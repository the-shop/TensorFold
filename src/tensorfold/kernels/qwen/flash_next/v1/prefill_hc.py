"""A prompt chunk's forward: the reference blocks with hyper-connections in the fused decode's arithmetic."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.kernels.qwen.flash_next.v1 import base
from tensorfold.kernels.qwen.flash_next.v1.hc import RINV, hc_norm

_HC_NORMED = r"""
  // Thread (e, r): element e of row r's streams: bf16((h * rinv(stream)) * scale), kernels.hc_project's normed input.
  const int e = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  const float rv = stream_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]);
"""

_HC_ACT = r"""
  // Thread (c, r): output c of the down + inject rows, / S, then SiLU (c < LOW) or the gate 2 sigmoid (hc_project's).
  const int c = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  const float v4 = float(bfloat(float(DN[size_t(r) * ND + c]) / float(S)));
  if (c < LOW) ACT[size_t(r) * LOW + c] = bfloat(bsilu(v4));
  else INJ[size_t(r) * S + (c - LOW)] = bfloat(2.0f * bsig(v4));
"""

_HC_MIX = r"""
  // Thread (d, r): dim d of row r's block input: mean over streams of bf16(sigmoid(up) * normed).
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat(total / float(S));
"""

_HEADER = base.QDOT_HEADER + RINV


def _qmm(x: mx.array, w: Any) -> mx.array:
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    return prefill_mm.matmul(x, w.weight, w.scales, w.biases, bits=w.bits, group=w.group_size)


def hyper_connection(hc: Any, h: mx.array, pending: tuple[mx.array, mx.array] | None, *, streams: int,
                     eps: mx.array) -> tuple[mx.array, mx.array, mx.array | None]:
    """h [R, S*D] and the pending (branch, inject gates) or None -> (h written back, block input, inject gates)."""

    if pending is None:
        h_new, ssp = hc_norm(h, streams=streams)
    else:
        h_new, ssp = hc_norm(h, streams=streams, write_back="plain", branch=(pending[0],), inject=pending[1])
    mixed, inj = project(h_new, ssp, hc.down, hc.up, hc.scale, hc.low, streams=streams, eps=eps, matmul=_qmm)
    return h_new, mixed, inj


def project(h_new: mx.array, ssp: mx.array, down: Any, up: Any, norm_scale: mx.array, low: int, *, streams: int,
            eps: mx.array, matmul: Any) -> tuple[mx.array, mx.array | None]:
    """hc_project through ``matmul(x, weights)`` for the down and up products: (block input, inject gates or None)."""

    rows, wide = h_new.shape
    dims = wide // streams
    normed_k = base.kernel("tf_hc_normed", _HC_NORMED, ["HN", "SSP", "NW", "eps"], ["NORMED"], header=_HEADER)
    normed = normed_k(inputs=[h_new, ssp, norm_scale, eps], template=[("S", streams), ("D", dims)],
                      grid=(wide, rows, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, wide)],
                      output_dtypes=[mx.bfloat16])[0]
    dn = matmul(normed, down)                                           # [R, LOW (+ S inject rows)]
    nd = int(dn.shape[1])
    act_k = base.kernel("tf_hc_act", _HC_ACT, ["DN"], ["ACT", "INJ"], header=_HEADER)
    act, inj = act_k(inputs=[dn], template=[("S", streams), ("LOW", low), ("ND", nd)],
                     grid=(nd, rows, 1), threadgroup=(min(nd, 256), 1, 1),
                     output_shapes=[(rows, low), (rows, streams)], output_dtypes=[mx.bfloat16, mx.bfloat16])
    up = matmul(act, up)                                                # [R, S*D]
    mix_k = base.kernel("tf_hc_mix", _HC_MIX, ["UP", "NORMED"], ["MIXED"], header=_HEADER)
    mixed = mix_k(inputs=[up, normed], template=[("S", streams), ("D", dims)],
                  grid=(dims, rows, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, dims)],
                  output_dtypes=[mx.bfloat16])[0]
    return mixed, (inj if nd > low else None)


QUEUE_LAYERS = 2                    # layers a slice; MLX holds a queued slice's buffers, so this bounds memory


def embed(model: Any, tokens: np.ndarray) -> mx.array:
    """A chunk's residual streams before layer 0: the token embeddings tiled over the streams, [L, S*D]."""

    h = model.model.embed_tokens(mx.array(tokens.astype(np.int32)))[0]
    return mx.tile(h, (1, model.args.hc_count))


def layer_step(model: Any, i: int, h: mx.array, pending: tuple[mx.array, mx.array] | None, tokens: np.ndarray,
               c: Any, after: mx.array | None = None) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    """Layer ``i`` on one chunk: (streams, pending (MoE branch, inject gates)) for the next layer. ``after``: an
    array the MoE's input waits for (the previous prompt call's MoE output: the expert window's calls in order)."""

    fused = model.__dict__["fused"]
    streams = model.args.hc_count
    eps = fused.eps
    layer, entry = model.layers[i], fused.layers[i]
    if "ple" in layer:
        if pending is not None:
            h = hc_norm(h, streams=streams, write_back="plain", branch=(pending[0],), inject=pending[1])[0]
            pending = None
        h = h + layer.ple(h[None], tokens, c)[0]
    h, mixed, inj = hyper_connection(entry["attn_hc"], h, pending, streams=streams, eps=eps)
    mixer = layer.linear_attn if layer.is_linear else layer.self_attn
    branch = mixer(mixed[None], c)[0]
    h, mixed, inj2 = hyper_connection(entry["mlp_hc"], h, (branch, inj), streams=streams, eps=eps)
    if after is not None:
        mixed = mx.depends([mixed], [after])[0]
    return h, (layer.mlp(mixed[None])[0], inj2)


def finish(model: Any, h: mx.array, pending: tuple[mx.array, mx.array] | None) -> tuple[mx.array, mx.array]:
    """After the last layer: (the streams before the final mixer [L, S*D], the mixed hidden state [1, L, D])."""

    fused = model.__dict__["fused"]
    h, mixed, _ = hyper_connection(fused.mixer, h, pending, streams=model.args.hc_count, eps=fused.eps)
    return h, mixed[None]


def hidden(model: Any, tokens: np.ndarray, cache: list[Any]) -> mx.array:
    """Qwen4Exp.hidden for a prompt chunk (batch 1); the same graph and bits however it is sliced."""

    h = embed(model, tokens)                                            # [L, S*D]
    pending = None
    queued = None
    depth = model.__dict__.get("prefill_queue", QUEUE_LAYERS)          # 1 where memory is tight
    states: list[mx.array] = []
    for i, c in enumerate(cache[:len(model.layers)]):
        h, pending = layer_step(model, i, h, pending, tokens, c)
        states += c.state                   # evaluated with its layer, so the chunk's inputs they read go with it
        if depth and (i + 1) % depth == 0:
            step = (h, *pending, *states)
            states = []
            mx.async_eval(*step)
            if queued is not None:
                mx.eval(*queued)
            queued = step
    h, mixed = finish(model, h, pending)
    model.__dict__["last_streams"] = h                                  # [L, S*D]: the streams before the mixer
    return mixed


def hidden_groups(model: Any, chunks: list[np.ndarray], cache: list[Any], *, before: Any = None,
                  after: Any = None) -> list[tuple[mx.array, mx.array]]:
    """Several consecutive prompt chunks layer-major: every chunk through layer i before layer i + 1, each
    (layer, chunk) call exactly the call ``hidden`` makes for that chunk (same rows, same cache state, same
    inputs), so every output and cache bit equals feeding the chunks one by one. Returns per chunk (streams before
    the final mixer, mixed hidden state [1, L, D]).

    The MoE input of each call waits for the previous call's MoE output (``mx.depends``), across chunks and
    layers, and every call is its own async_eval in program order: the expert window and the routing box are
    shared, so their host calls must run in the order they were queued. ``before(i, j)`` runs before a call is
    built and ``after(i, j)`` once it is queued (a layer's cache then holds chunk j's state, not yet evaluated)."""

    hs = [embed(model, t) for t in chunks]
    pends: list[Any] = [None] * len(chunks)
    last = None
    queued = None
    for i, c in enumerate(cache[:len(model.layers)]):
        for j, tokens in enumerate(chunks):
            if before is not None:
                before(i, j)
            hs[j], pends[j] = layer_step(model, i, hs[j], pends[j], tokens, c, after=last)
            last = pends[j][0]
            step = (hs[j], *pends[j], *c.state)
            mx.async_eval(*step)
            if queued is not None:
                mx.eval(*queued)
            queued = step
            if after is not None:
                after(i, j)
    if queued is not None:
        mx.eval(*queued)
    return [finish(model, h, pending) for h, pending in zip(hs, pends)]
