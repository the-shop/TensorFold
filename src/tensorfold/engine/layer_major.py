"""Layer-major prompt groups for Flash Next (qwen4_exp): several consecutive prompt chunks, every chunk through layer i
before layer i + 1, each (layer, chunk) call exactly the call the chunk-by-chunk prefill path makes.

This module sits outside the packages a prompt snapshot's kernel fingerprint hashes (``families/qwen4_exp`` and
``kernels/qwen/flash_next/v1``): the groups compute the same bits, so states saved by chunk-by-chunk prefill (and by
earlier releases with the same kernels) keep their keys. Its per-layer body mirrors
``kernels/qwen/flash_next/v1/prefill_hc.hidden``; the tests compare the two bit for bit."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np


def supports(model: Any) -> bool:
    """Whether ``model`` (the engine's family model) can run prompt groups: Flash Next on the fast prefill path."""

    inner = getattr(model, "model", None)
    if getattr(model, "fused", None) is None or inner is None or "fused" not in getattr(inner, "__dict__", {}):
        return False
    if not hasattr(inner, "layers") or not hasattr(getattr(inner, "args", None), "hc_count"):
        return False
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    return bool(prefill_mm.fast_prefill())


def frontier_bytes_per_token(model: Any) -> int:
    """What a group carries per prompt token between layers: residual streams [S*D], MoE branch [D] and inject
    gates [S], bf16 (25,608 bytes for the 8-bit Flash Next checkpoint: S 4, D 2560)."""

    args = model.model.args
    streams, dims = int(args.hc_count), int(args.hidden_size)
    return 2 * (streams * dims + dims + streams)


def embed(model: Any, tokens: np.ndarray) -> Any:
    h = model.model.embed_tokens(mx.array(tokens.astype(np.int32)))[0]
    return mx.tile(h, (1, model.args.hc_count))                          # [L, S*D]


def layer_step(model: Any, i: int, h: Any, pending: Any, tokens: np.ndarray, c: Any, after: Any = None) -> tuple:
    """prefill_hc.hidden's body for layer ``i`` on one chunk; ``after``: an array the MoE input waits for."""

    from tensorfold.kernels.qwen.flash_next.v1.hc import hc_norm
    from tensorfold.kernels.qwen.flash_next.v1.prefill_hc import hyper_connection

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


def finish(model: Any, h: Any, pending: Any) -> tuple:
    """After the last layer: (streams before the final mixer [L, S*D], mixed hidden state [1, L, D])."""

    from tensorfold.kernels.qwen.flash_next.v1.prefill_hc import hyper_connection

    fused = model.__dict__["fused"]
    h, mixed, _ = hyper_connection(fused.mixer, h, pending, streams=model.args.hc_count, eps=fused.eps)
    return h, mixed[None]


def hidden_groups(model: Any, chunks: list[np.ndarray], cache: list[Any], *, before: Any = None,
                  after: Any = None) -> list[tuple[Any, Any]]:
    """The chunks layer-major on the Qwen4Exp ``model``: per chunk (streams before the final mixer, mixed hidden).

    The MoE input of each call waits for the previous call's MoE output (``mx.depends``), across chunks and layers,
    and every call is its own async_eval in program order: the SSD pool's routing box and expert window are shared,
    so their host calls must run in the order they were queued. ``before(i, j)`` runs before a call is built and
    ``after(i, j)`` once it is queued (layer i's cache then holds chunk j's state, perhaps not yet evaluated)."""

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


def prefill_group(model: Any, inputs: list[Any], cache: list[Any], *, before: Any = None,
                  after: Any = None) -> list[tuple[Any, Any]]:
    """Consecutive prompt chunks (each more than ``fused_rows`` rows) of the engine's Flash Next ``model``,
    layer-major; per chunk (its streams before the final mixer, its mixed hidden state [1, R, D])."""

    chunks = []
    for one in inputs:
        tokens = np.asarray(one, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        if tokens.shape[0] != 1 or tokens.shape[1] <= model.fused_rows:
            raise ValueError("a layer-major group takes batch-1 chunks of more than fused_rows rows")
        chunks.append(tokens)
    if not supports(model):
        raise RuntimeError("layer-major prompt groups need Flash Next's fast prefill path")
    if "_resolved_prefill_identity" in model.__dict__:
        _ = model.prefill_key           # refuse a changed prefill mode before reading or updating a keyed cache
    return hidden_groups(model.model, chunks, cache[: model.layer_count], before=before, after=after)


def adopt_streams(model: Any, streams: Any) -> None:
    """Make one grouped chunk's residual streams [R, S*D] the ones the draft head's absorb and speculate read
    (``FlashNext.hidden`` sets the same attribute for a chunk it runs)."""

    model._streams = streams


__all__ = ["adopt_streams", "frontier_bytes_per_token", "hidden_groups", "prefill_group", "supports"]
