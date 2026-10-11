"""Flash Next with its routed experts streamed from the checkpoint into a GPU slot pool; the rest stays resident."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.streaming.pool import PARTS, Streamer, expert_nbytes, sources

PROJS = ("gate", "up", "down")
PREFILL_ROWS = 4096          # the largest prompt chunk a window call takes

# one threadgroup: the box's first E words become 1 for each expert some routed row picked, else 0
_PRESENT = r"""
  const uint t = thread_position_in_threadgroup.x;
  device uint32_t* box = (device uint32_t*)BOX;
  for (uint e = t; e < uint(NE); e += 1024) box[e] = 0u;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = t; i < uint(N); i += 1024) box[IDS[i]] = 1u;
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) OUT[0] = 1u;
"""

# _PRESENT with counts: the box's first E words become how many routed rows picked each expert (prompt seeding)
_COUNT = r"""
  const uint t = thread_position_in_threadgroup.x;
  device uint32_t* box = (device uint32_t*)BOX;
  for (uint e = t; e < uint(NE); e += 1024) box[e] = 0u;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = t; i < uint(N); i += 1024)
    atomic_fetch_add_explicit((device atomic_uint*)&box[IDS[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) OUT[0] = 1u;
"""


# a window call's tag first (box word E, the host checks it before and after reading the flags), then the flags
_TAG = r"""
  if (thread_position_in_threadgroup.x == 0) ((device uint32_t*)BOX)[NE] = TAG[0];
  threadgroup_barrier(mem_flags::mem_device);
"""


def pool_flags(env: dict | None = None) -> tuple[int, int]:
    """(decode prefetch k, prompt seed experts a layer) from TENSORFOLD_POOL_DECODE_PREFETCH (0..16) and
    TENSORFOLD_POOL_PROMPT_SEED=1 (TENSORFOLD_POOL_PROMPT_SEED_PER_LAYER, default 32); both off by default."""

    env = os.environ if env is None else env
    prefetch = min(16, max(0, int(env.get("TENSORFOLD_POOL_DECODE_PREFETCH", "0") or 0)))
    seed = 0
    if int(env.get("TENSORFOLD_POOL_PROMPT_SEED", "0") or 0) > 0:
        seed = max(1, int(env.get("TENSORFOLD_POOL_PROMPT_SEED_PER_LAYER", "32") or 32))
    return prefetch, seed


def expert_names(layers: int) -> dict[tuple[int, str, str], str]:
    return {(i, proj, part): f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}_proj.{part}"
            for i in range(layers) for proj in PROJS for part in PARTS}


def switch_keys(key: str) -> bool:
    """Weights the pool serves instead of the model: the decoder layers' routed stacks (not the MTP head's)."""

    return key.startswith("model.layers.") and ".mlp.switch_mlp." in key


def attach(model: Any, model_dir: Path, gib: float) -> Streamer:
    """Build the pool for ``gib`` GiB of experts and route every decoder MoE through it."""

    from tensorfold.streaming.build import load as hostsync

    cfg = model.args
    found, shapes = sources(model_dir, expert_names(cfg.num_hidden_layers))
    slots = int(gib * 2**30) // expert_nbytes(found, 0)
    prefetch, seed = pool_flags()
    streamer = Streamer(found, shapes, layers=cfg.num_hidden_layers, experts=cfg.num_experts,
                        top_k=cfg.num_experts_per_tok, slots=slots, box_rows=PREFILL_ROWS, hostsync=hostsync(),
                        decode_prefetch=prefetch, prompt_seed=seed)
    for index, layer in enumerate(model.layers):
        moe = layer.mlp
        moe.__dict__["streamer"] = streamer
        moe.__dict__["stream_layer"] = index
        for proj in PROJS:                       # the stacks stay on disk: placeholders keep the modules' names
            linear = getattr(moe.switch_mlp, f"{proj}_proj")
            for part in PARTS:
                setattr(linear, part, mx.zeros((1,), dtype=getattr(linear, part).dtype))
    model.__dict__["streamer"] = streamer
    return streamer


def moe_rows(moe: Any, x: mx.array, logits: mx.array, shared: tuple[Any, Any, Any],
             guess_logits: mx.array | None = None) -> tuple[str, tuple[mx.array, ...]]:
    """FusedDecode's MoE for streamed experts: the resident kernels' arithmetic, weights read through slots.

    ``guess_logits``: the next layer's router on this layer's input; its top decode_prefetch ids go to the host
    with this call's routing, for reads behind it (they change what is resident, never what is computed)."""

    from tensorfold.kernels.qwen.flash_next.v1 import stream_experts

    streamer, layer = moe.__dict__["streamer"], moe.__dict__["stream_layer"]
    cfg_top = moe.top_k
    token = stream_experts.route(logits, cfg_top, streamer.box)
    guess = 0
    if guess_logits is not None and streamer.decode_prefetch:
        guess = streamer.decode_prefetch
        token = mx.depends([token], [stream_experts.route(guess_logits, guess, streamer.guess_box)])[0]
    token = streamer.hold(token, "rows", layer, int(x.shape[0]), guess)
    slot_of, layer_id = mx.depends([streamer.slot_of, streamer.layer_ids[layer]], [token])
    pool = _Pool(streamer.pool)
    gate, up, down = shared
    act, picks, weights = stream_experts.gateup(x, logits, cfg_top, streamer.experts, pool, slot_of, layer_id,
                                                (gate, up))
    return "grouped", (stream_experts.down(act, picks, streamer.experts, pool, slot_of, layer_id, down), weights,
                       logits)


def moe_chunk(moe: Any, x: mx.array) -> mx.array:
    """SparseMoE on a prompt chunk: its routing, then the layer's picked experts loaded into the window slots."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm
    from tensorfold.kernels.qwen.flash_next.v1.base import kernel

    streamer, layer = moe.__dict__["streamer"], moe.__dict__["stream_layer"]
    experts, weights = moe.route(x)
    ids = experts.reshape(-1).astype(mx.uint32)
    present = (kernel("q4_stream_count_tagged", _TAG + _COUNT, ["IDS", "BOX", "TAG"], ["OUT"]) if streamer.prompt_seed
               else kernel("q4_stream_present_tagged", _TAG + _PRESENT, ["IDS", "BOX", "TAG"], ["OUT"]))
    tag = streamer.next_signal()
    token = present(inputs=[ids, streamer.box, mx.array([tag & 0xFFFFFFFF], dtype=mx.uint32)],
                    template=[("NE", streamer.experts), ("N", int(ids.size))],
                    grid=(1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]
    token = streamer.hold(token, "window_tagged", layer, int(x.shape[1]), expect=tag)
    views = streamer.window_views()
    switch = copy.copy(moe.switch_mlp)
    for proj in PROJS:
        linear = copy.copy(getattr(moe.switch_mlp, f"{proj}_proj"))
        for part, array in zip(PARTS, mx.depends([getattr(views[proj], p) for p in PARTS], [token])):
            setattr(linear, part, array)
        setattr(switch, f"{proj}_proj", linear)
    if prefill_mm.active(int(x.shape[0]) * int(x.shape[1])) and x.shape[0] == 1 and x.dtype == mx.bfloat16:
        return prefill_mm.moe(moe, x, route=(experts, weights), switch=switch)
    routed = (switch(x, experts) * weights[..., None]).sum(axis=-2)
    return routed + moe.shared_expert(x) * mx.sigmoid(moe.shared_expert_gate(x))


class _Pool:
    """The pool's projections as the kernels' gate/up/down arguments."""

    def __init__(self, pool: dict) -> None:
        self.gate, self.up, self.down = pool["gate"], pool["up"], pool["down"]


__all__ = ["attach", "expert_names", "moe_chunk", "moe_rows", "pool_flags", "switch_keys"]
