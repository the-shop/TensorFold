"""A tiny deterministic Flash Next and the prompt-cache digests the numerics golden test pins (GPU).

Run as a script it prints the digests as JSON: the key tests run it on an edited copy of the package.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

TEXT = {
    "hidden_size": 512, "num_hidden_layers": 4, "vocab_size": 128, "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
    "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]},
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "output_gate_type": "sigmoid",
    "num_experts": 32, "num_experts_per_tok": 4, "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512, "hc_count": 4, "hc_lowrank": 64,
    "indexer_n_heads": 2, "indexer_head_dim": 128, "indexer_budget": 16, "indexer_compress_ratio": 4,
    "ple_layer_ids": [1], "ple_embed_dim": 512, "ple_conv_kernel_size": 4, "ngram_size": 3,
    "heads_per_ngram": 2, "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 8, "eos_token_id": 5,
}
ASSISTANT = (3, 4)                       # the tiny plan's reply marker: the canary cuts there
SPARSE_PROMPT = 300                      # past the patched dense limit: the selected-blocks attention runs


def tiny_model(bits: int, group: int, seed: int = 0) -> Any:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from tensorfold.families.qwen4_exp import model as q4
    from tensorfold.families.qwen4_exp.decode import FusedDecode

    mx.random.seed(seed)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": TEXT}))
    model.load_weights([(name, 0.1 * mx.random.normal(v.shape) if "norm" in name and name.endswith("weight") else v)
                        for name, v in tree_flatten(model.parameters())])
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=group, bits=bits,
                class_predicate=lambda path, m: hasattr(m, "to_quantized") and not path.endswith("mlp.gate"))
    mx.eval(model.parameters())
    model.__dict__["fused"] = FusedDecode(model)
    q4.select_by_kernels(model.layers)
    return model


def digests() -> dict[str, str]:
    """Each case's prompt-cache digest; every case runs twice and must repeat bit for bit."""

    import mlx.core as mx
    import numpy as np

    from tensorfold.engine import numerics
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.families.qwen4_exp import model as q4
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    mx.set_default_device(mx.gpu)
    plan = PrefillPlan(128, (), 64, ASSISTANT)
    grid = PrefillPlan(128)
    sparse = [int(t) for t in np.random.default_rng(5).integers(6, TEXT["vocab_size"], size=SPARSE_PROMPT)]
    out: dict[str, str] = {}

    def twice(name: str, run: Any) -> None:
        first, second = run(), run()
        if first != second:
            raise AssertionError(f"{name}: two runs of the same code differ ({first} vs {second})")
        out[name] = first

    for bits, group in ((8, 64), (4, 32)):
        runtime = FlashNext(tiny_model(bits, group), drafts=0)
        engine = LaneEngine(runtime, prefill_plan=plan)
        twice(f"q{bits}-canary", lambda: numerics.canary_digest(engine, plan, TEXT["vocab_size"]))
        dense_keys = q4.prefill.DENSE_KEYS
        q4.prefill.DENSE_KEYS = 64
        try:
            sparse_engine = LaneEngine(runtime, prefill_plan=grid)
            twice(f"q{bits}-sparse", lambda: numerics.cache_digest(sparse_engine.prefill_prefix(sparse))[:16])
        finally:
            q4.prefill.DENSE_KEYS = dense_keys
        previous = os.environ.get("TF_FLASH_PREFILL")
        os.environ["TF_FLASH_PREFILL"] = "0"
        try:
            twice(f"q{bits}-mlx-route", lambda: numerics.canary_digest(engine, plan, TEXT["vocab_size"]))
        finally:
            if previous is None:
                os.environ.pop("TF_FLASH_PREFILL", None)
            else:
                os.environ["TF_FLASH_PREFILL"] = previous
        del runtime, engine, sparse_engine
        mx.clear_cache()
    return out


def environment() -> str:
    """What the golden digests depend on besides the code: MLX and the GPU."""

    import mlx.core as mx

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return f"mlx={mx.__version__}|gpu={info.get('architecture', 'unknown')}"


def declared() -> str:
    from tensorfold.engine import numerics
    from tensorfold.families import families

    return f"{numerics.declared(families()['qwen4_exp'])}|e{numerics.ENGINE_NUMERICS}"


def kernels() -> str:
    """The family's key part for a model with one sparse-capable attention layer and a fixed prefill route."""

    from types import SimpleNamespace

    from tensorfold.families import families, kernel_version

    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(kernel_select=True))],
                            prefill_key="flash-prefill=fast;architecture=x;matmul=native")
    return kernel_version(families()["qwen4_exp"], model)


if __name__ == "__main__":
    from tensorfold.engine import numerics

    json.dump({"declared": declared(), "environment": environment(), "kernels": kernels(),
               "code": numerics.code_fingerprint(), "digests": digests()}, sys.stdout)
