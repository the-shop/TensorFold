"""The decode profiler on a tiny Flash Next with streamed experts (GPU): profile off, Mode A (stamps only) and Mode B
(eval barriers) give the same logits, tokens and pool counters; Mode A's records cover every MoE call and split
each round so the report's buckets close on the round's wall time."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from test_qwen4_exp_family import TEXT, _flat  # noqa: E402

from tensorfold.families.qwen4_exp import model as q4  # noqa: E402
from tensorfold.families.qwen4_exp import stream  # noqa: E402
from tensorfold.streaming import profile  # noqa: E402

# the smallest shapes the fused decode kernels take: 512 wide, experts 640 wide (the down kernel), 128-dim DeltaNet
# heads, 64 experts, 8 n-gram shards
SHAPES = dict(linear_key_head_dim=128, linear_value_head_dim=128, hc_lowrank=32, hidden_size=512,
              moe_intermediate_size=640, shared_expert_intermediate_size=640, ple_embed_dim=256, heads_per_ngram=4,
              split_ngram_parts=8, num_experts=64, num_experts_per_tok=10, indexer_head_dim=64, head_dim=64,
              indexer_budget=64)     # dense attention throughout (the select kernel wants real-sized scores)
STEPS = 10


@pytest.fixture(scope="module")
def hostsync():
    from tensorfold.streaming import build

    try:
        module = build.load()
    except RuntimeError as error:                                  # no cmake or nanobind here
        pytest.skip(str(error))
    if not hasattr(module, "gpu_profile"):
        pytest.skip("this hostsync build has no profiler")
    return module


@pytest.fixture(autouse=True)
def _profile_off(monkeypatch):
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    # one-row rounds: the load-time multi-row check routes padded rows of this random model past the experts
    monkeypatch.setattr(FlashNext, "check_windows", lambda self: (1, {}))
    profile.configure(None)
    yield
    profile.configure(None)


def build(folder: Path):
    """The tiny model, 4-bit g32, its routed experts written to ``folder`` and streamed through a small pool."""

    mx.random.seed(3)
    text = dict(TEXT)
    text.update(SHAPES)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": text}))
    params = []
    for name, value in _flat(model.parameters()):
        if name.endswith("norm.weight") or "layernorm" in name or "hc_norm" in name or name.endswith("norm_key.weight"):
            value = 0.1 * mx.random.normal(value.shape)
        params.append((name, value))
    model.load_weights(params)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=32, bits=4, class_predicate=lambda path, m: not path.endswith("mlp.gate")
                and hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0)
    mx.eval(model.parameters())
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for i, layer in enumerate(model.layers):
        for proj in stream.PROJS:
            linear = getattr(layer.mlp.switch_mlp, f"{proj}_proj")
            for part in stream.PARTS:
                tensors[f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}_proj.{part}"] = getattr(linear, part)
    mx.save_safetensors(str(folder / "model-00001-of-00001.safetensors"), tensors)
    from tensorfold.streaming.pool import expert_nbytes, sources

    found, _ = sources(folder, stream.expert_names(len(model.layers)))
    gib = ((64 + 64) * expert_nbytes(found, 0) + 1) / 2**30      # the smallest pool: 64 decode slots for 40 routed
    streamer = stream.attach(model, folder, gib)
    from tensorfold.families.qwen4_exp.decode import FusedDecode

    model.__dict__["fused"] = FusedDecode(model)
    q4.select_by_kernels(model.layers)
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    return FlashNext(model, None, drafts=0), streamer


def greedy(flash, steps: int = STEPS):
    cache = flash.model.make_cache()
    token, logits_seen, tokens = 7, [], []
    for _ in range(steps):
        logits = flash.head(flash.hidden(np.array([[token]], dtype=np.int64), cache))
        mx.eval(logits)
        logits_seen.append(logits)
        token = int(mx.argmax(logits.reshape(-1)).item())
        tokens.append(token)
    return logits_seen, tokens


def run_mode(tmp_path: Path, mode: str):
    path = tmp_path / f"{mode}.jsonl"
    if mode != "off":
        profile.configure(path, sync=mode == "B")
    flash, streamer = build(tmp_path / f"ckpt-{mode}")
    logits, tokens = greedy(flash)
    mx.synchronize()
    time.sleep(0.05)                           # completion handlers run on Metal's thread just after the buffers end
    stats = streamer.stats()
    rec = profile.ACTIVE
    drained = rec.drain() if rec is not None else None
    streamer.close()
    return logits, tokens, stats, drained


def test_profile_modes_keep_logits_tokens_and_pool_state(tmp_path, hostsync):
    handlers = hostsync.gpu_handlers()
    off = run_mode(tmp_path, "off")
    assert hostsync.gpu_handlers() == handlers                     # off: no completion handler added
    a = run_mode(tmp_path, "A")
    b = run_mode(tmp_path, "B")
    for other in (a, b):
        assert other[1] == off[1]
        assert all(bool(mx.array_equal(x, y).item()) for x, y in zip(off[0], other[0]))
        stats = dict(other[2])
        assert {k: stats[k] for k in ("hits", "misses", "bytes_read")} == \
               {k: off[2][k] for k in ("hits", "misses", "bytes_read")}
    calls, gpu, lost = a[3]
    assert lost == {"calls_dropped": 0, "gpu_dropped": 0}
    decode = [c for c in calls if c["kind"] == 0]
    assert sum(c["misses"] for c in decode) == a[2]["misses"]       # every miss is in a record (R2's closure)
    assert sum(c["unique"] for c in decode) == a[2]["hits"] + a[2]["misses"]
    for c in calls:
        assert c["t_submit"] <= c["t_sig"] <= c["t_ready"]
        if c["misses"]:
            assert c["t_sig"] <= c["t_read0"] <= c["t_read1"] <= c["t_ready"]
            assert c["bytes"] > 0
    values = {g[0]: g for g in gpu}
    for c in decode:                                                 # both marks' buffers reported, after the host
        sig, wait = values[c["signal"]], values[c["signal"] + 1]
        assert sig[1] == 0 and wait[1] == 1
        assert sig[3] <= c["t_sig"] and wait[4] >= c["t_ready"]


def test_rounds_flush_and_the_report_closes(tmp_path, hostsync):
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream

    path = tmp_path / "rounds.jsonl"
    profile.configure(path)
    flash, streamer = build(tmp_path / "ckpt")
    profile.ACTIVE.flush(job="load")                                 # load-time checks: a request of their own
    engine = LaneEngine(flash)
    s = LaneStream(stream_id="x", prompt_ids=[7], max_new_tokens=8)
    engine.add_stream(s)
    while engine.active_count:
        engine.step()
    mx.synchronize()
    time.sleep(0.05)
    profile.ACTIVE.flush(job="x", tokens=len(s.emitted))
    streamer.close()
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    rounds = [l for l in lines if l["type"] == "round"]
    assert len(rounds) >= 6 and all(len([c for c in r["calls"] if c["kind"] == 0]) == 4 for r in rounds)
    spec = importlib.util.spec_from_file_location("report", Path(__file__).parents[1] / "tools" / "decode_profile_report.py")
    report = importlib.util.module_from_spec(spec)
    sys.modules["report"] = report
    spec.loader.exec_module(report)
    got = report.summarize(rounds, attention={3})
    assert all(r["f"]["ok"] for r in got["rows"]), [r["f"]["why"] for r in got["rows"]]
    for r in got["rows"]:
        total = sum(v for k, v in r["b"].items() if k != "gap")
        assert abs(total - r["f"]["wall"]) <= 1e-6 * r["f"]["wall"] + 1
        assert all(v >= 0 for v in r["b"].values()), r["b"]
    broken = dict(rounds[1], calls=rounds[1]["calls"][:-1])           # a lost record flags the round
    assert not report.round_buckets(broken, {3}, None, None)["f"]["ok"]
    assert report.main([str(path), "--attention", "3"]) == 0
