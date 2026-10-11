"""Layer-major prompt groups (TF_PREFILL_LAYER_MAJOR) against chunk-by-chunk prefill, bit for bit (GPU).

Every check compares the same engine and model with the groups off and on: the first token and the decoded tokens,
every working-cache field (KV buffers with their 256-row capacity, offsets, pooled index keys, DeltaNet conv and
state, the n-gram history and conv tail, the draft head's cache) and every prompt checkpoint, inner ones included.
Routed experts resident and streamed through the SSD pool (native server and Python serve loop); the pool's window
calls are also run in an adversarial evaluation order, where only the order chain keeps the bits."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.engine.prefill_plan import PrefillPlan  # noqa: E402
from tensorfold.families.qwen4_exp import model as q4  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

TEXT = {
    "hidden_size": 512, "num_hidden_layers": 8, "vocab_size": 128, "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
    "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]},
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "output_gate_type": "sigmoid",
    "num_experts": 32, "num_experts_per_tok": 4, "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640, "hc_count": 4, "hc_lowrank": 64,
    "indexer_n_heads": 2, "indexer_head_dim": 128, "indexer_budget": 16, "indexer_compress_ratio": 4,
    "ple_layer_ids": [2], "ple_embed_dim": 512, "ple_conv_kernel_size": 4, "ngram_size": 3,
    "heads_per_ngram": 2, "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 8, "eos_token_id": 5,
}
OPENER, ASSIST = 7, (8, 9)
PLAN = PrefillPlan(2048, openers=(OPENER,), min_chunk=256, assistant=ASSIST)


@pytest.fixture(autouse=True)
def _gpu(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    monkeypatch.setenv("TF_FLASH_PREFILL", "1")
    monkeypatch.setenv("TF_FLASH_DRAFT_VOCAB", "0")
    monkeypatch.delenv("TF_PREFILL_LAYER_MAJOR", raising=False)
    monkeypatch.delenv("TF_PREFILL_GROUP_GIB", raising=False)
    yield
    mx.set_default_device(previous)


def _q8(module) -> None:
    from mlx.utils import tree_flatten

    module.load_weights([(n, 0.1 * mx.random.normal(v.shape) if "norm" in n and n.endswith("weight") else v)
                         for n, v in tree_flatten(module.parameters())])
    module.set_dtype(mx.bfloat16)
    nn.quantize(module, group_size=64, bits=8,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and not p.endswith("mlp.gate"))
    mx.eval(module.parameters())


def tiny(seed: int = 3, **overrides) -> q4.Qwen4Exp:
    """Eight layers (6 DeltaNet, 2 sparse attention, n-grams on layer 1), 8-bit group 64 but the routers."""

    from tensorfold.families.qwen4_exp.decode import FusedDecode

    mx.random.seed(seed)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": {**TEXT, **overrides}}))
    _q8(model)
    model.__dict__["fused"] = FusedDecode(model)
    q4.select_by_kernels(model.layers)
    return model


def runtime(model, monkeypatch, drafts: bool = True):
    from tensorfold.families.qwen4_exp.mtp import FlashMTP
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    monkeypatch.setattr(FlashNext, "check_windows", lambda self: (4, {1: 1.0, 4: 2.0}))
    monkeypatch.setattr(FlashNext, "_time_mtp_step", lambda self: 1.0)
    head = None
    if drafts:
        mx.random.seed(11)
        head = FlashMTP(model.args)
        _q8(head)
    return FlashNext(model, head, drafts=1 if drafts else 0)


def prompt(n: int, seed: int = 5) -> list[int]:
    """Message starts (resume points) and assistant headers, so the plan cuts chunks of uneven rows."""

    rng = np.random.default_rng(seed)
    t = rng.integers(10, 97, size=n).astype(np.int64)
    t[0] = OPENER
    t[100] = 5
    for at in (3000, 7100):
        if at < n:
            t[at] = OPENER
    for at in (5000, 5100, 9000):
        if at + 1 < n:
            t[at], t[at + 1] = ASSIST
    return [int(x) for x in t]


def bits(a) -> np.ndarray:
    a = mx.contiguous(a)
    if a.dtype in (mx.bfloat16, mx.float16):
        return np.array(a.view(mx.uint16))
    if a.dtype == mx.float32:
        return np.array(a.view(mx.uint32))
    return np.array(a)


def same_value(a, b, where: str, bad: list[str]) -> None:
    if isinstance(a, mx.array) or isinstance(b, mx.array):
        if not (isinstance(a, mx.array) and isinstance(b, mx.array)):
            bad.append(f"{where}: one is None")
        elif a.shape != b.shape or a.dtype != b.dtype:
            bad.append(f"{where}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
        elif not np.array_equal(bits(a), bits(b)):
            bad.append(f"{where}: bits differ")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            bad.append(f"{where}: differ")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            bad.append(f"{where}: lengths {len(a)} vs {len(b)}")
        for q, (u, v) in enumerate(zip(a, b)):
            same_value(u, v, f"{where}[{q}]", bad)
    elif a != b:
        bad.append(f"{where}: {a!r} != {b!r}")


def same_cache(x: list, y: list, where: str, bad: list[str]) -> None:
    if len(x) != len(y):
        bad.append(f"{where}: {len(x)} vs {len(y)} entries")
    for i, (a, b) in enumerate(zip(x, y)):
        if type(a) is not type(b) or set(vars(a)) != set(vars(b)):
            bad.append(f"{where}[{i}]: fields {set(vars(a)) ^ set(vars(b))}")
            continue
        for key in sorted(vars(a)):
            same_value(getattr(a, key), getattr(b, key), f"{where}[{i}].{key}", bad)


def engine(rt, layer_major: bool, plan=PLAN) -> LaneEngine:
    e = LaneEngine(rt, prefill_plan=plan)
    e.layer_major = layer_major
    return e


def prefill(e: LaneEngine, ids: list[int], *, checkpoints=(), max_new: int = 4, cache=None, cached: int = 0):
    """Add a stream; returns (stream, a copy of the working cache as its prefill leaves it, before the first token,
    with the prompt's last hidden state appended). The tiny model's expert width is 640 (inside 512..1024, as the
    real one's): the fused decode is then deterministic, so decoded tokens are compared too."""

    stream = LaneStream("s", list(ids), max_new)
    taken: list = []
    first = e._family_first

    def keep(stream, work, hidden, *args):
        pool = e.model.model.__dict__.get("streamer")
        stream.prefill_bytes = None if pool is None else pool.bytes_read          # before any decode row reads
        taken.extend(e.copy_single_cache(work))
        taken.append(SimpleNamespace(hidden=mx.array(hidden)))
        return first(stream, work, hidden, *args)

    e._family_first = keep
    e.add_stream(stream, checkpoints_at=list(checkpoints), cache=cache, cached_tokens=cached)
    del e._family_first
    while e.active_count:
        e.step()
    return stream, taken


def compare(serial, grouped, label: str) -> list[str]:
    (s1, w1), (s2, w2) = serial, grouped
    bad: list[str] = []
    if s1.emitted != s2.emitted:
        bad.append(f"{label}: tokens {s1.emitted} vs {s2.emitted}")
    same_cache(w1, w2, f"{label} work", bad)
    got = [len(t) for t, _ in s2.history_checkpoints]
    want = [len(t) for t, _ in s1.history_checkpoints]
    if got != want:
        bad.append(f"{label}: checkpoints at {got}, chunk by chunk at {want}")
    for (t1, c1), (t2, c2) in zip(s1.history_checkpoints, s2.history_checkpoints):
        if t1 != t2:
            bad.append(f"{label}: checkpoint tokens differ at {len(t1)}")
        same_cache(c1, c2, f"{label} checkpoint@{len(t1)}", bad)
    return bad


def test_groups_give_chunk_by_chunk_bits_with_the_draft_head_and_inner_checkpoints(monkeypatch):
    """10,250 tokens in 6 uneven chunks crossing 4,096 and 8,192 keys; checkpoints at 3 inner cuts and a resume."""

    rt = runtime(tiny(), monkeypatch)
    ids = prompt(10250)
    chunks = PLAN.chunks(ids)
    cuts = [chunks.floor(b) for b in (3000, 7100, 9000)]
    assert len(chunks.between(0, len(ids))) == 6 and all(0 < c < len(ids) for c in cuts)
    serial = engine(rt, False)
    one = prefill(serial, ids, checkpoints=cuts)
    grouped = engine(rt, True)
    two = prefill(grouped, ids, checkpoints=cuts)
    assert grouped.prefill_steps == 6 * 8 and serial.prefill_steps == 0
    assert grouped.prefill_chunks == serial.prefill_chunks == 6
    bad = compare(one, two, "fresh")
    assert [len(t) for t, _ in one[0].history_checkpoints] == sorted(set(cuts))
    # a follow-up resumed from the checkpoint at the second cut: the group starts there
    tokens, stored = two[0].history_checkpoints[1]
    more = ids + prompt(700, seed=9)[1:]
    res1 = prefill(engine(rt, False), more, cache=serial.copy_single_cache(stored), cached=len(tokens))
    res2 = prefill(engine(rt, True), more, cache=serial.copy_single_cache(stored), cached=len(tokens))
    assert res2[0].cached_tokens == len(tokens)
    bad += compare(res1, res2, "resumed")
    assert not bad, bad[:12]


def test_a_tail_of_at_most_fused_rows_runs_alone_after_the_group(monkeypatch):
    """The last chunk (10 rows) takes the fused decode path chunk by chunk too: the cache it starts from, its own
    output, the cache it leaves and the decoded tokens are equal."""

    rt = runtime(tiny(), monkeypatch, drafts=False)
    ids = prompt(7058)
    starts = PLAN.chunks(ids).between(0, len(ids))
    assert starts[-1][1] - starts[-1][0] == 10
    cut = starts[-1][0]
    one = prefill(engine(rt, False), ids, checkpoints=[cut])
    grouped = engine(rt, True)
    two = prefill(grouped, ids, checkpoints=[cut])
    assert grouped.prefill_steps == (len(starts) - 1) * 8
    bad = compare(one, two, "tail")
    (t1, c1), = one[0].history_checkpoints
    (t2, c2), = two[0].history_checkpoints
    assert t1 == t2 and len(t1) == cut
    same_cache(c1, c2, "tail input", bad)
    assert not bad, bad[:12]


def test_groups_split_at_the_cap_and_where_memory_has_no_room(monkeypatch):
    from tensorfold.server.cancellation import Cancellation, PrefillGuard

    rt = runtime(tiny(), monkeypatch, drafts=False)
    ids = prompt(8200)
    rows = [b - a for a, b in PLAN.chunks(ids).between(0, len(ids))]
    per = rt.frontier_bytes_per_token
    assert per == 2 * (4 * 512 + 512 + 4)
    asked: list[int] = []

    class Memory:
        def before_chunk(self, cache, n):
            pass

        def after_chunk(self, cache, n):
            pass

        def allow_checkpoint(self, cache, current=None):
            return True

        def frontier_fits(self, cache, nbytes):
            asked.append(nbytes)
            return nbytes <= 3 * 2048 * per             # room for three full chunks' activations

    serial = prefill(engine(rt, False), ids)
    grouped = engine(rt, True)
    grouped.prefill_guard = PrefillGuard(Cancellation(), Memory())
    got = prefill(grouped, ids)
    assert not compare(serial, got, "split")
    assert asked[0] == sum(rows) * per and grouped.prefill_steps == len(rows) * 8
    monkeypatch.setenv("TF_PREFILL_GROUP_GIB", str(2 * 2048 * per / 2**30))      # at most two full chunks
    asked.clear()
    capped = engine(rt, True)
    capped.prefill_guard = PrefillGuard(Cancellation(), Memory())
    assert not compare(serial, prefill(capped, ids), "capped")
    assert max(asked) <= 2 * 2048 * per


def test_the_memory_guard_counts_a_groups_activations():
    from tensorfold.server.prompt_memory import PromptMemory

    memory = PromptMemory.__new__(PromptMemory)
    memory.frontier, memory.profile, memory.bootstrap = 0, None, 100
    memory.runtime = SimpleNamespace(get_active_memory=lambda: 1000, get_cache_memory=lambda: 0)
    memory.pending_bytes = lambda: 0
    base = memory.projected(10)
    memory.frontier = 5000
    assert memory.projected(10) == base + 5000


def test_a_prefill_cancelled_inside_a_group_keeps_no_half_swept_cache_and_a_retry_equals_fresh(monkeypatch):
    from tensorfold.server.cancellation import Cancellation, PrefillGuard, RequestCancelled

    rt = runtime(tiny(), monkeypatch)
    ids = prompt(7300)
    chunks = PLAN.chunks(ids)
    cuts = [chunks.floor(3000), chunks.floor(7100)]
    fresh = prefill(engine(rt, False), ids, checkpoints=cuts)

    class Memory:
        def before_chunk(self, cache, n):
            pass

        def after_chunk(self, cache, n):
            pass

        def allow_checkpoint(self, cache, current=None):
            return True

    class Late(Cancellation):
        calls = 0

        def check(self):
            Late.calls += 1
            if Late.calls == 6:                        # a few layers into the group
                raise RequestCancelled("request cancelled")

    e = engine(rt, True)
    e.prefill_guard = PrefillGuard(Late(), Memory())
    stopped = LaneStream("stopped", ids, 4)
    with pytest.raises(RequestCancelled):
        e.add_stream(stopped, checkpoints_at=cuts)
    assert 0 < e.prefill_steps < 8 * len(chunks.between(0, len(ids)))
    assert stopped.history_checkpoints == [] and e._prefill_at is None
    e.prefill_guard = None
    again = prefill(e, ids, checkpoints=cuts)
    assert not compare(fresh, again, "retry")


def test_off_by_default_and_never_without_the_fast_prefill(monkeypatch):
    rt = runtime(tiny(), monkeypatch, drafts=False)
    e = LaneEngine(rt, prefill_plan=PLAN)
    assert e._groups_chunks() is False
    monkeypatch.setenv("TF_PREFILL_LAYER_MAJOR", "1")
    assert e._groups_chunks() is True
    monkeypatch.setenv("TF_FLASH_PREFILL", "0")
    assert rt.groups_prompt_chunks is False and e._groups_chunks() is False


def test_the_n_gram_read_ahead_asks_for_every_chunk_once_in_order(monkeypatch):
    rt = runtime(tiny(), monkeypatch, drafts=False)
    emb = next(layer.ple.ple_embedding for layer in rt.model.layers if "ple" in layer)
    ids = prompt(8200)
    asked: dict[bool, list[bytes]] = {}
    for layer_major in (False, True):
        asked[layer_major] = []
        emb.host = SimpleNamespace(read_ahead=lambda x, out=asked[layer_major]: out.append(np.asarray(x).tobytes()))
        prefill(engine(rt, layer_major), ids)
    emb.host = None
    assert asked[True] == asked[False] and len(asked[True]) == len(PLAN.chunks(ids).between(0, len(ids)))


# -- streamed experts ---------------------------------------------------------------------------------------
def streamed(folder: Path, seed: int = 3, **overrides):
    """The tiny model with its routed experts in a safetensors file served by the SSD pool (1 GiB of slots)."""

    from tensorfold.families.qwen4_exp import stream

    model = tiny(seed, **overrides)
    path = folder / "model.safetensors"
    if not path.exists():
        tensors = {}
        for i, layer in enumerate(model.layers):
            for proj in ("gate", "up", "down"):
                linear = getattr(layer.mlp.switch_mlp, f"{proj}_proj")
                for part in ("weight", "scales", "biases"):
                    tensors[f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}_proj.{part}"] = getattr(linear, part)
        mx.save_safetensors(str(path), tensors)
    return model, stream.attach(model, folder, gib=1.0)


@pytest.fixture
def hostsync():
    from tensorfold.streaming import build

    try:
        return build.load()
    except RuntimeError as error:                                  # no cmake or nanobind here
        pytest.skip(str(error))


@pytest.mark.parametrize("native", ["1", "0"])
def test_streamed_experts_give_the_same_bits_with_one_window_fill_per_layer_a_group(tmp_path, monkeypatch, native,
                                                                                    hostsync):
    monkeypatch.setenv("TENSORFOLD_POOL_NATIVE", native)
    ids = prompt(6200)
    chunks = PLAN.chunks(ids).between(0, len(ids))
    out = {}
    for layer_major in (False, True):
        model, pool = streamed(tmp_path)
        assert pool.native == (native == "1")
        rt = runtime(model, monkeypatch, drafts=False)
        before = pool.bytes_read
        out[layer_major] = prefill(engine(rt, layer_major), ids, checkpoints=[chunks[2][0]])
        out[layer_major, "loads"] = (out[layer_major][0].prefill_bytes - before) // pool_expert_nbytes(pool)
        assert pool.error is None
        pool.close()
    assert not compare(out[False], out[True], "streamed")
    experts, layers = TEXT["num_experts"], TEXT["num_hidden_layers"]
    assert out[True, "loads"] <= layers * experts < out[False, "loads"]


def pool_expert_nbytes(pool) -> int:
    from tensorfold.streaming.pool import expert_nbytes

    return expert_nbytes(pool.found, 0)


SMALL = PrefillPlan(24)                 # 24-row chunks: each routes to a different few of 256 experts


@pytest.mark.parametrize("native", ["1", "0"])
def test_many_experts_and_small_chunks_keep_the_window_calls_in_order(tmp_path, monkeypatch, native, hostsync):
    monkeypatch.setenv("TENSORFOLD_POOL_NATIVE", native)
    ids = prompt(192)
    out = {}
    for layer_major in (False, True):
        model, pool = streamed(tmp_path, num_experts=256)
        rt = runtime(model, monkeypatch, drafts=False)
        before = pool.bytes_read
        out[layer_major] = prefill(engine(rt, layer_major, SMALL), ids)
        out[layer_major, "loads"] = (out[layer_major][0].prefill_bytes - before) // pool_expert_nbytes(pool)
        assert pool.error is None
        pool.close()
    assert not compare(out[False], out[True], "small chunks")
    assert out[True, "loads"] < out[False, "loads"]


class _Lazy:
    """mlx.core for prefill_hc with every eval deferred (``chain``: keep mx.depends, else drop it)."""

    def __init__(self, chain: bool) -> None:
        self.chain = chain

    def __getattr__(self, name):
        return getattr(mx, name)

    def async_eval(self, *arrays):
        pass

    def eval(self, *arrays):
        pass

    def depends(self, inputs, deps):
        return mx.depends(inputs, deps) if self.chain else inputs


@pytest.mark.parametrize("native", ["1", "0"])
def test_the_order_chain_keeps_the_bits_when_mlx_evaluates_out_of_order(tmp_path, monkeypatch, native, hostsync):
    """Every call of a group evaluated at once, outputs and caches last chunk first: MLX may then encode the window
    calls out of program order. The chain keeps them in order (equal bits); without it the routing box's tags catch
    a call that read another's routing, and the request fails instead of reading the wrong experts."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_hc

    monkeypatch.setenv("TENSORFOLD_POOL_NATIVE", native)
    ids = prompt(192)
    chunks = [np.array([ids[a:b]], dtype=np.int64) for a, b in SMALL.chunks(ids).between(0, len(ids))]

    def group(chain: bool):
        model, pool = streamed(tmp_path, num_experts=256)
        rt = runtime(model, monkeypatch, drafts=False)
        cache = rt.make_cache()
        if chain is None:
            outs = []
            for chunk in chunks:
                out = rt.prefill(chunk, cache)
                outs.append((rt.last_streams, out))
                mx.eval(*outs[-1], *[a for c in cache for a in c.state])
        else:
            monkeypatch.setattr(prefill_hc, "mx", _Lazy(chain))
            outs = rt.prefill_group(chunks, cache)
            monkeypatch.setattr(prefill_hc, "mx", mx)
            mx.eval(*[x for pair in reversed(outs) for x in pair], *[a for c in reversed(cache) for a in c.state])
        error = pool.error
        pool.close()
        return outs, cache, error

    ref, ref_cache, error = group(None)
    assert error is None
    got, cache, error = group(True)
    assert error is None
    bad: list[str] = []
    same_value(ref[-1][1], got[-1][1], "last chunk", bad)
    same_cache(ref_cache, cache, "cache", bad)
    assert not bad, bad[:8]
    _, cache, error = group(False)                       # the negative control: no chain
    wrong: list[str] = []
    same_cache(ref_cache, cache, "cache", wrong)
    assert error is not None, f"an unchained group went unnoticed ({len(wrong)} cache fields differ)"
    assert "another call's routing" in str(error)


def test_layer_major_prefill_copies_marks_of_the_same_shape(monkeypatch):
    """A checkpoint assembled inside a group holds each attention layer's KV buffer at the capacity chunk-by-chunk
    prefill leaves at that cut (the 256-row steps), not the end of the group's."""

    rt = runtime(tiny(), monkeypatch, drafts=False)
    ids = prompt(6200)
    cut = PLAN.chunks(ids).floor(3000)
    s1, _ = prefill(engine(rt, False), ids, checkpoints=[cut])
    s2, _ = prefill(engine(rt, True), ids, checkpoints=[cut])
    shapes = [[tuple(c.keys.shape) for c in cache if isinstance(c, q4.AttentionCache) and c.keys is not None]
              for _, cache in (s1.history_checkpoints[0], s2.history_checkpoints[0])]
    assert shapes[0] == shapes[1] and shapes[0]
