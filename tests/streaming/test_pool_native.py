"""The extension's pool server against the Python serve loop, and the contiguous expert pack, on real hostsync."""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.streaming import pack as packs
from tensorfold.streaming import pool

LAYERS, EXPERTS, TOPK, BOX_ROWS = 3, 8, 2, 4
# odd part sizes, unlike one another and far from the 16 KiB record alignment
SPECS = {"weight": ([5, 12], "U32", 4), "scales": ([5, 3], "BF16", 2), "biases": ([5, 3], "F16", 2)}
PROJS = ("gate", "up", "down")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("TENSORFOLD_EXPERT_PACK", "TENSORFOLD_POOL_NATIVE", "TENSORFOLD_POOL_THREADS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def hs():
    from tensorfold.streaming import build

    try:
        module = build.load()
    except RuntimeError as error:                                  # no cmake or nanobind here
        pytest.skip(str(error))
    if not hasattr(module, "PoolServer"):
        pytest.skip("this hostsync build has no PoolServer")
    return module


def write_checkpoint(folder: Path, per_expert: bool = False, seed: int = 0) -> dict:
    """Two shards of random bytes, qwen4_exp names; returns (layer, expert) -> {(proj, part): bytes}."""

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth: dict = {}
    shards = [({}, bytearray()), ({}, bytearray())]
    index = {}
    for layer in range(LAYERS):
        for proj in PROJS:
            for part, (shape, dtype, size) in SPECS.items():
                one = int(np.prod(shape)) * size
                blobs = [rng.integers(0, 256, one, dtype=np.uint8).tobytes() for _ in range(EXPERTS)]
                for e, blob in enumerate(blobs):
                    truth.setdefault((layer, e), {})[(proj, part)] = blob
                shard = (layer + PROJS.index(proj)) % 2
                tensors, data = shards[shard]
                if per_expert:
                    for e in (5, 0, 7, 2, 1, 6, 3, 4):
                        name = f"language_model.model.layers.{layer}.mlp.experts.{e}.{proj}_proj.{part}"
                        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + one]}
                        data += blobs[e]
                        index[name] = shard
                    continue
                name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{part}"
                tensors[name] = {"dtype": dtype, "shape": [EXPERTS, *shape],
                                 "data_offsets": [len(data), len(data) + EXPERTS * one]}
                data += b"".join(blobs)
                index[name] = shard
    for i, (tensors, data) in enumerate(shards):
        header = json.dumps(tensors).encode()
        (folder / f"model-0000{i + 1}-of-00002.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
    text = {"num_hidden_layers": LAYERS, "num_experts": EXPERTS, "num_experts_per_tok": TOPK}
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text}))
    return truth


def names(per_expert: bool) -> dict:
    out = {}
    for layer in range(LAYERS):
        for proj in PROJS:
            for part in SPECS:
                if per_expert:
                    for e in range(EXPERTS):
                        out[(layer, proj, part, e)] = f"language_model.model.layers.{layer}.mlp.experts.{e}.{proj}_proj.{part}"
                else:
                    out[(layer, proj, part)] = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{part}"
    return out


def make(hs, folder, slots, native, pack=False, per_expert=False, workers=4):
    found, shapes = pool.sources(folder, names(per_expert))
    s = pool.Streamer(found, shapes, layers=LAYERS, experts=EXPERTS, top_k=TOPK, slots=slots, box_rows=BOX_ROWS,
                      hostsync=hs, workers=workers, native=native, pack=pack)
    assert s.native == native
    return s


def call(hs, s, kind, layer, ids, ok=True):
    """One MoE call as the GPU makes it: ids (or presence flags) in the box, the signal, then the host's release."""

    if kind == "window":
        words = [0] * EXPERTS
        for e in ids:
            words[e] = 1
    else:
        words = list(ids)
    hs.write_into(s.box, 0, struct.pack(f"<{len(words)}I", *words))
    signal, ready = s.marks(kind, layer, len(ids) // TOPK if kind == "rows" else 1)
    s.channel.signal(signal)
    assert s.channel.wait(ready, 10_000)
    assert (s.error is None) == ok


def snapshot(hs, s) -> dict:
    out = {f"{proj}.{part}": hs.peek(getattr(s.pool[proj], part), 0, getattr(s.pool[proj], part).nbytes)
           for proj in s.pool for part in pool.PARTS}
    out["slot_of"] = hs.peek(s.slot_of, 0, s.slot_of.nbytes)
    out["lru"] = s.lru_items()
    out["free"] = s.free_slots()
    out["window"] = s.window_layers()
    out["counters"] = (s.hits, s.misses, s.bytes_read)
    return out


def slot_part(hs, s, proj, part, slot) -> bytes:
    shape, _, size = SPECS[part]
    n = int(np.prod(shape)) * size
    return hs.peek(getattr(s.pool[proj], part), slot * n, n)


def check_resident(hs, s, truth):
    """Every LRU entry's slot and every window slot hold exactly their expert's checkpoint bytes."""

    table = np.frombuffer(hs.peek(s.slot_of, 0, s.slot_of.nbytes), dtype=np.int32).reshape(LAYERS, EXPERTS)
    for (layer, e), slot in s.lru_items():
        assert table[layer, e] == slot
        for (proj, part), blob in truth[(layer, e)].items():
            assert slot_part(hs, s, proj, part, slot) == blob, (layer, e, proj, part)
    for e, layer in enumerate(s.window_layers()):
        if layer >= 0:
            for (proj, part), blob in truth[(layer, e)].items():
                assert slot_part(hs, s, proj, part, e) == blob


SEQUENCE = [("rows", 0, [1, 2, 3, 4]), ("window", 1, [0, 5]), ("rows", 1, [5, 6, 7, 0]), ("rows", 2, [1, 1, 3, 3]),
            ("rows", 0, [2, 7, 4, 6]), ("rows", 2, [0, 4, 5, 6]), ("window", 0, [2]), ("rows", 1, [1, 2, 3, 4]),
            ("window", 0, [2, 3, 7]), ("window", 2, list(range(EXPERTS))), ("rows", 0, [0, 0, 0, 0])]


def random_sequence(seed: int, n: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        layer = int(rng.integers(0, LAYERS))
        if rng.random() < 0.2:
            out.append(("window", layer, sorted(set(rng.integers(0, EXPERTS, int(rng.integers(1, EXPERTS + 1))).tolist()))))
        else:
            rows = int(rng.integers(1, BOX_ROWS + 1))
            out.append(("rows", layer, rng.integers(0, EXPERTS, rows * TOPK).tolist()))
    return out


# -- native == Python ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("per_expert", [False, True])
@pytest.mark.parametrize("slots,sequence", [(2 * EXPERTS + 3, SEQUENCE), (2 * EXPERTS, random_sequence(1, 150)),
                                            (2 * EXPERTS + 5, random_sequence(2, 150))])
def test_native_serves_exactly_as_python(hs, tmp_path, per_expert, slots, sequence):
    truth = write_checkpoint(tmp_path, per_expert, seed=3)
    py = make(hs, tmp_path, slots, native=False, per_expert=per_expert)
    nat = make(hs, tmp_path, slots, native=True, per_expert=per_expert)
    for kind, layer, ids in sequence:
        call(hs, py, kind, layer, ids)
        call(hs, nat, kind, layer, ids)
    a, b = snapshot(hs, py), snapshot(hs, nat)
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == b[key], key
    check_resident(hs, nat, truth)
    assert nat.stats()["native"] == 1 and nat.stats()["packed"] == 0
    assert {k: nat.stats()[k] for k in ("hits", "misses", "bytes_read")} == py.stats()
    found, _ = pool.sources(tmp_path, names(per_expert))
    assert nat.stats()["preads"] * pool.expert_nbytes(found, 0) == 9 * nat.bytes_read     # 9 parts an expert
    py.close()
    nat.close()


def test_preads_are_nine_an_expert_unpacked_and_one_packed(hs, tmp_path):
    write_checkpoint(tmp_path)
    found, _ = pool.sources(tmp_path, names(False))
    packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    plain = make(hs, tmp_path, 2 * EXPERTS + 2, native=True, pack=False)
    packed = make(hs, tmp_path, 2 * EXPERTS + 2, native=True, pack=True)
    for s in (plain, packed):
        call(hs, s, "rows", 0, [1, 2, 3, 3])
        call(hs, s, "window", 1, [4, 5])
    assert plain.stats()["preads"] == 5 * 9 and packed.stats()["preads"] == 5
    assert plain.stats()["bytes_read"] == packed.stats()["bytes_read"] == 5 * pool.expert_nbytes(found, 0)
    plain.close()
    packed.close()


def test_threads_come_from_the_environment(hs, tmp_path, monkeypatch):
    write_checkpoint(tmp_path)
    monkeypatch.setenv("TENSORFOLD_POOL_THREADS", "3")
    found, shapes = pool.sources(tmp_path, names(False))
    s = pool.Streamer(found, shapes, layers=LAYERS, experts=EXPERTS, top_k=TOPK, slots=2 * EXPERTS, box_rows=4,
                      hostsync=hs, pack=False)
    assert s.native and s.stats()["threads"] == 3
    s.close()
    monkeypatch.setenv("TENSORFOLD_POOL_NATIVE", "0")
    s = pool.Streamer(found, shapes, layers=LAYERS, experts=EXPERTS, top_k=TOPK, slots=2 * EXPERTS, box_rows=4,
                      hostsync=hs)
    assert not s.native
    s.close()


def test_prefetch_or_seed_keeps_the_python_loop(hs, tmp_path):
    write_checkpoint(tmp_path)
    found, shapes = pool.sources(tmp_path, names(False))
    for flags in ({"decode_prefetch": 2}, {"prompt_seed": 4}):
        s = pool.Streamer(found, shapes, layers=LAYERS, experts=EXPERTS, top_k=TOPK, slots=3 * EXPERTS, box_rows=4,
                          hostsync=hs, native=True, **flags)
        assert not s.native
        s.close()


# -- errors ----------------------------------------------------------------------------------------------------

def test_a_routed_id_outside_the_model_stops_both_alike(hs, tmp_path):
    write_checkpoint(tmp_path)
    runs = [make(hs, tmp_path, 2 * EXPERTS + 3, native=n) for n in (False, True)]
    for s in runs:
        call(hs, s, "rows", 0, [1, 2, 3, 4])
        call(hs, s, "rows", 1, [2, 99, 2, 5], ok=False)       # the GPU still got its release
        assert isinstance(s.error, ValueError) and "99" in str(s.error)
        with pytest.raises(RuntimeError, match="stopped"):
            s.marks("rows", 0, 1)
    a, b = (snapshot(hs, s) for s in runs)
    for key in a:
        assert a[key] == b[key], key
    for s in runs:
        s.close()


def test_a_pool_without_room_for_a_call_raises_memory_error_in_both(hs, tmp_path):
    write_checkpoint(tmp_path)
    py = make(hs, tmp_path, 2 * EXPERTS, native=False)
    py.free = py.free[-2:]                                      # two decode slots
    nat = make(hs, tmp_path, 2 * EXPERTS, native=True)
    nat._server.close()
    nat._server = nat._native(EXPERTS + 2, 2, False)            # the server over a pool with two decode slots
    for s in (py, nat):
        call(hs, s, "rows", 0, [1, 2, 3, 3], ok=False)
        assert isinstance(s.error, MemoryError)
        assert (s.hits, s.misses) == (0, 2)
    assert py.lru_items() == nat.lru_items() and py.free_slots() == nat.free_slots() == []
    py.close()
    nat.close()


def test_a_failed_read_is_raised_on_the_next_call(hs, tmp_path):
    write_checkpoint(tmp_path)
    runs = [make(hs, tmp_path, 2 * EXPERTS + 2, native=n) for n in (False, True)]
    for shard in tmp_path.glob("model-*.safetensors"):          # every expert's bytes gone from under the pool
        with open(shard, "r+b") as f:
            f.truncate(64)
    for s in runs:
        call(hs, s, "rows", 2, [6, 7, 7, 6], ok=False)
        assert isinstance(s.error, RuntimeError) and "short read" in str(s.error)
        with pytest.raises(RuntimeError, match="stopped"):
            s.marks("window", 0, 1)
        s.close()


def test_no_signal_from_the_gpu_times_out_and_still_releases(hs, tmp_path, monkeypatch):
    write_checkpoint(tmp_path)
    monkeypatch.setattr(pool, "TIMEOUT_MS", 300)
    for native in (False, True):
        s = make(hs, tmp_path, 2 * EXPERTS, native=native)
        _, ready = s.marks("rows", 0, 1)
        assert s.channel.wait(ready, 10_000)
        assert isinstance(s.error, TimeoutError)
        s.close()


@pytest.mark.parametrize("layer", [LAYERS, -1, 1000])
@pytest.mark.parametrize("kind", ["rows", "window"])
def test_a_layer_outside_the_model_changes_nothing_in_both(hs, tmp_path, layer, kind):
    write_checkpoint(tmp_path)
    runs = [make(hs, tmp_path, 2 * EXPERTS + 2, native=n) for n in (False, True)]
    for s in runs:
        call(hs, s, "rows", 0, [1, 2, 3, 4])
        call(hs, s, kind, layer, [1, 6], ok=False)
        assert isinstance(s.error, ValueError) and "layer" in str(s.error)
    a, b = (snapshot(hs, s) for s in runs)
    for key in a:
        assert a[key] == b[key], key
    assert a["counters"][:2] == (0, 4) and len(a["lru"]) == 4
    for s in runs:
        s.close()


def test_the_server_refuses_arrays_of_the_wrong_kind(hs, tmp_path):
    write_checkpoint(tmp_path)
    s = make(hs, tmp_path, 2 * EXPERTS, native=True)
    order = [(proj, part) for proj in s.pool for part in pool.PARTS]
    dsts = [getattr(s.pool[proj], part) for proj, part in order]
    sizes = [d.nbytes // d.shape[0] for d in dsts]
    table = (np.zeros(LAYERS * EXPERTS * len(order), dtype=np.int32).tobytes(),
             np.zeros(LAYERS * EXPERTS * len(order), dtype=np.int64).tobytes())
    args = {"layers": LAYERS, "experts": EXPERTS, "top_k": TOPK, "slots": 2 * EXPERTS, "workers": 1, "timeout_ms": 100}
    bad_box = mx.zeros(s.box.shape, dtype=mx.int32)
    bad_table = mx.zeros(s.slot_of.shape, dtype=mx.uint32)
    mx.eval(bad_box, bad_table)
    for box, slot_of, part_sizes in ((bad_box, s.slot_of, sizes), (s.box, bad_table, sizes),
                                     (s.box, s.slot_of, [n + 2 for n in sizes])):
        with pytest.raises(ValueError):
            hs.PoolServer(s.channel, box, slot_of, dsts, part_sizes, *table, **args)
    s.close()


# -- close -----------------------------------------------------------------------------------------------------

def test_close_releases_queued_calls_the_gpu_never_signalled(hs, tmp_path):
    write_checkpoint(tmp_path)
    s = make(hs, tmp_path, 2 * EXPERTS + 2, native=True)
    readies = [s.marks("rows", 0, 1)[1] for _ in range(20)]             # a graph built and never evaluated
    start = time.monotonic()
    s.close()
    assert time.monotonic() - start < 3
    assert s.channel.value() >= readies[-1] and s.misses == 0
    assert isinstance(s.error, RuntimeError) and "closed before the GPU reached layer 0" in str(s.error)
    assert "closed before" in s.stats()["error"]                         # a forward on those slots must fail


def test_close_still_loads_a_call_whose_signal_arrived(hs, tmp_path):
    truth = write_checkpoint(tmp_path, seed=8)
    s = make(hs, tmp_path, 2 * EXPERTS + 2, native=True)
    hs.write_into(s.box, 0, struct.pack("<4I", 2, 5, 5, 2))
    signal, _ = s.marks("rows", 1, 2)
    _, last = s.marks("rows", 2, 1)                                      # queued behind it, never signalled
    s.channel.signal(signal)
    s.close()
    assert s.channel.value() >= last
    assert (s.hits, s.misses) == (0, 2)
    check_resident(hs, s, truth)
    assert "layer 2" in str(s.error)                                     # only the unsignalled call failed


def test_a_second_close_waits_for_the_first(hs, tmp_path):
    write_checkpoint(tmp_path)
    for _ in range(5):
        s = make(hs, tmp_path, 2 * EXPERTS + 2, native=True)
        _, ready = s.marks("rows", 0, 1)                                 # keeps the first close busy ~1 s
        done: list[float] = []

        def close(server=s._server, done=done):
            server.close()
            done.append(time.monotonic())

        closers = [threading.Thread(target=close) for _ in range(2)]
        for c in closers:
            c.start()
        for c in closers:
            c.join(10)
            assert not c.is_alive()
        assert len(done) == 2 and s.channel.value() >= ready
        s.close()                                                        # the fds after both: nothing left reading
        assert not s.fds


def test_concurrent_streamer_closes_close_each_fd_once(hs, tmp_path):
    write_checkpoint(tmp_path)
    for native in (True, False):
        s = make(hs, tmp_path, 2 * EXPERTS, native=native)
        s.marks("rows", 0, 1)                                            # never signalled: the first close waits
        errors: list[str] = []

        def close(s=s, errors=errors):
            try:
                s.close()
            except OSError as error:
                errors.append(repr(error))

        closers = [threading.Thread(target=close) for _ in range(3)]
        for c in closers:
            c.start()
        for c in closers:
            c.join(10)
            assert not c.is_alive()
        assert not errors and not s.fds
        s.close()


@pytest.mark.parametrize("native", [False, True])
def test_stats_show_a_failed_pool(hs, tmp_path, native):
    write_checkpoint(tmp_path)
    s = make(hs, tmp_path, 2 * EXPERTS, native=native)
    call(hs, s, "rows", 0, [1, 2, 3, 4])
    assert "error" not in s.stats()
    call(hs, s, "rows", 0, [1, 99, 3, 4], ok=False)
    assert s.stats()["error"] == "ValueError: routed expert 99 is outside the model"
    s.close()


# -- the pack --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("per_expert", [False, True])
def test_the_pack_holds_each_experts_parts_in_one_record(tmp_path, per_expert):
    truth = write_checkpoint(tmp_path, per_expert, seed=4)
    found, _ = pool.sources(tmp_path, names(per_expert))
    out = packs.build(found, EXPERTS, tmp_path / "experts.pack", buffer_bytes=3 * packs.ALIGNMENT)   # 3-expert groups
    lay = packs.validate(out, found, EXPERTS)
    data = out.read_bytes()
    assert len(data) == LAYERS * EXPERTS * packs.ALIGNMENT and lay.stride == packs.ALIGNMENT
    assert lay.order == [(proj, part) for proj in sorted(PROJS) for part in pool.PARTS]
    for (layer, e), parts in truth.items():
        at = lay.record(layer, e)
        assert data[at:at + lay.payload] == b"".join(parts[o] for o in lay.order)
        assert data[at + lay.payload:at + lay.stride] == bytes(lay.stride - lay.payload)


def test_the_tool_builds_and_checks_a_pack(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    try:
        import pack_experts
    finally:
        sys.path.pop(0)
    write_checkpoint(tmp_path, seed=5)
    assert pack_experts.main([str(tmp_path)]) == 0
    assert (tmp_path / "experts.pack").is_file() and (tmp_path / "experts.pack.json").is_file()
    assert pack_experts.main([str(tmp_path), "--check"]) == 0
    assert "valid" in capsys.readouterr().out


def test_a_stale_pack_is_refused(hs, tmp_path, monkeypatch):
    write_checkpoint(tmp_path, seed=6)
    found, _ = pool.sources(tmp_path, names(False))
    packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    write_checkpoint(tmp_path, per_expert=True, seed=6)            # the checkpoint rewritten under the pack
    found, _ = pool.sources(tmp_path, names(True))
    with pytest.raises(packs.PackError, match="stale|differs"):
        packs.validate(tmp_path / packs.PACK_NAME, found, EXPERTS)
    s = make(hs, tmp_path, 2 * EXPERTS, native=True, pack=None, per_expert=True)   # opt-in: not used unasked
    assert s.pack_path is None and not s.stats()["packed"]
    s.close()
    for asked in (True, tmp_path / packs.PACK_NAME):
        with pytest.raises(packs.PackError):
            make(hs, tmp_path, 2 * EXPERTS, native=True, pack=asked, per_expert=True)
    for value in ("1", str(tmp_path / packs.PACK_NAME)):
        monkeypatch.setenv(packs.ENV, value)
        with pytest.raises(packs.PackError):
            make(hs, tmp_path, 2 * EXPERTS, native=True, pack=None, per_expert=True)


def test_same_layout_new_bytes_is_refused(hs, tmp_path):
    """The checkpoint rewritten with other weights in the same tensor table (a re-download, a new fine-tune)."""

    write_checkpoint(tmp_path, seed=41)
    found, _ = pool.sources(tmp_path, names(False))
    packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    truth = write_checkpoint(tmp_path, seed=42)
    found, _ = pool.sources(tmp_path, names(False))
    with pytest.raises(packs.PackError):
        packs.validate(tmp_path / packs.PACK_NAME, found, EXPERTS)
    with pytest.raises(packs.PackError):
        make(hs, tmp_path, 2 * EXPERTS, native=True, pack=True)
    s = make(hs, tmp_path, 2 * EXPERTS, native=True)                 # unpacked serves the new bytes
    call(hs, s, "rows", 0, [1, 2, 3, 4])
    check_resident(hs, s, truth)
    s.close()


def test_new_bytes_behind_the_same_size_inode_and_mtime_are_caught_by_the_sample(tmp_path):
    write_checkpoint(tmp_path, seed=43)
    found, _ = pool.sources(tmp_path, names(False))
    packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    shard = min(tmp_path.glob("model-*.safetensors"))
    before = shard.stat()
    src, offset = pool.Streamer.source(found, 0, "gate", "weight", 0)      # a sampled part of expert 0
    assert Path(src.path) == shard
    with open(shard, "r+b") as f:                                          # in place: same inode and size
        f.seek(offset)
        f.write(bytes(255 - b for b in shard.read_bytes()[offset:offset + 16]))
    os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = shard.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)
    assert packs.source_hash(found) == json.loads(packs.index_path(tmp_path / packs.PACK_NAME).read_text())[
        "source_hash"]
    with pytest.raises(packs.PackError, match="differ"):
        packs.validate(tmp_path / packs.PACK_NAME, found, EXPERTS)


def test_a_checkpoint_changing_while_it_is_packed_leaves_no_pack(tmp_path):
    write_checkpoint(tmp_path, seed=45)
    found, _ = pool.sources(tmp_path, names(False))
    shard = min(tmp_path.glob("model-*.safetensors"))

    def touch(line):
        if line.startswith("layer 0:"):                                  # mid-build: a file is replaced in place
            st = shard.stat()
            os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    with pytest.raises(packs.PackError, match="changed while"):
        packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME, progress=touch)
    assert not list(tmp_path.glob("experts.pack*"))


def test_a_corrupt_pack_record_is_refused(tmp_path):
    write_checkpoint(tmp_path, seed=44)
    found, _ = pool.sources(tmp_path, names(False))
    out = packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    lay = packs.validate(out, found, EXPERTS)
    with open(out, "r+b") as f:
        f.seek(lay.record(2, EXPERTS - 1) + lay.part_offsets[4])
        f.write(b"\xff" * 8)
    with pytest.raises(packs.PackError, match="corrupt"):
        packs.validate(out, found, EXPERTS)


def test_a_truncated_pack_is_refused(tmp_path):
    write_checkpoint(tmp_path)
    found, _ = pool.sources(tmp_path, names(False))
    out = packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    with open(out, "r+b") as f:
        f.truncate(packs.ALIGNMENT)
    with pytest.raises(packs.PackError, match="bytes"):
        packs.validate(out, found, EXPERTS)


@pytest.mark.parametrize("alignment", [1000, 2048, 3 * 4096])
def test_an_odd_alignment_is_refused(hs, tmp_path, alignment, capsys):
    write_checkpoint(tmp_path)
    found, _ = pool.sources(tmp_path, names(False))
    with pytest.raises(packs.PackError, match="power of two"):
        packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME, alignment=alignment)
    out = packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    index = json.loads(packs.index_path(out).read_text())
    index["alignment"] = alignment                                        # an index edited or from another build
    packs.index_path(out).write_text(json.dumps(index))
    with pytest.raises(packs.PackError, match="power of two"):
        packs.validate(out, found, EXPERTS)
    with pytest.raises(packs.PackError):
        make(hs, tmp_path, 2 * EXPERTS, native=True, pack=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    try:
        import pack_experts
    finally:
        sys.path.pop(0)
    with pytest.raises(SystemExit):
        pack_experts.main([str(tmp_path), "--alignment", str(alignment)])
    assert "power of two" in capsys.readouterr().err


def test_packs_are_opt_in(tmp_path):
    write_checkpoint(tmp_path)
    found, _ = pool.sources(tmp_path, names(False))
    packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    assert packs.choose(tmp_path, {}) is None                              # a pack beside the checkpoint: unused
    for off in ("0", "off", ""):
        assert packs.choose(tmp_path, {packs.ENV: off}) is None
    for on in ("1", "true"):
        assert packs.choose(tmp_path, {packs.ENV: on}) == tmp_path / packs.PACK_NAME
    assert packs.choose(tmp_path, {packs.ENV: "/x/y.pack"}) == Path("/x/y.pack")
    assert packs.choose(tmp_path, {packs.ENV: "1"}, pack=False) is None
    assert packs.choose(tmp_path, {}, pack=True) == tmp_path / packs.PACK_NAME


@pytest.mark.parametrize("per_expert", [False, True])
def test_packed_reads_land_the_same_bytes_as_unpacked_and_python(hs, tmp_path, per_expert):
    truth = write_checkpoint(tmp_path, per_expert, seed=7)
    found, _ = pool.sources(tmp_path, names(per_expert))
    packs.build(found, EXPERTS, tmp_path / packs.PACK_NAME)
    py = make(hs, tmp_path, 2 * EXPERTS + 1, native=False, per_expert=per_expert)
    plain = make(hs, tmp_path, 2 * EXPERTS + 1, native=True, pack=False, per_expert=per_expert)
    packed = make(hs, tmp_path, 2 * EXPERTS + 1, native=True, pack=True, per_expert=per_expert, workers=3)
    assert packed.stats()["packed"] == 1 and packed.pack_path == tmp_path / packs.PACK_NAME
    for kind, layer, ids in SEQUENCE + random_sequence(9, 120):
        for s in (py, plain, packed):
            call(hs, s, kind, layer, ids)
    a, b, c = (snapshot(hs, s) for s in (py, plain, packed))
    for key in a:
        assert a[key] == b[key] == c[key], key
    check_resident(hs, packed, truth)
    for s in (py, plain, packed):
        s.close()
