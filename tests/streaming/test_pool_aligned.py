"""TENSORFOLD_POOL_ALIGNED_READS: the same expert bytes, read so macOS F_NOCACHE leaves none in the page cache."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.streaming import page_cache, pool

EXPERTS, TOPK = 8, 2
PROJS = ("gate", "up", "down")
# the real model's part sizes: a 1,638,400-byte weight (past the 1 MiB that unaligned F_NOCACHE reads leave
# uncached) and 51,200-byte scales and biases (read directly either way)
BIG = {"weight": ([640, 640], "U32", 4), "scales": ([640, 40], "BF16", 2), "biases": ([640, 40], "BF16", 2)}
darwin = pytest.mark.skipif(sys.platform != "darwin", reason="the F_NOCACHE page-cache residue is a macOS behaviour")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("TENSORFOLD_POOL_NATIVE", "TENSORFOLD_POOL_THREADS",
                 "TENSORFOLD_POOL_ALIGNED_READS"):
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


def write_checkpoint(folder: Path, layers: int = 1, pad: int = 1234) -> dict:
    """One shard whose tensors start off any 4 KiB boundary and whose last tensor (a weight) ends the file at an
    unaligned end; returns (layer, expert) -> {(proj, part): bytes}."""

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    truth: dict = {}
    tensors, data = {"__metadata__": {"pad": "x" * pad}}, bytearray()
    for layer in range(layers):
        for proj in PROJS:
            for part in ("scales", "biases", "weight"):             # a weight last: the file ends inside an expert run
                shape, dtype, size = BIG[part]
                one = int(np.prod(shape)) * size
                blob = rng.integers(0, 256, EXPERTS * one, dtype=np.uint8).tobytes()
                for e in range(EXPERTS):
                    truth.setdefault((layer, e), {})[(proj, part)] = blob[e * one:(e + 1) * one]
                name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{part}"
                tensors[name] = {"dtype": dtype, "shape": [EXPERTS, *shape],
                                 "data_offsets": [len(data), len(data) + len(blob)]}
                data += blob
    header = json.dumps(tensors).encode()
    path = folder / "model-00001-of-00001.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    assert (8 + len(header)) % 4096 and path.stat().st_size % 4096
    text = {"num_hidden_layers": layers, "num_experts": EXPERTS, "num_experts_per_tok": TOPK}
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text}))
    return truth


def names(layers: int = 1) -> dict:
    return {(layer, proj, part): f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{part}"
            for layer in range(layers) for proj in PROJS for part in BIG}


def make(hs, folder, native, layers=1, workers=4):
    found, shapes = pool.sources(folder, names(layers))
    s = pool.Streamer(found, shapes, layers=layers, experts=EXPERTS, top_k=TOPK, slots=2 * EXPERTS, box_rows=4,
                      hostsync=hs, workers=workers, native=native)
    assert s.native == native
    return s


def call(hs, s, kind, layer, ids):
    if kind == "window":
        words = [1 if e in ids else 0 for e in range(EXPERTS)]
    else:
        words = list(ids)
    hs.write_into(s.boxes, 4 * s.region_word(s.value + 1), struct.pack(f"<{len(words)}I", *words))
    signal, ready = s.marks(kind, layer, len(ids) // TOPK if kind == "rows" else 1)
    s.channel.signal(signal)
    assert s.channel.wait(ready, 20_000)


def part_bytes(hs, s, proj, part, slot) -> bytes:
    shape, _, size = BIG[part]
    n = int(np.prod(shape)) * size
    return hs.peek(getattr(s.pool[proj], part), slot * n, n)


def check_window(hs, s, truth, layer):
    for e in range(EXPERTS):
        for proj in PROJS:
            for part in BIG:
                assert part_bytes(hs, s, proj, part, e) == truth[(layer, e)][(proj, part)], (e, proj, part)


def expert_residency(ranges: dict) -> int:
    return sum(page_cache.resident(path, spans) for path, spans in ranges.items())


# -- the leak and its fix ---------------------------------------------------------------------------------------

@darwin
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("aligned", ["0", "1"])
def test_unaligned_part_reads_leave_no_page_cache(hs, tmp_path, monkeypatch, native, aligned):
    monkeypatch.setenv("TENSORFOLD_POOL_ALIGNED_READS", aligned)
    truth = write_checkpoint(tmp_path)
    ranges = page_cache.expert_ranges(tmp_path)       # reads the header (cached) before the eviction
    s = make(hs, tmp_path, native)
    for path in page_cache.model_files(tmp_path):
        page_cache.evict(path)
    assert expert_residency(ranges) == 0
    try:
        call(hs, s, "window", 0, list(range(EXPERTS)))
        assert s.error is None
        check_window(hs, s, truth, 0)
        read = s.bytes_read
        left = expert_residency(ranges)
    finally:
        s.close()
    if aligned == "1":
        assert left <= 0.01 * read, f"{left} of {read} bytes left in the page cache"
    else:                                   # today's reads: all but the first MiB of each weight stays cached (~34%)
        assert left > 0.25 * read, f"only {left} of {read} bytes cached: the probe no longer sees the leak"


# -- the same bytes ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("aligned", [False, True])
def test_pread_into_bytes_match_at_any_offset_and_length(hs, tmp_path, aligned):
    path = tmp_path / "blob"
    blob = np.random.default_rng(3).integers(0, 256, 6 << 20, dtype=np.uint8).tobytes()
    path.write_bytes(blob)
    fd = pool._uncached(path)
    edge = (1 << 20) - 16384
    dst = mx.zeros((4 << 20,), dtype=mx.uint8)
    mx.eval(dst)
    try:
        for offset in (0, 1, 4095, 4096, 12344, 3 << 20, 16383):
            for n in (1, 4096, 4097, 51_200, edge, edge + 1, 1_638_400, 1_638_401, 2 << 20):
                for at in (0, 56, 4096):
                    hs.pread_into(dst, at, fd, offset, n, aligned=aligned)
                    assert hs.peek(dst, at, n) == blob[offset:offset + n], (offset, n, at)
        for n in (1, 4097, 1_638_401, 3 << 20):                           # a range ending at the end of the file
            hs.pread_into(dst, 56, fd, len(blob) - n, n, aligned=aligned)
            assert hs.peek(dst, 56, n) == blob[-n:], n
        with pytest.raises(RuntimeError, match="short read"):                # past the end: an error, not garbage
            hs.pread_into(dst, 0, fd, len(blob) - 1_638_400 + 1, 1_638_400, aligned=aligned)
    finally:
        os.close(fd)


@pytest.mark.parametrize("native", [True, False])
def test_switch_reads_the_same_slots_and_counts(hs, tmp_path, monkeypatch, native):
    truth = write_checkpoint(tmp_path, layers=2)
    snaps = []
    for aligned in ("0", "1"):
        monkeypatch.setenv("TENSORFOLD_POOL_ALIGNED_READS", aligned)
        s = make(hs, tmp_path, native, layers=2)
        try:
            call(hs, s, "window", 1, [0, 3, 7])
            call(hs, s, "rows", 0, [1, 5, 1, 7])
            call(hs, s, "rows", 1, [7, 2, 5, 0])
            call(hs, s, "window", 0, list(range(EXPERTS)))
            assert s.error is None
            check_window(hs, s, truth, 0)
            for (layer, e), slot in s.lru_items():
                for proj in PROJS:
                    for part in BIG:
                        assert part_bytes(hs, s, proj, part, slot) == truth[(layer, e)][(proj, part)]
            snaps.append((s.stats(), s.lru_items(), s.free_slots(), s.window_layers(),
                          [part_bytes(hs, s, proj, part, slot) for proj in PROJS for part in BIG
                           for slot in range(2 * EXPERTS)]))
        finally:
            s.close()
    assert snaps[0] == snaps[1]                         # counters too: still one pread a part (native)


def test_a_part_cut_short_by_the_end_of_file_fails(hs, tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_POOL_ALIGNED_READS", "1")
    write_checkpoint(tmp_path)
    path = page_cache.model_files(tmp_path)[0]
    with open(path, "r+b") as f:
        f.truncate(path.stat().st_size - 1)
    s = make(hs, tmp_path, native=True)
    try:
        call(hs, s, "window", 0, [EXPERTS - 1])
        assert isinstance(s.error, RuntimeError) and "short read" in str(s.error)
    finally:
        s.close()


def test_aligned_reads_switch(monkeypatch):
    assert pool.aligned_reads({}) is True                                 # on by default
    assert pool.aligned_reads({"TENSORFOLD_POOL_ALIGNED_READS": ""}) is True
    for on in ("1", "on", "yes", "true", " TRUE "):
        assert pool.aligned_reads({"TENSORFOLD_POOL_ALIGNED_READS": on}) is True
    for off in ("0", "off", "no", "false"):
        assert pool.aligned_reads({"TENSORFOLD_POOL_ALIGNED_READS": off}) is False


@darwin
def test_page_cache_skips_what_is_not_a_regular_file(tmp_path):
    write_checkpoint(tmp_path)
    (tmp_path / "stray.safetensors").mkdir()                              # mmap of a directory: EINVAL
    assert [p.name for p in page_cache.model_files(tmp_path)] == ["model-00001-of-00001.safetensors"]
    assert page_cache.main(["evict", str(tmp_path)]) == 0


# -- the page-cache tool ----------------------------------------------------------------------------------------

@darwin
def test_page_cache_tool_sees_and_evicts_cached_pages(tmp_path, capsys):
    write_checkpoint(tmp_path)
    path = page_cache.model_files(tmp_path)[0]
    path.read_bytes()                                                     # an ordinary cached read
    assert page_cache.resident(path) > 0.9 * path.stat().st_size
    spans = page_cache.expert_ranges(tmp_path)[path]
    assert sum(n for _, n in spans) == EXPERTS * len(PROJS) * sum(int(np.prod(s)) * z for s, _, z in BIG.values())
    assert page_cache.resident(path, spans) <= page_cache.resident(path)
    assert page_cache.main(["evict", str(tmp_path)]) == 0
    assert page_cache.resident(path) == 0
    assert page_cache.main(["resident", str(tmp_path), "--experts"]) == 0
    total = capsys.readouterr().out.strip().splitlines()[-1].split("\t")
    assert total[0] == "total" and int(total[1]) <= page_cache.PAGE     # the header page it read itself, at most
