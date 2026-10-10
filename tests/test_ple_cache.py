"""The n-gram (PLE) row cache in front of SSDTable: the bytes of the files, its arena's size, reads only for misses."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from tests.test_ple_ssd import COUNTS, TEXT, _checkpoint, _recorded, _same, mlx_cpu  # noqa: F401
from tensorfold.families.qwen4_exp import host_table, ssd_table
from tensorfold.families.qwen4_exp.ssd_table import RowCache, SSDTable

ROW = 80 + 10 + 10              # the test shards: 20 words, 5 scales and 5 biases a row


def _bytes(rows: int) -> int:
    """A cache of exactly ``rows`` test rows, index included."""

    return RowCache.bytes_for(rows, (80, 10, 10))


@pytest.fixture
def tables():
    made = []
    yield lambda files, **kw: made.append(SSDTable(files, **kw)) or made[-1]
    for table in made:
        table.close()


@pytest.fixture(params=[0, 64, 10**9], ids=["numpy", "split", "python"])
def probes(request, monkeypatch):
    """The cache's index probed through numpy always, past 64 ids, or one id at a time always."""

    monkeypatch.setattr(ssd_table, "SMALL", request.param)
    return request.param


def _cases(rows: int) -> list:
    inner = np.cumsum(COUNTS)[:-1]
    edges = np.concatenate([inner - 1, inner, [0, rows - 1]])
    rng = np.random.default_rng(11)
    return [rng.integers(0, rows, (13, 16)),                     # random
            np.array([5, 5, 5, 200, 5, 0, 200]),                 # repeated within a gather
            np.array([5, 200, 0]),                               # all of them again: every row hits
            edges, edges[::-1],                                  # both sides of every shard boundary
            np.arange(rows), np.arange(rows)[::-1],              # every row: past a small arena
            np.array(7), np.empty((0, 16), dtype=np.int64), []]


@pytest.mark.parametrize("cache_rows", [1, 7, 64, sum(COUNTS), 10 * sum(COUNTS)])
def test_cached_rows_equal_the_files_byte_for_byte(tmp_path, tables, cache_rows, probes):
    files, table = _checkpoint(tmp_path)
    off, on = tables(files), tables(files, cache_bytes=_bytes(cache_rows))
    held = min(cache_rows, on.rows)                              # past the table: capped at its rows
    assert off.cache is None and on.cache.slots == held and on.cache.nbytes == _bytes(held)
    for _ in range(2):                                           # the second pass reads what the first kept
        for ids in _cases(on.rows):
            flat = np.asarray(ids, dtype=np.int64).reshape(-1)
            want = tuple(part[flat] for part in table)
            _same(off.gather(ids), want)
            _same(on.gather(ids), want)
            assert len(on.cache) <= cache_rows


def test_the_arena_is_sized_from_the_tables_rows_and_never_grows(tmp_path, tables):
    files, _ = _checkpoint(tmp_path)
    ssd = tables(files, cache_bytes=_bytes(10) + ROW - 1)        # 10 rows and their index fit; an 11th does not
    cache = ssd.cache
    assert (cache.widths, cache.slots, cache.nbytes, cache.arena.shape) == ((80, 10, 10), 10, _bytes(10), (10, ROW))
    arena = cache.arena
    rng = np.random.default_rng(2)
    for _ in range(50):
        ssd.gather(rng.integers(0, ssd.rows, (4, 16)))
        assert len(cache) == 10 and cache.arena is arena and arena.shape == (10, ROW)
    with pytest.raises(ValueError, match="holds no 100-byte row"):
        SSDTable(files, cache_bytes=ROW - 1)


@pytest.mark.parametrize("mib", [1, 64])
def test_every_byte_the_cache_holds_is_in_its_fixed_arrays_within_the_budget(mib, probes):
    cache = RowCache(mib * 2**20, (160, 10, 10))                 # the 8-bit Flash Next row
    arrays = (cache.arena, cache._buckets, cache._ids, cache._where, cache._ref)
    assert cache.nbytes == sum(a.nbytes for a in arrays) <= mib * 2**20
    assert cache.arena.nbytes > 0.8 * mib * 2**20                # the index is a small share
    assert cache._buckets.size >= 4 * cache.slots                # a quarter full at most: short probes
    keys = np.unique(np.random.default_rng(0).integers(0, 320_000_000, 3 * cache.slots))
    for chunk in np.array_split(keys, 7):                        # fill, evict, re-index past the tombstone bound
        cache.put(chunk, [np.zeros((chunk.size, w), np.uint8) for w in cache.widths])
        assert len(cache) <= cache.slots
    assert cache.nbytes == sum(a.nbytes for a in (cache.arena, cache._buckets, cache._ids, cache._where, cache._ref))
    held = cache.items()
    assert len(held) == len(cache) == cache.slots and sorted(held.values()) == list(range(cache.slots))
    found = cache._find(np.array(sorted(held), dtype=np.int64))
    assert found.tolist() == [held[k] for k in sorted(held)]


def test_no_cache_without_bytes(tmp_path, tables):
    files, _ = _checkpoint(tmp_path)
    assert tables(files).cache is None and tables(files, cache_bytes=0).cache is None
    read = host_table.from_checkpoint(tmp_path, "emb", len(COUNTS), ssd=True)
    kept = host_table.from_checkpoint(tmp_path, "emb", len(COUNTS), ssd=True, cache_bytes=_bytes(4))
    assert read.cache is None and kept.cache.slots == 4
    read.close()
    kept.close()


def test_rows_that_hit_are_not_read_again_and_misses_alone_are_read(tmp_path, monkeypatch, tables):
    files, table = _checkpoint(tmp_path, counts=(10_000,))
    ssd = tables(files, cache_bytes=_bytes(100))
    reads = _recorded(monkeypatch)
    ssd.gather([9999, 5, 6, 5])
    assert len(reads) == 6                                       # 5-6 and 9999, three components each
    reads.clear()
    pool = ssd._pool.map
    monkeypatch.setattr(ssd._pool, "map", lambda *a: pytest.fail("the thread pool ran for rows all cached"))
    _same(ssd.gather([6, 9999, 5, 9999]), tuple(part[[6, 9999, 5, 9999]] for part in table))
    assert reads == []
    monkeypatch.setattr(ssd._pool, "map", pool)
    _same(ssd.gather([5, 7, 9999, 4000]), tuple(part[[5, 7, 9999, 4000]] for part in table))
    path, *entries = files[0]
    data = 8 + int.from_bytes(path.read_bytes()[:8], "little")
    want = sorted(data + entry["data_offsets"][0] + row * width
                  for entry, width in zip(entries, (80, 10, 10)) for row in (7, 4000))
    assert sorted(offset for _, offset, _ in reads) == want
    assert (ssd.cache.hits, ssd.cache.misses) == (3 + 2, 3 + 2)


def test_rows_used_since_the_clock_hand_passed_get_a_second_chance(tmp_path, monkeypatch, tables, probes):
    files, table = _checkpoint(tmp_path)
    ssd = tables(files, cache_bytes=_bytes(3))
    for ids in ([1], [2], [3], [4]):                             # all referenced: one sweep clears them, 1 goes
        ssd.gather(ids)
    assert ssd.cache.items() == {4: 0, 2: 1, 3: 2}
    ssd.gather([2])                                              # 2 used again: the hand passes it, 3 goes
    ssd.gather([5])
    assert ssd.cache.items() == {4: 0, 2: 1, 5: 2}
    reads = _recorded(monkeypatch)
    _same(ssd.gather([2, 4, 5]), tuple(part[[2, 4, 5]] for part in table))
    assert reads == []
    _same(ssd.gather([3]), tuple(part[[3]] for part in table))
    assert len(reads) == 3 and len(ssd.cache) == 3


def test_a_gather_wider_than_the_arena_keeps_its_last_rows(tmp_path, tables):
    files, table = _checkpoint(tmp_path)
    ssd = tables(files, cache_bytes=_bytes(5))
    ids = np.arange(40, 100)
    _same(ssd.gather(ids), tuple(part[ids] for part in table))
    assert sorted(ssd.cache.items()) == list(range(95, 100))
    for key, slot in ssd.cache.items().items():                   # each slot holds its own row's bytes
        packed = np.concatenate([part[key].view(np.uint8) for part in table])
        assert ssd.cache.arena[slot].tobytes() == packed.tobytes()


@pytest.mark.parametrize("cache_rows", [8, 40, 1000])
def test_concurrent_gathers_through_one_cache_each_get_their_own_rows(tmp_path, tables, cache_rows, probes):
    files, table = _checkpoint(tmp_path)
    ssd, start = tables(files, cache_bytes=_bytes(cache_rows)), Barrier(8)

    def run(seed: int) -> None:
        rng = np.random.default_rng(seed)
        start.wait(timeout=30)
        for _ in range(60):
            ids = rng.integers(0, 40, (int(rng.integers(1, 9)), 16))       # a narrow range: hits, misses, evictions
            if rng.random() < 0.3:
                ids = rng.integers(0, ssd.rows, (int(rng.integers(1, 9)), 16))
            flat = ids.reshape(-1)
            _same(ssd.gather(ids), tuple(part[flat] for part in table))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(run, range(8)))
    cache = ssd.cache
    held = cache.items()
    assert len(cache) == len(held) <= cache_rows and cache.hits > 0 and len(set(held.values())) == len(held)
    assert cache._find(np.array(list(held), dtype=np.int64)).tolist() == list(held.values())
    for key, slot in held.items():
        packed = np.concatenate([part[key].view(np.uint8) for part in table])
        assert cache.arena[slot].tobytes() == packed.tobytes()


def test_put_skips_rows_another_gather_kept_meanwhile():
    cache = RowCache(RowCache.bytes_for(3, (4, 1, 1)), (4, 1, 1))
    parts = [np.full((2, 4), 9, np.uint8), np.full((2, 1), 8, np.uint8), np.full((2, 1), 7, np.uint8)]
    cache.put(np.array([10, 11]), parts)
    cache.put(np.array([11, 12]), [p + 1 for p in parts])
    held = cache.items()
    assert sorted(held) == [10, 11, 12]
    assert cache.arena[held[11]].tolist() == [9, 9, 9, 9, 8, 7]
    assert cache.arena[held[12]].tolist() == [10, 10, 10, 10, 9, 8]


def test_close_drops_the_cache(tmp_path):
    files, _ = _checkpoint(tmp_path)
    ssd = SSDTable(files, cache_bytes=_bytes(8))
    ssd.gather([1, 2])
    ssd.close()
    assert ssd.cache is None
    with pytest.raises(ValueError, match="closed"):
        ssd.gather([1])
    assert os.path.exists(files[0][0])


def _flash_next(folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text('{"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 32}}')


@pytest.mark.parametrize("flags, error", [
    (["--ple-on-ssd", "--ple-cache-mib", "-1"], "0 or a positive MiB count"),
    (["--ple-cache-mib", "64"], "only with --ple-on-ssd"),
    (["--ple-on-ssd", "--ple-cache-mib", "64", "--backend", "cuda"], "Mac \\(MLX\\) loader only"),
])
def test_the_flag_is_refused_where_it_cannot_cache(tmp_path, monkeypatch, capsys, flags, error):
    import re

    from tensorfold import cli, hub

    _flash_next(tmp_path / "flash")
    for name in ("pull", "resolve"):
        monkeypatch.setattr(hub, name, lambda *a, **k: pytest.fail("went past the flag check"))
    assert cli.main(["serve", str(tmp_path / "flash"), "--no-update-check", *flags]) == 1
    assert re.search(error, capsys.readouterr().err)


def test_the_flag_reaches_the_metal_loader_only_when_given(tmp_path, mlx_cpu):
    from types import SimpleNamespace

    from tensorfold import cli

    class Reached(Exception):
        pass

    def load(path, **options):
        seen.append(options)
        raise Reached

    seen = []
    family = SimpleNamespace(title="fixture", model_type="qwen4_exp", package=SimpleNamespace(load=load))
    for flag in ([], ["--ple-on-ssd"], ["--ple-on-ssd", "--ple-cache-mib", "512"]):
        args = cli.build_parser().parse_args(["serve", str(tmp_path), "--snapshot-dir", "none", *flag])
        with pytest.raises(Reached):
            cli._serve_mlx(args, family, tmp_path, 0, [], 1 << 30)
    assert "ple_cache_mib" not in seen[0] and "ple_cache_mib" not in seen[1] and seen[2]["ple_cache_mib"] == 512


def test_greedy_tokens_and_logits_are_the_same_with_the_cache_on(tmp_path, monkeypatch, mlx_cpu):
    """A tiny Flash Next on the CPU device decodes the same greedy tokens, bit-equal logits, cache on or off."""

    mx = mlx_cpu
    import json

    import mlx.nn as nn
    import mlx_lm.utils
    from mlx.utils import tree_flatten

    from tensorfold.families.qwen4_exp import model as q4

    config = {"model_type": "qwen4_exp", "text_config": TEXT, "quantization": {"group_size": 32, "bits": 4}}
    mx.random.seed(0)
    model = q4.Qwen4Exp(q4.Config.from_dict(config))
    for shard in model.layers[1].ple.ple_embedding.shards:
        shard.weight = shard.weight.astype(mx.bfloat16)
    nn.quantize(model, group_size=32, bits=4, class_predicate=lambda path, _: ".ple_embedding.shards." in path)
    params = {"language_model." + k.replace(".ple_embedding.shards.", ".ple_embedding.ngram_embedding.shard_"): v
              for k, v in tree_flatten(model.parameters())}
    owner = {n: int(n.split(".shard_")[1].split(".")[0]) if ".shard_" in n else i for i, n in enumerate(params)}
    for f in range(2):
        mx.save_safetensors(str(tmp_path / f"model-0000{f + 1}-of-00002.safetensors"),
                            {n: a for n, a in params.items() if owner[n] % 2 == f}, metadata={"format": "mlx"})
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: "tokenizer")
    monkeypatch.setenv("TF_FLASH_FUSED", "0")
    monkeypatch.delenv("TF_NGRAM_HOST", raising=False)

    def decode(slots: int | None = None, **options) -> tuple[list[int], list, object]:
        loaded, _ = q4.load(tmp_path, ple_on_ssd=True, **options)
        table = loaded.layers[1].ple.ple_embedding.host.table
        caches = loaded.__dict__.get("ple_caches")                # what /health reads: only with the flag
        assert caches == ({"model.layers.1.ple.ple_embedding": table} if options else None)
        if slots is not None:                                    # a few rows only: every step evicts
            table.cache = RowCache(slots * sum(table.cache.widths), table.cache.widths)
        cache = loaded.make_cache()
        logits = [loaded(np.array([[11, 42, 5, 17, 42, 11, 60, 3, 5, 5, 90, 17]]), cache)]
        tokens = []
        for _ in range(16):
            tokens.append(int(mx.argmax(logits[-1][0, -1]).item()))
            logits.append(loaded(np.array([[tokens[-1]]]), cache))
        logits.append(loaded(np.array([[11, 42, 5, 17] + tokens[:4]]), cache))      # n-grams seen before: hits
        return tokens, logits, table

    off_tokens, off_logits, off = decode()
    assert off.cache is None
    for slots in (None, 3):
        on_tokens, on_logits, on = decode(slots, ple_cache_mib=1)
        assert on.cache is not None and len(on.cache) <= on.cache.slots
        assert on.cache.hits > 0 if slots is None else on.cache.misses > 2 * on.cache.slots    # hits, or evictions
        assert on_tokens == off_tokens
        assert all(mx.array_equal(a, b).item() for a, b in zip(off_logits, on_logits, strict=True))
        on.close()
    off.close()


def test_the_cache_counts_against_the_memory_budget(tmp_path, monkeypatch, capsys):
    import json
    import struct
    import sys
    from types import ModuleType

    from tensorfold import cli
    from tensorfold.families import qwen4_exp
    from tensorfold.server.memory_budget import GIB, PROCESS_BYTES

    folder = tmp_path / "flash"
    _flash_next(folder)
    ple = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    rows = 3 * GIB // 80                                       # room for a 2 GiB cache: not capped below
    header = {ple: {"dtype": "U32", "shape": [rows, 20], "data_offsets": [0, rows * 80]},
              "language_model.model.norm.weight": {"dtype": "BF16", "shape": [512],
                                                   "data_offsets": [rows * 80, rows * 80 + 1024]}}
    text = json.dumps(header).encode()
    with open(folder / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text)
        f.truncate(8 + len(text) + rows * 80 + 1024)           # sparse: 3 GiB of tables on SSD, 1 KiB of weights
    limits = []
    core = ModuleType("mlx.core")
    core.set_memory_limit = limits.append
    core.set_cache_limit = lambda value: None
    core.device_info = lambda: {"max_recommended_working_set_size": 64 * GIB, "memory_size": 128 * GIB}
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    for key, value in qwen4_exp.MLX_ENV.items():
        monkeypatch.setenv(key, os.environ.get(key, value))
    monkeypatch.setenv("TENSORFOLD_MEMORY_LIMIT_GB", "5")
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: None)

    class Reached(Exception):
        pass

    served = []
    monkeypatch.setattr(cli, "_serve_mlx", lambda *a, **k: served.append(a) or _raise(Reached))
    args = cli.build_parser().parse_args(["serve", str(folder), "--no-update-check", "--backend", "mlx",
                                          "--ple-on-ssd"])
    assert cli._host_reserve(args) == PROCESS_BYTES
    with pytest.raises(Reached):
        cli.cmd_serve(args)
    assert limits[-1] == 5 * GIB - PROCESS_BYTES               # no cache: MLX's share as before
    args.ple_cache_mib = 1536
    assert cli._host_reserve(args) == PROCESS_BYTES + 1536 * 2**20
    with pytest.raises(Reached):
        cli.cmd_serve(args)
    assert limits[-1] == 5 * GIB - PROCESS_BYTES - 1536 * 2**20  # the arena comes out of MLX's share
    args.ple_cache_mib = 2048                                   # 3 GiB + 2 GiB: nothing left for the weights
    with pytest.raises(ValueError, match="do not fit"):
        cli.cmd_serve(args)
    assert len(served) == 2
    capsys.readouterr()
    monkeypatch.setattr(qwen4_exp, "ple_cache_bytes", lambda path, mib: 10 * 2**20)    # a table smaller than asked
    with pytest.raises(Reached):
        cli.cmd_serve(args)
    assert limits[-1] == 5 * GIB - PROCESS_BYTES - 10 * 2**20 and "reserves only that" in capsys.readouterr().out
    count = len(limits)
    monkeypatch.setattr(qwen4_exp, "ple_cache_bytes", lambda path, mib: _raise(ValueError("less than one row")))
    with pytest.raises(ValueError, match="less than one row"):
        cli.cmd_serve(args)
    assert len(limits) == count                                 # refused before MLX's limit was set or weights read


def _raise(exc):
    raise exc


def _check(cache: RowCache) -> None:
    """The index agrees with the slots: each live row once, findable at its slot, and the bucket counts add up."""

    live = np.flatnonzero(cache._ids >= 0)
    assert live.tolist() == list(range(len(cache))) and len(cache) <= cache.slots
    ids = cache._ids[live]
    assert np.unique(ids).size == ids.size
    assert (cache._buckets[cache._where[live]] == live).all()
    pointed = np.flatnonzero(cache._buckets >= 0)
    assert pointed.size == live.size and (cache._where[cache._buckets[pointed]] == pointed).all()
    assert int(np.count_nonzero(cache._buckets == ssd_table.GONE)) == cache._gone
    assert int(np.count_nonzero(cache._buckets != ssd_table.EMPTY)) == len(cache) + cache._gone
    assert cache._find(ids).tolist() == live.tolist()


def _parts(ids: np.ndarray, widths=(4, 1, 1)) -> list[np.ndarray]:
    """Rows whose bytes name their id, so a slot's bytes show which row it holds."""

    word = (ids.astype(np.uint32).view(np.uint8).reshape(-1, 4))
    return [np.tile(word, (1, w // 4)) if w >= 4 else word[:, :w] for w in widths]


def test_a_put_past_the_free_slots_evicts_only_rows_held_before_it(probes):
    cache = RowCache(RowCache.bytes_for(10, (4, 1, 1)), (4, 1, 1))
    first, second = np.arange(100, 108), np.arange(200, 205)
    cache.put(first, _parts(first))
    _check(cache)
    cache.put(second, _parts(second))
    _check(cache)
    held = cache.items()
    assert all(k in held for k in second.tolist())
    assert len([k for k in first.tolist() if k not in held]) == 3 and len(held) == 10
    for key, slot in held.items():
        assert cache.arena[slot, :4].view(np.uint32)[0] == key


@pytest.mark.parametrize("slots", [1, 2, 5, 10, 17, 64])
def test_the_index_stays_consistent_through_random_puts_and_takes(slots, probes):
    rng = np.random.default_rng(slots)
    for step in range(300):
        if step % 15 == 0:                                       # often part full: puts that cross into eviction
            cache = RowCache(RowCache.bytes_for(slots, (4, 1, 1)), (4, 1, 1))
        hi = int(rng.choice([3 * slots, 50 * slots, 10**9]))
        size = int(rng.choice([1, 2, slots // 2 + 1, slots, 3 * slots + 1]))
        ids = np.unique(rng.integers(0, hi, size))
        outs = [np.empty((ids.size, w), np.uint8) for w in (4, 1, 1)]
        miss = cache.take(ids, outs)
        hit = np.setdiff1d(np.arange(ids.size), miss)
        assert (outs[0][hit].view(np.uint32).reshape(-1) == ids[hit].astype(np.uint32)).all()
        cache.put(ids[miss], _parts(ids[miss]))
        _check(cache)
        for key, slot in cache.items().items():
            assert cache.arena[slot, :4].view(np.uint32)[0] == np.uint32(key)


@pytest.mark.parametrize("slots", [1, 2, 5, 17, 64])
def test_many_threads_of_mixed_gathers_keep_bytes_and_index_right(tmp_path, tables, slots, probes):
    files, table = _checkpoint(tmp_path)
    ssd, start = tables(files, cache_bytes=_bytes(slots)), Barrier(12)

    def run(seed: int) -> None:
        rng = np.random.default_rng(seed)
        start.wait(timeout=30)
        for _ in range(60):
            ids = rng.integers(0, int(rng.choice([20, 60, ssd.rows])), int(rng.choice([1, 3, 16, 70, 200])))
            _same(ssd.gather(ids), tuple(part[ids] for part in table))

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(run, range(12)))
    _check(ssd.cache)


@pytest.mark.parametrize("ids, slots, kept", [
    ([90, 3, 50, 7, 1, 2, 80], 5, {50, 7, 1, 2, 80}),         # the last five asked for, not the five highest ids
    ([1, 90, 2, 90], 2, {2, 90}),                             # 90 asked for last of all, 2 before it
    ([[9, 8], [7, 6], [9, 5]], 3, {6, 9, 5}),                 # in request order across a batch's rows
])
def test_a_gather_wider_than_the_arena_keeps_the_rows_it_used_last(tmp_path, tables, ids, slots, kept):
    files, table = _checkpoint(tmp_path)
    ssd = tables(files, cache_bytes=_bytes(slots))
    flat = np.asarray(ids).reshape(-1)
    _same(ssd.gather(ids), tuple(part[flat] for part in table))
    assert set(ssd.cache.items()) == kept
    _check(ssd.cache)
    partial = tables(files, cache_bytes=_bytes(slots))       # some rows cached already: misses alone are ranked
    partial.gather([flat[-1]])
    partial.gather(ids)
    assert set(partial.cache.items()) == kept


def test_the_cache_is_capped_at_the_table_and_refused_below_one_row(tmp_path, tables, capsys, monkeypatch):
    from tensorfold import cli
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.ssd_table import cache_bytes, cache_share
    from tensorfold.server.memory_budget import PROCESS_BYTES

    files, table = _checkpoint(tmp_path)
    whole = tables(files, cache_bytes=10**9)                    # far past the table: every row, no more
    assert whole.cache.slots == whole.rows and whole.cache.nbytes == _bytes(whole.rows)
    ids = np.arange(whole.rows)[::-1]
    _same(whole.gather(ids), tuple(part[ids] for part in table))
    assert len(whole.cache) == whole.rows

    widths = (80, 10, 10)
    assert cache_share(3, 2) == 3 * 2**20 // 2
    assert cache_bytes(1, [(10, widths), (10**9, widths)]) == _bytes(10) + 2**20 // 2
    with pytest.raises(ValueError, match="less than one 100-byte row"):
        cache_bytes(1, [(10, widths)] * 2**15)

    folder = tmp_path / "flash"
    _flash_next(folder)
    stem = "language_model.model.layers.{}.ple.ple_embedding.ngram_embedding.shard_{}.{}"
    from tests.test_ple_ssd import _write
    shards = {stem.format(layer, i, part): array for layer in (1, 2) for i, (w, sc, b) in enumerate(_shards_of(table))
              for part, array in zip(("weight", "scales", "biases"), (w, sc, b))}
    _write(folder / "model-00001-of-00001.safetensors", shards)
    rows = sum(COUNTS)
    assert qwen4_exp.ple_cache_bytes(folder, 1) == 2 * _bytes(rows) < 2**20      # two tables, each held whole
    assert qwen4_exp.ple_cache_bytes(folder, 1) == cache_bytes(1, [(rows, widths)] * 2)

    args = cli.build_parser().parse_args(["serve", str(folder), "--ple-on-ssd", "--ple-cache-mib", "64"])
    args.ple_cache_bytes = qwen4_exp.ple_cache_bytes(folder, 64)
    assert cli._host_reserve(args) == PROCESS_BYTES + 2 * _bytes(rows)


def _shards_of(table):
    """The test table's rows back as shards of COUNTS rows, as weight, scales and biases arrays."""

    bounds = np.cumsum((0,) + COUNTS)
    return [tuple(part[a:b] for part in table) for a, b in zip(bounds[:-1], bounds[1:])]


def test_counters_count_evictions_and_re_indexes():
    cache = RowCache(RowCache.bytes_for(4, (4, 1, 1)), (4, 1, 1))
    assert cache.stats() == {"hits": 0, "misses": 0, "evictions": 0, "rebuilds": 0, "rows": 0, "slots": 4,
                             "bytes": cache.nbytes}
    for start in range(0, 40, 2):                                # 2 new rows a put: evictions once full
        ids = np.arange(start, start + 2)
        cache.take(ids, [np.empty((2, w), np.uint8) for w in (4, 1, 1)])
        cache.put(ids, _parts(ids))
    stats = cache.stats()
    assert (stats["misses"], stats["evictions"], stats["rows"]) == (40, 36, 4) and stats["rebuilds"] > 0
    _check(cache)


def test_health_reports_the_row_caches_only_when_they_exist(tmp_path, tables):
    import http.client
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.ssd_table import cache_stats
    from tensorfold.server.http import make_handler

    files, _ = _checkpoint(tmp_path)
    one, two = tables(files, cache_bytes=_bytes(8)), tables(files, cache_bytes=_bytes(20))
    one.gather([1, 2, 3])
    one.gather([1, 2, 9])
    two.gather(np.arange(20))
    two.gather(np.arange(20, 30))                                # full: 10 rows evicted
    assert cache_stats({"a": tables(files)}) is None

    def health(inner) -> dict:
        app = SimpleNamespace(served_name="t", model_ids=["t"], max_batch_size=1,
                              _model=SimpleNamespace(model=inner))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=3)
            connection.request("GET", "/health")
            return json.loads(connection.getresponse().read())
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join()

    assert "ple_cache" not in health(SimpleNamespace())
    got = health(SimpleNamespace(ple_caches={"layers.1": one, "layers.2": two}))["ple_cache"]
    assert got["tables"]["layers.1"] == {"hits": 2, "misses": 4, "evictions": 0, "rebuilds": 0, "rows": 4,
                                         "slots": 8, "bytes": _bytes(8)}
    assert got["tables"]["layers.2"]["evictions"] == 10 and got["tables"]["layers.2"]["rows"] == 20
    assert got["total"] == {k: got["tables"]["layers.1"][k] + got["tables"]["layers.2"][k] for k in got["total"]}
