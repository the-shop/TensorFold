"""Time SSDTable.gather with and without its row cache (--ple-cache-mib) on a synthetic 8-bit n-gram table.

    python tools/bench_ple_cache.py [--dir DIR] [--rows-per-shard N] [--cache-mib MIB]

Writes a table shaped like Flash Next's 8-bit one (160-byte words, 10-byte scales and biases a row) into DIR
(a temporary folder by default, removed after), then times decode-shaped gathers of 16 ids at several shares of
cached rows, a prompt chunk's gather, and the cache's own bookkeeping. Every cached gather is also checked byte
for byte against the uncached one. Not a test: the numbers depend on the disk and the machine.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from tensorfold.families.qwen4_exp.ssd_table import RowCache, SSDTable

PARTS = (("weight", "U32", 40), ("scales", "BF16", 5), ("biases", "BF16", 5))     # 8-bit, group 32, 1280 dims


def write_table(folder: Path, shards: int, rows: int, files: int = 2) -> list[tuple]:
    """``shards`` shards of ``rows`` random rows over ``files`` safetensors files; SSDTable's file list."""

    rng = np.random.default_rng(0)
    entries: list[tuple | None] = [None] * shards
    for f in range(files):
        mine = [i for i in range(shards) if i % files == f]
        header, at = {}, 0
        for i in mine:
            for part, dtype, cols in PARTS:
                size = rows * cols * (4 if dtype == "U32" else 2)
                header[f"emb.shard_{i}.{part}"] = {"dtype": dtype, "shape": [rows, cols],
                                                   "data_offsets": [at, at + size]}
                at += size
        text = json.dumps(header).encode()
        path = folder / f"model-{f:05d}.safetensors"
        with open(path, "wb") as out:
            out.write(struct.pack("<Q", len(text)) + text)
            for i in mine:
                for part, dtype, cols in PARTS:
                    size = rows * cols * (4 if dtype == "U32" else 2)
                    out.write(rng.integers(0, 256, size, dtype=np.uint8).tobytes())
        for i in mine:
            entries[i] = (path, *(header[f"emb.shard_{i}.{p}"] for p, _, _ in PARTS))
    return entries


def timed(table: SSDTable, batches: list[np.ndarray]) -> np.ndarray:
    out = []
    for ids in batches:
        start = time.perf_counter()
        table.gather(ids)
        out.append(time.perf_counter() - start)
    return np.array(out) * 1e3


def decode_ids(rng, rows: int, hot: np.ndarray, share: float) -> np.ndarray:
    """16 ids, one per head's slice of the table as Flash Next's lookups fall; ``share`` of them from ``hot``."""

    width = rows // 16
    ids = np.arange(16) * width + rng.integers(0, width, 16)
    pick = rng.random(16) < share
    ids[pick] = rng.choice(hot, int(pick.sum()))
    return ids.astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="where to write the synthetic table (default: a temporary folder)")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--rows-per-shard", type=int, default=100_000)
    ap.add_argument("--cache-mib", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    folder = Path(args.dir) if args.dir else Path(tempfile.mkdtemp(prefix="ple-cache-bench-"))
    folder.mkdir(parents=True, exist_ok=True)
    try:
        files = write_table(folder, args.shards, args.rows_per_shard)
        off = SSDTable(files)
        on = SSDTable(files, cache_bytes=args.cache_mib * 1024**2)
        rows = off.rows
        print(f"table: {rows:,} rows x {off.wrow + 2 * off.grow} B ({off.nbytes / 2**20:.0f} MiB), F_NOCACHE reads; "
              f"cache {on.cache.nbytes / 2**20:.0f} MiB = {on.cache.slots:,} rows")
        rng = np.random.default_rng(1)
        hot = rng.choice(rows, min(on.cache.slots, 50_000), replace=False).astype(np.int64)
        on.gather(hot)                                            # these rows are cached from here on
        for share in (0.0, 0.5, 0.75, 1.0):
            batches = [decode_ids(rng, rows, hot, share) for _ in range(args.steps)]
            for ids in batches[:20]:
                a, b = off.gather(ids), on.gather(ids)
                assert all(x.tobytes() == y.tobytes() for x, y in zip(a, b)), "cache changed the bytes"
            t_off, t_on = timed(off, batches), timed(on, batches)
            print(f"decode 16 ids, {share:4.0%} of rows cached: off median {np.median(t_off):.3f} ms "
                  f"p90 {np.percentile(t_off, 90):.3f} | on median {np.median(t_on):.3f} ms "
                  f"p90 {np.percentile(t_on, 90):.3f}")
        chunk = [rng.integers(0, rows, 32768).astype(np.int64) for _ in range(3)]
        print(f"prompt chunk 32768 ids (cold): off {np.median(timed(off, chunk)):.1f} ms | "
              f"on {np.median(timed(on, chunk)):.1f} ms (on, same chunks again: {np.median(timed(on, chunk)):.1f} ms)")

        cache = RowCache(args.cache_mib * 1024**2, (off.wrow, off.grow, off.grow))
        keys = np.unique(rng.choice(rows, 16, replace=False)).astype(np.int64)
        parts = [np.zeros((keys.size, w), np.uint8) for w in cache.widths]
        outs = [np.empty((keys.size, w), np.uint8) for w in cache.widths]
        cache.put(keys, parts)
        n = 20_000
        start = time.perf_counter()
        for _ in range(n):
            cache.take(keys, outs)
        take = (time.perf_counter() - start) / n * 1e6
        fresh = [np.unique(rng.choice(rows, 16, replace=False)).astype(np.int64) for _ in range(n)]
        start = time.perf_counter()
        for k in fresh:
            cache.put(k, [np.zeros((k.size, w), np.uint8) for w in cache.widths])
        put = (time.perf_counter() - start) / n * 1e6
        print(f"bookkeeping, 16 rows: take (all hit) {take:.1f} us, put (16 new, evicting) {put:.1f} us")
        off.close()
        on.close()
    finally:
        if not args.dir:
            shutil.rmtree(folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
