"""Time the SSD expert pool's miss reads: the Python path, the native server unpacked, and the native server packed.

    python tools/bench_pool_io.py MODEL_DIR [--experts 512] [--per-call 16] [--threads 4,8,16,32] [--pack PATH]

Each run reads ``--experts`` random (layer, expert) pairs (a fresh sample per run, so no run reads another's bytes)
into a small pool, ``--per-call`` experts a call, as decode misses arrive: (a) python: the Streamer's reads (a
ThreadPoolExecutor mapping hostsync.pread_into over every part of every miss; its LRU and peek costs not counted),
(b) native: one PoolServer call per batch, signalled and awaited like the GPU does (LRU, reads, slot table),
(c) packed: the same with the expert pack (one pread an expert). Reads are uncached (F_NOCACHE). Pool memory is
about 2 x per-call experts plus a record buffer a thread for packed runs."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from tensorfold.streaming import pack as packs
from tensorfold.streaming.pool import _DTYPES, PARTS, Streamer, _uncached, sources


class Bench:
    """A virtual model: layer v of the bench has ``per_call`` experts, each a random real (layer, expert)."""

    def __init__(self, model_dir: Path, hs, per_call: int, pack: Path | None) -> None:
        self.hs, self.per_call = hs, per_call
        names, self.experts = packs.model_experts(model_dir)
        self.found, shapes = sources(model_dir, names)
        self.layers = sorted({key[0] for key in self.found})
        self.order = [(proj, part) for proj in sorted({key[1] for key in self.found}) for part in PARTS]
        slots = 2 * per_call
        self.dsts = [mx.zeros((slots, *shapes[o][0]), dtype=_DTYPES[shapes[o][1]]) for o in self.order]
        self.slot_of = mx.zeros((1,), dtype=mx.int32)
        self.box = mx.array(np.arange(max(per_call, 8), dtype=np.uint32))
        mx.eval(*self.dsts, self.slot_of, self.box)
        self.fds = {p: _uncached(p) for p in {s.path for s in self.found.values()}}
        self.sizes = [Streamer.source(self.found, self.layers[0], proj, part, 0)[0].nbytes for proj, part in self.order]
        self.expert_bytes = sum(self.sizes)
        self.lay = None
        if pack is not None:
            self.lay = packs.validate(pack, self.found, self.experts)
            self.pack_fd = _uncached(pack)

    def sample(self, n: int, seed: int) -> list[tuple[int, int]]:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(self.layers) * self.experts, size=n, replace=False)
        return [(self.layers[int(p) // self.experts], int(p) % self.experts) for p in picks]

    def python(self, picks: list, threads: int) -> float:
        slots = list(range(self.per_call, 2 * self.per_call))
        with ThreadPoolExecutor(threads) as readers:
            start = time.perf_counter()
            for at in range(0, len(picks), self.per_call):
                need = list(zip(picks[at:at + self.per_call], slots))
                reads = [(p, layer, e, slot) for (layer, e), slot in need for p in range(len(self.order))]

                def read(item):
                    p, layer, e, slot = item
                    src, offset = Streamer.source(self.found, layer, *self.order[p], e)
                    self.hs.pread_into(self.dsts[p], slot * src.nbytes, self.fds[src.path], offset, src.nbytes)

                list(readers.map(read, reads))
            return time.perf_counter() - start

    def native(self, picks: list, threads: int, packed: bool) -> tuple[float, int]:
        calls = -(-len(picks) // self.per_call)
        fds = np.full((calls, self.per_call, len(self.order)), -1, dtype=np.int32)
        offsets = np.zeros_like(fds, dtype=np.int64)
        records = np.full((calls, self.per_call), -1, dtype=np.int64)
        for i, (layer, e) in enumerate(picks):
            c, v = divmod(i, self.per_call)
            for p, (proj, part) in enumerate(self.order):
                src, offset = Streamer.source(self.found, layer, proj, part, e)
                fds[c, v, p], offsets[c, v, p] = self.fds[src.path], offset
            if packed:
                records[c, v] = self.lay.record(layer, e)
        slot_of = mx.zeros((calls * self.per_call,), dtype=mx.int32)
        mx.eval(slot_of)
        channel = self.hs.Channel()
        server = self.hs.PoolServer(channel, self.box, slot_of, self.dsts, self.sizes, fds.tobytes(), offsets.tobytes(),
                                    calls, self.per_call, 1, 2 * self.per_call, threads, 60_000)
        if packed:
            server.set_pack(self.pack_fd, records.tobytes(), self.lay.stride, self.lay.part_offsets,
                            self.lay.alignment)
        value = 0
        start = time.perf_counter()
        for c in range(calls):
            n = min(self.per_call, len(picks) - c * self.per_call)
            self.hs.write_into(self.box, 0, struct.pack(f"<{n}I", *range(n)))
            server.submit(0, c, n, value + 1, value + 2)
            channel.signal(value + 1)
            if not channel.wait(value + 2, 60_000):
                raise TimeoutError("the pool server never answered")
            value += 2
        took = time.perf_counter() - start
        error = server.error()
        _, misses, read, preads = server.counters()
        server.close()
        if error is not None:
            raise RuntimeError(f"pool server: {error}")
        if misses != len(picks) or read != len(picks) * self.expert_bytes:
            raise RuntimeError(f"expected {len(picks)} misses, got {misses} ({read} bytes)")
        return took, preads


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--experts", type=int, default=512, help="experts read a run (default 512)")
    ap.add_argument("--per-call", type=int, default=16, help="misses a call (default 16)")
    ap.add_argument("--threads", default="4,8,16,32")
    ap.add_argument("--pack", type=Path, default=None, help="the expert pack (default MODEL_DIR/experts.pack)")
    ap.add_argument("--modes", default="python,native,packed")
    ap.add_argument("--json", type=Path, default=None, help="also write the results here")
    args = ap.parse_args(argv)
    from tensorfold.streaming.build import load

    model_dir = args.model_dir.expanduser().resolve()
    pack = (args.pack or model_dir / packs.PACK_NAME).expanduser()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if "packed" in modes and not packs.index_path(pack).is_file():
        print(f"no expert pack at {pack}: skipping packed runs (build it with tools/pack_experts.py)", file=sys.stderr)
        modes.remove("packed")
    bench = Bench(model_dir, load(), args.per_call, pack if "packed" in modes else None)
    print(f"{bench.expert_bytes / 2**20:.2f} MiB an expert, {args.experts} experts a run "
          f"({args.experts * bench.expert_bytes / 2**30:.2f} GiB), {args.per_call} a call")
    results, seed = [], 0
    for threads in [int(t) for t in args.threads.split(",")]:
        for mode in modes:
            seed += 1
            picks = bench.sample(args.experts, seed)
            if mode == "python":
                took, preads = bench.python(picks, threads), len(picks) * len(bench.order)
            else:
                took, preads = bench.native(picks, threads, mode == "packed")
            gbs = len(picks) * bench.expert_bytes / took / 1e9
            per_call = took / -(-len(picks) // args.per_call) * 1e3
            results.append({"mode": mode, "threads": threads, "seconds": round(took, 4), "GB_s": round(gbs, 3),
                            "ms_per_call": round(per_call, 3), "preads": int(preads)})
            print(f"{mode:>7} threads={threads:<3} {gbs:6.2f} GB/s  {per_call:7.2f} ms a call  {preads} preads",
                  flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=1))
    for fd in bench.fds.values():
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
