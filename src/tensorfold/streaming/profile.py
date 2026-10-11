"""Decode profiler (opt-in): host stamps at the points a decode round already passes, the expert pool's per-call
records and the GPU start/end of the command buffers that carry its marks, written as JSONL per request.

TENSORFOLD_DECODE_PROFILE=<path.jsonl> turns it on (Mode A: no MLX op, eval or sync is added, so tokens cannot
change). TENSORFOLD_DECODE_PROFILE_SYNC=1 adds Mode B: eval barriers inside the forward and the round for exact
per-component GPU time (a scheduling change only; accept its numbers only when its token shas match).
Off (unset), ``ACTIVE`` is None and every hook is one attribute test."""

from __future__ import annotations

import json
import os
import threading
import time
import weakref
from typing import Any

ACTIVE: Recorder | None = None
CLOCK_TOLERANCE_NS = 100_000          # steady_clock, mach_absolute_time and Metal GPU times share one domain on macOS

now = time.monotonic_ns


class Recorder:
    """Rounds and prompt chunks of the current request, flushed with the pool's and the GPU's records."""

    def __init__(self, path: str | os.PathLike, *, sync: bool = False) -> None:
        self.path, self.sync = str(path), bool(sync)
        self.lock = threading.Lock()
        self.rounds: list[dict] = []
        self.chunks: list[dict] = []
        self.events: list[dict] = []
        self.runs_outside: list[dict] = []      # forwards with no round open (prompt chunks, load-time checks)
        self.cur: dict | None = None
        self.run: dict | None = None
        self.streamers: weakref.WeakSet = weakref.WeakSet()
        self.extensions: dict[int, Any] = {}
        self.clock_offset_ns: int | None = None
        self.requests = 0

    # -- wiring -----------------------------------------------------------------------------------------------
    def attach(self, streamer: Any) -> None:
        """Start a pool's call records (and, once per extension, the GPU's buffer times)."""

        self.streamers.add(streamer)
        streamer.profile_on()
        hs = streamer.hs
        if hasattr(hs, "gpu_profile") and id(hs) not in self.extensions:
            self.extensions[id(hs)] = hs
            self.clock_offset_ns = check_clock(hs)
            hs.gpu_profile(True)

    # -- rounds -------------------------------------------------------------------------------------------------
    def begin_round(self, **meta: Any) -> dict:
        rec = {"t0": now(), "runs": [], **meta}
        self.cur = rec
        return rec

    def mark(self, name: str) -> None:
        if self.cur is not None:
            self.cur[name] = now()

    def end_round(self, rec: dict, **meta: Any) -> None:
        rec.update(meta)
        rec["t_end"] = now()
        self.cur = None
        with self.lock:
            self.rounds.append(rec)

    # -- forwards (FusedDecode.run) -----------------------------------------------------------------------------
    def begin_run(self, layers: int, rows: int) -> dict:
        run = {"layers": layers, "rows": rows, "t_start": now(), "q": [], "b": []}
        self.run = run
        (self.cur["runs"] if self.cur is not None else self.runs_outside).append(run)
        return run

    def end_run(self, run: dict) -> None:
        run["t_end"] = now()
        self.run = None

    def ple(self, started: int) -> None:
        if self.run is not None:
            self.run["ple_ns"] = self.run.get("ple_ns", 0) + now() - started

    # -- prompt chunks and request events -------------------------------------------------------------------------
    def chunk(self, started: int, ended: int, **meta: Any) -> None:
        with self.lock:
            self.chunks.append({"t0": started, "t_end": ended, **meta})

    def event(self, name: str, started: int, ended: int) -> None:
        with self.lock:
            self.events.append({"name": name, "t0": started, "t_end": ended})

    # -- output -------------------------------------------------------------------------------------------------
    def drain(self) -> tuple[list[dict], list[tuple], dict]:
        """Every pool's call records (as dicts) and the GPU's buffer records since the last drain."""

        calls, dropped = [], 0
        for streamer in list(self.streamers):
            got, lost = streamer.profile_drain()
            calls += got
            dropped += lost
        gpu: list[tuple] = []
        gpu_dropped = 0
        for hs in self.extensions.values():
            got, lost = hs.gpu_drain()
            gpu += [tuple(int(v) for v in g) for g in got]
            gpu_dropped += int(lost)
        return calls, gpu, {"calls_dropped": dropped, "gpu_dropped": gpu_dropped}

    def flush(self, **meta: Any) -> None:
        """One request done: a header line, then one line per prompt chunk and per round, with the calls and GPU
        buffer times inside each one's span attached."""

        with self.lock:
            rounds, self.rounds = self.rounds, []
            chunks, self.chunks = self.chunks, []
            events, self.events = self.events, []
            outside, self.runs_outside = self.runs_outside, []
        calls, gpu, lost = self.drain()
        by_signal: dict[int, dict] = {}
        for value, wait, buffer, start, end in gpu:
            by_signal[value] = {"buffer": buffer, "start": start, "end": end, "wait": wait}
        for call in calls:
            sig, ready = by_signal.get(call["signal"]), by_signal.get(call["signal"] + 1)
            if sig is not None:
                call["sig_buffer"], call["sig_start"], call["sig_end"] = sig["buffer"], sig["start"], sig["end"]
            if ready is not None:
                call["wait_buffer"], call["wait_start"], call["wait_end"] = ready["buffer"], ready["start"], ready["end"]
        spans = [("round", r) for r in rounds] + [("chunk", c) for c in chunks]
        spans.sort(key=lambda s: s[1]["t0"])
        for _, span in spans:
            span["calls"] = []
        orphans = []
        starts = [s[1]["t0"] for s in spans]
        import bisect

        for call in sorted(calls, key=lambda c: c["t_submit"]):
            i = bisect.bisect_right(starts, call["t_submit"]) - 1
            if i >= 0 and call["t_submit"] <= spans[i][1]["t_end"]:
                spans[i][1]["calls"].append(call)
            else:
                orphans.append(call)
        self.requests += 1
        header = {"type": "request", "n": self.requests, "sync": self.sync, "clock_offset_ns": self.clock_offset_ns,
                  "rounds": len(rounds), "chunks": len(chunks), "events": events, "orphan_calls": orphans,
                  "outside_runs": len(outside), **lost, **meta}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(header) + "\n")
            for kind, span in spans:
                handle.write(json.dumps({"type": kind, **span}) + "\n")


def check_clock(hs: Any) -> int:
    """The extension's steady_clock against Python's monotonic clock: the smallest bracketed offset (ns)."""

    best = None
    for _ in range(9):
        a = now()
        c = int(hs.clock_ns())
        b = now()
        off = c - (a + b) // 2
        if best is None or b - a < best[0]:
            best = (b - a, off)
    offset = best[1]
    assert abs(offset) < CLOCK_TOLERANCE_NS, f"decode profile: C++ steady_clock is {offset} ns off Python's clock"
    return offset


def configure(path: str | os.PathLike | None, *, sync: bool = False) -> Recorder | None:
    """Turn the profiler on (a path) or off (None). Pools built later attach themselves; existing ones are not
    reached from here, so configure before loading the model."""

    global ACTIVE
    if path is None:
        if ACTIVE is not None:
            for streamer in list(ACTIVE.streamers):
                streamer.profile_off()
            for hs in ACTIVE.extensions.values():
                hs.gpu_profile(False)
        ACTIVE = None
        return None
    ACTIVE = Recorder(path, sync=sync)
    return ACTIVE


def from_env(env: dict | None = None) -> Recorder | None:
    env = os.environ if env is None else env
    path = (env.get("TENSORFOLD_DECODE_PROFILE") or "").strip()
    if not path:
        return None
    sync = (env.get("TENSORFOLD_DECODE_PROFILE_SYNC", "0") or "0").strip() not in ("0", "", "off", "no", "false")
    return configure(path, sync=sync)


from_env()

__all__ = ["ACTIVE", "Recorder", "check_clock", "configure", "from_env"]
