"""A GPU-resident pool of expert slots, filled from the checkpoint by a host thread that answers each layer's routing."""

from __future__ import annotations

import os
import queue
import struct
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.streaming import profile as _profile
from tensorfold.streaming.checkpoint import headers

PARTS = ("weight", "scales", "biases")
_DTYPES = {"U32": mx.uint32, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
TIMEOUT_MS = 60_000          # a layer's loads finish long before this; past it the forward fails
_KINDS = {"rows": 0, "window": 1}
_ERRORS = {"TimeoutError": TimeoutError, "ValueError": ValueError, "MemoryError": MemoryError,
           "IndexError": IndexError, "KeyError": KeyError, "RuntimeError": RuntimeError}


def pool_threads(env: dict | None = None) -> int:
    """Reader threads for misses: TENSORFOLD_POOL_THREADS (1..256), 8 by default."""

    env = os.environ if env is None else env
    return min(256, max(1, int(env.get("TENSORFOLD_POOL_THREADS", "8") or 8)))


def native_wanted(env: dict | None = None) -> bool:
    """TENSORFOLD_POOL_NATIVE=0 keeps the Python serve loop; otherwise the extension's runs when it can."""

    env = os.environ if env is None else env
    return (env.get("TENSORFOLD_POOL_NATIVE", "1") or "1").strip().lower() not in ("0", "off", "no", "false")


def aligned_reads(env: dict | None = None) -> bool:
    """On by default: each unaligned expert part over ~1 MiB is read as a page-aligned superset into an aligned
    buffer, then copied out: same bytes, but macOS F_NOCACHE then leaves none of them in the page cache (an unaligned
    read leaves all but its first MiB there). TENSORFOLD_POOL_ALIGNED_READS=0 restores the plain unaligned reads."""

    env = os.environ if env is None else env
    return (env.get("TENSORFOLD_POOL_ALIGNED_READS", "1") or "1").strip().lower() not in ("0", "off", "no", "false")


def window_reuse_wanted(env: dict | None = None) -> bool:
    """TENSORFOLD_POOL_WINDOW_REUSE=0 reads every window expert from the checkpoint (v0.2.0); otherwise a window
    expert already in the decode LRU is copied from its slot."""

    env = os.environ if env is None else env
    return (env.get("TENSORFOLD_POOL_WINDOW_REUSE", "1") or "1").strip().lower() not in ("0", "off", "no", "false")


@dataclass
class Source:
    """Where expert 0 of one tensor starts in a checkpoint file, and the bytes of one expert."""

    path: Path
    offset: int
    nbytes: int


@dataclass
class Slots:
    """One projection's pool: weight, scales and biases for every slot, in the resident stacks' layout."""

    weight: mx.array
    scales: mx.array
    biases: mx.array


def sources(model_dir: Path, names: dict[tuple, str]) -> tuple[dict, dict]:
    """Sources and each (proj, part)'s (shape, dtype) for keys (layer, proj, part) of stacks or (..., expert)."""

    shards = headers(model_dir)
    found, shapes = {}, {}
    for key, name in names.items():
        hits = [(p, base, h[name]) for p, (base, h) in shards.items() if name in h]
        if len(hits) != 1:
            raise ValueError(f"expert tensor {name}: expected in exactly one checkpoint file, found {len(hits)}")
        path, base, entry = hits[0]
        shape, (begin, end) = entry["shape"], entry["data_offsets"]
        stacked = len(key) == 3
        found[key] = Source(path, base + begin, (end - begin) // shape[0] if stacked else end - begin)
        spec = (tuple(shape[1:] if stacked else shape), entry["dtype"])
        if shapes.setdefault((key[1], key[2]), spec) != spec:
            raise ValueError(f"expert tensor {name}: shape or dtype differs between layers or experts")
    return found, shapes


def expert_nbytes(found: dict, layer: int) -> int:
    """Bytes of one expert of ``layer``: every projection's weight, scales and biases."""

    return sum(Streamer.source(found, layer, key[1], key[2], 0)[0].nbytes
               for key in found if key[0] == layer and (len(key) == 3 or key[3] == 0))


def _uncached(path: Path) -> int:
    """A read-only fd whose reads bypass the page cache (macOS F_NOCACHE) or skip readahead (Linux): the pool caches."""

    fd = os.open(path, os.O_RDONLY)
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)       # 48: F_NOCACHE in <sys/fcntl.h>
    elif hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    return fd


class Streamer:
    """Slots [0, E): one layer's window (expert e in slot e) for prompt chunks; [E, S): an LRU for decode rows.

    With an extension that has one, the extension's PoolServer serves the calls (no Python per call); otherwise the Python thread below does. Both keep the same LRU, slots and
    counters for the same calls. A window expert already in the LRU is copied from its slot instead of read
    (``window_reuse``; None follows TENSORFOLD_POOL_WINDOW_REUSE), which leaves the LRU as it was. ``pack`` (opt-in, native only): None follows TENSORFOLD_EXPERT_PACK, False is
    off, True is MODEL_DIR/experts.pack, a path is that pack; a pack in use must validate or the pool refuses it."""

    def __init__(self, found: dict, shapes: dict, *, layers: int, experts: int, top_k: int, slots: int,
                 box_rows: int, hostsync: Any, workers: int | None = None, native: bool | None = None,
                 pack: str | Path | bool | None = None, window_reuse: bool | None = None) -> None:
        if slots < 2 * experts:
            raise ValueError(f"the expert pool needs a layer's {experts} window slots plus as many for decode, "
                             f"got {slots}")
        self.hs, self.found, self.layers, self.experts, self.top_k = hostsync, found, layers, experts, top_k
        arrays = []
        self.pool: dict[str, Slots] = {}
        for proj in sorted({key[1] for key in found}):
            parts = [mx.zeros((slots, *shapes[(proj, part)][0]), dtype=_DTYPES[shapes[(proj, part)][1]])
                     for part in PARTS]
            self.pool[proj] = Slots(*parts)
            arrays += parts
        self.slot_of = mx.zeros((layers * experts,), dtype=mx.int32)
        # calls alternate between two regions of the box; the host checks the next one is clear before releasing a
        # call: the next call's routing written before this one was answered fails it ("calls ran out of order").
        # A call two ahead shares this region and is not seen here; by analysis it ends in a pool timeout. Either
        # way the request fails; the engine's order chain keeps both from happening
        self.box_words = max(box_rows * top_k, experts, 8)
        self.boxes = mx.zeros((2 * self.box_words,), dtype=mx.uint32)
        self._clear = bytes(4 * self.box_words)                         # what a read region is reset to
        self.layer_ids = [mx.array([i] + [0] * 7, dtype=mx.int32) for i in range(layers)]
        mx.eval(*arrays, self.slot_of, self.boxes, *self.layer_ids)
        self.fds = {path: _uncached(path) for path in {s.path for s in found.values()}}
        self.channel = hostsync.Channel()
        self.value = 0
        self.window: list[int] = [-1] * experts                         # the layer whose expert sits in slot e
        self.lru: OrderedDict[tuple[int, int], int] = OrderedDict()     # (layer, expert) -> slot, oldest first
        self.free = list(range(slots - 1, experts - 1, -1))
        self._server = None
        self._close_lock = threading.Lock()
        self.pack_path: Path | None = None                              # the expert pack the native server reads
        self._error: BaseException | None = None
        self.hits = self.misses = self.bytes_read = 0
        self.aligned = aligned_reads()
        self.window_reuse = window_reuse_wanted() if window_reuse is None else bool(window_reuse)
        self.window_reads = self.window_copies = self.copy_bytes = 0
        workers = pool_threads() if workers is None else int(workers)
        self._bytes_lock = threading.Lock()                             # reader threads add to bytes_read
        self._readers = ThreadPoolExecutor(workers, thread_name_prefix="expert-read")
        self._jobs: queue.Queue = queue.Queue()
        self._prof: list | None = None                                  # the Python serve loop's call records
        self._prof_cur: dict | None = None
        self._prof_mu = threading.Lock()
        self._prof_dropped = 0
        if native is None:
            native = native_wanted()
        if native and hasattr(hostsync, "PoolServer"):
            self._readers.shutdown(wait=False)
            try:
                self._server = self._native(slots, workers, pack)
                self._server.set_regions(self.box_words)
            except BaseException:
                for fd in self.fds.values():
                    os.close(fd)
                raise
            self._thread = None
            if _profile.ACTIVE is not None:
                _profile.ACTIVE.attach(self)
            return
        self._thread = threading.Thread(target=self._serve, name="expert-stream", daemon=True)
        self._thread.start()
        if _profile.ACTIVE is not None:
            _profile.ACTIVE.attach(self)

    # -- decode profiler (streaming/profile.py) ---------------------------------------------------------------------
    _PROF_KEYS = ("kind", "layer", "rows", "unique", "misses", "bytes", "signal", "t_submit", "t_sig", "t_read0",
                  "t_read1", "t_ready")
    PROF_CAP = 1 << 16                                                  # records kept between drains (as native)

    def profile_on(self) -> None:
        if self._server is not None:
            self._server.profile(True)
        else:
            with self._prof_mu:
                self._prof, self._prof_dropped = [], 0

    def profile_off(self) -> None:
        if self._server is not None:
            self._server.profile(False)
        else:
            with self._prof_mu:
                self._prof = None

    def profile_drain(self) -> tuple[list[dict], int]:
        """Each call since the last drain: kind (0 rows, 1 window), layer, rows, unique experts, misses (expert
        reads), bytes read, the signal value and the host stamps (ns: submitted, signalled, reads start and end,
        released)."""

        if self._server is not None:
            got, dropped = self._server.drain_profile()
            return [dict(zip(self._PROF_KEYS, (int(v) for v in rec))) for rec in got], int(dropped)
        with self._prof_mu:
            got, self._prof = (self._prof or []), ([] if self._prof is not None else None)
            dropped, self._prof_dropped = self._prof_dropped, 0
        return got, dropped

    def _native(self, slots: int, workers: int, pack: str | Path | bool | None) -> Any:
        """The extension's PoolServer over this pool: every (layer, expert, part)'s fd and offset, maybe a pack."""

        from tensorfold.streaming import pack as packs

        order = [(proj, part) for proj in self.pool for part in PARTS]
        fds = np.full((self.layers, self.experts, len(order)), -1, dtype=np.int32)
        offsets = np.zeros((self.layers, self.experts, len(order)), dtype=np.int64)
        sizes = [0] * len(order)
        for layer in sorted({key[0] for key in self.found}):
            for p, (proj, part) in enumerate(order):
                for e in range(self.experts):
                    src, offset = self.source(self.found, layer, proj, part, e)
                    fds[layer, e, p], offsets[layer, e, p], sizes[p] = self.fds[src.path], offset, src.nbytes
        server = self.hs.PoolServer(self.channel, self.boxes, self.slot_of,
                                    [getattr(self.pool[proj], part) for proj, part in order], sizes,
                                    fds.tobytes(), offsets.tobytes(), self.layers, self.experts, self.top_k, slots,
                                    workers, TIMEOUT_MS, aligned=self.aligned, window_reuse=self.window_reuse)
        path = packs.choose(self._model_dir(), pack=pack)
        if path is None:
            return server
        lay = packs.validate(path, self.found, self.experts)
        if lay.order != order or lay.part_nbytes != sizes:
            raise packs.PackError(f"expert pack {path}: its part order or sizes differ from the pool's")
        records = np.full((self.layers, self.experts), -1, dtype=np.int64)
        for li, layer in enumerate(lay.layers):
            records[layer] = (li * self.experts + np.arange(self.experts, dtype=np.int64)) * lay.stride
        fd = _uncached(path)
        self.fds[path] = fd
        try:
            server.set_pack(fd, records.tobytes(), lay.stride, lay.part_offsets, lay.alignment)
        except ValueError as error:
            raise packs.PackError(f"expert pack {path}: {error}") from None
        self.pack_path = path
        return server

    def _model_dir(self) -> Path:
        return Path(os.path.commonpath([str(Path(s.path).parent) for s in self.found.values()]))

    @property
    def native(self) -> bool:
        return self._server is not None

    @property
    def error(self) -> BaseException | None:
        if self._error is None and self._server is not None:
            found = self._server.error()
            if found is not None:
                kind, message = found
                self._error = _ERRORS.get(kind, RuntimeError)(message)
        return self._error

    @error.setter
    def error(self, value: BaseException | None) -> None:
        self._error = value

    def _counter(index: int, name: str, method: str = "counters") -> property:   # a property factory (class body)
        def get(self) -> int:
            if self._server is not None:
                return int(getattr(self._server, method)()[index])
            return self.__dict__[name]

        def put(self, value: int) -> None:
            self.__dict__[name] = value

        return property(get, put)

    hits = _counter(0, "_hits")
    misses = _counter(1, "_misses")
    bytes_read = _counter(2, "_bytes_read")
    window_reads = _counter(0, "_window_reads", "window_counters")
    window_copies = _counter(1, "_window_copies", "window_counters")
    copy_bytes = _counter(2, "_copy_bytes", "window_counters")
    del _counter

    def lru_items(self) -> list[tuple[tuple[int, int], int]]:
        """The decode LRU as ((layer, expert), slot), oldest first."""

        if self._server is not None:
            return [((layer, e), slot) for layer, e, slot in self._server.lru_items()]
        return list(self.lru.items())

    def free_slots(self) -> list[int]:
        return list(self._server.free_slots()) if self._server is not None else list(self.free)

    def window_layers(self) -> list[int]:
        return list(self._server.window()) if self._server is not None else list(self.window)

    @property
    def box(self) -> mx.array:
        """The routing box region the next call's kernel writes (calls alternate between two)."""

        at = self.region_word(self.value + 1)
        return self.boxes[at:at + self.box_words]

    def region_word(self, signal: int) -> int:
        """Where in the box the call of ``signal`` finds its routing (a word offset)."""

        return (signal - 1) // 2 % 2 * self.box_words

    def marks(self, kind: str, layer: int, rows: int) -> tuple[int, int]:
        """(signal, ready) for one MoE call: "rows" reads rows x top_k ids from the box, "window" E presence flags."""

        if self.error is not None:
            raise RuntimeError("expert streaming stopped") from self.error
        signal, ready = self.value + 1, self.value + 2
        self.value = ready
        if self._server is not None:
            self._server.submit(_KINDS[kind], layer, rows, signal, ready)
            return signal, ready
        self._jobs.put((kind, layer, rows, signal, ready, _profile.now() if self._prof is not None else 0))
        return signal, ready

    def hold(self, token: mx.array, kind: str, layer: int, rows: int) -> mx.array:
        """``token`` once the GPU has signalled this call to the host and the host has answered it (graph marks)."""

        signal, ready = self.marks(kind, layer, rows)
        return self.hs.gpu_wait(self.hs.gpu_signal(token, self.channel, signal), self.channel, ready)

    def window_views(self) -> dict[str, Slots]:
        """Each projection's window slots [0, E) as stacks of the model's shape: expert e of the loaded layer."""

        return {proj: Slots(*(getattr(s, p)[:self.experts] for p in PARTS)) for proj, s in self.pool.items()}

    def stats(self) -> dict[str, int | str]:
        """Hits, misses and bytes read; window experts read, and copied from the LRU (with the bytes copied); the
        native server's read counts; and "error" once a call failed (the pool then refuses every later call)."""

        if self._server is not None:
            hits, misses, read, preads = self._server.counters()
            reads, copies, copied = self._server.window_counters()
            out = {"hits": int(hits), "misses": int(misses), "bytes_read": int(read), "window_reads": int(reads),
                   "window_copies": int(copies), "copy_bytes": int(copied), "native": 1,
                   "packed": int(self._server.packed()), "preads": int(preads), "threads": int(self._server.workers())}
            return self._with_error(out)
        out = {"hits": int(self.hits), "misses": int(self.misses), "bytes_read": int(self.bytes_read),
               "window_reads": int(self.window_reads), "window_copies": int(self.window_copies),
               "copy_bytes": int(self.copy_bytes)}
        return self._with_error(out)

    def _with_error(self, out: dict) -> dict:
        error = self.error
        if error is not None:
            out["error"] = f"{type(error).__name__}: {error}"
        return out

    def _serve(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            kind, layer, rows, signal, ready, submitted = job
            rec = None
            if submitted and self._prof is not None:
                rec = dict.fromkeys(self._PROF_KEYS, 0)
                rec.update(kind=_KINDS[kind], layer=layer, rows=rows, signal=signal, t_submit=submitted)
            self._prof_cur = rec
            try:
                if not self.channel.wait(signal, TIMEOUT_MS):
                    raise TimeoutError(f"no routing from the GPU for layer {layer}")
                if rec is not None:
                    rec["t_sig"] = _profile.now()
                if kind == "rows":
                    if rows * self.top_k > self.box_words:
                        raise IndexError("peek: the array is not evaluated or the range passes its end")
                    at = 4 * self.region_word(signal)
                    ids = np.frombuffer(self.hs.peek(self.boxes, at, 4 * rows * self.top_k), dtype=np.uint32)
                    routed = sorted({int(e) for e in ids})
                    if rec is not None:
                        rec["unique"] = len(routed)
                    self._check(layer, routed)
                    self._load(layer, routed)
                else:
                    self._check(layer, [])
                    at = 4 * self.region_word(signal)
                    flags = np.frombuffer(self.hs.peek(self.boxes, at, 4 * self.experts), dtype=np.uint32)
                    if rec is not None:
                        rec["unique"] = int(np.count_nonzero(flags))
                    self._fill_window(layer, np.flatnonzero(flags).tolist())
            except BaseException as error:  # noqa: BLE001  # the GPU still gets its release; the next call raises
                self._record(error)
            finally:
                try:                                   # a failed call's region is cleared too: the one two later
                    self._check_order(signal, layer)   # checks it is
                except BaseException as error:  # noqa: BLE001
                    self._record(error)
            if rec is not None:
                rec["t_ready"] = _profile.now()
                self._prof_cur = None
                with self._prof_mu:                 # profile_off/drain may run on another thread
                    if self._prof is None:
                        pass
                    elif len(self._prof) < self.PROF_CAP:
                        self._prof.append(rec)
                    else:
                        self._prof_dropped += 1
            self.channel.signal(ready)

    def _record(self, error: BaseException) -> None:
        """Keep the first error (as the native server does): later failures are its consequences."""

        if self.error is None:
            self.error = error

    def _check_order(self, signal: int, layer: int) -> None:
        """This call's region cleared; the next call's must still be clear (the native server checks the same)."""

        size = 4 * self.box_words
        self.hs.write_into(self.boxes, 4 * self.region_word(signal), self._clear)
        if np.frombuffer(self.hs.peek(self.boxes, 4 * self.region_word(signal + 2), size), dtype=np.uint8).any():
            raise RuntimeError(f"the expert pool's call after layer {layer}'s wrote its routing before that call was "
                               "answered: calls ran out of order")

    def _check(self, layer: int, routed: list[int]) -> None:
        """A call's layer and (sorted) routed ids inside the model, before anything changes (the native check too)."""

        if not 0 <= layer < self.layers:
            raise ValueError(f"layer {layer} is outside the model")
        bad = [e for e in routed if not 0 <= e < self.experts]
        if bad:
            raise ValueError(f"routed expert {bad[0]} is outside the model")

    def _load(self, layer: int, ids: list[int]) -> None:
        """Every id resident in the LRU for ``layer``, then its slot written into the table."""

        need, table = [], []
        for e in ids:
            if not 0 <= e < self.experts:
                raise ValueError(f"routed expert {e} is outside the model")
            slot = self.lru.pop((layer, e), None)
            if slot is None:
                slot = self.free.pop() if self.free else self._evict({(layer, i) for i in ids})
                need.append((e, slot))
                self.misses += 1
            else:
                self.hits += 1
            self.lru[(layer, e)] = slot
            table.append((e, slot))
        self._read_all(layer, need)
        for e, slot in table:
            self.hs.write_into(self.slot_of, 4 * (layer * self.experts + e), struct.pack("<i", slot))

    def _fill_window(self, layer: int, ids: list[int]) -> None:
        reuse = self.window_reuse and self._error is None       # after a failure an LRU slot may be partly read
        need, copy = [], []
        for e in ids:
            if self.window[e] == layer:
                continue
            src = self.lru.get((layer, e)) if reuse else None
            if src is None:
                need.append((e, e))
            else:
                copy.append((e, src))
        self._read_all(layer, need)
        self._copy_all(copy)
        self.window_reads += len(need)
        self.window_copies += len(copy)
        self.copy_bytes += len(copy) * sum(a.nbytes // a.shape[0] for s in self.pool.values()
                                           for a in (s.weight, s.scales, s.biases))
        for e, _ in need + copy:
            self.window[e] = layer

    def _copy_all(self, copy: list[tuple[int, int]]) -> None:
        """(dst, src) slot pairs, every part: src in the LRU, dst in the window."""

        arrays = [getattr(s, part) for s in self.pool.values() for part in PARTS]

        def one(item):
            (dst, src), a = item
            n = a.nbytes // a.shape[0]
            if hasattr(self.hs, "copy_within"):
                self.hs.copy_within(a, dst * n, src * n, n)
            else:
                self.hs.write_into(a, dst * n, self.hs.peek(a, src * n, n))

        list(self._readers.map(one, [(pair, a) for pair in copy for a in arrays]))

    def _evict(self, keep: set) -> int:
        for key in self.lru:
            if key not in keep:
                return self.lru.pop(key)
        raise MemoryError("the expert pool is smaller than one call's experts")

    def _read_all(self, layer: int, need: list[tuple[int, int]]) -> None:
        rec = self._prof_cur
        if rec is not None:
            rec["misses"] = len(need)
            rec["bytes"] = len(need) * expert_nbytes(self.found, layer)
            if need:
                rec["t_read0"] = _profile.now()
        reads = [(proj, part, e, slot) for e, slot in need for proj in self.pool for part in PARTS]
        list(self._readers.map(lambda item: self._read(layer, *item), reads))
        if rec is not None and need:
            rec["t_read1"] = _profile.now()

    @staticmethod
    def source(found: dict, layer: int, proj: str, part: str, expert: int) -> tuple[Source, int]:
        """The tensor holding ``expert`` and the file offset of its bytes."""

        one = found.get((layer, proj, part, expert))
        if one is not None:
            return one, one.offset
        stack = found[(layer, proj, part)]
        return stack, stack.offset + expert * stack.nbytes

    def _read(self, layer: int, proj: str, part: str, expert: int, slot: int) -> None:
        src, offset = self.source(self.found, layer, proj, part, expert)
        args = (getattr(self.pool[proj], part), slot * src.nbytes, self.fds[src.path], offset, src.nbytes)
        if self.aligned:
            self.hs.pread_into(*args, aligned=True)
        else:
            self.hs.pread_into(*args)
        with self._bytes_lock:
            self.bytes_read += src.nbytes

    def close(self) -> None:
        """Stop serving and close the files; idempotent, and safe from several threads at once."""

        with self._close_lock:
            if self._server is not None:
                self._server.close()
            else:
                self._close_python()
            while self.fds:
                os.close(self.fds.popitem()[1])

    def _close_python(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5)
        self._readers.shutdown(wait=True)


__all__ = ["PARTS", "Slots", "Source", "Streamer", "aligned_reads", "expert_nbytes", "native_wanted",
           "pool_threads", "sources", "window_reuse_wanted"]
