"""A GPU-resident pool of expert slots, filled from the checkpoint by a host thread that answers each layer's routing."""

from __future__ import annotations

import os
import queue
import struct
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_all
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

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


def job_log_path(env: dict | None = None) -> str:
    """TF_POOL_JOBLOG=path appends one line per pool call (diagnostics): kind layer rows seen_ns ready_ns bytes preads."""

    env = os.environ if env is None else env
    return (env.get("TF_POOL_JOBLOG", "") or "").strip()


def _now_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW) if hasattr(time, "CLOCK_UPTIME_RAW") else time.monotonic_ns()


def native_wanted(env: dict | None = None) -> bool:
    """TENSORFOLD_POOL_NATIVE=0 keeps the Python serve loop; otherwise the extension's runs when it can."""

    env = os.environ if env is None else env
    return (env.get("TENSORFOLD_POOL_NATIVE", "1") or "1").strip().lower() not in ("0", "off", "no", "false")


@dataclass
class Source:
    """Where expert 0 of one tensor starts in a checkpoint file, and the bytes of one expert."""

    path: Path
    offset: int
    nbytes: int


@dataclass
class Pending:
    """A speculative read in flight: the slot it fills and its part reads (a decode guess, or a prompt seed)."""

    slot: int
    reads: list[Future]
    seed: bool


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

    With decode prefetch and prompt seeding off and an extension that has one, the extension's PoolServer serves
    the calls (no Python per call); otherwise the Python thread below does. Both keep the same LRU, slots and
    counters for the same calls. ``pack`` (opt-in, native only): None follows TENSORFOLD_EXPERT_PACK, False is
    off, True is MODEL_DIR/experts.pack, a path is that pack; a pack in use must validate or the pool refuses it."""

    def __init__(self, found: dict, shapes: dict, *, layers: int, experts: int, top_k: int, slots: int,
                 box_rows: int, hostsync: Any, workers: int | None = None, decode_prefetch: int = 0,
                 prompt_seed: int = 0, native: bool | None = None, pack: str | Path | bool | None = None) -> None:
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
        workers = pool_threads() if workers is None else int(workers)
        self._bytes_lock = threading.Lock()                             # reader threads add to bytes_read
        self._readers = ThreadPoolExecutor(workers, thread_name_prefix="expert-read")
        # decode_prefetch=k: each decode row's k guesses for the next layer (GPU-written) are read behind the call;
        # prompt_seed=n: after a prompt, each layer's n most-routed experts are read into the LRU, warmest last
        self.decode_prefetch, self.prompt_seed = int(decode_prefetch), int(prompt_seed)
        self.pending: dict[tuple[int, int], Pending] = {}
        self.speculative: dict[tuple[int, int], str] = {}               # loaded ahead of demand, not yet routed
        self.prefetch_issued = self.prefetch_hits = self.prefetch_misses = self.prefetch_cancelled = 0
        self.seeded = self.seed_hits = 0
        self.guess_box = None
        if self.decode_prefetch:
            self.guess_box = mx.zeros((max(box_rows * self.decode_prefetch, 8),), dtype=mx.uint32)
            mx.eval(self.guess_box)
        self.prompt_counts = np.zeros((layers, experts), dtype=np.int64) if self.prompt_seed else None
        self._prompted = False                                          # window calls since the last seed
        self._seeders = ThreadPoolExecutor(2, thread_name_prefix="expert-seed") if self.prompt_seed else None
        self._jobs: queue.Queue = queue.Queue()
        self._log_path = job_log_path()
        self._log: list[tuple] = []
        self._notes: list[str] = []
        self._log_lock = threading.Lock()
        if native is None:
            native = native_wanted()
        if native and not self.decode_prefetch and not self.prompt_seed and hasattr(hostsync, "PoolServer"):
            self._readers.shutdown(wait=False)
            try:
                self._server = self._native(slots, workers, pack)
                self._server.set_regions(self.box_words)
            except BaseException:
                for fd in self.fds.values():
                    os.close(fd)
                raise
            self._thread = None
            self._start_log()
            return
        self._thread = threading.Thread(target=self._serve, name="expert-stream", daemon=True)
        self._thread.start()
        self._start_log()

    def _start_log(self) -> None:
        if not self._log_path:
            return
        if self._server is not None:
            self._server.set_log(True)
        threading.Thread(target=self._write_log, name="expert-joblog", daemon=True).start()

    def take_log(self) -> list[tuple]:
        """Pool calls logged since the last take: (kind, layer, rows, seen_ns, ready_ns, bytes, preads)."""

        if self._server is not None:
            return [tuple(int(v) for v in row) for row in self._server.take_log()]
        with self._log_lock:
            out, self._log = self._log, []
        return out

    def note(self, text: str) -> None:
        """A marker line in the call log (when TF_POOL_JOBLOG is set), at the uptime clock's now."""

        if self._log_path:
            with self._log_lock:
                self._notes.append(f"# {_now_ns()} {text}")

    def _write_log(self) -> None:
        while True:
            time.sleep(1.0)
            rows = self.take_log()
            with self._log_lock:
                notes, self._notes = self._notes, []
            lines = [" ".join(str(v) for v in row) for row in rows]
            # notes and calls interleaved by time; a note is "# ns text", a call "kind layer rows seen ready ..."
            merged = sorted([(int(n.split()[1]), n) for n in notes] + [(int(r[3]), line) for r, line in zip(rows, lines)])
            if merged:
                try:
                    with open(self._log_path, "a") as f:
                        f.write("".join(line + "\n" for _, line in merged))
                except OSError:
                    pass

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
                                    workers, TIMEOUT_MS)
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

    def _counter(index: int, name: str) -> property:   # a property factory, run in the class body
        def get(self) -> int:
            return int(self._server.counters()[index]) if self._server is not None else self.__dict__[name]

        def put(self, value: int) -> None:
            self.__dict__[name] = value

        return property(get, put)

    hits = _counter(0, "_hits")
    misses = _counter(1, "_misses")
    bytes_read = _counter(2, "_bytes_read")
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

    def marks(self, kind: str, layer: int, rows: int, guess: int = 0) -> tuple[int, int]:
        """(signal, ready) for one MoE call: "rows" reads rows x top_k ids from the box, "window" E presence flags
        (or counts); ``guess`` > 0: rows x guess ids for layer + 1 wait in the guess box."""

        if self.error is not None:
            raise RuntimeError("expert streaming stopped") from self.error
        signal, ready = self.value + 1, self.value + 2
        self.value = ready
        if self._server is not None:
            self._server.submit(_KINDS[kind], layer, rows, signal, ready)
            return signal, ready
        self._jobs.put((kind, layer, rows, signal, ready, guess))
        return signal, ready

    def hold(self, token: mx.array, kind: str, layer: int, rows: int, guess: int = 0) -> mx.array:
        """``token`` once the GPU has signalled this call to the host and the host has answered it (graph marks)."""

        signal, ready = self.marks(kind, layer, rows, guess)
        return self.hs.gpu_wait(self.hs.gpu_signal(token, self.channel, signal), self.channel, ready)

    def window_views(self) -> dict[str, Slots]:
        """Each projection's window slots [0, E) as stacks of the model's shape: expert e of the loaded layer."""

        return {proj: Slots(*(getattr(s, p)[:self.experts] for p in PARTS)) for proj, s in self.pool.items()}

    def stats(self) -> dict[str, int | str]:
        """Hits, misses and bytes read; with decode prefetch or prompt seeding on, their counters too; the native
        server's read counts; and "error" once a call failed (the pool then refuses every later call)."""

        if self._server is not None:
            hits, misses, read, preads = self._server.counters()
            out = {"hits": int(hits), "misses": int(misses), "bytes_read": int(read), "native": 1,
                   "packed": int(self._server.packed()), "preads": int(preads), "threads": int(self._server.workers())}
            return self._with_error(out)
        out = {"hits": int(self.hits), "misses": int(self.misses), "bytes_read": int(self.bytes_read)}
        if self.decode_prefetch:
            out.update(prefetch_issued=self.prefetch_issued, prefetch_hits=self.prefetch_hits,
                       prefetch_misses=self.prefetch_misses, prefetch_cancelled=self.prefetch_cancelled)
        if self.prompt_seed:
            out.update(seeded=self.seeded, seed_hits=self.seed_hits)
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
            kind, layer, rows, signal, ready, guess = job
            after: list = []
            seen = 0
            try:
                if not self.channel.wait(signal, TIMEOUT_MS):
                    raise TimeoutError(f"no routing from the GPU for layer {layer}")
                if self._log_path:
                    seen, bytes0 = _now_ns(), self.bytes_read
                if kind == "rows":
                    if rows * self.top_k > self.box_words:
                        raise IndexError("peek: the array is not evaluated or the range passes its end")
                    at = 4 * self.region_word(signal)
                    ids = np.frombuffer(self.hs.peek(self.boxes, at, 4 * rows * self.top_k), dtype=np.uint32)
                    routed = sorted({int(e) for e in ids})
                    self._check(layer, routed)
                    routed_reads = self._resolve(layer, routed)
                    self._load(layer, routed)
                    for read in routed_reads:              # a guess this call routed to: its bytes before release
                        read.result()
                    keep = {(layer, e) for e in routed}
                    if guess and layer + 1 < self.layers:  # read before release: the next layer rewrites the box
                        guesses = np.frombuffer(self.hs.peek(self.guess_box, 0, 4 * rows * guess), dtype=np.uint32)
                        after.append((self._prefetch, layer + 1, list(dict.fromkeys(int(e) for e in guesses)), keep))
                    if self._prompted:
                        after.append((self._seed, self.prompt_counts.copy(), keep))
                        self.prompt_counts[:] = 0
                        self._prompted = False
                else:
                    self._check(layer, [])
                    self._resolve(layer, [])
                    at = 4 * self.region_word(signal)
                    flags = np.frombuffer(self.hs.peek(self.boxes, at, 4 * self.experts), dtype=np.uint32)
                    if self.prompt_counts is not None:
                        self.prompt_counts[layer] += flags
                        self._prompted = True
                    self._fill_window(layer, np.flatnonzero(flags).tolist())
            except BaseException as error:  # noqa: BLE001  # the GPU still gets its release; the next call raises
                self._record(error)
            finally:
                try:                                   # a failed call's region is cleared too: the one two later
                    self._check_order(signal, layer)   # checks it is
                except BaseException as error:  # noqa: BLE001
                    self._record(error)
            if seen:
                with self._log_lock:
                    self._log.append((_KINDS[kind], layer, rows, seen, _now_ns(), self.bytes_read - bytes0, 0))
            self.channel.signal(ready)
            for step, *args in after:             # speculative reads start once the GPU runs on
                if self.error is not None:
                    break
                try:
                    step(*args)
                except BaseException as error:  # noqa: BLE001  # recorded; the next call raises it
                    self._record(error)

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

    def _resolve(self, layer: int, routed: list[int]) -> list[Future]:
        """Before a call: the reads of speculative loads it routed to (to wait for), stale guesses cancelled.

        A guess the call did not route to is cancelled if its reads have not all started; once they end, a
        partly read slot goes back to the free list (reused first), a fully read one stays in the LRU. Seeds
        still in flight keep reading, except a routed seed still queued behind other seeds: its queued reads are
        cancelled and the call reads it on demand instead."""

        if not self.pending:
            return []
        wanted = {(layer, e) for e in routed}
        waits: list[Future] = []
        stale = []
        for key, entry in list(self.pending.items()):
            if key in wanted and entry.seed and any([read.cancel() for read in entry.reads]):  # noqa: C419  # cancel all
                stale.append((key, entry))        # dropped below, then a demand miss on the readers
            elif key in wanted:
                waits += entry.reads
                del self.pending[key]
            elif not entry.seed or all(read.done() for read in entry.reads):
                stale.append((key, entry))
        for _, entry in stale:                    # every stale read cancelled before waiting on any
            if not entry.seed:
                for read in entry.reads:
                    read.cancel()
        for key, entry in stale:              # a cancelled read never runs; wait() would wait for a worker to see it
            wait_all([read for read in entry.reads if not read.cancelled()])
            del self.pending[key]
            if any(read.cancelled() or read.exception() is not None for read in entry.reads):
                if self.lru.get(key) == entry.slot:
                    del self.lru[key]
                self.speculative.pop(key, None)
                self.free.append(entry.slot)
                if not entry.seed:
                    self.prefetch_cancelled += 1
                    self.prefetch_misses += 1
        return waits

    def _speculate(self, layer: int, e: int, slot: int, seed: bool) -> None:
        """Start reading expert ``e`` of ``layer`` into ``slot``, already the newest entry of the LRU."""

        pool = self._seeders if seed else self._readers
        reads = [pool.submit(self._read, layer, proj, part, e, slot) for proj in self.pool for part in PARTS]
        self.pending[(layer, e)] = Pending(slot, reads, seed)
        self.speculative[(layer, e)] = "seed" if seed else "prefetch"

    def _prefetch(self, layer: int, guesses: list[int], keep: set) -> None:
        """Read the guessed experts of ``layer`` into free or least-recent slots, never evicting an expert of the
        current call, another guess of this call or a read in flight; stop when only those are left."""

        guesses = [e for e in guesses if 0 <= e < self.experts]
        keep = keep | {(layer, e) for e in guesses if (layer, e) in self.lru}   # one guess never evicts another
        for e in guesses:
            key = (layer, e)
            if key in self.lru:
                continue
            if self.free:
                slot = self.free.pop()
            else:
                victim = next((k for k in self.lru if k not in keep and k not in self.pending), None)
                if victim is None:
                    return
                slot = self._drop(victim)
            self.lru[key] = slot
            self._speculate(layer, e, slot, seed=False)
            self.prefetch_issued += 1

    def _seed(self, counts: np.ndarray, keep: set) -> None:
        """Each layer's ``prompt_seed`` most-routed experts of the prompt resident, the most-routed newest in the
        LRU; resident ones are only moved. Never evicts the current call's experts or reads in flight."""

        picks = []
        for layer in range(self.layers):
            order = np.argsort(-counts[layer], kind="stable")[:self.prompt_seed]
            picks += [(int(counts[layer, e]), layer, int(e)) for e in order if counts[layer, e] > 0]
        room = len(self.lru) + len(self.free) - len(keep) - len(self.pending)
        picks = sorted(picks, key=lambda p: (-p[0], p[1], p[2]))[:max(room, 0)]
        placed: set = set()
        for _, layer, e in reversed(picks):       # least routed first: the most routed end up newest
            key = (layer, e)
            placed.add(key)
            if key in self.lru:
                self.lru.move_to_end(key)
                continue
            if self.free:
                slot = self.free.pop()
            else:
                victim = next((k for k in self.lru if k not in keep and k not in self.pending and k not in placed),
                              None)
                if victim is None:
                    return
                slot = self._drop(victim)
            self.lru[key] = slot
            self._speculate(layer, e, slot, seed=True)
            self.seeded += 1

    def _drop(self, key: tuple[int, int]) -> int:
        """Evict ``key`` from the LRU: its slot, a speculative load that was never routed counted as wasted."""

        if self.speculative.pop(key, None) == "prefetch":
            self.prefetch_misses += 1
        return self.lru.pop(key)

    def _load(self, layer: int, ids: list[int]) -> None:
        """Every id resident in the LRU for ``layer``, then its slot written into the table."""

        need, table = [], []
        for e in ids:
            if not 0 <= e < self.experts:
                raise ValueError(f"routed expert {e} is outside the model")
            slot = self.lru.pop((layer, e), None)
            used = self.speculative.pop((layer, e), None) if self.speculative else None
            if used == "prefetch":
                self.prefetch_hits += 1
            elif used == "seed":
                self.seed_hits += 1
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
        need = [(e, e) for e in ids if self.window[e] != layer]
        self._read_all(layer, need)
        for e, _ in need:
            self.window[e] = layer

    def _evict(self, keep: set) -> int:
        for key in self.lru:
            if key not in keep and key not in self.pending:
                return self._drop(key)
        raise MemoryError("the expert pool is smaller than one call's experts")

    def _read_all(self, layer: int, need: list[tuple[int, int]]) -> None:
        reads = [(proj, part, e, slot) for e, slot in need for proj in self.pool for part in PARTS]
        list(self._readers.map(lambda item: self._read(layer, *item), reads))

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
        self.hs.pread_into(getattr(self.pool[proj], part), slot * src.nbytes, self.fds[src.path], offset, src.nbytes)
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
        for entry in self.pending.values():
            for read in entry.reads:
                read.cancel()
        if self._seeders is not None:
            self._seeders.shutdown(wait=True)
        self._readers.shutdown(wait=True)


__all__ = ["PARTS", "Pending", "Slots", "Source", "Streamer", "expert_nbytes", "native_wanted", "pool_threads",
           "sources"]
