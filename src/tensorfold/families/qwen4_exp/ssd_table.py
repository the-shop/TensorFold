"""Read n-gram (PLE) rows from the checkpoint's files at each lookup, holding no table in memory."""

from __future__ import annotations

import os
import struct
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WORKERS = 16                # reads in flight at once: os.pread releases the GIL
MAX_READ = 1 << 20          # bytes one read of adjacent rows may cover
_KINDS = (("weight", "U32", 4), ("scales", "BF16", 2), ("biases", "BF16", 2))


def _no_cache(fd: int) -> None:
    """Keep the file's pages out of the cache (macOS F_NOCACHE) or its readahead off (Linux FADV_RANDOM)."""

    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)       # 48: F_NOCACHE in <sys/fcntl.h>
    elif hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)


def _release(fds: list[int], pool: ThreadPoolExecutor) -> None:
    pool.shutdown(wait=True)
    while fds:
        os.close(fds.pop())


def _span(entry: object, kind: tuple[str, str, int], data: int, size: int, name: str) -> tuple[int, int, int]:
    """(rows, bytes a row, file offset of row 0) of one tensor; refuses a dtype, shape or range it can't read."""

    part, dtype, item = kind
    if not isinstance(entry, dict) or entry.get("dtype") != dtype:
        raise ValueError(f"{name}: the n-gram {part} must be a {dtype} tensor")
    shape, span = entry.get("shape"), entry.get("data_offsets")
    if not (isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)):
        raise ValueError(f"{name}: the n-gram {part} shape {shape} is not [rows, columns]")
    if not (isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span) and span[0] >= 0
            and span[1] - span[0] == shape[0] * shape[1] * item and data + span[1] <= size):
        raise ValueError(f"{name}: the n-gram {part} bytes {span} disagree with its shape or pass the file's end")
    return shape[0], shape[1] * item, data + span[0]


def _fill(reads: list[tuple[int, int, int, memoryview, int]]) -> None:
    """Each (fd, offset, size, out, at) read into ``out[at:]``, finishing short reads; end of file means it changed."""

    for fd, offset, size, out, at in reads:
        done = 0
        while done < size:
            data = os.pread(fd, size - done, offset + done)
            if not data:
                raise OSError(f"short read of the n-gram tables at byte {offset + done}: the checkpoint changed")
            out[at + done:at + done + len(data)] = data
            done += len(data)


EMPTY, GONE = -1, -2            # a bucket never used, and one whose row was evicted
MIB = 1024**2
SMALL = 64                      # ids a cache lookup or insert probes one by one; more go through numpy at once
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


class RowCache:
    """Rows read before, by global id, in a fixed arena of slots evicted in CLOCK order (second chance).

    Every byte it holds is in fixed numpy arrays sized at start, ``nbytes`` in all: the arena, an open-addressing
    hash table of slot numbers (linear probing, at most a quarter full of rows), each slot's row id and bucket,
    and its reference bit."""

    def __init__(self, nbytes: int, widths: tuple[int, ...]) -> None:
        self.widths = tuple(int(w) for w in widths)
        row = sum(self.widths)
        best = (0, 0)
        for bits in range(2, 40):           # the most slots whose arena and index fit in nbytes
            buckets = 1 << bits
            slots = min(buckets // 4, (int(nbytes) - 4 * buckets) // (row + 8 + 4 + 1))
            if slots > best[0]:
                best = (slots, bits)
            if 4 * buckets > int(nbytes):
                break
        self.slots, bits = best
        if self.slots < 1:
            raise ValueError(f"a row cache of {nbytes} bytes holds no {row}-byte row")
        self._shift = np.uint64(64 - bits)
        self._mask = (1 << bits) - 1
        self.arena = np.empty((self.slots, row), dtype=np.uint8)
        self._buckets = np.full(1 << bits, EMPTY, dtype=np.int32)     # bucket -> slot, EMPTY or GONE
        self._ids = np.full(self.slots, -1, dtype=np.int64)            # slot -> row id (-1: unused)
        self._where = np.zeros(self.slots, dtype=np.int32)             # slot -> its bucket
        self._ref = np.zeros(self.slots, dtype=np.uint8)               # used since the hand last passed
        self.nbytes = sum(int(a.nbytes) for a in (self.arena, self._buckets, self._ids, self._where, self._ref))
        self._bucket_at, self._id_at, self._where_at = (memoryview(a) for a in (self._buckets, self._ids, self._where))
        self._used = self._gone = self._hand = 0
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = self.rebuilds = 0

    def stats(self) -> dict[str, int]:
        """Counters since start and the cache's size: for /health."""

        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions, "rebuilds": self.rebuilds,
                    "rows": self._used, "slots": self.slots, "bytes": self.nbytes}

    @staticmethod
    def bytes_for(slots: int, widths: tuple[int, ...]) -> int:
        """The bytes a cache of exactly ``slots`` rows of ``widths`` takes, its index included."""

        return int(slots) * (sum(widths) + 8 + 4 + 1) + 4 * (1 << max(2, (4 * int(slots) - 1).bit_length()))

    def __len__(self) -> int:
        return self._used

    def items(self) -> dict[int, int]:
        """Cached row id -> slot (for tests and inspection)."""

        with self._lock:
            live = np.flatnonzero(self._ids >= 0)
            return dict(zip(self._ids[live].tolist(), live.tolist()))

    def _bucket(self, ids: np.ndarray) -> np.ndarray:
        return ((ids.astype(np.uint64) * _GOLDEN) >> self._shift).astype(np.int64)

    def _find(self, ids: np.ndarray) -> np.ndarray:
        """Each id's slot, or -1."""

        if ids.size <= SMALL:                       # a decode step's few ids: Python's loop beats numpy's calls
            buckets, owner, mask, found = self._bucket_at, self._id_at, self._mask, []
            for key, at in zip(ids.tolist(), self._bucket(ids).tolist()):
                while True:
                    slot = buckets[at]
                    if slot == EMPTY or (slot >= 0 and owner[slot] == key):
                        found.append(slot if slot >= 0 else -1)
                        break
                    at = (at + 1) & mask
            return np.array(found, dtype=np.int64)
        found = np.full(ids.size, -1, dtype=np.int64)
        todo, at = np.arange(ids.size), self._bucket(ids)
        while todo.size:
            slot = self._buckets[at]
            match = (slot >= 0) & (self._ids[np.maximum(slot, 0)] == ids[todo])
            found[todo[match]] = slot[match]
            more = (slot != EMPTY) & ~match
            todo, at = todo[more], (at[more] + 1) & self._mask
        return found

    def _place(self, ids: np.ndarray, slots: np.ndarray) -> None:
        """Index rows ``ids`` (absent) at ``slots``: each takes the first free bucket of its probe, one id a bucket."""

        if ids.size <= SMALL:
            buckets, where, mask = self._bucket_at, self._where_at, self._mask
            for slot, at in zip(slots.tolist(), self._bucket(ids).tolist()):
                while buckets[at] >= 0:
                    at = (at + 1) & mask
                self._gone -= buckets[at] == GONE
                buckets[at] = slot
                where[slot] = at
            return
        todo, at = np.arange(ids.size), self._bucket(ids)
        while todo.size:
            free = np.flatnonzero(self._buckets[at] < 0)
            _, first = np.unique(at[free], return_index=True)
            won = free[first]
            self._gone -= int(np.count_nonzero(self._buckets[at[won]] == GONE))
            self._buckets[at[won]] = slots[todo[won]]
            self._where[slots[todo[won]]] = at[won]
            left = np.ones(todo.size, dtype=bool)
            left[won] = False
            todo, at = todo[left], (at[left] + 1) & self._mask

    def _victims(self, need: int, among: int) -> np.ndarray:
        """``need`` (at most ``among``) of slots ``[0, among)``, the ones holding rows, to reuse, swept from the hand:
        a referenced slot loses its bit and is passed once."""

        chosen = []
        self._hand %= among
        while need:
            window = (self._hand + np.arange(min(among, max(64, 4 * need)))) % among
            zero = np.flatnonzero(self._ref[window] == 0)[:need]
            end = int(zero[-1]) + 1 if zero.size == need else window.size
            self._ref[window[:end]] = 0
            taken = window[zero]
            self._ref[taken] = 1                    # passed again in this sweep, they are cleared, not retaken
            chosen.append(taken)
            need -= taken.size
            self._hand = int((self._hand + end) % among)
        return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)

    def take(self, ids: np.ndarray, outs: list[np.ndarray]) -> np.ndarray:
        """Copy the cached rows of unique ``ids`` into ``outs``' rows; the positions of the rest, to be read."""

        with self._lock:
            slots = self._find(ids)
            hit = np.flatnonzero(slots >= 0)
            rows = None
            if hit.size:
                self._ref[slots[hit]] = 1
                rows = self.arena[slots[hit]]
            self.hits += int(hit.size)
            self.misses += int(ids.size - hit.size)
        if rows is not None:
            at = 0
            for out, width in zip(outs, self.widths):
                out[hit] = rows[:, at:at + width]
                at += width
        return np.flatnonzero(slots < 0)

    def put(self, ids: np.ndarray, parts: list[np.ndarray]) -> None:
        """Keep rows ``ids`` (unique), given as one array a component; past the arena, only the last ones (a gather
        passes the ones its request used last)."""

        keep = min(int(ids.size), self.slots)
        if not keep:
            return
        rows = np.concatenate([part[-keep:] for part in parts], axis=1)
        keys = np.asarray(ids[-keep:], dtype=np.int64)
        with self._lock:
            have = self._find(keys)
            self._ref[have[have >= 0]] = 1          # another gather kept it meanwhile: the same bytes
            new = np.flatnonzero(have < 0)
            if not new.size:
                return
            held = self._used
            fresh = min(new.size, self.slots - held)
            slots = np.arange(held, held + fresh, dtype=np.int64)
            if fresh < new.size:                    # evict only rows held before this put, never its own slots
                old = self._victims(new.size - fresh, held)
                self._buckets[self._where[old]] = GONE
                self._gone += old.size
                self.evictions += int(old.size)
                slots = np.concatenate([slots, old])
            self._used = held + fresh
            self._ids[slots] = keys[new]
            self._ref[slots] = 1
            self.arena[slots] = rows[new]
            if 8 * (self._used + self._gone) > 3 * self._buckets.size:
                self._buckets.fill(EMPTY)           # rows and tombstones past 3/8 of the table: index afresh
                self._gone = 0
                self.rebuilds += 1
                live = np.flatnonzero(self._ids >= 0)
                self._place(self._ids[live], live)
            else:
                self._place(keys[new], slots)


def cache_share(mib: int, tables: int) -> int:
    """Each n-gram table's bytes of a --ple-cache-mib of ``mib``: an even share."""

    return int(mib) * MIB // max(1, int(tables))


def cache_bytes(mib: int, tables: list[tuple[int, tuple[int, ...]]]) -> int:
    """Bytes the row caches of ``tables`` ((rows, widths) each) take for ``mib``: each table's share, capped at all
    its rows and their index. Refuses a share too small for one row."""

    share, total = cache_share(mib, len(tables)), 0
    for rows, widths in tables:
        if share < RowCache.bytes_for(1, widths):
            raise ValueError(f"--ple-cache-mib {mib} gives each of {len(tables)} n-gram tables {share:,} bytes, "
                             f"less than one {sum(widths)}-byte row and its index")
        total += min(share, RowCache.bytes_for(rows, widths))
    return total


def cache_stats(tables: dict[str, "SSDTable"]) -> dict | None:
    """Each named table's row cache counters and size, and their totals; None when no table has a cache."""

    each = {name: t.cache.stats() for name, t in tables.items() if getattr(t, "cache", None) is not None}
    if not each:
        return None
    return {"tables": each, "total": {k: sum(s[k] for s in each.values()) for k in next(iter(each.values()))}}


class SSDTable:
    """HostTable's rows, byte for byte, read from the checkpoint at each gather: deduplicated, coalesced, parallel.

    With ``cache_bytes``, rows already read come from a RowCache of that size instead of the files."""

    def __init__(self, files: list[tuple[Path, dict, dict, dict]], *, nocache: bool = True,
                 cache_bytes: int = 0) -> None:
        self._fds: list[int] = []
        self._pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="ple-ssd")
        self._closer = weakref.finalize(self, _release, self._fds, self._pool)
        self.cache: RowCache | None = None
        try:
            self._layout(files, nocache)
            if cache_bytes > 0:                       # never more than every row of the table and its index
                widths = (self.wrow, self.grow, self.grow)
                self.cache = RowCache(min(int(cache_bytes), RowCache.bytes_for(self.rows, widths)), widths)
        except BaseException:
            self.close()
            raise

    def _layout(self, files: list[tuple[Path, dict, dict, dict]], nocache: bool) -> None:
        if not files:
            raise ValueError("the n-gram table has no shards")
        opened: dict[Path, tuple[int, int, int]] = {}
        starts, fidx, bases, widths = [0], [], [], None
        total = 0
        for path, *entries in files:
            path = Path(path)
            if path not in opened:
                fd = os.open(path, os.O_RDONLY)
                self._fds.append(fd)
                if nocache:
                    _no_cache(fd)
                size, head = os.fstat(fd).st_size, os.pread(fd, 8, 0)
                data = 8 + struct.unpack("<Q", head)[0] if len(head) == 8 else size + 1
                if data > size:
                    raise ValueError(f"{path.name}: truncated safetensors header")
                opened[path] = (len(self._fds) - 1, data, size)
            index, data, size = opened[path]
            (rows, wrow, w0), (srows, grow, s0), (brows, brow, b0) = (
                _span(entry, kind, data, size, path.name) for entry, kind in zip(entries, _KINDS))
            if not rows == srows == brows or wrow not in (8 * grow, 16 * grow) or brow != grow:
                raise ValueError(f"{path.name}: an n-gram shard is not 4- or 8-bit rows with a scale and bias every 32")
            if widths not in (None, (wrow, grow)):
                raise ValueError(f"{path.name}: the n-gram shards differ in row width")
            widths = (wrow, grow)
            total += rows * (wrow + 2 * grow)
            starts.append(starts[-1] + rows)
            fidx.append(index)
            bases.append((w0, s0, b0))
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.fidx = np.array(fidx, dtype=np.int64)
        self.bases = np.array(bases, dtype=np.int64)        # [shard, component]: file offset of row 0
        self.wrow, self.grow = widths
        self.nbytes = total
        self._fd_of = np.array(self._fds, dtype=np.int64)

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows ``ids`` (global) -> words [n, W] uint32, scales and biases [n, G] (bf16 bits as uint16)."""

        if not self._closer.alive:
            raise ValueError("the n-gram table is closed")
        flat = np.asarray(ids).reshape(-1)
        if flat.size and flat.dtype.kind not in "iu":
            raise TypeError(f"n-gram row ids must be integers, not {flat.dtype}")
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise ValueError(f"n-gram row ids must lie in [0, {self.rows})")
        unique, inverse = np.unique(flat.astype(np.int64), return_inverse=True)
        widths = (self.wrow, self.grow, self.grow)
        outs = [np.empty((unique.size, width), dtype=np.uint8) for width in widths]
        cache = self.cache
        if cache is None:
            self._read(unique, outs)
        else:
            miss = cache.take(unique, outs)
            if miss.size == unique.size:
                self._read(unique, outs)
                read = outs
            elif miss.size:
                read = [np.empty((miss.size, width), dtype=np.uint8) for width in widths]
                self._read(unique[miss], read)
                for out, part in zip(outs, read):
                    out[miss] = part
            room = cache.slots - (unique.size - miss.size)     # the request's cached rows stay
            if miss.size > room:                    # more than fits: keep the rows the request used last
                last = np.full(unique.size, -1, dtype=np.int64)
                np.maximum.at(last, inverse.reshape(-1), np.arange(flat.size))
                pick = np.argsort(last[miss], kind="stable")[miss.size - max(0, room):]
                if pick.size:
                    cache.put(unique[miss[pick]], [part[pick] for part in read])
            elif miss.size:
                cache.put(unique[miss], read)
        words, scales, biases = (out[inverse] for out in outs)
        return words.view(np.uint32), scales.view(np.uint16), biases.view(np.uint16)

    def _read(self, unique: np.ndarray, outs: list[np.ndarray]) -> None:
        """Rows ``unique`` (sorted, distinct) from the files into ``outs``, one array a component."""

        if not unique.size:
            return
        shard = np.searchsorted(self.starts, unique, side="right") - 1
        local, where = unique - self.starts[shard], self.fidx[shard]
        reads = []
        for part, (width, out) in enumerate(zip((self.wrow, self.grow, self.grow), outs)):
            reads += self._reads(where, self.bases[shard, part] + local * width, width, out)
        batches = min(WORKERS, len(reads))
        if batches > 1:
            list(self._pool.map(_fill, [reads[i::batches] for i in range(batches)]))
        else:
            _fill(reads)

    def _reads(self, where: np.ndarray, offsets: np.ndarray, width: int, out: np.ndarray) -> list[tuple]:
        """One read per run of rows adjacent in one file, each at most MAX_READ bytes, into ``out``'s rows."""

        n = offsets.size
        if not n:
            return []
        cut = np.ones(n, dtype=bool)
        cut[1:] = (where[1:] != where[:-1]) | (offsets[1:] != offsets[:-1] + width)
        first = np.flatnonzero(cut)
        cut |= (np.arange(n) - first[np.cumsum(cut) - 1]) % max(1, MAX_READ // width) == 0
        first = np.flatnonzero(cut)
        rows = np.diff(first, append=n)
        view = memoryview(out.reshape(-1))
        return list(zip(self._fd_of[where[first]].tolist(), offsets[first].tolist(), (rows * width).tolist(),
                        [view] * first.size, (first * width).tolist()))

    def prefetch(self, workers: int = 8) -> float:
        """Nothing to warm: rows are read at each lookup, so this reads nothing (0 seconds)."""

        return 0.0

    def close(self) -> None:
        """Close the checkpoint's files and the read threads (also done at exit) and drop the row cache; a later
        gather raises."""

        self._closer()
        self.cache = None
