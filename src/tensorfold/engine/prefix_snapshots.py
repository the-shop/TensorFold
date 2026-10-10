"""Store system-prefix caches as tensors and JSON keyed by model and tokens, restoring layer classes by import path."""

from __future__ import annotations

import hashlib
import importlib
import io
import itertools
import json
import os
from pathlib import Path
import struct
import time
from typing import Any, Callable, Sequence
import zlib

import mlx.core as mx
import numpy as np

FORMAT = 1
DEFAULT_DIR = Path.home() / ".cache" / "tensorfold" / "prefix-snapshots"
PARTIAL = ".partial.safetensors"
# a partial without its writer's pid (0.1.0 and earlier) may be another server's write in progress until it is this old
UNNAMED_PARTIAL_SECONDS = 3600
# a partial untouched this long is abandoned whoever named it: its pid may have been reused, or be one we may not signal
STALE_PARTIAL_SECONDS = 24 * 3600
_WRITES = itertools.count()               # this process's writes: two threads never share a partial file
# crc32 of the metadata (but this key), each tensor's name, dtype, shape and offsets, and its bytes; a file without it,
# or whose bytes no longer match it, is never loaded
CHECKSUM = "checksum"
_UNSEALED = "crc32:????????"               # written first, then replaced in place once the tensor bytes are on disk
_READ = 16 * 1024**2


def remove_stale_partials(directory: Path) -> int:
    """Delete the partial writes of processes that have ended (a server stopped mid-write); return the bytes freed."""

    return clean_partials(directory)[1]


def clean_partials(directory: Path) -> tuple[int, int]:
    """``remove_stale_partials`` as (files removed, bytes freed): an empty partial counts as a file."""

    if not directory.is_dir():
        return 0, 0
    removed, freed, now = 0, 0, time.time()
    for path in directory.glob(f"*{PARTIAL}"):
        try:
            stat = path.stat()
            if not _abandoned(path.name, stat.st_mtime, now):
                continue
            path.unlink()
        except OSError:
            continue                          # gone already, or not ours to remove
        removed += 1
        freed += stat.st_size
    return removed, freed


def _abandoned(name: str, mtime: float, now: float) -> bool:
    if now - mtime > STALE_PARTIAL_SECONDS:
        return True
    parts = name[:-len(PARTIAL)].split(".")
    writer = parts[1].split("-")[0] if len(parts) == 2 else ""
    if writer.isdigit():
        return not _running(int(writer))
    return now - mtime > UNNAMED_PARTIAL_SECONDS


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                   # another user's process: running
        return True
    return True


def snapshot_key(model_id: str, tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(model_id.encode())
    digest.update(b"\0")
    digest.update(",".join(str(int(t)) for t in tokens).encode())
    return digest.hexdigest()[:32]


WEIGHT_SAMPLES = 16          # evenly spaced spans of each shard's tensor bytes, besides its whole header and its tail
WEIGHT_SPAN = 4096
TENSOR_SPAN = 256            # bytes read at the start, middle and end of every tensor named in a shard's header
_WEIGHTS: dict[str, str] = {}


def weights_fingerprint(model_dir: Path) -> str:
    """sha256 over each weight shard's name, size, whole header and sampled bytes, and the config and index JSON.

    Part of the snapshot key: new weights at the same path (requantized, retrained, a replaced shard) are another
    model's prefixes. Bytes only, not mtime or inode, so touching or rewriting the same bytes keeps the fingerprint;
    the key also holds the resolved model path, so a checkpoint copied or moved elsewhere starts without snapshots.
    Read once a process.
    Sampled: the start, middle and end of every tensor and evenly spaced spans of each shard, so replacing any one
    tensor changes it, but an edit confined to bytes between those samples (a few weights inside a tensor) does not.
    """

    root = Path(model_dir).resolve()
    known = _WEIGHTS.get(str(root))
    if known is not None:
        return known
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.json")):
        if path.name == "config.json" or path.name.endswith(".index.json"):
            digest.update(f"{path.name}\n".encode())
            digest.update(path.read_bytes())
    for path in sorted(root.glob("*.safetensors")):
        if path.name.endswith(PARTIAL):
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            digest.update(f"{path.name}|{size}\n".encode())
            head = os.pread(fd, 8, 0)
            start = 8 + struct.unpack("<Q", head)[0] if len(head) == 8 else 0
            start = min(start, size)
            raw = os.pread(fd, start, 0)
            digest.update(raw)
            body = size - start
            spans = {(start + body * i // WEIGHT_SAMPLES, WEIGHT_SPAN) for i in range(WEIGHT_SAMPLES)}
            spans.add((max(start, size - WEIGHT_SPAN), WEIGHT_SPAN))
            spans |= _tensor_spans(raw[8:], start, size)
            for at, n in sorted(spans):
                digest.update(os.pread(fd, min(n, size - at), at))
        finally:
            os.close(fd)
    value = digest.hexdigest()[:16]
    _WEIGHTS[str(root)] = value
    return value


def _tensor_spans(header: bytes, start: int, size: int) -> set[tuple[int, int]]:
    """(offset, bytes) at the start, middle and end of each tensor in a shard header (none if it does not parse)."""

    try:
        tensors = json.loads(header)
    except ValueError:
        return set()
    spans: set[tuple[int, int]] = set()
    for name, spec in tensors.items() if isinstance(tensors, dict) else ():
        try:
            begin, end = (start + int(x) for x in spec["data_offsets"])
        except (KeyError, TypeError, ValueError):
            continue
        end = min(end, size)
        if end <= begin:
            continue
        n = min(TENSOR_SPAN, end - begin)
        spans.update({(begin, n), (begin + (end - begin - n) // 2, n), (end - n, n)})
    return spans


def save_snapshot(directory: Path, model_id: str, tokens: Sequence[int], cache: list[Any],
                  *, keep: int = 8) -> Path | None:
    """Write one snapshot unless it is already there; keep the ``keep`` newest."""

    directory.mkdir(parents=True, exist_ok=True)
    key = snapshot_key(model_id, tokens)
    target = directory / f"{key}.safetensors"
    if _touch(target):
        return None
    arrays, meta = snapshot_arrays(model_id, tokens, cache)
    # this process's own (startup knows whether its writer ended), and this write's own (threads never share one)
    partial = target.with_name(f"{key}.{os.getpid()}-{next(_WRITES)}{PARTIAL}")
    try:
        mx.save_safetensors(str(partial), arrays, metadata=meta)
        with open(partial, "r+b") as handle:
            seal(handle)
        sync_file(partial)
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)   # a full disk would otherwise keep these bytes, outside every byte budget
        raise
    sync_dir(directory)
    keep_newest(directory, model_id, keep)
    return target


def _touch(target: Path) -> bool:
    """Mark an existing snapshot as just used; False when it is not there (pruned meanwhile too): write it then."""

    try:
        os.utime(target)
        return True
    except FileNotFoundError:
        return False


def snapshot_arrays(model_id: str, tokens: Sequence[int],
                    cache: list[Any]) -> tuple[dict[str, mx.array], dict[str, str]]:
    """The arrays and metadata of one snapshot file, its checksum still to be sealed."""

    arrays: dict[str, mx.array] = {}
    layers: list[dict[str, Any]] = []
    # an entry with ``stored = False`` (a stream's drafter state) is left out: the family adds a fresh one on resume
    for index, item in enumerate(item for item in cache if getattr(item, "stored", True)):
        materialize = getattr(item, "materialize", None)
        if materialize is not None:
            materialize()
        cls = type(item)
        entry: dict[str, Any] = {"class": f"{cls.__module__}:{cls.__qualname__}", "plain": {},
                                 "arrays": [], "lists": {}, "numpy": []}
        transient = set(getattr(cls, "transient", ()))    # runtime-only state the class rebuilds itself
        for name, value in vars(item).items():
            if name in transient:
                continue
            if isinstance(value, mx.array):
                arrays[f"{index}.{name}"] = value
                entry["arrays"].append(name)
            elif isinstance(value, np.ndarray):          # host-side state (token history of an n-gram layer)
                arrays[f"{index}.{name}"] = mx.array(value)
                entry["numpy"].append(name)
            elif isinstance(value, list) and any(isinstance(v, mx.array) for v in value):
                slots = []
                for slot, element in enumerate(value):
                    if isinstance(element, mx.array):
                        arrays[f"{index}.{name}.{slot}"] = element
                        slots.append(slot)
                    elif element is not None:
                        raise TypeError(f"cannot store {name}[{slot}] of {cls.__name__}")
                entry["lists"][name] = {"length": len(value), "slots": slots}
            elif value is None or isinstance(value, (bool, int, float, str)):
                entry["plain"][name] = value
            else:
                raise TypeError(f"cannot store attribute {name} of {cls.__name__}")
        layers.append(entry)
    meta = {"format": str(FORMAT), "model": model_id, "tokens": json.dumps([int(t) for t in tokens]),
            "layers": json.dumps(layers), "saved": str(time.time()), CHECKSUM: _UNSEALED}
    return arrays, meta


def seal(handle: Any) -> str:
    """Replace the unsealed checksum of a written snapshot (an open file or BytesIO) with its value, in place."""

    handle.seek(0)
    size = struct.unpack("<Q", handle.read(8))[0]
    raw = handle.read(size)
    header = json.loads(raw)

    def read(offset: int, n: int) -> bytes:
        handle.seek(offset)
        return handle.read(n)

    value = _checksum(read, header, 8 + size)
    marker = json.dumps(_UNSEALED).encode()
    at = raw.find(marker)
    if at < 0 or raw.find(marker, at + 1) >= 0 or len(value) != len(_UNSEALED):
        raise ValueError("snapshot header has no single unsealed checksum")
    handle.seek(8 + at + 1)
    handle.write(value.encode())
    handle.flush()
    return value


class _Short(Exception):
    """The file ends before its header says its bytes do."""


def verify_snapshot(path: Path) -> str:
    """"ok" when the file's bytes match its checksum, "unsealed" without one, "corrupt" when they differ, it is short
    or its header does not parse. Any other OSError (permissions, an I/O error, a vanished file) is raised: the file
    may be fine, so it is never treated as corrupt."""

    with open(path, "rb") as handle:
        try:
            head = handle.read(8)
            if len(head) != 8:
                raise _Short
            size = struct.unpack("<Q", head)[0]
            raw = handle.read(size)
            if len(raw) != size:
                raise _Short
            header = json.loads(raw)
            if not isinstance(header, dict):
                return "corrupt"
            stored = (header.get("__metadata__") or {}).get(CHECKSUM)
            if not stored or stored == _UNSEALED:
                return "unsealed"

            def read(offset: int, n: int) -> bytes:
                handle.seek(offset)
                return handle.read(n)

            return "ok" if _checksum(read, header, 8 + size) == stored else "corrupt"
        except OSError:
            raise
        except Exception:  # noqa: BLE001 - short, unparsable or malformed header: corrupt
            return "corrupt"


def _checksum(read: Callable[[int, int], bytes], header: dict[str, Any], base: int) -> str:
    meta = {k: v for k, v in (header.get("__metadata__") or {}).items() if k != CHECKSUM}
    crc = zlib.crc32(json.dumps(meta, sort_keys=True).encode())
    for name in sorted(k for k in header if k != "__metadata__"):
        spec = header[name]
        begin, end = (int(x) for x in spec["data_offsets"])
        crc = zlib.crc32(json.dumps([name, spec["dtype"], spec["shape"], [begin, end]]).encode(), crc)
        at = begin
        while at < end:
            n = min(_READ, end - at)
            data = read(base + at, n)
            if len(data) != n:
                raise _Short(f"short read at {base + at}")
            crc = zlib.crc32(data, crc)
            at += n
    return f"crc32:{crc:08x}"


def encode_snapshot(model_id: str, tokens: Sequence[int], cache: list[Any]) -> memoryview:
    """One snapshot file's bytes, sealed, in host memory: evaluated on the calling thread (the one owning the arrays),
    so another thread can write them with ``write_encoded`` without touching MLX."""

    arrays, meta = snapshot_arrays(model_id, tokens, cache)
    buffer = io.BytesIO()
    mx.save_safetensors(buffer, arrays, metadata=meta)
    del arrays
    seal(buffer)
    return buffer.getbuffer()


STREAM_CHUNK = 64 * 1024**2          # host bytes one streamed piece holds at most


def _dtype_name(dtype: Any) -> str:
    names = {mx.bool_: "BOOL", mx.uint8: "U8", mx.uint16: "U16", mx.uint32: "U32", mx.uint64: "U64",
             mx.int8: "I8", mx.int16: "I16", mx.int32: "I32", mx.int64: "I64", mx.float16: "F16",
             mx.bfloat16: "BF16", mx.float32: "F32"}
    for extra, name in (("float64", "F64"), ("complex64", "C64")):
        if hasattr(mx, extra):
            names[getattr(mx, extra)] = name
    if dtype not in names:
        raise TypeError(f"cannot stream a {dtype} array")
    return names[dtype]


def stream_snapshot(directory: Path, model_id: str, tokens: Sequence[int], cache: list[Any]) -> Path | None:
    """Write one snapshot into a partial file straight from its arrays, ``STREAM_CHUNK`` bytes of host memory at a
    time (no whole host copy), on the calling thread (the one owning the arrays). Returns the partial, unsealed and
    not synced: ``finish_streamed`` seals, syncs and renames it (on any thread). None when the snapshot is there."""

    directory.mkdir(parents=True, exist_ok=True)
    key = snapshot_key(model_id, tokens)
    if _touch(directory / f"{key}.safetensors"):
        return None
    arrays, meta = snapshot_arrays(model_id, tokens, cache)
    names = sorted(arrays)
    header: dict[str, Any] = {"__metadata__": meta}
    offset = 0
    for name in names:
        array = arrays[name]
        header[name] = {"dtype": _dtype_name(array.dtype), "shape": [int(d) for d in array.shape],
                        "data_offsets": [offset, offset + int(array.nbytes)]}
        offset += int(array.nbytes)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 8)                # tensor bytes 8-byte aligned, as safetensors writers do
    partial = directory / f"{key}.{os.getpid()}-{next(_WRITES)}{PARTIAL}"
    try:
        with open(partial, "wb") as handle:
            handle.write(struct.pack("<Q", len(raw)))
            handle.write(raw)
            for name in names:
                flat = arrays[name].reshape(-1)          # row-major, as the file stores it (a view when contiguous)
                step = max(1, STREAM_CHUNK // max(1, flat.itemsize))
                for at in range(0, int(flat.size), step):
                    host = np.array(mx.contiguous(flat[at:at + step]).view(mx.uint8))
                    handle.write(memoryview(host))
                    del host
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return partial


def finish_streamed(partial: Path, tokens: Sequence[int], model_id: str) -> Path:
    """Seal a ``stream_snapshot`` partial, sync it and rename it into place (no MLX: any thread)."""

    target = partial.parent / f"{snapshot_key(model_id, tokens)}.safetensors"
    try:
        with open(partial, "r+b") as handle:
            seal(handle)
        sync_file(partial)
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    sync_dir(partial.parent)
    return target


def write_encoded(directory: Path, model_id: str, tokens: Sequence[int], payload: Any) -> Path | None:
    """Write ``encode_snapshot``'s bytes as the snapshot of ``tokens`` (None when that file is already there)."""

    directory.mkdir(parents=True, exist_ok=True)
    key = snapshot_key(model_id, tokens)
    target = directory / f"{key}.safetensors"
    if _touch(target):
        return None
    partial = target.with_name(f"{key}.{os.getpid()}-{next(_WRITES)}{PARTIAL}")
    try:
        with open(partial, "wb") as handle:
            handle.write(payload)
            handle.flush()
            full_sync(handle.fileno())
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    sync_dir(directory)
    return target


def sync_file(path: Path) -> None:
    """The file's bytes on disk before it is renamed into place: a crash then leaves the old name or a whole file."""

    fd = os.open(path, os.O_RDONLY)
    try:
        full_sync(fd)
    finally:
        os.close(fd)


def full_sync(fd: int) -> None:
    """Flush a file to stable storage: F_FULLFSYNC on macOS, where fsync leaves the bytes in the drive's cache."""

    try:
        import fcntl

        flag = getattr(fcntl, "F_FULLFSYNC", None)
        if flag is not None:
            fcntl.fcntl(fd, flag)
            return
    except OSError:
        pass                                      # a file system without it: fsync below
    os.fsync(fd)


def sync_dir(directory: Path) -> None:
    """The rename itself on disk (best effort: some file systems cannot sync a directory)."""

    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def newest_first(directory: Path) -> list[Path]:
    """Finished snapshot files, newest first; a file another thread or process removes meanwhile is skipped."""

    found: list[tuple[float, Path]] = []
    for path in directory.glob("*.safetensors"):
        if path.name.endswith(PARTIAL):
            continue
        try:
            found.append((path.stat().st_mtime, path))
        except OSError:
            continue
    return [path for _, path in sorted(found, key=lambda item: item[0], reverse=True)]


def keep_newest(directory: Path, model_id: str, keep: int) -> None:
    """Keep the newest ``keep`` of this model only: another model's blocks are not this one's to evict. Past ``keep``,
    blocks of an older key of this model (other weights or kernels) go first, oldest first."""

    ours = []
    for path in newest_first(directory):
        try:
            model = str(read_metadata(path).get("model", ""))
        except Exception:  # noqa: BLE001 - an unreadable file is left alone
            continue
        if model.split("|")[0] == model_id.split("|")[0]:
            ours.append((path, model == model_id))
    over, stale = len(ours) - keep, 0
    for older_key in (True, False):
        for path, current in reversed(ours):
            if over <= 0:
                break
            if current != older_key:
                path.unlink(missing_ok=True)
                over -= 1
                stale += 1 if older_key else 0
    if stale:
        print(f"[tensorfold] removed {stale} snapshot(s) under an older key of this model (other weights or "
              f"kernels) from {directory}", flush=True)


def load_snapshots(directory: Path, model_id: str, *, limit: int | None = None, allow: Any = None):
    """Yield newest model snapshots one at a time so unread blocks do not occupy memory."""

    if not directory.is_dir():
        return
    count = 0
    for path in newest_first(directory):
        if limit is not None and count >= limit:
            return
        try:
            if read_metadata(path).get("model") != model_id:
                continue                          # another configuration's block: not read at all
        except CorruptSnapshot as exc:
            discard_corrupt(path, exc)
            continue
        except Exception:  # noqa: BLE001 - an unreadable file is skipped, never fatal
            continue
        if allow is not None and not allow(path):
            continue
        loaded = load_snapshot(path, model_id)
        if loaded is None:
            continue
        count += 1
        yield loaded


def load_snapshot(path: Path, model_id: str) -> tuple[list[int], list[Any]] | None:
    """One stored block as (tokens, cache), evaluated in the calling thread; None if unusable."""

    try:
        started = time.perf_counter()
        state = verify_snapshot(path)
    except FileNotFoundError:
        return None                               # removed meanwhile
    except OSError as exc:                        # unreadable now (permissions, I/O): kept, maybe fine later
        print(f"[tensorfold] could not read snapshot {path.name}: {type(exc).__name__}: {exc}; left in place",
              flush=True)
        return None
    if state != "ok":
        if state == "corrupt":
            path.unlink(missing_ok=True)          # never read again, never counted against a budget
        print(f"[tensorfold] refused snapshot {path.name}: "
              + ("its bytes do not match its checksum; deleted" if state == "corrupt"
                 else "no checksum (written before checksums); left in place, not loaded")
              + f" ({time.perf_counter() - started:.2f}s)", flush=True)
        return None
    try:
        arrays, meta = mx.load(str(path), return_metadata=True)
        # Evaluate lazy loads here because the scheduler thread has no CPU stream for their arrays.
        mx.eval(list(arrays.values()))
    except Exception:  # noqa: BLE001 - a bad file is skipped, never fatal
        return None
    if meta.get("format") != str(FORMAT) or meta.get("model") != model_id:
        return None
    cache: list[Any] = []
    for index, entry in enumerate(json.loads(meta["layers"])):
        module_name, qualname = entry["class"].split(":")
        cls: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        item = cls.__new__(cls)
        for name, value in entry["plain"].items():
            setattr(item, name, value)
        for name in entry["arrays"]:
            setattr(item, name, arrays[f"{index}.{name}"])
        for name in entry.get("numpy", []):
            setattr(item, name, np.array(arrays[f"{index}.{name}"]))
        for name, spec in entry["lists"].items():
            values: list[Any] = [None] * int(spec["length"])
            for slot in spec["slots"]:
                values[slot] = arrays[f"{index}.{name}.{slot}"]
            setattr(item, name, values)
        cache.append(item)
    return [int(t) for t in json.loads(meta["tokens"])], cache


class DiskBlocks:
    """Index stored prefixes for on-demand loading, refreshing metadata only when files change and touching used blocks for startup priority."""

    def __init__(self, directory: Path, model_id: str) -> None:
        self.directory = Path(directory)
        self.model_id = model_id
        self._known: dict[Path, tuple[float, list[int] | None]] = {}

    def blocks(self) -> list[tuple[Path, list[int]]]:
        if not self.directory.is_dir():
            return []
        known: dict[Path, tuple[float, list[int] | None]] = {}
        for path in self.directory.glob("*.safetensors"):
            if path.name.endswith(".partial.safetensors"):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            entry = self._known.get(path)
            if entry is None or entry[0] != mtime:
                tokens: list[int] | None = None
                try:
                    meta = read_metadata(path)
                    if meta.get("model") == self.model_id and meta.get(CHECKSUM):       # unsealed: never loaded
                        tokens = [int(t) for t in json.loads(meta["tokens"])]
                except CorruptSnapshot as exc:
                    discard_corrupt(path, exc)
                    continue
                except Exception:  # noqa: BLE001 - an unreadable file is skipped, kept for a later look
                    tokens = None
                entry = (mtime, tokens)
            known[path] = entry
        self._known = known
        return [(path, tokens) for path, (_, tokens) in known.items() if tokens]

    def best(self, prompt: Sequence[int], longer_than: int, usable: Any = None,
             skip: Any = ()) -> tuple[Path, list[int]] | None:
        """The longest stored strict prefix of ``prompt`` longer than ``longer_than``, of a length ``usable`` takes,
        not one of the paths in ``skip``."""

        best: tuple[Path, list[int]] | None = None
        for path, tokens in self.blocks():
            if path in skip or (usable is not None and not usable(len(tokens))):
                continue
            if longer_than < len(tokens) < len(prompt) and list(prompt[:len(tokens)]) == tokens:
                if best is None or len(tokens) > len(best[1]):
                    best = (path, tokens)
        return best

    def touch(self, tokens: Sequence[int]) -> None:
        """Mark the stored block with exactly these tokens as just used (newest first at startup)."""

        wanted = [int(t) for t in tokens]
        for path, (mtime, known) in list(self._known.items()):
            if known is not None and len(known) == len(wanted) and known == wanted:
                try:
                    os.utime(path)
                    self._known[path] = (path.stat().st_mtime, known)
                except OSError:
                    pass


class CorruptSnapshot(ValueError):
    """A finished snapshot file whose header is unreadable or whose size disagrees with it: never usable."""


_MAX_HEADER = 256 * 1024**2


def read_metadata(path: Path) -> dict[str, str]:
    """A safetensors file's metadata from its header, without loading any tensor.

    Raises ``CorruptSnapshot`` when the header length passes the file (garbage), the header does not parse, or the
    tensors it lists do not end where the file does (truncated); OSError when the file cannot be read at all.
    """

    with open(path, "rb") as handle:
        total = os.fstat(handle.fileno()).st_size
        head = handle.read(8)
        if len(head) != 8:
            raise CorruptSnapshot("shorter than a header")
        size = struct.unpack("<Q", head)[0]
        if size > total - 8 or size > _MAX_HEADER:
            raise CorruptSnapshot(f"header of {size} B in a {total} B file")
        raw = handle.read(size)
    if len(raw) != size:
        raise CorruptSnapshot("header cut short")
    try:
        header = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CorruptSnapshot(f"header does not parse: {exc}") from None
    if not isinstance(header, dict):
        raise CorruptSnapshot("header is not a table")
    try:
        end = max((int(spec["data_offsets"][1]) for name, spec in header.items() if name != "__metadata__"),
                  default=0)
    except (KeyError, TypeError, ValueError, IndexError):
        raise CorruptSnapshot("tensor entry without offsets") from None
    if 8 + size + end != total:
        raise CorruptSnapshot(f"tensors end at {8 + size + end} B, the file at {total} B")
    meta = header.get("__metadata__") or {}
    if not isinstance(meta, dict):
        raise CorruptSnapshot("metadata is not a table")
    return dict(meta)


def discard_corrupt(path: Path, why: Any) -> None:
    """Delete a corrupt snapshot file and say so: it would otherwise sit outside every budget, skipped forever."""

    try:
        size = path.stat().st_size
        path.unlink()
    except OSError:
        return
    print(f"[tensorfold] deleted corrupt snapshot {path.name} ({size} B): {why}", flush=True)


def blocks_to_warm(directory: Path, model_id: str) -> list[list[int]]:
    """Return longest uncovered token prefixes from other kernel configurations of the same model, newest first; other models' token ids are incompatible."""

    if not directory.is_dir():
        return []
    have: list[list[int]] = []
    other: list[tuple[float, list[int]]] = []
    for path in directory.glob("*.safetensors"):
        if path.name.endswith(".partial.safetensors"):
            continue
        try:
            meta = read_metadata(path)
            tokens = [int(t) for t in json.loads(meta["tokens"])]
        except Exception:  # noqa: BLE001 - an unreadable file is skipped
            continue
        if meta.get("model") == model_id:
            have.append(tokens)
        elif str(meta.get("model", "")).split("|")[0] == model_id.split("|")[0]:
            try:
                other.append((path.stat().st_mtime, tokens))
            except OSError:
                continue
    other.sort(key=lambda item: item[0], reverse=True)
    wanted: list[list[int]] = []
    for _, tokens in other:
        if any(tokens == h for h in have) or any(tokens == w or w[:len(tokens)] == tokens for w in wanted):
            continue
        wanted = [w for w in wanted if tokens[:len(w)] != w]  # a longer block covers its prefixes
        wanted.append(tokens)
    return wanted


__all__ = ["CHECKSUM", "DEFAULT_DIR", "CorruptSnapshot", "DiskBlocks", "blocks_to_warm", "clean_partials",
           "discard_corrupt", "encode_snapshot", "finish_streamed", "full_sync", "keep_newest", "load_snapshot",
           "load_snapshots", "newest_first", "read_metadata", "remove_stale_partials", "save_snapshot", "seal",
           "snapshot_arrays", "snapshot_key", "stream_snapshot", "sync_dir", "sync_file", "verify_snapshot",
           "weights_fingerprint", "write_encoded"]
