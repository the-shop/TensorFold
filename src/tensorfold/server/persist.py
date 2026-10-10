"""Each turn's conversation checkpoint to disk behind the request: encoded on the scheduler thread, written on its own."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

from tensorfold.server.checkpoints import extends, prune_conversations, write_with_room


# a BytesIO payload's allocation over its length (CPython over-allocates as it grows; ~1.12x measured)
HOST_OVERHEAD = 1.125
ENCODE_PEAK = 1.25              # what encoding one checkpoint may take at its peak, over the checkpoint's bytes


@dataclass
class _Write:
    tokens: list[int]
    payload: Any                    # encode_snapshot's bytes: host memory, held until written
    partial: Path | None = None     # or a stream_snapshot file already on disk, to seal, sync and rename

    @property
    def nbytes(self) -> int:
        return len(self.payload)

    @property
    def held(self) -> int:
        return int(len(self.payload) * HOST_OVERHEAD)

    def discard(self) -> None:
        """A write that will not happen: its host bytes go, and a streamed partial is removed."""

        self.payload = b""
        if self.partial is not None:
            self.partial.unlink(missing_ok=True)


class TurnWriter:
    """A bounded write-behind queue of sealed snapshot bytes for the session directory.

    Only the newest checkpoint of a conversation waits: a newer one that continues (or equals) a queued one replaces
    it. Past ``max_pending`` the oldest waiting write is dropped (a later turn or the shutdown save covers it). Bytes
    queued or being written are ``inflight_bytes``: memory no admission can reclaim. ``stop`` writes what waits.
    """

    def __init__(self, directory: Path, model_id: str, *, limit_bytes: int, max_pending: int = 2) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.directory = Path(directory)
        self.model_id = model_id
        self.limit_bytes = int(limit_bytes)
        self.max_pending = int(max_pending)
        self._pending: list[_Write] = []           # oldest first
        self._writing: _Write | None = None
        self._cond = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self.queued = self.written = self.coalesced = self.dropped = self.failed = 0
        self.streamed = 0                       # of ``queued``: files streamed from the arrays, no host copy
        # finished turns whose checkpoint is still being copied on the scheduler thread, not yet queued: a kill now
        # loses them (``stats``' "idle" is False until they are queued and written)
        self._copying = 0

    @property
    def inflight_bytes(self) -> int:
        with self._cond:
            return sum(w.held for w in self._pending) + (self._writing.held if self._writing else 0)

    def copy_started(self) -> None:
        with self._cond:
            self._copying += 1

    def copy_done(self) -> None:
        with self._cond:
            self._copying = max(0, self._copying - 1)
            self._cond.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._cond:
            return {"queued": self.queued, "streamed": self.streamed, "written": self.written,
                    "coalesced": self.coalesced,
                    "dropped": self.dropped, "failed": self.failed, "pending": len(self._pending),
                    "turn_copy_pending": self._copying,
                    "idle": not (self._copying or self._pending or self._writing is not None),
                    "inflight_bytes": sum(w.held for w in self._pending)
                    + (self._writing.held if self._writing else 0)}

    def has(self, tokens: list[int]) -> bool:
        """Whether these exact tokens are queued, being written, or already on disk: no need to encode them again."""

        from tensorfold.engine.prefix_snapshots import snapshot_key

        with self._cond:
            if any(w.tokens == tokens for w in self._pending) or (self._writing and self._writing.tokens == tokens):
                return True
        return (self.directory / f"{snapshot_key(self.model_id, tokens)}.safetensors").exists()

    def submit(self, tokens: list[int], payload: Any = b"", *, partial: Path | None = None) -> bool:
        """Queue sealed bytes (``encode_snapshot``), or with ``partial`` a ``stream_snapshot`` file to finish."""

        write = _Write(list(tokens), payload, partial)
        with self._cond:
            if self._stopping:
                write.discard()
                return False
            kept = [w for w in self._pending if not (w.tokens == write.tokens or extends(write.tokens, w.tokens))]
            for gone in self._pending:
                if gone not in kept:
                    gone.discard()
            self.coalesced += len(self._pending) - len(kept)
            while len(kept) >= self.max_pending:
                kept.pop(0).discard()
                self.dropped += 1
            kept.append(write)
            self._pending = kept
            self.queued += 1
            self.streamed += write.partial is not None
            if self._thread is None or not self._thread.is_alive():     # first use, or a thread that died
                self._thread = threading.Thread(target=self._run, name="tensorfold-persist", daemon=True)
                self._thread.start()
            self._cond.notify_all()
        return True

    def wait_idle(self, timeout: float) -> bool:
        """Block until nothing is queued or being written (tests, shutdown)."""

        deadline = time.monotonic() + timeout
        with self._cond:
            while self._pending or self._writing is not None or self._copying:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
        return True

    def stop(self, timeout: float = 60.0) -> bool:
        """Write everything queued, then end the thread; True when it drained within ``timeout``."""

        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                return False
        with self._cond:
            if not self._pending:
                return True
            # the thread ended with writes still queued: drop them (their bytes too) and say so
            self.dropped += len(self._pending)
            for gone in self._pending:
                gone.discard()
            self._pending = []
            self._cond.notify_all()
        return False

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stopping:
                    self._cond.wait()
                if not self._pending:
                    return                         # stopping, and drained
                write = self._pending.pop(0)
                self._writing = write
            try:
                self._write(write)
            except Exception as exc:  # noqa: BLE001 - this thread outlives any one write
                with self._cond:
                    self.failed += 1
                _log(f"[tensorfold] turn checkpoint write failed: {type(exc).__name__}: {exc}")
            finally:
                write.discard()                    # a partial a failed finish left behind is removed too
                with self._cond:
                    self._writing = None
                    self._cond.notify_all()

    def _write(self, write: _Write) -> None:
        from tensorfold.engine.prefix_snapshots import finish_streamed, write_encoded

        started = time.perf_counter()
        if write.partial is not None:
            size = write.partial.stat().st_size
            if size > self.limit_bytes:
                raise OSError(f"{size} B passes the {self.limit_bytes} B session budget")
            path: Path | None = finish_streamed(write.partial, write.tokens, self.model_id)
            write.partial = None
        else:
            if write.nbytes > self.limit_bytes:
                raise OSError(f"{write.nbytes} B passes the {self.limit_bytes} B session budget")
            path = write_with_room(self.directory, self.model_id, write.nbytes,
                                   lambda: write_encoded(self.directory, self.model_id, write.tokens, write.payload))
        with self._cond:
            self.written += 1
        try:
            prune_conversations(self.directory, self.model_id, self.limit_bytes)
        except Exception as exc:  # noqa: BLE001 - the file is written; a failed prune is retried next time
            _log(f"[tensorfold] session prune failed: {type(exc).__name__}: {exc}")
        if path is not None:
            try:
                size = path.stat().st_size
            except OSError:
                size = write.nbytes
            _log(f"[tensorfold] persisted turn checkpoint tokens={len(write.tokens)} "
                 f"({size / 1024**3:.2f} GiB) in {time.perf_counter() - started:.2f}s")


def _log(line: str) -> None:
    """A log line that never raises (a closed or broken stdout must not end the writer thread)."""

    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001
        pass


__all__ = ["TurnWriter"]
