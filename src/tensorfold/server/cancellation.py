import select
import socket
import threading
from typing import Any, Callable


class RequestCancelled(Exception):
    pass


class Cancellation:
    def __init__(self, disconnected: Callable[[], bool] | None = None):
        self._event = threading.Event()
        self._disconnected = disconnected

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        if not self._event.is_set() and self._disconnected is not None and self._disconnected():
            self.cancel()
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise RequestCancelled("request cancelled")


def socket_cancellation(connection: socket.socket) -> Cancellation:
    def disconnected() -> bool:
        try:
            if connection.fileno() < 0:
                return True
            ready, _, _ = select.select([connection], [], [], 0)
            if not ready:
                return False
            return connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except BlockingIOError:
            return False
        except OSError:
            return True
        except ValueError:
            return False

    return Cancellation(disconnected)


class PrefillGuard:
    def __init__(self, cancellation: Cancellation, memory: Any = None):
        self.cancellation, self.memory = cancellation, memory

    def before_chunk(self, cache: Any, tokens: int) -> None:
        self.cancellation.check()
        if self.memory is not None:
            self.memory.before_chunk(cache, tokens)
        self.cancellation.check()

    def after_chunk(self, cache: Any, tokens: int) -> None:
        self.cancellation.check()
        if self.memory is not None:
            self.memory.after_chunk(cache, tokens)
        self.cancellation.check()

    def frontier_fits(self, cache: Any, nbytes: int) -> bool:
        """Room for a layer-major group's carried activations (``nbytes``) without evicting anything."""

        fits = getattr(self.memory, "frontier_fits", None)
        return fits is None or fits(cache, nbytes)

    def carry_frontier(self, nbytes: int) -> None:
        """Count ``nbytes`` of group activations in every admission check until it is set back to 0."""

        if self.memory is not None:
            self.memory.frontier = int(nbytes)

    def allow_checkpoint(self, cache: Any, current: Any = None) -> bool:
        """``current``: what the copy's room is measured beside, when not the cache being copied (a checkpoint a
        layer-major group assembled: the working cache and the copy)."""

        self.cancellation.check()
        if self.memory is None:
            return True
        return self.memory.allow_checkpoint(cache) if current is None else self.memory.allow_checkpoint(cache, current)
