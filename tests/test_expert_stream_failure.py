"""A failed SSD expert pool call fails the request: no token from a forward on stale expert slots is returned."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine.family_common import check_expert_stream  # noqa: E402
from tensorfold.engine.lane_engine import LaneStream  # noqa: E402
from tests.lane_fakes import FakeEngine, FakeFamily  # noqa: E402


class FailingPool:
    """A streamer whose ``fail_after``-th forward fails a pool call (the GPU is released, the error recorded)."""

    def __init__(self, fail_after: int) -> None:
        self.error, self.forwards, self.fail_after = None, 0, fail_after


class PooledFamily(FakeFamily):
    def __init__(self, pool: FailingPool) -> None:
        super().__init__()
        self.__dict__["streamer"] = pool

    def hidden(self, inputs, cache, parents=None):
        pool = self.__dict__["streamer"]
        pool.forwards += 1
        if pool.forwards == pool.fail_after:
            pool.error = MemoryError("the expert pool is smaller than one call's experts")
        return super().hidden(inputs, cache, parents)


def stream() -> LaneStream:
    return LaneStream(stream_id="s", prompt_ids=[3, 14, 15], max_new_tokens=20)


def test_the_check_finds_the_pool_of_a_runtime_or_a_backbone():
    check_expert_stream(SimpleNamespace())                                     # no pool
    healthy = SimpleNamespace(error=None)
    check_expert_stream(SimpleNamespace(model=SimpleNamespace(streamer=healthy)))
    failed = SimpleNamespace(error=TimeoutError("no routing from the GPU for layer 3"))
    for model in (SimpleNamespace(model=SimpleNamespace(streamer=failed)), SimpleNamespace(streamer=failed)):
        with pytest.raises(RuntimeError, match="expert streaming failed") as raised:
            check_expert_stream(model)
        assert raised.value.__cause__ is failed.error


def test_a_failed_pool_call_in_a_decode_round_returns_no_tokens():
    pool = FailingPool(fail_after=10**9)
    engine = FakeEngine(PooledFamily(pool))
    engine.add_stream(stream())
    assert any(engine.step().values())                                         # healthy rounds land tokens
    pool.fail_after = pool.forwards + 1
    with pytest.raises(RuntimeError, match="expert streaming failed"):
        engine.step()


def test_a_failed_pool_call_in_the_prompt_fails_the_prefill():
    engine = FakeEngine(PooledFamily(FailingPool(fail_after=1)))
    with pytest.raises(RuntimeError, match="expert streaming failed"):
        engine.add_stream(stream())
    assert not engine.streams
