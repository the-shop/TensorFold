"""Admission releases the family's last-round rollback rows before it measures, and only with no stream live."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from tensorfold.server.app import ChatApp
from tensorfold.server.errors import RequestError
from tests.lane_fakes import FakeEngine
from tests.test_lane_server import FakeTokenizer
from tests.test_memory_window import LIVE, Runtime, SizedFamily

MIB = 2**20


class RowsFamily(SizedFamily):
    """A whole prompt of at most ``fused_rows`` tokens leaves per-row states behind (as FlashNext's fused kernels did
    before its ``prefill`` copied them out) until ``release_rounds``; anything else leaves none."""

    fused_rows = 16

    def __init__(self, rows_bytes: int, delay: float = 0.0) -> None:
        super().__init__(delay=delay)
        self.rows_bytes, self.held, self.releases = int(rows_bytes), 0, []

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> Any:
        import numpy as np

        if np.array(inputs).size <= self.fused_rows and not cache[0].rows[0]:     # a whole short prompt
            self.held = self.rows_bytes
        return super().hidden(inputs, cache, parents)

    def release_rounds(self) -> None:
        engine = getattr(self, "engine", None)          # set once the app has built it (startup sizing calls first)
        self.releases.append(engine.active_count if engine is not None else 0)
        self.held = 0


class RowsRuntime(Runtime):
    def __init__(self, weights: int, family: RowsFamily) -> None:
        super().__init__(weights)
        self.family = family

    def get_active_memory(self) -> int:
        return self.weights + self.family.held + sum(item.keys.nbytes + item.values.nbytes for item in list(LIVE))


def served(rows_bytes: int, *, lanes: int = 1, delay: float = 0.0) -> ChatApp:
    model = RowsFamily(rows_bytes, delay=delay)
    model.args = type("Args", (), {"num_attention_heads": 1, "head_dim": 128})()
    app = ChatApp(model, FakeTokenizer(), served_name="fake", lanes=lanes, default_max_tokens=16, checkpoint_slots=4,
                  use_proposer=False, engine_factory=FakeEngine, memory_budget_bytes=int(1.5 * 2**30),
                  memory_overhead_bytes=0, memory_runtime=RowsRuntime(2**30, model), context_window=8192)
    model.engine = app.scheduler.engine
    return app


def test_open_job_releases_rounds_before_it_begins_admission():
    app = served(0)
    try:
        order: list[str] = []
        engine, memory = app.scheduler.engine, app.prompt_memory
        release, begin = engine.release_rounds, memory.begin
        engine.release_rounds = lambda: order.append("release") or release()
        memory.begin = lambda *a, **k: order.append("begin") or begin(*a, **k)
        app.chat([{"role": "user", "content": "hello there"}], max_tokens=4)
        assert order[:2] == ["release", "begin"]
    finally:
        app.close()


def test_a_live_stream_keeps_its_rows_when_another_request_opens():
    app = served(0, lanes=2, delay=0.004)
    family = app.scheduler.engine.model
    try:
        family.releases.clear()
        opened: list[int] = []
        engine = app.scheduler.engine
        release = engine.release_rounds
        engine.release_rounds = lambda: opened.append(engine.active_count) or release()
        done: dict[str, Any] = {}
        long = threading.Thread(target=lambda: done.update(r=app.chat([{"role": "user", "content": "long one"}],
                                                                        max_tokens=60)))
        long.start()
        while not app.scheduler.active:
            time.sleep(0.001)
        app.chat([{"role": "user", "content": "short"}], max_tokens=2)
        long.join(timeout=30)
        assert done["r"]["content"]
        assert any(active > 0 for active in opened)          # the second request opened beside a live stream ...
        assert family.releases and all(active == 0 for active in family.releases)    # ... and released nothing
    finally:
        app.close()


def test_a_failed_short_prefill_does_not_refuse_the_next_request():
    # 1 GiB of weights in a 1.5 GiB budget: a request fits with ~0.25 GiB to spare, but not beside 400 MiB of rows
    app = served(400 * MIB)
    family = app.scheduler.engine.model
    try:
        with pytest.raises(RequestError, match="GiB"):
            app.chat([{"role": "user", "content": "hi"}], max_tokens=4)         # 5 tokens: the fused path's rows
        assert family.held == 400 * MIB                                          # left behind by the failed prefill
        reply = app.chat([{"role": "user", "content": "a prompt longer than sixteen tokens"}], max_tokens=4)
        assert reply["content"] is not None and reply["completion_tokens"] > 0
    finally:
        app.close()


def test_without_the_release_the_failed_prefill_would_refuse_everything():
    """The control for the test above: the same rows, never released, refuse a request that fits on its own."""

    app = served(400 * MIB)
    try:
        with pytest.raises(RequestError, match="GiB"):
            app.chat([{"role": "user", "content": "hi"}], max_tokens=4)
        app.scheduler.engine.release_rounds = lambda: None
        with pytest.raises(RequestError, match="GiB"):
            app.chat([{"role": "user", "content": "a prompt longer than sixteen tokens"}], max_tokens=4)
    finally:
        app.close()
