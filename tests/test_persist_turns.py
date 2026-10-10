"""--persist-turns: each turn's conversation checkpoint reaches disk behind the reply, and survives a crash."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine import prefix_snapshots as ps  # noqa: E402
from tensorfold.server.persist import TurnWriter  # noqa: E402
from tests.lane_fakes import FakeBatchItem, FakeEngine, FakeFamily  # noqa: E402
from tests.test_lane_server import expected_reply, make_app  # noqa: E402

MODEL = "/models/fake|weights=abc"


class DiskItem(FakeBatchItem):
    """The fake cache layer, storable: its history goes to disk as a numpy array and comes back as rows."""

    transient = ("rows",)

    def materialize(self) -> None:
        self.history = np.array(self.rows[0], dtype=np.int64)

    def __getattr__(self, name: str) -> Any:
        if name == "rows":
            self.rows = [[int(t) for t in self.__dict__["history"].reshape(-1).tolist()]]
            return self.rows
        raise AttributeError(name)


class DiskFamily(FakeFamily):
    def make_cache(self) -> list[Any]:
        return [DiskItem([[]])]


class DiskEngine(FakeEngine):
    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        super().__init__(DiskFamily(), **kwargs)

    @staticmethod
    def copy_single_cache(cache: list[Any]) -> list[Any]:
        return [DiskItem([list(r) for r in item.rows]) for item in cache]


SYSTEM = {"role": "system", "content": "You are a careful assistant who answers in full sentences."}
TURN1 = [SYSTEM, {"role": "user", "content": "Tell me about the sea and its tides, slowly."}]


def _turn2(reply: str) -> list[dict[str, Any]]:
    return [*TURN1, {"role": "assistant", "content": reply}, {"role": "user", "content": "And the moon?"}]


def _app(tmp_path, **kwargs: Any):
    return make_app(engine_factory=DiskEngine, lanes=1, snapshot_dir=tmp_path / "prefix-snapshots",
                    model_id=MODEL, **kwargs)


def _written(app, n: int, timeout: float = 10.0) -> bool:
    """The reply returns before the scheduler thread queues its checkpoint: wait for ``n`` writes."""

    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app.persist.stats()["written"] >= n and app.persist.wait_idle(timeout):
            return True
        time.sleep(0.01)
    return False


def _sessions(tmp_path):
    return sorted((tmp_path / "session-snapshots").glob("*.safetensors"))


def test_a_turn_is_on_disk_before_any_stop_and_a_new_server_resumes_it_with_identical_tokens(tmp_path):
    first = _app(tmp_path, persist_turns=True)
    try:
        reply1 = first.chat(TURN1, max_tokens=6)
        assert _written(first, 1)
        files = _sessions(tmp_path)
        assert len(files) == 1 and ps.verify_snapshot(files[0]) == "ok"
        assert first.persist.stats()["written"] == 1
    finally:
        first.scheduler.stop(timeout=5.0)       # a crash: no shutdown hook, no conversation save
    assert len(_sessions(tmp_path)) == 1
    turn2 = _turn2(reply1["content"])
    second = _app(tmp_path)
    cold = make_app(engine_factory=DiskEngine, lanes=1)
    try:
        resumed = second.chat(turn2, max_tokens=8)
        fresh = cold.chat(turn2, max_tokens=8)
    finally:
        second.scheduler.stop(timeout=5.0)
        cold.scheduler.stop(timeout=5.0)
    tokens, text = expected_reply(second, turn2, 8)
    stored = len(ps.read_metadata(files[0])["tokens"].split(","))
    assert resumed["cached_tokens"] == stored > 0 and fresh["cached_tokens"] == 0
    assert resumed["runtime"]["token_sha"] == fresh["runtime"]["token_sha"]
    assert resumed["content"] == fresh["content"] == text


def test_replies_are_identical_with_persisting_on_and_off(tmp_path):
    on, off = _app(tmp_path / "on", persist_turns=True), _app(tmp_path / "off")
    try:
        shas = []
        for app in (on, off):
            first = app.chat(TURN1, max_tokens=6)
            second = app.chat(_turn2(first["content"]), max_tokens=6)
            shas.append((first["runtime"]["token_sha"], second["runtime"]["token_sha"], second["cached_tokens"]))
        assert shas[0] == shas[1]
        assert _written(on, 2) and on.persist.stats()["written"] == 2
        assert off.persist is None and not (tmp_path / "off" / "session-snapshots").exists()
    finally:
        on.scheduler.stop(timeout=5.0)
        off.scheduler.stop(timeout=5.0)


def test_a_graceful_close_drains_the_writer_and_saves_nothing_twice(tmp_path):
    app = _app(tmp_path, persist_turns=True)
    app.chat(TURN1, max_tokens=6)
    app.close()
    assert app.persist.stats()["pending"] == 0 and app.persist.stats()["written"] == 1
    tokens = sorted(len(ps.read_metadata(path)["tokens"].split(",")) for path in _sessions(tmp_path))
    assert len(tokens) == len(set(tokens))              # the shutdown save skipped the persisted turn


def test_a_background_request_is_not_persisted(tmp_path):
    app = _app(tmp_path, persist_turns=True)
    try:
        app.chat(TURN1, max_tokens=4, sampling={"priority": "background"})
        app.chat(TURN1, max_tokens=4)              # the foreground turn resumes the background's checkpoint
        assert _written(app, 1) and app.persist.stats()["queued"] == 1        # and persists it, once
    finally:
        app.scheduler.stop(timeout=5.0)


def _kv(n: int) -> list[Any]:
    from mlx_lm.models.cache import KVCache

    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, n, 64)), mx.ones((1, 2, n, 64)))
    mx.eval(kv.state)
    return [kv]


def _persisting_scheduler(tmp_path, budget: int):
    """A scheduler with the real PromptMemory (a fake runtime counting the store's arrays) and a turn writer."""

    from tests.test_prompt_memory import Runtime, controller

    from tensorfold.server.checkpoints import CheckpointStore
    from tensorfold.server.memory_budget import cache_nbytes
    from tensorfold.server.scheduler import ChatJob, Scheduler

    runtime = Runtime(resident=1000)
    store = CheckpointStore(4, copier=lambda c: c, sizer=cache_nbytes)
    memory = controller(budget=budget, store=store, runtime=runtime)
    cache = _kv(256)
    runtime.caches.append(cache)
    tokens = list(range(1, 257))
    store.insert(tokens, cache, last_prompt=[*tokens, 0])
    memory.observe_cache(cache)
    scheduler = Scheduler(DiskEngine(), lanes=1, eos_ids=frozenset(), checkpoints=store, prompt_memory=memory,
                          model_id=MODEL)
    scheduler.persist = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    job = ChatJob(job_id="j", prompt_ids=[*tokens, 0], max_tokens=16, temperature=0.0, persist_tokens=tokens)
    return scheduler, memory, job, cache_nbytes(cache)


def test_the_host_copy_is_made_only_when_it_fits_beside_the_next_turn(tmp_path):
    scheduler, memory, job, size = _persisting_scheduler(tmp_path / "roomy", budget=1 << 30)
    scheduler._persist_turn(job)
    assert scheduler.persist.stop(5.0) and scheduler.persist.stats()["written"] == 1
    assert scheduler.persist.stats()["streamed"] == 0

    scheduler, memory, job, size = _persisting_scheduler(tmp_path / "tight", budget=1 << 30)
    entry = scheduler.checkpoints._entries[0]
    memory.prompt, memory.reply = len(job.prompt_ids), 16
    need = memory.projected(memory.prompt, current_cache=entry.cache)   # the next turn, resuming from the entry
    memory.prompt = memory.reply = 0
    memory.budget = need + size // 2                    # the next turn fits; the copy beside it does not
    assert memory.room_for(0, tokens=len(job.prompt_ids), reply=16, current_cache=entry.cache)
    assert not memory.room_for(int(size * 1.25), tokens=len(job.prompt_ids), reply=16, current_cache=entry.cache)
    scheduler._persist_turn(job)                        # no room for a host copy: streamed from the arrays instead
    assert scheduler.persist.stop(5.0)
    stats = scheduler.persist.stats()
    assert stats["streamed"] == 1 and stats["written"] == 1 and stats["inflight_bytes"] == 0
    [path] = list((tmp_path / "tight").glob("*.safetensors"))
    assert ps.verify_snapshot(path) == "ok" and not list((tmp_path / "tight").glob("*.partial.safetensors"))
    tokens, cache = ps.load_snapshot(path, MODEL)
    original = entry.cache[0].state
    assert tokens == list(range(1, 257))
    assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(cache[0].state, original))


def test_the_next_turns_reserve_never_passes_the_context_window(tmp_path):
    scheduler, memory, job, size = _persisting_scheduler(tmp_path, budget=1 << 30)
    entry = scheduler.checkpoints._entries[0]
    memory.window = len(job.prompt_ids) + 64
    job.max_tokens = 65536                              # what Pi sends
    memory.prompt, memory.reply = len(job.prompt_ids), 64
    need = memory.projected(memory.prompt, current_cache=entry.cache)
    memory.prompt = memory.reply = 0
    memory.budget = need + 2 * size                      # room for the copy beside a window-sized next turn only
    assert not memory.room_for(int(size * 1.25), tokens=len(job.prompt_ids), reply=65536, current_cache=entry.cache)
    scheduler._persist_turn(job)
    assert scheduler.persist.stop(5.0) and scheduler.persist.stats()["streamed"] == 0
    assert scheduler.persist.stats()["written"] == 1


class Mixed:
    def __init__(self):
        self.a = mx.arange(12, dtype=mx.bfloat16).reshape(3, 4).T        # not contiguous
        self.b = mx.array([True, False, True])
        self.c = mx.arange(6, dtype=mx.int64)
        self.d = mx.ones((2, 2), dtype=mx.float16)
        self.e = mx.array(7, dtype=mx.uint32)                           # 0-d
        self.f = mx.zeros((0, 3), dtype=mx.float32)                      # empty
        self.offset = 3


class Big:
    def __init__(self):
        self.keys = mx.ones((1, 4, 1000, 64), dtype=mx.bfloat16)        # 512 KiB


def test_a_streamed_snapshot_equals_an_encoded_one_for_every_dtype(tmp_path):
    partial = ps.stream_snapshot(tmp_path, MODEL, [1, 2], [Mixed()])
    assert partial.name.endswith(".partial.safetensors")
    path = ps.finish_streamed(partial, [1, 2], MODEL)
    assert ps.verify_snapshot(path) == "ok" and ps.read_metadata(path)["model"] == MODEL
    tokens, [loaded] = ps.load_snapshot(path, MODEL)
    want = Mixed()
    for name in "abcdef":
        got, ref = getattr(loaded, name), getattr(want, name)
        assert got.dtype == ref.dtype and got.shape == ref.shape and bool(mx.array_equal(got, ref).item()), name


def test_a_streamed_write_uses_bounded_host_pieces(tmp_path, monkeypatch):
    sizes = []
    real = ps.np.array

    def record(value, *args, **kwargs):
        out = real(value, *args, **kwargs)
        sizes.append(out.nbytes)
        return out

    monkeypatch.setattr(ps, "STREAM_CHUNK", 4096)
    monkeypatch.setattr(ps.np, "array", record)

    partial = ps.stream_snapshot(tmp_path, MODEL, [5], [Big()])
    monkeypatch.undo()
    assert sizes and max(sizes) <= 4096 and sum(sizes) == 4 * 1000 * 64 * 2
    path = ps.finish_streamed(partial, [5], MODEL)
    assert bool(mx.array_equal(ps.load_snapshot(path, MODEL)[1][0].keys, Big().keys).item())


def test_a_superseded_streamed_partial_is_removed(tmp_path, monkeypatch):
    gate = Gate(monkeypatch)
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    writer.submit([9], _payload([9]))
    assert gate.started.wait(5.0)
    first = ps.stream_snapshot(tmp_path, MODEL, [1, 2], [Layer()])
    writer.submit([1, 2], partial=first)
    writer.submit([1, 2, 3], _payload([1, 2, 3]))             # the same conversation, one turn on
    assert not first.exists()
    gate.release.set()
    assert writer.stop(5.0) and not list(tmp_path.glob("*.partial.safetensors"))


def test_a_system_block_is_never_persisted_as_a_conversation(tmp_path):
    scheduler, memory, job, size = _persisting_scheduler(tmp_path, budget=1 << 30)
    scheduler.checkpoints._entries[0].pinned = True      # resumed from prefix-snapshots, say
    scheduler._persist_turn(job)
    assert scheduler.persist.stats()["queued"] == 0 and list(tmp_path.iterdir()) == []


def test_room_for_uses_the_given_turn_not_whichever_job_began_last(tmp_path):
    scheduler, memory, job, size = _persisting_scheduler(tmp_path, budget=1 << 30)
    memory.begin(1 << 22, 1 << 16, admit=False)          # another request, far larger, just began
    assert memory.room_for(size, tokens=len(job.prompt_ids), reply=16) is True
    assert memory.room_for(size) is True and (memory.prompt, memory.reply) == (1 << 22, 1 << 16)
    assert memory.projected(memory.prompt) > memory.budget        # its own numbers would have refused the copy


def test_a_quick_follow_up_waits_for_the_last_turns_copy_instead_of_being_refused(tmp_path):
    from tensorfold.server.errors import RequestError

    scheduler, memory, job, size = _persisting_scheduler(tmp_path, budget=1 << 30)
    entry = scheduler.checkpoints._entries[0]
    memory.begin(len(job.prompt_ids) + 64, 64, admit=False)
    need = memory.projected(memory.prompt, current_cache=entry.cache)
    memory.budget = need + size // 2
    pending = [2 * size]                                 # the last turn's copy, queued or being written
    memory.pending_bytes = lambda: pending[0]
    with pytest.raises(RequestError):
        memory.require(current_cache=entry.cache, keep=entry)   # nothing waits for the writer: refused
    waited = []

    def drain(timeout: float) -> bool:
        waited.append(timeout)
        pending[0] = 0                                   # written: its bytes are freed
        return True

    pending[0] = 2 * size
    memory.wait_pending = drain
    memory.require(current_cache=entry.cache, keep=entry)       # admitted once the copy is on disk
    assert waited == [memory.pending_wait_s] and len(scheduler.checkpoints) == 1

    pending[0] = 2 * size
    memory.wait_pending = lambda timeout: False          # the writer did not finish in time: refused, not hung
    with pytest.raises(RequestError):
        memory.require(current_cache=entry.cache, keep=entry)


def test_the_app_wires_the_writers_bytes_and_drain_into_admission(tmp_path):
    from tests.test_prompt_memory import Runtime

    app = _app(tmp_path, persist_turns=True, memory_budget_bytes=1 << 34, memory_runtime=Runtime())
    try:
        assert app.prompt_memory.wait_pending == app.persist.wait_idle
        app.chat(TURN1, max_tokens=4)
        assert _written(app, 1)
        assert app.prompt_memory.pending_bytes() == app.persist.inflight_bytes == 0
    finally:
        app.close()


# -- the writer --------------------------------------------------------------


class Gate:
    """A ``write_encoded`` that waits for the test to let each write through."""

    def __init__(self, monkeypatch) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.seen: list[list[int]] = []
        real = ps.write_encoded

        def write(directory, model_id, tokens, payload):
            self.started.set()
            assert self.release.wait(10.0)
            self.seen.append(list(tokens))
            return real(directory, model_id, tokens, payload)

        monkeypatch.setattr(ps, "write_encoded", write)


class Layer:
    def __init__(self, n: int = 64) -> None:
        self.keys = mx.ones((n, 64))


def _payload(tokens, n: int = 64):
    return ps.encode_snapshot(MODEL, tokens, [Layer(n)])


def test_queued_bytes_are_in_flight_until_written_then_released(tmp_path, monkeypatch):
    gate = Gate(monkeypatch)
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    one, two = _payload([1, 2]), _payload([7, 8])
    writer.submit([1, 2], one)
    assert gate.started.wait(5.0)
    writer.submit([7, 8], two)
    from tensorfold.server.persist import HOST_OVERHEAD

    held = int(len(one) * HOST_OVERHEAD) + int(len(two) * HOST_OVERHEAD)
    assert writer.inflight_bytes == held                       # one being written, one waiting
    gate.release.set()
    assert writer.wait_idle(10.0) and writer.inflight_bytes == 0
    assert writer.stats()["written"] == 2 and len(list(tmp_path.glob("*.safetensors"))) == 2
    assert writer.stop(5.0)


def test_only_the_newest_checkpoint_of_a_conversation_waits(tmp_path, monkeypatch):
    gate = Gate(monkeypatch)
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    writer.submit([9], _payload([9]))                          # being written: holds the thread
    assert gate.started.wait(5.0)
    writer.submit([1, 2], _payload([1, 2]))
    writer.submit([1, 2, 3, 4], _payload([1, 2, 3, 4]))        # the same conversation, one turn on
    writer.submit([5, 6], _payload([5, 6]))                    # another conversation
    assert writer.stats()["coalesced"] == 1 and writer.stats()["pending"] == 2
    gate.release.set()
    assert writer.stop(10.0)
    assert gate.seen == [[9], [1, 2, 3, 4], [5, 6]]


def test_the_queue_is_bounded_and_drops_its_oldest(tmp_path, monkeypatch):
    gate = Gate(monkeypatch)
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30, max_pending=2)
    writer.submit([9], _payload([9]))
    assert gate.started.wait(5.0)
    for tokens in ([1], [2], [3]):
        writer.submit(tokens, _payload(tokens))
    assert writer.stats()["dropped"] == 1 and writer.stats()["pending"] == 2
    gate.release.set()
    assert writer.stop(10.0) and gate.seen == [[9], [2], [3]]


def test_stop_writes_everything_queued_then_refuses_more(tmp_path, monkeypatch):
    gate = Gate(monkeypatch)
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    writer.submit([1], _payload([1]))
    writer.submit([2], _payload([2]))
    assert gate.started.wait(5.0)
    threading.Timer(0.2, gate.release.set).start()
    assert writer.stop(10.0)
    assert gate.seen == [[1], [2]] and writer.inflight_bytes == 0
    assert writer.submit([3], _payload([3])) is False
    found = {tuple(tokens) for _, tokens in ps.DiskBlocks(tmp_path, MODEL).blocks()}
    assert found == {(1,), (2,)}


def test_a_failed_write_is_counted_and_leaves_nothing_behind(tmp_path, monkeypatch):
    def full(path, *args, **kwargs):
        raise OSError(28, "No space left on device")

    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    monkeypatch.setattr(ps, "full_sync", full)
    writer.submit([1], _payload([1]))
    assert writer.wait_idle(5.0)
    assert writer.stats()["failed"] == 1 and list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    writer.submit([2], _payload([2]))                          # the next one is tried again
    assert writer.stop(5.0) and writer.stats()["written"] == 1


def test_a_write_past_the_session_budget_is_refused(tmp_path):
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=100)
    writer.submit([1], _payload([1]))
    assert writer.stop(5.0) and writer.stats()["failed"] == 1 and list(tmp_path.iterdir()) == []


def test_the_written_bytes_equal_a_direct_save(tmp_path):
    writer = TurnWriter(tmp_path / "a", MODEL, limit_bytes=1 << 30)
    writer.submit([4, 5], _payload([4, 5]))
    assert writer.stop(5.0)
    [path] = list((tmp_path / "a").glob("*.safetensors"))
    tokens, cache = ps.load_snapshot(path, MODEL)
    assert tokens == [4, 5] and bool(mx.array_equal(cache[0].keys, Layer().keys).item())
    direct = ps.save_snapshot(tmp_path / "b", MODEL, [4, 5], [Layer()])
    assert direct.name == path.name                            # the same key: either path finds the other's file


def test_has_sees_queued_and_written_checkpoints(tmp_path, monkeypatch):
    gate = Gate(monkeypatch)
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    writer.submit([1, 2], _payload([1, 2]))
    assert gate.started.wait(5.0) and writer.has([1, 2]) and not writer.has([1, 3])
    gate.release.set()
    assert writer.stop(5.0) and writer.has([1, 2])


# -- memory accounting -------------------------------------------------------


def test_in_flight_bytes_count_as_held_never_as_reclaimable():
    from tests.test_prompt_memory import Runtime, controller

    from tensorfold.server.checkpoints import CheckpointStore

    runtime = Runtime(resident=1000)
    store = CheckpointStore(4, copier=lambda c: c, sizer=lambda c: 0)
    memory = controller(budget=10_000, store=store, runtime=runtime)
    held, used = memory.held(), memory._used()
    fits = memory.would_fit(1, 1)
    pending = [0]
    memory.pending_bytes = lambda: pending[0]
    pending[0] = 4000
    assert memory.held() == held + 4000 and memory._used() == used + 4000
    assert memory.room_for(4000) is True and memory.room_for(6000) is False
    pending[0] = 9500
    assert fits and memory.would_fit(1, 1) is False            # no eviction can free bytes on their way to disk
    pending[0] = 0
    assert memory.held() == held and memory.room_for(6000) is True


def test_a_broken_stdout_never_ends_the_writer(tmp_path, monkeypatch):
    import builtins

    real_print = builtins.print

    def broken(*args, **kwargs):
        if "turn checkpoint" in str(args[0] if args else ""):
            raise BrokenPipeError(32, "Broken pipe")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", broken)
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    writer.submit([1], _payload([1]))
    assert writer.wait_idle(5.0)
    writer.submit([2], _payload([2]))
    assert writer.stop(5.0) and writer.stats()["written"] == 2


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_writer_thread_that_died_is_started_again_and_stop_reports_lost_writes(tmp_path, monkeypatch):
    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)

    def dies(write):
        raise SystemExit                       # not an Exception: ends the thread as an unexpected error would

    monkeypatch.setattr(writer, "_write", dies)
    writer.submit([1], _payload([1]))
    writer._thread.join(5.0)
    assert not writer._thread.is_alive() and writer.inflight_bytes == 0
    monkeypatch.undo()
    writer.submit([2], _payload([2]))         # a new thread takes it
    assert writer.stop(5.0) and writer.stats()["written"] == 1

    stuck = TurnWriter(tmp_path / "s", MODEL, limit_bytes=1 << 30)
    monkeypatch.setattr(stuck, "_write", dies)
    stuck.submit([3], _payload([3]))
    stuck._thread.join(5.0)
    with stuck._cond:                          # a write queued after the thread died, before anyone restarted it
        from tensorfold.server.persist import _Write

        stuck._pending.append(_Write([4], _payload([4])))
    assert stuck.stop(1.0) is False and stuck.inflight_bytes == 0 and stuck.stats()["dropped"] == 1


def test_flags_that_cannot_work_say_so_at_startup(tmp_path, capsys):
    for kwargs, why in (({"snapshot_dir": None}, "snapshots are off"),
                        ({"checkpoint_slots": 0}, "the prompt cache is off")):
        settings = {"engine_factory": DiskEngine, "lanes": 1, "model_id": MODEL,
                    "snapshot_dir": tmp_path / "prefix-snapshots", **kwargs}
        app = make_app(persist_turns=True, checkpoint_last_user=True, **settings)
        app.scheduler.stop(timeout=5.0)
        out = capsys.readouterr().out
        assert app.persist is None and f"--persist-turns does nothing here: {why}" in out
        assert ("--checkpoint-last-user does nothing here" in out) == (kwargs.get("checkpoint_slots") == 0)
    app = _app(tmp_path, persist_turns=True, checkpoint_last_user=True)
    app.scheduler.stop(timeout=5.0)
    assert "does nothing here" not in capsys.readouterr().out


def _disk_full_after(monkeypatch, room: list[int]):
    """write_encoded / save_snapshot fail with ENOSPC while ``room[0]`` is below the bytes they need."""

    import errno

    real_write, real_save = ps.write_encoded, ps.save_snapshot

    def used(directory):
        return sum(p.stat().st_size for p in directory.glob("*.safetensors"))

    def write(directory, model_id, tokens, payload):
        if used(directory) + len(payload) > room[0]:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(directory, model_id, tokens, payload)

    def save(directory, model_id, tokens, cache, **kwargs):
        if used(directory) + 40_000 > room[0]:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_save(directory, model_id, tokens, cache, **kwargs)

    monkeypatch.setattr(ps, "write_encoded", write)
    monkeypatch.setattr(ps, "save_snapshot", save)
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda d: usage(room[0], used(Path(d)),
                                                              max(0, room[0] - used(Path(d)))))


def test_a_full_disk_makes_room_from_this_models_oldest_conversations(tmp_path, monkeypatch, capsys):
    import os

    writer = TurnWriter(tmp_path, MODEL, limit_bytes=1 << 30)
    one = _payload([1], n=128)
    for tokens in ([1], [2], [3]):
        writer.submit(tokens, _payload(tokens, n=128))
        assert writer.wait_idle(5.0)
    files = sorted(tmp_path.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    for n, path in enumerate(files):
        os.utime(path, (1_000_000 + n, 1_000_000 + n))
    other = ps.save_snapshot(tmp_path, "/models/other|weights=1", [9], [Layer(128)])      # another model's
    room = [sum(p.stat().st_size for p in tmp_path.glob("*.safetensors")) + len(one) // 2]
    _disk_full_after(monkeypatch, room)
    writer.submit([4], _payload([4], n=128))
    assert writer.stop(5.0) and writer.stats()["failed"] == 0 and writer.stats()["written"] == 4
    assert not files[0].exists() and files[1].exists() and other.exists()
    assert "disk full: removed 1 oldest saved conversation(s)" in capsys.readouterr().out


def test_a_disk_with_nothing_of_ours_to_free_fails_the_write_and_counts_it(tmp_path, monkeypatch):
    from tensorfold.server.checkpoints import WRITE_FAILURES, CheckpointEntry, spill_conversation

    _disk_full_after(monkeypatch, [0])
    before = dict(WRITE_FAILURES)
    entry = CheckpointEntry([1, 2], [Layer()], [1, 2], nbytes=10)
    assert spill_conversation(entry, tmp_path, MODEL, limit_bytes=1 << 30) is False
    assert WRITE_FAILURES["spill"] == before["spill"] + 1


def test_health_reports_snapshot_write_failures(tmp_path):
    from tensorfold.server.http import _write_failures

    app = _app(tmp_path, persist_turns=True)
    try:
        failures = _write_failures(app)
        assert set(failures) == {"spill", "shutdown_save", "turn"} and failures["turn"] == 0
    finally:
        app.scheduler.stop(timeout=5.0)


def test_an_explicit_session_dir_keeps_two_servers_with_sibling_snapshot_dirs_apart(tmp_path):
    from tensorfold import cli

    lanes = {}
    for name in ("l", "v"):
        lanes[name] = make_app(engine_factory=DiskEngine, lanes=1, model_id=MODEL, persist_turns=True,
                               snapshot_dir=tmp_path / f"snap-{name}", session_dir=tmp_path / f"sessions-{name}")
    try:
        lanes["l"].chat(TURN1, max_tokens=4)
        assert _written(lanes["l"], 1)
        assert len(list((tmp_path / "sessions-l").glob("*.safetensors"))) == 1
        assert not (tmp_path / "sessions-v").exists() and not (tmp_path / "session-snapshots").exists()
    finally:
        for app in lanes.values():
            app.scheduler.stop(timeout=5.0)
    derived = _app(tmp_path / "d")
    try:
        assert derived.scheduler.session_dir == tmp_path / "d" / "session-snapshots"
    finally:
        derived.scheduler.stop(timeout=5.0)
    assert cli.build_parser().parse_args(["serve", "m"]).session_dir is None
    assert cli.build_parser().parse_args(["serve", "m", "--session-dir", "x"]).session_dir == "x"


def test_health_shows_a_turn_copy_pending_from_the_reply_until_it_is_queued(tmp_path, monkeypatch):
    app = _app(tmp_path, persist_turns=True)
    seen = {}
    real = app.scheduler._persist_turn
    release = threading.Event()

    def slow(job):
        seen["during"] = app.persist.stats()
        release.wait(5.0)
        real(job)

    monkeypatch.setattr(app.scheduler, "_persist_turn", slow)
    try:
        threading.Timer(0.2, release.set).start()
        app.chat(TURN1, max_tokens=4)
        right_after = app.persist.stats()                      # the reply is out, the copy not queued yet
        assert right_after["turn_copy_pending"] == 1 and right_after["idle"] is False
        assert seen["during"]["turn_copy_pending"] == 1
        assert app.persist.wait_idle(5.0) and _written(app, 1)
        assert app.persist.stats()["idle"] is True and app.persist.stats()["turn_copy_pending"] == 0
    finally:
        app.scheduler.stop(timeout=5.0)
