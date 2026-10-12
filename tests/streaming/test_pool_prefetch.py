"""Decode prefetch and prompt seeding in the expert pool: guesses, eviction safety, stale guesses, seeds, counters."""

from __future__ import annotations

import struct
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from test_pool import EXPERTS, LAYERS, TOPK, FakeSync, checkpoint, slot_bytes

from tensorfold.streaming import pool


def make(tmp_path, slots, workers=8, **flags):
    found, shapes = checkpoint(tmp_path)
    fake = FakeSync()
    s = pool.Streamer(found, shapes, layers=LAYERS, experts=EXPERTS, top_k=TOPK, slots=slots, box_rows=4,
                      hostsync=fake, workers=workers, **flags)
    return s, fake


def call(s, fake, kind, layer, ids, guesses=()):
    """One MoE call as the GPU makes it: routing (ids, or per-expert counts for "window") and the next layer's
    guesses (one row's k ids after another) in their boxes, then signal and wait for the host's release."""

    words = list(ids)
    if kind == "window":
        words = [0] * EXPERTS
        for e in ids:
            words[e] += 1
    fake.write_into(s.boxes, 4 * s.region_word(s.value + 1), struct.pack(f"<{len(words)}I", *words))
    rows = len(ids) // TOPK if kind == "rows" else 1
    k = len(guesses) // rows if guesses else 0
    if guesses:
        fake.write_into(s.guess_box, 0, struct.pack(f"<{len(guesses)}I", *guesses))
    signal, ready = s.marks(kind, layer, rows, k)
    s.channel.signal(signal)
    assert s.channel.wait(ready, 5000) and s.error is None


def settle(s, check, timeout=5.0):
    """Wait for the host thread's reads behind a call (they start after the release)."""

    end = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < end, "speculative reads never started"
        time.sleep(0.005)
    for entry in list(s.pending.values()):
        for read in entry.reads:
            if not read.cancelled():
                read.result(timeout)


def table(s, fake):
    return np.frombuffer(fake.buf(s.slot_of), dtype=np.int32).reshape(LAYERS, EXPERTS)


def resident_bytes_ok(s, fake, key):
    layer, e = key
    return (slot_bytes(s, fake, s.lru[key]) == layer * 16 + e).all()


# -- flags off -------------------------------------------------------------------------------------------------

class OldLRU:
    """The pool's decode LRU before prefetch and seeding, as the reference the flags-off path must equal."""

    def __init__(self, slots):
        self.lru, self.free = OrderedDict(), list(range(slots - 1, EXPERTS - 1, -1))
        self.hits = self.misses = 0
        self.table = {}

    def load(self, layer, ids):
        for e in ids:
            slot = self.lru.pop((layer, e), None)
            if slot is None:
                if self.free:
                    slot = self.free.pop()
                else:
                    keep = {(layer, i) for i in ids}
                    victim = next(k for k in self.lru if k not in keep)
                    slot = self.lru.pop(victim)
                self.misses += 1
            else:
                self.hits += 1
            self.lru[(layer, e)] = slot
            self.table[(layer, e)] = slot


SEQUENCE = [("rows", 0, [1, 2, 3, 4]), ("window", 1, [0, 5]), ("rows", 1, [5, 6, 7, 0]), ("rows", 2, [1, 1, 3, 3]),
            ("rows", 0, [2, 7, 4, 6]), ("rows", 2, [0, 4, 5, 6]), ("window", 0, [2]), ("rows", 1, [1, 2, 3, 4])]


@pytest.mark.parametrize("flags", [{}, {"decode_prefetch": 0, "prompt_seed": 0}])
def test_flags_off_is_the_old_lru_slot_for_slot(tmp_path, flags):
    s, fake = make(tmp_path, slots=2 * EXPERTS + 3, **flags)
    ref = OldLRU(2 * EXPERTS + 3)
    for kind, layer, ids in SEQUENCE:
        call(s, fake, kind, layer, ids)
        if kind == "rows":
            ref.load(layer, sorted(set(ids)))
    assert list(s.lru.items()) == list(ref.lru.items())
    assert s.free == ref.free and (s.hits, s.misses) == (ref.hits, ref.misses)
    for (layer, e), slot in ref.table.items():
        assert table(s, fake)[layer, e] == slot
    assert s.stats() == {"hits": ref.hits, "misses": ref.misses, "bytes_read": s.bytes_read,
                         "window_reads": s.window_reads, "window_copies": s.window_copies,
                         "copy_bytes": s.copy_bytes}
    assert (s.window_reads, s.window_copies) == (2, 1)                  # (0, 2) was resident: copied
    assert s.guess_box is None and s.prompt_counts is None and s._seeders is None
    assert not s.pending and not s.speculative
    s.close()


def test_prefetch_on_without_guesses_changes_nothing(tmp_path):
    a, fa = make(tmp_path, slots=2 * EXPERTS + 3)
    b, fb = make(tmp_path, slots=2 * EXPERTS + 3, decode_prefetch=4)
    for kind, layer, ids in SEQUENCE:
        call(a, fa, kind, layer, ids)
        call(b, fb, kind, layer, ids)
    assert list(a.lru.items()) == list(b.lru.items()) and (a.hits, a.misses) == (b.hits, b.misses)
    assert (fa.buf(a.slot_of) == fb.buf(b.slot_of)).all()
    assert (fa.buf(a.pool["up"].weight) == fb.buf(b.pool["up"].weight)).all()
    a.close()
    b.close()


# -- prediction ------------------------------------------------------------------------------------------------

def test_guesses_for_the_next_layer_are_read_and_counted_when_routed(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, decode_prefetch=2)
    call(s, fake, "rows", 0, [1, 2, 1, 2], guesses=[5, 6, 6, 7])        # two rows: guesses {5, 6, 7} for layer 1
    settle(s, lambda: s.prefetch_issued == 3)
    for e in (5, 6, 7):
        assert (1, e) in s.lru and resident_bytes_ok(s, fake, (1, e))
    misses = s.misses
    call(s, fake, "rows", 1, [5, 6, 0, 6])
    assert s.prefetch_hits == 2 and s.misses == misses + 1               # only expert 0 was a demand read
    for e in (5, 6, 0):
        assert (slot_bytes(s, fake, table(s, fake)[1, e]) == 16 + e).all()
    assert s.speculative == {(1, 7): "prefetch"}                         # read, unused, still resident
    call(s, fake, "rows", 2, [0, 1, 0, 1], guesses=[3, 4, 3, 4])        # the last layer guesses nothing
    time.sleep(0.05)
    assert s.prefetch_issued == 3
    s.close()


def test_a_guess_already_resident_is_not_read_again(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, decode_prefetch=2)
    call(s, fake, "rows", 1, [3, 4, 3, 4])
    before = s.bytes_read
    call(s, fake, "rows", 0, [1, 2, 1, 2], guesses=[3, 4, 3, 5])
    settle(s, lambda: s.prefetch_issued == 1)
    assert set(s.speculative) == {(1, 5)} and s.bytes_read - before == (2 + 1) * pool.expert_nbytes(s.found, 0)  # 2 demand, 1 guess
    s.close()


# -- eviction safety -------------------------------------------------------------------------------------------

def test_guesses_never_evict_the_current_calls_experts(tmp_path):
    s, fake = make(tmp_path, slots=2 * EXPERTS + 2, decode_prefetch=4)  # 10 LRU slots
    call(s, fake, "rows", 2, [0, 1, 0, 1])                               # the two coldest, not in the next call
    current = [0, 1, 2, 3, 4, 5, 6, 7]
    call(s, fake, "rows", 0, current)                                    # 8 more: the LRU is full
    slots_before = {k: s.lru[k] for k in s.lru if k[0] == 0}
    call(s, fake, "rows", 0, current, guesses=[0, 1, 2, 3, 4, 5, 6, 7])  # 4 rows x 2: more than the LRU spares
    settle(s, lambda: s.prefetch_issued == 2)
    time.sleep(0.05)
    assert s.prefetch_issued == 2                                       # stopped at the current call's experts
    assert (2, 0) not in s.lru and (2, 1) not in s.lru
    assert {k: s.lru[k] for k in s.lru if k[0] == 0} == slots_before
    for key in slots_before:
        assert resident_bytes_ok(s, fake, key)
    assert len(set(s.lru.values())) == len(s.lru) == 10
    s.close()


def test_demand_reads_never_evict_a_read_in_flight(tmp_path):
    s, fake = make(tmp_path, slots=2 * EXPERTS + 2, prompt_seed=1)
    gate = threading.Event()
    real = s._read

    def gated(layer, proj, part, expert, slot):
        if (layer, expert) == (1, 4):
            gate.wait(5)
        real(layer, proj, part, expert, slot)

    call(s, fake, "window", 1, [4, 4, 4])
    s._read = gated
    call(s, fake, "rows", 0, [0, 1, 0, 1])                               # seeds (1, 4) behind this call
    end = time.monotonic() + 5
    while s.seeded < 1:
        assert time.monotonic() < end
        time.sleep(0.005)
    slot = s.lru[(1, 4)]
    call(s, fake, "rows", 2, [0, 1, 2, 3, 4, 5, 6, 7])                   # the LRU fills: the seed stays put
    assert s.lru[(1, 4)] == slot and (1, 4) in s.pending
    gate.set()
    for read in s.pending[(1, 4)].reads:                                 # done: a seed hit (else a demand read)
        read.result(5)
    call(s, fake, "rows", 1, [4, 4, 4, 4])
    assert s.seed_hits == 1 and (slot_bytes(s, fake, table(s, fake)[1, 4]) == 16 + 4).all()
    s.close()


def test_a_resident_guess_is_never_evicted_by_another_guess(tmp_path):
    s, fake = make(tmp_path, slots=2 * EXPERTS + 2, decode_prefetch=2)  # 10 LRU slots
    call(s, fake, "rows", 1, [3, 3])                                     # (1, 3): the coldest
    call(s, fake, "rows", 2, [0, 1, 2, 3, 4, 5, 6, 7])
    call(s, fake, "rows", 0, [1, 1])                                     # the LRU is full
    assert not s.free and next(iter(s.lru)) == (1, 3)
    slot = s.lru[(1, 3)]
    call(s, fake, "rows", 0, [1, 1], guesses=[3, 5])                     # (1, 3) resident, (1, 5) not
    settle(s, lambda: s.prefetch_issued == 1)
    assert s.lru[(1, 3)] == slot and resident_bytes_ok(s, fake, (1, 3))
    assert (2, 0) not in s.lru and (1, 5) in s.lru and resident_bytes_ok(s, fake, (1, 5))
    call(s, fake, "rows", 1, [3, 5])
    assert s.prefetch_hits == 1 and s.misses == 1 + 8 + 1               # (1, 3) a plain hit, (1, 5) a guess
    s.close()


def test_a_routed_seed_still_queued_is_read_on_demand_not_behind_the_seeds(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, prompt_seed=3)
    gate = threading.Event()
    real = s._read

    def gated(layer, proj, part, expert, slot):
        if (layer, expert) in ((1, 6), (1, 4)):                          # both seed threads stuck on these
            gate.wait(10)
        real(layer, proj, part, expert, slot)

    call(s, fake, "window", 1, [2] * 3 + [4] * 2 + [6])                  # seeds read least routed first: 6, 4, 2
    s._read = gated
    try:
        call(s, fake, "rows", 0, [0, 0])
        end = time.monotonic() + 5
        while s.seeded < 3:
            assert time.monotonic() < end
            time.sleep(0.005)
        misses = s.misses
        started = time.monotonic()
        call(s, fake, "rows", 1, [2, 2])                                 # (1, 2) queued behind the stuck seeds
        assert time.monotonic() - started < 2 and not gate.is_set()
        assert s.misses == misses + 1 and s.seed_hits == 0 and (1, 2) not in s.pending
        assert (slot_bytes(s, fake, table(s, fake)[1, 2]) == 16 + 2).all()
        assert len(set(s.lru.values())) == len(s.lru)
    finally:
        gate.set()
        s.close()


def test_a_seed_still_reading_is_not_copied_into_the_window(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, prompt_seed=1)
    gate = threading.Event()
    real = s._read

    def gated(layer, proj, part, expert, slot):
        if (layer, expert) == (1, 6) and slot >= EXPERTS:                # only the seed's read into its LRU slot
            gate.wait(10)
        real(layer, proj, part, expert, slot)

    call(s, fake, "window", 1, [6] * 3)
    s._read = gated
    try:
        call(s, fake, "rows", 0, [0, 0])                                 # seeds (1, 6) behind this call
        end = time.monotonic() + 5
        while s.seeded < 1:
            assert time.monotonic() < end
            time.sleep(0.005)
        call(s, fake, "window", 0, [6])                                  # window slot 6 now holds layer 0's
        assert (1, 6) in s.lru and (1, 6) in s.pending
        reads, copies = s.window_reads, s.window_copies
        call(s, fake, "window", 1, [6])
        assert (s.window_reads - reads, s.window_copies - copies) == (1, 0)
        assert (slot_bytes(s, fake, 6) == 16 + 6).all()
    finally:
        gate.set()
        s.close()


# -- stale guesses ---------------------------------------------------------------------------------------------

def test_stale_guesses_are_cancelled_and_partial_slots_freed(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, workers=1, decode_prefetch=2)
    gate, started = threading.Event(), threading.Event()
    real = s._read

    def gated(layer, proj, part, expert, slot):
        if (layer, expert) == (1, 3):
            started.set()
            gate.wait(5)
        real(layer, proj, part, expert, slot)

    s._read = gated
    call(s, fake, "rows", 0, [1, 2, 1, 2], guesses=[3, 4, 3, 4])
    assert started.wait(5)                                               # (1, 3) mid-read, the rest queued
    end = time.monotonic() + 5
    while s.prefetch_issued < 2:                                         # (1, 4) is issued after (1, 3) starts
        assert time.monotonic() < end
        time.sleep(0.005)
    guessed = {s.lru[(1, 3)], s.lru[(1, 4)]}
    threading.Timer(0.2, gate.set).start()
    call(s, fake, "rows", 1, [0, 5, 0, 5])                               # routed elsewhere: both guesses stale
    assert (s.prefetch_cancelled, s.prefetch_misses, s.prefetch_hits) == (2, 2, 0)
    assert (1, 3) not in s.lru and (1, 4) not in s.lru and not s.pending and not s.speculative
    assert {table(s, fake)[1, 0], table(s, fake)[1, 5]} == guessed      # freed slots are reused first
    for e in (0, 5):
        assert (slot_bytes(s, fake, table(s, fake)[1, e]) == 16 + e).all()
    assert len(set(s.lru.values())) == len(s.lru)
    s.close()


def test_a_window_call_drops_every_pending_guess(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, workers=1, decode_prefetch=2)
    gate = threading.Event()
    real = s._read

    def gated(layer, proj, part, expert, slot):
        gate.wait(5)
        real(layer, proj, part, expert, slot)

    call(s, fake, "rows", 0, [1, 2, 1, 2])                               # demand reads, ungated
    s._read = gated
    call(s, fake, "rows", 0, [1, 2, 1, 2], guesses=[6, 7, 6, 7])
    settle_start = time.monotonic()
    while s.prefetch_issued < 2:
        assert time.monotonic() - settle_start < 5
        time.sleep(0.005)
    threading.Timer(0.1, gate.set).start()
    call(s, fake, "window", 1, [6, 7])                                   # a new prompt: the guesses are stale
    assert not s.pending and s.prefetch_cancelled == 2 and s.prefetch_hits == 0
    assert (1, 6) not in s.lru and (1, 7) not in s.lru and not s.speculative
    s.close()


def test_a_fully_read_unrouted_guess_stays_and_counts_when_used_later(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, decode_prefetch=1)
    call(s, fake, "rows", 0, [1, 2], guesses=[7])
    settle(s, lambda: s.prefetch_issued == 1)
    call(s, fake, "rows", 1, [0, 3])
    assert (1, 7) in s.lru and s.prefetch_misses == 0 and s.prefetch_cancelled == 0
    call(s, fake, "rows", 1, [7, 3])
    assert s.prefetch_hits == 1
    s.close()


def test_an_unused_guess_evicted_counts_as_a_miss(tmp_path):
    s, fake = make(tmp_path, slots=2 * EXPERTS + 2, decode_prefetch=1)
    call(s, fake, "rows", 0, [1, 2], guesses=[7])
    settle(s, lambda: s.prefetch_issued == 1)
    call(s, fake, "rows", 1, [0, 3])
    call(s, fake, "rows", 2, [0, 1, 2, 3, 4, 5, 6, 7])                   # 5 free slots, then the 3 oldest go
    assert (1, 7) not in s.lru and s.prefetch_misses == 1 and s.prefetch_hits == 0
    s.close()


# -- prompt seeding --------------------------------------------------------------------------------------------

def test_seeds_rank_each_layers_prompt_routing_and_end_warmest(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, prompt_seed=2)
    call(s, fake, "window", 0, [3] * 5 + [1] * 4 + [7])                  # layer 0: 3 > 1 > 7
    call(s, fake, "window", 1, [2] * 9)
    call(s, fake, "window", 0, [1])                                      # a second chunk: 1 ties 3 at 5
    assert s.prompt_counts[0, 1] == 5 and s.prompt_counts[0, 3] == 5 and s.prompt_counts[1, 2] == 9
    call(s, fake, "rows", 0, [6, 6])                                     # decode starts: seed behind this call
    settle(s, lambda: s.seeded == 3)
    assert list(s.lru)[-4:] == [(0, 6), (0, 3), (0, 1), (1, 2)]          # ties by expert id; most routed newest
    for key in ((0, 1), (0, 3), (1, 2)):
        assert resident_bytes_ok(s, fake, key)
    assert not s.prompt_counts.any() and s.stats()["seeded"] == 3
    call(s, fake, "rows", 0, [3, 6])
    assert s.seed_hits == 1 and s.stats()["seed_hits"] == 1
    call(s, fake, "rows", 1, [0, 1])
    time.sleep(0.05)
    assert s.seeded == 3                                                 # no prompt since: nothing more to seed
    s.close()


def test_seeds_stay_within_capacity_and_keep_the_current_call(tmp_path):
    s, fake = make(tmp_path, slots=2 * EXPERTS + 2, prompt_seed=EXPERTS)  # 10 LRU slots
    for layer in range(LAYERS):
        call(s, fake, "window", layer, list(range(EXPERTS)) + [layer])
    call(s, fake, "rows", 2, [5, 6, 5, 6])
    settle(s, lambda: s.seeded == 10 - 2)
    time.sleep(0.05)
    assert s.seeded == 10 - 2 and len(s.lru) == 10 and not s.free
    assert (2, 5) in s.lru and (2, 6) in s.lru
    assert len(set(s.lru.values())) == 10
    for key in s.lru:
        assert resident_bytes_ok(s, fake, key)
    s.close()


def test_resident_seeds_are_only_warmed(tmp_path):
    s, fake = make(tmp_path, slots=3 * EXPERTS, prompt_seed=1)
    call(s, fake, "rows", 0, [4, 5])
    call(s, fake, "rows", 1, [1, 2])
    call(s, fake, "window", 0, [4, 4, 5])
    before = s.bytes_read
    call(s, fake, "rows", 1, [1, 2])
    time.sleep(0.05)
    assert s.seeded == 0 and s.bytes_read == before and list(s.lru)[-1] == (0, 4)
    s.close()


# -- counters and flags ----------------------------------------------------------------------------------------

def test_health_reports_the_new_counters_only_when_on(tmp_path):
    from tensorfold.server.http import _expert_stats

    off, _ = make(tmp_path, slots=3 * EXPERTS)
    on, _ = make(tmp_path, slots=3 * EXPERTS, decode_prefetch=2, prompt_seed=4)
    window = {"window_reads", "window_copies", "copy_bytes"}
    for s, keys in ((off, {"hits", "misses", "bytes_read", *window}),
                    (on, {"hits", "misses", "bytes_read", *window, "prefetch_issued", "prefetch_hits", "prefetch_misses",
                          "prefetch_cancelled", "seeded", "seed_hits"})):
        inner = SimpleNamespace()
        inner.__dict__["streamer"] = s
        stats = _expert_stats(SimpleNamespace(_model=SimpleNamespace(model=inner)))
        assert set(stats) == keys and all(v == 0 for v in stats.values())
        s.close()


@pytest.mark.parametrize("env,want", [({}, (0, 0)), ({"TENSORFOLD_POOL_DECODE_PREFETCH": "4"}, (4, 0)),
                                      ({"TENSORFOLD_POOL_DECODE_PREFETCH": "99"}, (16, 0)),
                                      ({"TENSORFOLD_POOL_DECODE_PREFETCH": "-3"}, (0, 0)),
                                      ({"TENSORFOLD_POOL_PROMPT_SEED": "1"}, (0, 32)),
                                      ({"TENSORFOLD_POOL_PROMPT_SEED": "0",
                                        "TENSORFOLD_POOL_PROMPT_SEED_PER_LAYER": "8"}, (0, 0)),
                                      ({"TENSORFOLD_POOL_PROMPT_SEED": "1",
                                        "TENSORFOLD_POOL_PROMPT_SEED_PER_LAYER": "8"}, (0, 8))])
def test_pool_flags(env, want):
    from tensorfold.families.qwen4_exp.stream import pool_flags

    assert pool_flags(env) == want


# -- the GPU side of moe_rows ----------------------------------------------------------------------------------

def test_moe_rows_sends_guesses_only_when_prefetch_is_on(monkeypatch):
    from tensorfold.families.qwen4_exp import stream
    from tensorfold.kernels.qwen.flash_next.v1 import stream_experts

    routes, holds = [], []
    monkeypatch.setattr(stream_experts, "route", lambda logits, k, box: routes.append((k, box)) or mx.array([1]))
    monkeypatch.setattr(stream_experts, "gateup", lambda *a: (mx.zeros((1,)), mx.zeros((1,)), mx.zeros((1,))))
    monkeypatch.setattr(stream_experts, "down", lambda *a: mx.zeros((1,)))

    def fake(prefetch):
        s = SimpleNamespace(box="box", guess_box="guess", decode_prefetch=prefetch, slot_of=mx.zeros((1,)),
                            layer_ids=[mx.zeros((1,))], experts=8,
                            pool={"gate": None, "up": None, "down": None})
        s.hold = lambda token, kind, layer, rows, guess=0: holds.append(guess) or token
        moe = SimpleNamespace(top_k=2)
        moe.__dict__.update(streamer=s, stream_layer=0)
        return moe

    x, logits = mx.zeros((1, 4)), mx.zeros((1, 9))
    stream.moe_rows(fake(0), x, logits, (None, None, None), guess_logits=logits)
    assert routes == [(2, "box")] and holds == [0]
    routes.clear(), holds.clear()
    stream.moe_rows(fake(3), x, logits, (None, None, None))
    assert routes == [(2, "box")] and holds == [0]
    routes.clear(), holds.clear()
    stream.moe_rows(fake(3), x, logits, (None, None, None), guess_logits=logits)
    assert routes == [(2, "box"), (3, "guess")] and holds == [3]


# -- the kernels (tiny, Metal only) ----------------------------------------------------------------------------

def _metal():
    try:
        return mx.metal.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _metal(), reason="needs a Metal GPU")
def test_route_writes_each_rows_top_guesses_into_the_guess_box():
    from tensorfold.kernels.qwen.flash_next.v1 import stream_experts

    rng = np.random.default_rng(3)
    ne, k, rows = 64, 4, 3
    logits = mx.array(rng.normal(0, 1, (rows, ne + 1)).astype(np.float32))
    box = mx.zeros((rows * k + 8,), dtype=mx.uint32)
    mx.eval(box)
    mx.eval(stream_experts.route(logits, k, box))
    got = np.array(box)[:rows * k].reshape(rows, k)
    want = np.argsort(-np.array(logits)[:, :ne], axis=1, kind="stable")[:, :k]
    assert (got == want).all()


@pytest.mark.skipif(not _metal(), reason="needs a Metal GPU")
def test_the_count_kernel_counts_and_the_present_kernel_flags():
    from tensorfold.families.qwen4_exp import stream
    from tensorfold.kernels.qwen.flash_next.v1.base import kernel

    ne = 16
    ids = mx.array([3, 3, 5, 0, 3, 15, 5], dtype=mx.uint32)
    for name, source, want in (("q4_stream_count", stream._COUNT, {0: 1, 3: 3, 5: 2, 15: 1}),
                               ("q4_stream_present", stream._PRESENT, {0: 1, 3: 1, 5: 1, 15: 1})):
        box = mx.full((ne,), 7, dtype=mx.uint32)
        mx.eval(box)
        run = kernel(name, source, ["IDS", "BOX"], ["OUT"])
        mx.eval(run(inputs=[ids, box], template=[("NE", ne), ("N", int(ids.size))], grid=(1024, 1, 1),
                    threadgroup=(1024, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0])
        assert np.array(box).tolist() == [want.get(e, 0) for e in range(ne)]
