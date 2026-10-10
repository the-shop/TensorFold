import pytest

from tensorfold.server.app import CheckpointStore


@pytest.mark.parametrize("pinned", [False, True])
def test_an_oversized_checkpoint_cannot_exceed_the_store_budget(pinned):
    store = CheckpointStore(2, copier=lambda c: c, budget_bytes=1000, sizer=lambda c: c[0])
    store.insert([1, 2], [1200], last_prompt=[1, 2], pinned=pinned)
    assert store.nbytes <= 1000
    assert len(store) == 0


def test_rejecting_an_oversized_checkpoint_preserves_a_usable_prefix():
    store = CheckpointStore(2, copier=lambda c: c, budget_bytes=1000, sizer=lambda c: c[0])
    store.insert([1], [800], last_prompt=[1])
    store.insert([2], [1200], last_prompt=[2])
    assert store.match([1, 3]) == (1, [800], [1])
    assert store.nbytes == 800


def _order(store):
    return [entry.cache[0] for entry in store._entries]


def test_a_conversation_that_moved_on_loses_its_older_checkpoint_before_another_conversations_newest():
    store = CheckpointStore(3, copier=lambda c: c)
    store.insert([5, 6], ["b"], last_prompt=[5, 6, 7])                  # conversation b, the least recently used
    store.insert([1, 2], ["a1"], last_prompt=[1, 2, 3])
    store.insert([1, 2, 3, 4], ["a2"], last_prompt=[1, 2, 3, 4, 5])     # a's next turn continues a1
    store.insert([9], ["c"], last_prompt=[9, 9])                         # over the three slots
    assert _order(store) == ["c", "a2", "b"]
    assert store.evict_one() and _order(store) == ["c", "a2"]           # then the least recently used
    store.insert([1, 2, 3, 4, 5, 6], ["a3"], last_prompt=[1, 2, 3, 4, 5, 6, 7])
    assert store.evict_one() and _order(store) == ["a3", "c"]           # memory pressure: a2 first, c stays


def test_one_turns_checkpoints_are_all_its_newest():
    store = CheckpointStore(3, copier=lambda c: c)
    store.insert([5, 6], ["b"], last_prompt=[5, 6, 7])
    store.insert([1, 2], ["stable"], last_prompt=[1, 2, 3, 4])           # a turn's stable prefix and its history
    store.insert([1, 2, 3], ["history"], last_prompt=[1, 2, 3, 4])
    store.insert([9], ["c"], last_prompt=[9, 9])
    assert _order(store) == ["c", "history", "stable"]                   # the next turn may diverge before 3


def test_a_reply_end_the_next_turn_rendered_differently_goes_before_a_superseded_prefix():
    store = CheckpointStore(3, copier=lambda c: c)
    store.insert([1, 2, 3], ["history0"], last_prompt=[1, 2, 3, 8, 9])            # turn 0's history checkpoint
    store.insert([1, 2, 3, 8, 9, 50, 51], ["reply0"], last_prompt=[1, 2, 3, 8, 9])  # its reply end
    # turn 1 re-renders the reply (52 for 50): the reply end is dead, the history checkpoint only superseded
    store.insert([1, 2, 3, 8, 52, 51, 4], ["history1"], last_prompt=[1, 2, 3, 8, 52, 51, 4, 8, 9])
    store.insert([7], ["other"], last_prompt=[7, 7])                               # over the three slots
    assert _order(store) == ["other", "history1", "history0"]


def test_another_conversations_reply_end_is_not_dead():
    store = CheckpointStore(2, copier=lambda c: c)
    store.insert([5, 6, 7, 40, 41], ["reply-b"], last_prompt=[5, 6, 7])
    store.insert([1, 2, 3, 4], ["a"], last_prompt=[1, 2, 3, 4, 5])
    store.insert([1, 2, 3, 4, 6], ["a2"], last_prompt=[1, 2, 3, 4, 6, 7])
    assert _order(store) == ["a2", "reply-b"]            # a's superseded prefix went, b's reply end stayed
