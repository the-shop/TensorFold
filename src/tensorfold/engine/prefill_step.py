"""The largest prompt chunk a family offers whose working memory still leaves room for a long context."""

from __future__ import annotations

from typing import Any, Sequence

# tokens of context a larger chunk must leave room for (or the model's window, if smaller)
CONTEXT_FLOOR = 131072


def synthetic_probe(count: int, vocab: int = 0) -> list[int]:
    """Probe ids when no real text is at hand: 1000, 1001, ... (wrapped into ``vocab``, 0: no limit, so a small
    embedding is never read past its table)."""

    return [(1000 + i) % vocab if vocab else 1000 + i for i in range(count)]


def choose(make_engine: Any, steps: Sequence[int], budget: int, tokens: Sequence[int], window: int = 0,
           vocab: int = 0) -> int:
    """Measure one chunk on the smallest step, then take the largest step whose chunk and the context floor fit;
    ``vocab`` (0: unknown) keeps the synthetic probe inside the model's embedding."""

    import mlx.core as mx

    from tensorfold.engine.family_common import cache_arrays
    from tensorfold.server.memory_budget import CacheMemory

    steps = sorted({int(s) for s in steps}, reverse=True)
    small = steps[-1]
    if len(steps) == 1:
        return small
    engine = make_engine(small)
    text = [int(t) for t in tokens] or synthetic_probe(small + 64, vocab)
    probe = (text * (-(-(small + 64) // len(text))))[:small + 64]
    tighten = getattr(engine.model, "tighten_prefill", None)
    while True:
        mx.synchronize()
        mx.clear_cache()
        held = int(mx.get_active_memory())
        mx.reset_peak_memory()
        cache = engine.prefill_prefix(probe, cache=None, cached_tokens=0)
        mx.eval(*cache_arrays(cache))
        work = max(0, int(mx.get_peak_memory()) - int(mx.get_active_memory()))
        per_token = CacheMemory.from_cache(cache).bytes_per_token
        del cache
        getattr(engine, "release_rounds", lambda: None)()
        mx.clear_cache()
        context = min(int(window) or CONTEXT_FLOOR, CONTEXT_FLOOR) * per_token
        for step in steps:
            if held + work * step // small + context <= int(budget):
                return step
        if tighten is None or not tighten():      # nothing fits: a model that can, keeps less of a chunk in flight
            return small


__all__ = ["CONTEXT_FLOOR", "choose", "synthetic_probe"]
