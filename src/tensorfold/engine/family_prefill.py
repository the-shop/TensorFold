"""The family rounds' prefill: prompt chunks, checkpoints at chunk starts, the first token and first drafts."""

from __future__ import annotations

from functools import partial
import os
import time
from typing import Any, Iterator, Sequence

from tensorfold.engine.family_common import cache_arrays, check_expert_stream, drop_spares, stream_note
from tensorfold.engine.prefill_plan import PromptChunks


def layer_major_wanted(env: dict | None = None) -> bool:
    """TF_PREFILL_LAYER_MAJOR=1: a prompt's chunks run in layer-major groups where the model can (off by default)."""

    env = os.environ if env is None else env
    return (env.get("TF_PREFILL_LAYER_MAJOR", "0") or "0").strip().lower() in ("1", "on", "yes", "true")


def group_cap_bytes(env: dict | None = None) -> int:
    """The most a layer-major group may carry between layers: TF_PREFILL_GROUP_GIB (default 1 GiB)."""

    env = os.environ if env is None else env
    return int(float(env.get("TF_PREFILL_GROUP_GIB", "1") or 1) * 2**30)


def drain(steps: Iterator[Any]) -> Any:
    """Run a generator of prefill steps to its end and return its value."""

    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


class FamilyPrefill:
    """Prefill for ``FamilyRounds``."""

    _prefill_at: int | None = None                     # the prompt position the working cache holds whole
    # layer-major prompt groups: None follows TF_PREFILL_LAYER_MAJOR; the model must offer prefill_group
    layer_major: bool | None = None
    prefill_steps = 0                                  # (layer, chunk) calls of layer-major groups: progress too

    def _family_feed(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]],
                     prompt_data: Any = None) -> Any:
        """Absorb the ``chunks`` ([begin, end) of ``tokens``); the last hidden state [1, 1, D], draft head fed too."""

        return drain(self._family_feed_steps(tokens, cache, chunks, prompt_data))

    def _family_feed_steps(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]],
                           prompt_data: Any = None) -> Iterator[None]:
        """``_family_feed`` as steps: it yields between chunks, each chunk's forward, draft absorb and eval whole."""

        import mlx.core as mx

        last = None
        feed = getattr(self.model, "prefill", None) or self.model.hidden
        self._fed_rows = 0
        chunks = list(chunks)
        ahead = getattr(self.model, "prefetch_prompt", None)
        if ahead is not None and chunks:
            ahead(tokens, *chunks[0])
        for n, (begin, end) in enumerate(chunks):
            if n:
                yield                                         # between chunks: the scheduler may run decode rounds
            if ahead is not None and n + 1 < len(chunks):
                ahead(tokens, *chunks[n + 1])                 # its host reads run while this chunk computes
            chunk = [int(t) for t in tokens[begin:end]]
            if self.prefill_guard is not None:
                self.prefill_guard.before_chunk(cache, len(chunk))
            self._prefill_at = None                    # a chunk in flight: the cache holds no prompt prefix whole
            stream_note(self.model, f"chunk {begin} {end}")
            inputs = mx.array([chunk], dtype=mx.uint32)
            hidden = (self.model.prefill_vision(inputs, cache, prompt_data, begin, end)
                      if prompt_data is not None else feed(inputs, cache))
            self._fed_rows = len(chunk)
            self.prefill_chunks += 1
            last = hidden[:, -1:, :]
            drafting = getattr(self.model, "mtp", None) is not None
            if drafting:
                nxt = [int(t) for t in tokens[begin + 1:end + 1]]
                if nxt:
                    self.model.absorb_draft_context(hidden[:, :len(nxt)], mx.array(nxt, dtype=mx.uint32), cache,
                                                    start=0)
            # an earlier chunk is read only through its caches (and a taps head's taps): MLX skips its last layer
            reads_last = n + 1 == len(chunks) or (drafting and getattr(self.model, "draft_reads_hidden", True))
            mx.eval(*((last,) if reads_last else ()), *cache_arrays(cache))
            check_expert_stream(self.model)
            stream_note(self.model, f"chunk-done {begin} {end}")
            self._prefill_at = end
            if self.prefill_guard is not None:
                self.prefill_guard.after_chunk(cache, len(chunk))
        return last

    def _family_start(self, cache: list[Any] | None, cached_tokens: int, chunks: Any) -> tuple[list[Any], int]:
        """The working cache and where its prefill starts: a stored state only at one of the prompt's chunk starts."""

        if cache is None or int(cached_tokens) not in chunks:
            return self.model.make_cache(), 0
        adopt = getattr(self.model, "adopt_cache", None)
        return (adopt(cache) if adopt is not None else cache), int(cached_tokens)

    def _family_prefill(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                        checkpoints_at: Sequence[int]) -> list[Any]:
        return drain(self._family_prefill_steps(stream, cache=cache, cached_tokens=cached_tokens,
                                                checkpoints_at=checkpoints_at))

    def _family_prefill_steps(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                              checkpoints_at: Sequence[int]) -> Iterator[None]:
        """The prompt's prefill, yielding only between its chunks; the last chunk and the first token go together."""

        prompt = stream.prompt_ids
        if not prompt:
            raise ValueError(f"{stream.stream_id}: empty prompt")
        prepared = getattr(stream, "prompt_data", None)
        # an image prompt never resumes, so it needs no cut at message starts: its chunks are the step grid alone
        chunks = self.prompt_chunks(prompt) if prepared is None else PromptChunks(
            None, len(prompt), step=getattr(self.prefill_plan, "step", None) or self.prefill_step)
        work, start = self._family_start(cache, cached_tokens, chunks)
        if prepared is not None:
            if start or cache is not None:
                raise ValueError("image prompts require a fresh cache")
            prepared = self.model.encode_vision(prepared, work)
            checkpoints_at = ()
        cached_tokens = self._prefill_at = start
        stream.history_checkpoints = []
        try:
            if prepared is None and self._groups_chunks():
                hidden = yield from self._family_prefill_grouped(stream, prompt, work, chunks, start, checkpoints_at)
            else:
                fed = False
                for boundary in sorted({chunks.floor(int(b)) for b in checkpoints_at}):
                    if not start < boundary < len(prompt):
                        continue
                    if fed:
                        yield
                    yield from self._family_feed_steps(prompt, work, chunks.between(start, boundary))
                    fed = True
                    if self.prefill_guard is None or self.prefill_guard.allow_checkpoint(work):
                        stream.history_checkpoints.append((list(prompt[:boundary]),
                                                           drop_spares(self.copy_single_cache(work))))
                    start = boundary
                if fed:
                    yield
                hidden = yield from self._family_feed_steps(prompt, work, chunks.between(start, len(prompt)),
                                                            prepared)
        except BaseException:
            at = self._prefill_at                      # stopped between chunks: keep the progress, a taken prefix too
            kept = [len(tokens) for tokens, _ in stream.history_checkpoints]
            if prepared is None and at is not None and at in chunks and at not in kept:
                stream.history_checkpoints.append((list(prompt[:at]), drop_spares(self.copy_single_cache(work))))
            raise
        first = self._family_first(stream, work, hidden, cached_tokens, self._fed_rows - 1)
        first = int(first.item()) if hasattr(first, "item") else int(first)
        check_expert_stream(self.model)
        self._family_commit_first(stream, first)
        return work

    def _groups_chunks(self) -> bool:
        wanted = layer_major_wanted() if self.layer_major is None else bool(self.layer_major)
        return wanted and bool(getattr(self.model, "groups_prompt_chunks", False))

    def _family_prefill_grouped(self, stream: Any, prompt: Sequence[int], work: list[Any], chunks: Any, start: int,
                                checkpoints_at: Sequence[int]) -> Iterator[None]:
        """The prompt's chunks from ``start`` in layer-major groups (the same chunks, calls and bits as one by one);
        a checkpoint inside a group is assembled from each layer's state as the layer passes it. A chunk of at most
        ``fused_rows`` rows (the fused decode path) and a group that would be one chunk run alone. Yields only
        between groups, every call of the group evaluated."""

        cuts = sorted({f for f in (chunks.floor(int(b)) for b in checkpoints_at) if start < f < len(prompt)})
        todo = list(chunks.between(start, len(prompt)))
        hidden = None
        fed = False
        while todo:
            group = self._next_group(todo, work)
            if fed:
                yield
            fed = True
            if len(group) == 1:
                hidden = yield from self._family_feed_steps(prompt, work, group)
            else:
                hidden = self._family_feed_group(stream, prompt, work, group, cuts)
            end = group[-1][1]
            if end in cuts and (self.prefill_guard is None or self.prefill_guard.allow_checkpoint(work)):
                stream.history_checkpoints.append((list(prompt[:end]), drop_spares(self.copy_single_cache(work))))
        return hidden

    def _next_group(self, todo: list[tuple[int, int]], work: list[Any]) -> list[tuple[int, int]]:
        """Take the next group off ``todo``: consecutive chunks of more than ``fused_rows`` rows whose carried
        activations stay under the group cap and fit the memory budget now (halved until they do; one chunk runs
        alone)."""

        small = int(getattr(self.model, "fused_rows", 16))
        if todo[0][1] - todo[0][0] <= small:
            return [todo.pop(0)]
        per = int(self.model.frontier_bytes_per_token)
        cap = group_cap_bytes()
        group, rows = [todo[0]], todo[0][1] - todo[0][0]
        for begin, end in todo[1:]:
            if end - begin <= small or (rows + end - begin) * per > cap:
                break
            group.append((begin, end))
            rows += end - begin
        while len(group) > 1 and self.prefill_guard is not None and not self.prefill_guard.frontier_fits(
                work, rows * per):
            group = group[:(len(group) + 1) // 2]
            rows = sum(end - begin for begin, end in group)
        del todo[:len(group)]
        return group

    def _family_feed_group(self, stream: Any, prompt: Sequence[int], work: list[Any], group: list[tuple[int, int]],
                           cuts: Sequence[int]) -> Any:
        """One layer-major group: its chunks' forwards, the draft head's absorbs in chunk order, and every inner
        checkpoint; the last chunk's hidden state [1, R, D] (evaluated). The working cache holds no prompt prefix
        whole until the group ends: a failure keeps only the checkpoints already assembled."""

        import mlx.core as mx
        import numpy as np

        guard = self.prefill_guard
        model = self.model
        layers = int(model.layer_count)
        rows = sum(end - begin for begin, end in group)
        if guard is not None:
            guard.carry_frontier(rows * int(model.frontier_bytes_per_token))
        try:
            if guard is not None:
                guard.before_chunk(work, group[0][1] - group[0][0])
            self._prefill_at = None
            stream_note(model, f"group {group[0][0]} {group[-1][1]} chunks={len(group)}")
            # inner checkpoints: each layer's state copied as the layer passes the cut, the draft head's after
            marks: dict[int, list[Any]] = {end: [None] * layers for _, end in group[:-1] if end in cuts}
            ahead = getattr(model, "prefetch_prompt", None)
            ple = next((i for i, layer in enumerate(model.layers) if "ple" in layer), None)
            if ahead is not None:
                for begin, end in group[:2]:
                    ahead(prompt, begin, end)

            def before(i: int, j: int) -> None:
                if j == 0 and guard is not None:
                    guard.cancellation.check()           # between calls: the queued ones run to their end

            def after(i: int, j: int) -> None:
                self.prefill_steps += 1
                end = group[j][1]
                if end in marks:
                    marks[end][i] = self.copy_single_cache([work[i]])[0]
                if i == ple and ahead is not None and j + 2 < len(group):
                    ahead(prompt, *group[j + 2])

            inputs = [np.array([[int(t) for t in prompt[begin:end]]], dtype=np.int64) for begin, end in group]
            outs = model.prefill_group(inputs, work, before=before, after=after)
            drafting = getattr(model, "mtp", None) is not None
            last = None
            for j, ((begin, end), (streams, hidden)) in enumerate(zip(group, outs)):
                model.adopt_streams(streams)
                if drafting:
                    nxt = [int(t) for t in prompt[begin + 1:end + 1]]
                    if nxt:
                        model.absorb_draft_context(hidden[:, :len(nxt)], mx.array(nxt, dtype=mx.uint32), work,
                                                   start=0)
                last = hidden[:, -1:]
                reads_last = j + 1 == len(group) or (drafting and getattr(model, "draft_reads_hidden", True))
                mx.eval(*((last,) if reads_last else ()), *cache_arrays(work[layers:]))
                if end in marks:
                    taken = marks.pop(end) + self.copy_single_cache(work[layers:])
                    if guard is None or guard.allow_checkpoint(taken, current=work + taken):
                        stream.history_checkpoints.append((list(prompt[:end]), drop_spares(taken)))
            mx.eval(*cache_arrays(work))
            check_expert_stream(model)
            stream_note(model, f"group-done {group[0][0]} {group[-1][1]}")
            self._fed_rows = group[-1][1] - group[-1][0]
            self.prefill_chunks += len(group)
        finally:
            if guard is not None:
                guard.carry_frontier(0)
        self._prefill_at = group[-1][1]
        if guard is not None:
            guard.after_chunk(work, group[-1][1] - group[-1][0])
        return last

    def _family_first(self, stream: Any, work: list[Any], hidden: Any, cached_tokens: int, row: int) -> Any:
        """Draw or force the first token before the draft head or the next forward reads it."""

        import mlx.core as mx

        prompt_len = len(stream.prompt_ids)
        stream.emitted = []
        stream.pending = []
        stream.cache_len = prompt_len
        stream.cached_tokens = int(cached_tokens)
        stream.started_at = time.perf_counter()
        token = self._draw(self.model.head(hidden), stream.sampling, [prompt_len])
        forced = self._forced_next(stream, token)
        if forced is not None:
            token = mx.array([forced], dtype=mx.uint32)
        if self.family_mtp and stream.drafts:
            # the head reads the prompt's last position and the first token, and drafts the one after it
            firsts = self.model.speculate(work, token, prompt_len - 1, stream.sampling, start=row)
            # the first drafts settle at the next round, so the first token goes out without the draft forward
            self._next[stream.stream_id] = partial(self.model.settle, work, 1, firsts.reshape(-1)[:1],
                                                   prompt_len + 1, stream.sampling, self._depth(stream))
        elif self.pipelined:
            self._queue_next(stream, work, token)
        return token

    @staticmethod
    def _family_commit_first(stream: Any, first: int) -> None:
        stream.commit([first])
        stream.pending = [first]

    def _queue_next(self, stream: Any, cache: list[Any], token: Any) -> None:
        """Feed ``token`` (a GPU array, not read yet) and queue the draw of the one after it."""

        import mlx.core as mx

        hidden = self.model.hidden(token.reshape(1, 1), cache)
        stream.cache_len += 1
        nxt = self._draw(self.model.head(hidden), stream.sampling, [stream.cache_len])
        mx.async_eval(nxt)
        self._inflight[stream.stream_id] = nxt

    def _family_prefill_prefix(self, prompt_ids: Sequence[int], *, cache: list[Any] | None,
                               cached_tokens: int) -> list[Any]:
        if not prompt_ids:
            raise ValueError("empty prefix")
        chunks = self.prompt_chunks(prompt_ids)
        work, start = self._family_start(cache, cached_tokens, chunks)
        self._family_feed(prompt_ids, work, chunks.between(start, len(prompt_ids)))
        return drop_spares(work)

    def _family_add_stream(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                           checkpoints_at: Sequence[int]) -> Iterator[None]:
        """Prefill a stream a chunk a step; after the last chunk it takes part in the rounds."""

        work = yield from self._family_prefill_steps(stream, cache=cache, cached_tokens=cached_tokens,
                                                     checkpoints_at=checkpoints_at)
        self.streams.append(stream)
        if stream.finished:
            stream.finished_at = time.perf_counter()
            self._release_stream_state(stream.stream_id)
            return
        self._live.append((stream, work))
