"""Sample drafts over listed token ids with the target's keyed rule; committed tokens still use its full vocabulary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

VOCAB_FILE = Path(__file__).with_name("cuda") / "draft_vocab.txt"


def draft_ids(path: Path = VOCAB_FILE, multiple: int = 64, vocab: int = 0) -> np.ndarray:
    """Sort ids to break ties by token id, padding with the smallest unlisted ids to a multiple of ``multiple``;
    ids at or past ``vocab`` (0: no limit) are left out: a draft is embedded next, and the gather has no bound check."""

    listed = {int(line) for line in Path(path).read_text().split() if line.strip()}
    if vocab:
        listed = {t for t in listed if 0 <= t < vocab}
    if not listed:
        raise ValueError(f"{path}: no token ids" + (f" below {vocab}" if vocab else ""))
    extra, candidate = [], 0
    while (len(listed) + len(extra)) % multiple:
        if vocab and candidate >= vocab:
            raise ValueError(f"{path}: {vocab} token ids can't fill a multiple of {multiple}")
        if candidate not in listed:
            extra.append(candidate)
        candidate += 1
    return np.array(sorted(listed | set(extra)), dtype=np.uint32)


def draft_head_for(lm_head: Any, vocab: int, path: Path = VOCAB_FILE) -> tuple[Any, Any]:
    """The listed draft ids the model can embed and score (below ``vocab`` and the head's rows) and their cut head;
    (None, None) when no such subset fills a multiple of 64 rows: drafts then score the whole head."""

    try:
        ids = draft_ids(path, vocab=min(int(vocab), int(lm_head.weight.shape[0])))
    except ValueError:
        return None, None
    return ids, cut_head(lm_head, ids)


def cut_head(lm_head: Any, ids: np.ndarray) -> nn.QuantizedLinear:
    """The rows of a 4-bit quantized vocabulary head for ``ids`` as a quantized linear of their own."""

    index = mx.array(ids.astype(np.int32))
    out = nn.QuantizedLinear(int(lm_head.weight.shape[1]) * 32 // lm_head.bits, len(ids), bias=False,
                             group_size=lm_head.group_size, bits=lm_head.bits)
    out.weight = mx.take(lm_head.weight, index, axis=0)
    out.scales = mx.take(lm_head.scales, index, axis=0)
    out.biases = mx.take(lm_head.biases, index, axis=0)
    mx.eval(out.weight, out.scales, out.biases)
    return out


def sample(logits: mx.array, ids: mx.array, sampling: Any, positions: Sequence[int] | mx.array) -> mx.array:
    """Return lazy uint32 token ids [R] using keyed sampling over ``ids``, or greedy selection when ``sampling`` is None."""

    from tensorfold.engine.gpu_sampling import sample as gpu_sample

    return gpu_sample(logits.reshape(-1, logits.shape[-1]), sampling, positions, ids=ids)
