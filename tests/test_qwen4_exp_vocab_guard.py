"""Token ids past the embedding are refused (the GPU gathers would read past the table), in-range ids are untouched."""

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.model import check_ids


def test_check_ids_bounds():
    check_ids(np.array([[0, 5, 96]]), 97)
    check_ids(np.zeros((1, 0), dtype=np.int64), 97)
    for bad in (97, 248045, -1):
        with pytest.raises(ValueError, match=str(bad)):
            check_ids(np.array([[3, bad, 4]]), 97)


@pytest.mark.parametrize("rows", [1, 8, 300])
def test_hidden_refuses_out_of_vocab_ids(rows):
    import test_qwen4_exp_prefill as T

    model = T.tiny_model()
    cache = model.make_cache()
    tokens = np.random.default_rng(0).integers(6, 97, size=(1, rows))
    tokens[0, rows // 2] = 97 + 1000
    with pytest.raises(ValueError, match="outside this model's vocabulary of 97"):
        model.hidden(tokens, cache)
