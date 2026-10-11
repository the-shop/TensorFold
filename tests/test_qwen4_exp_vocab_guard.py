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


def test_mtp_draft_ids_stay_inside_the_embedding():
    """MTP drafts come from the listed ids and are embedded next: none may reach past the model's table."""

    from tensorfold.families.qwen4_exp.draft_head import draft_ids

    full = draft_ids()
    assert int(full.max()) >= 65536                           # the list reaches past a small test model's table
    np.testing.assert_array_equal(draft_ids(vocab=248320), full)   # the real checkpoint's 248,320: the same ids
    small = draft_ids(vocab=65536)
    assert int(small.max()) < 65536 and len(small) % 64 == 0
    assert set(full[full < 65536].tolist()) <= set(small.tolist())


def test_draft_head_for_cuts_rows_the_head_has(tmp_path):
    import mlx.core as mx
    from mlx import nn

    from tensorfold.families.qwen4_exp.draft_head import draft_head_for, draft_ids

    mx.random.seed(0)
    lm = nn.QuantizedLinear.from_linear(nn.Linear(64, 320, bias=False), group_size=32, bits=4)
    for vocab in (320, 248320):                               # the embedding's rows, or a head shorter than it
        ids, head = draft_head_for(lm, vocab)
        assert int(ids.max()) < 320 and len(ids) % 64 == 0
        np.testing.assert_array_equal(np.array(head.weight), np.array(lm.weight)[ids.astype(np.int64)])
    assert draft_head_for(lm, 300) == (None, None)            # every id below 300 listed: 300 rows, not 64s
    listed = tmp_path / "ids.txt"
    listed.write_text("1\n2\n70000\n")
    with pytest.raises(ValueError, match="multiple of 64"):
        draft_ids(listed, vocab=10)
    listed.write_text("5\n")
    with pytest.raises(ValueError, match="no token ids below 3"):
        draft_ids(listed, vocab=3)
