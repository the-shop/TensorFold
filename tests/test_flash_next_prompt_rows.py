"""A prompt chunk the fused kernels take leaves the cache its own recurrent states, not views pinning the held rows."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.qwen4_exp.model import LinearCache
from tensorfold.families.qwen4_exp.runtime import AttentionCache, FlashNext

ROW = (4, 32, 32)            # 16 KiB of float32 a row: a held buffer is far past MLX's 16 KiB donation slack
CONV = (3, 2048)             # 12 KiB of bfloat16 a row
LAYERS = 3                   # two DeltaNet layers and an attention layer


def held(rows: int) -> int:
    return -(-rows // 8) * 8


def family() -> FlashNext:
    """FlashNext over a fake body whose fused forward does what FusedDecode._gdn does to a DeltaNet cache."""

    flash = FlashNext.__new__(FlashNext)
    flash.layer_count = LAYERS
    fused = SimpleNamespace(row_states={}, last_streams=None)

    def hidden(tokens, cache):
        rows = tokens.shape[1]
        if rows <= flash.fused_rows:
            for index, c in enumerate(cache):
                if isinstance(c, LinearCache):
                    base = 1000.0 * index + c.offset
                    ssm_rows = (base + mx.arange(held(rows) * int(np.prod(ROW)), dtype=mx.float32)).reshape(
                        held(rows), *ROW)
                    conv_rows = (base + mx.arange(held(rows) * int(np.prod(CONV)))).astype(mx.bfloat16).reshape(
                        held(rows), *CONV)
                    c.conv, c.ssm = conv_rows[rows - 1:rows], ssm_rows[rows - 1:rows]
                    c.offset += rows
                    fused.row_states[index] = [(conv_rows, ssm_rows, 0)]
            fused.last_streams = mx.zeros((rows, 8))
        else:                    # a prompt chunk's path: states of the cache's own, no rows kept
            for index, c in enumerate(cache):
                if isinstance(c, LinearCache):
                    c.conv, c.ssm = mx.zeros((1, *CONV), mx.bfloat16), mx.full((1, *ROW), float(rows))
                    c.offset += rows
            flash.model.__dict__["last_streams"] = mx.zeros((rows, 8))
        out = mx.zeros((1, rows, 8))
        mx.eval(out, *[a for c in cache if isinstance(c, LinearCache) for a in (c.conv, c.ssm)])
        return out

    flash.fused = fused
    flash.model = SimpleNamespace(hidden=hidden)
    return flash


def cache() -> list:
    return [LinearCache(), LinearCache(), AttentionCache()]


def measure(arrays) -> int:
    mx.eval(*arrays)
    gc.collect()
    return int(mx.get_active_memory())


@pytest.mark.parametrize("rows", [1, 8, 13, 16])
def test_a_fused_prompt_chunk_keeps_standalone_exact_states_and_no_rows(rows):
    flash, work = family(), cache()
    expected = None

    def snapshot(c):
        return np.array(c.ssm), np.array(c.conv.astype(mx.float32))

    tokens = mx.array([list(range(rows))], dtype=mx.uint32)
    reference = family()
    ref_cache = cache()
    reference.hidden(tokens, ref_cache)
    expected = [snapshot(c) for c in ref_cache[:2]]
    del reference, ref_cache

    before = measure([])
    flash.prefill(tokens, work)
    states = [a for c in work[:2] for a in (c.ssm, c.conv)]
    after = measure(states)

    assert flash.fused.row_states == {}
    for c, (ssm, conv) in zip(work[:2], expected):
        assert c.ssm.shape == (1, *ROW) and c.conv.shape == (1, *CONV)
        got_ssm, got_conv = snapshot(c)
        assert np.array_equal(got_ssm, ssm) and np.array_equal(got_conv, conv)       # bit-exact copies
    one_row = 2 * (int(np.prod(ROW)) * 4 + int(np.prod(CONV)) * 2)
    # only the two layers' own rows stay: the held buffers (8 or 16 rows a layer) are freed, not pinned by views
    assert after - before <= one_row + 2 * 16384, (after - before, one_row)
    assert work[2].keys is None                                                      # attention caches untouched


def test_the_held_buffers_would_stay_pinned_without_the_copy():
    """The control: the plain fused forward (``hidden``) leaves views that keep every held row alive."""

    flash, work = family(), cache()
    before = measure([])
    flash.hidden(mx.array([list(range(13))], dtype=mx.uint32), work)
    flash.fused.row_states.clear()
    after = measure([a for c in work[:2] for a in (c.ssm, c.conv)])
    held_bytes = 2 * held(13) * (int(np.prod(ROW)) * 4 + int(np.prod(CONV)) * 2)
    assert after - before >= held_bytes


def test_a_chunk_past_the_fused_rows_is_left_as_the_prompt_path_made_it():
    flash, work = family(), cache()
    sentinel = [(mx.zeros((1,)), mx.zeros((1,)), 0)]
    flash.fused.row_states[7] = sentinel
    flash.prefill(mx.array([list(range(17))], dtype=mx.uint32), work)
    kept = [(c.conv, c.ssm) for c in work[:2]]
    assert flash.fused.row_states == {7: sentinel}                                   # no rows dropped
    for c, (conv, ssm) in zip(work[:2], kept):
        assert c.conv is conv and c.ssm is ssm                                        # no copies made
        assert np.array_equal(np.array(c.ssm), np.full((1, *ROW), 17.0, dtype=np.float32))


def test_without_the_fused_kernels_prefill_is_hidden():
    flash, work = family(), cache()
    flash.fused = None
    flash.model.__dict__["last_streams"] = None

    def hidden(tokens, c):
        c[0].ssm, c[0].conv = mx.ones((1, *ROW)), mx.ones((1, *CONV))
        return mx.zeros((1, tokens.shape[1], 8))

    flash.model.hidden = hidden
    flash.prefill(mx.array([[1, 2, 3]], dtype=mx.uint32), work)
    ssm = work[0].ssm
    assert ssm.shape == (1, *ROW)
