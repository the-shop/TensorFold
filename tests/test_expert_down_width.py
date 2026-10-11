"""expert_down_y reads exactly its activation row at every expert width it takes (none past the row or the buffer)."""

import types

import mlx.core as mx
import numpy as np
import pytest

from tensorfold.kernels.qwen.flash_next.v1 import experts, stream_experts


@pytest.fixture(autouse=True)
def _gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _q(w, bits):
    group = 32 if bits == 4 else 64
    wq, s, b = mx.quantize(w, group_size=group, bits=bits)
    return types.SimpleNamespace(weight=wq, scales=s, biases=b, bits=bits, group_size=group), group


def _case(width, bits, rows=3, n_experts=8, top_k=4, dims=256, seed=0):
    mx.random.seed(seed)
    down, group = _q((mx.random.normal((n_experts, dims, width)) * 0.05).astype(mx.bfloat16), bits)
    shared, _ = _q((mx.random.normal((dims, width)) * 0.05).astype(mx.bfloat16), bits)
    act = mx.random.normal((rows, top_k + 1, width)).astype(mx.bfloat16)
    picks = mx.array(np.random.default_rng(seed).integers(0, n_experts, (rows, top_k)), dtype=mx.uint32)
    mx.eval(down.weight, shared.weight, act, picks)
    return act, picks, down, shared, group


def _reference(act, picks, down, shared, bits, group):
    w = mx.dequantize(down.weight, down.scales, down.biases, group_size=group, bits=bits).astype(mx.float32)
    ws = mx.dequantize(shared.weight, shared.scales, shared.biases, group_size=group, bits=bits).astype(mx.float32)
    a = np.array(act.astype(mx.float32))
    p = np.array(picks)
    w, ws = np.array(w), np.array(ws)
    rows, slots, _ = a.shape
    out = np.zeros((rows, slots, w.shape[1]), np.float32)
    for r in range(rows):
        for k in range(slots):
            out[r, k] = (ws if k == slots - 1 else w[p[r, k]]) @ a[r, k]
    return out


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("width", [64, 128, 256, 512, 640, 1024])
def test_down_matches_reference_and_repeats(width, bits):
    if bits == 8 and width % 64:
        pytest.skip("8-bit rows come in groups of 64")
    act, picks, down, shared, group = _case(width, bits)
    ref = _reference(act, picks, down, shared, bits, group)
    outs = []
    for _ in range(3):
        junk = [mx.random.normal((n,)) * 1e4 for n in (1 << 10, 1 << 14, 1 << 18)]   # stale data in MLX's free pool
        mx.eval(*junk)
        del junk
        y = experts.expert_down_y(act, picks, down, shared)
        outs.append(np.array(y.astype(mx.float32)))
    assert all(np.array_equal(outs[0], o) for o in outs[1:])
    err = np.linalg.norm(outs[0] - ref) / np.linalg.norm(ref)
    assert err < 1e-2, err


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("width", [512, 640, 1024])
def test_wide_rows_keep_the_original_source(width, bits):
    """Widths the original kernel covered compile from the unchanged text (the same pipeline, the same bits)."""

    assert experts.down_width(width, bits) is False
    want = experts._EXPERT_DOWN_Y if bits == 4 else experts.q8(experts._EXPERT_DOWN_Y, experts._Q8_DOWN)
    assert experts.down_source(experts._EXPERT_DOWN_Y, False, bits) == want
    stream = stream_experts._down_source()
    want = stream if bits == 4 else experts.q8(stream, experts._Q8_DOWN)
    assert experts.down_source(stream, False, bits) == want


@pytest.mark.parametrize(("width", "bits"), [(64, 4), (128, 8), (448, 8), (480, 4), (32, 4)])
def test_narrow_rows_take_the_narrow_variant(width, bits):
    assert experts.down_width(width, bits) is True
    for source in (experts._EXPERT_DOWN_Y, stream_experts._down_source()):
        assert "const bool first = int(lane) < NC;" in experts.down_source(source, True, bits)


@pytest.mark.parametrize(("width", "bits"), [(1040, 4), (1088, 8), (72, 4), (96, 8), (496, 4), (48, 4), (16, 4), (1008, 4),
                                              (0, 4), (0, 8), (-64, 8), (-32, 4)])
def test_unsupported_widths_are_refused(width, bits):
    with pytest.raises(ValueError):
        experts.down_width(width, bits)


@pytest.mark.parametrize("width", [16, 48, 496])
def test_q4_rows_off_the_group_size_never_reach_the_kernel(width):
    """A 4-bit lane reads one scale per 32 inputs: a row of 16 mod 32 would take the next group's scale."""

    act, picks, down, shared, _ = _case(64, 4)
    act = mx.zeros((act.shape[0], act.shape[1], width), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="multiple of 32"):
        experts.expert_down_y(act, picks, down, shared)
