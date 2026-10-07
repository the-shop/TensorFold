"""The Flash Next fused kernels on 8-bit weights (groups of 64; n-gram rows groups of 32) against MLX's own ops."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.kernels.qwen.flash_next.v1 import embed, experts, stream_experts  # noqa: E402


def _q(shape, key, bits=8, group=64):
    w = (mx.random.normal(shape, key=key) * 0.02).astype(mx.bfloat16)
    return w, mx.quantize(w, group_size=group, bits=bits)


class _Lin:
    def __init__(self, q, bits=8, group=64):
        self.weight, self.scales, self.biases = q
        self.bits, self.group_size = bits, group


def _close(a, b, tol=2e-2):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    err = float(mx.max(mx.abs(a - b)).item())
    scale = float(mx.max(mx.abs(b)).item())
    assert err <= tol * max(scale, 1e-3), (err, scale)


@pytest.mark.parametrize("rows", [1, 3])
def test_expert_kernels_q8(rows):
    keys = mx.random.split(mx.random.key(7), 8)
    E, D, N, K = 64, 512, 640, 4
    _, g = _q((E, N, D), keys[0])
    _, u = _q((E, N, D), keys[1])
    _, dn = _q((E, D, N), keys[2])
    _, sg = _q((N, D), keys[3])
    _, su = _q((N, D), keys[4])
    _, sd = _q((D, N), keys[5])
    x = (mx.random.normal((rows, D), key=keys[6])).astype(mx.bfloat16)
    logits = mx.random.normal((rows, E + 1), key=keys[7]).astype(mx.float32)
    G, U, DN, SG, SU, SD = (_Lin(t) for t in (g, u, dn, sg, su, sd))
    act, picks, wts = experts.expert_gateup(x, logits, K, E, G, U, shared=(SG, SU))
    y = experts.expert_down_y(act, picks, DN, SD)
    mx.eval(act, picks, wts, y)
    top = np.argsort(-np.array(logits[:, :E]), axis=-1, kind="stable")[:, :K]
    assert (np.array(picks) == top).all()
    for r in range(rows):
        for k in range(K + 1):
            if k < K:
                e = int(top[r, k])
                wg, wu, wd = (mx.dequantize(*(a[e] for a in t), group_size=64, bits=8) for t in (g, u, dn))
            else:
                wg, wu, wd = (mx.dequantize(*t, group_size=64, bits=8) for t in (sg, su, sd))
            gv = (x[r:r + 1].astype(mx.float32) @ wg.T.astype(mx.float32))
            uv = (x[r:r + 1].astype(mx.float32) @ wu.T.astype(mx.float32))
            a = nn.silu(gv) * uv
            _close(act[r, k][None], a)
            _close(y[r, k][None], act[r, k][None].astype(mx.float32) @ wd.T.astype(mx.float32))

    # the slot pool's kernels: expert e lives in slot SLOTOF[e] of a shuffled pool, same results
    perm = np.random.default_rng(0).permutation(E)
    inv = np.argsort(perm)

    class _Slots:
        def __init__(self, q):
            self.weight, self.scales, self.biases = (a[mx.array(perm)] for a in q)

    class _Pool:
        gate, up, down = _Slots(g), _Slots(u), _Slots(dn)

    slot_of = mx.array(inv.astype(np.int32))
    layer = mx.array([0] * 8, dtype=mx.int32)
    act2, picks2, wts2 = stream_experts.gateup(x, logits, K, E, _Pool, slot_of, layer, (SG, SU))
    y2 = stream_experts.down(act2, picks2, E, _Pool, slot_of, layer, SD)
    assert mx.array_equal(act, act2).item() and mx.array_equal(picks, picks2).item()
    assert mx.array_equal(wts, wts2).item() and mx.array_equal(y, y2).item()


@pytest.mark.parametrize("bits,group", [(4, 32), (8, 64)])
def test_embed_rows(bits, group):
    table, q = _q((100, 256), mx.random.key(3), bits, group)
    emb = _Lin(q, bits, group)
    ids = mx.array([5, 99, 0], dtype=mx.uint32)
    out = embed.embed_rows(ids, emb, tile=2)
    ref = mx.dequantize(*q, group_size=group, bits=bits)[ids]
    assert mx.array_equal(out, mx.concatenate([ref, ref], axis=-1)).item()


def test_ple_rows_q8():
    _, q = _q((50, 160), mx.random.key(4), 8, 32)

    class _Host:
        def gather(self, ids):
            flat = np.asarray(ids).reshape(-1)
            return (np.array(q[0])[flat], np.array(q[1].view(mx.uint16))[flat], np.array(q[2].view(mx.uint16))[flat])

    class _Emb:
        dims, host, quant_bits, quant_group = 160, _Host(), 8, 32

    tables = embed.PleTables(_Emb())
    ids = np.array([[1, 2], [49, 0]])
    out = embed.ple_lookup(ids, tables)
    ref = mx.dequantize(*q, group_size=32, bits=8)[mx.array(ids.reshape(-1))].reshape(2, 320)
    assert mx.array_equal(out, ref).item()
