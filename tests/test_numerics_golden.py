"""Prompt snapshots of Flash Next key on its declared numerics and a canary, not its source; this test pins a tiny
model's prompt-cache bits per declared numerics so an edit that moves them is caught where the canary cannot see.

It prefills a tiny deterministic model (8- and 4-bit, the full-attention layer second so attention output reaches
the cache) through the canary's reply-cut plan, the selected-blocks attention (DENSE_KEYS patched to 64, and
unpatched past 4096 keys), chunks of 8192 and 8448 tokens (expert rows at and past MAX_SORTED_ROWS), last chunks of
1-4 rows (PIPE_ROWS, SCALAR_ROWS) and MLX's matmul route, and compares the caches' digests with the ones recorded for
this NUMERICS_VERSION, MLX and GPU. It covers what those cases reach on that model, not every shape a real model
takes (other head counts, expert counts or MTP). Another MLX or GPU has no record and skips. When bits move on
purpose: bump ``NUMERICS_VERSION`` in families/qwen4_exp/__init__.py (or ``ENGINE_NUMERICS`` in engine/numerics.py
for shared engine code), then record with ``TF_RECORD_NUMERICS_GOLDEN=1`` and commit
tests/fixtures/numerics_golden.json (then tests/test_numerics_reviewed.py's digest).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

TESTS = Path(__file__).resolve().parent
GOLDEN = TESTS / "fixtures" / "numerics_golden.json"
SCRIPT = TESTS / "numerics_tiny.py"


def tiny_report(src: Path | None = None) -> dict:
    """numerics_tiny.py's report in a fresh process, on ``src`` (a package tree's parent) or the installed one."""

    import tensorfold

    env = dict(os.environ)
    env["PYTHONPATH"] = str(src if src is not None else Path(tensorfold.__file__).resolve().parents[1])
    env.pop("TF_FLASH_PREFILL", None)
    done = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=600,
                          check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_prompt_cache_bits_match_the_declared_numerics():
    report = tiny_report()
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.is_file() else {}
    recorded = golden.get(report["declared"], {}).get(report["environment"])
    if os.environ.get("TF_RECORD_NUMERICS_GOLDEN") == "1":
        golden.setdefault(report["declared"], {})[report["environment"]] = report["digests"]
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        return
    if recorded is None:
        pytest.skip(f"no golden digests for {report['declared']} on {report['environment']}: record them with "
                    "TF_RECORD_NUMERICS_GOLDEN=1 (on the code whose bits they are)")
    moved = sorted(k for k in report["digests"] if recorded.get(k) != report["digests"][k])
    assert not moved, (
        f"prompt-cache bits moved for {moved} under the same declared numerics {report['declared']} on "
        f"{report['environment']}: stored prompt snapshots would be reused with other bits. If the change is meant, "
        "bump NUMERICS_VERSION (families/qwen4_exp/__init__.py) or ENGINE_NUMERICS (engine/numerics.py) and record "
        "with TF_RECORD_NUMERICS_GOLDEN=1; if MLX or macOS changed rather than the code, record for it.")
    assert set(recorded) <= set(report["digests"]), "a recorded case no longer runs"


def test_the_cases_straddle_every_size_the_prefill_switches_on():
    from tensorfold.families.qwen4_exp import model as q4
    from tensorfold.kernels.qwen.flash_next.v1 import gdn, hc, prefill_mm
    from tests import numerics_tiny as tiny

    top = tiny.TEXT["num_experts_per_tok"]
    assert tiny.LONG_EVEN * top <= prefill_mm.MAX_SORTED_ROWS < tiny.LONG_SPLIT * top     # one sorted gather, split
    assert tiny.SPARSE_PROMPT > 64 and min(tiny.LONG_TAIL) > q4.prefill.DENSE_KEYS        # selected-blocks attention
    assert set(range(1, gdn.PIPE_ROWS + 1)) <= set(tiny.TAILS) and max(tiny.TAILS) > hc.SCALAR_ROWS
    assert min(tiny.TAILS) <= hc.SCALAR_ROWS
    assert tiny.TEXT["layer_types"][-1] != "full_attention"       # attention output reaches a later layer's state
