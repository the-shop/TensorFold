"""Prompt snapshots of Flash Next key on its declared numerics, not its source: this test is what keeps that safe.

It prefills a tiny deterministic model (8- and 4-bit; the canary through a reply-cut plan, a long prompt through the
selected-blocks attention, MLX's matmul route) and compares the caches' digests with the ones recorded for this
NUMERICS_VERSION, MLX and GPU. A change that moves them without a bump fails here. When bits move on purpose: bump
``NUMERICS_VERSION`` in families/qwen4_exp/__init__.py (or ``ENGINE_NUMERICS`` in engine/numerics.py for shared
engine code), then record with ``TF_RECORD_NUMERICS_GOLDEN=1`` and commit tests/fixtures/numerics_golden.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

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
    done = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=600)
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
