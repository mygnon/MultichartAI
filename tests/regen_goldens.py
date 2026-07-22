"""Regenerate golden files for test_golden_burn.py.  Run, then REVIEW the diff
before committing -- goldens are the byte-exact contract of the renderer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))

from test_golden_burn import GOLDEN, build  # noqa: E402

GOLDEN.mkdir(parents=True, exist_ok=True)
(GOLDEN / "MiniMain_nomodules.txt").write_text(build([]), encoding="utf-8", newline="\n")
(GOLDEN / "MiniMain_m5_m6.txt").write_text(build(["M5", "M6"]), encoding="utf-8", newline="\n")
print("goldens regenerated:", *GOLDEN.iterdir(), sep="\n  ")
