import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from burner.instruments import INSTRUMENTS, MODULE_REGISTRY  # noqa: E402
from burner.reader import BurnSource, KeptModule  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

MINIMAIN = (FIXTURES / "MiniMain_crypto.txt")

MODULE_PARAMS = {
    "M1": {"STP": 6.1},
    "M2": {"ATRSTP": 40.6},
    "M3": {"EXITBAR": 45.0},
    "M4": {"DAYRANGE": 6.04},
    "M5": {"PT_Base": 0.246},
    "M6": {"Length": 140.0, "std": 5.1},
}


def make_src(kept_labels, main_params=None, sha="0" * 64, ctx_key="btc",
             state_path=None):
    kept = [KeptModule(label=lb, signal=MODULE_REGISTRY[lb],
                       params=dict(MODULE_PARAMS[lb])) for lb in kept_labels]
    return BurnSource(
        ctx=INSTRUMENTS[ctx_key],
        state_path=state_path or FIXTURES / "fake_state_full.json",
        state_sha256=sha,
        main_params=main_params or {"Length": 8.0, "BandMult": 4.75,
                                    "ATRMult": 5.5, "ReentryBars": 0.0},
        winner={"oos_np": 498.13, "mdd_full": -329.96, "pass": False},
        kept=kept,
        stage4_final={"net_profit": 2335.01, "max_intraday_drawdown": -329.96,
                      "romad": 7.0766},
        timeframe="hourly",
    )


def fake_state(final_kept=("M5", "M6"), tail_discard=True, with_stage3=True,
               with_stage4=True):
    """Minimal-but-complete state.json dict for reader tests."""
    steps = []
    romad = 5.0
    for i, lb in enumerate(final_kept, 1):
        romad += 1.0
        steps.append({"step": i, "candidate": lb, "signal": f"sig_{lb}",
                      "enabled": [], "net_profit": 1000.0 + i,
                      "max_intraday_drawdown": -100.0 - i, "romad": romad,
                      "prev_romad": romad - 1.0, "delta_romad_pct": 1.0,
                      "decision": "KEEP", "valid": True})
    if tail_discard:
        steps.append({"step": len(steps) + 1, "candidate": "M2", "signal": "sig_M2",
                      "enabled": [], "net_profit": 1.0,
                      "max_intraday_drawdown": -999.0, "romad": 0.1,
                      "prev_romad": romad, "delta_romad_pct": -99.0,
                      "decision": "discard", "valid": True})
    state = {
        "stage1": {"symbol": "BTCUSDT HOT"},
        "stage2": {
            "winner": {"idx": 0,
                       "params": {"Length": 8.0, "BandMult": 4.75,
                                  "ATRMult": 5.5, "ReentryBars": 0.0},
                       "oos_np": 498.13, "mdd_full": -329.96, "pass": False,
                       "break_ratio": 1.469},
            "main_champ": {"Length": 8.0, "BandMult": 4.75, "ATRMult": 5.5,
                           "ReentryBars": 0.0},
        },
        "stage3": {"modules": {lb: {"signal": sig,
                                    "params": dict(MODULE_PARAMS[lb])}
                               for lb, sig in MODULE_REGISTRY.items()}}
                  if with_stage3 else None,
        "stage4": {"baseline": {"net_profit": 1888.66,
                                "max_intraday_drawdown": -329.96},
                   "steps": steps,
                   "final_kept": list(final_kept),
                   "final_romad": romad} if with_stage4 else None,
    }
    return state


@pytest.fixture
def write_state(tmp_path):
    def _write(state, name="state.json"):
        p = tmp_path / name
        p.write_text(json.dumps(state), encoding="utf-8")
        return p
    return _write
