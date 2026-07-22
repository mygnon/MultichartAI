"""End-to-end burn against the REAL DualAnchor state.json / Strategy sources,
written into a temp burned_root (repo's burned/ untouched)."""
import json

import pytest

from burner.burn import burn_all
from burner.instruments import REPO_ROOT

STATE = REPO_ROOT / "results" / "btc_dualanchor_hourly_pipeline" / "state.json"


@pytest.mark.skipif(not STATE.exists(), reason="dualanchor results absent")
def test_full_six_instrument_burn(tmp_path):
    results = burn_all("DualAnchorBreakout", "dualanchor", "hourly",
                       ["btc", "eth", "bnb", "txf", "nq", "gc"],
                       burned_root=tmp_path)
    assert [r.status for r in results] == ["burned"] * 6
    out = tmp_path / "DualAnchorBreakout"
    assert (out / "burn_report.json").exists()

    # BTC kept M5+M1+M2 per state.json -- manifest must mirror it
    mf = json.loads((out / "DualAnchorBreakout_BTC_H1_v1.manifest.json")
                    .read_text(encoding="utf-8"))
    assert [m["id"] for m in mf["exit_modules"]] == ["M5", "M1", "M2"]
    assert mf["params"] == {"Length": 8.0, "BandMult": 4.75, "ATRMult": 5.5,
                            "ReentryBars": 0.0}
    assert mf["stage4_final"]["romad"] == 7.0766

    # rerun -> everything reused, byte-identical
    again = burn_all("DualAnchorBreakout", "dualanchor", "hourly",
                     ["btc"], burned_root=tmp_path)
    assert again[0].status == "reused"


@pytest.mark.skipif(not STATE.exists(), reason="dualanchor results absent")
def test_no_modules_burn(tmp_path):
    results = burn_all("DualAnchorBreakout", "dualanchor", "hourly", ["gc"],
                       no_modules=True, burned_root=tmp_path)
    assert results[0].status == "burned"
    txt = (tmp_path / "DualAnchorBreakout" / "DualAnchorBreakout_GC_H1_v1.txt")
    content = txt.read_text(encoding="utf-8")
    assert "burned exit-module declarations" not in content
    assert "OMS signal emit" in content
