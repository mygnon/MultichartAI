import pytest

from burner.instruments import INSTRUMENTS
from burner.reader import BurnInputError, load_burn_source
from conftest import fake_state

CTX = INSTRUMENTS["btc"]


def test_main_champ_and_kept_extraction(write_state):
    p = write_state(fake_state(final_kept=("M5", "M6")))
    src = load_burn_source(p, CTX, "hourly")
    assert src.main_params == {"Length": 8.0, "BandMult": 4.75,
                              "ATRMult": 5.5, "ReentryBars": 0.0}
    assert [m.label for m in src.kept] == ["M5", "M6"]
    assert src.kept[1].signal == "RescueTeamExit"
    assert src.kept[1].params == {"Length": 140.0, "std": 5.1}


def test_stage4_final_ignores_trailing_discard(write_state):
    p = write_state(fake_state(final_kept=("M5", "M6"), tail_discard=True))
    src = load_burn_source(p, CTX, "hourly")
    # last KEEP step, not the trailing discard
    assert src.stage4_final["net_profit"] == 1002.0
    assert src.stage4_final["max_intraday_drawdown"] == -102.0


def test_empty_final_kept_falls_back_to_baseline(write_state):
    p = write_state(fake_state(final_kept=(), tail_discard=True))
    src = load_burn_source(p, CTX, "hourly")
    assert src.kept == []
    assert src.stage4_final["net_profit"] == 1888.66
    assert src.stage4_final["romad"] == round(1888.66 / 329.96, 4)


def test_missing_stage3_raises(write_state):
    p = write_state(fake_state(with_stage3=False))
    with pytest.raises(BurnInputError):
        load_burn_source(p, CTX, "hourly")


def test_missing_stage4_raises(write_state):
    p = write_state(fake_state(with_stage4=False))
    with pytest.raises(BurnInputError):
        load_burn_source(p, CTX, "hourly")


def test_no_modules_skips_stage3(write_state):
    p = write_state(fake_state(with_stage3=False))
    # stage3 missing is fine when burning bare main
    src = load_burn_source(p, CTX, "hourly", no_modules=True)
    assert src.kept == []


def test_missing_file_raises(tmp_path):
    with pytest.raises(BurnInputError):
        load_burn_source(tmp_path / "nope.json", CTX, "hourly")
