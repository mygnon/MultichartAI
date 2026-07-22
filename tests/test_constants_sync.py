"""burner/instruments.py duplicates identity constants from the pipeline
scripts (state.json does not carry them).  Lock the copy against the source."""
import pytest

from burner.equivalence import default_pipeline_script, scrape_pipeline_constants
from burner.instruments import INSTRUMENTS

SCRIPT = default_pipeline_script("dualanchor")


@pytest.fixture(scope="module")
def consts():
    if not SCRIPT.exists():
        pytest.skip(f"{SCRIPT.name} absent")
    return scrape_pipeline_constants(SCRIPT)


def test_all_six_instruments_scraped(consts):
    assert set(consts["instruments"]) == set(INSTRUMENTS)


def test_identity_fields_match(consts):
    for key, row in consts["instruments"].items():
        ctx = INSTRUMENTS[key]
        assert row["chart_symbol"] == ctx.chart_symbol, key
        assert row["tokens"] == ctx.tokens, key
        assert row["symbol_class"] == ctx.symbol_class, key
        # main signal suffix must match the variant the burner reads from Strategy/
        assert row["main_signal"].endswith("_" + ctx.variant_suffix), key


def test_full_ranges_present(consts):
    for klass in ("crypto", "futures"):
        rng = consts["full_ranges"][klass]
        assert len(rng) == 2 and all("/" in d for d in rng)
