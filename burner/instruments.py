"""Static 6-instrument table for the burner.

state.json does NOT record the workspace path / main-signal name / window
tokens -- those live only in the pipeline script constants.  This table
duplicates the identity part of the pipeline INSTRUMENTS rows;
tests/test_constants_sync.py regex-scrapes a pipeline script to keep the two
from drifting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

INST_ENUM = ("BTC", "ETH", "BNB", "TXF", "NQ", "GC")
TF_MAP = {"hourly": "H1", "daily": "D1", "240": "M240"}


@dataclass(frozen=True)
class InstrumentCtx:
    key: str            # pipeline key, e.g. "btc"
    inst: str           # strategy_id enum, e.g. "BTC"
    chart_symbol: str   # MC chart window symbol, e.g. "BTCUSDT HOT"
    symbol: str         # manifest symbol, e.g. "BTCUSDT"
    symbol_class: str   # "crypto" | "futures"
    variant_suffix: str # main-source suffix: "crypto" | "NQ"
    tokens: List[str]   # activate_chart_by_symbol match tokens


def _mk(key, inst, chart_symbol, tokens, symbol_class) -> InstrumentCtx:
    suffix = "crypto" if symbol_class == "crypto" else "NQ"
    symbol = chart_symbol[:-4] if chart_symbol.endswith(" HOT") else chart_symbol
    return InstrumentCtx(key=key, inst=inst, chart_symbol=chart_symbol, symbol=symbol,
                         symbol_class=symbol_class, variant_suffix=suffix,
                         tokens=list(tokens))


INSTRUMENTS: Dict[str, InstrumentCtx] = {c.key: c for c in [
    _mk("btc", "BTC", "BTCUSDT HOT", ["BTCUSDT"], "crypto"),
    _mk("eth", "ETH", "ETHUSDT HOT", ["ETHUSDT"], "crypto"),
    _mk("bnb", "BNB", "BNBUSDT HOT", ["BNBUSDT"], "crypto"),
    _mk("txf", "TXF", "TWF.TXF HOT", ["TXF"], "futures"),
    _mk("nq", "NQ", "CME.NQ HOT", ["CME.NQ"], "futures"),
    _mk("gc", "GC", "CME.GC HOT", ["CME.GC"], "futures"),
]}

# Module label -> MC signal name (mirrors the pipelines' MODULES table).
MODULE_REGISTRY: Dict[str, str] = {
    "M1": "SFJ_15Dworkshop_lesson4_ATRstop",
    "M2": "SFJ_15Dworkshop_lesson9_1_TrailingStop",
    "M3": "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit",
    "M4": "SFJ_15Dworkshop_lesson11_3_high_volatility_exit",
    "M5": "QuantPass_PT_Exit",
    "M6": "RescueTeamExit",
}


def state_json_path(key: str, strat_key: str, tf: str) -> Path:
    return REPO_ROOT / "results" / f"{key}_{strat_key}_{tf}_pipeline" / "state.json"


def main_source_path(name: str, ctx: InstrumentCtx) -> Path:
    return REPO_ROOT / "Strategy" / f"{name}_{ctx.variant_suffix}.txt"
