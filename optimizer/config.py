from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class DateRange:
    from_date: str  # "YYYY/MM/DD"
    to_date: str    # "YYYY/MM/DD"


@dataclass
class ParamAxis:
    name: str       # exact name as shown in MC Inputs tab
    start: float
    stop: float
    step: float

    def values(self) -> np.ndarray:
        n = round((self.stop - self.start) / self.step) + 1
        return np.linspace(self.start, self.stop, int(n))

    def count(self) -> int:
        return len(self.values())


@dataclass
class StrategyConfig:
    name: str                   # display name for logging / filenames
    mc_signal_name: str         # exact signal name in MC Format Signals dialog
    timeframe: str              # "daily" | "hourly"
    bar_period: int             # bar size in minutes (1440=daily, 60=hourly)
    params: List[ParamAxis]
    chart_workspace: str        # full path to MC workspace (.wsp) file
    chart_symbol: str           # symbol shown in chart title (for window matching)
    insample: DateRange = field(default_factory=lambda: DateRange("2019/01/01", "2026/01/01"))
    outsample: DateRange = field(default_factory=lambda: DateRange("2018/01/01", "2019/01/01"))

    def total_runs(self) -> int:
        n = 1
        for p in self.params:
            n *= p.count()
        return int(n)


# ---------------------------------------------------------------------------
# Strategy definitions
# Edit mc_signal_name, chart_workspace, chart_symbol, and ParamAxis.name
# to match your actual MultiCharts64 setup before running.
# ---------------------------------------------------------------------------

BREAKOUT_DAILY = StrategyConfig(
    name="Breakout_Daily",
    mc_signal_name="_2021Basic_Break_NQ",
    timeframe="daily",
    bar_period=1440,
    params=[
        ParamAxis("LE", start=1,  stop=50, step=1),  # 50 values — long entry lookback
        ParamAxis("SE", start=1,  stop=50, step=1),  # 50 values — short entry lookback
        # STP (default 1.5) and LMT (default 6) held at defaults
    ],
    chart_workspace=r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp",  # TODO
    chart_symbol="TWF.TXF HOT",                # TODO: verify symbol label
)

BREAKOUT_HOURLY = StrategyConfig(
    name="Breakout_Hourly",
    mc_signal_name="_2021Basic_Break_NQ",
    timeframe="hourly",
    bar_period=60,
    params=[
        ParamAxis("LE", start=5,  stop=100, step=5),  # 20 values — long entry lookback
        ParamAxis("SE", start=5,  stop=100, step=5),  # 20 values — short entry lookback
        # STP (default 1.5) and LMT (default 6) held at defaults
    ],
    chart_workspace=r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp",  # TODO
    chart_symbol="TWF.TXF HOT",
)

SUPERTREND_DAILY = StrategyConfig(
    name="Supertrend_Daily",
    mc_signal_name="SFJ_SuperTrend_NQ",
    timeframe="daily",
    bar_period=1440,
    params=[
        ParamAxis("ATRLength",   start=1,   stop=50,  step=1),    # 50 values
        ParamAxis("Multiplier",  start=0.5, stop=5.0, step=0.25), # 19 values
    ],
    chart_workspace=r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_SuperTrend_AI.wsp",  # TODO
    chart_symbol="TWF.TXF HOT",
)

SUPERTREND_HOURLY = StrategyConfig(
    name="Supertrend_Hourly",
    mc_signal_name="SFJ_SuperTrend_NQ",
    timeframe="hourly",
    bar_period=60,
    params=[
        ParamAxis("ATRLength",   start=5,   stop=100, step=5),    # 20 values
        ParamAxis("Multiplier",  start=0.5, stop=5.0, step=0.5),  # 10 values
    ],
    chart_workspace=r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_SuperTrend_AI.wsp",  # TODO
    chart_symbol="TWF.TXF HOT",
)

ALL_STRATEGIES: List[StrategyConfig] = [
    BREAKOUT_DAILY,
    BREAKOUT_HOURLY,
    SUPERTREND_DAILY,
    SUPERTREND_HOURLY,
]

STRATEGY_MAP = {s.name.lower().replace("_", ""): s for s in ALL_STRATEGIES}
STRATEGY_MAP.update({
    "breakout_daily":    BREAKOUT_DAILY,
    "breakout_hourly":   BREAKOUT_HOURLY,
    "supertrend_daily":  SUPERTREND_DAILY,
    "supertrend_hourly": SUPERTREND_HOURLY,
})

# ---------------------------------------------------------------------------
# Plateau detection settings
# ---------------------------------------------------------------------------
PLATEAU_NEIGHBORHOOD_RADIUS = 2   # (2r+1)×(2r+1) window around each point
PLATEAU_MIN_TRADES = 10           # discard runs with fewer trades (noise filter)
PLATEAU_TOP_N = 20                # number of top candidates to report

# ---------------------------------------------------------------------------
# Automation settings
# ---------------------------------------------------------------------------
MC_PROCESS_NAME = "MultiCharts64.exe"
OPTIMIZATION_TIMEOUT_SECONDS = 7200   # 2 hours max per strategy
POLL_INTERVAL_SECONDS = 10
RESULTS_OUTPUT_DIR = r"C:\Users\Tim\MultichartAI\results"

# Column name mapping from MC CSV export to internal names.
# Adjust if your MC version uses different column headers.
MC_COLUMN_MAP = {
    "Net Profit":                    "NetProfit",
    "Max Strategy Drawdown":         "MaxDrawdown",
    "Maximum Strategy Drawdown":     "MaxDrawdown",
    "Max. Strategy Drawdown":        "MaxDrawdown",
    "Maximum Intraday Drawdown":     "MaxDrawdown",
    "Total Number of Trades":        "TotalTrades",
    "Total Trades":                  "TotalTrades",
    "# of Trades":                   "TotalTrades",
}
