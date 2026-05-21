"""Decode A02+A03 MCReports to CSV using the auto-detecting decode function."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

import mc_automation as mc
from config import DateRange, ParamAxis, StrategyConfig

WORKSPACE = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp"
INSAMPLE  = DateRange("2019/01/01", "2026/01/01")

RUNS = [
    (
        r"C:\Users\Tim\Documents\Optimization TOUCHANCE Futures CME.CL HOT CME 1 Day 20260521-005414.MCReport",
        r"C:\Users\Tim\MultichartAI\results\cl_cl_daily_search\CLCL_02_fine_lese_raw.csv",
        "CLCL_02_fine_lese",
        [ParamAxis("LE",1,5,1), ParamAxis("SE",1,5,1), ParamAxis("STP",1,3,0.5), ParamAxis("LMT",3,7,1), ParamAxis("LenLE",95,105,5)],
    ),
    (
        r"C:\Users\Tim\Documents\Optimization TOUCHANCE Futures CME.CL HOT CME 1 Day 20260521-005627.MCReport",
        r"C:\Users\Tim\MultichartAI\results\cl_cl_daily_search\CLCL_03_high_lmt_raw.csv",
        "CLCL_03_high_lmt",
        [ParamAxis("LE",1,5,1), ParamAxis("SE",1,5,1), ParamAxis("STP",1,3,1), ParamAxis("LMT",8,20,2), ParamAxis("LenLE",95,105,5)],
    ),
]

for mcreport, out_csv, name, params in RUNS:
    print(f"\n=== {name} ===")
    cfg = StrategyConfig(
        name=name,
        mc_signal_name="_2021Basic_Break_CL",
        timeframe="daily",
        bar_period=1440,
        params=params,
        chart_workspace=WORKSPACE,
        chart_symbol="CME.CL HOT",
        insample=INSAMPLE,
    )
    ok = mc._decode_mcreport_to_csv(mcreport, cfg, out_csv)
    print("Success:", ok)
    if ok:
        df = pd.read_csv(out_csv)
        print(f"Rows: {len(df)}, columns: {list(df.columns[:8])}")
        np_col = next((c for c in df.columns if "Net" in c), None)
        if np_col:
            print(f"Best {np_col}: {df[np_col].max():.0f}")
            best = df.loc[df[np_col].idxmax()]
            print(f"  LE={best['LE']:.4g} SE={best['SE']:.4g} STP={best['STP']:.4g} LMT={best['LMT']:.4g}  NP={best[np_col]:.0f}  MDD={best.get('Max Intraday Drawdown', float('nan')):.0f}  trades={best.get('Total Trades', float('nan')):.0f}")
