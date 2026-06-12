"""
search_gc_xtreme_hourly3.py — SFJ_XtremeStop_NQ on CME.GC HOT Hourly, Round 3

R2 champion: X=7, LY=0.60%, SY=1.80%, NP=426,580, MDD=-46,720, trades=206 (gap -46.7%)
R1→R2 gain: +1.77% (regime shift: X=1→X=7; R1's X=1 was artifact of step=5 grid)

R2 key findings:
  - X=7 confirmed by A01/A04/A10 triple convergence at step=0.05 precision
  - R1's X=1 was a false optimum due to R1's step-5 X grid missing X=7
  - A11 (step=0.01 zoom3) UI-failed — fine precision gap remains
  - A03/A05/A07/A09 also UI-failed (5/11 = 45% failure rate)
  - Near-zero LY (A02): X=10, LY=0.07, NP=389K, 602 trades — more frequent but lower NP
  - SY=1.80 confirmed as local peak in A04 (SY range up to 2.5) and A10 (SY up to 2.1)

R3 focus:
  - Fill A11 gap: step=0.01 precision around X=7, LY=0.60, SY=1.80
  - Verify SY ceiling: does SY>1.8 improve NP? (A03 wide_sy_x1 failed in R2)
  - Compare X=1 and X=7 directly at equal fine resolution
  - Cover X=20-100 (A07 failed twice in R2)
  - Confirm global X=7 optimum with broader fine search
  - Adaptive zooms with targeted small combos to reduce UI failure risk

Attempt schedule (11 attempts, <=5,000 combos each):
  A01 ultra_fine    : X(5-9 s1)xLY(0.52-0.68 s0.01)xSY(1.66-1.94 s0.01)  = 5x17x29=2465
  A02 sy_extended   : X(1-8 s1)xLY(0.5-1.0 s0.1)xSY(1.5-6.0 s0.25)       = 8x6x19=912
  A03 x_compare     : X(1-10 s1)xLY(0.3-1.2 s0.1)xSY(1.0-3.0 s0.1)       = 10x10x21=2100
  A04 high_x_confirm: X(20-100 s5)xLY(0.4-1.0 s0.1)xSY(1.0-3.0 s0.25)    = 17x7x9=1071
  A05 fine_sy_sweep : X(5-9 s1)xLY(0.4-0.8 s0.05)xSY(1.5-2.5 s0.02)      = 5x9x51=2295
  A06 x_boundary    : X(1-15 s1)xLY(0.5-0.7 s0.05)xSY(1.6-2.0 s0.05)     = 15x5x9=675
  A07 global_fine   : X(1-50 s2)xLY(0.4-0.8 s0.1)xSY(1.2-2.4 s0.2)       = 25x5x7=875
  A08 wide_sy_check : X(5-10 s1)xLY(0.3-0.8 s0.1)xSY(2.0-8.0 s0.25)      = 6x6x25=900
  A09 adaptive_zoom1: X+-10 s2, LY+-0.50 s0.1, SY+-0.50 s0.1              <= 9x11x11=1089
  A10 adaptive_zoom2: X+-5  s1, LY+-0.20 s0.02, SY+-0.20 s0.02            <= 11x21x21=4851
  A11 adaptive_zoom3: X+-3  s1, LY+-0.10 s0.01, SY+-0.10 s0.01            <= 7x21x21=3087
"""
from __future__ import annotations
import argparse
import ctypes
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mc_automation as mc
import plateau as plateau_mod
from config import DateRange, ParamAxis, StrategyConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\SFJ_XtremeStop_AI.wsp"
SYMBOL     = "CME.GC HOT"
SIGNAL     = "SFJ_XtremeStop_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_xtreme_hourly3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0   # USD

X_LO,  X_HI  = 1.0,   10000.0
LY_LO, LY_HI = 0.005, 50.0
SY_LO, SY_HI = 0.01,  50.0

SEED_X  = 7.0
SEED_LY = 0.60
SEED_SY = 1.80
SEED_NP = 426_580.0

PREFIX = "GCXH3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_xtreme_hourly3_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s -- %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(val: float, step: float) -> float:
    return round(round(val / step) * step, 8)


def zoom(center: float, radius: float, step: float,
         lo: float, hi: float) -> Tuple[float, float, float]:
    start = max(lo, _snap(center - radius, step))
    stop  = min(hi, _snap(center + radius, step))
    if stop <= start:
        stop = start + step
    return (start, stop, step)


def n_vals(t: Tuple[float, float, float]) -> int:
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name: str,
         x:  Tuple[float, float, float],
         ly: Tuple[float, float, float],
         sy: Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    x  = _safe(x,  X_LO,  X_HI)
    ly = _safe(ly, LY_LO, LY_HI)
    sy = _safe(sy, SY_LO, SY_HI)

    combos = n_vals(x) * n_vals(ly) * n_vals(sy)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("X",  *x),
            ParamAxis("LY", *ly),
            ParamAxis("SY", *sy),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"{PREFIX}{name}_raw.csv"


def run_or_load(name, cfg, conn, from_csv):
    csv_path = csv_for(name)
    if from_csv or csv_path.exists():
        if csv_path.exists():
            try:
                df = mc.load_results_csv(str(csv_path), cfg)
                log.info("Loaded %s: %d rows", name, len(df))
                return df
            except Exception as e:
                log.warning("Could not load %s: %s", csv_path, e)
        else:
            log.warning("No CSV for %s", name)
        return None
    log.info("=== Starting %s%s (%d combos) ===", PREFIX, name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done %.1f min -- %s", (time.time() - t0) / 60, Path(raw_csv).name)
        return mc.load_results_csv(raw_csv, cfg)
    except Exception as e:
        log.error("  FAILED: %s", e, exc_info=True)
        return None


def _validate_df(df, cfg):
    if df is None or df.empty:
        return False
    for p in cfg.params:
        if p.name not in df.columns:
            continue
        lo = min(p.start, p.stop) - abs(p.step) * 2
        hi = max(p.start, p.stop) + abs(p.step) * 2
        col = pd.to_numeric(df[p.name], errors="coerce")
        if not col.between(lo, hi).all():
            log.warning("  INVALID: %s out of [%.4g,%.4g] got [%.4g,%.4g]",
                        p.name, lo, hi, col.min(), col.max())
            return False
    return True


def champion(df, fb_x, fb_ly, fb_sy):
    """Target-chasing mode: highest NP row drives zoom seed."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ** TARGET MET: X=%.4g LY=%.4g SY=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["X"]), float(best["LY"]), float(best["SY"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["X"]), float(best["LY"]), float(best["SY"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: X=%.4g LY=%.4g SY=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["X"]), float(best["LY"]), float(best["SY"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["X"]), float(best["LY"]), float(best["SY"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_x, fb_ly, fb_sy,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, x, ly, sy, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "X": x, "LY": ly, "SY": sy,
        "objective": obj, "net_profit": np_, "max_drawdown": mdd,
        "total_trades": trades, "target_met": met,
        "combos": combos,
        "rows": len(df) if df is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }


def save_json(best_entry, attempt_log, target_met):
    payload = {
        "round": 3,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY NEXT BAR C[X]*(1+LY*0.01) STOP; SELLSHORT NEXT BAR C[X]*(1-SY*0.01) STOP",
            "exits":       "Reversal only -- no STP or LMT",
            "r2_champion": "X=7 LY=0.60 SY=1.80 NP=426,580 MDD=-46,720 trades=206 (gap -46.7%)",
            "r3_focus":    "step=0.01 precision (A11 failed R2); SY>1.8; X=20-100; confirm ceiling",
            "history":     "R1 X=1 was artifact of step-5 X grid; R2 revealed X=7 at unit-step",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_gc_xtreme_hourly3.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_x  = SEED_X
    best_ly = SEED_LY
    best_sy = SEED_SY
    best_np  = SEED_NP
    best_obj = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("=" * 62)
    log.info("  SFJ_XtremeStop_NQ on CME.GC HOT Hourly -- Round 3")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD  (800,000)", TARGET_NP)
    log.info("  R2 champion: X=%.4g  LY=%.4g  SY=%.4g  NP=%.0f",
             SEED_X, SEED_LY, SEED_SY, SEED_NP)
    log.info("=" * 62)

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_x, best_ly, best_sy
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_x, best_ly, best_sy,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-22s]  no valid data", attempt_num, name)
            return

        x, ly, sy, obj, np_, mdd, tr, met = champion(
            df, best_x, best_ly, best_sy)

        if np_ > best_np:
            best_x  = x
            best_ly = ly
            best_sy = sy
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, x, ly, sy,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-22s]  X=%.4g LY=%.4g SY=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, x, ly, sy, obj, np_, mdd, tr,
                 "**TARGET**" if met else ("%.0f/800K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
                 best_np, max(0, TARGET_NP - best_np), best_x, best_ly, best_sy)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # -----------------------------------------------------------------------
    # A01  ultra_fine — X(5-9 s1)xLY(0.52-0.68 s0.01)xSY(1.66-1.94 s0.01) = 5x17x29=2465
    # Fill R2-A11 gap: step=0.01 ultra-fine precision around champion
    # -----------------------------------------------------------------------
    A = 1
    _n = "01_ultra_fine"
    _c = _cfg(_n, (5, 9, 1), (0.52, 0.68, 0.01), (1.66, 1.94, 0.01))
    log.info("A01  X(5-9 s1)xLY(0.52-0.68 s0.01)xSY(1.66-1.94 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A02  sy_extended — X(1-8 s1)xLY(0.5-1.0 s0.1)xSY(1.5-6.0 s0.25) = 8x6x19=912
    # Retry R2-A03 (wide_sy_x1) that failed: test SY>1.8 at small X
    # -----------------------------------------------------------------------
    A = 2
    _n = "02_sy_extended"
    _c = _cfg(_n, (1, 8, 1), (0.5, 1.0, 0.1), (1.5, 6.0, 0.25))
    log.info("A02  X(1-8 s1)xLY(0.5-1.0 s0.1)xSY(1.5-6.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A03  x_compare — X(1-10 s1)xLY(0.3-1.2 s0.1)xSY(1.0-3.0 s0.1) = 10x10x21=2100
    # Direct X=1 vs X=7 comparison at equal fine resolution with wide LY/SY
    # -----------------------------------------------------------------------
    A = 3
    _n = "03_x_compare"
    _c = _cfg(_n, (1, 10, 1), (0.3, 1.2, 0.1), (1.0, 3.0, 0.1))
    log.info("A03  X(1-10 s1)xLY(0.3-1.2 s0.1)xSY(1.0-3.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A04  high_x_confirm — X(20-100 s5)xLY(0.4-1.0 s0.1)xSY(1.0-3.0 s0.25) = 17x7x9=1071
    # Retry R2-A07 that failed twice: confirm X=20-100 not competitive
    # -----------------------------------------------------------------------
    A = 4
    _n = "04_high_x_confirm"
    _c = _cfg(_n, (20, 100, 5), (0.4, 1.0, 0.1), (1.0, 3.0, 0.25))
    log.info("A04  X(20-100 s5)xLY(0.4-1.0 s0.1)xSY(1.0-3.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A05  fine_sy_sweep — X(5-9 s1)xLY(0.4-0.8 s0.05)xSY(1.5-2.5 s0.02) = 5x9x51=2295
    # Fine SY step=0.02 across wider SY range: is SY>1.8 better at X=7?
    # -----------------------------------------------------------------------
    A = 5
    _n = "05_fine_sy_sweep"
    _c = _cfg(_n, (5, 9, 1), (0.4, 0.8, 0.05), (1.5, 2.5, 0.02))
    log.info("A05  X(5-9 s1)xLY(0.4-0.8 s0.05)xSY(1.5-2.5 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A06  x_boundary — X(1-15 s1)xLY(0.5-0.7 s0.05)xSY(1.6-2.0 s0.05) = 15x5x9=675
    # Precise X boundary: confirm X=7 vs neighbours at fine LY/SY
    # -----------------------------------------------------------------------
    A = 6
    _n = "06_x_boundary"
    _c = _cfg(_n, (1, 15, 1), (0.5, 0.7, 0.05), (1.6, 2.0, 0.05))
    log.info("A06  X(1-15 s1)xLY(0.5-0.7 s0.05)xSY(1.6-2.0 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A07  global_fine — X(1-50 s2)xLY(0.4-0.8 s0.1)xSY(1.2-2.4 s0.2) = 25x5x7=875
    # Broader global confirm with moderate precision
    # -----------------------------------------------------------------------
    A = 7
    _n = "07_global_fine"
    _c = _cfg(_n, (1, 50, 2), (0.4, 0.8, 0.1), (1.2, 2.4, 0.2))
    log.info("A07  X(1-50 s2)xLY(0.4-0.8 s0.1)xSY(1.2-2.4 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A08  wide_sy_check — X(5-10 s1)xLY(0.3-0.8 s0.1)xSY(2.0-8.0 s0.25) = 6x6x25=900
    # Test high SY (>2%) at champion X range: R1-A08 found X=55,SY=4 regime
    # -----------------------------------------------------------------------
    A = 8
    _n = "08_wide_sy_check"
    _c = _cfg(_n, (5, 10, 1), (0.3, 0.8, 0.1), (2.0, 8.0, 0.25))
    log.info("A08  X(5-10 s1)xLY(0.3-0.8 s0.1)xSY(2.0-8.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A09  adaptive_zoom1 — X+-10 s2, LY+-0.50 s0.1, SY+-0.50 s0.1
    # -----------------------------------------------------------------------
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,  10.0,  2.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.50,  0.1, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.50,  0.1, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A09  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A10  adaptive_zoom2 — X+-5 s1, LY+-0.20 s0.02, SY+-0.20 s0.02
    # -----------------------------------------------------------------------
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   5.0,  1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.20, 0.02, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.20, 0.02, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A10  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A11  adaptive_zoom3 — X+-3 s1, LY+-0.10 s0.01, SY+-0.10 s0.01
    # -----------------------------------------------------------------------
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   3.0,  1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.10, 0.01, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.10, 0.01, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A11  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    log.info("=" * 62)
    log.info("  SFJ_XtremeStop_NQ CME.GC HOT Hourly Round-3 COMPLETE")
    log.info("  Champion: X=%.4g  LY=%.4g  SY=%.4g", best_x, best_ly, best_sy)
    log.info("  Best NP: %.0f USD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    log.info("  R2 champion: 426,580  R2->R3 gain: +%.2f%%",
             100 * (best_np - SEED_NP) / SEED_NP if best_np > SEED_NP else 0)
    log.info("  Target 800,000 USD: %s", "** MET" if target_met
             else f"NOT MET (gap +{max(0, TARGET_NP-best_np):,.0f})")
    log.info("=" * 62)

    if not best_entry:
        best_entry = {
            "X": best_x, "LY": best_ly, "SY": best_sy,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone -- results at: {out}")
    print(f"R3 best: NP={best_np:,.0f} USD  (X={best_x} LY={best_ly} SY={best_sy})")
    print(f"R2->R3 gain: +{100*(best_np-SEED_NP)/SEED_NP:.2f}%" if best_np > SEED_NP else "R2->R3 gain: 0%")
    print(f"Target NP>800,000 USD: {'MET' if target_met else 'NOT MET'}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _auto_elevate():
    script  = str(Path(__file__).resolve())
    workdir = str(Path(__file__).resolve().parent)
    extra   = [a for a in sys.argv[1:] if a != "--_elevated"]
    quoted  = [f'"{a}"' if " " in a else a for a in extra]
    all_args = f'"{script}" ' + " ".join(quoted) + " --_elevated"
    print("[auto-elevate] Requesting elevation -- approve the UAC prompt.")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Administrator manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="SFJ_XtremeStop_NQ CME.GC HOT Hourly NP>800K USD Round-3 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1-11)")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0

    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
