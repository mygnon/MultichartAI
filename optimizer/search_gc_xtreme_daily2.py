"""
search_gc_xtreme_daily2.py — SFJ_XtremeStop_NQ on CME.GC HOT Daily, Round 2

R1 champion (A01 partial, 795/2560 rows): X=1, LY=0.50, SY=2.0, NP=306,870, MDD=-64,970, trades=200
R1 best Obj (A09 complete): X=1, LY=1.105, SY=2.8, NP=288,050, MDD=-37,310, Obj=2,223,876, trades=37

R1 key findings:
  - X=1 uniquely dominant; large X (650) → NP=7,320; medium X (30) → NP=167,410
  - Flat/noisy landscape at X=1: A09/A10/A11 zooms converged to 3 different champions
    (1.105/2.8, 1.0/1.95, 0.38/2.12) -- grid-shift artifact from LY_LO=0.005 offset
  - A01 result (306,870) came from partial 31% export; zooms couldn't reproduce it
    because step=0.1 grids starting at 0.005 don't hit LY=0.50 or LY=0.60 exactly
  - A03 low-trade regime: X=2, LY=0.5, SY=4.5, NP=271,010, 11 trades, Obj=1.76M
  - A05 high_pct UI-failed -- unexplored territory
  - Near-zero LY (A06): X=1, LY=0.005, SY=3.0, NP=269,190 -- viable, lower NP

R2 focus:
  - Aligned grids starting at LY=0.5/SY=1.0 to correctly hit R1 champion coordinates
  - Fine X=1-10 unit-step with clean LY/SY sweep to find true peak
  - High-SY regime: X=1-10, SY=3-8 (inspired by A03 X=2 SY=4.5)
  - Near-zero LY precision (NQ Daily pattern: LY=0.025)
  - Retry high_pct (A05 UI-failed)
  - Confirm X>5 is truly inferior
  - Stable adaptive zooms using aligned step grids

R2 attempt schedule (11 attempts, <=5,000 combos each):
  A01 fine_x_aligned  : X(1-10 s1)xLY(0.5-2.5 s0.1)xSY(1.5-3.5 s0.1)        = 10x21x21=4410
  A02 near_zero_ly    : X(1-5 s1)xLY(0.005-0.45 s0.025)xSY(1.5-5.0 s0.25)    = 5x19x15=1425
  A03 high_sy_regime  : X(1-10 s1)xLY(0.3-1.5 s0.1)xSY(3.0-8.0 s0.25)        = 10x13x21=2730
  A04 retry_high_pct  : X(1-181 s20)xLY(5-15 s0.5)xSY(5-15 s0.5)             = 10x21x21=4410
  A05 x1_sy_fine      : X(1-5 s1)xLY(0.3-1.5 s0.1)xSY(1.5-5.0 s0.1)          = 5x13x36=2340
  A06 global_confirm  : X(1-100 s5)xLY(0.5-3.0 s0.5)xSY(1.0-5.0 s0.5)        = 20x6x9=1080
  A07 asym_extreme    : X(1-5 s1)xLY(0.005-0.1 s0.005)xSY(2.0-8.0 s0.25)     = 5x20x25=2500
  A08 x_boundary_fine : X(1-20 s1)xLY(0.3-1.5 s0.2)xSY(1.5-4.0 s0.25)        = 20x7x11=1540
  A09 adaptive_zoom1  : X+-10 s2, LY+-1.0 s0.2, SY+-1.0 s0.2                  <= 11x11x11=1331
  A10 adaptive_zoom2  : X+-3  s1, LY+-0.5 s0.05, SY+-0.5 s0.05               <= 7x21x21=3087
  A11 adaptive_zoom3  : X+-2  s1, LY+-0.2 s0.02, SY+-0.2 s0.02               <= 5x21x21=2205
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_xtreme_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0   # USD

X_LO,  X_HI  = 1.0,   1000.0
LY_LO, LY_HI = 0.005, 50.0
SY_LO, SY_HI = 0.005, 50.0

SEED_X  = 1.0
SEED_LY = 0.50
SEED_SY = 2.0
SEED_NP = 306_870.0

PREFIX = "GCXD2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_xtreme_daily2_{int(time.time())}.log"
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
        timeframe="daily",
        bar_period=1440,
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
        "round": 2,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY NEXT BAR C[X]*(1+LY*0.01) STOP; SELLSHORT NEXT BAR C[X]*(1-SY*0.01) STOP",
            "exits":       "Reversal only -- no STP or LMT",
            "r1_champion": "X=1 LY=0.50 SY=2.0 NP=306,870 MDD=-64,970 trades=200 (partial 795/2560 rows; gap -61.6%)",
            "r1_best_obj": "X=1 LY=1.105 SY=2.8 NP=288,050 MDD=-37,310 Obj=2,223,876 trades=37 (complete result)",
            "r2_focus":    "Aligned grids to hit R1 peak; fine X=1-10; high-SY regime (X=2 SY=4.5); retry high_pct",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_gc_xtreme_daily2.json"
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
    log.info("  SFJ_XtremeStop_NQ on CME.GC HOT Daily -- Round 2")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD  Timeframe: Daily", TARGET_NP)
    log.info("  R1 champion: X=%.4g  LY=%.4g  SY=%.4g  NP=%.0f",
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
    # A01  fine_x_aligned — X(1-10 s1)xLY(0.5-2.5 s0.1)xSY(1.5-3.5 s0.1) = 10x21x21=4410
    # ALIGNED grids: LY starts at 0.5 (hits R1 champion LY=0.5 exactly)
    # SY starts at 1.5 (hits R1 champion SY=2.0 exactly at step 0.1)
    # -----------------------------------------------------------------------
    A = 1
    _n = "01_fine_x_aligned"
    _c = _cfg(_n, (1, 10, 1), (0.5, 2.5, 0.1), (1.5, 3.5, 0.1))
    log.info("A01  X(1-10 s1)xLY(0.5-2.5 s0.1)xSY(1.5-3.5 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A02  near_zero_ly — X(1-5 s1)xLY(0.005-0.45 s0.025)xSY(1.5-5.0 s0.25) = 5x19x15=1425
    # NQ Daily pattern: LY=0.025 SY=3.7; test near-zero LY precision at small X
    # -----------------------------------------------------------------------
    A = 2
    _n = "02_near_zero_ly"
    _c = _cfg(_n, (1, 5, 1), (0.005, 0.455, 0.025), (1.5, 5.0, 0.25))
    log.info("A02  X(1-5 s1)xLY(0.005-0.455 s0.025)xSY(1.5-5.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A03  high_sy_regime — X(1-10 s1)xLY(0.3-1.5 s0.1)xSY(3.0-8.0 s0.25) = 10x13x21=2730
    # R1-A03 found X=2 SY=4.5 NP=271K 11 trades: explore high SY territory
    # -----------------------------------------------------------------------
    A = 3
    _n = "03_high_sy_regime"
    _c = _cfg(_n, (1, 10, 1), (0.3, 1.5, 0.1), (3.0, 8.0, 0.25))
    log.info("A03  X(1-10 s1)xLY(0.3-1.5 s0.1)xSY(3.0-8.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A04  retry_high_pct — X(1-181 s20)xLY(5-15 s0.5)xSY(5-15 s0.5) = 10x21x21=4410
    # Retry R1-A05 that UI-failed: TXF Hourly used LY=5.155 SY=5.79
    # -----------------------------------------------------------------------
    A = 4
    _n = "04_retry_high_pct"
    _c = _cfg(_n, (1, 181, 20), (5.0, 15.0, 0.5), (5.0, 15.0, 0.5))
    log.info("A04  X(1-181 s20)xLY(5-15 s0.5)xSY(5-15 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A05  x1_sy_fine — X(1-5 s1)xLY(0.3-1.5 s0.1)xSY(1.5-5.0 s0.1) = 5x13x36=2340
    # Fine SY step=0.1 sweep at small X; SY=4.5 from A03 and SY=2.0 from A01 both covered
    # -----------------------------------------------------------------------
    A = 5
    _n = "05_x1_sy_fine"
    _c = _cfg(_n, (1, 5, 1), (0.3, 1.5, 0.1), (1.5, 5.0, 0.1))
    log.info("A05  X(1-5 s1)xLY(0.3-1.5 s0.1)xSY(1.5-5.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A06  global_confirm — X(1-100 s5)xLY(0.5-3.0 s0.5)xSY(1.0-5.0 s0.5) = 20x6x9=1080
    # Confirm no undiscovered X regime exists globally
    # -----------------------------------------------------------------------
    A = 6
    _n = "06_global_confirm"
    _c = _cfg(_n, (1, 96, 5), (0.5, 3.0, 0.5), (1.0, 5.0, 0.5))
    log.info("A06  X(1-96 s5)xLY(0.5-3.0 s0.5)xSY(1.0-5.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A07  asym_extreme — X(1-5 s1)xLY(0.005-0.1 s0.005)xSY(2.0-8.0 s0.25) = 5x20x25=2500
    # Ultra-near-zero LY precision; inspired by NQ Daily (LY=0.025 SY=3.7)
    # -----------------------------------------------------------------------
    A = 7
    _n = "07_asym_extreme"
    _c = _cfg(_n, (1, 5, 1), (0.005, 0.100, 0.005), (2.0, 8.0, 0.25))
    log.info("A07  X(1-5 s1)xLY(0.005-0.1 s0.005)xSY(2.0-8.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A08  x_boundary_fine — X(1-20 s1)xLY(0.3-1.5 s0.2)xSY(1.5-4.0 s0.25) = 20x7x11=1540
    # Fine X=1-20 unit-step with moderate LY/SY: confirm X=1 vs X=2-20
    # -----------------------------------------------------------------------
    A = 8
    _n = "08_x_boundary_fine"
    _c = _cfg(_n, (1, 20, 1), (0.3, 1.5, 0.2), (1.5, 4.0, 0.25))
    log.info("A08  X(1-20 s1)xLY(0.3-1.5 s0.2)xSY(1.5-4.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A09  adaptive_zoom1 — X+-10 s2, LY+-1.0 s0.2, SY+-1.0 s0.2
    # -----------------------------------------------------------------------
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,  10.0,  2.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  1.0,   0.2, LY_LO, LY_HI)
        _sy = zoom(best_sy,  1.0,   0.2, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A09  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A10  adaptive_zoom2 — X+-3 s1, LY+-0.5 s0.05, SY+-0.5 s0.05
    # -----------------------------------------------------------------------
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   3.0,  1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.50, 0.05, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.50, 0.05, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A10  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A11  adaptive_zoom3 — X+-2 s1, LY+-0.2 s0.02, SY+-0.2 s0.02
    # -----------------------------------------------------------------------
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   2.0,  1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.20, 0.02, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.20, 0.02, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A11  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    log.info("=" * 62)
    log.info("  SFJ_XtremeStop_NQ CME.GC HOT Daily Round-2 COMPLETE")
    log.info("  Champion: X=%.4g  LY=%.4g  SY=%.4g", best_x, best_ly, best_sy)
    log.info("  Best NP: %.0f USD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    log.info("  R1 champion: 306,870  R1->R2 gain: +%.2f%%",
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
    print(f"R2 best: NP={best_np:,.0f} USD  (X={best_x} LY={best_ly} SY={best_sy})")
    print(f"R1->R2 gain: +{100*(best_np-SEED_NP)/SEED_NP:.2f}%" if best_np > SEED_NP else "R1->R2 gain: 0%")
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
        description="SFJ_XtremeStop_NQ CME.GC HOT Daily NP>800K USD Round-2 search")
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
