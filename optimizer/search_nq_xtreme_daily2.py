"""
search_nq_xtreme_daily2.py — SFJ_XtremeStop_NQ on CME.NQ HOT Daily, Round 2

R1 champion: X=9  LY=0.03  SY=3.7  NP=490,840  MDD=-108,285  trades=75
R1 gap: -38.6% from 800K target
R1 gain path: 295K->399K->414K->486K(+17.2% jump)->491K(+0.9%)

R1 key findings:
  - LY converges to near-zero (0.03%) — unique regime: long enters whenever price
    is barely above 9-day-ago close; SY=3.7% provides asymmetric short exit only
    on significant drops
  - A11 grid step=0.02 means LY = 0.01, 0.03, 0.05... LY=0.02 was NEVER TESTED
  - A10->A11 gain only +0.9% — may be near ceiling, but not yet confirmed 0%
  - X=9 confirmed best in X(4-14) range; X=8 and X=10 at unit step not yet precise
  - SY=3.7 stable across zooms but step=0.1 — SY=3.65/3.75 etc not yet tested
  - A01 global_wide only 554/2816 rows — most param combos produce 0 trades on NQ daily
  - 75 trades/7yr (~10.7/yr); structural ceiling concern: 75 x $6.5K = $491K
  - High-pct (5-15%) does produce daily trades (2312 rows) but NP=319K < regime champion
  - X=1 (TXF Daily analogy) not separately tested; A04 covered X=1 but at coarse LY/SY

R2 focus:
  - A01 xy_fine: X=5-15 unit + LY=0.01-0.15 step=0.01 + SY=3.3-4.5 (fill gaps)
  - A02 sy_fine: SY at step=0.05 around 3.7 (is 3.70 the true peak?)
  - A03 x1_scan: X=1-5 with broader LY/SY (TXF Daily analogy: X=1 regime?)
  - A04 broad_ly0: X=1-30 step=2 with ultra-low LY; confirm X landscape
  - A05 high_trades: Low SY to get more reversals; can more trades compensate?
  - A06 global_confirm: X=1-100 with low-LY focus; any hidden regime?
  - A07 medium_pct: LY=0.5-2.5 with various SY; is LY=0.75 regime improvable?
  - A08 large_sy: Very high SY=4-8 with low LY; fewer trades but bigger per-trade?
  - A09-A11: adaptive zoom -> confirm ceiling via convergence

Attempt schedule (11 attempts, <=5000 combos each):
  A01 xy_fine        : X(5-15 s1)xLY(0.01-0.15 s0.01)xSY(3.3-4.5 s0.1)  = 11x15x13=2145
  A02 sy_fine        : X(7-12 s1)xLY(0.01-0.1 s0.01)xSY(3.4-4.1 s0.05)  = 6x10x15=900
  A03 x1_scan        : X(1-5 s1)xLY(0.5-4 s0.25)xSY(1-8 s0.25)           = 5x15x29=2175
  A04 broad_ly0      : X(1-30 s2)xLY(0.01-0.1 s0.01)xSY(2.5-5.5 s0.25)  = 15x10x13=1950
  A05 high_trades    : X(1-15 s2)xLY(0.01-0.1 s0.01)xSY(1.0-3.5 s0.25)  = 8x10x11=880
  A06 global_confirm : X(1-100 s5)xLY(0.01-0.1 s0.01)xSY(2.0-6.0 s0.5)  = 20x10x9=1800
  A07 medium_pct     : X(5-20 s1)xLY(0.5-2.5 s0.25)xSY(3.0-5.5 s0.25)   = 16x9x11=1584
  A08 large_sy       : X(5-15 s2)xLY(0.01-0.1 s0.01)xSY(4.0-8.0 s0.25)  = 6x10x17=1020
  A09 adaptive_zoom1 : X+-20 s5, LY+-0.75 s0.1, SY+-0.75 s0.1            <= 9x16x16=2304
  A10 adaptive_zoom2 : X+-10 s2, LY+-0.05 s0.01, SY+-0.50 s0.05          <= 11x11x21=2541
  A11 adaptive_zoom3 : X+-5  s1, LY+-0.02 s0.005, SY+-0.20 s0.02         <= 11x9x21=2079
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
SYMBOL     = "CME.NQ HOT"
SIGNAL     = "SFJ_XtremeStop_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_xtreme_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0   # USD

X_LO,  X_HI  = 1.0,   10000.0
LY_LO, LY_HI = 0.005, 50.0
SY_LO, SY_HI = 0.01,  50.0

# R1 champion seed
SEED_X  = 9.0
SEED_LY = 0.03
SEED_SY = 3.7
SEED_NP = 490_840.0

PREFIX = "NQXD2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_xtreme_daily2_{int(time.time())}.log"
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
            "logic":    "BUY NEXT BAR C[X]*(1+LY*0.01) STOP; SELLSHORT NEXT BAR C[X]*(1-SY*0.01) STOP",
            "exits":    "Reversal only -- no STP or LMT",
            "r1_champion": "X=9 LY=0.03 SY=3.7 NP=490,840 MDD=-108,285 trades=75 (gap -38.6%)",
            "r2_focus": "LY=0.02 gap fill; SY step=0.05 precision; X unit confirm; X=1 TXF-analog; global confirm",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_xtreme_daily2.json"
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
    log.info("  SFJ_XtremeStop_NQ on CME.NQ HOT Daily -- Round 2")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD  (800,000)", TARGET_NP)
    log.info("  R1 Seed: X=%.4g  LY=%.4g  SY=%.4g  NP=%.0f",
             SEED_X, SEED_LY, SEED_SY, SEED_NP)
    log.info("  NOTE: LY->0 regime; LY=0.02 gap; SY=3.7 precision; X=9 confirm")
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
    # A01  xy_fine -- fill LY=0.02 gap + confirm X and SY around champion
    #      X(5-15 s1)xLY(0.01-0.15 s0.01)xSY(3.3-4.5 s0.1) = 11x15x13=2145
    # -----------------------------------------------------------------------
    A = 1
    _n = "01_xy_fine"
    _c = _cfg(_n, (5, 15, 1), (0.01, 0.15, 0.01), (3.3, 4.5, 0.1))
    log.info("A01  X(5-15 s1)xLY(0.01-0.15 s0.01)xSY(3.3-4.5 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A02  sy_fine -- SY at step=0.05 precision (is 3.70 the exact peak?)
    #      X(7-12 s1)xLY(0.01-0.1 s0.01)xSY(3.4-4.1 s0.05) = 6x10x15=900
    # -----------------------------------------------------------------------
    A = 2
    _n = "02_sy_fine"
    _c = _cfg(_n, (7, 12, 1), (0.01, 0.1, 0.01), (3.4, 4.1, 0.05))
    log.info("A02  X(7-12 s1)xLY(0.01-0.1 s0.01)xSY(3.4-4.1 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A03  x1_scan -- TXF Daily found X=1 uniquely optimal; check NQ daily
    #      X(1-5 s1)xLY(0.5-4 s0.25)xSY(1-8 s0.25) = 5x15x29=2175
    # -----------------------------------------------------------------------
    A = 3
    _n = "03_x1_scan"
    _c = _cfg(_n, (1, 5, 1), (0.5, 4.0, 0.25), (1.0, 8.0, 0.25))
    log.info("A03  X(1-5 s1)xLY(0.5-4 s0.25)xSY(1-8 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A04  broad_ly0 -- X=1-30 step=2 with ultra-low LY; confirm X landscape
    #      X(1-30 s2)xLY(0.01-0.1 s0.01)xSY(2.5-5.5 s0.25) = 15x10x13=1950
    # -----------------------------------------------------------------------
    A = 4
    _n = "04_broad_ly0"
    _c = _cfg(_n, (1, 30, 2), (0.01, 0.1, 0.01), (2.5, 5.5, 0.25))
    log.info("A04  X(1-30 s2)xLY(0.01-0.1 s0.01)xSY(2.5-5.5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A05  high_trades -- low SY for more reversals; can frequency compensate?
    #      X(1-15 s2)xLY(0.01-0.1 s0.01)xSY(1.0-3.5 s0.25) = 8x10x11=880
    # -----------------------------------------------------------------------
    A = 5
    _n = "05_high_trades"
    _c = _cfg(_n, (1, 15, 2), (0.01, 0.1, 0.01), (1.0, 3.5, 0.25))
    log.info("A05  X(1-15 s2)xLY(0.01-0.1 s0.01)xSY(1.0-3.5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A06  global_confirm -- wide X sweep with low-LY focus
    #      X(1-100 s5)xLY(0.01-0.1 s0.01)xSY(2.0-6.0 s0.5) = 20x10x9=1800
    # -----------------------------------------------------------------------
    A = 6
    _n = "06_global_confirm"
    _c = _cfg(_n, (1, 100, 5), (0.01, 0.1, 0.01), (2.0, 6.0, 0.5))
    log.info("A06  X(1-100 s5)xLY(0.01-0.1 s0.01)xSY(2.0-6.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A07  medium_pct -- moderate LY=0.5-2.5 with various SY; R1 A04 best here
    #      X(5-20 s1)xLY(0.5-2.5 s0.25)xSY(3.0-5.5 s0.25) = 16x9x11=1584
    # -----------------------------------------------------------------------
    A = 7
    _n = "07_medium_pct"
    _c = _cfg(_n, (5, 20, 1), (0.5, 2.5, 0.25), (3.0, 5.5, 0.25))
    log.info("A07  X(5-20 s1)xLY(0.5-2.5 s0.25)xSY(3.0-5.5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A08  large_sy -- very high SY with low LY; fewer trades, bigger moves?
    #      X(5-15 s2)xLY(0.01-0.1 s0.01)xSY(4.0-8.0 s0.25) = 6x10x17=1020
    # -----------------------------------------------------------------------
    A = 8
    _n = "08_large_sy"
    _c = _cfg(_n, (5, 15, 2), (0.01, 0.1, 0.01), (4.0, 8.0, 0.25))
    log.info("A08  X(5-15 s2)xLY(0.01-0.1 s0.01)xSY(4.0-8.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A09  adaptive_zoom1 -- X+-20 s5, LY+-0.75 s0.1, SY+-0.75 s0.1
    # -----------------------------------------------------------------------
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,  20.0,  5.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.75,  0.1, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.75,  0.1, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A09  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A10  adaptive_zoom2 -- X+-10 s2, LY+-0.05 s0.01, SY+-0.50 s0.05
    #      Tight LY radius since LY is already near-zero
    # -----------------------------------------------------------------------
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,  10.0,  2.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.05, 0.01, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.50, 0.05, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A10  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # -----------------------------------------------------------------------
    # A11  adaptive_zoom3 -- X+-5 s1, LY+-0.02 s0.005, SY+-0.20 s0.02
    #      Ultra-fine LY (step=0.005) since champion is at LY=0.02-0.03
    # -----------------------------------------------------------------------
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   5.0,  1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.02, 0.005, LY_LO, LY_HI)
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
    log.info("  SFJ_XtremeStop_NQ CME.NQ HOT Daily Round-2 COMPLETE")
    log.info("  Champion: X=%.4g  LY=%.4g  SY=%.4g", best_x, best_ly, best_sy)
    log.info("  Best NP: %.0f USD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    log.info("  R1 seed NP: %.0f  R2 gain: %+.1f%%",
             SEED_NP, (best_np - SEED_NP) / SEED_NP * 100 if SEED_NP else 0)
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
    print(f"R1->R2 gain: {(best_np - SEED_NP)/SEED_NP*100:+.1f}%  ({SEED_NP:,.0f} -> {best_np:,.0f})")
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
        description="SFJ_XtremeStop_NQ CME.NQ HOT Daily NP>800K USD Round-2 search")
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
