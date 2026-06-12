"""
search_txf_xtreme_daily2.py — SFJ_XtremeStop_NQ on TWF.TXF HOT Daily, Round 2

R1 champion: X=1  LY=3.1  SY=4.26  NP=3,998,400 TWD  MDD=-1,019,000  trades=12
R1 gain trajectory: A01→A09→A11 = 3,270K→3,650K→3,998K (+9.3% last jump — NOT converged)

R1 critical gaps:
  - LY=3.1 was at the LEFT BOUNDARY of zoom3 (tested LY=3.1-3.5); LY<3.1 completely unknown
  - SY=4.26 inside zoom3 range (4.06-4.46); SY>4.5 at fine step unexplored
  - X=2,3,4 never tested (A03 used step=5: X=1,6,11...)
  - A02 (X=1000-2500) UI-failed — large lookback completely unknown
  - X=100-900 gap between A01 (X=5-1005 step=100) and A02
  - A10 (zoom2) UI-failed — A11 used zoom1 seed, may have missed finer optimum
  - Tight LY/SY (0.1-2.5%) at X=1 never tested (A04 only tested with X=5+)

Attempt schedule (11 attempts, ≤5,000 combos each):
  A01 retry_large_x  : X(1000-2500 s150)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5)=11×20×20=4400
  A02 x_single_fine  : X(1-9 s1)×LY(2.5-4.5 s0.2)×SY(3.5-5.5 s0.2)     =  9×11×11=1089
  A03 ly_low_bound   : X(1-5 s1)×LY(0.5-3.2 s0.1)×SY(3.5-5.5 s0.2)      =  5×28×11=1540
  A04 sy_high_bound  : X(1-5 s1)×LY(2.5-4.0 s0.1)×SY(4.0-7.0 s0.1)      =  5×16×31=2480
  A05 asym_extreme   : X(1-5 s1)×LY(0.5-2.5 s0.1)×SY(4.0-8.0 s0.2)      =  5×21×21=2205
  A06 medium_x_fine  : X(5-50 s5)×LY(1.5-5.0 s0.25)×SY(1.5-5.5 s0.25)   = 10×15×17=2550
  A07 global_bridge  : X(100-1000 s100)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5) = 10×20×20=4000
  A08 x1_tight       : X(1-5 s1)×LY(0.1-2.5 s0.1)×SY(0.1-3.5 s0.1)      =  5×25×35=4375
  A09 adaptive_zoom1 : X±20 s5, LY±0.75 s0.1, SY±0.75 s0.1             → ≤9×16×16=2304
  A10 adaptive_zoom2 : X±10 s2, LY±0.50 s0.05, SY±0.50 s0.05           → ≤11×21×21=4851
  A11 ultra_fine_final: X±5 s1, LY±0.20 s0.02, SY±0.20 s0.02           → ≤11×21×21=4851
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


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\SFJ_XtremeStop_AI.wsp"
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "SFJ_XtremeStop_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_xtreme_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0   # TWD

X_LO,  X_HI  = 1.0,   3000.0
LY_LO, LY_HI = 0.01,  50.0
SY_LO, SY_HI = 0.01,  50.0

# Seed from R1 champion
SEED_X  = 1.0
SEED_LY = 3.1
SEED_SY = 4.26
SEED_NP = 3_998_400.0

PREFIX = "TXFXD2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_xtreme_daily2_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        log.info("  Done %.1f min — %s", (time.time() - t0) / 60, Path(raw_csv).name)
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
        log.info("  ★ TARGET MET: X=%.4g LY=%.4g SY=%.4g  "
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
            "exits":       "Reversal only — no STP or LMT",
            "defaults":    "X=150, LY=3.25, SY=3.25",
            "r1_champion": "X=1 LY=3.1 SY=4.26 NP=3,998,400 MDD=-1,019,000 trades=12 (R1→R2 gap: -50%)",
            "r2_focus":    "LY<3.1 boundary; SY>4.46; X=2-9 unit; large X retry; tight LY/SY at X=1; bridge X=100-1000",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_xtreme_daily2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

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

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_XtremeStop_NQ on TWF.TXF HOT Daily — Round 2")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f TWD  (8,000,000)", TARGET_NP)
    log.info("  R1 champion: X=%.4g  LY=%.4g  SY=%.4g  NP=%.0f",
             SEED_X, SEED_LY, SEED_SY, SEED_NP)
    log.info("  R2 focus: LY<3.1 boundary; SY>4.46; X=2-9; large X retry; bridge X=100-1000")
    log.info("══════════════════════════════════════════════════════════════")

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
                 "★TARGET★" if met else ("%.0f/8M" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
                 best_np, max(0, TARGET_NP - best_np), best_x, best_ly, best_sy)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  retry_large_x — X=1000-2500 daily bars (4-10 year lookback)
    #      Retry of R1-A02 which UI-failed. LY/SY wide range.
    #      X(1000-2500 s150)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5) = 11×20×20=4400
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_retry_large_x"
    _c = _cfg(_n, (1000, 2500, 150), (0.5, 10.0, 0.5), (0.5, 10.0, 0.5))
    log.info("A01  X(1000-2500 s150)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A02  x_single_fine — X=1-9 unit step (test X=2,3,4 which R1 missed)
    #      R1-A03 used X step=5, missing X=2,3,4. Champion zone LY/SY.
    #      X(1-9 s1)×LY(2.5-4.5 s0.2)×SY(3.5-5.5 s0.2) = 9×11×11=1089
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_x_single_fine"
    _c = _cfg(_n, (1, 9, 1), (2.5, 4.5, 0.2), (3.5, 5.5, 0.2))
    log.info("A02  X(1-9 s1)×LY(2.5-4.5 s0.2)×SY(3.5-5.5 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ly_low_bound — explore LY<3.1 which was boundary in R1-zoom3
    #      LY=3.1 was the LEFT BOUNDARY of zoom3 (tested 3.1-3.5); must go lower.
    #      X(1-5 s1)×LY(0.5-3.2 s0.1)×SY(3.5-5.5 s0.2) = 5×28×11=1540
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ly_low_bound"
    _c = _cfg(_n, (1, 5, 1), (0.5, 3.2, 0.1), (3.5, 5.5, 0.2))
    log.info("A03  X(1-5 s1)×LY(0.5-3.2 s0.1)×SY(3.5-5.5 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A04  sy_high_bound — explore SY>4.46 which was boundary in R1-zoom3
    #      zoom3 SY range was 4.06-4.46; champion at 4.26 (inside); test wider SY.
    #      X(1-5 s1)×LY(2.5-4.0 s0.1)×SY(4.0-7.0 s0.1) = 5×16×31=2480
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_sy_high_bound"
    _c = _cfg(_n, (1, 5, 1), (2.5, 4.0, 0.1), (4.0, 7.0, 0.1))
    log.info("A04  X(1-5 s1)×LY(2.5-4.0 s0.1)×SY(4.0-7.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A05  asym_extreme — very low LY + high SY (asymmetric short-heavy)
    #      Explores regime where long bar is low (1-2%) and short bar is high (4-8%)
    #      X(1-5 s1)×LY(0.5-2.5 s0.1)×SY(4.0-8.0 s0.2) = 5×21×21=2205
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_asym_extreme"
    _c = _cfg(_n, (1, 5, 1), (0.5, 2.5, 0.1), (4.0, 8.0, 0.2))
    log.info("A05  X(1-5 s1)×LY(0.5-2.5 s0.1)×SY(4.0-8.0 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A06  medium_x_fine — X=5-50 step=5 with finer LY/SY
    #      R1-A01 used step=100 (X=5,105,...), R1-A03 step=5 (X=1,6,11...);
    #      this fills X=5-50 at step=5 with finer LY/SY than R1.
    #      X(5-50 s5)×LY(1.5-5.0 s0.25)×SY(1.5-5.5 s0.25) = 10×15×17=2550
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_medium_x_fine"
    _c = _cfg(_n, (5, 50, 5), (1.5, 5.0, 0.25), (1.5, 5.5, 0.25))
    log.info("A06  X(5-50 s5)×LY(1.5-5.0 s0.25)×SY(1.5-5.5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A07  global_bridge — fill gap X=100-1000 (between R1-A01 and A01 large_x)
    #      R1-A01 best was X=5; R1 large_x (A02) failed; X=100-900 unknown.
    #      X(100-1000 s100)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5) = 10×20×20=4000
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_global_bridge"
    _c = _cfg(_n, (100, 1000, 100), (0.5, 10.0, 0.5), (0.5, 10.0, 0.5))
    log.info("A07  X(100-1000 s100)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A08  x1_tight — tight LY/SY (0.1-3.5%) at X=1-5
    #      R1-A04 only tested tight LY/SY with X=5+ (step=100). This tests
    #      tight entry at X=1 specifically (lower bar but more frequent entries).
    #      X(1-5 s1)×LY(0.1-2.5 s0.1)×SY(0.1-3.5 s0.1) = 5×25×35=4375
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_x1_tight"
    _c = _cfg(_n, (1, 5, 1), (0.1, 2.5, 0.1), (0.1, 3.5, 0.1))
    log.info("A08  X(1-5 s1)×LY(0.1-2.5 s0.1)×SY(0.1-3.5 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — X±20 s5, LY±0.75 s0.1, SY±0.75 s0.1
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,  20.0,  5.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.75,  0.1, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.75,  0.1, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A09  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — X±10 s2, LY±0.50 s0.05, SY±0.50 s0.05
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,  10.0,  2.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.50, 0.05, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.50, 0.05, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A10  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A11  ultra_fine_final — X±5 s1, LY±0.20 s0.02, SY±0.20 s0.02
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_ultra_fine_final"
    log.info("A11  ultra_fine_final — center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   5.0,  1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.20, 0.02, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.20, 0.02, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A11  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_XtremeStop_NQ TWF.TXF HOT Daily Round-2 COMPLETE")
    log.info("  Champion: X=%.4g  LY=%.4g  SY=%.4g", best_x, best_ly, best_sy)
    log.info("  Best NP: %.0f TWD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    r1_np = 3_998_400.0
    log.info("  R1→R2 gain: %.3f%%", (best_np - r1_np) / r1_np * 100)
    log.info("  Target 8,000,000 TWD: %s", "★ MET" if target_met
             else f"NOT MET (gap +{max(0, TARGET_NP-best_np):,.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "X": best_x, "LY": best_ly, "SY": best_sy,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"R2 best: NP={best_np:,.0f} TWD  (X={best_x} LY={best_ly} SY={best_sy})")
    r1_np = 3_998_400.0
    print(f"R1→R2 gain: {(best_np - r1_np)/r1_np*100:+.3f}%")
    print(f"Target NP>8,000,000 TWD: {'MET ✅' if target_met else 'NOT MET'}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

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
    print("[auto-elevate] Requesting elevation — approve the UAC prompt.")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Administrator manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="SFJ_XtremeStop_NQ TWF.TXF HOT Daily NP>8M TWD Round-2 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1–11)")
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
