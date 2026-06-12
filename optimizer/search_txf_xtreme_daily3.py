"""
search_txf_xtreme_daily3.py — SFJ_XtremeStop_NQ on TWF.TXF HOT Daily, Round 3

R2 champion: X=1  LY=3.04  SY=4.46  NP=4,321,800 TWD  MDD=-1,041,000  trades=10
R1→R2 gain:  +8.09%  (zoom2 step=0.05 triggered trade change vs zoom1 step=0.1)
R2 internal: A10(4,321,600) ≈ A11(4,321,800) = +0.005% — LOCAL CONVERGENCE CONFIRMED

R2 confirmed regions:
  - X=1 uniquely optimal (X=2-9 worse; X=100-1000 far worse; large X structural dead zone)
  - LY<3.1 not better (A03 tested LY=0.5-3.2 step=0.1; best still LY=3.1)
  - LY=3.04 at step=0.02 precision (between step=0.1 grid points 3.0 and 3.1)
  - SY=4.46 at step=0.02 precision (A04 found SY=4.3 at step=0.1; zoom2 refined to 4.45-4.46)
  - Large X (1000-2500): 3x failure = structural data limit (IS=1750 bars < X+min_history)
  - Tight entry regime: X=1 LY=2.2 SY=3.5 → 30 trades NP=2,955,600 MDD=-756K (viable alt)

R3 focus: Ceiling CONFIRMATION via triple convergence proof.
  - Ultra-fine (step=0.01) precision around champion
  - Super-fine (step=0.005) for micro-precision in champion zone
  - Global confirm (no hidden regime)
  - SY extension above 4.5% (not tested at fine resolution)
  - LY extension below 3.04 (confirm floor)
  - Adaptive zooms will re-confirm A10≈A11

Attempt schedule (11 attempts, ≤5,000 combos each):
  A01 step01_precision : X(1-3 s1)×LY(2.9-3.2 s0.01)×SY(4.2-4.7 s0.01) = 3×31×51=4743
  A02 x_unit_confirm  : X(1-10 s1)×LY(2.5-4.0 s0.1)×SY(3.5-5.5 s0.1)   = 10×16×21=3360
  A03 global_confirm  : X(1-500 s50)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5)   = 11×20×20=4400
  A04 sy_high_scan    : X(1-5 s1)×LY(2.0-4.0 s0.1)×SY(4.5-8.0 s0.1)    =  5×21×36=3780
  A05 ly_fine_down    : X(1-5 s1)×LY(1.5-3.1 s0.05)×SY(4.2-4.7 s0.05)  =  5×33×11=1815
  A06 super_fine_005  : X(1-3 s1)×LY(3.0-3.1 s0.005)×SY(4.4-4.5 s0.005)= 3×21×21=1323
  A07 medium_x_confirm: X(2-20 s1)×LY(2.8-3.3 s0.05)×SY(4.2-4.7 s0.05) = 19×11×11=2299
  A08 x1_broad_scan   : X(1-5 s1)×LY(0.5-3.0 s0.1)×SY(0.5-4.2 s0.1)    =  5×26×38=4940
  A09 adaptive_zoom1  : X±20 s5, LY±0.75 s0.1, SY±0.75 s0.1            → ≤9×16×16=2304
  A10 adaptive_zoom2  : X±10 s2, LY±0.50 s0.05, SY±0.50 s0.05          → ≤11×21×21=4851
  A11 ultra_fine_final: X±5  s1, LY±0.20 s0.02, SY±0.20 s0.02          → ≤11×21×21=4851
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_xtreme_daily3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0   # TWD

X_LO,  X_HI  = 1.0,   3000.0
LY_LO, LY_HI = 0.01,  50.0
SY_LO, SY_HI = 0.01,  50.0

# Seed from R2 champion
SEED_X  = 1.0
SEED_LY = 3.04
SEED_SY = 4.46
SEED_NP = 4_321_800.0

PREFIX = "TXFXD3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_xtreme_daily3_{int(time.time())}.log"
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
        "round": 3,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY NEXT BAR C[X]*(1+LY*0.01) STOP; SELLSHORT NEXT BAR C[X]*(1-SY*0.01) STOP",
            "exits":       "Reversal only — no STP or LMT",
            "defaults":    "X=150, LY=3.25, SY=3.25",
            "r1_champion": "X=1 LY=3.1 SY=4.26 NP=3,998,400 MDD=-1,019,000 trades=12",
            "r2_champion": "X=1 LY=3.04 SY=4.46 NP=4,321,800 MDD=-1,041,000 trades=10 (R1→R2 +8.09%; A10≈A11 local conv)",
            "r3_focus":    "Ceiling confirmation: step=0.01/0.005 ultra-fine; global confirm; SY extension; triple convergence proof",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_xtreme_daily3.json"
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
    log.info("  SFJ_XtremeStop_NQ on TWF.TXF HOT Daily — Round 3")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f TWD  (8,000,000)", TARGET_NP)
    log.info("  R2 champion: X=%.4g  LY=%.4g  SY=%.4g  NP=%.0f",
             SEED_X, SEED_LY, SEED_SY, SEED_NP)
    log.info("  R3 focus: CEILING CONFIRMATION — step=0.01/0.005 precision + global confirm")
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
    # A01  step01_precision — ultra-fine step=0.01 around champion zone
    #      Tests whether step=0.01 gives any improvement beyond R2's step=0.02.
    #      X(1-3 s1)×LY(2.9-3.2 s0.01)×SY(4.2-4.7 s0.01) = 3×31×51=4743
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_step01_precision"
    _c = _cfg(_n, (1, 3, 1), (2.9, 3.2, 0.01), (4.2, 4.7, 0.01))
    log.info("A01  X(1-3 s1)×LY(2.9-3.2 s0.01)×SY(4.2-4.7 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A02  x_unit_confirm — confirm X=1 uniquely optimal with fine LY/SY
    #      R2-A02 used LY/SY step=0.2; now use step=0.1 for finer resolution.
    #      X(1-10 s1)×LY(2.5-4.0 s0.1)×SY(3.5-5.5 s0.1) = 10×16×21=3360
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_x_unit_confirm"
    _c = _cfg(_n, (1, 10, 1), (2.5, 4.0, 0.1), (3.5, 5.5, 0.1))
    log.info("A02  X(1-10 s1)×LY(2.5-4.0 s0.1)×SY(3.5-5.5 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A03  global_confirm — confirm no hidden regime in broad landscape
    #      X(1-500 s50)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5) = 11×20×20=4400
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_global_confirm"
    _c = _cfg(_n, (1, 500, 50), (0.5, 10.0, 0.5), (0.5, 10.0, 0.5))
    log.info("A03  X(1-500 s50)×LY(0.5-10 s0.5)×SY(0.5-10 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A04  sy_high_scan — SY=4.5-8% at fine step (R2-A04 stopped at SY=7 step=0.1)
    #      Does very high SY trigger fewer but larger trades at X=1?
    #      X(1-5 s1)×LY(2.0-4.0 s0.1)×SY(4.5-8.0 s0.1) = 5×21×36=3780
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_sy_high_scan"
    _c = _cfg(_n, (1, 5, 1), (2.0, 4.0, 0.1), (4.5, 8.0, 0.1))
    log.info("A04  X(1-5 s1)×LY(2.0-4.0 s0.1)×SY(4.5-8.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ly_fine_down — confirm LY floor with step=0.05 precision
    #      R2-A03 (step=0.1) confirmed LY<3.1 not better; this step=0.05 confirms floor.
    #      X(1-5 s1)×LY(1.5-3.1 s0.05)×SY(4.2-4.7 s0.05) = 5×33×11=1815
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ly_fine_down"
    _c = _cfg(_n, (1, 5, 1), (1.5, 3.1, 0.05), (4.2, 4.7, 0.05))
    log.info("A05  X(1-5 s1)×LY(1.5-3.1 s0.05)×SY(4.2-4.7 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A06  super_fine_005 — step=0.005 micro-precision in champion zone
    #      Final check: does step=0.005 give anything beyond step=0.02?
    #      X(1-3 s1)×LY(3.0-3.1 s0.005)×SY(4.4-4.5 s0.005) = 3×21×21=1323
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_super_fine_005"
    _c = _cfg(_n, (1, 3, 1), (3.0, 3.1, 0.005), (4.4, 4.5, 0.005))
    log.info("A06  X(1-3 s1)×LY(3.0-3.1 s0.005)×SY(4.4-4.5 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A07  medium_x_confirm — X=2-20 with fine LY/SY (confirm X=1 optimal)
    #      R2-A02 used step=0.2; now step=0.05 for X=2-20 to rule out any X>1.
    #      X(2-20 s1)×LY(2.8-3.3 s0.05)×SY(4.2-4.7 s0.05) = 19×11×11=2299
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_medium_x_confirm"
    _c = _cfg(_n, (2, 20, 1), (2.8, 3.3, 0.05), (4.2, 4.7, 0.05))
    log.info("A07  X(2-20 s1)×LY(2.8-3.3 s0.05)×SY(4.2-4.7 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A08  x1_broad_scan — comprehensive X=1-5 broad LY/SY scan
    #      Ensures no regime below LY=2.5 or below SY=4.2 was missed.
    #      X(1-5 s1)×LY(0.5-3.0 s0.1)×SY(0.5-4.2 s0.1) = 5×26×38=4940
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_x1_broad_scan"
    _c = _cfg(_n, (1, 5, 1), (0.5, 3.0, 0.1), (0.5, 4.2, 0.1))
    log.info("A08  X(1-5 s1)×LY(0.5-3.0 s0.1)×SY(0.5-4.2 s0.1)  %d combos",
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
    log.info("  SFJ_XtremeStop_NQ TWF.TXF HOT Daily Round-3 COMPLETE")
    log.info("  Champion: X=%.4g  LY=%.4g  SY=%.4g", best_x, best_ly, best_sy)
    log.info("  Best NP: %.0f TWD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    r2_np = 4_321_800.0
    log.info("  R2→R3 gain: %.3f%%", (best_np - r2_np) / r2_np * 100)
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
    print(f"R3 best: NP={best_np:,.0f} TWD  (X={best_x} LY={best_ly} SY={best_sy})")
    r2_np = 4_321_800.0
    print(f"R2→R3 gain: {(best_np - r2_np)/r2_np*100:+.3f}%")
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
        description="SFJ_XtremeStop_NQ TWF.TXF HOT Daily NP>8M TWD Round-3 ceiling confirmation")
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
