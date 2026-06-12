"""
search_txf_xtreme_hourly3.py — SFJ_XtremeStop_NQ on TWF.TXF HOT Hourly, Round 3

R2 champion: X=63, LY=5.16, SY=5.8, NP=5,405,400 TWD (gap −32.4%)
R1→R2 gain: +0.51%; MDD improved −9.6% (-747K→-676K)
R2 regime:  Low-trade (~11/7yr); X=63 confirmed by A08=A10 convergence
R2 gaps:    A02/A04/A06/A09/A11 UI-failed; tight_pct (LY/SY<3%) 3rd consecutive fail

R3 strategy:
  1. Ultra-fine LY/SY precision around champion (A01: step=0.01)
  2. Fine X sweep step=1 bridging X=58-70 with tight LY/SY grid (A02)
  3. Confirm SY lower bound: SY=5.0-5.8 region (A03)
  4. Retry tight_pct (4th attempt, LY/SY=0.1-3%): try narrower range to reduce combos (A04)
  5. Retry asym_high_sy with different range to avoid UI failure (A05)
  6. Bridge: X=55-75 step=2 with finer LY/SY grid (A06)
  7. Global confirmation: wide re-scan to ensure no missed regime (A07)
  8. Three adaptive zooms for triple convergence proof (A08–A10)
  9. Ultra-fine final zoom (A11: LY/SY step=0.005)

Convergence target: A08=A09=A10 triple confirmation → ceiling declared

Attempt schedule (11 attempts, ≤5,000 combos each):
  A01 ultra_fine_pct   : X(61-65 s1)×LY(4.9-5.4 s0.01)×SY(5.5-6.1 s0.01)  = 5×51×61=15555→trim
                         → X(62-64 s1)×LY(5.0-5.3 s0.02)×SY(5.6-6.0 s0.02) = 3×16×21=1008
  A02 x_precision      : X(58-70 s1)×LY(4.9-5.5 s0.1)×SY(5.4-6.2 s0.1)   = 13×7×9=819
  A03 sy_low_bound     : X(55-80 s5)×LY(4.5-6.0 s0.25)×SY(4.5-6.0 s0.25) = 6×7×7=294 too small
                         → X(40-100 s5)×LY(4.0-6.5 s0.25)×SY(4.0-6.5 s0.25)=13×11×11=1573
  A04 tight_pct_v4     : X(40-300 s20)×LY(0.5-3.0 s0.2)×SY(0.5-3.0 s0.2) = 14×13×13=2366
  A05 asym_sy_v2       : X(40-200 s20)×LY(3.5-6.0 s0.25)×SY(6.0-12 s0.5) = 9×11×13=1287
  A06 x_bridge_fine    : X(55-75 s2)×LY(4.8-5.6 s0.1)×SY(5.4-6.4 s0.1)  = 11×9×11=1089
  A07 global_confirm   : X(5-1005 s50)×LY(1.0-10 s0.5)×SY(1.0-10 s0.5)   = 20×19×19=7220→trim
                         → X(5-1005 s50)×LY(1.0-10 s1.0)×SY(1.0-10 s1.0)  = 20×10×10=2000
  A08 adaptive_zoom1   : X±8  s2,  LY±0.30 s0.05, SY±0.30 s0.05          → ≤9×13×13=1521
  A09 adaptive_zoom2   : X±4  s1,  LY±0.15 s0.02, SY±0.15 s0.02          → ≤9×16×16=2304
  A10 adaptive_zoom3   : X±2  s1,  LY±0.08 s0.01, SY±0.08 s0.01          → ≤5×17×17=1445
  A11 ultra_fine_final : X±1  s1,  LY±0.05 s0.005,SY±0.05 s0.005         → ≤3×21×21=1323
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_xtreme_hourly3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0   # TWD

X_LO,  X_HI  = 1.0,   10000.0
LY_LO, LY_HI = 0.01,  50.0
SY_LO, SY_HI = 0.01,  50.0

# R2 champion seed
SEED_X  = 63.0
SEED_LY = 5.16
SEED_SY = 5.8
SEED_NP = 5_405_400.0

PREFIX = "TXFXH3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_xtreme_hourly3_{int(time.time())}.log"
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
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY NEXT BAR C[X]*(1+LY*0.01) STOP; SELLSHORT NEXT BAR C[X]*(1-SY*0.01) STOP",
            "exits":    "Reversal only — no STP or LMT",
            "defaults": "X=150, LY=3.25, SY=3.25",
            "r1_champion": "X=65 LY=4.8 SY=6.0 NP=5,377,800 MDD=-747,400 trades=11",
            "r2_champion": "X=63 LY=5.16 SY=5.8 NP=5,405,400 MDD=-675,800 trades=11 (R1→R2 +0.51%)",
            "r3_focus": "Ultra-fine LY/SY precision; X step=1 scan; triple convergence proof; retry tight_pct v4",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_xtreme_hourly3.json"
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
    log.info("  SFJ_XtremeStop_NQ on TWF.TXF HOT Hourly — Round 3")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f TWD  (8,000,000)", TARGET_NP)
    log.info("  R2 seed: X=%.4g  LY=%.4g  SY=%.4g  NP=%.0f",
             SEED_X, SEED_LY, SEED_SY, SEED_NP)
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
    # A01  ultra_fine_pct — step=0.02 precision around R2 champion
    #      X(62-64 s1)×LY(5.0-5.3 s0.02)×SY(5.6-6.0 s0.02) = 3×16×21=1008
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ultra_fine_pct"
    _c = _cfg(_n, (62, 64, 1), (5.0, 5.3, 0.02), (5.6, 6.0, 0.02))
    log.info("A01  X(62-64 s1)×LY(5.0-5.3 s0.02)×SY(5.6-6.0 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A02  x_precision — X step=1 full scan around champion, moderate pct grid
    #      X(58-70 s1)×LY(4.9-5.5 s0.1)×SY(5.4-6.2 s0.1) = 13×7×9=819
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_x_precision"
    _c = _cfg(_n, (58, 70, 1), (4.9, 5.5, 0.1), (5.4, 6.2, 0.1))
    log.info("A02  X(58-70 s1)×LY(4.9-5.5 s0.1)×SY(5.4-6.2 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A03  sy_low_bound — test if SY<5.5 gives viable results
    #      X(40-100 s5)×LY(4.0-6.5 s0.25)×SY(4.0-6.5 s0.25) = 13×11×11=1573
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_sy_low_bound"
    _c = _cfg(_n, (40, 100, 5), (4.0, 6.5, 0.25), (4.0, 6.5, 0.25))
    log.info("A03  X(40-100 s5)×LY(4.0-6.5 s0.25)×SY(4.0-6.5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A04  tight_pct_v4 — 4th retry, narrower X range to reduce combos
    #      X(40-300 s20)×LY(0.5-3.0 s0.2)×SY(0.5-3.0 s0.2) = 14×13×13=2366
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_tight_pct_v4"
    _c = _cfg(_n, (40, 300, 20), (0.5, 3.0, 0.2), (0.5, 3.0, 0.2))
    log.info("A04  X(40-300 s20)×LY(0.5-3 s0.2)×SY(0.5-3 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A05  asym_sy_v2 — retry high-SY with tighter X range to avoid UI failure
    #      X(40-200 s20)×LY(3.5-6.0 s0.25)×SY(6.0-12 s0.5) = 9×11×13=1287
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_asym_sy_v2"
    _c = _cfg(_n, (40, 200, 20), (3.5, 6.0, 0.25), (6.0, 12.0, 0.5))
    log.info("A05  X(40-200 s20)×LY(3.5-6 s0.25)×SY(6-12 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A06  x_bridge_fine — X=55-75 step=2, finer pct grid around champion
    #      X(55-75 s2)×LY(4.8-5.6 s0.1)×SY(5.4-6.4 s0.1) = 11×9×11=1089
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_x_bridge_fine"
    _c = _cfg(_n, (55, 75, 2), (4.8, 5.6, 0.1), (5.4, 6.4, 0.1))
    log.info("A06  X(55-75 s2)×LY(4.8-5.6 s0.1)×SY(5.4-6.4 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A07  global_confirm — wide re-scan to confirm no undiscovered regime
    #      X(5-1005 s50)×LY(1-10 s1)×SY(1-10 s1) = 20×10×10=2000
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_global_confirm"
    _c = _cfg(_n, (5, 1005, 50), (1.0, 10.0, 1.0), (1.0, 10.0, 1.0))
    log.info("A07  X(5-1005 s50)×LY(1-10 s1)×SY(1-10 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A08  adaptive_zoom1 — X±8 s2, LY±0.30 s0.05, SY±0.30 s0.05
    #      ≤ 9×13×13 = 1521 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_adaptive_zoom1"
    log.info("A08  adaptive_zoom1 — center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   8.0, 2.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.30, 0.05, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.30, 0.05, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A08  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom2 — X±4 s1, LY±0.15 s0.02, SY±0.15 s0.02
    #      ≤ 9×16×16 = 2304 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom2"
    log.info("A09  adaptive_zoom2 — center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   4.0, 1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.15, 0.02, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.15, 0.02, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A09  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom3 — X±2 s1, LY±0.08 s0.01, SY±0.08 s0.01
    #      ≤ 5×17×17 = 1445 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom3"
    log.info("A10  adaptive_zoom3 — center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   2.0, 1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.08, 0.01, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.08, 0.01, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A10  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # A11  ultra_fine_final — X±1 s1, LY±0.05 s0.005, SY±0.05 s0.005
    #      ≤ 3×21×21 = 1323 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_ultra_fine_final"
    log.info("A11  ultra_fine_final — center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        _x  = zoom(best_x,   1.0, 1.0,  X_LO,  X_HI)
        _ly = zoom(best_ly,  0.05, 0.005, LY_LO, LY_HI)
        _sy = zoom(best_sy,  0.05, 0.005, SY_LO, SY_HI)
        _c  = _cfg(_n, _x, _ly, _sy)
        log.info("A11  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
             best_np, best_x, best_ly, best_sy)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_XtremeStop_NQ TWF.TXF HOT Hourly Round-3 COMPLETE")
    log.info("  Champion: X=%.4g  LY=%.4g  SY=%.4g", best_x, best_ly, best_sy)
    log.info("  Best NP: %.0f TWD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    log.info("  R2→R3 gain: %.2f%%",
             100.0 * (best_np - SEED_NP) / SEED_NP if SEED_NP > 0 else 0)
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
    print(f"R2→R3 gain: {100.0*(best_np-SEED_NP)/SEED_NP:.2f}%")
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
        description="SFJ_XtremeStop_NQ TWF.TXF HOT Hourly NP>8M TWD Round-3 search")
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
