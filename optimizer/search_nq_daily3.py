"""
search_nq_daily3.py — Breakout Daily NP > 700,000 on CME.NQ HOT, Round 3

Round-2 findings (2026-05-18):
  Best: LE=1  SE=78  STP=5.5  LMT=3.75  NP=341,410  MDD=-72,655  trades=28
  Gap to 700K: -358,590 (-51%)  (only +3.4% progress from R1's 330K)

Critical R2 discoveries:
  1. STP=8 and STP=25 give IDENTICAL result (NP=325K, trades=22) — above STP≈5-6,
     stop is never hit; exits are driven by SE (time-based) or LMT
  2. SE=77-78, LMT=3.75 confirmed as local NP peak with step-1 fine scan (A01)
  3. High LMT (8-25) + high SE: worse than tight LMT=3.75 (A03 gave only 309K)
  4. SE=145-150 also competitive (A09: NP=322K) — ultra-long SE untested finely
  5. A04 (STP=2): 53 trades but NP=317K — more trades doesn't help

Round-3 strategy: Genuinely new territory
  H1. Ultra-fine LMT: LMT=0.5-4 step 0.05 — maybe the exact LMT value matters sharply
  H2. Sub-1 LMT (0.1-1.4): very tight exits not yet tested
  H3. Ultra-high SE (150-400): rare but potentially very large breakouts
  H4. SE=40-75 with tight LMT — mid-SE fine scan with new LMT knowledge
  H5. LE=1 fine: LE=1 + SE step 1 over full 70-92 range
  H6. Sub-zone precision: LE=1 × SE=74-82 × STP=5-7 × LMT=3-5 step 0.1
  H7. SE=60-150 step 2 comprehensive sweep
  H8. Very high SE (200-400 step 10) — unknown territory
  H9. STP precision: STP=1-4 step 0.5 with SE=75-85 — maybe tighter STP IS better
  H10. Adaptive zoom around best R3 NP
  H11. Dense zoom (adaptive)
  H12. Global boundary ultra-wide

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  LE(1-3) × SE(74-82 s2) × STP(5-6 s0.5) × LMT(0.5-4 s0.05)     3×5×3×72 =3240
  A02  LE(1-4) × SE(70-90 s5) × STP(4-7 s1)   × LMT(0.1-1.5 s0.1)    4×5×4×15 =1200
  A03  LE(1-5) × SE(150-400 s25)× STP(3-7 s1)  × LMT(2-6 s1)          5×11×5×5 =1375
  A04  LE(1-4) × SE(40-75 s5)  × STP(4-7 s1)  × LMT(2-6 s0.5)        4×8×4×9  =1152
  A05  LE(1-2) × SE(70-90 s1)  × STP(4-7 s0.5)× LMT(3-5 s0.25)       2×21×7×9 =2646
  A06  LE(1-2) × SE(74-82 s1)  × STP(5-7 s0.25)× LMT(3-5 s0.1)       2×9×9×21 =3402
  A07  LE(1-3) × SE(60-150 s2) × STP(5-7 s1)  × LMT(3-5 s1)          3×46×3×3 =1242
  A08  LE(1-4) × SE(200-400 s10)× STP(3-8 s1)  × LMT(2-6 s1)         4×21×6×5 =2520
  A09  LE(1-3) × SE(75-85 s1)  × STP(1-4 s0.5)× LMT(2-6 s0.5)        3×11×7×9 =2079
  A10  Adaptive zoom around best NP from A01-A09
  A11  Dense zoom (progressive shrink, ≤5000)
  A12  LE(1-8 s2) × SE(30-400 s30) × STP(0.5-25 s5) × LMT(0.5-15 s3) 4×13×6×6 =1872
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
from typing import Dict, List, Optional, Tuple

import numpy as np
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp"
SYMBOL     = "CME.NQ HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_daily3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 700_000.0
MAX_ATTEMPTS = 12

STP_LO, STP_HI = 0.05, 500.0
LMT_LO, LMT_HI = 0.05, 500.0
SE_LO,  SE_HI  = 1.0,  500.0
LE_LO,  LE_HI  = 1.0,  100.0

# R2 best: LE=1 SE=78 STP=5.5 LMT=3.75 NP=341,410
SEED_LE,  SEED_SE  = 1.0,  78.0
SEED_STP, SEED_LMT = 5.5,   3.75
SEED_NP   = 341_410.0
SEED_OBJ  = 1_604_305.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_daily3_{int(time.time())}.log"
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


def _cfg(name, le, se, stp, lmt):
    def _safe(t):
        s, e, step = t
        if s == e:
            return (max(LE_LO, s - step), min(LE_HI, s + step), step)
        return t
    le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)
    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"NQD3_{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
            ParamAxis("LE",  *le),
            ParamAxis("SE",  *se),
            ParamAxis("STP", *stp),
            ParamAxis("LMT", *lmt),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name):
    return OUTPUT_DIR / f"NQD3_{name}_raw.csv"


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
    log.info("=== Attempt %-30s  (%d combos) ===", name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done in %.1f min", (time.time() - t0) / 60)
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
            log.warning("  INVALID: %s out of range — skipping.", p.name)
            return False
    return True


def champion(df, fb_le, fb_se, fb_stp, fb_lmt):
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best.get("TotalTrades", 0)))
        return (float(best["LE"]), float(best["SE"]),
                float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best.get("TotalTrades", 0)), True)

    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        return fb_le, fb_se, fb_stp, fb_lmt, 0.0, 0.0, 0.0, 0, False

    best = pos.loc[pos["NetProfit"].idxmax()]
    le  = float(best["LE"]);  se  = float(best["SE"])
    stp = float(best["STP"]); lmt = float(best["LMT"])
    np_ = float(best.get("NetProfit", 0))
    mdd = float(best.get("MaxDrawdown", 0))
    tr  = int(best.get("TotalTrades", 0))
    obj = float(best["Objective"])
    log.info("  Champion (NP-max): LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
             le, se, stp, lmt, np_, mdd, obj, tr)
    return le, se, stp, lmt, obj, np_, mdd, tr, False


def _entry(attempt, name, df, le, se, stp, lmt, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "combos": combos, "rows": len(df) if df is not None else 0,
        "LE": le, "SE": se, "STP": stp, "LMT": lmt,
        "net_profit":   round(np_, 0),
        "max_drawdown": round(mdd, 0),
        "objective":    round(obj, 0),
        "total_trades": trades,
        "target_met":   met,
    }


def save_json(best, best_np, log_, met):
    above = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    payload = {
        "strategy":           "Breakout_NQ_Daily  (target NP>700K round-3)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "r2_best": {
            "LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_nq_daily3.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met    = False
    attempt_log   = []
    best_entry    = {}
    best_np_entry = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Round-3 NQ Daily NP>700K")
    log.info("  Seed (R2 best): LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    log.info("  Gap: %.0f (+%.1f%%)", TARGET_NP - best_np,
             (TARGET_NP - best_np) / best_np * 100)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry, best_np_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_le, best_se, best_stp, best_lmt,
                                      0.0, 0.0, 0.0, 0, False, combos))
            log.info("  [A%02d %-25s]  no valid data", attempt_num, name)
            return

        le, se, stp, lmt, obj, np_, mdd, tr, met = champion(
            df, best_le, best_se, best_stp, best_lmt)

        if np_ > best_np:
            best_le, best_se = le, se
            best_stp, best_lmt = stp, lmt
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, le, se, stp, lmt,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", 0) > best_entry.get("net_profit", 0))):
            best_entry = e
        if not best_np_entry or e.get("net_profit", 0) > best_np_entry.get("net_profit", 0):
            best_np_entry = e

        log.info("  [A%02d %-25s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  %s",
                 attempt_num, name, le, se, stp, lmt,
                 obj, np_, mdd, tr, "★TARGET★" if met else "")
        log.info("       Global best NP=%.0f (need +%.1f%%)",
                 best_np,
                 (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else 0)

    # ──────────────────────────────────────────────────────────────────────
    # A01  Ultra-fine LMT step 0.05: LMT=0.5-4 around SE=74-82
    #      LE(1-3) × SE(74-82 s2) × STP(5-6 s0.5) × LMT(0.5-4 s0.05)  3×5×3×72=3240
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ultrafine_lmt"
    _c = _cfg(_n, (1,3,1), (74,82,2), (5.0,6.0,0.5), (0.5,4.0,0.05))
    log.info("A01  LE(1-3) × SE(74-82 s2) × STP(5-6 s0.5) × LMT(0.5-4 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  Sub-1 LMT (0.1-1.4 step 0.1) — very tight exits never tested
    #      LE(1-4) × SE(70-90 s5) × STP(4-7 s1) × LMT(0.1-1.5 s0.1)  4×5×4×15=1200
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_sub1_lmt"
    _c = _cfg(_n, (1,4,1), (70,90,5), (4.0,7.0,1.0), (0.1,1.5,0.1))
    log.info("A02  LE(1-4) × SE(70-90 s5) × STP(4-7 s1) × LMT(0.1-1.5 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  Ultra-high SE (150-400 step 25) — rare giant breakouts
    #      LE(1-5) × SE(150-400 s25) × STP(3-7 s1) × LMT(2-6 s1)  5×11×5×5=1375
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ultra_high_se"
    _c = _cfg(_n, (1,5,1), (150,400,25), (3.0,7.0,1.0), (2.0,6.0,1.0))
    log.info("A03  LE(1-5) × SE(150-400 s25) × STP(3-7 s1) × LMT(2-6 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  Mid SE (40-75) with tight LMT — R1 mid-SE region with new LMT
    #      LE(1-4) × SE(40-75 s5) × STP(4-7 s1) × LMT(2-6 s0.5)  4×8×4×9=1152
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_mid_se_tight_lmt"
    _c = _cfg(_n, (1,4,1), (40,75,5), (4.0,7.0,1.0), (2.0,6.0,0.5))
    log.info("A04  LE(1-4) × SE(40-75 s5) × STP(4-7 s1) × LMT(2-6 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  LE=1 fine sweep: SE step 1 over SE=70-90
    #      LE(1-2) × SE(70-90 s1) × STP(4-7 s0.5) × LMT(3-5 s0.25)  2×21×7×9=2646
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_le1_fine"
    _c = _cfg(_n, (1,2,1), (70,90,1), (4.0,7.0,0.5), (3.0,5.0,0.25))
    log.info("A05  LE(1-2) × SE(70-90 s1) × STP(4-7 s0.5) × LMT(3-5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  Sub-zone precision: SE=74-82 step 1, LMT=3-5 step 0.1
    #      LE(1-2) × SE(74-82 s1) × STP(5-7 s0.25) × LMT(3-5 s0.1)  2×9×9×21=3402
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_subzone_precision"
    _c = _cfg(_n, (1,2,1), (74,82,1), (5.0,7.0,0.25), (3.0,5.0,0.1))
    log.info("A06  LE(1-2) × SE(74-82 s1) × STP(5-7 s0.25) × LMT(3-5 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  SE comprehensive: SE=60-150 step 2
    #      LE(1-3) × SE(60-150 s2) × STP(5-7 s1) × LMT(3-5 s1)  3×46×3×3=1242
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_se_comprehensive"
    _c = _cfg(_n, (1,3,1), (60,150,2), (5.0,7.0,1.0), (3.0,5.0,1.0))
    log.info("A07  LE(1-3) × SE(60-150 s2) × STP(5-7 s1) × LMT(3-5 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  Very high SE (200-400 step 10) — unknown territory
    #      LE(1-4) × SE(200-400 s10) × STP(3-8 s1) × LMT(2-6 s1)  4×21×6×5=2520
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_very_high_se"
    _c = _cfg(_n, (1,4,1), (200,400,10), (3.0,8.0,1.0), (2.0,6.0,1.0))
    log.info("A08  LE(1-4) × SE(200-400 s10) × STP(3-8 s1) × LMT(2-6 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  STP precision: STP=1-4 step 0.5 with SE=75-85
    #      LE(1-3) × SE(75-85 s1) × STP(1-4 s0.5) × LMT(2-6 s0.5)  3×11×7×9=2079
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_stp_precision"
    _c = _cfg(_n, (1,3,1), (75,85,1), (1.0,4.0,0.5), (2.0,6.0,0.5))
    log.info("A09  LE(1-3) × SE(75-85 s1) × STP(1-4 s0.5) × LMT(2-6 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  Adaptive zoom around R3 best NP
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_zoom_r3"
    for _r_se, _r_stp, _r_lmt in [(6, 0.5, 1.5), (4, 0.4, 1.0), (3, 0.3, 0.75), (2, 0.2, 0.5)]:
        _le10  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se10  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp10 = zoom(best_stp, _r_stp, 0.25, STP_LO, STP_HI)
        _lmt10 = zoom(best_lmt, _r_lmt, 0.05, LMT_LO, LMT_HI)
        _c = _cfg(_n, _le10, _se10, _stp10, _lmt10)
        if _c.total_runs() <= 5000:
            break
    log.info("A10  Zoom best LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le10, _se10, _stp10, _lmt10, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt in [(4,0.3,0.5), (3,0.25,0.4), (2,0.2,0.3), (1,0.15,0.2)]:
        _le11  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se11  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, _r_stp, 0.25, STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, _r_lmt, 0.05, LMT_LO, LMT_HI)
        _c = _cfg(_n, _le11, _se11, _stp11, _lmt11)
        if _c.total_runs() <= 5000:
            break
    log.info("A11  Dense zoom LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le11, _se11, _stp11, _lmt11, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A12  Global boundary ultra-wide
    #      LE(1-8 s2) × SE(30-400 s30) × STP(0.5-25 s5) × LMT(0.5-15 s3)  4×13×6×6=1872
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (1,8,2), (30,400,30), (0.5,25.0,5.0), (0.5,15.0,3.0))
    log.info("A12  LE(1-8 s2) × SE(30-400 s30) × STP(0.5-25 s5) × LMT(0.5-15 s3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    if not best_entry:
        best_entry = _entry(0, "r2_seed", None,
                            SEED_LE, SEED_SE, SEED_STP, SEED_LMT,
                            SEED_OBJ, SEED_NP, 0.0, 0, False, 0)
    if not best_np_entry:
        best_np_entry = best_entry

    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL — Round-3 NQ Daily")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best NP: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
             best_np_entry.get("LE"), best_np_entry.get("SE"),
             best_np_entry.get("STP"), best_np_entry.get("LMT"),
             best_np_entry.get("net_profit", 0),
             best_np_entry.get("max_drawdown", 0),
             best_np_entry.get("objective", 0),
             best_np_entry.get("total_trades", 0))
    log.info("  Target (NP>700K): %s", "MET ✓" if target_met else
             "NOT MET — best NP=%.0f" % best_np)
    log.info("══════════════════════════════════════════════════════════════")
    log.info("")
    log.info("  %-3s %-28s %6s %10s %10s %12s %6s  %s",
             "A#", "Name", "Rows", "NP", "MDD", "Objective", "Trd", "★")
    for e in attempt_log:
        log.info("  A%02d %-28s %6d %10.0f %10.0f %12.0f %6d  %s",
                 e["attempt"], e["name"], e.get("rows", 0),
                 e.get("net_profit", 0), e.get("max_drawdown", 0),
                 e.get("objective", 0), e.get("total_trades", 0),
                 "★" if e.get("target_met") else "")

    out = save_json(best_entry, best_np_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>700K: {'MET ✓' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Admin manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="Round-3 NQ Daily — target NP > 700K USD")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--attempt", type=int, default=1, metavar="N")
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
