"""
search_nq_daily2.py — Breakout Daily NP > 700,000 on CME.NQ HOT, Round 2

Round-1 findings (2026-05-18):
  Best: LE=2  SE=77  STP=4.5  LMT=3.5  NP=330,160  MDD=-83,390  trades=32
  Best obj: LE=2  SE=78  STP=5.5  LMT=4.0  obj=1,409,577  NP=320,020

Key R1 discoveries:
  - SE=70-85 is the ONLY productive region — all best attempts converged here
  - LMT=3.5-5 (tight) beats LMT=14-30 (loose) — counterintuitive for daily
  - STP=4-7 optimal; STP<1 or STP>8 untested at SE=75-85
  - Low SE (3-20) performs poorly (NP=194K) — daily needs long lookback
  - LE=1-4 all work; LE=2 appears slightly best
  - Only ~30 trades in 7 years — very selective breakout system
  - Gap to 700K: -369,840 (-53%) — large, need to push hard

Round-2 strategy:
  H1. Fine SE step 1 around SE=68-92 — find exact peak (coarse R1 may have missed it)
  H2. LMT precision around 3.5 — LMT=1-7 step 0.25 with right SE
  H3. High LMT + high SE combo — SE=75-90 + LMT=8-25 (R1 never combined these)
  H4. Tight STP (0.5-3) — smaller losses = higher NP?
  H5. Large STP (8-30) — let winners run more (untested in SE=75-85 zone)
  H6. Very large STP (30-100) — extreme stop, allow outlier mega-moves
  H7. LE×SE fine grid — LE=1-6 × SE=70-90 step 1
  H8. Precision zoom around R1 best
  H9. SE=90-200 fine — extend beyond known region
  H10. 4D medium with R2 knowledge
  H11. Dense zoom (adaptive)
  H12. Wide boundary — sanity check with larger STP range

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  LE(1-4) × SE(68-92 s1)  × STP(3-6 s0.5)   × LMT(2.5-5.5 s0.5)   4×25×7×7 =4900
  A02  LE(1-3) × SE(72-84 s3)  × STP(3.5-7 s0.5)  × LMT(1-7 s0.25)      3×5×8×25 =3000
  A03  LE(1-4) × SE(60-95 s5)  × STP(2-7 s1)      × LMT(8-25 s2)        4×8×6×9  =1728
  A04  LE(1-4) × SE(65-90 s5)  × STP(0.5-3 s0.25) × LMT(3-7 s0.5)       4×6×11×9 =2376
  A05  LE(1-4) × SE(65-90 s5)  × STP(8-25 s1)     × LMT(3-8 s1)         4×6×18×6 =2592
  A06  LE(1-4) × SE(65-90 s5)  × STP(25-80 s5)    × LMT(2-8 s1)         4×6×12×7 =2016
  A07  LE(1-6) × SE(70-90 s1)  × STP(4-6 s0.5)    × LMT(3-5 s0.5)       6×21×5×5 =3150
  A08  LE(1-3) × SE(73-81 s1)  × STP(3.5-5.5 s0.25)× LMT(2.5-5 s0.25)  3×9×9×11 =2673
  A09  LE(1-4) × SE(90-200 s5) × STP(3-7 s1)      × LMT(3-7 s1)         4×23×5×5 =2300
  A10  LE(1-5) × SE(68-92 s3)  × STP(3-7 s0.5)    × LMT(2-6 s0.5)       5×9×9×9  =3645
  A11  Dense zoom around best NP (progressive shrink, ≤5000)
  A12  LE(1-10 s2) × SE(50-180 s10) × STP(1-25 s4) × LMT(1-15 s2)       5×14×7×8 =3920
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 700_000.0
MAX_ATTEMPTS = 12

STP_LO, STP_HI = 0.05, 200.0
LMT_LO, LMT_HI = 0.5,  200.0
SE_LO,  SE_HI  = 1.0,  400.0
LE_LO,  LE_HI  = 1.0,  100.0

# R1 best NP: LE=2 SE=77 STP=4.5 LMT=3.5 NP=330,160
SEED_LE,  SEED_SE  = 2.0,  77.0
SEED_STP, SEED_LMT = 4.5,   3.5
SEED_NP   = 330_160.0
SEED_OBJ  = 1_307_179.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_daily2_{int(time.time())}.log"
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
         le:  Tuple[float, float, float],
         se:  Tuple[float, float, float],
         stp: Tuple[float, float, float],
         lmt: Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t):
        s, e, step = t
        if s == e:
            return (max(LE_LO, s - step), min(LE_HI, s + step), step)
        return t
    le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)
    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000 limit!", name, combos)
    return StrategyConfig(
        name=f"NQD2_{name}",
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


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"NQD2_{name}_raw.csv"


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
        log.info("  Done in %.1f min — %s", (time.time() - t0) / 60, Path(raw_csv).name)
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
            log.warning("  INVALID: %s out of range [%.4g, %.4g] — skipping.",
                        p.name, lo, hi)
            return False
    return True


def champion(df, fb_le, fb_se, fb_stp, fb_lmt):
    """Return (LE, SE, STP, LMT, obj, NP, MDD, trades, target_met).
    Zoom seed = NP-max (target chasing)."""
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
        "strategy":           "Breakout_NQ_Daily  (target NP>700K round-2)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "r1_best": {
            "LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_nq_daily2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met   = False
    attempt_log  = []
    best_entry   = {}
    best_np_entry = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Round-2 NQ Daily NP>700K  (all-4-param variation enforced)")
    log.info("  Seed (R1 best): LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    log.info("  Gap to 700K: %.0f (+%.1f%%)", TARGET_NP - best_np,
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
    # A01  Fine SE scan step 1 around SE=68-92
    #      LE(1-4) × SE(68-92 s1) × STP(3-6 s0.5) × LMT(2.5-5.5 s0.5)  4×25×7×7=4900
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_se_scan"
    _c = _cfg(_n, (1,4,1), (68,92,1), (3.0,6.0,0.5), (2.5,5.5,0.5))
    log.info("A01  LE(1-4) × SE(68-92 s1) × STP(3-6 s0.5) × LMT(2.5-5.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  LMT precision LMT=1-7 step 0.25 around best SE
    #      LE(1-3) × SE(72-84 s3) × STP(3.5-7 s0.5) × LMT(1-7 s0.25)  3×5×8×25=3000
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_lmt_precision"
    _c = _cfg(_n, (1,3,1), (72,84,3), (3.5,7.0,0.5), (1.0,7.0,0.25))
    log.info("A02  LE(1-3) × SE(72-84 s3) × STP(3.5-7 s0.5) × LMT(1-7 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  High LMT + high SE combo (R1 never combined SE=75-90 with LMT=8-25)
    #      LE(1-4) × SE(60-95 s5) × STP(2-7 s1) × LMT(8-25 s2)  4×8×6×9=1728
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_high_lmt_high_se"
    _c = _cfg(_n, (1,4,1), (60,95,5), (2.0,7.0,1.0), (8,25,2))
    log.info("A03  LE(1-4) × SE(60-95 s5) × STP(2-7 s1) × LMT(8-25 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  Tight STP (0.5-3) with SE=65-90 — smaller losses improve NP?
    #      LE(1-4) × SE(65-90 s5) × STP(0.5-3 s0.25) × LMT(3-7 s0.5)  4×6×11×9=2376
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_tight_stp"
    _c = _cfg(_n, (1,4,1), (65,90,5), (0.5,3.0,0.25), (3.0,7.0,0.5))
    log.info("A04  LE(1-4) × SE(65-90 s5) × STP(0.5-3 s0.25) × LMT(3-7 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  Large STP (8-25) with SE=65-90 — let winners run (untested zone)
    #      LE(1-4) × SE(65-90 s5) × STP(8-25 s1) × LMT(3-8 s1)  4×6×18×6=2592
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_large_stp"
    _c = _cfg(_n, (1,4,1), (65,90,5), (8.0,25.0,1.0), (3.0,8.0,1.0))
    log.info("A05  LE(1-4) × SE(65-90 s5) × STP(8-25 s1) × LMT(3-8 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  Very large STP (25-80) with SE=65-90 — extreme stop
    #      LE(1-4) × SE(65-90 s5) × STP(25-80 s5) × LMT(2-8 s1)  4×6×12×7=2016
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_extreme_stp"
    _c = _cfg(_n, (1,4,1), (65,90,5), (25.0,80.0,5.0), (2.0,8.0,1.0))
    log.info("A06  LE(1-4) × SE(65-90 s5) × STP(25-80 s5) × LMT(2-8 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  LE×SE fine grid: LE=1-6 × SE=70-90 step 1
    #      LE(1-6) × SE(70-90 s1) × STP(4-6 s0.5) × LMT(3-5 s0.5)  6×21×5×5=3150
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_le_se_fine"
    _c = _cfg(_n, (1,6,1), (70,90,1), (4.0,6.0,0.5), (3.0,5.0,0.5))
    log.info("A07  LE(1-6) × SE(70-90 s1) × STP(4-6 s0.5) × LMT(3-5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  Precision zoom around R1 best (LE=2, SE=77, STP=4.5, LMT=3.5)
    #      LE(1-3) × SE(73-81 s1) × STP(3.5-5.5 s0.25) × LMT(2.5-5 s0.25)  3×9×9×11=2673
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_r1_precision"
    _c = _cfg(_n, (1,3,1), (73,81,1), (3.5,5.5,0.25), (2.5,5.0,0.25))
    log.info("A08  LE(1-3) × SE(73-81 s1) × STP(3.5-5.5 s0.25) × LMT(2.5-5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  SE extension SE=90-200 (beyond R1 boundary, finer step)
    #      LE(1-4) × SE(90-200 s5) × STP(3-7 s1) × LMT(3-7 s1)  4×23×5×5=2300
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_se_extension"
    _c = _cfg(_n, (1,4,1), (90,200,5), (3.0,7.0,1.0), (3.0,7.0,1.0))
    log.info("A09  LE(1-4) × SE(90-200 s5) × STP(3-7 s1) × LMT(3-7 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  4D medium with R2 knowledge
    #      LE(1-5) × SE(68-92 s3) × STP(3-7 s0.5) × LMT(2-6 s0.5)  5×9×9×9=3645
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_4d_medium"
    _c = _cfg(_n, (1,5,1), (68,92,3), (3.0,7.0,0.5), (2.0,6.0,0.5))
    log.info("A10  LE(1-5) × SE(68-92 s3) × STP(3-7 s0.5) × LMT(2-6 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom around best NP (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt in [(5,0.5,2), (4,0.4,1.5), (3,0.3,1), (2,0.2,0.75)]:
        _le11  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se11  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, _r_stp, 0.25, STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, _r_lmt, 0.25, LMT_LO, LMT_HI)
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
    # A12  Wide boundary with larger STP range
    #      LE(1-10 s2) × SE(50-180 s10) × STP(1-25 s4) × LMT(1-15 s2)  5×14×7×8=3920
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (1,10,2), (50,180,10), (1.0,25.0,4.0), (1.0,15.0,2.0))
    log.info("A12  LE(1-10 s2) × SE(50-180 s10) × STP(1-25 s4) × LMT(1-15 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    if not best_entry:
        best_entry = _entry(0, "r1_seed", None,
                            SEED_LE, SEED_SE, SEED_STP, SEED_LMT,
                            SEED_OBJ, SEED_NP, 0.0, 0, False, 0)
    if not best_np_entry:
        best_np_entry = best_entry

    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL RESULT — Round-2 NQ Daily")
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
    log.info("  Per-attempt summary:")
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
        description="Round-2 NQ Daily search — target NP > 700K USD")
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
