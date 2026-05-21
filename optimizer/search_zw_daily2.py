"""
search_zw_daily2.py — Breakout Daily NP > 700,000 on CBOT.ZW HOT, Round 2

R1 KEY FINDING (2026-05-20):
  Best: LE=25  SE=1  STP=5.25  LMT=5.25  NP=+17,615  MDD=-9,355  trades=34
  Also: LE=13  SE=5  STP=5.0   LMT=5.0   NP=+13,348  MDD=-7,452   trades=31
  First positive NP found — gap to target: -682,385 (-97.5%)

R1 pattern:
  High LE (25-day breakout) + ultra-short SE (1-5 days) + small STP/LMT (~5¢)
  Only 34 trades in 7yr ≈ 5 trades/year — very low frequency
  A09 (LE=10-40) found LE=25 gave NP=-1,088; adaptive zoom then turned it positive.
  A12 (wide boundary) independently confirmed LE=13 also positive.

R2 strategy — precision refinement of LE=25,SE=1 region + ceiling probe:
  1. LE fine sweep (20-35 step 1) at SE=1-3 — find exact LE optimum
  2. STP×LMT 2D fine grid at LE=25, SE=1 (step 0.5¢)
  3. SE sweep at LE=25 — confirm SE=1 is truly optimal
  4. High LE (30-60) — push LE further to see if NP rises
  5. LE=10-20 zone — confirm/extend A12's LE=13 finding
  6. Sub-5¢ LMT with LE=25, SE=1 — CL-daily-style tiny targets
  7. High LMT (10-100¢) with LE=25 — run winners wider
  8. Finest STP×LMT at LE=25 SE=1 (step 0.25¢, ZW min-tick resolution)
  9. Medium STP/LMT range (5-50¢) at LE=25
  10. Wide STP (100-500¢) at LE=25 — no-stop regime
  11. Adaptive zoom from R2 best
  12. Wide boundary R2 — catch any other missed LE/SE regimes

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  le_fine_sweep:   LE(20-35 s1) × SE(1-3 s1) × STP(4-8 s1) × LMT(4-8 s1)     16×3×5×5=1200
  A02  stp_lmt_2d:      LE(24-26 s1) × SE(1-3 s1) × STP(2-10 s0.5)× LMT(2-10 s0.5) 3×3×17×17=2601
  A03  se_sweep:        LE(23-27 s1) × SE(1-30 s2) × STP(4-8 s2) × LMT(4-8 s2)      5×16×3×3=720
  A04  high_le_push:    LE(30-60 s5) × SE(1-5 s1) × STP(3-15 s3) × LMT(3-15 s3)     7×5×5×5=875
  A05  le13_zone:       LE(10-20 s1) × SE(1-5 s1) × STP(3-10 s2) × LMT(3-10 s2)     11×5×4×4=880
  A06  sub5_lmt:        LE(23-27 s1) × SE(1-3 s1) × STP(3-10 s1) × LMT(0.5-5 s0.5)  5×3×8×10=1200
  A07  high_lmt:        LE(23-27 s1) × SE(1-5 s1) × STP(3-15 s3) × LMT(10-100 s10)  5×5×5×10=1250
  A08  finest_stplmt:   LE(24-26 s1) × SE(1-2 s1) × STP(4-7 s0.25)× LMT(4-7 s0.25) 3×2×13×13=1014
  A09  medium_range:    LE(22-28 s1) × SE(1-3 s1) × STP(5-50 s5) × LMT(5-50 s5)     7×3×10×10=2100
  A10  wide_stp2:       LE(23-27 s1) × SE(1-3 s1) × STP(50-500 s50)× LMT(5-50 s5)   5×3×10×10=1500
  A11  adaptive_zoom    (from best NP found in R2)
  A12  wide_boundary2:  LE(5-55 s5) × SE(1-15 s5) × STP(1-20 s5) × LMT(1-20 s5)     11×4×4×4=704
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp"
SYMBOL     = "CBOT.ZW HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\zw_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.25, 2000.0
LMT_LO, LMT_HI = 0.25, 2000.0
SE_LO,  SE_HI  = 1.0,   500.0
LE_LO,  LE_HI  = 1.0,   100.0

# R1 best
SEED_LE,  SEED_SE  = 25.0, 1.0
SEED_STP, SEED_LMT = 5.25, 5.25
SEED_NP   = 17_615.0
SEED_OBJ  = 33_168.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_zw_daily2_{int(time.time())}.log"
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
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"ZWD2_{name}",
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
    return OUTPUT_DIR / f"ZWD2_{name}_raw.csv"


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
    log.info("=== Starting ZWD2_%s (%d combos) ===", name, cfg.total_runs())
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


def champion(df, fb_le, fb_se, fb_stp, fb_lmt):
    """Priority: target met → positive NP → least-negative NP."""
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

    pos = df[df["Objective"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        le  = float(best["LE"]);  se  = float(best["SE"])
        stp = float(best["STP"]); lmt = float(best["LMT"])
        np_ = float(best.get("NetProfit", 0))
        mdd = float(best.get("MaxDrawdown", 0))
        tr  = int(best.get("TotalTrades", 0))
        obj = float(best["Objective"])
        log.info("  NP-Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 le, se, stp, lmt, obj, np_, mdd, tr)
        return le, se, stp, lmt, obj, np_, mdd, tr, False

    # All NP negative — track least-negative for zoom direction
    np_col = pd.to_numeric(df.get("NetProfit", pd.Series(dtype=float)), errors="coerce")
    if not np_col.isna().all():
        best = df.loc[np_col.idxmax()]
        le  = float(best["LE"]);  se  = float(best["SE"])
        stp = float(best["STP"]); lmt = float(best["LMT"])
        np_ = float(best.get("NetProfit", 0))
        mdd = float(best.get("MaxDrawdown", 0))
        tr  = int(best.get("TotalTrades", 0))
        log.info("  All-neg best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "NP=%.0f  MDD=%.0f  trades=%d",
                 le, se, stp, lmt, np_, mdd, tr)
        return le, se, stp, lmt, 0.0, np_, mdd, tr, False

    return fb_le, fb_se, fb_stp, fb_lmt, 0.0, 0.0, 0.0, 0, False


def _entry(attempt, name, df, le, se, stp, lmt, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "combos": combos, "rows": len(df) if df is not None else 0,
        "LE": le, "SE": se, "STP": stp, "LMT": lmt,
        "net_profit": round(np_, 0), "max_drawdown": round(mdd, 0),
        "objective":  round(obj, 0), "total_trades": trades,
        "target_met": met,
    }


def save_json(best, log_, met):
    above    = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    best_np  = max(log_, key=lambda x: x.get("net_profit", -1e18), default={})
    best_obj = max(log_, key=lambda x: x.get("objective", 0), default={})
    payload = {
        "strategy":           "Breakout_ZW_Daily (target NP>700K round-2)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1 day)",
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
        "best_obj_attempt":      best_obj,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_zw_daily2.json"
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
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ZW Daily Breakout NP>700K Round-2 — zoom on LE=25 SE=1 discovery")
    log.info("  R1 best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  Obj=%.0f",
             SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP, SEED_OBJ)
    log.info("  R1 also: LE=13 SE=5 STP=5 LMT=5 NP=13348")
    log.info("  Pattern: high LE + very short SE + small STP/LMT")
    log.info("  Target: %.0f  Gap: %.0f (-%.1f%%)",
             TARGET_NP, TARGET_NP - SEED_NP,
             (TARGET_NP - SEED_NP) / SEED_NP * 100)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_le, best_se, best_stp, best_lmt,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-28s]  no valid data", attempt_num, name)
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
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-28s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  %s",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f (gap +%.0f to target)",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  le_fine_sweep — LE=20–35 step 1 at SE=1-3 (find exact LE optimum)
    #      LE(20-35 s1) × SE(1-3 s1) × STP(4-8 s1) × LMT(4-8 s1) = 16×3×5×5=1200
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_le_fine_sweep"
    _c = _cfg(_n, (20, 35, 1), (1, 3, 1), (4.0, 8.0, 1.0), (4.0, 8.0, 1.0))
    log.info("A01  LE(20-35 s1) × SE(1-3 s1) × STP(4-8 s1) × LMT(4-8 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  stp_lmt_2d — STP×LMT 2D fine grid at LE=25, SE=1-3 (step 0.5¢)
    #      LE(24-26 s1) × SE(1-3 s1) × STP(2-10 s0.5) × LMT(2-10 s0.5) = 3×3×17×17=2601
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_stp_lmt_2d"
    _c = _cfg(_n, (24, 26, 1), (1, 3, 1), (2.0, 10.0, 0.5), (2.0, 10.0, 0.5))
    log.info("A02  LE(24-26 s1) × SE(1-3 s1) × STP(2-10 s0.5) × LMT(2-10 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  se_sweep — SE=1–30 step 2 at LE=25 (confirm SE=1 truly optimal)
    #      LE(23-27 s1) × SE(1-30 s2) × STP(4-8 s2) × LMT(4-8 s2) = 5×16×3×3=720
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_se_sweep"
    _c = _cfg(_n, (23, 27, 1), (1, 30, 2), (4.0, 8.0, 2.0), (4.0, 8.0, 2.0))
    log.info("A03  LE(23-27 s1) × SE(1-30 s2) × STP(4-8 s2) × LMT(4-8 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  high_le_push — LE=30–60 step 5 (push longer breakout lookback)
    #      LE(30-60 s5) × SE(1-5 s1) × STP(3-15 s3) × LMT(3-15 s3) = 7×5×5×5=875
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_high_le_push"
    _c = _cfg(_n, (30, 60, 5), (1, 5, 1), (3.0, 15.0, 3.0), (3.0, 15.0, 3.0))
    log.info("A04  LE(30-60 s5) × SE(1-5 s1) × STP(3-15 s3) × LMT(3-15 s3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  le13_zone — LE=10–20 step 1 (A12 found LE=13 positive)
    #      LE(10-20 s1) × SE(1-5 s1) × STP(3-10 s2) × LMT(3-10 s2) = 11×5×4×4=880
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_le13_zone"
    _c = _cfg(_n, (10, 20, 1), (1, 5, 1), (3.0, 10.0, 2.0), (3.0, 10.0, 2.0))
    log.info("A05  LE(10-20 s1) × SE(1-5 s1) × STP(3-10 s2) × LMT(3-10 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  sub5_lmt — LMT=0.5–5¢ step 0.5 at LE=25 (CL-daily-style tight targets)
    #      LE(23-27 s1) × SE(1-3 s1) × STP(3-10 s1) × LMT(0.5-5 s0.5) = 5×3×8×10=1200
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_sub5_lmt"
    _c = _cfg(_n, (23, 27, 1), (1, 3, 1), (3.0, 10.0, 1.0), (0.5, 5.0, 0.5))
    log.info("A06  LE(23-27 s1) × SE(1-3 s1) × STP(3-10 s1) × LMT(0.5-5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  high_lmt — LMT=10–100¢ at LE=25 (wider profit-taking)
    #      LE(23-27 s1) × SE(1-5 s1) × STP(3-15 s3) × LMT(10-100 s10) = 5×5×5×10=1250
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_high_lmt"
    _c = _cfg(_n, (23, 27, 1), (1, 5, 1), (3.0, 15.0, 3.0), (10.0, 100.0, 10.0))
    log.info("A07  LE(23-27 s1) × SE(1-5 s1) × STP(3-15 s3) × LMT(10-100 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  finest_stplmt — ZW min-tick resolution (step 0.25¢) at LE=25 SE=1
    #      LE(24-26 s1) × SE(1-2 s1) × STP(4-7 s0.25) × LMT(4-7 s0.25) = 3×2×13×13=1014
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_finest_stplmt"
    _c = _cfg(_n, (24, 26, 1), (1, 2, 1), (4.0, 7.0, 0.25), (4.0, 7.0, 0.25))
    log.info("A08  LE(24-26 s1) × SE(1-2 s1) × STP(4-7 s0.25) × LMT(4-7 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  medium_range — STP/LMT up to 50¢ at LE=22-28 (medium stops/targets)
    #      LE(22-28 s1) × SE(1-3 s1) × STP(5-50 s5) × LMT(5-50 s5) = 7×3×10×10=2100
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_medium_range"
    _c = _cfg(_n, (22, 28, 1), (1, 3, 1), (5.0, 50.0, 5.0), (5.0, 50.0, 5.0))
    log.info("A09  LE(22-28 s1) × SE(1-3 s1) × STP(5-50 s5) × LMT(5-50 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  wide_stp2 — STP=50–500¢ at LE=25 (essentially no-stop regime)
    #      LE(23-27 s1) × SE(1-3 s1) × STP(50-500 s50) × LMT(5-50 s5) = 5×3×10×10=1500
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_wide_stp2"
    _c = _cfg(_n, (23, 27, 1), (1, 3, 1), (50.0, 500.0, 50.0), (5.0, 50.0, 5.0))
    log.info("A10  LE(23-27 s1) × SE(1-3 s1) × STP(50-500 s50) × LMT(5-50 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom — zoom around R2 best NP
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom"
    log.info("A11  adaptive_zoom — center: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        cfg11 = None
        for r_le, r_se, r_stp, r_lmt, step_stp, step_lmt in [
            (3, 5, 3.0, 3.0, 0.5, 0.5),
            (2, 3, 2.0, 2.0, 0.5, 0.5),
            (2, 2, 1.5, 1.5, 0.5, 0.5),
            (1, 1, 1.0, 1.0, 0.25, 0.25),
        ]:
            _le  = zoom(best_le,  r_le,  1.0,     LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  1.0,     SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, step_stp, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, step_lmt, LMT_LO, LMT_HI)
            cfg11 = _cfg(_n, _le, _se, _stp, _lmt)
            if cfg11.total_runs() <= 5000:
                break
        if cfg11 is not None:
            log.info("A11  LE%s × SE%s × STP%s × LMT%s  %d combos",
                     _le, _se, _stp, _lmt, cfg11.total_runs())
            _update(run_or_load(_n, cfg11, conn, from_csv), cfg11, _n, A, cfg11.total_runs())
    log.info("After A11 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A12  wide_boundary2 — catch any other LE/SE regime missed by R1
    #      LE(5-55 s5) × SE(1-15 s5) × STP(1-20 s5) × LMT(1-20 s5) = 11×4×4×4=704
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_wide_boundary2"
    _c = _cfg(_n, (5, 55, 5), (1, 16, 5), (1.0, 21.0, 5.0), (1.0, 21.0, 5.0))
    log.info("A12  LE(5-55 s5) × SE(1-16 s5) × STP(1-21 s5) × LMT(1-21 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ZW Daily Round-2 COMPLETE")
    log.info("  Best NP: %.0f  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g",
             best_np, best_le, best_se, best_stp, best_lmt)
    log.info("  Target %.0f: %s", TARGET_NP, "★ MET" if target_met else f"NOT MET (gap +{TARGET_NP - best_np:.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LE": best_le, "SE": best_se, "STP": best_stp, "LMT": best_lmt,
            "net_profit": best_np, "max_drawdown": 0, "objective": best_obj,
            "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>700K: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
    ap = argparse.ArgumentParser(description="ZW Daily Breakout NP>700K Round-2 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1–12)")
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
