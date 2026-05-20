"""
search_zw_hourly5.py — Breakout Hourly NP > 700,000 on CBOT.ZW HOT, Round 5

R4 KEY FINDING (2026-05-20):
  Best: LE=1  SE=900  STP=40  LMT=30  NP=−788  MDD=−38,495  trades=365
  Gap to zero: only −788 USD!  SE=900 is near-breakeven.

R4 monotonic SE trend (all with LE=1, STP=40, LMT=30):
  SE=500-1000 step 25: best at SE=900 → NP=−788
  (SE=700 step 100 in A04 gave −6,622 — coarse step missed SE=900 peak)

Two key parameters at boundary of R4 test:
  1. STP: A09 range was STP=10-40 step 10; STP=40 was the UPPER LIMIT → may go higher
  2. SE:  A09 range was SE=500-1000 step 25; SE=900 was the BEST → what about SE>1000?

R5 strategy — push beyond R4 boundaries:
  1. se_1000_1500    : SE=1000–1500 (extend the monotonic trend)
  2. stp_50_100      : STP=50–100 at SE=900 (STP boundary push)
  3. se_900_zoom     : fine zoom around SE=900 (step 10) with wider STP/LMT
  4. se_1500_2000    : SE=1500–2000 (extend further)
  5. stp_100_500     : STP=100–500 at SE=900 (extremely wide stop)
  6. lmt_zoom        : LMT sweep at SE=900/STP=40 (LMT was only coarsely tested)
  7. se_stp_2d       : SE=850–1200 × STP=30–80 2D grid
  8. se_1000_fine    : SE=950–1100 step 10 (highest resolution around SE=1000 crossover)
  9. le_at_high_se   : LE=1–6 at SE=900 (confirm LE=1 is best at high SE)
  10. se_zoom_stp_hi : SE=900–1200 × STP=40–120 (joint push on both boundaries)
  11. dense_zoom     : adaptive progressive zoom around best NP
  12. boundary_r5    : global boundary — SE up to 3000, STP up to 1000

ZW context:
  1 pt = $50/contract.  SE=900 bars = 900 hours ÷ 6.5 hrs/day ≈ 138 trading days ≈ 6.5 months lookback.
  At 365 trades/7yr ≈ 52 trades/year, MDD≈-38K, NP≈0.

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 se_1000_1500:   LE(1-3 s1) × SE(1000-1500 s25)  × STP(30-60 s10) × LMT(20-50 s10) 3×21×4×4=1008
  A02 stp_50_100:     LE(1-3 s1) × SE(800-1000 s25)   × STP(50-150 s10) × LMT(20-50 s10) 3×9×11×4=1188
  A03 se_900_zoom:    LE(1-3 s1) × SE(850-960 s10)    × STP(20-80 s10)  × LMT(20-50 s10) 3×12×7×4=1008
  A04 se_1500_2000:   LE(1-2 s1) × SE(1500-2000 s50)  × STP(30-60 s10) × LMT(20-50 s10) 2×11×4×4=352
  A05 stp_100_500:    LE(1-3 s1) × SE(800-1000 s50)   × STP(100-500 s100)× LMT(20-60 s20) 3×5×5×3=225
  A06 lmt_zoom:       LE(1-3 s1) × SE(850-950 s25)    × STP(30-60 s10)  × LMT(10-100 s10) 3×5×4×10=600
  A07 se_stp_2d:      LE(1-2 s1) × SE(850-1200 s25)   × STP(30-80 s10)  × LMT(20-50 s10) 2×15×6×4=720
  A08 se_1000_fine:   LE(1-3 s1) × SE(950-1100 s10)   × STP(30-70 s10)  × LMT(20-50 s10) 3×16×5×4=960
  A09 le_at_high_se:  LE(1-8 s1) × SE(850-950 s25)    × STP(30-60 s10)  × LMT(20-50 s10) 8×5×4×4=640
  A10 se_stp_hi:      LE(1-2 s1) × SE(900-1300 s25)   × STP(40-120 s20) × LMT(20-50 s10) 2×17×5×4=680
  A11 dense_zoom      (adaptive progressive shrink from best NP found)
  A12 boundary_r5:    LE(1-5 s2) × SE(500-3000 s250)  × STP(10-200 s30) × LMT(10-100 s30) 3×11×7×4=924
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\zw_hourly5_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI =  0.01, 5000.0
LMT_LO, LMT_HI =  0.25, 5000.0
SE_LO,  SE_HI  =  1.0,  5000.0
LE_LO,  LE_HI  =  1.0,  100.0

# R4 best: LE=1 SE=900 STP=40 LMT=30 NP=-788
SEED_LE,  SEED_SE  = 1.0,   900.0
SEED_STP, SEED_LMT = 40.0,   30.0
SEED_NP = -788.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_zw_hourly5_{int(time.time())}.log"
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
            return (max(LE_LO, s - step), min(SE_HI, s + step), step)
        return t
    le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)
    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"ZWH5_{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
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
    return OUTPUT_DIR / f"ZWH5_{name}_raw.csv"


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
    log.info("=== Starting ZWH5_%s (%d combos) ===", name, cfg.total_runs())
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
        log.info("  ★ POSITIVE NP: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
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
        "strategy":           "Breakout_ZW_Hourly (target NP>700K round-5)",
        "symbol":             SYMBOL,
        "timeframe":          "Hourly (60 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "r4_best_np":         SEED_NP,
        "r4_best_params":     {"LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT},
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "best_params":        best,
        "best_np_attempt":    best_np,
        "best_obj_attempt":   best_obj,
        "attempts_above_target": above,
        "attempt_log":        log_,
    }
    out = OUTPUT_DIR / "final_params_zw_hourly5.json"
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
    best_obj = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ZW Hourly Breakout NP>700K Round-5 — push SE>900 + STP>40")
    log.info("  Symbol: %s  Timeframe: hourly (60 min)", SYMBOL)
    log.info("  R4 best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g NP=%.0f",
             SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP)
    log.info("  R4 key finding: SE=900 STP=40 gives NP=-788 (near-zero!)")
    log.info("  Both SE and STP were at their test UPPER LIMITS in R4.")
    log.info("  R5 pushes SE to 1500-2000+ and STP to 50-500.")
    log.info("  ZW: 1 pt = 1¢/bu × 5000 bu = $50/contract")
    log.info("  Target: %.0f USD", TARGET_NP)
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
        log.info("       Global best NP=%.0f (gap %.1f%%)",
                 best_np,
                 (TARGET_NP - best_np) / max(abs(best_np), 1) * 100
                 if best_np < TARGET_NP else 0.0)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  se_1000_1500 — extend monotonic SE trend beyond R4 boundary
    #      LE(1-3 s1) × SE(1000-1500 s25) × STP(30-60 s10) × LMT(20-50 s10)
    #      = 3×21×4×4 = 1008
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_se_1000_1500"
    _c = _cfg(_n, (1, 3, 1), (1000, 1500, 25), (30.0, 60.0, 10.0), (20.0, 50.0, 10.0))
    log.info("A01  LE(1-3 s1) × SE(1000-1500 s25) × STP(30-60 s10) × LMT(20-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  stp_50_100 — STP was capped at 40 in R4; push it higher at SE=900
    #      LE(1-3 s1) × SE(800-1000 s25) × STP(50-150 s10) × LMT(20-50 s10)
    #      = 3×9×11×4 = 1188
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_stp_50_100"
    _c = _cfg(_n, (1, 3, 1), (800, 1000, 25), (50.0, 150.0, 10.0), (20.0, 50.0, 10.0))
    log.info("A02  LE(1-3 s1) × SE(800-1000 s25) × STP(50-150 s10) × LMT(20-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  se_900_zoom — fine resolution around SE=900 with wider STP/LMT
    #      LE(1-3 s1) × SE(850-960 s10) × STP(20-80 s10) × LMT(20-50 s10)
    #      = 3×12×7×4 = 1008
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_se_900_zoom"
    _c = _cfg(_n, (1, 3, 1), (850, 960, 10), (20.0, 80.0, 10.0), (20.0, 50.0, 10.0))
    log.info("A03  LE(1-3 s1) × SE(850-960 s10) × STP(20-80 s10) × LMT(20-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  se_1500_2000 — extend even further
    #      LE(1-2 s1) × SE(1500-2000 s50) × STP(30-60 s10) × LMT(20-50 s10)
    #      = 2×11×4×4 = 352
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_se_1500_2000"
    _c = _cfg(_n, (1, 2, 1), (1500, 2000, 50), (30.0, 60.0, 10.0), (20.0, 50.0, 10.0))
    log.info("A04  LE(1-2 s1) × SE(1500-2000 s50) × STP(30-60 s10) × LMT(20-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  stp_100_500 — extremely wide stop at best SE zone
    #      LE(1-3 s1) × SE(800-1000 s50) × STP(100-500 s100) × LMT(20-60 s20)
    #      = 3×5×5×3 = 225
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_stp_100_500"
    _c = _cfg(_n, (1, 3, 1), (800, 1000, 50), (100.0, 500.0, 100.0), (20.0, 60.0, 20.0))
    log.info("A05  LE(1-3 s1) × SE(800-1000 s50) × STP(100-500 s100) × LMT(20-60 s20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  lmt_zoom — LMT was only coarsely tested (step 10) in R4; try finer
    #      LE(1-3 s1) × SE(850-950 s25) × STP(30-60 s10) × LMT(10-100 s10)
    #      = 3×5×4×10 = 600
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_lmt_zoom"
    _c = _cfg(_n, (1, 3, 1), (850, 950, 25), (30.0, 60.0, 10.0), (10.0, 100.0, 10.0))
    log.info("A06  LE(1-3 s1) × SE(850-950 s25) × STP(30-60 s10) × LMT(10-100 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  se_stp_2d — joint SE×STP sweep around the near-zero region
    #      LE(1-2 s1) × SE(850-1200 s25) × STP(30-80 s10) × LMT(20-50 s10)
    #      = 2×15×6×4 = 720
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_se_stp_2d"
    _c = _cfg(_n, (1, 2, 1), (850, 1200, 25), (30.0, 80.0, 10.0), (20.0, 50.0, 10.0))
    log.info("A07  LE(1-2 s1) × SE(850-1200 s25) × STP(30-80 s10) × LMT(20-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  se_1000_fine — highest resolution around SE=1000 crossover point
    #      LE(1-3 s1) × SE(950-1100 s10) × STP(30-70 s10) × LMT(20-50 s10)
    #      = 3×16×5×4 = 960
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_se_1000_fine"
    _c = _cfg(_n, (1, 3, 1), (950, 1100, 10), (30.0, 70.0, 10.0), (20.0, 50.0, 10.0))
    log.info("A08  LE(1-3 s1) × SE(950-1100 s10) × STP(30-70 s10) × LMT(20-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  le_at_high_se — confirm LE=1 is optimal at SE=900+
    #      LE(1-8 s1) × SE(850-950 s25) × STP(30-60 s10) × LMT(20-50 s10)
    #      = 8×5×4×4 = 640
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_le_at_high_se"
    _c = _cfg(_n, (1, 8, 1), (850, 950, 25), (30.0, 60.0, 10.0), (20.0, 50.0, 10.0))
    log.info("A09  LE(1-8 s1) × SE(850-950 s25) × STP(30-60 s10) × LMT(20-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  se_stp_hi — joint push on both SE and STP upper boundaries
    #      LE(1-2 s1) × SE(900-1300 s25) × STP(40-120 s20) × LMT(20-50 s10)
    #      = 2×17×5×4 = 680
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_se_stp_hi"
    _c = _cfg(_n, (1, 2, 1), (900, 1300, 25), (40.0, 120.0, 20.0), (20.0, 50.0, 10.0))
    log.info("A10  LE(1-2 s1) × SE(900-1300 s25) × STP(40-120 s20) × LMT(20-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  dense_zoom — adaptive progressive zoom around best NP found
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_s, _se_s, _stp_s, _lmt_s in [
        (2, 150, 40, 20, 1, 25, 10, 10),
        (2, 100, 30, 15, 1, 10, 5,  5),
        (1,  75, 20, 10, 1, 10, 5,  5),
        (1,  50, 15, 10, 1,  5, 5,  5),
    ]:
        _le  = zoom(best_le,  _r_le,  _le_s,  LE_LO,  LE_HI)
        _se  = zoom(best_se,  _r_se,  _se_s,  SE_LO,  SE_HI)
        _stp = zoom(best_stp, _r_stp, _stp_s, STP_LO, STP_HI)
        _lmt = zoom(best_lmt, _r_lmt, _lmt_s, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _le, _se, _stp, _lmt)
        if _c.total_runs() <= 5000:
            break
    log.info("A11  dense_zoom(LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)  %d combos",
             best_le, best_se, best_stp, best_lmt, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A12  boundary_r5 — global boundary with extended SE (up to 3000)
    #      LE(1-5 s2) × SE(500-3000 s250) × STP(10-200 s30) × LMT(10-100 s30)
    #      = 3×11×7×4 = 924
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary_r5"
    _c = _cfg(_n, (1, 5, 2), (500, 3000, 250), (10.0, 200.0, 30.0), (10.0, 100.0, 30.0))
    log.info("A12  LE(1-5 s2) × SE(500-3000 s250) × STP(10-200 s30) × LMT(10-100 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ROUND 5 COMPLETE")
    log.info("  Best NP  : %.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)
    log.info("  Best Obj : %.0f", best_obj)
    log.info("  Target   : %.0f  Met: %s", TARGET_NP, "YES ★" if target_met else "NO")
    log.info("══════════════════════════════════════════════════════════════")
    save_json(best_entry if best_entry else (attempt_log[-1] if attempt_log else {}),
              attempt_log, target_met)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _is_admin() -> bool:
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
    ap = argparse.ArgumentParser(description="ZW Hourly Breakout NP>700K Round-5 search")
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
