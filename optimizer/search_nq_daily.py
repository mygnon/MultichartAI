"""
search_nq_daily.py — Breakout Daily NP > 700,000 on CME.NQ HOT, Round 1

No prior NQ daily results — starting fresh with wide exploration.
Target: NP > 700,000 USD

NQ vs TWF context:
  - NQ in USD (1 NQ pt = $20);  TWF in TWD (1 TXF pt = NT$200)
  - USD/TWD ≈ 32;  700K USD ≈ 22.4M TWD
  - TWF daily best: LE=5 SE=49 STP=0.2 LMT=16  NP=4,074,200 TWD (~127K USD)
  - NQ hourly best: LE=1 SE=9  STP=2  LMT=14   NP=656,575 USD (6168 trades)
  - Daily bars = much fewer trades; SE and LE likely larger than hourly

Round-1 strategy: Wide exploration across all parameter regions
  R1.1  Coarse 4D scan — identify which LE×SE×STP×LMT region gives positive NP
  R1.2  Low SE (3-20) — short lookback like NQ hourly (SE=9 won there)
  R1.3  Mid SE (20-70) — medium-term breakout
  R1.4  High SE (60-180) — long-term trend-following
  R1.5  Very low STP (0.1-1.0) — tight stops, many small losses
  R1.6  High LMT (30-80) — large profit targets, few big wins
  R1.7  LE probe (1-20) — sweep LE with proven SE/STP/LMT from A01-A04
  R1.8  Adaptive zoom around best NP found so far
  R1.9  4D medium breadth with accumulated knowledge
  R1.10 Fine STP×LMT around best SE region
  R1.11 Dense zoom (progressive shrink)
  R1.12 Boundary sweep — global sanity check

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  LE(1-10 s3) × SE(5-100 s10) × STP(1-6 s1)    × LMT(5-30 s5)       4×10×6×6 =1440
  A02  LE(1-6 s1)  × SE(3-20 s1)   × STP(1-6 s1)    × LMT(5-20 s3)       6×18×6×6 =3888
  A03  LE(2-8 s1)  × SE(20-70 s5)  × STP(1-5 s1)    × LMT(8-25 s3)       7×11×5×6 =2310
  A04  LE(2-8 s1)  × SE(60-180 s10)× STP(1-5 s1)    × LMT(10-30 s5)      7×13×5×5 =2275
  A05  LE(2-7 s1)  × SE(10-60 s5)  × STP(0.1-1 s0.1)× LMT(5-20 s3)      6×11×10×6=3960
  A06  LE(2-7 s1)  × SE(10-60 s5)  × STP(1-4 s1)    × LMT(30-80 s5)      6×11×4×11=2904
  A07  LE(1-19 s2) × SE(10-50 s5)  × STP(1-5 s1)    × LMT(8-20 s3)       10×9×5×5 =2250
  A08  Adaptive zoom around best NP (LE±2 × SE±15 × STP±1 × LMT±5)
  A09  LE(1-6 s1)  × SE(5-55 s5)   × STP(0.5-3 s0.5)× LMT(5-25 s4)      6×11×6×6 =2376
  A10  Adaptive fine zoom around NP champion
  A11  Dense zoom (progressive shrink, ≤5000)
  A12  LE(1-17 s4) × SE(5-200 s20) × STP(0.5-8 s1)  × LMT(5-50 s5)      5×10×8×10=4000
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_daily_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 700_000.0
MAX_ATTEMPTS = 12

# Generous bounds — actual per-attempt ranges are much tighter
STP_LO, STP_HI = 0.05, 50.0
LMT_LO, LMT_HI = 2.0,  200.0
SE_LO,  SE_HI  = 1.0,  400.0
LE_LO,  LE_HI  = 1.0,  100.0

# No prior NQ daily results — seed with neutral starting point
SEED_LE,  SEED_SE  = 5.0,  20.0
SEED_STP, SEED_LMT = 3.0,  15.0
SEED_NP   = 0.0
SEED_OBJ  = 0.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_daily_{int(time.time())}.log"
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
# Range helpers
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


# ─────────────────────────────────────────────────────────────────────────────
# Config factory — ALWAYS vary all 4 params to avoid column-mapping bug
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(name: str,
         le:  Tuple[float, float, float],
         se:  Tuple[float, float, float],
         stp: Tuple[float, float, float],
         lmt: Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t: Tuple[float, float, float]) -> Tuple[float, float, float]:
        s, e, step = t
        if s == e:
            return (max(LE_LO, s - step), min(LE_HI, s + step), step)
        return t
    le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)

    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000 limit!", name, combos)
    return StrategyConfig(
        name=f"NQD_{name}",
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


# ─────────────────────────────────────────────────────────────────────────────
# Run or load CSV
# ─────────────────────────────────────────────────────────────────────────────

def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"NQD_{name}_raw.csv"


def run_or_load(name: str, cfg: StrategyConfig,
                conn: Optional[mc.MultiChartsConnection],
                from_csv: bool) -> Optional[pd.DataFrame]:
    csv_path = csv_for(name)
    if from_csv or csv_path.exists():
        if csv_path.exists():
            try:
                df = mc.load_results_csv(str(csv_path), cfg)
                log.info("Loaded %s: %d rows from %s", name, len(df), csv_path.name)
                return df
            except Exception as e:
                log.warning("Could not load %s: %s", csv_path, e)
        else:
            log.warning("No CSV for %s at %s", name, csv_path)
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


# ─────────────────────────────────────────────────────────────────────────────
# Validate CSV param columns are within expected range
# ─────────────────────────────────────────────────────────────────────────────

def _validate_df(df: pd.DataFrame, cfg: StrategyConfig) -> bool:
    if df is None or df.empty:
        return False
    for p in cfg.params:
        if p.name not in df.columns:
            continue
        lo = min(p.start, p.stop) - abs(p.step) * 2
        hi = max(p.start, p.stop) + abs(p.step) * 2
        col = pd.to_numeric(df[p.name], errors="coerce")
        if not col.between(lo, hi).all():
            log.warning("  INVALID: %s column out of expected range [%.4g, %.4g]. "
                        "Got min=%.4g max=%.4g. Skipping.",
                        p.name, lo, hi, col.min(), col.max())
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Champion selector — NP-max for zoom seed (target chasing)
# ─────────────────────────────────────────────────────────────────────────────

def champion(df: pd.DataFrame,
             fb_le: float, fb_se: float,
             fb_stp: float, fb_lmt: float,
             ) -> Tuple[float, float, float, float, float, float, float, int, bool]:
    """Return (LE, SE, STP, LMT, objective, NP, MDD, trades, target_met).
    Zoom seed = row with highest NP (target-chasing mode)."""
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

    # Use highest NP as zoom seed (not highest objective) — chase the target
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


# ─────────────────────────────────────────────────────────────────────────────
# Record helpers
# ─────────────────────────────────────────────────────────────────────────────

def _entry(attempt: int, name: str, df: Optional[pd.DataFrame],
           le: float, se: float, stp: float, lmt: float,
           obj: float, np_: float, mdd: float, trades: int,
           met: bool, combos: int) -> Dict:
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


def save_json(best: Dict, best_np: Dict, log_: List[dict], met: bool) -> Path:
    above = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    payload = {
        "strategy":           "Breakout_NQ_Daily  (target NP>700K round-1)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "best_params":            best,
        "best_np_attempt":        best_np,
        "attempts_above_target":  above,
        "attempt_log":            log_,
    }
    out = OUTPUT_DIR / "final_params_nq_daily.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn: Optional[mc.MultiChartsConnection],
               from_csv: bool, start_attempt: int) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met   = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}
    best_np_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Round-1 NQ Daily NP>700K  (all-4-param variation enforced)")
    log.info("  Symbol: %s   Signal: %s   Timeframe: daily", SYMBOL, SIGNAL)
    log.info("  Seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  (no prior NP)",
             best_le, best_se, best_stp, best_lmt)
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df: Optional[pd.DataFrame], cfg: StrategyConfig,
                name: str, attempt_num: int, combos: int) -> None:
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry, best_np_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            entry = _entry(attempt_num, name, df,
                           best_le, best_se, best_stp, best_lmt,
                           0.0, 0.0, 0.0, 0, False, combos)
            attempt_log.append(entry)
            log.info("  [A%02d %-25s]  no valid data", attempt_num, name)
            return

        le, se, stp, lmt, obj, np_, mdd, tr, met = champion(
            df, best_le, best_se, best_stp, best_lmt)

        # Update zoom seed: NP-max (target chasing)
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

        # Track best overall (obj-champion) and best NP separately
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
                 obj, np_, mdd, tr,
                 "★TARGET★" if met else "")
        log.info("       Global best NP=%.0f (need +%.1f%%)",
                 best_np,
                 (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else float("inf"))

    # ──────────────────────────────────────────────────────────────────────
    # A01  Coarse 4D scan  LE(1-10 s3) × SE(5-100 s10) × STP(1-6) × LMT(5-30 s5)
    #      Goal: find which region (low/mid/high SE) has positive NP  (4×10×6×6=1440)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_coarse_scan"
    _c = _cfg(_n, (1, 10, 3), (5, 100, 10), (1.0, 6.0, 1.0), (5, 30, 5))
    log.info("A01  LE(1-10 s3) × SE(5-100 s10) × STP(1-6) × LMT(5-30 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  Low SE (3-20) — NQ hourly favored SE=9; test on daily too
    #      LE(1-6) × SE(3-20 s1) × STP(1-6) × LMT(5-20 s3)   (6×18×6×6=3888)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_low_se"
    _c = _cfg(_n, (1, 6, 1), (3, 20, 1), (1.0, 6.0, 1.0), (5, 20, 3))
    log.info("A02  LE(1-6) × SE(3-20 s1) × STP(1-6) × LMT(5-20 s3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  Mid SE (20-70) — medium-term breakout
    #      LE(2-8) × SE(20-70 s5) × STP(1-5) × LMT(8-25 s3)  (7×11×5×6=2310)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_mid_se"
    _c = _cfg(_n, (2, 8, 1), (20, 70, 5), (1.0, 5.0, 1.0), (8, 25, 3))
    log.info("A03  LE(2-8) × SE(20-70 s5) × STP(1-5) × LMT(8-25 s3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  High SE (60-180) — long-term trend-following
    #      LE(2-8) × SE(60-180 s10) × STP(1-5) × LMT(10-30 s5)  (7×13×5×5=2275)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_high_se"
    _c = _cfg(_n, (2, 8, 1), (60, 180, 10), (1.0, 5.0, 1.0), (10, 30, 5))
    log.info("A04  LE(2-8) × SE(60-180 s10) × STP(1-5) × LMT(10-30 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  Very low STP (0.1-1.0 step 0.1) — tight stops like TWF daily (STP=0.2)
    #      LE(2-7) × SE(10-60 s5) × STP(0.1-1 s0.1) × LMT(5-20 s3)  (6×11×10×6=3960)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_very_low_stp"
    _c = _cfg(_n, (2, 7, 1), (10, 60, 5), (0.1, 1.0, 0.1), (5, 20, 3))
    log.info("A05  LE(2-7) × SE(10-60 s5) × STP(0.1-1 s0.1) × LMT(5-20 s3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  High LMT (30-80) — large profit targets, few big wins
    #      LE(2-7) × SE(10-60 s5) × STP(1-4) × LMT(30-80 s5)  (6×11×4×11=2904)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_lmt"
    _c = _cfg(_n, (2, 7, 1), (10, 60, 5), (1.0, 4.0, 1.0), (30, 80, 5))
    log.info("A06  LE(2-7) × SE(10-60 s5) × STP(1-4) × LMT(30-80 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  LE probe — wide LE sweep with proven STP/LMT region from A01-A06
    #      LE(1-19 s2) × SE(10-50 s5) × STP(1-5) × LMT(8-20 s3)  (10×9×5×5=2250)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_le_probe"
    _c = _cfg(_n, (1, 19, 2), (10, 50, 5), (1.0, 5.0, 1.0), (8, 20, 3))
    log.info("A07  LE(1-19 s2) × SE(10-50 s5) × STP(1-5) × LMT(8-20 s3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  Adaptive zoom around best NP found so far
    #      LE±2 × SE±15 step 2 × STP±1.5 step 0.25 × LMT±5 step 1
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_zoom_best"
    for _r_se, _r_stp, _r_lmt in [(15, 1.5, 5), (10, 1.0, 4), (8, 0.8, 3), (5, 0.5, 2)]:
        _le8  = zoom(best_le,  2,      1,    LE_LO,  LE_HI)
        _se8  = zoom(best_se,  _r_se,  2,    SE_LO,  SE_HI)
        _stp8 = zoom(best_stp, _r_stp, 0.25, STP_LO, STP_HI)
        _lmt8 = zoom(best_lmt, _r_lmt, 1.0,  LMT_LO, LMT_HI)
        _c = _cfg(_n, _le8, _se8, _stp8, _lmt8)
        if _c.total_runs() <= 5000:
            break
    log.info("A08  Zoom best LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le8, _se8, _stp8, _lmt8, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  4D medium breadth — broad with finer step
    #      LE(1-6) × SE(5-55 s5) × STP(0.5-3 s0.5) × LMT(5-25 s4)  (6×11×6×6=2376)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_4d_medium"
    _c = _cfg(_n, (1, 6, 1), (5, 55, 5), (0.5, 3.0, 0.5), (5, 25, 4))
    log.info("A09  LE(1-6) × SE(5-55 s5) × STP(0.5-3 s0.5) × LMT(5-25 s4)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  Fine STP×LMT precision zoom around NP champion
    #      LE±1 × SE±8 step 2 × STP±0.8 step 0.1 × LMT±4 step 0.5
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_stp_lmt_fine"
    for _r_se, _r_stp, _r_lmt in [(8, 0.8, 4), (6, 0.6, 3), (4, 0.4, 2), (3, 0.3, 1.5)]:
        _le10  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se10  = zoom(best_se,  _r_se,  2,    SE_LO,  SE_HI)
        _stp10 = zoom(best_stp, _r_stp, 0.1,  STP_LO, STP_HI)
        _lmt10 = zoom(best_lmt, _r_lmt, 0.5,  LMT_LO, LMT_HI)
        _c = _cfg(_n, _le10, _se10, _stp10, _lmt10)
        if _c.total_runs() <= 5000:
            break
    log.info("A10  Fine zoom LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le10, _se10, _stp10, _lmt10, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom around NP champion (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt in [(6, 0.5, 3), (4, 0.4, 2), (3, 0.3, 1.5), (2, 0.2, 1)]:
        _le11  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se11  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, _r_stp, 0.1,  STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, _r_lmt, 0.5,  LMT_LO, LMT_HI)
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
    # A12  Boundary sweep — global sanity check
    #      LE(1-17 s4) × SE(5-200 s20) × STP(0.5-8 s1) × LMT(5-50 s5)  (5×10×8×10=4000)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (1, 17, 4), (5, 200, 20), (0.5, 8.0, 1.0), (5, 50, 5))
    log.info("A12  LE(1-17 s4) × SE(5-200 s20) × STP(0.5-8) × LMT(5-50 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    if not best_entry:
        best_entry = _entry(0, "seed", None,
                            SEED_LE, SEED_SE, SEED_STP, SEED_LMT,
                            0.0, 0.0, 0.0, 0, False, 0)
    if not best_np_entry:
        best_np_entry = best_entry

    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL RESULT — Round-1 NQ Daily")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best NP: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
             best_np_entry.get("LE", best_le), best_np_entry.get("SE", best_se),
             best_np_entry.get("STP", best_stp), best_np_entry.get("LMT", best_lmt),
             best_np_entry.get("net_profit", best_np),
             best_np_entry.get("max_drawdown", 0),
             best_np_entry.get("objective", best_obj),
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
# Entry point with auto-elevation
# ─────────────────────────────────────────────────────────────────────────────

def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _auto_elevate() -> None:
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
        print(f"[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Round-1 NQ Daily search — target NP > 700K USD")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyze existing CSVs without running MC64")
    ap.add_argument("--attempt", type=int, default=1, metavar="N",
                    help="Resume from attempt N (skip earlier)")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0

    conn: Optional[mc.MultiChartsConnection] = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
