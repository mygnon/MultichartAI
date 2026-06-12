"""
search_txf_hunter_hourly2.py — SFJ_HUNTER_NQ on TWF.TXF HOT Hourly, Round 2

Strategy logic (from SFJ_HUNTER_NQ.docx):
  input: STP(2000), LMT(2500), LEN(250)
  ATR = AvgTrueRange(20)
  condition1 = C > AVERAGE(C, LEN)
  IF condition1 AND marketposition<>1 AND EntriesToday(D)=0 THEN
      BUY NEXT BAR Close[1] + 2*ATR STOP
  SetStopLoss(STP)      -- fixed TWD stop loss
  SetProfitTarget(LMT)  -- fixed TWD profit target

Long-only. Max 1 entry per day. ATR multiplier fixed at 2.
3 params: STP (stop loss TWD), LMT (profit target TWD), LEN (MA period).
Target NP > 7,000,000 TWD. ≤5,000 combos/attempt. 11 attempts.

R1 Champion: LEN=6, STP=23,900, LMT=607,400 → NP=4,974,600 MDD=-1,007,000 trades=103 (gap −28.9%)

R1 Key findings:
  - LMT=607K is a TRUE PEAK (A09 ran to 660K; top-10K export confirms max LMT=607K)
  - LEN is completely INERT: LEN=6/8/10/12 all give identical NP=4,974,600 (MA filter rarely binding)
  - STP=23,900-24,000 is optimal; ±100 TWD drop from 24,000 already reduces NP
  - R/R = 607K/24K ≈ 25:1; ~11.7% win rate; ~12 wins out of 103 trades; each win ≈607K TWD
  - Zoom overflow bug in R1: A09/A10/A11 each ran 35K-94K combos (exceeded 5000); fix: zoom_fixed()
  - Gap −28.9%; still improving (+3.6%→+1.7%) → R2 needed

R2 Attempt schedule:
  A01 fine_confirm    : LEN(2-12 s1)×STP(17K-31K s1K)×LMT(550K-680K s10K)  = 2310
  A02 lmt_extend      : LEN(4-10 s1)×STP(20K-28K s2K)×LMT(650K-1.5M s50K)  = 630
  A03 lmt_extreme     : LEN(4-10 s1)×STP(15K-35K s5K)×LMT(1M-5M s250K)     = 595
  A04 stp_landscape   : LEN(4-10 s1)×STP(1K-100K s5K)×LMT(575K-625K s25K)  = 420
  A05 medium_lmt      : LEN(40-100 s10)×STP(5K-25K s2K)×LMT(80K-600K s40K) = 1078
  A06 tight_exits     : LEN(2-12 s1)×STP(200-4200 s400)×LMT(500-10500 s1K)  = 1331
  A07 global_confirm  : LEN(1-200 s20)×STP(2K-80K s8K)×LMT(10K-2M s100K)   = 2541
  A08 stp_fine        : LEN(4-10 s1)×STP(21K-27K s500)×LMT(580K-640K s10K) = 637
  A09-A11 adaptive zoom — zoom_fixed() guarantees ≤5000 combos
"""
from __future__ import annotations
import argparse
import ctypes
import json
import logging
import math
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260101_SFJ_HUNTER_AI.wsp"
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "SFJ_HUNTER_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_hunter_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 7_000_000.0

LEN_LO, LEN_HI = 1.0,     2000.0
STP_LO, STP_HI = 200.0,   500_000.0
LMT_LO, LMT_HI = 200.0, 5_000_000.0

# R1 champion as seed
SEED_LEN = 6.0
SEED_STP = 23_900.0
SEED_LMT = 607_400.0
SEED_NP  = 4_974_600.0

PREFIX = "TXFHH2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_hunter_hourly2_{int(time.time())}.log"
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
    """Original zoom helper — use zoom_fixed for the adaptive attempts."""
    start = max(lo, _snap(center - radius, step))
    stop  = min(hi, _snap(center + radius, step))
    if stop <= start:
        stop = start + step
    return (start, stop, step)


def zoom_fixed(center: float, radius: float, n_target: int,
               step_min: float, lo: float, hi: float) -> Tuple[float, float, float]:
    """
    Returns (start, stop, step) with AT MOST n_target values.
    Step is computed via math.ceil so combos are guaranteed ≤ n_target.
    Use this to prevent zoom grids from exceeding 5000 combos.
    """
    lo_val = max(lo, center - radius)
    hi_val = min(hi, center + radius)
    if lo_val >= hi_val:
        return (max(lo, center - step_min), min(hi, center + step_min), step_min)
    rng = hi_val - lo_val
    # ceil(range / (n-1) / step_min) * step_min ensures n_vals ≤ n_target
    step = max(step_min, math.ceil(rng / max(1, n_target - 1) / step_min) * step_min)
    return (lo_val, hi_val, step)


def n_vals(t: Tuple[float, float, float]) -> int:
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name: str,
         length: Tuple[float, float, float],
         stp:    Tuple[float, float, float],
         lmt:    Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    length = _safe(length, LEN_LO, LEN_HI)
    stp    = _safe(stp,    STP_LO, STP_HI)
    lmt    = _safe(lmt,    LMT_LO, LMT_HI)

    combos = n_vals(length) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("STP", *stp),
            ParamAxis("LMT", *lmt),
            ParamAxis("LEN", *length),
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


def champion(df, fb_len, fb_stp, fb_lmt):
    """Target-chasing mode: highest-NP row drives zoom seed."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LEN=%.4g STP=%.4g LMT=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LEN"]), float(best["STP"]), float(best["LMT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["LEN"]), float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LEN=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LEN"]), float(best["STP"]), float(best["LMT"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["LEN"]), float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_len, fb_stp, fb_lmt,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, length, stp, lmt, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "LEN": length, "STP": stp, "LMT": lmt,
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
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":     "BUY when C > AVERAGE(C,LEN) AND EntriesToday=0 → next bar STOP at Close[1]+2×ATR(20)",
            "exits":     "SetStopLoss(STP) + SetProfitTarget(LMT) — fixed TWD amounts; max 1 entry/day",
            "r1_champion": "LEN=6 STP=23900 LMT=607400 NP=4974600 MDD=-1007000 trades=103 (gap -28.9%)",
            "r1_findings": "LMT=607K true peak (A09 ran to 660K, max in top-10K=607K); LEN inert (6/8/10/12 all identical); zoom overflow bug fixed in R2 via zoom_fixed()",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_hunter_hourly2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_LEN = SEED_LEN
    best_STP = SEED_STP
    best_LMT = SEED_LMT
    best_np  = SEED_NP
    best_obj = 0.0
    target_met  = False
    attempt_log: List[dict] = []
    best_entry:  Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER_NQ on TWF.TXF HOT Hourly — Round 2")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  R1 seed: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    log.info("  R1 finding: LMT=607K true peak; LEN inert; zoom_fixed() replaces buggy zoom")
    log.info("  Target: %.0f TWD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_LEN, best_STP, best_LMT
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_LEN, best_STP, best_LMT,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        length, stp, lmt, obj, np_, mdd, tr, met = champion(
            df, best_LEN, best_STP, best_LMT)

        if np_ > best_np:
            best_LEN, best_STP, best_LMT = length, stp, lmt
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, length, stp, lmt,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  LEN=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, length, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/7M" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  fine_confirm — tight grid around R1 champion
    #      LEN(2-12 s1) × STP(17K-31K s1K) × LMT(550K-680K s10K) = 11×15×14=2310
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_confirm"
    _c = _cfg(_n, (2, 12, 1), (17000, 31000, 1000), (550000, 680000, 10000))
    log.info("A01  LEN(2-12 s1)×STP(17K-31K s1K)×LMT(550K-680K s10K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A02  lmt_extend — probe LMT beyond A09's 660K limit
    #      LEN(4-10 s1) × STP(20K-28K s2K) × LMT(650K-1.5M s50K) = 7×5×18=630
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_lmt_extend"
    _c = _cfg(_n, (4, 10, 1), (20000, 28000, 2000), (650000, 1500000, 50000))
    log.info("A02  LEN(4-10 s1)×STP(20K-28K s2K)×LMT(650K-1.5M s50K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A03  lmt_extreme — very large profit targets (1M-5M)
    #      LEN(4-10 s1) × STP(15K-35K s5K) × LMT(1M-5M s250K) = 7×5×17=595
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_lmt_extreme"
    _c = _cfg(_n, (4, 10, 1), (15000, 35000, 5000), (1000000, 5000000, 250000))
    log.info("A03  LEN(4-10 s1)×STP(15K-35K s5K)×LMT(1M-5M s250K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A04  stp_landscape — wide STP survey at champion LMT
    #      LEN(4-10 s1) × STP(1K-100K s5K) × LMT(575K-625K s25K) = 7×20×3=420
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_stp_landscape"
    _c = _cfg(_n, (4, 10, 1), (1000, 100000, 5000), (575000, 625000, 25000))
    log.info("A04  LEN(4-10 s1)×STP(1K-100K s5K)×LMT(575K-625K s25K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A05  medium_lmt — explore the medium LMT regime (80K-600K, A06 in R1)
    #      LEN(40-100 s10) × STP(5K-25K s2K) × LMT(80K-600K s40K) = 7×11×14=1078
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_medium_lmt"
    _c = _cfg(_n, (40, 100, 10), (5000, 25000, 2000), (80000, 600000, 40000))
    log.info("A05  LEN(40-100 s10)×STP(5K-25K s2K)×LMT(80K-600K s40K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A06  tight_exits — scalping regime (STP≤4K, LMT≤10K)
    #      LEN(2-12 s1) × STP(200-4200 s400) × LMT(500-10500 s1K) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_tight_exits"
    _c = _cfg(_n, (2, 12, 1), (200, 4200, 400), (500, 10500, 1000))
    log.info("A06  LEN(2-12 s1)×STP(200-4200 s400)×LMT(500-10500 s1K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A07  global_confirm — coarse-wide sweep to rule out undiscovered regimes
    #      LEN(1-200 s20) × STP(2K-80K s8K) × LMT(10K-2M s100K) = 11×11×21=2541
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_global_confirm"
    _c = _cfg(_n, (1, 200, 20), (2000, 80000, 8000), (10000, 2000000, 100000))
    log.info("A07  LEN(1-200 s20)×STP(2K-80K s8K)×LMT(10K-2M s100K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A08  stp_fine — precise STP sweep (step=500 TWD) at champion LMT
    #      LEN(4-10 s1) × STP(21K-27K s500) × LMT(580K-640K s10K) = 7×13×7=637
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_stp_fine"
    _c = _cfg(_n, (4, 10, 1), (21000, 27000, 500), (580000, 640000, 10000))
    log.info("A08  LEN(4-10 s1)×STP(21K-27K s500)×LMT(580K-640K s10K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — wide zoom centered on best
    #      Uses zoom_fixed() with n_target=15 per dim → ≤15^3=3375 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len  = 8.0
        r_stp  = max(10000.0, best_STP * 0.42)
        r_lmt  = max(150000.0, best_LMT * 0.42)
        _len = zoom_fixed(best_LEN, r_len,  15,     1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp,  15,  1000.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt,  15, 20000.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A09  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — medium zoom centered on best
    #      Uses zoom_fixed() with n_target=11 per dim → ≤11^3=1331 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len  = 4.0
        r_stp  = max(4000.0, best_STP * 0.17)
        r_lmt  = max(40000.0, best_LMT * 0.10)
        _len = zoom_fixed(best_LEN, r_len, 11,    1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp, 11,  500.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt, 11, 5000.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A10  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — fine zoom centered on best
    #      Uses zoom_fixed() with n_target=9 per dim → ≤9^3=729 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len  = 2.0
        r_stp  = max(2000.0, best_STP * 0.09)
        r_lmt  = max(20000.0, best_LMT * 0.05)
        _len = zoom_fixed(best_LEN, r_len, 9,   1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp, 9, 200.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt, 9, 2000.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A11  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER_NQ TXF Hourly Round-2 COMPLETE")
    log.info("  Champion: LEN=%.4g STP=%.4g LMT=%.4g", best_LEN, best_STP, best_LMT)
    log.info("  Best NP: %.0f TWD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    log.info("  Target %.0f TWD: %s", TARGET_NP,
             "★ MET" if target_met else "NOT MET (gap +%.0f)" % max(0, TARGET_NP - best_np))
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LEN": best_LEN, "STP": best_STP, "LMT": best_LMT,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>7M TWD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
    parser = argparse.ArgumentParser(
        description="SFJ_HUNTER_NQ TXF Hourly R2 parameter search")
    parser.add_argument("--from-csv",  action="store_true",
                        help="Re-analyse existing CSVs without running MC64")
    parser.add_argument("--attempt",   type=int, default=1,
                        help="Start from this attempt number (1-11)")
    parser.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()

    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
