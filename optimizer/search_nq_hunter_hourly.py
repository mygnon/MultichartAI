"""
search_nq_hunter_hourly.py — SFJ_HUNTER_NQ on CME.NQ HOT Hourly, Round 1

Strategy logic (from SFJ_HUNTER_NQ.docx):
  input: STP(2000), LMT(2500), LEN(250)
  ATR = AvgTrueRange(20)
  condition1 = C > AVERAGE(C, LEN)
  IF condition1 AND marketposition<>1 AND EntriesToday(D)=0 THEN
      BUY NEXT BAR Close[1] + 2*ATR STOP
  SetStopLoss(STP)      -- fixed USD stop loss
  SetProfitTarget(LMT)  -- fixed USD profit target

Long-only. Max 1 entry per day. ATR multiplier fixed at 2.
3 params: STP (stop loss USD), LMT (profit target USD), LEN (MA period).
Target NP > 800,000 USD. ≤5,000 combos/attempt. 11 attempts.

R1 Attempt schedule:
  A01 global_wide   : LEN(10-510 s50)×STP(500-10500 s1K)×LMT(1K-21K s2K)     = 1331
  A02 large_len     : LEN(200-1000 s80)×STP(500-10500 s1K)×LMT(1K-21K s2K)   = 1331
  A03 short_len     : LEN(2-52 s5)×STP(500-10500 s1K)×LMT(1K-21K s2K)        = 1331
  A04 tight_exits   : LEN(2-52 s5)×STP(50-1050 s100)×LMT(100-2100 s200)      = 1331
  A05 wide_exits    : LEN(2-52 s5)×STP(5K-55K s5K)×LMT(10K-110K s10K)        = 1331
  A06 high_reward   : LEN(2-52 s5)×STP(500-5500 s500)×LMT(10K-110K s10K)     = 1331
  A07 medium_fine   : LEN(5-205 s20)×STP(500-10500 s1K)×LMT(1K-21K s1K)      = 2541
  A08 large_lmt     : LEN(2-52 s5)×STP(2K-22K s2K)×LMT(20K-220K s20K)        = 1331
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
SYMBOL     = "CME.NQ HOT"
SIGNAL     = "SFJ_HUNTER_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_hunter_hourly_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0   # USD

LEN_LO, LEN_HI = 1.0,      2000.0
STP_LO, STP_HI = 50.0,   100_000.0
LMT_LO, LMT_HI = 100.0, 1_000_000.0

# Strategy defaults as seed
SEED_LEN = 250.0
SEED_STP = 2_000.0
SEED_LMT = 2_500.0
SEED_NP  = 0.0

PREFIX = "NQHH_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_hunter_hourly_{int(time.time())}.log"
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


def zoom_fixed(center: float, radius: float, n_target: int,
               step_min: float, lo: float, hi: float) -> Tuple[float, float, float]:
    """
    Returns (start, stop, step) with AT MOST n_target values.
    Step is computed via math.ceil so combos are guaranteed ≤ n_target.
    """
    lo_val = max(lo, center - radius)
    hi_val = min(hi, center + radius)
    if lo_val >= hi_val:
        return (max(lo, center - step_min), min(hi, center + step_min), step_min)
    rng = hi_val - lo_val
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
        "round": 1,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C > AVERAGE(C,LEN) AND EntriesToday=0 → next bar STOP at Close[1]+2×ATR(20)",
            "exits":    "SetStopLoss(STP) + SetProfitTarget(LMT) — fixed USD amounts; max 1 entry/day",
            "params":   "STP (stop loss USD), LMT (profit target USD), LEN (MA period)",
            "defaults": "STP=2000, LMT=2500, LEN=250",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_hunter_hourly.json"
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
    log.info("  SFJ_HUNTER_NQ on CME.NQ HOT Hourly — Round 1")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Defaults: LEN=%.4g STP=%.4g LMT=%.4g",
             SEED_LEN, SEED_STP, SEED_LMT)
    log.info("  Target: %.0f USD", TARGET_NP)
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
                 "★TARGET★" if met else ("NP=%.0f/800K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  global_wide — coarse survey of small USD STP/LMT range
    #      LEN(10-510 s50) × STP(500-10500 s1K) × LMT(1K-21K s2K) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_global_wide"
    _c = _cfg(_n, (10, 510, 50), (500, 10500, 1000), (1000, 21000, 2000))
    log.info("A01  LEN(10-510 s50)×STP(500-10.5K s1K)×LMT(1K-21K s2K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A02  large_len — test large MA period
    #      LEN(200-1000 s80) × STP(500-10500 s1K) × LMT(1K-21K s2K) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_large_len"
    _c = _cfg(_n, (200, 1000, 80), (500, 10500, 1000), (1000, 21000, 2000))
    log.info("A02  LEN(200-1000 s80)×STP(500-10.5K s1K)×LMT(1K-21K s2K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A03  short_len — small MA period (likely inert as per TXF findings)
    #      LEN(2-52 s5) × STP(500-10500 s1K) × LMT(1K-21K s2K) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_short_len"
    _c = _cfg(_n, (2, 52, 5), (500, 10500, 1000), (1000, 21000, 2000))
    log.info("A03  LEN(2-52 s5)×STP(500-10.5K s1K)×LMT(1K-21K s2K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A04  tight_exits — very tight USD stop/target (scalping regime)
    #      LEN(2-52 s5) × STP(50-1050 s100) × LMT(100-2100 s200) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_tight_exits"
    _c = _cfg(_n, (2, 52, 5), (50, 1050, 100), (100, 2100, 200))
    log.info("A04  LEN(2-52 s5)×STP(50-1050 s100)×LMT(100-2100 s200)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A05  wide_exits — large USD stop/target
    #      LEN(2-52 s5) × STP(5K-55K s5K) × LMT(10K-110K s10K) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_wide_exits"
    _c = _cfg(_n, (2, 52, 5), (5000, 55000, 5000), (10000, 110000, 10000))
    log.info("A05  LEN(2-52 s5)×STP(5K-55K s5K)×LMT(10K-110K s10K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A06  high_reward — low STP + high LMT (like TXF R/R≈25:1 regime)
    #      LEN(2-52 s5) × STP(500-5500 s500) × LMT(10K-110K s10K) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_reward"
    _c = _cfg(_n, (2, 52, 5), (500, 5500, 500), (10000, 110000, 10000))
    log.info("A06  LEN(2-52 s5)×STP(500-5.5K s500)×LMT(10K-110K s10K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A07  medium_fine — moderate finer grid
    #      LEN(5-205 s20) × STP(500-10500 s1K) × LMT(1K-21K s1K) = 11×11×21=2541
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_medium_fine"
    _c = _cfg(_n, (5, 205, 20), (500, 10500, 1000), (1000, 21000, 1000))
    log.info("A07  LEN(5-205 s20)×STP(500-10.5K s1K)×LMT(1K-21K s1K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A08  large_lmt — large LMT territory ($20K-$220K), low-freq regime
    #      LEN(2-52 s5) × STP(2K-22K s2K) × LMT(20K-220K s20K) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_large_lmt"
    _c = _cfg(_n, (2, 52, 5), (2000, 22000, 2000), (20000, 220000, 20000))
    log.info("A08  LEN(2-52 s5)×STP(2K-22K s2K)×LMT(20K-220K s20K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — wide zoom centered on best NP champion
    #      zoom_fixed() with n_target=15 per dim → ≤15^3=3375 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 8.0
        r_stp = max(2000.0, best_STP * 0.42)
        r_lmt = max(10000.0, best_LMT * 0.42)
        _len = zoom_fixed(best_LEN, r_len,  15,    1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp,  15,   50.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt,  15,  500.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A09  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — medium zoom centered on best
    #      zoom_fixed() with n_target=11 per dim → ≤11^3=1331 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 4.0
        r_stp = max(500.0, best_STP * 0.17)
        r_lmt = max(2000.0, best_LMT * 0.10)
        _len = zoom_fixed(best_LEN, r_len, 11,   1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp, 11,  50.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt, 11, 200.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A10  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — fine zoom centered on best
    #      zoom_fixed() with n_target=9 per dim → ≤9^3=729 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 2.0
        r_stp = max(200.0, best_STP * 0.09)
        r_lmt = max(1000.0, best_LMT * 0.05)
        _len = zoom_fixed(best_LEN, r_len, 9,   1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp, 9,  50.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt, 9, 100.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A11  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER_NQ NQ Hourly Round-1 COMPLETE")
    log.info("  Champion: LEN=%.4g STP=%.4g LMT=%.4g", best_LEN, best_STP, best_LMT)
    log.info("  Best NP: %.0f USD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    log.info("  Target %.0f USD: %s", TARGET_NP,
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
    print(f"Target NP>800K USD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="SFJ_HUNTER_NQ CME.NQ HOT Hourly R1 parameter search")
    parser.add_argument("--from-csv",  action="store_true",
                        help="Re-analyse existing CSVs without running MC64")
    parser.add_argument("--attempt",   type=int, default=1,
                        help="Start from this attempt number (1-11)")
    parser.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

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
