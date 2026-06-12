"""
search_nq_hunter_hourly2.py — SFJ_HUNTER_NQ on CME.NQ HOT Hourly, Round 2

R1 Champion: LEN=7, STP=550, LMT=13,500 → NP=$500,015 MDD=-$21,485 trades=657 (gap -37.5%)

R1 Key findings:
  - TWO equal peaks at ~$492-500K:
      Peak 1: LEN=7 STP=550 LMT=13,500 → $500K (R/R=24.5:1, 657 trades, win rate ~9.3%)
      Peak 2: LEN=7 STP=250 LMT=22,500 → $492K (R/R=90:1,  590 trades, win rate ~4.8%)
  - STP=550 is the precise optimum: bell curve confirmed (STP=450→$492K < STP=550→$500K > STP=650→$483K)
  - LEN=7 uniquely optimal: LEN=6/8 both give $484K (-3%)
  - LMT=13,500 is the Peak 1 top
  - Zoom non-monotonic (A09=$492K → A10=$500K → A11=$492K) — near noise plateau
  - UNEXPLORED: LMT=25K-200K with STP=100-600 (A08 only tested STP=2K-22K; A09 LMT max=23K)
  - LMT=13.5K regime is high-frequency (94/yr); reaching $800K requires +60% gain — unlikely in Peak 1

R2 Attempt schedule:
  A01 peak2_zoom     : LEN(4-10 s1)×STP(100-600 s50)×LMT(18K-30K s1K)        = 819
  A02 lmt_bridge     : LEN(4-10 s1)×STP(100-700 s100)×LMT(25K-275K s25K)     = 539
  A03 ultra_lmt      : LEN(4-10 s1)×STP(200-1200 s200)×LMT(100K-600K s50K)   = 462
  A04 fine_peak1     : LEN(5-9 s1)×STP(350-800 s50)×LMT(12K-15.5K s250)      = 5×10×15=750
  A05 len_scan       : LEN(1-200 s10)×STP(400-700 s100)×LMT(12K-15K s1K)     = 21×4×4=336
  A06 global_confirm : LEN(1-200 s20)×STP(100-10100 s1K)×LMT(5K-105K s10K)   = 1331
  A07 high_lmt_allSTP: LEN(4-10 s1)×STP(200-600 s100)×LMT(20K-100K s10K)     = 7×5×9=315
  A08 stp_loLMT      : LEN(5-9 s1)×STP(50-1550 s100)×LMT(12K-16K s500)       = 5×16×9=720
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_hunter_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0   # USD

LEN_LO, LEN_HI = 1.0,      2000.0
STP_LO, STP_HI = 50.0,   100_000.0
LMT_LO, LMT_HI = 100.0, 1_000_000.0

# R1 champion as seed
SEED_LEN = 7.0
SEED_STP = 550.0
SEED_LMT = 13_500.0
SEED_NP  = 500_015.0

PREFIX = "NQHH2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_hunter_hourly2_{int(time.time())}.log"
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

def zoom_fixed(center: float, radius: float, n_target: int,
               step_min: float, lo: float, hi: float) -> Tuple[float, float, float]:
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
    """Target-chasing: highest-NP row drives zoom seed."""
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
            "logic":        "BUY when C > AVERAGE(C,LEN) AND EntriesToday=0 → next bar STOP at Close[1]+2×ATR(20)",
            "exits":        "SetStopLoss(STP) + SetProfitTarget(LMT) — fixed USD; max 1 entry/day",
            "r1_champion":  "LEN=7 STP=550 LMT=13500 NP=500015 MDD=-21485 trades=657 (gap -37.5%)",
            "r1_findings":  "Two peaks ~$492-500K: Peak1(STP=550,LMT=13.5K,R/R=24.5:1,657tr) Peak2(STP=250,LMT=22.5K,R/R=90:1,590tr); LEN=7 unique; STP=550 bell-curve; LMT=25K-200K+STP<600 unexplored",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_hunter_hourly2.json"
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
    log.info("  SFJ_HUNTER_NQ on CME.NQ HOT Hourly — Round 2")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  R1 seed: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    log.info("  R1 findings: Peak1(STP=550,LMT=13.5K)=$500K; Peak2(STP=250,LMT=22.5K)=$492K")
    log.info("  Focus: Peak2 zoom + LMT=25K-600K bridge + global confirm")
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
    # A01  peak2_zoom — fine-tune around Peak 2 (R/R=90:1 regime)
    #      LEN(4-10 s1) × STP(100-600 s50) × LMT(18K-30K s1K) = 7×11×13=1001
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_peak2_zoom"
    _c = _cfg(_n, (4, 10, 1), (100, 600, 50), (18000, 30000, 1000))
    log.info("A01  LEN(4-10 s1)×STP(100-600 s50)×LMT(18K-30K s1K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A02  lmt_bridge — unexplored LMT=25K-275K with low STP (key gap from R1)
    #      LEN(4-10 s1) × STP(100-700 s100) × LMT(25K-275K s25K) = 7×7×11=539
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_lmt_bridge"
    _c = _cfg(_n, (4, 10, 1), (100, 700, 100), (25000, 275000, 25000))
    log.info("A02  LEN(4-10 s1)×STP(100-700 s100)×LMT(25K-275K s25K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ultra_lmt — very large LMT ($100K-$600K), test extreme low-frequency
    #      LEN(4-10 s1) × STP(200-1200 s200) × LMT(100K-600K s50K) = 7×6×11=462
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ultra_lmt"
    _c = _cfg(_n, (4, 10, 1), (200, 1200, 200), (100000, 600000, 50000))
    log.info("A03  LEN(4-10 s1)×STP(200-1200 s200)×LMT(100K-600K s50K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A04  fine_peak1 — ultra-fine grid around Peak 1 champion
    #      LEN(5-9 s1) × STP(350-800 s50) × LMT(12K-15.5K s250) = 5×10×15=750
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_fine_peak1"
    _c = _cfg(_n, (5, 9, 1), (350, 800, 50), (12000, 15500, 250))
    log.info("A04  LEN(5-9 s1)×STP(350-800 s50)×LMT(12K-15.5K s250)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A05  len_scan — confirm LEN=7 is optimal across wide LEN range
    #      LEN(1-200 s10) × STP(400-700 s100) × LMT(12K-15K s1K) = 21×4×4=336
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_len_scan"
    _c = _cfg(_n, (1, 200, 10), (400, 700, 100), (12000, 15000, 1000))
    log.info("A05  LEN(1-200 s10)×STP(400-700 s100)×LMT(12K-15K s1K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A06  global_confirm — broad sweep to rule out undiscovered regimes
    #      LEN(1-200 s20) × STP(100-10100 s1K) × LMT(5K-105K s10K) = 11×11×11=1331
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_global_confirm"
    _c = _cfg(_n, (1, 200, 20), (100, 10100, 1000), (5000, 105000, 10000))
    log.info("A06  LEN(1-200 s20)×STP(100-10.1K s1K)×LMT(5K-105K s10K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A07  mid_lmt_loSTP — medium LMT ($20K-$100K) with very low STP (bridge)
    #      LEN(4-10 s1) × STP(200-600 s100) × LMT(20K-100K s10K) = 7×5×9=315
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_mid_lmt_loSTP"
    _c = _cfg(_n, (4, 10, 1), (200, 600, 100), (20000, 100000, 10000))
    log.info("A07  LEN(4-10 s1)×STP(200-600 s100)×LMT(20K-100K s10K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A08  stp_fine_loLMT — fine STP sweep around Peak 1, wider LMT
    #      LEN(5-9 s1) × STP(50-1550 s100) × LMT(12K-16K s500) = 5×16×9=720
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_stp_fine"
    _c = _cfg(_n, (5, 9, 1), (50, 1550, 100), (12000, 16000, 500))
    log.info("A08  LEN(5-9 s1)×STP(50-1550 s100)×LMT(12K-16K s500)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — wide zoom on overall best champion
    #      zoom_fixed() with n_target=15 → ≤15^3=3375 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 8.0
        r_stp = max(1000.0, best_STP * 0.42)
        r_lmt = max(5000.0, best_LMT * 0.42)
        _len = zoom_fixed(best_LEN, r_len,  15,    1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp,  15,   50.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt,  15,  200.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A09  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — medium zoom
    #      zoom_fixed() with n_target=11 → ≤11^3=1331 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 4.0
        r_stp = max(300.0, best_STP * 0.17)
        r_lmt = max(1000.0, best_LMT * 0.10)
        _len = zoom_fixed(best_LEN, r_len, 11,   1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp, 11,  50.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt, 11, 100.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A10  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — fine zoom
    #      zoom_fixed() with n_target=9 → ≤9^3=729 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 2.0
        r_stp = max(100.0, best_STP * 0.09)
        r_lmt = max(500.0, best_LMT * 0.05)
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
    log.info("  SFJ_HUNTER_NQ NQ Hourly Round-2 COMPLETE")
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
        description="SFJ_HUNTER_NQ CME.NQ HOT Hourly R2 parameter search")
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
