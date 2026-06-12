"""
search_gc_hunter_hourly2.py — SFJ_HUNTER_NQ on CME.GC HOT Hourly, Round 2

R1 Champion: LEN=156, STP=9155, LMT=7000 → NP=$376,180 MDD=-$44,700 trades=126 (gap -53.0%)

R1 Key findings:
  - COMPLETELY DIFFERENT regime from NQ: GC uses WIDE STP + MODERATE LMT (R/R≈0.76:1)
    Win rate ~75.2%, avg $2,986/trade. NQ used STP=$550 LMT=$13,500 (R/R=24.5:1, win rate 9.3%)
  - TWO competitive regimes found in R1:
      Regime 2 (champion): LEN=145-165, STP=8500-9500, LMT=7000, NP≈$376K (18 trades/yr)
      Regime 1: LEN=85-105, STP=4500, LMT=16000, NP≈$366K (100 trades/7yr)
  - LEN inert in range 156-164: all give same NP at same STP/LMT (confirmed by A10)
  - STP=9155 is between A01 grid points (8500 and 9500); zoom found the true peak
  - LMT landscape at STP=8500 has TWO humps:
      Primary: LMT=7000 → $371K (129 trades)
      Valley: LMT=9000-17000 (dips to $244-298K)
      Secondary hump: LMT=19000-21000 → $320-337K (40-56 trades)
      UNEXPLORED: LMT=22000-100000 with STP=8500-10500 (A08 only tested STP=2K-22K)
  - STP=10500-30000 territory not covered by A01 (stopped at 10500)
  - NQ-style tight STP (A04) terrible on GC: max $80K; GC needs wide STP

R2 Attempt schedule:
  A01 fine_peak        : LEN(150-170 s2)×STP(7500-11500 s300)×LMT(5500-9000 s250)   = 11×14×15=2310
  A02 lmt_secondary    : LEN(148-168 s5)×STP(7000-11000 s1K)×LMT(22K-82K s10K)     = 5×5×7=175
  A03 stp_extend       : LEN(148-168 s5)×STP(10500-25500 s2500)×LMT(5000-9000 s1K)  = 5×7×5=175
  A04 regime1_fine     : LEN(75-115 s5)×STP(3500-6000 s500)×LMT(12K-21K s1K)        = 9×6×10=540
  A05 regime3_short    : LEN(32-62 s5)×STP(1500-4000 s500)×LMT(14K-22K s1K)         = 7×6×9=378
  A06 global_confirm   : LEN(10-310 s30)×STP(6000-12000 s1500)×LMT(5000-9000 s1K)   = 11×5×5=275
  A07 len_scan         : LEN(2-200 s10)×STP(8500-10000 s500)×LMT(6000-8000 s500)     = 20×4×5=400
  A08 med_STP_highLMT  : LEN(80-120 s5)×STP(4000-6000 s500)×LMT(20K-80K s10K)       = 9×5×7=315
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
SYMBOL     = "CME.GC HOT"
SIGNAL     = "SFJ_HUNTER_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_hunter_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0   # USD

LEN_LO, LEN_HI = 1.0,      2000.0
STP_LO, STP_HI = 50.0,   100_000.0
LMT_LO, LMT_HI = 100.0, 1_000_000.0

# R1 champion as seed
SEED_LEN = 156.0
SEED_STP = 9_155.0
SEED_LMT = 7_000.0
SEED_NP  = 376_180.0

PREFIX = "GCHH2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_hunter_hourly2_{int(time.time())}.log"
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
            "logic":       "BUY when C > AVERAGE(C,LEN) AND EntriesToday=0 → next bar STOP at Close[1]+2×ATR(20)",
            "exits":       "SetStopLoss(STP) + SetProfitTarget(LMT) — fixed USD amounts; max 1 entry/day",
            "r1_champion": "LEN=156 STP=9155 LMT=7000 NP=376180 MDD=-44700 trades=126 (gap -53.0%)",
            "r1_regime":   "Wide STP + moderate LMT (R/R=0.76:1, win rate ~75%, 18 trades/yr). LEN inert 156-164.",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_gc_hunter_hourly2.json"
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
    log.info("  SFJ_HUNTER_NQ on CME.GC HOT Hourly — Round 2")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  R1 champion: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             SEED_LEN, SEED_STP, SEED_LMT, SEED_NP)
    log.info("  Target: %.0f USD  (gap %.0f)", TARGET_NP, TARGET_NP - SEED_NP)
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
    # A01  fine_peak — fine grid around R1 champion
    #      LEN(150-170 s2) × STP(7500-11500 s300) × LMT(5500-9000 s250) = 11×14×15=2310
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_peak"
    _c = _cfg(_n, (150, 170, 2), (7500, 11400, 300), (5500, 9000, 250))
    log.info("A01  LEN(150-170 s2)×STP(7500-11.4K s300)×LMT(5500-9K s250)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A02  lmt_secondary — explore secondary LMT hump at high STP (LMT=22K-82K)
    #      A01's LMT data shows secondary hump at LMT=19-21K; test above 21K with champion STP
    #      LEN(148-168 s5) × STP(7000-11000 s1K) × LMT(22K-82K s10K) = 5×5×7=175
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_lmt_secondary"
    _c = _cfg(_n, (148, 168, 5), (7000, 11000, 1000), (22000, 82000, 10000))
    log.info("A02  LEN(148-168 s5)×STP(7K-11K s1K)×LMT(22K-82K s10K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A03  stp_extend — test STP=10500-25500 territory (above A01's max 10500)
    #      LEN(148-168 s5) × STP(10500-25500 s2500) × LMT(5000-9000 s1K) = 5×7×5=175
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_stp_extend"
    _c = _cfg(_n, (148, 168, 5), (10500, 25500, 2500), (5000, 9000, 1000))
    log.info("A03  LEN(148-168 s5)×STP(10.5K-25.5K s2.5K)×LMT(5K-9K s1K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A04  regime1_fine — fine-tune Regime 1 (LEN=85-105, STP=4500, LMT=16K)
    #      LEN(75-115 s5) × STP(3500-6000 s500) × LMT(12K-21K s1K) = 9×6×10=540
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_regime1_fine"
    _c = _cfg(_n, (75, 115, 5), (3500, 6000, 500), (12000, 21000, 1000))
    log.info("A04  LEN(75-115 s5)×STP(3.5K-6K s500)×LMT(12K-21K s1K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A05  regime3_short — fine-tune Regime 3 (LEN=40-52, STP=2500, LMT=17K)
    #      LEN(32-62 s5) × STP(1500-4000 s500) × LMT(14K-22K s1K) = 7×6×9=378
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_regime3_short"
    _c = _cfg(_n, (32, 62, 5), (1500, 4000, 500), (14000, 22000, 1000))
    log.info("A05  LEN(32-62 s5)×STP(1.5K-4K s500)×LMT(14K-22K s1K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A06  global_confirm — verify no undiscovered region at high STP + all LEN
    #      LEN(10-310 s30) × STP(6000-12000 s1500) × LMT(5000-9000 s1K) = 11×5×5=275
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_global_confirm"
    _c = _cfg(_n, (10, 310, 30), (6000, 12000, 1500), (5000, 9000, 1000))
    log.info("A06  LEN(10-310 s30)×STP(6K-12K s1.5K)×LMT(5K-9K s1K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A07  len_scan — LEN landscape at champion STP/LMT with fine LEN step
    #      LEN(2-200 s10) × STP(8500-10000 s500) × LMT(6000-8000 s500) = 20×4×5=400
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_len_scan"
    _c = _cfg(_n, (2, 200, 10), (8500, 10000, 500), (6000, 8000, 500))
    log.info("A07  LEN(2-200 s10)×STP(8.5K-10K s500)×LMT(6K-8K s500)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A08  med_STP_highLMT — medium STP + high LMT with LEN=80-120
    #      Explores regime 1-type STP with high-LMT secondary hump territory
    #      LEN(80-120 s5) × STP(4000-6000 s500) × LMT(20K-80K s10K) = 9×5×7=315
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_med_STP_highLMT"
    _c = _cfg(_n, (80, 120, 5), (4000, 6000, 500), (20000, 80000, 10000))
    log.info("A08  LEN(80-120 s5)×STP(4K-6K s500)×LMT(20K-80K s10K)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — wide zoom on best NP found across A01-A08
    #      zoom_fixed() with n_target=15 per dim → ≤15^3=3375 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 15.0
        r_stp = max(2000.0, best_STP * 0.42)
        r_lmt = max(5000.0, best_LMT * 0.42)
        _len = zoom_fixed(best_LEN, r_len,  15,    1.0, LEN_LO, LEN_HI)
        _stp = zoom_fixed(best_STP, r_stp,  15,   50.0, STP_LO, STP_HI)
        _lmt = zoom_fixed(best_LMT, r_lmt,  15,  250.0, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _len, _stp, _lmt)
        log.info("A09  LEN%s STP%s LMT%s  %d combos", _len, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_LEN, best_STP, best_LMT)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — medium zoom on best
    #      zoom_fixed() with n_target=11 per dim → ≤11^3=1331 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 8.0
        r_stp = max(500.0, best_STP * 0.17)
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
    # A11  adaptive_zoom3 — fine zoom on best
    #      zoom_fixed() with n_target=9 per dim → ≤9^3=729 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_LEN, best_STP, best_LMT, best_np)
    if start_attempt <= A:
        r_len = 4.0
        r_stp = max(200.0, best_STP * 0.09)
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
    gain_pct = (best_np - SEED_NP) / SEED_NP * 100 if SEED_NP else 0
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER_NQ GC Hourly Round-2 COMPLETE")
    log.info("  Champion: LEN=%.4g STP=%.4g LMT=%.4g", best_LEN, best_STP, best_LMT)
    log.info("  Best NP: %.0f USD  (R1 gain +%.2f%%  target gap %.0f)",
             best_np, gain_pct, max(0, TARGET_NP - best_np))
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
    print(f"R1 NP={SEED_NP:.0f}  R2 NP={best_np:.0f}  gain={gain_pct:+.2f}%")
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
        description="SFJ_HUNTER_NQ CME.GC HOT Hourly R2 parameter search")
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
