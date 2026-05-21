"""
search_cl_cl_daily4.py — _2021Basic_Break_CL on CME.CL HOT Daily, Round 4

R3 summary (12 attempts, 2026-05-21):
  Best NP:  $101,490  LE=69 SE=34 STP=4 LMT=6  NEW regime discovered (gap -85%)
  Best Obj: 484,944   LE=69 SE=34 STP=4 LMT=6  (MDD only -$21,240 — far better risk-adj)
  Old best: $90,950   LE=1  SE=1  STP=4 LMT=5  (3 rounds confirmed)

R3 key findings:
  - LE=69 + SE=34 is a genuinely distinct regime from LE=1+SE=1
  - High LE + mid SE → lower trade count (~12) but better MDD (-21K vs -63K)
  - SE>70 with LE=1 = worse; very high STP/LMT = worse; micro-STP with LE=1 = worse
  - sub-$1 LMT UI-failed (MC64 cannot handle LMT<1 on CL)

Round-4 explores around the new LE=69 regime:
  1. Fine LE/SE 2D sweep in new regime core (LE=55-85, SE=20-50)
  2. Fine STP/LMT in new regime (LE=67-71, SE=32-36)
  3. Wide LE sweep (LE=40-130) at optimal SE~34
  4. Very high LE (LE=130-200)
  5. Wide STP range (STP=0.5-20) in new regime
  6. Wide LMT range (LMT=1-25) in new regime
  7. Broad 2D LE/SE landscape (LE=30-110, SE=10-60)
  8. Medium-fine new regime (LE=50-90, SE=25-45)
  9. Ultra-fine zoom around R3 best (LE=67-71, SE=32-36, STP step=0.2, LMT step=0.2)
  10. Adaptive zoom from R4 running best
  11. Micro STP in new regime (STP=0.1-2.8)
  12. Global boundary R4 (LE/SE/STP/LMT full boundary)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 fine_le_se         : LE(55-85 s5)×SE(20-50 s5)×STP(3-5 s1)×LMT(5-7 s1)×LenLE(95-105 s5)  = 1323
  A02 fine_stp_lmt       : LE(67-71 s2)×SE(32-36 s2)×STP(2-7 s0.5)×LMT(4-9 s0.5)×LenLE(95-105) = 3267
  A03 wide_le            : LE(40-130 s10)×SE(30-40 s5)×STP(3-5 s1)×LMT(5-7 s1)×LenLE(95-105)   = 810
  A04 very_high_le       : LE(130-200 s10)×SE(20-60 s10)×STP(3-5 s1)×LMT(4-8 s2)×LenLE(95-105) = 1080
  A05 wide_stp           : LE(65-75 s5)×SE(30-40 s5)×STP(0.5-19.5 s2)×LMT(5-7 s1)×LenLE(95-105)= 810
  A06 wide_lmt           : LE(65-75 s5)×SE(30-40 s5)×STP(3-5 s1)×LMT(1-25 s2)×LenLE(95-105)    = 1053
  A07 le_se_wide_2d      : LE(30-110 s5)×SE(10-60 s5)×STP(4-6 s2)×LMT(5-7 s2)×LenLE(95-105)   = 2244
  A08 new_regime_v2      : LE(50-90 s5)×SE(25-45 s5)×STP(3-5 s1)×LMT(5-7 s1)×LenLE(95-105)    = 1215
  A09 ultra_zoom_new     : LE(67-71 s1)×SE(32-36 s1)×STP(3.5-4.5 s0.2)×LMT(5-7 s0.2)×LenLE(95-105) = 4950
  A10 adaptive_zoom      : (dynamic from R4 best NP)
  A11 micro_stp_new      : LE(65-75 s5)×SE(30-40 s5)×STP(0.1-2.8 s0.3)×LMT(3-10 s1)×LenLE(95-105)= 2160
  A12 global_boundary4   : LE(1-200 s20)×SE(1-500 s50)×STP(2-8 s2)×LMT(4-8 s2)×LenLE(95-105)  = 4356
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
SYMBOL     = "CME.CL HOT"
SIGNAL     = "_2021Basic_Break_CL"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\cl_cl_daily4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0

STP_LO,   STP_HI   = 0.01,  500.0
LMT_LO,   LMT_HI   = 1.0,   500.0   # sub-$1 LMT UI-fails on CL — floor at 1
SE_LO,    SE_HI    = 1.0,   500.0
LE_LO,    LE_HI    = 1.0,   500.0
LENLE_LO, LENLE_HI = 2.0,   500.0

# R3 best as seeds (new regime discovered in A11)
SEED_LE, SEED_SE    = 69.0,   34.0
SEED_STP, SEED_LMT  = 4.0,    6.0
SEED_LENLE           = 100.0
SEED_NP              = -1_000_000.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_cl_cl_daily4_{int(time.time())}.log"
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
         le:    Tuple[float, float, float],
         se:    Tuple[float, float, float],
         stp:   Tuple[float, float, float],
         lmt:   Tuple[float, float, float],
         lenle: Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    le    = _safe(le,    LE_LO,    LE_HI)
    se    = _safe(se,    SE_LO,    SE_HI)
    stp   = _safe(stp,   STP_LO,   STP_HI)
    lmt   = _safe(lmt,   LMT_LO,   LMT_HI)
    lenle = _safe(lenle, LENLE_LO, LENLE_HI)

    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt) * n_vals(lenle)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"CLCL4_{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
            ParamAxis("LE",    *le),
            ParamAxis("SE",    *se),
            ParamAxis("STP",   *stp),
            ParamAxis("LMT",   *lmt),
            ParamAxis("LenLE", *lenle),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"CLCL4_{name}_raw.csv"


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
    log.info("=== Starting CLCL4_%s (%d combos) ===", name, cfg.total_runs())
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


def champion(df, fb_le, fb_se, fb_stp, fb_lmt, fb_lenle):
    """Priority: target met → highest NP (target-chasing mode)."""
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
                 float(best["Objective"]), int(best["TotalTrades"]))
        le    = float(best["LE"])
        se    = float(best["SE"])
        stp   = float(best["STP"])
        lmt   = float(best["LMT"])
        lenle = float(best.get("LenLE", fb_lenle))
        return le, se, stp, lmt, lenle, float(best["Objective"]), float(best["NetProfit"]), float(best["MaxDrawdown"]), int(best["TotalTrades"]), True

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        # Target-chasing: zoom toward NP-max, not Obj-max
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        le    = float(best["LE"])
        se    = float(best["SE"])
        stp   = float(best["STP"])
        lmt   = float(best["LMT"])
        lenle = float(best.get("LenLE", fb_lenle))
        return le, se, stp, lmt, lenle, float(best["Objective"]), float(best["NetProfit"]), float(best["MaxDrawdown"]), int(best["TotalTrades"]), False

    best = df.loc[df["NetProfit"].idxmax()]
    lenle = float(best.get("LenLE", fb_lenle))
    return fb_le, fb_se, fb_stp, fb_lmt, fb_lenle, 0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]), int(best["TotalTrades"]), False


def _entry(attempt, name, df, le, se, stp, lmt, lenle, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt,
        "name": name,
        "LE": le, "SE": se, "STP": stp, "LMT": lmt, "LenLE": lenle,
        "objective": obj,
        "net_profit": np_,
        "max_drawdown": mdd,
        "total_trades": trades,
        "target_met": met,
        "combos": combos,
        "rows": len(df) if df is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }


def save_json(best_entry, attempt_log, target_met):
    r3_best = {
        "LE": 69, "SE": 34, "STP": 4.0, "LMT": 6.0,
        "net_profit": 101490, "max_drawdown": -21240,
        "objective": 484944, "total_trades": 12,
        "note": "R3 A11 new regime — gap -85.5%",
    }
    payload = {
        "round": 4,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "long_entry":  "BUY HIGHEST(H,LE) when MA(LenLE)>MA(LenLE*2)",
            "short_entry": "SELL LOWEST(L,SE) when MA(LenLE)<=MA(LenLE*2)",
            "exits":       "STP×ATR stop + LMT×ATR limit",
            "lenle_note":  "LenLE=100 fixed (MC64 does not vary via automation)",
            "r3_best":     r3_best,
            "r4_focus":    "New LE=69 regime: fine STP/LMT, wide LE, very high LE, SE variations",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_cl_cl_daily4.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,   best_se    = SEED_LE,    SEED_SE
    best_stp,  best_lmt   = SEED_STP,   SEED_LMT
    best_lenle             = SEED_LENLE
    best_np   = SEED_NP
    best_obj  = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  CL Daily _2021Basic_Break_CL NP>700K — Round 4")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R3 best: LE=69 SE=34 STP=4 LMT=6  NP=$101,490 (gap -$598,510)")
    log.info("  R4 focus: deep-dive LE=69 regime, STP/LMT fine, LE range 40-200")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_le, best_se, best_stp, best_lmt, best_lenle
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_le, best_se, best_stp, best_lmt, best_lenle,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        le, se, stp, lmt, lenle, obj, np_, mdd, tr, met = champion(
            df, best_le, best_se, best_stp, best_lmt, best_lenle)

        if np_ > best_np:
            best_le, best_se  = le, se
            best_stp, best_lmt = stp, lmt
            best_lenle = lenle
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, le, se, stp, lmt, lenle,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  fine_le_se — map LE/SE 2D landscape of new regime
    #      LE(55-85 s5) × SE(20-50 s5) × STP(3-5 s1) × LMT(5-7 s1) × LenLE(95-105 s5)
    #      = 7×7×3×3×3 = 1323
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_le_se"
    _c = _cfg(_n, (55, 85, 5), (20, 50, 5), (3.0, 5.0, 1.0), (5.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A01  LE(55-85 s5)×SE(20-50 s5)×STP(3-5)×LMT(5-7)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  fine_stp_lmt — full STP/LMT grid in R3-best core
    #      LE(67-71 s2) × SE(32-36 s2) × STP(2-7 s0.5) × LMT(4-9 s0.5) × LenLE(95-105 s5)
    #      = 3×3×11×11×3 = 3267
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_fine_stp_lmt"
    _c = _cfg(_n, (67, 71, 2), (32, 36, 2), (2.0, 7.0, 0.5), (4.0, 9.0, 0.5), (95.0, 105.0, 5.0))
    log.info("A02  LE(67-71 s2)×SE(32-36 s2)×STP(2-7 s0.5)×LMT(4-9 s0.5)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  wide_le — sweep LE=40-130 at optimal SE~34
    #      LE(40-130 s10) × SE(30-40 s5) × STP(3-5 s1) × LMT(5-7 s1) × LenLE(95-105 s5)
    #      = 10×3×3×3×3 = 810
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_wide_le"
    _c = _cfg(_n, (40, 130, 10), (30, 40, 5), (3.0, 5.0, 1.0), (5.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A03  LE(40-130 s10)×SE(30-40 s5)×STP(3-5)×LMT(5-7)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  very_high_le — LE=130-200, never explored
    #      LE(130-200 s10) × SE(20-60 s10) × STP(3-5 s1) × LMT(4-8 s2) × LenLE(95-105 s5)
    #      = 8×5×3×3×3 = 1080
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_very_high_le"
    _c = _cfg(_n, (130, 200, 10), (20, 60, 10), (3.0, 5.0, 1.0), (4.0, 8.0, 2.0), (95.0, 105.0, 5.0))
    log.info("A04  LE(130-200 s10)×SE(20-60 s10)×STP(3-5)×LMT(4-8 s2)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  wide_stp — STP=0.5-19.5 sweep in new regime
    #      LE(65-75 s5) × SE(30-40 s5) × STP(0.5-19.5 s2) × LMT(5-7 s1) × LenLE(95-105 s5)
    #      = 3×3×10×3×3 = 810
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_wide_stp"
    _c = _cfg(_n, (65, 75, 5), (30, 40, 5), (0.5, 19.5, 2.0), (5.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A05  LE(65-75 s5)×SE(30-40 s5)×STP(0.5-19.5 s2)×LMT(5-7)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  wide_lmt — LMT=1-25 sweep in new regime
    #      LE(65-75 s5) × SE(30-40 s5) × STP(3-5 s1) × LMT(1-25 s2) × LenLE(95-105 s5)
    #      = 3×3×3×13×3 = 1053
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_wide_lmt"
    _c = _cfg(_n, (65, 75, 5), (30, 40, 5), (3.0, 5.0, 1.0), (1.0, 25.0, 2.0), (95.0, 105.0, 5.0))
    log.info("A06  LE(65-75 s5)×SE(30-40 s5)×STP(3-5)×LMT(1-25 s2)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  le_se_wide_2d — broad LE/SE landscape at fixed STP/LMT
    #      LE(30-110 s5) × SE(10-60 s5) × STP(4-6 s2) × LMT(5-7 s2) × LenLE(95-105 s5)
    #      = 17×11×2×2×3 = 2244
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_le_se_wide_2d"
    _c = _cfg(_n, (30, 110, 5), (10, 60, 5), (4.0, 6.0, 2.0), (5.0, 7.0, 2.0), (95.0, 105.0, 5.0))
    log.info("A07  LE(30-110 s5)×SE(10-60 s5)×STP(4-6 s2)×LMT(5-7 s2)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  new_regime_v2 — medium-fine LE/SE in new regime zone
    #      LE(50-90 s5) × SE(25-45 s5) × STP(3-5 s1) × LMT(5-7 s1) × LenLE(95-105 s5)
    #      = 9×5×3×3×3 = 1215
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_new_regime_v2"
    _c = _cfg(_n, (50, 90, 5), (25, 45, 5), (3.0, 5.0, 1.0), (5.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A08  LE(50-90 s5)×SE(25-45 s5)×STP(3-5)×LMT(5-7)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  ultra_zoom_new — ultra-fine around R3 best
    #      LE(67-71 s1) × SE(32-36 s1) × STP(3.5-4.5 s0.2) × LMT(5-7 s0.2) × LenLE(95-105 s5)
    #      = 5×5×6×11×3 = 4950
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_ultra_zoom_new"
    _c = _cfg(_n, (67, 71, 1), (32, 36, 1), (3.5, 4.5, 0.2), (5.0, 7.0, 0.2), (95.0, 105.0, 5.0))
    log.info("A09  LE(67-71 s1)×SE(32-36 s1)×STP(3.5-4.5 s0.2)×LMT(5-7 s0.2)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — zoom around best NP found in A01-A09
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_le, r_se, r_stp, r_lmt in [
            (5, 8, 1.0, 3.0),
            (3, 5, 0.8, 2.0),
            (2, 4, 0.6, 1.5),
            (1, 2, 0.4, 1.0),
        ]:
            _le  = zoom(best_le,  r_le,  1.0,  LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  1.0,  SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.2,  STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
            _c = _cfg(_n, _le, _se, _stp, _lmt, (95.0, 105.0, 5.0))
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LE%s SE%s STP%s LMT%s  %d combos", _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  micro_stp_new — STP=0.1-2.8 in new regime, with wider LMT
    #      LE(65-75 s5) × SE(30-40 s5) × STP(0.1-2.8 s0.3) × LMT(3-10 s1) × LenLE(95-105 s5)
    #      = 3×3×10×8×3 = 2160
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_micro_stp_new"
    _c = _cfg(_n, (65, 75, 5), (30, 40, 5), (0.1, 2.8, 0.3), (3.0, 10.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A11  LE(65-75 s5)×SE(30-40 s5)×STP(0.1-2.8 s0.3)×LMT(3-10)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A12  global_boundary4 — full boundary R4 (LE up to 200, SE up to 500)
    #      LE(1-200 s20) × SE(1-500 s50) × STP(2-8 s2) × LMT(4-8 s2) × LenLE(95-105 s5)
    #      = 11×11×4×3×3 = 4356
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_global_boundary4"
    _c = _cfg(_n, (1, 200, 20), (1, 500, 50), (2.0, 8.0, 2.0), (4.0, 8.0, 2.0), (95.0, 105.0, 5.0))
    log.info("A12  LE(1-200 s20)×SE(1-500 s50)×STP(2-8 s2)×LMT(4-8 s2)×LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  CL Daily _2021Basic_Break_CL Round-4 COMPLETE")
    log.info("  Best NP: %.0f  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)
    log.info("  Target %.0f: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{TARGET_NP - best_np:.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LE": best_le, "SE": best_se, "STP": best_stp, "LMT": best_lmt,
            "LenLE": best_lenle, "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
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
    ap = argparse.ArgumentParser(
        description="CL Daily _2021Basic_Break_CL NP>700K Round-4 search")
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
