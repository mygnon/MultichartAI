"""
search_zw_cl_daily2.py — _2021Basic_Break_CL (UPDATED) on CBOT.ZW HOT Daily, Round 1 (v2)

STRATEGY UPDATE (2026-05-20):
  Previous version: long-only (SE was declared but unused — dead param)
  UPDATED version:  long/short dual-direction

Updated strategy logic:
  INPUT: LE(3), SE(9), STP(1.5), LMT(6), LenLE(100)
  ATR = AvgTrueRange(10)
  IF AVERAGE(CLOSE, LenLE) > AVERAGE(CLOSE, LenLE*2) THEN
    IF marketposition <> 1  THEN BUY NEXT BAR HIGHEST(H,LE) STOP       {uptrend → LONG}
  ELSE
    IF marketposition <> -1 THEN SELLSHORT NEXT BAR LOWEST(L,SE) STOP  {downtrend → SHORT}
  IF marketposition=1  THEN SELL("LX-STP")      NEXT BAR ENTRYPRICE - STP*ATR STOP
  IF marketposition=-1 THEN buytocover("SX-STP") NEXT BAR ENTRYPRICE + STP*ATR STOP
  IF marketposition=1  THEN SELL("LX-LMT")      NEXT BAR ENTRYPRICE + LMT*ATR LIMIT
  IF marketposition=-1 THEN buytocover("SX-LMT") NEXT BAR ENTRYPRICE - LMT*ATR LIMIT

Key insight:
  - Uptrend   (MA(LenLE) > MA(LenLE×2)): BUY  breakout above LE-day high
  - Downtrend (ELSE):                     SHORT breakdown below SE-day low
  - SE is NOW ACTIVE — controls short entry lookback (NOT a dead param!)
  - STP/LMT are ATR multiples (symmetric for both directions)
  - ZW daily ATR ≈ 15-25¢ ($750-$1,250/contract)
  - R/R at default = LMT/STP = 6/1.5 = 4:1

All 5 parameters are meaningful:
  LE    — Long-entry breakout lookback (1-30+ days)
  SE    — Short-entry breakdown lookback (1-30+ days)
  STP   — Stop loss in ATR multiples (0.25-10×)
  LMT   — Profit target in ATR multiples (1-30×)
  LenLE — Trend-filter MA period (10-400 days)

R1_old partial results (old strategy, suspect):
  Best: LE=7 SE=8 STP=3.0 LMT=5.0 LenLE=100  NP=25,512  trades=10
  Note: R1 used SE=8-10 as "dead range" — those results may not transfer

WORKSPACE REQUIREMENT:
  Confirm "CBOT.ZW HOT - Daily" chart in MC64 has _2021Basic_Break_CL applied
  (updated version with ELSE/SELLSHORT clause). Save workspace before running.

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 symmetric_broad  : LE(1-8 s1) × SE(1-8 s1) × STP(1-3 s1) × LMT(4-10 s2)  × LenLE(60-180 s30) 8×8×3×4×5=3840
  A02 fine_lese        : LE(1-5 s1) × SE(1-5 s1) × STP(1-3 s0.5)× LMT(3-7 s1)  × LenLE(60-150 s30) 5×5×5×5×4=2500
  A03 high_lmt         : LE(1-5 s1) × SE(1-5 s1) × STP(1-3 s1) × LMT(8-20 s2)  × LenLE(60-150 s30) 5×5×3×7×4=2100
  A04 tight_stp        : LE(1-8 s1) × SE(1-8 s1) × STP(0.25-1 s0.25)× LMT(2-8 s2)× LenLE(60-150 s30) 8×8×4×4×4=4096
  A05 wide_stp         : LE(1-5 s1) × SE(1-5 s1) × STP(3-10 s1) × LMT(3-10 s2) × LenLE(60-150 s30) 5×5×8×4×4=3200
  A06 short_lenle      : LE(1-6 s1) × SE(1-6 s1) × STP(1-3 s1) × LMT(3-9 s2)   × LenLE(10-60 s10)  6×6×3×4×6=2592
  A07 long_lenle       : LE(1-6 s1) × SE(1-6 s1) × STP(1-3 s1) × LMT(3-9 s2)   × LenLE(120-360 s60) 6×6×3×4×5=2160
  A08 high_le          : LE(10-30 s5)× SE(1-10 s1)× STP(1-3 s1) × LMT(4-10 s2) × LenLE(60-150 s30) 5×10×3×4×4=2400
  A09 high_se          : LE(1-10 s1) × SE(10-30 s5)× STP(1-3 s1)× LMT(4-10 s2) × LenLE(60-150 s30) 10×5×3×4×4=2400
  A10 very_high_lmt    : LE(1-5 s1) × SE(1-5 s1) × STP(1-4 s1) × LMT(20-50 s10)× LenLE(60-150 s30) 5×5×4×4×4=1600
  A11 adaptive_zoom      (from best NP found)
  A12 wide_boundary    : LE(1-25 s6) × SE(1-25 s6)× STP(1-5 s2) × LMT(2-20 s6) × LenLE(50-300 s50) 5×5×3×4×6=1800
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
SIGNAL     = "_2021Basic_Break_CL"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\zw_cl_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0

STP_LO,   STP_HI   = 0.1,   50.0
LMT_LO,   LMT_HI   = 0.5,  200.0
SE_LO,    SE_HI    = 1.0,  200.0
LE_LO,    LE_HI    = 1.0,  100.0
LENLE_LO, LENLE_HI = 2.0,  500.0

# No reliable prior data for new strategy — neutral seed
SEED_LE, SEED_SE    = 3.0,   9.0
SEED_STP, SEED_LMT  = 1.5,   6.0
SEED_LENLE           = 100.0
SEED_NP              = -1_000_000.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_zw_cl_daily2_{int(time.time())}.log"
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
        name=f"ZWCL2_{name}",
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
    return OUTPUT_DIR / f"ZWCL2_{name}_raw.csv"


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
    log.info("=== Starting ZWCL2_%s (%d combos) ===", name, cfg.total_runs())
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
    """Priority: target met → positive NP → least-negative NP."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]), float(best.get("LenLE", fb_lenle)),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best.get("TotalTrades", 0)))
        return (float(best["LE"]), float(best["SE"]),
                float(best["STP"]), float(best["LMT"]), float(best.get("LenLE", fb_lenle)),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best.get("TotalTrades", 0)), True)

    pos = df[df["Objective"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        le    = float(best["LE"]);    se    = float(best["SE"])
        stp   = float(best["STP"]);   lmt   = float(best["LMT"])
        lenle = float(best.get("LenLE", fb_lenle))
        np_   = float(best.get("NetProfit", 0))
        mdd   = float(best.get("MaxDrawdown", 0))
        tr    = int(best.get("TotalTrades", 0))
        obj   = float(best["Objective"])
        log.info("  NP-Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 le, se, stp, lmt, lenle, obj, np_, mdd, tr)
        return le, se, stp, lmt, lenle, obj, np_, mdd, tr, False

    # All NP negative — track least-negative for zoom direction
    np_col = pd.to_numeric(df.get("NetProfit", pd.Series(dtype=float)), errors="coerce")
    if not np_col.isna().all():
        best  = df.loc[np_col.idxmax()]
        le    = float(best["LE"]);    se    = float(best["SE"])
        stp   = float(best["STP"]);   lmt   = float(best["LMT"])
        lenle = float(best.get("LenLE", fb_lenle))
        np_   = float(best.get("NetProfit", 0))
        mdd   = float(best.get("MaxDrawdown", 0))
        tr    = int(best.get("TotalTrades", 0))
        log.info("  All-neg best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g  "
                 "NP=%.0f  MDD=%.0f  trades=%d",
                 le, se, stp, lmt, lenle, np_, mdd, tr)
        return le, se, stp, lmt, lenle, 0.0, np_, mdd, tr, False

    return fb_le, fb_se, fb_stp, fb_lmt, fb_lenle, 0.0, 0.0, 0.0, 0, False


def _entry(attempt, name, df, le, se, stp, lmt, lenle, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "combos": combos, "rows": len(df) if df is not None else 0,
        "LE": le, "SE": se, "STP": stp, "LMT": lmt, "LenLE": lenle,
        "net_profit": round(np_, 0), "max_drawdown": round(mdd, 0),
        "objective":  round(obj, 0), "total_trades": trades,
        "target_met": met,
    }


def save_json(best, log_, met):
    above    = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    best_np  = max(log_, key=lambda x: x.get("net_profit", -1e18), default={})
    best_obj = max(log_, key=lambda x: x.get("objective", 0), default={})
    payload = {
        "strategy":           "Breakout_ZW_CL_Daily_v2 (target NP>700K round-1-v2)",
        "symbol":             SYMBOL,
        "signal":             SIGNAL,
        "timeframe":          "Daily (1 day)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "strategy_notes": {
            "version":    "UPDATED — long/short dual-direction",
            "long_entry": "BUY HIGHEST(H,LE) when MA(LenLE)>MA(LenLE*2) [uptrend]",
            "short_entry":"SELLSHORT LOWEST(L,SE) when MA(LenLE)<=MA(LenLE*2) [downtrend]",
            "exits":      "STP*ATR stop + LMT*ATR limit (symmetric long/short)",
            "ATR_basis":  "AvgTrueRange(10) — ZW daily ATR ≈ 15-25¢ = $750-1250/contract",
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "best_obj_attempt":      best_obj,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_zw_cl_daily2.json"
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
    log.info("  ZW Daily _2021Basic_Break_CL (UPDATED) NP>700K — Round 1 v2")
    log.info("  Symbol: %s  Signal: %s (UPDATED v2)", SYMBOL, SIGNAL)
    log.info("  Uptrend:   BUY breakout above LE-day high")
    log.info("  Downtrend: SHORT breakdown below SE-day low")
    log.info("  STP/LMT in ATR multiples; ZW ATR≈20¢, LMT=6→120¢=$6000/trade")
    log.info("  SE is now ACTIVE (not dead) — full 5-param search")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_le, best_se, best_stp, best_lmt, best_lenle
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_le, best_se, best_stp, best_lmt, best_lenle,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-22s]  no valid data", attempt_num, name)
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

        log.info("  [A%02d %-22s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, le, se, stp, lmt, lenle, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f (gap +%.0f to target)",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  symmetric_broad — LE≈SE range, medium STP/LMT, LenLE survey
    #      LE(1-8 s1) × SE(1-8 s1) × STP(1-3 s1) × LMT(4-10 s2) × LenLE(60-180 s30)
    #      = 8×8×3×4×5 = 3840
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_symmetric_broad"
    _c = _cfg(_n, (1, 8, 1), (1, 8, 1), (1.0, 3.0, 1.0), (4.0, 10.0, 2.0), (60.0, 180.0, 30.0))
    log.info("A01  LE(1-8) × SE(1-8) × STP(1-3 s1) × LMT(4-10 s2) × LenLE(60-180 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A02  fine_lese — LE=1–5, SE=1–5, finer STP/LMT step
    #      LE(1-5 s1) × SE(1-5 s1) × STP(1-3 s0.5) × LMT(3-7 s1) × LenLE(60-150 s30)
    #      = 5×5×5×5×4 = 2500
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_fine_lese"
    _c = _cfg(_n, (1, 5, 1), (1, 5, 1), (1.0, 3.0, 0.5), (3.0, 7.0, 1.0), (60.0, 150.0, 30.0))
    log.info("A02  LE(1-5) × SE(1-5) × STP(1-3 s0.5) × LMT(3-7 s1) × LenLE(60-150 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A03  high_lmt — LMT=8–20×ATR (large profit targets, R/R=5-13:1 if STP=1.5)
    #      LE(1-5 s1) × SE(1-5 s1) × STP(1-3 s1) × LMT(8-20 s2) × LenLE(60-150 s30)
    #      = 5×5×3×7×4 = 2100
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_high_lmt"
    _c = _cfg(_n, (1, 5, 1), (1, 5, 1), (1.0, 3.0, 1.0), (8.0, 20.0, 2.0), (60.0, 150.0, 30.0))
    log.info("A03  LE(1-5) × SE(1-5) × STP(1-3 s1) × LMT(8-20 s2) × LenLE(60-150 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A04  tight_stp — STP=0.25–1.0×ATR (aggressive risk control)
    #      LE(1-8 s1) × SE(1-8 s1) × STP(0.25-1 s0.25) × LMT(2-8 s2) × LenLE(60-150 s30)
    #      = 8×8×4×4×4 = 4096
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_tight_stp"
    _c = _cfg(_n, (1, 8, 1), (1, 8, 1), (0.25, 1.0, 0.25), (2.0, 8.0, 2.0), (60.0, 150.0, 30.0))
    log.info("A04  LE(1-8) × SE(1-8) × STP(0.25-1 s0.25) × LMT(2-8 s2) × LenLE(60-150 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A05  wide_stp — STP=3–10×ATR (loose stops, let trades develop)
    #      LE(1-5 s1) × SE(1-5 s1) × STP(3-10 s1) × LMT(3-10 s2) × LenLE(60-150 s30)
    #      = 5×5×8×4×4 = 3200
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_wide_stp"
    _c = _cfg(_n, (1, 5, 1), (1, 5, 1), (3.0, 10.0, 1.0), (3.0, 10.0, 2.0), (60.0, 150.0, 30.0))
    log.info("A05  LE(1-5) × SE(1-5) × STP(3-10 s1) × LMT(3-10 s2) × LenLE(60-150 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A06  short_lenle — LenLE=10–60 (short MA = more switching long/short)
    #      LE(1-6 s1) × SE(1-6 s1) × STP(1-3 s1) × LMT(3-9 s2) × LenLE(10-60 s10)
    #      = 6×6×3×4×6 = 2592
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_short_lenle"
    _c = _cfg(_n, (1, 6, 1), (1, 6, 1), (1.0, 3.0, 1.0), (3.0, 9.0, 2.0), (10.0, 60.0, 10.0))
    log.info("A06  LE(1-6) × SE(1-6) × STP(1-3 s1) × LMT(3-9 s2) × LenLE(10-60 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A07  long_lenle — LenLE=120–360 (strict trend filter — fewer but cleaner trades)
    #      LE(1-6 s1) × SE(1-6 s1) × STP(1-3 s1) × LMT(3-9 s2) × LenLE(120-360 s60)
    #      = 6×6×3×4×5 = 2160
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_long_lenle"
    _c = _cfg(_n, (1, 6, 1), (1, 6, 1), (1.0, 3.0, 1.0), (3.0, 9.0, 2.0), (120.0, 360.0, 60.0))
    log.info("A07  LE(1-6) × SE(1-6) × STP(1-3 s1) × LMT(3-9 s2) × LenLE(120-360 s60)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A08  high_le — LE=10–30 (wider long breakout channel)
    #      LE(10-30 s5) × SE(1-10 s1) × STP(1-3 s1) × LMT(4-10 s2) × LenLE(60-150 s30)
    #      = 5×10×3×4×4 = 2400
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_high_le"
    _c = _cfg(_n, (10, 30, 5), (1, 10, 1), (1.0, 3.0, 1.0), (4.0, 10.0, 2.0), (60.0, 150.0, 30.0))
    log.info("A08  LE(10-30 s5) × SE(1-10) × STP(1-3 s1) × LMT(4-10 s2) × LenLE(60-150 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A09  high_se — SE=10–30 (wider short breakdown channel)
    #      LE(1-10 s1) × SE(10-30 s5) × STP(1-3 s1) × LMT(4-10 s2) × LenLE(60-150 s30)
    #      = 10×5×3×4×4 = 2400
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_high_se"
    _c = _cfg(_n, (1, 10, 1), (10, 30, 5), (1.0, 3.0, 1.0), (4.0, 10.0, 2.0), (60.0, 150.0, 30.0))
    log.info("A09  LE(1-10) × SE(10-30 s5) × STP(1-3 s1) × LMT(4-10 s2) × LenLE(60-150 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A10  very_high_lmt — LMT=20–50×ATR (maximize per-trade profit)
    #      LE(1-5 s1) × SE(1-5 s1) × STP(1-4 s1) × LMT(20-50 s10) × LenLE(60-150 s30)
    #      = 5×5×4×4×4 = 1600
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_very_high_lmt"
    _c = _cfg(_n, (1, 5, 1), (1, 5, 1), (1.0, 4.0, 1.0), (20.0, 50.0, 10.0), (60.0, 150.0, 30.0))
    log.info("A10  LE(1-5) × SE(1-5) × STP(1-4 s1) × LMT(20-50 s10) × LenLE(60-150 s30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom — zoom around best NP found in R1v2
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom"
    log.info("A11  adaptive_zoom — center: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_lenle, best_np)
    if start_attempt <= A:
        cfg11 = None
        for r_le, r_se, r_stp, r_lmt, r_lenle, step_stp, step_lmt, step_lenle in [
            (3, 3, 1.5, 4.0, 40, 0.25, 1.0, 20),
            (2, 2, 1.0, 3.0, 30, 0.25, 1.0, 20),
            (2, 2, 0.75, 2.0, 20, 0.25, 0.5, 20),
            (1, 1, 0.5, 1.5, 15, 0.25, 0.5, 15),
        ]:
            _le    = zoom(best_le,    r_le,    1.0,        LE_LO,    LE_HI)
            _se    = zoom(best_se,    r_se,    1.0,        SE_LO,    SE_HI)
            _stp   = zoom(best_stp,   r_stp,   step_stp,   STP_LO,   STP_HI)
            _lmt   = zoom(best_lmt,   r_lmt,   step_lmt,   LMT_LO,   LMT_HI)
            _lenle = zoom(best_lenle, r_lenle, step_lenle, LENLE_LO, LENLE_HI)
            cfg11  = _cfg(_n, _le, _se, _stp, _lmt, _lenle)
            if cfg11.total_runs() <= 5000:
                break
        if cfg11 is not None:
            log.info("A11  LE%s × SE%s × STP%s × LMT%s × LenLE%s  %d combos",
                     _le, _se, _stp, _lmt, _lenle, cfg11.total_runs())
            _update(run_or_load(_n, cfg11, conn, from_csv), cfg11, _n, A, cfg11.total_runs())
    log.info("After A11 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # A12  wide_boundary — coarse global sweep across all dimensions
    #      LE(1-25 s6) × SE(1-25 s6) × STP(1-5 s2) × LMT(2-20 s6) × LenLE(50-300 s50)
    #      = 5×5×3×4×6 = 1800
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_wide_boundary"
    _c = _cfg(_n, (1, 25, 6), (1, 25, 6), (1.0, 5.0, 2.0), (2.0, 20.0, 6.0), (50.0, 300.0, 50.0))
    log.info("A12  LE(1-25 s6) × SE(1-25 s6) × STP(1-5 s2) × LMT(2-20 s6) × LenLE(50-300 s50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ZW Daily _2021Basic_Break_CL v2 Round-1 COMPLETE")
    log.info("  Best NP: %.0f  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)
    log.info("  Target %.0f: %s", TARGET_NP, "★ MET" if target_met else f"NOT MET (gap +{TARGET_NP - best_np:.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LE": best_le, "SE": best_se, "STP": best_stp, "LMT": best_lmt, "LenLE": best_lenle,
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
    ap = argparse.ArgumentParser(
        description="ZW Daily _2021Basic_Break_CL v2 NP>700K Round-1 search")
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
