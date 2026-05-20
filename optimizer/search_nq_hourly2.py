"""
search_nq_hourly2.py — Breakout Hourly NP > 700,000 on CME.NQ HOT, Round 2

Round-1 findings (2026-05-18):
  Best: LE=1  SE=11  STP=5    LMT=8   NP=621,220  MDD=-80,975   obj=4,765,845  trades=5779
  Gap to 700K: +78,780 (+12.7%)

Key R1 observations:
  - LE=1 (shortest possible) outperformed LE=3-6 significantly
  - SE=10-11 (very short lookback) beats SE>20 and especially SE>50
  - LMT=8-9 (tight target) beats LMT=18-35; tight profits on NQ hourly are key
  - STP insensitive: STP=2-7 all give similar NP (~500K), STP>8 worse
  - A08 (low_lmt, LMT=5-18) was the breakthrough that found LMT=9 → A11 found LMT=8
  - TWF high-SE region (SE=65-85) completely ineffective for NQ

Round-2 strategy: Fine zoom + LMT/SE precision around LE=1, SE=11 zone
  R2.1  Ultra-fine: LE=1-4, SE=7-15, STP=4-6.5, LMT=5-12
  R2.2  Ultra-low LMT 3-8 probe
  R2.3  STP wide scan 2-8 on core region
  R2.4  Very low SE 3-12 step 1
  R2.5  LE range 1-6 on core
  R2.6  Higher LMT 10-20 (confirm LMT>12 is worse)
  R2.7  Very low STP 0.5-2 on core
  R2.8  High STP 7-15
  R2.9  SE+LMT cross: SE=5-20 × LMT=5-10
  R2.10 4D medium: LE=1-5, SE=5-20, STP=3-7, LMT=5-14
  R2.11 Dense zoom around best NP (adaptive)
  R2.12 Global boundary: LE=1-5, SE=5-200 s25, STP=1-10, LMT=3-50 s5

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  LE(1-4) × SE(7-15 s1) × STP(4-6.5 s0.5) × LMT(5-12 s1)     4×9×6×8  =1728
  A02  LE(1-4) × SE(7-15 s1) × STP(4-7 s0.5)   × LMT(3-8 s1)       4×9×7×6  =1512
  A03  LE(1-3) × SE(8-14 s1) × STP(2-8 s0.5)   × LMT(6-12 s1)      3×7×13×7 =1911
  A04  LE(1-4) × SE(3-12 s1) × STP(4-7 s0.5)   × LMT(5-12 s1)      4×10×7×8 =2240
  A05  LE(1-6) × SE(8-14 s1) × STP(4-7 s0.5)   × LMT(6-12 s1)      6×7×7×7  =2058
  A06  LE(1-4) × SE(8-14 s1) × STP(4-7 s0.5)   × LMT(10-20 s1)     4×7×7×11 =2156
  A07  LE(1-5) × SE(5-15 s1) × STP(0.5-2 s0.5) × LMT(4-14 s1)      5×11×4×11=2420
  A08  LE(1-4) × SE(8-14 s1) × STP(7-15 s0.5)  × LMT(5-12 s1)      4×7×17×8 =3808
  A09  LE(1-3) × SE(5-20 s1) × STP(4-6 s0.5)   × LMT(5-10 s1)      3×16×5×6 =1440
  A10  LE(1-5) × SE(5-20 s1) × STP(3-7 s1)     × LMT(5-14 s1)      5×16×5×10=4000
  A11  Dense zoom around best NP (progressive shrink, ≤5000)
  A12  LE(1-5) × SE(5-205 s25)×STP(1-10 s1)    × LMT(3-50 s5)      5×9×10×10=4500
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
SYMBOL     = "CME.NQ HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.1,  20.0
LMT_LO, LMT_HI = 2.0, 200.0
SE_LO,  SE_HI  = 3.0, 400.0
LE_LO,  LE_HI  = 1.0, 300.0

# Seed = Round-1 best
SEED_LE,  SEED_SE  = 1.0, 11.0
SEED_STP, SEED_LMT = 5.0,  8.0
SEED_NP   = 621_220.0
SEED_OBJ  = 4_765_845.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_hourly2_{int(time.time())}.log"
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
        name=f"NQH2_{name}",
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
    return OUTPUT_DIR / f"NQH2_{name}_raw.csv"


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
    log.info("=== Starting NQH2_%s (%d combos) ===", name, cfg.total_runs())
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
    if pos.empty:
        return fb_le, fb_se, fb_stp, fb_lmt, 0.0, 0.0, 0.0, 0, False
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
    above   = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    best_np  = max(log_, key=lambda x: x.get("net_profit", 0), default={})
    best_obj = max(log_, key=lambda x: x.get("objective", 0), default={})
    payload = {
        "strategy":           "Breakout_NQ_Hourly  (target NP>700K round-2)",
        "symbol":             SYMBOL,
        "timeframe":          "Hourly (60 min)",
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
    out = OUTPUT_DIR / "final_params_nq_hourly2.json"
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
    log.info("  NQ Hourly Breakout NP>700K Round-2")
    log.info("  R1 best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP)
    log.info("  Target: %.0f  Gap: %.0f (+%.1f%%)",
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

        log.info("  [A%02d %-25s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  %s",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f (need +%.1f%%)",
                 best_np, (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else 999)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  Ultra-fine zoom: SE=7-15, LMT=5-12  (4×9×6×8=1728)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_zoom"
    _c = _cfg(_n, (1, 4, 1), (7, 15, 1), (4.0, 6.5, 0.5), (5, 12, 1))
    log.info("A01  LE(1-4) × SE(7-15 s1) × STP(4-6.5 s0.5) × LMT(5-12 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  Ultra-low LMT 3-8  (4×9×7×6=1512)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ultra_low_lmt"
    _c = _cfg(_n, (1, 4, 1), (7, 15, 1), (4.0, 7.0, 0.5), (3, 8, 1))
    log.info("A02  LE(1-4) × SE(7-15 s1) × STP(4-7 s0.5) × LMT(3-8 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  Wide STP scan 2-8 on core  (3×7×13×7=1911)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_stp_wide"
    _c = _cfg(_n, (1, 3, 1), (8, 14, 1), (2.0, 8.0, 0.5), (6, 12, 1))
    log.info("A03  LE(1-3) × SE(8-14 s1) × STP(2-8 s0.5) × LMT(6-12 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  Very low SE 3-12 step 1  (4×10×7×8=2240)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_very_low_se"
    _c = _cfg(_n, (1, 4, 1), (3, 12, 1), (4.0, 7.0, 0.5), (5, 12, 1))
    log.info("A04  LE(1-4) × SE(3-12 s1) × STP(4-7 s0.5) × LMT(5-12 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  LE range 1-6  (6×7×7×7=2058)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_le_range"
    _c = _cfg(_n, (1, 6, 1), (8, 14, 1), (4.0, 7.0, 0.5), (6, 12, 1))
    log.info("A05  LE(1-6) × SE(8-14 s1) × STP(4-7 s0.5) × LMT(6-12 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  Higher LMT 10-20 confirm ceiling  (4×7×7×11=2156)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_higher_lmt"
    _c = _cfg(_n, (1, 4, 1), (8, 14, 1), (4.0, 7.0, 0.5), (10, 20, 1))
    log.info("A06  LE(1-4) × SE(8-14 s1) × STP(4-7 s0.5) × LMT(10-20 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Very low STP 0.5-2  (5×11×4×11=2420)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_very_low_stp"
    _c = _cfg(_n, (1, 5, 1), (5, 15, 1), (0.5, 2.0, 0.5), (4, 14, 1))
    log.info("A07  LE(1-5) × SE(5-15 s1) × STP(0.5-2 s0.5) × LMT(4-14 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  High STP 7-15  (4×7×17×8=3808)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_high_stp"
    _c = _cfg(_n, (1, 4, 1), (8, 14, 1), (7.0, 15.0, 0.5), (5, 12, 1))
    log.info("A08  LE(1-4) × SE(8-14 s1) × STP(7-15 s0.5) × LMT(5-12 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  SE+LMT cross: SE=5-20, LMT=5-10  (3×16×5×6=1440)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_se_lmt_cross"
    _c = _cfg(_n, (1, 3, 1), (5, 20, 1), (4.0, 6.0, 0.5), (5, 10, 1))
    log.info("A09  LE(1-3) × SE(5-20 s1) × STP(4-6 s0.5) × LMT(5-10 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  4D medium  (5×16×5×10=4000)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_4d_medium"
    _c = _cfg(_n, (1, 5, 1), (5, 20, 1), (3.0, 7.0, 1.0), (5, 14, 1))
    log.info("A10  LE(1-5) × SE(5-20 s1) × STP(3-7 s1) × LMT(5-14 s1)  %d combos",
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
    for _r_se, _r_stp, _r_lmt in [(5, 1.5, 4), (4, 1.0, 3), (3, 0.75, 2), (2, 0.5, 1.5)]:
        _le11  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se11  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, _r_stp, 0.25, STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, _r_lmt, 0.5,  LMT_LO, LMT_HI)
        _c = _cfg(_n, _le11, _se11, _stp11, _lmt11)
        if _c.total_runs() <= 5000:
            break
    log.info("A11  Dense LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le11, _se11, _stp11, _lmt11, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A12  Global boundary  (5×9×10×10=4500)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (1, 5, 1), (5, 205, 25), (1.0, 10.0, 1.0), (3, 52, 5))
    log.info("A12  LE(1-5) × SE(5-205 s25) × STP(1-10) × LMT(3-52 s5)  %d combos",
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

    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL — NQ Hourly Breakout NP>700K Round-2")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best NP: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
             best_entry.get("LE"), best_entry.get("SE"),
             best_entry.get("STP"), best_entry.get("LMT"),
             best_entry.get("net_profit", 0), best_entry.get("max_drawdown", 0),
             best_entry.get("objective", 0), best_entry.get("total_trades", 0))
    log.info("  Target NP>700K: %s", "MET ✓" if target_met else
             "NOT MET — best NP=%.0f" % best_np)
    log.info("")
    log.info("  %-3s %-28s %6s %10s %10s %12s %6s  %s",
             "A#", "Name", "Rows", "NP", "MDD", "Objective", "Trd", "★")
    for e in attempt_log:
        log.info("  A%02d %-28s %6d %10.0f %10.0f %12.0f %6d  %s",
                 e["attempt"], e["name"], e.get("rows", 0),
                 e.get("net_profit", 0), e.get("max_drawdown", 0),
                 e.get("objective", 0), e.get("total_trades", 0),
                 "★" if e.get("target_met") else "")

    out = save_json(best_entry, attempt_log, target_met)
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
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Administrator manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="NQ Hourly Breakout NP>700K Round-2 search")
    ap.add_argument("--from-csv",  action="store_true")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N")
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
