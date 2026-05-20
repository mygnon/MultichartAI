"""
search_hourly_target2.py — Breakout Hourly NP > 6,000,000, Round 2

Round-1 findings (2026-05-18):
  Best NP:  LE=5  SE=75  STP=5    LMT=32   NP=5,210,000  MDD=-1,070,000  obj=25,368,318  trades=775
  Best Obj: LE=25 SE=65  STP=6    LMT=10   NP=4,622,600  MDD=-694,400    obj=30,772,510  trades=641
  Gap to 6M: +790,000 (+15.2%)

Two clusters found:
  Cluster A  SE=70-80, LMT=32+    → high NP (~5.2M), higher MDD (~1.07M)
  Cluster B  SE=14,    LMT=10-22  → high obj (~30M),  lower MDD (~762K), NP ~4.79M
  Cluster C  LE=25, SE=65, LMT=10 → best obj (30.77M), NP=4.62M (boundary find)

Round-2 strategy: Push Cluster A NP above 6M
  R2.1  Finer SE around 75, extend LMT to 46 (step 2)
  R2.2  LMT 36-60 scan (beyond R1's 34 cap) with SE=65-90
  R2.3  Very high SE (80-150) + extended LMT 22-40
  R2.4  A12 regime zoom: LE=15-40, SE=45-80, STP=4.5-8, LMT=8-26
  R2.5  STP range 3-10 on SE=70-80 zone
  R2.6  LE 5-30 × SE 60-90 cross-scan
  R2.7  Ultra-high LMT 50-100 on SE=65-85
  R2.8  Precision on R1 best NP (SE=70-80 step 1, LMT=29-37 step 1)
  R2.9  Low STP 0.5-3 with high SE=65-85
  R2.10 4D medium: LE=4-10, SE=60-90, STP=4-7, LMT=26-40
  R2.11 Dense zoom around best NP found so far (adaptive)
  R2.12 Global boundary (coarser, new regimes)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  LE(4-6) × SE(60-90 s3) × STP(4-6 s0.5) × LMT(28-46 s2)       3×11×5×10=1650
  A02  LE(4-6) × SE(65-90 s5) × STP(4-7 s0.5) × LMT(36-60 s2)       3×6×7×13=1638
  A03  LE(4-6) × SE(80-150 s10) × STP(4-7 s0.5) × LMT(22-40 s2)     3×8×7×10=1680
  A04  LE(15-40 s5) × SE(45-80 s5) × STP(4.5-8 s0.5) × LMT(8-26 s2) 6×8×8×10=3840
  A05  LE(4-6) × SE(65-85 s5) × STP(3-10 s0.5) × LMT(28-42 s2)      3×5×15×8=1800
  A06  LE(5-30 s5) × SE(60-90 s5) × STP(4-7 s0.5) × LMT(28-38 s2)   6×7×7×6=1764
  A07  LE(4-6) × SE(65-85 s5) × STP(4-7 s0.5) × LMT(50-100 s4)      3×5×7×14=1470
  A08  LE(4-6) × SE(70-80 s1) × STP(4.25-5.75 s0.25) × LMT(29-37 s1) 3×11×7×9=2079
  A09  LE(4-6) × SE(65-85 s5) × STP(0.5-3 s0.5) × LMT(26-42 s2)     3×5×6×9=810
  A10  LE(4-10 s1) × SE(60-90 s5) × STP(4-7 s0.5) × LMT(26-40 s2)   7×7×7×8=2744
  A11  Dense zoom around best NP params (progressive shrink, ≤5000)
  A12  LE(5-100 s25) × SE(5-200 s25) × STP(1-10) × LMT(10-60 s5)    4×9×10×11=3960
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
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\hourly_target2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 6_000_000.0

STP_LO, STP_HI = 0.1,  20.0
LMT_LO, LMT_HI = 2.0, 200.0
SE_LO,  SE_HI  = 5.0, 400.0
LE_LO,  LE_HI  = 1.0, 300.0

# Seed = Round-1 best NP result
SEED_LE,  SEED_SE  = 5.0,  75.0
SEED_STP, SEED_LMT = 5.0,  32.0
SEED_NP   = 5_210_000.0
SEED_OBJ  = 25_368_318.0

# Round-1 obj champion (for reference)
R1_OBJ_LE, R1_OBJ_SE   = 25.0, 65.0
R1_OBJ_STP, R1_OBJ_LMT = 6.0,  10.0
R1_OBJ_NP, R1_OBJ_OBJ  = 4_622_600.0, 30_772_510.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_hourly_target2_{int(time.time())}.log"
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
        name=f"BHT2_{name}",
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
    return OUTPUT_DIR / f"BHT2_{name}_raw.csv"


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
    log.info("=== Starting BHT2_%s (%d combos) ===", name, cfg.total_runs())
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
    # Use highest NP (not obj) to track the NP-champion for zoom
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
    best_np = max(log_, key=lambda x: x.get("net_profit", 0), default={})
    best_obj = max(log_, key=lambda x: x.get("objective", 0), default={})
    payload = {
        "strategy":           "Breakout_Hourly  (target NP>6M round-2)",
        "symbol":             SYMBOL,
        "timeframe":          "Hourly (60 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "r1_best_np": {
            "LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "r1_best_obj": {
            "LE": R1_OBJ_LE, "SE": R1_OBJ_SE,
            "STP": R1_OBJ_STP, "LMT": R1_OBJ_LMT,
            "net_profit": R1_OBJ_NP, "objective": R1_OBJ_OBJ,
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "best_obj_attempt":      best_obj,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_hourly_target2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track best NP params (used for A11 dense zoom)
    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Hourly Breakout NP>6M Round-2")
    log.info("  R1 best NP: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
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

        # Update zoom params based on NP (our primary target)
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
                 "★TARGET★" if met else ("%.0f/6M" % np_))
        log.info("       Global best NP=%.0f (need +%.1f%%)",
                 best_np, (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else 0)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  Fine SE zoom + extend LMT to 46  (3×11×5×10=1650)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_se_fine_lmt46"
    _c = _cfg(_n, (4, 6, 1), (60, 90, 3), (4.0, 6.0, 0.5), (28, 46, 2))
    log.info("A01  LE(4-6) × SE(60-90 s3) × STP(4-6 s0.5) × LMT(28-46 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  LMT 36-60 extension with SE=65-90  (3×6×7×13=1638)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_lmt60_se90"
    _c = _cfg(_n, (4, 6, 1), (65, 90, 5), (4.0, 7.0, 0.5), (36, 60, 2))
    log.info("A02  LE(4-6) × SE(65-90 s5) × STP(4-7 s0.5) × LMT(36-60 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  Very high SE 80-150 + extended LMT 22-40  (3×8×7×10=1680)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_very_high_se"
    _c = _cfg(_n, (4, 6, 1), (80, 150, 10), (4.0, 7.0, 0.5), (22, 40, 2))
    log.info("A03  LE(4-6) × SE(80-150 s10) × STP(4-7 s0.5) × LMT(22-40 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  A12-regime zoom: LE=15-40, SE=45-80, STP=4.5-8, LMT=8-26  (6×8×8×10=3840)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_boundary_regime"
    _c = _cfg(_n, (15, 40, 5), (45, 80, 5), (4.5, 8.0, 0.5), (8, 26, 2))
    log.info("A04  LE(15-40 s5) × SE(45-80 s5) × STP(4.5-8 s0.5) × LMT(8-26 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  STP range 3-10 on SE=65-85 zone  (3×5×15×8=1800)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_stp_wide"
    _c = _cfg(_n, (4, 6, 1), (65, 85, 5), (3.0, 10.0, 0.5), (28, 42, 2))
    log.info("A05  LE(4-6) × SE(65-85 s5) × STP(3-10 s0.5) × LMT(28-42 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  LE 5-30 × SE 60-90 cross-scan  (6×7×7×6=1764)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_le_se_cross"
    _c = _cfg(_n, (5, 30, 5), (60, 90, 5), (4.0, 7.0, 0.5), (28, 38, 2))
    log.info("A06  LE(5-30 s5) × SE(60-90 s5) × STP(4-7 s0.5) × LMT(28-38 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Ultra-high LMT 50-100 with SE=65-85  (3×5×7×14=1470)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ultra_lmt"
    _c = _cfg(_n, (4, 6, 1), (65, 85, 5), (4.0, 7.0, 0.5), (50, 102, 4))
    log.info("A07  LE(4-6) × SE(65-85 s5) × STP(4-7 s0.5) × LMT(50-102 s4)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  Precision on R1 NP champion: SE=70-80 step1, LMT=29-37 step1  (3×11×7×9=2079)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_precision_np"
    _c = _cfg(_n, (4, 6, 1), (70, 80, 1), (4.25, 5.75, 0.25), (29, 37, 1))
    log.info("A08  LE(4-6) × SE(70-80 s1) × STP(4.25-5.75 s0.25) × LMT(29-37 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  Low STP 0.5-3 with high SE=65-85  (3×5×6×9=810)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_low_stp_high_se"
    _c = _cfg(_n, (4, 6, 1), (65, 85, 5), (0.5, 3.0, 0.5), (26, 42, 2))
    log.info("A09  LE(4-6) × SE(65-85 s5) × STP(0.5-3 s0.5) × LMT(26-42 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  4D medium: LE=4-10, SE=60-90, STP=4-7, LMT=26-40  (7×7×7×8=2744)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_4d_medium"
    _c = _cfg(_n, (4, 10, 1), (60, 90, 5), (4.0, 7.0, 0.5), (26, 40, 2))
    log.info("A10  LE(4-10 s1) × SE(60-90 s5) × STP(4-7 s0.5) × LMT(26-40 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom around best NP found  (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt in [(8, 1.0, 4), (6, 0.75, 3), (4, 0.5, 2), (3, 0.5, 1.5)]:
        _le11  = zoom(best_le,  2,      1,    LE_LO,  LE_HI)
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
    # A12  Global boundary R2  (4×9×10×11=3960)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (5, 100, 25), (5, 205, 25), (1.0, 10.0, 1.0), (10, 60, 5))
    log.info("A12  LE(5-100 s25) × SE(5-205 s25) × STP(1-10) × LMT(10-60 s5)  %d combos",
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
    log.info("  FINAL — Hourly Breakout NP>6M Round-2")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best NP: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
             best_entry.get("LE", best_le), best_entry.get("SE", best_se),
             best_entry.get("STP", best_stp), best_entry.get("LMT", best_lmt),
             best_entry.get("net_profit", best_np),
             best_entry.get("max_drawdown", 0),
             best_entry.get("objective", best_obj),
             best_entry.get("total_trades", 0))
    log.info("  Target NP>6M: %s", "MET ✓" if target_met else
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
    print(f"Target NP>6M: {'MET ✓' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
    ap = argparse.ArgumentParser(description="Hourly Breakout NP>6M Round-2 search")
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
