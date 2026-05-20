"""
search_hourly_target.py — Breakout Hourly NP > 6,000,000, Round 1

Existing best (search_hourly.py, 2026-05-14):
  LE=5  SE=15  STP=5.0  LMT=22.25  NP=4,914,000  MDD=-1,385,600  trades=2704

Key observations from prior hourly search:
  - SE=15 (fewer, higher-quality trades) beats SE=5 (too many, noisy trades)
  - LMT was capped at 24 — upper boundary never explored
  - STP was capped at 8 — also potentially limiting
  - LE seems stable at 5; wider range not tested
  - Hourly trades are frequent (2000-5000+) — MDD control is critical
  - Gap to 6M: +1,086,000 (+22%)

Round-1 strategy: Systematic 4D sweeps with all-4-params varying
  H1. SE 15-150: higher SE = more selective entries → fewer but bigger wins
  H2. LMT 25-80: extending beyond old 24 cap — hourly moves are smaller
  H3. STP 5-15: wider stops let trades breathe on hourly timeframe
  H4. LE 5-50: longer entry lookback probe
  H5. Low STP (0.5-3): test if tight stops help hourly (hypothesis: no)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  LE 4-6 × SE 5-80 step 5 × STP 4-6 step 0.5 × LMT 18-26 step 0.5
       SE systematic scan with proven STP/LMT             (3×16×5×17=4080)
  A02  LE 4-6 × SE 12-24 step 2 × STP 4-7 step 0.5 × LMT 22-50 step 2
       LMT above old cap (22-50)                          (3×7×7×15=2205)
  A03  LE 4-6 × SE 5-100 step 10 × STP 3-7 step 1 × LMT 18-34 step 2
       Wide SE + extended LMT                             (3×10×5×9=1350)
  A04  LE 4-6 × SE 30-150 step 10 × STP 4-6 step 0.5 × LMT 18-26 step 1
       High SE 30-150 probe                               (3×13×5×9=1755)
  A05  LE 5-60 step 5 × SE 12-20 step 2 × STP 4-6 step 0.5 × LMT 20-27 step 1
       LE range probe 5-60                                (12×5×5×8=2400)
  A06  LE 4-6 × SE 12-22 step 2 × STP 6-14 step 1 × LMT 18-30 step 2
       High STP 6-14 (wider stop)                         (3×6×9×7=1134)
  A07  LE 4-6 × SE 12-22 step 2 × STP 4-7 step 0.5 × LMT 40-100 step 10
       Ultra-high LMT 40-100                              (3×6×7×7=882)
  A08  LE 4-6 × SE 12-18 step 1 × STP 4-6 step 0.25 × LMT 20-27 step 0.5
       Precision zoom around existing best                (3×7×9×15=2835)
  A09  LE 4-6 × SE 12-22 step 2 × STP 0.5-3 step 0.5 × LMT 18-30 step 2
       Low STP hypothesis test (0.5-3)                   (3×6×6×7=756)
  A10  LE 3-10 step 1 × SE 5-40 step 5 × STP 3-8 step 0.5 × LMT 18-34 step 2
       4D medium grid                                     (8×8×11×9=6336 → trim)
       LE 4-8 step 1 × SE 5-30 step 5 × STP 3-8 step 0.5 × LMT 18-32 step 2
                                                          (5×6×11×8=2640)
  A11  Dense zoom around best after A01-A10  (progressive shrink)
  A12  LE 5-100 step 20 × SE 5-200 step 20 × STP 1-10 step 1 × LMT 10-60 step 5
       Global boundary                                    (5×10×10×11=5500 → trim)
       LE 5-100 step 20 × SE 5-200 step 20 × STP 1-8 step 1 × LMT 10-50 step 5
                                                          (5×10×8×9=3600)
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\hourly_target_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 6_000_000.0

STP_LO, STP_HI = 0.1,  20.0
LMT_LO, LMT_HI = 2.0, 200.0
SE_LO,  SE_HI  = 5.0, 400.0
LE_LO,  LE_HI  = 1.0, 300.0

# Seed from prior hourly search (2026-05-14)
SEED_LE,  SEED_SE  = 5.0,  15.0
SEED_STP, SEED_LMT = 5.0,  22.25
SEED_NP   = 4_914_000.0
SEED_OBJ  = 17_427_393.0  # NP²/|MDD|

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_hourly_target_{int(time.time())}.log"
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
        name=f"BHT_{name}",
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
    return OUTPUT_DIR / f"BHT_{name}_raw.csv"


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
    log.info("=== Attempt %-30s  (%d combos) ===", name, cfg.total_runs())
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
    best = pos.loc[pos["Objective"].idxmax()]
    le  = float(best["LE"]);  se  = float(best["SE"])
    stp = float(best["STP"]); lmt = float(best["LMT"])
    np_ = float(best.get("NetProfit", 0))
    mdd = float(best.get("MaxDrawdown", 0))
    tr  = int(best.get("TotalTrades", 0))
    obj = float(best["Objective"])
    log.info("  Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
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
    payload = {
        "strategy":           "Breakout_Hourly  (target NP>6M round-1)",
        "symbol":             SYMBOL,
        "timeframe":          "Hourly (60 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "seed": {"LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
                 "net_profit": SEED_NP, "objective": SEED_OBJ},
        "best_params":           best,
        "best_np_attempt":       best_np,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_hourly_target.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np, best_obj  = SEED_NP,  SEED_OBJ
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Hourly Breakout NP>6M Round-1")
    log.info("  Seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    log.info("  Target: %.0f  Gap: %.0f (+%.1f%%)",
             TARGET_NP, TARGET_NP - best_np,
             (TARGET_NP - best_np) / best_np * 100)
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

        if obj > best_obj or (met and np_ > best_np):
            best_le, best_se = le, se
            best_stp, best_lmt = stp, lmt
            best_obj = max(obj, best_obj)
        if np_ > best_np:
            best_np = np_
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
        log.info("       Global best: NP=%.0f (need +%.1f%%)",
                 best_np, (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else 0)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  SE scan with proven STP/LMT  (3×16×5×17=4080)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_se_scan"
    _c = _cfg(_n, (4,6,1), (5,80,5), (4.0,6.0,0.5), (18,26,0.5))
    log.info("A01  LE(4-6) × SE(5-80 s5) × STP(4-6 s0.5) × LMT(18-26 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  LMT above old cap 22-50  (3×7×7×15=2205)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_high_lmt"
    _c = _cfg(_n, (4,6,1), (12,24,2), (4.0,7.0,0.5), (22,50,2))
    log.info("A02  LE(4-6) × SE(12-24 s2) × STP(4-7 s0.5) × LMT(22-50 s2)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  Wide SE + extended LMT  (3×10×5×9=1350)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_wide_se_lmt"
    _c = _cfg(_n, (4,6,1), (5,100,10), (3.0,7.0,1.0), (18,34,2))
    log.info("A03  LE(4-6) × SE(5-100 s10) × STP(3-7) × LMT(18-34 s2)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  High SE 30-150 probe  (3×13×5×9=1755)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_high_se"
    _c = _cfg(_n, (4,6,1), (30,150,10), (4.0,6.0,0.5), (18,26,1))
    log.info("A04  LE(4-6) × SE(30-150 s10) × STP(4-6 s0.5) × LMT(18-26)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  LE range probe 5-60  (12×5×5×8=2400)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_le_probe"
    _c = _cfg(_n, (5,60,5), (12,20,2), (4.0,6.0,0.5), (20,27,1))
    log.info("A05  LE(5-60 s5) × SE(12-20 s2) × STP(4-6 s0.5) × LMT(20-27)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  High STP 6-14  (3×6×9×7=1134)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_stp"
    _c = _cfg(_n, (4,6,1), (12,22,2), (6.0,14.0,1.0), (18,30,2))
    log.info("A06  LE(4-6) × SE(12-22 s2) × STP(6-14) × LMT(18-30 s2)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Ultra-high LMT 40-100  (3×6×7×7=882)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ultra_lmt"
    _c = _cfg(_n, (4,6,1), (12,22,2), (4.0,7.0,0.5), (40,100,10))
    log.info("A07  LE(4-6) × SE(12-22 s2) × STP(4-7 s0.5) × LMT(40-100 s10)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  Precision zoom around existing best  (3×7×9×15=2835)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_precision_best"
    _c = _cfg(_n, (4,6,1), (12,18,1), (4.0,6.0,0.25), (20,27,0.5))
    log.info("A08  LE(4-6) × SE(12-18 s1) × STP(4-6 s0.25) × LMT(20-27 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  Low STP hypothesis 0.5-3  (3×6×6×7=756)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_low_stp"
    _c = _cfg(_n, (4,6,1), (12,22,2), (0.5,3.0,0.5), (18,30,2))
    log.info("A09  LE(4-6) × SE(12-22 s2) × STP(0.5-3 s0.5) × LMT(18-30 s2)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  4D medium grid  (5×6×11×8=2640)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_4d_medium"
    _c = _cfg(_n, (4,8,1), (5,30,5), (3.0,8.0,0.5), (18,32,2))
    log.info("A10  LE(4-8) × SE(5-30 s5) × STP(3-8 s0.5) × LMT(18-32 s2)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom around best after A01-A10  (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt in [(10, 1.5, 5), (7, 1.0, 4), (5, 0.75, 3), (3, 0.5, 2)]:
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
    # A12  Global boundary  (5×10×8×9=3600)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (5,100,20), (5,200,20), (1.0,8.0,1.0), (10,50,5))
    log.info("A12  LE(5-100 s20) × SE(5-200 s20) × STP(1-8) × LMT(10-50 s5)  %d combos", _c.total_runs())
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
                            SEED_OBJ, SEED_NP, 0.0, 0, False, 0)

    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL — Hourly Breakout NP>6M Round-1")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
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
    ap = argparse.ArgumentParser(description="Hourly Breakout NP>6M search")
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
