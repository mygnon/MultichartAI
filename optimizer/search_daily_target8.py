"""
search_daily_target8.py — Breakout Daily NP > 6,000,000, Round 8

Reference: Round-5 seed (LE=5 SE=50 STP=2.1 LMT=17, NP=4,908,600 with old data).

After Rounds 6-7, the low-STP region (STP 0.1-0.5) plateaus at NP~4.09M.
Round-5 used STP 1-5 territory — this was NEVER cleanly scanned with all-4-params
varying (the column-mapping bug contaminated all Round-5 fixed-param attempts).

Round-8 strategy: fully explore STP 0.3-5 with correct 4D sweeps:
  - Systematic scan of the STP gap 0.3-1.5 (between R7 best and R5 territory)
  - Fine scan of STP 1.5-4 (R5 seed zone — never cleanly validated)
  - SE × STP 2D sweeps at various LMT levels
  - High LMT (20-35) paired with mid-high STP (matching R5's LMT=17-60 attempts)

Seeds:
  R7 best (high obj): LE=5 SE=49 STP=0.19 LMT=16  NP=4,089,000  obj=26,405,434
  R5 seed (old data): LE=5 SE=50 STP=2.1  LMT=17  NP=4,908,600  (reference direction)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  LE 4-6 × SE 46-54 step 1 × STP 0.3-1.5 step 0.1 × LMT 14-19 step 0.5
       STP gap 0.3-1.5 fine scan                    (3×9×13×11=3861)
  A02  LE 4-6 × SE 46-54 step 2 × STP 1.5-4 step 0.25 × LMT 15-20 step 0.5
       R5 STP zone 1.5-4 precision                  (3×5×11×11=1815)
  A03  LE 3-8 × SE 35-70 step 5 × STP 1.5-3.5 step 0.5 × LMT 15-20 step 1
       Wide SE × R5-STP 2D                          (6×8×5×6=1440)
  A04  LE 4-6 × SE 44-56 step 2 × STP 1.8-2.8 step 0.1 × LMT 15-20 step 0.5
       Fine scan around R5 seed (STP≈2.1)           (3×7×11×11=2541)
  A05  LE 5-7 × SE 60-110 step 5 × STP 1.5-3.5 step 0.5 × LMT 15-20 step 1
       High SE + R5-STP                             (3×11×5×6=990)
  A06  LE 4-6 × SE 46-54 step 2 × STP 1.8-2.5 step 0.1 × LMT 12-24 step 0.5
       LMT full scan with STP≈2                     (3×5×8×25=3000)
  A07  LE 3-8 × SE 44-56 step 2 × STP 2-4 step 0.25 × LMT 15-22 step 1
       Mid-high STP 2-4                             (6×7×9×8=3024)
  A08  LE 4-8 × SE 60-100 step 5 × STP 1.5-4 step 0.5 × LMT 15-22 step 1
       High SE + wider STP+LMT                      (5×9×6×8=2160)
  A09  LE 4-6 × SE 46-54 step 2 × STP 4-10 step 1 × LMT 15-25 step 1
       Very high STP 4-10                           (3×5×7×11=1155)
  A10  LE 4-6 × SE 46-54 step 1 × STP 2-4 step 0.25 × LMT 20-35 step 1
       High STP + high LMT                          (3×9×9×16=3888)
  A11  Zoom around best after A01-A10  (progressive shrink)
  A12  LE 1-20 step 2 × SE 10-200 step 20 × STP 0.5-5 step 0.5 × LMT 14-22 step 2
       Global R5-style boundary                     (10×10×10×5=5000)
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_target8_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 6_000_000.0

STP_LO, STP_HI = 0.01, 20.0
LMT_LO, LMT_HI = 2.0, 100.0
SE_LO,  SE_HI  = 1.0, 400.0
LE_LO,  LE_HI  = 1.0, 100.0

# Round-7 best (highest objective, current data 2026-05-17)
SEED_LE,  SEED_SE  = 5.0,  49.0
SEED_STP, SEED_LMT = 0.19, 16.0
SEED_NP   = 4_089_000.0
SEED_OBJ  = 26_405_434.0

# Round-5 reference direction (old data — use as search hypothesis, not NP baseline)
R5_REF = {"LE": 5, "SE": 50, "STP": 2.1, "LMT": 17, "NP_old": 4_908_600}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_daily_target8_{int(time.time())}.log"
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
        name=f"BD8_{name}",
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


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"BD8_{name}_raw.csv"


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
        "strategy":           "Breakout_Daily  (target NP>6M round-8)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "r7_best":  {"LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
                     "net_profit": SEED_NP, "objective": SEED_OBJ},
        "r5_reference": R5_REF,
        "best_params":           best,
        "best_np_attempt":       best_np,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_daily_target8.json"
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
    log.info("  Round-8 Breakout Daily NP>6M  (R5 territory + STP gap)")
    log.info("  Seed (R7): LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    log.info("  R5 ref:    LE=5    SE=50   STP=2.1  LMT=17    NP=4908600 (old data)")
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
    # A01  STP gap 0.3-1.5 fine  (3×9×13×11=3861)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_stp_gap"
    _c = _cfg(_n, (4,6,1), (46,54,1), (0.3,1.5,0.1), (14,19,0.5))
    log.info("A01  LE(4-6) × SE(46-54 s1) × STP(0.3-1.5 s0.1) × LMT(14-19 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  R5 STP zone 1.5-4 precision  (3×5×11×11=1815)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_r5_stp_zone"
    _c = _cfg(_n, (4,6,1), (46,54,2), (1.5,4.0,0.25), (15,20,0.5))
    log.info("A02  LE(4-6) × SE(46-54 s2) × STP(1.5-4 s0.25) × LMT(15-20 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  Wide SE × R5-STP 2D  (6×8×5×6=1440)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_wide_se_r5stp"
    _c = _cfg(_n, (3,8,1), (35,70,5), (1.5,3.5,0.5), (15,20,1))
    log.info("A03  LE(3-8) × SE(35-70 s5) × STP(1.5-3.5 s0.5) × LMT(15-20)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  Fine scan R5 seed zone STP≈2.1  (3×7×11×11=2541)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_r5_seed_fine"
    _c = _cfg(_n, (4,6,1), (44,56,2), (1.8,2.8,0.1), (15,20,0.5))
    log.info("A04  LE(4-6) × SE(44-56 s2) × STP(1.8-2.8 s0.1) × LMT(15-20 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  High SE + R5-STP  (3×11×5×6=990)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_high_se_r5stp"
    _c = _cfg(_n, (5,7,1), (60,110,5), (1.5,3.5,0.5), (15,20,1))
    log.info("A05  LE(5-7) × SE(60-110 s5) × STP(1.5-3.5 s0.5) × LMT(15-20)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  LMT full scan 12-24 with STP≈2  (3×5×8×25=3000)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_lmt_scan_stp2"
    _c = _cfg(_n, (4,6,1), (46,54,2), (1.8,2.5,0.1), (12,24,0.5))
    log.info("A06  LE(4-6) × SE(46-54 s2) × STP(1.8-2.5 s0.1) × LMT(12-24 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Mid-high STP 2-4 wide  (6×7×9×8=3024)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_stp2to4"
    _c = _cfg(_n, (3,8,1), (44,56,2), (2.0,4.0,0.25), (15,22,1))
    log.info("A07  LE(3-8) × SE(44-56 s2) × STP(2-4 s0.25) × LMT(15-22)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  High SE + wider STP+LMT  (5×9×6×8=2160)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_high_se_wide"
    _c = _cfg(_n, (4,8,1), (60,100,5), (1.5,4.0,0.5), (15,22,1))
    log.info("A08  LE(4-8) × SE(60-100 s5) × STP(1.5-4 s0.5) × LMT(15-22)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  Very high STP 4-10  (3×5×7×11=1155)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_very_high_stp"
    _c = _cfg(_n, (4,6,1), (46,54,2), (4.0,10.0,1.0), (15,25,1))
    log.info("A09  LE(4-6) × SE(46-54 s2) × STP(4-10) × LMT(15-25)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  High STP + high LMT  (3×9×9×16=3888)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_high_stp_lmt"
    _c = _cfg(_n, (4,6,1), (46,54,1), (2.0,4.0,0.25), (20,35,1))
    log.info("A10  LE(4-6) × SE(46-54 s1) × STP(2-4 s0.25) × LMT(20-35)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom around best after A01-A10
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt in [(5, 0.5, 3), (4, 0.4, 2.5), (3, 0.3, 2), (2, 0.2, 1.5)]:
        _le11  = zoom(best_le,  1,      1,     LE_LO,  LE_HI)
        _se11  = zoom(best_se,  _r_se,  1,     SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, _r_stp, 0.05,  STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, _r_lmt, 0.5,   LMT_LO, LMT_HI)
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
    # A12  Global R5-style boundary  (10×10×10×5=5000)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (1,20,2), (10,200,20), (0.5,5.0,0.5), (14,22,2))
    log.info("A12  LE(1-20 s2) × SE(10-200 s20) × STP(0.5-5 s0.5) × LMT(14-22 s2)  %d combos", _c.total_runs())
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
    log.info("  FINAL — Round-8 Breakout Daily")
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
    ap = argparse.ArgumentParser(description="Round-8 Breakout Daily — R5 territory scan")
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
    import sys
    sys.exit(main())
