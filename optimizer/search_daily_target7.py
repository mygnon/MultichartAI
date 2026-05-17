"""
search_daily_target7.py — Breakout Daily NP > 6,000,000, Round 7

Key findings from Round 6 (2026-05-17):
  - Best: LE=5 SE=49 STP=0.2 LMT=16  NP=4,074,200  obj=26,173,298  trades=53
  - Best NP: LE=5 SE=50 STP=0.25 LMT=16  NP=4,089,800  obj=26,021,257  trades=47
  - Very low STP (0.2–0.25) is the ONLY region that broke NP=4M
  - LMT=16 consistently outperforms LMT=17 in this low-STP zone
  - Gap to 6M: −1,925,800 (−32%)

Round-7 strategy:
  Deep dive into the low-STP landscape — the only region that produces NP > 4M.
  Explore STP 0.02–0.5 systematically, widen SE/LE axes around the promising zone.

Attempt schedule (12 attempts, each ≤ 5,000 combos):
  A01  LE 4-6 × SE 45-55 step 1 × STP 0.05-0.45 step 0.05 × LMT 13-19 step 0.5
       Core low-STP fine scan                        (3×11×9×13=3861)
  A02  LE 4-6 × SE 45-55 step 1 × STP 0.02-0.18 step 0.02 × LMT 14-19 step 0.5
       Ultra-low STP 0.02–0.18 zone                  (3×11×9×11=3267)
  A03  LE 4-6 × SE 40-70 step 5 × STP 0.05-0.5 step 0.05 × LMT 14-18 step 1
       Wider SE + low STP                             (3×7×10×5=1050)
  A04  LE 4-8 × SE 70-120 step 5 × STP 0.05-0.4 step 0.05 × LMT 14-18 step 1
       High-SE + low STP                              (5×11×8×5=2200)
  A05  LE 1-10 step 1 × SE 47-53 step 1 × STP 0.1-0.4 step 0.1 × LMT 14-18 step 1
       LE probe 1-10 with tight SE                    (10×7×4×5=1400)
  A06  LE 4-6 × SE 44-54 step 1 × STP 0.1-0.35 step 0.05 × LMT 13-19 step 0.5
       Precision zoom around A11 best (SE=49 STP=0.2)(3×11×6×13=2574)
  A07  LE 4-6 × SE 45-55 step 1 × STP 0.02-0.15 step 0.01 × LMT 14-18 step 1
       Very-low STP: does STP<0.1 add more trades?   (3×11×14×5=2310)
  A08  LE 5-7 × SE 80-150 step 5 × STP 0.1-0.5 step 0.1 × LMT 14-18 step 1
       High-SE (80-150) + low STP                    (3×15×5×5=1125)
  A09  LE 4-6 × SE 45-55 step 2 × STP 0.1-0.4 step 0.1 × LMT 10-22 step 0.5
       LMT scan 10-22 — does NP peak below LMT=14?  (3×6×4×25=1800)
  A10  LE 1-4 step 1 × SE 40-65 step 5 × STP 0.05-0.4 step 0.05 × LMT 14-18 step 1
       LE=1-4 + low STP                               (4×6×8×5=960)
  A11  LE 3-8 × SE 40-70 step 5 × STP 0.1-0.5 step 0.1 × LMT 14-20 step 1
       Medium 4D with low STP                         (6×7×5×7=1470)
  A12  LE 1-15 step 2 × SE 20-200 step 20 × STP 0.1-0.5 step 0.1 × LMT 12-22 step 2
       Global boundary check                          (8×10×5×6=2400)

Champion selection:
  Primary:  NP > 6,000,000 → highest objective = NP²/|MDD|
  Fallback: highest NP (track progress toward target)
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

import numpy as np
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_target7_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 6_000_000.0
MAX_ATTEMPTS = 12

# Generous global bounds (per-attempt ranges are much tighter)
STP_LO, STP_HI = 0.01, 20.0
LMT_LO, LMT_HI = 2.0, 100.0
SE_LO,  SE_HI  = 1.0, 400.0
LE_LO,  LE_HI  = 1.0, 100.0

# Round-6 verified best (2026-05-17) — highest objective
SEED_LE,  SEED_SE  = 5.0,  49.0
SEED_STP, SEED_LMT = 0.2,  16.0
SEED_NP   = 4_074_200.0
SEED_OBJ  = 26_173_298.0  # NP²/|MDD| = 4074200²/634200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_daily_target7_{int(time.time())}.log"
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
# Range helpers
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


# ─────────────────────────────────────────────────────────────────────────────
# Config factory — ALWAYS vary all 4 params to avoid column-mapping bug
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(name: str,
         le:  Tuple[float, float, float],
         se:  Tuple[float, float, float],
         stp: Tuple[float, float, float],
         lmt: Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t: Tuple[float, float, float]) -> Tuple[float, float, float]:
        s, e, step = t
        if s == e:
            return (max(LE_LO, s - step), min(LE_HI, s + step), step)
        return t
    le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)

    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000 limit!", name, combos)
    return StrategyConfig(
        name=f"BD7_{name}",
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


# ─────────────────────────────────────────────────────────────────────────────
# Run or load CSV
# ─────────────────────────────────────────────────────────────────────────────

def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"BD7_{name}_raw.csv"


def run_or_load(name: str, cfg: StrategyConfig,
                conn: Optional[mc.MultiChartsConnection],
                from_csv: bool) -> Optional[pd.DataFrame]:
    csv_path = csv_for(name)
    if from_csv or csv_path.exists():
        if csv_path.exists():
            try:
                df = mc.load_results_csv(str(csv_path), cfg)
                log.info("Loaded %s: %d rows from %s", name, len(df), csv_path.name)
                return df
            except Exception as e:
                log.warning("Could not load %s: %s", csv_path, e)
        else:
            log.warning("No CSV for %s at %s", name, csv_path)
        return None

    log.info("=== Attempt %-30s  (%d combos) ===", name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done in %.1f min — %s", (time.time() - t0) / 60, Path(raw_csv).name)
        return mc.load_results_csv(raw_csv, cfg)
    except Exception as e:
        log.error("  FAILED: %s", e, exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Validate CSV param columns are within expected ranges
# ─────────────────────────────────────────────────────────────────────────────

def _validate_df(df: pd.DataFrame, cfg: StrategyConfig) -> bool:
    if df is None or df.empty:
        return False
    for p in cfg.params:
        if p.name not in df.columns:
            continue
        lo = min(p.start, p.stop) - abs(p.step) * 2
        hi = max(p.start, p.stop) + abs(p.step) * 2
        col = pd.to_numeric(df[p.name], errors="coerce")
        if not col.between(lo, hi).all():
            log.warning("  INVALID: %s column out of expected range [%.4g, %.4g]. "
                        "Got min=%.4g max=%.4g. Skipping.",
                        p.name, lo, hi, col.min(), col.max())
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Champion selector
# ─────────────────────────────────────────────────────────────────────────────

def champion(df: pd.DataFrame,
             fb_le: float, fb_se: float,
             fb_stp: float, fb_lmt: float,
             ) -> Tuple[float, float, float, float, float, float, float, int, bool]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Record helpers
# ─────────────────────────────────────────────────────────────────────────────

def _entry(attempt: int, name: str, df: Optional[pd.DataFrame],
           le: float, se: float, stp: float, lmt: float,
           obj: float, np_: float, mdd: float, trades: int,
           met: bool, combos: int) -> Dict:
    return {
        "attempt": attempt, "name": name,
        "combos": combos, "rows": len(df) if df is not None else 0,
        "LE": le, "SE": se, "STP": stp, "LMT": lmt,
        "net_profit":   round(np_, 0),
        "max_drawdown": round(mdd, 0),
        "objective":    round(obj, 0),
        "total_trades": trades,
        "target_met":   met,
    }


def save_json(best: Dict, log_: List[dict], met: bool) -> Path:
    above   = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    best_np = max(log_, key=lambda x: x.get("net_profit", 0), default={})
    payload = {
        "strategy":           "Breakout_Daily  (target NP>6M round-7)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "round6_best": {
            "LE": SEED_LE, "SE": SEED_SE,
            "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_daily_target7.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn: Optional[mc.MultiChartsConnection],
               from_csv: bool, start_attempt: int) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met   = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Round-7 Breakout Daily NP>6M  (deep low-STP exploration)")
    log.info("  Seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    log.info("  Target: %.0f  Gap: %.0f (+%.1f%%)",
             TARGET_NP, TARGET_NP - best_np,
             (TARGET_NP - best_np) / best_np * 100)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df: Optional[pd.DataFrame], cfg: StrategyConfig,
                name: str, attempt_num: int, combos: int) -> None:
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            entry = _entry(attempt_num, name, df,
                           best_le, best_se, best_stp, best_lmt,
                           0.0, 0.0, 0.0, 0, False, combos)
            attempt_log.append(entry)
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
                 attempt_num, name, le, se, stp, lmt,
                 obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/6M" % np_))
        log.info("       Global best so far: NP=%.0f (need +%.1f%%)",
                 best_np, (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else 0)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  LE 4-6 × SE 45-55 step 1 × STP 0.05-0.45 step 0.05 × LMT 13-19 step 0.5
    #      Core low-STP fine scan  (3×11×9×13=3861)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_low_stp_core"
    _c = _cfg(_n, (4,6,1), (45,55,1), (0.05,0.45,0.05), (13,19,0.5))
    log.info("A01  LE(4-6) × SE(45-55 s1) × STP(0.05-0.45 s0.05) × LMT(13-19 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  LE 4-6 × SE 45-55 step 1 × STP 0.02-0.18 step 0.02 × LMT 14-19 step 0.5
    #      Ultra-low STP zone  (3×11×9×11=3267)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ultra_low_stp"
    _c = _cfg(_n, (4,6,1), (45,55,1), (0.02,0.18,0.02), (14,19,0.5))
    log.info("A02  LE(4-6) × SE(45-55 s1) × STP(0.02-0.18 s0.02) × LMT(14-19 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  LE 4-6 × SE 40-70 step 5 × STP 0.05-0.5 step 0.05 × LMT 14-18 step 1
    #      Wider SE + low STP  (3×7×10×5=1050)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_wider_se_low_stp"
    _c = _cfg(_n, (4,6,1), (40,70,5), (0.05,0.5,0.05), (14,18,1))
    log.info("A03  LE(4-6) × SE(40-70 s5) × STP(0.05-0.5 s0.05) × LMT(14-18)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  LE 4-8 × SE 70-120 step 5 × STP 0.05-0.4 step 0.05 × LMT 14-18 step 1
    #      High-SE + low STP  (5×11×8×5=2200)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_high_se_low_stp"
    _c = _cfg(_n, (4,8,1), (70,120,5), (0.05,0.4,0.05), (14,18,1))
    log.info("A04  LE(4-8) × SE(70-120 s5) × STP(0.05-0.4 s0.05) × LMT(14-18)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  LE 1-10 step 1 × SE 47-53 step 1 × STP 0.1-0.4 step 0.1 × LMT 14-18 step 1
    #      LE probe 1-10 with tight SE  (10×7×4×5=1400)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_le_probe"
    _c = _cfg(_n, (1,10,1), (47,53,1), (0.1,0.4,0.1), (14,18,1))
    log.info("A05  LE(1-10) × SE(47-53 s1) × STP(0.1-0.4 s0.1) × LMT(14-18)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  Precision zoom around A11 best (LE=5 SE=49 STP=0.2 LMT=16)
    #      LE 4-6 × SE 44-54 step 1 × STP 0.1-0.35 step 0.05 × LMT 13-19 step 0.5
    #      (3×11×6×13=2574)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_zoom_r6best"
    _le6  = zoom(best_le,  1,    1,    LE_LO,  LE_HI)
    _se6  = zoom(best_se,  5,    1,    SE_LO,  SE_HI)
    _stp6 = zoom(best_stp, 0.12, 0.05, STP_LO, STP_HI)
    _lmt6 = zoom(best_lmt, 3,    0.5,  LMT_LO, LMT_HI)
    # Progressive shrink if over 5000
    for _r_se, _r_stp, _r_lmt in [(5,0.12,3), (4,0.1,2.5), (3,0.08,2), (2,0.06,1.5)]:
        _se6  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp6 = zoom(best_stp, _r_stp, 0.05, STP_LO, STP_HI)
        _lmt6 = zoom(best_lmt, _r_lmt, 0.5,  LMT_LO, LMT_HI)
        _c = _cfg(_n, _le6, _se6, _stp6, _lmt6)
        if _c.total_runs() <= 5000:
            break
    log.info("A06  Zoom best LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le6, _se6, _stp6, _lmt6, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  LE 4-6 × SE 45-55 step 1 × STP 0.02-0.15 step 0.01 × LMT 14-18 step 1
    #      Very-low STP: does STP<0.1 add more NP?  (3×11×14×5=2310)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_micro_stp"
    _c = _cfg(_n, (4,6,1), (45,55,1), (0.02,0.15,0.01), (14,18,1))
    log.info("A07  LE(4-6) × SE(45-55 s1) × STP(0.02-0.15 s0.01) × LMT(14-18)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  LE 5-7 × SE 80-150 step 5 × STP 0.1-0.5 step 0.1 × LMT 14-18 step 1
    #      High-SE (80-150) + low STP  (3×15×5×5=1125)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_high_se_range"
    _c = _cfg(_n, (5,7,1), (80,150,5), (0.1,0.5,0.1), (14,18,1))
    log.info("A08  LE(5-7) × SE(80-150 s5) × STP(0.1-0.5 s0.1) × LMT(14-18)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  LE 4-6 × SE 45-55 step 2 × STP 0.1-0.4 step 0.1 × LMT 10-22 step 0.5
    #      LMT scan 10-22 with low STP  (3×6×4×25=1800)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_lmt_scan"
    _c = _cfg(_n, (4,6,1), (45,55,2), (0.1,0.4,0.1), (10,22,0.5))
    log.info("A09  LE(4-6) × SE(45-55 s2) × STP(0.1-0.4 s0.1) × LMT(10-22 s0.5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  LE 1-4 step 1 × SE 40-65 step 5 × STP 0.05-0.4 step 0.05 × LMT 14-18 step 1
    #      LE=1-4 + low STP  (4×6×8×5=960)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_le_low_stp"
    _c = _cfg(_n, (1,4,1), (40,65,5), (0.05,0.4,0.05), (14,18,1))
    log.info("A10  LE(1-4) × SE(40-65 s5) × STP(0.05-0.4 s0.05) × LMT(14-18)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom around current best after A01-A10
    #      Progressive shrink from LE±1 × SE±5 × STP±0.1 × LMT±3
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt in [(5,0.1,3), (4,0.08,2.5), (3,0.06,2), (2,0.05,1.5)]:
        _le11  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se11  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, _r_stp, 0.01, STP_LO, STP_HI)
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
    # A12  Boundary: LE 1-15 step 2 × SE 20-200 step 20 × STP 0.1-0.5 step 0.1 × LMT 12-22 step 2
    #      Global boundary check  (8×10×5×6=2400)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (1,15,2), (20,200,20), (0.1,0.5,0.1), (12,22,2))
    log.info("A12  LE(1-15 s2) × SE(20-200 s20) × STP(0.1-0.5 s0.1) × LMT(12-22 s2)  %d combos", _c.total_runs())
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
    log.info("  FINAL RESULT — Round-7 Breakout Daily")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
             best_entry.get("LE", best_le), best_entry.get("SE", best_se),
             best_entry.get("STP", best_stp), best_entry.get("LMT", best_lmt),
             best_entry.get("net_profit", best_np),
             best_entry.get("max_drawdown", 0),
             best_entry.get("objective", best_obj),
             best_entry.get("total_trades", 0))
    log.info("  Target (NP>6M): %s", "MET ✓" if target_met else
             "NOT MET — best NP=%.0f" % best_np)
    log.info("══════════════════════════════════════════════════════════════")
    log.info("")
    log.info("  Per-attempt summary:")
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
# Entry point with auto-elevation
# ─────────────────────────────────────────────────────────────────────────────

def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _auto_elevate() -> None:
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
        print(f"[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Round-7 Breakout Daily search — target NP > 6M")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--attempt",  type=int, default=1, metavar="N")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0

    conn: Optional[mc.MultiChartsConnection] = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
