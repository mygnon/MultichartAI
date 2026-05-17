"""
search_daily_target6.py — Breakout Daily NP > 6,000,000, Round 6

Key findings from Round 5 (May 2026):
  - Column-mapping bug: MUST vary ALL 4 params per attempt (no fixed params)
    When any param is fixed (start=stop), MCReport packs multiple values per row
    and the export comes out garbled.
  - Current data gives max NP=3,779,400 for LE=5 SE=50 STP=2.2 LMT=17 (A06/A10)
  - This is 23% below Round-1's 4,908,600 — data was updated between runs
  - ZERO combos found with NP > 4M in 5022 tested combos (A06+A10)
  - Gap to 6M target: +59% from current best

Round-6 strategy:
  All attempts vary ALL 4 parameters (no exceptions).
  Explore territory not yet cleanly tested:
    H1. LMT 17-40 with varied SE — is there a higher NP plateau at LMT>20?
    H2. SE 80-250 — very selective long-lookback breakouts capture bigger moves
    H3. LE 1-4 — shorter entry lookback = more entries = higher total NP
    H4. Very low STP (0.25-1.0) + high LMT — high win rate with large targets
    H5. LMT 30-70 — outlier mega-move capture (few but very large wins)

Attempt schedule (12 attempts, each ≤ 5,000 combos):
  A01  LE 3-7 × SE 35-75 step 5 × STP 1-3 step 0.5 × LMT 17-35 step 1
       Core H1 — does NP rise again above LMT=20?             (5×9×5×19=4275)
  A02  LE 2-8 × SE 15-100 step 5 × STP 1-3 step 1 × LMT 14-20 step 1
       Wider SE with proven LMT range                          (7×18×3×7=2646)
  A03  LE 1-5 × SE 35-70 step 5 × STP 0.5-3 step 0.5 × LMT 15-25 step 1
       H3: LE=1-4 probe                                        (5×8×6×11=2640)
  A04  LE 4-6 × SE 35-65 step 5 × STP 0.5-2 step 0.25 × LMT 20-35 step 1
       H4: low STP with LMT 20-35                              (3×7×7×16=2352)
  A05  LE 4-6 × SE 50-120 step 10 × STP 1-3 step 0.5 × LMT 20-40 step 2
       H2+H1: high SE + high LMT                               (3×8×5×11=1320)
  A06  LE 4-6 × SE 45-60 step 1 × STP 1.5-3 step 0.25 × LMT 15-20 step 1
       Fine precision near current best                        (3×16×7×6=2016)
  A07  LE 1-4 × SE 30-80 step 5 × STP 1.5-3 step 0.5 × LMT 15-25 step 1
       H3: LE=1-4 with wider SE                                (4×11×4×11=1936)
  A08  LE 3-7 × SE 80-200 step 10 × STP 1-3 step 0.5 × LMT 15-25 step 1
       H2: ultra-high SE                                       (5×13×5×11=3575)
  A09  LE 3-7 × SE 30-70 step 5 × STP 0.25-1.5 step 0.25 × LMT 15-25 step 1
       H4: very low STP region                                 (5×9×6×11=2970)
  A10  LE 3-8 × SE 40-70 step 5 × STP 1-4 step 0.5 × LMT 30-60 step 5
       H5: ultra-high LMT                                      (6×7×7×7=2058)
  A11  LE 4-6 × SE 44-56 step 1 × STP 1.8-2.8 step 0.2 × LMT 15-22 step 0.5
       Dense precision at current best region                  (3×13×6×15=3510)
  A12  LE 1-12 step 2 × SE 10-200 step 10 × STP 1-3 step 1 × LMT 15-25 step 5
       Boundary sweep — find missed peaks                       (6×20×3×3=1080)

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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_target6_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 6_000_000.0
MAX_ATTEMPTS = 12

# Bounds (generous — actual per-attempt ranges are much tighter)
STP_LO, STP_HI = 0.1,  20.0
LMT_LO, LMT_HI = 2.0, 100.0
SE_LO,  SE_HI  = 1.0, 400.0
LE_LO,  LE_HI  = 1.0, 100.0

# Round-5 verified best (valid 4D data, current data as of 2026-05-16)
SEED_LE,  SEED_SE  = 5.0,  50.0
SEED_STP, SEED_LMT = 2.2,  17.0
SEED_NP   = 3_779_400.0
SEED_OBJ  = 18_608_474.0  # NP²/|MDD| = 3779400²/767600

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_daily_target6_{int(time.time())}.log"
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
    # Safety: ensure every param varies (start ≠ stop)
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
        name=f"BD6_{name}",
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
    return OUTPUT_DIR / f"BD6_{name}_raw.csv"


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
# Validate that loaded CSV has params within expected ranges
# ─────────────────────────────────────────────────────────────────────────────

def _validate_df(df: pd.DataFrame, cfg: StrategyConfig) -> bool:
    """Return True if param column ranges in df match the StrategyConfig spec."""
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
                        "Got min=%.4g max=%.4g. Skipping this attempt.",
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
    """Return (LE, SE, STP, LMT, objective, NP, MDD, trades, target_met)."""
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
    above = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    best_np = max(log_, key=lambda x: x.get("net_profit", 0), default={})
    payload = {
        "strategy":           "Breakout_Daily  (target NP>6M round-6)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "round5_best":        {
            "LE": SEED_LE, "SE": SEED_SE,
            "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "best_params":            best,
        "best_np_attempt":        best_np,
        "attempts_above_target":  above,
        "attempt_log":            log_,
    }
    out = OUTPUT_DIR / "final_params_daily_target6.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
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
    log.info("  Round-6 Breakout Daily NP>6M  (all-4-param variation enforced)")
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
                 "★TARGET★" if met else ("NP=%.0f/6M" % np_))
        log.info("       Global best so far: NP=%.0f (need +%.1f%%)",
                 best_np, (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else 0)

    # ──────────────────────────────────────────────────────────────────────
    # A01  LE 3-7 × SE 35-75 step 5 × STP 1-3 step 0.5 × LMT 17-35 step 1
    #      Does NP keep rising above LMT=20?  (5×9×5×19=4275)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_lmt_hi_profile"
    _c = _cfg(_n, (3,7,1), (35,75,5), (1.0,3.0,0.5), (17,35,1))
    log.info("A01  LE(3-7) × SE(35-75 s5) × STP(1-3 s0.5) × LMT(17-35)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  LE 2-8 × SE 15-100 step 5 × STP 1-3 step 1 × LMT 14-20 step 1
    #      Wider SE with proven LMT range  (7×18×3×7=2646)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_wide_se"
    _c = _cfg(_n, (2,8,1), (15,100,5), (1.0,3.0,1.0), (14,20,1))
    log.info("A02  LE(2-8) × SE(15-100 s5) × STP(1-3 s1) × LMT(14-20)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  LE 1-5 × SE 35-70 step 5 × STP 0.5-3 step 0.5 × LMT 15-25 step 1
    #      LE=1-4 with medium SE  (5×8×6×11=2640)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_le_low"
    _c = _cfg(_n, (1,5,1), (35,70,5), (0.5,3.0,0.5), (15,25,1))
    log.info("A03  LE(1-5) × SE(35-70 s5) × STP(0.5-3 s0.5) × LMT(15-25)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  LE 4-6 × SE 35-65 step 5 × STP 0.5-2 step 0.25 × LMT 20-35 step 1
    #      Low STP with mid-high LMT  (3×7×7×16=2352)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_low_stp_high_lmt"
    _c = _cfg(_n, (4,6,1), (35,65,5), (0.5,2.0,0.25), (20,35,1))
    log.info("A04  LE(4-6) × SE(35-65 s5) × STP(0.5-2 s0.25) × LMT(20-35)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  LE 4-6 × SE 50-130 step 10 × STP 1-3 step 0.5 × LMT 18-40 step 2
    #      High SE + high LMT  (3×9×5×12=1620)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_high_se_lmt"
    _c = _cfg(_n, (4,6,1), (50,130,10), (1.0,3.0,0.5), (18,40,2))
    log.info("A05  LE(4-6) × SE(50-130 s10) × STP(1-3 s0.5) × LMT(18-40 s2)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  Zoom around current best  LE±2 × SE±12 step 2 × STP±1 step 0.2 × LMT±4 step 0.5
    #      Fine neighbourhood  (~5 × 13 × 11 × 17 = 12155 → trim to safe)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_zoom_best"
    _le6  = zoom(best_le,  2,   1,   LE_LO,  LE_HI)
    _se6  = zoom(best_se,  12,  2,   SE_LO,  SE_HI)
    _stp6 = zoom(best_stp, 1.0, 0.2, STP_LO, STP_HI)
    _lmt6 = zoom(best_lmt, 4,   0.5, LMT_LO, LMT_HI)
    # Progressively shrink until ≤5000 combos (fixed radii to avoid infinite loop)
    for _r_se, _r_stp, _r_lmt in [(12,1.0,4), (8,0.8,3), (6,0.6,2), (4,0.4,1)]:
        _se6  = zoom(best_se,  _r_se,  2,   SE_LO,  SE_HI)
        _stp6 = zoom(best_stp, _r_stp, 0.2, STP_LO, STP_HI)
        _lmt6 = zoom(best_lmt, _r_lmt, 0.5, LMT_LO, LMT_HI)
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
    # A07  LE 1-4 × SE 30-80 step 5 × STP 1.5-3 step 0.5 × LMT 15-25 step 1
    #      LE=1-4 focus with wider SE  (4×11×4×11=1936)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_le_low_wide_se"
    _c = _cfg(_n, (1,4,1), (30,80,5), (1.5,3.0,0.5), (15,25,1))
    log.info("A07  LE(1-4) × SE(30-80 s5) × STP(1.5-3 s0.5) × LMT(15-25)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  LE 3-7 × SE 80-200 step 10 × STP 1-3 step 0.5 × LMT 15-25 step 1
    #      Ultra-high SE  (5×13×5×11=3575)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_ultra_se"
    _c = _cfg(_n, (3,7,1), (80,200,10), (1.0,3.0,0.5), (15,25,1))
    log.info("A08  LE(3-7) × SE(80-200 s10) × STP(1-3 s0.5) × LMT(15-25)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  LE 3-7 × SE 30-70 step 5 × STP 0.25-1.5 step 0.25 × LMT 15-25 step 1
    #      Very low STP region  (5×9×6×11=2970)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_very_low_stp"
    _c = _cfg(_n, (3,7,1), (30,70,5), (0.25,1.5,0.25), (15,25,1))
    log.info("A09  LE(3-7) × SE(30-70 s5) × STP(0.25-1.5 s0.25) × LMT(15-25)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  LE 3-8 × SE 40-70 step 5 × STP 1-4 step 0.5 × LMT 30-60 step 5
    #      Ultra-high LMT  (6×7×7×7=2058)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_ultra_lmt"
    _c = _cfg(_n, (3,8,1), (40,70,5), (1.0,4.0,0.5), (30,60,5))
    log.info("A10  LE(3-8) × SE(40-70 s5) × STP(1-4 s0.5) × LMT(30-60 s5)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom: LE±1 × SE±5 step 1 × STP±0.4 step 0.1 × LMT±3 step 0.5
    #      Precision around current champion  (~3 × 11 × 9 × 13 = 3861)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    _le11  = zoom(best_le,  1,   1,   LE_LO,  LE_HI)
    _se11  = zoom(best_se,  5,   1,   SE_LO,  SE_HI)
    _stp11 = zoom(best_stp, 0.4, 0.1, STP_LO, STP_HI)
    _lmt11 = zoom(best_lmt, 3,   0.5, LMT_LO, LMT_HI)
    # Progressively shrink SE radius until ≤5000 combos
    for _r_se11 in [5, 4, 3, 2]:
        _se11 = zoom(best_se, _r_se11, 1, SE_LO, SE_HI)
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
    # A12  Boundary: LE 1-12 step 2 × SE 10-200 step 10 × STP 1-3 step 1 × LMT 15-25 step 5
    #      Global boundary check  (6×20×3×3=1080)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (1,12,2), (10,200,10), (1.0,3.0,1.0), (15,25,5))
    log.info("A12  LE(1-12 s2) × SE(10-200 s10) × STP(1-3) × LMT(15-25 s5)  %d combos", _c.total_runs())
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
    log.info("  FINAL RESULT — Round-6 Breakout Daily")
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
        description="Round-6 Breakout Daily search — target NP > 6M")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--attempt", type=int, default=1, metavar="N")
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
