"""
search_daily_target4.py — Round 4: Breakout Daily NP > 6,000,000

Learnings from Rounds 1-3:
  Round-1 best (verified clean): LE=5, SE=50, STP=2.1, LMT=17
    → NP=4,908,600  MDD=-767,600  obj=31,389,205  trades=50
  Round-3 A06 (STP=18 no-stop regime):  LE=5, SE=140, STP=18, LMT=18.5
    → NP=5,436,600  MDD=-1,122,600  obj=26,328,719  trades=23
  LMT=17 confirmed optimal for STP≈2.1 regime (tested up to LMT=60)

Column-mapping bug (critical, recurring):
  When only 1 parameter varies, MCReport packs multiple values per row.
  FIX: every attempt must vary AT LEAST 2 parameters.

Round-4 focus:
  A01  SE×STP 2D  — core landscape (SE 30-200, STP 1.25-5.5): never cleanly mapped
  A02  SE×LMT 2D  — find LMT optimum for new (SE,STP) champion
  A03  LE×SE 2D   — explore LE=1-4 which has never been tested below 5
  A04  SE×STP fine — zoom around A01 champion
  A05  SE high (200-400) × STP — test very selective breakout regime
  A06  LE×LMT 2D  — optimise LMT for each LE level
  A07  4D wide    — broad neighbourhood of current best
  A08  STP low × SE — very tight stop territory (STP 0.25-1.5)
  A09  LE low × SE — LE 1-8 focus with wide SE
  A10  LE×SE boundary — wide coarse scan to spot missed peaks
  A11  4D micro   — precision refinement
  A12  Final dense 4D
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_target4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 6_000_000.0
MAX_ATTEMPTS = 12

STP_LO, STP_HI = 0.25, 20.0
LMT_LO, LMT_HI = 2.0,  100.0
SE_LO,  SE_HI  = 2.0,  500.0
LE_LO,  LE_HI  = 1.0,  100.0

# Verified clean seed from Round-1
SEED_LE,  SEED_SE  = 5.0,  50.0
SEED_STP, SEED_LMT = 2.1,  17.0
SEED_NP   = 4_908_600.0
SEED_OBJ  = 31_389_205.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_daily_target4_{int(time.time())}.log"
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


def fixed(v: float) -> Tuple[float, float, float]:
    return (v, v, 1.0)


def n_vals(t: Tuple[float, float, float]) -> int:
    start, stop, step = t
    return max(1, round((stop - start) / step) + 1)


def cap_step(t: Tuple[float, float, float], max_n: int) -> Tuple[float, float, float]:
    """Widen step so the axis fits within max_n values."""
    start, stop, step = t
    n = n_vals(t)
    if n <= max_n:
        return t
    new_step = (stop - start) / (max_n - 1)
    # round up to a 'nice' multiple of the original step
    mult = max(1, int(np.ceil(new_step / step)))
    return (start, stop, step * mult)


# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(name: str,
         le:  Tuple[float, float, float],
         se:  Tuple[float, float, float],
         stp: Tuple[float, float, float],
         lmt: Tuple[float, float, float]) -> StrategyConfig:
    return StrategyConfig(
        name=f"BDT4_{name}",
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


def _trim4d(le, se, stp, lmt, limit=5000) -> Tuple:
    """Progressively reduce 4D grid until total_runs ≤ limit."""
    for le2, se2, stp2, lmt2 in [
        (le, se, stp, lmt),
        (cap_step(le, 5),  cap_step(se, 10), stp, lmt),
        (cap_step(le, 5),  cap_step(se, 8),  cap_step(stp, 6), cap_step(lmt, 8)),
        (cap_step(le, 4),  cap_step(se, 7),  cap_step(stp, 5), cap_step(lmt, 7)),
        (cap_step(le, 4),  cap_step(se, 6),  cap_step(stp, 4), cap_step(lmt, 6)),
    ]:
        cfg = _cfg("_probe", le2, se2, stp2, lmt2)
        if cfg.total_runs() <= limit:
            return le2, se2, stp2, lmt2
    return cap_step(le, 3), cap_step(se, 5), cap_step(stp, 4), cap_step(lmt, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Run or load CSV  (one retry on failure)
# ─────────────────────────────────────────────────────────────────────────────

def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"BDT4_{name}_raw.csv"


def run_or_load(name: str, cfg: StrategyConfig,
                conn: Optional[mc.MultiChartsConnection],
                from_csv: bool,
                retry: bool = True) -> Optional[pd.DataFrame]:
    csv = csv_for(name)
    if from_csv or csv.exists():
        if csv.exists():
            try:
                df = mc.load_results_csv(str(csv), cfg)
                log.info("Loaded %s: %d rows from %s", name, len(df), csv.name)
                return df
            except Exception as e:
                log.warning("Could not load %s: %s", csv, e)
        else:
            log.warning("No CSV for %s at %s", name, csv)
        return None

    log.info("=== Attempt %-30s (%d combos) ===", name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done in %.1f min — %s", (time.time()-t0)/60, Path(raw_csv).name)
        return mc.load_results_csv(raw_csv, cfg)
    except Exception as e:
        log.error("  FAILED: %s", e)
        if retry and conn is not None:
            log.warning("  Retrying after 20s pause...")
            time.sleep(20)
            try:
                raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
                log.info("  Retry succeeded in %.1f min — %s",
                         (time.time()-t0)/60, Path(raw_csv).name)
                return mc.load_results_csv(raw_csv, cfg)
            except Exception as e2:
                log.error("  Retry also FAILED: %s", e2, exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Champion selector — argmax objective
# ─────────────────────────────────────────────────────────────────────────────

def champion(df: pd.DataFrame,
             fallback_le: float, fallback_se: float,
             fallback_stp: float, fallback_lmt: float
             ) -> Tuple[float, float, float, float, float, float, float, int]:
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    best_row = df.loc[df["Objective"].idxmax()]
    obj = float(best_row["Objective"])
    if obj <= 0:
        log.info("  No profitable row — keeping fallback params")
        return (fallback_le, fallback_se, fallback_stp, fallback_lmt,
                0.0, 0.0, 0.0, 0)
    le  = float(best_row["LE"])
    se  = float(best_row["SE"])
    stp = float(best_row["STP"])
    lmt = float(best_row["LMT"])
    np_ = float(best_row.get("NetProfit",   0))
    mdd = float(best_row.get("MaxDrawdown", 0))
    tr  = int(best_row.get("TotalTrades",  0))
    log.info("  Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
             le, se, stp, lmt, obj, np_, mdd, tr)
    return le, se, stp, lmt, obj, np_, mdd, tr


def merge_best(dfs: List[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["Objective"] = plateau_mod.compute_objective(combined)
    combined.sort_values("Objective", ascending=False, inplace=True)
    combined.drop_duplicates(subset=keys, keep="first", inplace=True)
    return combined.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_json(le, se, stp, lmt, obj, np_, mdd, trades,
              attempt_log: List[dict], target_met: bool) -> Path:
    best_attempt = max(attempt_log, key=lambda x: x.get("objective", 0), default=None)
    payload = {
        "strategy":           "Breakout_Daily  (target NP>6M round-4)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|  [NetProfit>0 AND MaxDrawdown<0]",
        "target_np":          TARGET_NP,
        "target_met":         target_met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "seed_round1_best": {
            "LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "best_params": {
            "LE": le, "SE": se, "STP": stp, "LMT": lmt,
            "net_profit":   round(np_, 0),
            "max_drawdown": round(mdd, 0),
            "objective":    round(obj, 0),
            "total_trades": trades,
        },
        "best_objective_attempt": best_attempt,
        "attempt_log": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_daily_target4.json"
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
    target_met = False
    stagnant   = 0
    attempt_log: List[dict] = []

    # accumulate dataframes for merged champion selection
    se_stp_dfs:  List[pd.DataFrame] = []
    le_se_dfs:   List[pd.DataFrame] = []
    stp_lmt_dfs: List[pd.DataFrame] = []

    log.info("Round-4 seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  OBJ=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np, best_obj)
    log.info("Rule: every attempt varies >=2 params (prevents column-mapping bug)")

    def _update(df, name, attempt_num, accumulator=None):
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, stagnant
        if df is None:
            attempt_log.append({"attempt": attempt_num, "name": name,
                                 "rows": 0, "LE": best_le, "SE": best_se,
                                 "STP": best_stp, "LMT": best_lmt,
                                 "net_profit": 0, "max_drawdown": 0,
                                 "objective": 0, "total_trades": 0})
            return
        if accumulator is not None:
            accumulator.append(df)
        le, se, stp, lmt, obj, np_, mdd, tr = champion(
            df, best_le, best_se, best_stp, best_lmt)
        improved = obj > best_obj * 1.05
        if obj > best_obj:
            best_le, best_se = le, se
            best_stp, best_lmt = stp, lmt
            best_np, best_obj = np_, obj
        if best_np >= TARGET_NP:
            target_met = True
        stagnant = 0 if improved else stagnant + 1
        attempt_log.append({
            "attempt": attempt_num, "name": name, "rows": len(df),
            "LE": le, "SE": se, "STP": stp, "LMT": lmt,
            "net_profit":   round(np_, 0),
            "max_drawdown": round(mdd, 0),
            "objective":    round(obj, 0),
            "total_trades": tr,
        })
        log.info("  [A%02d %-28s] LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  [%s stagnant=%d]",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "TARGET!" if best_np >= TARGET_NP else "no", stagnant)

    def _done(attempt_num: int) -> bool:
        if attempt_num > MAX_ATTEMPTS:
            return True
        if target_met and stagnant >= 2:
            return True
        return False

    A = 0

    # ─── A01: SE × STP 2D — primary unmapped surface ─────────────────────────
    # LE=5, LMT=17 fixed (both confirmed near-optimal)
    # SE: 30-200 step 10 = 18 values | STP: 1.25-5.5 step 0.25 = 18 values → 324 combos
    A += 1
    _name = "01_se_stp_2d"
    if start_attempt <= A and not _done(A):
        _se1  = (30.0,  200.0, 10.0)   # 18 values
        _stp1 = (1.25,  5.5,   0.25)   # 18 values  →  324 combos
        _c = _cfg(_name, fixed(SEED_LE), _se1, _stp1, fixed(SEED_LMT))
        log.info("A01 SE×STP 2D — LE=%.4g LMT=%.4g, SE %s STP %s (%d combos)",
                 SEED_LE, SEED_LMT, _se1, _stp1, _c.total_runs())
        df = run_or_load(_name, _c, conn, from_csv, retry=True)
        _update(df, _name, A, se_stp_dfs)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A02: SE × LMT 2D (with STP champion from A01) ───────────────────────
    # Finds optimal LMT for the new (SE, STP) configuration
    # SE: 20-180 step 10 = 17 values | LMT: 5-40 step 2 = 18 values → 306 combos
    A += 1
    _name = "02_se_lmt_2d"
    if start_attempt <= A and not _done(A):
        _se2  = (20.0, 180.0, 10.0)    # 17 values
        _lmt2 = (5.0,  40.0,   2.0)    # 18 values  →  306 combos
        _c = _cfg(_name, fixed(best_le), _se2, fixed(best_stp), _lmt2)
        log.info("A02 SE×LMT 2D — LE=%.4g STP=%.4g, SE %s LMT %s (%d combos)",
                 best_le, best_stp, _se2, _lmt2, _c.total_runs())
        df = run_or_load(_name, _c, conn, from_csv)
        if df is not None:
            stp_lmt_dfs.append(df)
            se_stp_dfs.append(df)
            m = merge_best(stp_lmt_dfs, ["SE", "LMT"])
            _, best_se_m, _, best_lmt_m, obj_m, np_m, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj_m > best_obj:
                best_se, best_lmt = best_se_m, best_lmt_m
                best_obj, best_np = obj_m, np_m
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A03: LE × SE 2D (includes LE=1-4, never tested before) ─────────────
    # LE: 1-20 step 1 = 20 values | SE: 20-200 step 10 = 19 values → 380 combos
    A += 1
    _name = "03_le_se_2d"
    if start_attempt <= A and not _done(A):
        _le3 = (1.0,  20.0,  1.0)      # 20 values  — includes LE=1-4
        _se3 = (20.0, 200.0, 10.0)     # 19 values  →  380 combos
        _c = _cfg(_name, _le3, _se3, fixed(best_stp), fixed(best_lmt))
        log.info("A03 LE×SE 2D — STP=%.4g LMT=%.4g, LE %s SE %s (%d combos)",
                 best_stp, best_lmt, _le3, _se3, _c.total_runs())
        df = run_or_load(_name, _c, conn, from_csv)
        if df is not None:
            le_se_dfs.append(df)
            m = merge_best(le_se_dfs, ["LE", "SE"])
            best_le_m, best_se_m, _, _, obj_m, np_m, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj_m > best_obj:
                best_le, best_se = best_le_m, best_se_m
                best_obj, best_np = obj_m, np_m
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df, _name, A, le_se_dfs)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A04: SE × STP fine zoom (around current champion) ───────────────────
    # SE: best±20 step 2 | STP: best±1 step 0.1 → ~21×21=441 combos
    A += 1
    _name = "04_se_stp_fine"
    if start_attempt <= A and not _done(A):
        _se4  = zoom(best_se,  20, 2,   SE_LO,  SE_HI)
        _stp4 = zoom(best_stp,  1, 0.1, STP_LO, STP_HI)
        _c = _cfg(_name, fixed(best_le), _se4, _stp4, fixed(best_lmt))
        combos = _c.total_runs()
        if combos > 5000:
            _se4  = zoom(best_se,  15, 2,   SE_LO, SE_HI)
            _stp4 = zoom(best_stp,  1, 0.1, STP_LO, STP_HI)
            _c = _cfg(_name, fixed(best_le), _se4, _stp4, fixed(best_lmt))
        log.info("A04 SE×STP fine — LE=%.4g LMT=%.4g, SE %s STP %s (%d combos)",
                 best_le, best_lmt, _se4, _stp4, _c.total_runs())
        df = run_or_load(_name, _c, conn, from_csv)
        _update(df, _name, A, se_stp_dfs)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A05: SE high territory (SE 200-400) × STP ───────────────────────────
    # Very selective breakout: few trades but potentially high quality
    # SE: 200-400 step 10 = 21 values | STP: 1.5-4.0 step 0.5 = 6 values → 126 combos
    A += 1
    _name = "05_se_high_stp"
    if start_attempt <= A and not _done(A):
        _se5  = (200.0, 400.0, 10.0)   # 21 values
        _stp5 = (1.5,    4.0,  0.5)    # 6 values  →  126 combos
        _c = _cfg(_name, fixed(best_le), _se5, _stp5, fixed(best_lmt))
        log.info("A05 SE-high × STP — LE=%.4g LMT=%.4g, SE %s STP %s (%d combos)",
                 best_le, best_lmt, _se5, _stp5, _c.total_runs())
        df = run_or_load(_name, _c, conn, from_csv)
        if df is not None:
            se_stp_dfs.append(df)
            m = merge_best(se_stp_dfs, ["SE", "STP"])
            _, best_se_m, best_stp_m, _, obj_m, np_m, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj_m > best_obj:
                best_se, best_stp = best_se_m, best_stp_m
                best_obj, best_np = obj_m, np_m
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A06: LE × LMT 2D (optimise LMT for each LE level) ──────────────────
    # LE: 1-20 step 1 = 20 values | LMT: best±15 step 1 = ≤31 values → ≤620 combos
    A += 1
    _name = "06_le_lmt_2d"
    if start_attempt <= A and not _done(A):
        _le6  = (1.0, 20.0, 1.0)                                  # 20 values
        _lmt6 = zoom(best_lmt, 15, 1, LMT_LO, LMT_HI)            # ≤31 values
        _c = _cfg(_name, _le6, fixed(best_se), fixed(best_stp), _lmt6)
        combos = _c.total_runs()
        if combos > 5000:
            _lmt6 = zoom(best_lmt, 10, 1, LMT_LO, LMT_HI)
            _c = _cfg(_name, _le6, fixed(best_se), fixed(best_stp), _lmt6)
        log.info("A06 LE×LMT 2D — SE=%.4g STP=%.4g, LE %s LMT %s (%d combos)",
                 best_se, best_stp, _le6, _lmt6, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A07: 4D wide grid around best ───────────────────────────────────────
    A += 1
    _name = "07_4d_wide"
    if start_attempt <= A and not _done(A):
        _le7  = zoom(best_le,   4,   1,    LE_LO,  LE_HI)
        _se7  = zoom(best_se,  25,   5,    SE_LO,  SE_HI)
        _stp7 = zoom(best_stp,  1,   0.25, STP_LO, STP_HI)
        _lmt7 = zoom(best_lmt,  8,   1,    LMT_LO, LMT_HI)
        le7, se7, stp7, lmt7 = _trim4d(_le7, _se7, _stp7, _lmt7)
        _c = _cfg(_name, le7, se7, stp7, lmt7)
        log.info("A07 4D wide — LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 le7, se7, stp7, lmt7, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A08: STP low × SE (very tight stop territory) ───────────────────────
    # STP 0.25-1.5 with wide SE — maybe tight stop + selective entry works
    # SE: 30-150 step 10 = 13 values | STP: 0.25-1.5 step 0.125 = 11 values → 143 combos
    A += 1
    _name = "08_stp_low_se"
    if start_attempt <= A and not _done(A):
        _se8  = (30.0, 150.0,  10.0)   # 13 values
        _stp8 = (0.25,   1.5,  0.125)  # 11 values  →  143 combos
        _c = _cfg(_name, fixed(best_le), _se8, _stp8, fixed(best_lmt))
        log.info("A08 STP-low × SE — LE=%.4g LMT=%.4g, SE %s STP %s (%d combos)",
                 best_le, best_lmt, _se8, _stp8, _c.total_runs())
        df = run_or_load(_name, _c, conn, from_csv)
        _update(df, _name, A, se_stp_dfs)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A09: LE low (1-8) × SE wide — unexplored LE territory ──────────────
    # LE: 1-8 step 1 = 8 values | SE: 30-200 step 10 = 18 values → 144 combos
    A += 1
    _name = "09_le_low_se"
    if start_attempt <= A and not _done(A):
        _le9 = (1.0,   8.0,  1.0)      # 8 values
        _se9 = (30.0, 200.0, 10.0)     # 18 values  →  144 combos
        _c = _cfg(_name, _le9, _se9, fixed(best_stp), fixed(best_lmt))
        log.info("A09 LE-low × SE — STP=%.4g LMT=%.4g, LE %s SE %s (%d combos)",
                 best_stp, best_lmt, _le9, _se9, _c.total_runs())
        df = run_or_load(_name, _c, conn, from_csv)
        if df is not None:
            le_se_dfs.append(df)
            m = merge_best(le_se_dfs, ["LE", "SE"])
            best_le_m, best_se_m, _, _, obj_m, np_m, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj_m > best_obj:
                best_le, best_se = best_le_m, best_se_m
                best_obj, best_np = obj_m, np_m
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A10: Wide LE × SE boundary — coarse global scan ─────────────────────
    # LE: 1-60 step 4 = 15 values | SE: 10-300 step 15 = 20 values → 300 combos
    A += 1
    _name = "10_le_se_boundary"
    if start_attempt <= A and not _done(A):
        _le10 = (1.0,  61.0,  4.0)     # 15 values (includes LE=1)
        _se10 = (10.0, 305.0, 15.0)    # 20 values  →  300 combos
        _c = _cfg(_name, _le10, _se10, fixed(best_stp), fixed(best_lmt))
        log.info("A10 LE×SE boundary — STP=%.4g LMT=%.4g, LE %s SE %s (%d combos)",
                 best_stp, best_lmt, _le10, _se10, _c.total_runs())
        df = run_or_load(_name, _c, conn, from_csv)
        if df is not None:
            le_se_dfs.append(df)
            m = merge_best(le_se_dfs, ["LE", "SE"])
            best_le_m, best_se_m, _, _, obj_m, np_m, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj_m > best_obj:
                best_le, best_se = best_le_m, best_se_m
                best_obj, best_np = obj_m, np_m
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A11: 4D micro precision ─────────────────────────────────────────────
    A += 1
    _name = "11_4d_micro"
    if start_attempt <= A and not _done(A):
        _le11  = zoom(best_le,   2,   1,    LE_LO,  LE_HI)
        _se11  = zoom(best_se,  10,   2,    SE_LO,  SE_HI)
        _stp11 = zoom(best_stp,  0.5, 0.1,  STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt,  4,   0.5,  LMT_LO, LMT_HI)
        le11, se11, stp11, lmt11 = _trim4d(_le11, _se11, _stp11, _lmt11)
        _c = _cfg(_name, le11, se11, stp11, lmt11)
        log.info("A11 4D micro — LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 le11, se11, stp11, lmt11, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── A12: Final dense 4D ─────────────────────────────────────────────────
    A += 1
    _name = "12_4d_dense"
    if start_attempt <= A and not _done(A):
        _le12  = zoom(best_le,   3,   1,    LE_LO,  LE_HI)
        _se12  = zoom(best_se,  15,   3,    SE_LO,  SE_HI)
        _stp12 = zoom(best_stp,  0.6, 0.1,  STP_LO, STP_HI)
        _lmt12 = zoom(best_lmt,  5,   1,    LMT_LO, LMT_HI)
        le12, se12, stp12, lmt12 = _trim4d(_le12, _se12, _stp12, _lmt12)
        _c = _cfg(_name, le12, se12, stp12, lmt12)
        log.info("A12 4D dense — LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 le12, se12, stp12, lmt12, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np, best_obj)

    # ─── Final summary ────────────────────────────────────────────────────────
    best_entry = max(attempt_log, key=lambda x: x.get("objective", 0), default={})
    log.info("")
    log.info("══════════════════════════════════════════════════════")
    log.info("  FINAL (Round-4, Breakout Daily NP>6M search)")
    log.info("══════════════════════════════════════════════════════")
    log.info("  LE=%.4g  SE=%.4g  STP=%.4g  LMT=%.4g",
             best_le, best_se, best_stp, best_lmt)
    log.info("  NetProfit=%.0f  MaxDD=%.0f",
             best_np, best_entry.get("max_drawdown", 0))
    log.info("  Objective=%.0f  TargetMet=%s", best_obj,
             "YES ✓" if target_met else "NO — best NP was %.0f" % best_np)

    out = save_json(
        best_entry.get("LE",  best_le),  best_entry.get("SE",  best_se),
        best_entry.get("STP", best_stp), best_entry.get("LMT", best_lmt),
        best_entry.get("objective",    best_obj),
        best_entry.get("net_profit",   best_np),
        best_entry.get("max_drawdown", 0),
        best_entry.get("total_trades", 0),
        attempt_log, target_met,
    )
    print(f"\nDone — results at: {out}")
    print(f"Target (NP>6M): {'MET ✓' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Round-4 Breakout Daily target search — NP > 6M, SE×STP surface")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyze existing CSVs without MC automation")
    ap.add_argument("--attempt", type=int, default=1, metavar="N",
                    help="Start from attempt N (1-12)")
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        log.warning("Not running as Administrator — will attempt anyway (MC64 may run at same level)")


    conn: Optional[mc.MultiChartsConnection] = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
