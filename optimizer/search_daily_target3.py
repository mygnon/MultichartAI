"""
search_daily_target3.py — Round 3: Breakout Daily NP > 6,000,000

Key finding from Rounds 1 & 2:
  - Round-1 best: LE=5, SE=50, STP=2.1, LMT=17, NP=4,908,600, obj=31,389,205
  - Round-2 A02 partial decode: SE=125 gives NP=5,005,800 but obj only ~12M (MDD too large)
    → SE=50 is still optimal for objective; higher SE lowers objective despite higher NP
  - CRITICAL GAP: LMT>17 was NEVER properly tested (both Round-1 A01 and Round-2 A01 failed).
    This is the single most important missing data point.

Round-3 focus:
  H1. LMT 17-60 step 0.5 with LE=5, SE=50, STP=2.1 — primary push for NP>6M
      If LMT=30 gives NP≈5.5M with similar MDD, obj would be 5.5M²/767K≈39.4M
  H2. SE 50-160 step 5 (finer than Round-2 step=25): find if SE=55-120 improves on SE=50
  H3. STP fine around 2.1 with expanded LMT: tighter stop may help with high LMT
  H4. 2D SE×LMT and STP×LMT surfaces with correct fixed params
  H5. Full 4D grid in expanded LMT space

Retry logic: if an attempt fails (dialog error), wait 15s and retry once.
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_target3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 6_000_000.0
MAX_ATTEMPTS = 12

STP_LO, STP_HI = 0.25, 8.0
LMT_LO, LMT_HI = 2.0,  80.0
SE_LO,  SE_HI  = 2.0,  300.0
LE_LO,  LE_HI  = 1.0,  100.0

# Seed from Round-1 best
SEED_LE,  SEED_SE  = 5.0, 50.0
SEED_STP, SEED_LMT = 2.1, 17.0
SEED_NP   = 4_908_600.0
SEED_OBJ  = 31_389_205.0

_LOG_FILE = OUTPUT_DIR / f"search_daily_target3_{int(time.time())}.log"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(name: str,
         le:  Tuple[float, float, float],
         se:  Tuple[float, float, float],
         stp: Tuple[float, float, float],
         lmt: Tuple[float, float, float]) -> StrategyConfig:
    return StrategyConfig(
        name=f"BDT3_{name}",
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
# Run or load CSV  (with one retry on failure)
# ─────────────────────────────────────────────────────────────────────────────

def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"BDT3_{name}_raw.csv"


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
        "strategy":           "Breakout_Daily  (target NP>6M round-3)",
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
    out = OUTPUT_DIR / "final_params_daily_target3.json"
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
    stp_lmt_dfs: List[pd.DataFrame] = []
    le_se_dfs:   List[pd.DataFrame] = []

    log.info("Round-3 seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  OBJ=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np, best_obj)
    log.info("Focus: LMT probe up to %.0f, SE probe up to %.0f", LMT_HI, SE_HI)

    def _update(df, name, attempt_num):
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, stagnant
        if df is None:
            attempt_log.append({"attempt": attempt_num, "name": name,
                                 "rows": 0, "LE": best_le, "SE": best_se,
                                 "STP": best_stp, "LMT": best_lmt,
                                 "net_profit": 0, "max_drawdown": 0,
                                 "objective": 0, "total_trades": 0})
            return
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

    # ─── A01: PRIMARY — LMT high probe (CRITICAL MISSING DATA POINT) ──────────
    # LE=5, SE=50, STP=2.1, LMT 17-60 step 0.5  →  87 combos
    # This has FAILED in both Round-1 and Round-2. Retry enabled.
    A += 1
    _name = "01_lmt_high"
    if start_attempt <= A and not _done(A):
        _lmt1 = (17.0, 60.0, 0.5)              # 87 combos
        _c = _cfg(_name, fixed(SEED_LE), fixed(SEED_SE), fixed(SEED_STP), _lmt1)
        log.info("A01 PRIMARY LMT probe — LE=%.4g SE=%.4g STP=%.4g, LMT %s (%d combos) [retry ON]",
                 SEED_LE, SEED_SE, SEED_STP, _lmt1, _c.total_runs())
        df1 = run_or_load(_name, _c, conn, from_csv, retry=True)
        if df1 is not None:
            stp_lmt_dfs.append(df1)
        _update(df1, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A02: SE fine probe (step=5 vs Round-2's step=25) ────────────────────
    # LE=5, STP=2.1, LMT=best, SE 30-155 step 5  →  26 combos
    A += 1
    _name = "02_se_fine"
    if start_attempt <= A and not _done(A):
        _se2 = (30.0, 155.0, 5.0)              # 26 combos
        _c = _cfg(_name, fixed(best_le), _se2, fixed(best_stp), fixed(best_lmt))
        log.info("A02 SE fine probe — LE=%.4g STP=%.4g LMT=%.4g, SE %s (%d combos)",
                 best_le, best_stp, best_lmt, _se2, _c.total_runs())
        df2 = run_or_load(_name, _c, conn, from_csv)
        if df2 is not None:
            le_se_dfs.append(df2)
        _update(df2, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A03: STP probe with updated LMT ─────────────────────────────────────
    # LE=best, SE=best, LMT=best, STP 0.25-4.0 step 0.25  →  16 combos
    A += 1
    _name = "03_stp_probe"
    if start_attempt <= A and not _done(A):
        _stp3 = (0.25, 4.0, 0.25)              # 16 combos
        _c = _cfg(_name, fixed(best_le), fixed(best_se), _stp3, fixed(best_lmt))
        log.info("A03 STP probe — LE=%.4g SE=%.4g LMT=%.4g, STP %s (%d combos)",
                 best_le, best_se, best_lmt, _stp3, _c.total_runs())
        df3 = run_or_load(_name, _c, conn, from_csv)
        if df3 is not None:
            stp_lmt_dfs.append(df3)
        _update(df3, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A04: SE×LMT 2D surface (correct params) ─────────────────────────────
    # LE=best, STP=best, SE 20-160 step 10, LMT 10-65 step 5  →  15×12=180 combos
    A += 1
    _name = "04_se_lmt_2d"
    if start_attempt <= A and not _done(A):
        _se4  = (20.0, 160.0, 10.0)            # 15 values
        _lmt4 = (10.0, 65.0,  5.0)             # 12 values → 180 combos
        _c = _cfg(_name, fixed(best_le), _se4, fixed(best_stp), _lmt4)
        log.info("A04 SE×LMT 2D — LE=%.4g STP=%.4g, SE %s LMT %s (%d combos)",
                 best_le, best_stp, _se4, _lmt4, _c.total_runs())
        df4 = run_or_load(_name, _c, conn, from_csv)
        if df4 is not None:
            stp_lmt_dfs.append(df4)
            le_se_dfs.append(df4)
            m = merge_best(stp_lmt_dfs, ["SE","LMT"])
            _, best_se_tmp, _, best_lmt_tmp, obj4, np4, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj4 > best_obj:
                best_se, best_lmt = best_se_tmp, best_lmt_tmp
                best_obj, best_np = obj4, np4
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df4, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A05: STP×LMT 2D with expanded LMT range ────────────────────────────
    # LE=best, SE=best, STP 0.5-5 step 0.5, LMT 10-65 step 3  →  10×19=190 combos
    A += 1
    _name = "05_stp_lmt_2d"
    if start_attempt <= A and not _done(A):
        _stp5 = (0.5, 5.0, 0.5)               # 10 values
        _lmt5 = (10.0, 67.0, 3.0)             # 20 values → 200 combos
        _c = _cfg(_name, fixed(best_le), fixed(best_se), _stp5, _lmt5)
        log.info("A05 STP×LMT 2D — LE=%.4g SE=%.4g, STP %s LMT %s (%d combos)",
                 best_le, best_se, _stp5, _lmt5, _c.total_runs())
        df5 = run_or_load(_name, _c, conn, from_csv)
        if df5 is not None:
            stp_lmt_dfs.append(df5)
            m = merge_best(stp_lmt_dfs, ["STP","LMT"])
            _, _, best_stp_tmp, best_lmt_tmp, obj5, np5, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj5 > best_obj:
                best_stp, best_lmt = best_stp_tmp, best_lmt_tmp
                best_obj, best_np = obj5, np5
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df5, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A06: LE×SE 2D with updated STP/LMT ─────────────────────────────────
    # STP=best, LMT=best, LE 1-20 step 1, SE 20-160 step 8  →  20×19=380 combos
    A += 1
    _name = "06_le_se_2d"
    if start_attempt <= A and not _done(A):
        _le6 = (1.0, 20.0, 1.0)               # 20 values
        _se6 = (20.0, 162.0, 8.0)             # 19 values → 380 combos
        _c = _cfg(_name, _le6, _se6, fixed(best_stp), fixed(best_lmt))
        log.info("A06 LE×SE 2D — STP=%.4g LMT=%.4g, LE %s SE %s (%d combos)",
                 best_stp, best_lmt, _le6, _se6, _c.total_runs())
        df6 = run_or_load(_name, _c, conn, from_csv)
        if df6 is not None:
            le_se_dfs.append(df6)
            m = merge_best(le_se_dfs, ["LE","SE"])
            best_le_tmp, best_se_tmp, _, _, obj6, np6, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj6 > best_obj:
                best_le, best_se = best_le_tmp, best_se_tmp
                best_obj, best_np = obj6, np6
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df6, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A07: LMT fine zoom around new best ──────────────────────────────────
    # Zoom LMT with step 0.25 around current best
    A += 1
    _name = "07_lmt_fine"
    if start_attempt <= A and not _done(A):
        _lmt7 = zoom(best_lmt, 8, 0.25, LMT_LO, LMT_HI)  # ≤65 values
        _c = _cfg(_name, fixed(best_le), fixed(best_se), fixed(best_stp), _lmt7)
        log.info("A07 LMT fine — LE=%.4g SE=%.4g STP=%.4g, LMT %s (%d combos)",
                 best_le, best_se, best_stp, _lmt7, _c.total_runs())
        df7 = run_or_load(_name, _c, conn, from_csv)
        if df7 is not None:
            stp_lmt_dfs.append(df7)
        _update(df7, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A08: 4D grid around best ────────────────────────────────────────────
    # LE ±3 step 1, SE ±15 step 3, STP ±0.5 step 0.25, LMT ±6 step 1
    A += 1
    _name = "08_4d_grid"
    if start_attempt <= A and not _done(A):
        _le8  = zoom(best_le,  3,   1,    LE_LO,  LE_HI)
        _se8  = zoom(best_se,  15,  3,    SE_LO,  SE_HI)
        _stp8 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)
        _lmt8 = zoom(best_lmt, 6,   1,    LMT_LO, LMT_HI)
        _c = _cfg(_name, _le8, _se8, _stp8, _lmt8)
        combos = _c.total_runs()
        if combos > 5000:
            _le8  = zoom(best_le,  2,   1,    LE_LO,  LE_HI)
            _se8  = zoom(best_se,  10,  2,    SE_LO,  SE_HI)
            _stp8 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)
            _lmt8 = zoom(best_lmt, 5,   1,    LMT_LO, LMT_HI)
            _c = _cfg(_name, _le8, _se8, _stp8, _lmt8)
        log.info("A08 4D grid — LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le8, _se8, _stp8, _lmt8, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A09: LMT extended (if still below target) ───────────────────────────
    # Push LMT even higher: from best+5 to 80
    A += 1
    _name = "09_lmt_extend"
    if start_attempt <= A and not _done(A):
        _lmt9_lo = max(LMT_LO, best_lmt + 5.0)
        _lmt9 = (_lmt9_lo, 80.0, 1.0)         # varies
        _se9  = zoom(best_se, 10, 5, SE_LO, SE_HI)  # ≤5 values
        _c = _cfg(_name, fixed(best_le), _se9, fixed(best_stp), _lmt9)
        combos = _c.total_runs()
        if combos > 5000:
            _lmt9 = (_lmt9_lo, 80.0, 2.0)
            _c = _cfg(_name, fixed(best_le), _se9, fixed(best_stp), _lmt9)
        log.info("A09 LMT extend — LE=%.4g SE=%s STP=%.4g, LMT %s (%d combos)",
                 best_le, _se9, best_stp, _lmt9, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A10: Wide LE×SE boundary check ──────────────────────────────────────
    # LE 1-60 step 4, SE 10-200 step 10  →  15×20=300 combos
    A += 1
    _name = "10_le_se_boundary"
    if start_attempt <= A and not _done(A):
        _le10 = (1.0, 61.0, 4.0)              # 16 values
        _se10 = (10.0, 200.0, 10.0)           # 20 values → 320 combos
        _c = _cfg(_name, _le10, _se10, fixed(best_stp), fixed(best_lmt))
        log.info("A10 LE×SE boundary — STP=%.4g LMT=%.4g, LE %s SE %s (%d combos)",
                 best_stp, best_lmt, _le10, _se10, _c.total_runs())
        df10 = run_or_load(_name, _c, conn, from_csv)
        if df10 is not None:
            le_se_dfs.append(df10)
            m = merge_best(le_se_dfs, ["LE","SE"])
            best_le_tmp, best_se_tmp, _, _, obj10, np10, _, _ = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj10 > best_obj:
                best_le, best_se = best_le_tmp, best_se_tmp
                best_obj, best_np = obj10, np10
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df10, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A11: 4D micro precision ──────────────────────────────────────────────
    # LE ±2 step 1, SE ±8 step 2, STP ±0.4 step 0.1, LMT ±4 step 0.5
    A += 1
    _name = "11_4d_micro"
    if start_attempt <= A and not _done(A):
        _le11  = zoom(best_le,  2,   1,    LE_LO,  LE_HI)
        _se11  = zoom(best_se,  8,   2,    SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, 0.4, 0.1,  STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, 4,   0.5,  LMT_LO, LMT_HI)
        _c = _cfg(_name, _le11, _se11, _stp11, _lmt11)
        combos = _c.total_runs()
        if combos > 5000:
            _stp11 = zoom(best_stp, 0.3, 0.1, STP_LO, STP_HI)
            _lmt11 = zoom(best_lmt, 3,   0.5, LMT_LO, LMT_HI)
            _c = _cfg(_name, _le11, _se11, _stp11, _lmt11)
        log.info("A11 4D micro — LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le11, _se11, _stp11, _lmt11, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── A12: Final dense 4D ─────────────────────────────────────────────────
    # LE ±3 step 1, SE ±12 step 2, STP ±0.5 step 0.25, LMT ±5 step 1
    A += 1
    _name = "12_4d_dense"
    if start_attempt <= A and not _done(A):
        _le12  = zoom(best_le,  3,   1,    LE_LO,  LE_HI)
        _se12  = zoom(best_se,  12,  2,    SE_LO,  SE_HI)
        _stp12 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)
        _lmt12 = zoom(best_lmt, 5,   1,    LMT_LO, LMT_HI)
        _c = _cfg(_name, _le12, _se12, _stp12, _lmt12)
        combos = _c.total_runs()
        if combos > 5000:
            _le12  = zoom(best_le,  2,   1,    LE_LO,  LE_HI)
            _se12  = zoom(best_se,  8,   2,    SE_LO,  SE_HI)
            _stp12 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)
            _lmt12 = zoom(best_lmt, 4,   1,    LMT_LO, LMT_HI)
            _c = _cfg(_name, _le12, _se12, _stp12, _lmt12)
        log.info("A12 4D dense — LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le12, _se12, _stp12, _lmt12, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Final summary ────────────────────────────────────────────────────────
    best_entry = max(attempt_log, key=lambda x: x.get("objective", 0), default={})
    log.info("")
    log.info("══════════════════════════════════════════════════════")
    log.info("  FINAL (Round-3, Breakout Daily NP>6M search)")
    log.info("══════════════════════════════════════════════════════")
    log.info("  LE=%.4g  SE=%.4g  STP=%.4g  LMT=%.4g",
             best_le, best_se, best_stp, best_lmt)
    log.info("  NetProfit=%.0f  MaxDD=%.0f",
             best_np, best_entry.get("max_drawdown", 0))
    log.info("  Objective=%.0f  TargetMet=%s", best_obj,
             "YES ✓" if target_met else "NO — best NP was %.0f" % best_np)

    out = save_json(
        best_entry.get("LE", best_le), best_entry.get("SE", best_se),
        best_entry.get("STP", best_stp), best_entry.get("LMT", best_lmt),
        best_entry.get("objective", best_obj),
        best_entry.get("net_profit", best_np),
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
        description="Round-3 Breakout Daily target search — NP > 6M, LMT expanded")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyze existing CSVs without MC automation")
    ap.add_argument("--attempt", type=int, default=1, metavar="N",
                    help="Start from attempt N (1-12)")
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        print("[ERROR] Must run as Administrator to automate MultiCharts64.")
        print("        Please relaunch your terminal as Administrator.")
        return 1

    conn: Optional[mc.MultiChartsConnection] = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
