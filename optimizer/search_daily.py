"""
Adaptive 10-attempt Breakout Daily parameter search.

Each attempt uses the PREVIOUS attempt's plateau champion as its center,
alternating between zooming in (finer step) and expanding out (wider radius)
to converge on the best stable plateau.

Attempt schedule
────────────────
Phase 1 — LE × SE discovery  (STP/LMT held at defaults)
  01  Wide initial scan      LE/SE 2-80 step 4    (~400 combos)
  02  Zoom-in                LE/SE ±12 step 2     (~169 combos)
  03  Expand-out boundary    LE/SE ±24 step 3     (~289 combos)
  04  Fine zoom              LE/SE ±8 step 1      (~289 combos)

Phase 2 — STP × LMT discovery  (LE/SE fixed at phase-1 best)
  05  Wide STP×LMT           STP 0.5-6 ×0.5 / LMT 2-20 ×1  (11×19=209 combos)
  06  Zoom-in                STP ±2.0 ×0.25 / LMT ±5 ×0.5  (~varies ≤300)
  07  Fine zoom              STP ±0.75 ×0.1 / LMT ±3 ×0.25  (~varies ≤300)

Phase 3 — Cross-validate + 4D
  08  Re-verify LE×SE with refined STP/LMT  (±8 step 1)
  09  Wide boundary check    LE/SE ±30 step 4
  10  4D grid                LE ±3 step 1 × SE ±3 step 1 × STP ±0.5 step 0.25 × LMT ±2 step 1

Usage
─────
  python search_daily.py
  python search_daily.py --attempt 5   # resume from attempt 5
  python search_daily.py --from-csv    # re-analyze existing CSVs only
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
from scipy.ndimage import minimum_filter

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mc_automation as mc
import plateau as plateau_mod
from config import DateRange, ParamAxis, StrategyConfig, PLATEAU_NEIGHBORHOOD_RADIUS


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp"
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

DEFAULT_STP = 1.5
DEFAULT_LMT = 6.0

LE_LO, LE_HI   = 2, 100    # start from 2 to avoid LE=1 boundary trap
SE_LO, SE_HI   = 2, 100
STP_LO, STP_HI = 0.5, 8.0  # start from 0.5 to avoid STP boundary trap
LMT_LO, LMT_HI = 2.0, 24.0 # start from 2.0 to avoid LMT boundary trap

_LOG_FILE = OUTPUT_DIR / f"search_daily_{int(time.time())}.log"
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


# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(name: str,
         le:  Tuple[float, float, float],
         se:  Tuple[float, float, float],
         stp: Tuple[float, float, float],
         lmt: Tuple[float, float, float]) -> StrategyConfig:
    return StrategyConfig(
        name=f"BD_{name}",
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
    return OUTPUT_DIR / f"BD_{name}_raw.csv"


def run_or_load(name: str, cfg: StrategyConfig,
                conn: Optional[mc.MultiChartsConnection],
                from_csv: bool) -> Optional[pd.DataFrame]:
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

    log.info("=== %s  (%d combos) ===", name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done in %.1f min — %s", (time.time()-t0)/60, Path(raw_csv).name)
        return mc.load_results_csv(raw_csv, cfg)
    except Exception as e:
        log.error("  FAILED: %s", e, exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Plateau helpers — return (best_val1, best_val2, plateau_score)
# ─────────────────────────────────────────────────────────────────────────────

def best_le_se(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Return (LE, SE, plateau_score) of the plateau champion."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    le_vals = np.sort(df["LE"].unique())
    se_vals = np.sort(df["SE"].unique())
    grid = np.full((len(le_vals), len(se_vals)), 0.0)
    le_idx = {v: i for i, v in enumerate(le_vals)}
    se_idx = {v: j for j, v in enumerate(se_vals)}
    for _, row in df.iterrows():
        i, j = le_idx.get(row["LE"]), se_idx.get(row["SE"])
        if i is not None and j is not None:
            grid[i, j] = max(grid[i, j], row["Objective"])
    r = PLATEAU_NEIGHBORHOOD_RADIUS
    scores = minimum_filter(np.clip(grid, 0, None), size=2*r+1, mode="nearest")
    fi, fj = np.unravel_index(np.argmax(scores), scores.shape)
    le_best, se_best = float(le_vals[fi]), float(se_vals[fj])
    ps = float(scores[fi, fj])
    log.info("  LE×SE plateau champion: LE=%.4g  SE=%.4g  (plateau_score=%.0f)",
             le_best, se_best, ps)
    return le_best, se_best, ps


def best_stp_lmt(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Return (STP, LMT, plateau_score) of the plateau champion."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    stp_vals = np.sort(df["STP"].unique())
    lmt_vals = np.sort(df["LMT"].unique())
    grid = np.full((len(stp_vals), len(lmt_vals)), 0.0)
    si = {v: i for i, v in enumerate(stp_vals)}
    li = {v: j for j, v in enumerate(lmt_vals)}
    for _, row in df.iterrows():
        i, j = si.get(row["STP"]), li.get(row["LMT"])
        if i is not None and j is not None:
            grid[i, j] = max(grid[i, j], row["Objective"])
    r = PLATEAU_NEIGHBORHOOD_RADIUS
    scores = minimum_filter(np.clip(grid, 0, None), size=2*r+1, mode="nearest")
    fi, fj = np.unravel_index(np.argmax(scores), scores.shape)
    stp_best, lmt_best = float(stp_vals[fi]), float(lmt_vals[fj])
    ps = float(scores[fi, fj])
    log.info("  STP×LMT plateau champion: STP=%.4g  LMT=%.4g  (plateau_score=%.0f)",
             stp_best, lmt_best, ps)
    return stp_best, lmt_best, ps


def best_4d(df: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
    """Return (LE, SE, STP, LMT, objective, plateau_score) of 4D champion."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    def _uniq(col): return np.sort(df[col].unique())
    LEs, SEs, STPs, LMTs = _uniq("LE"), _uniq("SE"), _uniq("STP"), _uniq("LMT")
    idx = {
        "LE":  {v: i for i, v in enumerate(LEs)},
        "SE":  {v: i for i, v in enumerate(SEs)},
        "STP": {v: i for i, v in enumerate(STPs)},
        "LMT": {v: i for i, v in enumerate(LMTs)},
    }
    shape = (len(LEs), len(SEs), len(STPs), len(LMTs))
    grid = np.zeros(shape)
    for _, row in df.iterrows():
        try:
            i = idx["LE"][row["LE"]];  j = idx["SE"][row["SE"]]
            k = idx["STP"][row["STP"]]; l = idx["LMT"][row["LMT"]]
            grid[i, j, k, l] = max(grid[i, j, k, l], row["Objective"])
        except Exception:
            pass
    r = PLATEAU_NEIGHBORHOOD_RADIUS
    scores = minimum_filter(np.clip(grid, 0, None), size=2*r+1, mode="nearest")
    fi = np.argmax(scores)
    i, j, k, l = np.unravel_index(fi, shape)
    le_v, se_v = float(LEs[i]), float(SEs[j])
    stp_v, lmt_v = float(STPs[k]), float(LMTs[l])
    log.info("  4D champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  obj=%.0f  plateau=%.0f",
             le_v, se_v, stp_v, lmt_v, float(grid[i,j,k,l]), float(scores.flat[fi]))
    return le_v, se_v, stp_v, lmt_v, float(grid[i,j,k,l]), float(scores.flat[fi])


def merge_le_se_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["Objective"] = plateau_mod.compute_objective(combined)
    combined.sort_values("Objective", ascending=False, inplace=True)
    combined.drop_duplicates(subset=["LE", "SE"], keep="first", inplace=True)
    return combined.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-attempt metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _row_metrics(df: pd.DataFrame, **params) -> Dict:
    """Look up a row by exact param values and return its metrics."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in params.items():
        if col not in df.columns:
            continue
        if df[col].nunique() <= 1:
            continue  # fixed param — all rows share this value, no need to filter
        mask &= np.isclose(df[col].astype(float), float(val), atol=0.01)
    rows = df[mask]
    if rows.empty:
        return {"net_profit": 0.0, "max_drawdown": 0.0,
                "objective": 0.0, "total_trades": 0}
    r = rows.iloc[0]
    obj = float(plateau_mod.compute_objective(rows).iloc[0])
    return {
        "net_profit":   round(float(r.get("NetProfit",   0)), 0),
        "max_drawdown": round(float(r.get("MaxDrawdown", 0)), 0),
        "objective":    round(obj, 0),
        "total_trades": int(r.get("TotalTrades", 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_json(le, se, stp, lmt, obj, plateau_score, np_, mdd, trades,
              attempt_log: List[dict]) -> Path:
    best_attempt = max(attempt_log, key=lambda x: x.get("plateau_score", 0), default=None)
    payload = {
        "strategy": "Breakout_Daily  (adaptive 10-attempt)",
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "insample": f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|  [NetProfit>0 AND MaxDrawdown<0]",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "plateau_radius": PLATEAU_NEIGHBORHOOD_RADIUS,
        "best_params": {
            "LE": le, "SE": se, "STP": stp, "LMT": lmt,
            "net_profit":    round(np_, 0),
            "max_drawdown":  round(mdd, 0),
            "objective":     round(obj, 0),
            "plateau_score": round(plateau_score, 0),
            "total_trades":  trades,
        },
        "best_plateau_attempt": best_attempt,
        "attempt_log": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_daily.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main adaptive search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn: Optional[mc.MultiChartsConnection],
               from_csv: bool, start_attempt: int) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,  best_se  = 12.0, 12.0   # start away from boundary
    best_stp, best_lmt = DEFAULT_STP, DEFAULT_LMT
    attempt_log: List[dict] = []
    le_se_dfs:   List[pd.DataFrame] = []

    def _record(attempt_num: int, name: str, df: Optional[pd.DataFrame],
                le: float, se: float, stp: float, lmt: float,
                plateau_score: float) -> None:
        metrics = _row_metrics(df, LE=le, SE=se, STP=stp, LMT=lmt) if df is not None else {}
        entry = {
            "attempt":       attempt_num,
            "name":          name,
            "rows":          len(df) if df is not None else 0,
            "LE": le, "SE": se, "STP": stp, "LMT": lmt,
            "plateau_score": round(plateau_score, 0),
            **metrics,
        }
        attempt_log.append(entry)
        log.info("  [Attempt %d] LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "plateau=%.0f  obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 attempt_num, le, se, stp, lmt,
                 plateau_score,
                 metrics.get("objective", 0),
                 metrics.get("net_profit", 0),
                 metrics.get("max_drawdown", 0),
                 metrics.get("total_trades", 0))

    # ── Attempt 1: Wide LE×SE  ────────────────────────────────────────────────
    A = 1
    _name = "01_wide"
    if start_attempt <= A:
        _cfg1 = _cfg(_name, le=(2, 80, 4), se=(2, 80, 4),
                     stp=fixed(DEFAULT_STP), lmt=fixed(DEFAULT_LMT))
        df1 = run_or_load(_name, _cfg1, conn, from_csv)
        if df1 is not None:
            le_se_dfs.append(df1)
            best_le, best_se, ps1 = best_le_se(df1)
            _record(A, _name, df1, best_le, best_se, DEFAULT_STP, DEFAULT_LMT, ps1)
    else:
        p = csv_for(_name)
        if p.exists():
            df1 = mc.load_results_csv(str(p), _cfg(_name,(2,80,4),(2,80,4),
                                                    fixed(DEFAULT_STP),fixed(DEFAULT_LMT)))
            if df1 is not None:
                le_se_dfs.append(df1)
                best_le, best_se, _ = best_le_se(df1)
    log.info("After Attempt 1 — best LE=%.4g  SE=%.4g", best_le, best_se)

    # ── Attempt 2: Zoom-in ±12 step 2  ───────────────────────────────────────
    A = 2
    _name = "02_zoom"
    if start_attempt <= A:
        _le2 = zoom(best_le, 12, 2, LE_LO, LE_HI)
        _se2 = zoom(best_se, 12, 2, SE_LO, SE_HI)
        _cfg2 = _cfg(_name, _le2, _se2, fixed(DEFAULT_STP), fixed(DEFAULT_LMT))
        df2 = run_or_load(_name, _cfg2, conn, from_csv)
        if df2 is not None:
            le_se_dfs.append(df2)
            merged = merge_le_se_dfs(le_se_dfs)
            best_le, best_se, ps2 = best_le_se(merged)
            _record(A, _name, df2, best_le, best_se, DEFAULT_STP, DEFAULT_LMT, ps2)
    else:
        p = csv_for(_name)
        if p.exists():
            _le2 = zoom(best_le, 12, 2, LE_LO, LE_HI)
            _se2 = zoom(best_se, 12, 2, SE_LO, SE_HI)
            df2 = mc.load_results_csv(str(p), _cfg(_name,_le2,_se2,
                                                    fixed(DEFAULT_STP),fixed(DEFAULT_LMT)))
            if df2 is not None:
                le_se_dfs.append(df2)
                merged = merge_le_se_dfs(le_se_dfs)
                best_le, best_se, _ = best_le_se(merged)
    log.info("After Attempt 2 — best LE=%.4g  SE=%.4g", best_le, best_se)

    # ── Attempt 3: Expand-out ±24 step 3  ────────────────────────────────────
    A = 3
    _name = "03_expand"
    if start_attempt <= A:
        _le3 = zoom(best_le, 24, 3, LE_LO, LE_HI)
        _se3 = zoom(best_se, 24, 3, SE_LO, SE_HI)
        _cfg3 = _cfg(_name, _le3, _se3, fixed(DEFAULT_STP), fixed(DEFAULT_LMT))
        df3 = run_or_load(_name, _cfg3, conn, from_csv)
        if df3 is not None:
            le_se_dfs.append(df3)
            merged = merge_le_se_dfs(le_se_dfs)
            best_le, best_se, ps3 = best_le_se(merged)
            _record(A, _name, df3, best_le, best_se, DEFAULT_STP, DEFAULT_LMT, ps3)
    else:
        p = csv_for(_name)
        if p.exists():
            _le3 = zoom(best_le, 24, 3, LE_LO, LE_HI)
            _se3 = zoom(best_se, 24, 3, SE_LO, SE_HI)
            df3 = mc.load_results_csv(str(p), _cfg(_name,_le3,_se3,
                                                    fixed(DEFAULT_STP),fixed(DEFAULT_LMT)))
            if df3 is not None:
                le_se_dfs.append(df3)
                merged = merge_le_se_dfs(le_se_dfs)
                best_le, best_se, _ = best_le_se(merged)
    log.info("After Attempt 3 — best LE=%.4g  SE=%.4g", best_le, best_se)

    # ── Attempt 4: Fine zoom ±8 step 1  ──────────────────────────────────────
    A = 4
    _name = "04_fine"
    if start_attempt <= A:
        _le4 = zoom(best_le, 8, 1, LE_LO, LE_HI)
        _se4 = zoom(best_se, 8, 1, SE_LO, SE_HI)
        _cfg4 = _cfg(_name, _le4, _se4, fixed(DEFAULT_STP), fixed(DEFAULT_LMT))
        df4 = run_or_load(_name, _cfg4, conn, from_csv)
        if df4 is not None:
            le_se_dfs.append(df4)
            merged = merge_le_se_dfs(le_se_dfs)
            best_le, best_se, ps4 = best_le_se(merged)
            _record(A, _name, df4, best_le, best_se, DEFAULT_STP, DEFAULT_LMT, ps4)
    else:
        p = csv_for(_name)
        if p.exists():
            _le4 = zoom(best_le, 8, 1, LE_LO, LE_HI)
            _se4 = zoom(best_se, 8, 1, SE_LO, SE_HI)
            df4 = mc.load_results_csv(str(p), _cfg(_name,_le4,_se4,
                                                    fixed(DEFAULT_STP),fixed(DEFAULT_LMT)))
            if df4 is not None:
                le_se_dfs.append(df4)
                merged = merge_le_se_dfs(le_se_dfs)
                best_le, best_se, _ = best_le_se(merged)
    log.info("After Attempt 4 — best LE=%.4g  SE=%.4g", best_le, best_se)

    # ── Attempt 5: Wide STP×LMT  ─────────────────────────────────────────────
    A = 5
    _name = "05_stp_lmt_wide"
    stp_lmt_dfs: List[pd.DataFrame] = []
    if start_attempt <= A:
        _cfg5 = _cfg(_name, fixed(best_le), fixed(best_se),
                     stp=(0.5, 6.0, 0.5), lmt=(2.0, 20.0, 1.0))
        df5 = run_or_load(_name, _cfg5, conn, from_csv)
        if df5 is not None:
            stp_lmt_dfs.append(df5)
            best_stp, best_lmt, ps5 = best_stp_lmt(df5)
            _record(A, _name, df5, best_le, best_se, best_stp, best_lmt, ps5)
    else:
        p = csv_for(_name)
        if p.exists():
            df5 = mc.load_results_csv(str(p), _cfg(_name, fixed(best_le), fixed(best_se),
                                                    (0.5,6.0,0.5),(2.0,20.0,1.0)))
            if df5 is not None:
                stp_lmt_dfs.append(df5)
                best_stp, best_lmt, _ = best_stp_lmt(df5)
    log.info("After Attempt 5 — best STP=%.4g  LMT=%.4g", best_stp, best_lmt)

    # ── Attempt 6: Zoom STP ±2.0 ×0.25 / LMT ±5 ×0.5  ───────────────────────
    A = 6
    _name = "06_stp_lmt_zoom"
    if start_attempt <= A:
        _stp6 = zoom(best_stp, 2.0, 0.25, STP_LO, STP_HI)
        _lmt6 = zoom(best_lmt, 5.0, 0.5,  LMT_LO, LMT_HI)
        _cfg6 = _cfg(_name, fixed(best_le), fixed(best_se), _stp6, _lmt6)
        df6 = run_or_load(_name, _cfg6, conn, from_csv)
        if df6 is not None:
            stp_lmt_dfs.append(df6)
            merged_sl = pd.concat(stp_lmt_dfs, ignore_index=True)
            merged_sl["Objective"] = plateau_mod.compute_objective(merged_sl)
            merged_sl.drop_duplicates(subset=["STP","LMT"], keep="first", inplace=True)
            best_stp, best_lmt, ps6 = best_stp_lmt(merged_sl)
            _record(A, _name, df6, best_le, best_se, best_stp, best_lmt, ps6)
    else:
        p = csv_for(_name)
        if p.exists():
            _stp6 = zoom(best_stp, 2.0, 0.25, STP_LO, STP_HI)
            _lmt6 = zoom(best_lmt, 5.0, 0.5,  LMT_LO, LMT_HI)
            df6 = mc.load_results_csv(str(p), _cfg(_name, fixed(best_le), fixed(best_se),
                                                    _stp6, _lmt6))
            if df6 is not None:
                stp_lmt_dfs.append(df6)
                merged_sl = pd.concat(stp_lmt_dfs, ignore_index=True)
                merged_sl["Objective"] = plateau_mod.compute_objective(merged_sl)
                merged_sl.drop_duplicates(subset=["STP","LMT"], keep="first", inplace=True)
                best_stp, best_lmt, _ = best_stp_lmt(merged_sl)
    log.info("After Attempt 6 — best STP=%.4g  LMT=%.4g", best_stp, best_lmt)

    # ── Attempt 7: Fine STP ±0.75 ×0.1 / LMT ±3 ×0.25  ─────────────────────
    A = 7
    _name = "07_stp_lmt_fine"
    if start_attempt <= A:
        _stp7 = zoom(best_stp, 0.75, 0.1,  STP_LO, STP_HI)
        _lmt7 = zoom(best_lmt, 3.0,  0.25, LMT_LO, LMT_HI)
        _cfg7 = _cfg(_name, fixed(best_le), fixed(best_se), _stp7, _lmt7)
        df7 = run_or_load(_name, _cfg7, conn, from_csv)
        if df7 is not None:
            stp_lmt_dfs.append(df7)
            merged_sl = pd.concat(stp_lmt_dfs, ignore_index=True)
            merged_sl["Objective"] = plateau_mod.compute_objective(merged_sl)
            merged_sl.drop_duplicates(subset=["STP","LMT"], keep="first", inplace=True)
            best_stp, best_lmt, ps7 = best_stp_lmt(merged_sl)
            _record(A, _name, df7, best_le, best_se, best_stp, best_lmt, ps7)
    else:
        p = csv_for(_name)
        if p.exists():
            _stp7 = zoom(best_stp, 0.75, 0.1,  STP_LO, STP_HI)
            _lmt7 = zoom(best_lmt, 3.0,  0.25, LMT_LO, LMT_HI)
            df7 = mc.load_results_csv(str(p), _cfg(_name, fixed(best_le), fixed(best_se),
                                                    _stp7, _lmt7))
            if df7 is not None:
                stp_lmt_dfs.append(df7)
                merged_sl = pd.concat(stp_lmt_dfs, ignore_index=True)
                merged_sl["Objective"] = plateau_mod.compute_objective(merged_sl)
                merged_sl.drop_duplicates(subset=["STP","LMT"], keep="first", inplace=True)
                best_stp, best_lmt, _ = best_stp_lmt(merged_sl)
    log.info("After Attempt 7 — best STP=%.4g  LMT=%.4g", best_stp, best_lmt)

    # ── Attempt 8: Re-verify LE×SE with refined STP/LMT  ────────────────────
    A = 8
    _name = "08_re_verify_le_se"
    if start_attempt <= A:
        _le8 = zoom(best_le, 8, 1, LE_LO, LE_HI)
        _se8 = zoom(best_se, 8, 1, SE_LO, SE_HI)
        _cfg8 = _cfg(_name, _le8, _se8, fixed(best_stp), fixed(best_lmt))
        df8 = run_or_load(_name, _cfg8, conn, from_csv)
        if df8 is not None:
            le_se_dfs.append(df8)
            merged = merge_le_se_dfs(le_se_dfs)
            best_le, best_se, ps8 = best_le_se(merged)
            _record(A, _name, df8, best_le, best_se, best_stp, best_lmt, ps8)
    else:
        p = csv_for(_name)
        if p.exists():
            _le8 = zoom(best_le, 8, 1, LE_LO, LE_HI)
            _se8 = zoom(best_se, 8, 1, SE_LO, SE_HI)
            df8 = mc.load_results_csv(str(p), _cfg(_name,_le8,_se8,
                                                    fixed(best_stp),fixed(best_lmt)))
            if df8 is not None:
                le_se_dfs.append(df8)
                merged = merge_le_se_dfs(le_se_dfs)
                best_le, best_se, _ = best_le_se(merged)
    log.info("After Attempt 8 — best LE=%.4g  SE=%.4g", best_le, best_se)

    # ── Attempt 9: Wide boundary check ±30 step 4  ───────────────────────────
    A = 9
    _name = "09_boundary"
    if start_attempt <= A:
        _le9 = zoom(best_le, 30, 4, LE_LO, LE_HI)
        _se9 = zoom(best_se, 30, 4, SE_LO, SE_HI)
        _cfg9 = _cfg(_name, _le9, _se9, fixed(best_stp), fixed(best_lmt))
        df9 = run_or_load(_name, _cfg9, conn, from_csv)
        if df9 is not None:
            le_se_dfs.append(df9)
            merged = merge_le_se_dfs(le_se_dfs)
            best_le, best_se, ps9 = best_le_se(merged)
            _record(A, _name, df9, best_le, best_se, best_stp, best_lmt, ps9)
    else:
        p = csv_for(_name)
        if p.exists():
            _le9 = zoom(best_le, 30, 4, LE_LO, LE_HI)
            _se9 = zoom(best_se, 30, 4, SE_LO, SE_HI)
            df9 = mc.load_results_csv(str(p), _cfg(_name,_le9,_se9,
                                                    fixed(best_stp),fixed(best_lmt)))
            if df9 is not None:
                le_se_dfs.append(df9)
                merged = merge_le_se_dfs(le_se_dfs)
                best_le, best_se, _ = best_le_se(merged)
    log.info("After Attempt 9 — best LE=%.4g  SE=%.4g", best_le, best_se)

    # ── Attempt 10: 4D grid  ─────────────────────────────────────────────────
    A = 10
    _name = "10_4d"
    final_le, final_se, final_stp, final_lmt = best_le, best_se, best_stp, best_lmt
    final_obj, final_plateau = 0.0, 0.0
    final_np, final_mdd, final_trades = 0.0, 0.0, 0

    _le10  = zoom(best_le,  3, 1,    LE_LO,  LE_HI)
    _se10  = zoom(best_se,  3, 1,    SE_LO,  SE_HI)
    _stp10 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)
    _lmt10 = zoom(best_lmt, 2.0, 1.0,  LMT_LO, LMT_HI)
    _cfg10 = _cfg(_name, _le10, _se10, _stp10, _lmt10)
    log.info("Attempt 10 4D grid: LE=%s SE=%s STP=%s LMT=%s  (%d combos)",
             _le10, _se10, _stp10, _lmt10, _cfg10.total_runs())

    if start_attempt <= A:
        df10 = run_or_load(_name, _cfg10, conn, from_csv)
        if df10 is not None:
            final_le, final_se, final_stp, final_lmt, final_obj, final_plateau = best_4d(df10)
            mask = ((df10["LE"] == final_le) & (df10["SE"] == final_se) &
                    (df10["STP"] == final_stp) & (df10["LMT"] == final_lmt))
            row = df10[mask]
            if not row.empty:
                r = row.iloc[0]
                final_np    = float(r["NetProfit"])
                final_mdd   = float(r["MaxDrawdown"])
                final_trades = int(r.get("TotalTrades", 0))
            _record(A, _name, df10,
                    final_le, final_se, final_stp, final_lmt, final_plateau)
    else:
        p = csv_for(_name)
        if p.exists():
            df10 = mc.load_results_csv(str(p), _cfg10)
            if df10 is not None:
                final_le, final_se, final_stp, final_lmt, final_obj, final_plateau = best_4d(df10)
                mask = ((df10["LE"] == final_le) & (df10["SE"] == final_se) &
                        (df10["STP"] == final_stp) & (df10["LMT"] == final_lmt))
                row = df10[mask]
                if not row.empty:
                    r = row.iloc[0]
                    final_np    = float(r["NetProfit"])
                    final_mdd   = float(r["MaxDrawdown"])
                    final_trades = int(r.get("TotalTrades", 0))

    # ── Final summary  ───────────────────────────────────────────────────────
    log.info("")
    log.info("══════════════════════════════════════════")
    log.info("  FINAL RECOMMENDATION  (Breakout Daily)")
    log.info("══════════════════════════════════════════")
    log.info("  LE=%.4g  SE=%.4g  STP=%.4g  LMT=%.4g",
             final_le, final_se, final_stp, final_lmt)
    log.info("  NetProfit=%.0f  MaxDD=%.0f  Trades=%d",
             final_np, final_mdd, final_trades)
    log.info("  Objective=%.0f  PlateauScore=%.0f", final_obj, final_plateau)

    out = save_json(final_le, final_se, final_stp, final_lmt,
                    final_obj, final_plateau, final_np, final_mdd, final_trades,
                    attempt_log)
    print(f"\nDone — results at: {out}")
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
    ap = argparse.ArgumentParser(description="Adaptive 10-attempt Breakout Daily search")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyze existing CSVs; do not launch MC automation")
    ap.add_argument("--attempt", type=int, default=1, metavar="N",
                    help="Start from attempt N (1-10); previous CSVs loaded from disk")
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
