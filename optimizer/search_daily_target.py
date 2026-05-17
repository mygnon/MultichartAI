"""
search_daily_target.py — Target-driven Breakout Daily search: NP > 6,000,000

Starting from known best (LE=5, SE=18, STP=1.75, LMT=7, NP=4.66M).

Search hypotheses, in priority order:
  H1. Higher LMT (>7) → larger profit per winning trade → NP jumps proportionally
  H2. Different SE/LE ratio → different trend-capture dynamics
  H3. STP/LMT ratio optimization → better win-rate / payout balance
  H4. Full 4D grid confirmation

Attempt schedule (up to 15 attempts, each ≤ 5,000 combos):
  Phase 1 — Quick 1-D probes   (3 attempts, ≤40 combos each)
  Phase 2 — 2-D surface scans  (4 attempts, ≤400 combos each)
  Phase 3 — Fine 3-D / 4-D     (4 attempts, ≤3000 combos each)
  Phase 4 — Extended (if NP<6M)(4 attempts, various)

Termination: stops when NP > 6M found AND 2 consecutive attempts
             produce no improvement > 5% in objective.
Hard limit: 15 attempts.

Selection criterion: argmax(NetProfit² / |MaxDrawdown|)
NP target:          > 6,000,000
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_target_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP   = 6_000_000.0
MAX_ATTEMPTS = 15

STP_LO, STP_HI = 0.5, 8.0
LMT_LO, LMT_HI = 2.0, 24.0
LE_LO,  LE_HI  = 2, 100
SE_LO,  SE_HI  = 2, 100

# Seed from previous max-objective run (search_daily_maxobj.py)
SEED_LE,  SEED_SE  = 5.0, 18.0
SEED_STP, SEED_LMT = 1.75, 7.0
SEED_NP   = 4_657_800.0
SEED_OBJ  = 26_619_756.0

_LOG_FILE = OUTPUT_DIR / f"search_daily_target_{int(time.time())}.log"
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
        name=f"BDT_{name}",
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
    return OUTPUT_DIR / f"BDT_{name}_raw.csv"


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

    log.info("=== Attempt %-30s (%d combos) ===", name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done in %.1f min — %s", (time.time()-t0)/60, Path(raw_csv).name)
        return mc.load_results_csv(raw_csv, cfg)
    except Exception as e:
        log.error("  FAILED: %s", e, exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Champion selector — argmax objective
# ─────────────────────────────────────────────────────────────────────────────

def champion(df: pd.DataFrame,
             fallback_le: float, fallback_se: float,
             fallback_stp: float, fallback_lmt: float
             ) -> Tuple[float, float, float, float, float, float, float, int]:
    """Return (LE, SE, STP, LMT, objective, net_profit, max_drawdown, trades)
    of the best-objective row.  Falls back to seed values if no profitable row."""
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


def merge_dfs_by_key(dfs: List[pd.DataFrame],
                     keys: List[str]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["Objective"] = plateau_mod.compute_objective(combined)
    combined.sort_values("Objective", ascending=False, inplace=True)
    combined.drop_duplicates(subset=keys, keep="first", inplace=True)
    return combined.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-attempt metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _row_metrics(df: pd.DataFrame, **params) -> Dict:
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in params.items():
        if col not in df.columns:
            continue
        if df[col].nunique() <= 1:
            continue
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

def save_json(le, se, stp, lmt, obj, np_, mdd, trades,
              attempt_log: List[dict], target_met: bool) -> Path:
    best_attempt = max(attempt_log, key=lambda x: x.get("objective", 0), default=None)
    payload = {
        "strategy":          "Breakout_Daily  (target NP>6M search)",
        "symbol":            SYMBOL,
        "timeframe":         "Daily (1440 min)",
        "insample":          f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function":"NetProfit² / |MaxDrawdown|  [NetProfit>0 AND MaxDrawdown<0]",
        "target_np":         TARGET_NP,
        "target_met":        target_met,
        "generated_at":      datetime.now().isoformat(timespec="seconds"),
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
    out = OUTPUT_DIR / "final_params_daily_target.json"
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

    # Initialise from previous known-best run
    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met = False
    stagnant   = 0          # consecutive attempts without ≥5% obj improvement
    attempt_log: List[dict] = []

    log.info("Seed params: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g "
             "NP=%.0f OBJ=%.0f  target=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np, best_obj, TARGET_NP)

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
        entry = {
            "attempt": attempt_num, "name": name, "rows": len(df),
            "LE": le, "SE": se, "STP": stp, "LMT": lmt,
            "net_profit":   round(np_, 0),
            "max_drawdown": round(mdd, 0),
            "objective":    round(obj, 0),
            "total_trades": tr,
        }
        attempt_log.append(entry)
        log.info("  [Attempt %02d %-30s] LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  "
                 "[target=%s  stagnant=%d]",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "YES" if best_np >= TARGET_NP else "no", stagnant)

    def _done(attempt_num: int) -> bool:
        if attempt_num > MAX_ATTEMPTS:
            log.info("Max attempts (%d) reached — stopping.", MAX_ATTEMPTS)
            return True
        if target_met and stagnant >= 2:
            log.info("Target met and %d consecutive stagnant attempts — stopping.", stagnant)
            return True
        return False

    A = 0  # attempt counter

    # ─── Phase 1: Quick 1-D probes ──────────────────────────────────────────
    # H1: Higher LMT is the single most promising lever.
    # Probe: LE=5, SE=18, STP=1.75, LMT 5-24 step 0.5 (39 combos)
    A += 1
    _name = "01_lmt_probe"
    if start_attempt <= A and not _done(A):
        _lmt1 = (5.0, 24.0, 0.5)
        _c = _cfg(_name, fixed(SEED_LE), fixed(SEED_SE), fixed(SEED_STP), _lmt1)
        log.info("H1: LMT probe — fixed LE=%.4g SE=%.4g STP=%.4g, LMT %s (%d combos)",
                 SEED_LE, SEED_SE, SEED_STP, _lmt1, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # H2: SE might need to be higher (try wide SE range with best LMT found)
    A += 1
    _name = "02_se_probe"
    if start_attempt <= A and not _done(A):
        _se2 = (5.0, 80.0, 5.0)  # 16 combos
        _c = _cfg(_name, fixed(best_le), _se2, fixed(best_stp), fixed(best_lmt))
        log.info("H2: SE probe — LE=%.4g STP=%.4g LMT=%.4g, SE %s (%d combos)",
                 best_le, best_stp, best_lmt, _se2, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # H3: LE might also need adjustment with new LMT/SE
    A += 1
    _name = "03_le_probe"
    if start_attempt <= A and not _done(A):
        _le3 = (2.0, 50.0, 3.0)  # 17 combos
        _c = _cfg(_name, _le3, fixed(best_se), fixed(best_stp), fixed(best_lmt))
        log.info("H3: LE probe — SE=%.4g STP=%.4g LMT=%.4g, LE %s (%d combos)",
                 best_se, best_stp, best_lmt, _le3, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Phase 2: 2-D surface scans ─────────────────────────────────────────
    # Wide LE×SE with best STP/LMT
    A += 1
    _name = "04_le_se_wide"
    le_se_dfs: List[pd.DataFrame] = []
    if start_attempt <= A and not _done(A):
        _le4 = (2.0, 60.0, 3.0)   # 21 values
        _se4 = (5.0, 80.0, 4.0)   # 20 values  → 420 combos
        _c = _cfg(_name, _le4, _se4, fixed(best_stp), fixed(best_lmt))
        log.info("Wide LE×SE: %s × %s STP=%.4g LMT=%.4g (%d combos)",
                 _le4, _se4, best_stp, best_lmt, _c.total_runs())
        df4 = run_or_load(_name, _c, conn, from_csv)
        if df4 is not None:
            le_se_dfs.append(df4)
        _update(df4, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # STP×LMT 2-D with fixed LE/SE
    A += 1
    _name = "05_stp_lmt_2d"
    stp_lmt_dfs: List[pd.DataFrame] = []
    if start_attempt <= A and not _done(A):
        _stp5 = (0.5, 5.0, 0.25)   # 19 values
        _lmt5 = (4.0, 18.0, 1.0)   # 15 values  → 285 combos
        _c = _cfg(_name, fixed(best_le), fixed(best_se), _stp5, _lmt5)
        log.info("STP×LMT 2D: LE=%.4g SE=%.4g, STP %s LMT %s (%d combos)",
                 best_le, best_se, _stp5, _lmt5, _c.total_runs())
        df5 = run_or_load(_name, _c, conn, from_csv)
        if df5 is not None:
            stp_lmt_dfs.append(df5)
        _update(df5, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # Zoom LE×SE around current best with refined STP/LMT
    A += 1
    _name = "06_le_se_zoom"
    if start_attempt <= A and not _done(A):
        _le6 = zoom(best_le, 8, 1, LE_LO, LE_HI)   # ≤17 values
        _se6 = zoom(best_se, 10, 1, SE_LO, SE_HI)  # ≤21 values  → ≤357 combos
        _c = _cfg(_name, _le6, _se6, fixed(best_stp), fixed(best_lmt))
        log.info("LE×SE zoom: %s × %s STP=%.4g LMT=%.4g (%d combos)",
                 _le6, _se6, best_stp, best_lmt, _c.total_runs())
        df6 = run_or_load(_name, _c, conn, from_csv)
        if df6 is not None:
            le_se_dfs.append(df6)
            merged = merge_dfs_by_key(le_se_dfs, ["LE","SE"])
            best_le, best_se, best_stp_tmp, best_lmt_tmp, obj, np_, mdd, tr = \
                champion(merged, best_le, best_se, best_stp, best_lmt)
            if obj > best_obj:
                best_obj, best_np = obj, np_
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df6, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # STP×LMT fine zoom
    A += 1
    _name = "07_stp_lmt_fine"
    if start_attempt <= A and not _done(A):
        _stp7 = zoom(best_stp, 1.5, 0.25, STP_LO, STP_HI)  # ≤13 values
        _lmt7 = zoom(best_lmt, 4.0, 0.5,  LMT_LO, LMT_HI)  # ≤17 values  → ≤221 combos
        _c = _cfg(_name, fixed(best_le), fixed(best_se), _stp7, _lmt7)
        log.info("STP×LMT fine: LE=%.4g SE=%.4g, STP %s LMT %s (%d combos)",
                 best_le, best_se, _stp7, _lmt7, _c.total_runs())
        df7 = run_or_load(_name, _c, conn, from_csv)
        if df7 is not None:
            stp_lmt_dfs.append(df7)
            merged_sl = merge_dfs_by_key(stp_lmt_dfs, ["STP","LMT"])
            _, _, best_stp, best_lmt, obj, np_, mdd, tr = \
                champion(merged_sl, best_le, best_se, best_stp, best_lmt)
            if obj > best_obj:
                best_obj, best_np = obj, np_
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df7, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Phase 3: Fine 3-D / 4-D verification ───────────────────────────────
    # Re-verify LE×SE with final STP/LMT
    A += 1
    _name = "08_le_se_verify"
    if start_attempt <= A and not _done(A):
        _le8 = zoom(best_le, 10, 1, LE_LO, LE_HI)   # ≤21 values
        _se8 = zoom(best_se, 12, 1, SE_LO, SE_HI)   # ≤25 values  → ≤525 combos
        _c = _cfg(_name, _le8, _se8, fixed(best_stp), fixed(best_lmt))
        log.info("LE×SE verify: %s × %s STP=%.4g LMT=%.4g (%d combos)",
                 _le8, _se8, best_stp, best_lmt, _c.total_runs())
        df8 = run_or_load(_name, _c, conn, from_csv)
        if df8 is not None:
            le_se_dfs.append(df8)
            merged = merge_dfs_by_key(le_se_dfs, ["LE","SE"])
            best_le, best_se, _, _, obj, np_, mdd, tr = \
                champion(merged, best_le, best_se, best_stp, best_lmt)
            if obj > best_obj:
                best_obj, best_np = obj, np_
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df8, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # 4-D grid around current best
    A += 1
    _name = "09_4d_grid"
    if start_attempt <= A and not _done(A):
        _le9  = zoom(best_le,  4, 1,    LE_LO,  LE_HI)   # ≤9 values
        _se9  = zoom(best_se,  5, 1,    SE_LO,  SE_HI)   # ≤11 values
        _stp9 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)  # ≤5 values
        _lmt9 = zoom(best_lmt, 2.0, 0.5,  LMT_LO, LMT_HI)  # ≤9 values
        _c = _cfg(_name, _le9, _se9, _stp9, _lmt9)
        log.info("4D grid: LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le9, _se9, _stp9, _lmt9, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Phase 4: Extended search if NP < 6M ────────────────────────────────
    # High LMT zone (LMT 8-24) — main push to cross 6M threshold
    A += 1
    _name = "10_high_lmt"
    if start_attempt <= A and not _done(A):
        _le10  = zoom(best_le, 6, 1, LE_LO, LE_HI)    # ≤13 values
        _se10  = zoom(best_se, 6, 1, SE_LO, SE_HI)    # ≤13 values
        _lmt10 = (max(LMT_LO, best_lmt + 1.0), 24.0, 0.5)  # push above current best
        _c = _cfg(_name, _le10, _se10, fixed(best_stp), _lmt10)
        combos = _c.total_runs()
        if combos > 5000:
            _le10  = zoom(best_le, 4, 1, LE_LO, LE_HI)
            _se10  = zoom(best_se, 4, 1, SE_LO, SE_HI)
            _c = _cfg(_name, _le10, _se10, fixed(best_stp), _lmt10)
        log.info("High LMT: LE=%s SE=%s STP=%.4g LMT %s (%d combos)",
                 _le10, _se10, best_stp, _lmt10, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # STP×LMT 2D with high LMT range included
    A += 1
    _name = "11_stp_lmt_high"
    if start_attempt <= A and not _done(A):
        _stp11 = zoom(best_stp, 2.0, 0.25, STP_LO, STP_HI)  # ≤17 values
        _lmt11 = (max(LMT_LO, best_lmt - 2.0),
                  min(LMT_HI, best_lmt + 8.0), 0.5)          # 21 values
        _c = _cfg(_name, fixed(best_le), fixed(best_se), _stp11, _lmt11)
        combos = _c.total_runs()
        if combos > 5000:
            _stp11 = zoom(best_stp, 1.5, 0.25, STP_LO, STP_HI)
            _c = _cfg(_name, fixed(best_le), fixed(best_se), _stp11, _lmt11)
        log.info("STP×LMT high: LE=%.4g SE=%.4g STP %s LMT %s (%d combos)",
                 best_le, best_se, _stp11, _lmt11, _c.total_runs())
        df11 = run_or_load(_name, _c, conn, from_csv)
        if df11 is not None:
            stp_lmt_dfs.append(df11)
            merged_sl = merge_dfs_by_key(stp_lmt_dfs, ["STP","LMT"])
            _, _, best_stp, best_lmt, obj, np_, mdd, tr = \
                champion(merged_sl, best_le, best_se, best_stp, best_lmt)
            if obj > best_obj:
                best_obj, best_np = obj, np_
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df11, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # Wide LE×SE boundary check with updated best STP/LMT
    A += 1
    _name = "12_le_se_boundary"
    if start_attempt <= A and not _done(A):
        _le12 = (2.0, 100.0, 5.0)   # 20 values
        _se12 = (2.0, 100.0, 5.0)   # 20 values  → 400 combos
        _c = _cfg(_name, _le12, _se12, fixed(best_stp), fixed(best_lmt))
        log.info("Wide boundary: LE %s SE %s STP=%.4g LMT=%.4g (%d combos)",
                 _le12, _se12, best_stp, best_lmt, _c.total_runs())
        df12 = run_or_load(_name, _c, conn, from_csv)
        if df12 is not None:
            le_se_dfs.append(df12)
            merged = merge_dfs_by_key(le_se_dfs, ["LE","SE"])
            best_le, best_se, _, _, obj, np_, mdd, tr = \
                champion(merged, best_le, best_se, best_stp, best_lmt)
            if obj > best_obj:
                best_obj, best_np = obj, np_
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df12, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # Dense 4-D grid around refined best
    A += 1
    _name = "13_4d_dense"
    if start_attempt <= A and not _done(A):
        _le13  = zoom(best_le,  3, 1,    LE_LO,  LE_HI)   # ≤7
        _se13  = zoom(best_se,  5, 1,    SE_LO,  SE_HI)   # ≤11
        _stp13 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)  # ≤5
        _lmt13 = zoom(best_lmt, 3.0, 0.5,  LMT_LO, LMT_HI)  # ≤13
        _c = _cfg(_name, _le13, _se13, _stp13, _lmt13)
        log.info("Dense 4D: LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le13, _se13, _stp13, _lmt13, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # Final 4-D micro grid — precision confirmation
    A += 1
    _name = "14_4d_micro"
    if start_attempt <= A and not _done(A):
        _le14  = zoom(best_le,  2, 1,    LE_LO,  LE_HI)   # ≤5
        _se14  = zoom(best_se,  3, 1,    SE_LO,  SE_HI)   # ≤7
        _stp14 = zoom(best_stp, 0.3, 0.1, STP_LO, STP_HI)  # ≤7
        _lmt14 = zoom(best_lmt, 2.0, 0.25, LMT_LO, LMT_HI) # ≤17
        _c = _cfg(_name, _le14, _se14, _stp14, _lmt14)
        log.info("Micro 4D: LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le14, _se14, _stp14, _lmt14, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # Bonus attempt 15: very high LMT × wide SE (if still below target)
    A += 1
    _name = "15_high_lmt_se"
    if start_attempt <= A and not _done(A):
        _le15  = zoom(best_le, 3, 1, LE_LO, LE_HI)           # ≤7
        _se15  = (5.0, 80.0, 5.0)                             # 16 values
        _stp15 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)   # ≤5
        _lmt15 = (12.0, 24.0, 1.0)                            # 13 values → 7×16×5×13 = 7280 — too many
        # Reduce: fix STP, vary LMT and SE
        _lmt15 = (10.0, 24.0, 1.0)  # 15 values
        _c = _cfg(_name, _le15, _se15, fixed(best_stp), _lmt15)
        combos = _c.total_runs()
        if combos > 5000:
            _le15 = fixed(best_le)
            _c = _cfg(_name, _le15, _se15, fixed(best_stp), _lmt15)
        log.info("High LMT×SE: LE=%s SE=%s STP=%.4g LMT %s (%d combos)",
                 _le15, _se15, best_stp, _lmt15, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Final summary ────────────────────────────────────────────────────────
    log.info("")
    log.info("══════════════════════════════════════════════════════")
    log.info("  FINAL RECOMMENDATION  (Breakout Daily, NP>6M search)")
    log.info("══════════════════════════════════════════════════════")
    log.info("  LE=%.4g  SE=%.4g  STP=%.4g  LMT=%.4g",
             best_le, best_se, best_stp, best_lmt)
    log.info("  NetProfit=%.0f  MaxDD=%.0f", best_np,
             next((e["max_drawdown"] for e in reversed(attempt_log)
                   if e.get("objective", 0) == best_obj), 0))
    log.info("  Objective=%.0f  TargetMet=%s", best_obj,
             "YES ✓" if target_met else "NO — best NP was %.0f" % best_np)

    best_entry = max(attempt_log, key=lambda x: x.get("objective", 0), default={})
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
        description="Target-driven Breakout Daily search — NP > 6M")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyze existing CSVs; do not launch MC automation")
    ap.add_argument("--attempt", type=int, default=1, metavar="N",
                    help="Start from attempt N (1-15)")
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
