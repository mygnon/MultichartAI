"""
search_daily_target2.py — Round 2: Breakout Daily NP > 6,000,000

Seed from round-1 winner (A15): LE=5, SE=50, STP=2.1, LMT=17, NP=4,908,600.

Round-1 key finding:
  - NP never exceeded ~4.9M across 15 attempts
  - LMT was capped at 24 in round-1 probes — NEVER explored LMT > 24
  - SE was capped at 80 in round-1 probes — NEVER explored SE > 80
  - A15 winner (SE=50, LMT=17) was at the BOUNDARY of the search space

Round-2 new hypotheses:
  H1. LMT=25-100: extreme profit target; rare but massive winners → NP jump
  H2. SE=100-500: ultra-selective short entries; combined with high LMT → big NP
  H3. STP=0.25-1.0 + high LMT: asymmetric R:R (tight stop / huge target)
  H4. LE=1-4 + high SE/LMT: faster long entries with selective shorts
  H5. 2D / 4D dense grids in the newly expanded space

Hard limit: 15 attempts, each ≤ 5,000 combos.
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_target2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 6_000_000.0
MAX_ATTEMPTS = 15

# Expanded search bounds — key change from round-1
STP_LO, STP_HI = 0.25, 10.0
LMT_LO, LMT_HI = 2.0,  100.0   # was 24.0
SE_LO,  SE_HI  = 2.0,  500.0   # was 100.0
LE_LO,  LE_HI  = 1.0,  100.0

# Seed from round-1 A15 best
SEED_LE,  SEED_SE  = 5.0, 50.0
SEED_STP, SEED_LMT = 2.1, 17.0
SEED_NP   = 4_908_600.0
SEED_OBJ  = 31_389_205.0

_LOG_FILE = OUTPUT_DIR / f"search_daily_target2_{int(time.time())}.log"
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
        name=f"BDT2_{name}",
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
    return OUTPUT_DIR / f"BDT2_{name}_raw.csv"


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
    """Return (LE, SE, STP, LMT, objective, net_profit, max_drawdown, trades)."""
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


def merge_best(dfs: List[pd.DataFrame],
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
        "strategy":           "Breakout_Daily  (target NP>6M round-2)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|  [NetProfit>0 AND MaxDrawdown<0]",
        "target_np":          TARGET_NP,
        "target_met":         target_met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "round1_best": {
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
    out = OUTPUT_DIR / "final_params_daily_target2.json"
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

    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met = False
    stagnant   = 0
    attempt_log: List[dict] = []

    log.info("Round-2 seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g "
             "NP=%.0f OBJ=%.0f  target=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np, best_obj, TARGET_NP)
    log.info("New ranges: LMT up to %.0f, SE up to %.0f, STP down to %.4g",
             LMT_HI, SE_HI, STP_LO)

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

    A = 0
    stp_lmt_dfs: List[pd.DataFrame] = []
    le_se_dfs:   List[pd.DataFrame] = []

    # ─── Phase 1: Break the LMT=24 and SE=80 ceilings ──────────────────────

    # A01 — LMT high probe (KEY: first attempt beyond round-1 ceiling)
    # LE=5, SE=50, STP=2.1  LMT 17-60 step 1  →  44 combos
    A += 1
    _name = "01_lmt_high"
    if start_attempt <= A and not _done(A):
        _lmt1 = (SEED_LMT, 60.0, 1.0)           # 44 combos
        _c = _cfg(_name, fixed(SEED_LE), fixed(SEED_SE), fixed(SEED_STP), _lmt1)
        log.info("H1: LMT high probe — LE=%.4g SE=%.4g STP=%.4g, LMT %s (%d combos)",
                 SEED_LE, SEED_SE, SEED_STP, _lmt1, _c.total_runs())
        df1 = run_or_load(_name, _c, conn, from_csv)
        if df1 is not None:
            stp_lmt_dfs.append(df1)
        _update(df1, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A02 — SE very high probe
    # LE=5, STP=2.1, LMT=best, SE 50-450 step 25  →  17 combos
    A += 1
    _name = "02_se_high"
    if start_attempt <= A and not _done(A):
        _se2 = (50.0, 450.0, 25.0)              # 17 combos
        _c = _cfg(_name, fixed(SEED_LE), _se2, fixed(SEED_STP), fixed(best_lmt))
        log.info("H2: SE high probe — LE=%.4g STP=%.4g LMT=%.4g, SE %s (%d combos)",
                 SEED_LE, SEED_STP, best_lmt, _se2, _c.total_runs())
        df2 = run_or_load(_name, _c, conn, from_csv)
        if df2 is not None:
            le_se_dfs.append(df2)
        _update(df2, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A03 — STP low probe (asymmetric R:R)
    # LE=5, SE=best, LMT=best, STP 0.25-3.5 step 0.25  →  14 combos
    A += 1
    _name = "03_stp_low"
    if start_attempt <= A and not _done(A):
        _stp3 = (0.25, 3.5, 0.25)               # 14 combos
        _c = _cfg(_name, fixed(best_le), fixed(best_se), _stp3, fixed(best_lmt))
        log.info("H3: STP probe — LE=%.4g SE=%.4g LMT=%.4g, STP %s (%d combos)",
                 best_le, best_se, best_lmt, _stp3, _c.total_runs())
        df3 = run_or_load(_name, _c, conn, from_csv)
        if df3 is not None:
            stp_lmt_dfs.append(df3)
        _update(df3, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A04 — LE fine probe (LE=1-25 step 1)
    # SE=best, STP=best, LMT=best, LE 1-25 step 1  →  25 combos
    A += 1
    _name = "04_le_fine"
    if start_attempt <= A and not _done(A):
        _le4 = (1.0, 25.0, 1.0)                 # 25 combos
        _c = _cfg(_name, _le4, fixed(best_se), fixed(best_stp), fixed(best_lmt))
        log.info("H4: LE fine — SE=%.4g STP=%.4g LMT=%.4g, LE %s (%d combos)",
                 best_se, best_stp, best_lmt, _le4, _c.total_runs())
        df4 = run_or_load(_name, _c, conn, from_csv)
        if df4 is not None:
            le_se_dfs.append(df4)
        _update(df4, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Phase 2: 2D surface scans in expanded space ────────────────────────

    # A05 — SE × LMT 2D (main expanded surface)
    # LE=5, STP=best, SE 20-250 step 10, LMT 10-60 step 3  →  24×17=408 combos
    A += 1
    _name = "05_se_lmt_2d"
    if start_attempt <= A and not _done(A):
        _se5  = (20.0, 250.0, 10.0)             # 24 values
        _lmt5 = (10.0, 60.0,  3.0)              # 17 values → 408 combos
        _c = _cfg(_name, fixed(best_le), _se5, fixed(best_stp), _lmt5)
        log.info("SE×LMT 2D: LE=%.4g STP=%.4g, SE %s LMT %s (%d combos)",
                 best_le, best_stp, _se5, _lmt5, _c.total_runs())
        df5 = run_or_load(_name, _c, conn, from_csv)
        if df5 is not None:
            stp_lmt_dfs.append(df5)
            le_se_dfs.append(df5)
            m = merge_best(stp_lmt_dfs, ["SE","LMT"])
            _, best_se_tmp, _, best_lmt_tmp, obj5, np5, mdd5, tr5 = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj5 > best_obj:
                best_se, best_lmt = best_se_tmp, best_lmt_tmp
                best_obj, best_np = obj5, np5
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df5, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A06 — STP × LMT 2D (high LMT range)
    # LE=best, SE=best, STP 0.25-4 step 0.25, LMT 15-65 step 3  →  16×17=272 combos
    A += 1
    _name = "06_stp_lmt_high"
    if start_attempt <= A and not _done(A):
        _stp6 = (0.25, 4.0,  0.25)             # 16 values
        _lmt6 = (15.0, 65.0, 3.0)              # 17 values → 272 combos
        _c = _cfg(_name, fixed(best_le), fixed(best_se), _stp6, _lmt6)
        log.info("STP×LMT high: LE=%.4g SE=%.4g, STP %s LMT %s (%d combos)",
                 best_le, best_se, _stp6, _lmt6, _c.total_runs())
        df6 = run_or_load(_name, _c, conn, from_csv)
        if df6 is not None:
            stp_lmt_dfs.append(df6)
            m = merge_best(stp_lmt_dfs, ["STP","LMT"])
            _, _, best_stp_tmp, best_lmt_tmp, obj6, np6, mdd6, tr6 = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj6 > best_obj:
                best_stp, best_lmt = best_stp_tmp, best_lmt_tmp
                best_obj, best_np = obj6, np6
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df6, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A07 — LE × SE 2D (wide, with updated STP/LMT)
    # STP=best, LMT=best, LE 1-20 step 1, SE 20-200 step 10  →  20×19=380 combos
    A += 1
    _name = "07_le_se_2d"
    if start_attempt <= A and not _done(A):
        _le7 = (1.0, 20.0, 1.0)                # 20 values
        _se7 = (20.0, 200.0, 10.0)             # 19 values → 380 combos
        _c = _cfg(_name, _le7, _se7, fixed(best_stp), fixed(best_lmt))
        log.info("LE×SE 2D: STP=%.4g LMT=%.4g, LE %s SE %s (%d combos)",
                 best_stp, best_lmt, _le7, _se7, _c.total_runs())
        df7 = run_or_load(_name, _c, conn, from_csv)
        if df7 is not None:
            le_se_dfs.append(df7)
            m = merge_best(le_se_dfs, ["LE","SE"])
            best_le_tmp, best_se_tmp, _, _, obj7, np7, mdd7, tr7 = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj7 > best_obj:
                best_le, best_se = best_le_tmp, best_se_tmp
                best_obj, best_np = obj7, np7
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df7, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Phase 3: 4D grids in expanded space ────────────────────────────────

    # A08 — 4D grid (medium zoom)
    # LE ±4 step 1, SE ±25 step 5, STP ±0.75 step 0.25, LMT ±10 step 2
    A += 1
    _name = "08_4d_wide"
    if start_attempt <= A and not _done(A):
        _le8  = zoom(best_le,  4,    1,    LE_LO,  LE_HI)
        _se8  = zoom(best_se,  25,   5,    SE_LO,  SE_HI)
        _stp8 = zoom(best_stp, 0.75, 0.25, STP_LO, STP_HI)
        _lmt8 = zoom(best_lmt, 10,   2,    LMT_LO, LMT_HI)
        _c = _cfg(_name, _le8, _se8, _stp8, _lmt8)
        combos = _c.total_runs()
        if combos > 5000:
            _le8  = zoom(best_le,  3,    1,    LE_LO,  LE_HI)
            _se8  = zoom(best_se,  20,   5,    SE_LO,  SE_HI)
            _stp8 = zoom(best_stp, 0.5,  0.25, STP_LO, STP_HI)
            _lmt8 = zoom(best_lmt, 8,    2,    LMT_LO, LMT_HI)
            _c = _cfg(_name, _le8, _se8, _stp8, _lmt8)
        log.info("4D wide: LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le8, _se8, _stp8, _lmt8, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A09 — Ultra-high LMT zone (LMT 50-100 for the first time)
    # LE=1-10 step 1, SE=best ±30 step 10, STP=best fixed, LMT 50-100 step 5
    A += 1
    _name = "09_ultra_lmt"
    if start_attempt <= A and not _done(A):
        _le9  = (1.0, 10.0, 1.0)               # 10 values
        _se9  = zoom(best_se, 30, 10, SE_LO, SE_HI)   # ≤7 values
        _lmt9 = (50.0, 100.0, 5.0)             # 11 values → 10×7×11 = 770 combos
        _c = _cfg(_name, _le9, _se9, fixed(best_stp), _lmt9)
        log.info("Ultra-LMT: LE=%s SE=%s STP=%.4g LMT=%s (%d combos)",
                 _le9, _se9, best_stp, _lmt9, _c.total_runs())
        df9 = run_or_load(_name, _c, conn, from_csv)
        if df9 is not None:
            stp_lmt_dfs.append(df9)
        _update(df9, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A10 — Ultra-high SE zone (SE 100-500)
    # LE=best, STP=best, LMT=best ±10 step 2, SE 80-400 step 20
    A += 1
    _name = "10_ultra_se"
    if start_attempt <= A and not _done(A):
        _se10  = (80.0, 400.0, 20.0)            # 17 values
        _lmt10 = zoom(best_lmt, 10, 2, LMT_LO, LMT_HI)  # ≤11 values → ≤187 combos
        _c = _cfg(_name, fixed(best_le), _se10, fixed(best_stp), _lmt10)
        log.info("Ultra-SE: LE=%.4g STP=%.4g, SE %s LMT %s (%d combos)",
                 best_le, best_stp, _se10, _lmt10, _c.total_runs())
        df10 = run_or_load(_name, _c, conn, from_csv)
        if df10 is not None:
            le_se_dfs.append(df10)
        _update(df10, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Phase 4: Precision refinement + alternative frontiers ───────────────

    # A11 — 4D fine zoom around current best
    # LE ±3 step 1, SE ±20 step 4, STP ±0.5 step 0.25, LMT ±6 step 1
    A += 1
    _name = "11_4d_fine"
    if start_attempt <= A and not _done(A):
        _le11  = zoom(best_le,  3,   1,    LE_LO,  LE_HI)
        _se11  = zoom(best_se,  20,  4,    SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, 6,   1,    LMT_LO, LMT_HI)
        _c = _cfg(_name, _le11, _se11, _stp11, _lmt11)
        combos = _c.total_runs()
        if combos > 5000:
            _le11  = zoom(best_le,  2,   1,    LE_LO,  LE_HI)
            _se11  = zoom(best_se,  15,  3,    SE_LO,  SE_HI)
            _stp11 = zoom(best_stp, 0.5, 0.25, STP_LO, STP_HI)
            _lmt11 = zoom(best_lmt, 5,   1,    LMT_LO, LMT_HI)
            _c = _cfg(_name, _le11, _se11, _stp11, _lmt11)
        log.info("4D fine: LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le11, _se11, _stp11, _lmt11, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A12 — Wide boundary scan with updated params
    # LE 1-60 step 4, SE 20-400 step 20, STP=best, LMT=best  →  15×20=300 combos
    A += 1
    _name = "12_le_se_boundary"
    if start_attempt <= A and not _done(A):
        _le12 = (1.0, 60.0, 4.0)               # 15 values
        _se12 = (20.0, 400.0, 20.0)            # 20 values → 300 combos
        _c = _cfg(_name, _le12, _se12, fixed(best_stp), fixed(best_lmt))
        log.info("LE×SE boundary: STP=%.4g LMT=%.4g, LE %s SE %s (%d combos)",
                 best_stp, best_lmt, _le12, _se12, _c.total_runs())
        df12 = run_or_load(_name, _c, conn, from_csv)
        if df12 is not None:
            le_se_dfs.append(df12)
            m = merge_best(le_se_dfs, ["LE","SE"])
            best_le_tmp, best_se_tmp, _, _, obj12, np12, mdd12, tr12 = \
                champion(m, best_le, best_se, best_stp, best_lmt)
            if obj12 > best_obj:
                best_le, best_se = best_le_tmp, best_se_tmp
                best_obj, best_np = obj12, np12
                if best_np >= TARGET_NP:
                    target_met = True
        _update(df12, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A13 — Extreme R:R: very low STP + very high LMT
    # LE=best, SE=best ±20 step 5, STP 0.25-1.5 step 0.25, LMT 30-80 step 5
    A += 1
    _name = "13_extreme_rr"
    if start_attempt <= A and not _done(A):
        _stp13 = (0.25, 1.5, 0.25)             # 6 values
        _lmt13 = (30.0, 80.0, 5.0)             # 11 values
        _se13  = zoom(best_se, 20, 5, SE_LO, SE_HI)   # ≤9 values → 6×11×9 = 594 combos
        _c = _cfg(_name, fixed(best_le), _se13, _stp13, _lmt13)
        log.info("Extreme R:R: LE=%.4g SE=%s, STP %s LMT %s (%d combos)",
                 best_le, _se13, _stp13, _lmt13, _c.total_runs())
        df13 = run_or_load(_name, _c, conn, from_csv)
        if df13 is not None:
            stp_lmt_dfs.append(df13)
        _update(df13, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A14 — 4D micro-precision around best
    # LE ±2 step 1, SE ±10 step 2, STP ±0.4 step 0.1, LMT ±5 step 1
    A += 1
    _name = "14_4d_micro"
    if start_attempt <= A and not _done(A):
        _le14  = zoom(best_le,  2,   1,   LE_LO,  LE_HI)
        _se14  = zoom(best_se,  10,  2,   SE_LO,  SE_HI)
        _stp14 = zoom(best_stp, 0.4, 0.1, STP_LO, STP_HI)
        _lmt14 = zoom(best_lmt, 5,   1,   LMT_LO, LMT_HI)
        _c = _cfg(_name, _le14, _se14, _stp14, _lmt14)
        combos = _c.total_runs()
        if combos > 5000:
            _stp14 = zoom(best_stp, 0.3, 0.1, STP_LO, STP_HI)
            _lmt14 = zoom(best_lmt, 4,   1,   LMT_LO, LMT_HI)
            _c = _cfg(_name, _le14, _se14, _stp14, _lmt14)
        log.info("4D micro: LE=%s SE=%s STP=%s LMT=%s (%d combos)",
                 _le14, _se14, _stp14, _lmt14, _c.total_runs())
        _update(run_or_load(_name, _c, conn, from_csv), _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # A15 — Final frontier: ultra-high SE × high LMT joint sweep
    # LE=best ±2, SE 100-500 step 25, STP=best, LMT=best ±15 step 3
    A += 1
    _name = "15_ultra_se_lmt"
    if start_attempt <= A and not _done(A):
        _le15  = zoom(best_le, 2, 1, LE_LO, LE_HI)    # ≤5 values
        _se15  = (100.0, 500.0, 25.0)                  # 17 values
        _lmt15 = zoom(best_lmt, 15, 3, LMT_LO, LMT_HI)  # ≤11 values → ≤935 combos
        _c = _cfg(_name, _le15, _se15, fixed(best_stp), _lmt15)
        combos = _c.total_runs()
        if combos > 5000:
            _le15 = fixed(best_le)
            _c = _cfg(_name, _le15, _se15, fixed(best_stp), _lmt15)
        log.info("Ultra SE×LMT: LE=%s SE=%s STP=%.4g LMT %s (%d combos)",
                 _le15, _se15, best_stp, _lmt15, _c.total_runs())
        df15 = run_or_load(_name, _c, conn, from_csv)
        if df15 is not None:
            le_se_dfs.append(df15)
        _update(df15, _name, A)
    log.info("After A%02d — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             A, best_le, best_se, best_stp, best_lmt, best_np)

    # ─── Final summary ────────────────────────────────────────────────────────
    best_entry = max(attempt_log, key=lambda x: x.get("objective", 0), default={})
    best_mdd   = best_entry.get("max_drawdown", 0)
    log.info("")
    log.info("══════════════════════════════════════════════════════")
    log.info("  FINAL (Round-2, Breakout Daily NP>6M search)")
    log.info("══════════════════════════════════════════════════════")
    log.info("  LE=%.4g  SE=%.4g  STP=%.4g  LMT=%.4g",
             best_le, best_se, best_stp, best_lmt)
    log.info("  NetProfit=%.0f  MaxDD=%.0f", best_np, best_mdd)
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
        description="Round-2 Target-driven Breakout Daily — NP > 6M (expanded LMT/SE)")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyze existing CSVs without MC automation")
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
