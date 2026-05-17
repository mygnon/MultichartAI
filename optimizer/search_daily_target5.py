"""
search_daily_target5.py — Breakout Daily NP > 6,000,000, Round 5

Verified clean seed (Round-1):
  LE=5, SE=50, STP=2.1, LMT=17  →  NP=4,908,600  MDD=-767,600  obj=31,389,205  trades=50

Gap to target:  6,000,000 - 4,908,600 = 1,091,400  (+22.2%)

Column-mapping bug (critical, from Round-4 notes):
  When only 1 parameter varies, MCReport packs multiple values per row.
  ALL attempts MUST vary ≥ 2 parameters simultaneously.

Round-5 new hypotheses (territory not yet cleanly explored):
  H1. SE 60-200 + LMT 18-40  — longer lookback captures larger breakouts
  H2. LE 1-4                  — shorter entry lookback → more entries (never tested below 5)
  H3. Wide-stop (STP 10-25)   — no premature exits → higher win rate with large LMT
  H4. Very high LMT (35-60)   — outlier mega-move capture

Attempt schedule (12 attempts, each ≤ 5,000 combos):
  A01  SE 30-160 step 5 × LMT 15-42 step 1    LE=5 STP=2.1   coarse H1 surface
  A02  LE 1-12 step 1 × SE zoom(best_se,15,5)  STP=best LMT=best   LE probe
  A03  SE zoom × LMT zoom fine 2D              LE=best STP=best     refine H1
  A04  STP 0.25-8.0 step 0.25 × SE zoom        LE=best LMT=best     STP scan
  A05  STP 8-30 step 0.5  × SE zoom            LE=best LMT=best     wide-stop H3
  A06  4D medium  LE±2 × SE±8 step 2 × STP±1 step 0.25 × LMT±3 step 1
  A07  SE 100-300 step 10 × LMT zoom            LE=best STP=best     ultra-high SE
  A08  LE 1-6 step 1 × SE 30-130 step 5         STP=best LMT=best   LE-low 2D
  A09  SE zoom(best_se,10,2) × STP zoom          LE=best LMT=best    fine SE×STP
  A10  4D precision  LE±1 × SE±4 step 1 × STP±0.4 step 0.1 × LMT±2 step 0.5
  A11  LMT 35-80 step 1 × SE zoom(best_se,20,5)  LE=best STP=best   ultra-LMT H4
  A12  Boundary  LE 1-20 step 2 × SE 10-250 step 10  STP=best LMT=best

Selection criterion:
  Primary:   rows where NetProfit > 6,000,000 — pick best obj = NP² / |MDD|
  Fallback:  if no NP > 6M found, pick highest NP overall (closest to target)

Usage
─────
  python search_daily_target5.py
  python search_daily_target5.py --from-csv          # re-analyse existing CSVs
  python search_daily_target5.py --attempt 4         # resume from attempt 4
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\daily_target5_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP    = 6_000_000.0
MAX_ATTEMPTS = 12

STP_LO, STP_HI = 0.25, 35.0
LMT_LO, LMT_HI = 2.0,  80.0
SE_LO,  SE_HI  = 1.0,  400.0
LE_LO,  LE_HI  = 1.0,  100.0

# Verified clean seed from Round-1
SEED_LE,  SEED_SE  = 5.0,  50.0
SEED_STP, SEED_LMT = 2.1,  17.0
SEED_NP   = 4_908_600.0
SEED_OBJ  = 31_389_205.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_daily_target5_{int(time.time())}.log"
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
    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  Attempt %s: %d combos EXCEEDS 5000 limit!", name, combos)
    return StrategyConfig(
        name=f"BD5_{name}",
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
    return OUTPUT_DIR / f"BD5_{name}_raw.csv"


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
# Champion selector
# Primary: best objective where NP > 6M
# Fallback: best NP overall (closest to target)
# ─────────────────────────────────────────────────────────────────────────────

def champion(df: pd.DataFrame,
             fallback_le: float, fallback_se: float,
             fallback_stp: float, fallback_lmt: float,
             ) -> Tuple[float, float, float, float, float, float, float, int, bool]:
    """Return (LE, SE, STP, LMT, objective, net_profit, max_drawdown, trades, target_met)."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    # First choice: NP > 6M with highest objective
    df_target = df[df["NetProfit"] > TARGET_NP]
    if not df_target.empty:
        best = df_target.loc[df_target["Objective"].idxmax()]
        log.info("  ★ TARGET MET in this attempt: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best.get("TotalTrades", 0)))
        return (float(best["LE"]), float(best["SE"]),
                float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best.get("TotalTrades", 0)), True)

    # Fallback: best overall objective (guide exploration toward higher NP)
    pos = df[df["Objective"] > 0]
    if pos.empty:
        log.info("  No profitable row — keeping fallback params")
        return (fallback_le, fallback_se, fallback_stp, fallback_lmt,
                0.0, 0.0, 0.0, 0, False)
    best = pos.loc[pos["Objective"].idxmax()]
    le  = float(best["LE"]);  se  = float(best["SE"])
    stp = float(best["STP"]); lmt = float(best["LMT"])
    np_ = float(best.get("NetProfit", 0))
    mdd = float(best.get("MaxDrawdown", 0))
    tr  = int(best.get("TotalTrades", 0))
    obj = float(best["Objective"])
    log.info("  Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  [NP<6M target not met]",
             le, se, stp, lmt, obj, np_, mdd, tr)
    return le, se, stp, lmt, obj, np_, mdd, tr, False


def best_np_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return the row with the highest NP in df (regardless of objective)."""
    if df is None or df.empty:
        return None
    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        return None
    return pos.loc[pos["NetProfit"].idxmax()]


def merge_dfs(dfs: List[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["Objective"] = plateau_mod.compute_objective(combined)
    combined.sort_values("Objective", ascending=False, inplace=True)
    combined.drop_duplicates(subset=keys, keep="first", inplace=True)
    return combined.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-attempt record builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_entry(attempt_num: int, name: str, df: Optional[pd.DataFrame],
                 le: float, se: float, stp: float, lmt: float,
                 obj: float, np_: float, mdd: float, trades: int,
                 target_met: bool, combos: int) -> Dict:
    return {
        "attempt":       attempt_num,
        "name":          name,
        "combos":        combos,
        "rows":          len(df) if df is not None else 0,
        "LE":  le,  "SE":  se,
        "STP": stp, "LMT": lmt,
        "net_profit":   round(np_, 0),
        "max_drawdown": round(mdd, 0),
        "objective":    round(obj, 0),
        "total_trades": trades,
        "target_met":   target_met,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_json(best_entry: Dict, attempt_log: List[dict],
              target_met: bool) -> Path:
    # Among all attempts, find the one closest to (or above) 6M
    above_target = [e for e in attempt_log if e.get("net_profit", 0) >= TARGET_NP]
    best_np_entry = max(attempt_log,
                        key=lambda x: x.get("net_profit", 0),
                        default={})

    payload = {
        "strategy":           "Breakout_Daily  (target NP>6M round-5)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|  (NP>0 AND MDD<0)",
        "selection_criterion": "Primary: NP>6M highest obj | Fallback: highest NP overall",
        "target_np":          TARGET_NP,
        "target_met":         target_met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "seed_round1_best":   {
            "LE": SEED_LE, "SE": SEED_SE,
            "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP,
            "objective":  SEED_OBJ,
        },
        "best_params":        best_entry,
        "best_np_attempt":    best_np_entry,
        "attempts_above_target": above_target,
        "attempt_log":        attempt_log,
    }
    out = OUTPUT_DIR / "final_params_daily_target5.json"
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
    all_dfs:   List[pd.DataFrame] = []
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Round-5 Breakout Daily NP>6M Search")
    log.info("  Seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  obj=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np, best_obj)
    log.info("  Target: %.0f  Gap: %.0f (+%.1f%%)",
             TARGET_NP, TARGET_NP - best_np,
             (TARGET_NP - best_np) / best_np * 100)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df: Optional[pd.DataFrame], name: str, attempt_num: int,
                combos: int) -> None:
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty:
            entry = _build_entry(attempt_num, name, df,
                                 best_le, best_se, best_stp, best_lmt,
                                 0.0, 0.0, 0.0, 0, False, combos)
            attempt_log.append(entry)
            log.info("  [A%02d %-25s]  no data", attempt_num, name)
            return

        all_dfs.append(df)
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

        entry = _build_entry(attempt_num, name, df, le, se, stp, lmt,
                             obj, np_, mdd, tr, met, combos)
        attempt_log.append(entry)

        if not best_entry or (met and not best_entry.get("target_met")) \
                or (met and entry.get("objective", 0) > best_entry.get("objective", 0)) \
                or (not best_entry.get("target_met") and entry.get("net_profit", 0) > best_entry.get("net_profit", 0)):
            best_entry = entry

        log.info("  [A%02d %-25s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  %s",
                 attempt_num, name, le, se, stp, lmt,
                 obj, np_, mdd, tr,
                 "★TARGET★" if met else
                 ("NP=%.0f/6M" % np_))
        log.info("       Global best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
                 best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A01  SE 30-160 step 5 × LMT 15-42 step 1   LE=5 STP=2.1
    #      Coarse H1 surface  (27 × 28 = 756 combos)
    # ─────────────────────────────────────────────────────────────────────────
    A = 1
    _name = "01_se_lmt_coarse"
    _le1  = fixed(SEED_LE)
    _se1  = (30.0, 160.0, 5.0)
    _stp1 = fixed(SEED_STP)
    _lmt1 = (15.0, 42.0, 1.0)
    _c1   = _cfg(_name, _le1, _se1, _stp1, _lmt1)
    log.info("A01  SE %s × LMT %s  LE=%.4g STP=%.4g  (%d combos)",
             _se1, _lmt1, SEED_LE, SEED_STP, _c1.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c1, conn, from_csv), _name, A, _c1.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A02  LE 1-12 step 1 × SE zoom(best_se, 15, 5)   STP=best LMT=best
    #      LE probe below 5  (12 × 7 = 84 combos typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 2
    _name = "02_le_probe"
    _le2  = (1.0, 12.0, 1.0)
    _se2  = zoom(best_se, 15.0, 5.0, SE_LO, SE_HI)
    _stp2 = fixed(best_stp)
    _lmt2 = fixed(best_lmt)
    _c2   = _cfg(_name, _le2, _se2, _stp2, _lmt2)
    log.info("A02  LE %s × SE %s  STP=%.4g LMT=%.4g  (%d combos)",
             _le2, _se2, best_stp, best_lmt, _c2.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c2, conn, from_csv), _name, A, _c2.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A03  SE zoom × LMT zoom fine 2D   LE=best STP=best
    #      Refine H1: SE ±25 step 2, LMT ±6 step 0.5  (≤26 × 25 = 650 typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 3
    _name = "03_se_lmt_fine"
    _le3  = fixed(best_le)
    _se3  = zoom(best_se, 25.0, 2.0, SE_LO, SE_HI)
    _stp3 = fixed(best_stp)
    _lmt3 = zoom(best_lmt, 6.0, 0.5, LMT_LO, LMT_HI)
    _c3   = _cfg(_name, _le3, _se3, _stp3, _lmt3)
    log.info("A03  SE %s × LMT %s  LE=%.4g STP=%.4g  (%d combos)",
             _se3, _lmt3, best_le, best_stp, _c3.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c3, conn, from_csv), _name, A, _c3.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A04  STP 0.25-8 step 0.25 × SE zoom(best_se, 10, 5)   LE=best LMT=best
    #      STP fine scan  (31 × 5 = 155 combos typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 4
    _name = "04_stp_scan"
    _le4  = fixed(best_le)
    _se4  = zoom(best_se, 10.0, 5.0, SE_LO, SE_HI)
    _stp4 = (0.25, 8.0, 0.25)
    _lmt4 = fixed(best_lmt)
    _c4   = _cfg(_name, _le4, _se4, _stp4, _lmt4)
    log.info("A04  STP %s × SE %s  LE=%.4g LMT=%.4g  (%d combos)",
             _stp4, _se4, best_le, best_lmt, _c4.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c4, conn, from_csv), _name, A, _c4.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A05  STP 8-30 step 0.5 × SE zoom(best_se, 15, 5)   LE=best LMT=best
    #      Wide-stop (no-stop) regime H3  (44 × 7 = 308 combos typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 5
    _name = "05_wide_stop"
    _le5  = fixed(best_le)
    _se5  = zoom(best_se, 15.0, 5.0, SE_LO, SE_HI)
    _stp5 = (8.0, 30.0, 0.5)
    _lmt5 = fixed(best_lmt)
    _c5   = _cfg(_name, _le5, _se5, _stp5, _lmt5)
    log.info("A05  STP %s × SE %s  LE=%.4g LMT=%.4g  (%d combos)",
             _stp5, _se5, best_le, best_lmt, _c5.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c5, conn, from_csv), _name, A, _c5.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A06  4D medium: LE±2 × SE±8 step 2 × STP±1 step 0.25 × LMT±3 step 1
    #      Neighbourhood sweep  (~5 × 9 × 9 × 7 = 2835 combos typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 6
    _name = "06_4d_medium"
    _le6  = zoom(best_le,  2.0, 1.0,  LE_LO,  LE_HI)
    _se6  = zoom(best_se,  8.0, 2.0,  SE_LO,  SE_HI)
    _stp6 = zoom(best_stp, 1.0, 0.25, STP_LO, STP_HI)
    _lmt6 = zoom(best_lmt, 3.0, 1.0,  LMT_LO, LMT_HI)
    _c6   = _cfg(_name, _le6, _se6, _stp6, _lmt6)
    combos6 = _c6.total_runs()
    if combos6 > 5000:
        _se6  = zoom(best_se, 6.0, 2.0, SE_LO, SE_HI)
        _stp6 = zoom(best_stp, 0.75, 0.25, STP_LO, STP_HI)
        _c6   = _cfg(_name, _le6, _se6, _stp6, _lmt6)
        combos6 = _c6.total_runs()
    log.info("A06  LE %s × SE %s × STP %s × LMT %s  (%d combos)",
             _le6, _se6, _stp6, _lmt6, combos6)
    if start_attempt <= A:
        _update(run_or_load(_name, _c6, conn, from_csv), _name, A, combos6)
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A07  SE 100-300 step 10 × LMT zoom(best_lmt, 8, 1)   LE=best STP=best
    #      Ultra-high SE H2  (21 × 17 = 357 combos typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 7
    _name = "07_ultra_se"
    _le7  = fixed(best_le)
    _se7  = (100.0, 300.0, 10.0)
    _stp7 = fixed(best_stp)
    _lmt7 = zoom(best_lmt, 8.0, 1.0, LMT_LO, LMT_HI)
    _c7   = _cfg(_name, _le7, _se7, _stp7, _lmt7)
    log.info("A07  SE %s × LMT %s  LE=%.4g STP=%.4g  (%d combos)",
             _se7, _lmt7, best_le, best_stp, _c7.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c7, conn, from_csv), _name, A, _c7.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A08  LE 1-6 step 1 × SE 30-130 step 5   STP=best LMT=best
    #      LE-low 2D scan (6 × 21 = 126 combos)
    # ─────────────────────────────────────────────────────────────────────────
    A = 8
    _name = "08_le_low_se_2d"
    _le8  = (1.0, 6.0, 1.0)
    _se8  = (30.0, 130.0, 5.0)
    _stp8 = fixed(best_stp)
    _lmt8 = fixed(best_lmt)
    _c8   = _cfg(_name, _le8, _se8, _stp8, _lmt8)
    log.info("A08  LE %s × SE %s  STP=%.4g LMT=%.4g  (%d combos)",
             _le8, _se8, best_stp, best_lmt, _c8.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c8, conn, from_csv), _name, A, _c8.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A09  SE zoom(best_se,10,2) × STP zoom(best_stp,2,0.25)   LE=best LMT=best
    #      Fine SE×STP surface  (11 × 17 = 187 combos typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 9
    _name = "09_se_stp_fine"
    _le9  = fixed(best_le)
    _se9  = zoom(best_se,  10.0, 2.0,  SE_LO,  SE_HI)
    _stp9 = zoom(best_stp, 2.0,  0.25, STP_LO, STP_HI)
    _lmt9 = fixed(best_lmt)
    _c9   = _cfg(_name, _le9, _se9, _stp9, _lmt9)
    log.info("A09  SE %s × STP %s  LE=%.4g LMT=%.4g  (%d combos)",
             _se9, _stp9, best_le, best_lmt, _c9.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c9, conn, from_csv), _name, A, _c9.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A10  4D precision: LE±1 × SE±4 step 1 × STP±0.4 step 0.1 × LMT±2 step 0.5
    #      Dense neighbourhood  (3 × 9 × 9 × 9 = 2187 combos typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 10
    _name = "10_4d_precision"
    _le10  = zoom(best_le,  1.0,  1.0,  LE_LO,  LE_HI)
    _se10  = zoom(best_se,  4.0,  1.0,  SE_LO,  SE_HI)
    _stp10 = zoom(best_stp, 0.4,  0.1,  STP_LO, STP_HI)
    _lmt10 = zoom(best_lmt, 2.0,  0.5,  LMT_LO, LMT_HI)
    _c10   = _cfg(_name, _le10, _se10, _stp10, _lmt10)
    combos10 = _c10.total_runs()
    if combos10 > 5000:
        _se10 = zoom(best_se, 3.0, 1.0, SE_LO, SE_HI)
        _c10  = _cfg(_name, _le10, _se10, _stp10, _lmt10)
        combos10 = _c10.total_runs()
    log.info("A10  LE %s × SE %s × STP %s × LMT %s  (%d combos)",
             _le10, _se10, _stp10, _lmt10, combos10)
    if start_attempt <= A:
        _update(run_or_load(_name, _c10, conn, from_csv), _name, A, combos10)
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A11  LMT 35-80 step 1 × SE zoom(best_se, 20, 5)   LE=best STP=best
    #      Ultra-high LMT H4  (46 × 9 = 414 combos typical)
    # ─────────────────────────────────────────────────────────────────────────
    A = 11
    _name = "11_ultra_lmt"
    _le11  = fixed(best_le)
    _se11  = zoom(best_se, 20.0, 5.0, SE_LO, SE_HI)
    _stp11 = fixed(best_stp)
    _lmt11 = (35.0, 80.0, 1.0)
    _c11   = _cfg(_name, _le11, _se11, _stp11, _lmt11)
    log.info("A11  LMT %s × SE %s  LE=%.4g STP=%.4g  (%d combos)",
             _lmt11, _se11, best_le, best_stp, _c11.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c11, conn, from_csv), _name, A, _c11.total_runs())
    log.info("After A11 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # A12  Boundary sweep  LE 1-20 step 2 × SE 10-250 step 10   STP=best LMT=best
    #      Final global check  (10 × 25 = 250 combos)
    # ─────────────────────────────────────────────────────────────────────────
    A = 12
    _name = "12_boundary"
    _le12  = (1.0, 20.0, 2.0)
    _se12  = (10.0, 250.0, 10.0)
    _stp12 = fixed(best_stp)
    _lmt12 = fixed(best_lmt)
    _c12   = _cfg(_name, _le12, _se12, _stp12, _lmt12)
    log.info("A12  LE %s × SE %s  STP=%.4g LMT=%.4g  (%d combos)",
             _le12, _se12, best_stp, best_lmt, _c12.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_name, _c12, conn, from_csv), _name, A, _c12.total_runs())
    log.info("After A12 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL RESULT  (Round-5 Breakout Daily)")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best params:  LE=%.4g  SE=%.4g  STP=%.4g  LMT=%.4g",
             best_entry.get("LE", best_le), best_entry.get("SE", best_se),
             best_entry.get("STP", best_stp), best_entry.get("LMT", best_lmt))
    log.info("  NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
             best_entry.get("net_profit", best_np),
             best_entry.get("max_drawdown", 0),
             best_entry.get("objective", best_obj),
             best_entry.get("total_trades", 0))
    log.info("  Target (NP>6M): %s", "MET ✓" if target_met else
             "NOT MET — closest NP=%.0f" % best_np)
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = _build_entry(0, "seed", None,
                                  SEED_LE, SEED_SE, SEED_STP, SEED_LMT,
                                  SEED_OBJ, SEED_NP, 0.0, 0, False, 0)

    out = save_json(best_entry, attempt_log, target_met)

    # ── Per-attempt summary table ─────────────────────────────────────────
    log.info("")
    log.info("  Per-attempt summary:")
    log.info("  %-4s %-28s %6s %10s %10s %12s %6s  %s",
             "Att", "Name", "Rows", "NP", "MDD", "Objective", "Trd", "Target")
    for e in attempt_log:
        log.info("  A%02d %-28s %6d %10.0f %10.0f %12.0f %6d  %s",
                 e["attempt"], e["name"], e.get("rows", 0),
                 e.get("net_profit", 0), e.get("max_drawdown", 0),
                 e.get("objective", 0), e.get("total_trades", 0),
                 "★" if e.get("target_met") else "")
    log.info("")

    print(f"\nDone — results at: {out}")
    print(f"Target NP>6M: {'MET ✓' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _auto_elevate() -> None:
    """Re-launch this script elevated (UAC prompt) and exit the current process."""
    import shlex
    script  = str(Path(__file__).resolve())
    workdir = str(Path(__file__).resolve().parent)
    extra   = [a for a in sys.argv[1:] if a != "--_elevated"]
    quoted  = [f'"{a}"' if " " in a else a for a in extra]
    all_args = f'"{script}" ' + " ".join(quoted) + " --_elevated"
    print("[auto-elevate] Requesting elevation — a UAC prompt will appear.")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). "
              "Please run as Administrator manually.")
    else:
        print(f"[auto-elevate] Elevated process launched (code={ret}).")
    sys.exit(0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Round-5 Breakout Daily search — target NP > 6M")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyse existing CSVs; do not launch MC automation")
    ap.add_argument("--attempt", type=int, default=1, metavar="N",
                    help="Start from attempt N (1-12)")
    ap.add_argument("--_elevated", action="store_true",
                    help=argparse.SUPPRESS)
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
