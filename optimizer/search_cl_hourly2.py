"""
search_cl_hourly2.py — Breakout Hourly NP > 700,000 on CME.CL HOT, Round 2

Round-1 findings (2026-05-19):
  Best: LE=1  SE=54  STP=1.3  LMT=23  NP=101,480  MDD=-41,220  Obj=249,835  trades=2455
  Gap to 700K: -598,520 (-85.5%)

R1 A10 dense-zoom landscape (SE=44-64, STP=0.8-1.8, LMT=19-27):
  SE profile:  44→85K / 52→95K / 54→101K (peak) / 56→92K / 62→86K (small bump) / 64→82K
  LMT profile: 19→69K / 21→86K / 22→93K / 23→101K (peak) / 25→97K / 26→97K / 27→87K
  STP profile: 0.8→63K / 1.1→86K / 1.2→97K / 1.3→101K (peak) / 1.5→96K / 1.8→85K

Unexplored zones for R2:
  SE > 64 (only step-10 coarse in R1, with wrong LMT)
  SE < 44 (A05/A08 garbled in R1)
  LMT > 27 (drops after 23 within tested range)
  STP > 1.8 and STP < 0.8
  LE > 1 (not explored in depth)
  Very high SE (100-200)
  High LMT combined with high STP

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  SE fine 64-100:   LE(1-4 s1) × SE(64-100 s4) × STP(1.1-1.5 s0.1) × LMT(21-27 s1)  4×10×5×7=1400
  A02  Very high SE 100-200: LE(1-4 s1) × SE(100-200 s10) × STP(1.0-1.6 s0.2) × LMT(20-28 s2) 4×11×4×5=880
  A03  High LMT 27-60:   LE(1-4 s1) × SE(50-60 s2) × STP(1.1-1.5 s0.1) × LMT(27-60 s3)  4×6×5×12=1440
  A04  Low SE 20-44:     LE(1-4 s1) × SE(20-44 s4) × STP(1.1-1.5 s0.1) × LMT(21-27 s1)  4×7×5×7=980
  A05  High STP 2-5:     LE(1-4 s1) × SE(50-60 s2) × STP(2-5 s0.5) × LMT(20-30 s2)      4×6×7×6=1008
  A06  SE=46-64 deep:    LE(1-3 s1) × SE(46-64 s2) × STP(1.1-1.5 s0.1) × LMT(22-26 s1)  3×10×5×5=750
  A07  Adaptive zoom
  A08  Low STP 0.2-0.8:  LE(1-4 s1) × SE(50-60 s2) × STP(0.2-0.8 s0.2) × LMT(21-27 s1)  4×6×4×7=672
  A09  Fine LMT 23-40:   LE(1-3 s1) × SE(50-58 s2) × STP(1.1-1.5 s0.1) × LMT(23-40 s1)  3×5×5×18=1350
  A10  Dense zoom
  A11  LE survey 2-8:    LE(2-8 s2) × SE(50-60 s2) × STP(1.1-1.5 s0.1) × LMT(21-27 s1)  4×6×5×7=840
  A12  Wide boundary R2: LE(1-5 s1) × SE(20-100 s10) × STP(0.5-3.0 s0.5) × LMT(15-35 s5) 5×9×6×5=1350
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
from typing import Dict, List, Tuple

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
SYMBOL     = "CME.CL HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\cl_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.01, 500.0
LMT_LO, LMT_HI = 1.0,  500.0
SE_LO,  SE_HI  = 1.0,  500.0
LE_LO,  LE_HI  = 1.0,  100.0

# R1 best params
SEED_LE,  SEED_SE  = 1.0, 54.0
SEED_STP, SEED_LMT = 1.3, 23.0
SEED_NP   = 101_480.0
SEED_OBJ  = 249_835.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_cl_hourly2_{int(time.time())}.log"
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
        name=f"CLH2_{name}",
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
    return OUTPUT_DIR / f"CLH2_{name}_raw.csv"


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
    log.info("=== Starting CLH2_%s (%d combos) ===", name, cfg.total_runs())
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
    best = pos.loc[pos["NetProfit"].idxmax()]
    le  = float(best["LE"]);  se  = float(best["SE"])
    stp = float(best["STP"]); lmt = float(best["LMT"])
    np_ = float(best.get("NetProfit", 0))
    mdd = float(best.get("MaxDrawdown", 0))
    tr  = int(best.get("TotalTrades", 0))
    obj = float(best["Objective"])
    log.info("  NP-Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
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
    above    = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    best_np  = max(log_, key=lambda x: x.get("net_profit", 0), default={})
    best_obj = max(log_, key=lambda x: x.get("objective", 0), default={})
    payload = {
        "strategy":           "Breakout_CL_Hourly  (target NP>700K round-2)",
        "symbol":             SYMBOL,
        "timeframe":          "Hourly (1 hour)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "r1_best": {
            "LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "best_obj_attempt":      best_obj,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_cl_hourly2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  CL Hourly Breakout NP>700K Round-2 — unexplored territory")
    log.info("  R1 best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  Obj=%.0f",
             SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP, SEED_OBJ)
    log.info("  SE>64, LMT>27, STP<0.8 all unexplored")
    log.info("  Target: %.0f  Gap: %.0f (-%.1f%%)",
             TARGET_NP, TARGET_NP - SEED_NP,
             (TARGET_NP - SEED_NP) / SEED_NP * 100)
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

        if np_ > best_np:
            best_le, best_se = le, se
            best_stp, best_lmt = stp, lmt
            best_np = np_
        if obj > best_obj:
            best_obj = obj
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
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f (need +%.1f%%)",
                 best_np, (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else 999)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  SE fine 64-100  (4×10×5×7=1400)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_se_64_100"
    _c = _cfg(_n, (1, 4, 1), (64, 100, 4), (1.1, 1.5, 0.1), (21, 27, 1))
    log.info("A01  LE(1-4 s1) × SE(64-100 s4) × STP(1.1-1.5 s0.1) × LMT(21-27 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  Very high SE 100-200  (4×11×4×5=880)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_se_100_200"
    _c = _cfg(_n, (1, 4, 1), (100, 200, 10), (1.0, 1.6, 0.2), (20, 28, 2))
    log.info("A02  LE(1-4 s1) × SE(100-200 s10) × STP(1.0-1.6 s0.2) × LMT(20-28 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  High LMT 27-60  (4×6×5×12=1440)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_high_lmt"
    _c = _cfg(_n, (1, 4, 1), (50, 60, 2), (1.1, 1.5, 0.1), (27, 60, 3))
    log.info("A03  LE(1-4 s1) × SE(50-60 s2) × STP(1.1-1.5 s0.1) × LMT(27-60 s3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  Low SE 20-44  (4×7×5×7=980)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_low_se"
    _c = _cfg(_n, (1, 4, 1), (20, 44, 4), (1.1, 1.5, 0.1), (21, 27, 1))
    log.info("A04  LE(1-4 s1) × SE(20-44 s4) × STP(1.1-1.5 s0.1) × LMT(21-27 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  High STP 2-5  (4×6×7×6=1008)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_high_stp"
    _c = _cfg(_n, (1, 4, 1), (50, 60, 2), (2, 5, 0.5), (20, 30, 2))
    log.info("A05  LE(1-4 s1) × SE(50-60 s2) × STP(2-5 s0.5) × LMT(20-30 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  SE=46-64 deep step 2  (3×10×5×5=750)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_se_fine"
    _c = _cfg(_n, (1, 3, 1), (46, 64, 2), (1.1, 1.5, 0.1), (22, 26, 1))
    log.info("A06  LE(1-3 s1) × SE(46-64 s2) × STP(1.1-1.5 s0.1) × LMT(22-26 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Adaptive zoom from R2 best so far
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_zoom_best"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_step, _se_step, _stp_step, _lmt_step in [
        (3, 10, 0.5, 5, 1, 2, 0.1, 1),
        (2,  8, 0.4, 4, 1, 2, 0.1, 1),
        (2,  6, 0.3, 3, 1, 1, 0.1, 1),
        (1,  4, 0.2, 2, 1, 1, 0.1, 1),
    ]:
        _le7  = zoom(best_le,  _r_le,  _le_step,  LE_LO,  LE_HI)
        _se7  = zoom(best_se,  _r_se,  _se_step,  SE_LO,  SE_HI)
        _stp7 = zoom(best_stp, _r_stp, _stp_step, STP_LO, STP_HI)
        _lmt7 = zoom(best_lmt, _r_lmt, _lmt_step, LMT_LO, LMT_HI)
        _c = _cfg(_n, _le7, _se7, _stp7, _lmt7)
        if _c.total_runs() <= 5000:
            break
    log.info("A07  Zoom: LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le7, _se7, _stp7, _lmt7, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  Low STP 0.2-0.8  (4×6×4×7=672)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_low_stp"
    _c = _cfg(_n, (1, 4, 1), (50, 60, 2), (0.2, 0.8, 0.2), (21, 27, 1))
    log.info("A08  LE(1-4 s1) × SE(50-60 s2) × STP(0.2-0.8 s0.2) × LMT(21-27 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  Fine LMT 23-40  (3×5×5×18=1350)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_fine_lmt"
    _c = _cfg(_n, (1, 3, 1), (50, 58, 2), (1.1, 1.5, 0.1), (23, 40, 1))
    log.info("A09  LE(1-3 s1) × SE(50-58 s2) × STP(1.1-1.5 s0.1) × LMT(23-40 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  Dense zoom
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_dense_zoom"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_step, _se_step, _stp_step, _lmt_step in [
        (3, 10, 0.5, 5, 1, 2, 0.1, 1),
        (2,  8, 0.4, 4, 1, 2, 0.1, 1),
        (2,  6, 0.3, 3, 1, 1, 0.1, 1),
        (1,  4, 0.2, 2, 1, 1, 0.1, 1),
    ]:
        _le10  = zoom(best_le,  _r_le,  _le_step,  LE_LO,  LE_HI)
        _se10  = zoom(best_se,  _r_se,  _se_step,  SE_LO,  SE_HI)
        _stp10 = zoom(best_stp, _r_stp, _stp_step, STP_LO, STP_HI)
        _lmt10 = zoom(best_lmt, _r_lmt, _lmt_step, LMT_LO, LMT_HI)
        _c = _cfg(_n, _le10, _se10, _stp10, _lmt10)
        if _c.total_runs() <= 5000:
            break
    log.info("A10  Dense: LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le10, _se10, _stp10, _lmt10, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  LE survey 2-8  (4×6×5×7=840)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_le_survey"
    _c = _cfg(_n, (2, 8, 2), (50, 60, 2), (1.1, 1.5, 0.1), (21, 27, 1))
    log.info("A11  LE(2-8 s2) × SE(50-60 s2) × STP(1.1-1.5 s0.1) × LMT(21-27 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A12  Wide boundary R2  (5×9×6×5=1350)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_wide_boundary"
    _c = _cfg(_n, (1, 5, 1), (20, 100, 10), (0.5, 3.0, 0.5), (15, 35, 5))
    log.info("A12  LE(1-5 s1) × SE(20-100 s10) × STP(0.5-3.0 s0.5) × LMT(15-35 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    if not best_entry:
        best_entry = _entry(0, "r1_seed", None,
                            SEED_LE, SEED_SE, SEED_STP, SEED_LMT,
                            SEED_OBJ, SEED_NP, 0.0, 0, False, 0)

    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL — CL Hourly Breakout NP>700K Round-2")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
             "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
             best_entry.get("LE"), best_entry.get("SE"),
             best_entry.get("STP"), best_entry.get("LMT"),
             best_entry.get("net_profit", 0), best_entry.get("max_drawdown", 0),
             best_entry.get("objective", 0), best_entry.get("total_trades", 0))
    log.info("  Target NP>700K: %s", "MET ✓" if target_met else
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
    print(f"Target NP>700K: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
    ap = argparse.ArgumentParser(description="CL Hourly Breakout NP>700K Round-2 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1–12)")
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
