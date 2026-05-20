"""
search_cl_daily3.py — Breakout Daily NP > 700,000 on CME.CL HOT, Round 3

R2 findings: first positive NP found!
  Best NP: LE=2 SE=17 STP=1.5 LMT=1  NP=+15,510  MDD=-39,830  (but only 2% of target)
  Also:    LE=4 SE=6  STP=2.0 LMT=1  NP=+13,930  MDD=-27,760
  Pattern: LMT=1 essential; STP=1.5-2 optimal; SE=6-17 (ultra-low); STP>3 = dead

R3 strategy — dense precision search around the two productive zones:
  Zone A: SE=5-25 (ultra-low), STP=0.5-3, LMT=1, LE=1-6
  Zone B: SE=60-80, STP=1-3, LMT=1, LE=1-5 (confirm secondary zone)
  Goal: find if there's a higher NP ceiling anywhere in SE=2-30 space
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\cl_daily3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.01, 1000.0
LMT_LO, LMT_HI = 1.0,  1000.0
SE_LO,  SE_HI  = 1.0,   500.0
LE_LO,  LE_HI  = 1.0,   100.0

# R2 best NP seed
SEED_LE,  SEED_SE  = 2.0, 17.0
SEED_STP, SEED_LMT = 1.5,  1.0
SEED_NP   = 15_510.0
SEED_OBJ  = 6_040.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_cl_daily3_{int(time.time())}.log"
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
        name=f"CLD3_{name}",
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
    return OUTPUT_DIR / f"CLD3_{name}_raw.csv"


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
    log.info("=== Starting CLD3_%s (%d combos) ===", name, cfg.total_runs())
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
    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        best = df.loc[df["NetProfit"].idxmax()]
        le  = float(best["LE"]);  se  = float(best["SE"])
        stp = float(best["STP"]); lmt = float(best["LMT"])
        np_ = float(best.get("NetProfit", 0))
        mdd = float(best.get("MaxDrawdown", 0))
        tr  = int(best.get("TotalTrades", 0))
        log.info("  Best (all negative): LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
                 le, se, stp, lmt, np_)
        return le, se, stp, lmt, 0.0, np_, mdd, tr, False
    best = pos.loc[pos["NetProfit"].idxmax()]
    le  = float(best["LE"]);  se  = float(best["SE"])
    stp = float(best["STP"]); lmt = float(best["LMT"])
    np_ = float(best.get("NetProfit", 0))
    mdd = float(best.get("MaxDrawdown", 0))
    tr  = int(best.get("TotalTrades", 0))
    obj = float(best["Objective"]) if "Objective" in best else 0.0
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
    best_np  = max(log_, key=lambda x: x.get("net_profit", -999999), default={})
    best_obj = max(log_, key=lambda x: x.get("objective", 0), default={})
    payload = {
        "strategy":           "Breakout_CL_Daily  (target NP>700K round-3)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1440 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "best_params":        best,
        "best_np_attempt":    best_np,
        "best_obj_attempt":   best_obj,
        "attempts_above_target": above,
        "attempt_log":        log_,
    }
    out = OUTPUT_DIR / "final_params_cl_daily3.json"
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
    log.info("  CL Daily Breakout NP>700K Round-3 — dense precision search")
    log.info("  Symbol: %s  Timeframe: daily (1440 min)", SYMBOL)
    log.info("  R2 seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP)
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_le, best_se, best_stp, best_lmt,
                                      0, best_np, 0, 0, False, combos))
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
                    and e.get("net_profit", -999999) > best_entry.get("net_profit", -999999))):
            best_entry = e

        log.info("  [A%02d %-25s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  %s",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f", best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  Ultra-low SE 2-15 — explore SE < 17  (5×7×6×3=630)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ultra_low_se"
    _c = _cfg(_n, (1, 5, 1), (2, 14, 2), (0.5, 3.0, 0.5), (1, 3, 1))
    log.info("A01  LE(1-5 s1) × SE(2-14 s2) × STP(0.5-3 s0.5) × LMT(1-3 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  SE=15-25 fine grid — R2 winner zone  (6×6×10×2=720)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_se17_fine"
    _c = _cfg(_n, (1, 6, 1), (15, 20, 1), (0.5, 5.0, 0.5), (1, 2, 1))
    log.info("A02  LE(1-6 s1) × SE(15-20 s1) × STP(0.5-5 s0.5) × LMT(1-2 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  SE=5-10 fine — R2 SE=6 was promising  (6×6×10×2=720)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_se5_10_fine"
    _c = _cfg(_n, (1, 6, 1), (4, 12, 1), (0.5, 5.0, 0.5), (1, 2, 1))
    log.info("A03  LE(1-6 s1) × SE(4-12 s1) × STP(0.5-5 s0.5) × LMT(1-2 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  SE=17 micro-step — maximise within winner  (6×5×10×2=600)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_se17_micro"
    _c = _cfg(_n, (1, 6, 1), (14, 22, 2), (0.5, 5.0, 0.5), (1, 2, 1))
    log.info("A04  LE(1-6 s1) × SE(14-22 s2) × STP(0.5-5 s0.5) × LMT(1-2 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  LE sweep with SE=17 locked  (8×5×10×2=800)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_le_sweep_se17"
    _c = _cfg(_n, (1, 8, 1), (15, 19, 1), (0.5, 5.0, 0.5), (1, 2, 1))
    log.info("A05  LE(1-8 s1) × SE(15-19 s1) × STP(0.5-5 s0.5) × LMT(1-2 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  SE=20-30 — test just above winner zone  (5×6×10×2=600)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_se20_30"
    _c = _cfg(_n, (1, 5, 1), (20, 30, 2), (0.5, 5.0, 0.5), (1, 2, 1))
    log.info("A06  LE(1-5 s1) × SE(20-30 s2) × STP(0.5-5 s0.5) × LMT(1-2 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Zoom toward global best NP
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_zoom_best"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_step, _se_step, _stp_step, _lmt_step in [
        (3, 10, 1.5, 1, 1, 2, 0.5, 1),
        (2,  6, 1.0, 1, 1, 1, 0.2, 1),
        (2,  4, 0.8, 1, 1, 1, 0.2, 1),
        (1,  3, 0.5, 1, 1, 1, 0.2, 1),
    ]:
        _le  = zoom(best_le,  _r_le,  _le_step,  LE_LO,  LE_HI)
        _se  = zoom(best_se,  _r_se,  _se_step,  SE_LO,  SE_HI)
        _stp = zoom(best_stp, _r_stp, _stp_step, STP_LO, STP_HI)
        _lmt = zoom(best_lmt, _r_lmt, _lmt_step, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _le, _se, _stp, _lmt)
        if _c.total_runs() <= 5000:
            break
    log.info("A07  zoom(LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)  %d combos",
             best_le, best_se, best_stp, best_lmt, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  SE=5-30 × STP=0.5-3 comprehensive  (5×14×6×2=840)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_se5_30_comp"
    _c = _cfg(_n, (1, 5, 1), (5, 30, 2), (0.5, 3.0, 0.5), (1, 2, 1))
    log.info("A08  LE(1-5 s1) × SE(5-30 s2) × STP(0.5-3 s0.5) × LMT(1-2 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  Dense zoom around best after A08  (adaptive)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_dense_zoom"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_step, _se_step, _stp_step, _lmt_step in [
        (3,  8, 1.0, 1, 1, 1, 0.2, 1),
        (2,  5, 0.8, 1, 1, 1, 0.2, 1),
        (2,  4, 0.6, 1, 1, 1, 0.2, 1),
        (1,  3, 0.4, 1, 1, 1, 0.2, 1),
    ]:
        _le  = zoom(best_le,  _r_le,  _le_step,  LE_LO,  LE_HI)
        _se  = zoom(best_se,  _r_se,  _se_step,  SE_LO,  SE_HI)
        _stp = zoom(best_stp, _r_stp, _stp_step, STP_LO, STP_HI)
        _lmt = zoom(best_lmt, _r_lmt, _lmt_step, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _le, _se, _stp, _lmt)
        if _c.total_runs() <= 5000:
            break
    log.info("A09  dense_zoom(LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)  %d combos",
             best_le, best_se, best_stp, best_lmt, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  LMT=2-4 with best SE zone — test if LMT>1 ever works  (5×8×6×3=720)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_lmt_test"
    _c = _cfg(_n, (1, 5, 1), (5, 20, 3), (0.5, 3.0, 0.5), (1, 3, 1))
    log.info("A10  LE(1-5 s1) × SE(5-20 s3) × STP(0.5-3 s0.5) × LMT(1-3 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  LE=2 locked, SE/STP precision  (1×15×10×2=300 — wide SE range)
    # Uses _safe to expand fixed LE=2
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_le2_precision"
    _c = _cfg(_n, (2, 2, 1), (5, 25, 1), (0.2, 2.0, 0.2), (1, 2, 1))
    log.info("A11  LE(1-3 s1) × SE(5-25 s1) × STP(0.2-2 s0.2) × LMT(1-2 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A12  Boundary check — verify ceiling  (5×15×8×2=1200)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_ceiling_check"
    _c = _cfg(_n, (1, 5, 1), (5, 33, 2), (0.5, 4.0, 0.5), (1, 2, 1))
    log.info("A12  LE(1-5 s1) × SE(5-33 s2) × STP(0.5-4 s0.5) × LMT(1-2 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f", best_np)

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ROUND 3 COMPLETE")
    log.info("  Best NP  : %.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)
    log.info("  Target   : %.0f  Met: %s", TARGET_NP, "YES ★" if target_met else "NO")
    log.info("══════════════════════════════════════════════════════════════")
    save_json(best_entry if best_entry else attempt_log[-1], attempt_log, target_met)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _is_admin() -> bool:
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
    ap = argparse.ArgumentParser(description="CL Daily Breakout NP>700K Round-3 search")
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
