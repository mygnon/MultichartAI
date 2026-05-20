"""
search_gc_hourly.py — Breakout Hourly NP > 700,000 on CME.GC HOT, Round 1

No prior knowledge for GC hourly. Wide coarse scan to establish parameter ranges.
Strategy: _2021Basic_Break_NQ on Gold futures (CME.GC HOT), hourly bars.
Insample: 2019/01/01 – 2026/01/01. Objective = NetProfit² / |MaxDrawdown|.

GC contract: 100 troy oz, tick=0.10=$10/contract.
Typical hourly ATR ~1–5 pts ($100–500/contract).

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  Wide coarse:  LE(1-4) × SE(5-55 s5)  × STP(1-10 s1)   × LMT(2-22 s2)      4×11×10×11=4840
  A02  Low SE:       LE(1-4) × SE(3-20 s2)  × STP(1-8  s1)   × LMT(2-20 s2)      4×9×8×10  =2880
  A03  Sub-1 STP:    LE(1-4) × SE(5-30 s5)  × STP(0.2-2 s0.2)× LMT(2-20 s2)      4×6×10×10 =2400
  A04  High LMT:     LE(1-4) × SE(5-30 s5)  × STP(1-8  s1)   × LMT(20-60 s5)     4×6×8×9   =1728
  A05  High SE:      LE(1-4) × SE(50-200 s25)× STP(1-8 s1)   × LMT(5-30 s5)      4×7×8×6   =1344
  A06  Large STP:    LE(1-4) × SE(5-30 s5)  × STP(10-50 s5)  × LMT(10-50 s5)     4×6×9×9   =1944
  A07  Zoom best (adaptive progressive shrink around best_np params)
  A08  Fine STP (step 0.1) around best
  A09  Fine LMT (step 0.5) around best
  A10  LE range 1-10 with best SE/STP/LMT
  A11  Dense zoom (progressive shrink, ≤5000)
  A12  Global boundary: LE(1-5) × SE(5-205 s25) × STP(0.5-50 s5) × LMT(2-100 s10)  5×9×10×10=4500
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
SYMBOL     = "CME.GC HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_hourly_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.05, 200.0
LMT_LO, LMT_HI = 0.5,  500.0
SE_LO,  SE_HI  = 3.0,  500.0
LE_LO,  LE_HI  = 1.0,  300.0

# No prior knowledge for GC — start from scratch
SEED_LE,  SEED_SE  = 3.0,  20.0
SEED_STP, SEED_LMT = 3.0,  10.0
SEED_NP   = 0.0
SEED_OBJ  = 0.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_hourly_{int(time.time())}.log"
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
        name=f"GCH_{name}",
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
    return OUTPUT_DIR / f"GCH_{name}_raw.csv"


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
    log.info("=== Starting GCH_%s (%d combos) ===", name, cfg.total_runs())
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
    # target-chasing mode: zoom toward NP-max, not Obj-max
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
        "strategy":           "Breakout_GC_Hourly  (target NP>700K round-1)",
        "symbol":             SYMBOL,
        "timeframe":          "Hourly (60 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "best_params":           best,
        "best_np_attempt":       best_np,
        "best_obj_attempt":      best_obj,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_gc_hourly.json"
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
    log.info("  GC Hourly Breakout NP>700K USD Round-1")
    log.info("  Symbol: %s  Timeframe: Hourly  Insample: 2019–2026", SYMBOL)
    log.info("  No prior knowledge — wide coarse scan")
    log.info("  Target: %.0f USD", TARGET_NP)
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
        log.info("       Global best NP=%.0f%s",
                 best_np,
                 ("  (need +%.1f%%)" % ((TARGET_NP - best_np) / best_np * 100))
                 if best_np > 0 else "")

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  Wide coarse scan (4×11×10×11=4840)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_coarse_scan"
    _c = _cfg(_n, (1, 4, 1), (5, 55, 5), (1.0, 10.0, 1.0), (2.0, 22.0, 2.0))
    log.info("A01  LE(1-4) × SE(5-55 s5) × STP(1-10 s1) × LMT(2-22 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  Low SE focus (SE=3-20, like NQ hourly's SE=9 sweet spot)  (4×9×8×10=2880)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_low_se"
    _c = _cfg(_n, (1, 4, 1), (3, 19, 2), (1.0, 8.0, 1.0), (2.0, 20.0, 2.0))
    log.info("A02  LE(1-4) × SE(3-19 s2) × STP(1-8 s1) × LMT(2-20 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  Sub-1 STP (fractional stops may be key for Gold)  (4×6×10×10=2400)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_sub_stp"
    _c = _cfg(_n, (1, 4, 1), (5, 30, 5), (0.2, 2.0, 0.2), (2.0, 20.0, 2.0))
    log.info("A03  LE(1-4) × SE(5-30 s5) × STP(0.2-2 s0.2) × LMT(2-20 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  High LMT zone (larger profit targets for Gold)  (4×6×8×9=1728)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_high_lmt"
    _c = _cfg(_n, (1, 4, 1), (5, 30, 5), (1.0, 8.0, 1.0), (20.0, 60.0, 5.0))
    log.info("A04  LE(1-4) × SE(5-30 s5) × STP(1-8 s1) × LMT(20-60 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  High SE (SE=50-200, weekly-scale breakout)  (4×7×8×6=1344)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_high_se"
    _c = _cfg(_n, (1, 4, 1), (50, 200, 25), (1.0, 8.0, 1.0), (5.0, 30.0, 5.0))
    log.info("A05  LE(1-4) × SE(50-200 s25) × STP(1-8 s1) × LMT(5-30 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  Large STP (wide stops for Gold volatility)  (4×6×9×9=1944)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_large_stp"
    _c = _cfg(_n, (1, 4, 1), (5, 30, 5), (10.0, 50.0, 5.0), (10.0, 50.0, 5.0))
    log.info("A06  LE(1-4) × SE(5-30 s5) × STP(10-50 s5) × LMT(10-50 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Zoom around best NP from A01-A06 (adaptive progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_zoom_best"
    for _r_se, _r_stp, _r_lmt, _stp_step, _lmt_step in [
        (10, 5.0, 8.0, 0.5, 1.0),
        (7,  4.0, 6.0, 0.5, 1.0),
        (5,  3.0, 4.0, 0.5, 1.0),
        (3,  2.0, 3.0, 0.5, 1.0),
    ]:
        _le7  = zoom(best_le,  1,        1,          LE_LO,  LE_HI)
        _se7  = zoom(best_se,  _r_se,    1,          SE_LO,  SE_HI)
        _stp7 = zoom(best_stp, _r_stp,   _stp_step,  STP_LO, STP_HI)
        _lmt7 = zoom(best_lmt, _r_lmt,   _lmt_step,  LMT_LO, LMT_HI)
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
    # A08  Fine STP (step 0.1) around best STP
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_fine_stp"
    for _r_se, _r_stp, _r_lmt in [(5, 1.5, 4), (4, 1.0, 3), (3, 0.8, 2)]:
        _le8  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se8  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp8 = zoom(best_stp, _r_stp, 0.1,  STP_LO, STP_HI)
        _lmt8 = zoom(best_lmt, _r_lmt, 1.0,  LMT_LO, LMT_HI)
        _c = _cfg(_n, _le8, _se8, _stp8, _lmt8)
        if _c.total_runs() <= 5000:
            break
    log.info("A08  Fine STP: LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le8, _se8, _stp8, _lmt8, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  Fine LMT (step 0.5) around best LMT
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_fine_lmt"
    for _r_se, _r_stp, _r_lmt in [(5, 2.0, 5), (4, 1.5, 4), (3, 1.0, 3)]:
        _le9  = zoom(best_le,  1,      1,    LE_LO,  LE_HI)
        _se9  = zoom(best_se,  _r_se,  1,    SE_LO,  SE_HI)
        _stp9 = zoom(best_stp, _r_stp, 0.5,  STP_LO, STP_HI)
        _lmt9 = zoom(best_lmt, _r_lmt, 0.5,  LMT_LO, LMT_HI)
        _c = _cfg(_n, _le9, _se9, _stp9, _lmt9)
        if _c.total_runs() <= 5000:
            break
    log.info("A09  Fine LMT: LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le9, _se9, _stp9, _lmt9, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  LE range 1-10 with best SE/STP/LMT
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_le_range"
    for _r_se, _r_stp, _r_lmt in [(6, 2.0, 4), (4, 1.5, 3), (3, 1.0, 2)]:
        _le10  = (1, min(10, int(LE_HI)), 1)
        _se10  = zoom(best_se,  _r_se,  1,    SE_LO, SE_HI)
        _stp10 = zoom(best_stp, _r_stp, 0.5,  STP_LO, STP_HI)
        _lmt10 = zoom(best_lmt, _r_lmt, 1.0,  LMT_LO, LMT_HI)
        _c = _cfg(_n, _le10, _se10, _stp10, _lmt10)
        if _c.total_runs() <= 5000:
            break
    log.info("A10  LE range: LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le10, _se10, _stp10, _lmt10, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt, _stp_step, _lmt_step in [
        (4, 1.5, 5.0, 0.25, 0.5),
        (3, 1.0, 4.0, 0.25, 0.5),
        (2, 0.75, 3.0, 0.25, 0.5),
        (2, 0.5,  2.0, 0.25, 0.5),
    ]:
        _le11  = zoom(best_le,  1,        1,           LE_LO,  LE_HI)
        _se11  = zoom(best_se,  _r_se,    1,           SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, _r_stp,   _stp_step,   STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, _r_lmt,   _lmt_step,   LMT_LO, LMT_HI)
        _c = _cfg(_n, _le11, _se11, _stp11, _lmt11)
        if _c.total_runs() <= 5000:
            break
    log.info("A11  Dense: LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le11, _se11, _stp11, _lmt11, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A12  Global boundary check  (5×9×10×10=4500)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_boundary"
    _c = _cfg(_n, (1, 5, 1), (5, 205, 25), (0.5, 50.0, 5.5), (2.0, 100.0, 10.0))
    log.info("A12  LE(1-5) × SE(5-205 s25) × STP(0.5-50 s5.5) × LMT(2-100 s10)  %d combos",
             _c.total_runs())
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
    log.info("  FINAL — GC Hourly Breakout NP>700K Round-1")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Best NP: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
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
    print(f"Target NP>700K: {'MET ✓' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
    ap = argparse.ArgumentParser(description="GC Hourly Breakout NP>700K Round-1 search")
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
