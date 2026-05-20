"""
search_zw_hourly2.py — Breakout Hourly NP > 700,000 on CBOT.ZW HOT, Round 2

R1 findings:
  - Zero profitable combinations across 12 attempts (~14,000+ combos)
  - SE direction is the only signal: SE=150 gave best NP=-19,418
  - SE=80→100→150: NP=-30K→-21K→-19K  (monotonically improving at SE boundary)
  - R1 covered SE=5-150; R2 extends to SE=150-500

R2 strategy: chase the high-SE direction and verify if any SE>150 breaks into positive NP.

CBOT.ZW HOT context:
  5,000 bushels/contract
  Quoted in cents/bushel
  1 point = 1 cent/bu × 5,000 = $50/contract
  SE=150 bars ≈ ~2.5 weeks of hourly data
  SE=300 bars ≈ ~5 weeks; SE=500 ≈ ~8 weeks
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
SYMBOL     = "CBOT.ZW HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\zw_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.01, 2000.0
LMT_LO, LMT_HI = 1.0,  2000.0
SE_LO,  SE_HI  = 1.0,   500.0
LE_LO,  LE_HI  = 1.0,   100.0

# Seed from R1 best (least negative): LE=1 SE=150 STP=25 LMT=30 NP=-19,418
SEED_LE,  SEED_SE  = 1.0, 150.0
SEED_STP, SEED_LMT = 25.0, 30.0
SEED_NP   = 0.0
SEED_OBJ  = 0.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_zw_hourly2_{int(time.time())}.log"
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
        name=f"ZWH2_{name}",
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
    return OUTPUT_DIR / f"ZWH2_{name}_raw.csv"


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
    log.info("=== Starting ZWH2_%s (%d combos) ===", name, cfg.total_runs())
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
        "strategy":           "Breakout_ZW_Hourly  (target NP>700K round-2)",
        "symbol":             SYMBOL,
        "timeframe":          "Hourly (60 min)",
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
    out = OUTPUT_DIR / "final_params_zw_hourly2.json"
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
    log.info("  ZW Hourly Breakout NP>700K Round-2 — high SE extension")
    log.info("  Symbol: %s  Timeframe: hourly (60 min)", SYMBOL)
    log.info("  R1 seed: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g", SEED_LE, SEED_SE, SEED_STP, SEED_LMT)
    log.info("  R1 best NP=-19,418 at SE=150 (boundary); extending SE=150-500")
    log.info("  ZW: 1 pt = 1¢/bu × 5000 bu = $50/contract")
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
        log.info("       Global best NP=%.0f (need +%.1f%%)",
                 best_np, (TARGET_NP - best_np) / best_np * 100 if best_np > 0 else 999)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  Very high SE: 150-300  (4×16×6×6=2304)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_very_high_se"
    _c = _cfg(_n, (1, 4, 1), (150, 300, 10), (5.0, 30.0, 5.0), (10, 60, 10))
    log.info("A01  LE(1-4 s1) × SE(150-300 s10) × STP(5-30 s5) × LMT(10-60 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  Ultra high SE: 200-500  (4×16×5×5=1600)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ultra_high_se"
    _c = _cfg(_n, (1, 4, 1), (200, 500, 20), (5.0, 25.0, 5.0), (10, 50, 10))
    log.info("A02  LE(1-4 s1) × SE(200-500 s20) × STP(5-25 s5) × LMT(10-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  SE=150 fine STP  (4×16×8×5=2560)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_se150_stp_fine"
    _c = _cfg(_n, (1, 4, 1), (120, 270, 10), (2.0, 30.0, 4.0), (10, 50, 10))
    log.info("A03  LE(1-4 s1) × SE(120-270 s10) × STP(2-30 s4) × LMT(10-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  High SE + high LMT  (4×16×4×8=2048)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_high_se_high_lmt"
    _c = _cfg(_n, (1, 4, 1), (150, 300, 10), (10.0, 40.0, 10.0), (30, 100, 10))
    log.info("A04  LE(1-4 s1) × SE(150-300 s10) × STP(10-40 s10) × LMT(30-100 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  SE=150 tight step, confirm plateau  (3×16×6×5=1440)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_se150_plateau"
    _c = _cfg(_n, (1, 3, 1), (130, 280, 10), (5.0, 30.0, 5.0), (15, 55, 10))
    log.info("A05  LE(1-3 s1) × SE(130-280 s10) × STP(5-30 s5) × LMT(15-55 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  SE=300-500 large STP  (4×11×5×6=1320)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_se300_plus"
    _c = _cfg(_n, (1, 4, 1), (300, 500, 20), (5.0, 25.0, 5.0), (10, 60, 10))
    log.info("A06  LE(1-4 s1) × SE(300-500 s20) × STP(5-25 s5) × LMT(10-60 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Zoom toward best NP (if any found)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_zoom_best"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_step, _se_step, _stp_step, _lmt_step in [
        (3, 60, 10.0, 20, 1, 10, 2.0, 5),
        (2, 40, 8.0,  15, 1,  8, 2.0, 5),
        (2, 30, 6.0,  10, 1,  5, 1.0, 5),
        (1, 20, 4.0,   8, 1,  4, 1.0, 5),
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
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  LE sweep at high SE  (8×11×5×5=2200)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_le_sweep_high_se"
    _c = _cfg(_n, (1, 8, 1), (130, 230, 10), (5.0, 25.0, 5.0), (10, 50, 10))
    log.info("A08  LE(1-8 s1) × SE(130-230 s10) × STP(5-25 s5) × LMT(10-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  Very high LMT 100-300  (3×16×5×5=1200)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_very_high_lmt"
    _c = _cfg(_n, (1, 3, 1), (150, 300, 10), (5.0, 25.0, 5.0), (50, 200, 25))
    log.info("A09  LE(1-3 s1) × SE(150-300 s10) × STP(5-25 s5) × LMT(50-200 s25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  Dense zoom around best
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_dense_zoom"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_step, _se_step, _stp_step, _lmt_step in [
        (3, 50, 8.0, 15, 1, 5, 2.0, 5),
        (2, 30, 5.0, 10, 1, 5, 1.0, 5),
        (2, 20, 4.0,  8, 1, 4, 1.0, 5),
        (1, 15, 3.0,  5, 1, 3, 1.0, 5),
    ]:
        _le  = zoom(best_le,  _r_le,  _le_step,  LE_LO,  LE_HI)
        _se  = zoom(best_se,  _r_se,  _se_step,  SE_LO,  SE_HI)
        _stp = zoom(best_stp, _r_stp, _stp_step, STP_LO, STP_HI)
        _lmt = zoom(best_lmt, _r_lmt, _lmt_step, LMT_LO, LMT_HI)
        _c   = _cfg(_n, _le, _se, _stp, _lmt)
        if _c.total_runs() <= 5000:
            break
    log.info("A10  dense_zoom(LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)  %d combos",
             best_le, best_se, best_stp, best_lmt, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  LE=1 full SE sweep  (1 s1)×(100-500 s25)×(5-30 s5)×(10-60 s10)
    #       = 1×17×6×6=612
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_le1_full_se"
    _c = _cfg(_n, (1, 2, 1), (100, 500, 25), (5.0, 30.0, 5.0), (10, 60, 10))
    log.info("A11  LE(1-2 s1) × SE(100-500 s25) × STP(5-30 s5) × LMT(10-60 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A12  Wide boundary high SE  (5×9×7×6=1890)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_wide_high_se"
    _c = _cfg(_n, (1, 5, 1), (150, 500, 50), (5.0, 35.0, 5.0), (10, 60, 10))
    log.info("A12  LE(1-5 s1) × SE(150-500 s50) × STP(5-35 s5) × LMT(10-60 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ROUND 2 COMPLETE")
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
    ap = argparse.ArgumentParser(description="ZW Hourly Breakout NP>700K Round-2 search")
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
