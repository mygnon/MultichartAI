"""
search_gc_daily4.py — Breakout Daily NP > 700,000 on CME.GC HOT, Round 4

Round-3 findings (2026-05-19):
  Best: LE=4  SE=49  STP=0.9  LMT=7  NP=312,520  MDD=-38,660  Obj=2,526,352  trades=33
  Gap to 700K: -387,480 (-55%)

R3 confirmed:
  LMT=7 is a sharp unique integer spike (LMT=6.5→277K, LMT=7.5→247K)
  STP=0.9 confirmed sweet spot (STP<0.7 worse, STP>1.1 worse)
  SE=45-53 flat plateau at 312K; SE=45-48 gives 309K
  LE=4 > LE=6(293K) > LE=3(286K) > LE=2(285K) > LE=5(283K)
  A07 zoom covered SE=45-53, LE=2-6 — the main known zone

R4 strategy — probe unexplored territory only:
  1. SE=55-120 survey    — beyond SE=53 plateau, totally unexplored
  2. Very high SE 120-250 — very long lookback regime
  3. Very tight STP 0.1-0.5 — below the explored STP floor
  4. High LE 7-20 survey   — breakout length > 6 days
  5. Low SE zone 5-35      — short-period regime
  6. LE=1-4 wide SE survey — low LE with diverse SE
  7. Very high LE 20-50    — trend-following long LE regime
  8. Adaptive zoom
  9. SE=54-80 fine         — just above the plateau
  10. Dense zoom
  11. STP fine 0.1-0.9 at best zone
  12. Wide boundary sweep  — catch anything missed

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  SE survey 55-120:  LE(2-6 s1) × SE(55-120 s10) × STP(0.7-1.0 s0.1) × LMT(6-9 s1)    5×7×4×4=560
  A02  Very high SE 120+: LE(1-5 s1) × SE(120-270 s25) × STP(0.7-1.0 s0.1) × LMT(5-10 s1)  5×7×4×6=840
  A03  Tight STP 0.1-0.5: LE(3-6 s1) × SE(45-55 s2)  × STP(0.1-0.5 s0.1) × LMT(6-9 s1)    4×6×5×4=480
  A04  High LE 7-20:      LE(7-20 s2) × SE(40-65 s5)  × STP(0.7-1.0 s0.1) × LMT(6-9 s1)    7×6×4×4=672
  A05  Low SE 5-35:       LE(2-6 s1) × SE(5-35 s5)   × STP(0.7-1.0 s0.1) × LMT(5-10 s1)   5×7×4×6=840
  A06  LE=1-4 wide SE:    LE(1-4 s1) × SE(20-80 s5)  × STP(0.7-1.0 s0.1) × LMT(5-10 s1)   4×13×4×6=1248
  A07  Very high LE 20-50:LE(20-50 s5)× SE(20-55 s5) × STP(0.7-1.0 s0.1) × LMT(5-10 s1)   7×8×4×6=1344
  A08  Adaptive zoom
  A09  SE=54-80 fine:     LE(3-6 s1) × SE(54-70 s2)  × STP(0.7-1.0 s0.1) × LMT(5-10 s1)   4×9×4×6=864
  A10  Dense zoom
  A11  STP fine 0.1-0.9:  LE(3-6 s1) × SE(45-53 s2)  × STP(0.1-0.9 s0.1) × LMT(6-9 s1)    4×5×9×4=720
  A12  Wide boundary:     LE(1-25 s4) × SE(5-200 s25) × STP(0.5-5.0 s1.0) × LMT(4-15 s1)   7×8×5×12=3360
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_daily4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.05, 500.0
LMT_LO, LMT_HI = 1.0,  500.0
SE_LO,  SE_HI  = 1.0,  500.0
LE_LO,  LE_HI  = 1.0,  100.0

# R3 best params (same as R2 — R3 confirmed ceiling)
SEED_LE,  SEED_SE  = 4.0, 49.0
SEED_STP, SEED_LMT = 0.9, 7.0
SEED_NP   = 312_520.0
SEED_OBJ  = 2_526_352.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_daily4_{int(time.time())}.log"
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
        name=f"GCD4_{name}",
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
    return OUTPUT_DIR / f"GCD4_{name}_raw.csv"


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
    log.info("=== Starting GCD4_%s (%d combos) ===", name, cfg.total_runs())
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
        "strategy":           "Breakout_GC_Daily  (target NP>700K round-4)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1 day)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "r3_best": {
            "LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "best_obj_attempt":      best_obj,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_gc_daily4.json"
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
    log.info("  GC Daily Breakout NP>700K Round-4 — unexplored territory")
    log.info("  R3 best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  Obj=%.0f",
             SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP, SEED_OBJ)
    log.info("  SE>53 unexplored; high LE unexplored; STP<0.6 unexplored")
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
    # A01  SE survey 55-120 step 10  (5×7×4×4=560)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_se_55_120"
    _c = _cfg(_n, (2, 6, 1), (55, 120, 10), (0.7, 1.0, 0.1), (6, 9, 1))
    log.info("A01  LE(2-6 s1) × SE(55-120 s10) × STP(0.7-1.0 s0.1) × LMT(6-9 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  Very high SE 120-270 step 25  (5×7×4×6=840)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_se_120_270"
    _c = _cfg(_n, (1, 5, 1), (120, 270, 25), (0.7, 1.0, 0.1), (5, 10, 1))
    log.info("A02  LE(1-5 s1) × SE(120-270 s25) × STP(0.7-1.0 s0.1) × LMT(5-10 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  Very tight STP 0.1-0.5  (4×6×5×4=480)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_tight_stp"
    _c = _cfg(_n, (3, 6, 1), (45, 55, 2), (0.1, 0.5, 0.1), (6, 9, 1))
    log.info("A03  LE(3-6 s1) × SE(45-55 s2) × STP(0.1-0.5 s0.1) × LMT(6-9 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  High LE 7-20 step 2  (7×6×4×4=672)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_high_le"
    _c = _cfg(_n, (7, 20, 2), (40, 65, 5), (0.7, 1.0, 0.1), (6, 9, 1))
    log.info("A04  LE(7-20 s2) × SE(40-65 s5) × STP(0.7-1.0 s0.1) × LMT(6-9 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  Low SE zone 5-35 step 5  (5×7×4×6=840)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_low_se"
    _c = _cfg(_n, (2, 6, 1), (5, 35, 5), (0.7, 1.0, 0.1), (5, 10, 1))
    log.info("A05  LE(2-6 s1) × SE(5-35 s5) × STP(0.7-1.0 s0.1) × LMT(5-10 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  LE=1-4 wide SE survey  (4×13×4×6=1248)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_le1_wide_se"
    _c = _cfg(_n, (1, 4, 1), (20, 80, 5), (0.7, 1.0, 0.1), (5, 10, 1))
    log.info("A06  LE(1-4 s1) × SE(20-80 s5) × STP(0.7-1.0 s0.1) × LMT(5-10 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Very high LE 20-50 step 5  (7×8×4×6=1344)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_very_high_le"
    _c = _cfg(_n, (20, 50, 5), (20, 55, 5), (0.7, 1.0, 0.1), (5, 10, 1))
    log.info("A07  LE(20-50 s5) × SE(20-55 s5) × STP(0.7-1.0 s0.1) × LMT(5-10 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A08  Adaptive zoom (progressive shrink from R4 best so far)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_zoom_best"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_step, _se_step, _stp_step, _lmt_step in [
        (3, 10, 0.5, 4, 1, 2, 0.1, 1),
        (2,  8, 0.4, 3, 1, 2, 0.1, 1),
        (2,  6, 0.3, 2, 1, 1, 0.1, 1),
        (1,  4, 0.2, 1, 1, 1, 0.1, 1),
    ]:
        _le8  = zoom(best_le,  _r_le,  _le_step,  LE_LO,  LE_HI)
        _se8  = zoom(best_se,  _r_se,  _se_step,  SE_LO,  SE_HI)
        _stp8 = zoom(best_stp, _r_stp, _stp_step, STP_LO, STP_HI)
        _lmt8 = zoom(best_lmt, _r_lmt, _lmt_step, LMT_LO, LMT_HI)
        _c = _cfg(_n, _le8, _se8, _stp8, _lmt8)
        if _c.total_runs() <= 5000:
            break
    log.info("A08  Zoom: LE=%s SE=%s STP=%s LMT=%s  %d combos",
             _le8, _se8, _stp8, _lmt8, _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  SE=54-80 fine step 2  (4×9×4×6=864)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_se_54_80"
    _c = _cfg(_n, (3, 6, 1), (54, 70, 2), (0.7, 1.0, 0.1), (5, 10, 1))
    log.info("A09  LE(3-6 s1) × SE(54-70 s2) × STP(0.7-1.0 s0.1) × LMT(5-10 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  Dense zoom (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_dense_zoom"
    for _r_le, _r_se, _r_stp, _r_lmt, _le_step, _se_step, _stp_step, _lmt_step in [
        (3, 10, 0.5, 4, 1, 2, 0.1, 1),
        (2,  8, 0.4, 3, 1, 2, 0.1, 1),
        (2,  6, 0.3, 2, 1, 1, 0.1, 1),
        (1,  4, 0.2, 1, 1, 1, 0.1, 1),
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
    # A11  STP fine 0.1-0.9 at best zone  (4×5×9×4=720)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_stp_fine"
    _c = _cfg(_n, (3, 6, 1), (45, 53, 2), (0.1, 0.9, 0.1), (6, 9, 1))
    log.info("A11  LE(3-6 s1) × SE(45-53 s2) × STP(0.1-0.9 s0.1) × LMT(6-9 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A12  Wide boundary sweep  (7×8×5×12=3360)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_wide_boundary"
    _c = _cfg(_n, (1, 25, 4), (5, 200, 25), (0.5, 5.0, 1.0), (4, 15, 1))
    log.info("A12  LE(1-25 s4) × SE(5-200 s25) × STP(0.5-5.0 s1.0) × LMT(4-15 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    if not best_entry:
        best_entry = _entry(0, "r3_seed", None,
                            SEED_LE, SEED_SE, SEED_STP, SEED_LMT,
                            SEED_OBJ, SEED_NP, 0.0, 0, False, 0)

    log.info("")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL — GC Daily Breakout NP>700K Round-4")
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
    ap = argparse.ArgumentParser(description="GC Daily Breakout NP>700K Round-4 search")
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
