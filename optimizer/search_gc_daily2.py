"""
search_gc_daily2.py — Breakout Daily NP > 700,000 on CME.GC HOT, Round 2

Round-1 findings (2026-05-19):
  Best NP:  LE=4  SE=50  STP=1.05  LMT=7   NP=306,410  MDD=-38,660  Obj=2,428,533  trades=32
  Best Obj: LE=2  SE=50  STP=1.05  LMT=5   NP=271,390  MDD=-29,970  Obj=2,457,542  trades=42
  Gap to 700K: -393,590 (-56%)

Key R1 insights:
  SE=50-58:  Identical results — plateau at SE=50 (same trades regardless)
  SE=25-50:  Rising zone (289K→296K); SE<25 drops to ~200K
  STP=3+:    Same result for LE=1 (stop never hit above STP≈3 for LE=1)
  STP=1.05:  Tight stop selectively cuts losers → 306K vs 296K (3.3% gain)
  STP=0.05:  Too tight, gets stopped out → 259K
  LMT=7:     Best for NP with LE=4 (vs LMT=5 best for Obj with LE=2)
  Trades:    Only 20-42 per 7 years (2-6 per year)
  MDD:       Only -38K to -50K (very low drawdown)

Round-2 strategy — fine-tune the critical STP zone and SE below-50 region:
  1. Fine STP 0.1-3.0 step 0.1 around the sweet spot
  2. SE below 50 fine step 1 (rising zone 30-50)
  3. Fine LMT 3-20 step 1 with optimal LE/SE/STP
  4. LE fine 1-8 step 1 with optimal SE/STP/LMT
  5. Ultra-fine: LE=4 + SE=48-52 + STP=0.5-2.0 + LMT=5-15
  6. Try very high SE (55-150 step 5) — A08 R1 failed at 100-250 step 25
  7. Explore if STP<0.05 (sub-tick stop) changes anything
  8. Wide STP 3-30 to confirm the dead zone

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  Fine STP:    LE(2-6 s1) × SE(45-55 s2) × STP(0.1-3.0 s0.1) × LMT(5-11 s2)   5×6×30×4=3600
  A02  SE fine:     LE(3-5 s1) × SE(30-50 s1) × STP(0.5-2.0 s0.5) × LMT(5-11 s2)   3×21×4×4=1008
  A03  LMT fine:    LE(3-5 s1) × SE(45-55 s2) × STP(0.5-1.5 s0.25) × LMT(3-20 s1)  3×6×5×18=1620
  A04  LE fine:     LE(1-8 s1) × SE(45-55 s2) × STP(0.5-1.5 s0.25) × LMT(5-11 s2)  8×6×5×4=960
  A05  Ultrafine:   LE(2-6 s1) × SE(46-52 s1) × STP(0.5-2.0 s0.25) × LMT(5-11 s2)  5×7×7×4=980
  A06  High SE:     LE(2-6 s1) × SE(52-100 s5) × STP(0.5-2.0 s0.5) × LMT(5-11 s2)  5×10×4×4=800
  A07  Adaptive zoom
  A08  STP dead zone: LE(2-6 s1) × SE(45-55 s2) × STP(3-30 s3) × LMT(5-11 s2)      5×6×10×4=1200
  A09  SE=35-50 fine: LE(2-6 s1) × SE(35-50 s1) × STP(0.5-1.5 s0.25) × LMT(5-11 s2) 5×16×5×4=1600
  A10  Dense zoom (adaptive)
  A11  LE=4 deep:   LE(3-6 s1) × SE(40-55 s1) × STP(0.3-1.8 s0.3) × LMT(5-15 s2)  4×16×6×6=2304
  A12  Wide explore: LE(1-10 s2) × SE(15-55 s5) × STP(0.5-5.0 s0.5) × LMT(3-25 s2) 5×9×10×12=5400 — trim
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.05, 500.0
LMT_LO, LMT_HI = 1.0,  500.0
SE_LO,  SE_HI  = 1.0,  500.0
LE_LO,  LE_HI  = 1.0,  100.0

# R1 best NP params
SEED_LE,  SEED_SE  = 4.0, 50.0
SEED_STP, SEED_LMT = 1.05, 7.0
SEED_NP   = 306_410.0
SEED_OBJ  = 2_428_533.0

# R1 best Obj params
SEED_OBJ_LE,  SEED_OBJ_SE  = 2.0, 50.0
SEED_OBJ_STP, SEED_OBJ_LMT = 1.05, 5.0
SEED_OBJ_NP   = 271_390.0
SEED_OBJ_OBJ  = 2_457_542.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_daily2_{int(time.time())}.log"
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
        name=f"GCD2_{name}",
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
    return OUTPUT_DIR / f"GCD2_{name}_raw.csv"


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
    log.info("=== Starting GCD2_%s (%d combos) ===", name, cfg.total_runs())
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
        "strategy":           "Breakout_GC_Daily  (target NP>700K round-2)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1 day)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "r1_best_np": {
            "LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "r1_best_obj": {
            "LE": SEED_OBJ_LE, "SE": SEED_OBJ_SE,
            "STP": SEED_OBJ_STP, "LMT": SEED_OBJ_LMT,
            "net_profit": SEED_OBJ_NP, "objective": SEED_OBJ_OBJ,
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "best_obj_attempt":      best_obj,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_gc_daily2.json"
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
    log.info("  GC Daily Breakout NP>700K Round-2 — fine-tuning STP and SE")
    log.info("  R1 best NP:  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP)
    log.info("  R1 best Obj: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  Obj=%.0f",
             SEED_OBJ_LE, SEED_OBJ_SE, SEED_OBJ_STP, SEED_OBJ_LMT,
             SEED_OBJ_NP, SEED_OBJ_OBJ)
    log.info("  Target: %.0f  Gap: %.0f (-%.1f%%)",
             TARGET_NP, TARGET_NP - SEED_NP,
             (TARGET_NP - SEED_NP) / SEED_NP * 100)
    log.info("  Focus: STP=0.1-3.0 fine, SE=30-55 step 1, LMT and LE fine-tune")
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
    # A01  Fine STP 0.1-3.0 step 0.1  (5×6×30×4=3600)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_stp"
    _c = _cfg(_n, (2, 6, 1), (45, 55, 2), (0.1, 3.0, 0.1), (5, 11, 2))
    log.info("A01  LE(2-6 s1) × SE(45-55 s2) × STP(0.1-3.0 s0.1) × LMT(5-11 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  SE fine 30-50 step 1  (3×21×4×4=1008)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_se_fine"
    _c = _cfg(_n, (3, 5, 1), (30, 50, 1), (0.5, 2.0, 0.5), (5, 11, 2))
    log.info("A02  LE(3-5 s1) × SE(30-50 s1) × STP(0.5-2.0 s0.5) × LMT(5-11 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  LMT fine 3-20 step 1  (3×6×5×18=1620)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_lmt_fine"
    _c = _cfg(_n, (3, 5, 1), (45, 55, 2), (0.5, 1.5, 0.25), (3, 20, 1))
    log.info("A03  LE(3-5 s1) × SE(45-55 s2) × STP(0.5-1.5 s0.25) × LMT(3-20 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  LE fine 1-8 step 1  (8×6×5×4=960)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_le_fine"
    _c = _cfg(_n, (1, 8, 1), (45, 55, 2), (0.5, 1.5, 0.25), (5, 11, 2))
    log.info("A04  LE(1-8 s1) × SE(45-55 s2) × STP(0.5-1.5 s0.25) × LMT(5-11 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  Ultrafine LE=2-6 × SE=46-52 step 1  (5×7×7×4=980)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ultrafine"
    _c = _cfg(_n, (2, 6, 1), (46, 52, 1), (0.5, 2.0, 0.25), (5, 11, 2))
    log.info("A05  LE(2-6 s1) × SE(46-52 s1) × STP(0.5-2.0 s0.25) × LMT(5-11 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  High SE 55-100 step 5  (5×10×4×4=800)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_se"
    _c = _cfg(_n, (2, 6, 1), (55, 100, 5), (0.5, 2.0, 0.5), (5, 11, 2))
    log.info("A06  LE(2-6 s1) × SE(55-100 s5) × STP(0.5-2.0 s0.5) × LMT(5-11 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Adaptive zoom around R2 best  (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_zoom_best"
    for _r_se, _r_stp, _r_lmt, _stp_step, _lmt_step in [
        (8, 1.0, 8, 0.1, 1),
        (6, 0.8, 6, 0.1, 1),
        (4, 0.6, 4, 0.1, 1),
        (3, 0.4, 3, 0.1, 1),
    ]:
        _le7  = zoom(best_le,  2,        1,          LE_LO,  LE_HI)
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
    # A08  STP dead zone 3-30  (5×6×10×4=1200)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_stp_dead_zone"
    _c = _cfg(_n, (2, 6, 1), (45, 55, 2), (3, 30, 3), (5, 11, 2))
    log.info("A08  LE(2-6 s1) × SE(45-55 s2) × STP(3-30 s3) × LMT(5-11 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  SE=35-50 fine step 1  (5×16×5×4=1600)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_se_35_50_fine"
    _c = _cfg(_n, (2, 6, 1), (35, 50, 1), (0.5, 1.5, 0.25), (5, 11, 2))
    log.info("A09  LE(2-6 s1) × SE(35-50 s1) × STP(0.5-1.5 s0.25) × LMT(5-11 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  Dense zoom (adaptive, progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_dense_zoom"
    for _r_se, _r_stp, _r_lmt, _stp_step, _lmt_step in [
        (8, 1.0, 8, 0.1, 1),
        (6, 0.8, 6, 0.1, 1),
        (4, 0.6, 4, 0.1, 1),
        (3, 0.4, 3, 0.1, 1),
    ]:
        _le10  = zoom(best_le,  2,        1,          LE_LO,  LE_HI)
        _se10  = zoom(best_se,  _r_se,    1,          SE_LO,  SE_HI)
        _stp10 = zoom(best_stp, _r_stp,   _stp_step,  STP_LO, STP_HI)
        _lmt10 = zoom(best_lmt, _r_lmt,   _lmt_step,  LMT_LO, LMT_HI)
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
    # A11  LE=4 deep: SE=40-55 step 1  (4×16×6×6=2304)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_le4_deep"
    _c = _cfg(_n, (3, 6, 1), (40, 55, 1), (0.3, 1.8, 0.3), (5, 15, 2))
    log.info("A11  LE(3-6 s1) × SE(40-55 s1) × STP(0.3-1.8 s0.3) × LMT(5-15 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A12  Wide boundary check  (5×9×10×9=4050 — verify within limit)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_wide_boundary"
    _c = _cfg(_n, (1, 9, 2), (15, 55, 5), (0.5, 5.0, 0.5), (3, 19, 2))
    log.info("A12  LE(1-9 s2) × SE(15-55 s5) × STP(0.5-5.0 s0.5) × LMT(3-19 s2)  %d combos",
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
    log.info("  FINAL — GC Daily Breakout NP>700K Round-2")
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
    ap = argparse.ArgumentParser(description="GC Daily Breakout NP>700K Round-2 search")
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
