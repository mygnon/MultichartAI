"""
search_gc_hourly4.py — Breakout Hourly NP > 700,000 on CME.GC HOT, Round 4

Round-3 findings (2026-05-19):
  Best NP:  LE=2   SE=35  STP=3.2  LMT=20  NP=292,030  MDD=-62,570  obj=1,362,978  trades=2109
  Best Obj: LE=16  SE=34  STP=3.6  LMT=22  NP=268,050  MDD=-46,150  obj=1,556,897  trades=1138
  Gap to 700K: -407,970 (-58%)

LE distribution (A09 R3):
  LE=16 → max 268K  (best in high-LE regime)
  LE=13 → max 252K
  LE=10 → max 231K
  LE=19 → max 246K
  LE=22 → max 201K (drops off)

LE distribution (A10 R3):
  LE=2  → max 292K  (global best NP)
  LE=1  → max 284K  (close second)
  LE=3  → max 241K  (drops sharply)
  LE=7-10 → 233-234K

High SE (A08 R3): SE=90 + LMT=15 gave 276K best. LMT much lower than low-SE optimum.

Never explored territory:
  - LE=16 + high SE (60-90)
  - Very high LE (>25)
  - Very low STP (0.5-2.0)
  - High LMT (>24) with LE=2
  - LE=16 combined with wide LMT range

Round-4 strategy — exhaust remaining unexplored regions:
  1. Fine zoom LE=16 regime (best Obj)
  2. LE=16 + high SE (not yet tested)
  3. LE=16 with wide STP and LMT ranges
  4. Very high LE (25-50, may have third peak)
  5. Very low STP (0.5-2.0) — never probed
  6. High LMT (24-50) with LE=2 seed
  7. LE survey across 1-50 step 5

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  LE=16 fine:     LE(13-19 s2) × SE(30-40 s2) × STP(3.0-4.2 s0.4) × LMT(20-26 s1)   4×6×4×7=672
  A02  LE=16+high SE:  LE(13-19 s2) × SE(55-90 s5) × STP(3.0-5.0 s0.5) × LMT(18-26 s2)   4×8×5×5=800
  A03  LE=16 fine STP: LE(13-19 s2) × SE(30-38 s2) × STP(2.4-5.2 s0.4) × LMT(20-26 s1)   4×5×8×7=1120
  A04  Very high LE:   LE(25-49 s4) × SE(30-38 s2) × STP(3.2-4.4 s0.4) × LMT(20-26 s2)   7×5×4×4=560
  A05  Low STP:        LE(1-3)      × SE(33-37 s2) × STP(0.5-2.5 s0.5)  × LMT(18-24 s1)   3×3×5×7=315
  A06  High LMT:       LE(1-3)      × SE(33-37 s2) × STP(2.8-4.0 s0.4)  × LMT(24-50 s2)   3×3×4×14=504
  A07  Adaptive zoom (NP-max seed)
  A08  LE=16 high LMT: LE(13-19 s2) × SE(30-40 s2) × STP(3.2-4.4 s0.4) × LMT(24-36 s2)   4×6×4×7=672
  A09  LE=1-2 wide SE: LE(1-2)      × SE(40-70 s3) × STP(3.0-5.0 s0.5)  × LMT(18-26 s2)   2×11×5×5=550
  A10  LE survey:      LE(1-50 s5)  × SE(32-38 s2) × STP(3.2-4.0 s0.4)  × LMT(20-24 s2)   10×4×3×3=360
  A11  Dense zoom (adaptive)
  A12  LE=16 low STP:  LE(13-19 s2) × SE(30-40 s2) × STP(1.0-3.0 s0.5)  × LMT(20-28 s2)   4×6×5×5=600
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_hourly4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.05, 100.0
LMT_LO, LMT_HI = 0.5,  500.0
SE_LO,  SE_HI  = 3.0,  300.0
LE_LO,  LE_HI  = 1.0,  100.0

# R3 best NP params (global seed)
SEED_LE,  SEED_SE  = 2.0, 35.0
SEED_STP, SEED_LMT = 3.2, 20.0
SEED_NP   = 292_030.0
SEED_OBJ  = 1_362_978.0

# R3 best Obj params (high-LE regime)
SEED_HIGH_LE,  SEED_HIGH_SE  = 16.0, 34.0
SEED_HIGH_STP, SEED_HIGH_LMT = 3.6,  22.0
SEED_HIGH_NP   = 268_050.0
SEED_HIGH_OBJ  = 1_556_897.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_hourly4_{int(time.time())}.log"
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
        name=f"GCH4_{name}",
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
    return OUTPUT_DIR / f"GCH4_{name}_raw.csv"


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
    log.info("=== Starting GCH4_%s (%d combos) ===", name, cfg.total_runs())
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
        "strategy":           "Breakout_GC_Hourly  (target NP>700K round-4)",
        "symbol":             SYMBOL,
        "timeframe":          "Hourly (60 min)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "r3_best_np": {
            "LE": SEED_LE, "SE": SEED_SE, "STP": SEED_STP, "LMT": SEED_LMT,
            "net_profit": SEED_NP, "objective": SEED_OBJ,
        },
        "r3_best_obj": {
            "LE": SEED_HIGH_LE, "SE": SEED_HIGH_SE,
            "STP": SEED_HIGH_STP, "LMT": SEED_HIGH_LMT,
            "net_profit": SEED_HIGH_NP, "objective": SEED_HIGH_OBJ,
        },
        "best_params":           best,
        "best_np_attempt":       best_np,
        "best_obj_attempt":      best_obj,
        "attempts_above_target": above,
        "attempt_log":           log_,
    }
    out = OUTPUT_DIR / "final_params_gc_hourly4.json"
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
    log.info("  GC Hourly Breakout NP>700K Round-4 — exhausting remaining territory")
    log.info("  R3 best NP:  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP)
    log.info("  R3 best Obj: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f  Obj=%.0f",
             SEED_HIGH_LE, SEED_HIGH_SE, SEED_HIGH_STP, SEED_HIGH_LMT,
             SEED_HIGH_NP, SEED_HIGH_OBJ)
    log.info("  Target: %.0f  Gap: %.0f (-%.1f%%)",
             TARGET_NP, TARGET_NP - SEED_NP,
             (TARGET_NP - SEED_NP) / SEED_NP * 100)
    log.info("  Focus: LE=16 regime, LE=16+high SE, very high LE, low STP, high LMT")
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
    # A01  LE=16 fine zoom  (4×6×4×7=672)
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_le16_fine"
    _c = _cfg(_n, (13, 19, 2), (30, 40, 2), (3.0, 4.2, 0.4), (20, 26, 1))
    log.info("A01  LE(13-19 s2) × SE(30-40 s2) × STP(3.0-4.2 s0.4) × LMT(20-26 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A02  LE=16 + high SE 55-90  (4×8×5×5=800)
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_le16_high_se"
    _c = _cfg(_n, (13, 19, 2), (55, 90, 5), (3.0, 5.0, 0.5), (18, 26, 2))
    log.info("A02  LE(13-19 s2) × SE(55-90 s5) × STP(3.0-5.0 s0.5) × LMT(18-26 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A03  LE=16 fine STP (2.4-5.2)  (4×5×8×7=1120)
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_le16_fine_stp"
    _c = _cfg(_n, (13, 19, 2), (30, 38, 2), (2.4, 5.2, 0.4), (20, 26, 1))
    log.info("A03  LE(13-19 s2) × SE(30-38 s2) × STP(2.4-5.2 s0.4) × LMT(20-26 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A04  Very high LE 25-49  (7×5×4×4=560)
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_very_high_le"
    _c = _cfg(_n, (25, 49, 4), (30, 38, 2), (3.2, 4.4, 0.4), (20, 26, 2))
    log.info("A04  LE(25-49 s4) × SE(30-38 s2) × STP(3.2-4.4 s0.4) × LMT(20-26 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A05  Very low STP 0.5-2.5 with LE=2 seed  (3×3×5×7=315)
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_low_stp"
    _c = _cfg(_n, (1, 3, 1), (33, 37, 2), (0.5, 2.5, 0.5), (18, 24, 1))
    log.info("A05  LE(1-3) × SE(33-37 s2) × STP(0.5-2.5 s0.5) × LMT(18-24 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A06  High LMT 24-50 with LE=2 seed  (3×3×4×14=504)
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_lmt"
    _c = _cfg(_n, (1, 3, 1), (33, 37, 2), (2.8, 4.0, 0.4), (24, 50, 2))
    log.info("A06  LE(1-3) × SE(33-37 s2) × STP(2.8-4.0 s0.4) × LMT(24-50 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A07  Adaptive zoom around R4 NP-best (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_zoom_best"
    for _r_se, _r_stp, _r_lmt, _stp_step, _lmt_step in [
        (6, 1.5, 5.0, 0.2, 1),
        (5, 1.2, 4.0, 0.2, 1),
        (4, 1.0, 3.0, 0.2, 1),
        (3, 0.8, 2.0, 0.2, 1),
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
    # A08  LE=16 with higher LMT range 24-36  (4×6×4×7=672)
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_le16_high_lmt"
    _c = _cfg(_n, (13, 19, 2), (30, 40, 2), (3.2, 4.4, 0.4), (24, 36, 2))
    log.info("A08  LE(13-19 s2) × SE(30-40 s2) × STP(3.2-4.4 s0.4) × LMT(24-36 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A09  LE=1-2 with wider SE 40-70  (2×11×5×5=550)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_le12_wide_se"
    _c = _cfg(_n, (1, 2, 1), (40, 70, 3), (3.0, 5.0, 0.5), (18, 26, 2))
    log.info("A09  LE(1-2) × SE(40-70 s3) × STP(3.0-5.0 s0.5) × LMT(18-26 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A10  LE survey 1-50 step 5  (10×4×3×3=360)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_le_survey"
    _c = _cfg(_n, (1, 46, 5), (32, 38, 2), (3.2, 4.0, 0.4), (20, 24, 2))
    log.info("A10  LE(1-46 s5) × SE(32-38 s2) × STP(3.2-4.0 s0.4) × LMT(20-24 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)

    # ──────────────────────────────────────────────────────────────────────
    # A11  Dense zoom R4 (progressive shrink)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_dense_zoom"
    for _r_se, _r_stp, _r_lmt, _stp_step, _lmt_step in [
        (5, 1.5, 5.0, 0.2, 1),
        (4, 1.2, 4.0, 0.2, 1),
        (3, 1.0, 3.0, 0.2, 1),
        (2, 0.8, 2.0, 0.2, 1),
    ]:
        _le11  = zoom(best_le,  2,        1,          LE_LO,  LE_HI)
        _se11  = zoom(best_se,  _r_se,    1,          SE_LO,  SE_HI)
        _stp11 = zoom(best_stp, _r_stp,   _stp_step,  STP_LO, STP_HI)
        _lmt11 = zoom(best_lmt, _r_lmt,   _lmt_step,  LMT_LO, LMT_HI)
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
    # A12  LE=16 with low STP 1.0-3.0  (4×6×5×5=600)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_le16_low_stp"
    _c = _cfg(_n, (13, 19, 2), (30, 40, 2), (1.0, 3.0, 0.5), (20, 28, 2))
    log.info("A12  LE(13-19 s2) × SE(30-40 s2) × STP(1.0-3.0 s0.5) × LMT(20-28 s2)  %d combos",
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
    log.info("  FINAL — GC Hourly Breakout NP>700K Round-4")
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
    ap = argparse.ArgumentParser(description="GC Hourly Breakout NP>700K Round-4 search")
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
