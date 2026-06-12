"""
search_txf_osc_daily2.py — _2021Basic_Osc_NQ on TWF.TXF HOT Daily, Round 2

R1 results:
  NP-best:  LEN=4  LE=-1.5  SE=1.75 STP=1.75 LMT=21   NP=4,756,400 MDD=-1,014,000 trades=26
  Obj-best: LEN=6  LE=1.0   SE=2.5  STP=1.0  LMT=10   NP=4,475,400 MDD=-567,400   trades=48

R1 gaps: A03(high_lmt), A06(high_len), A08(global_d2), A10(zoom2) all UI-failed.
         Two different regimes discovered: (1) negative LE high-NP; (2) positive LE high-Obj.

R2 design — 8 fixed attempts + 3 adaptive zooms (11 total), ≤5,000 combos each:
  A01 obj_champ_zoom  : LEN(3-9 s1)   × LE(0-2 s0.5)    × SE(1.5-3.0 s0.5)  × STP(0.5-1.5 s0.25) × LMT(7-17 s2)  = 7×5×4×5×6  = 4,200  (zoom R1 Obj-best)
  A02 np_champ_zoom   : LEN(2-6 s1)   × LE(-2-0 s0.5)   × SE(1.0-2.25 s0.25)× STP(1.25-2.25 s0.25)× LMT(17-25 s2) = 5×5×6×5×5  = 3,750  (zoom R1 NP-best)
  A03 re_high_lmt     : LEN(5-25 s5)  × LE(-1.5-0.5 s0.5)× SE(1.5-3.5 s0.5) × STP(0.5-1.5 s0.5)  × LMT(30-80 s10)= 5×5×5×3×6  = 2,250  (retry R1 A03 fail)
  A04 re_high_len     : LEN(10-50 s8) × LE(-2-1 s0.5)   × SE(1.0-3.0 s0.5)  × STP(0.5-1.5 s0.5)  × LMT(10-30 s5) = 6×7×5×3×5  = 3,150  (retry R1 A06 fail)
  A05 pos_le_wide     : LEN(2-14 s3)  × LE(0-3 s0.5)    × SE(1.5-3.5 s0.5)  × STP(0.5-2.0 s0.5)  × LMT(5-25 s5)  = 5×7×5×4×5  = 3,500  (positive LE survey)
  A06 re_global_d2    : LEN(2-32 s5)  × LE(-3-2 s1)     × SE(0.5-4.5 s1)    × STP(0.25-1.75 s0.5) × LMT(5-35 s10) = 7×6×5×4×4  = 3,360  (retry R1 A08 fail)
  A07 lmt_bridge      : LEN(3-9 s2)   × LE(-1-1 s0.5)   × SE(1.5-3.0 s0.5)  × STP(0.5-2.0 s0.5)  × LMT(8-44 s4)  = 4×5×4×4×10 = 3,200  (LMT=8-44 regime map)
  A08 high_pos_le     : LEN(5-25 s5)  × LE(2-5 s0.5)    × SE(2.0-5.0 s0.5)  × STP(0.5-2.0 s0.5)  × LMT(5-25 s5)  = 5×7×7×4×5  = 4,900  (extreme positive LE)
  A09-A11 adaptive zooms
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260523SFJ_BASIC_OSC_AI.wsp"
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "_2021Basic_Osc_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_osc_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 7_000_000.0   # TWD

LEN_LO, LEN_HI = 1.0,   200.0
LE_LO,  LE_HI  = -10.0, 10.0
SE_LO,  SE_HI  = 0.0,   10.0
STP_LO, STP_HI = 0.01,  50.0
LMT_LO, LMT_HI = 0.5,   200.0

# R1 NP-champion as seed
SEED_LEN, SEED_LE  = 4.0,  -1.5
SEED_SE,  SEED_STP = 1.75,  1.75
SEED_LMT           = 21.0
SEED_NP            = 4_756_400.0

PREFIX = "TXFOSD2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_osc_daily2_{int(time.time())}.log"
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
         len_:  Tuple[float, float, float],
         le:    Tuple[float, float, float],
         se:    Tuple[float, float, float],
         stp:   Tuple[float, float, float],
         lmt:   Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    len_ = _safe(len_, LEN_LO, LEN_HI)
    le   = _safe(le,   LE_LO,  LE_HI)
    se   = _safe(se,   SE_LO,  SE_HI)
    stp  = _safe(stp,  STP_LO, STP_HI)
    lmt  = _safe(lmt,  LMT_LO, LMT_HI)

    combos = n_vals(len_) * n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
            ParamAxis("LEN", *len_),
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
    return OUTPUT_DIR / f"{PREFIX}{name}_raw.csv"


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
    log.info("=== Starting %s%s (%d combos) ===", PREFIX, name, cfg.total_runs())
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


def champion(df, fb_len, fb_le, fb_se, fb_stp, fb_lmt):
    """Priority: target met → highest NP (target-chasing mode)."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LEN"]), float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["LEN"]), float(best["LE"]), float(best["SE"]),
                float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LEN"]), float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["LEN"]), float(best["LE"]), float(best["SE"]),
                float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_len, fb_le, fb_se, fb_stp, fb_lmt,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, len_, le, se, stp, lmt, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "LEN": len_, "LE": le, "SE": se, "STP": stp, "LMT": lmt,
        "objective": obj, "net_profit": np_, "max_drawdown": mdd,
        "total_trades": trades, "target_met": met,
        "combos": combos,
        "rows": len(df) if df is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }


def save_json(best_entry, attempt_log, target_met):
    payload = {
        "round": 2,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY when C crosses over BollingerBand(C,LEN,LE); SHORT when C crosses under BollingerBand(C,LEN,SE)",
            "exits":       "STP×ATR(10) stop + LMT×ATR(10) limit",
            "r1_np_best":  "LEN=4 LE=-1.5 SE=1.75 STP=1.75 LMT=21 NP=4,756,400 MDD=-1,014,000 trades=26",
            "r1_obj_best": "LEN=6 LE=1.0 SE=2.5 STP=1.0 LMT=10 NP=4,475,400 MDD=-567,400 trades=48 Obj=35,299,974",
            "r2_focus":    "Zoom both R1 regimes; retry 4 UI-failed attempts (high_lmt/high_len/global_d2); positive LE survey; LMT bridge; extreme positive LE",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_osc_daily2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_len = SEED_LEN
    best_le  = SEED_LE
    best_se  = SEED_SE
    best_stp = SEED_STP
    best_lmt = SEED_LMT
    best_np  = SEED_NP
    best_obj = 0.0
    target_met  = False
    attempt_log: List[dict] = []
    best_entry:  Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  TXF Daily _2021Basic_Osc_NQ  NP>7,000,000 TWD — Round 2")
    log.info("  Symbol: %s  Signal: %s  Timeframe: Daily", SYMBOL, SIGNAL)
    log.info("  R1 NP-best: LEN=4 LE=-1.5 SE=1.75 STP=1.75 LMT=21  NP=4,756,400 TWD")
    log.info("  R1 Obj-best: LEN=6 LE=1.0 SE=2.5 STP=1.0 LMT=10   NP=4,475,400 TWD  Obj=35.3M")
    log.info("  Target: %.0f TWD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_len, best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_len, best_le, best_se, best_stp, best_lmt,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        len_, le, se, stp, lmt, obj, np_, mdd, tr, met = champion(
            df, best_len, best_le, best_se, best_stp, best_lmt)

        if np_ > best_np:
            best_len = len_
            best_le  = le
            best_se  = se
            best_stp = stp
            best_lmt = lmt
            best_np  = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, len_, le, se, stp, lmt,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, len_, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/7M" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  obj_champ_zoom — zoom around R1 Obj-best (LEN=6 LE=1 SE=2.5 STP=1 LMT=10)
    #      LEN(3-9 s1) × LE(0-2 s0.5) × SE(1.5-3.0 s0.5) × STP(0.5-1.5 s0.25) × LMT(7-17 s2)
    #      = 7×5×4×5×6 = 4,200
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_obj_champ_zoom"
    _c = _cfg(_n, (3, 9, 1), (0.0, 2.0, 0.5), (1.5, 3.0, 0.5), (0.5, 1.5, 0.25), (7, 17, 2))
    log.info("A01  LEN(3-9 s1)×LE(0-2 s0.5)×SE(1.5-3.0 s0.5)×STP(0.5-1.5 s0.25)×LMT(7-17 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  np_champ_zoom — zoom around R1 NP-best (LEN=4 LE=-1.5 SE=1.75 STP=1.75 LMT=21)
    #      LEN(2-6 s1) × LE(-2-0 s0.5) × SE(1.0-2.25 s0.25) × STP(1.25-2.25 s0.25) × LMT(17-25 s2)
    #      = 5×5×6×5×5 = 3,750
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_np_champ_zoom"
    _c = _cfg(_n, (2, 6, 1), (-2.0, 0.0, 0.5), (1.0, 2.25, 0.25), (1.25, 2.25, 0.25), (17, 25, 2))
    log.info("A02  LEN(2-6 s1)×LE(-2-0 s0.5)×SE(1.0-2.25 s0.25)×STP(1.25-2.25 s0.25)×LMT(17-25 s2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  re_high_lmt — retry R1 A03 (high_lmt UI-failed): LMT=30-80×ATR
    #      LEN(5-25 s5) × LE(-1.5-0.5 s0.5) × SE(1.5-3.5 s0.5) × STP(0.5-1.5 s0.5) × LMT(30-80 s10)
    #      = 5×5×5×3×6 = 2,250
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_re_high_lmt"
    _c = _cfg(_n, (5, 25, 5), (-1.5, 0.5, 0.5), (1.5, 3.5, 0.5), (0.5, 1.5, 0.5), (30, 80, 10))
    log.info("A03  LEN(5-25 s5)×LE(-1.5-0.5 s0.5)×SE(1.5-3.5 s0.5)×STP(0.5-1.5 s0.5)×LMT(30-80 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  re_high_len — retry R1 A06 (high_len UI-failed): LEN=10-50
    #      LEN(10-50 s8) × LE(-2-1 s0.5) × SE(1.0-3.0 s0.5) × STP(0.5-1.5 s0.5) × LMT(10-30 s5)
    #      = 6×7×5×3×5 = 3,150
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_re_high_len"
    _c = _cfg(_n, (10, 50, 8), (-2.0, 1.0, 0.5), (1.0, 3.0, 0.5), (0.5, 1.5, 0.5), (10, 30, 5))
    log.info("A04  LEN(10-50 s8)×LE(-2-1 s0.5)×SE(1.0-3.0 s0.5)×STP(0.5-1.5 s0.5)×LMT(10-30 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  pos_le_wide — wide positive LE survey (LE=0-3)
    #      LEN(2-14 s3) × LE(0-3 s0.5) × SE(1.5-3.5 s0.5) × STP(0.5-2.0 s0.5) × LMT(5-25 s5)
    #      = 5×7×5×4×5 = 3,500
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_pos_le_wide"
    _c = _cfg(_n, (2, 14, 3), (0.0, 3.0, 0.5), (1.5, 3.5, 0.5), (0.5, 2.0, 0.5), (5, 25, 5))
    log.info("A05  LEN(2-14 s3)×LE(0-3 s0.5)×SE(1.5-3.5 s0.5)×STP(0.5-2.0 s0.5)×LMT(5-25 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  re_global_d2 — retry R1 A08 (global_d2 UI-failed): second wide sweep
    #      LEN(2-32 s5) × LE(-3-2 s1) × SE(0.5-4.5 s1) × STP(0.25-1.75 s0.5) × LMT(5-35 s10)
    #      = 7×6×5×4×4 = 3,360
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_re_global_d2"
    _c = _cfg(_n, (2, 32, 5), (-3.0, 2.0, 1.0), (0.5, 4.5, 1.0), (0.25, 1.75, 0.5), (5, 35, 10))
    log.info("A06  LEN(2-32 s5)×LE(-3-2 s1)×SE(0.5-4.5 s1)×STP(0.25-1.75 s0.5)×LMT(5-35 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  lmt_bridge — map LMT=8-44 (between A01's 10 and A09's 21)
    #      LEN(3-9 s2) × LE(-1-1 s0.5) × SE(1.5-3.0 s0.5) × STP(0.5-2.0 s0.5) × LMT(8-44 s4)
    #      = 4×5×4×4×10 = 3,200
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_lmt_bridge"
    _c = _cfg(_n, (3, 9, 2), (-1.0, 1.0, 0.5), (1.5, 3.0, 0.5), (0.5, 2.0, 0.5), (8, 44, 4))
    log.info("A07  LEN(3-9 s2)×LE(-1-1 s0.5)×SE(1.5-3.0 s0.5)×STP(0.5-2.0 s0.5)×LMT(8-44 s4)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  high_pos_le — extreme positive LE=2-5 (buy well above upper band)
    #      LEN(5-25 s5) × LE(2-5 s0.5) × SE(2.0-5.0 s0.5) × STP(0.5-2.0 s0.5) × LMT(5-25 s5)
    #      = 5×7×7×4×5 = 4,900
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_high_pos_le"
    _c = _cfg(_n, (5, 25, 5), (2.0, 5.0, 0.5), (2.0, 5.0, 0.5), (0.5, 2.0, 0.5), (5, 25, 5))
    log.info("A08  LEN(5-25 s5)×LE(2-5 s0.5)×SE(2.0-5.0 s0.5)×STP(0.5-2.0 s0.5)×LMT(5-25 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — zoom around best NP found in R2
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (3, 0.5, 0.5, 0.5, 5.0),
            (2, 0.5, 0.5, 0.25, 4.0),
            (2, 0.25, 0.25, 0.25, 3.0),
            (1, 0.25, 0.25, 0.25, 2.0),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.25, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.25, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.25, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
            _c = _cfg(_n, _len, _le, _se, _stp, _lmt)
            if _c.total_runs() <= 5000:
                break
        log.info("  A09 zoom: LEN%s LE%s SE%s STP%s LMT%s → %d combos",
                 _len, _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — second zoom with finer steps
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (2, 0.5, 0.5, 0.25, 4.0),
            (2, 0.25, 0.25, 0.25, 3.0),
            (1, 0.25, 0.25, 0.25, 2.0),
            (1, 0.25, 0.25, 0.125, 1.5),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.25, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.25, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.25, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
            _c = _cfg(_n, _len, _le, _se, _stp, _lmt)
            if _c.total_runs() <= 5000:
                break
        log.info("  A10 zoom: LEN%s LE%s SE%s STP%s LMT%s → %d combos",
                 _len, _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — final fine zoom
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (2, 0.5, 0.5, 0.25, 3.0),
            (1, 0.25, 0.25, 0.25, 2.0),
            (1, 0.25, 0.25, 0.125, 1.5),
            (1, 0.25, 0.25, 0.125, 1.0),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.25, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.25, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.25, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
            _c = _cfg(_n, _len, _le, _se, _stp, _lmt)
            if _c.total_runs() <= 5000:
                break
        log.info("  A11 zoom: LEN%s LE%s SE%s STP%s LMT%s → %d combos",
                 _len, _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ROUND 2 COMPLETE")
    log.info("  Best NP: %.0f  (target: %.0f  gap: %.0f)",
             best_np, TARGET_NP, TARGET_NP - best_np)
    if best_entry:
        log.info("  Champion: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  MDD=%.0f  trades=%d",
                 best_entry.get("LEN"), best_entry.get("LE"), best_entry.get("SE"),
                 best_entry.get("STP"), best_entry.get("LMT"),
                 best_entry.get("objective", 0),
                 best_entry.get("max_drawdown", 0),
                 best_entry.get("total_trades", 0))
    log.info("  TARGET %s", "MET ✅" if target_met else "NOT MET")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LEN": best_len, "LE": best_le, "SE": best_se,
            "STP": best_stp, "LMT": best_lmt,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>7,000,000 TWD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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


def main():
    ap = argparse.ArgumentParser(description="TXF Daily _2021Basic_Osc_NQ NP>7M TWD Round-2 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1–11)")
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
