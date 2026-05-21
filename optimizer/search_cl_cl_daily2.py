"""
search_cl_cl_daily2.py — _2021Basic_Break_CL on CME.CL HOT Daily, Round 2

Round 1 summary (search_cl_cl_daily.py — 12 attempts):
  Best NP:  $90,950  LE=1 SE=1 STP=4 LMT=5  obj=131,907  (target $700K — gap 87%)
  Best Obj: $155,648 LE=20 SE=30 STP=2 LMT=6  NP=76,460
  Ceiling:  ~$91K confirmed; strategy marginally profitable on CL daily

Key Round-1 findings:
  - LE=1 is the ONLY effective long-entry lookback (LE≥2 consistently worse)
  - SE=1 gives highest NP but SE=30-41 gives better risk-adjusted return
  - STP=4 is the NP sweet spot (STP=0.25-1 and STP>5 all inferior)
  - LMT=5-6 is optimal; LMT<3 and LMT>10 are much worse
  - LenLE never varies in practice (always 100); token range (95-105 s5) used

Round-2 exploration priorities:
  1. Ultra-fine STP around 4 (step=0.1) — does NP peak sharply at exactly 4?
  2. SE=2-15 fine sweep — fills gap between SE=1 (best NP) and SE=30 (best Obj)
  3. Fine LMT around 4-7 (step=0.5) — is there a sharper peak than LMT=5?
  4. LE=1 + high SE (SE=40-80) + low LMT (LMT=1-3) — building on A12's SE=41,LMT=2
  5. STP=5-20 systematically — unexplored above 4 in LE=1 regime
  6. LMT=1-2 with good STP — LMT=2 appeared in A12 best (SE=41)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 ultra_stp_fine   : LE(1-2 s1) × SE(1-3 s1) × STP(3.0-5.0 s0.25) × LMT(4-7 s1) × LenLE(95-105 s5)
  A02 se_fine_mid      : LE(1-2 s1) × SE(2-15 s1) × STP(3-5 s1)        × LMT(4-7 s1) × LenLE(95-105 s5)
  A03 lmt_fine         : LE(1-2 s1) × SE(1-4 s1) × STP(3-5 s0.5)       × LMT(3.5-7 s0.5) × LenLE(95-105 s5)
  A04 high_se_low_lmt  : LE(1-2 s1) × SE(25-70 s5) × STP(3-8 s1)       × LMT(1-4 s1) × LenLE(95-105 s5)
  A05 stp_high_le1     : LE(1-2 s1) × SE(1-5 s1) × STP(5-18 s1)        × LMT(3-8 s1) × LenLE(95-105 s5)
  A06 lmt_very_low     : LE(1-3 s1) × SE(1-6 s1) × STP(3-6 s1)         × LMT(1-3 s1) × LenLE(95-105 s5)
  A07 se_mid_stp_fine  : LE(1-2 s1) × SE(5-20 s2) × STP(3.5-5 s0.25)   × LMT(4-7 s1) × LenLE(95-105 s5)
  A08 le_variety_v2    : LE(1-8 s1) × SE(1-3 s1) × STP(3.5-5 s0.5)     × LMT(4-7 s1) × LenLE(95-105 s5)
  A09 very_fine_stp    : LE(1-2 s1) × SE(1-2 s1) × STP(3.5-4.5 s0.1)   × LMT(4-6 s1) × LenLE(95-105 s5)
  A10 se_high_stp_med  : LE(1-2 s1) × SE(30-70 s5) × STP(3-8 s1)       × LMT(3-6 s1) × LenLE(95-105 s5)
  A11 global_boundary2 : LE(1-30 s3) × SE(1-30 s5)× STP(3-7 s2)        × LMT(3-8 s2) × LenLE(95-105 s5)
  A12 adaptive_zoom    : (dynamic from best NP in A01-A11)
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
SIGNAL     = "_2021Basic_Break_CL"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\cl_cl_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0

STP_LO,   STP_HI   = 0.1,   50.0
LMT_LO,   LMT_HI   = 0.5,  200.0
SE_LO,    SE_HI    = 1.0,  200.0
LE_LO,    LE_HI    = 1.0,  200.0
LENLE_LO, LENLE_HI = 2.0,  500.0

# Round-1 best as seeds
SEED_LE, SEED_SE    = 1.0,   1.0
SEED_STP, SEED_LMT  = 4.0,   5.0
SEED_LENLE           = 100.0
SEED_NP              = -1_000_000.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_cl_cl_daily2_{int(time.time())}.log"
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
         le:    Tuple[float, float, float],
         se:    Tuple[float, float, float],
         stp:   Tuple[float, float, float],
         lmt:   Tuple[float, float, float],
         lenle: Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    le    = _safe(le,    LE_LO,    LE_HI)
    se    = _safe(se,    SE_LO,    SE_HI)
    stp   = _safe(stp,   STP_LO,   STP_HI)
    lmt   = _safe(lmt,   LMT_LO,   LMT_HI)
    lenle = _safe(lenle, LENLE_LO, LENLE_HI)

    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt) * n_vals(lenle)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"CLCL2_{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
            ParamAxis("LE",    *le),
            ParamAxis("SE",    *se),
            ParamAxis("STP",   *stp),
            ParamAxis("LMT",   *lmt),
            ParamAxis("LenLE", *lenle),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"CLCL2_{name}_raw.csv"


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
    log.info("=== Starting CLCL2_%s (%d combos) ===", name, cfg.total_runs())
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


def champion(df, fb_le, fb_se, fb_stp, fb_lmt, fb_lenle):
    """Priority: target met → positive NP → least-negative NP."""
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
                 float(best["Objective"]), int(best["TotalTrades"]))
        le   = float(best["LE"])
        se   = float(best["SE"])
        stp  = float(best["STP"])
        lmt  = float(best["LMT"])
        lenle = float(best.get("LenLE", fb_lenle))
        return le, se, stp, lmt, lenle, float(best["Objective"]), float(best["NetProfit"]), float(best["MaxDrawdown"]), int(best["TotalTrades"]), True

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        le   = float(best["LE"])
        se   = float(best["SE"])
        stp  = float(best["STP"])
        lmt  = float(best["LMT"])
        lenle = float(best.get("LenLE", fb_lenle))
        return le, se, stp, lmt, lenle, float(best["Objective"]), float(best["NetProfit"]), float(best["MaxDrawdown"]), int(best["TotalTrades"]), False

    best = df.loc[df["NetProfit"].idxmax()]
    lenle = float(best.get("LenLE", fb_lenle))
    return fb_le, fb_se, fb_stp, fb_lmt, fb_lenle, 0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]), int(best["TotalTrades"]), False


def _entry(attempt, name, df, le, se, stp, lmt, lenle, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt,
        "name": name,
        "LE": le, "SE": se, "STP": stp, "LMT": lmt, "LenLE": lenle,
        "objective": obj,
        "net_profit": np_,
        "max_drawdown": mdd,
        "total_trades": trades,
        "target_met": met,
        "combos": combos,
        "rows": len(df) if df is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }


def save_json(best_entry, attempt_log, target_met):
    r1_best = {
        "LE": 1, "SE": 1, "STP": 4.0, "LMT": 5.0,
        "net_profit": 90950, "max_drawdown": -62710,
        "objective": 131907, "total_trades": 34,
    }
    payload = {
        "round": 2,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "long_entry":    "BUY HIGHEST(H,LE) when MA(LenLE)>MA(LenLE*2)",
            "short_entry":   "SELL LOWEST(L,SE) when MA(LenLE)<=MA(LenLE*2)",
            "exits":         "STP×ATR stop + LMT×ATR limit",
            "lenle_note":    "LenLE=100 fixed (MC64 does not vary via automation)",
            "round1_best":   r1_best,
        },
        "best_params":           best_entry,
        "all_attempts":          attempt_log,
    }
    out = OUTPUT_DIR / "final_params_cl_cl_daily2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,   best_se    = SEED_LE,    SEED_SE
    best_stp,  best_lmt   = SEED_STP,   SEED_LMT
    best_lenle             = SEED_LENLE
    best_np   = SEED_NP
    best_obj  = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  CL Daily _2021Basic_Break_CL NP>700K — Round 2")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  Round-1 best: LE=1 SE=1 STP=4 LMT=5  NP=$90,950 (gap $609K)")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_le, best_se, best_stp, best_lmt, best_lenle
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_le, best_se, best_stp, best_lmt, best_lenle,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-22s]  no valid data", attempt_num, name)
            return

        le, se, stp, lmt, lenle, obj, np_, mdd, tr, met = champion(
            df, best_le, best_se, best_stp, best_lmt, best_lenle)

        if np_ > best_np:
            best_le, best_se  = le, se
            best_stp, best_lmt = stp, lmt
            best_lenle = lenle
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, le, se, stp, lmt, lenle,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-22s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  ultra_stp_fine — very fine STP grid around Round-1 best
    #      LE(1-2 s1) × SE(1-3 s1) × STP(3.0-5.0 s0.25) × LMT(4-7 s1) × LenLE(95-105 s5)
    #      = 2×3×9×4×3 = 648
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ultra_stp_fine"
    _c = _cfg(_n, (1, 2, 1), (1, 3, 1), (3.0, 5.0, 0.25), (4.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A01  LE(1-2) × SE(1-3) × STP(3.0-5.0 s0.25) × LMT(4-7) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  se_fine_mid — SE=2-15 fills the gap between R1 A02 (SE=1-5) and A07 (SE=10-30)
    #      LE(1-2 s1) × SE(2-15 s1) × STP(3-5 s1) × LMT(4-7 s1) × LenLE(95-105 s5)
    #      = 2×14×3×4×3 = 1008
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_se_fine_mid"
    _c = _cfg(_n, (1, 2, 1), (2, 15, 1), (3.0, 5.0, 1.0), (4.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A02  LE(1-2) × SE(2-15) × STP(3-5) × LMT(4-7) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  lmt_fine — fine LMT with moderate STP steps
    #      LE(1-2 s1) × SE(1-4 s1) × STP(3.0-5.0 s0.5) × LMT(3.5-7.0 s0.5) × LenLE(95-105 s5)
    #      = 2×4×5×8×3 = 960
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_lmt_fine"
    _c = _cfg(_n, (1, 2, 1), (1, 4, 1), (3.0, 5.0, 0.5), (3.5, 7.0, 0.5), (95.0, 105.0, 5.0))
    log.info("A03  LE(1-2) × SE(1-4) × STP(3-5 s0.5) × LMT(3.5-7 s0.5) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  high_se_low_lmt — extends R1 A12 finding (SE=41,LMT=2 → NP=79K)
    #      LE(1-2 s1) × SE(25-70 s5) × STP(3-7 s2) × LMT(1-4 s1) × LenLE(95-105 s5)
    #      = 2×10×3×4×3 = 720
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_high_se_low_lmt"
    _c = _cfg(_n, (1, 2, 1), (25, 70, 5), (3.0, 7.0, 2.0), (1.0, 4.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A04  LE(1-2) × SE(25-70 s5) × STP(3-7 s2) × LMT(1-4) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  stp_high_le1 — STP=5-18 with LE=1 (unexplored high-STP territory)
    #      LE(1-2 s1) × SE(1-5 s1) × STP(5-18 s1) × LMT(3-7 s2) × LenLE(95-105 s5)
    #      = 2×5×14×3×3 = 1260
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_stp_high_le1"
    _c = _cfg(_n, (1, 2, 1), (1, 5, 1), (5.0, 18.0, 1.0), (3.0, 7.0, 2.0), (95.0, 105.0, 5.0))
    log.info("A05  LE(1-2) × SE(1-5) × STP(5-18) × LMT(3-7 s2) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  lmt_very_low — LMT=1-3 systematically (R1 A08 best: LE=1,SE=4,STP=4,LMT=3)
    #      LE(1-3 s1) × SE(1-6 s1) × STP(3-6 s1) × LMT(1-3 s1) × LenLE(95-105 s5)
    #      = 3×6×4×3×3 = 648
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_lmt_very_low"
    _c = _cfg(_n, (1, 3, 1), (1, 6, 1), (3.0, 6.0, 1.0), (1.0, 3.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A06  LE(1-3) × SE(1-6) × STP(3-6) × LMT(1-3) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  se_mid_stp_fine — mid SE range with fine STP
    #      LE(1-2 s1) × SE(5-20 s3) × STP(3.5-5.0 s0.25) × LMT(4-7 s1) × LenLE(95-105 s5)
    #      = 2×6×7×4×3 = 1008
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_se_mid_stp_fine"
    _c = _cfg(_n, (1, 2, 1), (5, 20, 3), (3.5, 5.0, 0.25), (4.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A07  LE(1-2) × SE(5-20 s3) × STP(3.5-5 s0.25) × LMT(4-7) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  le_variety_v2 — test LE=1-8 with tighter STP range
    #      LE(1-8 s1) × SE(1-3 s1) × STP(3.5-5.0 s0.5) × LMT(4-7 s1) × LenLE(95-105 s5)
    #      = 8×3×4×4×3 = 1152
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_le_variety_v2"
    _c = _cfg(_n, (1, 8, 1), (1, 3, 1), (3.5, 5.0, 0.5), (4.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A08  LE(1-8) × SE(1-3) × STP(3.5-5 s0.5) × LMT(4-7) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  very_fine_stp — STP step=0.1 around STP=4 with LE=1 only
    #      LE(1-2 s1) × SE(1-2 s1) × STP(3.5-4.5 s0.1) × LMT(4-6 s1) × LenLE(95-105 s5)
    #      = 2×2×11×3×3 = 396
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_very_fine_stp"
    _c = _cfg(_n, (1, 2, 1), (1, 2, 1), (3.5, 4.5, 0.1), (4.0, 6.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A09  LE(1-2) × SE(1-2) × STP(3.5-4.5 s0.1) × LMT(4-6) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  se_high_stp_med — high SE with medium STP
    #      LE(1-2 s1) × SE(30-70 s5) × STP(3-8 s1) × LMT(3-6 s1) × LenLE(95-105 s5)
    #      = 2×9×6×4×3 = 1296
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_se_high_stp_med"
    _c = _cfg(_n, (1, 2, 1), (30, 70, 5), (3.0, 8.0, 1.0), (3.0, 6.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A10  LE(1-2) × SE(30-70 s5) × STP(3-8) × LMT(3-6) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  global_boundary2 — wide coarse sweep to find any untouched peaks
    #      LE(1-30 s3) × SE(1-30 s5) × STP(3-7 s2) × LMT(3-8 s2) × LenLE(95-105 s5)
    #      = 10×7×3×3×3 = 1890
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_global_boundary2"
    _c = _cfg(_n, (1, 30, 3), (1, 30, 5), (3.0, 7.0, 2.0), (3.0, 8.0, 2.0), (95.0, 105.0, 5.0))
    log.info("A11  LE(1-30 s3) × SE(1-30 s5) × STP(3-7 s2) × LMT(3-8 s2) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A12  adaptive_zoom — zoom around best NP found in A01-A11
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_adaptive_zoom"
    log.info("A12  adaptive_zoom — center: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        _le  = zoom(best_le,  1.0, 1.0, LE_LO,  LE_HI)
        _se  = zoom(best_se,  3.0, 1.0, SE_LO,  SE_HI)
        _stp = zoom(best_stp, 1.0, 0.25, STP_LO, STP_HI)
        _lmt = zoom(best_lmt, 2.0, 0.5, LMT_LO, LMT_HI)
        _c = _cfg(_n, _le, _se, _stp, _lmt, (95.0, 105.0, 5.0))
        while _c.total_runs() > 5000:
            _le  = zoom(best_le,  1.0, 1.0, LE_LO,  LE_HI)
            _se  = zoom(best_se,  2.0, 1.0, SE_LO,  SE_HI)
            _stp = zoom(best_stp, 0.75, 0.25, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, 1.5, 0.5, LMT_LO, LMT_HI)
            _c = _cfg(_n, _le, _se, _stp, _lmt, (95.0, 105.0, 5.0))
            break
        log.info("A12  LE%s SE%s STP%s LMT%s  %d combos", _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  CL Daily _2021Basic_Break_CL Round-2 COMPLETE")
    log.info("  Best NP: %.0f  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g LenLE=%.4g",
             best_np, best_le, best_se, best_stp, best_lmt, best_lenle)
    log.info("  Target %.0f: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{TARGET_NP - best_np:.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LE": best_le, "SE": best_se, "STP": best_stp, "LMT": best_lmt,
            "LenLE": best_lenle, "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

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
    ap = argparse.ArgumentParser(
        description="CL Daily _2021Basic_Break_CL NP>700K Round-2 search")
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
