"""
search_cl_cl_daily3.py — _2021Basic_Break_CL on CME.CL HOT Daily, Round 3

R1+R2 summary (24 attempts, 2026-05-21):
  Best NP:  $90,950  LE=1 SE=1 STP=4 LMT=5  (ceiling confirmed — gap -87%)
  Best Obj: $136,679 LE=1 SE=35 STP=4 LMT=5  NP=$85,200

R1+R2 parameter coverage:
  LE:  1–30     SE: 1–70     STP: 0.25–18 (step as fine as 0.1)    LMT: 1–50

Round-3 explores ONLY the truly untouched regions:
  1. High SE (SE=70-200)       — SE>70 never explored
  2. Very high SE (SE=200-500) — far boundary
  3. High LE (LE=30-100)       — LE>30 never explored
  4. Very high STP (STP=20-200) — confirm dead-param territory
  5. Very high LMT (LMT=50-300) — far LMT boundary
  6. Micro STP (STP=0.02-0.20) — below STP=0.25 floor
  7. Sub-$1 LMT (LMT=0.1-0.9) — below LMT=1 floor
  8. LE=1 + SE=70-150 fine     — high-SE with optimal STP/LMT
  9. LE=30-80 + SE=5-30        — high-LE medium-SE regime
  10. Ultra-zoom (STP step=0.05, LMT step=0.1)
  11. Adaptive zoom from R3 best
  12. Global boundary R3 (LE/SE/STP/LMT all wide)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 high_se         : LE(1-2 s1) × SE(70-200 s15)  × STP(3-6 s1)         × LMT(4-8 s1)     × LenLE(95-105 s5)  = 1200
  A02 very_high_se    : LE(1-2 s1) × SE(200-500 s50) × STP(3-6 s1)         × LMT(4-8 s1)     × LenLE(95-105 s5)  = 840
  A03 high_le         : LE(30-100 s10) × SE(1-5 s1)  × STP(3-6 s1)         × LMT(4-7 s1)     × LenLE(95-105 s5)  = 1920
  A04 very_high_stp   : LE(1-2 s1) × SE(1-4 s1)     × STP(20-200 s20)     × LMT(4-6 s1)     × LenLE(95-105 s5)  = 720
  A05 very_high_lmt   : LE(1-2 s1) × SE(1-4 s1)     × STP(3-6 s1)         × LMT(50-300 s25) × LenLE(95-105 s5)  = 1056
  A06 micro_stp       : LE(1-2 s1) × SE(1-4 s1)     × STP(0.02-0.20 s0.02)× LMT(4-7 s1)     × LenLE(95-105 s5)  = 960
  A07 sub1_lmt        : LE(1-3 s1) × SE(1-4 s1)     × STP(2-5 s1)         × LMT(0.1-0.9 s0.1)× LenLE(95-105 s5) = 1296
  A08 le1_high_se_fine: LE(1-2 s1) × SE(70-150 s10) × STP(3-6 s0.5)       × LMT(3-7 s1)     × LenLE(95-105 s5)  = 1890
  A09 high_le_med_se  : LE(30-80 s10) × SE(5-30 s5) × STP(3-6 s1)         × LMT(4-8 s1)     × LenLE(95-105 s5)  = 2160
  A10 ultra_zoom      : LE(1-2 s1) × SE(1-2 s1)     × STP(3.8-4.2 s0.05) × LMT(4.5-5.5 s0.1)× LenLE(95-105 s5) = 1188
  A11 adaptive_zoom   : (dynamic from R3 best NP)
  A12 global_boundary3: LE(1-100 s10) × SE(1-500 s50) × STP(2-6 s2)       × LMT(3-7 s2)     × LenLE(95-105 s5)  = 3267
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\cl_cl_daily3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0

STP_LO,   STP_HI   = 0.01,  500.0
LMT_LO,   LMT_HI   = 0.1,   500.0
SE_LO,    SE_HI    = 1.0,   500.0
LE_LO,    LE_HI    = 1.0,   200.0
LENLE_LO, LENLE_HI = 2.0,   500.0

# R1+R2 best as seeds
SEED_LE, SEED_SE    = 1.0,   1.0
SEED_STP, SEED_LMT  = 4.0,   5.0
SEED_LENLE           = 100.0
SEED_NP              = -1_000_000.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_cl_cl_daily3_{int(time.time())}.log"
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
        name=f"CLCL3_{name}",
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
    return OUTPUT_DIR / f"CLCL3_{name}_raw.csv"


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
    log.info("=== Starting CLCL3_%s (%d combos) ===", name, cfg.total_runs())
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
        le    = float(best["LE"])
        se    = float(best["SE"])
        stp   = float(best["STP"])
        lmt   = float(best["LMT"])
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
        le    = float(best["LE"])
        se    = float(best["SE"])
        stp   = float(best["STP"])
        lmt   = float(best["LMT"])
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
    r2_best = {
        "LE": 1, "SE": 1, "STP": 4.0, "LMT": 5.0,
        "net_profit": 90950, "max_drawdown": -62710,
        "objective": 131907, "total_trades": 34,
        "note": "ceiling confirmed across R1+R2 (24 attempts)",
    }
    payload = {
        "round": 3,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "long_entry":    "BUY HIGHEST(H,LE) when MA(LenLE)>MA(LenLE*2)",
            "short_entry":   "SELL LOWEST(L,SE) when MA(LenLE)<=MA(LenLE*2)",
            "exits":         "STP×ATR stop + LMT×ATR limit",
            "lenle_note":    "LenLE=100 fixed (MC64 does not vary via automation)",
            "r1r2_best":     r2_best,
            "r3_focus":      "Unexplored: SE>70, LE>30, STP>18, LMT>50, STP<0.25, LMT<1",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_cl_cl_daily3.json"
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
    log.info("  CL Daily _2021Basic_Break_CL NP>700K — Round 3")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R1+R2 best: LE=1 SE=1 STP=4 LMT=5  NP=$90,950 (gap $609K)")
    log.info("  R3 focus: SE>70, LE>30, STP>18, LMT>50, STP<0.25, LMT<1")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_le, best_se, best_stp, best_lmt, best_lenle
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_le, best_se, best_stp, best_lmt, best_lenle,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
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

        log.info("  [A%02d %-24s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  high_se — SE=70-200, never explored in R1+R2
    #      LE(1-2 s1) × SE(70-200 s15) × STP(3-6 s1) × LMT(4-8 s1) × LenLE(95-105 s5)
    #      = 2×10×4×5×3 = 1200
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_high_se"
    _c = _cfg(_n, (1, 2, 1), (70, 200, 15), (3.0, 6.0, 1.0), (4.0, 8.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A01  LE(1-2) × SE(70-200 s15) × STP(3-6) × LMT(4-8) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  very_high_se — SE=200-500, far boundary
    #      LE(1-2 s1) × SE(200-500 s50) × STP(3-6 s1) × LMT(4-8 s1) × LenLE(95-105 s5)
    #      = 2×7×4×5×3 = 840
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_very_high_se"
    _c = _cfg(_n, (1, 2, 1), (200, 500, 50), (3.0, 6.0, 1.0), (4.0, 8.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A02  LE(1-2) × SE(200-500 s50) × STP(3-6) × LMT(4-8) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  high_le — LE=30-100, never explored
    #      LE(30-100 s10) × SE(1-5 s1) × STP(3-6 s1) × LMT(4-7 s1) × LenLE(95-105 s5)
    #      = 8×5×4×4×3 = 1920
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_high_le"
    _c = _cfg(_n, (30, 100, 10), (1, 5, 1), (3.0, 6.0, 1.0), (4.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A03  LE(30-100 s10) × SE(1-5) × STP(3-6) × LMT(4-7) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  very_high_stp — STP=20-200, confirm dead-param territory
    #      LE(1-2 s1) × SE(1-4 s1) × STP(20-200 s20) × LMT(4-6 s1) × LenLE(95-105 s5)
    #      = 2×4×10×3×3 = 720
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_very_high_stp"
    _c = _cfg(_n, (1, 2, 1), (1, 4, 1), (20.0, 200.0, 20.0), (4.0, 6.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A04  LE(1-2) × SE(1-4) × STP(20-200 s20) × LMT(4-6) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  very_high_lmt — LMT=50-300, far LMT boundary
    #      LE(1-2 s1) × SE(1-4 s1) × STP(3-6 s1) × LMT(50-300 s25) × LenLE(95-105 s5)
    #      = 2×4×4×11×3 = 1056
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_very_high_lmt"
    _c = _cfg(_n, (1, 2, 1), (1, 4, 1), (3.0, 6.0, 1.0), (50.0, 300.0, 25.0), (95.0, 105.0, 5.0))
    log.info("A05  LE(1-2) × SE(1-4) × STP(3-6) × LMT(50-300 s25) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  micro_stp — STP=0.02-0.20, below the R1 floor of 0.25
    #      LE(1-2 s1) × SE(1-4 s1) × STP(0.02-0.20 s0.02) × LMT(4-7 s1) × LenLE(95-105 s5)
    #      = 2×4×10×4×3 = 960
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_micro_stp"
    _c = _cfg(_n, (1, 2, 1), (1, 4, 1), (0.02, 0.20, 0.02), (4.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A06  LE(1-2) × SE(1-4) × STP(0.02-0.20 s0.02) × LMT(4-7) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  sub1_lmt — LMT=0.1-0.9, below the R2 floor of 1.0
    #      LE(1-3 s1) × SE(1-4 s1) × STP(2-5 s1) × LMT(0.1-0.9 s0.1) × LenLE(95-105 s5)
    #      = 3×4×4×9×3 = 1296
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_sub1_lmt"
    _c = _cfg(_n, (1, 3, 1), (1, 4, 1), (2.0, 5.0, 1.0), (0.1, 0.9, 0.1), (95.0, 105.0, 5.0))
    log.info("A07  LE(1-3) × SE(1-4) × STP(2-5) × LMT(0.1-0.9 s0.1) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  le1_high_se_fine — LE=1 with fine SE=70-150 grid
    #      LE(1-2 s1) × SE(70-150 s10) × STP(3-6 s0.5) × LMT(3-7 s1) × LenLE(95-105 s5)
    #      = 2×9×7×5×3 = 1890
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_le1_high_se_fine"
    _c = _cfg(_n, (1, 2, 1), (70, 150, 10), (3.0, 6.0, 0.5), (3.0, 7.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A08  LE(1-2) × SE(70-150 s10) × STP(3-6 s0.5) × LMT(3-7) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  high_le_med_se — LE=30-80 with medium SE
    #      LE(30-80 s10) × SE(5-30 s5) × STP(3-6 s1) × LMT(4-8 s1) × LenLE(95-105 s5)
    #      = 6×6×4×5×3 = 2160
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_high_le_med_se"
    _c = _cfg(_n, (30, 80, 10), (5, 30, 5), (3.0, 6.0, 1.0), (4.0, 8.0, 1.0), (95.0, 105.0, 5.0))
    log.info("A09  LE(30-80 s10) × SE(5-30 s5) × STP(3-6) × LMT(4-8) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  ultra_zoom — STP step=0.05, LMT step=0.1 around known peak
    #      LE(1-2 s1) × SE(1-2 s1) × STP(3.8-4.2 s0.05) × LMT(4.5-5.5 s0.1) × LenLE(95-105 s5)
    #      = 2×2×9×11×3 = 1188
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_ultra_zoom"
    _c = _cfg(_n, (1, 2, 1), (1, 2, 1), (3.8, 4.2, 0.05), (4.5, 5.5, 0.1), (95.0, 105.0, 5.0))
    log.info("A10  LE(1-2) × SE(1-2) × STP(3.8-4.2 s0.05) × LMT(4.5-5.5 s0.1) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom — zoom around best NP found in A01-A10
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom"
    log.info("A11  adaptive_zoom — center: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_se, r_stp, r_lmt in [(12, 1.0, 4), (8, 0.8, 3), (6, 0.6, 2), (4, 0.4, 1)]:
            _le  = zoom(best_le,  1.0,   1.0,  LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  1.0,  SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.2,  STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
            _c = _cfg(_n, _le, _se, _stp, _lmt, (95.0, 105.0, 5.0))
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LE%s SE%s STP%s LMT%s  %d combos", _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A12  global_boundary3 — wide coarse sweep covering full unexplored range
    #      LE(1-100 s10) × SE(1-500 s50) × STP(2-6 s2) × LMT(3-7 s2) × LenLE(95-105 s5)
    #      = 11×11×3×3×3 = 3267
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_global_boundary3"
    _c = _cfg(_n, (1, 100, 10), (1, 500, 50), (2.0, 6.0, 2.0), (3.0, 7.0, 2.0), (95.0, 105.0, 5.0))
    log.info("A12  LE(1-100 s10) × SE(1-500 s50) × STP(2-6 s2) × LMT(3-7 s2) × LenLE(95-105)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  CL Daily _2021Basic_Break_CL Round-3 COMPLETE")
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
        description="CL Daily _2021Basic_Break_CL NP>700K Round-3 search")
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
