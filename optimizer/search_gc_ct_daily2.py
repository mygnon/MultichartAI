"""
search_gc_ct_daily2.py — SFJ_15Dworkshop_lesson5_countertrend_LS on CME.GC HOT Daily, Round 2

R1 best: LL=3 SL=1.04 LS=51 SS=2.3 NP=450,640 MDD=-50,780 Obj=3,999,142 trades=66
R1 key discovery: Asymmetric short-LL regime (LL=3-4, SL≈1.04, LS=50-51, SS=2.3-2.4).
  TXF tight-SS analog failed; GC hourly tight-SS analog failed on daily.
  NQ daily analog (LL=4 SL=1.0 LS=50 SS=2.4) gave 423K → confirmed direction.

R2 strategy: Fine-tune asymmetric short-LL regime; extend LS; probe SS landscape.
  1. LL step=1 (LL=2-8), SL fine step=0.05, LS=46-58 step=2
  2. Extend LS to 60-100 (fewer short entries?)
  3. Higher SS (2.5-4.0) — rarer, bigger short entries
  4. Lower SS (1.4-2.1) — more but smaller short entries
  5. Broad SL scan (0.6-1.8) to confirm SL=1.04 peak
  6. Very tiny LL (LL=2) deep dive

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 ll_fine          : LL(2-8 s1)×SL(0.8-1.3 s0.1)×LS(46-56 s2)×SS(2.0-2.6 s0.1)   = 7×6×6×7  = 1764
  A02 ss_high          : LL(2-6 s1)×SL(0.9-1.15 s0.05)×LS(46-60 s2)×SS(2.4-3.5 s0.2)  = 5×6×8×6  = 1440
  A03 ls_extend        : LL(2-5 s1)×SL(0.85-1.15 s0.1)×LS(55-90 s5)×SS(2.0-3.0 s0.2)  = 4×4×8×6  =  768
  A04 sl_fine          : LL(2-6 s1)×SL(0.6-1.6 s0.1)×LS(46-58 s2)×SS(2.0-2.6 s0.2)   = 5×11×7×4 = 1540
  A05 ss_low           : LL(3-7 s1)×SL(0.8-1.2 s0.1)×LS(40-55 s3)×SS(1.4-2.0 s0.1)   = 5×5×6×7  = 1050
  A06 ll_tiny_fine     : LL(2-4 s1)×SL(0.7-1.3 s0.05)×LS(44-58 s2)×SS(2.0-2.8 s0.1)  = 3×13×8×9 = 2808
  A07 broad_zone       : LL(2-10 s1)×SL(0.7-1.4 s0.1)×LS(42-62 s4)×SS(1.8-2.8 s0.2)  = 9×8×6×6  = 2592
  A08 high_ls_ss       : LL(2-5 s1)×SL(0.9-1.2 s0.1)×LS(60-100 s5)×SS(2.5-4.0 s0.3)  = 4×4×9×6  =  864
  A09 global_r2        : LL(2-52 s10)×SL(0.5-2.5 s0.5)×LS(10-80 s10)×SS(1.0-3.5 s0.5) = 6×5×8×6  = 1440
  A10 adaptive_zoom    : (dynamic from R2 best NP)
  A11 fine_zoom        : (dynamic, SL step=0.02, SS step=0.05)
  A12 ultra_fine_verify: (dynamic, SL step=0.01, SS step=0.02)
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260521SFJ_Bollinger_AI.wsp"
SYMBOL     = "CME.GC HOT"
SIGNAL     = "SFJ_15Dworkshop_lesson5_countertrend_LS"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_ct_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.1,  20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.1,  20.0

# R1 best as seed
SEED_LL, SEED_SL = 3.0,  1.04
SEED_LS, SEED_SS = 51.0, 2.3
SEED_NP          = 450_640.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_ct_daily2_{int(time.time())}.log"
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
         ll:  Tuple[float, float, float],
         sl:  Tuple[float, float, float],
         ls:  Tuple[float, float, float],
         ss:  Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    ll = _safe(ll, LL_LO, LL_HI)
    sl = _safe(sl, SL_LO, SL_HI)
    ls = _safe(ls, LS_LO, LS_HI)
    ss = _safe(ss, SS_LO, SS_HI)

    combos = n_vals(ll) * n_vals(sl) * n_vals(ls) * n_vals(ss)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"GCCTD2_{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
            ParamAxis("LENGTH_LONG",  *ll),
            ParamAxis("STDDEV_LONG",  *sl),
            ParamAxis("LENGTH_SHORT", *ls),
            ParamAxis("STDDEV_SHORT", *ss),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"GCCTD2_{name}_raw.csv"


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
    log.info("=== Starting GCCTD2_%s (%d combos) ===", name, cfg.total_runs())
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


def champion(df, fb_ll, fb_sl, fb_ls, fb_ss):
    """Priority: target met → highest NP (target-chasing mode)."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                 float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                 float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_ll, fb_sl, fb_ls, fb_ss,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, ll, sl, ls, ss, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "LENGTH_LONG": ll, "STDDEV_LONG": sl,
        "LENGTH_SHORT": ls, "STDDEV_SHORT": ss,
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
        "timeframe": "daily",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_best":  "LL=3 SL=1.04 LS=51 SS=2.3 NP=450640 MDD=-50780 trades=66 (short-LL asymmetric regime)",
            "r2_focus": "Fine-tune short-LL regime; LL step=1; SS landscape; LS extension to 90",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_gc_ct_daily2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_ll, best_sl = SEED_LL, SEED_SL
    best_ls, best_ss = SEED_LS, SEED_SS
    best_np  = SEED_NP
    best_obj = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  GC Daily countertrend_LS NP>800K — Round 2")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R1 best: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R1 discovery: Short-LL asymmetric regime (LL=3-4, SL≈1.04, LS=50-51, SS=2.3-2.4)")
    log.info("  R2 focus: fine-tune regime, SS landscape, LS extension, SL scan")
    log.info("  Target: %.0f USD  (gap: %.0f)", TARGET_NP, TARGET_NP - SEED_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_ll, best_sl, best_ls, best_ss
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_ll, best_sl, best_ls, best_ss,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        ll, sl, ls, ss, obj, np_, mdd, tr, met = champion(
            df, best_ll, best_sl, best_ls, best_ss)

        if np_ > best_np:
            best_ll, best_sl = ll, sl
            best_ls, best_ss = ls, ss
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, ll, sl, ls, ss,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, ll, sl, ls, ss, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/800K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  ll_fine — LL step=1 scan (2-8), fine SL and SS grid
    #      LL(2-8 s1)×SL(0.8-1.3 s0.1)×LS(46-56 s2)×SS(2.0-2.6 s0.1) = 7×6×6×7 = 1764
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ll_fine"
    _c = _cfg(_n, (2, 8, 1), (0.8, 1.3, 0.1), (46, 56, 2), (2.0, 2.6, 0.1))
    log.info("A01  LL(2-8 s1)×SL(0.8-1.3 s0.1)×LS(46-56 s2)×SS(2.0-2.6 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  ss_high — probe higher SS (2.4-3.5) for rarer, bigger short entries
    #      LL(2-6 s1)×SL(0.9-1.15 s0.05)×LS(46-60 s2)×SS(2.4-3.5 s0.2) = 5×6×8×6 = 1440
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ss_high"
    _c = _cfg(_n, (2, 6, 1), (0.9, 1.15, 0.05), (46, 60, 2), (2.4, 3.5, 0.2))
    log.info("A02  LL(2-6 s1)×SL(0.9-1.15 s0.05)×LS(46-60 s2)×SS(2.4-3.5 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ls_extend — extend LS to 55-90 with R1 params zone
    #      LL(2-5 s1)×SL(0.85-1.15 s0.1)×LS(55-90 s5)×SS(2.0-3.0 s0.2) = 4×4×8×6 = 768
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ls_extend"
    _c = _cfg(_n, (2, 5, 1), (0.85, 1.15, 0.1), (55, 90, 5), (2.0, 3.0, 0.2))
    log.info("A03  LL(2-5 s1)×SL(0.85-1.15 s0.1)×LS(55-90 s5)×SS(2.0-3.0 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  sl_fine — broad SL scan (0.6-1.6) to confirm SL=1.04 is the peak
    #      LL(2-6 s1)×SL(0.6-1.6 s0.1)×LS(46-58 s2)×SS(2.0-2.6 s0.2) = 5×11×7×4 = 1540
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_sl_fine"
    _c = _cfg(_n, (2, 6, 1), (0.6, 1.6, 0.1), (46, 58, 2), (2.0, 2.6, 0.2))
    log.info("A04  LL(2-6 s1)×SL(0.6-1.6 s0.1)×LS(46-58 s2)×SS(2.0-2.6 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ss_low — lower SS (1.4-2.1) for more frequent short entries
    #      LL(3-7 s1)×SL(0.8-1.2 s0.1)×LS(40-55 s3)×SS(1.4-2.0 s0.1) = 5×5×6×7 = 1050
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ss_low"
    _c = _cfg(_n, (3, 7, 1), (0.8, 1.2, 0.1), (40, 55, 3), (1.4, 2.0, 0.1))
    log.info("A05  LL(3-7 s1)×SL(0.8-1.2 s0.1)×LS(40-55 s3)×SS(1.4-2.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ll_tiny_fine — deep dive on LL=2-4 with very fine SL/SS step
    #      LL(2-4 s1)×SL(0.7-1.3 s0.05)×LS(44-58 s2)×SS(2.0-2.8 s0.1) = 3×13×8×9 = 2808
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ll_tiny_fine"
    _c = _cfg(_n, (2, 4, 1), (0.7, 1.3, 0.05), (44, 58, 2), (2.0, 2.8, 0.1))
    log.info("A06  LL(2-4 s1)×SL(0.7-1.3 s0.05)×LS(44-58 s2)×SS(2.0-2.8 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  broad_zone — wider grid around the LL=2-10 zone
    #      LL(2-10 s1)×SL(0.7-1.4 s0.1)×LS(42-62 s4)×SS(1.8-2.8 s0.2) = 9×8×6×6 = 2592
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_broad_zone"
    _c = _cfg(_n, (2, 10, 1), (0.7, 1.4, 0.1), (42, 62, 4), (1.8, 2.8, 0.2))
    log.info("A07  LL(2-10 s1)×SL(0.7-1.4 s0.1)×LS(42-62 s4)×SS(1.8-2.8 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  high_ls_ss — very long LS (60-100) with very high SS (2.5-4.0)
    #      LL(2-5 s1)×SL(0.9-1.2 s0.1)×LS(60-100 s5)×SS(2.5-4.0 s0.3) = 4×4×9×6 = 864
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_high_ls_ss"
    _c = _cfg(_n, (2, 5, 1), (0.9, 1.2, 0.1), (60, 100, 5), (2.5, 4.0, 0.3))
    log.info("A08  LL(2-5 s1)×SL(0.9-1.2 s0.1)×LS(60-100 s5)×SS(2.5-4.0 s0.3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  global_r2 — broad global re-survey with R1 knowledge applied
    #      LL(2-52 s10)×SL(0.5-2.5 s0.5)×LS(10-80 s10)×SS(1.0-3.5 s0.5) = 6×5×8×6 = 1440
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_global_r2"
    _c = _cfg(_n, (2, 52, 10), (0.5, 2.5, 0.5), (10, 80, 10), (1.0, 3.5, 0.5))
    log.info("A09  LL(2-52 s10)×SL(0.5-2.5 s0.5)×LS(10-80 s10)×SS(1.0-3.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — zoom around best NP found in A01-A09
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (5,  0.4, 10, 0.5),
            (4,  0.3,  7, 0.35),
            (3,  0.2,  5, 0.25),
            (2,  0.15, 3, 0.2),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.05, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.05, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  fine_zoom — tighter zoom, SL step=0.02, SS step=0.05
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_fine_zoom"
    log.info("A11  fine_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (4,  0.15, 7, 0.2),
            (3,  0.1,  5, 0.15),
            (2,  0.08, 4, 0.1),
            (1,  0.06, 3, 0.08),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.02, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.05, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  ultra_fine_verify — finest zoom, SL step=0.01, SS step=0.02
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_ultra_fine_verify"
    log.info("A12  ultra_fine_verify — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (3,  0.1,  6, 0.15),
            (2,  0.08, 4, 0.1),
            (1,  0.06, 3, 0.08),
            (1,  0.04, 2, 0.06),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.01,  SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.02,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A12  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  GC Daily countertrend_LS Round-2 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R1 seed NP: %.0f → R2 best: %.0f  (Δ %.0f  %.1f%%)",
             SEED_NP, best_np, best_np - SEED_NP,
             (best_np - SEED_NP) / SEED_NP * 100)
    log.info("  Target %.0f: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{TARGET_NP - best_np:.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LENGTH_LONG": best_ll, "STDDEV_LONG": best_sl,
            "LENGTH_SHORT": best_ls, "STDDEV_SHORT": best_ss,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>800K: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="GC Daily countertrend_LS NP>800K Round-2 search")
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
