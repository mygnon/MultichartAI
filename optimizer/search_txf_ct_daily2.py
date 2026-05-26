"""
search_txf_ct_daily2.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Daily, Round 2

R1 findings (search_txf_ct_daily.py):
  Best NP : LL=20 SL=0.28 LS=56 SS=1.27  NP=3,270,600  MDD=-1,189,800  Obj=8,990,439  trades=48
  Gap: -59.1% from 8M target (need 2.45× more NP)
  MDD is fixed at -1,189,800 across A04-A12 (structural worst-trade floor)
  ~7 trades/year (48 trades / 7 years) — daily bars generate very few signals
  NP gain rate collapsed: A09→A10→A11→A12 = 3.7%→2.3%→0.2%→0% — ceiling forming at ~3.27M

  Top results from R1:
    Main regime : LL=20  SL=0.28 LS=56 SS=1.27  NP=3,270,600  Obj=8,990,439
    NQ-analog   : LL=10  SL=0.50 LS=60 SS=1.40  NP=3,069,800  Obj=8,241,798
    TXF-analog  : LL=20  SL=0.30 LS=53 SS=1.30  NP=3,081,400  Obj=7,980,355
    A01 tight-SS: LL=27  SL=0.20 LS=55 SS=0.50  NP=2,759,000  Obj=6,867,630

  Unexplored regions in R1:
    - LS > 60 (R1 max was LS=60 but with step=4/5/10)
    - Fine LS step=1 around LS=56 (R1 adaptive zoom used step=1 but only A09-A12)
    - Fine LL step=1 around LL=20 (partially done in A11)
    - SL < 0.2 (ultra-tight SL below R1's coarse lower bound)
    - SS=0.3-0.7 tight-SS regime (A01 hint at SS=0.5 was not zoomed)
    - SS=2.5-4.0 loose-SS regime (only coarse scan in A01)
    - LL=2-8 ultra-short (A08 step=7, very coarse)
    - SL=0.22-0.35 fine step=0.01 (R1 best SL=0.28 only found at step=0.02)

R2 goal: Exhaustive coverage of all unexplored regions + ultra-fine tune around R1 best.
R2 strategy:
  A01 fine_ls      : LS step=1 around LS=56, tight LL/SL/SS
  A02 fine_ll      : LL step=1 around LL=20, tight SL/LS/SS
  A03 long_ls      : LS=65-100 (unexplored; longer short periods may find new regime)
  A04 tight_ss     : SS=0.10-0.60 tight short-entry band (A01 hint at SS=0.5)
  A05 loose_ss     : SS=2.5-4.0 loose bands (rarely explored)
  A06 ultra_short_ll: LL=2-8 (ultra-short long bands; A08 was too coarse at step=7)
  A07 fine_sl      : SL step=0.01 around SL=0.28 (precise SL peak)
  A08 a01_tight_ss_zone: Zoom on A01 tight-SS peak (LL=27, SL=0.2, LS=55, SS=0.5)
  A09 global_r2    : Different coarse global scan (fill remaining gaps)
  A10 adaptive_zoom: step=(1,0.02,1,0.01) cascade from R2 best
  A11 fine_zoom    : step=(1,0.01,1,0.01) cascade from R2 best
  A12 ultra_fine_ss: SS step=0.005 ultra-fine final tune

Attempt schedule (≤5,000 combos each):
  A01  1300  LL(18-22 s1)×SL(0.24-0.32 s0.02)×LS(50-62 s1)×SS(1.20-1.35 s0.05)
  A02  1560  LL(14-26 s1)×SL(0.24-0.34 s0.02)×LS(52-60 s2)×SS(1.22-1.34 s0.04)
  A03   720  LL(15-25 s5)×SL(0.2-0.6 s0.1)×LS(65-100 s5)×SS(1.0-2.0 s0.2)
  A04   924  LL(15-25 s5)×SL(0.2-0.5 s0.05)×LS(50-65 s5)×SS(0.10-0.60 s0.05)
  A05   630  LL(15-25 s5)×SL(0.2-0.6 s0.1)×LS(40-65 s5)×SS(2.5-4.0 s0.25)
  A06  1440  LL(2-8 s2)×SL(0.3-1.2 s0.1)×LS(40-65 s5)×SS(1.0-2.0 s0.2)
  A07  1120  LL(18-22 s1)×SL(0.22-0.35 s0.01)×LS(53-59 s2)×SS(1.22-1.34 s0.04)
  A08  1350  LL(22-30 s2)×SL(0.10-0.30 s0.05)×LS(50-60 s2)×SS(0.30-0.70 s0.05)
  A09  1296  LL(3-28 s5)×SL(0.15-1.65 s0.3)×LS(15-65 s10)×SS(0.8-3.3 s0.5)
  A10  ≤5000 adaptive zoom step=(1,0.02,1,0.01) cascade
  A11  ≤5000 fine zoom step=(1,0.01,1,0.01) cascade
  A12  ≤5000 ultra-fine SS step=0.005 final tune
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
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "SFJ_15Dworkshop_lesson5_countertrend_LS"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.05, 20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.05, 20.0

# R1 NP-champion as seed
SEED_LL, SEED_SL = 20.0, 0.28
SEED_LS, SEED_SS = 56.0, 1.27
SEED_NP          = 3_270_600.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_daily2_{int(time.time())}.log"
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
        name=f"TXFCTD2_{name}",
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
    return OUTPUT_DIR / f"TXFCTD2_{name}_raw.csv"


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
    log.info("=== Starting TXFCTD2_%s (%d combos) ===", name, cfg.total_runs())
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
    """Priority: target met → highest Obj; else highest NP (target-chasing mode)."""
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
            "r1_best":  "LL=20 SL=0.28 LS=56 SS=1.27 NP=3270600 MDD=-1189800 trades=48",
            "r1_gap":   "-59.1% from 8M target; NP gain rate collapsed to 0% at R1 A12",
            "r2_focus": "Exhaustive unexplored: LS>60, LS step=1, LL step=1, tight-SS, loose-SS, LL<8, fine SL",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_daily2.json"
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
    log.info("  TXF Daily countertrend_LS NP>8,000,000 TWD — Round 2")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R1 best: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R1 gap: -59.1%% from 8M. R2 exhausts all unexplored regions.")
    log.info("  Target: %.0f TWD", TARGET_NP)
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
                 "★TARGET★" if met else ("%.0f/8M" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  fine_ls — LS step=1 around LS=56, narrow LL/SL/SS
    #      LL(18-22 s1)×SL(0.24-0.32 s0.02)×LS(50-62 s1)×SS(1.20-1.35 s0.05) = 5×5×13×4=1300
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_ls"
    _c = _cfg(_n, (18, 22, 1), (0.24, 0.32, 0.02), (50, 62, 1), (1.20, 1.35, 0.05))
    log.info("A01  LL(18-22 s1)×SL(0.24-0.32 s0.02)×LS(50-62 s1)×SS(1.20-1.35 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  fine_ll — LL step=1 around LL=20, narrow SL/LS/SS
    #      LL(14-26 s1)×SL(0.24-0.34 s0.02)×LS(52-60 s2)×SS(1.22-1.34 s0.04) = 13×6×5×4=1560
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_fine_ll"
    _c = _cfg(_n, (14, 26, 1), (0.24, 0.34, 0.02), (52, 60, 2), (1.22, 1.34, 0.04))
    log.info("A02  LL(14-26 s1)×SL(0.24-0.34 s0.02)×LS(52-60 s2)×SS(1.22-1.34 s0.04)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  long_ls — LS=65-100 unexplored territory
    #      LL(15-25 s5)×SL(0.2-0.6 s0.1)×LS(65-100 s5)×SS(1.0-2.0 s0.2) = 3×5×8×6=720
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_long_ls"
    _c = _cfg(_n, (15, 25, 5), (0.2, 0.6, 0.1), (65, 100, 5), (1.0, 2.0, 0.2))
    log.info("A03  LL(15-25 s5)×SL(0.2-0.6 s0.1)×LS(65-100 s5)×SS(1.0-2.0 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  tight_ss_regime — SS=0.10-0.60 (A01 hint at SS=0.5 unexplored in zoom)
    #      LL(15-25 s5)×SL(0.20-0.50 s0.05)×LS(50-65 s5)×SS(0.10-0.60 s0.05) = 3×7×4×11=924
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_tight_ss_regime"
    _c = _cfg(_n, (15, 25, 5), (0.20, 0.50, 0.05), (50, 65, 5), (0.10, 0.60, 0.05))
    log.info("A04  LL(15-25 s5)×SL(0.20-0.50 s0.05)×LS(50-65 s5)×SS(0.10-0.60 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  loose_ss — SS=2.5-4.0 loose bands (under-explored)
    #      LL(15-25 s5)×SL(0.2-0.6 s0.1)×LS(40-65 s5)×SS(2.5-4.0 s0.25) = 3×5×6×7=630
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_loose_ss"
    _c = _cfg(_n, (15, 25, 5), (0.2, 0.6, 0.1), (40, 65, 5), (2.5, 4.0, 0.25))
    log.info("A05  LL(15-25 s5)×SL(0.2-0.6 s0.1)×LS(40-65 s5)×SS(2.5-4.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ultra_short_ll — LL=2-8 (finer than A08 R1 step=7)
    #      LL(2-8 s2)×SL(0.3-1.2 s0.1)×LS(40-65 s5)×SS(1.0-2.0 s0.2) = 4×10×6×6=1440
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ultra_short_ll"
    _c = _cfg(_n, (2, 8, 2), (0.3, 1.2, 0.1), (40, 65, 5), (1.0, 2.0, 0.2))
    log.info("A06  LL(2-8 s2)×SL(0.3-1.2 s0.1)×LS(40-65 s5)×SS(1.0-2.0 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  fine_sl — SL step=0.01 precision (R1 found SL=0.28 at step=0.02)
    #      LL(18-22 s1)×SL(0.22-0.35 s0.01)×LS(53-59 s2)×SS(1.22-1.34 s0.04) = 5×14×4×4=1120
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_fine_sl"
    _c = _cfg(_n, (18, 22, 1), (0.22, 0.35, 0.01), (53, 59, 2), (1.22, 1.34, 0.04))
    log.info("A07  LL(18-22 s1)×SL(0.22-0.35 s0.01)×LS(53-59 s2)×SS(1.22-1.34 s0.04)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  a01_tight_ss_zone — Zoom on R1 A01 tight-SS peak (LL=27, SL=0.2, LS=55, SS=0.5)
    #      LL(22-30 s2)×SL(0.10-0.30 s0.05)×LS(50-60 s2)×SS(0.30-0.70 s0.05) = 5×5×6×9=1350
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_a01_tight_ss_zone"
    _c = _cfg(_n, (22, 30, 2), (0.10, 0.30, 0.05), (50, 60, 2), (0.30, 0.70, 0.05))
    log.info("A08  LL(22-30 s2)×SL(0.10-0.30 s0.05)×LS(50-60 s2)×SS(0.30-0.70 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  global_r2 — Different coarse global (fill remaining gaps)
    #      LL(3-28 s5)×SL(0.15-1.65 s0.3)×LS(15-65 s10)×SS(0.8-3.3 s0.5) = 6×6×6×6=1296
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_global_r2"
    _c = _cfg(_n, (3, 28, 5), (0.15, 1.65, 0.3), (15, 65, 10), (0.8, 3.3, 0.5))
    log.info("A09  LL(3-28 s5)×SL(0.15-1.65 s0.3)×LS(15-65 s10)×SS(0.8-3.3 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — SL step=0.02, SS step=0.01 cascade
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.08, 3, 0.06),   # 5×9×7×13=4095
            (2, 0.06, 2, 0.06),   # 5×7×5×13=2275
            (2, 0.04, 2, 0.04),   # 5×5×5×9=1125
            (1, 0.04, 1, 0.04),   # 3×5×3×9=405
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.02, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.01, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  fine_zoom — SL step=0.01, SS step=0.01 fine cascade
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_fine_zoom"
    log.info("A11  fine_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.06, 2, 0.06),   # 5×13×5×13=4225
            (2, 0.04, 2, 0.05),   # 5×9×5×11=2475
            (1, 0.05, 1, 0.05),   # 3×11×3×11=1089
            (1, 0.03, 1, 0.04),   # 3×7×3×9=567
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.01, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.01, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  ultra_fine_ss — SS step=0.005 ultra-precision final tune
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_ultra_fine_ss"
    log.info("A12  ultra_fine_ss — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (1, 0.04, 2, 0.06),   # 3×9×5×25=3375
            (1, 0.04, 1, 0.06),   # 3×9×3×25=2025
            (1, 0.02, 1, 0.06),   # 3×5×3×25=1125
            (1, 0.02, 1, 0.04),   # 3×5×3×17=765
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.02,  SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.005, SS_LO, SS_HI)
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
    log.info("  TXF Daily countertrend_LS Round-2 COMPLETE")
    log.info("  R1 seed NP: %.0f → R2 best: %.0f  (Δ %.0f  %+.1f%%)",
             SEED_NP, best_np, best_np - SEED_NP, (best_np / SEED_NP - 1) * 100)
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  Target %.0f TWD: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{max(0, TARGET_NP - best_np):.0f})")
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
    print(f"Target NP>8M TWD: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="TXF Daily countertrend_LS NP>8M TWD Round-2 search")
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
