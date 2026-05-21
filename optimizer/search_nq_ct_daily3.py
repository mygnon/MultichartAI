"""
search_nq_ct_daily3.py — SFJ_15Dworkshop_lesson5_countertrend_LS on CME.NQ HOT Daily, Round 3

R2 findings (search_nq_ct_daily2.py):
  Best NP : LL=8  SL=0.7 LS=47 SS=1.9  NP=431,500  MDD=-81,535  Obj=2,283,587  trades=36
  Alt Obj : LL=2  SL=any LS=12 SS=2.0  NP=408,575  MDD=-75,690  Obj=2,205,600  trades=58
  Gap: -38.4% from 700K target  (improvement from R1 +11.3%)
  Key: MDD=-81,535 shared by 63% of A10 profitable combos (structural floor — same worst trade)
  Key: LL=2, LS=12 breaks to MDD=-75,690 (different worst-trade sequence, SL-insensitive)
  Key: LL=2, LS=10 gives MDD=-72,655 (best MDD floor) but lower NP=376K

R3 focus:
  A01 ultra_fine_a10    : very fine around A10 winner (step=1 period, step=0.05 stddev)
  A02 fine_ls_scan      : fine LS=38-55 scan with main regime
  A03 ll2_regime        : explore LL=2-5 × LS=8-20 (different MDD floor region)
  A04 ll2_fine          : ultra-fine around LL=2, LS=12, SS=2.0 winner
  A05 ls_35_50          : LS=35-50 step 1 with main regime LL/SL
  A06 ss_ultra_fine     : SS=1.6-2.2 step 0.05 with tight LL/SL/LS
  A07 ss_range          : SS=1.5-2.5 wider scan across LS variants
  A08 ll_extend         : LL=10-20, can longer LL improve?
  A09 very_fine_ls40    : LL(6-12)×SL(0.55-0.85 s0.05)×LS(40-50 s1)×SS(1.75-2.05 s0.05)
  A10 adaptive_zoom     : fine steps (step=1 period, step=0.05 stddev)
  A11 ll2_wide          : LL=2-4 × broad SL/LS sweep to characterize full LL=2 regime
  A12 global_r3         : different coarse global (fills R1+R2 gaps)

Attempt schedule:
  A01  2310   LL(6-11 s1)×SL(0.55-0.85 s0.05)×LS(43-53 s1)×SS(1.7-2.1 s0.1)
  A02  3402   LL(6-12 s1)×SL(0.50-0.90 s0.05)×LS(38-55 s1)×SS(1.8-2.0 s0.1)
  A03  2496   LL(2-5 s1)×SL(0.10-1.50 s0.20)×LS(8-20 s1)×SS(1.5-2.5 s0.2)
  A04  2184   LL(2-4 s1)×SL(0.10-0.80 s0.10)×LS(10-22 s1)×SS(1.7-2.3 s0.1)
  A05  3360   LL(6-12 s1)×SL(0.50-0.90 s0.10)×LS(35-50 s1)×SS(1.7-2.2 s0.1)
  A06  1404   LL(7-10 s1)×SL(0.60-0.80 s0.10)×LS(44-52 s1)×SS(1.60-2.20 s0.05)
  A07  1375   LL(6-10 s1)×SL(0.50-0.90 s0.10)×LS(44-52 s2)×SS(1.50-2.50 s0.10)
  A08  3960   LL(10-20 s1)×SL(0.50-1.00 s0.10)×LS(44-55 s1)×SS(1.7-2.1 s0.1)
  A09  3773   LL(6-12 s1)×SL(0.55-0.85 s0.05)×LS(40-50 s1)×SS(1.75-2.05 s0.05)
  A10  ≤5000  adaptive zoom (step=1 period, step=0.05 stddev)
  A11  3960   LL(2-4 s1)×SL(0.10-2.00 s0.20)×LS(8-50 s2)×SS(1.5-2.5 s0.2)
  A12  3136   LL(2-58 s8)×SL(0.10-3.00 s0.50)×LS(2-58 s8)×SS(0.10-3.00 s0.50)
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
SYMBOL     = "CME.NQ HOT"
SIGNAL     = "SFJ_15Dworkshop_lesson5_countertrend_LS"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_ct_daily3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.05, 20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.05, 20.0

# Seeds from R2 best NP winner
SEED_LL, SEED_SL = 8.0,  0.7
SEED_LS, SEED_SS = 47.0, 1.9
SEED_NP          = 431_500.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_ct_daily3_{int(time.time())}.log"
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
        name=f"NQCTD3_{name}",
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
    return OUTPUT_DIR / f"NQCTD3_{name}_raw.csv"


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
    log.info("=== Starting NQCTD3_%s (%d combos) ===", name, cfg.total_runs())
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
        "round": 3,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "daily",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r2_best":  "LL=8 SL=0.7 LS=47 SS=1.9 NP=431500 MDD=-81535 trades=36",
            "r2_alt":   "LL=2 SL=any LS=12 SS=2.0 NP=408575 MDD=-75690 Obj=2205600 trades=58 (better MDD floor)",
            "r3_focus": "ultra-fine A10, LL=2 regime, LS=35-55 fine, SS ultra-fine, ll_extend",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_ct_daily3.json"
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
    log.info("  NQ Daily countertrend_LS NP>700K — Round 3")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  Timeframe: DAILY (1440 min bars)")
    log.info("  R2 best NP: LL=8 SL=0.7 LS=47 SS=1.9 NP=431500 MDD=-81535")
    log.info("  R2 alt: LL=2 SL=any LS=12 SS=2.0 NP=408575 MDD=-75690")
    log.info("  Target: %.0f USD  Gap from R2: +%.0f", TARGET_NP, TARGET_NP - SEED_NP)
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
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  ultra_fine_a10 — very fine around R2 A10 winner
    #      LL(6-11 s1)×SL(0.55-0.85 s0.05)×LS(43-53 s1)×SS(1.7-2.1 s0.1)  = 6×7×11×5 = 2310
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ultra_fine_a10"
    _c = _cfg(_n, (6, 11, 1), (0.55, 0.85, 0.05), (43, 53, 1), (1.7, 2.1, 0.1))
    log.info("A01  LL(6-11 s1)×SL(0.55-0.85 s0.05)×LS(43-53 s1)×SS(1.7-2.1 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  fine_ls_scan — fine LS=38-55 with main regime LL/SL
    #      LL(6-12 s1)×SL(0.50-0.90 s0.05)×LS(38-55 s1)×SS(1.8-2.0 s0.1)  = 7×9×18×3 = 3402
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_fine_ls_scan"
    _c = _cfg(_n, (6, 12, 1), (0.50, 0.90, 0.05), (38, 55, 1), (1.8, 2.0, 0.1))
    log.info("A02  LL(6-12 s1)×SL(0.50-0.90 s0.05)×LS(38-55 s1)×SS(1.8-2.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ll2_regime — explore LL=2-5 × LS=8-20 (different MDD floor)
    #      LL(2-5 s1)×SL(0.10-1.50 s0.20)×LS(8-20 s1)×SS(1.5-2.5 s0.2)  = 4×8×13×6 = 2496
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ll2_regime"
    _c = _cfg(_n, (2, 5, 1), (0.10, 1.50, 0.20), (8, 20, 1), (1.5, 2.5, 0.2))
    log.info("A03  LL(2-5 s1)×SL(0.10-1.50 s0.20)×LS(8-20 s1)×SS(1.5-2.5 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ll2_fine — ultra-fine around LL=2, LS=12, SS=2.0 winner
    #      LL(2-4 s1)×SL(0.10-0.80 s0.10)×LS(10-22 s1)×SS(1.7-2.3 s0.1)  = 3×8×13×7 = 2184
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ll2_fine"
    _c = _cfg(_n, (2, 4, 1), (0.10, 0.80, 0.10), (10, 22, 1), (1.7, 2.3, 0.1))
    log.info("A04  LL(2-4 s1)×SL(0.10-0.80 s0.10)×LS(10-22 s1)×SS(1.7-2.3 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ls_35_50 — LS=35-50 step 1 with main regime
    #      LL(6-12 s1)×SL(0.50-0.90 s0.10)×LS(35-50 s1)×SS(1.7-2.2 s0.1)  = 7×5×16×6 = 3360
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ls_35_50"
    _c = _cfg(_n, (6, 12, 1), (0.50, 0.90, 0.10), (35, 50, 1), (1.7, 2.2, 0.1))
    log.info("A05  LL(6-12 s1)×SL(0.50-0.90 s0.10)×LS(35-50 s1)×SS(1.7-2.2 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ss_ultra_fine — SS=1.60-2.20 step 0.05 with tight LL/SL/LS center
    #      LL(7-10 s1)×SL(0.60-0.80 s0.10)×LS(44-52 s1)×SS(1.60-2.20 s0.05)  = 4×3×9×13 = 1404
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ss_ultra_fine"
    _c = _cfg(_n, (7, 10, 1), (0.60, 0.80, 0.10), (44, 52, 1), (1.60, 2.20, 0.05))
    log.info("A06  LL(7-10 s1)×SL(0.60-0.80 s0.10)×LS(44-52 s1)×SS(1.60-2.20 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  ss_range — SS=1.5-2.5 wider scan across LS variants
    #      LL(6-10 s1)×SL(0.50-0.90 s0.10)×LS(44-52 s2)×SS(1.50-2.50 s0.10)  = 5×5×5×11 = 1375
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ss_range"
    _c = _cfg(_n, (6, 10, 1), (0.50, 0.90, 0.10), (44, 52, 2), (1.50, 2.50, 0.10))
    log.info("A07  LL(6-10 s1)×SL(0.50-0.90 s0.10)×LS(44-52 s2)×SS(1.50-2.50 s0.10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ss, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  ll_extend — LL=10-20, can longer LL periods improve NP?
    #      LL(10-20 s1)×SL(0.50-1.00 s0.10)×LS(44-55 s1)×SS(1.7-2.1 s0.1)  = 11×6×12×5 = 3960
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_ll_extend"
    _c = _cfg(_n, (10, 20, 1), (0.50, 1.00, 0.10), (44, 55, 1), (1.7, 2.1, 0.1))
    log.info("A08  LL(10-20 s1)×SL(0.50-1.00 s0.10)×LS(44-55 s1)×SS(1.7-2.1 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  very_fine_ls40 — finest resolution around LS=40-50 core
    #      LL(6-12 s1)×SL(0.55-0.85 s0.05)×LS(40-50 s1)×SS(1.75-2.05 s0.05)  = 7×7×11×7 = 3773
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_very_fine_ls40"
    _c = _cfg(_n, (6, 12, 1), (0.55, 0.85, 0.05), (40, 50, 1), (1.75, 2.05, 0.05))
    log.info("A09  LL(6-12 s1)×SL(0.55-0.85 s0.05)×LS(40-50 s1)×SS(1.75-2.05 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — fine steps (step=1 period, step=0.05 stddev)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (8,  0.5,  8, 0.5),
            (5,  0.3,  5, 0.3),
            (3,  0.2,  3, 0.2),
            (2,  0.15, 2, 0.15),
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
    # A11  ll2_wide — LL=2-4 × broad SL/LS to characterize full LL=2 regime
    #      LL(2-4 s1)×SL(0.10-2.00 s0.20)×LS(8-50 s2)×SS(1.5-2.5 s0.2)  = 3×10×22×6 = 3960
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_ll2_wide"
    _c = _cfg(_n, (2, 4, 1), (0.10, 2.00, 0.20), (8, 50, 2), (1.5, 2.5, 0.2))
    log.info("A11  LL(2-4 s1)×SL(0.10-2.00 s0.20)×LS(8-50 s2)×SS(1.5-2.5 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  global_r3 — different coarse global sweep (R1 used step 20, R2 used step 11)
    #      LL(2-58 s8)×SL(0.10-3.00 s0.50)×LS(2-58 s8)×SS(0.10-3.00 s0.50)  = 8×7×8×7 = 3136
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_global_r3"
    _c = _cfg(_n, (2, 58, 8), (0.10, 3.00, 0.50), (2, 58, 8), (0.10, 3.00, 0.50))
    log.info("A12  LL(2-58 s8)×SL(0.10-3.00 s0.50)×LS(2-58 s8)×SS(0.10-3.00 s0.50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  NQ Daily countertrend_LS Round-3 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
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
        description="NQ Daily countertrend_LS NP>700K Round-3 search")
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
