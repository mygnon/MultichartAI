"""
search_gc_ct_hourly.py — SFJ_15Dworkshop_lesson5_countertrend_LS on CME.GC HOT Hourly, Round 1

No prior data. Strategy: BUY when Close crosses over lower BB; SELLSHORT when Close crosses under upper BB.
Reversal-only exits (no STP or LMT).

R1 strategy: broad exploration — no prior data, need to map the full parameter landscape.
  Reference: NQ CT best: LL=17 SL=0.2 LS=45 SS=1.4 (ultra-tight SL)
             TXF CT best: LL=22 SL=0.425 LS=43 SS=1.771

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 global_broad      : LL(2-27 s5)×SL(0.2-2.0 s0.3)×LS(5-55 s10)×SS(0.5-3.0 s0.5)     = 6×7×6×6  = 1512
  A02 global_fill       : LL(4-29 s5)×SL(0.35-1.85 s0.3)×LS(10-60 s10)×SS(0.75-2.75 s0.5) = 6×6×6×5  = 1080
  A03 nq_analog         : LL(12-22 s2)×SL(0.1-0.5 s0.1)×LS(38-52 s2)×SS(0.8-2.0 s0.2)    = 6×5×8×7  = 1680
  A04 txf_analog        : LL(16-28 s2)×SL(0.2-0.8 s0.1)×LS(35-55 s4)×SS(1.3-2.3 s0.2)    = 7×7×6×6  = 1764
  A05 tight_sl_short_ll : LL(2-16 s2)×SL(0.1-0.5 s0.1)×LS(30-60 s5)×SS(0.8-2.0 s0.3)    = 8×5×7×5  = 1400
  A06 moderate_sl       : LL(10-25 s3)×SL(0.5-1.5 s0.2)×LS(25-55 s5)×SS(1.0-2.5 s0.3)    = 6×6×7×6  = 1512
  A07 long_ls           : LL(8-24 s4)×SL(0.2-1.0 s0.2)×LS(55-90 s5)×SS(0.8-2.0 s0.3)     = 5×5×8×5  = 1000
  A08 short_ls          : LL(6-22 s2)×SL(0.2-0.8 s0.2)×LS(10-40 s5)×SS(0.5-1.5 s0.25)    = 9×4×7×5  = 1260
  A09 tight_ss          : LL(10-26 s4)×SL(0.1-0.5 s0.1)×LS(30-60 s5)×SS(0.2-0.8 s0.1)    = 5×5×7×7  = 1225
  A10 adaptive_zoom     : (dynamic from R1 best NP)
  A11 global_wide       : LL(5-80 s15)×SL(0.15-2.25 s0.3)×LS(5-80 s15)×SS(0.4-2.8 s0.4)  = 6×8×6×7  = 2016
  A12 global_final      : LL(5-75 s10)×SL(0.2-2.2 s0.4)×LS(5-75 s10)×SS(0.5-3.0 s0.5)    = 8×6×8×6  = 2304

NOTE: CME.GC HOT chart with SFJ_15Dworkshop_lesson5_countertrend_LS signal must be open
      in 20260521SFJ_Bollinger_AI.wsp before running.
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_ct_hourly_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.1,  20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.1,  20.0

# Initial seed — NQ CT analog (no prior GC data)
SEED_LL, SEED_SL = 17.0, 0.2
SEED_LS, SEED_SS = 45.0, 1.4
SEED_NP          = 0.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_ct_hourly1_{int(time.time())}.log"
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
        name=f"GCCT1_{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
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
    return OUTPUT_DIR / f"GCCT1_{name}_raw.csv"


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
    log.info("=== Starting GCCT1_%s (%d combos) ===", name, cfg.total_runs())
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
        "round": 1,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "hourly",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_focus": "Broad exploration — no prior data. NQ analog: LL=17 SL=0.2 LS=45 SS=1.4",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_gc_ct_hourly.json"
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
    log.info("  GC Hourly countertrend_LS NP>800K — Round 1")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  No prior data — broad exploration")
    log.info("  Target: %.0f USD", TARGET_NP)
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
    # A01  global_broad — LL(2-27 s5)×SL(0.2-2.0 s0.3)×LS(5-55 s10)×SS(0.5-3.0 s0.5) = 1512
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_global_broad"
    _c = _cfg(_n, (2, 27, 5), (0.2, 2.0, 0.3), (5, 55, 10), (0.5, 3.0, 0.5))
    log.info("A01  LL(2-27 s5)×SL(0.2-2.0 s0.3)×LS(5-55 s10)×SS(0.5-3.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  global_fill — staggered grid to fill A01 gaps = 1080
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_global_fill"
    _c = _cfg(_n, (4, 29, 5), (0.35, 1.85, 0.3), (10, 60, 10), (0.75, 2.75, 0.5))
    log.info("A02  LL(4-29 s5)×SL(0.35-1.85 s0.3)×LS(10-60 s10)×SS(0.75-2.75 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  nq_analog — parameters around NQ CT champion (LL=17 SL=0.2 LS=45 SS=1.4) = 1680
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_nq_analog"
    _c = _cfg(_n, (12, 22, 2), (0.1, 0.5, 0.1), (38, 52, 2), (0.8, 2.0, 0.2))
    log.info("A03  LL(12-22 s2)×SL(0.1-0.5 s0.1)×LS(38-52 s2)×SS(0.8-2.0 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  txf_analog — parameters around TXF CT champion (LL=22 SL=0.425 LS=43 SS=1.771) = 1764
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_txf_analog"
    _c = _cfg(_n, (16, 28, 2), (0.2, 0.8, 0.1), (35, 55, 4), (1.3, 2.3, 0.2))
    log.info("A04  LL(16-28 s2)×SL(0.2-0.8 s0.1)×LS(35-55 s4)×SS(1.3-2.3 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  tight_sl_short_ll — ultra-tight SL with short LL (high-frequency entry) = 1400
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_tight_sl_short_ll"
    _c = _cfg(_n, (2, 16, 2), (0.1, 0.5, 0.1), (30, 60, 5), (0.8, 2.0, 0.3))
    log.info("A05  LL(2-16 s2)×SL(0.1-0.5 s0.1)×LS(30-60 s5)×SS(0.8-2.0 s0.3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  moderate_sl — moderate SL=0.5-1.5, mid-range regime = 1512
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_moderate_sl"
    _c = _cfg(_n, (10, 25, 3), (0.5, 1.5, 0.2), (25, 55, 5), (1.0, 2.5, 0.3))
    log.info("A06  LL(10-25 s3)×SL(0.5-1.5 s0.2)×LS(25-55 s5)×SS(1.0-2.5 s0.3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  long_ls — longer LS=55-90 (slower short-entry BB) = 1000
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_long_ls"
    _c = _cfg(_n, (8, 24, 4), (0.2, 1.0, 0.2), (55, 90, 5), (0.8, 2.0, 0.3))
    log.info("A07  LL(8-24 s4)×SL(0.2-1.0 s0.2)×LS(55-90 s5)×SS(0.8-2.0 s0.3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  short_ls — shorter LS=10-40 (faster short-entry BB) = 1260
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_short_ls"
    _c = _cfg(_n, (6, 22, 2), (0.2, 0.8, 0.2), (10, 40, 5), (0.5, 1.5, 0.25))
    log.info("A08  LL(6-22 s2)×SL(0.2-0.8 s0.2)×LS(10-40 s5)×SS(0.5-1.5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  tight_ss — very tight SS=0.2-0.8 (like TXF daily breakthrough) = 1225
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_tight_ss"
    _c = _cfg(_n, (10, 26, 4), (0.1, 0.5, 0.1), (30, 60, 5), (0.2, 0.8, 0.1))
    log.info("A09  LL(10-26 s4)×SL(0.1-0.5 s0.1)×LS(30-60 s5)×SS(0.2-0.8 s0.1)  %d combos",
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
            (10, 1.0, 10, 1.0),
            (7,  0.7,  7, 0.7),
            (5,  0.5,  5, 0.5),
            (3,  0.3,  3, 0.3),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.1,  SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.1,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  global_wide — staggered coarse global sweep (larger LL/LS range) = 2016
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_global_wide"
    _c = _cfg(_n, (5, 80, 15), (0.15, 2.25, 0.3), (5, 80, 15), (0.4, 2.8, 0.4))
    log.info("A11  LL(5-80 s15)×SL(0.15-2.25 s0.3)×LS(5-80 s15)×SS(0.4-2.8 s0.4)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  global_final — final coarse global to catch any remaining zones = 2304
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_global_final"
    _c = _cfg(_n, (5, 75, 10), (0.2, 2.2, 0.4), (5, 75, 10), (0.5, 3.0, 0.5))
    log.info("A12  LL(5-75 s10)×SL(0.2-2.2 s0.4)×LS(5-75 s10)×SS(0.5-3.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  GC Hourly countertrend_LS Round-1 COMPLETE")
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
        description="GC Hourly countertrend_LS NP>800K Round-1 search")
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
