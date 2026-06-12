"""
search_gc_hunter2_daily.py — SFJ_HUNTER2_NQ on CME.GC HOT Daily, Round 1

Strategy:
  BUY  when C > AVERAGE(C, LEN_L) AND EntriesToday(D)=0 → next bar STOP at Close + ATR_L×ATR(20)
  SHORT when C < AVERAGE(C, LEN_S) AND EntriesToday(D)=0 → next bar STOP at Close − ATR_S×ATR(20)
  Reversal exits only — no STP/LMT. 4 active params.
  Defaults: LEN_L=250, LEN_S=250, ATR_multiplier_L=2, ATR_multiplier_S=4.

Workspace: 20260101_SFJ_HUNTER_AI.wsp (CME.GC HOT Daily chart with SFJ_HUNTER2_NQ applied)
Target: NP > 700,000 USD, maximise NP²/|MDD|

Cross-instrument references:
  GC Hourly  ceiling $384,820 (LEN_L=5 LEN_S=37 ATR_L=0.8 ATR_S=5.9 288 trades/7yr)
  NQ Daily   ceiling $433,700 (LEN_L=6 LEN_S=85 ATR_L=0.08 ATR_S=1.15 72 trades/7yr)
  TXF Daily  ceiling 5.39M TWD (LEN_L=15 LEN_S=65 ATR_L=0.17 ATR_S=1.1 79 trades/7yr)
→ Daily bars yield far fewer trades; broad exploration covers tight-ATR, wide-ATR,
  GC-hourly analog (high ATR_S) and asymmetric MA regimes.

Attempt schedule (11 attempts, ≤5,000 combos each):
  A01 global_wide       : LEN_L(2-50 s8)×LEN_S(10-200 s30)×ATR_L(0.1-3.1 s0.5)×ATR_S(0.5-5.5 s1.0) = 7×7×7×6=2058
  A02 short_lenl        : LEN_L(1-15 s1)×LEN_S(30-180 s30)×ATR_L(0.05-0.55 s0.1)×ATR_S(0.5-2.5 s0.5) = 15×6×6×5=2700
  A03 gc_hourly_analog  : LEN_L(1-10 s1)×LEN_S(15-60 s5)×ATR_L(0.3-1.3 s0.25)×ATR_S(3.0-8.0 s1.0) = 10×10×5×6=3000
  A04 tight_atrl        : LEN_L(2-16 s2)×LEN_S(50-150 s25)×ATR_L(0.02-0.22 s0.04)×ATR_S(0.5-2.5 s0.5) = 8×5×6×5=1200
  A05 high_atrs         : LEN_L(2-20 s2)×LEN_S(20-100 s20)×ATR_L(0.1-1.6 s0.25)×ATR_S(2.0-8.0 s1.0) = 10×5×7×7=2450
  A06 medium_ma         : LEN_L(10-60 s10)×LEN_S(50-300 s50)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-3.0 s0.5) = 6×6×5×6=1080
  A07 short_lens        : LEN_L(2-20 s2)×LEN_S(5-50 s5)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-3.0 s0.5) = 10×10×5×6=3000
  A08 asym_ma           : LEN_L(1-10 s1)×LEN_S(100-400 s50)×ATR_L(0.05-0.55 s0.1)×ATR_S(0.5-2.5 s0.5) = 10×7×6×5=2100
  A09 adaptive_zoom1    : LL±3 LS±5 s1, ATR_L±0.1 s0.05, ATR_S±0.2 s0.1 → ≤7×11×5×5=1925
  A10 adaptive_zoom2    : LL±2 LS±3 s1, ATR_L±0.05 s0.025, ATR_S±0.15 s0.05 → ≤5×7×5×7=1225
  A11 adaptive_zoom3    : LL±2 LS±3 s1, ATR_L±0.05 s0.01, ATR_S±0.1 s0.05 → ≤5×7×11×5=1925
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260101_SFJ_HUNTER_AI.wsp"
SYMBOL     = "CME.GC HOT"
SIGNAL     = "SFJ_HUNTER2_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_hunter2_daily_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0   # USD

LL_LO, LL_HI = 1.0, 1000.0
LS_LO, LS_HI = 1.0, 1000.0
AL_LO, AL_HI = 0.01,  30.0
AS_LO, AS_HI = 0.1,   30.0

# Starting from strategy defaults (no prior GC daily HUNTER2 search)
SEED_LL   = 250.0
SEED_LS   = 250.0
SEED_ATRL = 2.0
SEED_ATRS = 4.0
SEED_NP   = 0.0

PREFIX = "GCH2D_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_hunter2_daily_{int(time.time())}.log"
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
         ll:   Tuple[float, float, float],
         ls:   Tuple[float, float, float],
         atrl: Tuple[float, float, float],
         atrs: Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    ll   = _safe(ll,   LL_LO, LL_HI)
    ls   = _safe(ls,   LS_LO, LS_HI)
    atrl = _safe(atrl, AL_LO, AL_HI)
    atrs = _safe(atrs, AS_LO, AS_HI)

    combos = n_vals(ll) * n_vals(ls) * n_vals(atrl) * n_vals(atrs)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
            ParamAxis("LEN_L",            *ll),
            ParamAxis("LEN_S",            *ls),
            ParamAxis("ATR_multiplier_L", *atrl),
            ParamAxis("ATR_multiplier_S", *atrs),
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


def champion(df, fb_ll, fb_ls, fb_atrl, fb_atrs):
    """Target-chasing mode: highest NP row drives zoom seed."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LEN_L"]), float(best["LEN_S"]),
                 float(best["ATR_multiplier_L"]), float(best["ATR_multiplier_S"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["LEN_L"]), float(best["LEN_S"]),
                float(best["ATR_multiplier_L"]), float(best["ATR_multiplier_S"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LEN_L"]), float(best["LEN_S"]),
                 float(best["ATR_multiplier_L"]), float(best["ATR_multiplier_S"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["LEN_L"]), float(best["LEN_S"]),
                float(best["ATR_multiplier_L"]), float(best["ATR_multiplier_S"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_ll, fb_ls, fb_atrl, fb_atrs,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, ll, ls, atrl, atrs, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "LEN_L": ll, "LEN_S": ls,
        "ATR_multiplier_L": atrl, "ATR_multiplier_S": atrs,
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
        "timeframe": "Daily (1440 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":      "BUY when C > AVERAGE(C,LEN_L) → next bar STOP at Close+ATR_L×ATR(20)",
            "exits":      "Reversal only — no STP or LMT; max 1 entry per day",
            "params":     "LEN_L (MA long filter), LEN_S (MA short filter), ATR_multiplier_L, ATR_multiplier_S",
            "references": "GC Hourly ceiling $384,820 (LEN_L=5 LEN_S=37 ATR_L=0.8 ATR_S=5.9 288 trades); NQ Daily ceiling $433,700 (LEN_L=6 LEN_S=85 ATR_L=0.08 ATR_S=1.15 72 trades); TXF Daily ceiling 5.39M TWD (LEN_L=15 LEN_S=65 ATR_L=0.17 ATR_S=1.1 79 trades)",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_gc_hunter2_daily.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_ll   = SEED_LL
    best_ls   = SEED_LS
    best_atrl = SEED_ATRL
    best_atrs = SEED_ATRS
    best_np   = SEED_NP
    best_obj  = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER2_NQ on CME.GC HOT Daily — Round 1")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_ll, best_ls, best_atrl, best_atrs
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_ll, best_ls, best_atrl, best_atrs,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        ll, ls, atrl, atrs, obj, np_, mdd, tr, met = champion(
            df, best_ll, best_ls, best_atrl, best_atrs)

        if np_ > best_np:
            best_ll, best_ls     = ll, ls
            best_atrl, best_atrs = atrl, atrs
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, ll, ls, atrl, atrs,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, ll, ls, atrl, atrs, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  global_wide — broad landscape with wide step sizes
    #      LEN_L(2-50 s8)×LEN_S(10-200 s30)×ATR_L(0.1-3.1 s0.5)×ATR_S(0.5-5.5 s1.0) = 7×7×7×6=2058
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_global_wide"
    _c = _cfg(_n, (2, 50, 8), (10, 200, 30), (0.1, 3.1, 0.5), (0.5, 5.5, 1.0))
    log.info("A01  LEN_L(2-50 s8)×LEN_S(10-200 s30)×ATR_L(0.1-3.1 s0.5)×ATR_S(0.5-5.5 s1.0)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A02  short_lenl — short LEN_L (1-15) with tight ATR_L (NQ/TXF daily analog)
    #      LEN_L(1-15 s1)×LEN_S(30-180 s30)×ATR_L(0.05-0.55 s0.1)×ATR_S(0.5-2.5 s0.5) = 15×6×6×5=2700
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_short_lenl"
    _c = _cfg(_n, (1, 15, 1), (30, 180, 30), (0.05, 0.55, 0.1), (0.5, 2.5, 0.5))
    log.info("A02  LEN_L(1-15 s1)×LEN_S(30-180 s30)×ATR_L(0.05-0.55 s0.1)×ATR_S(0.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A03  gc_hourly_analog — GC Hourly regime transposed to daily:
    #      short MA filter + moderate-to-high ATR_L + very high ATR_S
    #      LEN_L(1-10 s1)×LEN_S(15-60 s5)×ATR_L(0.3-1.3 s0.25)×ATR_S(3.0-8.0 s1.0) = 10×10×5×6=3000
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_gc_hourly_analog"
    _c = _cfg(_n, (1, 10, 1), (15, 60, 5), (0.3, 1.3, 0.25), (3.0, 8.0, 1.0))
    log.info("A03  LEN_L(1-10 s1)×LEN_S(15-60 s5)×ATR_L(0.3-1.3 s0.25)×ATR_S(3.0-8.0 s1.0)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A04  tight_atrl — very tight ATR_L (NQ/TXF daily pattern: ATR_L=0.05-0.22)
    #      LEN_L(2-16 s2)×LEN_S(50-150 s25)×ATR_L(0.02-0.22 s0.04)×ATR_S(0.5-2.5 s0.5) = 8×5×6×5=1200
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_tight_atrl"
    _c = _cfg(_n, (2, 16, 2), (50, 150, 25), (0.02, 0.22, 0.04), (0.5, 2.5, 0.5))
    log.info("A04  LEN_L(2-16 s2)×LEN_S(50-150 s25)×ATR_L(0.02-0.22 s0.04)×ATR_S(0.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A05  high_atrs — high ATR_S range (2-8) with moderate ATR_L
    #      LEN_L(2-20 s2)×LEN_S(20-100 s20)×ATR_L(0.1-1.6 s0.25)×ATR_S(2.0-8.0 s1.0) = 10×5×7×7=2450
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_high_atrs"
    _c = _cfg(_n, (2, 20, 2), (20, 100, 20), (0.1, 1.6, 0.25), (2.0, 8.0, 1.0))
    log.info("A05  LEN_L(2-20 s2)×LEN_S(20-100 s20)×ATR_L(0.1-1.6 s0.25)×ATR_S(2.0-8.0 s1.0)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A06  medium_ma — medium MA lengths (10-60 long, 50-300 short)
    #      LEN_L(10-60 s10)×LEN_S(50-300 s50)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-3.0 s0.5) = 6×6×5×6=1080
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_medium_ma"
    _c = _cfg(_n, (10, 60, 10), (50, 300, 50), (0.1, 1.1, 0.25), (0.5, 3.0, 0.5))
    log.info("A06  LEN_L(10-60 s10)×LEN_S(50-300 s50)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-3.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A07  short_lens — short LEN_S (5-50) for faster short-side filter
    #      LEN_L(2-20 s2)×LEN_S(5-50 s5)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-3.0 s0.5) = 10×10×5×6=3000
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_short_lens"
    _c = _cfg(_n, (2, 20, 2), (5, 50, 5), (0.1, 1.1, 0.25), (0.5, 3.0, 0.5))
    log.info("A07  LEN_L(2-20 s2)×LEN_S(5-50 s5)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-3.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A08  asym_ma — asymmetric: short LEN_L (1-10), very long LEN_S (100-400)
    #      LEN_L(1-10 s1)×LEN_S(100-400 s50)×ATR_L(0.05-0.55 s0.1)×ATR_S(0.5-2.5 s0.5) = 10×7×6×5=2100
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_asym_ma"
    _c = _cfg(_n, (1, 10, 1), (100, 400, 50), (0.05, 0.55, 0.1), (0.5, 2.5, 0.5))
    log.info("A08  LEN_L(1-10 s1)×LEN_S(100-400 s50)×ATR_L(0.05-0.55 s0.1)×ATR_S(0.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — fixed safe radii: LL±3 LS±5 s1, ATR_L±0.1 s0.05, ATR_S±0.2 s0.1
    #      Guarantees ≤ 7×11×5×5 = 1925 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   3.0, 1.0,  LL_LO, LL_HI)
        _ls   = zoom(best_ls,   5.0, 1.0,  LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.1, 0.05, AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.2, 0.1,  AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A09  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — tighter: LL±2 LS±3 s1, ATR_L±0.05 s0.025, ATR_S±0.15 s0.05
    #      ≤ 5×7×5×7 = 1225 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   2.0, 1.0,   LL_LO, LL_HI)
        _ls   = zoom(best_ls,   3.0, 1.0,   LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.05, 0.025, AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.15, 0.05,  AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A10  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — finest: LL±2 LS±3 s1, ATR_L±0.05 s0.01, ATR_S±0.1 s0.05
    #      ≤ 5×7×11×5 = 1925 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   2.0, 1.0,  LL_LO, LL_HI)
        _ls   = zoom(best_ls,   3.0, 1.0,  LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.05, 0.01, AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.1,  0.05, AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A11  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER2_NQ CME.GC HOT Daily Round-1 COMPLETE")
    log.info("  Champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g",
             best_ll, best_ls, best_atrl, best_atrs)
    log.info("  Best NP: %.0f USD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{max(0, TARGET_NP - best_np):.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LEN_L": best_ll, "LEN_S": best_ls,
            "ATR_multiplier_L": best_atrl, "ATR_multiplier_S": best_atrs,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>700K USD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="SFJ_HUNTER2_NQ CME.GC HOT Daily NP>700K USD Round-1 search")
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
