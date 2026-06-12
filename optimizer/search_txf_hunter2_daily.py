"""
search_txf_hunter2_daily.py — SFJ_HUNTER2_NQ on TWF.TXF HOT Daily, Round 1

Strategy:
  BUY  when C > AVERAGE(C, LEN_L) AND EntriesToday(D)=0 → next bar STOP at Close + ATR_L×ATR(20)
  SHORT when C < AVERAGE(C, LEN_S) AND EntriesToday(D)=0 → next bar STOP at Close − ATR_S×ATR(20)
  Reversal exits only. 4 active params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S.
  Defaults: LEN_L=250, LEN_S=250, ATR_multiplier_L=2, ATR_multiplier_S=4.

Workspace: 20260101_SFJ_HUNTER_AI.wsp (TWF.TXF HOT Daily chart with SFJ_HUNTER2_NQ applied)
Target: NP > 9,000,000 TWD, maximise NP²/|MDD|

R1 strategy: 8 wide+focused fixed attempts + 3 adaptive zooms to map the daily landscape.
  A01 global_wide    : LEN_L(2-50 s8)×LEN_S(10-500 s70)×ATR_L(0.1-3.1 s0.5)×ATR_S(0.5-4.5 s1.0) = 7×8×7×5=1960
  A02 short_lenl     : LEN_L(1-15 s1)×LEN_S(50-350 s50)×ATR_L(0.1-1.6 s0.25)×ATR_S(0.5-2.5 s0.5) = 15×7×7×5=3675
  A03 medium_ma      : LEN_L(10-60 s5)×LEN_S(50-350 s50)×ATR_L(0.1-1.6 s0.25)×ATR_S(1.0-3.0 s0.5) = 11×7×7×5=2695
  A04 long_ma        : LEN_L(20-150 s10)×LEN_S(100-500 s50)×ATR_L(0.5-2.5 s0.5)×ATR_S(1.0-3.0 s0.5) = 14×9×5×5=3150
  A05 fine_atrl      : LEN_L(2-20 s2)×LEN_S(200-400 s50)×ATR_L(0.05-1.05 s0.05)×ATR_S(1.0-2.5 s0.5) = 10×5×21×4=4200
  A06 high_atr       : LEN_L(5-30 s5)×LEN_S(50-350 s75)×ATR_L(2.0-8.0 s1.0)×ATR_S(2.0-6.0 s1.0) = 6×5×7×5=1050
  A07 ultra_short_lens: LEN_L(2-20 s2)×LEN_S(5-100 s10)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-2.5 s0.5) = 10×11×5×5=2750
  A08 asym_ma        : LEN_L(1-10 s1)×LEN_S(100-400 s50)×ATR_L(0.1-1.1 s0.25)×ATR_S(1.0-4.0 s0.5) = 10×7×5×7=2450
  A09 adaptive_zoom1 : dynamic (fine step ATR_L=0.025)
  A10 adaptive_zoom2 : dynamic (tighter)
  A11 adaptive_zoom3 : dynamic (finest ATR_L step=0.01)
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
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "SFJ_HUNTER2_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_hunter2_daily_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 9_000_000.0

LL_LO, LL_HI   = 1.0,   1000.0
LS_LO, LS_HI   = 1.0,   1000.0
AL_LO, AL_HI   = 0.05,  30.0
AS_LO, AS_HI   = 0.1,   30.0

# Start from strategy defaults (no prior daily search)
SEED_LL   = 250.0
SEED_LS   = 250.0
SEED_ATRL = 2.0
SEED_ATRS = 4.0
SEED_NP   = 0.0

PREFIX = "TXFH2D_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_hunter2_daily_{int(time.time())}.log"
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
            "logic":    "BUY when C > AVERAGE(C,LEN_L) → next bar STOP at Close+ATR_L×ATR(20)",
            "exits":    "Reversal only — no STP or LMT; max 1 entry per day",
            "defaults": "LEN_L=250, LEN_S=250, ATR_multiplier_L=2, ATR_multiplier_S=4",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_hunter2_daily.json"
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
    log.info("  SFJ_HUNTER2_NQ on TWF.TXF HOT Daily — Round 1")
    log.info("  Defaults: LEN_L=250 LEN_S=250 ATR_L=2 ATR_S=4")
    log.info("  Target: %.0f TWD  (NP > 9M)", TARGET_NP)
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
                 "★TARGET★" if met else ("%.0f/9M" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  global_wide — map the full parameter space on daily bars
    #      LEN_L(2-50 s8)×LEN_S(10-500 s70)×ATR_L(0.1-3.1 s0.5)×ATR_S(0.5-4.5 s1.0) = 7×8×7×5=1960
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_global_wide"
    _c = _cfg(_n, (2, 50, 8), (10, 500, 70), (0.1, 3.1, 0.5), (0.5, 4.5, 1.0))
    log.info("A01  LEN_L(2-50 s8)×LEN_S(10-500 s70)×ATR_L(0.1-3.1 s0.5)×ATR_S(0.5-4.5 s1.0)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A02  short_lenl — short MA for longs (like hourly regime-1)
    #      LEN_L(1-15 s1)×LEN_S(50-350 s50)×ATR_L(0.1-1.6 s0.25)×ATR_S(0.5-2.5 s0.5) = 15×7×7×5=3675
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_short_lenl"
    _c = _cfg(_n, (1, 15, 1), (50, 350, 50), (0.1, 1.6, 0.25), (0.5, 2.5, 0.5))
    log.info("A02  LEN_L(1-15 s1)×LEN_S(50-350 s50)×ATR_L(0.1-1.6 s0.25)×ATR_S(0.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A03  medium_ma — medium MA lengths (10-60 bars)
    #      LEN_L(10-60 s5)×LEN_S(50-350 s50)×ATR_L(0.1-1.6 s0.25)×ATR_S(1.0-3.0 s0.5) = 11×7×7×5=2695
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_medium_ma"
    _c = _cfg(_n, (10, 60, 5), (50, 350, 50), (0.1, 1.6, 0.25), (1.0, 3.0, 0.5))
    log.info("A03  LEN_L(10-60 s5)×LEN_S(50-350 s50)×ATR_L(0.1-1.6 s0.25)×ATR_S(1.0-3.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A04  long_ma — long MA lengths (20-150 bars = 1 month to 7 months)
    #      LEN_L(20-150 s10)×LEN_S(100-500 s50)×ATR_L(0.5-2.5 s0.5)×ATR_S(1.0-3.0 s0.5) = 14×9×5×5=3150
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_long_ma"
    _c = _cfg(_n, (20, 150, 10), (100, 500, 50), (0.5, 2.5, 0.5), (1.0, 3.0, 0.5))
    log.info("A04  LEN_L(20-150 s10)×LEN_S(100-500 s50)×ATR_L(0.5-2.5 s0.5)×ATR_S(1.0-3.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A05  fine_atrl — fine ATR_L sweep in tight-ATR regime
    #      LEN_L(2-20 s2)×LEN_S(200-400 s50)×ATR_L(0.05-1.05 s0.05)×ATR_S(1.0-2.5 s0.5) = 10×5×21×4=4200
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_fine_atrl"
    _c = _cfg(_n, (2, 20, 2), (200, 400, 50), (0.05, 1.05, 0.05), (1.0, 2.5, 0.5))
    log.info("A05  LEN_L(2-20 s2)×LEN_S(200-400 s50)×ATR_L(0.05-1.05 s0.05)×ATR_S(1.0-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A06  high_atr — high ATR regime (large multipliers like strategy defaults)
    #      LEN_L(5-30 s5)×LEN_S(50-350 s75)×ATR_L(2.0-8.0 s1.0)×ATR_S(2.0-6.0 s1.0) = 6×5×7×5=1050
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_atr"
    _c = _cfg(_n, (5, 30, 5), (50, 350, 75), (2.0, 8.0, 1.0), (2.0, 6.0, 1.0))
    log.info("A06  LEN_L(5-30 s5)×LEN_S(50-350 s75)×ATR_L(2.0-8.0 s1.0)×ATR_S(2.0-6.0 s1.0)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A07  ultra_short_lens — both MA very short (quick signal flip)
    #      LEN_L(2-20 s2)×LEN_S(5-100 s10)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-2.5 s0.5) = 10×11×5×5=2750
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ultra_short_lens"
    _c = _cfg(_n, (2, 20, 2), (5, 100, 10), (0.1, 1.1, 0.25), (0.5, 2.5, 0.5))
    log.info("A07  LEN_L(2-20 s2)×LEN_S(5-100 s10)×ATR_L(0.1-1.1 s0.25)×ATR_S(0.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A08  asym_ma — asymmetric: very short long-MA, long short-MA
    #      LEN_L(1-10 s1)×LEN_S(100-400 s50)×ATR_L(0.1-1.1 s0.25)×ATR_S(1.0-4.0 s0.5) = 10×7×5×7=2450
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_asym_ma"
    _c = _cfg(_n, (1, 10, 1), (100, 400, 50), (0.1, 1.1, 0.25), (1.0, 4.0, 0.5))
    log.info("A08  LEN_L(1-10 s1)×LEN_S(100-400 s50)×ATR_L(0.1-1.1 s0.25)×ATR_S(1.0-4.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — fine step: ATR_L=0.025, ATR_S=0.1
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (6, 30, 0.2,  0.5),
            (5, 25, 0.15, 0.4),
            (4, 20, 0.125, 0.4),
            (3, 15, 0.1,  0.3),
            (2, 10, 0.075, 0.25),
        ]:
            _ll   = zoom(best_ll,   r_ll,   1.0,   LL_LO, LL_HI)
            _ls   = zoom(best_ls,   r_ls,   5.0,   LS_LO, LS_HI)
            _atrl = zoom(best_atrl, r_atrl, 0.025, AL_LO, AL_HI)
            _atrs = zoom(best_atrs, r_atrs, 0.1,   AS_LO, AS_HI)
            _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
            if _c.total_runs() <= 5000:
                break
        log.info("A09  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — tighter zoom
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (4, 20, 0.1,  0.3),
            (3, 15, 0.075, 0.25),
            (2, 10, 0.05,  0.2),
            (2,  5, 0.05,  0.15),
            (1,  5, 0.05,  0.15),
        ]:
            _ll   = zoom(best_ll,   r_ll,   1.0,   LL_LO, LL_HI)
            _ls   = zoom(best_ls,   r_ls,   5.0,   LS_LO, LS_HI)
            _atrl = zoom(best_atrl, r_atrl, 0.025, AL_LO, AL_HI)
            _atrs = zoom(best_atrs, r_atrs, 0.1,   AS_LO, AS_HI)
            _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — finest zoom, ATR_L step=0.01
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (4, 20, 0.15, 0.3),
            (3, 15, 0.1,  0.25),
            (2, 10, 0.08, 0.2),
            (2,  5, 0.06, 0.15),
            (1,  5, 0.05, 0.1),
        ]:
            _ll   = zoom(best_ll,   r_ll,   1.0,  LL_LO, LL_HI)
            _ls   = zoom(best_ls,   r_ls,   5.0,  LS_LO, LS_HI)
            _atrl = zoom(best_atrl, r_atrl, 0.01, AL_LO, AL_HI)
            _atrs = zoom(best_atrs, r_atrs, 0.1,  AS_LO, AS_HI)
            _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER2_NQ TXF HOT Daily Round-1 COMPLETE")
    log.info("  Champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g",
             best_ll, best_ls, best_atrl, best_atrs)
    log.info("  Best NP: %.0f  Target %.0f TWD: %s",
             best_np, TARGET_NP,
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
    print(f"Target NP>9M TWD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="SFJ_HUNTER2_NQ TWF.TXF HOT Daily NP>9M TWD Round-1 search")
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
