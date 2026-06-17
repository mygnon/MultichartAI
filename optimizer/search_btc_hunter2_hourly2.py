"""
search_btc_hunter2_hourly2.py — SFJ_HUNTER2_crypto on BTCUSDT HOT Hourly, Round 2 (IS 2022/01-2026/01)

Strategy logic:
  input: LEN_L(250), LEN_S(250), ATR_multiplier_L(2), ATR_multiplier_S(4)
  ATR = AvgTrueRange(20)
  BUY  when C > AVERAGE(C, LEN_L) AND EntriesToday(D)=0
       → next bar STOP at Close + ATR_multiplier_L × ATR
  SHORT when C < AVERAGE(C, LEN_S) AND EntriesToday(D)=0
       → next bar STOP at Close − ATR_multiplier_S × ATR
  Exits: reversal only (no STP/LMT). Max 1 entry per day.

4 active params, target NP > 100,000 USD, <=5,000 combos/attempt. IS chart-trimmed to 2022/01-2026/01.

Attempt schedule (11 attempts):
  A01 global_wide  : LEN_L(10-250 s60)×LEN_S(10-250 s60)×ATR_L(0.5-5.0 s0.5)×ATR_S(0.5-5.0 s0.5) = 5×5×10×10=2500
  A02 long_ma      : LEN_L(100-500 s50)×LEN_S(100-500 s50)×ATR_L(1-5 s1)×ATR_S(1-5 s1) = 9×9×5×5=2025
  A03 short_ma     : LEN_L(5-95 s10)×LEN_S(5-95 s10)×ATR_L(0.5-5.5 s1)×ATR_S(0.5-5.5 s1) = 10×10×6×6=3600
  A04 vshort_ma    : LEN_L(3-30 s3)×LEN_S(3-30 s3)×ATR_L(0.5-6.5 s1)×ATR_S(0.5-6.5 s1) = 10×10×7×7=4900
  A05 asym_ma      : LEN_L(5-95 s10)×LEN_S(100-400 s50)×ATR_L(0.5-5.5 s1)×ATR_S(0.5-5.5 s1) = 10×7×6×6=2520
  A06 fine_atr     : LEN_L(10-250 s60)×LEN_S(10-250 s60)×ATR_L(0.25-3.0 s0.25)×ATR_S(0.25-3.0 s0.25) = 5×5×12×12=3600
  A07 ultra_long   : LEN_L(200-600 s50)×LEN_S(200-600 s50)×ATR_L(0.5-4.5 s1)×ATR_S(0.5-4.5 s1) = 9×9×5×5=2025
  A08 high_atr     : LEN_L(20-220 s50)×LEN_S(20-220 s50)×ATR_L(3-12 s1)×ATR_S(3-12 s1) = 5×5×10×10=2500
  A09 adaptive_zoom1: dynamic from best (moderate radius)
  A10 adaptive_zoom2: dynamic from best (tighter)
  A11 adaptive_zoom3: dynamic from best (tightest, step=1/0.1)
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_HUNTER_AI.wsp"
SYMBOL     = "BTCUSDT HOT"
SIGNAL     = "SFJ_HUNTER2_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\btc_hunter2_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LL_LO, LL_HI = 2.0, 1000.0   # LEN_L bounds
LS_LO, LS_HI = 2.0, 1000.0   # LEN_S bounds
AL_LO, AL_HI = 0.1,   30.0   # ATR_multiplier_L bounds
AS_LO, AS_HI = 0.1,   30.0   # ATR_multiplier_S bounds

# Round-2 has no prior champion — seed from strategy defaults
SEED_LL   = 250.0
SEED_LS   = 550.0
SEED_ATRL = 4.5
SEED_ATRS = 2.5
SEED_NP   = 0.0

PREFIX = "BTCH2H2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_hunter2_hourly2_{int(time.time())}.log"
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
        timeframe="hourly",
        bar_period=60,
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
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C > AVERAGE(C,LEN_L) → next bar STOP at Close+ATR_L×ATR(20)",
            "exits":    "Reversal only — no STP or LMT; max 1 entry per day",
            "params":   "LEN_L (MA long filter), LEN_S (MA short filter), ATR_multiplier_L, ATR_multiplier_S",
            "defaults": "LEN_L=250, LEN_S=250, ATR_multiplier_L=2, ATR_multiplier_S=4",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_btc_hunter2_hourly2.json"
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
    log.info("  SFJ_HUNTER2_crypto on BTCUSDT HOT Hourly — Round 2 (IS 2022/01-2026/01 chart-trimmed)")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S")
    log.info("  Entry: trend STOP breakout above/below MA; Exits: reversal only")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("══════════════════════════════════════════════════════════════")

    # Trim the chart to the IS window (this is what restricts the optimization period)
    if not from_csv and conn is not None:
        mc.ensure_chart_ready(conn, _cfg("seed", (240, 260, 10), (240, 260, 10),
                                         (1.5, 2.5, 0.5), (3.5, 4.5, 0.5)))
        log.info("Trimming chart data range to IS window %s ~ %s ...", *IS_RANGE)
        mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
        log.info("Chart trimmed (verify leftmost ~2022/01, rightmost ~2026/01).")

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
                 "★TARGET★" if met else ("%.0f/900K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  global_wide — broad sweep of entire parameter space
    #      LEN_L(10-250 s60)×LEN_S(10-250 s60)×ATR_L(0.5-5.0 s0.5)×ATR_S(0.5-5.0 s0.5) = 5×5×10×10=2500
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_high_atr_confirm"
    _c = _cfg(_n, (80, 180, 20), (180, 280, 20), (2.0, 6.0, 0.5), (8.0, 16.0, 1.0))
    log.info("A01  LEN_L(10-250 s60)×LEN_S(10-250 s60)×ATR_L(0.5-5.0 s0.5)×ATR_S(0.5-5.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A02  long_ma — larger MA periods (100-500), same as TXF R1 A02
    #      LEN_L(100-500 s50)×LEN_S(100-500 s50)×ATR_L(1-5 s1)×ATR_S(1-5 s1) = 9×9×5×5=2025
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ultralong_A07"
    _c = _cfg(_n, (200, 350, 25), (450, 650, 25), (3.5, 6.0, 0.5), (1.5, 4.0, 0.5))
    log.info("A02  LEN_L(100-500 s50)×LEN_S(100-500 s50)×ATR_L(1-5 s1)×ATR_S(1-5 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A03  short_ma — short MA periods (5-95)
    #      LEN_L(5-95 s10)×LEN_S(5-95 s10)×ATR_L(0.5-5.5 s1)×ATR_S(0.5-5.5 s1) = 10×10×6×6=3600
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_high_atrS_sweep"
    _c = _cfg(_n, (100, 140, 20), (200, 240, 20), (2.0, 6.0, 0.5), (6.0, 20.0, 1.0))
    log.info("A03  LEN_L(5-95 s10)×LEN_S(5-95 s10)×ATR_L(0.5-5.5 s1)×ATR_S(0.5-5.5 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A04  vshort_ma — very short MAs (3-30), wide ATR to catch momentum bursts
    #      LEN_L(3-30 s3)×LEN_S(3-30 s3)×ATR_L(0.5-6.5 s1)×ATR_S(0.5-6.5 s1) = 10×10×7×7=4900
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ultralong_X"
    _c = _cfg(_n, (250, 500, 50), (500, 800, 50), (2.0, 6.0, 0.5), (1.0, 4.0, 0.5))
    log.info("A04  LEN_L(3-30 s3)×LEN_S(3-30 s3)×ATR_L(0.5-6.5 s1)×ATR_S(0.5-6.5 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A05  asym_ma — asymmetric: short LEN_L, long LEN_S (TXF found LEN_L<<LEN_S)
    #      LEN_L(5-95 s10)×LEN_S(100-400 s50)×ATR_L(0.5-5.5 s1)×ATR_S(0.5-5.5 s1) = 10×7×6×6=2520
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_mid_freq"
    _c = _cfg(_n, (10, 110, 20), (50, 250, 40), (1.0, 5.0, 0.5), (2.0, 8.0, 1.0))
    log.info("A05  LEN_L(5-95 s10)×LEN_S(100-400 s50)×ATR_L(0.5-5.5 s1)×ATR_S(0.5-5.5 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A06  fine_atr — fine ATR resolution (0.25 step), moderate MA
    #      LEN_L(10-250 s60)×LEN_S(10-250 s60)×ATR_L(0.25-3.0 s0.25)×ATR_S(0.25-3.0 s0.25) = 5×5×12×12=3600
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_A08_fine"
    _c = _cfg(_n, (110, 130, 5), (210, 230, 5), (3.0, 5.0, 0.25), (9.0, 15.0, 0.5))
    log.info("A06  LEN_L(10-250 s60)×LEN_S(10-250 s60)×ATR_L(0.25-3.0 s0.25)×ATR_S(0.25-3.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A07  ultra_long — very long MA periods (200-600), check if slow-trend works
    #      LEN_L(200-600 s50)×LEN_S(200-600 s50)×ATR_L(0.5-4.5 s1)×ATR_S(0.5-4.5 s1) = 9×9×5×5=2025
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_asym_long"
    _c = _cfg(_n, (5, 50, 5), (300, 600, 50), (1.0, 4.0, 0.5), (2.0, 6.0, 0.5))
    log.info("A07  LEN_L(200-600 s50)×LEN_S(200-600 s50)×ATR_L(0.5-4.5 s1)×ATR_S(0.5-4.5 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A08  high_atr — high ATR multipliers (3-12), catching large NQ moves
    #      LEN_L(20-220 s50)×LEN_S(20-220 s50)×ATR_L(3-12 s1)×ATR_S(3-12 s1) = 5×5×10×10=2500
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_low_MDD_hunt"
    _c = _cfg(_n, (200, 400, 40), (400, 700, 50), (3.0, 6.0, 0.5), (2.0, 5.0, 0.5))
    log.info("A08  LEN_L(20-220 s50)×LEN_S(20-220 s50)×ATR_L(3-12 s1)×ATR_S(3-12 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — zoom around R1 best (moderate radius)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (30, 30, 1.5, 1.5),
            (20, 20, 1.25, 1.25),
            (15, 15, 1.0,  1.0),
            (10, 10, 0.75, 0.75),
            (8,  8,  0.5,  0.5),
            (6,  6,  0.5,  0.5),
            (5,  5,  0.375, 0.375),
        ]:
            _ll   = zoom(best_ll,   r_ll,   1.0,  LL_LO, LL_HI)
            _ls   = zoom(best_ls,   r_ls,   1.0,  LS_LO, LS_HI)
            _atrl = zoom(best_atrl, r_atrl, 0.25, AL_LO, AL_HI)
            _atrs = zoom(best_atrs, r_atrs, 0.25, AS_LO, AS_HI)
            _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
            if _c.total_runs() <= 5000:
                break
        log.info("A09  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — tighter zoom with finer step
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (15, 15, 0.75, 0.75),
            (12, 12, 0.625, 0.625),
            (10, 10, 0.5,  0.5),
            (8,  8,  0.375, 0.375),
            (6,  6,  0.25, 0.25),
        ]:
            _ll   = zoom(best_ll,   r_ll,   1.0,  LL_LO, LL_HI)
            _ls   = zoom(best_ls,   r_ls,   1.0,  LS_LO, LS_HI)
            _atrl = zoom(best_atrl, r_atrl, 0.25, AL_LO, AL_HI)
            _atrs = zoom(best_atrs, r_atrs, 0.25, AS_LO, AS_HI)
            _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — finest zoom, step=1 for LEN, step=0.1 for ATR
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (10, 10, 0.5, 0.5),
            (8,  8,  0.4, 0.4),
            (6,  6,  0.3, 0.3),
            (5,  5,  0.25, 0.25),
            (4,  4,  0.2, 0.2),
        ]:
            _ll   = zoom(best_ll,   r_ll,   1.0,  LL_LO, LL_HI)
            _ls   = zoom(best_ls,   r_ls,   1.0,  LS_LO, LS_HI)
            _atrl = zoom(best_atrl, r_atrl, 0.1,  AL_LO, AL_HI)
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
    log.info("  SFJ_HUNTER2_crypto BTCUSDT HOT Hourly Round-2 COMPLETE")
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
    print(f"Target NP>900K USD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="SFJ_HUNTER2_crypto BTCUSDT HOT Hourly NP>100K USD Round-2 search")
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
