"""
search_txf_hunter2_daily2.py — SFJ_HUNTER2_NQ on TWF.TXF HOT Daily, Round 2

Strategy:
  BUY  when C > AVERAGE(C, LEN_L) AND EntriesToday(D)=0 → next bar STOP at Close + ATR_L×ATR(20)
  SHORT when C < AVERAGE(C, LEN_S) AND EntriesToday(D)=0 → next bar STOP at Close − ATR_S×ATR(20)
  Reversal exits only. 4 active params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S.

R1 champion: LEN_L=15, LEN_S=65, ATR_L=0.17, ATR_S=1.1, NP=5,393,400 TWD, 79 trades/7yr

R2 strategy: Fine-tune productive zone (LEN_S=20-150); keep LEN_S ≤ 150 to avoid MC64 timeout.
  A01 fine_around_r1  : LEN_L(10-20 s2)×LEN_S(50-80 s5)×ATR_L(0.07-0.28 s0.07)×ATR_S(0.7-1.5 s0.2) = 6×7×4×5=840
  A02 fine_atrl       : LEN_L(13-17 s2)×LEN_S(60-70 s2)×ATR_L(0.10-0.25 s0.01)×ATR_S(0.9-1.3 s0.2) = 3×6×16×3=864
  A03 lower_lens      : LEN_L(10-25 s5)×LEN_S(20-60 s5)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-1.5 s0.5) = 4×9×7×3=756
  A04 higher_lens     : LEN_L(10-25 s5)×LEN_S(70-140 s10)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-2.0 s0.5) = 4×8×7×4=896
  A05 short_lenl      : LEN_L(2-14 s2)×LEN_S(40-90 s10)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-1.5 s0.5) = 7×6×7×3=882
  A06 high_atrl       : LEN_L(10-25 s5)×LEN_S(55-75 s5)×ATR_L(0.3-1.2 s0.1)×ATR_S(0.5-2.0 s0.5) = 4×5×10×4=800
  A07 lenl_scan       : LEN_L(1-50 s1)×LEN_S(63-67 s2)×ATR_L(0.15-0.20 s0.01)×ATR_S(0.9-1.3 s0.2) = 50×3×6×3=2700
  A08 global_r2       : LEN_L(5-35 s5)×LEN_S(10-110 s20)×ATR_L(0.1-0.5 s0.1)×ATR_S(0.5-2.0 s0.5) = 7×6×5×4=840
  A09 adaptive_zoom1  : dynamic (fine step ATR_L=0.025)
  A10 adaptive_zoom2  : dynamic (tighter)
  A11 adaptive_zoom3  : dynamic (finest ATR_L step=0.01)

NOTE: All attempts use LEN_S ≤ 150 to avoid the ~2-hour timeout caused by large MA warm-up
periods in MC64 daily backtests. This was discovered in R1 where A05 (LEN_S=200-400, 4200
combos) and A06 (LEN_S=50-350, 1050 combos) both hit the 2-hour limit.
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_hunter2_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 9_000_000.0

LL_LO, LL_HI   = 1.0,   1000.0
LS_LO, LS_HI   = 1.0,   150.0   # Hard cap at 150 — avoids MC64 timeout on daily bars
AL_LO, AL_HI   = 0.05,  30.0
AS_LO, AS_HI   = 0.1,   30.0

# R1 champion seed
SEED_LL   = 15.0
SEED_LS   = 65.0
SEED_ATRL = 0.17
SEED_ATRS = 1.1
SEED_NP   = 5_393_400.0

PREFIX = "TXFH2D2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_hunter2_daily2_{int(time.time())}.log"
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
        "round": 2,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY when C > AVERAGE(C,LEN_L) → next bar STOP at Close+ATR_L×ATR(20)",
            "exits":       "Reversal only — no STP or LMT; max 1 entry per day",
            "r1_champion": "LEN_L=15 LEN_S=65 ATR_L=0.17 ATR_S=1.1 NP=5393400 MDD=-994800 trades=79",
            "r2_focus":    "Fine-tune LEN_S=20-150 zone; LEN_S>150 avoided (MC64 daily timeout)",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_hunter2_daily2.json"
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
    log.info("  SFJ_HUNTER2_NQ on TWF.TXF HOT Daily — Round 2")
    log.info("  R1 champion: LEN_L=15 LEN_S=65 ATR_L=0.17 ATR_S=1.1  NP=5,393,400")
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
    # A01  fine_around_r1 — tighten around R1 champion with finer steps
    #      LEN_L(10-20 s2)×LEN_S(50-80 s5)×ATR_L(0.07-0.28 s0.07)×ATR_S(0.7-1.5 s0.2) = 6×7×4×5=840
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_around_r1"
    _c = _cfg(_n, (10, 20, 2), (50, 80, 5), (0.07, 0.28, 0.07), (0.7, 1.5, 0.2))
    log.info("A01  LEN_L(10-20 s2)×LEN_S(50-80 s5)×ATR_L(0.07-0.28 s0.07)×ATR_S(0.7-1.5 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A02  fine_atrl — ultra-fine ATR_L grid around 0.17 (step=0.01)
    #      LEN_L(13-17 s2)×LEN_S(60-70 s2)×ATR_L(0.10-0.25 s0.01)×ATR_S(0.9-1.3 s0.2) = 3×6×16×3=864
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_fine_atrl"
    _c = _cfg(_n, (13, 17, 2), (60, 70, 2), (0.10, 0.25, 0.01), (0.9, 1.3, 0.2))
    log.info("A02  LEN_L(13-17 s2)×LEN_S(60-70 s2)×ATR_L(0.10-0.25 s0.01)×ATR_S(0.9-1.3 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A03  lower_lens — explore shorter LEN_S (20-60), faster entry flips
    #      LEN_L(10-25 s5)×LEN_S(20-60 s5)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-1.5 s0.5) = 4×9×7×3=756
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_lower_lens"
    _c = _cfg(_n, (10, 25, 5), (20, 60, 5), (0.05, 0.35, 0.05), (0.5, 1.5, 0.5))
    log.info("A03  LEN_L(10-25 s5)×LEN_S(20-60 s5)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-1.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A04  higher_lens — explore longer LEN_S (70-140, safe range below timeout)
    #      LEN_L(10-25 s5)×LEN_S(70-140 s10)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-2.0 s0.5) = 4×8×7×4=896
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_higher_lens"
    _c = _cfg(_n, (10, 25, 5), (70, 140, 10), (0.05, 0.35, 0.05), (0.5, 2.0, 0.5))
    log.info("A04  LEN_L(10-25 s5)×LEN_S(70-140 s10)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-2.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A05  short_lenl — shorter LEN_L for more frequent entries
    #      LEN_L(2-14 s2)×LEN_S(40-90 s10)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-1.5 s0.5) = 7×6×7×3=882
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_short_lenl"
    _c = _cfg(_n, (2, 14, 2), (40, 90, 10), (0.05, 0.35, 0.05), (0.5, 1.5, 0.5))
    log.info("A05  LEN_L(2-14 s2)×LEN_S(40-90 s10)×ATR_L(0.05-0.35 s0.05)×ATR_S(0.5-1.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A06  high_atrl — explore larger ATR_L (0.3-1.2), different entry behaviour
    #      LEN_L(10-25 s5)×LEN_S(55-75 s5)×ATR_L(0.3-1.2 s0.1)×ATR_S(0.5-2.0 s0.5) = 4×5×10×4=800
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_atrl"
    _c = _cfg(_n, (10, 25, 5), (55, 75, 5), (0.3, 1.2, 0.1), (0.5, 2.0, 0.5))
    log.info("A06  LEN_L(10-25 s5)×LEN_S(55-75 s5)×ATR_L(0.3-1.2 s0.1)×ATR_S(0.5-2.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A07  lenl_scan — dense LEN_L sweep (step=1) with fixed optimal LEN_S=65
    #      LEN_L(1-50 s1)×LEN_S(63-67 s2)×ATR_L(0.15-0.20 s0.01)×ATR_S(0.9-1.3 s0.2) = 50×3×6×3=2700
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_lenl_scan"
    _c = _cfg(_n, (1, 50, 1), (63, 67, 2), (0.15, 0.20, 0.01), (0.9, 1.3, 0.2))
    log.info("A07  LEN_L(1-50 s1)×LEN_S(63-67 s2)×ATR_L(0.15-0.20 s0.01)×ATR_S(0.9-1.3 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A08  global_r2 — medium-density landscape overview (verify R1 map)
    #      LEN_L(5-35 s5)×LEN_S(10-110 s20)×ATR_L(0.1-0.5 s0.1)×ATR_S(0.5-2.0 s0.5) = 7×6×5×4=840
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_global_r2"
    _c = _cfg(_n, (5, 35, 5), (10, 110, 20), (0.1, 0.5, 0.1), (0.5, 2.0, 0.5))
    log.info("A08  LEN_L(5-35 s5)×LEN_S(10-110 s20)×ATR_L(0.1-0.5 s0.1)×ATR_S(0.5-2.0 s0.5)  %d combos",
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
            (6, 20, 0.15, 0.4),
            (5, 15, 0.12, 0.35),
            (4, 12, 0.1,  0.3),
            (3, 10, 0.075, 0.25),
            (2,  8, 0.05,  0.2),
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
            (4, 12, 0.08, 0.25),
            (3, 10, 0.06, 0.2),
            (2,  8, 0.05, 0.15),
            (2,  5, 0.04, 0.12),
            (1,  5, 0.04, 0.12),
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
            (4, 12, 0.12, 0.25),
            (3, 10, 0.09, 0.2),
            (2,  8, 0.07, 0.15),
            (2,  5, 0.05, 0.12),
            (1,  5, 0.04, 0.1),
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
    log.info("  SFJ_HUNTER2_NQ TXF HOT Daily Round-2 COMPLETE")
    log.info("  Champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g",
             best_ll, best_ls, best_atrl, best_atrs)
    log.info("  Best NP: %.0f  Target %.0f TWD: %s",
             best_np, TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{max(0, TARGET_NP - best_np):.0f})")
    log.info("  R1→R2 gain: %.1f%%", 100 * (best_np - SEED_NP) / SEED_NP)
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
    print(f"R1→R2 gain: {100 * (best_np - SEED_NP) / SEED_NP:.1f}%")
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
        description="SFJ_HUNTER2_NQ TWF.TXF HOT Daily NP>9M TWD Round-2 search")
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
