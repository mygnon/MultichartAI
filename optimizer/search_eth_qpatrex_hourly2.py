"""
search_eth_qpatrex_hourly2.py  QuantPassATRex on ETHUSDT HOT Hourly, Round 2

R1 Summary (best NP=$5,030, gap -49.7%):
  Champion: Len=25 Su=1.504 Ni=2.61 NP=$5,030 MDD=-$583 trades=73 Obj=43,354
  Zoom MONOTONIC UPWARD: A09=4,691 -> A10=4,807 -> A11=5,030 (still climbing!)
  Three regimes:
    Short-Len (Len=22-26, Su=1.5, Ni=2.5-2.9): NP=4.6-5.0K, 70-91 trades -- CHAMPION
    Mid-Len   (Len=40-55, Su=1.0-1.15, Ni=3.0-3.6): NP=4.0-4.4K, 92-144 trades
    Long-Len  (Len=160-165, Su=0.67-0.68, Ni=3.0-5.0): NP=4.45K, 82 trades, low MDD

R2 Plan (11 attempts, <=5000 combos each):
  Push the still-climbing short-Len peak with ultra-fine grid, and zoom each regime.

  A01 shortlen_ultrafine Len(20-30 s1) xSu(1.30-1.70 s0.02)  xNi(2.30-2.90 s0.03)  [11x21x21=4851]
  A02 shortlen_widerSu   Len(20-30 s1) xSu(1.70-2.50 s0.04)  xNi(2.0-3.5 s0.075)   [11x21x21=4851]
  A03 midlen_fine        Len(35-50 s1) xSu(0.85-1.30 s0.04)  xNi(2.5-3.6 s0.08)    [16x12x15=2880]
  A04 longlen_fine       Len(140-180 s2)xSu(0.50-0.90 s0.03) xNi(3.0-5.5 s0.2)     [21x14x14=4116]
  A05 vshort_fine        Len(2-22 s2)  xSu(1.0-3.0 s0.10)    xNi(1.0-4.5 s0.18)    [11x21x20=4620]
  A06 bridge_25_45       Len(25-45 s2) xSu(1.0-2.0 s0.07)    xNi(2.0-3.5 s0.10)    [11x15x16=2640]
  A07 highni_short       Len(20-30 s1) xSu(0.5-1.5 s0.07)    xNi(5.0-12.0 s0.5)    [11x15x15=2475]
  A08 global_recheck     Len(10-510 s50)xSu(0.3-3.3 s0.30)   xNi(1.0-10.0 s0.9)    [11x11x11=1331]
  A09-A11 adaptive zoom -- zoom_fixed() guarantees <=5000 combos each
"""
from __future__ import annotations
import argparse
import ctypes
import json
import logging
import math
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_QuantPassATRex_AI.wsp"
SYMBOL     = "ETHUSDT HOT"
SIGNAL     = "QuantPassATRex"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\eth_qpatrex_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 10_000.0   # USD

LEN_LO,  LEN_HI  = 1.0,   2000.0
SU_LO,   SU_HI   = 0.01,  50.0
NI_LO,   NI_HI   = 0.01,  100.0

# R1 champion as seed
SEED_LEN = 25.0
SEED_SU  = 1.504
SEED_NI  = 2.61
SEED_NP  = 5030.0

PREFIX = "ETHQP2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_eth_qpatrex_hourly2_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s -- %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zoom_fixed(center: float, radius: float, n_target: int,
               step_min: float, lo: float, hi: float) -> Tuple[float, float, float]:
    lo_val = max(lo, center - radius)
    hi_val = min(hi, center + radius)
    if lo_val >= hi_val:
        return (max(lo, center - step_min), min(hi, center + step_min), step_min)
    rng = hi_val - lo_val
    step = max(step_min, math.ceil(rng / max(1, n_target - 1) / step_min) * step_min)
    return (lo_val, hi_val, step)


def n_vals(t: Tuple[float, float, float]) -> int:
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name: str,
         length: Tuple[float, float, float],
         su:     Tuple[float, float, float],
         ni:     Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi, step_min):
        s, e, step = t
        if abs(s - e) < step_min * 0.5:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    length = _safe(length, LEN_LO, LEN_HI, 1.0)
    su     = _safe(su,     SU_LO,  SU_HI,  0.01)
    ni     = _safe(ni,     NI_LO,  NI_HI,  0.01)

    combos = n_vals(length) * n_vals(su) * n_vals(ni)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    # MC64 input order: Len, Su_Multiple, Ni_Multiple
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("Len",         *length),
            ParamAxis("Su_Multiple", *su),
            ParamAxis("Ni_Multiple", *ni),
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
        log.info("  Done %.1f min -- %s", (time.time() - t0) / 60, Path(raw_csv).name)
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


def champion(df, fb_len, fb_su, fb_ni):
    """Target-chasing mode: highest-NP row drives zoom seed."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  TARGET MET: Len=%.4g Su=%.4g Ni=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["Len"]), float(best["Su_Multiple"]), float(best["Ni_Multiple"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["Len"]), float(best["Su_Multiple"]), float(best["Ni_Multiple"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: Len=%.4g Su=%.4g Ni=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["Len"]), float(best["Su_Multiple"]), float(best["Ni_Multiple"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["Len"]), float(best["Su_Multiple"]), float(best["Ni_Multiple"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_len, fb_su, fb_ni,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, length, su, ni, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "Len": length, "Su_Multiple": su, "Ni_Multiple": ni,
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
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "ATRex=ATR(Len)+StdDev(C,Len,1); LE_Su: C>C[1]+ATRex[1]*Su -> H stop; LE_Ni: C>L[1]+ATRex[1]*Ni -> market; both long+short, reversal exits",
            "contract": "_Crypto1MUSD=Round(1000000/C,0) -- ~$1M notional per trade",
            "r1_result": "R1 best NP=$5,030 (Len=25 Su=1.504 Ni=2.61); zoom monotonic upward A09->A10->A11; gap -49.7%",
            "r2_plan":   "Ultra-fine zoom around A11 champion + mid-Len fine + long-Len fine + v-short + bridge + high-Ni + global recheck",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_eth_qpatrex_hourly2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_LEN = SEED_LEN
    best_SU  = SEED_SU
    best_NI  = SEED_NI
    best_np  = SEED_NP
    best_obj = 0.0
    target_met  = False
    attempt_log: List[dict] = []
    best_entry:  Dict = {}

    log.info("==============================================================")
    log.info("  QuantPassATRex on ETHUSDT HOT Hourly -- Round 2")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD  (contract: ~$1M notional, _Crypto1MUSD)", TARGET_NP)
    log.info("  R1 champion: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             SEED_LEN, SEED_SU, SEED_NI, SEED_NP)
    log.info("  Goal: push still-climbing short-Len peak + fine-zoom 3 regimes")
    log.info("==============================================================")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_LEN, best_SU, best_NI
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_LEN, best_SU, best_NI,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        length, su, ni, obj, np_, mdd, tr, met = champion(
            df, best_LEN, best_SU, best_NI)

        if np_ > best_np:
            best_LEN, best_SU, best_NI = length, su, ni
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, length, su, ni,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  Len=%.4g Su=%.4g Ni=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, length, su, ni, obj, np_, mdd, tr,
                 "TARGET" if met else ("NP=%.0f/10K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # -----------------------------------------------------------------------
    # A01  shortlen_ultrafine -- ultra-fine zoom around A11 champion (Len=25)
    #      R1 zoom was monotonic upward -- need step=1/0.02/0.03 to find true peak
    #      Len(20-30 s1) x Su(1.30-1.70 s0.02) x Ni(2.30-2.90 s0.03) = 11x21x21=4851
    # -----------------------------------------------------------------------
    A = 1
    _n = "01_shortlen_ultrafine"
    _c = _cfg(_n, (20, 30, 1), (1.30, 1.70, 0.02), (2.30, 2.90, 0.03))
    log.info("A01  Len(20-30 s1)xSu(1.30-1.70 s0.02)xNi(2.30-2.90 s0.03)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A02  shortlen_widerSu -- higher Su with short Len (Su=1.7-2.5 unexplored)
    #      BTC champion was Su=2.52 -- check if ETH has similar high-Su mode
    #      Len(20-30 s1) x Su(1.70-2.50 s0.04) x Ni(2.0-3.5 s0.075) = 11x21x21=4851
    # -----------------------------------------------------------------------
    A = 2
    _n = "02_shortlen_widerSu"
    _c = _cfg(_n, (20, 30, 1), (1.70, 2.50, 0.04), (2.0, 3.5, 0.075))
    log.info("A02  Len(20-30 s1)xSu(1.70-2.50 s0.04)xNi(2.0-3.5 s0.075)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A03  midlen_fine -- fine zoom of A03/A06 mid-Len high-freq regime
    #      R1 A03: Len=40 Su=1.0 Ni=3.0 NP=$4,439 (144 trades, second best)
    #      Len(35-50 s1) x Su(0.85-1.30 s0.04) x Ni(2.5-3.6 s0.08) = 16x12x15=2880
    # -----------------------------------------------------------------------
    A = 3
    _n = "03_midlen_fine"
    _c = _cfg(_n, (35, 50, 1), (0.85, 1.30, 0.04), (2.5, 3.6, 0.08))
    log.info("A03  Len(35-50 s1)xSu(0.85-1.30 s0.04)xNi(2.5-3.6 s0.08)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A04  longlen_fine -- fine zoom of A05/A07 long-Len stable regime
    #      R1 A05/A07: Len=160-165 Su=0.67-0.68 Ni=3.0-5.0 NP=$4.45K MDD=-606
    #      Len(140-180 s2) x Su(0.50-0.90 s0.03) x Ni(3.0-5.5 s0.2) = 21x14x14=4116
    # -----------------------------------------------------------------------
    A = 4
    _n = "04_longlen_fine"
    _c = _cfg(_n, (140, 180, 2), (0.50, 0.90, 0.03), (3.0, 5.5, 0.2))
    log.info("A04  Len(140-180 s2)xSu(0.50-0.90 s0.03)xNi(3.0-5.5 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A05  vshort_fine -- very short Len exploration (Len=2-22, untested at fine resolution)
    #      BTC's best Obj was at Len=13 -- check if ETH has similar v-short peak
    #      Len(2-22 s2) x Su(1.0-3.0 s0.10) x Ni(1.0-4.5 s0.18) = 11x21x20=4620
    # -----------------------------------------------------------------------
    A = 5
    _n = "05_vshort_fine"
    _c = _cfg(_n, (2, 22, 2), (1.0, 3.0, 0.10), (1.0, 4.5, 0.18))
    log.info("A05  Len(2-22 s2)xSu(1.0-3.0 s0.10)xNi(1.0-4.5 s0.18)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A06  bridge_25_45 -- bridge between short-Len and mid-Len regimes
    #      R1 missed Len=25-45 between champion (Len=25) and mid-Len (Len=40)
    #      Len(25-45 s2) x Su(1.0-2.0 s0.07) x Ni(2.0-3.5 s0.10) = 11x15x16=2640
    # -----------------------------------------------------------------------
    A = 6
    _n = "06_bridge_25_45"
    _c = _cfg(_n, (25, 45, 2), (1.0, 2.0, 0.07), (2.0, 3.5, 0.10))
    log.info("A06  Len(25-45 s2)xSu(1.0-2.0 s0.07)xNi(2.0-3.5 s0.10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A07  highni_short -- low Su + high Ni asymmetric (untested zone)
    #      Check if rare market reversals can beat A11's NP
    #      Len(20-30 s1) x Su(0.5-1.5 s0.07) x Ni(5.0-12.0 s0.5) = 11x15x15=2475
    # -----------------------------------------------------------------------
    A = 7
    _n = "07_highni_short"
    _c = _cfg(_n, (20, 30, 1), (0.5, 1.5, 0.07), (5.0, 12.0, 0.5))
    log.info("A07  Len(20-30 s1)xSu(0.5-1.5 s0.07)xNi(5.0-12.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A08  global_recheck -- broad sanity check (different cell from R1 A01)
    #      Confirm no missed regime in broader Len/Su/Ni space
    #      Len(10-510 s50) x Su(0.3-3.3 s0.30) x Ni(1.0-10.0 s0.9) = 11x11x11=1331
    # -----------------------------------------------------------------------
    A = 8
    _n = "08_global_recheck"
    _c = _cfg(_n, (10, 510, 50), (0.3, 3.3, 0.30), (1.0, 10.0, 0.9))
    log.info("A08  Len(10-510 s50)xSu(0.3-3.3 s0.30)xNi(1.0-10.0 s0.9)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A09  adaptive_zoom1 -- wide zoom on best NP found across A01-A08
    #      zoom_fixed() with n_target=15 per dim -> <=15^3=3375 combos
    # -----------------------------------------------------------------------
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             best_LEN, best_SU, best_NI, best_np)
    if start_attempt <= A:
        r_len = max(10.0,  best_LEN * 0.20)
        r_su  = max(0.20,  best_SU  * 0.30)
        r_ni  = max(0.30,  best_NI  * 0.30)
        _len = zoom_fixed(best_LEN, r_len, 15,  1.0, LEN_LO, LEN_HI)
        _su  = zoom_fixed(best_SU,  r_su,  15, 0.01, SU_LO,  SU_HI)
        _ni  = zoom_fixed(best_NI,  r_ni,  15, 0.01, NI_LO,  NI_HI)
        _c   = _cfg(_n, _len, _su, _ni)
        log.info("A09  Len%s Su%s Ni%s  %d combos", _len, _su, _ni, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A10  adaptive_zoom2 -- medium zoom on best
    # -----------------------------------------------------------------------
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             best_LEN, best_SU, best_NI, best_np)
    if start_attempt <= A:
        r_len = max(5.0,   best_LEN * 0.10)
        r_su  = max(0.10,  best_SU  * 0.13)
        r_ni  = max(0.15,  best_NI  * 0.13)
        _len = zoom_fixed(best_LEN, r_len, 11,  1.0, LEN_LO, LEN_HI)
        _su  = zoom_fixed(best_SU,  r_su,  11, 0.01, SU_LO,  SU_HI)
        _ni  = zoom_fixed(best_NI,  r_ni,  11, 0.01, NI_LO,  NI_HI)
        _c   = _cfg(_n, _len, _su, _ni)
        log.info("A10  Len%s Su%s Ni%s  %d combos", _len, _su, _ni, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A11  adaptive_zoom3 -- ultra-fine zoom on best
    # -----------------------------------------------------------------------
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             best_LEN, best_SU, best_NI, best_np)
    if start_attempt <= A:
        r_len = max(3.0,   best_LEN * 0.05)
        r_su  = max(0.04,  best_SU  * 0.06)
        r_ni  = max(0.07,  best_NI  * 0.06)
        _len = zoom_fixed(best_LEN, r_len, 9,  1.0, LEN_LO, LEN_HI)
        _su  = zoom_fixed(best_SU,  r_su,  9, 0.01, SU_LO,  SU_HI)
        _ni  = zoom_fixed(best_NI,  r_ni,  9, 0.01, NI_LO,  NI_HI)
        _c   = _cfg(_n, _len, _su, _ni)
        log.info("A11  Len%s Su%s Ni%s  %d combos", _len, _su, _ni, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    r1_np = SEED_NP
    gain  = (best_np - r1_np) / r1_np * 100 if r1_np else 0
    pct   = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  QuantPassATRex ETHUSDT Hourly Round-2 COMPLETE")
    log.info("  Champion: Len=%.4g Su=%.4g Ni=%.4g", best_LEN, best_SU, best_NI)
    log.info("  Best NP: %.0f USD  (%.1f%% of target %.0f)",
             best_np, pct, TARGET_NP)
    log.info("  R1->R2 gain: %+.2f%%  (%.0f -> %.0f)", gain, r1_np, best_np)
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "MET" if target_met else "NOT MET (gap +%.0f)" % max(0, TARGET_NP - best_np))
    log.info("==============================================================")

    if not best_entry:
        best_entry = {
            "Len": best_LEN, "Su_Multiple": best_SU, "Ni_Multiple": best_NI,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone -- results at: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    print("[auto-elevate] Requesting elevation -- approve the UAC prompt.")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Administrator manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="QuantPassATRex ETHUSDT HOT Hourly R2 parameter search")
    parser.add_argument("--from-csv",  action="store_true",
                        help="Re-analyse existing CSVs without running MC64")
    parser.add_argument("--attempt",   type=int, default=1,
                        help="Start from this attempt number (1-11)")
    parser.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

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
