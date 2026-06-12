"""
search_bnb_qpatrex_hourly2.py  QuantPassATRex on BNBUSDT HOT Hourly, Round 2

R1 Summary (best NP=$34,691, gap -65.3%):
  NP-max regime (A04 = A09 = A10 = A11 quad convergence):
    Len=27 Su=3.0 (BOUNDARY!) Ni=3.68 NP=$34,691 MDD=-$23,347 12 trades Obj=51,546
  Obj-max regime (3 sub-regimes -- never zoomed in R1):
    A05: Len=300 (BOUNDARY!) Su=0.97 Ni=1.25 NP=$27,438 MDD=-$7,874 51 tr Obj=95,614
    A01: Len=205 Su=0.59 Ni=2.16 NP=$26,061 MDD=-$7,571 113 tr Obj=89,709
    A02: Len=75 Su=0.86 Ni=2.05 NP=$23,412 MDD=-$6,202 129 tr Obj=88,376

R2 Plan (11 attempts, <=5000 combos each):
  Priorities: break both boundaries + zoom each Obj-max sub-regime.

  A01 npmax_pushSu      Len(22-35 s1)  xSu(2.5-5.0 s0.15)  xNi(3.0-5.5 s0.15)  [14x17x17=4046]
  A02 objmax_pushLen    Len(250-450 s10)xSu(0.7-1.3 s0.05) xNi(0.8-2.0 s0.1)   [21x13x13=3549]
  A03 a01_finezoom      Len(170-240 s5)xSu(0.4-0.9 s0.04)  xNi(1.5-3.0 s0.1)   [15x14x16=3360]
  A04 a02_finezoom      Len(55-100 s3) xSu(0.6-1.2 s0.04)  xNi(1.5-3.0 s0.1)   [16x16x16=4096]
  A05 bridge_30_75      Len(30-75 s3)  xSu(0.7-2.5 s0.12)  xNi(1.5-4.0 s0.15)  [16x16x17=4352]
  A06 bridge_100_200    Len(100-200 s5)xSu(0.4-1.5 s0.08)  xNi(1.0-3.5 s0.15)  [21x14x17=4998]
  A07 verylongLen       Len(350-800 s30)xSu(0.5-1.4 s0.06) xNi(0.5-2.0 s0.1)   [16x16x16=4096]
  A08 objmax_global     Len(50-500 s45)xSu(0.3-2.0 s0.15)  xNi(0.5-4.0 s0.3)   [11x12x12=1584]
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
SYMBOL     = "BNBUSDT HOT"
SIGNAL     = "QuantPassATRex"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_qpatrex_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO,  LEN_HI  = 1.0,   2000.0
SU_LO,   SU_HI   = 0.01,  50.0
NI_LO,   NI_HI   = 0.01,  100.0

# R1 NP-max as seed (highest NP, even though MDD is worse)
SEED_LEN = 27.0
SEED_SU  = 3.0
SEED_NI  = 3.68
SEED_NP  = 34691.0

PREFIX = "BNBQP2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_qpatrex_hourly2_{int(time.time())}.log"
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
            "r1_result": "R1 NP-max Len=27 Su=3.0(boundary!) Ni=3.68 NP=$34,691; Obj-max A05 Len=300(boundary!) Su=0.97 Ni=1.25 Obj=95,614",
            "r2_plan":  "Push Su>3 + Len>300 boundaries + zoom 3 obj-max sub-regimes + bridges",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_qpatrex_hourly2.json"
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
    log.info("  QuantPassATRex on BNBUSDT HOT Hourly -- Round 2")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD  (contract: ~$1M notional, _Crypto1MUSD)", TARGET_NP)
    log.info("  R1 NP-max: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             SEED_LEN, SEED_SU, SEED_NI, SEED_NP)
    log.info("  R1 Obj-max A05: Len=300 Su=0.97 Ni=1.25 NP=$27,438 Obj=95,614")
    log.info("  Goal: break Su>3 + Len>300 boundaries + zoom obj-max sub-regimes")
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
                 "TARGET" if met else ("NP=%.0f/100K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # -----------------------------------------------------------------------
    # A01  npmax_pushSu -- break Su=3.0 boundary
    #      R1 A04 had Su=3.0 at boundary; explore Su 2.5-5.0
    #      Len(22-35 s1) x Su(2.5-5.0 s0.15) x Ni(3.0-5.5 s0.15) = 14x17x17=4046
    # -----------------------------------------------------------------------
    A = 1
    _n = "01_npmax_pushSu"
    _c = _cfg(_n, (22, 35, 1), (2.5, 5.0, 0.15), (3.0, 5.5, 0.15))
    log.info("A01  Len(22-35 s1)xSu(2.5-5.0 s0.15)xNi(3.0-5.5 s0.15)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A02  objmax_pushLen -- break Len=300 boundary + zoom A05
    #      R1 A05 had Len=300 at boundary; explore Len 250-450
    #      Len(250-450 s10) x Su(0.7-1.3 s0.05) x Ni(0.8-2.0 s0.1) = 21x13x13=3549
    # -----------------------------------------------------------------------
    A = 2
    _n = "02_objmax_pushLen"
    _c = _cfg(_n, (250, 450, 10), (0.7, 1.3, 0.05), (0.8, 2.0, 0.1))
    log.info("A02  Len(250-450 s10)xSu(0.7-1.3 s0.05)xNi(0.8-2.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A03  a01_finezoom -- zoom R1 A01 Obj-max sub-regime (Len=205)
    #      R1 A01: Len=205 Su=0.59 Ni=2.16 NP=$26,061 Obj=89,709
    #      Len(170-240 s5) x Su(0.4-0.9 s0.04) x Ni(1.5-3.0 s0.1) = 15x14x16=3360
    # -----------------------------------------------------------------------
    A = 3
    _n = "03_a01_finezoom"
    _c = _cfg(_n, (170, 240, 5), (0.4, 0.9, 0.04), (1.5, 3.0, 0.1))
    log.info("A03  Len(170-240 s5)xSu(0.4-0.9 s0.04)xNi(1.5-3.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A04  a02_finezoom -- zoom R1 A02 Obj-max sub-regime (Len=75)
    #      R1 A02: Len=75 Su=0.86 Ni=2.05 NP=$23,412 Obj=88,376 129 trades (highest freq)
    #      Len(55-100 s3) x Su(0.6-1.2 s0.04) x Ni(1.5-3.0 s0.1) = 16x16x16=4096
    # -----------------------------------------------------------------------
    A = 4
    _n = "04_a02_finezoom"
    _c = _cfg(_n, (55, 100, 3), (0.6, 1.2, 0.04), (1.5, 3.0, 0.1))
    log.info("A04  Len(55-100 s3)xSu(0.6-1.2 s0.04)xNi(1.5-3.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A05  bridge_30_75 -- bridge between NP-max (Len=27) and obj-max A02 (Len=75)
    #      Unexplored zone in R1
    #      Len(30-75 s3) x Su(0.7-2.5 s0.12) x Ni(1.5-4.0 s0.15) = 16x16x17=4352
    # -----------------------------------------------------------------------
    A = 5
    _n = "05_bridge_30_75"
    _c = _cfg(_n, (30, 75, 3), (0.7, 2.5, 0.12), (1.5, 4.0, 0.15))
    log.info("A05  Len(30-75 s3)xSu(0.7-2.5 s0.12)xNi(1.5-4.0 s0.15)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A06  bridge_100_200 -- bridge between A02 (Len=75) and A01 (Len=205)
    #      Unexplored zone in R1
    #      Len(100-200 s5) x Su(0.4-1.5 s0.08) x Ni(1.0-3.5 s0.15) = 21x14x17=4998
    # -----------------------------------------------------------------------
    A = 6
    _n = "06_bridge_100_200"
    _c = _cfg(_n, (100, 200, 5), (0.4, 1.5, 0.08), (1.0, 3.5, 0.15))
    log.info("A06  Len(100-200 s5)xSu(0.4-1.5 s0.08)xNi(1.0-3.5 s0.15)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A07  verylongLen -- explore Len 350-800 (beyond R1 boundary)
    #      Test if obj-max regime extends further
    #      Len(350-800 s30) x Su(0.5-1.4 s0.06) x Ni(0.5-2.0 s0.1) = 16x16x16=4096
    # -----------------------------------------------------------------------
    A = 7
    _n = "07_verylongLen"
    _c = _cfg(_n, (350, 800, 30), (0.5, 1.4, 0.06), (0.5, 2.0, 0.1))
    log.info("A07  Len(350-800 s30)xSu(0.5-1.4 s0.06)xNi(0.5-2.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A08  objmax_global -- broad Obj-max landscape recheck
    #      Confirm no missed obj-max sub-regime in broader space
    #      Len(50-500 s45) x Su(0.3-2.0 s0.15) x Ni(0.5-4.0 s0.3) = 11x12x12=1584
    # -----------------------------------------------------------------------
    A = 8
    _n = "08_objmax_global"
    _c = _cfg(_n, (50, 500, 45), (0.3, 2.0, 0.15), (0.5, 4.0, 0.3))
    log.info("A08  Len(50-500 s45)xSu(0.3-2.0 s0.15)xNi(0.5-4.0 s0.3)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A09  adaptive_zoom1 -- wide zoom on best NP found across A01-A08
    # -----------------------------------------------------------------------
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             best_LEN, best_SU, best_NI, best_np)
    if start_attempt <= A:
        r_len = max(15.0,  best_LEN * 0.25)
        r_su  = max(0.25,  best_SU  * 0.35)
        r_ni  = max(0.35,  best_NI  * 0.35)
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
        r_len = max(8.0,   best_LEN * 0.12)
        r_su  = max(0.10,  best_SU  * 0.15)
        r_ni  = max(0.15,  best_NI  * 0.15)
        _len = zoom_fixed(best_LEN, r_len, 11,  1.0, LEN_LO, LEN_HI)
        _su  = zoom_fixed(best_SU,  r_su,  11, 0.01, SU_LO,  SU_HI)
        _ni  = zoom_fixed(best_NI,  r_ni,  11, 0.01, NI_LO,  NI_HI)
        _c   = _cfg(_n, _len, _su, _ni)
        log.info("A10  Len%s Su%s Ni%s  %d combos", _len, _su, _ni, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A11  adaptive_zoom3 -- fine zoom on best
    # -----------------------------------------------------------------------
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             best_LEN, best_SU, best_NI, best_np)
    if start_attempt <= A:
        r_len = max(4.0,   best_LEN * 0.06)
        r_su  = max(0.05,  best_SU  * 0.08)
        r_ni  = max(0.07,  best_NI  * 0.08)
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
    log.info("  QuantPassATRex BNBUSDT Hourly Round-2 COMPLETE")
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
        description="QuantPassATRex BNBUSDT HOT Hourly R2 parameter search")
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
