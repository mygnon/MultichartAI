"""
search_bnb_qpatrex_hourly5.py  QuantPassATRex on BNBUSDT HOT Hourly, Round 5

R4 BREAKTHROUGH (+8.04% gain):
  A09 NP champion: Len=235 Su=0.58575 Ni=2.1275 NP=$39,921 MDD=-$6,337 95tr Obj=251,498
  A02 Obj/MDD king: Len=98 Su=0.72 Ni=1.625 NP=$31,393 MDD=-$3,933 (lowest ever) 182tr Obj=250,571
  NP-max regime 4th confirm at $35,560

R4 key insight: Su step=0.005 unlocked +8% NP jump (R3 used step=0.01).
R5 strategy: push step precision further on remaining axes (Su 0.0025, Ni 0.0025/0.005)
              + explore A09 region with lower Ni + bridge sub-peaks + mid-Su zone.

R5 Plan (11 attempts, <=5000 combos each):
  A01 A09_su_finer    Len(230-245 s1)xSu(0.555-0.620 s0.005) xNi(2.08-2.18 s0.005)   [16x14x21=4704]
  A02 A09_ni_finer    Len(232-238 s1)xSu(0.575-0.605 s0.0025)xNi(2.10-2.16 s0.0025)  [7x13x25=2275]
  A03 A02_lenwide     Len(91-105 s1) xSu(0.69-0.76 s0.005)   xNi(1.55-1.72 s0.01)    [15x15x18=4050]
  A04 A02_sufiner     Len(88-104 s1) xSu(0.700-0.745 s0.005) xNi(1.58-1.72 s0.01)    [17x10x15=2550]
  A05 longLen_zoom    Len(450-480 s3)xSu(0.45-0.55 s0.01)    xNi(1.95-2.10 s0.01)    [11x11x16=1936]
  A06 bridge_subhunt  Len(110-220 s10)xSu(0.55-0.78 s0.025)  xNi(1.70-2.15 s0.045)   [12x10x11=1320]
  A07 A09_lowerNi     Len(232-242 s1)xSu(0.575-0.600 s0.005) xNi(1.95-2.10 s0.0075)  [11x6x21=1386]
  A08 midSu_bridge    Len(90-240 s10)xSu(0.62-0.71 s0.01)    xNi(1.65-2.10 s0.045)   [16x10x11=1760]
  A09-A11 adaptive zoom -- tighter than R4
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_qpatrex_hourly5_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO,  LEN_HI  = 1.0,   2000.0
SU_LO,   SU_HI   = 0.01,  50.0
NI_LO,   NI_HI   = 0.01,  100.0

# R4 NP champion as seed
SEED_LEN = 235.0
SEED_SU  = 0.58575
SEED_NI  = 2.1275
SEED_NP  = 39921.0

PREFIX = "BNBQP5_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_qpatrex_hourly5_{int(time.time())}.log"
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
        "round": 5,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "ATRex=ATR(Len)+StdDev(C,Len,1); LE_Su: C>C[1]+ATRex[1]*Su -> H stop; LE_Ni: C>L[1]+ATRex[1]*Ni -> market; both long+short, reversal exits",
            "contract": "_Crypto1MUSD=Round(1000000/C,0) -- ~$1M notional per trade",
            "r1_result": "R1 NP-max Len=27 Su=3.0 Ni=3.68 NP=$34,691",
            "r2_result": "R2 NP-max Len=37 Su=2.43 Ni=3.63 NP=$35,856 (+3.4%)",
            "r3_result": "R3 NP champion REGIME-SHIFT: A11 Len=237.5 Su=0.58 Ni=2.085 NP=$36,951 (+3.05%)",
            "r4_result": "R4 BREAKTHROUGH +8.04%: A09 Len=235 Su=0.58575 Ni=2.1275 NP=$39,921 Obj=251,498",
            "r5_plan":   "Push axis precision further (Su 0.0025, Ni 0.0025/0.005) + bridge sub-peaks + mid-Su exploration",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_qpatrex_hourly5.json"
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
    log.info("  QuantPassATRex on BNBUSDT HOT Hourly -- Round 5")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD  (contract: ~$1M notional, _Crypto1MUSD)", TARGET_NP)
    log.info("  R4 NP champion: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             SEED_LEN, SEED_SU, SEED_NI, SEED_NP)
    log.info("  R4 Obj champion A02: Len=98 Su=0.72 Ni=1.625 Obj=250,571 MDD=-$3,933")
    log.info("  Goal: push axis precision (Su/Ni step ~0.0025) + bridge sub-peaks")
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
    # A01  A09_su_finer -- finer Su around R4 A09 with broader range
    #      Len(230-245 s1) x Su(0.555-0.620 s0.005) x Ni(2.08-2.18 s0.005) = 16x14x21=4704
    # -----------------------------------------------------------------------
    A = 1
    _n = "01_A09_su_finer"
    _c = _cfg(_n, (230, 245, 1), (0.555, 0.620, 0.005), (2.08, 2.18, 0.005))
    log.info("A01  Len(230-245 s1)xSu(0.555-0.620 s0.005)xNi(2.08-2.18 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A02  A09_ni_finer -- ultra-fine Ni around A09 (step=0.0025)
    #      Len(232-238 s1) x Su(0.575-0.605 s0.0025) x Ni(2.10-2.16 s0.0025) = 7x13x25=2275
    # -----------------------------------------------------------------------
    A = 2
    _n = "02_A09_ni_finer"
    _c = _cfg(_n, (232, 238, 1), (0.575, 0.605, 0.0025), (2.10, 2.16, 0.0025))
    log.info("A02  Len(232-238 s1)xSu(0.575-0.605 s0.0025)xNi(2.10-2.16 s0.0025)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A03  A02_lenwide -- wide Len for A02 region (Len=98 zone) with finer Su
    #      R4 A02: Len=98 Su=0.72 Ni=1.625 Obj=250,571
    #      Len(91-105 s1) x Su(0.69-0.76 s0.005) x Ni(1.55-1.72 s0.01) = 15x15x18=4050
    # -----------------------------------------------------------------------
    A = 3
    _n = "03_A02_lenwide"
    _c = _cfg(_n, (91, 105, 1), (0.69, 0.76, 0.005), (1.55, 1.72, 0.01))
    log.info("A03  Len(91-105 s1)xSu(0.69-0.76 s0.005)xNi(1.55-1.72 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A04  A02_sufiner -- A02 region with denser Su around 0.72
    #      Len(88-104 s1) x Su(0.700-0.745 s0.005) x Ni(1.58-1.72 s0.01) = 17x10x15=2550
    # -----------------------------------------------------------------------
    A = 4
    _n = "04_A02_sufiner"
    _c = _cfg(_n, (88, 104, 1), (0.700, 0.745, 0.005), (1.58, 1.72, 0.01))
    log.info("A04  Len(88-104 s1)xSu(0.700-0.745 s0.005)xNi(1.58-1.72 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A05  longLen_zoom -- long-Len R4 A06/A07 region precision (Len=454-460)
    #      Len(450-480 s3) x Su(0.45-0.55 s0.01) x Ni(1.95-2.10 s0.01) = 11x11x16=1936
    # -----------------------------------------------------------------------
    A = 5
    _n = "05_longLen_zoom"
    _c = _cfg(_n, (450, 480, 3), (0.45, 0.55, 0.01), (1.95, 2.10, 0.01))
    log.info("A05  Len(450-480 s3)xSu(0.45-0.55 s0.01)xNi(1.95-2.10 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A06  bridge_subhunt -- sub-peak hunt Len 110-220
    #      R4 A03 found Len=235 best in bridge, but maybe sub-peak in 110-220
    #      Len(110-220 s10) x Su(0.55-0.78 s0.025) x Ni(1.70-2.15 s0.045) = 12x10x11=1320
    # -----------------------------------------------------------------------
    A = 6
    _n = "06_bridge_subhunt"
    _c = _cfg(_n, (110, 220, 10), (0.55, 0.78, 0.025), (1.70, 2.15, 0.045))
    log.info("A06  Len(110-220 s10)xSu(0.55-0.78 s0.025)xNi(1.70-2.15 s0.045)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A07  A09_lowerNi -- test Ni < 2.10 with A09 region (untested)
    #      Len(232-242 s1) x Su(0.575-0.600 s0.005) x Ni(1.95-2.10 s0.0075) = 11x6x21=1386
    # -----------------------------------------------------------------------
    A = 7
    _n = "07_A09_lowerNi"
    _c = _cfg(_n, (232, 242, 1), (0.575, 0.600, 0.005), (1.95, 2.10, 0.0075))
    log.info("A07  Len(232-242 s1)xSu(0.575-0.600 s0.005)xNi(1.95-2.10 s0.0075)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A08  midSu_bridge -- explore mid-Su (0.62-0.71) across all Lens (untested)
    #      Gap between A09 (Su=0.585) and A02 (Su=0.72) -- may contain hidden peak
    #      Len(90-240 s10) x Su(0.62-0.71 s0.01) x Ni(1.65-2.10 s0.045) = 16x10x11=1760
    # -----------------------------------------------------------------------
    A = 8
    _n = "08_midSu_bridge"
    _c = _cfg(_n, (90, 240, 10), (0.62, 0.71, 0.01), (1.65, 2.10, 0.045))
    log.info("A08  Len(90-240 s10)xSu(0.62-0.71 s0.01)xNi(1.65-2.10 s0.045)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A09  adaptive_zoom1 -- tight zoom on best NP found across A01-A08
    # -----------------------------------------------------------------------
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             best_LEN, best_SU, best_NI, best_np)
    if start_attempt <= A:
        r_len = max(5.0,   best_LEN * 0.06)
        r_su  = max(0.05,  best_SU  * 0.08)
        r_ni  = max(0.08,  best_NI  * 0.08)
        _len = zoom_fixed(best_LEN, r_len, 15,  1.0, LEN_LO, LEN_HI)
        _su  = zoom_fixed(best_SU,  r_su,  15, 0.01, SU_LO,  SU_HI)
        _ni  = zoom_fixed(best_NI,  r_ni,  15, 0.01, NI_LO,  NI_HI)
        _c   = _cfg(_n, _len, _su, _ni)
        log.info("A09  Len%s Su%s Ni%s  %d combos", _len, _su, _ni, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A10  adaptive_zoom2 -- medium zoom
    # -----------------------------------------------------------------------
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             best_LEN, best_SU, best_NI, best_np)
    if start_attempt <= A:
        r_len = max(2.0,   best_LEN * 0.025)
        r_su  = max(0.025, best_SU  * 0.04)
        r_ni  = max(0.04,  best_NI  * 0.04)
        _len = zoom_fixed(best_LEN, r_len, 11,  1.0, LEN_LO, LEN_HI)
        _su  = zoom_fixed(best_SU,  r_su,  11, 0.01, SU_LO,  SU_HI)
        _ni  = zoom_fixed(best_NI,  r_ni,  11, 0.01, NI_LO,  NI_HI)
        _c   = _cfg(_n, _len, _su, _ni)
        log.info("A10  Len%s Su%s Ni%s  %d combos", _len, _su, _ni, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A11  adaptive_zoom3 -- ultra-fine zoom
    # -----------------------------------------------------------------------
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             best_LEN, best_SU, best_NI, best_np)
    if start_attempt <= A:
        r_len = max(1.0,   best_LEN * 0.012)
        r_su  = max(0.012, best_SU  * 0.02)
        r_ni  = max(0.02,  best_NI  * 0.02)
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
    r4_np = SEED_NP
    gain  = (best_np - r4_np) / r4_np * 100 if r4_np else 0
    pct   = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  QuantPassATRex BNBUSDT Hourly Round-5 COMPLETE")
    log.info("  Champion: Len=%.4g Su=%.4g Ni=%.4g", best_LEN, best_SU, best_NI)
    log.info("  Best NP: %.0f USD  (%.1f%% of target %.0f)",
             best_np, pct, TARGET_NP)
    log.info("  R4->R5 gain: %+.2f%%  (%.0f -> %.0f)", gain, r4_np, best_np)
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
        description="QuantPassATRex BNBUSDT HOT Hourly R5 parameter search")
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
