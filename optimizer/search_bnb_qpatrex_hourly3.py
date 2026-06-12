"""
search_bnb_qpatrex_hourly3.py  QuantPassATRex on BNBUSDT HOT Hourly, Round 3

R1 best NP=$34,691 (Len=27 Su=3.0 Ni=3.68 NP-max regime; Su=3.0 boundary)
R2 best NP=$35,856 (Len=37 Su=2.43 Ni=3.63 NP-max regime) +3.4%
R2 Obj-max breakthrough A04: Len=100 Su=0.72 Ni=1.5 NP=$30,608 MDD=-$4,464
                              192 trades, Obj=209,881 (R1->R2 Obj +119%, 8.1x BTC's best)

R2 key insights:
  - Su>3.0 ruled out (Su=3.0 was boundary artifact, true peak Su=2.4-2.8)
  - Obj-max is a CONTINUOUS RIDGE Len=100-470 with consistent Su=0.5-0.72 Ni=1.3-2.2
  - Multiple Obj-max sub-regimes confirmed (A03, A04, A06, A07, A08 all >130K Obj)
  - Both regimes plateau at ~$35K NP, gap to $100K = -64%

R3 Plan (11 attempts, <=5000 combos each):
  Headline: ultra-fine zoom A04 Obj champion + zoom each Obj-ridge peak + boundary explore

  A01 A04_ultrafine     Len(86-118 s2)xSu(0.55-0.95 s0.025)xNi(1.2-1.85 s0.04)  [17x17x17=4913]
  A02 A07_ultrafine     Len(440-510 s4)xSu(0.4-0.65 s0.02) xNi(1.1-1.85 s0.05)  [18x13x16=3744]
  A03 A03_finezoom      Len(210-260 s4)xSu(0.5-0.75 s0.02) xNi(1.8-2.6 s0.05)   [13x13x17=2873]
  A04 A06_finezoom      Len(120-170 s3)xSu(0.55-0.95 s0.03)xNi(1.0-1.7 s0.05)   [17x14x15=3570]
  A05 ridge_connect     Len(100-470 s30)xSu(0.5-0.8 s0.03) xNi(1.0-2.5 s0.15)   [13x11x11=1573]
  A06 Len_lt_100        Len(30-100 s5)xSu(0.4-1.0 s0.05)   xNi(0.8-2.5 s0.15)   [15x13x12=2340]
  A07 Len_gt_500        Len(500-1000 s40)xSu(0.3-0.8 s0.04)xNi(0.8-2.0 s0.1)    [13x13x13=2197]
  A08 npmax_confirm     Len(30-45 s1)xSu(2.2-2.8 s0.04)   xNi(3.2-4.2 s0.07)    [16x16x15=3840]
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_qpatrex_hourly3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO,  LEN_HI  = 1.0,   2000.0
SU_LO,   SU_HI   = 0.01,  50.0
NI_LO,   NI_HI   = 0.01,  100.0

# R2 NP-max champion as seed
SEED_LEN = 37.0
SEED_SU  = 2.43
SEED_NI  = 3.63
SEED_NP  = 35856.0

PREFIX = "BNBQP3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_qpatrex_hourly3_{int(time.time())}.log"
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
        "round": 3,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "ATRex=ATR(Len)+StdDev(C,Len,1); LE_Su: C>C[1]+ATRex[1]*Su -> H stop; LE_Ni: C>L[1]+ATRex[1]*Ni -> market; both long+short, reversal exits",
            "contract": "_Crypto1MUSD=Round(1000000/C,0) -- ~$1M notional per trade",
            "r1_result": "R1 NP-max Len=27 Su=3.0 Ni=3.68 NP=$34,691; Obj-max A05 Obj=95,614",
            "r2_result": "R2 NP-max Len=37 Su=2.43 Ni=3.63 NP=$35,856 (+3.4%); A04 Obj-max Len=100 Su=0.72 Ni=1.5 Obj=209,881 (+119%)",
            "r3_plan":  "Ultra-fine zoom A04/A07/A03/A06 + Obj ridge connectivity + Len boundary explorations + NP-max confirm",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_qpatrex_hourly3.json"
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
    log.info("  QuantPassATRex on BNBUSDT HOT Hourly -- Round 3")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD  (contract: ~$1M notional, _Crypto1MUSD)", TARGET_NP)
    log.info("  R2 NP-max: Len=%.4g Su=%.4g Ni=%.4g  NP=%.0f",
             SEED_LEN, SEED_SU, SEED_NI, SEED_NP)
    log.info("  R2 A04 Obj-king: Len=100 Su=0.72 Ni=1.5 NP=$30,608 MDD=-$4,464 Obj=209,881")
    log.info("  Goal: ultra-fine zoom 4 Obj-max sub-regimes + ridge connectivity + bounds")
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
    # A01  A04_ultrafine -- ultra-fine zoom of R2 A04 Obj champion (Len=100)
    #      R2 A04: Len=100 Su=0.72 Ni=1.5 Obj=209,881 (best Obj across BTC/ETH/BNB)
    #      Len(86-118 s2) x Su(0.55-0.95 s0.025) x Ni(1.2-1.85 s0.04) = 17x17x17=4913
    # -----------------------------------------------------------------------
    A = 1
    _n = "01_A04_ultrafine"
    _c = _cfg(_n, (86, 118, 2), (0.55, 0.95, 0.025), (1.2, 1.85, 0.04))
    log.info("A01  Len(86-118 s2)xSu(0.55-0.95 s0.025)xNi(1.2-1.85 s0.04)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A02  A07_ultrafine -- ultra-fine zoom of R2 A07 (Len=470, Obj=201,705)
    #      Len(440-510 s4) x Su(0.4-0.65 s0.02) x Ni(1.1-1.85 s0.05) = 18x13x16=3744
    # -----------------------------------------------------------------------
    A = 2
    _n = "02_A07_ultrafine"
    _c = _cfg(_n, (440, 510, 4), (0.4, 0.65, 0.02), (1.1, 1.85, 0.05))
    log.info("A02  Len(440-510 s4)xSu(0.4-0.65 s0.02)xNi(1.1-1.85 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A03  A03_finezoom -- fine zoom of R2 A03 (Len=230, Obj=184,086)
    #      Len(210-260 s4) x Su(0.5-0.75 s0.02) x Ni(1.8-2.6 s0.05) = 13x13x17=2873
    # -----------------------------------------------------------------------
    A = 3
    _n = "03_A03_finezoom"
    _c = _cfg(_n, (210, 260, 4), (0.5, 0.75, 0.02), (1.8, 2.6, 0.05))
    log.info("A03  Len(210-260 s4)xSu(0.5-0.75 s0.02)xNi(1.8-2.6 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A04  A06_finezoom -- fine zoom of R2 A06 (Len=140, Obj=142,886)
    #      Len(120-170 s3) x Su(0.55-0.95 s0.03) x Ni(1.0-1.7 s0.05) = 17x14x15=3570
    # -----------------------------------------------------------------------
    A = 4
    _n = "04_A06_finezoom"
    _c = _cfg(_n, (120, 170, 3), (0.55, 0.95, 0.03), (1.0, 1.7, 0.05))
    log.info("A04  Len(120-170 s3)xSu(0.55-0.95 s0.03)xNi(1.0-1.7 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A05  ridge_connect -- Obj-max ridge connectivity (Len=100 to 470)
    #      Sweep Len axis to map continuous Obj-max region
    #      Len(100-470 s30) x Su(0.5-0.8 s0.03) x Ni(1.0-2.5 s0.15) = 13x11x11=1573
    # -----------------------------------------------------------------------
    A = 5
    _n = "05_ridge_connect"
    _c = _cfg(_n, (100, 470, 30), (0.5, 0.8, 0.03), (1.0, 2.5, 0.15))
    log.info("A05  Len(100-470 s30)xSu(0.5-0.8 s0.03)xNi(1.0-2.5 s0.15)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A06  Len_lt_100 -- explore Len 30-100 with Obj-max params
    #      Bridge gap between NP-max (Len=37) and Obj-max ridge start (Len=100)
    #      Len(30-100 s5) x Su(0.4-1.0 s0.05) x Ni(0.8-2.5 s0.15) = 15x13x12=2340
    # -----------------------------------------------------------------------
    A = 6
    _n = "06_Len_lt_100"
    _c = _cfg(_n, (30, 100, 5), (0.4, 1.0, 0.05), (0.8, 2.5, 0.15))
    log.info("A06  Len(30-100 s5)xSu(0.4-1.0 s0.05)xNi(0.8-2.5 s0.15)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A07  Len_gt_500 -- explore Len 500-1000 (beyond R2 A07's Len=470)
    #      Check if Obj-max ridge extends beyond 500
    #      Len(500-1000 s40) x Su(0.3-0.8 s0.04) x Ni(0.8-2.0 s0.1) = 13x13x13=2197
    # -----------------------------------------------------------------------
    A = 7
    _n = "07_Len_gt_500"
    _c = _cfg(_n, (500, 1000, 40), (0.3, 0.8, 0.04), (0.8, 2.0, 0.1))
    log.info("A07  Len(500-1000 s40)xSu(0.3-0.8 s0.04)xNi(0.8-2.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 -- best NP=%.0f  (Len=%.4g Su=%.4g Ni=%.4g)",
             best_np, best_LEN, best_SU, best_NI)

    # -----------------------------------------------------------------------
    # A08  npmax_confirm -- fine confirm NP-max regime around R2 A11
    #      R2 A11: Len=37 Su=2.43 Ni=3.63 NP=$35,856
    #      Len(30-45 s1) x Su(2.2-2.8 s=0.04) x Ni(3.2-4.2 s0.07) = 16x16x15=3840
    # -----------------------------------------------------------------------
    A = 8
    _n = "08_npmax_confirm"
    _c = _cfg(_n, (30, 45, 1), (2.2, 2.8, 0.04), (3.2, 4.2, 0.07))
    log.info("A08  Len(30-45 s1)xSu(2.2-2.8 s0.04)xNi(3.2-4.2 s0.07)  %d combos",
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
        r_len = max(15.0,  best_LEN * 0.20)
        r_su  = max(0.20,  best_SU  * 0.25)
        r_ni  = max(0.25,  best_NI  * 0.25)
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
        r_len = max(6.0,   best_LEN * 0.08)
        r_su  = max(0.08,  best_SU  * 0.10)
        r_ni  = max(0.10,  best_NI  * 0.10)
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
        r_len = max(3.0,   best_LEN * 0.04)
        r_su  = max(0.04,  best_SU  * 0.05)
        r_ni  = max(0.05,  best_NI  * 0.05)
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
    r2_np = SEED_NP
    gain  = (best_np - r2_np) / r2_np * 100 if r2_np else 0
    pct   = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  QuantPassATRex BNBUSDT Hourly Round-3 COMPLETE")
    log.info("  Champion: Len=%.4g Su=%.4g Ni=%.4g", best_LEN, best_SU, best_NI)
    log.info("  Best NP: %.0f USD  (%.1f%% of target %.0f)",
             best_np, pct, TARGET_NP)
    log.info("  R2->R3 gain: %+.2f%%  (%.0f -> %.0f)", gain, r2_np, best_np)
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
        description="QuantPassATRex BNBUSDT HOT Hourly R3 parameter search")
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
