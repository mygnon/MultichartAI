"""
search_btc_qpatr_breakout_hourly2.py  QuantPassATR_Breakout on BTCUSDT HOT Hourly, Round 2

R1 Summary:
  A10 NP champion: Len=212.8 Multiple=3.2825 NP=$2,748 MDD=-$689 90tr Obj=10,960
  A10 > A09 ($2,697) > A11 ($2,697) -- A10 still climbing slightly
  A04 surprise: Len=2 Mult=3.55 NP=$2,445 77tr Obj=9,469 (very-short Len sub-peak)
  A07 high_multi: Len=155 Mult=4.91 NP=$2,319 MDD=-$1,136 (Multi>4.5 catastrophic)
  R1 vs QPATRex BTC Hourly: NP -16.5% but MDD -50% and 7x trades (MDD/NP 25% vs 42%)

R2 Plan (11 attempts, <=5000 combos each, 2D grids):
  Goal: ultra-fine A10 zoom + hunt A04 sub-peak + bridge gaps + axis-precision.

  A01 A10_ultrafine   Len(200-225 s1)  xMult(3.10-3.40 s0.005) [26x61=1586]
  A02 vshort_Len      Len(2-15 s1)     xMult(2.5-4.5 s0.025)   [14x81=1134]
  A03 bridge_50_150   Len(50-150 s2)   xMult(2.5-4.5 s0.04)    [51x51=2601]
  A04 bridge_150_200  Len(150-200 s2)  xMult(2.8-4.0 s0.025)   [26x49=1274]
  A05 Multi_axis_fine Len(200-220 s2)  xMult(2.0-4.5 s0.015)   [11x167=1837]
  A06 Len_axis_fine   Len(180-250 s1)  xMult(3.15-3.45 s0.01)  [71x31=2201]
  A07 A06_region      Len(30-60 s1)    xMult(3.5-4.5 s0.02)    [31x51=1581]
  A08 superlong       Len(300-1000 s30)xMult(0.5-4.0 s0.05)    [24x71=1704]
  A09-A11 adaptive zoom (2D)
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_QuantPassATR_Breakout_AI.wsp"
SYMBOL     = "BTCUSDT HOT"
SIGNAL     = "QuantPassATR_Breakout"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\btc_qpatr_breakout_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 10_000.0   # USD

LEN_LO,  LEN_HI   = 1.0,   2000.0
MULT_LO, MULT_HI  = 0.01,  50.0

# R1 A10 champion as seed
SEED_LEN  = 212.8
SEED_MULT = 3.2825
SEED_NP   = 2748.0

PREFIX = "BTCQPB2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_qpatr_breakout_hourly2_{int(time.time())}.log"
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
         mult:   Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi, step_min):
        s, e, step = t
        if abs(s - e) < step_min * 0.5:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    length = _safe(length, LEN_LO,  LEN_HI,  1.0)
    mult   = _safe(mult,   MULT_LO, MULT_HI, 0.01)

    combos = n_vals(length) * n_vals(mult)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("Len",      *length),
            ParamAxis("Multiple", *mult),
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


def champion(df, fb_len, fb_mult):
    """Target-chasing mode: highest-NP row drives zoom seed."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  TARGET MET: Len=%.4g Mult=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["Len"]), float(best["Multiple"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["Len"]), float(best["Multiple"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: Len=%.4g Mult=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["Len"]), float(best["Multiple"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["Len"]), float(best["Multiple"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_len, fb_mult,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, length, mult, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "Len": length, "Multiple": mult,
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
            "logic":    "ATR=AvgTrueRange(Len); LE/SE: C><C[1]+/-ATR[1]*Mult -> market; reversal exits",
            "contract": "_Crypto1MUSD=Round(1000000/C,0)",
            "r1_result": "R1 A10 NP champion: Len=212.8 Mult=3.2825 NP=$2,748 MDD=-$689 90tr Obj=10,960; A04 sub-peak Len=2 NP=$2,445",
            "r2_plan":  "Ultra-fine A10 (Mult step=0.005) + A04 sub-peak hunt + bridges + axis precision + superlong",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_btc_qpatr_breakout_hourly2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_LEN  = SEED_LEN
    best_MULT = SEED_MULT
    best_np   = SEED_NP
    best_obj  = 0.0
    target_met  = False
    attempt_log: List[dict] = []
    best_entry:  Dict = {}

    log.info("==============================================================")
    log.info("  QuantPassATR_Breakout on BTCUSDT HOT Hourly -- Round 2")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("  R1 A10: Len=%.4g Mult=%.4g NP=%.0f (still climbing)",
             SEED_LEN, SEED_MULT, SEED_NP)
    log.info("==============================================================")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_LEN, best_MULT, best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_LEN, best_MULT,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        length, mult, obj, np_, mdd, tr, met = champion(df, best_LEN, best_MULT)

        if np_ > best_np:
            best_LEN, best_MULT = length, mult
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, length, mult, obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  Len=%.4g Mult=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, length, mult, obj, np_, mdd, tr,
                 "TARGET" if met else ("NP=%.0f/10K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # A01  A10_ultrafine (Mult step=0.005, captures sub-peaks around 3.28)
    A = 1
    _n = "01_A10_ultrafine"
    _c = _cfg(_n, (200, 225, 1), (3.10, 3.40, 0.005))
    log.info("A01  Len(200-225 s1)xMult(3.10-3.40 s0.005)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A02  vshort_Len (R1 A04 Len=2 sub-peak hunt)
    A = 2
    _n = "02_vshort_Len"
    _c = _cfg(_n, (2, 15, 1), (2.5, 4.5, 0.025))
    log.info("A02  Len(2-15 s1)xMult(2.5-4.5 s0.025)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A03  bridge_50_150
    A = 3
    _n = "03_bridge_50_150"
    _c = _cfg(_n, (50, 150, 2), (2.5, 4.5, 0.04))
    log.info("A03  Len(50-150 s2)xMult(2.5-4.5 s0.04)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A04  bridge_150_200
    A = 4
    _n = "04_bridge_150_200"
    _c = _cfg(_n, (150, 200, 2), (2.8, 4.0, 0.025))
    log.info("A04  Len(150-200 s2)xMult(2.8-4.0 s0.025)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A05  Multi_axis_fine
    A = 5
    _n = "05_Multi_axis_fine"
    _c = _cfg(_n, (200, 220, 2), (2.0, 4.5, 0.015))
    log.info("A05  Len(200-220 s2)xMult(2.0-4.5 s0.015)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A06  Len_axis_fine
    A = 6
    _n = "06_Len_axis_fine"
    _c = _cfg(_n, (180, 250, 1), (3.15, 3.45, 0.01))
    log.info("A06  Len(180-250 s1)xMult(3.15-3.45 s0.01)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A07  A06_region (R1 A06 Len=42 zone)
    A = 7
    _n = "07_A06_region"
    _c = _cfg(_n, (30, 60, 1), (3.5, 4.5, 0.02))
    log.info("A07  Len(30-60 s1)xMult(3.5-4.5 s0.02)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A08  superlong (Len > 300 untested in R1)
    A = 8
    _n = "08_superlong"
    _c = _cfg(_n, (300, 1000, 30), (0.5, 4.0, 0.05))
    log.info("A08  Len(300-1000 s30)xMult(0.5-4.0 s0.05)  %d combos", _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A09  adaptive_zoom1
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: Len=%.4g Mult=%.4g NP=%.0f",
             best_LEN, best_MULT, best_np)
    if start_attempt <= A:
        r_len  = max(8.0,   best_LEN  * 0.10)
        r_mult = max(0.15,  best_MULT * 0.15)
        _len  = zoom_fixed(best_LEN,  r_len,  41, 1.0, LEN_LO,  LEN_HI)
        _mult = zoom_fixed(best_MULT, r_mult, 81, 0.01, MULT_LO, MULT_HI)
        _c   = _cfg(_n, _len, _mult)
        log.info("A09  Len%s Mult%s  %d combos", _len, _mult, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A10  adaptive_zoom2
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: Len=%.4g Mult=%.4g NP=%.0f",
             best_LEN, best_MULT, best_np)
    if start_attempt <= A:
        r_len  = max(4.0,   best_LEN  * 0.05)
        r_mult = max(0.08,  best_MULT * 0.08)
        _len  = zoom_fixed(best_LEN,  r_len,  31, 1.0, LEN_LO,  LEN_HI)
        _mult = zoom_fixed(best_MULT, r_mult, 61, 0.01, MULT_LO, MULT_HI)
        _c   = _cfg(_n, _len, _mult)
        log.info("A10  Len%s Mult%s  %d combos", _len, _mult, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # A11  adaptive_zoom3
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: Len=%.4g Mult=%.4g NP=%.0f",
             best_LEN, best_MULT, best_np)
    if start_attempt <= A:
        r_len  = max(2.0,   best_LEN  * 0.025)
        r_mult = max(0.04,  best_MULT * 0.04)
        _len  = zoom_fixed(best_LEN,  r_len,  21, 1.0, LEN_LO,  LEN_HI)
        _mult = zoom_fixed(best_MULT, r_mult, 41, 0.01, MULT_LO, MULT_HI)
        _c   = _cfg(_n, _len, _mult)
        log.info("A11  Len%s Mult%s  %d combos", _len, _mult, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 -- best NP=%.0f  (Len=%.4g Mult=%.4g)", best_np, best_LEN, best_MULT)

    # Final summary
    r1_np = SEED_NP
    gain  = (best_np - r1_np) / r1_np * 100 if r1_np else 0
    pct   = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  QuantPassATR_Breakout BTCUSDT Hourly Round-2 COMPLETE")
    log.info("  Champion: Len=%.4g Multiple=%.4g", best_LEN, best_MULT)
    log.info("  Best NP: %.0f USD  (%.1f%% of target %.0f)", best_np, pct, TARGET_NP)
    log.info("  R1->R2 gain: %+.2f%%  (%.0f -> %.0f)", gain, r1_np, best_np)
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "MET" if target_met else "NOT MET (gap +%.0f)" % max(0, TARGET_NP - best_np))
    log.info("==============================================================")

    if not best_entry:
        best_entry = {
            "Len": best_LEN, "Multiple": best_MULT,
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
        description="QuantPassATR_Breakout BTCUSDT HOT Hourly R2 parameter search")
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
