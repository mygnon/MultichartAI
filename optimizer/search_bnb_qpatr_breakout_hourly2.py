"""
search_bnb_qpatr_breakout_hourly2.py  QuantPassATR_Breakout on BNBUSDT HOT Hourly, Round 2

R1 Summary:
  A04=A09=A10=A11 NP-max: Len=3 Multiple=2.965 NP=$35,634 MDD=-$6,255 82tr Obj=202,995
  A03 Obj-max: Len=145 Multiple=2.91 NP=$32,506 MDD=-$4,610 (lowest!) 98tr Obj=229,204 (highest!)
  A01 Len=115 NP=$33,811 108tr Obj=193,210 (third regime)
  A06 Len=88 NP=$27,341 146tr (highest freq, untested fine)
  BNB/BTC ratio 12.97x (vs QPATRex 12.1x -- consistent cross-symbol scaling)
  2nd place across all 6 strategy x symbol Hourly searches.

R2 Plan (11 attempts, <=5000 combos each, 2D grids):
  Goal: fine-zoom A03 Obj champion + A09 NP champion + bridge gaps + axis precision.

  A01 A03_ultrafine       Len(125-165 s1)  xMult(2.80-3.05 s0.005) [41x51=2091]
  A02 A09_ultrafine       Len(2-15 s1)     xMult(2.80-3.10 s0.005) [14x61=854]
  A03 A01_region_zoom     Len(100-130 s1)  xMult(2.55-2.90 s0.005) [31x71=2201]
  A04 bridge_130_220      Len(130-220 s2)  xMult(2.70-3.05 s0.01)  [46x36=1656]
  A05 A06_highfreq_fine   Len(75-110 s1)   xMult(2.30-2.70 s0.01)  [36x41=1476]
  A06 bridge_3_50         Len(3-50 s1)     xMult(2.70-3.20 s0.01)  [48x51=2448]
  A07 Multi_axis_fine     Len(140-150 s1)  xMult(2.60-3.20 s0.005) [11x121=1331]
  A08 global_recheck      Len(5-300 s10)   xMult(0.5-5.0 s0.05)    [30x91=2730]
  A09-A11 adaptive zoom (tighter than R1)
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


WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_QuantPassATR_Breakout_AI.wsp"
SYMBOL     = "BNBUSDT HOT"
SIGNAL     = "QuantPassATR_Breakout"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_qpatr_breakout_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 100_000.0

LEN_LO,  LEN_HI   = 1.0,   2000.0
MULT_LO, MULT_HI  = 0.01,  50.0

# R1 NP-max champion as seed
SEED_LEN  = 3.0
SEED_MULT = 2.965
SEED_NP   = 35634.0

PREFIX = "BNBQPB2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_qpatr_breakout_hourly2_{int(time.time())}.log"
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


def zoom_fixed(center, radius, n_target, step_min, lo, hi):
    lo_val = max(lo, center - radius)
    hi_val = min(hi, center + radius)
    if lo_val >= hi_val:
        return (max(lo, center - step_min), min(hi, center + step_min), step_min)
    rng = hi_val - lo_val
    step = max(step_min, math.ceil(rng / max(1, n_target - 1) / step_min) * step_min)
    return (lo_val, hi_val, step)


def n_vals(t):
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name, length, mult):
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


def csv_for(name):
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
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  TARGET MET: Len=%.4g Mult=%.4g  NP=%.0f MDD=%.0f obj=%.0f tr=%d",
                 float(best["Len"]), float(best["Multiple"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["Len"]), float(best["Multiple"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: Len=%.4g Mult=%.4g  obj=%.0f NP=%.0f MDD=%.0f tr=%d",
                 float(best["Len"]), float(best["Multiple"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["Len"]), float(best["Multiple"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_len, fb_mult, 0.0,
            float(best["NetProfit"]), float(best["MaxDrawdown"]),
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
            "r1_result": "R1 NP-max A04=A09=A10=A11: Len=3 Mult=2.965 NP=$35,634; Obj-max A03: Len=145 Mult=2.91 NP=$32,506 Obj=229,204",
            "r2_plan":  "Fine-zoom A03 Obj champion + A09 NP champion + bridges + axis precision",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_qpatr_breakout_hourly2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


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
    log.info("  QuantPassATR_Breakout on BNBUSDT HOT Hourly -- Round 2")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("  R1 NP-max: Len=%.4g Mult=%.4g NP=%.0f",
             SEED_LEN, SEED_MULT, SEED_NP)
    log.info("  R1 Obj-max A03: Len=145 Mult=2.91 NP=$32,506 Obj=229,204 (highest)")
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

        log.info("  [A%02d %-24s]  Len=%.4g Mult=%.4g  obj=%.0f NP=%.0f MDD=%.0f tr=%d  %s",
                 attempt_num, name, length, mult, obj, np_, mdd, tr,
                 "TARGET" if met else ("NP=%.0f/100K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    attempts_config = [
        ("01_A03_ultrafine",     (125, 165, 1),  (2.80, 3.05, 0.005)),
        ("02_A09_ultrafine",     (2, 15, 1),     (2.80, 3.10, 0.005)),
        ("03_A01_region_zoom",   (100, 130, 1),  (2.55, 2.90, 0.005)),
        ("04_bridge_130_220",    (130, 220, 2),  (2.70, 3.05, 0.01)),
        ("05_A06_highfreq_fine", (75, 110, 1),   (2.30, 2.70, 0.01)),
        ("06_bridge_3_50",       (3, 50, 1),     (2.70, 3.20, 0.01)),
        ("07_Multi_axis_fine",   (140, 150, 1),  (2.60, 3.20, 0.005)),
        ("08_global_recheck",    (5, 300, 10),   (0.5, 5.0, 0.05)),
    ]

    for idx, (n, len_range, mult_range) in enumerate(attempts_config, 1):
        _c = _cfg(n, len_range, mult_range)
        log.info("A%02d  %s  %d combos", idx, n, _c.total_runs())
        if start_attempt <= idx:
            _update(run_or_load(n, _c, conn, from_csv), _c, n, idx, _c.total_runs())
        log.info("After A%02d -- best NP=%.0f  (Len=%.4g Mult=%.4g)",
                 idx, best_np, best_LEN, best_MULT)

    # Adaptive zooms (tighter than R1)
    zoom_configs = [
        (9,  "09_adaptive_zoom1", 0.15, 0.20, 31, 61),
        (10, "10_adaptive_zoom2", 0.08, 0.10, 21, 41),
        (11, "11_adaptive_zoom3", 0.04, 0.05, 15, 31),
    ]
    for A, _n, r_len_pct, r_mult_pct, n_len, n_mult in zoom_configs:
        log.info("A%02d  %s -- center: Len=%.4g Mult=%.4g NP=%.0f",
                 A, _n, best_LEN, best_MULT, best_np)
        if start_attempt <= A:
            r_len  = max(5.0 if A == 9 else (3.0 if A == 10 else 2.0),  best_LEN  * r_len_pct)
            r_mult = max(0.15 if A == 9 else (0.08 if A == 10 else 0.04), best_MULT * r_mult_pct)
            _len  = zoom_fixed(best_LEN,  r_len,  n_len,  1.0, LEN_LO,  LEN_HI)
            _mult = zoom_fixed(best_MULT, r_mult, n_mult, 0.01, MULT_LO, MULT_HI)
            _c   = _cfg(_n, _len, _mult)
            log.info("A%02d  Len%s Mult%s  %d combos", A, _len, _mult, _c.total_runs())
            _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
        log.info("After A%02d -- best NP=%.0f  (Len=%.4g Mult=%.4g)",
                 A, best_np, best_LEN, best_MULT)

    r1_np = SEED_NP
    gain  = (best_np - r1_np) / r1_np * 100 if r1_np else 0
    pct   = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  QuantPassATR_Breakout BNBUSDT Hourly Round-2 COMPLETE")
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
        description="QuantPassATR_Breakout BNBUSDT HOT Hourly R2 parameter search")
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
