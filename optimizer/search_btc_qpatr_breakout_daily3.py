"""
search_btc_qpatr_breakout_daily3.py  QuantPassATR_Breakout on BTCUSDT HOT Daily, Round 3

R1+R2 Summary:
  R1 reported NP-max Len=11 Mult=1.95-1.96 NP=$3,227 (DATA DRIFT ARTIFACT)
  R1 reported Obj-max Len=42 Mult=1.95 NP=$2,898 Obj=14,063 (DATA DRIFT ARTIFACT)
  R2 REVEALED R1 was overnight data drift -- A03 bridge_11_42 with step=0.01 directly
    tested Len=11 Mult=1.95-1.96 and found <$2,495 (R1's $3,227 cannot be reproduced)
  R2 TRUE NP-max ⭐ (4-conv A01=A03=A09=A10): Len=13 Mult=1.86-1.88 NP=$2,495 MDD=-$900 11tr Obj=6,915
  R2 high-freq alt A08: Len=9 Mult=1.0 NP=$1,825 MDD=-$590 58tr (full coverage rows=2254/2254)

R3 Plan (11 attempts, <=5000 combos each, 2D grids):
  Goal: definitively confirm R2 ceiling, prove R1 was drift, redo R2 failures.

  A01 verify_R1_phantom    Len(8-14 s1)    xMult(1.90-2.00 s0.005)  [7x21=147]  -- R1 "$3,227" DEAD test
  A02 R2_full_coverage     Len(10-16 s1)   xMult(1.84-1.92 s0.005)  [7x17=119]  -- R2 champion full grid
  A03 R1_A06_verify        Len(38-46 s1)   xMult(1.90-2.00 s0.005)  [9x21=189]  -- R1 Obj champion test
  A04 redo_high_multi      Len(5-100 s2)   xMult(3.0-10.0 s0.1)     [48x71=3408]
  A05 Mult_deep_below_1    Len(5-50 s1)    xMult(0.1-1.0 s0.02)     [46x46=2116]
  A06 Mult_1_to_1.5        Len(5-50 s1)    xMult(1.0-1.5 s0.01)     [46x51=2346]
  A07 highfreq_Len_short   Len(1-20 s1)    xMult(0.5-2.0 s0.02)     [20x76=1520]
  A08 global_recheck       Len(5-300 s10)  xMult(0.5-5.0 s0.05)     [30x91=2730]
  A09-A11 adaptive zoom (ultra-tight from R2 champion Len=13 Mult=1.87)
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
SYMBOL     = "BTCUSDT HOT"
SIGNAL     = "QuantPassATR_Breakout"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\btc_qpatr_breakout_daily3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 10_000.0

LEN_LO,  LEN_HI   = 1.0,   2000.0
MULT_LO, MULT_HI  = 0.01,  50.0

# R2 NP-max champion as seed (4-conv)
SEED_LEN  = 13.0
SEED_MULT = 1.87
SEED_NP   = 2495.0

PREFIX = "BTCQPBD3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_qpatr_breakout_daily3_{int(time.time())}.log"
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
    mult   = _safe(mult,   MULT_LO, MULT_HI, 0.005)

    combos = n_vals(length) * n_vals(mult)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
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
        "round": 3,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "ATR=AvgTrueRange(Len); LE/SE: C><C[1]+/-ATR[1]*Mult -> market; reversal exits",
            "contract": "_Crypto1MUSD=Round(1000000/C,0)",
            "r1_result":"R1 reported $3,227 (Len=11 Mult=1.96) and $2,898 (Len=42 Mult=1.95) -- BOTH DATA DRIFT ARTIFACTS",
            "r2_result":"R2 TRUE ceiling: Len=13 Mult=1.86-1.88 NP=$2,495 (4-conv); A08 alt Len=9 Mult=1.0 NP=$1,825 58tr (low MDD)",
            "r3_plan":  "Verify R1 phantom dead + R2 full coverage + R1 A06 verify + redo high_multi + Mult<1 deep + global",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_btc_qpatr_breakout_daily3.json"
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
    log.info("  QuantPassATR_Breakout on BTCUSDT HOT Daily -- Round 3")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("  R2 TRUE NP-max (4-conv): Len=13 Mult=1.87 NP=$2,495")
    log.info("  R1 reported $3,227 was DATA DRIFT ARTIFACT")
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
                 "TARGET" if met else ("NP=%.0f/10K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    attempts_config = [
        ("01_verify_R1_phantom",  (8, 14, 1),       (1.90, 2.00, 0.005)),
        ("02_R2_full_coverage",   (10, 16, 1),      (1.84, 1.92, 0.005)),
        ("03_R1_A06_verify",      (38, 46, 1),      (1.90, 2.00, 0.005)),
        ("04_redo_high_multi",    (5, 100, 2),      (3.0, 10.0, 0.1)),
        ("05_Mult_deep_below_1",  (5, 50, 1),       (0.1, 1.0, 0.02)),
        ("06_Mult_1_to_1.5",      (5, 50, 1),       (1.0, 1.5, 0.01)),
        ("07_highfreq_Len_short", (1, 20, 1),       (0.5, 2.0, 0.02)),
        ("08_global_recheck",     (5, 300, 10),     (0.5, 5.0, 0.05)),
    ]

    for idx, (n, len_range, mult_range) in enumerate(attempts_config, 1):
        _c = _cfg(n, len_range, mult_range)
        log.info("A%02d  %s  %d combos", idx, n, _c.total_runs())
        if start_attempt <= idx:
            _update(run_or_load(n, _c, conn, from_csv), _c, n, idx, _c.total_runs())
        log.info("After A%02d -- best NP=%.0f  (Len=%.4g Mult=%.4g)",
                 idx, best_np, best_LEN, best_MULT)

    # Adaptive zooms (ultra-tight from R2 champion Len=13 Mult=1.87)
    zoom_configs = [
        (9,  "09_adaptive_zoom1", 0.15, 0.06, 25, 33),
        (10, "10_adaptive_zoom2", 0.08, 0.03, 17, 21),
        (11, "11_adaptive_zoom3", 0.04, 0.015, 11, 15),
    ]
    for A, _n, r_len_pct, r_mult_pct, n_len, n_mult in zoom_configs:
        log.info("A%02d  %s -- center: Len=%.4g Mult=%.4g NP=%.0f",
                 A, _n, best_LEN, best_MULT, best_np)
        if start_attempt <= A:
            r_len  = max(3.0 if A == 9 else (2.0 if A == 10 else 1.0),  best_LEN  * r_len_pct)
            r_mult = max(0.08 if A == 9 else (0.04 if A == 10 else 0.02), best_MULT * r_mult_pct)
            _len  = zoom_fixed(best_LEN,  r_len,  n_len,  1.0, LEN_LO,  LEN_HI)
            _mult = zoom_fixed(best_MULT, r_mult, n_mult, 0.005, MULT_LO, MULT_HI)
            _c   = _cfg(_n, _len, _mult)
            log.info("A%02d  Len%s Mult%s  %d combos", A, _len, _mult, _c.total_runs())
            _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
        log.info("After A%02d -- best NP=%.0f  (Len=%.4g Mult=%.4g)",
                 A, best_np, best_LEN, best_MULT)

    r2_np = SEED_NP
    gain  = (best_np - r2_np) / r2_np * 100 if r2_np else 0
    pct   = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  QuantPassATR_Breakout BTCUSDT Daily Round-3 COMPLETE")
    log.info("  Champion: Len=%.4g Multiple=%.4g", best_LEN, best_MULT)
    log.info("  Best NP: %.0f USD  (%.1f%% of target %.0f)", best_np, pct, TARGET_NP)
    log.info("  R2->R3 gain: %+.2f%%  (%.0f -> %.0f)", gain, r2_np, best_np)
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
        description="QuantPassATR_Breakout BTCUSDT HOT Daily R3 parameter search")
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
