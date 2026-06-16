"""
search_btc_supertrend_hourly2.py — SFJ_SuperTrend_crypto on BTCUSDT HOT Hourly, Round 2

Strategy (from Knowledge/SFJ_SuperTrend_crypto.docx):
  inputs ATRLength(10), Multiplier(3).  Classic SuperTrend:
    Up = C - Mult*ATR(ATRLength) ; Dn = C + Mult*ATR(ATRLength) ; trend flips on close vs bands.
    TREND=1 and C crosses over Dn -> BUY _Crypto1MUSD next bar market
    TREND=-1 and C crosses under Up -> SELLSHORT _Crypto1MUSD next bar market
  2 params only: ATRLength (ATR period), Multiplier (band width).  _Crypto1MUSD=Round(1e6/C,0).

IN-SAMPLE WINDOW = 2022/01/01 - 2026/01/01 (chart trimmed via mc.set_instrument_data_range BEFORE
the attempts; the optimization runs the chart's loaded range, signal date is a no-op).
Objective = NetProfit^2 / |MaxDrawdown| ; target NP > 100,000 USD.

Priors (no SuperTrend crypto prior; 2-param ATR-breakout analog = QPATR_Breakout):
  BTC QPATR_Breakout Hourly: Len=212 Mult=3.27 ($2,748); BNB SuperTrend champ ATR=79 Mult=6.625.
  SuperTrend default ATRLength=10 Mult=3.

R1 Plan (11 attempts, <=5000 combos each):
  A01 global_coarse  ATR(2-400 s8)     Mult(0.5-10 s0.25) = 1950
  A02 short_ATR      ATR(1-60 s1)      Mult(0.5-8 s0.25)  = 1860
  A03 mid_ATR        ATR(40-200 s4)    Mult(1-6 s0.125)   = 1681
  A04 long_ATR       ATR(150-600 s10)  Mult(1-6 s0.25)    = 966
  A05 low_mult       ATR(2-200 s4)     Mult(0.3-3 s0.1)   = 1400
  A06 high_mult      ATR(2-200 s4)     Mult(4-12 s0.25)   = 1650
  A07 vlong_ATR      ATR(400-1500 s25) Mult(1-6 s0.25)    = 945
  A08 fine_default   ATR(2-50 s1)      Mult(1.5-5 s0.125) = 1421
  A09-A11 adaptive zoom (progressively tighter from running best NP)
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


WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_SuperTrend_AI.wsp"
SYMBOL     = "BTCUSDT HOT"
SIGNAL     = "SFJ_SuperTrend_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\btc_supertrend_hourly2_search")

IS_RANGE   = ("2022/01/01", "2026/01/01")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op (signal date does not restrict)

TARGET_NP  = 100_000.0

AT_LO, AT_HI = 1.0,  5000.0    # ATRLength
ML_LO, ML_HI = 0.1,  50.0      # Multiplier

SEED_AT, SEED_ML = 151.0, 9.15   # R1 champion
SEED_NP = 0.0

PREFIX = "BTCSTH2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_supertrend_hourly2_{int(time.time())}.log"
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


def _snap(val, step):
    return round(round(val / step) * step, 8)


def zoom(center, radius, step, lo, hi):
    start = max(lo, _snap(center - radius, step))
    stop  = min(hi, _snap(center + radius, step))
    if stop <= start:
        stop = start + step
    return (start, stop, step)


def n_vals(t):
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name, at, ml):
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    at = _safe(at, AT_LO, AT_HI)
    ml = _safe(ml, ML_LO, ML_HI)

    combos = n_vals(at) * n_vals(ml)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("ATRLength",  *at),
            ParamAxis("Multiplier", *ml),
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


def champion(df, fb_at, fb_ml):
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  TARGET MET: ATR=%.4g Mult=%.4g  NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["ATRLength"]), float(best["Multiplier"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["ATRLength"]), float(best["Multiplier"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: ATR=%.4g Mult=%.4g  obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["ATRLength"]), float(best["Multiplier"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["ATRLength"]), float(best["Multiplier"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_at, fb_ml,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, at, ml, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "ATRLength": at, "Multiplier": ml,
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
        "is_range": IS_RANGE,
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "SuperTrend: Up=C-Mult*ATR, Dn=C+Mult*ATR, trend flip; BUY on TREND=1 & C cross over Dn; SHORT on TREND=-1 & C cross under Up; market entries; reversal",
            "contract": "_Crypto1MUSD=Round(1000000/C,0)",
            "is_window": "2022/01/01-2026/01/01 (chart trimmed via set_instrument_data_range)",
            "priors":   "BTC QPATR_Breakout Hourly Len=212 Mult=3.27 ($2,748); BTC CT $4,155; BNB SuperTrend champ ATR=79 Mult=6.625. SuperTrend default ATRLength=10 Mult=3",
            "r1_plan":  "global + short/mid/long/vlong ATR + low/high mult + fine-default + adaptive zoom",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_btc_supertrend_hourly2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_at, best_ml = SEED_AT, SEED_ML
    best_np  = SEED_NP
    best_obj = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("==============================================================")
    log.info("  BTC Hourly SFJ_SuperTrend_crypto NP>100K -- Round 2")
    log.info("  IS window %s ~ %s  (chart-trimmed)", IS_RANGE[0], IS_RANGE[1])
    log.info("  R1 champ ATR=151 Mult=9.15 NP=1986 (long-ATR wide-band 127tr; confirming peak)", )
    log.info("==============================================================")

    if not from_csv and conn is not None:
        mc.ensure_chart_ready(conn, _cfg("seed", (146, 156, 2), (8.5, 9.75, 0.25)))
        log.info("Trimming chart data range to IS window %s ~ %s ...", *IS_RANGE)
        mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
        log.info("Chart trimmed. (verify rightmost bar ~2026/01, leftmost ~2022/01)")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_at, best_ml, best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df, best_at, best_ml,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-18s]  no valid data", attempt_num, name)
            return

        at, ml, obj, np_, mdd, tr, met = champion(df, best_at, best_ml)

        if np_ > best_np:
            best_at, best_ml = at, ml
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, at, ml, obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-18s]  ATR=%.4g Mult=%.4g  obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, at, ml, obj, np_, mdd, tr,
                 "TARGET" if met else ("%.0f/100K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f", best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    attempts_config = [
        ("01_Mult_sweep",      (140, 165, 5),  (6.0, 12.0, 0.125)),
        ("02_ATR_sweep",       (80, 250, 5),   (8.5, 10.0, 0.125)),
        ("03_high_mult_push",  (120, 180, 5),  (9.0, 15.0, 0.25)),
        ("04_fine_zoom",       (145, 160, 1),  (8.5, 9.75, 0.05)),
        ("05_ATR_bridge_low",  (40, 150, 5),   (7.0, 11.0, 0.25)),
        ("06_ATR_bridge_high", (150, 350, 10), (6.0, 11.0, 0.25)),
        ("07_vhigh_mult",      (100, 200, 10), (11.0, 20.0, 0.5)),
        ("08_short_ATR_recheck",(10, 60, 2),   (5.0, 10.0, 0.25)),
    ]

    for idx, (n, at_r, ml_r) in enumerate(attempts_config, 1):
        _c = _cfg(n, at_r, ml_r)
        log.info("A%02d  %s  %d combos", idx, n, _c.total_runs())
        if start_attempt <= idx:
            _update(run_or_load(n, _c, conn, from_csv), _c, n, idx, _c.total_runs())
        log.info("After A%02d -- best NP=%.0f  (ATR=%.4g Mult=%.4g)", idx, best_np, best_at, best_ml)

    # Adaptive zooms (progressively tighter around running best NP)
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: ATR=%.4g Mult=%.4g  NP=%.0f", best_at, best_ml, best_np)
    if start_attempt <= A:
        for r_at, s_at, r_ml, s_ml in [(20, 2, 1.0, 0.1), (12, 1, 0.6, 0.05), (8, 1, 0.4, 0.05)]:
            _at = zoom(best_at, r_at, s_at, AT_LO, AT_HI)
            _ml = zoom(best_ml, r_ml, s_ml, ML_LO, ML_HI)
            _c  = _cfg(_n, _at, _ml)
            if _c.total_runs() <= 5000:
                break
        log.info("A09  ATR%s Mult%s  %d combos", _at, _ml, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (ATR=%.4g Mult=%.4g)", best_np, best_at, best_ml)

    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: ATR=%.4g Mult=%.4g  NP=%.0f", best_at, best_ml, best_np)
    if start_attempt <= A:
        for r_at, s_at, r_ml, s_ml in [(10, 1, 0.5, 0.05), (6, 1, 0.3, 0.025), (4, 1, 0.2, 0.025)]:
            _at = zoom(best_at, r_at, s_at, AT_LO, AT_HI)
            _ml = zoom(best_ml, r_ml, s_ml, ML_LO, ML_HI)
            _c  = _cfg(_n, _at, _ml)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  ATR%s Mult%s  %d combos", _at, _ml, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (ATR=%.4g Mult=%.4g)", best_np, best_at, best_ml)

    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: ATR=%.4g Mult=%.4g  NP=%.0f", best_at, best_ml, best_np)
    if start_attempt <= A:
        for r_at, s_at, r_ml, s_ml in [(5, 1, 0.25, 0.025), (3, 1, 0.15, 0.0125), (2, 1, 0.1, 0.0125)]:
            _at = zoom(best_at, r_at, s_at, AT_LO, AT_HI)
            _ml = zoom(best_ml, r_ml, s_ml, ML_LO, ML_HI)
            _c  = _cfg(_n, _at, _ml)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  ATR%s Mult%s  %d combos", _at, _ml, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 -- best NP=%.0f  (ATR=%.4g Mult=%.4g)", best_np, best_at, best_ml)

    pct = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  BTC Hourly SuperTrend_crypto Round-2 COMPLETE")
    log.info("  Champion: ATRLength=%.4g Multiplier=%.4g", best_at, best_ml)
    log.info("  Best NP: %.0f USD  (%.1f%% of target %.0f)", best_np, pct, TARGET_NP)
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "MET" if target_met else "NOT MET (gap +%.0f)" % max(0, TARGET_NP - best_np))
    log.info("==============================================================")

    if not best_entry:
        best_entry = {
            "ATRLength": best_at, "Multiplier": best_ml,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone -- results at: {out}")
    return 0


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
    ap = argparse.ArgumentParser(
        description="BTC Hourly SFJ_SuperTrend_crypto NP>100K Round-2 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1-11)")
    ap.add_argument("--probe-trim", action="store_true",
                    help="just trim the chart to the IS window and read it back")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0

    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    if args.probe_trim:
        mc.ensure_chart_ready(conn, _cfg("seed", (146, 156, 2), (8.5, 9.75, 0.25)))
        print(f"\nProbe: trimming chart to IS window {IS_RANGE[0]} ~ {IS_RANGE[1]}")
        mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
        time.sleep(1.0)
        rb_from, rb_to = mc.read_instrument_data_range(conn)
        print(f"  readback From={rb_from} To={rb_to}")
        return 0

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
