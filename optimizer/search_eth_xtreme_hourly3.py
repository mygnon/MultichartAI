"""
search_eth_xtreme_hourly3.py — SFJ_XtremeStop_Crypto on ETHUSDT HOT Hourly, Round 3

Strategy (from Knowledge/SFJ_XtremeStop_Crypto.docx):
  INPUT X(150), LY(3.25), SY(3.25).  Reversal-only stop-breakout vs the close X bars ago:
    if pos<>1  : BUY  _Crypto1MUSD next bar at C[X]*(1+LY*0.01) STOP
    if pos<>-1 : SHORT _Crypto1MUSD next bar at C[X]*(1-SY*0.01) STOP
  X = lookback bars; LY/SY = long/short breakout percent.  _Crypto1MUSD=Round(1e6/C,0).

IN-SAMPLE WINDOW = 2022/01/01 - 2026/01/01 (chart trimmed via mc.set_instrument_data_range BEFORE
the attempts; the optimization runs the chart's loaded range, signal date is a no-op).
Objective = NetProfit^2 / |MaxDrawdown| ; target NP > 100,000 USD.

Priors (XtremeStop on futures; no crypto prior):
  BNB crypto X=905 LY=3.0 SY=10.65 (asym long-X); BTC crypto X=60 LY=12.75 SY=12.4 (sym high-pct). ETH between.
  Crypto default X=150 LY=3.25 SY=3.25 (more volatile -> larger % breakouts plausible).

R1 Plan (11 attempts, <=5000 combos each):
  A01 global_coarse  X(5-305 s30)    LY(0.5-8 s0.5)   SY(0.5-8 s0.5)    = 2816
  A02 short_X        X(1-50 s3)      LY(0.5-7 s0.5)   SY(0.5-7 s0.5)    = 3332
  A03 mid_X          X(80-300 s20)   LY(1-8 s0.5)     SY(1-8 s0.5)      = 2700
  A04 long_X         X(300-1000 s50) LY(1-8 s0.5)     SY(1-8 s0.5)      = 3375
  A05 high_pct       X(5-305 s30)    LY(5-15 s0.5)    SY(5-15 s0.5)     = 4851
  A06 low_pct        X(5-205 s20)    LY(0.2-3 s0.2)   SY(0.2-3 s0.2)    = 2475
  A07 asym_LY_high   X(10-160 s15)   LY(2-12 s0.5)    SY(0.5-5 s0.5)    = 2310
  A08 asym_SY_high   X(10-160 s15)   LY(0.5-5 s0.5)   SY(2-12 s0.5)     = 2310
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


WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_XtremeStop_AI.wsp"
SYMBOL     = "ETHUSDT HOT"
SIGNAL     = "SFJ_XtremeStop_Crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\eth_xtreme_hourly3_search")

IS_RANGE   = ("2022/01/01", "2026/01/01")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op (signal date does not restrict)

TARGET_NP  = 100_000.0

X_LO,  X_HI  = 1.0,   5000.0
LY_LO, LY_HI = 0.1,   50.0
SY_LO, SY_HI = 0.1,   50.0

# No crypto XtremeStop prior — seed at signal defaults X=150 LY=3.25 SY=3.25
SEED_X, SEED_LY, SEED_SY = 796.0, 5.85, 3.35
SEED_NP = 0.0

PREFIX = "ETHXTH3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_eth_xtreme_hourly3_{int(time.time())}.log"
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


def _cfg(name, x, ly, sy):
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    x  = _safe(x,  X_LO,  X_HI)
    ly = _safe(ly, LY_LO, LY_HI)
    sy = _safe(sy, SY_LO, SY_HI)

    combos = n_vals(x) * n_vals(ly) * n_vals(sy)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("X",  *x),
            ParamAxis("LY", *ly),
            ParamAxis("SY", *sy),
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


def champion(df, fb_x, fb_ly, fb_sy):
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  TARGET MET: X=%.4g LY=%.4g SY=%.4g  NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["X"]), float(best["LY"]), float(best["SY"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["X"]), float(best["LY"]), float(best["SY"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: X=%.4g LY=%.4g SY=%.4g  obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["X"]), float(best["LY"]), float(best["SY"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["X"]), float(best["LY"]), float(best["SY"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_x, fb_ly, fb_sy,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, x, ly, sy, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "X": x, "LY": ly, "SY": sy,
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
        "is_range": IS_RANGE,
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "Reversal-only stop-breakout vs C[X]: BUY C[X]*(1+LY*0.01) STOP; SHORT C[X]*(1-SY*0.01) STOP",
            "contract": "_Crypto1MUSD=Round(1000000/C,0)",
            "is_window": "2022/01/01-2026/01/01 (chart trimmed via set_instrument_data_range)",
            "priors":   "XtremeStop futures: NQ Hourly X=10 LY=1.48 SY=2.18; GC X=7 LY=0.67 SY=1.80; TXF X=63 LY=5.155 SY=5.79. Crypto default X=150 LY=3.25 SY=3.25",
            "r1_plan":  "global + short/mid/long X + high/low pct + asym LY/SY + adaptive zoom",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_eth_xtreme_hourly3.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_x, best_ly, best_sy = SEED_X, SEED_LY, SEED_SY
    best_np  = SEED_NP
    best_obj = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("==============================================================")
    log.info("  ETH Hourly SFJ_XtremeStop_Crypto NP>100K -- Round 3")
    log.info("  IS window %s ~ %s  (chart-trimmed)", IS_RANGE[0], IS_RANGE[1])
    log.info("  R2 champ X=796 LY=5.85 SY=3.35 NP=3682 (long-X asym 46tr; confirm LY/SY, X bounded)", )
    log.info("==============================================================")

    if not from_csv and conn is not None:
        mc.ensure_chart_ready(conn, _cfg("seed", (792, 800, 2), (5.4, 6.3, 0.25), (3.0, 3.7, 0.25)))
        log.info("Trimming chart data range to IS window %s ~ %s ...", *IS_RANGE)
        mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
        log.info("Chart trimmed. (verify rightmost bar ~2026/01, leftmost ~2022/01)")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_x, best_ly, best_sy
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_x, best_ly, best_sy,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-20s]  no valid data", attempt_num, name)
            return

        x, ly, sy, obj, np_, mdd, tr, met = champion(df, best_x, best_ly, best_sy)

        if np_ > best_np:
            best_x, best_ly, best_sy = x, ly, sy
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, x, ly, sy, obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-20s]  X=%.4g LY=%.4g SY=%.4g  obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, x, ly, sy, obj, np_, mdd, tr,
                 "TARGET" if met else ("%.0f/100K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f", best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    attempts_config = [
        ("01_SY_sweep",      (792, 800, 2),   (5.5, 6.25, 0.25),(2.0, 5.0, 0.1)),
        ("02_LY_sweep",      (792, 800, 2),   (3.0, 9.0, 0.25), (3.0, 3.7, 0.1)),
        ("03_fine_zoom",     (790, 802, 1),   (5.4, 6.4, 0.05), (3.0, 3.7, 0.05)),
        ("04_LYSY_grid",     (794, 798, 2),   (4.0, 8.0, 0.25), (2.5, 4.5, 0.25)),
        ("05_X_confirm_long",(800, 1000, 10), (4.0, 7.0, 0.5),  (2.5, 4.0, 0.5)),
        ("06_X_shorter",     (400, 800, 20),  (5.0, 7.0, 0.5),  (2.5, 4.0, 0.5)),
        ("07_X_bridge",      (700, 900, 10),  (5.0, 7.0, 0.25), (2.5, 4.0, 0.25)),
        ("08_champ_ultrafine",(794, 798, 1),  (5.6, 6.1, 0.05), (3.15, 3.55, 0.05)),
    ]

    for idx, (n, x_r, ly_r, sy_r) in enumerate(attempts_config, 1):
        _c = _cfg(n, x_r, ly_r, sy_r)
        log.info("A%02d  %s  %d combos", idx, n, _c.total_runs())
        if start_attempt <= idx:
            _update(run_or_load(n, _c, conn, from_csv), _c, n, idx, _c.total_runs())
        log.info("After A%02d -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)",
                 idx, best_np, best_x, best_ly, best_sy)

    # Adaptive zooms (progressively tighter around running best NP)
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        for r_x, s_x, r_p, s_p in [(40, 5, 1.5, 0.25), (20, 2, 1.0, 0.2), (10, 1, 0.6, 0.1)]:
            _x  = zoom(best_x,  r_x, s_x, X_LO, X_HI)
            _ly = zoom(best_ly, r_p, s_p, LY_LO, LY_HI)
            _sy = zoom(best_sy, r_p, s_p, SY_LO, SY_HI)
            _c  = _cfg(_n, _x, _ly, _sy)
            if _c.total_runs() <= 5000:
                break
        log.info("A09  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)", best_np, best_x, best_ly, best_sy)

    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        for r_x, s_x, r_p, s_p in [(20, 2, 0.8, 0.1), (10, 1, 0.5, 0.1), (6, 1, 0.3, 0.05)]:
            _x  = zoom(best_x,  r_x, s_x, X_LO, X_HI)
            _ly = zoom(best_ly, r_p, s_p, LY_LO, LY_HI)
            _sy = zoom(best_sy, r_p, s_p, SY_LO, SY_HI)
            _c  = _cfg(_n, _x, _ly, _sy)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)", best_np, best_x, best_ly, best_sy)

    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: X=%.4g LY=%.4g SY=%.4g  NP=%.0f",
             best_x, best_ly, best_sy, best_np)
    if start_attempt <= A:
        for r_x, s_x, r_p, s_p in [(10, 1, 0.4, 0.05), (5, 1, 0.25, 0.05), (3, 1, 0.15, 0.025)]:
            _x  = zoom(best_x,  r_x, s_x, X_LO, X_HI)
            _ly = zoom(best_ly, r_p, s_p, LY_LO, LY_HI)
            _sy = zoom(best_sy, r_p, s_p, SY_LO, SY_HI)
            _c  = _cfg(_n, _x, _ly, _sy)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  X%s LY%s SY%s  %d combos", _x, _ly, _sy, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 -- best NP=%.0f  (X=%.4g LY=%.4g SY=%.4g)", best_np, best_x, best_ly, best_sy)

    pct = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  ETH Hourly XtremeStop_Crypto Round-3 COMPLETE")
    log.info("  Champion: X=%.4g LY=%.4g SY=%.4g", best_x, best_ly, best_sy)
    log.info("  Best NP: %.0f USD  (%.1f%% of target %.0f)", best_np, pct, TARGET_NP)
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "MET" if target_met else "NOT MET (gap +%.0f)" % max(0, TARGET_NP - best_np))
    log.info("==============================================================")

    if not best_entry:
        best_entry = {
            "X": best_x, "LY": best_ly, "SY": best_sy,
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
        description="ETH Hourly SFJ_XtremeStop_Crypto NP>100K Round-3 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1-11)")
    ap.add_argument("--probe-trim", action="store_true",
                    help="just trim the chart to the IS window and read it back (fast fix test)")
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
        mc.ensure_chart_ready(conn, _cfg("seed", (792, 800, 2), (5.4, 6.3, 0.25), (3.0, 3.7, 0.25)))
        print(f"\nProbe: trimming chart to IS window {IS_RANGE[0]} ~ {IS_RANGE[1]}")
        mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
        time.sleep(1.0)
        rb_from, rb_to = mc.read_instrument_data_range(conn)
        print(f"  readback From={rb_from} To={rb_to}")
        return 0

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
