"""
search_bnb_macd_hourly.py -- SFJ_MACD_Strategy03_crypto on BNBUSDT HOT Hourly, Round 1 (IS 2022/01-2026/01)

Strategy (Knowledge/SFJ_MACD_Strategy03_crypto.docx): MACD momentum, long+short, MARKET entries/exits.
  input: FastLength(12), SlowLength(26), MACDLength(9)
  var0 = MACD(Close, FastLength, SlowLength)      (MACD line)
  var1 = XAverage(var0, MACDLength)               (signal line)
  var2 = var0 - var1                              (histogram)
  ENTRY: var0 crosses OVER 0 -> BUY market ; var0 crosses UNDER 0 -> SHORT market
  EXIT : var2 crosses OVER 0 -> BuyToCover (cover short) ; var2 crosses UNDER 0 -> Sell (exit long)
  Contract: _Crypto1MUSD = Round(1,000,000/C, 0) ~ $1M notional per trade.

3 params (FastLength, SlowLength, MACDLength), all integers. MACD valid when Fast < Slow.
Target NP > 100,000 USD. Objective = NetProfit^2 / |MaxDrawdown| (bigger=better). <=5000 combos/attempt.
IS window 2022/01/01-2026/01/01 chart-trimmed (MC64 ignores signal Begin-date; only the loaded chart
data range restricts the optimization). Reports BOTH NP-max and Obj-max champions.

Attempt schedule (12 attempts, 3D grids, <=5000):
  A01 global_wide   Fast(4-40 s4)  x Slow(10-100 s10) x MACD(3-21 s3)  = 10x10x7 = 700
  A02 default_zone  Fast(6-20 s2)  x Slow(18-40 s2)   x MACD(5-15 s2)  = 8x12x6  = 576
  A03 fast_fine     Fast(2-30 s1)  x Slow(20-40 s5)   x MACD(6-12 s2)  = 29x5x4  = 580
  A04 slow_fine     Fast(8-16 s2)  x Slow(15-60 s3)   x MACD(4-16 s2)  = 5x16x7  = 560
  A05 macd_fine     Fast(8-16 s2)  x Slow(20-40 s4)   x MACD(2-30 s2)  = 5x6x15  = 450
  A06 short_fast    Fast(2-12 s1)  x Slow(10-40 s2)   x MACD(3-15 s3)  = 11x16x5 = 880
  A07 long_slow     Fast(10-30 s2) x Slow(40-120 s8)  x MACD(5-20 s3)  = 11x11x6 = 726
  A08 wide_macd     Fast(6-24 s3)  x Slow(20-60 s5)   x MACD(2-38 s4)  = 7x9x10  = 630
  A09 global_bound  Fast(2-62 s6)  x Slow(10-150 s14) x MACD(2-30 s4)  = 11x11x8 = 968
  A10-A12 adaptive zoom (3D) around the running Obj-max
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_MACD03_AI.wsp"
SYMBOL     = "BNBUSDT HOT"
SIGNAL     = "SFJ_MACD_Strategy03_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_macd_hourly_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

FAST_LO, FAST_HI = 1.0,  200.0
SLOW_LO, SLOW_HI = 2.0,  400.0
MACD_LO, MACD_HI = 1.0,  100.0

SEED_FAST, SEED_SLOW, SEED_MACD = 12.0, 26.0, 9.0

PREFIX = "BNBMACDH_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_macd_hourly_{int(time.time())}.log"
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
_RUN_T0 = time.time()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _cfg(name, fast, slow, macd) -> StrategyConfig:
    def _safe(t, lo, hi, step_min):
        s, e, step = t
        if abs(s - e) < step_min * 0.5:
            return (max(lo, s - step), min(hi, s + step), step)
        return t
    fast = _safe(fast, FAST_LO, FAST_HI, 1.0)
    slow = _safe(slow, SLOW_LO, SLOW_HI, 1.0)
    macd = _safe(macd, MACD_LO, MACD_HI, 1.0)
    combos = n_vals(fast) * n_vals(slow) * n_vals(macd)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[ParamAxis("FastLength", *fast),
                ParamAxis("SlowLength", *slow),
                ParamAxis("MACDLength", *macd)],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name) -> Path:
    return OUTPUT_DIR / f"{PREFIX}{name}_raw.csv"


_user32 = ctypes.windll.user32
_STRAY_KW = ["Optimization", "最佳化", "優化", "Optimis"]


def _cleanup_stray_windows():
    try:
        mc._close_optimization_report()
    except Exception:
        pass
    victims = []

    def _cb(hwnd, _):
        try:
            n = _user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            _user32.GetWindowTextW(hwnd, buf, n + 1)
            t = buf.value
            if "MultiCharts" in t:
                return True
            if _user32.IsWindowVisible(hwnd) and any(k.lower() in t.lower() for k in _STRAY_KW):
                victims.append(hwnd)
        except Exception:
            pass
        return True

    try:
        _user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)(_cb), 0)
    except Exception:
        pass
    for hwnd in victims:
        try:
            _user32.PostMessageW(hwnd, 0x0010, 0, 0)
            time.sleep(0.4)
        except Exception:
            pass
    if victims:
        time.sleep(0.6)


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
    for attempt in (1, 2):
        _cleanup_stray_windows()
        try:
            mc.ensure_chart_ready(conn, cfg)
        except Exception as e:
            log.warning("  ensure_chart_ready: %s", e)
        t0 = time.time()
        try:
            raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
            log.info("  Done %.1f min -- %s (attempt %d)", (time.time() - t0) / 60,
                     Path(raw_csv).name, attempt)
            return mc.load_results_csv(raw_csv, cfg)
        except Exception as e:
            log.warning("  attempt %d FAILED: %s", attempt, e)
            if attempt == 2:
                log.error("  %s: giving up after 2 attempts", name)
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


def _row(df, idx):
    b = df.loc[idx]
    return {"FastLength": float(b["FastLength"]), "SlowLength": float(b["SlowLength"]),
            "MACDLength": float(b["MACDLength"]),
            "objective": float(b["Objective"]), "net_profit": float(b["NetProfit"]),
            "max_drawdown": float(b["MaxDrawdown"]), "total_trades": int(b["TotalTrades"])}


def _np_max_row(df):
    df = df.copy(); df["Objective"] = plateau_mod.compute_objective(df)
    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        return None
    return _row(pos, pos["NetProfit"].idxmax())


def _obj_max_row(df):
    df = df.copy(); df["Objective"] = plateau_mod.compute_objective(df)
    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        return None
    return _row(pos, pos["Objective"].idxmax())


def save_json(obj_best, np_best, attempt_log, target_met):
    payload = {
        "round": 1, "strategy": SIGNAL, "symbol": SYMBOL, "timeframe": "Hourly (60 min)",
        "is_window": "2022/01/01 - 2026/01/01 (chart-trimmed)",
        "target_np": TARGET_NP, "target_met": target_met,
        "objective_def": "NetProfit^2 / |MaxDrawdown|",
        "notes": {
            "logic": "MACD(C,Fast,Slow) crosses 0 -> market entry (over=BUY, under=SHORT); histogram (MACD-signal) crosses 0 -> exit",
            "contract": "_Crypto1MUSD = Round(1000000/C, 0) ~ $1M notional/trade",
            "params": "FastLength, SlowLength, MACDLength (all int); defaults 12/26/9; valid when Fast<Slow",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_macd_hourly.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_fast, seed_slow, seed_macd = SEED_FAST, SEED_SLOW, SEED_MACD
    obj_best: Dict = {}
    np_best: Dict = {}
    target_met = False
    attempt_log: List[dict] = []

    log.info("==============================================================")
    log.info("  SFJ_MACD_Strategy03_crypto on BNBUSDT HOT Hourly -- Round 1 (IS 2022/01-2026/01)")
    log.info("  Params: FastLength, SlowLength, MACDLength.  Obj=NP^2/|MDD|.  Target NP %.0f USD", TARGET_NP)
    log.info("  Default: 12/26/9.  Run start: %s", datetime.now().isoformat())
    log.info("==============================================================")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (10, 14, 2), (24, 28, 2), (7, 11, 2)))
                mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
                ok = True
                break
            except Exception as e:
                log.warning("  chart-trim attempt %d failed: %s", _try + 1, e)
                try:
                    mc._close_optimization_report()
                except Exception:
                    pass
                time.sleep(1.0)
        if not ok:
            log.error("Chart-trim FAILED after 3 tries -- aborting.")
            return 1
        log.info("Chart trimmed (verify leftmost ~2022/01, rightmost ~2026/01).")

    def _update(df, cfg, name, attempt_num, combos, t_attempt):
        nonlocal seed_fast, seed_slow, seed_macd, obj_best, np_best, target_met
        elapsed = time.time() - t_attempt
        ok = df is not None and not df.empty and _validate_df(df, cfg)
        om = _obj_max_row(df) if ok else None
        nm = _np_max_row(df) if om else None
        entry = {"attempt": attempt_num, "name": name, "combos": combos,
                 "rows": len(df) if df is not None else 0,
                 "obj_max": om, "np_max": nm,
                 "elapsed_sec": round(elapsed, 1), "timestamp": datetime.now().isoformat()}
        attempt_log.append(entry)
        if om is None:
            log.info("  [A%02d %-16s] no valid data", attempt_num, name)
            save_json(obj_best, np_best, attempt_log, target_met)
            return
        if not obj_best or om["objective"] > obj_best.get("objective", -1):
            obj_best = {**om, "attempt": attempt_num, "name": name}
            seed_fast, seed_slow, seed_macd = om["FastLength"], om["SlowLength"], om["MACDLength"]
        if not np_best or nm["net_profit"] > np_best.get("net_profit", -1):
            np_best = {**nm, "attempt": attempt_num, "name": name}
            if nm["net_profit"] > TARGET_NP:
                target_met = True
        log.info("  [A%02d %-16s] obj_max F=%.4g S=%.4g M=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d (%.1fs)",
                 attempt_num, name, om["FastLength"], om["SlowLength"], om["MACDLength"],
                 om["net_profit"], om["max_drawdown"], om["objective"], om["total_trades"], elapsed)
        log.info("       Best Obj=%.0f (F=%.4g S=%.4g M=%.4g) | NP-max=%.0f gap_to_target=%.0f",
                 obj_best.get("objective", 0), seed_fast, seed_slow, seed_macd,
                 np_best.get("net_profit", 0), max(0, TARGET_NP - np_best.get("net_profit", 0)))
        save_json(obj_best, np_best, attempt_log, target_met)

    def _do(A, name, cfg):
        log.info("A%02d  %s  %d combos", A, name, cfg.total_runs())
        if start_attempt <= A:
            t = time.time()
            _update(run_or_load(name, cfg, conn, from_csv), cfg, name, A, cfg.total_runs(), t)

    # static attempts (3D)
    _do(1, "01_global_wide",  _cfg("01_global_wide",  (4,40,4),  (10,100,10), (3,21,3)))
    _do(2, "02_default_zone", _cfg("02_default_zone", (6,20,2),  (18,40,2),   (5,15,2)))
    _do(3, "03_fast_fine",    _cfg("03_fast_fine",    (2,30,1),  (20,40,5),   (6,12,2)))
    _do(4, "04_slow_fine",    _cfg("04_slow_fine",    (8,16,2),  (15,60,3),   (4,16,2)))
    _do(5, "05_macd_fine",    _cfg("05_macd_fine",    (8,16,2),  (20,40,4),   (2,30,2)))
    _do(6, "06_short_fast",   _cfg("06_short_fast",   (2,12,1),  (10,40,2),   (3,15,3)))
    _do(7, "07_long_slow",    _cfg("07_long_slow",    (10,30,2), (40,120,8),  (5,20,3)))
    _do(8, "08_wide_macd",    _cfg("08_wide_macd",    (6,24,3),  (20,60,5),   (2,38,4)))
    _do(9, "09_global_bound", _cfg("09_global_bound", (2,62,6),  (10,150,14), (2,30,4)))

    # adaptive zooms around the running Obj-max
    for A, nm_, (rf_mul, rf_min, rs_mul, rs_min, rm_mul, rm_min, nF, nS, nM) in [
        (10, "10_adaptive_zoom1", (0.40, 8.0, 0.40, 16.0, 0.40, 6.0, 13, 13, 11)),
        (11, "11_adaptive_zoom2", (0.20, 4.0, 0.20, 8.0,  0.22, 3.0, 11, 11, 9)),
        (12, "12_adaptive_zoom3", (0.10, 2.0, 0.10, 4.0,  0.12, 2.0, 9,  9,  7)),
    ]:
        log.info("A%02d  %s -- center F=%.4g S=%.4g M=%.4g Obj=%.0f",
                 A, nm_, seed_fast, seed_slow, seed_macd, obj_best.get("objective", 0))
        if start_attempt <= A:
            _fast = zoom_fixed(seed_fast, max(rf_min, seed_fast * rf_mul), nF, 1.0, FAST_LO, FAST_HI)
            _slow = zoom_fixed(seed_slow, max(rs_min, seed_slow * rs_mul), nS, 1.0, SLOW_LO, SLOW_HI)
            _macd = zoom_fixed(seed_macd, max(rm_min, seed_macd * rm_mul), nM, 1.0, MACD_LO, MACD_HI)
            _c = _cfg(nm_, _fast, _slow, _macd)
            log.info("A%02d  F%s S%s M%s  %d combos", A, _fast, _slow, _macd, _c.total_runs())
            t = time.time()
            _update(run_or_load(nm_, _c, conn, from_csv), _c, nm_, A, _c.total_runs(), t)

    log.info("==============================================================")
    log.info("  SFJ_MACD_Strategy03_crypto BNBUSDT Hourly Round-1 COMPLETE (%.1f min)", (time.time()-_RUN_T0)/60)
    if obj_best:
        log.info("  Obj-max (your criterion): F=%.4g S=%.4g M=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
                 obj_best["FastLength"], obj_best["SlowLength"], obj_best["MACDLength"],
                 obj_best["net_profit"], obj_best["max_drawdown"], obj_best["objective"], obj_best["total_trades"])
    if np_best:
        log.info("  NP-max: F=%.4g S=%.4g M=%.4g NP=%.0f Obj=%.0f",
                 np_best["FastLength"], np_best["SlowLength"], np_best["MACDLength"],
                 np_best["net_profit"], np_best["objective"])
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "MET" if target_met else "NOT MET (best NP %.0f)" % np_best.get("net_profit", 0))
    log.info("==============================================================")
    out = save_json(obj_best, np_best, attempt_log, target_met)
    print(f"\nDone -- results at: {out}")
    return 0


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
    ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Administrator manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="SFJ_MACD_Strategy03_crypto BNBUSDT HOT Hourly R1 search")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--attempt", type=int, default=1, metavar="N")
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
