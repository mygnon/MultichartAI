"""
search_bnb_qpatrsi_hourly.py — SFJ_TurtleChannel_crypto on BNBUSDT HOT Hourly, Round 1 (IS 2022/01-2026/01)

Strategy (Strategy/SFJ_TurtleChannel_crypto.txt): Turtle dual Donchian-channel breakout,
long+short, STOP entries, channel trailing exit, _Crypto1MUSD.
  HHe=Highest(High,EntryLen)[1]; LLe=Lowest(Low,EntryLen)[1] (entry channel)
  HHx=Highest(High,ExitLen)[1];  LLx=Lowest(Low,ExitLen)[1]  (exit channel)
  Long  : flat -> Buy at HHe STOP ; exit Sell at LLx STOP
  Short : flat -> SellShort at LLe STOP ; exit BuyToCover at HHx STOP
2 params: EntryLen (breakout channel int), ExitLen (trailing-exit channel int, < EntryLen).
Target NP>100,000. Objective = NetProfit^2/|MaxDrawdown|. <=5000 combos/attempt. NP-max + Obj-max.
Attempt schedule (8 static 2D + 3 zoom): EntryLen x ExitLen grids.
  A09-A11 adaptive zoom (2D) around the running Obj-max
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260622_SFJ_TurtleChannel_crypto_AI.wsp"
SYMBOL     = "BNBUSDT HOT"
SIGNAL     = "SFJ_TurtleChannel_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_turtlechannel_hourly_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO, LEN_HI = 5.0, 300.0    # EntryLen
GAP_LO, GAP_HI = 2.0, 100.0    # ExitLen

SEED_LEN = 20.0
SEED_GAP = 10.0
SEED_NP  = 0.0

PREFIX = "BNBTUR_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_turtlechannel_hourly_{int(time.time())}.log"
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


def _cfg(name, length, gap) -> StrategyConfig:
    def _safe(t, lo, hi, step_min):
        s, e, step = t
        if abs(s - e) < step_min * 0.5:
            return (max(lo, s - step), min(hi, s + step), step)
        return t
    length = _safe(length, LEN_LO, LEN_HI, 1.0)
    gap    = _safe(gap,    GAP_LO, GAP_HI, 1.0)
    combos = n_vals(length) * n_vals(gap)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[ParamAxis("EntryLen", *length), ParamAxis("ExitLen", *gap)],
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
    return {"EntryLen": float(b["EntryLen"]), "ExitLen": float(b["ExitLen"]),
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
            "logic": "Turtle dual-channel: Buy at Highest(High,EntryLen)[1] STOP / SellShort at Lowest(Low,EntryLen)[1] STOP (flat-only); exit on opposite ExitLen-bar channel (Sell at LLx / BuyToCover at HHx); long+short",
            "contract": "_Crypto1MUSD = Round(1000000/C, 0) ~ $1M notional/trade",
            "params": "EntryLen (breakout channel int), ExitLen (exit channel int); defaults 20/10",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_turtlechannel_hourly.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_len, seed_gap = SEED_LEN, SEED_GAP
    obj_best: Dict = {}
    np_best: Dict = {}
    target_met = False
    attempt_log: List[dict] = []

    log.info("==============================================================")
    log.info("  SFJ_TurtleChannel_crypto on BNBUSDT HOT Hourly -- Round 1 (IS 2022/01-2026/01 chart-trimmed)")
    log.info("  Params: EntryLen, ExitLen.  Obj=NP^2/|MDD|.  Target NP %.0f USD", TARGET_NP)
    log.info("  Run start: %s", datetime.now().isoformat())
    log.info("==============================================================")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (28, 32, 2), (30, 34, 2)))
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
        nonlocal seed_len, seed_gap, obj_best, np_best, target_met
        elapsed = time.time() - t_attempt
        om = _obj_max_row(df) if (df is not None and not df.empty and _validate_df(df, cfg)) else None
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
            seed_len, seed_gap = om["EntryLen"], om["ExitLen"]
        if not np_best or nm["net_profit"] > np_best.get("net_profit", -1):
            np_best = {**nm, "attempt": attempt_num, "name": name}
            if nm["net_profit"] > TARGET_NP:
                target_met = True
        log.info("  [A%02d %-16s] obj_max EntryLen=%.4g ExitLen=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d (%.1fs)",
                 attempt_num, name, om["EntryLen"], om["ExitLen"], om["net_profit"], om["max_drawdown"],
                 om["objective"], om["total_trades"], elapsed)
        log.info("       Best Obj=%.0f (Len=%.4g Gap=%.4g) | NP-max=%.0f gap_to_target=%.0f",
                 obj_best.get("objective", 0), seed_len, seed_gap, np_best.get("net_profit", 0),
                 max(0, TARGET_NP - np_best.get("net_profit", 0)))
        save_json(obj_best, np_best, attempt_log, target_met)

    def _do(A, name, cfg):
        log.info("A%02d  %s  %d combos", A, name, cfg.total_runs())
        if start_attempt <= A:
            t = time.time()
            _update(run_or_load(name, cfg, conn, from_csv), cfg, name, A, cfg.total_runs(), t)

    # static attempts
    _do(1, "01_global",      _cfg("01_global",      (10,100,5),  (2,40,2)))
    _do(2, "02_classic",     _cfg("02_classic",     (10,60,2),   (3,25,1)))
    _do(3, "03_short_entry", _cfg("03_short_entry", (5,40,1),    (2,20,1)))
    _do(4, "04_long_entry",  _cfg("04_long_entry",  (40,200,5),  (5,50,3)))
    _do(5, "05_tight_exit",  _cfg("05_tight_exit",  (15,60,3),   (1,15,1)))
    _do(6, "06_wide_exit",   _cfg("06_wide_exit",   (20,100,5),  (10,60,5)))
    _do(7, "07_vlong_entry", _cfg("07_vlong_entry", (100,300,10),(5,60,5)))
    _do(8, "08_dense_low",   _cfg("08_dense_low",   (5,50,1),    (2,12,1)))

    # adaptive zooms around the running Obj-max
    for A, nm_, (rl_mul, rl_min, rg_mul, rg_min, nL, nG) in [
        (9,  "09_adaptive_zoom1", (0.30, 12.0, 0.30, 6.0, 41, 25)),
        (10, "10_adaptive_zoom2", (0.15, 6.0,  0.18, 3.0, 31, 21)),
        (11, "11_adaptive_zoom3", (0.07, 3.0,  0.10, 2.0, 21, 17)),
    ]:
        log.info("A%02d  %s -- center Len=%.4g Gap=%.4g Obj=%.0f",
                 A, nm_, seed_len, seed_gap, obj_best.get("objective", 0))
        if start_attempt <= A:
            _len = zoom_fixed(seed_len, max(rl_min, seed_len * rl_mul), nL, 1.0, LEN_LO, LEN_HI)
            _gap = zoom_fixed(seed_gap, max(rg_min, seed_gap * rg_mul), nG, 1.0, GAP_LO, GAP_HI)
            _c = _cfg(nm_, _len, _gap)
            log.info("A%02d  Len%s Gap%s  %d combos", A, _len, _gap, _c.total_runs())
            t = time.time()
            _update(run_or_load(nm_, _c, conn, from_csv), _c, nm_, A, _c.total_runs(), t)

    log.info("==============================================================")
    log.info("  SFJ_TurtleChannel_crypto BNBUSDT Hourly Round-1 COMPLETE (%.1f min)", (time.time()-_RUN_T0)/60)
    if obj_best:
        log.info("  Obj-max (your criterion): EntryLen=%.4g ExitLen=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
                 obj_best["EntryLen"], obj_best["ExitLen"], obj_best["net_profit"],
                 obj_best["max_drawdown"], obj_best["objective"], obj_best["total_trades"])
    if np_best:
        log.info("  NP-max: EntryLen=%.4g ExitLen=%.4g NP=%.0f Obj=%.0f",
                 np_best["EntryLen"], np_best["ExitLen"], np_best["net_profit"], np_best["objective"])
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
    ap = argparse.ArgumentParser(description="SFJ_TurtleChannel_crypto BNBUSDT HOT Hourly R1 search")
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
