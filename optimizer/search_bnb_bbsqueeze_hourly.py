"""
search_bnb_bbsqueeze_hourly.py -- SFJ_BBSqueeze_crypto on BNBUSDT HOT Hourly, R1 (IS 2022/01-2026/01)

Strategy (Strategy/SFJ_BBSqueeze_crypto.txt): Bollinger-Band squeeze breakout (volatility-regime
filter) + ATR(14) chandelier trailing exit, long+short, STOP entries, _Crypto1MUSD.
  UpBB=BollingerBand(C,BBLen,BBmult); DnBB=BollingerBand(C,BBLen,-BBmult); BWidth=UpBB-DnBB
  squeeze = BWidth < Average(BWidth, SqueezeLen)
  while flat & squeeze: Buy next bar at UpBB STOP / SellShort next bar at DnBB STOP
  Exit  : ATR chandelier trail (ATRMult x ATR(14))

4 params: BBLen (int), BBmult (std-devs, fractional), SqueezeLen (int), ATRMult (fractional).
Target NP > 100,000 USD. Objective = NetProfit^2 / |MaxDrawdown|. <=5000 combos/attempt.
IS window 2022/01/01-2026/01/01 chart-trimmed. Reports BOTH NP-max and Obj-max champions.

Attempt schedule (12 attempts, 4D grids, <=5000):
  A01 global  BBLen(10-60) BBmult(1.5-3) SqueezeLen(20-100) ATRMult(2-5)  A02 default
  A03 bblen_fine  A04 bbmult_fine(BBmult 1-4 s0.25)  A05 sqzlen_scan(SqueezeLen 10-200)
  A06 mult_fine(ATRMult 1-8)  A07 short_bblen  A08 long_bblen  A09 global_bound
  A10-A12 adaptive zoom (4D) around the running Obj-max
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260622_SFJ_BBSqueeze_crypto_AI.wsp"
SYMBOL     = "BNBUSDT HOT"
SIGNAL     = "SFJ_BBSqueeze_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_bbsqueeze_hourly_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

TR_LO,   TR_HI   = 5.0,  200.0   # BBLen
RL_LO,   RL_HI   = 1.0,  6.0     # BBmult
TH_LO,   TH_HI   = 10.0, 300.0   # SqueezeLen
MULT_LO, MULT_HI = 0.5,  20.0

SEED_TR, SEED_RL, SEED_TH, SEED_MULT = 20.0, 2.0, 50.0, 3.0  # BBLen, BBmult, SqueezeLen, ATRMult

PREFIX = "BNBBBS_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_bbsqueeze_hourly_{int(time.time())}.log"
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


def _cfg(name, trend, rsilen, thresh, mult) -> StrategyConfig:
    def _safe(t, lo, hi, step_min):
        s, e, step = t
        if abs(s - e) < step_min * 0.5:
            return (max(lo, s - step), min(hi, s + step), step)
        return t
    trend  = _safe(trend,  TR_LO,   TR_HI,   1.0)
    rsilen = _safe(rsilen, RL_LO,   RL_HI,   0.25)  # BBmult frac
    thresh = _safe(thresh, TH_LO,   TH_HI,   1.0)
    mult   = _safe(mult,   MULT_LO, MULT_HI, 0.25)
    combos = n_vals(trend) * n_vals(rsilen) * n_vals(thresh) * n_vals(mult)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[ParamAxis("BBLen",      *trend),
                ParamAxis("BBmult",     *rsilen),
                ParamAxis("SqueezeLen", *thresh),
                ParamAxis("ATRMult",   *mult)],
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
    return {"BBLen": float(b["BBLen"]), "BBmult": float(b["BBmult"]),
            "SqueezeLen": float(b["SqueezeLen"]), "ATRMult": float(b["ATRMult"]),
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
            "logic": "Bollinger squeeze breakout: UpBB/DnBB=BollingerBand(C,BBLen,+/-BBmult); squeeze=BWidth<Average(BWidth,SqueezeLen); while flat&squeeze Buy at UpBB STOP / SellShort at DnBB STOP; ATR(14) chandelier trailing exit; long+short",
            "contract": "_Crypto1MUSD = Round(1000000/C, 0) ~ $1M notional/trade",
            "params": "BBLen (int), BBmult (std-devs frac), SqueezeLen (int), ATRMult (trail frac); defaults 20/2.0/50/3.0",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_bbsqueeze_hourly.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sT, sR, sH, sM = SEED_TR, SEED_RL, SEED_TH, SEED_MULT
    obj_best: Dict = {}
    np_best: Dict = {}
    target_met = False
    attempt_log: List[dict] = []

    log.info("==============================================================")
    log.info("  SFJ_BBSqueeze_crypto on BNBUSDT HOT Hourly -- Round 1 (IS 2022/01-2026/01)")
    log.info("  Params: BBLen, BBmult, SqueezeLen, ATRMult.  Obj=NP^2/|MDD|.  Target %.0f", TARGET_NP)
    log.info("  Default: 100/14/40/3.0.  Run start: %s", datetime.now().isoformat())
    log.info("==============================================================")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (80, 120, 20), (12, 16, 2),
                                                 (35, 45, 5), (2.5, 3.5, 0.5)))
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
        nonlocal sT, sR, sH, sM, obj_best, np_best, target_met
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
            sT, sR, sH, sM = om["BBLen"], om["BBmult"], om["SqueezeLen"], om["ATRMult"]
        if not np_best or nm["net_profit"] > np_best.get("net_profit", -1):
            np_best = {**nm, "attempt": attempt_num, "name": name}
            if nm["net_profit"] > TARGET_NP:
                target_met = True
        log.info("  [A%02d %-16s] obj_max BBLen=%.4g BBm=%.4g Sqz=%.4g M=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d (%.1fs)",
                 attempt_num, name, om["BBLen"], om["BBmult"], om["SqueezeLen"], om["ATRMult"],
                 om["net_profit"], om["max_drawdown"], om["objective"], om["total_trades"], elapsed)
        log.info("       Best Obj=%.0f | NP-max=%.0f gap_to_target=%.0f",
                 obj_best.get("objective", 0), np_best.get("net_profit", 0),
                 max(0, TARGET_NP - np_best.get("net_profit", 0)))
        save_json(obj_best, np_best, attempt_log, target_met)

    def _do(A, name, cfg):
        log.info("A%02d  %s  %d combos", A, name, cfg.total_runs())
        if start_attempt <= A:
            t = time.time()
            _update(run_or_load(name, cfg, conn, from_csv), cfg, name, A, cfg.total_runs(), t)

    # static attempts (4D)
    _do(1, "01_global",      _cfg("01_global",      (10,60,10),(1.5,3,0.5),  (20,100,20),(2,5,1)))
    _do(2, "02_default",     _cfg("02_default",     (15,30,5), (1.5,2.5,0.5),(30,80,10), (2,4,0.5)))
    _do(3, "03_bblen_fine",  _cfg("03_bblen_fine",  (5,50,5),  (1.5,2.5,0.5),(40,60,10), (2.5,4,0.5)))
    _do(4, "04_bbmult_fine", _cfg("04_bbmult_fine", (15,30,5), (1,4,0.25),   (30,70,20), (2.5,4,0.5)))
    _do(5, "05_sqzlen_scan", _cfg("05_sqzlen_scan", (15,30,5), (1.5,2.5,0.5),(10,200,10),(2.5,4,0.5)))
    _do(6, "06_mult_fine",   _cfg("06_mult_fine",   (15,30,5), (1.5,2.5,0.5),(40,60,20), (1,8,0.25)))
    _do(7, "07_short_bblen", _cfg("07_short_bblen", (5,25,2),  (1.5,2.5,0.5),(30,70,20), (2,5,0.5)))
    _do(8, "08_long_bblen",  _cfg("08_long_bblen",  (40,120,10),(1.5,3,0.5), (40,100,30),(2,5,0.5)))
    _do(9, "09_global_bound",_cfg("09_global_bound",(5,150,15),(1,4,0.5),    (20,150,40),(2,6,1)))

    # adaptive zooms around the running Obj-max
    for A, nm_, (rT_mul, rT_min, rR_mul, rR_min, rH_mul, rH_min, rM_mul, rM_min, nT, nR, nH, nM) in [
        (10, "10_adaptive_zoom1", (0.40, 20, 0.40, 5, 0.30, 8, 0.40, 1.5, 9, 7, 9, 9)),
        (11, "11_adaptive_zoom2", (0.20, 10, 0.22, 3, 0.18, 4, 0.22, 0.75, 9, 7, 9, 9)),
        (12, "12_adaptive_zoom3", (0.10, 5,  0.12, 2, 0.10, 2, 0.12, 0.5, 7, 7, 7, 7)),
    ]:
        log.info("A%02d  %s -- center T=%.4g RL=%.4g Th=%.4g M=%.4g Obj=%.0f",
                 A, nm_, sT, sR, sH, sM, obj_best.get("objective", 0))
        if start_attempt <= A:
            _T = zoom_fixed(sT, max(rT_min, sT * rT_mul), nT, 1.0,  TR_LO,   TR_HI)
            _R = zoom_fixed(sR, max(rR_min, sR * rR_mul), nR, 0.25, RL_LO,   RL_HI)
            _H = zoom_fixed(sH, max(rH_min, sH * rH_mul), nH, 1.0,  TH_LO,   TH_HI)
            _M = zoom_fixed(sM, max(rM_min, sM * rM_mul), nM, 0.25, MULT_LO, MULT_HI)
            _c = _cfg(nm_, _T, _R, _H, _M)
            log.info("A%02d  T%s RL%s Th%s M%s  %d combos", A, _T, _R, _H, _M, _c.total_runs())
            t = time.time()
            _update(run_or_load(nm_, _c, conn, from_csv), _c, nm_, A, _c.total_runs(), t)

    log.info("==============================================================")
    log.info("  SFJ_BBSqueeze_crypto BNBUSDT Hourly Round-1 COMPLETE (%.1f min)", (time.time()-_RUN_T0)/60)
    if obj_best:
        log.info("  Obj-max (your criterion): T=%.4g RL=%.4g Th=%.4g M=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
                 obj_best["BBLen"], obj_best["BBmult"], obj_best["SqueezeLen"], obj_best["ATRMult"],
                 obj_best["net_profit"], obj_best["max_drawdown"], obj_best["objective"], obj_best["total_trades"])
    if np_best:
        log.info("  NP-max: T=%.4g RL=%.4g Th=%.4g M=%.4g NP=%.0f Obj=%.0f",
                 np_best["BBLen"], np_best["BBmult"], np_best["SqueezeLen"], np_best["ATRMult"],
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
    ap = argparse.ArgumentParser(description="SFJ_BBSqueeze_crypto BNBUSDT HOT Hourly R1 search")
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
