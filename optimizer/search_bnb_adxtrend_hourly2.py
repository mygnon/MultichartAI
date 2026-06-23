"""
search_bnb_adxtrend_hourly.py -- SFJ_ADXtrend_crypto on BNBUSDT HOT Hourly, Round 2 integer-grid + boundary confirm

Strategy (Strategy/SFJ_ADXtrend_crypto.txt): DMI directional crossover gated by an ADX
trend-strength filter + ATR(14) chandelier trailing exit, long+short, MARKET entries, _Crypto1MUSD.
  diPlus=DMIPlus(DMILen); diMinus=DMIMinus(DMILen); adxv=ADX(DMILen)
  Long  : flat & adxv>ADXThresh & diPlus cross over diMinus  -> Buy market
  Short : flat & adxv>ADXThresh & diMinus cross over diPlus    -> SellShort market
  Exit  : ATR chandelier trail (ATRMult x ATR(14))
3 params: DMILen (DMI/ADX period int), ADXThresh (strength gate int), ATRMult (trail, fractional).
Target NP > 100,000 USD. Objective = NetProfit^2 / |MaxDrawdown| (bigger=better). <=5000 combos/attempt.
IS window 2022/01/01-2026/01/01 chart-trimmed (MC64 ignores signal Begin-date; only the loaded chart
data range restricts the optimization). Reports BOTH NP-max and Obj-max champions.

Attempt schedule (12 attempts, 3D grids, <=5000):
  A01 global_wide  Length(5-100 s5)  x ATRL(7-28 s7)  x Mult(1.5-6 s0.5)  = 20x4x10 = 800
  A02 default_zone Length(10-40 s2)  x ATRL(10-20 s2) x Mult(2-4 s0.5)    = 16x6x5  = 480
  A03 len_fine     Length(5-60 s2)   x ATRL(10-18 s4) x Mult(2.5-4.5 s0.5)= 28x3x5  = 420
  A04 atrl_fine    Length(15-35 s5)  x ATRL(5-40 s2)  x Mult(2.5-4 s0.5)  = 5x18x4  = 360
  A05 mult_fine    Length(15-35 s5)  x ATRL(10-20 s3) x Mult(1-8 s0.25)   = 5x4x29  = 580
  A06 short_len    Length(3-25 s1)   x ATRL(8-20 s4)  x Mult(2-5 s0.5)    = 23x4x7  = 644
  A07 long_len     Length(40-150 s5) x ATRL(10-25 s5) x Mult(2-5 s0.5)    = 23x4x7  = 644
  A08 wide_mult    Length(10-50 s5)  x ATRL(8-24 s4)  x Mult(1-10 s0.5)   = 9x5x19  = 855
  A09 global_bound Length(5-200 s15) x ATRL(5-50 s9)  x Mult(1-9 s1)      = 14x6x9  = 756
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260622_SFJ_ADXtrend_crypto_AI.wsp"
SYMBOL     = "BNBUSDT HOT"
SIGNAL     = "SFJ_ADXtrend_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_adxtrend_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO,  LEN_HI  = 2.0,  100.0   # DMILen
ATRL_LO, ATRL_HI = 5.0,  60.0   # ADXThresh (integer)
MULT_LO, MULT_HI = 0.5,  20.0

SEED_LEN, SEED_ATRL, SEED_MULT = 12.0, 26.0, 7.8  # DMILen, ADXThresh, ATRMult (R1 champion)

PREFIX = "BNBADX2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_adxtrend_hourly2_{int(time.time())}.log"
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


def zoom_int(center, radius, n_target, lo, hi):
    c = round(center)
    lo_val = max(round(lo), c - round(radius)); hi_val = min(round(hi), c + round(radius))
    if lo_val >= hi_val:
        return (max(round(lo), c - 1), min(round(hi), c + 1), 1.0)
    step = max(1, round((hi_val - lo_val) / max(1, n_target - 1)))
    return (float(lo_val), float(hi_val), float(step))


def n_vals(t):
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name, length, atrl, mult) -> StrategyConfig:
    def _safe(t, lo, hi, step_min):
        s, e, step = t
        if abs(s - e) < step_min * 0.5:
            return (max(lo, s - step), min(hi, s + step), step)
        return t
    length = _safe(length, LEN_LO,  LEN_HI,  1.0)
    atrl   = _safe(atrl,   ATRL_LO, ATRL_HI, 1.0)   # ADXThresh integer
    mult   = _safe(mult,   MULT_LO, MULT_HI, 0.25)
    combos = n_vals(length) * n_vals(atrl) * n_vals(mult)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[ParamAxis("DMILen",    *length),
                ParamAxis("ADXThresh", *atrl),
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
    return {"DMILen": float(b["DMILen"]), "ADXThresh": float(b["ADXThresh"]),
            "ATRMult": float(b["ATRMult"]),
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
        "round": 2, "strategy": SIGNAL, "symbol": SYMBOL, "timeframe": "Hourly (60 min)",
        "is_window": "2022/01/01 - 2026/01/01 (chart-trimmed)",
        "target_np": TARGET_NP, "target_met": target_met,
        "objective_def": "NetProfit^2 / |MaxDrawdown|",
        "notes": {
            "logic": "ADX-gated DMI crossover: long flat & ADX(DMILen)>ADXThresh & DI+ cross over DI-; short mirror; ATR(14) chandelier trailing exit; flat-only; long+short",
            "contract": "_Crypto1MUSD = Round(1000000/C, 0) ~ $1M notional/trade",
            "params": "DMILen (DMI/ADX period int), ADXThresh (strength gate int), ATRMult (trail frac); defaults 14/25/3.0",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_adxtrend_hourly2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_len, seed_atrl, seed_mult = SEED_LEN, SEED_ATRL, SEED_MULT
    obj_best: Dict = {}
    np_best: Dict = {}
    target_met = False
    attempt_log: List[dict] = []

    log.info("==============================================================")
    log.info("  SFJ_ADXtrend_crypto on BNBUSDT HOT Hourly -- Round 2 integer-grid + boundary confirm (IS 2022/01-2026/01)")
    log.info("  Params: DMILen, ADXThresh, ATRMult.  Obj=NP^2/|MDD|.  Target NP %.0f USD", TARGET_NP)
    log.info("  Default: 20/14/3.0.  Run start: %s", datetime.now().isoformat())
    log.info("==============================================================")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (10, 14, 2), (24, 28, 2), (7, 8.5, 0.5)))
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
        nonlocal seed_len, seed_atrl, seed_mult, obj_best, np_best, target_met
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
            seed_len, seed_atrl, seed_mult = om["DMILen"], om["ADXThresh"], om["ATRMult"]
        if not np_best or nm["net_profit"] > np_best.get("net_profit", -1):
            np_best = {**nm, "attempt": attempt_num, "name": name}
            if nm["net_profit"] > TARGET_NP:
                target_met = True
        log.info("  [A%02d %-16s] obj_max DMILen=%.4g ADXTh=%.4g M=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d (%.1fs)",
                 attempt_num, name, om["DMILen"], om["ADXThresh"], om["ATRMult"],
                 om["net_profit"], om["max_drawdown"], om["objective"], om["total_trades"], elapsed)
        log.info("       Best Obj=%.0f (LEN=%.4g ATRL=%.4g MULT=%.4g) | NP-max=%.0f gap_to_target=%.0f",
                 obj_best.get("objective", 0), seed_len, seed_atrl, seed_mult,
                 np_best.get("net_profit", 0), max(0, TARGET_NP - np_best.get("net_profit", 0)))
        save_json(obj_best, np_best, attempt_log, target_met)

    def _do(A, name, cfg):
        log.info("A%02d  %s  %d combos", A, name, cfg.total_runs())
        if start_attempt <= A:
            t = time.time()
            _update(run_or_load(name, cfg, conn, from_csv), cfg, name, A, cfg.total_runs(), t)

    # static attempts (3D)
    # static attempts (INTEGER DMILen/ADXThresh; ATRMult fractional) around L12/Th26/M7.8
    _do(1, "01_rule5_retest", _cfg("01_rule5_retest", (8,16,1),   (22,30,2),    (6.5,9,0.5)))   # 270
    _do(2, "02_dmilen_fine",  _cfg("02_dmilen_fine",  (4,30,1),   (24,28,2),    (7,8.5,0.5)))
    _do(3, "03_adx_scan",     _cfg("03_adx_scan",     (10,14,2),  (10,45,1),    (7,8.5,0.5)))   # ADXThresh full sweep
    _do(4, "04_mult_push",    _cfg("04_mult_push",    (10,14,2),  (24,28,2),    (5,16,0.5)))    # ATRMult 5-16
    _do(5, "05_ultra_wide_m", _cfg("05_ultra_wide_m", (8,16,2),   (22,30,2),    (9,20,0.5)))    # M beyond 7.8
    _do(6, "06_short_dmilen", _cfg("06_short_dmilen", (2,12,1),   (20,32,2),    (7,9,0.5)))
    _do(7, "07_long_dmilen",  _cfg("07_long_dmilen",  (14,40,2),  (20,32,2),    (6.5,9,0.5)))
    _do(8, "08_high_adx",     _cfg("08_high_adx",     (8,16,2),   (28,50,2),    (6.5,9,0.5)))   # high ADX gate
    _do(9, "09_global_int",   _cfg("09_global_int",   (2,40,3),   (12,45,3),    (4,12,1)))

    # adaptive zooms around the running Obj-max (Length/ATRL integer; ATRMult fractional)
    for A, nm_, (rl_mul, rl_min, ra_mul, ra_min, rm_mul, rm_min, nL, nA, nM) in [
        (10, "10_adaptive_zoom1", (0.40, 8.0, 0.40, 6.0, 0.40, 1.5, 13, 11, 11)),
        (11, "11_adaptive_zoom2", (0.20, 4.0, 0.22, 3.0, 0.22, 0.75, 11, 11, 11)),
        (12, "12_adaptive_zoom3", (0.10, 2.0, 0.12, 2.0, 0.12, 0.5, 9, 9, 9)),
    ]:
        log.info("A%02d  %s -- center LEN=%.4g ATRL=%.4g MULT=%.4g Obj=%.0f",
                 A, nm_, seed_len, seed_atrl, seed_mult, obj_best.get("objective", 0))
        if start_attempt <= A:
            _len  = zoom_int(seed_len,   max(rl_min, seed_len * rl_mul),   nL, LEN_LO, LEN_HI)
            _atrl = zoom_int(seed_atrl,  max(ra_min, seed_atrl * ra_mul),  nA, ATRL_LO, ATRL_HI)
            _mult = zoom_fixed(seed_mult, max(rm_min, seed_mult * rm_mul),  nM, 0.25, MULT_LO, MULT_HI)
            _c = _cfg(nm_, _len, _atrl, _mult)
            log.info("A%02d  LEN%s ATRL%s MULT%s  %d combos", A, _len, _atrl, _mult, _c.total_runs())
            t = time.time()
            _update(run_or_load(nm_, _c, conn, from_csv), _c, nm_, A, _c.total_runs(), t)

    log.info("==============================================================")
    log.info("  SFJ_ADXtrend_crypto BNBUSDT Hourly Round-2 COMPLETE (%.1f min)", (time.time()-_RUN_T0)/60)
    if obj_best:
        log.info("  Obj-max (your criterion): LEN=%.4g ATRL=%.4g MULT=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
                 obj_best["DMILen"], obj_best["ADXThresh"], obj_best["ATRMult"],
                 obj_best["net_profit"], obj_best["max_drawdown"], obj_best["objective"], obj_best["total_trades"])
    if np_best:
        log.info("  NP-max: LEN=%.4g ATRL=%.4g MULT=%.4g NP=%.0f Obj=%.0f",
                 np_best["DMILen"], np_best["ADXThresh"], np_best["ATRMult"],
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
    ap = argparse.ArgumentParser(description="SFJ_ADXtrend_crypto BNBUSDT HOT Hourly R1 search")
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
