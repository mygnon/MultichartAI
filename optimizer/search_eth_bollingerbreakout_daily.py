"""
search_eth_bollingerbreakout_daily.py -- SFJ_BollingerBreakout_crypto on ETHUSDT HOT Daily, Round 1 (IS 2022/01-2026/01)

Strategy (Strategy/SFJ_BollingerBreakout_crypto.txt): pure Bollinger-band breakout + ReentryBars cooldown + ATR chandelier
trailing stop, long+short, STOP entries, _Crypto1MUSD sizing.
  UpBB=BollingerBand(Close,BBLen,BBmult)[1]; DnBB=BollingerBand(Close,BBLen,-BBmult)[1] (non-lagged)
  Long  entry : Buy       next bar at UpBand STOP (flat-only)
  Short entry : SellShort  next bar at DnBand STOP (flat-only)
  Exits : ATR(14) chandelier trail (ATRMult x ATR(14)) under/over the running extreme.

4 params: BBLen (Bollinger period, int), BBmult (band stdev mult, frac), ATRMult (trail, frac), ReentryBars (cooldown, int).
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260622_SFJ_BollingerBreakout_crypto_AI.wsp"
SYMBOL     = "ETHUSDT HOT"
SIGNAL     = "SFJ_BollingerBreakout_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\eth_bollingerbreakout_daily_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO,  LEN_HI  = 2.0,  200.0   # BBLen
ATRL_LO, ATRL_HI = 0.5,  6.0    # BBmult (frac)
MULT_LO, MULT_HI = 0.5,  20.0   # ATRMult
RE_LO,   RE_HI   = 0.0,  300.0   # ReentryBars (int)

SEED_LEN, SEED_ATRL, SEED_MULT, SEED_RE = 20.0, 2.0, 3.0, 0.0  # BBLen, BBmult, ATRMult, ReentryBars

PREFIX = "ETHBOD_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_eth_bollingerbreakout_daily_{int(time.time())}.log"
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


def _cfg(name, length, atrl, mult, reentry) -> StrategyConfig:
    def _safe(t, lo, hi, step_min):
        s, e, step = t
        if abs(s - e) < step_min * 0.5:
            return (max(lo, s - step), min(hi, s + step), step)
        return t
    length  = _safe(length,  LEN_LO,  LEN_HI,  1.0)
    atrl    = _safe(atrl,    ATRL_LO, ATRL_HI, 0.25)  # BBmult fractional
    mult    = _safe(mult,    MULT_LO, MULT_HI, 0.25)
    reentry = _safe(reentry, RE_LO,   RE_HI,   1.0)   # ReentryBars integer
    combos = n_vals(length) * n_vals(atrl) * n_vals(mult) * n_vals(reentry)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[ParamAxis("BBLen",     *length),
                ParamAxis("BBmult",    *atrl),
                ParamAxis("ATRMult",     *mult),
                ParamAxis("ReentryBars", *reentry)],
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
            "ATRMult": float(b["ATRMult"]), "ReentryBars": float(b["ReentryBars"]),
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
        "round": 1, "strategy": SIGNAL, "symbol": SYMBOL, "timeframe": "Daily",
        "is_window": "2022/01/01 - 2026/01/01 (chart-trimmed)",
        "target_np": TARGET_NP, "target_met": target_met,
        "objective_def": "NetProfit^2 / |MaxDrawdown|",
        "notes": {
            "logic": "Pure Bollinger-band breakout + ReentryBars cooldown: UpBB=BollingerBand(Close,BBLen,BBmult)[1] DnBB=BollingerBand(Close,BBLen,-BBmult)[1]; flat & flatBars>=ReentryBars -> Buy STOP@UpBB / SellShort STOP@DnBB; ATR(14) chandelier trailing exits; flat-only; long+short",
            "contract": "_Crypto1MUSD = Round(1000000/C, 0) ~ $1M notional/trade",
            "params": "BBLen (Bollinger period int), BBmult (band stdev mult frac), ATRMult (trail frac), ReentryBars (flat cooldown int); defaults 20/2.0/3.0/0",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_eth_bollingerbreakout_daily.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_len, seed_atrl, seed_mult, seed_re = SEED_LEN, SEED_ATRL, SEED_MULT, SEED_RE
    obj_best: Dict = {}
    np_best: Dict = {}
    target_met = False
    attempt_log: List[dict] = []

    log.info("==============================================================")
    log.info("  SFJ_BollingerBreakout_crypto on ETHUSDT HOT Daily -- Round 1 (IS 2022/01-2026/01)")
    log.info("  Params: BBLen, BBmult, ATRMult, ReentryBars.  Obj=NP^2/|MDD|.  Target NP %.0f USD", TARGET_NP)
    log.info("  Default: 20/14/3.0.  Run start: %s", datetime.now().isoformat())
    log.info("==============================================================")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (18, 22, 2), (1.5, 2.5, 0.5), (2.5, 3.5, 0.5), (0, 4, 2)))
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
        nonlocal seed_len, seed_atrl, seed_mult, seed_re, obj_best, np_best, target_met
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
            seed_len, seed_atrl, seed_mult, seed_re = om["BBLen"], om["BBmult"], om["ATRMult"], om["ReentryBars"]
        if not np_best or nm["net_profit"] > np_best.get("net_profit", -1):
            np_best = {**nm, "attempt": attempt_num, "name": name}
            if nm["net_profit"] > TARGET_NP:
                target_met = True
        log.info("  [A%02d %-16s] obj_max Long=%.4g Short=%.4g Mult=%.4g Re=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d (%.1fs)",
                 attempt_num, name, om["BBLen"], om["BBmult"], om["ATRMult"], om["ReentryBars"],
                 om["net_profit"], om["max_drawdown"], om["objective"], om["total_trades"], elapsed)
        log.info("       Best Obj=%.0f (Long=%.4g Short=%.4g Mult=%.4g Re=%.4g) | NP-max=%.0f gap_to_target=%.0f",
                 obj_best.get("objective", 0), seed_len, seed_atrl, seed_mult, seed_re,
                 np_best.get("net_profit", 0), max(0, TARGET_NP - np_best.get("net_profit", 0)))
        save_json(obj_best, np_best, attempt_log, target_met)

    def _do(A, name, cfg):
        log.info("A%02d  %s  %d combos", A, name, cfg.total_runs())
        if start_attempt <= A:
            t = time.time()
            _update(run_or_load(name, cfg, conn, from_csv), cfg, name, A, cfg.total_runs(), t)

    # static attempts (4D: BBLen x BBmult x ATRMult x ReentryBars)
    _do(1, "01_global",       _cfg("01_global",       (10,80,10), (1,4,0.5),   (3,9,2),    (0,20,5)))   # broad sweep + cooldown
    _do(2, "02_classic",      _cfg("02_classic",      (10,40,5),  (1.5,3,0.5), (2,6,1),    (0,24,4)))   # classic BB zone
    _do(3, "03_reentry_scan", _cfg("03_reentry_scan", (15,25,5),  (2,3,0.5),   (6,8,0.5),  (0,40,4)))   # deep ReentryBars scan
    _do(4, "04_wide_trail",   _cfg("04_wide_trail",   (15,40,5),  (1.5,3,0.5), (6,16,1),   (0,15,5)))   # wide trail x cooldown
    _do(5, "05_bbmult_fine",  _cfg("05_bbmult_fine",  (15,30,5),  (0.5,4,0.25),(6,8,1),    (0,20,5)))   # nail BBmult 0.5-4
    _do(6, "06_bblen_scan",   _cfg("06_bblen_scan",   (5,100,5),  (2,3,0.5),   (6,8,1),    (0,15,5)))   # nail BBLen 5-100
    _do(7, "07_short_bb",     _cfg("07_short_bb",     (3,30,3),   (1,3,0.5),   (3,7,1),    (0,12,4)))   # short BBLen
    _do(8, "08_wide_band",    _cfg("08_wide_band",    (20,80,10), (2.5,5,0.5), (4,10,2),   (0,20,10)))  # wide bands (high BBmult)
    _do(9, "09_bblen_short2", _cfg("09_bblen_short2", (2,20,1),   (2,4,0.5),   (6,10,1),   (0,15,5)))   # short BBLen 2-20 fine (zoom found BBLen=2)

    # adaptive zooms around the running Obj-max (BBLen/ReentryBars integer; BBmult/ATRMult fractional)
    for A, nm_, (rl_mul, rl_min, ra_mul, ra_min, rm_mul, rm_min, rr_mul, rr_min, nL, nA, nM, nR) in [
        (10, "10_adaptive_zoom1", (0.40, 8.0, 0.40, 0.75, 0.40, 1.5, 0.60, 6.0, 7, 7, 7, 5)),
        (11, "11_adaptive_zoom2", (0.22, 4.0, 0.30, 0.50, 0.22, 0.75, 0.40, 4.0, 7, 7, 7, 5)),
        (12, "12_adaptive_zoom3", (0.12, 2.0, 0.20, 0.40, 0.12, 0.5, 0.25, 3.0, 7, 5, 5, 5)),
    ]:
        log.info("A%02d  %s -- center Long=%.4g Short=%.4g Mult=%.4g Re=%.4g Obj=%.0f",
                 A, nm_, seed_len, seed_atrl, seed_mult, seed_re, obj_best.get("objective", 0))
        if start_attempt <= A:
            _len  = zoom_fixed(seed_len,  max(rl_min, seed_len * rl_mul),   nL, 1.0,  LEN_LO,  LEN_HI)
            _atrl = zoom_fixed(seed_atrl, max(ra_min, seed_atrl * ra_mul),  nA, 0.25, ATRL_LO, ATRL_HI)
            _mult = zoom_fixed(seed_mult, max(rm_min, seed_mult * rm_mul),  nM, 0.25, MULT_LO, MULT_HI)
            _re   = zoom_fixed(seed_re,   max(rr_min, seed_re * rr_mul),    nR, 1.0,  RE_LO,   RE_HI)
            _c = _cfg(nm_, _len, _atrl, _mult, _re)
            log.info("A%02d  Long%s Short%s Mult%s Re%s  %d combos", A, _len, _atrl, _mult, _re, _c.total_runs())
            t = time.time()
            _update(run_or_load(nm_, _c, conn, from_csv), _c, nm_, A, _c.total_runs(), t)

    log.info("==============================================================")
    log.info("  SFJ_BollingerBreakout_crypto ETHUSDT Daily Round-1 COMPLETE (%.1f min)", (time.time()-_RUN_T0)/60)
    if obj_best:
        log.info("  Obj-max (your criterion): Long=%.4g Short=%.4g Mult=%.4g Re=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
                 obj_best["BBLen"], obj_best["BBmult"], obj_best["ATRMult"], obj_best["ReentryBars"],
                 obj_best["net_profit"], obj_best["max_drawdown"], obj_best["objective"], obj_best["total_trades"])
    if np_best:
        log.info("  NP-max: Long=%.4g Short=%.4g Mult=%.4g Re=%.4g NP=%.0f Obj=%.0f",
                 np_best["BBLen"], np_best["BBmult"], np_best["ATRMult"], np_best["ReentryBars"],
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
    ap = argparse.ArgumentParser(description="SFJ_BollingerBreakout_crypto ETHUSDT HOT Daily R1 search")
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
