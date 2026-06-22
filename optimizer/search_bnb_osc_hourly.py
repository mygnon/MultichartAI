"""
search_bnb_osc_hourly.py -- _2021Basic_Osc_crypto on BNBUSDT HOT Hourly, Round 1 (IS 2022/01-2026/01)

Strategy (Knowledge/_2021Basic_Osc_crypto.docx): BB-oscillator, long+short, STOP entries, ATR exits.
  INPUT: LEN(5), LE(-1), SE(1.75), STP(1), LMT(7.5);  ATR = AvgTrueRange(10);
  BUY  _Crypto1MUSD NEXT BAR H STOP  when C CROSS OVER  BollingerBand(C, LEN, LE)
  SHORT _Crypto1MUSD NEXT BAR L STOP when C CROSS UNDER BollingerBand(C, LEN, SE)
  Stop:  exit at EntryPrice -/+ STP x ATR(10)
  Limit: exit at EntryPrice +/- LMT x ATR(10)
  Contract: _Crypto1MUSD = Round(1,000,000/C, 0) ~ $1M notional per trade.

5 params (LEN, LE, SE, STP, LMT).  LE/SE = STDDEV multipliers (LE can be negative -> band below MA).
Target NP > 100,000 USD.  Objective = NetProfit^2 / |MaxDrawdown| (bigger=better).  <=5000 combos/attempt.
IS window 2022/01/01-2026/01/01 chart-trimmed (MC64 ignores signal Begin-date; only the loaded chart
data range restricts the optimization).  Reports BOTH NP-max and Obj-max champions.

Attempt schedule (12 attempts, 5D grids, <=5000):
  A01 global_wide      LEN(5-25 s5)  x LE(-2-1 s1)     x SE(0.5-2.5 s0.5) x STP(0.5-3 s0.5)   x LMT(3-12 s3)   = 2400
  A02 default_zone     LEN(3-13 s2)  x LE(-2-0 s0.5)   x SE(1-3 s0.5)     x STP(0.5-2 s0.5)   x LMT(5-15 s2.5) = 3000
  A03 breakout_regime  LEN(5-25 s5)  x LE(0.5-2 s0.5)  x SE(-1.5-0.5 s0.5)x STP(0.5-2 s0.5)   x LMT(3-9 s3)    = 1200
  A04 tight_exit       LEN(5-20 s5)  x LE(-1.5-0 s0.5) x SE(0.5-2 s0.5)   x STP(0.25-1 s0.25) x LMT(2-8 s2)    = 1024
  A05 long_lmt         LEN(5-20 s5)  x LE(-1.5-0.5 s0.5)x SE(1-3 s0.5)    x STP(0.5-2 s0.5)   x LMT(8-20 s3)   = 2000
  A06 global_boundary  LEN(5-45 s10) x LE(-3-2 s1)     x SE(-1-4 s1)      x STP(0.5-3 s0.5)   x LMT(3-12 s3)   = 4320
  A07 len_fine         LEN(3-25 s2)  x LE(-1.5-0 s0.5) x SE(0.5-2 s0.5)   x STP(0.5-2 s0.5)   x LMT(3-9 s3)    = 2304
  A08 stp_lmt_fine     LEN(5-20 s5)  x LE(-1.5-0 s0.5) x SE(0.5-2 s0.5)   x STP(0.25-2 s0.25) x LMT(2-12 s2)   = 3072
  A09 high_lmt_ratio   LEN(5-25 s5)  x LE(-2-0 s0.5)   x SE(1.5-3.5 s0.5) x STP(0.25-1 s0.25) x LMT(10-20 s2)  = 3000
  A10-A12 adaptive zoom (5D) around the running Obj-max
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_BASIC_OSC_AI.wsp"
SYMBOL     = "BNBUSDT HOT"
SIGNAL     = "_2021Basic_Osc_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_osc_hourly_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO, LEN_HI = 2.0,   200.0
LE_LO,  LE_HI  = -10.0, 10.0
SE_LO,  SE_HI  = -10.0, 10.0
STP_LO, STP_HI = 0.1,   50.0
LMT_LO, LMT_HI = 0.5,   200.0

SEED_LEN, SEED_LE  = 5.0,  -1.0
SEED_SE,  SEED_STP = 1.75,  1.0
SEED_LMT           = 7.5

PREFIX = "BNBOSCH_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_osc_hourly_{int(time.time())}.log"
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

def _snap(val: float, step: float) -> float:
    return round(round(val / step) * step, 8)


def zoom(center: float, radius: float, step: float,
         lo: float, hi: float) -> Tuple[float, float, float]:
    start = max(lo, _snap(center - radius, step))
    stop  = min(hi, _snap(center + radius, step))
    if stop <= start:
        stop = start + step
    return (start, stop, step)


def n_vals(t: Tuple[float, float, float]) -> int:
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name: str,
         len_:  Tuple[float, float, float],
         le:    Tuple[float, float, float],
         se:    Tuple[float, float, float],
         stp:   Tuple[float, float, float],
         lmt:   Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    len_ = _safe(len_, LEN_LO, LEN_HI)
    le   = _safe(le,   LE_LO,  LE_HI)
    se   = _safe(se,   SE_LO,  SE_HI)
    stp  = _safe(stp,  STP_LO, STP_HI)
    lmt  = _safe(lmt,  LMT_LO, LMT_HI)

    combos = n_vals(len_) * n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("LEN", *len_),
            ParamAxis("LE",  *le),
            ParamAxis("SE",  *se),
            ParamAxis("STP", *stp),
            ParamAxis("LMT", *lmt),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
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
    return {"LEN": float(b["LEN"]), "LE": float(b["LE"]), "SE": float(b["SE"]),
            "STP": float(b["STP"]), "LMT": float(b["LMT"]),
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
            "logic": "BUY H STOP when C crosses over BollingerBand(C,LEN,LE); SHORT L STOP when C crosses under BollingerBand(C,LEN,SE)",
            "exits": "STP x ATR(10) stop + LMT x ATR(10) limit",
            "contract": "_Crypto1MUSD = Round(1000000/C, 0) ~ $1M notional/trade",
            "params": "LEN=BB period, LE=long-entry STDDEV mult (can be neg), SE=short-entry STDDEV mult, STP=ATR stop, LMT=ATR limit; defaults LEN=5 LE=-1 SE=1.75 STP=1 LMT=7.5",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_osc_hourly.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_len, seed_le, seed_se, seed_stp, seed_lmt = SEED_LEN, SEED_LE, SEED_SE, SEED_STP, SEED_LMT
    obj_best: Dict = {}
    np_best: Dict = {}
    target_met = False
    attempt_log: List[dict] = []

    log.info("==============================================================")
    log.info("  _2021Basic_Osc_crypto on BNBUSDT HOT Hourly -- Round 1 (IS 2022/01-2026/01 chart-trimmed)")
    log.info("  Params: LEN, LE, SE, STP, LMT.  Obj=NP^2/|MDD|.  Target NP %.0f USD", TARGET_NP)
    log.info("  Default: LEN=5 LE=-1 SE=1.75 STP=1 LMT=7.5")
    log.info("  Run start: %s", datetime.now().isoformat())
    log.info("==============================================================")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (5, 7, 1), (-2, 0, 1),
                                                 (1, 3, 1), (0.5, 1.5, 0.5), (5, 9, 2)))
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
        nonlocal seed_len, seed_le, seed_se, seed_stp, seed_lmt
        nonlocal obj_best, np_best, target_met
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
            log.info("  [A%02d %-18s] no valid data", attempt_num, name)
            save_json(obj_best, np_best, attempt_log, target_met)
            return
        if not obj_best or om["objective"] > obj_best.get("objective", -1):
            obj_best = {**om, "attempt": attempt_num, "name": name}
            seed_len, seed_le, seed_se, seed_stp, seed_lmt = (
                om["LEN"], om["LE"], om["SE"], om["STP"], om["LMT"])
        if not np_best or nm["net_profit"] > np_best.get("net_profit", -1):
            np_best = {**nm, "attempt": attempt_num, "name": name}
            if nm["net_profit"] > TARGET_NP:
                target_met = True
        log.info("  [A%02d %-18s] obj_max LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f MDD=%.0f Obj=%.0f tr=%d (%.1fs)",
                 attempt_num, name, om["LEN"], om["LE"], om["SE"], om["STP"], om["LMT"],
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

    # static attempts (5D sweeps)
    _do(1, "01_global_wide",     _cfg("01_global_wide",     (5,25,5),  (-2,1,1),     (0.5,2.5,0.5),  (0.5,3.0,0.5),  (3,12,3)))
    _do(2, "02_default_zone",    _cfg("02_default_zone",    (3,13,2),  (-2,0,0.5),   (1,3,0.5),      (0.5,2.0,0.5),  (5,15,2.5)))
    _do(3, "03_breakout_regime", _cfg("03_breakout_regime", (5,25,5),  (0.5,2.0,0.5),(-1.5,0.5,0.5), (0.5,2.0,0.5),  (3,9,3)))
    _do(4, "04_tight_exit",      _cfg("04_tight_exit",      (5,20,5),  (-1.5,0.0,0.5),(0.5,2.0,0.5), (0.25,1.0,0.25),(2,8,2)))
    _do(5, "05_long_lmt",        _cfg("05_long_lmt",        (5,20,5),  (-1.5,0.5,0.5),(1,3,0.5),     (0.5,2.0,0.5),  (8,20,3)))
    _do(6, "06_global_boundary", _cfg("06_global_boundary", (5,45,10), (-3,2,1),     (-1,4,1),       (0.5,3.0,0.5),  (3,12,3)))
    _do(7, "07_len_fine",        _cfg("07_len_fine",        (3,25,2),  (-1.5,0.0,0.5),(0.5,2.0,0.5), (0.5,2.0,0.5),  (3,9,3)))
    _do(8, "08_stp_lmt_fine",    _cfg("08_stp_lmt_fine",    (5,20,5),  (-1.5,0.0,0.5),(0.5,2.0,0.5), (0.25,2.0,0.25),(2,12,2)))
    _do(9, "09_high_lmt_ratio",  _cfg("09_high_lmt_ratio",  (5,25,5),  (-2,0,0.5),   (1.5,3.5,0.5),  (0.25,1.0,0.25),(10,20,2)))

    # adaptive zooms around the running Obj-max
    for A, nm_, radii in [
        (10, "10_adaptive_zoom1", [(4,0.75,0.75,0.5,3.0),(3,0.5,0.5,0.5,2.5),(2,0.5,0.5,0.25,2.0),(2,0.25,0.25,0.25,1.5)]),
        (11, "11_adaptive_zoom2", [(3,0.5,0.5,0.5,3.0),(2,0.5,0.5,0.25,2.0),(2,0.25,0.25,0.25,1.5),(1,0.25,0.25,0.25,1.0)]),
        (12, "12_adaptive_zoom3", [(2,0.5,0.5,0.5,2.0),(2,0.25,0.25,0.25,1.5),(1,0.25,0.25,0.25,1.0),(1,0.25,0.25,0.125,1.0)]),
    ]:
        log.info("A%02d  %s -- center LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  Obj=%.0f",
                 A, nm_, seed_len, seed_le, seed_se, seed_stp, seed_lmt, obj_best.get("objective", 0))
        if start_attempt <= A:
            for r_len, r_le, r_se, r_stp, r_lmt in radii:
                _len = zoom(seed_len, r_len, 1.0,  LEN_LO, LEN_HI)
                _le  = zoom(seed_le,  r_le,  0.25, LE_LO,  LE_HI)
                _se  = zoom(seed_se,  r_se,  0.25, SE_LO,  SE_HI)
                _stp = zoom(seed_stp, r_stp, 0.25, STP_LO, STP_HI)
                _lmt = zoom(seed_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
                _c = _cfg(nm_, _len, _le, _se, _stp, _lmt)
                if _c.total_runs() <= 5000:
                    break
            log.info("A%02d  LEN%s LE%s SE%s STP%s LMT%s -> %d combos",
                     A, _len, _le, _se, _stp, _lmt, _c.total_runs())
            t = time.time()
            _update(run_or_load(nm_, _c, conn, from_csv), _c, nm_, A, _c.total_runs(), t)

    log.info("==============================================================")
    log.info("  _2021Basic_Osc_crypto BNBUSDT Hourly Round-1 COMPLETE (%.1f min)", (time.time()-_RUN_T0)/60)
    if obj_best:
        log.info("  Obj-max (your criterion): LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
                 obj_best["LEN"], obj_best["LE"], obj_best["SE"], obj_best["STP"], obj_best["LMT"],
                 obj_best["net_profit"], obj_best["max_drawdown"], obj_best["objective"], obj_best["total_trades"])
    if np_best:
        log.info("  NP-max: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g NP=%.0f Obj=%.0f",
                 np_best["LEN"], np_best["LE"], np_best["SE"], np_best["STP"], np_best["LMT"],
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
    ap = argparse.ArgumentParser(description="_2021Basic_Osc_crypto BNBUSDT HOT Hourly R1 search")
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
