"""
search_btc_osc_hourly2.py -- _2021Basic_Osc_crypto on BTCUSDT HOT Hourly, Round 2 (boundary confirm)

R1 ceiling: Obj-max $946 (LEN=6 LE=2.25 SE=-0.5 STP=2 LMT=9, MDD=-187, 66tr, Obj=4790, A10=A11=A12 converged);
            NP-max  $950 (LEN=7 LE=2.5 SE=-0.5 STP=2 LMT=9, MDD=-227, 104tr).
BTC = MOMENTUM/breakout regime (LE>0 buy breakout, SE<0 short breakdown) -- OPPOSITE of ETH's deep-LE
mean-reversion. R1 LE=2.25 sits just past the A03 breakout-grid boundary (LE max 2); R2 confirms the
ceiling by pushing LE higher (2-5 momentum boundary) + SE deeper + LEN fine + Rule-5 retest.

Same 5-param strategy / crypto infra as R1. IS window 2022/01/01-2026/01/01 chart-trimmed.
Target NP>100,000.  Objective = NP^2/|MDD|.  <=5000 combos/attempt.  Reports NP-max + Obj-max.

Attempt schedule (9 attempts):
  B01 rule5_retest   LEN(43-47 s1)  x LE(-3.5--2.5 s0.5) x SE(1.5-2.5 s0.5) x STP(0.75-1.25 s0.25) x LMT(15-18 s1)  = 540   (Rule-5 re-measure champion)
  B02 LE_push_deep   LEN(42-50 s2)  x LE(-6--2.5 s0.5)   x SE(1.5-2.5 s0.5) x STP(0.75-1.25 s0.25) x LMT(14-20 s2)  = 1440  (push LE below -3)
  B03 LMT_push_high  LEN(42-50 s2)  x LE(-4--2.5 s0.5)   x SE(1.5-2.5 s0.5) x STP(0.75-1.25 s0.25) x LMT(12-28 s2)  = 1620  (push LMT above 16.5)
  B04 LEN_fine       LEN(36-56 s1)  x LE(-3.5--2.5 s0.5) x SE(1.5-2.5 s0.5) x STP(0.75-1.25 s0.25) x LMT(15-18 s1)  = 2268
  B05 deepLE_highLMT LEN(42-50 s4)  x LE(-6--3 s0.5)     x SE(1.5-2.5 s0.5) x STP(0.75-1.25 s0.25) x LMT(14-24 s2)  = 1134  (combined corner)
  B06 SE_STP_refine  LEN(43-49 s2)  x LE(-3.5--2.5 s0.5) x SE(1-3.5 s0.5)   x STP(0.5-2 s0.25)     x LMT(15-18 s1)  = 2016
  B07-B09 adaptive zoom (5D) around running Obj-max
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
SYMBOL     = "BTCUSDT HOT"
SIGNAL     = "_2021Basic_Osc_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\btc_osc_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO, LEN_HI = 2.0,   200.0
LE_LO,  LE_HI  = -10.0, 10.0
SE_LO,  SE_HI  = -10.0, 10.0
STP_LO, STP_HI = 0.1,   50.0
LMT_LO, LMT_HI = 0.5,   200.0

# R1 champion seeds (zoom center if no attempt run yet)
SEED_LEN, SEED_LE  = 6.0,  2.25
SEED_SE,  SEED_STP = -0.5,  2.0
SEED_LMT           = 9.0

PREFIX = "BTCOSCH2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_osc_hourly2_{int(time.time())}.log"
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
        "round": 2, "strategy": SIGNAL, "symbol": SYMBOL, "timeframe": "Hourly (60 min)",
        "is_window": "2022/01/01 - 2026/01/01 (chart-trimmed)",
        "target_np": TARGET_NP, "target_met": target_met,
        "objective_def": "NetProfit^2 / |MaxDrawdown|",
        "r1_champion": {"obj_max": "LEN=6 LE=2.25 SE=-0.5 STP=2 LMT=9 NP=946 MDD=-187 66tr Obj=4790 (A10=A11=A12)",
                        "np_max": "LEN=7 LE=2.5 SE=-0.5 STP=2 LMT=9 NP=950 MDD=-227 104tr"},
        "notes": {
            "logic": "BUY H STOP when C crosses over BollingerBand(C,LEN,LE); SHORT L STOP when C crosses under BollingerBand(C,LEN,SE)",
            "exits": "STP x ATR(10) stop + LMT x ATR(10) limit",
            "contract": "_Crypto1MUSD = Round(1000000/C, 0) ~ $1M notional/trade",
            "r2_goal": "confirm ceiling: push LE above 2.25 (momentum), SE deeper, fine LEN 2-14, Rule-5 retest",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_btc_osc_hourly2.json"
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
    log.info("  _2021Basic_Osc_crypto BTCUSDT HOT Hourly -- Round 2 boundary confirm (IS 2022/01-2026/01)")
    log.info("  R1 ceiling Obj $4,790 / NP $950 (LEN=6 LE=2.25 SE=-0.5 STP=2 LMT=9 MOMENTUM)")
    log.info("  Pushing LE>2.25, SE deeper, LEN 2-14.  Obj=NP^2/|MDD|.  Target NP %.0f", TARGET_NP)
    log.info("  Run start: %s", datetime.now().isoformat())
    log.info("==============================================================")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (5, 7, 1), (2.0, 2.5, 0.25),
                                                 (-1, 0, 0.5), (1.5, 2.5, 0.5), (8, 10, 1)))
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
            log.info("  [B%02d %-18s] no valid data", attempt_num, name)
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
        log.info("  [B%02d %-18s] obj_max LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f MDD=%.0f Obj=%.0f tr=%d (%.1fs)",
                 attempt_num, name, om["LEN"], om["LE"], om["SE"], om["STP"], om["LMT"],
                 om["net_profit"], om["max_drawdown"], om["objective"], om["total_trades"], elapsed)
        log.info("       Best Obj=%.0f | NP-max=%.0f gap_to_target=%.0f",
                 obj_best.get("objective", 0), np_best.get("net_profit", 0),
                 max(0, TARGET_NP - np_best.get("net_profit", 0)))
        save_json(obj_best, np_best, attempt_log, target_met)

    def _do(A, name, cfg):
        log.info("B%02d  %s  %d combos", A, name, cfg.total_runs())
        if start_attempt <= A:
            t = time.time()
            _update(run_or_load(name, cfg, conn, from_csv), cfg, name, A, cfg.total_runs(), t)

    # static boundary-confirm attempts (5D)
    _do(1, "01_rule5_retest",   _cfg("01_rule5_retest",   (4,8,1),    (1.75,2.75,0.25),(-1,0,0.5),    (1.5,2.5,0.5),    (8,10,1)))
    _do(2, "02_LE_push_high",   _cfg("02_LE_push_high",   (4,10,2),   (2,5,0.5),       (-1.5,0,0.5),  (1.5,2.5,0.5),    (7,11,2)))
    _do(3, "03_SE_push_deep",   _cfg("03_SE_push_deep",   (4,10,2),   (1.5,3,0.5),     (-3,0.5,0.5),  (1.5,2.5,0.5),    (7,11,2)))
    _do(4, "04_LEN_fine",       _cfg("04_LEN_fine",       (2,14,1),   (2,2.75,0.25),   (-1,0,0.5),    (1.5,2.5,0.5),    (8,10,1)))
    _do(5, "05_highLE_corner",  _cfg("05_highLE_corner",  (3,9,2),    (2.5,5,0.5),     (-2,0,0.5),    (1.5,2.5,0.5),    (7,11,2)))
    _do(6, "06_LMT_STP_refine", _cfg("06_LMT_STP_refine", (4,8,2),    (2,2.75,0.25),   (-1,0,0.5),    (1,3,0.25),       (5,13,1)))

    # adaptive zooms around the running Obj-max
    for A, nm_, radii in [
        (7, "07_adaptive_zoom1", [(4,0.75,0.75,0.5,3.0),(3,0.5,0.5,0.5,2.5),(2,0.5,0.5,0.25,2.0),(2,0.25,0.25,0.25,1.5)]),
        (8, "08_adaptive_zoom2", [(3,0.5,0.5,0.5,3.0),(2,0.5,0.5,0.25,2.0),(2,0.25,0.25,0.25,1.5),(1,0.25,0.25,0.25,1.0)]),
        (9, "09_adaptive_zoom3", [(2,0.5,0.5,0.5,2.0),(2,0.25,0.25,0.25,1.5),(1,0.25,0.25,0.25,1.0),(1,0.25,0.25,0.125,1.0)]),
    ]:
        log.info("B%02d  %s -- center LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  Obj=%.0f",
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
            log.info("B%02d  LEN%s LE%s SE%s STP%s LMT%s -> %d combos",
                     A, _len, _le, _se, _stp, _lmt, _c.total_runs())
            t = time.time()
            _update(run_or_load(nm_, _c, conn, from_csv), _c, nm_, A, _c.total_runs(), t)

    log.info("==============================================================")
    log.info("  _2021Basic_Osc_crypto BTCUSDT Hourly Round-2 COMPLETE (%.1f min)", (time.time()-_RUN_T0)/60)
    if obj_best:
        log.info("  Obj-max: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
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
    ap = argparse.ArgumentParser(description="_2021Basic_Osc_crypto BTCUSDT HOT Hourly R2 boundary-confirm")
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
