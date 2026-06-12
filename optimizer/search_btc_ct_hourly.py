"""
search_btc_ct_hourly.py — SFJ_15Dworkshop_lesson5_countertrend_LS_crypto on BTCUSDT HOT Hourly, Round 1

Strategy logic (BB counter-trend, from docx):
  INPUT: LENGTH_LONG(46), STDDEV_LONG(3.8), LENGTH_SHORT(46), STDDEV_SHORT(3.8)
  DN_LONG  = BollingerBand(C, LENGTH_LONG,  -STDDEV_LONG)   -- lower band for long
  UP_SHORT = BollingerBand(C, LENGTH_SHORT,  STDDEV_SHORT)  -- upper band for short
  BUY       when C crosses OVER  DN_LONG   (counter-trend long at lower BB)
  SELLSHORT when C crosses UNDER UP_SHORT  (counter-trend short at upper BB)
  No STP/LMT -- reversal on opposite signal
  Contract: _Crypto1MUSD = Round(1,000,000/C, 0)  ~$1M notional per trade

Parameters:
  LENGTH_LONG  (LL) : BB period for long entry  (default 46)
  STDDEV_LONG  (SL) : BB std-dev for long entry (default 3.8)
  LENGTH_SHORT (LS) : BB period for short entry (default 46)
  STDDEV_SHORT (SS) : BB std-dev for short entry (default 3.8)

Symbol: BTCUSDT HOT (~$1M notional, crypto)
Timeframe: Hourly (60-min bars)
Target: NP > 10,000 USD

Reference (similar BB countertrend on futures):
  NQ Hourly TARGET MET R3: LL=17 SL=0.2 LS=45 SS=1.4 NP=$751,230 trades=1614
  GC Hourly ceiling $437,930 R3: LL=14 SL=0.1 LS=59 SS=0.45
  TXF Hourly 8M MET R4: LL=22 SL=0.425 LS=43 SS=1.77

Reference (other strategies on BTC Hourly):
  QPATR_Breakout BTC Hourly: $2,748 (Len=212 Mult=3.27, 90 trades)
  QPATRex BTC Hourly: $3,293 (Len=125, 8 trades)

Expected: ~$3K-$10K (high-freq countertrend BB might do well on BTC)

Attempt schedule (12 attempts, <=5,000 combos each):
  A01 broad_sweep      : LL(10-80 s5)xSL(1.5-4.5 s1)xLS(10-80 s5)xSS(1.5-4.5 s1)   = 3600
  A02 fine_stddev      : LL(20-70 s10)xSL(1.0-5.5 s0.5)xLS(20-70 s10)xSS(1.0-5.5 s0.5) = 3600
  A03 medium_fine      : LL(20-70 s5)xSL(2.0-4.5 s0.5)xLS(20-70 s5)xSS(2.0-4.5 s0.5)= 4356
  A04 long_period      : LL(60-160 s10)xSL(2.0-4.5 s0.5)xLS(60-160 s10)xSS(2.0-4.5 s0.5)= 4356
  A05 short_period     : LL(2-30 s2)xSL(1.0-4.0 s1)xLS(2-30 s2)xSS(1.0-4.0 s1)     = 3600
  A06 asymmetric       : LL(20-80 s5)xSL(2.0-5.0 s1)xLS(5-50 s5)xSS(2.0-5.0 s1)    = 2080
  A07 high_stddev      : LL(20-80 s10)xSL(3.5-6.5 s0.5)xLS(20-80 s10)xSS(3.5-6.5 s0.5)= 2401
  A08 low_stddev       : LL(20-80 s10)xSL(1.0-2.8 s0.2)xLS(20-80 s10)xSS(1.0-2.8 s0.2)= 4900
  A09 default_region   : LL(38-56 s2)xSL(3.0-4.5 s0.3)xLS(38-56 s2)xSS(3.0-4.5 s0.3) = 3600
  A10 adaptive_zoom    : (dynamic from R1 best NP)
  A11 very_long_period : LL(80-200 s20)xSL(3.0-5.5 s0.5)xLS(80-200 s20)xSS(3.0-5.5 s0.5)= 1764
  A12 global_boundary  : LL(5-155 s15)xSL(1.0-6.0 s1)xLS(5-155 s15)xSS(1.0-6.0 s1) = 4356
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


WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_Bollinger_AI.wsp"
SYMBOL     = "BTCUSDT HOT"
SIGNAL     = "SFJ_15Dworkshop_lesson5_countertrend_LS_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\btc_ct_hourly_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 10_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.1,  20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.1,  20.0

SEED_LL, SEED_SL = 46.0, 3.8
SEED_LS, SEED_SS = 46.0, 3.8
SEED_NP          = -1_000_000.0

PREFIX = "BTCCT1_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_ct_hourly_{int(time.time())}.log"
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
         ll:  Tuple[float, float, float],
         sl:  Tuple[float, float, float],
         ls:  Tuple[float, float, float],
         ss:  Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    ll = _safe(ll, LL_LO, LL_HI)
    sl = _safe(sl, SL_LO, SL_HI)
    ls = _safe(ls, LS_LO, LS_HI)
    ss = _safe(ss, SS_LO, SS_HI)

    combos = n_vals(ll) * n_vals(sl) * n_vals(ls) * n_vals(ss)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("LENGTH_LONG",  *ll),
            ParamAxis("STDDEV_LONG",  *sl),
            ParamAxis("LENGTH_SHORT", *ls),
            ParamAxis("STDDEV_SHORT", *ss),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
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


def champion(df, fb_ll, fb_sl, fb_ls, fb_ss):
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  TARGET MET: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                 float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                 float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_ll, fb_sl, fb_ls, fb_ss,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, ll, sl, ls, ss, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "LENGTH_LONG": ll, "STDDEV_LONG": sl,
        "LENGTH_SHORT": ls, "STDDEV_SHORT": ss,
        "objective": obj, "net_profit": np_, "max_drawdown": mdd,
        "total_trades": trades, "target_met": met,
        "combos": combos,
        "rows": len(df) if df is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }


def save_json(best_entry, attempt_log, target_met):
    payload = {
        "round": 1,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only -- no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "default":  "LENGTH=46, STDDEV=3.8",
            "contract": "_Crypto1MUSD=Round(1000000/C,0)",
            "reference":"NQ Hourly TARGET MET LL=17 SL=0.2 LS=45 SS=1.4 NP=$751K; GC Hourly LL=14 SL=0.1 LS=59 SS=0.45 NP=$438K",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_btc_ct_hourly.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_ll, best_sl = SEED_LL, SEED_SL
    best_ls, best_ss = SEED_LS, SEED_SS
    best_np  = SEED_NP
    best_obj = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("==============================================================")
    log.info("  BTC Hourly countertrend_LS_crypto NP>10K -- Round 1")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  Strategy: BB counter-trend reversal (no STP/LMT)")
    log.info("  Default params: LENGTH=46, STDDEV=3.8")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("==============================================================")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_ll, best_sl, best_ls, best_ss
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_ll, best_sl, best_ls, best_ss,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        ll, sl, ls, ss, obj, np_, mdd, tr, met = champion(
            df, best_ll, best_sl, best_ls, best_ss)

        if np_ > best_np:
            best_ll, best_sl = ll, sl
            best_ls, best_ss = ls, ss
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, ll, sl, ls, ss,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, ll, sl, ls, ss, obj, np_, mdd, tr,
                 "TARGET" if met else ("%.0f/10K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    attempts_config = [
        ("01_broad_sweep",      (10, 80, 5),   (1.5, 4.5, 1.0),  (10, 80, 5),   (1.5, 4.5, 1.0)),
        ("02_fine_stddev",      (20, 70, 10),  (1.0, 5.5, 0.5),  (20, 70, 10),  (1.0, 5.5, 0.5)),
        ("03_medium_fine",      (20, 70, 5),   (2.0, 4.5, 0.5),  (20, 70, 5),   (2.0, 4.5, 0.5)),
        ("04_long_period",      (60, 160, 10), (2.0, 4.5, 0.5),  (60, 160, 10), (2.0, 4.5, 0.5)),
        ("05_short_period",     (2, 30, 2),    (1.0, 4.0, 1.0),  (2, 30, 2),    (1.0, 4.0, 1.0)),
        ("06_asymmetric",       (20, 80, 5),   (2.0, 5.0, 1.0),  (5, 50, 5),    (2.0, 5.0, 1.0)),
        ("07_high_stddev",      (20, 80, 10),  (3.5, 6.5, 0.5),  (20, 80, 10),  (3.5, 6.5, 0.5)),
        ("08_low_stddev",       (20, 80, 10),  (1.0, 2.8, 0.2),  (20, 80, 10),  (1.0, 2.8, 0.2)),
        ("09_default_region",   (38, 56, 2),   (3.0, 4.5, 0.3),  (38, 56, 2),   (3.0, 4.5, 0.3)),
    ]

    for idx, (n, ll_r, sl_r, ls_r, ss_r) in enumerate(attempts_config, 1):
        _c = _cfg(n, ll_r, sl_r, ls_r, ss_r)
        log.info("A%02d  %s  %d combos", idx, n, _c.total_runs())
        if start_attempt <= idx:
            _update(run_or_load(n, _c, conn, from_csv), _c, n, idx, _c.total_runs())
        log.info("After A%02d -- best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
                 idx, best_np, best_ll, best_sl, best_ls, best_ss)

    # A10 adaptive zoom
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom -- center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (10, 1.0, 10, 1.0),
            (7,  0.7,  7, 0.7),
            (5,  0.5,  5, 0.5),
            (3,  0.3,  3, 0.3),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.1,  SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.1,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # A11 very long period
    A = 11
    _n = "11_very_long_period"
    _c = _cfg(_n, (80, 200, 20), (3.0, 5.5, 0.5), (80, 200, 20), (3.0, 5.5, 0.5))
    log.info("A11  LL(80-200 s20)xSL(3.0-5.5 s0.5)xLS(80-200 s20)xSS(3.0-5.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 -- best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # A12 global boundary
    A = 12
    _n = "12_global_boundary"
    _c = _cfg(_n, (5, 155, 15), (1.0, 6.0, 1.0), (5, 155, 15), (1.0, 6.0, 1.0))
    log.info("A12  LL(5-155 s15)xSL(1.0-6.0 s1)xLS(5-155 s15)xSS(1.0-6.0 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 -- best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    log.info("==============================================================")
    log.info("  BTC Hourly countertrend_LS_crypto Round-1 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  Target %.0f: %s", TARGET_NP,
             "MET" if target_met else f"NOT MET (gap +{max(0, TARGET_NP - best_np):.0f})")
    log.info("==============================================================")

    if not best_entry:
        best_entry = {
            "LENGTH_LONG": best_ll, "STDDEV_LONG": best_sl,
            "LENGTH_SHORT": best_ls, "STDDEV_SHORT": best_ss,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone -- results at: {out}")
    print(f"Target NP>10K: {'MET' if target_met else 'NOT MET -- best NP=%.0f' % best_np}")
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
        description="BTC Hourly countertrend_LS_crypto NP>10K Round-1 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1-12)")
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
