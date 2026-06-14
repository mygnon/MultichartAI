"""
search_btc_ct_daily2.py — SFJ_15Dworkshop_lesson5_countertrend_LS_crypto on BTCUSDT HOT Daily, Round 2

R1 Summary (11 attempts):
  Champion ⭐ A10=A11: LL=49 SL=2.4 LS=123 SS=1.2 NP=$3,592.66 MDD=-$398.78 16tr Obj=32,366
  NEW regime: mid LL≈49, long LS≈123, moderate SL≈2.4, tight SS≈1.2 (unlike Hourly LL=104 SL=4.05).
  Alt high-freq: A04 LL=4 LS=45 $2,687 47tr; A08 LL=14 LS=60 SS=0.5 $2,482 49tr.
  4 EXPORT FAILURES (0-6 rows): A02 hourly_analog (ultra-short LL 2-12),
    A03 long_period (LL 100-250), A06 high_stddev (SS 4.5-7), A07 asym — UNTESTED, redo.
  Gap to $10K -64.1%.

R2 Plan (11 attempts, <=5000 combos each): fine-tune champion + redo all failed regions.
  A01 champion_microfine  LL(46-52 s1)xSL(2.3-2.5 s0.05)xLS(119-127 s1)xSS(1.1-1.3 s0.05) = 1575
  A02 LS_100_160_bridge   LL(45-53 s2)xSL(2.2-2.6 s0.1)xLS(100-160 s5)xSS(1.0-1.4 s0.1)   = 1625
  A03 SS_sweep            LL(45-53 s4)xSL(2.2-2.6 s0.1)xLS(118-128 s5)xSS(0.3-2.0 s0.1)    = 810
  A04 SL_fine            LL(45-53 s2)xSL(1.5-3.5 s0.1)xLS(118-128 s5)xSS(1.0-1.4 s0.1)     = 1575
  A05 redo_long_period   LL(60-150 s10)xSL(1.5-3.5 s0.25)xLS(60-150 s10)xSS(0.5-2.5 s0.5)  = 4500
  A06 redo_high_stddev   LL(20-90 s10)xSL(3.0-5.0 s0.25)xLS(20-90 s10)xSS(1.0-3.0 s0.5)    = 2880
  A07 redo_asym          LL(40-70 s5)xSL(2.0-3.0 s0.2)xLS(5-30 s2)xSS(1.0-3.0 s0.5)        = 2730
  A08 LL_30_70_fine      LL(30-70 s5)xSL(2.0-3.0 s0.1)xLS(40-130 s10)xSS(1.0-2.0 s0.25)    = 3465
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


WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_Bollinger_AI.wsp"
SYMBOL     = "BTCUSDT HOT"
SIGNAL     = "SFJ_15Dworkshop_lesson5_countertrend_LS_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\btc_ct_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 10_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.1,  20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.1,  20.0

# R1 champion as seed (A10=A11)
SEED_LL, SEED_SL = 49.0, 2.4
SEED_LS, SEED_SS = 123.0, 1.2
SEED_NP          = 3592.66

PREFIX = "BTCCTD2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_ct_daily2_{int(time.time())}.log"
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


def _cfg(name, ll, sl, ls, ss):
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
        timeframe="daily",
        bar_period=1440,
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
        "round": 2,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only -- no STP or LMT",
            "contract": "_Crypto1MUSD=Round(1000000/C,0)",
            "r1_result": "R1 champion A10=A11: LL=49 SL=2.4 LS=123 SS=1.2 NP=$3,593 MDD=-$399 16tr Obj=32,366; alt A04 LL=4 LS=45 $2,687 47tr; A02/A03/A06/A07 export-failed",
            "r2_plan":  "Champion microfine + LS 100-160 bridge + SS sweep + SL fine + redo long_period/high_stddev/asym + LL 30-70 fine + adaptive zoom",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_btc_ct_daily2.json"
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
    log.info("  BTC Daily countertrend_LS_crypto NP>10K -- Round 2")
    log.info("  R1 champion: LL=49 SL=2.4 LS=123 SS=1.2 NP=$3,593 Obj=32,366")
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
        ("01_champion_microfine",   (46, 52, 1),    (2.3, 2.5, 0.05),  (119, 127, 1),  (1.1, 1.3, 0.05)),
        ("02_LS_100_160_bridge",    (45, 53, 2),    (2.2, 2.6, 0.1),   (100, 160, 5),  (1.0, 1.4, 0.1)),
        ("03_SS_sweep",             (45, 53, 4),    (2.2, 2.6, 0.1),   (118, 128, 5),  (0.3, 2.0, 0.1)),
        ("04_SL_fine",              (45, 53, 2),    (1.5, 3.5, 0.1),   (118, 128, 5),  (1.0, 1.4, 0.1)),
        ("05_redo_long_period",     (60, 150, 10),  (1.5, 3.5, 0.25),  (60, 150, 10),  (0.5, 2.5, 0.5)),
        ("06_redo_high_stddev",     (20, 90, 10),   (3.0, 5.0, 0.25),  (20, 90, 10),   (1.0, 3.0, 0.5)),
        ("07_redo_asym",            (40, 70, 5),    (2.0, 3.0, 0.2),   (5, 30, 2),     (1.0, 3.0, 0.5)),
        ("08_LL_30_70_fine",        (30, 70, 5),    (2.0, 3.0, 0.1),   (40, 130, 10),  (1.0, 2.0, 0.25)),
    ]

    for idx, (n, ll_r, sl_r, ls_r, ss_r) in enumerate(attempts_config, 1):
        _c = _cfg(n, ll_r, sl_r, ls_r, ss_r)
        log.info("A%02d  %s  %d combos", idx, n, _c.total_runs())
        if start_attempt <= idx:
            _update(run_or_load(n, _c, conn, from_csv), _c, n, idx, _c.total_runs())
        log.info("After A%02d -- best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
                 idx, best_np, best_ll, best_sl, best_ls, best_ss)

    # Adaptive zooms (progressively tighter around running best NP)
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 -- center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [(10, 0.5, 10, 0.5), (6, 0.3, 6, 0.3), (4, 0.2, 4, 0.2)]:
            _ll = zoom(best_ll, r_ll, 2.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.1, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 2.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.1, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A09  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 -- best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 -- center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [(5, 0.25, 5, 0.25), (3, 0.15, 3, 0.15), (2, 0.1, 2, 0.1)]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.05,SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.05,SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 -- best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 -- center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [(2, 0.1, 2, 0.1), (1, 0.05, 1, 0.05), (1, 0.025, 1, 0.025)]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.025,SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.025,SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 -- best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    pct = (best_np / TARGET_NP * 100) if TARGET_NP else 0
    log.info("==============================================================")
    log.info("  BTC Daily countertrend_LS_crypto Round-2 COMPLETE")
    log.info("  Champion: LL=%.4g SL=%.4g LS=%.4g SS=%.4g", best_ll, best_sl, best_ls, best_ss)
    log.info("  Best NP: %.0f USD  (%.1f%% of target %.0f)", best_np, pct, TARGET_NP)
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "MET" if target_met else "NOT MET (gap +%.0f)" % max(0, TARGET_NP - best_np))
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
        description="BTC Daily countertrend_LS_crypto NP>10K Round-2 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1-11)")
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
