"""
search_eth_ct_daily.py — SFJ_15Dworkshop_lesson5_countertrend_LS_crypto on ETHUSDT HOT Daily, Round 1

IN-SAMPLE WINDOW = 2022/01/01 - 2026/01/01 (user-specified).  The optimization runs over the
CHART's loaded data range (the signal/wizard date is a no-op per the OOS-isolation findings),
so this script TRIMS the chart to 2022/01-2026/01 via mc.set_instrument_data_range BEFORE the
attempts (the DTN_DATETIMECHANGE nudge fix makes it actually apply).  insample stays WIDE = no-op.

Objective = NetProfit^2 / |MaxDrawdown| (plateau.compute_objective); target NP > 10,000 USD.

Priors (no ETH Daily CT prior — this is the first):
  ETH Hourly CT ceiling $5,005 (LL=111 SL=4.025 LS=115 SS=4.725, rare-extreme 25tr;
    Obj-max LL=110 LS=110 SS=4.7).
  BTC Daily CT champion LL=49 SL=2.4 LS=123 SS=1.2 (16tr); alt LL=40 LS=130 SS=1.0;
    futures high-freq LL=4 LS=45 SS=1.75; tight-SL LL=14 LS=60 SS=0.5.
  Futures CT Daily: NQ LL=8 SL=0.7 LS=47 SS=1.86; GC LL=3 SL=1.28 LS=50 SS=2.3.
  Daily bars ~1,460 over the 2022-2026 IS window — long warmup matters for large LL/LS.

R1 Plan (11 attempts, <=5000 combos each):
  A01 global_coarse        LL(10-130 s20)  SL(0.5-4.5 s0.5)  LS(10-130 s20)  SS(1.0-4.5 s0.5)   = 3528
  A02 btc_daily_regime     LL(30-70 s5)    SL(1.5-3.0 s0.25) LS(100-150 s5)  SS(0.8-2.0 s0.2)   = 4851
  A03 eth_hourly_analog    LL(95-130 s5)   SL(3.5-5.0 s0.25) LS(95-130 s5)   SS(4.0-5.0 s0.25)  = 2240
  A04 futures_daily        LL(2-14 s2)     SL(0.5-2.0 s0.25) LS(35-60 s5)    SS(1.0-3.0 s0.25)  = 2646
  A05 mid_period           LL(15-60 s5)    SL(1.0-4.0 s0.5)  LS(15-60 s5)    SS(1.0-4.0 s0.5)   = 4900
  A06 high_stddev          LL(10-90 s10)   SL(4.5-7.0 s0.5)  LS(10-90 s10)   SS(4.5-7.0 s0.5)   = 2916
  A07 asym_longLL_shortLS  LL(60-130 s10)  SL(3.0-5.0 s0.25) LS(2-20 s2)     SS(3.0-5.0 s0.5)   = 3600
  A08 tight_SL             LL(2-20 s3)     SL(0.1-1.0 s0.15) LS(20-60 s5)    SS(0.2-2.0 s0.3)   = 3087
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
SYMBOL     = "ETHUSDT HOT"
SIGNAL     = "SFJ_15Dworkshop_lesson5_countertrend_LS_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\eth_ct_daily_search")

# IN-SAMPLE chart data range (user-specified); the chart is trimmed to this before attempts.
IS_RANGE   = ("2022/01/01", "2026/01/01")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op (signal date does not restrict)

TARGET_NP  = 10_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.1,  20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.1,  20.0

# No ETH Daily prior — neutral seed at signal defaults (LENGTH=46, STDDEV=3.8)
SEED_LL, SEED_SL = 46.0, 3.8
SEED_LS, SEED_SS = 46.0, 3.8
SEED_NP          = 0.0

PREFIX = "ETHCTD1_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_eth_ct_daily_{int(time.time())}.log"
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
        "round": 1,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily",
        "is_range": IS_RANGE,
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only -- no STP or LMT",
            "contract": "_Crypto1MUSD=Round(1000000/C,0)",
            "is_window": "2022/01/01-2026/01/01 (chart trimmed via set_instrument_data_range)",
            "priors":   "ETH Hourly CT $5,005 (LL=111 LS=115 SS=4.725); BTC Daily CT LL=49 SL=2.4 LS=123 SS=1.2; futures CT Daily NQ LL=8 LS=47 / GC LL=3 LS=50",
            "r1_plan":  "Global coarse + BTC-daily-regime + ETH-hourly-analog + futures-daily + mid period + high stddev + asym longLL/shortLS + tight SL + adaptive zoom",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_eth_ct_daily.json"
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
    log.info("  ETH Daily countertrend_LS_crypto NP>10K -- Round 1")
    log.info("  IS window %s ~ %s  (chart-trimmed)", IS_RANGE[0], IS_RANGE[1])
    log.info("  Priors: ETH Hourly CT $5,005 (LL=111 LS=115); BTC Daily CT LL=49 LS=123")
    log.info("==============================================================")

    # --- TRIM the chart to the IS window (this is what restricts the optimization) ---
    if not from_csv and conn is not None:
        mc.ensure_chart_ready(conn, _cfg("seed", (44, 48, 2), (3.7, 3.9, 0.1),
                                         (44, 48, 2), (3.7, 3.9, 0.1)))
        log.info("Trimming chart data range to IS window %s ~ %s ...", *IS_RANGE)
        mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
        log.info("Chart trimmed. (verify rightmost bar ~2026/01, leftmost ~2022/01)")

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
        ("01_global_coarse",        (10, 130, 20),  (0.5, 4.5, 0.5),   (10, 130, 20),  (1.0, 4.5, 0.5)),
        ("02_btc_daily_regime",     (30, 70, 5),    (1.5, 3.0, 0.25),  (100, 150, 5),  (0.8, 2.0, 0.2)),
        ("03_eth_hourly_analog",    (95, 130, 5),   (3.5, 5.0, 0.25),  (95, 130, 5),   (4.0, 5.0, 0.25)),
        ("04_futures_daily",        (2, 14, 2),     (0.5, 2.0, 0.25),  (35, 60, 5),    (1.0, 3.0, 0.25)),
        ("05_mid_period",           (15, 60, 5),    (1.0, 4.0, 0.5),   (15, 60, 5),    (1.0, 4.0, 0.5)),
        ("06_high_stddev",          (10, 90, 10),   (4.5, 7.0, 0.5),   (10, 90, 10),   (4.5, 7.0, 0.5)),
        ("07_asym_longLL_shortLS",  (60, 130, 10),  (3.0, 5.0, 0.25),  (2, 20, 2),     (3.0, 5.0, 0.5)),
        ("08_tight_SL",             (2, 20, 3),     (0.1, 1.0, 0.15),  (20, 60, 5),    (0.2, 2.0, 0.3)),
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
    log.info("  ETH Daily countertrend_LS_crypto Round-1 COMPLETE")
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
        description="ETH Daily countertrend_LS_crypto NP>10K Round-1 search")
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
        mc.ensure_chart_ready(conn, _cfg("seed", (44, 48, 2), (3.7, 3.9, 0.1),
                                         (44, 48, 2), (3.7, 3.9, 0.1)))
        print(f"\nProbe: trimming chart to IS window {IS_RANGE[0]} ~ {IS_RANGE[1]}")
        mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
        time.sleep(1.0)
        rb_from, rb_to = mc.read_instrument_data_range(conn)
        def _fmt(t):
            return f"{t[0]:04d}/{t[1]:02d}/{t[2]:02d}" if t else "None"
        exp_from = tuple(int(x) for x in IS_RANGE[0].split("/"))
        exp_to = tuple(int(x) for x in IS_RANGE[1].split("/"))
        print("\n================ PROBE RESULT ================")
        print(f"  target   From={IS_RANGE[0]}  To={IS_RANGE[1]}")
        print(f"  readback From={_fmt(rb_from)}  To={_fmt(rb_to)}")
        print(f"  persisted: From {'OK' if rb_from == exp_from else 'X'} / To {'OK' if rb_to == exp_to else 'X'}")
        print("  --> SCREENSHOT chart: leftmost bar ~2022/01, rightmost ~2026/01.")
        print("=============================================")
        return 0

    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
