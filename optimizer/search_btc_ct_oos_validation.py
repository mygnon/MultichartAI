"""
search_btc_ct_oos_validation.py — Full-period / out-of-sample (OOS) validation of the
already-optimized exit modules on top of the BTC Hourly CT champion.

This is NOT re-optimization.  The main signal and each module are FIXED at their
already-found values; we evaluate them over multiple date ranges to answer:
  1. Do the IS-optimized modules improve NetProfit and reduce Max Drawdown over the
     FULL period (2019 -> data end)?
  2. Does the main strategy (and main+module) make or LOSE money in the recent
     out-of-sample period (2026/01/01 -> data end)?

Main signal FIXED at Obj-max champion LL=104 SL=4.15 LS=68 SS=4.95 (Status ON).
Module params FIXED at their IS max-NP values (from btc_ct_exit_modules_search):
  M1 ATRstop STP=20.9 | M2 TrailingStop ATRSTP=28.4 | M3 EntryBarsAfterExit EXITBAR=338
  M4 high_volatility DAYRANGE=4.38 | M5 PT_Exit PT_Base=0.4 | M6 RescueTeamExit Length=80 std=3.9

Method: for each period (FULL, OOS) run a tiny micro-grid per config so MC64 exports
a clean CSV (Critical Rule 1: every checked param must vary), then pick the exact
fixed-param row.  Baseline = main only (all modules OFF), main 81-combo micro-grid.
Each module = module param swept over a small bracket around its fixed value (main
stays at workspace champion, verified per-row).  Drawdown = Max Intraday Drawdown
(reliable CSV metric); true Max Strategy Drawdown is read best-effort for the baseline.

IS reference (2019-2026) is reused from results/btc_ct_exit_modules_search/.

PREREQUISITES (one-time manual setup in MC64):
  - Workspace 20260101_SFJ_Bollinger_AI.wsp open, BTCUSDT HOT HOURLY tab active
  - Main signal inputs set to 104 / 4.15 / 68 / 4.95, Status ON
  - All 6 module signals inserted on the chart, Status OFF (module current value
    irrelevant — the bracket sweep sets it)
  - Workspace saved (must contain full data incl. out-of-sample after 2026/01/01)

CLI:
  py search_btc_ct_oos_validation.py                 # FULL + OOS, baseline + M1-M6 + teardown
  py search_btc_ct_oos_validation.py --module 0      # baseline only (both periods) -- smoke
  py search_btc_ct_oos_validation.py --module 3      # only module 3 (both periods)
  py search_btc_ct_oos_validation.py --period oos     # only the OOS period
  py search_btc_ct_oos_validation.py --no-strategy-dd # skip the perf-report Strategy DD read
  py search_btc_ct_oos_validation.py --manual-status  # toggle Status by hand
  py search_btc_ct_oos_validation.py --from-csv       # re-analyze existing CSVs
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
from typing import Dict, List, Optional, Tuple

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mc_automation as mc
import plateau as plateau_mod
from config import DateRange, ParamAxis, StrategyConfig


WORKSPACE   = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_Bollinger_AI.wsp"
SYMBOL      = "BTCUSDT HOT"
MAIN_SIGNAL = "SFJ_15Dworkshop_lesson5_countertrend_LS_crypto"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\btc_ct_oos_validation_search")

# Obj-max champion (R5) — set in MC64 manually, Status ON
CHAMPION = {
    "LENGTH_LONG":  104.0,
    "STDDEV_LONG":  4.15,
    "LENGTH_SHORT": 68.0,
    "STDDEV_SHORT": 4.95,
}

# Evaluation periods. Far-future End lets MC64 clamp to the actual data end.
PERIODS = {
    "full": DateRange("2019/01/01", "2027/01/01"),   # 2019 -> data end
    "oos":  DateRange("2026/01/01", "2027/01/01"),   # OOS only -> data end
}

# IS reference (2019-2026) reused from the prior exit-module run
IS_REF_JSON = Path(r"C:\Users\Tim\MultichartAI\results\btc_ct_exit_modules_search"
                   r"\final_params_btc_ct_exit_modules.json")

# (index, signal, [(param, fixed_value, step), ...]) — module FIXED at IS max-NP value
MODULES: List[Tuple[int, str, List[Tuple[str, float, float]]]] = [
    (1, "SFJ_15Dworkshop_lesson4_ATRstop",                 [("STP",      20.9,  0.1)]),
    (2, "SFJ_15Dworkshop_lesson9_1_TrailingStop",          [("ATRSTP",   28.4,  0.1)]),
    (3, "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit",   [("EXITBAR",  338.0, 1.0)]),
    (4, "SFJ_15Dworkshop_lesson11_3_high_volatility_exit", [("DAYRANGE", 4.38,  0.01)]),
    (5, "QuantPass_PT_Exit",                               [("PT_Base",  0.4,   0.001)]),
    (6, "RescueTeamExit",                                  [("Length",   80.0,  20.0),
                                                            ("std",      3.9,   0.1)]),
]
ALL_MODULE_NAMES = [m[1] for m in MODULES]

LL_LO, LL_HI = 2.0, 500.0
SL_LO, SL_HI = 0.1, 20.0
LS_LO, LS_HI = 2.0, 500.0
SS_LO, SS_HI = 0.1, 20.0
GEN_LO, GEN_HI = 0.001, 100000.0   # generic clamp for module params

PREFIX = "BTCCTOOS_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_ct_oos_validation_{int(time.time())}.log"
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


def n_vals(start, stop, step):
    return max(1, round((stop - start) / step) + 1)


def _bracket(value: float, step: float, lo: float, hi: float, smoke: bool):
    """Small symmetric bracket around a fixed value so the wizard has a varying
    axis (Critical Rule 1).  3 values in smoke, 5 otherwise."""
    n = 1 if smoke else 2
    start = max(lo, round(value - n * step, 8))
    stop = min(hi, round(value + n * step, 8))
    if stop <= start:
        stop = start + step
    return (start, stop, step)


def _baseline_cfg(period: str) -> StrategyConfig:
    # 81-combo micro-grid around the champion: all 4 params vary (Critical Rule 1)
    return StrategyConfig(
        name=f"{PREFIX}{period}_A00_baseline",
        mc_signal_name=MAIN_SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("LENGTH_LONG",  103.0, 105.0, 1.0),
            ParamAxis("STDDEV_LONG",  4.125, 4.175, 0.025),
            ParamAxis("LENGTH_SHORT", 67.0,  69.0,  1.0),
            ParamAxis("STDDEV_SHORT", 4.925, 4.975, 0.025),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=PERIODS[period],
    )


def _module_cfg(period: str, idx: int, signal: str, axes, smoke: bool) -> StrategyConfig:
    params = []
    for (nm, val, step) in axes:
        lo, hi = GEN_LO, GEN_HI
        params.append(ParamAxis(nm, *_bracket(val, step, lo, hi, smoke)))
    combos = 1
    for p in params:
        combos *= n_vals(p.start, p.stop, p.step)
    if combos > 5000:
        log.warning("Module %d (%s): %d combos EXCEEDS 5000!", idx, signal, combos)
    return StrategyConfig(
        name=f"{PREFIX}{period}_M{idx}_{signal[:26]}",
        mc_signal_name=signal,
        timeframe="hourly",
        bar_period=60,
        params=params,
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=PERIODS[period],
    )


def csv_for(cfg: StrategyConfig) -> Path:
    return OUTPUT_DIR / f"{cfg.name}_raw.csv"


def _apply_status(conn, active_module: Optional[str], manual: bool) -> float:
    t0 = time.time()
    status_map = {name: (name == active_module) for name in ALL_MODULE_NAMES}
    if manual:
        print("\n" + "=" * 60)
        print("MANUAL STATUS SETUP (Format Objects > Signals):")
        print(f"  {MAIN_SIGNAL}: ON (inputs 104/4.15/68/4.95)")
        for name, want in status_map.items():
            print(f"  {name}: {'ON' if want else 'OFF'}")
        print("=" * 60)
        input("Set the checkboxes in MC64, click OK, then press Enter here...")
        return time.time() - t0
    mc.set_signal_statuses(conn, status_map, verify=False, protected=[MAIN_SIGNAL])
    el = time.time() - t0
    log.info("[TIMING] status_apply=%.1fs", el)
    return el


def run_or_load(cfg: StrategyConfig, conn, from_csv: bool):
    csv_path = csv_for(cfg)
    if from_csv or csv_path.exists():
        if csv_path.exists():
            try:
                df = mc.load_results_csv(str(csv_path), cfg)
                log.info("Loaded %s: %d rows", cfg.name, len(df))
                return df
            except Exception as e:
                log.warning("Could not load %s: %s", csv_path, e)
        else:
            log.warning("No CSV for %s", cfg.name)
        return None
    log.info("=== Starting %s (%d combos) ===", cfg.name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done %.1f min -- %s", (time.time() - t0) / 60, Path(raw_csv).name)
        return mc.load_results_csv(raw_csv, cfg)
    except Exception as e:
        log.error("  FAILED: %s", e, exc_info=True)
        return None


def _main_fixed_ok(df) -> bool:
    """Per-row proof the main signal stayed at the champion."""
    main_cols = [c for c in CHAMPION if c in df.columns]
    if not main_cols:
        log.warning("  Main-signal columns not in CSV — fixed-value check skipped")
        return True
    for c in main_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        tol = max(1e-6, abs(CHAMPION[c]) * 1e-4)
        if not ((col - CHAMPION[c]).abs() <= tol).all():
            log.error("  INVALID: main param %s NOT fixed at %.6g (got %.6g..%.6g)",
                      c, CHAMPION[c], col.min(), col.max())
            return False
    return True


def _pick_champion_row(df) -> Optional[dict]:
    """Exact champion row from the baseline micro-grid (NP-max fallback)."""
    m = df
    for nm, v in CHAMPION.items():
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    if not m.empty:
        return m.iloc[0]
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    return df.loc[pd.to_numeric(df["NetProfit"], errors="coerce").idxmax()]


def _pick_fixed_row(df, axes) -> Optional[dict]:
    """Row matching each module param's fixed value (closest fallback)."""
    m = df
    for (nm, val, step) in axes:
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - val).abs() < step * 0.5]
    if not m.empty:
        return m.iloc[0]
    # fallback: closest by sum of squared distance on first axis
    nm0, val0, _ = axes[0]
    if nm0 in df.columns:
        idx = (pd.to_numeric(df[nm0], errors="coerce") - val0).abs().idxmin()
        return df.loc[idx]
    return None


def _row_metrics(row) -> dict:
    return {
        "net_profit": float(row["NetProfit"]),
        "max_intraday_drawdown": float(row["MaxDrawdown"]),
        "total_trades": int(row["TotalTrades"]),
    }


def _read_strategy_dd(conn, period: str) -> Optional[dict]:
    """Best-effort: apply the main champion over the period and read the true
    Max Strategy Drawdown from the Strategy Performance Report.  Non-fatal."""
    try:
        cfg = _baseline_cfg(period)
        params = dict(CHAMPION)
        mc.set_params_and_date_for_single_run(conn, cfg, params, PERIODS[period])
        rpt = mc.read_performance_report(conn)
        if rpt:
            log.info("  [%s] perf-report: NP=%s StrategyDD=%s",
                     period, rpt.get("NetProfit"), rpt.get("MaxDrawdown"))
        return rpt
    except Exception as e:
        log.warning("  [%s] strategy-DD read failed (non-fatal): %s", period, e)
        return None


def save_json(payload):
    out = OUTPUT_DIR / "final_params_btc_ct_oos_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def _load_is_ref() -> dict:
    """IS reference (2019-2026) from the prior exit-module run."""
    ref = {"baseline": None, "modules": {}}
    try:
        with open(IS_REF_JSON, encoding="utf-8") as f:
            d = json.load(f)
        bm = d.get("baseline_measured") or d.get("baseline_reference")
        if bm:
            ref["baseline"] = {"net_profit": bm["net_profit"],
                               "max_intraday_drawdown": bm["max_drawdown"],
                               "total_trades": bm.get("total_trades")}
        for m in d.get("modules", []):
            if m.get("valid"):
                ref["modules"][m["module"]] = {
                    "net_profit": m["net_profit"],
                    "max_intraday_drawdown": m["max_drawdown"],
                    "total_trades": m["total_trades"],
                    "best_params": m["best_params"],
                }
    except Exception as e:
        log.warning("Could not load IS reference: %s", e)
    return ref


def run(conn, from_csv, only_module, only_period, manual_status, smoke, want_strategy_dd):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_t0 = time.time()

    periods = [only_period] if only_period else list(PERIODS.keys())

    payload = {
        "strategy_main": MAIN_SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "champion_fixed": CHAMPION,
        "module_fixed_params": {idx: {nm: val for nm, val, _ in axes}
                                for idx, sig, axes in MODULES},
        "periods": {k: [v.from_date, v.to_date] for k, v in PERIODS.items()},
        "is_reference": _load_is_ref(),
        "results": {},      # period -> {"baseline": {...}, "modules": {idx: {...}}}
        "strategy_dd": {},  # period -> perf-report dict (baseline, true Strategy DD)
        "smoke": smoke,
        "run_started_at": datetime.now().isoformat(),
        "run_finished_at": None,
        "run_total_sec": None,
    }
    out_json = OUTPUT_DIR / "final_params_btc_ct_oos_validation.json"
    if out_json.exists():
        try:
            with open(out_json, encoding="utf-8") as f:
                old = json.load(f)
            payload["results"] = old.get("results", {})
            payload["strategy_dd"] = old.get("strategy_dd", {})
        except Exception:
            pass

    log.info("==============================================================")
    log.info("  BTC Hourly CT — OOS / FULL-PERIOD module validation%s",
             " [SMOKE]" if smoke else "")
    log.info("  Main FIXED: LL=104 SL=4.15 LS=68 SS=4.95 (Status ON)")
    log.info("  IS ref: %s", payload["is_reference"].get("baseline"))
    log.info("==============================================================")

    for period in periods:
        pres = payload["results"].get(period, {"baseline": None, "modules": {}})

        # ---- baseline (main only) ----
        if only_module is None or only_module == 0:
            cfg0 = _baseline_cfg(period)
            if not (from_csv or csv_for(cfg0).exists()):
                _apply_status(conn, None, manual_status)
            df0 = run_or_load(cfg0, conn, from_csv)
            # NOTE: the baseline is a micro-grid where main params DELIBERATELY vary
            # (Critical Rule 1); we pick the exact champion row afterwards.  Do NOT
            # apply the per-row main-fixed check here — that is only for module runs.
            if df0 is not None and not df0.empty:
                row = _pick_champion_row(df0)
                pres["baseline"] = _row_metrics(row)
                log.info("  [%s] BASELINE main-only: NP=%.2f intradayDD=%.2f tr=%d",
                         period, pres["baseline"]["net_profit"],
                         pres["baseline"]["max_intraday_drawdown"],
                         pres["baseline"]["total_trades"])
            else:
                log.warning("  [%s] baseline FAILED (no data)", period)
            payload["results"][period] = pres
            save_json(payload)

        # ---- each module (fixed at IS-optimized param) ----
        base_np = (pres.get("baseline") or {}).get("net_profit")
        base_mdd = (pres.get("baseline") or {}).get("max_intraday_drawdown")
        for idx, signal, axes in MODULES:
            if only_module is not None and only_module != idx:
                continue
            cfg = _module_cfg(period, idx, signal, axes, smoke)
            if not (from_csv or csv_for(cfg).exists()):
                _apply_status(conn, signal, manual_status)
            df = run_or_load(cfg, conn, from_csv)
            ent = {"signal": signal, "fixed_params": {nm: val for nm, val, _ in axes},
                   "combos": cfg.total_runs(), "rows": len(df) if df is not None else 0}
            if df is not None and not df.empty and _main_fixed_ok(df):
                row = _pick_fixed_row(df, axes)
                if row is not None:
                    ent.update(_row_metrics(row))
                    ent["actual_params"] = {nm: float(row[nm]) for nm, _, _ in axes if nm in row}
                    if base_np is not None:
                        ent["delta_np_vs_baseline"] = round(ent["net_profit"] - base_np, 2)
                        ent["delta_np_pct"] = round(
                            (ent["net_profit"] - base_np) / base_np * 100, 2) if base_np else None
                    if base_mdd is not None:
                        ent["delta_mdd_vs_baseline"] = round(
                            ent["max_intraday_drawdown"] - base_mdd, 2)
                    ent["valid"] = True
                    log.info("  [%s] M%d %s: NP=%.2f (%s%% vs base) intradayDD=%.2f tr=%d",
                             period, idx, ent["fixed_params"], ent["net_profit"],
                             ent.get("delta_np_pct"), ent["max_intraday_drawdown"],
                             ent["total_trades"])
                else:
                    ent["valid"] = False
                    log.warning("  [%s] M%d: fixed-param row not found", period, idx)
            else:
                ent["valid"] = False
                log.warning("  [%s] M%d: no valid data", period, idx)
            pres["modules"][str(idx)] = ent
            payload["results"][period] = pres
            save_json(payload)

        # ---- true Max Strategy Drawdown for the baseline (best-effort) ----
        if want_strategy_dd and not from_csv and (only_module is None or only_module == 0):
            sdd = _read_strategy_dd(conn, period)
            if sdd:
                payload["strategy_dd"][period] = sdd
                save_json(payload)

    # ---- per-module verdict flags (full + oos) ----
    full = payload["results"].get("full", {})
    oos = payload["results"].get("oos", {})
    fb = (full.get("baseline") or {})
    ob = (oos.get("baseline") or {})
    verdicts = {}
    for idx, signal, axes in MODULES:
        fm = (full.get("modules", {}) or {}).get(str(idx))
        om = (oos.get("modules", {}) or {}).get(str(idx))
        if not (fm and fm.get("valid")):
            continue
        v = {
            "improves_full_np": bool(fb and fm["net_profit"] > fb["net_profit"]),
            "reduces_full_mdd": bool(fb and abs(fm["max_intraday_drawdown"]) < abs(fb["max_intraday_drawdown"])),
        }
        if om and om.get("valid"):
            v["oos_profitable"] = bool(om["net_profit"] > 0)
            v["oos_beats_baseline"] = bool(ob and om["net_profit"] > ob["net_profit"])
        verdicts[str(idx)] = v
    payload["verdicts"] = verdicts

    # ---- teardown ----
    if not from_csv and only_module is None and not only_period:
        try:
            _apply_status(conn, None, manual_status)
            log.info("Teardown: all modules OFF")
        except Exception as e:
            log.warning("Teardown failed: %s", e)

    payload["run_finished_at"] = datetime.now().isoformat()
    payload["run_total_sec"] = round(time.time() - run_t0, 1)

    # ---- summary ----
    log.info("==============================================================")
    log.info("  OOS VALIDATION SUMMARY")
    if fb:
        log.info("  FULL baseline (main only): NP=%.2f intradayDD=%.2f tr=%d",
                 fb["net_profit"], fb["max_intraday_drawdown"], fb["total_trades"])
    if ob:
        log.info("  OOS  baseline (main only): NP=%.2f intradayDD=%.2f tr=%d  -> %s",
                 ob["net_profit"], ob["max_intraday_drawdown"], ob["total_trades"],
                 "PROFITABLE" if ob["net_profit"] > 0 else "LOSS")
    for idx, signal, axes in MODULES:
        fm = (full.get("modules", {}) or {}).get(str(idx))
        om = (oos.get("modules", {}) or {}).get(str(idx))
        if fm and fm.get("valid"):
            log.info("  M%d full NP=%.0f (%s%%) DD=%.0f | oos NP=%.0f  verdict=%s",
                     idx, fm["net_profit"], fm.get("delta_np_pct"),
                     fm["max_intraday_drawdown"],
                     om["net_profit"] if (om and om.get("valid")) else None,
                     verdicts.get(str(idx)))
    log.info("==============================================================")
    out = save_json(payload)
    print(f"\nDone -- results at: {out}")
    return 0


def _is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _auto_elevate():
    script = str(Path(__file__).resolve())
    workdir = str(Path(__file__).resolve().parent)
    extra = [a for a in sys.argv[1:] if a != "--_elevated"]
    quoted = [f'"{a}"' if " " in a else a for a in extra]
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
    ap = argparse.ArgumentParser(description="BTC Hourly CT OOS/full-period module validation")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--module", type=int, default=None, metavar="N",
                    help="Only config N (0=baseline, 1-6=module)")
    ap.add_argument("--period", choices=list(PERIODS.keys()), default=None,
                    help="Only this period")
    ap.add_argument("--manual-status", action="store_true")
    ap.add_argument("--no-strategy-dd", action="store_true",
                    help="Skip the perf-report true Max Strategy Drawdown read")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0

    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run(conn, args.from_csv, args.module, args.period,
               args.manual_status, args.smoke, not args.no_strategy_dd)


if __name__ == "__main__":
    sys.exit(main())
