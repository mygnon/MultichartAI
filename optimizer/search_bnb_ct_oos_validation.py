"""
search_bnb_ct_oos_validation.py — Full-period / out-of-sample (OOS) validation of the
already-optimized exit modules on top of the BNB Hourly CT champion.

GOAL (same as the BTC version): with the main signal and each module FIXED at their
already-found values, find which (if any) modules added to the main strategy
(1) improve NetProfit and reduce Max Drawdown over the FULL period, and
(2) do NOT lose money in the recent out-of-sample period.

>>> DATE-ISOLATION FIX (learned from the BTC run) <<<
MultiCharts IGNORES a signal's *Begin date* for restricting the backtest START —
the strategy always computes from the first loaded bar.  Only the *End date*
actually changes the backtest (verified on BTC: IS end 2026 differs from full end).
So we DO NOT use a 2026-start "oos" period (that silently returned a duplicate of
full on BTC).  Instead we measure two END-date periods on the SAME data in the SAME
session and derive OOS by difference:
    IS   = 2019/01/01 -> 2026/01/01   (end date works)
    FULL = 2019/01/01 -> data end     (end date works)
    OOS net profit (per config) = NP_full - NP_is
A module is OOS-safe if its OOS net contribution is >= 0 (ideally > baseline's).
True OOS-only Max Drawdown is not cleanly separable in MC64; we report full-period
Max Intraday Drawdown and whether the worst drawdown deepened from IS to FULL.

Main signal FIXED at Obj-max champion LL=122 SL=4.025 LS=29 SS=4.2 (Status ON).
Module params FIXED at their IS max-NP values (from bnb_ct_exit_modules_search):
  M1 ATRstop STP=5.7 | M2 TrailingStop ATRSTP=11.0 | M3 EntryBarsAfterExit EXITBAR=76
  M4 high_volatility DAYRANGE=4.68 | M5 PT_Exit PT_Base=0.065 | M6 RescueTeamExit Length=80 std=3.2

SPEED MODE: status fast-path (one-pass read, diffs only), no reopen-verify,
period-grouped so the date is set only twice total; per-config timestamps in JSON.

PREREQUISITES (one-time manual setup in MC64):
  - Workspace 20260101_SFJ_Bollinger_AI.wsp open with FULL data (incl. OOS after 2026/01/01)
  - BNBUSDT HOT HOURLY tab active
  - Main signal inputs set to 122 / 4.025 / 29 / 4.2, Status ON
  - All 6 module signals inserted on the chart, Status OFF
  - Workspace saved

CLI:
  py search_bnb_ct_oos_validation.py                 # IS + FULL, baseline + M1-M6 + teardown
  py search_bnb_ct_oos_validation.py --module 0      # baseline only (both periods)
  py search_bnb_ct_oos_validation.py --period full    # only the full period
  py search_bnb_ct_oos_validation.py --manual-status  # toggle Status by hand
  py search_bnb_ct_oos_validation.py --from-csv       # re-analyze existing CSVs
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
SYMBOL      = "BNBUSDT HOT"
MAIN_SIGNAL = "SFJ_15Dworkshop_lesson5_countertrend_LS_crypto"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\bnb_ct_oos_validation_search")

# BNB Hourly Obj-max champion — set in MC64 manually, Status ON
CHAMPION = {
    "LENGTH_LONG":  122.0,
    "STDDEV_LONG":  4.025,
    "LENGTH_SHORT": 29.0,
    "STDDEV_SHORT": 4.2,
}

# Two END-date periods on the SAME data (begin-date does NOT restrict in MC64).
# OOS is derived as FULL - IS.
PERIODS = {
    "is":   DateRange("2019/01/01", "2026/01/01"),   # in-sample (end works)
    "full": DateRange("2019/01/01", "2027/01/01"),   # 2019 -> data end
}
PERIOD_ORDER = ["is", "full"]   # group by period -> date set only twice total

# (index, signal, [(param, fixed_value, step), ...]) — module FIXED at IS max-NP value
MODULES: List[Tuple[int, str, List[Tuple[str, float, float]]]] = [
    (1, "SFJ_15Dworkshop_lesson4_ATRstop",                 [("STP",      5.7,   0.1)]),
    (2, "SFJ_15Dworkshop_lesson9_1_TrailingStop",          [("ATRSTP",   11.0,  0.1)]),
    (3, "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit",   [("EXITBAR",  7.0,   1.0)]),
    (4, "SFJ_15Dworkshop_lesson11_3_high_volatility_exit", [("DAYRANGE", 3.3,   0.01)]),
    (5, "QuantPass_PT_Exit",                               [("PT_Base",  0.065, 0.001)]),
    (6, "RescueTeamExit",                                  [("Length",   80.0,  20.0),
                                                            ("std",      3.2,   0.1)]),
]
ALL_MODULE_NAMES = [m[1] for m in MODULES]

GEN_LO, GEN_HI = 0.001, 100000.0

PREFIX = "BNBCTOOS_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_ct_oos_validation_{int(time.time())}.log"
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


def _bracket(value, step, smoke):
    n = 1 if smoke else 2
    start = max(GEN_LO, round(value - n * step, 8))
    stop = min(GEN_HI, round(value + n * step, 8))
    if stop <= start:
        stop = start + step
    return (start, stop, step)


def _baseline_cfg(period: str) -> StrategyConfig:
    return StrategyConfig(
        name=f"{PREFIX}{period}_A00_baseline",
        mc_signal_name=MAIN_SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("LENGTH_LONG",  121.0, 123.0, 1.0),
            ParamAxis("STDDEV_LONG",  4.0,   4.05,  0.025),
            ParamAxis("LENGTH_SHORT", 28.0,  30.0,  1.0),
            ParamAxis("STDDEV_SHORT", 4.175, 4.225, 0.025),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=PERIODS[period],
    )


def _module_cfg(period: str, idx: int, signal: str, axes, smoke: bool) -> StrategyConfig:
    params = [ParamAxis(nm, *_bracket(val, step, smoke)) for (nm, val, step) in axes]
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
        print(f"  {MAIN_SIGNAL}: ON (inputs 122/4.025/29/4.2)")
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


def _pick_champion_row(df):
    m = df
    for nm, v in CHAMPION.items():
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    if not m.empty:
        return m.iloc[0]
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    return df.loc[pd.to_numeric(df["NetProfit"], errors="coerce").idxmax()]


def _pick_fixed_row(df, axes):
    m = df
    for (nm, val, step) in axes:
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - val).abs() < step * 0.5]
    if not m.empty:
        return m.iloc[0]
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


def save_json(payload):
    out = OUTPUT_DIR / "final_params_bnb_ct_oos_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def run(conn, from_csv, only_module, only_period, manual_status, smoke):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_t0 = time.time()
    periods = [only_period] if only_period else PERIOD_ORDER

    payload = {
        "strategy_main": MAIN_SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "champion_fixed": CHAMPION,
        "module_fixed_params": {idx: {nm: val for nm, val, _ in axes}
                                for idx, sig, axes in MODULES},
        "periods": {k: [v.from_date, v.to_date] for k, v in PERIODS.items()},
        "method": "IS vs FULL via END date only (MC64 ignores begin-date); OOS = FULL - IS",
        "results": {},
        "oos_derived": {},
        "verdicts": {},
        "smoke": smoke,
        "run_started_at": datetime.now().isoformat(),
        "run_finished_at": None,
        "run_total_sec": None,
    }
    out_json = OUTPUT_DIR / "final_params_bnb_ct_oos_validation.json"
    if out_json.exists():
        try:
            with open(out_json, encoding="utf-8") as f:
                old = json.load(f)
            payload["results"] = old.get("results", {})
        except Exception:
            pass

    log.info("==============================================================")
    log.info("  BNB Hourly CT — OOS / FULL-PERIOD module validation%s [SPEED]",
             " [SMOKE]" if smoke else "")
    log.info("  Main FIXED: LL=122 SL=4.025 LS=29 SS=4.2 (Status ON)")
    log.info("  IS=2019->2026/01  FULL=2019->dataend  OOS=FULL-IS")
    log.info("==============================================================")

    def _eval(period, cfg, active_signal):
        """Run/load one config for one period; return (entry, df).  Timed."""
        t0 = time.time()
        status_sec = 0.0
        if not (from_csv or csv_for(cfg).exists()):
            status_sec = _apply_status(conn, active_signal, manual_status)
        opt_t0 = time.time()
        df = run_or_load(cfg, conn, from_csv)
        ent = {"started_at": datetime.fromtimestamp(t0).isoformat(),
               "finished_at": datetime.now().isoformat(),
               "status_sec": round(status_sec, 1),
               "optimize_sec": round(time.time() - opt_t0, 1),
               "total_sec": round(time.time() - t0, 1),
               "rows": len(df) if df is not None else 0}
        return ent, (df if (df is not None and not df.empty) else None)

    for period in periods:
        pres = payload["results"].get(period, {"baseline": None, "modules": {}})

        if only_module is None or only_module == 0:
            cfg0 = _baseline_cfg(period)
            ent, df0 = _eval(period, cfg0, None)
            if df0 is not None:
                row = _pick_champion_row(df0)   # main varies by design -> pick champion
                ent.update(_row_metrics(row)); ent["valid"] = True
                pres["baseline"] = ent
                log.info("  [%s] BASELINE: NP=%.2f intradayDD=%.2f tr=%d [%.0fs]",
                         period, ent["net_profit"], ent["max_intraday_drawdown"],
                         ent["total_trades"], ent["total_sec"])
            else:
                ent["valid"] = False; pres["baseline"] = ent
                log.warning("  [%s] baseline FAILED", period)
            payload["results"][period] = pres
            save_json(payload)

        base_np = (pres.get("baseline") or {}).get("net_profit")
        base_mdd = (pres.get("baseline") or {}).get("max_intraday_drawdown")
        for idx, signal, axes in MODULES:
            if only_module is not None and only_module != idx:
                continue
            cfg = _module_cfg(period, idx, signal, axes, smoke)
            ent, df = _eval(period, cfg, signal)
            ent["signal"] = signal
            ent["fixed_params"] = {nm: val for nm, val, _ in axes}
            ent["combos"] = cfg.total_runs()
            if df is not None and _main_fixed_ok(df):
                row = _pick_fixed_row(df, axes)
                if row is not None:
                    ent.update(_row_metrics(row))
                    ent["actual_params"] = {nm: float(row[nm]) for nm, _, _ in axes if nm in row}
                    if base_np is not None:
                        ent["delta_np_vs_baseline"] = round(ent["net_profit"] - base_np, 2)
                        ent["delta_np_pct"] = round((ent["net_profit"] - base_np) / base_np * 100, 2) if base_np else None
                    if base_mdd is not None:
                        ent["delta_mdd_vs_baseline"] = round(ent["max_intraday_drawdown"] - base_mdd, 2)
                    ent["valid"] = True
                    log.info("  [%s] M%d %s: NP=%.2f (%s%%) DD=%.2f tr=%d [%.0fs]",
                             period, idx, ent["fixed_params"], ent["net_profit"],
                             ent.get("delta_np_pct"), ent["max_intraday_drawdown"],
                             ent["total_trades"], ent["total_sec"])
                else:
                    ent["valid"] = False
                    log.warning("  [%s] M%d: fixed-param row not found", period, idx)
            else:
                ent["valid"] = False
                log.warning("  [%s] M%d: no valid data / main not fixed", period, idx)
            pres["modules"][str(idx)] = ent
            payload["results"][period] = pres
            save_json(payload)

    # ---- derive OOS = FULL - IS, and verdicts ----
    is_r = payload["results"].get("is", {})
    full_r = payload["results"].get("full", {})

    def _get(d, key, mod):
        x = (d.get(key) if mod is None else (d.get("modules", {}) or {}).get(str(mod)))
        return x if (x and x.get("valid")) else None

    def _np(d, key=None, mod=None):
        x = _get(d, key, mod); return x.get("net_profit") if x else None

    def _dd(d, key=None, mod=None):
        x = _get(d, key, mod); return x.get("max_intraday_drawdown") if x else None

    b_is_np, b_full_np = _np(is_r, "baseline"), _np(full_r, "baseline")
    b_oos_np = (b_full_np - b_is_np) if (b_is_np is not None and b_full_np is not None) else None
    bf_dd, bi_dd = _dd(full_r, "baseline"), _dd(is_r, "baseline")
    payload["oos_derived"]["baseline"] = {
        "is_np": b_is_np, "full_np": b_full_np,
        "oos_np": round(b_oos_np, 2) if b_oos_np is not None else None,
        "is_dd": bi_dd, "full_dd": bf_dd,
        "dd_deepened_in_oos": bool(bf_dd is not None and bi_dd is not None and abs(bf_dd) > abs(bi_dd)),
    }
    for idx, signal, axes in MODULES:
        m_is_np, m_full_np = _np(is_r, None, idx), _np(full_r, None, idx)
        m_oos_np = (m_full_np - m_is_np) if (m_is_np is not None and m_full_np is not None) else None
        m_full_dd = _dd(full_r, None, idx)
        payload["oos_derived"][str(idx)] = {
            "is_np": m_is_np, "full_np": m_full_np,
            "oos_np": round(m_oos_np, 2) if m_oos_np is not None else None,
            "full_dd": m_full_dd,
        }
        v = {}
        if b_full_np is not None and m_full_np is not None:
            v["improves_full_np"] = bool(m_full_np > b_full_np)
        if bf_dd is not None and m_full_dd is not None:
            v["reduces_full_mdd"] = bool(abs(m_full_dd) < abs(bf_dd))
        if m_oos_np is not None:
            v["oos_np_positive"] = bool(m_oos_np > 0)
        if b_oos_np is not None and m_oos_np is not None:
            v["oos_beats_baseline"] = bool(m_oos_np > b_oos_np)
        payload["verdicts"][str(idx)] = v

    if not from_csv and only_module is None and not only_period:
        try:
            _apply_status(conn, None, manual_status)
            log.info("Teardown: all modules OFF")
        except Exception as e:
            log.warning("Teardown failed: %s", e)

    payload["run_finished_at"] = datetime.now().isoformat()
    payload["run_total_sec"] = round(time.time() - run_t0, 1)

    log.info("==============================================================")
    log.info("  BNB OOS VALIDATION SUMMARY  (total %.1f min)", (time.time() - run_t0) / 60)
    bd = payload["oos_derived"]["baseline"]
    log.info("  BASELINE main-only: IS NP=%s | FULL NP=%s | OOS(=F-IS) NP=%s -> %s | FULL DD=%s (deepened=%s)",
             bd["is_np"], bd["full_np"], bd["oos_np"],
             ("PROFIT" if (bd["oos_np"] or 0) > 0 else "LOSS") if bd["oos_np"] is not None else "?",
             bd["full_dd"], bd["dd_deepened_in_oos"])
    for idx, signal, axes in MODULES:
        od = payload["oos_derived"].get(str(idx), {})
        log.info("  M%d full NP=%s (oos=%s) DD=%s | verdict=%s",
                 idx, od.get("full_np"), od.get("oos_np"), od.get("full_dd"),
                 payload["verdicts"].get(str(idx)))
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
    ap = argparse.ArgumentParser(description="BNB Hourly CT OOS/full-period module validation")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--module", type=int, default=None, metavar="N",
                    help="Only config N (0=baseline, 1-6=module)")
    ap.add_argument("--period", choices=list(PERIODS.keys()), default=None)
    ap.add_argument("--manual-status", action="store_true")
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
               args.manual_status, args.smoke)


if __name__ == "__main__":
    sys.exit(main())
