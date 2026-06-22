"""
search_eth_qpatrsi_exit_modules.py — Exit-module independent optimization on top of the
QuantPassRSI ETH Hourly champion.

Main signal FIXED at Obj-max champion (Len=12 RSI_Gap=15), Status ON.
Six exit modules tested INDEPENDENTLY: one module enabled at a time, exhaustive
optimization over its input range, max-NetProfit row selected, compared against
the same-day baseline (A00) and the reference baseline NP=$3,408 / MDD=-$704 (R1 Obj-max ETH Hourly QuantPassRSI champion).

Modules (all <=5000 combos):
  M1 SFJ_15Dworkshop_lesson4_ATRstop                STP      0.1-100  s0.1   (1000)
  M2 SFJ_15Dworkshop_lesson9_1_TrailingStop         ATRSTP   0.1-100  s0.1   (1000)
  M3 SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit  EXITBAR  1-1000   s1     (1000)
  M4 SFJ_15Dworkshop_lesson11_3_high_volatility_exit DAYRANGE 0.01-10 s0.01  (1000)
  M5 QuantPass_PT_Exit                              PT_Base  0.001-1  s0.001 (1000)
  M6 RescueTeamExit                                 Length 20-600 s20 x std 3-6 s0.1 (930)

PREREQUISITES (one-time manual setup in MC64):
  - Workspace 20260101_QuantPassRSI_AI.wsp open, ETHUSDT HOT 60-Minute tab active
  - Main signal QuantPassRSI inputs set to Len=12 RSI_Gap=15, Status ON
  - All 6 module signals inserted on the chart, Status OFF
  - Workspace saved

CLI:
  py search_eth_qpatrsi_exit_modules.py                  # A00 baseline + M1-M6 + teardown
  py search_eth_qpatrsi_exit_modules.py --module 3       # only module 3 (status set for it)
  py search_eth_qpatrsi_exit_modules.py --module 0       # baseline A00 only
  py search_eth_qpatrsi_exit_modules.py --smoke --module 1   # tiny end-to-end smoke test
  py search_eth_qpatrsi_exit_modules.py --manual-status  # user toggles Status by hand
  py search_eth_qpatrsi_exit_modules.py --from-csv       # re-analyze existing CSVs
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


WORKSPACE   = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_QuantPassRSI_AI.wsp"
SYMBOL      = "ETHUSDT HOT"
MAIN_SIGNAL = "QuantPassRSI"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\eth_qpatrsi_exit_modules_search")
INSAMPLE    = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE    = ("2022/01/01", "2026/01/01")

# ETH Hourly QuantPassRSI Obj-max champion (R1 ceiling) — main signal set to these in MC64 (Format Objects)
CHAMPION = {
    "Len":     12.0,
    "RSI_Gap": 15.0,
}
# Reference baseline (R1 Obj-max ETH Hourly QuantPassRSI champion, IS 2022/01-2026/01)
BASELINE_REF = {"net_profit": 3407.61, "max_drawdown": -704.0, "total_trades": 22}

# (index, signal_name, [(param, start, stop, step), ...])
MODULES: List[Tuple[int, str, List[Tuple[str, float, float, float]]]] = [
    (1, "SFJ_15Dworkshop_lesson4_ATRstop",                 [("STP",      0.1,   100.0, 0.1)]),
    (2, "SFJ_15Dworkshop_lesson9_1_TrailingStop",          [("ATRSTP",   0.1,   100.0, 0.1)]),
    (3, "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit",   [("EXITBAR",  1.0,   1000.0, 1.0)]),
    (4, "SFJ_15Dworkshop_lesson11_3_high_volatility_exit", [("DAYRANGE", 0.01,  10.0,  0.01)]),
    (5, "QuantPass_PT_Exit",                               [("PT_Base",  0.001, 1.0,   0.001)]),
    (6, "RescueTeamExit",                                  [("Length",   20.0,  600.0, 20.0),
                                                            ("std",      3.0,   6.0,   0.1)]),
]
ALL_MODULE_NAMES = [m[1] for m in MODULES]

PREFIX = "ETHRSIEX_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_eth_qpatrsi_exit_modules_{int(time.time())}.log"
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


def _module_cfg(idx: int, signal: str, axes, smoke: bool) -> StrategyConfig:
    params = []
    for (nm, s, e, st) in axes:
        if smoke:
            e = min(e, s + 2 * st)   # 3 values per axis
        params.append(ParamAxis(nm, s, e, st))
    combos = 1
    for p in params:
        combos *= n_vals(p.start, p.stop, p.step)
    if combos > 5000:
        log.warning("Module %d (%s): %d combos EXCEEDS 5000!", idx, signal, combos)
    return StrategyConfig(
        name=f"{'SMOKE_' if smoke else ''}{PREFIX}M{idx}_{signal[:30]}",
        mc_signal_name=signal,
        timeframe="hourly",
        bar_period=60,
        params=params,
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def _baseline_cfg() -> StrategyConfig:
    # 9-combo micro-grid around the champion: both params vary (Critical Rule 1)
    return StrategyConfig(
        name=f"{PREFIX}A00_baseline",
        mc_signal_name=MAIN_SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("Len",     11.0, 13.0, 1.0),
            ParamAxis("RSI_Gap", 14.0, 16.0, 1.0),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(cfg: StrategyConfig) -> Path:
    return OUTPUT_DIR / f"{cfg.name}_raw.csv"


def _apply_status(conn, active_module: Optional[str], manual: bool):
    """Set main ON (never touched), active module ON, all other modules OFF."""
    status_map = {name: (name == active_module) for name in ALL_MODULE_NAMES}
    if manual:
        print("\n" + "=" * 60)
        print("MANUAL STATUS SETUP REQUIRED (Format Objects > Signals):")
        print(f"  {MAIN_SIGNAL}: ON (Len=12 RSI_Gap=15)")
        for name, want in status_map.items():
            print(f"  {name}: {'ON' if want else 'OFF'}")
        print("=" * 60)
        input("Set the checkboxes in MC64, click OK, then press Enter here...")
        return
    mc.set_signal_statuses(conn, status_map, verify=True, protected=[MAIN_SIGNAL])


def run_or_load(cfg: StrategyConfig, conn, from_csv: bool):
    csv_path = csv_for(cfg)
    if from_csv or csv_path.exists():
        if csv_path.exists():
            try:
                df = mc.load_results_csv(str(csv_path), cfg)
                log.info("Loaded %s: %d rows", cfg.name, len(df))
                return df, True   # loaded from cache — status was not (re)applied
            except Exception as e:
                log.warning("Could not load %s: %s", csv_path, e)
        else:
            log.warning("No CSV for %s", cfg.name)
        return None, True
    log.info("=== Starting %s (%d combos) ===", cfg.name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done %.1f min -- %s", (time.time() - t0) / 60, Path(raw_csv).name)
        return mc.load_results_csv(raw_csv, cfg), False
    except Exception as e:
        log.error("  FAILED: %s", e, exc_info=True)
        return None, False


def _validate_module_df(df, cfg: StrategyConfig, smoke: bool) -> bool:
    """Range check + range-span check on module params, plus per-row proof that
    the main signal stayed fixed at the champion values."""
    if df is None or df.empty:
        return False
    for p in cfg.params:
        if p.name not in df.columns:
            log.warning("  INVALID: param column '%s' missing (cols=%s)",
                        p.name, list(df.columns)[:12])
            return False
        col = pd.to_numeric(df[p.name], errors="coerce")
        lo = min(p.start, p.stop) - abs(p.step) * 2
        hi = max(p.start, p.stop) + abs(p.step) * 2
        if not col.between(lo, hi).all():
            log.warning("  INVALID: %s out of [%.4g,%.4g] got [%.4g,%.4g]",
                        p.name, lo, hi, col.min(), col.max())
            return False
        # Span check: a short span is EITHER a clean sparse-trade TRUNCATION (MC drops the
        # high-param rows that make 0 trades — still usable, pick best-NP from the exported
        # subset) OR an MCReport packing GARBLE (columns shifted). Distinguish by the start:
        # a clean truncation still begins at the declared minimum; garble does not.
        if not smoke:
            span = abs(p.stop - p.start)
            got = col.max() - col.min()
            if span > 0 and got < 0.9 * span:
                lo_decl = min(p.start, p.stop)
                if abs(col.min() - lo_decl) <= abs(p.step) * 1.5:
                    log.warning("  PARTIAL: %s exported [%.4g,%.4g] of declared [%.4g,%.4g] "
                                "(sparse-trade truncation — accepting best-NP from exported subset)",
                                p.name, col.min(), col.max(), p.start, p.stop)
                else:
                    log.warning("  INVALID: %s span %.4g < 90%% of declared %.4g and start "
                                "%.4g != declared %.4g (garbled export)",
                                p.name, got, span, col.min(), lo_decl)
                    return False
    # Main-signal fixed-value proof (columns present only if MC exports them)
    main_cols = [c for c in CHAMPION if c in df.columns]
    if main_cols:
        for c in main_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            tol = max(1e-6, abs(CHAMPION[c]) * 1e-4)
            if not ((col - CHAMPION[c]).abs() <= tol).all():
                log.error("  INVALID: main param %s NOT fixed at %.6g "
                          "(got %.6g..%.6g) — Status/inputs setup is wrong!",
                          c, CHAMPION[c], col.min(), col.max())
                return False
        log.info("  Main-signal fixed-value check passed (%s)", main_cols)
    else:
        log.warning("  Main-signal columns not in CSV — fixed-value check skipped")
    return True


def _best_np_row(df) -> dict:
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    best = df.loc[pd.to_numeric(df["NetProfit"], errors="coerce").idxmax()]
    return best


def save_json(payload):
    name = ("SMOKE_" if payload.get("smoke") else "") + "final_params_eth_qpatrsi_exit_modules.json"
    out = OUTPUT_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def run_search(conn, from_csv: bool, only_module: Optional[int],
               manual_status: bool, smoke: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "strategy_main": MAIN_SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "champion_fixed": CHAMPION,
        "baseline_reference": BASELINE_REF,
        "baseline_measured": None,
        "mode": "independent (one module at a time)",
        "smoke": smoke,
        "modules": [],
        "timestamp": datetime.now().isoformat(),
    }
    # Resume from existing JSON if present (keeps prior module results)
    out_json = OUTPUT_DIR / (("SMOKE_" if smoke else "")
                             + "final_params_eth_qpatrsi_exit_modules.json")
    if out_json.exists():
        try:
            with open(out_json, encoding="utf-8") as f:
                old = json.load(f)
            payload["baseline_measured"] = old.get("baseline_measured")
            payload["modules"] = old.get("modules", [])
        except Exception:
            pass

    log.info("==============================================================")
    log.info("  ETH Hourly QuantPassRSI exit-module optimization%s", " [SMOKE]" if smoke else "")
    log.info("  Main fixed: Len=12 RSI_Gap=15 (ON); IS 2022/01-2026/01 (chart-trimmed)")
    log.info("  Baseline ref: NP=%.2f MDD=%.2f", BASELINE_REF["net_profit"],
             BASELINE_REF["max_drawdown"])
    log.info("==============================================================")

    baseline_np = BASELINE_REF["net_profit"]
    baseline_mdd = BASELINE_REF["max_drawdown"]

    # ── Switch to the CHART window, then trim the chart to the IS window ──
    #    connect() picks the largest-area MC window, which can be the Study Editor when
    #    it is open — so ALWAYS ensure_chart_ready first (it matches the workspace stem in
    #    the chart title, not the Study Editor) before the Format-Instruments right-click.
    #    3-attempt retry guards against a transient right-click miss on the graph panel.
    if not from_csv and conn is not None:
        log.info("Switching to chart window + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _baseline_cfg())   # focus the CHART (not Study Editor)
                mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
                ok = True
                break
            except Exception as e:
                log.warning("  chart-trim attempt %d failed: %s", _try + 1, e)
                try:
                    mc._close_optimization_report()
                except Exception:
                    pass
                time.sleep(1.2)
        if not ok:
            log.error("  set_instrument_data_range FAILED after 3 tries — aborting")
            return 1
        log.info("Chart trimmed (verify leftmost ~2022/01, rightmost ~2026/01).")

    # ── A00: same-day baseline (all modules OFF) ─────────────────────────────
    run_baseline = (only_module is None or only_module == 0)
    if run_baseline:
        cfg0 = _baseline_cfg()
        if not (from_csv or csv_for(cfg0).exists()):
            _apply_status(conn, None, manual_status)
        df0, _ = run_or_load(cfg0, conn, from_csv)
        if df0 is not None and not df0.empty:
            # Exact champion row if present, else NP-max of the micro-grid
            m = df0
            for nm, v in CHAMPION.items():
                if nm in m.columns:
                    m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
            row = m.iloc[0] if not m.empty else _best_np_row(df0)
            baseline_np = float(row["NetProfit"])
            baseline_mdd = float(row["MaxDrawdown"])
            payload["baseline_measured"] = {
                "net_profit": baseline_np,
                "max_drawdown": baseline_mdd,
                "total_trades": int(row["TotalTrades"]),
                "exact_champion_row": not m.empty,
                "drift_vs_reference_pct": round(
                    (baseline_np - BASELINE_REF["net_profit"])
                    / BASELINE_REF["net_profit"] * 100, 3),
                "timestamp": datetime.now().isoformat(),
            }
            log.info("A00 baseline: NP=%.2f MDD=%.2f (drift %+.3f%%)",
                     baseline_np, baseline_mdd,
                     payload["baseline_measured"]["drift_vs_reference_pct"])
            # ABORT if the baseline is far from the IS champion — means the chart is NOT on the
            # IS window (e.g. left on FULL range by a prior OOS run; the Data-Range trim did not
            # actually re-restrict the backtest). Running modules here would be on the wrong period.
            _drift = abs(payload["baseline_measured"]["drift_vs_reference_pct"])
            if _drift > 10.0:
                log.error("  A00 baseline drift %.1f%% > 10%% — chart is NOT on the IS window "
                          "(2022/01-2026/01). ABORTING. Manually set Format Instruments Data Range "
                          "to 2022/01/01-2026/01/01, verify the chart rightmost bar ~2026/01, then re-run.",
                          _drift)
                save_json(payload)
                return 1
        else:
            log.warning("A00 baseline failed — module deltas will use the reference")
        save_json(payload)
    elif payload.get("baseline_measured"):
        baseline_np = payload["baseline_measured"]["net_profit"]
        baseline_mdd = payload["baseline_measured"]["max_drawdown"]

    # ── M1-M6: one module at a time ──────────────────────────────────────────
    for idx, signal, axes in MODULES:
        if only_module is not None and only_module != idx:
            continue
        cfg = _module_cfg(idx, signal, axes, smoke)
        log.info("--- Module %d: %s (%d combos) ---", idx, signal, cfg.total_runs())

        if not (from_csv or csv_for(cfg).exists()):
            _apply_status(conn, signal, manual_status)
        df, _cached = run_or_load(cfg, conn, from_csv)

        entry = {
            "module": idx,
            "signal": signal,
            "params_range": {nm: [s, e, st] for nm, s, e, st in axes},
            "combos": cfg.total_runs(),
            "rows": len(df) if df is not None else 0,
            "timestamp": datetime.now().isoformat(),
        }
        if df is not None and _validate_module_df(df, cfg, smoke):
            best = _best_np_row(df)
            best_params = {p.name: float(best[p.name]) for p in cfg.params}
            np_ = float(best["NetProfit"])
            mdd = float(best["MaxDrawdown"])
            entry.update({
                "best_params": best_params,
                "net_profit": np_,
                "max_drawdown": mdd,
                "objective": float(best["Objective"]),
                "total_trades": int(best["TotalTrades"]),
                "delta_np_vs_baseline": round(np_ - baseline_np, 2),
                "delta_np_pct": round((np_ - baseline_np) / baseline_np * 100, 2),
                "delta_mdd_vs_baseline": round(mdd - baseline_mdd, 2),
                "valid": True,
            })
            log.info("  Module %d BEST: %s NP=%.2f (%+.2f%% vs baseline) "
                     "MDD=%.2f tr=%d Obj=%.0f",
                     idx, best_params, np_, entry["delta_np_pct"],
                     mdd, entry["total_trades"], entry["objective"])
        else:
            entry["valid"] = False
            log.warning("  Module %d: NO VALID DATA", idx)

        # Replace any previous entry for this module
        payload["modules"] = [m for m in payload["modules"] if m.get("module") != idx]
        payload["modules"].append(entry)
        payload["modules"].sort(key=lambda m: m["module"])
        save_json(payload)

    # ── Teardown: all modules OFF ────────────────────────────────────────────
    if not from_csv and only_module is None:
        try:
            _apply_status(conn, None, manual_status)
            log.info("Teardown: all modules OFF")
        except Exception as e:
            log.warning("Teardown status reset failed: %s", e)

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("==============================================================")
    log.info("  SUMMARY (baseline NP=%.2f MDD=%.2f)", baseline_np, baseline_mdd)
    for m in payload["modules"]:
        if m.get("valid"):
            log.info("  M%d %-48s NP=%.0f (%+.2f%%) MDD=%.0f  %s",
                     m["module"], m["signal"], m["net_profit"],
                     m["delta_np_pct"], m["max_drawdown"], m["best_params"])
        else:
            log.info("  M%d %-48s FAILED", m["module"], m["signal"])
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
        description="ETH Hourly QuantPassRSI exit-module independent optimization")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--module", type=int, default=None, metavar="N",
                    help="Run only module N (1-6); 0 = baseline A00 only")
    ap.add_argument("--manual-status", action="store_true",
                    help="Pause for manual Status checkbox setup instead of automation")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny 3-combo-per-axis end-to-end test (SMOKE_ prefix)")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0

    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run_search(conn, args.from_csv, args.module,
                      args.manual_status, args.smoke)


if __name__ == "__main__":
    sys.exit(main())
