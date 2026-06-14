"""
search_btc_ct_daily_exit_modules.py — Exit-module independent optimization on top of the
SFJ_15Dworkshop_lesson5_countertrend_LS_crypto BTC DAILY champion.

Main signal FIXED at champion (LL=49 SL=2.4 LS=123 SS=1.2, R1/R2 7-conv), Status ON.
Six exit modules tested INDEPENDENTLY: one module enabled at a time, exhaustive
optimization over its input range, max-NetProfit row selected, compared against
the same-day baseline (A00) and the reference baseline NP=$3,592.66 / MDD=-$398.78 / 16tr.
NOTE: BTC Daily NP is data-vintage dependent (R3 re-test gave $3,065/13tr) — A00
re-measures same-day so module deltas are computed against the current data.

Prior art: the SAME experiment on BNB/BTC/ETH Hourly (2026-06-13) found ALL 18
module tests hurt (-75% to -102%).  BTC Daily is the same rare-extreme regime
(13-16tr/7yr) — same outcome expected.

SPEED MODE (this version):
  - set_signal_statuses fast path: one-pass state read, only diffs toggled
  - reopen-verify SKIPPED by default (14 flawless toggles on BNB+BTC; the
    per-row main-param CSV check remains the ultimate proof) — re-enable
    with --verify-status
  - per-phase timestamps recorded in JSON (status_sec / optimize_sec / total_sec)

Modules (all <=5000 combos):
  M1 SFJ_15Dworkshop_lesson4_ATRstop                STP      0.1-100  s0.1   (1000)
  M2 SFJ_15Dworkshop_lesson9_1_TrailingStop         ATRSTP   0.1-100  s0.1   (1000)
  M3 SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit  EXITBAR  1-1000   s1     (1000)
  M4 SFJ_15Dworkshop_lesson11_3_high_volatility_exit DAYRANGE 0.01-10 s0.01  (1000)
  M5 QuantPass_PT_Exit                              PT_Base  0.001-1  s0.001 (1000)
  M6 RescueTeamExit                                 Length 20-600 s20 x std 3-6 s0.1 (930)

PREREQUISITES (one-time manual setup in MC64):
  - Workspace 20260101_SFJ_Bollinger_AI.wsp open, BTCUSDT HOT DAILY tab active
  - Main signal inputs set to 49 / 2.4 / 123 / 1.2, Status ON
  - All 6 module signals inserted on the chart, Status OFF
  - Workspace saved

CLI:
  py search_btc_ct_daily_exit_modules.py                  # A00 baseline + M1-M6 + teardown
  py search_btc_ct_daily_exit_modules.py --module 3       # only module 3
  py search_btc_ct_daily_exit_modules.py --module 0       # baseline A00 only
  py search_btc_ct_daily_exit_modules.py --smoke --module 1   # tiny end-to-end smoke test
  py search_btc_ct_daily_exit_modules.py --verify-status  # re-enable reopen-verify (slower)
  py search_btc_ct_daily_exit_modules.py --manual-status  # user toggles Status by hand
  py search_btc_ct_daily_exit_modules.py --from-csv       # re-analyze existing CSVs
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
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\btc_ct_daily_exit_modules_search")
INSAMPLE    = DateRange("2019/01/01", "2026/01/01")

# BTC Daily champion (R1/R2 7-conv) — set in MC64 manually
CHAMPION = {
    "LENGTH_LONG":  49.0,
    "STDDEV_LONG":  2.4,
    "LENGTH_SHORT": 123.0,
    "STDDEV_SHORT": 1.2,
}
# Reference baseline (R1/R2 stable; NP data-vintage dependent — A00 re-measures)
BASELINE_REF = {"net_profit": 3592.66, "max_drawdown": -398.78, "total_trades": 16}

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

PREFIX = "BTCCTDEX_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_ct_daily_exit_modules_{int(time.time())}.log"
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
        timeframe="daily",
        bar_period=1440,
        params=params,
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def _baseline_cfg() -> StrategyConfig:
    # 81-combo micro-grid around the champion: all 4 params vary (Critical Rule 1)
    return StrategyConfig(
        name=f"{PREFIX}A00_baseline",
        mc_signal_name=MAIN_SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
            ParamAxis("LENGTH_LONG",  48.0,  50.0,  1.0),
            ParamAxis("STDDEV_LONG",  2.35,  2.45,  0.05),
            ParamAxis("LENGTH_SHORT", 122.0, 124.0, 1.0),
            ParamAxis("STDDEV_SHORT", 1.15,  1.25,  0.05),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(cfg: StrategyConfig) -> Path:
    return OUTPUT_DIR / f"{cfg.name}_raw.csv"


def _apply_status(conn, active_module: Optional[str], manual: bool, verify: bool) -> float:
    """Set main ON (never touched), active module ON, others OFF.
    Returns elapsed seconds."""
    t0 = time.time()
    status_map = {name: (name == active_module) for name in ALL_MODULE_NAMES}
    if manual:
        print("\n" + "=" * 60)
        print("MANUAL STATUS SETUP REQUIRED (Format Objects > Signals):")
        print(f"  {MAIN_SIGNAL}: ON (inputs 49/2.4/123/1.2)")
        for name, want in status_map.items():
            print(f"  {name}: {'ON' if want else 'OFF'}")
        print("=" * 60)
        input("Set the checkboxes in MC64, click OK, then press Enter here...")
        return time.time() - t0
    mc.set_signal_statuses(conn, status_map, verify=verify, protected=[MAIN_SIGNAL])
    el = time.time() - t0
    log.info("[TIMING] status_apply=%.1fs (verify=%s)", el, verify)
    return el


def run_or_load(cfg: StrategyConfig, conn, from_csv: bool):
    csv_path = csv_for(cfg)
    if from_csv or csv_path.exists():
        if csv_path.exists():
            try:
                df = mc.load_results_csv(str(csv_path), cfg)
                log.info("Loaded %s: %d rows", cfg.name, len(df))
                return df, True
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


def _validate_module_df(df, cfg: StrategyConfig, smoke: bool):
    """Validate a module CSV. Returns (status, coverage) where status is:
      "invalid" — garbled (missing column, values out of declared range, or the
                  main signal not fixed at the champion) → discard.
      "partial" — values CLEAN and in range, main signal fixed, but a param does
                  not span >=90% of its declared range (MC64 sparse-trade export
                  truncated the high end) → ACCEPT best-NP from available rows.
      "valid"   — full coverage.
    coverage = {param: [observed_min, observed_max], ...} for reporting."""
    if df is None or df.empty:
        return "invalid", {}
    coverage = {}
    partial = False
    for p in cfg.params:
        if p.name not in df.columns:
            log.warning("  INVALID: param column '%s' missing (cols=%s)",
                        p.name, list(df.columns)[:12])
            return "invalid", {}
        col = pd.to_numeric(df[p.name], errors="coerce")
        lo = min(p.start, p.stop) - abs(p.step) * 2
        hi = max(p.start, p.stop) + abs(p.step) * 2
        if not col.between(lo, hi).all():
            # values OUTSIDE declared range = genuinely garbled (MCReport packing bug)
            log.warning("  INVALID: %s out of [%.4g,%.4g] got [%.4g,%.4g] — garbled",
                        p.name, lo, hi, col.min(), col.max())
            return "invalid", {}
        coverage[p.name] = [float(col.min()), float(col.max())]
        if not smoke:
            span = abs(p.stop - p.start)
            got = col.max() - col.min()
            if span > 0 and got < 0.9 * span:
                # CLEAN values but incomplete coverage — MC64 omitted sparse-trade
                # combos at a range boundary.  Usable but partial, not garbled.
                log.warning("  PARTIAL: %s covers [%.4g,%.4g] of declared [%.4g,%.4g] "
                            "(MC64 sparse-trade export truncation) — accepting best-NP "
                            "from available rows", p.name, col.min(), col.max(),
                            p.start, p.stop)
                partial = True
    main_cols = [c for c in CHAMPION if c in df.columns]
    if main_cols:
        for c in main_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            tol = max(1e-6, abs(CHAMPION[c]) * 1e-4)
            if not ((col - CHAMPION[c]).abs() <= tol).all():
                log.error("  INVALID: main param %s NOT fixed at %.6g "
                          "(got %.6g..%.6g) — Status/inputs setup is wrong!",
                          c, CHAMPION[c], col.min(), col.max())
                return "invalid", {}
        log.info("  Main-signal fixed-value check passed (%s)", main_cols)
    else:
        log.warning("  Main-signal columns not in CSV — fixed-value check skipped")
    return ("partial" if partial else "valid"), coverage


def _best_np_row(df) -> dict:
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    best = df.loc[pd.to_numeric(df["NetProfit"], errors="coerce").idxmax()]
    return best


def save_json(payload):
    name = ("SMOKE_" if payload.get("smoke") else "") + "final_params_btc_ct_daily_exit_modules.json"
    out = OUTPUT_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def run_search(conn, from_csv: bool, only_module: Optional[int],
               manual_status: bool, smoke: bool, verify_status: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_t0 = time.time()

    payload = {
        "strategy_main": MAIN_SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily",
        "champion_fixed": CHAMPION,
        "baseline_reference": BASELINE_REF,
        "baseline_measured": None,
        "mode": "independent (one module at a time)",
        "speed_mode": {"status_verify": verify_status,
                       "status_fast_path": "one-pass state read, diffs only"},
        "smoke": smoke,
        "modules": [],
        "run_started_at": datetime.now().isoformat(),
        "run_finished_at": None,
        "run_total_sec": None,
    }
    out_json = OUTPUT_DIR / (("SMOKE_" if smoke else "")
                             + "final_params_btc_ct_daily_exit_modules.json")
    if out_json.exists():
        try:
            with open(out_json, encoding="utf-8") as f:
                old = json.load(f)
            payload["baseline_measured"] = old.get("baseline_measured")
            payload["modules"] = old.get("modules", [])
        except Exception:
            pass

    log.info("==============================================================")
    log.info("  BTC Daily CT exit-module optimization%s [SPEED MODE]",
             " [SMOKE]" if smoke else "")
    log.info("  Main fixed: LL=49 SL=2.4 LS=123 SS=1.2 (Status ON)")
    log.info("  Baseline ref: NP=%.2f MDD=%.2f", BASELINE_REF["net_profit"],
             BASELINE_REF["max_drawdown"])
    log.info("==============================================================")

    baseline_np = BASELINE_REF["net_profit"]
    baseline_mdd = BASELINE_REF["max_drawdown"]

    # ── A00: same-day baseline (all modules OFF) ─────────────────────────────
    run_baseline = (only_module is None or only_module == 0)
    if run_baseline:
        a00_t0 = time.time()
        cfg0 = _baseline_cfg()
        status_sec = 0.0
        if not (from_csv or csv_for(cfg0).exists()):
            status_sec = _apply_status(conn, None, manual_status, verify_status)
        df0, _ = run_or_load(cfg0, conn, from_csv)
        if df0 is not None and not df0.empty:
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
                "status_sec": round(status_sec, 1),
                "total_sec": round(time.time() - a00_t0, 1),
                "timestamp": datetime.now().isoformat(),
            }
            log.info("A00 baseline: NP=%.2f MDD=%.2f (drift %+.3f%%) [%.0fs]",
                     baseline_np, baseline_mdd,
                     payload["baseline_measured"]["drift_vs_reference_pct"],
                     time.time() - a00_t0)
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
        mod_t0 = time.time()
        cfg = _module_cfg(idx, signal, axes, smoke)
        log.info("--- Module %d: %s (%d combos) [%s] ---",
                 idx, signal, cfg.total_runs(), datetime.now().strftime("%H:%M:%S"))

        status_sec = 0.0
        if not (from_csv or csv_for(cfg).exists()):
            status_sec = _apply_status(conn, signal, manual_status, verify_status)
        opt_t0 = time.time()
        df, _cached = run_or_load(cfg, conn, from_csv)
        optimize_sec = time.time() - opt_t0

        entry = {
            "module": idx,
            "signal": signal,
            "params_range": {nm: [s, e, st] for nm, s, e, st in axes},
            "combos": cfg.total_runs(),
            "rows": len(df) if df is not None else 0,
            "started_at": datetime.fromtimestamp(mod_t0).isoformat(),
            "finished_at": datetime.now().isoformat(),
            "status_sec": round(status_sec, 1),
            "optimize_sec": round(optimize_sec, 1),
            "total_sec": round(time.time() - mod_t0, 1),
        }
        status, coverage = _validate_module_df(df, cfg, smoke) if df is not None else ("invalid", {})
        if status in ("valid", "partial"):
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
                "coverage_status": status,
                "param_coverage": coverage,
            })
            log.info("  Module %d BEST%s: %s NP=%.2f (%+.2f%% vs baseline) "
                     "MDD=%.2f tr=%d Obj=%.0f [total %.0fs = status %.0fs + opt %.0fs]",
                     idx, " [PARTIAL coverage]" if status == "partial" else "",
                     best_params, np_, entry["delta_np_pct"],
                     mdd, entry["total_trades"], entry["objective"],
                     entry["total_sec"], status_sec, optimize_sec)
        else:
            entry["valid"] = False
            entry["coverage_status"] = "invalid"
            log.warning("  Module %d: NO VALID DATA (garbled/missing)", idx)

        payload["modules"] = [m for m in payload["modules"] if m.get("module") != idx]
        payload["modules"].append(entry)
        payload["modules"].sort(key=lambda m: m["module"])
        save_json(payload)

    # ── Teardown: all modules OFF ────────────────────────────────────────────
    if not from_csv and only_module is None:
        try:
            _apply_status(conn, None, manual_status, verify_status)
            log.info("Teardown: all modules OFF")
        except Exception as e:
            log.warning("Teardown status reset failed: %s", e)

    payload["run_finished_at"] = datetime.now().isoformat()
    payload["run_total_sec"] = round(time.time() - run_t0, 1)

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("==============================================================")
    log.info("  SUMMARY (baseline NP=%.2f MDD=%.2f)  total run %.1f min",
             baseline_np, baseline_mdd, (time.time() - run_t0) / 60)
    for m in payload["modules"]:
        if m.get("valid"):
            log.info("  M%d %-48s NP=%.0f (%+.2f%%) MDD=%.0f  [%.0fs]  %s",
                     m["module"], m["signal"], m["net_profit"],
                     m["delta_np_pct"], m["max_drawdown"],
                     m.get("total_sec", 0), m["best_params"])
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
        description="BTC Daily CT exit-module independent optimization (speed mode)")
    ap.add_argument("--from-csv", action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--module", type=int, default=None, metavar="N",
                    help="Run only module N (1-6); 0 = baseline A00 only")
    ap.add_argument("--manual-status", action="store_true",
                    help="Pause for manual Status checkbox setup instead of automation")
    ap.add_argument("--verify-status", action="store_true",
                    help="Re-enable reopen-verify after each status apply (slower, safer)")
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
                      args.manual_status, args.smoke, args.verify_status)


if __name__ == "__main__":
    sys.exit(main())
