"""
search_btc_hunter2_oos_validation.py — FULL-PERIOD (incl. OOS) validation of the IS-optimized
HUNTER2 exit modules on BTCUSDT HOT Hourly.

Main signal SFJ_HUNTER2_crypto FIXED at Obj-max champion (LEN_L=225 LEN_S=825 ATR_multiplier_L=4.25 ATR_multiplier_S=1.75), Status ON.
Each of the 6 exit modules is FIXED at its IS-best param (from
results/btc_hunter2_exit_modules_search) and re-tested over the FULL range 2021/03/01-2026/06/10
(chart-trimmed).  A module PASSES only if, on the full period, it BOTH increases NetProfit AND
reduces the Max (Strategy) Drawdown vs the main-only full-period baseline (A00).

IS-best module params (from the exit-module optimization, IS 2022/01-2026/01):
  M1 ATRstop            STP=9.2
  M2 TrailingStop       ATRSTP=44.4
  M3 EntryBarsAfterExit EXITBAR=131
  M4 high_volatility    DAYRANGE=6.21
  M5 QuantPass_PT_Exit  PT_Base=0.671
  M6 RescueTeamExit     Length=280, std=3.5

Each module is run as a 3-value (per axis) micro-grid centred on its fixed value (Critical Rule 1:
all params vary), and the EXACT IS-best row is picked.

PREREQUISITES (MC64, run AS ADMINISTRATOR):
  - Workspace 20260101_SFJ_HUNTER_AI.wsp open, full data to 2026/06/10
  - BTCUSDT HOT 60-Minute chart tab ACTIVE and VISIBLE
  - Main signal SFJ_HUNTER2_crypto inputs = LEN_L 225 LEN_S 825 ATR_multiplier_L 4.25 ATR_multiplier_S 1.75, Status ON
  - All 6 module signals inserted on the chart, Status OFF; workspace saved

CLI:
  py search_btc_hunter2_oos_validation.py                 # A00 full baseline + M1-M6 + teardown
  py search_btc_hunter2_oos_validation.py --module 6      # only module 6
  py search_btc_hunter2_oos_validation.py --manual-status # toggle Status by hand
  py search_btc_hunter2_oos_validation.py --from-csv      # re-analyze existing CSVs
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


WORKSPACE   = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_HUNTER_AI.wsp"
SYMBOL      = "BTCUSDT HOT"
MAIN_SIGNAL = "SFJ_HUNTER2_crypto"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\btc_hunter2_oos_validation_search")
INSAMPLE    = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
FULL_RANGE  = ("2021/03/01", "2026/06/10")

CHAMPION = {"LEN_L": 225.0, "LEN_S": 825.0, "ATR_multiplier_L": 4.25, "ATR_multiplier_S": 1.75}

# (index, signal, [(param, fixed_value, step, lo, hi)])  — micro-grid = fixed +/- 1 step
MODULES: List[Tuple[int, str, List[Tuple[str, float, float, float, float]]]] = [
    (1, "SFJ_15Dworkshop_lesson4_ATRstop",                 [("STP",      9.2,   0.1,   0.1,   100.0)]),
    (2, "SFJ_15Dworkshop_lesson9_1_TrailingStop",          [("ATRSTP",   44.4,  0.1,   0.1,   100.0)]),
    (3, "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit",   [("EXITBAR",  131.0, 1.0,   1.0,   1000.0)]),
    (4, "SFJ_15Dworkshop_lesson11_3_high_volatility_exit", [("DAYRANGE", 6.21,  0.01,  0.01,  10.0)]),
    (5, "QuantPass_PT_Exit",                               [("PT_Base",  0.671, 0.001, 0.001, 1.0)]),
    (6, "RescueTeamExit",                                  [("Length",   280.0, 20.0,  20.0,  600.0),
                                                            ("std",      3.5,   0.1,   3.0,   6.0)]),
]
ALL_MODULE_NAMES = [m[1] for m in MODULES]

PREFIX = "BTCH2OOS_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_hunter2_oos_validation_{int(time.time())}.log"
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


def _axis(name, val, step, lo, hi):
    start = max(lo, round(val - step, 8))
    stop = min(hi, round(val + step, 8))
    if stop <= start:
        stop = start + step
    return ParamAxis(name, start, stop, step)


def _module_cfg(idx: int, signal: str, axes) -> StrategyConfig:
    params = [_axis(nm, v, st, lo, hi) for (nm, v, st, lo, hi) in axes]
    return StrategyConfig(
        name=f"{PREFIX}M{idx}_{signal[:30]}",
        mc_signal_name=signal,
        timeframe="hourly",
        bar_period=60,
        params=params,
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def _baseline_cfg() -> StrategyConfig:
    return StrategyConfig(
        name=f"{PREFIX}A00_baseline",
        mc_signal_name=MAIN_SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("LEN_L",            224.0, 226.0, 1.0),
            ParamAxis("LEN_S",            824.0, 826.0, 1.0),
            ParamAxis("ATR_multiplier_L", 4.0,   4.5,   0.25),
            ParamAxis("ATR_multiplier_S", 1.5,   2.0,   0.25),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(cfg: StrategyConfig) -> Path:
    return OUTPUT_DIR / f"{cfg.name}_raw.csv"


def _apply_status(conn, active_module: Optional[str], manual: bool):
    status_map = {name: (name == active_module) for name in ALL_MODULE_NAMES}
    if manual:
        print("\n" + "=" * 60)
        print("MANUAL STATUS SETUP (Format Objects > Signals):")
        print(f"  {MAIN_SIGNAL}: ON (LEN_L=225 LEN_S=825 ATR_L=4.25 ATR_S=1.75)")
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


def _pick_exact(df, axes):
    """Pick the row matching the fixed module param values exactly."""
    m = df
    for (nm, v, st, lo, hi) in axes:
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    if m.empty:
        return None
    return m.iloc[0]


def _validate_main_fixed(df) -> bool:
    main_cols = [c for c in CHAMPION if c in df.columns]
    if not main_cols:
        log.warning("  Main-signal columns not in CSV — fixed-value check skipped")
        return True
    for c in main_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        tol = max(1e-6, abs(CHAMPION[c]) * 1e-4)
        if not ((col - CHAMPION[c]).abs() <= tol).all():
            log.error("  INVALID: main %s NOT fixed at %.6g (got %.6g..%.6g)",
                      c, CHAMPION[c], col.min(), col.max())
            return False
    log.info("  Main-signal fixed-value check passed (%s)", main_cols)
    return True


def save_json(payload):
    out = OUTPUT_DIR / "final_params_btc_hunter2_oos_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def run_search(conn, from_csv: bool, only_module: Optional[int], manual_status: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "strategy_main": MAIN_SIGNAL, "symbol": SYMBOL, "timeframe": "Hourly (60 min)",
        "champion_fixed": CHAMPION, "full_range": FULL_RANGE,
        "method": "Each module FIXED at IS-best param; full-period 2021/03-2026/06; "
                  "PASS = NP > baseline AND abs(MDD) < abs(baseline MDD).",
        "baseline_full": None, "modules": [],
        "timestamp": datetime.now().isoformat(),
    }
    out_json = OUTPUT_DIR / "final_params_btc_hunter2_oos_validation.json"
    if out_json.exists():
        try:
            old = json.load(open(out_json, encoding="utf-8"))
            payload["baseline_full"] = old.get("baseline_full")
            payload["modules"] = old.get("modules", [])
        except Exception:
            pass

    log.info("==============================================================")
    log.info("  BTC HUNTER2 exit-module FULL-PERIOD validation")
    log.info("  Main fixed LEN_L=225 LEN_S=825 ATR_L=4.25 ATR_S=1.75 (ON); range %s ~ %s",
             FULL_RANGE[0], FULL_RANGE[1])
    log.info("==============================================================")

    # ── Trim chart to FULL range (3-attempt retry: the right-click can transiently miss) ──
    if not from_csv and conn is not None:
        _trim_ok = False
        for _att in (1, 2, 3):
            try:
                log.info("Trimming chart to FULL range %s ~ %s (attempt %d) ...", *FULL_RANGE, _att)
                try:
                    mc._close_optimization_report()
                except Exception:
                    pass
                mc.set_instrument_data_range(conn, FULL_RANGE[0], FULL_RANGE[1])
                _trim_ok = True
                log.info("Chart trimmed (verify leftmost ~2021/03, rightmost ~2026/06).")
                break
            except Exception as e:
                log.warning("  set_instrument_data_range attempt %d FAILED: %s", _att, e)
                time.sleep(1.0)
        if not _trim_ok:
            log.error("  set_instrument_data_range FAILED after 3 tries — aborting")
            return 1

    base_np = base_mdd = None
    if payload.get("baseline_full"):
        base_np = payload["baseline_full"]["net_profit"]
        base_mdd = payload["baseline_full"]["max_drawdown"]

    # ── A00 full-period baseline (all modules OFF) ───────────────────────────
    if only_module is None or only_module == 0:
        cfg0 = _baseline_cfg()
        if not (from_csv or csv_for(cfg0).exists()):
            _apply_status(conn, None, manual_status)
        df0 = run_or_load(cfg0, conn, from_csv)
        # NOTE: do NOT call _validate_main_fixed here — the A00 baseline micro-grid
        # intentionally VARIES the main params (ATR 150-152, Mult 9.10-9.20); we pick
        # the exact champion (225/825/4.25/1.75) row below.  That check is for MODULE runs only.
        if df0 is not None and not df0.empty:
            m = df0
            for nm, v in CHAMPION.items():
                if nm in m.columns:
                    m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
            row = m.iloc[0] if not m.empty else df0.loc[
                pd.to_numeric(df0["NetProfit"], errors="coerce").idxmax()]
            base_np = float(row["NetProfit"]); base_mdd = float(row["MaxDrawdown"])
            payload["baseline_full"] = {
                "net_profit": base_np, "max_drawdown": base_mdd,
                "total_trades": int(row["TotalTrades"]),
                "timestamp": datetime.now().isoformat()}
            log.info("A00 FULL baseline: NP=%.2f MDD=%.2f tr=%d",
                     base_np, base_mdd, int(row["TotalTrades"]))
        else:
            log.warning("A00 full baseline failed")
        save_json(payload)

    # ── M1-M6: each fixed at IS-best, full period ────────────────────────────
    for idx, signal, axes in MODULES:
        if only_module is not None and only_module != idx:
            continue
        cfg = _module_cfg(idx, signal, axes)
        log.info("--- Module %d: %s FIXED %s (%d combos) ---", idx, signal,
                 {nm: v for nm, v, *_ in axes}, cfg.total_runs())
        if not (from_csv or csv_for(cfg).exists()):
            _apply_status(conn, signal, manual_status)
        df = run_or_load(cfg, conn, from_csv)

        entry = {"module": idx, "signal": signal,
                 "fixed_params": {nm: v for nm, v, *_ in axes},
                 "rows": len(df) if df is not None else 0,
                 "timestamp": datetime.now().isoformat()}
        row = _pick_exact(df, axes) if (df is not None and not df.empty
                                        and _validate_main_fixed(df)) else None
        if row is not None:
            np_ = float(row["NetProfit"]); mdd = float(row["MaxDrawdown"])
            improves_np = (base_np is not None and np_ > base_np)
            reduces_mdd = (base_mdd is not None and abs(mdd) < abs(base_mdd))
            entry.update({
                "net_profit": np_, "max_drawdown": mdd,
                "total_trades": int(row["TotalTrades"]),
                "delta_np": round(np_ - base_np, 2) if base_np is not None else None,
                "delta_np_pct": round((np_ - base_np) / base_np * 100, 2) if base_np else None,
                "delta_mdd": round(mdd - base_mdd, 2) if base_mdd is not None else None,
                "improves_np": improves_np, "reduces_mdd": reduces_mdd,
                "pass": bool(improves_np and reduces_mdd), "valid": True})
            log.info("  M%d NP=%.2f (%+.2f%%) MDD=%.2f tr=%d  NP%s MDD%s  %s",
                     idx, np_, entry["delta_np_pct"] or 0, mdd, entry["total_trades"],
                     "UP" if improves_np else "dn", "DOWN" if reduces_mdd else "up",
                     "*** PASS ***" if entry["pass"] else "fail")
        else:
            entry["valid"] = False
            log.warning("  Module %d: NO VALID DATA / exact row not found", idx)

        payload["modules"] = [m for m in payload["modules"] if m.get("module") != idx]
        payload["modules"].append(entry)
        payload["modules"].sort(key=lambda m: m["module"])
        save_json(payload)

    # ── Teardown ─────────────────────────────────────────────────────────────
    if not from_csv and only_module is None:
        try:
            _apply_status(conn, None, manual_status)
            log.info("Teardown: all modules OFF")
        except Exception as e:
            log.warning("Teardown failed: %s", e)

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("==============================================================")
    if base_np is not None:
        log.info("  FULL baseline NP=%.2f MDD=%.2f", base_np, base_mdd)
    passers = [m for m in payload["modules"] if m.get("pass")]
    for m in payload["modules"]:
        if m.get("valid"):
            log.info("  M%d %-48s NP=%.0f (%+.2f%%) MDD=%.0f  %s",
                     m["module"], m["signal"], m["net_profit"],
                     m.get("delta_np_pct") or 0, m["max_drawdown"],
                     "PASS" if m.get("pass") else "fail")
        else:
            log.info("  M%d %-48s FAILED", m["module"], m["signal"])
    log.info("  >>> PASS (NP up AND MDD down): %s",
             [m["module"] for m in passers] or "NONE")
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
    ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Administrator manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="BTC HUNTER2 exit-module full-period OOS validation")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--module", type=int, default=None, metavar="N")
    ap.add_argument("--manual-status", action="store_true")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0

    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    return run_search(conn, args.from_csv, args.module, args.manual_status)


if __name__ == "__main__":
    sys.exit(main())
