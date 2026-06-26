"""
search_eth_closechannelbreakout_cumulative_oos.py -- CUMULATIVE greedy exit-module FULL-PERIOD
(incl. OOS) validation for ETH SFJ_CloseChannelBreakout_crypto.

Main signal FIXED at CC4 (Length=6 ATRMult=7.5 ReentryBars=12; the OOS de-facto champion), Status ON.
Each of the 6 exit modules is FIXED at its IS-best param and considered ONE AT A TIME in DESCENDING
IS ΔNP% order. Starting from main-only, we GREEDILY STACK: enable [main + kept-set + candidate] on the
FULL range 2021/03/01-2026/06/10 and KEEP the candidate only if it raises
    RoMaD = NetProfit / |Max Intraday Drawdown|
vs the current kept-set; else discard. Then move to the next candidate. (Differs from the per-module
*_oos_validation scripts which test each module independently vs the main-only baseline.)

RoMaD denominator = Max Intraday Drawdown = the pipeline's MaxDrawdown column (config.MC_COLUMN_MAP).
Keep-rule = RoMaD strictly increases (single criterion; MDD may tick up if NP rises enough).

Order (descending IS ΔNP% from results/eth_closechannelbreakout_exit_modules_search):
  M6 RescueTeamExit     Length=260 std=4.5   (+22.18%)
  M3 EntryBarsAfterExit EXITBAR=130          (+15.75%)
  M5 QuantPass_PT_Exit  PT_Base=0.479        (+13.34%)
  M2 TrailingStop       ATRSTP=14.7          (+12.00%)
  M4 high_volatility    DAYRANGE=4.98        (+9.86%)
  M1 ATRstop            STP=9.6              (+2.30%)

Mechanism: per step, Status = main + kept + candidate; run a micro-grid that VARIES ONLY the
candidate's representative param +/-1 step (Critical Rule 1) while every other enabled signal's params
stay UNCHECKED -> held at the user-fixed Format-Objects values; pick the exact candidate-param row ->
NP + Max Intraday Drawdown from the CSV. M6's `std` is the varied axis (its `Length` collides with the
main `Length` -> never varied; excluded from the main-fixed check whenever M6 is enabled).

PREREQUISITES (MC64, run AS ADMINISTRATOR):
  - Workspace 20260101_SFJ_CloseChannelBreakout_crypto_AI.wsp open, ETHUSDT HOT 60-min tab ACTIVE,
    data through 2026/06/10, Binance connected, Study Editor closed.
  - Main inputs = 6 / 7.5 / 12, Status ON; all 6 modules inserted with inputs at IS-best
    (Length=260 std=4.5; EXITBAR=130; PT_Base=0.479; ATRSTP=14.7; DAYRANGE=4.98; STP=9.6), Status OFF;
    workspace saved.

CLI:
  py search_eth_closechannelbreakout_cumulative_oos.py            # A00 + greedy M6..M1 + teardown
  py search_eth_closechannelbreakout_cumulative_oos.py --only 1   # A00 baseline + first candidate only (smoke)
  py search_eth_closechannelbreakout_cumulative_oos.py --manual-status
  py search_eth_closechannelbreakout_cumulative_oos.py --from-csv # recompute decisions from existing CSVs
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
from config import DateRange, ParamAxis, StrategyConfig


WORKSPACE   = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260622_SFJ_CloseChannelBreakout_crypto_AI.wsp"
SYMBOL      = "ETHUSDT HOT"
MAIN_SIGNAL = "SFJ_CloseChannelBreakout_crypto"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\eth_closechannelbreakout_cumulative_oos_search")
INSAMPLE    = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
FULL_RANGE  = ("2021/03/01", "2026/06/10")

CHAMPION = {"Length": 6.0, "ATRMult": 7.5, "ReentryBars": 12.0}

# (label, signal, fixed_params, vary_param, step, lo, hi)  -- descending IS ΔNP% order
ORDER: List[Tuple[str, str, Dict[str, float], str, float, float, float]] = [
    ("M6", "RescueTeamExit",                               {"Length": 260.0, "std": 4.5}, "std",     0.1,   3.0,   6.0),
    ("M3", "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit", {"EXITBAR": 130.0},            "EXITBAR", 1.0,   1.0,   1000.0),
    ("M5", "QuantPass_PT_Exit",                            {"PT_Base": 0.479},            "PT_Base", 0.001, 0.001, 1.0),
    ("M2", "SFJ_15Dworkshop_lesson9_1_TrailingStop",       {"ATRSTP": 14.7},              "ATRSTP",  0.1,   0.1,   100.0),
    ("M4", "SFJ_15Dworkshop_lesson11_3_high_volatility_exit", {"DAYRANGE": 4.98},         "DAYRANGE",0.01,  0.01,  10.0),
    ("M1", "SFJ_15Dworkshop_lesson4_ATRstop",              {"STP": 9.6},                  "STP",     0.1,   0.1,   100.0),
]
ALL_MODULE_NAMES = [m[1] for m in ORDER]
# module-param names per signal (for main-fixed-check collision exclusion)
MODULE_PARAM_NAMES = {sig: set(fp.keys()) for (_, sig, fp, *_ ) in ORDER}

EPS = 1e-9   # RoMaD strict-increase tolerance

PREFIX = "ETHCCBCUM_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_eth_closechannelbreakout_cumulative_oos_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s -- %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(str(_LOG_FILE), encoding="utf-8")],
)
log = logging.getLogger(__name__)


def _axis(name, val, step, lo, hi) -> ParamAxis:
    start = max(lo, round(val - step, 8))
    stop = min(hi, round(val + step, 8))
    if stop <= start:
        stop = start + step
    return ParamAxis(name, start, stop, step)


def _baseline_cfg() -> StrategyConfig:
    return StrategyConfig(
        name=f"{PREFIX}S0_baseline",
        mc_signal_name=MAIN_SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[ParamAxis("Length", 5.0, 7.0, 1.0),
                ParamAxis("ATRMult", 7.0, 8.0, 0.5),
                ParamAxis("ReentryBars", 11.0, 13.0, 1.0)],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def _candidate_cfg(step: int, label: str, signal: str, vary_param, val, step_v, lo, hi) -> StrategyConfig:
    return StrategyConfig(
        name=f"{PREFIX}S{step}_{label}",
        mc_signal_name=signal,
        timeframe="hourly",
        bar_period=60,
        params=[_axis(vary_param, val, step_v, lo, hi)],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(cfg: StrategyConfig) -> Path:
    return OUTPUT_DIR / f"{cfg.name}_raw.csv"


def _apply_status(conn, enabled_modules: List[str], manual: bool):
    """Status ON for main (always) + every module in enabled_modules; OFF for the rest."""
    status_map = {name: (name in enabled_modules) for name in ALL_MODULE_NAMES}
    if manual:
        print("\n" + "=" * 60)
        print("MANUAL STATUS SETUP (Format Objects > Signals):")
        print(f"  {MAIN_SIGNAL}: ON (Length=6 ATRMult=7.5 ReentryBars=12)")
        for name in ALL_MODULE_NAMES:
            print(f"  {name}: {'ON' if status_map[name] else 'OFF'}")
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
        mc.ensure_chart_ready(conn, cfg)
    except Exception as e:
        log.warning("ensure_chart_ready: %s", e)
    try:
        raw = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done %.1f min -- %s", (time.time() - t0) / 60, Path(raw).name)
        return mc.load_results_csv(raw, cfg)
    except Exception as e:
        log.error("  FAILED: %s", e, exc_info=True)
        return None


def _validate_main_fixed(df, enabled_modules: List[str]) -> bool:
    """Main params must be held at CC4, EXCLUDING any CHAMPION name that collides with an enabled
    module param name (e.g. main 'Length' vs RescueTeamExit 'Length')."""
    collide = set()
    for sig in enabled_modules:
        collide |= MODULE_PARAM_NAMES.get(sig, set())
    main_cols = [c for c in CHAMPION if c in df.columns and c not in collide]
    if not main_cols:
        log.warning("  main-fixed check skipped (all main cols collide/absent)")
        return True
    for c in main_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        tol = max(1e-6, abs(CHAMPION[c]) * 1e-4)
        if not ((col - CHAMPION[c]).abs() <= tol).all():
            log.error("  INVALID: main %s NOT fixed at %.6g (got %.6g..%.6g) -- setup wrong!",
                      c, CHAMPION[c], col.min(), col.max())
            return False
    log.info("  main-fixed check passed (%s)", main_cols)
    return True


def _pick(df, vary_param, val):
    if vary_param in df.columns:
        m = df[(pd.to_numeric(df[vary_param], errors="coerce") - val).abs() < 1e-6]
        if not m.empty:
            return m.iloc[0]
    return None


def save_json(payload):
    out = OUTPUT_DIR / "final_params_eth_closechannelbreakout_cumulative_oos.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


def run_search(conn, from_csv: bool, only_k: Optional[int], manual_status: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_main": MAIN_SIGNAL, "symbol": SYMBOL, "timeframe": "Hourly (60 min)",
        "champion_fixed": CHAMPION, "full_range": FULL_RANGE,
        "method": "Cumulative greedy stacking in descending IS dNP% order; keep iff RoMaD "
                  "(=NetProfit/|Max Intraday Drawdown|) strictly increases vs current kept-set; "
                  "full period (incl OOS).",
        "baseline_full": None, "steps": [], "final_kept_set": None,
        "timestamp": datetime.now().isoformat(),
    }

    log.info("==============================================================")
    log.info("  ETH CloseChannelBreakout CUMULATIVE exit-module OOS validation")
    log.info("  Main fixed Length=6 ATRMult=7.5 ReentryBars=12 (ON); range %s ~ %s",
             FULL_RANGE[0], FULL_RANGE[1])
    log.info("==============================================================")

    if not from_csv and conn is not None:
        try:
            mc.ensure_chart_ready(conn, _baseline_cfg())   # bring the chart foreground BEFORE the trim
            log.info("Trimming chart to FULL range %s ~ %s ...", *FULL_RANGE)
            mc.set_instrument_data_range(conn, FULL_RANGE[0], FULL_RANGE[1])
            log.info("Chart trimmed (verify leftmost ~2021/03, rightmost ~2026/06).")
        except Exception as e:
            log.error("  set_instrument_data_range FAILED: %s -- aborting", e)
            return 1

    # ── A00 baseline: main only ──────────────────────────────────────────────
    cfg0 = _baseline_cfg()
    if not (from_csv or csv_for(cfg0).exists()):
        _apply_status(conn, [], manual_status)
    df0 = run_or_load(cfg0, conn, from_csv)
    if df0 is None or df0.empty:
        log.error("A00 baseline failed -- aborting"); save_json(payload); return 2
    m = df0
    for nm, v in CHAMPION.items():
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    row0 = m.iloc[0] if not m.empty else df0.loc[pd.to_numeric(df0["NetProfit"], errors="coerce").idxmax()]
    cur_np = float(row0["NetProfit"]); cur_midd = float(row0["MaxDrawdown"])
    cur_romad = cur_np / abs(cur_midd) if cur_midd else 0.0
    payload["baseline_full"] = {"net_profit": cur_np, "max_intraday_drawdown": cur_midd,
                                "romad": round(cur_romad, 4), "total_trades": int(row0["TotalTrades"]),
                                "exact_champion_row": not m.empty}
    log.info("A00 FULL baseline: NP=%.2f MIDD=%.2f RoMaD=%.4f tr=%d",
             cur_np, cur_midd, cur_romad, int(row0["TotalTrades"]))
    save_json(payload)

    # ── Greedy cumulative stacking ───────────────────────────────────────────
    kept: List[str] = []
    n_candidates = len(ORDER) if only_k is None else min(only_k, len(ORDER))
    for step, (label, signal, fixed, vary_param, step_v, lo, hi) in enumerate(ORDER, start=1):
        if step > n_candidates:
            break
        enabled = kept + [signal]
        cfg = _candidate_cfg(step, label, signal, vary_param, fixed[vary_param], step_v, lo, hi)
        log.info("--- Step %d  candidate %s %s (vary %s; kept=%s) ---",
                 step, label, fixed, vary_param, [l for l, s, *_ in ORDER if s in kept])
        if not (from_csv or csv_for(cfg).exists()):
            _apply_status(conn, enabled, manual_status)
        df = run_or_load(cfg, conn, from_csv)

        ent = {"step": step, "candidate": label, "signal": signal, "fixed_params": fixed,
               "vary_param": vary_param, "enabled": [signal] + kept,
               "rows": len(df) if df is not None else 0, "timestamp": datetime.now().isoformat()}
        row = None
        if df is not None and not df.empty and _validate_main_fixed(df, enabled):
            row = _pick(df, vary_param, fixed[vary_param])
        if row is not None:
            np_ = float(row["NetProfit"]); midd = float(row["MaxDrawdown"])
            romad = np_ / abs(midd) if midd else 0.0
            keep = romad > cur_romad + EPS
            ent.update({
                "net_profit": np_, "max_intraday_drawdown": midd, "romad": round(romad, 4),
                "total_trades": int(row["TotalTrades"]),
                "delta_np_pct": round((np_ - cur_np) / cur_np * 100, 2) if cur_np else None,
                "delta_mdd_pct": round((abs(midd) - abs(cur_midd)) / abs(cur_midd) * 100, 2) if cur_midd else None,
                "delta_romad_pct": round((romad - cur_romad) / cur_romad * 100, 2) if cur_romad else None,
                "prev_romad": round(cur_romad, 4), "decision": "KEEP" if keep else "discard",
                "valid": True})
            log.info("  %s: NP=%.2f MIDD=%.2f RoMaD=%.4f (prev %.4f, %+.2f%%) tr=%d  -> %s",
                     label, np_, midd, romad, cur_romad, ent["delta_romad_pct"] or 0,
                     ent["total_trades"], ent["decision"])
            if keep:
                kept.append(signal)
                cur_np, cur_midd, cur_romad = np_, midd, romad
        else:
            ent["valid"] = False
            log.warning("  Step %d %s: NO VALID DATA / exact row not found", step, label)

        payload["steps"] = [s for s in payload["steps"] if s.get("step") != step]
        payload["steps"].append(ent)
        payload["steps"].sort(key=lambda s: s["step"])
        payload["final_kept_set"] = [l for l, s, *_ in ORDER if s in kept]
        save_json(payload)

    # ── Teardown ─────────────────────────────────────────────────────────────
    if not from_csv and only_k is None:
        try:
            _apply_status(conn, kept, manual_status)  # leave the FINAL kept-set enabled for live use
            log.info("Teardown: left final kept-set enabled: %s", payload["final_kept_set"])
        except Exception as e:
            log.warning("Teardown failed: %s", e)

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("==============================================================")
    log.info("  A00 baseline: NP=%.2f MIDD=%.2f RoMaD=%.4f",
             payload["baseline_full"]["net_profit"], payload["baseline_full"]["max_intraday_drawdown"],
             payload["baseline_full"]["romad"])
    for s in payload["steps"]:
        if s.get("valid"):
            log.info("  S%d %-4s NP=%.0f MIDD=%.0f RoMaD=%.3f (%+.2f%%)  %s",
                     s["step"], s["candidate"], s["net_profit"], s["max_intraday_drawdown"],
                     s["romad"], s.get("delta_romad_pct") or 0, s["decision"])
        else:
            log.info("  S%d %-4s FAILED", s["step"], s["candidate"])
    log.info("  >>> FINAL KEPT SET: %s  (RoMaD %.4f -> %.4f)",
             payload["final_kept_set"] or "NONE (main only)",
             payload["baseline_full"]["romad"], round(cur_romad, 4))
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
    ap = argparse.ArgumentParser(description="ETH CloseChannelBreakout cumulative exit-module OOS validation")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--only", type=int, default=None, metavar="K",
                    help="evaluate only the first K candidates (A00 baseline always runs)")
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

    return run_search(conn, args.from_csv, args.only, args.manual_status)


if __name__ == "__main__":
    sys.exit(main())
