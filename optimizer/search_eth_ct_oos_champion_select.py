"""
search_eth_ct_oos_champion_select.py — Out-of-sample (OOS) selection among the 9 converged
ETH Hourly CT champions (rounds 1-5).

Workspace 20260101_SFJ_Bollinger_AI.wsp now has full data 2021/03/01 – 2026/06/10, which
extends past the IS cutoff 2026/01/01 used in all 5 prior rounds — so OOS is finally measurable.

TASK: re-test each previously-converged champion on this workspace and pick the parameter set
that earns the MOST out-of-sample profit (2026/01/01 -> 2026/06/10) WITHOUT its drawdown
breaking the established Max Drawdown.  Candidate selection / validation — NOT a new grid.

METHOD — TWO-PASS, TRIM-CHART (reliable MCReport route, NOT the live report reader):
  KEY: setting the signal's End date does NOT trim the optimization backtest — the
  optimization always runs over the CHART's currently-loaded data range.  So IS vs FULL
  is isolated by trimming the CHART data range between two separate invocations:
    PASS IS  : MC64 Format Instrument (F5) -> Settings -> data "To" = 2026/01/01,
               then  py ... --period is    -> NP_IS,   MDD_IS  (per candidate)
    PASS FULL: set data "To" = 2026/06/10, then  py ... --period full -> NP_FULL, MDD_FULL
    THEN     : py ... --from-csv   ->  OOS_NP = NP_FULL - NP_IS ;  PASS when abs(MDD_FULL)<=abs(MDD_IS)
  Each candidate is evaluated by an 81-combo micro-grid centred on its exact params (all 4
  axes vary +/-1 step per Critical Rule 1 so the CSV is clean); the EXACT candidate row is
  picked.  Metric = NetProfit + Max INTRADAY Drawdown — the SAME metric the prior rounds
  recorded, so "did OOS break the established Max Drawdown" is apples-to-apples.
  Winner = max OOS_NP among PASS candidates; also report overall max-OOS_NP and lowest MDD_FULL.
  Per-candidate cleanup+retry makes all 9 complete reliably.

PREREQUISITES (MC64, elevated):
  - 20260101_SFJ_Bollinger_AI.wsp open, ETHUSDT HOT Hourly chart, the
    SFJ_15Dworkshop_lesson5_countertrend_LS_crypto signal applied (Status ON)
  - The CHART data "To" date MUST be set to match the pass (2026/01/01 for is, 2026/06/10 for full)

CLI (run the two passes with the chart trimmed accordingly, then from-csv):
  py search_eth_ct_oos_champion_select.py --period is     # chart To=2026/01/01 first
  py search_eth_ct_oos_champion_select.py --period full   # chart To=2026/06/10 first
  py search_eth_ct_oos_champion_select.py --from-csv      # compute OOS = full - is
  py search_eth_ct_oos_champion_select.py --period is --candidate C9   # smoke one candidate
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
from typing import Dict, List, Optional

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mc_automation as mc
from config import DateRange, ParamAxis, StrategyConfig


WORKSPACE   = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_Bollinger_AI.wsp"
SYMBOL      = "ETHUSDT HOT"
SIGNAL      = "SFJ_15Dworkshop_lesson5_countertrend_LS_crypto"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\eth_ct_oos_champion_select_search")

# IMPORTANT — TWO-PASS / TRIM-CHART method.
# Finding: setting the signal's End date does NOT trim the optimization backtest;
# the optimization always runs over the CHART's currently-loaded data range.  So we
# isolate IS vs FULL by trimming the CHART data range (Format Instrument F5 -> Settings
# -> "To" date) between two separate invocations, NOT via the signal date:
#   Pass IS  : set chart data "To" = 2026/01/01, then run with --period is
#   Pass FULL: set chart data "To" = 2026/06/10, then run with --period full
#   Then run --from-csv :  OOS_NP(per candidate) = NP_full - NP_is
# The cfg.insample below is a WIDE range on purpose so the (ineffective, slow) signal
# date-set is a no-op and never misleads — the chart range is the real control.
WIDE = DateRange("2019/01/01", "2027/01/01")
# Instrument data ranges set AUTOMATICALLY via Format Instruments -> Settings (set_instrument_data_range).
# IS = 2022/01/01-2026/01/01 ; FULL = 2021/03/01-2026/06/10  (user-specified).
RANGES = {"is": ("2022/01/01", "2026/01/01"), "full": ("2021/03/01", "2026/06/10")}
PERIOD_ORDER = ["is", "full"]

# step per axis for the micro-grid (centre = exact candidate value -> exact row exists)
STEP = {"LENGTH_LONG": 1.0, "STDDEV_LONG": 0.025, "LENGTH_SHORT": 1.0, "STDDEV_SHORT": 0.025}
LO   = {"LENGTH_LONG": 2.0, "STDDEV_LONG": 0.1, "LENGTH_SHORT": 2.0, "STDDEV_SHORT": 0.1}
HI   = {"LENGTH_LONG": 500.0, "STDDEV_LONG": 20.0, "LENGTH_SHORT": 500.0, "STDDEV_SHORT": 20.0}

# 4 converged champions (ETH R1-R3 NP-max + Obj-max plateau; verified == eth_ct_hourly{,2,3} JSONs).
# id, regime, LL, SL, LS, SS, prior IS NP, prior IS MDD (= MCReport Max Intraday DD)
CANDIDATES = [
    ("E1", "NP-max R1",        112.0, 4.05,  99.0,  4.6,   4868.0, -748.0),
    ("E2", "NP-max R2/R3 ***", 111.0, 4.025, 115.0, 4.725, 5005.0, -748.0),
    ("E3", "Obj-max R2",       110.0, 4.30,  110.0, 4.7,   4848.0, -580.0),
    ("E4", "Obj-max R3 twin",  109.0, 4.40,  109.0, 4.75,  4848.0, -580.0),
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_eth_ct_oos_champion_select_{int(time.time())}.log"
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


def _axis(name, val):
    st = STEP[name]
    start = max(LO[name], round(val - st, 8))
    stop = min(HI[name], round(val + st, 8))
    if stop <= start:
        stop = start + st
    return ParamAxis(name, start, stop, st)


def _cfg(cid, period, ll, sl, ls, ss) -> StrategyConfig:
    return StrategyConfig(
        name=f"ETHCTSEL_{period}_{cid}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[_axis("LENGTH_LONG", ll), _axis("STDDEV_LONG", sl),
                _axis("LENGTH_SHORT", ls), _axis("STDDEV_SHORT", ss)],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=WIDE,   # signal date is a no-op; CHART trim controls the period
    )


def csv_for(cfg: StrategyConfig) -> Path:
    return OUTPUT_DIR / f"{cfg.name}_raw.csv"


_user32 = ctypes.windll.user32
# Titles of STRAY modal windows that block the next right-click -> Optimize launch.
# (Do NOT match the MC main window, whose title contains "MultiCharts".)
_STRAY_KW = ["Optimization", "最佳化", "優化", "Optimis",
             "Format Objects", "Format Signals", "Format Strategies", "Format Signal",
             "格式物件", "格式訊號", "格式策略"]


def _cleanup_stray_windows():
    """Close any leftover optimization wizard / report / Format dialog that would
    block the next candidate's right-click -> Optimize.  Root cause of the
    intermittent 'wizard not found after 30s' (every-other-candidate) failures:
    the previous run leaves a modal open, the next launch can't reach the menu."""
    try:
        mc._close_optimization_report()
    except Exception as e:
        log.debug("close_optimization_report: %s", e)

    victims = []

    def _cb(hwnd, _):
        try:
            n = _user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            _user32.GetWindowTextW(hwnd, buf, n + 1)
            t = buf.value
            if "MultiCharts" in t:           # never touch the main app window
                return True
            if _user32.IsWindowVisible(hwnd) and any(k.lower() in t.lower() for k in _STRAY_KW):
                victims.append((hwnd, t))
        except Exception:
            pass
        return True

    proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    try:
        _user32.EnumWindows(proc(_cb), 0)
    except Exception as e:
        log.debug("EnumWindows: %s", e)

    for hwnd, t in victims:
        try:
            _user32.PostMessageW(hwnd, 0x0010, 0, 0)   # WM_CLOSE
            time.sleep(0.4)
            log.info("  cleanup: closed stray window '%s'", t[:40])
        except Exception:
            pass
    if victims:
        time.sleep(0.6)


def run_or_load(cfg, conn, from_csv):
    p = csv_for(cfg)
    if from_csv or p.exists():
        if p.exists():
            try:
                df = mc.load_results_csv(str(p), cfg)
                log.info("Loaded %s: %d rows", cfg.name, len(df))
                return df
            except Exception as e:
                log.warning("Could not load %s: %s", p, e)
        else:
            log.warning("No CSV for %s", cfg.name)
        return None
    log.info("=== Starting %s (%d combos) ===", cfg.name, cfg.total_runs())
    # Up to 2 attempts: proactively clean stray modals before each, so a leftover
    # wizard/dialog from the previous candidate can't block this launch.
    for attempt in (1, 2):
        _cleanup_stray_windows()
        t0 = time.time()
        try:
            raw = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
            log.info("  Done %.1f min (attempt %d)", (time.time() - t0) / 60, attempt)
            return mc.load_results_csv(raw, cfg)
        except Exception as e:
            log.warning("  attempt %d FAILED: %s", attempt, e)
            if attempt == 2:
                log.error("  %s: giving up after 2 attempts", cfg.name, exc_info=True)
    return None


def _pick(df, ll, sl, ls, ss):
    """Exact candidate row from the micro-grid."""
    m = df
    for nm, v in (("LENGTH_LONG", ll), ("STDDEV_LONG", sl),
                  ("LENGTH_SHORT", ls), ("STDDEV_SHORT", ss)):
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    if m.empty:
        return None
    return m.iloc[0]


def save_json(payload):
    out = OUTPUT_DIR / "final_params_eth_ct_oos_champion_select.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def run(conn, from_csv, only_candidate, only_period):
    run_t0 = time.time()
    out_path = OUTPUT_DIR / "final_params_eth_ct_oos_champion_select.json"

    # MERGED single-run, AUTO data-range: default does BOTH passes; the script itself
    # sets the chart instrument Data Range (Format Instruments -> Settings) per pass.
    # --period restricts to one pass; --from-csv just re-computes from existing CSVs.
    if from_csv:
        periods = []
    elif only_period:
        periods = [only_period]
    else:
        periods = PERIOD_ORDER   # full merged: IS then FULL in one invocation

    payload = {
        "symbol": SYMBOL, "signal": SIGNAL, "timeframe": "Hourly (60 min)",
        "method": "MERGED auto-range; set_instrument_data_range per pass; micro-grid MCReport CSV; "
                  "metric=Max Intraday Drawdown; OOS_NP=NP_full-NP_is; PASS=abs(MDD_full)<=abs(MDD_is).",
        "ranges": RANGES,
        "results": {},     # period -> {cid: {...}}
        "candidates": {},  # cid -> merged is/full + verdict
        "winner": None,
        "run_started_at": datetime.now().isoformat(),
        "run_finished_at": None, "run_total_sec": None,
    }
    if out_path.exists():
        try:
            payload["results"] = json.load(open(out_path, encoding="utf-8")).get("results", {})
        except Exception:
            pass

    if not from_csv:
        mc.ensure_chart_ready(conn, _cfg("C1", "is", *CANDIDATES[0][2:6]))
        # Fresh start for a full merged run: clear stale CSVs so both passes re-measure
        # on the auto-set ranges (otherwise run_or_load would load old cached CSVs).
        if not only_period:
            for _f in OUTPUT_DIR.glob("ETHCTSEL_*_raw.csv"):
                try:
                    _f.unlink()
                except Exception:
                    pass
            payload["results"] = {}
            log.info("Cleared stale ETHCTSEL CSVs for a fresh merged run.")

    for period in periods:
        # --- AUTO-set the chart instrument data range for this pass ---
        rng = RANGES[period]
        log.info("==============================================================")
        log.info("  ETH Hourly CT — OOS CHAMPION SELECTION  [PASS: %s]  range %s ~ %s",
                 period, rng[0], rng[1])
        log.info("==============================================================")
        try:
            mc.set_instrument_data_range(conn, rng[0], rng[1])
        except Exception as e:
            log.error("  set_instrument_data_range FAILED for %s: %s — aborting pass", period, e)
            continue
        pres = payload["results"].get(period, {})
        for (cid, regime, ll, sl, ls, ss, pnp, pmdd) in CANDIDATES:
            if only_candidate and cid != only_candidate:
                continue
            t0 = time.time()
            cfg = _cfg(cid, period, ll, sl, ls, ss)
            log.info("--- [%s] %s %s LL=%g SL=%g LS=%g SS=%g [%s] ---",
                     period, cid, regime, ll, sl, ls, ss, datetime.now().strftime("%H:%M:%S"))
            df = run_or_load(cfg, conn, from_csv)
            ent = {"id": cid, "regime": regime,
                   "params": {"LENGTH_LONG": ll, "STDDEV_LONG": sl,
                              "LENGTH_SHORT": ls, "STDDEV_SHORT": ss},
                   "timestamp": datetime.now().isoformat(),
                   "elapsed_sec": round(time.time() - t0, 1),
                   "rows": len(df) if df is not None else 0}
            row = _pick(df, ll, sl, ls, ss) if (df is not None and not df.empty) else None
            if row is not None:
                ent.update({"net_profit": float(row["NetProfit"]),
                            "max_intraday_drawdown": float(row["MaxDrawdown"]),
                            "total_trades": int(row["TotalTrades"]), "valid": True})
                log.info("  [%s] %s NP=%.2f MDD=%.2f tr=%d [%.0fs]", period, cid,
                         ent["net_profit"], ent["max_intraday_drawdown"],
                         ent["total_trades"], ent["elapsed_sec"])
            else:
                ent["valid"] = False
                log.warning("  [%s] %s: candidate row not found", period, cid)
            pres[cid] = ent
            payload["results"][period] = pres
            save_json(payload)

    # ---- merge is/full + verdicts ----
    is_r, full_r = payload["results"].get("is", {}), payload["results"].get("full", {})
    for (cid, regime, ll, sl, ls, ss, pnp, pmdd) in CANDIDATES:
        ci, cf = is_r.get(cid), full_r.get(cid)
        c = {"id": cid, "regime": regime,
             "params": {"LENGTH_LONG": ll, "STDDEV_LONG": sl, "LENGTH_SHORT": ls, "STDDEV_SHORT": ss},
             "prior_is_np": pnp, "prior_is_mdd": pmdd}
        if ci and ci.get("valid"):
            c.update({"np_is": ci["net_profit"], "mdd_is": ci["max_intraday_drawdown"],
                      "tr_is": ci["total_trades"],
                      "is_drift_vs_prior_pct": round((ci["net_profit"] - pnp) / pnp * 100, 1) if pnp else None})
        if cf and cf.get("valid"):
            c.update({"np_full": cf["net_profit"], "mdd_full": cf["max_intraday_drawdown"],
                      "tr_full": cf["total_trades"]})
        if ci and ci.get("valid") and cf and cf.get("valid"):
            c["oos_np"] = round(c["np_full"] - c["np_is"], 2)
            c["mdd_break"] = abs(c["mdd_full"]) > abs(c["mdd_is"])
            c["pass"] = not c["mdd_break"]
            c["valid"] = True
        else:
            c["valid"] = False
        payload["candidates"][cid] = c

    valid = [c for c in payload["candidates"].values() if c.get("valid")]
    passing = [c for c in valid if c.get("pass")]
    winner = max(passing, key=lambda c: c["oos_np"]) if passing else None
    max_any = max(valid, key=lambda c: c["oos_np"]) if valid else None
    low_mdd = min(valid, key=lambda c: abs(c["mdd_full"])) if valid else None
    payload["winner"] = {
        "best_oos_np_passing": winner["id"] if winner else None,
        "best_oos_np_passing_params": winner["params"] if winner else None,
        "best_oos_np_passing_value": winner["oos_np"] if winner else None,
        "max_oos_np_any": max_any["id"] if max_any else None,
        "lowest_full_mdd": low_mdd["id"] if low_mdd else None,
    }
    payload["run_finished_at"] = datetime.now().isoformat()
    payload["run_total_sec"] = round(time.time() - run_t0, 1)
    save_json(payload)

    # ---- console summary ----
    log.info("==============================================================")
    log.info("  SUMMARY (sorted by OOS_NP)   total %.1f min", (time.time() - run_t0) / 60)
    log.info("  %-3s %-13s %9s %9s %9s %9s %4s", "id", "regime", "NP_is", "NP_full", "OOS_NP", "MDD_full", "pass")
    for c in sorted(valid, key=lambda c: c["oos_np"], reverse=True):
        log.info("  %-3s %-13s %9.0f %9.0f %9.0f %9.0f %4s", c["id"], c["regime"],
                 c["np_is"], c["np_full"], c["oos_np"], c["mdd_full"], "Y" if c["pass"] else "n")
    # sanity: IS==FULL for all -> set_instrument_data_range did NOT take effect
    # (the Format Instruments Data Range wasn't actually applied this pass).
    if valid and all(abs(c["np_full"] - c["np_is"]) < 1e-6 for c in valid):
        log.warning("  !! IS==FULL for ALL candidates -> set_instrument_data_range did NOT apply. "
                    "Check the Format Instrument UIA dump in the log; verify via screenshot.")
    if not valid:
        log.info("  (Only one pass present so far — IS or FULL missing.)")
    if winner:
        log.info("  >>> WINNER: %s %s  OOS_NP=%.2f  MDD_full=%.0f (<= MDD_is=%.0f)",
                 winner["id"], winner["params"], winner["oos_np"], winner["mdd_full"], winner["mdd_is"])
    else:
        log.info("  >>> No PASS candidate. Max-OOS overall: %s", max_any["id"] if max_any else None)
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
    ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Administrator manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="ETH Hourly CT OOS champion selection (auto-range merged)")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--candidate", metavar="Cn", default=None)
    ap.add_argument("--period", choices=PERIOD_ORDER, default=None,
                    help="restrict to one pass (is/full); default runs BOTH (merged)")
    ap.add_argument("--probe-instrument", action="store_true",
                    help="just open Format Instruments, set the IS range, and dump the dialog "
                         "(for verifying/debugging set_instrument_data_range)")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0

    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()

    if args.probe_instrument:
        mc.ensure_chart_ready(conn, _cfg("C1", "is", *CANDIDATES[0][2:6]))
        # Verify the FIX on the hard case: expand to the FULL range (this is the
        # expansion to 2026/06 that failed 3x).  Then REOPEN the dialog and read the
        # pickers back to confirm the change persisted (not just the in-call readback).
        tgt_from, tgt_to = RANGES["full"]     # 2021/03/01 ~ 2026/06/10
        print(f"\nProbe: setting Data Range -> {tgt_from} ~ {tgt_to} (the failing expansion)")
        mc.set_instrument_data_range(conn, tgt_from, tgt_to)
        time.sleep(1.0)
        rb_from, rb_to = mc.read_instrument_data_range(conn)
        def _fmt(t):
            return f"{t[0]:04d}/{t[1]:02d}/{t[2]:02d}" if t else "None"
        exp_to = tuple(int(x) for x in tgt_to.split("/"))
        ok_to = (rb_to == exp_to)
        print("\n================ PROBE RESULT ================")
        print(f"  target  From={tgt_from}  To={tgt_to}")
        print(f"  readback From={_fmt(rb_from)}  To={_fmt(rb_to)}")
        print(f"  To-date persisted after reopen: {'YES ✓' if ok_to else 'NO ✗'}")
        print("  --> Now SCREENSHOT the chart: rightmost bar should be ~2026/06/10.")
        print("      If To persisted AND chart extends to Jun-2026, the fix works —")
        print("      run the full merged BAT next.  If not, we switch to manual-pause BAT.")
        print("=============================================")
        return 0

    return run(conn, args.from_csv, args.candidate, args.period)


if __name__ == "__main__":
    sys.exit(main())
