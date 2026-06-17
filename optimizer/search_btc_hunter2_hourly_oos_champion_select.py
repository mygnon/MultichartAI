"""
search_btc_hunter2_hourly_oos_champion_select.py — Out-of-sample (OOS) selection among the converged
BTC Hourly SFJ_HUNTER2_crypto champions (rounds 1-3, results/btc_hunter2_hourly{,2,3}_search/).

Workspace 20260101_SFJ_HUNTER_AI.wsp now has full data 2021/03/01 – 2026/06/10, which
extends past the IS cutoff 2026/01/01 used in all prior daily rounds — so OOS is measurable.

TASK: re-test each previously-converged BTC Hourly HUNTER2 champion on this workspace and pick the
parameter set that earns the MOST out-of-sample profit WITHOUT its drawdown breaking the
established Max Drawdown.  Candidate selection / validation — NOT a new grid.

METHOD — MERGED single-run, AUTO data-range (the proven ETH-Hourly route):
  MC64 ignores the signal Begin-date; only the CHART's loaded data range restricts the
  optimization backtest.  This script sets Format Instruments -> Settings -> Data Range
  itself, per pass (mc.set_instrument_data_range; the DTN_DATETIMECHANGE nudge fix makes
  the OK actually apply the From-To range):
    PASS is  : Data Range 2022/01/01 - 2026/01/01  -> NP_is,   MDD_is   (per candidate)
    PASS full: Data Range 2021/03/01 - 2026/06/10  -> NP_full, MDD_full
    THEN     : OOS_NP = NP_full - NP_is ;  PASS when abs(MDD_full) <= abs(MDD_is)
  Each candidate = an 81-combo micro-grid centred on its exact params (all 4 axes vary
  +/-1 step per Critical Rule 1 so the CSV is clean); the EXACT candidate row is picked.
  Metric = NetProfit + Max INTRADAY Drawdown (same metric the prior rounds recorded, so
  "did OOS break the established Max Drawdown" is apples-to-apples).
  Winner = max OOS_NP among PASS candidates; also report overall max-OOS_NP and lowest MDD_full.

NOTE on ranges: IS starts 2022/01, FULL starts 2021/03 (user-specified, same as the ETH run).
  So OOS_NP = full - is includes the extra early 2021/03-2022/01 segment as well as the late
  2026/01-2026/06 OOS segment — Pass2 is "full period incl. OOS" by the user's definition.

PREREQUISITES (MC64, elevated):
  - 20260101_SFJ_HUNTER_AI.wsp open, BTCUSDT HOT **60-Minute** chart tab ACTIVE, the
    SFJ_HUNTER2_crypto signal applied (Status ON)
  - Full data available to 2026/06/10 (the script sets the range itself)

CLI:
  py search_btc_hunter2_hourly_oos_champion_select.py                 # merged: IS then FULL, then verdict
  py search_btc_hunter2_hourly_oos_champion_select.py --period is     # one pass only
  py search_btc_hunter2_hourly_oos_champion_select.py --candidate BH1  # smoke one candidate
  py search_btc_hunter2_hourly_oos_champion_select.py --from-csv      # recompute from existing CSVs
  py search_btc_hunter2_hourly_oos_champion_select.py --probe-instrument  # verify auto data-range
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


WORKSPACE   = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_HUNTER_AI.wsp"
SYMBOL      = "BTCUSDT HOT"
SIGNAL      = "SFJ_HUNTER2_crypto"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\btc_hunter2_hourly_oos_champion_select_search")
PREFIX      = "BTCH2SEL_"

WIDE = DateRange("2019/01/01", "2027/01/01")   # signal date is a no-op; CHART range is the control
# IS = 2022/01/01-2026/01/01 ; FULL = 2021/03/01-2026/06/10  (user-specified).
RANGES = {"is": ("2022/01/01", "2026/01/01"), "full": ("2021/03/01", "2026/06/10")}
PERIOD_ORDER = ["is", "full"]

# micro-grid step per axis (centre = exact candidate value -> exact row exists)
STEP = {"LEN_L": 1.0, "LEN_S": 1.0, "ATR_multiplier_L": 0.25, "ATR_multiplier_S": 0.25}
LO   = {"LEN_L": 2.0, "LEN_S": 2.0, "ATR_multiplier_L": 0.1, "ATR_multiplier_S": 0.1}
HI   = {"LEN_L": 5000.0, "LEN_S": 5000.0, "ATR_multiplier_L": 30.0, "ATR_multiplier_S": 30.0}

# 4 DISTINCT converged BTC Hourly HUNTER2 regimes (from btc_hunter2_hourly{,2,3}_search).
# id, regime, LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S, prior IS NP, prior IS MDD (reference).
CANDIDATES = [
    ("BH1", "Obj-max ultra-long ***", 225.0, 825.0,  4.25, 1.75,  2720.0, -320.0),  # 192tr, R3 Obj=23,082 lowest MDD
    ("BH2", "NP-max high-ATR",         95.0, 176.0,  3.8,  11.6,  3282.0, -1351.0), # 14tr, sparse highest NP
    ("BH3", "vlong-LEN_S",            250.0, 1200.0, 5.0,  1.5,   2770.0, -419.0),  # 144tr, distinct
    ("BH4", "short-MA high-freq",      10.0, 70.0,   1.0,  5.0,   1744.0, -546.0),  # 338tr, R1 global high-freq
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_hunter2_hourly_oos_champion_select_{int(time.time())}.log"
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
        name=f"{PREFIX}{period}_{cid}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[_axis("LEN_L", ll), _axis("LEN_S", sl),
                _axis("ATR_multiplier_L", ls), _axis("ATR_multiplier_S", ss)],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=WIDE,   # signal date is a no-op; CHART trim controls the period
    )


def csv_for(cfg: StrategyConfig) -> Path:
    return OUTPUT_DIR / f"{cfg.name}_raw.csv"


_user32 = ctypes.windll.user32
_STRAY_KW = ["Optimization", "最佳化", "優化", "Optimis",
             "Format Objects", "Format Signals", "Format Strategies", "Format Signal",
             "格式物件", "格式訊號", "格式策略"]


def _cleanup_stray_windows():
    """Close leftover optimization wizard / report / Format dialog that would block the
    next candidate's right-click -> Optimize (root cause of intermittent timeouts)."""
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
            if "MultiCharts" in t:
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
    for nm, v in (("LEN_L", ll), ("LEN_S", sl),
                  ("ATR_multiplier_L", ls), ("ATR_multiplier_S", ss)):
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    if m.empty:
        return None
    return m.iloc[0]


def save_json(payload):
    out = OUTPUT_DIR / "final_params_btc_hunter2_hourly_oos_champion_select.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def run(conn, from_csv, only_candidate, only_period):
    run_t0 = time.time()
    out_path = OUTPUT_DIR / "final_params_btc_hunter2_hourly_oos_champion_select.json"

    if from_csv:
        periods = []
    elif only_period:
        periods = [only_period]
    else:
        periods = PERIOD_ORDER

    payload = {
        "symbol": SYMBOL, "signal": SIGNAL, "timeframe": "Hourly (60 min)",
        "method": "MERGED auto-range; set_instrument_data_range per pass; micro-grid MCReport CSV; "
                  "metric=Max Intraday Drawdown; OOS_NP=NP_full-NP_is; PASS=abs(MDD_full)<=abs(MDD_is).",
        "ranges": RANGES,
        "results": {},
        "candidates": {},
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
        mc.ensure_chart_ready(conn, _cfg("BH1", "is", *CANDIDATES[0][2:6]))
        if not only_period:
            for _f in OUTPUT_DIR.glob(f"{PREFIX}*_raw.csv"):
                try:
                    _f.unlink()
                except Exception:
                    pass
            payload["results"] = {}
            log.info("Cleared stale %s CSVs for a fresh merged run.", PREFIX)

    for period in periods:
        rng = RANGES[period]
        log.info("==============================================================")
        log.info("  BTC Hourly HUNTER2 — OOS CHAMPION SELECTION  [PASS: %s]  range %s ~ %s",
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
                   "params": {"LEN_L": ll, "LEN_S": sl,
                              "ATR_multiplier_L": ls, "ATR_multiplier_S": ss},
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
             "params": {"LEN_L": ll, "LEN_S": sl, "ATR_multiplier_L": ls, "ATR_multiplier_S": ss},
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
    log.info("  %-3s %-20s %9s %9s %9s %9s %4s", "id", "regime", "NP_is", "NP_full", "OOS_NP", "MDD_full", "pass")
    for c in sorted(valid, key=lambda c: c["oos_np"], reverse=True):
        log.info("  %-3s %-20s %9.0f %9.0f %9.0f %9.0f %4s", c["id"], c["regime"],
                 c["np_is"], c["np_full"], c["oos_np"], c["mdd_full"], "Y" if c["pass"] else "n")
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
    ap = argparse.ArgumentParser(description="BTC Hourly HUNTER2 OOS champion selection (auto-range merged)")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--candidate", metavar="BHn", default=None)
    ap.add_argument("--period", choices=PERIOD_ORDER, default=None,
                    help="restrict to one pass (is/full); default runs BOTH (merged)")
    ap.add_argument("--probe-instrument", action="store_true",
                    help="set the FULL range, reopen the dialog and read it back "
                         "(verify set_instrument_data_range applied)")
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
        mc.ensure_chart_ready(conn, _cfg("BH1", "is", *CANDIDATES[0][2:6]))
        tgt_from, tgt_to = RANGES["full"]
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
        print(f"  To-date persisted after reopen: {'YES' if ok_to else 'NO'}")
        print("  --> Now SCREENSHOT the chart: rightmost bar should be ~2026/06/10.")
        print("=============================================")
        return 0

    return run(conn, args.from_csv, args.candidate, args.period)


if __name__ == "__main__":
    sys.exit(main())
