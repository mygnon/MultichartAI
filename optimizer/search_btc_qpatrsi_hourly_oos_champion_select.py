"""
search_btc_qpatrsi_hourly_oos_champion_select.py — OOS selection among the converged
BTC Hourly QuantPassRSI champions (R1-R2, results/btc_qpatrsi_hourly{,2}_search/).

QuantPassRSI (2 params Len, RSI_Gap; momentum: RSI>100-Gap -> BUY market, RSI<Gap -> SHORT market,
reversal exits, _Crypto1MUSD). IS converged to Obj-max Len=58 RSI_Gap=33 (sparse 15tr, low MDD).
Candidates span: the Obj-max mid-len, the NP-max short-len, the high-gap DENSE regime (195tr), and a
very-long-len — so the OOS test reveals which holds its Max Drawdown out-of-sample.

TASK: re-test each previously-converged champion and pick the one that earns the MOST out-of-sample
profit WITHOUT its drawdown breaking the established Max Drawdown.

METHOD — MERGED single-run, AUTO data-range:
    PASS is  : Data Range 2022/01/01 - 2026/01/01  -> NP_is,   MDD_is   (per candidate)
    PASS full: Data Range 2021/03/01 - 2026/06/10  -> NP_full, MDD_full
    OOS_NP = NP_full - NP_is ;  PASS when abs(MDD_full) <= abs(MDD_is)
  Each candidate = a 9-combo micro-grid (both params +/-1 step, Critical Rule 1); exact row picked.
  Metric = Max Intraday Drawdown (same as the IS rounds). Winner = max OOS_NP among PASS.

PREREQUISITES (MC64, elevated):
  - 20260101_QuantPassRSI_AI.wsp open, BTCUSDT HOT 60-Minute chart tab ACTIVE, QuantPassRSI signal ON
  - Full data to 2026/06/10 (the script sets the range itself). CLOSE the Study Editor window.

CLI:
  py search_btc_qpatrsi_hourly_oos_champion_select.py                 # merged: IS then FULL, verdict
  py search_btc_qpatrsi_hourly_oos_champion_select.py --candidate RS1 # smoke one candidate
  py search_btc_qpatrsi_hourly_oos_champion_select.py --from-csv      # recompute from existing CSVs
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
from typing import Dict, List

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mc_automation as mc
from config import DateRange, ParamAxis, StrategyConfig


WORKSPACE   = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_QuantPassRSI_AI.wsp"
SYMBOL      = "BTCUSDT HOT"
SIGNAL      = "QuantPassRSI"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\btc_qpatrsi_hourly_oos_champion_select_search")
PREFIX      = "BTCRSISEL_"

WIDE = DateRange("2019/01/01", "2027/01/01")   # signal date is a no-op; CHART range is the control
RANGES = {"is": ("2022/01/01", "2026/01/01"), "full": ("2021/03/01", "2026/06/10")}
PERIOD_ORDER = ["is", "full"]

STEP = {"Len": 1.0, "RSI_Gap": 1.0}
LO   = {"Len": 2.0, "RSI_Gap": 1.0}
HI   = {"Len": 5000.0, "RSI_Gap": 99.0}

# 4 converged BTC Hourly QuantPassRSI candidates (btc_qpatrsi_hourly{,2}_search).
# id, regime, Len, RSI_Gap, prior IS NP, prior IS MDD (reference).
CANDIDATES = [
    ("RS1", "R2 Obj-max mid-len ***", 58.0, 33.0, 2010.0, -624.0),  # 15tr, R2 ceiling Obj=6,468 lowest MDD
    ("RS2", "NP-max short-len",         7.0,  5.0, 2207.0, -1156.0), # 11tr, highest IS NP, deep MDD
    ("RS3", "high-gap DENSE",          28.0, 38.0, 1372.0, -497.0),  # 195tr, lowest IS MDD, most active
    ("RS4", "very-long-len",          290.0, 45.0, 1633.0, -674.0),  # 11tr, long-RSI contrast
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_btc_qpatrsi_hourly_oos_champion_select_{int(time.time())}.log"
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


def _cfg(cid, period, length, gap) -> StrategyConfig:
    return StrategyConfig(
        name=f"{PREFIX}{period}_{cid}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[_axis("Len", length), _axis("RSI_Gap", gap)],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=WIDE,
    )


def csv_for(cfg: StrategyConfig) -> Path:
    return OUTPUT_DIR / f"{cfg.name}_raw.csv"


_user32 = ctypes.windll.user32
_STRAY_KW = ["Optimization", "最佳化", "優化", "Optimis"]


def _cleanup_stray_windows():
    try:
        mc._close_optimization_report()
    except Exception:
        pass
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
                victims.append(hwnd)
        except Exception:
            pass
        return True

    try:
        _user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)(_cb), 0)
    except Exception:
        pass
    for hwnd in victims:
        try:
            _user32.PostMessageW(hwnd, 0x0010, 0, 0)
            time.sleep(0.4)
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
        try:
            mc.ensure_chart_ready(conn, cfg)
        except Exception as e:
            log.warning("  ensure_chart_ready: %s", e)
        t0 = time.time()
        try:
            raw = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
            log.info("  Done %.1f min (attempt %d)", (time.time() - t0) / 60, attempt)
            return mc.load_results_csv(raw, cfg)
        except Exception as e:
            log.warning("  attempt %d FAILED: %s", attempt, e)
            if attempt == 2:
                log.error("  %s: giving up after 2 attempts", cfg.name)
    return None


def _pick(df, length, gap):
    m = df
    for nm, v in (("Len", length), ("RSI_Gap", gap)):
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    if m.empty:
        return None
    return m.iloc[0]


def save_json(payload):
    out = OUTPUT_DIR / "final_params_btc_qpatrsi_hourly_oos_champion_select.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def run(conn, from_csv, only_candidate, only_period):
    run_t0 = time.time()
    out_path = OUTPUT_DIR / "final_params_btc_qpatrsi_hourly_oos_champion_select.json"
    periods = [] if from_csv else ([only_period] if only_period else PERIOD_ORDER)

    payload = {
        "symbol": SYMBOL, "signal": SIGNAL, "timeframe": "Hourly (60 min)",
        "method": "MERGED auto-range; set_instrument_data_range per pass; micro-grid MCReport CSV; "
                  "metric=Max Intraday Drawdown; OOS_NP=NP_full-NP_is; PASS=abs(MDD_full)<=abs(MDD_is).",
        "ranges": RANGES, "results": {}, "candidates": {}, "winner": None,
        "run_started_at": datetime.now().isoformat(),
        "run_finished_at": None, "run_total_sec": None,
    }
    if out_path.exists():
        try:
            payload["results"] = json.load(open(out_path, encoding="utf-8")).get("results", {})
        except Exception:
            pass

    if not from_csv:
        mc.ensure_chart_ready(conn, _cfg("RS1", "is", *CANDIDATES[0][2:4]))
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
        log.info("  BTC Hourly QuantPassRSI -- OOS CHAMPION SELECTION  [PASS: %s]  range %s ~ %s",
                 period, rng[0], rng[1])
        log.info("==============================================================")
        try:
            mc.ensure_chart_ready(conn, _cfg(CANDIDATES[0][0], period, *CANDIDATES[0][2:4]))
            mc.set_instrument_data_range(conn, rng[0], rng[1])
        except Exception as e:
            log.error("  set_instrument_data_range FAILED for %s: %s -- aborting pass", period, e)
            continue
        pres = payload["results"].get(period, {})
        for (cid, regime, length, gap, pnp, pmdd) in CANDIDATES:
            if only_candidate and cid != only_candidate:
                continue
            t0 = time.time()
            cfg = _cfg(cid, period, length, gap)
            log.info("--- [%s] %s %s Len=%g RSI_Gap=%g [%s] ---",
                     period, cid, regime, length, gap, datetime.now().strftime("%H:%M:%S"))
            df = run_or_load(cfg, conn, from_csv)
            ent = {"id": cid, "regime": regime,
                   "params": {"Len": length, "RSI_Gap": gap},
                   "timestamp": datetime.now().isoformat(),
                   "elapsed_sec": round(time.time() - t0, 1),
                   "rows": len(df) if df is not None else 0}
            row = _pick(df, length, gap) if (df is not None and not df.empty) else None
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
    for (cid, regime, length, gap, pnp, pmdd) in CANDIDATES:
        ci, cf = is_r.get(cid), full_r.get(cid)
        c = {"id": cid, "regime": regime, "params": {"Len": length, "RSI_Gap": gap},
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

    log.info("==============================================================")
    log.info("  SUMMARY (sorted by OOS_NP)   total %.1f min", (time.time() - run_t0) / 60)
    log.info("  %-3s %-22s %9s %9s %9s %9s %4s", "id", "regime", "NP_is", "NP_full", "OOS_NP", "MDD_full", "pass")
    for c in sorted(valid, key=lambda c: c["oos_np"], reverse=True):
        log.info("  %-3s %-22s %9.0f %9.0f %9.0f %9.0f %4s", c["id"], c["regime"],
                 c["np_is"], c["np_full"], c["oos_np"], c["mdd_full"], "Y" if c["pass"] else "n")
    if valid and all(abs(c["np_full"] - c["np_is"]) < 1e-6 for c in valid):
        log.warning("  !! IS==FULL for ALL candidates -> data-range did NOT apply.")
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
    ap = argparse.ArgumentParser(description="BTC Hourly QuantPassRSI OOS champion selection")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--candidate", metavar="RSn", default=None)
    ap.add_argument("--period", choices=PERIOD_ORDER, default=None)
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0
    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()
    return run(conn, args.from_csv, args.candidate, args.period)


if __name__ == "__main__":
    sys.exit(main())
