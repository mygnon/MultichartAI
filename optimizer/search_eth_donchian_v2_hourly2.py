"""
search_eth_donchian_v2_hourly.py -- SFJ_DonchianATR_crypto_v2 on ETHUSDT HOT Hourly, R1 (IS 2022/01-2026/01)

v1 lost on EVERY combo (whipsaw, 119-3,602 trades). v2 adds 3 anti-chop filters, all optimizable:
  Length    - Donchian breakout lookback (STOP entry at prior channel edge)
  ATRLength - ATR period for the chandelier trailing stop
  ATRMult   - trailing-stop distance (ATR multiples)            [fractional]
  TrendLen  - trend-filter MA period (long only above MA, short only below)
  ReentryBars - flat-bar cooldown after an exit before a new entry
Long+short, MARKET-stop entries, _Crypto1MUSD sizing.

Target NP > 100,000 USD. Objective = NetProfit^2 / |MaxDrawdown| (bigger=better). <=5000 combos/attempt.
IS window 2022/01/01-2026/01/01 chart-trimmed. Reports BOTH NP-max and Obj-max champions.

Attempt schedule (12 attempts, 5D grids, <=5000):
  A01 global_wide  L(10-60 s10) ATRL(10-20 s10) M(2-4 s1)    T(50-200 s50) R(0-10 s5)  = 432
  A02 default_zone L(10-30 s5)  ATRL(10-20 s5)  M(2-4 s0.5)  T(60-140 s20) R(0-8 s4)   = 1125
  A03 len_mult     L(5-50 s5)   ATRL(10-18 s4)  M(1.5-5 s0.5)T(80-120 s40) R(2-8 s3)   = 1440
  A04 trend_filter L(15-25 s5)  ATRL(10-18 s4)  M(2.5-4 s0.5)T(30-300 s30) R(0-8 s4)   = 1080
  A05 reentry_scan L(15-25 s5)  ATRL(10-18 s4)  M(2.5-4 s0.5)T(80-160 s40) R(0-30 s3)  = 1188
  A06 mult_fine    L(15-30 s5)  ATRL(10-18 s4)  M(1-8 s0.5)  T(80-120 s40) R(3-9 s3)   = 1080
  A07 short_len    L(5-25 s2)   ATRL(8-16 s4)   M(2-4 s0.5)  T(50-150 s50) R(2-8 s3)   = 1485
  A08 long_len     L(40-120 s10)ATRL(10-20 s5)  M(2-5 s0.5)  T(100-300 s100)R(0-10 s5) = 1701
  A09 global_bound L(5-150 s15) ATRL(6-30 s8)   M(1-6 s1)    T(40-280 s60) R(0-20 s10) = 3960
  A10-A12 adaptive zoom (5D) around the running Obj-max
"""
from __future__ import annotations
import argparse
import ctypes
import json
import logging
import math
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260622_SFJ_DonchianATR_AI.wsp"
SYMBOL     = "ETHUSDT HOT"
SIGNAL     = "SFJ_DonchianATR_crypto_v2"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\eth_donchian_v2_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LEN_LO,  LEN_HI  = 2.0,   500.0
ATRL_LO, ATRL_HI = 2.0,   100.0
MULT_LO, MULT_HI = 0.5,   20.0
TR_LO,   TR_HI   = 5.0,   1000.0
RE_LO,   RE_HI   = 0.0,   200.0

SEED_LEN, SEED_ATRL, SEED_MULT, SEED_TR, SEED_RE = 160.0, 22.0, 2.0, 145.0, 2.0

PREFIX = "ETHDONV22_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_eth_donchian_v2_hourly2_{int(time.time())}.log"
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
_RUN_T0 = time.time()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zoom_fixed(center, radius, n_target, step_min, lo, hi):
    lo_val = max(lo, center - radius)
    hi_val = min(hi, center + radius)
    if lo_val >= hi_val:
        return (max(lo, center - step_min), min(hi, center + step_min), step_min)
    rng = hi_val - lo_val
    step = max(step_min, math.ceil(rng / max(1, n_target - 1) / step_min) * step_min)
    return (lo_val, hi_val, step)


def n_vals(t):
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name, length, atrl, mult, trend, reentry) -> StrategyConfig:
    def _safe(t, lo, hi, step_min):
        s, e, step = t
        if abs(s - e) < step_min * 0.5:
            return (max(lo, s - step), min(hi, s + step), step)
        return t
    length  = _safe(length,  LEN_LO,  LEN_HI,  1.0)
    atrl    = _safe(atrl,    ATRL_LO, ATRL_HI, 1.0)
    mult    = _safe(mult,    MULT_LO, MULT_HI, 0.25)
    trend   = _safe(trend,   TR_LO,   TR_HI,   1.0)
    reentry = _safe(reentry, RE_LO,   RE_HI,   1.0)
    combos = n_vals(length) * n_vals(atrl) * n_vals(mult) * n_vals(trend) * n_vals(reentry)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[ParamAxis("Length",      *length),
                ParamAxis("ATRLength",   *atrl),
                ParamAxis("ATRMult",     *mult),
                ParamAxis("TrendLen",    *trend),
                ParamAxis("ReentryBars", *reentry)],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name) -> Path:
    return OUTPUT_DIR / f"{PREFIX}{name}_raw.csv"


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
    for attempt in (1, 2):
        _cleanup_stray_windows()
        try:
            mc.ensure_chart_ready(conn, cfg)
        except Exception as e:
            log.warning("  ensure_chart_ready: %s", e)
        t0 = time.time()
        try:
            raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
            log.info("  Done %.1f min -- %s (attempt %d)", (time.time() - t0) / 60,
                     Path(raw_csv).name, attempt)
            return mc.load_results_csv(raw_csv, cfg)
        except Exception as e:
            log.warning("  attempt %d FAILED: %s", attempt, e)
            if attempt == 2:
                log.error("  %s: giving up after 2 attempts", name)
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


def _row(df, idx):
    b = df.loc[idx]
    return {"Length": float(b["Length"]), "ATRLength": float(b["ATRLength"]),
            "ATRMult": float(b["ATRMult"]), "TrendLen": float(b["TrendLen"]),
            "ReentryBars": float(b["ReentryBars"]),
            "objective": float(b["Objective"]), "net_profit": float(b["NetProfit"]),
            "max_drawdown": float(b["MaxDrawdown"]), "total_trades": int(b["TotalTrades"])}


def _np_max_row(df):
    df = df.copy(); df["Objective"] = plateau_mod.compute_objective(df)
    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        return None
    return _row(pos, pos["NetProfit"].idxmax())


def _obj_max_row(df):
    df = df.copy(); df["Objective"] = plateau_mod.compute_objective(df)
    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        return None
    return _row(pos, pos["Objective"].idxmax())


def save_json(obj_best, np_best, attempt_log, target_met):
    payload = {
        "round": 2, "strategy": SIGNAL, "symbol": SYMBOL, "timeframe": "Hourly (60 min)",
        "is_window": "2022/01/01 - 2026/01/01 (chart-trimmed)",
        "target_np": TARGET_NP, "target_met": target_met,
        "objective_def": "NetProfit^2 / |MaxDrawdown|",
        "notes": {
            "logic": "Donchian STOP-breakout entries (prior channel edge), trend-filtered (Close vs MA(TrendLen)), flat-only + ReentryBars cooldown; ATR chandelier trailing exits; long+short",
            "contract": "_Crypto1MUSD = Round(1000000/C, 0) ~ $1M notional/trade",
            "params": "Length, ATRLength, ATRMult (frac), TrendLen, ReentryBars; defaults 20/14/3.0/100/5",
            "v1_note": "v1 (no filters) lost on every combo (119-3602 trades, whipsaw); v2 adds trend filter + STOP entry + cooldown",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_eth_donchian_v2_hourly2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sL, sA, sM, sT, sR = SEED_LEN, SEED_ATRL, SEED_MULT, SEED_TR, SEED_RE
    obj_best: Dict = {}
    np_best: Dict = {}
    target_met = False
    attempt_log: List[dict] = []

    log.info("==============================================================")
    log.info("  SFJ_DonchianATR_crypto_v2 on ETHUSDT HOT Hourly -- Round 2 boundary confirm (IS 2022/01-2026/01)")
    log.info("  Params: Length, ATRLength, ATRMult, TrendLen, ReentryBars.  Obj=NP^2/|MDD|.  Target %.0f", TARGET_NP)
    log.info("  R1: Obj-max L160/ATRL22/M2/T145/R2 Obj8276 (412tr, TIGHT trail); NP-max L20/M6 $2004.")
    log.info("  R2 confirm ultra-long+M2 + NP-max short-L regime.  Run start: %s", datetime.now().isoformat())
    log.info("==============================================================")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (155, 165, 5), (20, 24, 2),
                                                 (1.5, 2.5, 0.5), (135, 155, 10), (0, 4, 2)))
                mc.set_instrument_data_range(conn, IS_RANGE[0], IS_RANGE[1])
                ok = True
                break
            except Exception as e:
                log.warning("  chart-trim attempt %d failed: %s", _try + 1, e)
                try:
                    mc._close_optimization_report()
                except Exception:
                    pass
                time.sleep(1.0)
        if not ok:
            log.error("Chart-trim FAILED after 3 tries -- aborting.")
            return 1
        log.info("Chart trimmed (verify leftmost ~2022/01, rightmost ~2026/01).")

    def _update(df, cfg, name, attempt_num, combos, t_attempt):
        nonlocal sL, sA, sM, sT, sR, obj_best, np_best, target_met
        elapsed = time.time() - t_attempt
        ok = df is not None and not df.empty and _validate_df(df, cfg)
        om = _obj_max_row(df) if ok else None
        nm = _np_max_row(df) if om else None
        entry = {"attempt": attempt_num, "name": name, "combos": combos,
                 "rows": len(df) if df is not None else 0,
                 "obj_max": om, "np_max": nm,
                 "elapsed_sec": round(elapsed, 1), "timestamp": datetime.now().isoformat()}
        attempt_log.append(entry)
        if om is None:
            log.info("  [A%02d %-16s] no valid data", attempt_num, name)
            save_json(obj_best, np_best, attempt_log, target_met)
            return
        if not obj_best or om["objective"] > obj_best.get("objective", -1):
            obj_best = {**om, "attempt": attempt_num, "name": name}
            sL, sA, sM, sT, sR = (om["Length"], om["ATRLength"], om["ATRMult"],
                                  om["TrendLen"], om["ReentryBars"])
        if not np_best or nm["net_profit"] > np_best.get("net_profit", -1):
            np_best = {**nm, "attempt": attempt_num, "name": name}
            if nm["net_profit"] > TARGET_NP:
                target_met = True
        log.info("  [A%02d %-16s] obj_max L=%.4g ATRL=%.4g M=%.4g T=%.4g R=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d (%.1fs)",
                 attempt_num, name, om["Length"], om["ATRLength"], om["ATRMult"], om["TrendLen"],
                 om["ReentryBars"], om["net_profit"], om["max_drawdown"], om["objective"],
                 om["total_trades"], elapsed)
        log.info("       Best Obj=%.0f | NP-max=%.0f gap_to_target=%.0f",
                 obj_best.get("objective", 0), np_best.get("net_profit", 0),
                 max(0, TARGET_NP - np_best.get("net_profit", 0)))
        save_json(obj_best, np_best, attempt_log, target_met)

    def _do(A, name, cfg):
        log.info("A%02d  %s  %d combos", A, name, cfg.total_runs())
        if start_attempt <= A:
            t = time.time()
            _update(run_or_load(name, cfg, conn, from_csv), cfg, name, A, cfg.total_runs(), t)

    # static attempts (5D R2: ultra-long+M2 (Obj) and short-L+M6 (NP) regimes)
    _do(1, "01_rule5_retest", _cfg("01_rule5_retest", (150,180,5), (20,24,2), (1.5,3,0.5), (125,165,20),(0,4,2)))   # 2268
    _do(2, "02_len_push",     _cfg("02_len_push",     (150,260,10),(20,24,2), (1.5,3,0.5), (125,165,20),(0,4,2)))   # push L>160
    _do(3, "03_mult_tight",   _cfg("03_mult_tight",   (150,180,5), (20,24,2), (1,4,0.25),  (140,160,20),(0,4,2)))   # M fine near 2
    _do(4, "04_trend_zoom",   _cfg("04_trend_zoom",   (150,180,10),(20,24,2), (1.5,3,0.5), (80,220,20), (0,4,2)))   # TrendLen fine
    _do(5, "05_atrl_fine",    _cfg("05_atrl_fine",    (150,180,10),(12,30,2), (1.5,3,0.5), (130,160,15),(0,4,2)))   # ATRL fine
    _do(6, "06_reentry_scan", _cfg("06_reentry_scan", (155,165,5), (20,24,2), (1.5,3,0.5), (135,155,10),(0,20,2)))  # R 0-20
    _do(7, "07_npmax_zoom",   _cfg("07_npmax_zoom",   (10,40,2),   (18,24,3), (4,8,0.5),   (30,60,15),  (0,2,2)))   # zoom NP-max short-L
    _do(8, "08_npmax_mult",   _cfg("08_npmax_mult",   (12,32,2),   (18,24,3), (4,9,0.5),   (30,50,10),  (0,2,2)))   # NP-max M plane
    _do(9, "09_vlong_len",    _cfg("09_vlong_len",    (180,360,15),(20,24,2), (1.5,3,0.5), (140,200,30),(0,2,2)))   # very-long L boundary

    # adaptive zooms around the running Obj-max
    for A, nm_, (rL, rA, rM, rT, rR, nL, nA, nM, nT, nR) in [
        (10, "10_adaptive_zoom1", (12, 6, 1.5, 60, 8, 7, 3, 7, 5, 3)),
        (11, "11_adaptive_zoom2", (6,  4, 1.0, 30, 5, 7, 3, 7, 5, 3)),
        (12, "12_adaptive_zoom3", (3,  2, 0.5, 16, 3, 7, 3, 7, 5, 3)),
    ]:
        log.info("A%02d  %s -- center L=%.4g ATRL=%.4g M=%.4g T=%.4g R=%.4g Obj=%.0f",
                 A, nm_, sL, sA, sM, sT, sR, obj_best.get("objective", 0))
        if start_attempt <= A:
            _L = zoom_fixed(sL, rL, nL, 1.0,  LEN_LO,  LEN_HI)
            _A = zoom_fixed(sA, rA, nA, 1.0,  ATRL_LO, ATRL_HI)
            _M = zoom_fixed(sM, rM, nM, 0.25, MULT_LO, MULT_HI)
            _T = zoom_fixed(sT, rT, nT, 1.0,  TR_LO,   TR_HI)
            _R = zoom_fixed(sR, rR, nR, 1.0,  RE_LO,   RE_HI)
            _c = _cfg(nm_, _L, _A, _M, _T, _R)
            log.info("A%02d  L%s ATRL%s M%s T%s R%s  %d combos", A, _L, _A, _M, _T, _R, _c.total_runs())
            t = time.time()
            _update(run_or_load(nm_, _c, conn, from_csv), _c, nm_, A, _c.total_runs(), t)

    log.info("==============================================================")
    log.info("  SFJ_DonchianATR_crypto_v2 ETHUSDT Hourly Round-2 COMPLETE (%.1f min)", (time.time()-_RUN_T0)/60)
    if obj_best:
        log.info("  Obj-max (your criterion): L=%.4g ATRL=%.4g M=%.4g T=%.4g R=%.4g NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
                 obj_best["Length"], obj_best["ATRLength"], obj_best["ATRMult"], obj_best["TrendLen"],
                 obj_best["ReentryBars"], obj_best["net_profit"], obj_best["max_drawdown"],
                 obj_best["objective"], obj_best["total_trades"])
    if np_best:
        log.info("  NP-max: L=%.4g ATRL=%.4g M=%.4g T=%.4g R=%.4g NP=%.0f Obj=%.0f",
                 np_best["Length"], np_best["ATRLength"], np_best["ATRMult"], np_best["TrendLen"],
                 np_best["ReentryBars"], np_best["net_profit"], np_best["objective"])
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "MET" if target_met else "NOT MET (best NP %.0f)" % np_best.get("net_profit", 0))
    log.info("==============================================================")
    out = save_json(obj_best, np_best, attempt_log, target_met)
    print(f"\nDone -- results at: {out}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    ap = argparse.ArgumentParser(description="SFJ_DonchianATR_crypto_v2 ETHUSDT HOT Hourly R1 search")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--attempt", type=int, default=1, metavar="N")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.from_csv and not _is_admin():
        _auto_elevate()
        return 0
    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection()
        conn.connect()
    return run_search(conn, args.from_csv, args.attempt)


if __name__ == "__main__":
    sys.exit(main())
