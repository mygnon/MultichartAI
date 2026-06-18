"""
search_bnb_hunter2_daily2.py — SFJ_HUNTER2_crypto on BNBUSDT HOT Daily, Round 2 (IS 2022/01-2026/01)

R1 found BNB Daily's regime = ULTRA-SHORT LEN_L (LEN_L pinned at the 2 floor) and Obj rose all the
way to A11 (NOT converged): A01 LEN_L=5 Obj 81,835 -> A09/A10 LEN_L=3 Obj 106,647 -> A11 LEN_L=2 +12%.

  R1 Obj-max = NP-max ⭐ (A11): LEN_L=2 LEN_S=68 ATR_L=1.0 ATR_S=0.9  NP=$31,731 MDD=-$8,424 Obj=119,521 56tr
  (BNB Daily $31,731 > BNB Hourly $21,435 — Daily>Hourly like BNB CT; $100K unreachable -68.3%)

R2 PURPOSE: (1) confirm LEN_L=2 floor (test LEN_L 2-12 — is smaller really better?); (2) fine-sweep
LEN_S~68, ATR_L~1.0, ATR_S~0.9; (3) RE-TEST high-ATR (R1 A08 failed); (4) check the short-LEN_S
sub-regime (R1 A03 LEN_S=6); robust run_or_load (cleanup+retry+re-focus); zoom GUARANTEED <=5000.
Zoom seed = Obj-max (user criterion NP²/|MDD|).

11 attempts, each <=5000 combos. Objective = NetProfit² / |MaxDrawdown|.

Attempt schedule (daily, LEN_L floor=2):
  A01 retest_center : LEN_L(2-6 s1)×LEN_S(64-72 s2)×ATR_L(0.75-1.25 s0.25)×ATR_S(0.7-1.1 s0.1) = 5×5×3×5=375  (Rule 5 drift)
  A02 atrs_fine     : LEN_L(2-6 s1)×LEN_S(60-76 s2)×ATR_L(0.75-1.5 s0.25)×ATR_S(0.3-1.5 s0.1) = 5×9×4×13=2340
  A03 lenS_sweep    : LEN_L(2-6 s1)×LEN_S(20-120 s5)×ATR_L(0.75-1.25 s0.25)×ATR_S(0.7-1.1 s0.2) = 5×21×3×3=945
  A04 atrl_sweep    : LEN_L(2-6 s1)×LEN_S(60-76 s4)×ATR_L(0.5-3.0 s0.25)×ATR_S(0.7-1.1 s0.2) = 5×5×11×3=825
  A05 lenL_confirm  : LEN_L(2-12 s1)×LEN_S(60-80 s5)×ATR_L(0.75-1.25 s0.25)×ATR_S(0.7-1.1 s0.2) = 11×5×3×3=495
  A06 short_lenS    : LEN_L(2-10 s2)×LEN_S(2-30 s2)×ATR_L(1-3 s0.5)×ATR_S(1-3 s0.5) = 5×15×5×5=1875
  A07 high_atr_retest: LEN_L(10-130 s30)×LEN_S(10-130 s30)×ATR_L(3-12 s1)×ATR_S(3-12 s1) = 5×5×10×10=2500  (R1 A08 failed)
  A08 mid_lenS      : LEN_L(2-6 s1)×LEN_S(40-160 s10)×ATR_L(0.75-1.5 s0.25)×ATR_S(0.5-1.5 s0.25) = 5×13×4×5=1300
  A09 adaptive_zoom1: dynamic around Obj-max best (moderate; guaranteed <=5000)
  A10 adaptive_zoom2: dynamic (tighter)
  A11 adaptive_zoom3: dynamic (finest)
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
from typing import Dict, List, Tuple

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mc_automation as mc
import plateau as plateau_mod
from config import DateRange, ParamAxis, StrategyConfig


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichart64\Tim\20260101_SFJ_HUNTER_AI.wsp"
SYMBOL     = "BNBUSDT HOT"
SIGNAL     = "SFJ_HUNTER2_crypto"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\bnb_hunter2_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2027/01/01")   # WIDE no-op; chart-trim is the real control
IS_RANGE   = ("2022/01/01", "2026/01/01")

TARGET_NP  = 100_000.0   # USD

LL_LO, LL_HI = 2.0, 1000.0
LS_LO, LS_HI = 2.0, 1000.0
AL_LO, AL_HI = 0.1,   30.0
AS_LO, AS_HI = 0.1,   30.0

# R2 seed = R1 Daily Obj-max champion (A11, LEN_L at the 2 floor)
SEED_LL   = 2.0
SEED_LS   = 68.0
SEED_ATRL = 1.0
SEED_ATRS = 0.9
SEED_NP   = 31730.9
SEED_OBJ  = 119521.0

PREFIX = "BNBH2D2_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_bnb_hunter2_daily2_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

_RUN_T0 = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _snap(val: float, step: float) -> float:
    return round(round(val / step) * step, 8)


def zoom(center: float, radius: float, step: float,
         lo: float, hi: float) -> Tuple[float, float, float]:
    start = max(lo, _snap(center - radius, step))
    stop  = min(hi, _snap(center + radius, step))
    if stop <= start:
        stop = start + step
    return (start, stop, step)


def n_vals(t: Tuple[float, float, float]) -> int:
    s, e, step = t
    return max(1, round((e - s) / step) + 1)


def _cfg(name, ll, ls, atrl, atrs) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    ll   = _safe(ll,   LL_LO, LL_HI)
    ls   = _safe(ls,   LS_LO, LS_HI)
    atrl = _safe(atrl, AL_LO, AL_HI)
    atrs = _safe(atrs, AS_LO, AS_HI)

    combos = n_vals(ll) * n_vals(ls) * n_vals(atrl) * n_vals(atrs)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
            ParamAxis("LEN_L",            *ll),
            ParamAxis("LEN_S",            *ls),
            ParamAxis("ATR_multiplier_L", *atrl),
            ParamAxis("ATR_multiplier_S", *atrs),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"{PREFIX}{name}_raw.csv"


_user32 = ctypes.windll.user32
_STRAY_KW = ["Optimization", "最佳化", "優化", "Optimis"]


def _cleanup_stray_windows():
    """Close a leftover optimization wizard/report that blocks the NEXT attempt's
    right-click -> Optimize (root cause of the R1 'wizard not found in 30s' failures)."""
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
            _user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
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
    # Robust: clean strays + re-focus chart + 2 attempts (R1 had wizard-open flakiness)
    for attempt in (1, 2):
        _cleanup_stray_windows()
        try:
            mc.ensure_chart_ready(conn, cfg)   # re-focus the CHART (not Study Editor) each time
        except Exception as e:
            log.warning("  ensure_chart_ready: %s", e)
        t0 = time.time()
        try:
            raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
            log.info("  Done %.1f min — %s (attempt %d)", (time.time() - t0) / 60,
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


def _obj_max_row(df):
    """Highest-Objective row with NP>0 — R2 seeds the zoom from THIS (user criterion)."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        return None
    b = pos.loc[pos["Objective"].idxmax()]
    return {
        "LEN_L": float(b["LEN_L"]), "LEN_S": float(b["LEN_S"]),
        "ATR_multiplier_L": float(b["ATR_multiplier_L"]),
        "ATR_multiplier_S": float(b["ATR_multiplier_S"]),
        "objective": float(b["Objective"]), "net_profit": float(b["NetProfit"]),
        "max_drawdown": float(b["MaxDrawdown"]), "total_trades": int(b["TotalTrades"]),
    }


def _np_max_row(df):
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)
    pos = df[df["NetProfit"] > 0]
    if pos.empty:
        return None
    b = pos.loc[pos["NetProfit"].idxmax()]
    return {
        "LEN_L": float(b["LEN_L"]), "LEN_S": float(b["LEN_S"]),
        "ATR_multiplier_L": float(b["ATR_multiplier_L"]),
        "ATR_multiplier_S": float(b["ATR_multiplier_S"]),
        "objective": float(b["Objective"]), "net_profit": float(b["NetProfit"]),
        "max_drawdown": float(b["MaxDrawdown"]), "total_trades": int(b["TotalTrades"]),
    }


def _entry(attempt, name, df, combos, elapsed, om, nm):
    return {
        "attempt": attempt, "name": name,
        "combos": combos, "rows": len(df) if df is not None else 0,
        "obj_max": om, "np_max": nm,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }


def save_json(obj_best, np_best, attempt_log, target_met):
    payload = {
        "round": 2,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "is_window": "2022/01/01 - 2026/01/01 (chart-trimmed)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "objective_def": "NetProfit^2 / |MaxDrawdown|",
        "r1_obj_max": {"LEN_L": 2, "LEN_S": 68, "ATR_multiplier_L": 1.0, "ATR_multiplier_S": 0.9,
                       "net_profit": 31730.9, "max_drawdown": -8424.02, "objective": 119521, "total_trades": 56},
        "notes": {
            "logic": "BUY when C > AVERAGE(C,LEN_L) & EntriesToday=0 → next bar STOP at Close+ATR_L×ATR(20)",
            "exits": "Reversal only; max 1 entry per day; _Crypto1MUSD sizing",
            "purpose": "Confirm the MID-LEN Obj-max regime + retest long-LEN/ultra-long regions R1 failed; zoom seed = Obj-max",
        },
        "obj_max_champion": obj_best,   # user criterion
        "np_max_champion":  np_best,
        "run_total_sec": round(time.time() - _RUN_T0, 1),
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_bnb_hunter2_daily2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # zoom seed (tracks Obj-max regime)
    seed_ll, seed_ls, seed_atrl, seed_atrs = SEED_LL, SEED_LS, SEED_ATRL, SEED_ATRS
    obj_best: Dict = {"LEN_L": SEED_LL, "LEN_S": SEED_LS, "ATR_multiplier_L": SEED_ATRL,
                      "ATR_multiplier_S": SEED_ATRS, "net_profit": SEED_NP,
                      "max_drawdown": -8424.0, "objective": SEED_OBJ, "total_trades": 56}
    np_best: Dict = {}
    target_met = False
    attempt_log: List[dict] = []

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER2_crypto BNBUSDT HOT Daily — Round 2 (confirm ULTRA-SHORT LEN_L + fine-sweep)")
    log.info("  Seed (R1 Obj-max): LEN_L=%.0f LEN_S=%.0f ATR_L=%.2f ATR_S=%.2f  Obj=%.0f",
             SEED_LL, SEED_LS, SEED_ATRL, SEED_ATRS, SEED_OBJ)
    log.info("  Objective: NP^2/|MDD| (bigger=better). Zoom seed = Obj-max row. Robust run_or_load (retry+cleanup).")
    log.info("  Run start: %s", datetime.now().isoformat())
    log.info("══════════════════════════════════════════════════════════════")

    if not from_csv and conn is not None:
        log.info("Switching to chart + trimming data range to IS window %s ~ %s ...", *IS_RANGE)
        ok = False
        for _try in range(3):
            try:
                mc.ensure_chart_ready(conn, _cfg("seed", (2, 6, 1), (64, 72, 2),
                                                 (0.75, 1.25, 0.25), (0.7, 1.1, 0.1)))
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
            log.error("Chart-trim FAILED after 3 tries — aborting.")
            return 1
        log.info("Chart trimmed (verify leftmost ~2022/01, rightmost ~2026/01).")

    def _update(df, cfg, name, attempt_num, combos, t_attempt):
        nonlocal seed_ll, seed_ls, seed_atrl, seed_atrs
        nonlocal obj_best, np_best, target_met

        elapsed = time.time() - t_attempt
        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df, combos, elapsed, None, None))
            log.info("  [A%02d %-18s]  no valid data", attempt_num, name)
            save_json(obj_best, np_best, attempt_log, target_met)
            return

        om = _obj_max_row(df)
        nm = _np_max_row(df)

        if om and om["objective"] > obj_best.get("objective", -1):
            obj_best = {**om, "attempt": attempt_num, "name": name}
            seed_ll, seed_ls   = om["LEN_L"], om["LEN_S"]
            seed_atrl, seed_atrs = om["ATR_multiplier_L"], om["ATR_multiplier_S"]
        if nm and (not np_best or nm["net_profit"] > np_best.get("net_profit", -1)):
            np_best = {**nm, "attempt": attempt_num, "name": name}
            if nm["net_profit"] > TARGET_NP:
                target_met = True

        attempt_log.append(_entry(attempt_num, name, df, combos, elapsed, om, nm))

        log.info("  [A%02d %-18s] obj_max LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g "
                 "NP=%.0f MDD=%.0f Obj=%.0f tr=%d  (%.1fs)",
                 attempt_num, name,
                 om["LEN_L"] if om else 0, om["LEN_S"] if om else 0,
                 om["ATR_multiplier_L"] if om else 0, om["ATR_multiplier_S"] if om else 0,
                 om["net_profit"] if om else 0, om["max_drawdown"] if om else 0,
                 om["objective"] if om else 0, om["total_trades"] if om else 0, elapsed)
        log.info("       Best Obj so far=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)  | NP-max=%.0f",
                 obj_best.get("objective", 0), seed_ll, seed_ls, seed_atrl, seed_atrs,
                 np_best.get("net_profit", 0))
        save_json(obj_best, np_best, attempt_log, target_met)

    def _do(A, name, cfg):
        log.info("A%02d  %s  %d combos", A, name, cfg.total_runs())
        if start_attempt <= A:
            t = time.time()
            _update(run_or_load(name, cfg, conn, from_csv), cfg, name, A, cfg.total_runs(), t)

    # static attempts
    _do(1, "01_retest_center",  _cfg("01_retest_center",  (2,6,1),    (64,72,2),    (0.75,1.25,0.25),(0.7,1.1,0.1)))
    _do(2, "02_atrs_fine",      _cfg("02_atrs_fine",      (2,6,1),    (60,76,2),    (0.75,1.5,0.25), (0.3,1.5,0.1)))
    _do(3, "03_lenS_sweep",     _cfg("03_lenS_sweep",     (2,6,1),    (20,120,5),   (0.75,1.25,0.25),(0.7,1.1,0.2)))
    _do(4, "04_atrl_sweep",     _cfg("04_atrl_sweep",     (2,6,1),    (60,76,4),    (0.5,3.0,0.25),  (0.7,1.1,0.2)))
    _do(5, "05_lenL_confirm",   _cfg("05_lenL_confirm",   (2,12,1),   (60,80,5),    (0.75,1.25,0.25),(0.7,1.1,0.2)))
    _do(6, "06_short_lenS",     _cfg("06_short_lenS",     (2,10,2),   (2,30,2),     (1,3,0.5),       (1,3,0.5)))
    _do(7, "07_high_atr_retest",_cfg("07_high_atr_retest",(10,130,30),(10,130,30),  (3,12,1),        (3,12,1)))
    _do(8, "08_mid_lenS",       _cfg("08_mid_lenS",       (2,6,1),    (40,160,10),  (0.75,1.5,0.25), (0.5,1.5,0.25)))

    # adaptive zooms around Obj-max seed
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  zoom1 — center LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  Obj=%.0f",
             seed_ll, seed_ls, seed_atrl, seed_atrs, obj_best.get("objective", 0))
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (50, 60, 1.0, 0.6), (40, 50, 0.75, 0.5), (30, 40, 0.5, 0.4),
            (20, 30, 0.5, 0.3), (15, 20, 0.375, 0.25),
        ]:
            _ll=zoom(seed_ll,r_ll,5,LL_LO,LL_HI); _ls=zoom(seed_ls,r_ls,5,LS_LO,LS_HI)
            _al=zoom(seed_atrl,r_atrl,0.25,AL_LO,AL_HI); _as=zoom(seed_atrs,r_atrs,0.1,AS_LO,AS_HI)
            _c=_cfg(_n,_ll,_ls,_al,_as)
            if _c.total_runs()<=5000: break
        log.info("A09  LL%s LS%s ATR_L%s ATR_S%s  %d combos",_ll,_ls,_al,_as,_c.total_runs())
        t=time.time(); _update(run_or_load(_n,_c,conn,from_csv),_c,_n,A,_c.total_runs(),t)

    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  zoom2 — center LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  Obj=%.0f",
             seed_ll, seed_ls, seed_atrl, seed_atrs, obj_best.get("objective", 0))
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (30, 40, 0.6, 0.4), (24, 30, 0.5, 0.3), (18, 24, 0.375, 0.25),
            (12, 18, 0.25, 0.2), (10, 12, 0.25, 0.15),
        ]:
            _ll=zoom(seed_ll,r_ll,5,LL_LO,LL_HI); _ls=zoom(seed_ls,r_ls,5,LS_LO,LS_HI)
            _al=zoom(seed_atrl,r_atrl,0.25,AL_LO,AL_HI); _as=zoom(seed_atrs,r_atrs,0.1,AS_LO,AS_HI)
            _c=_cfg(_n,_ll,_ls,_al,_as)
            if _c.total_runs()<=5000: break
        log.info("A10  LL%s LS%s ATR_L%s ATR_S%s  %d combos",_ll,_ls,_al,_as,_c.total_runs())
        t=time.time(); _update(run_or_load(_n,_c,conn,from_csv),_c,_n,A,_c.total_runs(),t)

    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  zoom3 — center LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  Obj=%.0f",
             seed_ll, seed_ls, seed_atrl, seed_atrs, obj_best.get("objective", 0))
    if start_attempt <= A:
        for r_ll, r_ls, r_atrl, r_atrs in [
            (20, 25, 0.4, 0.3), (15, 20, 0.3, 0.25), (12, 15, 0.25, 0.2),
            (10, 12, 0.2, 0.15), (8, 10, 0.2, 0.1),
        ]:
            _ll=zoom(seed_ll,r_ll,5,LL_LO,LL_HI); _ls=zoom(seed_ls,r_ls,5,LS_LO,LS_HI)
            _al=zoom(seed_atrl,r_atrl,0.1,AL_LO,AL_HI); _as=zoom(seed_atrs,r_atrs,0.1,AS_LO,AS_HI)
            _c=_cfg(_n,_ll,_ls,_al,_as)
            if _c.total_runs()<=5000: break
        log.info("A11  LL%s LS%s ATR_L%s ATR_S%s  %d combos",_ll,_ls,_al,_as,_c.total_runs())
        t=time.time(); _update(run_or_load(_n,_c,conn,from_csv),_c,_n,A,_c.total_runs(),t)

    # summary
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  BNB HUNTER2 Daily Round-2 COMPLETE  (%.1f min total)", (time.time()-_RUN_T0)/60)
    log.info("  Obj-max Champion (your criterion): LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  "
             "NP=%.0f MDD=%.0f Obj=%.0f tr=%d",
             obj_best["LEN_L"], obj_best["LEN_S"], obj_best["ATR_multiplier_L"],
             obj_best["ATR_multiplier_S"], obj_best["net_profit"], obj_best["max_drawdown"],
             obj_best["objective"], obj_best["total_trades"])
    if np_best:
        log.info("  NP-max Champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f Obj=%.0f",
                 np_best["LEN_L"], np_best["LEN_S"], np_best["ATR_multiplier_L"],
                 np_best["ATR_multiplier_S"], np_best["net_profit"], np_best["objective"])
    log.info("  R1 Obj-max=119,521  →  R2 Obj-max=%.0f  (%+.1f%%)",
             obj_best["objective"], (obj_best["objective"]/SEED_OBJ - 1) * 100)
    log.info("══════════════════════════════════════════════════════════════")

    out = save_json(obj_best, np_best, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Obj-max champion Obj={obj_best['objective']:.0f}  (R1 119,521)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

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
    print("[auto-elevate] Requesting elevation — approve the UAC prompt.")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}). Run as Administrator manually.")
    else:
        print("[auto-elevate] Elevated process launched.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="SFJ_HUNTER2_crypto BNBUSDT HOT Daily Round-2 (confirm ultra-short LEN_L + fine-sweep)")
    ap.add_argument("--from-csv",  action="store_true")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N")
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
