"""
search_nq_ct_hourly4.py — SFJ_15Dworkshop_lesson5_countertrend_LS on CME.NQ HOT Hourly, Round 4

R3 best: LL=17, SL=0.2, LS=45, SS=1.4, NP=751,230, MDD=-64,855, Obj=8,701,665, trades=1614
R3 winner sat at TWO boundary edges of A01:
  SL=0.2 was at the LOW edge (A01 range was SL 0.2–0.8)
  LL=17 was at the HIGH edge (A01 range was LL 12–18)
New target: NP > 800,000

R4 strategy: probe SL=0.1 (absolute minimum), LL>17, and fine SS/LS
  1. SL=0.1 (never tested) — A01 stopped at 0.2
  2. LL=17–28 with SL=0.1–0.4 (LL high edge never exceeded)
  3. SS below 1.0 (tight short bands)
  4. Fine LS scan with SL=0.1–0.2
  5. Very fine dense grid around R3 champion
  6. LL below 12 with SL=0.1 (LL=7 gave NP=678K in R3-A03)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 ultra_tight_sl   : LL(12-22 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s1)×SS(0.8-1.6 s0.2) = 11×4×11×5=2420
  A02 ll_extend_high   : LL(17-28 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2) = 12×4×6×5=1440
  A03 ss_low           : LL(14-22 s1)×SL(0.1-0.3 s0.1)×LS(41-51 s2)×SS(0.3-1.2 s0.1) = 9×3×6×10=1620
  A04 fine_ls          : LL(14-20 s1)×SL(0.1-0.4 s0.1)×LS(42-52 s1)×SS(0.8-1.6 s0.2) = 7×4×11×5=1540
  A05 ls_lower         : LL(14-20 s1)×SL(0.1-0.4 s0.1)×LS(30-44 s2)×SS(0.8-1.6 s0.2) = 7×4×8×5=1120
  A06 ls_higher        : LL(14-22 s1)×SL(0.1-0.4 s0.1)×LS(50-70 s4)×SS(0.8-1.8 s0.2) = 9×4×6×6=1296
  A07 very_fine_center : LL(14-20 s1)×SL(0.1-0.5 s0.1)×LS(43-49 s1)×SS(0.8-1.8 s0.1) = 7×5×7×11=2695
  A08 short_ll         : LL(4-13 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2)  = 10×4×6×5=1200
  A09 wide_ls          : LL(12-22 s2)×SL(0.1-0.3 s0.1)×LS(25-65 s5)×SS(0.8-1.8 s0.2) = 6×3×9×6=972
  A10 adaptive_zoom    : (dynamic from R4 best NP)
  A11 ll_high_extend   : LL(18-30 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2) = 13×4×6×5=1560
  A12 global_r4        : LL(5-75 s10)×SL(0.1-1.0 s0.1)×LS(5-75 s10)×SS(0.5-2.5 s0.5) = 8×10×8×5=3200
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260521SFJ_Bollinger_AI.wsp"
SYMBOL     = "CME.NQ HOT"
SIGNAL     = "SFJ_15Dworkshop_lesson5_countertrend_LS"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_ct_hourly4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.1,  20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.1,  20.0

# R3 best as seed
SEED_LL, SEED_SL = 17.0, 0.2
SEED_LS, SEED_SS = 45.0, 1.4
SEED_NP          = 751_230.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_ct_hourly4_{int(time.time())}.log"
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


def _cfg(name: str,
         ll:  Tuple[float, float, float],
         sl:  Tuple[float, float, float],
         ls:  Tuple[float, float, float],
         ss:  Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    ll = _safe(ll, LL_LO, LL_HI)
    sl = _safe(sl, SL_LO, SL_HI)
    ls = _safe(ls, LS_LO, LS_HI)
    ss = _safe(ss, SS_LO, SS_HI)

    combos = n_vals(ll) * n_vals(sl) * n_vals(ls) * n_vals(ss)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"NQCT4_{name}",
        mc_signal_name=SIGNAL,
        timeframe="hourly",
        bar_period=60,
        params=[
            ParamAxis("LENGTH_LONG",  *ll),
            ParamAxis("STDDEV_LONG",  *sl),
            ParamAxis("LENGTH_SHORT", *ls),
            ParamAxis("STDDEV_SHORT", *ss),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"NQCT4_{name}_raw.csv"


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
    log.info("=== Starting NQCT4_%s (%d combos) ===", name, cfg.total_runs())
    t0 = time.time()
    try:
        raw_csv = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
        log.info("  Done %.1f min — %s", (time.time() - t0) / 60, Path(raw_csv).name)
        return mc.load_results_csv(raw_csv, cfg)
    except Exception as e:
        log.error("  FAILED: %s", e, exc_info=True)
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


def champion(df, fb_ll, fb_sl, fb_ls, fb_ss):
    """Priority: target met → highest NP (target-chasing mode)."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                 float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                 float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["LENGTH_LONG"]), float(best["STDDEV_LONG"]),
                float(best["LENGTH_SHORT"]), float(best["STDDEV_SHORT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_ll, fb_sl, fb_ls, fb_ss,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, ll, sl, ls, ss, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "LENGTH_LONG": ll, "STDDEV_LONG": sl,
        "LENGTH_SHORT": ls, "STDDEV_SHORT": ss,
        "objective": obj, "net_profit": np_, "max_drawdown": mdd,
        "total_trades": trades, "target_met": met,
        "combos": combos,
        "rows": len(df) if df is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }


def save_json(best_entry, attempt_log, target_met):
    payload = {
        "round": 4,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "hourly",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r3_best":  "LL=17 SL=0.2 LS=45 SS=1.4 NP=751230 MDD=-64855 trades=1614",
            "r4_focus": "SL=0.1 (minimum), LL>17, SS<1.0, fine LS — all boundary directions from R3",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_ct_hourly4.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_ll, best_sl = SEED_LL, SEED_SL
    best_ls, best_ss = SEED_LS, SEED_SS
    best_np  = SEED_NP
    best_obj = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  NQ Hourly countertrend_LS NP>800K — Round 4")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R3 best: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R3 winner edges: SL=0.2 (low boundary), LL=17 (high boundary)")
    log.info("  R4 focus: SL=0.1, LL>17, SS<1.0, fine LS/SS grid")
    log.info("  Target: %.0f USD  (gap: %.0f)", TARGET_NP, TARGET_NP - SEED_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_ll, best_sl, best_ls, best_ss
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_ll, best_sl, best_ls, best_ss,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        ll, sl, ls, ss, obj, np_, mdd, tr, met = champion(
            df, best_ll, best_sl, best_ls, best_ss)

        if np_ > best_np:
            best_ll, best_sl = ll, sl
            best_ls, best_ss = ls, ss
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, ll, sl, ls, ss,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, ll, sl, ls, ss, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/800K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  ultra_tight_sl — SL=0.1 (absolute minimum, never tested)
    #      LL(12-22 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s1)×SS(0.8-1.6 s0.2)  = 11×4×11×5 = 2420
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ultra_tight_sl"
    _c = _cfg(_n, (12, 22, 1), (0.1, 0.4, 0.1), (41, 51, 1), (0.8, 1.6, 0.2))
    log.info("A01  LL(12-22 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s1)×SS(0.8-1.6 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  ll_extend_high — LL above R3's high edge (17), probe 17-28
    #      LL(17-28 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2)  = 12×4×6×5 = 1440
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ll_extend_high"
    _c = _cfg(_n, (17, 28, 1), (0.1, 0.4, 0.1), (41, 51, 2), (0.8, 1.6, 0.2))
    log.info("A02  LL(17-28 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ss_low — SS below 1.0 (tight short bands, never explored)
    #      LL(14-22 s1)×SL(0.1-0.3 s0.1)×LS(41-51 s2)×SS(0.3-1.2 s0.1)  = 9×3×6×10 = 1620
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ss_low"
    _c = _cfg(_n, (14, 22, 1), (0.1, 0.3, 0.1), (41, 51, 2), (0.3, 1.2, 0.1))
    log.info("A03  LL(14-22 s1)×SL(0.1-0.3 s0.1)×LS(41-51 s2)×SS(0.3-1.2 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  fine_ls — fine LS scan step=1 (R3 A01 used step=2, A04 used step=1 but fewer SL)
    #      LL(14-20 s1)×SL(0.1-0.4 s0.1)×LS(42-52 s1)×SS(0.8-1.6 s0.2)  = 7×4×11×5 = 1540
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_fine_ls"
    _c = _cfg(_n, (14, 20, 1), (0.1, 0.4, 0.1), (42, 52, 1), (0.8, 1.6, 0.2))
    log.info("A04  LL(14-20 s1)×SL(0.1-0.4 s0.1)×LS(42-52 s1)×SS(0.8-1.6 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ls_lower — LS below 41 (unexplored with tight SL)
    #      LL(14-20 s1)×SL(0.1-0.4 s0.1)×LS(30-44 s2)×SS(0.8-1.6 s0.2)  = 7×4×8×5 = 1120
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ls_lower"
    _c = _cfg(_n, (14, 20, 1), (0.1, 0.4, 0.1), (30, 44, 2), (0.8, 1.6, 0.2))
    log.info("A05  LL(14-20 s1)×SL(0.1-0.4 s0.1)×LS(30-44 s2)×SS(0.8-1.6 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ls_higher — LS above 51 with tight SL (R3 never went above 51 in tight-SL regime)
    #      LL(14-22 s1)×SL(0.1-0.4 s0.1)×LS(50-70 s4)×SS(0.8-1.8 s0.2)  = 9×4×6×6 = 1296
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ls_higher"
    _c = _cfg(_n, (14, 22, 1), (0.1, 0.4, 0.1), (50, 70, 4), (0.8, 1.8, 0.2))
    log.info("A06  LL(14-22 s1)×SL(0.1-0.4 s0.1)×LS(50-70 s4)×SS(0.8-1.8 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  very_fine_center — dense grid around R3 champion (LL=17,SL=0.2,LS=45,SS=1.4)
    #      LL(14-20 s1)×SL(0.1-0.5 s0.1)×LS(43-49 s1)×SS(0.8-1.8 s0.1)  = 7×5×7×11 = 2695
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_very_fine_center"
    _c = _cfg(_n, (14, 20, 1), (0.1, 0.5, 0.1), (43, 49, 1), (0.8, 1.8, 0.1))
    log.info("A07  LL(14-20 s1)×SL(0.1-0.5 s0.1)×LS(43-49 s1)×SS(0.8-1.8 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  short_ll — LL=4-13 with tight SL (R3-A03 found LL=7,SL=0.2 gives NP=678K)
    #      LL(4-13 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2)  = 10×4×6×5 = 1200
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_short_ll"
    _c = _cfg(_n, (4, 13, 1), (0.1, 0.4, 0.1), (41, 51, 2), (0.8, 1.6, 0.2))
    log.info("A08  LL(4-13 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  wide_ls — explore LS=25-65 with tight SL
    #      LL(12-22 s2)×SL(0.1-0.3 s0.1)×LS(25-65 s5)×SS(0.8-1.8 s0.2)  = 6×3×9×6 = 972
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_wide_ls"
    _c = _cfg(_n, (12, 22, 2), (0.1, 0.3, 0.1), (25, 65, 5), (0.8, 1.8, 0.2))
    log.info("A09  LL(12-22 s2)×SL(0.1-0.3 s0.1)×LS(25-65 s5)×SS(0.8-1.8 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — zoom around best NP found in A01-A09
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (10, 0.5, 10, 0.5),
            (7,  0.4,  7, 0.4),
            (5,  0.3,  5, 0.3),
            (3,  0.2,  3, 0.2),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.1,  SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.1,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  ll_high_extend — LL=18-30 with tight SL (R3 never tested LL>17 in tight-SL regime)
    #      LL(18-30 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2)  = 13×4×6×5 = 1560
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_ll_high_extend"
    _c = _cfg(_n, (18, 30, 1), (0.1, 0.4, 0.1), (41, 51, 2), (0.8, 1.6, 0.2))
    log.info("A11  LL(18-30 s1)×SL(0.1-0.4 s0.1)×LS(41-51 s2)×SS(0.8-1.6 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  global_r4 — global sweep focused on tight SL (0.1-1.0)
    #      LL(5-75 s10)×SL(0.1-1.0 s0.1)×LS(5-75 s10)×SS(0.5-2.5 s0.5)  = 8×10×8×5 = 3200
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_global_r4"
    _c = _cfg(_n, (5, 75, 10), (0.1, 1.0, 0.1), (5, 75, 10), (0.5, 2.5, 0.5))
    log.info("A12  LL(5-75 s10)×SL(0.1-1.0 s0.1)×LS(5-75 s10)×SS(0.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  NQ Hourly countertrend_LS Round-4 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R3 seed NP: %.0f → R4 best: %.0f  (Δ %.0f)",
             SEED_NP, best_np, best_np - SEED_NP)
    log.info("  Target %.0f: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{TARGET_NP - best_np:.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LENGTH_LONG": best_ll, "STDDEV_LONG": best_sl,
            "LENGTH_SHORT": best_ls, "STDDEV_SHORT": best_ss,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>800K: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="NQ Hourly countertrend_LS NP>800K Round-4 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1–12)")
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
