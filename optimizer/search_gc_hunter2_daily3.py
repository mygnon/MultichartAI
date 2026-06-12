"""
search_gc_hunter2_daily3.py — SFJ_HUNTER2_NQ on CME.GC HOT Daily, Round 3

R2 champion: LEN_L=7, LEN_S=5, ATR_L=0.295, ATR_S=2.07
  NP=$338,710  MDD=-$38,660  Obj=2,967,524  trades=51/7yr  gain R1→R2=+1.09%
  → R2 A09→A10→A11 still improving (335K→337K→338K); triple convergence not yet proven.
  → ATR_S=2.07 confirmed sharp peak: 2.06→332K regime switch, 2.08→337K declining.
  → ATR_L broad plateau 0.295–0.320 ($280 spread); below 0.290 switches to MDD=-44K regime.
  → MDD=-$38,660 structurally fixed: same worst-drawdown trade across all top results.

R3 goals:
  1. Ultra-fine ATR step=0.005 (ATR_S) and step=0.002 (ATR_L) around champion
  2. Check if 2.07 is truly the ATR_S peak at step=0.002 resolution
  3. Check ATR_L boundary transition at 0.290 precisely
  4. Bridge scan of LL×LS space at champion ATR to confirm no unexplored pair
  5. Triple-convergence proof via A09=A10=A11 (if identical → ceiling confirmed)

Expected outcome: ~$338K–$340K with A09=A10=A11 convergence → GC Daily HUNTER2 ceiling.

Attempt schedule (11 attempts, ≤5,000 combos each):
  A01 fine_atrs_precision : LL(5-9 s1)×LS(3-7 s1)×ATR_L(0.285-0.305 s0.002)×ATR_S(2.05-2.09 s0.005) =5×5×11×9=2475
  A02 atrs_below_peak     : LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.285-0.305 s0.005)×ATR_S(2.00-2.065 s0.005)=3×3×5×14=630
  A03 atrl_boundary       : LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.270-0.295 s0.003)×ATR_S(2.05-2.10 s0.005) =3×3×9×11=891
  A04 wide_ll_ls_bridge   : LL(1-15 s1)×LS(1-15 s1)×ATR_L(0.29-0.30 s0.005)×ATR_S(2.05-2.10 s0.025) =15×15×3×3=2025
  A05 high_atrs_confirm   : LL(5-9 s1)×LS(3-7 s1)×ATR_L(0.28-0.31 s0.01)×ATR_S(2.00-2.50 s0.025)    =5×5×4×21=2100
  A06 atrs_step002        : LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.290-0.300 s0.002)×ATR_S(2.04-2.10 s0.002) =3×3×6×31=1674
  A07 global_confirm_v2   : LL(3-15 s1)×LS(3-15 s1)×ATR_L(0.10-0.50 s0.10)×ATR_S(1.5-2.5 s0.25)     =13×13×5×5=4225
  A08 atrl_low_regime     : LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.05-0.28 s0.01)×ATR_S(2.0-2.5 s0.25)       =3×3×24×3=648
  A09 adaptive_zoom1      : LL±3 LS±3 s1, ATR_L±0.010 s0.005, ATR_S±0.025 s0.005 → ≤7×7×5×11=2695
  A10 adaptive_zoom2      : LL±2 LS±2 s1, ATR_L±0.005 s0.002, ATR_S±0.010 s0.005 → ≤5×5×6×5=750
  A11 adaptive_zoom3      : LL±2 LS±2 s1, ATR_L±0.004 s0.001, ATR_S±0.008 s0.002 → ≤5×5×9×9=2025
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260101_SFJ_HUNTER_AI.wsp"
SYMBOL     = "CME.GC HOT"
SIGNAL     = "SFJ_HUNTER2_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_hunter2_daily3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0   # USD

LL_LO, LL_HI = 1.0, 1000.0
LS_LO, LS_HI = 1.0, 1000.0
AL_LO, AL_HI = 0.01,  30.0
AS_LO, AS_HI = 0.1,   30.0

# R2 champion seed
SEED_LL   = 7.0
SEED_LS   = 5.0
SEED_ATRL = 0.295
SEED_ATRS = 2.07
SEED_NP   = 338_710.0

PREFIX = "GCH2D3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_hunter2_daily3_{int(time.time())}.log"
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
         ll:   Tuple[float, float, float],
         ls:   Tuple[float, float, float],
         atrl: Tuple[float, float, float],
         atrs: Tuple[float, float, float]) -> StrategyConfig:
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


def champion(df, fb_ll, fb_ls, fb_atrl, fb_atrs):
    """Target-chasing mode: highest NP row drives zoom seed."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LEN_L"]), float(best["LEN_S"]),
                 float(best["ATR_multiplier_L"]), float(best["ATR_multiplier_S"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["LEN_L"]), float(best["LEN_S"]),
                float(best["ATR_multiplier_L"]), float(best["ATR_multiplier_S"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LEN_L"]), float(best["LEN_S"]),
                 float(best["ATR_multiplier_L"]), float(best["ATR_multiplier_S"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["LEN_L"]), float(best["LEN_S"]),
                float(best["ATR_multiplier_L"]), float(best["ATR_multiplier_S"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_ll, fb_ls, fb_atrl, fb_atrs,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, ll, ls, atrl, atrs, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "LEN_L": ll, "LEN_S": ls,
        "ATR_multiplier_L": atrl, "ATR_multiplier_S": atrs,
        "objective": obj, "net_profit": np_, "max_drawdown": mdd,
        "total_trades": trades, "target_met": met,
        "combos": combos,
        "rows": len(df) if df is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }


def save_json(best_entry, attempt_log, target_met):
    payload = {
        "round": 3,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY when C > AVERAGE(C,LEN_L) → next bar STOP at Close+ATR_L×ATR(20)",
            "exits":       "Reversal only — no STP or LMT; max 1 entry per day",
            "r1_champion": "LEN_L=7 LEN_S=5 ATR_L=0.3 ATR_S=2.1 NP=335050 MDD=-38660 trades=51",
            "r2_champion": "LEN_L=7 LEN_S=5 ATR_L=0.295 ATR_S=2.07 NP=338710 MDD=-38660 trades=51 (+1.09%)",
            "r3_focus":    "Ultra-fine ATR step=0.001-0.005; triple convergence proof; confirm ceiling ~$338-340K",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_gc_hunter2_daily3.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_ll   = SEED_LL
    best_ls   = SEED_LS
    best_atrl = SEED_ATRL
    best_atrs = SEED_ATRS
    best_np   = SEED_NP
    best_obj  = 0.0
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER2_NQ on CME.GC HOT Daily — Round 3")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  Target: %.0f USD  R2 seed NP: %.0f", TARGET_NP, SEED_NP)
    log.info("  R2 champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g",
             SEED_LL, SEED_LS, SEED_ATRL, SEED_ATRS)
    log.info("  Goal: Triple-convergence A09=A10=A11 → confirm ceiling ~$338-340K")
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_ll, best_ls, best_atrl, best_atrs
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_ll, best_ls, best_atrl, best_atrs,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-24s]  no valid data", attempt_num, name)
            return

        ll, ls, atrl, atrs, obj, np_, mdd, tr, met = champion(
            df, best_ll, best_ls, best_atrl, best_atrs)

        if np_ > best_np:
            best_ll, best_ls     = ll, ls
            best_atrl, best_atrs = atrl, atrs
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, ll, ls, atrl, atrs,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-24s]  LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, ll, ls, atrl, atrs, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f  (R2=%.0f  gain=%.1f%%)",
                 best_np, max(0, TARGET_NP - best_np),
                 SEED_NP, (best_np - SEED_NP) / SEED_NP * 100)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  fine_atrs_precision — very fine ATR step=0.002/0.005 around champion
    #      LL(5-9 s1)×LS(3-7 s1)×ATR_L(0.285-0.305 s0.002)×ATR_S(2.05-2.09 s0.005)
    #      n_ll=5, n_ls=5, n_atrl=11, n_atrs=9 → 5×5×11×9=2475
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_atrs_precision"
    _c = _cfg(_n, (5, 9, 1), (3, 7, 1), (0.285, 0.305, 0.002), (2.05, 2.09, 0.005))
    log.info("A01  LL(5-9 s1)×LS(3-7 s1)×ATR_L(0.285-0.305 s0.002)×ATR_S(2.05-2.09 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A02  atrs_below_peak — ATR_S 2.00-2.065 at step=0.005 (below the sharp peak)
    #      LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.285-0.305 s0.005)×ATR_S(2.00-2.065 s0.005)
    #      n_atrl=5, n_atrs=14 → 3×3×5×14=630
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_atrs_below_peak"
    _c = _cfg(_n, (6, 8, 1), (4, 6, 1), (0.285, 0.305, 0.005), (2.00, 2.065, 0.005))
    log.info("A02  LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.285-0.305 s0.005)×ATR_S(2.00-2.065 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A03  atrl_boundary — examine ATR_L below 0.290 regime switch precisely
    #      LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.270-0.295 s0.003)×ATR_S(2.05-2.10 s0.005)
    #      n_atrl≈9, n_atrs=11 → 3×3×9×11=891
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_atrl_boundary"
    _c = _cfg(_n, (6, 8, 1), (4, 6, 1), (0.270, 0.295, 0.003), (2.05, 2.10, 0.005))
    log.info("A03  LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.270-0.295 s0.003)×ATR_S(2.05-2.10 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A04  wide_ll_ls_bridge — LL×LS bridge sweep at champion ATR values
    #      LL(1-15 s1)×LS(1-15 s1)×ATR_L(0.29-0.30 s0.005)×ATR_S(2.05-2.10 s0.025)
    #      15×15×3×3=2025
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_wide_ll_ls_bridge"
    _c = _cfg(_n, (1, 15, 1), (1, 15, 1), (0.29, 0.30, 0.005), (2.05, 2.10, 0.025))
    log.info("A04  LL(1-15 s1)×LS(1-15 s1)×ATR_L(0.29-0.30 s0.005)×ATR_S(2.05-2.10 s0.025)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A05  high_atrs_confirm — confirm no secondary peak in ATR_S 2.0-2.5
    #      LL(5-9 s1)×LS(3-7 s1)×ATR_L(0.28-0.31 s0.01)×ATR_S(2.00-2.50 s0.025)
    #      5×5×4×21=2100
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_high_atrs_confirm"
    _c = _cfg(_n, (5, 9, 1), (3, 7, 1), (0.28, 0.31, 0.01), (2.00, 2.50, 0.025))
    log.info("A05  LL(5-9 s1)×LS(3-7 s1)×ATR_L(0.28-0.31 s0.01)×ATR_S(2.00-2.50 s0.025)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A06  atrs_step002 — ATR_S at step=0.002 to resolve 2.07 exactly
    #      LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.290-0.300 s0.002)×ATR_S(2.04-2.10 s0.002)
    #      3×3×6×31=1674
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_atrs_step002"
    _c = _cfg(_n, (6, 8, 1), (4, 6, 1), (0.290, 0.300, 0.002), (2.04, 2.10, 0.002))
    log.info("A06  LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.290-0.300 s0.002)×ATR_S(2.04-2.10 s0.002)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A07  global_confirm_v2 — confirm no undiscovered LL×LS regime
    #      LL(3-15 s1)×LS(3-15 s1)×ATR_L(0.10-0.50 s0.10)×ATR_S(1.5-2.5 s0.25)
    #      13×13×5×5=4225
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_global_confirm_v2"
    _c = _cfg(_n, (3, 15, 1), (3, 15, 1), (0.10, 0.50, 0.10), (1.5, 2.5, 0.25))
    log.info("A07  LL(3-15 s1)×LS(3-15 s1)×ATR_L(0.10-0.50 s0.10)×ATR_S(1.5-2.5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A08  atrl_low_regime — check ATR_L=0.05-0.28 (very tight entry stop)
    #      LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.05-0.28 s0.01)×ATR_S(2.0-2.5 s0.25)
    #      3×3×24×3=648
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_atrl_low_regime"
    _c = _cfg(_n, (6, 8, 1), (4, 6, 1), (0.05, 0.28, 0.01), (2.0, 2.5, 0.25))
    log.info("A08  LL(6-8 s1)×LS(4-6 s1)×ATR_L(0.05-0.28 s0.01)×ATR_S(2.0-2.5 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — ATR_L±0.010 s0.005, ATR_S±0.025 s0.005
    #      LL±3 LS±3 s1 → ≤ 7×7×5×11=2695
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   3.0,  1.0,   LL_LO, LL_HI)
        _ls   = zoom(best_ls,   3.0,  1.0,   LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.010, 0.005, AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.025, 0.005, AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A09  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — ATR_L±0.005 s0.002, ATR_S±0.010 s0.005
    #      LL±2 LS±2 s1 → ≤ 5×5×6×5=750
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   2.0,  1.0,  LL_LO, LL_HI)
        _ls   = zoom(best_ls,   2.0,  1.0,  LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.005, 0.002, AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.010, 0.005, AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A10  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — finest: ATR_L±0.004 s0.001, ATR_S±0.008 s0.002
    #      LL±2 LS±2 s1 → ≤ 5×5×9×9=2025
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   2.0,  1.0,   LL_LO, LL_HI)
        _ls   = zoom(best_ls,   2.0,  1.0,   LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.004, 0.001, AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.008, 0.002, AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A11  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER2_NQ CME.GC HOT Daily Round-3 COMPLETE")
    log.info("  Champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g",
             best_ll, best_ls, best_atrl, best_atrs)
    log.info("  Best NP: %.0f USD  (R2 seed %.0f  gain %.1f%%  target %.0f)",
             best_np, SEED_NP, (best_np - SEED_NP) / SEED_NP * 100, TARGET_NP)
    log.info("  R1 NP=335,050  R2 NP=338,710 (+1.09%)  R3 NP=%.0f (%.1f%%)",
             best_np, (best_np - SEED_NP) / SEED_NP * 100)
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{max(0, TARGET_NP - best_np):.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LEN_L": best_ll, "LEN_S": best_ls,
            "ATR_multiplier_L": best_atrl, "ATR_multiplier_S": best_atrs,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"R2 seed: NP=${SEED_NP:,.0f}  (LEN_L=7 LEN_S=5 ATR_L=0.295 ATR_S=2.07)")
    print(f"R3 best: NP=${best_np:,.0f}  gain={(best_np-SEED_NP)/SEED_NP*100:+.2f}%")
    print(f"Target NP>700K USD: {'MET ✅' if target_met else 'NOT MET'}")
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
        description="SFJ_HUNTER2_NQ CME.GC HOT Daily NP>700K USD Round-3 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1–11)")
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
