"""
search_nq_hunter2_hourly3.py — SFJ_HUNTER2_NQ on CME.NQ HOT Hourly, Round 3

R1 champion: LEN_L=8, LEN_S=89, ATR_L=0.3,  ATR_S=2.5  NP=630,845 USD (gain R0→R1 baseline)
R2 champion: LEN_L=8, LEN_S=89, ATR_L=0.25, ATR_S=2.5  NP=634,865 USD (R1→R2 gain +0.6%)
Gap to target: 900,000 - 634,865 = 265,135 (−29.5%)

R2 UI-failed (untested territory):
  A02: LEN_L=1-30 wide scan at LEN_S=85-95 step=2
  A04: LEN_S=40-88 (below champion)
  A06: LEN_S=100-250 (above champion)
  A08: LEN_L=1-10 × LEN_S=60-130 wide sweep
  A11: finest adaptive zoom (1225 combos)

R3 strategy: Fill all UI-failed gaps; probe ultra-tight ATR_L and high LEN_S (TXF analog: LEN_S=290); confirm ceiling.

4 active params, target NP > 900,000 USD, ≤5,000 combos/attempt.

Attempt schedule (11 attempts):
  A01 lens_very_short  : LEN_L(1-7 s1)×LEN_S(83-97 s1)×ATR_L(0.1-0.5 s0.1)×ATR_S(2.0-3.0 s0.25) = 7×15×5×5=2625
  A02 lower_lens       : LEN_L(4-16 s2)×LEN_S(40-88 s4)×ATR_L(0.1-0.5 s0.1)×ATR_S(2.0-3.0 s0.25) = 7×13×5×5=2275
  A03 high_lens        : LEN_L(4-14 s2)×LEN_S(100-290 s10)×ATR_L(0.1-0.5 s0.1)×ATR_S(1.5-3.0 s0.25)= 6×20×5×7=4200
  A04 deep_lenl        : LEN_L(1-10 s1)×LEN_S(55-130 s15)×ATR_L(0.1-0.5 s0.1)×ATR_S(1.5-3.5 s0.5) = 10×6×5×5=1500
  A05 fine_atrs        : LEN_L(6-12 s2)×LEN_S(85-95 s2)×ATR_L(0.15-0.4 s0.05)×ATR_S(1.5-3.5 s0.1) = 4×6×6×21=3024
  A06 ultra_tight_atrl : LEN_L(6-12 s2)×LEN_S(85-95 s2)×ATR_L(0.01-0.2 s0.01)×ATR_S(2.0-3.0 s0.25) = 4×6×20×5=2400
  A07 high_atrl        : LEN_L(5-20 s5)×LEN_S(80-120 s10)×ATR_L(0.5-3.0 s0.5)×ATR_S(1.0-4.0 s1)   = 4×5×6×4=480
  A08 global_landscape : LEN_L(2-50 s4)×LEN_S(20-410 s30)×ATR_L(0.1-0.9 s0.2)×ATR_S(1.0-4.0 s1)   = 13×14×5×4=3640
  A09 adaptive_zoom1   : fixed safe radii — LL±5 LS±5 s1, ATR±0.2 s0.1 → ≤11×11×5×5=3025
  A10 adaptive_zoom2   : finer — LL±4 LS±4 s1, ATR±0.15 s0.05 → ≤9×9×7×7=3969
  A11 adaptive_zoom3   : finest — LL±3 LS±3 s1, ATR±0.1 s0.05 → ≤7×7×5×5=1225
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
SYMBOL     = "CME.NQ HOT"
SIGNAL     = "SFJ_HUNTER2_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_hunter2_hourly3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 900_000.0   # USD

LL_LO, LL_HI = 1.0, 1000.0
LS_LO, LS_HI = 1.0, 1000.0
AL_LO, AL_HI = 0.01,  30.0
AS_LO, AS_HI = 0.1,   30.0

# Seeded from R2 champion
SEED_LL   = 8.0
SEED_LS   = 89.0
SEED_ATRL = 0.25
SEED_ATRS = 2.5
SEED_NP   = 634_865.0

PREFIX = "NQH2H3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_hunter2_hourly3_{int(time.time())}.log"
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
        timeframe="hourly",
        bar_period=60,
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
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY when C > AVERAGE(C,LEN_L) → next bar STOP at Close+ATR_L×ATR(20)",
            "exits":       "Reversal only — no STP or LMT; max 1 entry per day",
            "r1_champion": "LEN_L=8 LEN_S=89 ATR_L=0.3 ATR_S=2.5 NP=630845 MDD=-83660 trades=883",
            "r2_champion": "LEN_L=8 LEN_S=89 ATR_L=0.25 ATR_S=2.5 NP=634865 MDD=-82930 trades=883 (R1→R2 +0.6%)",
            "r3_focus":    "Fill R2 UI-fails: LEN_L=1-7, LEN_S=40-88, LEN_S=100-290; ultra-tight ATR_L; global landscape",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_hunter2_hourly3.json"
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
    log.info("  SFJ_HUNTER2_NQ on CME.NQ HOT Hourly — Round 3")
    log.info("  Signal: %s  Symbol: %s", SIGNAL, SYMBOL)
    log.info("  R2 champion: LEN_L=8 LEN_S=89 ATR_L=0.25 ATR_S=2.5  NP=634865 USD")
    log.info("  Target: %.0f USD", TARGET_NP)
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
                 "★TARGET★" if met else ("%.0f/900K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f  R2→R3 gain=%.1f%%",
                 best_np, max(0, TARGET_NP - best_np),
                 (best_np - SEED_NP) / SEED_NP * 100)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  lens_very_short — LEN_L=1-7 (R2 A02 UI-failed this region)
    #      LEN_L(1-7 s1)×LEN_S(83-97 s1)×ATR_L(0.1-0.5 s0.1)×ATR_S(2.0-3.0 s0.25) = 7×15×5×5=2625
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_lens_very_short"
    _c = _cfg(_n, (1, 7, 1), (83, 97, 1), (0.1, 0.5, 0.1), (2.0, 3.0, 0.25))
    log.info("A01  LEN_L(1-7 s1)×LEN_S(83-97 s1)×ATR_L(0.1-0.5 s0.1)×ATR_S(2.0-3.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A02  lower_lens — LEN_S=40-88 (R2 A04 UI-failed entirely)
    #      LEN_L(4-16 s2)×LEN_S(40-88 s4)×ATR_L(0.1-0.5 s0.1)×ATR_S(2.0-3.0 s0.25) = 7×13×5×5=2275
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_lower_lens"
    _c = _cfg(_n, (4, 16, 2), (40, 88, 4), (0.1, 0.5, 0.1), (2.0, 3.0, 0.25))
    log.info("A02  LEN_L(4-16 s2)×LEN_S(40-88 s4)×ATR_L(0.1-0.5 s0.1)×ATR_S(2.0-3.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A03  high_lens — LEN_S=100-290 (R2 A06 UI-failed; TXF champion had LEN_S=290)
    #      LEN_L(4-14 s2)×LEN_S(100-290 s10)×ATR_L(0.1-0.5 s0.1)×ATR_S(1.5-3.0 s0.25) = 6×20×5×7=4200
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_high_lens"
    _c = _cfg(_n, (4, 14, 2), (100, 290, 10), (0.1, 0.5, 0.1), (1.5, 3.0, 0.25))
    log.info("A03  LEN_L(4-14 s2)×LEN_S(100-290 s10)×ATR_L(0.1-0.5 s0.1)×ATR_S(1.5-3.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A04  deep_lenl — LEN_L=1-10 wide LEN_S (R2 A08 UI-failed)
    #      LEN_L(1-10 s1)×LEN_S(55-130 s15)×ATR_L(0.1-0.5 s0.1)×ATR_S(1.5-3.5 s0.5) = 10×6×5×5=1500
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_deep_lenl"
    _c = _cfg(_n, (1, 10, 1), (55, 130, 15), (0.1, 0.5, 0.1), (1.5, 3.5, 0.5))
    log.info("A04  LEN_L(1-10 s1)×LEN_S(55-130 s15)×ATR_L(0.1-0.5 s0.1)×ATR_S(1.5-3.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A05  fine_atrs — fine ATR_S step=0.1 (coarsely tested before)
    #      LEN_L(6-12 s2)×LEN_S(85-95 s2)×ATR_L(0.15-0.4 s0.05)×ATR_S(1.5-3.5 s0.1) = 4×6×6×21=3024
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_fine_atrs"
    _c = _cfg(_n, (6, 12, 2), (85, 95, 2), (0.15, 0.4, 0.05), (1.5, 3.5, 0.1))
    log.info("A05  LEN_L(6-12 s2)×LEN_S(85-95 s2)×ATR_L(0.15-0.4 s0.05)×ATR_S(1.5-3.5 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ultra_tight_atrl — ATR_L=0.01-0.2 (NQ CT hourly found SL=0.2 breakthrough)
    #      LEN_L(6-12 s2)×LEN_S(85-95 s2)×ATR_L(0.01-0.2 s0.01)×ATR_S(2.0-3.0 s0.25) = 4×6×20×5=2400
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ultra_tight_atrl"
    _c = _cfg(_n, (6, 12, 2), (85, 95, 2), (0.01, 0.2, 0.01), (2.0, 3.0, 0.25))
    log.info("A06  LEN_L(6-12 s2)×LEN_S(85-95 s2)×ATR_L(0.01-0.2 s0.01)×ATR_S(2.0-3.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A07  high_atrl — ATR_L=0.5-3.0 (moderate/high ATR_L regime, TXF analog ATR_L=0.97)
    #      LEN_L(5-20 s5)×LEN_S(80-120 s10)×ATR_L(0.5-3.0 s0.5)×ATR_S(1.0-4.0 s1) = 4×5×6×4=480
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_high_atrl"
    _c = _cfg(_n, (5, 20, 5), (80, 120, 10), (0.5, 3.0, 0.5), (1.0, 4.0, 1.0))
    log.info("A07  LEN_L(5-20 s5)×LEN_S(80-120 s10)×ATR_L(0.5-3.0 s0.5)×ATR_S(1.0-4.0 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A08  global_landscape — broad sweep to find undiscovered regimes
    #      LEN_L(2-50 s4)×LEN_S(20-410 s30)×ATR_L(0.1-0.9 s0.2)×ATR_S(1.0-4.0 s1) = 13×14×5×4=3640
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_global_landscape"
    _c = _cfg(_n, (2, 50, 4), (20, 410, 30), (0.1, 0.9, 0.2), (1.0, 4.0, 1.0))
    log.info("A08  LEN_L(2-50 s4)×LEN_S(20-410 s30)×ATR_L(0.1-0.9 s0.2)×ATR_S(1.0-4.0 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — fixed safe radii: LL±5 LS±5 step=1, ATR±0.2 step=0.1
    #      Guarantees ≤ 11×11×5×5 = 3025 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   5.0, 1.0,  LL_LO, LL_HI)
        _ls   = zoom(best_ls,   5.0, 1.0,  LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.2, 0.1,  AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.2, 0.1,  AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A09  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — tighter: LL±4 LS±4 step=1, ATR±0.15 step=0.05
    #      ≤ 9×9×7×7 = 3969 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   4.0, 1.0,  LL_LO, LL_HI)
        _ls   = zoom(best_ls,   4.0, 1.0,  LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.15, 0.05, AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.15, 0.05, AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A10  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — finest: LL±3 LS±3 step=1, ATR±0.1 step=0.05
    #      ≤ 7×7×5×5 = 1225 combos
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g  NP=%.0f",
             best_ll, best_ls, best_atrl, best_atrs, best_np)
    if start_attempt <= A:
        _ll   = zoom(best_ll,   3.0, 1.0,  LL_LO, LL_HI)
        _ls   = zoom(best_ls,   3.0, 1.0,  LS_LO, LS_HI)
        _atrl = zoom(best_atrl, 0.1,  0.05, AL_LO, AL_HI)
        _atrs = zoom(best_atrs, 0.1,  0.05, AS_LO, AS_HI)
        _c    = _cfg(_n, _ll, _ls, _atrl, _atrs)
        log.info("A11  LL%s LS%s ATR_L%s ATR_S%s  %d combos",
                 _ll, _ls, _atrl, _atrs, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g)",
             best_np, best_ll, best_ls, best_atrl, best_atrs)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    gain_pct = (best_np - SEED_NP) / SEED_NP * 100 if SEED_NP > 0 else 0.0
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  SFJ_HUNTER2_NQ CME.NQ HOT Hourly Round-3 COMPLETE")
    log.info("  Champion: LEN_L=%.4g LEN_S=%.4g ATR_L=%.4g ATR_S=%.4g",
             best_ll, best_ls, best_atrl, best_atrs)
    log.info("  Best NP: %.0f USD  (target %.0f  gap %.0f)",
             best_np, TARGET_NP, max(0, TARGET_NP - best_np))
    log.info("  Target %.0f USD: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{max(0, TARGET_NP - best_np):.0f})")
    log.info("  R2→R3 gain: %.1f%%", gain_pct)
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
    print(f"Target NP>900K USD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
    print(f"R2→R3 gain: {gain_pct:.1f}%")
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
        description="SFJ_HUNTER2_NQ CME.NQ HOT Hourly NP>900K USD Round-3 search")
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
