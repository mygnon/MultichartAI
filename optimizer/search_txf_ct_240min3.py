"""
search_txf_ct_240min3.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT 240-Minute, Round 3

R1 champion: LL=12, SL=0.45, LS=52, SS=0.1,   NP=6,243,800 (tight-SS regime)
R2 champion: LL=12, SL=0.7,  LS=52, SS=0.125, NP=6,742,400, MDD=-1,003,800, Obj=45,287,863, trades=412
R2 Obj-max:  LL=10, SL=0.6,  LS=52, SS=0.13,  NP=6,737,200, MDD=-930,600,   Obj=48,774,837

R1→R2 gain: +8.0%. Target 8M not met (gap −15.7%). Continuing to R3.

Key R2 discoveries:
  - SL shifted significantly up: 0.45 → 0.70
  - SS slightly up: 0.10 → 0.125
  - Two candidates: NP-max (LL=12 SL=0.7 MDD=-1004K) vs Obj-max (LL=10 SL=0.6 MDD=-931K)
  - LS=52 stable throughout
  - Hourly SS regime (SS=1.4-2.0) useless at 240-min (A07: NP=2,379,000)

R3 focus: fine-tune both LL=10 and LL=12 regimes; high-res SL/SS; extend SL up to 1.2; fine LS step=1.
  1. ll10_sl_fine   — LL=8-14, SL=0.50-0.75 step=0.025, fine SS around LL=10 Obj-max zone
  2. ll12_sl_fine   — LL=10-16, SL=0.55-0.80 step=0.025, fine SS around LL=12 NP-max zone
  3. sl_high        — SL=0.70-1.10 to test if SL > 0.70 helps NP
  4. ss_fine        — SS=0.08-0.18 step=0.01 to precisely map SS peak
  5. ls_fine        — LS=46-60 step=1 to find exact LS optimum
  6. ll_step1       — LL=7-15 step=1 to confirm optimal LL integer
  7. high_sl_ext    — SL=0.75-1.20 with wider LL scan
  8. tight_ss_ultra — Ultra-tight SS=0.05-0.10 step=0.005
  9. global_r3      — Broad sweep with tighter SS coverage (SS step=0.05)
  10-12. Adaptive zooms

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 ll10_sl_fine  : LL(8-14 s2)×SL(0.50-0.75 s0.025)×LS(50-54 s2)×SS(0.10-0.16 s0.01)  = 4×11×3×7 = 924
  A02 ll12_sl_fine  : LL(10-16 s2)×SL(0.55-0.80 s0.025)×LS(50-54 s2)×SS(0.10-0.16 s0.01) = 4×11×3×7 = 924
  A03 sl_high       : LL(10-14 s2)×SL(0.70-1.10 s0.025)×LS(48-56 s2)×SS(0.10-0.15 s0.01) = 3×17×5×6 = 1530
  A04 ss_fine       : LL(10-14 s2)×SL(0.55-0.75 s0.025)×LS(50-54 s2)×SS(0.08-0.18 s0.01) = 3×9×3×11 = 891
  A05 ls_fine       : LL(10-14 s2)×SL(0.60-0.75 s0.025)×LS(46-60 s1)×SS(0.10-0.14 s0.01) = 3×7×15×5 = 1575
  A06 ll_step1      : LL(7-15 s1)×SL(0.60-0.75 s0.025)×LS(50-54 s2)×SS(0.10-0.14 s0.01)  = 9×7×3×5 = 945
  A07 high_sl_ext   : LL(10-16 s2)×SL(0.75-1.20 s0.025)×LS(48-56 s2)×SS(0.10-0.15 s0.01) = 4×19×5×6 = 2280
  A08 tight_ss_ultra: LL(8-16 s2)×SL(0.55-0.75 s0.025)×LS(48-56 s2)×SS(0.05-0.10 s0.005) = 5×9×5×11 = 2475
  A09 global_r3     : LL(4-64 s10)×SL(0.3-1.3 s0.25)×LS(10-80 s10)×SS(0.05-0.30 s0.05)   = 7×5×8×6 = 1680
  A10 adaptive_zoom : (dynamic from best NP)
  A11 adaptive_zoom2: (dynamic from best NP)
  A12 adaptive_zoom3: (dynamic from best NP)
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
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "SFJ_15Dworkshop_lesson5_countertrend_LS"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_240min3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0   # TWD

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.05, 20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.05, 20.0

# R2 NP-max champion as seed
SEED_LL, SEED_SL = 12.0, 0.7
SEED_LS, SEED_SS = 52.0, 0.125
SEED_NP          = 6_742_400.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_240min3_{int(time.time())}.log"
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
        name=f"TXFCT240_3_{name}",
        mc_signal_name=SIGNAL,
        timeframe="minute",
        bar_period=240,
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
    return OUTPUT_DIR / f"TXFCT240_3_{name}_raw.csv"


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
    log.info("=== Starting TXFCT240_3_%s (%d combos) ===", name, cfg.total_runs())
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
        "round": 3,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "240min",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_champion": "LL=12 SL=0.45 LS=52 SS=0.1   NP=6,243,800",
            "r2_champion": "LL=12 SL=0.7  LS=52 SS=0.125 NP=6,742,400 MDD=-1,003,800 Obj=45,287,863 trades=412",
            "r2_obj_max":  "LL=10 SL=0.6  LS=52 SS=0.13  NP=6,737,200 MDD=-930,600   Obj=48,774,837",
            "r3_focus": "Fine-tune LL=10/12 with SL step=0.025; SS step=0.01; LS step=1; extend SL to 1.2",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_240min3.json"
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
    log.info("  TXF 240-Minute countertrend_LS NP>8M TWD — Round 3")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R2 NP-max: LL=12 SL=0.7  LS=52 SS=0.125 NP=6,742,400 MDD=-1,003,800")
    log.info("  R2 Obj-max: LL=10 SL=0.6 LS=52 SS=0.13  NP=6,737,200 MDD=-930,600 Obj=48.8M")
    log.info("  R2 seed: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  Target: %.0f TWD  (gap from seed: −15.7%%)", TARGET_NP)
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
                 "★TARGET★" if met else ("%.0f/8M" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  ll10_sl_fine — fine scan around LL=10 Obj-max zone (SL step=0.025)
    #      LL(8-14 s2)×SL(0.50-0.75 s0.025)×LS(50-54 s2)×SS(0.10-0.16 s0.01) = 4×11×3×7 = 924
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ll10_sl_fine"
    _c = _cfg(_n, (8, 14, 2), (0.50, 0.75, 0.025), (50, 54, 2), (0.10, 0.16, 0.01))
    log.info("A01  LL(8-14 s2)×SL(0.50-0.75 s0.025)×LS(50-54 s2)×SS(0.10-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  ll12_sl_fine — fine scan around LL=12 NP-max zone (SL step=0.025)
    #      LL(10-16 s2)×SL(0.55-0.80 s0.025)×LS(50-54 s2)×SS(0.10-0.16 s0.01) = 4×11×3×7 = 924
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ll12_sl_fine"
    _c = _cfg(_n, (10, 16, 2), (0.55, 0.80, 0.025), (50, 54, 2), (0.10, 0.16, 0.01))
    log.info("A02  LL(10-16 s2)×SL(0.55-0.80 s0.025)×LS(50-54 s2)×SS(0.10-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  sl_high — probe SL=0.70-1.10 to see if NP keeps rising
    #      LL(10-14 s2)×SL(0.70-1.10 s0.025)×LS(48-56 s2)×SS(0.10-0.15 s0.01) = 3×17×5×6 = 1530
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_sl_high"
    _c = _cfg(_n, (10, 14, 2), (0.70, 1.10, 0.025), (48, 56, 2), (0.10, 0.15, 0.01))
    log.info("A03  LL(10-14 s2)×SL(0.70-1.10 s0.025)×LS(48-56 s2)×SS(0.10-0.15 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ss_fine — fine SS scan step=0.01 across SS=0.08-0.18
    #      LL(10-14 s2)×SL(0.55-0.75 s0.025)×LS(50-54 s2)×SS(0.08-0.18 s0.01) = 3×9×3×11 = 891
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ss_fine"
    _c = _cfg(_n, (10, 14, 2), (0.55, 0.75, 0.025), (50, 54, 2), (0.08, 0.18, 0.01))
    log.info("A04  LL(10-14 s2)×SL(0.55-0.75 s0.025)×LS(50-54 s2)×SS(0.08-0.18 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ls_fine — fine LS scan step=1 (LS=46-60)
    #      LL(10-14 s2)×SL(0.60-0.75 s0.025)×LS(46-60 s1)×SS(0.10-0.14 s0.01) = 3×7×15×5 = 1575
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ls_fine"
    _c = _cfg(_n, (10, 14, 2), (0.60, 0.75, 0.025), (46, 60, 1), (0.10, 0.14, 0.01))
    log.info("A05  LL(10-14 s2)×SL(0.60-0.75 s0.025)×LS(46-60 s1)×SS(0.10-0.14 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ll_step1 — LL scan at integer resolution (LL=7-15 step=1)
    #      LL(7-15 s1)×SL(0.60-0.75 s0.025)×LS(50-54 s2)×SS(0.10-0.14 s0.01) = 9×7×3×5 = 945
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ll_step1"
    _c = _cfg(_n, (7, 15, 1), (0.60, 0.75, 0.025), (50, 54, 2), (0.10, 0.14, 0.01))
    log.info("A06  LL(7-15 s1)×SL(0.60-0.75 s0.025)×LS(50-54 s2)×SS(0.10-0.14 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  high_sl_ext — extend SL even higher (SL=0.75-1.20)
    #      LL(10-16 s2)×SL(0.75-1.20 s0.025)×LS(48-56 s2)×SS(0.10-0.15 s0.01) = 4×19×5×6 = 2280
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_high_sl_ext"
    _c = _cfg(_n, (10, 16, 2), (0.75, 1.20, 0.025), (48, 56, 2), (0.10, 0.15, 0.01))
    log.info("A07  LL(10-16 s2)×SL(0.75-1.20 s0.025)×LS(48-56 s2)×SS(0.10-0.15 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ss, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  tight_ss_ultra — ultra-tight SS=0.05-0.10 step=0.005 (finer than R2)
    #      LL(8-16 s2)×SL(0.55-0.75 s0.025)×LS(48-56 s2)×SS(0.05-0.10 s0.005) = 5×9×5×11 = 2475
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_tight_ss_ultra"
    _c = _cfg(_n, (8, 16, 2), (0.55, 0.75, 0.025), (48, 56, 2), (0.05, 0.10, 0.005))
    log.info("A08  LL(8-16 s2)×SL(0.55-0.75 s0.025)×LS(48-56 s2)×SS(0.05-0.10 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  global_r3 — broad sweep with tighter SS coverage
    #      LL(4-64 s10)×SL(0.3-1.3 s0.25)×LS(10-80 s10)×SS(0.05-0.30 s0.05) = 7×5×8×6 = 1680
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_global_r3"
    _c = _cfg(_n, (4, 64, 10), (0.3, 1.3, 0.25), (10, 80, 10), (0.05, 0.30, 0.05))
    log.info("A09  LL(4-64 s10)×SL(0.3-1.3 s0.25)×LS(10-80 s10)×SS(0.05-0.30 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — wide zoom around best NP
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (8,  0.20, 10, 0.08),
            (6,  0.15,  8, 0.06),
            (4,  0.10,  6, 0.04),
            (3,  0.08,  4, 0.03),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.025, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 2.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.01,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom2 — medium zoom
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom2"
    log.info("A11  adaptive_zoom2 — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (4,  0.10, 6, 0.04),
            (3,  0.08, 4, 0.03),
            (2,  0.06, 3, 0.025),
            (2,  0.05, 2, 0.01),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.025, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 2.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.01,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  adaptive_zoom3 — tight final zoom
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_adaptive_zoom3"
    log.info("A12  adaptive_zoom3 — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (3,  0.08, 4, 0.03),
            (2,  0.06, 3, 0.025),
            (2,  0.05, 2, 0.01),
            (2,  0.025, 2, 0.01),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.025, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 2.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.01,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A12  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  TXF 240-Minute countertrend_LS Round-3 COMPLETE")
    log.info("  Best NP: %.0f TWD  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R2 seed: 6,742,400  R3 best: %.0f  gain: %+.1f%%",
             best_np, (best_np / 6_742_400 - 1) * 100)
    log.info("  Target %.0f TWD: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{max(0,TARGET_NP - best_np):.0f})")
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
    print(f"Target NP>8M TWD: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="TXF 240-Minute countertrend_LS NP>8M TWD Round-3 search")
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
