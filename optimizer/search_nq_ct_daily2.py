"""
search_nq_ct_daily2.py — SFJ_15Dworkshop_lesson5_countertrend_LS on CME.NQ HOT Daily, Round 2

R1 findings (search_nq_ct_daily.py):
  Best NP : LL=7  SL=0.6 LS=55 SS=2.0  NP=387,590  MDD=-81,535  Obj=1,842,473  trades=28
  Best Obj: LL=15 SL=2.5 LS=25 SS=2.5  NP=332,130  MDD=-53,485  Obj=2,062,454  trades=10
  Gap: -44.6% from 700K target
  MDD=-81,535 is structural floor (recurring across many combos)
  Two regimes: (1) asymmetric high-freq (LL short+tight-SL, LS moderate, SS=2.0);
               (2) symmetric moderate (LL=LS~15-25, SL=SS~2.5)

R2 focus:
  A01 fine_np_winner   : fine grid around LL=7,SL=0.6,LS=55,SS=2.0
  A02 ultra_fine_center: extremely fine around winner (step=1 periods, step=0.1 stddev)
  A03 obj_winner_fine  : fine grid around LL=15,SL=2.5,LS=25,SS=2.5
  A04 extended_ls      : LS=55-120, short LL — can LS go higher?
  A05 ultra_tight_sl   : SL=0.05-0.35 analog of hourly breakthrough (SL=0.2)
  A06 very_short_ll    : LL=2-8 × broad SL × moderate LS
  A07 ss_fine_scan     : SS=1.6-2.4 fine × LL/SL center region
  A08 ll_sl_sweep      : LL=2-25 × SL=0.1-1.5 with LS=55, SS=2.0 fixed (narrow)
  A09 tight_ss         : SS=0.5-1.7 (tight short entry, analog of A12 R1 hint)
  A10 adaptive_zoom    : dynamic from R2 best NP (fine steps: period=1, stddev=0.1)
  A11 high_freq_daily  : LL=2-10 × SL=0.1-0.5 × LS=2-20 — can high-freq work on daily?
  A12 global_r2        : different coarse global sweep (fills gaps from R1 global)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01  3150   LL(4-12 s1)×SL(0.30-1.20 s0.15)×LS(46-64 s2)×SS(1.6-2.4 s0.2)
  A02  1260   LL(5-10 s1)×SL(0.40-0.90 s0.10)×LS(50-62 s2)×SS(1.8-2.2 s0.1)
  A03  2058   LL(11-21 s2)×SL(2.0-3.2 s0.2)×LS(20-32 s2)×SS(2.0-3.2 s0.2)
  A04  1848   LL(4-14 s1)×SL(0.30-1.20 s0.30)×LS(55-120 s5)×SS(1.5-2.5 s0.5)
  A05  1764   LL(2-15 s1)×SL(0.05-0.35 s0.05)×LS(40-65 s5)×SS(1.5-2.5 s0.5)
  A06  1764   LL(2-8 s1)×SL(0.20-1.40 s0.20)×LS(40-80 s5)×SS(1.5-3.0 s0.5)
  A07  3528   LL(5-12 s1)×SL(0.40-1.00 s0.10)×LS(50-62 s2)×SS(1.6-2.4 s0.1)
  A08  3240   LL(2-25 s1)×SL(0.10-1.50 s0.10)×LS(53-57 s2)×SS(1.9-2.1 s0.1)
  A09  2352   LL(3-14 s1)×SL(0.30-1.20 s0.30)×LS(45-75 s5)×SS(0.50-1.70 s0.20)
  A10  ≤5000  adaptive zoom (fine: step=1 period, step=0.1 stddev)
  A11  1800   LL(2-10 s1)×SL(0.10-0.50 s0.10)×LS(2-20 s2)×SS(0.5-2.0 s0.5)
  A12  2304   LL(3-80 s11)×SL(0.20-4.20 s0.80)×LS(3-80 s11)×SS(0.20-4.20 s0.80)
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_ct_daily2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.05, 20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.05, 20.0

# Seeds from R1 best NP winner
SEED_LL, SEED_SL = 7.0,  0.6
SEED_LS, SEED_SS = 55.0, 2.0
SEED_NP          = 387_590.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_ct_daily2_{int(time.time())}.log"
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
        name=f"NQCTD2_{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
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
    return OUTPUT_DIR / f"NQCTD2_{name}_raw.csv"


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
    log.info("=== Starting NQCTD2_%s (%d combos) ===", name, cfg.total_runs())
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
        "round": 2,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "daily",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_best":  "LL=7 SL=0.6 LS=55 SS=2.0 NP=387590 MDD=-81535 trades=28",
            "r1_obj":   "LL=15 SL=2.5 LS=25 SS=2.5 NP=332130 MDD=-53485 Obj=2062454 trades=10",
            "r2_focus": "fine NP winner, ultra-tight SL, Obj winner region, extended LS, tight SS",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_ct_daily2.json"
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
    log.info("  NQ Daily countertrend_LS NP>700K — Round 2")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  Timeframe: DAILY (1440 min bars)")
    log.info("  R1 best NP: LL=7 SL=0.6 LS=55 SS=2.0 NP=387590")
    log.info("  R1 best Obj: LL=15 SL=2.5 LS=25 SS=2.5 NP=332130 Obj=2062454")
    log.info("  Target: %.0f USD  Gap from R1: +%.0f", TARGET_NP, TARGET_NP - SEED_NP)
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
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  fine_np_winner — fine grid centered on R1 NP winner
    #      LL(4-12 s1)×SL(0.30-1.20 s0.15)×LS(46-64 s2)×SS(1.6-2.4 s0.2)  = 9×7×10×5 = 3150
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_np_winner"
    _c = _cfg(_n, (4, 12, 1), (0.30, 1.20, 0.15), (46, 64, 2), (1.6, 2.4, 0.2))
    log.info("A01  LL(4-12 s1)×SL(0.30-1.20 s0.15)×LS(46-64 s2)×SS(1.6-2.4 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  ultra_fine_center — extremely fine around winner
    #      LL(5-10 s1)×SL(0.40-0.90 s0.10)×LS(50-62 s2)×SS(1.8-2.2 s0.1)  = 6×6×7×5 = 1260
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ultra_fine_center"
    _c = _cfg(_n, (5, 10, 1), (0.40, 0.90, 0.10), (50, 62, 2), (1.8, 2.2, 0.1))
    log.info("A02  LL(5-10 s1)×SL(0.40-0.90 s0.10)×LS(50-62 s2)×SS(1.8-2.2 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  obj_winner_fine — fine grid around R1 Obj winner (LL=15, SL=2.5, LS=25, SS=2.5)
    #      LL(11-21 s2)×SL(2.0-3.2 s0.2)×LS(20-32 s2)×SS(2.0-3.2 s0.2)  = 6×7×7×7 = 2058
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_obj_winner_fine"
    _c = _cfg(_n, (11, 21, 2), (2.0, 3.2, 0.2), (20, 32, 2), (2.0, 3.2, 0.2))
    log.info("A03  LL(11-21 s2)×SL(2.0-3.2 s0.2)×LS(20-32 s2)×SS(2.0-3.2 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  extended_ls — does LS>55 improve NP? (LS=55-120, short LL)
    #      LL(4-14 s1)×SL(0.30-1.20 s0.30)×LS(55-120 s5)×SS(1.5-2.5 s0.5)  = 11×4×14×3 = 1848
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_extended_ls"
    _c = _cfg(_n, (4, 14, 1), (0.30, 1.20, 0.30), (55, 120, 5), (1.5, 2.5, 0.5))
    log.info("A04  LL(4-14 s1)×SL(0.30-1.20 s0.30)×LS(55-120 s5)×SS(1.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ultra_tight_sl — SL=0.05-0.35, analog of hourly SL=0.2 breakthrough
    #      LL(2-15 s1)×SL(0.05-0.35 s0.05)×LS(40-65 s5)×SS(1.5-2.5 s0.5)  = 14×7×6×3 = 1764
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ultra_tight_sl"
    _c = _cfg(_n, (2, 15, 1), (0.05, 0.35, 0.05), (40, 65, 5), (1.5, 2.5, 0.5))
    log.info("A05  LL(2-15 s1)×SL(0.05-0.35 s0.05)×LS(40-65 s5)×SS(1.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  very_short_ll — LL=2-8, broader SL and LS
    #      LL(2-8 s1)×SL(0.20-1.40 s0.20)×LS(40-80 s5)×SS(1.5-3.0 s0.5)  = 7×7×9×4 = 1764
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_very_short_ll"
    _c = _cfg(_n, (2, 8, 1), (0.20, 1.40, 0.20), (40, 80, 5), (1.5, 3.0, 0.5))
    log.info("A06  LL(2-8 s1)×SL(0.20-1.40 s0.20)×LS(40-80 s5)×SS(1.5-3.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  ss_fine_scan — fine SS scan (1.6-2.4) around winner regime
    #      LL(5-12 s1)×SL(0.40-1.00 s0.10)×LS(50-62 s2)×SS(1.6-2.4 s0.1)  = 8×7×7×9 = 3528
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ss_fine_scan"
    _c = _cfg(_n, (5, 12, 1), (0.40, 1.00, 0.10), (50, 62, 2), (1.6, 2.4, 0.1))
    log.info("A07  LL(5-12 s1)×SL(0.40-1.00 s0.10)×LS(50-62 s2)×SS(1.6-2.4 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  ll_sl_sweep — fix LS≈55, SS≈2.0; sweep LL and SL finely
    #      LL(2-25 s1)×SL(0.10-1.50 s0.10)×LS(53-57 s2)×SS(1.9-2.1 s0.1)  = 24×15×3×3 = 3240
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_ll_sl_sweep"
    _c = _cfg(_n, (2, 25, 1), (0.10, 1.50, 0.10), (53, 57, 2), (1.9, 2.1, 0.1))
    log.info("A08  LL(2-25 s1)×SL(0.10-1.50 s0.10)×LS(53-57 s2)×SS(1.9-2.1 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  tight_ss — explore SS=0.5-1.7 (tight short-entry bands, R1 A12 hint)
    #      LL(3-14 s1)×SL(0.30-1.20 s0.30)×LS(45-75 s5)×SS(0.50-1.70 s0.20)  = 12×4×7×7 = 2352
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_tight_ss"
    _c = _cfg(_n, (3, 14, 1), (0.30, 1.20, 0.30), (45, 75, 5), (0.50, 1.70, 0.20))
    log.info("A09  LL(3-14 s1)×SL(0.30-1.20 s0.30)×LS(45-75 s5)×SS(0.50-1.70 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — fine zoom (step=1 period, step=0.1 stddev) from R2 best NP
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (10, 1.0, 10, 1.0),
            (6,  0.6,  6, 0.6),
            (4,  0.4,  4, 0.4),
            (3,  0.3,  3, 0.3),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0, LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.1, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0, LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.1, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  high_freq_daily — very short LL+LS, tight stddev (high-freq mean-reversion)
    #      LL(2-10 s1)×SL(0.10-0.50 s0.10)×LS(2-20 s2)×SS(0.5-2.0 s0.5)  = 9×5×10×4 = 1800
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_high_freq_daily"
    _c = _cfg(_n, (2, 10, 1), (0.10, 0.50, 0.10), (2, 20, 2), (0.5, 2.0, 0.5))
    log.info("A11  LL(2-10 s1)×SL(0.10-0.50 s0.10)×LS(2-20 s2)×SS(0.5-2.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  global_r2 — coarse global sweep, different coverage from R1
    #      LL(3-80 s11)×SL(0.20-4.20 s0.80)×LS(3-80 s11)×SS(0.20-4.20 s0.80)  = 8×6×8×6 = 2304
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_global_r2"
    _c = _cfg(_n, (3, 80, 11), (0.20, 4.20, 0.80), (3, 80, 11), (0.20, 4.20, 0.80))
    log.info("A12  LL(3-80 s11)×SL(0.20-4.20 s0.80)×LS(3-80 s11)×SS(0.20-4.20 s0.80)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  NQ Daily countertrend_LS Round-2 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
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
    print(f"Target NP>700K: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="NQ Daily countertrend_LS NP>700K Round-2 search")
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
