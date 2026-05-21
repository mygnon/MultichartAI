"""
search_nq_ct_daily4.py — SFJ_15Dworkshop_lesson5_countertrend_LS on CME.NQ HOT Daily, Round 4

R3 findings (search_nq_ct_daily3.py):
  Best NP : LL=8  SL=0.7 LS=48 SS=1.85  NP=456,050  MDD=-81,535  Obj=2,550,826  trades=38
  Alt     : LL=2  SL=any LS=12 SS=1.9   NP=422,335  MDD=-75,690  Obj=2,356,544  trades=78
  Gap: -34.9% from 700K  (R3 improvement: +5.7%, decelerating from R2 +11.3%)
  SS=1.85 confirmed sweet spot (step=0.05 found it; step=0.1 missed it in R1/R2)
  SL inert in LL=2 regime — all SL(0.1-0.9) give same NP

  Progression:  R1=387,590  R2=431,500  R3=456,050
  Rate of gain: +11.3%      +5.7%       — ceiling forming ~460-470K?

R4 goal: final exhaustive sweep to either find 700K or confirm ceiling.
  A01 ss_step01        : SS=1.80-1.90 step 0.01 — confirm true SS peak
  A02 sl_fine          : SL=0.62-0.78 step 0.02 with very tight LS/SS center
  A03 ll2_ext_ls       : LL=2-4 × LS=12-40 — does LL=2 improve with longer LS?
  A04 ll2_fine_ss      : LL=2-3 × LS=10-16 fine × SS=1.70-2.00 step 0.05
  A05 high_ls_extreme  : LS=100-500 (extreme-move fades only)
  A06 high_freq_fine   : LL=2-8 × LS=5-12 — 100+ trades regime possible?
  A07 center_finest    : LL(7-10)×SL(0.62-0.78 s0.01)×LS(46-50)×SS(1.82-1.88 s0.01)
  A08 sl_broad         : LL(6-12)×SL(0.1-0.6 s0.05)×LS(45-52)×SS(1.83-1.87 s0.01)
  A09 very_short_ls    : LL(2-10)×SL(0.1-1.0)×LS(2-9) — ultra-short LS regime
  A10 adaptive_zoom    : finest possible (step=1 period, step=0.01 stddev)
  A11 ss_below_18      : SS=1.4-1.7 (below current best) — fully explored?
  A12 global_r4        : coarse global (different from R1-R3, includes LL=2-30)

Attempt schedule (≤5,000 combos each):
  A01  2156   LL(7-10 s1)×SL(0.55-0.85 s0.05)×LS(45-51 s1)×SS(1.800-1.900 s0.010)
  A02  2548   LL(7-10 s1)×SL(0.58-0.82 s0.02)×LS(45-51 s1)×SS(1.820-1.880 s0.010)
  A03  1350   LL(2-4 s1)×SL(0.5-1.5 s0.2)×LS(12-40 s2)×SS(1.7-2.1 s0.1)
  A04   980   LL(2-3 s1)×SL(0.1-1.0 s0.1)×LS(10-16 s1)×SS(1.70-2.00 s0.05)
  A05   900   LL(2-10 s2)×SL(0.5-2.5 s0.5)×LS(100-500 s50)×SS(2.0-5.0 s1.0)
  A06  1344   LL(2-8 s1)×SL(0.1-0.8 s0.1)×LS(5-12 s1)×SS(1.5-2.5 s0.5)
  A07  2380   LL(7-10 s1)×SL(0.62-0.78 s0.01)×LS(46-50 s1)×SS(1.820-1.880 s0.010)
  A08  3080   LL(6-12 s1)×SL(0.10-0.60 s0.05)×LS(45-52 s1)×SS(1.830-1.870 s0.010)
  A09  2160   LL(2-10 s1)×SL(0.1-1.0 s0.1)×LS(2-9 s1)×SS(1.5-2.5 s0.5)
  A10  ≤5000  adaptive zoom (step=1 period, step=0.01 stddev)
  A11  2205   LL(6-12 s1)×SL(0.5-0.9 s0.1)×LS(44-52 s1)×SS(1.40-1.70 s0.05)
  A12  4900   LL(2-29 s3)×SL(0.2-4.0 s0.6)×LS(2-29 s3)×SS(0.2-4.0 s0.6)
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_ct_daily4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 700_000.0

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.05, 20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.05, 20.0

# Seeds from R3 best NP winner
SEED_LL, SEED_SL = 8.0,  0.7
SEED_LS, SEED_SS = 48.0, 1.85
SEED_NP          = 456_050.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_ct_daily4_{int(time.time())}.log"
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
        name=f"NQCTD4_{name}",
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
    return OUTPUT_DIR / f"NQCTD4_{name}_raw.csv"


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
    log.info("=== Starting NQCTD4_%s (%d combos) ===", name, cfg.total_runs())
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
        "timeframe": "daily",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r3_best":  "LL=8 SL=0.7 LS=48 SS=1.85 NP=456050 MDD=-81535 trades=38",
            "r3_alt":   "LL=2 SL=any LS=12 SS=1.9  NP=422335 MDD=-75690 trades=78",
            "r4_focus": "SS step=0.01, SL step=0.02, LL=2 wider, extreme LS, high-freq fine",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_ct_daily4.json"
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
    log.info("  NQ Daily countertrend_LS NP>700K — Round 4 (Final)")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  Timeframe: DAILY (1440 min bars)")
    log.info("  R3 best: LL=8 SL=0.7 LS=48 SS=1.85 NP=456050")
    log.info("  Progression: 387K→431K→456K (rates +11.3%%→+5.7%%)")
    log.info("  Target: %.0f USD  Gap from R3: +%.0f", TARGET_NP, TARGET_NP - SEED_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_ll, best_sl, best_ls, best_ss
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_ll, best_sl, best_ls, best_ss,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-26s]  no valid data", attempt_num, name)
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

        log.info("  [A%02d %-26s]  LL=%.4g SL=%.4g LS=%.4g SS=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, ll, sl, ls, ss, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  ss_step01 — SS=1.80-1.90 step 0.01, confirm exact SS peak
    #      LL(7-10 s1)×SL(0.55-0.85 s0.05)×LS(45-51 s1)×SS(1.800-1.900 s0.010)  = 4×7×7×11 = 2156
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ss_step01"
    _c = _cfg(_n, (7, 10, 1), (0.55, 0.85, 0.05), (45, 51, 1), (1.800, 1.900, 0.010))
    log.info("A01  LL(7-10 s1)×SL(0.55-0.85 s0.05)×LS(45-51 s1)×SS(1.800-1.900 s0.010)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  sl_fine — SL step=0.02, tight center LS/SS
    #      LL(7-10 s1)×SL(0.58-0.82 s0.02)×LS(45-51 s1)×SS(1.820-1.880 s0.010)  = 4×13×7×7 = 2548
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_sl_fine"
    _c = _cfg(_n, (7, 10, 1), (0.58, 0.82, 0.02), (45, 51, 1), (1.820, 1.880, 0.010))
    log.info("A02  LL(7-10 s1)×SL(0.58-0.82 s0.02)×LS(45-51 s1)×SS(1.820-1.880 s0.010)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ll2_ext_ls — LL=2-4 × LS=12-40, does LL=2 improve with longer LS?
    #      LL(2-4 s1)×SL(0.5-1.5 s0.2)×LS(12-40 s2)×SS(1.7-2.1 s0.1)  = 3×6×15×5 = 1350
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ll2_ext_ls"
    _c = _cfg(_n, (2, 4, 1), (0.5, 1.5, 0.2), (12, 40, 2), (1.7, 2.1, 0.1))
    log.info("A03  LL(2-4 s1)×SL(0.5-1.5 s0.2)×LS(12-40 s2)×SS(1.7-2.1 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ll2_fine_ss — LL=2-3 fine SS=1.70-2.00 step 0.05 around LL=2 winner
    #      LL(2-3 s1)×SL(0.1-1.0 s0.1)×LS(10-16 s1)×SS(1.70-2.00 s0.05)  = 2×10×7×7 = 980
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ll2_fine_ss"
    _c = _cfg(_n, (2, 3, 1), (0.1, 1.0, 0.1), (10, 16, 1), (1.70, 2.00, 0.05))
    log.info("A04  LL(2-3 s1)×SL(0.1-1.0 s0.1)×LS(10-16 s1)×SS(1.70-2.00 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  high_ls_extreme — LS=100-500 (fade only extreme daily moves)
    #      LL(2-10 s2)×SL(0.5-2.5 s0.5)×LS(100-500 s50)×SS(2.0-5.0 s1.0)  = 5×5×9×4 = 900
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_high_ls_extreme"
    _c = _cfg(_n, (2, 10, 2), (0.5, 2.5, 0.5), (100, 500, 50), (2.0, 5.0, 1.0))
    log.info("A05  LL(2-10 s2)×SL(0.5-2.5 s0.5)×LS(100-500 s50)×SS(2.0-5.0 s1.0)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  high_freq_fine — LL=2-8 × LS=5-12 (can 100+ trades work?)
    #      LL(2-8 s1)×SL(0.1-0.8 s0.1)×LS(5-12 s1)×SS(1.5-2.5 s0.5)  = 7×8×8×3 = 1344
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_freq_fine"
    _c = _cfg(_n, (2, 8, 1), (0.1, 0.8, 0.1), (5, 12, 1), (1.5, 2.5, 0.5))
    log.info("A06  LL(2-8 s1)×SL(0.1-0.8 s0.1)×LS(5-12 s1)×SS(1.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  center_finest — SL step=0.01 + SS step=0.01 around winner core
    #      LL(7-10 s1)×SL(0.62-0.78 s0.01)×LS(46-50 s1)×SS(1.820-1.880 s0.010)  = 4×17×5×7 = 2380
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_center_finest"
    _c = _cfg(_n, (7, 10, 1), (0.62, 0.78, 0.01), (46, 50, 1), (1.820, 1.880, 0.010))
    log.info("A07  LL(7-10 s1)×SL(0.62-0.78 s0.01)×LS(46-50 s1)×SS(1.820-1.880 s0.010)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  sl_broad — wider SL scan (0.1-0.6) with very tight SS
    #      LL(6-12 s1)×SL(0.10-0.60 s0.05)×LS(45-52 s1)×SS(1.830-1.870 s0.010)  = 7×11×8×5 = 3080
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_sl_broad"
    _c = _cfg(_n, (6, 12, 1), (0.10, 0.60, 0.05), (45, 52, 1), (1.830, 1.870, 0.010))
    log.info("A08  LL(6-12 s1)×SL(0.10-0.60 s0.05)×LS(45-52 s1)×SS(1.830-1.870 s0.010)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  very_short_ls — LS=2-9 (ultra-short short-entry BB)
    #      LL(2-10 s1)×SL(0.1-1.0 s0.1)×LS(2-9 s1)×SS(1.5-2.5 s0.5)  = 9×10×8×3 = 2160
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_very_short_ls"
    _c = _cfg(_n, (2, 10, 1), (0.1, 1.0, 0.1), (2, 9, 1), (1.5, 2.5, 0.5))
    log.info("A09  LL(2-10 s1)×SL(0.1-1.0 s0.1)×LS(2-9 s1)×SS(1.5-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — finest (step=1 period, step=0.01 stddev)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (8,  0.5,  8, 0.5),
            (5,  0.3,  5, 0.3),
            (3,  0.15, 3, 0.15),
            (2,  0.1,  2, 0.1),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.01, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.01, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  ss_below_18 — SS=1.40-1.70 step 0.05 (below current best range)
    #      LL(6-12 s1)×SL(0.5-0.9 s0.1)×LS(44-52 s1)×SS(1.40-1.70 s0.05)  = 7×5×9×7 = 2205
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_ss_below_18"
    _c = _cfg(_n, (6, 12, 1), (0.5, 0.9, 0.1), (44, 52, 1), (1.40, 1.70, 0.05))
    log.info("A11  LL(6-12 s1)×SL(0.5-0.9 s0.1)×LS(44-52 s1)×SS(1.40-1.70 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  global_r4 — coarse global, different from R1-R3
    #      LL(2-29 s3)×SL(0.2-4.0 s0.6)×LS(2-29 s3)×SS(0.2-4.0 s0.6)  = 10×7×10×7 = 4900
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_global_r4"
    _c = _cfg(_n, (2, 29, 3), (0.2, 4.0, 0.6), (2, 29, 3), (0.2, 4.0, 0.6))
    log.info("A12  LL(2-29 s3)×SL(0.2-4.0 s0.6)×LS(2-29 s3)×SS(0.2-4.0 s0.6)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  NQ Daily countertrend_LS Round-4 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  Progression: R1=387590  R2=431500  R3=456050  R4=%.0f", best_np)
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
        description="NQ Daily countertrend_LS NP>700K Round-4 (Final) search")
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
