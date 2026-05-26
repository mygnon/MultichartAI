"""
search_txf_ct_daily.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Daily, Round 1

Strategy: BUY when C crosses over lower BB; SELLSHORT when C crosses under upper BB.
Reversal exits only — no STP or LMT.
Params: LENGTH_LONG (LL), STDDEV_LONG (SL), LENGTH_SHORT (LS), STDDEV_SHORT (SS)
Target: NP > 8,000,000 TWD
Timeframe: DAILY (1440-min bars), 2019/01/01–2026/01/01

No prior TXF daily CT data — R1 is a broad exploration.
Reference points:
  NQ CT daily best (R4 ceiling): LL=8  SL=0.7  LS=47  SS=1.86  NP=$460,770
  TXF CT hourly best (R4):       LL=22 SL=0.425 LS=43 SS=1.771 NP=8,101,400 TWD
  TXF CT hourly alt low-MDD:     LL=22 SL=0.40  LS=36 SS=1.40  Obj=83M  MDD=-609K

R1 strategy — broad 8-attempt exploration + 4 adaptive zooms:
  A01 global_broad    : Wide coarse sweep (LL step=5, SL step=0.3, LS step=10, SS step=0.5)
  A02 global_fill     : Offset grid from A01 (fill gaps in parameter space)
  A03 nq_analog       : Center on NQ daily champion (LL=8, SL=0.7, LS=48, SS=1.86)
  A04 txf_hourly_analog: Center on TXF hourly champion (LL=22, SL=0.425, LS=43, SS=1.771)
  A05 ultra_tight_sl  : SL=0.1-0.5 (NQ hourly breakthrough: SL=0.2 unlocked 700K)
  A06 short_ls        : LS=10-40 (TXF hourly LS=36 low-MDD regime analog)
  A07 tight_ss        : SS=0.4-1.4 (tight short-entry band regime)
  A08 ll_broad        : LL=1-50 broad sweep (check high-LL and low-LL regimes)
  A09 adaptive_zoom   : step=(1,0.05,1,0.05) cascade from best (round 1)
  A10 medium_zoom     : step=(1,0.02,1,0.02) cascade from best (round 2)
  A11 fine_zoom       : step=(1,0.01,1,0.01) cascade from best (round 3)
  A12 fine_ss_tune    : Fine-tune SS step=0.01 around best

Attempt schedule (≤5,000 combos each):
  A01  1512  LL(2-27 s5)×SL(0.2-2.0 s0.3)×LS(5-55 s10)×SS(0.5-3.0 s0.5)
  A02  1080  LL(4-29 s5)×SL(0.35-1.85 s0.3)×LS(10-60 s10)×SS(0.75-2.75 s0.5)
  A03  3234  LL(4-14 s2)×SL(0.3-1.3 s0.1)×LS(36-60 s4)×SS(1.2-2.4 s0.2)
  A04  1764  LL(16-28 s2)×SL(0.2-0.8 s0.1)×LS(33-53 s4)×SS(1.3-2.3 s0.2)
  A05  1575  LL(10-28 s3)×SL(0.1-0.5 s0.05)×LS(35-55 s5)×SS(1.0-2.0 s0.25)
  A06  1715  LL(6-22 s4)×SL(0.2-0.8 s0.1)×LS(10-40 s5)×SS(1.0-2.2 s0.2)
  A07  2310  LL(6-22 s4)×SL(0.4-1.4 s0.2)×LS(30-60 s5)×SS(0.4-1.4 s0.1)
  A08   800  LL(1-50 s7)×SL(0.3-1.5 s0.3)×LS(20-60 s10)×SS(1.0-2.5 s0.5)
  A09  ≤5000 adaptive zoom step=(1,0.05,1,0.05)
  A10  ≤5000 adaptive zoom step=(1,0.02,1,0.02)
  A11  ≤5000 adaptive zoom step=(1,0.01,1,0.01)
  A12  ≤5000 fine SS step=0.01 around best

NOTE: Requires TWF.TXF HOT daily chart open in workspace 20260521SFJ_Bollinger_AI.wsp
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_daily_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.05, 20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.05, 20.0

# No prior TXF daily CT data — use NQ daily champion as seed
SEED_LL, SEED_SL = 8.0,  0.7
SEED_LS, SEED_SS = 48.0, 1.86
SEED_NP          = 0.0   # unknown — first run

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_daily_{int(time.time())}.log"
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
        name=f"TXFCTD1_{name}",
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
    return OUTPUT_DIR / f"TXFCTD1_{name}_raw.csv"


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
    log.info("=== Starting TXFCTD1_%s (%d combos) ===", name, cfg.total_runs())
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
    """Priority: target met → highest Obj; else highest NP (target-chasing mode)."""
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
        "round": 1,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "daily",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":     "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":     "Reversal only — no STP or LMT",
            "params":    "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_focus":  "R1 first-ever TXF daily CT search; broad exploration before zooming",
            "reference": "NQ daily ceiling $460K (LL=8 SL=0.7 LS=47 SS=1.86); TXF hourly 8M (LL=22 SL=0.425 LS=43 SS=1.771)",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_daily.json"
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
    log.info("  TXF Daily countertrend_LS NP>8,000,000 TWD — Round 1")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  No prior data — R1 broad exploration")
    log.info("  Reference: NQ daily ceiling $460K (LL=8 SL=0.7 LS=47 SS=1.86)")
    log.info("  Reference: TXF hourly 8M (LL=22 SL=0.425 LS=43 SS=1.771)")
    log.info("  Target: %.0f TWD", TARGET_NP)
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
    # A01  global_broad — Wide coarse scan of full parameter space
    #      LL(2-27 s5)×SL(0.2-2.0 s0.3)×LS(5-55 s10)×SS(0.5-3.0 s0.5) = 6×7×6×6=1512
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_global_broad"
    _c = _cfg(_n, (2, 27, 5), (0.2, 2.0, 0.3), (5, 55, 10), (0.5, 3.0, 0.5))
    log.info("A01  LL(2-27 s5)×SL(0.2-2.0 s0.3)×LS(5-55 s10)×SS(0.5-3.0 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  global_fill — Offset grid to fill A01 gaps
    #      LL(4-29 s5)×SL(0.35-1.85 s0.3)×LS(10-60 s10)×SS(0.75-2.75 s0.5) = 6×6×6×5=1080
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_global_fill"
    _c = _cfg(_n, (4, 29, 5), (0.35, 1.85, 0.3), (10, 60, 10), (0.75, 2.75, 0.5))
    log.info("A02  LL(4-29 s5)×SL(0.35-1.85 s0.3)×LS(10-60 s10)×SS(0.75-2.75 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  nq_analog — Center on NQ daily champion (LL=8, SL=0.7, LS=48, SS=1.86)
    #      LL(4-14 s2)×SL(0.3-1.3 s0.1)×LS(36-60 s4)×SS(1.2-2.4 s0.2) = 6×11×7×7=3234
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_nq_analog"
    _c = _cfg(_n, (4, 14, 2), (0.3, 1.3, 0.1), (36, 60, 4), (1.2, 2.4, 0.2))
    log.info("A03  LL(4-14 s2)×SL(0.3-1.3 s0.1)×LS(36-60 s4)×SS(1.2-2.4 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  txf_hourly_analog — Center on TXF hourly champion (LL=22, SL=0.425, LS=43, SS=1.771)
    #      LL(16-28 s2)×SL(0.2-0.8 s0.1)×LS(33-53 s4)×SS(1.3-2.3 s0.2) = 7×7×6×6=1764
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_txf_hourly_analog"
    _c = _cfg(_n, (16, 28, 2), (0.2, 0.8, 0.1), (33, 53, 4), (1.3, 2.3, 0.2))
    log.info("A04  LL(16-28 s2)×SL(0.2-0.8 s0.1)×LS(33-53 s4)×SS(1.3-2.3 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ultra_tight_sl — SL=0.1-0.5 (NQ hourly SL=0.2 breakthrough analog)
    #      LL(10-28 s3)×SL(0.1-0.5 s0.05)×LS(35-55 s5)×SS(1.0-2.0 s0.25) = 7×9×5×5=1575
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ultra_tight_sl"
    _c = _cfg(_n, (10, 28, 3), (0.1, 0.5, 0.05), (35, 55, 5), (1.0, 2.0, 0.25))
    log.info("A05  LL(10-28 s3)×SL(0.1-0.5 s0.05)×LS(35-55 s5)×SS(1.0-2.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  short_ls — LS=10-40 (TXF hourly LS=36 low-MDD regime analog for daily)
    #      LL(6-22 s4)×SL(0.2-0.8 s0.1)×LS(10-40 s5)×SS(1.0-2.2 s0.2) = 5×7×7×7=1715
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_short_ls"
    _c = _cfg(_n, (6, 22, 4), (0.2, 0.8, 0.1), (10, 40, 5), (1.0, 2.2, 0.2))
    log.info("A06  LL(6-22 s4)×SL(0.2-0.8 s0.1)×LS(10-40 s5)×SS(1.0-2.2 s0.2)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  tight_ss — SS=0.4-1.4 (tight short-entry band regime)
    #      LL(6-22 s4)×SL(0.4-1.4 s0.2)×LS(30-60 s5)×SS(0.4-1.4 s0.1) = 5×6×7×11=2310
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_tight_ss"
    _c = _cfg(_n, (6, 22, 4), (0.4, 1.4, 0.2), (30, 60, 5), (0.4, 1.4, 0.1))
    log.info("A07  LL(6-22 s4)×SL(0.4-1.4 s0.2)×LS(30-60 s5)×SS(0.4-1.4 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  ll_broad — LL=1-50 broad sweep (high-LL and low-LL regimes)
    #      LL(1-50 s7)×SL(0.3-1.5 s0.3)×LS(20-60 s10)×SS(1.0-2.5 s0.5) = 8×5×5×4=800
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_ll_broad"
    _c = _cfg(_n, (1, 50, 7), (0.3, 1.5, 0.3), (20, 60, 10), (1.0, 2.5, 0.5))
    log.info("A08  LL(1-50 s7)×SL(0.3-1.5 s0.3)×LS(20-60 s10)×SS(1.0-2.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom — SL step=0.05, SS step=0.05 (round-1 zoom)
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom"
    log.info("A09  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.25, 3, 0.25),   # 5×11×7×11=4235
            (2, 0.20, 3, 0.20),   # 5×9×7×9=2835
            (2, 0.15, 2, 0.20),   # 5×7×5×9=1575
            (2, 0.10, 2, 0.15),   # 5×5×5×7=875
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.05, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.05, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A09  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  medium_zoom — SL step=0.02, SS step=0.02 (round-2 zoom)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_medium_zoom"
    log.info("A10  medium_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.10, 3, 0.10),   # 5×11×7×11=4235
            (2, 0.08, 2, 0.08),   # 5×9×5×9=2025
            (2, 0.06, 2, 0.06),   # 5×7×5×7=1225
            (2, 0.04, 2, 0.04),   # 5×5×5×5=625
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.02, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.02, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  fine_zoom — SL step=0.01, SS step=0.01 (round-3 zoom)
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_fine_zoom"
    log.info("A11  fine_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.06, 2, 0.06),   # 5×13×5×13=4225
            (2, 0.04, 2, 0.05),   # 5×9×5×11=2475
            (1, 0.05, 1, 0.05),   # 3×11×3×11=1089
            (1, 0.04, 1, 0.04),   # 3×9×3×9=729
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.01, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.01, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  fine_ss_tune — Fine SS step=0.01 with tight LL/SL/LS (final fine-tune)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_fine_ss_tune"
    log.info("A12  fine_ss_tune — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.04, 2, 0.12),   # 5×9×5×25=5625  — may exceed
            (1, 0.04, 2, 0.12),   # 3×9×5×25=3375 ✓
            (1, 0.04, 1, 0.12),   # 3×9×3×25=2025 ✓
            (1, 0.02, 1, 0.10),   # 3×5×3×21=945 ✓
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.02, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.01, SS_LO, SS_HI)
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
    log.info("  TXF Daily countertrend_LS Round-1 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  Target %.0f TWD: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{max(0, TARGET_NP - best_np):.0f})")
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
        description="TXF Daily countertrend_LS NP>8M TWD Round-1 search")
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
