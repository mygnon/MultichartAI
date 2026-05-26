"""
search_txf_ct_240min2.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT 240-Minute, Round 2

R1 champion: LL=12, SL=0.45, LS=52, SS=0.1, NP=6,243,800 TWD, MDD=-930,600, Obj=41,892,369, trades=412
R1 dominant regime: tight-SS (SS≈0.05-0.1), moderate SL (0.45), LL=12, LS=52
R1 runner-up (A04 txf_daily_analog): LL=18, SL=0.2, LS=50, SS=0.15, NP=5,164,800

Revised target: NP > 8,000,000 TWD (gap from R1 champion: −21.9%)

Key analogs for reference:
  TXF Hourly (60-min)  best: LL=22 SL=0.425 LS=43 SS=1.771  NP=8,101,400 TWD (795 trades)
  TXF Hourly LS=36 regime: LL=22 SL=0.42  LS=36 SS=1.43    NP=7,653,600 TWD (low MDD)
  TXF Daily  (1440-min) best: LL=25 SL=0.165 LS=50 SS=0.275  NP=4,019,800 TWD (45 trades)
  NQ  Hourly            best: LL=17 SL=0.2   LS=45 SS=1.4    NP=$751,230 USD (1614 trades)

240-min structure: TXF session ≈ 5h/day → ~1.25 bars/day → ~2250 bars in 7yr.
R1 trades=412 → ~59 trades/year (between daily ~6/yr and hourly ~114/yr). Good frequency.

R2 strategy: fine-tune tight-SS regime + probe unexplored adjacent zones.
  1. tight_ss_fine  — high-res scan near R1 champion (LL 8-18, SL 0.30-0.60, LS 44-60, SS 0.05-0.20)
  2. ll_scan        — wider LL range (6-24) with tight SS to find true LL peak
  3. sl_landscape   — SL scan (0.15-0.65) to map full SL sensitivity with tight SS
  4. ls_scan        — LS range (40-70, step=2) to map full LS landscape
  5. ss_landscape   — SS range (0.05-0.50) to fully characterize SS sensitivity
  6. low_sl         — lower SL (0.15-0.50) with tight SS and wider LS
  7. hourly_regime  — probe TXF hourly-like SS (1.4-2.0) to see if it works at 240-min
  8. ls36_zone      — probe lower LS (28-42) like TXF hourly LS=36 low-MDD regime
  9. global_r2      — broad global to ensure no unexplored territory
  10-12. adaptive zooms from best NP

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 tight_ss_fine : LL(8-18 s2)×SL(0.30-0.60 s0.05)×LS(44-60 s2)×SS(0.05-0.20 s0.025) = 6×7×9×7 = 2646
  A02 ll_scan       : LL(6-24 s2)×SL(0.35-0.55 s0.05)×LS(46-56 s2)×SS(0.08-0.20 s0.04)  = 10×5×6×4 = 1200
  A03 sl_landscape  : LL(10-16 s2)×SL(0.15-0.65 s0.05)×LS(46-58 s2)×SS(0.08-0.15 s0.01) = 4×11×7×8 = 2464
  A04 ls_scan       : LL(10-16 s2)×SL(0.35-0.55 s0.05)×LS(40-70 s2)×SS(0.08-0.20 s0.04) = 4×5×16×4 = 1280
  A05 ss_landscape  : LL(8-16 s2)×SL(0.35-0.55 s0.05)×LS(46-58 s4)×SS(0.05-0.50 s0.05)  = 5×5×4×10 = 1000
  A06 low_sl        : LL(10-20 s2)×SL(0.15-0.50 s0.05)×LS(44-60 s2)×SS(0.10-0.30 s0.05) = 6×8×9×5  = 2160
  A07 hourly_regime : LL(18-26 s2)×SL(0.35-0.55 s0.05)×LS(38-52 s2)×SS(1.4-2.0 s0.1)   = 5×5×8×7  = 1400
  A08 ls36_zone     : LL(10-20 s2)×SL(0.30-0.60 s0.05)×LS(28-42 s2)×SS(0.08-0.20 s0.04) = 6×7×8×4  = 1344
  A09 global_r2     : LL(4-64 s10)×SL(0.1-2.6 s0.5)×LS(10-80 s10)×SS(0.05-2.05 s0.5)   = 7×6×8×5  = 1680
  A10 adaptive_zoom : (dynamic from best NP)
  A11 adaptive_zoom2: (dynamic from best NP, tighter)
  A12 adaptive_zoom3: (dynamic from best NP, finest)
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_240min2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0   # TWD (revised up from 800K)

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.05, 20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.05, 20.0

# R1 champion seed
SEED_LL, SEED_SL = 12.0, 0.45
SEED_LS, SEED_SS = 52.0, 0.1
SEED_NP          = 6_243_800.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_240min2_{int(time.time())}.log"
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
        name=f"TXFCT240_2_{name}",
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
    return OUTPUT_DIR / f"TXFCT240_2_{name}_raw.csv"


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
    log.info("=== Starting TXFCT240_2_%s (%d combos) ===", name, cfg.total_runs())
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
        "timeframe": "240min",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_champion": "LL=12 SL=0.45 LS=52 SS=0.1 NP=6,243,800 MDD=-930,600 Obj=41,892,369 trades=412",
            "r2_focus": "Fine-tune tight-SS regime (SS=0.05-0.20); probe LL/SL/LS sensitivity; explore hourly-like SS regime; target 8M",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_240min2.json"
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
    log.info("  TXF 240-Minute countertrend_LS NP>8M TWD — Round 2")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  Timeframe: 240-minute bars  (TXF session ≈5h/day → ~1.25 bars/day)")
    log.info("  R1 champion: LL=12 SL=0.45 LS=52 SS=0.1 NP=6,243,800 (tight-SS regime)")
    log.info("  R2 seed: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  Target: %.0f TWD  (gap from seed: −21.9%%)", TARGET_NP)
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
    # A01  tight_ss_fine — high-res zoom near R1 champion (LL=12 SL=0.45 LS=52 SS=0.1)
    #      LL(8-18 s2)×SL(0.30-0.60 s0.05)×LS(44-60 s2)×SS(0.05-0.20 s0.025) = 6×7×9×7 = 2646
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_tight_ss_fine"
    _c = _cfg(_n, (8, 18, 2), (0.30, 0.60, 0.05), (44, 60, 2), (0.05, 0.20, 0.025))
    log.info("A01  LL(8-18 s2)×SL(0.30-0.60 s0.05)×LS(44-60 s2)×SS(0.05-0.20 s0.025)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  ll_scan — wider LL range with tight SS to find true LL peak
    #      LL(6-24 s2)×SL(0.35-0.55 s0.05)×LS(46-56 s2)×SS(0.08-0.20 s0.04) = 10×5×6×4 = 1200
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ll_scan"
    _c = _cfg(_n, (6, 24, 2), (0.35, 0.55, 0.05), (46, 56, 2), (0.08, 0.20, 0.04))
    log.info("A02  LL(6-24 s2)×SL(0.35-0.55 s0.05)×LS(46-56 s2)×SS(0.08-0.20 s0.04)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  sl_landscape — SL scan (0.15-0.65) to map full SL sensitivity with tight SS
    #      LL(10-16 s2)×SL(0.15-0.65 s0.05)×LS(46-58 s2)×SS(0.08-0.15 s0.01) = 4×11×7×8 = 2464
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_sl_landscape"
    _c = _cfg(_n, (10, 16, 2), (0.15, 0.65, 0.05), (46, 58, 2), (0.08, 0.15, 0.01))
    log.info("A03  LL(10-16 s2)×SL(0.15-0.65 s0.05)×LS(46-58 s2)×SS(0.08-0.15 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ls_scan — LS range (40-70, step=2) to map full LS landscape with tight SS
    #      LL(10-16 s2)×SL(0.35-0.55 s0.05)×LS(40-70 s2)×SS(0.08-0.20 s0.04) = 4×5×16×4 = 1280
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ls_scan"
    _c = _cfg(_n, (10, 16, 2), (0.35, 0.55, 0.05), (40, 70, 2), (0.08, 0.20, 0.04))
    log.info("A04  LL(10-16 s2)×SL(0.35-0.55 s0.05)×LS(40-70 s2)×SS(0.08-0.20 s0.04)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ss_landscape — SS range (0.05-0.50) to fully characterize SS sensitivity
    #      LL(8-16 s2)×SL(0.35-0.55 s0.05)×LS(46-58 s4)×SS(0.05-0.50 s0.05) = 5×5×4×10 = 1000
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ss_landscape"
    _c = _cfg(_n, (8, 16, 2), (0.35, 0.55, 0.05), (46, 58, 4), (0.05, 0.50, 0.05))
    log.info("A05  LL(8-16 s2)×SL(0.35-0.55 s0.05)×LS(46-58 s4)×SS(0.05-0.50 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  low_sl — lower SL (0.15-0.50) with tight SS, mimicking NQ ultra-tight SL
    #      LL(10-20 s2)×SL(0.15-0.50 s0.05)×LS(44-60 s2)×SS(0.10-0.30 s0.05) = 6×8×9×5 = 2160
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_low_sl"
    _c = _cfg(_n, (10, 20, 2), (0.15, 0.50, 0.05), (44, 60, 2), (0.10, 0.30, 0.05))
    log.info("A06  LL(10-20 s2)×SL(0.15-0.50 s0.05)×LS(44-60 s2)×SS(0.10-0.30 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  hourly_regime — probe TXF hourly-like SS (1.4-2.0) at 240-min bars
    #      LL(18-26 s2)×SL(0.35-0.55 s0.05)×LS(38-52 s2)×SS(1.4-2.0 s0.1) = 5×5×8×7 = 1400
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_hourly_regime"
    _c = _cfg(_n, (18, 26, 2), (0.35, 0.55, 0.05), (38, 52, 2), (1.4, 2.0, 0.1))
    log.info("A07  LL(18-26 s2)×SL(0.35-0.55 s0.05)×LS(38-52 s2)×SS(1.4-2.0 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  ls36_zone — probe lower LS (28-42) like TXF hourly LS=36 low-MDD regime
    #      LL(10-20 s2)×SL(0.30-0.60 s0.05)×LS(28-42 s2)×SS(0.08-0.20 s0.04) = 6×7×8×4 = 1344
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_ls36_zone"
    _c = _cfg(_n, (10, 20, 2), (0.30, 0.60, 0.05), (28, 42, 2), (0.08, 0.20, 0.04))
    log.info("A08  LL(10-20 s2)×SL(0.30-0.60 s0.05)×LS(28-42 s2)×SS(0.08-0.20 s0.04)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  global_r2 — broad global sweep to catch any unexplored territory
    #      LL(4-64 s10)×SL(0.1-2.6 s0.5)×LS(10-80 s10)×SS(0.05-2.05 s0.5) = 7×6×8×5 = 1680
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_global_r2"
    _c = _cfg(_n, (4, 64, 10), (0.1, 2.6, 0.5), (10, 80, 10), (0.05, 2.05, 0.5))
    log.info("A09  LL(4-64 s10)×SL(0.1-2.6 s0.5)×LS(10-80 s10)×SS(0.05-2.05 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — wide zoom around best NP found so far
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (10, 0.30, 14, 0.15),
            (7,  0.20, 10, 0.10),
            (5,  0.15,  7, 0.07),
            (4,  0.10,  5, 0.05),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.05, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 2.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.025, SS_LO, SS_HI)
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
            (6,  0.15, 8, 0.06),
            (4,  0.10, 6, 0.04),
            (3,  0.08, 4, 0.03),
            (2,  0.05, 3, 0.025),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.05, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 2.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.025, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  adaptive_zoom3 — tight zoom for final precision
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_adaptive_zoom3"
    log.info("A12  adaptive_zoom3 — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (4,  0.10, 6, 0.04),
            (3,  0.08, 4, 0.03),
            (2,  0.06, 3, 0.025),
            (2,  0.05, 2, 0.025),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.05, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 2.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.025, SS_LO, SS_HI)
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
    log.info("  TXF 240-Minute countertrend_LS Round-2 COMPLETE")
    log.info("  Best NP: %.0f TWD  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R1 seed: 6,243,800  R2 best: %.0f  gain: %+.1f%%",
             best_np, (best_np / 6_243_800 - 1) * 100)
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
        description="TXF 240-Minute countertrend_LS NP>8M TWD Round-2 search")
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
