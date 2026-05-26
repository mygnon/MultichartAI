"""
search_txf_ct_hourly2.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Hourly, Round 2

R1 best: LL=22, SL=0.4, LS=44, SS=1.8, NP=7,641,000, MDD=-1,225,600, trades=773
R1 7M target: MET ✅

R2 target: NP > 8,000,000 TWD  (gap from R1 best: +359,000 TWD = +4.7%)

Key R1 insights:
- Champion regime: LL≈20-25, SL≈0.4, LS≈42-46, SS≈1.8 — asymmetric short long-BB
- SL=0.4 is the TXF sweet spot — ultra-tight SL=0.25 HURTS (6.6M vs 7.64M)
  → Different from NQ hourly where SL=0.2 was breakthrough; TXF optimal SL is higher
- SS=1.8 (A02 step=0.2) — finer SS scan (step=0.05) may find SS=1.75 or 1.85 peak
- LS=44 > LS=45 (7.64M vs 7.38M) — need step=1 LS scan around 38-52
- A10 adaptive zoom returned 0 NP due to IME bug (now fixed) — must re-run fine zoom
- High-freq (LL=7,LS=20) gave only 3.97M — wrong regime for TXF

R2 strategy:
  1. Fine grid around champion (LL step=1, SL step=0.05, LS step=1, SS step=0.10)
  2. SS step=0.05 scan (A02 used step=0.2, missed intermediate values)
  3. SL step=0.02 scan (fine SL between 0.10-0.50)
  4. LL extend (explore LL=24-42)
  5. LS fine scan (step=1, wider range 36-54)
  6. SS above 2.0 (unexplored at fine grain)
  7. SL ultra-tight (0.05-0.25, verify A11 finding)
  8. Short LL regime (LL=8-18)
  9. Global R2 overview
  10. Adaptive zoom (step=0.05 for SL/SS — re-run with IME fix)
  11. Finest center (SL step=0.02, SS step=0.02 around champion peak)
  12. Cross-regime (short LL+LS combo)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 fine_r1_best  : LL(18-26 s1)×SL(0.25-0.55 s0.05)×LS(38-50 s2)×SS(1.60-2.00 s0.10) = 9×7×7×5=2205
  A02 ss_fine       : LL(20-24 s1)×SL(0.30-0.50 s0.05)×LS(40-48 s1)×SS(1.40-2.00 s0.05) = 5×5×9×13=2925
  A03 sl_fine       : LL(18-26 s2)×SL(0.10-0.50 s0.02)×LS(42-50 s2)×SS(1.60-2.00 s0.20) = 5×21×5×3=1575
  A04 ll_extend     : LL(24-42 s2)×SL(0.20-0.60 s0.10)×LS(38-52 s2)×SS(1.40-2.20 s0.20) = 10×5×8×5=2000
  A05 ls_fine       : LL(20-24 s1)×SL(0.30-0.50 s0.05)×LS(36-54 s1)×SS(1.60-2.00 s0.20) = 5×5×19×3=1425
  A06 ss_above2     : LL(18-28 s2)×SL(0.20-0.50 s0.10)×LS(38-52 s2)×SS(2.00-3.00 s0.20) = 6×4×8×6=1152
  A07 sl_ultra      : LL(16-28 s2)×SL(0.05-0.25 s0.05)×LS(38-52 s2)×SS(1.20-2.00 s0.20) = 7×5×8×5=1400
  A08 ll_short      : LL(8-18 s2)×SL(0.20-0.60 s0.10)×LS(36-52 s2)×SS(1.20-2.00 s0.20)  = 6×5×9×5=1350
  A09 global_r2     : LL(5-65 s10)×SL(0.20-2.00 s0.20)×LS(5-65 s10)×SS(0.50-2.50 s0.50) = 7×10×7×5=2450
  A10 adaptive_zoom : (dynamic from R2 best NP, step=0.05 for SL/SS — IME-safe)
  A11 finest_center : LL(20-24 s1)×SL(0.32-0.48 s0.02)×LS(42-46 s1)×SS(1.70-1.90 s0.02) = 5×9×5×11=2475
  A12 cross_regime  : LL(8-20 s2)×SL(0.10-0.50 s0.10)×LS(20-40 s4)×SS(1.00-2.00 s0.20)  = 7×5×6×6=1260
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_hourly2_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.1,  20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.1,  20.0

# R1 champion as seed
SEED_LL, SEED_SL = 22.0, 0.4
SEED_LS, SEED_SS = 44.0, 1.8
SEED_NP          = 7_641_000.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_hourly2_{int(time.time())}.log"
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
        name=f"TXFCT2_{name}",
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
    return OUTPUT_DIR / f"TXFCT2_{name}_raw.csv"


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
    log.info("=== Starting TXFCT2_%s (%d combos) ===", name, cfg.total_runs())
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
        "timeframe": "hourly",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_best":  "LL=22 SL=0.4 LS=44 SS=1.8 NP=7641000 MDD=-1225600 trades=773",
            "r1_notes": "SL=0.4 is TXF sweet spot (NOT ultra-tight unlike NQ); SS=1.8 needs finer scan",
            "r2_focus": "Fine SL/SS/LL/LS scans; SS step=0.05; re-run adaptive zoom with IME fix",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_hourly2.json"
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
    log.info("  TXF Hourly countertrend_LS NP>8,000,000 TWD — Round 2")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R1 best: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R2 focus: fine SL/SS/LS scans, adaptive zoom (IME-fixed), LL extend")
    log.info("  Target: %.0f TWD  (gap from R1: +%.0f = +%.1f%%)",
             TARGET_NP, TARGET_NP - SEED_NP, (TARGET_NP / SEED_NP - 1) * 100)
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
    # A01  fine_r1_best — fine grid around champion (LL/SL/SS all tightened)
    #      LL(18-26 s1)×SL(0.25-0.55 s0.05)×LS(38-50 s2)×SS(1.60-2.00 s0.10) = 9×7×7×5 = 2205
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_fine_r1_best"
    _c = _cfg(_n, (18, 26, 1), (0.25, 0.55, 0.05), (38, 50, 2), (1.60, 2.00, 0.10))
    log.info("A01  LL(18-26 s1)×SL(0.25-0.55 s0.05)×LS(38-50 s2)×SS(1.60-2.00 s0.10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  ss_fine — SS step=0.05 (R1 used step=0.2, may have missed peak)
    #      LL(20-24 s1)×SL(0.30-0.50 s0.05)×LS(40-48 s1)×SS(1.40-2.00 s0.05) = 5×5×9×13 = 2925
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ss_fine"
    _c = _cfg(_n, (20, 24, 1), (0.30, 0.50, 0.05), (40, 48, 1), (1.40, 2.00, 0.05))
    log.info("A02  LL(20-24 s1)×SL(0.30-0.50 s0.05)×LS(40-48 s1)×SS(1.40-2.00 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  sl_fine — SL step=0.02 (very fine SL 0.10-0.50)
    #      LL(18-26 s2)×SL(0.10-0.50 s0.02)×LS(42-50 s2)×SS(1.60-2.00 s0.20) = 5×21×5×3 = 1575
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_sl_fine"
    _c = _cfg(_n, (18, 26, 2), (0.10, 0.50, 0.02), (42, 50, 2), (1.60, 2.00, 0.20))
    log.info("A03  LL(18-26 s2)×SL(0.10-0.50 s0.02)×LS(42-50 s2)×SS(1.60-2.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ll_extend — LL=24-42 (explore longer long period)
    #      LL(24-42 s2)×SL(0.20-0.60 s0.10)×LS(38-52 s2)×SS(1.40-2.20 s0.20) = 10×5×8×5 = 2000
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ll_extend"
    _c = _cfg(_n, (24, 42, 2), (0.20, 0.60, 0.10), (38, 52, 2), (1.40, 2.20, 0.20))
    log.info("A04  LL(24-42 s2)×SL(0.20-0.60 s0.10)×LS(38-52 s2)×SS(1.40-2.20 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ls_fine — LS step=1 wider range (R1 step=2 may have missed LS peak)
    #      LL(20-24 s1)×SL(0.30-0.50 s0.05)×LS(36-54 s1)×SS(1.60-2.00 s0.20) = 5×5×19×3 = 1425
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ls_fine"
    _c = _cfg(_n, (20, 24, 1), (0.30, 0.50, 0.05), (36, 54, 1), (1.60, 2.00, 0.20))
    log.info("A05  LL(20-24 s1)×SL(0.30-0.50 s0.05)×LS(36-54 s1)×SS(1.60-2.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ss_above2 — SS=2.0-3.0 (unexplored at fine grain)
    #      LL(18-28 s2)×SL(0.20-0.50 s0.10)×LS(38-52 s2)×SS(2.00-3.00 s0.20) = 6×4×8×6 = 1152
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ss_above2"
    _c = _cfg(_n, (18, 28, 2), (0.20, 0.50, 0.10), (38, 52, 2), (2.00, 3.00, 0.20))
    log.info("A06  LL(18-28 s2)×SL(0.20-0.50 s0.10)×LS(38-52 s2)×SS(2.00-3.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  sl_ultra — very tight SL=0.05-0.25 (verify A11 finding; step=0.05)
    #      LL(16-28 s2)×SL(0.05-0.25 s0.05)×LS(38-52 s2)×SS(1.20-2.00 s0.20) = 7×5×8×5 = 1400
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_sl_ultra"
    _c = _cfg(_n, (16, 28, 2), (0.05, 0.25, 0.05), (38, 52, 2), (1.20, 2.00, 0.20))
    log.info("A07  LL(16-28 s2)×SL(0.05-0.25 s0.05)×LS(38-52 s2)×SS(1.20-2.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  ll_short — short LL=8-18 (explore if shorter long period works on TXF)
    #      LL(8-18 s2)×SL(0.20-0.60 s0.10)×LS(36-52 s2)×SS(1.20-2.00 s0.20) = 6×5×9×5 = 1350
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_ll_short"
    _c = _cfg(_n, (8, 18, 2), (0.20, 0.60, 0.10), (36, 52, 2), (1.20, 2.00, 0.20))
    log.info("A08  LL(8-18 s2)×SL(0.20-0.60 s0.10)×LS(36-52 s2)×SS(1.20-2.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  global_r2 — global overview step=10/0.2/10/0.5
    #      LL(5-65 s10)×SL(0.20-2.00 s0.20)×LS(5-65 s10)×SS(0.50-2.50 s0.50) = 7×10×7×5 = 2450
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_global_r2"
    _c = _cfg(_n, (5, 65, 10), (0.20, 2.00, 0.20), (5, 65, 10), (0.50, 2.50, 0.50))
    log.info("A09  LL(5-65 s10)×SL(0.20-2.00 s0.20)×LS(5-65 s10)×SS(0.50-2.50 s0.50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — SL/SS step=0.05 (finer than R1 which used step=0.1)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (5, 0.30, 5, 0.30),
            (4, 0.25, 4, 0.25),
            (3, 0.20, 3, 0.20),
            (2, 0.15, 2, 0.15),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,  LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.05, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,  LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.05, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  finest_center — SL step=0.02, SS step=0.02 (extreme fine-tuning)
    #      LL(20-24 s1)×SL(0.32-0.48 s0.02)×LS(42-46 s1)×SS(1.70-1.90 s0.02) = 5×9×5×11 = 2475
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_finest_center"
    _c = _cfg(_n, (20, 24, 1), (0.32, 0.48, 0.02), (42, 46, 1), (1.70, 1.90, 0.02))
    log.info("A11  LL(20-24 s1)×SL(0.32-0.48 s0.02)×LS(42-46 s1)×SS(1.70-1.90 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  cross_regime — short LL+LS combo (check if any other regime hidden)
    #      LL(8-20 s2)×SL(0.10-0.50 s0.10)×LS(20-40 s4)×SS(1.00-2.00 s0.20) = 7×5×6×6 = 1260
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_cross_regime"
    _c = _cfg(_n, (8, 20, 2), (0.10, 0.50, 0.10), (20, 40, 4), (1.00, 2.00, 0.20))
    log.info("A12  LL(8-20 s2)×SL(0.10-0.50 s0.10)×LS(20-40 s4)×SS(1.00-2.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  TXF Hourly countertrend_LS Round-2 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R1 seed NP: %.0f → R2 best: %.0f  (Δ %.0f  +%.1f%%)",
             SEED_NP, best_np, best_np - SEED_NP, (best_np / SEED_NP - 1) * 100)
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
        description="TXF Hourly countertrend_LS NP>8M TWD Round-2 search")
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
