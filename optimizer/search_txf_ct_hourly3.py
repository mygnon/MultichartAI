"""
search_txf_ct_hourly3.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Hourly, Round 3

R1 best: LL=22, SL=0.40, LS=44, SS=1.80, NP=7,641,000 (7M MET ✅)
R2 best: LL=22, SL=0.42, LS=43, SS=1.76, NP=7,845,000 (+2.7%)
R3 target: NP > 8,000,000 TWD  (gap from R2: +155,000 TWD = +2.0%)

Key R2 insights:
- SL=0.42 > SL=0.40: A03 found LL=22,SL=0.42,LS=44,SS=1.8 → 7,773K vs 7,641K (+132K)
- LS=43 > LS=44 (when SL=0.42, SS=1.76): A11 found LS=43,SS=1.76 → 7,845K
- SS=1.76 is better than SS=1.75 or SS=1.80 at this LS=43 regime
- A11 used SL step=0.02, SS step=0.02 — R3 needs step=0.01 to find true peak
- SL=0.4 at LS=43,SS=1.75 gives 7,654K — interaction between SL/LS/SS is precise
- LL=24, SL=0.2 secondary regime exists (7,571K) — worth fine-tuning
- SS>2.0 confirmed worse (6,747K) — upper bound set
- A10 adaptive_zoom confirmed: best in range is LS=43,SS=1.75,SL=0.40 at step=0.05
  → actual peak SS=1.76,SL=0.42 found only at step=0.02 (A11)
  → R3 step=0.01 to pin exact peak

R3 strategy:
  1. SL/SS step=0.01 around champion (LS=43 regime)
  2. SL/SS step=0.01 around LS=44 regime (R2 A03 peak was 7,773K)
  3. Broad SL step=0.01 (range 0.36-0.50)
  4. SS step=0.005 finest around peak (1.700-1.820)
  5. LL=24/SL=0.2 regime fine-tuning
  6. LL range scan (LL=21-28) at fine grain
  7. Ultimate finest: SL+SS step=0.005 (extreme precision)
  8. LS broad step=1 (check LS=38-50)
  9. Global overview (sanity check)
  10. Adaptive zoom step=(1,0.01,1,0.01)
  11. All-fine combined (SL/SS step=0.01, LS=41-47)
  12. LS=42 regime (step=1 scan, fine SL/SS)

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 sl_ss01_ls43  : LL(21-23 s1)×SL(0.38-0.48 s0.01)×LS(41-45 s1)×SS(1.720-1.820 s0.010) = 3×11×5×11=1815
  A02 sl_ss01_ls44  : LL(21-23 s1)×SL(0.38-0.48 s0.01)×LS(42-46 s1)×SS(1.720-1.860 s0.010) = 3×11×5×15=2475
  A03 sl01_broad    : LL(20-26 s1)×SL(0.36-0.50 s0.01)×LS(42-45 s1)×SS(1.740-1.820 s0.020) = 7×15×4×5=2100
  A04 ss005_fine    : LL(20-24 s1)×SL(0.40-0.45 s0.01)×LS(41-46 s1)×SS(1.700-1.820 s0.005) = 5×6×6×25=4500
  A05 ll24_regime   : LL(22-28 s1)×SL(0.14-0.26 s0.02)×LS(40-50 s2)×SS(1.600-2.000 s0.100) = 7×7×6×5=1470
  A06 ll_scan_fine  : LL(21-28 s1)×SL(0.38-0.46 s0.02)×LS(41-46 s1)×SS(1.700-1.840 s0.020) = 8×5×6×8=1920
  A07 ultimate_fine : LL(21-23 s1)×SL(0.40-0.46 s0.01)×LS(41-45 s1)×SS(1.720-1.800 s0.005) = 3×7×5×17=1785
  A08 ls_broad      : LL(21-23 s1)×SL(0.40-0.44 s0.01)×LS(38-50 s1)×SS(1.700-1.840 s0.020) = 3×5×13×8=1560
  A09 global_r3     : LL(5-65 s10)×SL(0.20-2.00 s0.20)×LS(5-65 s10)×SS(0.50-2.50 s0.50)   = 7×10×7×5=2450
  A10 adaptive_zoom : (dynamic from R3 best, step=0.01 for SL/SS)
  A11 all_fine      : LL(20-24 s1)×SL(0.38-0.48 s0.01)×LS(41-47 s1)×SS(1.720-1.820 s0.010) = 5×11×7×11=4235
  A12 ls42_regime   : LL(20-24 s1)×SL(0.38-0.48 s0.01)×LS(40-45 s1)×SS(1.700-1.840 s0.020) = 5×11×6×8=2640
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_hourly3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.1,  20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.1,  20.0

# R2 champion as seed
SEED_LL, SEED_SL = 22.0, 0.42
SEED_LS, SEED_SS = 43.0, 1.76
SEED_NP          = 7_845_000.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_hourly3_{int(time.time())}.log"
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
        name=f"TXFCT3_{name}",
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
    return OUTPUT_DIR / f"TXFCT3_{name}_raw.csv"


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
    log.info("=== Starting TXFCT3_%s (%d combos) ===", name, cfg.total_runs())
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
        "timeframe": "hourly",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_best":  "LL=22 SL=0.40 LS=44 SS=1.80 NP=7641000",
            "r2_best":  "LL=22 SL=0.42 LS=43 SS=1.76 NP=7845000",
            "r3_focus": "SL/SS step=0.01; SS step=0.005 extreme fine; LL=24/SL=0.2 regime; adaptive zoom",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_hourly3.json"
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
    log.info("  TXF Hourly countertrend_LS NP>8,000,000 TWD — Round 3")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R2 best: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R3 focus: SL/SS step=0.01; SS step=0.005; adaptive zoom IME-fixed")
    log.info("  Target: %.0f TWD  (gap from R2: +%.0f = +%.1f%%)",
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
    # A01  sl_ss01_ls43 — SL/SS step=0.01 around LS=43 regime
    #      LL(21-23 s1)×SL(0.38-0.48 s0.01)×LS(41-45 s1)×SS(1.720-1.820 s0.010) = 3×11×5×11=1815
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_sl_ss01_ls43"
    _c = _cfg(_n, (21, 23, 1), (0.38, 0.48, 0.01), (41, 45, 1), (1.720, 1.820, 0.010))
    log.info("A01  LL(21-23 s1)×SL(0.38-0.48 s0.01)×LS(41-45 s1)×SS(1.720-1.820 s0.010)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  sl_ss01_ls44 — SL/SS step=0.01 around LS=44 regime (R2 A03 gave 7,773K)
    #      LL(21-23 s1)×SL(0.38-0.48 s0.01)×LS(42-46 s1)×SS(1.720-1.860 s0.010) = 3×11×5×15=2475
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_sl_ss01_ls44"
    _c = _cfg(_n, (21, 23, 1), (0.38, 0.48, 0.01), (42, 46, 1), (1.720, 1.860, 0.010))
    log.info("A02  LL(21-23 s1)×SL(0.38-0.48 s0.01)×LS(42-46 s1)×SS(1.720-1.860 s0.010)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  sl01_broad — SL step=0.01 broader LL range
    #      LL(20-26 s1)×SL(0.36-0.50 s0.01)×LS(42-45 s1)×SS(1.740-1.820 s0.020) = 7×15×4×5=2100
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_sl01_broad"
    _c = _cfg(_n, (20, 26, 1), (0.36, 0.50, 0.01), (42, 45, 1), (1.740, 1.820, 0.020))
    log.info("A03  LL(20-26 s1)×SL(0.36-0.50 s0.01)×LS(42-45 s1)×SS(1.740-1.820 s0.020)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ss005_fine — SS step=0.005 (finest SS scan around peak)
    #      LL(20-24 s1)×SL(0.40-0.45 s0.01)×LS(41-46 s1)×SS(1.700-1.820 s0.005) = 5×6×6×25=4500
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ss005_fine"
    _c = _cfg(_n, (20, 24, 1), (0.40, 0.45, 0.01), (41, 46, 1), (1.700, 1.820, 0.005))
    log.info("A04  LL(20-24 s1)×SL(0.40-0.45 s0.01)×LS(41-46 s1)×SS(1.700-1.820 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ll24_regime — fine-tune LL=24/SL=0.2 secondary regime
    #      LL(22-28 s1)×SL(0.14-0.26 s0.02)×LS(40-50 s2)×SS(1.600-2.000 s0.100) = 7×7×6×5=1470
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ll24_regime"
    _c = _cfg(_n, (22, 28, 1), (0.14, 0.26, 0.02), (40, 50, 2), (1.600, 2.000, 0.100))
    log.info("A05  LL(22-28 s1)×SL(0.14-0.26 s0.02)×LS(40-50 s2)×SS(1.600-2.000 s0.100)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ll_scan_fine — check LL=21-28 range at fine SL/SS
    #      LL(21-28 s1)×SL(0.38-0.46 s0.02)×LS(41-46 s1)×SS(1.700-1.840 s0.020) = 8×5×6×8=1920
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ll_scan_fine"
    _c = _cfg(_n, (21, 28, 1), (0.38, 0.46, 0.02), (41, 46, 1), (1.700, 1.840, 0.020))
    log.info("A06  LL(21-28 s1)×SL(0.38-0.46 s0.02)×LS(41-46 s1)×SS(1.700-1.840 s0.020)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  ultimate_fine — SL step=0.01, SS step=0.005 (extreme precision)
    #      LL(21-23 s1)×SL(0.40-0.46 s0.01)×LS(41-45 s1)×SS(1.720-1.800 s0.005) = 3×7×5×17=1785
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ultimate_fine"
    _c = _cfg(_n, (21, 23, 1), (0.40, 0.46, 0.01), (41, 45, 1), (1.720, 1.800, 0.005))
    log.info("A07  LL(21-23 s1)×SL(0.40-0.46 s0.01)×LS(41-45 s1)×SS(1.720-1.800 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  ls_broad — LS step=1 broader scan (check LS=38-50)
    #      LL(21-23 s1)×SL(0.40-0.44 s0.01)×LS(38-50 s1)×SS(1.700-1.840 s0.020) = 3×5×13×8=1560
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_ls_broad"
    _c = _cfg(_n, (21, 23, 1), (0.40, 0.44, 0.01), (38, 50, 1), (1.700, 1.840, 0.020))
    log.info("A08  LL(21-23 s1)×SL(0.40-0.44 s0.01)×LS(38-50 s1)×SS(1.700-1.840 s0.020)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  global_r3 — global overview (sanity check; keep perspective)
    #      LL(5-65 s10)×SL(0.20-2.00 s0.20)×LS(5-65 s10)×SS(0.50-2.50 s0.50) = 7×10×7×5=2450
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_global_r3"
    _c = _cfg(_n, (5, 65, 10), (0.20, 2.00, 0.20), (5, 65, 10), (0.50, 2.50, 0.50))
    log.info("A09  LL(5-65 s10)×SL(0.20-2.00 s0.20)×LS(5-65 s10)×SS(0.50-2.50 s0.50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — SL/SS step=0.01 (finer than R2 which used step=0.05)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (4, 0.10, 4, 0.10),
            (3, 0.08, 3, 0.08),
            (2, 0.06, 2, 0.06),
            (2, 0.05, 2, 0.05),
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
    # A11  all_fine — SL/SS step=0.01, LS=41-47 (comprehensive fine grid)
    #      LL(20-24 s1)×SL(0.38-0.48 s0.01)×LS(41-47 s1)×SS(1.720-1.820 s0.010) = 5×11×7×11=4235
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_all_fine"
    _c = _cfg(_n, (20, 24, 1), (0.38, 0.48, 0.01), (41, 47, 1), (1.720, 1.820, 0.010))
    log.info("A11  LL(20-24 s1)×SL(0.38-0.48 s0.01)×LS(41-47 s1)×SS(1.720-1.820 s0.010)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  ls42_regime — LS=40-45 with fine SL/SS (check LS=42 potential)
    #      LL(20-24 s1)×SL(0.38-0.48 s0.01)×LS(40-45 s1)×SS(1.700-1.840 s0.020) = 5×11×6×8=2640
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_ls42_regime"
    _c = _cfg(_n, (20, 24, 1), (0.38, 0.48, 0.01), (40, 45, 1), (1.700, 1.840, 0.020))
    log.info("A12  LL(20-24 s1)×SL(0.38-0.48 s0.01)×LS(40-45 s1)×SS(1.700-1.840 s0.020)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  TXF Hourly countertrend_LS Round-3 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R2 seed NP: %.0f → R3 best: %.0f  (Δ %.0f  +%.1f%%)",
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
        description="TXF Hourly countertrend_LS NP>8M TWD Round-3 search")
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
