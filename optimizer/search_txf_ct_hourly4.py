"""
search_txf_ct_hourly4.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Hourly, Round 4

R1 best: LL=22, SL=0.40, LS=44, SS=1.800, NP=7,641,000 (7M MET ✅)
R2 best: LL=22, SL=0.42, LS=43, SS=1.760, NP=7,845,000 (+2.7%)
R3 best: LL=22, SL=0.43, LS=43, SS=1.765, NP=7,928,200 (+1.1%)
R4 target: NP > 8,000,000 TWD  (gap from R3: +71,800 TWD = +0.9%)

Key R3 insights:
- SL=0.43 > SL=0.42: confirmed across A01/A04/A07/A10
- SS=1.765 > SS=1.760 (found at step=0.005 in A04/A07); step=0.001 may reveal SS=1.763-1.767 peak
- LS=43 confirmed dominant over LS=42/44 at SL=0.43
- A11 showed LS=44 with SS=1.82 gives NP=7,401,800 (lower NP but better MDD)
- Gain rate collapsing: R1→R2 +204K, R2→R3 +83K; R4 targets squeeze + new regimes
- LL=23-30 unexplored at fine grain; LL>30 never tried; tight-SS regime (SS<1.5) never tried

R4 strategy:
  1. Ultra-fine SS (step=0.001) around confirmed peak SS=1.765
  2. SL step=0.005 (finer than R3's 0.01) + SS step=0.001 combined
  3. SS step=0.001 over broader range (1.750-1.785)
  4. Comprehensive fine grid: SL step=0.005, SS step=0.002
  5. LS broad scan with tight SL/SS (check if LS≠43 can win with finer params)
  6. LL=23-30 exploration (never carefully explored > LL=28)
  7. Tight SS regime: SS=0.60-1.50 (analogous to NQ SL=0.2 breakthrough — never tried for TXF)
  8. Large LL regime: LL=30-80 (completely unexplored territory)
  9. Global coarse with different grid from R3 (new perspective)
  10. Adaptive zoom with step=(1, 0.005, 1, 0.001) — finest dynamic probe
  11. SL step=0.005 comprehensive
  12. Final wide sweep

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 ss001_peak       : LL(21-23 s1)×SL(0.425-0.435 s0.005)×LS(42-44 s1)×SS(1.760-1.772 s0.001) = 3×3×3×13=351
  A02 sl_ss_ultrafine  : LL(21-23 s1)×SL(0.420-0.445 s0.005)×LS(42-44 s1)×SS(1.763-1.769 s0.001) = 3×6×3×7=378
  A03 ss001_wide       : LL(21-23 s1)×SL(0.42-0.44 s0.01)×LS(42-44 s1)×SS(1.750-1.785 s0.001)    = 3×3×3×36=972
  A04 all_ultrafine    : LL(20-24 s1)×SL(0.420-0.445 s0.005)×LS(41-45 s1)×SS(1.758-1.780 s0.002) = 5×6×5×12=1800
  A05 ls_fine_scan     : LL(21-23 s1)×SL(0.42-0.44 s0.01)×LS(38-50 s1)×SS(1.762-1.770 s0.002)    = 3×3×13×5=585
  A06 ll_extend        : LL(23-30 s1)×SL(0.38-0.48 s0.02)×LS(40-48 s2)×SS(1.720-1.820 s0.020)    = 8×6×5×6=1440
  A07 tight_ss_regime  : LL(18-26 s2)×SL(0.10-0.60 s0.10)×LS(38-58 s4)×SS(0.60-1.50 s0.10)       = 5×6×6×10=1800
  A08 large_ll_regime  : LL(30-80 s10)×SL(0.20-0.80 s0.15)×LS(30-70 s10)×SS(1.00-2.50 s0.50)     = 6×5×5×4=600
  A09 global_r4        : LL(5-55 s5)×SL(0.20-1.60 s0.40)×LS(10-60 s10)×SS(0.50-2.50 s0.50)       = 11×4×6×5=1320
  A10 adaptive_zoom    : (dynamic from best, SL step=0.005, SS step=0.001)
  A11 sl_fine_all      : LL(20-24 s1)×SL(0.415-0.450 s0.005)×LS(41-45 s1)×SS(1.758-1.782 s0.006) = 5×8×5×5=1000
  A12 final_sweep      : LL(20-24 s1)×SL(0.41-0.46 s0.01)×LS(40-46 s1)×SS(1.750-1.820 s0.010)    = 5×6×7×8=1680
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_hourly4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.1,  20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.1,  20.0

# R3 champion as seed
SEED_LL, SEED_SL = 22.0, 0.43
SEED_LS, SEED_SS = 43.0, 1.765
SEED_NP          = 7_928_200.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_hourly4_{int(time.time())}.log"
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
        name=f"TXFCT4_{name}",
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
    return OUTPUT_DIR / f"TXFCT4_{name}_raw.csv"


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
    log.info("=== Starting TXFCT4_%s (%d combos) ===", name, cfg.total_runs())
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
            "r1_best":  "LL=22 SL=0.40 LS=44 SS=1.800 NP=7641000",
            "r2_best":  "LL=22 SL=0.42 LS=43 SS=1.760 NP=7845000",
            "r3_best":  "LL=22 SL=0.43 LS=43 SS=1.765 NP=7928200",
            "r4_focus": "SL step=0.005, SS step=0.001; tight-SS regime; large-LL regime; adaptive zoom",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_hourly4.json"
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
    log.info("  TXF Hourly countertrend_LS NP>8,000,000 TWD — Round 4")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R3 best: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R4 focus: SL step=0.005, SS step=0.001; tight-SS; large-LL; adaptive zoom")
    log.info("  Target: %.0f TWD  (gap from R3: +%.0f = +%.1f%%)",
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
    # A01  ss001_peak — SS step=0.001 around confirmed peak SS=1.765
    #      LL(21-23 s1)×SL(0.425-0.435 s0.005)×LS(42-44 s1)×SS(1.760-1.772 s0.001) = 3×3×3×13=351
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ss001_peak"
    _c = _cfg(_n, (21, 23, 1), (0.425, 0.435, 0.005), (42, 44, 1), (1.760, 1.772, 0.001))
    log.info("A01  LL(21-23 s1)×SL(0.425-0.435 s0.005)×LS(42-44 s1)×SS(1.760-1.772 s0.001)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  sl_ss_ultrafine — SL step=0.005, SS step=0.001 (finest combined)
    #      LL(21-23 s1)×SL(0.420-0.445 s0.005)×LS(42-44 s1)×SS(1.763-1.769 s0.001) = 3×6×3×7=378
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_sl_ss_ultrafine"
    _c = _cfg(_n, (21, 23, 1), (0.420, 0.445, 0.005), (42, 44, 1), (1.763, 1.769, 0.001))
    log.info("A02  LL(21-23 s1)×SL(0.420-0.445 s0.005)×LS(42-44 s1)×SS(1.763-1.769 s0.001)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ss001_wide — SS step=0.001 over broader SS range
    #      LL(21-23 s1)×SL(0.42-0.44 s0.01)×LS(42-44 s1)×SS(1.750-1.785 s0.001) = 3×3×3×36=972
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ss001_wide"
    _c = _cfg(_n, (21, 23, 1), (0.42, 0.44, 0.01), (42, 44, 1), (1.750, 1.785, 0.001))
    log.info("A03  LL(21-23 s1)×SL(0.42-0.44 s0.01)×LS(42-44 s1)×SS(1.750-1.785 s0.001)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  all_ultrafine — SL step=0.005, SS step=0.002, LL/LS expanded
    #      LL(20-24 s1)×SL(0.420-0.445 s0.005)×LS(41-45 s1)×SS(1.758-1.780 s0.002) = 5×6×5×12=1800
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_all_ultrafine"
    _c = _cfg(_n, (20, 24, 1), (0.420, 0.445, 0.005), (41, 45, 1), (1.758, 1.780, 0.002))
    log.info("A04  LL(20-24 s1)×SL(0.420-0.445 s0.005)×LS(41-45 s1)×SS(1.758-1.780 s0.002)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ls_fine_scan — LS=38-50 full scan with tight SL/SS
    #      LL(21-23 s1)×SL(0.42-0.44 s0.01)×LS(38-50 s1)×SS(1.762-1.770 s0.002) = 3×3×13×5=585
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ls_fine_scan"
    _c = _cfg(_n, (21, 23, 1), (0.42, 0.44, 0.01), (38, 50, 1), (1.762, 1.770, 0.002))
    log.info("A05  LL(21-23 s1)×SL(0.42-0.44 s0.01)×LS(38-50 s1)×SS(1.762-1.770 s0.002)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ll_extend — LL=23-30 (never carefully explored > LL=28 in R3)
    #      LL(23-30 s1)×SL(0.38-0.48 s0.02)×LS(40-48 s2)×SS(1.720-1.820 s0.020) = 8×6×5×6=1440
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ll_extend"
    _c = _cfg(_n, (23, 30, 1), (0.38, 0.48, 0.02), (40, 48, 2), (1.720, 1.820, 0.020))
    log.info("A06  LL(23-30 s1)×SL(0.38-0.48 s0.02)×LS(40-48 s2)×SS(1.720-1.820 s0.020)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  tight_ss_regime — SS=0.60-1.50 (analogous to NQ SL=0.2 breakthrough)
    #      Never tried tight SS<1.5 on TXF; NQ breakthrough was SS=1.4
    #      LL(18-26 s2)×SL(0.10-0.60 s0.10)×LS(38-58 s4)×SS(0.60-1.50 s0.10) = 5×6×6×10=1800
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_tight_ss_regime"
    _c = _cfg(_n, (18, 26, 2), (0.10, 0.60, 0.10), (38, 58, 4), (0.60, 1.50, 0.10))
    log.info("A07  LL(18-26 s2)×SL(0.10-0.60 s0.10)×LS(38-58 s4)×SS(0.60-1.50 s0.10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  large_ll_regime — LL=30-80 (completely unexplored territory)
    #      LL(30-80 s10)×SL(0.20-0.80 s0.15)×LS(30-70 s10)×SS(1.00-2.50 s0.50) = 6×5×5×4=600
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_large_ll_regime"
    _c = _cfg(_n, (30, 80, 10), (0.20, 0.80, 0.15), (30, 70, 10), (1.00, 2.50, 0.50))
    log.info("A08  LL(30-80 s10)×SL(0.20-0.80 s0.15)×LS(30-70 s10)×SS(1.00-2.50 s0.50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  global_r4 — Coarse global with different grid (new perspective vs R3)
    #      LL(5-55 s5)×SL(0.20-1.60 s0.40)×LS(10-60 s10)×SS(0.50-2.50 s0.50) = 11×4×6×5=1320
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_global_r4"
    _c = _cfg(_n, (5, 55, 5), (0.20, 1.60, 0.40), (10, 60, 10), (0.50, 2.50, 0.50))
    log.info("A09  LL(5-55 s5)×SL(0.20-1.60 s0.40)×LS(10-60 s10)×SS(0.50-2.50 s0.50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — SL step=0.005, SS step=0.001 (finest dynamic probe)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.06, 2, 0.012),
            (2, 0.05, 2, 0.010),
            (2, 0.04, 2, 0.008),
            (2, 0.03, 2, 0.006),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.005, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.001, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  sl_fine_all — SL step=0.005, SS step=0.006 comprehensive
    #      LL(20-24 s1)×SL(0.415-0.450 s0.005)×LS(41-45 s1)×SS(1.758-1.782 s0.006) = 5×8×5×5=1000
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_sl_fine_all"
    _c = _cfg(_n, (20, 24, 1), (0.415, 0.450, 0.005), (41, 45, 1), (1.758, 1.782, 0.006))
    log.info("A11  LL(20-24 s1)×SL(0.415-0.450 s0.005)×LS(41-45 s1)×SS(1.758-1.782 s0.006)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  final_sweep — Wide sweep around champion with all params
    #      LL(20-24 s1)×SL(0.41-0.46 s0.01)×LS(40-46 s1)×SS(1.750-1.820 s0.010) = 5×6×7×8=1680
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_final_sweep"
    _c = _cfg(_n, (20, 24, 1), (0.41, 0.46, 0.01), (40, 46, 1), (1.750, 1.820, 0.010))
    log.info("A12  LL(20-24 s1)×SL(0.41-0.46 s0.01)×LS(40-46 s1)×SS(1.750-1.820 s0.010)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  TXF Hourly countertrend_LS Round-4 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R3 seed NP: %.0f → R4 best: %.0f  (Δ %.0f  +%.1f%%)",
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
        description="TXF Hourly countertrend_LS NP>8M TWD Round-4 search")
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
