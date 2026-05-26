"""
search_txf_ct_hourly6.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Hourly, Round 6

R1 best: LL=22, SL=0.400, LS=44, SS=1.800, NP=7,641,000 (7M MET ✅)
R2 best: LL=22, SL=0.420, LS=43, SS=1.760, NP=7,845,000 (+2.7%)
R3 best: LL=22, SL=0.430, LS=43, SS=1.765, NP=7,928,200 (+1.1%)
R4 best: LL=22, SL=0.425, LS=43, SS=1.771, NP=8,101,400 (8M MET ✅)
R5 best: LL=22, SL=0.425, LS=43, SS=1.765, NP=8,042,600 (data-shifted; 9M NOT met)
R6 target: NP > 9,000,000 TWD  (gap from R4: +898,600 TWD = +11.1%)

Key R5 discoveries — two new high-Obj regimes:
  (A) LS=36 low-drawdown regime (A03 ls_short):
      LL=22, SL=0.40, LS=36, SS=1.40, NP=7,131,800, MDD=-609,400, Obj=83,463,359 ← RECORD Obj!
      Scanned at coarse step: LS step=4, SS step=0.20 → true peak unknown, NP can be higher
  (B) SS=0.80 extreme-tight regime (A06 ss_extreme_tight):
      LL=22, SL=0.40, LS=45, SS=0.80, NP=6,425,800, MDD=-563,800, Obj=73,236,796

R6 strategy — deep exploration of newly discovered low-drawdown regimes:
  1. Fine LS=33-39 around regime (A) — step=1, SS step=0.02
  2. Ultra-fine SL/SS around LS=36 peak — step=0.01
  3. Broader SL scan in LS=36 regime
  4. SS=0.80 regime fine-tune (regime B)
  5. SS transition 0.80-1.40 bridge
  6. LS=34-42 with SS=1.10-1.60 (full A/B bridge zone)
  7. Specific zoom around LL=22/SL=0.40/LS=36/SS=1.40 discovery
  8. Global fresh scan (different grid)
  9. LS=34-42, SS=1.40-1.70 (check if LS=36 regime extends to SS>1.4)
  10. Adaptive zoom step=(1,0.02,1,0.02) from best
  11. LL=22 LS=35-41 comprehensive step=1
  12. Main regime verify + push with current data

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 ls36_ss_fine  : LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(33-39 s1)×SS(1.25-1.55 s0.02) = 5×6×7×16=3360
  A02 ls36_ultra    : LL(20-24 s1)×SL(0.38-0.44 s0.01)×LS(34-40 s2)×SS(1.34-1.50 s0.01) = 5×7×4×17=2380
  A03 ls36_sl_broad : LL(20-24 s1)×SL(0.30-0.55 s0.05)×LS(33-39 s1)×SS(1.30-1.52 s0.02) = 5×6×7×12=2520
  A04 ss08_fine     : LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(40-55 s3)×SS(0.60-1.00 s0.05) = 5×4×6×9=1080
  A05 ss_transition : LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(36-50 s2)×SS(0.80-1.40 s0.10) = 5×4×8×7=1120
  A06 ls36_full_ss  : LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(34-42 s2)×SS(1.10-1.60 s0.05) = 5×6×5×11=1650
  A07 ls36_zoomed   : LL(21-23 s1)×SL(0.36-0.46 s0.01)×LS(33-39 s1)×SS(1.34-1.50 s0.02) = 3×11×7×9=2079
  A08 global_r6     : LL(5-65 s5)×SL(0.10-0.60 s0.10)×LS(5-65 s10)×SS(0.50-2.00 s0.50)  = 13×6×7×4=2184
  A09 ls36_ss_ext   : LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(34-42 s2)×SS(1.40-1.70 s0.05) = 5×6×5×7=1050
  A10 adaptive_zoom : (dynamic from best, SL step=0.02, SS step=0.02)
  A11 ls35_41_combo : LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(35-41 s1)×SS(1.30-1.56 s0.04) = 5×6×7×8=1680
  A12 verify_main   : LL(20-24 s1)×SL(0.415-0.440 s0.005)×LS(41-46 s1)×SS(1.755-1.815 s0.005) = 5×6×6×13=2340
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_hourly6_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 9_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.1,  20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.1,  20.0

# R4 NP-champion as seed (highest confirmed NP)
SEED_LL, SEED_SL = 22.0, 0.425
SEED_LS, SEED_SS = 43.0, 1.771
SEED_NP          = 8_101_400.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_hourly6_{int(time.time())}.log"
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
        name=f"TXFCT6_{name}",
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
    return OUTPUT_DIR / f"TXFCT6_{name}_raw.csv"


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
    log.info("=== Starting TXFCT6_%s (%d combos) ===", name, cfg.total_runs())
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
        "round": 6,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "hourly",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r4_best":  "LL=22 SL=0.425 LS=43 SS=1.771 NP=8101400 (8M MET)",
            "r5_disco": "LS=36/SS=1.40 Obj=83M (MDD=-609K); SS=0.80/LS=45 Obj=73M (MDD=-564K)",
            "r6_focus": "Deep explore LS=36/SS=1.40 low-drawdown regime; SS=0.80 regime; bridge zone",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_hourly6.json"
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
    log.info("  TXF Hourly countertrend_LS NP>9,000,000 TWD — Round 6")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R4 best: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R5 discovery: LS=36/SS=1.40 Obj=83M (MDD=-609K); SS=0.80/LS=45 Obj=73M")
    log.info("  R6 focus: deep explore LS=36 low-drawdown regime; SS=0.80 regime; bridge")
    log.info("  Target: %.0f TWD  (gap from R4: +%.0f = +%.1f%%)",
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
                 "★TARGET★" if met else ("%.0f/9M" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, max(0, TARGET_NP - best_np))

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  ls36_ss_fine — LS=33-39 step=1, SS step=0.02 (R5 A03 found LS=36 at coarse step=4)
    #      LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(33-39 s1)×SS(1.25-1.55 s0.02) = 5×6×7×16=3360
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ls36_ss_fine"
    _c = _cfg(_n, (20, 24, 1), (0.36, 0.46, 0.02), (33, 39, 1), (1.25, 1.55, 0.02))
    log.info("A01  LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(33-39 s1)×SS(1.25-1.55 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  ls36_ultra — SL/SS step=0.01, LS=34-40 step=2 (ultra-fine around LS=36 peak)
    #      LL(20-24 s1)×SL(0.38-0.44 s0.01)×LS(34-40 s2)×SS(1.34-1.50 s0.01) = 5×7×4×17=2380
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ls36_ultra"
    _c = _cfg(_n, (20, 24, 1), (0.38, 0.44, 0.01), (34, 40, 2), (1.34, 1.50, 0.01))
    log.info("A02  LL(20-24 s1)×SL(0.38-0.44 s0.01)×LS(34-40 s2)×SS(1.34-1.50 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ls36_sl_broad — Broader SL scan at LS=33-39 to find SL peak
    #      LL(20-24 s1)×SL(0.30-0.55 s0.05)×LS(33-39 s1)×SS(1.30-1.52 s0.02) = 5×6×7×12=2520
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ls36_sl_broad"
    _c = _cfg(_n, (20, 24, 1), (0.30, 0.55, 0.05), (33, 39, 1), (1.30, 1.52, 0.02))
    log.info("A03  LL(20-24 s1)×SL(0.30-0.55 s0.05)×LS(33-39 s1)×SS(1.30-1.52 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ss08_fine — SS=0.80 extreme-tight regime fine-tune (R5 A06: Obj=73M)
    #      LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(40-55 s3)×SS(0.60-1.00 s0.05) = 5×4×6×9=1080
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ss08_fine"
    _c = _cfg(_n, (20, 24, 1), (0.35, 0.50, 0.05), (40, 55, 3), (0.60, 1.00, 0.05))
    log.info("A04  LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(40-55 s3)×SS(0.60-1.00 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ss_transition — SS=0.80-1.40 bridge between the two new regimes
    #      LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(36-50 s2)×SS(0.80-1.40 s0.10) = 5×4×8×7=1120
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ss_transition"
    _c = _cfg(_n, (20, 24, 1), (0.35, 0.50, 0.05), (36, 50, 2), (0.80, 1.40, 0.10))
    log.info("A05  LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(36-50 s2)×SS(0.80-1.40 s0.10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ls36_full_ss — LS=34-42, SS=1.10-1.60 (comprehensive bridge A/B zone)
    #      LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(34-42 s2)×SS(1.10-1.60 s0.05) = 5×6×5×11=1650
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ls36_full_ss"
    _c = _cfg(_n, (20, 24, 1), (0.36, 0.46, 0.02), (34, 42, 2), (1.10, 1.60, 0.05))
    log.info("A06  LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(34-42 s2)×SS(1.10-1.60 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  ls36_zoomed — SL step=0.01, SS step=0.02 zoom around LS=36/SS=1.40 discovery
    #      LL(21-23 s1)×SL(0.36-0.46 s0.01)×LS(33-39 s1)×SS(1.34-1.50 s0.02) = 3×11×7×9=2079
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ls36_zoomed"
    _c = _cfg(_n, (21, 23, 1), (0.36, 0.46, 0.01), (33, 39, 1), (1.34, 1.50, 0.02))
    log.info("A07  LL(21-23 s1)×SL(0.36-0.46 s0.01)×LS(33-39 s1)×SS(1.34-1.50 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  global_r6 — Fresh global with finer LL step (catches LL=10,15,20,25...)
    #      LL(5-65 s5)×SL(0.10-0.60 s0.10)×LS(5-65 s10)×SS(0.50-2.00 s0.50) = 13×6×7×4=2184
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_global_r6"
    _c = _cfg(_n, (5, 65, 5), (0.10, 0.60, 0.10), (5, 65, 10), (0.50, 2.00, 0.50))
    log.info("A08  LL(5-65 s5)×SL(0.10-0.60 s0.10)×LS(5-65 s10)×SS(0.50-2.00 s0.50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  ls36_ss_ext — Check LS=34-42 at SS=1.40-1.70 (does LS=36 peak extend higher SS?)
    #      LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(34-42 s2)×SS(1.40-1.70 s0.05) = 5×6×5×7=1050
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_ls36_ss_ext"
    _c = _cfg(_n, (20, 24, 1), (0.36, 0.46, 0.02), (34, 42, 2), (1.40, 1.70, 0.05))
    log.info("A09  LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(34-42 s2)×SS(1.40-1.70 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — SL step=0.02, SS step=0.02 (wider than R4's ultra-fine)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (3, 0.08, 3, 0.10),
            (2, 0.08, 2, 0.08),
            (2, 0.06, 2, 0.06),
            (2, 0.04, 2, 0.04),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.02,  SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.02,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  ls35_41_combo — LS=35-41 step=1, SS step=0.04 (bridge A/B with step=1 LS)
    #      LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(35-41 s1)×SS(1.30-1.56 s0.04) = 5×6×7×8=1680
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_ls35_41_combo"
    _c = _cfg(_n, (20, 24, 1), (0.36, 0.46, 0.02), (35, 41, 1), (1.30, 1.56, 0.04))
    log.info("A11  LL(20-24 s1)×SL(0.36-0.46 s0.02)×LS(35-41 s1)×SS(1.30-1.56 s0.04)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  verify_main — Re-verify main regime champion with current data
    #      LL(20-24 s1)×SL(0.415-0.440 s0.005)×LS(41-46 s1)×SS(1.755-1.815 s0.005) = 5×6×6×13=2340
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_verify_main"
    _c = _cfg(_n, (20, 24, 1), (0.415, 0.440, 0.005), (41, 46, 1), (1.755, 1.815, 0.005))
    log.info("A12  LL(20-24 s1)×SL(0.415-0.440 s0.005)×LS(41-46 s1)×SS(1.755-1.815 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  TXF Hourly countertrend_LS Round-6 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R4 seed NP: %.0f → R6 best: %.0f  (Δ %.0f  +%.1f%%)",
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
    print(f"Target NP>9M TWD: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
        description="TXF Hourly countertrend_LS NP>9M TWD Round-6 search")
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
