"""
search_txf_ct_hourly5.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Hourly, Round 5

R1 best: LL=22, SL=0.400, LS=44, SS=1.800, NP=7,641,000 (7M MET ✅)
R2 best: LL=22, SL=0.420, LS=43, SS=1.760, NP=7,845,000 (+2.7%)
R3 best: LL=22, SL=0.430, LS=43, SS=1.765, NP=7,928,200 (+1.1%)
R4 best: LL=22, SL=0.425, LS=43, SS=1.771, NP=8,101,400 (8M MET ✅ +2.2%)
R5 target: NP > 9,000,000 TWD  (gap from R4: +898,600 TWD = +11.1%)

Key R4 insights:
- SL=0.425 is the true peak (between R2's 0.42 and R3's 0.43 — found only at step=0.005)
- Two valid 8M combos: LS=43/SS=1.771 (NP-best) and LS=44/SS=1.758 (Obj-best)
- A07 tight-SS regime DISCOVERY: LL=22, SL=0.40, LS=38, SS=1.50 → Obj=54,280,002 (record!)
  but NP=7,128,600 at COARSE step=0.10 — very low MDD=-936,200; true peak unexplored
- LL>22 worse (LL=23-30 topped at 7.12M); LL=30-80 far worse (4.56M)
- Main LL=22 regime micro-tuning unlikely to yield +11% to reach 9M

R5 strategy — new territory over micro-fine tuning:
  1. Fine-tune LS=34-44 low-drawdown regime from A07 (coarse step → fine step)
  2. LS=35-43 at SS=1.4-1.9 fine (the A07 regime with finer resolution)
  3. Very short LS=20-36 (never explored below 38)
  4. Long LS=50-90 (never carefully explored above 50)
  5. Ultra-tight SL=0.05-0.25 (like NQ's SL=0.2 breakthrough — never tried on TXF)
  6. Extremely tight SS=0.20-0.80 (counter-trend with very tight short entry — never tried)
  7. Low LL regime LL=12-20 (never carefully explored below 18)
  8. Medium SL=0.5-1.2 (gap between tight and moderate, poorly explored)
  9. Global coarse with new grid (different perspective)
  10. Adaptive zoom step=(1,0.05,1,0.05) — broader than R4's fine steps
  11. LS=35-43 fine scan of the tight-SS regime
  12. Main regime push with expanded params

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 ls38_broad      : LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(34-44 s2)×SS(1.20-1.90 s0.10) = 5×4×6×8=960
  A02 ls38_fine       : LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(34-44 s2)×SS(1.40-1.80 s0.05) = 5×4×6×9=1080
  A03 ls_short        : LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(20-36 s4)×SS(1.20-2.00 s0.20) = 5×6×5×5=750
  A04 ls_long         : LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(50-90 s10)×SS(1.30-2.50 s0.30) = 5×6×5×5=750
  A05 sl_ultra_tight  : LL(18-26 s2)×SL(0.05-0.25 s0.05)×LS(35-55 s5)×SS(1.20-2.00 s0.20) = 5×5×5×5=625
  A06 ss_extreme_tight: LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(30-60 s5)×SS(0.20-0.80 s0.10) = 5×6×7×7=1470
  A07 ll_low_regime   : LL(12-20 s2)×SL(0.30-0.60 s0.05)×LS(35-55 s5)×SS(1.40-2.00 s0.20) = 5×7×5×4=700
  A08 sl_medium       : LL(18-26 s2)×SL(0.50-1.20 s0.10)×LS(35-55 s5)×SS(1.20-2.20 s0.20) = 5×8×5×6=1200
  A09 global_r5       : LL(5-65 s10)×SL(0.10-1.90 s0.30)×LS(5-65 s10)×SS(0.40-2.50 s0.35) = 7×7×7×7=2401
  A10 adaptive_zoom   : (dynamic from best, SL step=0.05, SS step=0.05 — broader exploration)
  A11 ls38_vfine      : LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(35-43 s1)×SS(1.40-1.90 s0.05) = 5×4×9×11=1980
  A12 main_regime_push: LL(20-24 s1)×SL(0.415-0.440 s0.005)×LS(41-46 s1)×SS(1.755-1.815 s0.005) = 5×6×6×13=2340
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_hourly5_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 9_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.1,  20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.1,  20.0

# R4 NP-champion as seed
SEED_LL, SEED_SL = 22.0, 0.425
SEED_LS, SEED_SS = 43.0, 1.771
SEED_NP          = 8_101_400.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_hourly5_{int(time.time())}.log"
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
        name=f"TXFCT5_{name}",
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
    return OUTPUT_DIR / f"TXFCT5_{name}_raw.csv"


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
    log.info("=== Starting TXFCT5_%s (%d combos) ===", name, cfg.total_runs())
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
        "round": 5,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "hourly",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r1_best":  "LL=22 SL=0.400 LS=44 SS=1.800 NP=7641000",
            "r2_best":  "LL=22 SL=0.420 LS=43 SS=1.760 NP=7845000",
            "r3_best":  "LL=22 SL=0.430 LS=43 SS=1.765 NP=7928200",
            "r4_best":  "LL=22 SL=0.425 LS=43 SS=1.771 NP=8101400 (8M MET)",
            "r5_focus": "New territory: LS=38 regime fine-tune, LS<38, LS>50, SL<0.25, SS<0.8, LL<18",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_hourly5.json"
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
    log.info("  TXF Hourly countertrend_LS NP>9,000,000 TWD — Round 5")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R4 best: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R5 focus: new territory exploration — LS=38 regime, LS<38, LS>50, SL<0.25, SS<0.8, LL<18")
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
    # A01  ls38_broad — LS=34-44 low-drawdown regime (R4 A07 best: LS=38 at boundary)
    #      Coarse scan to map the full LS=34-44, SS=1.2-1.9 landscape
    #      LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(34-44 s2)×SS(1.20-1.90 s0.10) = 5×4×6×8=960
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ls38_broad"
    _c = _cfg(_n, (20, 24, 1), (0.35, 0.50, 0.05), (34, 44, 2), (1.20, 1.90, 0.10))
    log.info("A01  LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(34-44 s2)×SS(1.20-1.90 s0.10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  ls38_fine — LS=34-44, finer SS step around 1.40-1.80
    #      LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(34-44 s2)×SS(1.40-1.80 s0.05) = 5×4×6×9=1080
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_ls38_fine"
    _c = _cfg(_n, (20, 24, 1), (0.35, 0.50, 0.05), (34, 44, 2), (1.40, 1.80, 0.05))
    log.info("A02  LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(34-44 s2)×SS(1.40-1.80 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ls_short — Very short LS=20-36 (below boundary of all prior scans)
    #      LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(20-36 s4)×SS(1.20-2.00 s0.20) = 5×6×5×5=750
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ls_short"
    _c = _cfg(_n, (18, 26, 2), (0.30, 0.55, 0.05), (20, 36, 4), (1.20, 2.00, 0.20))
    log.info("A03  LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(20-36 s4)×SS(1.20-2.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ls_long — Long LS=50-90 (above all prior scans — unexplored territory)
    #      LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(50-90 s10)×SS(1.30-2.50 s0.30) = 5×6×5×5=750
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ls_long"
    _c = _cfg(_n, (18, 26, 2), (0.30, 0.55, 0.05), (50, 90, 10), (1.30, 2.50, 0.30))
    log.info("A04  LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(50-90 s10)×SS(1.30-2.50 s0.30)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  sl_ultra_tight — SL=0.05-0.25 (like NQ SL=0.2 breakthrough, never tried on TXF)
    #      LL(18-26 s2)×SL(0.05-0.25 s0.05)×LS(35-55 s5)×SS(1.20-2.00 s0.20) = 5×5×5×5=625
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_sl_ultra_tight"
    _c = _cfg(_n, (18, 26, 2), (0.05, 0.25, 0.05), (35, 55, 5), (1.20, 2.00, 0.20))
    log.info("A05  LL(18-26 s2)×SL(0.05-0.25 s0.05)×LS(35-55 s5)×SS(1.20-2.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  ss_extreme_tight — SS=0.20-0.80 short entry (extremely tight, never tried)
    #      LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(30-60 s5)×SS(0.20-0.80 s0.10) = 5×6×7×7=1470
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_ss_extreme_tight"
    _c = _cfg(_n, (18, 26, 2), (0.30, 0.55, 0.05), (30, 60, 5), (0.20, 0.80, 0.10))
    log.info("A06  LL(18-26 s2)×SL(0.30-0.55 s0.05)×LS(30-60 s5)×SS(0.20-0.80 s0.10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  ll_low_regime — LL=12-20 (below all prior careful exploration)
    #      LL(12-20 s2)×SL(0.30-0.60 s0.05)×LS(35-55 s5)×SS(1.40-2.00 s0.20) = 5×7×5×4=700
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ll_low_regime"
    _c = _cfg(_n, (12, 20, 2), (0.30, 0.60, 0.05), (35, 55, 5), (1.40, 2.00, 0.20))
    log.info("A07  LL(12-20 s2)×SL(0.30-0.60 s0.05)×LS(35-55 s5)×SS(1.40-2.00 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  sl_medium — SL=0.5-1.2 (gap between tight and moderate, rarely explored)
    #      LL(18-26 s2)×SL(0.50-1.20 s0.10)×LS(35-55 s5)×SS(1.20-2.20 s0.20) = 5×8×5×6=1200
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_sl_medium"
    _c = _cfg(_n, (18, 26, 2), (0.50, 1.20, 0.10), (35, 55, 5), (1.20, 2.20, 0.20))
    log.info("A08  LL(18-26 s2)×SL(0.50-1.20 s0.10)×LS(35-55 s5)×SS(1.20-2.20 s0.20)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  global_r5 — Different coarse global (new grid vs R3/R4)
    #      LL(5-65 s10)×SL(0.10-1.90 s0.30)×LS(5-65 s10)×SS(0.40-2.50 s0.35) = 7×7×7×7=2401
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_global_r5"
    _c = _cfg(_n, (5, 65, 10), (0.10, 1.90, 0.30), (5, 65, 10), (0.40, 2.50, 0.35))
    log.info("A09  LL(5-65 s10)×SL(0.10-1.90 s0.30)×LS(5-65 s10)×SS(0.40-2.50 s0.35)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — SL/SS step=0.05 (broader than R4's 0.005/0.001)
    #      to cast wider net around best candidate found so far
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (3, 0.20, 3, 0.25),
            (3, 0.20, 3, 0.20),
            (2, 0.20, 2, 0.20),
            (2, 0.15, 2, 0.15),
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.05,  SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.05,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  ls38_vfine — LS=35-43 step=1, SS step=0.05 (very fine scan of tight-SS regime)
    #      LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(35-43 s1)×SS(1.40-1.90 s0.05) = 5×4×9×11=1980
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_ls38_vfine"
    _c = _cfg(_n, (20, 24, 1), (0.35, 0.50, 0.05), (35, 43, 1), (1.40, 1.90, 0.05))
    log.info("A11  LL(20-24 s1)×SL(0.35-0.50 s0.05)×LS(35-43 s1)×SS(1.40-1.90 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  main_regime_push — Push the known 8M regime with expanded SS range
    #      LL(20-24 s1)×SL(0.415-0.440 s0.005)×LS(41-46 s1)×SS(1.755-1.815 s0.005) = 5×6×6×13=2340
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_main_regime_push"
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
    log.info("  TXF Hourly countertrend_LS Round-5 COMPLETE")
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R4 seed NP: %.0f → R5 best: %.0f  (Δ %.0f  +%.1f%%)",
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
        description="TXF Hourly countertrend_LS NP>9M TWD Round-5 search")
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
