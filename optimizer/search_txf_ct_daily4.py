"""
search_txf_ct_daily4.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Daily, Round 4

Progression:
  R1: LL=20 SL=0.28 LS=56 SS=1.27  NP=3,270,600  MDD=-1,189,800  (baseline)
  R2: LL=23 SL=0.13 LS=49 SS=0.27  NP=3,978,600  MDD=-931,000   (+21.7% — new regime found)
  R3: LL=25 SL=0.165 LS=50 SS=0.275 NP=4,019,800 MDD=-931,000   (+1.0% — converging)
  Rate: R1→R2 +21.7%, R2→R3 +1.0% → NQ daily pattern; ceiling forming at ~4.0-4.1M

  Current best: LL=25 SL=0.165 LS=50 SS=0.275  NP=4,019,800  MDD=-931,000  trades=45
  8M gap: -49.8% — structural (45 trades/7yr × ~$89K/trade = $4.0M ceiling)

  Unexplored regions after R1-R3:
    - LL=30-100 in tight-SS regime (only LL≤30 explored; could high LL change trade frequency?)
    - SS < 0.20 (all searches stopped at SS=0.20)
    - LS=100-300 with tight SS (extreme long LS)
    - LL=2-14 fine step=1 in tight regime (only coarse step=2 in R2)
    - Fine LS step=1 around LS=50 (R3 A05 used mixed SL/LS but found LS=50 optimal)
    - Ultra-fine SL at step=0.002 around SL=0.165 (R3 finest was step=0.005)
    - Completely symmetric regime (SL=SS): not yet tried deliberately

R4 strategy — final exhaustive sweep + ceiling confirmation:
  A01 ultra_fine_peak    : Finest possible around champion (SL step=0.005, SS step=0.005)
  A02 very_low_ss        : SS=0.05-0.18 regime (below all prior searches)
  A03 high_ll_tight      : LL=30-80 in tight-SS regime
  A04 ls_fine_step1      : LS step=1 around LS=50 in tight regime
  A05 ultra_short_ll_fine: LL=2-14 step=1 fine (only coarse=2 in R2)
  A06 extreme_large_ls   : LS=100-300 (regime with infrequent trades)
  A07 global_r4          : New global scan (different grid from R1-R3)
  A08 symmetric_regime   : SL=SS deliberately (symmetric BB)
  A09 final_broad_sl     : SL=0.05-0.50 full scan around optimal LS/SS
  A10 adaptive_zoom      : step=(1,0.005,1,0.005) cascade from R4 best
  A11 fine_zoom          : step=(1,0.002,1,0.002) ultra-precision
  A12 ultra_fine_verify  : Confirm peak at SL step=0.002, SS step=0.001

Attempt schedule (≤5,000 combos each):
  A01  2700  LL(23-27 s1)×SL(0.140-0.195 s0.005)×LS(48-52 s1)×SS(0.255-0.295 s0.005)
  A02   480  LL(20-28 s2)×SL(0.10-0.22 s0.04)×LS(45-55 s5)×SS(0.05-0.19 s0.02)
  A03   336  LL(30-60 s5)×SL(0.10-0.25 s0.05)×LS(45-55 s5)×SS(0.20-0.35 s0.05)
  A04  1500  LL(23-27 s1)×SL(0.14-0.20 s0.02)×LS(44-58 s1)×SS(0.255-0.295 s0.01)
  A05   756  LL(2-14 s1)×SL(0.10-0.30 s0.05)×LS(45-60 s5)×SS(0.20-0.35 s0.05)
  A06   180  LL(20-30 s5)×SL(0.10-0.20 s0.05)×LS(100-300 s50)×SS(0.20-0.35 s0.05)
  A07   864  LL(5-30 s5)×SL(0.05-0.35 s0.1)×LS(20-70 s10)×SS(0.10-0.60 s0.1)
  A08  1296  LL(15-28 s3)×SL(0.05-0.40 s0.05)×LS(40-65 s5)×SS(0.05-0.40 s0.05)  [SL≈SS band]
  A09   840  LL(20-26 s2)×SL(0.05-0.50 s0.05)×LS(48-52 s1)×SS(0.24-0.30 s0.01)
  A10  ≤5000 adaptive zoom step=(1,0.005,1,0.005)
  A11  ≤5000 fine zoom step=(1,0.002,1,0.002)
  A12  ≤5000 ultra-fine verify: SL step=0.002, SS step=0.001
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_daily4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.05, 20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.05, 20.0

# R3 NP-champion as seed
SEED_LL, SEED_SL = 25.0, 0.165
SEED_LS, SEED_SS = 50.0, 0.275
SEED_NP          = 4_019_800.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_daily4_{int(time.time())}.log"
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
        name=f"TXFCTD4_{name}",
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
    return OUTPUT_DIR / f"TXFCTD4_{name}_raw.csv"


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
    log.info("=== Starting TXFCTD4_%s (%d combos) ===", name, cfg.total_runs())
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
            "r1_best":  "LL=20 SL=0.28 LS=56 SS=1.27 NP=3270600 MDD=-1189800",
            "r2_best":  "LL=23 SL=0.13 LS=49 SS=0.27 NP=3978600 MDD=-931000 (+21.7%)",
            "r3_best":  "LL=25 SL=0.165 LS=50 SS=0.275 NP=4019800 MDD=-931000 (+1.0%)",
            "r4_focus": "Final ceiling confirmation: high-LL, very-low-SS, extreme-LS, ultra-fine peak",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_daily4.json"
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
    log.info("  TXF Daily countertrend_LS NP>8,000,000 TWD — Round 4")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R3 champion: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  Gain rate: R1→R2 +21.7%%, R2→R3 +1.0%% — ceiling likely at ~4.0M")
    log.info("  R4: final exhaustive sweep of unexplored regions + ceiling confirmation")
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
    # A01  ultra_fine_peak — Finest possible around R3 champion
    #      LL(23-27 s1)×SL(0.140-0.195 s0.005)×LS(48-52 s1)×SS(0.255-0.295 s0.005) = 5×12×5×9=2700
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ultra_fine_peak"
    _c = _cfg(_n, (23, 27, 1), (0.140, 0.195, 0.005), (48, 52, 1), (0.255, 0.295, 0.005))
    log.info("A01  LL(23-27 s1)×SL(0.140-0.195 s0.005)×LS(48-52 s1)×SS(0.255-0.295 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  very_low_ss — SS=0.05-0.18 below all prior searches
    #      LL(20-28 s2)×SL(0.10-0.22 s0.04)×LS(45-55 s5)×SS(0.05-0.19 s0.02) = 5×4×3×8=480
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_very_low_ss"
    _c = _cfg(_n, (20, 28, 2), (0.10, 0.22, 0.04), (45, 55, 5), (0.05, 0.19, 0.02))
    log.info("A02  LL(20-28 s2)×SL(0.10-0.22 s0.04)×LS(45-55 s5)×SS(0.05-0.19 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  high_ll_tight — LL=30-80 in tight-SS regime (LL>30 never explored)
    #      LL(30-80 s10)×SL(0.10-0.25 s0.05)×LS(45-60 s5)×SS(0.20-0.35 s0.05) = 6×4×4×4=384
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_high_ll_tight"
    _c = _cfg(_n, (30, 80, 10), (0.10, 0.25, 0.05), (45, 60, 5), (0.20, 0.35, 0.05))
    log.info("A03  LL(30-80 s10)×SL(0.10-0.25 s0.05)×LS(45-60 s5)×SS(0.20-0.35 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ls_fine_step1 — LS step=1 precise scan around LS=50 in tight regime
    #      LL(23-27 s1)×SL(0.14-0.20 s0.02)×LS(44-58 s1)×SS(0.255-0.295 s0.01) = 5×4×15×5=1500
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ls_fine_step1"
    _c = _cfg(_n, (23, 27, 1), (0.14, 0.20, 0.02), (44, 58, 1), (0.255, 0.295, 0.01))
    log.info("A04  LL(23-27 s1)×SL(0.14-0.20 s0.02)×LS(44-58 s1)×SS(0.255-0.295 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ultra_short_ll_fine — LL=2-14 step=1 (finer than R2 step=2)
    #      LL(2-14 s1)×SL(0.10-0.30 s0.05)×LS(45-60 s5)×SS(0.20-0.35 s0.05) = 13×5×4×4=1040
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ultra_short_ll_fine"
    _c = _cfg(_n, (2, 14, 1), (0.10, 0.30, 0.05), (45, 60, 5), (0.20, 0.35, 0.05))
    log.info("A05  LL(2-14 s1)×SL(0.10-0.30 s0.05)×LS(45-60 s5)×SS(0.20-0.35 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  extreme_large_ls — LS=100-300 (regime with very infrequent signals)
    #      LL(20-30 s5)×SL(0.10-0.20 s0.05)×LS(100-300 s50)×SS(0.20-0.35 s0.05) = 3×3×5×4=180
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_extreme_large_ls"
    _c = _cfg(_n, (20, 30, 5), (0.10, 0.20, 0.05), (100, 300, 50), (0.20, 0.35, 0.05))
    log.info("A06  LL(20-30 s5)×SL(0.10-0.20 s0.05)×LS(100-300 s50)×SS(0.20-0.35 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  global_r4 — New global scan (different grid from all prior rounds)
    #      LL(5-30 s5)×SL(0.05-0.35 s0.1)×LS(20-70 s10)×SS(0.10-0.60 s0.1) = 6×4×6×6=864
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_global_r4"
    _c = _cfg(_n, (5, 30, 5), (0.05, 0.35, 0.1), (20, 70, 10), (0.10, 0.60, 0.1))
    log.info("A07  LL(5-30 s5)×SL(0.05-0.35 s0.1)×LS(20-70 s10)×SS(0.10-0.60 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  symmetric_regime — SL≈SS symmetric bands (deliberately unexplored)
    #      LL(15-28 s3)×SL(0.05-0.40 s0.05)×LS(40-65 s5)×SS(0.05-0.40 s0.05)
    #      = 6×8×6×8=2304  [picks where SL and SS happen to be similar]
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_symmetric_regime"
    _c = _cfg(_n, (15, 28, 3), (0.05, 0.40, 0.05), (40, 65, 5), (0.05, 0.40, 0.05))
    log.info("A08  LL(15-28 s3)×SL(0.05-0.40 s0.05)×LS(40-65 s5)×SS(0.05-0.40 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  final_broad_sl — Full SL scan at best LS/SS/LL neighborhood
    #      LL(20-26 s2)×SL(0.05-0.50 s0.05)×LS(48-52 s1)×SS(0.24-0.30 s0.01) = 4×10×5×7=1400
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_final_broad_sl"
    _c = _cfg(_n, (20, 26, 2), (0.05, 0.50, 0.05), (48, 52, 1), (0.24, 0.30, 0.01))
    log.info("A09  LL(20-26 s2)×SL(0.05-0.50 s0.05)×LS(48-52 s1)×SS(0.24-0.30 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — SL step=0.005, SS step=0.005 cascade
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.03, 2, 0.03),   # 5×13×5×13=4225
            (2, 0.02, 2, 0.02),   # 5×9×5×9=2025
            (1, 0.02, 1, 0.02),   # 3×9×3×9=729
            (1, 0.015, 1, 0.015), # 3×7×3×7=441
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.005, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.005, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  fine_zoom — SL step=0.002, SS step=0.002 ultra-precision
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_fine_zoom"
    log.info("A11  fine_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.02, 2, 0.02),   # 5×21×5×21=11025 — too many
            (2, 0.012, 2, 0.012), # 5×13×5×13=4225 ✓
            (2, 0.008, 2, 0.008), # 5×9×5×9=2025 ✓
            (1, 0.008, 1, 0.008), # 3×9×3×9=729 ✓
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.002, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.002, SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  ultra_fine_verify — SL step=0.002, SS step=0.001 (maximum precision)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_ultra_fine_verify"
    log.info("A12  ultra_fine_verify — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (1, 0.012, 2, 0.010),  # 3×13×5×21=4095 ✓
            (1, 0.010, 1, 0.010),  # 3×11×3×21=2079 ✓
            (1, 0.008, 1, 0.008),  # 3×9×3×17=1377 ✓
            (1, 0.006, 1, 0.006),  # 3×7×3×13=819 ✓
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.002, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.001, SS_LO, SS_HI)
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
    log.info("  TXF Daily countertrend_LS Round-4 COMPLETE")
    log.info("  R3 seed NP: %.0f → R4 best: %.0f  (Δ %.0f  %+.1f%%)",
             SEED_NP, best_np, best_np - SEED_NP, (best_np / SEED_NP - 1) * 100)
    log.info("  Best NP: %.0f  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  Target %.0f TWD: %s", TARGET_NP,
             "★ MET" if target_met else f"NOT MET (gap +{max(0, TARGET_NP - best_np):.0f})")
    log.info("  Gain trajectory: R1 3.27M → R2 3.98M (+21.7%%) → R3 4.02M (+1.0%%) → R4 %.0f",
             best_np)
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
        description="TXF Daily countertrend_LS NP>8M TWD Round-4 search")
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
