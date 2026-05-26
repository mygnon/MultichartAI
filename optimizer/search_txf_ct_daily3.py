"""
search_txf_ct_daily3.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT Daily, Round 3

R2 findings (search_txf_ct_daily2.py):
  Best NP : LL=23 SL=0.13 LS=49 SS=0.27  NP=3,978,600  MDD=-931,000  Obj=17,002,425  trades=43
  R1→R2 gain: +21.7% (3,270,600 → 3,978,600). Gap: -50.3% from 8M target.
  MDD improved: -931,000 (vs R1's -1,189,800) — new structural floor in tight-SS regime.

  Key R2 discoveries:
    (A) Ultra-tight regime: LL=23, SL=0.13, LS=49, SS=0.27  NP=3,978,600  Obj=17,002,425
        Found by A04 tight_ss_regime (SS=0.10-0.60) — completely missed in R1!
        SL=0.13 is ultra-tight (R1's SL=0.28 was too wide to find this)
    (B) Long-LS regime: LL=20, SL=0.30, LS=70, SS=1.40  NP=3,544,400  Obj=10,558,725
        A03 showed LS=70 beats LS=56 (+8.4%) — still unexplored LS=75-95
    (C) R2 A11→A12 regression: SS=0.27 → SS=0.265 gave NP drop (3,978,600→3,893,000)
        True SS peak is between 0.265 and 0.27 — needs SS step=0.002

  Structural: MDD floor at -931,000 in tight-SS regime (same floor across A10-A12)

R3 strategy — deep exploration of ultra-tight regime + new combinations:
  A01 tight_regime_fine   : Fine LL/SL/LS/SS around new champion (SL step=0.01, SS step=0.01)
  A02 tight_long_ls       : Tight SS + longer LS (combine A discovery + B discovery)
  A03 sl_ultra_broad      : Broad SL=0.05-0.25 scan in tight regime (find true SL peak)
  A04 ls_tight_regime     : LS range scan in tight-SS regime (LS=35-75)
  A05 ll_fine_tight       : LL step=1 fine scan in tight regime (LL=18-30)
  A06 long_ls_tight_ss    : LS=65-95 with tight SS (unexplored territory)
  A07 ss_fine_precise     : SS step=0.005 ultra-fine around SS=0.27
  A08 sl_fine_precise     : SL step=0.005 ultra-fine around SL=0.13
  A09 combine_regimes     : Combine A03 long-LS (LS=70) with tight-SS
  A10 adaptive_zoom       : step=(1,0.01,1,0.01) cascade from R3 best
  A11 fine_zoom           : step=(1,0.005,1,0.005) cascade from R3 best
  A12 ultra_fine_both     : SL step=0.005, SS step=0.002 final ultra-tune

Attempt schedule (≤5,000 combos each):
  A01  1575  LL(21-25 s1)×SL(0.10-0.18 s0.01)×LS(45-53 s2)×SS(0.22-0.34 s0.02)
  A02   600  LL(20-26 s2)×SL(0.10-0.20 s0.02)×LS(60-80 s5)×SS(0.20-0.40 s0.05)
  A03   880  LL(18-26 s2)×SL(0.05-0.25 s0.02)×LS(44-56 s4)×SS(0.20-0.35 s0.05)
  A04   560  LL(20-26 s2)×SL(0.10-0.18 s0.02)×LS(35-65 s5)×SS(0.20-0.35 s0.05)
  A05  1170  LL(18-30 s1)×SL(0.10-0.18 s0.02)×LS(45-53 s4)×SS(0.22-0.32 s0.02)
  A06   980  LL(18-26 s2)×SL(0.08-0.20 s0.02)×LS(65-95 s5)×SS(0.20-0.35 s0.05)
  A07  1680  LL(21-25 s1)×SL(0.10-0.16 s0.02)×LS(46-52 s2)×SS(0.220-0.320 s0.005)
  A08  1700  LL(21-25 s1)×SL(0.090-0.170 s0.005)×LS(46-52 s2)×SS(0.24-0.32 s0.02)
  A09   700  LL(18-24 s2)×SL(0.20-0.40 s0.05)×LS(60-80 s5)×SS(1.2-1.8 s0.1)
  A10  ≤5000 adaptive zoom step=(1,0.01,1,0.01) cascade
  A11  ≤5000 fine zoom step=(1,0.005,1,0.005) cascade
  A12  ≤5000 ultra-fine SL step=0.005, SS step=0.002 final tune
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_daily3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0

LL_LO, LL_HI = 2.0,  500.0
SL_LO, SL_HI = 0.05, 20.0
LS_LO, LS_HI = 2.0,  500.0
SS_LO, SS_HI = 0.05, 20.0

# R2 NP-champion (ultra-tight regime) as seed
SEED_LL, SEED_SL = 23.0, 0.13
SEED_LS, SEED_SS = 49.0, 0.27
SEED_NP          = 3_978_600.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_daily3_{int(time.time())}.log"
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
        name=f"TXFCTD3_{name}",
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
    return OUTPUT_DIR / f"TXFCTD3_{name}_raw.csv"


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
    log.info("=== Starting TXFCTD3_%s (%d combos) ===", name, cfg.total_runs())
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
        "round": 3,
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
            "r3_focus": "Ultra-tight SL/SS fine-tune: SL step=0.005-0.01, SS step=0.002-0.005; LS>60 + tight SS",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_daily3.json"
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
    log.info("  TXF Daily countertrend_LS NP>8,000,000 TWD — Round 3")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R2 champion: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  R2 gap: -50.3%% from 8M. R3 deep-dives ultra-tight SL/SS regime.")
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
    # A01  tight_regime_fine — Fine LL/SL/LS/SS around R2 champion
    #      LL(21-25 s1)×SL(0.10-0.18 s0.01)×LS(45-53 s2)×SS(0.22-0.34 s0.02) = 5×9×5×7=1575
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_tight_regime_fine"
    _c = _cfg(_n, (21, 25, 1), (0.10, 0.18, 0.01), (45, 53, 2), (0.22, 0.34, 0.02))
    log.info("A01  LL(21-25 s1)×SL(0.10-0.18 s0.01)×LS(45-53 s2)×SS(0.22-0.34 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  tight_long_ls — Tight SS + longer LS (combine R2 A03 and A04 regimes)
    #      LL(20-26 s2)×SL(0.10-0.20 s0.02)×LS(60-80 s5)×SS(0.20-0.40 s0.05) = 4×6×5×5=600
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_tight_long_ls"
    _c = _cfg(_n, (20, 26, 2), (0.10, 0.20, 0.02), (60, 80, 5), (0.20, 0.40, 0.05))
    log.info("A02  LL(20-26 s2)×SL(0.10-0.20 s0.02)×LS(60-80 s5)×SS(0.20-0.40 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  sl_ultra_broad — Broad SL=0.05-0.25 scan (find true SL lower bound)
    #      LL(18-26 s2)×SL(0.05-0.25 s0.02)×LS(44-56 s4)×SS(0.20-0.35 s0.05) = 5×11×4×4=880
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_sl_ultra_broad"
    _c = _cfg(_n, (18, 26, 2), (0.05, 0.25, 0.02), (44, 56, 4), (0.20, 0.35, 0.05))
    log.info("A03  LL(18-26 s2)×SL(0.05-0.25 s0.02)×LS(44-56 s4)×SS(0.20-0.35 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ls_tight_regime — Scan LS=35-75 in tight-SS regime (true LS range?)
    #      LL(20-26 s2)×SL(0.10-0.18 s0.02)×LS(35-65 s5)×SS(0.20-0.35 s0.05) = 4×5×7×4=560
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ls_tight_regime"
    _c = _cfg(_n, (20, 26, 2), (0.10, 0.18, 0.02), (35, 65, 5), (0.20, 0.35, 0.05))
    log.info("A04  LL(20-26 s2)×SL(0.10-0.18 s0.02)×LS(35-65 s5)×SS(0.20-0.35 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ll_fine_tight — LL step=1 full scan in tight regime (LL=18-30)
    #      LL(18-30 s1)×SL(0.10-0.18 s0.02)×LS(45-53 s4)×SS(0.22-0.32 s0.02) = 13×5×3×6=1170
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ll_fine_tight"
    _c = _cfg(_n, (18, 30, 1), (0.10, 0.18, 0.02), (45, 53, 4), (0.22, 0.32, 0.02))
    log.info("A05  LL(18-30 s1)×SL(0.10-0.18 s0.02)×LS(45-53 s4)×SS(0.22-0.32 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  long_ls_tight_ss — LS=65-95 with ultra-tight SS (unexplored combo)
    #      LL(18-26 s2)×SL(0.08-0.20 s0.02)×LS(65-95 s5)×SS(0.20-0.35 s0.05) = 5×7×7×4=980
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_long_ls_tight_ss"
    _c = _cfg(_n, (18, 26, 2), (0.08, 0.20, 0.02), (65, 95, 5), (0.20, 0.35, 0.05))
    log.info("A06  LL(18-26 s2)×SL(0.08-0.20 s0.02)×LS(65-95 s5)×SS(0.20-0.35 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  ss_fine_precise — SS step=0.005 ultra-fine around SS=0.27
    #      LL(21-25 s1)×SL(0.10-0.16 s0.02)×LS(46-52 s2)×SS(0.220-0.320 s0.005) = 5×4×4×21=1680
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_ss_fine_precise"
    _c = _cfg(_n, (21, 25, 1), (0.10, 0.16, 0.02), (46, 52, 2), (0.220, 0.320, 0.005))
    log.info("A07  LL(21-25 s1)×SL(0.10-0.16 s0.02)×LS(46-52 s2)×SS(0.220-0.320 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  sl_fine_precise — SL step=0.005 ultra-fine around SL=0.13
    #      LL(21-25 s1)×SL(0.090-0.170 s0.005)×LS(46-52 s2)×SS(0.24-0.32 s0.02) = 5×17×4×5=1700
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_sl_fine_precise"
    _c = _cfg(_n, (21, 25, 1), (0.090, 0.170, 0.005), (46, 52, 2), (0.24, 0.32, 0.02))
    log.info("A08  LL(21-25 s1)×SL(0.090-0.170 s0.005)×LS(46-52 s2)×SS(0.24-0.32 s0.02)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  combine_regimes — Combine R2 A03 long-LS (LS=60-80) with moderate SS
    #      LL(18-24 s2)×SL(0.20-0.40 s0.05)×LS(60-80 s5)×SS(1.2-1.8 s0.1) = 4×5×5×7=700
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_combine_regimes"
    _c = _cfg(_n, (18, 24, 2), (0.20, 0.40, 0.05), (60, 80, 5), (1.2, 1.8, 0.1))
    log.info("A09  LL(18-24 s2)×SL(0.20-0.40 s0.05)×LS(60-80 s5)×SS(1.2-1.8 s0.1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — SL step=0.01, SS step=0.01 cascade from R3 best
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2, 0.04, 3, 0.04),   # 5×9×7×9=2835
            (2, 0.04, 2, 0.04),   # 5×9×5×9=2025
            (2, 0.03, 2, 0.03),   # 5×7×5×7=1225
            (1, 0.03, 1, 0.03),   # 3×7×3×7=441
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
    # A11  fine_zoom — SL step=0.005, SS step=0.005 precision cascade
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_fine_zoom"
    log.info("A11  fine_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
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
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  ultra_fine_both — SL step=0.005, SS step=0.002 (maximum precision)
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_ultra_fine_both"
    log.info("A12  ultra_fine_both — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (1, 0.025, 2, 0.012),  # 3×11×5×13=2145
            (1, 0.020, 1, 0.012),  # 3×9×3×13=1053
            (1, 0.015, 1, 0.010),  # 3×7×3×11=693
            (1, 0.010, 1, 0.008),  # 3×5×3×9=405
        ]:
            _ll = zoom(best_ll, r_ll, 1.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.005, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)
            _ss = zoom(best_ss, r_ss, 0.002, SS_LO, SS_HI)
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
    log.info("  TXF Daily countertrend_LS Round-3 COMPLETE")
    log.info("  R2 seed NP: %.0f → R3 best: %.0f  (Δ %.0f  %+.1f%%)",
             SEED_NP, best_np, best_np - SEED_NP, (best_np / SEED_NP - 1) * 100)
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
        description="TXF Daily countertrend_LS NP>8M TWD Round-3 search")
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
