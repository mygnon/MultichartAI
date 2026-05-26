"""
search_txf_ct_240min4.py — SFJ_15Dworkshop_lesson5_countertrend_LS on TWF.TXF HOT 240-Minute, Round 4

R1: LL=12 SL=0.45 LS=52 SS=0.10  NP=6,243,800
R2: LL=12 SL=0.7  LS=52 SS=0.125 NP=6,742,400 (+8.0%)
R3: LL=10 SL=0.575 LS=53 SS=0.14 NP=6,951,200 (+3.1%), MDD=-930,600, Obj=51,922,610, trades=426
Target: NP > 8,000,000 TWD (gap from R3: −13.1%)

Key R3 discoveries:
  - LL=10 dominates (better NP AND better MDD than LL=12 regime)
  - SL=0.575 is the R3 peak (lower than R2's 0.7)
  - LS=53 (ODD) beats LS=52 — all step=2 zooms missed this!
  - SS=0.14 slightly higher than R2's 0.125
  - MDD stabilized at -930,600

Critical: Adaptive zooms in R3 (A10-A12) used LS step=2 → never sampled LS=53.
R4 must use LS step=1 in all attempts.

R4 focus: fine-tune around LL=10, SL=0.575, LS=53, SS=0.14 with step=1 for LS everywhere.
  1. seed_confirm   — Confirm R3 champion, scan LS=49-57 step=1
  2. sl_ultra_fine  — SL step=0.01 (0.52-0.63) to precisely locate SL peak
  3. ss_fine2       — SS step=0.005 (0.12-0.18) to confirm SS peak
  4. ll_fine        — LL step=1 (8-14) to confirm optimal LL integer
  5. ls_fine2       — LS step=1 wider (44-62) to check if LS>53 or LS<51 better
  6. high_ll_zone   — Probe LL=14-22 fine to ensure we haven't missed a high-LL regime
  7. sl_extend_fine — SL=0.60-0.85 step=0.01 to confirm SL ceiling
  8. global_r4      — Broad sweep with fine-SS to catch any unexplored territory
  9. ls_wide_step1  — LS=40-70 step=1 with optimal LL/SL to map full LS landscape

Attempt schedule (9 fixed + 3 adaptive, ≤5,000 combos each):
  A01 seed_confirm   : LL(8-14 s2)×SL(0.50-0.65 s0.025)×LS(49-57 s1)×SS(0.12-0.16 s0.01) = 4×7×9×5 = 1260
  A02 sl_ultra_fine  : LL(10-12 s2)×SL(0.52-0.63 s0.01)×LS(51-55 s1)×SS(0.12-0.16 s0.01)  = 2×12×5×5 = 600
  A03 ss_fine2       : LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(51-55 s1)×SS(0.12-0.18 s0.005) = 2×5×5×13 = 650
  A04 ll_fine        : LL(8-14 s1)×SL(0.55-0.65 s0.025)×LS(51-55 s1)×SS(0.12-0.16 s0.01)   = 7×5×5×5 = 875
  A05 ls_fine2       : LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(44-62 s1)×SS(0.12-0.16 s0.01)  = 2×5×19×5 = 950
  A06 high_ll_zone   : LL(14-22 s2)×SL(0.50-0.70 s0.025)×LS(48-56 s2)×SS(0.10-0.16 s0.01)  = 5×9×5×7 = 1575
  A07 sl_extend_fine : LL(10-14 s2)×SL(0.60-0.85 s0.01)×LS(51-55 s1)×SS(0.12-0.16 s0.01)   = 3×26×5×5 = 1950
  A08 global_r4      : LL(4-64 s10)×SL(0.3-1.3 s0.25)×LS(10-80 s10)×SS(0.05-0.30 s0.05)    = 7×5×8×6 = 1680
  A09 ls_wide_step1  : LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(40-70 s1)×SS(0.12-0.16 s0.01)  = 2×5×31×5 = 1550
  A10 adaptive_zoom  : (dynamic from best NP, LS step=1)
  A11 adaptive_zoom2 : (dynamic from best NP, LS step=1)
  A12 adaptive_zoom3 : (dynamic from best NP, LS step=1)
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
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_ct_240min4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 8_000_000.0   # TWD

LL_LO, LL_HI  = 2.0,  500.0
SL_LO, SL_HI  = 0.05, 20.0
LS_LO, LS_HI  = 2.0,  500.0
SS_LO, SS_HI  = 0.05, 20.0

# R3 champion
SEED_LL, SEED_SL = 10.0, 0.575
SEED_LS, SEED_SS = 53.0, 0.14
SEED_NP          = 6_951_200.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_ct_240min4_{int(time.time())}.log"
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
        name=f"TXFCT240_4_{name}",
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
    return OUTPUT_DIR / f"TXFCT240_4_{name}_raw.csv"


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
    log.info("=== Starting TXFCT240_4_%s (%d combos) ===", name, cfg.total_runs())
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
        "timeframe": "240min",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":    "BUY when C crosses over lower BB; SHORT when C crosses under upper BB",
            "exits":    "Reversal only — no STP or LMT",
            "params":   "LENGTH_LONG, STDDEV_LONG (long entry) / LENGTH_SHORT, STDDEV_SHORT (short entry)",
            "r3_champion": "LL=10 SL=0.575 LS=53 SS=0.14 NP=6,951,200 MDD=-930,600 Obj=51,922,610 trades=426",
            "r4_focus": "LS step=1 in all attempts (R3 zooms missed LS=53 with step=2); fine SL step=0.01; SS step=0.005; LL step=1",
            "key_insight": "LS=53 (odd) discovered via step=1 scan in R3 A05; step=2 always skips odd LS values",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_ct_240min4.json"
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
    log.info("  TXF 240-Minute countertrend_LS NP>8M TWD — Round 4")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R3 champion: LL=10 SL=0.575 LS=53 SS=0.14 NP=6,951,200 MDD=-930,600")
    log.info("  Key: LS=53 is ODD — step=2 zooms always skipped it; R4 uses LS step=1 everywhere")
    log.info("  R4 seed: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             SEED_LL, SEED_SL, SEED_LS, SEED_SS, SEED_NP)
    log.info("  Target: %.0f TWD  (gap from seed: −13.1%%)", TARGET_NP)
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
    # A01  seed_confirm — confirm R3 champion, LS step=1 around 49-57
    #      LL(8-14 s2)×SL(0.50-0.65 s0.025)×LS(49-57 s1)×SS(0.12-0.16 s0.01) = 4×7×9×5 = 1260
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_seed_confirm"
    _c = _cfg(_n, (8, 14, 2), (0.50, 0.65, 0.025), (49, 57, 1), (0.12, 0.16, 0.01))
    log.info("A01  LL(8-14 s2)×SL(0.50-0.65 s0.025)×LS(49-57 s1)×SS(0.12-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A02  sl_ultra_fine — SL step=0.01 precision scan (0.52-0.63)
    #      LL(10-12 s2)×SL(0.52-0.63 s0.01)×LS(51-55 s1)×SS(0.12-0.16 s0.01) = 2×12×5×5 = 600
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_sl_ultra_fine"
    _c = _cfg(_n, (10, 12, 2), (0.52, 0.63, 0.01), (51, 55, 1), (0.12, 0.16, 0.01))
    log.info("A02  LL(10-12 s2)×SL(0.52-0.63 s0.01)×LS(51-55 s1)×SS(0.12-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A03  ss_fine2 — SS step=0.005 precision scan (0.12-0.18)
    #      LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(51-55 s1)×SS(0.12-0.18 s0.005) = 2×5×5×13 = 650
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_ss_fine2"
    _c = _cfg(_n, (10, 12, 2), (0.55, 0.65, 0.025), (51, 55, 1), (0.12, 0.18, 0.005))
    log.info("A03  LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(51-55 s1)×SS(0.12-0.18 s0.005)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A04  ll_fine — LL step=1 integer scan (8-14)
    #      LL(8-14 s1)×SL(0.55-0.65 s0.025)×LS(51-55 s1)×SS(0.12-0.16 s0.01) = 7×5×5×5 = 875
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_ll_fine"
    _c = _cfg(_n, (8, 14, 1), (0.55, 0.65, 0.025), (51, 55, 1), (0.12, 0.16, 0.01))
    log.info("A04  LL(8-14 s1)×SL(0.55-0.65 s0.025)×LS(51-55 s1)×SS(0.12-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A05  ls_fine2 — LS step=1, wider range (44-62)
    #      LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(44-62 s1)×SS(0.12-0.16 s0.01) = 2×5×19×5 = 950
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_ls_fine2"
    _c = _cfg(_n, (10, 12, 2), (0.55, 0.65, 0.025), (44, 62, 1), (0.12, 0.16, 0.01))
    log.info("A05  LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(44-62 s1)×SS(0.12-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A06  high_ll_zone — probe LL=14-22, fine SL/SS (may have missed a high-LL peak)
    #      LL(14-22 s2)×SL(0.50-0.70 s0.025)×LS(48-56 s2)×SS(0.10-0.16 s0.01) = 5×9×5×7 = 1575
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_high_ll_zone"
    _c = _cfg(_n, (14, 22, 2), (0.50, 0.70, 0.025), (48, 56, 2), (0.10, 0.16, 0.01))
    log.info("A06  LL(14-22 s2)×SL(0.50-0.70 s0.025)×LS(48-56 s2)×SS(0.10-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A07  sl_extend_fine — SL=0.60-0.85 step=0.01, fine LS step=1
    #      LL(10-14 s2)×SL(0.60-0.85 s0.01)×LS(51-55 s1)×SS(0.12-0.16 s0.01) = 3×26×5×5 = 1950
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_sl_extend_fine"
    _c = _cfg(_n, (10, 14, 2), (0.60, 0.85, 0.01), (51, 55, 1), (0.12, 0.16, 0.01))
    log.info("A07  LL(10-14 s2)×SL(0.60-0.85 s0.01)×LS(51-55 s1)×SS(0.12-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A08  global_r4 — broad sweep to catch unexplored territory
    #      LL(4-64 s10)×SL(0.3-1.3 s0.25)×LS(10-80 s10)×SS(0.05-0.30 s0.05) = 7×5×8×6 = 1680
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_global_r4"
    _c = _cfg(_n, (4, 64, 10), (0.3, 1.3, 0.25), (10, 80, 10), (0.05, 0.30, 0.05))
    log.info("A08  LL(4-64 s10)×SL(0.3-1.3 s0.25)×LS(10-80 s10)×SS(0.05-0.30 s0.05)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A09  ls_wide_step1 — full LS landscape (40-70) at step=1
    #      LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(40-70 s1)×SS(0.12-0.16 s0.01) = 2×5×31×5 = 1550
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_ls_wide_step1"
    _c = _cfg(_n, (10, 12, 2), (0.55, 0.65, 0.025), (40, 70, 1), (0.12, 0.16, 0.01))
    log.info("A09  LL(10-12 s2)×SL(0.55-0.65 s0.025)×LS(40-70 s1)×SS(0.12-0.16 s0.01)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom — LS step=1 (critical — don't miss odd LS values)
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom"
    log.info("A10  adaptive_zoom — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (6,  0.12, 8, 0.05),
            (4,  0.08, 6, 0.04),
            (3,  0.06, 5, 0.03),
            (2,  0.05, 4, 0.025),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.025, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)   # step=1 to include odd LS
            _ss = zoom(best_ss, r_ss, 0.01,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A10  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom2
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom2"
    log.info("A11  adaptive_zoom2 — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (3,  0.06, 5, 0.03),
            (2,  0.05, 4, 0.025),
            (2,  0.04, 3, 0.02),
            (2,  0.03, 2, 0.01),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.025, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)   # step=1
            _ss = zoom(best_ss, r_ss, 0.01,  SS_LO, SS_HI)
            _c  = _cfg(_n, _ll, _sl, _ls, _ss)
            if _c.total_runs() <= 5000:
                break
        log.info("A11  LL%s SL%s LS%s SS%s  %d combos", _ll, _sl, _ls, _ss, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LL=%.4g SL=%.4g LS=%.4g SS=%.4g)",
             best_np, best_ll, best_sl, best_ls, best_ss)

    # ──────────────────────────────────────────────────────────────────────
    # A12  adaptive_zoom3 — tight final zoom
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_adaptive_zoom3"
    log.info("A12  adaptive_zoom3 — center: LL=%.4g SL=%.4g LS=%.4g SS=%.4g  NP=%.0f",
             best_ll, best_sl, best_ls, best_ss, best_np)
    if start_attempt <= A:
        for r_ll, r_sl, r_ls, r_ss in [
            (2,  0.05, 4, 0.025),
            (2,  0.04, 3, 0.02),
            (2,  0.03, 2, 0.01),
            (2,  0.025, 2, 0.01),
        ]:
            _ll = zoom(best_ll, r_ll, 2.0,   LL_LO, LL_HI)
            _sl = zoom(best_sl, r_sl, 0.025, SL_LO, SL_HI)
            _ls = zoom(best_ls, r_ls, 1.0,   LS_LO, LS_HI)   # step=1
            _ss = zoom(best_ss, r_ss, 0.01,  SS_LO, SS_HI)
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
    log.info("  TXF 240-Minute countertrend_LS Round-4 COMPLETE")
    log.info("  Best NP: %.0f TWD  LL=%.4g SL=%.4g LS=%.4g SS=%.4g",
             best_np, best_ll, best_sl, best_ls, best_ss)
    log.info("  R3 seed: 6,951,200  R4 best: %.0f  gain: %+.1f%%",
             best_np, (best_np / 6_951_200 - 1) * 100)
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
        description="TXF 240-Minute countertrend_LS NP>8M TWD Round-4 search")
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
