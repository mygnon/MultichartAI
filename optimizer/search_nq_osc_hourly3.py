"""
search_nq_osc_hourly3.py — _2021Basic_Osc_NQ on CME.NQ HOT 60-Minute, Round 3

R1+R2 summary (24 attempts, 40,383+ combos):
  Ceiling confirmed: $453,610  LEN=3 LE=-1.25 SE=1.5 STP=1.0 LMT=18.5
  MDD=-62,280  obj=3,303,822  trades=1427
  Gain rate R1→R2: +0.0%  (three adaptive zooms converged to exact R1 champion)
  Target: $800,000 — gap -43.3% (-$346,390)

R1+R2 explored territory (DONE — do not repeat):
  LEN=1-30 tested        → LEN=3 is optimal
  LE=-2.5 to +0.75       → LE=-1.25 is optimal
  SE=0.75-4.0            → SE=1.5 is optimal
  STP=0.25-2.5           → STP=1.0 is optimal
  LMT=0.5-40             → LMT=18.5 is optimal (LMT>22 gives LESS NP)

R3 focus: genuinely unexplored territory (NOT tested in R1+R2):
  1. ultra_tight_stp  — STP=0.05-0.25 (R2 min was 0.25); tiny stop, large limit
  2. very_high_lmt    — LMT=40-120 (R2 A01 only went to 40 step=2; 42-120 untested)
  3. wide_se          — SE=4-10 (R2 max was 4 coarsely; new high-SE regime)
  4. deep_le          — LE=-2.5 to -10 (R2 min was -2.5-3 coarsely; deep mean-reversion entry)
  5. low_lmt_scalp    — LMT=1-8 + STP=0.05-0.5 (very tight scalp regime, high win-rate hypothesis)
  6. tight_stp_hi_lmt — STP=0.1-0.5 + LMT=18-60 (tight stop + aggressive limit)
  7. global_r3        — wide sweep with LMT up to 65, STP as low as 0.1
  8. wide_se_fine     — fine-tune around R2 A09's result (LEN=17 LE=-2 SE=4 STP=1.5 LMT=30 NP=398K)
  A09-A11 adaptive zooms

Attempt schedule (8 fixed + 3 adaptive, ≤5,000 combos each):
  A01 ultra_tight_stp : LEN(1-5 s1)   ×LE(-2-0.5 s0.5) ×SE(0.5-2.5 s0.5) ×STP(0.05-0.25 s0.05)×LMT(15-25 s5) = 2250
  A02 very_high_lmt   : LEN(2-4 s1)   ×LE(-1.5-0 s0.5) ×SE(1.0-2.0 s0.5) ×STP(0.5-2.0 s0.5)  ×LMT(40-120 s10)= 1296
  A03 wide_se         : LEN(3-15 s3)  ×LE(-3-0 s1)     ×SE(4-10 s1)      ×STP(0.5-2.5 s0.5)  ×LMT(20-45 s5)  = 4200
  A04 deep_le         : LEN(2-5 s1)   ×LE(-10--2.5 s0.5)×SE(1.0-2.0 s0.5)×STP(0.5-1.5 s0.5)  ×LMT(15-25 s5)  = 1728
  A05 low_lmt_scalp   : LEN(2-5 s1)   ×LE(-2-0 s1)     ×SE(1.0-2.5 s0.5) ×STP(0.05-0.5 s0.05) ×LMT(1-8 s1)   = 3840
  A06 tight_stp_hi_lmt: LEN(2-4 s1)   ×LE(-1.5-0 s0.5) ×SE(1.0-2.0 s0.5) ×STP(0.1-0.5 s0.1)  ×LMT(18-60 s6)  = 1440
  A07 global_r3       : LEN(3-30 s5)  ×LE(-3-2 s1)     ×SE(0.5-5 s1)     ×STP(0.1-2.5 s1.2)  ×LMT(15-65 s10) = 3240
  A08 wide_se_fine    : LEN(8-20 s3)  ×LE(-3--1 s0.5)  ×SE(3-6 s0.5)     ×STP(0.75-2.25 s0.5) ×LMT(20-45 s5)  = 4200
  A09-A11 adaptive zooms
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260523SFJ_BASIC_OSC_AI.wsp"
SYMBOL     = "CME.NQ HOT"
SIGNAL     = "_2021Basic_Osc_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\nq_osc_hourly3_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 800_000.0   # USD

LEN_LO, LEN_HI = 1.0,   200.0
LE_LO,  LE_HI  = -10.0, 10.0
SE_LO,  SE_HI  = 0.0,   10.0
STP_LO, STP_HI = 0.01,  50.0
LMT_LO, LMT_HI = 0.5,   200.0

# R1+R2 confirmed champion
SEED_LEN, SEED_LE  = 3.0,  -1.25
SEED_SE,  SEED_STP = 1.5,   1.0
SEED_LMT           = 18.5
SEED_NP            = 453_610.0

PREFIX = "NQOSCH3_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_nq_osc_hourly3_{int(time.time())}.log"
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
         len_:  Tuple[float, float, float],
         le:    Tuple[float, float, float],
         se:    Tuple[float, float, float],
         stp:   Tuple[float, float, float],
         lmt:   Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t, lo, hi):
        s, e, step = t
        if s == e:
            return (max(lo, s - step), min(hi, s + step), step)
        return t

    len_ = _safe(len_, LEN_LO, LEN_HI)
    le   = _safe(le,   LE_LO,  LE_HI)
    se   = _safe(se,   SE_LO,  SE_HI)
    stp  = _safe(stp,  STP_LO, STP_HI)
    lmt  = _safe(lmt,  LMT_LO, LMT_HI)

    combos = n_vals(len_) * n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"{PREFIX}{name}",
        mc_signal_name=SIGNAL,
        timeframe="minute",
        bar_period=60,
        params=[
            ParamAxis("LEN", *len_),
            ParamAxis("LE",  *le),
            ParamAxis("SE",  *se),
            ParamAxis("STP", *stp),
            ParamAxis("LMT", *lmt),
        ],
        chart_workspace=WORKSPACE,
        chart_symbol=SYMBOL,
        insample=INSAMPLE,
    )


def csv_for(name: str) -> Path:
    return OUTPUT_DIR / f"{PREFIX}{name}_raw.csv"


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
    log.info("=== Starting %s%s (%d combos) ===", PREFIX, name, cfg.total_runs())
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


def champion(df, fb_len, fb_le, fb_se, fb_stp, fb_lmt):
    """Priority: target met → highest NP (target-chasing mode)."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LEN"]), float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best["TotalTrades"]))
        return (float(best["LEN"]), float(best["LE"]), float(best["SE"]),
                float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), True)

    pos = df[df["NetProfit"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        log.info("  NP-Champion: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 float(best["LEN"]), float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["Objective"]), float(best["NetProfit"]),
                 float(best["MaxDrawdown"]), int(best["TotalTrades"]))
        return (float(best["LEN"]), float(best["LE"]), float(best["SE"]),
                float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best["TotalTrades"]), False)

    best = df.loc[df["NetProfit"].idxmax()]
    return (fb_len, fb_le, fb_se, fb_stp, fb_lmt,
            0.0, float(best["NetProfit"]), float(best["MaxDrawdown"]),
            int(best["TotalTrades"]), False)


def _entry(attempt, name, df, len_, le, se, stp, lmt, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "LEN": len_, "LE": le, "SE": se, "STP": stp, "LMT": lmt,
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
        "timeframe": "60min",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":        "BUY when C crosses over BollingerBand(C,LEN,LE); SHORT when C crosses under BollingerBand(C,LEN,SE)",
            "exits":        "STP×ATR(10) stop + LMT×ATR(10) limit",
            "r1r2_champion": "LEN=3 LE=-1.25 SE=1.5 STP=1.0 LMT=18.5 NP=453,610 MDD=-62,280 obj=3,303,822 trades=1427",
            "r3_focus":     "Unexplored: ultra-tight STP(0.05-0.25), LMT(40-150), wide SE(4-10), deep LE(-3 to -10), low-LMT scalp(1-8)",
            "ceiling_r1r2": "$453,610 confirmed after 24 attempts — R3 probing genuinely new regimes",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_nq_osc_hourly3.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_len = SEED_LEN
    best_le  = SEED_LE
    best_se  = SEED_SE
    best_stp = SEED_STP
    best_lmt = SEED_LMT
    best_np  = SEED_NP
    best_obj = 0.0
    target_met  = False
    attempt_log: List[dict] = []
    best_entry:  Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  NQ 60-min _2021Basic_Osc_NQ  NP>800K — Round 3")
    log.info("  Symbol: %s  Signal: %s", SYMBOL, SIGNAL)
    log.info("  R1+R2 ceiling: LEN=3 LE=-1.25 SE=1.5 STP=1.0 LMT=18.5  NP=$453,610")
    log.info("  R3 explores: ultra-tight STP, LMT>40, SE>4, LE<-2.5, low-LMT scalp")
    log.info("  Target: %.0f USD  (gap: %.0f)", TARGET_NP, TARGET_NP - SEED_NP)
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_len, best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_len, best_le, best_se, best_stp, best_lmt,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-22s]  no valid data", attempt_num, name)
            return

        len_, le, se, stp, lmt, obj, np_, mdd, tr, met = champion(
            df, best_len, best_le, best_se, best_stp, best_lmt)

        if np_ > best_np:
            best_len = len_
            best_le  = le
            best_se  = se
            best_stp = stp
            best_lmt = lmt
            best_np  = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, len_, le, se, stp, lmt,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-22s]  LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  tr=%d  %s",
                 attempt_num, name, len_, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/800K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  ultra_tight_stp — STP=0.05-0.25 (never tested below 0.25 in R1+R2)
    #      Hypothesis: ultra-small stop → minimal loss per loser → different win/loss dynamic
    #      LEN(1-5 s1) × LE(-2-0.5 s0.5) × SE(0.5-2.5 s0.5) × STP(0.05-0.25 s0.05) × LMT(15-25 s5)
    #      = 5×6×5×5×3 = 2,250
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_ultra_tight_stp"
    _c = _cfg(_n, (1, 5, 1), (-2.0, 0.5, 0.5), (0.5, 2.5, 0.5), (0.05, 0.25, 0.05), (15, 25, 5))
    log.info("A01  LEN(1-5 s1) × LE(-2-0.5 s0.5) × SE(0.5-2.5 s0.5) × STP(0.05-0.25 s0.05) × LMT(15-25 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  very_high_lmt — LMT=40-120 (R2 A01 stopped at 40; 42-120 untested)
    #      LEN(2-4 s1) × LE(-1.5-0 s0.5) × SE(1.0-2.0 s0.5) × STP(0.5-2.0 s0.5) × LMT(40-120 s10)
    #      = 3×4×3×4×9 = 1,296
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_very_high_lmt"
    _c = _cfg(_n, (2, 4, 1), (-1.5, 0.0, 0.5), (1.0, 2.0, 0.5), (0.5, 2.0, 0.5), (40, 120, 10))
    log.info("A02  LEN(2-4 s1) × LE(-1.5-0 s0.5) × SE(1.0-2.0 s0.5) × STP(0.5-2.0 s0.5) × LMT(40-120 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  wide_se — SE=4-10 (R2 max was SE=4; completely new high-SE regime)
    #      Hypothesis: rare extreme-overshoot entries → bigger wins per trade
    #      LEN(3-15 s3) × LE(-3-0 s1) × SE(4-10 s1) × STP(0.5-2.5 s0.5) × LMT(20-45 s5)
    #      = 5×4×7×5×6 = 4,200
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_wide_se"
    _c = _cfg(_n, (3, 15, 3), (-3.0, 0.0, 1.0), (4.0, 10.0, 1.0), (0.5, 2.5, 0.5), (20, 45, 5))
    log.info("A03  LEN(3-15 s3) × LE(-3-0 s1) × SE(4-10 s1) × STP(0.5-2.5 s0.5) × LMT(20-45 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  deep_le — LE=-2.5 to -10 (R2 min was ~-2.5; deep mean-reversion entry)
    #      Hypothesis: entering only on extreme dips below MA → higher win rate
    #      LEN(2-5 s1) × LE(-10--2.5 s0.5) × SE(1.0-2.0 s0.5) × STP(0.5-1.5 s0.5) × LMT(15-25 s5)
    #      = 4×16×3×3×3 = 1,728
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_deep_le"
    _c = _cfg(_n, (2, 5, 1), (-10.0, -2.5, 0.5), (1.0, 2.0, 0.5), (0.5, 1.5, 0.5), (15, 25, 5))
    log.info("A04  LEN(2-5 s1) × LE(-10--2.5 s0.5) × SE(1.0-2.0 s0.5) × STP(0.5-1.5 s0.5) × LMT(15-25 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  low_lmt_scalp — LMT=1-8 + STP=0.05-0.5 (ultra-scalp regime)
    #      Hypothesis: tight profit target (1-8×ATR) → high win rate →
    #                  many trades × small-but-consistent profit > 800K
    #      LEN(2-5 s1) × LE(-2-0 s1) × SE(1.0-2.5 s0.5) × STP(0.05-0.5 s0.05) × LMT(1-8 s1)
    #      = 4×3×4×10×8 = 3,840
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_low_lmt_scalp"
    _c = _cfg(_n, (2, 5, 1), (-2.0, 0.0, 1.0), (1.0, 2.5, 0.5), (0.05, 0.5, 0.05), (1, 8, 1))
    log.info("A05  LEN(2-5 s1) × LE(-2-0 s1) × SE(1.0-2.5 s0.5) × STP(0.05-0.5 s0.05) × LMT(1-8 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  tight_stp_hi_lmt — STP=0.1-0.5 + LMT=18-60 (cut losses fast, let wins run)
    #      LEN(2-4 s1) × LE(-1.5-0 s0.5) × SE(1.0-2.0 s0.5) × STP(0.1-0.5 s0.1) × LMT(18-60 s6)
    #      = 3×4×3×5×8 = 1,440
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_tight_stp_hi_lmt"
    _c = _cfg(_n, (2, 4, 1), (-1.5, 0.0, 0.5), (1.0, 2.0, 0.5), (0.1, 0.5, 0.1), (18, 60, 6))
    log.info("A06  LEN(2-4 s1) × LE(-1.5-0 s0.5) × SE(1.0-2.0 s0.5) × STP(0.1-0.5 s0.1) × LMT(18-60 s6)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  global_r3 — wide sweep with STP as low as 0.1 and LMT up to 65
    #      LEN(3-30 s5) × LE(-3-2 s1) × SE(0.5-5 s1) × STP(0.1-2.5 s1.2) × LMT(15-65 s10)
    #      = 6×6×6×3×6 = 3,888
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_global_r3"
    _c = _cfg(_n, (3, 30, 5), (-3.0, 2.0, 1.0), (0.5, 5.0, 1.0), (0.1, 2.5, 1.2), (15, 65, 10))
    log.info("A07  LEN(3-30 s5) × LE(-3-2 s1) × SE(0.5-5 s1) × STP(0.1-2.5 s1.2) × LMT(15-65 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  wide_se_fine — fine-tune around R2 A09's result (LEN=17 SE=4 LMT=30 NP=398K)
    #      Extends the wide-SE / high-LEN regime discovered in R2
    #      LEN(8-20 s3) × LE(-3--1 s0.5) × SE(3-6 s0.5) × STP(0.75-2.25 s0.5) × LMT(20-45 s5)
    #      = 5×5×7×4×6 = 4,200
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_wide_se_fine"
    _c = _cfg(_n, (8, 20, 3), (-3.0, -1.0, 0.5), (3.0, 6.0, 0.5), (0.75, 2.25, 0.5), (20, 45, 5))
    log.info("A08  LEN(8-20 s3) × LE(-3--1 s0.5) × SE(3-6 s0.5) × STP(0.75-2.25 s0.5) × LMT(20-45 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — zoom around best NP found so far
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (3, 0.5, 0.5, 0.5, 4.0),
            (2, 0.5, 0.5, 0.25, 3.0),
            (2, 0.25, 0.25, 0.25, 2.5),
            (1, 0.25, 0.25, 0.25, 2.0),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.25, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.25, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.25, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
            _c = _cfg(_n, _len, _le, _se, _stp, _lmt)
            if _c.total_runs() <= 5000:
                break
        log.info("  A09 zoom: LEN%s LE%s SE%s STP%s LMT%s → %d combos",
                 _len, _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2 — second zoom with finer steps
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (2, 0.5, 0.5, 0.25, 3.0),
            (2, 0.25, 0.25, 0.25, 2.5),
            (1, 0.25, 0.25, 0.25, 2.0),
            (1, 0.25, 0.25, 0.125, 1.5),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.25, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.25, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.25, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
            _c = _cfg(_n, _len, _le, _se, _stp, _lmt)
            if _c.total_runs() <= 5000:
                break
        log.info("  A10 zoom: LEN%s LE%s SE%s STP%s LMT%s → %d combos",
                 _len, _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3 — final fine zoom
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (2, 0.5, 0.5, 0.25, 2.5),
            (1, 0.25, 0.25, 0.25, 2.0),
            (1, 0.25, 0.25, 0.125, 1.5),
            (1, 0.25, 0.25, 0.125, 1.0),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.25, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.25, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.25, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,  LMT_LO, LMT_HI)
            _c = _cfg(_n, _len, _le, _se, _stp, _lmt)
            if _c.total_runs() <= 5000:
                break
        log.info("  A11 zoom: LEN%s LE%s SE%s STP%s LMT%s → %d combos",
                 _len, _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A11 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ROUND 3 COMPLETE")
    log.info("  Best NP: %.0f  (target: %.0f  gap: %.0f)",
             best_np, TARGET_NP, TARGET_NP - best_np)
    if best_entry:
        log.info("  Champion: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  MDD=%.0f  trades=%d",
                 best_entry.get("LEN"), best_entry.get("LE"), best_entry.get("SE"),
                 best_entry.get("STP"), best_entry.get("LMT"),
                 best_entry.get("objective", 0),
                 best_entry.get("max_drawdown", 0),
                 best_entry.get("total_trades", 0))
    log.info("  TARGET %s", "MET ✅" if target_met else "NOT MET")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LEN": best_len, "LE": best_le, "SE": best_se,
            "STP": best_stp, "LMT": best_lmt,
            "net_profit": best_np, "max_drawdown": 0,
            "objective": best_obj, "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>800K USD: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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


def main():
    ap = argparse.ArgumentParser(description="NQ Osc 60-min _2021Basic_Osc_NQ NP>800K Round-3 search")
    ap.add_argument("--from-csv",  action="store_true",
                    help="Re-analyse existing CSVs without launching MC64")
    ap.add_argument("--attempt",   type=int, default=1, metavar="N",
                    help="Resume from attempt N (1–11)")
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
