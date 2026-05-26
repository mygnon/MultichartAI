"""
search_gc_osc_daily5.py — _2021Basic_Osc_NQ on CME.GC HOT Daily, Round 5

R4 champion: LEN=8 LE=-0.4 SE=2.2 STP=1.8 LMT=7.0 NP=$365,690 MDD=-$38,660 trades=29
Round progression: R1=$279,350 → R2=$325,750(+16.6%) → R3=$352,550(+8.2%) → R4=$365,690(+3.7%)
Gain halving: 16.6%→8.2%→3.7% → ceiling ~$370-380K likely; $400K may be structurally unreachable

R5 design — 8 fixed attempts + 3 adaptive zooms (11 total), ≤5,000 combos each:
  A01 se_step05     : LEN(7-9 s1) × LE(-0.5--0.3 s0.1)    × SE(2.0-2.4 s0.05)    × STP(1.7-2.0 s0.1)  × LMT(6.5-7.5 s0.5) = 3×3×9×4×3  =   972
  A02 le_step002    : LEN(7-9 s1) × LE(-0.55--0.25 s0.02)  × SE(2.1-2.3 s0.05)   × STP(1.7-2.0 s0.1)  × LMT(6.5-7.5 s0.5) = 3×16×5×4×3 = 2,880
  A03 stp_step005   : LEN(7-9 s1) × LE(-0.5--0.3 s0.1)    × SE(2.1-2.3 s0.05)    × STP(1.6-2.1 s0.05) × LMT(6.5-7.5 s0.5) = 3×3×5×11×3 = 1,485
  A04 lmt_step025   : LEN(7-9 s1) × LE(-0.5--0.3 s0.1)    × SE(2.1-2.3 s0.05)    × STP(1.7-2.0 s0.1)  × LMT(6.0-8.0 s0.25)= 3×3×5×4×9  = 1,620
  A05 len_fine      : LEN(6-11 s1) × LE(-0.5--0.3 s0.1)   × SE(2.1-2.3 s0.05)    × STP(1.7-2.0 s0.1)  × LMT(6.5-7.5 s0.5) = 6×3×5×4×3  = 1,080
  A06 global_fine2  : LEN(2-16 s2) × LE(-0.8-0 s0.2)      × SE(1.5-3.0 s0.5)     × STP(1.5-2.5 s0.5)  × LMT(6-8 s1)       = 8×5×4×3×3  = 1,440
  A07 all_fine      : LEN(7-9 s1) × LE(-0.55--0.25 s0.05)  × SE(2.05-2.35 s0.05)  × STP(1.65-1.95 s0.05)× LMT(6.5-7.5 s0.5)= 3×7×7×7×3  = 3,087
  A08 deeper_le     : LEN(7-9 s1) × LE(-0.8--0.1 s0.05)   × SE(2.1-2.3 s0.05)    × STP(1.7-2.0 s0.1)  × LMT(6.5-7.5 s0.5) = 3×15×5×4×3 = 2,700
  A09-A11 adaptive zooms (if A09=A10=A11 and gain<2% → ceiling confirmed)
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
SYMBOL     = "CME.GC HOT"
SIGNAL     = "_2021Basic_Osc_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\gc_osc_daily5_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 400_000.0   # USD

LEN_LO, LEN_HI = 1.0,   200.0
LE_LO,  LE_HI  = -10.0, 10.0
SE_LO,  SE_HI  = 0.0,   10.0
STP_LO, STP_HI = 0.01,  50.0
LMT_LO, LMT_HI = 0.5,   200.0

# R4 champion as seed
SEED_LEN, SEED_LE  = 8.0, -0.4
SEED_SE,  SEED_STP = 2.2,  1.8
SEED_LMT           = 7.0
SEED_NP            = 365_690.0

PREFIX = "GCOSD5_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_gc_osc_daily5_{int(time.time())}.log"
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
        timeframe="daily",
        bar_period=1440,
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
        "round": 5,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Daily (1440 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY when C crosses over BollingerBand(C,LEN,LE); SHORT when C crosses under BollingerBand(C,LEN,SE)",
            "exits":       "STP×ATR(10) stop + LMT×ATR(10) limit",
            "r4_champion": "LEN=8 LE=-0.4 SE=2.2 STP=1.8 LMT=7.0 NP=$365,690 MDD=-$38,660 trades=29",
            "r5_focus":    "Ultra-fine LE(step=0.02)/SE(step=0.05)/STP(step=0.05)/LMT(step=0.25) + ceiling confirmation; gain 16.6%→8.2%→3.7% halving pattern",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_gc_osc_daily5.json"
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
    log.info("  GC Daily _2021Basic_Osc_NQ  NP>400K — Round 5")
    log.info("  Symbol: %s  Signal: %s  Timeframe: Daily", SYMBOL, SIGNAL)
    log.info("  R4 seed: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             SEED_LEN, SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP)
    log.info("  Target: %.0f USD  (gain pattern: 16.6%%→8.2%%→3.7%%; ceiling likely ~$370-380K)",
             TARGET_NP)
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
            best_len = len_; best_le = le; best_se = se
            best_stp = stp; best_lmt = lmt; best_np = np_
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
                 "★TARGET★" if met else ("%.0f/400K" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  se_step05 — ultra-fine SE step=0.05 to find exact SE peak
    #      LEN(7-9 s1) × LE(-0.5--0.3 s0.1) × SE(2.0-2.4 s0.05) × STP(1.7-2.0 s0.1) × LMT(6.5-7.5 s0.5)
    #      = 3×3×9×4×3 = 972
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_se_step05"
    _c = _cfg(_n, (7, 9, 1), (-0.5, -0.3, 0.1), (2.0, 2.4, 0.05), (1.7, 2.0, 0.1), (6.5, 7.5, 0.5))
    log.info("A01  LEN(7-9 s1)×LE(-0.5--0.3 s0.1)×SE(2.0-2.4 s0.05)×STP(1.7-2.0 s0.1)×LMT(6.5-7.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  le_step002 — ultra-fine LE step=0.02
    #      LEN(7-9 s1) × LE(-0.55--0.25 s0.02) × SE(2.1-2.3 s0.05) × STP(1.7-2.0 s0.1) × LMT(6.5-7.5 s0.5)
    #      = 3×16×5×4×3 = 2,880
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_le_step002"
    _c = _cfg(_n, (7, 9, 1), (-0.55, -0.25, 0.02), (2.1, 2.3, 0.05), (1.7, 2.0, 0.1), (6.5, 7.5, 0.5))
    log.info("A02  LEN(7-9 s1)×LE(-0.55--0.25 s0.02)×SE(2.1-2.3 s0.05)×STP(1.7-2.0 s0.1)×LMT(6.5-7.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  stp_step005 — ultra-fine STP step=0.05
    #      LEN(7-9 s1) × LE(-0.5--0.3 s0.1) × SE(2.1-2.3 s0.05) × STP(1.6-2.1 s0.05) × LMT(6.5-7.5 s0.5)
    #      = 3×3×5×11×3 = 1,485
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_stp_step005"
    _c = _cfg(_n, (7, 9, 1), (-0.5, -0.3, 0.1), (2.1, 2.3, 0.05), (1.6, 2.1, 0.05), (6.5, 7.5, 0.5))
    log.info("A03  LEN(7-9 s1)×LE(-0.5--0.3 s0.1)×SE(2.1-2.3 s0.05)×STP(1.6-2.1 s0.05)×LMT(6.5-7.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  lmt_step025 — fine LMT step=0.25 to find if LMT=7.0 is exact peak
    #      LEN(7-9 s1) × LE(-0.5--0.3 s0.1) × SE(2.1-2.3 s0.05) × STP(1.7-2.0 s0.1) × LMT(6.0-8.0 s0.25)
    #      = 3×3×5×4×9 = 1,620
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_lmt_step025"
    _c = _cfg(_n, (7, 9, 1), (-0.5, -0.3, 0.1), (2.1, 2.3, 0.05), (1.7, 2.0, 0.1), (6.0, 8.0, 0.25))
    log.info("A04  LEN(7-9 s1)×LE(-0.5--0.3 s0.1)×SE(2.1-2.3 s0.05)×STP(1.7-2.0 s0.1)×LMT(6.0-8.0 s0.25)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  len_fine — LEN=6-11 full sweep with new champion knowledge
    #      LEN(6-11 s1) × LE(-0.5--0.3 s0.1) × SE(2.1-2.3 s0.05) × STP(1.7-2.0 s0.1) × LMT(6.5-7.5 s0.5)
    #      = 6×3×5×4×3 = 1,080
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_len_fine"
    _c = _cfg(_n, (6, 11, 1), (-0.5, -0.3, 0.1), (2.1, 2.3, 0.05), (1.7, 2.0, 0.1), (6.5, 7.5, 0.5))
    log.info("A05  LEN(6-11 s1)×LE(-0.5--0.3 s0.1)×SE(2.1-2.3 s0.05)×STP(1.7-2.0 s0.1)×LMT(6.5-7.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  global_fine2 — global sweep with current best knowledge
    #      LEN(2-16 s2) × LE(-0.8-0 s0.2) × SE(1.5-3.0 s0.5) × STP(1.5-2.5 s0.5) × LMT(6-8 s1)
    #      = 8×5×4×3×3 = 1,440
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_global_fine2"
    _c = _cfg(_n, (2, 16, 2), (-0.8, 0.0, 0.2), (1.5, 3.0, 0.5), (1.5, 2.5, 0.5), (6, 8, 1))
    log.info("A06  LEN(2-16 s2)×LE(-0.8-0 s0.2)×SE(1.5-3.0 s0.5)×STP(1.5-2.5 s0.5)×LMT(6-8 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  all_fine — simultaneous fine step on LE, SE, and STP
    #      LEN(7-9 s1) × LE(-0.55--0.25 s0.05) × SE(2.05-2.35 s0.05) × STP(1.65-1.95 s0.05) × LMT(6.5-7.5 s0.5)
    #      = 3×7×7×7×3 = 3,087
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_all_fine"
    _c = _cfg(_n, (7, 9, 1), (-0.55, -0.25, 0.05), (2.05, 2.35, 0.05), (1.65, 1.95, 0.05), (6.5, 7.5, 0.5))
    log.info("A07  LEN(7-9 s1)×LE(-0.55--0.25 s0.05)×SE(2.05-2.35 s0.05)×STP(1.65-1.95 s0.05)×LMT(6.5-7.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  deeper_le — explore wider LE range with ultra-fine step=0.05
    #      LEN(7-9 s1) × LE(-0.8--0.1 s0.05) × SE(2.1-2.3 s0.05) × STP(1.7-2.0 s0.1) × LMT(6.5-7.5 s0.5)
    #      = 3×15×5×4×3 = 2,700
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_deeper_le"
    _c = _cfg(_n, (7, 9, 1), (-0.8, -0.1, 0.05), (2.1, 2.3, 0.05), (1.7, 2.0, 0.1), (6.5, 7.5, 0.5))
    log.info("A08  LEN(7-9 s1)×LE(-0.8--0.1 s0.05)×SE(2.1-2.3 s0.05)×STP(1.7-2.0 s0.1)×LMT(6.5-7.5 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (2, 0.2,  0.2,  0.2,  1.5),
            (2, 0.15, 0.15, 0.15, 1.0),
            (1, 0.1,  0.1,  0.1,  0.5),
            (1, 0.08, 0.1,  0.05, 0.5),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.02, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.05, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.05, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.25, LMT_LO, LMT_HI)
            _c = _cfg(_n, _len, _le, _se, _stp, _lmt)
            if _c.total_runs() <= 5000:
                break
        log.info("  A09 zoom: LEN%s LE%s SE%s STP%s LMT%s → %d combos",
                 _len, _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  adaptive_zoom2
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_adaptive_zoom2"
    log.info("A10  adaptive_zoom2 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (2, 0.15, 0.15, 0.15, 1.0),
            (1, 0.1,  0.1,  0.1,  0.5),
            (1, 0.08, 0.1,  0.05, 0.5),
            (1, 0.06, 0.05, 0.05, 0.25),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.02, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.05, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.05, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.25, LMT_LO, LMT_HI)
            _c = _cfg(_n, _len, _le, _se, _stp, _lmt)
            if _c.total_runs() <= 5000:
                break
        log.info("  A10 zoom: LEN%s LE%s SE%s STP%s LMT%s → %d combos",
                 _len, _le, _se, _stp, _lmt, _c.total_runs())
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom3
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom3"
    log.info("A11  adaptive_zoom3 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (1, 0.1,  0.1,  0.1,  0.5),
            (1, 0.08, 0.1,  0.05, 0.5),
            (1, 0.06, 0.05, 0.05, 0.25),
            (1, 0.04, 0.05, 0.05, 0.25),
        ]:
            _len = zoom(best_len, r_len, 1.0,  LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.02, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.05, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.05, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.25, LMT_LO, LMT_HI)
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
    r5_gain_pct = (best_np - SEED_NP) / SEED_NP * 100
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ROUND 5 COMPLETE")
    log.info("  R4 seed: %.0f  R5 best: %.0f  gain: +%.1f%%", SEED_NP, best_np, r5_gain_pct)
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
    if r5_gain_pct < 2.0:
        log.info("  R5 gain <2%% — CEILING CONFIRMED at ~%.0f USD", best_np)
        log.info("  $400K structurally unreachable on GC Daily bars with this strategy")
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
    print(f"R5 gain: +{r5_gain_pct:.1f}% ({best_np:.0f} from {SEED_NP:.0f})")
    print(f"Target NP>400K USD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
    if r5_gain_pct < 2.0:
        print("Gain <2% — ceiling confirmed. $400K structurally unreachable.")
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
    ap = argparse.ArgumentParser(description="GC Daily _2021Basic_Osc_NQ NP>400K Round-5 search (ceiling confirmation)")
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
