"""
search_txf_osc_hourly4.py — _2021Basic_Osc_NQ on TWF.TXF HOT Hourly, Round 4

R1: LEN=11 LE=-1.25 SE=3.0 STP=1.0 LMT=36.0  NP=5,506,400 TWD  (+baseline)
R2: LEN=11 LE=-1.25 SE=3.0 STP=1.0 LMT=33.0  NP=5,755,000 TWD  (+4.5%)
R3: LEN=11 LE=-1.30 SE=3.0 STP=1.0 LMT=33.0  NP=5,901,600 TWD  (+2.5%)
Target: NP > 7,000,000 TWD  (gap from R3: +1,098,400 = +15.7%)

R3 confirmed (locked):
  LEN=11  (fine step=1 range 7-16 R2; range 10-30 R3 → LEN=11 wins)
  SE=3.0  (SE push 3-7 s0.5 R2 → SE=3 wins; SE below 3 step=0.1 R3 → SE=3 wins)
  STP=1.0 (fine step=0.1 range 0.5-2.0 R2 → STP=1.0 wins)

R3 refined:
  LE=-1.30 (fine step=0.1 range -2.0 to -0.5 R3 A02 → -1.30 wins vs -1.25)
  LMT=33   (fine step=1 range 25-45 R3 A01 → LMT=33 wins)

R4 unexplored:
  LE step=0.05 around -1.30  (only step=0.1 done so far)
  LMT step=0.5 around 33     (only step=1 done so far)
  SE step=0.05 above 3.0 (3.0-3.5; R2 tested 3.0/3.5/4.0/... s0.5 → 3.0 wins; but 3.1-3.4 untested)
  STP step=0.025 around 1.0  (only step=0.1 done so far)
  LEN above 11: A07 R3 found LEN=18 NP=4.9M — still lower; but step=1 scan of LEN=12-22 unexplored

R4 strategy:
  A01 le_ultrafine   : LEN(10-12 s1)×LE(-1.6-(-1.0) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(31-35 s1) = 3×13×3×3×5 = 1,755
  A02 lmt_ultrafine  : LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(28-40 s0.5) = 3×5×3×3×25 = 3,375
  A03 se_fine_above3 : LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(3.0-3.5 s0.05)×STP(0.9-1.1 s0.1)×LMT(31-35 s1) = 3×5×11×3×5 = 2,475
  A04 stp_ultrafine  : LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.9-3.1 s0.1)×STP(0.85-1.15 s0.025)×LMT(31-35 s1) = 3×5×3×13×5 = 2,925
  A05 le_lmt_joint   : LEN(10-12 s1)×LE(-1.5-(-1.1) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(30-36 s0.5) = 3×9×3×3×13 = 3,159
  A06 se_fine_below3 : LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.6-3.0 s0.05)×STP(0.9-1.1 s0.1)×LMT(31-35 s1) = 3×5×9×3×5 = 2,025
  A07 len_explore    : LEN(12-25 s1)×LE(-1.5-(-1.1) s0.1) ×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(31-35 s1) = 14×5×3×3×5 = 3,150
  A08 global_r4      : LEN(5-25 s5)×LE(-2-0 s0.5)×SE(1.5-3.5 s0.5)×STP(0.5-2.5 s1)×LMT(20-45 s5) = 5×5×5×3×6 = 2,250
  A09-A11 adaptive zooms (LE step=0.025, LMT step=0.5, SE step=0.05, STP step=0.05)
  Ceiling rule: if R4 gain <2% AND A09=A10=A11 same → declare ceiling; 7M structurally unreachable
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
SYMBOL     = "TWF.TXF HOT"
SIGNAL     = "_2021Basic_Osc_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\txf_osc_hourly4_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP  = 7_000_000.0   # TWD

LEN_LO, LEN_HI = 1.0,   500.0
LE_LO,  LE_HI  = -10.0, 10.0
SE_LO,  SE_HI  = 0.0,   10.0
STP_LO, STP_HI = 0.01,  50.0
LMT_LO, LMT_HI = 0.5,   200.0

# R3 champion as seed
SEED_LEN, SEED_LE  = 11.0, -1.30
SEED_SE,  SEED_STP = 3.0,   1.0
SEED_LMT           = 33.0
SEED_NP            = 5_901_600.0

PREFIX = "TXFOSH4_"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_txf_osc_hourly4_{int(time.time())}.log"
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
        timeframe="hourly",
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
        "round": 4,
        "strategy": SIGNAL,
        "symbol": SYMBOL,
        "timeframe": "Hourly (60 min)",
        "target_np": TARGET_NP,
        "target_met": target_met,
        "notes": {
            "logic":       "BUY when C crosses over BollingerBand(C,LEN,LE); SHORT when C crosses under BollingerBand(C,LEN,SE)",
            "exits":       "STP×ATR(10) stop + LMT×ATR(10) limit",
            "r3_champion": "LEN=11 LE=-1.30 SE=3.0 STP=1.0 LMT=33.0 NP=5,901,600 TWD MDD=-872,800 trades=558",
            "r4_focus":    "Ultra-fine LE(step=0.05) + LMT(step=0.5) + SE above 3.0(step=0.05) + STP(step=0.025) + LE×LMT joint + SE below 3.0 fine + LEN 12-25 + global",
        },
        "best_params":  best_entry,
        "all_attempts": attempt_log,
    }
    out = OUTPUT_DIR / "final_params_txf_osc_hourly4.json"
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
    log.info("  TXF Hourly _2021Basic_Osc_NQ  NP>7,000,000 TWD — Round 4")
    log.info("  Symbol: %s  Signal: %s  Timeframe: Hourly (60 min)", SYMBOL, SIGNAL)
    log.info("  R3 seed: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             SEED_LEN, SEED_LE, SEED_SE, SEED_STP, SEED_LMT, SEED_NP)
    log.info("  Target: %.0f TWD  (R3 gap: -15.7%%  R2→R3 gain: +2.5%%)", TARGET_NP)
    log.info("  LOCKED: LEN=11, SE=3.0, STP=1.0  NEW: LE=-1.30, LMT=33")
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
                 "★TARGET★" if met else ("%.0f/7M" % np_))
        log.info("       Global best NP=%.0f  gap=%.0f TWD",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  le_ultrafine — LE step=0.05 around -1.30 (R3 used step=0.1)
    #      LEN(10-12 s1)×LE(-1.6-(-1.0) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(31-35 s1)
    #      = 3×13×3×3×5 = 1,755
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_le_ultrafine"
    _c = _cfg(_n, (10, 12, 1), (-1.6, -1.0, 0.05), (2.9, 3.1, 0.1), (0.9, 1.1, 0.1), (31, 35, 1))
    log.info("A01  LEN(10-12 s1)×LE(-1.6-(-1.0) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(31-35 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  lmt_ultrafine — LMT step=0.5 around 33 (R3 used step=1)
    #      LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(28-40 s0.5)
    #      = 3×5×3×3×25 = 3,375
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_lmt_ultrafine"
    _c = _cfg(_n, (10, 12, 1), (-1.4, -1.2, 0.05), (2.9, 3.1, 0.1), (0.9, 1.1, 0.1), (28, 40, 0.5))
    log.info("A02  LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(28-40 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  se_fine_above3 — SE step=0.05 from 3.0 to 3.5 (R2 only tested s0.5: 3.0/3.5/4.0...)
    #      LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(3.0-3.5 s0.05)×STP(0.9-1.1 s0.1)×LMT(31-35 s1)
    #      = 3×5×11×3×5 = 2,475
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_se_fine_above3"
    _c = _cfg(_n, (10, 12, 1), (-1.4, -1.2, 0.05), (3.0, 3.5, 0.05), (0.9, 1.1, 0.1), (31, 35, 1))
    log.info("A03  LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(3.0-3.5 s0.05)×STP(0.9-1.1 s0.1)×LMT(31-35 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  stp_ultrafine — STP step=0.025 around 1.0 (R2 used step=0.1)
    #      LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.9-3.1 s0.1)×STP(0.85-1.15 s0.025)×LMT(31-35 s1)
    #      = 3×5×3×13×5 = 2,925
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_stp_ultrafine"
    _c = _cfg(_n, (10, 12, 1), (-1.4, -1.2, 0.05), (2.9, 3.1, 0.1), (0.85, 1.15, 0.025), (31, 35, 1))
    log.info("A04  LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.9-3.1 s0.1)×STP(0.85-1.15 s0.025)×LMT(31-35 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  le_lmt_joint — fine both LE(step=0.05) and LMT(step=0.5) simultaneously
    #      LEN(10-12 s1)×LE(-1.5-(-1.1) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(30-36 s0.5)
    #      = 3×9×3×3×13 = 3,159
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_le_lmt_joint"
    _c = _cfg(_n, (10, 12, 1), (-1.5, -1.1, 0.05), (2.9, 3.1, 0.1), (0.9, 1.1, 0.1), (30, 36, 0.5))
    log.info("A05  LEN(10-12 s1)×LE(-1.5-(-1.1) s0.05)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(30-36 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  se_fine_below3 — SE step=0.05 from 2.6 to 3.0 (R3 tested 2.4-3.0 at step=0.1)
    #      LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.6-3.0 s0.05)×STP(0.9-1.1 s0.1)×LMT(31-35 s1)
    #      = 3×5×9×3×5 = 2,025
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_se_fine_below3"
    _c = _cfg(_n, (10, 12, 1), (-1.4, -1.2, 0.05), (2.6, 3.0, 0.05), (0.9, 1.1, 0.1), (31, 35, 1))
    log.info("A06  LEN(10-12 s1)×LE(-1.4-(-1.2) s0.05)×SE(2.6-3.0 s0.05)×STP(0.9-1.1 s0.1)×LMT(31-35 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  len_explore — fine LEN(12-25) with R3 best LE/LMT (R3 A07 used step=3)
    #      LEN(12-25 s1)×LE(-1.5-(-1.1) s0.1)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(31-35 s1)
    #      = 14×5×3×3×5 = 3,150
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_len_explore"
    _c = _cfg(_n, (12, 25, 1), (-1.5, -1.1, 0.1), (2.9, 3.1, 0.1), (0.9, 1.1, 0.1), (31, 35, 1))
    log.info("A07  LEN(12-25 s1)×LE(-1.5-(-1.1) s0.1)×SE(2.9-3.1 s0.1)×STP(0.9-1.1 s0.1)×LMT(31-35 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  global_r4 — fresh global sweep (updated SE≤3.5, guided by locked params)
    #      LEN(5-25 s5)×LE(-2-0 s0.5)×SE(1.5-3.5 s0.5)×STP(0.5-2.5 s1)×LMT(20-45 s5)
    #      = 5×5×5×3×6 = 2,250
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_global_r4"
    _c = _cfg(_n, (5, 25, 5), (-2.0, 0.0, 0.5), (1.5, 3.5, 0.5), (0.5, 2.5, 1.0), (20, 45, 5))
    log.info("A08  LEN(5-25 s5)×LE(-2-0 s0.5)×SE(1.5-3.5 s0.5)×STP(0.5-2.5 s1)×LMT(20-45 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_len, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  adaptive_zoom1 — LE step=0.025, LMT step=0.5, SE step=0.05, STP step=0.05
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_adaptive_zoom1"
    log.info("A09  adaptive_zoom1 — center: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_len, best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        for r_len, r_le, r_se, r_stp, r_lmt in [
            (1, 0.10, 0.10, 0.05, 2.0),
            (1, 0.075, 0.10, 0.05, 1.5),
            (1, 0.05, 0.075, 0.05, 1.0),
            (1, 0.05, 0.05, 0.025, 0.75),
        ]:
            _len = zoom(best_len, r_len, 1.0,   LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.025, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.05,  SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.05,  STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,   LMT_LO, LMT_HI)
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
            (1, 0.075, 0.10, 0.05, 1.5),
            (1, 0.05,  0.075, 0.05, 1.0),
            (1, 0.05,  0.05, 0.025, 0.75),
            (1, 0.025, 0.05, 0.025, 0.5),
        ]:
            _len = zoom(best_len, r_len, 1.0,   LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.025, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.05,  SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.05,  STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,   LMT_LO, LMT_HI)
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
            (1, 0.05,  0.075, 0.05, 1.0),
            (1, 0.05,  0.05, 0.025, 0.75),
            (1, 0.025, 0.05, 0.025, 0.5),
            (1, 0.025, 0.025, 0.025, 0.5),
        ]:
            _len = zoom(best_len, r_len, 1.0,   LEN_LO, LEN_HI)
            _le  = zoom(best_le,  r_le,  0.025, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  0.05,  SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 0.05,  STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 0.5,   LMT_LO, LMT_HI)
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
    r4_gain_pct = (best_np - SEED_NP) / SEED_NP * 100 if SEED_NP > 0 else 0
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ROUND 4 COMPLETE")
    log.info("  R3 seed: %.0f  R4 best: %.0f  gain: +%.1f%%", SEED_NP, best_np, r4_gain_pct)
    log.info("  Best NP: %.0f TWD  (target: %.0f  gap: %.0f)",
             best_np, TARGET_NP, TARGET_NP - best_np)
    if best_entry:
        log.info("  Champion: LEN=%.4g LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  MDD=%.0f  trades=%d",
                 best_entry.get("LEN"), best_entry.get("LE"), best_entry.get("SE"),
                 best_entry.get("STP"), best_entry.get("LMT"),
                 best_entry.get("objective", 0),
                 best_entry.get("max_drawdown", 0),
                 best_entry.get("total_trades", 0))
    # zoom convergence check: compare A09/A10/A11 net_profit
    zooms = [e for e in attempt_log if e["name"] in
             ("09_adaptive_zoom1", "10_adaptive_zoom2", "11_adaptive_zoom3")]
    zoom_nps = [e.get("net_profit", 0) for e in zooms if e.get("net_profit", 0) > 0]
    zoom_converged = len(zoom_nps) >= 2 and len(set(zoom_nps)) == 1
    if r4_gain_pct < 2.0 and zoom_converged:
        log.info("  *** CEILING CONFIRMED: R4 gain <2%% AND A09=A10=A11 converged ***")
        log.info("  7,000,000 TWD target is structurally unreachable with this strategy/symbol/timeframe")
    elif r4_gain_pct < 2.0:
        log.info("  R4 gain <2%% — approaching ceiling; one more round if zooms not converged")
    log.info("  TARGET %s", "MET ✅" if target_met else "NOT MET — best NP=%.0f" % best_np)
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
    print(f"R4 gain: +{r4_gain_pct:.1f}% ({best_np:.0f} from {SEED_NP:.0f})")
    print(f"Target NP>7,000,000 TWD: {'MET ✅' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
    if r4_gain_pct < 2.0 and zoom_converged:
        print("CEILING CONFIRMED: R4 gain <2% AND zooms converged. 7M TWD structurally unreachable.")
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
    ap = argparse.ArgumentParser(description="TXF Hourly _2021Basic_Osc_NQ NP>7M TWD Round-4 search")
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
