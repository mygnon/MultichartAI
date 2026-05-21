"""
search_zw_daily.py — Breakout Daily NP > 700,000 on CBOT.ZW HOT, Round 1

ZW contract specs:
  CBOT.ZW HOT — Wheat futures (Chicago)
  5,000 bushels/contract, quoted in cents/bushel
  1 point = 1¢/bu × 5,000 = $50/contract
  Min tick = 0.25¢ = $12.50/contract
  Typical daily ATR: 15–30¢ ($750–$1,500/contract)
  To earn 700K USD over 7yr: need ~14,000 points ≈ 2,000 pts/year

No prior ZW daily data — this is Round 1, full exploratory survey.

Strategy: _2021Basic_Break_NQ, 4 parameters
  LE  — Long-entry breakout lookback (days)
  SE  — Short-entry breakout lookback (days)
  STP — Stop-loss (¢/bu)
  LMT — Profit-target (¢/bu)

WORKSPACE REQUIREMENT (must verify before running):
  Open 20260508_SFJ_BASIC_BREAK_AI.wsp in MC64 and confirm:
  1. "CBOT.ZW HOT - Daily" chart is open.
  2. _2021Basic_Break_NQ signal is applied (visible + checked in Format Signals).
  3. The strategy subchart (equity curve panel) is visible below the price bars.
     If not present, add the strategy to the chart, save the workspace.
  4. Right-click blank space NEXT TO the signal (not on price bars) to get
     "Optimize Strategy" in the context menu.

Attempt schedule (12 attempts, ≤5,000 combos each):
  A01 se_low       : LE(1-5 s1) × SE(5-50 s5)     × STP(10-50 s10) × LMT(10-50 s10)   5×10×5×5=1250
  A02 se_mid       : LE(1-5 s1) × SE(50-150 s10)   × STP(10-50 s10) × LMT(10-50 s10)   5×11×5×5=1375
  A03 se_high      : LE(1-4 s1) × SE(150-300 s25)  × STP(10-50 s10) × LMT(10-50 s10)   4×7×5×5=700
  A04 se_very_high : LE(1-3 s1) × SE(300-500 s50)  × STP(10-60 s10) × LMT(10-60 s10)   3×5×6×6=540
  A05 wide_stp     : LE(1-4 s1) × SE(10-150 s25)   × STP(100-500 s100)× LMT(20-80 s10) 4×7×5×7=980
  A06 tight_stp    : LE(1-4 s1) × SE(10-80 s10)    × STP(1-9 s1)    × LMT(5-30 s5)     4×8×9×6=1728
  A07 tiny_lmt     : LE(1-5 s1) × SE(10-100 s10)   × STP(5-25 s5)   × LMT(1-9 s1)      5×10×5×9=2250
  A08 large_lmt    : LE(1-3 s1) × SE(10-100 s25)   × STP(10-50 s10) × LMT(50-500 s50)  3×5×5×10=750
  A09 high_le      : LE(10-40 s5)× SE(20-100 s10)  × STP(10-40 s10) × LMT(10-40 s10)   7×9×4×4=1008
  A10 sub_tick_lmt : LE(1-3 s1) × SE(5-40 s5)      × STP(2-20 s2)   × LMT(0.5-4 s0.5)  3×8×10×8=1920
  A11 adaptive_zoom  (from best NP found so far)
  A12 wide_boundary: LE(1-15 s3)× SE(5-500 s50)    × STP(5-200 s50) × LMT(5-200 s50)   6×10×4×4=960
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

WORKSPACE  = r"C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp"
SYMBOL     = "CBOT.ZW HOT"
SIGNAL     = "_2021Basic_Break_NQ"
OUTPUT_DIR = Path(r"C:\Users\Tim\MultichartAI\results\zw_daily_search")
INSAMPLE   = DateRange("2019/01/01", "2026/01/01")

TARGET_NP = 700_000.0

STP_LO, STP_HI = 0.25, 2000.0
LMT_LO, LMT_HI = 0.25, 2000.0
SE_LO,  SE_HI  = 1.0,   500.0
LE_LO,  LE_HI  = 1.0,   100.0

# No prior ZW daily data — use neutral seed
SEED_LE,  SEED_SE  = 1.0, 50.0
SEED_STP, SEED_LMT = 20.0, 20.0
SEED_NP   = -1_000_000.0   # unknown — set very low so any result becomes best
SEED_OBJ  = 0.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = OUTPUT_DIR / f"search_zw_daily_{int(time.time())}.log"
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
         le:  Tuple[float, float, float],
         se:  Tuple[float, float, float],
         stp: Tuple[float, float, float],
         lmt: Tuple[float, float, float]) -> StrategyConfig:
    def _safe(t):
        s, e, step = t
        if s == e:
            return (max(LE_LO, s - step), min(LE_HI, s + step), step)
        return t
    le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)
    combos = n_vals(le) * n_vals(se) * n_vals(stp) * n_vals(lmt)
    if combos > 5000:
        log.warning("  %s: %d combos EXCEEDS 5000!", name, combos)
    return StrategyConfig(
        name=f"ZWD1_{name}",
        mc_signal_name=SIGNAL,
        timeframe="daily",
        bar_period=1440,
        params=[
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
    return OUTPUT_DIR / f"ZWD1_{name}_raw.csv"


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
    log.info("=== Starting ZWD1_%s (%d combos) ===", name, cfg.total_runs())
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


def champion(df, fb_le, fb_se, fb_stp, fb_lmt):
    """Priority: target met → positive NP → least-negative NP."""
    df = df.copy()
    df["Objective"] = plateau_mod.compute_objective(df)

    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        log.info("  ★ TARGET MET: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "NP=%.0f  MDD=%.0f  obj=%.0f  trades=%d",
                 float(best["LE"]), float(best["SE"]),
                 float(best["STP"]), float(best["LMT"]),
                 float(best["NetProfit"]), float(best["MaxDrawdown"]),
                 float(best["Objective"]), int(best.get("TotalTrades", 0)))
        return (float(best["LE"]), float(best["SE"]),
                float(best["STP"]), float(best["LMT"]),
                float(best["Objective"]), float(best["NetProfit"]),
                float(best["MaxDrawdown"]), int(best.get("TotalTrades", 0)), True)

    pos = df[df["Objective"] > 0]
    if not pos.empty:
        best = pos.loc[pos["NetProfit"].idxmax()]
        le  = float(best["LE"]);  se  = float(best["SE"])
        stp = float(best["STP"]); lmt = float(best["LMT"])
        np_ = float(best.get("NetProfit", 0))
        mdd = float(best.get("MaxDrawdown", 0))
        tr  = int(best.get("TotalTrades", 0))
        obj = float(best["Objective"])
        log.info("  NP-Champion: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d",
                 le, se, stp, lmt, obj, np_, mdd, tr)
        return le, se, stp, lmt, obj, np_, mdd, tr, False

    # All NP negative — track least-negative for zoom direction
    np_col = pd.to_numeric(df.get("NetProfit", pd.Series(dtype=float)), errors="coerce")
    if not np_col.isna().all():
        best = df.loc[np_col.idxmax()]
        le  = float(best["LE"]);  se  = float(best["SE"])
        stp = float(best["STP"]); lmt = float(best["LMT"])
        np_ = float(best.get("NetProfit", 0))
        mdd = float(best.get("MaxDrawdown", 0))
        tr  = int(best.get("TotalTrades", 0))
        log.info("  All-neg best: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "NP=%.0f  MDD=%.0f  trades=%d",
                 le, se, stp, lmt, np_, mdd, tr)
        return le, se, stp, lmt, 0.0, np_, mdd, tr, False

    return fb_le, fb_se, fb_stp, fb_lmt, 0.0, 0.0, 0.0, 0, False


def _entry(attempt, name, df, le, se, stp, lmt, obj, np_, mdd, trades, met, combos):
    return {
        "attempt": attempt, "name": name,
        "combos": combos, "rows": len(df) if df is not None else 0,
        "LE": le, "SE": se, "STP": stp, "LMT": lmt,
        "net_profit": round(np_, 0), "max_drawdown": round(mdd, 0),
        "objective":  round(obj, 0), "total_trades": trades,
        "target_met": met,
    }


def save_json(best, log_, met):
    above    = [e for e in log_ if e.get("net_profit", 0) >= TARGET_NP]
    best_np  = max(log_, key=lambda x: x.get("net_profit", -1e18), default={})
    best_obj = max(log_, key=lambda x: x.get("objective", 0), default={})
    payload = {
        "strategy":           "Breakout_ZW_Daily (target NP>700K round-1)",
        "symbol":             SYMBOL,
        "timeframe":          "Daily (1 day)",
        "insample":           f"{INSAMPLE.from_date} - {INSAMPLE.to_date}",
        "objective_function": "NetProfit² / |MaxDrawdown|",
        "target_np":          TARGET_NP,
        "target_met":         met,
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "best_params":        best,
        "best_np_attempt":    best_np,
        "best_obj_attempt":   best_obj,
        "attempts_above_target": above,
        "attempt_log":        log_,
    }
    out = OUTPUT_DIR / "final_params_zw_daily.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────────────────────

def run_search(conn, from_csv, start_attempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_le,  best_se  = SEED_LE,  SEED_SE
    best_stp, best_lmt = SEED_STP, SEED_LMT
    best_np  = SEED_NP
    best_obj = SEED_OBJ
    target_met = False
    attempt_log: List[dict] = []
    best_entry: Dict = {}

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ZW Daily Breakout NP>700K Round-1 — full exploratory survey")
    log.info("  Symbol: %s  Timeframe: daily (1440 min)", SYMBOL)
    log.info("  ZW: 1 pt = 1¢/bu × 5000 bu = $50/contract")
    log.info("  Daily ATR ~15-30¢ → STP/LMT in tens of cents for daily bars")
    log.info("  Target: %.0f USD", TARGET_NP)
    log.info("  Workspace: confirm CBOT.ZW HOT - Daily chart is open with strategy")
    log.info("══════════════════════════════════════════════════════════════")

    def _update(df, cfg, name, attempt_num, combos):
        nonlocal best_le, best_se, best_stp, best_lmt
        nonlocal best_np, best_obj, target_met, best_entry

        if df is None or df.empty or not _validate_df(df, cfg):
            attempt_log.append(_entry(attempt_num, name, df,
                                      best_le, best_se, best_stp, best_lmt,
                                      0, 0, 0, 0, False, combos))
            log.info("  [A%02d %-28s]  no valid data", attempt_num, name)
            return

        le, se, stp, lmt, obj, np_, mdd, tr, met = champion(
            df, best_le, best_se, best_stp, best_lmt)

        if np_ > best_np:
            best_le, best_se = le, se
            best_stp, best_lmt = stp, lmt
            best_np = np_
        if obj > best_obj:
            best_obj = obj
        if met:
            target_met = True

        e = _entry(attempt_num, name, df, le, se, stp, lmt,
                   obj, np_, mdd, tr, met, combos)
        attempt_log.append(e)

        if (not best_entry
                or (met and not best_entry.get("target_met"))
                or (met and e.get("objective", 0) > best_entry.get("objective", 0))
                or (not best_entry.get("target_met")
                    and e.get("net_profit", -1e18) > best_entry.get("net_profit", -1e18))):
            best_entry = e

        log.info("  [A%02d %-28s]  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  "
                 "obj=%.0f  NP=%.0f  MDD=%.0f  trades=%d  %s",
                 attempt_num, name, le, se, stp, lmt, obj, np_, mdd, tr,
                 "★TARGET★" if met else ("%.0f/700K" % np_))
        log.info("       Global best NP=%.0f (gap %+.1f)",
                 best_np, TARGET_NP - best_np)

        save_json(best_entry if best_entry else e, attempt_log, target_met)

    # ──────────────────────────────────────────────────────────────────────
    # A01  se_low — SE=5–50 step 5, medium STP/LMT
    #      LE(1-5 s1) × SE(5-50 s5) × STP(10-50 s10) × LMT(10-50 s10) = 5×10×5×5=1250
    # ──────────────────────────────────────────────────────────────────────
    A = 1
    _n = "01_se_low"
    _c = _cfg(_n, (1, 5, 1), (5, 50, 5), (10.0, 50.0, 10.0), (10.0, 50.0, 10.0))
    log.info("A01  LE(1-5 s1) × SE(5-50 s5) × STP(10-50 s10) × LMT(10-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A01 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A02  se_mid — SE=50–150 step 10, medium STP/LMT
    #      LE(1-5 s1) × SE(50-150 s10) × STP(10-50 s10) × LMT(10-50 s10) = 5×11×5×5=1375
    # ──────────────────────────────────────────────────────────────────────
    A = 2
    _n = "02_se_mid"
    _c = _cfg(_n, (1, 5, 1), (50, 150, 10), (10.0, 50.0, 10.0), (10.0, 50.0, 10.0))
    log.info("A02  LE(1-5 s1) × SE(50-150 s10) × STP(10-50 s10) × LMT(10-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A02 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A03  se_high — SE=150–300 step 25, medium STP/LMT
    #      LE(1-4 s1) × SE(150-300 s25) × STP(10-50 s10) × LMT(10-50 s10) = 4×7×5×5=700
    # ──────────────────────────────────────────────────────────────────────
    A = 3
    _n = "03_se_high"
    _c = _cfg(_n, (1, 4, 1), (150, 300, 25), (10.0, 50.0, 10.0), (10.0, 50.0, 10.0))
    log.info("A03  LE(1-4 s1) × SE(150-300 s25) × STP(10-50 s10) × LMT(10-50 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A03 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A04  se_very_high — SE=300–500 step 50
    #      LE(1-3 s1) × SE(300-500 s50) × STP(10-60 s10) × LMT(10-60 s10) = 3×5×6×6=540
    # ──────────────────────────────────────────────────────────────────────
    A = 4
    _n = "04_se_very_high"
    _c = _cfg(_n, (1, 3, 1), (300, 500, 50), (10.0, 60.0, 10.0), (10.0, 60.0, 10.0))
    log.info("A04  LE(1-3 s1) × SE(300-500 s50) × STP(10-60 s10) × LMT(10-60 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A04 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A05  wide_stp — STP=100–500¢ (essentially no stop — let winners run)
    #      LE(1-4 s1) × SE(10-150 s25) × STP(100-500 s100) × LMT(20-80 s10) = 4×7×5×7=980
    # ──────────────────────────────────────────────────────────────────────
    A = 5
    _n = "05_wide_stp"
    _c = _cfg(_n, (1, 4, 1), (10, 150, 25), (100.0, 500.0, 100.0), (20.0, 80.0, 10.0))
    log.info("A05  LE(1-4 s1) × SE(10-150 s25) × STP(100-500 s100) × LMT(20-80 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A05 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A06  tight_stp — STP=1–9¢ (very tight stops, sub-ATR)
    #      LE(1-4 s1) × SE(10-80 s10) × STP(1-9 s1) × LMT(5-30 s5) = 4×8×9×6=1728
    # ──────────────────────────────────────────────────────────────────────
    A = 6
    _n = "06_tight_stp"
    _c = _cfg(_n, (1, 4, 1), (10, 80, 10), (1.0, 9.0, 1.0), (5.0, 30.0, 5.0))
    log.info("A06  LE(1-4 s1) × SE(10-80 s10) × STP(1-9 s1) × LMT(5-30 s5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A06 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A07  tiny_lmt — LMT=1–9¢ (tight profit-taking, similar to CL daily pattern)
    #      LE(1-5 s1) × SE(10-100 s10) × STP(5-25 s5) × LMT(1-9 s1) = 5×10×5×9=2250
    # ──────────────────────────────────────────────────────────────────────
    A = 7
    _n = "07_tiny_lmt"
    _c = _cfg(_n, (1, 5, 1), (10, 100, 10), (5.0, 25.0, 5.0), (1.0, 9.0, 1.0))
    log.info("A07  LE(1-5 s1) × SE(10-100 s10) × STP(5-25 s5) × LMT(1-9 s1)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A07 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A08  large_lmt — LMT=50–500¢ (run profits very wide)
    #      LE(1-3 s1) × SE(10-100 s25) × STP(10-50 s10) × LMT(50-500 s50) = 3×5×5×10=750
    # ──────────────────────────────────────────────────────────────────────
    A = 8
    _n = "08_large_lmt"
    _c = _cfg(_n, (1, 3, 1), (10, 100, 25), (10.0, 50.0, 10.0), (50.0, 500.0, 50.0))
    log.info("A08  LE(1-3 s1) × SE(10-100 s25) × STP(10-50 s10) × LMT(50-500 s50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A08 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A09  high_le — LE=10–40 days (wide channel, trend-following entry)
    #      LE(10-40 s5) × SE(20-100 s10) × STP(10-40 s10) × LMT(10-40 s10) = 7×9×4×4=1008
    # ──────────────────────────────────────────────────────────────────────
    A = 9
    _n = "09_high_le"
    _c = _cfg(_n, (10, 40, 5), (20, 100, 10), (10.0, 40.0, 10.0), (10.0, 40.0, 10.0))
    log.info("A09  LE(10-40 s5) × SE(20-100 s10) × STP(10-40 s10) × LMT(10-40 s10)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A09 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A10  sub_tick_lmt — LMT=0.5–4¢ (very small profit target, like CL daily LMT=1)
    #      LE(1-3 s1) × SE(5-40 s5) × STP(2-20 s2) × LMT(0.5-4 s0.5) = 3×8×10×8=1920
    # ──────────────────────────────────────────────────────────────────────
    A = 10
    _n = "10_sub_tick_lmt"
    _c = _cfg(_n, (1, 3, 1), (5, 40, 5), (2.0, 20.0, 2.0), (0.5, 4.0, 0.5))
    log.info("A10  LE(1-3 s1) × SE(5-40 s5) × STP(2-20 s2) × LMT(0.5-4 s0.5)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A10 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A11  adaptive_zoom — zoom around current best NP, progressive shrink
    # ──────────────────────────────────────────────────────────────────────
    A = 11
    _n = "11_adaptive_zoom"
    log.info("A11  adaptive_zoom — center: LE=%.4g SE=%.4g STP=%.4g LMT=%.4g  NP=%.0f",
             best_le, best_se, best_stp, best_lmt, best_np)
    if start_attempt <= A:
        _le  = (best_le,  best_le,  1.0)
        _se  = (best_se,  best_se,  5.0)
        _stp = (best_stp, best_stp, 5.0)
        _lmt = (best_lmt, best_lmt, 5.0)
        cfg11 = None
        for r_le, r_se, r_stp, r_lmt in [
            (3, 30, 20, 20), (2, 20, 15, 15), (2, 15, 10, 10), (1, 10, 5, 5)
        ]:
            _le  = zoom(best_le,  r_le,  1.0, LE_LO,  LE_HI)
            _se  = zoom(best_se,  r_se,  5.0, SE_LO,  SE_HI)
            _stp = zoom(best_stp, r_stp, 5.0, STP_LO, STP_HI)
            _lmt = zoom(best_lmt, r_lmt, 5.0, LMT_LO, LMT_HI)
            cfg11 = _cfg(_n, _le, _se, _stp, _lmt)
            if cfg11.total_runs() <= 5000:
                break
        if cfg11 is not None:
            log.info("A11  LE%s × SE%s × STP%s × LMT%s  %d combos",
                     _le, _se, _stp, _lmt, cfg11.total_runs())
            _update(run_or_load(_n, cfg11, conn, from_csv), cfg11, _n, A, cfg11.total_runs())
    log.info("After A11 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # A12  wide_boundary — coarse global sweep to catch any missed region
    #      LE(1-15 s3) × SE(5-500 s50) × STP(5-200 s50) × LMT(5-200 s50) = 6×10×4×4=960
    # ──────────────────────────────────────────────────────────────────────
    A = 12
    _n = "12_wide_boundary"
    _c = _cfg(_n, (1, 16, 3), (5, 505, 50), (5.0, 205.0, 50.0), (5.0, 205.0, 50.0))
    log.info("A12  LE(1-16 s3) × SE(5-505 s50) × STP(5-205 s50) × LMT(5-205 s50)  %d combos",
             _c.total_runs())
    if start_attempt <= A:
        _update(run_or_load(_n, _c, conn, from_csv), _c, _n, A, _c.total_runs())
    log.info("After A12 — best NP=%.0f  (LE=%.4g SE=%.4g STP=%.4g LMT=%.4g)",
             best_np, best_le, best_se, best_stp, best_lmt)

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  ZW Daily Round-1 COMPLETE")
    log.info("  Best NP: %.0f  LE=%.4g SE=%.4g STP=%.4g LMT=%.4g",
             best_np, best_le, best_se, best_stp, best_lmt)
    log.info("  Target %.0f: %s", TARGET_NP, "★ MET" if target_met else f"NOT MET (gap +{TARGET_NP - best_np:.0f})")
    log.info("══════════════════════════════════════════════════════════════")

    if not best_entry:
        best_entry = {
            "LE": best_le, "SE": best_se, "STP": best_stp, "LMT": best_lmt,
            "net_profit": best_np, "max_drawdown": 0, "objective": best_obj,
            "total_trades": 0, "target_met": target_met,
        }

    out = save_json(best_entry, attempt_log, target_met)
    print(f"\nDone — results at: {out}")
    print(f"Target NP>700K: {'MET' if target_met else 'NOT MET — best NP=%.0f' % best_np}")
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
    ap = argparse.ArgumentParser(description="ZW Daily Breakout NP>700K Round-1 search")
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
