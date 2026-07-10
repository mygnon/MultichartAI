"""
run_ertrail_allinst_pipeline.py -- ONE-BAT unattended 4-stage pipeline for
ERTrailBreakout run across SIX instruments living in ONE workspace
(ERTrailBreakout_AI.wsp), switching to the correct chart Window per instrument.

  crypto charts (signal ERTrailBreakout_crypto):  BTCUSDT, ETHUSDT, BNBUSDT
  futures charts (signal ERTrailBreakout_NQ):      TWF.TXF, CME.NQ, CME.GC

Per instrument: Stage 1 IS optimization (self-judged convergence) -> Stage 2 OOS champion-select
-> Stage 3 exit-module IS optimization (6 modules) -> Stage 4 cumulative greedy full-period RoMaD
stack + write final params back into Format Objects Input Strings.

Strategy params (4): Length, BandMult, ATRMult, ReentryBars (Donchian entry + ER-adaptive chandelier trail (BandMult = trail ER-sensitivity 0-1)
). Date ranges by asset class. Each instrument writes results/<sub>_ertrail_hourly_pipeline/
state.json (+ CSVs).

KEY: all 6 charts share one MC workspace window. mc.activate_chart_by_symbol() MDI-activates +
MAXIMIZES the target instrument's chart before every wizard/dialog op (read-back verify, ABORT on
mismatch). set_signal_statuses retries the M5/M6 checkbox-revert + dialog-open flakiness. Each
instrument re-connects (refresh stale MC handle after multi-hour runs); Stage-4 teardown is best-effort.

CLI:
  py run_ertrail_allinst_pipeline.py                    # all 6 instruments x 4 stages (auto-elevates)
  py run_ertrail_allinst_pipeline.py --probe-windows    # read-only: dump open chart titles
  py run_ertrail_allinst_pipeline.py --instrument "BTCUSDT HOT" [--from-stage N]
  py run_ertrail_allinst_pipeline.py --resume-gaps      # re-run any instrument missing a complete stage4
  py run_ertrail_allinst_pipeline.py --from-csv         # re-analyse existing CSVs, no MC64
"""
from __future__ import annotations
import argparse, ctypes, json, logging, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import mc_automation as mc
import plateau as plateau_mod
from config import DateRange, ParamAxis, StrategyConfig

WORKSPACE = r"C:\Users\Tim\Downloads\Multichart64\Tim\ERTrailBreakout_AI.wsp"
INSAMPLE  = DateRange("2017/01/01", "2027/01/01")
TF, BARP  = "hourly", 60
RESULTS   = Path(r"C:\Users\Tim\MultichartAI\results")

CRYPTO_IS, CRYPTO_FULL = ("2022/01/01", "2026/01/01"), ("2021/03/01", "2026/06/10")
FUT_IS,    FUT_FULL    = ("2019/01/01", "2025/01/01"), ("2018/01/01", "2026/01/01")
SIG_CRYPTO = "ERTrailBreakout_crypto"
SIG_NQ     = "ERTrailBreakout_NQ"

# (key, symbol, match-tokens, signal, is_range, full_range, target, out_subdir)
INSTRUMENTS = [
    ("btc", "BTCUSDT HOT", ["BTCUSDT"], SIG_CRYPTO, CRYPTO_IS, CRYPTO_FULL, 100_000.0, "btc_ertrail_hourly_pipeline"),
    ("eth", "ETHUSDT HOT", ["ETHUSDT"], SIG_CRYPTO, CRYPTO_IS, CRYPTO_FULL,  10_000.0, "eth_ertrail_hourly_pipeline"),
    ("bnb", "BNBUSDT HOT", ["BNBUSDT"], SIG_CRYPTO, CRYPTO_IS, CRYPTO_FULL, 100_000.0, "bnb_ertrail_hourly_pipeline"),
    ("txf", "TWF.TXF HOT", ["TXF"],     SIG_NQ,     FUT_IS,    FUT_FULL, 10_000_000.0, "txf_ertrail_hourly_pipeline"),
    ("nq",  "CME.NQ HOT",  ["CME.NQ"],  SIG_NQ,     FUT_IS,    FUT_FULL, 10_000_000.0, "nq_ertrail_hourly_pipeline"),
    ("gc",  "CME.GC HOT",  ["CME.GC"],  SIG_NQ,     FUT_IS,    FUT_FULL, 10_000_000.0, "gc_ertrail_hourly_pipeline"),
]

# ---- 4-param search space (Length, BandMult, ATRMult, ReentryBars) ----
LEN_LO, LEN_HI   = 2.0, 300.0
BAND_LO, BAND_HI = 0.0, 1.0     # trail ER-sensitivity; 0 = fixed chandelier
ATR_LO, ATR_HI   = 0.5, 20.0
RE_LO,  RE_HI    = 0.0, 50.0
SEED = (20.0, 0.5, 7.0, 13.0)   # Length, BandMult, ATRMult, ReentryBars
CONV_THRESH = 0.005
MIN_ROUNDS, MAX_ROUNDS = 2, 5
P = ("Length", "BandMult", "ATRMult", "ReentryBars")
BOUNDS = {"Length": (LEN_LO, LEN_HI), "BandMult": (BAND_LO, BAND_HI),
          "ATRMult": (ATR_LO, ATR_HI), "ReentryBars": (RE_LO, RE_HI)}

MODULES: List[Tuple[str, str, List[Tuple[str, float, float, float]]]] = [
    ("M1", "SFJ_15Dworkshop_lesson4_ATRstop",                 [("STP",      0.1,   100.0, 0.1)]),
    ("M2", "SFJ_15Dworkshop_lesson9_1_TrailingStop",          [("ATRSTP",   0.1,   100.0, 0.1)]),
    ("M3", "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit",   [("EXITBAR",  1.0,   1000.0, 1.0)]),
    ("M4", "SFJ_15Dworkshop_lesson11_3_high_volatility_exit", [("DAYRANGE", 0.01,  10.0,  0.01)]),
    ("M5", "QuantPass_PT_Exit",                               [("PT_Base",  0.001, 1.0,   0.001)]),
    ("M6", "RescueTeamExit",                                  [("Length",   20.0,  600.0, 20.0),
                                                              ("std",       3.0,   6.0,   0.1)]),
]
ALL_MODULE_NAMES = [m[1] for m in MODULES]
# Live-trading signals that MUST be OFF during optimization/backtest — leaving
# them ON blocks dialogs / interferes with runs (user-diagnosed BNB/ETH failure).
# Missing-on-chart is tolerated (ensure-OFF is then trivially satisfied).
EXTRA_OFF_SIGNALS = ["WinTrade_TradeMode", "*_OrderMasterTXT"]
STATUS_ALL = ALL_MODULE_NAMES + EXTRA_OFF_SIGNALS
MODULE_PARAM_NAMES = {sig: {p[0] for p in axes} for (_, sig, axes) in MODULES}
VARY_AXIS = {"M1": ("STP", 0.1), "M2": ("ATRSTP", 0.1), "M3": ("EXITBAR", 1.0),
             "M4": ("DAYRANGE", 0.01), "M5": ("PT_Base", 0.001), "M6": ("std", 0.1)}
EPS = 1e-9
FORCE = False   # --force: ignore cached *_raw.csv and re-run the optimization

SYMBOL = MAIN_SIGNAL = None
TOKENS: List[str] = []
IS_RANGE = FULL_RANGE = None
TARGET_NP = 0.0
OUTPUT_DIR: Path = RESULTS / "ertrail_allinst_pipeline"
STATE_JSON: Path = OUTPUT_DIR / "state.json"

RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "ertrail_allinst_pipeline").mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s -- %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(str(RESULTS / "ertrail_allinst_pipeline" /
                                                      f"allinst_{int(time.time())}.log"), encoding="utf-8")])
log = logging.getLogger(__name__)


def _apply_ctx(inst):
    global SYMBOL, MAIN_SIGNAL, TOKENS, IS_RANGE, FULL_RANGE, TARGET_NP, OUTPUT_DIR, STATE_JSON
    key, symbol, tokens, signal, isr, full, target, sub = inst
    SYMBOL, MAIN_SIGNAL, TOKENS = symbol, signal, tokens
    IS_RANGE, FULL_RANGE, TARGET_NP = isr, full, target
    OUTPUT_DIR = RESULTS / sub
    STATE_JSON = OUTPUT_DIR / "state.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- helpers
def _seed_cfg():
    return StrategyConfig(name="MIDCH_seed", mc_signal_name=MAIN_SIGNAL, timeframe=TF, bar_period=BARP,
                          params=[ParamAxis("Length", 18, 22, 2), ParamAxis("BandMult", 0.5, 1.5, 0.5),
                                  ParamAxis("ATRMult", 6.5, 7.5, 0.5), ParamAxis("ReentryBars", 11, 13, 1)],
                          chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)


def _activate(conn):
    if conn is None:
        return
    mc.ensure_chart_ready(conn, _seed_cfg())
    mc.activate_chart_by_symbol(conn, SYMBOL, TOKENS)


def _save_state(state):
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_state():
    if STATE_JSON.exists():
        try:
            return json.load(open(STATE_JSON, encoding="utf-8"))
        except Exception:
            pass
    return {"stage1": None, "stage2": None, "stage3": None, "stage4": None}


def _snap(v, step):
    return round(round(v / step) * step, 8)


def n_vals(s, e, st):
    return max(1, round((e - s) / st) + 1)


def _safe(s, e, st, lo, hi):
    if abs(s - e) < st * 0.5:
        return (max(lo, s - st), min(hi, s + st), st)
    return (s, e, st)


def _main_cfg(name, L, B, A, Re):
    L = _safe(*L, LEN_LO, LEN_HI); B = _safe(*B, BAND_LO, BAND_HI)
    A = _safe(*A, ATR_LO, ATR_HI); Re = _safe(*Re, RE_LO, RE_HI)
    combos = n_vals(*L) * n_vals(*B) * n_vals(*A) * n_vals(*Re)
    if combos > 5000:
        log.warning("  %s: %d combos >5000!", name, combos)
    return StrategyConfig(name=name, mc_signal_name=MAIN_SIGNAL, timeframe=TF, bar_period=BARP,
                          params=[ParamAxis("Length", *L), ParamAxis("BandMult", *B),
                                  ParamAxis("ATRMult", *A), ParamAxis("ReentryBars", *Re)],
                          chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)


def _module_cfg(name, signal, axes, fixed_inputs=None):
    combos = 1
    for (_, s, e, st) in axes:
        combos *= n_vals(s, e, st)
    if combos > 5000:
        log.warning("  %s: %d combos >5000!", name, combos)
    return StrategyConfig(name=name, mc_signal_name=signal, timeframe=TF, bar_period=BARP,
                          params=[ParamAxis(p, s, e, st) for (p, s, e, st) in axes],
                          chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE, fixed_inputs=fixed_inputs)


def csv_for(name):
    return OUTPUT_DIR / f"{name}_raw.csv"


def run_or_load(cfg, conn, from_csv):
    p = csv_for(cfg.name)
    if from_csv or (p.exists() and not FORCE):
        if p.exists():
            try:
                df = mc.load_results_csv(str(p), cfg); log.info("Loaded %s: %d rows", cfg.name, len(df))
                return df
            except Exception as e:
                log.warning("load %s: %s", p, e)
        return None
    log.info("=== run %s (%d combos) ===", cfg.name, cfg.total_runs())
    t0 = time.time()
    for attempt in (1, 2):
        try:
            _activate(conn)
        except Exception as e:
            log.warning("_activate: %s", e)
        try:
            raw = mc.run_optimization_for_strategy(conn, cfg, str(OUTPUT_DIR))
            log.info("  done %.1f min", (time.time() - t0) / 60)
            return mc.load_results_csv(raw, cfg)
        except Exception as e:
            log.warning("  attempt %d failed: %s", attempt, e)
    return None


def _with_obj(df):
    df = df.copy(); df["Objective"] = plateau_mod.compute_objective(df); return df


def _row_dict(row):
    return {"Length": float(row["Length"]), "BandMult": float(row["BandMult"]),
            "ATRMult": float(row["ATRMult"]), "ReentryBars": float(row["ReentryBars"]),
            "net_profit": float(row["NetProfit"]), "max_drawdown": float(row["MaxDrawdown"]),
            "objective": float(row["Objective"]), "total_trades": int(row["TotalTrades"])}


def _valid_main(df):
    for nm in P:
        lo, hi = BOUNDS[nm]
        if nm not in df.columns:
            return False
        col = pd.to_numeric(df[nm], errors="coerce")
        if not col.between(lo - 1e6, hi + 1e6).all():
            return False
    return True


# ----------------------------------------------------------------------------- Stage 1
def _stage1_round(label, attempts, seed, conn, from_csv):
    rows: List[dict] = []
    best = {"objective": -1e18}
    bL, bB, bA, bRe = seed
    for nm, L, B, A, Re in attempts:
        cfg = _main_cfg(f"S1_{label}_{nm}", L, B, A, Re)
        df = run_or_load(cfg, conn, from_csv)
        if df is None or df.empty or not _valid_main(df):
            log.info("  [%s %s] no valid data", label, nm); continue
        d = _with_obj(df); pos = d[d["NetProfit"] > 0]
        if pos.empty:
            continue
        om = _row_dict(pos.loc[pos["Objective"].idxmax()])
        nm_ = _row_dict(pos.loc[pos["NetProfit"].idxmax()])
        rows.append(om); rows.append(nm_)
        if om["objective"] > best["objective"]:
            best = om; bL, bB, bA, bRe = om["Length"], om["BandMult"], om["ATRMult"], om["ReentryBars"]
        log.info("  [%s %-16s] Obj-max L=%.4g B=%.4g Atr=%.4g Re=%.4g NP=%.0f MDD=%.0f Obj=%.4g tr=%d",
                 label, nm, om["Length"], om["BandMult"], om["ATRMult"], om["ReentryBars"], om["net_profit"],
                 om["max_drawdown"], om["objective"], om["total_trades"])
    for zi, (rL, sL, rB, sB, rA, sA, rRe, sRe) in enumerate(
            [(4, 1, 0.375, 0.125, 1.5, 0.5, 4, 1),
             (3, 1, 0.25, 0.125, 1.0, 0.25, 3, 1),
             (2, 1, 0.25, 0.0625, 0.75, 0.25, 2, 1)], 1):
        Lz = (max(LEN_LO, _snap(bL - rL, sL)), min(LEN_HI, _snap(bL + rL, sL)), sL)
        Bz = (max(BAND_LO, _snap(bB - rB, sB)), min(BAND_HI, _snap(bB + rB, sB)), sB)
        Az = (max(ATR_LO, _snap(bA - rA, sA)), min(ATR_HI, _snap(bA + rA, sA)), sA)
        Rez = (max(RE_LO, _snap(bRe - rRe, sRe)), min(RE_HI, _snap(bRe + rRe, sRe)), sRe)
        cfg = _main_cfg(f"S1_{label}_zoom{zi}", Lz, Bz, Az, Rez)
        df = run_or_load(cfg, conn, from_csv)
        if df is None or df.empty or not _valid_main(df):
            continue
        d = _with_obj(df); pos = d[d["NetProfit"] > 0]
        if pos.empty:
            continue
        om = _row_dict(pos.loc[pos["Objective"].idxmax()]); rows.append(om)
        if om["objective"] > best["objective"]:
            best = om; bL, bB, bA, bRe = om["Length"], om["BandMult"], om["ATRMult"], om["ReentryBars"]
        log.info("  [%s zoom%d] Obj-max L=%.4g B=%.4g Atr=%.4g Re=%.4g Obj=%.4g", label, zi,
                 om["Length"], om["BandMult"], om["ATRMult"], om["ReentryBars"], om["objective"])
    return rows


def _confirm_attempts(sL, sB, sA, sRe):
    return [
        ("01_retest", (max(LEN_LO, sL-2), sL+2, 1), (max(BAND_LO, sB-0.5), sB+0.5, 0.25),
         (max(ATR_LO, sA-1), sA+1, 0.5), (max(RE_LO, sRe-2), sRe+2, 1)),
        ("02_len_fine", (max(LEN_LO, sL-8), sL+8, 1), (sB, sB, 0.25), (sA, sA, 0.5), (sRe, sRe, 1)),
        ("03_band_fine", (sL, sL, 1), (max(BAND_LO, sB-1), min(BAND_HI, sB+1), 0.125), (sA, sA, 0.5), (sRe, sRe, 1)),
        ("04_atr_fine", (sL, sL, 1), (sB, sB, 0.25), (max(ATR_LO, sA-3), min(ATR_HI, sA+3), 0.25), (sRe, sRe, 1)),
        ("05_re_fine", (sL, sL, 1), (sB, sB, 0.25), (sA, sA, 0.5), (0, 30, 1)),
        ("06_combo", (max(LEN_LO, sL-6), sL+6, 2), (max(BAND_LO, sB-0.75), min(BAND_HI, sB+0.75), 0.25),
         (max(ATR_LO, sA-2), min(ATR_HI, sA+2), 0.5), (max(RE_LO, sRe-4), sRe+4, 2)),
    ]


def stage1(conn, from_csv, state):
    log.info("############ [%s] STAGE 1: IS optimization ############", SYMBOL)
    if conn is not None:
        _activate(conn)
        mc.set_signal_statuses(conn, {n: False for n in STATUS_ALL}, verify=True, protected=[MAIN_SIGNAL])
        mc.set_instrument_data_range(conn, *IS_RANGE)
    r1_attempts = [
        ("01_global_wide", (5, 80, 5),   (0.0, 1.0, 0.25), (2.0, 14.0, 2.0), (0, 20, 10)),
        ("02_classic",     (10, 40, 3),  (0.0, 1.0, 0.2),  (5.0, 9.0, 1.0),  (10, 20, 5)),
        ("03_len_fine",    (2, 60, 2),   (0.25, 0.75, 0.25),(6.0, 8.0, 1.0),  (12, 16, 2)),
        ("04_band_fine",   (10, 30, 5),  (0.0, 1.0, 0.05), (6.0, 8.0, 1.0),  (12, 16, 2)),
        ("05_atr_fine",    (10, 30, 5),  (0.25, 0.75, 0.25),(3.0, 16.0, 0.5), (12, 16, 2)),
        ("06_reentry_fine",(10, 30, 5),  (0.25, 0.75, 0.25),(6.0, 8.0, 1.0),  (0, 30, 1)),
        ("07_wide_trail",  (5, 50, 5),   (0.0, 1.0, 0.25), (8.0, 20.0, 1.0), (5, 20, 5)),
        ("08_short_len",   (2, 20, 1),   (0.0, 1.0, 0.25), (5.0, 12.0, 1.0), (5, 20, 5)),
        ("09_global_bound",(5, 300, 20), (0.0, 1.0, 0.1),  (2.0, 18.0, 4.0), (0, 40, 20)),
    ]
    rows = _stage1_round("R1", r1_attempts, SEED, conn, from_csv)
    if not rows:
        log.error("[%s] Stage1 R1 produced no valid rows -- ABORT", SYMBOL); return None
    champ = max(rows, key=lambda r: r["objective"])
    conv = []
    for rnd in range(2, MAX_ROUNDS + 1):
        prev_obj = champ["objective"]
        s = (champ["Length"], champ["BandMult"], champ["ATRMult"], champ["ReentryBars"])
        rows += _stage1_round(f"R{rnd}", _confirm_attempts(*s), s, conn, from_csv)
        champ = max(rows, key=lambda r: r["objective"])
        gain = (champ["objective"] - prev_obj) / prev_obj if prev_obj > 0 else 1.0
        conv.append({"round": rnd, "prev_obj": prev_obj, "obj": champ["objective"], "gain_pct": round(gain * 100, 4)})
        log.info(">>> R%d Obj %.6g -> %.6g (gain %.3f%%)", rnd, prev_obj, champ["objective"], gain * 100)
        if rnd >= MIN_ROUNDS and gain <= CONV_THRESH:
            log.info(">>> CONVERGED at R%d", rnd); break
    npmax = max(rows, key=lambda r: r["net_profit"])
    cands = []; seen = set()
    def _key(r): return (round(r["Length"]), round(r["BandMult"], 3), round(r["ATRMult"], 2), round(r["ReentryBars"]))
    for r in [champ, npmax] + sorted(rows, key=lambda r: -r["objective"]):
        k = _key(r)
        if k in seen:
            continue
        seen.add(k); cands.append(r)
        if len(cands) >= 4:
            break
    state["stage1"] = {"symbol": SYMBOL, "champion_obj": champ, "champion_np": npmax,
                       "candidates": cands, "convergence": conv, "timestamp": datetime.now().isoformat()}
    _save_state(state)
    log.info(">>> [%s] Stage1 Obj-max: L=%.4g B=%.4g Atr=%.4g Re=%.4g NP=%.0f Obj=%.4g tr=%d", SYMBOL,
             champ["Length"], champ["BandMult"], champ["ATRMult"], champ["ReentryBars"],
             champ["net_profit"], champ["objective"], champ["total_trades"])
    return state


# ----------------------------------------------------------------------------- Stage 2
def _micro_main_cfg(name, L, B, A, Re):
    return StrategyConfig(name=name, mc_signal_name=MAIN_SIGNAL, timeframe=TF, bar_period=BARP,
                          params=[ParamAxis("Length", max(LEN_LO, L-1), L+1, 1.0),
                                  ParamAxis("BandMult", max(BAND_LO, B-0.25), B+0.25, 0.25),
                                  ParamAxis("ATRMult", max(ATR_LO, A-0.5), A+0.5, 0.5),
                                  ParamAxis("ReentryBars", max(RE_LO, Re-1), Re+1, 1.0)],
                          chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)


def _pick_main(df, L, B, A, Re):
    m = df
    for nm, v in (("Length", L), ("BandMult", B), ("ATRMult", A), ("ReentryBars", Re)):
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    return m.iloc[0] if not m.empty else None


def stage2(conn, from_csv, state):
    log.info("############ [%s] STAGE 2: OOS champion-select ############", SYMBOL)
    cands = state["stage1"]["candidates"]
    results = {"is": {}, "full": {}}
    for period, rng in (("is", IS_RANGE), ("full", FULL_RANGE)):
        if conn is not None:
            _activate(conn)
            mc.set_signal_statuses(conn, {n: False for n in STATUS_ALL}, verify=True, protected=[MAIN_SIGNAL])
            mc.set_instrument_data_range(conn, *rng)
        for i, c in enumerate(cands):
            cfg = _micro_main_cfg(f"S2_{period}_C{i}", c["Length"], c["BandMult"], c["ATRMult"], c["ReentryBars"])
            df = run_or_load(cfg, conn, from_csv)
            row = _pick_main(df, c["Length"], c["BandMult"], c["ATRMult"], c["ReentryBars"]) if df is not None and not df.empty else None
            if row is not None:
                results[period][str(i)] = {"net_profit": float(row["NetProfit"]),
                                           "max_drawdown": float(row["MaxDrawdown"]),
                                           "total_trades": int(row["TotalTrades"])}
                log.info("  [%s %s C%d] NP=%.0f MDD=%.0f", SYMBOL, period, i,
                         float(row["NetProfit"]), float(row["MaxDrawdown"]))
    rep = []
    for i, c in enumerate(cands):
        ci, cf = results["is"].get(str(i)), results["full"].get(str(i))
        if not ci or not cf:
            continue
        oos = cf["net_profit"] - ci["net_profit"]
        brk = abs(cf["max_drawdown"]) > abs(ci["max_drawdown"])
        rep.append({"idx": i, "params": {"Length": c["Length"], "BandMult": c["BandMult"],
                                         "ATRMult": c["ATRMult"], "ReentryBars": c["ReentryBars"]},
                    "np_is": ci["net_profit"], "mdd_is": ci["max_drawdown"],
                    "np_full": cf["net_profit"], "mdd_full": cf["max_drawdown"],
                    "oos_np": round(oos, 2), "mdd_break": brk, "pass": (not brk),
                    "break_ratio": round(abs(cf["max_drawdown"]) / abs(ci["max_drawdown"]), 3) if ci["max_drawdown"] else None})
    passing = [r for r in rep if r["pass"]]
    if passing:
        winner = max(passing, key=lambda r: r["oos_np"])
    elif rep:
        winner = sorted(rep, key=lambda r: (-r["oos_np"], r["break_ratio"] or 9))[0]
    else:
        log.error("[%s] Stage2 no valid candidates -- ABORT", SYMBOL); return None
    state["stage2"] = {"results": results, "report": rep, "winner": winner,
                       "main_champ": winner["params"], "timestamp": datetime.now().isoformat()}
    _save_state(state)
    log.info(">>> [%s] Stage2 WINNER %s OOS=%.0f pass=%s", SYMBOL, winner["params"], winner["oos_np"], winner["pass"])
    return state


# ----------------------------------------------------------------------------- set inputs
def _set_signal_inputs(conn, signal, params, commit=True):
    cfg = StrategyConfig(name=f"SET_{signal[:20]}", mc_signal_name=signal, timeframe=TF,
                         bar_period=BARP, params=[ParamAxis(k, v, v, 1.0) for k, v in params.items()],
                         chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)
    last = None
    for a in range(1, 4):
        try:
            mc.set_signal_input_strings(conn, signal, params, commit=commit); return
        except Exception as e:
            last = e; log.warning("_set_signal_inputs %s attempt %d failed: %s", signal, a, e)
            try:
                _activate(conn)
            except Exception:
                pass
            time.sleep(1.0)
    raise last


def _valid_main_fixed(df, champ, enabled_signals):
    collide = set()
    for s in enabled_signals:
        collide |= MODULE_PARAM_NAMES.get(s, set())
    for nm, v in champ.items():
        if nm in collide or nm not in df.columns:
            continue
        col = pd.to_numeric(df[nm], errors="coerce")
        tol = max(1e-6, abs(v) * 1e-4)
        if not ((col - v).abs() <= tol).all():
            log.error("  main %s NOT fixed at %.6g (got %.6g..%.6g)", nm, v, col.min(), col.max())
            return False
    return True


def _champ_tuple(champ):
    return (champ["Length"], champ["BandMult"], champ["ATRMult"], champ["ReentryBars"])


# ----------------------------------------------------------------------------- Stage 3
def stage3(conn, from_csv, state):
    log.info("############ [%s] STAGE 3: exit-module IS optimization ############", SYMBOL)
    champ = state["stage2"]["main_champ"]; ct = _champ_tuple(champ)
    if conn is not None:
        _activate(conn)
        mc.set_instrument_data_range(conn, *IS_RANGE)
        _set_signal_inputs(conn, MAIN_SIGNAL, champ)
        mc.set_signal_statuses(conn, {n: False for n in STATUS_ALL}, verify=True, protected=[MAIN_SIGNAL])
    df0 = run_or_load(_micro_main_cfg("S3_A00", *ct), conn, from_csv)
    row0 = _pick_main(df0, *ct) if df0 is not None and not df0.empty else None
    if row0 is None:
        log.error("[%s] Stage3 A00 baseline failed -- ABORT", SYMBOL); return None
    base_np = float(row0["NetProfit"]); base_mdd = float(row0["MaxDrawdown"])
    log.info("[%s] A00 IS baseline NP=%.2f MDD=%.2f tr=%d", SYMBOL, base_np, base_mdd, int(row0["TotalTrades"]))
    mods = {}
    for (label, signal, axes) in MODULES:
        if conn is not None:
            _activate(conn)
            mc.set_signal_statuses(conn, {n: (n == signal) for n in STATUS_ALL}, verify=True, protected=[MAIN_SIGNAL])
        cfg = _module_cfg(f"S3_{label}_{signal[:24]}", signal, axes,
                          fixed_inputs={MAIN_SIGNAL: champ})
        df = run_or_load(cfg, conn, from_csv)
        if df is None or df.empty or not _valid_main_fixed(df, champ, [signal]):
            log.error("  [%s] Stage3 %s INVALID -- ABORT", SYMBOL, label); return None
        d = _with_obj(df)
        best = d.loc[pd.to_numeric(d["NetProfit"], errors="coerce").idxmax()]
        params = {p[0]: float(best[p[0]]) for p in axes}
        np_ = float(best["NetProfit"]); mdd = float(best["MaxDrawdown"])
        mods[label] = {"signal": signal, "params": params, "net_profit": np_, "max_drawdown": mdd,
                       "total_trades": int(best["TotalTrades"]),
                       "delta_np_pct": round((np_ - base_np) / base_np * 100, 2) if base_np else None}
        log.info("  [%s] %s %s NP=%.0f (%+.2f%%) MDD=%.0f", SYMBOL, label, params, np_, mods[label]["delta_np_pct"] or 0, mdd)
    state["stage3"] = {"baseline": {"net_profit": base_np, "max_drawdown": base_mdd},
                       "modules": mods, "timestamp": datetime.now().isoformat()}
    _save_state(state)
    return state


# ----------------------------------------------------------------------------- Stage 4
def stage4(conn, from_csv, state):
    log.info("############ [%s] STAGE 4: cumulative greedy full-period OOS stack ############", SYMBOL)
    champ = state["stage2"]["main_champ"]; mods = state["stage3"]["modules"]; ct = _champ_tuple(champ)
    order = sorted(mods.items(), key=lambda kv: -(kv[1]["delta_np_pct"] or -1e9))
    if conn is not None:
        _activate(conn)
        mc.set_instrument_data_range(conn, *FULL_RANGE)
        _s4sigs = [(MAIN_SIGNAL, champ)] + [(info["signal"], info["params"]) for label, info in mods.items()]
        for _i, (_sig, _prm) in enumerate(_s4sigs):
            _set_signal_inputs(conn, _sig, _prm, commit=(_i == len(_s4sigs) - 1))
        mc.set_signal_statuses(conn, {n: False for n in STATUS_ALL}, verify=True, protected=[MAIN_SIGNAL])
    df0 = run_or_load(_micro_main_cfg("S4_A00", *ct), conn, from_csv)
    row0 = _pick_main(df0, *ct) if df0 is not None and not df0.empty else None
    if row0 is None:
        log.error("[%s] Stage4 A00 baseline failed -- ABORT", SYMBOL); return None
    cur_np = float(row0["NetProfit"]); cur_midd = float(row0["MaxDrawdown"])
    cur_romad = cur_np / abs(cur_midd) if cur_midd else 0.0
    log.info("[%s] A00 FULL baseline NP=%.2f MIDD=%.2f RoMaD=%.4f", SYMBOL, cur_np, cur_midd, cur_romad)
    kept: List[str] = []; steps = []
    for step, (label, info) in enumerate(order, 1):
        signal = info["signal"]; vax, vstep = VARY_AXIS[label]; vval = info["params"][vax]
        enabled = [mods[k]["signal"] for k in kept] + [signal]
        if conn is not None:
            _activate(conn)
            mc.set_signal_statuses(conn, {n: (n in enabled) for n in STATUS_ALL}, verify=True, protected=[MAIN_SIGNAL])
        _fx = {MAIN_SIGNAL: dict(champ)}
        for k in kept:
            _fx.setdefault(mods[k]["signal"], {}).update(mods[k]["params"])
        _cand_rest = {p: v for p, v in info["params"].items() if p != vax}
        if _cand_rest:
            _fx.setdefault(signal, {}).update(_cand_rest)
        cfg = StrategyConfig(name=f"S4_S{step}_{label}", mc_signal_name=signal, timeframe=TF, bar_period=BARP,
                             params=[ParamAxis(vax, max(0.0, round(vval - vstep, 8)), round(vval + vstep, 8), vstep)],
                             chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE,
                             fixed_inputs=_fx)
        df = run_or_load(cfg, conn, from_csv)
        row = None
        if df is not None and not df.empty and _valid_main_fixed(df, champ, enabled):
            m = df[(pd.to_numeric(df[vax], errors="coerce") - vval).abs() < 1e-6]
            row = m.iloc[0] if not m.empty else None
        ent = {"step": step, "candidate": label, "signal": signal, "enabled": enabled}
        if row is not None:
            np_ = float(row["NetProfit"]); midd = float(row["MaxDrawdown"])
            romad = np_ / abs(midd) if midd else 0.0
            keep = romad > cur_romad + EPS
            ent.update({"net_profit": np_, "max_intraday_drawdown": midd, "romad": round(romad, 4),
                        "prev_romad": round(cur_romad, 4),
                        "delta_romad_pct": round((romad - cur_romad) / cur_romad * 100, 2) if cur_romad else None,
                        "decision": "KEEP" if keep else "discard", "valid": True})
            log.info("  [%s] S%d %s NP=%.0f MIDD=%.0f RoMaD=%.4f (prev %.4f) -> %s",
                     SYMBOL, step, label, np_, midd, romad, cur_romad, ent["decision"])
            if keep:
                kept.append(label); cur_np, cur_midd, cur_romad = np_, midd, romad
        else:
            ent["valid"] = False
            log.error("  [%s] Stage4 S%d %s INVALID -- skipping", SYMBOL, step, label)
        steps.append(ent)
        state["stage4"] = {"baseline": {"net_profit": float(row0["NetProfit"]), "max_intraday_drawdown": float(row0["MaxDrawdown"])},
                           "steps": steps, "final_kept": [k for k in kept],
                           "final_romad": round(cur_romad, 4), "timestamp": datetime.now().isoformat()}
        _save_state(state)
    if conn is not None:
        try:        # best-effort write-back: results saved per step, stale-handle teardown must not fail a complete instrument
            _activate(conn)
            _t0 = time.time()
            _tdsigs = [(MAIN_SIGNAL, champ)] + [(mods[k]["signal"], mods[k]["params"]) for k in kept]
            for _i, (_sig, _prm) in enumerate(_tdsigs):
                _set_signal_inputs(conn, _sig, _prm, commit=(_i == len(_tdsigs) - 1))
                log.info("[TIMING] teardown inputs %s done (+%.1fs)", _sig, time.time() - _t0)
            kept_signals = {mods[k]["signal"] for k in kept}
            mc.set_signal_statuses(conn, {n: (n in kept_signals) for n in STATUS_ALL}, verify=True, protected=[MAIN_SIGNAL])
            for _i, (_sig, _prm) in enumerate(_tdsigs):   # post-statuses re-verify (fast path)
                _set_signal_inputs(conn, _sig, _prm, commit=(_i == len(_tdsigs) - 1))
            log.info("[TIMING] teardown inputs re-verified (+%.1fs)", time.time() - _t0)
            log.info("[%s] Final Format Objects inputs applied: main=%s kept=%s", SYMBOL, champ, {k: mods[k]["params"] for k in kept})
        except Exception as e:
            log.warning("[%s] Stage4 teardown write-back failed (results already saved): %s", SYMBOL, e)
    log.info(">>> [%s] Stage4 FINAL kept-set: %s  RoMaD %.4f", SYMBOL, kept or "NONE", cur_romad)
    return state


# ----------------------------------------------------------------------------- per-instrument driver

def apply_final_only(conn, state):
    """Re-apply the final deploy to the chart from state.json: write main champion
    + kept-module params into Format Objects inputs (read-back verified) and
    enable exactly the KEPT modules. Usable standalone via --apply-final."""
    champ = state["stage2"]["main_champ"]
    mods = state["stage3"]["modules"]
    kept = state["stage4"]["final_kept"]
    _activate(conn)
    _t0 = time.time()
    # ONE Format Objects session: set all signals' inputs, commit ONCE on the last
    sigs = [(MAIN_SIGNAL, champ)] + [(mods[k]["signal"], mods[k]["params"]) for k in kept]
    for _i, (_sig, _prm) in enumerate(sigs):
        _set_signal_inputs(conn, _sig, _prm, commit=(_i == len(sigs) - 1))
        log.info("[TIMING] apply-final inputs %s done (+%.1fs)", _sig, time.time() - _t0)
    kept_signals = {mods[k]["signal"] for k in kept}
    mc.set_signal_statuses(conn, {n: (n in kept_signals) for n in STATUS_ALL},
                           verify=True, protected=[MAIN_SIGNAL])
    # POST-STATUSES RE-VERIFY: the statuses session touches the same dialog; if
    # anything reverted the input strings, this pass re-fixes it (fast-path skip
    # when all correct, so it costs only one quick read-through).
    for _i, (_sig, _prm) in enumerate(sigs):
        _set_signal_inputs(conn, _sig, _prm, commit=(_i == len(sigs) - 1))
    log.info("[TIMING] apply-final inputs re-verified (+%.1fs)", time.time() - _t0)
    log.info("[%s] APPLY-FINAL done: main=%s kept=%s", SYMBOL, champ,
             {k: mods[k]["params"] for k in kept})
    return state


def run_instrument(inst, conn, args):
    _apply_ctx(inst)
    log.info("######################## INSTRUMENT %s (%s) ########################", inst[0].upper(), SYMBOL)
    if conn is not None:        # refresh the MC window handle (avoid stale handle after multi-hour runs)
        try:
            conn.connect()
        except Exception as e:
            log.warning("reconnect failed: %s", e)
    state = _load_state()
    if getattr(args, "apply_final", False):
        return apply_final_only(conn, state)
    if args.from_stage <= 1:
        state = stage1(conn, args.from_csv, state)
        if state is None: return None
    if args.from_stage <= 2:
        state = stage2(conn, args.from_csv, state)
        if state is None: return None
    if args.from_stage <= 3:
        state = stage3(conn, args.from_csv, state)
        if state is None: return None
    state = stage4(conn, args.from_csv, state)
    return state


# ----------------------------------------------------------------------------- driver
def _is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _auto_elevate():
    script = str(Path(__file__).resolve()); workdir = str(Path(__file__).resolve().parent)
    extra = [a for a in sys.argv[1:] if a != "--_elevated"]
    quoted = [f'"{a}"' if " " in a else a for a in extra]
    print("[auto-elevate] approve the UAC prompt.")
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable,
                                        f'"{script}" ' + " ".join(quoted) + " --_elevated", workdir, 1)
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", type=str, default=None, help="run one instrument (key or symbol)")
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--probe-windows", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore cached *_raw.csv and re-optimize")
    ap.add_argument("--apply-final", action="store_true", help="only re-apply final params+statuses to the chart from state.json")
    ap.add_argument("--resume-gaps", action="store_true",
                    help="resume any instrument missing a complete stage4 (in one process)")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    global FORCE
    FORCE = bool(args.force)
    if not args.from_csv and not _is_admin():
        _auto_elevate(); return 0
    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection(); conn.connect()
    if args.probe_windows:
        mc.ensure_chart_ready(conn, StrategyConfig(name="probe", mc_signal_name=SIG_CRYPTO, timeframe=TF,
                              bar_period=BARP, params=[ParamAxis("Length", 18, 22, 2)],
                              chart_workspace=WORKSPACE, chart_symbol="probe", insample=INSAMPLE))
        log.info("==== Open chart windows in workspace ====")
        for hwnd, title in mc.list_chart_windows(conn):
            log.info("  [0x%x] %s", hwnd, title)
        return 0
    if args.resume_gaps:
        plan = []
        for i in INSTRUMENTS:
            _apply_ctx(i); st = _load_state()
            if not st.get("stage4") or not (st.get("stage4") or {}).get("steps"):
                fs = 4 if (st.get("stage2") and st.get("stage3")) else 1
                plan.append((i, fs))
        if not plan:
            log.info("resume-gaps: nothing to do (all instruments have stage4)"); return 0
        log.info("resume-gaps plan: %s", [(i[0], fs) for i, fs in plan])
    else:
        todo = INSTRUMENTS
        if args.instrument:
            key = args.instrument.lower()
            todo = [i for i in INSTRUMENTS if i[0] == key or i[1].lower() == key]
            if not todo:
                log.error("Unknown --instrument '%s' (keys: %s)", args.instrument, [i[0] for i in INSTRUMENTS]); return 1
        plan = [(i, args.from_stage) for i in todo]
    t0 = time.time(); summary = []
    for inst, fstage in plan:
        a2 = argparse.Namespace(**vars(args)); a2.from_stage = fstage
        try:
            st = run_instrument(inst, conn, a2)
            if st is None:
                summary.append((inst[0], "ABORTED")); continue
            summary.append((inst[0], f"S2 OOS={st['stage2']['winner']['oos_np']:.0f} kept={st['stage4']['final_kept']}"))
        except Exception as e:
            log.exception("[%s] instrument FAILED: %s", inst[0], e)
            summary.append((inst[0], f"FAILED: {e}"))
    log.info("############ ALL DONE (%.1f min) ############", (time.time() - t0) / 60)
    for k, s in summary:
        log.info("  %-4s : %s", k.upper(), s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
