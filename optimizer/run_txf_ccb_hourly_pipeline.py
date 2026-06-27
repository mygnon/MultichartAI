"""
run_txf_ccb_hourly_pipeline.py -- ONE-BAT unattended 4-stage pipeline for the
FUTURES strategy SFJ_CloseChannelBreakout_NQ on TWF.TXF HOT HOURLY.

  Stage 1  IS optimization (R1 broad + confirm rounds until converged, Obj-max)  2019/01-2025/01
  Stage 2  OOS champion-select among Stage-1 candidates              IS vs FULL 2018/01-2026/01
  Stage 3  exit-module IS optimization (6 modules, main fixed)        2019/01-2025/01
  Stage 4  CUMULATIVE greedy full-period OOS stack (keep iff RoMaD up) 2018/01-2026/01

One MC64 connection, params handed stage->stage in memory, decisions made in code:
  - convergence: SELF-JUDGED -- R1 broad, then confirm rounds seeded at the running Obj-max
    until round-over-round Obj gain <= 0.5% (min 2 rounds, max 5). champion = Obj-max (NP^2/|MDD|).
  - Stage-2 pick: strict PASS (|MDD_full|<=|MDD_is|) with max OOS, else de-facto (max OOS, mildest break).
  - RoMaD = NetProfit / |Max Intraday Drawdown| (= MCReport MaxDrawdown column).
Programmatic Format-Objects setting (main after Stage 2; 6 modules after Stage 3) with CSV read-back
validation -> ABORT loudly if a held value is wrong (never silently skip). state.json after every stage;
--from-stage N resumes.

NOTE: the strategy is the futures version (no _Crypto1MUSD sizing) -> orders use the chart's default
sizing (Format Signals -> Properties -> default contracts). Run on TWF.TXF HOT data.

PREREQUISITES (MC64 as Administrator):
  SFJ_CloseChannelBreakout_AI.wsp open; TWF.TXF HOT HOURLY tab active+visible;
  data to 2026/01/01; data feed connected; Study Editor CLOSED; main SFJ_CloseChannelBreakout_NQ
  + 6 module signals inserted (Status arbitrary -- the orchestrator sets all Status & fixed inputs).

CLI:
  py run_txf_ccb_hourly_pipeline.py                 # full pipeline (auto-elevates)
  py run_txf_ccb_hourly_pipeline.py --from-stage 3  # resume from stage 3 (uses state.json)
  py run_txf_ccb_hourly_pipeline.py --from-csv      # re-analyse existing CSVs, no MC64
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

WORKSPACE   = r"C:\Users\Tim\Downloads\Multichart64\Tim\SFJ_CloseChannelBreakout_AI.wsp"
SYMBOL      = "TWF.TXF HOT"
MAIN_SIGNAL = "SFJ_CloseChannelBreakout_NQ"
OUTPUT_DIR  = Path(r"C:\Users\Tim\MultichartAI\results\txf_ccb_hourly_pipeline")
INSAMPLE    = DateRange("2017/01/01", "2027/01/01")          # wide no-op; chart-trim is control
IS_RANGE    = ("2019/01/01", "2025/01/01")
FULL_RANGE  = ("2018/01/01", "2026/01/01")
TF, BARP    = "hourly", 60

LEN_LO, LEN_HI   = 2.0, 300.0
ATR_LO, ATR_HI   = 0.5, 20.0
RE_LO,  RE_HI    = 0.0, 50.0
SEED = (20.0, 7.0, 13.0)   # Length, ATRMult, ReentryBars
TARGET_NP = 10_000_000.0

CONV_THRESH = 0.005   # <=0.5% round-over-round Obj gain => converged
MIN_ROUNDS, MAX_ROUNDS = 2, 5

# Stage-3 exit modules: (label, signal, [(param, lo, hi, step)]) -- user-specified ranges
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
MODULE_PARAM_NAMES = {sig: {p[0] for p in axes} for (_, sig, axes) in MODULES}
# Stage-4 representative (non-colliding) axis to vary per module
VARY_AXIS = {"M1": ("STP", 0.1), "M2": ("ATRSTP", 0.1), "M3": ("EXITBAR", 1.0),
             "M4": ("DAYRANGE", 0.01), "M5": ("PT_Base", 0.001), "M6": ("std", 0.1)}
EPS = 1e-9
STATE_JSON = OUTPUT_DIR / "state.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s -- %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(str(OUTPUT_DIR / f"pipeline_{int(time.time())}.log"),
                                                  encoding="utf-8")])
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- helpers
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


def _main_cfg(name, L, A, R):
    L = _safe(*L, LEN_LO, LEN_HI); A = _safe(*A, ATR_LO, ATR_HI); R = _safe(*R, RE_LO, RE_HI)
    combos = n_vals(*L) * n_vals(*A) * n_vals(*R)
    if combos > 5000:
        log.warning("  %s: %d combos >5000!", name, combos)
    return StrategyConfig(name=name, mc_signal_name=MAIN_SIGNAL, timeframe=TF, bar_period=BARP,
                          params=[ParamAxis("Length", *L), ParamAxis("ATRMult", *A),
                                  ParamAxis("ReentryBars", *R)],
                          chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)


def _module_cfg(name, signal, axes):
    combos = 1
    for (_, s, e, st) in axes:
        combos *= n_vals(s, e, st)
    if combos > 5000:
        log.warning("  %s: %d combos >5000!", name, combos)
    return StrategyConfig(name=name, mc_signal_name=signal, timeframe=TF, bar_period=BARP,
                          params=[ParamAxis(p, s, e, st) for (p, s, e, st) in axes],
                          chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)


def csv_for(name):
    return OUTPUT_DIR / f"{name}_raw.csv"


def run_or_load(cfg, conn, from_csv):
    p = csv_for(cfg.name)
    if from_csv or p.exists():
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
            mc.ensure_chart_ready(conn, cfg)
        except Exception as e:
            log.warning("ensure_chart_ready: %s", e)
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
    return {"Length": float(row["Length"]), "ATRMult": float(row["ATRMult"]),
            "ReentryBars": float(row["ReentryBars"]), "net_profit": float(row["NetProfit"]),
            "max_drawdown": float(row["MaxDrawdown"]), "objective": float(row["Objective"]),
            "total_trades": int(row["TotalTrades"])}


def _valid_main(df):
    for p in [("Length", LEN_LO, LEN_HI), ("ATRMult", ATR_LO, ATR_HI), ("ReentryBars", RE_LO, RE_HI)]:
        nm, lo, hi = p
        if nm not in df.columns:
            return False
        col = pd.to_numeric(df[nm], errors="coerce")
        if not col.between(lo - 1e6, hi + 1e6).all():
            return False
    return True


# ----------------------------------------------------------------------------- Stage 1
def _stage1_round(label, attempts, seed, conn, from_csv):
    """Run a set of static attempts + 3 adaptive zooms; return list of row-dicts."""
    rows: List[dict] = []
    best = {"objective": -1e18}
    bx, ba, br = seed
    for nm, L, A, R in attempts:
        cfg = _main_cfg(f"S1_{label}_{nm}", L, A, R)
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
            best = om; bx, ba, br = om["Length"], om["ATRMult"], om["ReentryBars"]
        log.info("  [%s %-16s] Obj-max L=%.4g A=%.4g R=%.4g NP=%.0f MDD=%.0f Obj=%.4g tr=%d",
                 label, nm, om["Length"], om["ATRMult"], om["ReentryBars"], om["net_profit"],
                 om["max_drawdown"], om["objective"], om["total_trades"])
    # 3 adaptive zooms around running Obj-max
    for zi, (rl, rs, ra, rast, rr, rrst) in enumerate(
            [(8, 1, 2.0, 0.5, 6, 1), (5, 1, 1.0, 0.25, 4, 1), (3, 1, 0.5, 0.25, 2, 1)], 1):
        Lz = (max(LEN_LO, _snap(bx - rl, rs)), min(LEN_HI, _snap(bx + rl, rs)), rs)
        Az = (max(ATR_LO, _snap(ba - ra, rast)), min(ATR_HI, _snap(ba + ra, rast)), rast)
        Rz = (max(RE_LO, _snap(br - rr, rrst)), min(RE_HI, _snap(br + rr, rrst)), rrst)
        cfg = _main_cfg(f"S1_{label}_zoom{zi}", Lz, Az, Rz)
        df = run_or_load(cfg, conn, from_csv)
        if df is None or df.empty or not _valid_main(df):
            continue
        d = _with_obj(df); pos = d[d["NetProfit"] > 0]
        if pos.empty:
            continue
        om = _row_dict(pos.loc[pos["Objective"].idxmax()]); rows.append(om)
        if om["objective"] > best["objective"]:
            best = om; bx, ba, br = om["Length"], om["ATRMult"], om["ReentryBars"]
        log.info("  [%s zoom%d] Obj-max L=%.4g A=%.4g R=%.4g Obj=%.4g", label, zi,
                 om["Length"], om["ATRMult"], om["ReentryBars"], om["objective"])
    return rows


def _confirm_attempts(sx, sa, sr):
    return [
        ("01_retest", (max(LEN_LO, sx-2), sx+2, 1), (max(ATR_LO, sa-1), sa+1, 0.5), (max(RE_LO, sr-2), sr+2, 1)),
        ("02_len_fine", (max(LEN_LO, sx-8), sx+8, 1), (sa, sa, 0.5), (sr, sr, 1)),
        ("03_atr_fine", (sx, sx, 1), (max(ATR_LO, sa-3), min(ATR_HI, sa+3), 0.25), (sr, sr, 1)),
        ("04_re_fine",  (sx, sx, 1), (sa, sa, 0.5), (0, 30, 1)),
        ("05_atr_low",  (sx, sx, 1), (0.5, min(ATR_HI, sa), 0.25), (sr, sr, 1)),
        ("06_combo",    (max(LEN_LO, sx-6), sx+6, 1), (max(ATR_LO, sa-2), min(ATR_HI, sa+2), 0.5), (max(RE_LO, sr-4), sr+4, 1)),
    ]


def stage1(conn, from_csv, state):
    log.info("############ STAGE 1: IS optimization (self-judged convergence) ############")
    if conn is not None:
        mc.ensure_chart_ready(conn, _main_cfg("S1_seed", (18, 22, 2), (6.5, 7.5, 0.5), (11, 13, 1)))
        mc.set_signal_statuses(conn, {n: False for n in ALL_MODULE_NAMES}, verify=True, protected=[MAIN_SIGNAL])
        mc.set_instrument_data_range(conn, *IS_RANGE)
    r1_attempts = [
        ("01_global_wide", (5, 100, 5),  (2.0, 12.0, 1.0), (0, 30, 5)),
        ("02_classic",     (5, 40, 2),   (5.0, 9.0, 0.5),  (8, 20, 2)),
        ("03_length_fine", (2, 60, 2),   (6.0, 8.0, 0.5),  (10, 16, 2)),
        ("04_atrmult_fine",(8, 28, 2),   (3.0, 12.0, 0.5), (10, 16, 2)),
        ("05_reentry_fine",(8, 28, 2),   (6.0, 8.0, 0.5),  (0, 30, 1)),
        ("06_wide_trail",  (5, 50, 5),   (8.0, 20.0, 0.5), (5, 20, 5)),
        ("07_short_len",   (2, 20, 1),   (4.0, 10.0, 0.5), (5, 20, 5)),
        ("08_long_len",    (40, 300, 10),(3.0, 9.0, 0.5),  (0, 20, 5)),
        ("09_global_bound",(5, 300, 15), (1.0, 18.0, 1.0), (0, 40, 10)),
    ]
    rows = _stage1_round("R1", r1_attempts, SEED, conn, from_csv)
    if not rows:
        log.error("Stage1 R1 produced no valid rows -- ABORT"); return None
    champ = max(rows, key=lambda r: r["objective"])
    conv = []
    for rnd in range(2, MAX_ROUNDS + 1):
        prev_obj = champ["objective"]
        sx, sa, sr = champ["Length"], champ["ATRMult"], champ["ReentryBars"]
        rows += _stage1_round(f"R{rnd}", _confirm_attempts(sx, sa, sr), (sx, sa, sr), conn, from_csv)
        champ = max(rows, key=lambda r: r["objective"])
        gain = (champ["objective"] - prev_obj) / prev_obj if prev_obj > 0 else 1.0
        conv.append({"round": rnd, "prev_obj": prev_obj, "obj": champ["objective"], "gain_pct": round(gain * 100, 4)})
        log.info(">>> R%d Obj %.6g -> %.6g (gain %.3f%%)", rnd, prev_obj, champ["objective"], gain * 100)
        if rnd >= MIN_ROUNDS and gain <= CONV_THRESH:
            log.info(">>> CONVERGED at R%d (gain %.3f%% <= %.3f%%)", rnd, gain * 100, CONV_THRESH * 100)
            break
    npmax = max(rows, key=lambda r: r["net_profit"])
    # candidate set: champion(Obj) + NP-max + next distinct high-Obj regimes
    cands = []
    seen = set()
    def _key(r): return (round(r["Length"]), round(r["ATRMult"], 2), round(r["ReentryBars"]))
    for r in [champ, npmax] + sorted(rows, key=lambda r: -r["objective"]):
        k = _key(r)
        if k in seen:
            continue
        seen.add(k); cands.append(r)
        if len(cands) >= 4:
            break
    state["stage1"] = {"champion_obj": champ, "champion_np": npmax, "candidates": cands,
                       "convergence": conv, "timestamp": datetime.now().isoformat()}
    _save_state(state)
    log.info(">>> Stage1 Obj-max champion: L=%.4g A=%.4g R=%.4g NP=%.0f MDD=%.0f Obj=%.4g tr=%d",
             champ["Length"], champ["ATRMult"], champ["ReentryBars"], champ["net_profit"],
             champ["max_drawdown"], champ["objective"], champ["total_trades"])
    log.info(">>> Stage1 candidates: %s", [(_key(c)) for c in cands])
    return state


# ----------------------------------------------------------------------------- Stage 2
def _micro_main_cfg(name, L, A, R):
    return StrategyConfig(name=name, mc_signal_name=MAIN_SIGNAL, timeframe=TF, bar_period=BARP,
                          params=[ParamAxis("Length", max(LEN_LO, L-1), L+1, 1.0),
                                  ParamAxis("ATRMult", max(ATR_LO, A-0.5), A+0.5, 0.5),
                                  ParamAxis("ReentryBars", max(RE_LO, R-1), R+1, 1.0)],
                          chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)


def _pick_main(df, L, A, R):
    m = df
    for nm, v in (("Length", L), ("ATRMult", A), ("ReentryBars", R)):
        if nm in m.columns:
            m = m[(pd.to_numeric(m[nm], errors="coerce") - v).abs() < 1e-6]
    return m.iloc[0] if not m.empty else None


def stage2(conn, from_csv, state):
    log.info("############ STAGE 2: OOS champion-select ############")
    cands = state["stage1"]["candidates"]
    results = {"is": {}, "full": {}}
    for period, rng in (("is", IS_RANGE), ("full", FULL_RANGE)):
        if conn is not None:
            mc.ensure_chart_ready(conn, _micro_main_cfg("S2_probe",
                                  cands[0]["Length"], cands[0]["ATRMult"], cands[0]["ReentryBars"]))
            mc.set_signal_statuses(conn, {n: False for n in ALL_MODULE_NAMES}, verify=True, protected=[MAIN_SIGNAL])
            mc.set_instrument_data_range(conn, *rng)
        for i, c in enumerate(cands):
            cfg = _micro_main_cfg(f"S2_{period}_C{i}", c["Length"], c["ATRMult"], c["ReentryBars"])
            df = run_or_load(cfg, conn, from_csv)
            row = _pick_main(df, c["Length"], c["ATRMult"], c["ReentryBars"]) if df is not None and not df.empty else None
            if row is not None:
                results[period][str(i)] = {"net_profit": float(row["NetProfit"]),
                                           "max_drawdown": float(row["MaxDrawdown"]),
                                           "total_trades": int(row["TotalTrades"])}
                log.info("  [%s C%d %s] NP=%.0f MDD=%.0f", period, i,
                         (c["Length"], c["ATRMult"], c["ReentryBars"]),
                         float(row["NetProfit"]), float(row["MaxDrawdown"]))
    # verdict
    rep = []
    for i, c in enumerate(cands):
        ci, cf = results["is"].get(str(i)), results["full"].get(str(i))
        if not ci or not cf:
            continue
        oos = cf["net_profit"] - ci["net_profit"]
        brk = abs(cf["max_drawdown"]) > abs(ci["max_drawdown"])
        rep.append({"idx": i, "params": {"Length": c["Length"], "ATRMult": c["ATRMult"], "ReentryBars": c["ReentryBars"]},
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
        log.error("Stage2 no valid candidates -- ABORT"); return None
    state["stage2"] = {"results": results, "report": rep, "winner": winner,
                       "main_champ": winner["params"], "timestamp": datetime.now().isoformat()}
    _save_state(state)
    log.info(">>> Stage2 WINNER %s OOS=%.0f pass=%s -> MAIN_CHAMP=%s",
             winner["params"], winner["oos_np"], winner["pass"], winner["params"])
    return state


# ----------------------------------------------------------------------------- set inputs
def _set_signal_inputs(conn, signal, params):
    cfg = StrategyConfig(name=f"SET_{signal[:20]}", mc_signal_name=signal, timeframe=TF,
                         bar_period=BARP, params=[ParamAxis(k, v, v, 1.0) for k, v in params.items()],
                         chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)
    mc.set_params_and_date_for_single_run(conn, cfg, params, DateRange(*IS_RANGE))


# ----------------------------------------------------------------------------- Stage 3
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


def stage3(conn, from_csv, state):
    log.info("############ STAGE 3: exit-module IS optimization ############")
    champ = state["stage2"]["main_champ"]
    if conn is not None:
        mc.ensure_chart_ready(conn, _micro_main_cfg("S3_probe", champ["Length"], champ["ATRMult"], champ["ReentryBars"]))
        mc.set_instrument_data_range(conn, *IS_RANGE)
        _set_signal_inputs(conn, MAIN_SIGNAL, champ)
    if conn is not None:
        mc.set_signal_statuses(conn, {n: False for n in ALL_MODULE_NAMES}, verify=True, protected=[MAIN_SIGNAL])
    df0 = run_or_load(_micro_main_cfg("S3_A00", champ["Length"], champ["ATRMult"], champ["ReentryBars"]), conn, from_csv)
    row0 = _pick_main(df0, champ["Length"], champ["ATRMult"], champ["ReentryBars"]) if df0 is not None and not df0.empty else None
    if row0 is None:
        log.error("Stage3 A00 baseline failed (main inputs not set?) -- ABORT"); return None
    base_np = float(row0["NetProfit"]); base_mdd = float(row0["MaxDrawdown"])
    log.info("A00 IS baseline NP=%.2f MDD=%.2f tr=%d", base_np, base_mdd, int(row0["TotalTrades"]))
    mods = {}
    for (label, signal, axes) in MODULES:
        if conn is not None:
            mc.set_signal_statuses(conn, {n: (n == signal) for n in ALL_MODULE_NAMES}, verify=True, protected=[MAIN_SIGNAL])
        cfg = _module_cfg(f"S3_{label}_{signal[:24]}", signal, axes)
        df = run_or_load(cfg, conn, from_csv)
        if df is None or df.empty or not _valid_main_fixed(df, champ, [signal]):
            log.error("  Stage3 %s INVALID (main not fixed / no data) -- ABORT", label); return None
        d = _with_obj(df)
        best = d.loc[pd.to_numeric(d["NetProfit"], errors="coerce").idxmax()]
        params = {p[0]: float(best[p[0]]) for p in axes}
        np_ = float(best["NetProfit"]); mdd = float(best["MaxDrawdown"])
        mods[label] = {"signal": signal, "params": params, "net_profit": np_, "max_drawdown": mdd,
                       "total_trades": int(best["TotalTrades"]),
                       "delta_np_pct": round((np_ - base_np) / base_np * 100, 2) if base_np else None}
        log.info("  %s %s NP=%.0f (%+.2f%%) MDD=%.0f", label, params, np_,
                 mods[label]["delta_np_pct"] or 0, mdd)
    state["stage3"] = {"baseline": {"net_profit": base_np, "max_drawdown": base_mdd},
                       "modules": mods, "timestamp": datetime.now().isoformat()}
    _save_state(state)
    return state


# ----------------------------------------------------------------------------- Stage 4
def stage4(conn, from_csv, state):
    log.info("############ STAGE 4: cumulative greedy full-period OOS stack ############")
    champ = state["stage2"]["main_champ"]; mods = state["stage3"]["modules"]
    order = sorted(mods.items(), key=lambda kv: -(kv[1]["delta_np_pct"] or -1e9))
    if conn is not None:
        mc.ensure_chart_ready(conn, _micro_main_cfg("S4_probe", champ["Length"], champ["ATRMult"], champ["ReentryBars"]))
        mc.set_instrument_data_range(conn, *FULL_RANGE)
        _set_signal_inputs(conn, MAIN_SIGNAL, champ)
        for label, info in mods.items():
            _set_signal_inputs(conn, info["signal"], info["params"])
    if conn is not None:
        mc.set_signal_statuses(conn, {n: False for n in ALL_MODULE_NAMES}, verify=True, protected=[MAIN_SIGNAL])
    df0 = run_or_load(_micro_main_cfg("S4_A00", champ["Length"], champ["ATRMult"], champ["ReentryBars"]), conn, from_csv)
    row0 = _pick_main(df0, champ["Length"], champ["ATRMult"], champ["ReentryBars"]) if df0 is not None and not df0.empty else None
    if row0 is None:
        log.error("Stage4 A00 baseline failed -- ABORT"); return None
    cur_np = float(row0["NetProfit"]); cur_midd = float(row0["MaxDrawdown"])
    cur_romad = cur_np / abs(cur_midd) if cur_midd else 0.0
    log.info("A00 FULL baseline NP=%.2f MIDD=%.2f RoMaD=%.4f", cur_np, cur_midd, cur_romad)
    kept: List[str] = []; steps = []
    for step, (label, info) in enumerate(order, 1):
        signal = info["signal"]; vax, vstep = VARY_AXIS[label]; vval = info["params"][vax]
        enabled = [mods[k]["signal"] for k in kept] + [signal]
        if conn is not None:
            mc.set_signal_statuses(conn, {n: (n in enabled) for n in ALL_MODULE_NAMES}, verify=True, protected=[MAIN_SIGNAL])
        cfg = StrategyConfig(name=f"S4_S{step}_{label}", mc_signal_name=signal, timeframe=TF,
                             bar_period=BARP,
                             params=[ParamAxis(vax, max(0.0, round(vval - vstep, 8)), round(vval + vstep, 8), vstep)],
                             chart_workspace=WORKSPACE, chart_symbol=SYMBOL, insample=INSAMPLE)
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
            log.info("  S%d %s NP=%.0f MIDD=%.0f RoMaD=%.4f (prev %.4f) -> %s",
                     step, label, np_, midd, romad, cur_romad, ent["decision"])
            if keep:
                kept.append(label); cur_np, cur_midd, cur_romad = np_, midd, romad
        else:
            ent["valid"] = False
            log.error("  Stage4 S%d %s INVALID -- skipping (main/kept not held?)", step, label)
        steps.append(ent)
        state["stage4"] = {"baseline": {"net_profit": float(row0["NetProfit"]), "max_intraday_drawdown": float(row0["MaxDrawdown"])},
                           "steps": steps, "final_kept": [k for k in kept],
                           "final_romad": round(cur_romad, 4), "timestamp": datetime.now().isoformat()}
        _save_state(state)
    if conn is not None:
        kept_signals = {mods[k]["signal"] for k in kept}
        mc.set_signal_statuses(conn, {n: (n in kept_signals) for n in ALL_MODULE_NAMES},
                               verify=False, protected=[MAIN_SIGNAL])
    log.info(">>> Stage4 FINAL kept-set: %s  RoMaD %.4f", kept or "NONE", cur_romad)
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
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.from_csv and not _is_admin():
        _auto_elevate(); return 0
    conn = None
    if not args.from_csv:
        conn = mc.MultiChartsConnection(); conn.connect()
    state = _load_state()
    t0 = time.time()
    if args.from_stage <= 1:
        state = stage1(conn, args.from_csv, state)
        if state is None: return 1
    if args.from_stage <= 2:
        state = stage2(conn, args.from_csv, state)
        if state is None: return 1
    if args.from_stage <= 3:
        state = stage3(conn, args.from_csv, state)
        if state is None: return 1
    state = stage4(conn, args.from_csv, state)
    if state is None: return 1
    log.info("############ PIPELINE COMPLETE (%.1f min) ############", (time.time() - t0) / 60)
    log.info("  Stage1 champ: %s", state["stage1"]["champion_obj"] and
             {k: state["stage1"]["champion_obj"][k] for k in ("Length", "ATRMult", "ReentryBars", "objective")})
    log.info("  Stage2 main : %s (OOS=%.0f)", state["stage2"]["main_champ"], state["stage2"]["winner"]["oos_np"])
    log.info("  Stage4 kept : %s  RoMaD=%.4f", state["stage4"]["final_kept"], state["stage4"]["final_romad"])
    print(f"\nDone -- state at: {STATE_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
