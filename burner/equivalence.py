"""Equivalence verification (spec 4.4) -- the hard gate.

Same-session A/B (decided over comparing to the manifest's stage4_final:
price data drifts daily, CLAUDE.md Critical Rule 5, so stale numbers are
advisory only):

  A = the ORIGINAL multi-signal deploy: main signal + kept modules enabled,
      stage2.main_champ + stage3 params applied, full-period backtest.
  B = the BURNED merged signal (compiled manually into MC64 beforehand),
      enabled alone, full-period backtest.

PASS: |NP_A - NP_B| / |NP_A| <= tolerance, and the same for MaxIntradayDD.

Measurement reuses the pipeline pattern: a degenerate 3-point optimization
grid via mc_automation.run_optimization_for_strategy -> CSV -> exact-row pick.
All four main params vary +-1 micro-step (MC garbles the CSV when any grid
axis is fixed -- Critical Rule 1); kept-module params are pinned via
fixed_inputs.

Workspace path / date ranges / chart symbols are regex-scraped READ-ONLY from
the pipeline script (they are not in state.json); tests/test_constants_sync.py
locks the instrument table against the same source.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .burn import BURNED_ROOT
from .instruments import INSTRUMENTS, REPO_ROOT, TF_MAP

OPTIMIZER_DIR = REPO_ROOT / "optimizer"

# how far each main param is nudged for the 3-point micro grid
_MICRO_STEP = {"default_int": 1.0, "default_float": 0.01}


def _micro_step(value: float) -> float:
    return _MICRO_STEP["default_int"] if value == int(value) \
        else _MICRO_STEP["default_float"]


# ---------------------------------------------------------------- scraping
def scrape_pipeline_constants(script: Path) -> Dict:
    """READ-ONLY regex scrape of WORKSPACE / FULL date ranges / instrument
    rows out of a run_*_allinst_pipeline.py (they exist nowhere else)."""
    text = script.read_text(encoding="utf-8")
    out: Dict = {"script": str(script)}
    m = re.search(r'WORKSPACE\s*=\s*r?"([^"]+)"', text)
    if not m:
        raise ValueError(f"{script.name}: WORKSPACE not found")
    out["workspace"] = m.group(1)

    def _range_pair(name: str) -> Tuple[str, str]:
        mm = re.search(name + r'\s*=\s*\("([^"]+)",\s*"([^"]+)"\),\s*\("([^"]+)",\s*"([^"]+)"\)',
                       text)
        if not mm:
            raise ValueError(f"{script.name}: {name} ranges not found")
        return (mm.group(3), mm.group(4))  # the FULL pair

    full_ranges = {"crypto": _range_pair(r"CRYPTO_IS,\s*CRYPTO_FULL"),
                   "futures": _range_pair(r"FUT_IS,\s*FUT_FULL")}
    out["full_ranges"] = full_ranges

    sigs = dict(re.findall(r'(SIG_CRYPTO|SIG_NQ)\s*=\s*"([^"]+)"', text))
    out["signals"] = sigs

    rows = re.findall(
        r'\(\s*"(\w+)",\s*"([^"]+)",\s*\[([^\]]*)\],\s*(SIG_CRYPTO|SIG_NQ),', text)
    insts = {}
    for key, symbol, tokens_raw, sig_var in rows:
        tokens = re.findall(r'"([^"]+)"', tokens_raw)
        klass = "crypto" if sig_var == "SIG_CRYPTO" else "futures"
        insts[key] = {"chart_symbol": symbol, "tokens": tokens,
                      "main_signal": sigs[sig_var], "symbol_class": klass,
                      "full_range": full_ranges[klass]}
    out["instruments"] = insts
    return out


def default_pipeline_script(strat_key: str) -> Path:
    return OPTIMIZER_DIR / f"run_{strat_key}_allinst_pipeline.py"


# ---------------------------------------------------------------- manifests
def latest_manifest(name: str, inst: str, tf_code: str,
                    burned_root: Path = BURNED_ROOT) -> Optional[Dict]:
    out_dir = burned_root / name
    pat = re.compile(re.escape(f"{name}_{inst}_{tf_code}_v") + r"(\d+)\.manifest\.json$")
    best, best_n = None, -1
    if out_dir.exists():
        for p in out_dir.iterdir():
            m = pat.match(p.name)
            if m and int(m.group(1)) > best_n:
                best_n, best = int(m.group(1)), p
    if best is None:
        return None
    return json.loads(best.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- dry run
def build_checklist(name: str, strat_key: str, tf: str, insts: List[str],
                    pipeline_script: Optional[str]) -> Dict:
    tf_code = TF_MAP[tf]
    script = Path(pipeline_script) if pipeline_script \
        else default_pipeline_script(strat_key)
    consts = scrape_pipeline_constants(script) if script.exists() else None
    items = []
    for key in insts:
        ctx = INSTRUMENTS[key]
        mf = latest_manifest(name, ctx.inst, tf_code)
        if mf is None:
            items.append({"inst": ctx.inst, "error": "no burned manifest found -- "
                          "run `py -m burner burn` first"})
            continue
        pc = (consts or {}).get("instruments", {}).get(key, {})
        enabled_a = [pc.get("main_signal", f"{name}_{ctx.variant_suffix}")] + \
                    [m["signal"] for m in mf["exit_modules"]]
        items.append({
            "inst": ctx.inst,
            "strategy_id": mf["strategy_id"],
            "compile_first": f"burned/{name}/{mf['strategy_id']}.txt "
                             f"(paste into MC64 Study Editor, compile)",
            "chart_symbol": pc.get("chart_symbol", ctx.chart_symbol),
            "full_range": pc.get("full_range"),
            "run_a": {"enable": enabled_a,
                      "main_inputs": mf["params"],
                      "module_inputs": {m["signal"]: m["params"]
                                        for m in mf["exit_modules"]}},
            "run_b": {"enable": [mf["strategy_id"]]},
            "advisory_stage4_final": mf["stage4_final"],
        })
    return {"name": name, "tf": tf_code,
            "workspace": (consts or {}).get("workspace"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "items": items}


# ---------------------------------------------------------------- live A/B
def _measure(conn, mc, cfgmod, signal_name: str, vary_params: Dict[str, float],
             fixed_inputs: Dict[str, Dict[str, float]], workspace: str,
             chart_symbol: str, out_dir: Path, run_name: str):
    """One full-period backtest via a 3-point-per-axis micro optimization."""
    import pandas as pd
    axes = []
    for p, v in vary_params.items():
        st = _micro_step(v)
        axes.append(cfgmod.ParamAxis(p, max(0.0, round(v - st, 8)),
                                     round(v + st, 8), st))
    cfg = cfgmod.StrategyConfig(
        name=run_name, mc_signal_name=signal_name, timeframe="hourly",
        bar_period=60, params=axes, chart_workspace=workspace,
        chart_symbol=chart_symbol,
        insample=cfgmod.DateRange("2017/01/01", "2027/01/01"),
        fixed_inputs=fixed_inputs or None)
    csv_path = mc.run_optimization_for_strategy(conn, cfg, str(out_dir))
    df = mc.load_results_csv(csv_path, cfg)
    if df is None or df.empty:
        raise RuntimeError(f"{run_name}: empty optimization export")
    mask = None
    for p, v in vary_params.items():
        cols = [c for c in df.columns if c == p or c.startswith(p + ".")]
        col = cols[0]
        if len(cols) > 1:  # duplicate column (e.g. M6 Length): pick the varying one
            col = next((c for c in cols
                        if pd.to_numeric(df[c], errors="coerce").nunique() > 1), col)
        m = (pd.to_numeric(df[col], errors="coerce") - v).abs() < 1e-6
        mask = m if mask is None else (mask & m)
    hit = df[mask]
    if hit.empty:
        raise RuntimeError(f"{run_name}: exact param row not found in export")
    row = hit.iloc[0]
    return float(row["NetProfit"]), float(row["MaxDrawdown"])


def run_verify(name: str, strat_key: str, tf: str, insts: List[str],
               dry_run: bool = False, pipeline_script: Optional[str] = None,
               tolerance: float = 0.005) -> int:
    tf_code = TF_MAP[tf]
    out_dir = BURNED_ROOT / name
    if dry_run:
        checklist = build_checklist(name, strat_key, tf, insts, pipeline_script)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "equivalence_checklist.json"
        path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"checklist written: {path}")
        bad = [i for i in checklist["items"] if "error" in i]
        for i in bad:
            print(f"  [{i['inst']}] {i['error']}")
        return 1 if bad else 0

    # ---- live mode: MC64 must be open with the workspace loaded ----
    sys.path.insert(0, str(OPTIMIZER_DIR))
    import config as cfgmod          # optimizer/config.py
    import mc_automation as mc       # optimizer/mc_automation.py

    script = Path(pipeline_script) if pipeline_script \
        else default_pipeline_script(strat_key)
    consts = scrape_pipeline_constants(script)
    report_items = []
    conn = mc.MultiChartsConnection()
    conn.connect()
    scratch = out_dir / "equivalence_runs"
    scratch.mkdir(parents=True, exist_ok=True)
    failed = 0
    for key in insts:
        ctx = INSTRUMENTS[key]
        mf = latest_manifest(name, ctx.inst, tf_code)
        pc = consts["instruments"][key]
        item = {"inst": ctx.inst, "strategy_id": mf["strategy_id"] if mf else None}
        try:
            if mf is None:
                raise RuntimeError("no burned manifest -- burn first")
            main_sig = pc["main_signal"]
            kept_sigs = [m["signal"] for m in mf["exit_modules"]]
            sid = mf["strategy_id"]
            status_all = kept_sigs + [main_sig, sid]

            mc.activate_chart_by_symbol(conn, pc["chart_symbol"], pc["tokens"])
            mc.set_instrument_data_range(conn, *pc["full_range"])

            # --- Run A: original multi-signal deploy ---
            mc.set_signal_statuses(
                conn, {s: (s == main_sig or s in kept_sigs) for s in status_all},
                verify=True, protected=[])
            fixed = {m["signal"]: dict(m["params"]) for m in mf["exit_modules"]}
            np_a, mdd_a = _measure(conn, mc, cfgmod, main_sig, mf["params"],
                                   fixed, consts["workspace"], pc["chart_symbol"],
                                   scratch, f"EQ_A_{ctx.inst}")

            # --- Run B: burned merged signal alone ---
            mc.set_signal_statuses(conn, {s: (s == sid) for s in status_all},
                                   verify=True, protected=[])
            np_b, mdd_b = _measure(conn, mc, cfgmod, sid, mf["params"],
                                   None, consts["workspace"], pc["chart_symbol"],
                                   scratch, f"EQ_B_{ctx.inst}")

            np_dev = abs(np_a - np_b) / abs(np_a) if np_a else abs(np_b)
            mdd_dev = abs(mdd_a - mdd_b) / abs(mdd_a) if mdd_a else abs(mdd_b)
            ok = np_dev <= tolerance and mdd_dev <= tolerance
            item.update({
                "run_a": {"net_profit": np_a, "max_intraday_drawdown": mdd_a},
                "run_b": {"net_profit": np_b, "max_intraday_drawdown": mdd_b},
                "np_dev": round(np_dev, 6), "mdd_dev": round(mdd_dev, 6),
                "tolerance": tolerance, "pass": ok,
                "advisory_manifest_stage4_final": mf["stage4_final"],
            })
            if not ok:
                failed += 1
        except Exception as e:  # noqa: BLE001 - report and continue per inst
            item.update({"pass": False, "error": str(e)})
            failed += 1
        report_items.append(item)
        mark = "PASS" if item.get("pass") else "FAIL"
        print(f"[{mark}] {ctx.inst}: {item}")

    report = {"name": name, "tf": tf_code, "tolerance": tolerance,
              "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "items": report_items}
    path = out_dir / "equivalence_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"report written: {path}")
    return 1 if failed else 0
