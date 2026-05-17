# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a **parameter plateau optimizer** for MultiCharts64 (MC64) trading strategies. It automates MC64's built-in optimizer via Windows UI automation, exports results as CSV, and applies a **plateau detection algorithm** to find parameter combinations that are stable (robust to small parameter perturbations) rather than just peak performers. Results include IS/OOS validation and heatmap visualizations.

The target strategies are breakout and SuperTrend systems trading TWF.TXF (Taiwan Futures) on daily and hourly timeframes.

## Running the Optimizer

All scripts must run as Administrator because MC64 runs elevated and Windows UIPI blocks cross-privilege UI automation.

```powershell
# From optimizer\ — run all 4 strategies (MC64 must be open with correct workspaces)
optimizer\run_optimizer.bat

# Or directly (auto-elevates via UAC):
cd optimizer
py main.py --strategies all

# Re-analyze existing CSVs without launching MC64:
py main.py --strategies all --from-csv ..\results

# Single strategy, dry run to check grid size:
py main.py --strategies breakout_daily --dry-run

# Adaptive multi-attempt search scripts (more thorough, strategy-specific):
py search_hourly.py
py search_hourly.py --from-csv        # re-analyze only
py search_hourly.py --attempt 5       # resume from attempt 5

# Breakout Daily NP>6M adaptive search (latest version):
py search_daily_target6.py
py search_daily_target6.py --from-csv        # re-analyze only
py search_daily_target6.py --attempt 6       # resume from attempt 6
```

## Install Dependencies

```powershell
pip install -r optimizer\requirements.txt
```

## Architecture

### Core pipeline (3 phases)

1. **MC64 UI Automation** (`mc_automation.py`) — Controls MultiCharts64 via pywinauto + pyautogui to run parameter sweeps and export CSVs. Uses raw ctypes `EnumWindows`/`EnumChildWindows` to bypass UIPI (pywinauto process-scoped specs fail across privilege levels). Key entry point: `run_optimization_for_strategy(conn, cfg, output_dir)`.

2. **Plateau detection** (`plateau.py`) — Reshapes the flat CSV into a 2D parameter grid. Objective = `NetProfit² / |MaxDrawdown|` (only where both are valid). Computes a sliding-minimum over a `(2r+1)×(2r+1)` neighborhood — a point's **plateau score** is the minimum objective of all its neighbors. High plateau score means the region is uniformly good, not just a spike. Radius defaults to 2 (configurable via `--radius`).

3. **Visualization & reporting** (`visualize.py`) — Dual-panel heatmaps (objective vs. plateau score), HTML summary report, JSON results files.

### Configuration (`config.py`)

All strategies are defined as `StrategyConfig` objects with:
- `mc_signal_name` — exact name in MC's Format Signals dialog
- `chart_workspace` — full path to `.wsp` file (must be open in MC64 before running)
- `params` — list of `ParamAxis(name, start, stop, step)` where `name` must match MC's Inputs tab exactly

`STRATEGY_MAP` maps lowercase/underscore-normalized names to configs. Adding a new strategy requires adding a `StrategyConfig` and updating `ALL_STRATEGIES` and `STRATEGY_MAP`.

### Adaptive search scripts (`search_hourly.py`, `search_daily*.py`)

These are standalone multi-phase scripts that perform sequential zooming:
- Phase 1: Wide LE×SE sweep to find stable region
- Phase 2: STP×LMT sweep with LE/SE fixed
- Phase 3: Cross-validate and optionally a 4D grid

Each attempt reads from an existing CSV if present (idempotent). The final champion is saved to `results/<subdir>/final_params_*.json`. These scripts hardcode `WORKSPACE`, `SYMBOL`, `SIGNAL`, and `OUTPUT_DIR` constants at the top — edit there when targeting a different strategy.

### Column name mapping

MC64 exports different column headers across versions. `MC_COLUMN_MAP` in `config.py` normalizes them to `NetProfit`, `MaxDrawdown`, `TotalTrades`. If a new MC version uses different headers, add entries there.

### Results layout

```
results/
  <StrategyName>_raw.csv              # raw MC64 optimization export
  <StrategyName>_plateau.json         # plateau candidates
  <StrategyName>_oos.json             # OOS validation results
  <StrategyName>_heatmap.png          # dual-panel heatmap
  report_<timestamp>.html             # full HTML report
  run_<timestamp>.log                 # run log
  hourly_search/
    BH_<attempt>_raw.csv              # per-attempt CSVs
    final_params_hourly.json          # best params from adaptive search
  daily_target6_search/               # latest Breakout Daily NP>6M search
    BD6_<attempt>_raw.csv             # per-attempt CSVs
    final_params_daily_target6.json   # best params (current champion)
    search_daily_target6_*.log        # run log
```

## Key Constraints

- **Must run as Administrator** — MC64 runs elevated; UI automation fails otherwise. `search_daily_target6.py` (and all `search_*.py`) auto-elevate via UAC using `ctypes.windll.shell32.ShellExecuteW(None, "runas", ...)`. Never skip this.
- **MC64 workspace must be open** before running — the automation finds the chart window by matching `chart_symbol` against open window titles.
- **pywinauto limitation** — never use process-scoped `Application(process=pid)` to reach MC64 dialogs; use `Desktop(backend="uia")` + ctypes window enumeration instead. This is the core UIPI workaround.
- **`ParamAxis.name` must match MC exactly** — case-sensitive; the automation types these names into the Inputs dialog.

## Critical Rules for Adaptive Search Scripts

These rules were learned through multiple failed rounds. Violating them silently corrupts results.

### 1. ALL 4 parameters MUST vary in every attempt

When any parameter is fixed (`start == stop`), MC64's MCReport export packs multiple metric sets per row. `pandas` misreads the columns — LE/SE/STP/LMT values appear completely wrong. Always call `_safe()` on every range before building a `StrategyConfig`:

```python
def _safe(t):
    s, e, step = t
    if s == e:
        return (max(LO, s - step), min(HI, s + step), step)
    return t
le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)
```

### 2. Validate every loaded CSV with `_validate_df()`

After loading a CSV, check that every parameter column's values fall within the expected range. If out of range, the CSV is garbled — discard it and record 0 results for that attempt.

```python
def _validate_df(df, cfg):
    for p in cfg.params:
        lo = min(p.start, p.stop) - abs(p.step) * 2
        hi = max(p.start, p.stop) + abs(p.step) * 2
        col = pd.to_numeric(df[p.name], errors="coerce")
        if not col.between(lo, hi).all():
            return False
    return True
```

### 3. Dynamic combo-reduction loops MUST progressively shrink

A `while combos > 5000` loop that recalculates with the same radius every iteration never converges — it fills the disk with log output. Use a `for` loop with decreasing radii and `break` on first fit:

```python
for r_se, r_stp, r_lmt in [(12, 1.0, 4), (8, 0.8, 3), (6, 0.6, 2), (4, 0.4, 1)]:
    _se  = zoom(best_se,  r_se,  2,   SE_LO, SE_HI)
    _stp = zoom(best_stp, r_stp, 0.2, STP_LO, STP_HI)
    _lmt = zoom(best_lmt, r_lmt, 0.5, LMT_LO, LMT_HI)
    _c = _cfg(name, _le, _se, _stp, _lmt)
    if _c.total_runs() <= 5000:
        break
```

### 4. Limit each attempt to ≤ 5000 combos

```python
combos = n_LE × n_SE × n_STP × n_LMT
assert combos <= 5000
```

### 5. NP numbers are not stable across days

TWF.TXF HOT price data can be updated overnight. The same parameters may give different NP values on different run dates. Always re-run the seed params in the same session to get a valid baseline before comparing across attempts.

## Breakout Daily Search — Current State (as of 2026-05-17)

- **Strategy:** `_2021Basic_Break_NQ`
- **Symbol:** `TWF.TXF HOT`
- **Workspace:** `C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp`
- **Insample:** 2019/01/01 – 2026/01/01
- **Objective:** `NP² / |MaxDrawdown|`
- **Target:** NP > 6,000,000 (not yet met)
- **Latest script:** `optimizer/search_daily_target6.py`
- **Detailed reference:** `optimizer/OPTIMIZATION_SKILLS.md`

### Best result so far (Round-6, A11)

| LE | SE | STP | LMT | NP | MDD | Objective | Trades |
|-----|-----|------|------|-----------|-----------|------------|--------|
| 5 | 49 | 0.2 | 16 | 4,074,200 | -634,200 | 26,173,298 | 53 |

Gap to target: −1,925,800 (−32%)

### Key parameter findings

| Region | Verdict |
|--------|---------|
| STP 0.2–0.5 | ✅ Best zone — more trades, smaller MDD, highest NP |
| STP > 2.0 | ❌ NP significantly lower |
| LMT 14–17 | ✅ Optimal |
| LMT > 20 | ❌ NP declines consistently |
| SE 45–55 | ✅ Highest NP |
| SE 80–110 | ⚠️ Smaller MDD but NP capped at ~3.5M |
| LE 5–7 | ✅ Stable |
| LE < 4 | ❌ MDD worsens |
