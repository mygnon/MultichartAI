# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a **parameter plateau optimizer** for MultiCharts64 (MC64) trading strategies. It automates MC64's built-in optimizer via Windows UI automation, exports results as CSV, and applies a **plateau detection algorithm** to find parameter combinations that are stable (robust to small parameter perturbations) rather than just peak performers. Results include IS/OOS validation and heatmap visualizations.

The primary strategy is `_2021Basic_Break_NQ` (NQ breakout system). A second strategy `_2021Basic_Break_CL` (CL ATR-based, with LenLE parameter) has also been tested on CL and ZW daily bars. A third strategy `SFJ_15Dworkshop_lesson5_countertrend_LS` (BB counter-trend reversal, no STP/LMT exits) has been tested on NQ hourly (target met) and NQ daily (ceiling confirmed).

### `_2021Basic_Break_NQ` search status

| Instrument | Timeframe | Target NP | Status |
|------------|-----------|-----------|--------|
| TWF.TXF HOT | Daily | > 6,000,000 TWD | Not met (-32%) |
| TWF.TXF HOT | Hourly | > 6,000,000 TWD | **✅ MET** |
| CME.NQ HOT | Hourly | > 700,000 USD | Not met (-6.2%) |
| CME.NQ HOT | Daily | > 700,000 USD | Not met (-50%) |
| CME.GC HOT | Hourly | > 700,000 USD | Not met (-58%) |
| CME.GC HOT | Daily | > 700,000 USD | Not met (-55%) |
| CME.CL HOT | Hourly | > 700,000 USD | Not met (-85%) |
| CME.CL HOT | Daily | > 700,000 USD | **Ceiling $15,510 (−97.8%) — strategy does not work** |
| CBOT.ZW HOT | Hourly | > 700,000 USD | Not met (all neg; R5 ready) |
| CBOT.ZW HOT | Daily | > 700,000 USD | **Ceiling $26K (−96%) — strategy does not work** |

### `_2021Basic_Break_CL` search status

| Instrument | Timeframe | Target NP | Status |
|------------|-----------|-----------|--------|
| CME.CL HOT | Daily | > 700,000 USD | **Ceiling $91K (−87%) — strategy does not work** |
| CBOT.ZW HOT | Daily | > 700,000 USD | **Ceiling $55K (−92%) — strategy does not work** |

### `SFJ_15Dworkshop_lesson5_countertrend_LS` search status

Strategy logic: BUY when Close crosses over lower BB; SELLSHORT when Close crosses under upper BB. Reversal exits only — no STP or LMT.
Params: LENGTH_LONG, STDDEV_LONG (long BB), LENGTH_SHORT, STDDEV_SHORT (short BB). Workspace: `20260521SFJ_Bollinger_AI.wsp`.

| Instrument | Timeframe | Target NP | Status |
|------------|-----------|-----------|--------|
| CME.NQ HOT | Hourly | > 700,000 USD | **✅ MET** (R3: NP=$751,230, LL=17 SL=0.2 LS=45 SS=1.4) |
| CME.NQ HOT | Daily | > 700,000 USD | **Ceiling $460,770 (−34.2%) — R4 confirmed** |

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
# TWF Hourly — COMPLETED (NP=6,043,200 TWD ✅)
py search_hourly_target2.py --from-csv

# TWF Daily NP>6M (latest rounds — R8 is current):
py search_daily_target8.py
py search_daily_target8.py --from-csv        # re-analyze only
py search_daily_target8.py --attempt 6       # resume from attempt 6

# NQ Hourly NP>700K — breakout (_2021Basic_Break_NQ, latest round):
py search_nq_hourly3.py
py search_nq_hourly3.py --from-csv

# NQ Hourly NP>700K — countertrend (SFJ_15Dworkshop_lesson5_countertrend_LS):
py search_nq_ct_hourly3.py --from-csv  # R3 completed — TARGET MET NP=751K
py search_nq_ct_hourly2.py --from-csv  # R2 completed
py search_nq_ct_hourly.py --from-csv   # R1 completed

# NQ Daily NP>700K — countertrend (ceiling $461K confirmed after R4):
py search_nq_ct_daily4.py --from-csv   # R4 completed — ceiling $460,770 confirmed
py search_nq_ct_daily3.py --from-csv   # R3 completed
py search_nq_ct_daily2.py --from-csv   # R2 completed
py search_nq_ct_daily.py --from-csv    # R1 completed

# NQ Daily NP>700K (latest round):
py search_nq_daily3.py
py search_nq_daily3.py --from-csv
py search_nq_daily3.py --attempt 6           # resume from attempt 6

# GC Hourly NP>700K (latest round — R5 is current):
py search_gc_hourly5.py
py search_gc_hourly5.py --from-csv

# GC Daily NP>700K (latest round — R4 is current):
py search_gc_daily4.py
py search_gc_daily4.py --from-csv

# ZW Hourly NP>700K (R5 — latest round; R3 all UI-failed):
py search_zw_hourly5.py
py search_zw_hourly5.py --from-csv
py search_zw_hourly5.py --attempt 6          # resume from attempt 6

# ZW Daily NP>700K (R2 complete — ceiling $26K confirmed):
py search_zw_daily2.py --from-csv

# CL Daily (_2021Basic_Break_CL, R1+R2 complete — ceiling $91K confirmed):
py search_cl_cl_daily2.py --from-csv

# ZW Daily (_2021Basic_Break_CL, R3 complete — ceiling $55K confirmed):
py search_zw_cl_daily4.py --from-csv
```

## Install Dependencies

```powershell
pip install -r optimizer\requirements.txt
```

## Architecture

### Core pipeline (3 phases)

1. **MC64 UI Automation** (`mc_automation.py`) — Controls MultiCharts64 via pywinauto + pyautogui to run parameter sweeps and export CSVs. Uses raw ctypes `EnumWindows`/`EnumChildWindows` to bypass UIPI (pywinauto process-scoped specs fail across privilege levels). Key entry point: `run_optimization_for_strategy(conn, cfg, output_dir)`.

   **Speed optimizations applied** (saves ~21s per attempt after the first):
   - *Date range cache* (`_configure_date_range_cache`): module-level cache skips step2 (~32s) when the date range is unchanged between attempts.
   - *Step1+2 skip*: when date is cached, `_open_format_signals` is skipped entirely (~20s saved) — `format_dlg` stays `None` and the code takes the direct right-click path.
   - *Step5a invoke() skip*: the WPF wizard CheckBox never responds to `invoke()` — always use `click_input()` directly (~1.5s saved).

2. **Plateau detection** (`plateau.py`) — Reshapes the flat CSV into a 2D parameter grid. Objective = `NetProfit² / |MaxDrawdown|` (only where both are valid). Computes a sliding-minimum over a `(2r+1)×(2r+1)` neighborhood — a point's **plateau score** is the minimum objective of all its neighbors. High plateau score means the region is uniformly good, not just a spike. Radius defaults to 2 (configurable via `--radius`).

3. **Visualization & reporting** (`visualize.py`) — Dual-panel heatmaps (objective vs. plateau score), HTML summary report, JSON results files.

### Configuration (`config.py`)

All strategies are defined as `StrategyConfig` objects with:
- `mc_signal_name` — exact name in MC's Format Signals dialog
- `chart_workspace` — full path to `.wsp` file (must be open in MC64 before running)
- `params` — list of `ParamAxis(name, start, stop, step)` where `name` must match MC's Inputs tab exactly

`STRATEGY_MAP` maps lowercase/underscore-normalized names to configs. Adding a new strategy requires adding a `StrategyConfig` and updating `ALL_STRATEGIES` and `STRATEGY_MAP`.

### Adaptive search scripts

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
  hourly_target2_search/           # TWF Hourly ✅ COMPLETED
  daily_target6_search/            # TWF Daily (R6, best NP)
  daily_target8_search/            # TWF Daily (R8 latest)
  nq_hourly3_search/               # NQ Hourly (R3 latest)
  nq_daily3_search/                # NQ Daily (R3 latest)
  gc_hourly4_search/               # GC Hourly R4 (latest completed)
  gc_daily3_search/                # GC Daily R3 (latest completed)
  cl_hourly4_search/               # CL Hourly R4 (latest)
  cl_daily3_search/                # CL Daily (_2021Basic_Break_NQ) R3 — ceiling $15K
  zw_hourly4_search/               # ZW Hourly R4 (latest completed)
  zw_daily_search/                 # ZW Daily (_2021Basic_Break_NQ) R1
  zw_daily2_search/                # ZW Daily (_2021Basic_Break_NQ) R2 — ceiling $26K confirmed
  cl_cl_daily_search/              # CL Daily (_2021Basic_Break_CL) R1
  cl_cl_daily2_search/             # CL Daily (_2021Basic_Break_CL) R2 — ceiling $91K confirmed
  zw_cl_daily2_search/             # ZW Daily (_2021Basic_Break_CL) R1
  zw_cl_daily3_search/             # ZW Daily (_2021Basic_Break_CL) R2
  zw_cl_daily4_search/             # ZW Daily (_2021Basic_Break_CL) R3 — ceiling $55K confirmed
  nq_ct_hourly_search/             # NQ Hourly countertrend R1 (best NP=694,910)
  nq_ct_hourly2_search/            # NQ Hourly countertrend R2 (confirmed 694,910 ceiling)
  nq_ct_hourly3_search/            # NQ Hourly countertrend R3 ✅ TARGET MET NP=751,230
  nq_ct_daily_search/              # NQ Daily countertrend R1 (best NP=387,590)
  nq_ct_daily2_search/             # NQ Daily countertrend R2 (best NP=431,500)
  nq_ct_daily3_search/             # NQ Daily countertrend R3 (best NP=456,050)
  nq_ct_daily4_search/             # NQ Daily countertrend R4 — ceiling $460,770 confirmed
```

Each search directory holds:
- `<PREFIX>_<attempt>_raw.csv` — raw MC export for each attempt
- `final_params_<name>.json` — champion parameters
- `search_<name>_*.log` — run log (where present)

## Key Constraints

- **Must run as Administrator** — MC64 runs elevated; UI automation fails otherwise. All `search_*.py` scripts auto-elevate via UAC using `ctypes.windll.shell32.ShellExecuteW(None, "runas", ...)`. Never skip this.
- **MC64 workspace must be open** before running — the automation finds the chart window by matching `chart_symbol` against open window titles.
- **pywinauto limitation** — never use process-scoped `Application(process=pid)` to reach MC64 dialogs; use `Desktop(backend="uia")` + ctypes window enumeration instead. This is the core UIPI workaround.
- **`ParamAxis.name` must match MC exactly** — case-sensitive; the automation types these names into the Inputs dialog.

## Critical Rules for Adaptive Search Scripts

These rules were learned through multiple failed rounds. Violating them silently corrupts results.

### 1. ALL parameters MUST vary in every attempt

When any parameter is fixed (`start == stop`), MC64's MCReport export packs multiple metric sets per row. `pandas` misreads the columns — parameter values appear completely wrong. Always call `_safe()` on every range before building a `StrategyConfig`:

```python
def _safe(t):
    s, e, step = t
    if s == e:
        return (max(LO, s - step), min(HI, s + step), step)
    return t
le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)
```

For `_2021Basic_Break_CL` (5-param strategy with LenLE): apply `_safe()` to LenLE too, using `(95, 105, 5)` as token range. MC64 never actually varies LenLE (always stays at 100), but the range must be non-degenerate to keep the CSV column layout correct.

**Exhaustive checkbox index is dynamic**: for N params, the Exhaustive option checkbox is at index `N+1` (not hardcoded 5). For 4-param strategies index=5, for 5-param strategies index=6. `mc_automation.py` uses `len(cfg.params) + 1` — never hardcode this.

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

Price data can be updated overnight. The same parameters may give different NP values on different run dates. Always re-run the seed params in the same session to get a valid baseline before comparing across attempts.

### 6. champion() zoom seed must use NP-max, not Obj-max

When chasing a target NP, zoom toward the highest NP row, not the highest Objective row. High Objective can come from low MDD with mediocre NP, which leads the search away from the target.

```python
# Target-chasing mode: NP-max as zoom seed
best = pos.loc[pos["NetProfit"].idxmax()]
```

---

## Current Best Results (as of 2026-05-21)

### TWF Hourly — **TARGET MET ✅**

| LE | SE | STP | LMT | NP (TWD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 3 | 76 | 4 | 32 | **6,043,200** | -1,087,800 | 33,572,593 | 886 |

Script: `search_hourly_target2.py` · Result: `results/hourly_target2_search/final_params_hourly_target2.json`

---

### TWF Daily — Not met (best 4,089,800 TWD, gap −32%)

| LE | SE | STP | LMT | NP (TWD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 5 | 49 | 0.2 | 16 | 4,074,200 | -634,200 | 26,173,298 | 53 |

Scripts: `search_daily_target6.py` (R6) → `search_daily_target8.py` (R8 current — exploring STP 0.3–5 gap)
Result: `results/daily_target6_search/final_params_daily_target6.json`

Key findings: STP 0.2–0.5 is the ONLY region producing NP>4M; LMT 14–17 best; SE 45–55 best; LE 5–7 stable. R7 deep-dived STP 0.02–0.5. R8 explores STP 0.3–5 (gap between R7 and original R5 territory, which used STP 2.1 with old data). NP ceiling appears firm at ~4.09M TWD across R6–R8.

---

### NQ Hourly — Not met 700K (best 656,575 USD, gap −6.2%)

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 1 | 9 | 2.0 | 14 | 656,575 | -89,130 | 4,836,651 | 6,168 |

Also worth noting — highest Objective (risk-adjusted): LE=1, SE=10, STP=0.9, LMT=14, NP=595,905, MDD=−59,855, Obj=5,932,717

Script: `search_nq_hourly3.py` · Result: `results/nq_hourly3_search/final_params_nq_hourly3.json`

Key findings: LE=1 only; SE=9–11 (ultra-short); LMT=14 beats LMT=8; NP ceiling ~657K across 3 rounds.

---

### NQ Daily — Not met 700K (best 350,220 USD, gap −50%)

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 1 | 78 | 5.5 | 3.9 | 350,220 | -72,655 | 1,688,171 | 28 |

Script: `search_nq_daily3.py` · Result: `results/nq_daily3_search/final_params_nq_daily3.json`

Key findings: SE=75–85 is the only productive zone; LMT=3.9 tight; **STP is a dead parameter above 5–6** (STP=8 and STP=25 give identical results — stop is never hit); only ~4 trades/year; NP ceiling ~350K.

---

### GC Hourly — Not met 700K (best 292,030 USD, gap −58%)

**Best NP** (unchanged through R4):

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 2 | 35 | 3.2 | 20 | 292,030 | -62,570 | 1,362,978 | 2109 |

**Best Obj** (R4 new record — high LE regime):

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 17 | 90 | 4.5 | 22 | 289,380 | -43,160 | 1,940,241 | 1138 |

Scripts: `search_gc_hourly.py` (R1) → `search_gc_hourly4.py` (R4 completed); `search_gc_hourly5.py` (R5 ready — probing SE>90 with LE=17)
Results: `results/gc_hourly4_search/final_params_gc_hourly4.json`

Key findings: SE=35 + LE=2 is NP-max zone; LE=17 + SE=90 discovered as separate high-Obj regime in R3/R4; SE≥50 hard wall in low-LE regime only; STP=3.2 (LE=2) or STP=4.5 (LE=17); LMT=20 (LE=2) or LMT=22 (LE=17); NP ceiling at 292K for LE=2; LE=17 regime still rising at SE=90 boundary (R5 will probe SE=90–200).

---

### GC Daily — Not met 700K (best 312,520 USD, gap −55%)

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 4 | 49 | 0.9 | 7 | 312,520 | -38,660 | 2,526,352 | 33 |

Scripts: `search_gc_daily.py` (R1) → `search_gc_daily3.py` (R3 completed); `search_gc_daily4.py` (R4 ready — probing unexplored SE/LE territory)
Results: `results/gc_daily3_search/final_params_gc_daily3.json`

Key findings: LMT=7 is a sharp unique integer spike (LMT=6.5→277K, LMT=7.5→247K); STP=0.9 is precise sweet spot (STP<0.7 or >1.1 worse); SE=45–53 flat plateau at 312K; LE=4 best in low-LE regime; very low STP (0.2–0.5) unexplored; high LE (7–50) unexplored; SE>55 unexplored.

---

### CL Hourly — Not met 700K (best 103,190 USD, gap −85%)

**Best NP:** LE=1, SE=54, STP=5.3, LMT=22, NP=103,190, MDD=-49,990, Obj=213,006, trades=2128

**Best Obj (risk-adjusted):**

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 1 | 55 | 1.2 | 36 | 99,120 | -32,700 | 300,452 | 2310 |

Script: `search_cl_hourly4.py` · Result: `results/cl_hourly4_search/final_params_cl_hourly4.json`

Key findings: LE=1 only (direct LE=1–8 sweep confirmed); SE=54–55 only productive zone (SE=2–20 useless, SE>60 drops); LMT dead above 22; three STP local peaks (~1.2, ~3.0, ~5.3) all within 3K NP; NP ceiling $103,190 — absolute, confirmed across 4 rounds / 48 attempts.

---

### CL Daily (_2021Basic_Break_NQ) — Ceiling $15K (−97.8%)

**Best NP:** LE=2, SE=17, STP=1.5, LMT=1, NP=15,510, MDD=-39,830, Obj=6,040, trades=169

**Best Obj (risk-adjusted):**

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 4 | 6 | 2.0 | 1 | 13,930 | -27,760 | 6,990 | 177 |

Scripts: `search_cl_daily.py` (R1), `search_cl_daily2.py` (R2), `search_cl_daily3.py` (R3)
Results: `results/cl_daily_search/`, `results/cl_daily2_search/`, `results/cl_daily3_search/`

Key findings: LMT=1 is the only profitable profit target (LMT≥2 loses money); SE=17 or SE=6 are the only two productive zones; LE=2 is extremely precise (LE=1 and LE≥3 all lose at SE=17); STP=1.5 is a narrow peak (STP=3.5 gives only $8,860); large STP (>5) and extreme STP (50–200) are completely unproductive; NP ceiling **$15,510** — absolute, confirmed across 3 rounds / ~30 attempts; strategy does not work on CL daily bars.

---

### ZW Hourly — Not met 700K (R1–R4 complete; R5 ready)

**R1/R2 best (least negative):** LE=1, SE=150, STP=25, LMT=30, NP=−19,418

- R1: SE=5–150, 12 attempts (~14K combos) — **zero profitable combinations**
- R2: SE=150–500, 12 attempts (~17K combos) — **zero profitable combinations**; 4 attempts UI-failed
- R3: ALL 12 ATTEMPTS UI-FAILED — ZW chart's strategy subchart not visible in workspace; zero data
- R4: Extended exploration (mini-LMT, micro-STP, wide-STP, ultra-SE, large-LE)

Scripts: `search_zw_hourly.py` (R1) → `search_zw_hourly4.py` (R4 completed); `search_zw_hourly5.py` (R5 ready)
Results: `results/zw_hourly4_search/`

---

### ZW Daily (_2021Basic_Break_NQ) — Ceiling $26K (−96%)

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 25 | 2 | 4 | 4 | **25,938** | -6,845 | 98,284 | 30 |

Scripts: `search_zw_daily.py` (R1), `search_zw_daily2.py` (R2 — ceiling confirmed)
Result: `results/zw_daily2_search/`

Key findings: LE=25 is uniquely optimal (LE=30 or LE=13 both ~17K); SE=1–2; STP=LMT≈4¢; only ~5 trades/year; wide STP never hit; **strategy does not work on ZW daily**.

---

### CL Daily (_2021Basic_Break_CL) — Ceiling $91K (−87%)

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 1 | 1 | 4 | 5 | **90,950** | -62,710 | 131,907 | 34 |

Scripts: `search_cl_cl_daily.py` (R1), `search_cl_cl_daily2.py` (R2 — ceiling confirmed)
Result: `results/cl_cl_daily2_search/final_params_cl_cl_daily2.json`

Key findings: LE=1, SE=1 is extremely precise; STP=4 is narrow sweet spot; LMT=5; LenLE always 100 (MC64 ignores automation); only ~34 trades in 7yr; explored LE=1–30, SE=1–70, STP=0.1–18, LMT=1–50 across 24 attempts; **strategy does not work on CL daily**.

---

### NQ Hourly (countertrend) — **TARGET MET ✅** (R3: NP=$751,230)

| LL | SL | LS | SS | NP (USD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 17 | 0.2 | 45 | 1.4 | **751,230** | -64,855 | 8,701,665 | 1,614 |

Also valid (R3 A04):

| LL | SL | LS | SS | NP (USD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 17 | 0.3 | 45 | 1.4 | 739,675 | -74,935 | 7,301,249 | 1,576 |

Scripts: `search_nq_ct_hourly.py` (R1) → `search_nq_ct_hourly2.py` (R2) → `search_nq_ct_hourly3.py` (R3 — target met)
Results: `results/nq_ct_hourly3_search/final_params_nq_ct_hourly3.json`

Key findings (36 attempts across R1–R3):
- **Two regimes exist**: (1) moderate-SL asymmetric (LL=21, SL=1.8, LS=49, SS=2.5, 508 trades, NP=694K); (2) ultra-tight-SL (LL=17, SL=0.2, LS=45, SS=1.4, 1614 trades, NP=751K). Regime 2 beats regime 1 in both NP and MDD.
- **SL=0.2 is the breakthrough**: R1/R2 never probed below SL=1.7. R3 A01 fine-tuned the R2-A12 discovery (LL=15, SL=0.5) and found SL=0.2 with LL=17 gives NP=751K.
- **Asymmetry remains required**: long entry ultra-tight bands (SL=0.2), short entry moderate bands (SS=1.4). Symmetric gives only ~447K.
- **High-frequency is better**: 1614 trades vs 508 — more entries, lower per-trade risk, lower MDD (-$64,855 vs -$82,805).
- R3 A09/A10 had no valid data (rows=0) — likely UI failure during those attempts.

---

### NQ Daily (countertrend) — Ceiling $460,770 (−34.2%)

| LL | SL | LS | SS | NP (USD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 8 | 0.7 | 47 | 1.86 | **460,770** | -81,535 | 2,603,900 | 40 |

Also valid (LL=2 regime, better MDD):

| LL | SL | LS | SS | NP (USD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 2 | any | 12 | 1.9 | 422,335 | -75,690 | 2,356,544 | 78 |

Scripts: `search_nq_ct_daily.py` (R1) → `search_nq_ct_daily2.py` (R2) → `search_nq_ct_daily3.py` (R3) → `search_nq_ct_daily4.py` (R4 — ceiling confirmed)
Results: `results/nq_ct_daily4_search/final_params_nq_ct_daily4.json`

Key findings (48 attempts across R1–R4, progression 387K→431K→456K→461K):
- **Gain rate converged**: R1→R2 +11.3%, R2→R3 +5.7%, R3→R4 +1.0% — ceiling firmly at ~$461K
- **SS=1.86 is the exact peak**: step=0.01 granularity required to find it (step=0.1 in R1/R2 missed it)
- **SL inert range 0.68-0.73**: all give exactly NP=460,770 — strategy is robust to SL in this range
- **MDD=-$81,535 structural floor**: 63%+ of profitable combos share this MDD (same worst trade); unavoidable with reversal-only exits
- **Two regimes**: (1) main asymmetric (LL=8, tight SL, LS=47, SS=1.86, 40 trades, NP=461K); (2) LL=2 high-freq (LL=2, SL inert, LS=12, SS=1.9, 78 trades, NP=422K, better MDD=-75,690)
- **Structural cap**: ~40 trades/7yr × avg $11.5K/trade = $461K; daily bar count limits entries beyond this
- **Ultra-tight SL did NOT help daily** (unlike hourly where SL=0.2 was the breakthrough): SL≤0.3 gives ≤391K
- **LL>8 worse**: LL=10-20 tops at 393K; only LL=6-9 competitive
- **Long LS (>55) worse**: LS=100-500 tops at 329K; LS=47 is the sweet spot

---

### ZW Daily (_2021Basic_Break_CL) — Ceiling $55K (−92%)

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 23 | 3 | 8 | 2 | **54,698** | -15,638 | — | ~30 |

Scripts: `search_zw_cl_daily2.py` (R1), `search_zw_cl_daily3.py` (R2), `search_zw_cl_daily4.py` (R3 — ceiling confirmed)
Result: `results/zw_cl_daily4_search/`

Key findings: LE=23–26 flat plateau; LMT=2 critical (LMT≥3 all worse); STP=7–10 plateau (stop never hit); MDD fixed at −$15,638 across all top results (same worst-drawdown trade regardless of params); **strategy does not work on ZW daily**.

---

## Detailed Reference

For full parameter findings, cross-instrument comparisons, and code patterns, see:

```
optimizer/OPTIMIZATION_SKILLS.md
```

Note: `OPTIMIZATION_SKILLS.md` is written in Traditional Chinese (繁體中文). It covers objective function details, parameter range tables, and per-instrument round history in deeper detail than this file.
