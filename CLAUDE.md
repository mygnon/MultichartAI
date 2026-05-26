# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a **parameter plateau optimizer** for MultiCharts64 (MC64) trading strategies. It automates MC64's built-in optimizer via Windows UI automation, exports results as CSV, and applies a **plateau detection algorithm** to find parameter combinations that are stable (robust to small parameter perturbations) rather than just peak performers. Results include IS/OOS validation and heatmap visualizations.

The primary strategy is `_2021Basic_Break_NQ` (NQ breakout system). A second strategy `_2021Basic_Break_CL` (CL ATR-based, with LenLE parameter) has also been tested on CL and ZW daily bars. A third strategy `SFJ_15Dworkshop_lesson5_countertrend_LS` (BB counter-trend reversal, no STP/LMT exits) has been tested on NQ hourly (target met) and NQ daily (ceiling confirmed). A fourth strategy `_2021Basic_Osc_NQ` (BB oscillator with ATR-based STP/LMT exits) has been tested across NQ/GC/TXF hourly and daily — all ceilings confirmed.

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
| CME.GC HOT | Hourly | > 800,000 USD | **Ceiling $437,930 (−45.3%) — R3 confirmed** |
| CME.GC HOT | Daily | > 800,000 USD | **Ceiling $467,320 (−41.6%) — R3 confirmed** |
| CME.NQ HOT | Hourly | > 700,000 USD | **✅ MET** (R3: NP=$751,230, LL=17 SL=0.2 LS=45 SS=1.4) |
| CME.NQ HOT | Daily | > 700,000 USD | **Ceiling $460,770 (−34.2%) — R4 confirmed** |
| TWF.TXF HOT | Hourly | > 9,000,000 TWD | **8M MET** (R4: NP=8,101,400); **9M not met** — R6 ceiling confirmed ~8.0M; LS=36 regime Obj=99M but NP ceiling 7.7M |
| TWF.TXF HOT | Daily  | > 8,000,000 TWD | **Ceiling 4,019,800 (−49.8%) — R4 confirmed** |
| TWF.TXF HOT | 240min | > 8,000,000 TWD | **Ceiling 6,958,400 (−13.0%) — R4 confirmed** (LL=10 SL=0.575 LS=53 SS=0.125) |

### `_2021Basic_Osc_NQ` search status

Strategy logic: BUY when C crosses over BollingerBand(C,LEN,LE); SELLSHORT when C crosses under BollingerBand(C,LEN,SE). ATR(10)-based stop (STP×ATR) and limit (LMT×ATR).
Params: LEN (BB period), LE (long-entry stddev, can be negative), SE (short-entry stddev), STP (ATR stop multiplier), LMT (ATR limit multiplier). Workspace: `20260523SFJ_BASIC_OSC_AI.wsp`.
Default: LEN=5, LE=-1, SE=1.75, STP=1, LMT=7.5.

| Instrument | Timeframe | Target NP | Status |
|------------|-----------|-----------|--------|
| CME.NQ HOT | Hourly | > 800,000 USD | **Ceiling $453,610 (−43.3%) — R3 confirmed** (LEN=3 LE=-1.25 SE=1.5 STP=1.0 LMT=18.5; $800K structurally unreachable) |
| CME.NQ HOT | Daily  | > 800,000 USD | **Ceiling $456,190 (−43.0%) — R2 confirmed** (LEN=12 LE=0.1 SE=2.5 STP=0.5 LMT=25.5; ~35 trades/7yr structural cap) |
| CME.GC HOT | Hourly | > 800,000 USD | **Ceiling $312,250 (−61.0%) — R3 confirmed** (LEN=2 LE=-0.75 SE=1 STP=1.6 LMT=30; R1→R2 +0.6%; R2→R3 0.0%; all new territory explored; $400K structurally unreachable) |
| CME.GC HOT | Daily  | > 400,000 USD | **Ceiling $365,690 (−8.6%) — R5 confirmed** (LEN=8 LE=-0.4 SE=2.2 STP=1.8 LMT=7; R4→R5 gain 0.0%; $400K structurally unreachable) |
| TWF.TXF HOT | Hourly | > 7,000,000 TWD | **Ceiling 5,970,000 (−14.7%) — R4 confirmed** (LEN=11 LE=-1.30 SE=3.0 STP=0.975 LMT=33; gains 4.5%/2.5%/1.15%; 7M structurally unreachable) |

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

# GC Hourly NP>800K — countertrend (SFJ_15Dworkshop_lesson5_countertrend_LS, ceiling $437,930 confirmed R3):
py search_gc_ct_hourly3.py --from-csv  # R3 completed — ceiling $437,930 confirmed (0% gain from R2)
py search_gc_ct_hourly2.py --from-csv  # R2 completed — best NP=437,930 (+2.6%; LL=14 SL=0.1 LS=59 SS=0.45)
py search_gc_ct_hourly.py --from-csv   # R1 completed — best NP=426,890 (tight-SS regime discovered)

# GC Daily NP>800K — countertrend (SFJ_15Dworkshop_lesson5_countertrend_LS, ceiling $467,320 confirmed R3):
py search_gc_ct_daily3.py --from-csv  # R3 completed — ceiling $467,320 confirmed (0% gain from R2)
py search_gc_ct_daily2.py --from-csv  # R2 completed — best NP=467,320 (LL=3 SL=1.28 LS=50 SS=2.3; +3.7%)
py search_gc_ct_daily.py --from-csv   # R1 completed — best NP=450,640 (LL=3 SL=1.04 LS=51 SS=2.3; asymmetric short-LL regime)

# GC Daily NP>400K — oscillator (_2021Basic_Osc_NQ, CEILING $365,690 confirmed R5):
py search_gc_osc_daily5.py --from-csv  # R5 completed — CEILING CONFIRMED $365,690 (R4→R5 gain 0.0%; ultra-fine sweeps exhausted)
py search_gc_osc_daily4.py --from-csv  # R4 completed — best $365,690 (LEN=8 LE=-0.4 SE=2.2 STP=1.8 LMT=7; +3.7%)
py search_gc_osc_daily3.py --from-csv  # R3 completed — best $352,550 (LEN=8 LE=-0.4 SE=2.25 STP=1.8 LMT=7; +8.2%)
py search_gc_osc_daily2.py --from-csv  # R2 completed — best $325,750 (LEN=6 LE=-0.6 SE=2.0 STP=1.6 LMT=7; +16.6%)
py search_gc_osc_daily.py --from-csv   # R1 completed — best $279,350 (LEN=11 LE=-1 SE=2.75 STP=0.75 LMT=12.5)

# TXF Hourly NP>7M TWD — oscillator (_2021Basic_Osc_NQ, R4 current):
py search_txf_osc_hourly4.py --from-csv  # R4 completed — CEILING 5,970,000 (LEN=11 LE=-1.30 SE=3.0 STP=0.975 LMT=33; gain 1.15%; 7M unreachable)
py search_txf_osc_hourly3.py --from-csv  # R3 completed — best 5,901,600 (LEN=11 LE=-1.30 SE=3.0 STP=1.0 LMT=33; +2.5%)
py search_txf_osc_hourly2.py --from-csv  # R2 completed — best 5,755,000 (LEN=11 LE=-1.25 SE=3.0 STP=1.0 LMT=33; +4.5%)
py search_txf_osc_hourly.py --from-csv   # R1 completed — best 5,506,400 (LEN=11 LE=-1.25 SE=3.0 STP=1.0 LMT=36; SE boundary hit)

# GC Hourly NP>800K — oscillator (_2021Basic_Osc_NQ, ceiling $312,250 confirmed R3):
py search_gc_osc_hourly3.py             # R3 complete — ceiling $312,250 confirmed (R2→R3 0.0%; all new territory exhausted)
py search_gc_osc_hourly3.py --from-csv  # re-analyze only
py search_gc_osc_hourly2.py --from-csv  # R2 completed — CEILING $312,250 confirmed (R1→R2 +0.6%; A09=A10=A11 immediate convergence; LEN=2 LE=-0.75 SE=1 STP=1.6 LMT=30)
py search_gc_osc_hourly.py --from-csv   # R1 completed — best $310,370 (LEN=2 LE=-0.5 SE=1 STP=1.5 LMT=30)

# NQ Daily NP>800K — oscillator (_2021Basic_Osc_NQ, ceiling $456,190 confirmed R2):
py search_nq_osc_daily2.py --from-csv   # R2 completed — CEILING $456,190 confirmed (A09=A10=A11 convergence; LEN=12 LE=0.1 SE=2.5 STP=0.5 LMT=25.5)
py search_nq_osc_daily.py --from-csv    # R1 completed — best $442,180 (LEN=11 LE=0.25 SE=2.5 STP=0.5 LMT=25.5)
py search_nq_osc_hourly3.py             # R3 COMPLETE — ceiling $453,610 confirmed
py search_nq_osc_hourly3.py --from-csv  # re-analyze only
py search_nq_osc_hourly2.py --from-csv  # R2 completed — ceiling $453,610 confirmed (R1=R2 champion)
py search_nq_osc_hourly.py --from-csv   # R1 completed — best $453,610 (LEN=3 LE=-1.25 SE=1.5 STP=1.0 LMT=18.5)

# NQ Hourly NP>700K — countertrend (SFJ_15Dworkshop_lesson5_countertrend_LS):
py search_nq_ct_hourly3.py --from-csv  # R3 completed — TARGET MET NP=751K
py search_nq_ct_hourly2.py --from-csv  # R2 completed
py search_nq_ct_hourly.py --from-csv   # R1 completed

# TXF 240-Minute NP>800K TWD — countertrend (SFJ_15Dworkshop_lesson5_countertrend_LS):
py search_txf_ct_240min4.py --from-csv # R4 complete — CEILING 6,958,400 (LL=10 SL=0.575 LS=53 SS=0.125; R3→R4 +0.1%)
py search_txf_ct_240min3.py --from-csv # R3 complete — NP=6,951,200 (LL=10 SL=0.575 LS=53 SS=0.14; +3.1%)
py search_txf_ct_240min2.py --from-csv # R2 complete — NP=6,742,400 (LL=12 SL=0.7 LS=52 SS=0.125; +8.0%)
py search_txf_ct_240min.py --from-csv  # R1 complete — NP=6,243,800 (LL=12 SL=0.45 LS=52 SS=0.1)

# TXF Daily NP>8M TWD — countertrend (SFJ_15Dworkshop_lesson5_countertrend_LS, ceiling $4.02M confirmed R4):
py search_txf_ct_daily4.py --from-csv # R4 completed — ceiling 4,019,800 confirmed (0% gain from R3)
py search_txf_ct_daily3.py --from-csv # R3 completed — best NP=4,019,800 (tight-SS regime; LL=25 SL=0.165 LS=50 SS=0.275)
py search_txf_ct_daily2.py --from-csv # R2 completed — best NP=3,978,600 (+21.7%; LL=23 SL=0.13 LS=49 SS=0.27)
py search_txf_ct_daily.py --from-csv  # R1 completed — best NP=3,270,600 (−59.1%)

# TXF Hourly NP>8M TWD — countertrend (SFJ_15Dworkshop_lesson5_countertrend_LS):
py search_txf_ct_hourly6.py --from-csv # R6 completed — 9M not met; LS=36 Obj=99M but NP ceiling 7.7M
py search_txf_ct_hourly5.py --from-csv # R5 completed — best 8.04M (9M not met); found LS=36/SS=1.4 regime
py search_txf_ct_hourly4.py --from-csv # R4 completed — 8M MET NP=8,101,400
py search_txf_ct_hourly3.py --from-csv # R3 completed — best NP=7,928,200
py search_txf_ct_hourly2.py --from-csv # R2 completed — best NP=7,845,000
py search_txf_ct_hourly.py --from-csv  # R1 completed — 7M MET, NP=7,641,000

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
  gc_ct_hourly_search/             # GC Hourly countertrend R1 — best NP=426,890 (tight-SS regime)
  gc_ct_hourly2_search/            # GC Hourly countertrend R2 — best NP=437,930 (+2.6%)
  gc_ct_hourly3_search/            # GC Hourly countertrend R3 — ceiling $437,930 confirmed (0% gain from R2)
  gc_ct_daily_search/              # GC Daily countertrend R1 — best NP=450,640 (LL=3 SL=1.04 LS=51 SS=2.3)
  gc_ct_daily2_search/             # GC Daily countertrend R2 — best NP=467,320 (+3.7%; SL=1.28)
  gc_ct_daily3_search/             # GC Daily countertrend R3 — ceiling $467,320 confirmed (0% gain from R2)
  gc_osc_hourly_search/            # GC Hourly oscillator (_2021Basic_Osc_NQ) R1 — best $310,370 (LEN=2 LE=-0.5 SE=1 STP=1.5 LMT=30)
  gc_osc_hourly2_search/           # GC Hourly oscillator (_2021Basic_Osc_NQ) R2 — CEILING $312,250 confirmed (LEN=2 LE=-0.75 SE=1 STP=1.6 LMT=30; R1→R2 +0.6%)
  gc_osc_hourly3_search/           # GC Hourly oscillator (_2021Basic_Osc_NQ) R3 — CEILING CONFIRMED $312,250 (R2→R3 0.0%; all new territory exhausted; deep LE catastrophic)
  txf_osc_hourly_search/           # TXF Hourly oscillator (_2021Basic_Osc_NQ) R1 — best 5,506,400 TWD (LEN=11 LE=-1.25 SE=3.0 STP=1.0 LMT=36; SE=3 boundary)
  txf_osc_hourly2_search/          # TXF Hourly oscillator (_2021Basic_Osc_NQ) R2 — best 5,755,000 TWD (LEN=11 LE=-1.25 SE=3.0 STP=1.0 LMT=33; +4.5%)
  txf_osc_hourly3_search/          # TXF Hourly oscillator (_2021Basic_Osc_NQ) R3 — best 5,901,600 TWD (LEN=11 LE=-1.30 SE=3.0 STP=1.0 LMT=33; +2.5%)
  txf_osc_hourly4_search/          # TXF Hourly oscillator (_2021Basic_Osc_NQ) R4 — CEILING 5,970,000 TWD (LEN=11 LE=-1.30 SE=3.0 STP=0.975 LMT=33; 7M unreachable)
  gc_osc_daily_search/             # GC Daily oscillator (_2021Basic_Osc_NQ) R1 — best $279,350 (LEN=11 LE=-1 SE=2.75 STP=0.75 LMT=12.5; 3 UI fails; zooms still improving)
  gc_osc_daily2_search/            # GC Daily oscillator (_2021Basic_Osc_NQ) R2 — COMPLETE: $325,750 (LEN=6 LE=-0.6 SE=2.0 STP=1.6 LMT=7; short-LEN regime; +16.6%)
  gc_osc_daily3_search/            # GC Daily oscillator (_2021Basic_Osc_NQ) R3 — COMPLETE: $352,550 (LEN=8 LE=-0.4 SE=2.25 STP=1.8 LMT=7; +8.2%)
  gc_osc_daily4_search/            # GC Daily oscillator (_2021Basic_Osc_NQ) R4 — COMPLETE: $365,690 (LEN=8 LE=-0.4 SE=2.2 STP=1.8 LMT=7; +3.7%)
  gc_osc_daily5_search/            # GC Daily oscillator (_2021Basic_Osc_NQ) R5 — CEILING CONFIRMED $365,690 (R4→R5 0.0%; $400K unreachable)
  nq_osc_daily_search/             # NQ Daily oscillator (_2021Basic_Osc_NQ) R1 — best $442,180 (LEN=11 LE=0.25 SE=2.5 STP=0.5 LMT=25.5)
  nq_osc_daily2_search/            # NQ Daily oscillator (_2021Basic_Osc_NQ) R2 — CEILING $456,190 confirmed (LEN=12 LE=0.1 SE=2.5 STP=0.5 LMT=25.5)
  nq_osc_hourly_search/            # NQ Hourly oscillator (_2021Basic_Osc_NQ) R1 — best $453,610 (LEN=3 LE=-1.25 SE=1.5 STP=1.0 LMT=18.5)
  nq_osc_hourly2_search/           # NQ Hourly oscillator (_2021Basic_Osc_NQ) R2 — ceiling $453,610 confirmed (R1=R2 champion)
  nq_osc_hourly3_search/           # NQ Hourly oscillator (_2021Basic_Osc_NQ) R3 — CEILING CONFIRMED $453,610 (35 attempts; deep LE catastrophic)
  nq_ct_hourly_search/             # NQ Hourly countertrend R1 (best NP=694,910)
  nq_ct_hourly2_search/            # NQ Hourly countertrend R2 (confirmed 694,910 ceiling)
  nq_ct_hourly3_search/            # NQ Hourly countertrend R3 ✅ TARGET MET NP=751,230
  txf_ct_240min_search/            # TXF 240-min countertrend R1 — NP=6,243,800 (LL=12 SL=0.45 LS=52 SS=0.1)
  txf_ct_240min2_search/           # TXF 240-min countertrend R2 — NP=6,742,400 (LL=12 SL=0.7 LS=52 SS=0.125; +8.0%)
  txf_ct_240min3_search/           # TXF 240-min countertrend R3 — NP=6,951,200 (LL=10 SL=0.575 LS=53 SS=0.14; +3.1%)
  txf_ct_240min4_search/           # TXF 240-min countertrend R4 — CEILING 6,958,400 confirmed (R3→R4 +0.1%)
  txf_ct_hourly_search/            # TXF Hourly countertrend R1 ✅ 7M MET NP=7,641,000
  txf_ct_hourly2_search/           # TXF Hourly countertrend R2 — best NP=7,845,000
  txf_ct_hourly3_search/           # TXF Hourly countertrend R3 — best NP=7,928,200
  txf_ct_hourly4_search/           # TXF Hourly countertrend R4 ✅ 8M MET NP=8,101,400
  txf_ct_hourly5_search/           # TXF Hourly countertrend R5 — best 8.04M; found LS=36/SS=1.4 low-MDD regime
  txf_ct_hourly6_search/           # TXF Hourly countertrend R6 — 9M not met; LS=36 Obj=99M but NP ceiling 7.7M; ceiling confirmed ~8.0M
  txf_ct_daily_search/             # TXF Daily countertrend R1 — best NP=3,270,600 (−59.1%)
  txf_ct_daily2_search/            # TXF Daily countertrend R2 — best 3,978,600 (+21.7%; tight-SS regime found)
  txf_ct_daily3_search/            # TXF Daily countertrend R3 — best NP=4,019,800 (tight-SS regime; SL=0.165 SS=0.275)
  txf_ct_daily4_search/            # TXF Daily countertrend R4 — ceiling 4,019,800 confirmed (0% gain from R3)
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

## Current Best Results (as of 2026-05-26)

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

### GC Hourly (countertrend) — Ceiling $437,930 (−45.3%); R3 confirmed

**Best NP** (NP-max, R2 champion confirmed R3):

| LL | SL | LS | SS | NP (USD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 14 | 0.1 | 59 | 0.45 | **437,930** | -46,930 | 4,086,569 | 1,984 |

**Best Obj** (risk-adjusted, new R3 record):

| LL | SL | LS | SS | NP (USD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 14 | 0.1 | 56 | 0.38 | **433,330** | -28,310 | **6,632,811** | 2,046 |

Scripts: `search_gc_ct_hourly.py` (R1) → `search_gc_ct_hourly2.py` (R2) → `search_gc_ct_hourly3.py` (R3 — ceiling confirmed)
Results: `results/gc_ct_hourly3_search/final_params_gc_ct_hourly3.json`

Key findings (36 attempts across R1–R3, progression 426,890→437,930→437,930):
- **Gain rate**: R1→R2 +2.6%, R2→R3 **0.0%** — ceiling confirmed at ~$438K
- **Tight-SS regime is the only viable regime** (LL=14, SL=0.1, LS=56-59, SS=0.38-0.45)
- **LL=14 is uniquely optimal**: LL=25+ (R3 A01/A02) → NP=313K; LL=12 (R2 A08) → NP=385K
- **SL=0.1 is the floor**: SL>0.1 (R3 A07 landscape) always worse; SL<0.1 tested but 0.1 remains optimal
- **SS sweet spot**: SS=0.38 maximizes Obj (MDD=-28,310); SS=0.45 maximizes NP (437,930, MDD=-46,930)
- **SS<0.2 is worse** (R3 A03): very-low SS gives NP≤340K
- **Short LS (<50) is worse** (R3 A05): NP≤341K
- **Extreme LS (≥100) is worse** (R3 A06): NP≤266K; LS=250 → 266K
- **LL scan (14-80, R3 A08)**: LL=14 is the optimal peak; higher LL monotonically worse
- **SL landscape (0.1-2.0, R3 A07)**: SL=0.1 is the global minimum and optimal
- **800K structurally unreachable**: tight-SS regime tops out at ~$438K; ~$220/trade avg × 2000 trades = $438K ceiling
- **Best risk-adjusted choice**: LL=14, SL=0.1, LS=56, SS=0.38 (Obj=6,632,811, MDD=-28,310 = 6.5% of NP)

---

### GC Daily (countertrend) — Ceiling $467,320 (−41.6%); R3 confirmed

**Champion** (LL=3, asymmetric short-LL regime, R2 champion confirmed R3):

| LL | SL | LS | SS | NP (USD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 3 | 1.28 | 50 | 2.3 | **467,320** | -50,780 | 4,300,669 | 64 |

Scripts: `search_gc_ct_daily.py` (R1) → `search_gc_ct_daily2.py` (R2) → `search_gc_ct_daily3.py` (R3 — ceiling confirmed)
Results: `results/gc_ct_daily3_search/final_params_gc_ct_daily3.json`

Key findings (36 attempts, R1–R3 — progression 450,640→467,320→467,320):
- **Gain rate**: R1→R2 +3.7%, R2→R3 **0.0%** — ceiling confirmed at $467,320
- **Asymmetric short-LL regime** (LL=3, SL=1.28, LS=50, SS=2.3): very short LL → many long entries; high SS → rare short entries only at extreme overbought
- **SL=1.27-1.28 is the precise peak** (R3 A01 ultra-fine step=0.01 confirmed): SL=1.3→465K, SL=1.27→467K
- **LS=50-51 plateau** (R3 A07 step=1 confirmed): flat region; LS<47 or LS>55 is worse
- **SS=2.3 confirmed stable** (R3 A03 landscape): SS=2.0-3.6 all tried; SS=2.3-2.4 best (463-467K); SS=3+ collapses to 254K (only 16 trades)
- **High LL (4-14) is worse** (R3 A04): best 421K at LL=4; only 54 trades
- **Tight SL (0.1-0.8) fails on daily** (R3 A06): best 430K at SL=0.8 — GC hourly-like regime doesn't translate
- **High SL (>1.4) fails** (R3 A08): best 283K at SL=1.4; much higher MDD
- **Extreme SS (3-8) collapses** (R3 A05): best 254K at SS=3.0; 16 trades is too few
- **LL=2 with fine SL still no better** (R3 A02): best 457K — LL=2 never beats LL=3
- **Global R3 sweep** confirms no unexplored territory: best 341K
- **Only ~64 trades/7yr**: structural ceiling at $467K confirmed (~64 trades × $7.3K/trade)
- **800K structurally unreachable**: would require ~110 trades (impossible at ~9/yr) or $12.5K/trade avg (impossible with reversal exits)
- **Best recommendation**: LL=3, SL=1.28, LS=50, SS=2.3 (NP=$467K, MDD=-$50.8K = 10.9% of NP)

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

### TXF Hourly (oscillator `_2021Basic_Osc_NQ`) — Ceiling 5,970,000 TWD (−14.7%); R4 confirmed

**Champion** (R4 A04 — STP ultra-fine step=0.025):

| LEN | LE | SE | STP | LMT | NP (TWD) | MDD | Objective | Trades |
|-----|----|----|-----|-----|---------|-----|-----------|--------|
| 11 | −1.30 | 3.0 | 0.975 | 33.0 | **5,970,000** | −841,800 | 42,338,917 | 563 |

**High-Obj alternative** (LEN=19 regime, R4 A07 — lower NP but best Obj):

| LEN | LE | SE | STP | LMT | NP (TWD) | MDD | Objective | Trades |
|-----|----|----|-----|-----|---------|-----|-----------|--------|
| 19 | −1.50 | 3.1 | 1.1 | 33.0 | 5,548,800 | −712,800 | **43,194,699** | 530 |

Scripts: `search_txf_osc_hourly.py` (R1) → `search_txf_osc_hourly2.py` (R2) → `search_txf_osc_hourly3.py` (R3) → `search_txf_osc_hourly4.py` (R4 — ceiling confirmed)
Results: `results/txf_osc_hourly4_search/final_params_txf_osc_hourly4.json`

Round progression (44 attempts, R1–R4): 5,506,400 → 5,755,000 → 5,901,600 → **5,970,000**

Key findings:
- **Gain rate**: R1→R2 +4.5%, R2→R3 +2.5%, R3→R4 +1.15% — halving each round; ceiling definitively confirmed
- **LEN=11 is uniquely optimal**: unlike NQ/GC (LEN=2-3), TXF Hourly prefers medium LEN=11; range 7-25 step=1 all confirmed
- **SE=3.0 is the true optimum**: confirmed from both sides at step=0.05 (range 2.6-3.5); NOT a boundary artifact
- **STP=0.975**: found at step=0.025 precision; not on 0.05 or 0.1 grid — coarser sweeps all showed STP=1.0
- **LE=-1.30**: step=0.05 precision confirmed; R3 step=0.1 found it; R4 step=0.05 confirmed no further improvement
- **LMT=33**: step=0.5 precision confirmed; R3 step=1 found it; R4 step=0.5 confirmed exact
- **LEN=19 regime**: LE=-1.5 SE=3.1 STP=1.1 LMT=33, NP=5.55M MDD=-712,800 Obj=43.2M — better risk-adjusted, lower NP
- **7M structurally unreachable**: all params confirmed at ultra-fine precision; no unexplored territory remains

---

### ZW Daily (_2021Basic_Break_CL) — Ceiling $55K (−92%)

| LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|----|-----|-----|-----|---------|-----|-----------|--------|
| 23 | 3 | 8 | 2 | **54,698** | -15,638 | — | ~30 |

Scripts: `search_zw_cl_daily2.py` (R1), `search_zw_cl_daily3.py` (R2), `search_zw_cl_daily4.py` (R3 — ceiling confirmed)
Result: `results/zw_cl_daily4_search/`

Key findings: LE=23–26 flat plateau; LMT=2 critical (LMT≥3 all worse); STP=7–10 plateau (stop never hit); MDD fixed at −$15,638 across all top results (same worst-drawdown trade regardless of params); **strategy does not work on ZW daily**.

---

### TXF Hourly (countertrend) — **TARGET MET ✅** (R4: NP=8,101,400 TWD); 9M ceiling confirmed R6

**Best NP** (champion by net profit, R4):

| LL | SL | LS | SS | NP (TWD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 22 | 0.425 | 43 | 1.771 | **8,101,400** | -1,461,200 | 44,916,974 | 795 |

**Best Obj** (LS=36 low-MDD regime, discovered R5/R6 — high Obj but NP<9M):

| LL | SL | LS | SS | NP (TWD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 22 | 0.42 | 36 | 1.43 | 7,653,600 | -588,800 | 99,486,401 | 1,039 |

Scripts: `search_txf_ct_hourly.py` (R1) → ... → `search_txf_ct_hourly4.py` (R4 — 8M target met) → `search_txf_ct_hourly5.py` (R5 — LS=36 regime found) → `search_txf_ct_hourly6.py` (R6 — 9M ceiling confirmed)
Results: `results/txf_ct_hourly4_search/final_params_txf_ct_hourly4.json`; R6: `results/txf_ct_hourly6_search/final_params_txf_ct_hourly6.json`

Key findings (72 attempts across R1–R6, progression 7.641M→7.845M→7.928M→8.101M→8.042M→8.043M):
- **Gain rate**: R1→R2 +2.7%, R2→R3 +1.1%, R3→R4 +2.2%; R4→R5/R6 slight data drift (−0.7%) — ceiling at ~8.0-8.1M
- **SL=0.425 is the true peak**: R3 found SL=0.43; R4 step=0.005 found SL=0.425 — the true peak that coarser steps missed
- **9M unreachable**: main regime ceiling ~8.0-8.1M; all R5/R6 attempts confirmed no path to 9M
- **LS=36 low-MDD regime** (R5/R6 discovery): LL=22, SL=0.42, LS=36, SS=1.43, NP=7.65M, MDD=-588K, Obj=99M — extremely attractive risk-adjusted but NP ceiling ~7.7M
- **Two regimes**: (1) main (LS=43-44, SS≈1.77, MDD=-1.46M, NP≈8.0M); (2) LS=36 (SS≈1.43, MDD=-589K, NP≈7.65M, Obj≈99M)
- **Asymmetric regime dominant**: LL=22 (moderate long), LS=43-44 (long short), SL tight (0.425), SS moderate (1.758-1.771)
- **TXF optimal SL ≈ 0.425** (NOT ultra-tight like NQ's SL=0.2)
- **LL=30-80 far worse**: best NP=4.56M — large LL regime not viable

---

### TXF 240-Minute (countertrend) — Ceiling 6,958,400 TWD (−13.0%); R4 confirmed

**Champion** (R4, LL=10 regime — best NP and best Obj):

| LL | SL | LS | SS | NP (TWD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 10 | 0.575 | 53 | 0.125 | **6,958,400** | -930,600 | 52,030,228 | 424 |

Also valid (LL=12 regime, same NP, worse MDD):

| LL | SL | LS | SS | NP (TWD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 12 | 0.72 | 53 | 0.14 | 6,957,200 | -1,003,000 | 48,257,858 | 412 |

Scripts: `search_txf_ct_240min.py` (R1) → `search_txf_ct_240min2.py` (R2) → `search_txf_ct_240min3.py` (R3) → `search_txf_ct_240min4.py` (R4 — ceiling confirmed)
Results: `results/txf_ct_240min4_search/final_params_txf_ct_240min4.json`

Round progression (48 attempts, R1–R4): 6,243,800 → 6,742,400 → 6,951,200 → **6,958,400**

Key findings:
- **Gain rate**: R1→R2 +8.0%, R2→R3 +3.1%, R3→R4 **+0.1%** — ceiling definitively confirmed
- **Tight-SS regime** (SS=0.10-0.14): entirely different from TXF hourly (SS=1.77) — hourly regime tested at 240-min gave only NP=2.38M
- **LL=10 dominates**: better NP AND better MDD than LL=12
- **SL≈0.575**: lower than TXF hourly (0.425) or TXF daily (0.165) — unique to 240-min
- **LS=53 (ODD)**: step=2 grids always skip odd LS values — must use step=1 to find these
- **MDD=-930,600 is structural**: ~13.4% of NP; the same worst-drawdown pattern repeated across all best combos
- **~424 trades/7yr** (~61/year): between daily (~6/yr) and hourly (~114/yr), as expected
- **8M structurally unreachable**: ~424 trades × ~$16.4K/trade = $6.96M ceiling; cannot increase per-trade NP further
- **TXF daily analog** (LL=18, SL=0.2, LS=50, SS=0.15) gave NP=5.16M in R1 — a useful early lead; tight-SS direction was correct but needed fine-tuning

---

### TXF Daily (countertrend) — Ceiling 4,019,800 TWD (−49.8%); R4 confirmed

**Main regime** (R1 best):

| LL | SL | LS | SS | NP (TWD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 20 | 0.28 | 56 | 1.27 | 3,270,600 | -1,189,800 | 8,987,671 | 48 |

**Tight-SS regime** (R3 champion — current best):

| LL | SL | LS | SS | NP (TWD) | MDD | Objective | Trades |
|----|----|----|-----|---------|-----|-----------|--------|
| 25 | 0.165 | 50 | 0.275 | **4,019,800** | -931,000 | 17,356,382 | 45 |

Scripts: `search_txf_ct_daily.py` (R1) → `search_txf_ct_daily2.py` (R2) → `search_txf_ct_daily3.py` (R3) → `search_txf_ct_daily4.py` (R4 — ceiling confirmed)
Results: `results/txf_ct_daily3_search/final_params_txf_ct_daily3.json`

Key findings (R1–R4, 48 attempts, progression 3.27M→3.98M→4.02M→4.02M):
- **Gain rate**: R1→R2 +21.7% (tight-SS regime discovered), R2→R3 +1.0%, R3→R4 **0.0%** — ceiling confirmed at 4,019,800
- **Two regimes**: (1) main (LL=20, SL=0.28, LS=56, SS=1.27, MDD=-1.19M, 48 trades); (2) tight-SS (LL=25, SL=0.165, LS=50, SS=0.275, MDD=-931K, 45 trades) — regime 2 strictly better in both NP and MDD
- **Structural ceiling**: ~45 trades/7yr × ~$89K/trade ≈ $4.0M — daily bar count limits entries
- **SS<0.20 is worse** (R4 A02): best 3,784,800 — tight regime requires SS=0.25-0.30
- **High LL (30-80) is worse** (R4 A03): best 3,395,200 — LL≈25 is optimal
- **Ultra-short LL (2-14) is much worse** (R4 A05): best 2,372,600
- **Extreme LS (100-300) is useless** (R4 A06): NP=1,268,200
- **SL=0.165 is the unique peak** (R4 A11/A12): SL=0.152 gives only 3,951,800 — step=0.005 required
- **8M target structurally unreachable** on daily bars with reversal-only exits; use hourly (R4 hourly met 8M)

---

## Detailed Reference

For full parameter findings, cross-instrument comparisons, and code patterns, see:

```
optimizer/OPTIMIZATION_SKILLS.md
```

Note: `OPTIMIZATION_SKILLS.md` is written in Traditional Chinese (繁體中文). It covers objective function details, parameter range tables, and per-instrument round history in deeper detail than this file.
