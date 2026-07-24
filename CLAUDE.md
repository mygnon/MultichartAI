# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a **parameter plateau optimizer** for MultiCharts64 (MC64) trading strategies. It automates MC64's built-in optimizer via Windows UI automation, exports results as CSV, and applies a **plateau detection algorithm** to find parameter combinations that are stable (robust to small parameter perturbations) rather than just peak performers. Results include IS/OOS validation and heatmap visualizations.

Workflow: the user opens the right MC64 workspace, requests a search round (target NP, ≤5000 combos per attempt, Objective = NP²/|MDD|), a `search_*.py` script (or a `run_*_pipeline.py` orchestrator) runs the attempts, and results are analyzed for ceiling confirmation via multi-attempt convergence.

**This file keeps only compact summary tables.** Full detail lives in: auto-memory files (`~/.claude/.../memory/project_*.md`), `results/*/state.json` + `final_params_*.json` (deploy params incl. kept-module values), and **this file's git history** (pre-2026-07-20 revisions hold the full per-reference 6-instrument matrices). Conventions: champion shown as param-tuple `NP / MDD / Obj / trades`; "ceiling X% (−n%)" = best NP and gap to target; OOS strict PASS = full |MDD| ≤ IS |MDD| (break× = full/IS MDD ratio); RoMaD = NP/|Max Intraday DD|.

## Search Status Summary (legacy searches — all CLOSED)

### `_2021Basic_Break_NQ` (breakout; LE, SE, STP, LMT)

| Instrument | TF | Target | Status |
|---|---|---|---|
| TWF.TXF | Hourly | >6M TWD | **✅ MET** (LE=3 SE=76 STP=4 LMT=32 NP=6,043,200) |
| TWF.TXF | Daily | >6M TWD | Not met — 4,089,800 (−32%) |
| CME.NQ | Hourly | >700K | Not met — $656,575 (−6.2%; LE=1 SE=9 STP=2 LMT=14) |
| CME.NQ | Daily | >700K | Not met — $350,220 (−50%; LE=1 SE=78 STP=5.5 LMT=3.9) |
| CME.GC | Hourly | >700K | Not met — $292,030 (−58%; LE=2 SE=35 STP=3.2 LMT=20) |
| CME.GC | Daily | >700K | Not met — $312,520 (−55%; LE=4 SE=49 STP=0.9 LMT=7) |
| CME.CL | Hourly | >700K | Ceiling $103,190 (−85%; LE=1 SE=54 STP=5.3 LMT=22) |
| CME.CL | Daily | >700K | $15,510 (−97.8%) — **does not work** |
| CBOT.ZW | Hourly | >700K | All negative — **does not work** |
| CBOT.ZW | Daily | >700K | $26K (−96%) — **does not work** |

### `_2021Basic_Break_CL` (ATR breakout w/ LenLE; LE, SE, STP, LMT, LenLE)

| Instrument | TF | Status |
|---|---|---|
| CME.CL | Daily | $91K (−87%; LE=1 SE=1 STP=4 LMT=5) — **does not work** |
| CBOT.ZW | Daily | $55K (−92%; LE=23 SE=3 STP=8 LMT=2) — **does not work** |

### `SFJ_15Dworkshop_lesson5_countertrend_LS` (BB counter-trend, reversal exits)

Params: LENGTH_LONG (LL), STDDEV_LONG (SL), LENGTH_SHORT (LS), STDDEV_SHORT (SS). Workspace: `20260521SFJ_Bollinger_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| CME.NQ | Hourly | **✅ MET** (LL=17 SL=0.2 LS=45 SS=1.4 NP=$751,230 MDD=−$64,855 1614tr) |
| CME.NQ | Daily | Ceiling $460,770 (−34.2%; LL=8 SL=0.7 LS=47 SS=1.86; 40tr) |
| CME.GC | Hourly | Ceiling $437,930 (−45.3%; LL=14 SL=0.1 LS=59 SS=0.45) |
| CME.GC | Daily | Ceiling $467,320 (−41.6%; LL=3 SL=1.28 LS=50 SS=2.3) |
| TWF.TXF | Hourly | **8M MET** (LL=22 SL=0.425 LS=43 SS=1.771 NP=8,101,400); 9M unreachable |
| TWF.TXF | 240min | Ceiling 6,958,400 (−13.0%; LL=10 SL=0.575 LS=53 SS=0.125) |
| TWF.TXF | Daily | Ceiling 4,019,800 (−49.8%; LL=25 SL=0.165 LS=50 SS=0.275) |

### `SFJ_15Dworkshop_lesson5_countertrend_LS_crypto` (BB counter-trend + `_Crypto1MUSD`)

Workspace `20260101_SFJ_Bollinger_AI.wsp`. `_Crypto1MUSD = Round(1,000,000/C,0)` (~$1M notional/trade).

| Instrument | TF | Ceiling | OOS champion-select |
|---|---|---|---|
| BNBUSDT | Hourly | $36,703 (−63.3%; LL222-236 SL3.8 LS22 SS4.2). ⭐Obj-max live: LL=122 SL=4.025 LS=29 SS=4.2 Obj=173,342 | all broke; least-bad B1 (LL=122) OOS −$10,046 |
| BTCUSDT | Hourly | $4,155 (−58.5%; LL=104 SL=4.05 LS=165 SS=4.95) | **C1 LL=107 SL=4.1 LS=139 SS=4.7 = only PASS** |
| ETHUSDT | Hourly | $5,005 (−50.0%; LL=111 SL=4.025 LS=115 SS=4.725) | best E2 (=NP-max) OOS +$100.5; none holds MDD |
| BNBUSDT | Daily | $42,546 (−57.5%; LL=18 SL=2.175 LS=21 SS=3.15); **Daily>Hourly**, only crypto Daily-CT works | all broke; least-bad BD4 hi-freq LL=4 LS=9 +$879 |
| BTCUSDT | Daily | $3,593 (−64.1%; LL=49 SL=2.4 LS=123 SS=1.2) | **D2 LL=40 LS=130 = only PASS, OOS +$606** (flips lower-freq) |
| ETHUSDT | Daily | $3,755 (−62%; LL=5 SL=1.75 LS=37 SS=1.375 hi-freq) | **ED4 LL=45 LS=100 = only PASS** (held MDD; OOS −$500) |

**CT exit modules (24/24 tests): ALL HURT — keep pure reversal exits** (any exit truncates rare-extreme holds + churns trades 5-15×).

### `SFJ_HUNTER2_NQ` (MA filter + ATR-stop entry, reversal exits, max 1 entry/day)

Params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S. Workspace `20260101_SFJ_HUNTER_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| TWF.TXF | Hourly | **✅ MET** (15/290/0.97/1.5 NP=9,121,800); 10M ceiling 9,129,400 |
| TWF.TXF | Daily | Ceiling 5,393,400 (−40.1%; 15/65/0.17/1.1) |
| CME.NQ | Hourly | Ceiling $634,865 (−29.5%; 8/89/0.25/2.5) |
| CME.NQ | Daily | Ceiling $433,700 (−51.8%; 6/85/0.08/1.15) |
| CME.GC | Hourly | Ceiling $384,820 (−45.0%; 5/37/0.8/5.9) |
| CME.GC | Daily | Ceiling $338,990 (−51.6%; 7/5/0.296/2.068 inverted MA) |

### `SFJ_HUNTER_NQ` (long-only MA + ATR stop entry, fixed STP/LMT, max 1 entry/day)

Params: LEN, STP, LMT. Workspace `20260101_SFJ_HUNTER_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| TWF.TXF | Hourly | Ceiling 5,001,800 (−28.5%; LEN=6 STP=22,837 LMT=606,660) |
| CME.NQ | Hourly | Ceiling $500,015 (−37.5%; LEN=7 STP=550 LMT=13,500) |
| CME.GC | Hourly | Ceiling $379,400 (−52.6%; LEN=144 STP=8,060 LMT=6,500) |
| CBOT.ZW | Hourly | $35,878 (−95.5%) — **does not work** |

### `_2021Basic_Osc_NQ` (BB oscillator, ATR(10)-based STP/LMT)

Params: LEN, LE (can be negative), SE, STP, LMT. Workspace `20260523SFJ_BASIC_OSC_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| CME.NQ | Hourly | Ceiling $453,610 (−43.3%; LEN=3 LE=−1.25 SE=1.5 STP=1.0 LMT=18.5) |
| CME.NQ | Daily | Ceiling $456,190 (−43.0%; LEN=12 LE=0.1 SE=2.5 STP=0.5 LMT=25.5) |
| CME.GC | Hourly | Ceiling $312,250 (−61.0%; LEN=2 LE=−0.75 SE=1 STP=1.6 LMT=30) |
| CME.GC | Daily | Ceiling $365,690 (−8.6%; LEN=8 LE=−0.4 SE=2.2 STP=1.8 LMT=7) |
| TWF.TXF | Hourly | Ceiling 5,970,000 (−14.7%; LEN=11 LE=−1.30 SE=3.0 STP=0.975 LMT=33) |
| TWF.TXF | Daily | Ceiling 5,065,400 (−27.6%; LEN=11 LE=1.2 SE=3.25 STP=1.25 LMT=17) |

### `SFJ_XtremeStop_NQ` (% breakout vs close X bars ago, reversal exits)

Params: X, LY, SY. Workspace `SFJ_XtremeStop_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| CME.NQ | Hourly | Ceiling $624,015 (−22.0%; X=10 LY=1.48 SY=2.18) |
| CME.NQ | Daily | Ceiling $491,590 (−38.6%; X=9 LY=0.025 SY=3.7) |
| CME.GC | Hourly | Ceiling $428,050 (−46.5%; X=7 LY=0.67 SY=1.80) |
| CME.GC | Daily | Ceiling $288,450 (−64.0%; X=1 LY=1.1 SY=2.8) |
| TWF.TXF | Hourly | Ceiling 5,411,000 (−32.4%; X=63 LY=5.155 SY=5.79) |
| TWF.TXF | Daily | Ceiling 4,820,400 (−39.8%; X=1 LY=3.1 SY=4.445) |

### `QuantPassATRex` (ATR+StdDev breakout, reversal exits, `_Crypto1MUSD`)

Params: Len, Su_Multiple, Ni_Multiple. Workspace `20260101_QuantPassATRex_AI.wsp`. (BTC/ETH target >10K, BNB >100K.)

| Instrument | TF | Ceiling (NP-max / Obj-max) |
|---|---|---|
| BTCUSDT | Hourly | $3,293 (−67.1%; Len=125 Su=1.525 Ni=6.15). Risk-adj Len=13 Su=2.52 Ni=3.0 Obj=26,020 |
| BTCUSDT | Daily | $3,511 (−64.9%; Len=24 Su=0.75 Ni=3.5). Obj-max Len=30 Su=0.74 Ni=1.875 |
| ETHUSDT | Hourly | $5,198 (−48.0%; Len=24 Su=1.48 Ni=2.47). ⭐Risk-adj Ni=5.4 Obj=46,902 |
| ETHUSDT | Daily | $3,161 (−68.4%; Len=14 Su=0.83 Ni=1.13) — worst Daily/Hourly drop |
| BNBUSDT | Hourly | $39,921 (−60.1%; Len=235 Su=0.586 Ni=2.128). ⭐Obj-max Len=94 Su=0.715 Ni=1.68 Obj=258,561 |
| BNBUSDT | Daily | $30,876 (−69.1%; Len=93 Su=0.37 Ni=0.53 Obj=163,939; MDD/NP 18.8% best) |

### `QuantPassATR_Breakout` (2-param ATR breakout, reversal exits, `_Crypto1MUSD`)

Params: Len, Multiple. Workspace `20260101_QuantPassATR_Breakout_AI.wsp`.

| Instrument | TF | Ceiling / OOS |
|---|---|---|
| BTCUSDT | Hourly | $2,748 (−72.5%; Len=212 Mult=3.27). **OOS WINNER QB4 Len=8 Mult=2.6 +$947** (IS champ broke 1.40×) |
| BTCUSDT | Daily | $2,544 (−74.6%; Len=13 Mult=1.845; Obj-max Len=10 Mult=1.05) |
| ETHUSDT | Hourly | $4,444 (−55.6%; Len=32 Mult=3.37 Obj=42,191; MDD/NP 10.5%) |
| ETHUSDT | Daily | $3,986 (−60.1%; Len=9 Mult=1.705; NP-max=Obj-max) |
| BNBUSDT | Hourly | $35,634 (−64.4%; Len=3 Mult=2.965). ⭐Obj-max Len=145 Mult=2.91 Obj=229,204 |
| BNBUSDT | Daily | $20,317 (−79.7%; Len=18 Mult=0.735 hi-freq) |

### `SFJ_XtremeStop_Crypto` (% breakout vs C[X], reversal exits, `_Crypto1MUSD`)

Params: X, LY, SY. Workspace `20260101_SFJ_XtremeStop_AI.wsp`. IS 2022/01-2026/01. 3 coins, 3 regimes.

| Instrument | TF | Ceiling | OOS |
|---|---|---|---|
| BNBUSDT | Hourly | ~$26-28K (X≈905 LY≈3.0 SY≈10.65 asym long-X) | Rule-5 FLIP; de-facto BX3 X=14 hi-freq +$8,334 (1.05×) |
| ETHUSDT | Hourly | $3,682 (X=796 LY=5.85 SY=3.35 long-X) | none PASS; only EX2 (X=720) +$439 |
| BTCUSDT | Hourly | $2,606 (X=60 LY=12.75 SY=12.4 sym high-pct) | none PASS; CX1 (=champ) +$570 |
| BNBUSDT | Daily | $23,605 (X=70 LY=0.5 SY=2.2) | all broke+lost; least-bad BD4 X=46 −$94 |
| BTCUSDT | Daily | $4,020 (X=268 LY=2.3 SY=0.1; **Daily>Hourly**) | **✅ ALL 4 PASS; WINNER CD4 X=55 LY=8 SY=6.5 +$532** |
| ETHUSDT | Daily | $3,387 (X=57 LY=3.45 SY=7.55) | (not run) |

**XtremeStop exit modules (BNB+ETH Hourly): TREND-pattern, 6/6 HELP** (BNB M4 +16.68% NP; ETH M6 +9.2%, M5 MDD−19.7%).

### `SFJ_SuperTrend_crypto` (ATR-band trend-flip, reversal, `_Crypto1MUSD`)

Params: ATRLength, Multiplier. Workspace `20260101_SFJ_SuperTrend_AI.wsp`. **Weakest crypto strategy**; wide bands win.

| Instrument | TF | Ceiling | OOS |
|---|---|---|---|
| BNBUSDT | Hourly | $17,453 (79/6.625) | none PASS; best SS3 +$827 |
| BTCUSDT | Hourly | $1,986 (151/9.15) | **TS1 (=champ) PASS +$1,102** (rare IS=OOS) |
| ETHUSDT | Hourly | $3,004 (45/10.45) | **ES1 (=champ) PASS +$2,096** |
| BNBUSDT | Daily | ~$19,169 (21/4.0) — strongest cell, Daily>Hourly | (not run) |
| BTCUSDT | Daily | $1,805 (3/6.5 sparse) | — |
| ETHUSDT | Daily | $2,413 (4/4.5) | **EDS2 4/1 hi-freq PASS +$206** |

**SuperTrend exit modules: HELP** (trend type); full-period joint test winner **M2 TrailingStop** (BTC ATRSTP=51.7, ETH 46.4).

### `SFJ_HUNTER2_crypto` (MA filter + ATR-stop entry, reversal exits, `_Crypto1MUSD`)

Params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S. Workspace `20260101_SFJ_HUNTER_AI.wsp`. Criterion=Obj-max.

| Instrument | TF | Ceiling (Obj-max) | OOS |
|---|---|---|---|
| BTCUSDT | Hourly | 225/825/4.25/1.75 NP=$2,720 Obj=23,082 (ultra-long) | none PASS; BH1 de-facto +$1,498 |
| ETHUSDT | Hourly | 495/635/2.8/0.25 NP=$3,699 (ultra-long) | none PASS; EH2 +$987 |
| BNBUSDT | Hourly | 136/187/3.75/2.75 NP=$20,118 Obj=71,169 (mid-LEN) | **BNH3 150/200/3.5/3.0 = only PASS +$7,654** |
| BNBUSDT | Daily | 2/68/1.0/0.9 NP=$31,731 Obj=119,521 (**Daily>Hourly**, strongest) | **BND1 (=champ) PASS +$2,112** |
| BTCUSDT | Daily | 4/112/0.55/0.75 NP=$2,948 (ultra-short) | **BTD4 27/110/0.35/0.4 PASS +$836** (flips low-freq) |
| ETHUSDT | Daily | 2/116/0.1/0.8 NP=$3,697 (ultra-short) | **EHD4 34/116/0.3/1.0 = only PASS +$698** |

**HUNTER2 exit modules: HELP on Hourly (BNB M5 +30.66% NP), HURT on Daily.** ⚠️ M6 Status reverts after OK on Daily auto-runs → use `--manual-status`.

### `QuantPassRSI` (RSI zero-threshold momentum, reversal exits, `_Crypto1MUSD`)

Params: Len, RSI_Gap. Workspace `20260101_QuantPassRSI_AI.wsp`. Weak (sparse; $100K unreachable).

| Instrument | TF | Status / OOS |
|---|---|---|
| BNBUSDT | Hourly | strongest (Len=56 Gap=34 ~$17.4K). OOS WINNER BS4 +$3,384 |
| BTCUSDT | Hourly | ~$2.2K (Len=62 Gap=34). OOS least-bad RS2 +$515 |
| ETHUSDT | Hourly | ~$3.4K (Len=12-17 Gap=15-19). OOS none PASS; exit modules ALL hurt |
| BTCUSDT | Daily | ~$1.6K. OOS WINNER BSD4 +$892 (low-freq flip) |
| ETHUSDT | Daily | ~$1.2K (weakest). OOS WINNER ESD3 dense Len=2 +$726 |
| BNBUSDT | Daily | ~$6K but Rule-5 data-unstable — **unusable** |

### `_2021Basic_Osc_crypto` (BB-oscillator, STOP entries, ATR(10) STP/LMT, `_Crypto1MUSD`)

Params: LEN, LE, SE, STP, LMT. Workspace `20260101_SFJ_BASIC_OSC_AI.wsp`. Same strategy, 3 OPPOSITE regimes.

| Instrument | TF | Ceiling (Obj-max) | OOS |
|---|---|---|---|
| BNBUSDT | Hourly | **strongest** 13/−0.5/3/1/19.5 NP=$15,113 Obj=119,051 | **WINNER BNO3 hi-freq 15/0/3/2/20 +$4,583 held MDD** |
| ETHUSDT | Hourly | 45/−3/2/1/17 NP=$1,656 (deep-LE mean-reversion) | EO4 +$301 held |
| BTCUSDT | Hourly | 6/2.25/−0.5/2/9 NP=$946 (momentum, LE>0) | **NO PASS — do not deploy** |

### `SFJ_MACD_Strategy03_crypto` (MACD zero-cross entry, histogram-cross exit, `_Crypto1MUSD`)

**Weakest tested.** BNB Hourly Obj-max 62/114/30 NP=$7,957 MDD−$7,065 (MDD/NP 88.8%); only long-period works. Not recommended.

## Self-Authored Breakout Reference Program

Shared engine (the combination that works): **non-lagged breakout level + intrabar STOP fill + wide ATR(14) chandelier trail + ReentryBars post-exit cooldown**. References are numbered variations of the *level construction* (each probes one axis); 4 params Length/BandMult/ATRMult/ReentryBars unless noted; `_crypto` signals use `_Crypto1MUSD`, `_NQ` signals use chart default contracts.

Each reference runs on 6 instruments (BNB/BTC/ETH + TXF/NQ/GC, all Hourly) in one workspace `<X>Breakout_AI.wsp` via `run_<key>_allinst_pipeline.py` + desktop BAT: **Stage-1** IS optimization (self-judged convergence) → **Stage-2** OOS champion-select over 4 regime-diverse candidates → **Stage-3** exit-module IS optimization (main params PINNED via `fixed_inputs`, read-back verified) → **Stage-4** cumulative greedy full-period RoMaD stack; teardown writes final params into the chart Input Strings. Crypto IS 2022/01-2026/01, FULL 2021/03-2026/06; futures IS 2019-2025, FULL 2018-2026. Exit modules: M1 ATRstop, M2 TrailingStop, M3 EntryBarsAfterExit, M4 high_volatility_exit, M5 QuantPass_PT_Exit, M6 RescueTeamExit. **Deploy params (OOS-selected main + kept modules) = `results/<sym>_<key>_hourly_pipeline/state.json`**; per-reference analysis = memory `project_<x>breakout.md`; full matrices = this file's git history.

### Early one-off strategies (BNB Hourly, pre-pipeline era)

| Strategy | Ceiling | Verdict |
|---|---|---|
| **Donchian v2** (+TrendLen) | $24,426 (Obj 100,089) | **OOS PASS DN3 +$5,704**; modules 6/6 HELP; v1 (no cooldown) all-negative |
| **ADXtrend** (DMILen, ADXThresh, ATRMult) | $23,938 | ADX gate=26 the only entry filter that helps IS; OOS no PASS |
| **VolatilityBreakout** (ATRLen, EntryMult, TrailMult) | $22,046 | EntryMult→floor (buffer rejected); OOS no PASS |
| **HeikinAshi** (HASmooth, ATRMult) | $20,689 | short-smooth + wide trail; OOS no PASS |
| **BBSqueeze** | $16,119 | squeeze filter active but OOS no PASS |
| **DonchianAsymV2** (asym + cooldown) | $14,969 | cooldown lifts bare asym +160% |
| **KeltnerTrend** | $10,132 | EMA band < High/Low channel |
| **RSIPullback** | ~$7-8.5K | ⚠️ R1/R2 were OOS-contaminated pre-fix |
| **ROCmomentum** / **DonchianAsym** | ~$5.6-5.8K | weak |
| **ParabolicSAR** / **TurtleChannel** / **FractalBreakout** / **ChannelClose** | broken / all-negative | reversal whipsaw / channel-EXIT whipsaw / lagged pivot / close-confirmed MARKET fills late |

### Tested references (6-instrument matrices)

| # | Reference (axis) | BNB IS Obj | OOS / stack highlights |
|---|---|---|---|
| — | **CloseChannelBreakout** (close-extreme channel; 3-param) | 98,395 ($22.0K) | TXF WINNER +1.09M 1.10×, GC WINNER +64K; BNB CC1 de-facto +$14.5K; crypto modules 6/6 HELP. ⚠️ The 5 per-instrument `run_*_ccb_*_pipeline.py` Stage-3/4 results ran BEFORE the pinning fix (0211a9) = **SUSPECT**; need fixed_inputs wiring + re-run |
| — | **BollingerBreakout** (stdev band; H+D matrix) | 95,486 ($21.2K) | **all 3 coins' DAILY pass OOS, Hourly only BNB → Daily more OOS-robust**; BNB Daily $25,576 = matrix max; module winner flips: M6 Hourly / M5 Daily |
| 5 | **RegChannelBreakout** (linear-regression channel, low-lag) | 110,693 | BTC only strict PASS; NQ stack $440K→$910K RoMaD 15.41; TXF stack 7.45M |
| 6 | **PivotBreakout** (daily pivot, session; 3-param) | 40,005 | weakest main, biggest module rescue: BNB M5 step +505% (largest ever), stack +210% → $41.25K; 0 strict PASS |
| 7 | **MidChannelBreakout** (range midpoint, laggy center) | 67,773 | BNB strict PASS +$12,525; NQ RoMaD 16.14 |
| 8 | `SFJ_VWAPBreakout` | — | built, untested |
| 9 | **HullBreakout** (near-zero-lag MA center) | 86,746 | 0 PASS, 3/6 profitable — low-lag wins IS, overfits OOS (lag law) |
| 10 | **KAMABreakout** (adaptive-lag center) | 68,224 | BNB strict PASS 1.00×; 6/6 OOS-profitable; stack $49.2K |
| 11 | **DayChannelBreakout** (session Donchian, L in days) | 19,759 | L→1-2 days everywhere; BTC+ETH strict PASS 1.00×; BNB stack +402% |
| 12 | **VWMABreakout** (volume-weighted center) | 102,264 | 0 PASS (ETH 3.45×) — volume-WEIGHT joins low-lag camp |
| 13 | **WeekChannelBreakout** | 15,870 | L→1 week; weakest main but mildest breaks (all ≤1.67×); ETH strict PASS |
| 14 | **ERBandBreakout** (ER-adaptive band width) | 113,179 | **1st anti-correlation breaker**: IS-top AND BNB strict PASS +$13,039; stack $47.9K |
| 15 | **OpenRangeBreakout** (daily-reset LW vol) | 60,197 | strongest session main; BTC strict PASS (BTC×session affinity); GC M6 +218% |
| 16 | **InnerChannelBreakout** (HighestLow/LowestHigh + buffer) | **140,235 #1** | BNB stack **$70.6K RoMaD 16.36** = record NP; BTC 1.047× near-PASS |
| 17 | **ConsensusBreakout** (AND of Donchian+Bollinger) | 64,193 | BNB strict PASS **+$14,719 = #1 strict-PASS profit**; IS < both parents |
| 18 | **DecayChannelBreakout** (age-fading levels) | 117,985 | decay confirmed (BandMult interior ×6) but 0 PASS (low-lag camp); GC 1st full-keep; ETH M5 step +251% |
| 19 | **UnionBreakout** (OR of Donchian+Bollinger) | 104,077 | IS beats both parents, OOS all-broke — **composition operator = IS↔OOS dial, not alpha** |
| 20 | **MADBandBreakout** | 138,777 | MAD ≫ stdev (+46% same center); BNB 1.056×/BTC 1.028× near-PASS |
| 21 | **SemiBandBreakout** (per-side dispersion) | 109,136 | direction split COSTS; dispersion axis final: MAD > Semi > stdev |
| 22 | **TypicalChannelBreakout** ((H+L+C)/3 extremes) | 119,203 | ETH strict PASS; **GC stack RoMaD 17.54 = all-time record cell**; TXF full-keep |
| 23 | **POCBreakout** (max-volume-bar anchor) | 46,522 | L→2-11 bars ×6; BNB strict PASS +$7,592; NQ M5 step +170.7%, stack +120% |
| 24 | **RangeSpikeBreakout** (max-range bar; POC control) | 106,997 | 0 PASS, BNB all-broke — **volume adds OOS robustness, not IS edge**; GC full-keep |
| 25 | **HeavyChannelBreakout** (volume-validated extremes) | 60,885 | BNB strict PASS +$14,223 (#2); NQ 1.012× RoMaD 16.22; BandMult→0 ×3 — **volume axis closes: VALIDATE > LOCATE > WEIGHT** |
| 26 | **MultiScaleBreakout** (3-scale level blend) | 52,033 | 0 PASS; L→2-5 except TXF 70 — **level-space averaging ≠ parameter-space plateau** |
| 27 | **RangeFracBreakout** (channel-width fraction) | 111,638 | anti-correlation breaker #3: strict PASS +$13,402, stack $51.1K (#2 NP); fraction lands 0.75-0.93 INSIDE; BTC M6 +326% |
| 28 | **PolarityChannelBreakout** (up/down-bar extremes) | 68,587 | IS champ ITSELF strict PASS +$13,437; 6/6 OOS-profitable; BandMult→0 (2nd validator) |
| 29 | **TrimChannelBreakout** (K-th order-statistic extreme) | 91,827 | K interior 2-3 ×5 (rank-validate); BNB strict PASS +$13,554; **TXF Obj 42.96M = futures record** |
| 30 | **FeedbackChannelBreakout** (loss-streak widening) | — | axis REJECTED (BandMult→0 ×5; cooldown suffices) but bare skeleton: BTC PASS + **GC 1st strict PASS +$59,130** RoMaD 15.48 |
| 31 | **WickBlendBreakout** (field = C + w·(H−C)) | 125,760 #3 | **w optimum INTERIOR ≈0.625-0.75**, beats both endpoints + Typical — "inward but not central" continuous; TXF Obj 40.77M #2; 0 PASS, futures OOS pick w→0 |
| 32 | **SpaceGateBreakout** (space cooldown: re-arm only BandMult·ATR away from last exit) | 60,317 | gate is MARKET-SPLIT: BNB→B=0 (time cooldown suffices), futures+ETH interior 0.5-2.0 — cooldown family: time (universal) > space (futures) > loss-streak (rejected); **BNB 2 candidates strict PASS, winner +$14,223 ties Heavy #2**; ETH stack RoMaD 13.97 = best ETH cell (M5 step +220.5%); TXF Obj 37.67M #4 futures; 5/6 OOS-profitable (TXF only loser) |
| 33 | **RegimeBlendBreakout** (ER-weighted blend of Donchian extreme + Bollinger band) | 107,721 | **COMPOSITION TRIAD CLOSES: the dynamic dial escapes the trade-off** — IS beats BOTH parents AND Union's static OR (104K) AND strict PASS 1.00× (+$13,800 #4) = 4th anti-correlation breaker; BNB all-4 candidates OOS-profitable, breaks ≤1.27×; stack $48.6K RoMaD 9.76; **NQ IS Obj 6.29M = best NQ IS ever** (but OOS 2.8× — NQ anti-correlation intact); TXF kept 5 modules; 3/6 OOS-profitable |
| 34 | **AgedChannelBreakout** (Donchian window excludes freshest K bars; K=BandMult, 0=Donchian) | 92,244 | **BNB ALL 4 candidates strict PASS 1.00× (first ever); winner +$15,318 = #1 all-time strict-PASS profit**; stack $53.7K = #2 all-time NP; extreme-age axis closes two-sided with Decay: recent extremes carry the signal (K→0-2 on strong cells, TXF/NQ K=0); TXF 1.054× near-PASS +508K; NQ IS Obj 4.81M (#2 NQ) but OOS 4.1× |
| 35 | **VolClockBreakout** (vol-clock Donchian: window N = Length·ATRbase/ATR14) | 55,917 | EVENT-TIME REJECTED for IS: vol-rescaling the lookback weakens every strong cell (BNB 55.9K ≪ Donchian 100K; TXF 26.6M; NQ 2.65M) — the fixed-bar window is already the right clock. But OOS-benign: **6/6 OOS-profitable** (3rd ever), BNB strict PASS +$14,247 (3rd ~14.2K bare-skeleton PASS: Heavy/SpaceGate/VolClock — same underlying regime); NQ 1.145× mild |
| 36 | **DualAnchorBreakout** (outermost of yesterday's extreme + rolling Donchian = anchor-space AND) | 25,116 | anchor-AND confirms "strictness costs IS" but WITHOUT Consensus's OOS payoff (0 strict PASS; BNB collapses to L=291 ultra-long, OOS −$5,250) — the AND reward depends on WHAT is AND-ed; **GC 178/0/1.75/24 stack RoMaD 17.77 = NEW ALL-TIME RECORD cell** (long-window tight-trail GC family again), OOS 1.03× near-PASS +56,840; **BNB+NQ both kept ALL 6 modules (first double full-keep)**, BNB stack +233%; NQ RoMaD 11.55; 4/6 OOS-profitable; zero-flake run |
| 37 | **BodySpikeBreakout** (max \|C−O\| bar; trio closes) | 107,357 | body ≈ range (≈ RangeSpike, both OOS-broke) — only VOLUME carries independent OOS info; **GC OOS +$135,970 = largest ever**; exposed seed-clamp bug (fixed 5b46b3b) |
| 38 | **HLMeanBreakout** (mean-of-highs shelf: Average(High,L) + BandMult·ATR) | 126,949 | **4th anti-correlation breaker**: BNB IS #2 all-time AND strict PASS 1.00× +$13,635; BTC 2/4 candidates PASS 1.00× (+$596); **NQ stack RoMaD 16.23 = new NQ record** (5 modules; dethrones NQ-Heavy 16.22); order-statistic gradient closes NON-monotone in rank: mean 127K > max(Donchian) 100K > K-th(Trim) 91.8K — the statistical shelf beats the spike, deep-inward endpoint wins (Law 5); TXF all-4-candidates OOS-NEGATIVE (winner −261,800, 2.94×); 5/6 OOS-profitable; triple-flake run (ETH/BNB/TXF), all recovered clean |

| 39 | **VolRatioBreakout** (vol-regime band width: EffBand = BandMult·ATR14·(ATR14/ATRslow), quadratic) | 138,191 #3 | **width-modulator axis closes as an IS↔OOS dial**: vol² modulation beats ERBand on IS (138.2K > 113.2K, #3 all-time behind Inner/MADBand) but loses ERBand's strict PASS (0 PASS; BNB 1.024× near-PASS +$9,691); **6/6 OOS-profitable (4th ever)**; TXF +704,000 = 2nd-largest TXF OOS (CCB +1.09M); NQ collapses to ultra-high-freq L=72 Re=0 2732tr; M3 kept 5/6 (historically least-kept); zero-flake run |

| 40 | **ERGateBreakout** (ERBand's ER gate on the Donchian EDGE: buffer = BandMult·ATR·(1−ER); B=0 = exact Donchian, nested A/B) | 85,779 | **ER-transplant 1/3: the OOS robustness transfers, the IS edge does NOT** — B interior 0.75-2.0 everywhere (the gate IS used, never →0) yet BNB IS 85.8K ≪ ERBand-on-SMA 113.2K; **BNB strict PASS 1.00× +$12,466**; **ETH stack RoMaD 12.99 = #2 ETH cell ever** (OOS +$1,860 high-tier, M5 step +39.5%); BTC IS Obj 17.6K (top BTC tier), break 1.125×; NQ all-4-candidates OOS-NEGATIVE (2nd all-negative cell, after HLMean-TXF); TXF +472,000 but kept only M5 (rare single-module keep); GC sparse 79tr weak; 5/6 OOS-profitable |

| 41 | **ERTrailBreakout** (ER on the TRAIL slot: EffTrail = ATRMult·ATR·(1+BandMult·(2ER−1)); B=0 = fixed trail) | 94,834 | **ER-transplant 2/3: the trail slot REJECTS ER modulation** — BandMult→0-0.25 on 5/6 cells (BNB winner exactly B=0; only TXF interior 0.75-1.0): the fixed wide chandelier is already right (Law 1 reconfirmed from inside; echoes VolClock's fixed-clock verdict). Futures IS strong (**NQ Obj 4.91M = #2 NQ ever**, GC 1.88M top-tier, TXF 25.2M) but crypto OOS flips negative: **BNB −$1,470 = 2nd BNB OOS loss ever**, ETH −$333; 0 strict PASS, 4/6 OOS-profitable; opposite market-split to ERGate (crypto-good/futures-bad) — ER value is slot×market specific |

### Untested queue (pipelines + BATs ready; all carry the 5b46b3b seed/zoom clamp fix)

42 ERPause (closes the ER-transplant factorial: gate=OOS-transfers, trail=rejected, pause=?), 43 DuoAdapt.

### Records ledger

- **Strict-PASS OOS profits (BNB):** **AgedChannel 15,318 (all-4-candidates PASS, first ever)** > Consensus 14,719 > VolClock 14,247 ≈ Heavy 14,223 ≈ SpaceGate 14,223 > RegimeBlend 13,800 > HLMean 13,635 > Trim 13,554 > Polarity 13,437 > RangeFrac 13,402 > ERBand 13,039
- **RoMaD cells:** **GC-DualAnchor 17.77** > GC-Typical 17.54 > **NQ-HLMean 16.23** > NQ-Heavy 16.22 > NQ-MidChannel 16.14 > GC-Feedback 15.48 > NQ-RegChannel 15.41
- **Stacked NP (BNB):** Inner 70.6K > AgedChannel 53.7K > RangeFrac 51.1K > Polarity 50.9K > HLMean 49.8K > VolRatio 49.6K ≈ WickBlend 49.4K ≈ KAMA 49.2K
- **Futures IS Obj (TXF):** Trim 42.96M > WickBlend 40.77M > RangeFrac 38.55M > Feedback 34.6M > Heavy 32.77M
- **Module single steps:** Pivot-BNB M5 +505% > RangeFrac-BTC M6 +326% > Decay-ETH M5 +251% > Union-GC M5 +243% > DayChannel-BNB M5 +220% > OpenRange-GC M6 +218%
- **Module keep total: 216/216 pinned cells keep ≥1 module** (every matrix since the pinning fix); M5 PT_Exit most-kept > M6 > M2/M1 > M4 > M3. GC full-keeps (all 6): Decay, RangeSpike, Heavy-era GC

### Laws & axis conclusions

1. **The engine is the alpha**: non-lagged level + intrabar STOP + wide ATR chandelier trail + ReentryBars cooldown. The EXIT is decisive (channel exit / SAR break it); a lagged pivot or close-confirmed MARKET entry breaks it.
2. **Entry filters fail OOS** (ADX/squeeze/RSI/trend/vol-buffer → floor); ReentryBars cooldown is the ONE entry-side lever kept >0. Cooldown family (refs 30+32): time cooldown universal > price-space re-arm gate market-split (futures+ETH keep it interior 0.5-2.0, BNB→0) > loss-streak escalation rejected.
3. **IS strength ≠ OOS robustness**: the IS Obj champion (lowest IS-MDD, sparse) is repeatedly OOS-worst; higher-freq / wider-MDD regimes generalize. Exceptions ("anti-correlation breakers", IS-top AND strict PASS): **ERBand, RangeFrac, RegimeBlend, HLMean**; near: Heavy, MADBand.
4. **Lag law** (5 confirmations): low-lag centers (Hull, VWMA, RegChannel, Decay) win IS but overfit OOS; laggy/adaptive centers (MidChannel, KAMA) are IS-weaker but OOS-robust.
5. **Field law**: monotone INWARD — Inner 140K > HLMean(mean-of-highs) 127K ≈ WickBlend(w≈0.7) 126K > Typical 119K > Donchian 100K > Close 98K; continuous form: the optimum is "inward but not central" (WickBlend w≈0.7 ≈ RangeFrac fraction 0.75-0.93). Order-statistic sweep (ref 38) is NON-monotone in rank: mean 127K > max 100K > K-th 91.8K — the deep-inward statistical shelf beats both the spike and the trimmed spike.
6. **Dispersion axis**: MAD 139K > SemiBand 109K > stdev 95K — robust dispersion helps, per-side splitting hurts.
7. **Volume axis**: VALIDATE (Heavy) > LOCATE (POC) > WEIGHT (VWMA); volume buys OOS robustness, not IS edge (RangeSpike/BodySpike price-only controls both OOS-broke). Event-anchor trio: body ≈ range, only volume independent.
8. **Validation family** (filter WHICH bars' extremes count — volume/polarity/rank): all 3 give BNB strict PASS and BandMult→0 (the filter replaces the price buffer).
9. **Anchor granularity**: rolling → day → week = edge↓, robustness↑; session anchors produce BTC strict PASSes (Day/Week/OpenRange). **Composition triad**: static AND/OR is an IS↔OOS dial, not alpha — but the ER-weighted dynamic BLEND (ref 33) escapes the dial (IS above both parents AND strict PASS).
10. **Extreme-age law** (refs 18+34, two-sided): recent extremes carry the signal — fading STALE levels helps (Decay BandMult interior ×6), while excluding the FRESHEST 1-2 bars is at best a small crypto refinement (Aged K→0-2; TXF/NQ take K=0).
11. **Width-modulator axis** (refs 14+39, two-sided): modulating band width by vol regime (VolRatio, quadratic ATR²/ATRslow) beats modulating by trend efficiency (ERBand) on IS (138.2K > 113.2K) but LOSES the strict PASS ERBand keeps — the modulator choice is another IS↔OOS dial; both are 6/6-profitable-grade mild-break designs.
12. **Exit modules on trend-breakout mains genuinely HELP (162/162) — but ONLY with verified pinning** of the main's champion params (`fixed_inputs`, read-back). Any module result without verified pinning is invalid (the CCB per-instrument pipelines' S3/4 remain SUSPECT). A module stack lifts NP/RoMaD, not generalization (Pivot-BNB +210% but still weak-OOS). On counter-trend/reversal mains modules HURT (CT 24/24, QPRSI-ETH).

## Burn-in Codegen (`burner/`)

Fuses each instrument's pipeline result (state.json main_champ + Stage-4 KEEP modules) into ONE self-contained EL signal with params baked as input defaults, for zero-config Portfolio Trader mounting at scale. Spec: `burn-in-codegen-spec.md` (+ `oms-spec.md`); procedure doc: `docs/burn_equivalence.md`.

```powershell
py -m burner burn --name DualAnchorBreakout --key dualanchor [--inst btc,gc] [--no-modules] [--dry-run]
py -m burner verify --name DualAnchorBreakout --key dualanchor --dry-run   # equivalence checklist (no MC64)
py -m burner verify --name DualAnchorBreakout --key dualanchor            # same-day A/B gate (MC64, signal compiled manually first)
py -m burner export-modules    # Knowledge/*.docx -> Strategy/modules/*.txt (7 files, diff-locked by tests)
```

Outputs `burned/{Name}/{Name}_{INST}_{TF}_v{n}.txt|.manifest.json` + `burn_report.json`; idempotent (identical rerun reuses v{n}, content change bumps, old versions never overwritten). Key rules: main body VERBATIM (only input defaults rewritten); module inputs/vars prefixed `m{n}_` (whitelist whole-token case-insensitive rename — M6 `Length` collides with main); M1/M3 unnamed orders get `"M1_LX"`-style names; module declarations HOISTED above the main's first executable statement (PL wants declarations first). **OMS emit block v4 (zero-throw, own-handle, pointer-free)**: writes `Z:\oms\signals\{id}.json` (ramdisk, path constant `templates.OMS_SIGNALS_DIR`); guarded `GetAppInfo(aiRealTimeCalc)=1` (backtests never write). Three field-found landmines shaped it: (v1) EL has NO try/catch and a file-builtin error DISARMS AOE (`C:\oms` missing-dir crash); (v2) **MC's FileAppend never releases its handle** (tmps stayed locked, rename/delete all sharing-violation, no .json ever published); (v3) **MC's DefineDLLFunc rejects byref pointer params** (WriteFile lplong → "Incorrect argument type"). So v4 uses NO EL file builtins and only by-value WinAPI: self-healing `CreateDirectoryA` + `GetFileAttributesA` skip-guard (ramdisk reboot wipe self-recovers), unique `.tmp` per write (GetTickCount+seq), `_lcreat`/`_lwrite`/`_lclose` (legacy kernel32, handle closed before rename, written-length verified), `MoveFileExA` retried 5×30ms, any failure → `DeleteFileA`+skip (next bar re-emits). Template bump requires recompiling the burned signals in MC64. Lock stress/audit tool: `py -m burner.tools.signal_lock_stress --file <json>|--scan-dir <dir>`. Equivalence gate (spec §4.4) = same-day A/B (original multi-signal vs burned signal, ±0.5%) — never compare against stale manifest numbers (Critical Rule 5). Tests: `py -m pytest tests/` (goldens byte-exact; regen via `py tests/regen_goldens.py` then review).

## Running the Optimizer

All scripts must run as Administrator because MC64 runs elevated and Windows UIPI blocks cross-privilege UI automation. MC64 must be open with the correct workspace before running.

```powershell
pip install -r optimizer\requirements.txt

# Core pipeline — run configured strategies:
optimizer\run_optimizer.bat
cd optimizer; py main.py --strategies all          # auto-elevates via UAC
py main.py --strategies all --from-csv ..\results  # re-analyze existing CSVs, no MC64
py main.py --strategies breakout_daily --dry-run   # check grid size
```

### Adaptive search scripts

Naming convention: `optimizer/search_<symbol>_<strategy>_<timeframe><round>.py` (round suffix omitted for R1). Strategy abbreviations: (none)=Break_NQ, `cl`=Break_CL, `ct`=countertrend_LS, `osc`=Osc_NQ, `xtreme`=XtremeStop, `hunter`/`hunter2`=HUNTER/HUNTER2, `qpatrex`=QuantPassATRex, `qpatr_breakout`=QuantPassATR_Breakout.

```powershell
cd optimizer
py search_btc_ct_hourly3.py              # run a round (11-12 attempts; auto-elevates)
py search_btc_ct_hourly3.py --from-csv   # re-analyze existing CSVs only
py search_btc_ct_hourly3.py --attempt 6  # resume from attempt 6
```

Each round's output goes to `results/<symbol>_<strategy>_<timeframe><round>_search/` containing `<PREFIX>_<attempt>_raw.csv` per attempt, `final_params_<name>.json` (champion + all attempts), and run logs. Desktop BAT launchers (`Run_*.bat`) exist for recent searches — the user double-clicks them because UAC cannot prompt from a background shell.

### One-BAT 4-stage pipeline orchestrators

`run_<key>_allinst_pipeline.py` (6-instrument, canonical: clone of the KAMA/MidChannel orchestrator) and the older `run_<sym>_ccb_*_pipeline.py` (single-instrument) run all 4 stages in one MC64 connection, handing params stage→stage in memory and writing `results/<dir>/state.json` after every stage. CLI: `--instrument <sym>`, `--from-stage N` (resume), `--from-csv` (re-analyze), `--_elevated`. Hardening in the allinst orchestrators: `mc.activate_chart_by_symbol` MDI-activates+maximizes the target chart before every wizard launch; `set_signal_statuses` retries; per-instrument `conn.connect()` reconnect (fixes multi-hour stale-handle); best-effort Stage-4 teardown writes final params into Input Strings. Programmatic Format-Objects input-setting is read-back-verified and ABORTs loudly on mismatch. Recurring flake: "Signal status not verified during apply" kills an instrument mid-run → restart MC64 + reopen workspace as Admin, re-run with `--instrument <sym> --from-stage N` via a desktop recovery BAT (cached CSVs replay; every recovery so far ran clean). **Pause/resume**: press `p` in the (elevated) console window to pause at the next safe point between MC64 operations, `p` again to resume (`pause_gate.py`, hooked in `mc_automation` — works in all pipelines/searches; MC64 must stay open while paused).

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

Standalone multi-attempt scripts performing sequential zooming: wide sweeps to find stable regions, then fixed-axis refinement, then adaptive zoom attempts (typically A09-A11) around the champion. Each attempt reads from an existing CSV if present (idempotent). The final champion is saved to `results/<subdir>/final_params_*.json`. These scripts hardcode `WORKSPACE`, `SYMBOL`, `SIGNAL`, and `OUTPUT_DIR` constants at the top — edit there when targeting a different strategy. `search_btc_qpatrex_hourly.py` is the canonical pattern for new scripts.

### Multi-signal module workflow (exit-module optimization)

`mc_automation.py` exposes a Signal Status API for charts with a fixed MAIN signal plus multiple exit-module signals toggled one at a time:
- `get_signal_status(format_dlg, name)` / `set_signal_status(format_dlg, name, enabled)` — read/toggle the Status checkbox with read-back verification (3-tier: checkbox click → row+Space → coordinate click). Never assumes a click worked.
- `set_signal_statuses(conn, status_map, verify=True, protected=[...])` — opens Format Signals, applies all statuses, OK-commits, reopens to verify; raises on mismatch; refuses to disable `protected` signals.
- In the optimization wizard, rows are matched by param name + `cfg.mc_signal_name`, so a module run uses `StrategyConfig(mc_signal_name=<module>, params=<module params>)`; the main signal's inputs stay unchecked at their Format-Signals values. The exported CSV contains ALL chart input columns — validate per row that the main signal's params stayed at the champion values (excluding columns whose name collides with an enabled module's param name, e.g. RescueTeamExit `Length`).
- `search_bnb_ct_exit_modules.py` is the canonical pattern (A00 same-day baseline re-measure + one attempt per module + teardown; `--module N`, `--manual-status`, `--smoke` flags). Probe UIA structure first with `probe_signal_status.py --dump / --toggle <name> / --wizard-dump`.
- ⚠️ The Data-Range trim does NOT re-restrict already-loaded wider data: an exit-module run launched right after an OOS run on FULL range silently runs full-period — `ensure_chart_ready` before `set_instrument_data_range`, set+verify the chart range first, and a baseline-drift>10% ABORT guard catches it.

### Column name mapping

MC64 exports different column headers across versions. `MC_COLUMN_MAP` in `config.py` normalizes them to `NetProfit`, `MaxDrawdown` (= "Max Intraday Drawdown"), `TotalTrades`. If a new MC version uses different headers, add entries there.

## Key Constraints

- **Must run as Administrator** — MC64 runs elevated; UI automation fails otherwise. All `search_*.py` / `run_*_pipeline.py` scripts auto-elevate via UAC using `ctypes.windll.shell32.ShellExecuteW(None, "runas", ...)` with the `--_elevated` flag. Never skip this.
- **MC64 workspace must be open** before running — the automation finds the chart window by matching `chart_symbol` against open window titles.
- **pywinauto limitation** — never use process-scoped `Application(process=pid)` to reach MC64 dialogs; use `Desktop(backend="uia")` + ctypes window enumeration instead. This is the core UIPI workaround.
- **`ParamAxis.name` must match MC exactly** — case-sensitive; the automation types these names into the Inputs dialog.
- **BAT files must be pure ASCII** — Traditional Chinese Windows uses CP950 (Big5). Non-ASCII characters such as the em dash `—` (U+2014) are invalid Big5 sequences and cause CMD misparsing (`echo` → `ec`+`ho`, `cd /d` → `cd`+`/d`), leaving the working directory wrong. Verify with: `[System.IO.File]::ReadAllBytes($path) | Where-Object { $_ -gt 127 }`.
- **MC64 connection in search scripts**: always `conn = mc.MultiChartsConnection(); conn.connect()`. Never `mc.connect_to_mc()` — that function does not exist.
- **Wizard optimization-TYPE = Regular (not Walk-Forward)** — the first optimization-wizard page ("Choose the Optimization Type") has 3 RadioButtons (Regular / Walk-Forward / Matrix) whose UIA **names are EMPTY**; each label ("Regular Optimization" …) is a SEPARATE sibling `Text` right after its radio. Clicking the label does nothing, and pywinauto `.select()` on the radio is a silent no-op → MC keeps its default (often Walk-Forward), silently producing WFO results. Fix (`_select_regular_optimization` in `mc_automation.py`): pair each radio with the following label Text, PHYSICALLY `click_input()` the 'regular' radio (DPI-correct; coord fallback), then VERIFY via `is_selected()` and retry. MC remembers the last-used type, so once set to Regular it stays sticky. Log shows `'Regular Optimization' already selected` / `Selected ... radio (verified)`.
- **MC64 export truncation** — sparse-trade grids (e.g. QPATR_Breakout Mult>2 with Len>20) export only partial rows; failed 3× across BTC/ETH/BNB Daily. Structural MC64 limitation; design grids to avoid these regions or accept the gap. Truncated-but-clean exports are accepted if they still start at the declared min, else garble→discard.
- **TRUE OOS isolation via chart Data Range** — MC64 ignores the signal Begin-date; only the CHART's loaded data range restricts the backtest. `mc.set_instrument_data_range(conn, from, to)` sets Format Instruments → Settings → Data Range (From-To radio + both date pickers) so an IS pass ≠ FULL pass, giving OOS = NP_full − NP_is. **Critical**: `DTM_SETSYSTEMTIME` sets the picker value but does NOT fire `DTN_DATETIMECHANGE`, so OK won't apply it (silent IS==FULL) — after the DTM set the code nudges the picker (click + Right + Up/Down, net-zero) to fire the notification. Verify with `--probe-instrument` + a chart screenshot. **⚠️ Settings-tab silent-fail (fixed ae19bdf):** when the data source shows "not connected", the dialog opens on the Lookup/Add-Symbol tab (no date pickers) and the old code clicked OK applying nothing → silent OOS-contamination. Fix: `set_instrument_data_range` retries the Settings-tab click (≤4×) until ≥2 `SysDateTimePick32` pickers are present, else Cancels and **raises**. Symptom of a contaminated IS run: NP collapses once the trim actually applies.
- **Wizard read-back compares display-rounded values** — a module target like DAYRANGE=5.11 can display as '5.1' and abort that step (KAMA-TXF M1). Prefer grid steps whose values round-trip through the wizard display.

## Critical Rules for Adaptive Search Scripts

These rules were learned through multiple failed rounds. Violating them silently corrupts results.

### 1. ALL parameters MUST vary in every attempt

When any parameter is fixed (`start == stop`), MC64's MCReport export packs multiple metric sets per row and `pandas` misreads the columns. Always call `_safe()` on every range before building a `StrategyConfig`:

```python
def _safe(t):
    s, e, step = t
    if s == e:
        return (max(LO, s - step), min(HI, s + step), step)
    return t
```

For `_2021Basic_Break_CL` (5-param with LenLE): apply `_safe()` to LenLE too, using `(95, 105, 5)` as token range — MC64 never varies it, but the range must be non-degenerate to keep the CSV layout correct.

**Exhaustive checkbox location**: the wizard CheckBox list is `CB[0]=select-all, CB[1..N]=one per grid row, CB[N+1]=Exhaustive` where **N = total grid rows (all inputs of all ENABLED signals)**, which equals `len(cfg.params)` only on single-signal charts. `mc_automation.py` locates it by name ("exhaustive"/"窮舉") first, then actual `DataItem` row count + 1, then legacy `len(cfg.params)+1`. Never hardcode.

### 2. Validate every loaded CSV with `_validate_df()`

After loading a CSV, check every parameter column's values fall within the expected range; if not, the CSV is garbled — discard it and record 0 results for that attempt.

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

A `while combos > 5000` loop recalculating with the same radius never converges. Use a `for` loop over decreasing radii and `break` on first fit:

```python
for r_se, r_stp, r_lmt in [(12, 1.0, 4), (8, 0.8, 3), (6, 0.6, 2), (4, 0.4, 1)]:
    _c = _cfg(name, _le, zoom(best_se, r_se, ...), zoom(best_stp, r_stp, ...), zoom(best_lmt, r_lmt, ...))
    if _c.total_runs() <= 5000:
        break
```

### 4. Limit each attempt to ≤ 5000 combos

### 5. NP numbers are not stable across days

Price data can be updated overnight (especially crypto). The same parameters may give different NP on different run dates. Always re-run the seed params in the same session to get a valid baseline before comparing across attempts. Several "breakthroughs" (BTC QPATRex Hourly R2 $3,507; BTC QPATR_Breakout Daily R1 $3,227) turned out to be data-drift artifacts — dedicate retest attempts in the next round to verify any cross-day champion.

### 6. champion() zoom seed must use NP-max, not Obj-max

When chasing a target NP, zoom toward the highest NP row, not the highest Objective row (high Obj can come from low MDD with mediocre NP):

```python
best = pos.loc[pos["NetProfit"].idxmax()]
```

### 7. Ceiling confirmation protocol

A ceiling is confirmed when: round-over-round gain ≤ ~0.1%, AND multiple attempts (including adaptive zooms) converge on byte-identical NP/MDD/trades, AND new-territory sweeps (global, boundary-push, alternative regimes) all fall short. Report both NP-max and Obj-max champions — they are often different regimes. ⚠️ Before declaring a ceiling, widen every axis (esp. STDDEV/Length) — a too-narrow axis caused a false ceiling on BNB CT Daily.

### 8. Clamp every seed, zoom center, and confirm-grid endpoint into declared bounds

`ParamAxis.values()` computes `np.linspace` sample counts from `stop-start` — an inverted range (stop < start) crashes with "Number of samples, -1, must be non-negative". A champion loaded from a cached CSV of an unclamped legacy grid can sit OUTSIDE `*_LO..*_HI` and invert both the confirm grids and the zoom ranges. Three-layer fix (in all `run_*_allinst_pipeline.py` since 5b46b3b): clamp the Stage-1 seed tuple `min(max(v, LO), HI)`, clamp the zoom center before building zoom ranges, and clamp confirm-grid uppers with `min(HI, ...)`. Fix the actual crash site and verify with the crashing input.

## Detailed Reference

- `optimizer/OPTIMIZATION_SKILLS.md` — objective function details, parameter range tables, per-instrument round history (Traditional Chinese)
- Auto-memory files (`~/.claude/projects/.../memory/project_*.md`) — full per-search findings, per-reference matrices, convergence tables, regime analysis
- `results/<search>/final_params_*.json` + `results/<pipeline>/state.json` — champion + all-attempt/all-stage data (state.json = authoritative deploy params)
- This file's git history (pre-2026-07-20) — the original full per-reference 6-instrument matrix tables
