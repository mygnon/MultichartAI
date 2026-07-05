# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a **parameter plateau optimizer** for MultiCharts64 (MC64) trading strategies. It automates MC64's built-in optimizer via Windows UI automation, exports results as CSV, and applies a **plateau detection algorithm** to find parameter combinations that are stable (robust to small parameter perturbations) rather than just peak performers. Results include IS/OOS validation and heatmap visualizations.

Workflow: the user opens the right MC64 workspace, requests a search round (target NP, ≤5000 combos per attempt, Objective = NP²/|MDD|), a `search_*.py` script (or a `run_*_pipeline.py` orchestrator) runs the attempts, and results are analyzed for ceiling confirmation via multi-attempt convergence.

**This file keeps only compact summary tables.** Full per-search detail (convergence tables, regime analysis, blow-by-blow OOS/exit-module numbers) lives in the auto-memory files (`~/.claude/.../memory/project_*.md`) and `results/*/final_params_*.json` + `results/*/state.json`. Conventions used below: champion shown as param-tuple `NP / MDD / Obj / trades`; "ceiling X% (−n%)" = best NP and gap to target; "OOS" rows = champion-select on FULL data (PASS = full |MDD| ≤ IS |MDD|); RoMaD = NP/|Max Intraday DD|.

## Search Status Summary

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

| Instrument | TF | Ceiling (NP-max / Obj-max) | OOS champion-select |
|---|---|---|---|
| BNBUSDT | Hourly | $36,703 (−63.3%; LL222-236 SL3.8 LS22 SS4.2 10tr). ⭐Obj-max live: LL=122 SL=4.025 LS=29 SS=4.2 $35,112 MDD−$7,112 Obj=173,342 | all broke; least-bad B1 (LL=122) OOS −$10,046 |
| BTCUSDT | Hourly | $4,155 (−58.5%; LL=104 SL=4.05 LS=165 SS=4.95 22tr). Obj-max LS=68 MDD−$431 Obj=38,853 | **C1 LL=107 SL=4.1 LS=139 SS=4.7 = only PASS** (held MDD, OOS −$378 least loss) |
| ETHUSDT | Hourly | $5,005 (−50.0%; LL=111 SL=4.025 LS=115 SS=4.725 25tr; Obj-max LS≈109 MDD−$580) | best E2 (=NP-max) OOS +$100.5; none holds MDD (all 1.5×+) |
| BNBUSDT | Daily | $42,546 (−57.5% of 100K; LL=18 SL=2.175 LS=21 SS=3.15 14tr); **BNB Daily>Hourly**; only crypto Daily-CT works | all broke; least-bad BD4 hi-freq LL=4 LS=9 +$879 (1.32×) |
| BTCUSDT | Daily | $3,593 (−64.1%; LL=49 SL=2.4 LS=123 SS=1.2 16tr); Daily<Hourly | **D2 LL=40 LS=130 = only PASS, OOS +$606** (FLIPS to lower-freq) |
| ETHUSDT | Daily | $3,755 (−62%; LL=5 SL=1.75 LS=37 SS=1.375 47tr hi-freq) | **ED4 LL=45 LS=100 = only PASS** (held MDD; OOS −$500, all lost) |

**CT exit modules (BNB/BTC/ETH Hourly + BTC Daily, 24/24 tests): ALL HURT — keep pure reversal exits.** Adding any exit truncates rare-extreme holds + churns (trades 5-15×). Full-period validation confirmed no module improves NP or cuts MDD. (Workspace `20260101` charts end ~2026/01 → IS==FULL for the validation runs.)

### `SFJ_HUNTER2_NQ` (MA filter + ATR-stop entry, reversal exits, max 1 entry/day)

Params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S. Workspace `20260101_SFJ_HUNTER_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| TWF.TXF | Hourly | **✅ MET** (LEN_L=15 LEN_S=290 ATR_L=0.97 ATR_S=1.5 NP=9,121,800); 10M ceiling 9,129,400 |
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
| BTCUSDT | Daily | $3,511 (−64.9%; Len=24 Su=0.75 Ni=3.5 13tr). Obj-max Len=30 Su=0.74 Ni=1.875 |
| ETHUSDT | Hourly | $5,198 (−48.0%; Len=24 Su=1.48 Ni=2.47). ⭐Risk-adj Ni=5.4 MDD−$493 Obj=46,902 |
| ETHUSDT | Daily | $3,161 (−68.4%; Len=14 Su=0.83 Ni=1.13 41tr) — worst Daily/Hourly drop |
| BNBUSDT | Hourly | $39,921 (−60.1%; Len=235 Su=0.586 Ni=2.128). ⭐Obj-max Len=94 Su=0.715 Ni=1.68 MDD−$3,905 Obj=258,561 |
| BNBUSDT | Daily | $30,876 (−69.1%; Len=93 Su=0.37 Ni=0.53 83tr Obj=163,939; MDD/NP 18.8% best) |

### `QuantPassATR_Breakout` (2-param ATR breakout, reversal exits, `_Crypto1MUSD`)

Params: Len, Multiple. Workspace `20260101_QuantPassATR_Breakout_AI.wsp`.

| Instrument | TF | Ceiling / OOS |
|---|---|---|
| BTCUSDT | Hourly | $2,748 (−72.5%; Len=212 Mult=3.27 90tr). **OOS: 2 PASS, WINNER QB4 Len=8 Mult=2.6 OOS +$947** (long-Len IS-champ QB1 broke 1.40×) |
| BTCUSDT | Daily | $2,544 (−74.6%; Len=13 Mult=1.845 13tr; Obj-max Len=10 Mult=1.05). Daily<Hourly |
| ETHUSDT | Hourly | $4,444 (−55.6%; Len=32 Mult=3.37 58tr Obj=42,191; MDD/NP 10.5%) |
| ETHUSDT | Daily | $3,986 (−60.1%; Len=9 Mult=1.705 14tr; NP-max=Obj-max). ETH Breakout Daily > QPATRex Daily |
| BNBUSDT | Hourly | $35,634 (−64.4%; 13-conv; Len=3 Mult=2.965 82tr). ⭐Obj-max Len=145 Mult=2.91 Obj=229,204 |
| BNBUSDT | Daily | $20,317 (−79.7%; Len=18 Mult=0.735 114tr hi-freq). Daily/Hourly −43% worst |

### `SFJ_XtremeStop_Crypto` (% breakout vs C[X], reversal exits, `_Crypto1MUSD`)

Params: X, LY, SY. Workspace `20260101_SFJ_XtremeStop_AI.wsp`. IS 2022/01-2026/01. **3 coins, 3 regimes.**

| Instrument | TF | Ceiling | OOS |
|---|---|---|---|
| BNBUSDT | Hourly | ~$26-28K (X≈905 LY≈3.0 SY≈10.65 asym long-X) | Rule-5 FLIP; de-facto BX3 X=14 short-X hi-freq +$8,334 (1.05×) |
| ETHUSDT | Hourly | $3,682 (X=796 LY=5.85 SY=3.35 long-X 46tr MDD−$450) | none PASS; only EX2 (X=720) +$439 (IS champ EX1 broke 5.7×) |
| BTCUSDT | Hourly | $2,606 (X=60 LY=12.75 SY=12.4 **sym high-pct** 13tr) | none PASS; CX1 (=champ) broke $2 +$570; hi-freq collapsed |
| BNBUSDT | Daily | $23,605 (X=70 LY=0.5 SY=2.2 43tr); Daily<Hourly | all broke+lost; least-bad BD4 X=46 −$94 (1.05×); IS champ collapsed |
| BTCUSDT | Daily | $4,020 (X=268 LY=2.3 SY=0.1 long-X 11tr); **Daily>Hourly** | **✅ ALL 4 PASS** (DD IS-locked); WINNER CD4 X=55 LY=8 SY=6.5 +$532 |
| ETHUSDT | Daily | $3,387 (X=57 LY=3.45 SY=7.55 15tr); Daily<Hourly | (not run) |

**XtremeStop exit modules (BNB+ETH Hourly): TREND-pattern, 6/6 HELP** (opposite of CT). BNB (main BX3): M4 DAYRANGE=5.08 +16.68% NP, M5 PT=0.27 MDD−8%. ETH (main EX2): M6 Len=540 +9.2% NP, M5 PT=0.589 +7.2% NP & MDD−19.7%.

### `SFJ_SuperTrend_crypto` (ATR-band trend-flip, reversal, `_Crypto1MUSD`)

Params: ATRLength, Multiplier. Workspace `20260101_SFJ_SuperTrend_AI.wsp`. **Weakest crypto strategy.** IS 2022/01-2026/01.

| Instrument | TF | Ceiling | OOS |
|---|---|---|---|
| BNBUSDT | Hourly | $17,453 (ATR=79 Mult=6.625 wide-band 220tr) | none PASS; best SS3 low-freq +$827 |
| BTCUSDT | Hourly | $1,986 (ATR=151 Mult=9.15 127tr; =BNB/8.8) | **TS1 (=champ) PASS, OOS +$1,102** (IS-best=OOS-best, rare) |
| ETHUSDT | Hourly | $3,004 (ATR=45 Mult=10.45 103tr) | **ES1 (=champ) PASS, OOS +$2,096** (like BTC) |
| BNBUSDT | Daily | **~$19,169 (ATR≈21 Mult≈4.0 11tr) — strongest SuperTrend cell; Daily>Hourly** | (not run) |
| BTCUSDT | Daily | $1,805 (ATR=3 Mult=6.5 10tr sparse spike) | — |
| ETHUSDT | Daily | $2,413 (ATR=4 Mult=4.5 12tr) | **EDS2 ATR=4 Mult=1 hi-freq PASS +$206** (wide-band champ too sparse) |

**SuperTrend exit modules (BTC+ETH Hourly): HELP** (trend type). M5 PT adds most NP IS; full-period OOS joint test (NP↑ AND MDD↓): **M2 TrailingStop PASSES on both** (BTC ATRSTP=51.7, ETH 46.4).

### `SFJ_HUNTER2_crypto` (MA filter + ATR-stop entry, reversal exits, `_Crypto1MUSD`)

Params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S. Workspace `20260101_SFJ_HUNTER_AI.wsp`. IS 2022/01-2026/01. Criterion=Obj-max. (regimes differ by coin/TF.)

| Instrument | TF | Ceiling (Obj-max) | OOS |
|---|---|---|---|
| BTCUSDT | Hourly | 225/825/4.25/1.75 NP=$2,720 MDD−$320 Obj=23,082 (ultra-long) | none PASS; BH1 (=champ) de-facto +$1,498 (1.28×) |
| ETHUSDT | Hourly | 495/635/2.8/0.25 NP=$3,699 Obj=31,283 (ultra-long) | none PASS; EH2 490/665 +$987 (1.93×); hi-freq collapsed |
| BNBUSDT | Hourly | 136/187/3.75/2.75 NP=$20,118 Obj=71,169 (mid-LEN) | **BNH3 150/200/3.5/3.0 = only PASS, OOS +$7,654** (only crypto HUNTER2 PASS) |
| BNBUSDT | Daily | 2/68/1.0/0.9 NP=$31,731 Obj=119,521 (ultra-short; **Daily>Hourly**, strongest) | **BND1 (=champ) PASS, OOS +$2,112** (IS=OOS aligned) |
| BTCUSDT | Daily | 4/112/0.55/0.75 NP=$2,948 (ultra-short) | **BTD4 27/110/0.35/0.4 PASS +$836** (FLIPS to low-freq) |
| ETHUSDT | Daily | 2/116/0.1/0.8 NP=$3,697 (ultra-short; R1 LEN_L=34 was false peak) | **EHD4 34/116/0.3/1.0 = only PASS +$698** (IS-best ultra-short collapsed −$1,041) |

**HUNTER2 exit modules: HELP on Hourly, HURT on Daily.** Hourly all positive (BNB M5 PT=0.069 +30.66% NP & MDD−25%). Daily: BTC+ETH all negative. M5 flips role by TF (Hourly NP-booster → Daily MDD-slasher). ⚠️ M6 Status reverts after OK on Daily auto-runs → use `--manual-status`.

### `QuantPassRSI` (RSI zero-threshold momentum, reversal exits, `_Crypto1MUSD`)

Params: Len, RSI_Gap. Workspace `20260101_QuantPassRSI_AI.wsp`. Weak (sparse 10-22tr; $100K unreachable).

| Instrument | TF | Status / OOS |
|---|---|---|
| BNBUSDT | Hourly | strongest QPRSI (Len=56 Gap=34 ~$17.4K MDD−$12.2K). OOS WINNER BS4 +$3,384 |
| BTCUSDT | Hourly | ~$2.2K (Len=62 Gap=34). OOS least-bad RS2 +$515 |
| ETHUSDT | Hourly | ~$3.4K (Len=12-17 Gap=15-19). OOS none PASS |
| BTCUSDT | Daily | ~$1.6K (Len=63 Gap≈48). OOS WINNER BSD4 +$892 (low-freq flip) |
| ETHUSDT | Daily | ~$1.2K (Len=50 Gap=47, weakest). OOS WINNER ESD3 dense Len=2 +$726 |
| BNBUSDT | Daily | ~$6K but Rule-5 data-unstable — **unusable** |

OOS law: sparse Obj-max champions collapse; denser regimes generalize. ETH Hourly exit modules ALL hurt (reversal type).

### `_2021Basic_Osc_crypto` (BB-oscillator, STOP entries, ATR(10) STP/LMT, `_Crypto1MUSD`)

Params: LEN, LE, SE, STP, LMT (LE/SE = STDDEV mults, LE can be negative). Workspace `20260101_SFJ_BASIC_OSC_AI.wsp`. **Same strategy, 3 OPPOSITE regimes.**

| Instrument | TF | Ceiling (Obj-max) |
|---|---|---|
| BNBUSDT | Hourly | **strongest** LEN=13 LE=−0.5 SE=3 STP=1 LMT=19.5 NP=$15,113 MDD−$1,919 Obj=119,051 (asym) |
| ETHUSDT | Hourly | LEN=45 LE=−3 SE=2 STP=1 LMT=17 NP=$1,656 (deep-LE mean-reversion) |
| BTCUSDT | Hourly | LEN=6 LE=2.25 SE=−0.5 STP=2 LMT=9 NP=$946 (momentum, LE>0) |

OOS: **BNB WINNER BNO3** (hi-freq LEN=15 LE=0 SE=3 STP=2 LMT=20) +$4,583 held MDD; **ETH EO4** +$301 held; **BTC NO PASS** (do not deploy). IS tight-MDD Obj-max overfits; hi-freq regime generalizes.

### `SFJ_MACD_Strategy03_crypto` (MACD zero-cross entry, histogram-cross exit, `_Crypto1MUSD`)

Params: FastLength, SlowLength, MACDLength. Workspace `20260101_SFJ_MACD03_AI.wsp`. **Weakest tested.** BNB Hourly Obj-max Fast=62 Slow=114 MACD=30 NP=$7,957 MDD−$7,065 (MDD/NP 88.8%); short/mid all-losing, only long-period works. Not recommended.

### Self-authored crypto strategies (BNB Hourly unless noted; `_Crypto1MUSD`; workspaces `20260622_*`)

Common engine that works = rolling/non-lagged breakout level + intrabar STOP fill + WIDE ATR(14) chandelier trail + ReentryBars post-exit cooldown.

| Strategy (params) | BNB Hourly ceiling | Notes / OOS |
|---|---|---|
| **Donchian v2** (Length, ATRLength, ATRMult, TrendLen, ReentryBars) | NP-max **$24,426** (2/12/16/15/10 wide-trail 66tr); Obj-max 3/16/7/10/17 Obj=100,089 | v1 all-negative (no anti-chop); **OOS WINNER DN3 4/—/7/5/17 +$5,704 = only PASS**. BTC mid-long+TrendLen-active; ETH ultra-long+tight M=2. BNB Daily 10/10/4.5/65/14 $19,045. Exit modules 6/6 HELP (M5 PT=0.15 +64% NP & MDD↓) |
| **ADXtrend** (DMILen, ADXThresh, ATRMult) | **$23,938** (12/26/7.8 Obj=86,068) | 2nd-strongest; ADX gate=26 the ONLY filter that genuinely helps. OOS: NO PASS (all broke) |
| **RegChannelBreakout** (RegLen, BandMult, ATRMult, ReentryBars) | **$23,576** (14/1.19/7/12 **Obj=110,693 #1 risk-adj**) | linear-regression channel = 5th breakout reference; see matrix below. Pinned exit modules: 6/6 KEEP (NQ stack $910K RoMaD 15.41). BTC = only strict OOS PASS |
| **PivotBreakout** (PivMult, ATRMult, ReentryBars) | $15,107 (0.25/13/22 Obj=40,005) | daily-pivot breakout (PP±PivMult*priorday-range, HighD/LowD/CloseD) = 6th reference; **WEAKEST main, biggest module beneficiary** (BNB +M5 → $41.25K +210%). see matrix below. OOS no strict PASS (NQ break 1.005× near-miss) |
| **MidChannelBreakout** (Length, BandMult, ATRMult, ReentryBars) | $18,029 (19/1.0/7/27 Obj=67,773) | Donchian range-MIDPOINT ±BandMult*ATR (center=(HH+LL)/2) = 7th reference; mid-tier. Pinned exit modules: 6/6 KEEP (NQ stack RoMaD 16.14 = best cell). **BNB strict OOS PASS +$12,525 (1.00×); 5/6 OOS-profitable, mild breaks** |
| **HullBreakout** (Length, BandMult, ATRMult, ReentryBars) | $21,928 (14/0.35/7/14 Obj=86,746) | Hull-MA center ±BandMult*ATR (near-zero-lag WMA-based MA) = 9th reference; mid-upper tier (IS ~ Bollinger). Pinned exit modules: 6/6 KEEP. **OOS WEAKER: no strict PASS, 3/6 profitable — low-lag center overfits** |
| **VolatilityBreakout** (ATRLen, EntryMult, TrailMult) | **$22,046** (45/~0.01/16.5) | EntryMult→floor (buffer rejected) → degenerates to wide trail. OOS: NO PASS; IS champ VB1 worst −$14,696; best VB3 125/2.25/8 +$4,924 (1.31×) |
| **CloseChannelBreakout** (Length, ATRMult, ReentryBars) | **$22,008** (8/7/14 Obj=98,395 **#1 risk-adj**) | see dedicated matrix below |
| **BollingerBreakout** (BBLen, BBmult, ATRMult, ReentryBars) | **$21,210** (13/1.55/7/13 Obj=95,486) | see dedicated matrix below |
| **HeikinAshi** (HASmooth, ATRMult) | $20,689 (3/8.25 Obj=67,428) | short-smooth + wide trail. OOS: NO PASS; best HA3 4/16 +$2,153 (1.57×) |
| **BBSqueeze** (BBLen, BBmult, SqueezeLen, ATRMult) | $16,119 (14/1.5/50/8) | squeeze filter active. OOS: NO PASS |
| **DonchianAsymV2** (LongLen, ShortLen, ATRMult, ReentryBars) | $14,969 (6/300/7/15) | asym+cooldown lifts bare asym +160%; ReentryBars kept >0 |
| **KeltnerTrend** (EMALen, BandMult, ATRMult) | $10,132 (107/1.0/5.4) | weaker than Donchian (EMA band < High/Low channel) |
| **RSIPullback** (TrendLen, RSILen, RSIThresh, ATRMult) | ~$7-8.5K (clean IS, not converged) | ⚠️ R1/R2 were OOS-contaminated pre-fix |
| **ROCmomentum** (ROCLen, ROCThresh, ATRMult) | $5,643 (148/2.25/3) | high-threshold thesis failed; wants low threshold |
| **DonchianAsym** (LongLen, ShortLen, ATRMult) | ~$5,754 (17/250/8) | only asym profitable; needs cooldown (→ V2) |
| **ParabolicSAR** (AfStep, AfMax) | $4,626 razor-spike (MDD/NP 267%) | **does not work** (always-in reversal whipsaws) |
| **TurtleChannel** (EntryLen, ExitLen) | all-negative | **does not work** — channel EXIT whipsaws (the exit is decisive, not the entry) |
| **FractalBreakout** (FracLen, ATRMult) | all-negative | **does not work** — lagged pivot vs rolling extreme |
| **ChannelClose** (ChanLen, ATRMult) | degenerate (MDD/NP 408%) | **does not work** — close-confirmed MARKET entry fills late (fixed by CloseChannelBreakout) |

### CloseChannelBreakout matrix (crypto `_crypto` + futures `_NQ` port)

Crypto = `SFJ_CloseChannelBreakout_crypto` (`_Crypto1MUSD`). Futures = `SFJ_CloseChannelBreakout_NQ` (no sizing → chart default contracts), via the 1-BAT 4-stage orchestrator `run_<sym>_ccb_*_pipeline.py` (IS→OOS-select→exit-modules→cumulative greedy RoMaD stack); futures IS 2019-2025 / FULL 2018-2026.

| Cell | IS champ (Length/ATRMult/ReentryBars) | OOS / deploy |
|---|---|---|
| BNB Hourly (crypto) | 8/7/14 $22,008 Obj=98,395 (#1 risk-adj) | CC3 10/7/12 strict PASS; **CC1 de-facto +$14,496 (broke $43)** |
| BTC Hourly (crypto) | 7/8/19 $1,866 | all broke; CC4 wide-trail de-facto +$495 (IS champ collapsed 2.98×) |
| ETH Hourly (crypto) | ~$4,100-4,700 wide-trail (Rule-5) | all broke; CC4 narrow-trail de-facto +$767 (wide-trail IS champ was overfit) |
| BNB Daily (crypto, pipeline) | 2/5/12 $25,576 (byte-identical to BNB BollingerBreakout Daily) | **deploy MAIN ONLY 5/5/5 OOS +$8,277** (3/4 PASS); exit 6/6 HURT → kept ∅ |
| BTC Daily (crypto, pipeline) | 2/3.25/10 $1,624 | **deploy MAIN ONLY 2/4.5/10** (only NP-max PASS, OOS −$67; Obj-max broke 3.15×); kept ∅ |
| ETH Daily (crypto) | ~$1.6-1.9K narrow-trail (sparse, Obj-creep = noise) | all broke; CC3 R1-coarse 24/3/12 +$314. Lesson: don't multi-round-chase Obj on sparse daily |
| **CME.NQ Hourly (futures)** | 4/8.5/9 NP=$407,505 Obj=3.29M (wide trail) | de-facto 2/7/10 (OOS −$12,735, all broke); **S4 keeps M2 TrailingStop ATRSTP=1.5 RoMaD 3.41→4.74; deploy MAIN+M2** |
| **TWF.TXF Hourly (futures)** | 10/3.5/13 NP=3,286,800 TWD Obj=23.4M (narrow trail) | **WINNER 10/3.5/13 OOS +1,090,800 break 1.10× (best futures cell); deploy MAIN ONLY** (RoMaD 8.59, kept ∅) |
| **CME.GC Hourly (futures)** | 30/4/31 NP=$136,380 Obj=1.04M (long-Len high-cooldown) | **WINNER 30/4/31 OOS +64,030 all-4-profitable break 1.95×; deploy MAIN ONLY** (RoMaD 5.72, kept ∅) |

**CloseChannelBreakout exit modules (crypto Hourly): TREND-pattern, 6/6 HELP** — M6 RescueTeam = max NP, M5 PT_Exit = best risk-adjusted (push BNB to Obj ~207K).
⚠️ **CCB pipeline Stage-3/4 results (futures NQ/TXF/GC Hourly + BNB/BTC Daily kept-sets, incl. the old "DD-slack" law) are SUSPECT** — they ran before the main-param pinning fix (0211a9), i.e. modules were optimized against DEFAULT main params. The 4 allinst pipelines' pinned re-runs overturned every "kept=∅" they produced; the 5 `run_*_ccb_*_pipeline.py` scripts still need fixed_inputs wiring + re-run before their module conclusions can be trusted. Futures 10M targets all unreachable (default sizing). The pipeline's Stage-4 teardown writes the final best params back into the Format Objects Input Strings.

### BollingerBreakout matrix (`SFJ_BollingerBreakout_crypto`; BBLen, BBmult, ATRMult, ReentryBars)

| | Hourly IS / OOS | Daily IS / OOS |
|---|---|---|
| BNB | $21,210 (13/1.55/7/13) / **✅ 2 PASS, BO3 9/1.05/7/17 +$9,687** | **$25,576 (2/1.0/5/12) / ✅ PASS BD2 full NP $30,718 = matrix max** |
| ETH | $5,267 (4/0.775/15.5/6) / ❌ all-broke | $2,291 (10/1.4/3/14) / ✅ PASS ED4 2/3.0/2.0/11 +$1,249 |
| BTC | $2,236 (8/1.5/15/22) / ❌ all-broke | ~$1,531-1,647 / ✅ PASS CD4 (sparse, low-confidence) |

**Matrix verdict: ALL 3 coins' DAILY pass OOS; Hourly only BNB → DAILY broadly more OOS-robust** (wide ATR trail, less hi-freq noise). BNB = only coin passing both TFs + only Daily>Hourly. Exit modules HELP (6/6); winner flips by TF: **M6 RescueTeam on Hourly (all coins), M5 PT_Exit on Daily** (M6 too sparse to fire).

### RegChannelBreakout 6-instrument Hourly matrix (`SFJ_RegChannelBreakout_{crypto,NQ}`; RegLen, BandMult, ATRMult, ReentryBars)

5th breakout reference = **linear-regression channel** (`MidReg=LinearRegValue; band=BandMult*StdDev`; the regression line LEADS an SMA). Crypto charts `_crypto`/`_Crypto1MUSD`, futures `_NQ`/default contracts. All 6 ran in ONE workspace via the 6-instrument orchestrator `run_rcb_allinst_pipeline.py` (`mc.activate_chart_by_symbol` MDI-activates+maximizes each chart). Crypto IS 2022-2026/FULL 2021/03-2026/06; futures IS 2019-2025/FULL 2018-2026.

| Inst | OOS-selected main (Reg/Band/Atr/Re) | Kept modules (Stage-4 greedy) | full NP main→stack | RoMaD |
|---|---|---|---|---|
| **BNB** | 14/1.19/7/12 (IS champ **Obj 110,693 #1 self-authored**; OOS +$4,005, 1.45×) | M5 PT=0.23, M2 ATRSTP=18.9, M1 STP=6.2 | $27.6K → **$41.8K** (MDD −7.3K→−6.7K) | 6.23 |
| **BTC** | 7/0.938/9/5 (**only strict PASS**, OOS +$1,322, MDD 1.00×) | M2=48.8, M3 EXITBAR=248, M1=11.7 | $3.9K → $4.2K | **10.22** |
| ETH | 40/1.0/16/5 (OOS +$221, 1.85×) | M5=0.667, M2=41, M1=13, M4 DAYRANGE=5.05 | $4.1K → $6.2K (MDD −32%) | 11.00 |
| TWF.TXF | 9/1.25/5/12 (OOS +117,400, 2.55×) | M2=5.2, M6=340/3.0, M5=0.089, M4=3.96, M3=33 (5!) | 3.92M → **7.45M** (+90%, MDD −40%) | 12.56 |
| CME.NQ | 13/1.0/8.5/8 (OOS −11,265 least-bad) | M6=140/3.4, M5=0.083 | $440K → **$910K** (+107%, MDD −45%) | **15.41** |
| CME.GC | 20/1.0/16/10 (OOS +93,480, 2.78×) | M6=80/3.9, M2=24.4, M4=5.58, M1=17.1 | $266K → $341K (MDD −18%) | 5.91 |

**Verdict: valid 5th breakout reference (BNB Obj 110,693 = #1 self-authored).** OOS-fragile like other breakouts — all break MDD EXCEPT **BTC (only strict PASS)**; 5/6 OOS-profitable (only NQ −). **Exit modules (re-run with main params correctly PINNED, 0211a9): 6/6 KEEP modules — NP +6..+107%, most MDD simultaneously shallower.** The old "6/6 HURT → MAIN only" was the pinning-bug artifact. Infra: `mc.set_signal_statuses` retries; per-instrument reconnect; teardown writes final main+module params into Input Strings.

### PivotBreakout 6-instrument Hourly matrix (`SFJ_PivotBreakout_{crypto,NQ}`; PivMult, ATRMult, ReentryBars)

6th breakout reference = **daily pivot point** (`PP=(HighD(1)+LowD(1)+CloseD(1))/3`; `UpLvl=PP+PivMult*priorday-range`, DnLvl mirror) — session-anchored levels (constant within the day), unlike all 5 prior rolling-intrabar references. Crypto `_crypto`, futures `_NQ`. All 6 ran via `run_pivot_allinst_pipeline.py` (same one-workspace orchestrator). Crypto IS 2022-2026/FULL 2021/03-2026/06; futures IS 2019-2025/FULL 2018-2026.

| Inst | OOS-selected main (Piv/Atr/Re) | Kept modules | full NP main→stack | RoMaD |
|---|---|---|---|---|
| **BNB** | 0.25/13.25/21 (OOS −1,335, 1.73×) | **M5 PT=0.213 ONLY (single step RoMaD +505% — largest ever)** | $13.3K → **$41.25K (+210%, MDD halved)** ≈ RCB-BNB | 7.63 |
| TXF | 0.325/3.5/15 (OOS +413,200, 1.31×) | M5=0.018, M4=4.9, M2=8.7, M1=6.1 | 3.12M → 4.02M | 8.26 |
| NQ | 1.0/6.25/31 (OOS +20,690, **1.005× near-miss**) | M6=480/3.1, M5=0.109, M3=39, M4=5.72 | $281K → $414K (+47%) | 8.45 |
| GC | 0.15/17.25/20 (OOS +85,010, 2.61×) | M6=520/3.0, M5=0.103, M4=4.94, M1=16.5 | $207K → $324K (+56%) | 4.39 |
| ETH | 0.2/16/30 (OOS −17 flat, 2.67×) | M6=40/4.1, M5=0.585, M4=5.58 | $3.5K → $6.0K (+73%, MDD −47%) | 9.14 |
| BTC | 0.25/10/27 (OOS +801, 1.16×) | M6=140/3.0 (IS ΔNP +154%!), M5=0.182 | $2.2K → $4.1K (+82%) | 10.58 |

**Verdict: WEAKEST reference MAIN-only, but the BIGGEST module beneficiary** — pinned re-run keeps modules 6/6 with **M5 PT_Exit kept 6/6** (session-anchored pivot entries + profit target = natural fit); BNB M5 alone lifts full NP +210% to $41.25K, tying RCB-BNB's stacked NP. IS edge still lowest (BNB $15,107 Obj 40K ≪ RegChannel/CC/Bollinger) and OOS none strict PASS (4/6 profitable), but post-stack RoMaD (BTC 10.58, ETH 9.14) is competitive. Infra: per-instrument `conn.connect()` reconnect (fixes multi-hour stale-handle) + best-effort Stage-4 teardown.

### MidChannelBreakout 6-instrument Hourly matrix (`SFJ_MidChannelBreakout_{crypto,NQ}`; Length, BandMult, ATRMult, ReentryBars)

7th breakout reference = **Donchian range-MIDPOINT ± ATR band** (`MidPt=(Highest(High,Length)+Lowest(Low,Length))/2; UpMid=MidPt+BandMult*ATR`) — center is the recent-range midpoint (mean-reversion-resistant, symmetric) vs the lagging SMA (Bollinger) / EMA (Keltner) centers. Ran via `run_midchannel_allinst_pipeline.py` (one-workspace orchestrator + all hardening: activate_chart_by_symbol, set_signal_statuses retry, per-instrument reconnect, best-effort teardown). Crypto IS 2022-2026/FULL 2021/03-2026/06; futures IS 2019-2025/FULL 2018-2026.

| Inst | IS champ (L/B/Atr/Re) NP Obj | OOS WINNER / deploy |
|---|---|---|
| Inst | OOS-selected main (L/B/Atr/Re) | Kept modules | full NP main→stack | RoMaD |
|---|---|---|---|---|
| **BNB** | 15/1.0/7/23 (**✅ strict PASS** OOS +$12,525, MDD 1.00×) | M5 PT=0.16, M2 ATRSTP=19.3 | $31.6K → **$39.8K** (MDD −23%) | 8.49 |
| TXF | 10/0.875/3.5/12 (OOS +1,009,000, 1.235×) | M6=100/3.4, M2=6.6, M4=4.25, M3=31 | 4.43M → 5.31M | 12.05 |
| NQ | 2/2.5/1.5/22 (OOS +90,830, 1.14×) | M5=0.028, M3=6, M1=1.5, M4=6.78 | $388K → $420K | **16.14 (highest of all 24 pinned cells)** |
| GC | 5/1.25/14/20 (OOS +33,100, 2.74×) | M5=0.07, M6=520/3.0, M2=24.9, M3=525, M1=17.0 (5!) | $201K → **$306K** (+52%, MDD −36%) | 5.57 |
| BTC | 4/0.475/17.75/8 (OOS +498, 1.37×) | M3=405, M1=29.3 | $2.9K → $3.1K | 7.38 |
| ETH | 3/0.1/15/7 (OOS −624, only loser) | M6=100/4.0, M5=0.6, M3=383, M1=18.0 | $4.4K → **$7.1K** (+62%, MDD −40%) | 9.85 |

**Verdict: solid mid-tier reference (BNB Obj 67,773 — above Pivot 40K, below RegChannel/CC/Bollinger).** Standout = **OOS robustness: BNB strict PASS (+$12,525, MDD held), 5/6 OOS-profitable, mild breaks** (BNB 1.00×, NQ 1.14×, TXF 1.235×, BTC 1.37×) — the range-midpoint center generalizes cleanly. **Pinned exit-module re-run: 6/6 KEEP; NQ post-stack RoMaD 16.14 = best cell across all four pinned references.** All runs zero-ABORT (full hardening).

### HullBreakout 6-instrument Hourly matrix (`SFJ_HullBreakout_{crypto,NQ}`; Length, BandMult, ATRMult, ReentryBars)

9th breakout reference = **Hull MA center ± ATR band** (`HullMA=WAverage(2*WAverage(C,n/2)−WAverage(C,n),√n)`) — a near-zero-lag WMA-based center that tracks price far faster than SMA/EMA. Ran via `run_hull_allinst_pipeline.py` (one-workspace orchestrator + all hardening). Crypto IS 2022-2026/FULL 2021/03-2026/06; futures IS 2019-2025/FULL 2018-2026. (First run under the fixed Regular-optimization selector — see Key Constraints.)

| Inst | IS champ (L/B/Atr/Re) NP Obj | OOS WINNER / deploy |
|---|---|---|
| Inst | OOS-selected main (L/B/Atr/Re) | Kept modules | full NP main→stack | RoMaD |
|---|---|---|---|---|
| **BNB** | 15/0.475/7/14 (OOS −2,341, 2.50×) | M5 PT=0.152, M2 ATRSTP=17.0 | $19.6K → $26.1K (MDD −16.0K stuck) | 1.77 (weakest cell) |
| TXF | 13/1.19/5.25/26 (OOS −406,800, 2.48×) | M6=380/3.5, M2=11.3, M1=7.7, M3=74 | 3.89M → **5.70M** (+47%, MDD −42%) | 7.75 |
| NQ | 15/3.0/2.0/20 (OOS **+29,110**, 1.43×) | M6=380/3.0, M5=0.079, M2=4.8, M4=6.78, M1=2.2 (5!) | $369K → $388K | 6.34 |
| GC | 60/2.5/14/20 (OOS −1,190, 4.11×) | M6=560/3.0, M4=3.2 (M4 step RoMaD +120%) | $193K → **$292.5K** (+51%, MDD −45%) | 5.21 |
| ETH | 31/0.35/11.5/19 (OOS +886, 1.73×) | M6=80/5.7, M3=148, M5=0.62, M1=13.6 | $4.8K → $6.8K (MDD −36%) | 10.97 |
| BTC | 7/1.0/9/5 (OOS +722, 1.49×) | M5=0.12, M2=51.0, M4=6.04 | $3.2K → $3.9K | 9.08 |

**Verdict: mid-upper tier IS (BNB Obj 86,746 ~ Bollinger/CC) but WEAKER OOS** — no strict PASS anywhere, only 3/6 OOS-profitable (BTC/ETH/NQ), larger breaks (GC 4.11×, BNB/TXF ~2.5×). The **low-lag Hull center wins IS but overfits OOS** — contrast MidChannel (laggier range-midpoint center, weaker IS but BNB strict PASS). **Pinned exit-module re-run: 6/6 KEEP** (old GC-keeps-M2 result also overturned — GC now keeps M6+M4, discards M2). BNB is Hull's weak spot: modules can't dent its −$16K MDD (RoMaD 1.77). All runs zero-ABORT.

### Self-authored ranking + FINAL LAWS

IS ranking (BNB Hourly, clean): Donchian v2 $24.4K (OOS PASS) ≈ ADXtrend $23.9K ≈ RegChannelBreakout $23.6K (**Obj 110,693 #1 risk-adj**; OOS BTC-only PASS) > VolatilityBreakout ≈ CloseChannelBreakout $22.0K (Obj 98K; OOS strong) > BollingerBreakout $21.2K (OOS PASS) > HeikinAshi $20.7K > BBSqueeze $16.1K > DonchianAsymV2 $15.0K > Keltner $10.1K > RSIPullback ~$8K > DonchianAsym $5.7K > ROCmomentum $5.6K > ParabolicSAR (broken); Turtle/Fractal all-negative; ChannelClose degenerate. **Eight breakout references tested** (all = non-lagged level + intrabar STOP + wide ATR trail + cooldown), by BNB-Hourly Obj: RegChannel 110K (NP $23.6K) > Donchian v2 100K ≈ CloseChannel 98K ($22K) > Bollinger 95K ($21K) > **Hull 87K ($22K, low-lag Hull center, OOS-weak)** > **MidChannel 68K ($18K, range-midpoint center, BNB strict OOS PASS)** > **Pivot 40K ($15K, daily-pivot, WEAKEST)**. (`SFJ_VWAPBreakout` = 9th ref built, not yet tested.) Reference-center lesson: **IS strength anti-correlates with OOS robustness across centers — the low-lag Hull wins IS but overfits OOS; the laggier range-midpoint (MidChannel) is IS-weaker but OOS-robust (BNB strict PASS).**

**Exit-module verdict (pinned re-runs, 4 refs × 6 inst = 24 cells): 24/24 KEEP ≥1 module.** NP +6..+210%, most cells' MDD simultaneously shallower. Kept frequency: **M5 PT_Exit 18/24** (6/6 on Pivot — session-anchored entry + profit target = natural fit) > M6 RescueTeam 14/24 > M2 Trailing 13/24 = M1 ATRstop 13/24 > M4 high_vol 12/24 > M3 EntryBars 10/24. Post-stack RoMaD leaders: MidChannel-NQ 16.14 > RCB-NQ 15.41 > RCB-TXF 12.56 ≈ Mid-TXF 12.05. The pre-fix "6/6 HURT on RegChannel/Hull/MidChannel/Pivot → MAIN only" and the "DD-slack" conditional law were BOTH artifacts of the silent main-param pinning failure (modules were optimized against DEFAULT main params); only CC/Bollinger/SuperTrend/Donchian "modules HELP" survived because those tests had correct mains.

**LAWS:** (1) Two self-authored are both IS-strong AND OOS-robust: **Donchian v2 + BollingerBreakout** (rolling breakout + wide ATR trail + cooldown). (2) Breakout REFERENCE: rolling extreme (Donchian) and statistical band (Bollinger) both work (Bollinger lowest MDD); a LAGGED pivot (Fractal) and CLOSE-confirmed MARKET entry (ChannelClose) both break it. (3) The EXIT is decisive — wide ATR chandelier trail ≫ channel exit / SAR. (4) Entry FILTERS fail OOS (ADX/squeeze/RSI/trend/vol-buffer → floor); the ReentryBars COOLDOWN is the one entry-side lever kept >0. (5) Asymmetry is necessary-not-sufficient (needs the cooldown). (6) **IS strength ≠ OOS robustness** — the IS NP/Obj champion (lowest IS-MDD) is repeatedly OOS-worst; IS-inferior higher-freq / tighter-band / wider-MDD regimes generalize best (thin-edge coins). Exception: thick-edge BNB Daily — the sparse wide-trail champion itself holds OOS. (7) **Exit modules on trend-breakout mains genuinely HELP (24/24) — but ONLY when the module optimization pins the main's champion params** (wizard-grid Current Value via `fixed_inputs`, read-back verified); any module result produced without verified pinning is invalid. (8) A weak-main reference can be rescued by its module stack (Pivot-BNB +210%), but a weak-OOS main stays weak-OOS — the stack lifts NP/RoMaD, not generalization.

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

`run_<sym>_ccb_*_pipeline.py` run all 4 stages (IS optimization → OOS champion-select → exit-module IS optimization → cumulative greedy full-period RoMaD stack) in one MC64 connection, handing params stage→stage in memory and writing `results/<dir>/state.json` after every stage (`--from-stage N` resumes; `--from-csv` re-analyzes). Programmatic Format-Objects input-setting is read-back-verified and ABORTs loudly on mismatch. `run_nq_ccb_hourly_pipeline.py` is the canonical futures pattern (self-judged Stage-1 convergence; Stage-4 teardown writes final best params back into Input Strings). Clone + change `SYMBOL`/`OUTPUT_DIR` for a new instrument.

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

## Detailed Reference

- `optimizer/OPTIMIZATION_SKILLS.md` — objective function details, parameter range tables, per-instrument round history (Traditional Chinese)
- Auto-memory files (`~/.claude/projects/.../memory/project_*.md`) — full per-search findings, convergence tables, regime analysis
- `results/<search>/final_params_*.json` + `results/<pipeline>/state.json` — champion + all-attempt/all-stage data
