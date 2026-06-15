# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a **parameter plateau optimizer** for MultiCharts64 (MC64) trading strategies. It automates MC64's built-in optimizer via Windows UI automation, exports results as CSV, and applies a **plateau detection algorithm** to find parameter combinations that are stable (robust to small parameter perturbations) rather than just peak performers. Results include IS/OOS validation and heatmap visualizations.

Workflow: the user opens the right MC64 workspace, requests a search round (target NP, ≤5000 combos per attempt, Objective = NP²/|MDD|), a `search_*.py` script runs 11-12 attempts, and results are analyzed for ceiling confirmation via multi-attempt convergence. Detailed per-search history lives in the auto-memory files and `results/*/final_params_*.json` — this file keeps only the summary tables.

## Search Status Summary

### `_2021Basic_Break_NQ` (breakout; params LE, SE, STP, LMT)

| Instrument | TF | Target | Status |
|---|---|---|---|
| TWF.TXF | Hourly | >6M TWD | **✅ MET** (LE=3 SE=76 STP=4 LMT=32 NP=6,043,200) |
| TWF.TXF | Daily | >6M TWD | Not met — best 4,089,800 (−32%) |
| CME.NQ | Hourly | >700K | Not met — best $656,575 (−6.2%; LE=1 SE=9 STP=2 LMT=14) |
| CME.NQ | Daily | >700K | Not met — best $350,220 (−50%; LE=1 SE=78 STP=5.5 LMT=3.9) |
| CME.GC | Hourly | >700K | Not met — best $292,030 (−58%; LE=2 SE=35 STP=3.2 LMT=20) |
| CME.GC | Daily | >700K | Not met — best $312,520 (−55%; LE=4 SE=49 STP=0.9 LMT=7) |
| CME.CL | Hourly | >700K | Ceiling $103,190 (−85%; LE=1 SE=54 STP=5.3 LMT=22) |
| CME.CL | Daily | >700K | Ceiling $15,510 (−97.8%) — **does not work** |
| CBOT.ZW | Hourly | >700K | All negative — does not work |
| CBOT.ZW | Daily | >700K | Ceiling $26K (−96%; LE=25 SE=2 STP=4 LMT=4) — **does not work** |

### `_2021Basic_Break_CL` (ATR breakout w/ LenLE; params LE, SE, STP, LMT, LenLE)

| Instrument | TF | Target | Status |
|---|---|---|---|
| CME.CL | Daily | >700K | Ceiling $91K (−87%; LE=1 SE=1 STP=4 LMT=5) — **does not work** |
| CBOT.ZW | Daily | >700K | Ceiling $55K (−92%; LE=23 SE=3 STP=8 LMT=2) — **does not work** |

### `SFJ_15Dworkshop_lesson5_countertrend_LS` (BB counter-trend, reversal exits only)

Params: LENGTH_LONG (LL), STDDEV_LONG (SL), LENGTH_SHORT (LS), STDDEV_SHORT (SS). Workspace: `20260521SFJ_Bollinger_AI.wsp`.

| Instrument | TF | Target | Status |
|---|---|---|---|
| CME.NQ | Hourly | >700K | **✅ MET** R3 (LL=17 SL=0.2 LS=45 SS=1.4 NP=$751,230 MDD=−$64,855 1614tr) |
| CME.NQ | Daily | >700K | Ceiling $460,770 (−34.2%; LL=8 SL=0.7 LS=47 SS=1.86; 40tr/7yr) |
| CME.GC | Hourly | >800K | Ceiling $437,930 (−45.3%; LL=14 SL=0.1 LS=59 SS=0.45); best Obj: LS=56 SS=0.38 Obj=6.63M MDD=−$28,310 |
| CME.GC | Daily | >800K | Ceiling $467,320 (−41.6%; LL=3 SL=1.28 LS=50 SS=2.3; 64tr/7yr) |
| TWF.TXF | Hourly | >9M TWD | **8M MET** (LL=22 SL=0.425 LS=43 SS=1.771 NP=8,101,400); 9M unreachable; low-MDD alt: LS=36 SS=1.43 NP=7.65M MDD=−589K Obj=99M |
| TWF.TXF | 240min | >8M TWD | Ceiling 6,958,400 (−13.0%; LL=10 SL=0.575 LS=53 SS=0.125) |
| TWF.TXF | Daily | >8M TWD | Ceiling 4,019,800 (−49.8%; LL=25 SL=0.165 LS=50 SS=0.275) |

### `SFJ_15Dworkshop_lesson5_countertrend_LS_crypto` (same BB counter-trend + `_Crypto1MUSD`)

Contract: `_Crypto1MUSD = Round(1,000,000/C, 0)` (~$1M notional/trade). Workspace: `20260101_SFJ_Bollinger_AI.wsp`.

| Instrument | TF | Target | Status |
|---|---|---|---|
| BNBUSDT | Hourly (exit modules) | improve NP | **ALL 6 exit modules HURT** (best M3 EntryBarsAfterExit EXITBAR=7: $8,636 −75.4%; worst M4 high_volatility: $17 −99.95%). Adding any exit truncates the rare-extreme holds and triggers re-entries (24tr→114-180tr). **Keep the pure reversal-exit champion.** A00 baseline byte-identical ($35,112.20, 0.000% drift); all CSVs full rows. See `results/bnb_ct_exit_modules_search/` |
| BTCUSDT | Hourly (exit modules) | improve NP | **ALL 6 exit modules HURT — strictly worse on BOTH NP and MDD** (best M6 RescueTeamExit Length=80 std=3.9: $891 −78.2%; worst M5 PT_Exit: $242 −94.1%; all MDD −$690 to −$946 vs baseline −$431). Optimizer pushed every exit param toward "never trigger" (STP=20.9, ATRSTP=28.4, EXITBAR=338, PT_Base=0.4). 23tr→173-210tr. A00 byte-identical ($4,092.62, 0.000% drift). See `results/btc_ct_exit_modules_search/` |
| BTCUSDT | Hourly (full-period validation) | improve NP+MDD | **CONFIRMED on full data: fixing each module at its IS param and re-testing — ALL 6 still HURT.** Modules −43.6% to −75.9% NP and ALL deepen DD, trade count explodes ~10×. No module improves NP or reduces Max DD. ⚠️ OOS-only could NOT be isolated: MC64 ignores signal Begin-date (only End-date restricts), and the workspace data ends ~2026/01/01 so there is no post-2026 OOS data. `search_btc_ct_oos_validation.py`, `results/btc_ct_oos_validation_search/` |
| BNBUSDT | Hourly (champion TRUE OOS) | pick most OOS-profit, no MDD break | **First clean OOS isolation (chart correctly trimmed to 2026/01 for IS pass): IS≠FULL.** OOS window 2026/01→2026/06 was a LOSS for every BNB CT champion: B1/B2 (LL=122 SL=4.025 LS=29 SS=4.2) OOS −$10,046 (full NP $25,066, MDD −$7,112→−$14,490); B3/B4 (LL=229 long) OOS −$10,487 (MDD −$14,441→−$18,202). **No champion profitable OOS; all broke MDD → none passes.** Least-bad = B1 (smallest OOS loss + lowest full MDD; the 8-conv live champion). KEY: chart-trim two-pass DOES isolate OOS when the chart is actually trimmed — the earlier BTC/ETH "can't isolate" was just the IS pass not being trimmed. `search_bnb_ct_oos_champion_select.py` |
| ETHUSDT | Hourly (champion TRUE OOS) | most OOS profit, no MDD break | **WINNER E2: LL=111 SL=4.025 LS=115 SS=4.725 — confirmed with clean OOS isolation (auto-set Data Range; IS $5,005 ≠ FULL $5,105, IS matches prior rounds 0.0% drift).** OOS 2026/01→06: **NP-max派 (E1/E2) PROFITABLE +$100.5**; Obj-max派 (E3/E4) LOST −$135. **No ETH champion holds Max DD** (all hit the same OOS −$1,124 drawdown > each established MDD) so none strictly PASSES — but E2 = best compromise: tied-best OOS (+$100.5), highest full NP ($5,105), DD-break mildest (1.50× vs E3/E4 1.94×). **Clean OOS FLIPS the earlier un-isolated E3/E4 pick**: Obj-max's low IS-MDD (−$580) was overfit to the calm period — it LOSES OOS and breaks DD harder (same trap as BTC C9). METHOD breakthrough: `DTM_SETSYSTEMTIME` sets the date picker but doesn't fire `DTN_DATETIMECHANGE` → OK won't apply (3 prior failures, IS==FULL); fix = nudge the picker (click+Right+Up/Down, net-zero) to fire the notification. Supersedes the old "OOS not isolable". `search_eth_ct_oos_champion_select.py`, `Run_ETH_CT_OOS_Select_MERGED.bat` |
| BTCUSDT | Hourly (champion TRUE OOS) | most OOS profit, no MDD break | **WINNER C1: LL=107 SL=4.1 LS=139 SS=4.7 — confirmed with clean OOS isolation (chart trimmed to 2026/01 for IS; IS $4,104 ≠ FULL $3,726).** OOS 2026/01→06 was a small loss for all (−$378 to −$806) but **C1 is the ONLY champion that did NOT break its Max DD (−$799 held) AND lost least OOS (−$378) → only PASS.** The in-sample low-MDD Obj-max C9 (LL=104 SL=4.15 LS=68 SS=4.95) BROKE its DD (−$431→−$725) and lost more OOS (−$603). BTC is the only crypto with a passing champion (BNB/ETH: all broke). Earlier full-period robustness pick now validated by true OOS. `search_btc_ct_oos_champion_select.py` |
| BTCUSDT | Daily (full-period validation) | improve NP+MDD | **CONFIRMED: NO module improves NP OR reduces MDD — cleanest negative of the 4.** Baseline main-only NP=$3,420 DD=−$478 16tr; every module NP −63% to −75.5% (M5 best $1,252; M1 $882; M2/M3/M4 $838; M6 $1,027) AND every module deepens DD (−$921 to −$1,298, 2-2.7×). ⚠️ OOS NOT measurable: IS==FULL byte-identical → BTC Daily chart has no data past 2026/01/01. (M2/M3/M4 gave identical $838/−$1298/73tr — sparse-daily bracket/convergence artifact; immaterial.) `search_btc_ct_daily_oos_validation.py` |
| BNBUSDT | Hourly (full-period validation) | improve NP+MDD | **CONFIRMED: NO module improves NP.** Baseline main-only NP=$25,066 DD=−$14,490 27tr (this data vintage; DD 2× the prior −$7,112). Only M3 EXITBAR=7 stays positive ($3,890, −84.5%); M1/M2/M4/M5/M6 all go NEGATIVE (−$4.9K to −$14.9K). Only M1/M3 cut DD slightly (M3 −$11,851 = −18%) but at −84.5% NP → fails joint test. ⚠️ OOS NOT measurable: IS(end 2026)==FULL(end 2027) byte-identical → BNB chart has no data past 2026/01/01 (workspace 20260101). Run interrupted after FULL M4 but IS==FULL so IS column = full result. `search_bnb_ct_oos_validation.py` |
| ETHUSDT | Hourly (full-period validation) | improve NP+MDD | **CONFIRMED: NO module improves NP — all destroy it.** Baseline main-only NP=$5,181 DD=−$2,380 22tr (this data vintage; DD 4× the prior $580). Modules: M1 −$129, M2 −$65, M3 −$3, M4 −$536, M5 −$552, M6 +$70 → all wipe out 98-110% of NP. M1-M5 DO cut Max DD (M1 −$1,501 = −37%) but only by churning the strategy to ~breakeven — fails the joint "NP↑ AND DD↓" test. ⚠️ **OOS NOT measurable: IS(end 2026)==FULL(end 2027) byte-identical → ETH chart has no data past 2026/01/01** (workspace named 20260101). Real OOS test needs post-2026 ETH data loaded. `search_eth_ct_oos_validation.py` |
| ETHUSDT | Hourly (exit modules) | improve NP | **ALL 6 exit modules HURT — most extreme of the trilogy: M4 (−$118) and M5 (−$39) are NET NEGATIVE even at best params** (best M3 EXITBAR=76: $459 −90.5%; 20tr→394-465tr, biggest explosion). A00 byte-identical ($4,847.50, 0.000% drift). **Trilogy verdict (BNB+BTC+ETH, 18/18 tests hurt): crypto CT champions must keep pure reversal exits.** Speed mode validated: 12.3 min vs BTC's 19 min (−35%; skip reopen-verify + one-pass status read). See `results/eth_ct_exit_modules_search/` |
| BTCUSDT | Daily (exit modules) | improve NP | **ALL 6 exit modules HURT (4th TF verified)** — all −52% to −66% (best M5 PT_Base=0.672 $1,452 −52.6%; M6 RescueTeamExit Length=20 std=3 $1,162 −62.1% [partial coverage]; M1/M3/M4 ~−65%), MDD all worse (−$810 to −$1,298 vs −$399); 13tr→46-75tr. A00 data-drifted $3,065/13tr (Rule 5). **M6 NOTE: MC64 only exported Length 20-500 of 20-600 (sparse-trade truncation, not a setup bug — log confirms End=600 was entered); validator now distinguishes garbled (discard) vs truncated-but-clean (accept best-NP + flag partial).** **Cross-TF verdict (24/24 tests hurt, BNB/BTC/ETH Hourly + BTC Daily): keep pure reversal exits.** Speed 11.7 min. See `results/btc_ct_daily_exit_modules_search/` |
| BNBUSDT | Hourly | >100K | **NP Ceiling $36,703 (−63.3%) — R4 confirmed** (7-conv R3 A09/A10/A11 + R4 A01/A09/A10/A11; R3→R4 +0.00%; NP plateau LL 222-236 SL 3.8-3.85 LS=22 SS 4.15-4.225 10tr MDD=−$14,441 Obj=93,284; found by LL 170-300 gap sweep). ⭐⭐⭐ **Obj-max (live use): LL=122 SL=4.025 LS=29 SS=4.2 NP=$35,112 MDD=−$7,112 24tr Obj=173,342** (8-conv). LL 2-500/LS 2-40/SS 3.4-6.0/SL 3.0-4.3 all bounded. BNB/BTC 8.83×. BNB Hourly cross-strategy: CT #2 NP (beats Breakout $35,634), #3 Obj |
| ETHUSDT | Hourly | >10K | **Ceiling $5,005 (−50.0%) — R3 confirmed** (7-conv R2 A09/A10/A11 + R3 A01/A09/A10/A11; LL=111 SL=4.025 LS=115 SS=4.725 25tr MDD=−$748 Obj=33,481; R2→R3 +0.00%; all gaps fine-swept). ⭐⭐ Obj-max: plateau LL 109-110 SL 4.3-4.4 LS 109-110 SS 4.7-4.75 NP=$4,848 MDD=−$580 Obj=40,545 (R2 A07 = R3 A02 same trade set). ETH/BTC 1.20×. On ETH, CT is #2 NP / #3 Obj (QPATRex still holds both crowns) — unlike BTC where CT swept all |
| BTCUSDT | Hourly | >10K | **Ceiling $4,155 (−58.5%) — R5 confirmed** (7-conv R4 A09/A10/A11 + R5 A01/A09/A10/A11; LL=104 SL=4.05 LS=165 SS=4.95 22tr MDD=−$498 Obj=34,690; R4→R5 +0.00%; all boundaries verified worse). ⭐⭐⭐ Obj-max: LL=104 SL=4.15 LS=68 SS=4.95 NP=$4,093 MDD=−$431 Obj=38,853 (R4 A01 = R5 A02 exact) — **MDD/NP 10.5% best of all BTC searches**. Note: R3's "11-conv ceiling" at LS=139 was a false ceiling — R4's LS 145-220 bridge found LS=165. Rare-extreme regime (22tr, opposite of futures 1614-1984tr) |
| BTCUSDT | Daily | >10K | **Ceiling ~$3,593 (−64.1%) — R3 confirmed** (LL=49 SL=2.4 LS=123 SS=1.2; R1/R2 7-conv 16tr MDD=−$399 Obj=32,366). R3 closed the LS>160 gap: LS 130-220/220-340/160-260 all worse (no Hourly-style hidden regime — LS=123 is the true peak). NEW mid-LL+long-LS+tight-SS regime (unlike Hourly LL=104 SL=4.05). R1 export-fails redone all worse. **Data drift Rule 5**: R3 same champion re-tested $3,065/13tr vs R1/R2 $3,593/16tr (Binance bar refresh; params unchanged, NP data-dependent). Daily < Hourly. $10K structurally unreachable |

### `SFJ_HUNTER2_NQ` (MA filter + ATR-stop entry, reversal exits, max 1 entry/day)

Params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S. Workspace: `20260101_SFJ_HUNTER_AI.wsp`.

| Instrument | TF | Target | Status |
|---|---|---|---|
| TWF.TXF | Hourly | >9M TWD | **✅ MET** (LEN_L=15 LEN_S=290 ATR_L=0.97 ATR_S=1.5 NP=9,121,800); 10M ceiling 9,129,400 (−8.7%) |
| TWF.TXF | Daily | >9M TWD | Ceiling 5,393,400 (−40.1%; LEN_L=15 LEN_S=65 ATR_L=0.17 ATR_S=1.1) |
| CME.NQ | Hourly | >900K | Ceiling $634,865 (−29.5%; LEN_L=8 LEN_S=89 ATR_L=0.25 ATR_S=2.5; 883tr/7yr) |
| CME.NQ | Daily | >900K | Ceiling $433,700 (−51.8%; LEN_L=6 LEN_S=85 ATR_L=0.08 ATR_S=1.15; 72tr/7yr) |
| CME.GC | Hourly | >700K | Ceiling $384,820 (−45.0%; LEN_L=5 LEN_S=37 ATR_L=0.8 ATR_S=5.9; 288tr/7yr) |
| CME.GC | Daily | >700K | Ceiling $338,990 (−51.6%; LEN_L=7 LEN_S=5 ATR_L=0.296 ATR_S=2.068; inverted MA; 51tr/7yr) |

### `SFJ_HUNTER_NQ` (long-only MA + ATR stop entry, fixed STP/LMT, max 1 entry/day)

Params: LEN, STP, LMT. Workspace: `20260101_SFJ_HUNTER_AI.wsp`.

| Instrument | TF | Target | Status |
|---|---|---|---|
| TWF.TXF | Hourly | >7M TWD | Ceiling 5,001,800 (−28.5%; LEN=6 STP=22,837 LMT=606,660) |
| CME.NQ | Hourly | >800K | Ceiling $500,015 (−37.5%; LEN=7 STP=550 LMT=13,500) |
| CME.GC | Hourly | >800K | Ceiling $379,400 (−52.6%; LEN=144 STP=8,060 LMT=6,500; 142tr/7yr) |
| CBOT.ZW | Hourly | >800K | Ceiling $35,878 (−95.5%) — **does not work** |

### `_2021Basic_Osc_NQ` (BB oscillator, ATR(10)-based STP/LMT)

Params: LEN, LE (can be negative), SE, STP, LMT. Workspace: `20260523SFJ_BASIC_OSC_AI.wsp`.

| Instrument | TF | Target | Status |
|---|---|---|---|
| CME.NQ | Hourly | >800K | Ceiling $453,610 (−43.3%; LEN=3 LE=−1.25 SE=1.5 STP=1.0 LMT=18.5) |
| CME.NQ | Daily | >800K | Ceiling $456,190 (−43.0%; LEN=12 LE=0.1 SE=2.5 STP=0.5 LMT=25.5) |
| CME.GC | Hourly | >800K | Ceiling $312,250 (−61.0%; LEN=2 LE=−0.75 SE=1 STP=1.6 LMT=30) |
| CME.GC | Daily | >400K | Ceiling $365,690 (−8.6%; LEN=8 LE=−0.4 SE=2.2 STP=1.8 LMT=7) |
| TWF.TXF | Hourly | >7M TWD | Ceiling 5,970,000 (−14.7%; LEN=11 LE=−1.30 SE=3.0 STP=0.975 LMT=33) |
| TWF.TXF | Daily | >7M TWD | Ceiling 5,065,400 (−27.6%; LEN=11 LE=1.2 SE=3.25 STP=1.25 LMT=17) |

### `SFJ_XtremeStop_NQ` (% breakout vs close X bars ago, reversal exits only)

Params: X, LY, SY. Workspace: `SFJ_XtremeStop_AI.wsp`.

| Instrument | TF | Target | Status |
|---|---|---|---|
| CME.NQ | Hourly | >800K | Ceiling $624,015 (−22.0%; X=10 LY=1.48 SY=2.18; 350tr/7yr) |
| CME.NQ | Daily | >800K | Ceiling $491,590 (−38.6%; X=9 LY=0.025 SY=3.7; 75tr/7yr) |
| CME.GC | Hourly | >800K | Ceiling $428,050 (−46.5%; X=7 LY=0.67 SY=1.80; 206tr/7yr) |
| CME.GC | Daily | >800K | Ceiling $288,450 (−64.0%; X=1 LY=1.1 SY=2.8; 37tr/7yr) |
| TWF.TXF | Hourly | >8M TWD | Ceiling 5,411,000 (−32.4%; X=63 LY=5.155 SY=5.79; 11tr/7yr) |
| TWF.TXF | Daily | >8M TWD | Ceiling 4,820,400 (−39.8%; X=1 LY=3.1 SY=4.445; 20tr/7yr) |

### `QuantPassATRex` (ATR+StdDev breakout, stop+market entries, reversal exits, `_Crypto1MUSD`)

Params: Len, Su_Multiple, Ni_Multiple. Workspace: `20260101_QuantPassATRex_AI.wsp`.

| Instrument | TF | Target | Status |
|---|---|---|---|
| BTCUSDT | Hourly | >10K | Ceiling $3,293 (−67.1%; Len=125 Su=1.525 Ni=6.15). Risk-adj: Len=13 Su=2.52 Ni=3.0 Obj=26,020 |
| BTCUSDT | Daily | >10K | Ceiling $3,511 (−64.9%; 8-conv; Len=24 Su=0.75 Ni=3.5 13tr). Obj-max: Len=30 Su=0.74 Ni=1.875 Obj=13,464 |
| ETHUSDT | Hourly | >10K | Ceiling $5,198 (−48.0%; Len=24 Su=1.48 Ni=2.47). ⭐⭐ Risk-adj: Ni=5.4 NP=$4,809 MDD=−$493 Obj=46,902 |
| ETHUSDT | Daily | >10K | Ceiling $3,161 (−68.4%; Len=14 Su=0.83 Ni=1.13 41tr) — worst Daily/Hourly drop (−39%) |
| BNBUSDT | Hourly | >100K | Ceiling $39,921 (−60.1%; Len=235 Su=0.58575 Ni=2.1275 95tr). ⭐⭐⭐ Obj-max: Len=94 Su=0.715 Ni=1.68 NP=$31,776 MDD=−$3,905 Obj=258,561 |
| BNBUSDT | Daily | >100K | Ceiling $30,876 (−69.1%; Len=93 Su=0.37 Ni=0.53 83tr Obj=163,939; MDD/NP 18.8% best in QPATRex) |

### `QuantPassATR_Breakout` (2-param ATR breakout, market entries, reversal exits, `_Crypto1MUSD`)

Params: Len, Multiple. Workspace: `20260101_QuantPassATR_Breakout_AI.wsp`.

| Instrument | TF | Target | Status |
|---|---|---|---|
| BTCUSDT | Hourly | >10K | Ceiling $2,748 (−72.5%; Len=212 Multiple=3.27 90tr; MDD/NP 25% vs QPATRex 42%) |
| BTCUSDT | Daily | >10K | Ceiling $2,544 (−74.6%; Len=13 Mult=1.845 13tr). Obj-max: Len=10 Mult=1.05 NP=$1,830 MDD=−$418 54tr. R1 "$3,227" was data-drift artifact. Daily < Hourly (opposite of QPATRex) |
| ETHUSDT | Hourly | >10K | Ceiling $4,444 (−55.6%; 8-conv; Len=32 Multiple=3.37 MDD=−$468 58tr Obj=42,191; **MDD/NP 10.5% best of all 4 BTC/ETH Hourly searches**) |
| ETHUSDT | Daily | >10K | Ceiling $3,986 (−60.1%; 9-conv; Len=9 Mult=1.705 14tr Obj=20,721; NP-max = Obj-max). ETH Breakout Daily > QPATRex Daily +26.1% (rare reversal) |
| BNBUSDT | Hourly | >100K | Ceiling $35,634 (−64.4%; 13-conv strongest ever; Len=3 Multiple=2.965 82tr). ⭐⭐⭐ Obj-max: Len=145 Multiple=2.91 NP=$32,506 MDD=−$4,610 Obj=229,204. BNB/BTC 12.97× |
| BNBUSDT | Daily | >100K | Ceiling $20,317 (−79.7%; 8-conv; Len=18 Mult=0.735 114tr HIGH-FREQ regime Obj=81,486; NP-max = Obj-max). Daily/Hourly −43% worst |

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
- In the optimization wizard, rows are matched by param name + `cfg.mc_signal_name`, so a module run uses `StrategyConfig(mc_signal_name=<module>, params=<module params>)`; the main signal's inputs stay unchecked at their Format-Signals values. The exported CSV contains ALL chart input columns — validate per row that the main signal's params stayed at the champion values.
- `search_bnb_ct_exit_modules.py` is the canonical pattern (A00 same-day baseline re-measure + one attempt per module + teardown; `--module N`, `--manual-status`, `--smoke` flags). Probe UIA structure first with `probe_signal_status.py --dump / --toggle <name> / --wizard-dump`.

### Column name mapping

MC64 exports different column headers across versions. `MC_COLUMN_MAP` in `config.py` normalizes them to `NetProfit`, `MaxDrawdown`, `TotalTrades`. If a new MC version uses different headers, add entries there.

## Key Constraints

- **Must run as Administrator** — MC64 runs elevated; UI automation fails otherwise. All `search_*.py` scripts auto-elevate via UAC using `ctypes.windll.shell32.ShellExecuteW(None, "runas", ...)` with the `--_elevated` flag. Never skip this.
- **MC64 workspace must be open** before running — the automation finds the chart window by matching `chart_symbol` against open window titles.
- **pywinauto limitation** — never use process-scoped `Application(process=pid)` to reach MC64 dialogs; use `Desktop(backend="uia")` + ctypes window enumeration instead. This is the core UIPI workaround.
- **`ParamAxis.name` must match MC exactly** — case-sensitive; the automation types these names into the Inputs dialog.
- **BAT files must be pure ASCII** — Traditional Chinese Windows uses CP950 (Big5). Non-ASCII characters such as the em dash `—` (U+2014) are invalid Big5 sequences and cause CMD misparsing (`echo` → `ec`+`ho`, `cd /d` → `cd`+`/d`), leaving the working directory wrong. Verify with: `[System.IO.File]::ReadAllBytes($path) | Where-Object { $_ -gt 127 }`.
- **MC64 connection in search scripts**: always `conn = mc.MultiChartsConnection(); conn.connect()`. Never `mc.connect_to_mc()` — that function does not exist.
- **MC64 export truncation** — sparse-trade grids (e.g. QPATR_Breakout Mult>2 with Len>20) export only partial rows; failed 3× across BTC/ETH/BNB Daily. Structural MC64 limitation; design grids to avoid these regions or accept the gap.
- **TRUE OOS isolation via chart Data Range** — MC64 ignores the signal Begin-date; only the CHART's loaded data range restricts the backtest. `mc.set_instrument_data_range(conn, from, to)` sets Format Instruments → Settings → Data Range (From-To radio + both date pickers) to trim/expand the chart so an IS pass (end 2026/01) ≠ FULL pass (end 2026/06), giving OOS = NP_full − NP_is. **Critical**: `DTM_SETSYSTEMTIME` sets the picker value but does NOT fire `DTN_DATETIMECHANGE`, so OK won't apply it (3 silent failures, IS==FULL) — after the DTM set the code nudges the picker (click + Right + Up/Down, net-zero) to fire the notification. Verify first with `--probe-instrument` (sets FULL range, reopens dialog, reads pickers back) + a chart screenshot. `mc.read_instrument_data_range` reads the current range. This supersedes the earlier "OOS-only not isolable" note.

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

A ceiling is confirmed when: round-over-round gain ≤ ~0.1%, AND multiple attempts (including adaptive zooms) converge on byte-identical NP/MDD/trades, AND new-territory sweeps (global, boundary-push, alternative regimes) all fall short. Report both NP-max and Obj-max champions — they are often different regimes.

## Detailed Reference

- `optimizer/OPTIMIZATION_SKILLS.md` — objective function details, parameter range tables, per-instrument round history (Traditional Chinese)
- Auto-memory files (`~/.claude/projects/.../memory/project_*_search.md`) — full per-search findings, convergence tables, regime analysis
- `results/<search>/final_params_*.json` — champion + all-attempt data for every round
