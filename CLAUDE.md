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
| ETHUSDT | Daily | >10K | **Ceiling $3,755 (−62%) — R2 A10=A11** (LL=5 SL=1.75 LS=37 SS=1.375 MDD=−$863 47tr Obj=16,336; NP-max=Obj-max). **HIGH-FREQ ultra-short-LL regime** (opposite of BTC Daily low-freq LL=49 LS=123). R1 LL=13 was NOT converged; R2 LL-sweep found LL=5 (+13.2%). ETH-hourly-analog/high-SS/asym regimes barren (0 trades). IS window 2022/01-2026/01. `search_eth_ct_daily{,2}.py` |
| ETHUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **WINNER ED4: LL=45 SL=2.25 LS=100 SS=1.0** (low-freq BTC-style 18tr) — ONLY strict PASS (full MDD −$1,002 = IS −$1,002, held) though OOS −$500 (ALL 4 lost OOS). Hi-freq champion ED1 (LL=5 LS=37) lost LEAST OOS (−$43) but BROKE MDD (−863→−1144) → fails. OOS criterion picks the low-freq longer-LS MDD-holder (same as BTC). Clean OOS (IS 2022/01-2026/01 ≠ FULL 2021/03-2026/06). `search_eth_ct_daily_oos_champion_select.py` |
| BNBUSDT | Daily | >10K / >100K | **Ceiling $42,546 (−57.5% of 100K) — R3=R4 +0.01%** (LL=18 SL=2.175 LS=21 SS=3.15 MDD=−$6,645 14tr Obj=272,408; NP-max=Obj-max; plateau LL 15-19 SL 2.1-2.25 LS 18-21 SS 3.0-3.15). ⚠️ **R1=R2 $35,182 (LL=24 SL=1.75) was a FALSE ceiling — R2 fixed SL too narrow (1.5-2.0); R3 widened SL→found real peak at SL~2.175 (+20.9%), R4 confirmed** (lesson: widen every axis, esp. STDDEV, before declaring a ceiling). >10K MET (425%); 100K unreachable. **BNB Daily $42.5K > BNB Hourly $36.7K (rare Daily>Hourly)**; BNB = only crypto whose Daily CT works (BTC/ETH ~$3.5-3.8K). `search_bnb_ct_daily{,2,3,4}.py` |
| BTCUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **WINNER D2: LL=40 SL=2.2 LS=130 SS=1.0** (mid-freq 20tr) — ONLY strict PASS (full MDD −$442 = IS −$442, held) AND highest OOS profit **+$606** (all 4 OOS-profitable). Ceiling champion D1 (LL=49 LS=123, highest NP) BROKE MDD (−399→−466) → fails. OOS FLIPS pick to lower-freq longer-LS regime. Clean OOS (IS 2022/01-2026/01 ≠ FULL; D1 FULL NP $3,592.66 = original exactly). `search_btc_ct_daily_oos_champion_select.py` |
| BNBUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 BROKE Max DD in OOS** (like BNB/ETH Hourly; BNB CT is OOS-fragile). Best compromise **BD4: LL=4 SL=1.5 LS=9 SS=1.2** (ultra-high-freq 109tr) — ONLY OOS-profitable **+$879**, lowest full MDD −$6,193, mildest break 1.32×. **CONTRARY to BTC/ETH Daily**: on BNB the HIGH-freq regime is OOS-best; the long-LS BD3 did NOT hold (1.41×). In-sample champion BD1 ($42,546) OOS-fragile (−$5,754, broke MDD). Clean OOS (IS = prior exactly 0% drift ≠ FULL). `search_bnb_ct_daily_oos_champion_select.py` |

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
| BTCUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **✅ 2 strict PASS — WINNER QB4: Len=8 Multiple=2.6** (short-Len high-freq) — OOS **+$947** (highest), full MDD −$880 = IS (HELD), full NP $2,469. QB3 (Len=7 Mult=2.56) also PASS (OOS +$819, MDD −$755 held). **🔴 IS-ceiling champion QB1 (Len=212 long-Len, lowest IS-MDD −$689) BROKE MDD (−$967, 1.40×) + lost OOS −$204**; QB2 same. **OOS-law generalizes beyond BollingerBreakout: long-Len IS-peak overfits, short-Len high-freq PASS.** Clean OOS (IS 0.0% drift ≠ FULL). `search_btc_qpatr_breakout_hourly_oos_champion_select.py` |
| BTCUSDT | Daily | >10K | Ceiling $2,544 (−74.6%; Len=13 Mult=1.845 13tr). Obj-max: Len=10 Mult=1.05 NP=$1,830 MDD=−$418 54tr. R1 "$3,227" was data-drift artifact. Daily < Hourly (opposite of QPATRex) |
| ETHUSDT | Hourly | >10K | Ceiling $4,444 (−55.6%; 8-conv; Len=32 Multiple=3.37 MDD=−$468 58tr Obj=42,191; **MDD/NP 10.5% best of all 4 BTC/ETH Hourly searches**) |
| ETHUSDT | Daily | >10K | Ceiling $3,986 (−60.1%; 9-conv; Len=9 Mult=1.705 14tr Obj=20,721; NP-max = Obj-max). ETH Breakout Daily > QPATRex Daily +26.1% (rare reversal) |
| BNBUSDT | Hourly | >100K | Ceiling $35,634 (−64.4%; 13-conv strongest ever; Len=3 Multiple=2.965 82tr). ⭐⭐⭐ Obj-max: Len=145 Multiple=2.91 NP=$32,506 MDD=−$4,610 Obj=229,204. BNB/BTC 12.97× |
| BNBUSDT | Daily | >100K | Ceiling $20,317 (−79.7%; 8-conv; Len=18 Mult=0.735 114tr HIGH-FREQ regime Obj=81,486; NP-max = Obj-max). Daily/Hourly −43% worst |

### `SFJ_XtremeStop_Crypto` (% breakout vs close X bars ago, reversal exits, `_Crypto1MUSD`)

Params: X (lookback bars), LY/SY (long/short breakout %). Logic: `pos≠1 → BUY C[X]*(1+LY*0.01) STOP`; `pos≠-1 → SHORT C[X]*(1-SY*0.01) STOP`. Workspace: `20260101_SFJ_XtremeStop_AI.wsp`. IS window 2022/01-2026/01 (chart-trimmed).

| Instrument | TF | Target | Status |
|---|---|---|---|
| BNBUSDT | Hourly | >100K | Ceiling ~$26-28K (−72%; R1-R3 params converged X≈905 LY≈3.0 SY≈10.65 asym long-X, 25-31tr; R2 $27,991 / R3 $25,665 data drift Rule 5; X & SY both bounded). $100K unreachable. Weaker than CT/QPATRex/Breakout |
| BNBUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **⚠️ Rule-5 FLIP across data refreshes. Original run: WINNER BX4 (X=70 LY=5.5 SY=4.5, mid-X high-freq 166tr) ONLY strict PASS (full MDD −$7,625=IS held) OOS +$3,024.** **2026-06-25 full-period re-test (data refreshed + FULL→2026/06/10): NO strict PASS — all 4 broke; de-facto best now BX3 (X=14 LY=6.5 SY=8.0 short-X high-freq) OOS +$8,334 (full NP $27,949 highest) break MILDEST 1.05×; BX4 now breaks 1.10× OOS +$362.** IS drift BX1 −11% / BX3 −6.2% (BX2/BX4 ~0%). Both runs agree: HIGH-freq OOS-robust (BX3/BX4 profitable), long-X BX1 + sym BX2 lose/collapse (BX2 −$8,518). **Lesson: data refresh can flip strict PASS/FAIL — re-test before deploy.** `search_bnb_xtreme_hourly_oos_champion_select.py` |
| BNBUSDT | Hourly (exit modules, main fixed at **BX3** X=14 LY=6.5 SY=8.0, IS) | improve NP | **🎯 TREND-pattern AGAIN: 6/6 modules HELP/neutral, 0 HURT** — even the SHORT-X high-freq BX3 main eats exit modules (XtremeStop family = modules help on BOTH long-X & short-X regimes; opposite of CT 24/24 hurt). Baseline (main-only BX3) NP=$19,614.2 MDD=−$8,392.1 60tr (A00 drift 0.000%, exact-champion row). ⭐ **M4 high_volatility_exit DAYRANGE=5.08 = best NP +16.68% ($22,885.7), MDD ~flat (−$8,410, +0.2%), Obj 62,274 highest** = best by your NP²/|MDD| (M4 rarely the NP-winner — usually M6/M5). ⭐⭐ **M5 QuantPass_PT_Exit PT_Base=0.27 = only MDD-improver: +4.68% NP AND MDD −8.0% (−$7,721).** M2 TrailingStop ATRSTP=66.3 +12.42%, M6 RescueTeam Length=580 std=3.9 +11.59% (both MDD flat); M3 +1.26%; M1 ATRstop STP=75 redundant (+0.0%, never-trigger). `search_bnb_xtreme_exit_modules.py` |
| ETHUSDT | Hourly | >100K | Ceiling $3,682 (−96.3%; R2=R3; X=796 LY=5.85 SY=3.35 long-X asym 46tr MDD=−$450 Obj=30,105; NP-max=Obj-max; X bounded both sides). Best crypto XtremeStop on BTC/ETH (lowest MDD). $100K unreachable. `search_eth_xtreme_hourly{,2,3}.py` |
| BTCUSDT | Hourly | >100K | Ceiling $2,606 (−97.4%; R1=R2; X=60 LY=12.75 SY=12.4 **symmetric high-pct** 13tr MDD=−$582 Obj=11,655; NP-max=Obj-max; X & % bounded). Distinct regime from BNB asym. `search_btc_xtreme_hourly{,2}.py` |
| BTCUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all broke, but CX1 (champion X=60 LY=12.75 SY=12.4) broke by only $2** (−582→−584, 1.003×) AND highest OOS **+$570** = de-facto winner. CX3 (long-X 69tr) OOS +$378 broke 1.16×. **High-freq CX2/CX4 (94/121tr) OOS-COLLAPSED** (−$1,175/−$1,272, MDD 2.3-2.9×) — OPPOSITE of BNB (low-freq robust on BTC). `search_btc_xtreme_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 broke.** Best-available EX2 (long-X low-SY X=720 LY=3 SY=1, 134tr) only OOS-profitable +$439 (broke 1.95×). **In-sample champion EX1 (X=796) OOS-WORST: −$1,097, MDD blew 5.7× (−$450→−$2,583)** — low IS-MDD = overfit to calm. ETH XtremeStop OOS-fragile (no safe regime). Re-validated 2026-06-25 (byte-identical, 0% drift). `search_eth_xtreme_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly (exit modules, main fixed at **EX2** X=720 LY=3 SY=1, IS) | improve NP | **🎯 TREND-pattern: 6/6 modules HELP/neutral, 0 HURT** — UNLIKE the CT/reversal pattern (24/24 hurt); long-X (720) XtremeStop behaves like a slow trend-follower. Baseline (main-only EX2) NP=$3,343.8 MDD=−$743.0 134tr (A00 drift 0.000%, exact-champion row). ⭐ **M6 RescueTeamExit Length=540 std=3.1 = best NP: $3,651.5 (+9.2%)** (MDD −$755 slightly deeper) — **M6 = Hourly cross-strategy NP-winner again**. ⭐⭐ **M5 QuantPass_PT_Exit PT_Base=0.589 = best JOINT: +7.2% NP ($3,584.7) AND MDD improved −19.7% (−$596.4), trades ~flat (134→139), Obj 21,546 highest** = best risk-adjusted (your NP²/|MDD|). M4 high_volatility DAYRANGE=4.68 +4.07% AND MDD −16%; M2/M3 +4%; M1 ATRstop STP=18.7 redundant (+0.0%, never-trigger). `search_eth_xtreme_exit_modules.py` |
| BNBUSDT | Daily | >100K | **CEILING $23,605 (−76.4%; R1/R2 9-conv; X=70 LY=0.5 SY=2.2 mid-X asym low-LY/mid-SY 43tr MDD=−$8,952 Obj=62,240; NP-max=Obj-max; R1→R2 +0.00%; X 20-160/LY 0.1-2.0/SY 0.5-4.0 all bounded).** First crypto XtremeStop Daily. **BNB Daily < Hourly** ($23.6K vs ~$26-28K; unlike CT/HUNTER2 where BNB Daily>Hourly). `search_bnb_xtreme_daily{,2}.py` |
| BNBUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 broke + all 4 LOST OOS** (weakest XtremeStop OOS cell). De-facto least-bad **BD4 X=46 LY=1.2 SY=3.0 (low-pct mid-X 61tr): OOS −$94 (≈flat), break MILDEST 1.05×.** 🔴 **IS ceiling champion BD1 (X=70, 43tr) = OOS DISASTER −$8,503, MDD blew 1.53× (−$8,952→−$13,682) = overfit.** ⚠️ **REFUTES "BNB Daily thick-edge": on XtremeStop the IS champ collapses + high-freq generalizes (standard OOS-law) — BNB-Daily-robustness is strategy-specific, NOT universal.** Clean OOS (IS drift 0.0%). `search_bnb_xtreme_daily_oos_champion_select.py` |
| BTCUSDT | Daily | >100K | **CEILING $4,020 (−96.0%; R1/R2 7-conv; X=268 LY=2.3 SY=0.1 LONG-X asym, SY at floor 43tr... 11tr MDD=−$1,742 Obj=9,282; NP-max=Obj-max; R1→R2 +0.00%; SY=0.1 genuine peak — swept 0.1-4.0 worse; X=268 verified 150-500).** Distinct from BTC Hourly sym high-pct. **BTC Daily > Hourly** ($4,020 vs $2,606; opposite of BNB/ETH). Dense alt X=42 LY=0.5 SY=0.2 (105tr $1,796 MDD−$831). `search_btc_xtreme_daily{,2}.py` |
| BTCUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **✅✅ ALL 4 strict PASS + all 4 OOS-profitable — FIRST all-pass crypto XtremeStop cell. WINNER CD4 X=55 LY=8.0 SY=6.5 (mid-X high-pct 13tr): OOS +$532 (highest), full NP $2,497, MDD held −$865.** All 4 full MDD == IS MDD (max DD structurally locked inside IS) → even 11tr sparse champ CD1 held (OOS +$268). **CONTRARY to BNB Daily (all broke): BTC's thin edge is OOS-robust because DD is IS-locked.** CD4 ≈ BTC Hourly sym high-pct family → high-% breakout generalizes across BTC TFs. Clean OOS (IS drift 0.0%). `search_btc_xtreme_daily_oos_champion_select.py` |
| ETHUSDT | Daily | >100K | **CEILING $3,387 (−96.6%; R1/R2; X=57 LY=3.45 SY=7.55 mid-X asym mid-LY/high-SY 15tr MDD=−$897 Obj=12,791; NP-max=Obj-max; R1→R2 +0.00%; 3rd distinct Daily regime vs BNB/BTC).** **ETH Daily < Hourly** ($3,387 vs $3,682; like BNB). Dense alt A06 X=58 LY=0.4 SY=3.25 (51tr); low-MDD A05 X=35 (32tr MDD−$625). OOS not yet run. `search_eth_xtreme_daily{,2}.py` |

### `SFJ_SuperTrend_crypto` (ATR-band trend-flip, market entries, reversal, `_Crypto1MUSD`)

Params: ATRLength, Multiplier. Logic: Up=C−Mult·ATR, Dn=C+Mult·ATR, trend flip; BUY TREND=1 & C↑Dn; SHORT TREND=−1 & C↓Up. Workspace: `20260101_SFJ_SuperTrend_AI.wsp`. **Weakest crypto strategy tested.** IS 2022/01-2026/01.

| Instrument | TF | Target | Status |
|---|---|---|---|
| BNBUSDT | Hourly | >100K | Ceiling $17,453 (−82.5%; R1=R2; ATR=79 Mult=6.625 wide-band 220tr MDD=−$7,614 Obj=40,008; NP-max=Obj-max). Both axes bounded (low-mult churns negative). $100K unreachable |
| BNBUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 broke MDD.** Best compromise **SS3 ATR=105 Mult=12** (low-freq 80tr): ONLY OOS-profitable +$827, lowest full MDD. SS4 long-ATR (489tr) OOS DISASTER (full NP **−$8,654**, MDD 3.2× blowup). In-sample champ SS1 OOS-fragile. `search_bnb_supertrend_hourly_oos_champion_select.py` |
| BTCUSDT | Hourly | >100K | Ceiling $1,986 (−98%; R1=R2; ATR=151 Mult=9.15 127tr MDD=−$586 Obj=6,733; NP-max=Obj-max; BTC=BNB/8.8 exactly). Wider bands than BNB. $100K unreachable |
| BTCUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **WINNER TS1: ATR=151 Mult=9.15** (in-sample champion ALSO wins OOS — rare) — strict PASS (full MDD −$586 = IS, held) + highest OOS **+$1,102** + lowest full MDD. TS4 (ATR=200 Mult=15.5, 53tr) also PASS. LOW-freq regimes hold MDD; high-freq broke (opposite of BNB CT/XtremeStop). Absolute NP tiny ($3K full). `search_btc_supertrend_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly | >100K | Ceiling $3,004 (−97%; R1=R2; ATR=45 Mult=10.45 widest-band 103tr MDD=−$621 Obj=14,534; NP-max=Obj-max). Both axes bounded. $100K unreachable. `search_eth_supertrend_hourly{,2}.py` |
| ETHUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **WINNER ES1: ATR=45 Mult=10.45** (in-sample champion ALSO wins OOS, like BTC) — strict PASS (full MDD −$621 = IS, held) + highest OOS **+$2,096** (full $5,100 vs IS $3,004) + lowest full MDD. All 4 OOS-profitable but only ES1 held MDD. `search_eth_supertrend_hourly_oos_champion_select.py` |
| BTCUSDT | Daily | >100K | Ceiling $1,805 (−98.2%; R1; X... ATR=3 Mult=6.5 10tr) — **sparse noisy spike** (zoom found lower nearby; 10tr degenerate). $100K unreachable. `search_btc_supertrend_daily.py` |
| ETHUSDT | Daily | >100K | Ceiling $2,413 (−97.6%; R1 6-conv stable; ATR=4 Mult=4.5 12tr MDD=−$855). Daily < Hourly. $100K unreachable. `search_eth_supertrend_daily.py` |
| BNBUSDT | Daily | >100K | **R1 ~$19,169 (−80.8%; ATRLength≈21-22 Multiplier≈4.0 mid-ATR 11tr MDD=−$10,256 Obj=35,831; NP-max=Obj-max; A10=A11; 11tr sparse — R2 not yet run).** 🎯 **STRONGEST SuperTrend cell of all** (≫ BTC Daily $1,805 / ETH Daily $2,413 / BNB Hourly $17,453); **BNB Daily > Hourly** (BNB Daily-works coin). Dense alt ATR=10 Mult=0.9 (130tr $10,933 MDD−$6,032). OOS not yet run. `search_bnb_supertrend_daily.py` |
| ETHUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **WINNER EDS2: ATR=4 Mult=1** (low-mult high-freq 117tr) — only OOS-profitable strict PASS (full MDD −$797 = IS, held; OOS +$206). In-sample wide-band champ EDS1 (ATR=4 Mult=4.5, 12tr) made more OOS (+$444) but BROKE MDD — Daily wide-band too sparse to hold (opposite of Hourly where wide-band champ held). `search_eth_supertrend_daily_oos_champion_select.py` |

**SuperTrend exit modules (BTC + ETH Hourly):** UNLIKE CT (24/24 modules hurt), exit modules **HELP** SuperTrend (trend strategy): BTC 4/6 + ETH 6/6 positive in-sample. M5 QuantPass_PT_Exit adds most NP (BTC +8.8% / ETH +14.6%) but MDD flat; M6 RescueTeam IS-strong but MDD deepens. **Full-period OOS joint test (NP↑ AND MaxDD↓): M2 TrailingStop PASSES on BOTH** (BTC ATRSTP=51.7, ETH ATRSTP=46.4) — the consistent robust improver; BTC also M3 EXITBAR=425 passes but ETH M3 EXITBAR=30 was IS-overfit (full NP −10.5%). `search_{btc,eth}_supertrend_exit_modules.py` + `..._oos_validation.py`.

### `SFJ_HUNTER2_crypto` (MA filter + ATR-stop entry, max 1 entry/day, reversal exits, `_Crypto1MUSD`)

Params: LEN_L, LEN_S, ATR_multiplier_L, ATR_multiplier_S. Logic: `C>avg(C,LEN_L) & EntriesToday=0 → BUY C[1]+ATR_L·ATR(20) STOP`; `C<avg(C,LEN_S) → SHORT C[1]−ATR_S·ATR STOP`. Workspace: `20260101_SFJ_HUNTER_AI.wsp`. IS 2022/01-2026/01.

| Instrument | TF | Target | Status |
|---|---|---|---|
| BTCUSDT | Hourly | >100K | **Obj-max (your criterion) WINNER: LEN_L=225 LEN_S=825 ATR_L=4.25 ATR_S=1.75** (R3; NP=$2,720 MDD=−$320 Obj=23,082 192tr; ultra-long low-MDD, all axes bounded). NP-max alt: high-ATR LEN_L≈95 LEN_S≈176 ATR_L=3.8 ATR_S=11.6 ($3,282, MDD−$1,351, 14tr sparse). $100K unreachable. ⚠️ high-ATR sparse zoom hit MCReport packing garble (LEN_L col misread) — use coarse grids there. `search_btc_hunter2_hourly{,2,3}.py` |
| BTCUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS but BH1 (Obj-max ultra-long 225/825/4.25/1.75) de-facto WINNER**: highest OOS **+$1,498**, best full NP $4,254, lowest full MDD −$411, mildest break 1.28×. **IS Obj-max = OOS-best (rare alignment).** High-freq BH4 (338tr) OOS-COLLAPSED (MDD 3.4×); NP-max high-ATR BH2 OOS-fragile. `search_btc_hunter2_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly | >100K | **Obj-max (your criterion) WINNER: LEN_L=495 LEN_S=635 ATR_L=2.8 ATR_S=0.25** (R3 CEILING CONFIRMED; NP=$3,699 MDD=−$437 Obj=31,283 182tr; same asym ultra-long family as BTC, all 4 axes bounded). R1 found 2 regimes (short-MA NP-max LEN_L=24 LS=5 vs ultra-long Obj-max); R1 zoom only refined NP-max → R2 zoomed ultra-long seeding Obj-max (Obj 25,882→30,979 +19.7%), R3 confirmed (R2 reproduced 0.0% drift; ATR_S 0.3→0.25 +0.98%; ATR_S<0.25 & wide/boundary all worse). NP-max alt: LEN_L=490 LS=665 ATR_L=2.75 ATR_S=0.3 NP=$3,724 (same regime). ETH/BTC Obj 1.36×. $100K unreachable −96.3%; weak (~$3.7K, SuperTrend/XtremeStop tier). `search_eth_hunter2_hourly{,2,3}.py` |
| ETHUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 broke Max DD.** 3 ultra-long candidates all OOS-PROFITABLE (+$573..+$987) but MDD ~1.9-2.0× deeper; **EH4 short-MA high-freq COLLAPSED** (OOS −$131, MDD 3.46×, like BTC BH4 — HUNTER2 high-freq OOS-fragile on both). De-facto best **EH2 LEN_L=490 LS=665 ATR_L=2.75 ATR_S=0.3**: max OOS +$987, highest full NP $4,711, break 1.93×; EH1 (R3 champ) tied (+$981, 2.00×); EH3 mildest break 1.87× lowest OOS. ETH ultra-long more OOS-fragile than BTC (1.28×). `search_eth_hunter2_hourly_oos_champion_select.py` |
| BNBUSDT | Hourly | >100K | **Obj-max (your criterion) WINNER: LEN_L=136 LEN_S=187 ATR_L=3.75 ATR_S=2.75** (R2 CEILING, R1→R2 reproduced 0.0%; NP=$20,118 MDD=−$5,687 Obj=71,169 271tr; **MID-LEN regime, distinct from BTC/ETH ultra-long**, all axes bounded). NP-max alt: LEN_L=140 LS=195 ATR_L=3.7 ATR_S=3.0 NP=$21,435. ⚠️ R1 had 5/11 wizard-open failures (fixed in R2 via robust run_or_load) + long-LEN/ultra-long retested all worse. BNB/BTC 7.4× BNB/ETH 5.4×. $100K unreachable −78.6%. `search_bnb_hunter2_hourly{,2}.py` |
| BNBUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **WINNER BNH3 LEN_L=150 LEN_S=200 ATR_L=3.5 ATR_S=3.0** (long-LEN variant) — ONLY strict PASS (full MDD −$7,104 = IS, held) + OOS +$7,654. All 4 OOS-PROFITABLE (+$7,084..+$14,087); **BNH4 high-ATR low-freq made most OOS +$14,087 but broke 1.20×**; IS champion BNH1 OOS-fragile (broke 1.39×). OOS FLIPS to longer-LEN (like CT). **BNB = only crypto HUNTER2 with a strict PASS.** `search_bnb_hunter2_hourly_oos_champion_select.py` |
| BNBUSDT | Daily | >100K | **Obj-max = NP-max WINNER: LEN_L=2 LEN_S=68 ATR_L=1.0 ATR_S=0.9** (R2 CEILING, byte-identical reproduce 0.0%; NP=$31,731 MDD=−$8,424 Obj=119,521 56tr; **ULTRA-SHORT LEN_L, LEN_L=2 floor confirmed best**, ATR_S/ATR_L/LEN_S all bounded). **BNB Daily > Hourly (+48% NP, like BNB CT — Daily-works coin); strongest HUNTER2 crypto result.** $100K unreachable −68.3%. `search_bnb_hunter2_daily{,2}.py` |
| BNBUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **WINNER BND1 (= IS champion 2/68/1.0/0.9)** — ONLY strict PASS (full MDD −$8,424 = IS, held) AND max OOS **+$2,112** AND highest full NP $33,843. **Rare IS-champion = OOS-winner alignment** (vs BNB Hourly which flipped). All 4 OOS-profitable but BND2/BND3 broke; BND4 low-freq broke 1.01×. `search_bnb_hunter2_daily_oos_champion_select.py` |
| BTCUSDT | Daily | >100K | **Obj-max = NP-max WINNER: LEN_L=4 LEN_S=112 ATR_L=0.55 ATR_S=0.75** (R2 CEILING, R1→R2 +0.96% sub-grid; NP=$2,948 MDD=−$642 Obj=13,547 70tr; ULTRA-SHORT LEN_L like BNB Daily, **MDD hard-floor −$642**, flat plateau LEN_L 2-5/ATR_L 0.25-0.55). R1 equiv pt 2/116/0.25/0.75. BTC Daily ≈ Hourly ($2,948 vs $2,720). BNB/BTC Daily 10.8×. $100K unreachable −97.1%. `search_btc_hunter2_daily{,2}.py` |
| BTCUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **WINNER BTD4 LEN_L=27 LEN_S=110 ATR_L=0.35 ATR_S=0.4** (longer-LEN low-freq 47tr) — strict PASS (MDD held −$642) + max OOS **+$836** + highest full NP $3,698. BTD2 (R1 LEN_L=2) also PASS (+$86). **IS champion BTD1 OOS-fragile** (OOS −$368, broke 1.29×). **OOS FLIPS to low-freq longer-LEN (BTC low-freq-robust, opposite of BNB Daily where IS champ held).** `search_btc_hunter2_daily_oos_champion_select.py` |
| ETHUSDT | Daily | >100K | **Obj-max = NP-max WINNER: LEN_L=2 LEN_S=116 ATR_L=0.1 ATR_S=0.8** (R3 CEILING; NP=$3,697 MDD=−$933 Obj=14,646 90tr; ULTRA-SHORT LEN_L like BTC/BNB Daily — LEN_L=2 AND ATR_L=0.1 both at floor). **⚠️ R1 LEN_L=34 was a FALSE local peak** (R1 "3-conv" Obj 10,488); R2 lenL_sweep+Obj-seed found the LEN_L=2 regime (+39% →30,979... Obj 14,578), R3 confirmed +0.47%. ETH Daily ≈ Hourly ($3,697 vs $3,699). $100K −96.3%. `search_eth_hunter2_daily{,2,3}.py` |
| ETHUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **WINNER EHD4 = the R1 "false-peak" LEN_L=34 LEN_S=116 ATR_L=0.3 ATR_S=1.0** (mid-LEN low-freq 31tr) — ONLY strict PASS (MDD held −$888) + only OOS-profitable **+$698** + highest full NP $3,749. **⚠️⚠️ IS-best = OOS-worst overfit: the R3 ultra-short champion (EHD1) OOS-COLLAPSED (−$1,041, MDD 1.79×); all 3 ultra-short candidates lost OOS + broke MDD.** The IS-inferior mid-LEN regime generalizes; same flip as BTC Daily (→low-freq). `search_eth_hunter2_daily_oos_champion_select.py` |

**HUNTER2 exit modules (Hourly: BTC/ETH/BNB; Daily: BNB/BTC/ETH):** trend-strategy pattern, but **modules help on Hourly, NOT on Daily**. Hourly IS: all positive (BTC 5/6, ETH 6/6, **BNB 6/6 — M5 QuantPass_PT_Exit PT=0.069 +30.66% NP AND MDD↓25% = strongest M5 ever**; BTC PT=0.671 +10.2%; ETH PT=0.677 +9.52%). **Daily IS: modules HURT NP** (BNB max +4.78% M4; **BTC + ETH ALL negative** — BTC −1.2..−5.5%, ETH −12.9..−17.5% [6/6, incl. M6 −12.9%] — low-freq daily already tight). **M5 QuantPass_PT_Exit flips role by TF: Hourly = NP-booster (NP↑+MDD↓); Daily = MDD-slasher** (BNB M5 −45%MDD; BTC −42%; ETH −52%). Full-period OOS (Hourly BTC/ETH): no module reduces main-dominated MDD but M5 adds NP at equal MDD (BTC +6.7%, ETH +10.09%). ⚠️ **M6 RescueTeamExit Status-checkbox toggle reverts after OK on Daily auto-runs** (works via `--manual-status`; ETH Daily M6 ran manually = Length=40 std=3 −12.9%). **M5 QuantPass_PT_Exit = consistent across SuperTrend + HUNTER2.** `search_{btc,eth,bnb}_hunter2{,_daily}_exit_modules.py` + `..._oos_validation.py`. (All scripts `ensure_chart_ready` before chart-trim — connect() picks the largest MC window = Study Editor when open. ⚠️ The Data-Range trim does NOT re-restrict already-loaded wider data: an exit-module run launched right after an OOS run on FULL range silently ran full-period — added a baseline-drift>10% ABORT guard; manually set + verify the chart range first. Truncated-but-clean exports [RescueTeam high-Length drops to 0-trade] are accepted if they still start at the declared min, else garble→discard.)

### `QuantPassRSI` (RSI zero-threshold momentum, market entries, reversal exits, `_Crypto1MUSD`)

Params: Len (RSI period), RSI_Gap (threshold). Logic: `RSI(C,Len)>100−RSI_Gap → BUY market`; `RSI(C,Len)<RSI_Gap → SHORT market` (momentum: long when RSI high). Workspace: `20260101_QuantPassRSI_AI.wsp`. Weak crypto strategy (sparse 10-22tr Hourly; $100K unreachable everywhere).

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | Strongest QPRSI: Obj-max Len=56 Gap=34 NP~$17.4K MDD−$12.2K (deep) 16tr. OOS WINNER BS4 +$3,384 |
| BTCUSDT | Hourly | Ceiling ~$2.2K (Obj-max Len=62 Gap=34 15tr). OOS least-bad RS2 +$515 |
| ETHUSDT | Hourly | Ceiling ~$3.4K (Len=12-17 Gap=15-19 22tr). OOS: NO strict PASS (all broke) |
| BTCUSDT | Daily | Ceiling ~$1.6K (Len=63 Gap≈48 near-degenerate 11tr). OOS WINNER BSD4 +$892 (low-freq flip) |
| ETHUSDT | Daily | Ceiling ~$1.2K (Len=50 Gap=47 10tr, weakest). OOS WINNER ESD3 dense Len=2 Gap=12 +$726 (sparse IS champ ESD1 collapsed) |
| BNBUSDT | Daily | ~$6K but **data-unstable (Rule 5 drift), unusable** |
| ETHUSDT | Hourly (exit modules) | ALL hurt (reversal/momentum type — same as CT) |

OOS law reconfirmed: **sparse IS-Obj-max champions overfit/collapse OOS; denser regimes (low Gap / short Len) generalize** (ESD3, BSD4). `search_{btc,eth,bnb}_qpatrsi_{hourly,daily}{,2}.py` + `..._oos_champion_select.py`.

### `_2021Basic_Osc_crypto` (BB-oscillator, STOP entries, ATR(10) STP/LMT exits, `_Crypto1MUSD`)

Params: LEN, LE, SE, STP, LMT (LE/SE = STDDEV multipliers, LE can be negative). `BUY H STOP` when C crosses over `BollingerBand(C,LEN,LE)`; `SHORT L STOP` when C crosses under `BollingerBand(C,LEN,SE)`; STP×ATR(10) stop + LMT×ATR(10) limit. Workspace: `20260101_SFJ_BASIC_OSC_AI.wsp`. **Same strategy, 3 coins, 3 OPPOSITE regimes.**

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **Strongest** — R2 ceiling Obj-max=NP-max LEN=13 LE=−0.5 SE=3 STP=1 LMT=19.5 NP=$15,113 MDD=−$1,919 Obj=119,051 166tr (B07=B08=B09); **asymmetric regime** (LE≈0 long / SE=3 short); MDD/NP 12.7% best. SE/LE confirmed (push worse), LMT crept 13.5→19.5 |
| ETHUSDT | Hourly | R2 ceiling Obj-max LEN=45 LE=−3 SE=2 STP=1 LMT=17 NP=$1,656 MDD=−$361 Obj=7,599 394tr (6-attempt byte-identical); **deep-LE mean-reversion regime**; LE=−3 confirmed peak (push to −6 worse) |
| BTCUSDT | Hourly | R2 ceiling Obj-max LEN=6 LE=2.25 SE=−0.5 STP=2 LMT=9 NP=$946 MDD=−$187 Obj=4,790 66tr (8-attempt byte-identical); **momentum/breakout regime** (LE>0); LE/SE confirmed (push worse) |

OOS (Data Range 2021/03-2026/06): **BNB WINNER BNO3** (NP-max hi-freq LEN=15 LE=0 SE=3 STP=2 LMT=20) OOS **+$4,583** held MDD −$6,506; **ETH WINNER EO4** (alt momentum LEN=20 LE=0 SE=3 STP=2 LMT=11) OOS +$301 held MDD −$645; **BTC NO PASS** (all 4 broke + OOS-loss; do not deploy). Law reconfirmed: **IS tight-MDD Obj-max champions overfit & break OOS; IS-MDD-wider high-freq regime generalizes** (BNO3, EO4). `search_{eth,btc,bnb}_osc_hourly{,2}.py` + `..._oos_champion_select.py`.

### `SFJ_MACD_Strategy03_crypto` (MACD zero-cross entries, histogram-cross exits, `_Crypto1MUSD`)

Params: FastLength, SlowLength, MACDLength (all int; valid Fast<Slow). `MACD(C,Fast,Slow)` crosses 0 → market entry (over=BUY, under=SHORT); histogram (`MACD−signal(MACDLength)`) crosses 0 → exit. Workspace: `20260101_SFJ_MACD03_AI.wsp`. **Weakest crypto strategy tested — poor risk profile.**

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | R2 ceiling Obj-max Fast=62 Slow=114 MACD=30 NP=$7,957 MDD=−$7,065 Obj=8,961 215tr; **short/mid MACD ALL-LOSING (whipsaw, every combo NP<0); only long-period works**; MDD/NP **88.8%** (NP-max even 105%, MDD>NP) — Obj 13× below Osc. Long-period bounded (push longer worse). Not recommended live |

### `SFJ_DonchianATR_crypto` / `_v2` (Donchian breakout + ATR chandelier trail, long+short, `_Crypto1MUSD`)

**Self-authored strategy** (`Strategy/SFJ_DonchianATR_crypto{,_v2}.txt`). Workspace: `20260622_SFJ_DonchianATR_AI.wsp`.

- **v1** (Length, ATRLength, ATRMult): market-breakout + ATR trail + always-in flip. **LOST on EVERY combo on BNB Hourly** (6,889 combos, NP −$94K..−$1.3K, 119-3,602 trades) — whipsaw / over-trading. Tight default trail (M=3) the main culprit.
- **v2** (Length, ATRLength, ATRMult, TrendLen, ReentryBars): adds 3 anti-chop filters — trend MA filter, **STOP-breakout entry at prior channel edge**, **flat-only + ReentryBars cooldown**. **Fixed it.**

**3 coins, 3 DISTINCT regimes** (the 3 anti-chop levers are coin-dependent — proves keeping all 3 was right):

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R3 ceiling, 2 regimes.** ⭐ Obj-max Length=3 ATRLength=16 ATRMult=7 TrendLen=10 ReentryBars=17 NP=$19,141 MDD=−$3,661 Obj=100,089 212tr. 🥇 NP-max Length=2 ATRLength=12 ATRMult=16 TrendLen=15 ReentryBars=10 NP=**$24,426** MDD=−$6,394 66tr (M=16 wide-trail low-freq; M>16 worse). **TrendLen filter →floor (not helpful here); wins from WIDE trail + cooldown.** NP=highest non-CT BNB |
| BTCUSDT | Hourly | **R2 ceiling (soft).** Obj-max Length=32 ATRLength=16 ATRMult=9 TrendLen=128 ReentryBars=3 NP=$1,337 MDD=−$449 Obj=3,981 174tr. NP-max Length=50 ATRMult=8.5 TrendLen=80 ReentryBars=0 NP=$1,446. **MID-long L + TrendLen ACTIVE (80-128, NEEDS the filter) + R~0** — opposite of BNB. ATRMult peak ~8.5-9 (M>9 worse). BNB/BTC 17× |
| ETHUSDT | Hourly | **R2 ceiling.** ⭐ Obj-max Length=167 ATRLength=22 ATRMult=2 TrendLen=107 ReentryBars=0 NP=$1,593 MDD=−$275 Obj=9,217 402tr (MDD/NP 17.3%). NP-max Length=20 ATRMult=6 TrendLen=40 NP=$1,844 MDD=−$995. **ULTRA-LONG L + TIGHT M=2 (!) + TrendLen active** — M=2 opposite of BNB/BTC wide trail. Obj > ETH Osc |
| BNBUSDT | Daily | **R2 ceiling (sparse 10-11tr, 5-conv).** Obj-max=NP-max Length=10 ATRLength=10 ATRMult=4.5 TrendLen=65 ReentryBars=14 NP=$19,045 MDD=−$7,041 Obj=51,509 10tr. NP-max alt ultra-short Length=2 ATRMult=5 TrendLen=5 NP=$20,461 MDD=−$9,097. **Daily < Hourly** (opposite of BNB CT/HUNTER2). TrendLen active. Wide-trail/denser worse. OOS-select script ready, not yet run |

OOS (Data Range 2021/03-2026/06): **BNB Hourly WINNER DN3** (Length=4 ATRMult=7 TrendLen=5 ReentryBars=17, the R2 Obj-max M7 regime) OOS **+$5,704** held MDD −$4,678, full NP **$26,511** — only strict PASS, strongest OOS of all BNB strategies. **ETH NO PASS** (all broke; ED1/ED2 ultra-long tight-M broke hardest 2.20× = overfit; best-compromise ED3 short-L M6 OOS +$322 broke 1.40×; not deployable). BTC OOS not yet run.

**BNB exit modules (main fixed at DN3, IS):** trend-strategy pattern — **ALL 6 modules HELP** (6/6 positive). ⭐ **M5 QuantPass_PT_Exit PT_Base=0.15 = +64.46% NP ($20,808→$34,220) AND MDD↓ (−$4,678→−$4,371)** — strongest M5 Hourly NP-boost yet; pushes Donchian v2 to Obj≈268K (> CT 173K, Osc 119K, in-sample). M6 RescueTeam +37.61% (MDD↓), M3 EXITBAR=138 +17.34%, M2 TrailingStop +8.21% (MDD↓), M1 +5.13%, M4 +3.99%. ⚠️ **Bug fixed: a module param name can COLLIDE with a main param** (RescueTeamExit's `Length` vs Donchian's `Length`) — the main-fixed-value check now excludes columns matching a module-param name. `search_bnb_donchian_v2_exit_modules.py` (chart-trims to IS first + baseline-drift>10% ABORT guard, since prior OOS run leaves chart on FULL).

`search_{bnb,btc,eth}_donchian_v2_hourly{,2,3}.py` + `search_bnb_donchian_v2_daily{,2}.py` + `..._oos_champion_select.py` + `search_bnb_donchian_v2_exit_modules.py`. v1 `search_bnb_donchian_hourly.py` (all-negative, abandoned).

### `SFJ_KeltnerTrend_crypto` (Keltner EMA±ATR band breakout, STOP entries, ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_KeltnerTrend_crypto.txt`). Workspace: `20260622_SFJ_KeltnerTrend_crypto_AI.wsp`. Params: EMALen (EMA+ATR period, int), BandMult (band half-width ATR mult, frac), ATRMult (trail, frac). `UpBand=EMA(EMALen)+BandMult*ATR; DnBand=EMA−BandMult*ATR`; Buy at UpBand STOP / SellShort at DnBand STOP (flat-only); ATR chandelier trail. ⚠️ `Upper`/`Lower` are PowerLanguage reserved words → renamed `UpBand`/`DnBand`.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R2 ceiling (CONFIRMED clean — re-ran post-fix, byte-identical, so NOT contaminated; only RSIPullback was).** Obj-max=NP-max EMALen≈107 BandMult=1.0 ATRMult≈5.4 NP=$10,132 MDD=−$7,183 Obj=14,291 408tr (R2 zoom-best $11,602). Short/mid-EMA all-losing; only long-EMA (~100-106) works. **Weaker than Donchian** (NP ~½, MDD/NP 70%) — High/Low channel beats EMA±ATR band |

`search_bnb_keltner_hourly{,2}.py`. ⚠️ Workspace's Binance source "not connected" → Format Instrument opened on Lookup tab → data-range trim silently failed (see Key Constraints).

### `SFJ_RSIPullback_crypto` (trend-filtered RSI pullback, market entries, ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_RSIPullback_crypto.txt`). Workspace: `20260622_SFJ_RSIPullback_crypto_AI.wsp`. Params: TrendLen (MA filter, int), RSILen (int), RSIThresh (long cross level, int), ATRMult (trail, frac). Long: `Close>MA(TrendLen) AND RSI(RSILen) cross over RSIThresh`; short mirror (100−RSIThresh); ATR(14) chandelier trail; flat-only.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | ⚠️ **R1/R2 IS were OOS-CONTAMINATED** (pre-fix; the "$25,303 strongest" was inflated by ~6 mo of OOS data). **CLEAN IS re-run R1** (post-fix, Settings tab reached): Obj-max TrendLen=5 RSILen=18.5 RSIThresh=19 ATRMult=7 NP=$7,030 MDD=−$2,357 Obj=20,970 11tr; NP-max TrendLen=20 RSILen=5 RSIThresh=50 ATRMult=7 NP=$8,462 228tr. **True IS only ~$7-8.5K (weak, not converged)** — clean "pullback" Obj-max wants deep dip (Th=19). R2 contaminated dir removed |

`search_bnb_rsipullback_hourly{,2}.py` (R2 contaminated, removed) + `..._oos_champion_select.py`.

### `SFJ_ROCmomentum_crypto` (ROC momentum-magnitude breakout, ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_ROCmomentum_crypto.txt`). Params: ROCLen (int), ROCThresh (% frac), ATRMult (frac). `ROCv=(C-C[ROCLen])/C[ROCLen]*100`; long ROCv cross over +ROCThresh, short cross under −ROCThresh; ATR(14) trail; flat-only. **Clean IS** (post-fix): Obj-max=NP-max ROCLen~148 ROCThresh=2.25 ATRMult=3 NP=$5,643 MDD=−$4,299 Obj=7,407 491tr (MDD/NP 76%). **Weak**: the high-threshold anti-whipsaw thesis FAILED — optimizer wants LOW threshold (2.25%) + long lookback (slow trend-follow); short/mid ROCLen all-losing. Not converged. `search_bnb_rocmomentum_hourly.py`.

### `SFJ_ParabolicSAR_crypto` (Wilder PSAR stop-and-reverse, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_ParabolicSAR_crypto.txt`). Params: AfStep, AfMax (both frac). Always-in stop-and-reverse; flip → Buy/SellShort. **Clean IS** (post-fix): only the slowest near-floor point AfStep=0.002 AfMax=0.01 was positive NP=$4,626 MDD=−$12,336 (MDD/NP **267%**) — all other settings all-losing; the positive point is a razor-thin spike (zooms all-negative). **DOES NOT WORK** — always-in reversal (no filter) whipsaws on crypto, like MACD zero-cross. `search_bnb_parabolicsar_hourly.py`.

### `SFJ_ADXtrend_crypto` (ADX-gated DMI crossover + ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_ADXtrend_crypto.txt`). Params: DMILen (int), ADXThresh (int), ATRMult (frac). `diPlus=DMIPlus(DMILen); diMinus=DMIMinus(DMILen); adxv=ADX(DMILen)`; long flat & adxv>ADXThresh & DI+ cross over DI−; short mirror; ATR(14) trail; flat-only.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R2 ceiling — 2nd-strongest self-authored (≈Donchian), clean IS.** ⭐ Best DMILen=12 ADXThresh=26 ATRMult=7.8 NP=**$23,938** MDD=−$6,658 Obj=86,068 155tr (MDD/NP 28%); stable 5-conv DMILen=12 ADXThresh=26 ATRMult=8 NP=$22,286. **ADX gate=26 confirmed TRUE peak (B03 full sweep)** — the ONLY self-authored where the trend-strength filter genuinely helps (vs RSIPullback TrendLen→floor). Wide ATR trail ~8 confirmed. **OOS: NO PASS** (all 4 broke + OOS-loss −$1.3K..−$6.2K; AD4 lowest-IS-MDD broke hardest 3.05× = overfit). IS strong ≠ OOS robust |

`search_bnb_adxtrend_hourly{,2}.py` + `..._oos_champion_select.py`.

### `SFJ_BBSqueeze_crypto` (Bollinger squeeze breakout — volatility-regime filter + ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_BBSqueeze_crypto.txt`). Params: BBLen (int), BBmult (std-devs frac), SqueezeLen (int), ATRMult (frac). `UpBB/DnBB=BollingerBand(C,BBLen,±BBmult)`; squeeze=`BWidth<Average(BWidth,SqueezeLen)`; while flat & squeeze Buy at UpBB STOP / SellShort at DnBB STOP; ATR(14) trail.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R2 ceiling — 3rd-strongest self-authored, clean IS.** ⭐ BBLen=14 BBmult=1.5 SqueezeLen=50 ATRMult=8 NP=$16,119 MDD=−$7,058 Obj=36,811 185tr (3-conv; MDD/NP 44%). Squeeze filter active (SqueezeLen~50); wide ATR trail ~8. **OOS: NO PASS** (all 4 broke + OOS-loss −$1.0K..−$5.1K). < Donchian/ADXtrend |

`search_bnb_bbsqueeze_hourly{,2}.py` + `..._oos_champion_select.py`.

### `SFJ_TurtleChannel_crypto` (Turtle dual Donchian channel — channel entry + channel exit, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_TurtleChannel_crypto.txt`). Params: EntryLen, ExitLen (both int). Buy/SellShort at the EntryLen-bar channel STOP (flat-only); exit on the opposite ExitLen-bar channel (Turtle System 1).

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | ❌ **ALL-NEGATIVE on every combo** (6,889+, NP −$96K..−$888; 139-4,890 trades = whipsaw). Same channel ENTRY as Donchian but the **channel EXIT (shorter Donchian) whipsaws** — exits winners early, re-enters. Least-bad EntryLen=260 ExitLen=10 still −$888. **Does not work** |

`search_bnb_turtlechannel_hourly.py`. **KEY LESSON: the decisive factor is the EXIT, not the channel entry** — Donchian's wide ATR chandelier trail (lets winners run) is what makes it work; the Turtle channel exit (reactive) and PSAR reversal both whipsaw on crypto.

### `SFJ_HeikinAshi_crypto` (smoothed Heikin-Ashi trend-flip + ATR chandelier trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_HeikinAshi_crypto.txt`). Params: HASmooth (OHLC EMA smoothing int), ATRMult (trail frac). `sO/sH/sL/sC=XAverage(O/H/L/C,HASmooth); HAClose=(sO+sH+sL+sC)/4; HAOpen=(HAOpen[1]+HAClose[1])/2`; flat-only: long on `HAClose cross over HAOpen` (market), short on cross under; ATR(14) chandelier trail exit. Workspace: `20260622_SFJ_SFJ_HeikinAshi_crypto_AI.wsp` (⚠️ actual filename has DOUBLE `SFJ_SFJ`).

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R2 ceiling CONFIRMED $20,689** (8-conv byte-identical: R1 A06/A09/A10/A11 + R2 A01/A09/A10/A11; R1→R2 +0.00%). NP-max = Obj-max: HASmooth=3 ATRMult=8.25 MDD=−$6,348 Obj=67,428 196tr (MDD/NP 30.7%). **Short-smooth + WIDE ATR trail wins**: tight-trail grids (ATRMult 2.5-5) all-NEGATIVE; HASmooth 1-2 floor worse + deeper MDD; ATRMult 8.25→20 all worse (8.25 interior peak). **#4 self-authored**. $100K −79.3% unreachable. `search_bnb_heikinashi_hourly{,2}.py` |
| BNBUSDT | Hourly (champion OOS) | **NO strict PASS — all 4 broke MDD (1.57×-3.81×).** Clean OOS (IS 0.0% drift, IS≠FULL). Best compromise **HA3 ultra-wide low-freq HASmooth=4 ATRMult=16** (71tr): only meaningful OOS profit **+$2,153**, mildest break 1.57×, lowest full MDD. **IS champion HA1 (3/8.25) OOS-LOST −$5,330 + broke 2.53× = overfit**; HA2 raw-HA (HASmooth=1) OOS-collapsed −$18,649 (3.81×, no-smoothing most fragile). Here WIDE-trail low-freq generalized (not the densest). `search_bnb_heikinashi_hourly_oos_champion_select.py` |

### `SFJ_VolatilityBreakout_crypto` (Larry-Williams volatility breakout STOP entry + ATR chandelier trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_VolatilityBreakout_crypto.txt`). Params: ATRLen (ATR period int), EntryMult (breakout buffer in ATR mults, frac), TrailMult (trail, frac). `ATRv=AvgTrueRange(ATRLen); BuyLevel=Close+EntryMult*ATRv; SellLevel=Close-EntryMult*ATRv`; flat-only STOP entries at BuyLevel/SellLevel; ATR chandelier trail exit (TrailMult×ATRv). Workspace: `20260622_SFJ_VolatilityBreakout_crypto_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R3 ceiling CONFIRMED $22,046** (ATRLen=45 EntryMult≈0.01-0.025 TrailMult=16.5 MDD≈−$8,056 Obj≈60,335 105tr; ATRLen byte-identical 5× at 45, both-sides bounded; R2→R3 +1.9%). NP-max = Obj-max. **🔴 The volatility-buffer novelty is REJECTED — EntryMult driven to floor every round (0.1→0.05→0.025→0.01, →0); strategy degenerates to "near-immediate breakout + WIDE ATR trail (16.5 interior peak, >17 worse)".** Strongest confirmation yet: entry mechanism irrelevant, wide ATR trail decisive. **#3 self-authored** (beats HeikinAshi). $100K −78% unreachable. `search_bnb_volatilitybreakout_hourly{,2,3}.py` |
| BNBUSDT | Hourly (champion OOS) | **NO strict PASS — all 4 broke MDD (1.31×-2.85×).** Clean OOS (IS 0.0% drift). Best compromise **VB3 long-ATRLen tight-trail ATRLen=125 EntryMult=2.25 TrailMult=8.0** (171tr): only meaningful OOS profit **+$4,924**, mildest break 1.31×, lowest full MDD. **🔴 IS champion VB1 (45/0.05/16.5) = OOS-WORST −$14,696 (full NP $21,628→$6,932, broke 2.31×) = most dramatic overfit-collapse yet.** 🔄 **OOS FLIP: the IS-rejected buffer (EntryMult=2.25) + tight trail (8) generalizes; wide-trail regimes (VB1/VB2) OOS-collapse** — wide ATR trail wins IS but amplifies single-trade OOS losses. `search_bnb_volatilitybreakout_hourly_oos_champion_select.py` |

### `SFJ_FractalBreakout_crypto` (Williams-fractal swing-pivot breakout + ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_FractalBreakout_crypto.txt`). Params: FracLen (fractal strength, int), ATRMult (trail frac). Last confirmed 2·FracLen+1 swing pivot → Buy STOP@lastHi / SellShort STOP@lastLo; ATR(14) chandelier trail. Workspace: `20260622_SFJ_FractalBreakout_crypto_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | ❌ **ALL-NEGATIVE on every combo** (1,345 combos, max NP −$901; tight-trail 1997tr whipsaw). Least-bad FracLen=3 ATRMult=13.5 (wide trail) still −$901. **Does not work.** KEY LESSON: Donchian's edge is the ROLLING extreme (Highest/Lowest), NOT a LAGGED confirmed pivot — fractal confirmation lag + entering at a stale prior pivot whipsaws. `search_bnb_fractalbreakout_hourly.py` |

### `SFJ_ChannelClose_crypto` (close-confirmed Donchian breakout, market entry + ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_ChannelClose_crypto.txt`). Params: ChanLen (Donchian lookback, int), ATRMult (trail frac). `Close > Highest(High,ChanLen)[1]` → Buy next-bar MARKET (mirror short); ATR(14) chandelier trail. Workspace: `20260622_SFJ_ChannelClose_crypto_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | ⚠️ **Near-dead — only the wide-trail corner is positive, degenerate** (ChanLen=28 ATRMult=16.5 NP=$2,952 MDD=−$12,053 **MDD/NP 408%** Obj=723; 6/9 grids all-negative). **Does not work usefully.** KEY LESSON: close-confirmed MARKET entry enters LATE at a worse price vs Donchian's intrabar STOP fill → far weaker. `search_bnb_channelclose_hourly.py` |

### `SFJ_CloseChannelBreakout_crypto` (CLOSE-channel breakout, intrabar STOP entry + cooldown + ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_CloseChannelBreakout_crypto.txt`). Params: Length (close-channel lookback int), ATRMult (chandelier trail frac), ReentryBars (cooldown int). `Buy STOP @ Highest(Close,Length)[1]` / `SellShort STOP @ Lowest(Close,Length)[1]` (flat-only + ReentryBars cooldown); ATR(14) chandelier trail. Workspace: `20260622_SFJ_CloseChannelBreakout_crypto_AI.wsp`. **The FIXED version of ChannelClose** (CLOSE channel not High/Low + intrabar STOP not market) — tests a 3rd breakout reference (close channel) vs Donchian (high/low) & Bollinger (stat band).

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R2 CEILING CONFIRMED $22,008 (−78%; 10-conv; Length=8 ATRMult=7 ReentryBars=14 MDD=−$4,922 Obj=98,395 226tr; NP-max=Obj-max; R1→R2 +0.00%; ATRMult=7 interior peak [0.5-7 & 7-20 both worse], Length=8 [swept 2-20], Re=14 [swept 0-30]).** ⭐ **Obj 98,395 = #1 risk-adjusted of all 17 self-authored** (> BollingerBreakout 95,486); MDD/NP 22.4%. Close channel = wick-immune → low MDD. `search_bnb_closechannelbreakout_hourly{,2}.py` |
| BNBUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **strict PASS = CC3 (Length=10 ATRMult=7 Re=12, OOS +$2,810, full MDD −$10,212=IS held). 🥇 de-facto WINNER = CC1 (champion 8/7/14): OOS +$14,496 (highest; IS $22K→full $36,504 ~2×) breaking MDD by only $43 (1.009×, −4,923→−4,966) = razor-thin.** CC1 OOS-profit $14,496 > BollingerBreakout BO3 $9,687 → close-channel = strong OOS generalizer. CC2 +$10,491 (1.08×); CC4 wide-trail low-freq −$4,153 (1.39×). Clean OOS (IS 0.0% drift). `search_bnb_closechannelbreakout_hourly_oos_champion_select.py` |
| BNBUSDT | Hourly (exit modules, main fixed at **CC1** 8/7/14, IS) | improve NP | **🎯 TREND-pattern: 6/6 HELP/neutral, 0 HURT.** Baseline (main-only CC1) NP=$22,008 MDD=−$4,922 226tr (A00 drift 0.000%). ⭐ **M6 RescueTeamExit Length=200 std=3.9 = best by Obj (206,860): +38.05% NP ($30,382) AND MDD improved −9.3% (−$4,462)** = best on BOTH axes. ⭐⭐ **M5 QuantPass_PT_Exit PT_Base=0.196 = highest raw NP +46.29% ($32,196)**, Obj 205,786 (≈M6), MDD +2.3%. M2 TrailingStop ATRSTP=21.1 +14.19% AND MDD −3.9%; M1 +8.15%, M3 EXITBAR=90 +8.10%; M4 redundant (+0.0%). (M6 ran clean despite its `Length` colliding with main `Length` — validator excludes module-name-collision cols.) `search_bnb_closechannelbreakout_exit_modules.py` |
| BTCUSDT | Hourly | >10K | **R2 CEILING $1,866 (−98.1%; 8-conv; Obj-max Length=7 ATRMult=8 ReentryBars=19 MDD=−$364 Obj=9,578 223tr; MDD/NP 19.5%; R1→R2 +0.00%; ATRMult=8 interior peak [<7 & >8 worse], Length=7, Re=19). NP-max alt Length=5 ATRMult=10.25 Re=19 NP=$1,975 MDD=−$587 162tr.** Same family as BNB (short Length, wide trail ~8, cooldown) but higher Re (19 vs 14). BTC ~BNB/11.8 (weaker). `search_btc_closechannelbreakout_hourly{,2}.py` |
| BTCUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 broke.** De-facto best **CC4 wide-trail low-freq (11/14.5/10, 106tr): OOS +$495 (highest) + lowest full MDD (−$688), broke 1.39×**; CC3 short-len +$298 (1.41×). 🔴 **IS Obj-max champion CC1 (7/8/19, lowest IS-MDD −$364) = OOS-WORST −$215, MDD blew 2.98× (−$364→−$1,085) = textbook overfit-to-calm.** **Coin contrast: BNB CC1 de-facto winner (+$14,496, broke $43) but BTC CC1 collapses (2.98×) — BNB thick-edge holds, BTC thin-edge low-IS-MDD champ breaks** (same as BTC on other strategies). Clean OOS (IS 0.0% drift). `search_btc_closechannelbreakout_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly | >10K | **R2 soft-ceiling ~$4,100-4,700 (Rule-5 drift; WIDE-trail regime distinct from BNB/BTC). R2 stable 7/15/7 NP=$4,133 Obj 34,462 78tr; Obj-max 2/12/6 NP=$4,138 Obj 35,975 138tr (but the 2/12/6 ultra-short point is data-fragile — re-tested $1,024 in OOS pass, −75% drift).** ATRMult≈12-15 wide-trail confirmed (narrow 6-10 decisively worse IS, A05); short Length 2-7; low cooldown ~6. MDD/NP ~11%. ETH ~BTC×2.4. `search_eth_closechannelbreakout_hourly{,2}.py` |
| ETHUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 broke. 🎯🔴 OOS FLIPS the IS regime: the IS wide-trail champions CC2/CC3 (7/15/7 & 4/15.5/6, low IS-MDD) OOS-COLLAPSED (−$930 MDD 3.03× / −$192 MDD 2.88×); the IS-INFERIOR narrow-trail CC4 (6/7.5/12, BNB/BTC-style, 233tr) is de-facto WINNER: OOS +$767 (highest) + lowest full MDD (−$826), broke 1.35× mildest.** **→ ETH's "wide-trail is different" IS-finding was OVERFIT; OOS pulls ETH back to the BNB/BTC narrow-trail family.** CC1 (2/12/6) unusable (IS drift −75%). Clean OOS (CC2/3/4 0.0% drift). `search_eth_closechannelbreakout_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly (exit modules, main fixed at **CC4** 6/7.5/12, IS) | improve NP | **🎯 TREND-pattern: 6/6 HELP, 0 HURT.** Baseline (main-only CC4) NP=$2,676.5 MDD=−$610.2 233tr (A00 drift 0.000%). ⭐ **M5 QuantPass_PT_Exit PT_Base=0.479 = best Obj (18,108): +13.34% NP ($3,033.7) AND MDD improved −17% (−$508.2)** = best risk-adjusted. ⭐⭐ **M6 RescueTeamExit Length=260 std=4.5 = highest raw NP +22.18% ($3,270.1)**, MDD flat, Obj 17,524. M3 EXITBAR=130 +15.75%; M2 TrailingStop ATRSTP=14.7 +12.0% AND MDD −9%; M1/M4 small. `search_eth_closechannelbreakout_exit_modules.py` |
| ETHUSDT | Hourly (exit modules, **CUMULATIVE greedy full-period OOS stack**, main CC4) | max RoMaD on full data | **NEW task type: greedy-stack modules (fixed at IS-best) in descending IS-ΔNP% order over FULL 2021/03-2026/06/10, keep iff RoMaD=NP/\|MaxIntradayDD\| strictly rises.** A00 full baseline NP=$3,443 MIDD=−$826 RoMaD 4.170 310tr (= OOS CC4 full NP, confirms data). **FINAL KEPT = {M6+M3+M5+M4}: RoMaD 4.170→6.570 (+57.5%), NP $3,443→$5,540 (+61%), MIDD −826→−843 (flat).** Stack: M6 RescueTeam(260/4.5) +18.3% → M3 EXITBAR=130 +7.2% → M5 PT=0.479 +20.6% → **M2 TrailingStop DISCARDED (RoMaD −31%, MDD blew +31% −843→−1,105 = IS +12% NP overfit, deepens OOS DD)** → M4 DAYRANGE=4.98 +3.1% → M1 STP=9.6 DISCARDED (NP −6%). **Greedy+full-period-RoMaD objectively drops the IS-pretty-but-OOS-DD-deepening module (M2); more realistic than per-module-vs-baseline.** teardown leaves {M6,M3,M5,M4} ON for live. `search_eth_closechannelbreakout_cumulative_oos.py` |
| ETHUSDT | Daily | >10K | **SOFT/NOISY ceiling ~$1,600-1,900 (sparse 21-37tr; Obj CREEPS each round so NOT byte-clean: R1 24/3/12 Obj 5,520 → R2 3/3/5 6,654 → R3 9/2.5/10 7,773). Stable invariants: NARROW trail ATRMult≈2.5-3 (Daily, opposite of Hourly's wide 12-15), MDD floor ≈−$455, NP ~$1.7-1.9K.** target −98% unreachable; ETH CC Daily ≪ Hourly ~$4.1K. `search_eth_closechannelbreakout_daily{,2,3}.py` |
| ETHUSDT | Daily (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 broke. 🎯 de-facto WINNER = CC3 = the R1 mid-len 24/3/12 (21tr): OOS +$314 (highest) + lowest full MDD (−$615), break 1.35× mildest.** 🔴 **The R2/R3 IS-Obj creep was NOISE-OVERFIT: IS-Obj order R3(7,773)>R2(6,654)>R1(5,520) but OOS REVERSES — R3 champ CC1 (9/2.5/10) OOS −$158, R2 CC2 (3/3/5) +$84 broke 1.86×, the coarse R1 CC3 wins.** **LESSON: on sparse (21-37tr) daily surfaces do NOT multi-round chase Obj — R1 coarse + OOS-select suffices.** Clean OOS (0.0% drift). `search_eth_closechannelbreakout_daily_oos_champion_select.py` |
| BNBUSDT | Daily (1-BAT 4-stage pipeline: IS→OOS→exit→cumulative) | full deployable | **🎯 RESULT: deploy MAIN ONLY Length=5 ATRMult=5 ReentryBars=5 — NO exit modules.** S1 IS Obj-max Length=2 ATRMult=5 Re=12 NP=$25,576 Obj=95,647 11tr (**byte-identical to BNB BollingerBreakout Daily — close-channel & Bollinger band CONVERGE to the same sparse 11tr trade set on BNB daily**). S2 OOS: **3/4 candidates strict PASS (thick-edge, MDD held 1.00×); WINNER 5/5/5 OOS +$8,277 full NP $30,288**. S3 exit-modules: **6/6 HURT badly (−88% to −113% NP, MDD deepens, tr 22→37-154)**. S4 cumulative greedy: **every module collapses RoMaD (4.27→0.09-0.80) → FINAL kept-set = ∅**. **Contrast: ETH CC Hourly modules 6/6 HELP (kept 4); BNB CC Daily modules 6/6 HURT (kept 0) — sparse daily (11-22tr) + main already ATR-trailed → any exit truncates the big trends & churns.** Single unattended BAT ran all 4 stages with programmatic main+module input-setting, no ABORT. `run_bnb_ccb_daily_pipeline.py` |
| BTCUSDT | Daily (1-BAT 4-stage pipeline) | full deployable | **🎯 RESULT: deploy MAIN ONLY Length=2 ATRMult=4.5 ReentryBars=10 — NO exit modules.** S1 IS Obj-max 2/3.25/10 NP=$1,624 MDD=−$400 Obj=6,597 27tr (Length=2 floor, same sparse family as BTC BollingerBreakout Daily). S2 OOS: **ONLY C1 (NP-max 2/4.5/10) strict PASS, full MDD −$769 held (1.00×), OOS −$67 (flat)**; 🔴 textbook OOS-law: IS-Obj-max C0 (lowest IS-MDD −$400) COLLAPSED OOS −$1,395 MDD 3.15×, C2 3.28× — low IS-MDD overfits, wider-trail NP-max holds. S3 exit-modules (base $1,701): **6/6 HURT (M5 −33% best … M1 −79%), tr 20→63-130**. S4 cumulative: A00 RoMaD 2.12 → every module 0.0025-1.08 → **FINAL kept = ∅**. **BTC matches BNB: CC Daily deploy = MAIN ONLY** (sparse daily, modules all hurt); but OOS mechanism differs — BNB thick-edge sparse champ PASSES, BTC thin-edge Obj-max breaks (only wide-trail NP-max holds). `run_btc_ccb_daily_pipeline.py` |
| **CME.NQ** | **Hourly (FUTURES `SFJ_CloseChannelBreakout_NQ`, default sizing; 1-BAT 4-stage)** | NP>10M | **🎯 RESULT: deploy MAIN 2/7/10 + M2 TrailingStop(ATRSTP=1.5) — FIRST CC pipeline that KEEPS a module.** IS 2019-2025 / FULL 2018-2026 on **CME.NQ HOT** (futures = no _Crypto1MUSD, chart default contracts ~1 contract). S1 Obj-max=NP-max Length=4 ATRMult=8.5 Re=9 NP=$407,505 MDD=−$50,530 Obj=3.29M 322tr (R1=R2 +0.00% 1-round converge; **short Len 2-7 + WIDE ATR trail 7-8.5 + Re 9-10** — same param shape as the crypto CC Hourly family). target 10M −96% unreachable (NP scale = default-sizing-dependent). S2 OOS: **NO strict PASS — all 4 broke**; de-facto WINNER C2 2/7/10 OOS −$12,735 (least-bad) break 2.10× (Obj-max C0 broke 2.67×, OOS-law again). S3 exit (base $348,750): **6/6 HURT NP, tr 518→1,530-3,886 churn**, BUT M2 keeps MDD ~flat. S4 cumulative (FULL): A00 RoMaD 3.41 → **M2 TrailingStop ATRSTP=1.5 KEEP RoMaD 4.74 (+39%)** (slashes full MIDD −$98,555→−$55,440, also fixes OOS DD-break 2.10×→1.18×); all others on top of M2 collapse → **FINAL kept = {M2}**. **NEW LAW: CC Hourly (322-518tr, big full-period DD) → the TrailingStop DD-slasher is KEPT (RoMaD +39%), unlike CC Daily (sparse → kept=∅).** First futures-strategy result in the CC matrix. `run_nq_ccb_hourly_pipeline.py` |
| **TWF.TXF** | **Hourly (FUTURES `SFJ_CloseChannelBreakout_NQ`, default sizing; 1-BAT 4-stage)** | NP>10M TWD | **🎯 RESULT: deploy MAIN ONLY Length=10 ATRMult=3.5 ReentryBars=13 — NO modules. BEST CC futures cell (OOS-profitable, mild break).** IS 2019-2025 / FULL 2018-2026 on **TWF.TXF HOT**. S1 Obj-max=NP-max 10/3.5/13 NP=3,286,800 TWD MDD=−461,400 Obj=23.4M 678tr (R1=R2 +0.00% 1-round; **MID Len 10 + NARROW trail 3.5 + Re 13 — TXF wants a narrower trail than NQ's 8.5 / crypto's 7-8**). target 10M TWD −67% IS (but FULL 4.38M). S2 OOS: no strict PASS but **WINNER C0 OOS +1,090,800 (FULL NP 4.38M > IS!), break only 1.104×** (C2 5/5/12 also OOS +657K break 1.09×) — **CC's most OOS-robust futures cell** (vs NQ all-broke ~2× negative). S3 exit (base 3.29M): **6/6 HURT hard, M4/M5/M6 go NEGATIVE (−115..−122%), tr 678→1,094-2,223**. S4 cumulative: A00 RoMaD **8.59** (very high) → every module collapses (M2 best=2.53, M4/M5/M6 RoMaD<0) → **FINAL kept = ∅**. **NEW LAW refinement: whether exit modules help depends on whether the MAIN already has low DD — TXF main (narrow trail, RoMaD 8.59, OOS-positive) has no DD headroom → kept=∅; NQ main (wide trail, big full DD) had headroom → M2 kept. Exit-module value is conditional on main-DD slack, not on TF alone.** `run_txf_ccb_hourly_pipeline.py` |
| **CME.GC** | **Hourly (FUTURES `SFJ_CloseChannelBreakout_NQ`, default sizing; 1-BAT 4-stage)** | NP>10M | **🎯 RESULT: deploy MAIN ONLY Length=30 ATRMult=4.0 ReentryBars=31 — NO modules.** IS 2019-2025 / FULL 2018-2026 on **CME.GC HOT**. S1 Obj-max 30/4/30 / NP-max 30/4/31 NP=$136,380 MDD=−$18,010 Obj=1.04M 456tr (R1=R2 +0.00% 1-round; **LONG Len 30 + mid-narrow trail 4.0 + HIGH cooldown 31 — 3rd distinct futures regime: NQ 4/8.5/9, TXF 10/3.5/13, GC 30/4/31**). target 10M −99% (GC smallest absolute NP). S2 OOS: no strict PASS but **ALL 4 OOS-PROFITABLE; WINNER C1 30/4/31 OOS +64,030 (FULL NP $200,410) AND mildest break 1.947×** (highest-OOS = mildest-break, clean pick). GC OOS-positive like TXF but DD expands more (1.95-3.18× vs TXF 1.1×). S3 exit (base $136K): **6/6 HURT, M4/M6 ≤−85%, tr 456→1,286-1,468, all deepen MDD ~3×**. S4 cumulative: A00 RoMaD **5.72** → every module collapses (M2 best 1.02, M4/M6 RoMaD<0) → **FINAL kept = ∅**. Confirms the conditional law (RoMaD already high → no module headroom; only low-RoMaD wide-trail NQ kept M2). **Stage-4 teardown now writes the final best params back into Format Objects Input Strings** (verified in log). `run_gc_ccb_hourly_pipeline.py` |

### CloseChannelBreakout 3-coin Hourly matrix (IS + OOS COMPLETE)
| | IS ceiling | OOS verdict |
|---|---|---|
| BNB | $22,008 (Obj 98K #1 self-authored) | ✅ CC3 strict PASS / CC1 de-facto +$14,496 (broke $43) |
| BTC | $1,866 | ❌ all-broke; CC4 wide-trail de-facto +$495 |
| ETH | ~$4,100-4,700 (wide-trail, Rule-5) | ❌ all-broke; **CC4 narrow-trail de-facto +$767 (IS wide-trail champ collapsed 3×)** |
**🎯 VERDICT: close-channel breakout is a valid 3rd reference (Donchian high/low, Bollinger band, Close channel). BNB thick-edge holds OOS (only coin with a PASS); BTC/ETH thin-edge all-break (OOS-law: IS low-MDD champ overfits, IS-inferior higher-freq/narrow-trail generalizes). ETH's IS wide-trail "difference" was overfit — OOS reconverges to the BNB/BTC narrow-trail family. Exit-modules TREND-pattern (6/6 HELP) on all coins: M6 RescueTeam = max NP, M5 QuantPass_PT_Exit = best risk-adjusted.**

### `SFJ_DonchianAsym_crypto` (asymmetric Donchian channel breakout + ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_DonchianAsym_crypto.txt`). Params: LongLen, ShortLen (asymmetric channels, int), ATRMult (trail frac). Buy STOP@Highest(High,LongLen)[1] / SellShort STOP@Lowest(Low,ShortLen)[1]; ATR(14) chandelier trail. Workspace: `20260622_SFJ_DonchianAsym_crypto_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R1 weak ~$5,754** (LongLen≈17 ShortLen≈250 ATRMult≈8 MDD≈−$8K Obj≈4,114 193tr; MDD/NP 138%). **ONLY the asymmetric regime is profitable (symmetric grids all-negative)** — short-LongLen aggressive long + very-long-ShortLen suppresses bad shorts on the up-trending asset. KEY: asymmetry is NECESSARY but NOT SUFFICIENT — bare asym Donchian lacks the anti-chop cooldown. `search_bnb_donchianasym_hourly.py` |

### `SFJ_DonchianAsymV2_crypto` (asymmetric Donchian + ReentryBars cooldown + ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_DonchianAsymV2_crypto.txt`). Params: LongLen, ShortLen (int), ATRMult (trail frac), ReentryBars (flat cooldown int). Adds a post-exit cooldown to DonchianAsym. Workspace: `20260622_SFJ_DonchianAsymV2_crypto_AI.wsp` (⚠️ first run wasted 2h — the chart had the OLD `SFJ_DonchianATR_crypto_v2` signal applied, not the new one; verify Format Signals shows the right signal + input count before running).

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R2 ceiling CONFIRMED $14,969** (LongLen=6 ShortLen=300 ATRMult=7 ReentryBars=15 MDD=−$5,326 Obj=42,074 224tr; ShortLen=300 byte-identical 5×, both-sides bounded incl. pushed to 500; R1→R2 +2.3%). NP-max=Obj-max. **Synthesis works: asym + cooldown lifts bare asym $5,754 → $14,969 (+160%), MDD/NP 138%→36%.** 🎯 **ReentryBars is the FIRST entry-side lever the optimizer keeps >0 (≈15) instead of driving to floor — confirms cooldown (not trend filter) is Donchian v2's real anti-chop mechanism.** #6 self-authored. `search_bnb_donchianasymv2_hourly{,2}.py` |

### `SFJ_BollingerBreakout_crypto` (pure Bollinger-band breakout STOP + cooldown + ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_BollingerBreakout_crypto.txt`). Params: BBLen (int), BBmult (stdev mult frac), ATRMult (trail frac), ReentryBars (cooldown int). `Buy STOP @ BollingerBand(C,BBLen,BBmult)[1]` / `SellShort STOP @ BollingerBand(C,BBLen,−BBmult)[1]`; flat-only + cooldown; ATR(14) chandelier trail. Tests: is a statistical volatility band a better breakout reference than the Donchian high/low channel? Workspace: `20260622_SFJ_BollingerBreakout_crypto_AI.wsp`.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R2 ceiling CONFIRMED $21,210** (BBLen=13 BBmult=1.55 ATRMult=7 ReentryBars=13 MDD=−$4,711 **Obj=95,486 #2 self-authored** 217tr; MDD/NP **22.2% lowest**; R1→R2 +9.5%; BBLen byte-identical, all 4 axes interior peaks). NP-max=Obj-max. **🎯 Statistical band BEATS Donchian high/low channel on risk-adjusted return** (same engine; Bollinger MDD far lower). ReentryBars cooldown engaged (13) — 2nd confirmation. ⚠️ 1st run's grids silently no-op'd (cloned wrong source; BBmult got channel-length ranges → 7/9 grids invalid) — fixed by parsing `_do` grids FROM FILE, not a hardcoded list. `search_bnb_bollingerbreakout_hourly{,2}.py` |
| BNBUSDT | Hourly (champion OOS) | **✅ WINNER BO3 = 2nd self-authored OOS PASS (after Donchian v2): BBLen=9 BBmult=1.05 ATRMult=7 ReentryBars=17** — OOS **+$9,687**, full MDD −$5,939 = IS MDD (HELD), full NP $30,941. **🎯 FIRST strategy where ALL 4 candidates were OOS-PROFITABLE (+$8,605..+$13,470)** — Bollinger breakout generalizes across regimes. 2 PASS (BO3 + BO4 ultra-short 3/1.5/7/14 +$8,605). **IS Obj-max champion BO1 (lowest IS-MDD −$4,711) OOS-best-profit +$11,813 but BROKE MDD 1.20× = overfit-to-calm**; BO2 short-BBLen highest OOS +$13,470 broke 1.05×. Strict criterion (profit + no MDD break) → BO3. `search_bnb_bollingerbreakout_hourly_oos_champion_select.py` |
| BNBUSDT | Hourly (exit modules, main fixed at **BO2** 8/1.5/7/14, IS) | **TREND-pattern: ALL 6 modules HELP NP (6/6 positive)** — like SuperTrend/Donchian v2 (vs CT 24/24 hurt). Baseline (main-only BO2) NP=$18,628 MDD=−$4,814 233tr (A00 drift −0.001%, main-fixed verified). ⭐ **M6 RescueTeamExit Length=100 std=3.5 = best: NP $29,459 (+58.15%) AND MDD improved −$4,393** (only module best on BOTH). M5 QuantPass_PT_Exit PT_Base=0.196 +55.35% ($28,938, MDD −$4,905) = consistent cross-strategy NP-booster. M2 TrailingStop ATRSTP=17.3 +19.57% also cuts MDD. M3 EXITBAR=86 +17.94%, M1 STP=6.4 +16.27% (both deepen MDD), M4 +0.82% (negligible). `search_bnb_bollingerbreakout_exit_modules.py` |
| BTCUSDT | Hourly | **R2 ceiling CONFIRMED — Obj-max BBLen=8 BBmult=1.5 ATRMult=15 ReentryBars=22 NP=$2,236 MDD=−$425 Obj=11,773 98tr (MDD/NP 19% best); NP-max BBLen=4 BBmult=1.5 ATRMult=14 Re=21 NP=$2,308 MDD=−$859** (Obj-max byte-identical 3×; ATRMult=15 interior peak — push to 20 worse; R1→R2 +14.6% Obj). Same family as BNB but BTC wants WIDER trail (15 vs BNB 7), shorter BBLen (8 vs 13), higher cooldown (22 vs 13); BBmult≈1.5 both. ReentryBars engaged (3rd confirm). BTC/BNB ≈9.5×; $100K −97.8% unreachable. `search_btc_bollingerbreakout_hourly{,2}.py` |
| BTCUSDT | Hourly (champion OOS) | **NO strict PASS — all 4 broke MDD (1.08×-5.07×)** (vs BNB's 2 PASS — BTC OOS-fragile, thin edge). Clean OOS (IS 0.0% drift). Max-OOS **CO1 (8/1.5/15/22) +$574 but broke 2.15×**; mildest+profitable **CO4 long-BBLen hi-freq 13/1.8/7.8/21 +$119, broke 1.35×, lowest full MDD −$714**. 🔴 **CO3 wide-band tight-trail (lowest IS-MDD −$409, ATRMult=9) OOS-COLLAPSED −$1,727 MDD 5.07× = worst overfit** (tight trail + low IS-MDD; reconfirms wide-trail-decisive / tight-trail-amplifies-OOS-tail). `search_btc_bollingerbreakout_hourly_oos_champion_select.py` |
| BTCUSDT | Hourly (exit modules, main fixed at **CO1** 8/1.5/15/22, IS) | **TREND-pattern: modules HELP/neutral, 0 HURT** (gains concentrated — baseline trail already wide=15). Baseline (main-only CO1) NP=$2,236 MDD=−$425 98tr (A00 drift −0.013%). ⭐ **M6 RescueTeamExit Length=260 std=3.2 = best: NP $3,140 (+40.46%) AND MDD improved −$400** (same as BNB — M6 = cross-coin winner). M5 QuantPass_PT_Exit PT_Base=0.134 +25.08% ($2,797, MDD −$483). M1/M4 = 0% (params→never-trigger: STP=28.8/DAYRANGE=5.84, redundant w/ wide main trail); M2 ATRSTP=66 +3.33%, M3 EXITBAR=220 +3.88% (MDD flat). `search_btc_bollingerbreakout_exit_modules.py` |
| ETHUSDT | Hourly | **R3 ceiling CONFIRMED $5,267 — Obj-max=NP-max BBLen=4 BBmult=0.775 ATRMult=15.5 ReentryBars=6 MDD=−$473 Obj=58,693 68tr (MDD/NP 9.0% = best of 3 coins)** (BBmult=0.775 interior peak — 0.625/1.0 both worse; ATRMult=15.5 interior; 8× byte-identical at BBmult≈0.8; R2→R3 +0.06%). Shortest BBLen + narrowest BBmult + wide trail (like BTC) + lowest cooldown of the 3. BBmult≈1.4-1.55 NOT universal (ETH=0.775); ReentryBars>0 all 3 (4th confirm). ETH/BTC ≈2.36×; $100K −94.7% unreachable. `search_eth_bollingerbreakout_hourly{,2,3}.py` |
| ETHUSDT | Hourly (champion OOS) | **NO strict PASS — all 4 broke MDD (1.25×-2.47×)** (like BTC; **only BNB BollingerBreakout is OOS-robust**). Clean OOS (IS 0.0% drift). Cleanest OOS-law demo yet: 🔴 **IS Obj-max champion EO1 (4/0.775/15.5/6, lowest IS-MDD −$473, 68tr low-freq) = OOS-WORST: only LOSER −$500 + broke hardest 2.47×** (the wide-trail/short-BBLen IS-peak overfits). 3 higher-freq regimes (195-257tr) all OOS-profitable; best **EO3 BNB-like tight-trail 6/2.0/7/15 +$913, lowest full MDD −$927, broke 1.73×** (the BNB regime generalizes best on ETH OOS); EO4 mildest break 1.25×. `search_eth_bollingerbreakout_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly (exit modules, main fixed at **EO3** 6/2.0/7/15, IS) | **TREND-pattern: ALL 6 modules HELP NP (6/6, 0 HURT)** (narrow main trail=7 → room, like BNB). Baseline (main-only EO3) NP=$2,623 MDD=−$535 257tr (A00 drift −0.006%). ⭐ **M6 RescueTeamExit Length=20 std=3.9 = best NP: $3,256 (+24.14%), MDD flat** — **M6 = cross-coin winner on ALL 3 Hourly (BNB+58%/BTC+40%/ETH+24%)**. M3 EntryBarsAfterExit EXITBAR=47 +19.51% AND MDD −14.4% (best joint NP↑+MDD↓ on ETH). M5 PT_Base=0.109 +17.74% but MDD +49.9% (deeper). M1/M2/M4 small. `search_eth_bollingerbreakout_exit_modules.py` |
| ETHUSDT | Daily | **R3 ceiling CONFIRMED $2,291 — Obj-max=NP-max BBLen=10 BBmult=1.4 ATRMult=3.0 ReentryBars=14 MDD=−$429 Obj=12,247 21tr (MDD/NP 18.7%)** (BBmult=1.4 interior peak — 0.6 & 2.5 worse; 7× byte-identical; R2→R3 +0.00%). Daily<Hourly −56%. ETH likes narrow BBmult both TFs (H 0.775 / D 1.4). $100K −97.7% unreachable. `search_eth_bollingerbreakout_daily{,2,3}.py` |
| ETHUSDT | Daily (champion OOS) | **✅ WINNER ED4 = strict OOS PASS: BBLen=2 BBmult=3.0 ATRMult=2.0 ReentryBars=11** — OOS **+$1,249** (highest), full MDD −$500 = IS (HELD), full NP $2,858. 45→63tr. **🔴 textbook OOS-law: IS champion ED1 (10/1.4/3/14, lowest IS-MDD, 21tr low-freq) = OOS-WORST = ONLY non-profitable +$9 + broke 1.49×; the IS-inferior ultra-short HIGHEST-freq ED4 (45tr) is the PASS.** `search_eth_bollingerbreakout_daily_oos_champion_select.py` |
| ETHUSDT | Daily (exit modules, main fixed at **ED4** 2/3.0/2.0/11, IS) | **TREND-pattern HELP/neutral, 0 HURT** (sparse 45tr). Baseline NP=$1,609 MDD=−$500 45tr (A00 drift −0.009%). ⭐ **M5 QuantPass_PT_Exit PT_Base=0.503 = best: NP $2,090 (+29.92%) AND MDD improved −$463** — **M5 wins on DAILY (vs M6 on Hourly); M6 RescueTeam ≈0% on daily (sparse, never fires usefully; export truncated Length>280=0-trade).** M3 EXITBAR=19 +5.93%, M2 +3.85%; M1/M4 0%. `search_eth_bollingerbreakout_daily_exit_modules.py` |
| BTCUSDT | Daily | **R2 soft-ceiling (sparse 10-19tr, noisy, does NOT converge cleanly) — Obj-max BBLen=3 BBmult=3.5 ATRMult=4.0 ReentryBars=4 NP=$1,531 MDD=−$541 19tr (most robust); NP-max BBLen=4 BBmult=2.8 ATRMult=5.5 Re=2 NP=$1,647 10tr.** Weakest cell of the matrix; Daily<Hourly −27%. ⚠️ top-Obj rows are 1-trade degenerate (ATRMult=12, excluded). $100K −98.5% unreachable. `search_btc_bollingerbreakout_daily{,2}.py` |
| BTCUSDT | Daily (champion OOS) | **✅ WINNER CD4 = strict OOS PASS (sparse, low-confidence): BBLen=4 BBmult=2.3 ATRMult=5.6 ReentryBars=3** — OOS **+$676**, full MDD −$765 = IS (HELD), full NP $2,287, 10→13tr. CD3 high-freq (30tr) also held MDD but OOS −$123. IS Obj-max CD1 (19tr) broke 1.68× + lost. ⚠️ 10-13tr → low confidence. `search_btc_bollingerbreakout_daily_oos_champion_select.py` |
| BNBUSDT | Daily | **R2 ceiling CONFIRMED — NP-max=Obj-max BBLen=2 BBmult=1.0 ATRMult=5.0 ReentryBars=12 NP=$25,576 MDD=−$6,839 Obj=95,647 11tr (8× byte-identical; BBmult=1.0 interior peak; BBLen=2 floor)**. ⚠️ R1 BBLen=3 gave $27,720 — Rule-5 data drift (same MDD, sparse 11-13tr). **STRONGEST CELL of the matrix; Daily > Hourly +20.6%** (BNB = the Daily-works coin, like its CT/HUNTER2). $100K −74.4% (matrix-closest). `search_bnb_bollingerbreakout_daily{,2}.py` |
| BNBUSDT | Daily (champion OOS) | **✅ WINNER BD2 = strict OOS PASS: BBLen=2 BBmult=1.0 ATRMult=5.0 ReentryBars=12** — OOS **+$5,142** (highest), full MDD −$6,839 = IS (HELD), **full NP $30,718 = matrix max**. 2 PASS (BD1 BBLen=3 also +$5,067). **🔄 BNB thick-edge EXCEPTION to the OOS-law: here the SPARSE low-freq champions (11-13tr) PASS while the high-freq BD3 (28tr) BROKE + lost OOS** — opposite of thin-edge coins; BNB's daily edge is thick enough that the wide-trail sparse champion is itself OOS-robust. `search_bnb_bollingerbreakout_daily_oos_champion_select.py` |

### BollingerBreakout 3-coin × 2-TF matrix (COMPLETE — IS + OOS all cells)
| | Hourly IS / OOS | Daily IS / OOS |
|---|---|---|
| BNB | $21,210 / ✅ 2 PASS (BO3) | **$25,576 / ✅ PASS (BD2; full NP $30,718 = matrix max)** |
| ETH | $5,267 / ❌ all-broke | $2,291 / ✅ PASS (ED4) |
| BTC | $2,236 / ❌ all-broke | ~$1,531-1,647 / ✅ PASS (CD4, sparse) |

**🎯 MATRIX VERDICT: 5 OOS-PASS configs = BNB-H(2), BNB-D, ETH-D, BTC-D. ALL 3 coins' DAILY pass; Hourly only on BNB → DAILY is broadly more OOS-robust (wide ATR trail captures big moves, less high-freq noise). BNB = only coin passing both TFs (thick edge) AND the only Daily>Hourly (strongest cell $25.6K IS / $30.7K full). OOS-law refinement: "IS sparse/low-MDD champ = OOS-worst, higher-freq generalizes" holds on THIN-edge cells (BTC/ETH); on THICK-edge BNB Daily it REVERSES — the sparse wide-trail champion itself holds OOS and the high-freq breaks. Exit-module winner flips by TF: M6 RescueTeamExit (Hourly all coins) → M5 QuantPass_PT_Exit (Daily; M6 too sparse to fire).**

**🎯 NEW PATTERN: on the thin-edge coins (BTC/ETH) the DAILY BollingerBreakout is OOS-robust (PASS) while the HOURLY is OOS-fragile (all-broke) — fewer trades but the wide ATR trail captures big moves with less high-freq noise erosion; BNB's edge is thick enough that Hourly already passes.** Exit-module winner flips by TF: **M6 RescueTeamExit on Hourly (all 3 coins), M5 QuantPass_PT_Exit on Daily** (M6 too sparse to fire on daily). OOS-PASS configs: BNB-H, ETH-D, BTC-D.

**17 self-authored strategies (clean-IS BNB Hourly), all confirmed: Donchian v2 $24.4K (OOS PASS DN3 +$5,704) ≈ ADXtrend $23.9K (OOS fail) > VolatilityBreakout $22.0K (OOS fail) ≈ CloseChannelBreakout $22.0K (✅ Obj 98K #1; OOS CC3 PASS / CC1 +$14,496 broke $43) > BollingerBreakout $21.2K (✅ OOS PASS BO3 +$9,687; Obj 95K) > HeikinAshi $20.7K (OOS fail) > BBSqueeze $16.1K (OOS fail) > DonchianAsymV2 $15.0K > Keltner $10.1K > RSIPullback ~$8K > DonchianAsym $5.7K > ROCmomentum $5.6K > ParabolicSAR $4.6K (broken) ; TurtleChannel / FractalBreakout all-negative ; ChannelClose degenerate.**
**🎯 CloseChannelBreakout (the FIXED ChannelClose: CLOSE channel + intrabar STOP, not High/Low + market) = the highest risk-adjusted (Obj 98K) AND highest de-facto OOS profit (+$14,496) of all self-authored — close-channel is a 3rd valid breakout reference (wick-immune → lowest MDD). Exit-modules push it to Obj ~207K (M6 RescueTeam +38% NP & MDD↓, or M5 PT_Exit +46% NP).**
**FINAL LAWS: (1) TWO self-authored strategies are both IS-strong AND OOS-robust (strict PASS): Donchian v2 AND BollingerBreakout — both are rolling/non-lagged breakout + WIDE ATR chandelier trail + post-exit cooldown. (2) The breakout REFERENCE matters: rolling extreme (Donchian) and statistical band (Bollinger) BOTH work — Bollinger band is best risk-adjusted (lowest MDD); but a LAGGED pivot (Fractal, all-negative) and a CLOSE-confirmed MARKET entry (ChannelClose, degenerate) BOTH break it → the edge needs the rolling/non-lagged level + intrabar STOP fill at the edge. (3) The EXIT is decisive — wide ATR chandelier trail >> channel exit (Turtle) / SAR (broken); but OOS a too-tight-IS-MDD wide trail can amplify tail losses (VB1). (4) Entry FILTERS fail OOS (ADX/squeeze/RSI/trend/vol-buffer → floor), but the ReentryBars post-exit COOLDOWN is the ONE entry-side lever that genuinely helps AND is kept >0 (DonchianAsymV2, BollingerBreakout). (5) Asymmetry (long≠short channel) is NECESSARY-not-sufficient on the up-trending asset (suppresses bad shorts) — only works WITH the cooldown. (6) IS strength != OOS robustness — the IS NP/Obj champion (lowest IS-MDD) is repeatedly the OOS-WORST / breaks MDD (HA1, VB1, BO1); IS-inferior higher-freq / tighter-band / wider-MDD regimes generalize best OOS (HA3, VB3, BO3).**

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
- **TRUE OOS isolation via chart Data Range** — MC64 ignores the signal Begin-date; only the CHART's loaded data range restricts the backtest. `mc.set_instrument_data_range(conn, from, to)` sets Format Instruments → Settings → Data Range (From-To radio + both date pickers) to trim/expand the chart so an IS pass (end 2026/01) ≠ FULL pass (end 2026/06), giving OOS = NP_full − NP_is. **Critical**: `DTM_SETSYSTEMTIME` sets the picker value but does NOT fire `DTN_DATETIMECHANGE`, so OK won't apply it (3 silent failures, IS==FULL) — after the DTM set the code nudges the picker (click + Right + Up/Down, net-zero) to fire the notification. Verify first with `--probe-instrument` (sets FULL range, reopens dialog, reads pickers back) + a chart screenshot. `mc.read_instrument_data_range` reads the current range. This supersedes the earlier "OOS-only not isolable" note. **⚠️ Settings-tab silent-fail (fixed ae19bdf, 2026-06-23):** when the data source shows "not connected" (e.g. Binance on a fresh workspace), the Format Instrument dialog opens on the **Lookup/Add-Symbol tab** which has NO date pickers; the old code clicked the Settings tab once, found 0 pickers, and **still clicked OK — silently applying nothing** (both passes ran on the loaded range → IS==FULL, and IS optimizations got silently OOS-contaminated). Fix: `set_instrument_data_range` now retries the Settings-tab click (≤4×) until ≥2 `SysDateTimePick32` pickers are present, else Cancels and **raises** instead of applying nothing. Log shows `Settings tab reached on attempt N (3 date pickers)` on success. Symptom of a contaminated IS run: NP collapses (e.g. RSIPullback $20.8K→$8K) once the trim actually applies. The RSIPullback/Keltner IS rounds run before this fix are OOS-contaminated.

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
