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
| BNBUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **WINNER BX4: X=70 LY=5.5 SY=4.5** (asym mid-X high-freq 166tr) — ONLY strict PASS (full MDD −$7,625 = IS, held) + OOS-profitable **+$3,024**. BX3 (short-X 57tr) highest OOS **+$8,743** but broke 1.05×. In-sample champ BX1 (long-X) OOS-fragile (−$2,140, broke). HIGH-freq OOS-robust (like BNB CT). Clean OOS (IS=prior drift ≠ FULL). `search_bnb_xtreme_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly | >100K | Ceiling $3,682 (−96.3%; R2=R3; X=796 LY=5.85 SY=3.35 long-X asym 46tr MDD=−$450 Obj=30,105; NP-max=Obj-max; X bounded both sides). Best crypto XtremeStop on BTC/ETH (lowest MDD). $100K unreachable. `search_eth_xtreme_hourly{,2,3}.py` |
| BTCUSDT | Hourly | >100K | Ceiling $2,606 (−97.4%; R1=R2; X=60 LY=12.75 SY=12.4 **symmetric high-pct** 13tr MDD=−$582 Obj=11,655; NP-max=Obj-max; X & % bounded). Distinct regime from BNB asym. `search_btc_xtreme_hourly{,2}.py` |
| BTCUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all broke, but CX1 (champion X=60 LY=12.75 SY=12.4) broke by only $2** (−582→−584, 1.003×) AND highest OOS **+$570** = de-facto winner. CX3 (long-X 69tr) OOS +$378 broke 1.16×. **High-freq CX2/CX4 (94/121tr) OOS-COLLAPSED** (−$1,175/−$1,272, MDD 2.3-2.9×) — OPPOSITE of BNB (low-freq robust on BTC). `search_btc_xtreme_hourly_oos_champion_select.py` |
| ETHUSDT | Hourly (champion OOS) | most OOS profit, no MDD break | **NO strict PASS — all 4 broke.** Best-available EX2 (long-X low-SY X=720 LY=3 SY=1, 134tr) only OOS-profitable +$439 (broke 1.95×). **In-sample champion EX1 (X=796) OOS-WORST: −$1,097, MDD blew 5.7× (−$450→−$2,583)** — low IS-MDD = overfit to calm. ETH XtremeStop OOS-fragile (no safe regime). `search_eth_xtreme_hourly_oos_champion_select.py` |

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
| BNBUSDT | Hourly | **R2 ceiling (soft).** Stable 6-conv EMALen=100 BandMult=0.75 ATRMult=5.5 NP=$10,062 MDD=−$6,155 Obj=16,451 397tr; zoom-best EMALen=106 BandMult=1.0 ATRMult=5.34 NP=$11,602 Obj=19,328. **Short/mid-EMA ALL-LOSING; only long-EMA (~100-106) works.** **Weaker than Donchian** (NP ~½, Obj ~⅕, MDD/NP 60% vs 19-26%) — High/Low channel beats EMA±ATR band on BNB |

`search_bnb_keltner_hourly{,2}.py`.

### `SFJ_RSIPullback_crypto` (trend-filtered RSI pullback, market entries, ATR trail, `_Crypto1MUSD`)

**Self-authored** (`Strategy/SFJ_RSIPullback_crypto.txt`). Workspace: `20260622_SFJ_RSIPullback_crypto_AI.wsp`. Params: TrendLen (MA filter, int), RSILen (int), RSIThresh (long cross level, int), ATRMult (trail, frac). Long: `Close>MA(TrendLen) AND RSI(RSILen) cross over RSIThresh`; short mirror (100−RSIThresh); ATR(14) chandelier trail; flat-only.

| Instrument | TF | Status |
|---|---|---|
| BNBUSDT | Hourly | **R2 — STRONGEST non-CT BNB strategy.** ⭐ Best (B11) TrendLen=7 RSILen=6 RSIThresh=55 ATRMult=9.02 NP=**$25,303** MDD=−$5,800 Obj=**110,393** 179tr (MDD/NP 22.9%); stable 3-conv plateau TrendLen=8 RSIThresh=54 ATRMult=9 NP=$21,025. R1→R2 Obj +50.6%. **"Pullback" thesis FAILED → degenerated to momentum + wide trail**: TrendLen→floor (BNB dislikes trend filters, like Osc/Donchian), RSIThresh~55 (momentum cross, NOT a deep dip — deep-dip Th worse), ATRMult~9 (wide trail does the work). NP highest of all non-CT BNB; Obj #2 behind CT |

`search_bnb_rsipullback_hourly{,2}.py`. **3 self-authored strategies span 2 archetypes: breakout (Donchian High/Low, Keltner EMA±ATR) vs pullback (RSIPullback). On BNB: RSIPullback (NP $25.3K) > Donchian ($24.4K) > Keltner ($11.6K); all beat the canned momentum/oscillator strategies (MACD03 $8K, Osc $15K) except CT.**

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
