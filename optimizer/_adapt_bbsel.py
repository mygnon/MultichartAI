import io, ast, re
p = 'search_bnb_bollingerbreakout_hourly_oos_champion_select.py'
s = io.open(p, encoding='utf-8').read()

# ---- header / names ----
s = s.replace('search_bnb_volatilitybreakout_hourly_oos_champion_select.py', 'search_bnb_bollingerbreakout_hourly_oos_champion_select.py')
s = s.replace('SFJ_VolatilityBreakout_crypto', 'SFJ_BollingerBreakout_crypto')
s = s.replace('results/bnb_volatilitybreakout_hourly{,2,3}_search/', 'results/bnb_bollingerbreakout_hourly{,2}_search/')
s = s.replace('20260622_SFJ_VolatilityBreakout_crypto_AI.wsp', '20260622_SFJ_BollingerBreakout_crypto_AI.wsp')
s = s.replace(r'results\bnb_volatilitybreakout_hourly_oos_champion_select_search', r'results\bnb_bollingerbreakout_hourly_oos_champion_select_search')
s = s.replace('PREFIX      = "BNBVBSEL_"', 'PREFIX      = "BNBBOSEL_"')
s = s.replace('search_bnb_volatilitybreakout_hourly_oos_champion_select_{int', 'search_bnb_bollingerbreakout_hourly_oos_champion_select_{int')
s = s.replace('final_params_bnb_volatilitybreakout_hourly_oos_champion_select.json', 'final_params_bnb_bollingerbreakout_hourly_oos_champion_select.json')
s = s.replace('--candidate VB1', '--candidate BO1')
s = s.replace('--candidate", metavar="VBn"', '--candidate", metavar="BOn"')
s = s.replace('3 params ATRLen, EntryMult, TrailMult; volatility-breakout STOP entry + ATR chandelier trailing exit,',
              '4 params BBLen, BBmult, ATRMult, ReentryBars; pure Bollinger-band breakout STOP + cooldown + ATR chandelier exit,')
s = s.replace('R3 ceiling $22,046 = #3 self-authored BNB strategy:\nATRLen=45, TrailMult=16.5 (wide trail = the decisive lever), EntryMult driven to ~0 floor\n(the volatility-buffer novelty is rejected; strategy = near-immediate breakout + wide ATR trail).',
              'R2 ceiling $21,210 = #2 self-authored by Obj (95,486): BBLen=13 BBmult=1.55 ATRMult=7 ReentryBars=13;\nstatistical band beats Donchian channel on risk-adjusted return (lowest MDD -$4,711); cooldown engaged.')
s = s.replace('BNB Hourly SFJ_BollingerBreakout_crypto champions (R1-R3, results/bnb_bollingerbreakout_hourly{,2,3}_search/).',
              'BNB Hourly SFJ_BollingerBreakout_crypto champions (R1-R2, results/bnb_bollingerbreakout_hourly{,2}_search/).')

# ---- STEP/LO/HI: 4 params ----
s = s.replace('STEP = {"ATRLen": 1.0, "EntryMult": 0.025, "TrailMult": 0.25}',
              'STEP = {"BBLen": 1.0, "BBmult": 0.05, "ATRMult": 0.25, "ReentryBars": 1.0}')
s = s.replace('LO   = {"ATRLen": 2.0, "EntryMult": 0.01, "TrailMult": 0.5}',
              'LO   = {"BBLen": 2.0, "BBmult": 0.25, "ATRMult": 0.5, "ReentryBars": 0.0}')
s = s.replace('HI   = {"ATRLen": 200.0, "EntryMult": 5.0, "TrailMult": 20.0}',
              'HI   = {"BBLen": 200.0, "BBmult": 6.0, "ATRMult": 20.0, "ReentryBars": 300.0}')

# ---- CANDIDATES (4 regime-diverse, 4-param) ----
old_c = re.search(r'CANDIDATES = \[.*?\]\n', s, re.S).group(0)
new_c = '''CANDIDATES = [
    # id, regime, BBLen, BBmult, ATRMult, ReentryBars, prior IS NP, prior IS MDD (intraday)
    ("BO1", "champion (Obj-max) ***", 13.0, 1.55, 7.0, 13.0, 21210.0, -4711.0),  # R2 NP/Obj-max, 217tr, lowest MDD
    ("BO2", "short-BBLen",            8.0,  1.5,  7.0, 14.0, 18628.0, -4814.0),  # 233tr, distinct shorter period
    ("BO3", "low-BBmult",             9.0,  1.05, 7.0, 17.0, 20767.0, -5888.0),  # 218tr, tight band distinct regime
    ("BO4", "ultra-short hi-freq",    3.0,  1.5,  7.0, 14.0, 12696.0, -6841.0),  # 263tr, high-freq generalizer
]
'''
s = s.replace(old_c, new_c)

# ---- _cfg signature + params ----
s = s.replace('def _cfg(cid, period, atrlen, entry, mult) -> StrategyConfig:',
              'def _cfg(cid, period, bblen, bbmult, mult, reentry) -> StrategyConfig:')
s = s.replace('params=[_axis("ATRLen", atrlen), _axis("EntryMult", entry), _axis("TrailMult", mult)],',
              'params=[_axis("BBLen", bblen), _axis("BBmult", bbmult), _axis("ATRMult", mult), _axis("ReentryBars", reentry)],')

# ---- _pick ----
s = s.replace('def _pick(df, atrlen, entry, mult):',
              'def _pick(df, bblen, bbmult, mult, reentry):')
s = s.replace('    for nm, v in (("ATRLen", atrlen), ("EntryMult", entry), ("TrailMult", mult)):',
              '    for nm, v in (("BBLen", bblen), ("BBmult", bbmult), ("ATRMult", mult), ("ReentryBars", reentry)):')

# ---- slices [2:5] -> [2:6] ----
s = s.replace('_cfg("VB1", "is", *CANDIDATES[0][2:5])', '_cfg("BO1", "is", *CANDIDATES[0][2:6])')
s = s.replace('_cfg(CANDIDATES[0][0], period, *CANDIDATES[0][2:5])', '_cfg(CANDIDATES[0][0], period, *CANDIDATES[0][2:6])')

# ---- run loop unpacking + calls (both occurrences) ----
s = s.replace('for (cid, regime, atrlen, entry, mult, pnp, pmdd) in CANDIDATES:',
              'for (cid, regime, bblen, bbmult, mult, reentry, pnp, pmdd) in CANDIDATES:')
s = s.replace('cfg = _cfg(cid, period, atrlen, entry, mult)',
              'cfg = _cfg(cid, period, bblen, bbmult, mult, reentry)')
s = s.replace('row = _pick(df, atrlen, entry, mult) if (df is not None and not df.empty) else None',
              'row = _pick(df, bblen, bbmult, mult, reentry) if (df is not None and not df.empty) else None')
s = s.replace('"params": {"ATRLen": atrlen, "EntryMult": entry, "TrailMult": mult},',
              '"params": {"BBLen": bblen, "BBmult": bbmult, "ATRMult": mult, "ReentryBars": reentry},', 1)
# second occurrence (candidate-build section)
s = s.replace('"params": {"ATRLen": atrlen, "EntryMult": entry, "TrailMult": mult},',
              '"params": {"BBLen": bblen, "BBmult": bbmult, "ATRMult": mult, "ReentryBars": reentry},')

# ---- log line ----
s = s.replace('log.info("--- [%s] %s %s ATRLen=%g EntryMult=%g TrailMult=%g [%s] ---",',
              'log.info("--- [%s] %s %s BBLen=%g BBmult=%g ATRMult=%g Re=%g [%s] ---",')
s = s.replace('period, cid, regime, atrlen, entry, mult, datetime.now().strftime("%H:%M:%S"))',
              'period, cid, regime, bblen, bbmult, mult, reentry, datetime.now().strftime("%H:%M:%S"))')

# ---- summary header ----
s = s.replace('  BNB Hourly VolatilityBreakout', '  BNB Hourly BollingerBreakout')
s = s.replace('description="BNB Hourly VolatilityBreakout OOS champion selection',
              'description="BNB Hourly BollingerBreakout OOS champion selection')

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
ast.parse(s)
print('SYNTAX OK | residual ATRLen/EntryMult/TrailMult/VolatilityBreakout/VB:',
      s.count('ATRLen')+s.count('EntryMult')+s.count('TrailMult')+s.count('VolatilityBreakout')+s.count('"VB')+s.count('atrlen')+s.count('entry,'))
