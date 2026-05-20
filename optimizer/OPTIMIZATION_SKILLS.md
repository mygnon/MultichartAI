# Breakout 策略最佳化技術手冊

本手冊涵蓋 `_2021Basic_Break_NQ` 策略在四種商品/時間框架組合的所有搜尋結果與技術規則。

---

## 搜尋概覽

| 搜尋軌道 | 商品 | 時間框架 | 目標 NP | 目前最佳 NP | 狀態 | 腳本 |
|----------|------|---------|---------|------------|------|------|
| TWF Daily | TWF.TXF HOT | Daily (1440m) | > 6,000,000 TWD | 4,089,800 TWD | 未達標 (-32%) | `search_daily_target6.py` |
| TWF Hourly | TWF.TXF HOT | Hourly (60m) | > 6,000,000 TWD | **6,043,200 TWD** | **✅ 達標** | `search_hourly_target2.py` |
| NQ Hourly | CME.NQ HOT | Hourly (60m) | > 700,000 USD | 656,575 USD | 未達標 (-6.2%) | `search_nq_hourly3.py` |
| NQ Daily | CME.NQ HOT | Daily (1440m) | > 700,000 USD | 350,220 USD | 未達標 (-50%) | `search_nq_daily3.py` |

> **貨幣換算**：NQ 以 USD 計價；TWF 以 TWD 計價；匯率約 32 TWD/USD。  
> 700K USD ≈ 22.4M TWD（遠超 TWF 6M TWD 目標，NQ 700K 是非常高的門檻）。

---

## Workspace

```
C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp
```

所有搜尋腳本共用此 workspace。Insample：`2019/01/01 – 2026/01/01`。

---

## 目標函數

```
Objective = NP² / |MaxDrawdown|   (限 NP > 0 且 MaxDrawdown < 0)
```

實作位置：`plateau.py → compute_objective()`

---

## 四個參數說明

| 參數 | 名稱 | 意義 | 全域範圍 |
|------|------|------|---------|
| LE | Long Entry lookback | 多頭突破回顧 K 棒數 | 1 – 200 |
| SE | Short Entry lookback | 空頭突破（或時間出場）回顧 K 棒數 | 1 – 400 |
| STP | Stop | 停損距離 | 0.05 – 200 |
| LMT | Limit | 獲利目標距離 | 0.05 – 200 |

---

## 最佳結果歷程

### TWF Hourly — **目標達成** ✅

| Round | 腳本 | LE | SE | STP | LMT | NP (TWD) | MDD | Objective | Trades |
|-------|------|----|----|-----|-----|---------|-----|-----------|--------|
| R1 A03 | search_hourly_target.py | 5 | 75 | 5 | 32 | 5,210,000 | — | — | — |
| **R2 A11** | **search_hourly_target2.py** | **3** | **76** | **4** | **32** | **6,043,200** | -1,087,800 | 33,572,593 | **886** |

結果檔：`results/hourly_target2_search/final_params_hourly_target2.json`

---

### TWF Daily — 未達標（目前最佳 4,089,800 TWD，缺口 -32%）

| Round | 腳本 | LE | SE | STP | LMT | NP (TWD) | MDD | Objective | Trades |
|-------|------|----|----|-----|-----|---------|-----|-----------|--------|
| R5 | search_daily_target5.py | 5 | 50 | 2.2 | 17 | 3,779,400 | -767,600 | 18,608,474 | 26 |
| R6 A09 (最高NP) | search_daily_target6.py | 5 | 50 | 0.25 | 16 | **4,089,800** | -642,800 | 26,021,257 | 47 |
| **R6 A11 (最高Obj)** | search_daily_target6.py | **5** | **49** | **0.2** | **16** | 4,074,200 | -634,200 | **26,173,298** | **53** |

結果檔：`results/daily_target6_search/final_params_daily_target6.json`

**TWF Daily 參數規律（截至 R6）：**

| 方向 | 結論 |
|------|------|
| STP 0.2–0.5 | ✅ 最佳：交易次數多，MDD 反而小 |
| STP > 2.0 | ❌ NP 明顯偏低 |
| LMT 14–17 | ✅ 最佳區間 |
| LMT > 20 | ❌ NP 持續下降 |
| SE 45–55 | ✅ NP 最高 |
| SE 80–110 | ⚠️ MDD 較小但 NP 卡在 3.5M |
| LE 5–7 | ✅ 穩定 |
| LE < 4 | ❌ MDD 惡化 |

---

### NQ Hourly — 未達標 700K（目前最佳 656,575 USD，缺口 -6.2%）

| Round | 腳本 | LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|-------|------|----|----|-----|-----|---------|-----|-----------|--------|
| R1 A11 (**600K 達標**) | search_nq_hourly.py | 1 | 11 | 5.0 | 8.0 | **621,220** | -80,975 | 4,765,845 | 5,779 |
| R2 A07 | search_nq_hourly2.py | 1 | 9 | 2.0 | 14.0 | **656,575** | -89,130 | 4,836,651 | 6,168 |
| R2 A02 (最高Obj) | search_nq_hourly2.py | 1 | 10 | 0.9 | 14.0 | 595,905 | **-59,855** | **5,932,717** | 6,504 |
| R3 A01–A11 | search_nq_hourly3.py | 1 | 9 | 2.0 | 14.0 | 656,575 | -89,130 | 4,836,651 | 6,168 |
| R3 A08 (折衷) | search_nq_hourly3.py | 1 | 9 | 1.5 | 14.0 | 655,835 | **-80,900** | **5,316,682** | 6,265 |

結果檔：`results/nq_hourly3_search/final_params_nq_hourly3.json`

**NQ Hourly 參數規律：**

| 方向 | 結論 |
|------|------|
| LE = 1 | ✅ 唯一有效的 LE，更長的 LE 無效 |
| SE = 9–11 | ✅ 最佳：超短回望，高頻訊號 |
| SE > 20 | ❌ NP 急降（高 SE 在 hourly NQ 不適用） |
| STP = 1.5–2 | ✅ 最佳 NP；STP=0.9 最高 Objective（但 NP 偏低） |
| STP < 1 | ⚠️ NP 下降但 MDD 大幅改善，Obj 更好 |
| LMT = 14 | ✅ 明顯優於 LMT=8（R1 最佳是 LMT=8，R2 換成 LMT=14 進步 +35K） |
| 每年交易次數 | ~880 次（6,168 trades / 7 年） |
| **NP 天花板** | **~656K — R2/R3 各 12 次嘗試都收斂到此** |

> **最高 Objective 推薦**（風險調整後更優）：LE=1, SE=10, STP=0.9, LMT=14, NP=595,905, MDD=-59,855, Obj=5,932,717

---

### NQ Daily — 未達標 700K（目前最佳 350,220 USD，缺口 -50%）

| Round | 腳本 | LE | SE | STP | LMT | NP (USD) | MDD | Objective | Trades |
|-------|------|----|----|-----|-----|---------|-----|-----------|--------|
| R1 A11 | search_nq_daily.py | 2 | 77 | 4.5 | 3.5 | 330,160 | -83,390 | 1,307,179 | 32 |
| R2 A02 | search_nq_daily2.py | 1 | 78 | 5.5 | 3.75 | 341,410 | -72,655 | 1,604,305 | 28 |
| **R3 A01** | search_nq_daily3.py | **1** | **78** | **5.5** | **3.9** | **350,220** | -72,655 | **1,688,171** | **28** |

結果檔：`results/nq_daily3_search/final_params_nq_daily3.json`

**NQ Daily 參數規律：**

| 方向 | 結論 |
|------|------|
| SE = 75–85 | ✅ 唯一有效核心區間（R1 ~ R3 所有嘗試都收斂到這裡） |
| SE < 30 | ❌ NP 最差（R1 A02 low_se: NP=194K） |
| SE > 150 | ⚠️ 尚可但遜於 SE=78（R3 A03: NP=318K） |
| LMT = 3.5–4.0 | ✅ 最佳，精確值 LMT=3.9 在 R3 確認 |
| LMT > 8 | ❌ NP 下降（LMT=8: NP=309K < LMT=3.9: NP=350K） |
| LMT < 1 | ❌ NP 下降（85 trades 但 NP=298K） |
| STP = 5–7 | ✅ 有效範圍 |
| **STP > 5–6** | **⚠️ 死參數！** STP=8 與 STP=25 結果完全相同 → 停損從未被觸及 |
| LE = 1–2 | ✅ 最佳；LE=4 也可但略遜 |
| 每年交易次數 | ~4 次（28 trades / 7 年），非常低頻 |
| **NP 天花板** | **~350K — 三輪 36 次嘗試，進展停滯（R1→R3 只從 330K→350K）** |

> **重要發現：STP 是死參數**（R2 A05 vs A06：STP=8 和 STP=25 結果完全相同）。出場機制完全由 SE 計時或 LMT 目標控制，停損價格永遠不會被觸及。

---

## 跨商品比較

| 維度 | TWF Daily | TWF Hourly | NQ Hourly | NQ Daily |
|------|-----------|------------|-----------|----------|
| 最佳 LE | 5 | 3 | **1** | 1–2 |
| 最佳 SE | 49 | 76 | **9–11** | **75–85** |
| 最佳 STP | 0.2 | 4 | 1.5–2 | 5–6（死參數） |
| 最佳 LMT | 16 | 32 | 14 | **3.9** |
| 每年交易次數 | ~8 次 | ~127 次 | ~880 次 | ~4 次 |
| NP 達標率 | 未達標 | ✅ 達標 | 未達標 | 未達標 |

**關鍵洞察：**
- NQ Hourly 與 TWF Hourly 行為完全不同：NQ 需要極短 SE（9 vs 76），極低 LE（1 vs 3）
- NQ Daily 與 TWF Daily 也截然不同：SE=78 vs SE=49，LMT=3.9 vs LMT=16
- TWF Hourly 是迄今唯一達標的軌道（NP=6,043,200 TWD）
- NQ Daily 700K 目標難度極高：每年只有 4 筆交易，NP 天花板約 350K，距目標差 50%

---

## 關鍵技術規則（血淚教訓）

### 規則 1：所有 4 個參數在每次嘗試都必須變動

**原因**：MC64 匯出 CSV 時，若某參數固定（start == stop），MCReport 每列塞入多組數據，pandas 欄位錯位，出現 LE=50 SE=0.75 STP=50 的假結果。

```python
def _safe(t: tuple) -> tuple:
    s, e, step = t
    if s == e:
        return (max(LO, s - step), min(HI, s + step), step)
    return t
le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)
```

### 規則 2：每次嘗試 ≤ 5000 combos；縮減用遞進收縮

```python
# 錯誤：固定值不會收斂 → 無限迴圈
while _c.total_runs() > 5000:
    _se = zoom(best_se, 8, 2, ...)   # 每次相同！

# 正確：每次縮小半徑
for r_se, r_stp, r_lmt in [(12,1.0,4), (8,0.8,3), (6,0.6,2), (4,0.4,1)]:
    _se  = zoom(best_se,  r_se,  2, SE_LO, SE_HI)
    _stp = zoom(best_stp, r_stp, 0.2, STP_LO, STP_HI)
    _lmt = zoom(best_lmt, r_lmt, 0.5, LMT_LO, LMT_HI)
    _c = _cfg(_n, _le, _se, _stp, _lmt)
    if _c.total_runs() <= 5000:
        break
```

### 規則 3：讀取 CSV 後必須驗證欄位範圍

```python
def _validate_df(df: pd.DataFrame, cfg: StrategyConfig) -> bool:
    for p in cfg.params:
        if p.name not in df.columns:
            continue
        lo = min(p.start, p.stop) - abs(p.step) * 2
        hi = max(p.start, p.stop) + abs(p.step) * 2
        col = pd.to_numeric(df[p.name], errors="coerce")
        if not col.between(lo, hi).all():
            return False   # 捨棄此次結果
    return True
```

### 規則 4：腳本必須以 Administrator 權限執行

MC64 以提升權限執行，跨權限 UI 自動化受 UIPI 封鎖。所有搜尋腳本包含 `_auto_elevate()` 自動提權：

```python
if not args.from_csv and not _is_admin():
    _auto_elevate()   # ShellExecuteW "runas"
```

### 規則 5：不能用 `Application(process=pid)` 連接 MC64

必須用 `Desktop(backend="uia")` + ctypes `EnumWindows`。

### 規則 6：champion() 的 zoom seed 使用 NP-max（非 Obj-max）

追標時 zoom 要往最高 NP 方向縮放，不是往最高 Objective 縮放（Objective 可能因 MDD 小而高但 NP 不夠）：

```python
# 追標模式：NP-max 作為 zoom seed
best = pos.loc[pos["NetProfit"].idxmax()]
```

---

## 腳本架構模板

```python
# 標準流程
cfg = _cfg(name, _safe(le), _safe(se), _safe(stp), _safe(lmt))
# assert cfg.total_runs() <= 5000
df  = run_or_load(name, cfg, conn, from_csv)
if _validate_df(df, cfg):
    le, se, stp, lmt, obj, np_, mdd, tr, met = champion(df, ...)
    attempt_log.append(_entry(...))
save_json(best_entry, best_np_entry, attempt_log, target_met)
```

---

## 輔助函數

```python
def zoom(center, radius, step, lo, hi):
    start = max(lo, round(round((center - radius) / step) * step, 8))
    stop  = min(hi, round(round((center + radius) / step) * step, 8))
    if stop <= start:
        stop = start + step
    return (start, stop, step)

def n_vals(t):
    s, e, step = t
    return max(1, round((e - s) / step) + 1)
```

---

## 結果目錄結構

```
results/
  hourly_target2_search/           # TWF Hourly ✅ 達標
    BHT2_*_raw.csv
    final_params_hourly_target2.json

  daily_target6_search/            # TWF Daily 未達標
    BD6_*_raw.csv
    final_params_daily_target6.json

  nq_hourly3_search/               # NQ Hourly 未達標 700K
    NQH3_*_raw.csv
    final_params_nq_hourly3.json

  nq_daily3_search/                # NQ Daily 未達標 700K
    NQD3_*_raw.csv
    final_params_nq_daily3.json
```

---

## 常用指令

```powershell
cd C:\Users\Tim\MultichartAI\optimizer

# TWF Daily（需 MC64 開啟）
python search_daily_target6.py
python search_daily_target6.py --from-csv      # 只重新分析

# TWF Hourly（已達標，備用）
python search_hourly_target2.py --from-csv

# NQ Hourly（700K 目標，缺口 -6.2%）
python search_nq_hourly3.py
python search_nq_hourly3.py --from-csv

# NQ Daily（700K 目標，缺口 -50%）
python search_nq_daily3.py
python search_nq_daily3.py --from-csv

# 從第 N 個 attempt 開始
python search_nq_daily3.py --attempt 6
```
