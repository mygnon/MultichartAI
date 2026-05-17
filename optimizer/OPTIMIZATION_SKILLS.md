# Breakout Daily 最佳化技術手冊

## 策略基本資訊

| 項目 | 內容 |
|------|------|
| 策略名稱 | `_2021Basic_Break_NQ` |
| 商品 | TWF.TXF HOT（台灣期貨） |
| 時間框架 | Daily (1440 min) |
| 樣本內 | 2019/01/01 – 2026/01/01 |
| Workspace | `C:\Users\Tim\Downloads\Multichartx86\Tim\20260508_SFJ_BASIC_BREAK_AI.wsp` |

---

## 目標函數

```
Objective = NP² / |MaxDrawdown|   (限 NP > 0 且 MaxDrawdown < 0)
```

- **主目標**：找到 NP > 6,000,000 且 Objective 最高的參數
- **備選**：若未達 6M，取 NP 最高的參數
- 實作位置：`plateau.py → compute_objective()`

---

## 四個參數說明

| 參數 | 名稱 | 意義 | 全域範圍 |
|------|------|------|---------|
| LE | Long Entry lookback | 多頭突破回顧 K 棒數 | 1 – 200 |
| SE | Short Entry lookback | 空頭突破回顧 K 棒數 | 1 – 400 |
| STP | Stop ATR multiplier | 停損距離（倍 ATR） | 0.05 – 20 |
| LMT | Limit ATR multiplier | 獲利目標距離（倍 ATR） | 2 – 100 |

### 已探索的規律（截至 Round-6，2026-05-17）

| 方向 | 結論 |
|------|------|
| LMT > 20 | ❌ NP 持續下降，不值得繼續探 |
| LMT 14–17 | ✅ 最佳區間 |
| STP 0.2–0.5 | ✅ 最有潛力：交易次數多、MDD 反而更小 |
| STP > 2.0 | ❌ NP 偏低 |
| SE 80–110 | ⚠️ MDD 較小但 NP 卡在 3.5M |
| SE 45–55 | ✅ NP 最高區間 |
| LE 1–4 | ❌ MDD 變大，NP 下降 |
| LE 5–7 | ✅ 穩定區間 |

---

## 目前最佳結果（歷次搜尋）

| 版次 | LE | SE | STP | LMT | NP | MDD | Objective | Trades |
|------|-----|-----|------|------|-----------|-----------|------------|--------|
| Round-1（2026-05-15 舊資料） | 5 | 50 | 2.1 | 17 | 4,908,600 | -769,200 | 31,389,205 | — |
| Round-5 最佳 | 5 | 50 | 2.2 | 17 | 3,779,400 | -767,600 | 18,608,474 | 26 |
| Round-6 A09 最高NP | 5 | 50 | 0.25 | 16 | **4,089,800** | -642,800 | 26,021,257 | 47 |
| **Round-6 A11 最高Obj** | **5** | **49** | **0.2** | **16** | 4,074,200 | -634,200 | **26,173,298** | **53** |

> **注意**：Round-1 與後續差異是因為 2026-05-15→16 之間價格資料有更新，非程式錯誤。

---

## 關鍵規則（血淚教訓）

### 規則 1：所有 4 個參數在每次嘗試都必須變動

**原因**：MC64 匯出 CSV 時，若某個參數固定（start == stop），MCReport 會每列塞入多組數據，導致 pandas 讀取時欄位錯位，LE=50 SE=0.75 STP=50 之類的假結果。

```python
# 錯誤示範（產生欄位錯位）
ParamAxis("LE", 5, 5, 1)   # start == stop → 只有 1 個值

# 正確做法：強制每個參數有 ≥2 個值
def _safe(t: tuple) -> tuple:
    s, e, step = t
    if s == e:
        return (max(LO, s - step), min(HI, s + step), step)
    return t
le, se, stp, lmt = _safe(le), _safe(se), _safe(stp), _safe(lmt)
```

### 規則 2：每次嘗試 ≤ 5000 combos

```python
combos = n_LE × n_SE × n_STP × n_LMT
assert combos <= 5000
```

**縮減 combos 的 while loop 必須用遞進縮小**，否則會無限迴圈：

```python
# 錯誤：固定值不會收斂
while _c.total_runs() > 5000:
    _se = zoom(best_se, 8, 2, ...)   # 每次相同 → 無限迴圈！
    _c = _cfg(...)

# 正確：每次縮小半徑
for radius in [(12, 1.0, 4), (8, 0.8, 3), (6, 0.6, 2), (4, 0.4, 1)]:
    _se  = zoom(best_se,  radius[0], 2, ...)
    _stp = zoom(best_stp, radius[1], 0.2, ...)
    _lmt = zoom(best_lmt, radius[2], 0.5, ...)
    _c = _cfg(...)
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
            log.warning("INVALID: %s 欄位超出預期範圍，捨棄此次結果", p.name)
            return False
    return True
```

### 規則 4：腳本必須以 Administrator 權限執行

MC64 本身以提升權限執行，跨權限 UI 自動化受 UIPI 封鎖。

```python
def _is_admin() -> bool:
    return bool(ctypes.windll.shell32.IsUserAnAdmin())

def _auto_elevate() -> None:
    script  = str(Path(__file__).resolve())
    workdir = str(Path(__file__).resolve().parent)
    extra   = [a for a in sys.argv[1:] if a != "--_elevated"]
    all_args = f'"{script}" ' + " ".join(extra) + " --_elevated"
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, all_args, workdir, 1)
    sys.exit(0)

# main() 開頭加這段
if not args.from_csv and not _is_admin():
    _auto_elevate()
```

### 規則 5：不能用 Application(process=pid) 連接 MC64

pywinauto 的 process-scoped spec 在跨權限情況下失敗。必須用 `Desktop(backend="uia")` + ctypes EnumWindows。

---

## 腳本架構模式

```python
# 每個 attempt 的標準流程
def run_attempt(name, le, se, stp, lmt, conn, from_csv):
    cfg = _cfg(name,
               _safe((le_start, le_stop, le_step)),
               _safe((se_start, se_stop, se_step)),
               _safe((stp_start, stp_stop, stp_step)),
               _safe((lmt_start, lmt_stop, lmt_step)))

    assert cfg.total_runs() <= 5000, f"{name}: {cfg.total_runs()} combos 超過 5000！"

    df = run_or_load(name, cfg, conn, from_csv)  # 有 CSV 就讀，沒有就跑 MC64

    if not _validate_df(df, cfg):
        # 記錄空結果，繼續下一個 attempt
        return

    le, se, stp, lmt, obj, np_, mdd, trades, met = champion(df, ...)
    attempt_log.append(_entry(...))
    save_json(best_entry, attempt_log, target_met)
```

---

## 輔助函數

### zoom — 以中心點生成範圍

```python
def zoom(center: float, radius: float, step: float,
         lo: float, hi: float) -> tuple:
    start = max(lo, round(round((center - radius) / step) * step, 8))
    stop  = min(hi, round(round((center + radius) / step) * step, 8))
    if stop <= start:
        stop = start + step
    return (start, stop, step)
```

### n_vals — 計算該範圍的值數量

```python
def n_vals(t: tuple) -> int:
    s, e, step = t
    return max(1, round((e - s) / step) + 1)
```

### champion — 從 DataFrame 選出最佳行

```python
def champion(df, fb_le, fb_se, fb_stp, fb_lmt):
    df["Objective"] = compute_objective(df)   # NP²/|MDD|
    above = df[df["NetProfit"] > TARGET_NP]
    if not above.empty:
        best = above.loc[above["Objective"].idxmax()]
        return ..., True   # 目標達成
    pos = df[df["Objective"] > 0]
    if pos.empty:
        return fb_le, fb_se, fb_stp, fb_lmt, 0, 0, 0, 0, False
    best = pos.loc[pos["Objective"].idxmax()]
    return ..., False
```

---

## 搜尋策略建議（Round-7 起）

### 優先探索低 STP 精細區

```
LE  : 4–6       step 1      →  3 vals
SE  : 45–55     step 1      → 11 vals
STP : 0.05–0.4  step 0.05   →  8 vals
LMT : 14–18     step 0.5    →  9 vals
Total: 3 × 11 × 8 × 9 = 2,376 combos ✓
```

### 其他值得探索的區域

1. **SE = 80–120 + 低 STP**：A08 的 SE=110 有小 MDD，加上低 STP 可能有更多獲利
2. **LE = 3–8 + SE = 45–60 + STP = 0.1–0.5 wide sweep**：比 Round-6 A09 再精細

### 不需要再探的區域

- LMT > 25（NP 明確下降）
- LE < 4（MDD 惡化）
- STP > 3（NP 偏低）
- 4D 全面掃描大範圍（浪費嘗試次數，不如集中精細）

---

## 結果目錄結構

```
results/
  daily_target6_search/
    BD6_01_lmt_hi_profile_raw.csv
    BD6_02_wide_se_raw.csv
    ...
    BD6_12_boundary_raw.csv
    final_params_daily_target6.json    ← 最終結果
    search_daily_target6_*.log
```

## 常用指令

```powershell
# 執行搜尋（會自動提升權限，MC64 必須已開啟）
cd C:\Users\Tim\MultichartAI\optimizer
python search_daily_target6.py

# 只重新分析已有的 CSV（不需要 MC64）
python search_daily_target6.py --from-csv

# 從第 N 個 attempt 開始（跳過前面已完成的）
python search_daily_target6.py --attempt 6
```
