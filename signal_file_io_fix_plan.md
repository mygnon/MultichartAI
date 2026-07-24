# PT 訊號檔案 I/O 錯誤修復規劃

## 問題描述

MultiCharts Portfolio Trader 執行 Automate Order Execution 一段時間後跳出錯誤並自動解除 AOE:

```
Source: DualAnchorBreakout_BTC_H1_v1 (BTCUSDT-60 Minute)
Message: Can't open/create file : "C:\oms\signals\DualAnchorBreakout_BTC_H1_v1.json.tmp"
```

## 根本原因分析

因果鏈:訊號程式碼寫入 `.tmp` 檔時 create 失敗 → EasyLanguage 拋出 runtime error → MC 的保護機制在訊號發生 runtime error 時自動關閉該策略的 AOE。

寫入失敗的可能原因(依可能性排序):

1. **檔案鎖定衝突(最可能)**:OMS(Python 端)輪詢讀取 json 時,恰好在 MC 要 create/rename `.tmp` 的瞬間持有檔案 handle,Windows 檔案鎖導致 create 失敗。策略數越多、輪詢越頻繁,碰撞機率越高,因此「跑一段時間後」才發生。
2. **防毒即時掃描**:Defender 或其他防毒對高頻寫入檔案短暫上鎖。
3. **同步/備份軟體**:監控該目錄的軟體同步時鎖檔。
4. **EasyLanguage 檔案 handle 洩漏**:開檔後未確實 close,跑久了 handle 耗盡。

## 修復方案

核心原則:**搬移到 ramdisk 只能降低碰撞機率,不能根治鎖定衝突;必須同時加上 retry 與讀寫規範**。這套系統只要錯一次就會解除 AOE 掉線,不能靠機率。

### 1. 路徑遷移:C:\oms\signals → Z:\oms\signals

- Z:\ 為 ramdisk,寫入延遲由毫秒級降至微秒級,鎖定時間窗口縮短數個數量級,碰撞機率大幅下降
- 免除幾百個策略高頻寫檔造成的 SSD 磨損與 I/O 排隊
- 保留 `\oms\signals\` 目錄結構,與現行邏輯一致,Phase 2 換 Python 引擎時路徑邏輯可沿用
- 所有引用此路徑的地方都要改:EasyLanguage 訊號碼、Python OMS 讀取端、任何設定檔

### 2. EasyLanguage 寫入端:加入 retry 機制

- 開檔/建檔失敗時不可直接讓 runtime error 拋出(會解除 AOE)
- 改為迴圈重試 3~5 次,每次間隔數十 ms,全部失敗才報錯
- 這是擋掉瞬間鎖定造成 AOE 解除的最關鍵修改

### 3. EasyLanguage 寫入端:唯一暫存檔名

- `.tmp` 檔名加上時間戳或亂數,例如 `DualAnchorBreakout_BTC_H1_v1.json.1721790000123.tmp`
- 寫完後 rename 成正式的 `.json` 檔名(atomic write 模式)
- 避免多次寫入或殘留檔搶同一個 `.tmp` 檔名
- 順帶檢查:寫檔用的檔案 handle 每次都必須確實 close,避免 handle 洩漏

### 4. Python OMS 讀取端規範

- glob 只匹配 `*.json`,絕對不能掃到 `*.tmp`(注意 `*.json*` 這類 pattern 會誤掃)
- 開檔讀取後立刻 close(用 `with open(...)`,不長時間持有 handle)
- 遇到 `PermissionError` / `OSError` 時跳過本輪,下次輪詢再讀,不可 crash 或重試佔住檔案

### 5. Ramdisk 相關注意事項(需在計畫中確認/處理)

- **重啟後 Z:\ 內容全部消失**:
  - 訊號檔為暫態資料,消失無妨
  - 但 OMS 重啟後的補單/對帳邏輯必須以「向 MC/交易所重新查詢實際部位」為準,不可依賴舊訊號檔存在
  - 任何需要持久化的狀態(策略上下架清單、MDD 追蹤水位等)必須放 C:\ 或定期落地,不可只存在 Z:\
- **ramdisk 軟體設定**:若有「定期備份到硬碟」功能(如 Primo Ramdisk),備份當下會鎖檔,對此目錄應關閉或大幅調低頻率
- **開機順序**:確保 Z:\ 在 MC/PT 啟動前已掛載,且 `Z:\oms\signals\` 目錄存在(可在開機腳本或 OMS 啟動時自動建立),否則 PT 開盤即報錯解除 AOE
- **防毒排除**:將 Z:\(或至少 Z:\oms)加入防毒即時掃描排除清單;原 C:\oms 若仍有持久化檔案也一併排除

## 驗收標準

1. PT + AOE 連續運行(例如 48 小時以上)不再出現 "Can't open/create file" 錯誤,AOE 不被解除
2. Python OMS 能正常讀到所有策略的最新訊號,無讀到半寫入(partial write)的 json
3. 模擬重開機:Z:\ 清空後,OMS 重啟能正確重建目錄、重新對帳,不因訊號檔消失而出錯
4. 手動製造鎖定(Python 端故意持有檔案 handle 數百 ms),EasyLanguage retry 能成功等到鎖釋放後寫入,不拋 runtime error

## 長期備案

若檔案輪詢在大規模(數百策略)下仍偶發不穩,將 MC ↔ OMS 通訊改為 named pipe 或本機 socket,徹底避開檔案鎖問題。此項不在本次範圍,列為 Phase 2 評估項目。
