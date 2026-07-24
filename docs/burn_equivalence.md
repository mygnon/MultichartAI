# 燒錄產物等價性驗證程序(spec §4.4 硬關卡)

驗證「燒錄合併訊號」與「原多訊號組態」在同一天、同一份資料上回測結果一致。
基準是**同日 A/B 對測**(不是 manifest 裡的 stage4_final —— 價格資料每天更新,
數週前的數字只作 advisory 參考;CLAUDE.md 鐵則 5)。

## 前置(每個要驗的商品各一次,手動)

1. `py -m burner burn --name <Name> --key <key>` 先燒出最新版(冪等,重跑無害)。
2. `py -m burner verify --name <Name> --key <key> --dry-run` 產出
   `burned/<Name>/equivalence_checklist.json` —— 每商品列出:要編譯的檔案、
   A/B 各自要開的訊號、input 值、FULL 日期範圍。
3. 開 MC64(**以系統管理員身分**)+ 對應 workspace(checklist 的 `workspace` 欄)。
4. **手動編譯燒錄訊號**(無自動化):PowerLanguage Editor → New Signal →
   名稱 = `strategy_id`(如 `DualAnchorBreakout_BTC_H1_v2`)→ 貼上
   `burned/<Name>/<strategy_id>.txt` 全文 → Compile。
   ⚠️ 首次編譯同時驗證 OMS 區塊的相容性 — v1 已實證:`GetAppInfo(aiRealTimeCalc)`、
   `DoubleQuote`、`FormatDate/FormatTime`、`ELDateToDateTime/ELTimeToDateTime`、
   `ComputerDateTime`、`DefineDLLFunc`+`MoveFileExA`;**v2 新增待驗**:
   `CreateDirectoryA`、`GetFileAttributesA`、`DeleteFileA`、`GetTickCount`、
   `Sleep`。任一不過 → 回報,調整 templates.py。
   v2 emit 行為:零拋錯設計 — 訊號檔寫入 `Z:\oms\signals\`(ramdisk),
   每次 emit 自癒建目錄(重開機清空自動恢復);Z: 未掛載或 rename 重試
   5 次仍失敗 → 本 bar 靜默跳過(下一 bar 重發,OMS 靠 heartbeat 判 stale),
   絕不拋 runtime error(拋錯會解除 AOE)。
5. 把編譯好的燒錄訊號 **Insert 到對應商品的 chart** 上(Status 可先關)。
   Study Editor 編譯完後**關閉**(pipeline 慣例)。

## 執行

```powershell
cd C:\Users\Tim\MultichartAI
py -m burner verify --name DualAnchorBreakout --key dualanchor --inst btc   # 先單商品
py -m burner verify --name DualAnchorBreakout --key dualanchor             # 再全 6
```

每商品流程(自動):
1. `activate_chart_by_symbol` 切到該 chart;`set_instrument_data_range` 設 FULL 範圍。
2. **Run A**:開主訊號 + KEPT 模組(其餘關)、套 main_champ + stage3 參數;
   以 3-point 微網格最佳化跑 full-period,取精確列 → NP_A / MDD_A。
3. **Run B**:只開燒錄訊號,同法量測 → NP_B / MDD_B。
4. PASS 條件:`|NP_A−NP_B|/|NP_A| ≤ 0.5%` 且 MDD 同(`--tolerance` 可調)。
5. 結果寫 `burned/<Name>/equivalence_report.json`(含 A/B 兩邊指標與
   advisory 的 manifest stage4_final 差異 = 資料飄移量)。

## FAIL 時

整組視為燒錄失敗(spec §4.4)。report 內有兩邊 NP/MDD;最可能原因依序:
1. 合併後模組求值順序 ≠ 多訊號順序(同 bar M1 stop 與 M3/M4/M5 market 並存的成交裁決)。
2. 未命名→命名訂單改變了 MC 的訂單覆蓋行為。
3. OMS 區塊意外影響(理論上 real-time guard 下不可能;可暫時刪除區塊重編譯排除)。
除錯:比對 A/B 的 List of Trades(MC 報表)找第一筆分歧交易。

## 負面對照(關卡有效性證明,做一次)

把燒錄訊號改一個參數(如 Length 8→9)重編譯再跑 verify —— 必須 FAIL。
