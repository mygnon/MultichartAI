# 多策略 OMS 規格書
**Portfolio Trader × 幣安 USDT-M 合約｜目標部位制下單機**

版本：v0.1 草案｜適用規模：1,000–5,000 隻策略

---

## 1. 系統總覽

```
┌─────────────────────┐    目標部位檔     ┌──────────────────────────┐
│ MC Portfolio Trader │ ──(atomic write)─▶│        Python OMS         │
│  (數千隻策略, 模擬  │                   │  ┌────────────────────┐  │
│   經紀商, 不實單)   │                   │  │ 1 訊號收集 Collector│  │
└─────────────────────┘                   │  │ 2 策略註冊表 Registry│ │
                                          │  │ 3 虛擬帳本 Ledger   │  │
┌─────────────┐   WebSocket/REST          │  │ 4 風控 MDD Monitor  │  │
│   Binance   │ ◀──────────────────────── │  │ 5 對帳引擎 Reconciler│ │
│ USDT-M 合約 │    差額下單 / 部位查詢     │  │ 6 執行器 Executor   │  │
└─────────────┘                           │  │ 7 儀表板 Dashboard  │  │
                                          │  └────────────────────┘  │
                                          └──────────────────────────┘
```

核心原則：**PT 只產生訊號，永遠不碰真實資金**。PT 的經紀商設定為模擬（paper），每隻策略把「目標部位」寫進檔案；OMS 把所有已上架策略的目標部位按 symbol 淨額加總，與幣安實際持倉比對，只對差額下單。補單、上下架平倉、斷線恢復全部由同一個對帳迴圈天然完成。

---

## 2. 訊號輸出層（PT 端）

### 2.1 EasyLanguage 輸出規格

每隻策略在每根 bar 收盤（以及進出場當下）呼叫共用函式，輸出當前目標部位。由 Claude Code 量產策略時直接嵌入此模板。

輸出方式：**每策略一檔，原子性覆寫**（先寫唯一檔名 `*.tmp` 再 MoveFileExA rename），避免 OMS 讀到半截檔案。

檔案路徑：`Z:\oms\signals\{strategy_id}.json`（Z: 為 ramdisk；見 §2.3）

```json
{
  "strategy_id": "BRK_BTCUSDT_H1_0042",
  "symbol": "BTCUSDT",
  "target_units": -1,
  "theoretical_equity": 1083.5,
  "bar_time": "2026-07-21T14:00:00Z",
  "emit_time": "2026-07-21T14:00:01.312Z",
  "schema": 1
}
```

| 欄位 | 說明 |
|---|---|
| `target_units` | 目標部位，單位為「口」（-N/0/+N），實際名目金額由 OMS 端的 `allocated_capital` 換算，PT 端不管資金 |
| `theoretical_equity` | 策略理論權益（PT 回測口徑），供 MDD 監控與追蹤誤差比對 |
| `bar_time` | 訊號所屬 bar 的收盤時間，用於過期判斷 |
| `emit_time` | 寫檔時間，作為 heartbeat |

### 2.1.1 寫入端零拋錯規範(template v2,burner/templates.py)

EL 無 try/catch,檔案內建函數一拋 runtime error 就會解除該策略的 AOE。因此 v2 emit 區塊:

1. **自癒目錄**:每次 emit 先 `CreateDirectoryA`(冪等)+ `GetFileAttributesA` 確認;Z: 未掛載/重開機清空 → 本 bar 靜默跳過,不寫不拋。開機順序不依賴任何 boot 腳本。
2. **唯一暫存檔名**:`{out}.{GetTickCount}.{seq}.tmp`,無同名爭搶,免 FileDelete。
3. **rename 重試**:`MoveFileExA`(BOOL 回傳,不拋錯)失敗 → Sleep(30ms) 重試,共 5 次;全敗 → `DeleteFileA` 清殘檔 + 跳過本 bar(下一 bar 重發,OMS 靠 §2.2 heartbeat 判 stale)。

### 2.1.2 讀取端 I/O 規範(collector 實作必守)

1. glob **只匹配 `*.json`** — 嚴禁 `*.json*` 之類 pattern(會誤掃 `.tmp`)。
2. `with open(...)` 即讀即關,任何時候不長時間持有 handle(持有中的 rename-onto 會被 Windows 檔案鎖擋下)。
3. `PermissionError` / `OSError` / `json.JSONDecodeError` 一律**跳過本輪、下輪再讀**,不 crash、不原地重試佔住檔案。

```python
def scan_signals(d: Path) -> dict[str, dict]:
    out = {}
    for p in d.glob("*.json"):          # never *.json*
        try:
            with open(p, encoding="utf-8") as f:   # open-read-close at once
                out[p.stem] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue                     # skip this cycle; next poll re-reads
    return out
```

驗收/稽核工具:`py -m burner.tools.signal_lock_stress`(`--file` 持鎖壓測寫入端 retry;`--scan-dir` 稽核 glob 規範與 .tmp 殘留)。

### 2.3 Ramdisk(Z:)注意事項

- **重啟後 Z:\ 全部消失**:訊號檔為暫態資料,消失無妨;寫入端自癒建目錄(§2.1.1),OMS 重啟後的補單/對帳一律以「向交易所重新查詢實際部位」為準(§6.3),不依賴舊訊號檔。
- **持久化狀態(registry、帳本、MDD 水位、equity_history)一律放 C:\**,不可只存在 Z:\。
- ramdisk 軟體若有「定期備份到硬碟」功能,對 `Z:\oms` 關閉或大幅調低(備份當下鎖檔)。
- 防毒即時掃描排除 `Z:\oms`(原 C:\oms 持久化目錄也一併排除)。

### 2.2 過期與心跳規則

- OMS 記錄每隻策略的 `emit_time`；超過 `stale_after = 3 × bar_interval` 未更新 → 標記 `STALE`。
- `STALE` 處理策略（registry 可設定，預設 `freeze`）：
  - `freeze`：沿用最後一次目標部位（適合 PT 短暫重啟）
  - `flat`：目標部位視為 0，OMS 平掉其份額（適合長時間失聯）
- `STALE` 超過 `stale_kill = 24h` → 強制 `flat` 並發告警。

---

## 3. 策略註冊表（Registry）

儲存：**PostgreSQL**（單機 Docker 即可；SQLite 在數千策略高頻寫入下鎖競爭會成為瓶頸）。

### 3.1 `strategies` 表

| 欄位 | 型別 | 說明 |
|---|---|---|
| `strategy_id` | text PK | 與訊號檔名一致 |
| `symbol` | text | 交易對 |
| `status` | enum | `probation` / `active` / `paused` / `delisted` |
| `allocated_capital` | numeric | 分配資金（USDT），1 口 = 1 × allocated_capital 名目 |
| `weight` | numeric | 部位縮放係數（probation 期用，0–1） |
| `mdd_limit` | numeric | 個別 MDD 門檻（見 §5.2） |
| `oos_mdd` | numeric | 樣本外回測 MDD，量產流程上架時寫入 |
| `oos_sharpe` | numeric | 樣本外 Sharpe，供儀表板排序 |
| `stale_policy` | enum | `freeze` / `flat` |
| `listed_at` / `delisted_at` | timestamptz | |
| `delist_reason` | text | `mdd_breach` / `tracking_error` / `manual` / `stale` |

### 3.2 狀態機

```
                 ┌──────────┐  觀察期滿且達標   ┌────────┐
  上架 ────────▶ │probation │ ───────────────▶ │ active │
                 └────┬─────┘                  └───┬────┘
                      │  MDD/追蹤誤差破限          │ MDD 破限 / 手動
                      ▼                            ▼
                 ┌──────────┐    只允許手動     ┌──────────┐
                 │ delisted │ ◀───────────────  │  paused  │
                 └──────────┘     重新上架      └──────────┘
```

- **上下架 = 改一個 status**。非 `probation/active` 的策略在淨額加總時目標部位一律視為 0，下一輪對帳自動把它的份額平掉——不需要任何額外的平倉程式碼。
- `probation`（試用期）：新策略上架後前 `probation_days = 14` 天以 `weight = 0.3` 跑，期滿且未破限自動升 `active`、weight 恢復 1.0。這是量產策略最重要的防線：過擬合的策略大多在上線初期就現形。
- 重新上架一律手動，且重新進入 `probation`。**不做自動復活**——MDD 破限代表策略與市場失配，自動復活等於重複踩同一個坑。

---

## 4. 虛擬帳本（Virtual Ledger）

交易所只持有淨部位，每隻策略的部位與損益存在於 OMS 自己的帳本。

### 4.1 `positions` 表（每策略一列，即時狀態）

`strategy_id, virtual_position(口), virtual_qty(幣), avg_entry_price, realized_pnl, unrealized_pnl, equity, peak_equity, current_dd`

### 4.2 `fills_allocation` 表（成交分攤）

每筆實際成交按「本輪各策略目標變化量」比例分攤：

```
本輪 BTCUSDT 淨目標變化 = Σ 各策略目標變化
策略 i 分攤數量 = 實際成交量 × (策略 i 目標變化 / 淨目標變化)
策略 i 分攤價格 = 該輪實際成交均價（含手續費攤提）
```

對向抵銷的部分（A 要買 1、B 要賣 1，交易所不動）在帳本內以**當時標記價格互轉**，視同 A、B 各自以 mark price 成交。

### 4.3 權益計算

每個對帳週期 mark-to-market：

```
equity_i = allocated_capital + realized_pnl_i + virtual_qty_i × (mark_price − avg_entry_i)
```

同時保留 PT 回傳的 `theoretical_equity`，兩條曲線都入庫（`equity_history` 表，每 bar 一筆）。

---

## 5. 風控與自動上下架（MDD Monitor）

每個對帳週期執行，順序：個別策略 → 組合層。

### 5.1 兩條權益曲線、三種下架觸發

| 觸發 | 依據 | 規則 | 動作 |
|---|---|---|---|
| **實盤 MDD 破限** | 虛擬帳本 equity | `current_dd > mdd_limit` | status → `delisted`，reason=`mdd_breach` |
| **理論 MDD 破限** | PT theoretical_equity | 同上 | 同上（策略本身失效，不等實盤虧到） |
| **追蹤誤差破限** | 兩條曲線差 | `(theoretical_equity − equity) / allocated_capital > 8%` 且持續 3 天 | status → `paused`，reason=`tracking_error`（滑價/成交假設有問題，先停不刪） |

### 5.2 MDD 門檻設定

```
mdd_limit = max( k × oos_mdd, floor )
建議預設：k = 1.5, floor = 10% × allocated_capital
```

理由：實盤 MDD 若已超過樣本外回測 MDD 的 1.5 倍，該策略大概率已失效；floor 避免 OOS MDD 極小的策略被雜訊誤殺。DD 分母用 `allocated_capital`（固定資金口徑），與回測口徑一致。

### 5.3 組合層 kill switch

- 組合總權益 DD > `portfolio_mdd_limit`（建議 15–20%）→ 全部策略 `paused`、清空所有部位、停止開新倉，**需人工解鎖**。
- 單一 symbol 淨名目 > `symbol_cap`（建議總資金 20%）→ 該 symbol 所有策略目標部位等比例縮放。
- 幣安 API 連續失敗 > 60s → 停止下單、告警，只做唯讀監控。

---

## 6. 對帳引擎（Reconciler）——系統心臟

### 6.1 主迴圈（每 `cycle = 5s`）

```
1. 掃描 signals/ 目錄，載入全部訊號檔（含 stale 判定）
2. 讀 registry，過濾 status ∈ {probation, active}
3. 每策略換算目標數量：
   target_qty_i = target_units_i × weight_i × allocated_capital_i / mark_price
4. 按 symbol 加總 → net_target_qty[symbol]
5. 取幣安實際持倉（user data stream 本地快取 + 每 60s REST /positionRisk 校正）
6. diff = net_target_qty − actual_qty
7. 若 |diff × mark_price| > deadband → 送執行器
8. 更新虛擬帳本、跑 MDD Monitor、寫 equity_history
```

### 6.2 Deadband（防抖動）

`deadband = max(minNotional × 1.2, net_target_notional × 0.5%)`

避免因 mark price 波動造成目標數量微幅變化而反覆下單。**這是淨額制系統最容易漏掉、也最燒手續費的一個參數。**

### 6.3 冪等與崩潰恢復

- 所有下單帶 `newClientOrderId = oms-{symbol}-{cycle_ts}-{seq}`，重啟後可用 client id 查回未確認訂單，不重複下。
- OMS 重啟後第一輪：先全量 REST 拉持倉與掛單 → 取消所有殘留掛單 → 正常進入迴圈。因為是目標部位制，**崩潰期間漏掉的所有訊號，重啟第一輪對帳就全部補齊**。

---

## 7. 執行器（Executor）

- 訂單型態：`diff` 名目 < 5,000 USDT → 直接市價（taker）；≥ 5,000 USDT → post-only 限價追單（貼對手價 −1 tick，每 2s 重掛，10s 未成交轉市價）。
- 數量處理：依 `exchangeInfo` 的 `stepSize`/`minQty`/`minNotional` 取整；取整後低於 minNotional 則本輪跳過（留給 deadband 下輪累積）。
- 持倉模式：**單向持倉（One-way）**，與淨額制天然一致。
- Rate limit 預算：order 類請求集中由 Executor 單一佇列發送，全域限速 5 orders/s（遠低於幣安上限，留餘裕給查詢）。淨額制下數千策略實際下單頻率很低，正常情況每輪只有少數 symbol 有 diff。
- 帳戶隔離：策略交易用獨立子帳戶，API key 僅開「讀取 + 合約交易」、綁 IP 白名單、**不開提現**。儀表板用另一組唯讀 key。

---

## 8. 儀表板（Dashboard）

Streamlit 或 FastAPI + 簡單前端，功能優先序：

1. 策略總表：status、equity、current_dd / mdd_limit、追蹤誤差、oos_sharpe，可排序篩選，一鍵 pause / delist / 重新上架（進 probation）。
2. 組合權益曲線 + 組合 DD、各 symbol 淨部位與名目占比。
3. 事件日誌：所有自動上下架事件（時間、策略、觸發原因、當時 DD 數值）。
4. 對帳健康度：訊號檔 stale 數量、上輪 diff 清單、API 錯誤率。

---

## 9. 部署架構（兩機制：開發機 + 實盤機）

### 9.1 機器分工

| | 開發機（Win11 桌機） | 實盤機（Win11 筆電） |
|---|---|---|
| 跑什麼 | Claude Code、最佳化 pipeline（UI 自動化）、燒錄 codegen | MC64 + PT（燒錄訊號）、OMS 全套、幣安連線 |
| 不跑什麼 | 任何實盤連線 | Claude Code、UI 自動化、最佳化 |
| MC 用途 | 開發、回測、最佳化、等價性驗證 | 純實盤 |

實盤機原則：**上面的東西越少越好**。不裝開發工具，Windows Update 設定延遲＋固定重啟時段，停用睡眠/休眠，有線網路（筆電內建電池兼作 UPS，建議再備手機熱點作網路備援）。

### 9.2 實盤機服務結構

```
Z:\oms\                 # ramdisk（暫態，重啟即清空；見 §2.3）
└── signals\            # PT 燒錄訊號寫入的目標部位檔（微秒級寫入、零 SSD 磨損）

C:\oms\                 # 持久化（重啟保留）
├── collector\      # 訊號檔掃描（讀 Z:\oms\signals）
├── core\           # 淨額計算、虛擬帳本、MDD monitor（單一程序）
├── executor\       # 幣安下單佇列（唯一持有交易 key）
├── dashboard\
└── db\             # PostgreSQL
```

OMS 以 Docker Desktop（WSL2）跑既有 compose，`Z:\oms\signals\` 以 bind mount 掛進 collector；設定開機自啟（collector 啟動時自建 signals 目錄，與寫入端自癒互為備援）。若 Docker Desktop 在筆電上不穩，備案是原生 Python venv + Windows 服務（NSSM），計畫中列為決策點。

### 9.3 燒錄產物同步與上架流程（開發機 → 實盤機）

1. 開發機：燒錄完成的 `burned/{Name}/`（.txt + manifest + burn_report）commit 進 git
2. 實盤機：`git pull` 取得新策略（唯一的傳輸通道，天然有版本與稽核軌跡）
3. 實盤機：於**固定上架窗口**（如每日一次、低波動時段）編譯新訊號、PT 加列掛載——上架窗口以外的時間不對 MC/PT 做任何操作
4. OMS 上架：讀 manifest 寫入 registry（`oos_mdd` → `mdd_limit`），策略進 `probation`
5. 驗證：上架後第一個對帳週期確認該策略訊號檔出現、目標部位為 0 或與 PT 一致

上架窗口內若需批次加列，可在實盤機執行受控的掛載腳本（重用 mc_automation 的子集），但僅限窗口內、僅限掛載操作，跑完即退出。

### 9.4 實盤機健康監控

- OMS 每 60s 送 heartbeat 至 Telegram/LINE bot；斷訊即人工介入
- 監控項：PT 訊號檔更新延遲、對帳 diff 異常量、API 錯誤率、磁碟/記憶體
- 筆電效能水位：PT 掛載策略數逐步增加時，監控 bar 運算延遲（訊號檔 `emit_time` − `bar_time`），延遲 > 5s 即為筆電容量上限訊號，考慮升級或分機

---

## 10. 分階段上線

### 10.1 Phase 1：PT + OMS（數百隻試跑）

| 階段 | 內容 | 出場條件 |
|---|---|---|
| P0 | 10 隻策略、testnet、全流程跑通 | 對帳/補單/上下架各演練一次（手動改倉測補單） |
| P1 | 50 隻策略、實盤最小資金 | 追蹤誤差 < 3%/月，無 rate limit 錯誤 |
| P2 | 100 → 300 → 500 隻逐步加載 | 下列五項全數達標 |
| P3 | 數千隻全量（或觸發 Phase 2） | — |

**Phase 1「方案可行」的量測定義（P2 出場條件）**：

1. **PT 容量曲線**：每次加載後記錄 `emit_time − bar_time` 延遲分布；延遲 p95 < 5s。曲線外推出筆電容量天花板，作為 P3 或 Phase 2 的決策輸入
2. **追蹤誤差**：虛擬帳本實盤權益 vs PT 理論權益，< 3%/月
3. **fills 分攤守恆**：Σ 各策略虛擬損益 ≡ 帳戶實際損益（每日對帳，容差 < 0.1%）
4. **上架窗口吞吐**：單一窗口可完成的「pull → 編譯 → PT 加列 → registry 上架」隻數 ≥ 20（低於此值，數千隻的上架期程不可行）
5. **MDD 下架實彈演練**：人工將一隻策略 `mdd_limit` 調至必破，驗證自動下架 → 份額平倉 → 事件記錄全程正確，至少一次

### 10.2 Phase 2：Python 引擎取代 PT（影子運行切換）

前提：Phase 1 達標後，Python 策略引擎以燒錄 manifest 為輸入 codegen 產生（見 burn-in-codegen-spec.md；manifest 頂層無 EL 專屬欄位即為此預留）。OMS 為目標部位制，**引擎替換對 OMS 透明**——只是訊號檔來源改變，OMS 零修改。

切換關卡（依序，全數通過才可切換）：

1. **歷史逐筆比對**：Python 引擎與 MC 在同一段歷史資料的交易清單逐筆比對（進出場時間、價格、方向），比對基準用 mc_automation 既有的 MCReport CSV 匯出；要求 100% 吻合
2. **影子運行（shadow run）**：Python 引擎在實盤機與 PT 並行，讀同樣的幣安即時資料，目標部位寫入 `signals_shadow\`，OMS 照常只吃 PT 的 `signals\`。獨立比對程序逐 bar 比對兩目錄，任何偏差記錄告警
3. **切換門檻**：連續 4 週影子運行零目標部位偏差（含斷線重連、資料補齊等即時場景）
4. **切換與回退**：OMS 的訊號目錄指向改為 `signals_shadow\`（一個設定值）；PT 降為影子繼續並行 2 週作回退保險，之後才下線

影子運行的價值在於暴露歷史回測測不到的即時差異：bar 未收完的處理、行情延遲、重連補資料的 bar 重算。這一關不可用回測比對替代。

---

## 11. 已知風險與待決事項

1. **PT 單機容量**：數千隻策略單台 PT 的 CPU/記憶體是實際瓶頸，可能需拆 2–4 台 PT（訊號目錄共用，strategy_id 不重複即可，OMS 端完全無感）。
2. **資金費率**：虛擬帳本目前未分攤 funding fee，建議按各策略持倉比例每 8h 分攤入 realized_pnl。
3. **同 symbol 部位互抵的損益歸屬**：§4.2 以 mark price 互轉是近似，長期高互抵率的 symbol 需驗證分攤誤差。
4. **mdd_limit 的 k 值**：1.5 是起始值，P1 階段後用實際下架策略的後續表現回頭校準。
