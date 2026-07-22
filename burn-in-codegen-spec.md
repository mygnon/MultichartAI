# 策略燒錄階段(Burn-in Codegen)規格與實作規劃 Prompt — v0.2

供 Claude Code plan mode 使用,在 **MultichartAI repo** 內實作。
v0.2 依據實際 repo(github.com/mygnon/MultichartAI)結構修訂:輸入介面改為既有的
`results/*/state.json`,出場模組拼裝改為「多訊號合併」問題,並新增以 mc_automation
回測驗證燒錄結果的等價性關卡。

---

## 0. 系統全貌與本任務定位

既有產線(勿修改):

```
Strategy/{Name}_crypto.txt, {Name}_NQ.txt   ← Claude Code 產生的主策略 EL 原始碼
optimizer/run_*_allinst_pipeline.py          ← 六商品 × 4 階段 UI 自動化 pipeline
  Stage 1: IS 高原最佳化(自判收斂)
  Stage 2: OOS 冠軍篩選(PASS = |MDD_full| ≤ |MDD_is|)
  Stage 3: 出場模組(M1–M6)逐一 IS 最佳化
  Stage 4: 貪婪 RoMaD 堆疊,KEEP 的模組與參數寫回 MC Format Objects
results/<inst>_<strat>_<tf>_pipeline/state.json   ← 每商品完整四階段結果
```

**本任務(燒錄階段)**:讀取 state.json,把「主策略最優參數 + Stage 4 KEEP 的出場
模組及其參數」燒成**每商品一隻自足的單一 EL 訊號**,供 Portfolio Trader 以
一列一訊號、零參數設定的方式掛載數千隻策略。

已定案設計決策(不要在計畫中挑戰):

1. **參數燒進程式碼**:每商品一隻獨立訊號,最優參數為 inputs 預設值,PT 端零設定。
2. **單一訊號輸出**:主策略與 KEEP 模組合併為一隻訊號。PT 千列規模下,每列掛
   1 隻主訊號 + 最多 6 隻模組訊號的管理成本不可接受。
3. **manifest 為單一事實來源**:EL 與 manifest 同一次產出,OMS 上架與未來
   Python 雙生成都吃 manifest。
4. **等價性驗證是硬關卡**:燒錄出的合併訊號,回測績效必須重現 Stage 4 終值
   (見 §4.4),不吻合即燒錄失敗。

## 1. 輸入介面(既有,只消費不修改)

每商品一份 `results/<sub>_pipeline/state.json`:

- `stage1.champion_obj / champion_np / candidates[]`:主策略參數 + NP/MDD/Obj/trades
- `stage2.results.is / oos`:各候選 IS 與 OOS 績效(取最終獲選者)
- `stage3`:六個出場模組各自的最優參數
- `stage4.baseline + steps[]`:貪婪堆疊軌跡;`decision: KEEP` 的模組即最終出場
  組合,每步含 enabled 清單、NP、max_intraday_drawdown、romad
- 主策略獲選參數以 `final_params_*.json` / stage2 champion-select 結論為準
  (計畫中列為澄清項:多候選時的最終獲選規則)

商品六種,對應既有訊號變體:
`BTCUSDT / ETHUSDT / BNBUSDT → {Name}_crypto`,`TWF.TXF / CME.NQ / CME.GC → {Name}_NQ`。

## 2. 命名規範

```
{Name}_{INST}_{TF}_v{n}
範例:DualAnchorBreakout_BTC_H1_v1
```

- 同一字串 = PowerLanguage 訊號名 = 檔名(.txt,沿用 repo 慣例)= manifest
  `strategy_id` = OMS registry 主鍵。
- INST 枚舉:`BTC, ETH, BNB, TXF, NQ, GC`;TF 枚舉:`H1, D1, M240`。
- `v{n}`:同商品 re-optimize 時遞增,舊檔保留不覆蓋。
- 產出目錄:`burned/{Name}/`,含 6 組(.txt + .manifest.json)+ `burn_report.json`。

## 3. Manifest Schema(`{strategy_id}.manifest.json`)

```json
{
  "schema": 1,
  "strategy_id": "DualAnchorBreakout_BTC_H1_v1",
  "base_strategy": "DualAnchorBreakout",
  "source_variant": "DualAnchorBreakout_crypto",
  "symbol": "BTCUSDT",
  "symbol_class": "crypto",
  "timeframe": "H1",
  "params": { "Length": 291, "BandMult": 2.4375, "ATRMult": 13.5, "ReentryBars": 13 },
  "exit_modules": [
    { "id": "M5", "signal": "QuantPass_PT_Exit", "params": { "PT_Base": 0.012 } },
    { "id": "M3", "signal": "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit", "params": { "EXITBAR": 48 } }
  ],
  "stage4_final": { "net_profit": 14118.4, "max_intraday_drawdown": -6800.13, "romad": 2.0762 },
  "dependencies": [],
  "oos": { "net_profit": 0, "mdd": 0, "pass": true },
  "source_state_json": "results/btc_dualanchor_hourly_pipeline/state.json",
  "source_state_sha256": "<state.json 雜湊>",
  "el_sha256": "<燒錄產出之 .txt 內容雜湊>",
  "burned_at": "2026-07-22T05:00:00Z"
}
```

- `stage4_final` 同時是 §4.4 等價性驗證的期望值,與 OMS 自動下架門檻的輸入
  (`mdd_limit = max(1.5 × oos_mdd, floor)`,見 oms-spec.md §5.2)。
- `source_state_sha256` + `el_sha256`:三方(輸入、產出、manifest)互相鎖定,
  防手改脫鉤。

## 4. 合併(Assembly)規則與驗證

### 4.1 模組合併的本質與已知風險

出場模組是**獨立訊號**,在 MC 內與主策略共享同一 strategy context
(marketposition、entryprice 等)。合併成單一訊號時此語意天然保留。
六個模組原始碼已入 repo:`Strategy/modules/*.txt`(verbatim,assembler
不得修改原始檔,所有變換在渲染時進行)。

模組清冊與合併風險(assembler 設計的直接依據):

| ID | 訊號 | inputs | vars | 訂單 | 合併風險 |
|---|---|---|---|---|---|
| M1 | lesson4_ATRstop | STP | ATR | SELL/BUYTOCOVER **未命名** stop | ⚠️ 未命名訂單、ATR 撞名 |
| M2 | lesson9_1_TrailingStop | ATRSTP | ATR, MP, POSH, POSL | "AtrLX"/"AtrSX" stop | ATR 撞名;MP[1] 序列相依 |
| M3 | lesson10_3_EntryBarsAfterExit | EXITBAR | — | SELL/BUYTOCOVER **未命名** market | ⚠️ 未命名訂單 |
| M4 | lesson11_3_high_volatility_exit | DAYRANGE | ATR | "DayRange_LX/sX" market | ATR 撞名 |
| M5 | QuantPass_PT_Exit | PT_Base | PT | "LX_PT"/"SX_PT" market | 低 |
| M6 | RescueTeamExit | **Length**, std | — | "Buytocover " market | ⚠️ **Length 與主策略 input 撞名** |

Assembler 必要變換(渲染時套用,依序):

1. **inputs / vars 前綴化**:`m{n}_` 前綴(如 `m6_Length`、`m2_POSH`)。
   M6 的 `Length` 與多數主策略的 `Length` input 直接衝突,不前綴必炸。
2. **訂單命名**:M1、M3 的 SELL/BUYTOCOVER 未命名。獨立訊號時各自 context
   互不干擾;合併進單一訊號後,**同名(含預設名)訂單會互相覆蓋**,M1 的
   stop 單可能被 M3 的 market 單蓋掉。渲染時為每張訂單補上唯一名稱
   (`"M1_LX"` 等)。獨立使用時加名稱不改變行為,此變換安全。
3. **ATR 去重決策**:M1/M2/M4 各自宣告 `ATR = AvgTrueRange(10)`,前綴化後
   為三個等值變數。預設保留三份(正確性優先);合併為單一 `m_ATR10` 列為
   計畫中的可選最佳化,需 golden test 證明等價。
4. **模組間順序**:合併後的程式碼順序固定為 M1→M6 中被 KEEP 者依 Stage 4
   堆疊順序排列;§4.4 等價性驗證負責裁決順序語意是否與多訊號一致。
5. **函數相依內聯**:部分主策略呼叫 PL 函數 `_Crypto1MUSD`(原始碼:
   `Strategy/modules/_Crypto1MUSD.txt`,回傳 ≈$1M 名目 / Close 的整數單位,
   C 無效時回 1)。燒錄時**預設將其內聯**為訊號內變數,使燒錄產物零外部相依
   (PT 部署機不需預先編譯任何函數);內聯與函數呼叫的等價由 golden test 與
   §4.4 共同保證。注意 repo 內兩種既有寫法並存(部分策略呼叫函數、部分已
   內聯且 min-guard 寫法略異),assembler 需以函數版語意為準統一。
6. **OMS 訊號輸出區塊(PT + OMS 架構必要)**:燒錄產物尾端固定嵌入目標部位
   輸出程式碼——每根 bar 收盤(及進出場當下)將 `strategy_id, symbol,
   target_units(=currentcontracts × marketposition 方向), theoretical_equity
   (netprofit + openpositionprofit), bar_time, emit_time` 以先寫 tmp 再
   rename 的原子性方式寫入 `C:\oms\signals\{strategy_id}.json`(格式契約:
   oms-spec.md §2.1)。此區塊為模板固定內容,不參與最佳化;§4.4 等價性驗證
   時允許存在(只寫檔,不影響交易邏輯)。

### 4.2 渲染

Jinja2 模板;產出檔頂部固定註解:strategy_id、燒錄時間、source_state_sha、
template 版本。`// AUTOGEN — do not edit by hand`。

### 4.3 靜態驗證(燒錄後、編譯前)

1. 語法預檢(括號/保留字/inputs 引用完整性)
2. 參數回讀:從產出 EL parse 回 inputs 預設值,與 manifest 逐一全等比對
3. 命名檢查:strategy_id 格式、枚舉、版本遞增
4. 任一關失敗 → 該商品標記失敗、不產生半成品,寫入 burn_report.json

### 4.4 等價性驗證(硬關卡,燒錄的靈魂)

用既有 `mc_automation` 把燒錄出的合併訊號掛上對應圖表,跑 full-period 回測,
NP 與 Max Intraday DD 必須與 manifest `stage4_final` 吻合(容差 ±0.5%,
計畫中可提案調整)。不吻合 = 合併語意有誤(最可能是模組求值順序或
intrabar 語意),整組作廢並輸出兩邊交易數供除錯。

## 5. Plan Mode 指示

你是本專案工程師,在 MultichartAI repo 內工作。先讀 `CLAUDE.md`、
`optimizer/run_hlmean_allinst_pipeline.py`(pipeline 全貌)、
`Strategy/DualAnchorBreakout_crypto.txt`(主策略樣本)、任一
`results/*_pipeline/state.json`,再制定計畫。
優先序:**等價性 > 可溯源性 > 整合順暢度 > 開發速度**。

計畫必須包含:

1. 模組化架構:state.json reader、assembler(§4.1)、renderer、
   static validator(§4.3)、equivalence runner(§4.4,重用 mc_automation)、
   manifest writer 的介面定義
2. state.json → 「最終獲選主策略參數」的抽取規則提案(stage1 candidates 與
   stage2 champion-select 的對應關係,先查證再提案)
3. 測試計畫:
   - golden file test:固定 state.json → 產出 EL 與 manifest 逐字元比對基準
   - 參數回讀 round-trip 自動化
   - 模組合併組合測試(任意 KEEP 子集不撞名、inputs 不重複、順序穩定)
   - 等價性驗證的 dry-run 模式(不開 MC,只產出待驗清單)
4. Milestone 切分,第一個 milestone:「單商品、零出場模組(pure reversal)
   的最小燒錄 + 靜態驗證 + golden test」;等價性驗證(需 MC64 環境)獨立成
   最後 milestone

開始規劃前:
1. 列出本 spec 的歧義與你不同意處(特別是 §4.1 模組求值順序、§1 最終獲選規則)
2. 一次列完澄清問題(至少:等價性容差、re-optimize 版本號由誰遞增、
   `_Crypto1MUSD` 部位語法在 PT 的行為是否一致、M1 stop 單與 M3/M4/M5
   market 單同 bar 並存時 MC 的成交優先序)

禁止事項:
- 不修改 optimizer/ 既有程式碼與 results/ 內容;mc_automation 只 import 重用
- 不實作 PT 掛載自動化、Python 訊號引擎、OMS 對接(另案;manifest schema 需
  預留,頂層不可出現 EL 專屬欄位)
- 產出碼中不得有硬編碼絕對路徑以外的環境假設(Windows 路徑沿用 repo 慣例)
