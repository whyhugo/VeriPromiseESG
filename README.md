# VeriPromiseESG

AI CUP - ESG 永續承諾驗證競賽 2026

## 簡介

隨著全球永續發展目標（SDGs）與淨零碳排政策推動、監管趨勢日益加劇，企業揭露 ESG 承諾已逐漸從自願性作法，轉變為投資人、監理機關與公民社會衡量企業責信與治理能力的重要指標。

然而，許多企業的永續報告書仍充斥語意模糊的承諾，缺乏具體佐證，引發漂綠疑慮。儘管 AI 發展迅速，但針對繁體中文 ESG 承諾的自動化分析仍面臨語料稀缺挑戰。因此，發展具備邏輯識別與可檢驗性的 NLP 技術，是提升永續資訊品質的關鍵。

為培養參賽者對 ESG 文本理解與資料驅動決策的實務能力，並促進產學界在語意理解與永續承諾稽核技術上的創新應用，特辦理「ESG 永續承諾驗證競賽 2026」。本競賽將提供源自真實產業情境建構之「VeriPromiseESG4K 標註語料庫」，讓參賽者開發能夠識別企業永續承諾與支撐證據之自然語言 AI 模型，提升企業永續承諾的內涵與可信度。

我們期待透過資料與模型的實作歷程，讓參賽者不只是完成一項技術任務，而是學會如何以 AI 回應現實世界的永續課題，為企業揭露資訊的透明度注入更多理性與信任。當技術成為理解承諾、守護未來的工具，這些模型所承載的，將不只是分數與成果，更是對永續責任的一次具體回應，與一份持續向前的承諾。

本競賽將提供從真實企業永續報告書中標註的 4,000 筆繁體中文 ESG 資料集（VeriPromiseESG4K），讓參賽者運用自然語言處理、大型語言模型等 AI 技術，開發能夠自動識別、分析和驗證企業永續承諾的 NLP 模型，完成以下四大核心任務，藉此推動企業永續資訊揭露的透明化與可信度：

- 子任務一：承諾語句識別（Commitment Classification）
  判斷給定的文本是否表達企業對未來行動的明確承諾。
- 子任務二：支持證據連結（Evidence Identification）
  判斷已識別的承諾語句是否附有具體執行計畫或支持佐證。
- 子任務三：證據清晰度評估（Clarity Classification）
  評估證據語句是否語意明確、無模糊字眼。
- 子任務四：預期驗證時機推論（Timeline Classification）
  根據語句推論承諾的預期完成時間。

此外，為使各參賽隊伍順利啟動模型開發專案，本單位提供相關技術支援與資源，包含：

- 三區巡迴課程：舉行免費巡迴課程，並同步開設線上場次，詳述競賽資訊，並示範範例程式碼（Sample Code）之使用方式，帶領參與者進行基礎實作。
- 範例代碼與討論區：待隊伍完成正式報名、進入競賽平臺後，本單位將統一提供範例程式碼，並開放官方「公開討論區」加入連結，供參賽者交流與提問。
- 官方諮詢表單：參賽隊伍於競賽期間如有任何疑義或遭遇困難，且不便透過官方討論區提問，可透過 Google 表單（表單連結）填寫相關申訴或諮詢內容，以獲得主辦單位的官方回覆或協助。

## 評估標準

「ESG 永續承諾驗證競賽 2026」（下稱本競賽）之參加者，須根據主辦單位提供之「訓練集」（Training Data）、「驗證集」（Validation Data）及「競賽測試集」（Test Data）共三種語料集，以及標註範例資料，開發能完成四大任務的 AI 模型，各競賽階段說明如下：

- 第一階段：主辦單位提供標註範例資料，並開放參賽報名。
- 第二階段：參賽隊伍須藉「訓練集」訓練模型，並對「驗證集」進行預測，最終將結果上傳至線上排名系統（Private Leaderboard）。
- 第三階段：主辦單位公佈完整的競賽測試集（Test Dataset）後，參賽隊伍須在截止時間前，至平臺上傳「競賽測試集」的預測結果，每日提交次數上限為 3 次。
  第四階段：競賽時間截止，系統會以 Test Dataset 的測試結果為最終排名依據。並公佈於 Private Leaderboard。

主辦單位將透過四項任務之加權綜合評分，來評比各參賽隊伍的系統效能和名次（詳細評分項目請參考「議題評估標準說明」與「議題評估公式」）。 各參賽隊伍應於規範時程內上傳相關報告書及原始程式碼，以證明無任何人為修正、作弊或抄襲之可能，未按時提交檔案之隊伍將不列入總排名。

## 議題評估標準說明

本競賽參考 ML-Promise 國際研究成果，採用多任務綜合評分機制。針對企業承諾驗證的四個核心任務，依據其難度與實務價值進行差異化權重設計，委員將以此成績高低為依據評估最後獎金名次。評分指標與權重分配說明如下：

1. 子任務一：承諾語句識別（Commitment Classification）
   - 評分權重：20%
   - 評分指標：Macro-F1 Score
   - 評估重點：評估模型識別 ESG 承諾語句的精確度與召回率平衡。
2. 子任務二：支持證據連結（Evidence Identification）
   - 評分權重：30%
   - 評分指標：Macro-F1 Score
   - 評估重點：評估模型判斷承諾是否具備充分支持證據的能力。
3. 子任務三：證據清晰度評估（Clarity Classification）
   - 評分權重：35%
   - 評分指標：Macro-F1 Score
   - 評估重點：評估「承諾-證據」對清晰度的三分類能力。
4. 子任務二：預期驗證時機推論（Timeline Classification）
   - 評分權重：15%
   - 評分指標：Macro-F1 Score
   - 評估重點：評估承諾適當驗證時間點的四分類預測能力。

## 議題評估公式

總分 = 承諾辨識 Macro-F1 × 0.20 +證據支持 Macro-F1 × 0.30 + 清晰度 Macro-F1 × 0.35 + 時機預測 Macro-F1 × 0.15

## Baseline Script

本專案已將官方 notebook baseline 整理為可重跑的 Python script：

```bash
python3 scripts/train_baseline.py --tensorboard
```

預設會讀取 `data/vpesg_4k_train_1000.json`，並將每次實驗保存到 `experiments/<timestamp>_<experiment-name>/`，包含：

- `config.json`：模型、訓練參數、資料分佈
- `history.json` / `history.csv`：每個 epoch 的 loss 與各任務 F1
- `metrics.json`：最佳模型在 validation split 上的最終指標
- `predictions_val.json`：validation 預測與 ground truth 對照
- `best_model.pt`：最佳 validation weighted score 的 checkpoint
- `tensorboard/`：使用 `--tensorboard` 時的 TensorBoard logs

查看 TensorBoard：

```bash
tensorboard --logdir experiments
```

常用調整範例：

```bash
python3 scripts/train_baseline.py \
  --experiment-name bert_weighted_len384 \
  --max-len 384 \
  --class-weight balanced \
  --tensorboard
```

---

## Reproduction Guide

### 1. 環境配置

- 作業系統：Linux；Python 3.12
- 硬體：單張 NVIDIA TITAN RTX（24 GB VRAM）、16 核 CPU、62 GB RAM
- 安裝：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 另需依 CUDA 版本安裝對應 PyTorch（本隊使用 torch 2.12.0 + CUDA 12.6）：
# pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 2. 資料放置

將官方三份資料放入 `data/`：`vpesg_4k_train_1000.json`、`vpesg4k_val_1000.json`、`vpesg4k_test_2000.json`。

### 3. 訓練三個骨幹（各 5-fold 交叉驗證、合併驗證集）

```bash
# (a) RoBERTa-large
python3 scripts/train_baseline.py --experiment-name roberta_large_mergeval_cv5 \
  --model-name hfl/chinese-roberta-wwm-ext-large \
  --merge-val --cv-folds 5 --cv-stratify-field evidence_status \
  --max-len 384 --batch-size 16 --lr 4e-5 --epochs 15 --class-weight balanced --fp16 --seed 42

# (b) MacBERT-large
python3 scripts/train_baseline.py --experiment-name macbert_large_mergeval_cv5 \
  --model-name hfl/chinese-macbert-large \
  --merge-val --cv-folds 5 --cv-stratify-field evidence_status \
  --max-len 384 --batch-size 16 --lr 4e-5 --epochs 15 --class-weight balanced --fp16 --seed 42

# (c) DeBERTa-v2-320M（不可用 fp16，batch 減半）
python3 scripts/train_baseline.py --experiment-name deberta_v2_320m_mergeval_cv5 \
  --model-name IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese \
  --merge-val --cv-folds 5 --cv-stratify-field evidence_status \
  --max-len 384 --batch-size 8 --lr 2e-5 --epochs 15 --class-weight balanced --seed 42
```

每個骨幹會在 `experiments/<timestamp>_<name>/fold{1..5}/best_model.pt` 產生 5 個 checkpoint，共 15 個。

### 4. Ensemble 推論 → 產生最終提交檔

```bash
ROB=$(ls -d experiments/*roberta_large_mergeval_cv5 | head -1)
MAC=$(ls -d experiments/*macbert_large_mergeval_cv5 | head -1)
DEB=$(ls -d experiments/*deberta_v2_320m_mergeval_cv5 | head -1)

python3 scripts/predict_test.py \
  --checkpoint $ROB/fold{1,2,3,4,5}/best_model.pt \
               $MAC/fold{1,2,3,4,5}/best_model.pt \
               $DEB/fold{1,2,3,4,5}/best_model.pt \
  --output submissions/ens_3way_raw.csv
```

輸出 `submissions/ens_3way_raw.csv`（2,000 列 + 表頭，欄位 `id,promise_status,verification_timeline,evidence_status,evidence_quality`），即為上傳 AIdea 之最終檔案。**注意：最終提交採未調參的純 soft-voting，不要套用 `tune_thresholds.py` 的偏置（理由見報告 §關鍵決策）。**

### 5. 重要模組的輸入／輸出

| 程式 | 輸入 | 輸出 |
|---|---|---|
| `scripts/train_baseline.py` | 訓練 JSON（`--merge-val` 時併入 val） | `experiments/<run>/`：`config.json`、`cv_summary.json`、各 fold 的 `metrics.json`、`best_model.pt` |
| `scripts/predict_test.py` | 1+ 個 `best_model.pt`、測試 JSON、（選用）`--bias-file` | 提交用 CSV |
| `scripts/tune_thresholds.py` | 某 CV run 目錄（可多個骨幹） | `class_bias.json`（**最終未採用**，保留作為過擬合分析紀錄） |

### 6. 關鍵參數速查

| 參數 | 值 | 說明 |
|---|---|---|
| `--merge-val` | 開 | 併入官方驗證集 → 2,000 筆訓練 |
| `--cv-folds` | 5 | 分層 5-fold CV |
| `--cv-stratify-field` | evidence_status | 分層依據欄位 |
| `--max-len` | 384 | tokenizer 截斷長度 |
| `--class-weight` | balanced | 對抗類別不平衡 |
| `--seed` | 42 | 固定亂數（含 CV 切分），確保可重現 |
