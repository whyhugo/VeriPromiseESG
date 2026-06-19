# VeriPromiseESG 本機實驗彙整

> 記錄截至 2026-06-17 在本機（WSL）所有訓練實驗的成果與分析，供遷移至新機器後繼續開發使用。

---

## 競賽背景

- **競賽**：AI CUP 2026 ESG 永續承諾驗證競賽
- **任務**：對繁體中文 ESG 報告書文句進行 4 個多分類子任務標注
- **評分公式**：

  ```
  總分 = evidence_quality × 0.35
       + evidence_status × 0.30
       + promise_status  × 0.20
       + verification_timeline × 0.15
  ```

  各子任務以 **Macro-F1** 計算。

- **資料集**（皆在 `data/`）：

  | 檔案 | 筆數 | 用途 |
  |---|---|---|
  | `vpesg_4k_train_1000.json` | 1,000 | 訓練 |
  | `vpesg4k_val_1000.json` | 1,000 | 官方驗證 |
  | `vpesg4k_test_2000.json` | 2,000 | 競賽測試（待提交） |

- **訓練腳本**：`scripts/train_baseline.py`
- **實驗目錄**：`experiments/<timestamp>_<name>/`

---

## 類別分佈（訓練集 1000 筆）

| Field | 類別 | 筆數 |
|---|---|---|
| promise_status | Yes | 814 |
| | No | 186 |
| verification_timeline | already | 366 |
| | between_2_and_5_years | 238 |
| | more_than_5_years | 197 |
| | N/A | 186 |
| | within_2_years | 13 |
| evidence_status | Yes | 677 |
| | N/A | 186 |
| | No | 137 |
| evidence_quality | Clear | 552 |
| | N/A | 323 |
| | Not Clear | 124 |
| | Misleading | **1** ← 極度稀少 |

文本長度：min=35 / avg=208 / max=816，超過 256 token 的樣本有 256 筆。

---

## 所有實驗結果

> 各實驗皆使用 80/20 train/val split（固定 random_state），訓練 10 epochs，lr=2e-5，dropout=0.1，warmup_ratio=0.1。

| 實驗 | 模型 | max_len | class_weight | fp16 | batch | seed | **Val Score** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| bert_baseline | bert-base-chinese | 256 | none | — | 8 | 42 | 0.5427 |
| bert_len384 | bert-base-chinese | 384 | none | — | 8 | 42 | 0.5690 |
| bert_len512 | bert-base-chinese | 512 | none | — | 8 | 42 | 0.5478 |
| bert_len384_class_weight | bert-base-chinese | 384 | balanced | — | 8 | 42 | 0.6009 |
| macbert_base_len384 | chinese-macbert-base | 384 | none | — | 8 | 42 | 0.5817 |
| roberta_wwm_len384 | chinese-roberta-wwm-ext | 384 | none | — | 8 | 42 | 0.5936 |
| bert_len384_cw_seed13 | bert-base-chinese | 384 | balanced | — | 8 | 13 | 0.5793 |
| bert_len384_cw_seed42 | bert-base-chinese | 384 | balanced | — | 8 | 42 | 0.6009 |
| bert_len384_cw_seed77 | bert-base-chinese | 384 | balanced | — | 8 | 77 | 0.5980 |
| bert_len384_cw_seed123 | bert-base-chinese | 384 | balanced | — | 8 | 123 | 0.5642 |
| bert_len384_cw_seed2026 | bert-base-chinese | 384 | balanced | — | 8 | 2026 | 0.5856 |
| roberta_wwm_len384_cw_seed13 | chinese-roberta-wwm-ext | 384 | balanced | — | 8 | 13 | 0.5841 |
| roberta_wwm_len384_cw_seed42 | chinese-roberta-wwm-ext | 384 | balanced | — | 8 | 42 | 0.5936 |
| **roberta_wwm_len384_cw_seed77** | **chinese-roberta-wwm-ext** | **384** | **balanced** | **—** | **8** | **77** | **0.6310** ✓ |
| roberta_wwm_len384_cw_seed2026_bs4_fp16 | chinese-roberta-wwm-ext | 384 | balanced | ✓ | 4 | 2026 | 0.5644 |
| roberta_wwm_len384_cw_seed123_bs4_fp16 (×2) | chinese-roberta-wwm-ext | 384 | balanced | ✓ | 2–4 | 123 | 未完成 |

---

## 最佳模型詳細分析

### 設定（roberta_wwm_len384_cw_seed77）

```
model_name : hfl/chinese-roberta-wwm-ext
max_len    : 384
batch_size : 8
epochs     : 10
lr         : 2e-5
class_weight: balanced
seed       : 77
fp16       : False
best_epoch : 8
```

### 各 Field Macro-F1

| Field | Macro-F1 | 權重 | 加權貢獻 |
|---|:---:|:---:|:---:|
| promise_status | 0.7865 | 20% | 0.1573 |
| verification_timeline | 0.6598 | 15% | 0.0990 |
| evidence_status | 0.7322 | 30% | 0.2197 |
| evidence_quality | 0.4429 | **35%** | 0.1550 |
| **Total** | | | **0.6310** |

### Epoch 別 Val Score

| Epoch | Train Loss | Val Score |
|:---:|:---:|:---:|
| 1 | 4.5996 | 0.3273 |
| 2 | 3.7688 | 0.5106 |
| 3 | 2.8814 | 0.5566 |
| 4 | 2.0860 | 0.5938 |
| 5 | 1.5339 | 0.6064 |
| 6 | 1.1485 | 0.5817 |
| 7 | 0.8600 | 0.5885 |
| **8** | **0.6709** | **0.6310** ✓ |
| 9 | 0.5220 | 0.6202 |
| 10 | 0.4401 | 0.6172 |

---

## 關鍵結論

1. **`class_weight=balanced` 必開**：幾乎在所有條件下帶來 +4–7% 提升，原因是各 field 的 minority class 占比偏低（No: 18.6%、Misleading: 0.1%）。

2. **`max_len=384` 最佳**：訓練資料平均 208 token，超過 256 的有 256 筆，加長至 384 有效；512 反而略差，推測因 batch 中 padding 增加而影響訓練效率。

3. **模型強弱**：`chinese-roberta-wwm-ext` > `bert-base-chinese` > `chinese-macbert-base`（base 尺寸下）。

4. **Seed 差異顯著**：相同模型 + balanced，RoBERTa 跑 4 個 seed 的分數區間達 0.584–0.631，其中 seed77 異常好，不宜過度信賴單一 seed 結果。

5. **fp16 + 小 batch 有害**：bs=4 + fp16 的結果（0.5644）明顯低於 bs=8 + fp16=off（0.6310），推測是小 batch 導致梯度不穩定，加上 fp16 精度損失加劇。

6. **evidence_quality 是核心瓶頸**：權重最高（35%）但 Macro-F1 最低（0.44），主因是 Misleading 類別僅 1 筆，模型完全無法學習。

---

## 新機器建議下一步

> 新機器配備 **24GB VRAM**，可直接跑 Large 模型 + 大 batch，無需 gradient accumulation。

### VRAM 使用量估算（fp16）

| 模型 | max_len | batch | 估計 VRAM |
|---|:---:|:---:|:---:|
| roberta-wwm-ext (base) | 384 | 8 | ~4 GB |
| roberta-wwm-ext (base) | 384 | 32 | ~10 GB |
| roberta-wwm-ext-large | 384 | 8 | ~8 GB |
| roberta-wwm-ext-large | 384 | 16 | ~14 GB |
| roberta-wwm-ext-large | 512 | 8 | ~11 GB |
| roberta-wwm-ext-large | 512 | 16 | ~20 GB |

### 建議執行順序

```text
Step 1. roberta_large + len384 + bs16 + seed77   ← 對標最佳 base 結果
Step 2. roberta_large + len512 + bs8  + seed77   ← 驗證更長 context 有無幫助
Step 3. 挑更好的設定，跑 seed 13/42/77/123/2026 五組
Step 4. 五模型 ensemble → 提交驗證集
```

---

### 優先嘗試（高 ROI）

**1. Large 模型 + 大 batch（首選）**

```bash
python3 scripts/train_baseline.py \
  --experiment-name roberta_large_len384_bs16_cw \
  --model-name hfl/chinese-roberta-wwm-ext-large \
  --max-len 384 \
  --batch-size 16 \
  --lr 4e-5 \
  --epochs 15 \
  --class-weight balanced \
  --fp16 \
  --seed 77
  --tensorboard
```

> batch 8→16 時，依線性 scaling rule lr 同步調整 2e-5→4e-5。`hfl/chinese-macbert-large` 也值得比較。

#### 2. Large 模型 + max_len 512（次選）

```bash
python3 scripts/train_baseline.py \
  --experiment-name roberta_large_len512_bs8_cw \
  --model-name hfl/chinese-roberta-wwm-ext-large \
  --max-len 512 \
  --batch-size 8 \
  --lr 2e-5 \
  --epochs 15 \
  --class-weight balanced \
  --fp16 \
  --seed 77
  --tensorboard
```

**3. 合併官方驗證集進訓練**

官方驗證集 `vpesg4k_val_1000.json` 有 1000 筆已標注資料，合併後訓練集達 2000 筆，直接倍增資料量，預期能有效提升稀少類別的學習。

**4. 對 evidence_quality 做 oversampling**

Misleading 類別僅 1 筆，可考慮：
- 將 Misleading 從訓練中排除，轉為 3 分類（Clear / Not Clear / N/A）
- 或對 Misleading 做重複 oversampling（×20 倍）搭配 class_weight

### 穩健提升

**5. Multi-seed ensemble**

對 seed 13/42/77/123/2026 各訓練一次，inference 時對各 field 的 logit/softmax 平均後 argmax，預期 +1–3%。一次約 30–40 分鐘，五組 3 小時內可完成。

```bash
for seed in 13 42 77 123 2026; do
  python3 scripts/train_baseline.py \
    --experiment-name roberta_large_len384_cw_seed${seed} \
    --model-name hfl/chinese-roberta-wwm-ext-large \
    --max-len 384 --batch-size 16 --lr 4e-5 \
    --epochs 15 --class-weight balanced --fp16 \
    --seed ${seed}
    --tensorboard
done
```

#### 6. 延長訓練至 15 epochs（已納入上方指令）

seed77 最佳點在 epoch 8，loss 仍持續下降，延長至 15 epochs 並搭配 early stopping（patience=3）可充分利用大模型容量。

**7. Per-class threshold 調整**

推論後對每個 field 的每個類別獨立調整 softmax 閾值，針對性提升 minority class 的 recall，直接優化 Macro-F1。

### 進階方向

**8. 使用 LLM 做 evidence_quality soft label**

用 Claude 或 GPT-4 對訓練集 evidence_quality 欄位打 soft label，補充 Misleading 類別的監督訊號。

**9. 更換 backbone 至 DeBERTa**

`IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese` 在繁體中文任務上表現往往優於 RoBERTa，24GB 可直接跑 large 版本。

### fp16 使用注意

- 24GB 空間充裕，fp16 僅帶來加速，不會重蹈本機 bs=4 的梯度不穩問題。
- 若跑 large + len512 + bs16（~20GB）接近上限，可先降到 bs=12 觀察穩定性。

---

## 檔案結構備忘

```
VeriPromiseESG/
├── data/
│   ├── vpesg_4k_train_1000.json     # 訓練集（1000）
│   ├── vpesg4k_val_1000.json        # 官方驗證集（1000）
│   └── vpesg4k_test_2000.json       # 競賽測試集（2000，無標籤）
├── scripts/
│   └── train_baseline.py            # 主訓練腳本
├── experiments/
│   └── <timestamp>_<name>/
│       ├── config.json              # 訓練設定
│       ├── metrics.json             # 最終 val 指標（weighted_score + per-field）
│       ├── history.json             # 每 epoch 的 loss / score
│       ├── predictions_val.json     # val 預測結果對照
│       └── best_model.pt            # 最佳 checkpoint
└── requirements.txt
```

---
---

# 新機器實驗與競賽提交彙整（2026-06-18，競賽最終日）

> 接續上方「舊機器」報告。本段記錄新機器（24GB VRAM）上的 Large 模型、交叉驗證、
> ensemble、threshold tuning，以及四次 AIdea 提交的實測結果與**關鍵教訓**。

## 最終成績

- **Public Leaderboard：0.6176，名次 31/143（前 22%）。**
- 最終排名以 **Private Leaderboard（完整 Test Dataset）** 為準，賽後公布。
- 獲獎門檻：前 30 名 **且** Private 前 25%。Public 31 名，差一名進前 30。
- 平台規則：測試集每日提交上限 3 次；本隊理解為「以最後一次提交計分」，故最終提交
  為 `ens_3way_raw.csv`。

## 資料策略（關鍵改動）

實際資料為 **1000 train / 1000 val / 2000 test**，且 train 與 val **皆有標籤**：

| 階段 | 用途 | 資料 |
|---|---|---|
| 模型選擇/調參 | 5-fold CV，取折平均±標準差 | train+val 合併 = **2000 筆 labeled** |
| 最終提交模型 | 用全 2000 筆訓練（或各 fold checkpoint 做 ensemble） | 2000 labeled |
| 推論/上傳 | 預測 → submission CSV | test **2000 筆 unlabeled** |

合併 val (`--merge-val`) 把訓練資料翻倍；改用 5-fold CV (`--cv-folds`) 取代會被單一 seed
誤導的 hold-out（舊機器同設定 5-seed 全距達 0.558–0.643）。

## 工具新增

- `scripts/train_baseline.py`：新增 `--merge-val`、`--cv-folds N`、`--cv-stratify-field`、
  `--full-train`。CV 模式輸出 `cv_summary.json`（各折分數 + 平均±標準差）。
- `scripts/predict_test.py`：載入 1+ checkpoint，對 test 2000 筆 soft-voting（softmax 平均）
  ，輸出符合 `sample_submission_format.csv` 的 CSV；`--bias-file` 套用 per-class 偏置。
- `scripts/tune_thresholds.py`：用各 fold checkpoint 重建 leak-free OOF 機率，對每類別做
  座標上升調整偏置以最大化 macro-F1；支援多 backbone 合併（同 seed fold 對齊）。

## 單模 5-fold CV 結果（merge-val，max_len384, bs16, 15ep, balanced）

| 模型 | lr | fp16 | **CV OOF weighted** | ps | vt | es | eq |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chinese-roberta-wwm-ext-**large** | 4e-5 | ✓ | **0.6224 ± 0.0165** | 0.809 | 0.599 | 0.697 | 0.462 |
| chinese-macbert-**large** | 4e-5 | ✓ | 0.6095 ± 0.0149 | 0.800 | 0.592 | 0.683 | 0.445 |
| Erlangshen-DeBERTa-v2-320M | 2e-5 | ✗ | 0.5948 ± 0.0182 | 0.785 | 0.532 | 0.680 | 0.440 |

> RoBERTa-large 最強；macbert/deberta 單模較弱，僅作 ensemble 多樣性用。

## Ensemble OOF（soft-voting，fold 對齊）

| 組合 | raw OOF | tuned OOF |
|---|:---:|:---:|
| roberta + macbert | 0.6252 | 0.6366 |
| roberta + macbert + deberta | 0.6323 | 0.6414 |

## 四次提交 vs 實際 LB —— calibration 分析（最重要）

| 提交檔 | 時間 | 本地 OOF | **Public LB** | OOF→LB 落差 |
|---|---|:---:|:---:|:---:|
| `roberta_large_cv5_ensemble.csv`（roberta raw） | 06-17 23:55 | 0.6239 | 0.6168 | **−0.007** ✓ |
| `ens_rob_mac_tuned.csv`（tuned） | 06-18 22:14 | 0.6366 | 0.6117 | **−0.025** ✗ |
| `ens_3way_tuned.csv`（tuned） | 06-18 23:05 | 0.6414 | 0.6172 | **−0.024** ✗ |
| `ens_3way_raw.csv`（raw，最終） | 06-18 23:41 | 0.6323 | **0.6176** | **−0.015** |

## 關鍵教訓

1. **Per-class threshold tuning 會過擬合，不可信。** 偏置是在「拿來評估的同一份 OOF」上
   座標上升最大化 macro-F1（in-sample 最佳化），灌水 ~0.01–0.015 在 test 上不存在。最明顯
   證據：`evidence_quality` 的偏置給 `Misleading +0.8`——而 Misleading 全資料僅 2 筆、OOF
   幾無 support，純啃噪音，到 test 反而逼出錯誤預測。**raw（argmax）的 OOF 才可信。**

2. **Calibration 對 raw 單模最準（−0.007），對 merge/ensemble 變大（−0.015）。** 合併 val +
   多模型雖提高 OOF，增益大多沒轉移到 test。所有候選實測擠在 **0.612–0.618**，差距在噪音內。

3. **Ensembling 方向正確但增益小**：LB 上 3-way > rob+mac，與 OOF 一致，但絕對提升有限。

4. **evidence_quality 是真正的天花板**（權重 35%，卡 ~0.44）。4 類 macro-F1 中 Misleading 全
   資料僅 2 筆 → 恆為 0 → 上限 0.75。標準監督式訓練無解。

5. **DeBERTa-v2 兩個坑**：(a) 與 AMP fp16 不相容（`unscale FP16 gradients`）→ 必須 fp32；
   (b) Erlangshen checkpoint 權重存成 fp16，載入後 encoder 輸出 Half 與 fp32 head 衝突
   （`mat1 and mat2 must have the same dtype`）→ 已在 `MultiTaskClassifier` 內 `.float()` 修正。

## 下次該怎麼做（若有後續賽季）

- **放棄在評估集上做 threshold tuning**；若要調，須用「巢狀 CV / 獨立 holdout」驗證偏置能轉移。
- **主攻 evidence_quality**：用 LLM 對訓練集打 soft label 擴充 Misleading（唯一能破 0.75 上限），
  或退化為 3 類 + 規則補 Misleading。
- 以 **raw ensemble 的 CV OOF** 作唯一決策指標（已驗證最可信）。
- 早點提交以累積 calibration 樣本，別把所有提交擠在最後一天。
