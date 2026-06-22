# 新聞標題黨偵測與內容破梗系統

本專案旨在建立一個中英文雙語的標題黨偵測與內容破梗系統。系統會先判斷使用者輸入的新聞標題是否具有標題黨特徵；若模型判定為標題黨，則進一步呼叫 Gemini API，根據新聞標題與內文生成一句話破梗，協助使用者在點擊前快速理解文章重點。

## 標題黨的定義

本專案的標題黨**不是指「事件誇張」或「內容離奇」**，而是指**標題與內文之間存在誇張、懸念、遮蔽或誤導**。即使事件離奇，只要標題忠實完整描述，仍視為非標題黨。據此歸納五類特徵：

1. **資訊落差**：標題刻意隱藏關鍵人/事/物/結果，逼讀者點擊解懸念。
2. **過度誇飾**：極端情緒詞，但事件層級撐不起（史上最、震驚、崩潰）。
3. **承諾落空**：標題承諾高價值資訊，內文只給常識/空泛/不相稱內容。
4. **格式化誘餌**：清單、排名、挑釁問句誘點（「第4個你想不到」）。
5. **軟文／業配**：標題以情感/知性語氣包裝，內文實為行銷軟文/廣告/業配（對應 WCD general-clickbait，T_LABEL=2）。

## 專案目標

- 偵測中文與英文新聞標題是否為標題黨。
- 將中文與英文資料集整理成統一的二元分類格式。
- 訓練並比較兩種模型：
  - TF-IDF + Logistic Regression baseline
  - fine-tuned multilingual transformer，採用 `xlm-roberta-base`
    （選用理由：跨語言共用表徵、sentence-pair 建模標題–內文落差、不依賴外部斷詞）
- 建立可互動的 Web 介面。
- 使用 Flask REST API 串接前端、分類模型與 Gemini API。
- 對判定為標題黨的文章生成一句話破梗。

## 預計使用技術

- 資料處理：Python, pandas
- Baseline 模型：scikit-learn
- Transformer 模型：PyTorch, Hugging Face Transformers
- 訓練環境：Google Colab GPU
- 後端：Flask REST API
- 前端：React + Vite
- 破梗生成：Gemini API

## 資料來源

本專案使用一個中文資料集與一個英文資料集。資料僅用於課程專案、模型訓練與實驗分析。

### 中文資料集

- 資料集：
  - WeChat Clickbait Dataset - `dataset\baseDataSet\all_labeled.csv`
- 參考專案：[WeSeewy/Chinese-Clickbait](https://github.com/WeSeewy/Chinese-Clickbait)
- 原始資料來源：[natsusaikou/WeChat-Clickbait](https://github.com/natsusaikou/WeChat-Clickbait)
- 資料內容：微信公眾號文章的 metadata、標題、內文與標題黨標註。

WCD 原始採三分類標註（`T_LABEL`），依原論文與官方程式碼定義如下：

| T_LABEL | 類別                        | 定義                                                                                                                       |
| ------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 0       | non-clickbait               | 非標題黨                                                                                                                   |
| 1       | malicious-clickbait（惡意） | 以低俗、色情標題惡意誘導點擊，內文含較多廣告與釣魚連結                                                                     |
| 2       | general-clickbait（一般）   | 政府、科普等發布者使用語言技巧（前指、清單體、文藝/知性語氣）吸引注意以擴大影響，多為行銷軟文/品牌內容，相對良性但仍屬誘導 |

中文標籤轉換方式（與 WCD 官方程式碼一致——[helper/utils.py](https://github.com/WeSeewy/Chinese-Clickbait/blob/main/helper/utils.py) 亦將 T_LABEL=2 映射為 label=1）：

```text
T_LABEL = 0      -> label = 0，非標題黨
T_LABEL = 1 or 2 -> label = 1，標題黨
```

> 註：T_LABEL=2（general-clickbait）涵蓋大量「文藝/知性語氣標題 + 行銷軟文內文」樣本，
> 例如標題用問句與引號反諷、內文為完整軟文最後標注「廣告」

### 英文資料集

- 資料集：Webis Clickbait Corpus 2017 / Webis-Clickbait-17
- 官方頁面：[Webis-Clickbait-17](https://webis.de/data/webis-clickbait-17.html)
- 下載頁面：[Zenodo Webis-Clickbait-17](https://zenodo.org/records/3346491)
- 預計使用檔案：`clickbait17-train-170630.zip`
- 資料內容：英文 Twitter/news clickbait 樣本與人工標註分數。

英文標籤轉換方式：

```text
truthMean >= 0.5 -> label = 1，標題黨
truthMean < 0.5  -> label = 0，非標題黨
```

大型 Webis archive 檔案，例如 `archives-clickbait17-train-170630-part[00-99].zip`，不是初版 supervised classifier 的必要檔案。初版訓練主要使用 `clickbait17-train-170630.zip`。

## 統一資料格式

資料前處理完成後，中英文資料會被轉成相同 CSV 欄位：

```csv
id,lang,title,content,label,source
```

預計輸出檔案：

```text
dataset/processed/unified_clickbait.csv
dataset/processed/train.csv
dataset/processed/valid.csv
dataset/processed/test.csv
```

## 資料分布與訓練風險

整體合併後的標籤分布看似接近平衡：

```text
Unified dataset: 36,194 rows
Label 0: 19,529
Label 1: 16,665
```

但是拆到語言層級後，資料存在嚴重的交叉不平衡（cross-lingual imbalance）：

```text
Chinese WCD: 16,656 rows
Label 0:  4,731
Label 1: 11,925

English Webis: 19,538 rows
Label 0: 14,798
Label 1:  4,740
```

也就是說，中文資料嚴重偏向「標題黨」，Label 1 大約是 Label 0 的 2.5 倍；英文資料則嚴重偏向「正常新聞」，Label 0 大約是 Label 1 的 3.1 倍。這會造成一個很大的訓練風險：如果直接 fine-tune `xlm-roberta-base`，模型可能偷懶學到語言捷徑，例如「看到中文就猜標題黨，看到英文就猜正常新聞」，而不是真正學會辨識標題黨的語意特徵。

目前資料切分結果如下：

```text
Train: 28,955 rows
lang: en 15,637 / zh 13,318
label: 0 15,631 / 1 13,324

Valid: 3,619 rows
lang: en 1,934 / zh 1,685
label: 0 1,916 / 1 1,703

Test: 3,620 rows
lang: en 1,967 / zh 1,653
label: 0 1,982 / 1 1,638
```

## 微調與資料增強策略

為了降低 cross-lingual imbalance 帶來的語言捷徑風險，fine-tuning 階段會加入三種資料增強與學習率調整策略：

- 回譯（back-translation）：需要呼叫翻譯 API，因此會做成獨立 preprocessing cell，先產生增強資料後再進入訓練流程。這可以讓同一語意以不同語言或不同表達方式出現，降低模型只依賴語言本身做判斷的機率。
- 其餘兩種文字增強：在訓練時 on-the-fly 進行，不直接改寫原始 CSV。這樣可以讓每個 epoch 看到略有變化的輸入，增加模型對詞彙擾動與句式變化的穩定性。
- `ReduceLROnPlateau`：當 validation loss 停滯時自動降低 learning rate，避免模型在後期用過大的更新幅度震盪，也能讓 fine-tuning 更穩定。

這些策略的目標不是單純增加資料量，而是讓模型更難依賴「語言 = label」的捷徑，迫使它學習標題與內文之間是否存在誇張、懸念、資訊缺口或誤導等真正與標題黨相關的語意特徵。

## XLM-RoBERTa 微調方法

模型訓練流程實作於 `notebooks/train_xlm_roberta_colab.ipynb`，設計目標是在 Google Colab GPU 上 fine-tune `xlm-roberta-base`，並將最佳 checkpoint 與評估結果存回 Google Drive。

### 訓練環境與輸入資料

- 執行環境：Google Colab GPU
- 主要套件：`transformers`, `datasets`, `accelerate`, `torch`, `scikit-learn`
- 模型：`xlm-roberta-base`
- 任務：二元分類，`0 = non-clickbait`，`1 = clickbait`
- 資料來源：`dataset/processed/train.csv`, `valid.csv`, `test.csv`
- 模型儲存位置：`models/xlm-roberta-clickbait/`
- 評估結果輸出：`results/transformer_metrics.json`

輸入格式使用 Hugging Face tokenizer 的 sentence pair 形式：

```text
title </s> content
```

也就是將新聞標題作為第一段文字，新聞內文作為第二段文字。這樣做的原因是標題黨判斷通常不只看標題本身，還需要比較標題與內文是否存在資訊落差、誇大、懸念或誤導。

### 主要超參數

```text
MODEL_NAME    = xlm-roberta-base
MAX_LENGTH    = 256
BATCH_SIZE    = 16
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-5
SEED          = 42
```

`MAX_LENGTH=256` 是在效能與 GPU 記憶體之間取平衡。若使用完整長文，訓練成本會明顯增加；本專案先保留標題與內文前段，讓模型能看到足夠上下文，同時維持 Colab 可訓練。

### 資料增強

Notebook 中使用三種增強策略，其中回譯是 preprocessing，其餘兩種是訓練時 on-the-fly 進行。

**1. 回譯 Back-translation**

回譯 cell 預設關閉：

```text
BACKTRANSLATE = False
BT_SAMPLE_RATIO = 0.3
```

若開啟，會使用翻譯工具對訓練集部分標題做回譯：

```text
中文：中文 -> 英文 -> 中文
英文：英文 -> 中文 -> 英文
```

回譯只處理 `title`，不處理 `content`。原因是標題是標題黨分類的核心訊號，而內文通常很長，直接翻譯成本高、速度慢，也容易遇到 API 額度限制。回譯後資料會輸出為：

```text
dataset/processed/train_augmented.csv
```

訓練時若偵測到 `train_augmented.csv` 存在，就優先使用該檔；否則使用原始 `train.csv`。

**2. Content 隨機截斷**

訓練集啟用 `CONTENT_CROP_PROB = 0.5`。每次取樣時，有 50% 機率從內文的隨機位置開始截取，而不是永遠從文章開頭截斷。這可以讓模型在不同 epoch 看到不同內文片段，降低模型只記住固定文章開頭模式的風險。

**3. Token Dropout**

訓練集啟用 `TOKEN_DROPOUT_PROB = 0.1`。模型輸入 token 化後，會隨機將部分非特殊 token 替換成 `<mask>`，但不會替換 padding 或特殊符號。這能讓模型不要過度依賴單一關鍵字，而是學習更穩定的語意特徵。

valid/test 不做任何資料增強，確保評估結果穩定且可重現。

### 訓練流程

Notebook 使用 `AutoModelForSequenceClassification` 載入 `xlm-roberta-base`，設定 `num_labels=2`。每個 batch 執行：

```text
forward -> loss -> backward -> gradient clipping -> optimizer step
```

訓練細節：

- optimizer：`AdamW`
- gradient clipping：`max_norm = 1.0`
- 每 100 step 印出 training loss 與目前 learning rate
- 每個 epoch 結束後在 validation set 上計算：
  - validation loss
  - accuracy
  - macro F1
- 只保存 validation macro F1 最好的 checkpoint

使用 macro F1 的原因是本任務存在 cross-lingual imbalance；只看 accuracy 可能會掩蓋某一類或某一語言表現很差的問題。

### Learning Rate Scheduler

訓練使用兩階段 learning rate 控制：

```text
Warmup: 前 10% training steps 使用 linear warmup
Plateau: warmup 後使用 ReduceLROnPlateau
```

`ReduceLROnPlateau` 設定：

```text
mode      = max
patience  = 1
factor    = 0.5
min_lr    = 1e-6
monitor   = validation macro F1
```

也就是說，如果 validation macro F1 連續一段時間沒有提升，learning rate 會降為原本的一半。這樣可以避免 fine-tuning 後期因 learning rate 過大而震盪，讓模型在接近收斂時用更小步伐更新。

### 測試與輸出

訓練完成後，notebook 會從 `models/xlm-roberta-clickbait/` 重新載入最佳 checkpoint，在 test set 上輸出：

- classification report
- confusion matrix
- test accuracy
- test macro F1

最後將結果存成：

```text
results/transformer_metrics.json
```

這份 JSON 可直接用於期末報告與簡報中的模型比較。

## 模型盲點與改進策略

在 demo 階段，最終分類模型暴露出兩個方向相反的盲點：

| 盲點         | 模型實際行為              | 範例                                                                       |
| ------------ | ------------------------- | -------------------------------------------------------------------------- |
| 語氣盲點     | 高信心**判標題黨（1）**   | LLM 改寫成平實標題、內文不變，模型仍判標題黨（false positive）             |
| 量級落差盲點 | 高信心**判非標題黨（0）** | 標題「我是世界冠軍」／內文「學校運動會第一名」，模型漏判（false negative） |

根本原因是 XLM-RoBERTa 以模式匹配為主，依賴主題關鍵字向量，無法分辨「同主題、語氣不同」的細微界線，也無法做「世界 > 學校」這種跨標題與內文的常識量級比較。

因此採取**兩個盲點分開治**的策略：

- **方法一（語氣對抗訓練，產出 G7）**：治語氣盲點。用 Gemini 把現有標題黨標題改寫成中性語氣、內文不動，標 0，與原標題（標 1）組成成對對抗資料併入訓練，強迫模型學「語氣」而非「主題」。語氣是表層語言特徵，模型學得動且標籤乾淨。
- **方法三（Gemini 輔助判斷 + 量級詞觸發器）**：治量級落差盲點。量級落差需要常識推理，交給 Gemini；不塞進訓練資料，避免模型學成「看到『世界／史上』就判標題黨」的大詞捷徑。

> 評估後未採用：方法二（純詞典判斷器）對語氣 false positive 是反向且易傷 recall；方法四（Bi-Encoder）對「同主題不同量級」的 cosine similarity 反而算成相似，不對症且需重訓、整體分數有下降風險。

## 系統流程

```text
使用者輸入新聞標題與內文
        |
        v
React 前端送出請求到 Flask REST API
        |
        v
Flask 載入訓練好的標題黨分類模型（G7）回傳分類結果與信心分數
        |
        v
Cascade 判斷是否呼叫 Gemini 輔助判斷：
  - 判標題黨（任何信心）        -> Gemini 輔助判斷 + 生成破梗
  - 判非標題黨且信心 < 0.80     -> Gemini 輔助判斷 + 生成破梗
  - 判非標題黨、高信心、命中量級詞 -> Gemini 輔助判斷 + 生成破梗
  - 判非標題黨、高信心、無量級詞   -> 跳過 Gemini（正常新聞，維持低成本）
        |
        v
前端顯示分類結果、信心分數，若為標題黨則顯示破梗內容
```

其中「量級詞觸發器」只是一個召喚開關：當標題出現超大量級宣稱詞（如世界、史上、全球、world's、ever）時，即使模型高信心判非標題黨，也強制送 Gemini 複核一次。判斷責任完全在 Gemini，詞典漏列頂多少複核幾筆，不會直接造成誤判。

## REST API 規劃

主要 endpoint：

```http
POST /api/analyze
```

Request：

```json
{
  "title": "string",
  "content": "string"
}
```

Response：

```json
{
  "baseline": {
    "label": 0,
    "is_clickbait": false,
    "confidence": 0.76
  },
  "transformer": {
    "label": 1,
    "is_clickbait": true,
    "confidence": 0.91,
    "gemini_used": true,
    "spoiler": "Gemini 生成的一句話破梗。"
  }
}
```

`gemini_used` 標記此次是否觸發 Gemini 輔助判斷（用於前端顯示 Cascade 標記）。當 Gemini 複核把標題黨判定翻轉為非標題黨時，`is_clickbait` 會更新為 `false` 且不輸出 `spoiler`。

## 注意事項

- Transformer 模型訓練建議在 Colab GPU 上進行
- 本機 CPU 可用於小樣本測試與最終 demo 的單筆推理

---

## Source code

[Github repo](https://github.com/Jx-study/NLP_Project)
