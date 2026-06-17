"""G9 雙塔（DualTower）共用模組：供 eval_edge_cases / explain_tokens 載入 G9。

G9 與 G5/G7/G8 的關鍵差異：
  - G5/G7/G8：單塔，tokenizer(title, content) 拼成一條序列 → AutoModelForSequenceClassification。
  - G9：雙塔，title 與 content 各自 tokenize（共享 XLM-R backbone），mean-pool 後合併
        [h_t; h_c; |h_t-h_c|; h_t⊙h_c] → 分類頭。backbone 與分類頭分開存：
          <dir>/                  ← AutoModel.save_pretrained 的 backbone + tokenizer
          <dir>/classifier_head.pt ← model.classifier 的 state_dict

本模組提供統一的 `predict_proba(title, content) -> P(clickbait)` 介面，
讓下游評估程式對單塔/雙塔用同一條 code path（is_dualtower 判斷一次即可）。

雙塔超參數須與 train_dualtower_colab.ipynb 一致，否則 tokenize 長度對不上。
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# 與 train_dualtower_colab.ipynb 的 config cell 一致
TITLE_MAX_LEN = 64
CONTENT_MAX_LEN = 256
CLICKBAIT_IDX = 1

# 單塔（G5/G7/G8）拼接序列長度，與 eval/explain 既有慣例一致
SINGLE_MAX_LEN = 256


class DualTowerClassifier(nn.Module):
    """Siamese 雙塔（共享 backbone），合併 [h_t; h_c; |h_t-h_c|; h_t⊙h_c]。

    定義須與 train_dualtower_colab.ipynb 的 cell-model 完全一致，否則
    classifier_head.pt 的 state_dict 形狀對不上。
    """

    def __init__(self, model_name, hidden_size=768, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(out.last_hidden_state, attention_mask)

    def encode_embeds(self, inputs_embeds, attention_mask):
        """從預先算好的 word embeddings 編碼。供 Integrated Gradients 用：
        IG 需要對 embedding 求梯度，而雙塔共享 backbone 的 embedding 層無法用
        LayerIntegratedGradients 區分兩塔（兩塔序列長度不同會互相污染），
        故改在外部各自算 embeds、以 inputs_embeds 餵入、用 plain IntegratedGradients。"""
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return self.mean_pool(out.last_hidden_state, attention_mask)

    def forward(self, title_input_ids, title_attention_mask,
                content_input_ids, content_attention_mask):
        h_t = self.encode(title_input_ids, title_attention_mask)
        h_c = self.encode(content_input_ids, content_attention_mask)
        v = torch.cat([h_t, h_c, (h_t - h_c).abs(), h_t * h_c], dim=-1)
        return self.classifier(v)

    def forward_embeds(self, title_embeds, title_attention_mask,
                       content_embeds, content_attention_mask):
        """以 inputs_embeds 形式做完整前向（供 IG）。"""
        h_t = self.encode_embeds(title_embeds, title_attention_mask)
        h_c = self.encode_embeds(content_embeds, content_attention_mask)
        v = torch.cat([h_t, h_c, (h_t - h_c).abs(), h_t * h_c], dim=-1)
        return self.classifier(v)


def is_dualtower(path):
    """雙塔 checkpoint 的判別特徵：同目錄存在 classifier_head.pt。"""
    from pathlib import Path
    return (Path(path) / "classifier_head.pt").exists()


class ModelWrapper:
    """統一封裝單塔/雙塔，對外只暴露 predict_proba(title, content)。

    下游評估程式不必再分 G5/G7/G8 vs G9 兩套 code path。
    """

    def __init__(self, name, path, device=None, base_model="xlm-roberta-base"):
        self.name = name
        self.path = str(path)
        self.device = device or torch.device("cpu")
        self.dual = is_dualtower(path)
        self.tok = AutoTokenizer.from_pretrained(self.path)

        if self.dual:
            self.mdl = DualTowerClassifier(base_model)
            # backbone 用 checkpoint 內的權重覆蓋（train notebook 以 save_pretrained 存於同目錄）
            self.mdl.encoder = AutoModel.from_pretrained(self.path)
            self.mdl.classifier.load_state_dict(
                torch.load(f"{self.path}/classifier_head.pt", map_location=self.device)
            )
        else:
            self.mdl = AutoModelForSequenceClassification.from_pretrained(self.path)

        self.mdl.to(self.device).eval()

    def _proba_single(self, title, content):
        enc = self.tok(
            str(title), str(content or ""),
            truncation=True, max_length=SINGLE_MAX_LEN, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            return float(self.mdl(**enc).logits.softmax(dim=-1)[0, CLICKBAIT_IDX])

    def _proba_dual(self, title, content):
        enc_t = self.tok(
            str(title), truncation=True, max_length=TITLE_MAX_LEN,
            padding="max_length", return_tensors="pt",
        ).to(self.device)
        enc_c = self.tok(
            str(content or ""), truncation=True, max_length=CONTENT_MAX_LEN,
            padding="max_length", return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.mdl(
                title_input_ids=enc_t["input_ids"],
                title_attention_mask=enc_t["attention_mask"],
                content_input_ids=enc_c["input_ids"],
                content_attention_mask=enc_c["attention_mask"],
            )
            return float(logits.softmax(dim=-1)[0, CLICKBAIT_IDX])

    def predict_proba(self, title, content):
        return self._proba_dual(title, content) if self.dual else self._proba_single(title, content)

    def predict(self, title, content, threshold=0.5):
        return int(self.predict_proba(title, content) >= threshold)

    def close(self):
        import gc
        del self.mdl, self.tok
        gc.collect()
