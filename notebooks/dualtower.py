"""G9 雙塔（DualTower）共用模組：供 eval_edge_cases / explain_tokens 載入 G9/G9v2。

G9v1 vs G9v2 差異：
  G9v1：合併 [h_t; h_c; |h_t-h_c|; h_t⊙h_c]，symmetric。
  G9v2：
    C1 常開：合併 [h_t; h_c_attn; h_t-h_c_attn; h_c_attn-h_t]（不對稱有號方向）
    C2 常開：TitleAwareAttentionPool（title 向量為 query 對 content token 加權）
    C3 flag：CrossAttentionAligner（塔頂一層 cross-attention title↔content）
    C4 flag：consistency loss（訓練時用，推理無影響）

checkpoint 目錄結構：
  <dir>/                     ← AutoModel.save_pretrained 的 backbone + tokenizer
  <dir>/classifier_head.pt   ← model.classifier 的 state_dict
  <dir>/attn_pool.pt         ← G9v2 TitleAwareAttentionPool（v2 才有）
  <dir>/cross_align.pt       ← G9v2 CrossAttentionAligner（C3=True 才有）
  <dir>/g9v2_config.json     ← G9v2 flag config（v2 才有）

本模組提供統一的 `predict_proba(title, content) -> P(clickbait)` 介面，
以及供 Integrated Gradients 用的 `forward_embeds`。
"""
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

TITLE_MAX_LEN  = 64
CONTENT_MAX_LEN = 256
CLICKBAIT_IDX  = 1
SINGLE_MAX_LEN = 256


# ── G9v2 子模組 ─────────────────────────────────────────────────────────────

class TitleAwareAttentionPool(nn.Module):
    """C2：以 title pooled 向量為 query，對 content token 加權 pooling。"""

    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale   = hidden_size ** -0.5

    def forward(self, h_title_pooled, content_hidden, content_mask):
        q = self.q_proj(h_title_pooled).unsqueeze(1)   # (B, 1, H)
        k = self.k_proj(content_hidden)                  # (B, L, H)
        scores = (q * k).sum(-1) * self.scale            # (B, L)
        scores = scores.masked_fill(content_mask == 0, float("-inf"))
        attn   = scores.softmax(dim=-1)
        return (attn.unsqueeze(-1) * content_hidden).sum(1)  # (B, H)


class CrossAttentionAligner(nn.Module):
    """C3：一層 multi-head cross-attention（title 為 Q, content 為 K/V）。"""

    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha  = nn.MultiheadAttention(hidden_size, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, title_hidden, title_mask, content_hidden, content_mask):
        content_key_mask = (content_mask == 0)
        attn_out, _ = self.mha(
            query=title_hidden, key=content_hidden, value=content_hidden,
            key_padding_mask=content_key_mask,
        )
        attn_out = self.norm(attn_out + title_hidden)
        title_mask_f = title_mask.unsqueeze(-1).float()
        return (attn_out * title_mask_f).sum(1) / title_mask_f.sum(1).clamp(min=1e-9)


# ── DualTowerClassifier（v1 + v2 統一類別）────────────────────────────────────

class DualTowerClassifier(nn.Module):
    """Siamese 雙塔分類器，支援 G9v1 與 G9v2 兩種合併方式。

    v2=False（G9v1）：[h_t; h_c; |h_t-h_c|; h_t⊙h_c]，4×768 = 3072
    v2=True  + cross=False：C1+C2，[h_t; h_c_attn; h_t-h_c_attn; h_c_attn-h_t]，3072
    v2=True  + cross=True ：C1+C2+C3，上述 + cross_align，5×768 = 3840
    """

    def __init__(self, model_name, hidden_size=768, dropout=0.1,
                 v2=False, use_cross=False):
        super().__init__()
        self.v2        = v2
        self.use_cross = use_cross and v2
        self.encoder   = AutoModel.from_pretrained(model_name)

        if v2:
            self.attn_pool = TitleAwareAttentionPool(hidden_size)
            if self.use_cross:
                self.cross_align = CrossAttentionAligner(hidden_size, num_heads=8,
                                                          dropout=dropout)
                merge_dim = hidden_size * 5
            else:
                merge_dim = hidden_size * 4
        else:
            merge_dim = hidden_size * 4

        self.classifier = nn.Sequential(
            nn.Linear(merge_dim, hidden_size),
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

    def encode_full(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        return self.mean_pool(hidden, attention_mask), hidden

    def _merge_v2(self, h_t, h_t_hidden, t_mask, h_c_hidden, c_mask):
        h_c_attn = self.attn_pool(h_t, h_c_hidden, c_mask)
        parts = [h_t, h_c_attn, h_t - h_c_attn, h_c_attn - h_t]
        if self.use_cross:
            align = self.cross_align(h_t_hidden, t_mask, h_c_hidden, c_mask)
            parts.append(align)
        return torch.cat(parts, dim=-1)

    def encode_embeds_full(self, inputs_embeds, attention_mask):
        """從 word embeddings 編碼，回傳 (pooled, token_hidden)（供 IG + v2 路徑）。"""
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        return self.mean_pool(hidden, attention_mask), hidden

    def forward(self, title_input_ids, title_attention_mask,
                content_input_ids, content_attention_mask):
        if self.v2:
            h_t, h_t_hid = self.encode_full(title_input_ids, title_attention_mask)
            _,   h_c_hid  = self.encode_full(content_input_ids, content_attention_mask)
            v = self._merge_v2(h_t, h_t_hid, title_attention_mask,
                                h_c_hid, content_attention_mask)
        else:
            h_t = self.encode(title_input_ids,   title_attention_mask)
            h_c = self.encode(content_input_ids, content_attention_mask)
            v   = torch.cat([h_t, h_c, (h_t - h_c).abs(), h_t * h_c], dim=-1)
        return self.classifier(v)

    def forward_embeds(self, title_embeds, title_attention_mask,
                       content_embeds, content_attention_mask):
        """以 inputs_embeds 形式做前向（供 IG，走與 forward() 相同的計算路徑）。

        精度說明：
          - eval 數字（eval_edge_cases 的機率）由 forward() 經 input_ids → 內建 embedding → encoder。
          - IG attribution（explain_tokens 的歸因）由此函式經 pre-extracted embeddings → encoder。
          - v1 路徑（mean-pool only）：兩者計算圖一致，差異僅 embedding lookup 浮點順序，已驗證 < 1e-5。
          - v2 + C3（cross_align）路徑：**等價性尚未驗證**。IG 端會把固定塔 expand 成
            (n_steps×1) 餵入 cross-attention，expand 產生非 contiguous 張量，
            nn.MultiheadAttention 對非 contiguous 輸入是否數值正確無法靜態確認。
            → explain_tokens 的雙塔 attribute 內含 forward vs forward_embeds 數值斷言；
              拿到 G9v2(C3) checkpoint 後須先通過該斷言，否則 v2 的 IG 結果不可用於報告。
        """
        if self.v2:
            h_t, h_t_hid = self.encode_embeds_full(title_embeds,   title_attention_mask)
            _,   h_c_hid  = self.encode_embeds_full(content_embeds, content_attention_mask)
            v = self._merge_v2(h_t, h_t_hid, title_attention_mask,
                                h_c_hid, content_attention_mask)
        else:
            h_t, _ = self.encode_embeds_full(title_embeds,   title_attention_mask)
            h_c, _ = self.encode_embeds_full(content_embeds, content_attention_mask)
            v = torch.cat([h_t, h_c, (h_t - h_c).abs(), h_t * h_c], dim=-1)
        return self.classifier(v)


# ── checkpoint 判別與載入工具 ────────────────────────────────────────────────

def is_dualtower(path):
    """雙塔 checkpoint 判別特徵：同目錄存在 classifier_head.pt。"""
    return (Path(path) / "classifier_head.pt").exists()


def _load_g9v2_config(path):
    """讀取 g9v2_config.json 或 g9_config.json；不存在則回傳 None（表示 v1）。"""
    for name in ("g9v2_config.json", "g9_config.json"):
        cfg_path = Path(path) / name
        if cfg_path.exists():
            return json.loads(cfg_path.read_text(encoding="utf-8"))
    return None


def load_dualtower(path, device, base_model="xlm-roberta-base"):
    """從 checkpoint 目錄載入 DualTowerClassifier（自動偵測 v1 vs v2）。"""
    cfg = _load_g9v2_config(path)
    if cfg is not None:
        # G9v2
        use_cross = cfg.get("USE_CROSS_ATTENTION", False)
        model = DualTowerClassifier(base_model, v2=True, use_cross=use_cross)
        model.encoder = AutoModel.from_pretrained(path)
        model.classifier.load_state_dict(
            torch.load(f"{path}/classifier_head.pt", map_location=device))
        model.attn_pool.load_state_dict(
            torch.load(f"{path}/attn_pool.pt", map_location=device))
        if use_cross:
            model.cross_align.load_state_dict(
                torch.load(f"{path}/cross_align.pt", map_location=device))
    else:
        # G9v1
        model = DualTowerClassifier(base_model, v2=False)
        model.encoder = AutoModel.from_pretrained(path)
        model.classifier.load_state_dict(
            torch.load(f"{path}/classifier_head.pt", map_location=device))
    return model.to(device).eval()


class ModelWrapper:
    """統一封裝單塔/雙塔，對外只暴露 predict_proba(title, content)。"""

    def __init__(self, name, path, device=None, base_model="xlm-roberta-base"):
        self.name   = name
        self.path   = str(path)
        self.device = device or torch.device("cpu")
        self.dual   = is_dualtower(path)
        self.tok    = AutoTokenizer.from_pretrained(self.path)

        if self.dual:
            self.mdl = load_dualtower(path, self.device, base_model)
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

    def predict_proba_batch(self, titles, contents, batch_size=32):
        """Batched inference; returns list of float probabilities."""
        probas = []
        for i in range(0, len(titles), batch_size):
            t_batch = [str(t) for t in titles[i:i + batch_size]]
            c_batch = [str(c or "") for c in contents[i:i + batch_size]]
            if self.dual:
                enc_t = self.tok(
                    t_batch, truncation=True, max_length=TITLE_MAX_LEN,
                    padding=True, return_tensors="pt",
                ).to(self.device)
                enc_c = self.tok(
                    c_batch, truncation=True, max_length=CONTENT_MAX_LEN,
                    padding=True, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    logits = self.mdl(
                        title_input_ids=enc_t["input_ids"],
                        title_attention_mask=enc_t["attention_mask"],
                        content_input_ids=enc_c["input_ids"],
                        content_attention_mask=enc_c["attention_mask"],
                    )
            else:
                enc = self.tok(
                    t_batch, c_batch,
                    truncation=True, max_length=SINGLE_MAX_LEN,
                    padding=True, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    logits = self.mdl(**enc).logits
            probas.extend(logits.softmax(dim=-1)[:, CLICKBAIT_IDX].tolist())
        return probas

    def predict_batch(self, titles, contents, threshold=0.5, batch_size=32):
        """Batched inference; returns list of int labels."""
        return [int(p >= threshold)
                for p in self.predict_proba_batch(titles, contents, batch_size)]

    def close(self):
        import gc
        del self.mdl, self.tok
        gc.collect()
