"""Token-level 判斷依據分析（報告 §6B）：模型憑哪些 token 判 clickbait？

**記憶體安全版**：拆成獨立 task，一次只跑一個，跑完即釋放（避免一次塞三件事爆記憶體）。
IG 一律設 internal_batch_size（把 n_steps 步插值分批，峰值砍到約 1/12），跑前檢查可用記憶體。

工具分層：
  1. 主力：Integrated Gradients（Captum LayerIntegratedGradients，作用在 embedding 層）
     → 每個 token 把預測推向/推離 clickbait 的貢獻量，畫紅藍 heatmap。
  2. 驗證：Occlusion（逐 token 刪除算 Δp）→ 系統化版的 probe_validate「拿掉『竟』就翻轉」。
  3.（Attention 留給 bertviz 當附錄，本腳本不畫——attention ≠ 解釋。）

四個 task（綁定 BugReport 敘事，各自獨立執行）：
  bug3      ：rt_zh_002_3B（超跑「竟」）並排 attribution。**只跨標題**（內文是雜訊，論點在單字過敏）。
  bug4      ：content-util 配對 attribution。**保留內文**——內文 attribution 近零正是「模型不看內文」的證據。
  aggregate ：驚訝類探針 top attribution token bar（抽樣，非全 72 條），對照 baseline_top_features.png。
  occlusion ：bug3 案例逐 token 刪除交叉驗證。**只刪標題 token**（內文與「竟」翻轉無關）。

實作地雷（已處理）：
  - SentencePiece subword 重組：合併帶 ▁ 的子詞回完整詞（同詞取絕對值最大、保留正負號）。
  - IG baseline 用 pad token、n_steps 可調、internal_batch_size 限峰值、檢查 convergence delta。

模型路徑沿用 probe_jing.py 慣例；G8 未訓練時自動 skip。

用法（PowerShell 中文輸出需 UTF-8，已於程式內設定 stdout）：
  python explain_tokens.py --task bug3        # 一次只跑一個，逐個確認記憶體
  python explain_tokens.py --task bug4
  python explain_tokens.py --task aggregate
  python explain_tokens.py --task occlusion
  python explain_tokens.py --task all         # 全跑（確認單個都安全後再用）
輸出：results/explain_<task>.json + results/explain_*.png
"""
import argparse
import gc
import io
import json
import os
import sys
from pathlib import Path

# Anaconda 常見的 libiomp5md.dll 重複初始化衝突（OMP Error #15）；須在 import torch 前設定
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# PowerShell / Windows 終端中文輸出避免 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# matplotlib 中文字型（Windows 內建，避免 heatmap 中文變方框）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results"
RESULTS.mkdir(exist_ok=True)

# 同 probe_jing：顯式模型路徑，存在才跑（未訓練的自動 skip）
MODELS = {
    "G5":   ROOT.parent / "models" / "xlm-roberta-clickbait",
    "G7":   ROOT.parent / "models" / "xlm-roberta-clickbait-g7",
    "G8":   ROOT.parent / "models" / "xlm-roberta-clickbait-g8",
    "G9":   ROOT.parent / "models" / "xlm-roberta-clickbait-g9-dualtower-run2",
}

CLICKBAIT_IDX = 1       # label=1 = clickbait
N_STEPS = 50            # IG 步數（≥50，並檢查 convergence delta）
IG_BATCH = 4            # internal_batch_size：把 50 步插值分批前向，記憶體峰值砍到約 1/12
MAX_LEN = 256
MIN_FREE_GB = 3.0       # 跑前可用記憶體低於此值就警告（兩個 XLM-R + IG 峰值的安全下限）

# 驚訝類探針（與 probe_jing 對齊）
SURPRISE = ["竟", "竟然", "居然", "沒想到", "不料", "豈料", "萬萬沒想到", "驚見", "赫然"]
PROBE_BASES = [
    ("台股今日收盤上漲50點", "成交量較昨日放大兩成",
     "台灣加權指數今日收盤上漲50點，成交量較昨日放大兩成，三大法人買超約30億元。"),
    ("北京郵電大學發布2019年碩士招生簡章", "招生名額較去年略有調整",
     "北京郵電大學日前發布2019年碩士研究生招生簡章，今年招生名額較去年略有調整，專業設置維持穩定。"),
    ("印度男子腹痛就醫", "醫師從其胃中取出100根鐵釘",
     "印度一名男子因長期腹痛就醫，X光檢查發現胃部有大量異物，醫師手術取出100根鐵釘與263枚硬幣。"),
    ("某科技公司公布第三季財報", "營收較去年同期成長6%",
     "該科技公司公布第三季財報，營收較去年同期成長6%，淨利符合分析師預期。"),
]


def check_memory():
    """跑前檢查可用記憶體；不足只警告不阻擋（讓使用者決定）。"""
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1e9
    except ImportError:
        # 無 psutil 時退回 Windows API（本機可用），失敗就跳過檢查
        try:
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("mem", ctypes.c_ulong),
                            ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong),
                            ("e", ctypes.c_ulonglong), ("f", ctypes.c_ulonglong),
                            ("g", ctypes.c_ulonglong)]
            ms = MS(); ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            free_gb = ms.c / 1e9
        except Exception:
            return
    tag = "✅" if free_gb >= MIN_FREE_GB else "⚠️"
    print(f"{tag} 可用記憶體：{free_gb:.1f} GB（建議 ≥ {MIN_FREE_GB} GB）")
    if free_gb < MIN_FREE_GB:
        print(f"   記憶體偏低，建議關閉其他程式後再跑，或一次只跑一個 --task。")


def _is_dualtower(path):
    """雙塔 checkpoint 判別特徵：同目錄存在 classifier_head.pt。"""
    return (Path(path) / "classifier_head.pt").exists()


class Explainer:
    """單一模型的 IG + occlusion 解釋器。用 close() 釋放記憶體。

    支援兩種架構，對外介面（attribute / occlude_title / predict_proba）相同：
      - 單塔（G5/G7/G8）：tokenizer(title, content) 拼一條序列，IG 作用在唯一序列，
        segment 用 [SEP] 切標題/內文。
      - 雙塔（G9）：title 與 content 各一條序列。IG 對「要解釋的那一塔」作用、另一塔固定，
        把兩塔 token 串起來回傳；segment 由『屬於哪一塔』決定（0=title 塔, 1=content 塔），
        而非靠 [SEP]——這是雙塔與單塔語意對齊的關鍵。
    """

    def __init__(self, name, path):
        self.name = name
        self.dual = _is_dualtower(path)
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.device = torch.device("cpu")

        if self.dual:
            from dualtower import load_dualtower, TITLE_MAX_LEN, CONTENT_MAX_LEN
            self.title_max = TITLE_MAX_LEN
            self.content_max = CONTENT_MAX_LEN
            self.mdl = load_dualtower(path, self.device)
            self.emb_layer = self.mdl.encoder.get_input_embeddings()
            # 雙塔的 IG forward 依「正在解釋哪一塔」而定，於 attribute 內動態綁定
            self.lig = None
        else:
            self.mdl = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
            self.mdl.to(self.device)
            self.emb_layer = self.mdl.get_input_embeddings()   # IG 作用在 word embedding 層
            self.lig = LayerIntegratedGradients(self._forward, self.emb_layer)

    def close(self):
        del self.mdl, self.lig, self.emb_layer, self.tok
        gc.collect()

    def _forward(self, input_ids, attention_mask):
        return self.mdl(input_ids=input_ids, attention_mask=attention_mask).logits.softmax(dim=-1)

    # ── 雙塔專用：固定一塔、對另一塔的 embeds 做 IG 的 forward ──────────
    def _dual_forward_embeds(self, var_embeds, var_mask, fixed_embeds, fixed_mask, target_tower):
        """plain IntegratedGradients 的 forward：var_embeds 是被解釋塔的 word embeddings
        （IG 對它求梯度），fixed_embeds 是另一塔（固定）。

        為何用 inputs_embeds 而非 LayerIntegratedGradients：雙塔共享同一個 embedding 層，
        該層在一次前向中被呼叫兩次（title 64 長、content 256 長）。LIG 會把它攔截到的
        embedding 輸出整批換成插值張量，無法區分是哪一塔的呼叫，導致 256 的插值張量
        流進 64 長的另一塔 → token_type embedding 維度衝突。改在外部各算 embeds、
        只把目標塔當 IG input，固定塔走 additional_forward_args，即可乾淨分離。

        IG 內部沿 batch 維擴張 var_embeds 成 (steps×1)，固定塔須同步 expand 對齊 batch。
        """
        bsz = var_embeds.shape[0]
        f_embeds = fixed_embeds.expand(bsz, -1, -1)
        f_mask = fixed_mask.expand(bsz, -1)
        v_mask = var_mask.expand(bsz, -1)
        if target_tower == "title":
            logits = self.mdl.forward_embeds(var_embeds, v_mask, f_embeds, f_mask)
        else:
            logits = self.mdl.forward_embeds(f_embeds, f_mask, var_embeds, v_mask)
        return logits.softmax(dim=-1)

    def _encode(self, title, content):
        # content 可能是 NaN（cu_empty 空內文）或非字串 → 統一轉乾淨字串
        content = "" if not isinstance(content, str) else content
        title = "" if not isinstance(title, str) else title
        return self.tok(title, content, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt").to(self.device)

    def _encode_dual(self, title, content):
        content = "" if not isinstance(content, str) else content
        title = "" if not isinstance(title, str) else title
        enc_t = self.tok(title, truncation=True, max_length=self.title_max,
                         padding="max_length", return_tensors="pt").to(self.device)
        enc_c = self.tok(content, truncation=True, max_length=self.content_max,
                         padding="max_length", return_tensors="pt").to(self.device)
        return enc_t, enc_c

    def predict_proba(self, title, content):
        if self.dual:
            enc_t, enc_c = self._encode_dual(title, content)
            with torch.no_grad():
                logits = self.mdl(
                    title_input_ids=enc_t["input_ids"], title_attention_mask=enc_t["attention_mask"],
                    content_input_ids=enc_c["input_ids"], content_attention_mask=enc_c["attention_mask"],
                )
                return float(logits.softmax(dim=-1)[0, CLICKBAIT_IDX])
        enc = self._encode(title, content)
        with torch.no_grad():
            return float(self.mdl(**enc).logits.softmax(dim=-1)[0, CLICKBAIT_IDX])

    def _segment_mask(self, enc):
        """回傳每個 token 屬於標題(0)或內文(1)；用於『只跨標題』的 task 過濾顯示。"""
        ids = enc["input_ids"].squeeze(0).tolist()
        sep_id = self.tok.sep_token_id
        seg, in_content = [], False
        for tid in ids:
            seg.append(1 if in_content else 0)
            if tid == sep_id:
                in_content = True
        return seg

    def _attribute_dual(self, title, content):
        """雙塔 IG：對 title 塔與 content 塔各自做 IG（對方固定），串接回傳。

        seg：0=title 塔 token、1=content 塔 token。pad token 不顯示（attn=0）。
        """
        enc_t, enc_c = self._encode_dual(title, content)
        pad_id = self.tok.pad_token_id

        def embed(ids):
            return self.emb_layer(ids)

        # ── Bug B sanity check：forward() vs forward_embeds() 數值一致性 ──────────
        # IG 走 forward_embeds（pre-extracted embeds 路徑）。對 v1（mean-pool only）已知
        # 與 forward() 等價；但 v2 + C3（cross_align）路徑未經驗證——IG 會把固定塔 expand
        # 成非 contiguous 張量餵進 nn.MultiheadAttention，數值是否正確無法靜態確認。
        # 此處在跑 IG 前先比對兩條路徑：若 v2+C3 路徑數值偏離，立即 fail
        with torch.no_grad():
            t_ids, t_m = enc_t["input_ids"], enc_t["attention_mask"]
            c_ids, c_m = enc_c["input_ids"], enc_c["attention_mask"]
            logits_ids = self.mdl(t_ids, t_m, c_ids, c_m)
            logits_emb = self.mdl.forward_embeds(embed(t_ids), t_m, embed(c_ids), c_m)
            max_diff = (logits_ids - logits_emb).abs().max().item()
            assert max_diff < 1e-4, (
                f"[Bug B] forward() vs forward_embeds() 偏離 {max_diff:.2e} (>1e-4)；"
                f"模型 {self.name} 的 IG 路徑數值不可信，attribution 不可用於報告。"
                f"（多半是 v2+C3 的 cross-attention 對 expand 後非 contiguous 張量算錯）"
            )

        def ig_one_tower(enc, fixed_enc, target_tower):
            ids, attn = enc["input_ids"], enc["attention_mask"]
            # baseline ids：非特殊 token 換 pad，特殊 token 保留
            base_ids = torch.full_like(ids, pad_id)
            for special in (self.tok.cls_token_id, self.tok.sep_token_id):
                if special is not None:
                    base_ids[ids == special] = special
            var_embeds = embed(ids)
            base_embeds = embed(base_ids)
            fixed_embeds = embed(fixed_enc["input_ids"])

            ig = IntegratedGradients(self._dual_forward_embeds)
            attrs, delta = ig.attribute(
                inputs=var_embeds, baselines=base_embeds,
                additional_forward_args=(attn, fixed_embeds,
                                         fixed_enc["attention_mask"], target_tower),
                target=CLICKBAIT_IDX,
                n_steps=N_STEPS, internal_batch_size=IG_BATCH,
                return_convergence_delta=True,
            )
            tok_attr = attrs.sum(dim=-1).squeeze(0)
            toks = self.tok.convert_ids_to_tokens(ids.squeeze(0).tolist())
            # 只保留非 pad token（雙塔 padding 到 max_length，pad 段無意義）
            keep = [i for i, m in enumerate(attn.squeeze(0).tolist()) if m == 1]
            del attrs
            return ([toks[i] for i in keep],
                    tok_attr[keep].detach(),
                    float(delta.mean()))

        t_tokens, t_attr, t_delta = ig_one_tower(enc_t, enc_c, "title")
        c_tokens, c_attr, c_delta = ig_one_tower(enc_c, enc_t, "content")

        tokens = t_tokens + c_tokens
        attr = torch.cat([t_attr, c_attr])
        attr = (attr / (attr.norm() + 1e-12)).numpy()
        seg = [0] * len(t_tokens) + [1] * len(c_tokens)
        proba = self.predict_proba(title, content)
        gc.collect()
        return tokens, attr, proba, float(t_delta + c_delta), seg

    def attribute(self, title, content):
        """回傳 (tokens, attr, proba, delta, seg)：token-level IG attribution。"""
        if self.dual:
            return self._attribute_dual(title, content)
        enc = self._encode(title, content)
        input_ids, attn = enc["input_ids"], enc["attention_mask"]
        pad_id = self.tok.pad_token_id
        baseline = torch.full_like(input_ids, pad_id)
        for special in (self.tok.cls_token_id, self.tok.sep_token_id):
            if special is not None:
                baseline[input_ids == special] = special

        attrs, delta = self.lig.attribute(
            inputs=input_ids, baselines=baseline,
            additional_forward_args=(attn,), target=CLICKBAIT_IDX,
            n_steps=N_STEPS, internal_batch_size=IG_BATCH,   # ← 限峰值的關鍵
            return_convergence_delta=True,
        )
        token_attr = attrs.sum(dim=-1).squeeze(0)
        token_attr = token_attr / (token_attr.norm() + 1e-12)
        tokens = self.tok.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        proba = self.predict_proba(title, content)
        seg = self._segment_mask(enc)
        del attrs, enc
        gc.collect()
        return tokens, token_attr.detach().numpy(), proba, float(delta), seg

    def occlude_title(self, title, content):
        """只刪標題 token 算 Δp（內文與『竟』翻轉無關，不逐 token 跑長內文）。"""
        if self.dual:
            return self._occlude_title_dual(title, content)
        base_p = self.predict_proba(title, content)
        enc = self._encode(title, content)
        ids = enc["input_ids"].squeeze(0).tolist()
        seg = self._segment_mask(enc)
        tokens = self.tok.convert_ids_to_tokens(ids)
        specials = set(self.tok.all_special_ids)
        deltas = []
        for i, tid in enumerate(ids):
            if tid in specials or seg[i] == 1:   # 跳過特殊 token 與內文段
                deltas.append(0.0)
                continue
            kept = ids[:i] + ids[i + 1:]
            sub = torch.tensor([kept], device=self.device)
            with torch.no_grad():
                p = float(self.mdl(input_ids=sub, attention_mask=torch.ones_like(sub))
                          .logits.softmax(dim=-1)[0, CLICKBAIT_IDX])
            deltas.append(base_p - p)
        return tokens, np.array(deltas), base_p, seg

    def _occlude_title_dual(self, title, content):
        """雙塔版 occlusion：刪標題塔的單一 token（content 塔固定）算 Δp。"""
        base_p = self.predict_proba(title, content)
        enc_t, enc_c = self._encode_dual(title, content)
        t_ids_full = enc_t["input_ids"].squeeze(0).tolist()
        t_mask = enc_t["attention_mask"].squeeze(0).tolist()
        c_ids, c_mask = enc_c["input_ids"], enc_c["attention_mask"]
        specials = set(self.tok.all_special_ids)
        # 只取非 pad 的標題 token（與 attribute_dual 的顯示對齊）
        keep = [i for i, m in enumerate(t_mask) if m == 1]
        tokens = [self.tok.convert_ids_to_tokens([t_ids_full[i]])[0] for i in keep]
        seg = [0] * len(keep)
        deltas = []
        for i in keep:
            if t_ids_full[i] in specials:
                deltas.append(0.0)
                continue
            kept_ids = [t_ids_full[j] for j in range(len(t_ids_full)) if j != i and t_mask[j] == 1]
            sub = torch.tensor([kept_ids], device=self.device)
            with torch.no_grad():
                logits = self.mdl(
                    title_input_ids=sub, title_attention_mask=torch.ones_like(sub),
                    content_input_ids=c_ids, content_attention_mask=c_mask,
                )
                p = float(logits.softmax(dim=-1)[0, CLICKBAIT_IDX])
            deltas.append(base_p - p)
        return tokens, np.array(deltas), base_p, seg


def _is_cjk(ch):
    """是否為中日韓表意文字或全形標點。
    XLM-R 的 SentencePiece 對中文是『詞首一個獨立 ▁，之後每個漢字 token 都不帶 ▁』，
    若沿用英文 BPE 的『不帶 ▁ = 續接』規則會把整句中文黏成一個詞——故中文須逐字獨立。"""
    return (
        "一" <= ch <= "鿿"      # CJK 統一表意
        or "㐀" <= ch <= "䶿"   # 擴展 A
        or "豈" <= ch <= "﫿"   # 相容表意
        or "　" <= ch <= "〿"   # CJK 標點
        or "＀" <= ch <= "￯"   # 全形字元
    )


def merge_subwords(tokens, scores, seg=None):
    """SentencePiece subword 合併回完整詞。同詞取絕對值最大（保留正負號）。
    規則：▁ 起新詞；中文字逐字獨立（不黏前詞）；英文 word-piece 才黏續接。
    傳 seg 時一併回傳每個合併詞的段別（0=標題,1=內文）。"""
    words, word_scores, word_seg = [], [], []
    special = {"<s>", "</s>", "<pad>", "<unk>", "[CLS]", "[SEP]", "[PAD]"}
    for i, (tok, sc) in enumerate(zip(tokens, scores)):
        if tok in special:
            continue
        is_start = tok.startswith("▁")
        clean = tok[1:] if is_start else tok
        if not clean:
            continue
        # 中文逐字獨立成詞；▁ 開頭或首詞也新起；其餘（英文 word-piece）黏前詞
        new_word = is_start or not words or _is_cjk(clean[0])
        if new_word:
            words.append(clean); word_scores.append(sc)
            if seg is not None:
                word_seg.append(seg[i])
        else:
            words[-1] += clean
            if abs(sc) > abs(word_scores[-1]):
                word_scores[-1] = sc
    if seg is not None:
        return words, np.array(word_scores), word_seg
    return words, np.array(word_scores)


def attr_heatmap(ax, words, scores, title, seg=None):
    """單欄 attribution heatmap（紅=推向 clickbait，藍=推離）。內文段標灰底。"""
    lim = max(abs(scores).max(), 1e-9)
    ax.imshow(scores.reshape(-1, 1), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(words)))
    labels = [f"{w} ·內" if (seg and seg[i]) else w for i, w in enumerate(words)]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([])
    ax.set_title(title, fontsize=10)
    for i, sc in enumerate(scores):
        ax.text(0, i, f"{sc:+.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(sc) > lim * 0.5 else "black")


def load_models():
    explainers = []
    for name, path in MODELS.items():
        if not path.exists():
            print(f"[skip] {name} 模型不存在：{path}")
            continue
        print(f"[load] {name}：{path}")
        explainers.append(Explainer(name, path))
    if not explainers:
        sys.exit("沒有可用模型，結束。")
    return explainers


def get_case(df, cid):
    row = df[df["id"] == cid]
    if not len(row):
        print(f"  [skip] 找不到 {cid}")
        return None
    return row.iloc[0]


# ---------- 四個 task ----------

def task_bug3(df, explainers):
    """Bug 3：超跑「竟」並排 attribution。只顯示標題段。"""
    print("\n=== Bug 3：rt_zh_002_3B（超跑「竟」FP）並排 attribution（只跨標題）===")
    case = get_case(df, "rt_zh_002_3B")
    if case is None:
        return {}
    n = len(explainers)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 6))
    axes = [axes] if n == 1 else axes
    rec = {}
    for ax, ex in zip(axes, explainers):
        tokens, attr, proba, delta, seg = ex.attribute(case["title"], case["content"])
        words, ws, wseg = merge_subwords(tokens, attr, seg)
        keep = [i for i, s in enumerate(wseg) if s == 0]   # 只留標題詞
        words, ws = [words[i] for i in keep], ws[np.array(keep)]
        attr_heatmap(ax, words, ws, f"{ex.name}  P(cb)={proba:.2f}")
        rec[ex.name] = {"proba": round(proba, 4), "convergence_delta": round(delta, 5),
                        "title_words": words, "scores": [round(float(s), 4) for s in ws]}
        print(f"  [{ex.name}] P(clickbait)={proba:.3f}  IG Δ={delta:+.4f}")
    fig.suptitle("Bug 3：『竟』是否撐起 clickbait 判定？", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "explain_bug3_jing.png")
    return rec


def task_bug4(df, explainers):
    """Bug 4：content-util 配對。保留內文 attribution（證明模型不看內文）。"""
    print("\n=== Bug 4：content-util 配對 attribution（保留內文，看是否近零）===")
    rec = {}
    for cid in ["rt_zh_b4_cu_normal", "rt_zh_b4_cu_empty"]:
        case = get_case(df, cid)
        if case is None:
            continue
        print(f"-- {cid} --")
        n = len(explainers)
        fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 8))
        axes = [axes] if n == 1 else axes
        rec[cid] = {}
        for ax, ex in zip(axes, explainers):
            tokens, attr, proba, delta, seg = ex.attribute(case["title"], case["content"])
            words, ws, wseg = merge_subwords(tokens, attr, seg)
            attr_heatmap(ax, words, ws, f"{ex.name}  P(cb)={proba:.2f}", wseg)
            content_attr = [float(s) for s, sg in zip(ws, wseg) if sg == 1]
            mean_abs = float(np.mean(np.abs(content_attr))) if content_attr else 0.0
            rec[cid][ex.name] = {"proba": round(proba, 4),
                                 "content_mean_abs_attr": round(mean_abs, 4),
                                 "n_content_words": len(content_attr)}
            print(f"  [{ex.name}] P(cb)={proba:.3f}  內文平均|attr|={mean_abs:.4f}"
                  f"（{len(content_attr)} 詞，近零=不看內文）")
        fig.suptitle(f"Bug 4：{cid} — 內文 attribution 是否近零？", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, f"explain_bug4_{cid}.png")
    return rec


def task_aggregate(explainers):
    """聚合：驚訝類探針 top attribution token bar（抽樣：每詞配 1 個 base，9 次/模型，非 72 次）。"""
    print("\n=== 聚合：驚訝類 top attribution token（抽樣，對照 baseline_top_features.png）===")
    fig, axes = plt.subplots(1, len(explainers), figsize=(5 * len(explainers), 5), squeeze=False)
    rec = {}
    for ax, ex in zip(axes[0], explainers):
        wsum, wcnt = {}, {}
        for k, marker in enumerate(SURPRISE):
            a, b, c = PROBE_BASES[k % len(PROBE_BASES)]   # 抽樣：每驚訝詞只配一個 base
            tokens, attr, _, _, seg = ex.attribute(f"{a}，{marker}{b}", c)
            words, ws, wseg = merge_subwords(tokens, attr, seg)
            for w, s, sg in zip(words, ws, wseg):
                if sg == 1:   # 只統計標題詞
                    continue
                wsum[w] = wsum.get(w, 0.0) + float(s)
                wcnt[w] = wcnt.get(w, 0) + 1
        avg = {w: wsum[w] / wcnt[w] for w in wsum}
        top = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ax.barh([w for w, _ in top][::-1], [v for _, v in top][::-1], color="#c0392b")
        ax.set_title(f"{ex.name}：驚訝類 top attribution token", fontsize=10)
        ax.set_xlabel("平均 attribution（推向 clickbait）")
        rec[ex.name] = {w: round(v, 4) for w, v in top}
        print(f"  [{ex.name}] top：{', '.join(w for w, _ in top[:5])}")
    fig.tight_layout()
    _save(fig, "explain_aggregate_surprise.png")
    return rec


def task_occlusion(df, explainers):
    """Occlusion 交叉驗證：bug3 案例只刪標題 token。"""
    print("\n=== Occlusion 交叉驗證：rt_zh_002_3B 標題各 token Δp ===")
    case = get_case(df, "rt_zh_002_3B")
    if case is None:
        return {}
    rec = {}
    for ex in explainers:
        tokens, deltas, base_p, seg = ex.occlude_title(case["title"], case["content"])
        words, wd, wseg = merge_subwords(tokens, deltas, seg)
        title_pairs = [(w, d) for w, d, s in zip(words, wd, wseg) if s == 0]
        top = sorted(title_pairs, key=lambda kv: kv[1], reverse=True)[:5]
        rec[ex.name] = {"base_proba": round(base_p, 4),
                        "top_delta_tokens": {w: round(float(d), 4) for w, d in top}}
        print(f"  [{ex.name}] base P={base_p:.3f}；刪後最降 clickbait："
              f"{', '.join(f'{w}({d:+.3f})' for w, d in top)}")
    return rec


def _save(fig, fname):
    out = RESULTS / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 存圖：{out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["bug3", "bug4", "aggregate", "occlusion", "all"],
                    required=True, help="一次只跑一個 task（記憶體安全）；確認單個都安全再用 all")
    args = ap.parse_args()

    check_memory()
    explainers = load_models()
    df = pd.read_csv(ROOT.parent / "dataset" / "processed" / "regression_test.csv")
    out = {"task": args.task, "models": [e.name for e in explainers], "n_steps": N_STEPS}

    if args.task in ("bug3", "all"):
        out["bug3"] = task_bug3(df, explainers)
    if args.task in ("bug4", "all"):
        out["bug4"] = task_bug4(df, explainers)
    if args.task in ("aggregate", "all"):
        out["aggregate"] = task_aggregate(explainers)
    if args.task in ("occlusion", "all"):
        out["occlusion"] = task_occlusion(df, explainers)

    for ex in explainers:
        ex.close()

    out_path = RESULTS / f"explain_{args.task}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存：{out_path}")


if __name__ == "__main__":
    main()
