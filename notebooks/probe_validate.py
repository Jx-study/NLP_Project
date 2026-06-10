"""兩個只讀驗證，把假設變事實，不寫任何訓練碼。

驗證 1（控制實驗）：rt_zh_002_3B 超跑案去掉「竟」是否翻成 0
    → 坐實 Bug 3（轉折詞）是主因，而非 Bug 1（事件離奇）。

驗證 2（資料審計）：adversarial_tone.csv 的 tw_neg 反例裡，
    「竟/竟然/居然」等轉折詞的數量與句式分布
    → 判斷是「資料假」還是「資料少/沒覆蓋到這種句式」。
"""
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

G7_DIR = "../models/xlm-roberta-clickbait-g7"
ADV_PATH = "../dataset/processed/adversarial_tone.csv"

# rt_zh_002_3B 原文（含「竟」）
ORIG_TITLE = "倫敦女子騎腳踏車不慎擦撞麥拉倫超跑，塑膠踏板竟當場刺穿碳纖維車殼"
CONTENT = ("英國倫敦昨日發生一起罕見的交通意外。一名 35 歲女子在騎乘外送腳踏車時，"
           "不小心撞上了停在路邊的千萬超級跑車（麥拉倫）。令人震驚的是，該超跑的碳纖維車殼"
           "竟然當場被腳踏車的塑膠踏板刺穿了一個大洞，而腳踏車卻毫損。超跑車主下車看到這一幕"
           "整個人都傻眼了，預估維修費用將高達台幣 80 萬元。")

# 變體：逐步移除觸發點，做消融
VARIANTS = [
    ("原文（含「竟」+物理衝突）", ORIG_TITLE),
    ("去掉標題「竟」", "倫敦女子騎腳踏車不慎擦撞麥拉倫超跑，塑膠踏板當場刺穿碳纖維車殼"),
    ("去掉「竟」+改中性動詞", "倫敦女子騎腳踏車擦撞麥拉倫超跑，塑膠踏板刺穿碳纖維車殼"),
    ("完全中性改寫", "倫敦女子騎腳踏車擦撞麥拉倫超跑，造成碳纖維車殼破損"),
]


def validation_1(tok, mdl):
    def predict(title, content):
        enc = tok(title, content, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(logits.argmax(dim=-1).item())
        return pred, round(float(probs[pred]), 3)

    print("=" * 80)
    print("驗證 1：rt_zh_002_3B 消融實驗（內文固定，只改標題）")
    print("=" * 80)
    print(f"{'pred':4} {'conf':5}  變體")
    print("-" * 80)
    for desc, title in VARIANTS:
        pred, conf = predict(title, CONTENT)
        print(f"{pred:<4} {conf:<5}  {desc}: {title}")
    print()


def validation_2():
    print("=" * 80)
    print("驗證 2：tw_neg 反例的轉折詞分布審計")
    print("=" * 80)
    adv = pd.read_csv(ADV_PATH)
    tw = adv[adv["id"].astype(str).str.endswith("_tw_neg")]
    print(f"tw_neg 總筆數：{len(tw)}（zh={sum(tw['lang']=='zh')}, en={sum(tw['lang']=='en')}）\n")

    zh = tw[tw["lang"] == "zh"]
    markers = ["竟然", "竟", "居然", "沒想到", "但是", "但", "卻", "然而", "雖然", "雖"]
    print("中文 tw_neg 各轉折詞出現次數（標題中）：")
    for m in markers:
        n = zh["title"].str.contains(m, regex=False).sum()
        print(f"  {m:4} : {n:3}  ({n/len(zh)*100:.0f}%)" if len(zh) else f"  {m}: 0")

    # 關鍵：有多少是「竟」開頭/前段（像 rt_zh_002_3B 那種句式）
    print("\n句式分析（「竟」類）：")
    jing = zh[zh["title"].str.contains("竟", regex=False)]
    print(f"  含「竟」的筆數：{len(jing)}")
    # 「竟」出現在標題前半 vs 後半
    if len(jing):
        front = sum(t.index("竟") < len(t) * 0.5 for t in jing["title"])
        print(f"    其中「竟」在標題前半段：{front}（這種句式才像 rt_zh_002_3B）")
        print(f"    「竟」在標題後半段：{len(jing) - front}")
        print("\n  「竟」類樣本舉例（最多 5 筆）：")
        for t in jing["title"].head(5):
            print(f"    - {t}")

    # 物理衝突類關鍵詞（rt_zh_002_3B 的第二觸發點）
    print("\n  tw_neg 裡有沒有「物理常識衝突」型的樣本？")
    print("  （tw_neg 是把真實非標題黨標題加轉折詞，按設計不會有荒誕事件，")
    print("   所以模型沒見過『荒誕事件+轉折詞』的 label=0 組合 → 這可能是漏判主因）")


def main():
    tok = AutoTokenizer.from_pretrained(G7_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(G7_DIR).eval()
    validation_1(tok, mdl)
    validation_2()


if __name__ == "__main__":
    main()
