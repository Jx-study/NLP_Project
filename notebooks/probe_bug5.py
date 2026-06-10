"""驗證 Bug 5（知性語氣）：rt_zh_003 以 0.998 高信心判錯（FN）。
診斷它是不是和 Bug 3 一樣，是「某類資料缺失」造成的系統性盲點。

Bug 5 樣本特徵：標題用學術/科普腔（「探討」「機制」「影響」），
但藏懸念缺口（「某種常見的早餐搭配」死不說是什麼）→ 應為 clickbait(1)，模型卻判 0。

兩個驗證：
  A. 消融：把 rt_zh_003 的學術腔逐步去掉/補上懸念缺口，看 pred 變化
  B. 資料審計：train.csv 的 label=1 裡，有多少是「學術腔 + 懸念缺口」這種？
     （若幾乎沒有 → 證實是缺這類資料的系統性盲點）
"""
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

G7_DIR = "../models/xlm-roberta-clickbait-g7"
TRAIN = "../dataset/processed/train.csv"

RT003_TITLE = "探討日常飲食對人體免疫機制的影響：某種常見的早餐搭配可能在無形中增加身體的發炎反應"
RT003_CONTENT = ("營養師指出，台灣人早餐愛吃的「油條配米漿」，因為兩者皆屬於高油、高糖、高熱量的精緻澱粉，"
                 "如果天天食用，長期下來容易使血糖快速上升，引發體內的慢性發炎反應。建議民眾適量食用，並多補充膳食纖維。")

# 消融變體：學術腔 vs 直白；有懸念缺口 vs 點名
VARIANTS = [
    ("原文（學術腔+懸念缺口）應1", RT003_TITLE),
    ("學術腔，但點名早餐 應0", "探討日常飲食對人體免疫機制的影響：油條配米漿可能增加身體的發炎反應"),
    ("直白腔+懸念缺口 應1", "天天吃這種早餐，竟然會讓身體慢性發炎？"),
    ("直白腔+點名 應0", "營養師：油條配米漿屬高油高糖，天天吃易引發慢性發炎"),
    ("純學術腔無缺口 應0", "研究探討精緻澱粉早餐與慢性發炎反應的關聯"),
]


def main():
    tok = AutoTokenizer.from_pretrained(G7_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(G7_DIR).eval()

    def predict(title, content):
        enc = tok(title, content, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(logits.argmax(dim=-1).item())
        return pred, round(float(probs[pred]), 3)

    print("=" * 85)
    print("驗證 A：rt_zh_003 消融（內文固定，改標題語氣/懸念）")
    print("=" * 85)
    print(f"{'pred':4} {'conf':5}  變體")
    print("-" * 85)
    for desc, title in VARIANTS:
        pred, conf = predict(title, RT003_CONTENT)
        print(f"{pred:<4} {conf:<5}  {desc} | {title[:38]}")

    print("\n" + "=" * 85)
    print("驗證 B：train.csv label=1 裡「學術/科普腔」標題有多少？")
    print("=" * 85)
    df = pd.read_csv(TRAIN)
    zh1 = df[(df["lang"] == "zh") & (df["label"] == 1)]
    print(f"中文 clickbait(label=1) 總數：{len(zh1)}")

    # 學術/科普腔關鍵詞
    academic = ["探討", "研究", "機制", "影響", "關聯", "分析", "解析", "原理",
                "為何", "如何", "之所以", "背後", "原因", "現象"]
    pattern = "|".join(academic)
    has_acad = zh1["title"].str.contains(pattern, regex=True)
    print(f"含學術/科普腔詞彙的 clickbait：{has_acad.sum()} ({has_acad.mean()*100:.1f}%)")

    # 更嚴：學術腔「開頭」（探討/研究/解析 等起手式）
    strict = zh1["title"].str.match(r"^(探討|研究|解析|淺談|論|關於|剖析)")
    print(f"以學術腔『開頭』的 clickbait：{strict.sum()} ({strict.mean()*100:.1f}%)")

    print("\n含學術腔詞的 clickbait 樣本舉例（最多 8 筆）：")
    for t in zh1[has_acad]["title"].head(8):
        print(f"  - {t}")

    print("\n判讀：")
    print("  rt_zh_003 是『純學術腔開頭(探討…) + 懸念缺口』。")
    print("  若 label=1 裡這種『學術腔開頭』樣本比例極低，")
    print("  代表模型把『探討/研究』開頭強烈關聯到 non-clickbait（新聞/論文腔）→ 系統性 FN 盲點。")


if __name__ == "__main__":
    main()
