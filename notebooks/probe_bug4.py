"""Bug 4（content utilization / 標題內文脫鉤）診斷。

reviewer #5：先前「同一明顯 clickbait 標題換四種內文仍全判 1」只能證明
「該標題上模型忽略內文」，不足以證明 Bug 4 的 FN 根因都是知性語氣。
本版改為**正反標題 × 多種內文交換**，更嚴謹地檢驗：

  驗證 A：BugReport 記錄的「知性腔 + 廣告內文」案例（FN 現況）
  驗證 B：content utilization —— 固定標題、交換多種內文，看預測是否變化
          測「明顯 clickbait 標題」與「明顯正常標題」兩端，
          若兩端換內文預測都不變 → 模型確實普遍忽略內文（content utilization 弱）
  驗證 C：固定內文、只換標題語氣（知性腔 vs 口語腔）→ 看標題語氣的影響

結論輸出 results/probe_bug4_content.json。模型路徑顯式記錄。
"""
import io
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

G7_DIR = "../models/xlm-roberta-clickbait-g7"

# ── A. BugReport 記錄案例（預期 label=1，疑似 G7 判 0）──
CASES_A = [
    ("zh", "真正的隐形富豪，都是怎样花钱的？", "广告", 1),
    ("zh", "探討優質睡眠與深層放鬆的生理機制：特定種類的植物精油成分有助於副交感神經的穩定", "广告", 1),
    ("zh", "關於空間美學與生活儀式感的重塑：在客廳角落刻意留白那一處空間能帶來視覺上的延伸", "LINE點擊連結加入好友，領取專屬優惠", 1),
    ("en", "Assessing the Long-term Financial Sustainability of Household Portfolios: Certain Low-risk Investment Vehicles Offer Hidden Disadvantages Over Time",
     "Click here to buy premium templates now!", 1),
]

# ── B. content utilization：正反兩種標題，各換多種內文 ──
# clickbait 標題（明顯誘導）
BAIT_TITLE = "醫生不敢說的秘密：某種常見食物竟悄悄傷害你的肝"
# 正常標題（忠實完整）
PLAIN_TITLE = "肝膽科醫師：加工肉品攝取過量增加肝臟代謝負擔，建議每週不超過兩次"
CONTENTS = [
    ("正常完整內文", "肝膽科醫師指出，加工肉品如香腸、培根因含大量亞硝酸鹽與飽和脂肪，長期過量食用會增加肝臟代謝負擔，建議每週攝取不超過兩次。"),
    ("廣告內文", "广告"),
    ("加好友內文", "點擊下方連結加入官方LINE好友，領取專屬健康優惠"),
    ("空內文", ""),
    ("不相關內文", "本週末天氣晴朗，氣溫攝氏28度，適合戶外活動，紫外線指數偏高請注意防曬。"),
]

# ── C. 固定內文、只換標題語氣 ──
FIXED_CONTENT = "薰衣草精油含芳樟醇，研究顯示吸入後可降低交感神經活性，有助入眠。"
TITLE_VARIANTS = [
    ("知性腔+模糊代稱(應1)", "探討優質睡眠的生理機制：特定種類的植物精油有助於副交感神經穩定", 1),
    ("知性腔+點名(應0)", "研究：薰衣草精油的芳樟醇可降低交感神經活性助眠", 0),
    ("口語腔+模糊代稱(應1)", "失眠救星！這款精油竟讓你秒睡，醫生都在用", 1),
    ("口語腔+點名(應0)", "薰衣草精油含芳樟醇，醫師：吸入有助入眠", 0),
]


def main():
    tok = AutoTokenizer.from_pretrained(G7_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(G7_DIR).eval()

    def predict(title, content):
        enc = tok(str(title), str(content), truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(logits.argmax(dim=-1).item())
        return pred, round(float(probs[pred]), 3)

    out = {"model_path": G7_DIR}

    # A
    print("=" * 85)
    print("驗證 A：BugReport「知性腔 + 廣告內文」案例（全應 label=1）")
    print("=" * 85)
    a_wrong = 0
    a_rows = []
    for lang, title, content, exp in CASES_A:
        pred, conf = predict(title, content)
        ok = pred == exp
        a_wrong += not ok
        print(f"  exp{exp} got{pred} ({conf}) {'OK' if ok else 'XX'}  {title[:34]} | {content[:12]}")
        a_rows.append({"title": title, "content": content, "exp": exp, "pred": pred, "conf": conf})
    print(f"  → Bug4 案例漏判 {a_wrong}/{len(CASES_A)}")
    out["A_known_cases"] = {"wrong": a_wrong, "total": len(CASES_A), "rows": a_rows}

    # B
    print("\n" + "=" * 85)
    print("驗證 B：content utilization（固定標題，換 5 種內文，看預測變不變）")
    print("=" * 85)
    out["B_content_utilization"] = {}
    for tlabel, title in [("clickbait標題", BAIT_TITLE), ("正常標題", PLAIN_TITLE)]:
        print(f"\n[{tlabel}] {title[:40]}")
        preds = []
        for desc, content in CONTENTS:
            pred, conf = predict(title, content)
            preds.append(pred)
            print(f"  pred{pred} ({conf})  {desc}")
        unchanged = len(set(preds)) == 1
        print(f"  → 換 5 種內文預測{'不變' if unchanged else '有變'}（{preds}）")
        out["B_content_utilization"][tlabel] = {"preds": preds, "unchanged": unchanged}

    # C
    print("\n" + "=" * 85)
    print("驗證 C：固定內文，只換標題語氣（知性腔 vs 口語腔）")
    print("=" * 85)
    c_rows = []
    for desc, title, exp in TITLE_VARIANTS:
        pred, conf = predict(title, FIXED_CONTENT)
        ok = pred == exp
        print(f"  exp{exp} got{pred} ({conf}) {'OK' if ok else 'XX'}  {desc}")
        c_rows.append({"desc": desc, "title": title, "exp": exp, "pred": pred, "conf": conf})
    out["C_title_tone"] = c_rows

    print("\n" + "=" * 85)
    print("綜合判讀：")
    print("  B 若兩端換內文預測都不變 → 模型 content utilization 弱（普遍忽略內文）")
    print("  C 若知性腔判0、口語腔判1（內文相同）→ 標題語氣是 FN 的主因之一")
    print("  兩者需一起看：Bug 4（content utilization）與 Bug 5（知性語氣）是否可分。")

    out_path = Path("../results/probe_bug4_content.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存：{out_path}")


if __name__ == "__main__":
    main()
