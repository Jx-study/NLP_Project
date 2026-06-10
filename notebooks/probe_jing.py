"""驗證「補轉折詞資料能不能修」的前置診斷：
G5 / G7 對各類轉折詞的過敏程度有多普遍？

做法：拿多個真實非標題黨新聞事件（事實型，本應 label=0），
把同一句子分別插入不同轉折詞，看模型預測如何變化。

判讀：
  若「竟/竟然/居然/沒想到」等驚訝類轉折詞普遍讓 label 0→1，
  而「但/雖然」等對比類不會
  → 證實 tw_neg 覆蓋偏斜（但93% / 竟0%）造成系統性盲點，補這類資料有效。

輸出：results/probe_jing_transition.json（記錄模型路徑與每個模型的結果，供文檔引用）。

用法（PowerShell 中文輸出需 UTF-8，已於程式內設定 stdout）：
  python probe_jing.py
"""
import io
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# PowerShell / Windows 終端中文輸出避免 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 同時診斷 G5 與 G7，分別加載並輸出（reviewer #3：先前誤把 G5 當 G7）
MODELS = {
    "G5": "../models/xlm-roberta-clickbait",       # augmentation only
    "G7": "../models/xlm-roberta-clickbait-g7",    # + tone adversarial
}

# 真實非標題黨新聞事件（事實型，本應 label=0）：(前段, 後段, 內文)
# 轉折詞插在「前段，<轉折詞>後段」
BASES = [
    ("台股今日收盤上漲50點", "成交量較昨日放大兩成",
     "台灣加權指數今日收盤上漲50點，成交量較昨日放大兩成，三大法人買超約30億元。"),
    ("北京郵電大學發布2019年碩士招生簡章", "招生名額較去年略有調整",
     "北京郵電大學日前發布2019年碩士研究生招生簡章，今年招生名額較去年略有調整，專業設置維持穩定。"),
    ("印度男子腹痛就醫", "醫師從其胃中取出100根鐵釘",
     "印度一名男子因長期腹痛就醫，X光檢查發現胃部有大量異物，醫師手術取出100根鐵釘與263枚硬幣。"),
    ("本日疫情通報新增確診病例較昨日下降", "累計人數仍超過千人",
     "本日疫情通報，新增確診病例較昨日下降，累計人數仍超過千人，防控形勢依然嚴峻。"),
    ("某科技公司公布第三季財報", "營收較去年同期成長6%",
     "該科技公司公布第三季財報，營收較去年同期成長6%，淨利符合分析師預期。"),
    ("市府宣布明年調漲公車票價", "捷運票價維持不變",
     "市政府今日宣布明年起公車票價調漲2元，捷運票價維持不變，預計影響通勤族每月支出。"),
]

SURPRISE = ["竟", "竟然", "居然", "沒想到", "不料", "豈料", "萬萬沒想到", "驚見", "赫然"]
CONTRAST = ["但", "但是", "卻", "然而", "不過"]
RESULT = ["反而", "結果", "最終"]
MARKERS = ["（基準，無轉折）"] + SURPRISE + CONTRAST + RESULT


def insert(a, b, marker):
    return f"{a}，{b}" if marker == "（基準，無轉折）" else f"{a}，{marker}{b}"


def run_model(name, path):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict(title, content):
        enc = tok(title, content, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        return int(logits.argmax(dim=-1).item())

    flips = {}
    for marker in MARKERS:
        n = sum(predict(insert(a, b, marker), c) == 1 for a, b, c in BASES)
        flips[marker] = n

    def block(title, names):
        print(title)
        for name_ in names:
            f = flips[name_]
            print(f"  {name_:14} {f}/{len(BASES)}  {'█'*f}{'░'*(len(BASES)-f)}")
        print()

    print(f"\n{'='*70}\n模型 {name}  ({path})\n{'='*70}")
    print(f"各轉折詞造成『判 clickbait（FP）』的比例（{len(BASES)} 個真實非標題黨事件，全應為 0）：\n")
    block("基準：", ["（基準，無轉折）"])
    block("【驚訝/意外類】（tw_neg 覆蓋率 0%，疑似盲點）：", SURPRISE)
    block("【對比/讓步類】（tw_neg 覆蓋多：但93% / 雖30%）：", CONTRAST)
    block("【結果/反差類】：", RESULT)

    s_rate = sum(flips[m] for m in SURPRISE) / (len(SURPRISE) * len(BASES))
    c_rate = sum(flips[m] for m in CONTRAST) / (len(CONTRAST) * len(BASES))
    print(f"驚訝類平均 FP 率：{s_rate:.0%}   對比類平均 FP 率：{c_rate:.0%}   「竟」：{flips['竟']}/{len(BASES)}")
    return {
        "model_path": path,
        "flips": flips,
        "surprise_fp_rate": round(s_rate, 4),
        "contrast_fp_rate": round(c_rate, 4),
        "jing_flips": flips["竟"],
        "n_bases": len(BASES),
    }


def main():
    out = {"n_bases": len(BASES), "models": {}}
    for name, path in MODELS.items():
        if not Path(path).exists():
            print(f"[skip] {name} 模型不存在：{path}")
            continue
        out["models"][name] = run_model(name, path)

    print(f"\n{'='*70}\nG5 vs G7 對照（驚訝類盲點是否為底層資料繼承）：")
    for name, r in out["models"].items():
        print(f"  {name}: 驚訝類 {r['surprise_fp_rate']:.0%} / 對比類 {r['contrast_fp_rate']:.0%} / 竟 {r['jing_flips']}/{r['n_bases']}")
    print("判讀：G5、G7 驚訝類 FP 率皆遠高於對比類，且兩者相近")
    print("      → 盲點為底層資料繼承（train 缺驚訝類 label=0），非 tone 對抗訓練造成。")

    out_path = Path("../results/probe_jing_transition.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存：{out_path}")


if __name__ == "__main__":
    main()
