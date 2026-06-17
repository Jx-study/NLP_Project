"""人工標注小工具：逐筆顯示新聞，按鍵填 label / keep / fit_type，直接寫回原 csv。

特性：
  - **斷點續標**：已有 label 的自動跳過；關掉再開從未標的繼續。
  - **逐筆寫回**：每標一筆立刻存檔（關掉不丟進度）。
  - **單鍵操作**（Windows msvcrt）：大多數筆三兩下搞定。

操作：
  label    : 0 / 1            （必填；Enter 跳過留空，該筆下次會再出現）
  keep     : k = 廢棄(0)       （預設 1，只標例外；Enter = keep 1）
  fit_type : n=normal t=transition a=academic x=none  （Enter 跳過）
  其他     : q = 存檔並退出     b = 上一筆重標     s = 跳過此筆不標

判準速記（詳見 ANNOTATION_GUIDE.md）：
  label=1 標題黨：資訊落差／過度誇飾／承諾落空／格式誘餌／軟文業配
  label=0 非標題黨：標題忠實完整（事件離奇但標題誠實 → 仍是 0）
  fit_type（給 adversarial_base 改寫用）：
    transition=有對比事實可造驚訝類轉折  academic=科普/生活/研究可造知性腔
    normal=一般新聞通用改寫  none=硬新聞(政治/災難)不套句式

用法：
  python label_tool.py                 # 標所有 scraped_*.csv
  python label_tool.py --file scraped_kknews.csv   # 只標單一檔
  python label_tool.py --only-unlabeled            # 只看未標的（預設行為）
"""
import argparse
import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import msvcrt
    def getkey():
        ch = msvcrt.getch()
        try:
            return ch.decode("utf-8", "ignore").lower()
        except Exception:
            return ""
except ImportError:
    # 非 Windows fallback（用 input 取第一個字）
    def getkey():
        s = input()
        return s[:1].lower() if s else "\r"

RAW_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw_scraped"
FIT_KEYS = {"n": "normal", "t": "transition", "a": "academic", "x": "none"}


def is_labeled(row):
    """label 欄已有 0/1 視為已標。"""
    v = row.get("label")
    return str(v).strip() in ("0", "1")


def prompt_label():
    """回傳 '0'/'1'/None(跳過)/特殊指令字串。"""
    print("  label [0/1] (Enter跳過, s=skip, b=上一筆, q=退出): ", end="", flush=True)
    while True:
        k = getkey()
        if k in ("0", "1"):
            print(k)
            return k
        if k in ("q", "b", "s"):
            print(k)
            return "__" + k
        if k in ("\r", "\n"):
            print("(跳過)")
            return None


def prompt_keep():
    print("  keep  [Enter=1可用, k=0廢棄]: ", end="", flush=True)
    k = getkey()
    if k == "k":
        print("0 (廢棄)")
        return 0
    print("1")
    return 1


def prompt_fit():
    print("  fit_type [n=normal t=transition a=academic x=none, Enter跳過]: ", end="", flush=True)
    while True:
        k = getkey()
        if k in FIT_KEYS:
            print(FIT_KEYS[k])
            return FIT_KEYS[k]
        if k in ("\r", "\n"):
            print("(跳過)")
            return None


def show_row(row, idx, total, done):
    print("\n" + "=" * 78)
    print(f"  [{done} 已標 | 第 {idx+1}/{total} 筆 | {row.get('source_site','?')} / {row.get('lang','?')}]")
    print("-" * 78)
    print(f"  標題：{row.get('title','')}")
    content = str(row.get("content", ""))
    print(f"  內文：{content[:220]}{'…' if len(content) > 220 else ''}")
    print("-" * 78)


def label_file(path: Path, only_unlabeled: bool):
    df = pd.read_csv(path, dtype=str)
    # 確保三欄存在
    for col in ("label", "keep", "fit_type"):
        if col not in df.columns:
            df[col] = ""

    todo = [i for i in range(len(df)) if not (only_unlabeled and is_labeled(df.iloc[i]))]
    if not todo:
        print(f"[{path.name}] 全部已標，跳過。")
        return 0

    print(f"\n### 開始標注 {path.name}（待標 {len(todo)} / 共 {len(df)} 筆）###")
    done = 0
    pos = 0
    while pos < len(todo):
        i = todo[pos]
        show_row(df.iloc[i], pos, len(todo), done)
        lab = prompt_label()

        if lab == "__q":
            print("\n存檔並退出。")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return done
        if lab == "__s":
            pos += 1
            continue
        if lab == "__b":
            pos = max(0, pos - 1)
            continue

        if lab in ("0", "1"):
            df.iat[i, df.columns.get_loc("label")] = lab
            df.iat[i, df.columns.get_loc("keep")] = str(prompt_keep())
            fit = prompt_fit()
            if fit is not None:
                df.iat[i, df.columns.get_loc("fit_type")] = fit
            done += 1
            # 逐筆寫回（關掉不丟）
            df.to_csv(path, index=False, encoding="utf-8-sig")
        pos += 1

    print(f"\n[{path.name}] 完成，本次標了 {done} 筆。")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="只標單一檔（檔名，如 scraped_kknews.csv）")
    ap.add_argument("--all", action="store_true", help="含已標的也重看（預設只看未標）")
    args = ap.parse_args()
    only_unlabeled = not args.all

    if args.file:
        files = [RAW_DIR / args.file]
    else:
        files = sorted(RAW_DIR.glob("scraped_*.csv"))
    if not files:
        sys.exit(f"找不到要標的檔案於 {RAW_DIR}")

    total = 0
    for p in files:
        if not p.exists():
            print(f"[skip] {p.name} 不存在")
            continue
        total += label_file(p, only_unlabeled)
    print(f"\n=== 本次共標注 {total} 筆 ===")


if __name__ == "__main__":
    main()
