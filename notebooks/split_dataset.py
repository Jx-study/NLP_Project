"""拆分標注後的爬蟲資料 → ood_test（平衡測試集）+ adversarial_base（對抗內文原料）。

拆分規則：
  - 來源：所有 scraped_*.csv 中 **keep=1 且已標 label(0/1)** 的列。
  - ood_test：四格（中/英 × label 0/1）**各抽相同 N 筆**（N = 最小格 × ood_ratio），保證平衡。
  - adversarial_base：其餘全部（含過剩的 label=0），當 Gemini 改寫的內文原料。
  - **全域去重 + 洩漏檢查**：同一篇（url / content_hash）絕不同時進兩批。
  - 隨機抽樣固定 seed，可復現。

輸出（dataset/raw_scraped/，已 gitignore）：
  ood_test.csv          intended_use=ood_test
  adversarial_base.csv  intended_use=adversarial_base

用法：
  python split_dataset.py                  # 預設 ood_ratio=0.8（每格 50×0.8=40）
  python split_dataset.py --ood-ratio 0.6  # 每格 50×0.6=30
  python split_dataset.py --seed 123
"""
import argparse
import glob
import io
import os
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw_scraped"
CELLS = [("zh", "0"), ("zh", "1"), ("en", "0"), ("en", "1")]


def load_labeled():
    """讀所有 scraped_*.csv，回傳 keep=1 且已標 label 的列（DataFrame）。"""
    frames = []
    for p in sorted(glob.glob(str(RAW_DIR / "scraped_*.csv"))):
        if os.path.getsize(p) < 50:
            continue
        frames.append(pd.read_csv(p, dtype=str))
    if not frames:
        sys.exit("找不到 scraped_*.csv")
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(str).str.strip()
    df["keep"] = df["keep"].astype(str).str.strip()
    df["lang"] = df["lang"].astype(str).str.strip()
    usable = df[(df["keep"] == "1") & (df["label"].isin(["0", "1"])) & (df["lang"].isin(["zh", "en"]))]
    return usable.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-ratio", type=float, default=0.8,
                    help="ood_test 每格抽樣比例（× 最小格）。預設 0.8")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_labeled()
    # 各格數量
    counts = {c: ((df["lang"] == c[0]) & (df["label"] == c[1])).sum() for c in CELLS}
    print("四格可用數量：")
    for c in CELLS:
        print(f"  {c[0]} label={c[1]}: {counts[c]}")
    min_cell = min(counts.values())
    per_cell = int(min_cell * args.ood_ratio)
    print(f"\n最小格 {min_cell} × ratio {args.ood_ratio} → ood_test 每格抽 {per_cell} 筆"
          f"（共 {per_cell * 4}）")
    if per_cell == 0:
        sys.exit("每格抽 0 筆，ratio 太小或資料不足")

    # 四格各抽 per_cell 進 ood_test，其餘進 adversarial_base
    ood_parts, used_idx = [], set()
    for c in CELLS:
        cell = df[(df["lang"] == c[0]) & (df["label"] == c[1])]
        picked = cell.sample(n=per_cell, random_state=args.seed)
        ood_parts.append(picked)
        used_idx.update(picked.index)
    ood = pd.concat(ood_parts).reset_index(drop=True)
    adv = df[~df.index.isin(used_idx)].reset_index(drop=True)

    ood = ood.copy(); adv = adv.copy()
    ood["intended_use"] = "ood_test"
    adv["intended_use"] = "adversarial_base"

    # 洩漏檢查：兩批 url / content_hash 不可交集
    def keyset(d, col):
        return set(d[col].dropna().astype(str).str.strip()) - {""}
    url_x = keyset(ood, "url") & keyset(adv, "url")
    hash_x = keyset(ood, "content_hash") & keyset(adv, "content_hash")
    assert not url_x, f"URL 洩漏：{len(url_x)} 筆同時在兩批"
    assert not hash_x, f"內文洩漏：{len(hash_x)} 筆同時在兩批"

    out_ood = RAW_DIR / "ood_test.csv"
    out_adv = RAW_DIR / "adversarial_base.csv"
    ood.to_csv(out_ood, index=False, encoding="utf-8-sig")
    adv.to_csv(out_adv, index=False, encoding="utf-8-sig")

    print(f"\n✅ 拆分完成（無洩漏）：")
    print(f"  ood_test:         {len(ood):>4} 筆 → {out_ood.name}（四格各 {per_cell}，平衡）")
    print(f"  adversarial_base: {len(adv):>4} 筆 → {out_adv.name}")
    print(f"\n  adversarial_base fit_type 分布：")
    print("   ", dict(adv["fit_type"].fillna("(空)").value_counts()))


if __name__ == "__main__":
    main()
