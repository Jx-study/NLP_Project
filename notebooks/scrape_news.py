"""新聞爬蟲骨架：RSS 拿清單 + requests 補內文（混合架構）。

用途（兩批，共用同一 schema）：
  adversarial_base — 對抗資料的內文來源（解 Bug 2 內文污染）。不需 label。
  ood_test        — 真實新聞泛化測試集。爬完由人工填 label（0/1）。

輸出 CSV schema：
  source_site, lang, title, content, url, pub_date, intended_use, label

設計：
  - 每個站只需在 SITES 加一筆 SiteConfig（RSS URL + 內文 CSS selector）
  - 通用流程：RSS 取 (title, url, date) → requests 進每個 url 補 content 前 N 字
  - rate limit、去重、斷點續跑、缺內文記錄，全部框架處理
  - 缺內文（selector 沒抓到 / 抓太短）→ 記到 _missing_content.csv，你按 url 手補

範例站：ETtoday（已附 RSS 與 selector）。其他站照 SiteConfig 補即可。

用法：
  python scrape_news.py --use adversarial_base --site ettoday --limit 50
  python scrape_news.py --use ood_test --site ettoday --limit 100
"""
import argparse
import csv
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup

# 以「腳本所在位置」錨定，避免從不同工作目錄執行時去重讀到空集合而失效（reviewer #1）
OUT_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw_scraped"
CONTENT_MAXLEN = 500          # 內文只取前 N 字，夠 Gemini 改寫/判斷
CONTENT_MINLEN = 100          # 內文短於此 → 視為抓失敗，記入 _missing
REQUEST_TIMEOUT = 12
SLEEP_BETWEEN = 1.5           # 禮貌性 rate limit（秒）
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

# schema：
#   label    ood_test 由人工標 0/1；adversarial_base 留空
#   keep     人工篩選用，1=可用 / 0=廢棄（壞資料/判不準），廢棄標 0 不刪
#   fit_type 標註這條適合改寫成哪種對抗句式：normal/transition/academic/none
SCHEMA = ["source_site", "lang", "title", "content", "url", "pub_date",
          "intended_use", "label", "keep", "fit_type"]


@dataclass
class SiteConfig:
    name: str                       # source_site 值
    lang: str                       # zh / en
    rss_urls: list                  # 一個站可有多個分類 RSS
    content_selectors: list         # 內文 CSS selector（依序嘗試，取第一個命中）
    # 可選：要排除的 url 關鍵字（業配/影音/圖集頁）
    url_exclude: list = field(default_factory=list)


# ── 站點設定（範例站：ETtoday）─────────────────────────────
# 其他站照樣補一筆。RSS 網址與 selector 我無法聯網驗證，跑不通時按 url 微調。
SITES = {
    "ettoday": SiteConfig(
        name="ettoday",
        lang="zh",
        rss_urls=[
            "https://feeds.feedburner.com/ettoday/news",        # 即時新聞
            "https://feeds.feedburner.com/ettoday/life",        # 生活
        ],
        # ETtoday 內文常見容器；依序嘗試
        content_selectors=[
            "div.story p",
            "div[itemprop='articleBody'] p",
            "article p",
        ],
        url_exclude=["/video/", "/album/", "supplier"],   # 影音/圖集/業配
    ),

    # ── BBC 英文（RSS 網址穩定；selector 可能隨改版微調）──
    "bbc": SiteConfig(
        name="bbc",
        lang="en",
        rss_urls=[
            "http://feeds.bbci.co.uk/news/rss.xml",            # 頭條
            "http://feeds.bbci.co.uk/news/world/rss.xml",      # 國際
        ],
        content_selectors=[
            "article div[data-component='text-block'] p",
            "main article p",
            "article p",
        ],
        url_exclude=["/videos/", "/av/", "/live/", "/sport/"],
    ),

    # ── UDN 聯合新聞網（主流 → label=0 為主）──
    # ⚠️ 實測：下面這個 6638 端點回傳的 item 全是空的（title/link 皆空、日期 1970），
    #    取到 0 筆。請至 https://udn.com/rss 找有效的分類 feed 網址替換。
    #    （feed XML 本身有效，是該端點內容為空，非腳本問題）
    "udn": SiteConfig(
        name="udn",
        lang="zh",
        rss_urls=[
            "https://udn.com/rssfeed/news/2/6638?ch=news",     # ❌ 空 feed，待換
        ],
        content_selectors=[
            "section.article-content__editor p",
            "div.article-content__paragraph p",
            "article p",
        ],
        url_exclude=["/vipender", "/photo/", "video"],
    ),

    # ── 待補：內容農場（OOD 的 label=1 主要來源）──
    # 農場站多無標準 RSS，可能需改用首頁列表頁解析（requests + 列表 selector）。
    # 若該站有 RSS 就照上面格式填；沒有則另寫 list-page 模式。
    # "coco01": SiteConfig(name="coco01", lang="zh", rss_urls=[...], content_selectors=[...]),
}


def fetch_content(url: str, selectors: list) -> str:
    """進單一文章頁，依 selectors 依序嘗試抓內文，回傳前 CONTENT_MAXLEN 字。抓不到回空字串。"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.encoding = resp.apparent_encoding or resp.encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        for sel in selectors:
            paras = soup.select(sel)
            if paras:
                text = " ".join(p.get_text(strip=True) for p in paras)
                text = text.strip()
                if len(text) >= CONTENT_MINLEN:
                    return text[:CONTENT_MAXLEN]
        return ""  # selector 都沒命中 / 太短
    except Exception as e:
        print(f"    [fetch error] {url[:60]} :: {e}")
        return ""


def load_global_seen() -> tuple:
    """全域去重（reviewer #1：避免同一 URL 進入 adversarial_base 與 ood_test 兩批）。

    掃描 OUT_DIR 下**所有** {use}_{site}.csv，回傳已出現的 (url 集合, content-hash 集合)。
    新抓的 entry 只要 url 或 content hash 命中任一，就跳過——跨批次、跨站點皆去重。
    """
    seen_urls, seen_hashes = set(), set()
    if not OUT_DIR.exists():
        return seen_urls, seen_hashes
    for p in OUT_DIR.glob("*.csv"):
        if p.name.startswith("_missing"):
            continue
        with open(p, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                u = (row.get("url") or "").strip()
                if u:
                    seen_urls.add(u)
                c = (row.get("content") or "").strip()
                if c:
                    seen_hashes.add(hashlib.md5(c.encode("utf-8")).hexdigest())
    return seen_urls, seen_hashes


def scrape_site(cfg: SiteConfig, intended_use: str, limit: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{intended_use}_{cfg.name}.csv"
    missing_path = OUT_DIR / f"_missing_content_{cfg.name}.csv"

    # 全域去重：掃所有現有輸出（含另一批 use）的 url，避免跨批次洩漏（reviewer #1）
    seen_urls, seen_hashes = load_global_seen()
    print(f"[{cfg.name}] 全域已存 {len(seen_urls)} url，目標再抓 {limit} 筆，use={intended_use}")

    # 1. RSS 取清單（url 命中任何現有批次就跳過）
    entries = []
    for rss in cfg.rss_urls:
        feed = feedparser.parse(rss)
        if feed.bozo:
            print(f"  [RSS warn] {rss} 解析有問題：{feed.bozo_exception}")
        for e in feed.entries:
            url = e.get("link", "")
            if not url or url in seen_urls:
                continue
            if any(x in url for x in cfg.url_exclude):
                continue
            entries.append({
                "title": e.get("title", "").strip(),
                "url": url,
                "pub_date": e.get("published", "") or e.get("updated", ""),
            })
    # 去重（同次 url）
    seen, uniq = set(), []
    for e in entries:
        if e["url"] not in seen:
            seen.add(e["url"])
            uniq.append(e)
    entries = uniq[:limit]
    print(f"  RSS 取得 {len(entries)} 筆待抓")

    # 2. 逐篇補內文 + 寫出
    new_file = not out_path.exists()
    miss_new = not missing_path.exists()
    written = missing = 0
    with open(out_path, "a", newline="", encoding="utf-8-sig") as fout, \
         open(missing_path, "a", newline="", encoding="utf-8-sig") as fmiss:
        w = csv.DictWriter(fout, fieldnames=SCHEMA)
        wm = csv.DictWriter(fmiss, fieldnames=["source_site", "title", "url", "pub_date"])
        if new_file:
            w.writeheader()
        if miss_new:
            wm.writeheader()

        dup_content = 0
        for i, e in enumerate(entries, 1):
            content = fetch_content(e["url"], cfg.content_selectors)
            if content:
                # 內文 hash 去重：同內容（不同 url）也視為重複，跨批次防洩漏
                h = hashlib.md5(content.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    dup_content += 1
                    print(f"  {i}/{len(entries)} DUP（內文已存在於其他批次/站點）{e['title'][:30]}")
                    continue
                seen_hashes.add(h)
                w.writerow({
                    "source_site": cfg.name,
                    "lang": cfg.lang,
                    "title": e["title"],
                    "content": content,
                    "url": e["url"],
                    "pub_date": e["pub_date"],
                    "intended_use": intended_use,
                    "label": "",       # ood_test 由人工填；adversarial_base 留空
                    "keep": "",        # 人工篩選填 1/0
                    "fit_type": "",    # 人工標 normal/transition/academic/none
                })
                written += 1
            else:
                wm.writerow({"source_site": cfg.name, "title": e["title"],
                             "url": e["url"], "pub_date": e["pub_date"]})
                missing += 1
                print(f"  {i}/{len(entries)} MISS {e['title'][:40]}")
            time.sleep(SLEEP_BETWEEN)

    print(f"\n[{cfg.name}] 寫出 {written} 筆 → {out_path}")
    if dup_content:
        print(f"[{cfg.name}] {dup_content} 筆因內文與其他批次重複而跳過（去重生效）")
    if missing:
        print(f"[{cfg.name}] {missing} 筆抓不到內文 → {missing_path}（按 url 手補）")


def verify_no_leak():
    """跨批次洩漏檢查（reviewer #1）：adversarial_base 與 ood_test 的
    url / 標題 / 內文 hash 交集必須為 0。訓練前務必執行此 assertion。
    """
    def collect(use):
        urls, titles, hashes = set(), set(), set()
        for p in OUT_DIR.glob(f"{use}_*.csv"):
            with open(p, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row.get("url"):
                        urls.add(row["url"].strip())
                    if row.get("title"):
                        titles.add(row["title"].strip())
                    c = (row.get("content") or "").strip()
                    if c:
                        hashes.add(hashlib.md5(c.encode("utf-8")).hexdigest())
        return urls, titles, hashes

    a_url, a_title, a_hash = collect("adversarial_base")
    o_url, o_title, o_hash = collect("ood_test")
    url_x, title_x, hash_x = a_url & o_url, a_title & o_title, a_hash & o_hash
    print(f"adversarial_base: {len(a_url)} url | ood_test: {len(o_url)} url")
    print(f"交集 — url:{len(url_x)}  title:{len(title_x)}  content-hash:{len(hash_x)}")
    assert not url_x, f"URL 洩漏！{len(url_x)} 筆同時在兩批：{list(url_x)[:3]}"
    assert not title_x, f"標題洩漏！{len(title_x)} 筆同時在兩批：{list(title_x)[:3]}"
    assert not hash_x, f"內文洩漏！{len(hash_x)} 筆內文同時在兩批"
    print("✅ 無洩漏：adversarial_base 與 ood_test 交集為 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use", choices=["adversarial_base", "ood_test"], help="這批爬來做什麼用")
    ap.add_argument("--site", help="SITES 裡的站名，如 ettoday")
    ap.add_argument("--limit", type=int, default=50, help="最多抓幾筆")
    ap.add_argument("--verify-no-leak", action="store_true",
                    help="只跑跨批次洩漏檢查（訓練前必跑），不爬資料")
    args = ap.parse_args()

    if args.verify_no_leak:
        verify_no_leak()
        return

    if not args.use or not args.site:
        raise SystemExit("請指定 --use 與 --site，或用 --verify-no-leak 做洩漏檢查")
    if args.site not in SITES:
        raise SystemExit(f"未知站點 {args.site}，請先在 SITES 加 SiteConfig。已有：{list(SITES)}")
    scrape_site(SITES[args.site], args.use, args.limit)


if __name__ == "__main__":
    main()
