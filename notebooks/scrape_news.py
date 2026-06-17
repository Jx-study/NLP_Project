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
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup

# Windows/Anaconda console 預設 cp1252，中文/箭頭/✅ 會爆 UnicodeEncodeError。
# 強制 stdout/stderr 為 UTF-8（與 probe_*.py 慣例一致）。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 以「腳本所在位置」錨定，避免從不同工作目錄執行時去重讀到空集合而失效（reviewer #1）
OUT_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw_scraped"
CONTENT_MAXLEN = 800          # 內文只取前 N 字，夠 Gemini 改寫/判斷
CONTENT_MINLEN = 100          # 內文短於此 → 視為抓失敗，記入 _missing
REQUEST_TIMEOUT = 12
SLEEP_BETWEEN = 5
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

# schema：
#   label        ood_test 由人工標 0/1；adversarial_base 留空
#   keep         人工篩選用，1=可用 / 0=廢棄（壞資料/判不準），廢棄標 0 不刪
#   fit_type     標註這條適合改寫成哪種對抗句式：normal/transition/academic/none
#   content_hash **完整正文**的 md5（content 欄是截短 800 字，hash 用全文算）。
#                跨批次去重以此欄為準，避免前 N 字相近的不同文章誤判重複。
SCHEMA = ["source_site", "lang", "title", "content", "url", "pub_date",
          "intended_use", "label", "keep", "fit_type", "content_hash"]


@dataclass
class SiteConfig:
    name: str                       # source_site 值
    lang: str                       # zh / en
    rss_urls: list                  # 一個站可有多個分類 RSS
    content_selectors: list         # 內文 CSS selector（依序嘗試，取第一個命中）
    # 可選：要排除的 url 關鍵字（業配/影音/圖集頁）
    url_exclude: list = field(default_factory=list)
    # ── list-page 模式（給無有效 RSS 的農場站，如 kknews）──
    # 填了 list_page_urls 就走 list-page：抓列表頁 → 用 list_link_selector 取連結
    # → 用 list_link_pattern 正則過濾出文章頁。三者皆空則走 RSS（預設）。
    list_page_urls: list = field(default_factory=list)
    list_link_selector: str = "a[href]"   # 列表頁抓連結的 selector
    list_link_pattern: str = ""           # 文章頁 url 的正則（如 r"/n/\w+\.html"）


# ── 站點設定（範例站：ETtoday）─────────────────────────────
# 其他站照樣補一筆。RSS 網址與 selector 我無法聯網驗證，跑不通時按 url 微調。
SITES = {
    "ettoday": SiteConfig(
        name="ettoday",
        lang="zh",
        rss_urls=[
            "https://feeds.feedburner.com/ettoday/news",        # 即時新聞（實測 ~50 筆正常）
            "https://feeds.feedburner.com/ettoday/life",        # 生活（實測非法 XML，0 筆，已自動 skip）
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

    # ── UDN 聯合新聞網（主流 → 標注後多 label=0）🟢 list-page ──
    # ⚠️ 實測（2026-06）：udn 的 RSS 對 feedparser 無效——entries 有 20 筆但 title/link 全空殼，
    #    抽不出任何 url。故改走 list-page：列表頁 /news/index 有 story 連結，內文 selector 實測命中。
    "udn": SiteConfig(
        name="udn",
        lang="zh",
        rss_urls=[],                                          # RSS 對 feedparser 無效，走 list-page
        content_selectors=[
            "section.article-content__editor p",              # 實測命中（21 段 1180 字）
            "div.article-content__paragraph p",
            "article p",
        ],
        url_exclude=["/vipender", "/photo/", "video"],
        list_page_urls=[
            "https://udn.com/news/index",                     # 即時總覽（實測 108 篇有標題）
            "https://udn.com/news/breaknews/1",               # 即時新聞
        ],
        list_link_selector="a[href]",
        list_link_pattern=r"/news/story/\d+/\d+",             # udn 文章頁 url 樣式
    ),

    # ════════════════════════════════════════════════════════════════════
    # label=1（標題黨）主要來源：小報 / 內容農場
    # ⚠️ 以下 RSS 端點與 selector 為「合理猜測」，作者無法聯網實測；
    #    跑不通時請依實際頁面微調（與 ettoday/bbc 同屬待驗骨架）。
    #    每站旁標信心：🟢較穩 / 🟡可能要改 / 🔴很可能要改 list-page 模式
    # ════════════════════════════════════════════════════════════════════

    # ── BuzzFeed 英文（label=1 主力：清單體/誘餌標題密度高）🟡 ──
    # BuzzFeed 有 RSS，但 selector 隨改版變動機率高。
    "buzzfeed": SiteConfig(
        name="buzzfeed",
        lang="en",
        rss_urls=[
            "https://www.buzzfeed.com/index.xml",              # 全站
            "https://www.buzzfeed.com/world.xml",              # 分類（可能需調）
        ],
        content_selectors=[
            "div[data-module='subbuzz-text'] p",
            "article p",
            "main p",
        ],
        url_exclude=["/quiz/", "/shopping/", "/video/"],
    ),

    # ── Daily Mail 英文小報（label=1 主力：標題誘導但內文是真新聞）🟢 ──
    # 實測（2026-06）：RSS 150 entry、內文 div[itemprop=articleBody] 2614 字。比 buzzfeed 好用
    # （buzzfeed label=1 多是 shopping/quiz 內文廢；小報內文有料，keep=1 率高）。
    "dailymail": SiteConfig(
        name="dailymail",
        lang="en",
        rss_urls=[
            "https://www.dailymail.co.uk/news/index.rss",      # 新聞（實測 150 entry）
            "https://www.dailymail.co.uk/femail/index.rss",    # 生活（誘導密度高）
        ],
        content_selectors=[
            "div[itemprop='articleBody'] p",                   # 實測命中（2614 字）
            "div.article-text p",
            "article p",
        ],
        url_exclude=["/video/", "/gallery/", "/promotions/"],
    ),

    # ── NY Post 英文小報（label=1：美式小報，內文真新聞）🟢 ──
    # 實測（2026-06）：RSS 23 entry、內文 div.entry-content 3280 字。
    #
    # B4 修復（2026-06）：nypost 主站 feed 以體育/時政硬新聞為主（標題忠實→多 label=0），
    # 撈不到英文娛樂 clickbait 正例。真正的 `Is X Dying?`/`Ending Explained`/`Why Was X?`
    # curiosity-gap 標題集中在同集團的娛樂站 decider.com 與八卦站 pagesix.com
    # （見 scraped_nypost.csv id22/25/26/27 的 url）。三站同屬 NYP 集團、皆 WordPress
    # 架構，內文同為 div.entry-content，故併入同一站爬（source_site 統一記 nypost）。
    "nypost": SiteConfig(
        name="nypost",
        lang="en",
        rss_urls=[
            "https://nypost.com/feed/",
            "https://decider.com/feed/",       # 娛樂導流站：Ending Explained / Is X? 設問鉤子密度高
            "https://pagesix.com/feed/",       # 八卦站：名人懸念/誇飾標題
        ],
        content_selectors=[
            "div.entry-content p",                             # 實測命中（3280 字）；decider/pagesix 同為 WordPress entry-content
            "div.entry-content div p",                         # entry-content 內若段落被包進子 div（decider 常見）
            "div.single__content p",
            "article p",
        ],
        url_exclude=["/video/", "/photos/"],
    ),

    # ── 三立新聞 setn 中文（label=1：網路版小編標題黨句式多）🟢 list-page ──
    # ⚠️ 注意：三立雖是電視台，但網路版標題很標題黨（「這款」「1動作」「內幕曝」「疑」等
    #    資訊落差/懸念句式多）→ 標注後 label=1 比例高，補中文 label=1（實測確認）。
    #    RSS 不穩，走 list-page：ViewAll 頁有 News.aspx?NewsID= 連結，內文 div#Content1 命中。
    "setn": SiteConfig(
        name="setn",
        lang="zh",
        rss_urls=[],
        content_selectors=[
            "div#Content1 p",                                  # 實測命中（905 字）
            "article p",
            "div.page-text p",
        ],
        url_exclude=["/video", "/Photo"],
        # 分頁 p=1..4（實測 5 頁合計 122 篇 unique，p=5 幾乎無新；頁間有 SLEEP_BETWEEN 間隔）
        list_page_urls=[
            f"https://www.setn.com/ViewAll.aspx?PageGroupID=0&p={n}" for n in range(1, 5)
        ],
        list_link_selector="a[href]",
        list_link_pattern=r"/News\.aspx\?NewsID=\d+",          # 三立文章頁 url 樣式
    ),

    # ── Upworthy 英文（label=1：情感勾子/懸念標題）🟡 ──
    "upworthy": SiteConfig(
        name="upworthy",
        lang="en",
        rss_urls=[
            "https://www.upworthy.com/rss",                    # 全站 RSS（待驗）
        ],
        content_selectors=[
            "div.rm-article-body p",
            "article p",
            "main p",
        ],
        url_exclude=["/video/"],
    ),

    # ── 每日頭條 kknews 中文（典型內容農場，補誘導文風）🟢 list-page ──
    # 實測（2026-06）：RSS 非法 XML 不可用，但 list-page 可爬：
    #   列表頁 https://kknews.cc/zh-tw 有文章連結 /n/xxxxx.html；內文 selector `article p` 命中。
    "kknews": SiteConfig(
        name="kknews",
        lang="zh",
        rss_urls=[],                                          # 無有效 RSS，走 list-page
        content_selectors=[
            "article p",                                      # 實測命中（61 段 3620 字）
            "div.article-content p",
        ],
        url_exclude=["/video/", "/photo/"],
        # 首頁 + 根分頁 ?page=N（每頁約 15 篇有標題）。頁數限 4（≈60 篇）避免農場站 429 限流；
        # 頁間有 SLEEP_BETWEEN 間隔。要更多可分多次跑（去重會自動跳過已抓）。
        list_page_urls=(
            ["https://kknews.cc/"]
            + [f"https://kknews.cc/?page={n}" for n in range(2, 5)]    # page 2..4
        ),
        list_link_selector="a[href]",
        list_link_pattern=r"/n/\w+\.html",                    # 文章頁 url 樣式
    ),
}

def collect_entries_from_listpage(cfg: SiteConfig) -> list:
    """list-page 模式（給無有效 RSS 的農場站）：
    抓 list_page_urls 各頁 → 用 list_link_selector 取連結 → 用 list_link_pattern 過濾出文章頁。
    回傳 [{title, url, pub_date}]；title 暫用連結文字（內文頁有更準的標題可日後補，這裡先夠用）。
    """
    import re
    from urllib.parse import urljoin
    pat = re.compile(cfg.list_link_pattern) if cfg.list_link_pattern else None
    entries, seen = [], set()
    for idx, page in enumerate(cfg.list_page_urls):
        if idx > 0:
            time.sleep(SLEEP_BETWEEN)   # 頁間禮貌間隔，避免農場站 429 限流
        try:
            resp = requests.get(page, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if (resp.encoding or "").lower() in ("iso-8859-1", "latin-1", None):
                resp.encoding = resp.apparent_encoding or resp.encoding
            soup = BeautifulSoup(resp.text, "html.parser")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  [list-page 429] {page[:50]} 被限流（Too Many Requests）。"
                      f"請等數分鐘後重試，或調大 SLEEP_BETWEEN。停止抓後續頁。")
                break   # 已被限流，續抓只會更糟，直接停
            print(f"  [list-page error] {page[:60]} :: {e}")
            continue
        except Exception as e:
            print(f"  [list-page error] {page[:60]} :: {e}")
            continue
        for a in soup.select(cfg.list_link_selector):
            href = a.get("href", "")
            if not href:
                continue
            url = urljoin(page, href)            # 相對 → 絕對
            if pat and not pat.search(url):
                continue
            # 標題優先用連結文字，退而用 title 屬性或 img alt
            # （農場站列表常有「圖片連結 + 文字連結」兩個 <a> 指向同 url，圖片連結文字為空）
            title = a.get_text(" ", strip=True) or a.get("title", "").strip()
            if not title:
                img = a.find("img")
                if img:
                    title = (img.get("alt") or "").strip()
            if not title:
                continue   # 空標題連結（多為圖片連結）→ 跳過，等同 url 的文字連結來補
            if url in seen:
                continue
            seen.add(url)
            entries.append({
                "title": title,
                "url": url,
                "pub_date": "",                          # 列表頁通常無日期，留空
            })
    print(f"  list-page 取得 {len(entries)} 篇文章連結")
    return entries


def fetch_content(url: str, selectors: list) -> str:
    """進單一文章頁，依 selectors 依序嘗試抓內文，回傳**完整正文**（不截短）。抓不到回空字串。
    截短交給呼叫端：去重用完整正文 hash（避免前 N 字相近的不同文章誤判重複），輸出時才截 CONTENT_MAXLEN。
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()   # 403/404/反爬頁直接拋錯，不當成正文解析
        # 編碼：優先信任 HTTP header 宣告的 charset。
        # requests 在 header 無 charset 時會 fallback 成 ISO-8859-1（常錯），此時才改用 chardet 猜測。
        # 不可無條件用 apparent_encoding 覆蓋——chardet 對英文頁常誤猜（如把 utf-8 猜成 Windows-1254）→ mojibake。
        if (resp.encoding or "").lower() in ("iso-8859-1", "latin-1", None):
            resp.encoding = resp.apparent_encoding or resp.encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        for sel in selectors:
            paras = soup.select(sel)
            if paras:
                # 用 " " 連接 inline tag 間文字，避免 includingChloe 這種黏字
                text = " ".join(p.get_text(" ", strip=True) for p in paras)
                text = " ".join(text.split())   # 壓縮多餘空白
                if len(text) >= CONTENT_MINLEN:
                    return text   # 完整正文，截短由呼叫端負責
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
        # 排除 _missing 記錄檔，以及 _fixture（測試資料，勿污染正式去重）
        if p.name.startswith("_missing") or p.name.startswith("_fixture"):
            continue
        with open(p, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                u = (row.get("url") or "").strip()
                if u:
                    seen_urls.add(u)
                # 優先用 content_hash（全文）；舊檔無此欄則回退用截短 content 算（向後相容）
                h = (row.get("content_hash") or "").strip()
                if not h:
                    c = (row.get("content") or "").strip()
                    h = hashlib.md5(c.encode("utf-8")).hexdigest() if c else ""
                if h:
                    seen_hashes.add(h)
    return seen_urls, seen_hashes


def scrape_site(cfg: SiteConfig, intended_use, limit: int):
    """intended_use 可為 None（統一爬，標注後再拆分）→ 存成 scraped_{site}.csv、intended_use 欄留空。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = intended_use if intended_use else "scraped"
    out_path = OUT_DIR / f"{prefix}_{cfg.name}.csv"
    missing_path = OUT_DIR / f"_missing_content_{cfg.name}.csv"

    # 全域去重：掃所有現有輸出的 url，避免重複（含跨用途，標注後拆分時防洩漏）
    seen_urls, seen_hashes = load_global_seen()
    print(f"[{cfg.name}] 全域已存 {len(seen_urls)} url，目標再抓 {limit} 筆，use={prefix}")

    # 1. 取清單：有 list_page_urls 走 list-page 模式，否則走 RSS
    raw = []
    if cfg.list_page_urls:
        raw = collect_entries_from_listpage(cfg)
    else:
        for rss in cfg.rss_urls:
            feed = feedparser.parse(rss)
            if feed.bozo:
                print(f"  [RSS warn] {rss} 解析有問題：{feed.bozo_exception}")
                if not feed.entries:
                    # XML 本身非法（站點端問題，非腳本 bug）→ 提示換 feed 或改 list-page
                    print(f"  [RSS skip] {rss} 取到 0 筆（XML 非法）。"
                          f"此站可能需換有效 feed 或改用 list-page 模式。")
                    continue
            for e in feed.entries:
                raw.append({
                    "title": e.get("title", "").strip(),
                    "url": e.get("link", ""),
                    "pub_date": e.get("published", "") or e.get("updated", ""),
                })

    # 統一過濾：跳過無 url / 已存在（全域去重）/ 命中 url_exclude / 同次重複
    entries, seen = [], set()
    for e in raw:
        url = e["url"]
        if not url or url in seen_urls or url in seen:
            continue
        if any(x in url for x in cfg.url_exclude):
            continue
        seen.add(url)
        entries.append(e)
    entries = entries[:limit]
    print(f"  取得 {len(entries)} 筆待抓")

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
                # 去重用**完整正文** hash（前 N 字相近的不同文章不會誤判重複）；輸出才截短
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
                    "content": content[:CONTENT_MAXLEN],   # 寫檔截短
                    "url": e["url"],
                    "pub_date": e["pub_date"],
                    "intended_use": intended_use or "",   # 統一爬時留空，標注後拆分才填
                    "label": "",       # 一律後標（Gemini 初篩 + 人工複核）
                    "keep": "",        # 人工篩選填 1/0
                    "fit_type": "",    # 人工標 normal/transition/academic/none
                    "content_hash": h,  # 完整正文 hash，供跨批次去重
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
            if p.name.startswith("_fixture"):
                continue
            with open(p, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row.get("url"):
                        urls.add(row["url"].strip())
                    if row.get("title"):
                        titles.add(row["title"].strip())
                    # 優先用 content_hash（全文）；舊檔回退截短 content
                    h = (row.get("content_hash") or "").strip()
                    if not h:
                        c = (row.get("content") or "").strip()
                        h = hashlib.md5(c.encode("utf-8")).hexdigest() if c else ""
                    if h:
                        hashes.add(h)
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
    ap.add_argument("--use", choices=["adversarial_base", "ood_test"], default=None,
                    help="（可選）這批的用途。**新流程：統一爬不必填**，存成 scraped_{site}.csv、"
                         "intended_use 欄留空，標注後再拆分。填了則存成 {use}_{site}.csv（舊行為）。")
    ap.add_argument("--site", help="SITES 裡的站名，如 ettoday")
    ap.add_argument("--limit", type=int, default=50,
                    help="最多**嘗試**幾個 RSS entry（非保證寫出數）。"
                         "MISS／HTTP 失敗／內文重複會讓實際寫出 < limit，且不會自動往後補抓。")
    ap.add_argument("--verify-no-leak", action="store_true",
                    help="只跑跨批次洩漏檢查（訓練前必跑），不爬資料")
    args = ap.parse_args()

    if args.verify_no_leak:
        verify_no_leak()
        return

    if not args.site:
        raise SystemExit("請指定 --site（--use 可選），或用 --verify-no-leak 做洩漏檢查")
    if args.site not in SITES:
        raise SystemExit(f"未知站點 {args.site}，請先在 SITES 加 SiteConfig。已有：{list(SITES)}")
    scrape_site(SITES[args.site], args.use, args.limit)


if __name__ == "__main__":
    main()
