"""誇大宣稱偵測：分層級判斷標題與內文的規模落差。

層級定義（0~5）：
    5 World      - 世界最、史上、全球第一、world's best、historic
    4 National   - 全國、舉國、nationwide
    3 Regional   - xx市最、xx縣最、全省、全市
    2 Local      - 學校、社區、村、local、school
    1 Personal   - 我、我家、同學
    0 None       - 無規模指涉

對外 API：
    hits_scale_lexicon(title)         舊 bool API：title 含明確誇大片語時 True
    scale_level(text)                  回傳 0~5 中最高的層級
    scale_mismatch(title, content)     title_level - content_level >= 1 → True
    scale_lexicon_score(title, content) 0/1/2/3，給 cascade 決策用
"""

import re

# ── World (5) ─────────────────────────────────────────────
# 中文以片語（含「最」「第一」等）為主，避免「世界冠軍」這類誤觸發
_ZH_WORLD_PATTERNS = [
    r"史上(最|空前|首次)",
    r"空前絕後",
    r"舉世(無雙|聞名|矚目)",
    r"(世界|全球)(最|第一|之最|級|頂級|頂尖|冠軍|紀錄)",
    r"震驚(世界|全球)",
]
_EN_WORLD_PATTERNS = [
    r"world'?s\s+(best|greatest|biggest|first|largest|most)",
    r"\bever\b",
    r"\bhistoric\b",
    r"\brecord[- ]breaking\b",
    r"\bgreatest\s+of\s+all\s+time\b",
]

# ── National (4) ─────────────────────────────────────────
_ZH_NATIONAL_PATTERNS = [
    r"震驚全國",
    r"舉國(歡騰|震驚|關注)",
    r"全國(最|第一|轟動|矚目|皆知)",
    r"(台灣|中國|日本|美國|韓國)(最|第一)",
]
_EN_NATIONAL_PATTERNS = [
    r"\bnationwide\b",
    r"\bnational(ly)?\s+(record|best|first|largest)\b",
]

# ── Regional (3) ─────────────────────────────────────────
# 「xx市最」「xx縣最」用 regex 捕捉
_ZH_REGIONAL_PATTERNS = [
    r"[一-鿿]{1,4}(市|縣|州|省)(最|第一)",
    r"全(省|市|縣|州)(最|皆知|轟動)?",
]
_EN_REGIONAL_PATTERNS = [
    r"\bstate(wide)?\s+(record|best|first)\b",
    r"\bcity(wide)?\s+(record|best|first)\b",
]

# ── Local (2) — 多作為 content 限定詞 ─────────────────────
# 「鎮」太容易誤觸發（重鎮、坐鎮…），移除單字版
_ZH_LOCAL_WORDS = ("學校", "社區", "校內", "全校", "班上")
_EN_LOCAL_WORDS = ("school", "local ", "neighborhood", "village", "campus")

# ── Personal (1) ─────────────────────────────────────────
_ZH_PERSONAL_WORDS = ("我家", "我的", "我跟", "我和", "同學", "朋友")
_EN_PERSONAL_WORDS = ("my friend", "my family", "my school", "i went", "i was")

# ── Content-only 規模指涉詞 ──────────────────────────────
# 出現在內文時表示「事件規模」，不需要誇大片語就算數
_ZH_CONTENT_WORLD = ("世界", "全球", "國際")
_ZH_CONTENT_NATIONAL = ("全國", "舉國", "台灣", "中國", "日本", "美國", "韓國")
_ZH_CONTENT_REGIONAL = ("市", "縣", "州", "省", "全省", "全市")
_EN_CONTENT_WORLD = ("worldwide", "global", "international")
_EN_CONTENT_NATIONAL = ("nationwide", "national")
_EN_CONTENT_REGIONAL = ("statewide", "citywide", "regional")


def _match_any(text: str, patterns) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def _contains_any(text_lower: str, words) -> bool:
    return any(w in text_lower for w in words)


def scale_level(text: str, mode: str = "title") -> int:
    """回傳文字中偵測到的最高規模層級（0~5）。

    mode="title"   嚴格模式：規模 3~5 須命中誇大片語（含「最」「第一」等）
    mode="content" 寬鬆模式：單字「世界/全國/市」等也算規模指涉
    """
    if not text:
        return 0
    low = text.lower()

    if _match_any(text, _ZH_WORLD_PATTERNS) or _match_any(low, _EN_WORLD_PATTERNS):
        return 5
    if _match_any(text, _ZH_NATIONAL_PATTERNS) or _match_any(low, _EN_NATIONAL_PATTERNS):
        return 4
    if _match_any(text, _ZH_REGIONAL_PATTERNS) or _match_any(low, _EN_REGIONAL_PATTERNS):
        return 3

    # content 模式：補抓單字規模指涉
    if mode == "content":
        if _contains_any(text, _ZH_CONTENT_WORLD) or _contains_any(low, _EN_CONTENT_WORLD):
            return 5
        if _contains_any(text, _ZH_CONTENT_NATIONAL) or _contains_any(low, _EN_CONTENT_NATIONAL):
            return 4
        if _contains_any(text, _ZH_CONTENT_REGIONAL) or _contains_any(low, _EN_CONTENT_REGIONAL):
            return 3

    if _contains_any(text, _ZH_LOCAL_WORDS) or _contains_any(low, _EN_LOCAL_WORDS):
        return 2
    if _contains_any(text, _ZH_PERSONAL_WORDS) or _contains_any(low, _EN_PERSONAL_WORDS):
        return 1
    return 0


def hits_scale_lexicon(title: str, content: str = "") -> bool:
    """舊 bool API：標題含 regional 以上的誇大片語才為 True。

    注意：純 local/personal 詞（學校、我家）不算誇大，回傳 False。
    """
    return scale_level(title) >= 3


def scale_mismatch(title: str, content: str) -> bool:
    """標題的規模宣稱高於內文證據 → True。

    判定：title_level >= 3（有實際誇大宣稱）且 content_level 在 1~2（個人/地方限定詞）
    且 title_level - content_level >= 1。
    """
    if not title or not content:
        return False
    t_level = scale_level(title, mode="title")
    c_level = scale_level(content, mode="content")
    if t_level < 3:
        return False  # title 沒有誇大宣稱，不存在 mismatch
    if c_level == 0 or c_level >= t_level:
        return False  # content 無規模指涉，或規模 >= title
    return (t_level - c_level) >= 1


def scale_lexicon_score(title: str, content: str) -> int:
    """0~3 分，供 cascade 決策用。

    3 = 標題明確片語命中 + 內文 mismatch（最高風險）
    2 = 標題命中（regional 以上）但內文無矛盾
    1 = 只有內文命中（可能是引用/報導）
    0 = 沒命中
    """
    t_level = scale_level(title, mode="title")
    c_level = scale_level(content, mode="content")

    if t_level >= 3 and scale_mismatch(title, content):
        return 3
    if t_level >= 3:
        return 2
    if c_level >= 3:
        return 1
    return 0
