import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.scale_lexicon import (
    hits_scale_lexicon,
    scale_level,
    scale_mismatch,
    scale_lexicon_score,
)


# ── hits_scale_lexicon (legacy bool API) ────────────────────────
def test_chinese_phrase_hits():
    assert hits_scale_lexicon("史上最強的方法") is True
    assert hits_scale_lexicon("震驚全國的消息") is True
    assert hits_scale_lexicon("台北市最強滷肉飯") is True


def test_english_phrase_hits():
    assert hits_scale_lexicon("The world's greatest discovery") is True
    assert hits_scale_lexicon("A historic moment") is True


def test_bare_world_does_not_hit():
    # 「世界各國」這種敘述性用法不應該觸發；要與「最/冠軍/第一」等搭配才算
    assert hits_scale_lexicon("世界各國反應不一") is False
    assert hits_scale_lexicon("台灣的世界排名") is False


def test_world_champion_phrase_hits():
    # 「世界冠軍」是規模宣稱片語，應該命中（後續由 mismatch 判斷是否真誇大）
    assert hits_scale_lexicon("我是世界冠軍") is True


def test_no_scale_word_misses():
    assert hits_scale_lexicon("我在學校運動會得第一名") is False
    assert hits_scale_lexicon("Local team wins school match") is False


def test_english_case_insensitive():
    assert hits_scale_lexicon("THE WORLD'S BEST") is True


def test_empty_title():
    assert hits_scale_lexicon("") is False


# ── scale_level: 0~5 階層偵測 ─────────────────────────────────
def test_scale_level_world():
    assert scale_level("史上最強的產品") == 5
    assert scale_level("全球第一") == 5
    assert scale_level("the world's best phone") == 5


def test_scale_level_national():
    assert scale_level("震驚全國的事件") == 4
    assert scale_level("舉國歡騰") == 4
    assert scale_level("a nationwide protest") == 4


def test_scale_level_regional():
    assert scale_level("台北市最強滷肉飯") == 3
    assert scale_level("新竹縣最大停車場") == 3
    assert scale_level("全省走透透") == 3


def test_scale_level_local():
    assert scale_level("學校運動會金牌") == 2
    assert scale_level("社區公告") == 2
    assert scale_level("local school event") == 2


def test_scale_level_personal():
    assert scale_level("我家後院") == 1
    assert scale_level("跟我同學一起") == 1


def test_scale_level_none():
    assert scale_level("今天天氣不錯") == 0
    assert scale_level("") == 0


# ── scale_mismatch: title_level - content_level >= 1 ─────────
def test_mismatch_world_title_local_content():
    # 經典 clickbait：標題「世界最」，內文「學校」
    assert scale_mismatch("史上最強運動員", "他在學校運動會拿到金牌") is True


def test_mismatch_regional_title_local_content():
    # 你提的案例：標題「市最強」、內文「學校」
    assert scale_mismatch("台北市最強跑者", "他在學校運動會奪冠") is True


def test_mismatch_world_title_personal_content():
    assert scale_mismatch("世界最強廚師", "我家媽媽煮的最好吃") is True


def test_no_mismatch_when_levels_align():
    # 標題、內文同層級 → 不視為 mismatch
    assert scale_mismatch("史上最強手機", "全球銷量破億，創下紀錄") is False
    assert scale_mismatch("全國最佳大學", "舉國公認的學術重鎮") is False


def test_no_mismatch_when_content_higher_than_title():
    # 標題小、內文大 → 不算 mismatch（不是誇大）
    assert scale_mismatch("學校運動會", "全國高中生齊聚一堂") is False


def test_no_mismatch_when_title_has_no_scale():
    # 標題沒有規模宣稱 → 不算 mismatch
    assert scale_mismatch("今天天氣不錯", "我在學校吃午餐") is False


def test_mismatch_empty_inputs():
    assert scale_mismatch("", "") is False
    assert scale_mismatch("史上最強", "") is False  # content 空 → 無證據可比


# ── scale_lexicon_score: 0/1/2/3 ─────────────────────────────
def test_score_no_match():
    assert scale_lexicon_score("今天天氣不錯", "我在學校吃午餐") == 0


def test_score_content_only_match():
    # 大詞只出現在 content（很可能是引用/報導）→ 1 分
    assert scale_lexicon_score("學校公告", "今天有全國比賽轉播") == 1


def test_score_title_bare_word():
    # 標題出現規模相關詞但非完整片語 → 2 分（這層級在新設計下其實偏少用）
    # 用 regional 觸發但內文沒矛盾的情境
    assert scale_lexicon_score("台北市最強小吃", "市民推薦的老店") == 2


def test_score_title_phrase_or_mismatch():
    # 標題明確片語命中 + 內文 mismatch → 3 分
    assert scale_lexicon_score("史上最強運動員", "他在學校運動會拿金牌") == 3
    assert scale_lexicon_score("台北市最強跑者", "他在學校運動會奪冠") == 3


def test_score_empty():
    assert scale_lexicon_score("", "") == 0
