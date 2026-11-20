import pytest

from normalize import normalize_text  # adjust the import to your layout

Z = "\u200c"

TABLE_1 = [
    ("خوب بود<br>ممنون", "خوب بود ممنون"),
    ("ببینید https://x.com/a?b=1 عالی بود", "ببینید عالی بود"),
    ("خوب بود\nممنون\r\nعالی\tبود", "خوب بود ممنون عالی بود"),
    (f"می{Z}شود  خوب   است", f"می{Z}شود خوب است"),
    ("كيفيت ممتاز", "کیفیت ممتاز"),
    ("خـــوب و زیبَا", "خوب و زیبا"),
    ("٥ ستاره و 5 ستاره", "۵ ستاره و ۵ ستاره"),
    ("عالیییییی و ههههه", "عالیی و هه"),
    ("1000 و ۱۰۰۰", "۱۰۰۰ و ۱۰۰۰"),
    ("عالی!!!!!! بد؟؟؟؟", "عالی!! بد؟؟"),
    ("https://only.link/abc", None),
    (f"توصیه نمی{Z}کنم نه", f"توصیه نمی{Z}کنم نه"),
    ("nano na", "nano na"),
    ("عــــالی", "عالی"),
    (f"می{Z}شود", f"می{Z}شود"),
    ("لب ھارو نرم میکنہ و قیمت مناسب", "لب هارو نرم میکنه و قیمت مناسب"),
    ("علامت ؟سوال است", "علامت؟ سوال است"),
    (r"می\u200Cتونست به", f"می{Z}تونست به"),        # literal escape text -> real ZWNJ
]

TABLE_2 = [
    (":)))))", ":))"),
    (": ((", ": (("),
    (": ( !", ": ( !"),
    ('گفتم: "سلام"', 'گفتم: " سلام "'),
    ('گفتی :"سیب"', 'گفتی: " سیب "'),
    ("خوب....", "خوب..."),
    ("خوب..", "خوب..."),
    ("خوب...سلام", "خوب... سلام"),
    ("1,000 و 3.5 و 12:30 و a.com", "۱۰۰۰ و ۳.۵ و ۱۲:۳۰ و a.com"),
    ("سلام ,خوب ;بد ?چرا", "سلام، خوب؛ بد؟ چرا"),
    ("{سلام} و [خوب]", "[ سلام ] و [ خوب ]"),
    ("گفت «سلام» و “خوب”", 'گفت " سلام " و " خوب "'),
    ("(سلام).", "( سلام )."),
    ("خوب)))", "خوب )"),
    ("عالی😍😍😍😍 بود", "عالی 😍😍 بود"),   # differs from the old table: emoji step runs last
    ("😍عالی", "😍 عالی"),
    ("قیمت: ۵۰٪ — خوب _ بد", "قیمت: ۵۰% - خوب - بد"),
    ("ممنون!!!!!! عالی !", "ممنون!! عالی!"),
    ("5 < 6 > 4", "۵ < ۶ > ۴"),
]

EMOJI_CASES = [
    ("قشنگه😍🔥 ممنون", "قشنگه 😍🔥 ممنون"),
    ("کیفیت خوبه👍🏼 فقط", "کیفیت خوبه 👍🏼 فقط"),
    ("خوبه!😍", "خوبه!😍"),
]


@pytest.mark.parametrize("src,expected", TABLE_1 + TABLE_2 + EMOJI_CASES)
def test_examples(src, expected):
    assert normalize_text(src) == expected


@pytest.mark.parametrize("src", [None, "", "   ", "\n\t"])
def test_empty_inputs(src):
    assert normalize_text(src) is None


def test_idempotent():
    for src, _ in TABLE_1 + TABLE_2:
        once = normalize_text(src)
        assert normalize_text(once) == once


V2_CASES = [
    (f"خانه{Z}ی من", f"خانه{Z}ی من"),                    # ی after ZWNJ survives
    (f"خانه{Z} من", "خانه من"),                          # stray ZWNJ
    ("\u0627\u0653\u0628", "\u0622\u0628"),              # decomposed آ recomposed
    ("\ufecb\ufe8e\ufedf\ufbfd", "عالی"),                # presentation forms
    ("من ا خوبم", "من خوبم"),                            # real stray letter still removed
]

@pytest.mark.parametrize("src,expected", V2_CASES)
def test_v2(src, expected):
    assert normalize_text(src) == expected

@pytest.mark.parametrize("src,expected", [
    ("کتاب ها و عکس های قشنگ", f"کتاب{Z}ها و عکس{Z}های قشنگ"),
    ("نمی کنم", f"نمی{Z}کنم"),
    ("۳ ها", "۳ ها"),
])
def test_join(src, expected):
    assert normalize_text(src, join_suffixes=True) == expected