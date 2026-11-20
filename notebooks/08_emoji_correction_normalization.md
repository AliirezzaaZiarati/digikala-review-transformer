# Emoji Correction — Text Normalization

> This document contains only the notebook cells directly related to detecting and correcting emojis attached to words, plus the post-correction diagnostic scan. General setup and the separate DuckDB comparison cell are omitted.


## Emoji-spacing correction


### 1. Identify and quantify emojis glued to words


```python
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Locate the repo root by walking up until project.duckdb is found
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "project.duckdb").exists())
SPLIT_DIR = ROOT / "data" / "processed"
SPLIT_NAMES = ["train_supervised", "train_pseudo_unlabeled", "validation", "test"]

# One emoji character (main emoji/symbol blocks plus variation selector)
EMOJI = "[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
# One word character (letter, digit, underscore, or ZWNJ)
WORD = r"[\w\u200c]"

# Zero-width boundaries: they match only the emoji itself
EMOJI_AFTER_WORD = rf"(?<={WORD}){EMOJI}"
EMOJI_BEFORE_WORD = rf"{EMOJI}(?={WORD})"
EMOJI_GLUED = rf"{EMOJI_AFTER_WORD}|{EMOJI_BEFORE_WORD}"

summary = []
for name in SPLIT_NAMES:
    df = pd.read_csv(SPLIT_DIR / f"{name}.csv", dtype=str)
    body = df["body"].fillna("")

    print(f"\n=== {name} ===")
    print("columns:", list(df.columns), "| rows:", len(df))

    has_after_word = body.str.contains(EMOJI_AFTER_WORD, regex=True)
    has_before_word = body.str.contains(EMOJI_BEFORE_WORD, regex=True)
    has_glued = has_after_word | has_before_word

    glued_body = body[has_glued]
    n_spaces = glued_body.str.count(EMOJI_AFTER_WORD).sum() + glued_body.str.count(EMOJI_BEFORE_WORD).sum()

    kinds = Counter()
    for text in glued_body:
        kinds.update(re.findall(EMOJI_GLUED, text))

    print(f"rows with word+emoji glued   : {has_after_word.sum():,}")
    print(f"rows with emoji+word glued   : {has_before_word.sum():,}")
    print(f"rows with either (to edit)   : {has_glued.sum():,} ({has_glued.mean() * 100:.2f}%)")
    print(f"total spaces to insert       : {n_spaces:,}")
    print("distinct emojis:", len(kinds))
    print("top 15 emojis  :", kinds.most_common(15))
    print("samples:")
    for text in glued_body.head(3):
        print("  ", text[:150])

    summary.append({
        "file": name,
        "rows": len(df),
        "rows_to_edit": int(has_glued.sum()),
        "spaces_to_insert": int(n_spaces),
    })

print()
print(pd.DataFrame(summary))
```


**Output**

```text

=== train_supervised ===
columns: ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'] | rows: 2860236
rows with word+emoji glued   : 25,377
rows with emoji+word glued   : 1
rows with either (to edit)   : 25,377 (0.89%)
total spaces to insert       : 26,463
distinct emojis: 458
top 15 emojis  : [('😍', 3506), ('👌', 3159), ('👍', 2170), ('❤', 1174), ('🙏', 1088), ('😂', 1015), ('😐', 728), ('😊', 661), ('🥰', 636), ('😁', 617), ('🌹', 474), ('😅', 461), ('😑', 373), ('🤩', 346), ('🥲', 338)]
samples:
   هنوز استفاده نکردم ولی طرحش قشنگه😍🔥 لازم نیست حتماا اندازه ناخن ببرید. میتونید طرح و قیچی کنید بزارید رو ناخن بعدش با استون و براش یا گوش پاک کن دورشو
   درود تراش سالم به دستم رسید و از کیفیت قابل قبولی هم برخورداره از تیغه ها راضی بودم فقط همونطور که در تصویر مشاهده میفرمایید من رنگ " بنفش " سفارش داد
   کیفیتش و سایزش خوبه ولی یجوریه انگار با گواش رنگش کردن🤦🏻♀️ احساس میکنی اگه با دست خیس بهش دست بزنی رنگش پاک میشه یا دستت رنگ میگیره... من خودم یه لایه

=== train_pseudo_unlabeled ===
columns: ['id', 'body'] | rows: 635824
rows with word+emoji glued   : 5,312
rows with emoji+word glued   : 2
rows with either (to edit)   : 5,312 (0.84%)
total spaces to insert       : 5,579
distinct emojis: 273
top 15 emojis  : [('😍', 705), ('👌', 625), ('👍', 425), ('😂', 235), ('🙏', 216), ('❤', 197), ('😐', 160), ('🥰', 137), ('😁', 135), ('😊', 123), ('🌹', 117), ('😅', 91), ('🤩', 90), ('😑', 83), ('👎', 77)]
samples:
   ترجمه و چاپ عالی و یک رمان کلاسیک بی نظیر👌🏻 فقط یه نکته اینکه قیمت پشت جلد کتاب ارزانتر بود!
   کیفیت جلد و کاغذ کتابهای نشر افق همیشه خیلی خوبه، ترجمه ها هم همیشه روان و خوب هست، اما قیمت روی جلد اکثر اوقات یکمی غیر منطقیه... کتاب های این مجموعه
   هنوز استفاده نکردم ولی سالم رسید فقط اینکه دیجی کالا یه کالای دیگه رو ارسال کرده اما به ضرر خودش کار کرده چون من این محصول رو که بسته دو عددی بود رو پ

=== validation ===
columns: ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'] | rows: 922794
rows with word+emoji glued   : 6,667
rows with emoji+word glued   : 0
rows with either (to edit)   : 6,667 (0.72%)
total spaces to insert       : 6,974
distinct emojis: 290
top 15 emojis  : [('😍', 958), ('👌', 777), ('👍', 643), ('🙏', 305), ('😂', 290), ('❤', 282), ('😐', 190), ('😊', 169), ('😁', 167), ('🌹', 141), ('🥰', 136), ('😅', 108), ('😑', 104), ('🥲', 94), ('🙂', 90)]
samples:
   بنظرم عالیه خیلی قشنگه🥺🥺
   عالی در کیفیت و بعد از استفاده بلافاصله لب ها را نرم می‌کند پیشنهاد میکنم قبل از خواب استفاده کنید🤩🤩
   خیلی خوبه من جلدهای دیگه اش رو هم دارم🤍

=== test ===
columns: ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'] | rows: 922805
rows with word+emoji glued   : 6,385
rows with emoji+word glued   : 1
rows with either (to edit)   : 6,385 (0.69%)
total spaces to insert       : 6,705
distinct emojis: 286
top 15 emojis  : [('😍', 854), ('👌', 810), ('👍', 582), ('🙏', 280), ('❤', 274), ('😂', 272), ('😐', 215), ('😁', 181), ('🥰', 165), ('😊', 162), ('🌹', 113), ('😅', 112), ('😑', 102), ('👎', 80), ('🤩', 79)]
samples:
   من دوستش نداشتم و راضی نبودم اصلا خوب نگه نداشت و فر موهام تا پایان مراسم باز شد باید انقد زیااد بزنین که حالت خشکی بگیره موها😑
   دوستای گلم من هر ۴ مدل رژهای کالیستا رو دارم و میتونم بگم از میون اونا هیدراکالر خوش بافت ترین و خوشبوترین هست‌. ماته اما بافت بسیار سبک و روانی داره‌
   کیفیت کتاب نسبت به قیمت خیلی خوبه👍🏼 فقط اندازه اش نسبت به کتاب‌های اصل کمی کوچیک تره

                     file     rows  rows_to_edit  spaces_to_insert
0        train_supervised  2860236         25377             26463
1  train_pseudo_unlabeled   635824          5312              5579
2              validation   922794          6667              6974
3                    test   922805          6385              6705
```


### 2. Apply spacing correction across the four data splits


```python
import os
import re
from pathlib import Path

import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "project.duckdb").exists())
SPLIT_DIR = ROOT / "data" / "processed"
SPLIT_NAMES = ["train_supervised", "train_pseudo_unlabeled", "validation", "test"]

EMOJI = "[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
WORD = r"[\w\u200c]"

# Same boundaries as the counting cell; the (?!\uFE0F) guard keeps keycap
# sequences such as digit + FE0F + 20E3 from being split
EMOJI_AFTER_WORD = rf"(?<={WORD})(?!\uFE0F){EMOJI}"
EMOJI_BEFORE_WORD = rf"{EMOJI}(?={WORD})"

# (rows_to_edit, spaces_to_insert) printed by the counting cell
EXPECTED = {
    "train_supervised": (25377, 26463),
    "train_pseudo_unlabeled": (5312, 5579),
    "validation": (6667, 6974),
    "test": (6385, 6705),
}

for name in SPLIT_NAMES:
    path = SPLIT_DIR / f"{name}.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    old = df["body"]
    mask = old.str.contains(EMOJI_AFTER_WORD, regex=True) | old.str.contains(EMOJI_BEFORE_WORD, regex=True)

    new = (
        old[mask]
        .str.replace(EMOJI_AFTER_WORD, r" \g<0>", regex=True)
        .str.replace(EMOJI_BEFORE_WORD, r"\g<0> ", regex=True)
    )

    n_rows = int(mask.sum())
    n_spaces = int((new.str.len() - old[mask].str.len()).sum())

    df.loc[mask, "body"] = new

    # No glued emoji may remain, and no other row may have changed
    left = df["body"].str.contains(EMOJI_AFTER_WORD, regex=True) | df["body"].str.contains(EMOJI_BEFORE_WORD, regex=True)
    assert not left.any(), f"{name}: glued emojis remain"
    assert (old[~mask] == df.loc[~mask, "body"]).all(), f"{name}: untouched rows changed"

    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

    status = "OK" if (n_rows, n_spaces) == EXPECTED[name] else "DIFFERENT FROM COUNTING CELL"
    print(f"{name}: rows edited {n_rows:,} | spaces inserted {n_spaces:,} | {status}")
```


**Output**

```text
train_supervised: rows edited 25,374 | spaces inserted 26,454 | DIFFERENT FROM COUNTING CELL
train_pseudo_unlabeled: rows edited 5,310 | spaces inserted 5,577 | DIFFERENT FROM COUNTING CELL
validation: rows edited 6,666 | spaces inserted 6,973 | DIFFERENT FROM COUNTING CELL
test: rows edited 6,385 | spaces inserted 6,705 | OK
```


### 3. Re-check the data after correction


```python
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Locate the repo root by walking up until project.duckdb is found
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "project.duckdb").exists())
SPLIT_DIR = ROOT / "data" / "processed"
SPLIT_NAMES = ["train_supervised", "train_pseudo_unlabeled", "validation", "test"]

# One emoji character (main emoji/symbol blocks plus variation selector)
EMOJI = "[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
# One word character (letter, digit, underscore, or ZWNJ)
WORD = r"[\w\u200c]"

# Zero-width boundaries: they match only the emoji itself
EMOJI_AFTER_WORD = rf"(?<={WORD}){EMOJI}"
EMOJI_BEFORE_WORD = rf"{EMOJI}(?={WORD})"
EMOJI_GLUED = rf"{EMOJI_AFTER_WORD}|{EMOJI_BEFORE_WORD}"

summary = []
for name in SPLIT_NAMES:
    df = pd.read_csv(SPLIT_DIR / f"{name}.csv", dtype=str)
    body = df["body"].fillna("")

    print(f"\n=== {name} ===")
    print("columns:", list(df.columns), "| rows:", len(df))

    has_after_word = body.str.contains(EMOJI_AFTER_WORD, regex=True)
    has_before_word = body.str.contains(EMOJI_BEFORE_WORD, regex=True)
    has_glued = has_after_word | has_before_word

    glued_body = body[has_glued]
    n_spaces = glued_body.str.count(EMOJI_AFTER_WORD).sum() + glued_body.str.count(EMOJI_BEFORE_WORD).sum()

    kinds = Counter()
    for text in glued_body:
        kinds.update(re.findall(EMOJI_GLUED, text))

    print(f"rows with word+emoji glued   : {has_after_word.sum():,}")
    print(f"rows with emoji+word glued   : {has_before_word.sum():,}")
    print(f"rows with either (to edit)   : {has_glued.sum():,} ({has_glued.mean() * 100:.2f}%)")
    print(f"total spaces to insert       : {n_spaces:,}")
    print("distinct emojis:", len(kinds))
    print("top 15 emojis  :", kinds.most_common(15))
    print("samples:")
    for text in glued_body.head(3):
        print("  ", text[:150])

    summary.append({
        "file": name,
        "rows": len(df),
        "rows_to_edit": int(has_glued.sum()),
        "spaces_to_insert": int(n_spaces),
    })

print()
print(pd.DataFrame(summary))
```


**Output**

```text

=== train_supervised ===
columns: ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'] | rows: 2860236
rows with word+emoji glued   : 4
rows with emoji+word glued   : 0
rows with either (to edit)   : 4 (0.00%)
total spaces to insert       : 9
distinct emojis: 1
top 15 emojis  : [('️', 9)]
samples:
   سری کتاب های ۴۰۰۰ essential english words واقعا خیلی عالین واسیه ار زبان آموزی چه self study چه با معلم خیلی مفیده میتونین فیلمو مشاهده بکنین نقد و بر
   سری کتاب های ۴۰۰۰ essential english words واقعا خیلی عالین واسیه ار زبان آموزی چه self study چه با معلم خیلی مفیده میتونین فیلمو مشاهده بکنین نقد و بر
   خیلی عالیی 💖💖👌👌👏👏 خیلی ناراحتم این مجموعه تموم شد ولی در عوض در این مدتی که این مجموعه رو میخوندم واقعا خیلی خیلی لذت می‌بردم و حتما بهتون پیشنهاد می‌

=== train_pseudo_unlabeled ===
columns: ['id', 'body'] | rows: 635824
rows with word+emoji glued   : 2
rows with emoji+word glued   : 0
rows with either (to edit)   : 2 (0.00%)
total spaces to insert       : 2
distinct emojis: 1
top 15 emojis  : [('️', 2)]
samples:
   سری کتاب های ۴۰۰۰ essential english words واقعا خیلی عالین واسیه ار زبان آموزی چه self study چه با معلم خیلی مفیده میتونین فیلمو مشاهده بکنین نقد و بر
   سری کتاب های ۴۰۰۰ essential english words واقعا خیلی عالین واسیه ار زبان آموزی چه self study چه با معلم خیلی مفیده میتونین فیلمو مشاهده بکنین نقد و بر

=== validation ===
columns: ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'] | rows: 922794
rows with word+emoji glued   : 1
rows with emoji+word glued   : 0
rows with either (to edit)   : 1 (0.00%)
total spaces to insert       : 1
distinct emojis: 1
top 15 emojis  : [('️', 1)]
samples:
   سری کتاب های ۴۰۰۰ essential english words واقعا خیلی عالین واسیه ار زبان آموزی چه self study چه با معلم خیلی مفیده میتونین فیلمو مشاهده بکنین نقد و بر

=== test ===
columns: ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'] | rows: 922805
rows with word+emoji glued   : 0
rows with emoji+word glued   : 0
rows with either (to edit)   : 0 (0.00%)
total spaces to insert       : 0
distinct emojis: 0
top 15 emojis  : []
samples:

                     file     rows  rows_to_edit  spaces_to_insert
0        train_supervised  2860236             4                 9
1  train_pseudo_unlabeled   635824             2                 2
2              validation   922794             1                 1
3                    test   922805             0                 0
```


---

### Scope note

- Setup-only imports, project-root discovery, and database connection/comparison cells were omitted.
- No notebook images were retained; image outputs, if present, are represented by placeholders.
- Code and text outputs are preserved from the saved notebook execution state; they have not been re-run.
