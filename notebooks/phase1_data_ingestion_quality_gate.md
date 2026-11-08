# Project 3 — Intelligent Analysis of Digikala Reviews

## Phase 1: Data Ingestion, Cleaning and Quality Gate

This document is the consolidated evidence report for the first phase of
the Digikala review analysis project.

It collects the statistical results, data-quality findings, cleaning-rule
logs, duplicate analysis, text-repetition analysis and text-length
measurements generated during the `00_data_ingest_quality_gate.ipynb`
notebook.

The report is designed to serve as a reusable source of evidence for
the subsequent descriptive-analysis, modeling and final-report stages.

---


| property | value |
| --- | --- |
| Project | Intelligent Analysis of Digikala Reviews |
| Notebook | notebooks/00_data_ingest_quality_gate.ipynb |
| Database | project.duckdb |
| Raw table | raw |
| Final cleaned table | clean_base |
| Total raw rows | 6,156,289 |
| Final cleaned rows | 6,151,947 |
| Report generation date | 2026-11-08 08:46:30 |
| DuckDB version | 2.0.0 |

---

## Table of Contents

1. [Data Funnel: Raw to Clean Base](#1-data-funnel-raw-to-clean_base)
2. [Cleaning Rules and Their Impact](#2-cleaning-rules-and-their-impact)
3. [Data Quality by Field](#3-data-quality-by-field)
4. [Before and After Data Profile](#4-before-and-after-data-profile)
5. [Final Target Distributions](#5-final-target-distributions)
6. [Duplicate Analysis](#6-duplicate-analysis)
7. [Repeated Text and Label Contradictions](#7-repeated-text-and-label-contradictions)
8. [Review Length Analysis](#8-review-length-analysis)
9. [Data Card: Findings and Technical Decisions](#9-data-card-findings-and-technical-decisions)

---



## 1. Data Funnel: Raw to Clean Base


### Goal

This section quantifies the reduction in dataset size from the original
raw table to the final `clean_base` table.

It distinguishes explicitly logged deletion rules from reductions that
occurred without corresponding row-level or rule-level logging.

### Method

The initial row count is obtained from `raw`. The final row count is
obtained from `clean_base`. Logged DROP counts are extracted from
`rule_summary`.

The difference between the total reduction and logged DROP counts is
reported separately. The notebook indicates that rows with empty bodies
after normalization were deleted without corresponding row-level logging.
The exact count is derived from the raw-to-clean difference and the
logged DROP totals at report-generation time.

All percentages below use the original raw row count as their denominator.



### 1.1 Dataset reduction summary


| metric | count | percentage_of_raw |
| --- | --- | --- |
| Raw rows | 6156289 | 100.0 |
| Final clean_base rows | 6151947 | 99.9294704975676 |
| Total removed rows | 4342 | 0.07052950243239069 |
| Logged DROP rows | 4342 | 0.07052950243239069 |
| Unlogged/derived reduction | 0 | 0.0 |

### 1.2 Step-by-step funnel


| stage | rows_before | removed | rows_after | removed_pct_raw | note |
| --- | --- | --- | --- | --- | --- |
| Raw dataset | 6156289 | 0 | 6156289 | 0.0 | Initial dataset |
| DUP_IDENTICAL | 6156289 | 2889 | 6153400 | 0.04692762149405267 | Same id, all compared columns identical: keep the lowest _row_idx |
| DUP_SAMEBODY_SAMEBAND | 6153400 | 119 | 6153281 | 0.00193298267836354 | Same id and body, same rating_band and status: keep one (prefer titled copy) |
| DUP_SAMEBODY_NO_TITLE | 6153281 | 5 | 6153276 | 8.121775959510672e-05 | Same id and body: drop the copy without title, keep the titled one |
| DUP_SAMEBODY_CONTRADICTION | 6153276 | 6 | 6153270 | 9.746131151412807e-05 | Same id and body: drop the copy whose rate contradicts recommendation_status |
| DUP_SAMEBODY_QUARANTINE | 6153270 | 114 | 6153156 | 0.0018517649187684334 | Same id and body, unresolved after earlier rules: drop all copies |
| DUP_DIFFBODY_QUARANTINE | 6153156 | 306 | 6152850 | 0.004970526887220532 | Same id but different body (edited review): drop all copies |
| MISS_body_norm | 6152850 | 903 | 6151947 | 0.014667927382876275 | body: missing-like text to NULL |

### 1.3 Main observations

- Raw dataset: 6,156,289 rows.
- Final `clean_base`: 6,151,947 rows.
- Total reduction: 4,342 rows (0.07%).
- Explicitly logged DROP rows: 4,342.
- Unlogged/derived reduction: 0.

The final funnel count is verified against the actual `clean_base` table.

### 1.4 Recommended visualization

- Waterfall chart: number of rows remaining after every deletion stage.
- Optional companion chart: percentage of the original dataset removed
  by each stage.

The unlogged reduction should be visually distinguished from the
explicitly logged cleaning rules.


## 2. Cleaning Rules and Their Impact


### Goal

This section answers which data-quality issues were most frequent and
which cleaning operations were applied.

### Method

The main source is `rule_summary`, which contains the rule ID,
description, action, affected-row count and reason.

The action categories are kept separate:

- `drop`: records removed from the dataset.
- `flag`: records retained but marked as invalid or requiring attention.
- `fix`: values standardized or corrected.
- `check`: validation rules and quality checks.

A row can be affected by multiple rules. Consequently, rule-level counts
must not be interpreted as a count of unique affected records.

The unique affected-row count is calculated separately from
`defect_manifest`.



### 2.1 Complete cleaning-rule register


| rule_id | description | action | n_rows | reason | pct_of_raw |
| --- | --- | --- | --- | --- | --- |
| DUP_IDENTICAL | Same id, all compared columns identical: keep the lowest _row_idx | drop | 2,889 | Exact copies carry no extra information | 0.05% |
| MISS_body_norm | body: missing-like text to NULL | drop | 903 | missing values stored as text; standardized | 0.01% |
| DUP_DIFFBODY_QUARANTINE | Same id but different body (edited review): drop all copies | drop | 306 | Edited versions can disagree on text and rating, so the label cannot be trusted | 0.00% |
| DUP_SAMEBODY_SAMEBAND | Same id and body, same rating_band and status: keep one (prefer titled copy) | drop | 119 | Copies differ only in minor columns, so the label is unchanged | 0.00% |
| DUP_SAMEBODY_QUARANTINE | Same id and body, unresolved after earlier rules: drop all copies | drop | 114 | Label cannot be trusted, so the id is excluded from training | 0.00% |
| DUP_SAMEBODY_CONTRADICTION | Same id and body: drop the copy whose rate contradicts recommendation_status | drop | 6 | Contradictory copy is likely a placeholder or wrong rate | 0.00% |
| DUP_SAMEBODY_NO_TITLE | Same id and body: drop the copy without title, keep the titled one | drop | 5 | Titled copy is the more complete record | 0.00% |
| RATE_invalid | rate missing, non-numeric or outside 0..5 | flag | 1 | row kept; only rating_band is invalid for it | 0.00% |
| MISS_title | title: missing-like text to NULL | fix | 2,866,656 | missing values stored as text; standardized | 46.56% |
| NORM_symbol_spacing | Spacing of punctuation/emoji runs, quotes, parentheses and brackets | fix | 1,052,395 | body text normalization; original text stays in raw | 17.09% |
| NORM_trim | Leading and trailing spaces/ZWNJs removed | fix | 913,957 | body text normalization; original text stays in raw | 14.85% |
| MISS_recommendation_status | recommendation_status: missing-like text to NULL | fix | 894,426 | missing values stored as text; standardized | 14.53% |
| NORM_space_ctrl | Unusual spaces and control chars converted to spaces | fix | 567,254 | body text normalization; original text stays in raw | 9.21% |
| MISS_seller_code | seller_code: missing-like text to NULL | fix | 302,181 | missing values stored as text; standardized | 4.91% |
| NORM_single_letters | Standalone single Arabic/Persian letters removed, except و | fix | 213,851 | body text normalization; original text stays in raw | 3.47% |
| NORM_spaces | Repeated spaces and ZWNJs collapsed | fix | 192,149 | body text normalization; original text stays in raw | 3.12% |
| NORM_repeats | Characters repeated 3+ times reduced to 2 (not digits, dots, parentheses) | fix | 143,300 | body text normalization; original text stays in raw | 2.33% |
| NORM_digits | Arabic-Indic and Latin digits converted to Persian digits | fix | 97,466 | body text normalization; original text stays in raw | 1.58% |
| NORM_dots | Two or more dots become exactly three dots | fix | 74,105 | body text normalization; original text stays in raw | 1.20% |
| NORM_arabic_letters | Arabic yeh/kaf variants converted to Persian | fix | 50,340 | body text normalization; original text stays in raw | 0.82% |
| NORM_diacritics | Diacritics and tatweel removed | fix | 45,574 | body text normalization; original text stays in raw | 0.74% |
| NORM_symbols_unify | Quotes, braces, percent, dashes, ? ; , converted to one standard form; thousands separators removed | fix | 14,673 | body text normalization; original text stays in raw | 0.24% |
| NORM_html | HTML entities decoded, tags removed | fix | 1,485 | body text normalization; original text stays in raw | 0.02% |
| NORM_invisible | Zero-width chars, bidi marks, BOM, soft hyphen removed (ZWNJ kept) | fix | 1,446 | body text normalization; original text stays in raw | 0.02% |
| MISS_body | body: missing-like text to NULL | fix | 637 | missing values stored as text; standardized | 0.01% |
| NORM_heh | Non-standard Arabic/Urdu heh variants converted to Persian heh | fix | 605 | body text normalization; original text stays in raw | 0.01% |
| NORM_empty | Empty text after cleaning converted to NULL | fix | 266 | body text normalization; original text stays in raw | 0.00% |
| NORM_url | URLs removed | fix | 8 | body text normalization; original text stays in raw | 0.00% |
| MISS_rate | rate: missing-like text to NULL | fix | 0 | missing values stored as text; standardized | 0.00% |
| NORM_escaped_zwnj | Literal ZWNJ Unicode escape sequences converted to actual ZWNJ | fix | 0 | body text normalization; original text stays in raw | 0.00% |
| MISS_is_buyer | is_buyer: missing-like text to NULL | fix | 0 | missing values stored as text; standardized | 0.00% |
| MISS_product_id | product_id: missing-like text to NULL | fix | 0 | missing values stored as text; standardized | 0.00% |
| MISS_id | id: missing-like text to NULL | fix | 0 | missing values stored as text; standardized | 0.00% |
| LABEL_rec_vocab | recommendation_status has only the 3 valid values | check | 0 | no mapping needed | 0.00% |

### 2.2 Summary by action type


| action | n_rules | total_rule_hits | largest_rule_count | pct_of_raw |
| --- | --- | --- | --- | --- |
| fix | 25 | 7,432,774 | 2,866,656 | 120.73% |
| drop | 7 | 4,342 | 2,889 | 0.07% |
| flag | 1 | 1 | 1 | 0.00% |
| check | 1 | 0 | 0 | 0.00% |

### 2.3 Unique affected rows in defect_manifest


| action | manifest_entries | unique_rows | n_rules | unique_rows_pct |
| --- | --- | --- | --- | --- |
| flag | 894427 | 894427 | 2 | 14.53% |
| drop | 4342 | 4342 | 7 | 0.07% |

Total manifest entries: 898,769

Unique rows represented in `defect_manifest`:
898,182 (14.59% of raw rows).

This is the number of unique row indices recorded in the manifest.
It does not include rows affected only by normalization rules that were
logged in `rule_summary` but not recorded individually in the manifest.

Among the surviving records, 2,210,449 rows
have a normalized body that differs from the original body in `raw`.

### 2.4 Normalization rules

Normalization is recorded as a sequence of transformations.
A row may be changed by multiple sequential rules, so the counts
must not be added to obtain a unique-row count.


| rule_id | description | n_rows | pct_of_raw |
| --- | --- | --- | --- |
| NORM_symbol_spacing | Spacing of punctuation/emoji runs, quotes, parentheses and brackets | 1,052,395 | 17.09% |
| NORM_trim | Leading and trailing spaces/ZWNJs removed | 913,957 | 14.85% |
| NORM_space_ctrl | Unusual spaces and control chars converted to spaces | 567,254 | 9.21% |
| NORM_single_letters | Standalone single Arabic/Persian letters removed, except و | 213,851 | 3.47% |
| NORM_spaces | Repeated spaces and ZWNJs collapsed | 192,149 | 3.12% |
| NORM_repeats | Characters repeated 3+ times reduced to 2 (not digits, dots, parentheses) | 143,300 | 2.33% |
| NORM_digits | Arabic-Indic and Latin digits converted to Persian digits | 97,466 | 1.58% |
| NORM_dots | Two or more dots become exactly three dots | 74,105 | 1.20% |
| NORM_arabic_letters | Arabic yeh/kaf variants converted to Persian | 50,340 | 0.82% |
| NORM_diacritics | Diacritics and tatweel removed | 45,574 | 0.74% |
| NORM_symbols_unify | Quotes, braces, percent, dashes, ? ; , converted to one standard form; thousands separators removed | 14,673 | 0.24% |
| NORM_html | HTML entities decoded, tags removed | 1,485 | 0.02% |
| NORM_invisible | Zero-width chars, bidi marks, BOM, soft hyphen removed (ZWNJ kept) | 1,446 | 0.02% |
| NORM_heh | Non-standard Arabic/Urdu heh variants converted to Persian heh | 605 | 0.01% |
| NORM_empty | Empty text after cleaning converted to NULL | 266 | 0.00% |
| NORM_url | URLs removed | 8 | 0.00% |
| NORM_escaped_zwnj | Literal ZWNJ Unicode escape sequences converted to actual ZWNJ | 0 | 0.00% |

### 2.5 Recommended visualization

- Horizontal bar chart of rule-level affected-row counts.
- Separate colors or panels for DROP, FLAG, FIX and CHECK.
- Logarithmic horizontal axis where necessary.

The chart should not combine unique affected rows with rule-level
counts as if they represented the same quantity.


## 3. Data Quality by Field


### Goal

Identify which dataset columns were associated with the recorded
quality problems.

### Method

The analysis groups `defect_manifest` records by `field` and `rule_id`.

The heatmap values represent the number of manifest entries.
A second table reports unique affected row indices per field.

This analysis only covers defects recorded in `defect_manifest`.
Rules recorded exclusively in `rule_summary` are not represented
in the heatmap.



### 3.1 Field-level summary


| field | manifest_entries | unique_affected_rows | pct_of_raw |
| --- | --- | --- | --- |
| recommendation_status | 894,426 | 894,426 | 14.53% |
| id | 3,439 | 3,439 | 0.06% |
| body | 903 | 903 | 0.01% |
| rate | 1 | 1 | 0.00% |

### 3.2 Rule-by-field heatmap data


| rule_id | body | id | rate | recommendation_status |
| --- | --- | --- | --- | --- |
| DUP_DIFFBODY_QUARANTINE | 0 | 306 | 0 | 0 |
| DUP_IDENTICAL | 0 | 2889 | 0 | 0 |
| DUP_SAMEBODY_CONTRADICTION | 0 | 6 | 0 | 0 |
| DUP_SAMEBODY_NO_TITLE | 0 | 5 | 0 | 0 |
| DUP_SAMEBODY_QUARANTINE | 0 | 114 | 0 | 0 |
| DUP_SAMEBODY_SAMEBAND | 0 | 119 | 0 | 0 |
| RATE_invalid | 0 | 0 | 1 | 0 |
| RECOMMENDATION_null | 0 | 0 | 0 | 894426 |
| body_null | 903 | 0 | 0 | 0 |

### 3.3 Recommended visualization

Heatmap configuration:

- Rows: cleaning rules (`rule_id`).
- Columns: data fields (`field`).
- Cell value: number of defect-manifest entries.
- Optional logarithmic color normalization if counts are highly skewed.

A small horizontal bar chart of unique affected rows per field can
accompany the heatmap.


## 4. Before and After Data Profile


### Goal

Compare the quality of the original dataset with the final cleaned table.

### Method

For shared columns, missing values are identified using NULL values,
blank strings and the configured placeholder vocabulary.

The comparison includes both absolute counts and percentages.

Rate statistics are calculated separately because `raw.rate` was
originally stored as text, while the cleaned rate is numeric and
restricted to the accepted range.



### 4.1 Missing-value comparison


| field | raw_missing | raw_missing_pct | clean_missing | clean_missing_pct | change_in_missing |
| --- | --- | --- | --- | --- | --- |
| id | 0 | 0.0 | 0 | 0.0 | 0 |
| title | 2866656 | 46.56467556997406 | 2864848 | 46.56815151365901 | -1808 |
| body | 637 | 0.010347142572416597 | 0 | 0.0 | -637 |
| rate | 0 | 0.0 | 1 | 1.625501650128e-05 | 1 |
| recommendation_status | 894426 | 14.528655168722587 | 893839 | 14.529367694487615 | -587 |
| product_id | 0 | 0.0 | 0 | 0.0 | 0 |
| is_buyer | 0 | 0.0 | 0 | 0.0 | 0 |
| seller_code | 302181 | 4.90849276244179 | 301903 | 4.9074382467859365 | -278 |

### 4.2 Unique ID comparison


| dataset | total_rows | unique_ids | duplicate_id_rows |
| --- | --- | --- | --- |
| raw | 6156289 | 6153060 | 3229 |
| clean_base | 6151947 | 6151947 | 0 |

### 4.3 Rate statistics


| dataset | total_rows | numeric_rows | null_or_non_numeric | min_rate | max_rate | mean_rate | median_rate | std_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 6156289 | 6156289 | 0 | 0.0 | 2500.0 | 3.648285939792628 | 4.0 | 1.8618541881512007 |
| clean_base | 6151947 | 6151946 | 1 | 0.0 | 5.0 | 3.647782725986217 | 4.0 | 1.5666765025642648 |

Raw rate values outside the accepted 0–5 range: 1.

The original rate inspection found a maximum value of 2500.0.
The cleaning pipeline restricts valid rate values to the 0–5 range
and stores invalid values as NULL.

### 4.4 Interpretation notes

- Missing-value percentages use the row count of each respective table.
- Rate statistics use successfully cast numeric values.
- Raw and cleaned statistics are not calculated from identical
  populations when invalid values are converted to NULL.
- A decrease in the missing count is not necessarily evidence of
  improved quality if records were deleted from the dataset.

### 4.5 Recommended visualization

A before/after horizontal bar chart of missing percentages by field.

The rate distribution before and after cleaning can also be shown
as a separate comparison if the original raw rate distribution
is retained.


## 5. Final Target Distributions


### Goal

Describe the distribution of the two targets that will be used
in the classification tasks.

### Recommendation target

`recommendation_status` contains three valid classes:

- `recommended`
- `not_recommended`
- `no_idea`

Missing values are reported separately.

### Rating-band target

The rating bands are derived from the cleaned numeric `rate`:

| Rating band | Threshold |
|---|---|
| low | 0 ≤ rate < 2.5 |
| medium | 2.5 ≤ rate < 4.0 |
| high | 4.0 ≤ rate ≤ 5.0 |

The NULL category represents records without a valid rating-band target.

All percentages are reported relative to the complete `clean_base`
dataset. The valid-label percentages are also shown separately.



### 5.1 Recommendation-status distribution


| class | n | pct_of_all | pct_of_valid |
| --- | --- | --- | --- |
| recommended | 4,160,041 | 67.62% | 79.12% |
| not_recommended | 504,448 | 8.20% | 9.59% |
| no_idea | 593,619 | 9.65% | 11.29% |
| NULL | 893,839 | 14.53% | 17.00% |

### 5.2 Rating-band distribution


| class | n | pct_of_all | pct_of_valid |
| --- | --- | --- | --- |
| low | 1,091,629 | 17.74% | 17.74% |
| medium | 1,058,354 | 17.20% | 17.20% |
| high | 4,001,963 | 65.05% | 65.05% |
| NULL | 1 | 0.00% | 0.00% |

### 5.3 Missing target counts

- Missing recommendation labels: 893,839
  (14.53%).
- Missing rating-band labels: 1
  (0.00%).

### 5.4 Modeling implications

Class imbalance should be evaluated using the class-specific distributions.

Macro F1, per-class Precision, Recall and F1, and Balanced Accuracy
should accompany Accuracy in the classification evaluation.

For the multi-task model, a missing target should mask only the loss
associated with that target. A valid target for the other head should
remain usable.

### 5.5 Recommended visualization

Two side-by-side bar charts:

1. Recommendation-status distribution.
2. Rating-band distribution.

Each bar should show both the number of records and the percentage.
NULL values should be visually distinguishable from valid classes.


### 5.6 Existing figures


### Raw rate distribution

![Raw rate distribution](figures/rate_distribution_curve.png)

Smoothed distribution of cleaned numeric rate values with rating-band boundaries.

### Rating-band distribution

![Rating-band distribution](figures/rating_band_distribution.png)

Distribution of low, medium and high rating bands.


## 6. Duplicate Analysis


### Goal

Document the duplicate-ID detection and removal process, including
the reason for each deletion rule.

### Duplicate categories

The duplicate groups are summarized into three conceptual categories:

1. `identical`: all compared fields are identical.
2. `same_body_other_cols_differ`: the body is identical but other
   columns differ.
3. `different_body`: the same ID has different review bodies.

The detailed cleaning process uses six rules.

### Method

Deleted records are identified using `defect_manifest`.
The original group sizes are reconstructed from `raw`.

The number of extra rows is calculated as the original group size
minus one. This is distinct from the number of removed rows,
especially for quarantined groups where every copy is deleted.



### 6.1 Duplicate category summary


| duplicate_type | affected_ids | original_rows | extra_rows | removed_rows |
| --- | --- | --- | --- | --- |
| identical | 2886 | 5775.0 | 2889.0 | 2889.0 |
| same_body_other_cols_differ | 187 | 374.0 | 187.0 | 244.0 |
| different_body | 153 | 306.0 | 153.0 | 306.0 |

### 6.2 Six detailed duplicate-removal rules


| rule_id | description | removed_rows | reason |
| --- | --- | --- | --- |
| DUP_IDENTICAL | Same id, all compared columns identical: keep the lowest _row_idx | 2889 | Exact copies carry no extra information |
| DUP_SAMEBODY_SAMEBAND | Same id and body, same rating_band and status: keep one (prefer titled copy) | 119 | Copies differ only in minor columns, so the label is unchanged |
| DUP_SAMEBODY_NO_TITLE | Same id and body: drop the copy without title, keep the titled one | 5 | Titled copy is the more complete record |
| DUP_SAMEBODY_CONTRADICTION | Same id and body: drop the copy whose rate contradicts recommendation_status | 6 | Contradictory copy is likely a placeholder or wrong rate |
| DUP_SAMEBODY_QUARANTINE | Same id and body, unresolved after earlier rules: drop all copies | 114 | Label cannot be trusted, so the id is excluded from training |
| DUP_DIFFBODY_QUARANTINE | Same id but different body (edited review): drop all copies | 306 | Edited versions can disagree on text and rating, so the label cannot be trusted |

### 6.3 Original duplicate group sizes


| id | original_group_size |
| --- | --- |
| 53766536 | 3 |
| 53973644 | 3 |
| 54490233 | 3 |
| 10021106 | 2 |
| 1002211 | 2 |
| 10051223 | 2 |
| 10115646 | 2 |
| 10131362 | 2 |
| 10179737 | 2 |
| 10221351 | 2 |
| 9509881 | 2 |
| 9562644 | 2 |
| 9595249 | 2 |
|  . . .  |  |
| 9651453 | 2 |
| 9666640 | 2 |
| 9696492 | 2 |
| 9696547 | 2 |
| 9719507 | 2 |
| 9730399 | 2 |
| 9785633 | 2 |
| 9795013 | 2 |
| 9798121 | 2 |
| 9820262 | 2 |
| 9836592 | 2 |
| 9897229 | 2 |
| 9923233 | 2 |
| 9968736 | 2 |
| 9977788 | 2 |
| 9993024 | 2 |

### 6.4 Duplicate examples


| duplicate_type | rule_id | id | _row_idx | title_sample | body_sample | rate | recommendation_status | product_id | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| different_body | DUP_DIFFBODY_QUARANTINE | 51133362 | 3464492 | nan | مو رو بسیار نرم می‌کنه مارک بسیار خوبیه | 5.0 | recommended | 8046768 | kept_row_idx=none (quarantined); edited copies (body differs): all copies dropped; rate=5.00, status=recommended |
| different_body | DUP_DIFFBODY_QUARANTINE | 33224903 | 982293 | nan | برای پوست های خشک عالیه  | 5.0 | recommended | 716918 | kept_row_idx=none (quarantined); edited copies (body differs): all copies dropped; rate=5.00, status=recommended |
| different_body | DUP_DIFFBODY_QUARANTINE | 54412507 | 2365609 | nan | خیلی راضی نبودم ..... | 4.0 | no_idea | 3830114 | kept_row_idx=none (quarantined); edited copies (body differs): all copies dropped; rate=4.00, status=no_idea |
| identical | DUP_IDENTICAL | 54252703 | 3294303 | تلوزیون ایران | تمام کنالهارو بخوبی دریافت میکنه | 5.0 | recommended | 10510452 | kept_row_idx=2025094; exact copy of the kept row |
| identical | DUP_IDENTICAL | 51675545 | 2776779 | پیلینگ صورت | خوب بود و برای جوش های سرسیاه مناسبه | 4.0 | recommended | 4332510 | kept_row_idx=2776729; exact copy of the kept row |
| identical | DUP_IDENTICAL | 34434326 | 1938017 | nan | اینکه توی دو روز قیمتش یه دفعه افزایش داد جالب نبود   اما خود کالا جالب و خوبه   بچه ها دوستش دارن   صدای قاری کودک هم داره | 4.0 | recommended | 5597247 | kept_row_idx=1937931; exact copy of the kept row |
| same_body_other_cols_differ | DUP_SAMEBODY_CONTRADICTION | 51960731 | 4006611 | nan | سنسوداین برند خوبی است اما قیمت های آن بالا  | 5.0 | not_recommended | 10157286 | kept_row_idx=3835620; status=not_recommended, rate=5.00 contradicts the status |
| same_body_other_cols_differ | DUP_SAMEBODY_CONTRADICTION | 12862914 | 3996800 | کرم مرطوب کننده BBk | کرم خوبیه  پیشنهاد میکنم | 0.0 | recommended | 2670905 | kept_row_idx=3561861; status=recommended, rate=0.00 contradicts the status |
| same_body_other_cols_differ | DUP_SAMEBODY_CONTRADICTION | 54505117 | 4004200 | عالی | ببینید بو یه چیز سلیقه‌ایه از نظر من این محصول بوی ملایم داره و نه تند  ضمنا من زیاد عرق نمیکنم و عرقم بو نداره پس این محصول برای من خوبه ولی به نظرم  | 0.0 | recommended | 4295397 | kept_row_idx=2020078; status=recommended, rate=0.00 contradicts the status |
| same_body_other_cols_differ | DUP_SAMEBODY_NO_TITLE | 40036247 | 4005898 | nan | عالی بود | 5.0 | nan | 6502581 | kept_row_idx=1347591; row has no title, another copy has one |
| same_body_other_cols_differ | DUP_SAMEBODY_NO_TITLE | 51787462 | 2288346 | nan | بسیار عالی و کاربردی | 4.0 | no_idea | 4131706 | kept_row_idx=359071; row has no title, another copy has one |
| same_body_other_cols_differ | DUP_SAMEBODY_NO_TITLE | 50235109 | 641544 | nan | تاثیر چندانی نداشت اما مضر هم نبود. | 3.0 | not_recommended | 5680302 | kept_row_idx=1704657; row has no title, another copy has one |
| same_body_other_cols_differ | DUP_SAMEBODY_QUARANTINE | 54349847 | 2374230 | توصیه میکنم | ممنون از اینکه ب روند تهیه آسان و دسترسی سریع ما کمک میکنید | 4.0 | recommended | 2411185 | kept_row_idx=none (quarantined); unresolved same-body copies: all copies dropped, never used for training |
| same_body_other_cols_differ | DUP_SAMEBODY_QUARANTINE | 54390812 | 2875187 | کیفیت مناسب | خرید این محصول را پیشنهاد میکنم | 4.0 | recommended | 9215716 | kept_row_idx=none (quarantined); unresolved same-body copies: all copies dropped, never used for training |
| same_body_other_cols_differ | DUP_SAMEBODY_QUARANTINE | 54318681 | 4004478 | nan | عالیه من قبلا هم گرفتم این بار دوم دوستمم چند بار گرفته ممنون از شرکت سازنده و دیجی کالا | 5.0 | nan | 209422 | kept_row_idx=none (quarantined); unresolved same-body copies: all copies dropped, never used for training |
| same_body_other_cols_differ | DUP_SAMEBODY_SAMEBAND | 54230762 | 4002875 | nan | برند خوبیه | 4.0 | recommended | 126815 | kept_row_idx=2867246; band=high, status=recommended, rate=4.00 |
| same_body_other_cols_differ | DUP_SAMEBODY_SAMEBAND | 52162265 | 3502542 | nan | خوش عطر و بو و به قیمت | 4.0 | recommended | 2184332 | kept_row_idx=2696859; band=high, status=recommended, rate=4.00 |
| same_body_other_cols_differ | DUP_SAMEBODY_SAMEBAND | 53907008 | 1617200 | خوب | داداشم چندین ساله اسفتاده میکنه بدشت راضیه پوستشم چرب ولی راضیه | 5.0 | recommended | 81867 | kept_row_idx=684792; band=high, status=recommended, rate=5.00 |

### 6.5 Duplicate-removal totals

- Unique duplicate IDs affected: 3,226.
- Rows removed under duplicate rules: 3,439.
- Duplicate-ID groups remaining in `clean_base`:
  0.
- Duplicate excess rows remaining in `clean_base`:
  0.

### 6.6 Recommended visualization

A compact bar chart comparing the three duplicate categories.

A second chart can show the number of removed rows under each
of the six detailed rules.

The example table can be included in the appendix of the final report.


## 7. Repeated Text and Label Contradictions


### Goal

Measure how frequently identical normalized review texts occur
and determine how often the same text is associated with conflicting
recommendation or rating signals.

### Method

Repetition statistics are calculated by grouping the final normalized
`body` column.

A text is considered repeated when it appears in more than one row.

The contradiction analysis is restricted to labeled records with
non-empty review bodies.

The four contradiction checks follow the original notebook's logic.
The categories are not mutually exclusive: some texts can satisfy
multiple contradiction conditions.



### 7.1 Overall repetition statistics


| metric | count | percentage |
| --- | --- | --- |
| Rows with non-empty body | 6,151,947 | 100.00% |
| Rows with unique text | 3,807,535 | 61.89% |
| Rows with repeated text | 2,344,412 | 38.11% |
| Distinct review bodies | 3,998,279 | — |
| Rows whose text appears more than 100 times | 1,365,608 | 22.20% |
| Distinct texts appearing more than 100 times | 1,878 | — |

### 7.2 Repetition levels


| repetition_bucket | distinct_texts | review_rows | pct_of_rows |
| --- | --- | --- | --- |
| 1 | 3,807,535 | 3,807,535 | 61.89% |
| 2 | 93,678 | 187,356 | 3.05% |
| 3-5 | 58,655 | 212,156 | 3.45% |
| 6-20 | 29,191 | 284,553 | 4.63% |
| 21-100 | 7,342 | 294,739 | 4.79% |
| >100 | 1,878 | 1,365,608 | 22.20% |

### 7.3 Label and rating contradictions


| conflict_type | affected_rows | distinct_texts | pct_of_labeled_rows |
| --- | --- | --- | --- |
| Recommendation contradiction | 981,533 | 6,618 | 18.67% |
| Low vs high rating band | 1,499,307 | 38,214 | 28.51% |
| Rate 0-1 vs rate 4-5 | 1,482,552 | 35,666 | 28.20% |
| Rate 0 vs rate 5 | 1,432,917 | 27,171 | 27.25% |

### 7.4 Important finding

1,365,608 rows (22.20%)
belong to only 1,878 distinct texts that appear
more than 100 times each.

This indicates a high concentration of repeated generic review texts.

The repeated-text analysis is relevant to evaluation design.
Product-grouped splitting is required, but it does not automatically
guarantee that identical review texts associated with different
products cannot appear in separate splits.

Exact-text overlap should therefore be audited after the split.

### 7.5 Existing figures


### Identical text label conflicts

![Identical text label conflicts](figures/identical_text_label_conflicts.png)

Four types of contradictory labels associated with identical review texts.

### Identical text repetition

![Identical text repetition](figures/identical_text_share.png)

Share of unique and repeated texts and repetition levels.


### 7.6 Existing example CSV files

The notebook has already exported example records for the contradiction
and repetition analyses.


- [conflicts/01_recommended_vs_not_recommended.csv](examples/conflicts/01_recommended_vs_not_recommended.csv)
- [conflicts/02_low_band_vs_high_band.csv](examples/conflicts/02_low_band_vs_high_band.csv)
- [conflicts/03_rate_0-1_vs_rate_4-5.csv](examples/conflicts/03_rate_0-1_vs_rate_4-5.csv)
- [conflicts/04_rate_0_vs_rate_5.csv](examples/conflicts/04_rate_0_vs_rate_5.csv)
- [repetition/appears_2x.csv](examples/repetition/appears_2x.csv)
- [repetition/appears_3_5x.csv](examples/repetition/appears_3_5x.csv)
- [repetition/appears_6_20x.csv](examples/repetition/appears_6_20x.csv)
- [repetition/appears_21_100x.csv](examples/repetition/appears_21_100x.csv)
- [repetition/appears_100plus.csv](examples/repetition/appears_100plus.csv)
- [repetition/top_100_most_frequent_texts.csv](examples/repetition/top_100_most_frequent_texts.csv)

### 7.7 Recommended visualization

The existing two figures should be retained:

- `identical_text_share.png`
- `identical_text_label_conflicts.png`

No additional visualization is essential for this section.


## 8. Review Length Analysis


### Goal

Describe the length distribution of cleaned Persian reviews and
compare review lengths across recommendation classes.

This analysis provides evidence for choosing a maximum sequence length
and truncation policy for Transformer training.

### Method

Word counts are calculated using whitespace-separated tokens.
Character counts are calculated using the length of the normalized body.

The class-level analysis uses the three recommendation classes
and reports NULL labels separately.



### 8.1 Overall review-length statistics


| n_reviews | mean_words | median_words | p90_words | p95_words | p99_words | max_words | mean_chars | median_chars | p90_chars | p95_chars | p99_chars | max_chars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6,151,947 | 10.05 | 6.00 | 22.00 | 32.00 | 62.00 | 908.00 | 47.84 | 29.00 | 105.00 | 151.00 | 294.00 | 4,053.00 |

### 8.2 Very short and very long reviews


| total_reviews | reviews_up_to_3_words | reviews_over_70_words | reviews_over_100_words | pct_up_to_3_words | pct_over_70_words | pct_over_100_words |
| --- | --- | --- | --- | --- | --- | --- |
| 6,151,947 | 1,680,942 | 42,312 | 15,430 | 27.32% | 0.69% | 0.25% |

### 8.3 Review length by recommendation class


| recommendation_class | n_reviews | mean_words | median_words | p90_words | p95_words | p99_words | max_words | mean_chars | median_chars | p95_chars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recommended | 4,160,041 | 9.20 | 6.00 | 20.00 | 28.00 | 55.00 | 908.00 | 43.70 | 27.00 | 135.00 |
| not_recommended | 504,448 | 15.11 | 10.00 | 33.00 | 45.00 | 79.00 | 630.00 | 72.56 | 48.00 | 215.00 |
| no_idea | 593,619 | 12.31 | 8.00 | 27.00 | 38.00 | 70.00 | 686.00 | 58.75 | 37.00 | 179.00 |
| NULL | 893,839 | 9.66 | 5.00 | 22.00 | 32.00 | 66.00 | 855.00 | 45.92 | 25.00 | 153.00 |

### 8.4 Initial truncation recommendation

The overall 99th percentile is 62 words.

A preliminary word-based cap of 70 words
would leave 42,312 reviews
(0.69%) above the cap.

This is a word-count-based estimate, not a Transformer tokenizer limit.

The final `max_length` must be selected after applying the actual
tokenizer of the chosen pretrained encoder to a representative
collection of reviews.

The tokenizer analysis should report:

- median token count;
- p90, p95 and p99 token counts;
- maximum token count;
- percentage of reviews exceeding candidate limits;
- expected truncation rate at each candidate maximum length.

### 8.5 Existing visualization


### Review length distribution

![Review length distribution](figures/review_length_distribution.png)

Distribution of review word counts and cumulative share.


### 8.6 Recommended new visualization

A box plot or violin plot comparing word counts across:

- recommended
- not_recommended
- no_idea

A separate chart can compare character lengths.

Because review lengths are often highly skewed, a logarithmic y-axis
or a restricted display range may improve readability. The full
statistics should remain available in the accompanying table.


## 9. Data Card: Findings and Technical Decisions


### Goal

Consolidate the major data-quality findings and explicitly connect
them to decisions in the next project stages.

The findings describe observed evidence. The technical decisions
are proposed implications for subsequent implementation and should
not be interpreted as completed modeling experiments.



| issue | finding | technical_decision | reason | stage |
| --- | --- | --- | --- | --- |
| Product-level leakage risk | The dataset contains millions of product reviews. Product-level grouping must be checked before splitting. | Use a product-grouped train/validation/test split. | Prevent reviews belonging to the same product from appearing in different splits. | Data splitting |
| Repeated review texts | 2,344,412 rows (38.11%) contain repeated text. | Audit exact-text overlap across future splits. | Repeated texts may create overly optimistic evaluation if copies cross split boundaries. | Data splitting and evaluation |
| Highly concentrated repeated texts | 1,365,608 rows are associated with 1,878 texts appearing more than 100 times. | Report repeated-text concentration and inspect its effect on evaluation. | A small vocabulary of generic reviews represents a substantial fraction of the dataset. | Evaluation |
| Recommendation class imbalance | The target distribution must be evaluated using the class-specific counts in Section 5. | Consider class weighting and report Macro F1 and per-class Precision, Recall and F1. | Accuracy alone may hide weak performance on minority classes. | Modeling and evaluation |
| Rating-band target | 1 rows have no valid rating_band. | Mask the rating-band loss for rows without a valid target. | A missing rating target should not remove a valid recommendation target. | Multi-task modeling |
| Persian text normalization | 17 normalization rules are recorded in rule_summary. | Apply the same documented normalization pipeline during training and inference. | Training and prediction text must follow consistent preprocessing. | Preprocessing |
| Review length | Word-count p99 is 62 words. | Use 70 words as an initial analysis-based truncation candidate, then verify with the actual tokenizer. | Word counts are not equal to subword tokenizer tokens. | Transformer configuration |
| Contradictory labels | Identical texts can occur with conflicting recommendation or rating labels. | Include contradiction analysis in error analysis and evaluate model behavior on repeated texts. | The same text does not always imply the same label. | Evaluation and error analysis |
| Product concentration | Product-frequency analysis has not been completed in this notebook. | Compute product-level concentration before creating the grouped split. | The distribution of reviews per product affects the split strategy. | Next analysis stage |

---

## 10. Overall Phase 1 Summary

This section provides a concise overview of the completed data
ingestion and cleaning stage.

### Dataset


| metric | value |
| --- | --- |
| Raw rows | 6,156,289 |
| Final clean_base rows | 6,151,947 |
| Total rows removed | 4,342 |
| Unique IDs in raw | 6,153,060 |
| Unique IDs in clean_base | 6,151,947 |
| Unique manifest-affected rows | 898,182 |
| Duplicate-removal rows | 3,439 |
| Duplicate-ID groups remaining in clean_base | 0 |
| Repeated-text rows | 2,344,412 |
| Repeated-text percentage | 38.11% |
| Recommendation missing rows | 893,839 |
| Rating-band missing rows | 1 |
| Review word-count p99 | 62 |
| Normalization rules | 17 |

---

## 11. Methodology and Reproducibility Notes

### Data sources

The report is generated from the following DuckDB tables:

| Table | Purpose |
|---|---|
| `raw` | Original dataset loaded from CSV |
| `clean_base` | Final cleaned dataset |
| `defect_manifest` | Row-level record of logged defects |
| `rule_summary` | Rule-level cleaning statistics |

### Data processing

The original dataset was loaded with text-oriented column types.
The cleaning pipeline standardized missing values, validated
recommendation labels, validated numeric rates, derived rating bands,
normalized Persian review text and removed duplicate IDs.

The final review text is stored in the `body` column of `clean_base`.

### Missing values

The following placeholders were treated as missing:

- `null`
- `nan`
- `none`
- `-`
- `n/a`
- `na`
- `undefined`

Blank strings and SQL NULL values were also treated as missing.

### Rating bands

| Band | Numeric range | Ordinal encoding |
|---|---|---|
| low | 0 ≤ rate < 2.5 | 0 |
| medium | 2.5 ≤ rate < 4.0 | 1 |
| high | 4.0 ≤ rate ≤ 5.0 | 2 |

### Important limitations

1. The unlogged/derived reduction is not recorded at row level in
   `defect_manifest`. The notebook indicates that the corresponding
   body-empty deletion logging operations were commented out. Its count
   is inferred from the raw-to-clean difference minus logged DROP totals.

2. The unique affected-row count from `defect_manifest` excludes
   normalization operations that were logged only in `rule_summary`.

3. Rule-level counts can overlap. They must not be summed to estimate
   unique affected records.

4. The duplicate example records are reconstructed from `raw`
   using the row indices stored in the manifest.

5. The text-repetition and contradiction statistics are calculated
   on the final cleaned and normalized `body` column.

6. The product-frequency concentration analysis has not been
   completed in this notebook and should be performed before
   creating the product-grouped split.

7. Word count is not equivalent to subword tokenizer length.
   The final Transformer truncation limit requires a separate
   tokenizer-based analysis.

8. The report does not include train/validation/test splitting,
   TF-IDF modeling or Transformer training results. These belong
   to subsequent project stages.

### Reproducibility

All major counts and statistics are queried directly from DuckDB
when this report is generated.

The report generation process does not modify `raw` or `clean_base`.

The report can be regenerated after updating the cleaning pipeline
by rerunning the final report-generation cell.

---

## 12. Report Generation Metadata


| property | value |
| --- | --- |
| Generated at | 2026-11-08 08:46:30 |
| Raw row count | 6,156,289 |
| Clean row count | 6,151,947 |
| DuckDB version | 2.0.0 |
| Output file | reports/phase1_data_ingestion_quality_gate.md |

---

_End of Phase 1 Data Ingestion and Quality Gate Report._
