# Project Handoff: Intelligent Analysis of Digikala Reviews (Project 3)

## 1. Project summary
Bootcamp project on Persian Digikala product reviews. The only model input is the review `body`.
- **Task A:** classify `recommendation_status` into `recommended` / `not_recommended` / `no_idea`.
- **Task B:** classify an ordered `rating_band` derived from `rate`: `low` (0 ≤ rate < 2.5), `medium` (2.5 ≤ rate < 4.0), `high` (4.0 ≤ rate ≤ 5.0), encoded 0/1/2.
- **Models:** a TF-IDF + Logistic Regression baseline and fine-tuned pretrained Transformer encoders with two classification heads.
- **Semi-supervised experiment:** pseudo-labeling for the recommendation head only, fixed confidence threshold 0.90.
- **Dataset:** public Digikala reviews (HuggingFace `RadeAI/Digikala_comments_products`), about 7M raw records.

### Fixed rules from the project definition
- Allowed model input: `body` only. `title`, `rate`, `recommendation_status`, `product_id`, seller fields, timestamps, likes, dislikes, advantages and disadvantages are never features. `product_id` is used only for grouped splitting.
- One shared product-grouped split for both tasks: 70% train / 15% validation / 15% test. A product must not appear in more than one split.
- Rows with a missing target stay in their product's split. Only the head with a valid target uses them (the loss of the other head is masked).
- Data-dependent preprocessing (TF-IDF vocabulary, class weights, normalization statistics) is fitted on train only.
- Masking protocol (after the split): within each recommendation class of the train split, sort by `stable_hash(id)` and move `floor(0.20 × class_count)` rows to the artificial unlabeled pool. Validation and test are never masked.
  ```python
  SPLIT_SEED = 2026
  MASK_SEED = "digikala-semi-supervised-v1"
  MASK_RATE = 0.20
  def stable_hash(row_id): int(sha256(f"{MASK_SEED}|{row_id}".encode()).hexdigest()[:16], 16)
  ```
- Required data artifacts: `train_supervised.csv`, `train_pseudo_unlabeled.csv` (only `id` and `body`), `natural_unlabeled.csv`, `validation.csv`, `test.csv`, `split_manifest.json`.
- Required notebooks: `01_descriptive_analysis`, `02_baselines`, `03_transfer_learning_finetuning`, `04_semi_supervised`. Also required: offline prediction script, README, final report with at least 20 analyzed errors, and `release_report.json`. (The repo's current notebook numbering differs from these required names; the required deliverables still have to be created or renamed later.)
- FastAPI and Docker are optional Phase 2.

## 2. Repository and technical setup
- Config file: `src/digikala_review_transformer/config.py` (with `__init__.py` beside it):
  - `ROOT` is the project root (three levels up from the config file).
  - Folders: `DATA_RAW = ROOT/data/raw`, `DATA_INTERIM = ROOT/data/interim`, `DATA_PROCESSED = ROOT/data/processed`, `ARTIFACTS = ROOT/artifacts`, `REPORTS = ROOT/reports`.
  - Constants: `SPLIT_SEED = 2026`, `MASK_SEED = "digikala-semi-supervised-v1"`, `MASK_RATE = 0.20`, `PSEUDO_LABEL_THRESHOLD = 0.90`.
  - Notebooks also import `DB_PATH` and `CLEAN_TABLE` from the config (DuckDB file and table name).
- The contents of the data and artifacts folders are not tracked by Git.
- Cleaned data lives in the DuckDB database (`DB_PATH`), table `clean_base` (`CLEAN_TABLE`), about 6,151,947 rows. It has a row-counter column `_row_idx` (never a model feature).
- Helper module: `src/digikala_review_transformer/split.py` (split logic).
- Language conventions: code comments in English. The user communicates in Persian.

## Folders and files tree structure

```
digikala-review-transformer/
├── .cache
├── .git
├── .venv
├── artifacts
│   ├── Step1_label_correction
│   │   ├── rating_band
│   │   │   └── label_correction_bands_rating_band.html
│   │   ├── rating_band_more_precise
│   │   │   └── rating_band_more_precise.html
│   │   └── recommendation_status
│   │       └── label_correction_bands.html
│   └── .gitkeep
├── data
│   ├── external
│   │   ├── parsivar_spell_extracted
│   │   │   ├── mybigram_lm.pckl
│   │   │   └── onegram.pckl
│   │   └── parsivar_spell.zip
│   ├── interim
│   │   ├── body_groups_rating_band.parquet
│   │   ├── body_groups_recommendation_status.parquet
│   │   └── clean_base_backup.duckdb
│   ├── processed
│   │   ├── label_correction_params.json
│   │   └── product_split.csv
│   ├── raw
│   │   ├── digikala-comments.csv
│   │   └── digikala-products.csv
│   └── .gitkeep
├── figs
│   ├── banner.png
│   ├── Data Quality Decision.png
│   ├── normalizing example 1.png
│   ├── normalizing example 2.png
│   ├── Screenshot 2026-10-31 052644.png
│   ├── Screenshot 2026-10-31 052934.png
│   ├── Screenshot From 2026-11-07 03-22-38.png
│   ├── Screenshot From 2026-11-07 03-25-04.png
│   ├── semi-supervised learning.png
│   └── three NN heads.png
├── models
│   └── pretrained
│       ├── mdeberta-v3-base
│       │   ├── .cache
│       │   ├── .gitattributes
│       │   ├── config.json
│       │   ├── generator_config.json
│       │   ├── pytorch_model.bin
│       │   ├── pytorch_model.generator.bin
│       │   ├── README.md
│       │   ├── spm.model
│       │   └── tokenizer_config.json
│       ├── roberta-fa-zwnj-base
│       │   ├── .cache
│       │   ├── .gitattributes
│       │   ├── config.json
│       │   ├── merges.txt
│       │   ├── pytorch_model.bin
│       │   ├── README.md
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer.json
│       │   ├── tokenizer_config.json
│       │   └── vocab.json
│       └── TookaBERT-Base
│           ├── .cache
│           ├── .gitattributes
│           ├── config.json
│           ├── generation_config.json
│           ├── model.safetensors
│           ├── pytorch_model.bin
│           ├── README.md
│           ├── special_tokens_map.json
│           ├── tokenizer.json
│           └── tokenizer_config.json
├── notebooks
│   ├── .gitkeep
│   ├── 00_data_ingest_quality_gate.ipynb
│   ├── 00_data_ingest_quality_gate.md
│   ├── 01_more_data_ingest_analysis.ipynb
│   ├── 02_fetch_models.ipynb
│   ├── 03_data_split_&_products_analysis.ipynb
│   └── 04_label_correction_step1.ipynb
├── reports
│   ├── examples
│   │   ├── conflicts
│   │   │   ├── 01_recommended_vs_not_recommended.csv
│   │   │   ├── 02_low_band_vs_high_band.csv
│   │   │   ├── 03_rate_0-1_vs_rate_4-5.csv
│   │   │   ├── 04_rate_0_vs_rate_5.csv
│   │   │   └── Columns definition.csv
│   │   └── repetition
│   │       ├── appears_100plus.csv
│   │       ├── appears_21_100x.csv
│   │       ├── appears_2x.csv
│   │       ├── appears_3_5x.csv
│   │       ├── appears_6_20x.csv
│   │       └── top_100_most_frequent_texts.csv
│   ├── figures
│   │   ├── .gitkeep
│   │   ├── group_split_&_lorenz_poster.png
│   │   ├── identical_text_label_conflicts.png
│   │   ├── identical_text_share.png
│   │   ├── latin_conversion_outcome.png
│   │   ├── lorenz_chart_split_curves.png
│   │   ├── Parsivar_rejection.png
│   │   ├── product_review_lorenz_by_split.png
│   │   ├── rate_distribution_curve.png
│   │   ├── rating_band_distribution.png
│   │   └── review_length_distribution.png
│   ├── label_correction_rating_band_bands
│   │   ├── final
│   │   │   └── label_correction_rating_band_groups.csv
│   │   ├── band_0.35.csv
│   │   ├── band_0.40.csv
│   │   ├── band_0.45.csv
│   │   ├── band_0.50.csv
│   │   ├── band_0.55.csv
│   │   ├── band_0.60.csv
│   │   ├── band_0.65.csv
│   │   ├── band_0.70.csv
│   │   ├── band_0.75.csv
│   │   ├── band_0.80.csv
│   │   ├── band_0.85.csv
│   │   ├── band_0.90.csv
│   │   ├── band_summary.csv
│   │   ├── label_correction_rating_band_knee.png
│   │   ├── label_correction_rating_band_sweep.csv
│   │   └── label_correction_rating_band_sweep.png
│   ├── label_correction_rating_band_bands_more_precise
│   │   ├── band_0.80.csv
│   │   ├── band_0.81.csv
│   │   ├── band_0.82.csv
│   │   ├── band_0.83.csv
│   │   ├── band_0.84.csv
│   │   ├── band_0.85.csv
│   │   ├── band_0.86.csv
│   │   ├── band_summary.csv
│   │   ├── label_correction_rating_band_sweep.csv
│   │   └── label_correction_rating_band_sweep.png
│   ├── label_correction_recommendation_status_bands
│   │   ├── final
│   │   │   └── label_correction_recommendation_status_groups.csv
│   │   ├── band_0.91.csv
│   │   ├── band_0.92.csv
│   │   ├── band_0.93.csv
│   │   ├── band_0.94.csv
│   │   ├── band_summary.csv
│   │   ├── label_correction_recommendation_status_knee.png
│   │   ├── label_correction_recommendation_status_sweep.csv
│   │   ├── label_correction_recommendation_status_sweep.html
│   │   └── label_correction_recommendation_status_sweep.png
│   ├── latin_comments_for_translation
│   │   ├── converted
│   │   │   ├── part_01.csv
│   │   │   ├── part_02.csv
│   │   │   └── part_03.csv
│   │   ├── parts
│   │   │   ├── part_01.csv
│   │   │   ├── part_02.csv
│   │   │   └── part_03.csv
│   │   ├── convert_persian_comments_all.csv
│   │   └── latin_comments_all.csv
│   ├── product_concentration_differences_vs_global.csv
│   ├── split_class_comparison.csv
│   ├── split_final_summary.json
│   └── top10_most_commented_products.csv
├── src
│   └── digikala_review_transformer
│       ├── __init__.py
│       ├── config.py
│       ├── folders_tree_list_helper.py
│       └── split.py
├── .gitignore
├── .python-version
├── Handoff.md
├── Personal Note.txt
├── Project-Definition3-digikala-review-transformer-fastapi-v2.ipynb
├── Project-Definition3-digikala-review-transformer-fastapi-v2.md
├── project.duckdb
├── pyproject.toml
├── README.md
└── uv.lock
```

## 3. What has been done so far
1. **Data ingestion and cleaning finished.** About 6.15M comments remain after cleaning. No duplicate `id` values exist. Persian normalization was applied as part of cleaning, so all `body` texts are already normalized.
2. **Observed data findings:**
   - Many reviews have a body that contradicts the label (for example the body "عالی" labeled `not_recommended`).
   - Ambiguous bodies ("معمولی", "بد نیست", "بد نبود", "خوبه بد نیست") carry the `no_idea` signal and must not be forced to `recommended`.
   - About 893,839 rows have no `recommendation_status` (about 14.5%); 1 row has no `rating_band`.
   - `rate = 0` covers 8.60% of the data (528,914 rows), and 80.5% of those rows are `recommended`. This is consistent with `rate = 0` often meaning "not rated", but the meaning of `rate` was not changed (see section 5).
   - Classmates report 60–70% validation accuracy. The task and class balance behind that figure are unknown.
3. **Strategy decisions** (from the earlier decision report):
   - **Duplicate cap:** at most 50 copies per body in `train_supervised` only, keeping the smallest `stable_hash(id)`. Not done yet.
   - **Soft sample weighting (variant C):** out-of-fold TF-IDF + LR probabilities from 5-fold `StratifiedGroupKFold` (group = `product_id`) give a per-row weight `w = max(0.1, 1 − C·(1 − Q))`. Recommendation head only; pseudo-labeled rows get w = 1. Stop rule: if the mean weight of `no_idea` is more than about 0.05 below the other classes, drop the idea. Fallback estimator: frozen-encoder embeddings + LR.
   - **Noise-robust training:** label smoothing 0.1, class weights from train, 1–2 epochs, early stopping on validation Macro F1, smaller learning rate for the encoder than for the heads. Label smoothing reduces how many rows pass the 0.90 pseudo-label threshold, so the accepted count must be reported.
   - **Order and evaluation:** correct labels first, then mask (documented deviation from the shared protocol). Report raw and corrected metrics on validation and test; the raw test set is the benchmark comparable with classmates.
   - **Variants:** A = raw labels (`recommendation_status`, `rating_band`), B = step-1 corrected (`step1_rec_label`, `step1_rate_band_label`), C = B plus sample weights. Compare them on the baseline and one Transformer, then use the best variant for the other two encoders. Optional ensemble at the end by averaging softmax outputs.
   - **Rejected for time:** Confident Learning with removal, co-teaching and DivideMix. An LLM is used only for auditing 20–30 samples.
4. **Models chosen, downloaded and ready** (in `models/pretrained/`):

   | Model | Family | Parameters | Role |
   | --- | --- | --- | --- |
   | `PartAI/TookaBERT-Base` | TookaBERT (Persian) | ~123M | Primary Persian model |
   | `HooshvareLab/roberta-fa-zwnj-base` | ParsBERT v3.0 (RoBERTa) | ~125M (verify) | Persian, ZWNJ-aware |
   | `microsoft/mdeberta-v3-base` | DeBERTa-v3 (multilingual) | ~278M | Multilingual |

5. **`max_length` determined experimentally** from token-length percentiles:

   | Model | Tokenizer | max_length |
   | --- | --- | --- |
   | TookaBERT-Base | BPE (Persian) | 128 |
   | roberta-fa-zwnj-base | Byte-level BPE (Persian) | 128 |
   | mdeberta-v3-base | SentencePiece (multilingual) | 160 |

   Dynamic padding is planned.
6. **Seeds and mask constants** are fixed in `config.py`.
7. **Product-grouped split finished and verified** (notebook `03_data_split_&_products_analysis.ipynb`).
   - Method: deterministic product-grouped split, 70/15/15 by row share, `SPLIT_SEED = 2026`, one shared split for both tasks. Rows with a missing recommendation label or invalid `rate` stay in their product's split.
   - Output: `data/processed/product_split.csv` (`product_id, split`; values `train` / `validation` / `test`). The `split` column was also added to `clean_base`.
   - Final sizes: train `[rows / products]`, validation `[rows / products]`, test `[rows / products]`.
   - Checks passed: no product in more than one split, row shares close to 70/15/15, class distributions similar across splits, reproducible across runs.
   - Reports: `reports/split_final_summary.json`, `reports/split_class_comparison.csv`, `reports/product_concentration_differences_vs_global.csv`, `reports/top10_most_commented_products.csv`, figures in `reports/figures/`.
   - The split is final and must not be changed.
   - ## **Split results and validation**

| **Split** | **Products** | **Share of products** | **Reviews** | **Share of reviews** | **n_rows (all)** |
| --- | --- | --- | --- | --- | --- |
| Train | 232,397 | 70.0869% | 4,306,348 | 69.9998% | 4306348 |
| Validation | 49,911 | 15.0523% | 922,794 | 15.0000% | 922794 |
| Test | 49,276 | 14.8608% | 922,805 | 15.0002% | 922805 |
| **Total** | **331,584** | **100%** | **6,151,947** | **100%** | 6151947 |
8. **Step 1 label correction (majority rule) finished and verified** (notebook `04_label_correction_step1.ipynb`).
   - **Goal:** reduce noise from repeated bodies that carry contradictory labels.
   - **Method:** group by normalized `body` over labeled **train** rows only; groups with `MIN_GROUP = 20` labeled rows; if the majority share ≥ threshold, relabel minority rows to the majority label. The train-derived table is applied to train, validation and test. Missing labels stay missing (checked first in the `UPDATE`, so the natural unlabeled pool is not polluted). Original columns are never modified.
   - **New columns in `clean_base`:** `split`, `step1_rec_label`, `step1_rate_band_label`. No extra tables or views were created.
   - **Thresholds (chosen by reading the newly added groups per band; Kneedle was unstable and used only as a hint):**

     | Target | Selected threshold | Reason |
     | --- | --- | --- |
     | `recommendation_status` | 0.932 | Just above the hedged high-frequency bodies «خوب بود» (0.9316), «خوبه» (0.9312), «خوب» (0.9299), whose ~6.7k `no_idea` rows are natural doubt |
     | `rating_band` | 0.84 | Chosen from sample reading; band needs a lower threshold (outside the originally planned 0.90–0.99 range, documented) |

   - **Group statistics (recommendation, train):** 3,681,586 labeled rows; 2,479,733 unique bodies; 5,918 groups with ≥ 20 labeled rows (24.59% of labeled train rows); 3,796 groups with a conflict. Known bodies: «عالی» (share 0.990) is corrected; «معمولی» (0.590, majority `no_idea`) and «بد نیست» (0.496) are never corrected.
   - **Rows corrected:**

     | Target | Majority-table groups | Train | Validation | Test | Total | % of labeled rows |
     | --- | --- | --- | --- | --- | --- | --- |
     | `recommendation_status` | 4,218 (2,122 with share = 1) | 10,793 | 2,471 | 2,537 | 15,801 | 0.301% |
     | `rating_band` | 2,650 (165 with share = 1) | 39,262 | 9,034 | 8,946 | 57,242 | 0.930% |

   - **Main transitions:** recommendation: about 98% of changes go toward `recommended` (`no_idea→recommended` 13,553; `not_recommended→recommended` 1,945). Band: about 96% go toward `high` (`low→high` 31,560; `medium→high` 23,330; `medium→low` 1,699).
   - **Verification passed:** row count unchanged (6,151,947), NULL counts equal between original and new columns, original columns unchanged, no changed row outside the majority table, re-run gives an identical column, group CSV totals match.
   - **Backup:** `data/interim/clean_base_backup.duckdb` (taken before the first write).
   - **Outputs:** `data/interim/body_groups_<target>.parquet`; `data/processed/label_correction_params.json` (final parameters per target); `reports/label_correction_<target>_bands/final/label_correction_<target>_groups.csv` (one row per corrected group); sweep CSV/PNG and knee PNG and per-band inspection CSVs under `reports/label_correction_*_bands*/`; interactive HTML tables under `artifacts/Step1_label_correction/`.
   - **Known limitations:** only bodies with ≥ 20 labeled train rows are corrected, so noise remains (band: 0.93% of rows changed while `rate = 0` alone is 8.6%); the train-derived table uses all labeled train rows, including rows that will later be masked; thresholds come from manual sample reading; the 1,699 `medium→low` rows run against the main direction and a spot-check of their texts is advisable; raw-label metrics must always be reported next to corrected ones.

## 4. Current status
- Done: data cleaning, product-grouped split, step-1 label correction (both targets).
- Not done yet: masking on `step1_rec_label`, duplicate cap, the six standard CSV artifacts and `split_manifest.json`, baselines, sample weighting (variant C), Transformer fine-tuning, semi-supervised experiment, error analysis, offline prediction script, README and final report, `release_report.json`.
- Immediate next step: masking. Within each class of `step1_rec_label` in the train split, move `floor(0.20 × class_count)` labeled rows (sorted by `stable_hash(id)`) to the artificial unlabeled pool; then write the standard files and `split_manifest.json`. The original `recommendation_status` stays available so pseudo-label accuracy can be reported against both label versions.

## 5. Open items
- Confirm with the instructor the correct-before-mask order and the duplicate cap.
- Decide whether the pseudo-unlabeled pool is also capped.
- Decide the size of the deterministic modeling subset if hardware is limited.
- `rate = 0`: evidence collected (8.60% of rows, 80.5% of them `recommended`, dominant `low→high` correction), but the definition of `rate` and of the bands was not changed. Any formal change needs documented evidence and approval. Handling of residual band noise is left to sample weighting and cautious reading of band Macro F1.
- Record the mask-after-correction deviation in `split_manifest.json` and the report.

## 6. Working-style notes for the assistant
- The user wants concise, direct answers without filler, and step-by-step plans when a roadmap is requested (for each step: purpose, what to do, inputs, outputs, expected result, approval criterion, fallback), with no extra steps. Steps whose results drive decisions come first. No code in roadmaps unless asked.
- Code style requested by the user: parametric (tunable ranges, steps and thresholds as parameters at the top of the notebook, no hard-coding). The first notebook cell holds imports and the DuckDB connection (`DB_PATH`, `CLEAN_TABLE` from config, `cur` cursor, `scalar` helper).
- Disk space is tight (DuckDB temp tables previously reached ~15 GB): prefer adding columns to `clean_base` over new tables or views, keep small intermediates as parquet in `data/interim/`, and never modify existing columns.
- Do not run tests or experiments in a sandbox or use tools unnecessarily; the user has limited tokens. Write code or guidance as text and let the user run it.
- The user wants to learn the reasoning behind each setting, not only get instructions.