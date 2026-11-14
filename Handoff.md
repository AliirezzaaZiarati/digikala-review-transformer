# Project 3: Digikala Review Classification — Technical Handoff

**Purpose of this file:** a new engineer or AI assistant with zero prior knowledge should be able to continue the project from this file alone.
**Last updated:** end of the baseline phase (TF-IDF + Logistic Regression). The phase is closed. The next phase has not started.
**Facts here are taken from real outputs.** Anything unverified is marked **(unverified)** or **(not inspected)**. Do not guess column names, helper signatures or file structures: inspect them first with a read-only cell.

---

## 0. Reader instructions

- **Language:** the user writes in **Persian**. Reply in Persian: concise, direct, no filler. Code, code comments, identifiers and file names are in **English**.
- **No sandbox:** you write code as text and the user runs it in Jupyter, then pastes outputs back.
- **Before each reply:** reason fully and verify claims against real outputs.
- **No images:** you cannot see generated charts. Ask the user to describe them (`display(Image(filename=..., width=1200))`).
- **No code in roadmaps or plans** unless the user asks.
- **Time is very tight:** do nothing that is not in the agreed steps.
- **Before each chart:** give a short design plan (5–8 lines).
- **Read-only inspection:** if you need an unknown helper's signature or output structure, request a small read-only inspection cell. Never guess.

---

## 1. Project overview

A class project (Persian bootcamp): classify Digikala (Iranian e-commerce) product reviews from the review text.

| Item | Definition |
|---|---|
| **Model input** | `body` only (the review text). Nothing else may enter a model (no `rate`, no `product_id`). |
| **Task A (rec)** | `recommendation_status`: `recommended` / `not_recommended` / `no_idea` |
| **Task B (band)** | ordered `rating_band`: `low` (<2.5), `medium` (2.5 to <4.0), `high` (4.0 to 5.0) |
| **Dataset** | HuggingFace `RadeAI/Digikala_comments_products` (about 7M raw records) |
| **Primary metric** | Macro F1 (also Accuracy, Balanced Accuracy) |
| **Benchmark** | Raw-label test results are the numbers comparable with classmates. Whether classmates use the same task and metric is unknown. |

**Required final deliverables:**
1. Four notebooks named `01_descriptive_analysis`, `02_baselines`, `03_transfer_learning_finetuning`, `04_semi_supervised`. Repo files keep their own numbering, and the README maps repo names to the required names. Known repo names: baseline = `07_baselines_tfidf`, masking = `06`, label correction = `04`.
2. An offline prediction script. It must warn when `n_features == 0`.
3. A README.
4. A final report with at least 20 analyzed errors.
5. `release_report.json`.

**Overall pipeline:**
1. Cleaning.
2. Product-grouped split.
3. Step-1 label correction.
4. Masking of labels for semi-supervised learning, plus a per-body cap (done earlier).
5. **Baseline: TF-IDF + LR (done, this handoff).**
6. Variant C (sample weighting).
7. Separate band analysis.
8. Transformers.
9. Semi-supervised learning.
10. Error analysis.
11. Offline script, README, final report, `release_report.json`.

---

## 2. Status snapshot

| Item | Status |
|---|---|
| Baseline phase (notebook 07, steps 0–14b, charts 01–08) | **DONE and closed.** All 8 charts approved by the user. The final Persian phase report was delivered. |
| Step 15 (optional tfidf13: ngram (1,3), 1,000,000 features, B models only, same C, decision on validation only) | **Not done.** The assistant recommended skipping it because of time pressure. **The user has not confirmed. Ask or assume skip.** |
| Next planned step | **Variant C** (sample weighting, rec head only) |
| Jupyter kernel | Restarted after the baseline phase. `vec`, `tr`, `DATA`, `MODELS`, `details`, `artifact_table`, `closeout_checks`, `checklist_frame` are **not in memory**. Load everything from files. |
| `closeout_checks.csv` | **Does not exist** (kernel restarted before the save cell ran). The 11 checks (all PASS) are recorded only in notebook cell output. Do not reconstruct by re-running cell 19. |
| `config.json` | Unchanged since `artifact_table.csv` was written. sha256 re-verified after restart. |

**Hard "do not" list:**
- Do **not** re-run notebook-07 cell 19 (Step 14a). It may rewrite `config.json`. Any change to `config.json` requires rewriting `artifact_table.csv`.
- Do **not** delete `data/interim/tfidf_cache/` until variant C is finished.
- Do **not** re-run cell 6 of notebook 06 without cell 7.
- DuckDB has a single writer. The baseline phase reads CSVs, not the DB.
- Do **not** `uv add timer` or `import timer`. `timer` is our own function in `utils.py`.

---

## 3. Repository layout

Root: `/home/alireza/AI-Bootcamp/Project-3-digikala-review/digikala-review-transformer` (`config.ROOT`).
Items marked **[verified]** were seen in outputs. Others are **(not inspected)**.

```
digikala-review-transformer/
├── .cache
├── .git
├── .venv
├── artifacts
│   ├── baseline_tfidf
│   │   ├── class_weights.json
│   │   ├── config.json
│   │   ├── tfidf12_band_A.joblib
│   │   ├── tfidf12_band_B.joblib
│   │   ├── tfidf12_rec_A.joblib
│   │   ├── tfidf12_rec_B.joblib
│   │   └── vectorizer_tfidf12.joblib
│   ├── Step1_label_correction
│   │   ├── rating_band
│   │   │   └── label_correction_bands_rating_band.html
│   │   ├── rating_band_more_precise
│   │   │   └── rating_band_more_precise.html
│   │   └── recommendation_status
│   │       └── label_correction_bands.html
│   └── .gitkeep
├── assets
│   └── fonts
│       ├── Poppins-Bold.ttf
│       ├── Poppins-Medium.ttf
│       ├── Poppins-Regular.ttf
│       ├── Vazirmatn-Bold.ttf
│       ├── Vazirmatn-Medium.ttf
│       └── Vazirmatn-Regular.ttf
├── data
│   ├── external
│   │   ├── parsivar_spell_extracted
│   │   │   ├── mybigram_lm.pckl
│   │   │   └── onegram.pckl
│   │   └── parsivar_spell.zip
│   ├── interim
│   │   ├── dropped_tables_archive
│   │   │   ├── body_backup_latin.parquet
│   │   │   ├── defect_manifest.parquet
│   │   │   ├── dup_plan.parquet
│   │   │   ├── dup_plan_s3.parquet
│   │   │   ├── len_sample.parquet
│   │   │   └── rule_summary.parquet
│   │   ├── duckdb_tmp
│   │   ├── explorer_candidates_tfidf12
│   │   │   ├── tfidf12_band_B.parquet
│   │   │   └── tfidf12_rec_B.parquet
│   │   ├── tfidf_cache
│   │   │   ├── ids_tfidf12_test.parquet
│   │   │   ├── ids_tfidf12_train.parquet
│   │   │   ├── ids_tfidf12_validation.parquet
│   │   │   ├── meta_tfidf12.json
│   │   │   ├── X_tfidf12_test.npz
│   │   │   ├── X_tfidf12_train.npz
│   │   │   └── X_tfidf12_validation.npz
│   │   ├── body_groups_rating_band.parquet
│   │   ├── body_groups_recommendation_status.parquet
│   │   ├── cap_dropped_rows.parquet
│   │   ├── cap_overcap_bodies.parquet
│   │   ├── clean_base_backup.duckdb
│   │   └── train_roles_mask.parquet
│   ├── processed
│   │   ├── label_correction_params.json
│   │   ├── natural_unlabeled.csv
│   │   ├── product_split.csv
│   │   ├── split_manifest.json
│   │   ├── test.csv
│   │   ├── train_pseudo_unlabeled.csv
│   │   ├── train_supervised.csv
│   │   └── validation.csv
│   ├── raw
│   │   ├── digikala-comments.csv
│   │   └── digikala-products.csv
│   └── .gitkeep
├── figs
│   ├── 41624398-907b-81f1-b466-0040ccdccd8f.png
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
│   ├── 04_label_correction_step1.ipynb
│   ├── 05_db_cleanup.ipynb
│   ├── 06_masking_and_artifacts.ipynb
│   ├── 07_baselines_tfidf.ipynb
│   └── 08_emoji_correction.ipynb
├── reports
│   ├── baseline_tfidf
│   │   ├── explorer
│   │   │   └── errors_explorer_baseline.html
│   │   ├── figures
│   │   │   ├── 01_dummy_vs_models_and_overlap.png
│   │   │   ├── 02_c_tuning_curves.png
│   │   │   ├── 03_confusion_matrices.png
│   │   │   ├── 04_per_class_metrics.png
│   │   │   ├── 05_label_views.png
│   │   │   ├── 06_a_vs_b.png
│   │   │   ├── 07_top_ngrams.png
│   │   │   └── 08_confidence.png
│   │   ├── tables
│   │   │   ├── ab_comparison.csv
│   │   │   ├── artifact_table.csv
│   │   │   ├── band_by_rate_zero.csv
│   │   │   ├── confidence_table.csv
│   │   │   ├── confusion_long.csv
│   │   │   ├── data_checks.csv
│   │   │   ├── dummy_baseline.csv
│   │   │   ├── final_fit.csv
│   │   │   ├── high_conf_errors.csv
│   │   │   ├── metrics_per_class.csv
│   │   │   ├── metrics_summary.csv
│   │   │   ├── overlap_with_train.csv
│   │   │   ├── reproducibility_check.json
│   │   │   ├── step1_changed_rows_check.csv
│   │   │   ├── timing.csv
│   │   │   ├── top_ngrams.csv
│   │   │   ├── tuning_results.csv
│   │   │   └── vocab_probe.csv
│   │   └── workflow.html
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
│   │   ├── Digikala Recommendation Distribution (Editorial).png
│   │   ├── group_split_&_lorenz_poster.png
│   │   ├── identical_text_label_conflicts.png
│   │   ├── identical_text_share.png
│   │   ├── latin_conversion_outcome.png
│   │   ├── lorenz_chart_split_curves.png
│   │   ├── Parsivar_rejection.png
│   │   ├── Poster - Tokenizer max_length (4_3).png
│   │   ├── product_review_lorenz_by_split.png
│   │   ├── rate_distribution_curve.png
│   │   ├── rating_band_distribution.png
│   │   ├── review_length_distribution.png
│   │   └── The Art of Emoji Spacing.png
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
│   │   ├── label_correction_recommendation_status_sweep.png
│   │   └── workflow.html
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
│   │   ├── English finglish convert.png
│   │   └── latin_comments_all.csv
│   ├── masking
│   │   ├── band_share_by_rec_missing.csv
│   │   ├── cap_effect.csv
│   │   ├── cap_effect_top20.png
│   │   ├── cap_roles_fingerprint.json
│   │   ├── cap_summary.csv
│   │   ├── class_distribution_by_segment.png
│   │   ├── csv_outputs.json
│   │   ├── dist_band_counts.csv
│   │   ├── dist_band_pct.csv
│   │   ├── dist_rec_counts.csv
│   │   ├── dist_rec_pct.csv
│   │   ├── handoff_update.md
│   │   ├── hash_check.json
│   │   ├── mask_overlap_raw_vs_corrected.json
│   │   ├── mask_plan.csv
│   │   ├── mask_roles_fingerprint.json
│   │   ├── masking_figures_charts.html
│   │   ├── mermaid-diagram.png
│   │   ├── phase_close_check.json
│   │   ├── pseudo_body_overlap_after_cap.csv
│   │   ├── pseudo_body_overlap_before_cap.csv
│   │   ├── pseudo_corrected_share.json
│   │   ├── pseudo_raw_vs_corrected_pseudo_after_cap.csv
│   │   ├── pseudo_raw_vs_corrected_pseudo_before_cap.csv
│   │   ├── role_counts_after_cap.csv
│   │   └── role_counts_after_mask.csv
│   ├── predictions
│   │   ├── tfidf12_band_A__test.parquet
│   │   ├── tfidf12_band_A__validation.parquet
│   │   ├── tfidf12_band_B__test.parquet
│   │   ├── tfidf12_band_B__validation.parquet
│   │   ├── tfidf12_rec_A__test.parquet
│   │   ├── tfidf12_rec_A__validation.parquet
│   │   ├── tfidf12_rec_B__test.parquet
│   │   └── tfidf12_rec_B__validation.parquet
│   ├── vocab_analysis
│   │   ├── distinctive_terms.csv
│   │   ├── distinctive_terms.html
│   │   ├── frequent_terms.html
│   │   ├── most-distinctive-bigrams-recommended-vs.png
│   │   └── vocabulary-analysis-all-raw.png
│   ├── model_registry.csv
│   ├── product_concentration_differences_vs_global.csv
│   ├── split_class_comparison.csv
│   ├── split_final_summary.json
│   └── top10_most_commented_products.csv
├── src
│   └── digikala_review_transformer
│       ├── templates
│       │   └── explorer_template.html
│       ├── __init__.py
│       ├── baseline.py
│       ├── closeout.py
│       ├── config.py
│       ├── data_io.py
│       ├── error_analysis.py
│       ├── explorer.py
│       ├── folders_tree_list_helper.py
│       ├── labels.py
│       ├── metrics.py
│       ├── plot_style.py
│       ├── plots.py
│       ├── predictions.py
│       ├── repro.py
│       ├── split.py
│       └── utils.py
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

**Naming rules for outputs:**
- Save each result once.
- No similarly named duplicates.
- Figures use `NN_name.png`, enforced by regex `\d{2}_[a-z0-9_]+\.png`.
- `save_figure` refuses a second file with the same number prefix.

**Audit result (`reports/`):**
- 142 files: 36 from the baseline phase, 106 from earlier phases.
- 0 similar names within a folder.
- 17 same-name files across folders, all in earlier-phase `label_correction_*` and `latin_comments_for_translation`.

---

## 4. Data and labels

### 4.1 Data lineage

- `clean_base` (in `project.duckdb`): **6,151,947** rows.
- Split by **product** (no product in two splits):

| Split | Rows | Products |
|---|---|---|
| train | 4,306,348 | 232,397 |
| validation | 922,794 | 49,911 |
| test | 922,805 | 49,276 |

- `train_role` inside train:

| `train_role` | Rows |
|---|---|
| `supervised` | 2,860,236 |
| `pseudo_unlabeled` | 635,824 |
| `cap_dropped_supervised` | 709,796 |
| `cap_dropped_pseudo` | 100,492 |

- **Masking:** 20% of each class (`MASK_RATE=0.20`) is moved to the pseudo-unlabeled pool. The mask is a hash: `substr(sha256('digikala-semi-supervised-v1|'||id),1,16)`.
- **Cap:** at most 50 copies of the same `body` per pool.
- `natural_unlabeled.csv` (468,079 rows) holds naturally unlabeled rows.
- **Step-1 correction** changed 15,801 rec rows (0.301%) and 57,242 band rows (0.930%).
- Config constants: `SPLIT_SEED=2026`, `MASK_SEED`, `MASK_RATE=0.20`, `PSEUDO_LABEL_THRESHOLD=0.90`, `DB_PATH`, `CLEAN_TABLE`, `ROOT`, `DATA_RAW`, `DATA_INTERIM`, `DATA_PROCESSED`, `ARTIFACTS`, `REPORTS`.

### 4.2 CSV columns and reading rules

- **`train_supervised.csv` columns:** `id, body, product_id, recommendation_status, step1_rec_label, rating_band, step1_rate_band_label, rate`.
- **Raw labels:** `recommendation_status`, `rating_band`.
- **Corrected labels (variant B):** `step1_rec_label`, `step1_rate_band_label`. **The corrected band column is `step1_rate_band_label`, NOT `step1_band_label`.**
- **Reading rules:**
  - Read CSVs as strings with `keep_default_na=False`. Bodies like "None" or "NA" must survive.
  - Only label columns convert "" to NA.
  - `rate` is numeric.
  - `id` is always `str`.
  - Use `data_io.read_split`.
- **Label codes:**
  - rec: `recommended=0`, `not_recommended=1`, `no_idea=2`.
  - band: `low=0`, `medium=1`, `high=2`.
- **Variants:**
  - A = raw labels. B = Step-1 corrected labels.
  - The user prioritizes B. Both are always reported, and the raw label is shown next to corrected results.
- **`MODEL_IDS` is a dict** `{(head, variant): model_id}`:
  - `('rec','A'):'tfidf12_rec_A'`, `('rec','B'):'tfidf12_rec_B'`
  - `('band','A'):'tfidf12_band_A'`, `('band','B'):'tfidf12_band_B'`
  - Use `set(MODEL_IDS.values())` for membership checks.
- **Training rows:**
  - rec trains only on rec-labeled rows: 2,392,157.
  - band trains on all 2,860,236 (the test split has 1 row with missing band).
  - No empty bodies.
- **Class counts in training (codes 0, 1, 2):**

| Model | Counts |
|---|---|
| rec_A | 1,832,583 / 266,817 / 292,757 |
| rec_B | 1,834,524 / 266,638 / 290,995 |
| band_A | 553,520 / 505,064 / 1,801,652 |
| band_B | 549,788 / 500,933 / 1,809,515 |

- **Validation distribution:** rec no_idea 11.07%, not_recommended 9.67%, recommended 79.26%. band high 65.85%.
- **Two evaluation views:** `raw` (vs raw label) and `corrected` (vs corrected label). Never mix splits or views in one comparison.

---

## 5. Baseline phase: design and decisions

**Design:**
- Four models: rec_A, rec_B, band_A, band_B.
- One shared vectorizer: word-level TF-IDF, ngram (1,2), no stopword removal, fitted on all 2,860,236 `train_supervised` bodies.
- Multinomial Logistic Regression (lbfgs), `class_weight='balanced'`. Only C is tuned.
- C is chosen on validation by Macro F1. If the F1 difference is below `TIE_TOL=0.0005`, the smaller C wins.
- Test is used only for final numbers.

**Decisions and deviations:**

| # | Decision | Reason |
|---|---|---|
| 1 | Token pattern `[\w\u200c]+` (deviation from sklearn default) | The default breaks tokens at ZWNJ. Ours gives 2,659 ZWNJ tokens per 50k bodies (default: 0). 3.8% of bodies contain ZWNJ. |
| 2 | No stopword removal | Negations ("نه"، "نیست"، "نبود") must survive. With the default pattern, the loss would be نه 0%, نیست 0.28%, نبود 0.36%. |
| 3 | `min_df=5`, `max_features=500,000` | Noise and memory. The cap is active (raw vocabulary about 4.5M, df≥5 about 0.59M). |
| 4 | Tuning on a 500k-row subset, then final fit on all rows with fresh C candidates | Full-data tuning was too slow. 12 of 18 tuning fits did not converge at `max_iter=200`. Final candidates were defined separately. |
| 5 | Final fits use `max_iter=1000`; ConvergenceWarning is never silenced | All 12 final fits converged. `converged` is logged. |
| 6 | No statistical tests, no bootstrap | User decision. Small differences (A vs B) are not called "better". |
| 7 | No seen/unseen-body evaluation; `overlap_with_train` table and `body_in_train` flag stay informational | User decision. |
| 8 | Latin→Persian conversion ignored (`normalize.py` not needed) | No effect on evaluation. |
| 9 | Explorer error tags are manual and optional (only `model_correct_label_wrong`, `unclear_text`, `model_error`) | No official error-category list exists. |
| 10 | Charts are built at the end of the phase | User request. |

**Statistical honesty rules:**
- The Step-1 correction was built from the majority label per body. It is **not independent** evidence.
- A higher metric on the corrected view comes from the corrected label itself, not from training.
- The instructor was not consulted about the correction.

---

## 6. Parameters (notebook cell 2)

| Group | Parameters |
|---|---|
| Vectorizer | `MODEL_TAG=tfidf12`, `NGRAM=(1,2)`, `MIN_DF=5`, `MAX_FEATURES=500_000`, `SUBLINEAR_TF=True`, `TOKEN_PATTERN=[\w\u200c]+` |
| Tuning | `C_GRID=[0.3,1,3,10]`, `CLASS_WEIGHT=balanced`, `SOLVER=lbfgs`, `MAX_ITER=200`, `TOL=1e-4`, `TUNE_N=500_000`, `PROBE_N=200_000`, `N_JOBS=4`, `TIE_TOL=0.0005`, `EXTEND_FACTOR=3.0`, `SLOWDOWN=1.5`, `FULL_FIT_MAX_MIN=5`, `MAX_PEAK_RAM_GB=20`, `PROBE_SIZES=[50k,100k,200k,300k]` |
| Final fit | `MAX_ITER_FINAL=1000`, `FINAL_C_CANDIDATES={"rec":[0.3,1.0],"band":[0.1,0.3]}`, `EXT_C_CANDIDATES={"rec":[0.1],"band":[0.03]}`, `N_JOBS_FINAL=8` |
| Analysis | `HIGH_CONF=0.90`, `LOW_CONF=0.5`, `CONF_BINS=[0,.5,.6,.7,.8,.9,1]`, `SHORT_MAX_WORDS=3`, `NEG_WORDS=["نه","نیست","نبود"]`, `NEG_FLAG_WORDS`, `NEG_PREFIXES=["نمی"]`, `TOPK_NGRAMS=25`, `TOPN_MIN_DF=100`, `RANK_TERMS`, `TOP_FEATURES_K=5` |
| Explorer | `EXPLORER_SEED=SPLIT_SEED`, `EXPLORER_QUOTAS`, `EXPLORER_MODEL_IDS`, `EXPLORER_CANDIDATES_DIR=data/interim/explorer_candidates_tfidf12`, `EXPLORER_HTML_PATH`, `EXPLORER_FONT_DIR`, `EXPLORER_TITLE/SUBTITLE`, `EXPLORER_ERROR_TAGS` |
| Reproducibility | `REPRO_N_ROWS=1000`, `REPRO_SEED`, `REPRO_PROBA_TOL=1e-5`, `REPRO_JSON_PATH`, `REPRO_DEMO_SENTENCES` (8 sentences) |
| Close-out | `CLOSE_ARTIFACT_DIR=ARTIFACTS/"baseline_tfidf"`, `CLOSE_TABLES_DIR=REPORTS/"baseline_tfidf"/"tables"`, `CLOSE_PRED_DIR`, `CLOSE_SPLITS=("validation","test")`, `CLOSE_REFERENCE_MODELS={"rec":"tfidf12_rec_B","band":"tfidf12_band_B"}`, `CLOSE_EXPECTED_C={"rec":0.3,"band":0.1}`, `CLOSE_EXPECTED_TRAIN_ROWS`, `CLOSE_EXPECTED_VOCAB=500_000`, `CLOSE_IDF_TOL=1e-4`, `CLOSE_ARTIFACT_TABLE_CSV`, `CLOSE_EXPECTED_TABLES`, `CLOSE_STEP11_CONFIG_KEYS` |
| Charts | `PLOT_FONT_DIR=ROOT/assets/fonts`, `FIGURES_DIR=REPORTS/baseline_tfidf/figures`, plus `CHART0N_*` parameters (split, label view, variant, heads) |

---

## 7. Key results

### 7.1 Final models

| Model | Selected C | Validation Macro F1 (final fit) | Iterations |
|---|---|---|---|
| rec_A | 0.3 | 0.684732 | 254 |
| rec_B | 0.3 | 0.687386 | 262 |
| band_A | 0.1 | 0.576202 | 207 |
| band_B | 0.1 | 0.583427 | 241 |

- All converged. `n_features_in = 500,000`.
- In each of the 4 selected fits, the chosen C had the highest Macro F1 among that model's candidates.
- Tuning-stage picks (500k subset) were rec C=1 and band C=0.3. They differ from the final picks.
- The final-fit F1 is about 0.010–0.011 above the tuning-stage F1.

### 7.2 Vectorizer

- Vocabulary 500,000 (447,644 bigrams, 7,620 with ZWNJ), median df 11, fit 39 s.
- nnz: train 57,544,256 (20.1 per row), validation 15,467,273, test 15,423,463. Dtype float32.
- Rows with zero features: train 3,724, validation 1,033, test 1,050.
- idf rebuilt from train document frequencies: max abs difference 7.73e-07 (confirms the vocabulary and idf come from train only).

### 7.3 Test results (raw label view unless stated)

| Model | Accuracy | Macro F1 | Balanced Acc | Corrected: Acc / F1 / BalAcc |
|---|---|---|---|---|
| rec_A | 0.8134 | 0.6847 | 0.7380 | 0.8166 / 0.6874 / 0.7442 |
| rec_B | 0.8137 | 0.6849 | 0.7378 | 0.8169 / 0.6876 / 0.7440 |
| band_A | 0.6672 | 0.5781 | 0.5891 | 0.6768 / 0.5858 / 0.6001 |
| band_B | 0.6656 | 0.5776 | 0.5894 | 0.6752 / 0.5852 / 0.6004 |
| dummy rec | 0.7891 | 0.2940 | 0.3333 | |
| dummy band | 0.6489 | 0.2624 | 0.3333 | |

- The dummy always predicts the train majority class (rec: recommended, band: high). A and B dummies are identical.
- Gap model vs dummy (B, test raw, percentage points): rec Accuracy +2.46, Macro F1 +39.09. band Accuracy +1.67, Macro F1 +31.52.
- Validation corrected Macro F1: rec_B 0.6874, band_B 0.5834.
- **A vs B (test raw):**
  - Differences are at most 0.16 points, with mixed signs.
  - Prediction disagreement: rec 1.079%, band 1.275%.
  - No statistical test exists, so neither variant is declared better. B is the reference by prior decision.

### 7.4 Per class (test, raw, B)

| Head | Class | Precision | Recall | F1 | True | Predicted |
|---|---|---|---|---|---|---|
| rec | recommended | 0.963 | 0.851 | 0.904 | 621,691 | 549,226 |
| rec | not_recommended | 0.651 | 0.775 | 0.708 | 77,160 | 91,778 |
| rec | no_idea | 0.356 | 0.588 | 0.444 | 88,994 | 146,841 (1.65×) |
| band | low | 0.529 | 0.514 | 0.521 | 165,299 | 160,709 |
| band | medium | 0.369 | 0.504 | 0.426 | 158,723 | 216,625 (1.36×) |
| band | high | 0.824 | 0.750 | 0.785 | 598,782 | 545,470 |

- **Confusion rec_B (test raw, row %):**

| True ↓ / Pred → | recommended | not_recommended | no_idea |
|---|---|---|---|
| recommended | 85.09 | 2.10 | 12.81 |
| not_recommended | 3.20 | 77.47 | 19.32 |
| no_idea | 19.97 | 21.27 | 58.77 |

- **Band ordinal errors (test raw):** band_A mean_abs_distance 0.434 (neighbor 69.68%, extreme 30.32%). band_B 0.435 (69.82% / 30.18%).
- **Weak classes:** `no_idea` and `medium`. Small classes are over-predicted because of `class_weight='balanced'`. The reason for the weakness at label level was not analyzed.

### 7.5 Confidence (validation, B)

| Bin | rec: share | rec: acc raw | rec: acc corrected | band: share | band: acc raw | band: acc corrected |
|---|---|---|---|---|---|---|
| 0–0.5 | 7.44% | 0.4545 | 0.4545 | 31.04% | 0.4966 | 0.4969 |
| 0.5–0.6 | 11.67% | 0.5646 | 0.5648 | 23.78% | 0.6290 | 0.6312 |
| 0.6–0.7 | 12.01% | 0.6837 | 0.6846 | 19.52% | 0.7481 | 0.7551 |
| 0.7–0.8 | 12.11% | 0.7902 | 0.7917 | 12.37% | 0.8067 | 0.8226 |
| 0.8–0.9 | 15.33% | 0.9010 | 0.9051 | 6.83% | 0.8431 | 0.8762 |
| 0.9–1.0 | 41.43% | 0.9651 | 0.9704 | 6.46% | 0.8935 | 0.9480 |

- **Observation (not diagnosed):** mid-bin mean confidence is below accuracy. In the top band bin, raw accuracy is 0.893 while mean confidence is 0.962. No calibration was done.
- **High-confidence errors (conf ≥ 0.9, validation, B):**

| Head | View | Errors | % of all errors | % of high-conf rows |
|---|---|---|---|---|
| rec | raw | 11,175 | 7.61 | 3.49 |
| rec | corrected | 9,471 | 6.56 | 2.96 |
| band | raw | 6,353 | 2.05 | 10.65 |
| band | corrected | 3,101 | 1.03 | 5.20 |

  - Top rec pairs: recommended→no_idea 3,863; no_idea→recommended 3,367; no_idea→not_recommended 1,110.
  - Top band pairs: high→medium 874; low→high 676; medium→low 501.

### 7.6 Error rates by flag (validation, B, vs corrected label; flag False / True)

| Flag | rec | band |
|---|---|---|
| short (≤3 words) | 20.9 / 10.9 | 36.1 / 23.2 |
| has negation | 14.7 / 36.8 | 30.6 / 42.8 |
| `body_in_train` | 23.3 / 9.2 | 37.6 / 23.8 |
| `label_changed` | 18.4 / 0.1 | 32.9 / 0.5 |
| `rate_is_zero` | | 29.1 / 68.5 |

- Flag shares: short 27.31%, negation 15.92%, `body_in_train` 36.42%, `rate_is_zero` 8.67%. Median 6 words, mean 10.1.
- `label_changed` has about 0 error because the correction comes from body-majority labels (circular).
- Flags are correlated, so no causal claims.

### 7.7 `rate = 0` in band

- Test: 79,264 rows (8.6%), all raw `low`. Validation: 80,013 rows.
- band accuracy is 0.264–0.267 on `rate=0` vs 0.703–0.705 on `rate>0`.
- About 19% of all band errors and about 45% of extreme errors (approximate).
- Hypothesis (**unverified**): `rate=0` means "no rating given", so it cannot be predicted from text. To be tested in the separate band analysis.

### 7.8 Overlap with train (informational)

| Group | validation | test |
|---|---|---|
| all | 36.42% (336,048 / 922,794) | 36.34% |
| recommended | 39.77% | 39.68% |
| label_missing | 41.78% | 41.81% |
| no_idea | 22.98% | 22.81% |
| not_recommended | 14.85% | 14.94% |

- This is exact body text found in train. It is not leakage (the split is by product). It makes evaluation optimistic for repeated texts.

### 7.9 Step-1 correction check (validation)

- rec: 2,471 changed rows. A and B predictions equal the corrected label in 99.76% and 99.88% of them.
- band: 9,034 rows, about 99.54%.
- band medium→low: 265 rows. 20 sampled bodies were all negative.
- The model agrees with the correction, but this is not independent evidence (circular).

### 7.10 Top n-grams (rec_B / band_B)

- rec recommended: عالیه (10.94), عالی (9.98), بهترین, خوبیه, خوبه, بد نبود, بد نیست, راضیم, خیلی خوبه, عالین.
- rec not_recommended: افتضاح, افتضاحه, اصلا, بدترین, متاسفم, راضی نیستم, نخرید.
- rec no_idea: مطمئن نیستم, معمولی, معمولیه, نمیدونم, ولی, اما, شک دارم.
- band low: افتضاح, اصلا, نخرید. band medium: معمولی, معمولیه, ولی, نبود, نیست, اما. band high: عالی, عالیه, بهترین, بد نبود, بد نیست.
- The df≥100 filter did not change the top lists.
- Three suspicious items were not checked: «نازک نیست», «حرف», «فرستادن».
- Coefficients are not true contribution shares (tf-idf values are not multiplied in).

### 7.11 Reproducibility (Step 13)

- A fresh Python process (child pid 47797, parent 25183, 12.6 s) reloaded the artifacts and predicted 1,000 validation rows.
- All 4 models: `max_abs_diff_proba = 0.0`, `n_pred_mismatch = 0`.

**Demo sentences (B models):**

| Sentence | rec (conf) | band (conf) |
|---|---|---|
| عالی بود، دقیقا مثل عکس. پیشنهاد می‌کنم | recommended 0.99 | high 0.79 |
| افتضاح بود، اصلا نخرید | not_recommended 1.00 | low 1.00 |
| معمولی بود | no_idea 0.96 | medium 0.97 |
| بد نیست | no_idea 0.96 | medium 0.90 |
| خوب نیست | not_recommended 0.75 | low 0.60 |
| کیفیت خوبه ولی قیمتش بالاست | no_idea 0.62 | medium 0.61 |
| نمیدونم، هنوز استفاده نکردم | no_idea 0.95 | medium 0.72 |
| `!!!` (0 features) | no_idea 0.38 | low 0.36 |

- A text with 0 features still gets a prediction. The offline script must warn when `n_features == 0`.

---

## 8. Artifacts and tables

### 8.1 Environment

python 3.12.16, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, joblib 1.6.0, matplotlib 3.11.2, fonttools 4.68.0. `arabic-reshaper` 3.1.0 and `python-bidi` 0.6.12 are installed but **unused**; they break Persian rendering in matplotlib 3.11.2.
Hardware: i9-13900H (14 cores), 32 GB RAM, RTX 4060 (sklearn runs on CPU only).
**Disk is tight (keep at least 10 GB free).** Prefer extra columns over extra tables, and keep small intermediates as parquet in `data/interim/`. Never modify existing `clean_base` columns.

### 8.2 Artifacts (sha256)

| File | sha256 |
|---|---|
| `vectorizer_tfidf12.joblib` (21.33 MB) | `f3f25e98e7e34707a52195dfe58b9bdb342deab7b6c664e13ba00d296b71b62d` |
| `tfidf12_rec_A.joblib` (6.00 MB) | `c70938d99072163090f9510d29f40821fc2f7a03508e4ab913c7c16368522303` |
| `tfidf12_rec_B.joblib` | `3e5a8bfd1e9ce00c1442739926ac097a190ab4ade2e1603d11fc3ea3c4dbbc4f` |
| `tfidf12_band_A.joblib` | `a35ef0b85a73db4195734d5c561d89d9f64d89c5adeb2348251c373a0b7c5e89` |
| `tfidf12_band_B.joblib` | `c6cc132907c65480cf4e89369c27c437f448ad906c1e80d3537e572755de35ca` |
| `class_weights.json` | `d72476dc537dab690b4db12d2a720664fdbb3545a847aa0b82b5b7ad3ef6a037` |
| `config.json` | `3ed8eab340f30cdb58f3d858c7b57ea5959b5fd2fe1c999ac8dbcbaf5ee37fb6` |

- `config.json` keys: `artifact_table_csv, conf_bins, explorer_quotas, explorer_seed, final_settings, high_conf, label_codes, library_versions, low_conf, model_tag, negation_prefixes, negation_words, phase_closed_at, reference_models, selected_models, short_max_words, short_max_words, train_rows_by_model`.
- Loading rule: a joblib may hold the object directly or a dict holding it. `repro._unwrap` handles both. Use `closeout.load_vectorizer`.

### 8.3 Tables in `reports/baseline_tfidf/tables/` (18 files)

| File | Contents (rows) |
|---|---|
| `data_checks.csv` | Step-1 sanity checks: `check, got, expected, ok` |
| `dummy_baseline.csv` | 16 rows. `model_id, head, variant, split, majority_class, n, accuracy, macro_f1, balanced_accuracy, macro_precision, macro_recall, label_view, mean_abs_distance, neighbor_error_pct, extreme_error_pct` |
| `vocab_probe.csv` | Vocabulary growth, nnz, time and RAM on nested subsets |
| `tuning_results.csv` | 18 rows. `model_id, C, val_macro_f1, val_accuracy, fit_seconds, n_iter, converged, n_train_rows, tune_n, val_n, extended, selected, at_edge` |
| `final_fit.csv` | 12 rows. `model_id, C, val_macro_f1, val_macro_recall, val_balanced_accuracy, val_accuracy, fit_seconds, n_iter, converged, n_train_rows, max_iter, selected, artifact, sha256, size_mb`. Only `val_*` metric columns. |
| `metrics_summary.csv` | 16 rows. `model_id, head, variant, split, n, accuracy, macro_f1, balanced_accuracy, macro_precision, macro_recall, label_view, mean_abs_distance, neighbor_error_pct, extreme_error_pct` |
| `metrics_per_class.csv` | 48 rows. `model_id, head, variant, split, label_view, class, precision, recall, f1, support, predicted_count` |
| `confusion_long.csv` | 144 rows. `model_id, head, variant, split, label_view, true, pred` (class names), `count, row_pct` (0–100) |
| `ab_comparison.csv` | 8 rows × 19 columns. `head, split, label_view`, `{accuracy,macro_f1,balanced_accuracy,macro_recall,mean_abs_distance}_{A,B}`, `*_diff_B_minus_A`, `pred_disagreement_pct` |
| `band_by_rate_zero.csv` | Band accuracy and class shares for `rate==0` / `>0` / missing |
| `top_ngrams.csv` | 300 rows (rec_B and band_B only). `model_id, filter ('all','df>=100'), class, rank, feature, coef, df_train, is_bigram, has_zwnj` |
| `confidence_table.csv` | 12 rows, validation only (no split column). `conf_from, conf_to, n, pct_rows, mean_conf, n_raw, accuracy_raw, n_corrected, accuracy_corrected, model_id` |
| `high_conf_errors.csv` | Errors with conf ≥ 0.9, total and per (true, pred) pair, per label view |
| `step1_changed_rows_check.csv` | How A and B predict rows whose raw and corrected labels differ |
| `overlap_with_train.csv` | 10 rows. `split, group, n, n_in_train, pct_in_train` |
| `artifact_table.csv` | 7 rows. `kind, path, size_mb, sha256` |
| `reproducibility_check.json` | Fresh-process comparison result |
| `timing.csv` | Append-only log. `time` is step END time. Several repeated runs exist and the reason was not recorded. Final-run seconds: step 6b 376, step 7 477, step 7b 136, step 5b 57. |

### 8.4 Predictions and registry

- **Predictions:** `reports/predictions/<model_id>__<split>.parquet` for 4 models × {validation, test}.
  - Contents: `id`, raw and corrected true codes (NA allowed), predicted code, `p_<class>` ×3, `conf`. No body or `product_id`.
  - Exact column names were **not re-verified for this handoff**. Use `predictions.read_predictions` and inspect.
  - Row order equals the split CSV (asserted for all 8).
- **Registry:** `reports/model_registry.csv`, 4 rows. Columns: `model_id, family, head, variant, train_label_col, ngram, C, class_weight, n_train_rows, artifact_path, created_at`.
- **Cache:** `data/interim/tfidf_cache/` holds the matrices for train, validation and test, the id order and the vectorizer. `baseline.load_cache` asserts the id order.

### 8.5 Charts (all approved)

Files are `reports/baseline_tfidf/figures/NN_*.png`, 3200×1800 each:
- 01 dummy vs models and overlap
- 02 C tuning curves
- 03 confusion matrices
- 04 per-class metrics
- 05 label views
- 06 B minus A
- 07 top n-grams
- 08 confidence

### 8.6 Error explorer

- **File:** `reports/baseline_tfidf/explorer/*.html`, a standalone file with embedded JSON and no server.
- **Layout:** two tabs (rec, band), three-column layout, keys ↑↓, tag shortcuts 0–3, localStorage prefix `digikala_explorer_baseline_v1::`. CSV export `explorer_tags_baseline.csv` (with BOM).
- **Colors:** it keeps its own purple theme. The user said no edits.
- **Candidates:** rec_B 7,300 rows, band_B 7,500 rows (seeded selection by error, high/low confidence, short, negation).
- **Candidate columns:** `model_id, pos, id, true_code, raw_code, pred_code, conf, true_name, pred_name, raw_name, is_error, label_changed, n_words, is_short, has_negation, body_in_train, rate, rate_is_zero, cand_reason, top_features_pred, body, p_<class>×3`.
- **Tagging:** manual and optional. The user keeps it.
- **Verification:** `verify_explorer_html` re-reads the HTML and compares it with the parquet files.

---

## 9. Module index (`src/digikala_review_transformer/`)

Signatures come from real `inspect.signature` output. The notebook only calls functions. Reusable logic lives in modules. Functions are parametric with short docstrings, and paths come from `config.py`.

| Module | Responsibility |
|---|---|
| `config` | Constants and paths (no functions) |
| `utils` | sha256, JSON, timer |
| `labels` | Fixed class order, label columns per head/variant, encoding, class weights |
| `data_io` | CSV reading with NA rules, sanity checks, per-head training data |
| `metrics` | Metrics, per-class table, confusion, A/B comparison, band by `rate=0` |
| `predictions` | Prediction parquet files and model registry |
| `baseline` | TF-IDF vectorizer, cache, LR training, C tuning, n-grams, explanations |
| `error_analysis` | Row flags, confidence table, high-confidence errors, explorer candidate selection |
| `explorer` | Build and verify the HTML explorer |
| `repro` | Fresh-process reproducibility; `predict_texts` is the core of the offline script |
| `closeout` | Phase close-out checks, artifact table, config merge |
| `plot_style` | "Aurora Glass" style (no purple), canvas, tiles, safe save |
| `plots` | 8 chart functions |

```python
# utils
sha256_file(path, chunk=1048576)
write_json(obj, path)
update_json(path, updates)
timer(name, csv_path=None, **extra)

# labels   (constants: CLASSES, LABEL_COLS, RAW_COL, CORR_COL -- structure not re-inspected)
label_map(head)                 # {class_name: int_code}
encode(series, head)            # nullable Int8; missing stays <NA>; unknown values raise
decode(codes, head)
class_weights(y, n_classes=3)   # {code: weight}, from train labels only

# data_io
read_split(path)
read_ids(path)
check_splits(splits, pseudo_ids, natural_ids, manifest, csv_paths,
             expected_rows, expected_rec_labeled, expected_band_missing)
build_head_data(df, head, variant)   # only rows with a valid label

# metrics   (private: _valid, _tag)
evaluate(y_true, y_pred, n_classes=3, ordinal=False)
per_class_table(y_true, y_pred, head)
confusion_long(y_true, y_pred, head)
evaluate_all(y_raw, y_corr, y_pred, head)
dummy_predict(y_train, n)                    # (code, predictions)
macro_f1_from_confusion(cm_long, classes)
build_metric_tables(pred_dir, models, splits=('validation','test'))
ab_comparison(summary, pred_dir, models, splits=('validation','test'))
band_by_rate_zero(pred_dir, band_models, rates, splits=('validation','test'))

# predictions   (constant: REGISTRY_COLS)
pred_path(pred_dir, model_id, split)
write_predictions(pred_dir, model_id, split, ids, y_raw, y_corr, proba, head)
read_predictions(pred_dir, model_id, split)
register_model(registry_csv, row)            # insert or replace by model_id

# baseline   (private: _centered_coef; constant: DEFAULT_TOKEN_PATTERN)
tokenize(body, pattern, lowercase=True)
vocab_growth_probe(bodies, sizes, ngram_range, token_pattern, min_df, seed)
make_vectorizer(ngram_range, token_pattern, min_df, max_features, sublinear_tf)
fit_and_cache(vec, bodies, ids, cache_dir, tag, vec_path)    # fits on train only
load_cache(cache_dir, tag, expected_ids)                     # asserts id order
train_lr(X, y, C, class_weight='balanced', solver='lbfgs',
         max_iter=200, tol=0.0001, sample_weight=None)       # returns (model, info)
fit_eval_one(model_id, C, X_train, pos, y, X_val, pos_val, y_val, head, train_params)
pick_c(df, tie_tol)
tune_c(X_train, X_val, data, c_grid, train_params, n_jobs, tune_n, seed,
       tie_tol, extend_factor=3.0)
fit_final_candidates(X_train, X_val, data, cand_c, train_params, n_jobs,
                     art_dir, tie_tol, prev=None)
top_ngrams(model, names, df_train, classes, k, min_df=None)
term_ranks(model, names, classes, terms)
explain_rows(model, X, names, target, k=5)

# error_analysis
in_train_mask(train_bodies, bodies)
body_overlap_share(train_bodies, splits, label_col)
negation_pattern(words, prefixes=())
add_flags(bodies, train_bodies, rate, neg_words, neg_prefixes, token_pattern, short_max=3)
confidence_table(p, bins)
high_conf_summary(p, head, high_conf)
step1_changed_check(preds, head)
select_candidates(df, quotas, high_conf, low_conf, seed)

# explorer   (private: _to_native, _font_face_css; constants: TEMPLATE_PATH, REQUIRED_COLUMNS)
load_candidates(model_id, candidates_dir)
build_records(candidate_df, proba_columns)
build_explorer_html(model_ids, candidates_dir, out_html_path, error_tags,
                    page_title, page_subtitle, font_dir=None, template_path=TEMPLATE_PATH)
verify_explorer_html(html_path, model_ids, candidates_dir)

# repro   (private: _unwrap; constants: HEAD_CLASS_NAMES, HEADS, VARIANTS)
predict_texts(texts, vectorizer, model, class_names)
    # one row per text: pred_code, pred_name, conf, n_features, p_<class>
reproducibility_check(model_tag, n_rows, seed, proba_tol, demo_sentences,
                      artifact_dir, predictions_dir, validation_csv)
run_fresh_process_check(model_tag, n_rows, seed, proba_tol, demo_sentences, out_json)
main()
check_frame(result)
demo_table(result, model_ids)

# closeout
library_versions()
make_check(name, passed, evidence, status_if_false='FAIL')
load_vectorizer(artifact_dir, model_tag)
model_details(artifact_dir, model_ids)
check_registry(registry_csv, model_ids, root, expected_columns)
check_prediction_ids(pred_dir, model_ids, splits, split_csv_paths)
check_vectorizer_fit_set(vectorizer, train_bodies)
build_artifact_table(named_paths, root)
merge_new_config_keys(config_path, additions, overwrite_keys=())

# plot_style   (private: _assert_no_purple)
init_style(font_dir)                     # once per session
fa(text)                                 # returns raw str; NO reshape/bidi
make_background(width=1600, height=900, seed=2026, grain=0.02)
new_canvas(title, subtitle, footer)
split_row(n_columns, weights=None, gap=0.015, top=0.84, bottom=0.065)
style_axes(ax)
add_tile(fig, left, bottom, width, height, tile_title=None,
         pad_left=0.05, pad_right=0.025, pad_bottom=0.09, pad_top=None)
glass_legend(ax, handles=None, labels=None, **legend_kwargs)
save_figure(fig, path)
style_smoke_figure()

# plots   (each saves one PNG and returns an info dict; reads saved tables only)
chart_01_dummy_vs_models_and_overlap(tables_dir, figures_dir, model_tag, split, label_view, variant, overlap_splits)
chart_02_c_tuning_curves(tables_dir, figures_dir, model_tag, heads, variants, highlight_variant)
chart_03_confusion_matrices(tables_dir, figures_dir, model_tag, split, label_view, variant, heads)
chart_04_per_class_metrics(tables_dir, figures_dir, model_tag, split, label_view, variant, heads)
chart_05_label_views(tables_dir, figures_dir, model_tag, split, variant, heads, label_views)
chart_06_a_vs_b(tables_dir, figures_dir, model_tag, split, label_view, heads)
chart_07_top_ngrams(tables_dir, figures_dir, model_tag, variant, heads, feature_filter, top_n)
chart_08_confidence(tables_dir, figures_dir, model_tag, split, variant, heads, high_conf_from)
```

**Notes:**
- **Renamed function:** `fit_final_models` → `fit_final_candidates`. The old name must not exist (verified: it does not).
- **Variant C:** `train_lr` already accepts `sample_weight`.
- **`split` and `folders_tree_list_helper`:** they exist in the package but were not inspected. Inspect before use.
- **Docstrings:** `baseline` and `error_analysis` module docstrings still say "more functions are added in later steps" (stale, harmless). `config`, `predictions.pred_path` and `predictions.read_predictions` have no docstring.
- **Offline script:** `predict_texts` is its core. It needs an argument parser, loading of the saved vectorizer and model, and the `n_features == 0` warning.

---

## 10. Notebook 07 map and in-memory state

**Cells:**

| Cell | Content |
|---|---|
| 1 | imports (+ `%load_ext autoreload`, `%autoreload 2`) |
| 2 | parameters |
| 3 | Step 1 |
| 4 | Step 2 |
| 5 | Step 4 |
| 6 | Step 5a |
| 7 | Step 5b |
| 8 | Step 6a |
| 9 | Step 6b |
| 10 | Step 7 |
| 11 | Step 7b |
| 12 | Step 8 |
| 13 | Step 9 |
| 14 | Step 10 |
| 15 | Step 11a |
| 16 | Step 11b |
| 17 | Step 13 |
| 18 | Step 12 |
| 19 | Step 14a (**do not re-run**) |
| 20 | chart pre-inspection |
| 21 | style smoke test |
| 22–29 | charts 01–08 |
| 30 | the failed `closeout_checks` save and module-index cell (only the module-index part was later run as a separate read-only cell) |

**Memory:** after the kernel restart, only parameters, imports and `info_08` exist. Rebuild anything else from files. The notebook still has its original cell order, so a later restart-and-run-all would re-run cell 19. Warn the user, or guard cell 19 before any run-all.

---

## 11. Conventions and user constraints

- **Code style:**
  - Parameters at the top of the notebook (cell 2), cell 1 = imports.
  - Meaningful variable names, short English comments.
  - The notebook only calls functions.
  - Functions are parametric with short docstrings, and paths come from `config.py`.
  - The user wants to learn the reasoning behind settings.
- **Never silence `ConvergenceWarning`.**
- **Never guess:** no assumed column names, structures or signatures. Inspect first.
- **Test discipline:** test is used for final numbers only. Decisions (C, model choice, thresholds) use validation only.
- **Comparison discipline:** same split and same label view in every comparison.
- **Instructor:** not consulted. Raw test is the classmate benchmark.
- **Charts ("Aurora Glass", NO PURPLE, no pink):**
  - figsize (16,9), dpi 200 → 3200×1800. Background #0B1020 with blurred aurora blobs and ~2% grain.
  - Text #E6E9F5, muted #9AA4C7. Glass tiles: white alpha 0.09, border alpha 0.28.
  - Class colors: recommended/high turquoise #2DD4BF, not_recommended/low coral #FB7185, no_idea/medium amber #FBBF24. B blue #38BDF8, A #94A3B8, dummy #64748B.
  - Latin text in Poppins, Persian only for n-grams and bodies in Vazirmatn (`fontfamily=PERSIAN_FONT`, raw string via `fa`).
  - Avoid Unicode minus U+2212 (Poppins support unknown). Check for overlap and clipping, and keep legends off data.
- **Persian text in matplotlib 3.11.2:** do not use `arabic_reshaper` or `bidi`. Raw strings render correctly.
- **Long outputs:** if Jupyter truncates ("Output is truncated"), the user opens it "in a text editor" and pastes it all, or send smaller cells.
- **Chart text:** English only, except Persian n-grams and bodies.

---

## 12. Past mistakes and dead ends (do not repeat)

- **Wrong band column name** in an old handoff (`step1_band_label`). Correct: `step1_rate_band_label`.
- **Wrong assumptions:** "band similar in pseudo and supervised"; an expected small extreme-error share in band (actually about 30%); expecting «بد نیست» among no_idea coefficients (it leans to recommended/high).
- **Wrong explanation:** B's higher score on its own label was first explained by memorization. The truth is that the gain comes from the corrected label itself.
- **Mixed-view comparison:** validation-corrected was compared with test-raw once. Always match the split and the view.
- **Weak asserts:** an `assert ... or True`, a wrong Step-4 assert (`cd == co`), a vacuous assert in cell 14. All fixed.
- **Wrong estimate:** the Step-7 time estimate. Replaced by measurement.
- **Unreliable claim:** "tuning on a subset biases C smaller". Do not repeat it.
- **Rejected methods:** Confident Learning with row removal, co-teaching, DivideMix, `char_wb`, unseen-body evaluation, bootstrap, class-weight search, a signed-BIGINT hash.
- **Assumed list:** a project-supplied error-category list. None exists.
- **Inspection pitfall:** `inspect.getsource(plots)` printed only imports (output truncated). Request helpers in small cells.
- **Chart diagnosis:** a diagnostic figure with 0 axes printed only a repr. Use axes and `display(Image(...))`. Glyph coverage was not the cause of the Persian rendering problem.
- **`timer` shadowing:** `import timer` shadowed our own function. Fixed.
- **Time-file check for "test unused":** weak evidence. The strong evidence is that the `tuning_results` and `final_fit` tables hold only `val_*` metric columns.
- **`step12_build_explorer`:** logged 0.19 s, probably only partial timer coverage (unconfirmed).
- **`closeout_checks.csv`:** never saved (kernel restart). See section 2.

---

## 13. Open observations (for the later error analysis)

1. **«بد نیست» as a single sentence** is predicted `no_idea` (0.96), although its bigram coefficient ranks 7th for recommended (rank 87 for no_idea). The cause is unanalyzed (possibly the single-word features «بد» and «نیست»).
2. **Mid-bin confidence below accuracy** and the top-bin overconfidence in band (conf 0.962 vs raw accuracy 0.893). Not diagnosed; no calibration.
3. **`rate=0` hypothesis** (no rating given) is unverified.
4. **Suspicious n-grams** «نازک نیست», «حرف», «فرستادن» were not checked.
5. **High-confidence errors** often have intrinsically ambiguous labels: «خوب بود» labeled no_idea → recommended 0.93; «معمولی» labeled recommended → no_idea 0.99; band «معمولی» high → medium 0.99.
6. **Why `no_idea` and `medium` are weak** was not analyzed at label level.
7. **Bodies are exact duplicates for about 36% of evaluation rows**, so metrics are optimistic for repeated texts.
8. **Chart 07 printout:** the `bigram_pct`, `zwnj_shown` and `zwnj_flag_mismatch` values were never shown to the assistant. Only 2 of 300 n-gram rows contain ZWNJ.

---

## 14. Remaining roadmap

1. **Step 15 (optional, skip unless the user insists).** Variant `tfidf13`: ngram (1,3), `MAX_FEATURES=1_000_000`, B models only, same C, decision on validation only. Needs a new `MODEL_TAG`, a new cache, and a check of disk space and RAM first.
2. **Variant C: sample weighting (rec head only).**
   - OOF weights: `w = max(0.1, 1 − C·(1 − Q))`, from 5-fold `StratifiedGroupKFold` by `product_id`.
   - **Q and C are defined in the user's roadmap, and their exact definitions are not stored in this handoff. Ask the user for the exact formula definition before implementing.**
   - Use `train_lr(..., sample_weight=...)` on the cached matrices.
   - Reuse `data/interim/tfidf_cache/` (do not delete it before this variant finishes).
   - Compare against rec_B on the same split and view.
3. **Separate band analysis,** including `rate=0` and the ordinal errors.
4. **Transformers** (models in `models/pretrained/`):
   - TookaBERT-Base, max_length 128.
   - roberta-fa-zwnj-base, max_length 128.
   - mdeberta-v3-base, max_length 160.
   - Label smoothing 0.1, 1–2 epochs, early stopping on validation Macro F1 with a best-epoch checkpoint, class weights from train (`labels.class_weights`).
   - Model selection on validation only. Use the same prediction-parquet and registry formats so the existing metric, A/B, confidence and chart code can be reused.
   - GPU is an RTX 4060, so check memory and batch size.
5. **Semi-supervised learning:** pseudo-label threshold 0.90 (`PSEUDO_LABEL_THRESHOLD`) on `train_pseudo_unlabeled` (635,824 rows) and possibly `natural_unlabeled` (468,079 rows). Re-evaluate on the same validation and test.
6. **Error analysis:** at least 20 analyzed errors. Use the explorer, the high-confidence errors and observations from section 13. The user may tag manually (3 tags).
7. **Offline prediction script:**
   - Built on `repro.predict_texts` with the saved vectorizer and model.
   - Must warn when `n_features == 0`.
   - Reference models: `tfidf12_rec_B` and `tfidf12_band_B`.
8. **README** (with notebook-name mapping), **final report**, **`release_report.json`.**

**Reference models for all later comparisons:** `tfidf12_rec_B` and `tfidf12_band_B`.
- Use test with the raw view for classmate comparison.
- Use validation for every decision.
- Always show the dummy line next to model results.

---

## 15. First actions for a new assistant

1. Greet in Persian, briefly. Confirm you read this handoff.
2. Ask the user two things:
   - Skip Step 15? (default: skip.)
   - The exact definition of Q and C for variant C.
3. Before any code that depends on files, ask for a small read-only inspection:
   - The `predictions` parquet columns.
   - The `labels` constants structure.
   - The `data_io` and `baseline` function behavior you will call.
4. For variant C, plan first (steps, inputs, outputs, time and RAM estimate), wait for approval, then give one cell at a time.
5. Keep every result saved once, with no similarly named duplicates, and update this handoff at the end of each phase.