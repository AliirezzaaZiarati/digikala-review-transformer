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
- Required notebooks: `01_descriptive_analysis`, `02_baselines`, `03_transfer_learning_finetuning`, `04_semi_supervised`. Also required: offline prediction script, README, final report with at least 20 analyzed errors, and `release_report.json`.
- FastAPI and Docker are optional Phase 2.

## 2. Repository and technical setup
- Config file: `src/digikala_review_transformer/config.py` (with `__init__.py` beside it):
  - `ROOT` is the project root (three levels up from the config file).
  - Folders: `DATA_RAW = ROOT/data/raw`, `DATA_INTERIM = ROOT/data/interim`, `DATA_PROCESSED = ROOT/data/processed`, `ARTIFACTS = ROOT/artifacts`, `REPORTS = ROOT/reports`.
  - Constants: `SPLIT_SEED = 2026`, `MASK_SEED = "digikala-semi-supervised-v1"`, `MASK_RATE = 0.20`, `PSEUDO_LABEL_THRESHOLD = 0.90`.
- The contents of the data and artifacts folders are not tracked by Git.
- Cleaned data lives in a DuckDB database, in a table reportedly named `clean_base` (verify the exact database path and table name in the repo before use).
- Language conventions: code comments in English. The user communicates in Persian.

## 3. What has been done so far
1. **Data ingestion and cleaning finished.** About 6M+ comments remain after cleaning. No duplicate `id` values exist. Persian normalization was applied as part of cleaning.
2. **Observed data findings**, from the table of the most frequent bodies:
   - Many reviews have a body that contradicts the label (for example the body "عالی" labeled `not_recommended`).
   - The 40 most frequent bodies cover about 522k rows. Only about 2.6k of those are `not_recommended`, and roughly 850 of those are legitimately ambiguous ("معمولی", "بد نیست", "بد نبود", "خوبه بد نیست"). Such ambiguous bodies must not be forced to `recommended`, since they carry the `no_idea` signal.
   - For the body "عالی", 2,721 of 3,816 low-band rows (71%) have exactly `rate = 0`. This may be a "not rated" placeholder, which is **not yet verified** and would make `rating_band` noise larger than recommendation noise.
   - A large share of rows (about 14%, as reported earlier) has no `recommendation_status`.
   - Classmates report 60–70% validation accuracy. The task and class balance behind that figure are unknown.
3. **Strategy decisions** (documented in a separate decision report):
   - **Step 1, majority-vote correction:** group rows by normalized body, using groups with at least N = 20 rows. If one label's share is at or above a threshold (to be swept in 0.90–0.99), relabel the minority rows. The majority table comes from train only, and validation and test are corrected through that train-derived table. Keep the original label column beside the corrected one.
   - **Duplicate cap:** at most 50 copies per body in train only, keeping the smallest `stable_hash(id)`.
   - **Step 2, soft sample weighting:** out-of-fold TF-IDF + LR probabilities from 5-fold `StratifiedGroupKFold` (group = `product_id`) give a per-row weight `w = max(0.1, 1 − C·(1 − Q))`. Recommendation head only. Stop rule: if the mean weight of `no_idea` is more than about 0.05 below the other classes, drop the idea. Fallback estimator: frozen-encoder embeddings + LR.
   - **Noise-robust training:** label smoothing 0.1, class weights from train, 1–2 epochs, early stopping on validation Macro F1, smaller learning rate for the encoder than for the heads. Note that label smoothing reduces how many rows pass the 0.90 pseudo-label threshold, so the accepted count should be reported.
   - **Order and evaluation:** correct labels first, then mask, which is a documented deviation from the shared protocol. Report both raw and corrected metrics on validation and test, with the raw test set as the benchmark comparable with classmates.
   - **Variants:** A = raw labels, B = step-1 corrected, C = B plus sample weights. Compare them on the baseline and one Transformer, then use the best variant for the other two encoders. Optionally ensemble the three at the end by averaging softmax outputs.
   - **Rejected for time:** Confident Learning with removal, co-teaching and DivideMix. An LLM is used only for auditing 20–30 samples.
4. **Models chosen, downloaded and ready:**

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
7. **Product-grouped split finished and verified.**
   - Method: deterministic product-grouped split, 70/15/15 by row share, `SPLIT_SEED = 2026`. One shared split for both tasks. Rows with a missing recommendation label or invalid `rate` stay in their product's split.
   - Main output: product-to-split mapping `[path/filename, planned: data/processed/product_split.csv]` with columns `product_id, split` (values `train` / `validation` / `test`). `clean_base` was left untouched. Row-level splits are obtained by joining on `product_id`.
   - Final sizes: train `[rows / products]`, validation `[rows / products]`, test `[rows / products]`.
   - Diagnostics done before the split: product concentration (largest product share `[x%]`, top-10 share `[x%]`, products without `product_id`: `[n]`), class distributions of `recommendation_status` (missing share `[x%]`) and `rating_band` over the full data, `rate` distribution and `rate = 0` share (`[x%]`, finding: `[short note]`).
   - Post-split checks passed: no product in more than one split, row shares close to 70/15/15, class distributions (including the missing-label share) similar across splits, one row got exactly one split, split reproducible across two runs, token-length percentiles consistent with `max_length` 128 / 128 / 160.
   - Split report written: `[path]`.
   - Other output files created in this phase: `[list of files and paths, if any beyond the mapping, e.g. split_manifest.json status]`.

## 4. Current status
- Data cleaning and the product-grouped split are finished and verified. The split is final and must not be changed.
- Not done yet: label correction (step 1) and its threshold sweep, masking, duplicate cap, final CSV artifacts `[adjust if already created]`, baselines, sample weighting, rating_band analysis, Transformer fine-tuning, semi-supervised experiment, error analysis, offline prediction script and final report.

## 5. Open items
- Confirm the duplicate cap and the correct-before-mask order with the instructor.
- Decide whether the pseudo-unlabeled pool is also capped.
- Decide the size of the deterministic modeling subset if hardware is limited.
- Verify the meaning of `rate = 0`. Changing band definitions requires documented evidence and approval.
- Final threshold values for step 1 come from the data.

## 6. Working-style notes for the assistant
- The user wants concise, direct answers without filler, and step-by-step plans when a roadmap is requested (purpose, inputs, outputs, expected result, approval criterion, fallback), with no extra steps.
- Do not run tests or experiments in a sandbox or use tools unnecessarily. The user has limited tokens. Write code or guidance as text and let the user run it.
- The user wants to learn the reasoning behind each setting, not only get instructions.