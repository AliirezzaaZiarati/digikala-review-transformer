# Project 3: Intelligent Analysis of Digikala Reviews

**Phase 1 (required): Semi-Supervised Persian Review Classification and Rating-Band Prediction with Transformers**

**Phase 2 (optional): Productize the trained artifacts with FastAPI and Docker after those topics are taught.**

This notebook defines a complete modeling project first and a separate deployment extension second. The word **Karvand** is used for each learner.

## 1. Problem Statement

Each Karvand must build an intelligent system that uses only the review body (`body`) to analyze Persian Digikala reviews. The required Phase 1 system predicts whether the user recommends the product, predicts an ordered `rating_band` derived from the numeric `rate`, and evaluates a public, reproducible semi-supervised protocol. The saved artifacts must also support a simple offline prediction script.

FastAPI and Docker are not required for Phase 1; they form an optional Phase 2 after the relevant deployment topics are taught.

### Central question

> Can the text of a Persian product review predict both the user's recommendation status and ordered rating band, and can unlabeled reviews improve the classifier?

## 2. Learning Objectives

By the end of Phase 1, each Karvand should be able to:

- audit and describe a multi-million-row text dataset;
- normalize Persian text without destroying sentiment or negation;
- prevent product-level data leakage;
- build a TF-IDF baseline and a fine-tuned pretrained Transformer;
- formulate recommendation and ordered rating-band classification tasks;
- run one reproducible semi-supervised experiment;
- analyze errors and class-specific behavior;
- save model, tokenizer, preprocessing, and split artifacts for reproducible offline inference.

Phase 2 optionally adds API design, service testing, and containerization after FastAPI and Docker have been taught.

## 3. Dataset and Scope

The complete public Digikala review dataset, containing approximately 7 million records, will be available to all Karvands. No labels or rows are hidden.

Dataset URL: https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/main

> Just download and extract the rar file!


The complete dataset must be used for data auditing and descriptive statistics. For memory-safe processing, use CSV chunks, Polars, DuckDB, or another streaming approach. Transformer training may use a deterministic, fully documented modeling subset when hardware is limited; the same seed and sampling rule must be used by all Karvands.

Before modeling, run a schema and quality gate: verify the required columns, ID uniqueness policy, parseable `rate` values, the valid rating range, and the observed label vocabulary. Count malformed, duplicate, missing, and out-of-range records and document every exclusion or correction; do not silently coerce invalid data.

## 4. Columns and Input Policy

The dataset includes columns such as `id`, `title`, `body`, `created_at`, `rate`, `recommendation_status`, `is_buyer`, `product_id`, `advantages`, `disadvantages`, `likes`, `dislikes`, `seller_title`, `seller_code`, and `true_to_size_rate`. Verify the actual schema before processing.

The only model input is `body`. The `title` column must not be used. Target columns and metadata such as `rate`, `recommendation_status`, `product_id`, seller identifiers, timestamps, likes, dislikes, advantages, and disadvantages must not be used as model features. `product_id` is allowed only for **grouped splitting**.

## 5. Task A: Recommendation Classification

Target column: `recommendation_status`. First inspect the actual unique values. If the public file uses Persian labels, different spellings, or extra categories, define and document a deterministic mapping to these canonical classes:

- `recommended`
- `not_recommended`
- `no_idea`

Rows without this label are not supervised classification examples. Because class imbalance is expected, Macro F1 and per-class Precision, Recall, and F1 are more important than Accuracy alone.

## 6. Task B: Rating-Band Classification

Use the raw `rate` column only to derive an ordered target named `rating_band`; never pass `rate` itself to the model. Unless the data audit shows a different scale, use the common 0–5 bands:

- `low`: 0 ≤ rate < 2.5
- `medium`: 2.5 ≤ rate < 4.0
- `high`: 4.0 ≤ rate ≤ 5.0

Treat missing, non-numeric, and out-of-range rates as invalid for this task and report them in the quality audit. Record these thresholds and the ordinal encoding `low=0`, `medium=1`, `high=2` in `split_manifest.json`, then apply them unchanged to validation and test. If the meaning or range of `rate` differs, document the evidence and obtain approval before changing the bands.

Required metrics are Macro F1, per-class Precision/Recall/F1, Confusion Matrix, Balanced Accuracy, and ordinal distance error. Predicting the exact numeric `rate` is an optional extension.

# Part A — Descriptive Statistics and Data Understanding

This part is mandatory and is worth 10 points. It must be completed before model training. The goal is to use evidence from the data to justify modeling decisions, not to produce decorative plots.

## A1. Dataset Size and Structure

Using the complete dataset, report the true number of records with a CSV parser, number of columns, data types, file size, number of unique products and sellers, and the distribution of review counts per product. Do not count physical lines as rows because a review body may contain newlines.

## A2. Missing-Value Patterns

For every column, report missing count and missing percentage. Analyze whether missing `recommendation_status` is related to `is_buyer`, `rate`, product frequency, or other available metadata. Explain whether missing labels appear to be random or structurally biased.

## A3. Target Distributions

Report counts and percentages for each recommendation class and for the derived `rating_band` classes. Visualize both class distributions and discuss imbalance.

For the raw `rate` column, report minimum, maximum, mean, median, standard deviation, quantiles, number of unique values, and the fraction of fractional values. Compare `rate` with recommendation status and explain why the raw numeric value is used only to derive `rating_band`, not as a model input.

Use the fixed band thresholds from Task B and report their counts before training.

## A4. Text-Length Analysis

For `body`, measure character length, word length, median, upper percentiles, very short reviews, and very long reviews. Compare length distributions across classes. Use this analysis to justify the Transformer `max_length` and truncation policy.

## A5. Repetition, Conflicts, and Product Concentration

Report duplicate IDs, duplicate review bodies, identical texts with conflicting labels, and the concentration of reviews among products. Explain why product concentration motivates a product-grouped train/validation/test split.

A temporal or seller-specific analysis is optional; it should not replace the core descriptive analyses above.

## A6. Vocabulary-Level Analysis (Optional Extension)

This analysis is valuable but not required for Phase 1. Interested Karvands may report frequent words and bigrams, class-specific terms, and differences between labeled and naturally unlabeled reviews. Stopword removal must be treated as an experiment rather than an unverified preprocessing rule.

## A7. Descriptive-Analysis Deliverable

Submit `01_descriptive_analysis.ipynb` containing:

- at least four readable plots;
- missing-value and target-distribution tables;
- text-length and repetition summaries;
- at least five defensible conclusions;
- a statement of whether each plot uses the full data or a bounded sample;
- the seed and sampling rule for every bounded sample;
- explicit links from findings to choices such as splitting, class weighting, max length, normalization, and evaluation metrics.

All global counts and descriptive statistics must be computed from the complete dataset, even if text examples and some plots use a documented sample. Vocabulary-level analysis is an optional extension.

# Part B — Cleaning, Splitting, and Label Protocol

Normalize Persian text carefully: convert Arabic `ي` to Persian `ی` and Arabic `ك` to Persian `ک`, normalize whitespace and zero-width characters, handle URLs and HTML, and make explicit decisions about numbers and emojis. Do not remove negation words without evidence.

**You can utilize public available tools like:**
- https://www.roshan-ai.ir/hazm/content/hazm/index.html
- https://github.com/ICTRC/Parsivar

## Data Split
- Create one shared, deterministic product-grouped split almost: 70% train, 15% validation, and 15% test.
- A product must not appear in more than one split.
- Try to make each split reasonably representative of both recommendation status and rating band.
- Rows with a missing target stay in their product's assigned split but are used only by a head for which the target is valid.
- Do not create separate row-level splits for the two tasks.

Fit the TF-IDF vocabulary, normalization statistics, class weights, and any other data-dependent preprocessing on the training split only.

## Public and Reproducible Masking Protocol

Masking is not intended to hide public labels. It creates the same semi-supervised experiment for all Karvands. Use exactly:

```python
SPLIT_SEED = 2026
MASK_SEED = "digikala-semi-supervised-v1"
MASK_RATE = 0.20
```

After the product-grouped split, select exactly 20% of labeled training rows within each recommendation class using the deterministic hash rule below. Move these rows to the artificial unlabeled pool. Never mask validation or test rows. The main comparison must use this common artificial pool; naturally unlabeled rows are an optional extension.


```python
import hashlib

SPLIT_SEED = 2026
MASK_SEED = "digikala-semi-supervised-v1"
MASK_RATE = 0.20

def stable_hash(row_id: str) -> int:
    key = f"{MASK_SEED}|{row_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16)

# Within each recommendation class in the training split, sort by
# stable_hash(id) and mask floor(MASK_RATE * class_count) rows.
# Publish counts, rounding behavior, and the full manifest.
```

## Standard Data Artifacts

Generate and document the following files:

```text
train_supervised.csv
train_pseudo_unlabeled.csv
natural_unlabeled.csv
validation.csv
test.csv
split_manifest.json
```

The artificial unlabeled file must contain only `id` and `body` for training purposes; `recommendation_status`, `rating_band`, and raw `rate` must be removed from that derived file. The original public dataset remains unchanged. `split_manifest.json` must record seeds, rating-band thresholds, split strategy, row counts, and masked counts per class.

# Part C — Modeling

Required baselines:

- TF-IDF + Logistic Regression for both classification targets;

The required deep-learning model is a pretrained encoder adapted to this dataset through transfer learning and fine-tuning.

## Mandatory Transfer Learning and Fine-Tuning

Each Karvand must select one publicly available pretrained Persian or multilingual encoder and adapt it to the Digikala review task. Suitable examples include ParsBERT-family models, `HooshvareLab/bert-fa-base-uncased`, or `xlm-roberta-base`; the selected checkpoint, tokenizer, license, and reason for selection must be documented.

### Required experiment

Fine-tune the selected encoder on the Digikala training split for both required classification heads. Use a smaller learning rate for the encoder than for the task head when appropriate. A resource-limited team may fine-tune only the final encoder block, but must document this choice.

The frozen feature-extraction comparison is an optional Phase 2 extension. Phase 1 must compare the TF-IDF baseline with the fine-tuned Transformer under the same product-grouped split and evaluation protocol. If a row lacks one target, mask only that head's loss; do not discard a valid target for the other head.

### Required report

- total and trainable parameter counts;
- tokenizer and maximum sequence length;
- learning rates and batch/gradient-accumulation settings;
- training and validation curves, selected checkpoint, and early-stopping rule;
- Macro F1 and per-class metrics for both heads;
- evidence of overfitting or instability;
- a reasoned explanation of whether fine-tuning improved the result.

Only the pretrained weights may come from outside the Digikala dataset. All task-specific training, validation, and test data must follow the project split. A smaller checkpoint is allowed when GPU memory is limited, but the limitation must be documented.

## Optional Exact-Rate Regression

Predicting the raw numeric `rate` is an optional extension. If used, add a regression head beside the recommendation and `rating_band` classification heads:

```text
total_loss = recommendation_loss + λ1 × rating_band_loss + λ2 × rate_regression_loss
```

Mask each loss when its target is unavailable. Report MAE, RMSE, and R² for the optional exact-rate output, but do not replace the required `rating_band` classification with regression.

## Semi-Supervised Learning

This experiment tests whether pseudo-labels can improve the required fine-tuned Transformer.

Use the same pretrained Transformer encoder selected for the transfer-learning task. The mandatory semi-supervised experiment applies only to the `recommendation_status` head. Pseudo-labeling for `rating_band` is optional.

### Procedure

1. Fine-tune the Transformer using `train_supervised`. This is the supervised model and the source of pseudo-label predictions.
2. Use that model to predict `recommendation_status` for rows in `train_pseudo_unlabeled`. For each row, take the class with the highest predicted probability. Accept the prediction as a pseudo-label only when that probability is at least `0.90`; discard lower-confidence predictions from pseudo-label training.
3. Create a second copy of the same model and continue fine-tuning it on `train_supervised` plus the accepted pseudo-labeled rows. Use the pseudo-labels only for the recommendation loss. If the model also has a `rating_band` head, mask that head's loss for pseudo-labeled rows because their rating-band targets are not available to the training procedure.
4. Compare the supervised and semi-supervised models on the same validation and test sets. Keep the model architecture, data split, preprocessing, and training settings the same. Do not use validation or test labels to create pseudo-labels or select them.

The public reference labels for `train_pseudo_unlabeled` may be used only after pseudo-label selection and training to audit pseudo-label quality. Do not provide those reference labels to the model, use them to select pseudo-labels, or tune the `0.90` threshold.

### Required report

Report:

- the number and percentage of rows that received accepted pseudo-labels;
- the pseudo-label class distribution;
- pseudo-label accuracy, calculated afterward against the public reference labels;
- Accuracy, Macro F1, Macro Precision, Macro Recall, and per-class metrics for both models on the same test set;
- the change in these test metrics after adding pseudo-labeled data.

A sweep over confidence thresholds such as `0.80`, `0.90`, and `0.95`, and pseudo-labeling for `rating_band`, are optional Phase 2 extensions.

## Evaluation and Error Analysis

For recommendation and `rating_band` classification, report Accuracy, Macro F1, per-class Precision/Recall/F1, Confusion Matrix, and Balanced Accuracy. For `rating_band`, also report ordinal distance error because confusing adjacent bands is less severe than confusing the lowest and highest bands.

Analyze at least 20 mistakes, including negation, sarcasm, short text, mixed opinions, spelling variation, duplicated text, contradictory labels, rating-band boundary cases, and disagreements between classical and fine-tuned models. Calibration, coverage-risk analysis, latency, memory use, and a 30-error audit are optional extensions.

If the optional exact-rate regression extension is implemented, additionally report MAE, RMSE, MSE, R², Spearman correlation, predicted-versus-actual plot, and residual analysis.

# Part D — Phase 2 Optional: FastAPI Prediction Service

This entire part is optional and should be attempted after FastAPI has been taught. It does not affect completion of Phase 1. The goal is to serve the saved TF-IDF and Transformer artifacts through a small, validated application.

Suggested endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service is alive |
| `GET` | `/ready` | Models and artifacts are ready |
| `POST` | `/normalize` | Persian text normalization |
| `POST` | `/tokenize` | Tokens, IDs, and attention mask |
| `POST` | `/predict/tfidf` | Classical model prediction |
| `POST` | `/predict/transformer` | Fine-tuned Transformer prediction |
| `POST` | `/predict/compare` | Side-by-side model comparison |

A minimal Phase 2 submission may implement only `/health`, `/normalize`, and `/predict`. Models should be loaded at startup rather than once per request.

## API Contract

Example request:

```json
{"text": "کیفیت محصول خیلی خوب بود اما بسته‌بندی ضعیف بود"}
```

The response should include model name, recommendation label and probabilities, `rating_band` and its probabilities, model version, token count, and latency. `predicted_rate` may be included only for the optional exact-rate regression extension. Requests must contain text only; users must not submit target labels or ratings. `/normalize` returns original and normalized text. `/tokenize` returns tokens, `input_ids`, `attention_mask`, token count, and truncation status.

## API Failure Modes and Tests (Optional Phase 2)

If the service is implemented, return `422` for empty or overlong text, `400` for an unknown model or tokenizer, `503` when artifacts are missing or models are not ready, and `500` for unexpected internal errors. Write at least five `TestClient` tests. Do not log raw user reviews; enforce a documented maximum text length.

# Phase 2 Optional Extension — Dockerize the FastAPI Service

Docker is not required for Phase 1 and does not affect its score. After implementing the optional FastAPI service, interested Karvands may package it as a Docker image so that it can run consistently on another machine.

### Minimum requirements

- provide a `Dockerfile`;
- install dependencies from `requirements.txt`;
- copy application code and model artifacts into the image or mount them at runtime;
- expose port `8000`;
- start the service with a production-style command such as `uvicorn app.main:app --host 0.0.0.0 --port 8000`;
- verify `/health` and `/ready` from inside the running container;
- document image size, model-loading time, and the CPU/GPU assumption.

### Minimal Dockerfile example

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Demonstration

```bash
docker build -t digikala-review-api .
docker run --rm -p 8000:8000 digikala-review-api
```

The report should discuss common container failure modes such as missing artifacts, an incorrect working directory, dependency incompatibility, a port mismatch, insufficient memory, and a model that expects CUDA while the container runs on CPU.

# Repository Structure

Phase 1 required files:

```text
project-3/
├── data/
├── notebooks/
│   ├── 01_descriptive_analysis.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_transfer_learning_finetuning.ipynb
│   └── 04_semi_supervised.ipynb
├── artifacts/
├── reports/
├── requirements.txt
└── README.md
```

The `app/`, `tests/`, and `Dockerfile` directories belong to the optional Phase 2 productization extension.

# Final Mini-Project

## Phase 1 required demonstration

A user provides one review body to an offline prediction script. The system normalizes and tokenizes it, runs the TF-IDF and fine-tuned Transformer classifiers, predicts both recommendation and `rating_band`, reports probabilities, and explains the result. Demonstrate short, ambiguous, negative, and mixed-opinion reviews.

## Phase 2 optional demonstration

Wrap the same artifacts in the optional FastAPI service and expose the predictions through HTTP. A side-by-side model comparison and latency report are optional extensions.

# Required Deliverables — Phase 1

- Complete, reproducible source code
- `01_descriptive_analysis.ipynb` over the full dataset
- `02_baselines.ipynb`
- `03_transfer_learning_finetuning.ipynb` with one fine-tuned pretrained encoder
- `04_semi_supervised.ipynb` with the fixed threshold `0.90`
- Split files and `split_manifest.json`
- Saved model, tokenizer, and preprocessing artifacts
- An offline prediction script
- README with installation and execution instructions
- Final report with at least 20 analyzed errors
- `release_report.json`
- Live demonstration or short presentation

FastAPI, API tests, frozen-feature extraction comparison, threshold sweeps, and Docker belong to the optional Phase 2 deliverables.

# Expected End-to-End Outcome

## Phase 1 required

```text
Full Data Audit and Descriptive Statistics
        ↓
Persian Text Normalization
        ↓
TF-IDF Baselines
        ↓
Fine-Tuned Transformer
        ↓
Recommendation and Rating-Band Classification
        ↓
One Controlled Semi-Supervised Experiment
        ↓
Error Analysis and Offline Inference
```

## Phase 2 optional

```text
Saved Artifacts
        ↓
FastAPI Prediction Service
        ↓
API Tests and Docker Packaging
```

The optional exact-rate regression extension may be added after the required two-head classification system.
