# Digikala Review Transformer

Persian Digikala review analysis from the review `body` only: recommendation status and rating-band classification, a TF-IDF baseline vs a fine-tuned Transformer, and a semi-supervised pseudo-labeling experiment.

## Setup

Requires Linux, an NVIDIA GPU with a recent driver, and [uv](https://docs.astral.sh/uv/).

    uv sync
    uv run python -c "import torch; print(torch.cuda.is_available())"

## Data

Download the public dataset ([RadeAI/Digikala_comments_products](https://huggingface.co/datasets/RadeAI/Digikala_comments_products)), extract the CSV files into `data/raw/` (create it if missing). Data and model weights are not tracked by Git.

## Layout

| Path | Content |
|---|---|
| `src/digikala_review_transformer/` | shared code (config, normalization, model, sample weighting, plots) |
| `scripts/` | long-running resumable scripts (e.g. `step2_run_oof.py`) |
| `notebooks/` | 01 descriptive analysis, 02 baselines, 03 fine-tuning, 04 semi-supervised |
| `data/` | raw, interim, processed (not tracked) |
| `artifacts/` | saved models, tokenizer, preprocessing, weighting config (not tracked) |
| `reports/` | figures, metrics and written reports |

Seeds and paths live in `src/digikala_review_transformer/config.py`.

## Sample weights (label-noise weighting)

Every `train_supervised` row gets one weight per head, `w_rec` and `w_band`, in [0.1, 1]. Rows whose recorded label looks suspicious get a low weight but are never removed. Weights are used only in the per-sample training loss on `train_supervised`; validation and test rows have no weights and are never used to compute them.

**Method.** A product-grouped 5-fold out-of-fold (OOF) TF-IDF + logistic regression gives, for each row, the probability of every class. Per head, on rows with a corrected label:

    p_y    = OOF probability of the recorded label
    m_c    = mean p_y over all OOF rows of class c
    Q      = min(1, p_y / m_c)
    C      = 1 - H_norm,  H_norm = -sum_k p_k * ln(p_k) / ln(3)
    w      = max(0.1, 1 - C * (1 - Q))

Rows with no TF-IDF feature get `w = 1` and are excluded from `m_c`. `w_rec` is NaN where the row has no recommendation label. Only corrected (Step 1) labels are used.

**Notebooks**
| Phase / Bootcamp Module | Notebook(s) | Key Outputs |
|---|---|---|
| `11_transfer_learning_finetuning` (Multi-Task Transformer Fine-Tuning: Stage 1 Variant B vs C & Stage 2 Encoder Comparison) | `notebooks/10_1st_transformer_finetune.ipynb` (Stage 1: TookaBERT B vs C) and Stage 2 4-run analysis notebook | `reports/transformer/tables/` (`primary_model.json`, `stage1_decision.json`, `run_comparison.csv`, `training_settings.csv`, `stage1_test_metrics.csv`, `stage1_vs_tfidf.csv`, `stage1_learning_summary.csv`, ...) & `reports/transformer/figures/01..06_*.png` |

**Outputs** (not tracked by Git).

| File | Content |
|---|---|
| `data/processed/step2_sample_weights.parquet` | `id`, `n_features_zero`, and `p_y_*`, `q_*`, `c_*`, `w_*` for `rec` and `band` (float32) |
| `artifacts/step2_weighting/weights_config.json` | parameters, class means, library versions, fit summaries, artifact checksums |
| `artifacts/step2_weighting/step2_decision.json` | keep/drop decision record |
| `reports/step2_weighting/` | tables and figures |

Reference SHA-256:

| File | SHA-256 |
|---|---|
| `step2_sample_weights.parquet` | `af4422a0db37e660f32ff5a51c9b7e1c285655275afb859d3eb37a3babc94d87` |
| `weights_config.json` | `96adb179ca6c61e1aa6c561a33f51bd88a3444ddabdbc4a04a30edf42dec6bac` |

**Reproduce.** Run `notebooks/09_step2_sample_weights.ipynb` cell by cell. The OOF fits run in a separate resumable script, started from the repository root after the notebook has written the params file:

    .venv/bin/python scripts/step2_run_oof.py \
        --params "$PWD/data/interim/step2_oof/step2_params.json" --n-jobs 3

Peak memory is about 1.4 GB per worker.

**Use in the Transformer loss.** Join the weights to the training rows by `id`. Compute the per-sample loss with `reduction='none'`, multiply by `w`, and divide by the sum of `w` over the valid rows of the batch, not by the batch size. For `rec`, rows with NaN `w_rec` have no label and are masked out. Pseudo-labeled rows use `w = 1`. This is compatible with label smoothing and class weights.

**Results.**

| Head | Rows with w < 0.5 | Mean w | OOF macro-F1 | Validation reference |
|---|---|---|---|---|
| rec | 1.04% | 0.9603 | 0.6767 | 0.6874 |
| band | 0.26% | 0.9781 | 0.5720 | 0.5834 |

Decision: **keep**. The weights are mild, so the effect on the Transformer is expected to be small; it is measured by a controlled comparison of a weighted run against a `w = 1` run with identical settings and seed.

**Limitations.**

- The manual review of 100 rows was qualitative. Row-level verdicts were not recorded and the review thresholds were not evaluated.
- All OK/WARN thresholds are engineering guesses, not tuned values.
- OOF macro-F1 is about 0.011 below the validation reference for both heads; the cause was not determined.
- Logistic-regression probabilities (class-balanced) are not calibrated.
- `Q` and `C` ignore the order of the rating bands.
- The TF-IDF vectorizer was fit on all of the training text (no labels).
- Folds use a single seed.