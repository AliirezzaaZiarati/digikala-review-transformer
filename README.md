# Digikala Review Intelligence

Predict the **recommendation status** and the **rating band** of Persian Digikala reviews from the review `body` only. A TF-IDF baseline is compared with a fine-tuned Transformer, and the models are served through a FastAPI + Streamlit demo. Bootcamp Project 3.

> **Central question:** can the text of a review predict both the recommendation and the ordered rating band, and can unlabeled reviews help?

## Full report

Every phase, decision, table and figure is documented in a Notion hub. An **edit-access link has been shared with the project mentors**; the read-only link is here:

**[Digikala Review Intelligence: Project 3 Hub](https://flowery-mongoose-733.notion.site/Digikala-Review-Intelligence-Project-3-Hub-40a24398907b81779568eb059725a1b3?source=copy_link)**

This README covers only installation and execution. The project has two repositories:

| Repo | Content |
| --- | --- |
| `digikala-review-transformer` (this one) | data, analysis, training, reports |
| [`digikala_reviews_prod`](https://github.com/AliirezzaaZiarati/digikala_reviews_prod) | FastAPI service, Streamlit UI, Docker |

## Tasks

| Task | Target | Classes |
| --- | --- | --- |
| A | `recommendation_status` | `recommended`, `not_recommended`, `no_idea` |
| B | `rating_band` (from `rate`) | `low` < 2.5, `medium` 2.5 to < 4.0, `high` 4.0 to 5.0 |

The only model input is `body`. `rate` is used only to derive the band, and `product_id` only for the product-grouped split.

## Phases

| Phase | Content | Status |
| --- | --- | --- |
| P1–P2 | Data ingest, quality gate, full-data descriptive analysis, `max_length` choice | done |
| P3 | Product-grouped 70/15/15 split, masking, label correction (majority vote per identical body) | done |
| P4 | TF-IDF + Logistic Regression baseline, label-noise sample weights | done |
| P5 | Fine-tuned multi-task Transformer (shared encoder, two heads) | done |
| P6 | FastAPI service + Streamlit UI + Docker | done |
| P7 | Error analysis (where the models disagree) | see Notion |
| Semi-supervised | Pseudo-labeling on the recommendation head (threshold 0.90) | pending |

## Results (test set, raw labels, Macro F1)

| Model | Recommendation | Rating band |
| --- | --- | --- |
| TF-IDF + LR | 0.6849 | 0.5776 |
| **TookaBERT (fine-tuned)** | **0.7175** | **0.6007** |

Primary model: `PartAI/TookaBERT-Base`. Also compared: `HooshvareLab/roberta-fa-zwnj-base` and `microsoft/mdeberta-v3-base`. Details, checkpoints and licenses are in the Notion hub.

## Setup

Requires Linux, an NVIDIA GPU with a recent driver, and [uv](https://docs.astral.sh/uv/). The Transformer runs used bf16 on a 16 GB GPU.

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Data

Download the [public dataset](https://huggingface.co/datasets/RadeAI/Digikala_comments_products) and extract the CSV into `data/raw/`. Data and model weights are not tracked by Git.

## Run

Run the notebooks in order (all code lives in `src/digikala_review_transformer/`; notebooks only call it).

| Notebook | Content |
| --- | --- |
| `00_data_ingest_quality_gate` | ingest, schema and quality gate |
| `01_more_data_ingest_analysis` | descriptive analysis, normalization, `max_length` |
| `02_fetch_models` | download pretrained encoders |
| `03_data_split_&_products_analysis` | product-grouped split |
| `04_label_correction_step1` | label correction (majority vote) |
| `05_db_cleanup` | DuckDB cleanup |
| `06_masking_and_artifacts` | 20% masking, split CSVs, `split_manifest.json` |
| `07_baselines_tfidf` | TF-IDF + LR baseline |
| `08_emoji_correction` | emoji spacing fix |
| `09_step2_sample_weights` | label-noise sample weights |
| `10_1st_transformer_finetune` | Stage 1: TookaBERT, variant B vs C |
| `11_all_supervised_finetuned` | Stage 2: all encoders, primary model |
| `12_rearranging_reports` | report organization |
| `13_error_analysis` | error analysis and explorer |

Long jobs are scripts: `scripts/step2_run_oof.py` (sample weights) and `scripts/train_transformer.py` (YAML configs in `configs/transformer/`).

## Demo (optional, `digikala_reviews_prod`)

FastAPI serves the TF-IDF and Transformer models (chosen in `models.yaml`; weights go in `artifacts/baseline/` and `artifacts/encoders/`). The Streamlit UI only calls the API.

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
uv run streamlit run ui/app.py
```

or with Docker:

```bash
docker compose up --build
```

## Reproducibility

```python
SPLIT_SEED = 2026
MASK_SEED  = "digikala-semi-supervised-v1"
MASK_RATE  = 0.20
PSEUDO_LABEL_THRESHOLD = 0.90
```

Seeds, band thresholds, split strategy and masked counts are in `data/processed/split_manifest.json`. Deviation from the shared protocol: masking is applied after label correction (documented in the report).

## Layout

| Path | Content |
| --- | --- |
| `src/digikala_review_transformer/` | shared code (normalization, labels, split, baseline, training, plots) |
| `scripts/` | training and resumable long-running scripts |
| `notebooks/` | analysis notebooks |
| `configs/transformer/` | run configs (`tooka`, `roberta`, `mdeberta`; variants B/C) |
| `data/`, `artifacts/` | data, models, tokenizers, weights (not tracked) |
| `reports/` | figures, tables, interactive explorers |