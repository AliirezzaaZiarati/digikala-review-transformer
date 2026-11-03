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
| `src/digikala_review_transformer/` | shared code (config, normalization, model) |
| `notebooks/` | 01 descriptive analysis, 02 baselines, 03 fine-tuning, 04 semi-supervised |
| `data/` | raw, interim, processed (not tracked) |
| `artifacts/` | saved models, tokenizer, preprocessing (not tracked) |
| `reports/` | figures, metrics and written reports |

Seeds and paths live in `src/digikala_review_transformer/config.py`.
