
"""Prediction files (one parquet per model and split) and the model registry."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .labels import CLASSES

TEHRAN_TZ = ZoneInfo("Asia/Tehran")

REGISTRY_COLS = ["model_id", "family", "head", "variant", "train_label_col", "ngram", "C",
                 "class_weight", "n_train_rows", "artifact_path", "created_at"]


def pred_path(pred_dir, model_id, split):
    return Path(pred_dir) / f"{model_id}__{split}.parquet"


def write_predictions(pred_dir, model_id, split, ids, y_raw, y_corr, proba, head):
    """Save id, raw/corrected true codes (NA allowed), pred, p_<class>..., conf. No body/product_id."""
    names = CLASSES[head]
    proba = np.asarray(proba, dtype=np.float32)
    n = len(ids)
    assert proba.shape == (n, len(names)) and len(y_raw) == n and len(y_corr) == n
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3), "probabilities do not sum to 1"

    df = pd.DataFrame({
        "id": np.asarray(ids).astype(str),
        "y_true_raw": pd.array(y_raw, dtype="Int8"),
        "y_true_corr": pd.array(y_corr, dtype="Int8"),
        "pred": proba.argmax(axis=1).astype("int8"),
    })

    for j, c in enumerate(names):
        df[f"p_{c}"] = proba[:, j]

    df["conf"] = proba.max(axis=1)

    path = pred_path(pred_dir, model_id, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    return path


def read_predictions(pred_dir, model_id, split):
    return pd.read_parquet(pred_path(pred_dir, model_id, split))


def register_model(registry_csv, row):
    """Insert or replace one row (by model_id) in the registry CSV.
    created_at uses the Asia/Tehran timezone.
    """
    row = {
        "created_at": datetime.now(TEHRAN_TZ).isoformat(timespec="seconds"),
        **row
    }

    missing = set(REGISTRY_COLS) - set(row)
    assert not missing, f"registry row lacks: {missing}"

    path = Path(registry_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    new = pd.DataFrame([{c: row[c] for c in REGISTRY_COLS}])

    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        new = pd.concat(
            [old[old["model_id"] != row["model_id"]], new],
            ignore_index=True
        )

    new.to_csv(path, index=False)

    return new
