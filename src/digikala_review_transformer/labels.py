"""Fixed class order, label columns per head/variant, encoding and class weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed order: the integer code of a class is its index here. Never rely on sklearn's classes_ order.
CLASSES = {
    "rec":  ["recommended", "not_recommended", "no_idea"],
    "band": ["low", "medium", "high"],
}
# (head, variant) -> label column. A = raw labels, B = step-1 corrected labels.
LABEL_COLS = {
    ("rec", "A"):  "recommendation_status",
    ("rec", "B"):  "step1_rec_label",
    ("band", "A"): "rating_band",
    ("band", "B"): "step1_rate_band_label",
}
RAW_COL  = {"rec": "recommendation_status", "band": "rating_band"}
CORR_COL = {"rec": "step1_rec_label", "band": "step1_rate_band_label"}


def label_map(head):
    """Return {class_name: int_code} for a head."""
    return {c: i for i, c in enumerate(CLASSES[head])}


def encode(series, head):
    """Map label strings to int codes (nullable Int8). Missing stays <NA>; unknown values raise."""
    out = series.map(label_map(head))
    bad = series.notna() & out.isna()
    if bad.any():
        raise ValueError(f"unknown labels for head '{head}': {series[bad].unique()[:10]}")
    return out.astype("Int8")


def decode(codes, head):
    """Map int codes back to class names."""
    names = CLASSES[head]
    return pd.Series(codes).map(lambda i: names[int(i)] if pd.notna(i) else pd.NA)


def class_weights(y, n_classes=3):
    """Balanced weights n / (k * n_c) as {code: weight}, computed from training labels only."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=n_classes)
    return {i: float(len(y) / (n_classes * c)) for i, c in enumerate(counts)}