"""CSV reading with NA rules, sanity checks and per-head training data."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import labels
from .utils import sha256_file

LABEL_COLUMNS = ["recommendation_status", "step1_rec_label", "rating_band", "step1_rate_band_label"]


def read_split(path):
    """Read a split CSV as strings (no default NA parsing, so bodies like 'None'/'NA' survive).
    Empty strings become <NA> only in label columns; rate becomes numeric; id stays str."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    for c in LABEL_COLUMNS:
        if c in df.columns:
            df[c] = df[c].where(df[c] != "", pd.NA)
    if "rate" in df.columns:
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    return df


def read_ids(path):
    """Read only the id column of a CSV as a string Series."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["id"], encoding="utf-8")["id"]


def check_splits(splits, pseudo_ids, natural_ids, manifest, csv_paths,
                 expected_rows, expected_rec_labeled, expected_band_missing):
    """Run all step-1 sanity checks. Returns a DataFrame(check, got, expected, ok).
    splits: {'train','validation','test'} -> DataFrame; csv_paths: {file_name: Path}."""
    rows = []

    def add(name, got, exp):
        rows.append({"check": name, "got": got, "expected": exp, "ok": bool(got == exp)})

    tr = splits["train"]
    for s, df in splits.items():
        add(f"{s}: rows", len(df), expected_rows[s])
        add(f"{s}: duplicate ids", int(df["id"].duplicated().sum()), 0)
        add(f"{s}: missing corrected band", int(df[labels.CORR_COL['band']].isna().sum()), expected_band_missing[s])
        add(f"{s}: raw vs corrected band missing differ",
            int((df[labels.RAW_COL['band']].isna() != df[labels.CORR_COL['band']].isna()).sum()), 0)
        add(f"{s}: raw vs corrected rec missing differ",
            int((df[labels.RAW_COL['rec']].isna() != df[labels.CORR_COL['rec']].isna()).sum()), 0)
        add(f"{s}: NULL body", int(df["body"].isna().sum()), 0)

    for a, b in combinations(splits, 2):
        add(f"shared ids {a}/{b}", int(splits[a]["id"].isin(splits[b]["id"]).sum()), 0)
        add(f"shared product_id {a}/{b}",
            len(set(splits[a]["product_id"]) & set(splits[b]["product_id"])), 0)

    add("train rec labeled (corrected)", int(tr[labels.CORR_COL['rec']].notna().sum()), expected_rec_labeled)
    add("train rec labeled (raw)", int(tr[labels.RAW_COL['rec']].notna().sum()), expected_rec_labeled)

    all_ids = pd.concat([d["id"] for d in splits.values()])
    add("pseudo ids present in train/val/test files", int(pseudo_ids.isin(all_ids).sum()), 0)
    add("natural ids not in train_supervised", int((~natural_ids.isin(tr["id"])).sum()), 0)
    add("natural ids that have a rec label", int(tr.loc[tr["id"].isin(natural_ids), labels.CORR_COL['rec']].notna().sum()), 0)
    add("natural count == train rows without rec label", len(natural_ids), int(tr[labels.CORR_COL['rec']].isna().sum()))

    for name, path in csv_paths.items():
        add(f"sha256 {name}", sha256_file(path), manifest["files"][name]["sha256"])
    return pd.DataFrame(rows)


def build_head_data(df, head, variant):
    """Training/eval data for one head+variant: rows with a valid label only.
    Returns DataFrame(pos, id, body, y); 'pos' is the row position in df (= row of the TF-IDF matrix)."""
    y = labels.encode(df[labels.LABEL_COLS[(head, variant)]], head)
    keep = y.notna().to_numpy()
    return pd.DataFrame({
        "pos": np.flatnonzero(keep),
        "id": df["id"].to_numpy()[keep],
        "body": df["body"].to_numpy()[keep],
        "y": y[keep].astype("int8").to_numpy(),
    })