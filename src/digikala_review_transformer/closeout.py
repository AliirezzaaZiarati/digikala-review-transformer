"""Phase close-out helpers: model details, registry/prediction checks, vectorizer fit-set check, config merge."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from .repro import _unwrap
from .utils import sha256_file, update_json


def library_versions():
    """Versions of the libraries that matter for loading the pickled artifacts."""
    return {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "scipy": scipy.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__}


def make_check(name, passed, evidence, status_if_false="FAIL"):
    """One checklist row. `status_if_false` lets soft checks report REVIEW instead of FAIL."""
    return {"check": name, "status": "PASS" if passed else status_if_false, "evidence": evidence}


def load_vectorizer(artifact_dir, model_tag):
    """Load the saved vectorizer from disk (same unwrap rule as the reproducibility test)."""
    path = Path(artifact_dir) / f"vectorizer_{model_tag}.joblib"
    return _unwrap(joblib.load(path), "transform", ("vectorizer", "vec"))


def model_details(artifact_dir, model_ids):
    """Load each saved LR model and report C, iterations, feature count and classes."""
    rows = []
    for model_id in model_ids:
        model = _unwrap(joblib.load(Path(artifact_dir) / f"{model_id}.joblib"),
                        "predict_proba", ("model", "clf", "estimator"))
        _, head, variant = model_id.rsplit("_", 2)       # e.g. tfidf12_rec_B -> rec, B
        n_iter_max = int(np.max(model.n_iter_))
        rows.append({
            "model_id": model_id, "head": head, "variant": variant,
            "C": float(model.C), "max_iter": int(model.max_iter), "n_iter_max": n_iter_max,
            "converged": bool(n_iter_max < int(model.max_iter)),   # lbfgs stops at max_iter if not converged
            "n_features_in": int(model.n_features_in_),
            "classes": [int(c) for c in model.classes_],
        })
    return pd.DataFrame(rows)


def check_registry(registry_csv, model_ids, root, expected_columns):
    """Validate the model registry without assuming column names. Returns (summary dict, registry DataFrame)."""
    registry = pd.read_csv(registry_csv, dtype=str, keep_default_na=False, encoding="utf-8")
    problems = []
    if set(registry.columns) != set(expected_columns):
        problems.append(f"columns differ: {list(registry.columns)} vs {list(expected_columns)}")
    if len(registry) != len(model_ids):
        problems.append(f"expected {len(model_ids)} rows, found {len(registry)}")

    # The id column is the only column that contains all model ids
    id_columns = [c for c in registry.columns if set(model_ids) <= set(registry[c])]
    # The path column is the only column whose values all end with .joblib
    path_columns = [c for c in registry.columns
                    if len(registry) > 0 and registry[c].str.endswith(".joblib").all()]
    if len(id_columns) != 1:
        problems.append(f"id column not unique: {id_columns}")
    if len(path_columns) != 1:
        problems.append(f"path column not unique: {path_columns}")
    id_column = id_columns[0] if len(id_columns) == 1 else None
    path_column = path_columns[0] if len(path_columns) == 1 else None

    if id_column and path_column:
        if registry[id_column].duplicated().any():
            problems.append("duplicate model ids")
        for _, row in registry.iterrows():
            artifact = Path(root) / row[path_column]
            if not artifact.exists():
                problems.append(f"missing artifact file: {artifact}")
            if Path(row[path_column]).stem != row[id_column]:
                problems.append(f"path stem != model id for {row[id_column]}")
    return {"ok": not problems, "problems": problems,
            "id_column": id_column, "path_column": path_column, "n_rows": len(registry)}, registry


def check_prediction_ids(pred_dir, model_ids, splits, split_csv_paths):
    """Compare the id column (value and order) of every prediction parquet with its split CSV."""
    rows = []
    for split in splits:
        csv_ids = pd.read_csv(split_csv_paths[split], usecols=["id"], dtype=str,
                              keep_default_na=False, encoding="utf-8")["id"].tolist()
        for model_id in model_ids:
            saved_ids = (pd.read_parquet(Path(pred_dir) / f"{model_id}__{split}.parquet", columns=["id"])["id"]
                         .astype(str).tolist())
            rows.append({"model_id": model_id, "split": split, "n_csv": len(csv_ids),
                         "n_parquet": len(saved_ids), "same_order": saved_ids == csv_ids})
    return pd.DataFrame(rows)


def check_vectorizer_fit_set(vectorizer, train_bodies):
    """Rebuild idf from document frequencies of `train_bodies` and compare with the saved `idf_`.
    A vectorizer fitted on a different document set would give a clearly different idf."""
    if not (vectorizer.use_idf and vectorizer.smooth_idf):
        raise ValueError("check assumes use_idf=True and smooth_idf=True")
    matrix = vectorizer.transform(train_bodies)                       # CSR, one row per document
    n_docs = matrix.shape[0]
    # each (row, term) pair appears once in .indices, so bincount = number of documents containing the term
    doc_freq = np.bincount(matrix.indices, minlength=matrix.shape[1])
    expected_idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0    # sklearn smooth idf formula
    abs_diff = np.abs(expected_idf - np.asarray(vectorizer.idf_, dtype=np.float64))
    return {"n_docs": int(n_docs), "n_features": int(matrix.shape[1]),
            "min_doc_freq": int(doc_freq.min()), "max_abs_idf_diff": float(abs_diff.max())}


def build_artifact_table(named_paths, root):
    """Rows: kind, path relative to root, size (MB), sha256. `named_paths` = list of (kind, path)."""
    rows = []
    for kind, path in named_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({"kind": kind,
                     "relative_path": path.resolve().relative_to(Path(root).resolve()).as_posix(),
                     "size_mb": round(path.stat().st_size / 1e6, 2),
                     "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def merge_new_config_keys(config_path, additions, overwrite_keys=()):
    """Add only keys absent from config.json (existing keys stay). Keys in `overwrite_keys` are always written.
    Returns which keys were written / unchanged / in conflict (existing value differs)."""
    config_path = Path(config_path)
    existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    # JSON round trip so tuples/Paths compare the same way as after saving
    additions = json.loads(json.dumps(additions, ensure_ascii=False, default=str))
    to_write, unchanged, conflicts = {}, [], []
    for key, value in additions.items():
        if key in overwrite_keys or key not in existing:
            to_write[key] = value
        elif existing[key] == value:
            unchanged.append(key)
        else:
            conflicts.append(key)
    if to_write:
        update_json(config_path, to_write)
    final_config = json.loads(config_path.read_text(encoding="utf-8"))
    return {"written": sorted(to_write), "unchanged": sorted(unchanged),
            "conflicts": sorted(conflicts), "all_keys": sorted(final_config)}