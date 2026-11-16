"""Step 2 (label-noise weighting) helpers.

Cell 3 functions: load the training table and the zero-feature flags.
More functions are added in later cells (folds, OOF assembly, weights, review sample).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from digikala_review_transformer import data_io, labels


def load_training_table(train_csv, heads, id_col="id", product_col="product_id", rate_col="rate"):
    """Read train_supervised once (via data_io.read_split) and drop the body.

    Returns a DataFrame with id, product_id, rate and one nullable Int8 column
    y_<head> per head (corrected-label codes; <NA> where the head has no label).
    Row i of this table is row i of the cached TF-IDF matrix.
    """
    raw = data_io.read_split(train_csv).reset_index(drop=True)

    # Required columns must exist (corrected label columns come from labels.CORR_COL)
    corrected_cols = {head: labels.CORR_COL[head] for head in heads}
    needed = [id_col, product_col, rate_col] + list(corrected_cols.values())
    missing = [col for col in needed if col not in raw.columns]
    assert not missing, f"columns missing in train csv: {missing}"

    # product_id is the fold group: it must never be missing or empty
    assert raw[product_col].notna().all(), "product_id has missing values"
    assert (raw[product_col].astype(str).str.len() > 0).all(), "product_id has empty values"

    table = pd.DataFrame({
        id_col: raw[id_col].astype(str).to_numpy(),
        product_col: raw[product_col].astype(str).to_numpy(),
        # rate is for analysis only; NaN stays NaN
        rate_col: pd.to_numeric(raw[rate_col], errors="coerce").to_numpy(dtype="float64", na_value=np.nan),
    })

    # Encode corrected labels; class order must match labels.CLASSES (code = list position)
    for head, column in corrected_cols.items():
        expected_map = {name: code for code, name in enumerate(labels.CLASSES[head])}
        assert labels.label_map(head) == expected_map, f"{head}: label_map differs from CLASSES order"
        table[f"y_{head}"] = labels.encode(raw[column], head)
    return table


def zero_feature_flags(x_npz, ids_parquet, expected_ids, id_col="id"):
    """Flag rows with no TF-IDF feature in the cached train matrix.

    Asserts that the cached id order equals expected_ids.
    The matrix is loaded only to count non-zeros per row, then released.
    Returns (boolean array aligned with expected_ids, info dict).
    """
    cache_ids = pd.read_parquet(ids_parquet)[id_col].astype(str).to_numpy()
    assert np.array_equal(cache_ids, np.asarray(expected_ids)), "cache id order differs from train table"

    matrix = sparse.load_npz(x_npz).tocsr()
    assert matrix.shape[0] == len(expected_ids), "matrix rows differ from id count"
    row_nnz = np.diff(matrix.indptr)
    info = {"n_rows": int(matrix.shape[0]), "n_features": int(matrix.shape[1]),
            "nnz": int(matrix.nnz), "dtype": str(matrix.dtype)}
    return row_nnz == 0, info


def build_load_checks(table, heads, expected_rows, expected_labeled, expected_zero_feature,
                      id_col="id", zero_col="n_features_zero"):
    """Return a check table (check, got, expected, ok) for the loaded training table."""
    rows = []

    def add_check(name, got, expected):
        rows.append({"check": name, "got": got, "expected": expected, "ok": bool(got == expected)})

    add_check("train rows", len(table), int(expected_rows))
    add_check("unique ids", int(table[id_col].nunique()), len(table))
    for head in heads:
        add_check(f"labeled rows ({head})", int(table[f"y_{head}"].notna().sum()),
                  int(expected_labeled[head]))
    add_check("zero-feature rows", int(table[zero_col].sum()), int(expected_zero_feature))
    return pd.DataFrame(rows)


def build_group_folds(table, head, n_folds, seed, shuffle=True, product_col="product_id"):
    """Assign each labeled row of one head to a fold with StratifiedGroupKFold.

    Groups are products (a product never spans two folds); strata are the corrected
    labels of the same head. Rows without a label for this head get <NA>.
    Returns a nullable Int8 Series aligned with table rows.
    """
    y_column = f"y_{head}"
    labeled_positions = np.flatnonzero(table[y_column].notna().to_numpy())
    y_labeled = table[y_column].to_numpy(dtype="int8", na_value=-1)[labeled_positions]
    # factorize products to integer group codes (faster than string groups)
    group_codes, _ = pd.factorize(table[product_col].to_numpy()[labeled_positions])

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    fold_of_labeled = np.full(len(labeled_positions), -1, dtype="int8")
    # X is not used by the splitter; a dummy column keeps memory small
    dummy_x = np.zeros((len(labeled_positions), 1), dtype="int8")
    for fold_index, (_, held_out_idx) in enumerate(splitter.split(dummy_x, y_labeled, group_codes)):
        fold_of_labeled[held_out_idx] = fold_index
    assert (fold_of_labeled >= 0).all(), f"{head}: some labeled rows were not assigned to a fold"

    folds = pd.Series(pd.NA, index=range(len(table)), dtype="Int8")
    folds.iloc[labeled_positions] = fold_of_labeled
    return folds


def check_folds(table, folds, head, n_folds, product_col="product_id"):
    """Return a check table (check, got, expected, ok) for the folds of one head."""
    y_column = f"y_{head}"
    labeled = table[y_column].notna().to_numpy()
    has_fold = folds.notna().to_numpy()
    rows = []

    def add_check(name, got, expected):
        rows.append({"head": head, "check": name, "got": got, "expected": expected,
                     "ok": bool(got == expected)})

    add_check("labeled rows with a fold", int((labeled & has_fold).sum()), int(labeled.sum()))
    add_check("unlabeled rows with a fold", int((~labeled & has_fold).sum()), 0)
    add_check("fold ids used", sorted(folds.dropna().astype(int).unique().tolist()),
              list(range(n_folds)))

    # a product must live in exactly one fold
    pairs = pd.DataFrame({
        "product": table[product_col].to_numpy()[has_fold],
        "fold": folds[has_fold].astype("int8").to_numpy(),
    })
    spanning_products = int((pairs.groupby("product")["fold"].nunique() > 1).sum())
    add_check("products in more than one fold", spanning_products, 0)

    # every training complement (4 folds) must contain every class
    n_classes = len(labels.CLASSES[head])
    class_by_fold = pd.crosstab(folds[has_fold].astype("int8").to_numpy(),
                                table[y_column][has_fold].astype("int8").to_numpy())
    class_by_fold = class_by_fold.reindex(columns=range(n_classes), fill_value=0)
    train_counts = class_by_fold.sum(axis=0).to_numpy()[None, :] - class_by_fold.to_numpy()
    add_check("(fold, class) cells with an empty training side", int((train_counts <= 0).sum()), 0)
    add_check("(fold, class) cells with no held-out row", int((class_by_fold.to_numpy() == 0).sum()), 0)
    return pd.DataFrame(rows)


def fold_balance_table(table, folds, head, product_col="product_id"):
    """Per-fold size, product count and class distribution (plus an 'all' row)."""
    class_names = labels.CLASSES[head]
    has_fold = folds.notna().to_numpy()
    frame = pd.DataFrame({
        "fold": folds[has_fold].astype("int8").to_numpy(),
        "y": table[f"y_{head}"][has_fold].astype("int8").to_numpy(),
        "product": table[product_col].to_numpy()[has_fold],
    })
    class_counts = pd.crosstab(frame["fold"], frame["y"]).reindex(
        columns=range(len(class_names)), fill_value=0)
    class_counts.columns = list(class_names)

    result = pd.DataFrame({
        "head": head,
        "fold": class_counts.index.astype(str),
        "n_rows": class_counts.sum(axis=1).to_numpy(),
        "n_products": frame.groupby("fold")["product"].nunique().reindex(class_counts.index).to_numpy(),
    })
    total_row = {"head": head, "fold": "all", "n_rows": len(frame),
                 "n_products": int(frame["product"].nunique())}
    for name in class_names:
        result[f"n_{name}"] = class_counts[name].to_numpy()
        total_row[f"n_{name}"] = int(class_counts[name].sum())
    result = pd.concat([result, pd.DataFrame([total_row])], ignore_index=True)

    result["row_share_pct"] = 100 * result["n_rows"] / len(frame)
    for name in class_names:
        result[f"{name}_pct"] = 100 * result[f"n_{name}"] / result["n_rows"]
    return result


def make_folds_frame(table, fold_by_head, heads, id_col="id"):
    """Build the step2_folds table: id, fold_<head> and y_<head> (corrected codes) per head."""
    frame = pd.DataFrame({id_col: table[id_col].to_numpy()})
    for head in heads:
        frame[f"fold_{head}"] = fold_by_head[head].to_numpy()
        frame[f"y_{head}"] = table[f"y_{head}"].to_numpy()
    return frame


def save_folds_once(folds_frame, path):
    """Write the folds parquet once. If it exists, assert the new folds are identical.

    The OOF files depend on these folds, so they must never change silently.
    """
    path = Path(path)
    if path.exists():
        existing = pd.read_parquet(path)
        assert existing.equals(folds_frame), (
            f"{path} exists and differs from the new folds; "
            "delete it deliberately only if the OOF files are also rebuilt")
        return "exists_identical"
    temporary = path.with_suffix(".tmp")
    folds_frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return "written"


def load_fold_meta(oof_dir, heads, n_folds):
    """Read the meta json of every (head, fold) into one table (asserts all files exist)."""
    rows = []
    for head in heads:
        for fold in range(n_folds):
            meta_path = Path(oof_dir) / f"oof_{head}_fold{fold}_meta.json"
            assert meta_path.exists(), f"missing meta file: {meta_path.name}"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append({
                "head": head, "fold": fold,
                "fit_seconds": meta["fit_seconds"], "n_iter": meta["n_iter"],
                "converged": meta["converged"], "n_train_rows": meta["n_train_rows"],
                "n_pred_rows": meta["n_pred_rows"], "peak_rss_mb": meta["peak_rss_mb"],
                "classes_": meta["classes_"],
                "recipe": json.dumps(meta["recipe"], sort_keys=True),
            })
    return pd.DataFrame(rows)


def check_fold_meta(meta_frame, params_snapshot, labeled_counts):
    """Return a check table (head, check, got, expected, ok) for the fold meta files.

    Every fold must have been fitted with the recipe in the current params snapshot.
    """
    rows = []

    def add_check(head, name, got, expected):
        rows.append({"head": head, "check": name, "got": got, "expected": expected,
                     "ok": bool(got == expected)})

    for head, group in meta_frame.groupby("head", sort=False):
        head_settings = params_snapshot["heads"][head]
        lr_settings = params_snapshot["lr"]
        recipes = [json.loads(text) for text in group["recipe"]]
        add_check(head, "distinct recipes", int(group["recipe"].nunique()), 1)
        add_check(head, "recipe C", recipes[0]["C"], head_settings["C"])
        add_check(head, "recipe label column", recipes[0]["label_col"], head_settings["label_col"])
        for key in ("class_weight", "solver", "tol", "max_iter"):
            add_check(head, f"recipe {key}", recipes[0][key], lr_settings[key])
        add_check(head, "recipe folds sha256", recipes[0]["folds_sha256"], params_snapshot["folds_sha256"])
        add_check(head, "classes_ in every fold", int(sum(c == [0, 1, 2] for c in group["classes_"])), len(group))
        add_check(head, "held-out rows total", int(group["n_pred_rows"].sum()), int(labeled_counts[head]))
        add_check(head, "train + held-out rows per fold",
                  int(((group["n_train_rows"] + group["n_pred_rows"]) == labeled_counts[head]).sum()), len(group))
    return pd.DataFrame(rows)


def assemble_oof(table, folds, head, oof_dir, n_folds, id_col="id", prob_sum_tol=1e-5):
    """Merge the fold files of one head into one OOF table (labeled rows, table order).

    Asserts: every labeled row covered exactly once, ids aligned with the table, the fold
    of each row equals the fold assignment, probabilities finite, in [0,1] and summing to 1.
    Returns (DataFrame id, fold, p_<class>..., info dict).
    """
    class_names = labels.CLASSES[head]
    prob_columns = [f"p_{name}" for name in class_names]

    parts = []
    for fold in range(n_folds):
        part = pd.read_parquet(Path(oof_dir) / f"oof_{head}_fold{fold}.parquet")
        assert list(part.columns) == [id_col] + prob_columns, f"fold {fold}: unexpected columns {list(part.columns)}"
        part["fold"] = np.int8(fold)
        parts.append(part)
    oof = pd.concat(parts, ignore_index=True)
    oof[id_col] = oof[id_col].astype(str)

    # exactly-once coverage: same size, unique ids, and every expected id is present
    labeled = table[f"y_{head}"].notna().to_numpy()
    expected_ids = table[id_col].to_numpy()[labeled]
    assert oof[id_col].is_unique, f"{head}: duplicate ids across folds"
    assert len(oof) == len(expected_ids), f"{head}: OOF rows {len(oof)} != labeled rows {len(expected_ids)}"
    indexer = pd.Index(oof[id_col]).get_indexer(expected_ids)
    assert (indexer >= 0).all(), f"{head}: some labeled rows have no OOF probability"

    # reorder to table order and verify alignment
    oof = oof.iloc[indexer].reset_index(drop=True)
    assert np.array_equal(oof[id_col].to_numpy(), expected_ids), f"{head}: id order differs from table"

    # each row must come from the fold it was assigned to
    expected_fold = folds.to_numpy(dtype="int8", na_value=-1)[labeled]
    assert np.array_equal(oof["fold"].to_numpy(), expected_fold), f"{head}: fold of a row differs from assignment"

    # probability sanity
    probabilities = oof[prob_columns].to_numpy()
    assert np.isfinite(probabilities).all(), f"{head}: non-finite probabilities"
    assert probabilities.min() >= 0 and probabilities.max() <= 1 + prob_sum_tol, f"{head}: probability outside [0,1]"
    max_sum_deviation = float(np.abs(probabilities.sum(axis=1) - 1.0).max())
    assert max_sum_deviation <= prob_sum_tol, f"{head}: probabilities do not sum to 1 ({max_sum_deviation})"

    oof["fold"] = oof["fold"].astype("int8")
    info = {"head": head, "n_rows": len(oof), "max_abs_sum_deviation": max_sum_deviation,
            "min_prob": float(probabilities.min()), "max_prob": float(probabilities.max())}
    return oof, info


def save_parquet_once(frame, path):
    """Write a parquet once. If it exists, assert the new frame is identical (no silent change)."""
    path = Path(path)
    if path.exists():
        existing = pd.read_parquet(path)
        assert existing.equals(frame), f"{path.name} exists and differs from the new frame"
        return "exists_identical"
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return "written"


def reference_validation_scores(final_fit_csv, reference_model_ids):
    """Read the validation scores of the selected final fit of each reference model.

    Returns a DataFrame indexed by head with model_id, C, val_macro_f1 and val_accuracy.
    Asserts that exactly one selected row exists per reference model.
    """
    final_fit = pd.read_csv(final_fit_csv)
    # "selected" may be read as bool or as text, so compare as text
    selected = final_fit[final_fit["selected"].astype(str) == "True"]

    rows = {}
    for head, model_id in reference_model_ids.items():
        model_rows = selected[selected["model_id"] == model_id]
        assert len(model_rows) == 1, f"{model_id}: expected exactly one selected row, got {len(model_rows)}"
        row = model_rows.iloc[0]
        rows[head] = {"model_id": model_id, "C": float(row["C"]),
                      "val_macro_f1": float(row["val_macro_f1"]),
                      "val_accuracy": float(row["val_accuracy"])}
    return pd.DataFrame.from_dict(rows, orient="index")


def oof_sanity_table(table, oof, head, fold_meta, reference_f1, max_above, max_below, id_col="id"):
    """OOF Macro F1 and accuracy (corrected label) for all rows and for each fold.

    The 'all' row is compared with the reference validation Macro F1 and gets a status:
    OK / WARN_above (possible leakage) / WARN_below (possible bug). Fold rows get "-".
    Warnings are informational and never stop the run.
    """
    class_names = labels.CLASSES[head]
    class_codes = list(range(len(class_names)))
    prob_columns = [f"p_{name}" for name in class_names]

    # Align OOF rows with the labeled rows of the table (same order, same ids)
    labeled = table[f"y_{head}"].notna().to_numpy()
    assert np.array_equal(oof[id_col].to_numpy(), table[id_col].to_numpy()[labeled]), \
        f"{head}: OOF ids are not aligned with the table"
    y_true = table[f"y_{head}"].to_numpy(dtype="int8", na_value=-1)[labeled]
    assert (y_true >= 0).all(), f"{head}: unexpected missing label"

    # Predicted class = argmax of the OOF probabilities (computed in float64)
    y_pred = oof[prob_columns].to_numpy(dtype="float64").argmax(axis=1)
    fold_values = oof["fold"].to_numpy()
    head_meta = fold_meta[fold_meta["head"] == head].set_index("fold")
    assert sorted(head_meta.index.tolist()) == sorted(np.unique(fold_values).tolist()), \
        f"{head}: meta folds differ from OOF folds"

    rows = []
    for scope in ["all"] + sorted(np.unique(fold_values).tolist()):
        if scope == "all":
            mask, meta_part = np.ones(len(y_true), dtype=bool), head_meta
        else:
            mask, meta_part = fold_values == scope, head_meta.loc[[scope]]

        macro_f1 = f1_score(y_true[mask], y_pred[mask], labels=class_codes,
                            average="macro", zero_division=0)
        accuracy = accuracy_score(y_true[mask], y_pred[mask])
        difference = macro_f1 - reference_f1

        if scope == "all":
            status = ("WARN_above" if difference > max_above
                      else "WARN_below" if difference < -max_below else "OK")
        else:
            status = "-"
        rows.append({
            "head": head, "scope": str(scope), "n_rows": int(mask.sum()),
            "macro_f1": macro_f1, "accuracy": accuracy,
            "reference_val_macro_f1": reference_f1, "f1_minus_reference": difference,
            "status": status,
            "n_iter": int(meta_part["n_iter"].max()),
            "converged": bool(meta_part["converged"].all()),
            "fit_seconds": round(float(meta_part["fit_seconds"].sum()), 1),
        })
    return pd.DataFrame(rows)

def normalized_entropy(probabilities, eps, n_classes):
    """Normalized Shannon entropy H_norm in [0, 1] for every row.

    Probabilities are clipped from below by eps ONLY inside the log;
    they are not renormalized.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    log_term = np.log(np.clip(probabilities, eps, None))
    return -(probabilities * log_term).sum(axis=1) / np.log(n_classes)


def compute_head_weights(table, oof, head, w_min, eps, zero_policy, prob_tol,
                         id_col="id", zero_col="n_features_zero"):
    """Compute p_y, Q, C and w for one head from its OOF probabilities.

    Formula:  Q = min(1, p_y / m_c),  C = 1 - H_norm,  w = max(w_min, 1 - C * (1 - Q))
    m_c is the mean OOF p_y of the recorded class over ALL OOF rows (not per fold),
    excluding zero-feature rows. Zero-feature rows get w = 1 (zero_policy 'weight_one').

    Returns (arrays, info). Each array has len(table) entries (float32), NaN where
    the head has no label. info holds class means and diagnostics for the report.
    """
    assert zero_policy == "weight_one", f"unknown zero-feature policy: {zero_policy}"
    class_names = labels.CLASSES[head]
    n_classes = len(class_names)
    prob_columns = [f"p_{name}" for name in class_names]

    # ── Align OOF rows with the labeled rows of the table ──
    labeled = table[f"y_{head}"].notna().to_numpy()
    assert np.array_equal(oof[id_col].to_numpy(), table[id_col].to_numpy()[labeled]), \
        f"{head}: OOF ids are not aligned with the table"
    y = table[f"y_{head}"].to_numpy(dtype="int8", na_value=-1)[labeled].astype("int64")
    assert y.min() >= 0 and y.max() < n_classes, f"{head}: label code out of range"
    zero_rows = table[zero_col].to_numpy()[labeled]

    # ── Probabilities in float64 (the OOF files hold float32) ──
    probabilities = oof[prob_columns].to_numpy(dtype="float64")
    assert probabilities.shape == (len(y), n_classes), f"{head}: unexpected probability shape"
    p_y = probabilities[np.arange(len(y)), y]          # probability of the recorded label

    # ── Class means m_c over all OOF rows, zero-feature rows excluded ──
    included = ~zero_rows
    class_means = np.zeros(n_classes)
    class_rows = np.zeros(n_classes, dtype="int64")
    for code in range(n_classes):
        selected = included & (y == code)
        assert selected.any(), f"{head}: class {code} has no rows for m_c"
        class_means[code] = p_y[selected].mean()
        class_rows[code] = int(selected.sum())
    assert np.all(class_means > 0), f"{head}: non-positive class mean"

    # ── Q: class-normalized agreement between label and model ──
    q = np.minimum(1.0, p_y / class_means[y])

    # ── C: model confidence from the full distribution (1 = decisive, 0 = confused) ──
    h_norm = normalized_entropy(probabilities, eps, n_classes)
    c_raw = 1.0 - h_norm
    assert c_raw.min() >= -prob_tol and c_raw.max() <= 1.0 + prob_tol, \
        f"{head}: C outside [0,1] by more than {prob_tol}"
    c = np.clip(c_raw, 0.0, 1.0)                        # numerical guard only

    # ── Weight, then the zero-feature policy ──
    w = np.maximum(w_min, 1.0 - c * (1.0 - q))
    w[zero_rows] = 1.0

    # ── Scatter back to full table length (NaN where the head has no label) ──
    def to_full(values):
        full = np.full(len(table), np.nan, dtype="float32")
        full[labeled] = values.astype("float32")
        return full

    arrays = {"p_y": to_full(p_y), "q": to_full(q), "c": to_full(c), "w": to_full(w)}
    info = {
        "head": head, "class_names": list(class_names),
        "class_means": class_means.tolist(), "class_rows_used": class_rows.tolist(),
        "n_labeled": len(y), "n_zero_feature_labeled": int(zero_rows.sum()),
        "n_probs_below_eps": int((probabilities < eps).sum()),
        "c_raw_min": float(c_raw.min()), "c_raw_max": float(c_raw.max()),
        "share_q_capped_at_1": float((q[included] >= 1.0).mean()),
        "share_w_at_floor": float((w[included] <= w_min + 1e-12).mean()),
        "mean_w": float(w.mean()),
    }
    return arrays, info


def build_weights_frame(table, arrays_by_head, heads, id_col="id", zero_col="n_features_zero"):
    """Assemble the weights table: id, zero-feature flag, then p_y/q/c/w per head."""
    frame = pd.DataFrame({id_col: table[id_col].to_numpy(),
                          zero_col: table[zero_col].to_numpy()})
    for head in heads:
        for name in ("p_y", "q", "c", "w"):
            frame[f"{name}_{head}"] = arrays_by_head[head][name]
    return frame


def check_weights(frame, table, heads, w_min, id_col="id", zero_col="n_features_zero", tol=1e-6):
    """Return a check table (head, check, got, expected, ok) for the weights frame."""
    rows = []

    def add_check(head, name, got, expected):
        rows.append({"head": head, "check": name, "got": got, "expected": expected,
                     "ok": bool(got == expected)})

    add_check("all", "row count", len(frame), len(table))
    add_check("all", "ids aligned with table",
              bool(np.array_equal(frame[id_col].to_numpy(), table[id_col].to_numpy())), True)
    add_check("all", "zero-feature rows", int(frame[zero_col].sum()), int(table[zero_col].sum()))

    for head in heads:
        labeled = table[f"y_{head}"].notna().to_numpy()
        zero_labeled = frame[zero_col].to_numpy() & labeled
        for name in ("p_y", "q", "c", "w"):
            values = frame[f"{name}_{head}"].to_numpy(dtype="float64")
            # NaN exactly where the head has no label, and no inf anywhere
            add_check(head, f"{name}: NaN exactly where label is missing",
                      bool(np.array_equal(np.isnan(values), ~labeled)), True)
            add_check(head, f"{name}: no inf", bool(not np.isinf(values).any()), True)
            if name in ("p_y", "q", "c"):
                add_check(head, f"{name} within [0, 1]",
                          bool(np.nanmin(values) >= -tol and np.nanmax(values) <= 1 + tol), True)
        weights = frame[f"w_{head}"].to_numpy(dtype="float64")
        add_check(head, "w >= W_MIN", bool(np.nanmin(weights) >= w_min - tol), True)
        add_check(head, "w <= 1", bool(np.nanmax(weights) <= 1 + tol), True)
        add_check(head, "zero-feature labeled rows with w == 1",
                  int((weights[zero_labeled] == 1.0).sum()), int(zero_labeled.sum()))
    return pd.DataFrame(rows)


def verify_weights_by_hand(table, oof, weights_frame, head, class_means, w_min, eps,
                           n_sample, seed, id_col="id", zero_col="n_features_zero"):
    """Recompute w for random rows with plain python math; return the max abs difference.

    This is an independent check of the vectorized formula (no numpy broadcasting,
    one row at a time, using the stored float32 OOF probabilities).
    """
    class_names = labels.CLASSES[head]
    n_classes = len(class_names)
    prob_columns = [f"p_{name}" for name in class_names]

    labeled_positions = np.flatnonzero(table[f"y_{head}"].notna().to_numpy())
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(labeled_positions), size=min(n_sample, len(labeled_positions)),
                       replace=False)

    probabilities = oof[prob_columns].to_numpy()
    y = table[f"y_{head}"].to_numpy(dtype="int8", na_value=-1)
    stored_w = weights_frame[f"w_{head}"].to_numpy()
    zero_flags = table[zero_col].to_numpy()

    max_difference = 0.0
    for pick in picks:
        position = labeled_positions[pick]
        assert oof[id_col].iat[pick] == table[id_col].iat[position], "id mismatch in hand check"
        p = [float(value) for value in probabilities[pick]]
        if zero_flags[position]:
            expected = 1.0
        else:
            label = int(y[position])
            q = min(1.0, p[label] / class_means[label])
            h_norm = -sum(pk * math.log(max(pk, eps)) for pk in p) / math.log(n_classes)
            c = min(1.0, max(0.0, 1.0 - h_norm))
            expected = max(w_min, 1.0 - c * (1.0 - q))
        max_difference = max(max_difference, abs(expected - float(stored_w[position])))
    return max_difference

# ── Step 2 / Cell 9: weight summary tables and acceptance status ──

def head_weight_view(table, weights, head, class_names,
                     id_col="id", zero_col="n_features_zero", rate_col="rate"):
    """Labeled rows of one head: class name, rate group, w, and (zero-feature masked) p_y, q, c."""
    # Both frames must describe the same rows in the same order
    if not table[id_col].equals(weights[id_col]):
        raise AssertionError("train table and weights frame are not row-aligned by id")

    label_present = table[f"y_{head}"].notna().to_numpy()
    weight_present = weights[f"w_{head}"].notna().to_numpy()
    if not np.array_equal(label_present, weight_present):
        raise AssertionError(f"{head}: rows with a label and rows with a weight differ")

    class_codes = table.loc[label_present, f"y_{head}"].astype("int64").to_numpy()
    is_zero_feature = table.loc[label_present, zero_col].to_numpy()
    rate = table.loc[label_present, rate_col].to_numpy(dtype="float64")
    rate_group = np.where(rate == 0, "rate==0", np.where(rate > 0, "rate>0", "rate_missing"))

    def masked(prefix):
        # p_y, q, c of zero-feature rows are not meaningful, so they are excluded from their means
        values = weights.loc[label_present, f"{prefix}_{head}"].to_numpy(dtype="float64")
        return np.where(is_zero_feature, np.nan, values)

    return pd.DataFrame({
        "class_name": np.asarray(class_names)[class_codes],
        "rate_group": rate_group,
        "w": weights.loc[label_present, f"w_{head}"].to_numpy(dtype="float64"),
        "p_y": masked("p_y"),
        "q": masked("q"),
        "c": masked("c"),
    })


def summarize_weights(subset, low_cutoff, half_cutoff):
    """Descriptive statistics of one group of rows."""
    w = subset["w"]
    return {
        "count": len(subset),
        "mean_w": w.mean(),
        "median_w": w.median(),
        "share_w_le_low": (w <= low_cutoff).mean(),
        "share_w_lt_half": (w < half_cutoff).mean(),
        "mean_p_y": subset["p_y"].mean(),
        "mean_q": subset["q"].mean(),
        "mean_c": subset["c"].mean(),
    }


def weight_summary_table(views_by_head, class_names_by_head, low_cutoff, half_cutoff,
                         rate_split_heads=("band",)):
    """Head x class summary, plus a rate==0 / rate>0 split for the heads in rate_split_heads."""
    rows = []

    def add(head, split, group, subset):
        rows.append({"head": head, "split": split, "group": group,
                     **summarize_weights(subset, low_cutoff, half_cutoff)})

    for head, view in views_by_head.items():
        add(head, "all", "ALL", view)
        for name in class_names_by_head[head]:
            add(head, "class", name, view[view["class_name"] == name])
        if head in rate_split_heads:
            for rate_group in ("rate==0", "rate>0"):
                in_group = view[view["rate_group"] == rate_group]
                add(head, "rate", rate_group, in_group)
                for name in class_names_by_head[head]:
                    add(head, "rate_x_class", f"{rate_group} | {name}",
                        in_group[in_group["class_name"] == name])
    return pd.DataFrame(rows)


def weight_quantile_table(views_by_head, class_names_by_head, quantiles):
    """Quantiles of w per head, for all rows and for each class."""
    rows = []
    for head, view in views_by_head.items():
        groups = [("ALL", view)] + [(name, view[view["class_name"] == name])
                                    for name in class_names_by_head[head]]
        for group, subset in groups:
            values = subset["w"].quantile(list(quantiles))
            row = {"head": head, "group": group, "min": subset["w"].min()}
            row.update({f"q{round(p * 100):02d}": v for p, v in zip(quantiles, values)})
            row["max"] = subset["w"].max()
            rows.append(row)
    return pd.DataFrame(rows)


def acceptance_status_table(views_by_head, ambiguous_class_by_head, mean_weight_min,
                            low_cutoff, low_share_max, gap_max):
    """OK / WARN per engineering threshold. Never raises on a quality miss."""
    rows = []
    for head, view in views_by_head.items():
        w = view["w"]
        mean_w = float(w.mean())
        low_share = float((w <= low_cutoff).mean())

        is_ambiguous = (view["class_name"] == ambiguous_class_by_head[head]).to_numpy()
        if not is_ambiguous.any():
            raise ValueError(f"{head}: ambiguous class not found in data")
        # signed gap: mean w of the ambiguous class minus mean w of all other classes pooled
        gap = float(w[is_ambiguous].mean() - w[~is_ambiguous].mean())

        rows.append({"head": head, "check": "mean_weight", "value": mean_w,
                     "threshold": mean_weight_min, "rule": "mean_w >= MEAN_WEIGHT_MIN",
                     "status": "OK" if mean_w >= mean_weight_min else "WARN"})
        rows.append({"head": head, "check": f"share_w_le_{low_cutoff}", "value": low_share,
                     "threshold": low_share_max, "rule": "share <= LOW_WEIGHT_SHARE_MAX",
                     "status": "OK" if low_share <= low_share_max else "WARN"})
        rows.append({"head": head, "check": f"ambiguous_gap_{ambiguous_class_by_head[head]}",
                     "value": gap, "threshold": gap_max, "rule": "|gap| <= AMBIGUOUS_CLASS_MEAN_GAP_MAX",
                     "status": "OK" if abs(gap) <= gap_max else "WARN"})
    return pd.DataFrame(rows)


def save_table_once(table, path, float_format="%.6f"):
    """Write a CSV once. Re-running with identical content is a no-op; different content raises."""
    path = Path(path)
    new_text = table.to_csv(index=False, float_format=float_format)
    if path.exists():
        if path.read_text(encoding="utf-8") == new_text:
            return "identical"
        raise FileExistsError(f"{path} already exists with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return "written"



# ── Step 2 / Cell 10: manual review sample ──

def attach_body(pool, bodies, id_col="id", body_col="body"):
    """Add the review text to the pool by id. Every pool id must exist in the text table."""
    if not bodies[id_col].is_unique:
        raise AssertionError("duplicate ids in the text table")
    body_by_id = bodies.set_index(id_col)[body_col]
    missing = ~pool[id_col].isin(body_by_id.index)
    if missing.any():
        raise AssertionError(f"{int(missing.sum())} pool ids are missing from the text table")
    pool = pool.copy()
    pool[body_col] = body_by_id.reindex(pool[id_col]).to_numpy()
    return pool


def eligible_review_pool(pool, seed, zero_col="n_features_zero", body_col="body"):
    """Drop zero-feature rows and blank bodies, then keep one random row per unique body."""
    body_text = pool[body_col].fillna("").astype(str).str.strip()
    keep = (~pool[zero_col].to_numpy(dtype=bool)) & (body_text != "").to_numpy()
    shuffled = pool[keep].sample(frac=1.0, random_state=seed)
    unique_body = shuffled.drop_duplicates(subset=body_col).reset_index(drop=True)
    counts = {"pool_rows": len(pool),
              "after_zero_and_blank_filters": int(keep.sum()),
              "unique_bodies": len(unique_body)}
    return unique_body, counts


def assign_review_stratum(pool, high_min_w, low_max_w):
    """Add w_min (NaN-ignoring minimum over both heads) and the stratum label."""
    w_min = np.fmin(pool["w_rec"].to_numpy(dtype="float64"),
                    pool["w_band"].to_numpy(dtype="float64"))
    if np.isnan(w_min).any():
        raise AssertionError("rows without any weight found in the review pool")
    stratum = np.where(w_min < low_max_w, "low", np.where(w_min >= high_min_w, "high", "mid"))
    pool = pool.copy()
    pool["w_min"] = w_min
    pool["stratum"] = stratum
    return pool


def draw_review_rows(pool, quotas, low_max_w, min_rate_zero, seed):
    """Positional indices of the review sample.

    Order of draws: (1) low stratum: half of the quota with w_rec < cutoff, half with
    w_band < cutoff; (2) top up rate==0 rows from the high/mid strata if fewer than
    min_rate_zero were drawn; (3) fill high and mid to their quotas.
    """
    rng = np.random.default_rng(seed)
    stratum = pool["stratum"].to_numpy()
    is_rate_zero = pool["rate"].to_numpy(dtype="float64") == 0
    rec_is_low = pool["w_rec"].to_numpy(dtype="float64") < low_max_w     # NaN compares False
    band_is_low = pool["w_band"].to_numpy(dtype="float64") < low_max_w
    chosen = np.zeros(len(pool), dtype=bool)

    def draw(mask, n):
        candidates = np.flatnonzero(mask & ~chosen)
        if len(candidates) < n:
            raise ValueError(f"only {len(candidates)} candidates for a draw of {n}")
        chosen[rng.choice(candidates, size=n, replace=False)] = True

    low_per_head = quotas["low"] // 2
    draw(rec_is_low, low_per_head)
    draw(band_is_low, quotas["low"] - low_per_head)

    rate_zero_missing = max(0, min_rate_zero - int((chosen & is_rate_zero).sum()))
    if rate_zero_missing:
        draw(is_rate_zero & (stratum != "low"), rate_zero_missing)

    for name in ("high", "mid"):
        still_needed = quotas[name] - int((chosen & (stratum == name)).sum())
        if still_needed < 0:
            raise AssertionError(f"stratum {name} exceeds its quota before filling")
        draw(stratum == name, still_needed)
    return np.flatnonzero(chosen)


def _label_names(codes, class_names):
    """Class names for nullable integer codes; empty string where the label is missing."""
    values = codes.astype("float64").to_numpy()
    names = np.full(len(values), "", dtype=object)
    present = ~np.isnan(values)
    names[present] = np.asarray(class_names, dtype=object)[values[present].astype(int)]
    return names


def build_review_frame(sample, class_names_by_head, id_col="id", body_col="body", rate_col="rate"):
    """Review file layout: text, corrected labels, all weight ingredients, empty verdict columns."""
    review = pd.DataFrame({
        id_col: sample[id_col].to_numpy(),
        body_col: sample[body_col].to_numpy(),
        "rec_label": _label_names(sample["y_rec"], class_names_by_head["rec"]),
        "band_label": _label_names(sample["y_band"], class_names_by_head["band"]),
        rate_col: sample[rate_col].to_numpy(dtype="float64"),
    })
    for column in ("p_y_rec", "q_rec", "c_rec", "w_rec", "p_y_band", "q_band", "c_band", "w_band"):
        review[column] = sample[column].to_numpy(dtype="float64")
    review["stratum"] = sample["stratum"].to_numpy()
    review["verdict_rec"] = ""
    review["verdict_band"] = ""
    return review


def check_review_sample(sample, quotas, high_min_w, low_max_w, min_rate_zero, low_per_head,
                        id_col="id", body_col="body", zero_col="n_features_zero"):
    """Integrity checks of the drawn sample. Returns a table (name, passed)."""
    by_stratum = lambda name: sample[sample["stratum"] == name]
    checks = [
        ("row count", len(sample) == sum(quotas.values())),
        ("unique ids", bool(sample[id_col].is_unique)),
        ("unique bodies", bool(sample[body_col].is_unique)),
        ("stratum quotas", sample["stratum"].value_counts().to_dict() == dict(quotas)),
        ("rate==0 rows >= minimum", int((sample["rate"] == 0).sum()) >= min_rate_zero),
        ("rec rows with w_rec < cutoff >= per-head minimum",
         int((sample["w_rec"] < low_max_w).sum()) >= low_per_head),
        ("band rows with w_band < cutoff >= per-head minimum",
         int((sample["w_band"] < low_max_w).sum()) >= low_per_head),
        ("low stratum: w_min < cutoff", bool((by_stratum("low")["w_min"] < low_max_w).all())),
        ("high stratum: w_min >= cutoff", bool((by_stratum("high")["w_min"] >= high_min_w).all())),
        ("mid stratum inside range",
         bool(by_stratum("mid")["w_min"].between(low_max_w, high_min_w, inclusive="left").all())),
        ("no zero-feature rows", not bool(sample[zero_col].to_numpy(dtype=bool).any())),
        ("no blank bodies", bool((sample[body_col].astype(str).str.strip() != "").all())),
    ]
    return pd.DataFrame(checks, columns=["check", "passed"])


def write_review_file_once(review, path, id_col="id", float_format="%.4f"):
    """Write the review CSV once. An existing file (possibly with verdicts) is never overwritten."""
    path = Path(path)
    if path.exists():
        existing_ids = pd.read_csv(path, usecols=[id_col], dtype=str, encoding="utf-8-sig")[id_col]
        if sorted(existing_ids) != sorted(review[id_col].astype(str)):
            raise FileExistsError(f"{path} exists but its ids differ from the new draw")
        return "exists: kept as is (ids match the new draw)"
    path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(path, index=False, encoding="utf-8-sig", float_format=float_format)
    return "written"


# ── Step 2 / Cell 11: manual review verdicts and decision ──

def load_review_verdicts(path, review_table, allowed_verdicts, id_col="id", weight_tol=1e-4):
    """Read the filled review CSV, verify it still matches the drawn sample, return it with verdicts."""
    filled = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = [id_col, "rec_label", "band_label", "w_rec", "w_band", "verdict_rec", "verdict_band"]
    missing_columns = [name for name in required if name not in filled.columns]
    if missing_columns:
        raise AssertionError(f"review file is missing columns: {missing_columns}")

    # Same ids as the drawn sample (an id column reformatted by a spreadsheet would fail here)
    expected_ids = review_table[id_col].astype(str).to_numpy()
    if not filled[id_col].is_unique or set(filled[id_col]) != set(expected_ids) \
            or len(filled) != len(expected_ids):
        raise AssertionError("ids in the review file differ from the drawn sample")
    filled = filled.set_index(id_col).loc[expected_ids]

    # Labels and weights must be untouched by the manual editing
    for head in ("rec", "band"):
        if not np.array_equal(filled[f"{head}_label"].to_numpy(dtype=str),
                              review_table[f"{head}_label"].to_numpy(dtype=str)):
            raise AssertionError(f"{head}_label was modified in the review file")
        file_w = pd.to_numeric(filled[f"w_{head}"], errors="coerce").to_numpy(dtype="float64")
        if not np.allclose(file_w, review_table[f"w_{head}"].to_numpy(dtype="float64"),
                           atol=weight_tol, equal_nan=True):
            raise AssertionError(f"w_{head} was modified in the review file")

    # Verdicts: valid values only, filled exactly where the head has a label
    reviewed = review_table.copy()
    for head in ("rec", "band"):
        verdicts = filled[f"verdict_{head}"].str.strip().str.lower().to_numpy(dtype=object)
        invalid = ~np.isin(verdicts, list(allowed_verdicts) + [""])
        if invalid.any():
            raise AssertionError(f"verdict_{head}: invalid values {sorted(set(verdicts[invalid]))}")
        has_label = review_table[f"{head}_label"].to_numpy(dtype=str) != ""
        unfilled = int((has_label & (verdicts == "")).sum())
        unexpected = int((~has_label & (verdicts != "")).sum())
        if unfilled or unexpected:
            raise AssertionError(f"verdict_{head}: {unfilled} labeled rows without a verdict, "
                                 f"{unexpected} verdicts on rows without a label")
        reviewed[f"verdict_{head}"] = verdicts
    return reviewed


def review_verdict_counts(reviewed, heads, verdict_values, low_max_w, high_min_w):
    """Verdict counts per head and weight group (grouped by that head's own w)."""
    rows = []
    for head in heads:
        w = reviewed[f"w_{head}"]
        verdict = reviewed[f"verdict_{head}"]
        groups = {"low": w < low_max_w,
                  "mid": (w >= low_max_w) & (w < high_min_w),
                  "high": w >= high_min_w}
        for group, mask in groups.items():
            mask = mask.to_numpy() & (verdict != "").to_numpy()
            row = {"head": head, "group": group, "n_rows": int(mask.sum())}
            row.update({name: int((verdict[mask] == name).sum()) for name in verdict_values})
            row["n_decided"] = row["n_rows"] - row.get("unsure", 0)
            for name in ("ok", "too_low", "too_high"):
                row[f"share_{name}"] = row[name] / row["n_decided"] if row["n_decided"] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def review_acceptance_table(counts, heads, ok_min_share_low, too_high_max_share_high):
    """OK / WARN of the two review thresholds. 'unsure' is outside the denominator."""
    rows = []
    for head in heads:
        low = counts[(counts["head"] == head) & (counts["group"] == "low")].iloc[0]
        high = counts[(counts["head"] == head) & (counts["group"] == "high")].iloc[0]
        for check, source, value, threshold, rule, passed in [
            ("low_group_really_suspicious", low, low["share_ok"], ok_min_share_low,
             "share(ok) >= REVIEW_SUSPICIOUS_MIN_SHARE_LOW", lambda v: v >= ok_min_share_low),
            ("high_group_clear_contradiction", high, high["share_too_high"], too_high_max_share_high,
             "share(too_high) <= REVIEW_CLEAR_CONTRADICTION_MAX_SHARE_HIGH",
             lambda v: v <= too_high_max_share_high),
        ]:
            status = "NO_DATA" if source["n_decided"] == 0 else ("OK" if passed(value) else "WARN")
            rows.append({"head": head, "check": check, "n_decided": int(source["n_decided"]),
                         "value": value, "threshold": threshold, "rule": rule, "status": status})
    return pd.DataFrame(rows)


def write_json_once(payload, path):
    """Write a JSON file once. Identical content is a no-op; different content raises."""
    path = Path(path)
    new_text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == new_text:
            return "identical"
        raise FileExistsError(f"{path} already exists with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return "written"