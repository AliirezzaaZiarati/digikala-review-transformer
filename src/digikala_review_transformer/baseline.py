"""TF-IDF baseline helpers (more functions are added in later steps)."""
import gc
import json
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .metrics import evaluate
from .utils import write_json

DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def tokenize(body, pattern, lowercase=True):
    """Tokens of one body exactly as the vectorizer would see them (unigrams only)."""
    s = body.lower() if lowercase else body
    return re.findall(pattern, s)


def vocab_growth_probe(bodies, sizes, ngram_range, token_pattern, min_df, seed):
    """Fit CountVectorizer (no pruning) on nested random subsets; report vocab size, nnz, time, RAM.
    bodies: pd.Series. Used only to decide how to fit the full vectorizer."""
    import psutil
    proc = psutil.Process()
    order = np.random.default_rng(seed).permutation(len(bodies))
    rows = []
    for n in sizes:
        gc.collect()
        rss0 = proc.memory_info().rss
        t0 = time.perf_counter()
        cv = CountVectorizer(token_pattern=token_pattern, ngram_range=ngram_range,
                             lowercase=True, min_df=1, dtype=np.int32)
        X = cv.fit_transform(bodies.iloc[order[:n]])
        sec = time.perf_counter() - t0
        rss1 = proc.memory_info().rss
        v = X.shape[1]
        df = np.bincount(X.indices, minlength=v)  # document frequency per feature
        rows.append({
            "n_docs": n, "raw_vocab": v, "vocab_df_ge_min": int((df >= min_df).sum()),
            "nnz": int(X.nnz), "nnz_per_doc": X.nnz / n, "seconds": sec,
            "rss_delta_gb": (rss1 - rss0) / 1e9,
            "matrix_gb": (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e9,
        })
        del cv, X, df
        gc.collect()
    return pd.DataFrame(rows)


def make_vectorizer(ngram_range, token_pattern, min_df, max_features, sublinear_tf):
    """Word-level TF-IDF, no stopword removal (negations must survive), float32 output."""
    return TfidfVectorizer(lowercase=True, token_pattern=token_pattern, ngram_range=ngram_range,
                           min_df=min_df, max_features=max_features,
                           sublinear_tf=sublinear_tf, dtype=np.float32)


def fit_and_cache(vec, bodies, ids, cache_dir, tag, vec_path):
    """Fit vec on bodies['train'] ONLY, transform every split, save matrices + id order + vectorizer.
    bodies: dict split -> pd.Series named 'body'; ids: dict split -> array of ids (same order).
    Returns (matrices dict, meta dict)."""
    for k, s in bodies.items():
        assert isinstance(s, pd.Series) and s.name == "body", f"{k}: only 'body' may enter the vectorizer"
        assert len(s) == len(ids[k]), f"{k}: bodies/ids length differ"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    X = {"train": vec.fit_transform(bodies["train"])}
    fit_sec = time.perf_counter() - t0
    # pruned terms are kept by sklearn in stop_words_; count them, then drop (it bloats the pickle)
    n_pruned = None
    if hasattr(vec, "stop_words_"):
        n_pruned = len(vec.stop_words_)
        delattr(vec, "stop_words_")
    t1 = time.perf_counter()
    for k in bodies:
        if k != "train":
            X[k] = vec.transform(bodies[k])
    transform_sec = time.perf_counter() - t1
    X = {k: m.astype(np.float32, copy=False).tocsr() for k, m in X.items()}

    V = X["train"].shape[1]
    df = np.bincount(X["train"].indices, minlength=V)   # document frequency of kept features
    meta = {
        "tag": tag, "vocab_size": int(V), "pruned_terms": n_pruned,
        "df_min_kept": int(df.min()), "df_median_kept": float(np.median(df)),
        "fit_seconds": fit_sec, "transform_seconds": transform_sec,
        "sklearn": sklearn.__version__,
        "shape": {k: list(m.shape) for k, m in X.items()},
        "nnz": {k: int(m.nnz) for k, m in X.items()},
        "nnz_per_doc": {k: m.nnz / m.shape[0] for k, m in X.items()},
        "zero_rows": {k: int((np.diff(m.indptr) == 0).sum()) for k, m in X.items()},
        "matrix_gb": {k: (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e9 for k, m in X.items()},
    }
    for k, m in X.items():
        sparse.save_npz(cache_dir / f"X_{tag}_{k}.npz", m, compressed=False)
        pd.DataFrame({"id": np.asarray(ids[k]).astype(str)}).to_parquet(
            cache_dir / f"ids_{tag}_{k}.parquet", index=False)
    joblib.dump(vec, vec_path)
    write_json(meta, cache_dir / f"meta_{tag}.json")
    return X, meta


def load_cache(cache_dir, tag, expected_ids):
    """Load cached matrices and assert the saved id order equals expected_ids (dict split -> ids)."""
    cache_dir = Path(cache_dir)
    X = {}
    for k, ids in expected_ids.items():
        saved = pd.read_parquet(cache_dir / f"ids_{tag}_{k}.parquet")["id"].astype(str).to_numpy()
        assert np.array_equal(saved, np.asarray(ids).astype(str)), f"{k}: id order differs from cache"
        X[k] = sparse.load_npz(cache_dir / f"X_{tag}_{k}.npz").tocsr()
        assert X[k].shape[0] == len(ids), f"{k}: row count differs"
    meta = json.loads((cache_dir / f"meta_{tag}.json").read_text(encoding="utf-8"))
    return X, meta

def train_lr(X, y, C, class_weight="balanced", solver="lbfgs", max_iter=200, tol=1e-4,
             sample_weight=None):
    """Fit multinomial logistic regression. Returns (model, info).
    ConvergenceWarning is recorded in info['converged'] (never silenced).
    sample_weight is optional (used later by the sample-weighting variant)."""
    clf = LogisticRegression(C=C, class_weight=class_weight, solver=solver,
                             max_iter=max_iter, tol=tol)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, y, sample_weight=sample_weight)
    sec = time.perf_counter() - t0
    assert list(clf.classes_) == [0, 1, 2], f"unexpected classes_: {clf.classes_}"
    conv = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    info = {"C": C, "fit_seconds": sec, "n_iter": int(np.max(clf.n_iter_)),
            "converged": len(conv) == 0, "n_rows": int(X.shape[0])}
    return clf, info


def fit_eval_one(model_id, C, X_train, pos, y, X_val, pos_val, y_val, head, train_params):
    """Fit on rows `pos` of X_train, score on rows `pos_val` of X_val. Returns (row dict, model)."""
    model, info = train_lr(X_train[pos], y, C, **train_params)
    m = evaluate(y_val, model.predict(X_val[pos_val]), n_classes=3, ordinal=(head == "band"))
    row = {"model_id": model_id, "C": C, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
           "fit_seconds": info["fit_seconds"], "n_iter": info["n_iter"], "converged": info["converged"],
           "n_train_rows": len(pos), "val_n": m["n"]}
    return row, model


def pick_c(df, tie_tol):
    """Smallest C whose validation Macro F1 is within tie_tol of the best."""
    best = df["val_macro_f1"].max()
    return float(df.loc[df["val_macro_f1"] >= best - tie_tol, "C"].min())


def tune_c(X_train, X_val, data, c_grid, train_params, n_jobs, tune_n, seed, tie_tol,
           extend_factor=3.0):
    """Tune C per model on validation Macro F1. data: model_id -> dict(head, pos, y, pos_val, y_val).
    tune_n=None uses all training rows; otherwise a seeded subsample (same rows for models with the
    same row count). If the winner is on the grid edge, one extra C is tried (once).
    Returns (results DataFrame, models dict {(model_id, C): model})."""
    sub = {}
    for mid, d in data.items():
        n = len(d["pos"])
        sub[mid] = (np.arange(n) if tune_n is None or tune_n >= n
                    else np.sort(np.random.default_rng(seed).choice(n, tune_n, replace=False)))

    def run(tasks):
        return Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(fit_eval_one)(mid, C, X_train, data[mid]["pos"][sub[mid]], data[mid]["y"][sub[mid]],
                                  X_val, data[mid]["pos_val"], data[mid]["y_val"], data[mid]["head"],
                                  train_params)
            for mid, C in tasks)

    out = run([(mid, C) for mid in data for C in c_grid])
    ext = []
    df0 = pd.DataFrame([r for r, _ in out])
    for mid in data:
        c = pick_c(df0[df0.model_id == mid], tie_tol)
        if c == min(c_grid):
            ext.append((mid, round(c / extend_factor, 4)))
        elif c == max(c_grid):
            ext.append((mid, round(c * extend_factor, 4)))
    if ext:
        print("grid-edge winners, extending once:", ext)
        out += run(ext)

    df = pd.DataFrame([r for r, _ in out])
    models = {(r["model_id"], r["C"]): m for r, m in out}
    df["tune_n"] = [len(sub[m]) for m in df["model_id"]]
    df["selected"] = False
    df["extended"] = ~df["C"].isin(c_grid)
    for mid in data:
        d = df[df.model_id == mid]
        df.loc[(df.model_id == mid) & (df.C == pick_c(d, tie_tol)), "selected"] = True
    df["at_edge"] = df.groupby("model_id")["C"].transform(
        lambda s: s.eq(s.min()) | s.eq(s.max())) & df["selected"]
    return df, models

def fit_final_candidates(X_train, X_val, data, cand_c, train_params, n_jobs, art_dir, tie_tol,
                         prev=None):
    """Fit every (model, C) candidate on ALL labeled train rows, score on validation, pick C per
    model with pick_c (validation Macro F1, smaller C on ties) and save ONLY the winner.
    data: model_id -> dict(head, pos, y, pos_val, y_val); cand_c: model_id -> list of C.
    prev: result of an earlier call; its rows join the selection and its saved winner is kept
    unless a new candidate beats it (then the artifact file is overwritten).
    Returns a DataFrame with one row per candidate."""
    art_dir = Path(art_dir)

    def one(mid, C):
        d = data[mid]
        model, info = train_lr(X_train[d["pos"]], d["y"], C, **train_params)
        m = evaluate(d["y_val"], model.predict(X_val[d["pos_val"]]), n_classes=3,
                     ordinal=(d["head"] == "band"))
        row = {"model_id": mid, "C": C, "val_macro_f1": m["macro_f1"],
               "val_macro_recall": m["macro_recall"], "val_balanced_accuracy": m["balanced_accuracy"],
               "val_accuracy": m["accuracy"], "fit_seconds": info["fit_seconds"],
               "n_iter": info["n_iter"], "converged": info["converged"],
               "n_train_rows": info["n_rows"], "max_iter": train_params["max_iter"]}
        return row, model

    tasks = [(mid, C) for mid, cs in cand_c.items() for C in cs]
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(m, c) for m, c in tasks)
    new = pd.DataFrame([r for r, _ in out])
    models = {(r["model_id"], r["C"]): m for r, m in out}
    new["selected"] = False
    new["artifact"] = ""
    old_art = {}
    if prev is not None:
        old_art = prev.loc[prev["selected"]].set_index("model_id")["artifact"].to_dict()
        old = prev.drop(columns=[c for c in ("sha256", "size_mb") if c in prev.columns]).copy()
        old["selected"] = False
        old["artifact"] = ""
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    for mid in cand_c:
        c = pick_c(df[df.model_id == mid], tie_tol)
        sel = (df.model_id == mid) & (df.C == c)
        df.loc[sel, "selected"] = True
        if (mid, c) in models:                      # a new candidate won: save it
            path = art_dir / f"{mid}.joblib"
            joblib.dump(models[(mid, c)], path)
            df.loc[sel, "artifact"] = str(path)
        else:                                       # the earlier winner stays
            df.loc[sel, "artifact"] = old_art[mid]
    return df


def _centered_coef(model):
    """Class coefficients minus their mean over classes (softmax is shift-invariant)."""
    return model.coef_ - model.coef_.mean(axis=0, keepdims=True)


def top_ngrams(model, names, df_train, classes, k, min_df=None):
    """Top-k features per class by centered coefficient. min_df filters rare features."""
    coef = _centered_coef(model)
    assert coef.shape == (len(classes), len(names)) and len(df_train) == len(names)
    cand = np.arange(len(names)) if min_df is None else np.flatnonzero(df_train >= min_df)
    rows = []
    for j, c in enumerate(classes):
        top = cand[np.argsort(-coef[j, cand])[:k]]
        for r, i in enumerate(top, 1):
            rows.append({"class": c, "rank": r, "feature": names[i], "coef": float(coef[j, i]),
                         "df_train": int(df_train[i]), "is_bigram": " " in names[i],
                         "has_zwnj": "\u200c" in names[i]})
    return pd.DataFrame(rows)


def term_ranks(model, names, classes, terms):
    """Rank (1 = most indicative) of given terms in each class, by centered coefficient."""
    coef = _centered_coef(model)
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for t in terms:
        if t not in idx:
            rows.append({"term": t, "in_vocab": False})
            continue
        i = idx[t]
        row = {"term": t, "in_vocab": True}
        for j, c in enumerate(classes):
            row[f"rank_{c}"] = int((coef[j] > coef[j, i]).sum()) + 1
            row[f"coef_{c}"] = round(float(coef[j, i]), 3)
        rows.append(row)
    return pd.DataFrame(rows)

def explain_rows(model, X, names, target, k=5):
    """Top-k features pushing each row toward its `target` class (centered coef * tf-idf value).
    X: sparse rows; target: int class code per row. Returns one string per row."""
    coef = _centered_coef(model)
    X = X.tocsr()
    out = []
    for i in range(X.shape[0]):
        s, e = X.indptr[i], X.indptr[i + 1]
        idx, val = X.indices[s:e], X.data[s:e]
        if len(idx) == 0:
            out.append("")
            continue
        contrib = val * coef[target[i], idx]
        top = np.argsort(-contrib)[:k]
        out.append(" | ".join(f"{names[idx[j]]} ({contrib[j]:+.2f})" for j in top))
    return out