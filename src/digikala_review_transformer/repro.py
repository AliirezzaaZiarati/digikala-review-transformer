"""Reproducibility check: reload saved artifacts in a fresh process and compare with saved predictions."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

# Class order must match label codes (0, 1, 2) used in training and in the p_<class> parquet columns
HEAD_CLASS_NAMES = {
    "rec": ["recommended", "not_recommended", "no_idea"],
    "band": ["low", "medium", "high"],
}
HEADS = ("rec", "band")
VARIANTS = ("A", "B")


def _unwrap(loaded_obj, required_attr, dict_keys):
    """Return the object that has `required_attr`; accept a plain object or a dict holding it."""
    if hasattr(loaded_obj, required_attr):
        return loaded_obj
    if isinstance(loaded_obj, dict):
        for key in dict_keys:
            if key in loaded_obj and hasattr(loaded_obj[key], required_attr):
                return loaded_obj[key]
    raise TypeError(
        f"Loaded object of type {type(loaded_obj).__name__} has no '{required_attr}' "
        f"(dict keys: {list(loaded_obj) if isinstance(loaded_obj, dict) else 'n/a'})"
    )


def predict_texts(texts, vectorizer, model, class_names):
    """Vectorize texts and predict. One output row per text (core of the future offline script)."""
    feature_matrix = vectorizer.transform(list(texts))
    proba = model.predict_proba(feature_matrix)
    pred_code = proba.argmax(axis=1)
    result = pd.DataFrame({
        "pred_code": pred_code,
        "pred_name": [class_names[code] for code in pred_code],
        "conf": proba.max(axis=1),
        # number of non-zero tf-idf features; 0 means no known token (prediction comes from intercept only)
        "n_features": np.asarray(feature_matrix.getnnz(axis=1)).ravel(),
    })
    for class_index, class_name in enumerate(class_names):
        result[f"p_{class_name}"] = proba[:, class_index]
    return result


def reproducibility_check(model_tag, n_rows, seed, proba_tol, demo_sentences,
                          artifact_dir, predictions_dir, validation_csv):
    """Runs inside the fresh process. Returns a JSON-serializable dict."""
    start_time = time.perf_counter()
    vectorizer = _unwrap(joblib.load(artifact_dir / f"vectorizer_{model_tag}.joblib"),
                         "transform", ("vectorizer", "vec"))

    # Read only id + body; ids stay strings
    validation = pd.read_csv(validation_csv, dtype=str, keep_default_na=False,
                             usecols=["id", "body"], encoding="utf-8")
    rng = np.random.RandomState(seed)
    sample_positions = np.sort(rng.choice(len(validation), size=n_rows, replace=False))
    sample = validation.iloc[sample_positions]
    sample_ids = sample["id"].tolist()
    sample_bodies = sample["body"].tolist()

    check_rows, demo_rows = [], []
    for head in HEADS:
        class_names = HEAD_CLASS_NAMES[head]
        proba_columns = [f"p_{name}" for name in class_names]
        for variant in VARIANTS:
            model_id = f"{model_tag}_{head}_{variant}"
            model = _unwrap(joblib.load(artifact_dir / f"{model_id}.joblib"),
                            "predict_proba", ("model", "clf", "estimator"))
            if list(model.classes_) != [0, 1, 2]:
                raise ValueError(f"{model_id}: unexpected classes_ {list(model.classes_)}")

            # --- compare 1,000 validation rows with the saved parquet ---
            fresh = predict_texts(sample_bodies, vectorizer, model, class_names)
            saved = pd.read_parquet(predictions_dir / f"{model_id}__validation.parquet")
            needed_columns = ["id", "pred", *proba_columns]
            absent = [c for c in needed_columns if c not in saved.columns]
            if absent:
                raise KeyError(f"{model_id}: parquet lacks {absent}; columns are {list(saved.columns)}")
            saved["id"] = saved["id"].astype(str)
            saved = saved.set_index("id")
            if not saved.index.is_unique:
                raise ValueError(f"{model_id}: duplicate ids in parquet")

            found_mask = pd.Index(sample_ids).isin(saved.index)
            n_ids_missing = int((~found_mask).sum())
            if n_ids_missing > 0:
                check_rows.append({"model_id": model_id, "n_rows": len(sample_ids),
                                   "n_ids_missing_in_parquet": n_ids_missing, "max_abs_diff_proba": None,
                                   "mean_abs_diff_proba": None, "max_abs_diff_conf": None,
                                   "n_pred_mismatch": None, "ok": False})
            else:
                saved_rows = saved.loc[sample_ids]
                proba_diff = np.abs(saved_rows[proba_columns].to_numpy(dtype=np.float64)
                                    - fresh[proba_columns].to_numpy(dtype=np.float64))
                # 'pred' in parquet may be a code or a class name; handle both
                saved_pred = saved_rows["pred"].to_numpy()
                if np.issubdtype(saved_pred.dtype, np.number):
                    n_pred_mismatch = int((saved_pred.astype(int) != fresh["pred_code"].to_numpy()).sum())
                else:
                    n_pred_mismatch = int(sum(str(a) != str(b)
                                              for a, b in zip(saved_pred, fresh["pred_name"])))
                conf_diff = (float(np.abs(saved_rows["conf"].to_numpy(dtype=np.float64)
                                          - fresh["conf"].to_numpy()).max())
                             if "conf" in saved_rows.columns else None)
                check_rows.append({
                    "model_id": model_id, "n_rows": len(sample_ids), "n_ids_missing_in_parquet": 0,
                    "max_abs_diff_proba": float(proba_diff.max()),
                    "mean_abs_diff_proba": float(proba_diff.mean()),
                    "max_abs_diff_conf": conf_diff,
                    "n_pred_mismatch": n_pred_mismatch,
                    "ok": bool(proba_diff.max() <= proba_tol and n_pred_mismatch == 0),
                })

            # --- demo sentences ---
            demo = predict_texts(demo_sentences, vectorizer, model, class_names)
            for sentence_index, sentence in enumerate(demo_sentences):
                row = demo.iloc[sentence_index]
                demo_rows.append({
                    "sentence_index": sentence_index, "sentence": sentence, "model_id": model_id,
                    "pred_name": str(row["pred_name"]), "conf": float(row["conf"]),
                    "n_features": int(row["n_features"]),
                    "proba_sum": float(sum(row[c] for c in proba_columns)),
                    "probas": {name: float(row[f"p_{name}"]) for name in class_names},
                })

    # Demo validity: probabilities sum to 1 and predicted class is one of the head's classes
    demo_ok = all(abs(r["proba_sum"] - 1.0) < 1e-6 for r in demo_rows)
    checks_ok = all(r["ok"] for r in check_rows)
    return {
        "model_tag": model_tag, "n_rows": n_rows, "seed": seed, "proba_tol": proba_tol,
        "child_pid": os.getpid(),
        "elapsed_sec": round(time.perf_counter() - start_time, 1),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "scipy": scipy.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "joblib": joblib.__version__},
        "checks": check_rows, "demo": demo_rows,
        "demo_ok": bool(demo_ok), "checks_ok": bool(checks_ok),
        "all_ok": bool(demo_ok and checks_ok),
    }


def main():
    """CLI entry point of the fresh process."""
    from digikala_review_transformer.config import ARTIFACTS, DATA_PROCESSED, REPORTS

    parser = argparse.ArgumentParser(description="Artifact reproducibility check (fresh process)")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proba-tol", type=float, required=True)
    parser.add_argument("--demo-json", required=True)   # UTF-8 JSON list of sentences
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    demo_sentences = json.loads(Path(args.demo_json).read_text(encoding="utf-8"))
    result = reproducibility_check(
        model_tag=args.model_tag, n_rows=args.n_rows, seed=args.seed, proba_tol=args.proba_tol,
        demo_sentences=demo_sentences,
        artifact_dir=ARTIFACTS / "baseline_tfidf",
        predictions_dir=REPORTS / "predictions",
        validation_csv=DATA_PROCESSED / "validation.csv",
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written here with explicit UTF-8 because demo sentences are Persian
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"child pid {os.getpid()}: all_ok={result['all_ok']}")


def run_fresh_process_check(model_tag, n_rows, seed, proba_tol, demo_sentences, out_json):
    """Called from the notebook: runs `main` in a NEW python process and returns its result dict."""
    from digikala_review_transformer.config import ROOT

    out_json = Path(out_json)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_json = Path(tmp_dir) / "demo_sentences.json"
        demo_json.write_text(json.dumps(list(demo_sentences), ensure_ascii=False), encoding="utf-8")
        command = [sys.executable, "-m", "digikala_review_transformer.repro",
                   "--model-tag", model_tag, "--n-rows", str(n_rows), "--seed", str(seed),
                   "--proba-tol", str(proba_tol), "--demo-json", str(demo_json),
                   "--out-json", str(out_json)]
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                   text=True, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"fresh process failed (code {completed.returncode}):\n"
                           f"{completed.stderr[-3000:]}")
    result = json.loads(out_json.read_text(encoding="utf-8"))
    result["parent_pid"] = os.getpid()
    return result


def check_frame(result):
    """Per-model comparison table."""
    return pd.DataFrame(result["checks"])


def demo_table(result, model_ids):
    """One row per demo sentence; one column per model: 'class (conf)'."""
    demo = pd.DataFrame(result["demo"])
    demo["cell"] = demo["pred_name"] + " (" + demo["conf"].map("{:.2f}".format) + ")"
    wide = (demo[demo["model_id"].isin(model_ids)]
            .pivot(index="sentence_index", columns="model_id", values="cell")[list(model_ids)])
    first = demo.drop_duplicates("sentence_index").set_index("sentence_index")[["sentence", "n_features"]]
    return first.join(wide).reset_index(drop=True)


if __name__ == "__main__":
    main()