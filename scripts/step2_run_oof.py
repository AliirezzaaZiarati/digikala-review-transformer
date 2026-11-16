"""Step 2: resumable OOF runner (one fresh LR per head and fold).

Reads everything from step2_params.json (written by notebook 09, Cell 5).
For each (head, fold): fit on the other folds, predict_proba on the held-out fold,
save parquet + meta json. A fold is "done" only when both files exist.
Each task runs in its own fresh process (spawn, one task per process), so the
peak RSS in the meta file belongs to that fit alone.
"""
import argparse
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def fold_paths(oof_dir, head, fold):
    """Return (parquet path, meta path) of one (head, fold)."""
    oof_dir = Path(oof_dir)
    return oof_dir / f"oof_{head}_fold{fold}.parquet", oof_dir / f"oof_{head}_fold{fold}_meta.json"


def make_recipe(params, head):
    """Everything that changes the OOF result. Stored in meta and compared on resume."""
    lr = params["lr"]
    return {
        "model_tag": params["model_tag"], "label_col": params["heads"][head]["label_col"],
        "C": params["heads"][head]["C"], "class_weight": lr["class_weight"],
        "solver": lr["solver"], "tol": lr["tol"], "max_iter": lr["max_iter"],
        "folds_sha256": params["folds_sha256"],
    }


def write_json_atomic(obj, path):
    """Write json to a temporary file, then rename (no half-written meta)."""
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def is_done(params, head, fold):
    """True if parquet and meta exist; raises if the stored recipe differs from the params."""
    parquet_path, meta_path = fold_paths(params["oof_dir"], head, fold)
    if not (parquet_path.exists() and meta_path.exists()):
        return False
    stored = json.loads(meta_path.read_text(encoding="utf-8"))["recipe"]
    if stored != make_recipe(params, head):
        raise RuntimeError(f"{meta_path.name}: stored recipe differs from current params. "
                           "Delete the fold files deliberately if the recipe really changed.")
    return True


def run_one_fold(params, head, fold):
    """Fit one fresh LR on the training folds and predict the held-out fold. Returns meta."""
    # heavy imports live in the worker process
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn import __version__ as sklearn_version
    from threadpoolctl import threadpool_limits

    from digikala_review_transformer.baseline import train_lr

    head_cfg, lr = params["heads"][head], params["lr"]
    id_col, class_names = params["id_col"], head_cfg["class_names"]
    parquet_path, meta_path = fold_paths(params["oof_dir"], head, fold)

    with threadpool_limits(limits=params["blas_threads_per_worker"]):
        # fold assignment and corrected labels (single source of truth: the folds parquet)
        folds = pd.read_parquet(params["folds_parquet"],
                                columns=[id_col, f"fold_{head}", f"y_{head}"])
        fold_values = folds[f"fold_{head}"].to_numpy(dtype="float64", na_value=np.nan)
        y_values = folds[f"y_{head}"].to_numpy(dtype="float64", na_value=np.nan)
        labeled = ~np.isnan(y_values)
        assert np.array_equal(labeled, ~np.isnan(fold_values)), "labels and folds disagree"

        held_out_pos = np.flatnonzero(fold_values == fold)
        train_pos = np.flatnonzero(labeled & (fold_values != fold))
        assert len(held_out_pos) > 0 and len(train_pos) > 0, "empty train or held-out side"
        assert len(held_out_pos) + len(train_pos) == int(labeled.sum()), "row accounting mismatch"
        assert len(set(held_out_pos) & set(train_pos)) == 0, "train and held-out overlap"
        y_train = y_values[train_pos].astype("int64")
        y_held_out = y_values[held_out_pos].astype("int64")
        n_classes = len(class_names)

        # load the cached matrix, slice both sides, then free the full matrix
        matrix = sparse.load_npz(params["x_train_npz"]).tocsr()
        assert matrix.shape[0] == len(folds), "matrix rows differ from folds rows"
        x_held_out = matrix[held_out_pos]
        x_train = matrix[train_pos]
        del matrix

        # fresh model with the baseline recipe; class_weight='balanced' is computed
        # by sklearn from y_train (the 4 training folds only)
        model, info = train_lr(x_train, y_train, C=head_cfg["C"], class_weight=lr["class_weight"],
                               solver=lr["solver"], max_iter=lr["max_iter"], tol=lr["tol"])
        assert list(model.classes_) == list(range(n_classes)), f"unexpected classes_: {model.classes_}"

        # out-of-fold probabilities for the held-out fold
        predict_start = time.perf_counter()
        proba = model.predict_proba(x_held_out)
        predict_seconds = time.perf_counter() - predict_start
        assert proba.shape == (len(held_out_pos), n_classes), "unexpected proba shape"
        assert np.isfinite(proba).all(), "non-finite probabilities"
        assert np.allclose(proba.sum(axis=1), 1.0, atol=params["prob_sum_tol"]), "probabilities do not sum to 1"

    # save the parquet first (temporary file, then rename), meta last
    oof_frame = pd.DataFrame({id_col: folds[id_col].astype(str).to_numpy()[held_out_pos]})
    for class_index, class_name in enumerate(class_names):
        oof_frame[f"p_{class_name}"] = proba[:, class_index]
    temporary = parquet_path.with_suffix(".parquet.tmp")
    oof_frame.to_parquet(temporary, index=False)
    temporary.replace(parquet_path)

    meta = {
        "head": head, "fold": int(fold), "recipe": make_recipe(params, head),
        "fit_seconds": info["fit_seconds"], "predict_seconds": predict_seconds,
        "n_iter": info["n_iter"], "converged": bool(info["converged"]),
        "n_train_rows": len(train_pos), "n_pred_rows": len(held_out_pos),
        "train_class_counts": np.bincount(y_train, minlength=n_classes).tolist(),
        "held_out_class_counts": np.bincount(y_held_out, minlength=n_classes).tolist(),
        "classes_": [int(code) for code in model.classes_],
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # KB on Linux
        "blas_threads": params["blas_threads_per_worker"],
        "sklearn_version": sklearn_version,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json_atomic(meta, meta_path)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Step 2 OOF runner (resumable)")
    parser.add_argument("--params", required=True, help="path to step2_params.json")
    parser.add_argument("--heads", nargs="+", default=None, help="default: all heads in params")
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="default: all folds")
    parser.add_argument("--n-jobs", type=int, default=None, help="default: n_jobs in params")
    args = parser.parse_args()

    params = json.loads(Path(args.params).read_text(encoding="utf-8"))
    heads = args.heads or params["head_names"]
    folds = args.folds if args.folds is not None else list(range(params["n_folds"]))
    n_jobs = args.n_jobs or params["n_jobs"]
    assert all(head in params["heads"] for head in heads), f"unknown head in {heads}"
    assert all(0 <= fold < params["n_folds"] for fold in folds), f"fold out of range: {folds}"

    # limit BLAS threads for every worker (children inherit the environment)
    for variable in THREAD_ENV_VARS:
        os.environ[variable] = str(params["blas_threads_per_worker"])

    # the folds file must be the one the params were written for, and aligned with the cache
    import pandas as pd

    from digikala_review_transformer.utils import sha256_file
    assert sha256_file(params["folds_parquet"]) == params["folds_sha256"], "folds parquet changed"
    folds_ids = pd.read_parquet(params["folds_parquet"], columns=[params["id_col"]])[params["id_col"]]
    cache_ids = pd.read_parquet(params["ids_train_parquet"])[params["id_col"]]
    assert folds_ids.astype(str).tolist() == cache_ids.astype(str).tolist(), "folds ids differ from cache ids"

    tasks = [(head, fold) for head in heads for fold in folds]
    pending = []
    for head, fold in tasks:
        if is_done(params, head, fold):
            print(f"[skip] {head} fold {fold}: already done", flush=True)
        else:
            pending.append((head, fold))
    print(f"[plan] {len(pending)} pending of {len(tasks)} | n_jobs={n_jobs} | "
          f"blas threads per worker={params['blas_threads_per_worker']}", flush=True)
    if not pending:
        return 0

    failed = []
    started = time.perf_counter()
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(pending)), mp_context=context,
                             max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_one_fold, params, head, fold): (head, fold) for head, fold in pending}
        for future in as_completed(futures):
            head, fold = futures[future]
            try:
                meta = future.result()
            except Exception as error:  # report and continue with the other folds  # noqa: BLE001
                failed.append((head, fold))
                print(f"[FAIL] {head} fold {fold}: {error!r}", flush=True)
                continue
            elapsed_min = (time.perf_counter() - started) / 60
            print(f"[done] {head} fold {fold} | fit {meta['fit_seconds']:.0f} s | n_iter {meta['n_iter']} | "
                  f"converged {meta['converged']} | peak RSS {meta['peak_rss_mb']:.0f} MB | "
                  f"elapsed {elapsed_min:.1f} min", flush=True)
            print(json.dumps(meta, indent=2), flush=True)

    print(f"[end] failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())