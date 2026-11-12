"""Metrics shared by all heads/models. Rows with NA true label are ignored."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .labels import CLASSES
from .predictions import read_predictions


def _valid(y_true, y_pred):
    """Keep only rows with a valid true label; return int arrays."""
    yt = pd.Series(y_true).reset_index(drop=True)
    yp = np.asarray(y_pred)
    assert len(yt) == len(yp), "y_true and y_pred length differ"
    m = yt.notna().to_numpy()
    return yt[m].astype(int).to_numpy(), yp[m].astype(int)


def evaluate(y_true, y_pred, n_classes=3, ordinal=False):
    """Return a dict of metrics. ordinal=True adds distance-based errors (band head)."""
    yt, yp = _valid(y_true, y_pred)
    labels = list(range(n_classes))
    out = {
        "n": int(len(yt)),  # noqa: RUF046
        "accuracy": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "macro_precision": precision_score(yt, yp, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(yt, yp, labels=labels, average="macro", zero_division=0),
    }
    if ordinal:
        d = np.abs(yt - yp)
        err = d > 0
        out["mean_abs_distance"] = float(d.mean())
        # shares are computed among misclassified rows only
        out["neighbor_error_pct"] = 100 * float((d[err] == 1).mean()) if err.any() else 0.0
        out["extreme_error_pct"] = 100 * float((d[err] == 2).mean()) if err.any() else 0.0
    return out


def per_class_table(y_true, y_pred, head):
    """Precision / recall / F1 / support / predicted count per class."""
    yt, yp = _valid(y_true, y_pred)
    n = len(CLASSES[head])
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=list(range(n)), zero_division=0)
    return pd.DataFrame({
        "class": CLASSES[head], "precision": p, "recall": r, "f1": f,
        "support": s, "predicted_count": np.bincount(yp, minlength=n),
    })


def confusion_long(y_true, y_pred, head):
    """Confusion matrix in long format with row-normalized percent."""
    yt, yp = _valid(y_true, y_pred)
    names = CLASSES[head]
    cm = confusion_matrix(yt, yp, labels=list(range(len(names))))
    rows = []
    for i, t in enumerate(names):
        tot = cm[i].sum()
        for j, p in enumerate(names):
            rows.append({"true": t, "pred": p, "count": int(cm[i, j]),
                         "row_pct": 100 * cm[i, j] / tot if tot else 0.0})
    return pd.DataFrame(rows)


def evaluate_all(y_raw, y_corr, y_pred, head):
    """Evaluate one prediction vector against raw and corrected labels (two rows)."""
    rows = []
    for view, y in (("raw", y_raw), ("corrected", y_corr)):
        d = evaluate(y, y_pred, n_classes=len(CLASSES[head]), ordinal=(head == "band"))
        d["label_view"] = view
        rows.append(d)
    return pd.DataFrame(rows)


def dummy_predict(y_train, n):
    """Majority class of the TRAIN labels, repeated n times. Returns (code, predictions)."""
    vals = pd.Series(y_train).dropna().astype(int)
    code = int(vals.value_counts().idxmax())
    return code, np.full(n, code, dtype=np.int8)

def macro_f1_from_confusion(cm_long, classes):
    """Recompute Macro F1 from a long confusion table (independent check of sklearn)."""
    f1s = []
    for c in classes:
        tp = cm_long.loc[(cm_long["true"] == c) & (cm_long["pred"] == c), "count"].sum()
        fp = cm_long.loc[(cm_long["true"] != c) & (cm_long["pred"] == c), "count"].sum()
        fn = cm_long.loc[(cm_long["true"] == c) & (cm_long["pred"] != c), "count"].sum()
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1s))


def _tag(df, **kw):
    """Insert constant columns at the front, in the given order."""
    for i, (k, v) in enumerate(kw.items()):
        df.insert(i, k, v)
    return df


def build_metric_tables(pred_dir, models, splits=("validation", "test")):
    """Summary / per-class / confusion tables for every model, split and label view, computed from
    the saved prediction files. models: model_id -> (head, variant)."""
    summ, pcs, cfs = [], [], []
    for mid, (head, var) in models.items():
        for split in splits:
            p = read_predictions(pred_dir, mid, split)
            summ.append(_tag(evaluate_all(p["y_true_raw"], p["y_true_corr"], p["pred"], head),
                             model_id=mid, head=head, variant=var, split=split))
            for view, col in (("raw", "y_true_raw"), ("corrected", "y_true_corr")):
                meta = dict(model_id=mid, head=head, variant=var, split=split, label_view=view)
                pcs.append(_tag(per_class_table(p[col], p["pred"], head), **meta))
                cfs.append(_tag(confusion_long(p[col], p["pred"], head), **meta))
    return (pd.concat(summ, ignore_index=True), pd.concat(pcs, ignore_index=True),
            pd.concat(cfs, ignore_index=True))


def ab_comparison(summary, pred_dir, models, splits=("validation", "test")):
    """B minus A on the SAME label view, plus the share of rows where A and B predict differently."""
    keys = ["head", "split", "label_view"]
    mets = ["accuracy", "macro_f1", "balanced_accuracy", "macro_recall", "mean_abs_distance"]
    a = summary[summary["variant"] == "A"].set_index(keys)[mets].add_suffix("_A")
    b = summary[summary["variant"] == "B"].set_index(keys)[mets].add_suffix("_B")
    out = a.join(b)
    for m in mets:
        out[f"{m}_diff_B_minus_A"] = out[f"{m}_B"] - out[f"{m}_A"]
    dis = {}
    for head in ("rec", "band"):
        ma = [m for m, (h, v) in models.items() if h == head and v == "A"][0]
        mb = [m for m, (h, v) in models.items() if h == head and v == "B"][0]
        for s in splits:
            pa = read_predictions(pred_dir, ma, s)["pred"].to_numpy()
            pb = read_predictions(pred_dir, mb, s)["pred"].to_numpy()
            dis[(head, s)] = 100 * float((pa != pb).mean())
    out = out.reset_index()
    out["pred_disagreement_pct"] = [dis[(h, s)] for h, s in zip(out["head"], out["split"])]
    return out


def band_by_rate_zero(pred_dir, band_models, rates, splits=("validation", "test")):
    """Band-head accuracy and class shares split by rate == 0 / rate > 0 / rate missing.
    rates: split -> array aligned with the prediction file (same order as the CSV)."""
    rows = []
    for mid, (head, var) in band_models.items():
        for split in splits:
            p = read_predictions(pred_dir, mid, split)
            r = pd.Series(rates[split]).reset_index(drop=True)
            assert len(r) == len(p), f"{mid}/{split}: rate length differs"
            groups = {"rate_zero": r.eq(0), "rate_positive": r.gt(0), "rate_missing": r.isna()}
            for view, col in (("raw", "y_true_raw"), ("corrected", "y_true_corr")):
                for g, m in groups.items():
                    ok = (p[col].notna() & m.to_numpy()).to_numpy()
                    if not ok.any():
                        continue
                    yt = p.loc[ok, col].astype(int).to_numpy()
                    yp = p.loc[ok, "pred"].astype(int).to_numpy()
                    rows.append({"model_id": mid, "split": split, "label_view": view, "group": g,
                                 "n": int(ok.sum()), "accuracy": float((yt == yp).mean()),
                                 "true_low_pct": 100 * float((yt == 0).mean()),
                                 "true_high_pct": 100 * float((yt == 2).mean()),
                                 "pred_low_pct": 100 * float((yp == 0).mean()),
                                 "mean_abs_distance": float(np.abs(yt - yp).mean())})
    return pd.DataFrame(rows)