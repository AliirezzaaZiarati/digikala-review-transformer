"""Error-analysis helpers (more functions are added in later steps)."""
import re

import numpy as np
import pandas as pd

from .labels import CLASSES


def in_train_mask(train_bodies, bodies):
    """Boolean array: body text exactly present in train_supervised."""
    return pd.Series(bodies).isin(pd.unique(pd.Series(train_bodies))).to_numpy()


def body_overlap_share(train_bodies, splits, label_col):
    """Percent of rows whose exact body also appears in train, per split and per label class.
    splits: dict name -> DataFrame with a 'body' column. Rows are counted, not unique bodies."""
    rows = []
    for name, df in splits.items():
        m = pd.Series(in_train_mask(train_bodies, df["body"]), index=df.index)
        lab = df[label_col].astype("object").where(df[label_col].notna(), "label_missing")
        rows.append({"split": name, "group": "all", "n": len(df), "n_in_train": int(m.sum())})
        for g, s in m.groupby(lab):
            rows.append({"split": name, "group": g, "n": len(s), "n_in_train": int(s.sum())})
    out = pd.DataFrame(rows)
    out["pct_in_train"] = 100 * out["n_in_train"] / out["n"]
    return out

def negation_pattern(words, prefixes=()):
    """Regex matching whole tokens: any of `words`, or any token starting with a prefix.
    ZWNJ counts as a word character (same convention as the vectorizer token pattern)."""
    alts = [re.escape(w) for w in sorted(words, key=len, reverse=True)]
    alts += [re.escape(p) + r"[\w\u200c]*" for p in prefixes]
    return r"(?<![\w\u200c])(?:" + "|".join(alts) + r")(?![\w\u200c])"


def add_flags(bodies, train_bodies, rate, neg_words, neg_prefixes, token_pattern, short_max=3):
    """Model-independent row flags (same order as `bodies`). Uses python `re`, not pandas .str."""
    tok = re.compile(token_pattern)
    neg = re.compile(negation_pattern(neg_words, neg_prefixes))
    b = pd.Series(bodies).reset_index(drop=True)
    r = pd.Series(rate, dtype="float64").reset_index(drop=True)
    n_words = np.fromiter((len(tok.findall(s)) for s in b), dtype=np.int32, count=len(b))
    has_neg = np.fromiter((neg.search(s) is not None for s in b), dtype=bool, count=len(b))
    return pd.DataFrame({
        "n_words": n_words,
        "is_short": n_words <= short_max,
        "has_negation": has_neg,
        "body_in_train": in_train_mask(train_bodies, b),
        "rate": r.to_numpy(),
        "rate_is_zero": r.eq(0).to_numpy(),
    })


def confidence_table(p, bins):
    """Accuracy per confidence bin against raw and corrected labels.
    p: prediction DataFrame. n counts all rows; n_<view> counts rows with a valid label."""
    edges = np.asarray(bins, dtype=float)
    assert np.all(np.diff(edges) > 0), "bins must increase"
    conf = p["conf"].to_numpy().astype(float)
    pred = p["pred"].to_numpy().astype(int)
    rows = []
    for k in range(len(edges) - 1):
        upper = edges[k + 1] + (1e-9 if k == len(edges) - 2 else 0.0)   # last bin includes the top edge
        m = (conf >= edges[k]) & (conf < upper)
        row = {"conf_from": edges[k], "conf_to": edges[k + 1], "n": int(m.sum()),
               "pct_rows": 100 * float(m.mean()),
               "mean_conf": float(conf[m].mean()) if m.any() else np.nan}
        for view, col in (("raw", "y_true_raw"), ("corrected", "y_true_corr")):
            yt = p[col].to_numpy(dtype="float64", na_value=np.nan)
            ok = m & ~np.isnan(yt)
            row[f"n_{view}"] = int(ok.sum())
            row[f"accuracy_{view}"] = float((yt[ok] == pred[ok]).mean()) if ok.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def high_conf_summary(p, head, high_conf):
    """Errors with conf >= high_conf: total ('ALL') and per (true, pred) pair, per label view."""
    names = CLASSES[head]
    rows = []
    for view, col in (("raw", "y_true_raw"), ("corrected", "y_true_corr")):
        yt = p[col].to_numpy(dtype="float64", na_value=np.nan)
        ok = ~np.isnan(yt)
        yt = yt[ok].astype(int)
        yp = p["pred"].to_numpy()[ok].astype(int)
        hi = p["conf"].to_numpy()[ok] >= high_conf
        err = yt != yp
        n_err, n_hi = int(err.sum()), int(hi.sum())
        pairs = [("ALL", "ALL", hi & err)]
        pairs += [(names[t], names[q], hi & err & (yt == t) & (yp == q))
                  for t in range(len(names)) for q in range(len(names)) if t != q]
        for t, q, m in pairs:
            c = int(m.sum())
            rows.append({"label_view": view, "true": t, "pred": q, "count": c,
                         "n_valid": int(ok.sum()), "n_high_conf": n_hi, "n_errors": n_err,
                         "pct_of_all_errors": 100 * c / n_err if n_err else 0.0,
                         "pct_of_high_conf_rows": 100 * c / n_hi if n_hi else 0.0})
    return pd.DataFrame(rows)


def step1_changed_check(preds, head):
    """Rows whose raw and corrected labels differ: how models A and B predict them.
    preds: dict variant -> prediction DataFrame (same ids, same order)."""
    names = CLASSES[head]
    ref = preds["B"]
    for p in preds.values():
        assert np.array_equal(p["id"].to_numpy(), ref["id"].to_numpy()), "id order differs between models"
    r = ref["y_true_raw"].to_numpy(dtype="float64", na_value=np.nan)
    c = ref["y_true_corr"].to_numpy(dtype="float64", na_value=np.nan)
    ch = ~np.isnan(r) & ~np.isnan(c) & (r != c)
    groups = [("ALL", "ALL", ch)]
    for a, b in sorted({(int(x), int(y)) for x, y in zip(r[ch], c[ch])}):
        groups.append((names[a], names[b], ch & (r == a) & (c == b)))
    rows = []
    for rl, cl, m in groups:
        row = {"head": head, "raw_label": rl, "corrected_label": cl, "n": int(m.sum())}
        for v, p in preds.items():
            pr = p["pred"].to_numpy().astype(int)
            eq_raw = 100 * float((pr[m] == r[m]).mean())
            eq_cor = 100 * float((pr[m] == c[m]).mean())
            row[f"{v}_pred_eq_raw_pct"] = eq_raw
            row[f"{v}_pred_eq_corrected_pct"] = eq_cor
            row[f"{v}_pred_other_pct"] = 100 - eq_raw - eq_cor
            row[f"{v}_mean_conf"] = float(p["conf"].to_numpy()[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def select_candidates(df, quotas, high_conf, low_conf, seed):
    """Seeded selection of rows for the explorer. df (unique index) needs: is_error, conf,
    label_changed, rate_is_zero, true_name, pred_name. quotas: category -> n; 'error_pair' is n PER
    (true, pred) error pair. A row keeps the first category that picked it (column cand_reason)."""
    rng = np.random.default_rng(seed)
    taken = pd.Series(False, index=df.index)
    parts = []

    def take(mask, n, reason):
        pool = df.index[(mask & ~taken).to_numpy()].to_numpy()
        if n <= 0 or len(pool) == 0:
            return
        pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
        taken.loc[pick] = True
        parts.append(pd.DataFrame({"idx": pick, "cand_reason": reason}))

    err = df["is_error"]
    take(err & (df["conf"] >= high_conf), quotas.get("high_conf_error", 0), "high_conf_error")
    take(df["label_changed"], quotas.get("label_changed_step1", 0), "label_changed_step1")
    take(err & df["rate_is_zero"], quotas.get("rate_zero_error", 0), "rate_zero_error")
    for (t, q), _ in df[err].groupby(["true_name", "pred_name"]):
        take(err & (df["true_name"] == t) & (df["pred_name"] == q),
             quotas.get("error_pair", 0), f"error_{t}_to_{q}")
    take(df["conf"] < low_conf, quotas.get("low_conf", 0), "low_conf")
    take(~err, quotas.get("correct_control", 0), "correct_control")
    sel = pd.concat(parts, ignore_index=True)
    out = df.loc[sel["idx"].to_numpy()].copy()
    out["cand_reason"] = sel["cand_reason"].to_numpy()
    return out