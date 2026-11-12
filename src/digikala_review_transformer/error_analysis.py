"""Error-analysis helpers (more functions are added in later steps)."""
import pandas as pd


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