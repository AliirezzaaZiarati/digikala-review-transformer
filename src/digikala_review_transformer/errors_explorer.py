"""Build the interactive error-explorer HTML from errors_rec.csv / errors_band.csv."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from digikala_review_transformer import config

MODELS = ("tfidf", "tooka_B", "tooka_C", "mdeberta_B", "roberta_B")
HEADS = {
    "rec": ("Recommendation", ["recommended", "not_recommended", "no_idea"]),
    "band": ("Rating band", ["low", "medium", "high"]),
}
TEMPLATE = Path(__file__).parent / "templates" / "errors_explorer_template.html"
ERR_DIR = config.ROOT / "reports" / "error_analysis"
PLACEHOLDER = "/*__DATA__*/null"


def _raw_idx(s: pd.Series, classes: list[str]) -> pd.Series:
    """Raw (pre-Step-1) label -> class index, -1 when missing/unknown."""
    num = pd.to_numeric(s, errors="coerce")
    by_name = s.astype(str).map({c: i for i, c in enumerate(classes)})
    out = num.where(num.isin(range(len(classes))), by_name)
    return out.fillna(-1).astype(int)


def build_errors_explorer(
    models=MODELS,
    errors_dir=ERR_DIR,
    out_path=None,
    template=TEMPLATE,
    max_body_chars: int | None = None,   # set e.g. 600 if the HTML gets too heavy
    copy_fonts: bool = True,
) -> Path:
    errors_dir = Path(errors_dir)
    out_path = Path(out_path or errors_dir / "errors_explorer.html")
    data = {"models": list(models), "heads": {}}

    for head, (label, classes) in HEADS.items():
        df = pd.read_csv(errors_dir / f"errors_{head}.csv", dtype={"id": str})
        m = {c: i for i, c in enumerate(classes)}
        y = df["y_true"].map(m).astype(int)
        raw = _raw_idx(df["y_true_raw"], classes)
        P = np.column_stack([df[f"{k}_pred"].map(m) for k in models]).astype(int)
        C = df[[f"{k}_conf" for k in models]].to_numpy(float).round(3)
        body = df["body"].fillna("").astype(str)
        if max_body_chars:
            body = body.str.slice(0, max_body_chars)
        rows = [
            [i, b, int(a), int(r), p.tolist(), c.tolist()]
            for i, b, a, r, p, c in zip(df["id"], body, y, raw, P, C)
        ]
        data["heads"][head] = {"label": label, "classes": classes, "rows": rows}

    html = Path(template).read_text(encoding="utf-8")
    assert PLACEHOLDER in html, f"placeholder {PLACEHOLDER!r} not found in template"
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html.replace(PLACEHOLDER, blob), encoding="utf-8")

    if copy_fonts:  # optional local fonts: <html dir>/fonts/Vazirmatn-*.ttf
        fdir = out_path.parent / "fonts"
        fdir.mkdir(exist_ok=True)
        for f in (config.ROOT / "assets" / "fonts").glob("Vazirmatn-*.ttf"):
            shutil.copy2(f, fdir / f.name)

    n = {h: len(v["rows"]) for h, v in data["heads"].items()}
    print(f"saved -> {out_path} | {out_path.stat().st_size / 1e6:.1f} MB | rows: {n}")
    return out_path