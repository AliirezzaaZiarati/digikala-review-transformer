"""Builds a standalone HTML error explorer (embedded JSON, no server) from candidate parquet files."""
from __future__ import annotations

import datetime
import html
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd

# Template lives next to this module: src/digikala_review_transformer/templates/
TEMPLATE_PATH = Path(__file__).parent / "templates" / "explorer_template.html"

# Columns that must exist in every candidate parquet (names verified in Step 12-pre)
REQUIRED_COLUMNS = [
    "id", "pos", "true_code", "pred_code", "conf", "true_name", "pred_name", "raw_name",
    "is_error", "label_changed", "n_words", "is_short", "has_negation", "body_in_train",
    "rate", "rate_is_zero", "cand_reason", "top_features_pred", "body",
]
FONT_FILES = {400: "Vazirmatn-Regular.ttf", 500: "Vazirmatn-Medium.ttf", 700: "Vazirmatn-Bold.ttf"}
DATA_SCRIPT_PATTERN = re.compile(
    r'<script id="explorer-data" type="application/json">(.*?)</script>', re.DOTALL)


def _to_native(value):
    """Convert numpy/pandas scalars to plain python values; NaN becomes None (JSON has no NaN)."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_candidates(model_id, candidates_dir):
    """Read one candidate parquet and validate it. Returns (dataframe, proba_columns)."""
    path = Path(candidates_dir) / f"{model_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    candidate_df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in candidate_df.columns]
    if missing:
        raise KeyError(f"{model_id}: missing columns {missing}")
    proba_columns = [c for c in candidate_df.columns if c.startswith("p_")]
    if len(proba_columns) != 3:
        raise ValueError(f"{model_id}: expected 3 p_<class> columns, found {proba_columns}")

    if candidate_df["id"].duplicated().any():
        raise ValueError(f"{model_id}: duplicate ids (a row must have exactly one cand_reason)")
    for column in ("true_name", "pred_name", "cand_reason", "body"):
        if candidate_df[column].isna().any():
            raise ValueError(f"{model_id}: nulls in '{column}'")
    # is_error must agree with the codes (guards against a mismatch between columns)
    codes_differ = candidate_df["true_code"].to_numpy() != candidate_df["pred_code"].to_numpy()
    if not (codes_differ == candidate_df["is_error"].to_numpy()).all():
        raise ValueError(f"{model_id}: is_error disagrees with true_code != pred_code")
    return candidate_df, proba_columns


def build_records(candidate_df, proba_columns):
    """One compact dict per row. Short keys keep the HTML small (mapping documented in the template)."""
    records = []
    for row in candidate_df.to_dict("records"):
        records.append({
            "id": str(row["id"]),                    # row id (string)
            "pos": int(row["pos"]),                  # position in validation.csv
            "t": row["true_name"],                   # corrected label (step 1)
            "p": row["pred_name"],                   # model prediction
            "r": _to_native(row["raw_name"]),        # raw label
            "c": round(float(row["conf"]), 4),       # confidence of the predicted class
            "e": int(bool(row["is_error"])),
            "lc": int(bool(row["label_changed"])),
            "sh": int(bool(row["is_short"])),
            "ng": int(bool(row["has_negation"])),
            "bt": int(bool(row["body_in_train"])),
            "rz": int(bool(row["rate_is_zero"])),
            "nw": int(row["n_words"]),
            "rate": _to_native(row["rate"]),
            "why": row["cand_reason"],               # why this row is in the candidate set
            "tf": _to_native(row["top_features_pred"]) or "",
            "b": str(row["body"]),
            "pr": [round(float(row[c]), 4) for c in proba_columns],   # probabilities, class order = classes
        })
    return records


def _font_face_css(font_dir, out_html_path):
    """@font-face rules: system font first, then a relative path to assets/fonts (best effort)."""
    if font_dir is None:
        return ""
    rules = []
    for weight, file_name in FONT_FILES.items():
        font_path = Path(font_dir) / file_name
        if not font_path.exists():
            continue
        try:
            relative = Path(os.path.relpath(font_path, Path(out_html_path).parent)).as_posix()
            url_part = f', url("{quote(relative)}") format("truetype")'
        except ValueError:      # e.g. different drives on Windows
            url_part = ""
        rules.append(f'@font-face{{font-family:"Vazirmatn";font-weight:{weight};'
                     f'src:local("Vazirmatn"){url_part};font-display:swap;}}')
    return "\n".join(rules)


def build_explorer_html(model_ids, candidates_dir, out_html_path, error_tags,
                        page_title, page_subtitle, font_dir=None, template_path=TEMPLATE_PATH):
    """Build the explorer HTML. `error_tags` is a dict {key: label}; empty dict hides tagging."""
    models_payload = []
    for model_id in model_ids:
        candidate_df, proba_columns = load_candidates(model_id, candidates_dir)
        reasons = [[str(k), int(v)] for k, v in candidate_df["cand_reason"].value_counts().items()]
        models_payload.append({
            "model_id": model_id,
            "classes": [c[2:] for c in proba_columns],     # strip the "p_" prefix
            "reasons": reasons,
            "rows": build_records(candidate_df, proba_columns),
        })

    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "error_tags": [{"key": k, "label": v} for k, v in error_tags.items()],
        "models": models_payload,
    }
    # "<" is escaped so the data can never close the <script> tag; \u003c is valid JSON
    data_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")

    template = Path(template_path).read_text(encoding="utf-8")
    for placeholder in ("__EXPLORER_DATA_JSON__", "__FONT_CSS__"):
        if template.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
    for placeholder in ("__PAGE_TITLE__", "__PAGE_SUBTITLE__"):
        if placeholder not in template:
            raise ValueError(f"template lacks {placeholder}")

    page = (template
            .replace("__PAGE_TITLE__", html.escape(page_title))
            .replace("__PAGE_SUBTITLE__", html.escape(page_subtitle))
            .replace("__FONT_CSS__", _font_face_css(font_dir, out_html_path))
            .replace("__EXPLORER_DATA_JSON__", data_json))     # data last: it may contain anything

    out_html_path = Path(out_html_path)
    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_html_path.write_text(page, encoding="utf-8")
    return {
        "html_path": str(out_html_path),
        "html_mb": round(out_html_path.stat().st_size / 1e6, 2),
        "rows_per_model": {m["model_id"]: len(m["rows"]) for m in models_payload},
    }


def verify_explorer_html(html_path, model_ids, candidates_dir):
    """Re-read the written HTML, extract the embedded JSON and compare it with the parquet files."""
    page = Path(html_path).read_text(encoding="utf-8")
    leftovers = [m for m in ("__EXPLORER_DATA_JSON__", "__PAGE_TITLE__", "__PAGE_SUBTITLE__", "__FONT_CSS__")
                 if m in page]
    if leftovers:
        raise ValueError(f"unreplaced placeholders: {leftovers}")
    if page.count('id="explorer-data"') != 1:
        raise ValueError("expected exactly one embedded data block")
    match = DATA_SCRIPT_PATTERN.search(page)
    if match is None:
        raise ValueError("embedded data block not found")
    payload = json.loads(match.group(1))

    if [m["model_id"] for m in payload["models"]] != list(model_ids):
        raise ValueError("model order/ids differ from requested")
    result = {"html_mb": round(Path(html_path).stat().st_size / 1e6, 2), "models": {}}
    for model_payload in payload["models"]:
        model_id = model_payload["model_id"]
        candidate_df, _ = load_candidates(model_id, candidates_dir)
        embedded_ids = [r["id"] for r in model_payload["rows"]]
        if embedded_ids != candidate_df["id"].astype(str).tolist():
            raise ValueError(f"{model_id}: embedded id order differs from parquet")
        if [r["b"] for r in model_payload["rows"]] != candidate_df["body"].astype(str).tolist():
            raise ValueError(f"{model_id}: embedded bodies differ from parquet")
        max_conf_diff = max(abs(r["c"] - c) for r, c in zip(model_payload["rows"], candidate_df["conf"]))
        if max_conf_diff > 1e-4:
            raise ValueError(f"{model_id}: conf differs by {max_conf_diff}")
        result["models"][model_id] = {
            "n_rows": len(embedded_ids),
            "n_errors": int(sum(r["e"] for r in model_payload["rows"])),
            "reasons": model_payload["reasons"],
        }
    return result