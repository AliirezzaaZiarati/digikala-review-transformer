"""Training pack: build from the split CSVs + Step 2 weights, verify, fingerprint.

Light imports only (pandas, pyarrow, numpy): this module is also used on the
training machine, where heavy analysis libraries are not installed.
"""

# ---- Section: Imports ----
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from digikala_review_transformer import config, data_io, labels

# ---- Section: Constants ----
HEADS = ("rec", "band")
SPLITS = ("train", "validation", "test")
SPLIT_FILES = {
    "train": "train_supervised.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
SOURCE_DIR = config.DATA_PROCESSED
WEIGHTS_PATH = config.DATA_PROCESSED / "step2_sample_weights.parquet"
MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1

# Plan: Step 2 weights live in [0.1, 1]; tolerance covers float32 rounding.
W_MIN, W_MAX, W_TOL = 0.1, 1.0, 1e-6

LABEL_COLUMNS = ["y_rec", "y_band", "y_rec_raw", "y_band_raw"]
PACK_COLUMNS = {
    "train": ["id", "body"] + LABEL_COLUMNS + ["w_rec", "w_band"],
    "validation": ["id", "body"] + LABEL_COLUMNS,
    "test": ["id", "body"] + LABEL_COLUMNS,
}


class PackError(RuntimeError):
    """Raised when the training pack cannot be built or is invalid."""


# ---- Section: Small helpers ----
def pack_file_name(split):
    return f"{split}.parquet"


def sha256_file(path, chunk_bytes=1 << 20):
    """Return the sha256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def count_csv_rows(path):
    """Count data rows with the csv module (independent of pandas parsing)."""
    csv.field_size_limit(2 ** 30)
    with open(path, encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def blank_to_na(series):
    """Turn empty / whitespace-only labels into NA before encoding."""
    text = series.astype("string")
    blank = text.isna() | text.str.strip().eq("").fillna(False)
    return series.mask(blank, pd.NA)


def relative_posix(path):
    return Path(path).resolve().relative_to(config.ROOT).as_posix()


def git_info():
    """Return the current commit and whether the working tree has changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=config.ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return {"commit": commit, "working_tree_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "working_tree_dirty": None}


def print_results(results):
    """Print [PASS]/[FAIL] lines; return True when everything passed."""
    for ok, message in results:
        print(("[PASS] " if ok else "[FAIL] ") + message)
    return all(ok for ok, _ in results)


def check_label_code_order():
    """The pack stores codes; confirm code == index in labels.CLASSES."""
    for head in HEADS:
        names = labels.CLASSES[head]
        codes = labels.encode(pd.Series(names), head).astype(int).tolist()
        if codes != list(range(len(names))):
            raise PackError(
                f"label codes for '{head}' are not 0..{len(names) - 1} in CLASSES order "
                f"-> inspect labels.encode before building the pack"
            )


# ---- Section: Build one split frame ----
def build_split_frame(split):
    """Read one split CSV and return the pack columns (without weights)."""
    raw = data_io.read_split(SOURCE_DIR / SPLIT_FILES[split])
    columns = {"id": raw["id"].astype(str), "body": raw["body"]}
    # Corrected labels first, then raw labels (same head order).
    for suffix, source_map in (("", labels.CORR_COL), ("_raw", labels.RAW_COL)):
        for head in HEADS:
            cleaned = blank_to_na(raw[source_map[head]])
            expected_missing = int(cleaned.isna().sum())
            encoded = labels.encode(cleaned, head)  # raises on unknown labels
            if int(encoded.isna().sum()) != expected_missing:
                raise PackError(
                    f"{split}: encoded NA count differs from blank count for "
                    f"'{source_map[head]}' -> check labels.encode handling of empty values"
                )
            columns[f"y_{head}{suffix}"] = encoded
    return pd.DataFrame(columns)


def attach_weights(frame):
    """Join Step 2 weights 1:1 on id (train only); weights stored as float32."""
    weights = pd.read_parquet(WEIGHTS_PATH, columns=["id", "w_rec", "w_band"])
    weights["id"] = weights["id"].astype(str)
    if not weights["id"].is_unique:
        raise PackError("duplicate ids in step2_sample_weights.parquet -> do not edit Step 2 outputs; report it")
    if len(weights) != len(frame):
        raise PackError(
            f"weights rows ({len(weights):,}) != train rows ({len(frame):,}) "
            f"-> weights file and train CSV are out of sync"
        )
    merged = frame.merge(weights, on="id", how="left", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise PackError("some train ids have no weight row -> check id dtype/format in both files")
    merged = merged.drop(columns="_merge")
    for head in HEADS:
        merged[f"w_{head}"] = merged[f"w_{head}"].astype("float32")
    return merged


# ---- Section: Frame checks and stats ----
def check_frame(frame, split):
    """Return [PASS]/[FAIL] results for one split frame."""
    results = []

    def add(ok, name, hint=""):
        results.append((bool(ok), f"{split}: {name}" if ok else f"{split}: {name} -> {hint}"))

    expected_columns = PACK_COLUMNS[split]
    add(list(frame.columns) == expected_columns, "columns match the plan",
        f"got {list(frame.columns)}, expected {expected_columns}")
    if list(frame.columns) != expected_columns:
        return results  # later checks need the right columns

    add(frame["id"].notna().all() and frame["id"].is_unique, "id is unique and non-null",
        "duplicate or missing ids; rebuild the pack")
    add(frame["body"].notna().all(), "body has no NA", "empty body cells; rebuild the pack")

    n_classes = {head: len(labels.CLASSES[head]) for head in HEADS}
    for head in HEADS:
        for suffix in ("", "_raw"):
            col = f"y_{head}{suffix}"
            codes = frame[col].dropna()
            add(codes.between(0, n_classes[head] - 1).all(), f"{col} codes are valid",
                f"codes outside 0..{n_classes[head] - 1}; rebuild the pack")

    if split == "train":
        for head in HEADS:
            labeled = frame[f"y_{head}"].notna().to_numpy()
            w = frame[f"w_{head}"].to_numpy(dtype="float64")[labeled]
            nan_count = int(np.isnan(w).sum())
            out_of_range = int(((w < W_MIN - W_TOL) | (w > W_MAX + W_TOL)).sum())
            add(nan_count == 0, f"w_{head} has no NaN on labeled rows",
                f"{nan_count:,} NaN weights on labeled rows")
            add(out_of_range == 0, f"w_{head} lies in [{W_MIN}, {W_MAX}]",
                f"{out_of_range:,} weights out of range")
    return results


def frame_stats(frame):
    """Counts for the manifest."""
    stats = {
        "rows": len(frame),
        "label_missing": {col: int(frame[col].isna().sum()) for col in LABEL_COLUMNS},
        "blank_body_rows": int(frame["body"].str.strip().eq("").sum()),
    }
    return stats


# ---- Section: Build the pack ----
def write_parquet_atomic(frame, path):
    """Write to a temp file, then replace (never leaves a half-written file)."""
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False, engine="pyarrow", compression="zstd")
    os.replace(temp, path)


def build_training_pack(pack_dir, force=False):
    """Build train/validation/test parquet files and manifest.json."""
    pack_dir = Path(pack_dir)
    manifest_path = pack_dir / MANIFEST_NAME
    if manifest_path.exists() and not force:
        raise PackError(
            f"{manifest_path} already exists -> the pack is frozen once built; "
            f"use --force only if you really want a new one"
        )
    if manifest_path.exists():
        manifest_path.unlink()  # manifest is the completion marker
    pack_dir.mkdir(parents=True, exist_ok=True)
    check_label_code_order()

    files, sources = {}, {}
    for split in SPLITS:
        print(f"--- {split} ---")
        source = SOURCE_DIR / SPLIT_FILES[split]
        frame = build_split_frame(split)
        if split == "train":
            frame = attach_weights(frame)

        csv_rows = count_csv_rows(source)
        results = check_frame(frame, split)
        results.append((len(frame) == csv_rows,
                        f"{split}: rows ({len(frame):,}) equal CSV data rows ({csv_rows:,})"
                        if len(frame) == csv_rows else
                        f"{split}: rows ({len(frame):,}) differ from CSV data rows ({csv_rows:,}) "
                        f"-> read_split dropped or merged rows; inspect the CSV"))
        if not print_results(results):
            raise PackError(f"{split} failed validation (see [FAIL] lines above); nothing more was written")

        out_path = pack_dir / pack_file_name(split)
        write_parquet_atomic(frame, out_path)
        written_rows = pq.ParquetFile(out_path).metadata.num_rows
        if written_rows != len(frame):
            raise PackError(f"{out_path.name}: wrote {written_rows:,} rows, expected {len(frame):,} -> disk problem? retry")

        files[out_path.name] = {
            **frame_stats(frame),
            "columns": list(frame.columns),
            "sha256": sha256_file(out_path),
            "size_mb": round(out_path.stat().st_size / 1e6, 1),
        }
        sources[relative_posix(source)] = {"rows": csv_rows, "sha256": sha256_file(source)}
        del frame

    sources[relative_posix(WEIGHTS_PATH)] = {"sha256": sha256_file(WEIGHTS_PATH)}
    manifest = {
        "format_version": FORMAT_VERSION,
        "code": git_info(),
        "label_codes": {head: list(labels.CLASSES[head]) for head in HEADS},
        "label_code_note": "code = index in the list",
        "weights": {"w_min": W_MIN, "w_max": W_MAX, "source": relative_posix(WEIGHTS_PATH)},
        "files": files,
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---- Section: Verify an existing pack ----
def verify_pack(pack_dir, check_content=False):
    """Check files against manifest.json. Returns (manifest or None, results)."""
    pack_dir = Path(pack_dir)
    results = []
    manifest_path = pack_dir / MANIFEST_NAME
    if not manifest_path.exists():
        results.append((False, f"{manifest_path} is missing -> build the pack or unzip it into data/"))
        return None, results
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for split in SPLITS:
        name = pack_file_name(split)
        entry = manifest["files"].get(name)
        path = pack_dir / name
        if entry is None or not path.exists():
            results.append((False, f"{name} missing -> re-extract the training pack zip"))
            continue
        sha_ok = sha256_file(path) == entry["sha256"]
        results.append((sha_ok, f"{name} sha256 matches manifest" if sha_ok else
                        f"{name} sha256 differs from manifest -> file is corrupted or edited; re-copy it"))
        parquet = pq.ParquetFile(path)
        rows_ok = parquet.metadata.num_rows == entry["rows"]
        results.append((rows_ok, f"{name} row count matches manifest ({entry['rows']:,})" if rows_ok else
                        f"{name} has {parquet.metadata.num_rows:,} rows, manifest says {entry['rows']:,} -> re-copy it"))
        cols_ok = list(parquet.schema_arrow.names) == PACK_COLUMNS[split]
        results.append((cols_ok, f"{name} columns match the plan" if cols_ok else
                        f"{name} columns are {parquet.schema_arrow.names} -> rebuild the pack"))
        if check_content and sha_ok and cols_ok:
            results.extend(check_frame(pd.read_parquet(path), split))
    return manifest, results


def pack_fingerprint(manifest):
    """Short id of the pack contents (used in token-cache keys)."""
    joined = "".join(manifest["files"][pack_file_name(s)]["sha256"] for s in SPLITS)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def print_summary(manifest):
    """Print a compact summary of a built pack."""
    print("\n=== Pack summary ===")
    for name, entry in manifest["files"].items():
        print(f"{name}: rows={entry['rows']:,}  size={entry['size_mb']} MB  sha256={entry['sha256'][:16]}...")
        print(f"   label_missing={entry['label_missing']}  blank_body_rows={entry['blank_body_rows']}")
    print(f"code commit: {manifest['code']['commit']}  dirty={manifest['code']['working_tree_dirty']}")
    print(f"fingerprint: {pack_fingerprint(manifest)}")