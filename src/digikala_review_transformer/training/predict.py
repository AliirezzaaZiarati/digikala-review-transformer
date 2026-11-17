"""Prediction, verification, and run-finalization helpers."""

from __future__ import annotations

# ---- Section: Imports ----
# Keep imports light so prediction works in the lean training environment.
import copy
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import config, labels, metrics, utils  # noqa: F401
from . import (
    checkpoint,
    data,
    loop,
    pack,
    run_config,
)
from . import (
    model as model_module,
)

# ---- Section: Constants and Errors ----
# Name run artifacts in one place so training, packaging, and notebooks agree.
RESOLVED_CONFIG_NAME = "config_resolved.yaml"
HISTORY_NAME = "history.csv"
TRAIN_LOG_CSV_NAME = "train_log.csv"
TRAIN_LOG_TEXT_NAME = "train.log"
RUN_SUMMARY_NAME = "run_summary.json"
VAL_PROBS_NAME = "val_probs_best.parquet"
TEST_PROBS_NAME = "test_probs.parquet"
BEST_MODEL_NAME = "best_model.pt"
LAST_CHECKPOINT_NAME = "last.pt"
ENCODER_CONFIG_NAME = "encoder_config.json"
TOKENIZER_DIR_NAME = "tokenizer"

LIBRARY_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "pyyaml",
    "tqdm",
)


class PredictionError(RuntimeError):
    """Raised when prediction outputs or final run artifacts are invalid."""


@dataclass(frozen=True)
class FinalizedRun:
    """Paths and summary dictionary produced when a training run finishes."""

    run_directory: Path
    summary_path: Path
    validation_probabilities_path: Path
    test_probabilities_path: Path
    summary: dict[str, Any]


# ---- Section: Probability Schema and Validation ----
# Enforce exact id alignment and float32 class-probability columns.
def probability_columns_for_head(head: str) -> list[str]:
    """Return the ordered probability column names for one head."""
    if head not in data.HEADS:
        raise PredictionError(f"Unknown head {head!r}; expected one of {data.HEADS}.")
    return [f"p_{class_name}" for class_name in labels.CLASSES[head]]


def expected_probability_columns() -> list[str]:
    """Return the exact schema of val_probs_best.parquet and test_probs.parquet."""
    columns = ["id"]
    for head in data.HEADS:
        columns.extend(probability_columns_for_head(head))
    return columns


def verify_probability_frame(
    frame: pd.DataFrame,
    split: data.SplitData,
    *,
    source_name: str = "probability frame",
) -> dict[str, np.ndarray]:
    """Validate id alignment, float32 dtypes, and row sums; return probabilities."""
    if not isinstance(frame, pd.DataFrame):
        raise PredictionError(f"{source_name} is not a pandas DataFrame.")

    expected_columns = expected_probability_columns()
    actual_columns = list(frame.columns)
    if actual_columns != expected_columns:
        raise PredictionError(
            f"{source_name} has columns {actual_columns}; "
            f"expected {expected_columns}."
        )

    expected_ids = np.asarray(split.ids)
    actual_ids = frame["id"].to_numpy()
    if len(actual_ids) != len(expected_ids):
        raise PredictionError(
            f"{source_name} has {len(actual_ids)} rows; "
            f"expected {len(expected_ids)} rows for split {split.split!r}."
        )
    if not np.array_equal(actual_ids, expected_ids):
        raise PredictionError(
            f"{source_name} ids are not exactly aligned with split {split.split!r}."
        )

    probabilities: dict[str, np.ndarray] = {}
    for head in data.HEADS:
        head_columns = probability_columns_for_head(head)
        for column_name in head_columns:
            if frame[column_name].dtype != np.float32:
                raise PredictionError(
                    f"{source_name} column {column_name!r} has dtype "
                    f"{frame[column_name].dtype}; expected float32."
                )

        values = frame.loc[:, head_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        if not np.all(np.isfinite(values)):
            raise PredictionError(
                f"{source_name} contains non-finite probabilities for head {head!r}."
            )
        if np.any(values < -1.0e-6) or np.any(values > 1.0 + 1.0e-6):
            raise PredictionError(
                f"{source_name} has out-of-range probabilities for head {head!r}."
            )
        if not np.allclose(values.sum(axis=1), 1.0, rtol=1.0e-4, atol=1.0e-4):
            raise PredictionError(
                f"{source_name} probability rows for head {head!r} do not sum to 1."
            )

        probabilities[head] = values

    return probabilities


def read_and_verify_probability_parquet(
    path: str | Path,
    split: data.SplitData,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read a saved probability parquet and verify its schema and id order."""
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise PredictionError(f"Probability parquet does not exist: {parquet_path}")

    try:
        frame = pd.read_parquet(parquet_path)
    except Exception as error:
        raise PredictionError(
            f"Could not read probability parquet {parquet_path}: "
            f"{type(error).__name__}: {error}"
        ) from error

    probabilities = verify_probability_frame(
        frame,
        split,
        source_name=str(parquet_path),
    )
    return frame, probabilities


# ---- Section: Atomic File Writing ----
# Write parquet and JSON outputs through same-directory temporary files.
def _is_retryable_replace_error(error: OSError) -> bool:
    winerror = getattr(error, "winerror", None)
    return isinstance(error, PermissionError) or winerror in {5, 32, 33}


def _replace_with_retries(
    source: Path,
    destination: Path,
    *,
    max_attempts: int = 5,
    retry_delay_seconds: float = 0.2,
) -> None:
    for attempt in range(max_attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            is_last_attempt = attempt + 1 == max_attempts
            if not _is_retryable_replace_error(error) or is_last_attempt:
                raise PredictionError(
                    f"Could not atomically replace {destination} "
                    f"after {attempt + 1} attempt(s)."
                ) from error
            time.sleep(retry_delay_seconds * (attempt + 1))


def _atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.",
            suffix=".tmp.parquet",
            dir=target.parent,
        )
        os.close(descriptor)
    except OSError as error:
        raise PredictionError(
            f"Could not create a temporary parquet beside {target}."
        ) from error

    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        _replace_with_retries(temporary_path, target)
        return target
    except PredictionError:
        raise
    except Exception as error:
        raise PredictionError(
            f"Could not write parquet {target}: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _json_clean(value: Any) -> Any:
    """Convert nested summary values into strict, portable JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return _json_clean(value.item())

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]

    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())

    raise PredictionError(
        f"Run summary contains an unsupported JSON value of type {type(value).__name__}."
    )


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and atomically write a UTF-8 JSON object."""
    if not isinstance(payload, Mapping):
        raise PredictionError("JSON payload must be a mapping.")

    cleaned = _json_clean(dict(payload))
    try:
        serialized = json.dumps(
            cleaned,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except Exception as error:
        raise PredictionError(
            f"Could not serialize JSON for {path}: "
            f"{type(error).__name__}: {error}"
        ) from error

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
    except OSError as error:
        raise PredictionError(
            f"Could not create a temporary JSON file beside {target}."
        ) from error

    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(serialized + "\n", encoding="utf-8")
        _replace_with_retries(temporary_path, target)
        return target
    except PredictionError:
        raise
    except OSError as error:
        raise PredictionError(f"Could not write JSON file {target}.") from error
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


# ---- Section: Split Prediction and Metric Calculation ----
# Predict in length-sorted batches, restore original row order, and score both label views.
def predict_split_to_parquet(
    model: nn.Module,
    split: data.SplitData,
    output_path: str | Path,
    *,
    batch_size: int,
    device: torch.device | str | None = None,
    bf16: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Predict a split, verify id alignment, and atomically write a parquet file."""
    resolved_device = loop._resolve_device(device)
    try:
        probabilities = loop.predict_probabilities(
            model,
            split,
            batch_size,
            resolved_device,
            bf16=bf16,
        )
        frame = loop.probability_frame(split, probabilities)
    except Exception as error:
        raise PredictionError(
            f"Could not predict probabilities for split {split.split!r}: "
            f"{type(error).__name__}: {error}"
        ) from error

    verified_probabilities = verify_probability_frame(
        frame,
        split,
        source_name=f"predicted {split.split}",
    )
    _atomic_write_parquet(frame, output_path)
    return frame, verified_probabilities


def _codes_for_view(
    split: data.SplitData,
    *,
    head: str,
    view: str,
) -> np.ndarray:
    if view == "corrected":
        source = split.y
    elif view == "raw":
        source = split.y_raw
    else:
        raise PredictionError(f"Unknown label view {view!r}.")

    if head not in source:
        raise PredictionError(
            f"Split {split.split!r} is missing {view} labels for head {head!r}."
        )

    return loop._codes_as_int64(
        source[head],
        split_name=f"{split.split}.{view}",
        head=head,
    )


def evaluate_probabilities_on_split(
    split: data.SplitData,
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute corrected and raw metric dicts for both heads from probabilities."""
    frame = loop.probability_frame(split, probabilities)
    verified = verify_probability_frame(
        frame,
        split,
        source_name=f"{split.split} probabilities",
    )

    results: dict[str, dict[str, dict[str, Any]]] = {
        "corrected": {},
        "raw": {},
    }
    for view in ("corrected", "raw"):
        for head in data.HEADS:
            true_codes = _codes_for_view(split, head=head, view=view)
            predicted_codes = verified[head].argmax(axis=1).astype(np.int64)
            results[view][head] = loop._metrics_for_codes(
                true_codes,
                predicted_codes,
                head,
            )

    return results


def build_quick_summary(
    metrics_by_view: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build a compact both-view metric block for run_summary.json."""
    summary: dict[str, Any] = {}
    for view in ("corrected", "raw"):
        if view not in metrics_by_view:
            raise PredictionError(f"Metrics are missing label view {view!r}.")

        view_block: dict[str, Any] = {}
        macro_f1_values = []
        for head in data.HEADS:
            if head not in metrics_by_view[view]:
                raise PredictionError(
                    f"Metrics for view {view!r} are missing head {head!r}."
                )

            head_metrics = dict(metrics_by_view[view][head])
            macro_f1 = float(head_metrics["macro_f1"])
            macro_f1_values.append(macro_f1)

            entry = {
                "n": int(head_metrics["n"]),
                "macro_f1": macro_f1,
                "accuracy": float(head_metrics["accuracy"]),
                "balanced_accuracy": float(head_metrics["balanced_accuracy"]),
                "macro_precision": float(head_metrics["macro_precision"]),
                "macro_recall": float(head_metrics["macro_recall"]),
            }
            if head == "band":
                for ordinal_key in (
                    "mean_abs_distance",
                    "neighbor_error_pct",
                    "extreme_error_pct",
                ):
                    if ordinal_key in head_metrics:
                        entry[ordinal_key] = float(head_metrics[ordinal_key])

            view_block[head] = entry

        view_block["mean_macro_f1"] = (
            float(np.mean(macro_f1_values))
            if all(math.isfinite(value) for value in macro_f1_values)
            else None
        )
        summary[view] = view_block

    return summary


# ---- Section: Environment, Artifacts, and Hash Records ----
# Collect versions, hardware details, model/tokenizer artifacts, and file digests.
def collect_library_versions() -> dict[str, str | None]:
    """Record Python and key training library versions."""
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
    }
    for distribution_name in LIBRARY_DISTRIBUTIONS:
        try:
            versions[distribution_name] = importlib_metadata.version(
                distribution_name
            )
        except importlib_metadata.PackageNotFoundError:
            versions[distribution_name] = None
    return versions


def collect_hardware_info(device: torch.device | str | None = None) -> dict[str, Any]:
    """Record OS, device, CUDA version, GPU name, total VRAM, and peak VRAM."""
    resolved_device = loop._resolve_device(device)
    cuda_available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "device": str(resolved_device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "bf16_supported": (
            bool(torch.cuda.is_bf16_supported()) if cuda_available else False
        ),
        "gpu_name": None,
        "gpu_total_memory_mb": None,
        "gpu_max_memory_allocated_mb": None,
        "gpu_max_memory_reserved_mb": None,
    }

    if cuda_available and resolved_device.type == "cuda":
        device_index = (
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(device_index)
        megabyte = float(1024**2)
        info.update(
            gpu_name=str(properties.name),
            gpu_total_memory_mb=round(properties.total_memory / megabyte, 2),
            gpu_max_memory_allocated_mb=round(
                torch.cuda.max_memory_allocated(device_index) / megabyte,
                2,
            ),
            gpu_max_memory_reserved_mb=round(
                torch.cuda.max_memory_reserved(device_index) / megabyte,
                2,
            ),
        )

    return info


def save_run_Auxiliary_artifacts(
    cfg: Mapping[str, Any],
    model: nn.Module,
    tokenizer_bundle: Any,
    run_directory: str | Path,
) -> dict[str, Path]:
    """Save config_resolved.yaml, tokenizer/, and encoder_config.json."""
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)

    resolved_config_path = directory / RESOLVED_CONFIG_NAME
    tokenizer_directory = directory / TOKENIZER_DIR_NAME
    encoder_config_path = directory / ENCODER_CONFIG_NAME

    try:
        run_config.dump_resolved_config(dict(cfg), resolved_config_path)
    except Exception as error:
        raise PredictionError(
            f"Could not write {resolved_config_path}: "
            f"{type(error).__name__}: {error}"
        ) from error

    if not hasattr(tokenizer_bundle, "save") or not callable(tokenizer_bundle.save):
        raise PredictionError(
            "tokenizer_bundle must provide a callable save(out_dir) method."
        )
    try:
        tokenizer_bundle.save(tokenizer_directory)
    except Exception as error:
        raise PredictionError(
            f"Could not save tokenizer to {tokenizer_directory}: "
            f"{type(error).__name__}: {error}"
        ) from error

    try:
        model_module.save_encoder_config(model, encoder_config_path)
    except Exception as error:
        raise PredictionError(
            f"Could not write {encoder_config_path}: "
            f"{type(error).__name__}: {error}"
        ) from error

    return {
        "resolved_config": resolved_config_path,
        "tokenizer_dir": tokenizer_directory,
        "encoder_config": encoder_config_path,
    }


def _tokenizer_summary(
    cfg: Mapping[str, Any],
    tokenizer_bundle: Any,
) -> dict[str, Any]:
    if hasattr(tokenizer_bundle, "info") and callable(tokenizer_bundle.info):
        info = copy.deepcopy(dict(tokenizer_bundle.info()))
    else:
        info = {}

    model_cfg = cfg.get("model", {})
    info.setdefault(
        "backend",
        getattr(
            tokenizer_bundle,
            "backend",
            model_cfg.get("tokenizer_backend"),
        ),
    )
    info.setdefault(
        "max_length",
        int(
            getattr(
                tokenizer_bundle,
                "max_length",
                model_cfg.get("max_length"),
            )
        ),
    )
    info.setdefault(
        "name",
        getattr(
            tokenizer_bundle,
            "name",
            Path(str(model_cfg.get("pretrained_path", ""))).name,
        ),
    )
    return info


def _pack_sha256_summary(manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if manifest is None:
        return None

    files = manifest.get("files", {})
    split_hashes: dict[str, Any] = {}
    for split_name in pack.SPLITS:
        file_name = pack.pack_file_name(split_name)
        if file_name in files:
            entry = files[file_name]
            split_hashes[file_name] = {
                "sha256": entry.get("sha256"),
                "rows": entry.get("rows"),
                "size_mb": entry.get("size_mb"),
            }

    return {
        "fingerprint": pack.pack_fingerprint(manifest),
        "files": split_hashes,
    }


def _artifact_sha256_summary(run_directory: Path) -> dict[str, str]:
    tracked_files = [
        RESOLVED_CONFIG_NAME,
        HISTORY_NAME,
        TRAIN_LOG_CSV_NAME,
        VAL_PROBS_NAME,
        TEST_PROBS_NAME,
        BEST_MODEL_NAME,
        ENCODER_CONFIG_NAME,
    ]
    hashes: dict[str, str] = {}
    for relative_name in tracked_files:
        file_path = run_directory / relative_name
        if not file_path.is_file():
            raise PredictionError(
                f"Required run artifact is missing before summary creation: {file_path}"
            )
        hashes[relative_name] = pack.sha256_file(file_path)

    tokenizer_dir = run_directory / TOKENIZER_DIR_NAME
    if not tokenizer_dir.is_dir():
        raise PredictionError(f"Saved tokenizer directory is missing: {tokenizer_dir}")

    tokenizer_files = sorted(
        path for path in tokenizer_dir.rglob("*") if path.is_file()
    )
    if not tokenizer_files:
        raise PredictionError(f"Saved tokenizer directory is empty: {tokenizer_dir}")

    for file_path in tokenizer_files:
        relative_path = file_path.relative_to(run_directory).as_posix()
        hashes[relative_path] = pack.sha256_file(file_path)

    return hashes


def _best_history_row(
    history: list[dict[str, Any]],
    best_epoch: int,
) -> dict[str, Any]:
    for row in history:
        if int(row.get("epoch", -1)) == int(best_epoch):
            return copy.deepcopy(dict(row))
    raise PredictionError(
        f"History does not contain a record for best_epoch={best_epoch}."
    )


# ---- Section: Finalize Run ----
# Load best_model.pt, verify validation probs, predict test, write run_summary.json, remove last.pt.
def finalize_run(
    cfg: Mapping[str, Any],
    model: nn.Module,
    tokenizer_bundle: Any,
    loop_result: loop.LoopResult,
    *,
    train_split: data.SplitData,
    validation_split: data.SplitData,
    test_split: data.SplitData,
    train_subset_split: data.SplitData | None = None,
    manifest: Mapping[str, Any] | None = None,
    load_info: Mapping[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> FinalizedRun:
    """Finish a trained run without using test metrics for any model decision."""
    finalize_started = time.monotonic()
    if not isinstance(loop_result, loop.LoopResult):
        raise PredictionError("loop_result must be a LoopResult instance.")

    run_directory = Path(loop_result.run_directory)
    if loop_result.skipped or checkpoint.run_status(run_directory) == "complete":
        summary_path = run_directory / RUN_SUMMARY_NAME
        val_probs_path = run_directory / VAL_PROBS_NAME
        test_probs_path = run_directory / TEST_PROBS_NAME
        if not summary_path.is_file():
            raise PredictionError(
                f"Run was marked complete, but {summary_path} is missing."
            )
        read_and_verify_probability_parquet(val_probs_path, validation_split)
        read_and_verify_probability_parquet(test_probs_path, test_split)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return FinalizedRun(
            run_directory=run_directory,
            summary_path=summary_path,
            validation_probabilities_path=val_probs_path,
            test_probabilities_path=test_probs_path,
            summary=summary,
        )

    if loop_result.status != "trained":
        raise PredictionError(
            f"Refusing to finalize a run with loop status {loop_result.status!r}."
        )
    if loop_result.best_epoch is None or loop_result.best_score is None:
        raise PredictionError("Cannot finalize a run without best_epoch and best_score.")

    settings = loop._validate_training_config(cfg)
    resolved_device = loop._resolve_device(device)
    loop._validate_bf16(settings["bf16"], resolved_device)

    best_model_path = run_directory / BEST_MODEL_NAME
    val_probs_path = run_directory / VAL_PROBS_NAME
    test_probs_path = run_directory / TEST_PROBS_NAME
    summary_path = run_directory / RUN_SUMMARY_NAME
    text_log_path = run_directory / TRAIN_LOG_TEXT_NAME

    save_run_Auxiliary_artifacts(
        cfg,
        model,
        tokenizer_bundle,
        run_directory,
    )

    # Always reload best_model.pt before any final verification or test prediction.
    checkpoint.load_best_weights(best_model_path, model)
    model.to(resolved_device)

    _, val_probabilities = read_and_verify_probability_parquet(
        val_probs_path,
        validation_split,
    )
    validation_metrics = evaluate_probabilities_on_split(
        validation_split,
        val_probabilities,
    )
    validation_quick = build_quick_summary(validation_metrics)

    with utils.timer("transformer_test_prediction"):
        _, test_probabilities = predict_split_to_parquet(  # noqa: RUF059
            model,
            test_split,
            test_probs_path,
            batch_size=settings["eval_batch_size"],
            device=resolved_device,
            bf16=settings["bf16"],
        )

    # Read back the written parquet to guarantee the on-disk file is intact and aligned.
    _, verified_test_probabilities = read_and_verify_probability_parquet(
        test_probs_path,
        test_split,
    )
    test_metrics = evaluate_probabilities_on_split(
        test_split,
        verified_test_probabilities,
    )
    test_quick = build_quick_summary(test_metrics)

    token_stats_by_split: dict[str, Any] = {
        "train": copy.deepcopy(dict(train_split.stats)),
        "validation": copy.deepcopy(dict(validation_split.stats)),
        "test": copy.deepcopy(dict(test_split.stats)),
    }
    if train_subset_split is not None:
        token_stats_by_split["train_subset"] = copy.deepcopy(
            dict(train_subset_split.stats)
        )

    split_rows: dict[str, int] = {
        "train": int(train_split.n_rows),
        "validation": int(validation_split.n_rows),
        "test": int(test_split.n_rows),
    }
    if train_subset_split is not None:
        split_rows["train_subset"] = int(train_subset_split.n_rows)

    best_row = _best_history_row(
        loop_result.history,
        int(loop_result.best_epoch),
    )
    finalize_seconds = time.monotonic() - finalize_started

    artifact_hashes = _artifact_sha256_summary(run_directory)
    model_info = model_module.model_summary(
        model,
        dict(load_info) if load_info is not None else {},
    )
    tokenizer_info = _tokenizer_summary(cfg, tokenizer_bundle)

    summary_payload: dict[str, Any] = {
        "status": "complete",
        "run_name": str(cfg["run_name"]),
        "seed": int(cfg.get("seed", config.SPLIT_SEED)),
        "variant": str(cfg["training"]["variant"]),
        "config_hash": run_config.config_hash(cfg),
        "config": copy.deepcopy(dict(cfg)),
        "model": model_info,
        "parameter_counts": dict(loop_result.parameter_counts),
        "tokenizer_backend": tokenizer_info.get("backend"),
        "tokenizer": tokenizer_info,
        "token_stats": token_stats_by_split,
        "encoder_load_info": (
            copy.deepcopy(dict(load_info)) if load_info is not None else {}
        ),
        "split_rows": split_rows,
        "training_schedule": {
            "batch_size": settings["batch_size"],
            "eval_batch_size": settings["eval_batch_size"],
            "gradient_accumulation_steps": 1,
            "steps_per_epoch": int(loop_result.steps_per_epoch),
            "total_scheduled_steps": int(loop_result.total_steps),
            "warmup_steps": (
                None
                if loop_result.warmup_steps is None
                else int(loop_result.warmup_steps)
            ),
            "warmup_ratio": settings["warmup_ratio"],
            "optimizer": "AdamW",
            "optimizer_fused": loop_result.optimizer_fused,
            "lr_encoder": float(cfg["training"]["lr_encoder"]),
            "lr_heads": float(cfg["training"]["lr_heads"]),
            "weight_decay": float(cfg["training"]["weight_decay"]),
            "max_grad_norm": settings["max_grad_norm"],
            "label_smoothing": float(cfg["training"]["label_smoothing"]),
            "loss_weight_band": float(cfg["training"]["loss_weight_band"]),
            "use_class_weights": bool(cfg["training"]["use_class_weights"]),
            "bf16": settings["bf16"],
        },
        "selection_and_stopping": {
            "criterion": "val_mean_macro_f1_corrected",
            "labels_used_for_selection": "step1_corrected_only",
            "test_used_for_any_decision": False,
            "max_epochs": settings["max_epochs"],
            "early_stopping_patience": settings["patience"],
            "early_stopping_min_delta": settings["min_delta"],
            "shared_across_heads": True,
            "epochs_completed": int(loop_result.epochs_done),
            "global_steps_completed": int(loop_result.global_step),
            "best_epoch": int(loop_result.best_epoch),
            "best_val_mean_macro_f1": float(loop_result.best_score),
            "stop_reason": str(loop_result.stop_reason),
            "best_epoch_history": best_row,
        },
        "validation_best": validation_quick,
        "test_quick": test_quick,
        "timing_seconds": {
            "train_Pure_epoch_sum": round(float(loop_result.train_seconds), 3),
            "loop_wall": round(float(loop_result.wall_seconds), 3),
            "finalize_wall": round(float(finalize_seconds), 3),
            "total_wall": round(
                float(loop_result.wall_seconds + finalize_seconds),
                3,
            ),
        },
        "library_versions": collect_library_versions(),
        "hardware": collect_hardware_info(resolved_device),
        "pack": _pack_sha256_summary(manifest),
        "artifact_sha256": artifact_hashes,
    }

    atomic_write_json(summary_payload, summary_path)
    checkpoint.cleanup_after_complete(run_directory)

    if text_log_path.is_file():
        loop._append_text_log(
            text_log_path,
            (
                f"[finalize] status=complete "
                f"best_epoch={loop_result.best_epoch} "
                f"val_mean_macro_f1_corr="
                f"{validation_quick['corrected']['mean_macro_f1']:.6f} "
                f"test_mean_macro_f1_corr="
                f"{test_quick['corrected']['mean_macro_f1']:.6f} "
                f"test_mean_macro_f1_raw="
                f"{test_quick['raw']['mean_macro_f1']:.6f} "
                f"last_pt_removed={not (run_directory / LAST_CHECKPOINT_NAME).exists()}"
            ),
        )

    return FinalizedRun(
        run_directory=run_directory,
        summary_path=summary_path,
        validation_probabilities_path=val_probs_path,
        test_probabilities_path=test_probs_path,
        summary=summary_payload,
    )