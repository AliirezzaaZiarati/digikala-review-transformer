"""Custom two-head transformer training loop."""

from __future__ import annotations

# ---- Section: Imports ----
# Keep training imports light and compatible with Windows spawn workers.
import copy
import csv
import inspect
import math
import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import config, labels, metrics, utils
from . import checkpoint, data, losses, run_config
from . import model as model_module


# ---- Section: Errors and Result Types ----
# Return plain result data so scripts can write the final run summary later.
class TrainingLoopError(RuntimeError):
    """Raised when training, evaluation, or run-state validation fails."""


@dataclass(frozen=True)
class SplitEvaluation:
    """Predictions and metrics aligned to the original order of one split."""

    probabilities: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    plain_ce: dict[str, float]
    corrected_metrics: dict[str, dict[str, Any]]
    raw_metrics: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TrainEpochResult:
    """Aggregated training statistics for one completed epoch."""

    summary: dict[str, float]
    seconds: float
    samples: int


@dataclass(frozen=True)
class LoopResult:
    """Outcome of one training-loop invocation."""

    run_directory: Path
    status: str
    skipped: bool
    stop_reason: str | None
    epochs_done: int
    global_step: int
    best_epoch: int | None
    best_score: float | None
    train_seconds: float
    wall_seconds: float
    optimizer_fused: bool | None
    parameter_counts: dict[str, Any]
    steps_per_epoch: int
    total_steps: int
    warmup_steps: int | None
    history: list[dict[str, Any]]
    best_model_path: Path
    validation_probabilities_path: Path


# ---- Section: Output Columns ----
# Use the project label order when building probability and history outputs.
def _train_log_columns() -> list[str]:
    columns = [
        "epoch",
        "step_in_epoch",
        "steps_per_epoch",
        "global_step",
        "total_steps",
        "progress_pct",
        "train_objective",
    ]
    columns.extend(f"plain_ce_{head}" for head in data.HEADS)
    columns.extend(
        [
            "lr_encoder",
            "lr_heads",
            "samples_seen",
            "samples_per_second",
            "elapsed_training_seconds",
            "eta_seconds",
        ]
    )
    return columns


def _history_columns() -> list[str]:
    """Return the ordered per-epoch history schema."""
    columns = ["epoch", "train_objective"]
    columns.extend(f"train_plain_ce_{head}" for head in data.HEADS)
    columns.extend(f"train_subset_plain_ce_{head}" for head in data.HEADS)
    columns.extend(f"val_plain_ce_{head}" for head in data.HEADS)
    columns.extend(f"val_acc_{head}" for head in data.HEADS)
    columns.extend(f"val_macro_f1_{head}" for head in data.HEADS)
    columns.extend(f"val_macro_f1_{head}_raw" for head in data.HEADS)
    columns.extend(f"train_subset_macro_f1_{head}" for head in data.HEADS)
    columns.extend(f"val_acc_{head}_raw" for head in data.HEADS)
    columns.extend(
        [
            "val_mean_macro_f1",
            "lr_end",
            "lr_end_heads",
            "epoch_seconds",
            "is_best",
        ]
    )
    return columns


# ---- Section: Input Validation ----
# Validate labels and weights before constructing a DataLoader.
def _codes_as_int64(
    values: Any,
    *,
    split_name: str,
    head: str,
) -> np.ndarray:
    """Convert nullable class codes to int64, using IGNORE_INDEX for missing."""
    try:
        array = np.asarray(values)
        if array.ndim != 1:
            raise TrainingLoopError(
                f"{split_name}.{head} labels must be one-dimensional."
            )

        if np.issubdtype(array.dtype, np.integer):
            codes = array.astype(np.int64, copy=False)
        else:
            codes = pd.Series(values, copy=False).to_numpy(
                dtype=np.int64,
                na_value=data.IGNORE_INDEX,
            )
    except TrainingLoopError:
        raise
    except Exception as error:
        raise TrainingLoopError(
            f"Could not convert {split_name}.{head} labels to class codes."
        ) from error

    n_classes = len(labels.CLASSES[head])
    invalid = (codes != data.IGNORE_INDEX) & (
        (codes < 0) | (codes >= n_classes)
    )
    if np.any(invalid):
        bad_codes = np.unique(codes[invalid])[:5].tolist()
        raise TrainingLoopError(
            f"{split_name}.{head} contains invalid class codes {bad_codes}."
        )

    return codes


def _validate_split(
    split: data.SplitData,
    *,
    name: str,
    require_raw_labels: bool,
    require_train_weights: bool = False,
) -> None:
    """Check split lengths, labels, and optional training weights."""
    try:
        n_rows = int(split.n_rows)
        ids = np.asarray(split.ids)
        lengths = np.asarray(split.lengths)
    except Exception as error:
        raise TrainingLoopError(f"{name} is not a valid SplitData object.") from error

    if n_rows < 1:
        raise TrainingLoopError(f"{name} is empty.")
    if ids.ndim != 1 or len(ids) != n_rows:
        raise TrainingLoopError(
            f"{name}.ids length does not match the split row count."
        )
    if lengths.ndim != 1 or len(lengths) != n_rows:
        raise TrainingLoopError(
            f"{name}.lengths length does not match the split row count."
        )
    if not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths < 1):
        raise TrainingLoopError(f"{name} has invalid token sequence lengths.")

    for head in data.HEADS:
        if head not in split.y:
            raise TrainingLoopError(f"{name}.y is missing head {head!r}.")

        corrected = _codes_as_int64(
            split.y[head],
            split_name=name,
            head=head,
        )
        if len(corrected) != n_rows:
            raise TrainingLoopError(
                f"{name}.y[{head!r}] length does not match the split."
            )

        if require_raw_labels:
            if head not in split.y_raw:
                raise TrainingLoopError(
                    f"{name}.y_raw is missing head {head!r}."
                )
            raw = _codes_as_int64(
                split.y_raw[head],
                split_name=f"{name}.raw",
                head=head,
            )
            if len(raw) != n_rows:
                raise TrainingLoopError(
                    f"{name}.y_raw[{head!r}] length does not match the split."
                )

        if require_train_weights:
            if head not in split.w:
                raise TrainingLoopError(
                    f"train.w is missing head {head!r}."
                )
            weights = np.asarray(split.w[head], dtype=np.float64)
            if weights.ndim != 1 or len(weights) != n_rows:
                raise TrainingLoopError(
                    f"train.w[{head!r}] length does not match the split."
                )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise TrainingLoopError(
                    f"train.w[{head!r}] contains invalid weights."
                )

        if name == "train" and not np.any(corrected != data.IGNORE_INDEX):
            raise TrainingLoopError(
                f"Training labels for head {head!r} are all masked."
            )


def _positive_integer(training: Mapping[str, Any], key: str) -> int:
    value = training.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TrainingLoopError(f"training.{key} must be an integer.")
    if int(value) < 1:
        raise TrainingLoopError(f"training.{key} must be at least 1.")
    return int(value)


def _validate_training_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Read and validate settings used directly by the training loop."""
    try:
        training = cfg["training"]
    except Exception as error:
        raise TrainingLoopError("Resolved config is missing the training section.") from error

    if not isinstance(training, Mapping):
        raise TrainingLoopError("The training config section must be a mapping.")

    settings = {
        "max_epochs": _positive_integer(training, "max_epochs"),
        "patience": _positive_integer(
            training,
            "early_stopping_patience",
        ),
        "batch_size": _positive_integer(training, "batch_size"),
        "eval_batch_size": _positive_integer(training, "eval_batch_size"),
        "log_every_steps": _positive_integer(training, "log_every_steps"),
        "num_workers": training.get("num_workers"),
        "max_grad_norm": training.get("max_grad_norm"),
        "min_delta": training.get("early_stopping_min_delta"),
        "warmup_ratio": training.get("warmup_ratio"),
        "bf16": training.get("bf16"),
    }

    workers = settings["num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)):
        raise TrainingLoopError("training.num_workers must be a non-negative integer.")
    if int(workers) < 0:
        raise TrainingLoopError("training.num_workers must be non-negative.")
    settings["num_workers"] = int(workers)

    try:
        max_grad_norm = float(settings["max_grad_norm"])
        min_delta = float(settings["min_delta"])
        warmup_ratio = float(settings["warmup_ratio"])
    except (TypeError, ValueError) as error:
        raise TrainingLoopError(
            "max_grad_norm, early_stopping_min_delta, and warmup_ratio "
            "must be numeric."
        ) from error

    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise TrainingLoopError("training.max_grad_norm must be finite and positive.")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise TrainingLoopError(
            "training.early_stopping_min_delta must be finite and non-negative."
        )
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise TrainingLoopError(
            "training.warmup_ratio must be in the interval [0, 1)."
        )
    if not isinstance(settings["bf16"], bool):
        raise TrainingLoopError("training.bf16 must be a boolean.")

    settings["max_grad_norm"] = max_grad_norm
    settings["min_delta"] = min_delta
    settings["warmup_ratio"] = warmup_ratio
    return settings


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve and validate the training device."""
    if device is None:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            resolved = torch.device(device)
        except Exception as error:
            raise TrainingLoopError(f"Invalid device {device!r}.") from error

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise TrainingLoopError("CUDA was requested, but torch.cuda.is_available() is false.")
    return resolved


def _validate_bf16(enabled: bool, device: torch.device) -> None:
    """Require a CUDA device with bf16 support when bf16 autocast is requested."""
    if not enabled:
        return
    if device.type != "cuda":
        raise TrainingLoopError(
            "training.bf16=true requires a CUDA device; "
            "set it to false only for an intentional non-CUDA run."
        )
    if not torch.cuda.is_bf16_supported():
        raise TrainingLoopError(
            "The selected CUDA device does not report bf16 support."
        )


# ---- Section: Optimizer and Scheduler ----
# Keep the encoder and heads on the separate learning rates from the project plan.
def _build_optimizer(
    model: nn.Module,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.optim.Optimizer, bool, str | None]:
    training = cfg["training"]
    try:
        parameter_groups = model_module.parameter_groups(
            model,
            lr_encoder=float(training["lr_encoder"]),
            lr_heads=float(training["lr_heads"]),
            weight_decay=float(training["weight_decay"]),
        )
    except Exception as error:
        raise TrainingLoopError("Could not build optimizer parameter groups.") from error

    if not parameter_groups:
        raise TrainingLoopError("Optimizer parameter groups are empty.")

    adamw_parameters = inspect.signature(torch.optim.AdamW).parameters
    can_request_fused = "fused" in adamw_parameters
    request_fused = device.type == "cuda" and can_request_fused

    if request_fused:
        try:
            optimizer = torch.optim.AdamW(parameter_groups, fused=True)
            return optimizer, True, None
        except (TypeError, RuntimeError, ValueError) as fused_error:
            try:
                optimizer = torch.optim.AdamW(parameter_groups, fused=False)
            except Exception as fallback_error:
                raise TrainingLoopError(
                    "AdamW could not be constructed with fused=True or fused=False."
                ) from fallback_error

            warning = (
                "[WARN] Fused AdamW was unavailable; using standard AdamW. "
                f"Reason: {fused_error}"
            )
            return optimizer, False, warning

    try:
        optimizer = torch.optim.AdamW(parameter_groups)
    except Exception as error:
        raise TrainingLoopError("Could not construct the AdamW optimizer.") from error

    return optimizer, False, None


def make_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    """Build linear warmup followed by linear decay over all scheduled steps."""
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise TrainingLoopError("total_steps must be a positive integer.")
    if total_steps < 1:
        raise TrainingLoopError("total_steps must be at least 1.")

    warmup_ratio = float(warmup_ratio)
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise TrainingLoopError("warmup_ratio must be in the interval [0, 1).")

    warmup_steps = int(total_steps * warmup_ratio)

    def scale_at_step(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(decay_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=scale_at_step,
    )
    return scheduler, warmup_steps


def _learning_rates(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Read the encoder and head learning rates from named parameter groups."""
    encoder_lrs = []
    head_lrs = []

    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name.startswith("encoder"):
            encoder_lrs.append(float(group["lr"]))
        elif name.startswith("heads"):
            head_lrs.append(float(group["lr"]))

    if not encoder_lrs or not head_lrs:
        raise TrainingLoopError(
            "Optimizer groups must be named encoder_* and heads_*."
        )

    if not np.allclose(encoder_lrs, encoder_lrs[0]):
        raise TrainingLoopError("Encoder parameter groups have different learning rates.")
    if not np.allclose(head_lrs, head_lrs[0]):
        raise TrainingLoopError("Head parameter groups have different learning rates.")

    return encoder_lrs[0], head_lrs[0]


# ---- Section: Batch and RNG Helpers ----
# Isolate DataLoader iterator seeding from the model's CPU RNG stream.
def _new_iterator_without_rng_consumption(loader: Any) -> Any:
    cpu_rng_state = torch.get_rng_state()
    try:
        return iter(loader)
    finally:
        torch.set_rng_state(cpu_rng_state)


def _shutdown_loader(loader: Any) -> None:
    """Close persistent worker processes when a training call ends."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:  # noqa: BLE001, S110
            pass

    try:
        loader._iterator = None
    except Exception:  # noqa: BLE001, S110
        pass


def _batch_indices(batch: Mapping[str, Any]) -> np.ndarray:
    if "index" not in batch:
        raise TrainingLoopError("A DataLoader batch is missing the index field.")

    value = batch["index"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    indices = np.asarray(value)

    if indices.ndim == 0:
        indices = indices.reshape(1)
    else:
        indices = indices.reshape(-1)

    if not np.issubdtype(indices.dtype, np.integer):
        raise TrainingLoopError("Batch indices must be integers.")
    return indices.astype(np.int64, copy=False)


def _to_device(
    value: Any,
    device: torch.device,
    *,
    non_blocking: bool,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    return torch.as_tensor(value, device=device)


def _training_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move model inputs, labels, and sample weights while leaving indices on CPU."""
    required = {"input_ids", "attention_mask"}
    for head in data.HEADS:
        required.add(f"y_{head}")
        required.add(f"w_{head}")

    missing = required - set(batch)
    if missing:
        raise TrainingLoopError(
            f"Training batch is missing fields: {sorted(missing)}."
        )

    non_blocking = device.type == "cuda"
    return {
        key: _to_device(batch[key], device, non_blocking=non_blocking)
        for key in required
    }


def _evaluation_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    include_labels: bool,
) -> dict[str, torch.Tensor]:
    required = {"input_ids", "attention_mask"}
    if include_labels:
        required.update(f"y_{head}" for head in data.HEADS)

    missing = required - set(batch)
    if missing:
        raise TrainingLoopError(
            f"Evaluation batch is missing fields: {sorted(missing)}."
        )

    non_blocking = device.type == "cuda"
    return {
        key: _to_device(batch[key], device, non_blocking=non_blocking)
        for key in required
    }


def _autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    return nullcontext()


# ---- Section: Prediction and Evaluation ----
# Restore length-sorted batches to the split's original row order by index.
def _metrics_for_codes(
    true_codes: np.ndarray,
    predicted_codes: np.ndarray,
    head: str,
) -> dict[str, Any]:
    valid = true_codes != data.IGNORE_INDEX
    if not np.any(valid):
        return {
            "n": 0,
            "accuracy": math.nan,
            "macro_f1": math.nan,
            "balanced_accuracy": math.nan,
            "macro_precision": math.nan,
            "macro_recall": math.nan,
        }

    return metrics.evaluate(
        true_codes[valid],
        predicted_codes[valid],
        n_classes=len(labels.CLASSES[head]),
        ordinal=(head == "band"),
    )


def _collect_split_outputs(
    model: nn.Module,
    split: data.SplitData,
    batch_size: int,
    device: torch.device,
    *,
    bf16: bool,
    compute_plain_ce: bool,
    compute_metrics: bool,
    include_raw: bool,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise TrainingLoopError("Evaluation batch_size must be a positive integer.")
    _validate_split(
        split,
        name=str(split.split),
        require_raw_labels=compute_metrics and include_raw,
    )

    n_rows = int(split.n_rows)
    if n_rows < 1:
        raise TrainingLoopError(f"Cannot evaluate an empty split: {split.split}.")

    corrected_codes: dict[str, np.ndarray] = {}
    raw_codes: dict[str, np.ndarray] = {}
    if compute_metrics:
        for head in data.HEADS:
            corrected_codes[head] = _codes_as_int64(
                split.y[head],
                split_name=str(split.split),
                head=head,
            )
            if include_raw:
                raw_codes[head] = _codes_as_int64(
                    split.y_raw[head],
                    split_name=f"{split.split}.raw",
                    head=head,
                )

    probabilities = {
        head: np.empty(
            (n_rows, len(labels.CLASSES[head])),
            dtype=np.float32,
        )
        for head in data.HEADS
    }
    seen = np.zeros(n_rows, dtype=bool)
    plain_accumulator = (
        losses.LossAccumulator(device) if compute_plain_ce else None
    )

    row_indices = np.arange(n_rows, dtype=np.int64)
    eval_batches = data.sorted_eval_batches(
        split.lengths,
        row_indices,
        batch_size,
    )
    loader = data.make_loader(
        split.dataset(),
        eval_batches,
        num_workers=0,
        persistent=False,
    )

    old_training_mode = model.training
    model.eval()

    try:
        iterator = _new_iterator_without_rng_consumption(loader)
        with torch.inference_mode():
            for batch in iterator:
                indices = _batch_indices(batch)
                if len(indices) < 1:
                    raise TrainingLoopError("Evaluation produced an empty batch.")
                if np.any(indices < 0) or np.any(indices >= n_rows):
                    raise TrainingLoopError("Evaluation batch has an out-of-range index.")
                if np.any(seen[indices]):
                    raise TrainingLoopError(
                        "Evaluation batches contain duplicate row indices."
                    )

                batch_device = _evaluation_batch_to_device(
                    batch,
                    device,
                    include_labels=compute_plain_ce,
                )

                with _autocast_context(bf16):
                    logits_by_head = model(
                        input_ids=batch_device["input_ids"],
                        attention_mask=batch_device["attention_mask"],
                    )

                if not isinstance(logits_by_head, Mapping):
                    raise TrainingLoopError(
                        "The model must return a mapping of head logits."
                    )

                for head in data.HEADS:
                    if head not in logits_by_head:
                        raise TrainingLoopError(
                            f"The model output is missing head {head!r}."
                        )

                    logits = logits_by_head[head]
                    expected_classes = len(labels.CLASSES[head])
                    if (
                        logits.ndim != 2
                        or logits.shape[0] != len(indices)
                        or logits.shape[1] != expected_classes
                    ):
                        raise TrainingLoopError(
                            f"Invalid logits shape for head {head!r}: "
                            f"{tuple(logits.shape)}."
                        )

                    probabilities_batch = torch.softmax(
                        logits.float(),
                        dim=-1,
                    )
                    if not torch.isfinite(probabilities_batch).all():
                        raise TrainingLoopError(
                            f"Non-finite probabilities for head {head!r}."
                        )

                    probabilities[head][indices] = (
                        probabilities_batch.detach().cpu().numpy().astype(
                            np.float32,
                            copy=False,
                        )
                    )

                if plain_accumulator is not None:
                    plain_accumulator.add_logits(
                        logits_by_head,
                        batch_device,
                    )

                seen[indices] = True

    except TrainingLoopError:
        raise
    except Exception as error:
        raise TrainingLoopError(
            f"Could not evaluate split {split.split!r}: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        model.train(old_training_mode)

    if not np.all(seen):
        missing_count = int((~seen).sum())
        raise TrainingLoopError(
            f"Evaluation did not return {missing_count} rows from "
            f"split {split.split!r}."
        )

    if plain_accumulator is not None:
        accumulator_summary = plain_accumulator.summary()
        plain_ce = {
            head: float(accumulator_summary[f"plain_ce_{head}"])
            for head in data.HEADS
        }
    else:
        plain_ce = {head: math.nan for head in data.HEADS}

    predicted_codes = {
        head: probabilities[head].argmax(axis=1).astype(np.int64)
        for head in data.HEADS
    }

    corrected_metrics: dict[str, dict[str, Any]] = {}
    raw_metrics: dict[str, dict[str, Any]] = {}

    if compute_metrics:
        for head in data.HEADS:
            corrected_metrics[head] = _metrics_for_codes(
                corrected_codes[head],
                predicted_codes[head],
                head,
            )
            if include_raw:
                raw_metrics[head] = _metrics_for_codes(
                    raw_codes[head],
                    predicted_codes[head],
                    head,
                )

    return probabilities, plain_ce, corrected_metrics, raw_metrics


def evaluate_split(
    model: nn.Module,
    split: data.SplitData,
    batch_size: int,
    device: torch.device | str,
    *,
    bf16: bool = False,
    include_raw: bool = True,
) -> SplitEvaluation:
    """Evaluate corrected labels and, optionally, the raw label view."""
    resolved_device = _resolve_device(device)
    _validate_bf16(bf16, resolved_device)

    probabilities, plain_ce, corrected, raw = _collect_split_outputs(
        model,
        split,
        batch_size,
        resolved_device,
        bf16=bf16,
        compute_plain_ce=True,
        compute_metrics=True,
        include_raw=include_raw,
    )
    predictions = {
        head: probabilities[head].argmax(axis=1).astype(np.int64)
        for head in data.HEADS
    }

    return SplitEvaluation(
        probabilities=probabilities,
        predictions=predictions,
        plain_ce=plain_ce,
        corrected_metrics=corrected,
        raw_metrics=raw,
    )


def predict_probabilities(
    model: nn.Module,
    split: data.SplitData,
    batch_size: int,
    device: torch.device | str,
    *,
    bf16: bool = False,
) -> dict[str, np.ndarray]:
    """Return probabilities in the original split row order."""
    resolved_device = _resolve_device(device)
    _validate_bf16(bf16, resolved_device)

    probabilities, _, _, _ = _collect_split_outputs(
        model,
        split,
        batch_size,
        resolved_device,
        bf16=bf16,
        compute_plain_ce=False,
        compute_metrics=False,
        include_raw=False,
    )
    return probabilities


def probability_frame(
    split: data.SplitData,
    probabilities: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Build the required id plus p_<class> probability table."""
    ids = np.asarray(split.ids)
    if ids.ndim != 1 or len(ids) != int(split.n_rows):
        raise TrainingLoopError("Split ids are not aligned with split rows.")

    missing_heads = set(data.HEADS) - set(probabilities)
    if missing_heads:
        raise TrainingLoopError(
            f"Probability output is missing heads: {sorted(missing_heads)}."
        )

    columns: dict[str, Any] = {"id": ids}
    for head in data.HEADS:
        values = np.asarray(probabilities[head], dtype=np.float32)
        n_classes = len(labels.CLASSES[head])
        if values.shape != (int(split.n_rows), n_classes):
            raise TrainingLoopError(
                f"Probability shape for {head!r} is {values.shape}; "
                f"expected {(int(split.n_rows), n_classes)}."
            )
        if not np.all(np.isfinite(values)):
            raise TrainingLoopError(
                f"Probability output for {head!r} contains non-finite values."
            )
        if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-4, atol=1e-4):
            raise TrainingLoopError(
                f"Probability rows for {head!r} do not sum to one."
            )

        for class_index, class_name in enumerate(labels.CLASSES[head]):
            column_name = f"p_{class_name}"
            if column_name in columns:
                raise TrainingLoopError(
                    f"Probability column name is duplicated: {column_name}."
                )
            columns[column_name] = values[:, class_index].astype(
                np.float32,
                copy=False,
            )

    return pd.DataFrame(columns)


# ---- Section: Atomic Tabular Outputs ----
# Keep history and probability files from being left partially written.
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
            final_attempt = attempt + 1 == max_attempts
            if not _is_retryable_replace_error(error) or final_attempt:
                raise TrainingLoopError(
                    f"Could not replace output file {destination} "
                    f"after {attempt + 1} attempt(s)."
                ) from error
            time.sleep(retry_delay_seconds * (attempt + 1))


def _atomic_write_frame(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    file_format: str,
) -> Path:
    target = Path(path)
    if file_format not in {"csv", "parquet"}:
        raise TrainingLoopError(f"Unsupported table format: {file_format!r}.")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.",
            suffix=f".tmp.{file_format}",
            dir=target.parent,
        )
        os.close(file_descriptor)
    except OSError as error:
        raise TrainingLoopError(
            f"Could not create a temporary output beside {target}."
        ) from error

    temporary_path = Path(temporary_name)
    try:
        if file_format == "parquet":
            frame.to_parquet(temporary_path, index=False)
        else:
            frame.to_csv(
                temporary_path,
                index=False,
                encoding="utf-8",
            )
        _replace_with_retries(temporary_path, target)
        return target
    except TrainingLoopError:
        raise
    except Exception as error:
        raise TrainingLoopError(f"Could not write output file {target}.") from error
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _write_probability_file(
    split: data.SplitData,
    probabilities: Mapping[str, np.ndarray],
    path: Path,
) -> None:
    frame = probability_frame(split, probabilities)
    _atomic_write_frame(frame, path, file_format="parquet")


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame.from_records(
        history,
        columns=_history_columns(),
    )
    _atomic_write_frame(frame, path, file_format="csv")


# ---- Section: Run Logging ----
# Keep terminal progress, train_log.csv, and train.log readable and resumable.
def _reset_run_logs(run_directory: Path) -> None:
    train_log_path = run_directory / "train_log.csv"
    text_log_path = run_directory / "train.log"

    with train_log_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=_train_log_columns(),
        )
        writer.writeheader()

    text_log_path.write_text("", encoding="utf-8")


def _ensure_train_log_header(path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            try:
                existing_columns = next(reader)
            except StopIteration:
                existing_columns = []
        if existing_columns != _train_log_columns():
            raise TrainingLoopError(
                f"{path} has an unexpected header; do not append incompatible logs."
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(
            stream,
            fieldnames=_train_log_columns(),
        ).writeheader()


def _append_train_log(path: Path, row: Mapping[str, Any]) -> None:
    _ensure_train_log_header(path)
    try:
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=_train_log_columns(),
                extrasaction="raise",
            )
            writer.writerow(dict(row))
            stream.flush()
    except Exception as error:
        raise TrainingLoopError(f"Could not append to {path}.") from error


def _prune_uncommitted_train_logs(
    path: Path,
    committed_global_step: int,
) -> None:
    """Drop logs from an epoch that was not committed to last.pt."""
    if not path.is_file() or path.stat().st_size == 0:
        _ensure_train_log_header(path)
        return

    try:
        frame = pd.read_csv(path, encoding="utf-8")
        if "global_step" not in frame.columns:
            raise TrainingLoopError(
                f"{path} is missing the global_step column."
            )
        steps = pd.to_numeric(frame["global_step"], errors="raise")
        frame = frame.loc[steps <= committed_global_step].copy()
        frame = frame.loc[:, _train_log_columns()]
        _atomic_write_frame(frame, path, file_format="csv")
    except TrainingLoopError:
        raise
    except Exception as error:
        raise TrainingLoopError(
            "Could not reconcile train_log.csv with the saved checkpoint."
        ) from error


def _append_text_log(path: Path, message: str) -> None:
    print(message, flush=True)
    try:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
            stream.flush()
    except OSError as error:
        raise TrainingLoopError(f"Could not append to {path}.") from error


# ---- Section: Training Epoch ----
# Train each batch once; the shared validation score is handled at epoch end.
def _train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: losses.TwoHeadLoss,
    state: checkpoint.TrainState,
    device: torch.device,
    *,
    bf16: bool,
    epoch_number: int,
    steps_per_epoch: int,
    total_steps: int,
    train_rows: int,
    max_epochs: int,
    log_every_steps: int,
    max_grad_norm: float,
    run_name: str,
    train_log_path: Path,
    text_log_path: Path,
) -> TrainEpochResult:
    model.train()
    epoch_accumulator = losses.LossAccumulator(device)
    interval_accumulator = losses.LossAccumulator(device)
    training_started = time.monotonic()
    samples_this_epoch = 0
    steps_seen = 0

    iterator = _new_iterator_without_rng_consumption(loader)
    try:
        for step_in_epoch, batch in enumerate(iterator, start=1):
            steps_seen += 1
            indices = _batch_indices(batch)
            if len(indices) < 1:
                raise TrainingLoopError("Training produced an empty batch.")

            batch_device = _training_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(bf16):
                logits_by_head = model(
                    input_ids=batch_device["input_ids"],
                    attention_mask=batch_device["attention_mask"],
                )
                loss_output = loss_fn(logits_by_head, batch_device)

            if not torch.isfinite(loss_output.total).all():
                raise TrainingLoopError(
                    f"Non-finite training loss at epoch {epoch_number}, "
                    f"step {step_in_epoch}."
                )

            epoch_accumulator.add(loss_output)
            interval_accumulator.add(loss_output)

            loss_output.total.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                    error_if_nonfinite=True,
                )
            except Exception as error:
                raise TrainingLoopError(
                    f"Gradient clipping failed at epoch {epoch_number}, "
                    f"step {step_in_epoch}."
                ) from error

            optimizer.step()
            scheduler.step()

            state.global_step += 1
            samples_this_epoch += len(indices)

            is_log_step = (
                step_in_epoch % log_every_steps == 0
                or step_in_epoch == steps_per_epoch
            )
            if not is_log_step:
                continue

            interval_summary = interval_accumulator.summary()
            elapsed_this_call = time.monotonic() - training_started
            elapsed_training = state.train_seconds + elapsed_this_call
            samples_seen = state.epochs_done * train_rows + samples_this_epoch

            samples_per_second = (
                samples_seen / elapsed_training
                if elapsed_training > 0
                else math.nan
            )
            remaining_steps = max(0, total_steps - state.global_step)
            eta_seconds = (
                remaining_steps * elapsed_training / state.global_step
                if state.global_step > 0
                else math.nan
            )
            encoder_lr, heads_lr = _learning_rates(optimizer)
            progress_pct = (
                100.0 * state.global_step / total_steps
                if total_steps > 0
                else math.nan
            )

            log_row: dict[str, Any] = {
                "epoch": epoch_number,
                "step_in_epoch": step_in_epoch,
                "steps_per_epoch": steps_per_epoch,
                "global_step": state.global_step,
                "total_steps": total_steps,
                "progress_pct": progress_pct,
                "train_objective": interval_summary["objective"],
                "lr_encoder": encoder_lr,
                "lr_heads": heads_lr,
                "samples_seen": samples_seen,
                "samples_per_second": samples_per_second,
                "elapsed_training_seconds": elapsed_training,
                "eta_seconds": eta_seconds,
            }
            for head in data.HEADS:
                log_row[f"plain_ce_{head}"] = interval_summary[
                    f"plain_ce_{head}"
                ]

            _append_train_log(train_log_path, log_row)
            _append_text_log(
                text_log_path,
                (
                    f"[train] run={run_name} "
                    f"epoch={epoch_number}/{max_epochs} "
                    f"step={step_in_epoch}/{steps_per_epoch} "
                    f"global_step={state.global_step}/{total_steps} "
                    f"progress={progress_pct:.2f}% "
                    f"objective={interval_summary['objective']:.6f} "
                    f"lr_encoder={encoder_lr:.3e} "
                    f"lr_heads={heads_lr:.3e} "
                    f"samples_per_second={samples_per_second:.2f} "
                    f"ETA={eta_seconds / 60.0:.1f} min"
                ),
            )
            interval_accumulator.reset()

    finally:
        del iterator

    if steps_seen != steps_per_epoch:
        raise TrainingLoopError(
            f"Training yielded {steps_seen} batches; expected {steps_per_epoch}."
        )

    summary = epoch_accumulator.summary()
    if summary["steps"] != steps_per_epoch:
        raise TrainingLoopError(
            "Training loss accumulator step count does not match the sampler."
        )
    if not math.isfinite(float(summary["objective"])):
        raise TrainingLoopError("Epoch training objective is not finite.")

    for head in data.HEADS:
        value = float(summary[f"plain_ce_{head}"])
        if not math.isfinite(value):
            raise TrainingLoopError(
                f"Training plain CE for head {head!r} is not finite."
            )

    return TrainEpochResult(
        summary=summary,
        seconds=time.monotonic() - training_started,
        samples=samples_this_epoch,
    )


# ---- Section: Best-Checkpoint Publication ----
# Commit last.pt first; if publishing best artifacts is interrupted, resume can repair them.
def _publish_best_artifacts(
    model: nn.Module,
    validation_split: data.SplitData,
    evaluation: SplitEvaluation,
    run_directory: Path,
) -> None:
    checkpoint.save_best_weights(
        run_directory / "best_model.pt",
        model,
    )
    _write_probability_file(
        validation_split,
        evaluation.probabilities,
        run_directory / "val_probs_best.parquet",
    )


def _repair_or_check_best_artifacts(
    model: nn.Module,
    state: checkpoint.TrainState,
    validation_split: data.SplitData,
    run_directory: Path,
    *,
    eval_batch_size: int,
    device: torch.device,
    bf16: bool,
    text_log_path: Path,
) -> None:
    if state.best_epoch is None:
        if state.epochs_done != 0:
            raise TrainingLoopError(
                "The resume checkpoint has completed epochs but no best epoch."
            )
        return

    best_model_path = run_directory / "best_model.pt"
    best_probs_path = run_directory / "val_probs_best.parquet"

    if state.best_epoch == state.epochs_done:
        if not state.history or not bool(state.history[-1].get("is_best", False)):
            raise TrainingLoopError(
                "Checkpoint best_epoch conflicts with its last history record."
            )

        _append_text_log(
            text_log_path,
            (
                "[resume] Rebuilding best_model.pt and val_probs_best.parquet "
                "from the committed last.pt weights."
            ),
        )
        evaluation = evaluate_split(
            model,
            validation_split,
            eval_batch_size,
            device,
            bf16=bf16,
            include_raw=True,
        )
        _publish_best_artifacts(
            model,
            validation_split,
            evaluation,
            run_directory,
        )
        return

    if not best_model_path.is_file() or not best_probs_path.is_file():
        raise TrainingLoopError(
            "Best artifacts are missing although the best epoch predates "
            "the latest checkpoint; they cannot be reconstructed from last.pt."
        )


# ---- Section: Run-Directory Helpers ----
# Avoid silently starting over when a partial run has lost its resume checkpoint.
def _check_fresh_run_directory(run_directory: Path) -> None:
    protected = [
        run_directory / "run_summary.json",
        run_directory / "history.csv",
        run_directory / "best_model.pt",
        run_directory / "val_probs_best.parquet",
    ]
    existing = [path.name for path in protected if path.exists()]
    if existing:
        raise TrainingLoopError(
            "Run directory has artifacts but no resumable last.pt: "
            f"{existing}. Recover the run or use a new run directory."
        )


def _build_history_row(
    *,
    epoch_number: int,
    train_summary: Mapping[str, Any],
    monitor_evaluation: SplitEvaluation,
    validation_evaluation: SplitEvaluation,
    val_mean_macro_f1: float,
    encoder_lr: float,
    heads_lr: float,
    epoch_seconds: float,
    is_best: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch_number,
        "train_objective": float(train_summary["objective"]),
        "val_mean_macro_f1": val_mean_macro_f1,
        # lr_end is the encoder learning rate; lr_end_heads is also recorded.
        "lr_end": encoder_lr,
        "lr_end_heads": heads_lr,
        "epoch_seconds": epoch_seconds,
        "is_best": bool(is_best),
    }

    for head in data.HEADS:
        row[f"train_plain_ce_{head}"] = float(
            train_summary[f"plain_ce_{head}"]
        )
        row[f"train_subset_plain_ce_{head}"] = float(
            monitor_evaluation.plain_ce[head]
        )
        row[f"train_subset_macro_f1_{head}"] = float(
            monitor_evaluation.corrected_metrics[head]["macro_f1"]
        )
        row[f"val_plain_ce_{head}"] = float(
            validation_evaluation.plain_ce[head]
        )
        row[f"val_acc_{head}"] = float(
            validation_evaluation.corrected_metrics[head]["accuracy"]
        )
        row[f"val_macro_f1_{head}"] = float(
            validation_evaluation.corrected_metrics[head]["macro_f1"]
        )
        row[f"val_macro_f1_{head}_raw"] = float(
            validation_evaluation.raw_metrics[head]["macro_f1"]
        )
        row[f"val_acc_{head}_raw"] = float(
            validation_evaluation.raw_metrics[head]["accuracy"]
        )

    return row


# ---- Section: Main Training Loop ----
# Train, evaluate, checkpoint, and early-stop using the shared corrected-label score.
def fit(
    cfg: Mapping[str, Any],
    model: nn.Module,
    train_split: data.SplitData,
    validation_split: data.SplitData,
    train_subset_split: data.SplitData,
    *,
    run_directory: str | Path | None = None,
    device: torch.device | str | None = None,
    max_epochs_this_call: int | None = None,
) -> LoopResult:
    """Run training and return with the best model loaded into ``model``.

    ``max_epochs_this_call`` is a smoke-test/resume control. It does not alter
    the resolved config hash or the configured maximum epoch count.
    """
    wall_started = time.monotonic()
    settings = _validate_training_config(cfg)
    training = cfg["training"]

    if max_epochs_this_call is not None and (
        isinstance(max_epochs_this_call, bool)
        or not isinstance(max_epochs_this_call, int)
        or max_epochs_this_call < 1
    ):
        raise TrainingLoopError(
            "max_epochs_this_call must be a positive integer or None."
        )

    resolved_device = _resolve_device(device)
    _validate_bf16(settings["bf16"], resolved_device)

    if run_directory is None:
        resolved_run_directory = Path(run_config.run_dir(cfg))
    else:
        resolved_run_directory = Path(run_directory).expanduser()
        if not resolved_run_directory.is_absolute():
            resolved_run_directory = resolved_run_directory.resolve()

    try:
        config_digest = run_config.config_hash(cfg)
    except Exception as error:
        raise TrainingLoopError("Could not hash the resolved run config.") from error

    run_status = checkpoint.run_status(resolved_run_directory)
    parameter_counts = dict(model_module.count_parameters(model))

    best_model_path = resolved_run_directory / "best_model.pt"
    best_probs_path = resolved_run_directory / "val_probs_best.parquet"

    if run_status == "complete":
        print(
            f"[run] {resolved_run_directory.name} is already complete; skipping.",
            flush=True,
        )
        return LoopResult(
            run_directory=resolved_run_directory,
            status="complete",
            skipped=True,
            stop_reason="already_complete",
            epochs_done=0,
            global_step=0,
            best_epoch=None,
            best_score=None,
            train_seconds=0.0,
            wall_seconds=time.monotonic() - wall_started,
            optimizer_fused=None,
            parameter_counts=parameter_counts,
            steps_per_epoch=0,
            total_steps=0,
            warmup_steps=None,
            history=[],
            best_model_path=best_model_path,
            validation_probabilities_path=best_probs_path,
        )

    _validate_split(
        train_split,
        name="train",
        require_raw_labels=False,
        require_train_weights=True,
    )
    _validate_split(
        validation_split,
        name="validation",
        require_raw_labels=True,
    )
    _validate_split(
        train_subset_split,
        name="train_subset",
        require_raw_labels=False,
    )

    if train_subset_split.n_rows < 1:
        raise TrainingLoopError("The train monitoring subset is empty.")

    resolved_run_directory.mkdir(parents=True, exist_ok=True)
    history_path = resolved_run_directory / "history.csv"
    train_log_path = resolved_run_directory / "train_log.csv"
    text_log_path = resolved_run_directory / "train.log"
    last_path = resolved_run_directory / "last.pt"

    if run_status == "fresh":
        _check_fresh_run_directory(resolved_run_directory)
        _reset_run_logs(resolved_run_directory)
    else:
        _ensure_train_log_header(train_log_path)
        if not text_log_path.exists():
            text_log_path.write_text("", encoding="utf-8")

    _append_text_log(
        text_log_path,
        (
            f"[run] start run={cfg.get('run_name', resolved_run_directory.name)} "
            f"device={resolved_device} "
            f"variant={training.get('variant')} "
            f"seed={config.SPLIT_SEED}"
        ),
    )
    _append_text_log(
        text_log_path,
        f"[model] parameter_counts={parameter_counts}",
    )

    model.to(resolved_device)

    sampler = data.LengthBucketedBatches(
        train_split.lengths,
        settings["batch_size"],
        seed=config.SPLIT_SEED,
    )
    steps_per_epoch = len(sampler)
    if steps_per_epoch < 1:
        raise TrainingLoopError("The training sampler produced no batches.")

    total_steps = settings["max_epochs"] * steps_per_epoch
    optimizer, optimizer_fused, optimizer_warning = _build_optimizer(
        model,
        cfg,
        resolved_device,
    )
    scheduler, warmup_steps = make_linear_scheduler(
        optimizer,
        total_steps,
        settings["warmup_ratio"],
    )
    loss_fn = losses.build_loss(
        cfg,
        train_split.y,
        resolved_device,
    )

    train_loader = data.make_loader(
        train_split.dataset(),
        sampler,
        num_workers=settings["num_workers"],
        persistent=(settings["num_workers"] > 0),
    )

    if optimizer_warning is not None:
        _append_text_log(text_log_path, optimizer_warning)

    try:
        if run_status == "resumable":
            payload = checkpoint.load_last(
                last_path,
                expected_hash=config_digest,
            )
            state = checkpoint.restore_last(
                payload,
                model,
                optimizer,
                scheduler,
            )

            if state.epochs_done > settings["max_epochs"]:
                raise TrainingLoopError(
                    "Checkpoint epochs_done exceeds configured max_epochs."
                )
            if len(state.history) != state.epochs_done:
                raise TrainingLoopError(
                    "Checkpoint history length does not match epochs_done."
                )

            expected_global_step = state.epochs_done * steps_per_epoch
            if state.global_step != expected_global_step:
                raise TrainingLoopError(
                    "Checkpoint global_step does not match completed epochs "
                    f"(saved={state.global_step}, expected={expected_global_step})."
                )

            _write_history(history_path, state.history)
            _prune_uncommitted_train_logs(
                train_log_path,
                state.global_step,
            )
            _repair_or_check_best_artifacts(
                model,
                state,
                validation_split,
                resolved_run_directory,
                eval_batch_size=settings["eval_batch_size"],
                device=resolved_device,
                bf16=settings["bf16"],
                text_log_path=text_log_path,
            )

            _append_text_log(
                text_log_path,
                (
                    f"[resume] restored epochs_done={state.epochs_done} "
                    f"global_step={state.global_step} "
                    f"best_epoch={state.best_epoch} "
                    f"best_score={state.best_score:.6f}"
                ),
            )
        else:
            state = checkpoint.TrainState()

        _append_text_log(
            text_log_path,
            (
                f"[optimizer] fused={optimizer_fused} "
                f"steps_per_epoch={steps_per_epoch} "
                f"total_steps={total_steps} "
                f"warmup_steps={warmup_steps}"
            ),
        )

        epochs_run_this_call = 0
        stop_reason: str | None = None

        if state.bad_epochs >= settings["patience"]:
            stop_reason = "early_stopping_patience"
        elif state.epochs_done >= settings["max_epochs"]:
            stop_reason = "max_epochs_reached"

        while stop_reason is None:
            if (
                max_epochs_this_call is not None
                and epochs_run_this_call >= max_epochs_this_call
            ):
                stop_reason = "paused_by_call_limit"
                break

            epoch_index = state.epochs_done
            epoch_number = epoch_index + 1
            epoch_started = time.monotonic()

            sampler.set_epoch(epoch_index)

            with utils.timer(f"transformer_train_epoch_{epoch_number}"):
                train_result = _train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    state,
                    resolved_device,
                    bf16=settings["bf16"],
                    epoch_number=epoch_number,
                    steps_per_epoch=steps_per_epoch,
                    total_steps=total_steps,
                    train_rows=int(train_split.n_rows),
                    max_epochs=settings["max_epochs"],
                    log_every_steps=settings["log_every_steps"],
                    max_grad_norm=settings["max_grad_norm"],
                    run_name=str(cfg.get("run_name", resolved_run_directory.name)),
                    train_log_path=train_log_path,
                    text_log_path=text_log_path,
                )

            with utils.timer(f"transformer_validation_epoch_{epoch_number}"):
                validation_evaluation = evaluate_split(
                    model,
                    validation_split,
                    settings["eval_batch_size"],
                    resolved_device,
                    bf16=settings["bf16"],
                    include_raw=True,
                )

            with utils.timer(f"transformer_train_subset_epoch_{epoch_number}"):
                monitor_evaluation = evaluate_split(
                    model,
                    train_subset_split,
                    settings["eval_batch_size"],
                    resolved_device,
                    bf16=settings["bf16"],
                    include_raw=False,
                )

            corrected_scores = [
                float(validation_evaluation.corrected_metrics[head]["macro_f1"])
                for head in data.HEADS
            ]
            if not all(math.isfinite(value) for value in corrected_scores):
                raise TrainingLoopError(
                    "Corrected validation Macro F1 must be finite for both heads."
                )

            val_mean_macro_f1 = float(np.mean(corrected_scores))
            improved = (
                val_mean_macro_f1
                > state.best_score + settings["min_delta"]
            )

            if improved:
                state.best_score = val_mean_macro_f1
                state.best_epoch = epoch_number
                state.bad_epochs = 0
            else:
                state.bad_epochs += 1

            state.epochs_done = epoch_number
            state.train_seconds += train_result.seconds
            epoch_seconds = time.monotonic() - epoch_started
            encoder_lr, heads_lr = _learning_rates(optimizer)

            history_row = _build_history_row(
                epoch_number=epoch_number,
                train_summary=train_result.summary,
                monitor_evaluation=monitor_evaluation,
                validation_evaluation=validation_evaluation,
                val_mean_macro_f1=val_mean_macro_f1,
                encoder_lr=encoder_lr,
                heads_lr=heads_lr,
                epoch_seconds=epoch_seconds,
                is_best=improved,
            )
            state.history.append(history_row)

            # Commit the epoch state first so an interrupted best-artifact write is repairable.
            checkpoint.save_last(
                last_path,
                model,
                optimizer,
                scheduler,
                state,
                config_digest,
            )
            _write_history(history_path, state.history)

            if improved:
                _publish_best_artifacts(
                    model,
                    validation_split,
                    validation_evaluation,
                    resolved_run_directory,
                )

            epochs_run_this_call += 1

            _append_text_log(
                text_log_path,
                (
                    f"[epoch] epoch={epoch_number}/{settings['max_epochs']} "
                    f"train_objective={history_row['train_objective']:.6f} "
                    f"val_mean_macro_f1={val_mean_macro_f1:.6f} "
                    f"best_epoch={state.best_epoch} "
                    f"is_best={improved} "
                    f"bad_epochs={state.bad_epochs}/"
                    f"{settings['patience']} "
                    f"epoch_seconds={epoch_seconds:.1f}"
                ),
            )

            if state.bad_epochs >= settings["patience"]:
                stop_reason = "early_stopping_patience"
            elif state.epochs_done >= settings["max_epochs"]:
                stop_reason = "max_epochs_reached"
            elif (
                max_epochs_this_call is not None
                and epochs_run_this_call >= max_epochs_this_call
            ):
                stop_reason = "paused_by_call_limit"

        if state.best_epoch is None:
            raise TrainingLoopError(
                "Training ended without selecting a best validation checkpoint."
            )
        if not best_model_path.is_file():
            raise TrainingLoopError(
                f"Best model checkpoint is missing: {best_model_path}"
            )
        if not best_probs_path.is_file():
            raise TrainingLoopError(
                f"Best validation probabilities are missing: {best_probs_path}"
            )

        checkpoint.load_best_weights(best_model_path, model)

        elapsed_wall = time.monotonic() - wall_started
        status = "paused" if stop_reason == "paused_by_call_limit" else "trained"

        _append_text_log(
            text_log_path,
            (
                f"[run] training_loop_status={status} "
                f"stop_reason={stop_reason} "
                f"epochs_done={state.epochs_done} "
                f"global_step={state.global_step} "
                f"best_epoch={state.best_epoch} "
                f"best_score={state.best_score:.6f} "
                f"train_seconds={state.train_seconds:.1f} "
                f"wall_seconds={elapsed_wall:.1f}"
            ),
        )

        return LoopResult(
            run_directory=resolved_run_directory,
            status=status,
            skipped=False,
            stop_reason=stop_reason,
            epochs_done=state.epochs_done,
            global_step=state.global_step,
            best_epoch=state.best_epoch,
            best_score=state.best_score,
            train_seconds=state.train_seconds,
            wall_seconds=elapsed_wall,
            optimizer_fused=optimizer_fused,
            parameter_counts=parameter_counts,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            history=copy.deepcopy(state.history),
            best_model_path=best_model_path,
            validation_probabilities_path=best_probs_path,
        )
    finally:
        _shutdown_loader(train_loader)