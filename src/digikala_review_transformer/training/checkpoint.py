"""Atomic checkpoint helpers for custom transformer training."""

from __future__ import annotations

# ---- Section: Imports ----
# Keep this module independent of the training loop.
import copy
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---- Section: Constants and Errors ----
# Version the checkpoint format so incompatible files fail clearly.
CHECKPOINT_FORMAT_VERSION = 1
_LAST_REQUIRED_KEYS = {
    "format_version",
    "config_hash",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "train_state",
    "rng_state",
}
_RNG_REQUIRED_KEYS = {
    "python",
    "numpy",
    "torch_cpu",
    "torch_cuda",
}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely saved or restored."""


# ---- Section: Atomic File Saving ----
# Write beside the target so replacement stays on the same filesystem.
def _is_retryable_replace_error(error: OSError) -> bool:
    """Return whether an atomic replace error may be transient on Windows."""
    winerror = getattr(error, "winerror", None)
    return isinstance(error, PermissionError) or winerror in {5, 32, 33}


def atomic_torch_save(
    payload: Any,
    path: str | Path,
    *,
    max_attempts: int = 5,
    retry_delay_seconds: float = 0.2,
) -> Path:
    """Save with a same-directory temporary file and an atomic replacement."""
    target = Path(path)

    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise CheckpointError("max_attempts must be a positive integer.")
    if max_attempts < 1:
        raise CheckpointError("max_attempts must be at least 1.")
    if isinstance(retry_delay_seconds, bool) or not isinstance(
        retry_delay_seconds, (int, float)
    ):
        raise CheckpointError("retry_delay_seconds must be a non-negative number.")
    if not math.isfinite(float(retry_delay_seconds)) or retry_delay_seconds < 0:
        raise CheckpointError("retry_delay_seconds must be finite and non-negative.")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(file_descriptor)
    except OSError as error:
        raise CheckpointError(
            f"Could not create a temporary checkpoint beside {target}."
        ) from error

    temporary_path = Path(temporary_name)

    try:
        try:
            torch.save(payload, temporary_path)
        except Exception as error:
            raise CheckpointError(
                f"Could not serialize the checkpoint for {target}."
            ) from error

        for attempt in range(max_attempts):
            try:
                os.replace(temporary_path, target)
                return target
            except OSError as error:
                is_final_attempt = attempt + 1 == max_attempts
                if not _is_retryable_replace_error(error) or is_final_attempt:
                    raise CheckpointError(
                        f"Could not atomically replace checkpoint {target} "
                        f"after {attempt + 1} attempt(s)."
                    ) from error

                if retry_delay_seconds:
                    time.sleep(float(retry_delay_seconds))

        raise CheckpointError(f"Could not atomically save checkpoint {target}.")

    finally:
        # Remove an abandoned temporary file after serialization or replacement fails.
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


# ---- Section: Training State ----
# Store loop state as plain fields so it remains easy to validate and resume.
def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CheckpointError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise CheckpointError(f"{name} must be a non-negative integer.")
    return int(value)


@dataclass
class TrainState:
    """State maintained by the training loop and stored in last.pt."""

    epochs_done: int = 0
    global_step: int = 0
    best_score: float = -math.inf
    best_epoch: int | None = None
    bad_epochs: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    train_seconds: float = 0.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.epochs_done = _nonnegative_integer("epochs_done", self.epochs_done)
        self.global_step = _nonnegative_integer("global_step", self.global_step)
        self.bad_epochs = _nonnegative_integer("bad_epochs", self.bad_epochs)

        score = float(self.best_score)
        if math.isnan(score) or score == math.inf:
            raise CheckpointError("best_score must be finite or negative infinity.")
        self.best_score = score

        if self.best_epoch is not None:
            self.best_epoch = _nonnegative_integer("best_epoch", self.best_epoch)
            if self.best_epoch == 0:
                raise CheckpointError("best_epoch must be at least 1 when set.")
            if self.best_epoch > self.epochs_done:
                raise CheckpointError("best_epoch cannot exceed epochs_done.")
            if not math.isfinite(self.best_score):
                raise CheckpointError("best_score must be finite when best_epoch is set.")

        if isinstance(self.train_seconds, bool):
            raise CheckpointError("train_seconds must be finite and non-negative.")
        self.train_seconds = float(self.train_seconds)
        if not math.isfinite(self.train_seconds) or self.train_seconds < 0:
            raise CheckpointError("train_seconds must be finite and non-negative.")

        if not isinstance(self.history, list):
            raise CheckpointError("history must be a list of epoch records.")
        if not all(isinstance(row, Mapping) for row in self.history):
            raise CheckpointError("Every history item must be a mapping.")
        self.history = [copy.deepcopy(dict(row)) for row in self.history]

    def to_dict(self) -> dict[str, Any]:
        """Return a validated, detached copy suitable for checkpointing."""
        self._validate()
        return {
            "epochs_done": self.epochs_done,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "history": copy.deepcopy(self.history),
            "train_seconds": self.train_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainState:
        """Build a validated training state from checkpoint data."""
        if not isinstance(payload, Mapping):
            raise CheckpointError("The saved train_state is not a mapping.")

        required = {
            "epochs_done",
            "global_step",
            "best_score",
            "best_epoch",
            "bad_epochs",
            "history",
            "train_seconds",
        }
        missing = required - set(payload)
        unexpected = set(payload) - required
        if missing or unexpected:
            raise CheckpointError(
                "The saved train_state has an invalid schema "
                f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})."
            )

        try:
            return cls(
                epochs_done=payload["epochs_done"],
                global_step=payload["global_step"],
                best_score=payload["best_score"],
                best_epoch=payload["best_epoch"],
                bad_epochs=payload["bad_epochs"],
                history=payload["history"],
                train_seconds=payload["train_seconds"],
            )
        except CheckpointError:
            raise
        except Exception as error:
            raise CheckpointError("Could not parse the saved train_state.") from error


# ---- Section: Random Number Generator State ----
# Preserve all RNGs used by Python, NumPy, PyTorch CPU, and available CUDA devices.
def capture_rng() -> dict[str, Any]:
    """Capture the current Python, NumPy, CPU, and CUDA RNG states."""
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [
            state.detach().cpu().clone()
            for state in torch.cuda.get_rng_state_all()
        ]

    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda_states,
    }


def _validate_rng_state(
    rng_state: Mapping[str, Any],
    *,
    check_cuda_compatibility: bool,
) -> None:
    """Validate RNG data before changing the current process state."""
    if not isinstance(rng_state, Mapping):
        raise CheckpointError("The saved rng_state is not a mapping.")

    missing = _RNG_REQUIRED_KEYS - set(rng_state)
    if missing:
        raise CheckpointError(
            f"The saved rng_state is missing keys: {sorted(missing)}."
        )

    try:
        random.Random(0).setstate(rng_state["python"])
        np.random.RandomState(0).set_state(rng_state["numpy"])
    except Exception as error:
        raise CheckpointError("The saved Python or NumPy RNG state is invalid.") from error

    cpu_state = rng_state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.device.type != "cpu"
    ):
        raise CheckpointError("The saved CPU RNG state must be a CPU uint8 tensor.")

    try:
        torch.Generator(device="cpu").set_state(cpu_state)
    except Exception as error:
        raise CheckpointError("The saved PyTorch CPU RNG state is invalid.") from error

    cuda_states = rng_state["torch_cuda"]
    if cuda_states is None:
        if check_cuda_compatibility and torch.cuda.is_available():
            raise CheckpointError(
                "This checkpoint has no CUDA RNG state, but CUDA is available now."
            )
        return

    if not isinstance(cuda_states, (list, tuple)):
        raise CheckpointError("The saved CUDA RNG state must be a list or tuple.")

    for index, state in enumerate(cuda_states):
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.device.type != "cpu"
        ):
            raise CheckpointError(
                f"CUDA RNG state {index} must be a CPU uint8 tensor."
            )

    if check_cuda_compatibility:
        if not torch.cuda.is_available():
            raise CheckpointError(
                "This checkpoint has CUDA RNG state, but CUDA is unavailable now."
            )
        current_device_count = torch.cuda.device_count()
        if len(cuda_states) != current_device_count:
            raise CheckpointError(
                "CUDA device count differs from the checkpoint "
                f"(saved={len(cuda_states)}, current={current_device_count})."
            )


def restore_rng(rng_state: Mapping[str, Any]) -> None:
    """Restore RNG states after validating their structure and device compatibility."""
    _validate_rng_state(rng_state, check_cuda_compatibility=True)

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])

    cuda_states = rng_state["torch_cuda"]
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(list(cuda_states))


# ---- Section: Last Checkpoint ----
# Save and restore the complete optimizer and loop state at epoch boundaries.
def _validate_last_payload(payload: Mapping[str, Any]) -> TrainState:
    if not isinstance(payload, Mapping):
        raise CheckpointError("The last checkpoint is not a mapping.")

    missing = _LAST_REQUIRED_KEYS - set(payload)
    if missing:
        raise CheckpointError(
            f"The last checkpoint is missing keys: {sorted(missing)}."
        )

    version = payload["format_version"]
    if type(version) is not int or version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            "Unsupported checkpoint format version "
            f"{version!r}; expected {CHECKPOINT_FORMAT_VERSION}."
        )

    if not isinstance(payload["config_hash"], str) or not payload["config_hash"]:
        raise CheckpointError("The checkpoint config_hash is missing or invalid.")

    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointError("The checkpoint model_state_dict is invalid.")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise CheckpointError("The checkpoint optimizer_state_dict is invalid.")

    scheduler_state = payload["scheduler_state_dict"]
    if scheduler_state is not None and not isinstance(scheduler_state, Mapping):
        raise CheckpointError("The checkpoint scheduler_state_dict is invalid.")

    train_state = TrainState.from_dict(payload["train_state"])
    _validate_rng_state(payload["rng_state"], check_cuda_compatibility=False)
    return train_state


def save_last(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any | None,
    state: TrainState,
    config_hash: str,
) -> Path:
    """Save all state needed to resume after the current epoch."""
    if not isinstance(state, TrainState):
        raise CheckpointError("state must be a TrainState instance.")
    if not isinstance(config_hash, str) or not config_hash:
        raise CheckpointError("config_hash must be a non-empty string.")

    try:
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config_hash": config_hash,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                None if scheduler is None else scheduler.state_dict()
            ),
            "train_state": state.to_dict(),
            "rng_state": capture_rng(),
        }
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("Could not collect the state for last.pt.") from error

    return atomic_torch_save(payload, path)


def load_last(path: str | Path, expected_hash: str) -> dict[str, Any]:
    """Load last.pt on CPU and reject a mismatched configuration hash."""
    checkpoint_path = Path(path)

    if not isinstance(expected_hash, str) or not expected_hash:
        raise CheckpointError("expected_hash must be a non-empty string.")
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Resume checkpoint does not exist: {checkpoint_path}")

    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise CheckpointError(
            f"Could not read resume checkpoint {checkpoint_path}."
        ) from error

    _validate_last_payload(payload)

    actual_hash = payload["config_hash"]
    if actual_hash != expected_hash:
        raise CheckpointError(
            "Resume checkpoint config hash does not match the resolved config "
            f"(saved={actual_hash}, current={expected_hash}). "
            "Use the original config or a new run directory."
        )

    return dict(payload)


def restore_last(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any | None,
) -> TrainState:
    """Restore model, optimizer, scheduler, RNG, and loop state from loaded data."""
    train_state = _validate_last_payload(payload)

    saved_scheduler_state = payload["scheduler_state_dict"]
    if (scheduler is None) != (saved_scheduler_state is None):
        raise CheckpointError(
            "The current scheduler setup does not match the saved checkpoint."
        )

    # Check RNG/device compatibility before mutating the model or optimizer.
    _validate_rng_state(payload["rng_state"], check_cuda_compatibility=True)

    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])

        if scheduler is not None:
            scheduler.load_state_dict(saved_scheduler_state)

        # Restore RNG last, after all state-loading work is complete.
        restore_rng(payload["rng_state"])
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError(
            "Could not restore model, optimizer, scheduler, or RNG state."
        ) from error

    return train_state


# ---- Section: Best Model Weights ----
# Keep best_model.pt limited to model weights, with floating tensors stored as fp32.
def save_best_weights(path: str | Path, model: torch.nn.Module) -> Path:
    """Save only the model state_dict, detached on CPU and in fp32 where applicable."""
    try:
        source_state = model.state_dict()
        cpu_state: dict[str, torch.Tensor] = {}

        for name, value in source_state.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise CheckpointError(
                    "Model state_dict must contain string keys and tensor values."
                )

            cpu_value = value.detach().to(device="cpu")
            if torch.is_floating_point(cpu_value):
                cpu_value = cpu_value.to(dtype=torch.float32)

            # Clone to make this best-checkpoint snapshot independent of later updates.
            cpu_state[name] = cpu_value.clone()

    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("Could not prepare best_model.pt weights.") from error

    return atomic_torch_save(cpu_state, path)


def load_best_weights(path: str | Path, model: torch.nn.Module) -> None:
    """Load the weights-only best checkpoint with strict key and shape checking."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Best model checkpoint does not exist: {checkpoint_path}")

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointError(
            f"Could not read best model checkpoint {checkpoint_path}."
        ) from error

    if not isinstance(state_dict, Mapping):
        raise CheckpointError("best_model.pt must contain a model state_dict.")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise CheckpointError(
            "best_model.pt must contain only string keys and tensor values."
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as error:
        raise CheckpointError(
            "Best model weights do not strictly match the current model."
        ) from error


# ---- Section: Run Status and Cleanup ----
# Use the completion summary as the final marker and last.pt as the resume marker.
def run_status(run_dir: str | Path) -> str:
    """Return 'complete', 'resumable', or 'fresh' for a transformer run directory."""
    directory = Path(run_dir)
    if directory.exists() and not directory.is_dir():
        raise CheckpointError(f"Run path is not a directory: {directory}")

    summary_path = directory / "run_summary.json"
    last_path = directory / "last.pt"

    if summary_path.exists():
        if not summary_path.is_file():
            raise CheckpointError(
                f"Run summary path is not a file: {summary_path}"
            )

        try:
            with summary_path.open("r", encoding="utf-8") as stream:
                summary = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(
                f"Could not read run summary {summary_path}."
            ) from error

        if not isinstance(summary, dict):
            raise CheckpointError(
                f"Run summary must contain a JSON object: {summary_path}"
            )

        status = summary.get("status")
        if not isinstance(status, str) or not status:
            raise CheckpointError(
                f"Run summary has no valid status field: {summary_path}"
            )
        if status == "complete":
            return "complete"

    if last_path.exists() and not last_path.is_file():
        raise CheckpointError(f"Resume checkpoint path is not a file: {last_path}")
    if last_path.is_file():
        return "resumable"

    return "fresh"


def cleanup_after_complete(run_dir: str | Path) -> None:
    """Delete last.pt only after the run has a valid complete summary."""
    directory = Path(run_dir)
    if run_status(directory) != "complete":
        raise CheckpointError(
            f"Refusing to remove last.pt from an incomplete run: {directory}"
        )

    last_path = directory / "last.pt"
    try:
        last_path.unlink(missing_ok=True)
    except OSError as error:
        raise CheckpointError(
            f"Could not remove completed run checkpoint {last_path}."
        ) from error