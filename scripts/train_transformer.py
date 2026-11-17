from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from digikala_review_transformer import config
from digikala_review_transformer.training import (
    checkpoint,
    data,
    loop,
    predict,
    run_config,
    smoke,
    tokenization,
)
from digikala_review_transformer.training import (
    model as model_lib,
)

# ---- Section: Zero-Copy Monitor Subset Helper ----

@dataclass(frozen=True)
class _TokenArraysSubsetView:
    """Zero-copy view over a parent TokenArrays for a subset of row indices."""

    ids: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray


def build_monitor_subset(
    train_split: data.SplitData,
    subset_rows: int,
    seed: int,
) -> data.SplitData:
    """
    Build the fixed train monitor subset (`purpose="monitor"`) directly from
    an in-memory `train_split` without re-reading or re-tokenizing parquet files.
    """
    row_indices = data.subset_row_indices(
        train_split.n_rows,
        int(subset_rows),
        int(seed),
        purpose="monitor",
    )
    parent_arrays = train_split.arrays
    subset_lengths = np.ascontiguousarray(parent_arrays.lengths[row_indices])
    subset_offsets = np.ascontiguousarray(parent_arrays.offsets[row_indices])
    arrays_view = _TokenArraysSubsetView(
        ids=parent_arrays.ids,
        offsets=subset_offsets,
        lengths=subset_lengths,
    )

    subset_y = {
        head: np.ascontiguousarray(train_split.y[head][row_indices])
        for head in data.HEADS
    }
    subset_y_raw = {
        head: np.ascontiguousarray(train_split.y_raw[head][row_indices])
        for head in data.HEADS
    }
    subset_w = {
        head: np.ascontiguousarray(train_split.w[head][row_indices])
        for head in data.HEADS
    }
    subset_stats = {
        "rows": len(row_indices),
        "total_tokens": int(subset_lengths.sum()),
        "max_token_id": int(train_split.stats.get("max_token_id", 0)),
    }
    return data.SplitData(
        split="train_monitor",
        ids=train_split.ids[row_indices].copy(),
        arrays=arrays_view,
        pad_id=int(train_split.pad_id),
        y=subset_y,
        y_raw=subset_y_raw,
        w=subset_w,
        stats=subset_stats,
        origin=train_split.origin,
        source_rows=row_indices,
    )


# ---- Section: CLI Argument Parsing ----

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser defined in plan section 5-11."""
    parser = argparse.ArgumentParser(
        description="Fine-tune multi-task two-head transformers on Digikala reviews.",
    )
    parser.add_argument(
        "--config",
        dest="configs",
        nargs="+",
        action="extend",
        required=True,
        metavar="YAML",
        help="One or more YAML run config paths executed sequentially.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the 9-step smoke test in artifacts/transformer_runs/smoke/ instead of full training.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Optional override for training.batch_size.",
    )
    return parser


def load_and_prepare_config(
    config_path: str | Path,
    *,
    batch_size_override: int | None = None,
) -> dict[str, Any]:
    """Load a YAML config, apply CLI overrides if requested, and validate."""
    resolved_path = Path(config_path).resolve()
    cfg = run_config.load_run_config(resolved_path)
    if batch_size_override is not None:
        if int(batch_size_override) < 1:
            raise run_config.ConfigError(
                f"[FAIL] --batch-size must be >= 1, got {batch_size_override} -> pass a positive integer"
            )
        cfg["training"]["batch_size"] = int(batch_size_override)
    return cfg


# ---- Section: Single-Config Execution ----

def _free_gpu_memory() -> None:
    """Run garbage collection and release cached CUDA blocks between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_config(
    config_path: str | Path,
    *,
    smoke_mode: bool = False,
    batch_size_override: int | None = None,
    smoke_directory: str | Path | None = None,
    require_cuda_in_smoke: bool = True,
) -> int:
    """
    Execute one config in either smoke mode or full training mode.
    Returns 0 on success and non-zero on failure.
    """
    cfg = load_and_prepare_config(
        config_path,
        batch_size_override=batch_size_override,
    )
    run_name = str(cfg["run_name"])

    if smoke_mode:
        cfg_for_smoke = dict(cfg)
        cfg_for_smoke["_config_path"] = str(Path(config_path))
        report = smoke.run_smoke(
            cfg_for_smoke,
            smoke_directory=smoke_directory,
            require_cuda=require_cuda_in_smoke,
        )
        _free_gpu_memory()
        return 0 if report.passed else 1

    target_dir = run_config.run_dir(cfg)
    status = checkpoint.run_status(target_dir)

    if status == "complete":
        print(
            f"[PASS] {run_name}: run is already complete at {target_dir} -> skipping",
            flush=True,
        )
        return 0

    expected_hash = run_config.config_hash(cfg)
    if status == "resumable":
        last_pt = target_dir / predict.LAST_CHECKPOINT_NAME
        payload = checkpoint.load_last(last_pt, expected_hash=expected_hash)
        done_ep = int(payload["state"]["epochs_done"])
        print(
            f"[PASS] {run_name}: found resumable checkpoint at {last_pt} "
            f"(epochs_done={done_ep}, hash={expected_hash[:12]}) -> resuming",
            flush=True,
        )
    else:
        print(
            f"[PASS] {run_name}: starting fresh run at {target_dir} "
            f"(hash={expected_hash[:12]})",
            flush=True,
        )

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    train_split: data.SplitData | None = None
    val_split: data.SplitData | None = None
    test_split: data.SplitData | None = None
    train_subset_split: data.SplitData | None = None
    model_instance: model_lib.TwoHeadClassifier | None = None

    try:
        pack_dir = run_config.project_path(cfg["data"]["pack_dir"])
        manifest = data.open_pack(pack_dir)

        model_path = run_config.project_path(cfg["model"]["pretrained_path"])
        tokenizer_bundle = tokenization.load_tokenizer(
            model_path,
            backend=str(cfg["model"]["tokenizer_backend"]),
            max_length=int(cfg["model"]["max_length"]),
        )
        cache_dir = tokenization.default_cache_dir(tokenizer_bundle)
        variant = str(cfg["training"]["variant"])

        print(
            f"[{run_name}] Loading splits from {pack_dir} "
            f"(variant={variant}, token_cache={cache_dir})...",
            flush=True,
        )
        train_split = data.load_split(
            "train",
            tokenizer_bundle,
            pack_dir,
            manifest,
            variant=variant,
            cache_dir=cache_dir,
            show_progress=True,
        )
        val_split = data.load_split(
            "validation",
            tokenizer_bundle,
            pack_dir,
            manifest,
            variant=variant,
            cache_dir=cache_dir,
            show_progress=True,
        )
        test_split = data.load_split(
            "test",
            tokenizer_bundle,
            pack_dir,
            manifest,
            variant=variant,
            cache_dir=cache_dir,
            show_progress=True,
        )
        train_subset_split = build_monitor_subset(
            train_split,
            subset_rows=int(cfg["training"]["train_subset_eval_rows"]),
            seed=int(cfg.get("seed", config.SPLIT_SEED)),
        )

        model_instance, load_info = model_lib.build_model(cfg)
        max_tok_id = max(
            int(train_split.stats["max_token_id"]),
            int(val_split.stats["max_token_id"]),
            int(test_split.stats["max_token_id"]),
        )
        model_lib.check_token_ids_fit(
            model_instance,
            max_token_id=max_tok_id,
            pad_id=tokenizer_bundle.pad_id,
        )
        print(model_lib.model_summary(model_instance, load_info), flush=True)

        loop_result = loop.fit(
            cfg,
            model_instance,
            train_split,
            val_split,
            train_subset_split,
            run_directory=target_dir,
            device=device,
        )

        finalized = predict.finalize_run(
            cfg,
            model_instance,
            tokenizer_bundle,
            loop_result,
            train_split=train_split,
            validation_split=val_split,
            test_split=test_split,
            train_subset_split=train_subset_split,
            manifest=manifest,
            load_info=load_info,
            device=device,
        )
        print(
            f"[PASS] {run_name}: completed in {loop_result.wall_seconds / 60.0:.1f} min | "
            f"best_epoch={loop_result.best_epoch} | "
            f"best_val_mean_macro_f1={loop_result.best_score:.4f} | "
            f"summary={finalized.summary_path}",
            flush=True,
        )
        return 0
    finally:
        del model_instance, train_split, val_split, test_split, train_subset_split
        _free_gpu_memory()


# ---- Section: Main Sequential Runner ----

def main(
    argv: Sequence[str] | None = None,
    *,
    smoke_directory: str | Path | None = None,
    require_cuda_in_smoke: bool = True,
) -> int:
    """Parse CLI arguments and run each config in sequence, stopping on first error."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_paths = [Path(p) for p in args.configs]
    total = len(config_paths)

    for idx, cfg_path in enumerate(config_paths, start=1):
        print("=" * 72, flush=True)
        print(
            f"RUN [{idx}/{total}]: config={cfg_path} "
            f"(smoke={bool(args.smoke)}, batch_size_override={args.batch_size})",
            flush=True,
        )
        print("=" * 72, flush=True)
        try:
            exit_code = run_single_config(
                cfg_path,
                smoke_mode=bool(args.smoke),
                batch_size_override=args.batch_size,
                smoke_directory=smoke_directory,
                require_cuda_in_smoke=require_cuda_in_smoke,
            )
        except torch.cuda.OutOfMemoryError as error:
            _free_gpu_memory()
            print(
                f"[FAIL] {cfg_path.name}: CUDA OutOfMemoryError ({error}) -> "
                f"re-run with a smaller --batch-size (e.g. --batch-size 64); "
                f"if epoch >= 1 finished, training will auto-resume from last.pt",
                flush=True,
            )
            return 1
        except Exception as error:  # noqa: BLE001
            _free_gpu_memory()
            text = str(error).strip()
            if text.startswith("[FAIL]"):
                print(text, flush=True)
            else:
                print(
                    f"[FAIL] {cfg_path.name}: {type(error).__name__}: {text} -> "
                    f"fix the reported error before re-running",
                    flush=True,
                )
            return 1

        if exit_code != 0:
            print(
                f"[FAIL] {cfg_path.name}: exited with code {exit_code} -> "
                f"stopping sequential execution",
                flush=True,
            )
            return exit_code

    print("=" * 72, flush=True)
    print(f"[PASS] All {total} config(s) finished successfully.", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())