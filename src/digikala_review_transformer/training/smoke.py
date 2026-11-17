from __future__ import annotations

import gc
import importlib
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from digikala_review_transformer import config
from digikala_review_transformer.training import (
    checkpoint,
    data,
    loop,
    losses,
    pack,
    predict,
    run_config,
    tokenization,
)
from digikala_review_transformer.training import (
    model as model_lib,
)

# ---- Section: Constants and Exceptions ----

DEFAULT_SMOKE_DIR = config.ARTIFACTS / "transformer_runs" / "smoke"

DEFAULT_SMOKE_TRAIN_ROWS = 2_000
DEFAULT_SMOKE_VAL_ROWS = 1_000
DEFAULT_SMOKE_TEST_ROWS = 1_000
DEFAULT_SMOKE_MONITOR_ROWS = 500
DEFAULT_MIN_FREE_DISK_GB = 2.0


class SmokeError(RuntimeError):
    """Raised when a smoke check fails critically and prevents subsequent checks."""


@dataclass
class SmokeCheckResult:
    """Outcome of a single smoke check."""

    name: str
    passed: bool
    detail: str
    fix: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class SmokeReport:
    """Complete summary of a smoke-test execution."""

    run_name: str
    config_path: str | None
    smoke_directory: Path
    passed: bool
    checks: list[SmokeCheckResult]
    warnings: list[str]
    peak_memory_mb: float
    train_samples_per_sec: float
    eval_samples_per_sec: float
    estimated_full_epoch_minutes: float
    estimated_full_run_minutes: float


# ---- Section: Formatting and Printing Helpers ----

def format_check_line(check: SmokeCheckResult) -> str:
    """Format a SmokeCheckResult using the project's [PASS] / [FAIL] convention."""
    if check.passed:
        return f"[PASS] {check.name}: {check.detail}"
    fix_text = check.fix or "inspect the traceback and configuration"
    return f"[FAIL] {check.name}: {check.detail} -> {fix_text}"


def _record_check(
    checks: list[SmokeCheckResult],
    all_warnings: list[str],
    check: SmokeCheckResult,
) -> None:
    """Append a check result, collect its warnings, and print immediately."""
    checks.append(check)
    print(format_check_line(check), flush=True)
    for warning_text in check.warnings:
        all_warnings.append(warning_text)
        print(f"[WARN] {check.name}: {warning_text}", flush=True)


def print_smoke_report(report: SmokeReport) -> None:
    """Print the final smoke-test verdict."""
    failed = [check for check in report.checks if not check.passed]
    print("-" * 72, flush=True)
    if not failed and report.passed:
        print(
            f"SMOKE RESULT ({report.run_name}): ALL PASSED "
            f"({len(report.checks)} checks, {len(report.warnings)} warnings)",
            flush=True,
        )
    else:
        print(
            f"SMOKE RESULT ({report.run_name}): {len(failed)} FAILED "
            f"out of {len(report.checks)} checks",
            flush=True,
        )
        for check in failed:
            print(f"  - {format_check_line(check)}", flush=True)
    print("-" * 72, flush=True)


# ---- Section: Individual Check Helpers ----

def _check_environment(
    cfg: dict[str, Any],
    smoke_dir: Path,
    *,
    require_cuda: bool,
    min_free_disk_gb: float,
) -> tuple[SmokeCheckResult, torch.device]:
    """Check 1: Python, PyTorch, CUDA, bf16 support, and free disk space."""
    py_ver = sys.version.split()[0]
    os_label = f"{platform.system()} {platform.release()}"
    torch_ver = torch.__version__

    probe_dir = smoke_dir.parent if smoke_dir.parent.exists() else config.ROOT
    disk_usage = shutil.disk_usage(probe_dir)
    free_gb = disk_usage.free / (1024**3)

    if free_gb < min_free_disk_gb:
        return (
            SmokeCheckResult(
                name="1. Environment & disk",
                passed=False,
                detail=f"free disk space is {free_gb:.2f} GB (< {min_free_disk_gb:.2f} GB required)",
                fix="free up disk space on the drive hosting the repository",
            ),
            torch.device("cpu"),
        )

    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        return (
            SmokeCheckResult(
                name="1. Environment & CUDA",
                passed=False,
                detail=f"CUDA is not available (python={py_ver}, torch={torch_ver}, os={os_label})",
                fix="install the CUDA-enabled PyTorch wheel and verify NVIDIA driver with nvidia-smi",
            ),
            torch.device("cpu"),
        )

    if cuda_available:
        device = torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024**3)
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        wants_bf16 = bool(cfg["training"]["bf16"])
        if wants_bf16 and not bf16_supported:
            return (
                SmokeCheckResult(
                    name="1. Environment & CUDA",
                    passed=False,
                    detail=f"config requests bf16=True but GPU '{props.name}' does not support bf16",
                    fix="use an Ampere+ GPU (e.g. RTX 30/40 series) or set training.bf16=False",
                ),
                device,
            )
        detail = (
            f"os={os_label}, python={py_ver}, torch={torch_ver}, "
            f"gpu='{props.name}' ({vram_gb:.2f} GB), bf16={bf16_supported}, "
            f"free_disk={free_gb:.1f} GB"
        )
        return (
            SmokeCheckResult(
                name="1. Environment & CUDA",
                passed=True,
                detail=detail,
            ),
            device,
        )

    detail = (
        f"os={os_label}, python={py_ver}, torch={torch_ver}, "
        f"device=cpu (CUDA not required), free_disk={free_gb:.1f} GB"
    )
    return (
        SmokeCheckResult(
            name="1. Environment & CPU",
            passed=True,
            detail=detail,
        ),
        torch.device("cpu"),
    )


def _check_training_pack(pack_dir: Path) -> tuple[SmokeCheckResult, dict[str, Any] | None]:
    """Check 2: Verify training pack files, sha256 checksums, row counts, labels, and weights."""
    try:
        manifest, results = pack.verify_pack(pack_dir, check_content=True)
    except Exception as error:  # noqa: BLE001
        return (
            SmokeCheckResult(
                name="2. Training pack integrity",
                passed=False,
                detail=f"{type(error).__name__}: {error}",
                fix="rebuild or re-extract data/training_pack and verify manifest.json checksums",
            ),
            None,
        )

    failed_messages = [msg for passed, msg in results if not passed]
    if failed_messages:
        return (
            SmokeCheckResult(
                name="2. Training pack integrity",
                passed=False,
                detail="; ".join(failed_messages),
                fix="rebuild or re-extract data/training_pack and verify manifest.json checksums",
            ),
            None,
        )

    fp = pack.pack_fingerprint(manifest)
    split_rows = {
        split: int(manifest["files"][pack.pack_file_name(split)]["rows"])
        for split in pack.SPLITS
    }
    detail = (
        f"fingerprint={fp}, rows(train={split_rows['train']:,}, "
        f"val={split_rows['validation']:,}, test={split_rows['test']:,})"
    )
    return (
        SmokeCheckResult(
            name="2. Training pack integrity",
            passed=True,
            detail=detail,
        ),
        manifest,
    )


def run_mdeberta_tokenizer_checks(
    model_path: Path,
    bundle: tokenization.TokenizerBundle,
    sample_texts: list[str],
) -> list[SmokeCheckResult]:
    """
    Run the 3 mDeBERTa-specific tokenizer checks required by the plan:
      1. Special token IDs match HF (first=1, last=2, pad=0, unk=3).
      2. Raw SentencePiece token IDs equal HF fast token IDs on all rows without [UNK].
      3. Count rows containing [UNK] under SentencePiece vs HF fast (reported as [WARN]).
    """
    results: list[SmokeCheckResult] = []

    expected_specials = {"first_id": 1, "last_id": 2, "pad_id": 0, "unk_id": 3}
    actual_specials = {
        "first_id": int(bundle.first_id),
        "last_id": int(bundle.last_id),
        "pad_id": int(bundle.pad_id),
        "unk_id": int(bundle.unk_id),
    }

    try:
        hf_tok = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    except Exception as error:  # noqa: BLE001
        results.append(
            SmokeCheckResult(
                name="3b-1. mDeBERTa special token IDs",
                passed=False,
                detail=f"could not load HF fast tokenizer from {model_path}: {error}",
                fix="re-download microsoft/mdeberta-v3-base into models/pretrained/mdeberta-v3-base",
            )
        )
        return results

    hf_specials = {
        "first_id": int(hf_tok.cls_token_id),
        "last_id": int(hf_tok.sep_token_id),
        "pad_id": int(hf_tok.pad_token_id),
        "unk_id": int(hf_tok.unk_token_id),
    }

    specials_ok = (actual_specials == expected_specials) and (hf_specials == expected_specials)
    if specials_ok:
        results.append(
            SmokeCheckResult(
                name="3b-1. mDeBERTa special token IDs",
                passed=True,
                detail=f"SentencePiece and HF special IDs match {expected_specials}",
            )
        )
    else:
        results.append(
            SmokeCheckResult(
                name="3b-1. mDeBERTa special token IDs",
                passed=False,
                detail=f"expected {expected_specials}, got spm={actual_specials}, hf={hf_specials}",
                fix="verify spm.model and tokenizer_config.json in models/pretrained/mdeberta-v3-base",
            )
        )

    sp_lists = bundle.encode(sample_texts)
    hf_encoded = hf_tok(
        sample_texts,
        add_special_tokens=True,
        truncation=True,
        max_length=int(bundle.max_length),
        padding=False,
        return_attention_mask=False,
    )["input_ids"]

    unk_id = int(bundle.unk_id)
    non_unk_rows = 0
    mismatch_rows = 0
    sp_unk_rows = 0
    hf_unk_rows = 0

    for sp_row, hf_row in zip(sp_lists, hf_encoded, strict=True):
        sp_ids = [int(x) for x in sp_row]
        hf_ids = [int(x) for x in hf_row]
        if unk_id in sp_ids:
            sp_unk_rows += 1
        if unk_id in hf_ids:
            hf_unk_rows += 1
        else:
            non_unk_rows += 1
            if sp_ids != hf_ids:
                mismatch_rows += 1

    if mismatch_rows == 0 and non_unk_rows > 0:
        results.append(
            SmokeCheckResult(
                name="3b-2. mDeBERTa SentencePiece vs HF parity (non-UNK rows)",
                passed=True,
                detail=(
                    f"100% identical across all {non_unk_rows:,} non-[UNK] rows "
                    f"(out of {len(sample_texts):,} sampled rows)"
                ),
            )
        )
    else:
        results.append(
            SmokeCheckResult(
                name="3b-2. mDeBERTa SentencePiece vs HF parity (non-UNK rows)",
                passed=False,
                detail=(
                    f"{mismatch_rows:,} mismatches across {non_unk_rows:,} non-[UNK] rows "
                    f"(out of {len(sample_texts):,} sampled rows)"
                ),
                fix="check SentencePiecePiece offset (+1) and special-token wrapping in tokenization.py",
            )
        )

    unk_warning = (
        f"rows with [UNK] on {len(sample_texts):,} sampled texts: "
        f"sentencepiece={sp_unk_rows:,} ({100.0 * sp_unk_rows / max(len(sample_texts), 1):.2f}%), "
        f"hf_fast={hf_unk_rows:,} ({100.0 * hf_unk_rows / max(len(sample_texts), 1):.2f}%)"
    )
    results.append(
        SmokeCheckResult(
            name="3b-3. mDeBERTa [UNK] row count",
            passed=True,
            detail=f"counted [UNK] rows on {len(sample_texts):,} texts",
            warnings=[unk_warning],
        )
    )
    return results


def _check_worst_case_batch(
    cfg: dict[str, Any],
    model: model_lib.TwoHeadClassifier,
    y_train: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[SmokeCheckResult, float]:
    """
    Check 6: Run forward, backward, grad clip, and optimizer step on a synthetic
    batch of full (batch_size, max_length) and record peak CUDA memory in MB.
    """
    batch_size = int(cfg["training"]["batch_size"])
    max_length = int(cfg["model"]["max_length"])
    use_bf16 = bool(cfg["training"]["bf16"]) and device.type == "cuda"

    try:
        model.to(device)
        model.train()

        loss_fn = losses.build_loss(cfg, y_train, device)
        param_groups = model_lib.parameter_groups(
            model,
            lr_encoder=float(cfg["training"]["lr_encoder"]),
            lr_heads=float(cfg["training"]["lr_heads"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        use_fused = device.type == "cuda"
        try:
            optimizer = torch.optim.AdamW(param_groups, fused=use_fused)
        except TypeError:
            optimizer = torch.optim.AdamW(param_groups)
        optimizer.zero_grad(set_to_none=True)

        # Warm up optimizer state tensors (exp_avg, exp_avg_sq) on a tiny batch so
        # the worst-case measurement reflects steady-state training memory (Step >= 2).
        warm_ids = torch.full((1, 8), fill_value=10, dtype=torch.long, device=device)
        warm_mask = torch.ones((1, 8), dtype=torch.long, device=device)
        warm_batch: dict[str, torch.Tensor] = {
            "input_ids": warm_ids,
            "attention_mask": warm_mask,
            "y_rec": torch.zeros(1, dtype=torch.long, device=device),
            "y_band": torch.zeros(1, dtype=torch.long, device=device),
            "w_rec": torch.ones(1, dtype=torch.float32, device=device),
            "w_band": torch.ones(1, dtype=torch.float32, device=device),
        }
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            warm_logits = model(input_ids=warm_ids, attention_mask=warm_mask)
            warm_loss = loss_fn(warm_logits, warm_batch)
        warm_loss.total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del warm_ids, warm_mask, warm_batch, warm_logits, warm_loss

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
        input_ids = torch.full(
            (batch_size, max_length),
            fill_value=10,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones(
            (batch_size, max_length),
            dtype=torch.long,
            device=device,
        )
        batch_tensors: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "y_rec": torch.arange(batch_size, dtype=torch.long, device=device) % 3,
            "y_band": torch.arange(batch_size, dtype=torch.long, device=device) % 3,
            "w_rec": torch.ones(batch_size, dtype=torch.float32, device=device),
            "w_band": torch.ones(batch_size, dtype=torch.float32, device=device),
        }

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits_by_head = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_out = loss_fn(logits_by_head, batch_tensors)

        loss_value = float(loss_out.total.detach().item())
        if not np.isfinite(loss_value):
            return (
                SmokeCheckResult(
                    name="6. Worst-case batch (forward/backward/step)",
                    passed=False,
                    detail=f"non-finite loss {loss_value} on shape=({batch_size}, {max_length})",
                    fix="check learning rates, label smoothing, and model initialization",
                ),
                0.0,
            )

        loss_out.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(cfg["training"]["max_grad_norm"]),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        peak_mb = 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        detail = (
            f"shape=({batch_size}, {max_length}), loss={loss_value:.4f}, "
            f"peak_vram={peak_mb:.1f} MB ({peak_mb / 1024.0:.2f} GB)"
        )
        return (
            SmokeCheckResult(
                name="6. Worst-case batch (forward/backward/step)",
                passed=True,
                detail=detail,
            ),
            peak_mb,
        )
    except torch.cuda.OutOfMemoryError as error:
        return (
            SmokeCheckResult(
                name="6. Worst-case batch (forward/backward/step)",
                passed=False,
                detail=f"CUDA OutOfMemoryError at shape=({batch_size}, {max_length}): {error}",
                fix="pass a smaller --batch-size (e.g. --batch-size 64 or --batch-size 32)",
            ),
            0.0,
        )
    except Exception as error:  # noqa: BLE001
        return (
            SmokeCheckResult(
                name="6. Worst-case batch (forward/backward/step)",
                passed=False,
                detail=f"{type(error).__name__}: {error}",
                fix="inspect model forward/backward and CUDA installation",
            ),
            0.0,
        )


# ---- Section: Main Smoke Runner ----

def run_smoke(
    cfg: dict[str, Any],
    *,
    smoke_directory: str | Path | None = None,
    train_rows: int = DEFAULT_SMOKE_TRAIN_ROWS,
    val_rows: int = DEFAULT_SMOKE_VAL_ROWS,
    test_rows: int = DEFAULT_SMOKE_TEST_ROWS,
    monitor_rows: int = DEFAULT_SMOKE_MONITOR_ROWS,
    num_workers: int | None = None,
    require_cuda: bool = True,
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB,
) -> SmokeReport:
    """
    Execute the complete 9-step (+ 3 mDeBERTa) smoke test in `artifacts/transformer_runs/smoke/`.
    """
    smoke_dir = (
        Path(smoke_directory).resolve()
        if smoke_directory is not None
        else DEFAULT_SMOKE_DIR.resolve()
    )
    run_name = str(cfg.get("run_name", "smoke"))
    cfg_path = cfg.get("_config_path")

    checks: list[SmokeCheckResult] = []
    warnings_list: list[str] = []
    peak_memory_mb = 0.0
    train_sps = 0.0
    eval_sps = 0.0
    est_epoch_minutes = 0.0
    est_run_minutes = 0.0

    model_label = Path(str(cfg["model"]["pretrained_path"])).name
    print("=" * 72, flush=True)
    print(
        f"STARTING SMOKE TEST: run_name={run_name}, "
        f"model={model_label}, variant={cfg['training']['variant']}, "
        f"batch_size={cfg['training']['batch_size']}, max_length={cfg['model']['max_length']}",
        flush=True,
    )
    print(f"Smoke output directory: {smoke_dir}", flush=True)
    print("=" * 72, flush=True)

    # ---- Check 1: Environment, CUDA, bf16, and Disk ----
    check_1, device = _check_environment(
        cfg,
        smoke_dir,
        require_cuda=require_cuda,
        min_free_disk_gb=min_free_disk_gb,
    )
    _record_check(checks, warnings_list, check_1)
    if not check_1.passed:
        report = SmokeReport(
            run_name=run_name,
            config_path=cfg_path,
            smoke_directory=smoke_dir,
            passed=False,
            checks=checks,
            warnings=warnings_list,
            peak_memory_mb=peak_memory_mb,
            train_samples_per_sec=train_sps,
            eval_samples_per_sec=eval_sps,
            estimated_full_epoch_minutes=est_epoch_minutes,
            estimated_full_run_minutes=est_run_minutes,
        )
        print_smoke_report(report)
        return report

    # ---- Check 2: Training Pack Integrity ----
    pack_dir = run_config.project_path(cfg["data"]["pack_dir"])
    check_2, manifest = _check_training_pack(pack_dir)
    _record_check(checks, warnings_list, check_2)
    if not check_2.passed or manifest is None:
        report = SmokeReport(
            run_name=run_name,
            config_path=cfg_path,
            smoke_directory=smoke_dir,
            passed=False,
            checks=checks,
            warnings=warnings_list,
            peak_memory_mb=peak_memory_mb,
            train_samples_per_sec=train_sps,
            eval_samples_per_sec=eval_sps,
            estimated_full_epoch_minutes=est_epoch_minutes,
            estimated_full_run_minutes=est_run_minutes,
        )
        print_smoke_report(report)
        return report

    # ---- Check 3: Model and Tokenizer Load ----
    model_path = run_config.project_path(cfg["model"]["pretrained_path"])
    backend = str(cfg["model"]["tokenizer_backend"])
    max_length = int(cfg["model"]["max_length"])
    is_mdeberta = (backend == "sentencepiece") or ("mdeberta" in model_path.name.lower())

    if is_mdeberta:
        missing_pkgs: list[str] = []
        for mod_name in ("sentencepiece", "google.protobuf"):
            try:
                importlib.import_module(mod_name)
            except Exception:  # noqa: BLE001
                missing_pkgs.append(mod_name)
        if missing_pkgs:
            _record_check(
                checks,
                warnings_list,
                SmokeCheckResult(
                    name="3. Model & tokenizer load",
                    passed=False,
                    detail=f"missing required packages for mDeBERTa: {missing_pkgs}",
                    fix="run `uv sync` to install sentencepiece and protobuf",
                ),
            )
            report = SmokeReport(
                run_name=run_name,
                config_path=cfg_path,
                smoke_directory=smoke_dir,
                passed=False,
                checks=checks,
                warnings=warnings_list,
                peak_memory_mb=peak_memory_mb,
                train_samples_per_sec=train_sps,
                eval_samples_per_sec=eval_sps,
                estimated_full_epoch_minutes=est_epoch_minutes,
                estimated_full_run_minutes=est_run_minutes,
            )
            print_smoke_report(report)
            return report

    try:
        tokenizer_bundle = tokenization.load_tokenizer(
            model_path,
            backend=backend,
            max_length=max_length,
        )
        model_instance, load_info = model_lib.build_model(cfg)
        param_counts = model_lib.count_parameters(model_instance)
        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="3. Model & tokenizer load",
                passed=True,
                detail=(
                    f"backend={tokenizer_bundle.backend}, model={tokenizer_bundle.name}, "
                    f"params(total={param_counts['total']:,}, trainable={param_counts['trainable']:,}), "
                    f"ignored_unexpected={load_info.get('ignored_unexpected_keys', [])}"
                ),
            ),
        )
    except Exception as error:  # noqa: BLE001
        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="3. Model & tokenizer load",
                passed=False,
                detail=f"{type(error).__name__}: {error}",
                fix=f"verify pretrained model files in {model_path} or re-run scripts/download_pretrained_models.py",
            ),
        )
        report = SmokeReport(
            run_name=run_name,
            config_path=cfg_path,
            smoke_directory=smoke_dir,
            passed=False,
            checks=checks,
            warnings=warnings_list,
            peak_memory_mb=peak_memory_mb,
            train_samples_per_sec=train_sps,
            eval_samples_per_sec=eval_sps,
            estimated_full_epoch_minutes=est_epoch_minutes,
            estimated_full_run_minutes=est_run_minutes,
        )
        print_smoke_report(report)
        return report

    # ---- Check 3b: mDeBERTa Extra Checks (when applicable) ----
    if is_mdeberta:
        import pyarrow.parquet as pq

        val_parquet_path = pack_dir / pack.pack_file_name("validation")
        val_total_rows = int(manifest["files"][pack.pack_file_name("validation")]["rows"])
        sample_idx = data.subset_row_indices(
            val_total_rows,
            min(val_rows, val_total_rows),
            config.SPLIT_SEED,
            purpose="smoke",
        )
        sample_table = pq.read_table(val_parquet_path, columns=["body"]).take(sample_idx)
        sample_texts = [text if text is not None else "" for text in sample_table.column("body").to_pylist()]

        for mdeb_check in run_mdeberta_tokenizer_checks(model_path, tokenizer_bundle, sample_texts):
            _record_check(checks, warnings_list, mdeb_check)

    # ---- Check 4: Token Cache Build and Re-read in a Temporary Directory ----
    try:
        with tempfile.TemporaryDirectory(prefix="dk_smoke_tok_cache_") as tmp_dir_str:
            tmp_cache_dir = Path(tmp_dir_str)
            smoke_cache_texts = [
                "این محصول کیفیت بسیار خوبی دارد و خرید آن را پیشنهاد می‌کنم.",
                "بسته‌بندی مناسب نبود اما نسبت به قیمت ارزش خرید داشت.",
                "اصلا راضی نبودم؛ بعد از دو روز خراب شد و مرجوع کردم.",
                "کیفیت معمولی\u200cتر از انتظارم بود ولی کار راه\u200cانداز است.",
            ] * 16
            fake_sha = "0123456789abcdef" * 4

            arrays_built, _, origin_1 = tokenization.get_split_tokens(
                tokenizer_bundle,
                split="validation",
                source_sha256=fake_sha,
                load_texts=lambda: smoke_cache_texts,
                expected_rows=len(smoke_cache_texts),
                cache_dir=tmp_cache_dir,
                show_progress=False,
            )
            arrays_cached, _, origin_2 = tokenization.get_split_tokens(
                tokenizer_bundle,
                split="validation",
                source_sha256=fake_sha,
                load_texts=lambda: (_ for _ in ()).throw(
                    RuntimeError("load_texts should not be called on a cache hit")
                ),
                expected_rows=len(smoke_cache_texts),
                cache_dir=tmp_cache_dir,
                show_progress=False,
            )

            ids_equal = np.array_equal(arrays_built.ids, arrays_cached.ids)
            offsets_equal = np.array_equal(arrays_built.offsets, arrays_cached.offsets)
            origins_ok = (origin_1 == "built") and (origin_2 == "cache")

            if ids_equal and offsets_equal and origins_ok:
                _record_check(
                    checks,
                    warnings_list,
                    SmokeCheckResult(
                        name="4. Temporary token cache build & re-read",
                        passed=True,
                        detail=(
                            f"origin=({origin_1} -> {origin_2}), "
                            f"rows={len(arrays_built.lengths)}, tokens={int(arrays_built.ids.size)}"
                        ),
                    ),
                )
            else:
                _record_check(
                    checks,
                    warnings_list,
                    SmokeCheckResult(
                        name="4. Temporary token cache build & re-read",
                        passed=False,
                        detail=(
                            f"ids_equal={ids_equal}, offsets_equal={offsets_equal}, "
                            f"origins=({origin_1}, {origin_2})"
                        ),
                        fix="inspect token cache serialization in tokenization.py",
                    ),
                )
    except Exception as error:  # noqa: BLE001
        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="4. Temporary token cache build & re-read",
                passed=False,
                detail=f"{type(error).__name__}: {error}",
                fix="check write permissions in system temp directory and tokenization.py",
            ),
        )

    # ---- Load Smoke Splits (no disk token cache: cache_dir=None) ----
    try:
        train_split_B = data.load_split(
            "train",
            tokenizer_bundle,
            pack_dir,
            manifest,
            variant="B",
            subset_rows=train_rows,
            subset_seed=config.SPLIT_SEED,
            cache_dir=None,
            show_progress=False,
        )
        train_split_C = data.load_split(
            "train",
            tokenizer_bundle,
            pack_dir,
            manifest,
            variant="C",
            subset_rows=train_rows,
            subset_seed=config.SPLIT_SEED,
            cache_dir=None,
            show_progress=False,
        )
        variant = str(cfg["training"]["variant"])
        train_split = train_split_C if variant == "C" else train_split_B

        val_split = data.load_split(
            "validation",
            tokenizer_bundle,
            pack_dir,
            manifest,
            variant=variant,
            subset_rows=val_rows,
            subset_seed=config.SPLIT_SEED,
            cache_dir=None,
            show_progress=False,
        )
        test_split = data.load_split(
            "test",
            tokenizer_bundle,
            pack_dir,
            manifest,
            variant=variant,
            subset_rows=test_rows,
            subset_seed=config.SPLIT_SEED,
            cache_dir=None,
            show_progress=False,
        )
        monitor_subset_size = min(monitor_rows, train_split.n_rows)
        train_monitor_split = data.load_split(
            "train",
            tokenizer_bundle,
            pack_dir,
            manifest,
            variant=variant,
            subset_rows=monitor_subset_size,
            subset_seed=config.SPLIT_SEED,
            cache_dir=None,
            show_progress=False,
        )
    except Exception as error:  # noqa: BLE001
        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="5. Smoke splits & B/C batch order equivalence",
                passed=False,
                detail=f"failed to load smoke splits: {type(error).__name__}: {error}",
                fix="inspect data.load_split and training pack files",
            ),
        )
        report = SmokeReport(
            run_name=run_name,
            config_path=cfg_path,
            smoke_directory=smoke_dir,
            passed=False,
            checks=checks,
            warnings=warnings_list,
            peak_memory_mb=peak_memory_mb,
            train_samples_per_sec=train_sps,
            eval_samples_per_sec=eval_sps,
            estimated_full_epoch_minutes=est_epoch_minutes,
            estimated_full_run_minutes=est_run_minutes,
        )
        print_smoke_report(report)
        return report

    # ---- Check 5: Variant B and C Batch Order Equivalence + Token ID Fit ----
    try:
        batch_size = int(cfg["training"]["batch_size"])
        sampler_B = data.LengthBucketedBatches(
            train_split_B.lengths,
            batch_size=batch_size,
            seed=config.SPLIT_SEED,
        )
        sampler_C = data.LengthBucketedBatches(
            train_split_C.lengths,
            batch_size=batch_size,
            seed=config.SPLIT_SEED,
        )
        sampler_B.set_epoch(0)
        sampler_C.set_epoch(0)
        fp_B_ep0 = data.batch_fingerprint(list(sampler_B), n_batches=5)
        fp_C_ep0 = data.batch_fingerprint(list(sampler_C), n_batches=5)

        sampler_B.set_epoch(1)
        sampler_C.set_epoch(1)
        fp_B_ep1 = data.batch_fingerprint(list(sampler_B), n_batches=5)
        fp_C_ep1 = data.batch_fingerprint(list(sampler_C), n_batches=5)

        ids_identical = np.array_equal(train_split_B.ids, train_split_C.ids)
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

        if ids_identical and (fp_B_ep0 == fp_C_ep0) and (fp_B_ep1 == fp_C_ep1) and (fp_B_ep0 != fp_B_ep1):
            _record_check(
                checks,
                warnings_list,
                SmokeCheckResult(
                    name="5. Variant B/C batch order equivalence",
                    passed=True,
                    detail=(
                        f"B and C batch fingerprints match (ep0={fp_B_ep0}, ep1={fp_B_ep1}), "
                        f"max_token_id={max_tok_id} fits embedding table"
                    ),
                ),
            )
        else:
            _record_check(
                checks,
                warnings_list,
                SmokeCheckResult(
                    name="5. Variant B/C batch order equivalence",
                    passed=False,
                    detail=(
                        f"ids_identical={ids_identical}, ep0=({fp_B_ep0} vs {fp_C_ep0}), "
                        f"ep1=({fp_B_ep1} vs {fp_C_ep1})"
                    ),
                    fix="ensure LengthBucketedBatches depends only on (seed, epoch, lengths)",
                ),
            )
    except Exception as error:  # noqa: BLE001
        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="5. Variant B/C batch order equivalence",
                passed=False,
                detail=f"{type(error).__name__}: {error}",
                fix="check LengthBucketedBatches and model embedding size",
            ),
        )
    finally:
        del train_split_B, train_split_C

    # ---- Check 6: Worst-Case Synthetic Batch (full batch_size, full max_length) ----
    check_6, peak_memory_mb = _check_worst_case_batch(
        cfg,
        model_instance,
        train_split.y,
        device,
    )
    _record_check(checks, warnings_list, check_6)

    # Free worst-case batch model and clear CUDA cache before 2-epoch training
    del model_instance
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not check_6.passed:
        report = SmokeReport(
            run_name=run_name,
            config_path=cfg_path,
            smoke_directory=smoke_dir,
            passed=False,
            checks=checks,
            warnings=warnings_list,
            peak_memory_mb=peak_memory_mb,
            train_samples_per_sec=train_sps,
            eval_samples_per_sec=eval_sps,
            estimated_full_epoch_minutes=est_epoch_minutes,
            estimated_full_run_minutes=est_run_minutes,
        )
        print_smoke_report(report)
        return report

    # ---- Check 7: 2-Epoch Training + Save after Epoch 1 + Auto-Resume Epoch 2 ----
    smoke_cfg = {
        "run_name": run_name,
        "seed": int(cfg.get("seed", config.SPLIT_SEED)),
        "output": {"root": str(smoke_dir.parent)},
        "data": dict(cfg["data"]),
        "model": dict(cfg["model"]),
        "training": dict(cfg["training"]),
    }
    smoke_cfg["training"]["max_epochs"] = 2
    smoke_cfg["training"]["early_stopping_patience"] = 2
    smoke_cfg["training"]["early_stopping_min_delta"] = 0.0
    if num_workers is not None:
        smoke_cfg["training"]["num_workers"] = int(num_workers)

    # When batch_size is reduced (e.g. via --batch-size on an 8 GB GPU), scale down
    # eval_batch_size proportionally so evaluation does not exceed available VRAM.
    configured_bs = int(smoke_cfg["training"]["batch_size"])
    configured_eval_bs = int(smoke_cfg["training"]["eval_batch_size"])
    if configured_bs < 128 and configured_eval_bs > configured_bs * 2:
        smoke_cfg["training"]["eval_batch_size"] = max(configured_bs * 2, 64)

    steps_per_ep = max(1, (train_split.n_rows + configured_bs - 1) // configured_bs)
    smoke_cfg["training"]["log_every_steps"] = max(1, min(5, steps_per_ep))

    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    loop_result_final: loop.LoopResult | None = None
    model_for_finalize: model_lib.TwoHeadClassifier | None = None
    eval_wall_seconds = 0.0

    try:
        # Phase 1: Train Epoch 1 only and stop to leave a resumable checkpoint
        model_ep1, load_info = model_lib.build_model(smoke_cfg)
        res_ep1 = loop.fit(
            smoke_cfg,
            model_ep1,
            train_split,
            val_split,
            train_monitor_split,
            run_directory=smoke_dir,
            device=device,
            max_epochs_this_call=1,
        )
        status_after_ep1 = checkpoint.run_status(smoke_dir)
        last_pt_exists = (smoke_dir / predict.LAST_CHECKPOINT_NAME).exists()

        del model_ep1
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if res_ep1.epochs_done != 1 or status_after_ep1 != "resumable" or not last_pt_exists:
            _record_check(
                checks,
                warnings_list,
                SmokeCheckResult(
                    name="7. 2-epoch training & checkpoint auto-resume",
                    passed=False,
                    detail=(
                        f"after epoch 1: epochs_done={res_ep1.epochs_done}, "
                        f"status={status_after_ep1}, last.pt={last_pt_exists}"
                    ),
                    fix="inspect checkpoint.save_last and loop.fit max_epochs_this_call handling",
                ),
            )
        else:
            # Phase 2: Build a fresh model and auto-resume from last.pt to complete Epoch 2
            model_ep2, load_info = model_lib.build_model(smoke_cfg)
            res_ep2 = loop.fit(
                smoke_cfg,
                model_ep2,
                train_split,
                val_split,
                train_monitor_split,
                run_directory=smoke_dir,
                device=device,
                max_epochs_this_call=None,
            )
            loop_result_final = res_ep2
            model_for_finalize = model_ep2

            history_rows = res_ep2.history
            epochs_recorded = [int(row["epoch"]) for row in history_rows]
            losses_finite = all(
                np.isfinite(float(row["train_objective"]))
                and np.isfinite(float(row["val_plain_ce_rec"]))
                and np.isfinite(float(row["val_plain_ce_band"]))
                and np.isfinite(float(row["val_mean_macro_f1"]))
                for row in history_rows
            )

            ep1_loss = float(history_rows[0]["train_objective"]) if len(history_rows) >= 1 else float("nan")
            ep2_loss = float(history_rows[-1]["train_objective"]) if len(history_rows) >= 2 else float("nan")

            check7_warnings: list[str] = []
            if np.isfinite(ep1_loss) and np.isfinite(ep2_loss) and ep2_loss >= ep1_loss:
                check7_warnings.append(
                    f"train_objective did not decrease on small smoke subset "
                    f"(epoch1={ep1_loss:.4f} -> epoch2={ep2_loss:.4f})"
                )

            resumed_cleanly = (
                res_ep2.epochs_done == 2
                and epochs_recorded == [1, 2]
                and res_ep2.global_step == 2 * steps_per_ep
                and losses_finite
            )

            if resumed_cleanly:
                _record_check(
                    checks,
                    warnings_list,
                    SmokeCheckResult(
                        name="7. 2-epoch training & checkpoint auto-resume",
                        passed=True,
                        detail=(
                            f"resumed epoch 1 -> 2 (steps={res_ep2.global_step}), "
                            f"train_objective=({ep1_loss:.4f} -> {ep2_loss:.4f}), "
                            f"best_val_mean_macro_f1={res_ep2.best_score:.4f} (epoch {res_ep2.best_epoch})"
                        ),
                        warnings=check7_warnings,
                    ),
                )
            else:
                _record_check(
                    checks,
                    warnings_list,
                    SmokeCheckResult(
                        name="7. 2-epoch training & checkpoint auto-resume",
                        passed=False,
                        detail=(
                            f"epochs_done={res_ep2.epochs_done}, epochs={epochs_recorded}, "
                            f"global_step={res_ep2.global_step}, losses_finite={losses_finite}"
                        ),
                        fix="inspect checkpoint restore and history tracking in loop.py",
                    ),
                )
    except Exception as error:  # noqa: BLE001
        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="7. 2-epoch training & checkpoint auto-resume",
                passed=False,
                detail=f"{type(error).__name__}: {error}",
                fix="inspect loop.fit and checkpoint.py",
            ),
        )

    # ---- Check 8: Finalize Run & Verify Prediction Alignment with `id` ----
    if loop_result_final is not None and model_for_finalize is not None:
        try:
            eval_t0 = time.perf_counter()
            finalized = predict.finalize_run(
                smoke_cfg,
                model_for_finalize,
                tokenizer_bundle,
                loop_result_final,
                train_split=train_split,
                validation_split=val_split,
                test_split=test_split,
                train_subset_split=train_monitor_split,
                manifest=manifest,
                load_info=load_info,
                device=device,
            )
            eval_wall_seconds = max(time.perf_counter() - eval_t0, 1e-6)

            val_df, _ = predict.read_and_verify_probability_parquet(
                finalized.validation_probabilities_path,
                val_split,
            )
            test_df, _ = predict.read_and_verify_probability_parquet(
                finalized.test_probabilities_path,
                test_split,
            )
            val_ids_match = np.array_equal(val_df["id"].to_numpy(), np.asarray(val_split.ids))
            test_ids_match = np.array_equal(test_df["id"].to_numpy(), np.asarray(test_split.ids))
            final_status = checkpoint.run_status(smoke_dir)
            last_removed = not (smoke_dir / predict.LAST_CHECKPOINT_NAME).exists()

            if val_ids_match and test_ids_match and final_status == "complete" and last_removed:
                _record_check(
                    checks,
                    warnings_list,
                    SmokeCheckResult(
                        name="8. Predictions & run finalization",
                        passed=True,
                        detail=(
                            f"val_probs ({len(val_df):,} rows) and test_probs ({len(test_df):,} rows) "
                            f"aligned with id; status={final_status}; last.pt removed"
                        ),
                    ),
                )
            else:
                _record_check(
                    checks,
                    warnings_list,
                    SmokeCheckResult(
                        name="8. Predictions & run finalization",
                        passed=False,
                        detail=(
                            f"val_ids_match={val_ids_match}, test_ids_match={test_ids_match}, "
                            f"status={final_status}, last_removed={last_removed}"
                        ),
                        fix="inspect predict.finalize_run and sorted_eval_batches restoration",
                    ),
                )
        except Exception as error:  # noqa: BLE001
            _record_check(
                checks,
                warnings_list,
                SmokeCheckResult(
                    name="8. Predictions & run finalization",
                    passed=False,
                    detail=f"{type(error).__name__}: {error}",
                    fix="inspect predict.finalize_run and probability parquet verification",
                ),
            )
    else:
        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="8. Predictions & run finalization",
                passed=False,
                detail="skipped because Check 7 did not complete",
                fix="resolve Check 7 failure first",
            ),
        )

    # ---- Check 9: Throughput (samples/sec) & Full-Epoch Wall-Time Estimate ----
    if loop_result_final is not None and loop_result_final.train_seconds > 0:
        total_smoke_train_samples = float(train_split.n_rows * loop_result_final.epochs_done)
        train_sps = total_smoke_train_samples / float(loop_result_final.train_seconds)

        # Estimate evaluation throughput from epoch wall_seconds minus train_seconds
        total_epoch_wall = sum(float(row["epoch_seconds"]) for row in loop_result_final.history)
        eval_only_seconds = max(total_epoch_wall - float(loop_result_final.train_seconds), eval_wall_seconds, 1e-3)
        eval_rows_per_epoch = float(val_split.n_rows + train_monitor_split.n_rows)
        total_smoke_eval_rows = eval_rows_per_epoch * float(loop_result_final.epochs_done)
        eval_sps = total_smoke_eval_rows / eval_only_seconds

        full_train_rows = float(manifest["files"][pack.pack_file_name("train")]["rows"])
        full_val_rows = float(manifest["files"][pack.pack_file_name("validation")]["rows"])
        full_monitor_rows = float(min(int(cfg["training"]["train_subset_eval_rows"]), int(full_train_rows)))

        est_train_sec = full_train_rows / max(train_sps, 1e-6)
        est_eval_sec = (full_val_rows + full_monitor_rows) / max(eval_sps, 1e-6)
        est_epoch_minutes = (est_train_sec + est_eval_sec) / 60.0
        configured_max_epochs = int(cfg["training"]["max_epochs"])
        est_run_minutes = est_epoch_minutes * configured_max_epochs

        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="9. Throughput & full-epoch time estimate",
                passed=True,
                detail=(
                    f"train={train_sps:.1f} samples/s, eval={eval_sps:.1f} samples/s -> "
                    f"est 1 full epoch ({int(full_train_rows):,} train + {int(full_val_rows):,} val): "
                    f"{est_epoch_minutes:.1f} min "
                    f"(~{est_run_minutes:.1f} min for {configured_max_epochs} epochs)"
                ),
            ),
        )
    else:
        _record_check(
            checks,
            warnings_list,
            SmokeCheckResult(
                name="9. Throughput & full-epoch time estimate",
                passed=False,
                detail="could not compute throughput because training did not complete",
                fix="resolve training failure in Check 7",
            ),
        )

    # Clean up model and GPU memory
    del model_for_finalize
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_passed = all(check.passed for check in checks)
    report = SmokeReport(
        run_name=run_name,
        config_path=cfg_path,
        smoke_directory=smoke_dir,
        passed=all_passed,
        checks=checks,
        warnings=warnings_list,
        peak_memory_mb=peak_memory_mb,
        train_samples_per_sec=train_sps,
        eval_samples_per_sec=eval_sps,
        estimated_full_epoch_minutes=est_epoch_minutes,
        estimated_full_run_minutes=est_run_minutes,
    )
    print_smoke_report(report)
    return report