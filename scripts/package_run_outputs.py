from __future__ import annotations

import argparse
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

from digikala_review_transformer import config
from digikala_review_transformer.training import checkpoint, predict

# ---- Section: Constants ----

DEFAULT_RUNS_ROOT = config.ARTIFACTS / "transformer_runs"
DEFAULT_OUTPUT_DIR = config.ROOT / "outputs_to_send"

REQUIRED_FILES = (
    predict.RESOLVED_CONFIG_NAME,
    predict.HISTORY_NAME,
    predict.TRAIN_LOG_CSV_NAME,
    predict.TRAIN_LOG_TEXT_NAME,
    predict.RUN_SUMMARY_NAME,
    predict.VAL_PROBS_NAME,
    predict.TEST_PROBS_NAME,
    predict.ENCODER_CONFIG_NAME,
)

EXCLUDED_NAMES = {
    predict.BEST_MODEL_NAME,
    predict.LAST_CHECKPOINT_NAME,
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
}
EXCLUDED_SUFFIXES = (".pt", ".pth", ".ckpt", ".tmp")


class PackageError(RuntimeError):
    """Raised when a run directory cannot be packaged safely."""


# ---- Section: Filtering and Validation ----

def should_exclude(relative_path: Path) -> bool:
    """Return True if a file or directory inside a run folder must be excluded from the zip."""
    for part in relative_path.parts:
        if part in EXCLUDED_NAMES or part.startswith("."):
            return True
    name_lower = relative_path.name.lower()
    return bool(any(name_lower.endswith(suffix) for suffix in EXCLUDED_SUFFIXES))


def validate_completed_run_dir(run_dir: Path) -> list[Path]:
    """
    Verify that `run_dir` is a completed run containing all required light artifacts
    and return the sorted list of files to include in the zip archive.
    """
    if not run_dir.is_dir():
        raise PackageError(
            f"[FAIL] Run directory not found: {run_dir} -> check the run name passed to --runs"
        )

    status = checkpoint.run_status(run_dir)
    if status != "complete":
        raise PackageError(
            f"[FAIL] Run '{run_dir.name}' has status={status!r} (expected 'complete') -> "
            f"finish training before packaging outputs"
        )

    last_ckpt_path = run_dir / predict.LAST_CHECKPOINT_NAME
    if last_ckpt_path.exists():
        raise PackageError(
            f"[FAIL] Run '{run_dir.name}' still contains '{predict.LAST_CHECKPOINT_NAME}' -> "
            f"run finalization did not finish cleanly"
        )
    
    missing_files = [
        name for name in REQUIRED_FILES if not (run_dir / name).is_file()
    ]
    if missing_files:
        raise PackageError(
            f"[FAIL] Run '{run_dir.name}' is missing required output files: {missing_files} -> "
            f"inspect {run_dir}"
        )

    tokenizer_dir = run_dir / predict.TOKENIZER_DIR_NAME
    tokenizer_files = (
        [p for p in tokenizer_dir.rglob("*") if p.is_file() and not should_exclude(p.relative_to(run_dir))]
        if tokenizer_dir.is_dir()
        else []
    )
    if not tokenizer_files:
        raise PackageError(
            f"[FAIL] Run '{run_dir.name}' is missing a non-empty '{predict.TOKENIZER_DIR_NAME}/' directory -> "
            f"verify finalize_run completed properly"
        )

    included_files: list[Path] = []
    for file_path in sorted(run_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(run_dir)
        if should_exclude(rel):
            continue
        included_files.append(file_path)

    return included_files


# ---- Section: Packaging One Run ----

def package_single_run(
    run_name: str,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Package one completed run into `<output_dir>/<run_name>.zip`, excluding `.pt` checkpoints.
    Archive entries are stored under `<run_name>/<relative_path>`.
    """
    runs_root_path = Path(runs_root).resolve()
    output_dir_path = Path(output_dir).resolve()
    run_dir = runs_root_path / run_name

    included_files = validate_completed_run_dir(run_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    final_zip_path = output_dir_path / f"{run_name}.zip"
    temp_zip_path = output_dir_path / f".{run_name}.zip.tmp"
    if temp_zip_path.exists():
        temp_zip_path.unlink()

    try:
        with zipfile.ZipFile(
            temp_zip_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for file_path in included_files:
                rel = file_path.relative_to(run_dir)
                arcname = (Path(run_name) / rel).as_posix()
                archive.write(file_path, arcname=arcname)
        temp_zip_path.replace(final_zip_path)
    finally:
        if temp_zip_path.exists():
            try:
                temp_zip_path.unlink()
            except OSError:
                pass

    size_mb = final_zip_path.stat().st_size / (1024**2)
    print(
        f"[PASS] Packaged '{run_name}': {len(included_files)} files -> "
        f"{final_zip_path} ({size_mb:.2f} MB; best_model.pt excluded)",
        flush=True,
    )
    return final_zip_path


# ---- Section: CLI Entry Point ----

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for packaging completed run outputs."""
    parser = argparse.ArgumentParser(
        description="Package lightweight run outputs into outputs_to_send/ (excluding .pt weights).",
    )
    parser.add_argument(
        "--runs",
        dest="runs",
        nargs="+",
        action="extend",
        required=True,
        metavar="RUN_NAME",
        help="One or more completed run names under artifacts/transformer_runs/.",
    )
    parser.add_argument(
        "--runs-root",
        dest="runs_root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Root directory containing transformer run folders.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the generated .zip archives.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and package each requested run."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    for run_name in args.runs:
        try:
            package_single_run(
                run_name,
                runs_root=args.runs_root,
                output_dir=args.output_dir,
            )
        except Exception as error:  # noqa: BLE001
            text = str(error).strip()
            if text.startswith("[FAIL]"):
                print(text, flush=True)
            else:
                print(
                    f"[FAIL] Could not package '{run_name}': {type(error).__name__}: {text} -> "
                    f"verify the run folder under {args.runs_root}",
                    flush=True,
                )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())