"""Select the Stage-2 variant (B or C) from two completed Stage-1 runs.

Rule (fixed before seeing results): C is selected only if its best validation
mean Macro F1 (corrected labels) exceeds that of B by at least MIN_GAIN;
otherwise B is selected.

stdout : a single letter (B or C), printed only when the decision is made.
stderr : human-readable details.
Exit codes: 0 = decided, 3 = runs not finished yet (retry later), 1 = error.
"""
import argparse
import json
import sys
from pathlib import Path

MIN_GAIN = 0.003
FLOAT_SLACK = 1e-12          # guards the exact-threshold case against float noise
CONSISTENCY_TOLERANCE = 1e-6
RUN_NAME_TEMPLATE = "tooka_{variant}"
EXIT_DECIDED, EXIT_ERROR, EXIT_NOT_READY = 0, 1, 3
# Observed value in real run_summary.json is "complete"; "completed" is accepted too
# (an unfinished run is already detected by the leftover last.pt).
COMPLETE_STATUSES = {"complete", "completed"}


class RunNotReady(Exception):
    pass


class RunInvalid(Exception):
    pass


def load_run_score(runs_root: Path, variant: str) -> tuple[float, int]:
    """Return (best validation mean Macro F1 on corrected labels, best epoch)."""
    run_name = RUN_NAME_TEMPLATE.format(variant=variant)
    run_dir = runs_root / run_name
    summary_path = run_dir / "run_summary.json"
    # A leftover last.pt means the run is unfinished (it is deleted on completion).
    if not summary_path.is_file() or (run_dir / "last.pt").exists():
        raise RunNotReady(f"{run_name}: not finished yet")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in COMPLETE_STATUSES:
        raise RunNotReady(f"{run_name}: status is {summary.get('status')!r}")
    if summary.get("variant") != variant:
        raise RunInvalid(f"{run_name}: variant in summary is {summary.get('variant')!r}")
    score = float(summary["validation_best"]["corrected"]["mean_macro_f1"])
    selection = summary["selection_and_stopping"]
    if abs(score - float(selection["best_val_mean_macro_f1"])) > CONSISTENCY_TOLERANCE:
        print(f"[WARN] {run_name}: validation_best and selection_and_stopping differ "
              f"({score} vs {selection['best_val_mean_macro_f1']})", file=sys.stderr)
    return score, int(selection["best_epoch"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="artifacts/transformer_runs")
    parser.add_argument("--min-gain", type=float, default=MIN_GAIN)
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    try:
        score_b, epoch_b = load_run_score(runs_root, "B")
        score_c, epoch_c = load_run_score(runs_root, "C")
    except RunNotReady as error:
        print(f"[WAIT] {error}", file=sys.stderr)
        return EXIT_NOT_READY
    except (RunInvalid, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"[FAIL] {error!r} -> inspect run_summary.json", file=sys.stderr)
        return EXIT_ERROR
    gain = score_c - score_b
    winner = "C" if gain >= args.min_gain - FLOAT_SLACK else "B"
    print(f"[INFO] B={score_b:.6f} (epoch {epoch_b}), C={score_c:.6f} (epoch {epoch_c}), "
          f"C-B={gain:+.6f}, threshold={args.min_gain}", file=sys.stderr)
    print(winner)
    return EXIT_DECIDED


if __name__ == "__main__":
    sys.exit(main())