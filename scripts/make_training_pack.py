"""Build data/training_pack/ from the split CSVs and Step 2 weights.

Run once on the analysis machine (Linux):
    uv run python scripts/make_training_pack.py
"""

import argparse
import sys
from pathlib import Path

from digikala_review_transformer import config, utils
from digikala_review_transformer.training import pack, run_config

DEFAULT_CONFIG = config.ROOT / "configs" / "transformer" / "tooka_B.yaml"


def main():
    parser = argparse.ArgumentParser(description="Build the training pack.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="any run config; only data.pack_dir is used")
    parser.add_argument("--force", action="store_true", help="rebuild even if manifest.json exists")
    args = parser.parse_args()

    try:
        cfg = run_config.load_run_config(Path(args.config))
        pack_dir = Path(run_config.project_path(cfg["data"]["pack_dir"]))
        print(f"Pack directory: {pack_dir}")
        with utils.timer("make_training_pack"):
            manifest = pack.build_training_pack(pack_dir, force=args.force)
    except (pack.PackError, run_config.ConfigError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    pack.print_summary(manifest)
    print("\n[PASS] training pack built")
    return 0


if __name__ == "__main__":
    sys.exit(main())