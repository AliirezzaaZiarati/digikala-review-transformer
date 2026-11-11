from pathlib import Path

# Project root: src/digikala_review_transformer/config.py -> three levels up
ROOT = Path(__file__).resolve().parents[2]

# Folders (data and artifacts contents are not tracked by Git)
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

# Cleaned data: DuckDB file lives directly in the project root
DB_PATH = ROOT / "project.duckdb"
CLEAN_TABLE = "clean_base"

# Fixed by the project definition: same split and same mask for everyone
SPLIT_SEED = 2026
MASK_SEED = "digikala-semi-supervised-v1"
MASK_RATE = 0.20

# Semi-supervised: accept a pseudo-label only at this confidence or higher
PSEUDO_LABEL_THRESHOLD = 0.90