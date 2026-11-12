## Update: masking, duplicate cap and artifacts (notebook `06_masking_and_artifacts`)
**Correction to earlier sections:** the corrected band column is `step1_rate_band_label` (not `step1_band_label`).
**Naming:** notebooks keep repo numbering; the README must contain a table mapping them to the required names.

### Decisions
- All work from now on uses the corrected labels (`step1_rec_label`, `step1_rate_band_label`); raw labels are reported next to them.
- Order: correct labels, then mask, then cap. Deviation from the shared protocol, recorded in `split_manifest.json`. Instructor confirmation field: "not requested; decided by the project author".
- Mask: labeled train rows only (NULL `step1_rec_label` never enters the pseudo pool), per class, sorted by (stable_hash(id), id), floor(0.20 * count). SQL hash = first 16 hex chars of sha256 compared as a string (verified 1000/1000 against Python).
- Mask plan: no_idea: 405,857 -> mask 81,171; not_recommended: 349,558 -> mask 69,911; recommended: 2,926,171 -> mask 585,234. Total masked 736,316.
- Overlap with a mask built on the raw label: 736,300 of 736,316.
- Cap: 50 copies per body, smallest (hash, id) kept, applied to each pool separately after masking; validation/test not capped. supervised: 3,570,032 -> 2,860,236 (dropped 709,796, 19.8821%, 2,110 bodies over cap); pseudo: 736,316 -> 635,824 (dropped 100,492, 13.6479%, 442 bodies over cap).

### Database state
- New column `train_role` in `clean_base` (train rows only): `supervised`, `pseudo_unlabeled`, `cap_dropped_supervised`, `cap_dropped_pseudo`; val/test are NULL. Role counts: {'supervised': 2860236, 'cap_dropped_supervised': 709796, 'pseudo_unlabeled': 635824, 'cap_dropped_pseudo': 100492}.
- Natural unlabeled = `train_role = 'supervised'` and `step1_rec_label IS NULL`.
- Original columns verified identical to `data/interim/clean_base_backup.duckdb` (pre-mask state). DB was compacted earlier (about 308 MB); do not run cell 6 of notebook 06 again without re-running cell 7.

### Files in `data/processed/` (+ `split_manifest.json` with sha256)
  - `train_supervised.csv`: 2,860,236 rows, columns ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'], 447.7 MB
  - `train_pseudo_unlabeled.csv`: 635,824 rows, columns ['id', 'body'], 68.6 MB
  - `natural_unlabeled.csv`: 468,079 rows, columns ['id', 'body'], 54.2 MB
  - `validation.csv`: 922,794 rows, columns ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'], 129.5 MB
  - `test.csv`: 922,805 rows, columns ['id', 'body', 'product_id', 'recommendation_status', 'step1_rec_label', 'rating_band', 'step1_rate_band_label', 'rate'], 129.3 MB
- `cap_dropped` rows (810,288) are not exported. 4 files + dropped = 6,151,947.

### Facts for the report
- Recommended share among labeled rows (supervised): 79.48% before cap, 76.69% after cap, validation 79.26%. Compute class weights on `train_supervised` after the cap.
- Pseudo pool rows changed by step 1: 0.2917% before cap, 0.1614% after cap (almost all no_idea -> recommended).
- Pseudo rows whose body also occurs in supervised: 35.51% before cap, 25.32% after cap. Report pseudo-label accuracy and acceptance rate on the whole pool and on non-duplicated bodies only.
- Band share (high) by group: pseudo 64.08%, supervised with recommendation 64.15%, supervised without recommendation (natural) 76.85%. The natural pool differs from labeled data: report pseudo-label stats for the artificial and natural pools separately.
- Cap shifts class shares away from validation/test; this is documented, Macro F1 is the main metric.

### Remaining steps
Baselines, sample weighting (variant C), Transformer fine-tuning, semi-supervised experiment, error analysis (>= 20 errors), offline prediction script, README (with notebook-name mapping and a note that `product_id` is used only for grouping and `rate` only for analysis), final report, `release_report.json`.
