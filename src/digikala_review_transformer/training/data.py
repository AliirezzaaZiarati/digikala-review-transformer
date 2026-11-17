"""Data layer: pack loading, token arrays + labels + weights, batching, loaders.

Batches are built inside Dataset.__getitem__ (it receives an array of row indices), so the
DataLoader runs with batch_size=None and a sampler that yields index arrays.
Light imports only: this module also runs on the training machine.
"""

# ---- Section: Imports ----
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from digikala_review_transformer import config, labels
from digikala_review_transformer.training import pack, tokenization
from digikala_review_transformer.training.run_config import VARIANTS

# ---- Section: Constants ----
HEADS = pack.HEADS
IGNORE_INDEX = -100
CHUNK_BATCHES = 50  # plan: chunks of batch_size * 50 rows are sorted by length
PREFETCH_FACTOR = 2
# Separate random streams so different subsets never coincide.
SUBSET_STREAMS = {"smoke": 0, "monitor": 1}
WEIGHT_COLUMNS = ["w_rec", "w_band"]


class DataError(RuntimeError):
    """Raised when pack files, labels or weights are not what the training code expects."""


# ---- Section: Pack and labels ----
def open_pack(pack_dir):
    """Verify pack files against manifest.json (sha256, rows, columns) and return the manifest."""
    manifest, results = pack.verify_pack(pack_dir)
    failed = [message for ok, message in results if not ok]
    if failed:
        raise DataError("training pack is not usable:\n" + "\n".join("[FAIL] " + m for m in failed))
    return manifest


def to_codes(series, head):
    """Nullable label series -> int64 codes, with IGNORE_INDEX where the label is missing."""
    codes = series.astype("Int8").to_numpy(dtype="int64", na_value=IGNORE_INDEX)
    n_classes = len(labels.CLASSES[head])
    bad = (codes != IGNORE_INDEX) & ((codes < 0) | (codes >= n_classes))
    if bad.any():
        raise DataError(f"{int(bad.sum()):,} '{head}' codes are outside 0..{n_classes - 1} -> rebuild the pack")
    return codes


def class_weight_vector(y, head):
    """Class weights (index = class code) from the training labels of one head."""
    n_classes = len(labels.CLASSES[head])
    valid = np.asarray(y)[np.asarray(y) != IGNORE_INDEX]
    present = np.bincount(valid, minlength=n_classes)
    if (present == 0).any():
        raise DataError(f"'{head}' has classes with no training rows ({present.tolist()}) "
                        f"-> class weights are undefined; check the data or the smoke subset")
    raw = labels.class_weights(valid, n_classes)
    return np.array([raw[code] for code in range(n_classes)], dtype=np.float32)


def subset_row_indices(n_rows, size, seed, purpose):
    """Sorted, reproducible row indices (whole range if size >= n_rows)."""
    if purpose not in SUBSET_STREAMS:
        raise DataError(f"unknown subset purpose {purpose!r} -> use one of {tuple(SUBSET_STREAMS)}")
    if size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng([int(seed), SUBSET_STREAMS[purpose]])
    return np.sort(rng.choice(n_rows, size=int(size), replace=False)).astype(np.int64)


# ---- Section: Dataset (batch level) ----
class BatchedTokenDataset(Dataset):
    """Flat token arrays + targets + weights. dataset[row_indices] returns one padded batch."""

    def __init__(self, arrays, y, w, pad_id):
        self.ids = arrays.ids
        self.offsets = arrays.offsets
        self.lengths = arrays.lengths
        n_rows = len(self.lengths)
        self.y = {head: np.ascontiguousarray(y[head], dtype=np.int64) for head in HEADS}
        self.w = {head: np.ascontiguousarray(w[head], dtype=np.float32) for head in HEADS}
        for head in HEADS:
            if len(self.y[head]) != n_rows or len(self.w[head]) != n_rows:
                raise DataError(f"'{head}' targets/weights have a different length than the token arrays")
        self.pad_id = int(pad_id)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        rows = np.asarray(index, dtype=np.int64)
        lengths = self.lengths[rows]
        steps = np.arange(int(lengths.max()), dtype=np.int64)
        inside = steps[None, :] < lengths[:, None]
        positions = self.offsets[rows][:, None] + steps[None, :]
        np.minimum(positions, len(self.ids) - 1, out=positions)  # padded slots read any valid cell
        input_ids = np.where(inside, self.ids[positions], self.pad_id).astype(np.int64)
        batch = {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(inside.astype(np.int64)),
            "index": torch.from_numpy(rows),
        }
        for head in HEADS:
            batch[f"y_{head}"] = torch.from_numpy(self.y[head][rows])
            batch[f"w_{head}"] = torch.from_numpy(self.w[head][rows])
        return batch


# ---- Section: One split in memory ----
@dataclass
class SplitData:
    """Everything the main process needs about one split (the dataset is the light part)."""

    split: str
    ids: np.ndarray
    arrays: object
    pad_id: int
    y: dict
    y_raw: dict
    w: dict
    stats: dict
    origin: str
    source_rows: object  # row positions inside the pack file when a subset was taken, else None

    @property
    def n_rows(self):
        return len(self.ids)

    @property
    def lengths(self):
        return self.arrays.lengths

    def dataset(self):
        return BatchedTokenDataset(self.arrays, self.y, self.w, self.pad_id)


def _weights_for(frame, head, variant, n_rows):
    """B -> all ones. C -> Step 2 weights, with NaN (rows without that label) replaced by 1."""
    if variant == "B" or f"w_{head}" not in frame.columns:
        return np.ones(n_rows, dtype=np.float32)
    weights = frame[f"w_{head}"].to_numpy(dtype="float32", na_value=np.nan)
    weights = np.where(np.isnan(weights), np.float32(1.0), weights).astype(np.float32)
    if not np.isfinite(weights).all():
        raise DataError(f"w_{head} has infinite values -> rebuild the pack")
    return weights


def load_split(split, bundle, pack_dir, manifest, variant, subset_rows=None,
               subset_seed=config.SPLIT_SEED, cache_dir=None, show_progress=True):
    """Read one pack split, tokenize (or read the token cache) and return SplitData."""
    if split not in pack.SPLITS:
        raise DataError(f"unknown split {split!r} -> use one of {pack.SPLITS}")
    if variant not in VARIANTS:
        raise DataError(f"unknown variant {variant!r} -> use one of {VARIANTS}")
    if subset_rows is not None and cache_dir is not None:
        raise DataError("a row subset must not use the disk token cache -> pass cache_dir=None")

    name = pack.pack_file_name(split)
    path = Path(pack_dir) / name
    entry = manifest["files"][name]
    columns = ["id"] + pack.LABEL_COLUMNS
    if split == "train" and variant == "C":
        columns += WEIGHT_COLUMNS

    source_rows = None
    if subset_rows is None:
        frame = pd.read_parquet(path, columns=columns)
        expected_rows = entry["rows"]

        def load_texts():
            return pd.read_parquet(path, columns=["body"])["body"].tolist()
    else:
        full = pd.read_parquet(path, columns=columns + ["body"])
        source_rows = subset_row_indices(len(full), subset_rows, subset_seed, "smoke")
        frame = full.iloc[source_rows].reset_index(drop=True)
        del full
        texts = frame["body"].tolist()
        expected_rows = len(frame)

        def load_texts():
            return texts

    if len(frame) != expected_rows:
        raise DataError(f"{name}: read {len(frame):,} rows, expected {expected_rows:,} -> re-copy the pack")

    arrays, stats, origin = tokenization.get_split_tokens(
        bundle, split, entry["sha256"], load_texts, expected_rows, cache_dir, show_progress)
    y = {head: to_codes(frame[f"y_{head}"], head) for head in HEADS}
    y_raw = {head: to_codes(frame[f"y_{head}_raw"], head) for head in HEADS}
    w = {head: _weights_for(frame, head, variant, len(frame)) for head in HEADS}
    return SplitData(split=split, ids=frame["id"].to_numpy(), arrays=arrays, pad_id=bundle.pad_id,
                     y=y, y_raw=y_raw, w=w, stats=stats, origin=origin, source_rows=source_rows)


# ---- Section: Batch order ----
class LengthBucketedBatches:
    """Training batches. The order depends only on (seed, epoch, lengths), never on the variant."""

    def __init__(self, lengths, batch_size, seed, chunk_batches=CHUNK_BATCHES):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.chunk_batches = int(chunk_batches)
        self.epoch = 0
        if self.batch_size < 1 or self.chunk_batches < 1:
            raise DataError("batch_size and chunk_batches must be at least 1")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def batches_for_epoch(self, epoch):
        """Shuffle rows, sort each chunk by length, cut into batches, shuffle batch order."""
        rng = np.random.default_rng([self.seed, int(epoch)])
        order = rng.permutation(len(self.lengths))
        chunk_rows = self.batch_size * self.chunk_batches
        batches = []
        for start in range(0, len(order), chunk_rows):
            chunk = order[start:start + chunk_rows]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches.extend(chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size))
        permutation = rng.permutation(len(batches))
        return [batches[i] for i in permutation]

    def __iter__(self):
        return iter(self.batches_for_epoch(self.epoch))


def sorted_eval_batches(lengths, row_indices, batch_size):
    """Evaluation batches: the given rows sorted by length (stable), cut into large batches."""
    rows = np.asarray(row_indices, dtype=np.int64)
    order = rows[np.argsort(np.asarray(lengths)[rows], kind="stable")]
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def batch_fingerprint(batches, n_batches=5):
    """Short hash of the first batches (used to prove that B and C see the same order)."""
    digest = hashlib.sha256()
    for batch in batches[:n_batches]:
        digest.update(np.asarray(batch, dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


# ---- Section: Loader ----
def make_loader(dataset, batches, num_workers, persistent=False):
    """DataLoader over index arrays. Workers always use 'spawn' so Linux behaves like Windows."""
    options = {"batch_size": None, "sampler": batches, "num_workers": int(num_workers),
               "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        options.update(multiprocessing_context="spawn", persistent_workers=bool(persistent),
                       prefetch_factor=PREFETCH_FACTOR)
    return DataLoader(dataset, **options)