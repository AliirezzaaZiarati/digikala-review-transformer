"""Tokenization for the three encoders: two backends and a flat-array token cache.

Backends (model.tokenizer_backend in the run config):
  hf_fast        Hugging Face fast tokenizer (TookaBERT, roberta-fa-zwnj).
  sentencepiece  raw spm.model, structure [CLS] + ids[:max_length-2] + [SEP] (mDeBERTa).

Only input_ids are cached; the attention mask is all ones before padding and is built in collate.
Light imports only: this module also runs on the training machine.
"""

# ---- Section: Imports ----
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm
from transformers import AutoTokenizer

from digikala_review_transformer import config, utils
from digikala_review_transformer.training.pack import sha256_file
from digikala_review_transformer.training.run_config import TOKENIZER_BACKENDS

# ---- Section: Constants ----
CACHE_FORMAT_VERSION = 1
CACHE_ROOT = config.DATA_INTERIM / "token_cache"
META_NAME = "meta.json"
BACKEND_INFO_NAME = "tokenizer_backend.json"
SP_MODEL_NAME = "spm.model"
# Piece names of the special tokens inside spm.model (checked against its own ids).
SP_PIECES = {"first": "[CLS]", "last": "[SEP]", "pad": "[PAD]", "unk": "[UNK]"}
HF_FINGERPRINT_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "vocab.txt", SP_MODEL_NAME,
)
PROBE_TEXT = "x"
ENCODE_CHUNK_ROWS = 100_000
STATS_CHUNK_ROWS = 200_000


class TokenizationError(RuntimeError):
    """Raised when a tokenizer or a token array is not what the training code expects."""


# ---- Section: Small helpers ----
def fingerprint_files(directory, names):
    """Hash of the tokenizer files that exist in a folder (at least one is required)."""
    directory = Path(directory)
    parts = [f"{name}:{sha256_file(directory / name)}" for name in names if (directory / name).is_file()]
    if not parts:
        raise TokenizationError(
            f"no tokenizer files found in {directory} -> download the model with "
            f"scripts/download_pretrained_models.py"
        )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# ---- Section: Tokenizer bundles ----
class TokenizerBundle:
    """Common interface: encode(texts) -> list of id lists (specials added, truncated)."""

    backend = ""
    structure = ""

    def __init__(self, source_dir, max_length):
        self.source_dir = Path(source_dir)
        self.max_length = int(max_length)
        self.name = self.source_dir.name
        self.fingerprint = ""
        self.first_id = self.last_id = self.pad_id = self.unk_id = None
        if self.max_length < 3:
            raise TokenizationError(f"max_length={self.max_length} is too small -> need room for 2 special tokens")

    def encode(self, texts):
        raise NotImplementedError

    def _save_files(self, out_dir):
        raise NotImplementedError

    def info(self):
        """Plain dict describing this tokenizer (written next to the saved files)."""
        return {
            "backend": self.backend,
            "source_model": self.name,
            "max_length": self.max_length,
            "special_ids": {"first": self.first_id, "last": self.last_id,
                            "pad": self.pad_id, "unk": self.unk_id},
            "structure": self.structure,
            "fingerprint": self.fingerprint,
        }

    def save(self, out_dir):
        """Save tokenizer files plus tokenizer_backend.json into out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_files(out_dir)
        (out_dir / BACKEND_INFO_NAME).write_text(json.dumps(self.info(), indent=2), encoding="utf-8")


class HfFastBundle(TokenizerBundle):
    backend = "hf_fast"
    structure = "tokenizer adds its own special tokens; truncation keeps the end token"

    def __init__(self, source_dir, max_length):
        super().__init__(source_dir, max_length)
        self.fingerprint = fingerprint_files(self.source_dir, HF_FINGERPRINT_FILES)
        self.tokenizer = AutoTokenizer.from_pretrained(self.source_dir)
        if not self.tokenizer.is_fast:
            raise TokenizationError(f"{self.name}: tokenizer is not a fast tokenizer -> install 'tokenizers'")
        if self.tokenizer.pad_token_id is None:
            raise TokenizationError(f"{self.name}: tokenizer has no pad token id -> cannot pad batches")

        # First/last ids come from a probe text (TookaBERT has no cls_token_id).
        options = {"return_token_type_ids": False, "return_attention_mask": False}
        full = self.tokenizer(PROBE_TEXT, **options)["input_ids"]
        bare = self.tokenizer(PROBE_TEXT, add_special_tokens=False, **options)["input_ids"]
        if len(full) < 3 or full[1:-1] != bare:
            raise TokenizationError(
                f"{self.name}: probe encoding {full} is not [start] + tokens + [end] -> inspect the tokenizer"
            )
        self.first_id, self.last_id = full[0], full[-1]
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.unk_token_id

    def encode(self, texts):
        return self.tokenizer(
            list(texts), truncation=True, max_length=self.max_length, add_special_tokens=True,
            padding=False, return_attention_mask=False, return_token_type_ids=False,
        )["input_ids"]

    def _save_files(self, out_dir):
        self.tokenizer.save_pretrained(out_dir)


class SentencePieceBundle(TokenizerBundle):
    backend = "sentencepiece"
    structure = "[CLS] + ids[:max_length-2] + [SEP]; attention mask is all ones"

    def __init__(self, source_dir, max_length):
        super().__init__(source_dir, max_length)
        model_file = self.source_dir / SP_MODEL_NAME
        if not model_file.is_file():
            raise TokenizationError(f"{model_file} not found -> this backend needs {SP_MODEL_NAME} in the model folder")
        self.fingerprint = fingerprint_files(self.source_dir, (SP_MODEL_NAME,))
        self.processor = spm.SentencePieceProcessor(model_file=str(model_file))

        # Special ids are read by piece name and cross-checked with sentencepiece's own ids.
        ids = {key: self.processor.piece_to_id(piece) for key, piece in SP_PIECES.items()}
        own = {"first": self.processor.bos_id(), "last": self.processor.eos_id(),
               "pad": self.processor.pad_id(), "unk": self.processor.unk_id()}
        for key, piece in SP_PIECES.items():
            if self.processor.id_to_piece(ids[key]) != piece or ids[key] != own[key]:
                raise TokenizationError(
                    f"{self.name}: special piece {piece} (id {ids[key]}) does not match sentencepiece "
                    f"{key} id {own[key]} -> this backend assumes the mDeBERTa layout"
                )
        self.first_id, self.last_id = ids["first"], ids["last"]
        self.pad_id, self.unk_id = ids["pad"], ids["unk"]

    def encode(self, texts):
        body_length = self.max_length - 2
        encoded = self.processor.encode(list(texts))
        return [[self.first_id, *row[:body_length], self.last_id] for row in encoded]

    def _save_files(self, out_dir):
        shutil.copyfile(self.source_dir / SP_MODEL_NAME, out_dir / SP_MODEL_NAME)


def load_tokenizer(model_path, backend, max_length):
    """Build the tokenizer bundle for a pretrained model folder."""
    if backend == "hf_fast":
        return HfFastBundle(model_path, max_length)
    if backend == "sentencepiece":
        return SentencePieceBundle(model_path, max_length)
    raise TokenizationError(f"unknown tokenizer backend {backend!r} -> use one of {TOKENIZER_BACKENDS}")


def load_saved_tokenizer(tokenizer_dir):
    """Load the tokenizer saved in a run's tokenizer/ folder (for later phases)."""
    tokenizer_dir = Path(tokenizer_dir)
    info_path = tokenizer_dir / BACKEND_INFO_NAME
    if not info_path.is_file():
        raise TokenizationError(f"{info_path} not found -> this folder was not written by TokenizerBundle.save")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    bundle = load_tokenizer(tokenizer_dir, info["backend"], info["max_length"])
    if bundle.info()["special_ids"] != info["special_ids"]:
        raise TokenizationError(
            f"special ids changed after reload ({bundle.info()['special_ids']} vs {info['special_ids']}) "
            f"-> the saved tokenizer files are damaged"
        )
    return bundle


# ---- Section: Flat token arrays ----
class TokenArrays:
    """All rows of one split: flat int32 ids plus int64 offsets (row i = ids[offsets[i]:offsets[i+1]])."""

    def __init__(self, ids, offsets):
        if ids.ndim != 1 or offsets.ndim != 1 or len(offsets) < 1:
            raise TokenizationError("token arrays must be 1-D and offsets must not be empty")
        self.ids = ids
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets) - 1

    @property
    def lengths(self):
        return np.diff(self.offsets)

    def row(self, index):
        return self.ids[self.offsets[index]:self.offsets[index + 1]]

    def equals(self, other):
        return np.array_equal(self.ids, other.ids) and np.array_equal(self.offsets, other.offsets)


def encode_texts(bundle, texts, show_progress=False):
    """Tokenize a list of strings in chunks and return TokenArrays."""
    texts = list(texts)
    if not all(isinstance(text, str) for text in texts):
        raise TokenizationError("texts contain non-string values -> check the body column for NA")
    lengths_parts, id_parts = [], []
    starts = range(0, len(texts), ENCODE_CHUNK_ROWS)
    for start in tqdm(starts, disable=not show_progress, desc=f"tokenize {bundle.name}", unit="chunk"):
        rows = bundle.encode(texts[start:start + ENCODE_CHUNK_ROWS])
        lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
        flat = np.fromiter(itertools.chain.from_iterable(rows), dtype=np.int32, count=int(lengths.sum()))
        lengths_parts.append(lengths)
        id_parts.append(flat)
    offsets = np.zeros(len(texts) + 1, dtype=np.int64)
    if texts:
        np.cumsum(np.concatenate(lengths_parts), out=offsets[1:])
        ids = np.concatenate(id_parts)
    else:
        ids = np.zeros(0, dtype=np.int32)
    return TokenArrays(ids, offsets)


def validate_arrays(arrays, bundle):
    """Every row must start with first_id, end with last_id and fit in max_length."""
    problems = []
    lengths = arrays.lengths
    if arrays.offsets[0] != 0 or arrays.offsets[-1] != len(arrays.ids):
        problems.append("offsets do not cover the id array")
    elif (lengths < 2).any():
        problems.append(f"{int((lengths < 2).sum()):,} rows have fewer than 2 tokens")
    else:
        if (lengths > bundle.max_length).any():
            problems.append(f"{int((lengths > bundle.max_length).sum()):,} rows exceed max_length={bundle.max_length}")
        if (arrays.ids[arrays.offsets[:-1]] != bundle.first_id).any():
            problems.append("some rows do not start with the start token")
        if (arrays.ids[arrays.offsets[1:] - 1] != bundle.last_id).any():
            problems.append("some rows do not end with the end token (truncation cut it?)")
    if problems:
        raise TokenizationError("token arrays are invalid: " + "; ".join(problems))


def token_stats(arrays, bundle):
    """Counts for the run summary: length, truncation upper bound, unk and max id."""
    n_rows, lengths = len(arrays), arrays.lengths
    unk_tokens = rows_with_unk = 0
    for start in range(0, n_rows, STATS_CHUNK_ROWS):
        stop = min(start + STATS_CHUNK_ROWS, n_rows)
        chunk = arrays.ids[arrays.offsets[start]:arrays.offsets[stop]]
        is_unk = chunk == bundle.unk_id
        if is_unk.any():
            unk_tokens += int(is_unk.sum())
            row_of_token = np.repeat(np.arange(stop - start), lengths[start:stop])
            rows_with_unk += len(np.unique(row_of_token[is_unk]))
    n_tokens = len(arrays.ids)
    at_max = int((lengths == bundle.max_length).sum())
    return {
        "rows": n_rows,
        "tokens": n_tokens,
        "mean_length": round(n_tokens / max(n_rows, 1), 3),
        "max_length_seen": int(lengths.max()) if n_rows else 0,
        "rows_at_max_length": at_max,
        "rows_at_max_length_pct": round(at_max / max(n_rows, 1) * 100, 4),
        "unk_tokens": unk_tokens,
        "unk_pct_of_tokens": round(unk_tokens / max(n_tokens, 1) * 100, 5),
        "rows_with_unk": rows_with_unk,
        "rows_with_unk_pct": round(rows_with_unk / max(n_rows, 1) * 100, 4),
        "max_token_id": int(arrays.ids.max()) if n_tokens else -1,
    }


# ---- Section: Disk cache ----
def default_cache_dir(bundle):
    """data/interim/token_cache/<model>_<max_length>/"""
    return CACHE_ROOT / f"{bundle.name}_{bundle.max_length}"


def cache_key(bundle, source_sha256):
    payload = {"format": CACHE_FORMAT_VERSION, "backend": bundle.backend, "tokenizer": bundle.fingerprint,
               "max_length": bundle.max_length, "source_sha256": source_sha256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _read_meta(cache_dir):
    fresh = {"format": CACHE_FORMAT_VERSION, "splits": {}}
    path = Path(cache_dir) / META_NAME
    if not path.is_file():
        return fresh
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fresh
    return meta if meta.get("format") == CACHE_FORMAT_VERSION else fresh


def _write_meta(cache_dir, meta):
    path = Path(cache_dir) / META_NAME
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _save_npy_atomic(array, path):
    temp = path.with_name(path.stem + ".tmp.npy")  # np.save needs the .npy suffix
    np.save(temp, array, allow_pickle=False)
    os.replace(temp, path)


def _read_cache(bundle, split, key, expected_rows, cache_dir):
    """Return (arrays, stats, reason); arrays is None when the cache cannot be used."""
    cache_dir = Path(cache_dir)
    entry = _read_meta(cache_dir)["splits"].get(split)
    if entry is None:
        return None, None, "no cache entry"
    if entry.get("key") != key:
        return None, None, "cache key differs (new pack, tokenizer or max_length)"
    try:
        ids = np.load(cache_dir / f"{split}_ids.npy", allow_pickle=False)
        offsets = np.load(cache_dir / f"{split}_offsets.npy", allow_pickle=False)
        if ids.dtype != np.int32 or offsets.dtype != np.int64:
            raise TokenizationError(f"unexpected dtypes {ids.dtype}/{offsets.dtype}")
        arrays = TokenArrays(ids, offsets)
        if len(arrays) != expected_rows or len(arrays) != entry["stats"]["rows"]:
            raise TokenizationError(f"cache has {len(arrays):,} rows, expected {expected_rows:,}")
        if len(ids) != entry["stats"]["tokens"]:
            raise TokenizationError("token count differs from meta.json")
        validate_arrays(arrays, bundle)
    except (OSError, EOFError, ValueError, KeyError, TokenizationError) as error:
        return None, None, f"cache unreadable: {error}"
    return arrays, entry["stats"], ""


def _write_cache(split, key, arrays, stats, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(cache_dir)
    meta["splits"].pop(split, None)  # no valid entry while files are being replaced
    _write_meta(cache_dir, meta)
    _save_npy_atomic(arrays.ids, cache_dir / f"{split}_ids.npy")
    _save_npy_atomic(arrays.offsets, cache_dir / f"{split}_offsets.npy")
    meta["splits"][split] = {"key": key, "stats": stats}
    _write_meta(cache_dir, meta)  # meta last: it marks the files as complete


def get_split_tokens(bundle, split, source_sha256, load_texts, expected_rows, cache_dir=None, show_progress=True):
    """Return (TokenArrays, stats, origin); origin is 'cache' or 'built'.

    load_texts is only called when the cache cannot be used. cache_dir=None means no disk cache.
    """
    key = cache_key(bundle, source_sha256)
    if cache_dir is not None:
        arrays, stats, reason = _read_cache(bundle, split, key, expected_rows, cache_dir)
        if arrays is not None:
            return arrays, stats, "cache"
        if reason != "no cache entry":
            print(f"[WARN] {split}: {reason} -> rebuilding")

    with utils.timer(f"tokenize_{bundle.name}_{split}"):
        arrays = encode_texts(bundle, load_texts(), show_progress=show_progress)
    if len(arrays) != expected_rows:
        raise TokenizationError(f"{split}: got {len(arrays):,} rows, expected {expected_rows:,} -> texts and manifest differ")
    validate_arrays(arrays, bundle)
    stats = token_stats(arrays, bundle)
    if cache_dir is not None:
        _write_cache(split, key, arrays, stats, cache_dir)
    return arrays, stats, "built"