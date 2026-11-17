"""Load, merge, validate and hash a training-run config (YAML with `extends`).

Light imports only (yaml, stdlib, project config), so it runs on the training machine.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

from digikala_review_transformer import config

# ── Expected structure: every key must exist with this type (no typos, no missing keys) ──
CONFIG_SCHEMA = {
    "run_name": str,
    "data": {"pack_dir": str},
    "model": {
        "pretrained_path": str,
        "max_length": int,
        "head_dropout": float,
        "tokenizer_backend": str,
    },    "training": {
        "variant": str,
        "max_epochs": int,
        "early_stopping_patience": int,
        "early_stopping_min_delta": float,
        "batch_size": int,
        "eval_batch_size": int,
        "lr_encoder": float,
        "lr_heads": float,
        "weight_decay": float,
        "warmup_ratio": float,
        "max_grad_norm": float,
        "label_smoothing": float,
        "loss_weight_band": float,
        "use_class_weights": bool,
        "bf16": bool,
        "num_workers": int,
        "log_every_steps": int,
        "train_subset_eval_rows": int,
    },
    "output": {"root": str},
}

RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9_]+")
VARIANTS = ("B", "C")
TOKENIZER_BACKENDS = ("hf_fast", "sentencepiece")

# These keys change speed or memory only, never the training result. They stay out of the
# hash so that e.g. num_workers can be lowered after a crash without blocking the resume.
RUNTIME_ONLY_KEYS = (
    ("training", "num_workers"),
    ("training", "log_every_steps"),
    ("training", "eval_batch_size"),
)


class ConfigError(ValueError):
    """Raised with one or more '[FAIL] reason -> fix' lines."""


# ── Reading and merging ──
def _read_yaml(path):
    """Read one YAML file (utf-8, BOM tolerated) and return its top-level mapping."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"[FAIL] config file not found: {path} -> check the path given to --config")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as error:
        raise ConfigError(f"[FAIL] {path.name} is not valid YAML: {error} -> fix the syntax") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"[FAIL] {path.name} is not a mapping -> the file must contain 'key: value' pairs")
    return data


def _deep_merge(parent, child):
    """Return parent updated by child; nested mappings are merged, other values replaced."""
    merged = copy.deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_extends(path, chain=()):
    """Read a YAML file and, if it has `extends`, merge it on top of its parent file."""
    path = Path(path)
    resolved = path.resolve()
    if resolved in chain:
        raise ConfigError(f"[FAIL] circular 'extends' at {path.name} -> remove the loop between config files")
    data = _read_yaml(path)
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    if not isinstance(parent_ref, str):
        raise ConfigError(f"[FAIL] 'extends' in {path.name} is not a file name -> write e.g. extends: base.yaml")
    parent = _load_with_extends(path.parent / parent_ref, chain + (resolved,))
    return _deep_merge(parent, data)


# ── Validation ──
def _type_ok(value, expected):
    """bool is not accepted as int or float; int is accepted as float."""
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def _check_structure(node, schema, prefix, errors):
    """Report unknown keys, missing keys, null values and wrong types."""
    for key in node:
        if key not in schema:
            if prefix == "" and key == "seed":
                errors.append("[FAIL] 'seed' is set in YAML -> remove it; the seed comes from "
                              "digikala_review_transformer.config.SPLIT_SEED")
            else:
                errors.append(f"[FAIL] unknown key '{prefix}{key}' -> fix the spelling or remove it")
    for key, expected in schema.items():
        name = f"{prefix}{key}"
        if key not in node:
            errors.append(f"[FAIL] missing key '{name}' -> add it to the config or to base.yaml")
        elif isinstance(expected, dict):
            if isinstance(node[key], dict):
                _check_structure(node[key], expected, f"{name}.", errors)
            else:
                errors.append(f"[FAIL] '{name}' must be a section -> write '{key}:' followed by indented keys")
        elif node[key] is None:
            errors.append(f"[FAIL] '{name}' is not set -> give it a value in the run config")
        elif not _type_ok(node[key], expected):
            hint = " (write 1.0e-5, not 1e-5: YAML reads 1e-5 as text)" if expected is float else ""
            errors.append(f"[FAIL] '{name}' must be {expected.__name__}, got {node[key]!r}{hint} -> fix the value")


def _normalize_floats(node, schema):
    """Turn ints written for float keys into floats, so the hash does not depend on '2' vs '2.0'."""
    for key, expected in schema.items():
        if isinstance(expected, dict):
            _normalize_floats(node[key], expected)
        elif expected is float:
            node[key] = float(node[key])


def _expect(errors, condition, reason, fix):
    if not condition:
        errors.append(f"[FAIL] {reason} -> {fix}")


def _check_values(cfg, errors):
    """Range and consistency checks (run only after the structure is valid)."""
    model, training = cfg["model"], cfg["training"]
    _expect(errors, RUN_NAME_PATTERN.fullmatch(cfg["run_name"]) is not None,
            f"run_name {cfg['run_name']!r} has invalid characters",
            "use letters, digits and underscore only (it becomes a folder name)")
    _expect(errors, training["variant"] in VARIANTS,
            f"training.variant is {training['variant']!r}", f"use one of {VARIANTS}")
    _expect(errors, model["max_length"] > 0, "model.max_length must be positive", "e.g. 128")
    _expect(errors, model["tokenizer_backend"] in TOKENIZER_BACKENDS,
            f"model.tokenizer_backend is {model['tokenizer_backend']!r}",
            f"use one of {TOKENIZER_BACKENDS}")
    _expect(errors, 0 <= model["head_dropout"] < 1, "model.head_dropout must be in [0, 1)", "e.g. 0.1")
    _expect(errors, training["max_epochs"] >= 1, "training.max_epochs must be at least 1", "e.g. 4")
    _expect(errors, training["early_stopping_patience"] >= 0,
            "training.early_stopping_patience must be >= 0", "e.g. 1")
    _expect(errors, training["early_stopping_min_delta"] >= 0,
            "training.early_stopping_min_delta must be >= 0", "e.g. 0.0")
    _expect(errors, training["batch_size"] >= 1, "training.batch_size must be at least 1", "e.g. 128")
    _expect(errors, training["eval_batch_size"] >= 1, "training.eval_batch_size must be at least 1", "e.g. 512")
    _expect(errors, training["lr_encoder"] > 0, "training.lr_encoder must be positive", "e.g. 2.0e-5")
    _expect(errors, training["lr_heads"] > 0, "training.lr_heads must be positive", "e.g. 1.0e-4")
    _expect(errors, training["lr_encoder"] < training["lr_heads"],
            "training.lr_encoder must be smaller than training.lr_heads (project requirement)",
            "lower lr_encoder or raise lr_heads")
    _expect(errors, training["weight_decay"] >= 0, "training.weight_decay must be >= 0", "e.g. 0.01")
    _expect(errors, 0 <= training["warmup_ratio"] < 1, "training.warmup_ratio must be in [0, 1)", "e.g. 0.06")
    _expect(errors, training["max_grad_norm"] > 0, "training.max_grad_norm must be positive", "e.g. 1.0")
    _expect(errors, 0 <= training["label_smoothing"] < 1,
            "training.label_smoothing must be in [0, 1)", "e.g. 0.1")
    _expect(errors, training["loss_weight_band"] > 0, "training.loss_weight_band must be positive", "e.g. 1.0")
    _expect(errors, training["num_workers"] >= 0, "training.num_workers must be >= 0", "use 0 to disable workers")
    _expect(errors, training["log_every_steps"] >= 1, "training.log_every_steps must be >= 1", "e.g. 200")
    _expect(errors, training["train_subset_eval_rows"] >= 1,
            "training.train_subset_eval_rows must be >= 1", "e.g. 100000")


# ── Public API ──
def load_run_config(config_path):
    """Load one run config (with `extends`), validate it, add the project seed, return a dict."""
    merged = _load_with_extends(config_path)
    errors = []
    _check_structure(merged, CONFIG_SCHEMA, "", errors)
    if not errors:
        _normalize_floats(merged, CONFIG_SCHEMA)
        _check_values(merged, errors)
    if errors:
        raise ConfigError(f"Config problems in {Path(config_path).name}:\n" + "\n".join(errors))
    merged["seed"] = config.SPLIT_SEED
    return merged


def config_hash(cfg):
    """SHA-256 of the resolved config, ignoring runtime-only keys (see RUNTIME_ONLY_KEYS)."""
    hashed = copy.deepcopy(cfg)
    for section, key in RUNTIME_ONLY_KEYS:
        hashed[section].pop(key, None)
    text = json.dumps(hashed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def project_path(path_text):
    """Relative paths in a config are relative to the project root."""
    path = Path(path_text)
    return path if path.is_absolute() else Path(config.ROOT) / path


def run_dir(cfg):
    """Output folder of one run: <output.root>/<run_name>."""
    return project_path(cfg["output"]["root"]) / cfg["run_name"]


def dump_resolved_config(cfg, path):
    """Write the final config (after extends and seed) as YAML for the run record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")