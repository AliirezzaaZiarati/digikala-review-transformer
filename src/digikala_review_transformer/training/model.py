"""Model: one shared encoder and two independent linear heads (rec, band).

Light imports only: this module also runs on the training machine.
"""

# ---- Section: Imports ----
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from digikala_review_transformer import labels
from digikala_review_transformer.training import pack, run_config

# ---- Section: Constants ----
HEADS = pack.HEADS
HEAD_INIT_STD = 0.02  # plan: normal init, std ~0.02, zero bias
FEATURE_POSITION = 0  # plan: last_hidden_state[:, 0]
POOLER_PREFIX = "pooler."
GROUP_ORDER = (("encoder", True), ("encoder", False), ("heads", True), ("heads", False))


class ModelError(RuntimeError):
    """Raised when the encoder or the model is not what the training code expects."""


# ---- Section: Seeding ----
def seed_everything(seed):
    """Seed python, numpy and torch (CPU and all GPUs)."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---- Section: Encoder ----
def load_encoder(pretrained_path):
    """Load the pretrained encoder in fp32 and drop its pooler. Returns (encoder, load_info)."""
    path = Path(pretrained_path)
    if not path.is_dir():
        raise ModelError(f"model folder not found: {path} -> run scripts/download_pretrained_models.py")
    encoder, info = AutoModel.from_pretrained(path, dtype=torch.float32, output_loading_info=True)

    missing = sorted(info["missing_keys"])
    missing_real = [key for key in missing if not key.startswith(POOLER_PREFIX)]
    if missing_real:
        raise ModelError(
            f"{len(missing_real)} encoder weights are missing in the checkpoint (first: {missing_real[:5]}) "
            f"-> they would be randomly initialised; check the model folder"
        )
    if info.get("mismatched_keys"):
        raise ModelError(f"checkpoint shapes differ from the config: {list(info['mismatched_keys'])[:5]} "
                         f"-> check the model folder")

    # The pooler is not used by this project (features come from position 0).
    if getattr(encoder, "pooler", None) is not None:
        encoder.pooler = None
    load_info = {
        "unexpected_keys": len(info["unexpected_keys"]),  # e.g. the masked-LM head, intentionally unused
        "pooler_keys_missing": len(missing) - len(missing_real),
    }
    return encoder, load_info


# ---- Section: Two-head classifier ----
class TwoHeadClassifier(nn.Module):
    """Shared encoder -> dropout -> one Linear per head. forward returns {head: logits (float32)}."""

    def __init__(self, encoder, head_dropout):
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(float(head_dropout))
        self.heads = nn.ModuleDict({head: nn.Linear(hidden, len(labels.CLASSES[head])) for head in HEADS})
        for layer in self.heads.values():
            nn.init.normal_(layer.weight, mean=0.0, std=HEAD_INIT_STD)
            nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        features = self.dropout(hidden[:, FEATURE_POSITION])
        return {head: layer(features).float() for head, layer in self.heads.items()}


def build_model(cfg):
    """Build the model for one run config. Returns (model, load_info)."""
    encoder, load_info = load_encoder(run_config.project_path(cfg["model"]["pretrained_path"]))
    seed_everything(cfg["seed"])  # right before the heads: same init for variants B and C
    model = TwoHeadClassifier(encoder, cfg["model"]["head_dropout"])
    return model, load_info


# ---- Section: Checks and summaries ----
def check_token_ids_fit(model, max_token_id, pad_id):
    """Every token id must exist in the embedding; the pad id must agree with the model config."""
    rows = model.encoder.get_input_embeddings().num_embeddings
    if max_token_id >= rows:
        raise ModelError(f"max token id {max_token_id} >= embedding rows {rows} -> tokenizer and model do not match")
    config_pad = getattr(model.encoder.config, "pad_token_id", None)
    if config_pad is not None and int(config_pad) != int(pad_id):
        raise ModelError(f"tokenizer pad id {pad_id} != model config pad id {config_pad} -> padding would hit a real token")


def count_parameters(model):
    """Parameter counts (shared tensors are counted once)."""
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "heads": sum(p.numel() for p in model.heads.parameters()),
    }


def model_summary(model, load_info):
    """Plain dict for the run summary."""
    return {
        "encoder_class": type(model.encoder).__name__,
        "hidden_size": int(model.encoder.config.hidden_size),
        "head_dropout": float(model.dropout.p),
        "classes": {head: list(labels.CLASSES[head]) for head in HEADS},
        "dtype": str(next(model.parameters()).dtype),
        "parameters": count_parameters(model),
        "load_info": load_info,
    }


def save_encoder_config(model, path):
    """Write the encoder's Hugging Face config as JSON (for later phases)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.encoder.config.to_dict(), indent=2, default=str), encoding="utf-8")


# ---- Section: Optimizer parameter groups ----
def parameter_groups(model, lr_encoder, lr_heads, weight_decay):
    """Four groups: encoder/heads x decay/no_decay. Weight decay only on tensors with ndim >= 2."""
    buckets = {key: [] for key in GROUP_ORDER}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        part = "encoder" if name.startswith("encoder.") else "heads"
        buckets[(part, parameter.ndim >= 2)].append(parameter)
    groups = []
    for (part, decays), params in buckets.items():
        if not params:
            continue
        groups.append({
            "params": params,
            "lr": float(lr_encoder if part == "encoder" else lr_heads),
            "weight_decay": float(weight_decay) if decays else 0.0,
            "name": f"{part}_{'decay' if decays else 'no_decay'}",
        })
    return groups