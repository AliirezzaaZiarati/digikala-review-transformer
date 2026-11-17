"""Loss for the two-head model: weighted, label-smoothed cross entropy per head.

Per head:  e = w * class_weight[target] * (target != IGNORE_INDEX)
           loss = sum(e * ce) / max(sum(e), EPS)       (a head with no valid row contributes 0)
Total:     loss_rec + loss_weight_band * loss_band
A separate plain cross entropy (no weights, no smoothing) is tracked for reporting.
Light imports only: this module also runs on the training machine.
"""

# ---- Section: Imports ----
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from digikala_review_transformer import labels
from digikala_review_transformer.training import data, pack

# ---- Section: Constants ----
HEADS = pack.HEADS
IGNORE_INDEX = data.IGNORE_INDEX
DENOMINATOR_FLOOR = 1e-8  # plan: max(sum(e), 1e-8)


class LossError(ValueError):
    """Raised when loss settings or class weights are invalid."""


# ---- Section: Output container ----
@dataclass
class LossOutput:
    """total keeps the graph; everything else is detached (for logging)."""

    total: torch.Tensor
    head_loss: dict
    plain_ce_sum: dict
    n_valid: dict


# ---- Section: Plain cross entropy (reporting only) ----
@torch.no_grad()
def plain_ce_sum(logits, target):
    """Return (sum of unweighted, unsmoothed CE over valid rows, number of valid rows)."""
    ce = F.cross_entropy(logits.float(), target, reduction="none", ignore_index=IGNORE_INDEX)
    return ce.sum(), (target != IGNORE_INDEX).sum()


# ---- Section: Training loss ----
class TwoHeadLoss:
    """Callable: loss_fn(logits_by_head, batch) -> LossOutput."""

    def __init__(self, class_weights, label_smoothing, loss_weight_band, device):
        if not 0.0 <= float(label_smoothing) < 1.0:
            raise LossError(f"label_smoothing={label_smoothing} must be in [0, 1) -> fix the config")
        if not float(loss_weight_band) > 0.0:
            raise LossError(f"loss_weight_band={loss_weight_band} must be positive -> fix the config")
        self.label_smoothing = float(label_smoothing)
        self.loss_weight_band = float(loss_weight_band)
        self.class_weights = {}
        for head in HEADS:
            if head not in class_weights:
                raise LossError(f"class weights for head '{head}' are missing -> pass one vector per head")
            vector = torch.as_tensor(np.asarray(class_weights[head]), dtype=torch.float32, device=device)
            n_classes = len(labels.CLASSES[head])
            if vector.shape != (n_classes,):
                raise LossError(f"class weights for '{head}' have shape {tuple(vector.shape)}, expected ({n_classes},)")
            if not bool(torch.isfinite(vector).all()) or not bool((vector > 0).all()):
                raise LossError(f"class weights for '{head}' must be finite and positive, got {vector.tolist()}")
            self.class_weights[head] = vector

    def head_loss(self, head, logits, target, sample_weight):
        """Weighted, label-smoothed CE of one head; rows with target == IGNORE_INDEX do not count."""
        ce = F.cross_entropy(logits.float(), target, reduction="none",
                             label_smoothing=self.label_smoothing, ignore_index=IGNORE_INDEX)
        valid = (target != IGNORE_INDEX).to(ce.dtype)
        class_weight = self.class_weights[head][target.clamp_min(0)]  # clamp: masked rows index class 0, then get e=0
        effective = sample_weight.float() * class_weight * valid
        return (effective * ce).sum() / effective.sum().clamp_min(DENOMINATOR_FLOOR)

    def __call__(self, logits, batch):
        losses = {head: self.head_loss(head, logits[head], batch[f"y_{head}"], batch[f"w_{head}"]) for head in HEADS}
        total = losses["rec"] + self.loss_weight_band * losses["band"]
        plain = {head: plain_ce_sum(logits[head], batch[f"y_{head}"]) for head in HEADS}
        return LossOutput(
            total=total,
            head_loss={head: value.detach() for head, value in losses.items()},
            plain_ce_sum={head: plain[head][0] for head in HEADS},
            n_valid={head: plain[head][1] for head in HEADS},
        )


def build_loss(cfg, y_train, device):
    """Build the loss for one run. Class weights come from the train labels of each head only."""
    training = cfg["training"]
    weights = {}
    for head in HEADS:
        if training["use_class_weights"]:
            weights[head] = data.class_weight_vector(y_train[head], head)
        else:
            weights[head] = np.ones(len(labels.CLASSES[head]), dtype=np.float32)
    return TwoHeadLoss(weights, training["label_smoothing"], training["loss_weight_band"], device)


# ---- Section: Running averages (device-side sums, one transfer at the end) ----
class LossAccumulator:
    """Accumulates objective and plain CE; summary() returns python floats."""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.objective = torch.zeros((), dtype=torch.float64, device=self.device)
        self.steps = 0
        self.plain = {head: torch.zeros((), dtype=torch.float64, device=self.device) for head in HEADS}
        self.count = {head: torch.zeros((), dtype=torch.int64, device=self.device) for head in HEADS}

    def add_plain(self, head, ce_sum, n_valid):
        self.plain[head] += ce_sum.detach().to(torch.float64)
        self.count[head] += n_valid.to(torch.int64)

    def add(self, output):
        """Add one training step (LossOutput)."""
        self.objective += output.total.detach().to(torch.float64)
        self.steps += 1
        for head in HEADS:
            self.add_plain(head, output.plain_ce_sum[head], output.n_valid[head])

    def add_logits(self, logits, batch):
        """Add plain CE of one evaluation batch (no objective)."""
        for head in HEADS:
            ce_sum, n_valid = plain_ce_sum(logits[head], batch[f"y_{head}"])
            self.add_plain(head, ce_sum, n_valid)

    def summary(self):
        nan = float("nan")
        result = {"objective": float(self.objective.item()) / self.steps if self.steps else nan, "steps": self.steps}
        for head in HEADS:
            count = int(self.count[head].item())
            result[f"plain_ce_{head}"] = float(self.plain[head].item()) / count if count else nan
            result[f"n_valid_{head}"] = count
        return result