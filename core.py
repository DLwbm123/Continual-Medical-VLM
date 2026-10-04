"""Independent, minimal components for the first-round feasibility study."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def load_biomedclip(snapshot):
    import open_clip
    from open_clip.factory import _MODEL_CONFIGS
    from open_clip.tokenizer import HFTokenizer
    snapshot = Path(snapshot)
    cfg = json.loads((snapshot / "open_clip_config.json").read_text())
    model_cfg = copy.deepcopy(cfg["model_cfg"])
    # The full checkpoint contains text weights; only the pinned BERT config is needed.
    model_cfg["text_cfg"]["hf_model_name"] = str(snapshot)
    model_cfg["text_cfg"]["hf_tokenizer_name"] = str(snapshot)
    model_cfg["text_cfg"]["hf_model_pretrained"] = False
    _MODEL_CONFIGS["biomedclip_local"] = model_cfg
    model, train_transform, val_transform = open_clip.create_model_and_transforms(
        "biomedclip_local", pretrained=str(snapshot / "open_clip_pytorch_model.bin"),
        pretrained_text=False, **{f"image_{k}": v for k, v in cfg["preprocess_cfg"].items()})
    context_length = model_cfg["text_cfg"]["context_length"]
    tokenizer = HFTokenizer(str(snapshot), context_length=context_length)
    return model, train_transform, val_transform, tokenizer, context_length


class ClinicalModel(nn.Module):
    """Parallel heads are concept prediction, not proof of diagnostic dependence."""
    def __init__(self, backbone, feature_dim, task_classes, concept_groups=()):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.diagnosis = nn.ModuleDict()
        self.concept_specs = list(concept_groups)
        if len({g["id"] for g in concept_groups}) != len(concept_groups):
            raise ValueError("Concept IDs must be unique")
        self.concepts = nn.ModuleDict({f"g{i}": nn.Linear(feature_dim, g["outputs"])
                                      for i, g in enumerate(concept_groups)})
        for task, classes in task_classes.items():
            self.add_task(task, classes)
        for parameter in self.backbone.text.parameters():
            parameter.requires_grad_(False)
        self.backbone.logit_scale.requires_grad_(False)

    def add_task(self, task, classes):
        if task in self.diagnosis:
            raise ValueError("Refusing to reset an existing task head")
        self.diagnosis[task] = nn.Linear(self.feature_dim, classes)

    def acquire(self, task):
        for name, head in self.diagnosis.items():
            head.requires_grad_(name == task)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.text.eval()
        return self

    def forward(self, images, task):
        h = self.backbone.encode_image(images, normalize=False)
        concepts = {g["id"]: self.concepts[f"g{i}"](h) for i, g in enumerate(self.concept_specs)}
        return self.diagnosis[task](h), concepts, F.normalize(h, dim=-1)


def masked_concept_loss(logits, targets, specs, reference):
    losses = []
    for spec in specs:
        key = spec["id"]
        if key not in targets:
            continue
        truth, valid = targets[key]["value"], targets[key]["valid"].bool()
        pred = logits[key]
        if spec["type"] == "categorical":
            if valid.ndim != 1 or truth.ndim != 1:
                raise ValueError("Categorical concepts need a per-sample validity mask")
            if valid.any():
                losses.append(F.cross_entropy(pred[valid], truth[valid].long()))
        elif spec["type"] in {"binary", "multilabel"}:
            if pred.shape != truth.shape or pred.shape != valid.shape:
                raise ValueError("BCE concepts require elementwise targets and validity")
            if valid.any():
                losses.append(F.binary_cross_entropy_with_logits(pred[valid], truth[valid].float()))
        else:
            raise ValueError("Unknown concept target type")
    return torch.stack(losses).mean() if losses else reference.sum() * 0


def multi_positive_alignment(image_features, text_features, positives, scale):
    scores = scale * F.normalize(image_features, dim=-1) @ F.normalize(text_features, dim=-1).T
    if positives.shape != scores.shape or not positives.any(1).all() or not positives.any(0).all():
        raise ValueError("Every image and text must have a known positive")
    def direction(s, p):
        return (torch.logsumexp(s, 1) - torch.logsumexp(s.masked_fill(~p, -torch.inf), 1)).mean()
    return (direction(scores, positives) + direction(scores.T, positives.T)) / 2


def frozen_teacher(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


def diagnosis_kd(student, teacher):
    return F.kl_div(F.log_softmax(student, -1), F.softmax(teacher.detach(), -1), reduction="batchmean")


def supported_concept_kd(student, teacher, targets, specs, reference):
    """Only verified labels and correct teacher predictions are retained."""
    losses, coverage = [], {}
    if set(student) != {s["id"] for s in specs} or set(teacher) != set(student):
        raise ValueError("Concept IDs are not aligned")
    for spec in specs:
        key = spec["id"]
        if key not in targets:
            coverage[key] = 0
            continue
        y, valid = targets[key]["value"], targets[key]["valid"].bool()
        t, s = teacher[key].detach(), student[key]
        if spec["type"] == "categorical":
            keep = valid & (t.argmax(-1) == y)
            if keep.any():
                losses.append(diagnosis_kd(s[keep], t[keep]))
        else:
            keep = valid & ((t.sigmoid() >= .5) == (y >= .5))
            if keep.any():
                p = t[keep].sigmoid().clamp(1e-6, 1 - 1e-6)
                losses.append((F.binary_cross_entropy_with_logits(s[keep], p, reduction="none")
                               + p * p.log() + (1 - p) * (1 - p).log()).mean())
        coverage[key] = int(keep.sum())
    return (torch.stack(losses).mean() if losses else reference.sum() * 0), coverage


def choose_replay(rows, per_class, seed):
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Replay may only use training samples")
    rng = random.Random(seed)
    selected = []
    for label in sorted({r["diagnosis_label"] for r in rows}):
        candidates = sorted(r["sample_id"] for r in rows if r["diagnosis_label"] == label)
        selected.extend(rng.sample(candidates, min(per_class, len(candidates))))
    return sorted(selected)


def validate_group_separation(rows):
    owners, unknown = {}, 0
    for row in rows:
        group = row.get("patient_id")
        if group is None:
            unknown += 1
            continue
        if group in owners and owners[group] != row["split"]:
            raise ValueError("Patient appears in multiple splits")
        owners[group] = row["split"]
    return {"verified": unknown == 0, "unknown_samples": unknown}


def classification_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    pred = probabilities.argmax(1)
    auc, support = {}, {}
    for k in range(probabilities.shape[1]):
        y = labels == k
        support[str(k)] = {"positive": int(y.sum()), "negative": int((~y).sum())}
        auc[str(k)] = float(roc_auc_score(y, probabilities[:, k])) if y.any() and (~y).any() else None
    defined = [v for v in auc.values() if v is not None]
    return {"accuracy": float(accuracy_score(labels, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, labels=list(range(probabilities.shape[1])), average="macro", zero_division=0)),
            "macro_ovr_auc": float(np.mean(defined)) if len(defined) == probabilities.shape[1] else None,
            "auc_undefined_reason": None if len(defined) == probabilities.shape[1] else "At least one class lacks positives or negatives",
            "per_class_auc": auc, "support": support, "n": len(labels)}


def save_checkpoint(path, model, optimizer, step, sampler_generator, config):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "sampler_rng": sampler_generator.get_state(), "config": config}, path)


def restore_checkpoint(path, model, optimizer, sampler_generator):
    # Only checkpoints generated by this project are accepted by the caller.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    sampler_generator.set_state(state["sampler_rng"])
    return state["step"]
