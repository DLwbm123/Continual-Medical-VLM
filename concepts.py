"""Typed concept targets, fixed vocabulary text alignment, and masked scoring."""
import math
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from core import classification_metrics


def targets_for(rows, specs, device="cpu"):
    targets = {}
    for g in specs:
        values, valid = [], []
        for row in rows:
            a = row.get("concept_annotations", {}).get(g["id"], {"valid": False})
            ok = a["valid"]
            values.append(a["value"] if ok else (-1 if g["type"] == "categorical" else [float("nan")] * g["outputs"]))
            valid.append(ok if g["type"] == "categorical" else [ok] * g["outputs"])
        targets[g["id"]] = {"value": torch.tensor(values, device=device), "valid": torch.tensor(valid, device=device)}
    return targets


def vocabulary_logits(features, text_features, specs, scale):
    result = {}
    for g in specs:
        scores = scale * features @ text_features[g["id"]].T
        # Two unique templates per binary attribute: present, absent.
        result[g["id"]] = scores if g["type"] == "categorical" else scores[:, 0::2] - scores[:, 1::2]
    return result


def concept_metrics(rows, probabilities, specs):
    targets = targets_for(rows, specs)
    output = {}
    for g in specs:
        y = targets[g["id"]]["value"].numpy()
        valid = targets[g["id"]]["valid"].numpy()
        p = np.asarray(probabilities[g["id"]])
        if g["type"] == "categorical":
            if not valid.any():
                output[g["id"]] = {"n": 0, "undefined_reason": "No valid labels"}; continue
            m = classification_metrics(y[valid], p[valid])
            m["values"] = g["values"]
            m["majority_accuracy"] = max(v["positive"] for v in m["support"].values()) / int(valid.sum())
            # Full-dictionary F1 is not reportable when a truth class has no support.
            if any(v["positive"] == 0 for v in m["support"].values()):
                m["macro_f1"] = None
                m["f1_undefined_reason"] = "At least one declared category has no true examples"
        else:
            entries = []
            for k, name in enumerate(g["values"]):
                v = valid[:, k]; truth = y[v, k]; pred = p[v, k] >= .5
                pos, neg = int(truth.sum()), int((1-truth).sum())
                entries.append({"value": name, "positive": pos, "negative": neg,
                    "accuracy": float(np.mean(truth == pred)) if len(truth) else None,
                    "f1": float(f1_score(truth, pred, zero_division=0)) if pos else None,
                    "auc": float(roc_auc_score(truth, p[v, k])) if pos and neg else None,
                    "undefined_reason": None if pos and neg else "No positive or negative support"})
            m = {"n": int(valid.any(1).sum()), "attributes": entries,
                 "accuracy": float(np.mean((y[valid] >= .5) == (p[valid] >= .5))) if valid.any() else None,
                 "macro_f1": float(np.mean([e["f1"] for e in entries])) if all(e["f1"] is not None for e in entries) else None,
                 "macro_ovr_auc": float(np.mean([e["auc"] for e in entries])) if all(e["auc"] is not None for e in entries) else None}
        output[g["id"]] = m
    return output


def wilson(correct, total):
    if total == 0: return None
    p, z = correct / total, 1.95996398454
    center = (p + z*z / (2*total)) / (1+z*z/total)
    delta = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [center-delta, center+delta]
