"""First-round single-task feasibility with verified, typed concept supervision."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from concepts import targets_for, vocabulary_logits, concept_metrics, wilson

from core import (ClinicalModel, classification_metrics, load_biomedclip,
                  masked_concept_loss, save_checkpoint, restore_checkpoint)


def load_rows(manifest, task, split):
    rows = []
    with open(manifest) as stream:
        for line in stream:
            row = json.loads(line)
            if row["task_id"] == task and row["split"] == split:
                rows.append(row)
    if not rows:
        raise ValueError("No samples for requested task/split")
    return rows


def image_tensor(row, data_root, preprocess):
    with Image.open(Path(data_root) / row["image_path"]) as image:
        image = image.convert("RGB")
        # Preserve the full field of view; no mask or label-dependent ROI.
        image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC,
                             color=(123, 117, 104))
        return preprocess(image)


@torch.no_grad()
def evaluate(model, rows, config, preprocess, text_features, output, label, concept_text=None):
    model.eval()
    probabilities, vl_probabilities, truths, ids = [], [], [], []
    specs = model.concept_specs
    cp = {g["id"]: [] for g in specs}; cv = {g["id"]: [] for g in specs}
    for start in range(0, len(rows), config["batch_size"]):
        chunk = rows[start:start + config["batch_size"]]
        x = torch.stack([image_tensor(r, config["data_root"], preprocess) for r in chunk]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["bf16"]):
            diagnosis, concepts, features = model(x, config["task"])
            ctext = vocabulary_logits(features, concept_text or {}, specs, model.backbone.logit_scale.exp())
        for g in specs:
            key = g["id"]
            for dest, logits in [(cp, concepts), (cv, ctext)]:
                p = logits[key].float().softmax(-1) if g["type"] == "categorical" else logits[key].float().sigmoid()
                dest[key].extend(p.cpu().numpy())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["bf16"]):
            vl = model.backbone.logit_scale.exp() * features @ text_features.T
        probabilities.extend(diagnosis.float().softmax(-1).cpu().numpy())
        vl_probabilities.extend(vl.float().softmax(-1).cpu().numpy())
        truths.extend(r["diagnosis_label"] for r in chunk)
        ids.extend(r["sample_id"] for r in chunk)
    output = Path(output)
    np.savez_compressed(output / (label + "_predictions.npz"), sample_ids=np.array(ids), labels=np.array(truths),
                        diagnosis=np.array(probabilities), vision_language=np.array(vl_probabilities),
                        **{f"concept_{k}": np.array(v) for k, v in cp.items()},
                        **{f"concept_vl_{k}": np.array(v) for k, v in cv.items()})
    return {"diagnosis": classification_metrics(truths, probabilities),
            "vision_language": classification_metrics(truths, vl_probabilities),
            "diagnosis_accuracy_wilson95": wilson(int(np.sum(np.argmax(probabilities, 1) == truths)), len(truths)),
            "concept": concept_metrics(rows, cp, specs) if specs else None,
            "concept_vision_language": concept_metrics(rows, cv, specs) if specs else None,
            "concept_unavailable_reason": None if specs else "No verified image-level concept annotations"}


def main(config):
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()) and not config.get("resume"):
        raise ValueError("Refusing to overwrite an existing run")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sampler = torch.Generator().manual_seed(seed)
    started = time.monotonic()
    backbone, _, preprocess, tokenizer, context_length = load_biomedclip(config["snapshot"])
    backbone = backbone.cuda().eval()
    with torch.no_grad():
        feature_dim = backbone.encode_image(torch.zeros(1, 3, 224, 224, device="cuda")).shape[-1]
    train = load_rows(config["manifest"], config["task"], "train")
    validation = load_rows(config["manifest"], config["task"], "val")
    class_map = {r["diagnosis_label"]: r["diagnosis_name"] for r in train}
    if sorted(class_map) != list(range(len(class_map))):
        raise ValueError("Training split does not cover the declared class space")
    texts = [config["prompt_template"].format(label=class_map[k]) for k in range(len(class_map))]
    with torch.no_grad():
        text_features = backbone.encode_text(tokenizer(texts, context_length=context_length).cuda(), normalize=True)
    if text_features.shape[-1] != feature_dim:
        raise ValueError("Image/text dimensions disagree")
    specs = json.loads(Path(config["concept_schema"]).read_text())["groups"] if config.get("concept_schema") else []
    concept_text = {}
    with torch.no_grad():
        for g in specs:
            concept_text[g["id"]] = backbone.encode_text(tokenizer(g["candidate_texts"], context_length=context_length).cuda(), normalize=True)
    model = ClinicalModel(backbone, feature_dim, {config["task"]: len(class_map)}, concept_groups=specs).cuda()
    model.acquire(config["task"])
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.visual.parameters() if p.requires_grad], "lr": config["backbone_lr"]},
        {"params": list(model.diagnosis[config["task"]].parameters()) + list(model.concepts.parameters()), "lr": config["head_lr"]}], weight_decay=config["weight_decay"])
    runtime = {"feature_dimension": feature_dim, "context_length": context_length,
               "preprocess": "Full-frame aspect-preserving letterbox to 224, then checkpoint evaluation transform; no random augmentation",
               "text_prompts": texts, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "frozen_text_parameters": sum(p.numel() for p in model.backbone.text.parameters()),
               "concept_heads": len(specs), "concept_truth": "BrEaST source radiologist labels" if specs else "absent", "test_evaluated": False,
               "gpu": torch.cuda.get_device_name(), "pid": os.getpid()}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print("RUNTIME", json.dumps(runtime), flush=True)
    results = {"protocol": "single-task concept feasibility" if specs else "classification engineering feasibility only", "runtime": runtime, "evaluations": {}}
    if config["mode"] == "smoke_overfit":
        initial = evaluate(model, validation, config, preprocess, text_features, output, "zero_shot_val", concept_text)
        results["zero_shot_validation"] = initial["vision_language"]
        results["zero_shot_concept_validation"] = initial["concept_vision_language"]
        selected = []
        for label in sorted(class_map):
            candidates = [r for r in train if r["diagnosis_label"] == label]
            order = torch.randperm(len(candidates), generator=sampler).tolist()
            selected.extend(candidates[i] for i in order[:config["overfit_per_class"]])
        train = selected
        results["overfit_before"] = evaluate(model, train, config, preprocess, text_features, output, "overfit_before", concept_text)
        (output / "overfit_sample_ids.json").write_text(json.dumps([r["sample_id"] for r in train]))
    step0 = restore_checkpoint(config["resume"], model, optimizer, sampler) if config.get("resume") else 0
    step_started = time.monotonic()
    with (output / "learning_curve.jsonl").open("a") as curve:
        for step in range(step0 + 1, config["updates"] + 1):
            model.train()
            indices = torch.randint(len(train), (min(config["batch_size"], len(train)),), generator=sampler).tolist()
            batch = [train[i] for i in indices]
            x = torch.stack([image_tensor(r, config["data_root"], preprocess) for r in batch]).cuda()
            y = torch.tensor([r["diagnosis_label"] for r in batch], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["bf16"]):
                diagnosis, concept, features = model(x, config["task"])
                vl = model.backbone.logit_scale.exp() * features @ text_features.T
                dy, lv = F.cross_entropy(diagnosis, y), F.cross_entropy(vl, y)
                targets = targets_for(batch, specs, "cuda")
                lc = masked_concept_loss(concept, targets, specs, features)
                cv_logits = vocabulary_logits(features, concept_text, specs, model.backbone.logit_scale.exp())
                lcv = masked_concept_loss(cv_logits, targets, specs, features)
                loss = dy + config["beta_vl"] * (lv + lcv) + config.get("alpha_concept", 1.) * lc
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            record = {"step": step, "diagnosis_loss": float(dy.detach()), "vl_loss": float(lv.detach()),
                      "concept_loss": float(lc.detach()), "concept_vl_loss": float(lcv.detach()), "loss": float(loss.detach()),
                      "batch_accuracy": float((diagnosis.argmax(-1) == y).float().mean()),
                      "elapsed_training_seconds": time.monotonic() - step_started}
            curve.write(json.dumps(record) + "\n"); curve.flush()
            if step == 20 or step % 100 == 0:
                print("UPDATE", json.dumps(record), flush=True)
            if config["mode"] == "single_task" and step % config["eval_every"] == 0:
                result = evaluate(model, validation, config, preprocess, text_features, output, f"val_{step}", concept_text)
                results["evaluations"][str(step)] = result
                print("VALIDATION", step, json.dumps(result), flush=True)
                (output / "results.partial.json").write_text(json.dumps(results, indent=2) + "\n")
    if config["mode"] == "smoke_overfit":
        results["overfit_after"] = evaluate(model, train, config, preprocess, text_features, output, "overfit_after", concept_text)
        results["overfit_gate_passed"] = results["overfit_after"]["diagnosis"]["accuracy"] >= .95
    else:
        results["final_validation"] = results["evaluations"].get(str(config["updates"])) or evaluate(
            model, validation, config, preprocess, text_features, output, "final_val", concept_text)
    save_checkpoint(output / "final.pt", model, optimizer, config["updates"], sampler, config)
    results.update({"status": "complete", "updates": config["updates"], "elapsed_seconds": time.monotonic() - started,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "seed": seed,
                    "concept_stage_gate_passed": False, "concept_stage_gate_reason": "Requires per-group validation and M1 review before M2" if specs else "Verified image-level concepts are unavailable"})
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("COMPLETED", json.dumps(results), flush=True)


if __name__ == "__main__":
    config = json.loads(Path(os.environ["RUN_CONFIG"]).read_text())
    config = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in config.items()}
    main(config)
