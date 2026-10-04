"""Patient-unit bootstrap for validation accuracy; reads private predictions only."""
import json
import os
from pathlib import Path
import numpy as np
from concepts import concept_metrics
from core import classification_metrics


def summarize(output, manifest, schema):
    output = Path(output)
    result = json.loads((output / "results.json").read_text())
    specs = json.loads(Path(schema).read_text())["groups"]
    all_rows = {r["sample_id"]: r for r in map(json.loads, Path(manifest).read_text().splitlines())}
    predictions = np.load(output / f"val_{result['updates']}_predictions.npz")
    rows = [all_rows[k] for k in predictions["sample_ids"]]
    if any(r["split"] != "val" for r in rows): raise ValueError("Expected validation only")
    if len({r["patient_id"] for r in rows}) != len(rows): raise ValueError("Bootstrap requires one row per patient")
    rng = np.random.default_rng(20261004)
    indices = rng.integers(0, len(rows), size=(1000, len(rows)))
    intervals = {}
    for g in specs:
        ann = [r["concept_annotations"][g["id"]] for r in rows]
        valid = np.array([a["valid"] for a in ann])
        p = predictions["concept_" + g["id"]]
        correctness = np.full(len(rows), np.nan)
        truth = np.array([a["value"] for a in ann if a["valid"]])
        if g["type"] == "categorical": correctness[valid] = p[valid].argmax(1) == truth
        else: correctness[valid] = ((p[valid] >= .5) == truth).mean(1)
        boot = np.nanmean(correctness[indices], axis=1)
        intervals[g["id"]] = {"accuracy": float(np.nanmean(correctness)),
            "patient_bootstrap95": np.quantile(boot[np.isfinite(boot)], [.025, .975]).tolist(),
            "valid_patients": int(valid.sum()), "bootstrap_replicates": 1000,
            "interpretation": "Case resampling; multilabel attributes are kept together; no seed uncertainty"}
    destination = output / "validation_accuracy_intervals.json"
    destination.write_text(json.dumps(intervals, indent=2) + "\n")
    train = [r for r in all_rows.values() if r["split"] == "train"]
    dummy = {}
    for g in specs:
        truth = [r["concept_annotations"][g["id"]]["value"] for r in train if r["concept_annotations"][g["id"]]["valid"]]
        if g["type"] == "categorical":
            value = np.eye(g["outputs"])[np.bincount(truth, minlength=g["outputs"]).argmax()]
        else:
            value = (np.mean(truth, axis=0) >= .5).astype(float)
        dummy[g["id"]] = np.tile(value, (len(rows), 1))
    label = np.bincount([r["diagnosis_label"] for r in train], minlength=2).argmax()
    baseline = {"diagnosis": classification_metrics([r["diagnosis_label"] for r in rows], np.tile(np.eye(2)[label], (len(rows), 1))),
                "concept": concept_metrics(rows, dummy, specs), "selection": "Training labels only; fixed constant prediction"}
    (output / "validation_majority_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    return intervals


if __name__ == "__main__":
    print(json.dumps(summarize(os.environ["RUN_OUTPUT"], os.environ["MANIFEST"], os.environ["CONCEPT_SCHEMA"]), indent=2))
