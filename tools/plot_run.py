"""Aggregate scientific learning curves; never includes patient-level outputs."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path(os.environ["RUN_OUTPUT"])
result = json.loads((out / "results.json").read_text())
curve = [json.loads(line) for line in (out / "learning_curve.jsonl").read_text().splitlines()]
fig = plt.figure(figsize=(11, 7), constrained_layout=True)
grid = fig.add_gridspec(2, 2)
ax = fig.add_subplot(grid[0, 0])
for key, label in [("diagnosis_loss", "Diagnosis"), ("concept_loss", "Concept heads"), ("vl_loss", "Class text"), ("concept_vl_loss", "Concept text")]:
    y = np.array([r[key] for r in curve])
    # Fixed non-overlapping windows, not a best-run or best-checkpoint selection.
    ax.plot(np.array([r["step"] for r in curve])[19::20], y.reshape(-1, 20).mean(1), label=label)
ax.set(xlabel="Optimizer step", ylabel="Loss (20-step mean)", yscale="log", title="Training objectives")
ax.legend(fontsize=8)
steps = sorted(map(int, result["evaluations"]))
ax = fig.add_subplot(grid[0, 1])
for branch, label in [("diagnosis", "Diagnosis head"), ("vision_language", "Image-class text")]:
    ax.plot(steps, [result["evaluations"][str(s)][branch]["accuracy"] for s in steps], marker="o", label=label)
ax.set(xlabel="Optimizer step", ylabel="Validation accuracy", ylim=(0, 1), title="Diagnosis: 49 validation cases")
ax.legend(fontsize=8)
ax = fig.add_subplot(grid[1, :])
for key in result["final_validation"]["concept"]:
    vals = [result["evaluations"][str(s)]["concept"][key]["macro_ovr_auc"] for s in steps]
    if all(v is None for v in vals): continue
    ax.plot(steps, vals, marker="o", label=key)
ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
ax.set(xlabel="Optimizer step", ylabel="Concept macro AUROC", ylim=(0, 1), title="Parallel concept heads: full declared group vocabulary")
ax.text(.01, .03, "Calcifications: undefined (no validation intraductal positives). Rare categories have very small support.", transform=ax.transAxes, fontsize=8)
ax.legend(ncol=3, fontsize=8, loc="lower right", bbox_to_anchor=(1, .1))
fig.suptitle("BrEaST M1 | seed 0 | fixed 1,000 updates | validation only", fontsize=13)
fig.savefig(out / "learning_curves.png", dpi=180)
plt.close(fig)
