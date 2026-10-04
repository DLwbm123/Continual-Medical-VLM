"""Validate the actual publisher split records; never infer clinical concepts."""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
from PIL import Image


def audit(data_root, output, inventory_file):
    data_root, output = Path(data_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Audit output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    physical = {}
    with open(inventory_file) as stream:
        for line in stream:
            row = json.loads(line)
            if row["role_hint"] == "ordinary_image_candidate":
                physical[row["relative_path"]] = row["bytes"]
    report = {"schema": 1, "datasets": {}, "errors": [], "data_modified": False,
              "patient_independence": "unverified: publisher splits have no group IDs",
              "concept_truth": "absent: records contain path, diagnosis ID and class name only",
              "duplicate_scope": "same source path only; no image hashes or near-duplicate audit"}
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w") as manifest:
        for split_file in sorted(data_root.glob("*/split_*.json")):
            task = split_file.parent.name
            obj = json.loads(split_file.read_text())
            seen, class_names, split_counts, dimensions, modes = {}, {}, {}, Counter(), Counter()
            sizes = []
            task_errors = []
            referenced = set()
            for phase in ("train", "val", "test"):
                counts = Counter()
                for idx, record in enumerate(obj[phase]):
                    if not isinstance(record, list) or len(record) != 3:
                        raise ValueError(f"Unsupported record schema in {task}/{phase}")
                    relative, label, name = record
                    rel = PurePosixPath(relative)
                    if rel.is_absolute() or ".." in rel.parts or not isinstance(label, int):
                        raise ValueError("Unsafe path or invalid diagnosis ID")
                    if "mask" in rel.name.lower():
                        raise ValueError("Mask-like classification input requires manual review")
                    if label in class_names and class_names[label] != name:
                        raise ValueError("Class ID has inconsistent names")
                    class_names[label] = name
                    path = data_root / task / relative
                    key = f"{task}/{relative}"
                    if key in seen:
                        task_errors.append({"kind": "repeated_source_path", "splits": [seen[key], phase], "path": relative})
                    seen[key] = phase
                    referenced.add(key)
                    if key not in physical or not path.is_file():
                        task_errors.append({"kind": "missing_image", "path": relative})
                        continue
                    try:
                        with Image.open(path) as image:
                            width, height = image.size
                            mode = image.mode
                    except Exception as exc:
                        task_errors.append({"kind": "unreadable_header", "path": relative, "error": type(exc).__name__})
                        continue
                    dimensions[(width, height)] += 1
                    modes[mode] += 1
                    sizes.append(physical[key])
                    counts[label] += 1
                    row = {"sample_id": f"{task.lower()}:{phase}:{idx:06d}", "task_id": task,
                           "split": phase, "image_path": key, "diagnosis_label": label,
                           "diagnosis_name": name, "width": width, "height": height,
                           "patient_id": None, "lesion_id": None, "slide_id": None,
                           "group_identity_status": "not_provided", "concept_annotations": [],
                           "concept_status": "not_annotated", "mask_path": None,
                           "annotation_source": "publisher_split_json"}
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                split_counts[phase] = dict(sorted(counts.items()))
            if sorted(class_names) != list(range(len(class_names))):
                raise ValueError(f"Non-contiguous label IDs in {task}")
            actual = {key for key in physical if key.startswith(task + "/")}
            extra = sorted(actual - referenced)
            report["datasets"][task] = {
                "class_names": {str(k): v for k, v in sorted(class_names.items())},
                "split_class_counts": split_counts, "num_images": sum(sum(x.values()) for x in split_counts.values()),
                "image_bytes": sum(sizes), "dimension_counts": {f"{w}x{h}": n for (w, h), n in dimensions.items()},
                "modes": dict(modes), "capability": "A", "verified_concept_labels": 0,
                "provided_patient_ids": 0, "provided_lesion_ids": 0, "provided_slide_ids": 0,
                "provided_masks": 0, "unreferenced_images": len(extra), "errors": task_errors,
                "header_check": "all referenced image headers opened; pixels not exhaustively decoded"}
            report["errors"].extend({"task": task, **x} for x in task_errors)
            if extra:
                report["errors"].append({"task": task, "kind": "unreferenced_images", "count": len(extra)})
            print("AUDITED", task, report["datasets"][task]["num_images"], "errors", len(task_errors), flush=True)
    report["status"] = "passed_classification_structure" if not report["errors"] else "needs_review"
    (output / "audit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "concept_coverage.json").write_text(json.dumps({"verified_annotations": 0, "anchor_task": None,
        "status": "blocked_missing_image_level_concept_truth", "values_imputed": False}, indent=2) + "\n")
    return report
