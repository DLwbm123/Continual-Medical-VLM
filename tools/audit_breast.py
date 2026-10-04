"""BrEaST v1 adapter. Paths/IDs and row-level labels stay in private output."""
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SOURCE = "https://doi.org/10.7937/9WKK-Q141"


def read_table(path):
    # This released workbook uses shared strings and a single data sheet.
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(t.itertext()) for t in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        records = []
        for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall(".//m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                value = cell.find("m:v", NS)
                text = "" if value is None else value.text
                column = "".join(c for c in cell.attrib["r"] if c.isalpha())
                values[column] = strings[int(text)] if cell.attrib.get("t") == "s" else text
            records.append(values)
    headers = records[0]
    return [{headers[k].strip(): v for k, v in row.items()} for row in records[1:] if row.get("A")]


def encode_annotation(raw, spec):
    raw = (raw or "").strip()
    missing = {"": "not_annotated", "not available": "not_annotated", "not applicable": "not_applicable",
               "indefinable": "not_assessable"}
    if raw in missing:
        return {"value": None, "valid": False, "status": missing[raw], "source": SOURCE,
                "applicable": False if raw == "not applicable" else None,
                "assessable": False if raw == "indefinable" else None,
                "annotated": raw not in {"", "not available"}, "reliability": "source radiologist annotation"}
    if spec["type"] == "multilabel":
        selected = raw.removeprefix("not circumscribed - ").split("&")
        if not set(selected) <= set(spec["values"]):
            raise ValueError(f"Unknown {spec['id']} value")
        value = [int(v in selected) for v in spec["values"]]
    elif spec["type"] == "binary":
        if raw not in {"yes", "no"}: raise ValueError("Unknown binary label")
        value = [int(raw == "yes")]
    else:
        value = spec["values"].index(raw)
    return {"value": value, "valid": True, "status": "annotated", "source": SOURCE,
            "applicable": True, "assessable": True, "annotated": True, "reliability": "source radiologist annotation"}


def assign_splits(rows, seed=0):
    rng = random.Random(seed)
    for label in sorted({r["diagnosis_label"] for r in rows}):
        group = sorted([r for r in rows if r["diagnosis_label"] == label], key=lambda r: r["sample_id"])
        rng.shuffle(group)
        a, b = int(.6 * len(group)), int(.2 * len(group))
        for i, row in enumerate(group):
            row["split"] = "train" if i < a else "val" if i < a + b else "test"
    return rows


def audit(root, output, schema_path):
    from PIL import Image
    root, output = Path(root), Path(output)
    if output.exists(): raise ValueError("Refusing to replace a locked audit/split")
    specs = json.loads(Path(schema_path).read_text())["groups"]
    table = read_table(root / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx")
    files = {p.name: p for p in (root / "images").rglob("*.png")}
    if len(files) != len(list((root / "images").rglob("*.png"))): raise ValueError("Duplicate basenames")
    rows, excluded, used = [], [], set()
    for record in table:
        case = record["CaseID"]
        name = record["Image_filename"]
        if "_tumor" in name or "_other" in name: raise ValueError("Mask used as image")
        if name in used: raise ValueError("Image reused across cases")
        with Image.open(files[name]) as im: size = im.size
        masks = []
        for field in ["Mask_tumor_filename", "Mask_other_filename"]:
            for mask in filter(None, record.get(field, "").split("&")):
                with Image.open(files[mask]) as im:
                    if im.size != size: raise ValueError("Mask size mismatch")
                masks.append(str(files[mask].relative_to(root))); used.add(mask)
        used.add(name)
        annotations = {g["id"]: encode_annotation(record.get(g["field"]), g) for g in specs}
        row = {"sample_id": "BrEaST:" + case, "patient_id": "BrEaST:source_case:" + case,
               "patient_group_basis": "Source publication: one image per patient; case ID, not a recovered clinical ID",
               "task_id": "BrEaST", "image_path": str(files[name].relative_to(root)), "mask_paths": masks,
               "image_size": size, "diagnosis_name": record["Classification"], "concept_annotations": annotations,
               "diagnosis_verification_source": record.get("Verification", "")}
        if row["diagnosis_name"] == "normal":
            row.update(split="excluded", exclusion_reason="Binary lesion anchor; normal has no target lesion")
            excluded.append(row)
        else:
            row["diagnosis_label"] = ["benign", "malignant"].index(row["diagnosis_name"]); rows.append(row)
    if len({r["sample_id"] for r in rows + excluded}) != len(table): raise ValueError("Repeated case ID")
    if set(files) != used: raise ValueError("Unreferenced image/mask files")
    assign_splits(rows)
    summary = {"source": SOURCE, "version": "TCIA v1, 2024-01-08; files dated 2023-12-15", "license": "CC BY 4.0",
               "n_cases": len(table), "n_images": len(table), "n_masks": len(files) - len(table),
               "excluded_normal": len(excluded), "group_disjoint": True, "group_basis": rows[0]["patient_group_basis"],
               "split_seed": 0, "split_rule": "Per diagnosis shuffle, floor(60%) train, floor(20%) validation, remainder test",
               "concepts": {}, "splits": {}}
    for split in ["train", "val", "test"]:
        selected = [r for r in rows if r["split"] == split]
        summary["splits"][split] = dict(Counter(r["diagnosis_name"] for r in selected))
        summary["concepts"][split] = {}
        for g in specs:
            ann = [r["concept_annotations"][g["id"]] for r in selected]
            valid = [a["value"] for a in ann if a["valid"]]
            counts = [sum(v == k for v in valid) for k in range(g["outputs"])] if g["type"] == "categorical" else [sum(v[k] for v in valid) for k in range(g["outputs"])]
            summary["concepts"][split][g["id"]] = {"valid_cases": len(valid), "status": dict(Counter(a["status"] for a in ann)),
                                                         "positive_counts": counts, "negative_counts": [len(valid)-v for v in counts]}
    output.mkdir(parents=True)
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + excluded))
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    import os
    print(json.dumps(audit(os.environ["DATA_ROOT"], os.environ["AUDIT_OUTPUT"], os.environ["CONCEPT_SCHEMA"]), indent=2))
