#!/usr/bin/env python3
"""Read-only, local dataset inventory. Does NOT infer labels/splits or train a model.

Python >= 3.10; standard library only. Source directories are never modified.
The output directory must be new (or empty) and outside every input root.
File/filename counts are NOT clinical case counts. Hashing is opt-in.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CONTAINER_EXT = {".npz", ".npy", ".h5", ".hdf5", ".zip", ".tar", ".gz", ".tgz", ".7z", ".parquet"}
ANNOTATION_EXT = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".yaml", ".yml", ".txt"}
MASK_PATTERN = re.compile(r"(?:^|[/_\-])(mask(?:s)?|seg(?:mentation)?|labelmap)(?:[/_\-.]|$)|_(?:tumor|other)\.png$", re.I)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def role_hint(path: Path, relative: str) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "possible_mask_filename_only" if MASK_PATTERN.search(relative) else "ordinary_image_candidate"
    if ext in {".dcm", ".dicom", ".nii", ".mha", ".mhd"} or relative.lower().endswith(".nii.gz"):
        return "medical_image_container_not_decoded"
    if ext in CONTAINER_EXT:
        return "container_not_enumerated"
    if ext in ANNOTATION_EXT:
        return "possible_metadata_not_validated"
    return "other"


def bounded_header(path: Path) -> dict[str, Any] | None:
    """Return field names only, not patient data. Oversized metadata is not parsed."""
    ext = path.suffix.lower()
    if ext not in {".csv", ".tsv", ".json", ".jsonl"}:
        return None
    try:
        if ext in {".csv", ".tsv"}:
            with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as f:
                first_line = f.readline(65537)
            if len(first_line) > 65536:
                return {"status": "skipped_header_too_long"}
            header = next(csv.reader([first_line], delimiter="\t" if ext == ".tsv" else ","), [])
            return {"status": "first_line_fields_only_unverified", "fields": header[:256], "truncated": len(header) > 256}
        if ext == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as f:
                text = f.readline(65537)
            if len(text) > 65536:
                return {"status": "skipped_first_record_too_long"}
            obj = json.loads(text)
        else:
            if path.stat().st_size > 2 * 1024 * 1024:
                return {"status": "skipped_json_over_2MiB"}
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(obj, list):
            obj = next((x for x in obj[:3] if isinstance(x, dict)), {})
        if isinstance(obj, dict):
            keys = list(obj)
            return {"status": "keys_only_unverified", "fields": keys[:256], "truncated": len(keys) > 256}
        return {"status": "not_a_record_mapping"}
    except (OSError, ValueError, UnicodeError, csv.Error) as exc:
        return {"status": "unreadable_or_unsupported", "error_type": type(exc).__name__}


def gpu_inventory() -> dict[str, Any]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        return {"available": r.returncode == 0, "returncode": r.returncode, "output": r.stdout.strip()[:4096]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error_type": type(exc).__name__}


def run(roots: list[Path], out: Path, max_files: int = 100000, hash_images: bool = False) -> dict[str, Any]:
    roots = [p.expanduser().resolve() for p in roots]
    out = out.expanduser().resolve()
    if max_files < 0:
        raise ValueError("max_files must be >= 0; zero means unlimited")
    if not roots or any(not p.is_dir() for p in roots):
        raise ValueError("Every --root must be an existing directory")
    for i, root in enumerate(roots):
        if out == root or out.is_relative_to(root):
            raise ValueError("Output must be outside all input roots")
        for other in roots[:i]:
            if root == other or root.is_relative_to(other) or other.is_relative_to(root):
                raise ValueError("Input roots must not overlap")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be new or empty; existing outputs are not overwritten")
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "raw_inventory_only_not_dataset_validation",
        "python": platform.python_version(), "gpu": gpu_inventory(),
        "hash_images": hash_images, "max_files_per_root": max_files,
        "source_data_modified": False, "network_requests": False,
        "labels_validated": False, "splits_validated": False,
        "concept_annotations_validated": False, "images_decoded": False,
        "warnings": [
            "Physical file counts are not sample, patient, class, or split counts.",
            "Mask and metadata roles are filename heuristics, not semantic validation.",
            "NPZ, DICOM, archives and spreadsheets are not decoded or enumerated.",
            "No hashes by default; hashing only detects identical encoded files, not near duplicates.",
            "Inventory may contain identifiers in filenames/field names; review before sharing.",
        ], "roots": [],
    }
    dup: dict[str, list[dict[str, str]]] = defaultdict(list)
    headers: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with (out / "file_inventory.jsonl").open("w", encoding="utf-8") as listing:
        for rid, root in enumerate(roots):
            ext_count: Counter[str] = Counter()
            role_count: Counter[str] = Counter()
            top: dict[str, Counter[str]] = defaultdict(Counter)
            count, total, symlinks, truncated = 0, 0, 0, False
            root_errors_start = len(errors)
            def walk_error(exc: OSError) -> None:
                errors.append({"root_id": str(rid), "path": str(exc.filename), "error_type": type(exc).__name__})
            for parent, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
                filtered = []
                for name in sorted(dirs):
                    if (Path(parent) / name).is_symlink():
                        symlinks += 1
                    else:
                        filtered.append(name)
                dirs[:] = filtered
                for name in sorted(names):
                    path = Path(parent) / name
                    if path.is_symlink():
                        symlinks += 1
                        continue
                    if max_files and count >= max_files:
                        truncated = True
                        break
                    relative = path.relative_to(root).as_posix()
                    try:
                        if not path.is_file():
                            continue
                        size = path.stat().st_size
                    except OSError as exc:
                        errors.append({"root_id": str(rid), "path": relative, "error_type": type(exc).__name__})
                        continue
                    role = role_hint(path, relative)
                    record: dict[str, Any] = {
                        "root_id": rid, "relative_path": relative,
                        "bytes": size, "extension": path.suffix.lower(), "role_hint": role,
                        "label": None, "split": None, "patient_id": None,
                    }
                    if hash_images and path.suffix.lower() in IMAGE_EXT:
                        try:
                            digest = sha256(path)
                            record["sha256"] = digest
                            dup[digest].append({"root_id": str(rid), "relative_path": relative})
                        except OSError as exc:
                            record["hash_error"] = type(exc).__name__
                            errors.append({"root_id": str(rid), "path": relative, "error_type": type(exc).__name__})
                    listing.write(json.dumps(record, ensure_ascii=False) + "\n")
                    header = bounded_header(path)
                    if header is not None:
                        headers.append({"root_id": rid, "relative_path": relative, **header})
                    count += 1
                    total += size
                    ext_count[path.suffix.lower() or "<no_extension>"] += 1
                    role_count[role] += 1
                    first = Path(relative).parts[0] if len(Path(relative).parts) > 1 else "<root_files>"
                    top[first]["physical_files"] += 1
                    top[first]["bytes"] += size
                if truncated:
                    break
            summary["roots"].append({
                "root_id": rid, "root_path": str(root), "physical_files_scanned": count,
                "bytes_scanned": total, "symlinks_skipped": symlinks,
                "coverage": "partial_file_limit" if truncated else ("partial_io_errors" if len(errors) > root_errors_start else "complete_non_symlink_file_listing"),
                "extension_counts": dict(sorted(ext_count.items())),
                "role_hint_counts": dict(sorted(role_count.items())),
                "top_level_groups": {k: dict(v) for k, v in sorted(top.items())},
            })
    summary["io_errors"] = errors
    write_json(out / "inventory_summary.json", summary)
    write_json(out / "metadata_field_candidates.json", headers)
    write_json(out / "exact_duplicate_candidates.json", {k: v for k, v in dup.items() if len(v) > 1})
    lines = ["# 本地数据只读盘点", "", "这不是数据验证报告；没有推断类别、患者、划分或概念标签。", ""]
    for root in summary["roots"]:
        lines.extend([f"## root_{root['root_id']}", f"目录：`{root['root_path']}`", f"扫描物理文件数：{root['physical_files_scanned']}", f"覆盖：{root['coverage']}", ""])
    lines.extend(["## Codex 后续必须完成", "核实数据版本、标注文件、类别映射、分组划分、重复样本、图像质量与概念真值覆盖。", "不得将本脚本输出的候选 mask 或字段名当成已验证医学标注。", "分享前检查目录与字段是否包含敏感标识。", ""])
    (out / "INVENTORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True, help="Existing local data root; may repeat; roots must not overlap")
    parser.add_argument("--out", type=Path, required=True, help="New/empty output directory, outside input roots")
    parser.add_argument("--max-files", type=int, default=100000, help="Physical files per root; 0 = unlimited")
    parser.add_argument("--hash-images", action="store_true", help="Read all ordinary encoded image files for SHA-256; may be slow")
    args = parser.parse_args()
    try:
        summary = run(args.root, args.out, args.max_files, args.hash_images)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps({"out": str(args.out.resolve()), "roots": len(summary["roots"]), "status": summary["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
