"""Read-only audit of the locally supplied 2026 E-problem data.

Usage: python E2026/audit_data.py --data-root ../E题数据/E题数据 --out E2026/data_audit.json
The 1 GB / 2.9 GB training pickles are intentionally not unpickled here.
Only run pickle loading on the contest files you trust.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class CompatibleUnpickler(pickle.Unpickler):
    """Allow NumPy 2's module path for the small specialty samples on NumPy 1.x."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def excel_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            strings = ["".join(n.itertext()) for n in root.findall("m:si", NS)]
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall(".//m:sheetData/m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                col = re.match(r"[A-Z]+", cell.attrib.get("r", ""))
                if not col:
                    continue
                value = cell.find("m:v", NS)
                if value is None:
                    inline = cell.find("m:is", NS)
                    values[col.group()] = "".join(inline.itertext()) if inline is not None else ""
                elif cell.attrib.get("t") == "s":
                    values[col.group()] = strings[int(value.text or 0)]
                else:
                    values[col.group()] = value.text or ""
            if values:
                rows.append(values)
    if not rows:
        return []
    columns = rows[0]
    return [{columns.get(col, col): value for col, value in row.items()} for row in rows[1:]]


def longest_zero_runs(array: np.ndarray) -> list[int]:
    """Longest all-zero consecutive interval per sample, before length checks."""
    if array.ndim != 3:
        return []
    result = []
    for item in array:
        zero = np.all(item == 0, axis=-1)
        best = current = 0
        for flag in zero:
            current = current + 1 if flag else 0
            best = max(best, current)
        result.append(int(best))
    return result


def summarize_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        obj = CompatibleUnpickler(handle).load()
    result = {"file": path.name, "top_keys": list(obj) if isinstance(obj, dict) else None}
    split = obj.get("test", obj) if isinstance(obj, dict) else obj
    if not isinstance(split, dict):
        result["error"] = "expected dictionary"
        return result
    result["fields"] = {
        key: {"type": type(value).__name__, "shape": list(value.shape) if hasattr(value, "shape") else None}
        for key, value in split.items()
    }
    result["longest_all_zero_run"] = {
        key: longest_zero_runs(value)
        for key, value in split.items()
        if key in ("text", "audio", "vision") and isinstance(value, np.ndarray)
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    assert root.is_dir(), root

    a1 = next(root.glob("附件1-*/MOSEI*/label-100.xlsx"))
    videos = list(a1.parent.rglob("*.mp4"))
    labels1 = excel_rows(a1)
    headers1 = sorted({key for row in labels1 for key in row})
    video_keys = {(p.parent.name, p.stem) for p in videos}
    label_keys = {(str(row.get("video_id", "")), str(row.get("clip_id", ""))) for row in labels1}
    # Excel numeric clip_id may be stored as 1 while filenames use 1, 01 or 001.
    normalize = lambda pair: (pair[0], str(int(float(pair[1]))) if pair[1] else "")
    video_keys_norm = {normalize(pair) for pair in video_keys}
    label_keys_norm = {normalize(pair) for pair in label_keys}

    a2dir = next(root.glob("附件2-*"))
    labels2 = excel_rows(a2dir / "label.xlsx")
    a2files = {p.name: p.stat().st_size for p in a2dir.glob("*.pkl")}

    specialty = {}
    for number in (3, 4):
        top = next(root.glob(f"附件{number}-*"))
        specialty[str(number)] = {}
        for version in ("对齐版本", "未对齐版本"):
            files = sorted(top.rglob(f"{version}/*.pkl"))
            videos_version = list(top.rglob(f"{version}/videos/*.mp4"))
            data = [summarize_pickle(p) for p in files]
            specialty[str(number)][version] = {
                "pickle_count": len(files),
                "video_count": len(videos_version),
                "pickle_total_bytes": sum(p.stat().st_size for p in files),
                "field_patterns": dict(Counter(",".join(sorted(item.get("fields", {}))) for item in data)),
                "first_sample": data[0] if data else None,
                "load_errors": [item for item in data if "error" in item],
            }

    report = {
        "attachment1": {
            "video_count": len(videos),
            "label_rows": len(labels1),
            "label_columns": headers1,
            "unique_video_keys": len(video_keys_norm),
            "unique_label_keys": len(label_keys_norm),
            "videos_without_label": sorted(video_keys_norm - label_keys_norm)[:20],
            "labels_without_video": sorted(label_keys_norm - video_keys_norm)[:20],
            "label_values": dict(Counter(row.get("annotation", "") for row in labels1)),
        },
        "attachment2": {
            "label_rows": len(labels2),
            "label_columns": sorted({key for row in labels2 for key in row}),
            "pickle_files_bytes": a2files,
            "pickle_internal_structure": "not_loaded_due_to_memory_limit",
        },
        "specialty": specialty,
        "environment": {"numpy": np.__version__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "attachment1_video_count": len(videos),
        "attachment1_label_rows": len(labels1),
        "attachment1_unmatched_video_count": len(video_keys_norm - label_keys_norm),
        "attachment1_unmatched_label_count": len(label_keys_norm - video_keys_norm),
        "attachment2_label_rows": len(labels2),
        "specialty_counts": {k: {v: d["pickle_count"] for v, d in x.items()} for k, x in specialty.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
