"""Validate every Q1 generated feature file and time map."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from audit_data import excel_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.features.resolve()
    rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    label_file = next(args.data_root.resolve().glob("附件1-*/MOSEI*/label-100.xlsx"))
    label_ids = {str(x["video_id"]) + "$_$" + str(int(float(x["clip_id"]))) for x in excel_rows(label_file)}
    ids = [r["sample_id"] for r in rows]
    problems = []
    totals = {"feature_bytes": 0, "mapping_bytes": 0, "word_count": 0,
              "text_bins": 0, "vision_bins": 0}
    for row in rows:
        sid = row["sample_id"]
        if row["status"] != "ok":
            problems.append(sid + ": extraction failed")
            continue
        npz_path = root / row["feature_file"]
        map_path = root / row["mapping_file"]
        if not npz_path.is_file() or not map_path.is_file():
            problems.append(sid + ": missing output")
            continue
        totals["feature_bytes"] += npz_path.stat().st_size
        totals["mapping_bytes"] += map_path.stat().st_size
        with np.load(npz_path) as feature:
            expected = {"text": (50, 259), "audio": (50, 21),
                        "vision": (50, 21), "time_edges_s": (51,)}
            for key, shape in expected.items():
                if key not in feature or feature[key].shape != shape:
                    problems.append(sid + ": wrong " + key + " shape")
                elif not np.isfinite(feature[key]).all():
                    problems.append(sid + ": nonfinite " + key)
            edges = feature["time_edges_s"]
            if not np.all(np.diff(edges) > 0):
                problems.append(sid + ": nonmonotone time edges")
            totals["text_bins"] += int(feature["text_observed"].sum())
            totals["vision_bins"] += int(feature["vision_observed"].sum())
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        if mapping["sample_id"] != sid or len(mapping["bins"]) != 50:
            problems.append(sid + ": map identity/count")
        if not mapping["text_timing_is_estimated"]:
            problems.append(sid + ": text timing provenance missing")
        words = mapping["words"]
        totals["word_count"] += len(words)
        assigned = sorted(i for b in mapping["bins"] for i in b["word_indices"])
        if assigned != list(range(len(words))):
            problems.append(sid + ": word-to-bin coverage")
        if abs(mapping["bins"][0]["start_s"]) > 1e-6:
            problems.append(sid + ": first bin not zero")
        if abs(mapping["bins"][-1]["end_s"] - mapping["duration_s"]) > 1e-6:
            problems.append(sid + ": last bin not duration")
    if len(rows) != 100 or len(set(ids)) != 100 or set(ids) != label_ids:
        problems.append("100-row ID/label coverage failed")
    summary = {"samples": len(rows), "unique_ids": len(set(ids)),
               "label_ids_matched": len(set(ids) & label_ids),
               "feature_files": len(list(root.glob("*.npz"))),
               "mapping_files": len([p for p in root.glob("*.json") if p.name != "summary.json"]),
               "totals": totals, "problems": problems,
               "text_timing": "energy-weighted estimate; not forced alignment"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
