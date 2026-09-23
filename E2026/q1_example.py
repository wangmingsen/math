"""Export a 50-row Q1 example alignment table from generated files."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-id")
    args = parser.parse_args()
    root = args.features.resolve()
    rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    selected = next((r for r in rows if r["sample_id"] == args.sample_id), None) if args.sample_id else rows[0]
    if selected is None:
        raise ValueError("sample ID not in manifest")
    mapping = json.loads((root / selected["mapping_file"]).read_text(encoding="utf-8"))
    with np.load(root / selected["feature_file"]) as feature:
        audio = feature["audio"]
        vision = feature["vision"]
        output = []
        for item in mapping["bins"]:
            i = item["index"]
            output.append({
                "sample_id": selected["sample_id"],
                "bin": i,
                "start_s": item["start_s"],
                "end_s": item["end_s"],
                "estimated_text": " ".join(mapping["words"][j]["token"] for j in item["word_indices"]),
                "frame_time_s": item["frame_time_s"],
                "audio_rms": float(audio[i, 16]),
                "face_detected": bool(vision[i, 18] > 0),
                "text_timing": "estimated_from_audio_energy",
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(selected["sample_id"], len(output), args.out)

if __name__ == "__main__":
    main()
