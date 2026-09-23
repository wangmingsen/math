"""Validate the A4 explanation tables without using labels or original media."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pred = rows(args.out / "predictions.csv")
    mod = rows(args.out / "modality_impacts.csv")
    local = rows(args.out / "local_evidence.csv")
    ids = [r["sample_id"] for r in pred]
    assert len(ids) == 20 and len(set(ids)) == 20
    assert len(mod) == 60 and len(local) == 300
    assert Counter((r["sample_id"], r["modality"]) for r in mod) == Counter(
        (sid, modality) for sid in ids for modality in ("text", "audio", "vision")
    )
    assert set(r["sample_id"] for r in local) == set(ids)
    for r in pred:
        probs = [float(r[f"prob_{c}"]) for c in ("negative", "neutral", "positive")]
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in probs)
        assert abs(sum(probs) - 1) < 1e-6
        assert int(r["predicted_polarity"]) == max(range(3), key=lambda i: probs[i])
        assert math.isfinite(float(r["predicted_intensity"]))
        shares = [float(r[f"{m}_support_share"]) for m in ("text", "audio", "vision")]
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in shares)
        assert abs(sum(shares) - 1) < 1e-6
    p = {r["sample_id"]: r for r in pred}
    assert p["13"]["vision_observed"] == "False"
    assert all(r["class_confidence_drop"] == "0.0" for r in mod
               if r["sample_id"] == "13" and r["modality"] == "vision")
    assert p["15"]["time_mapping_quality"] == "unverified"
    assert all(not r["estimated_start_s"] and not r["estimated_end_s"] for r in local
               if r["sample_id"] == "15")
    assert {r["sample_id"] for r in pred if r["text_truncated_at_50"] == "True"} == {"07", "18"}
    print(f"PASS: {len(pred)} predictions, {len(mod)} modality impacts, {len(local)} local rows")


if __name__ == "__main__":
    main()
