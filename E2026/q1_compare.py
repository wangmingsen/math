"""Compare original and ASR-refined Q1 artifacts without exporting sample content."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = list(csv.DictReader((a.old / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    audits = list(csv.DictReader((a.new / "alignment_audit.csv").open(encoding="utf-8-sig", newline="")))
    audit_by_id = {r["sample_id"]: r for r in audits}
    problems, changed, moved_words, total_words, face_bins, audio_bins, accepted = [], 0, 0, 0, 0, 0, 0
    shifts = []
    for row in rows:
        sid = row["sample_id"]
        if sid not in audit_by_id or audit_by_id[sid]["status"] != "ok":
            problems.append(sid + ": refinement failed")
            continue
        old = json.loads((a.old / row["mapping_file"]).read_text(encoding="utf-8"))
        new = json.loads((a.new / row["mapping_file"]).read_text(encoding="utf-8"))
        with np.load(a.old / row["feature_file"]) as x, np.load(a.new / row["feature_file"]) as y:
            if not np.array_equal(x["audio"], y["audio"]) or not np.array_equal(x["vision"], y["vision"]):
                problems.append(sid + ": audio/vision changed")
            if y["face_observed"].shape != (50,) or not np.array_equal(y["face_observed"], y["vision"][:, 18] > 0.5):
                problems.append(sid + ": face mask invalid")
            face_bins += int(y["face_observed"].sum())
            if y["audio_signal_present"].shape != (50,) or not np.array_equal(y["audio_signal_present"], y["audio"][:, 16] > 1e-6):
                problems.append(sid + ": audio signal mask invalid")
            audio_bins += int(y["audio_signal_present"].sum())
            if not np.array_equal(x["text"], y["text"]):
                changed += 1
            if not all(np.isfinite(y[k]).all() for k in ("text", "audio", "vision")):
                problems.append(sid + ": nonfinite features")
        if len(old["words"]) != len(new["words"]):
            problems.append(sid + ": transcript token count changed")
        old_bins = {i: b["index"] for b in old["bins"] for i in b["word_indices"]}
        new_bins = {i: b["index"] for b in new["bins"] for i in b["word_indices"]}
        if set(old_bins) != set(new_bins) or set(new_bins) != set(range(len(new["words"]))):
            problems.append(sid + ": token coverage failed")
        total_words += len(new["words"])
        moved_words += sum(old_bins.get(i) != new_bins.get(i) for i in new_bins)
        accepted += bool(new["alignment_quality"]["accepted"])
        for x, y in zip(old["words"], new["words"]):
            if x["token"] != y["token"] or not (0 <= y["start_s"] <= y["end_s"] <= new["duration_s"] + 1e-6):
                problems.append(sid + ": token identity/time invalid")
            shifts.append(abs((x["start_s"] + x["end_s"] - y["start_s"] - y["end_s"]) / 2))
        if any(new["words"][i]["end_s"] > new["words"][i+1]["start_s"] + 1e-6 for i in range(len(new["words"])-1)):
            problems.append(sid + ": overlapping token intervals")
    result = {"samples": len(rows), "asr_accepted": accepted, "asr_fallback": len(rows)-accepted,
              "text_feature_changed_samples": changed, "moved_tokens": moved_words,
              "all_tokens": total_words, "face_observed_bins": face_bins,
              "total_vision_bins": 50*len(rows), "audio_signal_bins": audio_bins,
              "median_absolute_center_shift_s": round(float(np.median(shifts)), 3),
              "p90_absolute_center_shift_s": round(float(np.quantile(shifts, 0.9)), 3),
              "problems": sorted(set(problems))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
