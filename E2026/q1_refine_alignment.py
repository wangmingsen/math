"""Refine Q1 token times using ASR word timestamps, retaining provenance.

Requires faster-whisper. The provided transcript remains the text source; ASR
supplies timing anchors only. Low-match clips retain the original estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from q1_extract import TEXT_HASH_DIM, text_features


def norm(word: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", word.lower()))


def align(words: list[dict], heard: list[dict], duration: float,
          min_match: float = 0.6) -> tuple[list[dict], dict]:
    source = [(i, norm(w["token"])) for i, w in enumerate(words) if norm(w["token"])]
    target = [(i, norm(w["word"])) for i, w in enumerate(heard) if norm(w["word"])]
    matcher = SequenceMatcher(None, [x[1] for x in source], [x[1] for x in target], autojunk=False)
    anchors = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            wi = source[block.a + offset][0]
            hi = target[block.b + offset][0]
            anchors[wi] = (float(heard[hi]["start_s"]), float(heard[hi]["end_s"]))
    coverage = len(anchors) / max(1, len(source))
    details = {"lexical_words": len(source), "asr_words": len(target),
               "exact_matched_words": len(anchors), "exact_match_fraction": round(coverage, 4),
               "accepted": coverage >= min_match and len(anchors) >= 3}
    if not details["accepted"]:
        return words, details

    # First use exact ASR word centers. Interpolate only between reliable anchors.
    anchor_ids = sorted(anchors)
    anchor_centers = [(anchors[i][0] + anchors[i][1]) / 2 for i in anchor_ids]
    centers = np.interp(np.arange(len(words)),
                        [-1, *anchor_ids, len(words)],
                        [0.0, *anchor_centers, duration])
    # Correct ASR timestamp overlap or inversions before producing nonoverlapping spans.
    centers = np.maximum.accumulate(np.clip(centers, 0.0, duration))
    boundaries = np.empty(len(words) + 1, dtype=float)
    boundaries[0], boundaries[-1] = 0.0, duration
    boundaries[1:-1] = (centers[:-1] + centers[1:]) / 2
    refined = []
    for i, word in enumerate(words):
        item = dict(word)
        item["start_s"] = float(boundaries[i])
        item["end_s"] = float(boundaries[i + 1])
        item["timing"] = "asr_exact_word_anchor" if i in anchors else "interpolated_between_asr_anchors"
        refined.append(item)
    return refined, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device="cpu", compute_type="int8",
                         download_root=str(args.model_cache))
    source = args.features.resolve()
    rows = list(csv.DictReader((source / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    if args.limit:
        rows = rows[:args.limit]
    reports = []
    for n, row in enumerate(rows, 1):
        try:
            mapping = json.loads((source / row["mapping_file"]).read_text(encoding="utf-8"))
            video = source.parents[1] / "E题数据" / "E题数据" / "附件1-数据" / "MOSEI-100" / mapping["video_relative_path"]
            if not video.is_file():
                # Resolve from the original data tree without assuming folder spelling.
                candidates = list(source.parents[1].glob("E题数据/E题数据/附件1-*/MOSEI*/" + mapping["video_relative_path"].replace("\\", "/")))
                if len(candidates) != 1:
                    raise FileNotFoundError(mapping["video_relative_path"])
                video = candidates[0]
            segments, _ = model.transcribe(str(video), language="en", beam_size=5,
                                           word_timestamps=True, condition_on_previous_text=False)
            heard = [{"word": w.word, "start_s": w.start, "end_s": w.end}
                     for segment in segments for w in (segment.words or [])
                     if w.start is not None and w.end is not None]
            old_words = mapping["words"]
            words, quality = align(old_words, heard, mapping["duration_s"])
            with np.load(source / row["feature_file"]) as old:
                arrays = {key: old[key].copy() for key in old.files}
            if quality["accepted"]:
                arrays["text"], bins = text_features(words, arrays["time_edges_s"])
                arrays["text_observed"] = np.asarray([bool(x) for x in bins], dtype=bool)
                for i, indices in enumerate(bins):
                    mapping["bins"][i]["word_indices"] = indices
                mapping["words"] = words
            # A face detector's failure does not mean a neutral expression.
            arrays["face_observed"] = arrays["vision"][:, 18] > 0.5
            arrays["audio_signal_present"] = arrays["audio"][:, 16] > 1e-6
            mapping["alignment_method"] = "asr_word_anchors_or_energy_fallback"
            mapping["text_timing_is_estimated"] = True
            mapping["alignment_quality"] = quality
            mapping["face_observed_bins"] = int(arrays["face_observed"].sum())
            mapping["audio_signal_bins"] = int(arrays["audio_signal_present"].sum())
            np.savez_compressed(args.out / row["feature_file"], **arrays)
            (args.out / row["mapping_file"]).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
            report = {"sample_id": row["sample_id"], "status": "ok", **quality,
                      "face_observed_bins": mapping["face_observed_bins"],
                      "audio_signal_bins": mapping["audio_signal_bins"]}
        except Exception as exc:
            report = {"sample_id": row["sample_id"], "status": "error", "error": str(exc)}
        reports.append(report)
        print(f"{n}/{len(rows)} {report['sample_id']} {report['status']} {report.get('exact_match_fraction', '')}", flush=True)
    with (args.out / "alignment_audit.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["sample_id", "status", "lexical_words", "asr_words", "exact_matched_words",
                  "exact_match_fraction", "accepted", "face_observed_bins", "audio_signal_bins", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reports)
    if not args.limit:
        (args.out / "manifest.csv").write_bytes((source / "manifest.csv").read_bytes())
    summary = {"processed": len(reports), "successful": sum(x["status"] == "ok" for x in reports),
               "asr_accepted": sum(x.get("accepted", False) for x in reports),
               "model": args.model, "timing": "ASR timestamp estimate, not forced alignment",
               "face_unobserved_samples": sum(x.get("face_observed_bins") == 0 for x in reports),
               "fully_silent_audio_samples": sum(x.get("audio_signal_bins") == 0 for x in reports)}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
