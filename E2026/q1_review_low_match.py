"""Audit Q1 low-ASR-match clips and decoded silence; keeps transcripts local.

Run with the original contest data only. The output contains raw transcripts
and should not be committed to a public repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from audit_data import excel_rows
from q1_extract import audio_wave
from q1_refine_alignment import align, norm


def lexical(text: str) -> list[str]:
    return [norm(x) for x in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", text) if norm(x)]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, lexical(a), lexical(b), autojunk=False).ratio()


def sample_id(row: dict) -> str:
    return str(row["video_id"]) + "$_$" + str(int(float(row["clip_id"])))


def stereo_peaks(path: Path) -> list[float]:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path),
                                   "-vn", "-ac", "2", "-ar", "16000", "-f", "f32le", "pipe:1"],
                                  timeout=90)
    audio = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    return np.max(np.abs(audio), axis=0).astype(float).tolist() if len(audio) else [0.0, 0.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="base.en")
    args = parser.parse_args()
    label_file = next(args.data_root.resolve().glob("附件1-*/MOSEI*/label-100.xlsx"))
    labels = excel_rows(label_file)
    by_id = {sample_id(row): row for row in labels}
    audits = list(csv.DictReader((args.features / "alignment_audit.csv").open(encoding="utf-8-sig", newline="")))
    low = [row for row in audits if row["accepted"].lower() == "false"]
    model = WhisperModel(args.model, device="cpu", compute_type="int8",
                         download_root=str(args.model_cache))
    results = []
    for n, row in enumerate(low, 1):
        sid = row["sample_id"]
        label = by_id[sid]
        video = label_file.parent / str(label["video_id"]) / (str(int(float(label["clip_id"]))) + ".mp4")
        wave = audio_wave(video)
        rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
        original = str(label["text"])
        if peak == 0:
            heard = []
            channels = stereo_peaks(video)
            category = "decoded_stereo_pcm_all_zero" if channels == [0.0, 0.0] else "mono_cancellation"
            base_quality = {"exact_match_fraction": 0.0, "accepted": False,
                            "exact_matched_words": 0, "lexical_words": len(lexical(original))}
        else:
            channels = None
            segments, _ = model.transcribe(str(video), language="en", beam_size=5,
                                           word_timestamps=True, condition_on_previous_text=False)
            heard = [{"word": w.word, "start_s": w.start, "end_s": w.end}
                     for segment in segments for w in (segment.words or [])
                     if w.start is not None and w.end is not None]
            import hashlib
            path = args.features / (hashlib.sha256(sid.encode()).hexdigest()[:20] + ".json")
            old_mapping = json.loads(path.read_text(encoding="utf-8"))
            _, base_quality = align(old_mapping["words"], heard, old_mapping["duration_s"])
            category = "base_alignment_candidate" if base_quality["accepted"] else "unresolved_low_match"
            if not base_quality["accepted"] and len(heard) < 3:
                category = "nonzero_audio_asr_too_few_words"
        hypothesis = " ".join(x["word"].strip() for x in heard)
        peers = [(sample_id(other), similarity(hypothesis, str(other["text"])))
                 for other in labels if str(other["video_id"]) == str(label["video_id"])]
        best_id, best_similarity = max(peers, key=lambda item: item[1])
        result = {
            "sample_id": sid, "category": category, "duration_s": round(len(wave) / 16000, 3),
            "pcm_rms": round(rms, 7), "pcm_peak": round(peak, 7),
            "stereo_peaks_if_mono_zero": channels,
            "tiny_exact_match_fraction": float(row["exact_match_fraction"]),
            "base_exact_match_fraction": base_quality["exact_match_fraction"],
            "base_exact_matched_words": base_quality["exact_matched_words"],
            "base_accepted": base_quality["accepted"],
            "same_video_best_label_id": best_id,
            "same_video_best_similarity": round(best_similarity, 4),
            "given_transcript": original, "base_asr_transcript": hypothesis,
            "video_relative_path": str(video.relative_to(label_file.parent)),
            "base_words": heard,
        }
        results.append(result)
        print(f"{n}/{len(low)} {sid} {category} tiny={row['exact_match_fraction']} base={base_quality['exact_match_fraction']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    queue_path = args.out.parent / "q1_low_match_manual_review.csv"
    with queue_path.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["sample_id", "category", "video_relative_path", "pcm_rms", "pcm_peak",
                  "tiny_exact_match_fraction", "base_exact_match_fraction", "given_transcript",
                  "base_asr_transcript", "manual_review_status", "manual_notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key, "") for key in fields} |
                            {"manual_review_status": "pending", "manual_notes": ""})
    summary = {"reviewed": len(results), "stereo_pcm_all_zero": sum(x["category"] == "decoded_stereo_pcm_all_zero" for x in results),
               "base_alignment_candidates": sum(x["category"] == "base_alignment_candidate" for x in results),
               "unresolved_low_match": sum(x["category"] == "unresolved_low_match" for x in results),
               "nonzero_audio_asr_too_few_words": sum(x["category"] == "nonzero_audio_asr_too_few_words" for x in results),
               "other_label_similarity_ge_0_8": sum(x["same_video_best_label_id"] != x["sample_id"] and x["same_video_best_similarity"] >= 0.8 for x in results),
               "model": args.model, "warning": "ASR hypotheses are not manual transcript ground truth"}
    (args.out.parent / "q1_low_match_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
