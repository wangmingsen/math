"""Q1 provisional three-modality extraction on the 100 supplied videos.

Text timestamps are energy-weighted estimates, not forced-alignment truth.
Usage:
  python E2026/q1_extract.py --data-root ../E题数据/E题数据 --out ../outputs/q1_features --limit 1
  python E2026/q1_extract.py --data-root ../E题数据/E题数据 --out ../outputs/q1_features
Only use the contest-provided videos and transcript table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from audit_data import excel_rows

N_BINS = 50
SAMPLE_RATE = 16000
TEXT_HASH_DIM = 256
AUDIO_BANDS = 16
TEXT_DIM = TEXT_HASH_DIM + 3
AUDIO_DIM = AUDIO_BANDS + 5
VISION_DIM = 8 + 4 + 4 + 5
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", re.UNICODE)
FACE_MODEL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if FACE_MODEL.empty():
    raise RuntimeError("OpenCV face cascade unavailable")


def audio_wave(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[:300])
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def speech_weights(wave: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, round(SAMPLE_RATE * 0.02))
    n = max(1, (len(wave) + hop - 1) // hop)
    energy = np.empty(n, dtype=np.float32)
    for i in range(n):
        part = wave[i * hop:(i + 1) * hop]
        energy[i] = float(np.sqrt(np.mean(part * part))) if len(part) else 0.0
    threshold = max(float(np.quantile(energy, 0.3) * 1.5),
                    float(np.quantile(energy, 0.9) * 0.1))
    weights = np.maximum(energy - threshold, 0)
    if not np.any(weights):
        weights[:] = 1
    times = np.minimum((np.arange(n) + 0.5) * hop / SAMPLE_RATE, duration)
    return times, weights


def word_intervals(text: str, wave: np.ndarray, duration: float) -> list[dict]:
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return []
    times, weights = speech_weights(wave, duration)
    cdf = np.cumsum(weights, dtype=np.float64)
    cdf /= cdf[-1]
    quantiles = np.linspace(0, 1, len(tokens) + 1)
    boundaries = np.interp(quantiles, np.r_[0.0, cdf], np.r_[0.0, times])
    boundaries[0], boundaries[-1] = 0.0, duration
    return [{"index": i, "token": token, "start_s": float(boundaries[i]),
             "end_s": float(boundaries[i + 1]), "timing": "energy_weighted_estimate"}
            for i, token in enumerate(tokens)]


def text_features(words: list[dict], edges: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    feature = np.zeros((N_BINS, TEXT_DIM), dtype=np.float32)
    assigned = [[] for _ in range(N_BINS)]
    for word in words:
        center = (word["start_s"] + word["end_s"]) / 2
        bin_id = int(np.clip(np.searchsorted(edges, center, side="right") - 1, 0, N_BINS - 1))
        assigned[bin_id].append(word["index"])
        token = word["token"].lower()
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        feature[bin_id, int.from_bytes(digest, "little") % TEXT_HASH_DIM] += 1
        feature[bin_id, TEXT_HASH_DIM] += 1
        feature[bin_id, TEXT_HASH_DIM + 1] += len(token)
        feature[bin_id, TEXT_HASH_DIM + 2] += float(not token.isalnum())
    for i, indices in enumerate(assigned):
        if indices:
            feature[i, :TEXT_HASH_DIM] /= len(indices)
            feature[i, TEXT_HASH_DIM + 1] /= len(indices)
            feature[i, TEXT_HASH_DIM + 2] /= len(indices)
    return feature, assigned


def audio_features(wave: np.ndarray, edges: np.ndarray) -> np.ndarray:
    result = np.zeros((N_BINS, AUDIO_DIM), dtype=np.float32)
    band_edges = np.geomspace(80, 8000, AUDIO_BANDS + 1)
    for i in range(N_BINS):
        lo = int(edges[i] * SAMPLE_RATE)
        hi = min(len(wave), max(lo + 1, int(edges[i + 1] * SAMPLE_RATE)))
        segment = wave[lo:hi]
        if len(segment) < 32:
            continue
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
        freq = np.fft.rfftfreq(len(segment), 1 / SAMPLE_RATE)
        total = spectrum.sum() + 1e-12
        for j in range(AUDIO_BANDS):
            take = (freq >= band_edges[j]) & (freq < band_edges[j + 1])
            result[i, j] = np.log1p(float(spectrum[take].sum() / max(1, take.sum())))
        centroid = float((freq * spectrum).sum() / total)
        result[i, AUDIO_BANDS:] = (
            float(np.sqrt(np.mean(segment ** 2))),
            float(np.mean(np.signbit(segment[1:]) != np.signbit(segment[:-1]))),
            centroid / 8000.0,
            float(np.sqrt((((freq - centroid) ** 2) * spectrum).sum() / total)) / 8000.0,
            float(np.exp(np.mean(np.log(spectrum + 1e-12))) / (np.mean(spectrum) + 1e-12)),
        )
    return result


def frame_feature(frame: np.ndarray, previous_gray: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = []
    for channel, bins, maximum in ((0, 8, 180), (1, 4, 256), (2, 4, 256)):
        values = cv2.calcHist([hsv], [channel], None, [bins], [0, maximum]).reshape(-1)
        hist.extend((values / max(1, values.sum())).tolist())
    faces = FACE_MODEL.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    area = max((w * h for _, _, w, h in faces), default=0) / (128 * 128)
    motion = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255) if previous_gray is not None else 0.0
    extra = [float(np.mean(gray) / 255), float(np.std(gray) / 255),
             float(len(faces) > 0), float(area), motion]
    return np.asarray(hist + extra, dtype=np.float32), gray


def video_features(path: Path, duration: float) -> tuple[np.ndarray, np.ndarray, list[float | None]]:
    targets = (np.arange(N_BINS) + 0.5) * duration / N_BINS
    result = np.zeros((N_BINS, VISION_DIM), dtype=np.float32)
    valid = np.zeros(N_BINS, dtype=bool)
    frame_times: list[float | None] = [None] * N_BINS
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    previous_gray = None
    frame_number = 0
    target_index = 0
    last_frame = None
    last_time = None
    while target_index < N_BINS:
        okay, frame = cap.read()
        if not okay:
            break
        last_frame = frame
        t = frame_number / fps
        last_time = t
        while target_index < N_BINS and t >= targets[target_index]:
            result[target_index], previous_gray = frame_feature(frame, previous_gray)
            valid[target_index] = True
            frame_times[target_index] = float(t)
            target_index += 1
        frame_number += 1
    if last_frame is not None:
        while target_index < N_BINS:
            result[target_index], previous_gray = frame_feature(last_frame, previous_gray)
            valid[target_index] = True
            frame_times[target_index] = float(last_time)
            target_index += 1
    cap.release()
    return result, valid, frame_times


def process(row: dict, base: Path, output: Path) -> dict:
    video_id = str(row["video_id"])
    clip_id = str(int(float(row["clip_id"])))
    path = base / video_id / (clip_id + ".mp4")
    if not path.is_file():
        matches = list((base / video_id).glob(clip_id + "*.mp4"))
        if len(matches) != 1:
            raise FileNotFoundError(path)
        path = matches[0]
    sample_id = video_id + "$_$" + clip_id
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata: {path}")
    duration = frames / fps
    wave = audio_wave(path)
    audio_duration = len(wave) / SAMPLE_RATE
    duration = min(duration, audio_duration)
    edges = np.linspace(0, duration, N_BINS + 1)
    words = word_intervals(str(row.get("text", "")), wave, duration)
    text_array, bins = text_features(words, edges)
    audio_array = audio_features(wave, edges)
    vision_array, vision_valid, frame_times = video_features(path, duration)
    if not all(np.isfinite(x).all() for x in (text_array, audio_array, vision_array)):
        raise RuntimeError(f"non-finite feature: {sample_id}")
    output.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(sample_id.encode()).hexdigest()[:20]
    feature_file = name + ".npz"
    mapping_file = name + ".json"
    np.savez_compressed(output / feature_file,
                        text=text_array, audio=audio_array, vision=vision_array,
                        text_observed=np.asarray([bool(x) for x in bins], dtype=bool),
                        audio_observed=np.ones(N_BINS, dtype=bool), vision_observed=vision_valid,
                        time_edges_s=edges.astype(np.float32))
    mapping = {
        "sample_id": sample_id, "video_relative_path": str(path.relative_to(base)),
        "alignment_method": "fixed_50_time_bins_with_energy_weighted_text_estimate",
        "text_timing_is_estimated": True,
        "audio_sample_rate": SAMPLE_RATE, "video_fps": fps, "duration_s": duration,
        "words": words,
        "bins": [{"index": i, "start_s": float(edges[i]), "end_s": float(edges[i + 1]),
                  "word_indices": bins[i], "frame_time_s": frame_times[i]}
                 for i in range(N_BINS)],
        "feature_shapes": {"text": list(text_array.shape), "audio": list(audio_array.shape),
                           "vision": list(vision_array.shape)},
    }
    (output / mapping_file).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"sample_id": sample_id, "feature_file": feature_file, "mapping_file": mapping_file,
            "duration_s": round(duration, 6), "audio_duration_s": round(audio_duration, 6),
            "word_count": len(words), "text_bins": sum(bool(x) for x in bins),
            "vision_bins": int(vision_valid.sum()), "text_dim": TEXT_DIM,
            "audio_dim": AUDIO_DIM, "vision_dim": VISION_DIM, "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    label_file = next(args.data_root.resolve().glob("附件1-*/MOSEI*/label-100.xlsx"))
    base = label_file.parent
    labels = excel_rows(label_file)
    if args.limit:
        labels = labels[:args.limit]
    results = []
    for index, row in enumerate(labels, start=1):
        try:
            result = process(row, base, args.out)
        except Exception as exc:
            result = {"sample_id": str(row.get("video_id")) + "$_$" + str(row.get("clip_id")),
                      "status": "error", "error": str(exc)}
        results.append(result)
        print(f"{index}/{len(labels)} {result['sample_id']} {result['status']}", flush=True)
    fields = ["sample_id", "feature_file", "mapping_file", "duration_s", "audio_duration_s",
              "word_count", "text_bins", "vision_bins", "text_dim", "audio_dim",
              "vision_dim", "status", "error"]
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = {"requested": len(labels), "completed": sum(x["status"] == "ok" for x in results),
               "failed": [x for x in results if x["status"] != "ok"],
               "alignment_method": "energy_weighted_estimate_not_forced_alignment",
               "feature_dims": {"text": TEXT_DIM, "audio": AUDIO_DIM, "vision": VISION_DIM}}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
