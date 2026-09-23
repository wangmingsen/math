"""Small reproducible aligned_50 baseline using only fields shared with A3.

Run: python E2026/baseline.py --data ../E题数据/E题数据/附件2-数据集特征文件/aligned_50.pkl --out E2026/baseline_metrics.json
This baseline hashes text_bert token IDs, pools nonzero audio/vision rows,
then fits a classifier and regressor. It is a diagnostic baseline, not the
final local-missingness model.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def text_features(bert: np.ndarray, bins: int = 256) -> np.ndarray:
    result = np.zeros((len(bert), bins), dtype=np.float32)
    for i, sample in enumerate(bert):
        ids = sample[0].astype(np.int64)
        mask = sample[1] != 0
        ids = ids[mask & (ids != 0) & (ids != 101) & (ids != 102)]
        if len(ids):
            result[i] = np.bincount(ids % bins, minlength=bins) / len(ids)
    return result


def pool(sequence: np.ndarray) -> np.ndarray:
    result = np.zeros((len(sequence), sequence.shape[-1] * 2), dtype=np.float32)
    for i, sample in enumerate(sequence):
        active = np.any(sample != 0, axis=1)
        if np.any(active):
            values = sample[active]
            result[i] = np.concatenate([values.mean(axis=0), values.std(axis=0)])
    return result


def features(split: dict) -> np.ndarray:
    return np.concatenate([
        text_features(np.asarray(split["text_bert"])),
        pool(np.asarray(split["audio"])),
        pool(np.asarray(split["vision"])),
    ], axis=1)


def mask_contiguous(split: dict, modality: str, fraction: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    copy = dict(split)
    original = np.asarray(split[modality])
    array = original.copy()
    for i, item in enumerate(original):
        active = np.flatnonzero(np.any(item != 0, axis=1))
        if len(active) == 0:
            continue
        start_valid, end_valid = int(active[0]), int(active[-1]) + 1
        length = max(1, int(round((end_valid - start_valid) * fraction)))
        start = int(rng.integers(start_valid, end_valid - length + 1))
        array[i, start:start + length] = 0
    copy[modality] = array
    return copy


def scores(classifier, regressor, matrix: np.ndarray, split: dict) -> dict:
    y_class = np.asarray(split["classification_labels"]).reshape(-1).astype(int)
    y_reg = np.asarray(split["regression_labels"]).reshape(-1)
    pred_class = classifier.predict(matrix)
    pred_reg = regressor.predict(matrix)
    corr = float(np.corrcoef(y_reg, pred_reg)[0, 1]) if np.std(pred_reg) > 0 else None
    return {
        "accuracy": float(accuracy_score(y_class, pred_class)),
        "f1_macro": float(f1_score(y_class, pred_class, average="macro")),
        "mae": float(mean_absolute_error(y_reg, pred_reg)),
        "pearson": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.data.open("rb") as handle:
        data = pickle.load(handle)
    train, valid = data["train"], data["valid"]
    x_train, x_valid = features(train), features(valid)
    y_class = np.asarray(train["classification_labels"]).reshape(-1).astype(int)
    y_reg = np.asarray(train["regression_labels"]).reshape(-1)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    classifier.fit(x_train, y_class)
    regressor.fit(x_train, y_reg)
    result = {
        "data_version": "aligned_50",
        "feature_definition": "256-bin hashed BERT token IDs + nonzero-row audio/vision mean and std",
        "train_count": len(x_train),
        "valid_count": len(x_valid),
        "feature_count": x_train.shape[1],
        "clean_valid": scores(classifier, regressor, x_valid, valid),
        "masked_valid": {},
    }
    for modality in ("audio", "vision"):
        altered = mask_contiguous(valid, modality, fraction=0.25, seed=2026)
        result["masked_valid"][modality] = scores(classifier, regressor, features(altered), valid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
