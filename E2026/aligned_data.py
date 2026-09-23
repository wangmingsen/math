"""Aligned E-problem data contract and read-only preprocessing summary.

Run: python E2026/aligned_data.py --data-root ../E题数据/E题数据 --out E2026/aligned_preprocess_summary.json
Only load contest-provided pickle files that you trust.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from audit_data import CompatibleUnpickler

EXPECTED = {'text_bert': (3, 50), 'audio': (50, 74), 'vision': (50, 35)}

def load_small(path: Path):
    with path.open('rb') as stream:
        return CompatibleUnpickler(stream).load()

def normalize(sample: dict, sample_id: str, source: str) -> dict:
    value = {'sample_id': sample_id, 'source': source}
    for key, shape in EXPECTED.items():
        if key not in sample:
            raise ValueError(f'{source}/{sample_id}: missing {key}')
        arr = np.asarray(sample[key])
        if arr.ndim == len(shape) + 1 and arr.shape[0] == 1:
            arr = arr[0]
        if tuple(arr.shape) != shape:
            raise ValueError(f'{source}/{sample_id}: {key} shape {arr.shape}, expected {shape}')
        if not np.isfinite(arr).all():
            raise ValueError(f'{source}/{sample_id}: {key} contains NaN/Inf')
        value[key] = arr
    return value

def zero_categories(array: np.ndarray) -> dict:
    zero = np.all(array == 0, axis=1)
    observed = ~zero
    before = np.cumsum(observed) > 0
    after = np.cumsum(observed[::-1])[::-1] > 0
    interior = zero & before & after
    right_edge = zero & before & ~after
    leading = zero & ~before
    return {'zero': zero, 'interior_unknown': interior, 'right_edge_zero': right_edge, 'leading_zero': leading}

def summarize(sample: dict) -> dict:
    t = sample['text_bert']
    text_valid = t[1] != 0
    result = {'sample_id': sample['sample_id'], 'source': sample['source'], 'text_valid_count': int(text_valid.sum())}
    for key in ('audio', 'vision'):
        cat = zero_categories(sample[key])
        result[key + '_nonzero_count'] = int((~cat['zero']).sum())
        result[key + '_interior_unknown_count'] = int(cat['interior_unknown'].sum())
        result[key + '_right_edge_zero_count'] = int(cat['right_edge_zero'].sum())
        result[key + '_all_zero'] = bool(cat['zero'].all())
    return result

def inject_local_gap(sample: dict, modality: str, fraction: float, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Inject a known contiguous gap; return sample, planned mask, actually changed mask."""
    if modality not in ('audio', 'vision'):
        raise ValueError('this injector supports audio or vision only')
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    original = sample[modality]
    observed = np.flatnonzero(np.any(original != 0, axis=1))
    if len(observed) == 0:
        raise ValueError('cannot inject a gap into an all-zero modality')
    left, right = int(observed[0]), int(observed[-1]) + 1
    span = min(right - left, max(1, round((right - left) * fraction)))
    start = int(np.random.default_rng(seed).integers(left, right - span + 1))
    planned = np.zeros(50, dtype=bool)
    planned[start:start + span] = True
    changed = planned & np.any(original != 0, axis=1)
    altered = dict(sample)
    altered[modality] = original.copy()
    altered[modality][planned] = 0
    return altered, planned, changed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    aligned = next(root.glob('附件2-*/aligned_50.pkl'))
    with aligned.open('rb') as stream:
        dataset = pickle.load(stream)
    records = []
    split_sizes = {}
    for split in ('train', 'valid', 'test'):
        block = dataset[split]
        split_sizes[split] = len(block['id'])
        for i, sample_id in enumerate(block['id']):
            sample = normalize({k: block[k][i] for k in EXPECTED}, str(sample_id), 'A2/' + split)
            records.append(summarize(sample))
    special_counts = {}
    for annex in (3, 4):
        top = next(root.glob(f'附件{annex}-*'))
        paths = sorted(top.rglob('对齐版本/*.pkl'))
        special_counts[str(annex)] = len(paths)
        for path in paths:
            raw = load_small(path)
            if annex == 3:
                if set(raw) != {'test'}:
                    raise ValueError(f'{path}: unexpected wrapper')
                raw = raw['test']
                sample_id = path.stem
            else:
                sample_id = str(raw['id'])
                if sample_id != path.stem:
                    raise ValueError(f'{path}: id {sample_id} differs from filename')
            records.append(summarize(normalize(raw, sample_id, f'A{annex}')))
    by_source = {}
    for source in sorted({r['source'] for r in records}):
        rows = [r for r in records if r['source'] == source]
        by_source[source] = {
            'count': len(rows),
            'audio_all_zero': sum(r['audio_all_zero'] for r in rows),
            'vision_all_zero': sum(r['vision_all_zero'] for r in rows),
            'audio_interior_unknown_samples': sum(r['audio_interior_unknown_count'] > 0 for r in rows),
            'vision_interior_unknown_samples': sum(r['vision_interior_unknown_count'] > 0 for r in rows),
            'text_valid_min': min(r['text_valid_count'] for r in rows),
            'text_valid_max': max(r['text_valid_count'] for r in rows),
        }
    anomalies = [r for r in records if r['audio_all_zero'] or r['vision_all_zero']]
    report = {'version': 'aligned_50', 'split_sizes': split_sizes, 'special_counts': special_counts, 'by_source': by_source, 'all_zero_anomalies': anomalies,
              'mask_semantics': {'text_valid': 'BERT attention mask', 'interior_unknown': 'all-zero between nonzero observations, not proven injected missingness', 'right_edge_zero': 'all-zero after last observed row, not proven padding', 'injected': 'known only when generated by inject_local_gap'}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({"version": report["version"], "split_sizes": split_sizes, "special_counts": special_counts, "by_source": by_source, "anomaly_count": len(anomalies)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
