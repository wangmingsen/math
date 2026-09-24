"""Audit the frozen Q2 model under known local gaps on A2 valid only.

A3 has no labels: its zero patterns are described, never scored. The A2 test
split is not read for model selection or for this audit.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np

from aligned_data import load_small, normalize
from q2_late_fusion import metric
from q2_missing_baseline import gap, text_gap
from q3_explain import predict


def gap_with_log(sequence, fraction, where=None, rng=None):
    """Mask a contiguous observed span; count planned and actually changed rows."""
    out = sequence.copy()
    active = np.any(sequence != 0, axis=2)
    planned = changed = eligible = 0
    for i in range(len(sequence)):
        index = np.flatnonzero(active[i])
        eligible += len(index)
        if not len(index):
            continue
        left, right = int(index[0]), int(index[-1]) + 1
        width = min(right-left, max(1, round((right-left)*fraction)))
        position = int(rng.integers(left, right-width+1)) if rng is not None else left+round((right-left-width)*where)
        planned += width
        changed += int(active[i, position:position+width].sum())
        out[i, position:position+width] = 0
    return out, {'eligible_nonzero_rows': eligible, 'planned_rows': planned,
                 'changed_nonzero_rows': changed,
                 'actual_changed_fraction': changed/eligible if eligible else None}


def zero_pattern(sequence):
    active = np.any(sequence != 0, axis=1)
    where = np.flatnonzero(active)
    if not len(where):
        return {'all_zero': True, 'interior_zero': 0, 'longest_interior_run': 0,
                'observed_rows': 0}
    between = ~active[where[0]:where[-1]+1]
    run = longest = 0
    for missing in between:
        run = run+1 if missing else 0
        longest = max(longest, run)
    return {'all_zero': False, 'interior_zero': int(between.sum()),
            'longest_interior_run': longest, 'observed_rows': int(active.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    if bundle.get('version') != 'q2_late_fusion_v1':
        raise ValueError('Unexpected Q2 model version')
    source = next(args.data_root.resolve().glob('附件2-*/aligned_50.pkl'))
    with source.open('rb') as stream:
        block = pickle.load(stream)['valid']
    ids = np.asarray(block['id'], dtype=str)
    with np.load(args.embeddings/'valid.npz') as z:
        if not np.array_equal(ids, z['sample_id'].astype(str)):
            raise ValueError('A2 valid embedding ID/order mismatch')
        segments = z['segments'].astype('f4')
        mask = z['segment_present'].astype(bool)
    text = bundle['text_pca'].transform(segments.reshape(-1, 768)).reshape(len(ids), 5, 48).astype('f4')
    text[~mask] = 0
    audio = np.asarray(block['audio'], 'f4')
    vision = np.asarray(block['vision'], 'f4')
    cls = np.asarray(block['classification_labels']).reshape(-1).astype(int)
    reg = np.asarray(block['regression_labels']).reshape(-1).astype('f4')

    def evaluate(t, m, a, v):
        p, r = predict(bundle, t, m, a, v)
        return metric(cls, reg, p, r)

    scenarios = {'clean': {'metrics': evaluate(text, mask, audio, vision)}}
    for fraction in (.10, .25, .50):
        for name, position in (('start', 0.), ('middle', .5), ('end', 1.)):
            a, alog = gap_with_log(audio, fraction, position)
            v, vlog = gap_with_log(vision, fraction, position)
            for modality, aa, vv, logs in (
                ('audio', a, vision, {'audio': alog}),
                ('vision', audio, v, {'vision': vlog}),
                ('audio_vision', a, v, {'audio': alog, 'vision': vlog}),
            ):
                scenarios[f'{modality}_{int(100*fraction)}pct_{name}'] = {
                    'metrics': evaluate(text, mask, aa, vv), 'injection': logs}
    for name, position in (('start', 0.), ('middle', .5), ('end', 1.)):
        t, m, changed = text_gap(text, mask, where=position)
        scenarios[f'text_1segment_{name}'] = {
            'metrics': evaluate(t, m, audio, vision),
            'injection': {'changed_text_segments': changed}}

    # Repeat the combined-gap experiment with independent positions. Neither
    # labels nor metrics from A3 are used to choose these seeds or fractions.
    random_trials = []
    for seed in (11, 23, 47, 83, 131):
        arng = np.random.default_rng(seed)
        vrng = np.random.default_rng(seed+10000)
        a, alog = gap_with_log(audio, .25, rng=arng)
        v, vlog = gap_with_log(vision, .25, rng=vrng)
        random_trials.append({'seed': seed, 'metrics': evaluate(text, mask, a, v),
                              'injection': {'audio': alog, 'vision': vlog}})

    a3_paths = sorted(next(args.data_root.resolve().glob('附件3-*')).rglob('对齐版本/*.pkl'))
    if len(a3_paths) != 30:
        raise ValueError('Expected 30 A3 aligned samples')
    patterns = {'audio': [], 'vision': []}
    for path in a3_paths:
        raw = load_small(path)
        if set(raw) != {'test'}:
            raise ValueError(f'Unexpected A3 wrapper: {path}')
        sample = normalize(raw['test'], path.stem, 'A3')
        for modality in patterns:
            patterns[modality].append(zero_pattern(sample[modality]))
    a3_summary = {}
    for modality, records in patterns.items():
        a3_summary[modality] = {
            'samples': len(records),
            'all_zero_samples': sum(r['all_zero'] for r in records),
            'interior_zero_samples': sum(r['interior_zero'] > 0 for r in records),
            'longest_interior_run_max': max(r['longest_interior_run'] for r in records),
            'observed_rows_min': min(r['observed_rows'] for r in records),
            'observed_rows_median': float(np.median([r['observed_rows'] for r in records])),
            'observed_rows_max': max(r['observed_rows'] for r in records),
        }
    report = {
        'protocol': 'Frozen Q2 candidate; metrics on A2 valid only. A2 test untouched in this audit; A3 summarized without labels.',
        'valid_samples': len(ids), 'A3_samples': len(a3_paths),
        'mask_semantics': 'Injection records planned positions and originally nonzero positions actually changed. A3 interior zeros have unknown cause.',
        'scenarios': scenarios, 'random_audio_vision_25pct': random_trials,
        'A3_observed_zero_patterns': a3_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'clean': scenarios['clean']['metrics'],
                      'dual_25pct_middle': scenarios['audio_vision_25pct_middle'],
                      'A3_zero_patterns': a3_summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
