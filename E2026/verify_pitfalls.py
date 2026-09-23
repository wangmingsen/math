from pathlib import Path
import json, pickle, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from audit_data import excel_rows, CompatibleUnpickler

root = Path(__file__).resolve().parents[2]
data_root = root / 'E题数据' / 'E题数据'
a2 = data_root / '附件2-数据集特征文件' / 'aligned_50.pkl'
with a2.open('rb') as f:
    d = pickle.load(f)

def norm_id(value):
    return str(value).strip().replace('$_$', '|')

ids = {split: [norm_id(v) for v in block['id']] for split, block in d.items()}
a1_xlsx = next(data_root.glob('附件1-*/MOSEI*/label-100.xlsx'))
a1_rows = excel_rows(a1_xlsx)
a1_ids = {norm_id(str(r['video_id']) + '$_$' + str(int(float(r['clip_id'])))) for r in a1_rows}
a2_ids = set().union(*(set(values) for values in ids.values()))

def zero_stats(block, key):
    arr = np.asarray(block[key])
    valid = np.asarray(block['text_bert'])[:, 1, :] != 0
    row_zero = np.all(arr == 0, axis=-1)
    interior = valid.copy()
    interior[:, 0] = False
    return {
        't0_zero_fraction': float(row_zero[:, 0].mean()),
        'samples_any_zero_on_bert_valid': int(np.any(row_zero & valid, axis=1).sum()),
        'samples_any_zero_after_t0_on_bert_valid': int(np.any(row_zero & interior, axis=1).sum()),
        'zero_rows_on_bert_valid': int((row_zero & valid).sum()),
        'nonzero_rows_on_bert_padding': int(((~row_zero) & (~valid)).sum()),
    }

train = d['train']
bert_valid = np.asarray(train['text_bert'])[:, 1, :] != 0
result = {
    'aligned_split_sizes': {k: len(v['id']) for k, v in d.items()},
    'split_unique_ids': {k: len(set(v)) for k, v in ids.items()},
    'a1_ids_overlap_a2': len(a1_ids & a2_ids),
    'a1_ids_overlap_by_split': {k: len(a1_ids & set(v)) for k, v in ids.items()},
    'bert_valid_length': {'min': int(bert_valid.sum(1).min()), 'median': float(np.median(bert_valid.sum(1))), 'max': int(bert_valid.sum(1).max())},
    'train_zero_patterns': {key: zero_stats(train, key) for key in ('text', 'audio', 'vision')},
    'class_vs_reg_sign': {},
}
for split, block in d.items():
    counts = {}
    for cls, reg in zip(np.asarray(block['classification_labels']).reshape(-1), np.asarray(block['regression_labels']).reshape(-1)):
        sign = 'negative' if reg < 0 else 'neutral' if reg == 0 else 'positive'
        key = str(int(cls)) + ':' + sign
        counts[key] = counts.get(key, 0) + 1
    result['class_vs_reg_sign'][split] = counts

specialty = {}
for annex in (3, 4):
    top = next(data_root.glob(f'附件{annex}-*'))
    specialty[str(annex)] = {}
    for version in ('对齐版本', '未对齐版本'):
        files = sorted(top.rglob(f'{version}/*.pkl'))
        patterns = {}
        ids_list = []
        nonzero_t0 = {'audio': 0, 'vision': 0}
        for path in files:
            with path.open('rb') as f:
                obj = CompatibleUnpickler(f).load()
            obj = obj.get('test', obj)
            pattern = ','.join(sorted(obj))
            patterns[pattern] = patterns.get(pattern, 0) + 1
            if 'id' in obj:
                ids_list.append(str(obj['id']))
            for key in nonzero_t0:
                arr = np.asarray(obj[key])
                if arr.ndim == 3:
                    arr = arr[0]
                nonzero_t0[key] += int(np.any(arr[0] != 0))
        specialty[str(annex)][version] = {'files': len(files), 'field_patterns': patterns, 'ids_present': len(ids_list), 'unique_ids': len(set(ids_list)), 'nonzero_t0_count': nonzero_t0, 'sample_id': ids_list[0] if ids_list else None}
result['specialty'] = specialty
out = Path(__file__).parent / 'pitfall_checks.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
