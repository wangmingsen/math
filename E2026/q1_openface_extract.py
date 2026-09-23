"""Extract 35 OpenFace action units on the Q1 fixed time grid."""
from __future__ import annotations
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
import numpy as np


def aggregate(csv_path: Path, edges: np.ndarray):
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        columns = [x.strip() for x in reader.fieldnames or []]
        aus = sorted(x for x in columns if re.fullmatch(r'AU\d+_[rc]', x))
        if len(aus) != 35:
            raise ValueError(f'Expected 35 AU columns, got {len(aus)}')
        values = np.zeros((50, 35), np.float32)
        count = np.zeros(50, np.int32)
        confidence = np.zeros(50, np.float32)
        frames = 0
        failed = 0
        nonfinite = 0
        for row in reader:
            frames += 1
            row = {k.strip(): v for k, v in row.items()}
            if int(float(row['success'])) != 1:
                failed += 1
                continue
            t = float(row['timestamp'])
            b = int(np.clip(np.searchsorted(edges, t, side='right') - 1, 0, 49))
            au = np.asarray([float(row[x]) for x in aus], np.float32)
            if not np.isfinite(au).all():
                nonfinite += int((~np.isfinite(au)).sum())
                au = np.nan_to_num(au)
            values[b] += au
            confidence[b] += float(row['confidence'])
            count[b] += 1
    occupied = count > 0
    values[occupied] /= count[occupied, None]
    confidence[occupied] /= count[occupied]
    return values, count, confidence, aus, frames, failed, nonfinite


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--q1-features', type=Path, required=True)
    p.add_argument('--openface', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    base = next(a.data_root.resolve().glob('附件1-*/MOSEI*'))
    source = a.q1_features.resolve()
    exe = a.openface.resolve()
    rows = list(csv.DictReader((source / 'manifest.csv').open(encoding='utf-8-sig', newline='')))
    if a.limit: rows = rows[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    raw = (a.out / 'raw_openface').resolve()
    raw.mkdir(exist_ok=True)
    report = []
    for n, row in enumerate(rows, 1):
        try:
            mapping = json.loads((source / row['mapping_file']).read_text(encoding='utf-8'))
            with np.load(source / row['feature_file']) as old:
                edges = old['time_edges_s'].astype(np.float64)
            video = base / mapping['video_relative_path']
            stem = Path(row['feature_file']).stem
            output_csv = raw / (stem + '.csv')
            if not output_csv.exists():
                command = [str(exe), '-f', str(video), '-out_dir', str(raw), '-of', stem,
                           '-mloc', 'model/main_clnf_general.txt', '-aus', '-pose', '-q']
                process = subprocess.run(command, cwd=exe.parent, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                         errors='replace', timeout=180)
                if process.returncode or not output_csv.exists():
                    raise RuntimeError(f'OpenFace exit={process.returncode}: {process.stderr[-300:]}')
            feature, count, confidence, aus, frames, failed, nonfinite = aggregate(output_csv, edges)
            if feature.shape != (50, 35) or not np.isfinite(feature).all():
                raise ValueError('Invalid OpenFace features')
            np.savez_compressed(a.out / row['feature_file'], vision_openface=feature,
                                face_valid=count > 0, face_frame_count=count,
                                face_confidence=confidence, time_edges_s=edges.astype(np.float32))
            item = {'sample_id': row['sample_id'], 'feature_file': row['feature_file'],
                    'status': 'ok', 'face_bins': int((count > 0).sum()),
                    'face_frames': int(count.sum()), 'video_frames': frames,
                    'failed_frames': failed, 'nonfinite_raw_cells': nonfinite}
        except Exception as exc:
            item = {'sample_id': row['sample_id'], 'feature_file': row['feature_file'],
                    'status': 'error', 'error': str(exc)}
        report.append(item)
        print(f"{n}/{len(rows)} {item['sample_id']} {item['status']}", flush=True)
    fields = ['sample_id', 'feature_file', 'status', 'face_bins', 'face_frames',
              'video_frames', 'failed_frames', 'nonfinite_raw_cells', 'error']
    with (a.out / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(report)
    summary = {'requested': len(rows), 'completed': sum(x['status']=='ok' for x in report),
               'failed': sum(x['status']!='ok' for x in report),
               'extractor': 'OpenFace 2.2.0 FeatureExtraction, main_clnf_general',
               'feature': '17 AU intensities + 18 AU occurrences',
               'output_shape': [50,35],
               'no_face_samples': sum(x.get('face_frames') == 0 for x in report),
               'total_face_frames': sum(x.get('face_frames',0) for x in report),
               'total_video_frames': sum(x.get('video_frames',0) for x in report)}
    (a.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary['failed']: raise SystemExit(1)


if __name__ == '__main__': main()
