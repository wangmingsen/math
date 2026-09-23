"""Probe all 100 Q1 source videos without modifying them.

Run: python E2026/probe_raw_videos.py --data-root ../E题数据/E题数据 --out E2026/raw_video_manifest.csv
"""
from __future__ import annotations
import argparse, csv, json, subprocess
from pathlib import Path
from audit_data import excel_rows

def probe(path: Path) -> dict:
    cmd = ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]
    run = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
    if run.returncode:
        return {'error': run.stderr.strip()[:300]}
    info = json.loads(run.stdout)
    streams = info.get('streams', [])
    video = next((s for s in streams if s.get('codec_type') == 'video'), {})
    audio = next((s for s in streams if s.get('codec_type') == 'audio'), {})
    duration = info.get('format', {}).get('duration') or video.get('duration')
    return {'duration_seconds': float(duration) if duration else None,
            'width': video.get('width'), 'height': video.get('height'),
            'frame_rate': video.get('avg_frame_rate') or video.get('r_frame_rate'),
            'video_codec': video.get('codec_name'), 'audio_codec': audio.get('codec_name'),
            'has_audio': bool(audio), 'has_video': bool(video), 'error': ''}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    label_path = next(root.glob('附件1-*/MOSEI*/label-100.xlsx'))
    base = label_path.parent
    labels = excel_rows(label_path)
    label_keys = {(str(r['video_id']), str(int(float(r['clip_id'])))) for r in labels}
    videos = sorted(base.rglob('*.mp4'))
    rows = []
    for path in videos:
        key = (path.parent.name, str(int(path.stem)))
        row = {'video_id': path.parent.name, 'clip_id': path.stem, 'sample_id': path.parent.name + '$_$' + path.stem,
               'relative_path': str(path.relative_to(base)), 'label_match': key in label_keys}
        row.update(probe(path))
        rows.append(row)
    columns = ['sample_id', 'video_id', 'clip_id', 'relative_path', 'label_match', 'duration_seconds', 'width', 'height', 'frame_rate', 'video_codec', 'audio_codec', 'has_audio', 'has_video', 'error']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    durations = [r['duration_seconds'] for r in rows if r['duration_seconds'] is not None]
    report = {'videos': len(rows), 'label_matched': sum(r['label_match'] for r in rows),
              'audio_streams': sum(r['has_audio'] for r in rows), 'video_streams': sum(r['has_video'] for r in rows),
              'probe_errors': [r['sample_id'] for r in rows if r['error']],
              'duration_min': min(durations) if durations else None, 'duration_max': max(durations) if durations else None}
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
