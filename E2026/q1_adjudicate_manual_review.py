"""Join a private Q1 video review export to the low-match audit without rewriting labels.

The per-sample output contains contest transcript context and must stay local.
Only the aggregate summary is suitable for the team repository.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DECISIONS = {'match', 'offset', 'mismatch', 'unintelligible', 'silent', 'other'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    a = p.parse_args()
    reviews = list(csv.DictReader(a.review.open(encoding='utf-8-sig', newline='')))
    audit = json.loads(a.audit.read_text(encoding='utf-8'))
    source = {r['sample_id']: r for r in audit}
    ids = [r['sample_id'] for r in reviews]
    if len(reviews) != 21 or len(set(ids)) != 21 or set(ids) != set(source):
        raise ValueError('Review must cover exactly the same 21 sample IDs as the audit')
    result = []
    for row in reviews:
        sid = row['sample_id']
        decision = row['manual_decision'].strip().lower()
        if decision not in DECISIONS:
            raise ValueError(f'{sid}: unsupported or pending decision {decision!r}')
        note = row['manual_notes'].strip()
        item = source[sid]
        if row['video_relative_path'].replace('\\', '/') != item['video_relative_path'].replace('\\', '/'):
            raise ValueError(f'{sid}: video path mismatch')
        silent_pcm = item['category'] == 'decoded_stereo_pcm_all_zero'
        flags = []
        if silent_pcm and ('音乐' in note or '人声' in note and '无人声' not in note):
            flags.append('manual_note_conflicts_with_zero_pcm')
        if decision == 'offset':
            flags.append('offset_requires_transcript_content_confirmation')
        if decision == 'other' and not note:
            flags.append('other_without_explanation')
        if decision == 'match' and silent_pcm:
            flags.append('speech_match_claimed_on_silent_pcm')
        if flags:
            disposition = 'needs_recheck'
        elif silent_pcm:
            disposition = 'no_speech_alignment_possible'
        elif decision == 'match':
            disposition = 'reported_transcript_match_time_unverified'
        elif decision == 'mismatch':
            disposition = 'reported_transcript_mismatch'
        elif decision == 'offset':
            disposition = 'reported_offset_time_unverified'
        elif '无人声' in note or '无声' in note or decision == 'silent':
            disposition = 'reported_no_speech'
        else:
            disposition = 'reported_unverifiable'
        result.append({
            'sample_id': sid,
            'video_relative_path': row['video_relative_path'],
            'automatic_category': item['category'],
            'pcm_rms': item['pcm_rms'],
            'manual_decision': decision,
            'manual_notes': note,
            'disposition': disposition,
            'review_flags': ';'.join(flags),
            'word_audio_alignment_usable': 'no',
        })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    summary = {
        'reviewed': len(result),
        'manual_decisions': dict(Counter(r['manual_decision'] for r in result)),
        'preliminary_dispositions': dict(Counter(r['disposition'] for r in result)),
        'needs_recheck': sum(r['disposition'] == 'needs_recheck' for r in result),
        'word_audio_alignment_usable': 0,
        'note': 'Manual review is a reported observation; no transcript or word timestamp was silently replaced.',
    }
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
