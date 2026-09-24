"""Audit a human A4 review CSV without treating decisions as ground-truth labels."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

ALLOWED={'pending','supports','contradicts','uncertain'}

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--time-maps',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    a=p.parse_args()
    reviews=read_csv(a.review);predictions=read_csv(a.predictions)
    if len(reviews)!=20 or len(predictions)!=20:raise ValueError('Expected 20 review and prediction rows')
    if set(reviews[0])!={'sample_id','review_decision','review_notes'}:raise ValueError('Unexpected review columns')
    by_id={r['sample_id']:r for r in reviews};by_pred={r['sample_id']:r for r in predictions}
    if len(by_id)!=20 or len(by_pred)!=20 or set(by_id)!=set(by_pred):
        raise ValueError('Duplicate or missing A4 IDs')
    if set(r['review_decision'] for r in reviews)-ALLOWED:raise ValueError('Unknown decision')
    contradictions=[]
    for sid,review in sorted(by_id.items()):
        mapping=json.loads((a.time_maps/(sid+'.json')).read_text(encoding='utf8'))
        if mapping['sample_id']!=sid:raise ValueError('Time-map ID mismatch')
        if review['review_decision']=='contradicts':
            prediction=by_pred[sid]
            cls=int(prediction['predicted_polarity'])
            contradictions.append({
                'sample_id':sid,'review_note_present':bool(review['review_notes'].strip()),
                'predicted_polarity':cls,
                'predicted_intensity':float(prediction['predicted_intensity']),
                'predicted_class_probability':float(prediction[['prob_negative','prob_neutral','prob_positive'][cls]]),
                'primary_reference_modality':prediction['primary_reference_modality'],
                'text_support_share':float(prediction['text_support_share']),
                'time_mapping_quality':prediction['time_mapping_quality'],
                'asr_exact_match_fraction':mapping['alignment_quality']['exact_match_fraction'],
                'asr_anchor_accepted':mapping['alignment_quality']['accepted']})
    a.snapshot.parent.mkdir(parents=True,exist_ok=True)
    a.summary.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.review,a.snapshot)
    digest=hashlib.sha256(a.snapshot.read_bytes()).hexdigest()
    if digest!=hashlib.sha256(a.review.read_bytes()).hexdigest():raise ValueError('Snapshot mismatch')
    summary={'source_file':a.review.name,'source_sha256':digest,'rows':len(reviews),
             'unique_ids':len(by_id),'decision_counts':dict(Counter(r['review_decision'] for r in reviews)),
             'notes_nonempty':sum(bool(r['review_notes'].strip()) for r in reviews),
             'contradiction_cases':contradictions,
             'interpretation':'Subjective human video review of overall prediction and local positions; not ground-truth test accuracy.'}
    a.summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
