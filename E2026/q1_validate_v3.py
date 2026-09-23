"""Independently check Q1 v3 against the official 100-row transcript table."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from audit_data import excel_rows
from q1_extract import TOKEN_PATTERN

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    root=a.features.resolve()
    label_file=next(a.data_root.resolve().glob('附件1-*/MOSEI*/label-100.xlsx'))
    labels={str(x['video_id'])+'$_$'+str(int(float(x['clip_id']))):x for x in excel_rows(label_file)}
    rows=list(csv.DictReader((root/'manifest.csv').open(encoding='utf-8-sig',newline='')))
    problems=[]
    totals={'text_bins':0,'audio_bins':0,'face_bins':0,'words':0}
    ids=[]
    for row in rows:
        sid=row['sample_id'];ids.append(sid)
        if sid not in labels:
            problems.append(sid+': not in supplied labels');continue
        mapping=json.loads((root/row['mapping_file']).read_text(encoding='utf-8'))
        expected=TOKEN_PATTERN.findall(str(labels[sid].get('text','')))
        actual=[x['token'] for x in mapping['words']]
        if actual!=expected or mapping['sample_id']!=sid:
            problems.append(sid+': transcript or identity mismatch')
        if sorted(j for b in mapping['bins'] for j in b['word_indices'])!=list(range(len(actual))):
            problems.append(sid+': word-to-bin mapping mismatch')
        if any(x.get('timing')=='asr_exact_word_anchor' for x in mapping['words']):
            problems.append(sid+': misleading exact timing label remains')
        with np.load(root/row['feature_file']) as d:
            for key,shape in {'text_bert':(50,768),'audio_opensmile':(50,50),
                              'vision_openface':(50,35),'time_edges_s':(51,)}.items():
                if d[key].shape!=shape or not np.isfinite(d[key]).all():
                    problems.append(sid+': invalid '+key)
            edges=d['time_edges_s']
            if not np.all(np.diff(edges)>0) or abs(float(edges[-1])-float(mapping['duration_s']))>1e-4:
                problems.append(sid+': invalid time grid')
            for i,b in enumerate(mapping['bins']):
                if abs(float(edges[i])-b['start_s'])>1e-4 or abs(float(edges[i+1])-b['end_s'])>1e-4:
                    problems.append(sid+': mapped seconds mismatch');break
            counts=d['text_word_count']
            if sum(counts)!=len(actual) or not np.array_equal(counts>0,d['text_present']):
                problems.append(sid+': text count/mask mismatch')
            face=d['face_valid']
            if not np.array_equal(d['face_frame_count']>0,face) or np.any(d['vision_openface'][~face]!=0):
                problems.append(sid+': face mask mismatch')
            for key,mask in [('text_bert',d['text_present']),('audio_opensmile',d['audio_signal_present'])]:
                if np.any(d[key][~mask]!=0):
                    problems.append(sid+': nonzero '+key+' in absent bin')
            totals['text_bins']+=int(d['text_present'].sum())
            totals['audio_bins']+=int(d['audio_signal_present'].sum())
            totals['face_bins']+=int(face.sum())
            totals['words']+=len(actual)
    if len(rows)!=100 or len(set(ids))!=100 or set(ids)!=set(labels):
        problems.append('100-row label coverage failed')
    summary={'samples':len(rows),'unique_ids':len(set(ids)),
             'label_ids_matched':len(set(ids)&set(labels)),
             'feature_files':len(list(root.glob('*.npz'))),
             'mapping_files':len([p for p in root.glob('*.json') if p.name!='summary.json']),
             'totals':totals,'problems':problems}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if problems:raise SystemExit(1)

if __name__=='__main__':main()
