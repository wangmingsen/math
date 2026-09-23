"""Join verified Q1 BERT, openSMILE and OpenFace features on one time grid."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    for key in ('source','bert','smile','openface','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    paths={k:getattr(a,k).resolve() for k in ('source','bert','smile','openface')}
    base=list(csv.DictReader((paths['source']/'manifest.csv').open(encoding='utf-8-sig',newline='')))
    branch={}
    for key in ('bert','smile','openface'):
        rows=list(csv.DictReader((paths[key]/'manifest.csv').open(encoding='utf-8-sig',newline='')))
        branch[key]={r['sample_id']:r for r in rows}
        if len(rows)!=len(base) or set(branch[key])!={r['sample_id'] for r in base}:
            raise ValueError(f'{key}: sample ID mismatch')
    a.out.mkdir(parents=True,exist_ok=True)
    report=[]
    for row in base:
        sample=row['sample_id']; name=row['feature_file']
        if any(branch[key][sample]['status']!='ok' for key in branch):
            raise ValueError(f'{sample}: incomplete extraction')
        with np.load(paths['source']/name) as old, \
             np.load(paths['bert']/name) as bert, \
             np.load(paths['smile']/name) as smile, \
             np.load(paths['openface']/name) as face:
            edges=old['time_edges_s'].astype(np.float32)
            for item in (bert,smile,face):
                if not np.allclose(edges,item['time_edges_s'],atol=1e-6):
                    raise ValueError(f'{sample}: timeline mismatch')
            text=bert['text_bert']; audio=smile['audio_opensmile']; vision=face['vision_openface']
            if text.shape!=(50,768) or audio.shape!=(50,50) or vision.shape!=(50,35):
                raise ValueError(f'{sample}: shape mismatch')
            if not all(np.isfinite(x).all() for x in (text,audio,vision)):
                raise ValueError(f'{sample}: nonfinite feature')
            signal=smile['signal_present'].astype(bool)
            valid_face=face['face_valid'].astype(bool)
            if np.any(~valid_face & np.any(vision!=0,axis=1)):
                raise ValueError(f'{sample}: invalid face bins are nonzero')
            np.savez_compressed(a.out/name,text_bert=text,audio_opensmile=audio,
                                vision_openface=vision,time_edges_s=edges,
                                text_present=bert['text_present'],
                                audio_signal_present=signal,face_valid=valid_face,
                                text_word_count=bert['word_count'],
                                audio_frame_count=smile['frame_count'],
                                face_frame_count=face['face_frame_count'],
                                face_confidence=face['face_confidence'])
            report.append({'sample_id':sample,'feature_file':name,
                           'mapping_file':row['mapping_file'],
                           'duration_s':row['duration_s'],
                           'text_bins':int(bert['text_present'].sum()),
                           'audio_signal_bins':int(signal.sum()),
                           'face_bins':int(valid_face.sum()),'status':'ok'})
        mapping=json.loads((paths['source']/row['mapping_file']).read_text(encoding='utf-8'))
        for word in mapping['words']:
            if word.get('timing')=='asr_exact_word_anchor':
                word['timing']='asr_midpoint_derived_estimate'
        mapping['q1_v3_provenance']={
            'transcript':'contest-provided label-100.xlsx, unchanged',
            'word_time':'Q1 v2 estimated intervals; accepted ASR anchors use neighboring midpoint boundaries',
            'bert':'last encoder hidden state; mean WordPieces per word; mean words per bin',
            'audio':'openSMILE eGeMAPSv02 LLD, mean+std per bin',
            'vision':'OpenFace AU intensity/occurrence mean over successful face frames',
            'bin_time':'50 equal-duration original-video bins'}
        (a.out/row['mapping_file']).write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
    with (a.out/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]))
        writer.writeheader();writer.writerows(report)
    summary={'samples':len(report),'text_shape':[50,768],
             'audio_shape':[50,50],'vision_shape':[50,35],
             'shared_timeline':'50 equal-duration bins per original video',
             'silent_samples':sum(x['audio_signal_bins']==0 for x in report),
             'no_face_samples':sum(x['face_bins']==0 for x in report),
             'text_present_bins':sum(x['text_bins'] for x in report),
             'audio_signal_bins':sum(x['audio_signal_bins'] for x in report),
             'face_valid_bins':sum(x['face_bins'] for x in report),
             'source_alignment':'Q1 v2 estimated transcript word times; not forced-alignment truth'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

