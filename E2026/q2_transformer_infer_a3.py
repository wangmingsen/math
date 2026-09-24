"""Predict A3 with the validation-selected aligned_50 ensemble; no labels."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from q2_transformer_distill import FusionTransformer,predict,read,tensors


def source_stages(name):
    return ('teacher','student') if name=='teacher_student_mean6' else (
        ('teacher',) if name=='teacher_mean3' else ('student',))


def interior_zero_count(values):
    active=np.any(values!=0,axis=1)
    indices=np.flatnonzero(active)
    return int((~active[indices[0]:indices[-1]+1]).sum()) if len(indices) else 0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    a=p.parse_args()
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else
                        a.device if a.device!='auto' else 'cpu')
    torch.set_num_threads(4)
    selected=json.loads(a.selection.read_text(encoding='utf8'))
    if selected['seeds']!=[2026,2027,2028]:raise ValueError('Unexpected selected seeds')
    data=read(a.data/'A3.npz')
    train=read(a.data/'train.npz')
    count=np.bincount(train['classification_labels'],minlength=3)
    prior={'probability':(count/count.sum()).astype('f4'),
           'intensity':float(train['regression_labels'].mean())}
    probabilities={};intensities={}
    normalizer=None;inputs=None
    for seed in selected['seeds']:
        for stage in ('teacher','student'):
            checkpoint=torch.load(a.runs/f'q2_transformer_seed_{seed}'/(stage+'.pt'),map_location='cpu')
            if checkpoint['train_seed']!=seed or checkpoint['mask_seed']!=seed+40000:
                raise ValueError('Checkpoint seed mismatch')
            if normalizer is None:
                normalizer=checkpoint['normalizer']
                inputs=tensors(data,normalizer,device)
            elif any(not np.array_equal(normalizer[key][j],checkpoint['normalizer'][key][j])
                     for key in normalizer for j in range(2)):
                raise ValueError('Train normalizer differs between runs')
            model=FusionTransformer().to(device)
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            _,prob,strength=predict(model,inputs,prior=prior)
            probabilities[(seed,stage)]=prob
            intensities[(seed,stage)]=strength
    class_stages=source_stages(selected['classification_component'])
    reg_stages=source_stages(selected['regression_component'])
    pmean=np.mean([probabilities[(seed,stage)] for seed in selected['seeds'] for stage in class_stages],axis=0)
    rmean=np.mean([intensities[(seed,stage)] for seed in selected['seeds'] for stage in reg_stages],axis=0)
    rmean=np.clip(rmean,-3,3)
    if (not np.isfinite(pmean).all() or not np.isfinite(rmean).all() or
            not np.allclose(pmean.sum(axis=1),1,atol=1e-6) or
            len(data['sample_id'])!=30 or len(set(data['sample_id']))!=30):
        raise ValueError('Invalid A3 predictions')
    available=inputs[1].cpu().numpy().any(axis=1)
    rows=[];quality=[]
    for i,sid in enumerate(data['sample_id']):
        label=int(pmean[i].argmax());intensity=float(rmean[i])
        flags=[]
        if not available[i]:flags.append('all_unavailable_prior')
        if label==0 and intensity>0 or label==2 and intensity<0:flags.append('polarity_intensity_sign')
        if label==1 and abs(intensity)>.5:flags.append('neutral_large_strength')
        rows.append({'sample_id':str(sid),'polarity':label,'intensity':intensity,
                     'prob_negative':float(pmean[i,0]),'prob_neutral':float(pmean[i,1]),
                     'prob_positive':float(pmean[i,2])})
        quality.append({'sample_id':str(sid),'flags':';'.join(flags),
                        'text_observed_segments':int(data['text_present'][i].sum()),
                        'audio_nonzero_rows':int(np.any(data['audio'][i]!=0,axis=1).sum()),
                        'vision_nonzero_rows':int(np.any(data['vision'][i]!=0,axis=1).sum()),
                        'audio_interior_zero_rows':interior_zero_count(data['audio'][i]),
                        'vision_interior_zero_rows':interior_zero_count(data['vision'][i])})
    a.out.mkdir(parents=True,exist_ok=True)
    for name,records in (('A3_predictions_transformer_candidate.csv',rows),
                         ('A3_prediction_quality_local.csv',quality)):
        with (a.out/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary={'samples':len(rows),'class_counts':{str(c):sum(r['polarity']==c for r in rows) for c in range(3)},
             'all_unavailable_prior':sum('all_unavailable_prior' in r['flags'] for r in quality),
             'polarity_intensity_flags':sum('polarity_intensity_sign' in r['flags'] for r in quality),
             'neutral_large_strength_flags':sum('neutral_large_strength' in r['flags'] for r in quality),
             'label_status':'A3 labels not supplied; no A3 performance metrics',
             'representation':'aligned_50 throughout'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
