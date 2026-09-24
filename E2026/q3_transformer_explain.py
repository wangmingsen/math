"""A4 predictions and occlusion explanations from the frozen Q2 ensemble.

The reported input effects are sensitivity under zero occlusion, not causal
attribution. A4 has no labels and its PKLs provide no video-second timestamps.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from q2_transformer_distill import FusionTransformer,predict,read,tensors
from q2_transformer_infer_a3 import source_stages


MODALITIES=('text','audio','vision')

def locate(mapping,start,end):
    positions=[p for p in mapping['positions'][start:end] if p['char_start'] is not None]
    if not positions:return '',None,None,'no_text_position'
    lo=min(p['char_start'] for p in positions);hi=max(p['char_end'] for p in positions)
    overlapping=[w for w in mapping['words'] if w['char_start']<hi and w['char_end']>lo]
    if overlapping:
        lo=min(w['char_start'] for w in overlapping)
        hi=max(w['char_end'] for w in overlapping)
    excerpt=mapping['raw_text'][lo:hi]
    times=[p for p in positions if p['start_s'] is not None]
    if not times:return excerpt,None,None,'time_unverified'
    return excerpt,min(p['start_s'] for p in times),max(p['end_s'] for p in times),'asr_estimate'


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--time-maps',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--train-data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    a=p.parse_args();torch.set_num_threads(4)
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else
                        a.device if a.device!='auto' else 'cpu')
    selected=json.loads(a.selection.read_text(encoding='utf8'))
    if selected['seeds']!=[2026,2027,2028]:raise ValueError('Unexpected Q2 seeds')
    with np.load(a.prepared) as z:
        ids=z['sample_id'].astype(str)
        text=z['text'].astype('f4')
        mask=z['text_present'].astype(bool)
        audio=z['audio'].astype('f4')
        vision=z['vision'].astype('f4')
    if len(ids)!=20 or len(set(ids))!=20:raise ValueError('Expected 20 unique A4 IDs')
    if text.shape!=(20,5,48) or audio.shape!=(20,50,74) or vision.shape!=(20,50,35):
        raise ValueError('Unexpected A4 tensor shapes')
    maps=[]
    for sid in ids:
        mapping=json.loads((a.time_maps/(sid+'.json')).read_text(encoding='utf8'))
        if mapping['sample_id']!=sid:raise ValueError('A4 map ID mismatch')
        maps.append(mapping)
    data={'text':text,'text_present':mask,'audio':audio,'vision':vision}
    train=read(a.train_data)
    counts=np.bincount(train['classification_labels'],minlength=3)
    prior={'probability':(counts/counts.sum()).astype('f4'),
           'intensity':float(train['regression_labels'].mean())}
    normalizer=None;models={}
    for seed in selected['seeds']:
        for stage in ('teacher','student'):
            checkpoint=torch.load(a.runs/f'q2_transformer_seed_{seed}'/(stage+'.pt'),map_location='cpu')
            if checkpoint['train_seed']!=seed or checkpoint['mask_seed']!=seed+40000:
                raise ValueError('Checkpoint seed mismatch')
            if normalizer is None:normalizer=checkpoint['normalizer']
            elif any(not np.array_equal(normalizer[key][j],checkpoint['normalizer'][key][j])
                     for key in normalizer for j in range(2)):
                raise ValueError('Normalizer mismatch')
            model=FusionTransformer().to(device)
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            models[(seed,stage)]=model

    def ensemble(values):
        inputs=tensors(values,normalizer,device)
        outputs={key:predict(model,inputs,prior=prior)[1:]
                 for key,model in models.items()}
        probability=np.mean([outputs[(seed,stage)][0] for seed in selected['seeds']
                             for stage in source_stages(selected['classification_component'])],axis=0)
        intensity=np.mean([outputs[(seed,stage)][1] for seed in selected['seeds']
                           for stage in source_stages(selected['regression_component'])],axis=0)
        return probability,np.clip(intensity,-3,3)

    variants={'baseline':data}
    for modality in MODALITIES:
        full={key:value.copy() for key,value in data.items()}
        if modality=='text':full['text'][:]=0;full['text_present'][:]=False
        else:full[modality][:]=0
        variants[f'{modality}_full']=full
        for j in range(5):
            local={key:value.copy() for key,value in data.items()}
            if modality=='text':
                local['text'][:,j]=0;local['text_present'][:,j]=False
            else:local[modality][:,j*10:(j+1)*10]=0
            variants[f'{modality}_{j}']=local
    outputs={name:ensemble(value) for name,value in variants.items()}
    base_prob,base_strength=outputs['baseline']
    if not np.isfinite(base_prob).all() or not np.isfinite(base_strength).all() or not np.allclose(base_prob.sum(axis=1),1,atol=1e-6):
        raise ValueError('Invalid A4 prediction')
    predictions=[];modality_rows=[];local_rows=[]
    for i,sid in enumerate(ids):
        label=int(base_prob[i].argmax());confidence=float(base_prob[i,label]);strength=float(base_strength[i])
        support={}
        for modality in MODALITIES:
            p2,r2=outputs[f'{modality}_full']
            drop=confidence-float(p2[i,label])
            support[modality]=max(0.,drop)
            modality_rows.append({'sample_id':sid,'modality':modality,
                                  'baseline_class_confidence':confidence,
                                  'ablated_class_confidence':float(p2[i,label]),
                                  'class_confidence_drop':drop,
                                  'support_score_positive_drop':support[modality],
                                  'intensity_change_signed':strength-float(r2[i]),
                                  'ablation_warning':'full-modality zero input may be outside training distribution'})
            for j in range(5):
                start,end=j*10,(j+1)*10
                p3,r3=outputs[f'{modality}_{j}']
                excerpt,first,last,time_quality=locate(maps[i],start,end)
                observed=(sum(item['char_start'] is not None for item in maps[i]['positions'][start:end])
                          if modality=='text' else
                          int(np.any(data[modality][i,start:end]!=0,axis=1).sum()))
                local_rows.append({'sample_id':sid,'modality':modality,
                                   'position_start':start,'position_end_exclusive':end,
                                   'observed_positions':observed,
                                   'class_confidence_drop':confidence-float(p3[i,label]),
                                   'intensity_change_signed':strength-float(r3[i]),
                                   'text_excerpt':excerpt,'estimated_start_s':first,
                                   'estimated_end_s':last,'time_quality':time_quality})
        total=sum(support.values())
        shares={key:(support[key]/total if total else 0.) for key in MODALITIES}
        primary=max(MODALITIES,key=lambda key:support[key]) if total>1e-8 else 'undetermined'
        coherence=('opposite_sign' if label==0 and strength>0 or label==2 and strength<0 else
                   'neutral_large_nonzero_strength' if label==1 and abs(strength)>.5 else 'none')
        predictions.append({'sample_id':sid,'predicted_polarity':label,
                            'predicted_intensity':strength,'coherence_qa_flag':coherence,
                            'prob_negative':float(base_prob[i,0]),
                            'prob_neutral':float(base_prob[i,1]),
                            'prob_positive':float(base_prob[i,2]),
                            'primary_reference_modality':primary,
                            'text_support_share':shares['text'],
                            'audio_support_share':shares['audio'],
                            'vision_support_share':shares['vision'],
                            'time_mapping_quality':('asr_estimate' if maps[i]['alignment_quality']['accepted'] else 'unverified'),
                            'text_truncated_at_50':maps[i]['text_truncated_at_50'],
                            'vision_observed':bool(np.any(data['vision'][i]!=0))})
    a.out.mkdir(parents=True,exist_ok=True)
    write_csv(a.out/'predictions.csv',predictions)
    write_csv(a.out/'modality_impacts.csv',modality_rows)
    write_csv(a.out/'local_evidence.csv',local_rows)
    summary={'samples':20,'model':'q2_transformer_selected_ensemble',
             'polarity_counts':{str(c):sum(row['predicted_polarity']==c for row in predictions) for c in range(3)},
             'primary_modality_counts':{key:sum(row['primary_reference_modality']==key for row in predictions)
                                        for key in (*MODALITIES,'undetermined')},
             'asr_estimated_time_samples':sum(row['time_mapping_quality']=='asr_estimate' for row in predictions),
             'time_unverified_samples':sum(row['time_mapping_quality']=='unverified' for row in predictions),
             'text_truncated_samples':sum(row['text_truncated_at_50'] for row in predictions),
             'no_vision_samples':sum(not row['vision_observed'] for row in predictions),
             'coherence_qa_flags':{key:sum(row['coherence_qa_flag']==key for row in predictions)
                                   for key in ('none','opposite_sign','neutral_large_nonzero_strength')},
             'interpretation':'occlusion sensitivity only; not causal importance or verified speech-video alignment'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
