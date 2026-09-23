"""A4 inference and reproducible modality/segment occlusion sensitivity.

Occlusion changes model inputs. Its deltas are sensitivity evidence, not causal
effects. Video seconds are included only when q3_time_map accepts ASR anchors.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from aligned_data import load_small, normalize
from q2_missing_baseline import features

MODALITIES=('text','audio','vision')


def predict(bundle,text,mask,audio,vision):
    x_text=features(text,mask,audio,vision,True)
    x_av=features(text,mask,audio,vision)[:,x_text.shape[1]:]
    wc=bundle['class_weight'];wr=bundle['reg_weight']
    probability=(1-wc)*bundle['text_model'][0].predict_proba(x_text)+wc*bundle['av_classifier'].predict_proba(x_av)
    intensity=(1-wr)*bundle['text_model'][1].predict(x_text)+wr*bundle['av_regressor'].predict(x_av)
    return probability,intensity


def remove(text,mask,audio,vision,modality,start=0,end=50):
    t=text.copy();m=mask.copy();a=audio.copy();v=vision.copy()
    if modality=='text':
        lo,hi=start//10,(end+9)//10
        t[:,lo:hi]=0;m[:,lo:hi]=False
    elif modality=='audio':a[:,start:end]=0
    elif modality=='vision':v[:,start:end]=0
    else:raise ValueError(modality)
    return t,m,a,v


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


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--text',type=Path,required=True)
    p.add_argument('--time-maps',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    bundle=joblib.load(a.model)
    if bundle.get('version')!='q2_late_fusion_v1':raise ValueError('Unexpected Q2 model version')
    with np.load(a.text) as z:
        ids=z['sample_id'].astype(str)
        segments=z['segments'].astype(np.float32)
        present=z['segment_present'].astype(bool)
    if len(ids)!=20 or len(set(ids))!=20:raise ValueError('Expected 20 unique A4 IDs')
    transformed=bundle['text_pca'].transform(segments.reshape(-1,768)).reshape(20,5,48).astype(np.float32)
    transformed[~present]=0
    root=next(a.data_root.resolve().glob('附件4-*'))
    paths=sorted(root.rglob('对齐版本/*.pkl'))
    samples=[];audio=[];vision=[];maps=[]
    for i,path in enumerate(paths):
        item=load_small(path);sid=str(item['id'])
        if sid!=ids[i] or sid!=path.stem:raise ValueError('A4 ID/order mismatch')
        sample=normalize(item,sid,'A4')
        samples.append(sample)
        audio.append(sample['audio']);vision.append(sample['vision'])
        mapping=json.loads((a.time_maps/(sid+'.json')).read_text(encoding='utf-8'))
        if mapping['sample_id']!=sid:raise ValueError('A4 time map ID mismatch')
        maps.append(mapping)
    audio=np.asarray(audio,dtype=np.float32);vision=np.asarray(vision,dtype=np.float32)
    probabilities,intensities=predict(bundle,transformed,present,audio,vision)
    predictions=[];modality_rows=[];local_rows=[]
    for i,sid in enumerate(ids):
        t=transformed[i:i+1];m=present[i:i+1];au=audio[i:i+1];vi=vision[i:i+1]
        label=int(probabilities[i].argmax());base_conf=float(probabilities[i,label]);base_strength=float(intensities[i])
        impacts={}
        for modality in MODALITIES:
            ablated=remove(t,m,au,vi,modality)
            p2,r2=predict(bundle,*ablated)
            drop=base_conf-float(p2[0,label])
            strength_drop=base_strength-float(r2[0])
            support=max(0.,drop)
            impacts[modality]=support
            modality_rows.append({'sample_id':sid,'modality':modality,
                                  'baseline_class_confidence':base_conf,
                                  'ablated_class_confidence':float(p2[0,label]),
                                  'class_confidence_drop':drop,
                                  'support_score_positive_drop':support,
                                  'intensity_change_signed':strength_drop,
                                  'ablation_warning':'full-modality zero input may be outside training distribution'})
            for j in range(5):
                start,end=j*10,(j+1)*10
                p3,r3=predict(bundle,*remove(t,m,au,vi,modality,start,end))
                excerpt,time_start,time_end,time_quality=locate(maps[i],start,end)
                observed=(sum(p['char_start'] is not None for p in maps[i]['positions'][start:end]) if modality=='text' else
                          int(np.any((au if modality=='audio' else vi)[0,start:end]!=0,axis=1).sum()))
                local_rows.append({'sample_id':sid,'modality':modality,
                                   'position_start':start,'position_end_exclusive':end,
                                   'observed_positions':observed,
                                   'class_confidence_drop':base_conf-float(p3[0,label]),
                                   'intensity_change_signed':base_strength-float(r3[0]),
                                   'text_excerpt':excerpt,
                                   'estimated_start_s':time_start,'estimated_end_s':time_end,
                                   'time_quality':time_quality})
        total=sum(impacts.values())
        shares={key:(impacts[key]/total if total>0 else 0.) for key in MODALITIES}
        primary=max(MODALITIES,key=lambda key:impacts[key]) if total>1e-8 else 'undetermined'
        coherence=('opposite_sign' if (label==0 and base_strength>0) or (label==2 and base_strength<0)
                   else 'neutral_large_nonzero_strength' if label==1 and abs(base_strength)>.5
                   else 'none')
        predictions.append({'sample_id':sid,'predicted_polarity':label,
                            'predicted_intensity':base_strength,'coherence_qa_flag':coherence,
                            'prob_negative':float(probabilities[i,0]),
                            'prob_neutral':float(probabilities[i,1]),
                            'prob_positive':float(probabilities[i,2]),
                            'primary_reference_modality':primary,
                            'text_support_share':shares['text'],
                            'audio_support_share':shares['audio'],
                            'vision_support_share':shares['vision'],
                            'time_mapping_quality':('asr_estimate' if maps[i]['alignment_quality']['accepted'] else 'unverified'),
                            'text_truncated_at_50':maps[i]['text_truncated_at_50'],
                            'vision_observed':bool(np.any(vi!=0))})
    a.out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('predictions.csv',predictions),('modality_impacts.csv',modality_rows),
                      ('local_evidence.csv',local_rows)]:
        with (a.out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'samples':len(predictions),
             'polarity_counts':{str(c):sum(x['predicted_polarity']==c for x in predictions) for c in range(3)},
             'primary_modality_counts':{key:sum(x['primary_reference_modality']==key for x in predictions)
                                        for key in (*MODALITIES,'undetermined')},
             'asr_estimated_time_samples':sum(x['time_mapping_quality']=='asr_estimate' for x in predictions),
             'time_unverified_samples':sum(x['time_mapping_quality']=='unverified' for x in predictions),
             'text_truncated_samples':sum(x['text_truncated_at_50'] for x in predictions),
             'no_vision_samples':sum(not x['vision_observed'] for x in predictions),
             'coherence_qa_flags':{key:sum(x['coherence_qa_flag']==key for x in predictions)
                                   for key in ('none','opposite_sign','neutral_large_nonzero_strength')},
             'interpretation':'occlusion sensitivity only; not causal importance or verified speech-video alignment'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
