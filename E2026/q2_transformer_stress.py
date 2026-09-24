"""A2-valid-only missingness matrix for frozen teacher and student.

Text is represented by five pooled segments, so requested text percentages
are rounded to whole segments. Both requested and actually changed rates are
reported; this is not a token-level 10% masking experiment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from q2_transformer_distill import FusionTransformer,make_tokens,predict,read,score,tensors


def selected_positions(left,right,count,position,blocks):
    span=right-left
    count=min(span,count)
    if count<=0:return []
    if blocks==1 or count==1:
        free=span-count
        start=left+round(free*position)
        return list(range(start,start+count))
    first=count//2;second=count-first
    free=span-count
    gap=min(max(1,free//2),free) if free else 0
    offset=round((free-gap)*position)
    start=left+offset
    return list(range(start,start+first))+list(range(start+first+gap,start+first+gap+second))


def inject(data,modalities,rate,position,blocks):
    out={key:value.copy() for key,value in data.items()}
    planned={key:0 for key in modalities};changed={key:0 for key in modalities};eligible={key:0 for key in modalities}
    for modality in modalities:
        if modality=='text':
            for i in range(len(out['text'])):
                indices=np.flatnonzero(data['text_present'][i])
                eligible[modality]+=len(indices)
                if not len(indices):continue
                count=max(1,round(len(indices)*rate))
                for j in selected_positions(0,len(indices),count,position,blocks):
                    out['text'][i,indices[j]]=0
                    out['text_present'][i,indices[j]]=False
                    planned[modality]+=1;changed[modality]+=1
        else:
            for i in range(len(out[modality])):
                active=np.any(data[modality][i]!=0,axis=1)
                indices=np.flatnonzero(active)
                eligible[modality]+=len(indices)
                if not len(indices):continue
                left,right=int(indices[0]),int(indices[-1])+1
                count=max(1,round((right-left)*rate))
                positions=selected_positions(left,right,count,position,blocks)
                planned[modality]+=len(positions)
                changed[modality]+=int(active[positions].sum())
                out[modality][i,positions]=0
    log={key:{'eligible':eligible[key],'planned':planned[key],
              'changed_original_nonzero':changed[key],
              'actual_changed_rate':changed[key]/eligible[key] if eligible[key] else None}
         for key in modalities}
    return out,log


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    a=p.parse_args()
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else
                        a.device if a.device!='auto' else 'cpu')
    torch.set_num_threads(4)
    valid=read(a.data/'valid.npz')
    checkpoints={name:torch.load(a.models/(name+'.pt'),map_location='cpu') for name in ('teacher','student')}
    norm=checkpoints['teacher']['normalizer']
    models={}
    for name,checkpoint in checkpoints.items():
        if any(not np.array_equal(norm[key][j],checkpoint['normalizer'][key][j])
               for key in norm for j in range(2)):
            raise ValueError('Teacher/student normalizer differs')
        model=FusionTransformer().to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval();models[name]=model
    train=read(a.data/'train.npz')
    count=np.bincount(train['classification_labels'],minlength=3)
    prior={'probability':(count/count.sum()).astype('f4'),
           'intensity':float(train['regression_labels'].mean())}
    groups={'T':('text',),'A':('audio',),'V':('vision',),
            'TA':('text','audio'),'TV':('text','vision'),
            'AV':('audio','vision'),'TAV':('text','audio','vision')}
    scenarios=[]
    configs=[('clean',None,None,None,None)]
    for group,modalities in groups.items():
        for rate in (.10,.20,.40,.60):
            for position_name,position in (('front',0.),('middle',.5),('back',1.)):
                for blocks in (1,2):
                    configs.append((group,modalities,rate,position_name,(position,blocks)))
    for index,(name,modalities,rate,position_name,detail) in enumerate(configs):
        if modalities is None:
            altered,log=valid,{}
        else:
            altered,log=inject(valid,modalities,rate,*detail)
        inputs=tensors(altered,norm,device)
        available=inputs[1].cpu().numpy().any(axis=1)
        row={'scenario':name if modalities is None else f'{name}_{int(rate*100)}_{position_name}_{detail[1]}block',
             'modalities':list(modalities) if modalities else [],
             'requested_rate':rate,'position':position_name,
             'blocks':detail[1] if detail else 0,
             'injection':log,'all_unavailable_count':int((~available).sum()),'models':{}}
        for model_name,model in models.items():
            _,prob,intensity=predict(model,inputs,prior=prior)
            row['models'][model_name]=score(valid['classification_labels'],prob,intensity,
                                            valid['regression_labels'])
        scenarios.append(row)
        if (index+1)%30==0:print(f'{index+1}/{len(configs)}',flush=True)
    report={'protocol':'A2 valid only; teacher/student weights frozen; A2 test and A3 not evaluated',
            'text_resolution_warning':'Five BERT segment vectors: requested text rates round to segment count; actual_changed_rate is authoritative.',
            'original_zero_warning':'A/V original zero rows have unknown cause; injected mask planned and changed counts are separate.',
            'all_unavailable_prior':{'probability':prior['probability'].tolist(),
                                     'intensity':prior['intensity'],'confidence':'low'},
            'scenarios':scenarios}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    brief={key:next(r for r in scenarios if r['scenario']==key)['models']
           for key in ('clean','AV_40_middle_1block','TAV_60_middle_2block')}
    print(json.dumps(brief,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
