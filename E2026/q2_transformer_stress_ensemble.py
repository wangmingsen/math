"""Run the full A2-valid stress matrix on the selected six-checkpoint ensemble."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from q2_transformer_distill import FusionTransformer,predict,read,score,tensors
from q2_transformer_stress import inject


def stages(name):
    if name=='teacher_mean3':return ('teacher',)
    if name=='student_mean3':return ('student',)
    if name=='teacher_student_mean6':return ('teacher','student')
    raise ValueError(name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    a=p.parse_args();torch.set_num_threads(4)
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else
                        a.device if a.device!='auto' else 'cpu')
    selected=json.loads(a.selection.read_text(encoding='utf8'))
    valid=read(a.data/'valid.npz');train=read(a.data/'train.npz')
    counts=np.bincount(train['classification_labels'],minlength=3)
    prior={'probability':(counts/counts.sum()).astype('f4'),
           'intensity':float(train['regression_labels'].mean())}
    models={};normalizer=None
    for seed in selected['seeds']:
        for stage in ('teacher','student'):
            checkpoint=torch.load(a.runs/f'q2_transformer_seed_{seed}'/(stage+'.pt'),map_location='cpu')
            if normalizer is None:normalizer=checkpoint['normalizer']
            elif any(not np.array_equal(normalizer[key][j],checkpoint['normalizer'][key][j])
                     for key in normalizer for j in range(2)):
                raise ValueError('Normalizer mismatch')
            model=FusionTransformer().to(device)
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            models[(seed,stage)]=model
    groups={'T':('text',),'A':('audio',),'V':('vision',),
            'TA':('text','audio'),'TV':('text','vision'),
            'AV':('audio','vision'),'TAV':('text','audio','vision')}
    configs=[('clean',(),0.,'none',0,0.)]
    for group,modalities in groups.items():
        for rate in (.10,.20,.40,.60):
            for position_name,position in (('front',0.),('middle',.5),('back',1.)):
                for blocks in (1,2):
                    configs.append((group,modalities,rate,position_name,blocks,position))
    rows=[]
    for index,(group,modalities,rate,position_name,blocks,position) in enumerate(configs):
        altered,log=inject(valid,modalities,rate,position,blocks) if modalities else (valid,{})
        inputs=tensors(altered,normalizer,device)
        outputs={key:predict(model,inputs,prior=prior)[1:]
                 for key,model in models.items()}
        p=np.mean([outputs[(seed,stage)][0] for seed in selected['seeds']
                   for stage in stages(selected['classification_component'])],axis=0)
        r=np.mean([outputs[(seed,stage)][1] for seed in selected['seeds']
                   for stage in stages(selected['regression_component'])],axis=0)
        r=np.clip(r,-3,3)
        available=inputs[1].cpu().numpy().any(axis=1)
        rows.append({'scenario':'clean' if group=='clean' else f'{group}_{int(rate*100)}_{position_name}_{blocks}block',
                     'modalities':list(modalities),'requested_rate':rate,'position':position_name,
                     'blocks':blocks,'injection':log,
                     'all_unavailable_count':int((~available).sum()),
                     'metrics':score(valid['classification_labels'],p,r,valid['regression_labels'])})
        if (index+1)%30==0:print(f'{index+1}/{len(configs)}',flush=True)
    report={'protocol':'A2 valid only; validation-selected ensemble frozen. No A2 test or A3 labels.',
            'component_selection':{'classification':selected['classification_component'],
                                   'regression':selected['regression_component']},
            'text_resolution_warning':'Five pooled text segments: actual_changed_rate may greatly exceed requested 10% or 20%.',
            'original_zero_warning':'A/V original zero cause unknown; planned and changed counts reported separately.',
            'scenarios':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({key:next(x for x in rows if x['scenario']==key)['metrics']
                      for key in ('clean','AV_40_middle_1block','TAV_60_middle_2block')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
