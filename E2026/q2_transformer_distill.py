"""Small masked fusion Transformer with a clean teacher and robust student.

This is an A2 train/valid experiment. It never reads A2 test or A3 labels.
Run with a Python environment containing PyTorch and NumPy.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEED=2026
CLASSES=3


def read(path):
    with np.load(path) as z:
        return {key:z[key].copy() for key in z.files}


def fit_normalizer(train):
    result={}
    for key,present in (('text',train['text_present']),
                        ('audio',np.any(train['audio']!=0,axis=2)),
                        ('vision',np.any(train['vision']!=0,axis=2))):
        rows=train[key][present]
        mean=rows.mean(axis=0).astype('f4')
        std=np.maximum(rows.std(axis=0),1e-3).astype('f4')
        result[key]=(mean,std)
    return result


def transform_values(values,present,normalizer):
    mean,std=normalizer
    z=np.clip((values-mean)/std,-5,5).astype('f4')
    z[~present]=0
    return z


def pool_av(values):
    n,_,d=values.shape
    active=np.any(values!=0,axis=2)
    before=np.cumsum(active,axis=1)>0
    after=np.cumsum(active[:,::-1],axis=1)[:,::-1]>0
    middle=(~active)&before&after
    x=values.reshape(n,5,10,d)
    m=active.reshape(n,5,10)
    counts=m.sum(axis=2)
    means=(x*m[:,:,:,None]).sum(axis=2)/np.maximum(counts[:,:,None],1)
    pooled=np.concatenate((means,counts[:,:,None]/10,
                           middle.reshape(n,5,10).mean(axis=2)[:,:,None]),axis=2).astype('f4')
    return pooled,counts>0


def make_tokens(data,norm):
    text_present=data['text_present'].astype(bool)
    text=transform_values(data['text'],text_present,norm['text'])
    text=np.concatenate((text,text_present[:,:,None].astype('f4')),axis=2)
    parts=[text];masks=[text_present]
    for key in ('audio','vision'):
        raw=data[key]
        active=np.any(raw!=0,axis=2)
        clean=transform_values(raw,active,norm[key])
        pooled,mask=pool_av(clean)
        parts.append(pooled);masks.append(mask)
    return parts,np.concatenate(masks,axis=1)


def augment(train,rng):
    out={key:value.copy() for key,value in train.items()}
    choices=rng.integers(0,6,size=len(train['text']))
    injected={'text':np.zeros((len(choices),5),bool),
              'audio':np.zeros((len(choices),50),bool),
              'vision':np.zeros((len(choices),50),bool)}
    # 0 clean; 1 audio; 2 vision; 3 both; 4 text; 5 text+both.
    for i,mode in enumerate(choices):
        if mode in (4,5):
            candidates=np.flatnonzero(out['text_present'][i])
            if len(candidates):
                j=int(rng.choice(candidates))
                out['text'][i,j]=0
                out['text_present'][i,j]=False
                injected['text'][i,j]=True
        for modality,selected in (('audio',mode in (1,3,5)),('vision',mode in (2,3,5))):
            if not selected:continue
            original=train[modality][i]
            observed=np.flatnonzero(np.any(original!=0,axis=1))
            if not len(observed):continue
            left,right=int(observed[0]),int(observed[-1])+1
            fraction=float(rng.uniform(.10,.50))
            width=min(right-left,max(1,round((right-left)*fraction)))
            start=int(rng.integers(left,right-width+1))
            out[modality][i,start:start+width]=0
            injected[modality][i,start:start+width]=True
    return out,injected


def fixed_dual_gap(data):
    out={key:value.copy() for key,value in data.items()}
    for modality in ('audio','vision'):
        for i,original in enumerate(data[modality]):
            observed=np.flatnonzero(np.any(original!=0,axis=1))
            if not len(observed):continue
            left,right=int(observed[0]),int(observed[-1])+1
            width=min(right-left,max(1,round((right-left)*.25)))
            start=left+round((right-left-width)*.5)
            out[modality][i,start:start+width]=0
    return out


class FusionTransformer(nn.Module):
    def __init__(self,width=64,heads=4,layers=2):
        super().__init__()
        self.project=nn.ModuleList([nn.Linear(d,width) for d in (49,76,37)])
        self.modality=nn.Parameter(torch.zeros(3,width))
        self.position=nn.Parameter(torch.zeros(5,width))
        self.cls=nn.Parameter(torch.zeros(1,1,width))
        layer=nn.TransformerEncoderLayer(width,heads,dim_feedforward=128,dropout=.1,
                                         activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers)
        self.classifier=nn.Linear(width,CLASSES)
        self.regressor=nn.Linear(width,1)
        nn.init.normal_(self.modality,std=.02)
        nn.init.normal_(self.position,std=.02)
        nn.init.normal_(self.cls,std=.02)

    def forward(self,parts,mask):
        tokens=[]
        for j,x in enumerate(parts):
            tokens.append(self.project[j](x)+self.modality[j]+self.position)
        x=torch.cat(tokens,dim=1)
        x=torch.cat((self.cls.expand(len(x),-1,-1),x),dim=1)
        padding=torch.cat((torch.zeros(len(x),1,dtype=torch.bool,device=x.device),~mask),dim=1)
        representation=self.encoder(x,src_key_padding_mask=padding)[:,0]
        return self.classifier(representation),self.regressor(representation).squeeze(1)


def tensors(data,norm,device):
    parts,mask=make_tokens(data,norm)
    return [torch.from_numpy(x).to(device) for x in parts],torch.from_numpy(mask).to(device)


def predict(model,inputs,batch=256,prior=None):
    model.eval();parts,mask=inputs;logits=[];scores=[]
    with torch.no_grad():
        for start in range(0,len(mask),batch):
            output=model([x[start:start+batch] for x in parts],mask[start:start+batch])
            logits.append(output[0].cpu().numpy());scores.append(output[1].cpu().numpy())
    z=np.concatenate(logits);p=torch.softmax(torch.from_numpy(z),dim=1).numpy()
    r=np.concatenate(scores)
    if prior is not None:
        unavailable=~mask.cpu().numpy().any(axis=1)
        if unavailable.any():
            p[unavailable]=prior['probability']
            r[unavailable]=prior['intensity']
            z[unavailable]=np.log(np.maximum(prior['probability'],1e-8))
    return z,p,r


def score(y,p,r,truth):
    label=p.argmax(axis=1)
    f1=[]
    for c in range(3):
        tp=int(((label==c)&(y==c)).sum())
        fp=int(((label==c)&(y!=c)).sum())
        fn=int(((label!=c)&(y==c)).sum())
        f1.append(2*tp/max(2*tp+fp+fn,1))
    return {'accuracy':float(np.mean(label==y)),
            'f1_macro':float(np.mean(f1)),
            'f1_by_class':[float(x) for x in f1],
            'mae':float(np.mean(np.abs(r-truth))),
            'pearson':float(np.corrcoef(r,truth)[0,1])}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--teacher-epochs',type=int,default=12)
    parser.add_argument('--student-epochs',type=int,default=18)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--train-seed',type=int,default=2026)
    parser.add_argument('--mask-seed',type=int,default=42026)
    args=parser.parse_args()
    random.seed(args.train_seed);np.random.seed(args.train_seed);torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.train_seed)
    torch.set_num_threads(4)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        args.device if args.device!='auto' else 'cpu')
    train=read(args.data/'train.npz');valid=read(args.data/'valid.npz')
    norm=fit_normalizer(train)
    clean_train=tensors(train,norm,device)
    clean_valid=tensors(valid,norm,device)
    dual_valid=tensors(fixed_dual_gap(valid),norm,device)
    y=torch.from_numpy(train['classification_labels']).to(device)
    r=torch.from_numpy(train['regression_labels']).to(device)
    counts=np.bincount(train['classification_labels'],minlength=3)
    weights=np.sqrt(counts.sum()/np.maximum(counts,1)).astype('f4')
    weights=weights/weights.mean()
    class_weights=torch.from_numpy(weights).to(device)
    prior={'probability':(counts/counts.sum()).astype('f4'),
           'intensity':float(train['regression_labels'].mean())}
    args.out.mkdir(parents=True,exist_ok=True)
    models={};reports={};valid_outputs={}
    rng=np.random.default_rng(args.mask_seed)
    injection_epochs=[]
    for stage,epochs in (('teacher',args.teacher_epochs),('student',args.student_epochs)):
        model=FusionTransformer().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
        best=-1e9;best_epoch=0;history=[]
        teacher_target=None
        if stage=='student':
            teacher_target=predict(models['teacher'],clean_train)
            target_logits=torch.from_numpy(teacher_target[0]).to(device)
            target_reg=torch.from_numpy(teacher_target[2]).to(device)
        for epoch in range(epochs):
            if stage=='student':
                augmented,injected=augment(train,rng)
                training_inputs=tensors(augmented,norm,device)
                original_audio=np.any(train['audio']!=0,axis=2)
                original_vision=np.any(train['vision']!=0,axis=2)
                injection_epochs.append({'text':injected['text'],
                    'audio':injected['audio'],'vision':injected['vision'],
                    'audio_changed':injected['audio']&original_audio,
                    'vision_changed':injected['vision']&original_vision})
            else:training_inputs=clean_train
            model.train()
            order=torch.randperm(len(y),device=device)
            losses=[]
            for indices in order.split(64):
                logits,reg=model([x[indices] for x in training_inputs[0]],training_inputs[1][indices])
                loss=F.cross_entropy(logits,y[indices],weight=class_weights)+.5*F.smooth_l1_loss(reg,r[indices])
                if stage=='student':
                    temperature=2.
                    loss=loss+.3*temperature**2*F.kl_div(
                        F.log_softmax(logits/temperature,dim=1),
                        F.softmax(target_logits[indices]/temperature,dim=1),reduction='batchmean')
                    loss=loss+.1*F.smooth_l1_loss(reg,target_reg[indices])
                optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();losses.append(float(loss.detach().cpu()))
            _,p_clean,r_clean=predict(model,clean_valid)
            _,p_dual,r_dual=predict(model,dual_valid)
            clean=score(valid['classification_labels'],p_clean,r_clean,valid['regression_labels'])
            dual=score(valid['classification_labels'],p_dual,r_dual,valid['regression_labels'])
            # Both clean and dual-gap validation determine the checkpoint.
            objective=(clean['f1_macro']+dual['f1_macro'])/2-.1*(clean['mae']+dual['mae'])/2
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),
                            'clean':clean,'dual_25pct_middle':dual,'selection_score':objective})
            print(json.dumps({'stage':stage,'epoch':epoch+1,'loss':round(float(np.mean(losses)),4),
                              'clean_f1':round(clean['f1_macro'],4),'dual_f1':round(dual['f1_macro'],4),
                              'clean_mae':round(clean['mae'],4)},ensure_ascii=False),flush=True)
            if objective>best+1e-5:
                best=objective;best_epoch=epoch+1
                state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            if epoch-best_epoch>=4:break
        model.load_state_dict(state)
        model.eval();models[stage]=model
        z,pred,reg=predict(model,clean_valid)
        zd,pd,rd=predict(model,dual_valid)
        valid_outputs[stage]={'clean_probability':pred,'clean_intensity':reg,
                              'dual_probability':pd,'dual_intensity':rd}
        reports[stage]={'best_epoch':best_epoch,'best':history[best_epoch-1],
                        'history':history,'model_parameters':sum(p.numel() for p in model.parameters())}
        torch.save({'state_dict':state,'normalizer':norm,'architecture':{'width':64,'heads':4,'layers':2},
                    'train_seed':args.train_seed,'mask_seed':args.mask_seed,'stage':stage},args.out/(stage+'.pt'))
    np.savez_compressed(args.out/'valid_predictions.npz',sample_id=valid['sample_id'],
                        classification_labels=valid['classification_labels'],
                        regression_labels=valid['regression_labels'],
                        **{stage+'_'+key:value for stage,output in valid_outputs.items() for key,value in output.items()})
    np.savez_compressed(args.out/'injection_masks.npz',sample_id=train['sample_id'],
        **{key:np.asarray([epoch[key] for epoch in injection_epochs],dtype=bool)
           for key in ('text','audio','vision','audio_changed','vision_changed')})
    report={'protocol':'A2 train/valid only; frozen train-only PCA; no A2 test or A3 labels',
            'device':str(device),'train_seed':args.train_seed,'mask_seed':args.mask_seed,'teacher_student':reports,
            'prior':{'probability':prior['probability'].tolist(),'intensity':prior['intensity'],
                     'usage':'all unavailable -> training prior and low-confidence flag at inference'},
            'injection_masks':'injection_masks.npz stores per-epoch planned masks and changed nonzero A/V rows',
            'mask_semantics':'text padding from supplied attention; A/V original zero cause unknown; injection masks known by construction',
            'candidate_selection':'Compare validation results against frozen Q2 late-fusion; do not assume Transformer improves.'}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({stage:reports[stage]['best'] for stage in reports},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
