"""Q2: compare clean and missingness-augmented aligned_50 baselines."""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,roc_auc_score

SEED=2026

def project(train,valid):
    pca=PCA(n_components=48,svd_solver='randomized',random_state=SEED)
    pca.fit(train['segments'][train['segment_present']])
    def convert(data):
        mask=data['segment_present'].astype(bool)
        values=pca.transform(data['segments'].reshape(-1,768)).reshape(len(mask),5,48).astype('f4')
        values[~mask]=0
        return values,mask
    return convert(train),convert(valid)

def pool(seq):
    n,_,d=seq.shape
    active=np.any(seq!=0,axis=2)
    before=np.cumsum(active,axis=1)>0
    after=np.cumsum(active[:,::-1],axis=1)[:,::-1]>0
    middle=(~active)&before&after
    x=seq.reshape(n,5,10,d)
    m=active.reshape(n,5,10)
    count=m.sum(axis=2)
    mean=(x*m[:,:,:,None]).sum(axis=2)/np.maximum(count[:,:,None],1)
    return mean.reshape(n,-1),count.astype('f4')/10,middle.reshape(n,5,10).mean(axis=2).astype('f4')

def features(text,mask,audio,vision,text_only=False):
    t=text.copy();t[~mask]=0
    pieces=[t.reshape(len(t),-1),mask.astype('f4')]
    if not text_only:
        for x in (audio,vision):pieces.extend(pool(x))
    result=np.concatenate(pieces,axis=1).astype('f4')
    assert np.isfinite(result).all()
    return result

def gap(seq,where=.5,fraction=.25,rng=None):
    out=seq.copy();active=np.any(seq!=0,axis=2);changed=0
    for i in range(len(seq)):
        index=np.flatnonzero(active[i])
        if not len(index):continue
        left,right=int(index[0]),int(index[-1])+1
        portion=float(rng.uniform(.15,.45)) if rng is not None else fraction
        width=min(right-left,max(1,round((right-left)*portion)))
        start=int(rng.integers(left,right-width+1)) if rng is not None else left+round((right-left-width)*where)
        changed+=int(active[i,start:start+width].sum())
        out[i,start:start+width]=0
    return out,changed

def text_gap(text,mask,where=.5,rng=None):
    out=text.copy();valid=mask.copy();changed=0
    for i in range(len(mask)):
        index=np.flatnonzero(valid[i])
        if not len(index):continue
        j=int(rng.choice(index)) if rng is not None else int(index[round((len(index)-1)*where)])
        valid[i,j]=False;out[i,j]=0;changed+=1
    return out,valid,changed

def fit(x,cls,reg):
    c=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,class_weight='balanced',random_state=SEED))
    r=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
    c.fit(x,cls);r.fit(x,reg)
    return c,r

def evaluate(models,x,cls,reg):
    c,r=models;pred=c.predict(x);prob=c.predict_proba(x);strength=r.predict(x)
    return {'accuracy':float(accuracy_score(cls,pred)),'f1_macro':float(f1_score(cls,pred,average='macro')),
            'auc_macro_ovr':float(roc_auc_score(cls,prob,multi_class='ovr',average='macro')),
            'mae':float(mean_absolute_error(reg,strength)),
            'pearson':float(np.corrcoef(reg,strength)[0,1]) if np.std(strength)>0 else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--embeddings',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with a.data.open('rb') as f:data=pickle.load(f)
    train,valid=data['train'],data['valid']
    for block in (train,valid):block.pop('text',None);block.pop('text_bert',None)
    emb={}
    for name,block in (('train',train),('valid',valid)):
        with np.load(a.embeddings/(name+'.npz')) as z:
            if not np.array_equal(np.asarray(block['id'],dtype=str),z['sample_id']):raise ValueError(name+' embedding ID/order mismatch')
            emb[name]={k:z[k].copy() for k in ('segments','segment_present')}
    (tt,tm),(vt,vm)=project(emb['train'],emb['valid'])
    ta,tv=np.asarray(train['audio'],'f4'),np.asarray(train['vision'],'f4')
    va,vv=np.asarray(valid['audio'],'f4'),np.asarray(valid['vision'],'f4')
    yc=np.asarray(train['classification_labels']).reshape(-1).astype(int)
    yr=np.asarray(train['regression_labels']).reshape(-1).astype('f4')
    vc=np.asarray(valid['classification_labels']).reshape(-1).astype(int)
    vr=np.asarray(valid['regression_labels']).reshape(-1).astype('f4')
    text_model=fit(features(tt,tm,ta,tv,True),yc,yr)
    clean=features(tt,tm,ta,tv);plain=fit(clean,yc,yr)
    rng=np.random.default_rng(SEED);views=[clean];injected={}
    aa,n=gap(ta,rng=rng);views.append(features(tt,tm,aa,tv));injected['audio']=n
    av,n=gap(tv,rng=rng);views.append(features(tt,tm,ta,av));injected['vision']=n
    tx,mx,n=text_gap(tt,tm,rng=rng);views.append(features(tx,mx,ta,tv));injected['text']=n
    robust=fit(np.concatenate(views),np.tile(yc,len(views)),np.tile(yr,len(views)))
    conditions={'clean':features(vt,vm,va,vv)};affected={'clean':0}
    for modality in ('audio','vision'):
        for label,pos in (('start',0.),('middle',.5),('end',1.)):
            if modality=='audio':x,n=gap(va,where=pos);matrix=features(vt,vm,x,vv)
            else:x,n=gap(vv,where=pos);matrix=features(vt,vm,va,x)
            key=f'{modality}_25pct_{label}';conditions[key]=matrix;affected[key]=n
    for label,pos in (('start',0.),('middle',.5),('end',1.)):
        x,m,n=text_gap(vt,vm,where=pos);key=f'text_1segment_{label}';conditions[key]=features(x,m,va,vv);affected[key]=n
    results={'text_only':{'clean':evaluate(text_model,features(vt,vm,va,vv,True),vc,vr)},'clean_fusion':{},'augmented_fusion':{}}
    for key,x in conditions.items():
        results['clean_fusion'][key]=evaluate(plain,x,vc,vr)
        results['augmented_fusion'][key]=evaluate(robust,x,vc,vr)
    report={'seed':SEED,'train':len(yc),'valid':len(vc),'class_counts':np.bincount(yc,minlength=3).tolist(),
            'protocol':'A2 train fits all transforms/models; A2 valid evaluates; A2 test untouched',
            'text':'same frozen BERT encoder for A2/A3, 5 segments, train-only PCA 48',
            'fusion':'5 audio/vision means plus occupancy and interior-zero fractions',
            'augmentation':'clean + one random contiguous gap per modality per training sample',
            'injected_training_positions':injected,'affected_validation_positions':affected,
            'naive_valid':{'majority_accuracy':float(np.mean(vc==np.bincount(yc).argmax())),
                           'train_mean_mae':float(np.mean(np.abs(vr-yr.mean())))},'metrics':results}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'naive_valid':report['naive_valid'],'text':results['text_only']['clean'],
                      'plain':results['clean_fusion']['clean'],'robust':results['augmented_fusion']['clean'],
                      'plain_audio_middle':results['clean_fusion']['audio_25pct_middle'],
                      'robust_audio_middle':results['augmented_fusion']['audio_25pct_middle']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
