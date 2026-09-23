"""Q2 conservative late fusion with train-only audio/vision gap augmentation.

Shared BERT embeddings for A2 and A3 are built by q2_text_embeddings.py.
Weights were chosen during valid-set development and fixed before one-time test.
"""
from __future__ import annotations
import argparse,csv,json,pickle
from pathlib import Path
import joblib,numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,roc_auc_score,classification_report
from aligned_data import load_small,normalize
from q2_missing_baseline import features,gap,text_gap,fit

SEED=2026
CLASS_WEIGHT=.1
REG_WEIGHT=.2

def metric(yc,yr,pred,score):
    label=pred.argmax(axis=1)
    return {'accuracy':float(accuracy_score(yc,label)),
            'f1_macro':float(f1_score(yc,label,average='macro')),
            'auc_macro_ovr':float(roc_auc_score(yc,pred,multi_class='ovr',average='macro')),
            'mae':float(mean_absolute_error(yr,score)),
            'pearson':float(np.corrcoef(yr,score)[0,1]),
            'f1_by_class':{str(i):float(f1_score(yc,label,labels=[i],average='macro',zero_division=0)) for i in range(3)}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();root=a.data_root.resolve();a.out_dir.mkdir(parents=True,exist_ok=True)
    with next(root.glob('附件2-*/aligned_50.pkl')).open('rb') as f:data=pickle.load(f)
    blocks={k:data[k] for k in ('train','valid','test')}
    for b in blocks.values():b.pop('text',None);b.pop('text_bert',None)
    embed={}
    for k in ('train','valid','test','A3'):
        with np.load(a.embeddings/(k+'.npz')) as z:
            embed[k]={x:z[x].copy() for x in ('sample_id','segments','segment_present')}
    for k in ('train','valid','test'):
        if not np.array_equal(np.asarray(blocks[k]['id'],dtype=str),embed[k]['sample_id']):
            raise ValueError(k+' ID mismatch')
    pca=PCA(n_components=48,svd_solver='randomized',random_state=SEED)
    pca.fit(embed['train']['segments'][embed['train']['segment_present']])
    def transform(k):
        z=embed[k];m=z['segment_present'].astype(bool)
        t=pca.transform(z['segments'].reshape(-1,768)).reshape(len(m),5,48).astype('f4')
        t[~m]=0
        return t,m
    train_text,train_mask=transform('train');valid_text,valid_mask=transform('valid');test_text,test_mask=transform('test');a3_text,a3_mask=transform('A3')
    tr,va=blocks['train'],blocks['valid']
    ta,tv=np.asarray(tr['audio'],'f4'),np.asarray(tr['vision'],'f4')
    vaa,vav=np.asarray(va['audio'],'f4'),np.asarray(va['vision'],'f4')
    yc=np.asarray(tr['classification_labels']).reshape(-1).astype(int)
    yr=np.asarray(tr['regression_labels']).reshape(-1).astype('f4')
    vc=np.asarray(va['classification_labels']).reshape(-1).astype(int)
    vr=np.asarray(va['regression_labels']).reshape(-1).astype('f4')
    text_x=features(train_text,train_mask,ta,tv,True)
    text_rng=np.random.default_rng(SEED)
    text_masked,text_masked_present,_=text_gap(train_text,train_mask,rng=text_rng)
    text_augmented=features(text_masked,text_masked_present,ta,tv,True)
    text_model=fit(np.concatenate([text_x,text_augmented]),np.tile(yc,2),np.tile(yr,2))
    text_dim=text_x.shape[1]
    rng=np.random.default_rng(SEED)
    views=[features(train_text,train_mask,ta,tv)[:,text_dim:]]
    audio_gap,n_audio=gap(ta,rng=rng);views.append(features(train_text,train_mask,audio_gap,tv)[:,text_dim:])
    vision_gap,n_vision=gap(tv,rng=rng);views.append(features(train_text,train_mask,ta,vision_gap)[:,text_dim:])
    av_x=np.concatenate(views)
    av_classifier=make_pipeline(StandardScaler(),LogisticRegression(C=.001,max_iter=500,class_weight='balanced',random_state=SEED))
    av_regressor=make_pipeline(StandardScaler(),Ridge(alpha=1000.0))
    av_classifier.fit(av_x,np.tile(yc,3));av_regressor.fit(av_x,np.tile(yr,3))
    def predict(t,m,a,v):
        text=features(t,m,a,v,True)
        av=features(t,m,a,v)[:,text.shape[1]:]
        p=(1-CLASS_WEIGHT)*text_model[0].predict_proba(text)+CLASS_WEIGHT*av_classifier.predict_proba(av)
        r=(1-REG_WEIGHT)*text_model[1].predict(text)+REG_WEIGHT*av_regressor.predict(av)
        return p,r
    cases={'clean':(valid_text,valid_mask,vaa,vav)};affected={'clean':0}
    for modality in ('audio','vision'):
        for fraction in (.10,.25,.50):
            for label,pos in (('start',0.),('middle',.5),('end',1.)):
                if modality=='audio': altered,n=gap(vaa,where=pos,fraction=fraction);case=(valid_text,valid_mask,altered,vav)
                else: altered,n=gap(vav,where=pos,fraction=fraction);case=(valid_text,valid_mask,vaa,altered)
                name=f'{modality}_{int(fraction*100)}pct_{label}';cases[name]=case;affected[name]=n
    for label,pos in (('start',0.),('middle',.5),('end',1.)):
        t,m,n=text_gap(valid_text,valid_mask,where=pos)
        name=f'text_1segment_{label}';cases[name]=(t,m,vaa,vav);affected[name]=n
    results={}
    for name,args in cases.items():results[name]=metric(vc,vr,*predict(*args))
    te=blocks['test']
    test_audio,test_vision=np.asarray(te['audio'],'f4'),np.asarray(te['vision'],'f4')
    test_class=np.asarray(te['classification_labels']).reshape(-1).astype(int)
    test_reg=np.asarray(te['regression_labels']).reshape(-1).astype('f4')
    test_metrics=metric(test_class,test_reg,*predict(test_text,test_mask,test_audio,test_vision))
    paths=sorted(next(root.glob('附件3-*')).rglob('对齐版本/*.pkl'))
    ids=[];audio=[];vision=[]
    for path in paths:
        item=load_small(path)
        if set(item)!={'test'}:raise ValueError('Invalid A3 wrapper')
        sample=normalize(item['test'],path.stem,'A3')
        ids.append(path.stem);audio.append(sample['audio']);vision.append(sample['vision'])
    if not np.array_equal(np.asarray(ids,dtype=str),embed['A3']['sample_id']):
        raise ValueError('A3 ID/order mismatch')
    probabilities,intensity=predict(a3_text,a3_mask,np.asarray(audio,'f4'),np.asarray(vision,'f4'))
    if len(ids)!=30 or len(set(ids))!=30:
        raise ValueError('A3 must have 30 unique sample IDs')
    known={str(item) for split in ('train','valid','test') for item in blocks[split]['id']}
    if set(ids)&known:
        raise ValueError('A3 sample ID overlaps A2')
    if (not np.isfinite(probabilities).all() or not np.isfinite(intensity).all()
            or not np.allclose(probabilities.sum(axis=1),1,atol=1e-6)
            or np.any((intensity < -3) | (intensity > 3))):
        raise ValueError('A3 probability or intensity output is invalid')
    rows=[]
    for i,sid in enumerate(ids):
        rows.append({'sample_id':sid,'predicted_polarity':int(probabilities[i].argmax()),
                     'predicted_intensity':float(intensity[i]),
                     'prob_negative':float(probabilities[i,0]),
                     'prob_neutral':float(probabilities[i,1]),
                     'prob_positive':float(probabilities[i,2])})
    with (a.out_dir/'A3_predictions_candidate.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    bundle={'text_model':text_model,'av_classifier':av_classifier,'av_regressor':av_regressor,'text_pca':pca,
            'class_weight':CLASS_WEIGHT,'reg_weight':REG_WEIGHT,'version':'q2_late_fusion_v1'}
    joblib.dump(bundle,a.out_dir/'model_candidate.joblib',compress=3)
    report={'train':len(yc),'valid':len(vc),'A3':len(ids),'seed':SEED,
            'class_weights':{'text':1-CLASS_WEIGHT,'audio_vision':CLASS_WEIGHT},
            'regression_weights':{'text':1-REG_WEIGHT,'audio_vision':REG_WEIGHT},
            'training':'A2 train only; text frozen BERT+PCA with one segment dropout copy; audio/vision clean and random gap augmented',
            'validation_scenario_affected_positions':affected,'validation':results,
            'A2_test_once':test_metrics,
            'A3_predictions':'candidate, no A3 labels or A2 test used',
            'limitations':'A3 gap mechanism may differ from synthetic gaps; text-gap robustness is not trained'}
    (a.out_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'train':len(yc),'valid':len(vc),'A3':len(ids),
                      'clean':results['clean'],'A2_test_once':test_metrics,'audio_middle':results['audio_25pct_middle'],
                      'vision_middle':results['vision_25pct_middle'],'text_middle':results['text_1segment_middle']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()


