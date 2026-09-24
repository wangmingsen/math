"""Paired A2-valid comparison of the old and new frozen Q2 candidates."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import f1_score,mean_absolute_error,roc_auc_score
from q3_explain import predict as predict_old


SEEDS=(2026,2027,2028)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--old-model',type=Path,required=True)
    p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with next(a.data_root.resolve().glob('附件2-*/aligned_50.pkl')).open('rb') as stream:
        block=pickle.load(stream)['valid']
    ids=np.asarray(block['id'],dtype=str)
    y=np.asarray(block['classification_labels']).reshape(-1).astype(int)
    r=np.asarray(block['regression_labels']).reshape(-1).astype('f4')
    model=joblib.load(a.old_model)
    with np.load(a.embeddings/'valid.npz') as z:
        if not np.array_equal(ids,z['sample_id'].astype(str)):raise ValueError('Old valid IDs differ')
        mask=z['segment_present'].astype(bool)
        text=model['text_pca'].transform(z['segments'].reshape(-1,768)).reshape(len(ids),5,48).astype('f4')
        text[~mask]=0
    p_old,r_old=predict_old(model,text,mask,np.asarray(block['audio'],'f4'),np.asarray(block['vision'],'f4'))
    selected=json.loads(a.selection.read_text(encoding='utf8'))
    runs=[np.load(a.runs/f'q2_transformer_seed_{seed}'/'valid_predictions.npz') for seed in SEEDS]
    if any(not np.array_equal(ids,z['sample_id'].astype(str)) for z in runs):
        raise ValueError('New valid IDs differ')
    def stages(name):
        return ('teacher','student') if name=='teacher_student_mean6' else (
            ('teacher',) if name=='teacher_mean3' else ('student',))
    p_new=np.mean([z[f'{stage}_clean_probability'] for z in runs
                   for stage in stages(selected['classification_component'])],axis=0)
    r_new=np.mean([z[f'{stage}_clean_intensity'] for z in runs
                   for stage in stages(selected['regression_component'])],axis=0)
    old_label=p_old.argmax(axis=1);new_label=p_new.argmax(axis=1)
    def changes(index):
        yf=y[index];rf=r[index]
        delta_f1=f1_score(yf,new_label[index],average='macro')-f1_score(yf,old_label[index],average='macro')
        delta_mae=mean_absolute_error(rf,r_new[index])-mean_absolute_error(rf,r_old[index])
        return delta_f1,delta_mae
    point=changes(np.arange(len(y)))
    rng=np.random.default_rng(20260924)
    trials=np.asarray([changes(rng.integers(0,len(y),len(y))) for _ in range(2000)])
    ci=np.quantile(trials,[.025,.975],axis=0)
    report={'protocol':'Same A2 valid IDs; frozen old/new candidates; 2000 paired bootstrap resamples, seed 20260924',
            'samples':len(ids),
            'old':{'f1_macro':float(f1_score(y,old_label,average='macro')),
                   'auc_macro_ovr':float(roc_auc_score(y,p_old,multi_class='ovr',average='macro')),
                   'mae':float(mean_absolute_error(r,r_old))},
            'new':{'f1_macro':float(f1_score(y,new_label,average='macro')),
                   'auc_macro_ovr':float(roc_auc_score(y,p_new,multi_class='ovr',average='macro')),
                   'mae':float(mean_absolute_error(r,r_new))},
            'delta_new_minus_old':{'f1_macro':float(point[0]),'f1_95pct_percentile_ci':ci[:,0].tolist(),
                                   'mae':float(point[1]),'mae_95pct_percentile_ci':ci[:,1].tolist()},
            'limitation':'Valid data selected the new model; bootstrap intervals describe sampling variability, not an untouched test result.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
