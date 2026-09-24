"""Select frozen ensemble components using A2 valid only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,mean_absolute_error


SEEDS=(2026,2027,2028)


def metrics(y,r,p,s):
    label=p.argmax(axis=1)
    return {'accuracy':float(accuracy_score(y,label)),
            'f1_macro':float(f1_score(y,label,average='macro')),
            'f1_by_class':{str(c):float(f1_score(y,label,labels=[c],average='macro',zero_division=0)) for c in range(3)},
            'auc_macro_ovr':float(roc_auc_score(y,p,multi_class='ovr',average='macro')),
            'mae':float(mean_absolute_error(r,s)),
            'pearson':float(np.corrcoef(r,s)[0,1])}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True,help='Parent containing q2_transformer_seed_<seed>')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    files=[np.load(a.runs/f'q2_transformer_seed_{seed}'/'valid_predictions.npz') for seed in SEEDS]
    ids=files[0]['sample_id'];y=files[0]['classification_labels'];r=files[0]['regression_labels']
    if any(not np.array_equal(z['sample_id'],ids) for z in files[1:]):
        raise ValueError('Valid ID/order mismatch across seeds')
    candidates={}
    for mode in ('clean','dual'):
        candidates[mode]={}
        for group,stages in (('teacher_mean3',('teacher',)),
                             ('student_mean3',('student',)),
                             ('teacher_student_mean6',('teacher','student'))):
            p3=np.mean([z[f'{stage}_{mode}_probability'] for z in files for stage in stages],axis=0)
            s3=np.mean([z[f'{stage}_{mode}_intensity'] for z in files for stage in stages],axis=0)
            candidates[mode][group]=metrics(y,r,p3,s3)
    classification=max(candidates['clean'],key=lambda key:candidates['clean'][key]['f1_macro'])
    regression=min(candidates['clean'],key=lambda key:candidates['clean'][key]['mae'])
    result={'selection_data':'A2 valid only; A2 test and A3 not used',
            'seeds':list(SEEDS),'classification_component':classification,
            'regression_component':regression,
            'candidate_metrics':candidates,
            'note':'Earlier Q2 linear candidate already inspected A2 test. This new selection has no independent test claim.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'classification_component':classification,'regression_component':regression,
                      'clean_class_f1':candidates['clean'][classification]['f1_macro'],
                      'clean_reg_mae':candidates['clean'][regression]['mae']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
