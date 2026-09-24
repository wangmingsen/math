"""Prepare A2 train/valid inputs for a small fusion Transformer experiment.

Uses the train-fitted PCA in the frozen Q2 bundle; never opens A2 test here.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import joblib
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    bundle=joblib.load(a.model)
    if bundle.get('version')!='q2_late_fusion_v1':raise ValueError('Unexpected PCA source')
    with next(a.data_root.resolve().glob('附件2-*/aligned_50.pkl')).open('rb') as f:
        data=pickle.load(f)
    a.out.mkdir(parents=True,exist_ok=True)
    summary={}
    for split in ('train','valid'):
        block=data[split]
        ids=np.asarray(block['id'],dtype=str)
        with np.load(a.embeddings/(split+'.npz')) as z:
            if not np.array_equal(ids,z['sample_id'].astype(str)):
                raise ValueError(split+' ID/order mismatch')
            segments=z['segments'].astype('f4')
            mask=z['segment_present'].astype(bool)
        text=bundle['text_pca'].transform(segments.reshape(-1,768)).reshape(len(ids),5,48).astype('f4')
        text[~mask]=0
        audio=np.asarray(block['audio'],'f4')
        vision=np.asarray(block['vision'],'f4')
        cls=np.asarray(block['classification_labels']).reshape(-1).astype('i8')
        reg=np.asarray(block['regression_labels']).reshape(-1).astype('f4')
        assert text.shape==(len(ids),5,48) and audio.shape==(len(ids),50,74) and vision.shape==(len(ids),50,35)
        np.savez_compressed(a.out/(split+'.npz'),sample_id=ids,text=text,text_present=mask,
                            audio=audio,vision=vision,classification_labels=cls,regression_labels=reg)
        summary[split]={'samples':len(ids),'classes':np.bincount(cls,minlength=3).tolist()}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
