"""Prepare A4 aligned inputs in the sklearn 1.4.1 environment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from aligned_data import load_small, normalize

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--text',type=Path,required=True)
    p.add_argument('--pca-bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    bundle=joblib.load(a.pca_bundle)
    if bundle.get('version')!='q2_late_fusion_v1':raise ValueError('Unexpected PCA source')
    with np.load(a.text) as z:
        ids=z['sample_id'].astype(str)
        segments=z['segments'].astype('f4')
        mask=z['segment_present'].astype(bool)
    if len(ids)!=20 or len(set(ids))!=20:raise ValueError('Expected 20 unique A4 IDs')
    text=bundle['text_pca'].transform(segments.reshape(-1,768)).reshape(20,5,48).astype('f4')
    text[~mask]=0
    paths=sorted(next(a.data_root.resolve().glob('附件4-*')).rglob('对齐版本/*.pkl'))
    if len(paths)!=20:raise ValueError(f'Expected 20 A4 pkl, found {len(paths)}')
    audio=[];vision=[]
    for i,path in enumerate(paths):
        item=load_small(path)
        if ids[i]!=path.stem or str(item['id'])!=ids[i]:raise ValueError('A4 ID/order mismatch')
        sample=normalize(item,ids[i],'A4')
        audio.append(sample['audio']);vision.append(sample['vision'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,sample_id=ids,text=text,text_present=mask,
                        audio=np.asarray(audio,'f4'),vision=np.asarray(vision,'f4'))
    print(json.dumps({'samples':len(ids),'out':str(a.out.resolve()),'text_shape':list(text.shape)},ensure_ascii=False))

if __name__=='__main__':main()
