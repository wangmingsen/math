"""Prepare the 30 unlabeled A3 aligned samples through the Q2 training path."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from aligned_data import load_small,normalize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    bundle=joblib.load(a.model)
    if bundle.get('version')!='q2_late_fusion_v1':raise ValueError('Unexpected PCA source')
    paths=sorted(next(a.data_root.resolve().glob('附件3-*')).rglob('对齐版本/*.pkl'))
    if len(paths)!=30:raise ValueError('Expected 30 A3 aligned samples')
    ids=[];audio=[];vision=[]
    for path in paths:
        item=load_small(path)
        if set(item)!={'test'}:raise ValueError('Unexpected A3 wrapper')
        sample=normalize(item['test'],path.stem,'A3')
        ids.append(path.stem);audio.append(sample['audio']);vision.append(sample['vision'])
    ids=np.asarray(ids,dtype=str)
    with np.load(a.embeddings/'A3.npz') as z:
        if not np.array_equal(ids,z['sample_id'].astype(str)):
            raise ValueError('A3 ID/order mismatch')
        segments=z['segments'].astype('f4')
        mask=z['segment_present'].astype(bool)
    text=bundle['text_pca'].transform(segments.reshape(-1,768)).reshape(len(ids),5,48).astype('f4')
    text[~mask]=0
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,sample_id=ids,text=text,text_present=mask,
                        audio=np.asarray(audio,'f4'),vision=np.asarray(vision,'f4'))
    print(json.dumps({'samples':len(ids),'labels':'not supplied','representation':'aligned_50'},ensure_ascii=False))


if __name__=='__main__':main()
