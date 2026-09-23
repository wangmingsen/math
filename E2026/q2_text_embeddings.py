"""Encode A2 and A3 text_bert tokens with one frozen BERT ONNX encoder.

Run on trusted contest pickle files only. No A2 test labels are used.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from aligned_data import load_small
from q1_bert_extract import HIDDEN_NODE, session_with_hidden_output

SPECIAL = (0, 101, 102)
SEGMENTS = 5


def encode_batch(tokens: np.ndarray, session):
    x = np.asarray(tokens, dtype=np.int64)
    if x.ndim != 3 or x.shape[1:] != (3,50):
        raise ValueError(f'Expected [batch,3,50], got {x.shape}')
    hidden = session.run([HIDDEN_NODE], {
        'input_ids': x[:,0,:], 'attention_mask': x[:,1,:],
        'token_type_ids': x[:,2,:]})[0].astype(np.float32)
    valid = (x[:,1,:] != 0) & ~np.isin(x[:,0,:], SPECIAL)
    if not np.isfinite(hidden).all():
        raise ValueError('Nonfinite BERT encoder output')
    whole_count = valid.sum(axis=1)
    if np.any(whole_count == 0):
        raise ValueError('A sample contains no lexical BERT token')
    sentence = (hidden * valid[:,:,None]).sum(axis=1) / whole_count[:,None]
    segments = np.zeros((len(x),SEGMENTS,768),np.float32)
    occupied = np.zeros((len(x),SEGMENTS),bool)
    for j in range(SEGMENTS):
        take = valid[:,j*10:(j+1)*10]
        count = take.sum(axis=1)
        occupied[:,j] = count > 0
        part = hidden[:,j*10:(j+1)*10]
        segments[:,j] = (part*take[:,:,None]).sum(axis=1)/np.maximum(count[:,None],1)
    return sentence,segments,occupied


def encode_array(tokens, session, batch_size):
    sentences=[];segments=[];occupied=[]
    for start in range(0,len(tokens),batch_size):
        a,b,c=encode_batch(np.asarray(tokens[start:start+batch_size]),session)
        sentences.append(a);segments.append(b);occupied.append(c)
        print(f'{min(start+batch_size,len(tokens))}/{len(tokens)}',flush=True)
    return np.concatenate(sentences),np.concatenate(segments),np.concatenate(occupied)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args()
    root=a.data_root.resolve()
    source=next(root.glob('附件2-*/aligned_50.pkl'))
    session=session_with_hidden_output(a.model.resolve())
    with source.open('rb') as f:data=pickle.load(f)
    a.out.mkdir(parents=True,exist_ok=True)
    summary={'model':'bert-base-uncased ONNX int8, last encoder layer',
             'pooling':'mean lexical token vectors, whole sentence and five 10-position segments',
             'splits':{}}
    for split in ('train','valid','test'):
        block=data[split]
        sentence,segments,occupied=encode_array(block['text_bert'],session,a.batch_size)
        ids=np.asarray(block['id'],dtype=str)
        np.savez_compressed(a.out/(split+'.npz'),sample_id=ids,
                            sentence=sentence,segments=segments,segment_present=occupied)
        summary['splits'][split]={'samples':len(ids),'segment_present':int(occupied.sum())}
    del data
    paths=sorted(next(root.glob('附件3-*')).rglob('对齐版本/*.pkl'))
    ids=[];tokens=[]
    for path in paths:
        raw=load_small(path)
        if set(raw)!={'test'}:raise ValueError(f'Unexpected A3 wrapper {path}')
        ids.append(path.stem)
        value=np.asarray(raw['test']['text_bert'])
        if value.shape==(1,3,50):value=value[0]
        tokens.append(value)
    sentence,segments,occupied=encode_array(np.asarray(tokens),session,a.batch_size)
    np.savez_compressed(a.out/'A3.npz',sample_id=np.asarray(ids,dtype=str),
                        sentence=sentence,segments=segments,segment_present=occupied)
    summary['splits']['A3']={'samples':len(ids),'segment_present':int(occupied.sum())}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

