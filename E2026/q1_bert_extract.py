"""Extract last-layer contextual BERT vectors for Q1 transcript words and bins."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from tokenizers import Tokenizer

HIDDEN_NODE = '/bert/encoder/layer.11/output/LayerNorm/Add_1_output_0'


def session_with_hidden_output(model_path: Path):
    model = onnx.load(str(model_path), load_external_data=False)
    if HIDDEN_NODE not in {o for node in model.graph.node for o in node.output}:
        raise ValueError('BERT final hidden node not found in this ONNX model')
    output = onnx.helper.make_tensor_value_info(HIDDEN_NODE, onnx.TensorProto.FLOAT,
                                                ['batch_size', 'sequence_length', 768])
    model.graph.output.append(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options,
                                providers=['CPUExecutionProvider'])


def word_vectors(words, tokenizer, session):
    if not words:
        return np.zeros((0,768),np.float32), 0
    sequence = [w['token'] for w in words]
    encoding = tokenizer.encode(sequence, is_pretokenized=True)
    if len(encoding.ids) > 512:
        raise ValueError(f'BERT input has {len(encoding.ids)} tokens, exceeds 512')
    inputs = {k: np.asarray([v], dtype=np.int64) for k,v in {
        'input_ids': encoding.ids,
        'attention_mask': encoding.attention_mask,
        'token_type_ids': encoding.type_ids,
    }.items()}
    hidden = session.run([HIDDEN_NODE], inputs)[0][0]
    if hidden.shape != (len(encoding.ids),768):
        raise ValueError(f'Unexpected BERT hidden shape: {hidden.shape}')
    vectors = np.zeros((len(words),768),np.float32)
    counts = np.zeros(len(words),np.int32)
    for i, wid in enumerate(encoding.word_ids):
        if wid is not None:
            vectors[wid] += hidden[i]
            counts[wid] += 1
    if not np.all(counts > 0):
        raise ValueError('At least one word has no WordPiece token')
    vectors /= counts[:,None]
    return vectors, len(encoding.ids)


def aggregate(words, vectors, edges):
    feature = np.zeros((50,768),np.float32)
    counts = np.zeros(50,np.int32)
    for word, vector in zip(words,vectors):
        midpoint = (word['start_s']+word['end_s'])/2
        b = int(np.clip(np.searchsorted(edges,midpoint,side='right')-1,0,49))
        feature[b] += vector
        counts[b] += 1
    used = counts > 0
    feature[used] /= counts[used,None]
    return feature,counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--q1-features',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    source=a.q1_features.resolve()
    rows=list(csv.DictReader((source/'manifest.csv').open(encoding='utf-8-sig',newline='')))
    if a.limit: rows=rows[:a.limit]
    a.out.mkdir(parents=True,exist_ok=True)
    tokenizer=Tokenizer.from_file(str(a.tokenizer.resolve()))
    session=session_with_hidden_output(a.model.resolve())
    report=[]
    for n,row in enumerate(rows,1):
        try:
            mapping=json.loads((source/row['mapping_file']).read_text(encoding='utf-8'))
            words=mapping['words']
            with np.load(source/row['feature_file']) as old:
                edges=old['time_edges_s'].astype(np.float64)
            vectors,pieces=word_vectors(words,tokenizer,session)
            feature,count=aggregate(words,vectors,edges)
            if feature.shape!=(50,768) or not np.isfinite(feature).all():
                raise ValueError('Invalid BERT vectors')
            np.savez_compressed(a.out/row['feature_file'],text_bert=feature,
                                text_present=count>0,word_count=count,
                                time_edges_s=edges.astype(np.float32))
            item={'sample_id':row['sample_id'],'feature_file':row['feature_file'],
                  'status':'ok','words':len(words),'wordpieces':pieces,
                  'text_bins':int((count>0).sum())}
        except Exception as exc:
            item={'sample_id':row['sample_id'],'feature_file':row['feature_file'],
                  'status':'error','error':str(exc)}
        report.append(item)
        print(f"{n}/{len(rows)} {item['sample_id']} {item['status']}",flush=True)
    fields=['sample_id','feature_file','status','words','wordpieces','text_bins','error']
    with (a.out/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(report)
    summary={'requested':len(rows),'completed':sum(x['status']=='ok' for x in report),
             'failed':sum(x['status']!='ok' for x in report),
             'model':'bert-base-uncased, quantized ONNX masked LM encoder',
             'layer':'last encoder hidden state',
             'output_shape':[50,768],
             'text_timing':'Q1 estimated transcript word intervals',
             'wordpiece_pooling':'mean within word',
             'bin_pooling':'mean of word vectors by interval midpoint'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['failed']:raise SystemExit(1)


if __name__=='__main__':main()
