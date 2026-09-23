"""Encode A4 with the frozen Q2 BERT path and audit raw-video links."""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from aligned_data import load_small, normalize
from q1_bert_extract import session_with_hidden_output
from q2_text_embeddings import encode_array


def duration(path: Path) -> float:
    cmd=['ffprobe','-v','error','-show_entries','format=duration',
         '-of','default=noprint_wrappers=1:nokey=1',str(path)]
    p=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=30)
    return float(p.stdout.strip())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    root=next(a.data_root.resolve().glob('附件4-*'))
    paths=sorted(root.rglob('对齐版本/*.pkl'))
    if len(paths)!=20:raise ValueError(f'Expected 20 A4 files, got {len(paths)}')
    tok=Tokenizer.from_file(str(a.tokenizer.resolve()))
    rows=[];tokens=[]
    for path in paths:
        raw=load_small(path)
        sid=str(raw['id'])
        if sid!=path.stem:raise ValueError(f'{path}: ID differs from filename')
        sample=normalize(raw,sid,'A4')
        video=path.parent/'videos'/(sid+'.mp4')
        if not video.is_file():raise FileNotFoundError(video)
        ids=sample['text_bert'][0].astype(np.int64)
        attention=sample['text_bert'][1]!=0
        encoded=tok.encode(str(raw['raw_text']))
        stored=ids[attention].tolist(); truncated=len(encoded.ids)>50; token_match=(stored==encoded.ids if not truncated else stored[:-1]==encoded.ids[:49] and stored[-1]==102)
        rows.append({'sample_id':sid,'feature_path':str(path),
                     'video_path':str(video),'duration_s':duration(video),
                     'text_tokens':int(attention.sum()),
                     'audio_observed_positions':int(np.any(sample['audio']!=0,axis=1).sum()),
                     'vision_observed_positions':int(np.any(sample['vision']!=0,axis=1).sum()),
                     'tokenizer_prefix_match':token_match,'text_truncated_at_50':truncated})
        tokens.append(sample['text_bert'])
    session=session_with_hidden_output(a.model.resolve())
    sentence,segments,present=encode_array(np.asarray(tokens),session,32)
    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/'A4_text.npz',sample_id=np.asarray([r['sample_id'] for r in rows]),
                        sentence=sentence,segments=segments,segment_present=present)
    with (a.out/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'samples':len(rows),'tokenizer_prefix_match':sum(x['tokenizer_prefix_match'] for x in rows),'text_truncated_at_50':sum(x['text_truncated_at_50'] for x in rows),
             'no_vision_samples':sum(x['vision_observed_positions']==0 for x in rows),
             'model':'same frozen BERT ONNX encoder and pooling as Q2'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
