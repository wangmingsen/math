"""Estimate A4 aligned-token seconds from original-video ASR word anchors.

The PKL has no timestamps. Accepted mappings are estimates, never ground truth.
Rejected mappings keep exact text offsets but no seconds.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from tokenizers import Tokenizer
from aligned_data import load_small
from q1_extract import TOKEN_PATTERN
from q1_refine_alignment import align


def word_spans(text):
    return [{'index':i,'token':m.group(),'char_start':m.start(),'char_end':m.end(),
             'start_s':None,'end_s':None,'timing':'unmapped'}
            for i,m in enumerate(TOKEN_PATTERN.finditer(text))]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--model-cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',default='base.en')
    a=p.parse_args()
    rows=list(csv.DictReader(a.manifest.open(encoding='utf-8-sig',newline='')))
    tok=Tokenizer.from_file(str(a.tokenizer.resolve()))
    whisper=WhisperModel(a.model,device='cpu',compute_type='int8',
                         download_root=str(a.model_cache.resolve()))
    a.out.mkdir(parents=True,exist_ok=True)
    report=[]
    for n,row in enumerate(rows,1):
        sid=row['sample_id']
        try:
            raw=load_small(Path(row['feature_path']))
            original=str(raw['raw_text'])
            words=word_spans(original)
            video=Path(row['video_path'])
            duration=float(row['duration_s'])
            segments,_=whisper.transcribe(str(video),language='en',beam_size=5,
                                          word_timestamps=True,condition_on_previous_text=False)
            heard=[{'word':w.word,'start_s':w.start,'end_s':w.end}
                   for segment in segments for w in (segment.words or [])
                   if w.start is not None and w.end is not None]
            aligned,quality=align(words,heard,duration,min_match=.6)
            encoded=tok.encode(original)
            stored=np.asarray(raw['text_bert'])[0]
            attention=np.asarray(raw['text_bert'])[1]!=0
            observed=stored[attention].astype(int).tolist()
            if observed[:-1]!=encoded.ids[:len(observed)-1] or observed[-1]!=102:
                raise ValueError('Stored tokens differ from transcript prefix')
            positions=[]
            for pos in range(50):
                if pos==0 or pos>=len(observed)-1:
                    positions.append({'position':pos,'text':'','char_start':None,
                                      'char_end':None,'word_index':None,
                                      'start_s':None,'end_s':None,'timing':'special_or_padding'})
                    continue
                start,end=encoded.offsets[pos]
                take=[w for w in aligned if w['char_start']<end and w['char_end']>start]
                if not take:raise ValueError(f'No source word for token position {pos}')
                first,last=take[0],take[-1]
                positions.append({'position':pos,'text':original[start:end],
                                  'char_start':start,'char_end':end,
                                  'word_index':first['index'],
                                  'start_s':first['start_s'] if quality['accepted'] else None,
                                  'end_s':last['end_s'] if quality['accepted'] else None,
                                  'timing':'asr_anchor_midpoint_estimate' if quality['accepted'] else 'unverified'})
            mapping={'sample_id':sid,'video_path':str(video),
                     'duration_s':duration,'raw_text':original,
                     'text_truncated_at_50':len(encoded.ids)>50,
                     'alignment_quality':quality,'words':aligned,'positions':positions,
                     'time_warning':'ASR-based estimate, not a supplied timestamp or human-verified word boundary'}
            (a.out/(sid+'.json')).write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
            item={'sample_id':sid,'status':'ok','asr_anchor_accepted':quality['accepted'],
                  'exact_match_fraction':quality['exact_match_fraction'],
                  'mapped_lexical_positions':sum(x['start_s'] is not None for x in positions),
                  'text_truncated_at_50':len(encoded.ids)>50}
        except Exception as exc:
            item={'sample_id':sid,'status':'error','error':str(exc)}
        report.append(item)
        print(f"{n}/{len(rows)} {sid} {item['status']} {item.get('exact_match_fraction','')}",flush=True)
    fields=['sample_id','status','asr_anchor_accepted','exact_match_fraction',
            'mapped_lexical_positions','text_truncated_at_50','error']
    with (a.out/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(report)
    summary={'samples':len(report),'successful':sum(x['status']=='ok' for x in report),
             'asr_anchor_accepted':sum(x.get('asr_anchor_accepted',False) for x in report),
             'unverified_time':sum(x.get('status')=='ok' and not x.get('asr_anchor_accepted') for x in report),
             'truncated_text':sum(x.get('text_truncated_at_50',False) for x in report),
             'timing':'ASR word-center midpoint estimate; no PKL timestamps'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['successful']!=len(report):raise SystemExit(1)

if __name__=='__main__':main()
