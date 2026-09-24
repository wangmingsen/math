"""Build and verify the Q1 all-100 paper table from the final feature archive."""
from __future__ import annotations
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
import numpy as np

FIELDS=['sample_id','duration_s','text_dimension','audio_dimension','vision_dimension',
        'aligned_windows','window_seconds','text_valid_windows','audio_signal_windows',
        'face_valid_windows','asr_anchor_accepted','feature_file','mapping_file']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--markdown',type=Path,required=True)
    a=p.parse_args()
    with zipfile.ZipFile(a.archive) as z:
        manifest=list(csv.DictReader(io.StringIO(z.read('manifest.csv').decode('utf-8-sig'))))
        if len(manifest)!=100 or len({r['sample_id'] for r in manifest})!=100:
            raise ValueError('Expected 100 unique IDs')
        rows=[]
        for item in manifest:
            feat=item['feature_file'];mapping=item['mapping_file']
            if feat not in z.namelist() or mapping not in z.namelist():
                raise ValueError('Missing feature or mapping')
            with np.load(io.BytesIO(z.read(feat))) as values:
                shapes=[values[key].shape for key in ('text_bert','audio_opensmile','vision_openface')]
                if shapes!=[(50,768),(50,50),(50,35)]:raise ValueError((feat,shapes))
                edges=values['time_edges_s']
                if edges.shape!=(51,) or np.any(np.diff(edges)<=0):raise ValueError('Bad time edges')
                duration=float(item['duration_s'])
                if not np.isclose(edges[0],0,atol=1e-6) or not np.isclose(edges[-1],duration,atol=1e-4):
                    raise ValueError('Time duration mismatch')
                tc=int(values['text_present'].sum())
                ac=int(values['audio_signal_present'].sum())
                vc=int(values['face_valid'].sum())
                aa=bool(values['asr_anchor_accepted'])
                if (tc,ac,vc,aa)!=(int(item['text_bins']),int(item['audio_signal_bins']),
                                  int(item['face_bins']),item['asr_anchor_accepted']=='True'):
                    raise ValueError('Manifest and tensor masks disagree')
            record=json.loads(z.read(mapping))
            if record['sample_id']!=item['sample_id']:raise ValueError('Mapping ID mismatch')
            rows.append({'sample_id':item['sample_id'],'duration_s':f'{duration:.3f}',
                         'text_dimension':'50×768','audio_dimension':'50×50',
                         'vision_dimension':'50×35','aligned_windows':50,
                         'window_seconds':f'{duration/50:.3f}',
                         'text_valid_windows':tc,'audio_signal_windows':ac,
                         'face_valid_windows':vc,'asr_anchor_accepted':aa,
                         'feature_file':feat,'mapping_file':mapping})
    a.csv.parent.mkdir(parents=True,exist_ok=True)
    for path in (a.csv,a.markdown):path.parent.mkdir(parents=True,exist_ok=True)
    with a.csv.open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    cols=['sample_id','duration_s','text_dimension','audio_dimension','vision_dimension',
          'window_seconds','text_valid_windows','audio_signal_windows',
          'face_valid_windows','asr_anchor_accepted']
    lines=['# 问题一：附件1全部100条样本特征汇总','',
           '每条均按原视频时长划分50个等时窗口。窗宽=时长/50；文本/声学/视觉维度分别为768/50/35。'
           '有效窗仅表示有题给文本词、非零音轨信号或成功人脸跟踪，不代表情感信息已验证。'
           'ASR锚点通过仍是估计词时间；不通过则使用回退定位。','',
           '| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
    for row in rows:lines.append('| '+' | '.join(str(row[c]) for c in cols)+' |')
    a.markdown.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'samples':len(rows),'text_valid_total':sum(r['text_valid_windows'] for r in rows),
                      'audio_signal_total':sum(r['audio_signal_windows'] for r in rows),
                      'face_valid_total':sum(r['face_valid_windows'] for r in rows),
                      'asr_accepted':sum(r['asr_anchor_accepted'] for r in rows),
                      'markdown':str(a.markdown.resolve())},ensure_ascii=False))

if __name__=='__main__':main()
