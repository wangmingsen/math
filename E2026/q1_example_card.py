"""Generate a traceable Q1 example card from the final archive (no label inference)."""
from __future__ import annotations
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--sample-id',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with zipfile.ZipFile(a.archive) as z:
        manifest=list(csv.DictReader(io.StringIO(z.read('manifest.csv').decode('utf-8-sig'))))
        row=next((r for r in manifest if r['sample_id']==a.sample_id),None)
        if row is None:raise ValueError('sample ID absent')
        mapping=json.loads(z.read(row['mapping_file']))
        with np.load(io.BytesIO(z.read(row['feature_file']))) as values:
            text=values['text_present'].astype(bool)
            audio=values['audio_signal_present'].astype(bool)
            face=values['face_valid'].astype(bool)
            face_frames=values['face_frame_count'].astype(int)
            audio_frames=values['audio_frame_count'].astype(int)
            shape={key:values[key].shape for key in ('text_bert','audio_opensmile','vision_openface')}
        if mapping['sample_id']!=a.sample_id or not mapping['asr_anchor_accepted']:
            raise ValueError('Example requires a matching accepted-ASR record')
        candidates=[b for b in mapping['bins'] if b['word_indices'] and audio[b['index']] and face[b['index']]]
        selected=candidates[:10]
        if len(selected)<8:raise ValueError('Too few three-modality example windows')
        video=mapping['video_relative_path']
        lines=[
            '# 问题一典型样本对齐卡', '',
            f'- 样本：`{a.sample_id}`；原视频相对路径：`{video}`；时长 {mapping["duration_s"]:.3f} s。',
            f'- 50个原视频等时窗，每窗约 {mapping["duration_s"]/50:.3f} s；文本/声学/视觉特征为 {shape["text_bert"]}/{shape["audio_opensmile"]}/{shape["vision_openface"]}。',
            '- 文本来自题给转写。词时间使用ASR中心估计及插值；ASR锚点通过自动门槛不代表人工确认的逐词真值。声学“有效”只表示波形非零，人脸“有效”只表示OpenFace跟踪成功。',
            '- 以下每行的帧时刻来自原视频采样，须人工确认跟踪脸属于表达主体。', '',
            '| 对齐窗 | 视频时段/s | 题给词 | 原视频帧时刻/s | 音频窗 | 人脸窗 | 音频帧数 | 有效人脸帧数 |',
            '|---:|---:|---|---:|---|---|---:|---:|']
        for b in selected:
            j=b['index']
            words=' '.join(mapping['words'][i]['token'] for i in b['word_indices'])
            lines.append(f'| {j} | {b["start_s"]:.3f}–{b["end_s"]:.3f} | {words} | '
                         f'{b["frame_time_s"]:.3f} | {audio[j]} | {face[j]} | '
                         f'{audio_frames[j]} | {face_frames[j]} |')
        lines += ['', '## 核查边界', '',
                  '这张表证明三支特征能落到相同原视频时间窗，并提供取帧位置；'
                  '它不证明词级时间的绝对误差，也不证明OpenFace跟踪的人脸就是情感表达主体。'
                  '实际画面截帧仅保留在本地 `outputs/paper/q1_example_frames`，未上传公开仓库。']
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'sample_id':a.sample_id,'rows':len(selected),'output':str(a.out.resolve())},ensure_ascii=False))

if __name__=='__main__':main()
