"""Extract time-indexed openSMILE eGeMAPSv02 LLDs for Q1's 100 videos.

The output is a separate feature version. It does not replace A2's supplied
74-dimensional audio and does not use labels. The fixed bins are Q1's own.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import opensmile
from opensmile.core.smile import Smile
from q1_extract import audio_wave


def smile_with_ascii_config():
    # openSMILE 2.6.0 encodes its config paths as ASCII. The project path is
    # Chinese, so copy only the bundled config files into a temporary ASCII path.
    module = Path(__import__('opensmile.core.smile', fromlist=['__file__']).__file__)
    temp = tempfile.TemporaryDirectory(prefix='q1_smile_')
    target = Path(temp.name) / 'config'
    if not str(target).isascii():
        temp.cleanup()
        raise RuntimeError('openSMILE needs an ASCII temporary directory on this machine')
    shutil.copytree(module.parent / 'config', target)
    class AsciiSmile(Smile):
        @property
        def default_config_root(self):
            return str(target)
    return temp, AsciiSmile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)


def aggregate(table, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    values = table.to_numpy(dtype=np.float32)
    nonfinite = int((~np.isfinite(values)).sum())
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    starts = table.index.get_level_values(0).total_seconds().to_numpy()
    ends = table.index.get_level_values(1).total_seconds().to_numpy()
    centers = (starts + ends) / 2
    target = np.clip(np.searchsorted(edges, centers, side='right') - 1, 0, 49)
    out = np.zeros((50, values.shape[1] * 2), dtype=np.float32)
    count = np.bincount(target, minlength=50).astype(np.int32)
    for i in range(50):
        part = values[target == i]
        if len(part):
            out[i] = np.r_[part.mean(axis=0), part.std(axis=0)]
    return out, count, nonfinite


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--q1-features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    base=next(a.data_root.resolve().glob('附件1-*/MOSEI*'))
    source=a.q1_features.resolve()
    rows=list(csv.DictReader((source/'manifest.csv').open(encoding='utf-8-sig',newline='')))
    if a.limit:rows=rows[:a.limit]
    a.out.mkdir(parents=True,exist_ok=True)
    temp,smile=smile_with_ascii_config()
    report=[]
    try:
        for n,row in enumerate(rows,1):
            try:
                mapping=json.loads((source/row['mapping_file']).read_text(encoding='utf-8'))
                with np.load(source/row['feature_file']) as old:
                    edges=old['time_edges_s'].astype(np.float64)
                    signal=old['audio_signal_present'].astype(bool)
                if signal.any():
                    wave=audio_wave(base/mapping['video_relative_path'])
                    table=smile.process_signal(wave,16000)
                    feature,count,nonfinite=aggregate(table,edges)
                else:
                    feature=np.zeros((50,50),dtype=np.float32)
                    count=np.zeros(50,dtype=np.int32)
                    nonfinite=0
                if feature.shape!=(50,50) or not np.isfinite(feature).all():
                    raise ValueError('invalid audio feature shape/value')
                np.savez_compressed(a.out/row['feature_file'],audio_opensmile=feature,
                                    frame_count=count,signal_present=signal,time_edges_s=edges.astype(np.float32))
                item={'sample_id':row['sample_id'],'feature_file':row['feature_file'],
                      'status':'ok','valid_bins':int(((count > 0) & signal).sum()),
                      'lld_frames':int(count.sum()),'nonfinite_raw_cells':nonfinite}
            except Exception as exc:
                item={'sample_id':row['sample_id'],'feature_file':row['feature_file'],
                      'status':'error','error':str(exc)}
            report.append(item)
            print(f"{n}/{len(rows)} {item['sample_id']} {item['status']}",flush=True)
    finally:
        temp.cleanup()
    fields=['sample_id','feature_file','status','valid_bins','lld_frames','nonfinite_raw_cells','error']
    with (a.out/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(report)
    summary={'requested':len(rows),'completed':sum(x['status']=='ok' for x in report),
             'failed':sum(x['status']!='ok' for x in report),
             'feature_set':'eGeMAPSv02','level':'LowLevelDescriptors',
             'lld_dim':25,'aggregation':'per fixed Q1 time bin mean+std',
             'output_shape':[50,50],
             'fully_silent_samples':sum(x.get('lld_frames')==0 for x in report),
             'nonfinite_raw_cells':sum(x.get('nonfinite_raw_cells',0) for x in report)}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['failed']:raise SystemExit(1)

if __name__=='__main__':main()
