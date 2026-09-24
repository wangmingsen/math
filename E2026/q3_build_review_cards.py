"""Build a local-only browser review page for A4 explanations and videos."""
from __future__ import annotations
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

HTML=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题三：新版模型视频审查</title>
<style>
body{font:16px/1.5 system-ui,"Microsoft YaHei",sans-serif;background:#f4f7fa;color:#17212d;margin:0}
header{background:#143b62;color:white;padding:16px 24px}header h1{margin:0;font-size:22px}
main{max-width:1100px;margin:auto;padding:18px}.card{background:white;border-radius:9px;padding:16px;margin:12px 0;box-shadow:0 1px 5px #0001}
video{width:100%;max-height:56vh;background:#111;border-radius:8px}.row{display:flex;gap:14px;flex-wrap:wrap}.stat{min-width:150px}
select,button,textarea{font:inherit;padding:7px;border:1px solid #9fb2c5;border-radius:6px;background:white}
button{cursor:pointer}button.primary{background:#215a91;color:white;border-color:#215a91}
textarea{width:100%;box-sizing:border-box;min-height:80px}.small{color:#587080;font-size:14px}
.warn{color:#9d3b00;font-weight:600}.evidence{border-left:4px solid #4380b4;padding:8px 12px;margin:10px 0;background:#f5f9fc}
pre{white-space:pre-wrap;word-wrap:break-word;font:inherit;margin:5px 0}
</style></head><body><header><h1>问题三：附件4新版模型视频审查</h1>
<div>20条原视频；位置到秒数若由ASR估计，会明确标注。遮蔽影响是模型敏感性，不是因果证明。</div></header>
<main><div class="card row"><label>样本 <select id="sample"></select></label><button id="export" class="primary">导出人工核查 CSV</button></div>
<div class="card"><div id="headline"></div><div id="warnings" class="warn"></div><video id="video" controls preload="metadata"></video></div>
<div class="card"><h2>逐模态遮蔽</h2><div id="impacts"></div></div>
<div class="card"><h2>局部证据</h2><div id="segments"></div></div>
<div class="card"><h2>人工核查</h2><div class="small">观看和听辨视频后，记录关键片段与模型预测是否相符；浏览器本地保存，完成后导出。</div>
<label>判断 <select id="decision"><option value="pending">待核查</option><option value="supports">证据与预测基本相符</option>
<option value="contradicts">证据与预测矛盾</option><option value="uncertain">无法判断</option></select></label>
<textarea id="notes" placeholder="记录听到的内容、画面、时间定位误差或模型问题"></textarea></div>
</main><script>
const data=__DATA__;const $=id=>document.getElementById(id);let current=0;
const names={text:'文本',audio:'语音/声学',vision:'视觉'},polarity=['负向','中性','正向'];
const key=id=>'q3-review-transformer-v1-'+id;function saved(id){try{return JSON.parse(localStorage.getItem(key(id)))||{}}catch{return {}}}
function save(){const x=data[current];localStorage.setItem(key(x.id),JSON.stringify({decision:$('decision').value,notes:$('notes').value}))}
function show(i){current=i;const x=data[i],p=x.pred;$('sample').value=x.id;
 $('headline').textContent=`${x.id} · ${polarity[Number(p.predicted_polarity)]} · 强度 ${Number(p.predicted_intensity).toFixed(3)} · 预测类别概率 ${Number(p['prob_'+['negative','neutral','positive'][Number(p.predicted_polarity)]]).toFixed(3)}`;
 const warnings=[];if(p.time_mapping_quality!=='asr_estimate')warnings.push('词时间无法核实，不提供秒数');
 if(p.text_truncated_at_50==='True')warnings.push('文本超过50位，后半句未进入模型');
 if(p.vision_observed==='False')warnings.push('视觉特征全零，不能提供面部局部证据');
 if(p.coherence_qa_flag!=='none')warnings.push('分类和连续强度存在需要解释的一致性问题');
 $('warnings').textContent=warnings.join('；');$('video').src=x.video;$('video').load();
 $('impacts').replaceChildren();for(const m of ['text','audio','vision']){const r=x.impacts[m],d=document.createElement('div');d.className='evidence';
  d.textContent=`${names[m]}：类别概率遮蔽差 ${Number(r.class_confidence_drop).toFixed(3)}；强度变化 ${Number(r.intensity_change_signed).toFixed(3)}；正向支持占比 ${Number(p[m+'_support_share']).toFixed(3)}`;$('impacts').appendChild(d)}
 $('segments').replaceChildren();for(const m of ['text','audio','vision']){const r=x.top[m];const d=document.createElement('div');d.className='evidence';
  if(!r){d.textContent=names[m]+'：无有效位置';$('segments').appendChild(d);continue}
  const time=r.time_quality==='asr_estimate'?`${Number(r.estimated_start_s).toFixed(2)}–${Number(r.estimated_end_s).toFixed(2)}秒（ASR估计）`:'秒数未验证';
  const title=document.createElement('strong');title.textContent=`${names[m]} · 位置 ${r.position_start}–${Number(r.position_end_exclusive)-1} · ${time}`;d.appendChild(title);
  const excerpt=document.createElement('pre');excerpt.textContent=r.text_excerpt||'无对应文本';d.appendChild(excerpt);
  const score=document.createElement('div');score.className='small';score.textContent=`遮蔽后原预测类别概率变化 ${Number(r.class_confidence_drop).toFixed(3)}；强度变化 ${Number(r.intensity_change_signed).toFixed(3)}`;d.appendChild(score);
  if(r.time_quality==='asr_estimate'){const b=document.createElement('button');b.textContent='跳到估计片段';b.onclick=()=>{$('video').currentTime=Math.max(0,Number(r.estimated_start_s));$('video').play()};d.appendChild(b)}
  $('segments').appendChild(d)}
 const old=saved(x.id);$('decision').value=old.decision||'pending';$('notes').value=old.notes||''}
data.forEach((x,i)=>{const o=document.createElement('option');o.value=x.id;o.textContent=x.id;$('sample').appendChild(o)});
$('sample').onchange=()=>show(data.findIndex(x=>x.id===$('sample').value));$('decision').onchange=save;$('notes').oninput=save;
$('export').onclick=()=>{const rows=[['sample_id','review_decision','review_notes'],...data.map(x=>{const s=saved(x.id);return[x.id,s.decision||'pending',s.notes||'']})];
 const esc=v=>'"'+String(v).replaceAll('"','""')+'"';const blob=new Blob(['\ufeff'+rows.map(r=>r.map(esc).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
 const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='q3_transformer_explanation_review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
show(data.findIndex(x=>x.id==='09')>=0?data.findIndex(x=>x.id==='09'):0);
</script></body></html>'''


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--explanations',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    pred={r['sample_id']:r for r in read_csv(a.explanations/'predictions.csv')}
    impacts=defaultdict(dict)
    for r in read_csv(a.explanations/'modality_impacts.csv'):
        impacts[r['sample_id']][r['modality']]=r
    local=defaultdict(lambda:defaultdict(list))
    for r in read_csv(a.explanations/'local_evidence.csv'):
        local[r['sample_id']][r['modality']].append(r)
    data=[]
    for row in read_csv(a.manifest):
        sid=row['sample_id']
        path=Path(row['video_path']).resolve()
        relative=Path(os.path.relpath(path,a.out.parent.resolve())).as_posix()
        top={m:max((r for r in local[sid][m] if int(r['observed_positions'])>0),
                   key=lambda r:float(r['class_confidence_drop']),default=None)
             for m in ('text','audio','vision')}
        data.append({'id':sid,'video':quote(relative,safe='/.'),'pred':pred[sid],
                     'impacts':impacts[sid],'top':top})
    payload=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    a.out.write_text(HTML.replace('__DATA__',payload),encoding='utf-8')
    print(json.dumps({'samples':len(data),'page':str(a.out.resolve()),'bytes':a.out.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()
