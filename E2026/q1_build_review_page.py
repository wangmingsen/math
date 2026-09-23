"""Build a local-only video review page from the detailed Q1 audit JSON.

The generated HTML embeds contest transcripts and must stay outside the public repo.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>问题一：原视频人工核查</title>
<style>
:root{font-family:system-ui,'Microsoft YaHei',sans-serif;color:#17212d;background:#f5f7fa}body{margin:0}header{background:#143b62;color:#fff;padding:18px 24px}h1{font-size:21px;margin:0 0 6px}header p{margin:0;color:#dce9f5}.layout{display:grid;grid-template-columns:260px minmax(0,1fr);min-height:calc(100vh - 92px)}aside{background:#e9eef4;padding:12px;overflow:auto;max-height:calc(100vh - 92px)}aside button{display:block;width:100%;text-align:left;border:0;border-radius:7px;background:#fff;margin:5px 0;padding:9px;cursor:pointer;color:#24384b}aside button.active{background:#d5e8fa;outline:2px solid #2865a0}main{padding:20px;max-width:1200px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}button,select{font:inherit;border:1px solid #9aafc2;border-radius:6px;padding:7px 10px;background:white;cursor:pointer}button.primary{background:#215a91;color:#fff;border-color:#215a91}video{width:100%;max-height:58vh;background:#111;border-radius:8px}.panel{background:#fff;border-radius:8px;padding:14px;margin-top:12px;box-shadow:0 1px 4px #0001}.label{font-weight:700;margin-bottom:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.5;margin:0}details summary{cursor:pointer}textarea{width:100%;min-height:74px;box-sizing:border-box;font:inherit;padding:8px}.muted{color:#5d6f82}.warn{color:#9a3200;font-weight:700}@media(max-width:760px){.layout{display:block}aside{max-height:180px}main{padding:12px}}
</style></head><body>
<header><h1>问题一：21 条原视频核查</h1><p>先听原视频，再看题给转写；识别文本仅作参考。每条判断保存在当前浏览器，点击“导出核查记录”交给小组。</p></header>
<div class="layout"><aside id="items"></aside><main>
<div class="toolbar"><button id="prev">上一条</button><strong id="counter"></strong><button id="next">下一条</button><button id="export" class="primary">导出核查记录 CSV</button></div>
<div id="status" class="warn"></div><video id="video" controls preload="metadata"></video>
<div class="panel"><div class="label">题给转写（请对照视频听辨）</div><pre id="given"></pre></div>
<div class="panel"><details><summary>展开自动识别文本（可能出错）</summary><pre id="asr"></pre></details></div>
<div class="panel"><label class="label" for="decision">人工判断</label><select id="decision"><option value="pending">未核查</option><option value="match">语音与题给文本一致</option><option value="offset">内容一致但时间有偏移</option><option value="mismatch">疑似片段或转写不对应</option><option value="unintelligible">音频有声但听不清</option><option value="silent">确认全静音</option><option value="other">其他</option></select><p class="muted">请记下听到的内容、可能的时间偏移和需要复核的秒数。自动识别不能替代人工听辨。</p><textarea id="notes" placeholder="核查备注……"></textarea></div>
<div class="panel muted" id="meta"></div>
</main></div>
<script>
const items = __DATA__;
let index=0;
const $=id=>document.getElementById(id);
const key=id=>'q1-review-'+id;
function saved(id){try{return JSON.parse(localStorage.getItem(key(id)))||{}}catch{return {}}}
function put(){let x=items[index];localStorage.setItem(key(x.id),JSON.stringify({decision:$('decision').value,notes:$('notes').value})) ;drawList()}
function drawList(){const root=$('items');root.replaceChildren();items.forEach((x,i)=>{const b=document.createElement('button');const d=saved(x.id);b.textContent=`${i+1}. ${x.id}  ${d.decision&&d.decision!=='pending'?'✓':''}`;b.className=i===index?'active':'';b.onclick=()=>show(i);root.appendChild(b)})}
function show(i){index=Math.max(0,Math.min(items.length-1,i));const x=items[index];$('counter').textContent=`${index+1} / ${items.length} · ${x.id}`;$('video').pause();$('video').src=x.src;$('video').load();$('given').textContent=x.given;$('asr').textContent=x.asr||'识别器没有得到可用文字';$('status').textContent=x.category==='decoded_stereo_pcm_all_zero'?'双声道解码全静音；仍可查看画面。':(x.category==='nonzero_audio_asr_too_few_words'?'有音频信号，但自动识别词数不足。':'音频非零，自动识别与题给文本低匹配。');$('meta').textContent=`原文件：${x.path}　|　音频 RMS：${x.rms}　|　tiny 匹配率：${x.tiny}　|　base 匹配率：${x.base}`;const d=saved(x.id);$('decision').value=d.decision||'pending';$('notes').value=d.notes||'';drawList()}
$('prev').onclick=()=>show(index-1);$('next').onclick=()=>show(index+1);$('decision').onchange=put;$('notes').oninput=put;
$('export').onclick=()=>{const header=['sample_id','manual_decision','manual_notes','video_relative_path'];const lines=[header,...items.map(x=>{let d=saved(x.id);return [x.id,d.decision||'pending',d.notes||'',x.path]})];const esc=v=>'"'+String(v).replaceAll('"','""')+'"';const csv='\ufeff'+lines.map(r=>r.map(esc).join(',')).join('\r\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='q1_manual_review_results.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight')show(index+1);if(e.key==='ArrowLeft')show(index-1)});
show(0);
</script></body></html>'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    rows=json.loads(a.audit.read_text(encoding='utf-8'))
    base=next(a.data_root.resolve().glob('附件1-*/MOSEI*'))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    ordered=[x for x in rows if x['category']!='decoded_stereo_pcm_all_zero']+[x for x in rows if x['category']=='decoded_stereo_pcm_all_zero']
    data=[]
    for x in ordered:
        video=base/x['video_relative_path']
        if not video.is_file():
            raise FileNotFoundError(video)
        relative=Path(os.path.relpath(video.resolve(),a.out.parent.resolve())).as_posix()
        data.append({'id':x['sample_id'],'src':quote(relative,safe='/.'),'path':x['video_relative_path'],
                     'given':x['given_transcript'],'asr':x['base_asr_transcript'],
                     'category':x['category'],'rms':x['pcm_rms'],
                     'tiny':x['tiny_exact_match_fraction'],'base':x['base_exact_match_fraction']})
    payload=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    a.out.write_text(HTML.replace('__DATA__',payload),encoding='utf-8')
    print(json.dumps({'videos':len(data),'missing':0,'page':str(a.out.resolve()),'bytes':a.out.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()