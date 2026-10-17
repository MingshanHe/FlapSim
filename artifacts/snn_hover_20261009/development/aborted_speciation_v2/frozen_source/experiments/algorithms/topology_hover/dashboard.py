"""Local, read-only campaign overview; generated HTML also works offline."""
import argparse
import json
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


def snapshot(root):
    def read(path,default=None):
        try:return json.loads(path.read_text())
        except (OSError,ValueError):return default
    runs=[]
    for stage in ['pilot','formal','transfer']:
        for p in sorted((root/stage).glob('*/status.json')):
            row=read(p)
            if row is None:continue
            row['folder']=str(p.parent.relative_to(root))
            row['has_visuals']=(p.parent/'visuals/explorer.html').exists()
            runs.append(row)
    return dict(campaign=read(root/'campaign_status.json',{}),runs=runs,
                evaluation=read(root/'evaluation_status.json'),delivery=read(root/'delivery_status.json'),
                failure=read(root/'campaign_failure.json') or read(root/'parallel_transfer_failure.json') or read(root/'finalization_failure.json'),
                expansion=read(root/'expansion_decision.json'),timestamp=time.time())


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>扑翼控制实验总览</title>
<style>body{background:#edf2f6;color:#1d3447;font:16px system-ui;margin:0}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:30px}a{color:#08709b}.card{background:white;padding:22px;margin:20px 0;border-radius:12px;border:1px solid #d9e3eb}.muted{color:#5d7081;font-size:14px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.num{font-size:29px;color:#087b88}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #e0e7ed;font-size:14px}progress{width:100%;height:20px;accent-color:#168594}.error{color:#a52f25}img{width:100%}@media(max-width:800px){main{padding:16px}.grid{grid-template-columns:1fr}.scroll{overflow:auto}}</style>
<main><h1>扑翼机器人 · NEAT / PPO / PID</h1><p>上部 7 g＋下部 8 g · 姿态稳定与保高 · yaw、水平位置不控制</p>
<p><a href="report.md">打开完整 report.md</a> · <a href="protocol.json">实验协议</a></p>
<div class="grid"><div class="card">已完成训练<div class="num" id="count">—</div></div><div class="card">当前阶段<div class="num" id="stage">—</div></div><div class="card">结果范围<div id="scope">—</div></div></div>
<div class="card"><div id="current"></div><progress id="progress" max="100" value="0"></progress><p class="muted" id="clock"></p><div class="error" id="failure"></div></div>
<div class="card"><h2>训练与验证</h2><p class="muted">这里是验证集选择成绩。不同训练种子使用不同验证场景；最终方法比较请看 report.md 中的独立测试。</p><div class="scroll"><table><thead><tr><th>方法／种子</th><th>状态</th><th>交互数</th><th>验证通过</th><th>倾角 RMS</th><th>演化记录</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<div class="card"><h2>怎样查看 NEAT 的演化</h2><p>进入某一行的“打开演化浏览器”，拖动代数后选择任意个体；点击父代可回溯家谱。蓝色连接为正权重，红色为负权重，线宽表示权重大小。</p><p>每代冠军可能来自不同家族。双亲家谱、祖先权重与偏置图另外保存；飞行激活图展示节点状态变化，飞行时权重固定。</p></div>
<p class="muted">本地服务运行时每 15 秒读取状态；离线打开时显示生成本文件时保存的快照。</p></main>
<script>const SNAPSHOT=__SNAPSHOT__;let last=SNAPSHOT;const el=id=>document.getElementById(id);const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(s,live){last=s;
let done=s.runs.filter(r=>r.status==='completed').length,planned=s.expansion?.expand_to_ten?70:35;
let reviewed=s.delivery?.status==='reviewed',tested=s.evaluation?.stage==='completed',trained=s.campaign.status==='training_complete';
el('count').textContent=`${done} / ${planned}`;
el('stage').textContent=reviewed?'报告已完成':s.delivery?.status==='ready_for_final_review'?'等待最终审阅':tested?'测试完成':s.evaluation?'独立测试':trained?'训练完成':'训练中';
el('scope').textContent=tested?'独立测试已完成，请查看完整报告。':'单次验证分数不能证明拓扑优势。';
let r=s.runs.find(x=>x.status!=='completed'&&s.campaign.current?.endsWith(x.folder));
let testing=s.evaluation&&!tested,percent=r?100*r.training_steps/r.budget:testing&&s.evaluation.total?100*s.evaluation.completed/s.evaluation.total:trained?100:0;
let testName=testing?({heldout:'20 秒独立测试',long:'60 秒长时测试',interventions:'连接干预测试'}[s.evaluation.stage]||s.evaluation.stage):'';
el('current').textContent=r?`${r.method} · seed ${r.seed} · 当前运行 ${percent.toFixed(1)}%`:reviewed?'训练、测试与可视化已完成。':tested?'测试已完成，正在整理报告与视频。':testing?`${testName} · 已完成 ${s.evaluation.completed} / ${s.evaluation.total}`:trained?'训练已完成，正在进行独立测试与结果整理。':'等待下一阶段';
el('progress').value=percent;el('clock').textContent=(live?'实时状态':'离线快照')+' · '+new Date(s.timestamp*1000).toLocaleString();
el('failure').textContent=s.failure?'执行出现问题：'+s.failure.message:'';
el('rows').innerHTML=s.runs.map(r=>{let v=r.best_validation;return `<tr><td>${esc(r.method)} / ${r.seed}</td><td>${r.status==='completed'?'完成':'训练中'}</td><td>${r.training_steps.toLocaleString()}</td><td>${v?(v.success_rate*100).toFixed(0)+'%':'—'}</td><td>${v?v.tilt_rms_deg.toFixed(2)+'°':'—'}</td><td>${r.has_visuals?`<a href="${esc(r.folder)}/visuals/explorer.html">打开演化浏览器</a>`:'—'}</td></tr>`}).join('')}
render(SNAPSHOT,false);async function refresh(){if(location.protocol==='file:')return;try{let r=await fetch('/api/status',{cache:'no-store'});if(r.ok)render(await r.json(),true)}catch(e){}}refresh();setInterval(refresh,15000);</script></html>'''


def build(root):
    (root/'dashboard.html').write_text(HTML.replace('__SNAPSHOT__',json.dumps(snapshot(root),ensure_ascii=False)),encoding='utf8')


def serve(root,port):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split('?')[0]=='/api/status':
                data=json.dumps(snapshot(root),ensure_ascii=False).encode()
                self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            else:super().do_GET()
        def log_message(self,*args):pass
    ThreadingHTTPServer(('127.0.0.1',port),partial(Handler,directory=str(root))).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--serve',type=int,metavar='PORT');a=p.parse_args()
    root=a.root.resolve();build(root)
    if a.serve:serve(root,a.serve)
