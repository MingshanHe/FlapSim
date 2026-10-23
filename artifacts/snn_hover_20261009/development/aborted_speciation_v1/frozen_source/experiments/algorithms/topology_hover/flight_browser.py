"""Offline inspection of recorded node states; policy weights never change here."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import write
from .networks import NumpyPolicy
from .visualize import INPUTS


HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NEAT 飞行节点回放</title>
<style>body{margin:0;background:#eef3f7;color:#20384b;font:15px system-ui}main{max-width:1300px;margin:auto;padding:26px}h1{font-size:28px}h2{font-size:18px}.card{background:white;border:1px solid #d8e2ea;border-radius:12px;padding:18px;margin:18px 0}.muted{color:#597080;font-size:14px}.split{display:grid;grid-template-columns:1.7fr 1fr;gap:18px}.split>.card{min-width:0}canvas{width:100%;display:block}#graph{height:650px;min-width:590px}#signals{height:200px}.scroll{overflow:auto}button,select{font:inherit;padding:7px 12px;margin:4px}input[type=range]{width:min(65%,650px)}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:7px 4px;text-align:right;border-bottom:1px solid #e2e8ee}td:first-child,th:first-child{text-align:left}.badge{display:inline-block;border-radius:6px;background:#eaf2f7;padding:8px;margin:3px}a{color:#087599}@media(max-width:1050px){.split{grid-template-columns:1fr}main{padding:14px}}</style>
<main><h1>NEAT 飞行节点回放 · __LABEL__</h1><p>真实记录的 100 Hz 策略输入、输出和节点状态。拖动时间查看一次控制计算；网络结构、权重和偏置全程固定。</p><p class="muted">__SCOPE__ · yaw 与水平位置不控制。本页展示加权信号，并不把较大的权重或贡献直接解释为因果重要性。</p>
<div class="card"><button id="play">▶ 播放</button><button id="previous">上一拍</button><button id="next">下一拍</button><label>速度 <select id="speed"><option value=".1">0.1×</option><option value=".5">0.5×</option><option value="1" selected>1×</option></select></label><p><input id="tick" type="range" min="0" max="__MAX__" value="0" step="1"> <span id="time"></span></p><div id="metrics"></div></div>
<div class="split"><div class="card scroll"><canvas id="graph"></canvas><p class="muted">连接：蓝＝正权重，红＝负权重，线宽＝|权重|，虚线＝使用上一拍节点状态。节点：蓝＝正状态，红＝负状态，白＝接近零；颜色在 ±1 饱和，数字保留实际归一化数值。</p></div><div class="card"><h2>查看节点的加权输入</h2><label>目标节点 <select id="node"></select></label><p id="equation"></p><div class="scroll"><table><thead><tr><th>来源</th><th>输入值</th><th>权重</th><th>乘积</th></tr></thead><tbody id="terms"></tbody></table></div><p class="muted">传感器连接使用本拍输入；来自隐藏节点或输出节点的连接使用上一拍状态。第 0 拍的历史状态为零。表中数值为显示用的浮点重算，记录值来自原控制器。</p></div></div>
<div class="card"><h2>三个残差输出</h2><canvas id="signals"></canvas><p class="muted">蓝：δα；橙：δβ；紫：δmotor。范围 [-1,1]，分别乘以 0.035 rad、0.035 rad、100 rad/s 后叠加固定 PID，再受执行器范围限制。上图记录的是残差策略输出。</p></div>
<p><a href="../report.md">返回实验报告</a> · <a href="../traces/__LABEL__.npz">下载原始轨迹</a></p></main>
<script>const DATA=__DATA__,SPEC=__SPEC__,LABELS=__LABELS__,el=id=>document.getElementById(id);let index=0,playing=false,lastTime=0,fraction=0;
const ids=SPEC.nodes.map(n=>n.id),nodeIndex=Object.fromEntries(ids.map((id,i)=>[id,i])),nin=SPEC.inputs;
const input=row=>row.slice(1,1+nin),states=row=>row.slice(1+nin+3),outputs=row=>row.slice(1+nin,1+nin+3),name=id=>id<0?LABELS[-id-1]:SPEC.outputs.includes(id)?['α output','β output','motor output'][SPEC.outputs.indexOf(id)]:`h${id}`;
const fmt=v=>(v>=0?'+':'')+v.toFixed(5);function stop(){playing=false;el('play').textContent='▶ 播放'}
function choose(i){index=Math.max(0,Math.min(DATA.length-1,i));el('tick').value=index;render()}
function setupCanvas(id,height){const canvas=el(id),w=canvas.clientWidth,ctx=canvas.getContext('2d');canvas.width=w*2;canvas.height=height*2;ctx.scale(2,2);return [ctx,w,height]}
function drawGraph(){const [ctx,w,h]=setupCanvas('graph',650),row=DATA[index],values={},pos={},hidden=SPEC.nodes.filter(n=>!SPEC.outputs.includes(n.id));
input(row).forEach((v,i)=>{values[-i-1]=v;pos[-i-1]=[160,40+i*(h-80)/(nin-1)]});states(row).forEach((v,i)=>values[ids[i]]=v);
SPEC.outputs.forEach((id,i)=>pos[id]=[w-100,145+i*175]);hidden.forEach((n,i)=>pos[n.id]=[w*(.5+.1*(i%2)),100+(i+1)*(h-200)/(hidden.length+1)]);
for(const e of SPEC.edges.filter(e=>e.enabled)){const a=pos[e.source],b=pos[e.target];ctx.strokeStyle=e.weight>=0?'#1677b2':'#ce4d4e';ctx.lineWidth=.5+Math.min(Math.abs(e.weight)*6,4);ctx.globalAlpha=.5;ctx.setLineDash(e.source>=0?[5,4]:[]);ctx.beginPath();if(e.source===e.target){ctx.arc(a[0]+17,a[1]-22,24,.2,Math.PI*1.8)}else{const dx=b[0]-a[0],dy=b[1]-a[1],d=Math.hypot(dx,dy),ex=b[0]-dx/d*25,ey=b[1]-dy/d*25;ctx.moveTo(a[0],a[1]);ctx.lineTo(ex,ey);ctx.moveTo(ex,ey);ctx.lineTo(ex-dx/d*9+dy/d*4,ey-dy/d*9-dx/d*4);ctx.moveTo(ex,ey);ctx.lineTo(ex-dx/d*9-dy/d*4,ey-dy/d*9+dx/d*4)}ctx.stroke()}
ctx.globalAlpha=1;ctx.setLineDash([]);for(const [key,p]of Object.entries(pos)){const id=+key,v=values[id],q=Math.min(1,Math.abs(v)),base=v>=0?[80,169,211]:[233,121,116];ctx.fillStyle=`rgb(${base.map(c=>Math.round(255+(c-255)*q)).join(',')})`;ctx.strokeStyle=id===+el('node').value?'#112c42':'#8198a9';ctx.lineWidth=id===+el('node').value?3:1;ctx.beginPath();ctx.arc(p[0],p[1],24,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.fillStyle='#1e3445';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText(v.toFixed(2),p[0],p[1]+4);if(id<0){ctx.textAlign='right';ctx.fillText(name(id),p[0]-30,p[1]+4)}else{ctx.fillText(name(id),p[0],p[1]+41)}}}
function drawSignals(){const [ctx,w,h]=setupCanvas('signals',200),left=40,right=w-20,top=20,bottom=h-25,T=DATA.at(-1)[0]||1,y=v=>top+(1-v)*(bottom-top)/2,x=t=>left+t/T*(right-left);ctx.font='12px system-ui';for(const v of [-1,0,1]){ctx.strokeStyle='#e0e7ec';ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(right,y(v));ctx.stroke();ctx.fillStyle='#526c7c';ctx.fillText(v,12,y(v)+4)}['#177bab','#d98228','#8960b2'].forEach((color,k)=>{ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();DATA.forEach((r,i)=>{const xx=x(r[0]),yy=y(outputs(r)[k]);i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)});ctx.stroke()});ctx.strokeStyle='#233e52';ctx.beginPath();ctx.moveTo(x(DATA[index][0]),top);ctx.lineTo(x(DATA[index][0]),bottom);ctx.stroke();ctx.fillText('0 s',left,h-4);ctx.textAlign='right';ctx.fillText(T.toFixed(2)+' s',right,h-4)}
function render(){const row=DATA[index],prev=index?states(DATA[index-1]):ids.map(()=>0),obs=input(row),state=states(row),target=+el('node').value,node=SPEC.nodes[nodeIndex[target]];let sum=node.bias;
el('time').textContent=`${row[0].toFixed(2)} s · 第 ${index} 拍`;el('metrics').innerHTML=outputs(row).map((v,i)=>`<span class="badge">${['δα','δβ','δmotor'][i]} ${fmt(v)}</span>`).join('');
const edges=SPEC.edges.filter(e=>e.enabled&&e.target===target).map(e=>{const value=e.source<0?obs[-e.source-1]:prev[nodeIndex[e.source]],term=value*e.weight;sum+=term;return {...e,value,term}}).sort((a,b)=>Math.abs(b.term)-Math.abs(a.term));
el('terms').innerHTML=edges.map(e=>`<tr><td>${name(e.source)}${e.source>=0?' (t−1)':''}</td><td>${fmt(e.value)}</td><td>${fmt(e.weight)}</td><td>${fmt(e.term)}</td></tr>`).join('')+`<tr><td>bias</td><td>1</td><td>${fmt(node.bias)}</td><td>${fmt(node.bias)}</td></tr>`;
el('equation').textContent=`tanh(${fmt(sum)}) ≈ ${fmt(Math.tanh(sum))}；记录状态 ${fmt(state[nodeIndex[target]])}`;drawGraph();drawSignals()}
el('node').innerHTML=SPEC.nodes.map(n=>`<option value="${n.id}">${name(n.id)} · ID ${n.id}</option>`).join('');el('node').onchange=render;el('tick').oninput=()=>{stop();choose(+el('tick').value)};el('previous').onclick=()=>{stop();choose(index-1)};el('next').onclick=()=>{stop();choose(index+1)};
el('play').onclick=()=>{if(playing){stop();return}if(index===DATA.length-1)choose(0);playing=true;fraction=index;lastTime=performance.now();el('play').textContent='■ 暂停';requestAnimationFrame(animate)};
function animate(now){if(!playing)return;fraction+=(now-lastTime)/1000*100*(+el('speed').value);lastTime=now;choose(Math.floor(fraction));if(index===DATA.length-1){stop();return}requestAnimationFrame(animate)}window.addEventListener('resize',render);render();
</script></html>'''


def build(root,label):
    root=Path(root)
    representative=json.loads((root/'representatives.json').read_text())['neat']
    spec=json.loads((root/representative['path']).read_text())
    data=np.load(root/'traces'/f'{label}.npz')['activations']
    episode=json.loads((root/'traces'/f'{label}.json').read_text())['episode']
    n=spec['inputs'];policy=NumpyPolicy(spec);max_error=0.
    assert data.shape[1]==1+n+3+len(spec['nodes'])
    np.testing.assert_allclose(np.diff(data[:,0]),.01,rtol=0,atol=1e-9)
    for row in data:
        action=policy.act(row[1:1+n].astype(np.float32))
        max_error=max(max_error,float(np.max(np.abs(action-row[1+n:1+n+3]))),
                      float(np.max(np.abs(policy.state-row[1+n+3:]))))
    assert max_error<2e-6, f'Recorded activations do not reproduce: {max_error}'
    scope=f"seed {episode['seed']} · {episode['case']['condition']} · {episode['elapsed']:.2f} s · 单个轨迹的解释工具，不是总体成功率"
    html=(HTML.replace('__LABEL__',label).replace('__SCOPE__',scope)
          .replace('__MAX__',str(len(data)-1)).replace('__DATA__',json.dumps(data.tolist()))
          .replace('__SPEC__',json.dumps(spec)).replace('__LABELS__',json.dumps(INPUTS)))
    out=root/'flight';out.mkdir(exist_ok=True)
    (out/f'{label}.html').write_text(html,encoding='utf8')
    write(out/f'{label}_verification.json',dict(status='passed',source_seed=episode['seed'],
        ticks=len(data),recording_hz=100,maximum_recomputed_state_or_action_error=max_error,
        weights_fixed_during_flight=True,source_policy=representative['path']))
    return out/f'{label}.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    for path in sorted((args.root/'traces').glob('neat_*.npz')):print(build(args.root,path.stem))
