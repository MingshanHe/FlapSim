"""Paired replay with exact state checks and millisecond neural recordings."""
import argparse,base64,gzip,json
from pathlib import Path
from .design import read_protocol
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write,record_criteria
from .environment import HoverEnv,Zero,rollout
from .networks import policy
from .evaluate import core_cases,extra_cases
from ..topology_hover.render import plot_trace


def record(task):
    spec,c,mode=task;env=HoverEnv(mode=mode,sensor_mode='height');p=policy(spec) if spec else Zero()
    temporal=hasattr(p,'capture')
    if temporal:p.capture=True
    acts=[];original=p.act
    def act(obs):
        y=original(obs);state=getattr(p,'voltage',getattr(p,'state',np.empty(0)))
        acts.append([env.air.steps*.002,*obs,*y,*state]);return y
    p.act=act;r=record_criteria(rollout(env,p,c,True),env)
    r['reference_position']=env.air.start.tolist()
    trace=np.asarray(r.pop('trace'));activations=np.asarray(acts)
    r['saturation_fraction']=np.mean(np.column_stack([np.abs(trace[:,8:10])>=.3-1e-9,(trace[:,10]<=3115.224079+1e-6)|(trace[:,10]>=3807.496097-1e-6)]),axis=0).tolist()
    internal=np.asarray(p.internal_trace) if temporal else np.empty((0,0))
    # Recompute the actor solely from recorded sensor inputs, independent of flight.
    replay=policy(spec) if spec else Zero()
    if temporal:replay.capture=True
    error=0.
    for row in activations:
        y=replay.act(row[1:12].astype(np.float32));error=max(error,float(np.max(np.abs(y-row[12:15]))))
    if temporal:
        np.testing.assert_allclose(replay.internal_trace,internal,rtol=0,atol=1e-10)
        error=max(error,float(np.max(np.abs(np.asarray(replay.internal_trace)-internal))))
    assert error<2e-6
    return r,trace,activations,internal,dict(maximum_recomputed_error=error,neural_substeps_per_action=10 if temporal else 1,weights_fixed_during_flight=True)


def neural_plots(root,label,spec,activations,internal):
    target=root/'figures';target.mkdir(exist_ok=True);N=len(spec['nodes']);ids=[n['id'] for n in spec['nodes']]
    hidden=[i for i,k in enumerate(ids) if k not in spec['outputs']];t=internal[:,0]
    z=internal[:,1:1+N];V=internal[:,1+N:1+2*N];I=internal[:,1+2*N:1+3*N]
    fig,axes=plt.subplots(3,1,figsize=(13,9),constrained_layout=True)
    for i in hidden:axes[0].scatter(t[z[:,i]>0],np.full(int(z[:,i].sum()),i),marker='|',s=12,color='#187ea2')
    axes[0].set(yticks=hidden,yticklabels=[ids[i] for i in hidden],ylabel='Hidden node ID',title='Recorded spikes (1 ms internal integration; inputs held for10 steps)')
    chosen=sorted(hidden,key=lambda i:z[:,i].sum(),reverse=True)[:3]
    for i in chosen:
        axes[1].plot(t,V[:,i],label=f'h{ids[i]} V; tau_m={spec["nodes"][i]["tau_m"]*1000:.1f}ms',lw=.7)
        axes[2].plot(t,I[:,i],label=f'h{ids[i]} I; tau_s={spec["nodes"][i]["tau_s"]*1000:.1f}ms',lw=.7)
    axes[1].axhline(1,color='black',ls=':',lw=.7);axes[1].set(xlim=(0,min(2,t[-1])),ylabel='Post-reset voltage',title='First2s; up to3 most-active hidden neurons (display selection only)')
    axes[2].set(xlim=(0,min(2,t[-1])),ylabel='Synaptic current',xlabel='Internal integration clock (s)')
    for ax in axes[1:]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(target/f'{label}_spikes.png',dpi=160);plt.close(fig)
    data=dict(spec=spec,acts=activations.tolist(),internal=internal.tolist())
    payload=base64.b64encode(gzip.compress(json.dumps(data,separators=(',',':')).encode())).decode()
    html=NEURAL_HTML.replace('__LABEL__',label).replace('__PAYLOAD__',payload)
    folder=root/'flight';folder.mkdir(exist_ok=True);(folder/f'{label}.html').write_text(html,encoding='utf8')


NEURAL_HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SNN 内部状态回放</title>
<style>body{background:#eef3f7;color:#203448;font:16px system-ui;max-width:1250px;margin:30px auto;padding:15px}.card{background:white;border-radius:12px;padding:18px;margin:16px 0}canvas{width:100%;height:280px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #dbe4ea;text-align:right}td:first-child,th:first-child{text-align:left}input{width:70%}button{padding:8px;font:inherit}p{line-height:1.6}.muted{color:#516879}</style>
<h1>SNN 内部状态回放 · __LABEL__</h1><p>每个控制决策内部计算10步。拖动查看真实记录的脉冲、电压、电流和有符号输出；网络结构和参数在飞行中固定。</p>
<p class="muted">内部积分时间是模型的计算时钟，并不是增加了传感器频率。每10个内部步骤读取一次11维输入；执行器仍为100 Hz。此处使用连续电流编码和连续读出，只有隐藏节点发放脉冲。</p>
<div class="card"><button id="play">播放</button> <input id="tick" type="range" value="0" min="0" step="1"><b id="time"></b><p id="action"></p></div>
<div class="card"><h2>全部隐藏节点的脉冲</h2><canvas id="raster"></canvas></div>
<div class="card"><h2>当前内部状态</h2><table><thead><tr><th>节点</th><th>脉冲</th><th>重置后电压 V</th><th>滤波电流 I</th><th>τm (ms)</th><th>τs (ms)</th></tr></thead><tbody id="nodes"></tbody></table></div>
<div class="card"><h2>当前100 Hz输入</h2><p id="obs"></p><p class="muted">顺序：roll、pitch、yaw、三轴角速度、三轴加速度、高度误差、估计垂直速度；均为策略使用的归一化值。没有水平位置反馈。</p></div>
<p><a href="../report.md">报告</a> · <a href="../traces/__LABEL__.npz">原始记录</a></p>
<script>async function boot(){const bytes=Uint8Array.from(atob('__PAYLOAD__'),c=>c.charCodeAt(0));const d=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text()),s=d.spec,N=s.nodes.length,el=id=>document.getElementById(id);let timer=null;el('tick').max=d.internal.length-1;
function render(){let k=+el('tick').value,r=d.internal[k],a=d.acts[Math.min(d.acts.length-1,Math.floor(k/10))];el('time').textContent=r[0].toFixed(3)+' s';el('action').textContent='本次完整决策的输出：α '+a[12].toFixed(5)+'，β '+a[13].toFixed(5)+'，motor '+a[14].toFixed(5)+'；控制器请求时刻 '+a[0].toFixed(2)+' s';el('obs').textContent=a.slice(1,12).map(x=>x.toFixed(4)).join(' · ');el('nodes').innerHTML=s.nodes.map((n,i)=>`<tr><td>${s.outputs.includes(n.id)?'读出':'隐藏'} #${n.id}</td><td>${r[1+i]?'●':'—'}</td><td>${r[1+N+i].toFixed(6)}</td><td>${r[1+2*N+i].toFixed(6)}</td><td>${(n.tau_m*1000).toFixed(3)}</td><td>${(n.tau_s*1000).toFixed(3)}</td></tr>`).join('');draw(k)}
function draw(k){let cv=el('raster'),w=cv.clientWidth,h=280;cv.width=w*2;cv.height=h*2;let c=cv.getContext('2d');c.scale(2,2);let hid=s.nodes.map((n,i)=>s.outputs.includes(n.id)?-1:i).filter(i=>i>=0),T=d.internal.at(-1)[0],xx=t=>65+t/T*(w-90);c.font='13px system-ui';hid.forEach((i,j)=>{let y=30+(j+.5)*(h-70)/hid.length;c.fillStyle='#395a70';c.fillText('h'+s.nodes[i].id,10,y+4);c.strokeStyle='#1a87a2';c.beginPath();for(const r of d.internal)if(r[1+i]){let x=xx(r[0]);c.moveTo(x,y-4);c.lineTo(x,y+4)}c.stroke()});c.strokeStyle='#e47c44';c.beginPath();c.moveTo(xx(d.internal[k][0]),15);c.lineTo(xx(d.internal[k][0]),h-30);c.stroke();c.fillText('0 s',65,h-5);c.fillText(T.toFixed(2)+' s',w-65,h-5)}
el('tick').oninput=render;el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='播放';return}el('play').textContent='暂停（0.1×）';timer=setInterval(()=>{el('tick').value=(+el('tick').value+10)%d.internal.length;render()},100)};window.onresize=render;render()}boot().catch(e=>document.body.append('读取失败：'+e));</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args();root=args.root
    assert json.loads((root/'evaluation_status.json').read_text())['status']=='completed'
    protocol=read_protocol(root);reps=json.loads((root/'representatives.json').read_text())
    selected=[dict(method='pid',name='pid',path=None,mode='pid')]+list(reps.values())
    rows=[json.loads(x) for x in (root/'evaluation/episodes.jsonl').read_text().splitlines()];tasks=[];metadata=[]
    cases=[(c['condition'],c,'core') for cid,c in core_cases(protocol) if c['seed']==protocol['test']['seeds'][0] and c['condition']!='ood']
    cases += [('angle30',c,'angle') for g,cid,c in extra_cases(protocol) if g=='angle' and c['tilt_deg']==30 and c['seed']==protocol['recovery']['angle_seeds'][0]]
    for e in selected:
        spec=json.loads(Path(e['path']).read_text()) if e['path'] else None
        for label,c,group in cases:
            reference=next(r for r in rows if r['policy']==e['name'] and r['group']==group and r['case']==c)
            tasks.append((spec,c,e['mode']));metadata.append((f'{e["method"]}_{label}',e,spec,reference))
    target=root/'traces';target.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (label,e,spec,reference),(episode,trace,acts,internal,verification) in zip(metadata,pool.map(record,tasks)):
            for k in ['return_sum','elapsed','tilt_rms_deg','height_rmse_m','minimum_clearance_m','xy_offset_m']:
                np.testing.assert_allclose(episode[k],reference[k],rtol=0,atol=1e-10)
            assert episode['success']==reference['success'] and episode['termination']==reference['termination']
            np.savez_compressed(target/f'{label}.npz',trace=trace,activations=acts,internal=internal)
            write(target/f'{label}.json',dict(policy=e,episode=episode,verification=verification,selection='Validation-selected policy; first predeclared case, never chosen by test outcome.'))
            plot_trace(root,label)
            if len(internal):neural_plots(root,label,spec,acts,internal)
            print(label,flush=True)
if __name__=='__main__':main()
