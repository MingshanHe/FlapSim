"""Scientific plots plus a self-contained, offline evolution explorer."""
import argparse,copy,csv,gzip,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from .networks import complexity
from .common import write

INPUTS=['roll','pitch','yaw','gyro x','gyro y','gyro z','accel x','accel y','accel z','height err','vertical speed']
OUTPUTS=['alpha residual','beta residual','motor residual']


def load_generations(folder):
    result=[]
    history=Path(folder)/'history.json'
    if not history.exists():return []
    committed={r['generation'] for r in json.loads(history.read_text())}
    for p in sorted(Path(folder).glob('generation_*.json.gz')):
        if int(p.name.split('_')[1].split('.')[0]) not in committed:continue
        with gzip.open(p,'rt') as f:result.append(json.load(f))
    return result


def champions(generations):
    return [max(g['individuals'],key=lambda x:x['fitness']) for g in generations]


def ancestry(generations,target):
    records={};birth={}
    for g in generations:
        for r in g['individuals']:
            if r['id'] not in records:records[r['id']]=r;birth[r['id']]=g['generation']
    path=[];seen=set();node=target
    while node in records and node not in seen:
        seen.add(node);r=records[node];path.append(dict(r,generation=birth[node]))
        parents=[records[x] for x in r['parents'] if x in records]
        if not parents:break
        node=max(parents,key=lambda x:x['fitness'])['id']
    return list(reversed(path)),records


def positions(spec,hidden_keys=None):
    inputs=[-i-1 for i in range(spec['inputs'])]
    hidden=hidden_keys or sorted(n['id'] for n in spec['nodes'] if n['id'] not in spec['outputs'])
    pos={k:(0.,1-i/(len(inputs)-1)) for i,k in enumerate(inputs)}
    pos.update({k:(1.,.8-i*.3) for i,k in enumerate(spec['outputs'])})
    pos.update({k:(.42+.16*(i%2),1-(i+1)/(len(hidden)+1)) for i,k in enumerate(hidden)})
    return pos


def draw_graph(ax,spec,title,hidden_keys=None):
    pos=positions(spec,hidden_keys);actual={n['id'] for n in spec['nodes']};incoming={k:0 for k in actual}
    for e in spec['edges']:
        if not e['enabled']:continue
        a,b=e['source'],e['target'];w=e['weight'];incoming[b]+=1
        color='#1575b6' if w>=0 else '#db4e50';width=.4+min(abs(w)*5,3)
        if a==b:
            x,y=pos[a];arrow=FancyArrowPatch((x-.01,y+.008),(x+.01,y+.008),connectionstyle='arc3,rad=-3',arrowstyle='-|>',mutation_scale=9,color=color,lw=width,alpha=.65)
        else:
            arrow=FancyArrowPatch(pos[a],pos[b],connectionstyle=f'arc3,rad={.10 if a>=0 else .03}',arrowstyle='-|>',mutation_scale=9,color=color,lw=width,alpha=.6,shrinkA=14,shrinkB=14)
        ax.add_patch(arrow)
    for k,(x,y) in pos.items():
        if k>=0 and k not in actual:continue
        color='#dae7f4' if k<0 else '#d9eee5' if k in spec['outputs'] else '#f9e7c2'
        ax.scatter([x],[y],s=340,c=color,edgecolors='#294153',zorder=3)
        label=INPUTS[-k-1] if k<0 else OUTPUTS[k] if k in spec['outputs'] else f'h{k}'
        if k<0:ax.text(x-.03,y,label,ha='right',va='center',fontsize=8)
        elif k in spec['outputs']:ax.text(x+.035,y,label,ha='left',va='center',fontsize=8)
        else:ax.text(x,y,str(k),ha='center',va='center',fontsize=8)
    ax.set(xlim=(-.28,1.30),ylim=(-.1,1.1));ax.axis('off');ax.set_title(title,fontsize=11)


def plots(folder,generations):
    folder=Path(folder);out=folder/'visuals';out.mkdir(exist_ok=True)
    ch=champions(generations);bygen={r['generation']:r for r in json.loads((folder/'history.json').read_text())}
    history=[bygen[g['generation']] for g in generations]
    gen=[g['generation'] for g in generations]
    fig,ax=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
    for field,label in [('fitness_max','best'),('fitness_mean','mean'),('fitness_min','worst')]:ax[0,0].plot(gen,[r[field] for r in history],label=label)
    ax[0,0].set_title('Training fitness (scenario changes each generation)');ax[0,0].legend()
    valid=[r for r in history if 'validation' in r]
    ax[0,1].plot([r['generation'] for r in valid],[r['validation']['mean_return'] for r in valid],'o-');ax[0,1].set_title('Fixed validation scenarios: mean return')
    for field in ['hidden','connections','recurrent_edges']:
        ax[1,0].plot(gen,[complexity(r['policy'])[field] for r in ch],label=field)
    ax[1,0].set_title('Generation champion complexity');ax[1,0].legend()
    ids=sorted({int(k) for g in generations for k in g['species']})
    counts=np.array([[len(g['species'].get(str(k),[])) for g in generations] for k in ids])
    ax[1,1].stackplot(gen,counts,labels=[f'species {i}' for i in ids]);ax[1,1].set_title('Species population')
    if len(ids)<10:ax[1,1].legend(fontsize=7)
    for a in ax.flat:a.set_xlabel('Generation');a.grid(alpha=.2)
    fig.savefig(out/'evolution_summary.png',dpi=160);plt.close(fig)
    selected=json.loads((folder/'selection.json').read_text()) if (folder/'selection.json').exists() else {'genome_id':ch[-1]['id'],'generation':gen[-1]}
    line,records=ancestry(generations,selected['genome_id']);write(out/'selected_ancestry.json',line)
    write(out/'all_parent_links.json',[dict(id=k,parents=r['parents']) for k,r in records.items()])
    # Both parents are shown, not only the single path used for weight heatmaps.
    births={}
    for generation in generations:
        for r in generation['individuals']:births.setdefault(r['id'],generation['generation'])
    ancestors=set();pending=[selected['genome_id']]
    while pending:
        key=pending.pop()
        if key in ancestors or key not in records:continue
        ancestors.add(key);pending.extend(records[key]['parents'])
    layers={g:sorted(k for k in ancestors if births[k]==g) for g in sorted(set(births[k] for k in ancestors))}
    coords={k:(g,(i+.5)/len(keys)) for g,keys in layers.items() for i,k in enumerate(keys)}
    fig,ax=plt.subplots(figsize=(11,max(4,min(12,max(map(len,layers.values()))*.22))),constrained_layout=True)
    main_ids={r['id'] for r in line}
    main_edges={(parent['id'],child['id']) for parent,child in zip(line,line[1:])}
    for child in ancestors:
        for parent in records[child]['parents']:
            if parent not in coords:continue
            x,y=coords[parent];u,v=coords[child]
            highlighted=(parent,child) in main_edges
            ax.plot([x,u],[y,v],color='#167aa2' if highlighted else '#9bacba',lw=1.7 if highlighted else .7,alpha=.9 if highlighted else .55,zorder=1)
    for key,(x,y) in coords.items():
        chosen=key==selected['genome_id']
        ax.scatter(x,y,s=100 if chosen else 32,c='#c55044' if chosen else '#167aa2' if key in main_ids else '#c6d1da',zorder=2)
        if len(ancestors)<=100 or key in main_ids:ax.annotate(str(key),(x,y),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6)
    ax.set(xticks=list(layers),yticks=[],xlabel='Birth generation (elite copies keep their ID)',
        title='Selected genome ancestry: both parents | red: selected | blue: heatmap path')
    ax.grid(axis='x',alpha=.2);fig.savefig(out/'selected_family_tree.png',dpi=170);plt.close(fig)
    with gzip.open(out/'population.csv.gz','wt',newline='',encoding='utf8') as fp, gzip.open(out/'weights.csv.gz','wt',newline='',encoding='utf8') as fw, gzip.open(out/'biases.csv.gz','wt',newline='',encoding='utf8') as fb:
        popcsv=csv.writer(fp);wc=csv.writer(fw);bc=csv.writer(fb)
        popcsv.writerow(['generation','genome_id','parents','species','fitness','nodes','enabled_connections'])
        wc.writerow(['generation','genome_id','source','target','weight','enabled','innovation'])
        bc.writerow(['generation','genome_id','node_id','bias','activation'])
        for g in generations:
            for r in g['individuals']:
                spec=r['policy'];popcsv.writerow([g['generation'],r['id'],json.dumps(r['parents']),r['species'],r['fitness'],len(spec['nodes']),sum(e['enabled'] for e in spec['edges'])])
                for e in spec['edges']:wc.writerow([g['generation'],r['id'],e['source'],e['target'],e['weight'],e['enabled'],e['innovation']])
                for n in spec['nodes']:bc.writerow([g['generation'],r['id'],n['id'],n['bias'],n['activation']])
    specs=[r['policy'] for r in ch]
    union=sorted({n['id'] for s in specs for n in s['nodes'] if n['id'] not in s['outputs']})
    frames=[]
    for i,(g,r) in enumerate(zip(gen,ch)):
        fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
        draw_graph(ax,r['policy'],f'Generation {g} champion #{r["id"]} | fitness {r["fitness"]:.4f}',union)
        fig.text(.5,.01,'Blue: positive weight | Red: negative | Thickness: magnitude | Champions may have different parents',ha='center',fontsize=9)
        path=out/f'champion_{g:04d}.png';fig.savefig(path,dpi=120);plt.close(fig)
        with Image.open(path) as im:frames.append(im.convert('RGB').copy())
    frames[0].save(out/'topology_evolution.gif',save_all=True,append_images=frames[1:],duration=900,loop=0)
    chosen=records[selected['genome_id']]['policy'];fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    draw_graph(ax,chosen,f'Validation-selected genome #{selected["genome_id"]} | generation {selected["generation"]}')
    fig.text(.5,.01,'Blue: positive | Red: negative | Thickness: magnitude | Non-input sources use previous state',ha='center',fontsize=8)
    fig.savefig(out/'selected_topology.png',dpi=180);plt.close(fig)
    # Show each inherited edge through one documented parent path. Missing/disabled != zero.
    edges=sorted({(e['source'],e['target']) for r in line for e in r['policy']['edges']})
    matrix=np.full((len(edges),len(line)),np.nan)
    for j,r in enumerate(line):
        values={(e['source'],e['target']):e for e in r['policy']['edges']}
        for i,key in enumerate(edges):
            if key in values and values[key]['enabled']:matrix[i,j]=values[key]['weight']
    fig,ax=plt.subplots(figsize=(max(8,len(line)*.65),max(5,min(16,len(edges)*.16))),constrained_layout=True)
    bound=max(.05,float(np.nanmax(np.abs(matrix)))) if matrix.size else .05
    cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#dddddd')
    im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=-bound,vmax=bound)
    ax.set_xticks(range(len(line)),[f'g{r["generation"]}\n#{r["id"]}' for r in line],fontsize=7)
    if len(edges)<100:ax.set_yticks(range(len(edges)),[f'{a} → {b}' for a,b in edges],fontsize=6)
    ax.set_title('Selected lineage weights | gray: absent/disabled | white: near zero')
    fig.colorbar(im,ax=ax,label='Weight');fig.savefig(out/'lineage_weights.png',dpi=160);plt.close(fig)
    # A compact reading aid alongside the complete heatmap and full CSV archive.
    variations=[np.ptp(row[np.isfinite(row)]) if np.isfinite(row).any() else 0. for row in matrix]
    top=sorted(np.argsort(variations)[-min(12,len(edges)):],key=lambda i:variations[i],reverse=True)
    fig,ax=plt.subplots(figsize=(11,4.8),constrained_layout=True)
    for color_index,i in enumerate(top):
        source,target=edges[i]
        ax.plot(range(len(line)),matrix[i],'o-',markersize=3,color=plt.get_cmap('tab20')(color_index),label=f'{source} -> {target}')
    ax.set_xticks(range(len(line)),[f'g{r["generation"]}\n#{r["id"]}' for r in line],fontsize=8)
    ax.set(ylabel='Weight',title='Selected lineage: up to 12 edges with greatest weight range')
    ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8,loc='best')
    if len(line)==1:
        ax.clear();ax.axis('off')
        ax.text(.5,.60,'The selected genome belongs to the initial population.',ha='center',va='center',fontsize=15,transform=ax.transAxes)
        ax.text(.5,.42,'There is no parent-to-child weight trajectory to display.\nInitial weights remain available in the heatmap, explorer and CSV archive.',ha='center',va='center',fontsize=12,transform=ax.transAxes)
    fig.savefig(out/'lineage_weight_curves.png',dpi=170);plt.close(fig)
    nodes=sorted({n['id'] for r in line for n in r['policy']['nodes']})
    biases=np.full((len(nodes),len(line)),np.nan)
    for j,r in enumerate(line):
        for n in r['policy']['nodes']:biases[nodes.index(n['id']),j]=n['bias']
    fig,ax=plt.subplots(figsize=(max(8,len(line)*.65),4),constrained_layout=True)
    bound=max(.01,float(np.nanmax(np.abs(biases))));im=ax.imshow(biases,aspect='auto',cmap=cmap,vmin=-bound,vmax=bound)
    ax.set_xticks(range(len(line)),[f'g{r["generation"]}\n#{r["id"]}' for r in line],fontsize=7)
    ax.set_yticks(range(len(nodes)),nodes);ax.set_title('Selected lineage node biases | gray: absent node')
    fig.colorbar(im,ax=ax,label='Bias');fig.savefig(out/'lineage_biases.png',dpi=160);plt.close(fig)
    events=[];seen=set()
    for g in generations:
        counts={k:0 for k in ['added_nodes','removed_nodes','added_edges','removed_edges','changed_weights','toggled_edges']}
        for r in g['individuals']:
            if r['id'] in seen:continue
            seen.add(r['id'])
            for key in counts:counts[key]+=len(r['mutation'].get(key,[]))
        events.append(counts)
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for key in ['added_nodes','removed_nodes','added_edges','removed_edges','toggled_edges']:ax[0].plot(gen,[e[key] for e in events],label=key)
    ax[0].legend(fontsize=7);ax[0].set_title('Recorded mutation events in newly born genomes')
    weights=[np.array([e['weight'] for r in g['individuals'] for e in r['policy']['edges'] if e['enabled']]) for g in generations]
    ax[1].boxplot(weights,showfliers=False,tick_labels=gen);ax[1].set_title('Enabled weight distribution across each population')
    for a in ax:a.set_xlabel('Generation');a.grid(alpha=.2)
    fig.savefig(out/'mutations_and_weights.png',dpi=160);plt.close(fig)
    write(out/'visualization_manifest.json',dict(generations=len(generations),archived_individual_evaluations=sum(len(g['individuals']) for g in generations),
        selected=selected,main_lineage='At each birth, follow the recorded parent with higher first-recorded training fitness; this is one path, not the whole ancestry.',
        mutation_events='Events are logged immediately around mutation, after crossover. Elite copies are counted once. Bias values are present in full genomes; bias mutation events are not separately classified.',
        graph_dynamics='All non-input nodes update synchronously. Any edge from a non-input node uses the previous actor tick; drawn left-to-right edges can therefore also carry one-tick memory.'))
    return out


HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>NEAT 演化浏览器</title>
<style>body{margin:0;background:#f2f5f8;color:#203045;font:15px system-ui}header{background:#14344b;color:white;padding:24px 32px}main{padding:20px 28px;max-width:1400px;margin:auto}.controls,.card{background:white;border-radius:10px;padding:16px;margin:12px 0;box-shadow:0 2px 8px #22334412}button,select,input{font:inherit;margin:5px;padding:5px}canvas{width:100%;min-width:540px;height:620px;background:white}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #e1e6ee;text-align:left}code{color:#145b79}.muted{color:#586e80}.split{display:grid;grid-template-columns:2fr 1fr;gap:16px}.split>.card{min-width:0;overflow:auto}@media(max-width:950px){.split{grid-template-columns:1fr}}h2{font-size:19px}.badge{display:inline-block;padding:6px 12px;background:#e8f1f7;border-radius:8px;margin:3px}#weights{max-height:350px;overflow:auto}a{color:#0c699b}</style>
<header><h1>NEAT 演化浏览器 · __NAME__</h1><div>完整种群、真实父代、出生时变异与权重。离线文件，无外部依赖。</div></header><main>
<div class="controls"><button id="play">▶ 播放冠军序列</button>代数 <input id="gen" type="range" min="0" max="__MAX__" value="0"><span id="glabel"></span> 个体 <select id="individual"></select><br>最小 |权重| <input id="threshold" type="range" min="0" max="1" step=".005" value="0"><span id="thlabel">0</span><label><input type="checkbox" id="disabled">显示禁用连接</label><button id="selected">查看验证选中的策略</button></div>
<div id="metrics" class="card"></div><div class="split"><div class="card"><canvas id="graph"></canvas><div class="muted">蓝色：正权重；红色：负权重；粗细：绝对值；虚线：禁用。节点内数字为基因 ID。所有来自非输入节点的连接使用上一时刻状态，代表记忆。不同代冠军不一定是一条父子链。</div></div><div class="card"><h2>来源与变异</h2><div id="parents"></div><div id="mutations"></div><h2>节点偏置</h2><div id="biases"></div></div></div>
<div class="card"><h2>连接权重</h2><div id="weights"></div></div><div class="card"><h2>训练和验证</h2><img width="100%" src="evolution_summary.png"><img width="100%" src="mutations_and_weights.png"><h2>选中策略的双亲家谱</h2><img width="100%" src="selected_family_tree.png"><h2>选中策略的祖先权重</h2><p class="muted">主祖先路径每次沿一个父代回溯；灰色表示连接不存在或被禁用。全部双亲关系保存在 all_parent_links.json。</p><img width="100%" src="lineage_weights.png"><img width="100%" src="lineage_weight_curves.png"><h2>祖先节点偏置</h2><img width="100%" src="lineage_biases.png"></div>
</main><script>
const DATA=__DATA__,SELECTION=__SELECTION__;const labels=__LABELS__;let playing=null;const el=id=>document.getElementById(id);let current;
function champion(g){return g.individuals.reduce((a,b)=>a.fitness>b.fitness?a:b)}
function changeGeneration(){let g=DATA[+el('gen').value];el('glabel').textContent=g.generation;let rows=[...g.individuals].sort((a,b)=>b.fitness-a.fitness);el('individual').innerHTML=rows.map(r=>`<option value="${r.id}">#${r.id} · fitness ${r.fitness.toFixed(4)} · species ${r.species}</option>`).join('');render()}
function jump(id){for(let i=+el('gen').value;i>=0;i--){if(DATA[i].individuals.some(r=>r.id===id)){el('gen').value=i;changeGeneration();el('individual').value=id;render();return}}}
function render(){let g=DATA[+el('gen').value];current=g.individuals.find(r=>r.id===+el('individual').value);let r=current,s=r.policy,edges=s.edges.filter(e=>e.enabled);el('metrics').innerHTML=[`个体 #${r.id}`,`代 ${g.generation}`,`fitness ${r.fitness.toFixed(5)}`,`节点 ${s.nodes.length}`,`启用连接 ${edges.length}`,`有记忆连接 ${edges.filter(e=>e.source>=0).length}`,`训练倾角 RMS ${r.episode.tilt_rms_deg.toFixed(2)}°`,`训练高度 RMS ${(r.episode.height_rmse_m*100).toFixed(2)} cm`,`训练片段 ${r.episode.elapsed.toFixed(2)} s`,...(r.id===SELECTION.genome_id&&SELECTION.validation?[`选中策略验证通过 ${(SELECTION.validation.success_rate*100).toFixed(0)}%`,`验证倾角 RMS ${SELECTION.validation.tilt_rms_deg.toFixed(2)}°`]:[])].map(t=>`<span class="badge">${t}</span>`).join('');el('parents').innerHTML=r.parents.length?`父代：${r.parents.map(id=>`<button onclick="jump(${id})">#${id}</button>`).join('')}`:'初始种群，无父代';let m=r.mutation;el('mutations').innerHTML=`<p>此个体出生时（交叉后）的变异：</p><ul>${Object.entries(m).map(([k,v])=>`<li>${k}：${v.length}</li>`).join('')}</ul><details><summary>展开事件明细</summary><pre style="font-size:11px;white-space:pre-wrap">${JSON.stringify(m,null,2)}</pre></details>`;el('biases').innerHTML='<table><tr><th>节点</th><th>偏置</th></tr>'+s.nodes.map(n=>`<tr><td>${n.id}</td><td>${n.bias.toFixed(6)}</td></tr>`).join('')+'</table>';el('weights').innerHTML='<table><tr><th>来源</th><th>目标</th><th>权重</th><th>启用</th><th>创新编号</th><th>时序</th></tr>'+[...s.edges].sort((a,b)=>Math.abs(b.weight)-Math.abs(a.weight)).map(e=>`<tr><td>${e.source<0?labels[-e.source-1]:e.source}</td><td>${e.target}</td><td style="color:${e.weight>=0?'#1575b6':'#db4e50'}">${e.weight.toFixed(7)}</td><td>${e.enabled?'是':'否'}</td><td>${e.innovation}</td><td>${e.source>=0?'上一时刻':'当前输入'}</td></tr>`).join('')+'</table>';draw()}
function draw(){if(!current)return;let canvas=el('graph'),ctx=canvas.getContext('2d'),w=canvas.clientWidth,h=620;canvas.width=w*2;canvas.height=h*2;ctx.scale(2,2);let s=current.policy,pos={},hidden=s.nodes.filter(n=>!s.outputs.includes(n.id)).sort((a,b)=>a.id-b.id);for(let i=0;i<s.inputs;i++)pos[-i-1]=[125,35+i*(h-70)/(s.inputs-1)];s.outputs.forEach((id,i)=>pos[id]=[w-95,140+i*160]);hidden.forEach((n,i)=>pos[n.id]=[w*(.43+.12*(i%2)),60+(i+1)*(h-120)/(hidden.length+1)]);let threshold=+el('threshold').value;el('thlabel').textContent=threshold;for(let e of s.edges){if((!e.enabled&&!el('disabled').checked)||Math.abs(e.weight)<threshold)continue;let a=pos[e.source],b=pos[e.target];ctx.strokeStyle=!e.enabled?'#bbb':e.weight>=0?'#1575b6':'#db4e50';ctx.lineWidth=.5+Math.min(Math.abs(e.weight)*7,5);ctx.globalAlpha=e.enabled?.65:.3;ctx.setLineDash(e.enabled?[]:[4,4]);ctx.beginPath();if(e.source===e.target){ctx.arc(a[0]+13,a[1]-15,20,.3,Math.PI*1.8)}else{let dx=b[0]-a[0],dy=b[1]-a[1],d=Math.hypot(dx,dy),ex=b[0]-dx/d*19,ey=b[1]-dy/d*19;ctx.moveTo(a[0],a[1]);ctx.lineTo(ex,ey);ctx.moveTo(ex,ey);ctx.lineTo(ex-dx/d*8+dy/d*4,ey-dy/d*8-dx/d*4);ctx.moveTo(ex,ey);ctx.lineTo(ex-dx/d*8-dy/d*4,ey-dy/d*8+dx/d*4)}ctx.stroke()}ctx.globalAlpha=1;ctx.setLineDash([]);for(let [key,p]of Object.entries(pos)){let id=+key;ctx.fillStyle=id<0?'#dae7f4':s.outputs.includes(id)?'#d9eee5':'#f9e7c2';ctx.strokeStyle='#294153';ctx.lineWidth=1;ctx.beginPath();ctx.arc(p[0],p[1],17,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.fillStyle='#203045';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText(id,p[0],p[1]+4);if(id<0){ctx.textAlign='right';ctx.fillText(labels[-id-1],p[0]-23,p[1]+4)}if(s.outputs.includes(id)){ctx.textAlign='left';ctx.fillText(['α','β','motor'][id],p[0]+23,p[1]+4)}}}
el('gen').oninput=changeGeneration;el('individual').onchange=render;el('threshold').oninput=draw;el('disabled').onchange=draw;window.onresize=draw;el('selected').onclick=()=>{el('gen').value=DATA.findIndex(g=>g.generation===SELECTION.generation);changeGeneration();el('individual').value=SELECTION.genome_id;render()};el('play').onclick=()=>{if(playing){clearInterval(playing);playing=null;el('play').textContent='▶ 播放冠军序列';return}el('play').textContent='■ 停止';playing=setInterval(()=>{el('gen').value=(+el('gen').value+1)%DATA.length;changeGeneration()},1000)};changeGeneration();
</script></html>'''


def build(folder):
    folder=Path(folder);generations=load_generations(folder)
    if not generations:return None
    out=plots(folder,generations)
    selection=json.loads((folder/'selection.json').read_text()) if (folder/'selection.json').exists() else dict(generation=generations[-1]['generation'],genome_id=champions(generations)[-1]['id'])
    html=HTML.replace('__NAME__',folder.name).replace('__MAX__',str(len(generations)-1)).replace('__DATA__',json.dumps(generations)).replace('__SELECTION__',json.dumps(selection)).replace('__LABELS__',json.dumps(INPUTS))
    (out/'explorer.html').write_text(html,encoding='utf8');return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(build(a.folder))
