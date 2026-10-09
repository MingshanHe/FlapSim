"""Complete population explorer, birth events, ancestry and learned time scales."""
import argparse,base64,copy,csv,gzip,io,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from ..topology_hover.visualize import HTML,INPUTS,draw_graph,load_generations,champions
from .common import write
from .networks import complexity


def lineage(generations,target):
    records={};birth={}
    for g in generations:
        for r in g['individuals']:
            if r['id'] not in records:records[r['id']]=r;birth[r['id']]=g['generation']
    branch=[];seen=set();node=target
    while node in records and node not in seen:
        seen.add(node);r=records[node];branch.append(dict(r,generation=birth[node]))
        parents=sorted(p for p in r['parents'] if p in records)
        if not parents:break
        node=parents[0]
    return list(reversed(branch)),records,birth


def heatmap(out,line,field):
    edges=sorted({(e['source'],e['target']) for r in line for e in r['policy']['edges']})
    nodes=sorted({n['id'] for r in line for n in r['policy']['nodes']})
    keys=edges if field=='weights' else nodes
    matrix=np.full((len(keys),len(line)),np.nan)
    for j,r in enumerate(line):
        for item in r['policy']['edges'] if field=='weights' else r['policy']['nodes']:
            if field=='weights':
                if item['enabled']:matrix[keys.index((item['source'],item['target'])),j]=item['weight']
            else:matrix[keys.index(item['id']),j]=item['bias'] if field=='biases' else 1000*item[field]
    fig,ax=plt.subplots(figsize=(max(8,len(line)*.48),max(4,min(17,len(keys)*.16))),constrained_layout=True)
    temporal=field.startswith('tau');cmap=plt.get_cmap('viridis' if temporal else 'RdBu').copy();cmap.set_bad('#d0d0d0')
    bound=max(.01,float(np.nanmax(np.abs(matrix))))
    im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0 if temporal else -bound,vmax=bound)
    ax.set_xticks(range(len(line)),[f'g{r["generation"]}\n#{r["id"]}' for r in line],fontsize=7)
    ax.set_yticks(range(len(keys)),[f'{k[0]} -> {k[1]}' if isinstance(k,tuple) else str(k) for k in keys],fontsize=6)
    ax.set_title(f'Selected ancestry branch: {field} | smaller-ID parent at each birth; gray = absent / disabled')
    fig.colorbar(im,ax=ax,label='Time constant (ms)' if temporal else field)
    fig.savefig(out/f'lineage_{field}.png',dpi=150);plt.close(fig)
    return matrix,keys


def build(folder):
    folder=Path(folder);generations=load_generations(folder)
    if not generations:return None
    out=folder/'visuals';out.mkdir(exist_ok=True)
    history=json.loads((folder/'history.json').read_text());ch=champions(generations);xs=[g['generation'] for g in generations]
    chosen=json.loads((folder/'selection.json').read_text());spec=json.loads((folder/'best_policy.json').read_text())
    line,records,birth=lineage(generations,chosen['genome_id'])
    if chosen['generation']==-1:
        line=[dict(records[chosen['genome_id']],generation=-1,policy=spec)]
    write(out/'selected_ancestry.json',line)
    write(out/'all_parent_links.json',[dict(id=k,generation=birth[k],parents=r['parents']) for k,r in records.items()])
    merges=[event for g in generations for event in g.get('species_merges',[])]
    write(out/'species_capacity_merges.json',merges)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for field,fun in [('best',np.max),('mean',np.mean),('worst',np.min)]:
        axes[0,0].plot(xs,[fun([r['fitness'] for r in g['individuals']]) for g in generations],label=field)
    axes[0,0].set_title('Training fitness: scenario and horizon change');axes[0,0].legend()
    val=[r for r in history if 'validation' in r]
    axes[0,1].plot([r['generation'] for r in val],[r['validation']['mean_return'] for r in val],'o-',label='generation champion')
    axes[0,1].plot([r['generation'] for r in val],[r['best_validation']['mean_return'] for r in val],label='selected')
    axes[0,1].set_title('Fixed validation mean return');axes[0,1].legend()
    for field in ['hidden','connections','effective_connections']:
        axes[1,0].plot(xs,[complexity(r['policy'])[field] for r in ch],label=field)
    axes[1,0].set_title('Generation champion complexity');axes[1,0].legend()
    ids=sorted({r['species'] for g in generations for r in g['individuals']})
    axes[1,1].stackplot(xs,[[sum(r['species']==s for r in g['individuals']) for g in generations] for s in ids],labels=[str(s) for s in ids])
    axes[1,1].set_title('Population by species')
    if len(ids)<12:axes[1,1].legend(fontsize=7)
    for ax in axes.flat:ax.set_xlabel('Generation');ax.grid(alpha=.2)
    fig.savefig(out/'evolution_summary.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,8),constrained_layout=True);draw_graph(ax,spec,f'{folder.name} | validation-selected genome #{chosen["genome_id"]}')
    fig.savefig(out/'selected_network.png',dpi=160);plt.close(fig)
    ancestors=set();queue=[chosen['genome_id']]
    while queue:
        k=queue.pop()
        if k in ancestors or k not in records:continue
        ancestors.add(k);queue+=records[k]['parents']
    levels={g:sorted(k for k in ancestors if birth[k]==g) for g in sorted({birth[k] for k in ancestors})}
    pos={k:(j-(len(keys)-1)/2,g) for g,keys in levels.items() for j,k in enumerate(keys)}
    fig,ax=plt.subplots(figsize=(max(8,min(20,max(map(len,levels.values()))*.7)),max(4,len(levels)*.42)),constrained_layout=True)
    for k in ancestors:
        for parent in records[k]['parents']:
            if parent in pos:ax.plot([pos[parent][0],pos[k][0]],[pos[parent][1],pos[k][1]],color='#9bb0bd',lw=.6)
        ax.scatter(*pos[k],s=125,c='#eec56d' if k==chosen['genome_id'] else '#bfdae8',zorder=2)
        ax.text(*pos[k],str(k),ha='center',va='center',fontsize=5)
    ax.set(ylabel='Birth generation',title='Both recorded parents of every selected ancestor (not causal credit)');ax.set_xticks([])
    fig.savefig(out/'selected_family_tree.png',dpi=160);plt.close(fig)
    weights,edgekeys=heatmap(out,line,'weights');heatmap(out,line,'biases')
    if spec['kind'] in ('snn','leaky_ann'):
        heatmap(out,line,'tau_m');heatmap(out,line,'tau_s')
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    variations=[np.ptp(r[np.isfinite(r)]) if np.isfinite(r).any() else 0 for r in weights]
    for i in np.argsort(variations)[-12:]:ax.plot(range(len(line)),weights[i],'o-',ms=3,label=str(edgekeys[i]))
    ax.set_xticks(range(len(line)),[f'g{r["generation"]}\n#{r["id"]}' for r in line],fontsize=7)
    ax.set(title='Ancestry branch: up to 12 edges with largest weight range',ylabel='Weight');ax.legend(ncol=4,fontsize=7);ax.grid(alpha=.2)
    if len(line)==1:ax.set_title('Selected initial genome: no inherited weight trajectory')
    fig.savefig(out/'lineage_weight_curves.png',dpi=160);plt.close(fig)
    events=[];seen=set()
    for g in generations:
        row={k:0 for k in ['added_nodes','removed_nodes','added_edges','removed_edges','toggled_edges','changed_weights','changed_nodes']}
        for r in g['individuals']:
            if r['id'] in seen:continue
            seen.add(r['id'])
            for k in row:row[k]+=len(r['mutation'].get(k,[]))
        events.append(row)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for k in ['added_nodes','removed_nodes','added_edges','removed_edges','toggled_edges']:
        axes[0].plot(xs,[r[k] for r in events],label=k)
    axes[0].legend(fontsize=7);axes[0].set_title('Newborn mutation events: elite copies counted once')
    axes[1].boxplot([[e['weight'] for r in g['individuals'] for e in r['policy']['edges'] if e['enabled']] for g in generations],showfliers=False,tick_labels=xs)
    axes[1].tick_params(axis='x',labelsize=6);axes[1].set_title('Population enabled-weight distribution')
    for ax in axes:ax.set_xlabel('Generation');ax.grid(alpha=.2)
    fig.savefig(out/'mutations_and_weights.png',dpi=160);plt.close(fig)
    with gzip.open(out/'population.csv.gz','wt',newline='') as f,gzip.open(out/'weights.csv.gz','wt',newline='') as wf,gzip.open(out/'nodes.csv.gz','wt',newline='') as nf:
        a,b,c=csv.writer(f),csv.writer(wf),csv.writer(nf)
        a.writerow(['generation','genome','species','fitness','parents','nodes','enabled_edges']);b.writerow(['generation','genome','source','target','weight','enabled','innovation']);c.writerow(['generation','genome','node','bias','tau_m_s','tau_s_s'])
        for g in generations:
            for r in g['individuals']:
                s=r['policy'];a.writerow([g['generation'],r['id'],r['species'],r['fitness'],json.dumps(r['parents']),len(s['nodes']),sum(e['enabled'] for e in s['edges'])])
                for e in s['edges']:b.writerow([g['generation'],r['id'],e['source'],e['target'],e['weight'],e['enabled'],e['innovation']])
                for n in s['nodes']:c.writerow([g['generation'],r['id'],n['id'],n['bias'],n.get('tau_m'),n.get('tau_s')])
    frames=[];hidden=sorted({n['id'] for r in ch for n in r['policy']['nodes'] if n['id'] not in spec['outputs']})
    for g,r in zip(generations,ch):
        fig,ax=plt.subplots(figsize=(10,7));draw_graph(ax,r['policy'],f'Generation {g["generation"]} champion #{r["id"]} (not one lineage)',hidden)
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=90);plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB').copy())
    frames[0].save(out/'topology_evolution.gif',save_all=True,append_images=frames[1:],duration=650,loop=0)
    # Keep every individual while reducing the offline payload; no external service.
    payload=base64.b64encode(gzip.compress(json.dumps(generations,separators=(',',':')).encode())).decode()
    html=HTML.replace('__NAME__',folder.name).replace('__MAX__',str(len(generations)-1)).replace('__SELECTION__',json.dumps(chosen)).replace('__LABELS__',json.dumps(INPUTS))
    html=html.replace('const DATA=__DATA__,SELECTION=',"async function boot(){const bytes=Uint8Array.from(atob('"+payload+"'),c=>c.charCodeAt(0));const DATA=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text()),SELECTION=")
    html=html.replace('</script>',"window.jump=jump;}boot().catch(e=>document.body.append('读取失败：'+e));</script>")
    html=html.replace('所有来自非输入节点的连接使用上一时刻状态，代表记忆。','SNN/Leaky ANN：隐藏状态按 1 ms 更新，读出为有符号连续值；ANN：按 10 ms 更新。连接存在不证明记忆有用。')
    html=html.replace('有记忆连接','隐藏来源连接')
    html=html.replace('节点偏置','节点偏置及时间常数')
    html=html.replace('<th>偏置</th>','<th>偏置</th><th>τm / τs (ms)</th>')
    html=html.replace('<td>${n.bias.toFixed(6)}</td>','<td>${n.bias.toFixed(6)}</td><td>${(n.tau_m*1000).toFixed(2)} / ${(n.tau_s*1000).toFixed(2)}</td>')
    html=html.replace("DATA.findIndex(g=>g.generation===SELECTION.generation)","Math.max(0,DATA.findIndex(g=>g.generation===SELECTION.generation))")
    html=html.replace('主祖先路径每次沿一个父代回溯','主祖先路径每次沿编号较小的父代回溯，不代表该父代贡献更大')
    if spec['kind'] in ('snn','leaky_ann'):html=html.replace('</main>','<div class="card"><h2>祖先时间常数</h2><img width="100%" src="lineage_tau_m.png"><img width="100%" src="lineage_tau_s.png"></div></main>')
    html=html.replace('</main>','<div class="card"><h2>物种容量约束</h2><p>超过16个物种时，合并遗传代表距离最近的两个物种，基因本身不改变。该步骤只限制分组，不能算作基因突变。</p><a href="species_capacity_merges.json">查看全部真实合并事件</a></div></main>')
    (out/'explorer.html').write_text(html,encoding='utf8')
    write(out/'visualization_manifest.json',dict(generations=len(generations),archived_individual_evaluations=sum(len(g['individuals']) for g in generations),unique_genomes=len(records),selected=chosen,
        main_lineage='Follow smaller-ID recorded parent at each birth. Both parents retained in family tree; no causal parent attribution.',
        mutation_events='Mutation measured after crossover. Count each genome once at first appearance.',capacity_merges=len(merges),time_constants_unit='seconds in exports; milliseconds in plots',
        graph_dynamics='Temporal hidden nodes update synchronously every1ms, readouts filtered continuously. Ordinary ANN uses10ms actor ticks. All evolved parameters stay fixed in flight.'))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();links=[]
    for folder in sorted((a.root/'training').iterdir()):
        if (folder/'results.json').exists() and (folder/'initial_population.json.gz').exists():
            out=build(folder);links.append(f'<li><a href="{out.relative_to(a.root).as_posix()}/explorer.html">{folder.name}</a></li>')
    (a.root/'evolution_index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>演化档案</title><body style="font:18px system-ui;max-width:900px;margin:40px auto"><h1>NEAT / SNN 演化档案</h1><p>25 次训练的完整种群、双亲、结构、权重、偏置及时间常数。训练片段分数不能用来跨代判断泛化性能。</p><ul>'+''.join(links)+'</ul></body></html>')
if __name__=='__main__':main()
