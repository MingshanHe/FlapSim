"""Build the FlapSim Run Board documents (runs/, blind/, status/now) from local result files.

    python docs/dashboard/board_sync.py [--status status.json]
Writes JSON documents under runs/board_docs/<collection>/<id>.json; Claude uploads them with the ArtifactData tool.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.algorithms.spike_steer.paired_report import paired  # noqa: E402
OUT = ROOT/'runs/board_docs'


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def blind_doc(plan, arm, seed, machine, report, note=None, thr=.95):
    tasks = {}
    for t, v in report.items():
        if not isinstance(v, dict) or 'ratio' not in v:
            continue
        tasks[t] = dict(ratio=v['ratio'], wins=v.get('wins'), n=v.get('n', sum(c.get('n', 0) for c in v.get('by_class', {}).values())),
                        crashes=v.get('crashes', 0), classes={k: c['ratio'] for k, c in v.get('by_class', {}).items()})
    mean = report.get('manoeuvre_mean', sum(x['ratio'] for x in tasks.values())/max(1, len(tasks)))
    crashes = sum(x['crashes'] for x in tasks.values())
    return dict(plan=plan, arm=arm, seed=seed, machine=machine, mean=mean, tasks=tasks, crashes=crashes,
                pass_=None, **{'pass': bool(mean <= thr and crashes == 0)}, note=note)


def val_curve(reselect):
    cs = sorted(reselect['centres'], key=lambda c: c['generation'])
    sel = reselect.get('selected') or {}
    return dict(x=[c['generation'] for c in cs], y=[c['objective'] for c in cs],
                excluded=[bool(c['crashes'] or c['over_hard']) for c in cs], selected=sel.get('generation'))


def build(status_file=None):
    docs = {'runs': {}, 'blind': {}}
    # ---- plan v6
    a6 = ROOT/'artifacts/spike_steer/plan_v6'
    if (a6/'claude_ea/reselect_s4.json').exists():
        h = jl(a6/'finetune_s4/history.jsonl')
        docs['runs']['v6_snn_s4'] = dict(plan='v6', arm='snn_es', seed=4, label='SNN + ES, seed 4', machine='CloudPC-2', status='done',
                                         val=val_curve(json.loads((a6/'claude_ea/reselect_s4.json').read_text())),
                                         train=dict(x=[r['generation'] for r in h], y=[r['center'] for r in h]))
        e = json.loads((a6/'claude_ea/evaluate_s4.json').read_text())
        docs['blind']['v6_snn_s4'] = blind_doc('v6', 'snn_es', 4, 'CloudPC-2', e['report'], note='selected gen 44')
    p6 = ROOT/'artifacts/spike_steer/ppo_direct/v6ppo'
    for s in range(1, 6):
        d = p6/f's{s}'
        if (d/'reselect.json').exists():
            r = json.loads((d/'reselect.json').read_text())
            docs['runs'][f'v6_ppo_s{s}'] = dict(plan='v6', arm='ppo', seed=s, label=f'MLP + PPO, seed {s}', machine='HomePC', status='done',
                                                val=val_curve(r), seedShade=1.0 if s == 1 else .7)
        if (d/'evaluate.json').exists():
            e = json.loads((d/'evaluate.json').read_text())
            sel = (e.get('selected') or {}).get('generation')
            docs['blind'][f'v6_ppo_s{s}'] = blind_doc('v6', 'ppo', s, 'HomePC (reference)', e['report'],
                                                       note='selected the imitation start (gen −1)' if sel == -1 else f'selected gen {sel}')
    # ---- plan v7
    v7 = ROOT/'runs/spike_steer/v7'
    for f in sorted(v7.glob('finetune_s*/history.jsonl')):
        s = int(f.parent.name.split('_s')[-1])
        h = jl(f)
        doc = dict(plan='v7', arm='snn_es', seed=s, label=f'SNN + ES, seed {s}', machine='CloudPC-2', status='running',
                   train=dict(x=[r['generation'] for r in h], y=[r['center'] for r in h]))
        rs = v7/f'reselect_s{s}.json'
        if rs.exists():
            doc['val'], doc['status'] = val_curve(json.loads(rs.read_text())), 'done'
        docs['runs'][f'v7_snn_s{s}'] = doc
    for f in sorted(v7.glob('evaluate_*.json')):
        e = json.loads(f.read_text())
        key = f.stem.replace('evaluate_', '')
        docs['blind'][f'v7_{key}'] = blind_doc('v7', e.get('arm', 'snn_es'), e.get('seed'), e.get('machine', 'CloudPC-2'), e['report'], note=e.get('note'))
    for f in sorted(v7.glob('finetune_s*_*/history.jsonl')):      # tagged ES variants run on this machine
        s_, tag = f.parent.name[len('finetune_s'):].split('_', 1)
        h = jl(f)
        doc = dict(plan='v7', arm='snn_es', seed=int(s_), label=f'SNN + ES ({tag}), seed {s_}', machine='CloudPC-2', status='running',
                   train=dict(x=[r['generation'] for r in h], y=[r['center'] for r in h]), seedShade=.6)
        rs = v7/f'reselect_s{s_}_{tag}.json'
        if rs.exists():
            doc['val'], doc['status'] = val_curve(json.loads(rs.read_text())), 'done'
        docs['runs'][f'v7_snn_{tag}_s{s_}'] = doc
    # ---- plan v7, HomePC PPO arm (pushed to ppo-full under artifacts/spike_steer/ppo_direct/v7ppo/s<S>/)
    p7 = ROOT/'artifacts/spike_steer/ppo_direct/v7ppo'
    for f in sorted(p7.glob('reselect_s*.json')):          # HomePC layout: files at the top of v7ppo/
        s = int(f.stem.split('_s')[-1])
        docs['runs'][f'v7_ppo_s{s}'] = dict(plan='v7', arm='mlp_ppo', seed=s, label=f'MLP + PPO, seed {s}', machine='HomePC', status='done',
                                            val=val_curve(json.loads(f.read_text())))
    for f in sorted(p7.glob('evaluate_*_s*.json')):
        e = json.loads(f.read_text())
        arm, s = e.get('arm'), int(f.stem.split('_s')[-1])
        docs['blind'][f'v7_{arm}_s{s}'] = blind_doc('v7', arm, s, 'HomePC', e['report'], note=e.get('note'))
    for d in sorted(p7.glob('s[0-9]*')):
        if not d.is_dir():
            continue
        s = int(d.name[1:])
        doc = dict(plan='v7', arm='mlp_ppo', seed=s, label=f'MLP + PPO, seed {s}', machine='HomePC', status='running')
        if (d/'reselect.json').exists():
            doc['val'], doc['status'] = val_curve(json.loads((d/'reselect.json').read_text())), 'done'
            docs['runs'][f'v7_ppo_s{s}'] = doc
        for f in d.glob('evaluate_*.json'):
            e = json.loads(f.read_text())
            arm = e.get('arm', 'mlp_ppo')
            docs['blind'][f'v7_{arm}_s{s}'] = blind_doc('v7', arm, s, e.get('machine', 'HomePC'), e['report'], note=e.get('note'))
    # ---- data relayed by message (GoogleCloud): ready-made documents in runs/board_inbox/<collection>/<id>.json
    inbox = ROOT/'runs/board_inbox'
    for f in sorted(inbox.glob('*/*.json')):
        docs.setdefault(f.parent.name, {})[f.stem] = json.loads(f.read_text())
    # ---- plan v7 full (confirmatory): CloudPC-2 SNN careful ES + HomePC PPO
    vf = ROOT/'runs/spike_steer/v7full'
    for f in sorted(vf.glob('finetune_s*_careful/history.jsonl')):
        s_ = int(f.parent.name[len('finetune_s'):].split('_')[0])
        h = jl(f)
        doc = dict(plan='v7full', arm='snn_es', seed=s_, label=f'SNN + ES careful, seed {s_}', machine='CloudPC-2', status='running',
                   train=dict(x=[r['generation'] for r in h], y=[r['center'] for r in h]))
        rs = vf/f'reselect_s{s_}_careful.json'
        if rs.exists():
            doc['val'], doc['status'] = val_curve(json.loads(rs.read_text())), 'done'
        docs['runs'][f'v7full_snn_s{s_}'] = doc
    for f in sorted(vf.glob('evaluate_*_s*_careful.json')):
        e = json.loads(f.read_text())
        s_ = e.get('seed')
        docs['blind'][f"v7full_{e.get('arm', 'snn_es')}_s{s_}"] = blind_doc('v7full', e.get('arm', 'snn_es'), s_, 'CloudPC-2', e['report'], note=e.get('note'))
    pf = ROOT/'artifacts/spike_steer/ppo_direct/v7full'
    for f in sorted(pf.glob('reselect_s*.json')):
        s_ = int(f.stem.split('_s')[-1])
        docs['runs'][f'v7full_ppo_s{s_}'] = dict(plan='v7full', arm='mlp_ppo', seed=s_, label=f'MLP + PPO, seed {s_}', machine='HomePC', status='done',
                                                 val=val_curve(json.loads(f.read_text())))
    for f in sorted(pf.glob('evaluate_*_s*.json')):
        e = json.loads(f.read_text())
        arm, s_ = e.get('arm'), int(f.stem.split('_s')[-1])
        docs['blind'][f'v7full_{arm}_s{s_}'] = blind_doc('v7full', arm, s_, 'HomePC', e['report'], note=e.get('note'))
    # ---- plan v8: frozen re-test (CloudPC-2 SNN + classical, HomePC PPO)
    for d8, mach in ((ROOT/'runs/spike_steer/v8', 'CloudPC-2'), (ROOT/'artifacts/spike_steer/ppo_direct/v8', 'HomePC')):
        for f in sorted(d8.glob('evaluate_*_s*.json')):
            e = json.loads(f.read_text())
            arm, s_ = e.get('arm'), int(f.stem.split('_s')[-1])
            doc = blind_doc('v8', arm, s_, mach, e['report'], note=e.get('note'))
            c8 = e.get('crashes', doc['crashes'])
            doc.update(crashes=c8, note=f"{c8} crashed of 192 (classical: 5)")
            if 'M4_pass' in e and e['M4_pass'] is not None:
                doc['pass'] = e['M4_pass']
            else:
                doc['pass'] = bool(doc['mean'] <= .95 and c8 <= 5)
            docs['blind'][f'v8_{arm}_s{s_}'] = doc
    # ---- plans v9 (bench10 scoring) and v10 (37-input layer): CloudPC-2 seeds 1-3, HomePC seeds 4-5 (pushed under plan_v*/homepc)
    for plan, rdir, adir, lab, tg in (('v9', 'v9', 'plan_v9', 'v9', '_careful'), ('v10', 'v10', 'plan_v10', 'v10', '_careful'), ('v11', 'v11', 'plan_v11', 'v11', '_v11')):
        cloud = ROOT/f'runs/spike_steer/{rdir}'
        opp_f = cloud/'test_opponent.json'
        opp_rows = json.loads(opp_f.read_text()) if opp_f.exists() else None
        opp_crash = sum(not r['flown'] for r in opp_rows) if opp_rows else None
        for root, mach in ((cloud, 'CloudPC-2'), (ROOT/f'artifacts/spike_steer/{adir}/homepc', 'HomePC')):
            for f in sorted(root.glob(f'finetune_s*{tg}/history.jsonl')):
                s_ = int(f.parent.name[len('finetune_s'):].split('_')[0])
                h = jl(f)
                doc = dict(plan=plan, arm='snn_es', seed=s_, label=f'SNN + ES ({lab}), seed {s_}', machine=mach, status='running',
                           train=dict(x=[r['generation'] for r in h], y=[r['center'] for r in h]))
                rs = root/f'reselect_s{s_}{tg}.json'
                if rs.exists():
                    doc['val'], doc['status'] = val_curve(json.loads(rs.read_text())), 'done'
                docs['runs'][f'{plan}_snn_s{s_}'] = doc
            for f in sorted(root.glob('cls_es_s*/history.jsonl')):
                s_ = int(f.parent.name[len('cls_es_s'):])
                h = jl(f)
                doc = dict(plan=plan, arm='cls_es', seed=s_, label=f'Classical + ES upper bound ({lab}), seed {s_}', machine=mach, status='running',
                           train=dict(x=[r['generation'] for r in h], y=[r['center'] for r in h]))
                rs = root/f'reselect_cls_es_s{s_}.json'
                if rs.exists():
                    doc['val'], doc['status'] = val_curve(json.loads(rs.read_text())), 'done'
                docs['runs'][f'{plan}_cls_s{s_}'] = doc
            for f in sorted(list(root.glob(f'evaluate_*_s*{tg}.json'))+list(root.glob('evaluate_snn9_s*.json'))):
                e = json.loads(f.read_text())
                arm, s_ = e.get('arm', 'snn_es'), e.get('seed')
                rows = e.get('rows', [])
                cn = sum(not r['flown'] for r in rows)
                doc = blind_doc(plan, arm, s_, mach, e['report'], note=e.get('note'))
                doc.update(crashes=cn)
                if opp_crash is not None:
                    doc['pass'] = bool(doc['mean'] <= .95 and cn <= opp_crash)
                    doc['note'] = f"{cn} crashed of {len(rows)} (classical: {opp_crash})"
                if opp_rows and rows and len(rows) == len(opp_rows):
                    pr = paired(opp_rows, rows)
                    doc.update(delta=pr['delta'], delta_ci=pr['delta_ci95'], wins=pr['wins'], classical_self=pr['classical_self'])
                if plan == 'v11':
                    doc['test'] = 'nominal'
                docs['blind'][f'{plan}_{arm}_s{s_}'] = doc
    # ---- plan v12: blind-only robustness tests (kick / ood) of frozen controllers
    for pdir, test in [('v12', t_) for t_ in ('kick2', 'kick', 'kick45', 'kick_early', 'kick_late', 'ood')]+[('v13', t_) for t_ in ('turb125', 'turb15', 'turb2', 'turb3', 'noise5', 'delay3', 'kick_pre', 'kick_double', 'kick_rollaxis', 'kick_pitchaxis')]:
        d = ROOT/f'runs/spike_steer/{pdir}/{test}'
        sf = d/'summary.json'
        if not sf.exists():
            continue
        S = json.loads(sf.read_text())
        for f in sorted(d.glob('evaluate_*.json')):
            e = json.loads(f.read_text())
            name = f.stem[len('evaluate_'):]
            p = e['paired']
            doc = blind_doc('v11', e['arm'], e.get('seed'), e.get('machine', 'CloudPC-2'), e['report'], note=e.get('note'))
            doc.update(test=test, crashes=p['crashes'], delta=p['delta'], delta_ci=p['delta_ci95'], wins=p['wins'], classical_self=p['classical_self'])
            doc['pass'] = e.get('H1_pass')
            docs['blind'][f'v12_{test}_{name}'] = doc
        any_p = next((S[k] for k in S if isinstance(S.get(k), dict) and 'classical_self' in S[k]), None)
        docs['blind'][f'v12_{test}_classical'] = dict(plan='v11', arm='classical', seed=None, machine='CloudPC-2', test=test,
                                                     mean=any_p['classical_self'] if any_p else None, tasks={}, crashes=S['classical']['crashes'],
                                                     **{'pass': None}, note=f"classical opponent's own ratio (reference); {S['classical']['crashes']} crashed of {S['classical']['n']}")
    if status_file:
        docs['status'] = {'now': json.loads(Path(status_file).read_text())}
    OUT.mkdir(parents=True, exist_ok=True)
    for col, items in docs.items():
        (OUT/col).mkdir(exist_ok=True)
        for k, v in items.items():
            v.pop('pass_', None)
            (OUT/col/f'{k}.json').write_text(json.dumps(v))
            print(col, k)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--status', default=None)
    build(p.parse_args().status)
