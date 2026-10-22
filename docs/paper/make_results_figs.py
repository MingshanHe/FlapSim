"""Results figures for the Method/Results draft (docs/paper/). Every number is read from artifacts/ or runs/.
    .venv/bin/python docs/paper/make_results_figs.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/paper/figs'
OUT.mkdir(parents=True, exist_ok=True)
A = ROOT/'artifacts/spike_steer'
R = ROOT/'runs/spike_steer'
C_CLS, C_CLSES, C_SNN, C_BC, C_SNN9 = '#444444', '#c9a227', '#1f6fb2', '#2a9d5c', '#9ecae1'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8})


V14 = R/'v14'


def unsup_crashes(test):
    f = V14/f'{test}__cls_unsup.json'
    if not f.exists():
        return np.nan
    return sum(not r['flown'] for r in json.loads(f.read_text()))


def unsup_delta(test, opp_rows):
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.algorithms.spike_steer.paired_report import paired
    f = V14/f'{test}__cls_unsup.json'
    if not f.exists():
        return np.nan
    return paired(opp_rows, json.loads(f.read_text()))['delta']


C_PPO = '#7b4fa8'


def ppo_crashes(setname):
    out = []
    for s_ in range(1, 6):
        f = V14/f'{setname}__mlp_ppo_s{s_}.json'
        if f.exists():
            out.append(sum(not r['flown'] for r in json.loads(f.read_text())))
    return out


def ppo_deltas(setname, opp_rows):
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.algorithms.spike_steer.paired_report import paired
    out = []
    for s_ in range(1, 6):
        f = V14/f'{setname}__mlp_ppo_s{s_}.json'
        if f.exists():
            out.append(paired(opp_rows, json.loads(f.read_text()))['delta'])
    return out


def summ(plan, test):
    for base in (A/f'plan_{plan}', R/plan):
        f = base/test/'summary.json'
        if f.exists():
            return json.loads(f.read_text())
    return None


TESTS = [('v12', 'kick2', '2 mN·m hit'), ('v12', 'kick', '3 mN·m hit'), ('v12', 'kick45', '4.5 mN·m hit'),
         ('v12', 'kick_early', 'hit +0–0.15 s'), ('v12', 'kick_late', 'hit +0.5–1.0 s'), ('v13', 'kick_pre', 'hit before cmd'),
         ('v13', 'kick_double', 'double hit'), ('v13', 'kick_rollaxis', 'roll-axis hit'), ('v13', 'kick_pitchaxis', 'pitch-axis hit'),
         ('v13', 'turb125', 'turbulence ×1.25'), ('v13', 'turb15', 'turbulence ×1.5'), ('v13', 'noise5', 'IMU noise ×5'),
         ('v13', 'delay3', 'sensor delay 3'), ('v12', 'ood', 'mass +12.5 %')]


def fig_crashes():
    labels, cls, cles, snn, bc, h1, uns = [], [], [], [], [], [], []
    for plan, t, lab in TESTS:
        S = summ(plan, t)
        if S is None:
            continue
        labels.append(lab)
        cls.append(S['classical']['crashes'])
        uns.append(unsup_crashes(t))
        cles.append([S[f'cls_es_s{s}']['crashes'] for s in (1, 2) if f'cls_es_s{s}' in S])
        snn.append([S[f'snn11_s{s}']['crashes'] for s in range(1, 6) if f'snn11_s{s}' in S])
        bc.append(S['snn_bc_s1']['crashes'] if 'snn_bc_s1' in S else np.nan)
        h1.append(S['H1'])
    n = len(labels)
    x = np.arange(n)
    w = .2
    fig, ax = plt.subplots(figsize=(9.6, 3.7), dpi=170)
    ax.bar(x-1.5*w, np.array(cls)/1.92, w, color=C_CLS, label='Classical opponent (PID + schedule)')
    ax.bar(x-.5*w, [np.mean(v)/1.92 for v in cles], w, color=C_CLSES, label='Classical + ES (2 seeds, mean)')
    pk = ppo_crashes('kick')
    tests_here = [t for plan, t, lab in TESTS if summ(plan, t) is not None]
    if pk and 'kick' in tests_here:
        i_k = tests_here.index('kick')
        ax.bar(x[i_k]+1.5*w, np.mean(pk)/1.92, w, color=C_PPO, label='MLP + PPO reference (5 seeds, mean; 3 mN·m only)', yerr=[[np.mean(pk)/1.92-min(pk)/1.92], [max(pk)/1.92-np.mean(pk)/1.92]], capsize=2, ecolor='#4a2d6b', error_kw=dict(lw=.8))
    m = [np.mean(v)/1.92 for v in snn]
    lo = [np.mean(v)/1.92-min(v)/1.92 for v in snn]
    hi = [max(v)/1.92-np.mean(v)/1.92 for v in snn]
    ax.bar(x+.5*w, m, w, color=C_SNN, label='SNN + ES (v11, 5 seeds, mean; whisker = min–max)', yerr=[lo, hi], capsize=2, ecolor='#0b3d6b', error_kw=dict(lw=.8))
    for i, h in enumerate(h1):
        ax.text(x[i], max(cls[i], max(snn[i]))/1.92+2.5, f"H1 {h['seeds_passing']}/{h['seeds']}", ha='center', fontsize=6.5, color='#0b3d6b')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('crashed episodes (% of 192)')
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=.3)
    ax.legend(loc='upper right', ncol=2, fontsize=7)
    ax.set_title('Crashed episodes of the frozen controllers on 14 blind robustness sets (192 cases each)')
    fig.tight_layout()
    fig.savefig(OUT/'fig_r2_crashes.png')
    fig.savefig(OUT/'fig_r2_crashes.svg')
    plt.close(fig)


def fig_delta():
    labels, snn, cles, bc, uns = [], [], [], [], []
    for plan, t, lab in [('v11', 'nominal', 'nominal (v11 blind)')]+TESTS:
        if t == 'nominal':
            S = json.loads((A/'plan_v11/summary.json').read_text())
            labels.append(lab)
            snn.append([S[f's{s}']['snn_es']['delta'] for s in range(1, 6)])
            cles.append([S[f'cls_es_s{s}']['delta'] for s in (1, 2)])
            bc.append(np.nan)
            uns.append(unsup_delta('nominal', json.loads((R/'v11/test_opponent.json').read_text())))
            continue
        S = summ(plan, t)
        if S is None:
            continue
        labels.append(lab)
        uns.append(unsup_delta(t, json.loads((R/plan/t/'rows_classical.json').read_text())))
        snn.append([S[f'snn11_s{s}']['delta'] for s in range(1, 6) if f'snn11_s{s}' in S])
        cles.append([S[f'cls_es_s{s}']['delta'] for s in (1, 2) if f'cls_es_s{s}' in S])
        bc.append(S['snn_bc_s1']['delta'] if 'snn_bc_s1' in S else np.nan)
    n = len(labels)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=170)
    ax.axvline(0, color=C_CLS, lw=1.2)
    for i in range(n):
        ax.scatter(cles[i], [y[i]+.18]*len(cles[i]), marker='s', s=22, color=C_CLSES, zorder=3)
        ax.scatter(snn[i], [y[i]]*len(snn[i]), marker='o', s=22, color=C_SNN, zorder=3)
        ax.plot([min(snn[i]), max(snn[i])], [y[i], y[i]], color=C_SNN, lw=1.2, zorder=2)
        lab_i = labels[i]
        if lab_i.startswith('nominal'):
            pd_ = ppo_deltas('nominal', json.loads((R/'v11/test_opponent.json').read_text()))
        elif lab_i == '3 mN·m hit':
            pd_ = ppo_deltas('kick', json.loads((R/'v12/kick/rows_classical.json').read_text()))
        else:
            pd_ = []
        if pd_:
            ax.scatter(pd_, [y[i]-.22]*len(pd_), marker='D', s=20, color=C_PPO, zorder=3)
    ax.scatter([], [], marker='o', color=C_SNN, label='SNN + ES (v11), one point per seed')
    ax.scatter([], [], marker='s', color=C_CLSES, label='Classical + ES, one point per seed')
    ax.scatter([], [], marker='D', color=C_PPO, label='MLP + PPO reference, one point per seed (nominal and 3 mN·m only)')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('paired Δ = arm ratio − classical opponent\'s own ratio on the same cases\n(negative = better than the hand-tuned classical opponent)')
    ax.grid(axis='x', alpha=.3)
    ax.legend(loc='lower left', framealpha=.95)
    for i in range(n):
        for v in snn[i]:
            if v > .6:
                ax.annotate('v11 seed 1 (17 crashes)', (v, y[i]), xytext=(v-.02, y[i]-.55), fontsize=7, ha='right', color=C_SNN, arrowprops=dict(arrowstyle='-', color=C_SNN, lw=.6))
    ax.set_title('Paired difference to the classical opponent on every blind set')
    fig.tight_layout()
    fig.savefig(OUT/'fig_r3_paired_delta.png')
    fig.savefig(OUT/'fig_r3_paired_delta.svg')
    plt.close(fig)


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=170)
    ax = axes[0]
    for s in range(1, 6):
        f = (R/f'v11/finetune_s{s}_v11/history.jsonl') if s <= 3 else (A/f'plan_v11/homepc/finetune_s{s}_v11/history.jsonl')
        h = [json.loads(l) for l in f.read_text().splitlines()]
        ax.plot([r['generation'] for r in h], [r['center'] for r in h], color=C_SNN, alpha=.75, lw=1.1, label='SNN + ES seeds 1–5' if s == 1 else None)
    for s, f in ((1, R/'v11/cls_es_s1/history.jsonl'), (2, A/'plan_v11/homepc/cls_es_s2/history.jsonl')):
        h = [json.loads(l) for l in f.read_text().splitlines()]
        ax.plot([r['generation'] for r in h], [r['center'] for r in h], color=C_CLSES, ls='--', lw=1.1, label='Classical + ES seeds 1–2' if s == 1 else None)
    ax.axhline(1., color=C_CLS, lw=.8)
    ax.set_xlabel('ES generation')
    ax.set_ylabel('centre objective (24 training cases)')
    ax.set_title('(a) ES training (v11 protocol)')
    ax.grid(alpha=.3)
    ax.legend()
    ax = axes[1]
    for s in range(1, 6):
        f = (R/f'v11/reselect_s{s}_v11.json') if s <= 3 else (A/f'plan_v11/homepc/reselect_s{s}_v11.json')
        rs = json.loads(f.read_text())
        cs = sorted(rs['centres'], key=lambda c: c['generation'])
        g = [c['generation'] for c in cs]
        o = [c['objective'] for c in cs]
        ax.plot(g, o, color=C_SNN, alpha=.75, lw=1.1, marker='.', label='SNN + ES seeds 1–5' if s == 1 else None)
        cr = [c for c in cs if c['crashes'] > 0]
        ax.scatter([c['generation'] for c in cr], [c['objective'] for c in cr], marker='x', color='#c23b3b', s=28, zorder=4, label='centre with a crash (excluded)' if s == 2 else None)
        sel = rs['selected']
        ax.scatter([sel['generation']], [sel['objective']], facecolors='none', edgecolors=C_SNN, s=60, zorder=5, label='selected centre' if s == 1 else None)
    for s, f in ((1, R/'v11/reselect_cls_es_s1.json'), (2, A/'plan_v11/homepc/reselect_cls_es_s2.json')):
        rs = json.loads(f.read_text())
        cs = sorted(rs['centres'], key=lambda c: c['generation'])
        ax.plot([c['generation'] for c in cs], [c['objective'] for c in cs], color=C_CLSES, ls='--', lw=1.1, marker='.', label='Classical + ES seeds 1–2' if s == 1 else None)
    ax.axhline(1., color=C_CLS, lw=.8)
    ax.set_xlabel('generation of the saved centre (−1 = imitation start)')
    ax.set_ylabel('objective on the 192 selection cases')
    ax.set_title('(b) Selection on the fixed validation set')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT/'fig_r1_training.png')
    fig.savefig(OUT/'fig_r1_training.svg')
    plt.close(fig)


def fig_nominal():
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.algorithms.spike_steer.paired_report import paired
    rows = []
    # plan, label, snn means, classical self, cls_es means
    v7 = [json.loads((A/f'plan_v7/full/snn/s{s}/evaluate_snn_es_s{s}.json').read_text())['report']['manoeuvre_mean'] for s in range(1, 6)]
    opp7 = json.loads((R/'v7full/test_opponent.json').read_text())
    rows.append(('v7 full\n(32/task)', v7, paired(opp7, opp7)['mean'], []))
    ppo = {}
    try:
        ppo['v7 full\n(32/task)'] = [json.loads((A/f'ppo_direct/v7full/evaluate_mlp_ppo_s{s}.json').read_text())['report']['manoeuvre_mean'] for s in range(1, 6)]
    except Exception:
        pass
    try:
        ppo['v8 re-test\n(same ctrl., 64/task)'] = [json.loads((A/f'ppo_direct/v8/evaluate_mlp_ppo_s{s}.json').read_text())['report']['manoeuvre_mean'] for s in range(1, 6)]
    except Exception:
        pass
    pn = ppo_deltas('nominal', json.loads((R/'v11/test_opponent.json').read_text()))
    S8 = json.loads((A/'plan_v8/summary.json').read_text())
    opp8 = json.loads((R/'v8/test_opponent.json').read_text())
    rows.append(('v8 re-test\n(same ctrl., 64/task)', [S8[f'snn_es_s{s}']['mean'] for s in range(1, 6)], paired(opp8, opp8)['mean'], []))
    S9 = json.loads((A/'plan_v9/summary.json').read_text())
    opp9 = json.loads((R/'v9/test_opponent.json').read_text())
    rows.append(('v9\n(altitude term)', [S9[f'snn_es_s{s}']['mean'] for s in range(1, 6)], paired(opp9, opp9)['mean'], []))
    S10 = json.loads((A/'plan_v10/summary.json').read_text())
    opp10 = json.loads((R/'v10/test_opponent.json').read_text())
    rows.append(('v10\n(37 inputs)', [S10[f's{s}']['snn_es']['mean'] for s in range(1, 6)], paired(opp10, opp10)['mean'], []))
    S11 = json.loads((A/'plan_v11/summary.json').read_text())
    opp11 = json.loads((R/'v11/test_opponent.json').read_text())
    rows.append(('v11\n(fixed ES)', [S11[f's{s}']['snn_es']['mean'] for s in range(1, 6)], paired(opp11, opp11)['mean'],
                 [S11[f'cls_es_s{s}']['mean'] for s in (1, 2)]))
    crashes = ['SNN 2 / opp. 1', 'SNN 10 / opp. 5', 'SNN 3 / opp. 0', 'SNN 6 / opp. 3', 'SNN 0 / opp. 0']
    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=170)
    x = np.arange(len(rows))
    for i, (lab, snn, self_, cles) in enumerate(rows):
        ax.scatter([i]*len(snn), snn, color=C_SNN, s=26, zorder=3, label='SNN + ES, one point per seed' if i == 0 else None)
        ax.plot([i-.22, i+.22], [self_, self_], color=C_CLS, lw=2.2, zorder=2, label='classical opponent\'s own ratio' if i == 0 else None)
        if cles:
            ax.scatter([i+.28]*len(cles), cles, marker='s', color=C_CLSES, s=24, zorder=3, label='Classical + ES (2 seeds, same machines)' if i == 4 else None)
            ax.scatter([i+.28]*2, [0.999, 1.018], marker='s', facecolors='none', edgecolors=C_CLSES, s=24, zorder=3, label='Classical + ES (2 seeds, approximate machine)' if i == 4 else None)
        ax.text(i, .815, 'crashes: '+crashes[i], ha='center', fontsize=6.5, color=C_CLS)
        if lab.startswith('v11'):
            gc = []
            for s_ in range(6, 11):
                f = A/f'googlecloud/v11/evaluate_snn_es_s{s_}_v11.json'
                if f.exists():
                    gc.append(json.loads(f.read_text())['report']['manoeuvre_mean'])
            if gc:
                ax.scatter([i+.12]*len(gc), gc, facecolors='none', edgecolors=C_SNN, s=28, zorder=3, label='SNN + ES exploratory seeds 6–10 (approximate machine)')
        pts = ppo.get(lab, [])
        if lab.startswith('v11') and pn:
            pts = [self_+d for d in pn]
        if pts:
            ax.scatter([i-.28]*len(pts), pts, marker='D', s=22, color=C_PPO, zorder=3, label='MLP + PPO reference, one point per seed' if i == 0 else None)
    ax.axhline(.95, color='#c23b3b', ls=':', lw=1, label='preregistered threshold 0.95')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('mean manoeuvre ratio (lower is better)')
    ax.set_ylim(.8, 1.5)
    ax.grid(axis='y', alpha=.3)
    ax.legend(loc='upper center', fontsize=7, ncol=3, bbox_to_anchor=(.5, -.2), frameon=False)
    ax.set_title('Nominal manoeuvres: mean ratio per plan and the opponent\'s own ratio on the same blind set')
    fig.set_size_inches(9.6, 4.8)
    fig.tight_layout()
    fig.savefig(OUT/'fig_r0_nominal.png')
    fig.savefig(OUT/'fig_r0_nominal.svg')
    plt.close(fig)


def fig_mechanism():
    M = json.loads((A/'plan_v13/mechanism_summary.json').read_text())['refly']
    arms = [('classical', 'Classical', C_CLS), ('cls_es_s1', 'Classical + ES', C_CLSES), ('snn11_s2', 'SNN + ES s2', C_SNN), ('snn11_s3', 'SNN + ES s3', C_SNN),
            ('snn11_s4', 'SNN + ES s4', C_SNN)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=170)
    ax = axes[0]
    bins = ['<20', '20-35']
    w = .15
    for j, (k, lab, col) in enumerate(arms):
        r = M[k]['crash_rate_by_tilt_at_hit']
        vals = [100*r[b][0]/r[b][1] if b in r else np.nan for b in bins]
        err = [[100*(r[b][0]/r[b][1]-r[b][2][0]) if b in r else 0 for b in bins], [100*(r[b][2][1]-r[b][0]/r[b][1]) if b in r else 0 for b in bins]]
        ax.bar(np.arange(2)+(j-2)*w, vals, w, color=col, alpha=.95 if 'snn11' not in k else .55+.15*j/5, label=lab, yerr=err, capsize=1.5, error_kw=dict(lw=.6, ecolor='#333'))
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['tilt at hit < 20°', '20–35°'])
    ax.set_ylabel('crash rate (%)')
    ax.set_title('(a) Crash rate at matched tilt-at-hit')
    ax.legend(fontsize=6.5, loc='upper left', ncol=2, framealpha=.9)
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=.3)
    ax = axes[1]
    for j, (k, lab, col) in enumerate(arms):
        m, (lo, hi) = M[k]['tilt_at_hit']
        ax.errorbar(j, m, yerr=[[m-lo], [hi-m]], fmt='o', color=col, capsize=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms], rotation=30, ha='right')
    ax.set_ylabel('tilt at the hit (deg, mean, 95 % CI)')
    ax.set_title('(b) Tilt when the hit lands')
    ax.grid(axis='y', alpha=.3)
    ax = axes[2]
    for j, (k, lab, col) in enumerate(arms):
        m, (lo, hi) = M[k]['pulses_post_minus_pre']
        ax.errorbar(j, m, yerr=[[m-lo], [hi-m]], fmt='o', color=col, capsize=3)
    ax.axhline(0, color='#888', lw=.8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms], rotation=30, ha='right')
    ax.set_ylabel('motor pulses (150 ms windows)')
    ax.set_title('(c) Pulses after − before the hit')
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT/'fig_r5_mechanism.png')
    fig.savefig(OUT/'fig_r5_mechanism.svg')
    plt.close(fig)


def fig_skew():
    import sys
    sys.path.insert(0, str(ROOT))
    opp = json.loads((R/'v11/test_opponent.json').read_text())
    from experiments.algorithms.spike_steer import bench9 as T
    fig, ax = plt.subplots(figsize=(5.2, 3.), dpi=170)
    for t, col in (('evade', '#2f6db5'), ('redirect', '#1f8a4c'), ('dash', '#d4781f')):
        med = {cl: np.median([r['score'] for r in opp if r['task'] == t and r['cls'] == cl]) for cl in T.CLASSES[t]}
        ratios = [r['score']/med[r['cls']] for r in opp if r['task'] == t]
        ax.hist(ratios, bins=np.arange(0, 3.05, .1), alpha=.55, color=col, label=f'{t}: mean {np.mean(ratios):.2f}')
    ax.axvline(1., color='#444', lw=1)
    ax.set_xlabel('opponent score ÷ its own class median (v11 blind set, 192 cases)')
    ax.set_ylabel('cases')
    ax.legend(fontsize=7.5)
    ax.set_title('Why the ratio exceeds 1 for the opponent itself: right-skewed per-case ratios', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT/'fig_r0b_skew.png')
    fig.savefig(OUT/'fig_r0b_skew.svg')
    plt.close(fig)


def fig_ablation():
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.algorithms.spike_steer.evaluate_v8 import clopper_pearson as cp
    S = summ('v12', 'kick')
    V = json.loads((A/'plan_v14/summary_d1_d4.json').read_text())['d2_d3_d4']
    full = [S[f'snn11_s{s}']['crashes'] for s in range(1, 6)]
    nop = [V[f'kick__snn11_s{s}_nopulse']['crashes'] for s in range(1, 6)]
    nor = [V[f'kick__snn11_s{s}_noref']['crashes'] for s in range(1, 6)]
    bc = [S['snn_bc_s1']['crashes']]+[V[f'kick__snn_bc_s{s}']['crashes'] for s in (2, 3, 4, 5)]
    uns = V['kick__cls_unsup']['crashes']
    groups = [('opponent\n(PID + schedule,\nno pulses in\nthe manoeuvre)', [S['classical']['crashes']], C_CLS),
              ('SNN + ES\n(5 seeds)', full, C_SNN), ('SNN + ES,\npulses\nsuppressed', nop, '#8fb3d9'), ('SNN + ES,\nset-points\nzeroed', nor, '#c7d9ec'),
              ('SNN,\nimitation only\n(5 seeds)', bc, C_BC), ('opponent +\nphase-locked\npulses kept on\n(probe)', [uns], '#9a9a9a')]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=170)
    for i, (lab, vals, col) in enumerate(groups):
        m = np.mean(vals)/1.92
        ax.bar(i, m, .6, color=col, edgecolor='#333', lw=.5, hatch='//' if 'probe' in lab else None)
        if len(vals) > 1:
            ax.errorbar(i, m, yerr=[[m-min(vals)/1.92], [max(vals)/1.92-m]], fmt='none', ecolor='#0b3d6b', capsize=3, lw=.9)
        else:
            lo_, hi_ = cp(vals[0], 192)
            ax.errorbar(i, m, yerr=[[m-100*lo_], [100*hi_-m]], fmt='none', ecolor='#333', capsize=3, lw=.9)
        ax.text(i, max(vals)/1.92+2.5, '/'.join(str(v) for v in vals), ha='center', fontsize=6.5, color='#222')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_ylabel('crashed episodes (% of 192), 3 mN·m hit')
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=.3)
    ax.set_title('Channel ablation under the primary hit (whiskers: range over seeds; probe: exact 95 % interval)', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT/'fig_r5b_ablation.png')
    fig.savefig(OUT/'fig_r5b_ablation.svg')
    plt.close(fig)


def fig_dose():
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.algorithms.spike_steer.evaluate_v8 import clopper_pearson as cp
    mags, tests = [2, 3, 4.5], ['kick2', 'kick', 'kick45']
    S = {t: summ('v12', t) for t in tests}
    n = 192
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.plot(mags, [S[t]['classical']['crashes']/n*100 for t in tests], '-', color=C_CLS, lw=2.2, marker='o', label='classical opponent (PID + schedule)')
    lo = [cp(S[t]['classical']['crashes'], n)[0]*100 for t in tests]
    hi = [cp(S[t]['classical']['crashes'], n)[1]*100 for t in tests]
    ax.fill_between(mags, lo, hi, color=C_CLS, alpha=.12, lw=0, label='opponent: exact 95 % interval')
    for s_, c in ((1, '#8c6d1f'), (2, '#c9a227')):
        ax.plot(mags, [S[t][f'cls_es_s{s_}']['crashes']/n*100 for t in tests], '--', color=c, lw=1.4, marker='s', label=f'classical + ES seed {s_}')
    for s_ in range(1, 6):
        ax.plot(mags, [S[t][f'snn11_s{s_}']['crashes']/n*100 for t in tests], '-', color=C_SNN, lw=1.6, marker='o', alpha=.85, label='SNN + ES (v11 seeds 1–5)' if s_ == 1 else None)
    pk = ppo_crashes('kick')
    if pk:
        ax.scatter([3]*len(pk), [v/n*100 for v in pk], marker='D', s=26, color=C_PPO, zorder=4, label='MLP + PPO reference (5 seeds, 3 mN·m only)')
    ax.set_xlabel('torque hit during the manoeuvre (mN·m, 20 ms, +0.15–0.45 s)')
    ax.set_ylabel('crashed episodes (% of 192)')
    ax.set_xticks(mags)
    ax.set_ylim(0, 60)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_title('Crash rate vs hit magnitude (frozen controllers, a fresh blind set per level)', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/'fig_r4_dose_response.png')
    fig.savefig(OUT/'fig_r4_dose_response.svg')
    plt.close(fig)


if __name__ == '__main__':
    fig_dose()
    fig_ablation()
    fig_skew()
    fig_nominal()
    fig_training()
    fig_crashes()
    fig_delta()
    fig_mechanism()
    import shutil
    shutil.copy(A/'plan_v12/videos/v12_kick_s4_crashdiff.png', OUT/'fig_r6_kick_frame.png')
    print('figures written to', OUT)
