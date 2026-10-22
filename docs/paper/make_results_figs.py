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
    w = .17
    fig, ax = plt.subplots(figsize=(9.6, 3.7), dpi=170)
    ax.bar(x-2*w, np.array(cls)/1.92, w, color=C_CLS, label='Classical opponent (preregistered; pulses off in segments)')
    ax.bar(x-w, np.array(uns)/1.92, w, color='#9a9a9a', hatch='//', edgecolor='#444', lw=.4, label='Classical, pulses kept on (plan v14)')
    ax.bar(x, [np.mean(v)/1.92 for v in cles], w, color=C_CLSES, label='Classical + ES (2 seeds, mean)')
    m = [np.mean(v)/1.92 for v in snn]
    lo = [np.mean(v)/1.92-min(v)/1.92 for v in snn]
    hi = [max(v)/1.92-np.mean(v)/1.92 for v in snn]
    ax.bar(x+w, m, w, color=C_SNN, label='SNN + ES (v11, 5 seeds, mean; whisker = min–max)', yerr=[lo, hi], capsize=2, ecolor='#0b3d6b', error_kw=dict(lw=.8))
    ax.bar(x+2*w, np.array(bc)/1.92, w, color=C_BC, label='SNN, imitation only (seed 1)')
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
        if not np.isnan(bc[i]):
            ax.scatter([bc[i]], [y[i]-.18], marker='^', s=26, color=C_BC, zorder=3)
        if not np.isnan(uns[i]):
            ax.scatter([uns[i]], [y[i]+.32], marker='D', s=22, color='#9a9a9a', edgecolors='#444', lw=.5, zorder=3)
    ax.scatter([], [], marker='D', color='#9a9a9a', edgecolors='#444', label='Classical, pulses kept on (v14)')
    ax.scatter([], [], marker='o', color=C_SNN, label='SNN + ES (v11), one point per seed')
    ax.scatter([], [], marker='s', color=C_CLSES, label='Classical + ES, one point per seed')
    ax.scatter([], [], marker='^', color=C_BC, label='SNN, imitation only')
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
    ax.set_title('Paired difference to the hand-tuned classical schedule on every blind set')
    fig.tight_layout()
    fig.savefig(OUT/'fig_r3_paired_delta.png')
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
    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=170)
    x = np.arange(len(rows))
    for i, (lab, snn, self_, cles) in enumerate(rows):
        ax.scatter([i]*len(snn), snn, color=C_SNN, s=26, zorder=3, label='SNN + ES, one point per seed' if i == 0 else None)
        ax.plot([i-.22, i+.22], [self_, self_], color=C_CLS, lw=2.2, zorder=2, label='classical opponent\'s own ratio' if i == 0 else None)
        if cles:
            ax.scatter([i+.28]*len(cles), cles, marker='s', color=C_CLSES, s=24, zorder=3, label='Classical + ES (2 seeds, same machines)' if i == 4 else None)
            ax.scatter([i+.28]*2, [0.999, 1.018], marker='s', facecolors='none', edgecolors=C_CLSES, s=24, zorder=3, label='Classical + ES (2 seeds, approximate machine)' if i == 4 else None)
        ax.text(i, .815, 'crashes: '+crashes[i], ha='center', fontsize=6.5, color=C_CLS)
        if i == 4:
            u = self_+unsup_delta('nominal', opp11)
            ax.scatter([i-.28], [u], marker='D', s=30, color='#9a9a9a', edgecolors='#444', lw=.6, zorder=4, label='Classical, pulses kept on (v14)')
    ax.axhline(.95, color='#c23b3b', ls=':', lw=1, label='preregistered threshold 0.95')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('mean manoeuvre ratio (lower is better)')
    ax.set_ylim(.8, 1.5)
    ax.grid(axis='y', alpha=.3)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('Nominal manoeuvres: mean ratio per plan and the opponent\'s own ratio on the same blind set')
    fig.tight_layout()
    fig.savefig(OUT/'fig_r0_nominal.png')
    plt.close(fig)


def fig_mechanism():
    M = json.loads((A/'plan_v13/mechanism_summary.json').read_text())['refly']
    arms = [('classical', 'Classical', C_CLS), ('cls_es_s1', 'Classical + ES', C_CLSES), ('snn11_s2', 'SNN + ES s2', C_SNN), ('snn11_s3', 'SNN + ES s3', C_SNN),
            ('snn11_s4', 'SNN + ES s4', C_SNN), ('snn_bc_s1', 'SNN imitation', C_BC)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=170)
    ax = axes[0]
    bins = ['<20', '20-35']
    w = .13
    for j, (k, lab, col) in enumerate(arms):
        r = M[k]['crash_rate_by_tilt_at_hit']
        vals = [100*r[b][0]/r[b][1] if b in r else np.nan for b in bins]
        err = [[100*(r[b][0]/r[b][1]-r[b][2][0]) if b in r else 0 for b in bins], [100*(r[b][2][1]-r[b][0]/r[b][1]) if b in r else 0 for b in bins]]
        ax.bar(np.arange(2)+(j-2.5)*w, vals, w, color=col, alpha=.95 if 'snn11' not in k else .55+.15*j/5, label=lab, yerr=err, capsize=1.5, error_kw=dict(lw=.6, ecolor='#333'))
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
    plt.close(fig)


if __name__ == '__main__':
    fig_skew()
    fig_nominal()
    fig_training()
    fig_crashes()
    fig_delta()
    fig_mechanism()
    import shutil
    shutil.copy(A/'plan_v12/fig_dose_response.png', OUT/'fig_r4_dose_response.png')
    shutil.copy(A/'plan_v12/videos/v12_kick_s4_crashdiff.png', OUT/'fig_r6_kick_frame.png')
    print('figures written to', OUT)
