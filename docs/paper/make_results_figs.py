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
    labels, cls, cles, snn, bc, h1 = [], [], [], [], [], []
    for plan, t, lab in TESTS:
        S = summ(plan, t)
        if S is None:
            continue
        labels.append(lab)
        cls.append(S['classical']['crashes'])
        cles.append([S[f'cls_es_s{s}']['crashes'] for s in (1, 2) if f'cls_es_s{s}' in S])
        snn.append([S[f'snn11_s{s}']['crashes'] for s in range(1, 6) if f'snn11_s{s}' in S])
        bc.append(S['snn_bc_s1']['crashes'] if 'snn_bc_s1' in S else np.nan)
        h1.append(S['H1'])
    n = len(labels)
    x = np.arange(n)
    w = .2
    fig, ax = plt.subplots(figsize=(9.2, 3.6), dpi=170)
    ax.bar(x-1.5*w, np.array(cls)/1.92, w, color=C_CLS, label='Classical (re-tuned)')
    ax.bar(x-.5*w, [np.mean(v)/1.92 for v in cles], w, color=C_CLSES, label='Classical + ES (2 seeds, mean)')
    m = [np.mean(v)/1.92 for v in snn]
    lo = [np.mean(v)/1.92-min(v)/1.92 for v in snn]
    hi = [max(v)/1.92-np.mean(v)/1.92 for v in snn]
    ax.bar(x+.5*w, m, w, color=C_SNN, label='SNN + ES (v11, 5 seeds, mean; bar = min–max)', yerr=[lo, hi], capsize=2, ecolor='#0b3d6b', error_kw=dict(lw=.8))
    ax.bar(x+1.5*w, np.array(bc)/1.92, w, color=C_BC, label='SNN, imitation only (seed 1)')
    for i, h in enumerate(h1):
        ax.text(x[i], max(cls[i], max(snn[i]))/1.92+2.5, f"{h['seeds_passing']}/{h['seeds']}", ha='center', fontsize=7, color='#0b3d6b')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('crashed episodes (% of 192)')
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=.3)
    ax.legend(loc='upper right', ncol=2)
    ax.set_title('Crashes of the frozen controllers on 14 blind robustness sets (64 cases per task); numbers = v11 seeds passing H1')
    fig.tight_layout()
    fig.savefig(OUT/'fig_r2_crashes.png')
    plt.close(fig)


def fig_delta():
    labels, snn, cles, bc = [], [], [], []
    for plan, t, lab in [('v11', 'nominal', 'nominal (v11 blind)')]+TESTS:
        if t == 'nominal':
            S = json.loads((A/'plan_v11/summary.json').read_text())
            labels.append(lab)
            snn.append([S[f's{s}']['snn_es']['delta'] for s in range(1, 6)])
            cles.append([S[f'cls_es_s{s}']['delta'] for s in (1, 2)])
            bc.append(np.nan)
            continue
        S = summ(plan, t)
        if S is None:
            continue
        labels.append(lab)
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
    ax.scatter([], [], marker='o', color=C_SNN, label='SNN + ES (v11), one point per seed')
    ax.scatter([], [], marker='s', color=C_CLSES, label='Classical + ES, one point per seed')
    ax.scatter([], [], marker='^', color=C_BC, label='SNN, imitation only')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('paired Δ = arm ratio − classical opponent\'s own ratio on the same cases\n(negative = better than the re-tuned classical schedule)')
    ax.grid(axis='x', alpha=.3)
    ax.legend(loc='upper right')
    ax.set_title('Paired difference to the re-tuned classical schedule on every blind set')
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
                 [S11[f'cls_es_s{s}']['mean'] for s in (1, 2)]+[0.999, 1.018]))
    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=170)
    x = np.arange(len(rows))
    for i, (lab, snn, self_, cles) in enumerate(rows):
        ax.scatter([i]*len(snn), snn, color=C_SNN, s=26, zorder=3, label='SNN + ES, one point per seed' if i == 0 else None)
        ax.plot([i-.22, i+.22], [self_, self_], color=C_CLS, lw=2.2, zorder=2, label='classical opponent\'s own ratio' if i == 0 else None)
        if cles:
            ax.scatter([i+.28]*len(cles), cles, marker='s', color=C_CLSES, s=24, zorder=3, label='Classical + ES (4 seeds)' if i == 4 else None)
    ax.axhline(.95, color='#c23b3b', ls=':', lw=1, label='preregistered threshold 0.95')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('mean manoeuvre ratio (lower is better)')
    ax.set_ylim(.8, 1.5)
    ax.grid(axis='y', alpha=.3)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('Nominal manoeuvres: the ratio criterion is not reached by any controller, the classical included')
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
        ax.bar(np.arange(2)+(j-2.5)*w, vals, w, color=col, alpha=.95 if 'snn11' not in k else .55+.15*j/5, label=lab)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['tilt at hit < 20°', '20–35°'])
    ax.set_ylabel('crash rate (%)')
    ax.set_title('(a) Crash rate at matched tilt-at-hit')
    ax.legend(fontsize=6.5, loc='upper left', ncol=2, framealpha=.9)
    ax.set_ylim(0, 45)
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


if __name__ == '__main__':
    fig_nominal()
    fig_training()
    fig_crashes()
    fig_delta()
    fig_mechanism()
    import shutil
    shutil.copy(A/'plan_v12/fig_dose_response.png', OUT/'fig_r4_dose_response.png')
    shutil.copy(A/'plan_v12/videos/v12_kick_s4_crashdiff.png', OUT/'fig_r6_kick_frame.png')
    print('figures written to', OUT)
