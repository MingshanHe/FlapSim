"""Tables and figures for the paper from the campaign outputs (missing inputs are skipped).

Survival rates carry Wilson 95 % intervals; continuous differences are paired by case
(both controllers flew the full episode) with bootstrap 95 % intervals; learned arms are
summarized per training seed with an exact two-sided sign test against the tuned hand law.

    python -m experiments.algorithms.wingbeat_snn.paper_stats --input artifacts/wingbeat_paper --output artifacts/wingbeat_paper/report
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import numpy as np

REF = 'B2plus_retuned'


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),)*2
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0., centre-half), min(1., centre+half)


def sign_test(k, n):
    """Two-sided exact binomial test of k successes in n with p = 1/2."""
    if n == 0:
        return float('nan')
    tail = sum(math.comb(n, i) for i in range(0, min(k, n-k)+1))/2**n
    return min(1., 2*tail)


def bootstrap(diffs, reps=4000, seed=0):
    d = np.asarray(diffs, float)
    if len(d) < 2:
        return (float(d.mean()) if len(d) else float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (reps, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def flown(r):
    return r['termination'] == 'time_limit'


def index(rows, name_key='controller'):
    out = defaultdict(dict)
    for r in rows:
        out[(r[name_key], str(r['gust_peak_m_s']))][r['seed']] = r
    return out


def survival_table(rows, names, levels, title):
    idx = index(rows)
    lines = [f'### {title}', '', '| controller | '+' | '.join(f'{l} m/s' for l in levels)+' |',
             '|---|'+'---|'*len(levels)]
    for n in names:
        cells = []
        for l in levels:
            eps = list(idx.get((n, l), {}).values())
            k = sum(map(flown, eps))
            lo, hi = wilson(k, len(eps))
            cells.append(f'{k}/{len(eps)} [{lo:.2f}, {hi:.2f}]' if eps else '-')
        lines.append(f'| {n} | '+' | '.join(cells)+' |')
    return lines


def paired_table(rows, pairs, levels, key, title):
    idx = index(rows)
    scale = 100. if key.endswith('_m') else 1.  # metres are reported in cm
    if scale != 1.:
        title = title.replace('(m)', '(cm)') if '(m)' in title else title+' (cm)'
    lines = [f'### {title}', '', '| comparison | '+' | '.join(f'{l} m/s' for l in levels)+' |',
             '|---|'+'---|'*len(levels)]
    for a, b in pairs:
        cells = []
        for l in levels:
            A, B = idx.get((a, l), {}), idx.get((b, l), {})
            diffs = [scale*(A[s][key]-B[s][key]) for s in A if s in B and flown(A[s]) and flown(B[s])]
            if not diffs:
                cells.append('-')
                continue
            m, lo, hi = bootstrap(diffs)
            better = sum(d < 0 for d in diffs)
            cells.append(f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}] ({better}/{len(diffs)} lower)')
        lines.append(f'| {a} − {b} | '+' | '.join(cells)+' |')
    return lines


def levels_of(rows):
    return sorted({str(r['gust_peak_m_s']) for r in rows}, key=float)


def figure_survival(rows, names, path, xlabel='peak fore-aft gust (m/s)'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    idx = index(rows)
    levels = levels_of(rows)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for n in names:
        x, y, lo, hi = [], [], [], []
        for l in levels:
            eps = list(idx.get((n, l), {}).values())
            if not eps:
                continue
            k = sum(map(flown, eps))
            a, b = wilson(k, len(eps))
            x.append(float(l)), y.append(k/len(eps)), lo.append(k/len(eps)-a), hi.append(b-k/len(eps))
        ax.errorbar(x, y, yerr=[lo, hi], marker='o', capsize=3, label=n)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('fraction completing 8 s')
    ax.set_ylim(-.05, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def seed_summary(rows, pattern, reference, levels, key='pitch_peak_deg'):
    """Arms named <arm>_s<seed>: survival per seed and paired differences to ``reference``."""
    idx = index(rows)
    arms = defaultdict(dict)
    for name in {r['controller'] for r in rows}:
        m = re.fullmatch(pattern, name)
        if m:
            arms[m.group(1)][int(m.group(2))] = name
    lines = ['| arm | seeds | '+' | '.join(f'survival {l}' for l in levels)+' | '+
             ' | '.join(f'Δ{key} vs {reference} at {l} (seeds lower)' for l in levels)+' |',
             '|---|---|'+'---|'*(2*len(levels))]
    for arm, seeds in sorted(arms.items()):
        surv, delta = [], []
        for l in levels:
            rates = [np.mean([flown(e) for e in idx[(n, l)].values()]) for n in seeds.values() if (n, l) in idx]
            surv.append(f'{np.mean(rates):.2f} ± {np.std(rates):.2f}' if rates else '-')
            per_seed = []
            ref = idx.get((reference, l), {})
            for n in seeds.values():
                A = idx.get((n, l), {})
                d = [A[s][key]-ref[s][key] for s in A if s in ref and flown(A[s]) and flown(ref[s])]
                if d:
                    per_seed.append(float(np.mean(d)))
            if per_seed:
                k = sum(x < 0 for x in per_seed)
                delta.append(f'{np.mean(per_seed):+.2f} ± {np.std(per_seed):.2f} ({k}/{len(per_seed)}, '
                             f'p={sign_test(k, len(per_seed)):.3f})')
            else:
                delta.append('-')
        lines.append(f'| {arm} | {len(seeds)} | '+' | '.join(surv)+' | '+' | '.join(delta)+' |')
    return lines


def section_e1(path):
    r = json.loads(path.read_text())
    s = r['summary']
    gimbal = s['gimbal_pitch_0.035']
    lines = ['## E1 open-loop authority (0.1 s burst, gimbal frozen)', '',
             f"paired cases: {r['seeds']}; wrapper equivalence max |Δqpos| = {r['equivalence_max_abs_qpos']}", '',
             '| input | Δ pitch rate (rad/s) | ratio to gimbal | Δ forward velocity (m/s) | ratio | peak current (A) |',
             '|---|---:|---:|---:|---:|---:|']
    for label in ['gimbal_pitch_0.035']+[k for k in s if k.startswith('split_') and ('psi90' in k or 'psi270' in k)]:
        e = s[label]
        lines.append(f"| {label} | {e['body_pitch_rate']:+.2f} | {abs(e['body_pitch_rate'])/abs(gimbal['body_pitch_rate']):.1f} | "
                     f"{e['forward_velocity']:+.3f} | {abs(e['forward_velocity'])/abs(gimbal['forward_velocity']):.1f} | "
                     f"{e['peak_current_A']:.2f} |")
    return lines


def section_phase(model_dir, closed_loop):
    lines = ['## E8 phase estimation', '']
    off = model_dir/'offline_test.json'
    if off.exists():
        r = json.loads(off.read_text())
        fam = defaultdict(list)
        for name, v in r.items():
            fam[re.sub(r'_s\d+$', '', name)].append(v)
        tuned = model_dir/'pll_tuned.json'
        if tuned.exists():
            fam['pll_tuned'].append(json.loads(tuned.read_text())['test'])
        lines += ['### offline, held-out flights (mean ± sd over seeds)', '',
                  '| estimator | seeds | RMS all (deg) | RMS gust | p95 gust | lock time (s) | never locked |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name, vs in sorted(fam.items()):
            def ms(k):
                x = [v[k] for v in vs if v.get(k) is not None]
                return f'{np.mean(x):.2f} ± {np.std(x):.2f}' if x else '-'
            lines.append(f'| {name} | {len(vs)} | {ms("rms_all")} | {ms("rms_gust")} | {ms("p95_gust")} | '
                         f'{ms("lock_time_s_mean")} | {sum(v["never_locked"] for v in vs)} |')
        lines.append('')
    if closed_loop.exists():
        rows = json.loads(closed_loop.read_text())['episodes']
        levels = levels_of(rows)
        names = sorted({r['controller'] for r in rows})
        lines += survival_table(rows, [n for n in names if not re.search(r'_s\d+$', n)], levels,
                                'closed loop, fixed phase sources (survival, Wilson 95 %)')
        lines += ['', '### closed loop, learned sources per seed vs default PLL (pitch peak)', '']
        lines += seed_summary(rows, r'law_(gru|snn_grad|snn_neat)_s(\d+)', 'law_pll', levels)
        spikes = [r['spikes_per_s'] for r in rows if 'snn_grad' in r['controller'] and r.get('spikes_per_s')]
        if spikes:
            lines += ['', f'SNN (gradient) spikes per second, whole network: {np.mean(spikes):.0f} ± {np.std(spikes):.0f}']
    return lines


def ops_note(model_dir):
    """Per 2 ms tick: GRU multiply-accumulates vs SNN dense input MACs + spike-driven accumulates."""
    for path in sorted(model_dir.glob('snn_grad_s*.json'))[:1]:
        spec = json.loads(path.read_text())
        hidden = len(spec['nodes'])-2
        return [f'Operation count per tick (proxy, not energy): GRU-32 ≈ {3*(8*32+32*32)+32*2} MAC; '
                f'SNN-{hidden}: {2*8*hidden} MAC for the analog input projection over two substeps, plus '
                f'(spikes per tick) × {hidden+2} accumulates.']
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    out = ['# Wingbeat-synchronous control: campaign tables', '',
           'Survival = episodes completing 8 s; [Wilson 95 % interval]. Paired Δ = mean over cases flown by both, '
           '[bootstrap 95 %], (cases where the first is lower).', '']
    if (a.input/'e1/p0_e1.json').exists():
        out += section_e1(a.input/'e1/p0_e1.json')+['']
    e2 = a.input/'e2_gust_curves.json'
    if e2.exists():
        rows = json.loads(e2.read_text())['episodes']
        levels = levels_of(rows)
        names = ['B1_pid', 'B2plus_hand_hp1s', REF]
        out += ['## E2 gust tolerance'] + ['']+survival_table(rows, names, levels, 'survival')+['']
        out += paired_table(rows, [(REF, 'B1_pid'), (REF, 'B2plus_hand_hp1s')], levels, 'pitch_peak_deg',
                            'pitch peak (deg), paired')+['']
        out += paired_table(rows, [(REF, 'B1_pid')], levels, 'height_rmse_m', 'height RMSE (m), paired')+['']
        fig = figure_survival(rows, names, a.output/'fig_gust_survival.png')
        if fig:
            out += [f'![gust survival]({fig.name})', '']
    for name, title in (('e3_lag_curve.json', 'E3 unknown actuator lag'), ('e7_h5_seeds.json', 'E7 lag-trained networks x seeds')):
        path = a.input/name
        if path.exists():
            r = json.loads(path.read_text())
            out += [f'## {title}', '', '| controller | lag (deg) | '+' | '.join(f'{p} m/s' for p in r['peaks'])+' |',
                    '|---|---:|'+'---|'*len(r['peaks'])]
            for ctl, by_lag in sorted(r['completed'].items()):
                for lag in sorted(by_lag, key=float):
                    cells = [f"{by_lag[lag][str(p)][0]}/{by_lag[lag][str(p)][1]}" for p in r['peaks']]
                    out.append(f'| {ctl} | {lag} | '+' | '.join(cells)+' |')
            out.append('')
    rob = a.input/'e4_robustness.json'
    if rob.exists():
        r = json.loads(rob.read_text())
        out += ['## E4 one-factor robustness (survival)', '', '| factor | '+' | '.join(
            f'{c} @ {p}' for c in sorted(r['completed']) for p in r['peaks'])+' |',
            '|---|'+'---|'*(len(r['completed'])*len(r['peaks']))]
        for factor in r['factors']:
            cells = [f"{r['completed'][c][factor][str(p)][0]}/{r['completed'][c][factor][str(p)][1]}"
                     for c in sorted(r['completed']) for p in r['peaks']]
            out.append(f'| {factor} | '+' | '.join(cells)+' |')
        out.append('')
    abl = a.input/'e5_ablation.json'
    if abl.exists():
        rows = json.loads(abl.read_text())['episodes']
        levels = levels_of(rows)
        names = sorted({r['controller'] for r in rows})
        out += ['## E5 ablation', '']+survival_table(rows, names, levels, 'survival')+['']
        for key in ('pitch_peak_deg', 'height_rmse_m', 'foreaft_max_m'):
            out += paired_table(rows, [(n, 'full') for n in names if n != 'full'], levels, key, f'{key}, paired vs full')+['']
    e6 = a.input/'e6_residual_seeds.json'
    if e6.exists():
        rows = json.loads(e6.read_text())['episodes']
        levels = levels_of(rows)
        out += ['## E6 learned residuals x seeds vs the retuned hand law', '']
        out += survival_table(rows, ['B1_pid', 'B2plus_hand_hp1s', REF], levels, 'hand-designed references')+['']
        out += seed_summary(rows, r'(snn_neat_delay|ann_neat|snn_fixed)_s(\d+)', REF, levels)+['']
    out += section_phase(a.input/'e8_models', a.input/'e8_phase_closed_loop.json')+['']
    out += ops_note(a.input/'e8_models')
    (a.output/'tables.md').write_text('\n'.join(out)+'\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
