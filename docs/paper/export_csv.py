"""Export every number behind the paper draft to editable CSV files under docs/paper/data/.
    .venv/bin/python docs/paper/export_csv.py
Files:
  nominal_v11_arms.csv          Table R0 (+ exploratory seeds, classical self ratio)
  nominal_plans_v7_v11.csv      per-plan SNN+ES means and opponent self-ratios (Fig. R0)
  robustness_crashes.csv        Table R1 incl. the pulses-on probe and the v9 centres (crashes per arm per scenario)
  robustness_paired_delta.csv   paired delta with 95 % CI, wins, crashes per arm per scenario (long format)
  dose_response.csv             crashes per arm at 2 / 3 / 4.5 mN m (Fig. R4)
  es_training_v11.csv           ES centre objective per generation, seeds 1-5 and classical+ES 1-2 (Fig. R1a)
  selection_v11.csv             selection objective per saved centre (Fig. R1b)
  mechanism_summary.csv         Fig. R5 numbers (tilt at hit, pulses, matched-bin crash rates)
  mechanism_refly_episodes.csv  per-episode features of the 192 primary cases re-flown with traces
  ablation_v14.csv              channel ablations, imitation starts 2-5, pulses-on probe (Fig. R5b)
  snn_vs_cls_es.csv             SNN seeds paired against classical+ES seed 1 on every set
  episodes/<set>__<arm>.csv     every blind episode of every arm (score, flown, termination, task metrics)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT/'artifacts/spike_steer'
R = ROOT/'runs/spike_steer'
OUT = ROOT/'docs/paper/data'
EP = OUT/'episodes'
OUT.mkdir(parents=True, exist_ok=True)
EP.mkdir(exist_ok=True)
import sys
sys.path.insert(0, str(ROOT))
from experiments.algorithms.spike_steer.paired_report import paired  # noqa: E402


def w(name, header, rows):
    with (OUT/name).open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(header)
        cw.writerows(rows)
    print(name, len(rows), 'rows')


def rows_of(f):
    d = json.loads(Path(f).read_text())
    return d if isinstance(d, list) else d['rows']


EPCOLS = ['task', 'seed', 'cls', 'dir_deg', 'dir2_deg', 't0', 'score', 'score_bench9', 'flown', 'termination', 'elapsed', 'pulses', 'goal_t',
          't_reach', 't_settle', 'overshoot_m', 'cross_m', 't_switch', 't_reach2', 'overrun_m', 'height_loss_cm', 'alt_dev_cm', 'tilt_peak_deg']


def write_episodes(setname, arm, rows):
    with (EP/f'{setname}__{arm}.csv').open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['arm']+EPCOLS)
        for r in rows:
            cw.writerow([arm]+[r.get(k) for k in EPCOLS])


# ---------------- nominal v11
S11 = json.loads((A/'plan_v11/summary.json').read_text())
opp11 = rows_of(R/'v11/test_opponent.json')
rows = [['classical_opponent', '', 'CloudPC-2', round(S11['classical_self_ratio'], 4), 0, '', '', '', 0]]
write_episodes('nominal_v11', 'classical', opp11)
for s in (1, 2):
    f = (R/f'v11/evaluate_cls_es_s{s}_v11.json') if s == 1 else (A/f'plan_v11/homepc/evaluate_cls_es_s{s}_v11.json')
    rr = rows_of(f)
    p = paired(opp11, rr)
    rows.append([f'cls_es', s, 'CloudPC-2' if s == 1 else 'HomePC', round(p['mean'], 4), round(p['delta'], 4), *[round(x, 4) for x in p['delta_ci95']], p['wins'], p['crashes']])
    write_episodes('nominal_v11', f'cls_es_s{s}', rr)
for s in range(1, 6):
    root = R/'v11' if s <= 3 else A/'plan_v11/homepc'
    for arm in ('snn_es', 'snn_bc'):
        f = root/f'evaluate_{arm}_s{s}_v11.json'
        if not f.exists():
            continue
        rr = rows_of(f)
        p = paired(opp11, rr)
        rows.append([arm, s, 'CloudPC-2' if s <= 3 else 'HomePC', round(p['mean'], 4), round(p['delta'], 4), *[round(x, 4) for x in p['delta_ci95']], p['wins'], p['crashes']])
        write_episodes('nominal_v11', f'{arm}_s{s}', rr)
for s, m in ((6, .997), (7, 1.093), (8, 1.058), (9, 1.013), (10, 1.016)):
    rows.append(['snn_es_exploratory', s, 'GoogleCloud (approximate; rows not archived)', m, '', '', '', '', 0])
un = rows_of(R/'v14/nominal__cls_unsup.json')
p = paired(opp11, un)
rows.append(['classical_pulses_on_probe', '', 'CloudPC-2', round(p['mean'], 4), round(p['delta'], 4), *[round(x, 4) for x in p['delta_ci95']], p['wins'], p['crashes']])
write_episodes('nominal_v11', 'classical_pulses_on_probe', un)
w('nominal_v11_arms.csv', ['arm', 'seed', 'machine', 'mean_ratio', 'paired_delta', 'delta_ci_lo', 'delta_ci_hi', 'wins_of_192', 'crashes'], rows)

# ---------------- plans v7-v11
rows = []
v7 = [json.loads((A/f'plan_v7/full/snn/s{s}/evaluate_snn_es_s{s}.json').read_text())['report']['manoeuvre_mean'] for s in range(1, 6)]
opp7 = rows_of(R/'v7full/test_opponent.json')
rows += [['v7full', s, v7[s-1], paired(opp7, opp7)['mean'], 32] for s in range(1, 6)]
S8 = json.loads((A/'plan_v8/summary.json').read_text())
opp8 = rows_of(R/'v8/test_opponent.json')
rows += [['v8', s, S8[f'snn_es_s{s}']['mean'], paired(opp8, opp8)['mean'], 64] for s in range(1, 6)]
S9 = json.loads((A/'plan_v9/summary.json').read_text())
opp9 = rows_of(R/'v9/test_opponent.json')
rows += [['v9', s, S9[f'snn_es_s{s}']['mean'], paired(opp9, opp9)['mean'], 64] for s in range(1, 6)]
S10 = json.loads((A/'plan_v10/summary.json').read_text())
opp10 = rows_of(R/'v10/test_opponent.json')
rows += [['v10', s, S10[f's{s}']['snn_es']['mean'], paired(opp10, opp10)['mean'], 64] for s in range(1, 6)]
rows += [['v11', s, S11[f's{s}']['snn_es']['mean'], S11['classical_self_ratio'], 64] for s in range(1, 6)]
w('nominal_plans_v7_v11.csv', ['plan', 'seed', 'snn_es_mean_ratio', 'opponent_self_ratio', 'cases_per_task'], [[a, b, round(c, 4), round(d, 4), e] for a, b, c, d, e in rows])

# ---------------- robustness
TESTS = [('v12', 'kick2'), ('v12', 'kick'), ('v12', 'kick_rep'), ('v12', 'kick45'), ('v12', 'kick_early'), ('v12', 'kick_late'), ('v13', 'kick_pre'), ('v13', 'kick_double'),
         ('v13', 'kick_rollaxis'), ('v13', 'kick_pitchaxis'), ('v13', 'turb125'), ('v13', 'turb15'), ('v13', 'turb2'), ('v13', 'turb3'), ('v13', 'noise5'), ('v13', 'delay3'), ('v12', 'ood')]
crash_rows, delta_rows, dose = [], [], []
for plan, t in TESTS:
    d = R/plan/t
    S = json.loads((d/'summary.json').read_text())
    opp = rows_of(d/'rows_classical.json')
    write_episodes(t, 'classical', opp)
    rec = {'scenario': t, 'classical': S['classical']['crashes']}
    for f in sorted(d.glob('rows_*.json')):
        arm = f.stem[len('rows_'):]
        if arm == 'classical':
            continue
        rr = rows_of(f)
        write_episodes(t, arm, rr)
        p = paired(opp, rr)
        rec[arm] = p['crashes']
        delta_rows.append([t, arm, round(p['mean'], 4), round(p['classical_self'], 4), round(p['delta'], 4), *[round(x, 4) for x in p['delta_ci95']], p['wins'], p['crashes'], p['classical_crashes'],
                           S.get(arm, {}).get('H1', '')])
    un = R/'v14'/f'{t}__cls_unsup.json'
    if un.exists():
        rr = rows_of(un)
        write_episodes(t, 'classical_pulses_on_probe', rr)
        p = paired(opp, rr)
        rec['classical_pulses_on_probe'] = p['crashes']
        delta_rows.append([t, 'classical_pulses_on_probe', round(p['mean'], 4), round(p['classical_self'], 4), round(p['delta'], 4), *[round(x, 4) for x in p['delta_ci95']], p['wins'], p['crashes'], p['classical_crashes'], ''])
    for s in (2, 3, 4, 5):
        bc = R/'v14'/f'{t}__snn_bc_s{s}.json'
        if bc.exists():
            rr = rows_of(bc)
            write_episodes(t, f'snn_bc_s{s}', rr)
            p = paired(opp, rr)
            rec[f'snn_bc_s{s}'] = p['crashes']
            delta_rows.append([t, f'snn_bc_s{s}', round(p['mean'], 4), round(p['classical_self'], 4), round(p['delta'], 4), *[round(x, 4) for x in p['delta_ci95']], p['wins'], p['crashes'], p['classical_crashes'], ''])
    rec['H1_v11_seeds_passing'] = S['H1']['seeds_passing']
    crash_rows.append(rec)
    if t in ('kick2', 'kick', 'kick45'):
        dose.append(rec)
cols = ['scenario', 'classical', 'classical_pulses_on_probe', 'cls_es_s1', 'cls_es_s2'] + [f'snn11_s{s}' for s in range(1, 6)] + [f'snn9_s{s}' for s in range(1, 6)] + [f'snn_bc_s{s}' for s in range(1, 6)] + ['H1_v11_seeds_passing']
w('robustness_crashes.csv', cols, [[r.get(c, '') for c in cols] for r in crash_rows])
w('robustness_paired_delta.csv', ['scenario', 'arm', 'mean_ratio', 'opponent_self_ratio', 'paired_delta', 'delta_ci_lo', 'delta_ci_hi', 'wins_of_192', 'crashes', 'opponent_crashes', 'H1_pass'], delta_rows)
w('dose_response.csv', ['magnitude_mNm']+cols[1:-1], [[{'kick2': 2, 'kick': 3, 'kick45': 4.5}[r['scenario']]]+[r.get(c, '') for c in cols[1:-1]] for r in dose])

# ---------------- training / selection
rows = []
for s in range(1, 6):
    f = (R/f'v11/finetune_s{s}_v11/history.jsonl') if s <= 3 else (A/f'plan_v11/homepc/finetune_s{s}_v11/history.jsonl')
    for l in f.read_text().splitlines():
        h = json.loads(l)
        rows.append(['snn_es', s, h['generation'], round(h['center'], 4), *[round(h['center_ratios'][t], 4) for t in ('evade', 'redirect', 'dash')], h['flown_center'], h['crashed_candidates'], round(h['best_candidate'], 4), round(h['mean'], 4), round(h['lr'], 5)])
for s, f in ((1, R/'v11/cls_es_s1/history.jsonl'), (2, A/'plan_v11/homepc/cls_es_s2/history.jsonl')):
    for l in f.read_text().splitlines():
        h = json.loads(l)
        rows.append(['cls_es', s, h['generation'], round(h['center'], 4), *[round(h['center_ratios'][t], 4) for t in ('evade', 'redirect', 'dash')], h['flown_center'], h['crashed_candidates'], round(h['best_candidate'], 4), round(h['mean'], 4), round(h['lr'], 5)])
w('es_training_v11.csv', ['arm', 'seed', 'generation', 'centre_objective', 'evade', 'redirect', 'dash', 'centre_flown_of_24', 'crashed_candidates_of_16', 'best_candidate', 'population_mean', 'lr'], rows)
rows = []
for s in range(1, 6):
    f = (R/f'v11/reselect_s{s}_v11.json') if s <= 3 else (A/f'plan_v11/homepc/reselect_s{s}_v11.json')
    rs = json.loads(f.read_text())
    for c in rs['centres']:
        rows.append(['snn_es', s, c['generation'], round(c['objective'], 4), *[round(c['ratios'][t], 4) for t in ('evade', 'redirect', 'dash')], c['crashes'], c['over_hard'], round(c['tilt_peak_max'], 1), int(c['generation'] == rs['selected']['generation'])])
for s, f in ((1, R/'v11/reselect_cls_es_s1.json'), (2, A/'plan_v11/homepc/reselect_cls_es_s2.json')):
    rs = json.loads(f.read_text())
    for c in rs['centres']:
        rows.append(['cls_es', s, c['generation'], round(c['objective'], 4), *[round(c['ratios'][t], 4) for t in ('evade', 'redirect', 'dash')], c['crashes'], c['over_hard'], round(c['tilt_peak_max'], 1), int(c['generation'] == rs['selected']['generation'])])
w('selection_v11.csv', ['arm', 'seed', 'generation', 'objective_192_cases', 'evade', 'redirect', 'dash', 'crashes', 'over_hard_tilt', 'tilt_peak_max_deg', 'selected'], rows)

# ---------------- mechanism
M = json.loads((A/'plan_v13/mechanism_summary.json').read_text())['refly']
rows = []
for arm, v in M.items():
    b = v['crash_rate_by_tilt_at_hit']
    rows.append([arm, v['n'], v['crashes'], round(v['tilt_at_hit'][0], 2), *[round(x, 2) for x in v['tilt_at_hit'][1]], round(v['d_tilt_peak_vs_classical'][0], 2), *[round(x, 2) for x in v['d_tilt_peak_vs_classical'][1]],
                 round(v['corrective_fraction'], 3), round(v['proj_on_induced'][0], 4), round(v['dref_norm'][0], 4), round(v['pulses_post_minus_pre'][0], 3), *[round(x, 3) for x in v['pulses_post_minus_pre'][1]],
                 b.get('<20', ['', ''])[0], b.get('<20', ['', ''])[1], b.get('20-35', ['', ''])[0], b.get('20-35', ['', ''])[1]])
w('mechanism_summary.csv', ['arm', 'n', 'crashes', 'tilt_at_hit_deg', 'ci_lo', 'ci_hi', 'tilt_peak_delta_vs_classical_deg', 'ci_lo', 'ci_hi', 'corrective_fraction', 'proj_on_induced', 'dref_norm_rad', 'pulses_post_minus_pre', 'ci_lo', 'ci_hi',
                            'crashes_tilt_lt20', 'n_tilt_lt20', 'crashes_tilt_20_35', 'n_tilt_20_35'], rows)
rows = []
keys = ['task', 'seed', 'cls', 'flown', 'termination', 't_hit_rel', 'tilt_at_hit', 'tilt_peak_post', 'induced_deg', 'd_roll', 'd_pitch', 'dref_roll', 'dref_pitch', 'dref_norm', 'proj_on_induced', 'corrective', 'ref_norm_pre', 'ref_norm_post', 'pulses_pre', 'pulses_post', 't_recover25', 'crash_after_hit']
for f in sorted((R/'v13').glob('refly_*.json')):
    arm = f.stem[len('refly_'):]
    for x in json.loads(f.read_text()):
        if x.get('hit_before_end'):
            rows.append([arm]+[x.get(k) for k in keys])
w('mechanism_refly_episodes.csv', ['arm']+keys, rows)

# ---------------- v14 ablations / probe / imitation starts / snn vs cls_es
V = json.loads((A/'plan_v14/summary_d1_d4.json').read_text())
rows = [[k, v['set'], v['arm'], v['crashes'], *[round(x, 4) for x in v['ci95']], v['classical_crashes'], round(v['delta'], 4), *[round(x, 4) for x in v['delta_ci95']], v['wins']] for k, v in V['d2_d3_d4'].items()]
w('ablation_v14.csv', ['key', 'set', 'arm', 'crashes', 'crash_rate_ci_lo', 'crash_rate_ci_hi', 'opponent_crashes', 'paired_delta', 'delta_ci_lo', 'delta_ci_hi', 'wins'], rows)
rows = [[t, k, round(v['delta'], 4), *[round(x, 4) for x in v['ci']], v['crashes'], v['ref_crashes']] for t, d in V['d1_snn_vs_cls_es_s1'].items() for k, v in d.items()]
w('snn_vs_cls_es.csv', ['scenario', 'snn_seed', 'paired_delta_vs_cls_es_s1', 'ci_lo', 'ci_hi', 'snn_crashes', 'cls_es_crashes'], rows)
print('done ->', OUT)
