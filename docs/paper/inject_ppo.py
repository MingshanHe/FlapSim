"""Insert the MLP + PPO reference numbers (plan v14 amendment 2) into draft_method_results.md: a Table R0 row, a sentence in
3.1 and one in the primary-scenario paragraph of 3.2.   .venv/bin/python docs/paper/inject_ppo.py"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.algorithms.spike_steer.paired_report import paired  # noqa: E402

R = ROOT/'runs/spike_steer'
P = ROOT/'docs/paper/draft_method_results.md'
s = P.read_text()
res = {}
for setname, oppf in (('nominal', R/'v11/test_opponent.json'), ('kick', R/'v12/kick/rows_classical.json')):
    opp = json.loads(oppf.read_text())
    res[setname] = []
    for seed in range(1, 6):
        f = R/'v14'/f'{setname}__mlp_ppo_s{seed}.json'
        if f.exists():
            res[setname].append(paired(opp, json.loads(f.read_text())))
assert len(res['nominal']) == 5 and len(res['kick']) == 5, 'PPO rows missing'
nom, kick = res['nominal'], res['kick']
fmt = lambda v: '%.3f' % v
row = '| MLP + PPO reference | 1–5 | CloudPC-2 (HomePC policies) | ' + ' / '.join(fmt(p['mean']) for p in nom) + ' | ' + ' / '.join('%+.3f' % p['delta'] for p in nom) + ' | ' + ' / '.join(str(p['wins']) for p in nom) + ' | ' + ' / '.join(str(p['crashes']) for p in nom) + ' |'
anchor = '| SNN, imitation only | 1 / 3 / 4 / 5 |'
i = s.index(anchor)
j = s.index('\n', i)
s = s[:j+1] + row + '\n' + s[j+1:]
old = "For reference, a MLP + PPO layer trained on the same tasks (plan v8, 5 seeds, three of which selected their imitation start) scored 1.109 ± 0.024 with 1 crash in 960 episodes on the v8 blind set, where the SNN + ES centres scored 0.97–1.07 with 10 crashes and the opponent crashed 5 times."
sig = sum(p['delta_ci95'][1] < 0 for p in nom)
new = ("For reference, the five frozen MLP + PPO policies trained on the same tasks (plan v7 full, HomePC; three of them selected their imitation start) score "
       + '–'.join(fmt(x) for x in (min(p['mean'] for p in nom), max(p['mean'] for p in nom))) + " on the v11 blind set (Table R0; paired Δ "
       + ' … '.join('%+.3f' % x for x in (min(p['delta'] for p in nom), max(p['delta'] for p in nom))) + f", {sig} of 5 intervals below 0; "
       + str(sum(p['crashes'] for p in nom)) + " crashes in 960 episodes), i.e. the same parity band as the SNN; on the v8 blind set they scored 1.109 ± 0.024 with 1 crash where the SNN + ES centres scored 0.97–1.07 with 10 crashes.")
assert old in s
s = s.replace(old, new, 1)
old2 = "Per task, the gain is in dash (2–10 crashes vs 19) and redirect (12–22 vs 24); evade crashes less too (6–13 vs 17) but its Δ has no consistent sign."
new2 = old2 + (" The MLP + PPO reference, which has the same set-point authority but no pulse channel, crashes "
               + '–'.join(str(x) for x in (min(p['crashes'] for p in kick), max(p['crashes'] for p in kick))) + " times on the same cases (Δ "
               + ' … '.join('%+.3f' % x for x in (min(p['delta'] for p in kick), max(p['delta'] for p in kick))) + ").")
assert old2 in s
s = s.replace(old2, new2, 1)
P.write_text(s)
print('nominal', [(round(p['mean'], 3), round(p['delta'], 3), p['crashes']) for p in nom])
print('kick', [(round(p['mean'], 3), round(p['delta'], 3), p['crashes']) for p in kick])
