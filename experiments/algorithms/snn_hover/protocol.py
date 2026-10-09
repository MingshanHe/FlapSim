"""Freeze the approved research design before any formal training/test access."""
import argparse,hashlib,json,platform,subprocess,time
from pathlib import Path
import importlib.metadata
from .common import write
from .evolution import METHODS

def freeze(root):
    if (root/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    core=['__init__.py','networks.py','environment.py','common.py','evolution.py','train.py','ppo.py']
    paths=[Path('experiments/algorithms/snn_hover')/f for f in core]
    paths+=list(Path('experiments/algorithms/topology_hover').glob('*.py'))+list(Path('experiments/algorithms/topology_hover').glob('*.cpp'))
    paths+=list(Path('experiments/algorithms/pid_airborne').glob('*.py'))+list(Path('experiments/algorithms/hover9').glob('*.py'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(status='frozen_before_formal_training',created_unix=time.time(),base_commit='3711664acf1e9660e304a6c8a473bb55e0eb102f',
        methods=METHODS+['rnn_ppo','ff_ppo'],seeds=list(range(2201,2206)),population=64,workers=16,
        bridge_budget=256000,final_budget=1024000,validation_interval=51200,validation_episodes=8,validation_seconds=8.,
        curriculum=[dict(until=256000,seconds=2.),dict(until=512000,seconds=4.),dict(until=1024000,seconds=8.)],
        allocation='Evolution charges complete requested episodes to its budget; early-termination actual steps are separately reported. PPO counts actual transitions. Never claim exact actual-interaction equality if these differ.',
        reward='Unchanged topology_hover reward; evolution fitness = return/requested duration - 0.00002*enabled edges for every evolution arm',
        selection='lexicographic validation strict-success rate, mean return, negative tilt RMS; initial policy eligible; no test selection',
        stop='Common fixed allocated budget; plateau diagnostic only, no data-dependent training extension',
        development_seed_range=[409900000,409999999],training_seed_range=[360000000,399999999],validation_seed_formula='410000000 + training_seed*10 + index(0..7)',
        test=dict(seeds=list(range(420000000,420000016)),seconds=20.,conditions=['nominal','stress','ood']),
        long_test=dict(seeds=list(range(420100000,420100008)),seconds=60.,conditions=['nominal','stress'],all_training_seeds=True),
        recovery=dict(angle_seeds=list(range(420200000,420200016)),angles=[5,10,15,20,30,45],directions=[0,45,90,135,180,225,270,315],
            pulse_seeds=list(range(420300000,420300016)),pulse_torques=[.000016,.000032,.000064],pulse_start=5.,pulse_duration=.1,seconds=20.,
            criterion='500ms tilt RMS below 10deg by 5s after release/pulse end, continuously for another 1s; full horizon and original clearance + height criteria'),
        factors=dict(seeds=list(range(420400000,420400008)),seconds=20.,names=['delay_10ms','motor_3pct','lower9g','pulse_1p5x']),
        interventions=dict(seeds=list(range(420500000,420500008)),seconds=20.,condition='stress',variants=['original','tau_initial','tau_shuffle','reset_each_tick','remove_recurrent','random_remove_0','random_remove_1','random_remove_2']),
        inference='CPU latency measured after training; all ten SNN internal updates counted; no energy claims',
        interpretation='H1/H2 primary; simultaneous bootstrap intervals and paired sign-flip tests with Holm adjustment; five training seeds exploratory',
        proposal_adjustments=[
            'Common budgets rounded down by 2.34375% from 262144/1048576 to 256000/1024000 to align full evolution generations and PPO rollouts; previous policies retain their 262144 budget label.',
            'Initial compatibility distances checked on development seed2198 (median0.14256, max0.20128); threshold0.25 selected before formal reward evaluation; no guarantee of multiple species.',
            'PPO curriculum set at each rollout boundary; an already active episode completes under its previous horizon.',
            'Analog temporal control has zero background; SNN background1.2 prevents quiescence. It tests the combined threshold/reset/background mechanism, not threshold alone.'
        ],source_hashes=hashes,packages={p:importlib.metadata.version(p) for p in ['neat-python','mujoco','numpy','torch','stable_baselines3']},python=platform.python_version())
    root.mkdir(parents=True,exist_ok=True);write(root/'protocol.json',protocol)
    write(root/'source_hashes.json',hashes)
    snapshot=root/'frozen_source'
    for p in paths:
        q=snapshot/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    (root/'research_proposal_approved.md').write_bytes(Path('research_proposal.md').read_bytes())
    write(root/'campaign_status.json',dict(status='ready',completed=0,total=35))
    return protocol

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();freeze(a.root)
if __name__=='__main__':main()
