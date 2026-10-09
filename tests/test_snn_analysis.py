"""Prevent test leakage, asymmetric interventions and seed-level pseudoreplication."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from experiments.algorithms.snn_hover.evaluate import core_cases,extra_cases,variants
from experiments.algorithms.snn_hover.analyze import interval,signflip,holm
from experiments.algorithms.snn_hover.common import validation_cases

PROTOCOL=Path('artifacts/snn_hover_20261009/protocol.json')

def test_case_design_is_balanced_and_held_out():
    p=json.loads(PROTOCOL.read_text());core=list(core_cases(p));extra=list(extra_cases(p))
    assert len(core)==48 and len(extra)==192
    assert len({cid for cid,c in core})==48
    assert len({(g,cid) for g,cid,c in extra})==192
    testseeds={c['seed'] for cid,c in core}|{c['seed'] for g,cid,c in extra}|set(p['interventions']['seeds'])
    for seed in p['seeds']:
        validation={c['seed'] for c in validation_cases(seed)}
        assert not testseeds&validation
        assert all(not p['training_seed_range'][0]<=s<=p['training_seed_range'][1] for s in testseeds|validation)
    for angle in p['recovery']['angles']:
        cases=[c for g,cid,c in extra if g=='angle' and c['tilt_deg']==angle]
        assert len(cases)==16
        assert all(sum(c['direction_deg']==d for c in cases)==2 for d in p['recovery']['directions'])
    # 70 new checkpoints + 20 old policies + PID; extra tests only final/PID.
    assert 91*len(core)+36*len(extra)+5*8*8==11600

def test_interventions_preserve_matched_deletion_and_tau_multisets():
    spec=dict(kind='snn',inputs=11,outputs=[0,1,2],nodes=[dict(id=i,bias=0,tau_m=.005+i*.003,tau_s=.003+i*.002) for i in range(7)],
        edges=[dict(source=a,target=b,weight=.2,enabled=True) for a,b in [(-1,3),(3,3),(3,4),(4,0),(5,1),(6,2)]])
    before=copy.deepcopy(spec);rows=dict(variants(spec,2201));assert spec==before
    for name in ['remove_recurrent','random_remove_0','random_remove_1','random_remove_2']:
        assert sum(not e['enabled'] for e in rows[name]['edges'])==2
    assert all(e['target']>=3 for e in rows['remove_recurrent']['edges'] if not e['enabled'])
    for output in [False,True]:
        def taus(s):return sorted((n['tau_m'],n['tau_s']) for n in s['nodes'] if (n['id'] in s['outputs'])==output)
        assert taus(spec)==taus(rows['tau_shuffle'])
    assert rows['reset_each_tick']['reset_each_tick'] and not rows['original'].get('reset_each_tick')

def test_five_seed_exact_test_resolution_and_multiplicity():
    assert signflip([1,1,1,1,1])==pytest.approx(2/32)
    assert signflip([0,0,0,0,0])==1
    assert holm([.03,.01])==pytest.approx([.03,.02])
    assert holm([.01,.03,.04])==pytest.approx([.03,.06,.06])
    assert interval([.5]*5)==pytest.approx([.5,.5])
    assert interval([0,0,1,1,1])==interval([0,0,1,1,1])

def test_test_instrumentation_preserves_pid_episode():
    from experiments.algorithms.snn_hover.evaluate import intervention_task
    from experiments.algorithms.snn_hover.common import evaluate
    from experiments.algorithms.snn_hover.environment import case
    task=(None,case(409900007,.05,'stress',2),'pid','height',False)
    reference=evaluate(task);instrumented=intervention_task(task)
    for key in reference:
        if key=='saturation_fraction':continue
        assert reference[key]==instrumented[key],key
    assert instrumented['actor_step_delta_rms']==[0.,0.,0.]
    assert np.isfinite(instrumented['command_step_delta_rms']).all()
    trace=evaluate((*task[:4],True))['trace']
    expected=np.mean(np.column_stack([np.abs(np.array(trace)[:,8:10])>=.3-1e-9,(np.array(trace)[:,10]<=3115.224079+1e-6)|(np.array(trace)[:,10]>=3807.496097-1e-6)]),axis=0)
    np.testing.assert_array_equal(instrumented['saturation_fraction'],expected)
