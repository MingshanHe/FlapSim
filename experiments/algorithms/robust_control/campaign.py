"""Finite, recoverable v3 queue, sealed code, paired seeds, held-out final tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
from pathlib import Path
import shutil
import re
import subprocess
import sys
import time
import traceback
from ..snn_pid_compare.common import ROOT, read, write, sha, event, heartbeat, prepare_source
from .networks import initial_spec
from .train_neat import allocated_budget
from .environment import scenarios
from .evaluate import worker_init, evaluate
from .reporting import generate


def job_name(task, mode, kind, method, seed):
    return f'{task}_{mode}_{kind}_{method}_seed{seed}'


def prepare(root, destination):
    if (root/'protocol.json').exists():
        protocol = read(root/'protocol.json')
        for relative, digest in protocol['source_hashes'].items():
            if sha(root/'frozen_source'/relative) != digest:
                raise ValueError(f'Frozen source changed: {relative}')
        for relative, digest in protocol['plant_asset_hashes'].items():
            if sha(root/'source'/relative) != digest:
                raise ValueError(f'Plant asset changed: {relative}')
        return protocol
    smoke = read(root/'smoke_final/results.json')
    if not smoke['passed']:
        raise ValueError('All trainer smoke checks must pass')
    import importlib
    for name,digest in smoke['source_hashes'].items():
        if sha(importlib.import_module(name).__file__) != digest:
            raise ValueError(f'Implementation changed after trainer smoke: {name}')
    confirmation = read(root/'velocity_confirmation/results.json')
    if len(confirmation) != 8 or any(r['termination'] != 'time_limit' for r in confirmation):
        raise ValueError('Velocity baseline must first complete the eight development command cases')
    confirmation_plan = read(root/'velocity_confirmation/plan.json')
    if confirmation_plan['environment_sha256'] != sha(Path(__file__).with_name('environment.py')):
        raise ValueError('Revalidate the changed velocity environment before sealing')
    prepare_source(root/'source')
    seeds = [6201,6202,6203,6204,6205]
    jobs = []
    # Prioritize residuals; failure of direct imitation no longer gates them.
    for task, modes in [('hover', ('residual','direct')), ('velocity', ('residual',))]:
        for mode in modes:
            for seed in seeds:
                for kind in ('snn','ann'):
                    for method in ('neat','ppo'):
                        jobs.append(dict(task=task, mode=mode, kind=kind, method=method, seed=seed,
                                         name=job_name(task, mode, kind, method, seed)))
    # Mechanistic control: distinguish evolving topology from weight search.
    for seed in seeds:
        jobs.append(dict(task='hover',mode='residual',kind='snn',method='evo',seed=seed,
                         name=job_name('hover','residual','snn','evo',seed)))
    for job in jobs:
        job['budget'] = allocated_budget(16,12,job['task'])
    initializations = {}
    historical = {}
    for method in ('snn_neat','ann_neat','rnn_ppo'):
        original = ROOT/f'artifacts/snn_hover_20261009/training/{method}_seed2201/best_policy.json'
        spec = read(original)
        if spec['inputs'] != 11:
            raise ValueError('Historical observation mismatch')
        target = root/'initialization'/f'historical_{method}_2201.json'
        write(target,dict(legacy_100hz=True,policy=spec))
        historical[method] = dict(path=str(target),sha256=sha(target),prior_source=str(original),
                                  prior_sha256=sha(original),policy_hz=100,inputs=11)
    for kind, bc_seed in [('snn',5101),('ann',5102)]:
        original = ROOT/f'runs/snn_pid_compare/20261010_v21/development/round2/{kind}_seed{bc_seed}/best_policy.json'
        direct = root/'initialization'/f'direct_{kind}.json'
        write(direct, read(original))
        initializations[f'hover_direct_{kind}'] = dict(path=str(direct), sha256=sha(direct),
            prior_source=str(original), prior_sha256=sha(original),
            selection='highest original nominal validation rank within architecture; fixed BC prior shared across five RL/evolution seeds')
        for task, inputs in [('hover',11),('velocity',16)]:
            for seed in seeds:
                p = root/'initialization'/f'{task}_residual_{kind}_{seed}.json'
                write(p, initial_spec(kind, seed, inputs=inputs, zero_readout=True))
                initializations[f'{task}_residual_{kind}_{seed}'] = dict(path=str(p), sha256=sha(p))
    frozen = root/'frozen_source'
    if frozen.exists():
        frozen.rename(root/f'frozen_source_incomplete_{time.time_ns()}')
    # Copy both packages and assets so child processes keep their exact code
    # even if subsequent exploratory work modifies the live workspace.
    for package in ('experiments','flapsim'):
        shutil.copytree(ROOT/package, frozen/package, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    xml = root/'source/robot_upper7_lower8.xml'
    meshdir = str(frozen/'flapsim/robot/IST_robot-V2/meshes')
    xml.write_text(re.sub(r'meshdir="[^"]+"',f'meshdir="{meshdir}"',xml.read_text()))
    provenance = read(root/'source/provenance.json')
    provenance.update(meshdir=meshdir,relocated_xml_sha256=sha(xml))
    write(root/'source/provenance.json',provenance)
    plant_hashes = {str(p.relative_to(root/'source')):sha(p)
                    for p in sorted((root/'source').rglob('*')) if p.is_file()}
    hashes = {str(p.relative_to(frozen)):sha(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
    protocol = dict(version='robust-control-v3.0', created_unix=time.time(),
        user_authorized_large_training=True, original_v21_retained=True,
        seeds=seeds, jobs=jobs, population=16, generations=12,
        validation_after_generations=[0,4,8,12], steps_per_run=allocated_budget(16,12),
        steps_per_task={t:allocated_budget(16,12,t) for t in ('hover','velocity')},
        total_allocated_training_steps=sum(j['budget'] for j in jobs),
        horizons_seconds=[4.,8.,16.], cases_per_generation=4,
        task_horizons_seconds=dict(hover=[4.,8.,16.],velocity=[8.,12.,20.]),
        velocity_controller=dict(feedforward=[7.,1.],kp=[.02,1.],ki=[.01,.3],
                                 startup_seconds=6.,reference_limit_rad=.6,reference_rate_rad_s=.6),
        workers_per_run=8, concurrent_runs=2, evaluation_workers=16,
        ppo=dict(rollout=500,window=500,epochs=4,learning_rate=.0003,
                 resume='optimizer/RNG preserved; unfinished episodes explicitly reset at rollout boundary'),
        backend='original_python_torch_eager_float64', physical_dt=.000005, control_dt=.002,
        observation_dimensions=dict(hover=11,velocity=16),
        residual_scale=[.035,.035,100.], initializations=initializations, historical=historical,
        source_hashes=hashes, frozen_source=str(frozen), final_tests_accessed=False,
        plant_asset_hashes=plant_hashes,
        final_counts=dict(nominal_20s=16,stress_20s=16,nominal_60s=4,stress_60s=4,
                          factor_delay_20s=8,factor_mass_20s=8,factor_motor_20s=8,ood_20s=8,
                          velocity_command_generalization_20s=8),
        robust_gate=dict(nominal=15/16,stress=14/16,long_nominal=1.,long_stress=.75),
        limits=['Simulation only; fixed teacher and prior shared across training seeds',
                'Different direct and residual priors/authority; paired comparison tests complete control designs',
                'NEAT allocated and consumed steps differ after early termination',
                'Velocity task assumes an additional physical horizontal velocity sensor',
                'No result may be rewritten to favor SNN or NEAT'],
        result_destination=str(destination), publication='local_only',repository=str(ROOT),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip())
    protocol['git_commit_at_seal']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(root/'protocol.json', protocol)
    return protocol


def final_cases(task):
    cases = []
    for seconds, count, condition in [(20.,16,'nominal'),(20.,16,'stress'),
                                       (60.,4,'nominal'),(60.,4,'stress')]:
        for c in scenarios('final',count,seconds,condition,task):
            if task == 'velocity' and seconds == 60.:
                # Long tests cover two horizontal axes and both vertical signs.
                i = c['seed'] % 1000
                c['velocity_command'] = [[.04,0,0],[0,.04,0],[0,0,.06],[0,0,-.06]][i]
            c['test_family'] = ('long_' if seconds == 60. else '')+condition
            cases.append(c)
    for key, value in [('delay_steps',3),('lower_mass_scale',1.125),('motor_uncertainty',.04)]:
        for c in scenarios('final',8,20.,task=task):
            c[key] = value
            c['test_family'] = 'factor_'+key
            cases.append(c)
    for c in scenarios('final',8,20.,'ood',task):
        c.update(delay_steps=3,motor_uncertainty=.04,test_family='ood')
        cases.append(c)
    if task == 'velocity':
        for i,c in enumerate(scenarios('final',8,20.,task=task)):
            vectors = [[.06,.02,0],[-.06,-.02,0],[.02,-.06,0],[-.02,.06,0],
                       [0,0,.075],[0,0,-.075],[.04,-.04,.05],[-.04,.04,-.05]]
            c.update(velocity_command=vectors[i],test_family='command_generalization')
            cases.append(c)
    return cases


def train_one(root, protocol, job):
    output = root/'training'/job['name']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        return job['name']
    key = f"{job['task']}_{job['mode']}_{job['kind']}"
    if job['mode'] == 'residual':
        key += f"_{job['seed']}"
    initial = protocol['initializations'][key]
    if sha(initial['path']) != initial['sha256']:
        raise ValueError('Initialization changed after protocol seal')
    module = 'train_ppo' if job['method'] == 'ppo' else 'train_neat'
    args = [sys.executable,'-u','-m','experiments.algorithms.robust_control.'+module,
            '--source',str(root/'source'),'--initial',initial['path'],'--output',str(output),
            '--mode',job['mode'],'--task',job['task'],'--seed',str(job['seed']),
            '--workers',str(protocol['workers_per_run'])]
    if job['method'] == 'ppo':
        args += ['--steps',str(job['budget']),'--resume-reset-episodes']
    else:
        args += ['--population',str(protocol['population']),'--generations',str(protocol['generations']),
                 '--validate-every','4']
        if job['method'] == 'evo':
            args.append('--fixed-topology')
    write(output/'invocation.json',dict(command=args,cwd=protocol['frozen_source'],job=job,
                                      initial_sha256=initial['sha256'],started_unix=time.time()))
    with (output/'train.log').open('a') as log:
        subprocess.run(args,cwd=protocol['frozen_source'],stdout=log,stderr=subprocess.STDOUT,check=True)
    return job['name']


def finish(root, protocol, destination):
    jobs = protocol['jobs']
    if not all((root/'training'/j['name']/'results.json').exists() for j in jobs):
        raise ValueError('Final testing requires every predeclared training run')
    selected = {j['name']:sha(root/'training'/j['name']/'best_policy.json') for j in jobs}
    sealed = root/'selection_frozen.json'
    if sealed.exists() and read(sealed)['policies'] != selected:
        raise ValueError('Selected policies changed after final-test seal')
    if not sealed.exists():
        write(sealed,dict(unix=time.time(),policies=selected,selection_used_final_tests=False))
    # Invoke evaluation from the same sealed code, not the editable workspace.
    args = [sys.executable,'-u','-m','experiments.algorithms.robust_control.campaign',
            '--root',str(root),'--destination',str(destination),'--final-only']
    subprocess.run(args,cwd=protocol['frozen_source'],check=True)


def final_evaluation(root, protocol, destination):
    sealed = read(root/'selection_frozen.json')['policies']
    for job in protocol['jobs']:
        if sha(root/'training'/job['name']/'best_policy.json') != sealed[job['name']]:
            raise ValueError('Changed policy after final-test seal')
    for task in ('hover','velocity'):
        cases = final_cases(task)
        write(root/f'final_cases_{task}.json',cases)
        models = [(task+'_pid',None,'pid')]
        if task == 'hover':
            models += [(f'hover_bc_{k}',protocol['initializations'][f'hover_direct_{k}']['path'],'direct')
                       for k in ('snn','ann')]
            models += [(f'historical_{method}_2201',v['path'],'residual')
                       for method,v in protocol['historical'].items()]
        models += [(j['name'],str(root/'training'/j['name']/'best_policy.json'),j['mode'])
                   for j in protocol['jobs'] if j['task'] == task]
        with ProcessPoolExecutor(protocol['evaluation_workers'],initializer=worker_init,
                                 initargs=(str(root/'source'),task)) as pool:
            for name, path, mode in models:
                target = root/'evaluation'/f'{name}.json'
                if target.exists():
                    continue
                event(root,'independent_final_evaluation',model=name,final_tests_accessed=True)
                result = evaluate(pool,read(path) if path else None,mode,cases)
                write(target,result)
                generate(root,destination)
    event(root,'completed',training_runs=len(protocol['jobs']),final_tests_accessed=True)
    generate(root,destination)
    commit_report(root,protocol,destination)


def commit_report(root,protocol,destination):
    """Local report commit only, on the declared branch; never contact a remote."""
    repository=Path(protocol['repository'])
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=repository,text=True).strip()
    if branch != protocol['branch']:
        write(root/'local_delivery.json',dict(committed=False,reason='branch changed',branch=branch))
        return
    paths=[str(destination.relative_to(repository))]
    subprocess.run(['git','add','--',*paths],cwd=repository,check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=repository).returncode
    if changed:
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost',
                        'commit','-m','Record complete v3 control study and held-out evaluations',
                        '--only','--',*paths],cwd=repository,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    write(root/'local_delivery.json',dict(committed=True,commit=commit,branch=branch,pushed=False))


def run(root, destination):
    protocol = prepare(root,destination)
    tasks = [j for j in protocol['jobs'] if not (root/'training'/j['name']/'results.json').exists()]
    completed = len(protocol['jobs'])-len(tasks)
    generate(root,destination)
    with ThreadPoolExecutor(protocol['concurrent_runs']) as pool:
        pending, index = {}, 0
        while index < len(tasks) or pending:
            pausing = (root/'pause_request.json').exists()
            while not pausing and index < len(tasks) and len(pending) < protocol['concurrent_runs']:
                job = tasks[index]
                pending[pool.submit(train_one,root,protocol,job)] = job
                index += 1
            if not pending:
                event(root,'paused_at_run_boundary',completed=completed,total=len(protocol['jobs']))
                return
            event(root,'training',runs=[j['name'] for j in pending.values()],completed=completed,
                  total=len(protocol['jobs']),steps_per_run=protocol['steps_per_run'],final_tests_accessed=False)
            done,_ = wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                pending.pop(future)
                completed += 1
            generate(root,destination)
    finish(root,protocol,destination)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=ROOT/'artifacts/robust_control_20261010')
    p.add_argument('--final-only',action='store_true')
    a = p.parse_args()
    root,destination = a.root.resolve(),a.destination.resolve()
    root.mkdir(parents=True,exist_ok=True)
    if a.final_only:
        final_evaluation(root,read(root/'protocol.json'),destination)
        return
    with (root/'campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with heartbeat(root):
                run(root,destination)
        except Exception:
            event(root,'failed',traceback=traceback.format_exc())
            generate(root,destination)
            raise
        finally:
            generate(root,destination)


if __name__ == '__main__':
    main()
