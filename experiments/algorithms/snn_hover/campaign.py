"""Immediate bounded job queue. Results are never silently replaced."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,platform,fcntl
from pathlib import Path
from .common import write

def progress(root,protocol):
    results=[]
    for p in sorted((root/'training').glob('*/results.json')):results.append(json.loads(p.read_text()))
    lines=['# NEAT-SNN 与 ANN／PPO：实验进行中','',
        '**尚未完成独立测试，当前不能给出 SNN 或拓扑优势结论。**','',
        f'已完成 {len(results)}/35 次正式训练；预算为每次 1,024,000 次分配交互，完整记录实际交互数。','',
        '[冻结协议](protocol.json) · [已审阅 proposal](../../research_proposal.md)','',
        '| 方法 | 种子 | 实际训练步 | 验证成功率 | 验证倾角 RMS | 经验平台期 |',
        '|---|---:|---:|---:|---:|---|']
    for r in results:
        v=r['best_validation'];lines.append(f'| {r["method"]} | {r["seed"]} | {r["training_steps"]} | {v["success_rate"]:.1%} | {v["tilt_rms_deg"]:.3f}° | {r.get("validation_plateau",False)} |')
    lines+=['','验证分数用于选模，不是独立泛化结论。开发、训练、最终测试分别存档；所有失败保留。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return len(results)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root
    control=root/'control';control.mkdir(exist_ok=True)
    lock=(control/'campaign.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another campaign already owns this study directory')
    protocol=json.loads((root/'protocol.json').read_text());(root/'logs').mkdir(exist_ok=True)
    for path,expected in protocol['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen source changed: {path}')
    pause=root/'pause_request.json'
    if pause.exists():
        if not a.resume:raise RuntimeError('Study paused by user. Explicit --resume is required; see exec.md')
        completed=root/'pause_completed.json'
        if not completed.exists():raise RuntimeError('Requested final run is still active; wait for pause_completed.json')
        snapshot=json.loads(completed.read_text());request=json.loads(pause.read_text())
        assert snapshot['status']=='paused' and snapshot['after_run']==request['after_run']
        assert json.loads((root/'training'/request['after_run']/'results.json').read_text())['status']=='completed'
        write(control/f'resume_{time.time_ns()}.json',dict(pause_request=request,pause_completed=snapshot,resumed_unix=time.time(),host=platform.node()))
        pause.unlink();completed.unlink()
    elif a.resume:raise RuntimeError('No recorded user pause to resume')
    with (control/'execution_hosts.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(),host=platform.node(),platform=platform.platform(),python=sys.version,
            executable=sys.executable,workers=protocol['workers'],existing_completed=sorted(x.parent.name for x in (root/'training').glob('*/results.json'))))+'\n')
    try:
        for seed in protocol['seeds']:
            for method in protocol['methods']:
                out=root/'training'/f'{method}_seed{seed}'
                if (out/'results.json').exists():
                    existing=json.loads((out/'results.json').read_text())
                    assert existing['status']=='completed' and existing['budget']==protocol['final_budget'],out
                    continue
                if out.exists():raise RuntimeError(f'Incomplete run retained; inspect before restarting: {out}')
                module='ppo' if method.endswith('_ppo') else 'train'
                command=[sys.executable,'-u','-m',f'experiments.algorithms.snn_hover.{module}','--output',str(out),'--method',method,
                    '--seed',str(seed),'--steps',str(protocol['final_budget']),'--workers',str(protocol['workers'])]
                if module=='train':command+=['--population',str(protocol['population'])]
                log=root/'logs'/f'{method}_seed{seed}.log'
                with log.open('w') as f:
                    proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    started=time.monotonic();last_report=0.
                    write(root/'campaign_status.json',dict(status='training',current=str(out),pid=proc.pid,command=command,started=time.time(),completed=progress(root,protocol),total=35))
                    while proc.poll() is None:
                        if time.monotonic()-started>14400:
                            os.killpg(proc.pid,signal.SIGTERM);raise TimeoutError(f'Four-hour individual run limit: {out}')
                        if time.monotonic()-last_report>60:
                            progress(root,protocol);last_report=time.monotonic()
                        time.sleep(5)
                if proc.returncode:raise RuntimeError(f'Run exited {proc.returncode}: {log}')
                count=progress(root,protocol);print(json.dumps(dict(completed=count,total=35,method=method,seed=seed)),flush=True)
                if pause.exists() and json.loads(pause.read_text())['after_run']==out.name:
                    snapshot=dict(status='paused',after_run=out.name,completed=count,total=35,remaining=35-count,paused_unix=time.time())
                    write(root/'pause_completed.json',snapshot);write(root/'campaign_status.json',snapshot)
                    return
        write(root/'campaign_status.json',dict(status='training_complete',completed=35,total=35))
        progress(root,protocol)
    except BaseException as error:
        write(root/'campaign_failure.json',dict(error=repr(error),time=time.time()));raise
if __name__=='__main__':main()
