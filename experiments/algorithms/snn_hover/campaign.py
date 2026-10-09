"""Immediate bounded job queue. Results are never silently replaced."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    protocol=json.loads((root/'protocol.json').read_text());(root/'logs').mkdir(exist_ok=True)
    for path,expected in protocol['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen source changed: {path}')
    try:
        for seed in protocol['seeds']:
            for method in protocol['methods']:
                out=root/'training'/f'{method}_seed{seed}'
                if (out/'results.json').exists():continue
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
        write(root/'campaign_status.json',dict(status='training_complete',completed=35,total=35))
        progress(root,protocol)
    except BaseException as error:
        write(root/'campaign_failure.json',dict(error=repr(error),time=time.time()));raise
if __name__=='__main__':main()
