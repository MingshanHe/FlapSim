"""Paired 60 s diagnostic of inertial drift versus oracle z/vz feedback."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .audit import init,evaluate,summarize
from ..neat_imu.train import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    policy=json.loads((a.root/'selected_pid.json').read_text())['parameters']
    tasks=[]
    for stress,seed in [(False,310920000),(True,310920001)]:
        for source in ['imu','truth']:
            extra=dict(acceleration_bias_range=.01,motor_uncertainty=.01,attitude_noise_deg=.02) if stress else {}
            label=f'{source}_{"stress" if stress else "nominal"}_60s'
            tasks.append((label,dict(policy,height_source=source,**extra),seed,60.,5. if stress else 2.,True))
    write_json(a.root/'duration_probe_protocol.json',dict(status='written_before_execution',
        tasks=[dict(label=t[0],parameters=t[1],seed=t[2],seconds=t[3],initial_degrees=t[4]) for t in tasks],
        purpose='Paired duration/drift diagnostic; no policy tuning or test-set selection follows. Truth source reads both z and vz.'))
    with ProcessPoolExecutor(max_workers=4,initializer=init,initargs=(str(a.root),)) as pool:
        rows=list(pool.map(evaluate,tasks))
    summaries={r['policy']:summarize([r]) for r in rows}
    write_json(a.root/'duration_probe_results.json',dict(summaries=summaries,episodes=rows))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
