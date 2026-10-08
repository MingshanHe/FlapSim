"""Freeze policies, run held-out comparisons, and render honest test videos."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from . import compare_env as common
from .train import write_json


def final_test(root):
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text())
    campaign=json.loads((root/'campaign_status.json').read_text())
    if campaign['status']!='completed' or len(campaign['completed'])!=4:
        raise RuntimeError('All four training runs must finish before final testing')
    destination=root/'final_test'; destination.mkdir(exist_ok=False)
    payloads={}; hashes={}; selections={}
    for method in ('neat','ppo'):
        for seed in protocol['seeds']:
            label=f'{method}_seed{seed}'; run=root/label
            path=run/'best_policy.json'
            payloads[label]=json.loads(path.read_text()); hashes[label]=hashlib.sha256(path.read_bytes()).hexdigest()
            selections[label]=json.loads((run/'results.json').read_text())
            write_json(destination/(label+'_frozen.json'),payloads[label])
    payloads['constant_trim']=json.loads((root/'trim_seed.json').read_text())
    write_json(destination/'frozen_selection.json',dict(policy_sha256=hashes,selections=selections,
               selection_basis='Validation only; all four policies frozen before any held-out test'))
    tasks=[(label,group,case) for label in payloads for group in ('nominal','stress') for case in protocol['test_'+group]]
    episodes={label:{group:[] for group in ('nominal','stress')} for label in payloads}
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=12,initializer=common.init_worker,initargs=(str(root),)) as pool:
        jobs=[(payloads[label],[case],True) for label,group,case in tasks]
        for index,((label,group,case),rows) in enumerate(zip(tasks,pool.map(common.evaluate,jobs))):
            episodes[label][group].extend(rows)
            if (index+1)%32==0:
                write_json(destination/f'{label}_{group}.json',episodes[label][group])
                print(json.dumps(dict(completed_episodes=index+1,total=len(tasks),label=label,group=group,
                                     summary=common.summarize(episodes[label][group]))),flush=True)
        # Deterministic replay of the first predeclared nominal case, no case selection.
        verification=[]
        for label in payloads:
            check=list(pool.map(common.evaluate,[(payloads[label],[protocol['test_nominal'][0]],True)]))[0][0]
            expected=episodes[label]['nominal'][0]
            np.testing.assert_array_equal(check['trace'],expected['trace'])
            verification.append(dict(policy=label,seed=check['seed'],trace_exact_match=True))
    summary={label:{group:common.summarize(rows) for group,rows in groups.items()} for label,groups in episodes.items()}
    result=dict(status='completed',summary=summary,test_control_steps=sum(r['steps'] for groups in episodes.values() for rows in groups.values() for r in rows),
                replay_verification=verification,wall_seconds=time.monotonic()-started)
    write_json(destination/'results.json',result)
    return result


def record_video(root,label,group='nominal'):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    root=Path(root); protocol=json.loads((root/'protocol.json').read_text())
    payload=json.loads((root/'final_test'/(label+'_frozen.json')).read_text())
    case=protocol['test_'+group][0]
    env=common.CompareEnv(root,render_mode='rgb_array'); env.configure_case(case)
    policy=common.make_policy(payload)
    directory=root/'videos'; directory.mkdir(exist_ok=True)
    path=directory/f'{label}_{group}.mp4'
    writer=imageio.get_writer(str(path),fps=25,codec='libx264')
    def frame(info):
        picture=Image.fromarray(env.render()); draw=ImageDraw.Draw(picture)
        draw.rectangle((0,0,picture.width,84),fill=(17,29,43))
        draw.text((10,7),f'{label} | upper 7 g + lower 8 g | IMU 6 -> control 3',fill='white')
        draw.text((10,25),f'{group}, fixed test seed {case["seed"]} | 4x slow motion',fill='white')
        height=env.sim.data.xpos[env.body,2]
        draw.text((10,43),f'Time {info.get("elapsed",0):.2f} s | height {height:.2f} m | pos RMSE {100*info.get("tracking_rmse",0):.1f} cm | tilt RMS {info.get("tilt_rmse_deg",0):.1f} deg',fill='white')
        status=info.get('termination_reason') or 'flying'
        draw.text((10,61),f'Status: {status} | goal: 8 s, RMSE <10 cm, tilt RMS <10 deg',fill='white')
        return np.asarray(picture)
    try:
        obs,info=env.reset(seed=case['seed']); policy.reset(obs); writer.append_data(frame(info))
        while True:
            obs,_,terminated,truncated,info=env.step(policy.act(obs))
            writer.append_data(frame(info))
            if terminated or truncated: break
        expected=json.loads((root/'final_test'/f'{label}_{group}.json').read_text())[0]
        assert info['elapsed']==expected['elapsed'] and info['termination_reason']==expected['termination']
        assert np.isclose(info['tracking_rmse'],expected['tracking_rmse_m'],atol=1e-12,rtol=0)
        # Hold the final status for one playback second without extending flight time.
        final_frame=frame(info)
        for _ in range(25): writer.append_data(final_frame)
        write_json(path.with_suffix('.json'),dict(label=label,case=case,elapsed=info['elapsed'],success=info['hover_success'],
                   termination=info['termination_reason'],tracking_rmse_m=info['tracking_rmse'],tilt_rmse_deg=info['tilt_rmse_deg'],slowdown=4))
    finally: writer.close(); env.close()
    return str(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--video-label'); parser.add_argument('--group',default='nominal',choices=['nominal','stress'])
    args=parser.parse_args()
    if args.video_label: print(record_video(args.root,args.video_label,args.group))
    else: print(json.dumps(final_test(args.root),indent=2))


if __name__=='__main__': main()
