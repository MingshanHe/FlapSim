"""Freeze every run, test without reselection, and show uncut first-case videos."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from . import common
from .environment import HistoryEnv
from ..neat_imu.train import write_json


def final_test(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    labels=[f'{m}_seed{s}' for m in ('neat','ppo') for s in protocol['seeds']]
    runs={label:json.loads((root/label/'results.json').read_text()) for label in labels}
    assert all(r['status']=='completed' for r in runs.values())
    out=root/'final_test';out.mkdir(exist_ok=False)
    payloads={label:json.loads((root/label/'best_policy.json').read_text()) for label in labels}
    payloads['shared_prior']=dict(format='zero_residual')
    for label,payload in payloads.items():write_json(out/f'{label}_frozen.json',payload)
    write_json(out/'frozen_selection.json',dict(runs=runs,
        hashes={label:hashlib.sha256((out/f'{label}_frozen.json').read_bytes()).hexdigest() for label in payloads},
        selection='Validation only. All models frozen before final testing; no post-test selection or tuning.'))
    groups=dict(nominal=protocol['test_nominal'],stress=protocol['test_stress'],
                zero_history=[dict(c,ablation='zero_history') for c in protocol['test_nominal']])
    tasks=[(label,group,c) for label in payloads for group in groups for c in groups[group]
           if label!='shared_prior' or group!='zero_history']
    episodes={label:{g:[] for g in groups if label!='shared_prior' or g!='zero_history'} for label in payloads}
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=12,initializer=common.init_worker,initargs=(str(root),)) as pool:
        jobs=[(payloads[label],[case],True) for label,group,case in tasks]
        for index,((label,group,case),rows) in enumerate(zip(tasks,pool.map(common.evaluate,jobs))):
            episodes[label][group].extend(rows)
            if len(episodes[label][group])==len(groups[group]):
                write_json(out/f'{label}_{group}.json',episodes[label][group])
                print(json.dumps(dict(completed=index+1,total=len(tasks),label=label,group=group,
                                     summary=common.summary(episodes[label][group]))),flush=True)
        replay=[]
        for label in payloads:
            row=list(pool.map(common.evaluate,[(payloads[label],[protocol['test_nominal'][0]],True)]))[0][0]
            np.testing.assert_array_equal(row['trace'],episodes[label]['nominal'][0]['trace'])
            replay.append(dict(label=label,seed=row['seed'],trace_exact_match=True))
    result=dict(status='completed',summary={label:{g:common.summary(rows) for g,rows in groups.items()} for label,groups in episodes.items()},
                replay_verification=replay,wall_seconds=time.monotonic()-started,
                test_control_steps=sum(r['steps'] for groups in episodes.values() for rows in groups.values() for r in rows),
                trace_columns=['seconds','roll_deg','pitch_deg','yaw_deg','alpha_rad','beta_rad','x_m','y_m','z_m','motor_rad_s',
                               'relative_x','relative_y','relative_z','estimated_x','estimated_y','estimated_z','estimated_vx','estimated_vy','estimated_vz','observer_error_m'])
    write_json(out/'results.json',result);return result


def video(root,label,group='nominal'):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    payload=json.loads((root/'final_test'/f'{label}_frozen.json').read_text())
    case=protocol['test_'+group][0]
    env=HistoryEnv(root,render_mode='rgb_array');env.configure_case(case)
    policy=common.make_policy(payload);out=root/'videos';out.mkdir(exist_ok=True)
    path=out/f'{label}_{group}.mp4';writer=imageio.get_writer(str(path),fps=25,codec='libx264')
    true_path=[];estimated_path=[]
    font=ImageFont.truetype('DejaVuSans.ttf',14)
    def frame(info):
        scene=Image.fromarray(env.render()).resize((960,720))
        picture=Image.new('RGB',(1280,720),(17,29,43));picture.paste(scene,(0,0));draw=ImageDraw.Draw(picture)
        draw.font=font
        draw.rectangle((0,0,picture.width,105),fill=(17,29,43))
        draw.text((10,6),f'{label} | 7+8 g | 9 IMU channels + inertial history | residual control',fill='white')
        draw.text((10,25),f'Predeclared {group} seed {case["seed"]} | uncut 4x slow motion',fill='white')
        draw.text((10,44),f'Time {info.get("elapsed",0):.2f} s | position RMS {100*info.get("tracking_rmse",0):.1f} cm | tilt RMS {info.get("tilt_rmse_deg",0):.1f} deg',fill='white')
        draw.text((10,63),f'Inertial estimate error {100*info.get("observer_error_m",0):.1f} cm | status {info.get("termination_reason") or "flying"}',fill='white')
        draw.text((10,82),'Success: 8 s, position RMS <10 cm, tilt RMS <10 deg; virtual-IMU simulation only',fill='white')
        truth=np.asarray(info.get('relative_position',np.zeros(3)))
        estimate=np.asarray(info.get('estimated_position',np.zeros(3)))
        def xy(p):return tuple(np.rint(np.array([1120.,300.])+np.clip(p[:2],-.6,.6)*np.array([200.,-200.])).astype(int))
        true_path.append(xy(truth));estimated_path.append(xy(estimate))
        draw.text((980,125),'Relative X-Y (fixed origin, +/-60 cm)',fill='white')
        draw.rectangle((1000,180,1240,420),outline=(110,125,140))
        draw.line((1000,300,1240,300),fill=(70,85,100));draw.line((1120,180,1120,420),fill=(70,85,100))
        draw.ellipse((1100,280,1140,320),outline=(200,180,90));draw.text((1143,286),'10 cm',fill=(200,180,90))
        if len(true_path)>1:
            draw.line(true_path,fill=(75,205,236),width=2);draw.line(estimated_path,fill=(250,151,75),width=2)
        for point,color in [(true_path[-1],(75,205,236)),(estimated_path[-1],(250,151,75))]:
            x,y=point;draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
        draw.text((980,440),'Cyan: true path (evaluation only)',fill=(75,205,236))
        draw.text((980,458),'Orange: IMU estimate',fill=(250,151,75))
        draw.text((980,493),f'Height change: {truth[2]*100:+.1f} cm',fill='white')
        draw.text((980,514),f'Estimated: {estimate[2]*100:+.1f} cm',fill=(250,151,75))
        draw.text((980,551),f'X/Y: {truth[0]*100:+.1f} / {truth[1]*100:+.1f} cm',fill='white')
        draw.text((980,580),'Camera follows the body;',fill=(190,200,210))
        draw.text((980,598),'the plot shows actual drift.',fill=(190,200,210))
        if np.max(np.abs(truth[:2]))>.6:draw.text((980,628),'Path exceeds plot boundary',fill=(250,151,75))
        return np.asarray(picture)
    try:
        obs,info=env.reset(seed=case['seed']);policy.reset(obs);writer.append_data(frame(info))
        while True:
            obs,_,done,truncated,info=env.step(policy.act(obs));writer.append_data(frame(info))
            if done or truncated:break
        expected=json.loads((root/'final_test'/f'{label}_{group}.json').read_text())[0]
        assert info['elapsed']==expected['elapsed'] and info['termination_reason']==expected['termination']
        np.testing.assert_allclose(info['tracking_rmse'],expected['tracking_rmse_m'],rtol=0,atol=1e-12)
        last=frame(info)
        for _ in range(25):writer.append_data(last)
        write_json(path.with_suffix('.json'),dict(label=label,case=case,elapsed=info['elapsed'],
            success=info['hover_success'],termination=info['termination_reason'],tracking_rmse_m=info['tracking_rmse'],
            tilt_rmse_deg=info['tilt_rmse_deg'],observer_final_error_m=info['observer_error_m'],slowdown=4,final_hold_playback_seconds=1))
    finally:writer.close();env.close()
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--video-label')
    p.add_argument('--group',choices=['nominal','stress'],default='nominal');a=p.parse_args()
    if a.video_label:print(video(a.root,a.video_label,a.group))
    else:print(json.dumps(final_test(a.root),indent=2))


if __name__=='__main__':main()
