"""Render physical qpos, including high-rate wing motion. Adapted from the frozen renderer."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from ..pid_airborne.results import portable
from ..topology_hover.environment import DEFAULT_SOURCE
from .environment import HoverEnv
from .common import write


def render(root,condition,third="ann_neat"):
    labels=[f'{m}_{condition}' for m in ['pid','snn_neat',third]]
    traces=[np.load(root/'traces'/f'{label}.npz')['trace'] for label in labels]
    metas=[json.loads((root/'traces'/f'{label}.json').read_text())['episode'] for label in labels]
    source_seed=metas[0]['seed'];horizon=metas[0]['case']['seconds']
    assert all(meta['seed']==source_seed and meta['case']['seconds']==horizon for meta in metas)
    videos=root/'videos';videos.mkdir(exist_ok=True)
    font=ImageFont.truetype('DejaVuSans.ttf',19);small=ImageFont.truetype('DejaVuSans.ttf',15)
    with portable(DEFAULT_SOURCE) as stage:
        shutil.copyfile(Path(DEFAULT_SOURCE)/'selected_pid.json',stage/'selected_pid.json')
        env=HoverEnv(source=stage);m=env.sim.model;d=env.sim.data
        renderer=mujoco.Renderer(m,height=500,width=480);closeup=mujoco.Renderer(m,height=210,width=210)
        camera=mujoco.MjvCamera();camera.distance=1.15;camera.azimuth=135;camera.elevation=-15
        def scene(row,distance=1.15,follow_height=False,target=None):
            d.qpos[:]=row[25:];d.qvel[:]=0.;d.time=float(row[0]);mujoco.mj_forward(m,d)
            np.testing.assert_allclose(d.xpos[env.air.env.body],row[1:4],rtol=0,atol=1e-9)
            camera.distance=distance;camera.lookat[:]=[row[1],row[2],row[3]+.015 if follow_height else .48]
            active_renderer=renderer if target is None else target
            active_renderer.update_scene(d,camera=camera);return Image.fromarray(active_renderer.render())
        path=videos/f'comparison_{condition}_{third}.mp4';writer=imageio.get_writer(path,fps=60,codec='libx264',quality=8,macro_block_size=None)
        last=None
        try:
            for frame in range(round(horizon*60)):
                t=(frame+1)/60;pic=Image.new('RGB',(1440,768),(17,31,46));draw=ImageDraw.Draw(pic)
                draw.text((20,12),f'7 g + 8 g | {condition.upper()} | same seed {source_seed} | playback 1x',font=font,fill='white')
                for col,(label,trace,meta) in enumerate(zip(labels,traces,metas)):
                    i=min(np.searchsorted(trace[:,0],t),len(trace)-1);row=trace[i];x=col*480
                    pic.paste(scene(row),(x,100));draw=ImageDraw.Draw(pic)
                    pic.paste(scene(row,.31,True,closeup),(x+252,344));draw=ImageDraw.Draw(pic)
                    draw.rectangle((x+251,343,x+463,555),outline=(110,155,180),width=1)
                    draw.text((x+252,322),'BODY CLOSE-UP',font=small,fill=(160,220,235))
                    draw.text((x+16,53),label,font=font,fill=(100,220,230))
                    ended=t>meta['elapsed']+.02 and meta['termination']!='time_limit'
                    draw.text((x+16,80),f't={row[0]:.2f} s'+(' | STOPPED: '+meta['termination'] if ended else ''),font=small,fill=(255,105,100) if ended else 'white')
                    draw.text((x+16,610),f'Height {row[3]:.3f} m | tilt {np.rad2deg(row[7]):.1f} deg',font=font,fill='white')
                    draw.text((x+16,644),f'alpha {np.rad2deg(row[8]):+.2f} / beta {np.rad2deg(row[9]):+.2f} deg',font=small,fill=(190,210,220))
                    draw.text((x+16,672),f'Motor target {row[10]:.0f} rad/s | actor {row[22]:+.2f}, {row[23]:+.2f}, {row[24]:+.2f}',font=small,fill=(190,210,220))
                    drift=np.linalg.norm(row[1:3]-np.asarray(meta['reference_position'])[:2])
                    draw.text((x+16,701),f'Episode: {"PASS" if meta["success"] else "FAIL"} | XY drift {drift:.2f} m',font=font,fill=(135,220,135) if meta['success'] else (255,150,120))
                draw.text((20,739),'Recorded states, no invented motion. Camera follows XY; height reference remains fixed. XY and yaw are uncontrolled.',font=small,fill=(170,190,205))
                last=np.asarray(pic);writer.append_data(last)
                if frame==min(300,round(horizon*60)-1):pic.save(videos/f'comparison_{condition}_{third}_preview.png')
            for _ in range(60):writer.append_data(last)
        finally:writer.close();renderer.close();closeup.close()
        # High-rate, slow-motion wing close-up from the same recorded SNN-NEAT trajectory.
        trace=traces[1];start=5. if trace[-1,0]>=6 else max(0.,trace[-1,0]-1.)
        end=min(start+1.,trace[-1,0]);segment=trace[(trace[:,0]>start)&(trace[:,0]<=end)][::2]
        renderer=mujoco.Renderer(m,height=500,width=480);slow=videos/f'snn_wings_{condition}_10x.mp4'
        writer=imageio.get_writer(slow,fps=25,codec='libx264',quality=8,macro_block_size=None)
        wing_angles={name:[] for name in ['left_flapping_joint','right_flapping_joint']}
        try:
            for row in segment:
                pic=Image.new('RGB',(960,620),(17,31,46));pic.paste(scene(row,.31,True),(0,65));draw=ImageDraw.Draw(pic)
                draw.text((20,16),f'SNN-NEAT | {condition} | actual wing motion | 10x slow',font=font,fill='white')
                draw.text((500,85),f'Simulation time {row[0]:.3f} s',font=font,fill='white')
                for j,name in enumerate(wing_angles):
                    value=np.rad2deg(d.joint(name).qpos[0]);wing_angles[name].append(float(value))
                    draw.text((500,140+j*40),f'{name}: {value:+.1f} deg',font=small,fill=(100,220,230))
                draw.text((500,250),'250 state samples/s -> 25 video fps',font=small,fill='white')
                draw.text((500,290),'No interpolated or artificial flapping.',font=small,fill='white')
                draw.text((20,582),'Camera follows the body. This close-up shows wing motion, not station keeping.',font=small,fill=(180,200,215))
                writer.append_data(np.asarray(pic))
        finally:writer.close();renderer.close()
    write(videos/f'verification_{condition}_{third}.json',dict(source_labels=labels,source_seed=source_seed,simulation_horizon_seconds=horizon,
        frame_position_check_atol=1e-9,flight_fps=60,playback_speed=1,hold_seconds=1,
        body_closeup_overlay=True,
        verdict_is_whole_episode_strict_criteria=True,
        wing_sample_hz=250,wing_playback_fps=25,wing_slowdown=10,wing_segment=[start,end],
        wing_joint_ranges_deg={name:dict(minimum=min(values),maximum=max(values),peak_to_peak=max(values)-min(values)) for name,values in wing_angles.items()},
        note='Reconstructed from recorded MuJoCo qpos. Paired rollout metrics checked separately to 1e-12. Terminated panels freeze and are visibly marked STOPPED.'))
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for condition,third in [('nominal','ann_neat'),('stress','ann_neat'),('stress','rnn_ppo'),('angle30','rnn_ppo')]:
        print(render(a.root,condition,third),flush=True)
if __name__=='__main__':main()
