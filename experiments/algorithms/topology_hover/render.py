"""Render recorded physical trajectories; never invent wing animation."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..pid_airborne.results import portable
from .environment import HoverEnv,DEFAULT_SOURCE
from .common import write


def plot_trace(root,label):
    data=np.load(root/'traces'/f'{label}.npz');x=data['trace'];a=data['activations']
    meta=json.loads((root/'traces'/f'{label}.json').read_text())['episode'];reference=meta['reference_position'][2]
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True,constrained_layout=True)
    axes[0,0].plot(x[:,0],x[:,3],label='True height');axes[0,0].plot(x[:,0],reference+x[:,20],label='Range observer');axes[0,0].axhline(reference,color='gray',ls='--');axes[0,0].set_ylabel('Height (m)')
    axes[0,1].plot(x[:,0],np.rad2deg(x[:,7]),label='True tilt');axes[0,1].set_ylabel('Tilt (deg)')
    axes[1,0].plot(x[:,0],np.rad2deg(x[:,8]),label='alpha');axes[1,0].plot(x[:,0],np.rad2deg(x[:,9]),label='beta');axes[1,0].set_ylabel('Command (deg)')
    axes[1,1].plot(x[:,0],x[:,10],label='Motor target');axes[1,1].set_ylabel('Motor command (rad/s)')
    for index,name in enumerate(['alpha','beta','motor']):axes[2,0].plot(a[:,0],a[:,12+index],label=name)
    axes[2,0].set_ylabel('Actor outputs');axes[2,0].set_xlabel('Time (s)')
    axes[2,1].plot(x[:,0],np.linalg.norm(x[:,1:3]-np.array(meta['reference_position'])[:2],axis=1),label='XY displacement');axes[2,1].set_ylabel('XY drift (m)');axes[2,1].set_xlabel('Time (s)')
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(f'{label} | seed {meta["seed"]} | yaw / XY uncontrolled')
    out=root/'figures';out.mkdir(exist_ok=True);fig.savefig(out/f'{label}.png',dpi=160);plt.close(fig)
    if label.startswith('neat') and a.shape[1]>15:
        spec_meta=json.loads((root/'representatives.json').read_text())['neat']
        spec=json.loads((root/spec_meta['path']).read_text());states=a[:,15:]
        fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
        im=ax.imshow(states.T,aspect='auto',origin='lower',extent=(a[0,0],a[-1,0],-.5,states.shape[1]-.5),cmap='RdBu',vmin=-1,vmax=1)
        ax.set_yticks(range(states.shape[1]),[str(n['id']) for n in spec['nodes']]);ax.set(xlabel='Time (s)',ylabel='Node ID',title='Node activation during flight (weights remain fixed)')
        fig.colorbar(im,ax=ax,label='tanh state');fig.savefig(out/f'{label}_activations.png',dpi=160);plt.close(fig)
        from .visualize import INPUTS
        observed=a[:,1:1+spec['inputs']]
        fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
        im=ax.imshow(observed.T,aspect='auto',origin='lower',extent=(a[0,0],a[-1,0],-.5,spec['inputs']-.5),cmap='RdBu',vmin=-1,vmax=1)
        ax.set_yticks(range(spec['inputs']),INPUTS[:spec['inputs']])
        ax.set(xlabel='Time (s)',title='Recorded actor inputs (100 Hz, fixed normalization)')
        fig.colorbar(im,ax=ax,label='Normalized input (color clipped at +/-1)')
        fig.savefig(out/f'{label}_observations.png',dpi=160);plt.close(fig)


def render(root,condition):
    labels=[f'{m}_{condition}' for m in ['pid','neat','rnn_ppo']]
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
        path=videos/f'comparison_{condition}.mp4';writer=imageio.get_writer(path,fps=60,codec='libx264',quality=8,macro_block_size=None)
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
                if frame==min(300,round(horizon*60)-1):pic.save(videos/f'comparison_{condition}_preview.png')
            for _ in range(60):writer.append_data(last)
        finally:writer.close();renderer.close();closeup.close()
        # High-rate, slow-motion wing close-up from the same recorded NEAT trajectory.
        trace=traces[1];start=5. if trace[-1,0]>=6 else max(0.,trace[-1,0]-1.)
        end=min(start+1.,trace[-1,0]);segment=trace[(trace[:,0]>start)&(trace[:,0]<=end)][::2]
        renderer=mujoco.Renderer(m,height=500,width=480);slow=videos/f'neat_wings_{condition}_10x.mp4'
        writer=imageio.get_writer(slow,fps=25,codec='libx264',quality=8,macro_block_size=None)
        wing_angles={name:[] for name in ['left_flapping_joint','right_flapping_joint']}
        try:
            for row in segment:
                pic=Image.new('RGB',(960,620),(17,31,46));pic.paste(scene(row,.31,True),(0,65));draw=ImageDraw.Draw(pic)
                draw.text((20,16),f'NEAT | {condition} | actual wing motion | 10x slow',font=font,fill='white')
                draw.text((500,85),f'Simulation time {row[0]:.3f} s',font=font,fill='white')
                for j,name in enumerate(wing_angles):
                    value=np.rad2deg(d.joint(name).qpos[0]);wing_angles[name].append(float(value))
                    draw.text((500,140+j*40),f'{name}: {value:+.1f} deg',font=small,fill=(100,220,230))
                draw.text((500,250),'250 state samples/s -> 25 video fps',font=small,fill='white')
                draw.text((500,290),'No interpolated or artificial flapping.',font=small,fill='white')
                draw.text((20,582),'Camera follows the body. This close-up shows wing motion, not station keeping.',font=small,fill=(180,200,215))
                writer.append_data(np.asarray(pic))
        finally:writer.close();renderer.close()
    write(videos/f'verification_{condition}.json',dict(source_labels=labels,source_seed=source_seed,simulation_horizon_seconds=horizon,
        frame_position_check_atol=1e-9,flight_fps=60,playback_speed=1,hold_seconds=1,
        body_closeup_overlay=True,
        verdict_is_whole_episode_strict_criteria=True,
        wing_sample_hz=250,wing_playback_fps=25,wing_slowdown=10,wing_segment=[start,end],
        wing_joint_ranges_deg={name:dict(minimum=min(values),maximum=max(values),peak_to_peak=max(values)-min(values)) for name,values in wing_angles.items()},
        note='Reconstructed from recorded MuJoCo qpos. Paired rollout metrics checked separately to 1e-12. Terminated panels freeze and are visibly marked STOPPED.'))
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for path in sorted((a.root/'traces').glob('*.npz')):plot_trace(a.root,path.stem)
    from .flight_browser import build
    for path in sorted((a.root/'traces').glob('neat_*.npz')):build(a.root,path.stem)
    for condition in ['nominal','stress']:print(render(a.root,condition),flush=True)

if __name__=='__main__':main()
