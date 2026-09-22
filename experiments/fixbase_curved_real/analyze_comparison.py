"""Read-only analysis of all 18 real runs against the saved single-wing sweep."""
from pathlib import Path
import csv
import json
import re
import sys
import hashlib
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inspect_planform import planform

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
OUTPUT = FOLDER/'analysis'
SIM = ROOT/'experiments/fixbase/results_single_wing_10_15hz'


def load_force(path):
    a = np.loadtxt(path, delimiter=',', skiprows=4, usecols=range(7), encoding='cp932')
    assert np.isfinite(a).all() and np.all(np.diff(a[:,0])>0)
    return a


def window_mean(a, start, end, column=3):
    mask = (a[:,0]>=start)&(a[:,0]<end)
    return float(a[mask,column].mean())


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    OUTPUT.mkdir(exist_ok=True)
    runs=[]
    sources=[]
    paths=sorted((FOLDER/'0831_curve_capton').rglob('*.csv.csv'))
    assert len(paths)==18
    for path in paths:
        a=load_force(path)
        filtered=path.with_name(path.name.replace('.csv.csv','.csvfc10Hz.csv'))
        b=load_force(filtered)
        np.testing.assert_allclose(a[:,0],b[:,0])
        encoder=path.with_name(path.name.replace('.csv.csv','_encoder.csv'))
        e=np.genfromtxt(encoder,delimiter=',',names=True)
        power=path.with_name(path.name.replace('.csv.csv','_power.csv'))
        p=np.genfromtxt(power,delimiter=',',names=True)
        hz=int(re.search(r'_(\d+)Hz_',path.name)[1])
        on=(a[:,0]>=.5)&(a[:,0]<4)
        off=(a[:,0]>=5.5)&(a[:,0]<a[-1,0]-.1)
        time=e['t_s']; rotations=-e['shaft_deg']/360
        interior=(time>=.5)&(time<time[-1]-.3)
        encoder_hz=float(np.polyfit(time[interior],rotations[interior],1)[0])
        delta=window_mean(a,.5,4)-window_mean(a,5.5,a[-1,0]-.1)
        delta_f=window_mean(b,.5,4)-window_mean(b,5.5,b[-1,0]-.1)
        alternatives=[window_mean(a,l,r)-window_mean(a,lo,a[-1,0]-.1)
                      for l,r in ((.5,3.5),(1,4),(.5,4)) for lo in (5.5,6)]
        # Event timing based on force vibration, independent of force DC level.
        blocks=[(t,np.linalg.norm(a[(a[:,0]>=t)&(a[:,0]<t+.2),1:3].std(0)))
                for t in np.arange(0,a[-1,0]-.2,.2)]
        threshold=np.median([v for t,v in blocks if .5<t<3.5])*.25
        active=[t for t,v in blocks if v>threshold]
        row=dict(trial=path.parent.name,target_hz=hz,encoder_hz=encoder_hz,
                 duration_s=float(a[-1,0]),raw_full_mean_mN=float(a[:,3].mean()*1000),
                 filtered_full_mean_mN=float(b[:,3].mean()*1000),
                 on_mean_mN=float(a[on,3].mean()*1000),off_baseline_mN=float(a[off,3].mean()*1000),
                 corrected_Fz_mN=delta*1000,filtered_corrected_Fz_mN=delta_f*1000,
                 window_sensitivity_min_mN=min(alternatives)*1000,
                 window_sensitivity_max_mN=max(alternatives)*1000,
                 Fx_on_std_N=float(a[on,1].std()),Fx_off_std_N=float(a[off,1].std()),
                 force_vibration_last_active_s=float(max(active)+.2),
                 encoder_end_s=float(time[-1]),real_voltage_V=float(p['voltage_v'].mean()),
                 real_current_A=float(p['current_a'].mean()),real_input_power_W=float(p['power_w'].mean()),
                 source=str(path.relative_to(FOLDER)))
        runs.append(row)
        for source in (path,filtered,encoder,power):
            sources.append({'file':str(source.relative_to(FOLDER)),
                            'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    write_csv(OUTPUT/'run_statistics.csv',runs)
    sim=json.loads((SIM/'sweep.json').read_text())
    summary=[]
    for item in sim:
        hz=item['target_wing_frequency_hz']
        selected=[r for r in runs if r['target_hz']==hz]
        corrected=np.array([r['corrected_Fz_mN'] for r in selected])
        f=np.mean([r['encoder_hz'] for r in selected])
        summary.append(dict(target_hz=hz,real_encoder_hz=f,real_mean_Fz_mN=float(corrected.mean()),
                   real_between_run_sd_mN=float(corrected.std(ddof=1)),
                   real_naive_full_mean_mN=float(np.mean([r['raw_full_mean_mN'] for r in selected])),
                   filtered_corrected_mean_mN=float(np.mean([r['filtered_corrected_Fz_mN'] for r in selected])),
                   simulation_Fz_mN=item['mean_lift_mN'],simulation_hz=item['measured_wing_frequency_hz'],
                   overprediction_percent=float(100*(item['mean_lift_mN']/corrected.mean()-1)),
                   real_input_power_W=float(np.mean([r['real_input_power_W'] for r in selected])),
                   n_runs=len(selected)))
    write_csv(OUTPUT/'comparison.csv',summary)

    radius,chord,center,dr=planform(ROOT/'flapsim/robot/IST_robot/meshes/left_wing.stl')
    radius2,chord2,_,dr2=planform(ROOT/'flapsim/robot/IST_robot/meshes/left_wing.stl',bins=2400)
    area=float(chord.sum()*dr); i2=float((chord*radius**2).sum()*dr)
    assert abs(chord2.sum()*dr2/area-1)<.001
    assert abs((chord2*radius2**2).sum()*dr2/i2-1)<.001
    geometry={'CAD_projected_area_cm2':area*1e4,'rectangle_area_cm2':.083*.03325*1e4,
              'CAD_to_rectangle_area_ratio':area/(.083*.03325),
              'CAD_radial_second_moment_m4':i2,'rectangle_radial_second_moment_m4':.03325*.083**3/3,
              'CAD_to_rectangle_radial_second_moment_ratio':i2/(.03325*.083**3/3),
              'note':'STL projection union in local y-z plane, positive span; includes wing frame silhouette'}
    np.savez(OUTPUT/'CAD_planform.npz',radius=radius,chord=chord,center=center,dr=dr)
    result={'protocol':{'on_window_s':[.5,4],'off_window_s':'[5.5, end-0.1]',
                        'baseline_assumption':'Tail is motor-off, same loading and sensor zero; pending user confirmation',
                        'replicates':'01/02/03 are repeat recordings, not necessarily independent wing specimens'},
            'summary':summary,'geometry':geometry,'sources':sources}
    (OUTPUT/'analysis.json').write_text(json.dumps(result,indent=2))

    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    x=np.array([r['target_hz'] for r in summary])
    ax.errorbar(x,[r['real_mean_Fz_mN'] for r in summary],
                yerr=[r['real_between_run_sd_mN'] for r in summary],fmt='o-',capsize=4,
                color='#2563eb',label='Measured: on minus off baseline (mean +/- run SD, n=3)')
    ax.plot(x,[r['simulation_Fz_mN'] for r in summary],'s-',color='#ea580c',label='Original single-wing simulation')
    ax.plot(x,[r['real_naive_full_mean_mN'] for r in summary],'x--',color='#94a3b8',label='Measured: naive whole-file mean')
    sensitivity_path=OUTPUT/'geometry_sensitivity/summary.json'
    if sensitivity_path.is_file():
        sensitivity=json.loads(sensitivity_path.read_text())
        selected=sorted([r for r in sensitivity if r['mode']=='full_planform'],key=lambda r:r['target_wing_frequency_hz'])
        ax.plot([r['target_wing_frequency_hz'] for r in selected],
                [r['mean_lift_mN'] for r in selected],'*',color='#16a34a',markersize=13,
                label='CAD geometry diagnostic: 10/15 Hz only, no fitted coefficients')
    for r in summary:
        ax.annotate(f"{r['real_mean_Fz_mN']:.2f}",(r['target_hz'],r['real_mean_Fz_mN']),
                    xytext=(0,12 if r['target_hz']==10 else -17),textcoords='offset points',ha='center',color='#1e40af')
    ax.set(xlabel='Nominal flapping frequency [Hz]',ylabel='Mean Fz [mN]',xticks=x,ylim=(-4,58),
           title='Single wing: separating baseline/window effects from model error')
    ax.grid(alpha=.2); ax.legend(fontsize=9,loc='upper left')
    fig.savefig(OUTPUT/'comparison.png',dpi=190); fig.savefig(OUTPUT/'comparison.svg');plt.close(fig)

    example=next(p for p in paths if p.parent.name=='01' and '_15Hz_' in p.name)
    a=load_force(example); b=load_force(example.with_name(example.name.replace('.csv.csv','.csvfc10Hz.csv')))
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    t=a[:,0]; smooth=sosfiltfilt(butter(4,3,fs=1200,output='sos'),a[:,3])
    axes[0].plot(t,a[:,3]*1000,lw=.35,alpha=.35,label='Raw Fz')
    axes[0].plot(t,smooth*1000,lw=2,label='3 Hz smooth, illustration only')
    axes[0].plot(t,b[:,3]*1000,lw=.6,alpha=.65,label='Provided fc10Hz file')
    axes[0].axvspan(.5,4,color='green',alpha=.10,label='On window')
    axes[0].axvspan(5.5,t[-1]-.1,color='gray',alpha=.2,label='Off baseline window')
    axes[0].set(xlabel='Force-file time [s]',ylabel='Fz [mN]',title='15 Hz / recording 01: the end of the force file is quiet')
    axes[0].legend(fontsize=8,ncol=3)
    axes[1].plot(t,a[:,1],lw=.5,label='Raw Fx (vibration indicator)')
    axes[1].set(xlabel='Force-file time [s]',ylabel='Fx [N]');axes[1].legend()
    e=np.genfromtxt(example.with_name(example.name.replace('.csv.csv','_encoder.csv')),delimiter=',',names=True)
    axes[2].plot(e['t_s'],-e['shaft_deg']/360)
    axes[2].set(xlabel='Encoder-file time [s]',ylabel='Output-shaft revolutions',
                title='Encoder continues to rotate: file clocks cannot be assumed synchronized')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'window_and_baseline.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axes[0].fill_between(radius*1000,(center-chord/2)*1000,(center+chord/2)*1000,alpha=.6,label='CAD projected planform')
    axes[0].plot([0,83,83,0,0],[-1.25,-1.25,32,32,-1.25],'--',label='Existing aerodynamic rectangle')
    axes[0].set(xlabel='Span coordinate [mm]',ylabel='Chord coordinate [mm]',title='Visual mesh is curved; aerodynamic strips are rectangular')
    axes[0].set_aspect('equal');axes[0].legend(fontsize=8)
    axes[1].plot(radius*1000,chord*radius**2*1e6,label='CAD c(r) r^2')
    axes[1].plot(radius*1000,np.full_like(radius,.03325)*radius**2*1e6,'--',label='Existing c r^2')
    axes[1].set(xlabel='Span coordinate [mm]',ylabel='Radial loading weight [scaled]',title=f'CAD / rectangle radial second moment = {geometry["CAD_to_rectangle_radial_second_moment_ratio"]:.3f}')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(OUTPUT/'planform_mismatch.png',dpi=190);plt.close(fig)
    print(json.dumps({'comparison':summary,'geometry':geometry},indent=2),flush=True)


if __name__=='__main__':
    analyze()
