"""Mean lift validation and constrained sensor-yaw identification.

Only yaw and per-record time phase are fitted. Force scale, Z, aerodynamic
coefficients, and geometry are not fitted. Frequencies 13--15 Hz are held out.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from analyze_comparison import load_force, write_csv
from rect_simulation import OUTPUT, FOLDER, AXES

HARMONICS = 4
NPHASE = 256


def harmonic_design(phase):
    return np.column_stack([np.ones(len(phase))] +
                           [v for k in range(1, HARMONICS+1) for v in (np.cos(k*phase), np.sin(k*phase))])


def harmonic_fit(t, force, frequency):
    design = harmonic_design(2*np.pi*frequency*t)
    coefficients = np.linalg.lstsq(design, force, rcond=None)[0]
    reconstructed = design@coefficients
    explained = 1-np.sum((force-reconstructed)**2, axis=0)/np.sum((force-force.mean(0))**2, axis=0)
    return coefficients, explained


def waveform(coef):
    return harmonic_design(np.arange(NPHASE)*2*np.pi/NPHASE)@coef


def sensor_harmonics(row):
    a = load_force(FOLDER/row['source'])
    keep = (a[:, 0] >= .5) & (a[:, 0] < 4)
    t = a[keep, 0]
    force = a[keep, 1:4]-np.array([row[f'{axis}_baseline'] for axis in AXES[:3]])
    # Track within-record speed drift with a narrow fundamental band. Choose
    # the strongest measured axis in that band; one shared phase for all axes.
    # This removes timing drift, not force amplitude or orientation errors.
    fs = 1/np.median(np.diff(a[:,0]))
    fundamental = sosfiltfilt(butter(3, [row['encoder_hz']-2.5,row['encoder_hz']+2.5],
                                   fs=fs,btype='bandpass',output='sos'), a[:,1:4],axis=0)
    axis = int(np.argmax(np.var(fundamental[keep],axis=0)))
    phase = np.unwrap(np.angle(hilbert(fundamental[:,axis])))[keep]
    frequency = np.polyfit(t,phase,1)[0]/(2*np.pi)
    design = harmonic_design(phase)
    coef = np.linalg.lstsq(design, force, rcond=None)[0]
    reconstructed = design@coef
    explained = 1-np.sum((force-reconstructed)**2,axis=0)/np.sum((force-force.mean(0))**2,axis=0)
    band = sosfiltfilt(butter(4,70,fs=fs,output='sos'),a[:,1:4],axis=0)[keep]
    band -= np.array([row[f'{axis}_baseline'] for axis in AXES[:3]])
    band_explained = 1-np.sum((band-reconstructed)**2,axis=0)/np.sum((band-band.mean(0))**2,axis=0)
    # DC is the declared baseline-subtracted window mean, not a fit adjustment.
    coef[0] = np.array([row[f'{axis}_mean'] for axis in AXES[:3]])
    return coef, float(frequency), explained, band_explained


def momentum_to_fixture_coefficients(aero, momentum, frequency):
    """F_fixture = F_aero - dP/dt in a shared real Fourier basis."""
    coef = np.array(aero,copy=True)
    for k in range(1,(len(coef)-1)//2+1):
        omega=2*np.pi*k*frequency
        coef[2*k-1] -= omega*momentum[2*k]
        coef[2*k] += omega*momentum[2*k-1]
    return coef


def simulation_harmonics(result):
    data = np.load(OUTPUT/result['source'])
    start, end = result['averaging_interval_s']
    t = data['time_s']
    keep = (t >= start) & (t <= end)
    f = np.column_stack([data[f'fixture_dynamic_{axis}_N'] for axis in AXES[:3]])
    coef, explained = harmonic_fit(t[keep], f[keep], result['measured_wing_frequency_hz'])
    if 'momentum_x' in data:
        # Sensor load = aerodynamic force - rate of robot linear momentum,
        # after weight tare. Differentiate Fourier coefficients, not impulsive
        # joint-stop reactions sampled at 2 kHz (which otherwise alias).
        aero = np.column_stack([data[f'aero_{axis}_N'] for axis in AXES[:3]])
        momentum = np.column_stack([data[f'momentum_{axis}'] for axis in 'xyz'])
        coef, _ = harmonic_fit(t[keep],aero[keep],result['measured_wing_frequency_hz'])
        pc, _ = harmonic_fit(t[keep],momentum[keep],result['measured_wing_frequency_hz'])
        coef=momentum_to_fixture_coefficients(coef,pc,result['measured_wing_frequency_hz'])
    # Period-mean inertia is zero. Use aerodynamic full-cycle force for DC;
    # separately report sampled-fixture integration residual as a check.
    coef[0] = np.array([result['mean_wrenches'][f'aero_{axis}_N'] for axis in AXES[:3]])
    return coef, explained


def yaw_rotate(values, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c,-s,0], [s,c,0], [0,0,1]])
    return values@R.T


def correlations(sim, real):
    # Fit the AC waveform. A DC horizontal resultant can be nearly zero and
    # cannot reliably identify orientation; mean-force errors are still reported.
    s, r = sim-sim.mean(0), real-real.mean(0)
    shifts = np.array([np.roll(s, k, axis=0) for k in range(NPHASE)])
    xy_scale = max(float(np.mean(np.sum(r[:,:2]**2, axis=1))), 1e-12)
    z_scale = max(float(np.mean(r[:,2]**2)), 1e-12)
    A = np.mean(shifts[:,:,0]*r[None,:,0]+shifts[:,:,1]*r[None,:,1], axis=1)/xy_scale
    B = np.mean(shifts[:,:,0]*r[None,:,1]-shifts[:,:,1]*r[None,:,0], axis=1)/xy_scale
    C = np.mean(shifts[:,:,2]*r[None,:,2], axis=1)/z_scale
    constant = ((np.mean(np.sum(s[:,:2]**2,axis=1))+xy_scale)/xy_scale
                +(np.mean(s[:,2]**2)+z_scale)/z_scale)
    return A, B, C, constant


def fit_yaw(pairs):
    angles = np.linspace(-np.pi, np.pi, 1441)
    terms = [correlations(p['sim_wave'], p['real_wave']) for p in pairs]
    costs = []
    for theta in angles:
        costs.append(np.mean([const-2*np.max(np.cos(theta)*A+np.sin(theta)*B+C) for A,B,C,const in terms]))
    costs = np.array(costs)
    best = int(np.argmin(costs))
    theta = float(angles[best])
    return theta, angles, costs


def evaluate(pair, theta):
    A, B, C, constant = correlations(pair['sim_wave'], pair['real_wave'])
    scores = np.cos(theta)*A+np.sin(theta)*B+C
    shift = int(np.argmax(scores))
    pred = yaw_rotate(np.roll(pair['sim_wave'], shift, axis=0), theta)
    actual = pair['real_wave']
    rmse = np.sqrt(np.mean((pred-actual)**2, axis=0))
    ac = actual-actual.mean(0)
    xy_real = np.sqrt(np.mean(np.sum(ac[:,:2]**2, axis=1)))
    xy_sim = np.sqrt(np.mean(np.sum((pred-pred.mean(0))[:,:2]**2, axis=1)))
    metrics = dict(phase_shift_deg=shift*360/NPHASE, yaw_deg=float(np.degrees(theta)),
                   normalized_cost=float(constant-2*scores[shift]),
                   Fx_rmse_mN=rmse[0]*1000, Fy_rmse_mN=rmse[1]*1000, Fz_rmse_mN=rmse[2]*1000,
                   xy_rms_real_mN=xy_real*1000, xy_rms_sim_mN=xy_sim*1000,
                   xy_rms_ratio=xy_sim/max(xy_real,1e-12),
                   xy_nrmse=float(np.sqrt(np.mean(np.sum((pred[:,:2]-actual[:,:2])**2,axis=1)))/max(xy_real,1e-12)))
    return pred, metrics


def compare():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    real = json.loads((OUTPUT/'real_runs.json').read_text())
    groups = json.loads((OUTPUT/'real_grouped.json').read_text())
    sims = json.loads((OUTPUT/'momentum_summary.json').read_text())
    assert len(sims) == 12 and len(real) == 36
    smap = {(r['kind'],r['nominal_hz']):r for r in sims}
    gmap = {(r['kind'],r['hz']):r for r in groups}
    comparisons = []
    for kind in ('curve','rect'):
        for hz in range(10,16):
            s, r = smap[kind,hz], gmap[kind,hz]
            row = dict(kind=kind,hz=hz,encoder_hz=r['encoder_hz'],simulation_hz=s['measured_wing_frequency_hz'],
                       real_lift_mN=r['Fz_mean']*1000,real_sd_mN=r['Fz_sd']*1000,
                       sim_lift_mN=s['mean_lift_mN'],relative_error_pct=(s['mean_lift_N']/r['Fz_mean']-1)*100)
            recorded=np.load(OUTPUT/s['source'])
            start,end=s['averaging_interval_s']
            for axis in AXES[:3]:
                row[f'real_{axis}_mN'] = r[f'{axis}_mean']*1000
                row[f'sim_{axis}_mN'] = s['mean_wrenches'][f'aero_{axis}_N']*1000
                row[f'fixture_mean_residual_{axis}_mN'] = (s['mean_wrenches'][f'fixture_dynamic_{axis}_N']-s['mean_wrenches'][f'aero_{axis}_N'])*1000
                endpoints=np.interp([start,end],recorded['time_s'],recorded[f'momentum_{axis[-1].lower()}'])
                row[f'mean_inertial_{axis}_mN'] = float(-np.diff(endpoints)[0]/(end-start)*1000)
            comparisons.append(row)
    write_csv(OUTPUT/'lift_comparison.csv', comparisons)
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,kind in zip(axes,('curve','rect')):
        rows=[r for r in comparisons if r['kind']==kind]
        x=[r['encoder_hz'] for r in rows]
        ax.errorbar(x,[r['real_lift_mN'] for r in rows],yerr=[r['real_sd_mN'] for r in rows],fmt='o-',color='black',capsize=4,label='Measured: mean +/- SD, n=3')
        ax.plot(x,[r['sim_lift_mN'] for r in rows],'s-',color='#1673b1',label='Simulation: geometry-specific BEM')
        ax.set(title=f'{kind}: single wing, pitch limit +/-26.1 deg',xlabel='Measured wing frequency [Hz]',ylabel='Mean Fz [mN]')
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(OUTPUT/'lift_validation.png',dpi=180);fig.savefig(OUTPUT/'lift_validation.svg');plt.close(fig)
    pairs=[]
    for r in real:
        sc,se = simulation_harmonics(smap[r['kind'],r['hz']])
        rc,f,re,be = sensor_harmonics(r)
        pairs.append(dict(kind=r['kind'],hz=r['hz'],repeat=r['repeat'],source=r['source'],
                          force_frequency_hz=f,encoder_hz=r['encoder_hz'],real_explained=re.tolist(),band_explained=be.tolist(),sim_explained=se.tolist(),
                          sim_wave=waveform(sc),real_wave=waveform(rc)))
    fits={};metrics=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,kind in zip(axes,('curve','rect','shared')):
        selected=[p for p in pairs if (kind=='shared' or p['kind']==kind)]
        train=[p for p in selected if p['hz']<=12]
        angle,grid,cost=fit_yaw(train)
        fit=dict(yaw_deg=float(np.degrees(angle)),training_frequencies_hz=[10,11,12],held_out_frequencies_hz=[13,14,15],
                 convention='F_sensor = Rz(yaw) @ F_sim; +Z unchanged',fit_parameters='One global yaw; one time phase per record; no amplitude/offset/tilt fit')
        smean=np.array([p['sim_wave'].mean(0)[:2] for p in train])
        rmean=np.array([p['real_wave'].mean(0)[:2] for p in train])
        dc_angle=np.arctan2(np.sum(smean[:,0]*rmean[:,1]-smean[:,1]*rmean[:,0]),np.sum(smean*rmean))
        fit['mean_force_only_yaw_deg']=float(np.degrees(dc_angle))
        fit['mean_force_fit_note']='Separate diagnostic on 10--12 Hz means; not substituted for waveform yaw'
        ax.plot(np.degrees(grid),cost-cost.min());ax.axvline(np.degrees(angle),color='red',ls='--')
        ax.set(title=f'{kind}: yaw={np.degrees(angle):.2f} deg',xlabel='Yaw [deg]',ylabel='Training loss above minimum');ax.grid(alpha=.2)
        near=np.degrees(grid[cost<=cost.min()+.01*max(cost.min(),.01)])
        fit['near_optimal_angle_span_deg']=[float(near.min()),float(near.max())]
        for split in ('train','test'):
            subset=[p for p in selected if (p['hz']<=12)==(split=='train')]
            values=[]
            for pair in subset:
                _,m=evaluate(pair,angle)
                single,_,_=fit_yaw([pair])
                m.update(fit=kind,split=split,kind=pair['kind'],hz=pair['hz'],repeat=pair['repeat'],
                         individual_best_yaw_deg=float(np.degrees(single)),force_frequency_hz=pair['force_frequency_hz'],
                         force_encoder_delta_hz=pair['force_frequency_hz']-pair['encoder_hz'],
                         **{f'real_{a}_harmonic_explained':pair['real_explained'][i] for i,a in enumerate(AXES[:3])},
                         **{f'real_{a}_70Hz_band_explained':pair['band_explained'][i] for i,a in enumerate(AXES[:3])})
                metrics.append(m);values.append(m)
            fit[f'{split}_xy_nrmse_mean']=float(np.mean([m['xy_nrmse'] for m in values]))
            fit[f'{split}_rmse_mN']={a:float(np.mean([m[f'{a}_rmse_mN'] for m in values])) for a in AXES[:3]}
            fit[f'{split}_xy_rms_ratio_mean']=float(np.mean([m['xy_rms_ratio'] for m in values]))
        fits[kind]=fit
    fig.savefig(OUTPUT/'yaw_identification.png',dpi=170);plt.close(fig)
    write_csv(OUTPUT/'orientation_metrics.csv',metrics)
    (OUTPUT/'orientation_fit.json').write_text(json.dumps(fits,indent=2))
    # Display held-out high-frequency waveforms with the frozen training yaw.
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ki,kind in enumerate(('curve','rect')):
        pair=next(p for p in pairs if p['kind']==kind and p['hz']==15 and p['repeat']=='01')
        pred,_=evaluate(pair,np.radians(fits[kind]['yaw_deg']))
        for i,ax in enumerate(axes[ki]):
            phase=np.arange(NPHASE)/NPHASE
            ax.plot(phase,pair['real_wave'][:,i]*1000,color='black',label='Measured: 1st-4th harmonics')
            ax.plot(phase,pred[:,i]*1000,color='#1673b1',label='Fixture simulation: rotated')
            ax.set(title=f'{kind}, 15 Hz held out, {AXES[i]}',xlabel='Cycle phase (time aligned)',ylabel='Force [mN]')
            ax.grid(alpha=.2)
            if i==0:ax.legend(fontsize=8)
    fig.savefig(OUTPUT/'force_waveform_validation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for kind,color,marker in [('curve','#1673b1','o'),('rect','#d55e00','s')]:
        rows=[r for r in comparisons if r['kind']==kind]
        s=np.array([[r[f'sim_{a}_mN'] for a in AXES[:3]] for r in rows])
        sr=yaw_rotate(s,np.radians(fits[kind]['yaw_deg']))
        for i,ax in enumerate(axes):
            ax.plot(range(10,16),[r[f'real_{AXES[i]}_mN'] for r in rows],marker+'-',color=color,label=f'{kind} measured')
            ax.plot(range(10,16),sr[:,i],marker+'--',color=color,label=f'{kind} sim, yaw fixed')
            ax.set(title=f'Mean {AXES[i]}',xlabel='Nominal wing frequency [Hz]',ylabel='Force [mN]');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(OUTPUT/'mean_force_xyz.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ki,kind in enumerate(('curve','rect')):
        row=next(r for r in real if r['kind']==kind and r['hz']==15 and r['repeat']=='01')
        a=load_force(FOLDER/row['source']);keep=(a[:,0]>=.5)&(a[:,0]<4)
        f,p=welch(a[keep,1:4],fs=1/np.median(np.diff(a[:,0])),nperseg=2048,axis=0)
        s=smap[kind,15];d=np.load(OUTPUT/s['source']);keep=d['time_s']>=.6
        values=np.column_stack([d[f'fixture_dynamic_{axis}_N'] for axis in AXES[:3]])
        values=yaw_rotate(values,np.radians(fits[kind]['yaw_deg']))
        sf,sp=welch(values[keep],fs=2000,nperseg=2048,axis=0)
        for i,ax in enumerate(axes[ki]):
            ax.semilogy(f,p[:,i]*1e6,color='black',label='Measured raw')
            ax.semilogy(sf,sp[:,i]*1e6,color='#1673b1',label='Sampled fixture (impulsive)')
            ax.axvspan(0,70,color='green',alpha=.08)
            ax.set(xlim=(0,350),ylim=(1e-3,1e5),title=f'{kind}, 15 Hz, {AXES[i]}',xlabel='Frequency [Hz]',ylabel='PSD [mN^2/Hz]')
            if i==0:ax.legend(fontsize=8)
    fig.savefig(OUTPUT/'force_spectra.png',dpi=160);plt.close(fig)
    arrays={}
    for p in pairs:
        key=f'{p["kind"]}_{p["hz"]}_{p["repeat"]}'
        arrays[key+'_sim']=p['sim_wave'];arrays[key+'_real']=p['real_wave']
    np.savez_compressed(OUTPUT/'harmonic_waveforms.npz',**arrays)
    (OUTPUT/'comparison_summary.json').write_text(json.dumps(dict(lift=comparisons,orientation=fits),indent=2))
    print(json.dumps(dict(lift=comparisons,orientation=fits),indent=2))
    return comparisons,fits


if __name__=='__main__':
    compare()
