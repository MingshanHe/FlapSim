"""Audit acquisition aliasing using every physics-step IMU sample.

This expensive, ideal sensor diagnostic is not a hardware-rate recommendation.
Position and velocity ground truth are used only for scoring / an explicitly
labeled initial-velocity upper bound.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from .imu_history_probe import collect, integrate
from .train import write_json


def run(root):
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text())
    case=protocol['cases'][0]
    arrays,episode=collect(root,case,200000)
    time=arrays['time']; rotation=arrays['rotation']; force=arrays['accel']
    truth=arrays['position']-arrays['position'][0]
    predictions={}; metrics=[]

    def evaluate(label,t,r,a,indices):
        p,v=integrate(t,r,a,arrays['velocity'][0])
        errors=np.linalg.norm(p-truth[indices],axis=1)
        metrics.append(dict(label=label,elapsed_s=float(t[-1]),
                            rmse_m=float(np.sqrt(np.mean(errors**2))),
                            final_error_m=float(errors[-1]),
                            final_velocity_error_m_s=float(np.linalg.norm(v[-1]-arrays['velocity'][indices[-1]]))))
        predictions[label]=np.column_stack([t,p,truth[indices]])

    evaluate('physics_rate_ideal',time,rotation,force,np.arange(len(time)))
    for rate in (100,500,2000):
        indices=np.arange(0,len(time),200000//rate)
        evaluate(f'raw_{rate}Hz',time[indices],rotation[indices],force[indices],indices)
    # Causal two-pole 100 Hz sensor low-pass BEFORE decimation. This is a
    # hypothetical analog/internal-oversampling front end, not measured hardware.
    rho=np.exp(-2*np.pi*100/200000)
    for initialization in ('first_sample','stationary_prior'):
        initial=force[0] if initialization=='first_sample' else rotation[0].T@np.array([0,0,9.81])
        filtered=force.copy()
        for _ in range(2):
            filtered,_=lfilter([1-rho],[1,-rho],filtered,axis=0,zi=(rho*initial)[None])
        for rate in (500,2000):
            indices=np.arange(0,len(time),200000//rate)
            evaluate(f'prefilter100Hz_2pole_{rate}Hz_{initialization}',time[indices],rotation[indices],filtered[indices],indices)
    write_json(root/'sampling_audit.json',dict(episode=episode,metrics=metrics,
        scope='Single first predeclared case; ideal attitude and oracle v0; causal prefilter sensitivity, not a deployed estimator',
        initial_velocity_m_s=arrays['velocity'][0].tolist(),
        peak_specific_force_m_s2=float(np.max(np.linalg.norm(force,axis=1))),
        source_rate_hz=200000,filter_cutoff_per_pole_hz=100))
    # Keep only control-rate scores and a short raw trace; avoid huge datasets.
    np.savez_compressed(root/'sampling_audit_traces.npz',**{k:v[::max(1,len(v)//1000)] for k,v in predictions.items()})
    np.savez_compressed(root/'sampling_short_trace.npz',time=time[:4001],accel=force[:4001],
                        filtered=filtered[:4001])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for label in ('raw_100Hz','raw_2000Hz','prefilter100Hz_2pole_2000Hz_stationary_prior','physics_rate_ideal'):
        a=predictions[label]
        axes[0].plot(a[:,0],100*np.linalg.norm(a[:,1:4]-a[:,4:7],axis=1),label=label)
    axes[0].set(xlabel='Time (s)',ylabel='Displacement error (cm)',title='Same flight; ideal attitude and initial velocity')
    axes[0].legend(fontsize=7)
    axes[1].plot(time[:4001]*1000,force[:4001,0],lw=.7,label='Physics-rate raw sensor')
    axes[1].plot(time[:4001]*1000,filtered[:4001,0],lw=1,label='Causal prefilter')
    axes[1].set(xlabel='First 20 ms',ylabel='Body-x specific force (m/s²)',title='High-frequency impulses require sensor modeling')
    axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/'sampling_audit.png',dpi=170);plt.close(fig)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.root),indent=2))
