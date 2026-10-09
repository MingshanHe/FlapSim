"""Predeclare shared training and held-out cases before policy optimization."""
import argparse
import hashlib
import json
from pathlib import Path
from ..neat_imu.activation import cases
from ..neat_imu.train import write_json
from .environment import NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root
    prior=Path('artifacts/neat_ppo_7g8g_20261008/neat_seed836/best_policy.json')
    (root/'attitude_prior.json').write_bytes(prior.read_bytes())
    validation=cases(range(302010000,302010008),8.)
    for c in validation[4:]:c.update(acceleration_bias=.005,attitude_bias_degrees=.05)
    nominal=cases(range(302020000,302020024),8.)
    stress=cases(range(302030000,302030024),8.,True)
    for c in stress:c.update(acceleration_bias=.03,attitude_bias_degrees=.3,accel_noise=.02,initial_velocity=.05)
    write_json(root/'protocol.json',dict(version=1,seeds=[941,942],nominal_steps=262144,population=96,
        controller='Shared frozen six-IMU NEAT attitude prior; actor adds +/-2 normalized alpha/beta corrections and sets motor residual',
        initial_actor='Zero residual for both algorithms; motor starts at hovering calibration',
        prior_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
        input_names=NAMES,raw_sensor_channels=9,actor_features=15,
        raw_scales=[.35]*3+[10.]*3+[9.81]*3,history_scales=[.5]*3+[2.]*3,
        history='Causal ideal high-rate virtual-IMU force rotation, gravity removal, low-pass 2 ms and double integration; zero initial p/v',
        limitations=['Ideal orientation at physics rate is optimistic, not hardware-equivalent IMU',
                     'PID screening failed; this is learned-attitude-prior residual control, not PID-inner success',
                     'No absolute-position observability or drift-free inertial navigation claim'],
        physics_dt=5e-6,control_dt=.01,validation_cases=validation,test_nominal=nominal,test_stress=stress,
        selection='Validation successes, then mean discounted return, then mean elapsed',
        success='8 s without termination, position RMSE <0.1 m, tilt RMS <10 deg',
        test_policy='Freeze all policies before any final cases; zero-history ablation on identical cases; first case videos',
        budget='NEAT complete generations until >=262144 transitions, PPO rounds matched count to complete 1024-transition rollouts',
        gamma=.995,neat_activations=['tanh','bipolar_sigmoid','softsign','sin','gauss','clamped']))


if __name__=='__main__':main()
