"""Safe JSON inference artifact; preserves neat-python activation semantics."""
import json
from pathlib import Path

import neat
import numpy as np

from .environment import ANGLE_SCALE, INPUT_NAMES


def encode_network(genome, config):
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return dict(format='flapsim-neat-imu-v2', inputs=network.input_nodes,
                outputs=network.output_nodes, input_names=INPUT_NAMES[len(network.input_nodes)],
                input_scale_rad=ANGLE_SCALE, output_names=['alpha_pitch', 'beta_roll', 'motor_speed'],
                motor_trim_rad_s=getattr(config, 'motor_speed', 3400.),
                motor_max_rad_s=getattr(config, 'max_motor_speed', 5000.),
                output_limit_rad=.3, nodes=[dict(key=key, bias=bias, response=response,
                activation='tanh', aggregation='sum', links=links)
                for key, _, _, bias, response, links in network.node_evals])


def decode_network(payload):
    if payload['format'] not in ('flapsim-neat-imu-v1', 'flapsim-neat-imu-v2'):
        raise ValueError('unsupported policy format')
    activations = neat.activations.ActivationFunctionSet()
    aggregations = neat.aggregations.AggregationFunctionSet()
    return neat.nn.FeedForwardNetwork(payload['inputs'], payload['outputs'], [
        (n['key'], activations.get(n['activation']), aggregations.get(n['aggregation']),
         n['bias'], n['response'], n['links']) for n in payload['nodes']])


class ImuPolicy:
    def __init__(self, path):
        self.payload = json.loads(Path(path).read_text())
        self.network = decode_network(self.payload)

    def act(self, normalized_rpy):
        return np.asarray(self.network.activate(normalized_rpy), dtype=float)

    def angles(self, roll, pitch, yaw):
        """For three-input policies: raw IMU radians -> alpha, beta, motor speed."""
        if len(self.payload['inputs']) != 3:
            raise ValueError('this policy requires position/motion inputs; use commands(observation)')
        obs = np.clip(np.array([roll, pitch, yaw])/self.payload['input_scale_rad'], -10, 10)
        return self.commands(obs)

    def commands(self, normalized_observation):
        """Targets before smoothing: alpha/beta in rad, shaft speed in rad/s."""
        action = np.clip(self.act(normalized_observation), -1, 1)
        angles = self.payload['output_limit_rad']*action[:2]
        if len(action) == 2:
            return angles
        trim, maximum = self.payload['motor_trim_rad_s'], self.payload['motor_max_rad_s']
        span = maximum-trim if action[2] >= 0 else trim
        return np.r_[angles, trim+action[2]*span]
