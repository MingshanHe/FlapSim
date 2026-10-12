"""Short validation of spike-timed steering inside the original v3 interface.

Question: can NEAT evolve a spiking circuit that, like an insect steering muscle, fires phase-locked motor
pulses only when needed? How close does it get to the classical IMU pulse law?

The controller runs every 2 ms tick:
  * gimbal and height: the retuned PID (kp x1.5), as in the classical baseline;
  * motor pulses: one spiking "motor neuron". Each of its spikes starts a 6 ms twitch, during which the
    speed command sits at the upper bound. The command then goes through the unchanged clip, smoothing
    and motor path.
Inputs, IMU only (12):
  * 8 phase cells: the PLL wingbeat phase, one-hot in 45 deg bins;
  * rectified lagged AHRS pitch (+, -);
  * rectified notched pitch rate (+, -).
The phase reference still comes from the classical PLL in this pilot. Which phase to fire at, and when, is
left entirely to the evolved network.

    python -m experiments.algorithms.spike_steer.pilot train --output runs/spike_steer/pilot_s1 --seed 1
    python -m experiments.algorithms.spike_steer.pilot test --policy runs/spike_steer/pilot_s1/best_policy.json \
        --output artifacts/spike_steer/pilot_test.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math
from pathlib import Path
import random
import time
import neat
import numpy as np
from ..snn_pid_compare.common import DT, prepare_source
from ..snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from ..robust_control.environment import ResearchEnv
from ..topology_hover.environment import case as make_case
from ..wingbeat_feasibility.preliminary import WingbeatPhase
from ..wingbeat_snn.plant import PHASE_OFFSET
from ..wingbeat_snn.networks import Graph, MAX_DELAY
from ..wingbeat_snn.evolve import DelayGenome, plain_tanh
from ..neuro_gates import gates
from ..neuro_gates.timing import PULSE_PHASE

PID_GAINS = (1.5, 1.)
TWITCH_TICKS = 3
N_PHASE = 8
INPUTS, HIDDEN = N_PHASE+4, 6
OUTPUTS = 1  # motor neurons; any of their spikes starts a twitch (set by --outputs)
PITCH_SCALE, RATE_SCALE = .1, 1.  # rad, rad/s: an input of 1 means a large error
SPLITS = dict(train=2_030_100_000, validation=2_031_100_000, test=gates.SPLITS['evaluate']+100_000)
HEIGHT_WEIGHT = .3  # objective J = pitch RMS (deg) + 0.3 x height RMSE (cm); lower is better
SPEC = dict(tau_m=.005, tau_s=.0025, spike_scale=5.)
CLASSICAL = dict(law_strong=dict(k_theta=5., k_q=.4, threshold=.5, width=math.radians(60.)),
                 law_moderate=dict(k_theta=5., k_q=.4, threshold=.8, width=math.radians(90.)))


class SpikingGraph(Graph):
    """The wingbeat_snn Graph with spiking output units: an output spike is a motor event."""

    def snn_substep(self, x):
        self.pointer = (self.pointer+1) % self.length
        self.inputs[self.pointer] = x
        p = self.pointer
        src, _, _, d = self.inp
        values_in = self.inputs[(p-d) % self.length, src]
        src, _, _, d = self.rec
        values_rec = self.spike_scale*self.spike_history[(p-1-d) % self.length, src]
        drive = self._drive(values_in, values_rec)
        self.current = self.a_s*self.current+(1-self.a_s)*drive
        self.voltage = self.a_m*self.voltage+(1-self.a_m)*self.current
        fired = self.voltage >= 1.
        self.voltage[fired] -= 1.
        self.spike_history[p] = fired
        self.spikes += int(fired.sum())
        return fired[self.outputs]


class Steer:
    """Motor-pulse decision for one tick; `spec` is an evolved network, `law` the classical pulse law."""

    def __init__(self, spec=None, law=None):
        self.spec, self.law = spec, law

    def reset(self):
        self.net = SpikingGraph(self.spec) if self.spec else None
        self.estimator = WingbeatPhase(PHASE_OFFSET)
        self.twitch, self.active = 0, False

    def features(self, reading, phase):
        x = np.zeros(INPUTS)
        width = 2*math.pi/N_PHASE
        x[int(((phase % (2*math.pi))+width/2)//width) % N_PHASE] = 1.
        th, q = reading[1]/PITCH_SCALE, self.estimator.slow/RATE_SCALE
        x[N_PHASE:] = (max(th, 0.), max(-th, 0.), max(q, 0.), max(-q, 0.))
        return x

    def step(self, env, now, t):
        """Returns (pulse this tick, a new twitch started this tick)."""
        reading = env.last_reading
        self.estimator.update(now, float(reading[4]))
        phase = self.estimator.crank(now)
        fire = False
        if self.net is not None:  # the network runs from takeoff so that it settles; it acts after the handover
            x = self.features(reading, phase)
            self.fired = self.net.snn_substep(x) | self.net.snn_substep(x)
            fire = bool(self.fired.any())
        if t < gates.HANDOVER_T:
            return False, False
        if self.law is not None:  # classical: motor up for the whole phase window while |u| is above threshold
            u = -(self.law['k_theta']*reading[1]+self.law['k_q']*self.estimator.slow)
            psi = PULSE_PHASE if u > 0 else PULSE_PHASE+math.pi
            fire = abs(u) > self.law['threshold'] and abs(math.remainder(phase-psi, 2*math.pi)) < self.law['width']/2
            onset, self.active = fire and not self.active, fire
            return fire, onset
        if self.twitch > 0:
            self.twitch -= 1
            return True, False
        if fire:
            self.twitch = TWITCH_TICKS-1
            return True, True
        return False, False


def cases(split, count, seconds, offset=0):
    return [make_case(SPLITS[split]+offset+i, seconds, 'stress', i % 8) for i in range(count)]


def pulse_sign(seed):
    """The disturbance-pulse signs HoverEnv.reset draws for a stress case (same RNG calls, same order)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 981]))
    rng.uniform(-1, 1, 3)  # acceleration bias
    rng.uniform(-1, 1, 3)  # attitude bias
    rng.uniform(-.015, .015)  # motor gain
    return rng.choice([-1., 1.], 3)


def balanced_cases(split, count, seconds, offset=0):
    """Stress cases cycling through the four (x, y) pulse-sign combinations, so that every batch needs
    steering both ways."""
    wanted = [(1., 1.), (1., -1.), (-1., 1.), (-1., -1.)]
    out, seed = [], SPLITS[split]+offset
    while len(out) < count:
        if tuple(pulse_sign(seed)[:2]) == wanted[len(out) % 4]:
            out.append(make_case(seed, seconds, 'stress', len(out) % 8))
        seed += 1
    return out


def run(env, c, steer):
    env.configure(c)
    env.reset(seed=c['seed'])
    p, (sp, sd) = env.p, PID_GAINS
    env.teacher.p = dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']])
    steer.reset()
    d = env.sim.data
    crank = int(env.sim.model.joint('carrier_2_joint').qposadr[0])
    oracle = gates.Oracle(env, 'derocked')  # metrics and spike-phase analysis only; never seen by the controller
    pitch, pulses, onsets = [], 0, []
    while True:
        t = env.air.steps*DT
        cmd = env.teacher_command()
        pulse, onset = steer.step(env, float(d.time), t)
        if pulse and t >= gates.HANDOVER_T:
            cmd[2] = HIGH[2]
            pulses += 1
            if onset:
                u = -(5.*oracle.queue[-1][1]+.4*oracle.rate[1])  # true need for +pitch acceleration
                which = int(np.flatnonzero(steer.fired)[0]) if steer.net is not None else -1
                onsets.append([round(math.degrees(math.remainder(float(d.qpos[crank]), 2*math.pi)) % 360, 1),
                               int(np.sign(u)), which])
        _, _, term, trunc, out = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        oracle.update()
        pitch.append((env.air.steps*DT, math.degrees(oracle.queue[-1][1])))
        if term or trunc:
            break
    m = out['episode']
    pt = np.array(pitch)
    late = pt[:, 0] >= 2.
    flown = m['termination'] == 'time_limit'
    pitch_rms = float(np.sqrt(np.mean(pt[late, 1]**2))) if late.any() else float('nan')
    J = pitch_rms+HEIGHT_WEIGHT*100*m['height_rmse_m'] if flown else 40.-m['elapsed']
    return dict(seed=c['seed'], termination=m['termination'], elapsed=m['elapsed'], objective=J,
                pitch_rms_deg=pitch_rms, pitch_peak_deg=float(np.abs(pt[late, 1]).max()) if late.any() else float('nan'),
                tilt_rms_deg=m['tilt_rms_deg'], height_rmse_m=m['height_rmse_m'], pulse_duty=pulses/max(1, env.air.steps),
                onsets=onsets)


_ENV = None


def worker_init(source):
    global _ENV
    _ENV = ResearchEnv(str(source), mode='direct', task='hover')


def task(args):
    name, c, spec, law = args
    return dict(run(_ENV, c, Steer(spec, law)), controller=name)


def pool(workers):
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(source),))


def make_config(out, population, compatibility=3.):
    text = (Path(__file__).with_name('neat_template.ini').read_text()
            .replace('{population}', str(population)).replace('{inputs}', str(INPUTS))
            .replace('{outputs}', str(OUTPUTS)).replace('{compatibility}', str(compatibility)))
    path = out/'neat.ini'
    path.write_text(text)
    cfg = neat.Config(DelayGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))
    cfg.genome_config.activation_defs.add('plain_tanh', plain_tanh)
    return cfg


def initialize(pop, cfg, seed):
    """Shared start: 6 hidden LIF units read every input; genome 0 has a zero readout (= PID only)."""
    rng = np.random.default_rng(seed)
    gc = cfg.genome_config
    hidden = list(range(OUTPUTS, OUTPUTS+HIDDEN))
    gc.node_indexer = itertools.count(OUTPUTS+HIDDEN)
    for index, g in enumerate(pop.population.values()):
        g.nodes.clear()
        g.connections.clear()
        for k in list(gc.output_keys)+hidden:
            node = g.create_node(gc, k)
            node.bias = 0. if k in gc.output_keys else float(rng.normal(.6, .2))
            g.nodes[k] = node
        for source, target in [(i, h) for i in gc.input_keys for h in hidden]+[(h, o) for h in hidden
                                                                             for o in gc.output_keys]:
            edge = g.create_connection(gc, source, target,
                                       gc.innovation_tracker.get_innovation_number(source, target, 'shared_initial'))
            readout = target in gc.output_keys
            edge.weight = 0. if (readout and index == 0) else float(rng.normal(0., 1. if readout else .6))
            edge.delay = float(rng.integers(0, 9))
            g.connections[(source, target)] = edge
    pop.species.speciate(cfg, pop.population, 0)


def export(g):
    return dict(kind='snn', inputs=INPUTS, outputs=list(range(OUTPUTS)), genome_id=g.key, **SPEC,
                nodes=[dict(id=k, bias=float(n.bias)) for k, n in sorted(g.nodes.items())],
                edges=[dict(source=a, target=b, weight=float(e.weight), delay=float(round(e.delay)),
                            enabled=bool(e.enabled)) for (a, b), e in sorted(g.connections.items())])


def summarize(rows):
    flown = [r for r in rows if r['termination'] == 'time_limit']
    f = lambda k: float(np.mean([r[k] for r in flown])) if flown else None
    return dict(episodes=len(rows), flown=len(flown), objective=float(np.mean([r['objective'] for r in rows])),
                pitch_rms_deg=f('pitch_rms_deg'), pitch_peak_deg=f('pitch_peak_deg'), tilt_rms_deg=f('tilt_rms_deg'),
                height_rmse_cm=None if not flown else 100*f('height_rmse_m'), pulse_duty=f('pulse_duty'))


def train(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    cfg = make_config(out, a.population, a.compatibility)
    pop = neat.Population(cfg)
    initialize(pop, cfg, a.seed)
    pick = balanced_cases if a.balanced else cases
    validation = pick('validation', a.validation_cases, a.seconds)
    best, history, start = None, [], time.monotonic()
    with pool(a.workers) as executor:
        def score(genomes, config):
            nonlocal best
            generation = len(history)
            offset = a.seed*100_000+generation*1_000 if a.balanced else a.seed*10_000+generation*a.cases  # v1: latter
            batch = pick('train', a.cases, a.seconds, offset=offset)
            specs = {gid: export(g) for gid, g in genomes}
            rows = list(executor.map(task, [(str(gid), c, specs[gid], None) for gid, _ in genomes for c in batch],
                                     chunksize=1))
            for k, (gid, g) in enumerate(genomes):
                g.fitness = -float(np.mean([r['objective'] for r in rows[k*a.cases:(k+1)*a.cases]]))
            champion_id, champion = max(genomes, key=lambda x: x[1].fitness)
            row = dict(generation=generation, fitness_max=champion.fitness,
                       fitness_mean=float(np.mean([g.fitness for _, g in genomes])), species=len(pop.species.species),
                       wall_s=round(time.monotonic()-start, 1),
                       champion_size=[len(specs[champion_id]['nodes']),
                                      sum(e['enabled'] for e in specs[champion_id]['edges'])])
            if generation % a.validate_every == a.validate_every-1 or generation == 0:
                vrows = list(executor.map(task, [('champion', c, specs[champion_id], None) for c in validation],
                                          chunksize=1))
                row['validation'] = summarize(vrows)
                if best is None or row['validation']['objective'] < best['objective']:
                    best = dict(objective=row['validation']['objective'], generation=generation)
                    (out/'best_policy.json').write_text(json.dumps(specs[champion_id], indent=1))
            history.append(row)
            with (out/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        pop.run(score, a.generations)
    result = dict(seed=a.seed, population=a.population, generations=a.generations, cases=a.cases, seconds=a.seconds,
                  outputs=OUTPUTS, balanced=a.balanced, compatibility=a.compatibility,
                  selected_generation=best['generation'], selected_validation_objective=best['objective'],
                  wall_s=time.monotonic()-start)
    (out/'results.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


def tune_law(a):
    """Classical pulse law tuned on the same kind of training cases as the network (grid, one budget)."""
    grid = [dict(k_theta=kt, k_q=kq, threshold=u0, width=math.radians(w))
            for kt in (3., 5., 8.) for kq in (.2, .4, .8) for u0 in (.4, .5, .65, .8) for w in (45., 60., 90.)]
    train_cases = balanced_cases('train', a.count, a.seconds, offset=900_000)
    with pool(a.workers) as executor:
        rows = list(executor.map(task, [(str(i), c, None, law) for i, law in enumerate(grid) for c in train_cases],
                                 chunksize=1))
    score = {i: float(np.mean([r['objective'] for r in rows if r['controller'] == str(i)])) for i in range(len(grid))}
    best = min(score, key=score.get)
    result = dict(grid=grid, objective=score, best=grid[best], best_objective=score[best], cases=len(train_cases),
                  episodes=len(rows))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(dict(best=grid[best], best_objective=score[best]), indent=1))


def test(a):
    spec = json.loads(Path(a.policy).read_text())
    laws = dict(CLASSICAL)
    if a.law:
        laws['law_tuned'] = json.loads(Path(a.law).read_text())['best']
    controllers = [('pid_retuned', None, None)]+[(k, None, v) for k, v in laws.items()]+[('snn_neat', spec, None)]
    test_cases = cases('test', a.count, 8.)
    with pool(a.workers) as executor:
        rows = list(executor.map(task, [(n, c, s, l) for n, s, l in controllers for c in test_cases], chunksize=1))
    summary = {n: summarize([r for r in rows if r['controller'] == n]) for n, _, _ in controllers}
    onsets = [o for r in rows if r['controller'] == 'snn_neat' for o in r['onsets']]
    phases = {}
    for need in (1, -1):  # where in the wingbeat does a motor neuron fire, by the true sign of the needed push?
        for which in sorted({o[2] for o in onsets}):
            hist = np.histogram([o[0] for o in onsets if o[1] == need and o[2] == which], bins=8, range=(0, 360))[0]
            phases[f"{'need_up' if need > 0 else 'need_down'}|output{which}"] = hist.tolist()
    result = dict(test_split=SPLITS['test'], policy=str(a.policy), summary=summary, spike_phase_hist_45deg=phases,
                  reference_phase_deg=dict(need_up=math.degrees(PULSE_PHASE), need_down=math.degrees(PULSE_PHASE)+180),
                  rows=rows)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(dict(summary=summary, spike_phase_hist_45deg=phases), indent=1))


def main():
    global OUTPUTS
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train', 'test', 'tune_law'])
    p.add_argument('--outputs', type=int, default=1, help='motor neurons (train)')
    p.add_argument('--balanced', action='store_true', help='every batch covers all four pulse-sign combinations')
    p.add_argument('--compatibility', type=float, default=3., help='NEAT speciation threshold')
    p.add_argument('--law', help='tune_law output whose best law joins the test')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--cases', type=int, default=3)
    p.add_argument('--seconds', type=float, default=6.)
    p.add_argument('--validation-cases', type=int, default=8)
    p.add_argument('--validate-every', type=int, default=4)
    p.add_argument('--policy')
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--workers', type=int, default=18)
    a = p.parse_args()
    OUTPUTS = a.outputs
    dict(train=train, test=test, tune_law=tune_law)[a.stage](a)


if __name__ == '__main__':
    main()
