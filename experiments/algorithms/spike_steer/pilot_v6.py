"""B' pilot: spike-timed steering evolved from RAW IMU signals (no PLL, no phase cells).

Plan and pre-registered criteria: docs/original_env_snn_neat/PLAN_B_emergence.md. The network sees only the
rectified raw IMU channels (AHRS roll/pitch, gyro x/y/z, accelerometer x/y/z) plus a constant; the wingbeat
oscillation in the gyro and accelerometer (10.7 rad/s, 24 m/s^2) is the only phase information. It must build its
own phase reference (a threshold-crossing unit plus synaptic delays) to time the motor pulses.

Reuse: TauGenome and gain-preserving tau (pilot_v4), canonical cold start (pilot_v5), cases, selection,
calibration and statistics (pilot_v3), the episode loop and the classical law (pilot).

Arms (--arm): main | nodelay (delays frozen at 0) | fixed (dense 6-hidden topology, weights and tau only) |
ahrs (gyro and accelerometer inputs zeroed) | rate (non-spiking tanh hidden units, same delays and inputs).

    python -m experiments.algorithms.spike_steer.pilot_v6 selftest
    python -m experiments.algorithms.spike_steer.pilot_v6 all --quick --output runs/spike_steer/pilot_v6_quick
    python -m experiments.algorithms.spike_steer.pilot_v6 all --arm main --seed 1 --output runs/spike_steer/pilot_v6_main_s1
    python -m experiments.algorithms.spike_steer.pilot_v6 analyze --spec artifacts/spike_steer/v6/main_s1/selected_calibrated.json \
        --output artifacts/spike_steer/v6/main_s1/mechanism.json
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import re
import tempfile
import time
import neat
import numpy as np
from . import pilot as P
from . import pilot_v3 as V
from . import pilot_v4 as W
from . import pilot_v5 as X
from ..wingbeat_snn.evolve import plain_tanh

V3 = W.V3
CHANNELS = ('roll', 'pitch', 'gx', 'gy', 'gz', 'ax', 'ay', 'az')
IDX = (0, 1, 3, 4, 5, 6, 7, 8)                      # indices in the 11-D reading
SCALE = np.array([.1, .1, 10., 10., 10., 25., 25., 25.])  # unit ~ 'large': 0.1 rad, wingbeat gyro and accel amplitude
INPUTS = 2*len(CHANNELS)+2                         # rectified pairs (+ then -) and constants 1, 0
OUTPUTS = 2
MIRROR = [i+len(CHANNELS) for i in range(len(CHANNELS))]+[i for i in range(len(CHANNELS))]+[2*len(CHANNELS), 2*len(CHANNELS)+1]
MASKS = dict(all=np.ones(len(CHANNELS)), ahrs=np.array([1., 1., 0., 0., 0., 0., 0., 0.]))
ARMS = dict(main=dict(), nodelay=dict(), fixed=dict(), ahrs=dict(mask='ahrs'), rate=dict(rate_hidden=True))
FIXED_HIDDEN = 6
OVERRIDES = dict(  # on top of pilot_v5.COLD; per arm
    main={'node_add_prob': '0.1'},
    nodelay={'node_add_prob': '0.1', 'delay_mutate_rate': '0.0', 'delay_replace_rate': '0.0'},
    fixed={'node_add_prob': '0.0', 'node_delete_prob': '0.0', 'conn_add_prob': '0.0', 'conn_delete_prob': '0.0',
           'enabled_mutate_rate': '0.0', 'initial_connection': 'unconnected'},
    ahrs={'node_add_prob': '0.1'},
    rate={'node_add_prob': '0.1'})


# ----------------------------------------------------------------------------------------- network
class RateHiddenGraph(P.SpikingGraph):
    """Ablation: hidden units are non-spiking (transmit tanh(voltage) every substep, same delays); outputs spike."""

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
        fired = (self.voltage >= 1.) & ~self.hidden
        self.voltage[fired] -= 1.
        self.spike_history[p] = np.where(self.hidden, np.tanh(self.voltage), fired.astype(float))
        self.spikes += int(fired.sum())
        return fired[self.outputs]


class SteerV6(V.SteerV3):
    """pilot_v3.SteerV3 with raw-IMU features. Calibration knobs: g_theta scales the angle inputs, g_q the gyro
    inputs, d_out the output bias, twitch the pulse length. The PLL still runs, but only the classical law reads it."""

    def __init__(self, spec=None, law=None, lockout=False, mask='all', rate_hidden=False):
        super().__init__(spec, law, lockout)
        self.mask, self.rate_hidden = MASKS[mask], rate_hidden

    def reset(self):
        cls = RateHiddenGraph if self.rate_hidden else P.SpikingGraph
        self.net = cls(self.spec) if self.spec else None
        self.estimator = P.WingbeatPhase(P.PHASE_OFFSET)
        self.twitch, self.active = 0, False
        if self.net is not None and self.calib['d_out']:
            self.net.bias[self.net.outputs] += self.calib['d_out']
        self.block, self.last_pool = 0, None

    def features(self, reading, phase):
        x = np.asarray(reading, dtype=float)[list(IDX)]/SCALE*self.mask
        x[:2] *= self.calib['g_theta']
        x[2:5] *= self.calib['g_q']
        return np.r_[np.maximum(x, 0.), np.maximum(-x, 0.), 1., 0.]


def task(args):
    """Worker entry (pilot_v3.run_all is pointed at this): the arm settings travel inside the spec."""
    name, c, spec, law, lockout = args
    arm = dict((spec or {}).get('arm', {}))
    return dict(P.run(P._ENV, c, SteerV6(spec, law, lockout, **arm)), controller=name)


# ----------------------------------------------------------------------------------------- genomes
def make_config(out, population, arm, outputs=OUTPUTS):
    text = (Path(V.__file__).with_name('neat_template_v3.ini').read_text()
            .replace('[DelayGenome]', '[TauGenome]')
            .replace('{population}', str(population)).replace('{inputs}', str(INPUTS))
            .replace('{outputs}', str(outputs)))
    for key, value in {**X.COLD, **OVERRIDES[arm]}.items():
        text, n = re.subn(rf'^{key}\s*=.*$', f'{key} = {value}', text, flags=re.M)
        assert n == 1, key
    path = Path(out)/'neat.ini'
    path.write_text(text)
    cfg = neat.Config(W.TauGenome, neat.DefaultReproduction, V.AdaptiveSpeciesSet, neat.DefaultStagnation, str(path))
    cfg.genome_config.activation_defs.add('plain_tanh', plain_tanh)
    return cfg


def export(g, arm, tied=False):
    spec = dict(kind='snn', inputs=INPUTS, genome_id=g.key, tau_m=float(g.tau_m), tau_s=float(g.tau_s),
                spike_scale=P.SPEC['spike_scale'], outputs=list(range(1 if tied else OUTPUTS)), arm=dict(ARMS[arm]), tied=False,
                nodes=[dict(id=k, bias=float(n.bias)) for k, n in sorted(g.nodes.items())],
                edges=[dict(source=a, target=b, weight=float(e.weight), delay=0. if arm == 'nodelay' else float(round(e.delay)),
                            enabled=bool(e.enabled)) for (a, b), e in sorted(g.connections.items())])
    return mirror_spec(spec) if tied else spec


def mirror_spec(spec):
    """Antagonist pool as the mirror image of the evolved pool: every +/- input pair swapped (constants kept), nodes
    and edges copied with pilot_v3.MIRROR_OFFSET. On raw IMU inputs the swap moves a gyro threshold detector by half
    a wingbeat, which is exactly the 180 deg the opposite pulse direction needs."""
    off = V.MIRROR_OFFSET
    def source(sid):
        return -(MIRROR[-sid-1]+1) if sid < 0 else sid+off
    out = dict(spec)
    out['nodes'] = list(spec['nodes'])+[dict(n, id=n['id']+off) for n in spec['nodes']]
    out['edges'] = list(spec['edges'])+[dict(e, source=source(e['source']), target=e['target']+off) for e in spec['edges']]
    out['outputs'] = list(spec['outputs'])+[o+off for o in spec['outputs']]
    out['tied'] = True
    return out


def silent_spec(arm):
    return dict(kind='snn', inputs=INPUTS, outputs=[0, 1], tau_m=.001, tau_s=.0005, spike_scale=P.SPEC['spike_scale'],
                arm=dict(ARMS[arm]), tied=False, nodes=[dict(id=0, bias=-3.), dict(id=1, bias=-3.)], edges=[])


def dense_population(pop, cfg):
    """Fixed-topology arm: 6 hidden LIF units read every input; outputs read every hidden unit; outputs start silent."""
    gc = cfg.genome_config
    n_out = len(gc.output_keys)
    hidden = list(range(n_out, n_out+FIXED_HIDDEN))
    gc.node_indexer = itertools.count(n_out+FIXED_HIDDEN)
    for g in pop.population.values():
        g.nodes.clear()
        g.connections.clear()
        for k in list(gc.output_keys)+hidden:
            node = g.create_node(gc, k)
            node.bias = -2. if k in gc.output_keys else float(np.clip(random.gauss(-.5, .5), -3., 3.))
            g.nodes[k] = node
        for s, t in [(i, h) for i in gc.input_keys for h in hidden]+[(h, o) for h in hidden for o in gc.output_keys]:
            edge = g.create_connection(gc, s, t, gc.innovation_tracker.get_innovation_number(s, t, 'shared_initial'))
            edge.weight, edge.delay, edge.enabled = float(np.clip(random.gauss(0., .5), -5., 5.)), 0., True
            g.connections[(s, t)] = edge
        tau_m = g.clip(math.exp(random.uniform(*map(math.log, W.TAU_WARM))))
        g.tau_m, g.tau_s = tau_m, g.clip(tau_m*random.uniform(.3, .7))
    pop.species.speciate(cfg, pop.population, 0)


# ----------------------------------------------------------------------------------------- stages
def stage_train(executor, out, a):
    arm = a.arm
    random.seed(a.seed)
    tied, balanced = bool(a.tied), a.fitness == 'balanced'
    X.N_OUT = W.OUTPUT_KEYS = 1 if tied else OUTPUTS
    cfg = make_config(out, a.population, arm, X.N_OUT)
    pop = neat.Population(cfg)
    if arm == 'fixed':
        dense_population(pop, cfg)
    else:
        X.strip_output_loops(pop)
        pop.species.speciate(cfg, pop.population, 0)
    validation = V.stratified(V.BASES['validation'], V.n(8), 6.)
    seen, validated, history, start = {}, [], [], time.monotonic()

    def score(genomes, config):
        generation = len(history)
        batch = V.stratified(V.BASES['train']+a.seed*100_000+generation*1_000, a.cases, 6.)
        specs = {gid: export(g, arm, tied) for gid, g in genomes}
        rows = V.run_all(executor, [(str(gid), c, specs[gid], None, False) for gid, _ in genomes for c in batch])
        for k, (gid, g) in enumerate(genomes):
            js = [r['objective'] for r in rows[k*a.cases:(k+1)*a.cases]]
            g.fitness = -float(.5*np.mean(js)+.5*np.max(js)) if balanced else -float(np.mean(js))
            seen.setdefault(gid, []).extend(js)
        best = max(genomes, key=lambda x: x[1].fitness)[1]
        duty = float(np.mean([r['pulse_duty'] for r in rows]))
        row = dict(generation=generation, fitness_max=best.fitness,
                   fitness_mean=float(np.mean([g.fitness for _, g in genomes])), species=len(pop.species.species),
                   threshold=cfg.species_set_config.compatibility_threshold, wall_s=round(time.monotonic()-start, 1),
                   tau_ms=W.tau_summary(genomes), best_structure=X.structure(best), mean_duty=round(duty, 4),
                   hidden_mean=float(np.mean([X.structure(g)['hidden'] for _, g in genomes])))
        if generation % a.validate_every == a.validate_every-1:
            ranked = sorted(genomes, key=lambda x: (len(seen[x[0]]) < 8, float(np.mean(seen[x[0]]))
                                                    if len(seen[x[0]]) >= 8 else -x[1].fitness))
            top = [gid for gid, _ in ranked[:a.validate_top]]
            vrows = V.run_all(executor, [(str(gid), c, specs[gid], None, False) for gid in top for c in validation])
            for gid in top:
                items = [r for r in vrows if r['controller'] == str(gid)]
                s = V.stats(items)
                validated.append(dict(generation=generation, genome=gid, flown=s['flown'], J=s['J'], spec=specs[gid],
                                      structure=X.structure(dict(genomes)[gid]), directions=V.directions(items)))
            row['validation'] = [dict(genome=v['genome'], flown=v['flown'], J=round(v['J'], 3),
                                      tau_ms=[round(1e3*v['spec']['tau_m'], 3), round(1e3*v['spec']['tau_s'], 3)],
                                      dirs=v['directions']['pass'], **v['structure'])
                                 for v in validated if v['generation'] == generation]
        k = len(pop.species.species)
        factor = 1.5 if k > 10 else 1.3 if k > 6 else 1.15 if k > 4 else (.87 if k < 3 else 1.)
        cfg.species_set_config.compatibility_threshold = float(np.clip(
            cfg.species_set_config.compatibility_threshold*factor, .02, 4.))
        history.append(row)
        with (out/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    pop.run(score, a.generations)
    unique = {}
    for v in sorted(validated, key=lambda v: (-v['flown'], v['J'])):
        unique.setdefault(v['genome'], v)
    best = list(unique.values())[:3]
    (out/'validated.json').write_text(json.dumps([dict(v, spec=None) for v in validated], indent=1))
    return dict(candidates=[v['spec'] for v in best], source=silent_spec(arm), tied=tied)


def concentration(rows, need):
    phases = np.radians([o[0] for r in rows for o in r['onsets'] if o[1] == need])
    return float(abs(np.mean(np.exp(1j*phases)))) if len(phases) else 0.


def stage_test(executor, out, calibrated, raw, law_ext, arm):
    plan = Path(__file__).resolve().parents[3]/'docs/original_env_snn_neat/PLAN_B_emergence.md'
    rule = dict(only=f'snn_v6 arm={arm} (calibrated, validation-selected)', plan=str(plan),
                H1=['flies 16/16 on the old and the fresh 16-case tests', 'J <= 1.05 x J_law_tuned on both'],
                H2a=['>= 20 onsets per direction, >= 70 % within +-67.5 deg of 135 / 315 deg', 'concentration R >= 0.7 per direction'],
                H2b_H2c='mechanism probes: `analyze` command (hidden-unit phase locking, input clamp, delay zeroing)',
                registered_before_test=True, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out/'preregistration.json').write_text(json.dumps(rule, indent=1))
    law_tuned = json.loads(V.V2_LAW.read_text())['best']
    arms = [('snn_v6', calibrated, None, False), ('snn_v6_raw', raw, None, False),
            ('law_tuned', None, law_tuned, False), ('law_ext', None, law_ext, False), ('pid_retuned', None, None, False)]
    sets = dict(old16=P.cases('test', 16, 8.)[:V.n(16, 2)], fresh16=V.consecutive(V.BASES['test3'], V.n(16, 2), 8.))
    report, all_rows = {}, {}
    for label, cases in sets.items():
        rows = V.run_all(executor, [(n, c, s, l, x) for n, s, l, x in arms for c in cases])
        all_rows[label] = rows
        by = {n: [r for r in rows if r['controller'] == n] for n, *_ in arms}
        table = {n: V.stats(v) for n, v in by.items()}
        snn, law = table['snn_v6'], table['law_tuned']
        dirs = {n: dict(V.directions(by[n]), R_up=concentration(by[n], 1), R_down=concentration(by[n], -1))
                for n in ('snn_v6', 'snn_v6_raw')}
        d = dirs['snn_v6']
        verdict = dict(flown=snn['flown'] == len(cases), J_ratio=snn['J']/law['J'], directions=d)
        verdict['H1'] = bool(verdict['flown'] and verdict['J_ratio'] <= 1.05)
        verdict['H2a'] = bool(d['pass'] and d['R_up'] >= .7 and d['R_down'] >= .7)
        report[label] = dict(table=table, directions=dirs,
                             paired_vs_law_tuned={n: V.paired(by[n], by['law_tuned']) for n in by if n != 'law_tuned'},
                             strata={n: V.strata(by[n]) for n in ('snn_v6', 'law_tuned')}, verdict=verdict)
        print(label, json.dumps({n: dict(flown=t['flown'], J=round(t['J'], 3), pitch=round(t['pitch_rms_deg'] or 0, 2),
                                         height=round(t['height_rmse_cm'] or 0, 1)) for n, t in table.items()}, indent=1),
              json.dumps(verdict), flush=True)
    hidden = [n['id'] for n in calibrated['nodes'] if n['id'] not in calibrated['outputs']]
    report['overall'] = dict(arm=arm, tied=bool(calibrated.get('tied', False)), fitness=getattr(stage_test, 'fitness', 'mean'), H1=report['old16']['verdict']['H1'] and report['fresh16']['verdict']['H1'],
                             H2a=report['old16']['verdict']['H2a'] and report['fresh16']['verdict']['H2a'],
                             tau_ms=[1e3*calibrated['tau_m'], 1e3*calibrated['tau_s']], hidden=len(hidden),
                             edges=sum(e['enabled'] for e in calibrated['edges']), calib=calibrated.get('calib'))
    (out/'test.json').write_text(json.dumps(dict(report=report, old_rows=all_rows['old16'], fresh_rows=all_rows['fresh16']),
                                            indent=1))
    print(json.dumps(report['overall'], indent=1), flush=True)
    return report


# ----------------------------------------------------------------------------------------- mechanism
class TracedSteer(SteerV6):
    """Logs, per tick: time, true crank phase (deg), hidden units that spiked, output onset."""

    def reset(self):
        super().reset()
        self.log = []

    def step(self, env, now, t):
        pulse, onset = super().step(env, now, t)
        if self.net is not None:
            n = self.net
            p = n.pointer
            fired = (n.spike_history[p] > .5) | (n.spike_history[(p-1) % n.length] > .5)
            crank = int(env.sim.model.joint('carrier_2_joint').qposadr[0])
            phi = math.degrees(float(env.sim.data.qpos[crank])) % 360.
            self.log.append((t, phi, [int(n.ids[i]) for i in np.flatnonzero(fired & n.hidden)], int(onset)))
        return pulse, onset


def traced_task(args):
    name, c, spec, variant = args
    arm = dict(spec.get('arm', {}))
    spec = json.loads(json.dumps(spec))
    if variant == 'nodelay_out':  # zero the delays of edges into the outputs (weights unchanged)
        for e in spec['edges']:
            if e['target'] < OUTPUTS:
                e['delay'] = 0.
    elif variant == 'nodelay_all':
        for e in spec['edges']:
            e['delay'] = 0.
    elif variant == 'clamp_fast':  # gyro and accelerometer inputs zeroed
        arm['mask'] = 'ahrs'
    elif variant.startswith('lesion_'):  # remove one hidden unit
        hid = int(variant.split('_')[1])
        spec['nodes'] = [n for n in spec['nodes'] if n['id'] != hid]
        spec['edges'] = [e for e in spec['edges'] if hid not in (e['source'], e['target'])]
    steer = TracedSteer(spec, None, False, **arm)
    res = P.run(P._ENV, c, steer)
    return dict(res, controller=name, variant=variant, log=steer.log)


def analyze(a):
    spec = json.loads(Path(a.spec).read_text())
    hidden = [n['id'] for n in spec['nodes'] if n['id'] >= OUTPUTS]
    cases = V.stratified(V.BASES['select'], V.n(8), 6.)
    variants = ['intact', 'nodelay_out', 'nodelay_all', 'clamp_fast']+[f'lesion_{h}' for h in hidden]
    with P.pool(a.workers) as ex:
        rows = list(ex.map(traced_task, [(v, c, spec, v) for v in variants for c in cases], chunksize=1))
    out = {}
    for v in variants:
        items = [r for r in rows if r['variant'] == v]
        late = [(t, phi, hs, on) for r in items for (t, phi, hs, on) in r['log'] if t >= 2.]
        locking = {}
        for h in hidden:
            ph = np.radians([phi for (_, phi, hs, _) in late if h in hs])
            n_cycles = sum(1 for r in items for _ in [0])*0  # placeholder for readability
            locking[h] = dict(spikes=int(len(ph)), R=float(abs(np.mean(np.exp(1j*ph)))) if len(ph) else 0.,
                              mean_deg=float(np.degrees(np.angle(np.mean(np.exp(1j*ph)))) % 360) if len(ph) else None,
                              per_wingbeat=float(len(ph)/max(1, sum(len([1 for x in r['log'] if x[0] >= 2.]) for r in items)/20.)))
        out[v] = dict(stats=V.stats(items), directions=V.directions(items), R_up=concentration(items, 1),
                      R_down=concentration(items, -1), hidden_locking=locking)
        print(f"{v:14s} flown {out[v]['stats']['flown']} J {out[v]['stats']['J']:.2f} up/dn R {out[v]['R_up']:.2f}/{out[v]['R_down']:.2f} "
              f"dirs {out[v]['directions']['pass']}  hidden R: " +
              ', '.join(f"{h}:{locking[h]['R']:.2f}@{locking[h]['mean_deg'] or 0:.0f}({locking[h]['per_wingbeat']:.2f}/beat)" for h in hidden),
              flush=True)
    J0 = out['intact']['stats']['J']
    verdict = dict(
        H2b=any(l['R'] >= .8 and .5 <= l['per_wingbeat'] <= 2.5 for l in out['intact']['hidden_locking'].values())
        and (out['clamp_fast']['stats']['J'] >= 1.15*J0 or not out['clamp_fast']['directions']['pass']),
        H2c=out['nodelay_out']['stats']['J'] >= 1.1*J0 or out['nodelay_all']['stats']['J'] >= 1.1*J0)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(dict(spec=a.spec, variants=out, verdict=verdict,
                                              rows=[dict(r, log=None) for r in rows]), indent=1))
    print(json.dumps(verdict), flush=True)


# ----------------------------------------------------------------------------------------- selftest / main
def selftest():
    s = SteerV6(None)
    s.calib = dict(g_theta=1., g_q=1., d_out=0., twitch=3)
    r = np.zeros(11)
    r[1], r[4], r[6] = .05, -5., 25.
    x = s.features(r, 0.)
    assert x.shape == (INPUTS,) and x[1] == .5 and x[11] == .5 and x[5] == 1. and x[-2] == 1. and x[-1] == 0., x
    s.mask = MASKS['ahrs']
    assert s.features(r, 0.)[11] == 0. and s.features(r, 0.)[1] == .5
    for arm in ARMS:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp, 8, arm)
            random.seed(1)
            pop = neat.Population(cfg)
            if arm == 'fixed':
                dense_population(pop, cfg)
                st = X.structure(next(iter(pop.population.values())))
                assert st['hidden'] == FIXED_HIDDEN and st['edges'] == INPUTS*FIXED_HIDDEN+FIXED_HIDDEN*OUTPUTS, st
            else:
                X.strip_output_loops(pop)
                pop.species.speciate(cfg, pop.population, 0)
                st = X.structure(next(iter(pop.population.values())))
                assert st['hidden'] == 0 and st['direct'] == INPUTS*OUTPUTS, st
            g = next(iter(pop.population.values()))
            for _ in range(10):
                g.mutate(cfg.genome_config)
            spec = export(g, arm)
            assert spec['inputs'] == INPUTS and spec['arm'] == ARMS[arm]
            if arm == 'nodelay':
                assert all(e['delay'] == 0. for e in spec['edges'])
            steer = SteerV6(spec, **ARMS[arm])
            steer.reset()
            rng = np.random.default_rng(0)
            for _ in range(50):
                steer.net.snn_substep(rng.random(INPUTS)*.3)
            if arm == 'rate' and X.structure(g)['hidden']:
                hist = steer.net.spike_history[:, steer.net.hidden]
                assert np.all(np.abs(hist) <= 1.) and not np.array_equal(hist, np.round(hist)), 'rate hidden units should be graded'
    # mirror map: swaps the +/- halves, keeps the constants; a tied export has two pools and 2x the edges
    assert sorted(MIRROR) == list(range(INPUTS)) and all(MIRROR[MIRROR[i]] == i for i in range(INPUTS)) and MIRROR[16:] == [16, 17]
    with tempfile.TemporaryDirectory() as tmp:
        X.N_OUT = W.OUTPUT_KEYS = 1
        cfg = make_config(tmp, 8, 'main', 1)
        random.seed(3)
        pop = neat.Population(cfg)
        X.strip_output_loops(pop)
        g = next(iter(pop.population.values()))
        assert X.structure(g)['direct'] == INPUTS
        m = export(g, 'main', tied=True)
        assert m['tied'] and m['outputs'] == [0, V.MIRROR_OFFSET] and len(m['edges']) == 2*INPUTS and len(m['nodes']) == 2
        a, b = P.SpikingGraph(m), P.SpikingGraph(m)
        rng = np.random.default_rng(4)
        for _ in range(100):  # mirrored inputs into the copy == original inputs into the original pool
            x = rng.random(INPUTS)*(rng.random(INPUTS) < .3)
            x[16], x[17] = 1., 0.
            xm = np.zeros(INPUTS)
            for i in range(INPUTS):
                xm[MIRROR[i]] = x[i]
            fa, fb = a.snn_substep(x), b.snn_substep(xm)
            assert fa[0] == fb[1] and fa[1] == fb[0], 'mirror equivalence failed'
        X.N_OUT = W.OUTPUT_KEYS = OUTPUTS
    net = P.SpikingGraph(silent_spec('main'))
    assert not any(net.snn_substep(np.ones(INPUTS)).any() for _ in range(50))
    assert set(OVERRIDES) == set(ARMS)
    print('selftest ok')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'all', 'analyze'])
    p.add_argument('--arm', default='main', choices=sorted(ARMS))
    p.add_argument('--output', default=None)
    p.add_argument('--spec', default=None)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--generations', type=int, default=40)
    p.add_argument('--cases', type=int, default=4)
    p.add_argument('--validate-every', type=int, default=4)
    p.add_argument('--validate-top', type=int, default=3)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--fitness', default='mean', choices=['mean', 'balanced'], help='balanced: 0.5 mean + 0.5 worst case per batch')
    p.add_argument('--tied', action='store_true', help='mirror-tied antagonist pair: evolve one pool, mirror it')
    a = p.parse_args()
    V.QUICK = a.quick
    stage_test.fitness = a.fitness
    if a.quick:
        a.population, a.generations, a.validate_every, a.validate_top = 8, 2, 1, 2
    if a.stage == 'selftest':
        selftest()
        return
    V.task = task                       # pilot_v3.run_all (used by its select/calibrate stages) now runs SteerV6
    V.CALIB_GRID = W.CALIB_GRID
    if a.stage == 'analyze':
        analyze(a)
        return
    out = Path(a.output or f'runs/spike_steer/pilot_v6_{a.arm}_s{a.seed}')  # scripts/run_v6.sh adds the fix tag
    out.mkdir(parents=True, exist_ok=True)
    law_ext = json.loads((V3/'law_ext.json').read_text())['best']
    choice = dict(tied=a.tied, source='cold', lockout=False)
    print(json.dumps(dict(arm=a.arm, seed=a.seed, tied=a.tied, fitness=a.fitness, population=a.population, generations=a.generations, law_ext=law_ext)), flush=True)
    with P.pool(a.workers) as executor:
        trained = stage_train(executor, out, a)
        raw = V.stage_select(executor, out, trained, choice)
        calibrated = V.stage_calibrate(executor, out, raw, choice)
        stage_test(executor, out, calibrated, raw, law_ext, a.arm)


if __name__ == '__main__':
    main()
