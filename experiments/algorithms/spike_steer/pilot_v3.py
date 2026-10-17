"""Spike-timed steering pilot v3. This follows the fix plan of the contingency workflow (2026-10-13).

Changes against v1/v2 (pilot.py, which stays untouched):
  * mirror-tied antagonist motor pools: the genome encodes one pool, and the other is its mirror image
    (phase cells +180 deg, pitch+/pitch- and rate+/rate- swapped), so both steering directions exist by
    construction;
  * a warm start from the v1 champion, plus zero-weight direct need->motor edges;
  * a gentle mutation template (neat_template_v3.ini) and a species-count controller;
  * batches stratified by pulse signs and by motor gain (low lift headroom when gain >= 1.008);
  * selection on validation (top 3 every 4 generations, then a 16-case final selection);
  * a 4-knob calibration (input gains, motor-neuron bias, twitch length), on the same 8 cases and with the
    same 108-point budget as the classical law;
  * a pass rule registered before the test, and computed from the onset rows.

    python -m experiments.algorithms.spike_steer.pilot_v3 selftest
    python -m experiments.algorithms.spike_steer.pilot_v3 all --output runs/spike_steer/pilot_v3_s3 --seed 3
"""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import random
import time
import neat
import numpy as np
from . import pilot as P
from ..neuro_gates import gates
from ..neuro_gates.timing import PULSE_PHASE
from ..topology_hover.environment import case as make_case
from ..wingbeat_snn.evolve import DelayGenome, plain_tanh

ROOT = Path(__file__).resolve().parents[3]
MIRROR = (4, 5, 6, 7, 0, 1, 2, 3, 9, 8, 11, 10)  # phase cell +180 deg; pitch+ <-> pitch-; rate+ <-> rate-
MIRROR_OFFSET = 100_000
NEED_INPUTS = (8, 9, 10, 11)  # pitch+, pitch-, rate+, rate- (input ids -9..-12)
GAIN_SPLIT = 1.008  # motor gain at or above this leaves <= ~45 rad/s of downward speed headroom
COMBOS = [(1., 1.), (1., -1.), (-1., 1.), (-1., -1.)]
BASES = dict(train=2_050_300_000, validation=2_050_600_000, d1=2_050_700_000, select=2_050_800_000,
             test3=2_052_100_000)
LOCKOUT_TICKS = 14
V1 = ROOT/'artifacts/spike_steer/v1/best_policy.json'
V2 = ROOT/'artifacts/spike_steer/v2/best_policy.json'
V2_LAW = ROOT/'artifacts/spike_steer/v2/law_tuned.json'
V2_TEST = ROOT/'artifacts/spike_steer/v2/test.json'
CALIB_GRID = [dict(g_theta=a, g_q=b, d_out=c, twitch=d) for a in (.6, 1., 1.6) for b in (.5, 1., 2.)
              for c in (-.3, -.15, 0., .15) for d in (2, 3, 4)]
QUICK = False  # --quick: tiny end-to-end smoke run of every stage (sizes below), to catch crashes before 2 h runs


def n(full, quick=4):
    return quick if QUICK else full


# ----------------------------------------------------------------------------------------- cases
def case_draw(seed):
    """(motor gain, pulse signs) exactly as HoverEnv.reset draws them for a stress case."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 981]))
    rng.uniform(-1, 1, 3)  # acceleration bias
    rng.uniform(-1, 1, 3)  # attitude bias
    gain = 1+rng.uniform(-.015, .015)
    return float(gain), rng.choice([-1., 1.], 3)


def stratified(base, count, seconds):
    """Each block of 4 covers the four (x, y) sign combinations, with exactly one high-gain case in a slot that
    rotates with the block index (about the natural 23 % share)."""
    out, seed = [], base
    while len(out) < count:
        block, pos = divmod(len(out), 4)
        want_sign, want_high = COMBOS[pos], pos == block % 4
        while True:
            gain, signs = case_draw(seed)
            seed += 1
            if tuple(signs[:2]) == want_sign and (gain >= GAIN_SPLIT) == want_high:
                out.append(make_case(seed-1, seconds, 'stress', len(out) % 8))
                break
    return out


def consecutive(base, count, seconds):
    return [make_case(base+i, seconds, 'stress', i % 8) for i in range(count)]


# ----------------------------------------------------------------------------------------- networks
def untied_pair(spec):
    """Both pools as one untied 2-output genome (output 0 = the original pool, output 1 = its mirror), so that each
    pool can evolve on its own. Hidden ids are moved out of the way of output key 1."""
    m = mirror_spec(spec)

    def remap(i):
        if i < 0 or i == 0:
            return i
        if i == MIRROR_OFFSET:
            return 1
        return 1000+i-MIRROR_OFFSET if i > MIRROR_OFFSET else 10+i
    return dict(m, nodes=[dict(nd, id=remap(nd['id'])) for nd in m['nodes']],
                edges=[dict(e, source=remap(e['source']), target=remap(e['target'])) for e in m['edges']],
                outputs=[0, 1], tied=False)


def mirror_spec(spec):
    """Append the antagonist pool: a mirrored copy of every node and edge."""
    def source(s):
        return -(MIRROR[-s-1]+1) if s < 0 else s+MIRROR_OFFSET
    out = dict(spec)
    out['nodes'] = list(spec['nodes'])+[dict(n, id=n['id']+MIRROR_OFFSET) for n in spec['nodes']]
    out['edges'] = list(spec['edges'])+[dict(e, source=source(e['source']), target=e['target']+MIRROR_OFFSET)
                                        for e in spec['edges']]
    out['outputs'] = list(spec['outputs'])+[o+MIRROR_OFFSET for o in spec['outputs']]
    out['tied'] = True
    return out


class SteerV3(P.Steer):
    """pilot.Steer with calibration knobs, a configurable twitch length, an optional lockout between the two
    pools, and a phase offset for the classical law. With the default knobs the network path reproduces
    pilot.Steer exactly (selftest)."""

    def __init__(self, spec=None, law=None, lockout=False):
        super().__init__(spec, law)
        self.calib = dict(g_theta=1., g_q=1., d_out=0., twitch=P.TWITCH_TICKS)
        self.calib.update((spec or {}).get('calib', {}))
        self.lockout = lockout

    def reset(self):
        super().reset()
        if self.net is not None and self.calib['d_out']:
            self.net.bias[self.net.outputs] += self.calib['d_out']
        self.block, self.last_pool = 0, None

    def features(self, reading, phase):
        x = super().features(reading, phase)
        x[P.N_PHASE:P.N_PHASE+2] *= self.calib['g_theta']
        x[P.N_PHASE+2:P.N_PHASE+4] *= self.calib['g_q']
        return x

    def pool_of(self, output_index):
        n = len(self.net.outputs)
        return 0 if not self.spec.get('tied') else int(output_index >= n//2)

    def step(self, env, now, t):
        reading = env.last_reading
        self.estimator.update(now, float(reading[4]))
        phase = self.estimator.crank(now)
        fired = None
        if self.net is not None:
            x = self.features(reading, phase)
            fired = self.net.snn_substep(x) | self.net.snn_substep(x)
            self.fired = fired
        if t < gates.HANDOVER_T:
            return False, False
        if self.law is not None:
            u = -(self.law['k_theta']*reading[1]+self.law['k_q']*self.estimator.slow)
            psi = PULSE_PHASE+math.radians(self.law.get('psi_off_deg', 0.))+(0. if u > 0 else math.pi)
            fire = abs(u) > self.law['threshold'] and abs(math.remainder(phase-psi, 2*math.pi)) < self.law['width']/2
            onset, self.active = fire and not self.active, fire
            return fire, onset
        if self.block > 0:
            self.block -= 1
        if self.twitch > 0:
            self.twitch -= 1
            return True, False
        if fired is not None and fired.any():
            pool = self.pool_of(int(np.flatnonzero(fired)[0]))
            if self.lockout and self.block > 0 and pool != self.last_pool:
                return False, False  # reciprocal inhibition between antagonist pools
            self.twitch = int(self.calib['twitch'])-1
            self.block, self.last_pool = LOCKOUT_TICKS, pool
            return True, True
        return False, False


def task(args):
    name, c, spec, law, lockout = args
    return dict(P.run(P._ENV, c, SteerV3(spec, law, lockout)), controller=name)


def run_all(executor, jobs):
    """jobs: list of (name, case, spec, law, lockout)."""
    return list(executor.map(task, jobs, chunksize=1))


# ----------------------------------------------------------------------------------------- NEAT
class AdaptiveSpeciesSet(neat.DefaultSpeciesSet):
    """Speciation that never leaves more than MAX_SPECIES species. If it would, the threshold rises and the
    previous state is re-speciated. This prevents both the per-species elite glut seen in v2 and neat-python's
    'pop_size < species x min_species_size' error, which a warm-started population hits after one generation."""
    MAX_SPECIES = 5

    def speciate(self, config, population, generation):
        saved = copy.deepcopy((self.species, self.genome_to_species, self.indexer))
        for _ in range(40):
            super().speciate(config, population, generation)
            if len(self.species) <= self.MAX_SPECIES:
                return
            self.species, self.genome_to_species, self.indexer = copy.deepcopy(saved)
            self.species_set_config.compatibility_threshold *= 1.25
        super().speciate(config, population, generation)


def make_config(out, population, outputs):
    text = (Path(__file__).with_name('neat_template_v3.ini').read_text()
            .replace('{population}', str(population)).replace('{inputs}', str(P.INPUTS))
            .replace('{outputs}', str(outputs)))
    path = out/'neat.ini'
    path.write_text(text)
    cfg = neat.Config(DelayGenome, neat.DefaultReproduction, AdaptiveSpeciesSet, neat.DefaultStagnation, str(path))
    cfg.genome_config.activation_defs.add('plain_tanh', plain_tanh)
    return cfg


def genome_from_spec(cfg, key, spec, direct_need=True):
    """A DelayGenome carrying the spec's nodes and edges (one pool), plus optional zero-weight need->motor edges."""
    gc = cfg.genome_config
    g = DelayGenome(key)
    for n in spec['nodes']:
        node = g.create_node(gc, n['id'])
        node.bias = float(n['bias'])
        g.nodes[n['id']] = node
    edges = [dict(e) for e in spec['edges']]
    if direct_need:
        have = {(e['source'], e['target']) for e in edges}
        for o in spec['outputs']:
            for i in NEED_INPUTS:
                if (-(i+1), o) not in have:
                    edges.append(dict(source=-(i+1), target=o, weight=0., delay=0., enabled=True))
    for e in edges:
        key2 = (e['source'], e['target'])
        conn = g.create_connection(gc, *key2, gc.innovation_tracker.get_innovation_number(*key2, 'shared_initial'))
        conn.weight, conn.delay, conn.enabled = float(e['weight']), float(e['delay']), bool(e['enabled'])
        g.connections[key2] = conn
    return g


def export(g, tied, calib=None):
    spec = dict(kind='snn', inputs=P.INPUTS, genome_id=g.key, **P.SPEC, outputs=list(range(OUTPUT_KEYS)),
                nodes=[dict(id=k, bias=float(n.bias)) for k, n in sorted(g.nodes.items())],
                edges=[dict(source=a, target=b, weight=float(e.weight), delay=float(round(e.delay)),
                            enabled=bool(e.enabled)) for (a, b), e in sorted(g.connections.items())])
    if tied:
        spec = mirror_spec(spec)
    if calib:
        spec['calib'] = dict(calib)
    return spec


OUTPUT_KEYS = 1  # set in train(): genome output keys are 0..OUTPUT_KEYS-1


def warm_population(pop, cfg, source, tied):
    """Genome 0 is the exact source; the others are the source after two gentle mutate() rounds."""
    gc = cfg.genome_config
    keys = sorted(pop.population)
    top = max([n['id'] for n in source['nodes']]+[31])
    gc.node_indexer = itertools.count(top+1)
    for i, k in enumerate(keys):
        g = genome_from_spec(cfg, k, source)
        if i > 0:
            for _ in range(2):
                g.mutate(gc)
        pop.population[k] = g
    distances = [pop.population[a].distance(pop.population[b], gc) for a, b in itertools.combinations(keys, 2)]
    cfg.species_set_config.compatibility_threshold = max(.02, float(np.median(distances)))
    pop.species = AdaptiveSpeciesSet(cfg.species_set_config, pop.reporters)  # drop the random-genome species
    pop.species.speciate(cfg, pop.population, 0)
    return cfg.species_set_config.compatibility_threshold


def source_spec(name):
    """Warm-start source: 'v1' (one pool; mirrored at export when tied) or 'v1_pair' (untied 2-output pair
    initialised as v1 plus its mirror)."""
    v1 = json.loads(V1.read_text())
    return v1 if name == 'v1' else untied_pair(v1)


# ----------------------------------------------------------------------------------------- scoring helpers
def stats(rows):
    flown = [r for r in rows if r['termination'] == 'time_limit']
    f = lambda k: float(np.mean([r[k] for r in flown])) if flown else None
    return dict(episodes=len(rows), flown=len(flown), J=float(np.mean([r['objective'] for r in rows])),
                pitch_rms_deg=f('pitch_rms_deg'), pitch_peak_deg=f('pitch_peak_deg'), tilt_rms_deg=f('tilt_rms_deg'),
                height_rmse_cm=None if not flown else 100*f('height_rmse_m'), pulse_duty=f('pulse_duty'))


def directions(rows):
    """Both-direction rule from the onset rows (true need from the oracle, true crank phase)."""
    out = {}
    for need, centre, label in ((1, 135., 'need_up'), (-1, 315., 'need_down')):
        phases = np.array([o[0] for r in rows for o in r['onsets'] if o[1] == need])
        inside = np.abs((phases-centre+180.) % 360.-180.) <= 67.5 if len(phases) else np.array([])
        out[label] = dict(onsets=int(len(phases)), in_window=float(inside.mean()) if len(phases) else 0.)
    out['pass'] = all(out[k]['onsets'] >= 20 and out[k]['in_window'] >= .7 for k in ('need_up', 'need_down'))
    return out


def paired(rows_a, rows_b, n_boot=2000, seed=0):
    a = {r['seed']: r['objective'] for r in rows_a}
    b = {r['seed']: r['objective'] for r in rows_b}
    common = sorted(set(a) & set(b))
    d = np.array([a[s]-b[s] for s in common])
    rng = np.random.default_rng(seed)
    boot = [rng.choice(d, len(d)).mean() for _ in range(n_boot)] if len(d) else [float('nan')]
    return dict(n=len(d), mean_dJ=float(d.mean()) if len(d) else None, ci95=[float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5))], wins=int((d < 0).sum()))


def strata(rows):
    out = {}
    for r in rows:
        gain, signs = case_draw(r['seed'])
        for key in (f"gain{'>=' if gain >= GAIN_SPLIT else '<'}{GAIN_SPLIT}", f"sx{int(signs[0]):+d}"):
            out.setdefault(key, []).append(r)
    return {k: stats(v) for k, v in sorted(out.items())}


# ----------------------------------------------------------------------------------------- stages
def stage_d1(executor, out):
    """Choose the warm start and the lockout: 16 fresh stratified cases, 6 s."""
    v1, v2 = json.loads(V1.read_text()), json.loads(V2.read_text())
    law = json.loads(V2_LAW.read_text())['best']
    arms = [('pid_retuned', None, None, False), ('law_tuned', None, law, False), ('v1', v1, None, False),
            ('mirror_v1', mirror_spec(v1), None, False), ('mirror_v1_lockout', mirror_spec(v1), None, True),
            ('v2_champion', v2, None, False)]
    cs = stratified(BASES['d1'], n(16), 6.)
    rows = run_all(executor, [(name, c, s, l, lo) for name, s, l, lo in arms for c in cs])
    table = {name: dict(stats([r for r in rows if r['controller'] == name]),
                        directions=directions([r for r in rows if r['controller'] == name])) for name, *_ in arms}
    J = {name: table[name]['J'] for name in table}
    # Tie the pools only if the mirrored push-up pool helps as is; otherwise let both pools evolve separately
    # from the same mirror(v1) start (an untied pair), so that evolution can retune the push-up phase.
    mirror_ok = J['mirror_v1'] <= J['v1'] and table['mirror_v1']['directions']['need_up']['in_window'] >= .7
    lockout = J['mirror_v1_lockout'] <= J['mirror_v1']
    choice = dict(tied=bool(mirror_ok), source='v1' if mirror_ok else 'v1_pair', lockout=bool(lockout and mirror_ok))
    result = dict(stage='d1', cases=[c['seed'] for c in cs], table=table, choice=choice, rows=rows)
    (out/'d1.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(J=J, choice=choice), indent=1), flush=True)
    return choice


def stage_train(executor, out, a, choice):
    global OUTPUT_KEYS
    source = source_spec(choice['source'])
    OUTPUT_KEYS = len(source['outputs'])
    tied = choice['tied']
    random.seed(a.seed)
    cfg = make_config(out, a.population, OUTPUT_KEYS)
    pop = neat.Population(cfg)
    threshold0 = warm_population(pop, cfg, source, tied)
    validation = stratified(BASES['validation'], n(8), 6.)
    seen, validated, history, start = {}, [], [], time.monotonic()

    def score(genomes, config):
        generation = len(history)
        batch = stratified(BASES['train']+a.seed*100_000+generation*1_000, a.cases, 6.)
        specs = {gid: export(g, tied) for gid, g in genomes}
        rows = run_all(executor, [(str(gid), c, specs[gid], None, choice['lockout'])
                                  for gid, _ in genomes for c in batch])
        for k, (gid, g) in enumerate(genomes):
            js = [r['objective'] for r in rows[k*a.cases:(k+1)*a.cases]]
            g.fitness = -float(np.mean(js))
            seen.setdefault(gid, []).extend(js)
        row = dict(generation=generation, fitness_max=max(g.fitness for _, g in genomes),
                   fitness_mean=float(np.mean([g.fitness for _, g in genomes])), species=len(pop.species.species),
                   threshold=cfg.species_set_config.compatibility_threshold, wall_s=round(time.monotonic()-start, 1))
        if generation % a.validate_every == a.validate_every-1:
            # best first: genomes with >= 8 evaluated cases by cumulative mean J, then the rest by batch J
            ranked = sorted(genomes, key=lambda x: (len(seen[x[0]]) < 8, float(np.mean(seen[x[0]]))
                                                    if len(seen[x[0]]) >= 8 else -x[1].fitness))
            top = [gid for gid, _ in ranked[:a.validate_top]]
            vrows = run_all(executor, [(str(gid), c, specs[gid], None, choice['lockout'])
                                       for gid in top for c in validation])
            for gid in top:
                s = stats([r for r in vrows if r['controller'] == str(gid)])
                validated.append(dict(generation=generation, genome=gid, flown=s['flown'], J=s['J'], spec=specs[gid]))
            row['validation'] = [dict(genome=v['genome'], flown=v['flown'], J=round(v['J'], 3))
                                 for v in validated if v['generation'] == generation]
        # species-count controller: aim for 3-4 species; pull back hard well before neat's 17-species limit
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
    for v in sorted(validated, key=lambda v: (-v['flown'], v['J'])):  # an elite may be validated more than once
        unique.setdefault(v['genome'], v)
    best = list(unique.values())[:3]
    (out/'validated.json').write_text(json.dumps([dict(v, spec=None) for v in validated], indent=1))
    return dict(candidates=[v['spec'] for v in best], source=export(genome_from_spec(cfg, -1, source, False), tied),
                threshold0=threshold0, tied=tied)


def stage_select(executor, out, trained, choice):
    cs = stratified(BASES['select'], n(16), 6.)
    arms = [(f'candidate{i}', s) for i, s in enumerate(trained['candidates'])]+[('warm_source', trained['source'])]
    rows = run_all(executor, [(a, c, s, None, choice['lockout']) for a, s in arms for c in cs])
    table = {a: stats([r for r in rows if r['controller'] == a]) for a, _ in arms}
    eligible = [a for a, _ in arms if table[a]['flown'] == len(cs)] or [a for a, _ in arms]
    name = min(eligible, key=lambda a: table[a]['J'])
    selected = dict(arms)[name]
    (out/'select.json').write_text(json.dumps(dict(table=table, selected=name, rows=rows), indent=1))
    (out/'selected_raw.json').write_text(json.dumps(selected, indent=1))
    print(json.dumps(dict(select=table, selected=name), indent=1), flush=True)
    return selected


def stage_calibrate(executor, out, spec, choice):
    cs = P.balanced_cases('train', n(8), 6., offset=900_000)  # exactly the classical law's tuning cases (v2)
    default = CALIB_GRID.index(dict(g_theta=1., g_q=1., d_out=0., twitch=3))
    grid = CALIB_GRID if not QUICK else [CALIB_GRID[default], CALIB_GRID[0], CALIB_GRID[-1]]
    jobs = [(str(i), c, dict(spec, calib=k), None, choice['lockout']) for i, k in enumerate(grid) for c in cs]
    rows = run_all(executor, jobs)
    table = {i: stats([r for r in rows if r['controller'] == str(i)]) for i in range(len(grid))}
    best = min(table, key=lambda i: (-table[i]['flown'], table[i]['J']))
    calibrated = dict(spec, calib=grid[best])
    (out/'calibrate.json').write_text(json.dumps(dict(grid=grid, table=table, best=grid[best], rows=rows), indent=1))
    (out/'selected_calibrated.json').write_text(json.dumps(calibrated, indent=1))
    print(json.dumps(dict(calibration_best=grid[best], J=table[best]['J'], flown=table[best]['flown'],
                          default_J=table[grid.index(CALIB_GRID[default])]['J'])), flush=True)
    return calibrated


def stage_law_ext(executor, out):
    base = json.loads(V2_LAW.read_text())['best']
    grid = [dict(base, psi_off_deg=po, threshold=base['threshold']*ts, width=math.radians(w), k_q=base['k_q']*kq)
            for po in (-30., -15., 0., 15.) for ts in (.8, 1., 1.25) for w in (45., 60., 90.) for kq in (.5, 1.)]
    grid = grid if not QUICK else grid[:3]
    cs = P.balanced_cases('train', n(8), 6., offset=900_000)
    rows = run_all(executor, [(str(i), c, None, law, False) for i, law in enumerate(grid) for c in cs])
    table = {i: stats([r for r in rows if r['controller'] == str(i)]) for i in range(len(grid))}
    best = min(table, key=lambda i: (-table[i]['flown'], table[i]['J']))
    (out/'law_ext.json').write_text(json.dumps(dict(grid=grid, table=table, best=grid[best], rows=rows), indent=1))
    print(json.dumps(dict(law_ext_best=grid[best], J=table[best]['J']), indent=1), flush=True)
    return grid[best]


def stage_test(executor, out, calibrated, raw, law_ext, choice):
    rule = dict(only='snn_v3 (calibrated, validation-selected)',
                pass_rule=['flies 16/16 on the old 16-case test',
                           'J_snn <= 1.05 x J_law_tuned (v2 108-point law)',
                           '>= 20 onsets for need-up and need-down; >= 70 % within +-67.5 deg of 135 / 315 deg'],
                robust_pass='the same on the fresh 16-case test', registered_before_test=True)
    (out/'preregistration.json').write_text(json.dumps(rule, indent=1))
    v1, v2 = json.loads(V1.read_text()), json.loads(V2.read_text())
    law_tuned = json.loads(V2_LAW.read_text())['best']
    lo = choice['lockout']
    new = [('snn_v3', calibrated, None, lo), ('snn_v3_raw', raw, None, lo), ('mirror_v1', mirror_spec(v1), None, lo),
           ('law_ext', None, law_ext, False)]
    old_cases = P.cases('test', 16, 8.)[:n(16, 2)]
    old_rows = run_all(executor, [(a, c, s, l, x) for a, s, l, x in new for c in old_cases])
    v2rows = json.loads(V2_TEST.read_text())['rows']
    rename = {'snn_neat': 'v2_champion'}
    keep = {c['seed'] for c in old_cases}
    old_rows += [dict(r, controller=rename.get(r['controller'], r['controller'])) for r in v2rows if r['seed'] in keep]
    fresh_cases = consecutive(BASES['test3'], n(16, 2), 8.)
    every = new+[('pid_retuned', None, None, False), ('law_strong', None, P.CLASSICAL['law_strong'], False),
                 ('law_moderate', None, P.CLASSICAL['law_moderate'], False), ('law_tuned', None, law_tuned, False),
                 ('v2_champion', v2, None, False)]
    fresh_rows = run_all(executor, [(a, c, s, l, x) for a, s, l, x in every for c in fresh_cases])
    report = {}
    for label, rows in (('old16', old_rows), ('fresh16', fresh_rows)):
        names = sorted({r['controller'] for r in rows})
        by = {a: [r for r in rows if r['controller'] == a] for a in names}
        snn = by['snn_v3']
        verdict = dict(flown=stats(snn)['flown'] == len(snn),
                       J_ratio=stats(snn)['J']/stats(by['law_tuned'])['J'],
                       directions=directions(snn))
        verdict['J_ok'] = verdict['J_ratio'] <= 1.05
        verdict['pass'] = verdict['flown'] and verdict['J_ok'] and verdict['directions']['pass']
        report[label] = dict(table={a: stats(v) for a, v in by.items()},
                             directions={a: directions(v) for a, v in by.items() if a.startswith(('snn', 'mirror', 'v2'))},
                             paired_vs_law_tuned={a: paired(v, by['law_tuned']) for a, v in by.items() if a != 'law_tuned'},
                             strata={a: strata(v) for a, v in by.items()}, verdict=verdict)
    report['overall'] = dict(PASS=report['old16']['verdict']['pass'],
                             ROBUST_PASS=report['old16']['verdict']['pass'] and report['fresh16']['verdict']['pass'])
    (out/'test.json').write_text(json.dumps(dict(report=report, old_rows=old_rows, fresh_rows=fresh_rows), indent=1))
    print(json.dumps({k: (v['verdict'] if 'verdict' in v else v) for k, v in report.items()}, indent=1), flush=True)
    for label in ('old16', 'fresh16'):
        print(label, json.dumps({a: dict(J=round(s['J'], 3), flown=s['flown'], pitch=s['pitch_rms_deg'] and round(
            s['pitch_rms_deg'], 2), height=s['height_rmse_cm'] and round(s['height_rmse_cm'], 2))
            for a, s in report[label]['table'].items()}, indent=1), flush=True)
    return report


# ----------------------------------------------------------------------------------------- self test
def selftest():
    """No simulation: mirror map, knob defaults, stratified sampler, warm population and species start."""
    assert sorted(MIRROR) == list(range(12)) and all(MIRROR[MIRROR[i]] == i for i in range(12))
    v1 = json.loads(V1.read_text())
    m = mirror_spec(v1)
    rng = np.random.default_rng(0)
    a, b = P.SpikingGraph(m), P.SpikingGraph(m)
    for _ in range(300):  # mirrored inputs into the copy == original inputs into the original pool
        x = rng.random(12)*(rng.random(12) < .3)
        xm = np.zeros(12)
        for i in range(12):
            xm[MIRROR[i]] = x[i]
        fa, fb = a.snn_substep(x), b.snn_substep(xm)
        assert fa[0] == fb[1] and fa[1] == fb[0], 'mirror equivalence failed'
    s1, s2 = P.Steer(v1), SteerV3(v1)
    s1.reset(), s2.reset()
    for _ in range(300):
        x = rng.random(12)
        assert np.array_equal(s1.net.snn_substep(x), s2.net.snn_substep(x))
    x = rng.random(12)
    s2.calib = dict(g_theta=1., g_q=1., d_out=0., twitch=3)
    s2.estimator = P.WingbeatPhase(P.PHASE_OFFSET)
    assert np.array_equal(P.Steer.features(s2, np.r_[0., .05, 0., 0., .3, 0.], .3),
                          SteerV3.features(s2, np.r_[0., .05, 0., 0., .3, 0.], .3))
    cs = stratified(BASES['train'], 8, 6.)
    for k, c in enumerate(cs):
        gain, signs = case_draw(c['seed'])
        assert tuple(signs[:2]) == COMBOS[k % 4] and (gain >= GAIN_SPLIT) == (k % 4 == (k//4) % 4)
    out = Path('runs/spike_steer/selftest_v3')
    out.mkdir(parents=True, exist_ok=True)
    global OUTPUT_KEYS
    OUTPUT_KEYS = 1
    cfg = make_config(out, 32, 1)
    pop = neat.Population(cfg)
    t0 = warm_population(pop, cfg, v1, True)
    g0 = pop.population[sorted(pop.population)[0]]
    spec0 = export(g0, True)
    ref = mirror_spec(v1)
    ref_edges = {(e['source'], e['target']): (e['weight'], e['delay'], e['enabled']) for e in ref['edges']}
    got = {(e['source'], e['target']): (e['weight'], e['delay'], e['enabled']) for e in spec0['edges']}
    extra = {k: v for k, v in got.items() if k not in ref_edges}
    assert all(got[k] == v for k, v in ref_edges.items()), 'warm genome 0 differs from mirror(v1)'
    assert all(v[0] == 0. for v in extra.values()) and len(extra) == 8, 'expected 8 zero-weight need->motor edges'
    species0 = len(pop.species.species)
    pair = untied_pair(v1)  # same phenotype as mirror(v1), but both pools evolvable
    ga, gb = P.SpikingGraph(m), P.SpikingGraph(pair)
    for _ in range(300):
        x = rng.random(12)*(rng.random(12) < .3)
        assert np.array_equal(ga.snn_substep(x), gb.snn_substep(x)), 'untied pair differs from mirror(v1)'
    counts = {}
    for tied, src in ((True, v1), (False, pair)):  # dummy-fitness evolution: no neat error, species stay capped
        OUTPUT_KEYS = len(src['outputs'])
        cfg = make_config(out, 32, OUTPUT_KEYS)
        pop = neat.Population(cfg)
        warm_population(pop, cfg, src, tied)
        draw = random.Random(1)

        def fitness(genomes, config):
            for _, g in genomes:
                g.fitness = draw.random()
        pop.run(fitness, 8)
        counts['tied' if tied else 'pair'] = len(pop.species.species)
        assert counts['tied' if tied else 'pair'] <= AdaptiveSpeciesSet.MAX_SPECIES
        export(pop.population[sorted(pop.population)[0]], tied)
    print(json.dumps(dict(selftest='ok', species=species0, threshold0=t0, species_after_8_dummy_generations=counts,
                          stratified_gains=[round(case_draw(c['seed'])[0], 4) for c in cs]), indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'all'])
    p.add_argument('--output', default='runs/spike_steer/pilot_v3_s3')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--generations', type=int, default=16)
    p.add_argument('--cases', type=int, default=4)
    p.add_argument('--validate-every', type=int, default=4)
    p.add_argument('--validate-top', type=int, default=3)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--quick', action='store_true', help='tiny end-to-end smoke run of every stage')
    a = p.parse_args()
    global QUICK
    QUICK = a.quick
    if QUICK:
        a.population, a.generations, a.validate_every, a.validate_top = 8, 2, 1, 2
    if a.stage == 'selftest':
        selftest()
        return
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with P.pool(a.workers) as executor:
        choice = stage_d1(executor, out)
        trained = stage_train(executor, out, a, choice)
        raw = stage_select(executor, out, trained, choice)
        calibrated = stage_calibrate(executor, out, raw, choice)
        law_ext = stage_law_ext(executor, out)
        stage_test(executor, out, calibrated, raw, law_ext, choice)


if __name__ == '__main__':
    main()
