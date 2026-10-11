"""Parallel evaluation of wingbeat controllers on paired cases and gust levels.

    python -m experiments.algorithms.wingbeat_snn.evaluate gust_scan --scratch <dir> --output <json>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import numpy as np
from ..snn_pid_compare.common import prepare_source
from .plant import Plant, HandLaw, cases, gusts

_PLANT = None


def worker_init(source):
    global _PLANT
    _PLANT = Plant(source)


def make_controller(spec):
    kind = spec['kind']
    if kind == 'pid':
        return None
    if kind == 'hand':
        return HandLaw(**spec.get('params', {}))
    if kind == 'graph':
        from .networks import GraphController
        return GraphController(spec['policy'], phase_inputs=spec['phase_inputs'])
    raise ValueError(kind)


def run_task(task):
    spec, c, peak = task
    out = _PLANT.run(c, make_controller(spec), gusts(peak) if peak else ())
    out.update(controller=spec['name'], gust_peak_m_s=peak)
    return out


def pool(source, workers):
    source = Path(source)
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(source),))


def evaluate(executor, specs, case_list, peaks):
    tasks = [(s, c, p) for p in peaks for c in case_list for s in specs]
    return list(executor.map(run_task, tasks))


def summarize(rows, keys=('tilt_rms_deg', 'pitch_peak_deg', 'foreaft_max_m', 'height_rmse_m', 'drive_activity',
                          'phase_error_rms_deg')):
    out = {}
    for r in rows:
        out.setdefault(str(r['gust_peak_m_s']), {}).setdefault(r['controller'], []).append(r)
    summary = {}
    for peak, by in out.items():
        for name, items in by.items():
            flown = [r for r in items if r['termination'] == 'time_limit']
            entry = dict(episodes=len(items), completed=len(flown))
            for k in keys:
                values = [r[k] for r in flown if k in r and r[k] == r[k]]
                entry[k] = float(np.mean(values)) if values else None
            summary.setdefault(peak, {})[name] = entry
    return summary


def paired(rows, a, b, key):
    """a-b per (gust, seed) where both flew the full episode; failures counted separately."""
    index = {(r['gust_peak_m_s'], r['seed'], r['controller']): r for r in rows}
    out = {}
    for peak in sorted({r['gust_peak_m_s'] for r in rows}):
        diffs, failures = [], {a: 0, b: 0}
        for (p, seed, name), r in index.items():
            if p != peak or name != a or (p, seed, b) not in index:
                continue
            other = index[(p, seed, b)]
            failures[a] += r['termination'] != 'time_limit'
            failures[b] += other['termination'] != 'time_limit'
            if r['termination'] == other['termination'] == 'time_limit':
                diffs.append(r[key]-other[key])
        out[str(peak)] = dict(mean=float(np.mean(diffs)) if diffs else None,
                              better=int(np.sum(np.array(diffs) < 0)), n=len(diffs), failures=failures)
    return out


BASELINES = [dict(name='B1_pid', kind='pid'),
             dict(name='B2_hand', kind='hand', params=dict(highpass_s=None)),
             dict(name='B2plus_hand_hp1s', kind='hand', params=dict(highpass_s=1.))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['gust_scan', 'compare'])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--policy', action='append', default=[], help='name=path/to/best_policy.json (compare)')
    p.add_argument('--levels', type=float, nargs='+', default=[0., .5, 1., 1.25, 1.5])
    p.add_argument('--count', type=int, default=8)
    a = p.parse_args()
    if a.stage == 'gust_scan':
        specs, case_list, levels, references = BASELINES, cases('development', 8), (0., .5, 1., 1.5, 2., 3.), ['B1_pid']
    else:
        # Held-out test split; the policies were selected on the validation split only.
        specs = [BASELINES[0], BASELINES[2]]
        for item in a.policy:
            name, path = item.split('=', 1)
            specs.append(dict(name=name, kind='graph', policy=json.loads(Path(path).read_text()), phase_inputs=True))
        case_list, levels, references = cases('test', a.count), a.levels, ['B1_pid', 'B2plus_hand_hp1s']
    with pool(a.scratch/'source', a.workers) as executor:
        rows = evaluate(executor, specs, case_list, levels)
    result = dict(stage=a.stage, summary=summarize(rows),
                  paired={f'{x["name"]}__minus__{ref}': {k: paired(rows, x['name'], ref, k) for k in
                                                         ('pitch_peak_deg', 'tilt_rms_deg', 'height_rmse_m')}
                          for ref in references for x in specs if x['name'] != ref},
                  episodes=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, indent=1)[:15000])


if __name__ == '__main__':
    main()
