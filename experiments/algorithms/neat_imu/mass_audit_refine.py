"""Independent, finer-sampled checks of the mass-distribution audit."""
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from .activation import cases, gyro_network, signed_map
from .mass_audit_run import BASE, ROOT, fixed, fly
from .mass_model import describe
from .robust import summarize
from .train import write_json


def main():
    destination = ROOT / 'refinement'
    destination.mkdir(exist_ok=True)
    if (destination / 'results.json').exists():
        raise RuntimeError('Do not overwrite completed measurements')
    started = time.monotonic()
    models = json.loads((ROOT / 'models.json').read_text())
    original = Path(models[0]['path'])
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    # Archive the exact XML used, including the unchanged source model.
    for model in models:
        path = Path(model['path'])
        (destination / (model['label'] + '.xml')).write_bytes(path.read_bytes())
    hashes = {model['label']: hashlib.sha256(Path(model['path']).read_bytes()).hexdigest()
              for model in models}
    write_json(destination / 'model_hashes.json', hashes)
    all_measurements = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        # More samples and more complete cycles than the first authority scan.
        commands = [(BASE, .05, 0.), (BASE, -.05, 0.),
                    (BASE, 0., .05), (BASE, 0., -.05)]
        selected = [models[0], models[2]]
        rows = list(pool.map(fixed, [(m['path'], c, True)
                                    for m in selected for c in commands]))
        all_measurements.extend(rows)
        authority = []
        for i, model in enumerate(selected):
            block = rows[i * 4:(i + 1) * 4]
            derivatives = [(np.asarray(block[k]['torque_com_Nm']) -
                            np.asarray(block[k + 1]['torque_com_Nm'])) / .1
                           for k in (0, 2)]
            authority.append(dict(model=model['label'], measurements=block,
                                  roll_pitch_authority_Nm_per_rad=
                                  np.column_stack(derivatives)[:2].tolist(),
                                  scope='Cycle-averaged aerodynamic moment about whole-system COM; '
                                        'not a free-flight base-link angular-acceleration Jacobian.'))
        write_json(destination / 'authority_fine.json', authority)
        print(json.dumps(dict(stage='authority', rows=authority)), flush=True)

        # Refine only the lift estimates whose residual exceeds 0.2%.
        matched = json.loads((ROOT / 'lift_matching.json').read_text())
        history = []
        for iteration in range(4):
            active = [i for i, row in enumerate(matched)
                      if abs(row['force_N'][2] / row['weight_N'] - 1.) > .002]
            if not active:
                break
            probes = list(pool.map(fixed, [
                (models[i]['path'], [matched[i]['command'][0] + delta, 0., 0.], True)
                for i in active for delta in (-30., 30.)]))
            all_measurements.extend(probes)
            tasks = []
            for index, i in enumerate(active):
                low, high = probes[2 * index:2 * index + 2]
                slope = (high['force_N'][2] - low['force_N'][2]) / 60.
                if slope <= 0:
                    raise RuntimeError('Local lift curve is not increasing; inspect measurements')
                step = np.clip((matched[i]['weight_N'] - matched[i]['force_N'][2]) / slope,
                               -150., 150.)
                tasks.append((models[i]['path'], [matched[i]['command'][0] + step, 0., 0.], True))
                history.append(dict(model=models[i]['label'], iteration=iteration,
                                    low=low, high=high, slope=slope))
            checks = list(pool.map(fixed, tasks))
            all_measurements.extend(checks)
            for i, row in zip(active, checks):
                row.update(model=models[i]['label'],
                           relative_vertical_lift_error=row['force_N'][2] / row['weight_N'] - 1.,
                           hover_attitude_rad=[0., 0., 0.],
                           scope='Vertical lift only; not an angular-moment trim or validated hovering state')
                history.append(dict(model=models[i]['label'], iteration=iteration, check=row))
                matched[i] = row
        write_json(destination / 'lift_refinement_history.json', history)
        write_json(destination / 'lift_refined.json', matched)
        print(json.dumps(dict(stage='lift', rows=matched)), flush=True)

        # Re-run paired free-flight comparisons at all four refined baselines.
        old = json.loads((ROOT / 'frozen_old_policy.json').read_text())
        batch = cases(range(898810000, 898810008), 8.)
        tasks = []
        for model, trim in zip(models, matched):
            constant = gyro_network(signed_map(trim), 0., 0., activation='clamped')
            policy = copy.deepcopy(old)
            policy['action_map'] = signed_map(trim)
            for label, payload in (('constant_neutral', constant), ('old_neat_rebased', policy)):
                tasks.extend((model, trim, payload, case, label, 'refined_weight_matched')
                             for case in batch)
        episodes = list(pool.map(fly, tasks))
    write_json(destination / 'episodes.json', episodes)

    # Position sensitivity is a geometric uncertainty study, not measured hardware data.
    sensitivity = []
    reference = Path(models[2]['path'])
    for shift_mm in (-5., 0., 5.):
        tree = ET.parse(reference)
        inertia = tree.find(".//body[@name='base_link']/inertial")
        position = np.fromstring(inertia.get('pos'), sep=' ')
        position[2] += shift_mm / 1000.
        inertia.set('pos', ' '.join(f'{x:.17g}' for x in position))
        path = destination / f'lower_com_shift_{shift_mm:+g}mm.xml'
        tree.write(path, encoding='utf-8', xml_declaration=True)
        sensitivity.append(dict(base_link_inertial_z_shift_mm=shift_mm,
                                scope='Assumed COM-position uncertainty, mass and body COM inertia unchanged',
                                properties=describe(path)))
    write_json(destination / 'com_sensitivity.json', sensitivity)
    summaries = [dict(model=model['label'], policy=label, summary=summarize([
        row for row in episodes if row['model'] == model['label'] and row['policy'] == label]))
        for model in models for label in ('constant_neutral', 'old_neat_rebased')]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
    result = dict(status='completed', original_model_sha256=original_hash,
                  source_model_unchanged=True, summaries=summaries,
                  control_steps=sum(row['steps'] for row in episodes),
                  fixed_physics_steps=sum(row['physics_steps'] for row in all_measurements),
                  wall_seconds=time.monotonic() - started)
    write_json(destination / 'results.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
