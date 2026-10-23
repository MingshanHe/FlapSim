"""Draw the active inference graph saved in best_policy.json."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(policy_path, output):
    payload = json.loads(Path(policy_path).read_text())
    inputs, outputs, nodes = payload['inputs'], payload['outputs'], payload['nodes']
    depth = {key: 0 for key in inputs}
    edges = []
    for node in nodes:
        sources = [source for source, _ in node['links']]
        depth[node['key']] = 1+max((depth.get(source, 0) for source in sources), default=0)
        edges.extend((source, node['key'], weight) for source, weight in node['links'])
    final_depth = max(depth.values(), default=0)+1
    for key in outputs:
        depth[key] = final_depth
    columns = {}
    for key, layer in depth.items():
        columns.setdefault(layer, []).append(key)
    positions = {}
    for layer, keys in sorted(columns.items()):
        for key, height in zip(keys, np.linspace(.1, .9, len(keys)) if len(keys) > 1 else [.5]):
            positions[key] = (layer/final_depth, height)
    fig, ax = plt.subplots(figsize=(max(9, len(columns)*2), max(5, len(inputs)*.4)))
    for source, target, weight in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.annotate('', (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#167aa4' if weight >= 0 else '#c85040',
                                    alpha=.55, lw=.5+min(abs(weight), 2.), shrinkA=14, shrinkB=14), zorder=1)
    names = dict(zip(inputs, payload['input_names']))
    names.update(zip(outputs, payload['output_names']))
    for key, (x, y) in positions.items():
        color = '#dcedf4' if key in inputs else '#e1edd5' if key in outputs else '#f4e1c1'
        ax.scatter([x], [y], s=650, color=color, edgecolor='#344250', zorder=3)
        ax.text(x, y, str(key) if key not in names else '', ha='center', va='center', fontsize=9, zorder=4)
        if key in inputs:
            ax.text(x-.05, y, names[key], ha='right', va='center', fontsize=10)
        elif key in outputs:
            ax.text(x+.05, y, names[key], ha='left', va='center', fontsize=10)
        else:
            bias = next(n['bias'] for n in nodes if n['key'] == key)
            ax.text(x, y-.065, f'b={bias:.2f}', ha='center', fontsize=8)
    active_hidden = sum(key not in inputs and key not in outputs for key in positions)
    ax.set_title(f'NEAT controller: {len(inputs)} inputs, {active_hidden} active hidden nodes, '
                 f'{len(outputs)} outputs\n{len(edges)} active connections; blue positive, red negative', pad=18)
    ax.set(xlim=(-.28, 1.4), ylim=(-.03, 1.04))
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    draw(args.policy, args.output)
