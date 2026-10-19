"""Figures for docs/algorithm.md (PNG + PDF in docs/figs/).

    python docs/figs/make_algorithm_figs.py              # diagrams + training curves (no simulation)
    MUJOCO_GL=osmesa python docs/figs/make_algorithm_figs.py --sim   # also the closed-loop trace figure (one episode)

Diagrams are drawn from the method description; data figures read archived results only:
artifacts/spike_steer/plan_v4/r5/finetune_s*/history.jsonl (ES), plan_v4/r5/claude_ea/reselect_r5_s*.json (selection),
confirm_m1/ (M1). The trace figure flies round-5 seed 4 on M1 nod test case 1 and logs the network state per tick.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R5 = ROOT/'artifacts/spike_steer/plan_v4/r5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})

INK, MUTED, GRIDC = '#1e252e', '#6b7380', '#d9dde3'
C_REFLEX, C_MUSCLE, C_LEARN, C_CLASSIC = '#dbe8f6', '#fde3c8', '#d9efdf', '#eceef1'
E_REFLEX, E_MUSCLE, E_LEARN, E_CLASSIC = '#2f6db5', '#d4781f', '#1f8a4c', '#8a929e'
SEEDS = [1, 2, 3, 4, 5]
SEED_COL = {1: '#2f6db5', 2: '#d4781f', 3: '#7a52b3', 4: '#1f8a4c', 5: '#c23b3b'}


def box(ax, x, y, w, h, text, fc='white', ec=INK, fs=10, bold=False, lw=1.3, ha='center', va='center', r=.02, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))
    ty = y+h/2+(.012 if sub else 0)
    ax.text(x+w/2 if ha == 'center' else x+.012, ty, text, ha=ha, va=va, fontsize=fs, color=INK,
            fontweight='bold' if bold else 'normal')
    if sub:
        ax.text(x+w/2 if ha == 'center' else x+.012, y+h/2-.03, sub, ha=ha, va='center', fontsize=fs-2, color=MUTED)
    return (x, y, w, h)


def arrow(ax, p0, p1, text=None, col=INK, lw=1.3, style='-|>', ls='-', fs=8.5, toff=(0, .012), rad=0., ha='center'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=col, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))
    if text:
        ax.text((p0[0]+p1[0])/2+toff[0], (p0[1]+p1[1])/2+toff[1], text, ha=ha, va='bottom', fontsize=fs, color=col)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


def save(fig, name):
    fig.savefig(OUT/f'{name}.png')
    fig.savefig(OUT/f'{name}.pdf')
    plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------------------------------------------------------------
def fig_architecture():
    fig, ax = canvas(13, 6.6)
    ax.text(.01, .965, 'Bio-inspired hierarchical controller (one 2 ms tick)', fontsize=14, fontweight='bold', color=INK)
    # sensors / commands
    box(ax, .01, .62, .12, .14, 'IMU + ToF', fc='white', sub='11-D reading, 500 Hz')
    box(ax, .01, .36, .12, .12, 'Task command', fc='white', sub='[nod, dash] ∈ {0,1}')
    # descending layer
    ax.add_patch(FancyBboxPatch((.17, .53), .40, .36, boxstyle='round,pad=0,rounding_size=.02', fc=C_LEARN, ec=E_LEARN, lw=1.6))
    ax.text(.18, .865, 'Descending command layer (learned)', fontsize=11.5, fontweight='bold', color=E_LEARN)
    box(ax, .19, .63, .10, .17, 'features', sub='20-D, rectified\n(ON/OFF halves)', fs=9.5)
    box(ax, .32, .63, .11, .17, 'delay lines', sub='11 taps\n0–20 ms', fs=9.5)
    box(ax, .455, .70, .10, .10, 'LIF SNN', sub='8 hidden', fs=9.5)
    box(ax, .455, .565, .10, .095, 'outputs', sub='up/down · g', fs=9.5)
    arrow(ax, (.29, .715), (.32, .715))
    arrow(ax, (.43, .75), (.455, .75))
    arrow(ax, (.505, .70), (.505, .66))
    arrow(ax, (.13, .69), (.19, .715))
    arrow(ax, (.13, .42), (.19, .66), rad=-.2)
    # gate
    ax.add_patch(FancyBboxPatch((.62, .58), .14, .25, boxstyle='round,pad=0,rounding_size=.02', fc='#fff7e0', ec='#b58a12', lw=1.3))
    ax.text(.69, .785, 'Command gate', ha='center', fontsize=10.5, fontweight='bold')
    ax.text(.69, .67, 'learned in charge if\ncommand on or < 1 s\nafter; else classical law', ha='center', va='center',
            fontsize=8.5, color=MUTED, linespacing=1.4)
    arrow(ax, (.555, .615), (.62, .64), col=E_LEARN)
    ax.text(.587, .585, 'fire, ref', ha='center', va='top', fontsize=8.5, color=E_LEARN)
    # classical law
    box(ax, .37, .345, .20, .13, 'Classical phase-locked\npulse law', fc=C_CLASSIC, ec=E_CLASSIC, fs=9.5,
        sub='law_tuned · pulse_tuned (tilt > 25°)')
    box(ax, .19, .345, .14, .13, 'PLL', fc=C_CLASSIC, ec=E_CLASSIC, sub='wingbeat phase\nfrom pitch gyro', fs=9.5)
    arrow(ax, (.13, .66), (.19, .44), col=E_CLASSIC, rad=.15)
    arrow(ax, (.33, .41), (.37, .41), col=E_CLASSIC)
    arrow(ax, (.57, .41), (.65, .58), 'fire_law, ref = 0', col=E_CLASSIC, toff=(.04, -.02))
    # refractory
    box(ax, .80, .70, .18, .11, 'Refractory 37 ms', fc='#fff7e0', ec='#b58a12', fs=9.5, sub='≤ 1 learned pulse onset / wingbeat')
    arrow(ax, (.76, .755), (.80, .755), 'fire', toff=(0, .005))
    # reflex layer
    ax.add_patch(FancyBboxPatch((.62, .14), .15, .26, boxstyle='round,pad=0,rounding_size=.02', fc=C_REFLEX, ec=E_REFLEX, lw=1.6))
    ax.text(.695, .365, 'Reflex layer (PID)', fontsize=11, fontweight='bold', color=E_REFLEX, ha='center')
    ax.text(.695, .24, 'attitude: α, β ← PID(e)\ne = [0, ref] − [roll, pitch]\naltitude: ω ← PID(z)',
            fontsize=9, color=INK, ha='center', va='center')
    arrow(ax, (.69, .58), (.69, .40), 'ref (pitch set-point)', col=E_REFLEX, toff=(.005, -.02), ha='left')
    ax.plot([.13, .15, .15], [.64, .64, .27], color=E_REFLEX, lw=1.3)
    arrow(ax, (.15, .27), (.62, .27), 'IMU reading', col=E_REFLEX, toff=(-.15, .005))
    # muscle layer
    ax.add_patch(FancyBboxPatch((.82, .40), .16, .21, boxstyle='round,pad=0,rounding_size=.02', fc=C_MUSCLE, ec=E_MUSCLE, lw=1.6))
    ax.text(.90, .575, 'Steering-muscle layer', fontsize=10.5, fontweight='bold', color=E_MUSCLE, ha='center')
    ax.text(.90, .47, 'fire → ω = 3807 rad/s\nfor 3 ticks (6 ms)\noverrides PID ω', fontsize=9, ha='center', va='center')
    arrow(ax, (.89, .70), (.89, .61), col=E_MUSCLE)
    # actuators
    box(ax, .82, .14, .16, .14, 'Actuators', fc='white', sub='α, β ±0.3 rad · ω 3115–3807\n8.2 ms smoothing', fs=10)
    arrow(ax, (.77, .21), (.82, .21), col=E_REFLEX)
    ax.text(.795, .225, 'α, β, ω', ha='center', va='bottom', fontsize=8.5, color=E_REFLEX)
    arrow(ax, (.90, .40), (.90, .28), 'ω pulse', col=E_MUSCLE, toff=(.005, -.01), ha='left')
    ax.text(.01, .04, 'Fixed interface: sensors, actuators, update rate and limits are those of the original environment. '
            'Pulse direction (pitch up / down) is set only by its timing within the wingbeat (≈135° up, ≈315° down).',
            fontsize=9, color=MUTED)
    save(fig, 'fig1_architecture')


def fig_snn(spec):
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0, 0, .64, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    asp = 13*.64/6.2

    def circ(x, y, r, **kw):
        ax.add_patch(Ellipse((x, y), 2*r, 2*r*asp, **kw))
    ax.text(.02, .955, 'Descending SNN: 20 inputs × 11 delays → 8 LIF → 2 spiking outputs + graded unit (2035 parameters)',
            fontsize=12.5, fontweight='bold', color=INK)
    names = ['roll', 'pitch', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    labels = [f'{n}+' for n in names]+[f'{n}−' for n in names]+['bias', '0', 'nod', 'dash']
    W_in = np.asarray(spec['W_in'])
    wsum = W_in.sum(2)
    ys = np.linspace(.88, .08, 20)
    hy = np.linspace(.80, .18, 8)
    xi, xd, xh, xo = .10, .25, .55, .82
    for i, (lab, y) in enumerate(zip(labels, ys)):
        col = '#2f6db5' if i < 16 else ('#1f8a4c' if i >= 18 else MUTED)
        circ(xi, y, .011, fc='white', ec=col, lw=1.4)
        ax.text(xi-.02, y, lab, ha='right', va='center', fontsize=8, color=col)
        for k in range(11):                                  # delay taps (schematic)
            ax.plot([xi+.015+k*.011], [y], 's', ms=2.2, color=col, alpha=.55)
    ax.text(xd-.02, .925, 'delay taps 0,2,…,20 ms', fontsize=8.5, color=MUTED, ha='center')
    wmax = np.abs(wsum).max()
    for h in range(8):
        for i in range(20):
            w = wsum[h, i]
            if abs(w) < .25*wmax:
                continue
            ax.plot([xi+.14, xh-.025], [ys[i], hy[h]], color=E_MUSCLE if w > 0 else E_REFLEX, lw=.4+2.2*abs(w)/wmax, alpha=.55)
    for h, y in enumerate(hy):
        circ(xh, y, .025, fc='#fff4e4', ec=INK, lw=1.3)
        ax.text(xh, y, f'h{h+1}', ha='center', va='center', fontsize=8.5)
    for (lab, y, col) in (('up', .66, '#c23b3b'), ('down', .50, '#c23b3b')):
        circ(xo, y, .03, fc='#fde8e6', ec=col, lw=1.5)
        ax.text(xo, y, lab, ha='center', va='center', fontsize=9)
    ax.add_patch(Rectangle((xo-.03, .25), .06, .08, fc=C_REFLEX, ec=E_REFLEX, lw=1.5))
    ax.text(xo, .29, 'g', ha='center', va='center', fontsize=11, fontweight='bold')
    for h, y in enumerate(hy):
        for yo in (.66, .50, .29):
            ax.plot([xh+.025, xo-.03], [y, yo], color=MUTED, lw=.5, alpha=.5)
    ax.text((xh+xo)/2, .88, 'hidden → output: 11 delays 0–20 ms, ×5', fontsize=8.5, color=MUTED, ha='center')
    ax.text(xo+.045, .58, 'either spike\n→ motor pulse\n(timing = direction)', fontsize=8.5, va='center')
    ax.text(xo+.045, .29, 'τ_g = 20 ms\nref = 0.7·tanh(g)\n→ PID set-point', fontsize=8.5, va='center')
    ax.text(.02, .02, 'Edges drawn: input→hidden weights summed over delays, |w| ≥ 25% of max (orange +, blue −), '
            'round-5 seed 4.', fontsize=8.5, color=MUTED)
    # right: LIF equations + parameter table
    ax2 = fig.add_axes([.65, 0, .35, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.add_patch(FancyBboxPatch((.02, .50), .94, .40, boxstyle='round,pad=0,rounding_size=.02', fc='#f7f8fa', ec=GRIDC))
    ax2.text(.06, .86, 'LIF dynamics (2 × 1 ms substeps per tick)', fontsize=10.5, fontweight='bold')
    eq = ('$I_h=\\sum_i\\sum_k W^{in}_{h,i,k}\\,x_i(t-2k\\,\\mathrm{ms})+b_h$\n'
          '$c_h\\leftarrow a_s c_h+(1-a_s) I_h$\n'
          '$v_h\\leftarrow a_m v_h+(1-a_m) c_h$\n'
          '$s_h=[v_h\\geq 1],\\;\\; v_h\\leftarrow v_h-s_h$\n'
          '$I_o=5\\sum_h\\sum_d W^{out}_{o,h,d}\\,s_h(t-d)+b_o$\n'
          '$ref=0.7\\,\\tanh(g)$')
    ax2.text(.06, .82, eq, fontsize=10, va='top', linespacing=1.7)
    rows = [('W_in', '8×20×11', '1760'), ('b_h', '8', '8'), ('W_out', '2×8×11', '176'), ('b_o', '2', '2'),
            ('W_g', '8×11', '88'), ('b_g', '1', '1'), ('total', '', '2035')]
    ax2.text(.06, .44, 'parameters (searched by ES)', fontsize=10.5, fontweight='bold')
    for r, (a, b, c) in enumerate(rows):
        y = .39-r*.045
        ax2.text(.08, y, a, fontsize=9.5, fontweight='bold' if a == 'total' else 'normal')
        ax2.text(.40, y, b, fontsize=9.5, color=MUTED)
        ax2.text(.80, y, c, fontsize=9.5, ha='right', fontweight='bold' if a == 'total' else 'normal')
    ax2.text(.06, .03, f"τ_m = {spec['tau_m']*1e3:.2f} ms, τ_s = {spec['tau_s']*1e3:.2f} ms\n(learned in imitation, fixed in ES)",
             fontsize=9, color=MUTED)
    save(fig, 'fig2_snn')


def fig_gating():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    t = np.linspace(0, 5, 2000)
    rows = [('takeoff guard (t < 1.5 s)', [(0, 1.5)], '#bfc5cd'),
            ('task command', [(2.4, 2.95)], E_LEARN),
            ('learned layer in charge', [(2.4, 3.95)], E_MUSCLE),
            ('classical law in charge', [(1.5, 2.4), (3.95, 5)], E_CLASSIC)]
    for r, (lab, spans, col) in enumerate(rows):
        y = len(rows)-r
        for a, b in spans:
            ax.add_patch(Rectangle((a, y-.32), b-a, .64, fc=col, ec='none', alpha=.85))
        ax.text(-.08, y, lab, ha='right', va='center', fontsize=9.5)
    ax.annotate('', xy=(3.95, 2.62), xytext=(2.95, 2.62), arrowprops=dict(arrowstyle='<->', color=INK))
    ax.text(3.45, 2.67, 'tail 1.0 s', ha='center', va='bottom', fontsize=9)
    ax.text(2.42, 4.45, 'T0 = 2.4 s', fontsize=9, color=E_LEARN)
    ax.text(2.97, 3.55, 'goal reached → command off', fontsize=8.5, color=E_LEARN)
    ax.axvline(2.4, color=E_LEARN, lw=.8, ls=':')
    ax.set_xlim(0, 5)
    ax.set_ylim(.4, 4.7)
    ax.set_yticks([])
    ax.set_xlabel('time in episode (s)')
    ax.spines['left'].set_visible(False)
    ax.set_title('Command gating: the learned layer acts only during a manoeuvre; the network and the PLL run every tick',
                 fontsize=11.5, loc='left', fontweight='bold')
    fig.text(.01, -.04, 'Not in charge and AHRS tilt > 25° → switch to pulse_tuned (righting reflex) for the rest of the episode.',
             fontsize=8.5, color=MUTED)
    save(fig, 'fig3_gating')


def fig_pipeline():
    fig, ax = canvas(14, 4.6)
    ax.text(.01, .93, 'Training and evaluation pipeline (one seed; 5 seeds in parallel)', fontsize=14, fontweight='bold')
    stages = [
        ('1  Classical\nopponents', 'tune PID / pulse law /\nset-point schedule\nper task (8 cases)', C_CLASSIC, E_CLASSIC),
        ('2  Imitation', 'surrogate-gradient BPTT\non opponent traces\n40 epochs, ~15 min', C_LEARN, E_LEARN),
        ('3  Add gate', 'gate, 1 s tail, law_tuned,\npulse_tuned (25°),\nrefractory 37 ms', '#fff7e0', '#b58a12'),
        ('4  OpenAI-ES', '2035-D, pop 32, σ 0.03\n8 nod + 4 dash / gen\n60 generations', C_MUSCLE, E_MUSCLE),
        ('5  Select centre', '13 saved centres\n64 + 64 val. cases\n0 crash, tilt < 70°', C_REFLEX, E_REFLEX),
        ('6  Blind test', 'fresh 16 + 16 cases\npaired vs strongest\nclassical + ablations', '#f3e8f7', '#7a52b3'),
        ('7  M1 confirm', 'frozen controllers\nnew blind set, prereg.\n≥ 3 / 5 seeds pass', '#fde8e6', '#c23b3b'),
    ]
    w, gap, y, h = .118, .022, .27, .55
    for k, (title, sub, fc, ec) in enumerate(stages):
        x = .01+k*(w+gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.015', fc=fc, ec=ec, lw=1.6))
        ax.text(x+w/2, y+h-.1, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=ec)
        ax.text(x+w/2, y+.18, sub, ha='center', va='center', fontsize=8.6, color=INK, linespacing=1.4)
        if k < len(stages)-1:
            arrow(ax, (x+w, y+h/2), (x+w+gap, y+h/2), lw=1.6)
    ax.text(.01+1*(w+gap)+w/2, .15, 'labels: when / which pulse,\nset-point', ha='center', fontsize=8.3, color=MUTED)
    ax.text(.01+3*(w+gap)+w/2, .15, 'objective: score ratio vs\nclassical + tilt penalty', ha='center', fontsize=8.3, color=MUTED)
    ax.text(.01+5*(w+gap)+w/2, .15, 'ratio = learned mean /\nopponent per-case median', ha='center', fontsize=8.3, color=MUTED)
    ax.text(.01, .04, 'Stage-1 opponents are both the teacher in stage 2 and the yardstick (normalisers N_t) in stages 4–7.',
            fontsize=9, color=E_CLASSIC)
    save(fig, 'fig4_pipeline')


def fig_imitation():
    fig, ax = canvas(13, 4.4)
    ax.text(.01, .93, 'Step 1 — imitation: learn when to fire (and which way) from the strongest classical controller',
            fontsize=13, fontweight='bold')
    box(ax, .01, .30, .16, .45, 'Classical opponent\nflies training cases', fc=C_CLASSIC, ec=E_CLASSIC, fs=10,
        sub='\n\nhover 24 · nod 16\ndash 16 · upset 16')
    arrow(ax, (.17, .525), (.21, .525), 'log every tick (t ≥ 2 s)', toff=(0, .24))
    box(ax, .21, .30, .20, .45, '', fc='white')
    ax.text(.31, .69, 'dataset per tick', ha='center', fontsize=10.5, fontweight='bold')
    ax.text(.225, .48, 'x(t): 20-D features\ny_up = fire ∧ u > 0\ny_down = fire ∧ u < 0\nr = ref / 0.7\n(fire rate ≈ 1–2 %)',
            fontsize=9.3, va='center', linespacing=1.5)
    arrow(ax, (.41, .525), (.45, .525), 'windows of 200 ticks', toff=(0, .24))
    box(ax, .45, .30, .20, .45, '', fc=C_LEARN, ec=E_LEARN)
    ax.text(.55, .69, 'SNN unrolled (BPTT)', ha='center', fontsize=10.5, fontweight='bold', color=E_LEARN)
    ax.text(.465, .48, 'forward: Heaviside spikes\nbackward: σ′((v−1)/0.25)\np = max over substeps\n'
            'of σ((v−1)/0.25)', fontsize=9.3, va='center', linespacing=1.5)
    arrow(ax, (.65, .525), (.69, .525))
    box(ax, .69, .30, .29, .45, '', fc='#f7f8fa', ec=GRIDC)
    ax.text(.835, .69, 'loss and optimiser', ha='center', fontsize=10.5, fontweight='bold')
    ax.text(.70, .53, '$L=\\overline{(1+2y)\\,\\mathrm{BCE}(p,y)}+5\\,\\overline{(ref/0.7-r)^2}$',
            fontsize=10.5, va='center')
    ax.text(.70, .39, 'Adam 3e-3 · grad-norm clip 1 · batch 32\n40 epochs · best validation epoch\n'
            'τ_m, τ_s learned (log-param., 0.3–10 ms)', fontsize=9, va='center', linespacing=1.5)
    ax.text(.01, .12, 'Why: from scratch, NEAT / RL never found phase-locked firing (wrong basin, flat objective). Imitation '
            'gives a controller that already flies with correctly phased pulses (val. precision ≈ 0.45, recall ≈ 0.65–0.70);\n'
            'it need not copy the teacher exactly — step 2 improves on it.', fontsize=9, color=MUTED, linespacing=1.5)
    save(fig, 'fig5_imitation')


def fig_es():
    fig, ax = canvas(13, 5.2)
    ax.text(.01, .94, 'Step 2 — OpenAI-ES: one generation', fontsize=13, fontweight='bold')
    box(ax, .01, .45, .12, .30, 'centre θ', fc=C_MUSCLE, ec=E_MUSCLE, bold=True, fs=11, sub='\n2035-D')
    arrow(ax, (.13, .60), (.17, .60))
    box(ax, .17, .40, .17, .40, '', fc='white')
    ax.text(.255, .74, 'antithetic population', ha='center', fontsize=10, fontweight='bold')
    ax.text(.255, .58, 'ε_j ~ N(0, I), j = 1..16\nθ ± 0.03·ε_j  →  32\n+ centre (logged)', ha='center', va='center',
            fontsize=9.3, linespacing=1.5)
    arrow(ax, (.34, .60), (.38, .60))
    box(ax, .38, .40, .20, .40, '', fc=C_LEARN, ec=E_LEARN)
    ax.text(.48, .74, 'closed-loop rollouts', ha='center', fontsize=10, fontweight='bold', color=E_LEARN)
    ax.text(.48, .575, 'same fresh batch for all\n(common random numbers)\n8 nod + 4 dash stress cases\n'
            'offset 100 000 + 100·gen', ha='center', va='center', fontsize=9.2, linespacing=1.5)
    arrow(ax, (.58, .60), (.62, .60))
    box(ax, .62, .40, .20, .40, '', fc='#f7f8fa', ec=GRIDC)
    ax.text(.72, .74, 'objective (lower = better)', ha='center', fontsize=10, fontweight='bold')
    ax.text(.63, .575, '$F=\\frac{1}{3}\\sum_t w_t\\left(\\frac{\\bar J_t}{N_t}+0.3\\,\\overline{[\\theta_{peak}-60°]_+}\\right)$',
            fontsize=9.5, va='center')
    ax.text(.72, .46, 'w_nod = 2, w_dash = 1\nN_t: classical normaliser', ha='center', fontsize=8.8, color=MUTED)
    arrow(ax, (.82, .60), (.86, .60))
    box(ax, .86, .40, .13, .40, '', fc='white')
    ax.text(.925, .74, 'update', ha='center', fontsize=10, fontweight='bold')
    ax.text(.925, .58, 'rank utility\n$u_j=[\\ln 17-\\ln(r_j{+}1)]_+$\n$\\hat g=\\frac{1}{σ}\\sum u_j ε_j$\nAdam, lr 0.02',
            ha='center', va='center', fontsize=9.2, linespacing=1.6)
    ax.annotate('', xy=(.07, .45), xytext=(.925, .40), arrowprops=dict(arrowstyle='-|>', color=E_MUSCLE, lw=1.6,
                                                                       connectionstyle='arc3,rad=-.25'))
    ax.text(.50, .08, 'θ ← θ + Adam(ĝ)   ·   every 5 generations the centre is saved (13 candidates over 60 generations)',
            ha='center', fontsize=9.5, color=E_MUSCLE)
    ax.text(.01, .015, 'Tilt penalty uses the simulator’s true tilt as a training signal only; the controller still sees '
            'only the IMU. Hover and upset are not trained (gate makes them identical to classical).',
            fontsize=8.5, color=MUTED)
    save(fig, 'fig6_es')


def fig_es_curves():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axs[0]
    for s in SEEDS:
        h = [json.loads(l) for l in (R5/f'finetune_s{s}/history.jsonl').read_text().splitlines() if l.strip()]
        g = np.array([r['generation'] for r in h])
        c = np.array([r['center'] for r in h])
        k = 5
        sm = np.array([np.median(c[i-k+1:i+1]) for i in range(k-1, len(c))])
        ax.plot(g, c, color=SEED_COL[s], alpha=.18, lw=.8)
        ax.plot(g[k-1:], sm, color=SEED_COL[s], lw=1.8, label=f'seed {s}')
    ax.set_xlabel('ES generation')
    ax.set_ylabel('training objective F of the centre\n(12 fresh cases / gen; lower = better)')
    ax.set_title('a  ES training (5-gen running median; faint = raw)', loc='left', fontweight='bold', fontsize=11)
    ax.axhline(1., color=MUTED, lw=.8, ls='--')
    ax.text(59, 1.005, 'classical = 1', ha='right', va='bottom', fontsize=8.5, color=MUTED)
    ax.set_ylim(.6, 1.6)
    ax.legend(frameon=False, fontsize=8.5, ncol=5, loc='upper left')
    ax = axs[1]
    for s in SEEDS:
        d = json.loads((R5/f'claude_ea/reselect_r5_s{s}.json').read_text())
        cs = sorted(d['centres'], key=lambda c: c['generation'])
        g = np.array([max(0, c['generation']) for c in cs])
        v = np.array([c['penalised16'] for c in cs])
        ok = np.array([c['crashes'] == 0 and c['over_hard'] == 0 for c in cs])
        ax.plot(g, np.minimum(v, 1.6), color=SEED_COL[s], lw=1.2, alpha=.8)
        ax.scatter(g[ok], np.minimum(v[ok], 1.6), s=26, color=SEED_COL[s], zorder=3)
        ax.scatter(g[~ok], np.minimum(v[~ok], 1.6), s=30, marker='x', color=SEED_COL[s], zorder=3)
        sel = d['selected']
        if sel is not None:
            ax.scatter([max(0, sel['generation'])], [sel['penalised16']], s=170, facecolors='none', edgecolors=SEED_COL[s],
                       lw=2, zorder=4)
    ax.axhline(1., color=MUTED, lw=.8, ls='--')
    ax.set_xlabel('ES generation of saved centre (0 = imitation start)')
    ax.set_ylabel('selection score on 64 + 64 val. cases\n(penalised; clipped at 1.6)')
    ax.set_title('b  Centre selection: ● safe  ✕ crash or tilt ≥ 70°  ○ selected', loc='left', fontweight='bold', fontsize=11)
    ax.set_ylim(.8, 1.65)
    fig.tight_layout()
    save(fig, 'fig7_es_curves')


def fig_evaluation():
    fig = plt.figure(figsize=(13, 4.6))
    ax = fig.add_axes([0, 0, .52, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.02, .93, 'Evaluation: paired against the strongest re-tuned classical controller', fontsize=12, fontweight='bold')
    box(ax, .02, .55, .28, .28, 'Learned controller', fc=C_LEARN, ec=E_LEARN, bold=True, sub='\nfrozen after selection')
    box(ax, .02, .17, .28, .28, 'Strongest classical', fc=C_CLASSIC, ec=E_CLASSIC, bold=True,
        sub='\n\nnod / dash: set-point schedule\nhover: law_tuned\nupset: pulse_tuned', fs=9.5)
    box(ax, .38, .30, .22, .40, 'Fresh blind set', fc='white', bold=True, sub='\n\nnod 16 · dash 16\nhover 16 · upset 48\nsame cases\nfor both')
    arrow(ax, (.30, .69), (.38, .58))
    arrow(ax, (.30, .31), (.38, .42))
    box(ax, .68, .30, .30, .40, '', fc='#f7f8fa', ec=GRIDC)
    ax.text(.83, .63, 'ratio per task', ha='center', fontsize=10.5, fontweight='bold')
    ax.text(.83, .50, '$\\frac{\\mathrm{mean}\\;J_{learned}}{\\mathrm{median}\\;J_{classical}}$', ha='center', fontsize=14)
    ax.text(.83, .37, '< 1 = better than classical', ha='center', fontsize=9, color=MUTED)
    arrow(ax, (.60, .50), (.68, .50))
    ax.text(.02, .05, 'Also: 4 ablations (clamp_fast, nodelay_all, noref, nopulse) and a same-size ANN.', fontsize=9, color=MUTED)
    ax2 = fig.add_axes([.58, .14, .40, .72])
    m1 = {4: (.911, .870), 1: (.951, .846), 2: (.934, .965), 3: (.973, 1.009), 5: (1.064, .930)}
    order = [4, 1, 2, 3, 5]
    xs = np.arange(len(order))
    ax2.bar(xs-.2, [m1[s][0] for s in order], .38, color='#7aa7d9', label='nod ratio')
    ax2.bar(xs+.2, [m1[s][1] for s in order], .38, color='#f0b27a', label='dash ratio')
    mean = [(m1[s][0]+m1[s][1])/2 for s in order]
    ax2.scatter(xs, mean, color=INK, zorder=3, s=30, label='manoeuvre mean')
    ax2.axhline(.95, color='#c23b3b', lw=1.2, ls='--')
    ax2.text(4.45, .952, 'M1 threshold 0.95', color='#c23b3b', ha='right', va='bottom', fontsize=8.5)
    ax2.axhline(1., color=MUTED, lw=.8)
    ax2.set_xticks(xs, [f'seed {s}\n{"PASS" if (m1[s][0]+m1[s][1])/2 <= .95 else "fail"}' for s in order])
    ax2.set_ylim(.8, 1.1)
    ax2.set_ylabel('ratio vs classical (lower = better)')
    ax2.set_title('M1 confirmatory test: 3 / 5 seeds pass (hover 1.000, upset 1.050, 0 crashes)',
                  fontsize=10.5, loc='left', fontweight='bold')
    ax2.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=3)
    save(fig, 'fig8_evaluation')


def fig_trace():
    """Closed-loop trace of one nod episode with the network state (needs MuJoCo)."""
    import os
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from experiments.algorithms.spike_steer import bench as B, confirm_m1 as C
    from experiments.algorithms.spike_steer.render_v3 import record
    from experiments.algorithms.spike_steer.render_net import NetLog
    C.install()
    gen, spec_path = C.specs()[4]
    spec = json.loads(spec_path.read_text())
    baseline = json.loads(Path('artifacts/spike_steer/confirm_m1/claude_ea/baseline.json').read_text())
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        B.prepare_source(source)
    B.worker_init(str(source), baseline['prereg']['dash_m'] or .25)
    c = B.cases('nod', 'test', None)[1]
    with NetLog() as log:
        res, _ = record(B._ENV, 'nod', c, 'learned', spec)
    tr = np.array(res['trace'])
    n = min(len(tr), len(log.rows))
    tr, net = tr[-n:], log.rows[-n:]
    t = tr[:, 0]
    cmd = tr[:, 11:13].max(1) > .5
    gate = np.zeros(n, bool)
    off = None
    for k in range(n):
        if k and cmd[k-1] and not cmd[k]:
            off = t[k]
        gate[k] = cmd[k] or (off is not None and t[k]-off < 1.)
    hs = np.array([r['hs'] for r in net])
    vh = np.array([r['vh'] for r in net])
    fo = np.array([r['fo'] for r in net])
    xs = np.array([r['x'] for r in net])
    ref = np.degrees([r['ref'] for r in net])
    fire = tr[:, 10] > .5
    w = (t >= 2.1) & (t <= 4.1)
    fig, axs = plt.subplots(5, 1, figsize=(12, 9.2), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1.2, .9, 1.6, .7]))
    a = axs[0]
    a.fill_between(t[w], 0, 1, where=cmd[w], color=E_LEARN, alpha=.25, step='post', label='nod command')
    a.fill_between(t[w], 0, 1, where=gate[w] & ~cmd[w], color=E_MUSCLE, alpha=.25, step='post', label='1 s tail (learned still in charge)')
    a.plot(t[w], xs[w, 3]+0, color=E_REFLEX, lw=.8, label='input gy+ (wingbeat oscillation)')
    a.set_yticks([])
    a.set_ylim(0, xs[w, 3].max()*1.35)
    a.legend(frameon=False, fontsize=8, ncol=3, loc='upper right')
    a.set_title('Closed-loop run of the learned layer — nod 30°, round-5 seed 4, M1 test case 1 (every value logged per 2 ms tick)',
                loc='left', fontweight='bold', fontsize=11)
    a = axs[1]
    a.plot(t[w], np.degrees(tr[w, 1]), color=INK, lw=1.6, label='pitch (true)')
    a.plot(t[w], ref[w], color=E_REFLEX, lw=1.4, ls='--', label='ref = 0.7·tanh(g) (to PID)')
    a.axhline(30, color=E_LEARN, lw=.8, ls=':')
    a.axhline(-30, color=E_LEARN, lw=.8, ls=':')
    a.set_ylabel('deg')
    a.legend(frameon=False, fontsize=8.5, loc='lower right', ncol=2)
    a = axs[2]
    for h in (0, 2, 7):
        a.plot(t[w], vh[w, h], lw=.8, label=f'v(h{h+1})')
    a.axhline(1., color='#c23b3b', lw=.8, ls='--')
    a.set_ylabel('membrane')
    a.legend(frameon=False, fontsize=8, ncol=3, loc='lower right')
    a = axs[3]
    for h in range(8):
        tt = t[w][hs[w][:, h]]
        a.vlines(tt, h+.6, h+1.4, color=INK, lw=.6)
    for o in range(2):
        tt = t[w][fo[w][:, o]]
        a.vlines(tt, 9+o-.4+.6, 9+o+.4+.6, color='#c23b3b', lw=1.2)
    a.set_yticks(list(range(1, 9))+[9.6, 10.6], [f'h{h}' for h in range(1, 9)]+['up', 'down'], fontsize=8)
    a.set_ylabel('spikes')
    a = axs[4]
    a.vlines(t[w][fire[w] & gate[w]], 0, 1, color=E_MUSCLE, lw=1, label='motor pulse (learned layer)')
    a.vlines(t[w][fire[w] & ~gate[w]], 0, 1, color=E_CLASSIC, lw=1, label='motor pulse (classical law)')
    a.set_yticks([])
    a.legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
    a.set_xlabel('time (s)')
    for a in axs:
        a.axvline(2.4, color=E_LEARN, lw=.8)
    fig.align_ylabels(axs)
    fig.tight_layout()
    save(fig, 'fig9_closed_loop_trace')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sim', action='store_true')
    a = p.parse_args()
    spec = json.loads((ROOT/'artifacts/spike_steer/plan_v4/r5/train/imitate_spec_s4.json').read_text())
    sel = json.loads((R5/'claude_ea/reselect_r5_s4.json').read_text())['selected']
    sel_spec = ROOT/sel['spec']
    spec_final = json.loads(sel_spec.read_text()) if sel_spec.exists() else spec
    fig_architecture()
    fig_snn(spec_final)
    fig_gating()
    fig_pipeline()
    fig_imitation()
    fig_es()
    fig_es_curves()
    fig_evaluation()
    if a.sim:
        fig_trace()


if __name__ == '__main__':
    main()
