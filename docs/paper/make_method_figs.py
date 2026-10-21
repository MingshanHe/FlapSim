"""Method figures for docs/paper/draft_method_results.md (plans v9-v11: evade / redirect / dash).

    PYTHONPATH=/root/flapsim .venv/bin/python docs/paper/make_method_figs.py

Outputs (PNG, 170 dpi) in docs/paper/figs/:
  fig_m1_architecture.png  the three-layer controller in one 2 ms tick
  fig_m2_snn.png           the descending SNN with the edge weights of the selected plan-v11 centre of seed 3
  fig_m4_pipeline.png      training / selection / blind-test / robustness pipeline with the plan-v11 numbers
  fig_m5_tasks.png         the three tasks in the heading frame + the classical normalisers (plan-v9 baseline.json)
fig_m3_gating.png is docs/figs/fig3_gating.png copied unchanged (the gating timeline did not change after plan v6).

Diagrams are drawn from the method description; no simulation. Drawing helpers (box / arrow / canvas / save) are adapted
from docs/figs/make_algorithm_figs.py, which draws the plan-v6 (nod / dash) versions.
"""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                               # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/paper/figs'
SPEC = ROOT/'artifacts/spike_steer/plan_v11/snn/s3/selected.json'
BASELINE = ROOT/'artifacts/spike_steer/plan_v9/baseline.json'
DPI = 170
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': DPI, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})

INK, MUTED, GRIDC = '#1e252e', '#6b7380', '#d9dde3'
C_REFLEX, C_MUSCLE, C_LEARN, C_CLASSIC = '#dbe8f6', '#fde3c8', '#d9efdf', '#eceef1'
E_REFLEX, E_MUSCLE, E_LEARN, E_CLASSIC = '#2f6db5', '#d4781f', '#1f8a4c', '#8a929e'
C_GATE, E_GATE = '#fff7e0', '#b58a12'
C_BLIND, E_BLIND = '#f3e8f7', '#7a52b3'
C_ROBUST, E_ROBUST = '#fde8e6', '#c23b3b'
CLS_COL = dict(pitch=E_REFLEX, roll=E_LEARN, diag=E_MUSCLE)      # direction classes, as in docs/figs/fig10_directions.png


# ------------------------------------------------------------------------------------------------------- helpers
def box(ax, x, y, w, h, text, fc='white', ec=INK, fs=7, bold=False, lw=1.0, r=.012, sub=None, subfs=None, tdy=.02, sdy=-.028,
        ls=1.3):
    """Rounded box with a title and an optional muted sub-text (title shifted up by tdy, sub-text centred at y+h/2+sdy)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))
    ty = y+h/2+(tdy if sub else 0)
    ax.text(x+w/2, ty, text, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal',
            linespacing=ls)
    if sub:
        ax.text(x+w/2, y+h/2+sdy, sub, ha='center', va='center', fontsize=subfs or fs-1.2, color=MUTED, linespacing=ls)
    return (x, y, w, h)


def arrow(ax, p0, p1, text=None, col=INK, lw=1.0, style='-|>', ls='-', fs=6, toff=(0, .01), rad=0., ha='center', va='bottom'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8, color=col, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))
    if text:
        ax.text((p0[0]+p1[0])/2+toff[0], (p0[1]+p1[1])/2+toff[1], text, ha=ha, va=va, fontsize=fs, color=col)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f'{name}.png', dpi=DPI)
    plt.close(fig)
    print('wrote', OUT/f'{name}.png')


# ------------------------------------------------------------------------------------------- M1: architecture
def fig_architecture():
    """Three-layer controller in one 2 ms tick (layout of the plan-v6 fig1_architecture, evade / redirect / dash labels)."""
    fig, ax = canvas(7.2, 4.4)
    ax.text(.99, .975, 'one 2 ms control tick (500 Hz)', ha='right', va='top', fontsize=6.5, color=MUTED, style='italic')
    # sensors / command
    box(ax, .01, .64, .135, .14, 'IMU + ToF', sub='11-D reading\n500 Hz, noisy', tdy=.025, sdy=-.022)
    box(ax, .01, .33, .135, .18, 'Task command', sub='evade channel or\ndash/redirect channel\n(unit vector along φ)',
        tdy=.045, sdy=-.022, subfs=5.6)
    # descending layer (learned)
    ax.add_patch(FancyBboxPatch((.165, .53), .405, .40, boxstyle='round,pad=0,rounding_size=.015', fc=C_LEARN, ec=E_LEARN, lw=1.3))
    ax.text(.175, .905, 'Descending layer (learned SNN)', fontsize=8, fontweight='bold', color=E_LEARN, va='center')
    box(ax, .18, .60, .11, .22, 'features', sub='25-D: 16 rectified\nIMU channels,\nbias, 8 command\nchannels', tdy=.065, sdy=-.028)
    box(ax, .305, .60, .095, .22, 'delay lines', sub='11 taps\n0–20 ms', tdy=.03, sdy=-.025)
    box(ax, .415, .74, .14, .10, 'LIF hidden layer', sub='8 neurons, 2 sub-steps', tdy=.015, sdy=-.02)
    box(ax, .415, .56, .135, .15, 'outputs', sub='via 11 axonal delays\n0–20 ms: 2 spiking →\npulse, 2 graded g_r, g_p', tdy=.045,
        sdy=-.025, subfs=5.3)
    arrow(ax, (.29, .71), (.305, .71))
    arrow(ax, (.40, .79), (.415, .79))
    arrow(ax, (.4825, .74), (.4825, .71))
    arrow(ax, (.145, .71), (.18, .72))
    arrow(ax, (.145, .42), (.18, .66), rad=-.2)
    # command gate
    ax.add_patch(FancyBboxPatch((.60, .575), .175, .29, boxstyle='round,pad=0,rounding_size=.015', fc=C_GATE, ec=E_GATE, lw=1.1))
    ax.text(.6875, .83, 'Command gate', ha='center', fontsize=7.5, fontweight='bold')
    ax.text(.6875, .70, 'learned pulse and (g_r, g_p)\nin charge while the\ncommand is on + 1 s tail;\notherwise the classical\nbase law, set-points = 0',
            ha='center', va='center', fontsize=5.9, color=MUTED, linespacing=1.35)
    arrow(ax, (.55, .635), (.60, .66), col=E_LEARN)
    # classical base law
    box(ax, .18, .33, .12, .14, 'PLL', fc=C_CLASSIC, ec=E_CLASSIC, sub='wingbeat phase\nfrom the pitch gyro', tdy=.03, sdy=-.022)
    box(ax, .335, .32, .22, .16, 'Classical base pulse law', fc=C_CLASSIC, ec=E_CLASSIC,
        sub='phase-locked: k_θ = 8, k_q = 0.4\nrighting reflex (tilt > 25°)\nonly outside the gate', tdy=.045, sdy=-.022)
    arrow(ax, (.145, .665), (.18, .43), col=E_CLASSIC, rad=.15)
    arrow(ax, (.30, .40), (.335, .40), col=E_CLASSIC)
    arrow(ax, (.555, .42), (.64, .575), col=E_CLASSIC)
    ax.text(.598, .50, 'pulse_law, set-points = (0, 0)', fontsize=5.8, color=E_CLASSIC, va='center', ha='right')
    # refractory
    box(ax, .81, .705, .18, .13, 'Refractory 37 ms', fc=C_GATE, ec=E_GATE, sub='≤ 1 learned pulse onset\nper wingbeat', tdy=.03,
        sdy=-.025, subfs=5.6)
    arrow(ax, (.775, .77), (.81, .77), 'pulse', toff=(0, .005))
    # reflex layer
    ax.add_patch(FancyBboxPatch((.60, .12), .175, .30, boxstyle='round,pad=0,rounding_size=.015', fc=C_REFLEX, ec=E_REFLEX, lw=1.3))
    ax.text(.6875, .385, 'Reflex layer (PID)', fontsize=7.5, fontweight='bold', color=E_REFLEX, ha='center')
    ax.text(.6875, .255, 'attitude: α, β ← PID(e)\ne = (g_r, g_p) − (roll, pitch)\naltitude: ω ← PID(z, v_z)',
            fontsize=5.5, color=INK, ha='center', va='center', linespacing=1.5)
    arrow(ax, (.6875, .575), (.6875, .42), col=E_REFLEX)
    ax.text(.697, .50, 'set-points\n(roll, pitch)', fontsize=6, color=E_REFLEX, va='center', ha='left', linespacing=1.3)
    ax.plot([.145, .16, .16], [.67, .67, .25], color=E_REFLEX, lw=1.0)
    arrow(ax, (.16, .25), (.60, .25), 'IMU reading', col=E_REFLEX, toff=(-.12, .006))
    # steering-muscle layer
    ax.add_patch(FancyBboxPatch((.81, .40), .18, .245, boxstyle='round,pad=0,rounding_size=.015', fc=C_MUSCLE, ec=E_MUSCLE, lw=1.3))
    ax.text(.90, .615, 'Steering-muscle layer', fontsize=7, fontweight='bold', color=E_MUSCLE, ha='center')
    ax.text(.90, .505, 'pulse → ω = 3807 rad/s\nfor 3 ticks (6 ms),\noverrides the PID ω;\n≈ +690 rad/s after the lag\n→ pitch torque, sign set\nby the wingbeat phase',
            fontsize=6, ha='center', va='center', linespacing=1.35)
    arrow(ax, (.90, .715), (.90, .645), col=E_MUSCLE)
    # actuators
    box(ax, .81, .12, .18, .16, 'Actuators', sub='α, β ±0.3 rad\nω 3115–3807 rad/s\n8.2 ms first-order lag', tdy=.045, sdy=-.022)
    arrow(ax, (.775, .20), (.81, .20), col=E_REFLEX)
    ax.text(.7925, .215, 'α β ω', ha='center', va='bottom', fontsize=5.5, color=E_REFLEX)
    arrow(ax, (.90, .40), (.90, .28), col=E_MUSCLE)
    ax.text(.91, .34, 'ω pulse', fontsize=6, color=E_MUSCLE, va='center', ha='left')
    ax.text(.01, .055, 'Fixed interface: sensors, actuators, the 2 ms update and the limits are those of the original environment. '
            'A pulse gives pitch torque only (nose-up or nose-down set by its timing in the\nwingbeat); roll manoeuvres go through the '
            'roll set-point g_r → PID → gimbal. Outside the gate every arm is tick-for-tick identical to the classical base law.',
            fontsize=5.8, color=MUTED, va='center', linespacing=1.4)
    save(fig, 'fig_m1_architecture')


# ----------------------------------------------------------------------------------------------------- M2: SNN
def fig_snn(spec):
    """Network graph with the actual (delay-summed) weights of a spec, LIF equations and the parameter count."""
    tau_m, tau_s, tau_g = spec['tau_m']*1e3, spec['tau_s']*1e3, spec['tau_g']*1e3
    print(f"time constants of {SPEC.relative_to(ROOT)}: tau_m = {tau_m:.3f} ms, tau_s = {tau_s:.3f} ms, tau_g = {tau_g:.1f} ms, "
          f"spike_scale = {spec['spike_scale']}, generation = {spec.get('generation')}")
    W_in, W_out, W_g = (np.asarray(spec[k], float) for k in ('W_in', 'W_out', 'W_g'))
    n_par = sum(np.asarray(spec[k], float).size for k in ('W_in', 'b_h', 'W_out', 'b_o', 'W_g', 'b_g'))
    assert n_par == 2564, n_par
    W, H = 7.2, 4.5
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, .60, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    asp = W*.60/H

    def circ(x, y, r, **kw):
        ax.add_patch(Ellipse((x, y), 2*r, 2*r*asp, **kw))
    names = ['roll', 'pitch', 'g$_x$', 'g$_y$', 'g$_z$', 'a$_x$', 'a$_y$', 'a$_z$']
    cmdn = ['evade·cos φ', 'evade·sin φ', 'dash·cos φ', 'dash·sin φ']
    labels = [f'{n}+' for n in names]+[f'{n}−' for n in names]+['bias']+[f'{c}+' for c in cmdn]+[f'{c}−' for c in cmdn]
    n_in = len(labels)
    assert n_in == W_in.shape[1] == 25
    ys = np.linspace(.905, .13, n_in)
    hy = np.linspace(.80, .24, 8)
    oy = dict(up=.70, down=.555, g_r=.38, g_p=.25)
    xi, xh, xo = .135, .54, .78
    ax.text(.02, .975, 'inputs (25)', fontsize=6.5, color=MUTED, va='center')
    ax.text(xi+.07, .945, 'delay taps 0, 2, …, 20 ms (11)', fontsize=6, color=MUTED, va='center', ha='center')
    ax.text(xh, .975, 'LIF hidden (8)', fontsize=6.5, color=MUTED, va='center', ha='center')
    ax.text(xo, .975, 'outputs', fontsize=6.5, color=MUTED, va='center', ha='center')
    for i, (lab, y) in enumerate(zip(labels, ys)):
        col = E_REFLEX if i < 16 else (E_LEARN if i >= 17 else MUTED)
        circ(xi, y, .007, fc='white', ec=col, lw=1.0)
        ax.text(xi-.015, y, lab, ha='right', va='center', fontsize=5.6, color=col)
        for k in range(11):                                   # delay taps (schematic)
            ax.plot([xi+.014+k*.0105], [y], 's', ms=1.4, color=col, alpha=.55)
    # input -> hidden edges: weights summed over the 11 delays
    wsum = W_in.sum(2)
    wmax = np.abs(wsum).max()
    for h in range(8):
        for i in range(n_in):
            w = wsum[h, i]
            if abs(w) < .25*wmax:
                continue
            ax.plot([xi+.135, xh-.022], [ys[i], hy[h]], color=E_MUSCLE if w > 0 else E_REFLEX, lw=.3+1.6*abs(w)/wmax, alpha=.55,
                    solid_capstyle='round')
    for h, y in enumerate(hy):
        circ(xh, y, .021, fc='#fff4e4', ec=INK, lw=1.0)
        ax.text(xh, y, f'h{h+1}', ha='center', va='center', fontsize=6.3)
    # hidden -> outputs: W_out (spiking) and W_g (graded), summed over the 11 axonal delays
    osum = np.r_[W_out.sum(2), W_g.sum(2)]                     # 4 x 8
    omax = np.abs(osum).max()
    for o, key in enumerate(('up', 'down', 'g_r', 'g_p')):
        for h in range(8):
            w = osum[o, h]
            if abs(w) < .10*omax:
                continue
            ax.plot([xh+.022, xo-.027], [hy[h], oy[key]], color=E_MUSCLE if w > 0 else E_REFLEX, lw=.3+1.6*abs(w)/omax, alpha=.55,
                    solid_capstyle='round')
    for lab in ('up', 'down'):
        circ(xo, oy[lab], .027, fc=C_ROBUST, ec=E_ROBUST, lw=1.2)
        ax.text(xo, oy[lab], lab, ha='center', va='center', fontsize=6.5)
    for lab, txt in (('g_r', '$g_r$'), ('g_p', '$g_p$')):
        ax.add_patch(Rectangle((xo-.027, oy[lab]-.027*asp), .054, .054*asp, fc=C_REFLEX, ec=E_REFLEX, lw=1.2))
        ax.text(xo, oy[lab], txt, ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax.text((xh+xo)/2, .905, 'hidden → outputs: 11 axonal\ndelays 0–20 ms, drive × 5', fontsize=6, color=MUTED, ha='center', va='center',
            linespacing=1.3)
    ax.text(xo+.04, .63, 'spiking (LIF):\neither spike →\none motor pulse\n(same twitch;\nthe timing\ncarries the\nmeaning)', fontsize=5.6,
            va='center', linespacing=1.3)
    ax.text(xo+.04, .315, 'graded units:\nτ$_g$ = 20 ms\nref = 0.7·tanh(g)\n→ roll / pitch\nPID set-points', fontsize=5.6, va='center',
            linespacing=1.3)
    fig.text(.012, .012, f'Edges: weights summed over the 11 delays, orange +, blue −, width ∝ |w|; input→hidden drawn for |w| ≥ 25 % of max, '
             f'hidden→output for |w| ≥ 10 % of max. The dash channel is shared by dash and\nredirect (direction of the current leg). '
             f'Weights: the selected centre of plan v11, seed 3 (ES generation {spec.get("generation")}).',
             fontsize=5.5, color=MUTED, va='bottom', linespacing=1.4)
    # right: LIF equations, time constants, parameter table
    ax2 = fig.add_axes([.60, 0, .40, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.add_patch(FancyBboxPatch((.03, .505), .95, .46, boxstyle='round,pad=0,rounding_size=.02', fc='#f7f8fa', ec=GRIDC))
    ax2.text(.07, .935, 'LIF dynamics (2 × 1 ms sub-steps per 2 ms tick)', fontsize=7.2, fontweight='bold', va='center')
    eq = ('$I_h=\\sum_i\\sum_k W^{in}_{h,i,k}\\,x_i(t-2k\\,\\mathrm{ms})+b_h$\n'
          '$c_h\\leftarrow a_s\\,c_h+(1-a_s)\\,I_h$\n'
          '$v_h\\leftarrow a_m\\,v_h+(1-a_m)\\,c_h$\n'
          '$s_h=[v_h\\geq 1],\\;\\; v_h\\leftarrow v_h-s_h$  (subtractive reset)\n'
          '$I_o=5\\sum_h\\sum_d W^{out}_{o,h,d}\\,s_h(t-d)+b_o$ → same LIF → spike\n'
          '$I_g=5\\sum_h\\sum_d W^{g}_{g,h,d}\\,s_h(t-d)+b_g$, $c_g$ as $c_h$,\n'
          '$g\\leftarrow a_g\\,g+(1-a_g)\\,c_g$,  $ref=0.7\\tanh(g)$\n'
          '$a_{m,s,g}=\\exp(-1\\,\\mathrm{ms}/\\tau_{m,s,g})$')
    ax2.text(.07, .90, eq, fontsize=6.4, va='top', linespacing=1.95)
    ax2.text(.07, .535, f'this centre: τ$_m$ = {tau_m:.2f} ms, τ$_s$ = {tau_s:.2f} ms (learned in\nimitation, fixed in ES); '
             f'τ$_g$ = {tau_g:.0f} ms fixed', fontsize=6.2, color=MUTED, va='center', linespacing=1.35)
    rows = [('W_in', '8 × 25 × 11', '2200'), ('b_h', '8', '8'), ('W_out', '2 × 8 × 11', '176'), ('b_o', '2', '2'),
            ('W_g', '2 × 8 × 11', '176'), ('b_g', '2', '2'), ('total', '', f'{n_par}')]
    ax2.text(.07, .455, 'parameters (searched by ES)', fontsize=7.2, fontweight='bold', va='center')
    for r, (a, b, c) in enumerate(rows):
        y = .405-r*.042
        ax2.text(.09, y, a, fontsize=6.4, fontweight='bold' if a == 'total' else 'normal', va='center')
        ax2.text(.45, y, b, fontsize=6.4, color=MUTED, va='center')
        ax2.text(.90, y, c, fontsize=6.4, ha='right', fontweight='bold' if a == 'total' else 'normal', va='center')
    ax2.plot([.08, .92], [.405-6*.042+.021]*2, color=GRIDC, lw=.8)
    save(fig, 'fig_m2_snn')


# ------------------------------------------------------------------------------------------------ M4: pipeline
def fig_pipeline():
    fig, ax = canvas(7.2, 3.15)
    stages = [
        ('1  Opponent', 'classical: 2-D set-point\nschedule per task and\ndirection class (7)\nevade, redirect:\n24 configs × 8\ntuning cases\n'
         'dash: 108 configs\n× 16 tuning cases', C_CLASSIC, E_CLASSIC),
        ('2  Imitation', 'BPTT with a surrogate\ngradient on the\nopponent\'s traces\n16 training +\n4 validation cases\nper task\n'
         '40 epochs, best\nvalidation epoch\n5 seeds → 5 starts', C_LEARN, E_LEARN),
        ('3  OpenAI-ES', '2564-D, population 16\n(8 antithetic pairs)\nσ 0.02, lr 0.01 · 0.85$^g$\n8 cases / task / gen.\n= 24 episodes\n'
         'per candidate (all\ndirections, turns,\nstrata every gen.)\ncrash-aware ranking\n16 generations\nshared pre-command prefix',
         C_MUSCLE, E_MUSCLE),
        ('4  Selection', 'imitation start +\n9 saved centres\n(odd generations)\n64 fixed validation\ncases per task\nexclude any crash\n'
         'or tilt ≥ 70°\npick the lowest\nobjective\n→ frozen controller', C_REFLEX, E_REFLEX),
        ('5  Blind test', 'fresh seed base\n64 cases per task\npaired with the\nopponent (same\ncases, both arms)\nprereg. pass: ratio ≤ 0.95\n'
         'and crashes ≤ opponent\n(≥ 3 of 5 seeds); paired Δ\nreported alongside', C_BLIND, E_BLIND),
        ('6  Robustness', 'frozen controllers,\nno training (v12–v13):\ntorque hits 2 / 3 / 4.5\nmN·m (timing, axis,\ndouble hit)\n'
         'turbulence × 1.25, × 1.5\nIMU noise × 5\nsensor delay 3 steps\nmass + 12.5 %\npass (H1): paired Δ CI < 0\nand crashes ≤ opponent,\n≥ 3 of 5 seeds', C_ROBUST, E_ROBUST),
    ]
    w, gap, y, h = .151, .0166, .30, .62
    for k, (title, sub, fc, ec) in enumerate(stages):
        x = .007+k*(w+gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.012', fc=fc, ec=ec, lw=1.2))
        ax.text(x+w/2, y+h-.05, title, ha='center', va='center', fontsize=7.2, fontweight='bold', color=ec)
        ax.text(x+w/2, y+h/2-.045, sub, ha='center', va='center', fontsize=5.5, color=INK, linespacing=1.36)
        if k < len(stages)-1:
            arrow(ax, (x+w, y+h/2), (x+w+gap, y+h/2), lw=1.2, ls='--' if k == 4 else '-', col=E_ROBUST if k == 4 else INK)
    notes = {1: 'labels per tick: pulse and\nits sign, set-points',
             2: 'objective: mean case score /\nclassical class mean\n+ 0.3 · [tilt$_{peak}$ − 60°]$_+$',
             4: 'ratio = case score / opponent\nclass median; Δ = ratio −\nopponent\'s own ratio'}
    for k, txt in notes.items():
        ax.text(.007+k*(w+gap)+w/2, .185, txt, ha='center', va='center', fontsize=5.4, color=MUTED, linespacing=1.35)
    ax.text(.007, .045, 'The stage-1 opponent is the teacher in 2, the normaliser in 3–4 and the paired comparator in 5–6. '
            'Stages 1–3 may run on any machine; selection and blind test of a seed run on its reference machine\n'
            '(CloudPC-2: seeds 1–3, HomePC: seeds 4–5; bit-identical episodes). The blind seed base is fixed in the preregistration before '
            'any blind episode; stage 6 flies the stage-4 controllers unchanged.', fontsize=5.6, color=E_CLASSIC, va='center',
            linespacing=1.4)
    save(fig, 'fig_m4_pipeline')


# --------------------------------------------------------------------------------------------------- M5: tasks
def _dir(phi):
    """Heading frame drawn with forward = up, left = left: unit vector of direction phi (deg, 0 forward, 90 left)."""
    a = math.radians(phi)
    return -math.sin(a), math.cos(a)


def fig_tasks(base):
    N = base['normalisers']
    klass = {0: 'pitch', 180: 'pitch', 90: 'roll', 270: 'roll', 45: 'diag', 135: 'diag', 225: 'diag', 315: 'diag'}
    S = 1.0/.15                                               # drawing scale: 0.15 m = 1 unit (same in panels a-c)
    fig = plt.figure(figsize=(7.2, 2.75))
    panels = [('(a) evade: 0.10 m along φ', 'window 1.5 s, t$_0$ ~ U(2.2, 2.8) s\n4 directions, class pitch / roll'),
              ('(b) redirect: φ$_1$, then φ$_1$ ± 90°', 'φ$_1$ ∈ {0, 90, 180, 270}°; switch to φ$_2$ at 0.08 m\nor 1.0 s, '
               'then 0.10 m along φ$_2$; window 4 s,\nt$_0$ = 2.4 s; class by φ$_2$'),
              ('(c) dash: 0.15 m along φ, stop', 'settle |v$_{along}$| < 0.1 m/s for 0.3 s\nwindow 4 s, t$_0$ = 2.4 s; 8 directions')]
    for k, (title, sub) in enumerate(panels):
        ax = fig.add_axes([.005+k*.245, .0, .238, .92])
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.95, 1.45)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, loc='left', fontsize=6.8, fontweight='bold', pad=2)
        ax.add_patch(plt.Circle((0, 0), .11, fc='#fff4e4', ec=INK, lw=1.0))
        ax.annotate('', xy=(0, .30), xytext=(0, .11), arrowprops=dict(arrowstyle='-|>', color=INK, lw=.9, mutation_scale=7))
        if k == 0:
            ax.text(.05, .21, 'heading', fontsize=5.3, color=MUTED)
        if k == 0:                                            # evade
            for phi in (0, 90, 180, 270):
                dx, dy = _dir(phi)
                L = .10*S
                ax.annotate('', xy=(dx*L, dy*L), xytext=(dx*.17, dy*.17),
                            arrowprops=dict(arrowstyle='-|>', color=CLS_COL[klass[phi]], lw=1.8, mutation_scale=10))
                ax.text(dx*(L+.2), dy*(L+.2), f'{phi}°', ha='center', va='center', fontsize=6.2, color=CLS_COL[klass[phi]])
        elif k == 1:                                          # redirect
            L1, L2 = .08*S, .10*S
            for phi1 in (0, 90, 180, 270):
                dx, dy = _dir(phi1)
                ax.annotate('', xy=(dx*L1, dy*L1), xytext=(dx*.17, dy*.17),
                            arrowprops=dict(arrowstyle='-|>', color=E_CLASSIC, lw=1.3, mutation_scale=8))
                for turn in (90, -90):
                    phi2 = (phi1+turn) % 360
                    ex, ey = _dir(phi2)
                    col = CLS_COL[klass[phi2]]
                    ax.annotate('', xy=(dx*L1+ex*L2, dy*L1+ey*L2), xytext=(dx*L1, dy*L1),
                                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8, mutation_scale=10))
                    ax.text(dx*L1+ex*(L2+.19), dy*L1+ey*(L2+.19), f'{phi2}°', ha='center', va='center', fontsize=5.6, color=col)
            ax.text(.07, .36, 'φ$_1$', fontsize=5.6, color=E_CLASSIC, va='center')
            ax.text(-.33, .62, 'φ$_2$', fontsize=5.6, color=CLS_COL['roll'], va='center', ha='center')
            ax.text(0, -1.33, 'grey: leg 1 (φ$_1$); colour: leg 2 (φ$_2$), 8 pairs', ha='center', fontsize=5.2, color=MUTED)
        else:                                                 # dash
            for phi in range(0, 360, 45):
                dx, dy = _dir(phi)
                L = .15*S
                col = CLS_COL[klass[phi]]
                ax.annotate('', xy=(dx*L, dy*L), xytext=(dx*.17, dy*.17),
                            arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8, mutation_scale=10))
                px, py = -dy*.07, dx*.07                      # stop bar at the goal
                ax.plot([dx*(L+.04)-px, dx*(L+.04)+px], [dy*(L+.04)-py, dy*(L+.04)+py], color=col, lw=1.6, solid_capstyle='round')
                ax.text(dx*(L+.28), dy*(L+.28), f'{phi}°', ha='center', va='center', fontsize=6.2, color=col)
        ax.text(0, -1.72, sub, ha='center', va='center', fontsize=5.4, color=MUTED, linespacing=1.35)
    fig.text(.12, .055, 'heading frame at the command onset: 0° forward (up), 90° left; same length scale in a–c',
             fontsize=5.4, color=MUTED, ha='left', va='center')
    # (d) classical normalisers per task and direction class
    ax = fig.add_axes([.785, .22, .205, .63])
    labs, vals, cols, groups = [], [], [], []
    for t in ('evade', 'redirect', 'dash'):
        for cl in ('pitch', 'roll', 'diag'):
            if cl in N[t]:
                labs.append(cl)
                vals.append(N[t][cl])
                cols.append(CLS_COL[cl])
                groups.append(t)
    ax.bar(range(len(vals)), vals, color=cols, width=.62)
    for i, v in enumerate(vals):
        ax.text(i, v+.12, f'{v:.2f}', ha='center', va='bottom', fontsize=5.4, color=INK)
    ax.set_xticks(range(len(vals)), labs, fontsize=5.2, rotation=90)
    ax.tick_params(axis='y', labelsize=6, length=2)
    ax.tick_params(axis='x', length=0, pad=1.5)
    for t in ('evade', 'redirect', 'dash'):                   # task group labels under the class ticks
        idx = [i for i, g in enumerate(groups) if g == t]
        ax.text(np.mean(idx), -1.9, t, ha='center', va='top', fontsize=5.6, color=INK, clip_on=False)
        ax.plot([min(idx)-.4, max(idx)+.4], [-1.75, -1.75], color=GRIDC, lw=.8, clip_on=False)
    ax.set_ylabel('classical opponent score\n(normaliser; lower = better)', fontsize=6)
    ax.set_title('(d) classical normalisers', fontsize=6.8, loc='left', fontweight='bold', pad=4)
    ax.set_ylim(0, 7.4)
    ax.spines['left'].set_color(GRIDC)
    ax.spines['bottom'].set_color(GRIDC)
    handles = [Patch(fc=CLS_COL['pitch'], label='pitch (0°, 180°)'), Patch(fc=CLS_COL['roll'], label='roll (90°, 270°)'),
               Patch(fc=CLS_COL['diag'], label='diagonal (45°, …)')]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=5.2, handlelength=1.1, handleheight=.8, borderaxespad=.2,
              labelspacing=.3)
    save(fig, 'fig_m5_tasks')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC.read_text())
    base = json.loads(BASELINE.read_text())
    fig_architecture()
    fig_snn(spec)
    fig_pipeline()
    fig_tasks(base)
    shutil.copy(ROOT/'docs/figs/fig3_gating.png', OUT/'fig_m3_gating.png')
    print('copied', OUT/'fig_m3_gating.png')


if __name__ == '__main__':
    main()
