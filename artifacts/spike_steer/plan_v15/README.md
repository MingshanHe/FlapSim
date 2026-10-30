# Plan v15: generalisation to changed plants (HomePC, 2026-10-24 19:50 to 10-25 03:05)

Frozen controllers, no training. 192 stress cases (3 tasks x 64, seed base 2_103_000_000), the same seeds on every plant.
Code: experiments/algorithms/spike_steer/blind15.py; raw rows in runs/spike_steer/v15/<plant>/; preregistered criteria in plan.json;
per-plant summaries summary_<plant>.json; table crash_table.md; criteria evaluation criteria_result.txt.

Result against the criteria (G1: >= 3/5 SNN11 seeds with crashes <= classical and degradation <= classical's + 0.05):
- G1 holds on 8 of the 10 flyable plants: wing mass +30 %, span -5 %, chord -10 %, left span -3 %, density -10 %, voltage -10 %,
  kt -10 %, lag x1.5. On the four hardest ones (voltage, lag, wing mass, span -5 %) classical crashes 6-17 vs SNN 0-14 (lag: 15 vs 0-6).
- G1 fails on lower body +25 % and lower +12.5 % with CoM -3 mm: nobody crashes; the classical score improves (its two nominal
  crashes are gone) while the SNN keeps a nominal-like score, so SNN is ~3-7 % worse than classical there (same as v12 'ood').
- Outside the fixed interface's thrust envelope, every controller crashes. For span +5 %, there is too much lift even at LOW 3115 rad/s,
  ending in height_escape or tilt. For feathering -20 % and 'unit B', there is too little lift, ending in low_height at about 1 s, before the manoeuvre starts. Not counted.

## MLP+PPO reference (HomePC, 2026-10-25 09:12-14:26)
ppo_blind14.py --sets v15_<plant> (same cases, same opponent rows, plant applied per worker); rows in ppo_rows/, comparison in ppo_compare.txt.
- G1 for PPO: 5/5 on 8 plants, 5/5 on asym97, 3/5 on both lower-body plants (PPO's nominal score is worse, 1.06-1.27, so its
  degradation on the lower-body plants is negative; in absolute score PPO and SNN are the same there, 0.96-1.02, both a little behind classical 0.92-0.94).
- Total crashes over the 10 flyable plants (1920 episodes): classical 61, cls_es 37/37, SNN11 1/4/10/15/33 (median 10), PPO 5/8/10/12/17 (median 10).
  Both learned layers generalise far better than the classical family; the SNN is not better than PPO on plant changes (same median, wider seed spread).
