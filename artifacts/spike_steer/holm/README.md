# Multiplicity correction for the descriptive robustness sets (2026-10-26, analysis only)
experiments/algorithms/spike_steer/holm.py. The family is 15 descriptive v12/v13 sets; the confirmatory kick and its replication kick_rep are excluded.
For every arm, two paired tests per set against the classical opponent: an exact McNemar test on crash pairs and a paired bootstrap on the score ratio (20 000 resamples).
Holm-Bonferroni is applied within each arm and test across the 15 sets, at alpha 0.05.

- SNN11 seeds: significantly fewer crashes than the opponent on 4 / 9 / 8 / 8 / 7 of 15 sets (s1-s5). The sets are the hit timing / axis / dose sets
  and turbulence x1.25-x2; turb15, turb2, kick_late and kick_pitchaxis survive for (almost) every seed. More crashes: only s1 on kick_rollaxis.
  Better score on 6-12 sets; worse score on ood (s1, s2) and kick_rollaxis (s1).
- PPO seeds: fewer crashes on 5-8 sets; worse score on ood for 5/5 seeds.
- cls_es: no crash difference on any set; better score on 4 sets.
The 3.2 conclusions survive correction. The sentence "the descriptive scenarios are uncorrected for multiplicity" in 3.5 can be replaced.
