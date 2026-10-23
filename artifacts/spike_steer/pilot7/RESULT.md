# Round-7 pilot (exploratory) -- stopped by the user, 2026-10-19 15:15 JST

Arms on top of the plan-v4 round-5 configuration (ideas from MonoRace, Bahnam et al., Sci. Robot. 2026):
- B = nod reward cap 0.85 + pulse penalty 0.001/pulse (claude-ea, seeds 1 and 4, stopped at generations 13 and 15)
- C = B + mid-manoeuvre pitch impulse in 1/4 of ES cases (LabHPPC seed 1 stopped early; seed 4 never started)
- A (control) = round 5, same seeds, generations 0-29 (existing data)

128 validation cases (round-5 selection set), plain round-5 score (LabPC preview, p7_preview.py):

| seed | gen | B score | B crashes | A score | A crashes |
|---|---|---|---|---|---|
| 1 | -1 | 1.090 | 0 | 1.090 | 0 |
| 1 | 4 | 1.189 | 1 | 1.190 | 2 |
| 1 | 9 | 1.819 | 16 | 0.912 | 0 |
| 4 | -1 | 0.993 | 0 | 0.993 | 0 |
| 4 | 4 | 1.076 | 1 | 0.964 | 0 |
| 4 | 9 | 1.409 | 3 | 1.362 | 2 |

Conclusion: no benefit; arm B is worse than the control (seed 1 gen 9: 16 crashes vs 0). Likely reasons: the nod cap
removes the gradient toward a better nod, and the pulse penalty removes pitch-braking pulses. The 8-case validation again
showed nothing (B s1 g9: 8/8 flown, composite 0.954). The user continues with the round-5 results (confirmatory M1: PASS 3/5).
