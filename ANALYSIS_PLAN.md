# Analysis plan for the closed-loop (moving people) study

Written on 2026-11-21, **before** any result of the main closed-loop study was inspected. The
only closed-loop data seen beforehand were a 12-scenario pilot (seeds `trial_seed(5005, i)`,
Map 1, 5 robots, 8 people), which used different seeds from the main study and is not part of
any reported result. Nothing below may be changed after the results are seen; any deviation must
be reported as a deviation.

## Fixed design
- Maps 1 and 2; robots {3, 5, 8}; people {4, 8, 12}; 30 paired scenarios per cell (seeds
  `trial_seed(5005, map, robots, people, i)`); every variant sees identical scenarios.
- Variants: `unaware`, `hard_exclusion`, `snapshot`, `predictive` (constant-velocity prediction,
  horizon 6, fixed a priori), `oracle` (true future positions, reference only), `lacam_snapshot`.
- Planner weights as in the static study (turn 1.0, social 1.0, sigma 0.8 m, r_p 1.0 m), chosen
  earlier only to reproduce the original Table 1.
- Pedestrians: constant personal speed 0.5-1.0 m/step along shortest corridor routes to random
  waypoints; they do not react to robots.

## Hypotheses
- **H1 (primary).** `predictive` has a lower personal-space intrusion rate than `snapshot`.
- **H2.** `predictive` has a larger mean minimum robot-human clearance than `snapshot`.
- **H3 (cost).** `predictive` increases time to goal relative to `snapshot`; we report by how much.
- **H4.** `predictive` has a lower intrusion rate than `lacam_snapshot` (modern solver, snapshot cost).

## Tests
- Unit of analysis: one scenario (paired across variants), pooled over all 9 cells x 2 maps
  (n = 540), and reported per cell as well.
- Continuous metrics (intrusion rate, mean minimum clearance, time to goal): two-sided Wilcoxon
  signed-rank on paired scenario values; effect size reported as the median paired difference and
  the mean relative change.
- Binary (all robots reached their goals): McNemar's exact test.
- Holm correction across the family {predictive vs snapshot, vs lacam_snapshot, vs unaware, vs
  hard_exclusion} for each metric; `oracle` comparisons are descriptive (it is not a realisable
  method).
- 95% bootstrap confidence intervals (2,000 resamples over scenarios).

## How results will be interpreted
- H1 is supported only if the Holm-adjusted p < 0.05 **and** the direction favours `predictive`.
  If not, the paper reports that prediction did not help under this pedestrian model.
- Per-cell results are reported in full, including cells where `predictive` is worse.

## Pre-registered robustness check (decided before results)
The pedestrian model above (steady, straight walking) favours constant-velocity prediction. A
second condition with less predictable people is therefore run on Map 1 (robots {3, 5, 8},
people 8, 30 scenarios): at every step each person pauses with probability 0.2 and picks a new
random waypoint with probability 0.1. H1 and H2 are tested there in the same way, and the result
is reported whatever it is.

## Amendment 1 (added 2026-11-21 22:43 LST, before any result of the main study, the robustness check or this condition was available; the main-study data exist only in memory until that run ends)
A third pedestrian condition with **reactive** people (social force model, Helbing and Molnar 1995)
is added at the authors' request, to test prediction against people who react to each other and to
robots, which constant-velocity prediction does not model. Design: Map 1, robots {3, 5, 8},
people 8, 30 scenarios (seeds `trial_seed(5005, 8, robots, 8, i)`), all six variants.

Pedestrian model: each person follows its corridor route at its personal speed; at every step a
social-force deviation is added, repulsion from other people
(A = 0.5 m, B = 0.3 m, contact distance 0.5 m) and from robots (same form, contact distance
0.6 m), capped at 0.3 m per step, and kept only if the new position stays within 0.5 m of a free
node (people do not walk through walls). A person slows to half speed when a robot is within
1.0 m ahead in its walking direction. These values were set from the cited model's typical
magnitudes before any run and are not tuned.

The oracle cannot know how people will react to robots that have not moved yet, so in this
condition it simulates the future with the robots held at their current nodes; it is an
approximate reference only.

H1 and H2 are tested exactly as in the main study (Wilcoxon signed-rank, Holm over the same
family) and the result is reported whatever it is.

## Amendment 2: uncertainty-aware prediction (added 2026-11-21 23:43 LST, before any code for it was written or run)
Motivation (from the results above): constant-velocity (CV) prediction treats the predicted
position as certain. It helped with steady walkers but not with erratic or reactive people.

**Method (fixed now, not tuned).** At plan step k the predicted position of person i is treated
as Gaussian, N(mu_ik, s_k^2 I), with mu_ik the CV prediction and s_k = s * k, s = 0.3 m per step.
The value s = 0.3 m (per 1 s step) is the typical between-person and within-person variability of
human walking speed (about 0.2-0.3 m/s) and is chosen from that argument, not from our simulator.
- Proximity term: the exact expectation of Eq. (2)'s Gaussian under this uncertainty,
  alpha * sigma^2 / (sigma^2 + s_k^2) * exp(-d^2 / (2 (sigma^2 + s_k^2))): the cost spreads out and
  its peak falls as uncertainty grows (its total mass is unchanged).
- Comfort term: Eq. (3) with the personal-space radius inflated by one standard deviation,
  r_p + s_k (the usual chance-constraint style inflation).
- Everything else (horizon 6, weights, planner) is identical to the `predictive` variant.

**Confirmatory study on fresh seeds** (base seed 6006, never used before; no tuning runs on them):
- Steady walkers: maps 1 and 2, robots {3, 5, 8}, people {4, 8, 12}, 15 scenarios per cell (n = 270).
- Erratic walkers (pause 0.2, turn 0.1): map 1, robots {3, 5, 8}, people 8, 30 scenarios (n = 90).
- Reactive walkers (social force, Amendment 1): map 1, robots {3, 5, 8}, people 8, 30 scenarios (n = 90).
- Variants: `snapshot`, `predictive` (CV), `predictive_unc` (new), `oracle` (reference).

**Hypotheses.**
- H5 (primary): `predictive_unc` has a lower intrusion rate than `snapshot`, tested separately in
  each of the three pedestrian conditions.
- H6: `predictive_unc` does not have a significantly lower mean minimum clearance than `snapshot`
  in any condition.
- H7: `predictive_unc` vs `predictive` (intrusions, clearance), reported in each condition.

(Note: this amendment was first appended at 23:43 with the backquoted variant names accidentally
dropped by the shell; they were restored a few minutes later, before any code for the method
existed. No other wording was changed.)
- Tests: paired two-sided Wilcoxon signed-rank; Holm over the 3 conditions x 2 comparisons
  (vs snapshot, vs predictive) for each metric.

**Success criterion (fixed now).** The paper may claim that uncertainty-aware prediction "makes
anticipation robust" only if H5 holds (Holm p < 0.05) in **all three** conditions and H6 holds.
Otherwise the paper reports exactly in which conditions it helps and in which it does not.

**Smoke test rule.** Before the confirmatory run, the code may be executed on at most 3 scenarios
with base seed 7007 (not part of any reported result), only to check that it runs and is
collision-free. Its outcomes will not be used to change s or any other setting.
