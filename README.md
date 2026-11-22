# human-aware-mrpp

Reference implementation, raw data and analysis for the manuscript

> **Human-Aware Multi-Robot Path Planning with Uncertainty-Aware Prediction of Pedestrian Motion**
> Saif Allah M. Abgenah, Abdusalam Aldrawi and Hamza Ali Juma (submitted to *IET Cyber-Systems and Robotics*).

The repository contains everything needed to regenerate every table and figure of the paper:
the planner, all baselines, the simulation, the experiment scripts with fixed seeds, the raw
per-scenario results, the statistical analysis and the analysis plan that was written before the
closed-loop results were seen (`ANALYSIS_PLAN.md`).

## What is implemented

| Module | Content | Paper |
|---|---|---|
| `hamrpp/maps.py` | Map 1 (11×11, 72 nodes, 296 directed edges) and Map 2 "hospital wing" (13×17, 133 nodes, 376 directed edges) | Sect. 5.1 |
| `hamrpp/costs.py` | Turn cost, proxemic social cost (Eqs. 1–3), uncertainty-aware social cost (Eqs. 5–6) | Sect. 4.2, 4.4 |
| `hamrpp/planners.py` | Social-cost Dijkstra, classic and hard-exclusion baselines, the earlier "v1" reservation scheme (ablation) | Sect. 4.3, 5.4 |
| `hamrpp/spacetime.py` | Space-time prioritized coordination with a full reservation table (proposed coordination layer, Algorithm 1) | Sect. 4.3 |
| `hamrpp/cbs.py` | Time-budgeted vanilla Conflict-Based Search | Sect. 5.4 |
| `hamrpp/modern.py` | PBS and ECBS(w = 1.5) with the same social cost | Sect. 5.4 |
| `hamrpp/lacam.py` | PIBT and LaCAM with a social cost-to-go | Sect. 5.4 |
| `hamrpp/dynamic.py` | Closed-loop simulation with moving people (steady, erratic, reactive social-force walkers), CV and uncertainty-aware prediction, oracle | Sect. 4.4, 5.3 |
| `hamrpp/metrics.py`, `hamrpp/stats.py` | Metrics; Wilcoxon, McNemar, Holm, bootstrap | Sect. 5.5, 5.6 |

All baselines are compact Python implementations written for this study with the same motion model
(synchronous steps, waits allowed, robots wait at their goals) and the same social cost; they are not
the original authors' C++ code.

## Requirements

Python 3.12 and the packages in `requirements.txt` (NumPy, SciPy, Matplotlib, pandas; exact versions
used for the paper are pinned there). No other dependencies; everything runs on one CPU core.

```
python -m pip install -r requirements.txt
```

## Reproducing the paper

Run from the repository root. Times are for the laptop used in the paper (Intel Core, Windows 11).

| Step | Command | Output | Time |
|---|---|---|---|
| Unit tests (21) | `python -m unittest discover -s tests -v` | — | < 5 s |
| Static experiments (Tables 2–6, Figs. 3–7) | `python experiments/run_experiments.py` | `results/raw/*` | ≈ 6 min |
| Closed loop, steady walkers + horizon sweep (Table 7, Figs. 8–9) | `python experiments/run_dynamic.py` | `results/raw/dynamic_trials.csv`, `dynamic_horizon_trials.csv` | ≈ 100 min |
| Robustness: erratic walkers (Table 8) | `python experiments/run_dynamic.py --robustness` | `dynamic_robustness_trials.csv` | ≈ 10 min |
| Robustness: reactive social-force walkers (Table 8) | `python experiments/run_dynamic.py --reactive` | `dynamic_reactive_trials.csv` | ≈ 10 min |
| Uncertainty-aware prediction, confirmatory (Table 9, Fig. 10) | `python experiments/run_uncertainty.py` | `uncertainty_trials.csv` | ≈ 40 min |
| Tables and figures | `python experiments/analyze.py`, `python experiments/analyze_dynamic.py`, `python experiments/analyze_uncertainty.py`, `python experiments/make_schematic.py` | `results/tables/*`, `results/figures/*` | ≈ 1 min |

The raw data used in the paper are already in `results/raw`, so the analysis scripts can be run
without repeating the experiments. Every scenario has a deterministic seed (`hamrpp.scenarios.trial_seed`),
so reruns give identical paths and metrics; only planning times vary with the machine.

## Seeds and study design

| Study | Base seed | Scenarios |
|---|---|---|
| Static comparison, Map 1 / Map 2 | 2026 / 2027 | 80 / 50 |
| Sensitivity | 3001 | 20 |
| Scalability | 4001 | 10 per robots × people cell |
| Closed loop (steady, erratic, reactive) | 5005 | 540 / 90 / 90 |
| Uncertainty-aware confirmatory study (new seeds) | 6006 | 270 / 90 / 90 |
| Smoke test only (not reported) | 7007 | 3 |

The closed-loop analysis followed `ANALYSIS_PLAN.md`, written before those results were seen and
amended twice before the corresponding data existed; the amendments are dated in the file.

## Main results (from `results/tables`)

- **Static people (Map 1, 80 scenarios).** The social cost raises mean minimum clearance from people
  from 1.23 m (classic Dijkstra) to 1.39 m at 5% longer paths, with no collisions or infeasible robots;
  hard exclusion leaves 30.8% of robots without a path. CBS, PBS, ECBS, PIBT and LaCAM with the same
  social cost give the same clearance. At ten robots, vanilla CBS exceeds its 2-s budget in 40–70% of
  scenarios and its fallback plans collide; the space-time planner, ECBS, PBS and LaCAM do not.
- **Moving people.** Constant-velocity prediction reduces personal-space intrusions by 19.6% against a
  snapshot social cost for steady walkers, but has no significant effect for erratic or reactive
  walkers. The uncertainty-aware prediction reduces intrusions by 25.7%, 13.5% and 12.6% for steady,
  erratic and reactive walkers (Holm-adjusted p ≤ 0.006) without significantly reducing clearance, at
  16–21% longer travel times (pre-registered criterion met).

## Licence and citation

MIT licence (`LICENSE`). Please cite the article (see `CITATION.cff`).
