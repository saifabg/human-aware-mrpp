"""Closed-loop experiment with moving people (new Sec. 6.x): does a predictive social cost reduce
personal-space intrusions compared with a snapshot social cost?

    python experiments/run_dynamic.py            # full study (about 30-45 min)
    python experiments/run_dynamic.py --quick    # 5 trials per cell, for a smoke test

Writes results/raw/dynamic_trials.csv and results/raw/dynamic_horizon_trials.csv.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, RESULTS, write_csv, save_json, environment  # noqa: E402
from hamrpp import map1, map2  # noqa: E402
from hamrpp.dynamic import closed_loop, VARIANTS  # noqa: E402
from hamrpp.scenarios import random_scenario, trial_seed  # noqa: E402

BASE_SEED = 5005
ROBOTS = [3, 5, 8]
HUMANS = [4, 8, 12]
TRIALS = 30
HORIZONS = [0, 2, 4, 6, 10]


def main(quick=False):
    n = 5 if quick else TRIALS
    rows = []
    t0 = time.time()
    for mname, mk in (("map1", map1), ("map2", map2)):
        g = mk()
        for nr in ROBOTS:
            for nh in HUMANS:
                for i in range(n):
                    seed = trial_seed(BASE_SEED, 1 if mname == "map1" else 2, nr, nh, i)
                    robots, _ = random_scenario(g, nr, 0, seed)
                    for v in VARIANTS:
                        r = closed_loop(g, robots, nh, v, seed)
                        r.update(map=mname, trial=i)
                        rows.append(r)
                print(f"dynamic {mname} robots={nr} humans={nh} done ({time.time() - t0:.0f} s)", flush=True)
    write_csv(os.path.join(RAW, "dynamic_trials.csv"), rows)
    # prediction-horizon sensitivity (map 1, 5 robots, 8 humans)
    hrows = []
    g = map1()
    for i in range(n):
        seed = trial_seed(BASE_SEED, 9, 5, 8, i)
        robots, _ = random_scenario(g, 5, 0, seed)
        for H in HORIZONS:
            r = closed_loop(g, robots, 8, "predictive", seed, pred_horizon=H)
            r.update(map="map1", trial=i, pred_horizon=H)
            hrows.append(r)
    write_csv(os.path.join(RAW, "dynamic_horizon_trials.csv"), hrows)
    env = environment(); env.update(experiment="dynamic", base_seed=BASE_SEED, robots=ROBOTS, humans=HUMANS,
                                    trials=n, horizons=HORIZONS, runtime_s=round(time.time() - t0, 1))
    save_json(os.path.join(RESULTS, "environment_dynamic.json"), env)


def robustness(quick=False):
    """Pre-registered robustness check (ANALYSIS_PLAN.md): less predictable pedestrians."""
    n = 5 if quick else TRIALS
    rows, g, t0 = [], map1(), time.time()
    for nr in ROBOTS:
        for i in range(n):
            seed = trial_seed(BASE_SEED, 7, nr, 8, i)
            robots, _ = random_scenario(g, nr, 0, seed)
            for v in VARIANTS:
                r = closed_loop(g, robots, 8, v, seed, pause_p=0.2, turn_p=0.1)
                r.update(map="map1", trial=i, pause_p=0.2, turn_p=0.1)
                rows.append(r)
        print(f"robustness robots={nr} done ({time.time() - t0:.0f} s)", flush=True)
    write_csv(os.path.join(RAW, "dynamic_robustness_trials.csv"), rows)


def reactive(quick=False):
    """Amendment 1 (ANALYSIS_PLAN.md): reactive social-force pedestrians."""
    n = 5 if quick else TRIALS
    rows, g, t0 = [], map1(), time.time()
    for nr in ROBOTS:
        for i in range(n):
            seed = trial_seed(BASE_SEED, 8, nr, 8, i)
            robots, _ = random_scenario(g, nr, 0, seed)
            for v in VARIANTS:
                r = closed_loop(g, robots, 8, v, seed, reactive=True)
                r.update(map="map1", trial=i, reactive=1)
                rows.append(r)
        print(f"reactive robots={nr} done ({time.time() - t0:.0f} s)", flush=True)
    write_csv(os.path.join(RAW, "dynamic_reactive_trials.csv"), rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--robustness", action="store_true", help="run only the pre-registered robustness check")
    ap.add_argument("--reactive", action="store_true", help="run only the reactive (social force) condition")
    a = ap.parse_args()
    if a.robustness:
        robustness(a.quick)
    elif a.reactive:
        reactive(a.quick)
    else:
        main(a.quick)
