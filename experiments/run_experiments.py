"""Runs every experiment in the paper and writes raw per-robot / per-trial CSV files to results/raw.

    python experiments/run_experiments.py            # everything (about 15-30 min on a laptop)
    python experiments/run_experiments.py --only comparison_map1 sensitivity

Experiments (paper section -> output):
  illustrative       Sec. 6.1, Fig. 4, Table 1   -> raw/illustrative_robots.csv
  comparison_map1    Sec. 6.3, Fig. 6, Tables 2-3 -> raw/comparison_map1_{robots,trials}.csv
  sensitivity        Sec. 6.4, Fig. 7            -> raw/sensitivity_{robots,trials}.csv
  scalability        Sec. 6.5, Fig. 8, Table 4   -> raw/scalability_{robots,trials}.csv  (all four methods)
  comparison_map2    Sec. 6.6, Fig. 9, Table 5   -> raw/comparison_map2_{robots,trials}.csv
  cbs_budget         Supplementary (10 robots)   -> raw/cbs_budget_trials.csv
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, METHOD_ORDER, run_method, write_csv, save_json, environment, RESULTS
from hamrpp import DEFAULT_CONFIG, map1, map2
from hamrpp.scenarios import illustrative, random_scenario, trial_seed

SEEDS = {"comparison_map1": 2026, "comparison_map2": 2027, "sensitivity": 3001, "scalability": 4001}
N_TRIALS = {"comparison_map1": 80, "comparison_map2": 50, "sensitivity": 20, "scalability": 10}
SOCIAL_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
ROBOT_COUNTS = [3, 5, 10]
HUMAN_COUNTS = [4, 8, 15]
CBS_BUDGETS = [(50, 0.5), (150, 2.0), (1000, 10.0)]


def exp_illustrative():
    g = map1(); robots, humans = illustrative()
    rows, trial = run_method(g, robots, humans, "human_aware", DEFAULT_CONFIG)
    write_csv(os.path.join(RAW, "illustrative_robots.csv"), rows)
    print("illustrative:", [(r["robot"], r["path_len"], r["turns"], round(r["min_human_dist"], 2)) for r in rows])


def exp_comparison(name, gmap):
    R, T = [], []
    for i in range(N_TRIALS[name]):
        seed = trial_seed(SEEDS[name], i)
        robots, humans = random_scenario(gmap, 3, 4, seed)
        for m in METHOD_ORDER:
            rows, trial = run_method(gmap, robots, humans, m)
            for r in rows:
                r.update(trial=i, seed=seed, method=m)
            trial.update(trial=i, seed=seed)
            R += rows; T.append(trial)
    print(f"{name}: {N_TRIALS[name]} trials done")
    write_csv(os.path.join(RAW, f"{name}_robots.csv"), R)
    write_csv(os.path.join(RAW, f"{name}_trials.csv"), T)


def exp_sensitivity():
    g = map1(); R, T = [], []
    for i in range(N_TRIALS["sensitivity"]):
        seed = trial_seed(SEEDS["sensitivity"], i)
        robots, humans = random_scenario(g, 3, 4, seed)
        for w in SOCIAL_WEIGHTS:
            cfg = DEFAULT_CONFIG.with_(social_weight=w)
            for m in ("human_aware_st", "human_aware"):
                rows, trial = run_method(g, robots, humans, m, cfg)
                for r in rows:
                    r.update(trial=i, seed=seed, social_weight=w, method=m)
                trial.update(trial=i, seed=seed, social_weight=w)
                R += rows; T.append(trial)
    write_csv(os.path.join(RAW, "sensitivity_robots.csv"), R)
    write_csv(os.path.join(RAW, "sensitivity_trials.csv"), T)
    print("sensitivity: done")


def exp_scalability():
    g = map1(); R, T = [], []
    for nr in ROBOT_COUNTS:
        for nh in HUMAN_COUNTS:
            for i in range(N_TRIALS["scalability"]):
                seed = trial_seed(SEEDS["scalability"], nr, nh, i)
                robots, humans = random_scenario(g, nr, nh, seed)
                for m in METHOD_ORDER:
                    rows, trial = run_method(g, robots, humans, m)
                    for r in rows:
                        r.update(trial=i, seed=seed, method=m)
                    trial.update(trial=i, seed=seed)
                    R += rows; T.append(trial)
            print(f"scalability: robots={nr} humans={nh} done")
    write_csv(os.path.join(RAW, "scalability_robots.csv"), R)
    write_csv(os.path.join(RAW, "scalability_trials.csv"), T)


def exp_cbs_budget():
    """Supplementary: does a larger CBS budget or a conflict-minimizing fallback rescue CBS where
    the 2 s budget binds (10 robots)? Same 10-robot scenarios as the scalability study."""
    g = map1(); T = []
    for nh in HUMAN_COUNTS:
        for i in range(N_TRIALS["scalability"]):
            seed = trial_seed(SEEDS["scalability"], 10, nh, i)
            robots, humans = random_scenario(g, 10, nh, seed)
            for (n_exp, secs) in CBS_BUDGETS:
                for fb in ("lowest_cost", "fewest_conflicts"):
                    cfg = DEFAULT_CONFIG.with_(cbs_max_expansions=n_exp, cbs_time_budget_s=secs, cbs_fallback=fb)
                    _, trial = run_method(g, robots, humans, "cbs", cfg)
                    trial.update(trial=i, seed=seed, budget_expansions=n_exp, budget_s=secs, fallback=fb)
                    T.append(trial)
        print(f"cbs_budget: humans={nh} done")
    write_csv(os.path.join(RAW, "cbs_budget_trials.csv"), T)


ALL = ["illustrative", "comparison_map1", "sensitivity", "scalability", "comparison_map2", "cbs_budget"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=ALL, choices=ALL)
    a = ap.parse_args()
    t0 = time.time()
    env = environment(); env["experiments"] = a.only; env["seeds"] = SEEDS; env["n_trials"] = N_TRIALS
    for e in a.only:
        t1 = time.time()
        if e == "illustrative": exp_illustrative()
        elif e == "comparison_map1": exp_comparison(e, map1())
        elif e == "comparison_map2": exp_comparison(e, map2())
        elif e == "sensitivity": exp_sensitivity()
        elif e == "scalability": exp_scalability()
        elif e == "cbs_budget": exp_cbs_budget()
        env[f"runtime_s_{e}"] = round(time.time() - t1, 1)
    env["runtime_s_total"] = round(time.time() - t0, 1)
    save_json(os.path.join(RESULTS, f"environment_{'_'.join(a.only) if len(a.only) < len(ALL) else 'all'}.json"), env)
