"""Confirmatory study for uncertainty-aware prediction (ANALYSIS_PLAN.md, Amendment 2).
Fresh base seed 6006; nothing was tuned on these scenarios.

    python experiments/run_uncertainty.py            # confirmatory study (about 30-45 min)
    python experiments/run_uncertainty.py --smoke    # <= 3 scenarios, seed 7007, code check only
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, RESULTS, write_csv, save_json, environment  # noqa: E402
from hamrpp import map1, map2  # noqa: E402
from hamrpp.dynamic import closed_loop, UNC_S_PER_STEP  # noqa: E402
from hamrpp.scenarios import random_scenario, trial_seed  # noqa: E402

BASE = 6006
VARIANTS = ["snapshot", "predictive", "predictive_unc", "oracle"]
CONDITIONS = {   # name -> (maps, robots, people, scenarios per cell, closed_loop kwargs)
    "steady": (("map1", "map2"), (3, 5, 8), (4, 8, 12), 15, {}),
    "erratic": (("map1",), (3, 5, 8), (8,), 30, {"pause_p": 0.2, "turn_p": 0.1}),
    "reactive": (("map1",), (3, 5, 8), (8,), 30, {"reactive": True}),
}
MAPS = {"map1": map1, "map2": map2}


def run(smoke=False):
    rows, t0 = [], time.time()
    for ci, (cname, (maps, R, H, n, kw)) in enumerate(CONDITIONS.items()):
        for mname in maps:
            g = MAPS[mname]()
            for nr in R:
                for nh in H:
                    for i in range(1 if smoke else n):
                        seed = trial_seed(7007 if smoke else BASE, ci, 1 if mname == "map1" else 2, nr, nh, i)
                        robots, _ = random_scenario(g, nr, 0, seed)
                        for v in VARIANTS:
                            r = closed_loop(g, robots, nh, v, seed, **kw)
                            r.update(condition=cname, map=mname, trial=i)
                            rows.append(r)
                    print(f"{cname} {mname} robots={nr} people={nh} done ({time.time() - t0:.0f} s)", flush=True)
                    if smoke and len(rows) >= 3 * len(VARIANTS):
                        return rows
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = run(a.smoke)
    name = "uncertainty_smoke_trials.csv" if a.smoke else "uncertainty_trials.csv"
    write_csv(os.path.join(RAW, name), rows)
    if not a.smoke:
        env = environment(); env.update(experiment="uncertainty", base_seed=BASE, s_per_step=UNC_S_PER_STEP,
                                        conditions={k: str(v) for k, v in CONDITIONS.items()})
        save_json(os.path.join(RESULTS, "environment_uncertainty.json"), env)
    print("done")
