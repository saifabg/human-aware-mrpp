"""Scenario generation. Every trial has its own seed derived from a base seed, so any single trial
can be regenerated independently and all methods see exactly the same scenario."""
import numpy as np

# The fixed illustrative scenario (Fig. 4, Table 1): robots 1->50, 10->60, 20->70, four humans.
ILLUSTRATIVE_ROBOTS = [(1, 50), (10, 60), (20, 70)]
ILLUSTRATIVE_HUMANS = [(2.5, 3.5), (5.5, 2.0), (3.0, 6.5), (7.0, 5.0)]


def illustrative():
    robots = [{"id": i, "start": s, "goal": g, "priority": 3 - i} for i, (s, g) in enumerate(ILLUSTRATIVE_ROBOTS)]
    return robots, list(ILLUSTRATIVE_HUMANS)


def random_scenario(gmap, n_robots, n_humans, seed):
    """Starts and goals: uniform over free nodes, all starts distinct, all goals distinct, and
    start != goal for each robot. Humans: uniform over the map's bounding box (so a human may
    stand inside an obstacle block). Priorities n_robots .. 1."""
    rng = np.random.default_rng(seed)
    nodes = np.array(gmap.nodes)
    while True:
        starts = rng.choice(nodes, n_robots, replace=False)
        goals = rng.choice(nodes, n_robots, replace=False)
        if all(s != g for s, g in zip(starts, goals)):
            break
    x0, y0, x1, y1 = gmap.bbox
    hum = rng.uniform([x0, y0], [x1, y1], size=(n_humans, 2))
    robots = [{"id": i, "start": int(s), "goal": int(g), "priority": n_robots - i}
              for i, (s, g) in enumerate(zip(starts, goals))]
    return robots, [(float(x), float(y)) for x, y in hum]


def trial_seed(base, *idx):
    """Deterministic per-trial seed."""
    return int(np.random.SeedSequence([base, *idx]).generate_state(1)[0])
