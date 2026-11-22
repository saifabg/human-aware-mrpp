"""Evaluation metrics (Sec. 5.2)."""
import itertools
import math

from .costs import turn_cost


def compress(path):
    """Drop consecutive duplicates (CBS wait actions) to get the geometric node sequence."""
    out = []
    for v in path:
        if not out or out[-1] != v:
            out.append(v)
    return out


def path_length(path):
    """Path length in nodes (time steps for CBS paths, which may contain waits)."""
    return len(path)


def turn_count(gmap, path, cfg):
    p = compress(path)
    n = 0
    for a, b, c in zip(p, p[1:], p[2:]):
        if turn_cost(gmap.pos[a], gmap.pos[b], gmap.pos[c], cfg.turn_noise_deg) > cfg.turn_count_threshold:
            n += 1
    return n


def min_human_distance(gmap, path, humans):
    if not humans or not path:
        return math.nan
    return min(math.hypot(gmap.pos[v][0] - hx, gmap.pos[v][1] - hy) for v in path for hx, hy in humans)


def collision_pairs(paths):
    """Number of robot pairs that occupy the same node at the same step at least once
    (synchronous motion, one node per step, robots wait at their goals). Infeasible robots
    (empty paths) are excluded."""
    ids = [a for a in paths if paths[a]]
    if len(ids) < 2:
        return 0
    T = max(len(paths[a]) for a in ids)
    at = lambda p, t: p[t] if t < len(p) else p[-1]
    n = 0
    for a, b in itertools.combinations(ids, 2):
        if any(at(paths[a], t) == at(paths[b], t) for t in range(T)):
            n += 1
    return n


def robot_rows(gmap, paths, robots, humans, cfg):
    rows = []
    for r in robots:
        p = paths.get(r["id"], [])
        feasible = bool(p)
        rows.append({
            "robot": r["id"], "priority": r["priority"], "start": r["start"], "goal": r["goal"],
            "feasible": int(feasible),
            "path_len": path_length(p) if feasible else math.nan,
            "turns": turn_count(gmap, p, cfg) if feasible else math.nan,
            "min_human_dist": min_human_distance(gmap, p, humans) if feasible else math.nan,
            "enters_personal_space": int(min_human_distance(gmap, p, humans) < cfg.personal_space) if feasible and humans else math.nan,
            "path": " ".join(map(str, p)),
        })
    return rows
