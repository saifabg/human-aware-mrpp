"""Time-budgeted Conflict-Based Search baseline (Algorithm 3, Sec. 5.3).

"Vanilla" CBS (Sharon et al., 2012): time-expanded low-level search over (node, t) with wait
actions, the same edge cost as the proposed planner (Eq. 1) read from a per-scenario social cost
cache, and a high level that branches on the first vertex or edge (swap) conflict. Agents stay
at their goal after arrival. No cardinal-conflict heuristics, bypass or disjoint splitting.
The high level stops after `cbs_max_expansions` expansions or `cbs_time_budget_s` seconds and
then returns the best node found so far (lowest cost, as in Algorithm 3; optionally fewest
conflicts, used only in the supplementary budget analysis)."""
import heapq
import itertools
import math
import time

from .config import DEFAULT_CONFIG
from .costs import turn_cost, SocialCache


def _at(path, t):
    return path[t] if t < len(path) else path[-1]


def find_conflicts(paths, first_only=True):
    """Vertex conflicts (same node, same t) and edge/swap conflicts, synchronous motion,
    agents wait at their goals. Returns a list of conflicts (first one if first_only)."""
    ids = [a for a in paths if paths[a]]
    if len(ids) < 2:
        return []
    T = max(len(paths[a]) for a in ids)
    out = []
    for t in range(T):
        for a, b in itertools.combinations(ids, 2):
            if _at(paths[a], t) == _at(paths[b], t):
                out.append(("vertex", a, b, _at(paths[a], t), t))
                if first_only:
                    return out
        if t + 1 < T:
            for a, b in itertools.combinations(ids, 2):
                ua, va = _at(paths[a], t), _at(paths[a], t + 1)
                ub, vb = _at(paths[b], t), _at(paths[b], t + 1)
                if ua == vb and va == ub and ua != va:
                    out.append(("edge", a, b, (ua, va), t))
                    if first_only:
                        return out
    return out


def low_level(gmap, start, goal, cfg, cache, vcons, econs, horizon, allowed=None, last_goal_con=None):
    """Time-expanded Dijkstra. vcons: set of (v, t); econs: set of (u, v, t) for moves t->t+1.
    `allowed` optionally restricts the node set. The goal is accepted only after the last vertex
    constraint on it (so the robot can wait there). Returns (path, cost) or ([], inf)."""
    pos = gmap.pos
    if last_goal_con is None:
        last_goal_con = max([t for (v, t) in vcons if v == goal], default=-1)
    start_state = (start, 0)
    if start_state in vcons:
        return [], math.inf
    if allowed is not None and (start not in allowed or goal not in allowed):
        return [], math.inf
    dist = {start_state: 0.0}
    parent = {start_state: None}
    pq = [(0.0, 0, start, None)]
    while pq:
        d, t, u, pnode = heapq.heappop(pq)
        if d > dist.get((u, t), math.inf):
            continue
        if u == goal and t > last_goal_con:
            path, s = [], (u, t)
            while s is not None:
                path.append(s[0]); s = parent[s]
            return path[::-1], d
        if t >= horizon:
            continue
        pu = pos[u]
        pp = pos[pnode] if pnode is not None else None
        moves = [(u, 1.0)] + list(gmap.adj[u])
        for v, w in moves:
            nt = t + 1
            if (v, nt) in vcons or (u, v, t) in econs:
                continue
            if allowed is not None and v not in allowed:
                continue
            if v == u:
                cost = 1.0 + cfg.social_weight * cache(u, u)
                npn = pnode
            else:
                cost = w + cfg.turn_weight * turn_cost(pp, pu, pos[v], cfg.turn_noise_deg) + cfg.social_weight * cache(u, v)
                npn = u
            nd = d + cost
            if nd < dist.get((v, nt), math.inf):
                dist[(v, nt)] = nd
                parent[(v, nt)] = (u, t)
                heapq.heappush(pq, (nd, nt, v, npn))
    return [], math.inf


def plan_cbs(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    """Returns ({robot id: path}, info) with info = {status, expansions, n_conflicts}."""
    cache = SocialCache(gmap, humans, cfg)
    base_h = 2 * gmap.n_nodes
    paths, costs = {}, {}
    cons = {r["id"]: (frozenset(), frozenset()) for r in robots}
    for r in robots:
        p, c = low_level(gmap, r["start"], r["goal"], cfg, cache, set(), set(), base_h)
        if not p:
            return {x["id"]: [] for x in robots}, {"status": "infeasible", "expansions": 0, "n_conflicts": 0}
        paths[r["id"]], costs[r["id"]] = p, c
    rob = {r["id"]: r for r in robots}
    tie = itertools.count()

    def score(node):
        return (node["cost"], len(find_conflicts(node["paths"], first_only=False))) \
            if cfg.cbs_fallback == "fewest_conflicts" else node["cost"]

    root = {"cons": cons, "paths": paths, "costs": costs, "cost": sum(costs.values())}
    open_ = [(root["cost"], next(tie), root)]
    best = root
    best_conf = len(find_conflicts(root["paths"], first_only=False)) if cfg.cbs_fallback == "fewest_conflicts" else None
    expansions = 0
    t0 = time.perf_counter()
    while open_:
        expansions += 1
        if expansions > cfg.cbs_max_expansions or time.perf_counter() - t0 > cfg.cbs_time_budget_s:
            return best["paths"], {"status": "timeout", "expansions": expansions - 1,
                                   "n_conflicts": len(find_conflicts(best["paths"], first_only=False))}
        _, _, node = heapq.heappop(open_)
        if cfg.cbs_fallback == "fewest_conflicts":
            nc = len(find_conflicts(node["paths"], first_only=False))
            if nc < best_conf or (nc == best_conf and node["cost"] < best["cost"]):
                best, best_conf = node, nc
        elif node["cost"] < best["cost"]:
            best = node
        conf = find_conflicts(node["paths"])
        if not conf:
            return node["paths"], {"status": "solved", "expansions": expansions, "n_conflicts": 0}
        kind, a, b, where, t = conf[0]
        for agent in (a, b):
            vc, ec = node["cons"][agent]
            if kind == "vertex":
                vc = vc | {(where, t)}
            else:
                u, v = where if agent == a else (where[1], where[0])
                ec = ec | {(u, v, t)}
            horizon = base_h + max([tt for (_, tt) in vc] + [tt for (_, _, tt) in ec] + [0])
            r = rob[agent]
            p, c = low_level(gmap, r["start"], r["goal"], cfg, cache, set(vc), set(ec), horizon)
            if not p:
                continue
            cons2 = dict(node["cons"]); cons2[agent] = (vc, ec)
            paths2 = dict(node["paths"]); paths2[agent] = p
            costs2 = dict(node["costs"]); costs2[agent] = c
            child = {"cons": cons2, "paths": paths2, "costs": costs2, "cost": sum(costs2.values())}
            heapq.heappush(open_, (child["cost"], next(tie), child))
    return best["paths"], {"status": "timeout", "expansions": expansions,
                           "n_conflicts": len(find_conflicts(best["paths"], first_only=False))}
