"""Two stronger multi-agent path finding baselines, implemented with the same social cost (Eq. 1),
motion model (synchronous steps, waits allowed, robots wait at their goals) and time budget as
the other planners:

* PBS  - Priority-Based Search (Ma et al., AAAI 2019): depth-first search over partial priority
         orderings; low level replans agents against all higher-priority paths.
* ECBS - Enhanced CBS with suboptimality bound w (Barer et al., SoCS 2014): focal search at the
         high level (fewest conflicts among nodes with cost <= w * LB) and at the low level
         (fewest conflicts with the other agents' paths among states with f <= w * f_min).

Both return the best plan found so far (fewest conflicts, then lowest cost) when the time budget
runs out. These are faithful but compact Python implementations, not the authors' C++ code; the
improvements of EECBS (explicit estimation, bypassing, conflict prioritization) and MAPF-LNS2
(large-neighbourhood repair) are not included."""
import bisect
import heapq
import itertools
import math
import time

from .config import DEFAULT_CONFIG
from .costs import turn_cost, SocialCache
from .cbs import low_level, find_conflicts


def _reservations(paths, ids, horizon):
    """Vertex/edge constraints that make a path avoid the paths of agents `ids`
    (incl. waiting at their goals until `horizon`)."""
    vc, ec = set(), set()
    for a in ids:
        p = paths.get(a)
        if not p:
            continue
        for t, v in enumerate(p):
            vc.add((v, t))
        for t in range(len(p) - 1, horizon + 1):
            vc.add((p[-1], t))
        for t in range(len(p) - 1):
            if p[t] != p[t + 1]:
                ec.add((p[t + 1], p[t], t))
    return vc, ec


def _cost(gmap, path, cfg, cache):
    c, prev = 0.0, None
    for t in range(len(path) - 1):
        u, v = path[t], path[t + 1]
        if u == v:
            c += 1.0 + cfg.social_weight * cache(u, u)
        else:
            w = dict(gmap.adj[u])[v]
            c += w + cfg.turn_weight * turn_cost(gmap.pos[prev] if prev is not None else None, gmap.pos[u], gmap.pos[v], cfg.turn_noise_deg) \
                + cfg.social_weight * cache(u, v)
            prev = u
    return c


def _better(a_conf, a_cost, b_conf, b_cost):
    return (a_conf, a_cost) < (b_conf, b_cost)


# ----------------------------------------------------------------------------- PBS
def plan_pbs(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    cache = SocialCache(gmap, humans, cfg)
    ids = [r["id"] for r in robots]
    rob = {r["id"]: r for r in robots}
    H0 = 2 * gmap.n_nodes
    t0 = time.perf_counter()

    def plan_one(a, paths, higher):
        horizon = H0 + max([len(paths[b]) for b in higher if paths.get(b)] + [0])
        vc, ec = _reservations(paths, higher, horizon)
        last = max([t for (v, t) in vc if v == rob[a]["goal"]], default=-1)
        p, c = low_level(gmap, rob[a]["start"], rob[a]["goal"], cfg, cache, vc, ec, horizon, None, last)
        return p, c

    def higher_than(prio, a):
        """All agents with higher priority than a (transitive closure of the partial order)."""
        out, stack = set(), [a]
        while stack:
            x = stack.pop()
            for (h, l) in prio:
                if l == x and h not in out:
                    out.add(h); stack.append(h)
        return out

    def lower_than(prio, a):
        out, stack = set(), [a]
        while stack:
            x = stack.pop()
            for (h, l) in prio:
                if h == x and l not in out:
                    out.add(l); stack.append(l)
        return out

    paths, costs = {}, {}
    for a in ids:
        p, c = plan_one(a, {}, [])
        if not p:
            return {x: [] for x in ids}, {"status": "infeasible", "expansions": 0}
        paths[a], costs[a] = p, c
    root = {"prio": frozenset(), "paths": paths, "costs": costs}
    stack = [root]
    best, best_key = root, (len(find_conflicts(paths, False)), sum(costs.values()))
    expansions = 0
    while stack:
        if time.perf_counter() - t0 > cfg.cbs_time_budget_s:
            return best["paths"], {"status": "timeout", "expansions": expansions}
        node = stack.pop(); expansions += 1
        confs = find_conflicts(node["paths"], first_only=False)
        key = (len(confs), sum(node["costs"].values()))
        if key < best_key:
            best, best_key = node, key
        if not confs:
            return node["paths"], {"status": "solved", "expansions": expansions}
        _, a, b, _, _ = confs[0]
        children = []
        for hi, lo in ((a, b), (b, a)):
            if (lo, hi) in node["prio"] or hi in lower_than(node["prio"], hi) and lo in higher_than(node["prio"], hi):
                continue
            if hi in lower_than(node["prio"], lo):      # would create a cycle
                continue
            prio = node["prio"] | {(hi, lo)}
            paths2, costs2 = dict(node["paths"]), dict(node["costs"])
            # replan lo and everything below it, in a topological order
            affected = [lo] + sorted(lower_than(prio, lo), key=lambda x: len(higher_than(prio, x)))
            ok = True
            for x in affected:
                hx = higher_than(prio, x)
                clash = x == lo or any(c for c in find_conflicts({x: paths2[x], **{h: paths2[h] for h in hx}}, False)
                                       if x in (c[1], c[2]))
                if not clash:
                    continue
                p, c = plan_one(x, paths2, hx)
                if not p:
                    ok = False; break
                paths2[x], costs2[x] = p, c
            if ok:
                children.append({"prio": prio, "paths": paths2, "costs": costs2})
        children.sort(key=lambda n: -sum(n["costs"].values()))   # cheaper child explored first
        stack.extend(children)
    return best["paths"], {"status": "timeout", "expansions": expansions}


# ----------------------------------------------------------------------------- ECBS
def _heuristic(gmap, goal, cfg, cache):
    """Admissible cost-to-go: backward Dijkstra with distance + social cost (turn cost >= 0 ignored)."""
    h = {goal: 0.0}
    pq = [(0.0, goal)]
    while pq:
        d, v = heapq.heappop(pq)
        if d > h[v]:
            continue
        for u, w in gmap.adj[v]:
            nd = d + w + cfg.social_weight * cache(u, v)
            if nd < h.get(u, math.inf):
                h[u] = nd; heapq.heappush(pq, (nd, u))
    return h


def _focal_low_level(gmap, start, goal, cfg, cache, hmap, vcons, econs, others, w, horizon):
    """Focal search over (node, t). Secondary key: number of vertex/edge conflicts with `others`
    (dict agent -> path, robots wait at goals). Returns (path, cost, f_min_lower_bound)."""
    pos = gmap.pos
    at = lambda p, t: p[t] if t < len(p) else p[-1]
    occ, moves = {}, set()
    T_o = max([len(p) for p in others.values()] + [1])
    for p in others.values():
        for t in range(T_o + horizon):
            occ[(at(p, t), t)] = occ.get((at(p, t), t), 0) + 1
            if t + 1 < len(p) and p[t] != p[t + 1]:
                moves.add((p[t + 1], p[t], t))
    last_goal_con = max([t for (v, t) in vcons if v == goal], default=-1)
    if (start, 0) in vcons:
        return [], math.inf, math.inf
    tie = itertools.count()
    s0 = (start, 0)
    g = {s0: 0.0}; parent = {s0: None}; prevnode = {s0: None}; conf = {s0: occ.get(s0, 0)}
    open_list = []                       # sorted list of (f, id, state)
    in_open = {}
    focal = []
    def push(s):
        f = g[s] + hmap.get(s[0], math.inf)
        k = (f, next(tie), s); bisect.insort(open_list, k); in_open[s] = k
        return f
    f0 = push(s0)
    f_min = f0
    heapq.heappush(focal, (conf[s0], f0, next(tie), s0))
    closed = set()
    while open_list:
        f_min_new = open_list[0][0]
        if f_min_new > f_min:            # bound grew: move newly eligible states into FOCAL
            for (f, _, s) in open_list:
                if f > w * f_min_new:
                    break
                if f > w * f_min:
                    heapq.heappush(focal, (conf[s], f, next(tie), s))
            f_min = f_min_new
        while focal:
            c, f, _, s = heapq.heappop(focal)
            if s in in_open and in_open[s][0] == f and f <= w * f_min + 1e-9:
                break
        else:
            return [], math.inf, math.inf
        open_list.remove(in_open.pop(s)); closed.add(s)
        u, t = s
        if u == goal and t > last_goal_con:
            path, x = [], s
            while x is not None:
                path.append(x[0]); x = parent[x]
            return path[::-1], g[s], f_min
        if t >= horizon:
            continue
        pp = pos[prevnode[s]] if prevnode[s] is not None else None
        for v, wt in [(u, 1.0)] + list(gmap.adj[u]):
            ns = (v, t + 1)
            if ns in vcons or (u, v, t) in econs or ns in closed:
                continue
            if v == u:
                step = 1.0 + cfg.social_weight * cache(u, u); pn = prevnode[s]
            else:
                step = wt + cfg.turn_weight * turn_cost(pp, pos[u], pos[v], cfg.turn_noise_deg) + cfg.social_weight * cache(u, v); pn = u
            ng = g[s] + step
            nc = conf[s] + occ.get(ns, 0) + (1 if (u, v, t) in moves else 0)
            if ng < g.get(ns, math.inf) - 1e-12:
                if ns in in_open:
                    open_list.remove(in_open.pop(ns))
                g[ns] = ng; parent[ns] = s; prevnode[ns] = pn; conf[ns] = nc
                f = push(ns)
                if f <= w * f_min + 1e-9:
                    heapq.heappush(focal, (nc, f, next(tie), ns))
    return [], math.inf, math.inf


def plan_ecbs(gmap, robots, humans, cfg=DEFAULT_CONFIG, w=1.5):
    cache = SocialCache(gmap, humans, cfg)
    ids = [r["id"] for r in robots]
    rob = {r["id"]: r for r in robots}
    hmaps = {a: _heuristic(gmap, rob[a]["goal"], cfg, cache) for a in ids}
    H0 = 2 * gmap.n_nodes
    t0 = time.perf_counter()

    def replan(a, cons, paths):
        vc, ec = cons
        horizon = H0 + max([t for (_, t) in vc] + [t for (_, _, t) in ec] + [0])
        others = {b: paths[b] for b in ids if b != a and paths.get(b)}
        return _focal_low_level(gmap, rob[a]["start"], rob[a]["goal"], cfg, cache, hmaps[a], set(vc), set(ec), others, w, horizon)

    paths, costs, lbs = {}, {}, {}
    cons0 = {a: (frozenset(), frozenset()) for a in ids}
    for a in ids:
        p, c, lb = replan(a, cons0[a], paths)
        if not p:
            return {x: [] for x in ids}, {"status": "infeasible", "expansions": 0}
        paths[a], costs[a], lbs[a] = p, c, lb
    tie = itertools.count()

    def make(cons, paths, costs, lbs):
        return {"cons": cons, "paths": paths, "costs": costs, "lbs": lbs, "cost": sum(costs.values()),
                "lb": sum(lbs.values()), "nconf": len(find_conflicts(paths, False))}

    root = make(cons0, paths, costs, lbs)
    open_ = [root]
    best = root
    expansions = 0
    while open_:
        if time.perf_counter() - t0 > cfg.cbs_time_budget_s:
            return best["paths"], {"status": "timeout", "expansions": expansions}
        lb_min = min(n["lb"] for n in open_)
        focal = [n for n in open_ if n["cost"] <= w * lb_min + 1e-9]
        node = min(focal, key=lambda n: (n["nconf"], n["cost"]))
        open_.remove(node); expansions += 1
        if _better(node["nconf"], node["cost"], best["nconf"], best["cost"]):
            best = node
        conf = find_conflicts(node["paths"])
        if not conf:
            return node["paths"], {"status": "solved", "expansions": expansions}
        kind, a, b, where, t = conf[0]
        for agent in (a, b):
            vc, ec = node["cons"][agent]
            if kind == "vertex":
                vc = vc | {(where, t)}
            else:
                u, v = where if agent == a else (where[1], where[0])
                ec = ec | {(u, v, t)}
            cons2 = dict(node["cons"]); cons2[agent] = (vc, ec)
            p, c, lb = replan(agent, (vc, ec), node["paths"])
            if not p:
                continue
            paths2 = dict(node["paths"]); paths2[agent] = p
            costs2 = dict(node["costs"]); costs2[agent] = c
            lbs2 = dict(node["lbs"]); lbs2[agent] = lb
            open_.append(make(cons2, paths2, costs2, lbs2))
    return best["paths"], {"status": "timeout", "expansions": expansions}
