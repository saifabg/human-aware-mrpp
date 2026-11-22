"""PIBT (Okumura et al., Artif. Intell. 2022) and LaCAM (Okumura, AAAI 2023) with social cost.

Both solvers move all robots one step at a time. The social cost enters through each robot's
cost-to-go h_i(v): a backward Dijkstra from the robot's goal with edge cost
d(u, v) + lambda_social * (W_social + W_comfort), so robots prefer next nodes that lead to their
goal along socially cheap routes ("social PIBT"). Vertex and swap conflicts are excluded by
construction.

* PIBT  - priority inheritance with backtracking, one-shot variant (priorities grow while a robot
          is away from its goal); stops when all robots are at their goals or after 4|V| steps.
* LaCAM - lazy constraints addition search over joint configurations, using PIBT as the
          configuration generator (complete).

Compact but faithful Python implementations, not the authors' C++ code (LaCAM* refinements,
e.g. anytime improvement, are not included)."""
import heapq
import math
import random
import time
from collections import deque

from .config import DEFAULT_CONFIG
from .costs import SocialCache


def social_cost_to_go(gmap, goal, cfg, cache):
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


class _Setup:
    def __init__(self, gmap, robots, humans, cfg, seed=0):
        self.g = gmap
        self.ids = [r["id"] for r in robots]
        self.starts = tuple(r["start"] for r in robots)
        self.goals = tuple(r["goal"] for r in robots)
        cache = SocialCache(gmap, humans, cfg)
        self.h = [social_cost_to_go(gmap, gl, cfg, cache) for gl in self.goals]
        self.nbrs = {v: [u for u, _ in gmap.adj[v]] for v in gmap.adj}
        self.rng = random.Random(seed)
        self.tie = {v: self.rng.random() for v in gmap.adj}      # fixed random tie-breaking
        self.n = len(robots)


def _pibt_step(S, Q, prio, forced=None):
    """One PIBT step from configuration Q. `forced` maps agent -> next node (LaCAM constraints).
    Returns the next configuration or None if the constraints cannot be met."""
    n = S.n
    nxt = [None] * n
    reserved = {}
    occ = {Q[i]: i for i in range(n)}
    if forced:
        for i, v in forced.items():
            if v != Q[i] and v not in S.nbrs[Q[i]]:
                return None
            if v in reserved:
                return None
            j = occ.get(v)
            if j is not None and j != i and j in forced and forced[j] == Q[i]:
                return None                  # forced swap
            nxt[i] = v; reserved[v] = i

    def func(i, banned=None):
        cands = [Q[i]] + S.nbrs[Q[i]]
        cands.sort(key=lambda v: (S.h[i].get(v, math.inf), S.tie[v]))
        for v in cands:
            if v in reserved or v == banned:
                continue
            j = occ.get(v)
            if j is not None and j != i and nxt[j] is not None and nxt[j] == Q[i]:
                continue                     # would swap with an agent that already decided
            nxt[i] = v; reserved[v] = i
            if j is not None and j != i and nxt[j] is None:
                if not func(j, banned=Q[i]):
                    continue                 # j stays on v; try the next candidate
            return True
        nxt[i] = Q[i]
        reserved[Q[i]] = i
        return False

    for i in sorted(range(n), key=lambda k: -prio[k]):
        if nxt[i] is None:
            func(i)
    conf = tuple(nxt)
    if len(set(conf)) < n:
        return None
    return conf


def _to_paths(S, configs):
    paths = {}
    for k, rid in enumerate(S.ids):
        p = [c[k] for c in configs]
        while len(p) > 1 and p[-1] == p[-2] == S.goals[k]:
            p.pop()
        paths[rid] = p
    return paths


def plan_pibt(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    S = _Setup(gmap, robots, humans, cfg)
    eps = [S.rng.random() for _ in range(S.n)]
    prio = list(eps)
    Q = S.starts
    configs = [Q]
    t0 = time.perf_counter()
    for _ in range(4 * gmap.n_nodes):
        if Q == S.goals:
            return _to_paths(S, configs), {"status": "solved"}
        if time.perf_counter() - t0 > cfg.cbs_time_budget_s:
            break
        prio = [prio[i] + 1 if Q[i] != S.goals[i] else eps[i] for i in range(S.n)]
        Q = _pibt_step(S, Q, prio)
        configs.append(Q)
    paths = _to_paths(S, configs)
    for k, rid in enumerate(S.ids):          # robots that never settled at their goal
        if configs[-1][k] != S.goals[k]:
            paths[rid] = []
    return paths, {"status": "timeout"}


def plan_lacam(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    S = _Setup(gmap, robots, humans, cfg)
    t0 = time.perf_counter()

    def order_of(Q):
        return sorted(range(S.n), key=lambda i: -S.h[i].get(Q[i], math.inf))

    def make(Q, parent):
        prio = [S.h[i].get(Q[i], 0.0) for i in range(S.n)]
        return {"Q": Q, "parent": parent, "order": order_of(Q), "prio": prio,
                "tree": deque([{"forced": {}, "depth": 0}])}

    root = make(S.starts, None)
    explored = {S.starts: root}
    open_ = [root]
    expansions = 0
    while open_:
        if time.perf_counter() - t0 > cfg.cbs_time_budget_s:
            paths, info = plan_pibt(gmap, robots, humans, cfg)      # best effort fallback
            return paths, {"status": "timeout", "expansions": expansions}
        N = open_[-1]
        if N["Q"] == S.goals:
            configs, x = [], N
            while x is not None:
                configs.append(x["Q"]); x = x["parent"]
            return _to_paths(S, configs[::-1]), {"status": "solved", "expansions": expansions}
        if not N["tree"]:
            open_.pop(); continue
        C = N["tree"].popleft(); expansions += 1
        if C["depth"] < S.n:
            i = N["order"][C["depth"]]
            cands = [N["Q"][i]] + S.nbrs[N["Q"][i]]
            S.rng.shuffle(cands)
            for v in cands:
                f = dict(C["forced"]); f[i] = v
                N["tree"].append({"forced": f, "depth": C["depth"] + 1})
        Q2 = _pibt_step(S, N["Q"], N["prio"], C["forced"])
        if Q2 is None:
            continue
        if Q2 in explored:
            open_.append(explored[Q2]); continue
        N2 = make(Q2, N)
        explored[Q2] = N2
        open_.append(N2)
    return {rid: [] for rid in S.ids}, {"status": "infeasible", "expansions": expansions}
