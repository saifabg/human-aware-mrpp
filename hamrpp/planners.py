"""Social-cost Dijkstra (Algorithm 1), priority coordination (Algorithm 2) and the two Dijkstra
baselines (classic, hard-exclusion dynamic obstacle). See Sec. 4 and 5.3 of the manuscript."""
import heapq
import math
import time

from .config import DEFAULT_CONFIG
from .costs import turn_cost, social_terms


def social_cost_dijkstra(gmap, start, goal, humans, cfg, excluded=None, allowed=None):
    """Algorithm 1. Single-robot Dijkstra under edge cost Eq. (1).

    excluded(v, t) -> bool implements the coordination check (Algorithm 2, line 3); t is the hop
    count at which the robot would reach v. `allowed` optionally restricts the node set (used by
    the hard-exclusion baseline). Returns the node path [start, ..., goal] or [] if unreachable.

    Note: the turn term depends on the predecessor of u, so the search is exact for the distance
    and social terms and approximate for the turn term (Sec. 4.3)."""
    if allowed is not None and (start not in allowed or goal not in allowed):
        return []
    pos = gmap.pos
    dist = {start: 0.0}
    prev = {start: None}
    hop = {start: 0}
    pq = [(0.0, 0, start)]
    counter = 1
    done = set()
    while pq:
        d, _, u = heapq.heappop(pq)
        if u in done or d > dist.get(u, math.inf):
            continue
        done.add(u)
        if u == goal:
            break
        pu = pos[u]
        pp = pos[prev[u]] if prev[u] is not None else None
        t_next = hop[u] + 1
        for v, w in gmap.adj[u]:
            if v in done:
                continue
            if allowed is not None and v not in allowed:
                continue
            if excluded is not None and excluded(v, t_next):
                continue
            pv = pos[v]
            w_turn = turn_cost(pp, pu, pv, cfg.turn_noise_deg)
            if cfg.social_weight and humans:
                s, c = social_terms(pu, pv, humans, cfg.sigma, cfg.personal_space, cfg.human_weight)
                w_soc = s + c
            else:
                w_soc = 0.0
            cost = w + cfg.turn_weight * w_turn + cfg.social_weight * w_soc
            nd = d + cost
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                hop[v] = t_next
                heapq.heappush(pq, (nd, counter, v)); counter += 1
    if goal not in done:
        return []
    path, x = [], goal
    while x is not None:
        path.append(x); x = prev[x]
    return path[::-1]


def priority_coordination(gmap, robots, humans, cfg, allowed=None):
    """Algorithm 2. Plans robots in decreasing priority. `robots` is a list of dicts with keys
    id, start, goal, priority. Returns {robot id: path or []}.

    occupied[r] keeps only the LAST node of each planned path (the known weakness, Sec. 4.4);
    reserved[(v, tau)] marks v for tau in t_k - 1 .. t_k + 1 along each planned path."""
    order = sorted(robots, key=lambda r: -r["priority"])  # stable for ties
    occupied = {}
    reserved = {}
    W = cfg.exclusion_window
    paths = {}
    for r in order:
        def excluded(v, t, rid=r["id"]):
            for other, node in occupied.items():
                if other != rid and node == v:
                    return True
            for tau in range(t - W, t + W + 1):
                owner = reserved.get((v, tau))
                if owner is not None and owner != rid:
                    return True
            return False
        path = social_cost_dijkstra(gmap, r["start"], r["goal"], humans, cfg, excluded, allowed)
        paths[r["id"]] = path
        if not path:
            continue
        for k, v in enumerate(path):
            for tau in range(k - cfg.reserve_halfwidth, k + cfg.reserve_halfwidth + 1):
                reserved.setdefault((v, tau), r["id"])
        occupied[r["id"]] = path[-1]
    return paths


def plan_classic(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    """Baseline 1: each robot planned independently, no humans (social weight zero), no coordination."""
    c = cfg.with_(social_weight=0.0)
    return {r["id"]: social_cost_dijkstra(gmap, r["start"], r["goal"], [], c) for r in robots}, {}


def hard_exclusion_nodes(gmap, humans, r_p):
    return {v for v, (x, y) in gmap.pos.items()
            if all(math.hypot(x - hx, y - hy) > r_p for hx, hy in humans)}


def plan_hard_exclusion(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    """Baseline 2 (dynamic obstacle): same coordination, but every node within r_p of a human is
    removed from the graph; no graded social cost."""
    allowed = hard_exclusion_nodes(gmap, humans, cfg.personal_space)
    c = cfg.with_(social_weight=0.0)
    return priority_coordination(gmap, robots, [], c, allowed=allowed), {}


def plan_human_aware(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    """Proposed planner: graded social cost (Eqs. 1-3) + priority coordination (Algorithm 2)."""
    return priority_coordination(gmap, robots, humans, cfg), {}


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, (time.perf_counter() - t0) * 1000.0


def _lazy_cbs(*a, **k):
    from .cbs import plan_cbs
    return plan_cbs(*a, **k)


def _lazy_lacam(name):
    def f(*a, **k):
        from . import lacam
        return getattr(lacam, name)(*a, **k)
    return f


def _lazy_modern(name):
    def f(*a, **k):
        from . import modern
        return getattr(modern, name)(*a, **k)
    return f


def _lazy_st(name):
    def f(*a, **k):
        from . import spacetime
        return getattr(spacetime, name)(*a, **k)
    return f


METHODS = {
    "classic": plan_classic,
    "hard_exclusion": plan_hard_exclusion,          # original reservation scheme (v1)
    "human_aware": plan_human_aware,                # original reservation scheme (v1)
    "hard_exclusion_st": _lazy_st("plan_hard_exclusion_st"),
    "unaware_st": _lazy_st("plan_unaware_st"),
    "human_aware_st": _lazy_st("plan_human_aware_st"),  # proposed (revised)
    "cbs": _lazy_cbs,
    "pbs": _lazy_modern("plan_pbs"),
    "ecbs": _lazy_modern("plan_ecbs"),
    "pibt": _lazy_lacam("plan_pibt"),
    "lacam": _lazy_lacam("plan_lacam"),
}

METHOD_LABELS = {
    "classic": "Classic Dijkstra",
    "hard_exclusion": "Hard exclusion (v1 reservations)",
    "human_aware": "Human-aware (v1 reservations)",
    "hard_exclusion_st": "Hard exclusion (space-time)",
    "unaware_st": "No social cost (space-time)",
    "human_aware_st": "Human-aware (space-time, proposed)",
    "cbs": "CBS + social cost",
    "pbs": "PBS + social cost",
    "ecbs": "ECBS(1.5) + social cost",
    "pibt": "PIBT + social cost",
    "lacam": "LaCAM + social cost",
}
