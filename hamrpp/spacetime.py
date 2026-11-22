"""Space-time prioritized coordination (the improved coordination layer; Sec. 4.4 of the revised
manuscript). This is the fix that the original paper proposed as future work: instead of keeping
only the last node of each higher-priority path and a short reservation window, every
higher-priority path is reserved in full in a space-time reservation table (cooperative
pathfinding, Silver 2005):

  * vertex reservations (v, t) for every step of the path,
  * the goal is reserved from the arrival time onwards (robots wait at their goals),
  * swap reservations: if a higher-priority robot moves a->b between t and t+1, the move b->a in
    the same step is forbidden.

Each robot is then planned with a time-expanded Dijkstra search (wait actions allowed) under the
same edge cost as Eq. (1). If a robot cannot be planned, it is moved to the front of the planning
order and all robots are replanned (at most `max_restarts` times); robots still without a path
are reported infeasible.

By construction, the returned paths of feasible robots have no vertex or swap conflicts with each
other (under the synchronous, wait-at-goal motion model used for evaluation)."""
from .config import DEFAULT_CONFIG
from .costs import SocialCache
from .cbs import low_level
from .planners import hard_exclusion_nodes


def _plan_order(gmap, robots, order, cfg, cache, allowed):
    vres, eres = set(), set()
    goal_from = {}          # goal node -> arrival time of the robot that parks there
    paths = {}
    horizon_base = 2 * gmap.n_nodes
    for rid in order:
        r = next(x for x in robots if x["id"] == rid)
        max_t = max([t for (_, t) in vres] + [0])
        horizon = horizon_base + max_t
        # parked robots block their goal from arrival to the horizon
        vc = set(vres)
        for g, t_arr in goal_from.items():
            for t in range(t_arr, horizon + 1):
                vc.add((g, t))
        last_goal = max([t for (v, t) in vc if v == r["goal"]], default=-1)
        if r["goal"] in goal_from:   # another robot parks on my goal for ever
            paths[rid] = []
            continue
        p, _ = low_level(gmap, r["start"], r["goal"], cfg, cache, vc, eres, horizon, allowed, last_goal)
        paths[rid] = p
        if not p:
            continue
        for t, v in enumerate(p):
            vres.add((v, t))
        for t in range(len(p) - 1):
            a, b = p[t], p[t + 1]
            if a != b:
                eres.add((b, a, t))
        goal_from[p[-1]] = len(p) - 1
    return paths


def _spacetime(gmap, robots, humans, cfg, allowed=None, max_restarts=3):
    cache = SocialCache(gmap, humans, cfg)
    order = [r["id"] for r in sorted(robots, key=lambda r: -r["priority"])]
    best = None
    restarts = 0
    for attempt in range(max_restarts + 1):
        paths = _plan_order(gmap, robots, order, cfg, cache, allowed)
        failed = [rid for rid in order if not paths[rid]]
        if best is None or len(failed) < sum(1 for v in best.values() if not v):
            best = paths
        if not failed:
            break
        restarts += 1
        first = failed[0]
        order = [first] + [x for x in order if x != first]
    return best, {"restarts": restarts}


def plan_human_aware_st(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    """Proposed planner (revised): graded social cost + space-time prioritized coordination."""
    return _spacetime(gmap, robots, humans, cfg)


def plan_hard_exclusion_st(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    """Hard-exclusion baseline with the same space-time coordination (no graded cost)."""
    allowed = hard_exclusion_nodes(gmap, humans, cfg.personal_space)
    return _spacetime(gmap, robots, [], cfg.with_(social_weight=0.0), allowed=allowed)


def plan_unaware_st(gmap, robots, humans, cfg=DEFAULT_CONFIG):
    """Ablation: space-time coordination with no social cost (humans ignored)."""
    return _spacetime(gmap, robots, [], cfg.with_(social_weight=0.0))
