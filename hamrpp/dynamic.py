"""Closed-loop evaluation with moving people and a predictive (time-dependent) social cost.

Pedestrian model. Each person walks along the corridor graph towards a random waypoint (a free
node) at a constant personal speed (uniform 0.5-1.0 m per step), picks a new waypoint on arrival,
and does not react to robots (a common simplification; see the threats-to-validity section).
Positions are continuous along graph edges.

Robots move one node per step (synchronous), replan every step from their current nodes
(receding horizon, Sec. 4.5) and stay at their goals once they reach them.

Social cost variants (the coordination layer is the space-time prioritized planner throughout):
  snapshot    social cost from the positions observed now, constant over the plan (Eqs. 2-3)
  predictive  social cost at plan step k uses constant-velocity predictions p + k*v_hat, held
              constant after `pred_horizon` steps (velocity estimated from the last two
              observations)
  oracle      as predictive, but with the true future positions (upper reference; not realisable)
"""
import heapq
import itertools
import math
import time

import numpy as np

from .config import DEFAULT_CONFIG
from .costs import turn_cost, social_terms, social_terms_uncertain, SocialCache
from .cbs import find_conflicts
from .planners import social_cost_dijkstra


# ------------------------------------------------------------------ pedestrians
class Crowd:
    """pause_p / turn_p > 0 give the less predictable pedestrians of the robustness check: each
    step a person pauses with probability pause_p and re-draws a random waypoint with probability
    turn_p. With the defaults (0, 0) no extra random numbers are drawn."""

    def __init__(self, gmap, n, rng, speed=(0.5, 1.0), pause_p=0.0, turn_p=0.0, reactive=False):
        self.g, self.rng = gmap, rng
        self.pause_p, self.turn_p = pause_p, turn_p
        self.reactive = reactive
        self.robots_xy = []
        self._free = np.array(list(gmap.pos.values()), float)
        nodes = gmap.nodes
        self.speed = rng.uniform(speed[0], speed[1], n)
        self.pos = []
        self.route = []
        for i in range(n):
            s = int(rng.choice(nodes))
            self.pos.append(np.array(gmap.pos[s], float))
            self.route.append(self._new_route(s))
        self.prev = [p.copy() for p in self.pos]

    def _new_route(self, s):
        g = self.g
        goal = s
        while goal == s:
            goal = int(self.rng.choice(g.nodes))
        cfg = DEFAULT_CONFIG.with_(social_weight=0.0, turn_weight=0.0)
        path = social_cost_dijkstra(g, s, goal, [], cfg)
        return [np.array(g.pos[v], float) for v in path[1:]]

    def nearest_node(self, p):
        return min(self.g.pos, key=lambda v: (self.g.pos[v][0] - p[0]) ** 2 + (self.g.pos[v][1] - p[1]) ** 2)

    # ---- reactive (social force) pedestrians, Amendment 1 of ANALYSIS_PLAN.md
    SF_A, SF_B, SF_R_HUMAN, SF_R_ROBOT, SF_CAP, WALL_TOL, SLOW_DIST = 0.5, 0.3, 0.5, 0.6, 0.3, 0.5, 1.0

    def _advance(self, i, budget):
        p = self.pos[i].copy()
        while budget > 1e-9:
            if not self.route[i]:
                self.route[i] = self._new_route(self.nearest_node(p))
            tgt = self.route[i][0]
            d = float(np.linalg.norm(tgt - p))
            if d <= budget:
                p = tgt.copy(); self.route[i].pop(0); budget -= d
            else:
                p = p + (tgt - p) * (budget / d); budget = 0.0
        return p

    def _step_reactive(self, robots_xy):
        old = [p.copy() for p in self.pos]
        self.prev = [p.copy() for p in self.pos]
        R = [np.array(r, float) for r in (robots_xy or [])]
        for i in range(len(self.pos)):
            if not self.route[i]:
                self.route[i] = self._new_route(self.nearest_node(self.pos[i]))
            head = self.route[i][0] - old[i]
            nh = float(np.linalg.norm(head))
            head = head / nh if nh > 1e-9 else np.zeros(2)
            speed = self.speed[i]
            if any(float(np.linalg.norm(r - old[i])) < self.SLOW_DIST and float(np.dot(r - old[i], head)) > 0 for r in R):
                speed *= 0.5
            newp = self._advance(i, speed)
            f = np.zeros(2)
            for j in range(len(old)):
                if j != i:
                    d = old[i] - old[j]; n = float(np.linalg.norm(d))
                    if n > 1e-9:
                        f += self.SF_A * math.exp((self.SF_R_HUMAN - n) / self.SF_B) * d / n
            for r in R:
                d = old[i] - r; n = float(np.linalg.norm(d))
                if n > 1e-9:
                    f += self.SF_A * math.exp((self.SF_R_ROBOT - n) / self.SF_B) * d / n
            nf = float(np.linalg.norm(f))
            if nf > self.SF_CAP:
                f = f * (self.SF_CAP / nf)
            cand = newp + f
            if float(np.min(np.linalg.norm(self._free - cand, axis=1))) <= self.WALL_TOL:
                self.pos[i] = cand
            else:
                self.pos[i] = newp

    def step(self, robots_xy=None):
        if self.reactive:
            self.robots_xy = list(robots_xy or [])
            return self._step_reactive(self.robots_xy)
        self.prev = [p.copy() for p in self.pos]
        for i in range(len(self.pos)):
            if self.turn_p > 0 and self.rng.random() < self.turn_p:
                self.route[i] = self._new_route(self.nearest_node(self.pos[i]))
            if self.pause_p > 0 and self.rng.random() < self.pause_p:
                continue
            budget = self.speed[i]
            while budget > 1e-9:
                if not self.route[i]:
                    self.route[i] = self._new_route(self.nearest_node(self.pos[i]))
                tgt = self.route[i][0]
                d = float(np.linalg.norm(tgt - self.pos[i]))
                if d <= budget:
                    self.pos[i] = tgt.copy(); self.route[i].pop(0); budget -= d
                else:
                    self.pos[i] = self.pos[i] + (tgt - self.pos[i]) * (budget / d); budget = 0.0

    def observed(self):
        return [tuple(p) for p in self.pos]

    def velocities(self):
        return [tuple(p - q) for p, q in zip(self.pos, self.prev)]

    def clone_future(self, k):
        """True positions for the next k steps (used only by the oracle variant)."""
        import copy
        c = copy.deepcopy(self)
        out = []
        for _ in range(k):
            c.step(self.robots_xy); out.append(c.observed())   # reactive: robots held where they are
        return out


# ------------------------------------------------------------------ time-dependent social cost
class TimeSocial:
    """social(u, v, t): W_social + W_comfort of moving u->v during plan step t -> t+1, evaluated
    against the human positions predicted for time t+1."""

    def __init__(self, gmap, frames, cfg):
        self.g, self.frames, self.cfg = gmap, frames, cfg
        self.memo = {}

    def __call__(self, u, v, t):
        k = min(t + 1, len(self.frames) - 1)
        key = (u, v, k)
        val = self.memo.get(key)
        if val is None:
            s, c = social_terms(self.g.pos[u], self.g.pos[v], self.frames[k], self.cfg.sigma, self.cfg.personal_space, self.cfg.human_weight)
            val = s + c
            self.memo[key] = val
        return val


class TimeSocialUncertain(TimeSocial):
    """Time-dependent uncertainty-aware social cost (Amendment 2): frame k uses s_k = s * k."""

    def __init__(self, gmap, frames, cfg, s_per_step=0.3):
        super().__init__(gmap, frames, cfg)
        self.s = s_per_step

    def __call__(self, u, v, t):
        k = min(t + 1, len(self.frames) - 1)
        key = (u, v, k)
        val = self.memo.get(key)
        if val is None:
            a, b = social_terms_uncertain(self.g.pos[u], self.g.pos[v], self.frames[k], self.s * k,
                                          self.cfg.sigma, self.cfg.personal_space, self.cfg.human_weight)
            val = a + b
            self.memo[key] = val
        return val


UNC_S_PER_STEP = 0.3   # fixed a priori in ANALYSIS_PLAN.md, Amendment 2


def frames_for(variant, crowd, cfg, pred_horizon):
    now = crowd.observed()
    if variant == "snapshot":
        return [now]
    if variant in ("predictive", "predictive_unc"):
        vel = crowd.velocities()
        return [[(x + k * vx, y + k * vy) for (x, y), (vx, vy) in zip(now, vel)] for k in range(pred_horizon + 1)]
    if variant == "oracle":
        return [now] + crowd.clone_future(pred_horizon)
    if variant == "unaware":
        return [[]]
    raise ValueError(variant)


def _low_level_td(gmap, start, goal, cfg, soc, vcons, econs, horizon, last_goal_con, allowed=None):
    pos = gmap.pos
    if (start, 0) in vcons or (allowed is not None and (start not in allowed or goal not in allowed)):
        return []
    dist = {(start, 0): 0.0}; parent = {(start, 0): None}
    pq = [(0.0, 0, start, None)]
    while pq:
        d, t, u, pn = heapq.heappop(pq)
        if d > dist.get((u, t), math.inf):
            continue
        if u == goal and t > last_goal_con:
            path, s = [], (u, t)
            while s is not None:
                path.append(s[0]); s = parent[s]
            return path[::-1]
        if t >= horizon:
            continue
        pp = pos[pn] if pn is not None else None
        for v, w in [(u, 1.0)] + list(gmap.adj[u]):
            nt = t + 1
            if (v, nt) in vcons or (u, v, t) in econs or (allowed is not None and v not in allowed):
                continue
            if v == u:
                cost = 1.0 + cfg.social_weight * soc(u, u, t); npn = pn
            else:
                cost = w + cfg.turn_weight * turn_cost(pp, pos[u], pos[v], cfg.turn_noise_deg) + cfg.social_weight * soc(u, v, t); npn = u
            nd = d + cost
            if nd < dist.get((v, nt), math.inf):
                dist[(v, nt)] = nd; parent[(v, nt)] = (u, t)
                heapq.heappush(pq, (nd, nt, v, npn))
    return []


def plan_spacetime_td(gmap, robots, soc, cfg, allowed=None, max_restarts=3):
    """Space-time prioritized planning (as in spacetime.py) with a time-dependent social cost.
    Robots that still cannot be planned after the restarts are held in place for this cycle and
    reserved as static obstacles, and the others are replanned around them."""
    order = [r["id"] for r in sorted(robots, key=lambda r: -r["priority"])]
    rob = {r["id"]: r for r in robots}
    best = None
    held = []
    for attempt in range(max_restarts + 1 + len(robots)):
        vres, eres, goal_from, paths = set(), set(), {}, {}
        for rid in held:                      # last pass: stuck robots are static obstacles
            goal_from[rob[rid]["start"]] = 0
            paths[rid] = [rob[rid]["start"]]
        for rid in [x for x in order if x not in held]:
            r = rob[rid]
            horizon = 2 * gmap.n_nodes + max([t for (_, t) in vres] + [0])
            vc = set(vres)
            for gnode, ta in goal_from.items():
                for t in range(ta, horizon + 1):
                    vc.add((gnode, t))
            last = max([t for (v, t) in vc if v == r["goal"]], default=-1)
            p = _low_level_td(gmap, r["start"], r["goal"], cfg, soc, vc, eres, horizon, last, allowed)
            paths[rid] = p
            if not p:
                continue
            for t, v in enumerate(p):
                vres.add((v, t))
            for t in range(len(p) - 1):
                if p[t] != p[t + 1]:
                    eres.add((p[t + 1], p[t], t))
            goal_from[p[-1]] = len(p) - 1
        failed = [rid for rid in order if not paths[rid]]
        if not failed:
            return paths                      # held robots (if any) have a one-node "stay" path
        if held or attempt >= max_restarts:
            held = held + [x for x in failed if x not in held]   # hold until nobody fails
            continue
        if best is None or len(failed) < sum(1 for v in best.values() if not v):
            best = paths
        order = [failed[0]] + [x for x in order if x != failed[0]]
    return best


# ------------------------------------------------------------------ closed loop
VARIANTS = ["unaware", "hard_exclusion", "snapshot", "predictive", "oracle", "lacam_snapshot"]


def closed_loop(gmap, robots, n_humans, variant, seed, cfg=DEFAULT_CONFIG, pred_horizon=6, max_steps=60,
                pause_p=0.0, turn_p=0.0, reactive=False):
    rng = np.random.default_rng(seed)
    crowd = Crowd(gmap, n_humans, rng, pause_p=pause_p, turn_p=turn_p, reactive=reactive)
    crowd.step([gmap.pos[r["start"]] for r in robots])   # one step so that velocities are defined
    cur = {r["id"]: r["start"] for r in robots}
    done_at = {r["id"]: None for r in robots}
    min_clear = {r["id"]: math.inf for r in robots}
    intrusions = {r["id"]: 0 for r in robots}
    robot_steps = {r["id"]: 0 for r in robots}
    travelled = {r["id"]: 0 for r in robots}
    collisions, plan_ms, stuck_cycles = 0, [], 0
    collided_pairs = set()
    for step in range(max_steps):
        # measure current clearance (robots that have not finished)
        hum = crowd.observed()
        for rid, v in cur.items():
            if done_at[rid] is None:
                x, y = gmap.pos[v]
                d = min(math.hypot(x - hx, y - hy) for hx, hy in hum)
                min_clear[rid] = min(min_clear[rid], d)
                robot_steps[rid] += 1
                if d < cfg.personal_space:
                    intrusions[rid] += 1
        if all(done_at[rid] is not None for rid in cur):
            break
        sub = [{"id": r["id"], "start": cur[r["id"]], "goal": r["goal"], "priority": r["priority"]} for r in robots]
        t0 = time.perf_counter()
        if variant == "lacam_snapshot":
            from .lacam import plan_lacam
            paths, _ = plan_lacam(gmap, sub, hum, cfg)
        else:
            allowed = None
            frames = frames_for("snapshot" if variant == "hard_exclusion" else variant, crowd, cfg, pred_horizon)
            c = cfg
            if variant in ("unaware", "hard_exclusion"):
                c = cfg.with_(social_weight=0.0); frames = [[]]
            if variant == "hard_exclusion":
                allowed = {v for v, (x, y) in gmap.pos.items() if all(math.hypot(x - hx, y - hy) > cfg.personal_space for hx, hy in hum)}
                allowed |= {cur[r["id"]] for r in robots}   # a robot may always stay where it is
            soc = TimeSocialUncertain(gmap, frames, c, UNC_S_PER_STEP) if variant == "predictive_unc" else TimeSocial(gmap, frames, c)
            paths = plan_spacetime_td(gmap, sub, soc, c, allowed)
        plan_ms.append((time.perf_counter() - t0) * 1000.0)
        nxt = {}
        for rid, v in cur.items():
            p = paths.get(rid) or []
            nxt[rid] = p[1] if len(p) > 1 else v
            if len(p) <= 1 and done_at[rid] is None:
                stuck_cycles += 1
        # execute and check robot-robot conflicts of the executed step
        for a, b in itertools.combinations(cur, 2):
            if nxt[a] == nxt[b] or (nxt[a] == cur[b] and nxt[b] == cur[a] and nxt[a] != cur[a]):
                collided_pairs.add((a, b)); collisions += 1
        for rid in cur:
            if nxt[rid] != cur[rid]:
                travelled[rid] += 1
            cur[rid] = nxt[rid]
            if done_at[rid] is None and cur[rid] == next(r["goal"] for r in robots if r["id"] == rid):
                done_at[rid] = step + 1
        crowd.step([gmap.pos[v] for v in cur.values()])
    n = len(robots)
    reached = [rid for rid in cur if done_at[rid] is not None]
    total_steps = sum(robot_steps.values())
    return {
        "variant": variant, "n_robots": n, "n_humans": n_humans, "seed": seed,
        "success_rate": len(reached) / n,
        "all_reached": int(len(reached) == n),
        "mean_time_to_goal": float(np.mean([done_at[r] for r in reached])) if reached else math.nan,
        "mean_moves": float(np.mean([travelled[r] for r in reached])) if reached else math.nan,
        "mean_min_clearance": float(np.mean([min_clear[r] for r in cur if min_clear[r] < math.inf])),
        "worst_min_clearance": float(min(min_clear.values())),
        "intrusion_steps": int(sum(intrusions.values())),
        "intrusion_rate": sum(intrusions.values()) / total_steps if total_steps else math.nan,
        "robots_with_intrusion": int(sum(1 for r in intrusions if intrusions[r] > 0)),
        "collision_events": collisions, "collision_pairs": len(collided_pairs),
        "mean_plan_ms": float(np.mean(plan_ms)) if plan_ms else math.nan,
        "max_plan_ms": float(np.max(plan_ms)) if plan_ms else math.nan,
        "stuck_cycles": stuck_cycles,
    }
