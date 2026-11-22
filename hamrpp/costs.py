"""Edge cost terms: turn cost and the graded social cost, Eqs. (1)-(3)."""
import math


def turn_cost(p_prev, p_u, p_v, noise_deg=10.0):
    """Angle between incoming (prev->u) and outgoing (u->v) directions, in units of 90 degrees.
    Angles below `noise_deg` cost zero (Sec. 4.3). Returns 0 when there is no predecessor."""
    if p_prev is None or p_u == p_v or p_prev == p_u:
        return 0.0
    ax, ay = p_u[0] - p_prev[0], p_u[1] - p_prev[1]
    bx, by = p_v[0] - p_u[0], p_v[1] - p_u[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    ang = math.degrees(math.acos(c))
    return 0.0 if ang < noise_deg else ang / 90.0


def edge_samples(p_u, p_v):
    return (p_u, ((p_u[0] + p_v[0]) / 2.0, (p_u[1] + p_v[1]) / 2.0), p_v)


def social_terms(p_u, p_v, humans, sigma=0.8, r_p=1.0, alpha=1.0):
    """Returns (W_social, W_comfort) for edge u->v against human positions (Eqs. 2-3)."""
    if not humans:
        return 0.0, 0.0
    pts = edge_samples(p_u, p_v)
    w_soc = 0.0
    w_com = 0.0
    two_s2 = 2.0 * sigma * sigma
    for hx, hy in humans:
        d2min = min((px - hx) ** 2 + (py - hy) ** 2 for px, py in pts)
        w_soc += alpha * math.exp(-d2min / two_s2)
        for px, py in pts:
            d = math.hypot(px - hx, py - hy)
            if d < r_p:
                w_com += (r_p - d) ** 2
    return w_soc, w_com


def social_terms_uncertain(p_u, p_v, humans, s, sigma=0.8, r_p=1.0, alpha=1.0):
    """Uncertainty-aware social cost (ANALYSIS_PLAN.md, Amendment 2). Each predicted human position
    is treated as N(mu, s^2 I). Proximity: exact expectation of the Gaussian term,
    alpha * sigma^2/(sigma^2+s^2) * exp(-d^2 / (2 (sigma^2+s^2))). Comfort: Eq. (3) with the radius
    inflated by one standard deviation, r_p + s. With s = 0 this equals social_terms()."""
    if not humans:
        return 0.0, 0.0
    pts = edge_samples(p_u, p_v)
    var = sigma * sigma + s * s
    scale = alpha * sigma * sigma / var
    r = r_p + s
    w_soc, w_com = 0.0, 0.0
    for hx, hy in humans:
        d2min = min((px - hx) ** 2 + (py - hy) ** 2 for px, py in pts)
        w_soc += scale * math.exp(-d2min / (2.0 * var))
        for px, py in pts:
            d = math.hypot(px - hx, py - hy)
            if d < r:
                w_com += (r - d) ** 2
    return w_soc, w_com


def social_field(x, y, humans, sigma=0.8, alpha=1.0):
    """Proximity term evaluated at a point (used for the heatmap, Fig. 5)."""
    return sum(alpha * math.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * sigma * sigma)) for hx, hy in humans)


class SocialCache:
    """W_social + W_comfort for every directed edge (and for waiting at a node), computed once per
    scenario. The CBS baseline uses this cache; the priority planners compute on demand."""

    def __init__(self, gmap, humans, cfg):
        self.vals = {}
        for u, nbrs in gmap.adj.items():
            pu = gmap.pos[u]
            s, c = social_terms(pu, pu, humans, cfg.sigma, cfg.personal_space, cfg.human_weight)
            self.vals[(u, u)] = s + c
            for v, _ in nbrs:
                s, c = social_terms(pu, gmap.pos[v], humans, cfg.sigma, cfg.personal_space, cfg.human_weight)
                self.vals[(u, v)] = s + c

    def __call__(self, u, v):
        return self.vals[(u, v)]
