"""All tunable parameters in one place (values as reported in the manuscript)."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PlannerConfig:
    # edge-cost weights, Eq. (1)
    turn_weight: float = 1.0          # lambda_turn
    social_weight: float = 1.0        # lambda_social
    # social cost, Eqs. (2)-(3)
    sigma: float = 0.8                # Gaussian spread (m)
    personal_space: float = 1.0       # r_p (m)
    human_weight: float = 1.0         # alpha_i (uniform)
    # turn cost (Sec. 4.3)
    turn_noise_deg: float = 10.0      # angles below this cost zero
    turn_count_threshold: float = 0.1 # a "turn" is counted when turn cost > 0.1 (Sec. 5.2)
    # priority / reservation coordination (Sec. 4.4)
    reserve_halfwidth: int = 1        # each path node reserved for t_k-1 .. t_k+1 (three timesteps)
    exclusion_window: int = 2         # a node is excluded if reserved within +-2 timesteps
    # CBS baseline (Sec. 5.3)
    cbs_max_expansions: int = 150
    cbs_time_budget_s: float = 2.0
    cbs_fallback: str = "lowest_cost"  # "lowest_cost" (as in the paper) or "fewest_conflicts"

    def with_(self, **kw):
        return replace(self, **kw)


DEFAULT_CONFIG = PlannerConfig()
