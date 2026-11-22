"""Human-aware multi-robot path planning with a graded (proxemic) social cost.

Reference implementation for the manuscript "Human-Aware Multi-Robot Path Planning with
Uncertainty-Aware Prediction of Pedestrian Motion" (Abgenah, Aldrawi and Juma).
"""
from .config import PlannerConfig, DEFAULT_CONFIG
from .maps import GridMap, map1, map2
from .planners import plan_classic, plan_hard_exclusion, plan_human_aware, METHODS
from .cbs import plan_cbs

__version__ = "1.0.0"
__all__ = ["PlannerConfig", "DEFAULT_CONFIG", "GridMap", "map1", "map2", "plan_classic",
           "plan_hard_exclusion", "plan_human_aware", "plan_cbs", "METHODS"]
