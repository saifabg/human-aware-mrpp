"""Shared helpers for the experiment scripts."""
import csv
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from hamrpp import DEFAULT_CONFIG  # noqa: E402
from hamrpp.planners import METHODS  # noqa: E402
from hamrpp.metrics import robot_rows, collision_pairs  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
RAW = os.path.join(RESULTS, "raw")
TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")
for d in (RAW, TABLES, FIGS):
    os.makedirs(d, exist_ok=True)

# main comparison: classic, hard_exclusion_st, human_aware_st (proposed), cbs
# ablations: human_aware / hard_exclusion with the original (v1) reservations, unaware_st
METHOD_ORDER = ["classic", "hard_exclusion", "human_aware", "hard_exclusion_st", "unaware_st", "human_aware_st", "cbs", "pbs", "ecbs", "pibt", "lacam"]
MAIN_METHODS = ["classic", "hard_exclusion_st", "human_aware_st", "cbs", "pbs", "ecbs", "pibt", "lacam"]


def run_method(gmap, robots, humans, method, cfg=DEFAULT_CONFIG):
    """Plan one scenario with one method. Returns (robot rows, trial row)."""
    t0 = time.perf_counter()
    paths, info = METHODS[method](gmap, robots, humans, cfg)
    ms = (time.perf_counter() - t0) * 1000.0
    rows = robot_rows(gmap, paths, robots, humans, cfg)
    valid = [r for r in rows if r["feasible"]]
    mean = lambda k: float(np.mean([r[k] for r in valid])) if valid else math.nan
    n_col = collision_pairs(paths)
    trial = {
        "method": method, "n_robots": len(robots), "n_humans": len(humans),
        "n_valid": len(valid), "n_infeasible": len(rows) - len(valid),
        "any_infeasible": int(len(valid) < len(rows)),
        "mean_path_len": mean("path_len"), "mean_turns": mean("turns"), "mean_min_human_dist": mean("min_human_dist"),
        "collisions": n_col, "any_collision": int(n_col > 0),
        "plan_ms": ms, "cbs_status": info.get("status", ""), "cbs_expansions": info.get("expansions", ""),
    }
    return rows, trial


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("",) else math.nan
            except (TypeError, ValueError):
                pass
    return rows


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def environment():
    import scipy, matplotlib, pandas
    info = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": platform.processor(), "machine": platform.machine(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "pandas": pandas.__version__,
        "config": DEFAULT_CONFIG.__dict__,
    }
    try:
        import os as _os
        info["cpu_count"] = _os.cpu_count()
    except Exception:
        pass
    return info
