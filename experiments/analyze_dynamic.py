"""Analysis of the closed-loop (moving people) study, exactly as specified in ANALYSIS_PLAN.md.

    python experiments/analyze_dynamic.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, TABLES, FIGS, read_csv, write_csv, save_json  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hamrpp.stats import wilcoxon_p, mcnemar_exact_p, holm, bootstrap_ci  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "savefig.dpi": 600})
W = 6.3
VARIANTS = ["unaware", "hard_exclusion", "snapshot", "predictive", "oracle", "lacam_snapshot"]
LABEL = {"unaware": "No social cost", "hard_exclusion": "Hard exclusion", "snapshot": "Snapshot social cost",
         "predictive": "Constant-velocity prediction", "oracle": "Oracle (true future, reference)",
         "lacam_snapshot": "LaCAM + snapshot social cost"}
COL = {"unaware": "#9d9d9d", "hard_exclusion": "#4c78a8", "snapshot": "#f2a3a4", "predictive": "#e45756",
       "oracle": "#54a24b", "lacam_snapshot": "#72b7b2"}
METRICS = [("intrusion_rate", "Intrusion rate (robot-steps inside r_p)", 3), ("mean_min_clearance", "Mean minimum clearance (m)", 2),
           ("mean_time_to_goal", "Time to goal (steps)", 2), ("success_rate", "Robots reaching goal (fraction)", 3),
           ("mean_plan_ms", "Planning time per cycle (ms)", 1), ("collision_events", "Robot-robot collisions", 2)]
FAMILY = ["snapshot", "lacam_snapshot", "unaware", "hard_exclusion"]   # predictive vs each (Holm family)
OUT = {}


def key(r):
    return (r["map"], int(r["n_robots"]), int(r["n_humans"]), int(r["trial"]))


def paired(rows, a, b, m):
    A = {key(r): r[m] for r in rows if r["variant"] == a}
    B = {key(r): r[m] for r in rows if r["variant"] == b}
    ks = sorted(set(A) & set(B))
    return np.array([A[k] for k in ks], float), np.array([B[k] for k in ks], float)


def fmt_p(p):
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def summary_table(rows, name):
    out = []
    for v in VARIANTS:
        rr = [r for r in rows if r["variant"] == v]
        if not rr:
            continue
        d = {"variant": v, "label": LABEL[v], "n_scenarios": len(rr)}
        for m, _, _ in METRICS:
            vals = [r[m] for r in rr]
            d[m] = float(np.nanmean(vals))
            d[m + "_lo"], d[m + "_hi"] = bootstrap_ci([[x] for x in vals])
        d["all_reached_pct"] = 100 * float(np.mean([r["all_reached"] for r in rr]))
        out.append(d)
    write_csv(os.path.join(TABLES, f"{name}_summary.csv"), out)
    with open(os.path.join(TABLES, f"{name}_summary.md"), "w", encoding="utf-8") as f:
        f.write("| Variant | n | " + " | ".join(lbl for _, lbl, _ in METRICS) + " | All robots reached (%) |\n")
        f.write("|---|---|" + "---|" * len(METRICS) + "---|\n")
        for d in out:
            cells = [f"{d[m]:.{k}f} [{d[m + '_lo']:.{k}f}, {d[m + '_hi']:.{k}f}]" for m, _, k in METRICS]
            f.write(f"| {d['label']} | {d['n_scenarios']} | " + " | ".join(cells) + f" | {d['all_reached_pct']:.1f} |\n")
    return out


def tests(rows, name):
    res = {}
    for m in ("intrusion_rate", "mean_min_clearance", "mean_time_to_goal"):
        ps, entries = [], []
        for b in FAMILY:
            x, y = paired(rows, "predictive", b, m)
            ok = ~(np.isnan(x) | np.isnan(y))
            p, n = wilcoxon_p(x, y)
            diff = x[ok] - y[ok]
            rel = 100 * (np.mean(x[ok]) - np.mean(y[ok])) / np.mean(y[ok]) if np.mean(y[ok]) else math.nan
            entries.append({"comparison": f"predictive vs {b}", "metric": m, "n": n, "p": p,
                            "median_diff": float(np.median(diff)) if len(diff) else math.nan,
                            "mean_pred": float(np.mean(x[ok])), "mean_other": float(np.mean(y[ok])), "rel_change_pct": float(rel),
                            "predictive_better_pct": float(100 * np.mean(diff < 0 if m != "mean_min_clearance" else diff > 0)) if len(diff) else math.nan})
            ps.append(p)
        for e, q in zip(entries, holm(ps)):
            e["p_holm"] = q
        # oracle: descriptive
        x, y = paired(rows, "oracle", "predictive", m)
        ok = ~(np.isnan(x) | np.isnan(y))
        entries.append({"comparison": "oracle vs predictive (descriptive)", "metric": m, "n": int(ok.sum()),
                        "p": wilcoxon_p(x, y)[0], "p_holm": math.nan,
                        "median_diff": float(np.median(x[ok] - y[ok])), "mean_pred": float(np.mean(y[ok])), "mean_other": float(np.mean(x[ok])),
                        "rel_change_pct": float(100 * (np.mean(x[ok]) - np.mean(y[ok])) / np.mean(y[ok])), "predictive_better_pct": math.nan})
        res[m] = entries
    # binary: all robots reached
    bins = []
    ps = []
    for b in FAMILY:
        x, y = paired(rows, "predictive", b, "all_reached")
        p, disc = mcnemar_exact_p(x, y); ps.append(p)
        bins.append({"comparison": f"predictive vs {b}", "metric": "all_reached", "p": p, "discordant": disc,
                     "rate_pred": float(np.mean(x)), "rate_other": float(np.mean(y))})
    for e, q in zip(bins, holm(ps)):
        e["p_holm"] = q
    res["all_reached"] = bins
    flat = [e for v in res.values() for e in v]
    write_csv(os.path.join(TABLES, f"{name}_tests.csv"), flat)
    with open(os.path.join(TABLES, f"{name}_tests.md"), "w", encoding="utf-8") as f:
        f.write("| Comparison | Metric | n | Mean (predictive) | Mean (other) | Relative change (%) | p | p (Holm) |\n|---|---|---|---|---|---|---|---|\n")
        for e in flat:
            if e["metric"] == "all_reached":
                f.write(f"| {e['comparison']} | all robots reached | — | {e['rate_pred']:.3f} | {e['rate_other']:.3f} | — | {fmt_p(e['p'])} | {fmt_p(e['p_holm'])} |\n")
            else:
                ph = "—" if math.isnan(e["p_holm"]) else fmt_p(e["p_holm"])
                f.write(f"| {e['comparison']} | {e['metric']} | {e['n']} | {e['mean_pred']:.3f} | {e['mean_other']:.3f} | {e['rel_change_pct']:+.1f} | {fmt_p(e['p'])} | {ph} |\n")
    return res


def per_cell(rows, name):
    out = []
    cells = sorted({(r["map"], int(r["n_robots"]), int(r["n_humans"])) for r in rows})
    for c in cells:
        rr = [r for r in rows if (r["map"], int(r["n_robots"]), int(r["n_humans"])) == c]
        d = {"map": c[0], "robots": c[1], "humans": c[2]}
        for v in VARIANTS:
            vv = [r for r in rr if r["variant"] == v]
            d[f"{v}_intrusion"] = float(np.mean([r["intrusion_rate"] for r in vv]))
            d[f"{v}_clearance"] = float(np.mean([r["mean_min_clearance"] for r in vv]))
            d[f"{v}_time"] = float(np.nanmean([r["mean_time_to_goal"] for r in vv]))
            d[f"{v}_success"] = float(np.mean([r["success_rate"] for r in vv]))
        x, y = paired(rr, "predictive", "snapshot", "intrusion_rate")
        d["p_pred_vs_snapshot_intrusion"] = wilcoxon_p(x, y)[0]
        out.append(d)
    write_csv(os.path.join(TABLES, f"{name}_per_cell.csv"), out)
    with open(os.path.join(TABLES, f"{name}_per_cell.md"), "w", encoding="utf-8") as f:
        f.write("| Map | Robots | People | Intrusion: snapshot | Intrusion: predictive | Intrusion: oracle | Intrusion: LaCAM | Clearance: snapshot | Clearance: predictive | Time: snapshot | Time: predictive | p (pred vs snap, unadjusted) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for d in out:
            f.write(f"| {d['map']} | {d['robots']} | {d['humans']} | {d['snapshot_intrusion']:.3f} | {d['predictive_intrusion']:.3f} | {d['oracle_intrusion']:.3f} | "
                    f"{d['lacam_snapshot_intrusion']:.3f} | {d['snapshot_clearance']:.2f} | {d['predictive_clearance']:.2f} | {d['snapshot_time']:.2f} | "
                    f"{d['predictive_time']:.2f} | {fmt_p(d['p_pred_vs_snapshot_intrusion'])} |\n")
    return out


def figures(rows, cells, hrows):
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.0))
    show = ["unaware", "hard_exclusion", "snapshot", "lacam_snapshot", "predictive", "oracle"]
    for ax, (m, ttl) in zip(axs, (("intrusion_rate", "(a) Intrusion rate"), ("mean_min_clearance", "(b) Mean min. clearance (m)"),
                                   ("mean_time_to_goal", "(c) Time to goal (steps)"))):
        for v in show:
            ys = [np.nanmean([r[m] for r in rows if r["variant"] == v and int(r["n_humans"]) == nh]) for nh in (4, 8, 12)]
            ax.plot((4, 8, 12), ys, marker="o", ms=3.2, color=COL[v], ls="--" if v == "oracle" else "-", label=LABEL[v])
        ax.set_xticks((4, 8, 12)); ax.set_xlabel("Number of people"); ax.set_title(ttl, loc="left"); ax.grid(alpha=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(os.path.join(FIGS, "fig10_closed_loop.png")); fig.savefig(os.path.join(FIGS, "fig10_closed_loop.pdf")); plt.close(fig)
    if hrows:
        Hs = sorted({int(r["pred_horizon"]) for r in hrows})
        fig, ax = plt.subplots(figsize=(W * 0.5, 2.4))
        ax.plot(Hs, [np.mean([r["intrusion_rate"] for r in hrows if int(r["pred_horizon"]) == h]) for h in Hs], marker="o", color="#e45756")
        ax.set_xlabel("Prediction horizon (steps; 0 = snapshot)"); ax.set_ylabel("Intrusion rate"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig11_horizon.png")); fig.savefig(os.path.join(FIGS, "fig11_horizon.pdf")); plt.close(fig)


def main():
    rows = read_csv(os.path.join(RAW, "dynamic_trials.csv"))
    OUT["main_summary"] = summary_table(rows, "dynamic_main")
    for mp in ("map1", "map2"):
        OUT[f"{mp}_summary"] = summary_table([r for r in rows if r["map"] == mp], f"dynamic_{mp}")
    OUT["main_tests"] = tests(rows, "dynamic_main")
    cells = per_cell(rows, "dynamic_main")
    OUT["per_cell"] = cells
    hp = os.path.join(RAW, "dynamic_horizon_trials.csv")
    hrows = read_csv(hp) if os.path.exists(hp) else []
    if hrows:
        Hs = sorted({int(r["pred_horizon"]) for r in hrows})
        OUT["horizon"] = [{"pred_horizon": h, "intrusion_rate": float(np.mean([r["intrusion_rate"] for r in hrows if int(r["pred_horizon"]) == h])),
                           "mean_min_clearance": float(np.mean([r["mean_min_clearance"] for r in hrows if int(r["pred_horizon"]) == h])),
                           "mean_time_to_goal": float(np.nanmean([r["mean_time_to_goal"] for r in hrows if int(r["pred_horizon"]) == h]))} for h in Hs]
        write_csv(os.path.join(TABLES, "dynamic_horizon_summary.csv"), OUT["horizon"])
    rp = os.path.join(RAW, "dynamic_robustness_trials.csv")
    if os.path.exists(rp):
        rr = read_csv(rp)
        OUT["robustness_summary"] = summary_table(rr, "dynamic_robustness")
        OUT["robustness_tests"] = tests(rr, "dynamic_robustness")
        per_cell(rr, "dynamic_robustness")
    xp = os.path.join(RAW, "dynamic_reactive_trials.csv")
    if os.path.exists(xp):
        xr = read_csv(xp)
        OUT["reactive_summary"] = summary_table(xr, "dynamic_reactive")
        OUT["reactive_tests"] = tests(xr, "dynamic_reactive")
        per_cell(xr, "dynamic_reactive")
    figures(rows, cells, hrows)
    save_json(os.path.join(TABLES, "dynamic_results_summary.json"), OUT)
    print("dynamic analysis done")


if __name__ == "__main__":
    main()
