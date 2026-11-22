"""Analysis of the uncertainty-aware prediction study exactly as in ANALYSIS_PLAN.md, Amendment 2.

    python experiments/analyze_uncertainty.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, TABLES, FIGS, read_csv, write_csv, save_json  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hamrpp.stats import wilcoxon_p, holm, bootstrap_ci  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 7.5})

CONDS = ["steady", "erratic", "reactive"]
VARS = ["snapshot", "predictive", "predictive_unc", "oracle"]
LABEL = {"snapshot": "Snapshot", "predictive": "CV prediction", "predictive_unc": "Uncertainty-aware prediction",
         "oracle": "Oracle (reference)"}
COL = {"snapshot": "#f2a3a4", "predictive": "#e45756", "predictive_unc": "#8b1a1a", "oracle": "#54a24b"}


def key(r):
    return (r["map"], int(r["n_robots"]), int(r["n_humans"]), int(r["trial"]))


def paired(rows, a, b, m):
    A = {key(r): r[m] for r in rows if r["variant"] == a}
    B = {key(r): r[m] for r in rows if r["variant"] == b}
    ks = sorted(set(A) & set(B))
    return np.array([A[k] for k in ks], float), np.array([B[k] for k in ks], float)


def main():
    rows = read_csv(os.path.join(RAW, "uncertainty_trials.csv"))
    summ = []
    for c in CONDS:
        rc = [r for r in rows if r["condition"] == c]
        for v in VARS:
            rv = [r for r in rc if r["variant"] == v]
            d = {"condition": c, "variant": v, "n": len(rv)}
            for m in ("intrusion_rate", "mean_min_clearance", "mean_time_to_goal", "success_rate", "collision_events", "mean_plan_ms"):
                vals = [r[m] for r in rv]
                d[m] = float(np.nanmean(vals)); d[m + "_lo"], d[m + "_hi"] = bootstrap_ci([[x] for x in vals])
            summ.append(d)
    write_csv(os.path.join(TABLES, "uncertainty_summary.csv"), summ)
    tests = []
    for m in ("intrusion_rate", "mean_min_clearance"):
        fam = []
        for c in CONDS:
            rc = [r for r in rows if r["condition"] == c]
            for b in ("snapshot", "predictive"):
                x, y = paired(rc, "predictive_unc", b, m)
                p, n = wilcoxon_p(x, y)
                fam.append({"metric": m, "condition": c, "comparison": f"predictive_unc vs {b}", "n": n, "p": p,
                            "mean_unc": float(np.mean(x)), "mean_other": float(np.mean(y)),
                            "rel_change_pct": float(100 * (np.mean(x) - np.mean(y)) / np.mean(y)),
                            "median_diff": float(np.median(x - y))})
        for e, q in zip(fam, holm([e["p"] for e in fam])):
            e["p_holm"] = q
        tests += fam
    # descriptive: time to goal
    for c in CONDS:
        rc = [r for r in rows if r["condition"] == c]
        for b in ("snapshot", "predictive"):
            x, y = paired(rc, "predictive_unc", b, "mean_time_to_goal")
            ok = ~(np.isnan(x) | np.isnan(y))
            tests.append({"metric": "mean_time_to_goal", "condition": c, "comparison": f"predictive_unc vs {b}", "n": int(ok.sum()),
                          "p": wilcoxon_p(x, y)[0], "p_holm": float("nan"), "mean_unc": float(np.mean(x[ok])), "mean_other": float(np.mean(y[ok])),
                          "rel_change_pct": float(100 * (np.mean(x[ok]) - np.mean(y[ok])) / np.mean(y[ok])), "median_diff": float(np.median(x[ok] - y[ok]))})
    write_csv(os.path.join(TABLES, "uncertainty_tests.csv"), tests)
    # pre-registered success criterion
    h5 = {c: next(e for e in tests if e["metric"] == "intrusion_rate" and e["condition"] == c and e["comparison"].endswith("snapshot")) for c in CONDS}
    h6 = {c: next(e for e in tests if e["metric"] == "mean_min_clearance" and e["condition"] == c and e["comparison"].endswith("snapshot")) for c in CONDS}
    h5_ok = {c: bool(h5[c]["p_holm"] < 0.05 and h5[c]["mean_unc"] < h5[c]["mean_other"]) for c in CONDS}
    h6_ok = {c: not bool(h6[c]["p_holm"] < 0.05 and h6[c]["mean_unc"] < h6[c]["mean_other"]) for c in CONDS}
    verdict = {"H5_supported_by_condition": h5_ok, "H6_supported_by_condition": h6_ok,
               "robustness_claim_allowed": all(h5_ok.values()) and all(h6_ok.values())}
    save_json(os.path.join(TABLES, "uncertainty_results.json"), {"summary": summ, "tests": tests, "verdict": verdict})
    with open(os.path.join(TABLES, "uncertainty_tests.md"), "w", encoding="utf-8") as f:
        f.write("| Condition | Metric | Comparison | n | Mean (unc.-aware) | Mean (other) | Change (%) | p | p (Holm) |\n|---|---|---|---|---|---|---|---|---|\n")
        for e in tests:
            ph = "—" if e["p_holm"] != e["p_holm"] else (f"{e['p_holm']:.1e}" if e["p_holm"] < 0.001 else f"{e['p_holm']:.3f}")
            pp = f"{e['p']:.1e}" if e["p"] < 0.001 else f"{e['p']:.3f}"
            f.write(f"| {e['condition']} | {e['metric']} | {e['comparison']} | {e['n']} | {e['mean_unc']:.3f} | {e['mean_other']:.3f} | {e['rel_change_pct']:+.1f} | {pp} | {ph} |\n")
        f.write(f"\nPre-registered verdict: {verdict}\n")
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for ax, m, ttl in ((axs[0], "intrusion_rate", "(a) Intrusion rate"), (axs[1], "mean_min_clearance", "(b) Mean min. clearance (m)")):
        w = 0.2
        for k, v in enumerate(VARS):
            vals = [next(d for d in summ if d["condition"] == c and d["variant"] == v) for c in CONDS]
            ys = [d[m] for d in vals]
            err = np.array([[d[m] - d[m + "_lo"] for d in vals], [d[m + "_hi"] - d[m] for d in vals]])
            ax.bar(np.arange(3) + (k - 1.5) * w, ys, w, yerr=err, capsize=1.5, color=COL[v], label=LABEL[v], error_kw={"lw": 0.7})
        ax.set_xticks(range(3)); ax.set_xticklabels(["Steady", "Erratic", "Reactive"]); ax.set_title(ttl, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIGS, "fig12_uncertainty.png"), dpi=600); fig.savefig(os.path.join(FIGS, "fig12_uncertainty.pdf")); plt.close(fig)
    print(verdict)


if __name__ == "__main__":
    main()
