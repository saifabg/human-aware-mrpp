"""Builds every table (CSV + Markdown) and figure (PNG 600 dpi + PDF) from results/raw.

    python experiments/analyze.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, TABLES, FIGS, read_csv, write_csv, save_json, MAIN_METHODS, METHOD_ORDER  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hamrpp import DEFAULT_CONFIG, map1, map2  # noqa: E402
from hamrpp.planners import METHOD_LABELS, METHODS  # noqa: E402
from hamrpp.costs import social_field  # noqa: E402
from hamrpp.maps import map2_free_cells  # noqa: E402
from hamrpp.metrics import robot_rows  # noqa: E402
from hamrpp.scenarios import illustrative  # noqa: E402
from hamrpp.stats import wilcoxon_p, mcnemar_exact_p, holm, bootstrap_ci  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "savefig.dpi": 600})
W = 6.3
COL = {"classic": "#9d9d9d", "hard_exclusion": "#9fb7d6", "human_aware": "#f2a3a4", "hard_exclusion_st": "#4c78a8",
       "unaware_st": "#c9c9c9", "human_aware_st": "#e45756", "cbs": "#72b7b2",
       "pbs": "#b279a2", "ecbs": "#f58518", "pibt": "#9d755d", "lacam": "#54a24b"}
SHORT = {"classic": "Classic", "hard_exclusion": "Hard excl. (v1)", "human_aware": "Human-aware (v1)",
         "hard_exclusion_st": "Hard excl.", "unaware_st": "No social cost", "human_aware_st": "Proposed",
         "cbs": "CBS", "pbs": "PBS", "ecbs": "ECBS", "pibt": "PIBT", "lacam": "LaCAM"}
SUMMARY = {}


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"))
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)


def fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def md_table(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")


# ------------------------------------------------------------------ descriptive summaries
def by(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def summarize(R, T, methods):
    out = []
    trials = sorted({int(t["trial"]) for t in T})
    for m in methods:
        rr, tt = by(R, method=m), by(T, method=m)
        if not tt:
            continue
        valid = [r for r in rr if r["feasible"] == 1]
        per_trial = lambda k: [[r[k] for r in valid if int(r["trial"]) == i] for i in trials]
        row = {"method": m, "label": METHOD_LABELS[m], "n_trials": len(tt), "n_robot_instances": len(rr)}
        for k, name in (("path_len", "path_len"), ("turns", "turns"), ("min_human_dist", "min_human_dist")):
            row[name] = float(np.mean([r[k] for r in valid])) if valid else math.nan
            row[name + "_ci"] = bootstrap_ci(per_trial(k))
        col = [[t["collisions"]] for t in sorted(tt, key=lambda t: t["trial"])]
        row["collisions_per_trial"] = float(np.mean([t["collisions"] for t in tt]))
        row["collisions_per_trial_ci"] = bootstrap_ci(col)
        inf = [[1 - r["feasible"] for r in rr if int(r["trial"]) == i] for i in trials]
        row["infeasible_pct"] = 100 * float(np.mean([1 - r["feasible"] for r in rr]))
        lo, hi = bootstrap_ci(inf)
        row["infeasible_pct_ci"] = (100 * lo, 100 * hi)
        row["trials_with_collision_pct"] = 100 * float(np.mean([t["any_collision"] for t in tt]))
        row["plan_ms"] = float(np.mean([t["plan_ms"] for t in tt]))
        row["plan_ms_median"] = float(np.median([t["plan_ms"] for t in tt]))
        row["timeout_pct"] = 100 * float(np.mean([t["cbs_status"] == "timeout" for t in tt])) if m in ("cbs", "pbs", "ecbs", "pibt", "lacam") else math.nan
        ent = [r["enters_personal_space"] for r in valid if not math.isnan(r["enters_personal_space"])]
        row["enters_personal_space_pct"] = 100 * float(np.mean(ent)) if ent else math.nan
        out.append(row)
    return out


def write_summary(name, S):
    flat = []
    for s in S:
        d = {k: v for k, v in s.items() if not k.endswith("_ci")}
        for k, v in s.items():
            if k.endswith("_ci"):
                d[k[:-3] + "_lo"], d[k[:-3] + "_hi"] = v
        flat.append(d)
    write_csv(os.path.join(TABLES, f"{name}_summary.csv"), flat)
    hdr = ["Method", "Path length (nodes) [95% CI]", "Turns [95% CI]", "Min. human dist. (m) [95% CI]",
           "Collisions / trial [95% CI]", "Infeasible (%) [95% CI]", "Plan time (ms)", "Search timeout (%)"]
    rows = []
    for s in S:
        ci = lambda k, d=2: f"{fmt(s[k], d)} [{fmt(s[k + '_ci'][0], d)}, {fmt(s[k + '_ci'][1], d)}]"
        rows.append([s["label"], ci("path_len"), ci("turns"), ci("min_human_dist"), ci("collisions_per_trial", 3),
                     ci("infeasible_pct", 1), fmt(s["plan_ms"], 1), fmt(s["timeout_pct"], 1)])
    md_table(os.path.join(TABLES, f"{name}_summary.md"), hdr, rows)


# ------------------------------------------------------------------ significance tests
COMPARISONS = [("human_aware_st", "classic"), ("human_aware_st", "hard_exclusion_st"), ("human_aware_st", "cbs"),
               ("human_aware_st", "pbs"), ("human_aware_st", "ecbs"), ("human_aware_st", "pibt"), ("human_aware_st", "lacam"),
               ("cbs", "hard_exclusion_st"), ("cbs", "classic"),
               ("human_aware_st", "human_aware"), ("human_aware_st", "unaware_st")]
CONT = [("mean_path_len", "Path length"), ("mean_turns", "Turns"), ("mean_min_human_dist", "Min. human distance")]
BIN = [("any_infeasible", "Infeasibility"), ("any_collision", "Collision")]


def significance(name, T):
    trials = sorted({int(t["trial"]) for t in T})
    get = lambda m, k: [next(t[k] for t in T if t["method"] == m and int(t["trial"]) == i) for i in trials]
    res = {c: {} for c in COMPARISONS}
    for k, _ in CONT:
        ps = []
        for a, b in COMPARISONS:
            p, n = wilcoxon_p(get(a, k), get(b, k))
            res[(a, b)][k] = {"p": p, "n": n}; ps.append(p)
        for (a, b), q in zip(COMPARISONS, holm(ps)):
            res[(a, b)][k]["p_holm"] = q
    for k, _ in BIN:
        ps = []
        for a, b in COMPARISONS:
            p, disc = mcnemar_exact_p(get(a, k), get(b, k))
            res[(a, b)][k] = {"p": p, "discordant": disc}; ps.append(p)
        for (a, b), q in zip(COMPARISONS, holm(ps)):
            res[(a, b)][k]["p_holm"] = q
    flat, rows = [], []
    for (a, b) in COMPARISONS:
        d = {"comparison": f"{METHOD_LABELS[a]} vs. {METHOD_LABELS[b]}"}
        cells = [d["comparison"]]
        for k, lab in CONT + BIN:
            d[k + "_p"] = res[(a, b)][k]["p"]; d[k + "_p_holm"] = res[(a, b)][k]["p_holm"]
            cells.append(f"{fmt_p(res[(a, b)][k]['p'])} ({fmt_p(res[(a, b)][k]['p_holm'])})")
        flat.append(d); rows.append(cells)
    write_csv(os.path.join(TABLES, f"{name}_significance.csv"), flat)
    md_table(os.path.join(TABLES, f"{name}_significance.md"),
             ["Comparison"] + [lab + " p (Holm)" for _, lab in CONT + BIN], rows)
    return {f"{a}__vs__{b}": v for (a, b), v in res.items()}


# ------------------------------------------------------------------ figures
def fig_paths_and_heatmap():
    g = map1(); robots, humans = illustrative()
    rows_out = []
    for m in ("human_aware_st", "human_aware"):
        paths, _ = METHODS[m](g, robots, humans, DEFAULT_CONFIG)
        for r in robot_rows(g, paths, robots, humans, DEFAULT_CONFIG):
            r["method"] = m; rows_out.append(r)
        if m == "human_aware_st":
            st_paths = paths
    write_csv(os.path.join(TABLES, "table1_illustrative.csv"), rows_out)
    md_table(os.path.join(TABLES, "table1_illustrative.md"), ["Method", "Robot", "Priority", "Path length", "Turns", "Min. human dist. (m)"],
             [[METHOD_LABELS[r["method"]], r["robot"], r["priority"], r["path_len"], r["turns"], fmt(r["min_human_dist"])] for r in rows_out])
    SUMMARY["illustrative"] = rows_out
    # Fig. 4
    fig, ax = plt.subplots(figsize=(W * 0.78, W * 0.78))
    for u, nb in g.adj.items():
        for v, _ in nb:
            if u < v:
                ax.plot(*zip(g.pos[u], g.pos[v]), color="#c8c8c8", lw=0.5, zorder=1)
    xs, ys = zip(*g.pos.values()); ax.scatter(xs, ys, s=5, color="#555", zorder=2)
    for i, (hx, hy) in enumerate(humans):
        ax.add_patch(plt.Circle((hx, hy), 0.22, color="#8c8c8c", zorder=3))
        ax.add_patch(plt.Circle((hx, hy), DEFAULT_CONFIG.personal_space, fill=False, ls="--", lw=0.8, color="#8c8c8c", zorder=3))
        ax.text(hx, hy - 0.42, f"H{i + 1}", ha="center", va="top", fontsize=7.5)
    cols = ["#d62728", "#1f4fd1", "#2ca02c"]
    for r, c in zip(robots, cols):
        p = [v for k, v in enumerate(st_paths[r["id"]]) if k == 0 or v != st_paths[r["id"]][k - 1]]
        ax.plot(*zip(*[g.pos[v] for v in p]), color=c, lw=2.2, zorder=4, label=f"Robot {r['id']} (P={r['priority']})")
        ax.scatter(*g.pos[r["start"]], marker="s", s=45, color=c, zorder=5)
        ax.scatter(*g.pos[r["goal"]], marker="*", s=110, color=c, zorder=5)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_xlim(-0.8, 8.8); ax.set_ylim(-0.8, 8.8)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout(); save(fig, "fig4_illustrative_paths")
    # Fig. 5
    fig, ax = plt.subplots(figsize=(W * 0.78, W * 0.66))
    X, Y = np.meshgrid(np.linspace(-0.8, 8.8, 300), np.linspace(-0.8, 8.8, 300))
    Z = np.vectorize(lambda x, y: social_field(x, y, humans, DEFAULT_CONFIG.sigma))(X, Y)
    cs = ax.contourf(X, Y, Z, levels=20, cmap="magma")
    fig.colorbar(cs, ax=ax, label="Proximity cost (Eq. 2 term)")
    for u, nb in g.adj.items():
        for v, _ in nb:
            if u < v:
                ax.plot(*zip(g.pos[u], g.pos[v]), color="white", lw=0.3, alpha=0.6)
    ax.scatter(xs, ys, s=3, color="white")
    for hx, hy in humans:
        ax.scatter(hx, hy, s=40, color="#3b82f6", edgecolor="white", zorder=4)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout(); save(fig, "fig5_social_cost_heatmap")


def fig_bars(S, name, methods):
    S = [s for s in S if s["method"] in methods]
    panels = [("path_len", "(a) Path length (nodes)", 2), ("turns", "(b) Turns", 2), ("min_human_dist", "(c) Min. human distance (m)", 2),
              ("collisions_per_trial", "(d) Robot–robot collisions / trial", 3), ("infeasible_pct", "(e) Infeasible robot instances (%)", 1),
              ("plan_ms", "(f) Planning time (ms, log scale)", 1)]
    fig, axs = plt.subplots(2, 3, figsize=(W, 5.0))
    for ax, (k, t, d) in zip(axs.flat, panels):
        vals = [s[k] for s in S]
        err = None
        if k + "_ci" in S[0]:
            err = np.array([[s[k] - s[k + "_ci"][0] for s in S], [s[k + "_ci"][1] - s[k] for s in S]])
            err = np.clip(err, 0, None)
        ax.bar(range(len(S)), vals, color=[COL[s["method"]] for s in S], width=0.68, yerr=err, capsize=2, error_kw={"lw": 0.8})
        ax.set_title(t, loc="left"); ax.set_xticks(range(len(S))); ax.set_xticklabels([SHORT[s["method"]] for s in S], fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
        ax.spines[["top", "right"]].set_visible(False)
        if k == "plan_ms":
            ax.set_yscale("log")
        top = max(vals) if k != "plan_ms" else None
        for i, v in enumerate(vals):
            y = v if err is None else v + err[1][i]
            ax.text(i, y * (1.15 if k == "plan_ms" else 1.0) + (0 if k == "plan_ms" else 0.02 * (top or 1)), fmt(v, min(d, 2)), ha="center", va="bottom", fontsize=5.5, rotation=90 if k in ("collisions_per_trial", "infeasible_pct") else 0)
        if k == "min_human_dist":
            ax.axhline(DEFAULT_CONFIG.personal_space, ls="--", lw=0.8, color="#555")
        if k != "plan_ms":
            ax.set_ylim(0, (max(vals) if err is None else max(np.array(vals) + err[1])) * 1.22 + 1e-9)
    fig.tight_layout(h_pad=1.0, w_pad=1.0); save(fig, name)


def fig_sensitivity(R):
    ws = sorted({r["social_weight"] for r in R})
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
    for m, ls in (("human_aware_st", "-"), ("human_aware", ":")):
        rr = [r for r in R if r["method"] == m and r["feasible"] == 1]
        pl = [np.mean([r["path_len"] for r in rr if r["social_weight"] == w]) for w in ws]
        md = [np.mean([r["min_human_dist"] for r in rr if r["social_weight"] == w]) for w in ws]
        ent = [100 * np.mean([r["enters_personal_space"] for r in rr if r["social_weight"] == w]) for w in ws]
        lab = "proposed" if m == "human_aware_st" else "v1 reservations"
        axs[0].plot(ws, pl, ls, marker="o", ms=3.5, color="#e45756", label=f"Path length ({lab})")
        ax2 = axs[0].twinx() if m == "human_aware_st" else ax2
        ax2.plot(ws, md, ls, marker="s", ms=3.5, color="#4c78a8", label=f"Min. human dist. ({lab})")
        axs[1].plot(ws, ent, ls, marker="^", ms=3.5, color="#54a24b", label=lab)
        SUMMARY.setdefault("sensitivity", {})[m] = {"weights": ws, "path_len": pl, "min_dist": md, "enter_pct": ent}
    axs[0].set_xlabel("social_weight"); axs[0].set_ylabel("Path length (nodes)", color="#e45756"); ax2.set_ylabel("Min. human distance (m)", color="#4c78a8")
    axs[0].set_title("(a) Efficiency vs. clearance", loc="left")
    h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[0].legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")
    axs[1].set_xlabel("social_weight"); axs[1].set_ylabel("Paths entering personal space (%)"); axs[1].set_title("(b) Personal-space entry", loc="left"); axs[1].legend()
    for a in axs: a.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "fig7_sensitivity")
    rows = []
    for m, d in SUMMARY["sensitivity"].items():
        for i, w in enumerate(ws):
            colls = sum(t["collisions"] for t in read_csv(os.path.join(RAW, "sensitivity_trials.csv")) if t["method"] == m and t["social_weight"] == w)
            rows.append({"method": m, "social_weight": w, "path_len": d["path_len"][i], "min_human_dist": d["min_dist"][i],
                         "enters_personal_space_pct": d["enter_pct"][i], "total_collisions": colls})
    write_csv(os.path.join(TABLES, "sensitivity_summary.csv"), rows)
    md_table(os.path.join(TABLES, "sensitivity_summary.md"), ["Method", "social_weight", "Path length", "Min. human dist. (m)", "Enter personal space (%)", "Collisions (total)"],
             [[METHOD_LABELS[r["method"]], r["social_weight"], fmt(r["path_len"]), fmt(r["min_human_dist"]), fmt(r["enters_personal_space_pct"], 1), int(r["total_collisions"])] for r in rows])


def scalability(T, R):
    rows = []
    for m in METHOD_ORDER:
        for nr in (3, 5, 10):
            for nh in (4, 8, 15):
                tt = [t for t in T if t["method"] == m and t["n_robots"] == nr and t["n_humans"] == nh]
                rr = [r for r in R if r["method"] == m and int(r["trial"]) in {int(t["trial"]) for t in tt}
                      and r.get("seed") in {t["seed"] for t in tt}]
                v = [r for r in rr if r["feasible"] == 1]
                rows.append({"method": m, "robots": nr, "humans": nh, "plan_ms": float(np.mean([t["plan_ms"] for t in tt])),
                             "infeasible_pct": 100 * sum(t["n_infeasible"] for t in tt) / (nr * len(tt)),
                             "collisions_total_10_trials": int(sum(t["collisions"] for t in tt)),
                             "trials_with_collision": int(sum(t["any_collision"] for t in tt)),
                             "min_human_dist": float(np.mean([r["min_human_dist"] for r in v])) if v else math.nan,
                             "path_len": float(np.mean([r["path_len"] for r in v])) if v else math.nan,
                             "cbs_timeout_pct": 100 * float(np.mean([t["cbs_status"] == "timeout" for t in tt])) if m in ("cbs", "pbs", "ecbs", "pibt", "lacam") else math.nan})
    write_csv(os.path.join(TABLES, "scalability_summary.csv"), rows)
    md_table(os.path.join(TABLES, "scalability_summary.md"),
             ["Method", "Robots", "Humans", "Plan time (ms)", "Infeasible (%)", "Collisions (10 trials)", "Min. human dist. (m)", "Search timeout (%)"],
             [[METHOD_LABELS[r["method"]], r["robots"], r["humans"], fmt(r["plan_ms"], 1), fmt(r["infeasible_pct"], 1), r["collisions_total_10_trials"],
               fmt(r["min_human_dist"]), fmt(r["cbs_timeout_pct"], 0)] for r in rows])
    SUMMARY["scalability"] = rows
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.0))
    for m in MAIN_METHODS + ["human_aware"]:
        for ax, k, ttl in zip(axs, ("plan_ms", "infeasible_pct", "collisions_total_10_trials"),
                              ("(a) Planning time (ms, log)", "(b) Infeasible robot instances (%)", "(c) Collisions (10 trials)")):
            ys = [np.mean([r[k] for r in rows if r["method"] == m and r["robots"] == nr]) for nr in (3, 5, 10)]
            ax.plot((3, 5, 10), ys, marker="o", ms=3.5, color=COL[m], ls=":" if m == "human_aware" else "-",
                    label=METHOD_LABELS[m])
            ax.set_title(ttl, loc="left"); ax.set_xticks((3, 5, 10)); ax.set_xlabel("Number of robots"); ax.grid(alpha=0.3)
    axs[0].set_yscale("log")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); save(fig, "fig8_scalability")


def cbs_budget(T):
    rows = []
    for (ne, s) in sorted({(int(t["budget_expansions"]), t["budget_s"]) for t in T}):
        for fb in ("lowest_cost", "fewest_conflicts"):
            tt = [t for t in T if int(t["budget_expansions"]) == ne and t["budget_s"] == s and t["fallback"] == fb]
            rows.append({"max_expansions": ne, "time_budget_s": s, "fallback": fb, "n_scenarios": len(tt),
                         "timeout_pct": 100 * float(np.mean([t["cbs_status"] == "timeout" for t in tt])),
                         "collisions_per_trial": float(np.mean([t["collisions"] for t in tt])),
                         "trials_with_collision_pct": 100 * float(np.mean([t["any_collision"] for t in tt])),
                         "plan_ms": float(np.mean([t["plan_ms"] for t in tt]))})
    write_csv(os.path.join(TABLES, "cbs_budget_summary.csv"), rows)
    md_table(os.path.join(TABLES, "cbs_budget_summary.md"), ["Max expansions", "Time budget (s)", "Fallback", "Scenarios", "Timeout (%)",
                                                              "Collisions / trial", "Trials with collision (%)", "Plan time (ms)"],
             [[r["max_expansions"], r["time_budget_s"], r["fallback"], r["n_scenarios"], fmt(r["timeout_pct"], 1), fmt(r["collisions_per_trial"]),
               fmt(r["trials_with_collision_pct"], 1), fmt(r["plan_ms"], 0)] for r in rows])
    SUMMARY["cbs_budget"] = rows


def fig_map2_layout():
    g = map2(); _, _, _, rooms = map2_free_cells()
    fig, ax = plt.subplots(figsize=(W * 0.55, W * 0.68))
    for u, nb in g.adj.items():
        for v, _ in nb:
            if u < v:
                ax.plot(*zip(g.pos[u], g.pos[v]), color="#bbbbbb", lw=0.5)
    xs, ys = zip(*g.pos.values()); ax.scatter(xs, ys, s=5, color="#444")
    for rm in rooms:
        for c, r in rm["cells"]:
            ax.add_patch(plt.Rectangle((c * 0.8 - 0.4, r * 0.8 - 0.4), 0.8, 0.8, color="#f2d7a1", zorder=0))
        c, r = rm["door"]; ax.add_patch(plt.Rectangle((c * 0.8 - 0.4, r * 0.8 - 0.4), 0.8, 0.8, color="#e45756", alpha=0.5, zorder=0))
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Map 2: rooms (tan) and single-cell doorways (red)", loc="left", fontsize=7.5)
    fig.tight_layout(); save(fig, "figS1_map2_layout")


def main():
    fig_paths_and_heatmap()
    for name, fig in (("comparison_map1", "fig6_comparison_map1"), ("comparison_map2", "fig9_comparison_map2")):
        R, T = read_csv(os.path.join(RAW, f"{name}_robots.csv")), read_csv(os.path.join(RAW, f"{name}_trials.csv"))
        S = summarize(R, T, METHOD_ORDER)
        write_summary(name, S)
        SUMMARY[name] = S
        SUMMARY[name + "_significance"] = significance(name, T)
        fig_bars(S, fig, MAIN_METHODS)
    fig_sensitivity(read_csv(os.path.join(RAW, "sensitivity_robots.csv")))
    scalability(read_csv(os.path.join(RAW, "scalability_trials.csv")), read_csv(os.path.join(RAW, "scalability_robots.csv")))
    if os.path.exists(os.path.join(RAW, "cbs_budget_trials.csv")):
        cbs_budget(read_csv(os.path.join(RAW, "cbs_budget_trials.csv")))
    fig_map2_layout()
    save_json(os.path.join(TABLES, "all_results_summary.json"), SUMMARY)
    print("tables ->", TABLES); print("figures ->", FIGS)


if __name__ == "__main__":
    main()
