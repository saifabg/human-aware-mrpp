"""Fig. 1: architecture and receding-horizon cycle of the planner.  python experiments/make_schematic.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGS  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 8})
BLUE, RED, TEAL, GREY, GOLD = "#4c78a8", "#e45756", "#72b7b2", "#9d9d9d", "#c9a227"
FILL = {BLUE: "#edf2fa", RED: "#fdeaea", TEAL: "#e6f4f3", GREY: "#f4f4f4", GOLD: "#fbf4dc"}


def box(ax, x, y, w, h, text, c, bold=False, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=FILL[c], ec=c, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, a, b, c="#444", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, ls=ls))


fig = plt.figure(figsize=(6.3, 3.25)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
# left: state layers
ax.text(0.13, 0.95, "Multi-layer state", ha="center", fontweight="bold", fontsize=8)
layers = [("Static map\n(graph, edge weights)", BLUE), ("Robots\n(position, goal, priority)", BLUE),
          ("People\n(observed positions,\nvelocity estimates)", GOLD), ("Predictions\n(mean + uncertainty s_k)", GOLD),
          ("Space-time social cost\nW(u, v, t), Eqs. (1)-(6)", RED)]
ys = [0.79, 0.64, 0.47, 0.30, 0.13]
for (t, c), y in zip(layers, ys):
    box(ax, 0.02, y, 0.22, 0.12, t, c, fs=7)
# right: cycle
cyc = [(0.36, 0.78, "1. Observe people\nand robots", GOLD), (0.66, 0.78, "2. Predict each person\n(constant velocity,\nuncertainty grows with k)", GOLD),
       (0.66, 0.44, "3. Plan all robots:\nspace-time prioritized\nsearch (Algorithm 1)", RED),
       (0.36, 0.44, "4. Execute one step,\nreplan next cycle\n(receding horizon)", TEAL)]
for x, y, t, c in cyc:
    box(ax, x, y, 0.26, 0.15, t, c, fs=7.4)
arrow(ax, (0.62, 0.855), (0.66, 0.855)); arrow(ax, (0.79, 0.78), (0.79, 0.59)); arrow(ax, (0.66, 0.515), (0.62, 0.515)); arrow(ax, (0.49, 0.59), (0.49, 0.78))
box(ax, 0.36, 0.13, 0.56, 0.15, "Reservation table: every committed path reserved in space and time\n(vertices, swaps, goals after arrival)  ->  plans are conflict-free by construction", TEAL, fs=7.2)
arrow(ax, (0.79, 0.44), (0.79, 0.28), TEAL)
ax.annotate("", xy=(0.355, 0.685), xytext=(0.245, 0.685), arrowprops=dict(arrowstyle="<|-|>", color=GREY, lw=1.2))
ax.text(0.30, 0.70, "read /\nupdate", ha="center", va="bottom", fontsize=6.5, color="#555")
ax.set_ylim(0.1, 1.0)
fig.savefig(os.path.join(FIGS, "fig1_architecture.png"), dpi=600); fig.savefig(os.path.join(FIGS, "fig1_architecture.pdf"))
print("ok")
