"""
plot_compare.py
Generates chart_all_queries.png for Slide 12 (CKKS vs BFV comparison).

Reads benchmark_results.json (produced by benchmark.py) and outputs:
  chart_all_queries.png  — all 5 query times at N=7,000 for CKKS vs BFV

Run:
    cd he_burnout
    python plot_compare.py
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "benchmark_results.json")) as f:
    data = json.load(f)

# Only use entries that have BFV data (added by Step 4b)
data = [r for r in data if "bfv_queries" in r]
if not data:
    print("No BFV data found. Run benchmark.py first.")
    exit()

CKKS_KEY_MAP = {
    "Q1": "Q1_avg_burn_rate",
    "Q2": "Q2_avg_mental_fatigue",
    "Q3": "Q3_weighted_burn_risk",
    "Q4": "Q4_scaled_hours",
    "Q5": "Q5_stress_index",
}

CKKS_C = "#3B5BA5"
BFV_C  = "#C0395A"
BAR_W  = 0.35

plt.rcParams.update({
    "font.family": "Calibri",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

r7k   = data[-1]  # N = 7000
qs    = list(r7k["bfv_queries"].keys())
q_labels = [r7k["bfv_queries"][q]["label"].replace(" ", "\n") for q in qs]
xq    = np.arange(len(qs))

ckks_q = [r7k["queries"][CKKS_KEY_MAP[q]]["he_time_s"] for q in qs]
bfv_q  = [r7k["bfv_queries"][q]["bfv_time_s"]          for q in qs]

MIN_HEIGHT = max(max(ckks_q), max(bfv_q)) * 0.04  # 4% of tallest bar

ckks_display = [max(v, MIN_HEIGHT) for v in ckks_q]
bfv_display  = [max(v, MIN_HEIGHT) for v in bfv_q]

fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar(xq - BAR_W/2, ckks_display, BAR_W, color=CKKS_C, label="CKKS", zorder=3)
bars2 = ax.bar(xq + BAR_W/2, bfv_display,  BAR_W, color=BFV_C,  label="BFV",  zorder=3)

for bar, actual in zip(bars1, ckks_q):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{actual:.3f}s", ha="center", va="bottom", fontsize=7, color=CKKS_C)
for bar, actual in zip(bars2, bfv_q):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{actual:.3f}s", ha="center", va="bottom", fontsize=7, color=BFV_C)

ax.set_xticks(xq); ax.set_xticklabels(q_labels, fontsize=8.5)
ax.set_xlabel("Query type")
ax.set_ylabel("Query time (seconds)")
ax.set_title("All queries at N = 7,000 — CKKS vs BFV")
ax.legend(fontsize=9, framealpha=0)
ax.grid(axis="y", alpha=0.25, zorder=0)
fig.tight_layout()

out = os.path.join(HERE, "chart_all_queries.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out}")
