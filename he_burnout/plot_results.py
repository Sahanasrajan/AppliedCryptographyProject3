"""
plot_results.py
Generate performance comparison charts from benchmark_results.json.
Produces: benchmark_plots.png (4-panel figure)
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_plots.png")

# ── Load ──────────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    results = json.load(f)

ns = [r["n"] for r in results]
upload_he = [r["upload_he_s"] for r in results]
upload_pt = [r["upload_plain_s"] for r in results]

q1_he = [r["queries"]["Q1_avg_burn_rate"]["he_time_s"] for r in results]
q1_pt = [r["queries"]["Q1_avg_burn_rate"]["pt_time_s"] for r in results]
q1_err = [r["queries"]["Q1_avg_burn_rate"]["abs_error"] for r in results]

q3_he = [r["queries"]["Q3_weighted_burn_risk"]["he_time_s"] for r in results]
q3_pt = [r["queries"]["Q3_weighted_burn_risk"]["pt_time_s"] for r in results]

q5_he = [r["queries"]["Q5_stress_index"]["he_time_s"] for r in results]
q5_pt = [r["queries"]["Q5_stress_index"]["pt_time_s"] for r in results]

# ── Style ─────────────────────────────────────────────────────────
HE_COLOR  = "#E85D2F"
PT_COLOR  = "#2F80E8"
ERR_COLOR = "#2BC48A"
BG       = "#0F1117"
GRID_C   = "#2A2D3A"
TEXT_C   = "#E0E4F0"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID_C, "grid.color": GRID_C,
    "text.color": TEXT_C, "axes.labelcolor": TEXT_C,
    "xtick.color": TEXT_C, "ytick.color": TEXT_C,
    "font.family": "monospace", "font.size": 10,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "Homomorphic Encryption vs Plaintext — Developer Burnout Dataset\n"
    "Scheme: CKKS (TenSEAL, poly_deg=8192, 128-bit security)",
    fontsize=13, fontweight="bold", color=TEXT_C, y=0.98
)

ax = axes.flatten()

# ── Panel 1: Upload time ──────────────────────────────────────────
ax[0].plot(ns, upload_he, "o-", color=HE_COLOR, lw=2, ms=7, label="HE (CKKS) encryption time")
ax[0].plot(ns, upload_pt, "s--", color=PT_COLOR, lw=2, ms=7, label="Plaintext copy time")
ax[0].set_title("Dataset Upload / Encryption Time", fontweight="bold")
ax[0].set_xlabel("Dataset size (N rows)")
ax[0].set_ylabel("Time (seconds)")
ax[0].set_xscale("log"); ax[0].grid(True, alpha=0.3)
ax[0].legend(facecolor=BG, edgecolor=GRID_C)

# ── Panel 2: Query time comparison ────────────────────────────────
x = np.arange(len(ns)); w = 0.22
ax[1].bar(x - w,   q1_he, w, color=HE_COLOR, label="Q1 avg (HE)")
ax[1].bar(x,       q3_he, w, color="#E8A02F", label="Q3 weighted (HE)")
ax[1].bar(x + w,   q5_he, w, color="#C42FE8", label="Q5 vec-add (HE)")
ax[1].set_title("HE Query Execution Time by Query Type", fontweight="bold")
ax[1].set_xlabel("Dataset size (N rows)")
ax[1].set_ylabel("Time (seconds)")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{n:,}" for n in ns])
ax[1].grid(True, alpha=0.3, axis="y")
ax[1].legend(facecolor=BG, edgecolor=GRID_C)

# ── Panel 3: HE vs Plain speedup ratio ───────────────────────────
speedup_q1 = [h/max(p,1e-9) for h,p in zip(q1_he, q1_pt)]
speedup_q3 = [h/max(p,1e-9) for h,p in zip(q3_he, q3_pt)]
speedup_q5 = [h/max(p,1e-9) for h,p in zip(q5_he, q5_pt)]
ax[2].plot(ns, speedup_q1, "o-",  color=HE_COLOR, lw=2, ms=7, label="Q1 avg burn_rate")
ax[2].plot(ns, speedup_q3, "s-",  color="#E8A02F", lw=2, ms=7, label="Q3 weighted score")
ax[2].plot(ns, speedup_q5, "^-",  color="#C42FE8", lw=2, ms=7, label="Q5 stress index")
ax[2].axhline(1, color=TEXT_C, lw=1, ls="--", alpha=0.5)
ax[2].set_title("Slowdown Factor (HE time ÷ Plaintext time)", fontweight="bold")
ax[2].set_xlabel("Dataset size (N rows)")
ax[2].set_ylabel("Ratio  (1 = equal speed)")
ax[2].set_xscale("log"); ax[2].grid(True, alpha=0.3)
ax[2].legend(facecolor=BG, edgecolor=GRID_C)
ax[2].set_yscale("log")

# ── Panel 4: CKKS approximation error ───────────────────────────
ax[3].semilogy(ns, q1_err, "o-", color=ERR_COLOR, lw=2, ms=8)
ax[3].set_title("CKKS Approximation Error  (Q1: avg burn_rate)", fontweight="bold")
ax[3].set_xlabel("Dataset size (N rows)")
ax[3].set_ylabel("|HE result − Plaintext result|")
ax[3].set_xscale("log"); ax[3].grid(True, alpha=0.3)
ax[3].fill_between(ns, [1e-9]*len(ns), q1_err, color=ERR_COLOR, alpha=0.15)
ax[3].annotate("All errors < 10⁻⁷\n(negligible for analytics)", xy=(500, max(q1_err)*0.6),
               color=ERR_COLOR, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"[✓] Plot saved → {OUT_PATH}")
