"""
benchmark.py
Full benchmark: HE vs Plaintext on Developer Burnout dataset.
 
Queries tested:
  Q1 - Average burn_rate
  Q2 - Average mental_fatigue_score  
  Q3 - Weighted burn-risk score (weighted sum across features)
  Q4 - Unit-scaled hours_per_week (÷ 40 → ratio)
  Q5 - Combined stress index (mental_fatigue + resource_allocation)
 
Performance sweep: varies dataset size N ∈ {100, 500, 1000, 3000, 7000}
"""
 
import os, sys, time, json
import numpy as np
import pandas as pd
 
# make sure local modules are found
sys.path.insert(0, os.path.dirname(__file__))
 
from generate_dataset import generate
from he_engine import HEQuerySystem, PlaintextQuerySystem
 
NUMERIC_COLS = [
    "designation", "resource_allocation", "mental_fatigue_score",
    "hours_per_week", "years_experience", "team_size", "burn_rate"
]
 
SIZES = [100, 500, 1000, 3000, 7000]
 
WEIGHTS_BURN_RISK = [0.1, 0.15, 0.35, 0.1, -0.05, -0.05, 0.2]  # one per numeric col
 
# ─────────────────────────────────────────────
# Single run at a given size
# ─────────────────────────────────────────────
 
def run_at_size(n: int, df_full: pd.DataFrame) -> dict:
    df = df_full.head(n).copy()
 
    print(f"\n{'='*60}")
    print(f"  N = {n:,} rows")
    print(f"{'='*60}")
 
    # ── Plaintext setup ──────────────────────
    pt = PlaintextQuerySystem()
    t0 = time.time()
    pt.upload_dataset(df, NUMERIC_COLS)
    pt_upload = time.time() - t0
 
    # ── HE setup ────────────────────────────
    he = HEQuerySystem(poly_modulus_degree=8192)
    he_upload = he.alice_upload_dataset(df, NUMERIC_COLS)
 
    results = {
        "n": n,
        "upload_he_s": round(he_upload, 4),
        "upload_plain_s": round(pt_upload, 6),
    }
 
    queries = {}
 
    # Q1: average burn_rate
    he_val, he_t = he.query_average("burn_rate")
    pt_val, pt_t = pt.query_average("burn_rate")
    queries["Q1_avg_burn_rate"] = {
        "he_result": round(he_val, 6),
        "pt_result": round(pt_val, 6),
        "he_time_s": round(he_t, 4),
        "pt_time_s": round(pt_t, 6),
        "abs_error": round(abs(he_val - pt_val), 8),
        "speedup_plain_vs_he": round(he_t / max(pt_t, 1e-9), 1),
    }
    print(f"  Q1 avg burn_rate  | HE={he_val:.4f}  PT={pt_val:.4f}  err={abs(he_val-pt_val):.2e}  HE_t={he_t:.3f}s  PT_t={pt_t:.5f}s")
 
    # Q2: average mental_fatigue
    he_val, he_t = he.query_average("mental_fatigue_score")
    pt_val, pt_t = pt.query_average("mental_fatigue_score")
    queries["Q2_avg_mental_fatigue"] = {
        "he_result": round(he_val, 6),
        "pt_result": round(pt_val, 6),
        "he_time_s": round(he_t, 4),
        "pt_time_s": round(pt_t, 6),
        "abs_error": round(abs(he_val - pt_val), 8),
    }
    print(f"  Q2 avg fatigue    | HE={he_val:.4f}  PT={pt_val:.4f}  err={abs(he_val-pt_val):.2e}  HE_t={he_t:.3f}s")
 
    # Q3: weighted burn-risk score (dot product)
    w = WEIGHTS_BURN_RISK[:n] if n < len(WEIGHTS_BURN_RISK) else WEIGHTS_BURN_RISK
    # pad/truncate weights to n
    w_full = [WEIGHTS_BURN_RISK[i % len(WEIGHTS_BURN_RISK)] for i in range(n)]
    he_val, he_t = he.query_weighted_sum("burn_rate", w_full)
    pt_val, pt_t = pt.query_weighted_sum("burn_rate", w_full)
    queries["Q3_weighted_burn_risk"] = {
        "he_result": round(he_val, 4),
        "pt_result": round(pt_val, 4),
        "he_time_s": round(he_t, 4),
        "pt_time_s": round(pt_t, 6),
        "abs_error": round(abs(he_val - pt_val), 6),
    }
    print(f"  Q3 weighted score | HE={he_val:.4f}  PT={pt_val:.4f}  err={abs(he_val-pt_val):.2e}  HE_t={he_t:.3f}s")
 
    # Q4: scaled hours (÷40)
    he_vec, he_t = he.query_scaled_column("hours_per_week", 1.0/40.0)
    pt_vec, pt_t = pt.query_scaled_column("hours_per_week", 1.0/40.0)
    max_err = max(abs(h - p) for h, p in zip(he_vec[:10], pt_vec[:10]))
    queries["Q4_scaled_hours"] = {
        "he_sample5": [round(v, 4) for v in he_vec[:5]],
        "pt_sample5": [round(v, 4) for v in pt_vec[:5]],
        "he_time_s": round(he_t, 4),
        "pt_time_s": round(pt_t, 6),
        "max_abs_error_first10": round(max_err, 8),
    }
    print(f"  Q4 scaled hours   | max_err={max_err:.2e}  HE_t={he_t:.3f}s")
 
    # Q5: stress index = mental_fatigue + resource_allocation
    he_vec, he_t = he.query_column_sum_two("mental_fatigue_score", "resource_allocation")
    pt_vec, pt_t = pt.query_column_sum_two("mental_fatigue_score", "resource_allocation")
    max_err = max(abs(h - p) for h, p in zip(he_vec[:10], pt_vec[:10]))
    queries["Q5_stress_index"] = {
        "he_sample5": [round(v, 3) for v in he_vec[:5]],
        "pt_sample5": [round(v, 3) for v in pt_vec[:5]],
        "he_time_s": round(he_t, 4),
        "pt_time_s": round(pt_t, 6),
        "max_abs_error_first10": round(max_err, 8),
    }
    print(f"  Q5 stress index   | max_err={max_err:.2e}  HE_t={he_t:.3f}s")
 
    results["queries"] = queries
    return results
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  HE Burnout Benchmark  — TenSEAL CKKS  vs  Plaintext    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
 
    print("[*] Generating full dataset (7000 rows) …")
    df_full = generate()
    csv_path = os.path.join(os.path.dirname(__file__), "developer_burnout_dataset.csv")
    df_full.to_csv(csv_path, index=False)
    print(f"[✓] Dataset saved → {csv_path}\n")
 
    all_results = []
    for n in SIZES:
        r = run_at_size(n, df_full)
        all_results.append(r)
 
    # Save JSON results
    out_json = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Results saved → {out_json}")
 
    # Print summary table
    print("\n\n╔══ SUMMARY TABLE ════════════════════════════════════════════╗")
    print(f"{'N':>8} │ {'Upload_HE':>10} │ {'Upload_PT':>10} │ {'Q1_HE_t':>9} │ {'Q1_PT_t':>9} │ {'Q1_err':>10}")
    print("─"*70)
    for r in all_results:
        q1 = r["queries"]["Q1_avg_burn_rate"]
        print(f"{r['n']:>8,} │ {r['upload_he_s']:>10.2f}s │ {r['upload_plain_s']:>9.5f}s │ "
              f"{q1['he_time_s']:>8.3f}s │ {q1['pt_time_s']:>8.5f}s │ {q1['abs_error']:>10.2e}")
    print("╚══════════════════════════════════════════════════════════════╝\n")
 