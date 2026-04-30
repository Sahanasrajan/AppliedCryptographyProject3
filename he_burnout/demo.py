"""
demo.py
Interactive demo of the HE Developer Burnout Query System.

Simulates the Alice → Carol → Alice pipeline interactively.
Run: python demo.py
"""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from generate_dataset import generate
from he_engine import HEQuerySystem, PlaintextQuerySystem

NUMERIC_COLS = ["designation", "resource_allocation", "mental_fatigue_score",
                "hours_per_week", "years_experience", "team_size", "burn_rate"]

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          HE Developer Burnout Query Demo                         ║
║          Dataset  : Developer Burnout (7000 samples)             ║
║          Scheme   : CKKS (TenSEAL) — 128-bit security           ║
║          Roles    : Alice (data owner) ↔ Carol (cloud server)    ║
╚══════════════════════════════════════════════════════════════════╝
"""

MENU = """
Available Queries (computed OVER ENCRYPTED DATA by Carol):
  1  →  Average burn_rate
  2  →  Average mental_fatigue_score
  3  →  Average resource_allocation
  4  →  Average hours_per_week
  5  →  Average years_experience
  6  →  Stress Index  =  mental_fatigue + resource_allocation  (per employee)
  7  →  Normalised hours  =  hours_per_week ÷ 40  (per employee)
  8  →  Weighted burn-risk score (dot product across all employees)
  0  →  Quit
"""

def print_compare(label, he_val, pt_val, he_t, pt_t):
    err = abs(he_val - pt_val) if isinstance(he_val, float) else None
    speedup = he_t / max(pt_t, 1e-9)
    print(f"\n  ┌─ {label}")
    print(f"  │  HE  result : {he_val}")
    print(f"  │  Plain result: {pt_val}")
    if err is not None:
        print(f"  │  Abs error  : {err:.2e}  (CKKS approximation noise)")
    print(f"  │  HE   time  : {he_t:.4f}s")
    print(f"  │  Plain time : {pt_t:.6f}s")
    print(f"  └─ Plaintext is ~{speedup:.0f}× faster (expected; HE provides confidentiality)")

def main():
    print(BANNER)

    # ── Load / generate dataset ──────────────
    csv = os.path.join(os.path.dirname(__file__), "developer_burnout_dataset.csv")
    if os.path.exists(csv):
        print(f"[*] Loading dataset from {csv} …")
        df = pd.read_csv(csv)
    else:
        print("[*] Generating dataset …")
        df = generate()
        df.to_csv(csv, index=False)

    n = len(df)
    print(f"[✓] Loaded {n:,} employee records\n")

    # ── Setup systems ─────────────────────────
    print("[Alice] Setting up HE context (CKKS, poly_deg=8192, 128-bit security) …")
    he = HEQuerySystem(poly_modulus_degree=8192)

    print("[Alice] Encrypting dataset columns and uploading to Carol …")
    he_upload_t = he.alice_upload_dataset(df, NUMERIC_COLS)

    pt = PlaintextQuerySystem()
    pt.upload_dataset(df, NUMERIC_COLS)

    print(f"[✓] HE upload took {he_upload_t:.2f}s\n")

    # Sample weights for Q8
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.05, 0.20, n).tolist()

    # ── Interactive loop ──────────────────────
    while True:
        print(MENU)
        choice = input("  Enter query number: ").strip()

        if choice == "0":
            print("\n[*] Exiting. Goodbye!")
            break

        elif choice == "1":
            he_val, he_t = he.query_average("burn_rate")
            pt_val, pt_t = pt.query_average("burn_rate")
            print_compare("Average Burn Rate (Carol computed on ciphertext)", round(he_val,6), round(pt_val,6), he_t, pt_t)

        elif choice == "2":
            he_val, he_t = he.query_average("mental_fatigue_score")
            pt_val, pt_t = pt.query_average("mental_fatigue_score")
            print_compare("Average Mental Fatigue Score", round(he_val,4), round(pt_val,4), he_t, pt_t)

        elif choice == "3":
            he_val, he_t = he.query_average("resource_allocation")
            pt_val, pt_t = pt.query_average("resource_allocation")
            print_compare("Average Resource Allocation", round(he_val,4), round(pt_val,4), he_t, pt_t)

        elif choice == "4":
            he_val, he_t = he.query_average("hours_per_week")
            pt_val, pt_t = pt.query_average("hours_per_week")
            print_compare("Average Hours Per Week", round(he_val,4), round(pt_val,4), he_t, pt_t)

        elif choice == "5":
            he_val, he_t = he.query_average("years_experience")
            pt_val, pt_t = pt.query_average("years_experience")
            print_compare("Average Years Experience", round(he_val,4), round(pt_val,4), he_t, pt_t)

        elif choice == "6":
            he_vec, he_t = he.query_column_sum_two("mental_fatigue_score", "resource_allocation")
            pt_vec, pt_t = pt.query_column_sum_two("mental_fatigue_score", "resource_allocation")
            print(f"\n  ┌─ Stress Index (mental_fatigue + resource_allocation) — first 10 employees")
            print(f"  │  HE  : {[round(v,3) for v in he_vec[:10]]}")
            print(f"  │  PT  : {[round(v,3) for v in pt_vec[:10]]}")
            max_err = max(abs(h-p) for h,p in zip(he_vec[:10],pt_vec[:10]))
            print(f"  │  Max abs error (first 10): {max_err:.2e}")
            print(f"  │  HE time: {he_t:.4f}s   Plain time: {pt_t:.6f}s")
            print(f"  └─ Plaintext is ~{he_t/max(pt_t,1e-9):.0f}× faster")

        elif choice == "7":
            he_vec, he_t = he.query_scaled_column("hours_per_week", 1/40.0)
            pt_vec, pt_t = pt.query_scaled_column("hours_per_week", 1/40.0)
            print(f"\n  ┌─ Normalised Hours (hours ÷ 40) — first 10 employees")
            print(f"  │  HE  : {[round(v,4) for v in he_vec[:10]]}")
            print(f"  │  PT  : {[round(v,4) for v in pt_vec[:10]]}")
            max_err = max(abs(h-p) for h,p in zip(he_vec[:10],pt_vec[:10]))
            print(f"  │  Max abs error (first 10): {max_err:.2e}")
            print(f"  └─ HE time: {he_t:.4f}s  |  Plain time: {pt_t:.6f}s")

        elif choice == "8":
            he_val, he_t = he.query_weighted_sum("burn_rate", weights)
            pt_val, pt_t = pt.query_weighted_sum("burn_rate", weights)
            print_compare("Weighted Burn-Risk Score (dot product)", round(he_val,4), round(pt_val,4), he_t, pt_t)

        else:
            print("  [!] Invalid choice. Please enter 0–8.")

        input("\n  Press Enter to continue …")

if __name__ == "__main__":
    main()
