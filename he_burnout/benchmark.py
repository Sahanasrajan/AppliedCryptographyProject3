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
# Step 4b: BFV comparison
# ─────────────────────────────────────────────

import copy
from Pyfhel import Pyfhel, PyCtxt

_SCALE        = 10_000
_WEIGHT_SCALE = 100
_N_POLY       = 8192
_T_BITS       = 36
_MAX_SLOTS    = _N_POLY // 2

def _bfv_setup():
    HE = Pyfhel()
    HE.contextGen(scheme='bfv', n=_N_POLY, t_bits=_T_BITS, sec=128)
    HE.keyGen(); HE.relinKeyGen(); HE.rotateKeyGen()
    return HE

def _bfv_encrypt_col(HE, values):
    int_vals = np.round(values.astype(float) * _SCALE).astype(np.int64)
    chunks = []
    for i in range(0, len(int_vals), _MAX_SLOTS):
        chunk = int_vals[i:i + _MAX_SLOTS]
        padded = np.zeros(_MAX_SLOTS, dtype=np.int64)
        padded[:len(chunk)] = chunk
        chunks.append(HE.encryptInt(padded).to_bytes())
    return chunks

def _bfv_tree_sum(HE, ct, n):
    acc = copy.deepcopy(ct)
    step = 1
    while step < n:
        acc = acc + HE.rotate(copy.deepcopy(acc), step)
        step *= 2
    return acc

def _bfv_avg(HE, chunks, n):
    sizes = [min(_MAX_SLOTS, n - i * _MAX_SLOTS) for i in range(len(chunks))]
    total = None
    for ct_bytes, size in zip(chunks, sizes):
        ct = PyCtxt(pyfhel=HE, bytestring=ct_bytes)
        s = _bfv_tree_sum(HE, ct, size)
        total = s if total is None else total + s
    return HE.decryptInt(total)[0] / _SCALE / n

def _bfv_weighted(HE, chunks, n, weights):
    sizes = [min(_MAX_SLOTS, n - i * _MAX_SLOTS) for i in range(len(chunks))]
    total = None; offset = 0
    for ct_bytes, size in zip(chunks, sizes):
        ct = PyCtxt(pyfhel=HE, bytestring=ct_bytes)
        w = np.zeros(_MAX_SLOTS, dtype=np.int64)
        w[:size] = np.round([weights[(offset+i) % len(weights)] * _WEIGHT_SCALE
                             for i in range(size)]).astype(np.int64)
        s = _bfv_tree_sum(HE, ct * HE.encodeInt(w), size)
        total = s if total is None else total + s
        offset += size
    return HE.decryptInt(total)[0] / _SCALE / _WEIGHT_SCALE

def _bfv_scale(HE, chunks, n, scalar):
    sizes = [min(_MAX_SLOTS, n - i * _MAX_SLOTS) for i in range(len(chunks))]
    result = []
    for ct_bytes, size in zip(chunks, sizes):
        ct = PyCtxt(pyfhel=HE, bytestring=ct_bytes)
        pt = HE.encodeInt(np.full(_MAX_SLOTS, int(round(scalar * _SCALE)), dtype=np.int64))
        result.extend(float(v) / _SCALE / _SCALE for v in HE.decryptInt(ct * pt)[:size])
    return result

def _bfv_add(HE, chunks_a, chunks_b, n):
    sizes = [min(_MAX_SLOTS, n - i * _MAX_SLOTS) for i in range(len(chunks_a))]
    result = []
    for a_bytes, b_bytes, size in zip(chunks_a, chunks_b, sizes):
        result.extend(float(v) / _SCALE for v in
                      HE.decryptInt(PyCtxt(pyfhel=HE, bytestring=a_bytes) +
                                    PyCtxt(pyfhel=HE, bytestring=b_bytes))[:size])
    return result

def run_bfv_at_size(n: int, df_full: pd.DataFrame) -> dict:
    """Run BFV queries at size n, return results dict to merge into main results."""
    df = df_full.head(n).copy()
    w_full = [WEIGHTS_BURN_RISK[i % len(WEIGHTS_BURN_RISK)] for i in range(n)]

    HE = _bfv_setup()
    t0 = time.time()
    cols = {col: _bfv_encrypt_col(HE, df[col].values) for col in NUMERIC_COLS}
    upload_bfv = time.time() - t0
    print(f"  [BFV] Upload: {upload_bfv:.3f}s")

    pt = PlaintextQuerySystem()
    pt.upload_dataset(df, NUMERIC_COLS)

    bfv_queries = {}
    for qid, label, fn_bfv, fn_pt in [
        ("Q1", "avg burn_rate",
            lambda: _bfv_avg(HE, cols["burn_rate"], n),
            lambda: pt.query_average("burn_rate")[0]),
        ("Q2", "avg mental_fatigue",
            lambda: _bfv_avg(HE, cols["mental_fatigue_score"], n),
            lambda: pt.query_average("mental_fatigue_score")[0]),
        ("Q3", "weighted burn_rate",
            lambda: _bfv_weighted(HE, cols["burn_rate"], n, w_full),
            lambda: pt.query_weighted_sum("burn_rate", w_full)[0]),
        ("Q4", "scaled hours",
            lambda: _bfv_scale(HE, cols["hours_per_week"], n, 1/40),
            lambda: pt.query_scaled_column("hours_per_week", 1/40)[0]),
        ("Q5", "stress index",
            lambda: _bfv_add(HE, cols["mental_fatigue_score"],
                             cols["resource_allocation"], n),
            lambda: pt.query_column_sum_two("mental_fatigue_score", "resource_allocation")[0]),
    ]:
        t0 = time.time(); bfv_val = fn_bfv(); bfv_t = time.time() - t0
        pt_val = fn_pt()

        if isinstance(bfv_val, list):
            err = max(abs(a-b) for a,b in zip(bfv_val[:10], pt_val[:10])) if isinstance(pt_val, list) else 0
            result = round(bfv_val[0], 4)
        else:
            err = abs(bfv_val - (pt_val[0] if isinstance(pt_val, list) else pt_val))
            result = round(float(bfv_val), 6)

        bfv_queries[qid] = {"label": label, "bfv_result": result,
                             "bfv_time_s": round(bfv_t, 4), "bfv_abs_error": round(err, 10)}
        print(f"  [BFV] {qid} {label:20s} | t={bfv_t:.3f}s  err={err:.2e}")

    return {"upload_bfv_s": round(upload_bfv, 4), "bfv_queries": bfv_queries}


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

    # Step 4b: BFV comparison —> merge into existing results
    print("\n\n── Step 4b: BFV Comparison ─────────────────────────────────")
    for r in all_results:
        bfv = run_bfv_at_size(r["n"], df_full)
        r["upload_bfv_s"] = bfv["upload_bfv_s"]
        r["bfv_queries"]  = bfv["bfv_queries"]

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
