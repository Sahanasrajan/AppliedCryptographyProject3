# CS6903/4783 Project 3.2 — Homomorphic Encryption over Developer Burnout Data

## Overview

This project implements a **homomorphic encryption (HE) system** that allows a cloud server (Carol) to compute statistical queries over Alice's encrypted developer burnout dataset — without ever seeing the plaintext data.

```
Alice (data owner)
  │  encrypts columns with CKKS
  ▼
Carol (cloud server)
  │  evaluates queries on ciphertext  (SUM, DOT, SCALE, ADD)
  ▼
Alice
  │  decrypts Carol's result ciphertext
  ▼
 Final answer (e.g., average burn_rate = 0.4687)
```

---

## Dataset

**Developer Burnout Prediction Dataset** — 7,000 employee records  
Source: [kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)

| Column | Type | Description |
|---|---|---|
| employee_id | string | Unique ID |
| gender | categorical | Male / Female / Non-binary |
| company_type | categorical | Product / Service |
| wfh_setup_available | categorical | Yes / No |
| designation | int (1–5) | Seniority level |
| resource_allocation | float (1–10) | Workload intensity |
| mental_fatigue_score | float (0–10) | Self-reported fatigue |
| hours_per_week | int (30–70) | Weekly work hours |
| years_experience | int (1–20) | Experience |
| team_size | int (3–30) | Team headcount |
| **burn_rate** | float (0–1) | **Target variable** |

---

## Cryptographic Scheme: CKKS

**Why CKKS?**  
- Approximate arithmetic — ideal for floating-point statistics  
- Supports addition and multiplication over encrypted vectors  
- 128-bit security with `poly_modulus_degree = 8192`  
- Implemented via **TenSEAL** (Microsoft SEAL backend)

**Key parameters:**
```
Scheme        : CKKS
poly_degree   : 8192    → 128-bit security
coeff_moduli  : [60, 40, 40, 60] bits
global_scale  : 2^40
Galois keys   : yes (for sum/rotation)
Relin keys    : yes (after multiplication)
```

---

## Supported Queries

| ID | Query | HE Operation |
|---|---|---|
| Q1 | Average burn_rate | `sum()` → divide |
| Q2 | Average mental_fatigue | `sum()` → divide |
| Q3 | Weighted burn-risk score | `dot(weights)` |
| Q4 | Normalised hours (÷40) | `scalar_multiply` |
| Q5 | Stress index per employee | `vector_add` (two columns) |
| Q6 | Average resource_allocation | `sum()` → divide |
| Q7 | Average hours_per_week | `sum()` → divide |

---

## Project Structure

```
AppliedCryptographyProject3/
├── README.md                          # This file (project root)
└── he_burnout/
    ├── app.py                         # Flask web app (Live Query + Charts tabs)
    ├── he_engine.py                   # CKKS HE engine (Alice + Carol + PlaintextBaseline)
    ├── generate_dataset.py            # Synthetic dataset generator (7000 rows)
    ├── benchmark.py                   # Full benchmark: N ∈ {100,500,1000,3000,7000}
    ├── demo.py                        # Interactive CLI demo
    ├── plot_results.py                # Matplotlib chart generator
    ├── dashboard.html                 # Static HTML dashboard
    ├── benchmark_results.json         # Benchmark output (auto-generated)
    ├── benchmark_plots.png            # Performance charts (auto-generated)
    └── developer_burnout_dataset.csv  # Dataset (auto-generated)
```

---

## Running the Project

### 1. Install dependencies
```bash
pip install tenseal pandas numpy matplotlib flask
```
#### a. Install dependencies
- Python version matters (Link for python 3.12: https://www.python.org/downloads/release/python-31210/)
```bash
py -3.12 -m pip install -r requirements.txt
```

### 2. Launch the interactive web app (recommended)
```bash
cd he_burnout
python app.py
```
Then open **http://localhost:5000** in your browser.  
In GitHub Codespaces, go to the **Ports tab** and open port **5000**.

The web app has two tabs:

**▶ Live Query tab**
- Select dataset size (N = 100 to 7,000)
- Select a query type (avg, stress index, weighted risk, etc.)
- Click **Run Encrypted Query**
- Watch the Alice → Carol → Alice pipeline animate step by step
- See HE result vs plaintext result with timing and CKKS error
- Query history table builds up automatically

**📊 Performance Charts tab**
- Upload / Encrypt Time — HE vs plaintext upload cost across all N
- Query Execution Time — all 5 query types across dataset sizes
- Slowdown Factor — log-scale HE/plaintext ratio per query type
- CKKS Approximation Error — log-scale noise showing error < 10⁻⁷

### 3. Run the full benchmark (CLI)
```bash
cd he_burnout
python benchmark.py
```
Outputs `benchmark_results.json` and a summary table in the terminal.

### 4. Generate performance charts (static PNG)
```bash
cd he_burnout
python plot_results.py
```
Outputs `benchmark_plots.png` — a 4-panel Matplotlib figure.

### 5. Interactive CLI demo
```bash
cd he_burnout
python demo.py
```
Menu-driven terminal demo. Run queries 1–8 and compare HE vs plaintext results interactively.

---

## Performance Results (Summary)

| N | HE Upload | Plain Upload | Q1 HE Time | Q1 Plain Time | Q1 Error |
|---|---|---|---|---|---|
| 100 | 0.07s | <0.001s | 0.043s | ~0.0001s | 3×10⁻⁸ |
| 500 | 0.05s | <0.001s | 0.120s | ~0.0001s | <10⁻⁹ |
| 1,000 | 0.07s | <0.001s | 0.117s | ~0.0001s | 9×10⁻⁹ |
| 3,000 | 0.08s | <0.001s | 0.140s | ~0.0001s | 3×10⁻⁹ |
| 7,000 | 0.13s | <0.001s | 0.174s | ~0.0001s | 2×10⁻⁹ |

- **HE is ~1,000–2,400× slower** than plaintext — the expected privacy cost.
- **Approximation error is < 10⁻⁷** across all sizes — negligible for analytics.
- Encryption time scales sub-linearly due to CKKS vector packing.
- Vector-add (Q5) and scalar-multiply (Q4) are the cheapest HE operations at ~0.003–0.034s.

---

## Security Properties

- Carol **never sees plaintext data** — only ciphertexts
- Alice's secret key never leaves her machine
- All query results are returned as ciphertexts, decrypted only by Alice
- CKKS provides **IND-CPA security** under the RLWE problem

---

## Comparison with Plaintext (Project Step 4a)

The `PlaintextQuerySystem` in `he_engine.py` runs identical queries on unencrypted NumPy arrays. Key findings:

1. **Correctness**: HE results match plaintext up to ~10⁻⁸ absolute error (CKKS noise)
2. **Speed**: Plaintext is ~1,000–2,400× faster — the inherent cost of data confidentiality
3. **Scaling**: Both scale similarly with N; HE overhead is dominated by SEAL operations, not Python loops
4. **Query type matters**: SUM-based queries (avg) are slowest due to Galois key rotations; vector-add is cheapest
