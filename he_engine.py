"""
he_engine.py
Homomorphic Encryption engine for Developer Burnout dataset queries.

Scheme: CKKS (Cheon-Kim-Kim-Song) via TenSEAL
  - Approximate arithmetic over real/complex numbers
  - Supports: addition, scalar multiplication, dot products
  - Perfect for statistical queries (avg, sum, weighted scores)

Alice: data owner — encrypts dataset, sends ciphertexts to Carol
Carol: cloud server — evaluates queries on ciphertext, returns result ciphertext
Alice: decrypts Carol's result to get plaintext answer
"""

import tenseal as ts
import numpy as np
import time
import json
import os
from typing import List, Tuple, Dict, Any


# ─────────────────────────────────────────────
# Key Setup (Alice)
# ─────────────────────────────────────────────

def alice_setup(poly_modulus_degree: int = 8192, coeff_mod_bit_sizes: List[int] = None) -> ts.Context:
    """
    Alice generates a CKKS TenSEAL context (public + secret keys).
    Returns the full context (with secret key) for Alice.
    
    Security: poly_modulus_degree=8192 gives ~128-bit security.
    """
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60]  # standard for CKKS depth-2 ops
    
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    ctx.generate_galois_keys()       # needed for sum / rotation
    ctx.generate_relin_keys()        # needed after multiplication
    ctx.global_scale = 2 ** 40
    return ctx


def get_carol_context(alice_ctx: ts.Context) -> ts.Context:
    """
    Carol's view: public context only (no secret key).
    Serialise + deserialise to simulate the real-world separation.
    """
    pub_ctx_bytes = alice_ctx.serialize(save_secret_key=False)
    carol_ctx = ts.context_from(pub_ctx_bytes)
    return carol_ctx


# ─────────────────────────────────────────────
# Alice: Encrypt
# ─────────────────────────────────────────────

def alice_encrypt_vector(ctx: ts.Context, data: List[float]) -> ts.CKKSVector:
    """Encrypt a list of floats into a single CKKS vector (packed)."""
    return ts.ckks_vector(ctx, data)


def alice_encrypt_column(ctx: ts.Context, values: np.ndarray) -> bytes:
    """
    Encrypt a numeric column as a packed CKKS vector.
    Returns serialised ciphertext bytes (what Alice sends to Carol).
    """
    enc = ts.ckks_vector(ctx, values.tolist())
    return enc.serialize()


# ─────────────────────────────────────────────
# Carol: Evaluate queries on ciphertext
# ─────────────────────────────────────────────

def carol_compute_sum(carol_ctx: ts.Context, ct_bytes: bytes) -> bytes:
    """Carol homomorphically sums all elements in the encrypted vector."""
    enc = ts.ckks_vector_from(carol_ctx, ct_bytes)
    result = enc.sum()
    return result.serialize()


def carol_compute_weighted_sum(carol_ctx: ts.Context, ct_bytes: bytes, weights: List[float]) -> bytes:
    """
    Carol evaluates weighted sum: ∑ wᵢ * xᵢ  (dot product).
    Weights are plaintext (Carol-side query parameters).
    """
    enc = ts.ckks_vector_from(carol_ctx, ct_bytes)
    result = enc.dot(weights)
    return result.serialize()


def carol_compute_element_multiply(carol_ctx: ts.Context, ct_bytes: bytes, scalar: float) -> bytes:
    """Carol multiplies all encrypted values by a plaintext scalar."""
    enc = ts.ckks_vector_from(carol_ctx, ct_bytes)
    result = enc * scalar
    return result.serialize()


def carol_compute_add_vectors(carol_ctx: ts.Context, ct_a_bytes: bytes, ct_b_bytes: bytes) -> bytes:
    """Carol adds two encrypted vectors element-wise."""
    enc_a = ts.ckks_vector_from(carol_ctx, ct_a_bytes)
    enc_b = ts.ckks_vector_from(carol_ctx, ct_b_bytes)
    result = enc_a + enc_b
    return result.serialize()


def carol_compute_poly_eval(carol_ctx: ts.Context, ct_bytes: bytes, coeffs: List[float]) -> bytes:
    """
    Carol evaluates polynomial: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2
    Useful for: normalisation, burnout-score transformations.
    """
    enc = ts.ckks_vector_from(carol_ctx, ct_bytes)
    # p(x) = a0 + a1*x + a2*x^2  (depth-2 CKKS)
    a0, a1, a2 = coeffs[0], coeffs[1], coeffs[2] if len(coeffs) > 2 else 0.0
    result = enc * a2        # a2 * x
    result = result * enc    # a2 * x^2  — NOTE: this uses one multiplicative level
    scaled = enc * a1        # a1 * x
    result = result + scaled
    result = result + a0
    return result.serialize()


# ─────────────────────────────────────────────
# Alice: Decrypt
# ─────────────────────────────────────────────

def alice_decrypt(alice_ctx: ts.Context, ct_bytes: bytes, n: int = None) -> List[float]:
    """Alice decrypts Carol's result ciphertext."""
    enc = ts.ckks_vector_from(alice_ctx, ct_bytes)
    decrypted = enc.decrypt()
    if n is not None:
        decrypted = decrypted[:n]
    return decrypted


def alice_decrypt_scalar(alice_ctx: ts.Context, ct_bytes: bytes) -> float:
    """Decrypt and return the first element (for scalar results like sum/avg)."""
    vals = alice_decrypt(alice_ctx, ct_bytes, n=1)
    return vals[0]


# ─────────────────────────────────────────────
# High-level Query API
# ─────────────────────────────────────────────

class HEQuerySystem:
    """
    Simulates the full Alice→Carol→Alice encrypted query pipeline.
    
    Usage:
        sys = HEQuerySystem()
        sys.alice_upload_dataset(df, numeric_cols)
        result = sys.query_average("burn_rate")
    """

    def __init__(self, poly_modulus_degree: int = 8192):
        print("[Alice] Generating CKKS context & keys …")
        self.alice_ctx = alice_setup(poly_modulus_degree)
        self.carol_ctx = get_carol_context(self.alice_ctx)
        self.encrypted_columns: Dict[str, bytes] = {}
        self.column_lengths: Dict[str, int] = {}
        self.n_rows = 0

    def alice_upload_dataset(self, df, numeric_cols: List[str]):
        """Alice encrypts each numeric column and 'uploads' to Carol."""
        self.n_rows = len(df)
        print(f"[Alice] Encrypting {self.n_rows} rows × {len(numeric_cols)} columns …")
        t0 = time.time()
        for col in numeric_cols:
            vals = df[col].astype(float).values
            self.encrypted_columns[col] = alice_encrypt_column(self.alice_ctx, vals)
            self.column_lengths[col] = len(vals)
        elapsed = time.time() - t0
        print(f"[Alice] Upload complete in {elapsed:.2f}s")
        return elapsed

    # ── Statistical Queries ──────────────────

    def query_sum(self, col: str) -> Tuple[float, float, float, float]:
        """Returns (he_result, plaintext_result, he_time, plain_time)"""
        ct = self.encrypted_columns[col]
        
        t0 = time.time()
        result_ct = carol_compute_sum(self.carol_ctx, ct)
        he_result = alice_decrypt_scalar(self.alice_ctx, result_ct)
        he_time = time.time() - t0

        return he_result, he_time

    def query_average(self, col: str) -> Tuple[float, float]:
        he_sum, he_time = self.query_sum(col)
        he_avg = he_sum / self.column_lengths[col]
        return he_avg, he_time

    def query_weighted_sum(self, col: str, weights: List[float]) -> Tuple[float, float]:
        ct = self.encrypted_columns[col]
        t0 = time.time()
        result_ct = carol_compute_weighted_sum(self.carol_ctx, ct, weights)
        he_result = alice_decrypt_scalar(self.alice_ctx, result_ct)
        he_time = time.time() - t0
        return he_result, he_time

    def query_scaled_column(self, col: str, scalar: float) -> Tuple[List[float], float]:
        """Multiply every encrypted value by scalar (e.g., unit conversion)."""
        ct = self.encrypted_columns[col]
        t0 = time.time()
        result_ct = carol_compute_element_multiply(self.carol_ctx, ct, scalar)
        he_result = alice_decrypt(self.alice_ctx, result_ct, self.column_lengths[col])
        he_time = time.time() - t0
        return he_result, he_time

    def query_column_sum_two(self, col_a: str, col_b: str) -> Tuple[List[float], float]:
        """Homomorphically add two encrypted columns."""
        ct_a = self.encrypted_columns[col_a]
        ct_b = self.encrypted_columns[col_b]
        t0 = time.time()
        result_ct = carol_compute_add_vectors(self.carol_ctx, ct_a, ct_b)
        he_result = alice_decrypt(self.alice_ctx, result_ct, self.column_lengths[col_a])
        he_time = time.time() - t0
        return he_result, he_time


# ─────────────────────────────────────────────
# Plaintext Baseline (for comparison)
# ─────────────────────────────────────────────

class PlaintextQuerySystem:
    """Identical query set executed on plaintext data for performance comparison."""

    def __init__(self):
        self.columns: Dict[str, np.ndarray] = {}

    def upload_dataset(self, df, numeric_cols: List[str]):
        for col in numeric_cols:
            self.columns[col] = df[col].astype(float).values

    def query_sum(self, col: str) -> Tuple[float, float]:
        t0 = time.time()
        result = float(np.sum(self.columns[col]))
        return result, time.time() - t0

    def query_average(self, col: str) -> Tuple[float, float]:
        t0 = time.time()
        result = float(np.mean(self.columns[col]))
        return result, time.time() - t0

    def query_weighted_sum(self, col: str, weights: List[float]) -> Tuple[float, float]:
        t0 = time.time()
        result = float(np.dot(self.columns[col], weights))
        return result, time.time() - t0

    def query_scaled_column(self, col: str, scalar: float) -> Tuple[List[float], float]:
        t0 = time.time()
        result = (self.columns[col] * scalar).tolist()
        return result, time.time() - t0

    def query_column_sum_two(self, col_a: str, col_b: str) -> Tuple[List[float], float]:
        t0 = time.time()
        result = (self.columns[col_a] + self.columns[col_b]).tolist()
        return result, time.time() - t0
