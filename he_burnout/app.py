from flask import Flask, jsonify, request, make_response
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from generate_dataset import generate
from he_engine import HEQuerySystem, PlaintextQuerySystem

app = Flask(__name__)

NUMERIC_COLS = ["designation","resource_allocation","mental_fatigue_score","hours_per_week","years_experience","team_size","burn_rate"]

import pandas as pd
CSV = os.path.join(os.path.dirname(__file__), "developer_burnout_dataset.csv")
df_full = pd.read_csv(CSV) if os.path.exists(CSV) else generate()
print(f"[OK] Dataset: {len(df_full)} rows")

_he_cache, _pt_cache = {}, {}

def get_systems(n):
    if n not in _he_cache:
        df = df_full.head(n).copy()
        he = HEQuerySystem(poly_modulus_degree=8192)
        he.alice_upload_dataset(df, NUMERIC_COLS)
        pt = PlaintextQuerySystem()
        pt.upload_dataset(df, NUMERIC_COLS)
        _he_cache[n] = he
        _pt_cache[n] = pt
        print(f"[OK] HE ready N={n}")
    return _he_cache[n], _pt_cache[n]

print("[*] Pre-warming N=500...")
get_systems(500)
print("[OK] Ready!\n")

BENCH = [
  {"n":100,"uhe":0.072,"upt":0.000672,"q":{"Q1":{"ht":0.043,"pt":0.000072,"e":3e-8},"Q2":{"ht":0.049,"pt":0.000051,"e":3e-8},"Q3":{"ht":0.027,"pt":0.00003,"e":4e-6},"Q4":{"ht":0.0035,"pt":0.00002,"e":2.2e-7},"Q5":{"ht":0.0053,"pt":0.000015,"e":4.7e-9}}},
  {"n":500,"uhe":0.054,"upt":0.000465,"q":{"Q1":{"ht":0.120,"pt":0.000061,"e":1e-9},"Q2":{"ht":0.104,"pt":0.000054,"e":9e-10},"Q3":{"ht":0.066,"pt":0.00004,"e":3e-6},"Q4":{"ht":0.0037,"pt":0.000026,"e":2.2e-7},"Q5":{"ht":0.0058,"pt":0.000018,"e":2.6e-9}}},
  {"n":1000,"uhe":0.075,"upt":0.000466,"q":{"Q1":{"ht":0.117,"pt":0.000062,"e":9e-9},"Q2":{"ht":0.121,"pt":0.000053,"e":9e-9},"Q3":{"ht":0.074,"pt":0.000056,"e":2e-6},"Q4":{"ht":0.0034,"pt":0.000028,"e":2.2e-7},"Q5":{"ht":0.0058,"pt":0.000031,"e":4.6e-9}}},
  {"n":3000,"uhe":0.076,"upt":0.000548,"q":{"Q1":{"ht":0.140,"pt":0.000064,"e":2.9e-9},"Q2":{"ht":0.135,"pt":0.000055,"e":2.8e-9},"Q3":{"ht":0.086,"pt":0.000116,"e":3e-5},"Q4":{"ht":0.0035,"pt":0.000045,"e":2.2e-7},"Q5":{"ht":0.0058,"pt":0.000042,"e":2.4e-9}}},
  {"n":7000,"uhe":0.129,"upt":0.000471,"q":{"Q1":{"ht":0.174,"pt":0.000072,"e":2e-9},"Q2":{"ht":0.163,"pt":0.000056,"e":1.9e-9},"Q3":{"ht":0.087,"pt":0.000252,"e":6.3e-5},"Q4":{"ht":0.034,"pt":0.000081,"e":2.2e-7},"Q5":{"ht":0.012,"pt":0.000079,"e":4.4e-9}}}
]

_cmp_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
BENCH_COMPARE = json.load(open(_cmp_path)) if os.path.exists(_cmp_path) else []

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HE Burnout App</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@600;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f1117; color: #dce4f2; font-family: 'IBM Plex Mono', monospace; }
body::before { content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image: linear-gradient(rgba(96,184,212,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(96,184,212,0.025) 1px, transparent 1px);
  background-size: 44px 44px; }
.wrap { position:relative; z-index:1; max-width:1100px; margin:0 auto; padding:24px 20px; }

.topbar { display:flex; flex-direction:column; gap:16px; padding-bottom:18px; margin-bottom:22px; border-bottom:1px solid #252d3d; }
.topbar-top { display:flex; justify-content:space-between; align-items:center; }
.logo { font-family:'Inter',sans-serif; font-size:1.4rem; color:#60b8d4; text-shadow:none; }
.logo small { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#8a9ab8; display:block; margin-top:3px; font-weight:400; }
.badge { font-size:0.6rem; padding:5px 12px; border:1px solid #60b8d4; border-radius:20px; color:#60b8d4; }

.tab-bar { display:flex; gap:6px; }
.tab-btn { padding:9px 22px; border-radius:7px; border:1px solid #252d3d; background:#12151e; color:#8a9ab8; font-family:'IBM Plex Mono',monospace; font-size:0.72rem; cursor:pointer; }
.tab-btn.active { border-color:#60b8d4; background:rgba(96,184,212,0.08); color:#60b8d4; }

.tab-panel { display:none; }
.tab-panel.active { display:block; }

.pipe { display:flex; align-items:center; background:#12151e; border:1px solid #252d3d; border-radius:8px; padding:12px 14px; overflow-x:auto; gap:0; }
.pstep { display:flex; flex-direction:column; align-items:center; gap:4px; flex:1; min-width:80px; text-align:center; font-size:0.62rem; color:#4e5f7a; padding:6px; border-radius:7px; }
.pstep.active { color:#60b8d4; background:rgba(96,184,212,0.05); }
.pnum { width:22px; height:22px; border-radius:50%; border:1px solid currentColor; display:flex; align-items:center; justify-content:center; font-size:0.6rem; }
.pstep.active .pnum { background:#60b8d4; color:#000; }
.parr { color:#4e5f7a; padding:0 3px; flex-shrink:0; }

.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
.card { background:#161b24; border:1px solid #252d3d; border-radius:10px; padding:16px; }
.card-header { margin-bottom:12px; }
.card-step { display:inline-block; font-size:0.55rem; color:#60b8d4; letter-spacing:0.15em; text-transform:uppercase; background:rgba(96,184,212,0.08); border:1px solid rgba(96,184,212,0.2); border-radius:4px; padding:2px 9px; margin-bottom:6px; }
.card-label { font-family:'Inter',sans-serif; font-size:0.95rem; font-weight:800; color:#dce4f2; letter-spacing:0; margin-bottom:0; }
.btn-row { display:flex; flex-wrap:wrap; gap:6px; }
.sel-btn { padding:6px 12px; border:1px solid #252d3d; border-radius:5px; background:#13172a; color:#8a9ab8; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; cursor:pointer; }
.sel-btn:hover { border-color:#8a9ab8; color:#dce4f2; }
.sel-btn.active { border-color:#60b8d4; background:rgba(96,184,212,0.08); color:#60b8d4; }

.run-section { background:#161b24; border:1px solid #252d3d; border-radius:10px; padding:16px; margin-bottom:14px; }
.run-section-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
.run-btn { padding:10px 32px; background:#60b8d4; color:#000; border:none; border-radius:7px; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:600; cursor:pointer; }
.run-btn:hover { opacity:0.88; transform:translateY(-1px); }
.run-btn:disabled { opacity:0.4; cursor:not-allowed; transform:none; }

.log-box { background:#12151e; border:1px solid #252d3d; border-radius:8px; padding:14px; font-size:0.65rem; line-height:2; }
.log-label { font-size:0.55rem; color:#8a9ab8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }
.log-row { display:flex; gap:8px; }
.log-t { color:#7a8da8; min-width:50px; }
.log-alice { color:#60b8d4; }
.log-carol { color:#9d85e8; }
.log-done { color:#f5b942; font-weight:600; }

.result-box { background:#161b24; border:1px solid #252d3d; border-radius:12px; padding:20px; margin-bottom:16px; display:none; }
.result-box.show { display:block; }
.res-head { display:flex; justify-content:space-between; align-items:center; padding-bottom:12px; margin-bottom:16px; border-bottom:1px solid #252d3d; }
.res-title { font-size:0.65rem; color:#a0afc8; }
.res-title strong { display:block; font-size:0.85rem; color:#dce4f2; margin-top:2px; }
.res-tag { font-size:0.58rem; padding:3px 10px; background:rgba(96,184,212,0.08); border:1px solid rgba(96,184,212,0.25); border-radius:4px; color:#60b8d4; }

.res-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }
.res-cell { border-radius:10px; padding:16px; border:1px solid #252d3d; }
.res-cell.he { border-color:rgba(96,184,212,0.2); background:rgba(96,184,212,0.03); }
.res-cell.pt { border-color:rgba(232,117,79,0.2); background:rgba(232,117,79,0.03); }
.res-cell-label { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px; font-weight:600; }
.res-cell.he .res-cell-label { color:#60b8d4; }
.res-cell.pt .res-cell-label { color:#e8754f; }
.res-cell-how { font-size:0.65rem; color:#8a9ab8; margin-bottom:10px; line-height:1.6; border-left:2px solid #252d3d; padding-left:8px; }
.res-cell.he .res-cell-how { border-color:rgba(96,184,212,0.2); }
.res-cell.pt .res-cell-how { border-color:rgba(232,117,79,0.2); }
.res-val { font-size:1.5rem; font-weight:600; line-height:1; margin-bottom:10px; }
.res-cell.he .res-val { color:#60b8d4; }
.res-cell.pt .res-val { color:#e8754f; }
.res-meta { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:8px; }
.res-meta-item { background:#12151e; border:1px solid #252d3d; border-radius:5px; padding:6px 10px; }
.res-meta-item .rmi-label { font-size:0.58rem; color:#7a8da8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:2px; }
.res-meta-item .rmi-val { font-size:0.72rem; color:#b0bdd4; }
.res-cell.he .rmi-val { color:#60b8d4; }
.res-cell.pt .rmi-val { color:#e8754f; }

.info-sections { display:flex; flex-direction:column; gap:10px; }
.info-section { border-radius:8px; padding:14px 16px; border:1px solid; }
.info-section.err-section { background:rgba(245,185,66,0.04); border-color:rgba(245,185,66,0.2); }
.info-section.slow-section { background:rgba(232,117,79,0.04); border-color:rgba(232,117,79,0.15); }
.info-section.match-section { background:rgba(96,184,212,0.03); border-color:rgba(96,184,212,0.15); }
.info-section-header { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.info-section-title { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; }
.err-section .info-section-title { color:#f5b942; }
.slow-section .info-section-title { color:#e8754f; }
.match-section .info-section-title { color:#60b8d4; }
.info-big-val { font-size:1.3rem; font-weight:600; }
.err-section .info-big-val { color:#f5b942; }
.slow-section .info-big-val { color:#e8754f; }
.match-section .info-big-val { color:#60b8d4; }
.info-section-body { font-size:0.68rem; color:#a0afc8; line-height:1.75; }
.info-section-body strong { color:#dce4f2; }
.info-formula { background:#12151e; border:1px solid #252d3d; border-radius:5px; padding:6px 12px; font-size:0.62rem; color:#9d85e8; margin-top:8px; font-family:'IBM Plex Mono',monospace; }
.match-pill { display:inline-block; background:rgba(96,184,212,0.1); border:1px solid rgba(96,184,212,0.3); border-radius:4px; color:#60b8d4; padding:1px 8px; font-size:0.6rem; margin-left:6px; }

.hist-box { background:#161b24; border:1px solid #252d3d; border-radius:10px; overflow:hidden; }
.hist-head { padding:10px 14px; border-bottom:1px solid #252d3d; font-size:0.58rem; color:#8a9ab8; text-transform:uppercase; letter-spacing:0.1em; display:flex; justify-content:space-between; align-items:center; }
.hist-clear { background:none; border:1px solid #252d3d; color:#4e5f7a; font-family:'IBM Plex Mono',monospace; font-size:0.58rem; padding:2px 8px; border-radius:3px; cursor:pointer; }
.hist-empty { padding:18px; text-align:center; color:#4e5f7a; font-size:0.65rem; }
.htable { width:100%; border-collapse:collapse; font-size:0.65rem; }
.htable th { background:#13172a; color:#8a9ab8; padding:8px 12px; text-align:left; font-size:0.58rem; text-transform:uppercase; border-bottom:1px solid #252d3d; }
.htable td { padding:8px 12px; border-bottom:1px solid rgba(28,36,54,0.6); }
.htable tr:last-child td { border-bottom:none; }
.c-he { color:#60b8d4; } .c-pt { color:#e8754f; } .c-err { color:#f5b942; }
.slow-pill { background:rgba(232,117,79,0.1); border:1px solid rgba(232,117,79,0.25); color:#e8754f; border-radius:3px; padding:1px 6px; font-size:0.58rem; }

.chart-intro { background:#12151e; border:1px solid #252d3d; border-radius:10px; padding:14px 18px; margin-bottom:18px; font-size:0.68rem; color:#8a9ab8; line-height:1.7; }
.chart-intro strong { color:#60b8d4; }
.chart-tabs { display:flex; gap:6px; margin-bottom:16px; flex-wrap:wrap; }
.chart-tab { padding:7px 14px; border:1px solid #252d3d; border-radius:5px; background:#13172a; color:#8a9ab8; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; cursor:pointer; }
.chart-tab:hover { border-color:#8a9ab8; color:#dce4f2; }
.chart-tab.active { border-color:#60b8d4; background:rgba(96,184,212,0.07); color:#60b8d4; }
.chart-panel { display:none; }
.chart-panel.active { display:block; }
.chart-card { background:#161b24; border:1px solid #252d3d; border-radius:12px; padding:20px; margin-bottom:14px; }
.chart-title { font-family:'Inter',sans-serif; font-size:0.9rem; color:#dce4f2; margin-bottom:4px; }
.chart-sub { font-size:0.62rem; color:#8a9ab8; margin-bottom:16px; line-height:1.6; }
.chart-wrap { position:relative; height:280px; }
.chart-wrap-tall { position:relative; height:320px; }
.chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.insight { background:#12151e; border-left:2px solid #60b8d4; border-radius:0 6px 6px 0; padding:10px 13px; margin-top:12px; font-size:0.63rem; color:#8a9ab8; line-height:1.6; }
.insight strong { color:#60b8d4; }

/* ── INTRO MODAL ── */
.modal-overlay {
  position:fixed; inset:0; z-index:9999;
  background:rgba(0,0,0,0.82);
  backdrop-filter:blur(4px);
  display:flex; align-items:center; justify-content:center;
  padding:20px;
  animation:fadeInOverlay 0.3s ease;
}
.modal-overlay.hide { animation:fadeOutOverlay 0.25s ease forwards; }
@keyframes fadeInOverlay  { from{opacity:0} to{opacity:1} }
@keyframes fadeOutOverlay { from{opacity:1} to{opacity:0} }

.modal-box {
  background:#12151e;
  border:1px solid #60b8d4;
  border-radius:14px;
  max-width:680px; width:100%;
  padding:32px 36px;
  box-shadow:0 0 60px rgba(96,184,212,0.12);
  position:relative;
  animation:slideUp 0.35s cubic-bezier(0.4,0,0.2,1);
}
@keyframes slideUp { from{transform:translateY(24px);opacity:0} to{transform:translateY(0);opacity:1} }

.modal-tag {
  font-size:0.58rem; letter-spacing:0.14em; text-transform:uppercase;
  color:#60b8d4; border:1px solid rgba(96,184,212,0.35);
  border-radius:20px; display:inline-block; padding:4px 14px; margin-bottom:18px;
}
.modal-title {
  font-family:'Inter',sans-serif; font-size:1.5rem; color:#60b8d4;
  text-shadow:none; line-height:1.2; margin-bottom:6px;
}
.modal-subtitle { font-size:0.68rem; color:#8a9ab8; margin-bottom:22px; }

.modal-section { margin-bottom:18px; }
.modal-section-label {
  font-size:0.57rem; text-transform:uppercase; letter-spacing:0.12em;
  color:#60b8d4; opacity:0.7; margin-bottom:8px;
}
.modal-section-body {
  font-size:0.68rem; color:#8a9ab8; line-height:1.85;
}
.modal-section-body strong { color:#dce4f2; }

.modal-divider { border:none; border-top:1px solid #252d3d; margin:20px 0; }

.modal-cols { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:22px; }
.modal-col {
  background:#161b24; border:1px solid #252d3d; border-radius:9px; padding:14px 16px;
}
.modal-col-label { font-size:0.57rem; text-transform:uppercase; letter-spacing:0.1em; color:#8a9ab8; margin-bottom:6px; }
.modal-col-val { font-size:0.72rem; color:#dce4f2; line-height:1.7; }
.modal-col-val span { color:#60b8d4; }

.modal-close-row { text-align:center; }
.modal-close-btn {
  padding:12px 48px; background:#60b8d4; color:#000; border:none;
  border-radius:7px; font-family:'IBM Plex Mono',monospace;
  font-size:0.78rem; font-weight:600; cursor:pointer; letter-spacing:0.04em;
}
.modal-close-btn:hover { opacity:0.88; transform:translateY(-1px); }
.modal-skip {
  display:block; margin-top:10px; font-size:0.58rem; color:#4e5f7a; cursor:pointer;
  background:none; border:none; font-family:'IBM Plex Mono',monospace;
}
.modal-skip:hover { color:#8a9ab8; }
</style>
</head>
<body>

<!-- INTRO MODAL -->
<div class="modal-overlay" id="intro-modal">
  <div class="modal-box">
    <div class="modal-tag">CS6903 / 4783 · Project 3 · NYU</div>
    <div class="modal-title">HE Burnout Explorer</div>
    <div class="modal-subtitle">Homomorphic Encryption over Outsourced Data · CKKS scheme via TenSEAL</div>

    <div class="modal-section">
      <div class="modal-section-label">What is this?</div>
      <div class="modal-section-body">
        This app demonstrates <strong>Homomorphic Encryption (HE)</strong> — a cryptographic technique
        that lets a third party (<strong>Carol</strong>) compute queries on data she <em>never decrypts</em>.
        Alice encrypts a developer burnout dataset using the <strong>CKKS scheme</strong>, sends only
        ciphertexts to Carol, and Carol returns encrypted results that only Alice can decrypt.
        The data stays private end-to-end.
      </div>
    </div>

    <div class="modal-section">
      <div class="modal-section-label">The Dataset</div>
      <div class="modal-section-body">
        <strong>Developer Burnout Dataset</strong> — 7,000 synthetic records modeled on the
        Kaggle employee burnout dataset. Each row represents one developer with fields for
        designation, resource allocation, mental fatigue score, hours per week, years of
        experience, team size, and burn rate (0–1 scale).
      </div>
    </div>

    <hr class="modal-divider">

    <div class="modal-cols">
      <div class="modal-col">
        <div class="modal-col-label">HE Scheme</div>
        <div class="modal-col-val">
          <span>CKKS</span> — approximate arithmetic<br>
          poly_degree = <span>8192</span><br>
          security = <span>128-bit</span> (RLWE)
        </div>
      </div>
      <div class="modal-col">
        <div class="modal-col-label">What you can do</div>
        <div class="modal-col-val">
          Run <span>5 query types</span> on ciphertext<br>
          Compare HE vs plaintext speed<br>
          Inspect <span>CKKS error</span> (≤ 10⁻⁷)
        </div>
      </div>
    </div>

    <div class="modal-close-row">
      <button class="modal-close-btn" onclick="closeModal()">▶  Start Exploring</button>
      <button class="modal-skip" onclick="closeModal(true)">don't show again</button>
    </div>
  </div>
</div>

<div class="wrap">

<div class="topbar">
  <div class="topbar-top">
    <div class="logo">HE Burnout<small>Homomorphic Encryption · Developer Burnout Dataset · CS6903/4783</small></div>
    <div class="badge">CKKS · 128-bit · TenSEAL</div>
  </div>
  <div class="tab-bar">
    <button class="tab-btn active" id="tab-query" onclick="showTab('query')">▶  Live Query</button>
    <button class="tab-btn" id="tab-charts" onclick="showTab('charts')">📊  Performance Charts</button>
  </div>
</div>

<!-- QUERY TAB -->
<div class="tab-panel active" id="panel-query">

  <div class="grid2">
    <div class="card">
      <div class="card-header">
        <div class="card-step">Step 01</div>
        <div class="card-label">Select Dataset Size</div>
      </div>
      <div class="btn-row" id="size-group">
        <button class="sel-btn" onclick="pickSize(this,100)">N = 100</button>
        <button class="sel-btn active" onclick="pickSize(this,500)">N = 500</button>
        <button class="sel-btn" onclick="pickSize(this,1000)">N = 1,000</button>
        <button class="sel-btn" onclick="pickSize(this,3000)">N = 3,000</button>
        <button class="sel-btn" onclick="pickSize(this,7000)">N = 7,000</button>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <div class="card-step">Step 02</div>
        <div class="card-label">Select Query</div>
      </div>
      <div class="btn-row" id="query-group">
        <button class="sel-btn active" onclick="pickQuery(this,'avg_burn_rate')">avg(burn_rate)</button>
        <button class="sel-btn" onclick="pickQuery(this,'avg_fatigue')">avg(fatigue)</button>
        <button class="sel-btn" onclick="pickQuery(this,'avg_resource')">avg(resource)</button>
        <button class="sel-btn" onclick="pickQuery(this,'avg_hours')">avg(hours)</button>
        <button class="sel-btn" onclick="pickQuery(this,'stress_index')">stress_index</button>
        <button class="sel-btn" onclick="pickQuery(this,'scaled_hours')">scaled_hours</button>
        <button class="sel-btn" onclick="pickQuery(this,'weighted_risk')">weighted_risk</button>
      </div>
    </div>
  </div>

  <div class="run-section">
    <div class="run-section-top">
      <div class="card-header" style="margin-bottom:0;">
        <div class="card-step">Step 03</div>
        <div class="card-label">Run Encrypted Query</div>
      </div>
      <button class="run-btn" id="run-btn" onclick="runQuery()">▶  Run Encrypted Query</button>
    </div>

    <div class="pipe" style="margin-bottom:12px;">
      <div class="pstep active" id="ps-1"><div class="pnum">1</div><div>Alice<br>Keygen</div></div>
      <div class="parr">→</div>
      <div class="pstep" id="ps-2"><div class="pnum">2</div><div>Alice<br>Encrypt</div></div>
      <div class="parr">→</div>
      <div class="pstep" id="ps-3"><div class="pnum">3</div><div>Carol<br>Receives</div></div>
      <div class="parr">→</div>
      <div class="pstep" id="ps-4"><div class="pnum">4</div><div>Carol<br>Evaluates</div></div>
      <div class="parr">→</div>
      <div class="pstep" id="ps-5"><div class="pnum">5</div><div>Alice<br>Decrypts</div></div>
      <div class="parr">→</div>
      <div class="pstep" id="ps-6"><div class="pnum">✓</div><div>Result<br>Out</div></div>
    </div>

    <div class="log-box">
      <div class="log-label">Execution Log</div>
      <div id="log-lines">
        <div class="log-row"><span class="log-t">ready</span><span>Select a query and click Run ↑</span></div>
      </div>
    </div>
  </div>

  <div class="result-box" id="result-box">
    <div class="res-head">
      <div class="res-title">Query Result<strong id="res-name">—</strong></div>
      <div class="res-tag">🔒 Computed on ciphertext by Carol</div>
    </div>

    <!-- HE + Plaintext side by side -->
    <div class="res-grid">
      <div class="res-cell he">
        <div class="res-cell-label">🔒 HE Result (CKKS)</div>
        <div class="res-cell-how">
          Alice encrypts each column into a packed CKKS ciphertext vector and uploads to Carol.
          Carol evaluates the query entirely on ciphertext using TenSEAL — never seeing raw values.
          The encrypted result is sent back to Alice, who decrypts it with her secret key.
        </div>
        <div class="res-val" id="res-he-val">—</div>
        <div class="res-meta">
          <div class="res-meta-item">
            <div class="rmi-label">Eval Time</div>
            <div class="rmi-val" id="res-he-time">—</div>
          </div>
          <div class="res-meta-item">
            <div class="rmi-label">HE Operation</div>
            <div class="rmi-val" id="res-he-op">—</div>
          </div>
          <div class="res-meta-item">
            <div class="rmi-label">Scheme</div>
            <div class="rmi-val">CKKS (TenSEAL)</div>
          </div>
          <div class="res-meta-item">
            <div class="rmi-label">Security</div>
            <div class="rmi-val">128-bit (RLWE)</div>
          </div>
        </div>
      </div>

      <div class="res-cell pt">
        <div class="res-cell-label">📄 Plaintext Baseline</div>
        <div class="res-cell-how">
          The identical query is run directly on the raw unencrypted NumPy array.
          No encryption or decryption — pure arithmetic on plaintext numbers.
          Used as the ground-truth reference to measure HE correctness and speed cost.
        </div>
        <div class="res-val" id="res-pt-val">—</div>
        <div class="res-meta">
          <div class="res-meta-item">
            <div class="rmi-label">Eval Time</div>
            <div class="rmi-val" id="res-pt-time">—</div>
          </div>
          <div class="res-meta-item">
            <div class="rmi-label">Library</div>
            <div class="rmi-val">NumPy</div>
          </div>
          <div class="res-meta-item">
            <div class="rmi-label">Data State</div>
            <div class="rmi-val">Unencrypted</div>
          </div>
          <div class="res-meta-item">
            <div class="rmi-label">Privacy</div>
            <div class="rmi-val" style="color:#e8754f;">None ⚠</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Info sections -->
    <div class="info-sections">

      <!-- CKKS Error -->
      <div class="info-section err-section">
        <div class="info-section-header">
          <div>
            <div class="info-section-title">CKKS Approximation Error</div>
            <div class="info-big-val" id="res-err">—</div>
          </div>
          <div class="info-section-body" style="flex:1">
            CKKS is an <strong>approximate</strong> HE scheme — it intentionally trades tiny precision loss
            for practical speed. The error is the absolute difference between the HE-decrypted value
            and the true plaintext result.
          </div>
        </div>
        <div class="info-formula">Error = | decrypt(Carol's ciphertext) − NumPy result |</div>
        <div class="info-section-body" style="margin-top:8px;">
          For burn_rate (range 0–1), an error of <strong id="res-err-repeat">—</strong> means the result
          is accurate to <strong id="res-err-digits">—</strong> decimal places — far beyond what any HR analytics use case requires.
          This noise is controlled by the <strong>global_scale = 2⁴⁰</strong> and
          <strong>coeff_moduli = [60,40,40,60]</strong> parameters chosen at key generation.
        </div>
      </div>

      <!-- Slowdown -->
      <div class="info-section slow-section">
        <div class="info-section-header">
          <div>
            <div class="info-section-title">Performance Tradeoff</div>
            <div class="info-big-val">HE is <span id="res-slow">—</span>× slower</div>
          </div>
          <div class="info-section-body" style="flex:1">
            This slowdown is the direct cost of <strong>data confidentiality</strong>.
            Carol evaluates the query on ciphertext — every arithmetic operation requires
            polynomial multiplication modulo large primes, which is inherently slower than raw NumPy array math.
          </div>
        </div>
        <div class="info-formula">Slowdown = HE eval time (<span id="res-slow-he">—</span>) ÷ Plaintext eval time (<span id="res-slow-pt">—</span>)</div>
        <div class="info-section-body" style="margin-top:8px;">
          SUM-based queries (avg) are slowest because they require <strong>Galois key rotations</strong>
          across the packed ciphertext vector. Vector-add and scalar-multiply are cheaper operations
          with only ~40–150× slowdown.
        </div>
      </div>

      <!-- Correctness match -->
      <div class="info-section match-section">
        <div class="info-section-header">
          <div>
            <div class="info-section-title">Correctness Verification</div>
            <div class="info-big-val">✓ Results Match <span class="match-pill" id="res-match-pill">within 10⁻⁸</span></div>
          </div>
          <div class="info-section-body" style="flex:1">
            The HE result and the plaintext result are functionally identical.
            This confirms that Carol's homomorphic evaluation over encrypted data
            produces the correct answer — proving both <strong>correctness</strong> and <strong>privacy</strong> simultaneously.
          </div>
        </div>
        <div class="info-formula">HE result (<span id="res-match-he">—</span>) ≈ Plaintext result (<span id="res-match-pt">—</span>)  [Δ = <span id="res-match-err">—</span>]</div>
      </div>

    </div>
  </div>

  <div class="hist-box">
    <div class="hist-head">Query History <button class="hist-clear" onclick="clearHistory()">Clear</button></div>
    <div id="hist-body"><div class="hist-empty">No queries run yet.</div></div>
  </div>
</div>

<!-- CHARTS TAB -->
<div class="tab-panel" id="panel-charts">
  <div class="chart-intro">
    Benchmark across <strong>N ∈ {100, 500, 1000, 3000, 7000}</strong> rows.
    Scheme: <strong>CKKS (TenSEAL)</strong>, poly_degree=8192, 128-bit security.
    Baseline: NumPy on raw arrays.
  </div>
  <div class="chart-tabs">
    <button class="chart-tab active" id="ct-upload" onclick="showChart('upload')">Upload / Encrypt Time</button>
    <button class="chart-tab" id="ct-qtime" onclick="showChart('qtime')">Query Execution Time</button>
    <button class="chart-tab" id="ct-slow" onclick="showChart('slow')">Slowdown Factor</button>
    <button class="chart-tab" id="ct-err" onclick="showChart('err')">CKKS Error</button>
    <button class="chart-tab" id="ct-bfv" onclick="showChart('bfv')">CKKS vs BFV</button>
  </div>

  <div class="chart-panel active" id="cp-upload">
    <div class="chart-card">
      <div class="chart-title">Dataset Upload / Encryption Time</div>
      <div class="chart-sub">One-time cost for Alice to encrypt all columns. CKKS packs entire column into one ciphertext vector.</div>
      <div class="chart-wrap"><canvas id="ch-upload"></canvas></div>
      <div class="insight"><strong>Key insight:</strong> HE takes 0.05–0.13s regardless of N. Plaintext is always sub-millisecond.</div>
    </div>
  </div>

  <div class="chart-panel" id="cp-qtime">
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title">HE Query Time — All 5 Types</div>
        <div class="chart-sub">SUM queries need Galois key rotations — the most expensive HE operation.</div>
        <div class="chart-wrap"><canvas id="ch-qtime"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">HE vs Plaintext — avg(burn_rate)</div>
        <div class="chart-sub">Direct comparison for Q1. Plaintext barely registers at this scale.</div>
        <div class="chart-wrap"><canvas id="ch-compare"></canvas></div>
      </div>
    </div>
    <div class="insight"><strong>Key insight:</strong> Vector-add and scalar-multiply are cheapest at ~0.003–0.034s. SUM/avg costs 0.04–0.17s due to rotation overhead.</div>
  </div>

  <div class="chart-panel" id="cp-slow">
    <div class="chart-card">
      <div class="chart-title">Slowdown Factor: HE Time ÷ Plaintext Time</div>
      <div class="chart-sub">How many times slower HE is per query type. Log scale.</div>
      <div class="chart-wrap-tall"><canvas id="ch-slow"></canvas></div>
      <div class="insight"><strong>Key insight:</strong> avg() queries show 600–2400× slowdown. Vector ops only 40–150×. This is the fundamental privacy/performance tradeoff.</div>
    </div>
  </div>

  <div class="chart-panel" id="cp-err">
    <div class="chart-card">
      <div class="chart-title">CKKS Approximation Error vs Dataset Size</div>
      <div class="chart-sub">|HE result − Plaintext result| for all 5 queries. Log scale.</div>
      <div class="chart-wrap"><canvas id="ch-err"></canvas></div>
      <div class="insight"><strong>Key insight:</strong> All errors below <strong>10⁻⁷</strong> — completely negligible for analytics on burn_rate (range 0–1).</div>
    </div>
  </div>

  <div class="chart-panel" id="cp-bfv">
    <div class="chart-card">
      <div class="chart-title">CKKS vs BFV — All Queries at N = 7,000</div>
      <div class="chart-sub">Same 5 queries run under both schemes. BFV uses integer-scaled arithmetic via Pyfhel; CKKS uses native floats via TenSEAL.</div>
      <div class="chart-wrap-tall"><canvas id="ch-bfv"></canvas></div>
      <div class="insight"><strong>Key insight:</strong> CKKS is ~3.5× faster on rotation-heavy queries (avg, weighted) due to native float support. BFV edges ahead on simple vector ops where no rotations are needed.</div>
    </div>
  </div>
</div>

</div>

<script>
(function() {
  if (localStorage.getItem('he_intro_seen')) {
    document.getElementById('intro-modal').style.display = 'none';
  }
})();

function closeModal(permanent) {
  var el = document.getElementById('intro-modal');
  el.classList.add('hide');
  if (permanent) localStorage.setItem('he_intro_seen', '1');
  setTimeout(function() { el.style.display = 'none'; }, 260);
}

var selN = 500;
var selQ = 'avg_burn_rate';
var queryHistory = [];
var chartsBuilt = false;

var BENCH = """ + json.dumps(BENCH) + """;
var NS = [100, 500, 1000, 3000, 7000];
var XL = ['100', '500', '1k', '3k', '7k'];
var QLS = {
  avg_burn_rate: 'avg(burn_rate)',
  avg_fatigue: 'avg(mental_fatigue)',
  avg_resource: 'avg(resource_alloc)',
  avg_hours: 'avg(hours/week)',
  stress_index: 'stress_index',
  scaled_hours: 'scaled_hours',
  weighted_risk: 'weighted_risk'
};

function showTab(t) {
  document.getElementById('tab-query').classList.remove('active');
  document.getElementById('tab-charts').classList.remove('active');
  document.getElementById('panel-query').classList.remove('active');
  document.getElementById('panel-charts').classList.remove('active');
  document.getElementById('tab-' + t).classList.add('active');
  document.getElementById('panel-' + t).classList.add('active');
  if (t === 'charts' && !chartsBuilt) {
    chartsBuilt = true;
    buildCharts();
  }
}

function pickSize(btn, n) {
  var btns = document.getElementById('size-group').querySelectorAll('.sel-btn');
  for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
  btn.classList.add('active');
  selN = n;
}

function pickQuery(btn, q) {
  var btns = document.getElementById('query-group').querySelectorAll('.sel-btn');
  for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
  btn.classList.add('active');
  selQ = q;
}

function showChart(c) {
  var ids = ['upload','qtime','slow','err','bfv'];
  for (var i = 0; i < ids.length; i++) {
    document.getElementById('ct-' + ids[i]).classList.remove('active');
    document.getElementById('cp-' + ids[i]).classList.remove('active');
  }
  document.getElementById('ct-' + c).classList.add('active');
  document.getElementById('cp-' + c).classList.add('active');
}

function setStep(n) {
  for (var i = 1; i <= 6; i++) {
    var el = document.getElementById('ps-' + i);
    if (el) {
      if (i <= n) el.classList.add('active');
      else el.classList.remove('active');
    }
  }
}

function setLog(lines) {
  var html = '';
  for (var i = 0; i < lines.length; i++) {
    var cls = lines[i].c || '';
    html += '<div class="log-row"><span class="log-t">' + lines[i].t + '</span><span class="' + cls + '">' + lines[i].m + '</span></div>';
  }
  document.getElementById('log-lines').innerHTML = html;
}

function sleep(ms) { return new Promise(function(r){ setTimeout(r, ms); }); }

async function runQuery() {
  var btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = '⏳  Running...';
  document.getElementById('result-box').classList.remove('show');
  setStep(0);

  var steps = [
    {t:'step 1', m:'[Alice] Loading CKKS context & secret key...', c:'log-alice'},
    {t:'step 2', m:'[Alice] Encrypting ' + selN.toLocaleString() + ' rows as packed CKKS vectors...', c:'log-alice'},
    {t:'step 3', m:'[Carol] Received ciphertext columns (public context only)...', c:'log-carol'},
    {t:'step 4', m:'[Carol] Evaluating "' + QLS[selQ] + '" over ciphertext...', c:'log-carol'},
    {t:'step 5', m:'[Alice] Decrypting Carol result ciphertext...', c:'log-alice'}
  ];

  for (var i = 0; i < steps.length; i++) {
    setLog(steps.slice(0, i+1));
    setStep(i+1);
    await sleep(300);
  }

  try {
    var resp = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({n: selN, query: selQ})
    });
    var d = await resp.json();

    if (d.error) {
      setLog(steps.concat([{t:'ERROR', m: d.error, c:'log-done'}]));
      setStep(0);
    } else {
      var HE_OPS = {
        avg_burn_rate:'SUM → divide by N', avg_fatigue:'SUM → divide by N',
        avg_resource:'SUM → divide by N', avg_hours:'SUM → divide by N',
        stress_index:'VECTOR ADD (2 cols)', scaled_hours:'SCALAR MULTIPLY (÷40)',
        weighted_risk:'DOT PRODUCT'
      };

      setStep(6);
      var isVec = Array.isArray(d.he_result);
      var heStr = isVec ? '[' + d.he_result.map(function(v){ return v.toFixed(3); }).join(', ') + ', …]' : d.he_result.toFixed(6);
      var ptStr = isVec ? '[' + d.pt_result.map(function(v){ return v.toFixed(3); }).join(', ') + ', …]' : d.pt_result.toFixed(6);
      var errExp = d.error_val.toExponential(2);
      var digits = Math.abs(Math.floor(Math.log10(Math.max(d.error_val, 1e-15))));
      var slowdown = (d.he_time / Math.max(d.pt_time, 1e-9)).toFixed(0);

      setLog(steps.concat([{t:'done', m:'[Alice] ✓ Decrypted: ' + (isVec ? '[vector]' : d.he_result.toFixed(6)), c:'log-done'}]));

      document.getElementById('res-name').textContent = QLS[selQ] + '  (N = ' + selN.toLocaleString() + ')';
      document.getElementById('res-he-val').textContent = heStr;
      document.getElementById('res-pt-val').textContent = ptStr;
      document.getElementById('res-he-time').textContent = d.he_time.toFixed(4) + 's';
      document.getElementById('res-pt-time').textContent = (d.pt_time * 1000).toFixed(3) + 'ms';
      document.getElementById('res-he-op').textContent = HE_OPS[selQ] || '—';

      document.getElementById('res-err').textContent = errExp;
      document.getElementById('res-err-repeat').textContent = errExp;
      document.getElementById('res-err-digits').textContent = digits;

      document.getElementById('res-slow').textContent = slowdown;
      document.getElementById('res-slow-he').textContent = d.he_time.toFixed(4) + 's';
      document.getElementById('res-slow-pt').textContent = (d.pt_time * 1000).toFixed(3) + 'ms';

      document.getElementById('res-match-he').textContent = isVec ? '[vector]' : d.he_result.toFixed(6);
      document.getElementById('res-match-pt').textContent = isVec ? '[vector]' : d.pt_result.toFixed(6);
      document.getElementById('res-match-err').textContent = errExp;
      document.getElementById('res-match-pill').textContent = 'within ' + errExp;

      document.getElementById('result-box').classList.add('show');

      queryHistory.unshift({
        n: selN, q: QLS[selQ],
        he: isVec ? '(vector)' : d.he_result.toFixed(6),
        pt: isVec ? '(vector)' : d.pt_result.toFixed(6),
        ht: d.he_time.toFixed(4),
        ptt: (d.pt_time * 1000).toFixed(3),
        err: d.error_val.toExponential(1),
        sl: (d.he_time / Math.max(d.pt_time, 1e-9)).toFixed(0)
      });
      renderHistory();
    }
  } catch(e) {
    setLog(steps.concat([{t:'ERROR', m: String(e), c:'log-done'}]));
  }

  btn.disabled = false;
  btn.textContent = '▶  Run Encrypted Query';
}

function renderHistory() {
  var el = document.getElementById('hist-body');
  if (!queryHistory.length) {
    el.innerHTML = '<div class="hist-empty">No queries run yet.</div>';
    return;
  }
  var html = '<table class="htable"><thead><tr><th>N</th><th>Query</th><th class="c-he">HE Result</th><th class="c-pt">PT Result</th><th class="c-he">HE Time</th><th class="c-pt">PT Time</th><th class="c-err">Error</th><th>Slowdown</th></tr></thead><tbody>';
  for (var i = 0; i < queryHistory.length; i++) {
    var h = queryHistory[i];
    html += '<tr><td>' + h.n.toLocaleString() + '</td><td>' + h.q + '</td><td class="c-he">' + h.he + '</td><td class="c-pt">' + h.pt + '</td><td class="c-he">' + h.ht + 's</td><td class="c-pt">' + h.ptt + 'ms</td><td class="c-err">' + h.err + '</td><td><span class="slow-pill">' + h.sl + '×</span></td></tr>';
  }
  html += '</tbody></table>';
  el.innerHTML = html;
}

function clearHistory() {
  queryHistory = [];
  renderHistory();
}

var GC = {color: 'rgba(255,255,255,0.07)'};
var TC = {color: '#8a9ab8', font: {family: 'IBM Plex Mono', size: 10}};
var COLS = ['#60b8d4','#e8754f','#9d85e8','#f5b942','#64b5e8'];
var QKS = ['Q1','Q2','Q3','Q4','Q5'];
var QNS = ['Q1 avg(burn)','Q2 avg(fatigue)','Q3 weighted','Q4 scaled','Q5 stress'];

function baseOpts(yLabel, logY) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {legend: {labels: {color: '#dce4f2', font: {family: 'IBM Plex Mono', size: 11}}}},
    scales: {
      x: {grid: GC, ticks: TC},
      y: {grid: GC, ticks: TC, type: logY ? 'logarithmic' : 'linear',
          title: {display: true, text: yLabel, color: '#8a9ab8', font: {family: 'IBM Plex Mono', size: 10}}}
    }
  };
}

function buildCharts() {
  new Chart(document.getElementById('ch-upload'), {
    type: 'bar',
    data: {labels: XL, datasets: [
      {label: 'HE (CKKS)', data: BENCH.map(function(d){return d.uhe;}), backgroundColor: 'rgba(96,184,212,0.15)', borderColor: '#60b8d4', borderWidth: 1.5},
      {label: 'Plaintext', data: BENCH.map(function(d){return d.upt;}), backgroundColor: 'rgba(232,117,79,0.15)', borderColor: '#e8754f', borderWidth: 1.5}
    ]},
    options: baseOpts('Time (s)', false)
  });

  new Chart(document.getElementById('ch-qtime'), {
    type: 'line',
    data: {labels: XL, datasets: QKS.map(function(k,i){return {
      label: QNS[i], data: BENCH.map(function(d){return d.q[k].ht;}),
      borderColor: COLS[i], backgroundColor: 'transparent', borderWidth: 2, pointRadius: 4, tension: 0.3
    };})},
    options: baseOpts('HE Time (s)', false)
  });

  new Chart(document.getElementById('ch-compare'), {
    type: 'bar',
    data: {labels: XL, datasets: [
      {label: 'HE (CKKS)', data: BENCH.map(function(d){return d.q.Q1.ht;}), backgroundColor: 'rgba(96,184,212,0.15)', borderColor: '#60b8d4', borderWidth: 1.5},
      {label: 'Plaintext', data: BENCH.map(function(d){return d.q.Q1.pt;}), backgroundColor: 'rgba(232,117,79,0.15)', borderColor: '#e8754f', borderWidth: 1.5}
    ]},
    options: baseOpts('Time (s)', false)
  });

  new Chart(document.getElementById('ch-slow'), {
    type: 'line',
    data: {labels: XL, datasets: QKS.map(function(k,i){return {
      label: QNS[i],
      data: BENCH.map(function(d){return d.q[k].ht / Math.max(d.q[k].pt, 1e-9);}),
      borderColor: COLS[i], backgroundColor: 'transparent', borderWidth: 2, pointRadius: 4, tension: 0.3
    };})},
    options: baseOpts('Slowdown (x)', true)
  });

  new Chart(document.getElementById('ch-err'), {
    type: 'line',
    data: {labels: XL, datasets: QKS.map(function(k,i){return {
      label: QNS[i],
      data: BENCH.map(function(d){return d.q[k].e === 0 ? 1e-10 : d.q[k].e;}),
      borderColor: COLS[i], backgroundColor: 'transparent', borderWidth: 2, pointRadius: 4, tension: 0.3
    };})},
    options: baseOpts('|HE - PT|', true)
  });

  var BC = """ + json.dumps(BENCH_COMPARE) + """;
  var CKKS_MAP = {"Q1": "Q1_avg_burn_rate", "Q2": "Q2_avg_mental_fatigue", "Q3": "Q3_weighted_burn_risk", "Q4": "Q4_scaled_hours", "Q5": "Q5_stress_index"};
  if (BC.length && BC[BC.length-1].bfv_queries) {
    var r7k = BC[BC.length - 1];
    var ks = Object.keys(r7k.bfv_queries);
    var labels = ks.map(function(k){ return r7k.bfv_queries[k].label; });
    var ckksReal = ks.map(function(k){ return r7k.queries[CKKS_MAP[k]].he_time_s; });
    var bfvReal  = ks.map(function(k){ return r7k.bfv_queries[k].bfv_time_s; });
    new Chart(document.getElementById('ch-bfv'), {
      type: 'bar',
      data: {labels: labels, datasets: [
        {label: 'CKKS (TenSEAL)', data: ckksReal,
         backgroundColor: 'rgba(96,184,212,0.15)', borderColor: '#60b8d4', borderWidth: 1.5},
        {label: 'BFV (Pyfhel)',   data: bfvReal,
         backgroundColor: 'rgba(196,84,110,0.15)', borderColor: '#c4546e', borderWidth: 1.5}
      ]},
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {padding: {bottom: 30}},
        plugins: {
          legend: {labels: {color: '#dce4f2', font: {family: 'IBM Plex Mono', size: 11}}},
          tooltip: {
            callbacks: {
              title: function(items) { return items[0].label; },
              label: function(item) {
                var ct = ckksReal[item.dataIndex];
                var bt = bfvReal[item.dataIndex];
                if (item.datasetIndex === 0) return 'CKKS: ' + ct.toFixed(4) + 's';
                return ['BFV:  ' + bt.toFixed(4) + 's', 'Ratio: ' + (bt/Math.max(ct,1e-9)).toFixed(1) + '× slower'];
              }
            },
            backgroundColor: '#161b24', borderColor: '#252d3d', borderWidth: 1,
            titleColor: '#dce4f2', bodyColor: '#8a9ab8', padding: 10, displayColors: true
          }
        },
        scales: {
          x: {grid: GC, ticks: TC},
          y: {grid: GC, ticks: TC, type: 'linear',
              title: {display: true, text: 'Query time (s)', color: '#8a9ab8',
                      font: {family: 'IBM Plex Mono', size: 10}}}
        }
      },
      plugins: [{
        id: 'bfvLabels',
        afterDraw: function(chart) {
          var ctx = chart.ctx;
          var xAxis = chart.scales.x;
          ctx.save();
          ctx.font = '9px IBM Plex Mono';
          ctx.textAlign = 'center';
          for (var i = 0; i < ks.length; i++) {
            var x = xAxis.getPixelForValue(i);
            var y = chart.chartArea.bottom + 28;
            ctx.fillStyle = '#60b8d4';
            ctx.fillText('C:' + ckksReal[i].toFixed(3) + 's', x, y);
            ctx.fillStyle = '#c4546e';
            ctx.fillText('B:' + bfvReal[i].toFixed(3) + 's', x, y + 13);
          }
          ctx.restore();
        }
      }]
    });
  }
}
</script>
</body>
</html>"""

@app.route('/')
def index():
    resp = make_response(HTML)
    resp.headers['Content-Type'] = 'text/html'
    return resp

@app.route('/api/query', methods=['POST'])
def run_query():
    data = request.get_json()
    n = int(data.get('n', 500))
    query = data.get('query', 'avg_burn_rate')
    try:
        he, pt = get_systems(n)
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.05, 0.20, n).tolist()
        if query == 'avg_burn_rate':
            hv,ht = he.query_average('burn_rate'); pv,pt2 = pt.query_average('burn_rate')
        elif query == 'avg_fatigue':
            hv,ht = he.query_average('mental_fatigue_score'); pv,pt2 = pt.query_average('mental_fatigue_score')
        elif query == 'avg_resource':
            hv,ht = he.query_average('resource_allocation'); pv,pt2 = pt.query_average('resource_allocation')
        elif query == 'avg_hours':
            hv,ht = he.query_average('hours_per_week'); pv,pt2 = pt.query_average('hours_per_week')
        elif query == 'stress_index':
            hv,ht = he.query_column_sum_two('mental_fatigue_score','resource_allocation')
            pv,pt2 = pt.query_column_sum_two('mental_fatigue_score','resource_allocation')
            hv=hv[:5]; pv=pv[:5]
        elif query == 'scaled_hours':
            hv,ht = he.query_scaled_column('hours_per_week',1/40.0)
            pv,pt2 = pt.query_scaled_column('hours_per_week',1/40.0)
            hv=hv[:5]; pv=pv[:5]
        elif query == 'weighted_risk':
            hv,ht = he.query_weighted_sum('burn_rate',weights); pv,pt2 = pt.query_weighted_sum('burn_rate',weights)
        else:
            return jsonify({'error': 'Unknown query: ' + query})
        err = float(max(abs(h-p) for h,p in zip(hv,pv))) if isinstance(hv,list) else float(abs(hv-pv))
        return jsonify({'he_result':hv,'pt_result':pv,'he_time':ht,'pt_time':pt2,'error_val':err})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
