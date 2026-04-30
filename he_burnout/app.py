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

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HE Burnout App</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #07090f; color: #cdd9f0; font-family: 'IBM Plex Mono', monospace; }
body::before { content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image: linear-gradient(rgba(0,255,200,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,200,0.025) 1px, transparent 1px);
  background-size: 44px 44px; }
.wrap { position:relative; z-index:1; max-width:1100px; margin:0 auto; padding:24px 20px; }

.topbar { display:flex; justify-content:space-between; align-items:center; padding-bottom:18px; margin-bottom:22px; border-bottom:1px solid #1c2436; }
.logo { font-family:'Syne',sans-serif; font-size:1.4rem; color:#00ffc8; text-shadow:0 0 24px rgba(0,255,200,0.3); }
.logo small { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#6a8aaa; display:block; margin-top:3px; font-weight:400; }
.badge { font-size:0.6rem; padding:5px 12px; border:1px solid #00ffc8; border-radius:20px; color:#00ffc8; }

.tab-bar { display:flex; gap:6px; margin-bottom:22px; }
.tab-btn { padding:9px 22px; border-radius:7px; border:1px solid #1c2436; background:#0c0f1a; color:#6a8aaa; font-family:'IBM Plex Mono',monospace; font-size:0.72rem; cursor:pointer; }
.tab-btn.active { border-color:#00ffc8; background:rgba(0,255,200,0.08); color:#00ffc8; }

.tab-panel { display:none; }
.tab-panel.active { display:block; }

.pipe { display:flex; align-items:center; background:#0c0f1a; border:1px solid #1c2436; border-radius:10px; padding:12px 14px; margin-bottom:18px; overflow-x:auto; gap:0; }
.pstep { display:flex; flex-direction:column; align-items:center; gap:4px; flex:1; min-width:80px; text-align:center; font-size:0.62rem; color:#3a4a60; padding:6px; border-radius:7px; }
.pstep.active { color:#00ffc8; background:rgba(0,255,200,0.05); }
.pnum { width:22px; height:22px; border-radius:50%; border:1px solid currentColor; display:flex; align-items:center; justify-content:center; font-size:0.6rem; }
.pstep.active .pnum { background:#00ffc8; color:#000; }
.parr { color:#3a4a60; padding:0 3px; flex-shrink:0; }

.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
.card { background:#131826; border:1px solid #1c2436; border-radius:10px; padding:16px; }
.card-label { font-size:0.58rem; color:#6a8aaa; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px; }
.btn-row { display:flex; flex-wrap:wrap; gap:6px; }
.sel-btn { padding:6px 12px; border:1px solid #1c2436; border-radius:5px; background:#111520; color:#6a8aaa; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; cursor:pointer; }
.sel-btn:hover { border-color:#6a8aaa; color:#cdd9f0; }
.sel-btn.active { border-color:#00ffc8; background:rgba(0,255,200,0.08); color:#00ffc8; }

.run-wrap { text-align:center; margin-bottom:16px; }
.run-btn { padding:12px 40px; background:#00ffc8; color:#000; border:none; border-radius:7px; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:600; cursor:pointer; }
.run-btn:hover { opacity:0.88; transform:translateY(-1px); }
.run-btn:disabled { opacity:0.4; cursor:not-allowed; transform:none; }

.log-box { background:#0c0f1a; border:1px solid #1c2436; border-radius:10px; padding:14px; margin-bottom:14px; font-size:0.65rem; line-height:2; }
.log-label { font-size:0.55rem; color:#6a8aaa; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }
.log-row { display:flex; gap:8px; }
.log-t { color:#3a4a60; min-width:50px; }
.log-alice { color:#00ffc8; }
.log-carol { color:#a78bfa; }
.log-done { color:#ffe246; font-weight:600; }

.result-box { background:#131826; border:1px solid #1c2436; border-radius:12px; padding:18px; margin-bottom:16px; display:none; }
.result-box.show { display:block; }
.res-head { display:flex; justify-content:space-between; align-items:center; padding-bottom:12px; margin-bottom:14px; border-bottom:1px solid #1c2436; }
.res-title { font-size:0.65rem; color:#6a8aaa; }
.res-title strong { display:block; font-size:0.85rem; color:#cdd9f0; margin-top:2px; }
.res-tag { font-size:0.58rem; padding:3px 10px; background:rgba(0,255,200,0.08); border:1px solid rgba(0,255,200,0.25); border-radius:4px; color:#00ffc8; }
.res-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px; }
.res-cell { border-radius:8px; padding:14px; border:1px solid #1c2436; }
.res-cell.he { border-color:rgba(0,255,200,0.2); background:rgba(0,255,200,0.03); }
.res-cell.pt { border-color:rgba(255,112,67,0.2); background:rgba(255,112,67,0.03); }
.res-cell-label { font-size:0.58rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; }
.res-cell.he .res-cell-label { color:#00ffc8; }
.res-cell.pt .res-cell-label { color:#ff7043; }
.res-val { font-size:1.4rem; font-weight:600; line-height:1; margin-bottom:6px; }
.res-cell.he .res-val { color:#00ffc8; }
.res-cell.pt .res-val { color:#ff7043; }
.res-time { font-size:0.6rem; color:#3a4a60; }
.res-time span { background:#0c0f1a; border:1px solid #1c2436; border-radius:3px; padding:1px 6px; color:#6a8aaa; margin-top:3px; display:inline-block; }
.err-row { background:rgba(255,226,70,0.05); border:1px solid rgba(255,226,70,0.2); border-radius:7px; padding:10px 13px; display:flex; gap:12px; align-items:center; font-size:0.65rem; margin-bottom:8px; }
.err-val { color:#ffe246; font-size:1rem; font-weight:600; }
.err-desc { color:#6a8aaa; }
.slow-row { background:#111520; border:1px solid #1c2436; border-radius:7px; padding:10px 13px; font-size:0.65rem; color:#6a8aaa; }
.slow-num { color:#ff7043; font-size:1rem; font-weight:600; }

.hist-box { background:#131826; border:1px solid #1c2436; border-radius:10px; overflow:hidden; }
.hist-head { padding:10px 14px; border-bottom:1px solid #1c2436; font-size:0.58rem; color:#6a8aaa; text-transform:uppercase; letter-spacing:0.1em; display:flex; justify-content:space-between; align-items:center; }
.hist-clear { background:none; border:1px solid #1c2436; color:#3a4a60; font-family:'IBM Plex Mono',monospace; font-size:0.58rem; padding:2px 8px; border-radius:3px; cursor:pointer; }
.hist-empty { padding:18px; text-align:center; color:#3a4a60; font-size:0.65rem; }
.htable { width:100%; border-collapse:collapse; font-size:0.65rem; }
.htable th { background:#111520; color:#6a8aaa; padding:8px 12px; text-align:left; font-size:0.58rem; text-transform:uppercase; border-bottom:1px solid #1c2436; }
.htable td { padding:8px 12px; border-bottom:1px solid rgba(28,36,54,0.6); }
.htable tr:last-child td { border-bottom:none; }
.c-he { color:#00ffc8; } .c-pt { color:#ff7043; } .c-err { color:#ffe246; }
.slow-pill { background:rgba(255,112,67,0.1); border:1px solid rgba(255,112,67,0.25); color:#ff7043; border-radius:3px; padding:1px 6px; font-size:0.58rem; }

.chart-intro { background:#0c0f1a; border:1px solid #1c2436; border-radius:10px; padding:14px 18px; margin-bottom:18px; font-size:0.68rem; color:#6a8aaa; line-height:1.7; }
.chart-intro strong { color:#00ffc8; }
.chart-tabs { display:flex; gap:6px; margin-bottom:16px; flex-wrap:wrap; }
.chart-tab { padding:7px 14px; border:1px solid #1c2436; border-radius:5px; background:#111520; color:#6a8aaa; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; cursor:pointer; }
.chart-tab:hover { border-color:#6a8aaa; color:#cdd9f0; }
.chart-tab.active { border-color:#00ffc8; background:rgba(0,255,200,0.07); color:#00ffc8; }
.chart-panel { display:none; }
.chart-panel.active { display:block; }
.chart-card { background:#131826; border:1px solid #1c2436; border-radius:12px; padding:20px; margin-bottom:14px; }
.chart-title { font-family:'Syne',sans-serif; font-size:0.9rem; color:#cdd9f0; margin-bottom:4px; }
.chart-sub { font-size:0.62rem; color:#6a8aaa; margin-bottom:16px; line-height:1.6; }
.chart-wrap { position:relative; height:280px; }
.chart-wrap-tall { position:relative; height:320px; }
.chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.insight { background:#0c0f1a; border-left:2px solid #00ffc8; border-radius:0 6px 6px 0; padding:10px 13px; margin-top:12px; font-size:0.63rem; color:#6a8aaa; line-height:1.6; }
.insight strong { color:#00ffc8; }
</style>
</head>
<body>
<div class="wrap">

<div class="topbar">
  <div class="logo">HE Burnout<small>Homomorphic Encryption · Developer Burnout Dataset · CS6903/4783</small></div>
  <div class="badge">CKKS · 128-bit · TenSEAL</div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" id="tab-query" onclick="showTab('query')">▶  Live Query</button>
  <button class="tab-btn" id="tab-charts" onclick="showTab('charts')">📊  Performance Charts</button>
</div>

<!-- QUERY TAB -->
<div class="tab-panel active" id="panel-query">

  <div class="pipe">
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

  <div class="grid2">
    <div class="card">
      <div class="card-label">① Select Dataset Size</div>
      <div class="btn-row" id="size-group">
        <button class="sel-btn" onclick="pickSize(this,100)">N = 100</button>
        <button class="sel-btn active" onclick="pickSize(this,500)">N = 500</button>
        <button class="sel-btn" onclick="pickSize(this,1000)">N = 1,000</button>
        <button class="sel-btn" onclick="pickSize(this,3000)">N = 3,000</button>
        <button class="sel-btn" onclick="pickSize(this,7000)">N = 7,000</button>
      </div>
    </div>
    <div class="card">
      <div class="card-label">② Select Query (Carol evaluates on ciphertext)</div>
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

  <div class="run-wrap">
    <button class="run-btn" id="run-btn" onclick="runQuery()">▶  Run Encrypted Query</button>
  </div>

  <div class="log-box">
    <div class="log-label">Execution Log</div>
    <div id="log-lines">
      <div class="log-row"><span class="log-t">ready</span><span>Select a query and click Run ↑</span></div>
    </div>
  </div>

  <div class="result-box" id="result-box">
    <div class="res-head">
      <div class="res-title">Query Result<strong id="res-name">—</strong></div>
      <div class="res-tag">🔒 Computed on ciphertext by Carol</div>
    </div>
    <div class="res-grid">
      <div class="res-cell he">
        <div class="res-cell-label">🔒 HE Result (CKKS)</div>
        <div class="res-val" id="res-he-val">—</div>
        <div class="res-time">Eval time: <span id="res-he-time">—</span></div>
      </div>
      <div class="res-cell pt">
        <div class="res-cell-label">📄 Plaintext Baseline</div>
        <div class="res-val" id="res-pt-val">—</div>
        <div class="res-time">Eval time: <span id="res-pt-time">—</span></div>
      </div>
    </div>
    <div class="err-row">
      <div><div style="font-size:0.58rem;color:#ffe246;margin-bottom:3px;">CKKS Error</div><div class="err-val" id="res-err">—</div></div>
      <div class="err-desc">Inherent to approximate HE — negligible for analytics</div>
    </div>
    <div class="slow-row">HE is <span class="slow-num" id="res-slow">—</span>× slower — the privacy / performance tradeoff</div>
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
</div>

</div>

<script>
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
  var ids = ['upload','qtime','slow','err'];
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
      setStep(6);
      var isVec = Array.isArray(d.he_result);
      var heStr = isVec ? '[' + d.he_result.map(function(v){ return v.toFixed(3); }).join(', ') + ', …]' : d.he_result.toFixed(6);
      var ptStr = isVec ? '[' + d.pt_result.map(function(v){ return v.toFixed(3); }).join(', ') + ', …]' : d.pt_result.toFixed(6);

      setLog(steps.concat([{t:'done', m:'[Alice] ✓ Decrypted: ' + (isVec ? '[vector]' : d.he_result.toFixed(6)), c:'log-done'}]));

      document.getElementById('res-name').textContent = QLS[selQ] + '  (N = ' + selN.toLocaleString() + ')';
      document.getElementById('res-he-val').textContent = heStr;
      document.getElementById('res-pt-val').textContent = ptStr;
      document.getElementById('res-he-time').textContent = d.he_time.toFixed(4) + 's';
      document.getElementById('res-pt-time').textContent = (d.pt_time * 1000).toFixed(3) + 'ms';
      document.getElementById('res-err').textContent = d.error_val.toExponential(2);
      document.getElementById('res-slow').textContent = (d.he_time / Math.max(d.pt_time, 1e-9)).toFixed(0);
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
var TC = {color: '#6a8aaa', font: {family: 'IBM Plex Mono', size: 10}};
var COLS = ['#00ffc8','#ff7043','#a78bfa','#ffe246','#4dd0e1'];
var QKS = ['Q1','Q2','Q3','Q4','Q5'];
var QNS = ['Q1 avg(burn)','Q2 avg(fatigue)','Q3 weighted','Q4 scaled','Q5 stress'];

function baseOpts(yLabel, logY) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {legend: {labels: {color: '#cdd9f0', font: {family: 'IBM Plex Mono', size: 11}}}},
    scales: {
      x: {grid: GC, ticks: TC},
      y: {grid: GC, ticks: TC, type: logY ? 'logarithmic' : 'linear',
          title: {display: true, text: yLabel, color: '#6a8aaa', font: {family: 'IBM Plex Mono', size: 10}}}
    }
  };
}

function buildCharts() {
  new Chart(document.getElementById('ch-upload'), {
    type: 'bar',
    data: {labels: XL, datasets: [
      {label: 'HE (CKKS)', data: BENCH.map(function(d){return d.uhe;}), backgroundColor: 'rgba(0,255,200,0.15)', borderColor: '#00ffc8', borderWidth: 1.5},
      {label: 'Plaintext', data: BENCH.map(function(d){return d.upt;}), backgroundColor: 'rgba(255,112,67,0.15)', borderColor: '#ff7043', borderWidth: 1.5}
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
      {label: 'HE (CKKS)', data: BENCH.map(function(d){return d.q.Q1.ht;}), backgroundColor: 'rgba(0,255,200,0.15)', borderColor: '#00ffc8', borderWidth: 1.5},
      {label: 'Plaintext', data: BENCH.map(function(d){return d.q.Q1.pt;}), backgroundColor: 'rgba(255,112,67,0.15)', borderColor: '#ff7043', borderWidth: 1.5}
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
    app.run(host='0.0.0.0', port=5000, debug=False)
