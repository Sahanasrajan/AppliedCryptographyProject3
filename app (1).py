"""
app.py — Flask web server for the HE Burnout interactive app.
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, jsonify, request, render_template_string
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from generate_dataset import generate
from he_engine import HEQuerySystem, PlaintextQuerySystem

app = Flask(__name__)

# ── Global state (loaded once at startup) ────────────────────────
NUMERIC_COLS = [
    "designation", "resource_allocation", "mental_fatigue_score",
    "hours_per_week", "years_experience", "team_size", "burn_rate"
]

print("[*] Loading dataset...")
CSV = os.path.join(os.path.dirname(__file__), "developer_burnout_dataset.csv")
import pandas as pd
if os.path.exists(CSV):
    df_full = pd.read_csv(CSV)
else:
    df_full = generate()
    df_full.to_csv(CSV, index=False)
print(f"[✓] Dataset loaded: {len(df_full)} rows")

# Cache HE systems per size to avoid re-encrypting every request
_he_cache = {}
_pt_cache = {}

def get_systems(n):
    if n not in _he_cache:
        df = df_full.head(n).copy()
        he = HEQuerySystem(poly_modulus_degree=8192)
        he.alice_upload_dataset(df, NUMERIC_COLS)
        pt = PlaintextQuerySystem()
        pt.upload_dataset(df, NUMERIC_COLS)
        _he_cache[n] = he
        _pt_cache[n] = pt
        print(f"[✓] HE system ready for N={n}")
    return _he_cache[n], _pt_cache[n]

# Pre-warm with N=500 at startup
print("[*] Pre-warming HE system (N=500)...")
get_systems(500)
print("[✓] Ready!\n")

# ── HTML Template ─────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HE Burnout — Live Query App</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Clash+Display:wght@500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:     #07090f;
  --bg2:    #0c0f1a;
  --bg3:    #111520;
  --card:   #131826;
  --border: #1c2436;
  --he:     #00ffc8;
  --pt:     #ff7043;
  --err:    #ffe246;
  --muted:  #3a4a60;
  --text:   #cdd9f0;
  --text2:  #6a8aaa;
  --glow:   0 0 24px rgba(0,255,200,0.18);
  --font:   'IBM Plex Mono', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  min-height: 100vh;
}

/* noise texture */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 0; opacity: 0.6;
}

.wrap { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 32px 24px; }

/* ── TOP BAR ── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 40px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}
.logo {
  font-family: 'Clash Display', 'IBM Plex Mono', monospace;
  font-size: 1.5rem; font-weight: 700;
  color: var(--he);
  text-shadow: 0 0 28px rgba(0,255,200,0.35);
  letter-spacing: -0.02em;
}
.logo span { color: var(--text2); font-weight: 500; font-size: 0.8rem; display:block; margin-top:2px; }
.badge {
  font-size: 0.65rem; padding: 6px 14px;
  border: 1px solid var(--he); border-radius: 20px;
  color: var(--he); background: rgba(0,255,200,0.05);
  box-shadow: var(--glow);
}

/* ── STEP INDICATOR ── */
.steps {
  display: flex; align-items: center; gap: 0;
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 20px;
  margin-bottom: 28px; overflow-x: auto;
  font-size: 0.68rem;
}
.step {
  display: flex; flex-direction: column; align-items: center;
  gap: 5px; flex: 1; min-width: 100px; text-align: center;
  color: var(--muted); padding: 8px;
  border-radius: 8px; transition: all 0.3s;
}
.step.active { color: var(--he); background: rgba(0,255,200,0.05); }
.step .snum {
  width: 26px; height: 26px; border-radius: 50%;
  border: 1px solid currentColor;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; font-weight: 600;
}
.step.active .snum { background: var(--he); color: #000; border-color: var(--he); box-shadow: var(--glow); }
.sarrow { color: var(--muted); padding: 0 6px; font-size: 0.9rem; }

/* ── CONTROLS PANEL ── */
.controls {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 16px; margin-bottom: 24px;
}
.ctrl-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px;
}
.ctrl-label {
  font-size: 0.62rem; color: var(--text2); text-transform: uppercase;
  letter-spacing: 0.12em; margin-bottom: 12px;
}
.btn-group { display: flex; flex-wrap: wrap; gap: 8px; }
.btn {
  padding: 8px 16px;
  border: 1px solid var(--border); border-radius: 6px;
  background: var(--bg3); color: var(--text2);
  font-family: var(--font); font-size: 0.7rem;
  cursor: pointer; transition: all 0.18s;
}
.btn:hover { border-color: var(--he); color: var(--he); }
.btn.active {
  border-color: var(--he); background: rgba(0,255,200,0.07);
  color: var(--he); box-shadow: var(--glow);
}

/* ── RUN BUTTON ── */
.run-wrap { text-align: center; margin-bottom: 28px; }
.run-btn {
  padding: 14px 48px;
  background: var(--he); color: #000;
  border: none; border-radius: 8px;
  font-family: var(--font); font-size: 0.85rem; font-weight: 600;
  cursor: pointer; letter-spacing: 0.05em;
  box-shadow: 0 0 30px rgba(0,255,200,0.3);
  transition: all 0.2s;
  position: relative; overflow: hidden;
}
.run-btn:hover { transform: translateY(-2px); box-shadow: 0 0 40px rgba(0,255,200,0.45); }
.run-btn:active { transform: translateY(0); }
.run-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.run-btn .spinner {
  display: none; width: 14px; height: 14px;
  border: 2px solid #000; border-top-color: transparent;
  border-radius: 50%; animation: spin 0.7s linear infinite;
  vertical-align: middle; margin-right: 8px;
}
.run-btn.loading .spinner { display: inline-block; }
.run-btn.loading .btn-text { opacity: 0.7; }

/* ── RESULT PANEL ── */
.result-panel {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 24px;
  margin-bottom: 24px; display: none;
}
.result-panel.visible { display: block; animation: slideUp 0.4s ease; }

.result-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 20px; padding-bottom: 14px;
  border-bottom: 1px solid var(--border);
}
.result-title { font-size: 0.75rem; color: var(--text2); }
.result-title strong { color: var(--text); display: block; font-size: 0.9rem; margin-top: 3px; }
.encrypt-tag {
  font-size: 0.62rem; padding: 4px 12px;
  background: rgba(0,255,200,0.08); border: 1px solid rgba(0,255,200,0.25);
  border-radius: 4px; color: var(--he);
}

.result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px; }

.res-box {
  border-radius: 10px; padding: 18px;
  border: 1px solid var(--border);
}
.res-box.he-box { border-color: rgba(0,255,200,0.2); background: rgba(0,255,200,0.03); }
.res-box.pt-box { border-color: rgba(255,112,67,0.2); background: rgba(255,112,67,0.03); }

.res-box .rb-label {
  font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em;
  margin-bottom: 10px;
}
.he-box .rb-label { color: var(--he); }
.pt-box .rb-label { color: var(--pt); }

.res-box .rb-val {
  font-size: 1.6rem; font-weight: 600; line-height: 1;
  margin-bottom: 10px;
}
.he-box .rb-val { color: var(--he); text-shadow: 0 0 16px rgba(0,255,200,0.3); }
.pt-box .rb-val { color: var(--pt); }

.res-box .rb-time {
  font-size: 0.65rem; color: var(--muted);
}
.res-box .rb-time .t { 
  display: inline-block; margin-top: 4px;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 4px; padding: 2px 8px; color: var(--text2);
}

.err-strip {
  background: rgba(255,226,70,0.05); border: 1px solid rgba(255,226,70,0.2);
  border-radius: 8px; padding: 12px 16px;
  display: flex; align-items: center; gap: 12px;
  font-size: 0.7rem;
}
.err-label { color: var(--err); font-weight: 600; font-size: 0.62rem; }
.err-num { color: var(--err); font-size: 1rem; font-weight: 600; }
.err-desc { color: var(--text2); }

.slowdown-strip {
  margin-top: 12px;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px;
  font-size: 0.7rem; color: var(--text2);
  display: flex; align-items: center; gap: 10px;
}
.slowdown-num { color: var(--pt); font-size: 1.1rem; font-weight: 600; }

/* ── LOG ── */
.log-panel {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px;
  margin-bottom: 24px;
}
.log-title { font-size: 0.65rem; color: var(--text2); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; }
.log-lines { font-size: 0.7rem; line-height: 2; }
.log-line { display: flex; gap: 12px; }
.log-line .lt { color: var(--muted); min-width: 60px; }
.log-line .lm { color: var(--text); }
.log-line.alice .lm { color: var(--he); }
.log-line.carol .lm { color: #a78bfa; }
.log-line.result .lm { color: var(--err); font-weight: 600; }

/* ── HISTORY ── */
.history {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; overflow: hidden;
}
.history-header {
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  font-size: 0.65rem; color: var(--text2);
  text-transform: uppercase; letter-spacing: 0.1em;
  display: flex; justify-content: space-between; align-items: center;
}
.clear-btn {
  background: none; border: 1px solid var(--border);
  color: var(--muted); font-family: var(--font); font-size: 0.62rem;
  padding: 3px 10px; border-radius: 4px; cursor: pointer;
}
.clear-btn:hover { border-color: var(--pt); color: var(--pt); }
.history-empty { padding: 24px; text-align: center; color: var(--muted); font-size: 0.7rem; }
table.htable { width: 100%; border-collapse: collapse; font-size: 0.7rem; }
table.htable th {
  background: var(--bg3); color: var(--text2); font-weight: 600;
  padding: 10px 16px; text-align: left; font-size: 0.62rem;
  text-transform: uppercase; letter-spacing: 0.05em;
}
table.htable td { padding: 10px 16px; border-bottom: 1px solid rgba(28,36,54,0.8); }
table.htable tr:last-child td { border-bottom: none; }
table.htable tr:hover td { background: rgba(0,255,200,0.02); }
.tag-he { color: var(--he); } .tag-pt { color: var(--pt); } .tag-err { color: var(--err); }
.tag-slow { display:inline-block; background: rgba(255,112,67,0.1); border:1px solid rgba(255,112,67,0.25); color:var(--pt); border-radius:4px; padding:1px 7px; font-size:0.62rem; }

@keyframes slideUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 700px) {
  .controls, .result-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="wrap">

<!-- TOP BAR -->
<div class="topbar">
  <div class="logo">
    🔐 HE Burnout
    <span>Homomorphic Encryption · Developer Burnout Dataset · CS6903/4783</span>
  </div>
  <div class="badge">CKKS · 128-bit security · TenSEAL</div>
</div>

<!-- PIPELINE STEPS -->
<div class="steps" id="steps">
  <div class="step" id="s1"><div class="snum">1</div><div>Alice<br>Keygen</div></div>
  <div class="sarrow">→</div>
  <div class="step" id="s2"><div class="snum">2</div><div>Alice<br>Encrypt</div></div>
  <div class="sarrow">→</div>
  <div class="step" id="s3"><div class="snum">3</div><div>Carol<br>Receives CT</div></div>
  <div class="sarrow">→</div>
  <div class="step" id="s4"><div class="snum">4</div><div>Carol<br>Evaluates</div></div>
  <div class="sarrow">→</div>
  <div class="step" id="s5"><div class="snum">5</div><div>Alice<br>Decrypts</div></div>
  <div class="sarrow">→</div>
  <div class="step" id="s6"><div class="snum">✓</div><div>Result<br>Revealed</div></div>
</div>

<!-- CONTROLS -->
<div class="controls">
  <div class="ctrl-card">
    <div class="ctrl-label">① Dataset Size (N rows)</div>
    <div class="btn-group" id="size-btns">
      <button class="btn" data-val="100">N = 100</button>
      <button class="btn active" data-val="500">N = 500</button>
      <button class="btn" data-val="1000">N = 1,000</button>
      <button class="btn" data-val="3000">N = 3,000</button>
      <button class="btn" data-val="7000">N = 7,000</button>
    </div>
  </div>
  <div class="ctrl-card">
    <div class="ctrl-label">② Query Function (Carol evaluates on ciphertext)</div>
    <div class="btn-group" id="query-btns">
      <button class="btn active" data-val="avg_burn_rate">avg(burn_rate)</button>
      <button class="btn" data-val="avg_fatigue">avg(fatigue)</button>
      <button class="btn" data-val="avg_resource">avg(resource)</button>
      <button class="btn" data-val="avg_hours">avg(hours)</button>
      <button class="btn" data-val="stress_index">stress_index</button>
      <button class="btn" data-val="scaled_hours">scaled_hours</button>
      <button class="btn" data-val="weighted_risk">weighted_risk</button>
    </div>
  </div>
</div>

<!-- RUN -->
<div class="run-wrap">
  <button class="run-btn" id="run-btn" onclick="runQuery()">
    <span class="spinner"></span>
    <span class="btn-text">▶  Run Encrypted Query</span>
  </button>
</div>

<!-- LOG -->
<div class="log-panel">
  <div class="log-title">Execution Log</div>
  <div class="log-lines" id="log">
    <div class="log-line"><span class="lt">ready</span><span class="lm">Select a query and click Run ↑</span></div>
  </div>
</div>

<!-- RESULT -->
<div class="result-panel" id="result-panel">
  <div class="result-header">
    <div class="result-title">
      Query Result
      <strong id="r-query-name">—</strong>
    </div>
    <div class="encrypt-tag">🔒 Computed on ciphertext by Carol</div>
  </div>
  <div class="result-grid">
    <div class="res-box he-box">
      <div class="rb-label">🔒 HE Result (CKKS)</div>
      <div class="rb-val" id="r-he-val">—</div>
      <div class="rb-time">Eval time: <span class="t" id="r-he-time">—</span></div>
    </div>
    <div class="res-box pt-box">
      <div class="rb-label">📄 Plaintext Baseline</div>
      <div class="rb-val" id="r-pt-val">—</div>
      <div class="rb-time">Eval time: <span class="t" id="r-pt-time">—</span></div>
    </div>
  </div>
  <div class="err-strip">
    <div>
      <div class="err-label">CKKS Approximation Error</div>
      <div class="err-num" id="r-err">—</div>
    </div>
    <div class="err-desc">Inherent to approximate HE — negligible for analytics</div>
  </div>
  <div class="slowdown-strip">
    <div>HE is <span class="slowdown-num" id="r-slow">—</span>× slower than plaintext</div>
    <div style="color:var(--muted)"> — this is the privacy/performance tradeoff of homomorphic encryption</div>
  </div>
</div>

<!-- HISTORY TABLE -->
<div class="history">
  <div class="history-header">
    Query History
    <button class="clear-btn" onclick="clearHistory()">Clear</button>
  </div>
  <div id="history-body">
    <div class="history-empty">No queries run yet.</div>
  </div>
</div>

</div><!-- /.wrap -->

<script>
let selectedN = 500;
let selectedQ = 'avg_burn_rate';
let history = [];
let stepTimer = null;

const QUERY_LABELS = {
  avg_burn_rate: 'avg(burn_rate)',
  avg_fatigue:   'avg(mental_fatigue)',
  avg_resource:  'avg(resource_alloc)',
  avg_hours:     'avg(hours/week)',
  stress_index:  'stress_index = fatigue + resource',
  scaled_hours:  'scaled_hours = hours ÷ 40',
  weighted_risk: 'weighted burn-risk score',
};

// btn clicks
document.getElementById('size-btns').addEventListener('click', e => {
  if (!e.target.dataset.val) return;
  document.querySelectorAll('#size-btns .btn').forEach(b => b.classList.remove('active'));
  e.target.classList.add('active');
  selectedN = +e.target.dataset.val;
});

document.getElementById('query-btns').addEventListener('click', e => {
  if (!e.target.dataset.val) return;
  document.querySelectorAll('#query-btns .btn').forEach(b => b.classList.remove('active'));
  e.target.classList.add('active');
  selectedQ = e.target.dataset.val;
});

function setStep(n) {
  document.querySelectorAll('.step').forEach((s,i) => {
    s.classList.toggle('active', i < n);
  });
}

function log(lines) {
  const el = document.getElementById('log');
  el.innerHTML = lines.map(l =>
    `<div class="log-line ${l.cls||''}"><span class="lt">${l.t}</span><span class="lm">${l.m}</span></div>`
  ).join('');
}

async function runQuery() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true; btn.classList.add('loading');
  document.getElementById('result-panel').classList.remove('visible');
  setStep(0);

  const steps_log = [
    { t: 'step 1', m: '[Alice] Retrieving CKKS context & secret key...', cls:'alice' },
    { t: 'step 2', m: `[Alice] Encrypting ${selectedN.toLocaleString()} rows into packed CKKS ciphertexts...`, cls:'alice' },
    { t: 'step 3', m: '[Carol] Received ciphertext columns (no secret key)...', cls:'carol' },
    { t: 'step 4', m: `[Carol] Evaluating  "${QUERY_LABELS[selectedQ]}"  on ciphertext...`, cls:'carol' },
    { t: 'step 5', m: '[Alice] Decrypting result ciphertext with secret key...', cls:'alice' },
  ];

  for (let i = 0; i < steps_log.length; i++) {
    log(steps_log.slice(0, i+1));
    setStep(i+1);
    await new Promise(r => setTimeout(r, 220));
  }

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n: selectedN, query: selectedQ })
    });
    const data = await res.json();

    if (data.error) {
      log([...steps_log, { t: 'ERROR', m: data.error, cls:'result' }]);
      setStep(0);
    } else {
      setStep(6);
      log([...steps_log,
        { t: 'done', m: `[Alice] ✓ Result decrypted: ${data.he_result}`, cls:'result' }
      ]);

      // populate result panel
      const isVec = Array.isArray(data.he_result);
      document.getElementById('r-query-name').textContent = QUERY_LABELS[selectedQ] + `  (N = ${selectedN.toLocaleString()})`;
      document.getElementById('r-he-val').textContent = isVec
        ? '[' + data.he_result.map(v=>v.toFixed(3)).join(', ') + ', …]'
        : data.he_result.toFixed ? data.he_result.toFixed(6) : data.he_result;
      document.getElementById('r-pt-val').textContent = isVec
        ? '[' + data.pt_result.map(v=>v.toFixed(3)).join(', ') + ', …]'
        : data.pt_result.toFixed ? data.pt_result.toFixed(6) : data.pt_result;
      document.getElementById('r-he-time').textContent = data.he_time.toFixed(4) + 's';
      document.getElementById('r-pt-time').textContent = (data.pt_time * 1000).toFixed(3) + 'ms';
      document.getElementById('r-err').textContent = data.error_val !== null
        ? data.error_val.toExponential(2) : '< 10⁻⁹';
      document.getElementById('r-slow').textContent =
        (data.he_time / Math.max(data.pt_time, 1e-9)).toFixed(0);

      document.getElementById('result-panel').classList.add('visible');

      // add to history
      history.unshift({
        n: selectedN, query: QUERY_LABELS[selectedQ],
        he: isVec ? '(vector)' : data.he_result.toFixed(6),
        pt: isVec ? '(vector)' : data.pt_result.toFixed(6),
        he_t: data.he_time.toFixed(4),
        pt_t: (data.pt_time * 1000).toFixed(3),
        err: data.error_val !== null ? data.error_val.toExponential(1) : '~0',
        slow: (data.he_time / Math.max(data.pt_time, 1e-9)).toFixed(0),
      });
      renderHistory();
    }
  } catch(e) {
    log([...steps_log, { t: 'ERROR', m: String(e), cls:'result' }]);
  }

  btn.disabled = false; btn.classList.remove('loading');
}

function renderHistory() {
  const el = document.getElementById('history-body');
  if (!history.length) { el.innerHTML = '<div class="history-empty">No queries run yet.</div>'; return; }
  el.innerHTML = `<table class="htable">
    <thead><tr>
      <th>N</th><th>Query</th>
      <th class="tag-he">HE Result</th><th class="tag-pt">PT Result</th>
      <th class="tag-he">HE Time</th><th class="tag-pt">PT Time</th>
      <th class="tag-err">Error</th><th>Slowdown</th>
    </tr></thead><tbody>` +
    history.map(h => `<tr>
      <td>${h.n.toLocaleString()}</td>
      <td>${h.query}</td>
      <td class="tag-he">${h.he}</td>
      <td class="tag-pt">${h.pt}</td>
      <td class="tag-he">${h.he_t}s</td>
      <td class="tag-pt">${h.pt_t}ms</td>
      <td class="tag-err">${h.err}</td>
      <td><span class="tag-slow">${h.slow}×</span></td>
    </tr>`).join('') +
    '</tbody></table>';
}

function clearHistory() { history = []; renderHistory(); }
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

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
            he_val, he_t = he.query_average('burn_rate')
            pt_val, pt_t = pt.query_average('burn_rate')
            err = abs(he_val - pt_val)
        elif query == 'avg_fatigue':
            he_val, he_t = he.query_average('mental_fatigue_score')
            pt_val, pt_t = pt.query_average('mental_fatigue_score')
            err = abs(he_val - pt_val)
        elif query == 'avg_resource':
            he_val, he_t = he.query_average('resource_allocation')
            pt_val, pt_t = pt.query_average('resource_allocation')
            err = abs(he_val - pt_val)
        elif query == 'avg_hours':
            he_val, he_t = he.query_average('hours_per_week')
            pt_val, pt_t = pt.query_average('hours_per_week')
            err = abs(he_val - pt_val)
        elif query == 'stress_index':
            he_val, he_t = he.query_column_sum_two('mental_fatigue_score', 'resource_allocation')
            pt_val, pt_t = pt.query_column_sum_two('mental_fatigue_score', 'resource_allocation')
            he_val = he_val[:5]; pt_val = pt_val[:5]
            err = max(abs(h-p) for h,p in zip(he_val, pt_val))
        elif query == 'scaled_hours':
            he_val, he_t = he.query_scaled_column('hours_per_week', 1/40.0)
            pt_val, pt_t = pt.query_scaled_column('hours_per_week', 1/40.0)
            he_val = he_val[:5]; pt_val = pt_val[:5]
            err = max(abs(h-p) for h,p in zip(he_val, pt_val))
        elif query == 'weighted_risk':
            he_val, he_t = he.query_weighted_sum('burn_rate', weights)
            pt_val, pt_t = pt.query_weighted_sum('burn_rate', weights)
            err = abs(he_val - pt_val)
        else:
            return jsonify({'error': f'Unknown query: {query}'})

        return jsonify({
            'he_result': he_val,
            'pt_result': pt_val,
            'he_time': he_t,
            'pt_time': pt_t,
            'error_val': float(err),
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
