"""Q-Strainer Dashboard — local HTTP server for viewing compute straining results.

Usage:
    py dashboard.py              # opens http://localhost:8050
    py dashboard.py --port 9000  # custom port

Serves a modern single-page dashboard that displays all demo run
results from the runs/ directory — showing how much compute/time/cost
was saved by the GPU workload strainer.
"""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

RUNS_DIR = Path(__file__).resolve().parent / "runs"
PORT = 8050


# ── HTML Template ────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Q-Strainer — GPU Compute Workload Dashboard</title>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2333;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --purple: #bc8cff;
    --orange: #f0883e;
    --cyan: #39d2c0;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.5;
  }

  /* ── Top Bar ─────────────────────────── */
  .topbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .topbar svg { flex-shrink: 0; }
  .topbar h1 {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.3px;
  }
  .topbar h1 span { color: var(--accent); }
  .topbar .version {
    font-size: 12px;
    color: var(--text-dim);
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2px 10px;
  }
  .topbar .spacer { flex: 1; }
  .topbar select {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    cursor: pointer;
    outline: none;
  }
  .topbar select:hover { border-color: var(--accent); }

  /* ── Layout ──────────────────────────── */
  .container { max-width: 1400px; margin: 0 auto; padding: 24px 32px; }

  /* ── KPI Row ─────────────────────────── */
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }
  .kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .kpi:hover { border-color: var(--accent); }
  .kpi .value {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.1;
  }
  .kpi .label {
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 6px;
  }
  .kpi .sub {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 2px;
  }
  .kpi.accent .value { color: var(--accent); }
  .kpi.green  .value { color: var(--green); }
  .kpi.red    .value { color: var(--red); }
  .kpi.purple .value { color: var(--purple); }
  .kpi.orange .value { color: var(--orange); }
  .kpi.cyan   .value { color: var(--cyan); }

  /* ── Sections ────────────────────────── */
  .section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 20px;
    overflow: hidden;
  }
  .section-header {
    padding: 16px 20px;
    font-size: 15px;
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-header .badge {
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-dim);
  }
  .section-body { padding: 20px; }

  /* ── Verdict donut ───────────────────── */
  .verdict-row {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    align-items: center;
  }
  .verdict-bar-container { flex: 1; min-width: 300px; }
  .verdict-bar-track {
    display: flex;
    height: 36px;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 12px;
  }
  .verdict-bar-seg { transition: width 0.5s; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; }
  .verdict-legend {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
  .verdict-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
  }
  .verdict-swatch {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    flex-shrink: 0;
  }

  /* ── Savings highlight ───────────────── */
  .savings-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }
  .saving-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
  }
  .saving-card .big {
    font-size: 36px;
    font-weight: 800;
    line-height: 1.1;
  }
  .saving-card .desc {
    font-size: 12px;
    color: var(--text-dim);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  /* ── Phase Cards ─────────────────────── */
  .phases-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
  }
  .phase-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    transition: transform 0.15s, border-color 0.2s;
  }
  .phase-card:hover { transform: translateY(-2px); border-color: var(--accent); }
  .phase-card .phase-num {
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .phase-card h3 {
    font-size: 15px;
    font-weight: 600;
    margin: 4px 0 12px;
  }
  .phase-card .metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .phase-card .metric {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
  }
  .phase-card .metric .k { color: var(--text-dim); }
  .phase-card .metric .v { font-weight: 600; font-variant-numeric: tabular-nums; }

  /* ── Decisions Table ─────────────────── */
  .dec-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .dec-table th {
    text-align: left;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
  }
  .dec-table td {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  .dec-table tr:last-child td { border-bottom: none; }
  .dec-table tr:hover td { background: rgba(88,166,255,0.04); }

  .verdict-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .verdict-SKIP        { background: rgba(248,81,73,0.15); color: var(--red); }
  .verdict-APPROXIMATE { background: rgba(210,153,34,0.15); color: var(--yellow); }
  .verdict-DEFER       { background: rgba(240,136,62,0.15); color: var(--orange); }
  .verdict-EXECUTE     { background: rgba(63,185,80,0.15);  color: var(--green); }

  /* ── Z-Score Bars ────────────────────── */
  .zscore-list { list-style: none; }
  .zscore-item {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    font-size: 13px;
  }
  .zscore-item .feat {
    width: 200px;
    text-align: right;
    color: var(--text-dim);
    font-family: 'Consolas','Courier New',monospace;
    flex-shrink: 0;
    font-size: 12px;
  }
  .zscore-bar-bg {
    flex: 1;
    height: 22px;
    background: var(--surface2);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }
  .zscore-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
    min-width: 2px;
  }
  .zscore-val {
    width: 50px;
    text-align: right;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
  }

  /* ── Quantum Section ──────────────────── */
  .quantum-hero {
    background: linear-gradient(135deg, rgba(188,140,255,0.08), rgba(88,166,255,0.08));
    border: 1px solid var(--purple);
    border-radius: 12px;
    margin-bottom: 20px;
    overflow: hidden;
  }
  .quantum-hero .section-header {
    border-bottom: 1px solid rgba(188,140,255,0.25);
  }
  .qubo-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
  }
  .qubo-metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
  }
  .qubo-metric .val {
    font-size: 24px;
    font-weight: 700;
    line-height: 1.2;
  }
  .qubo-metric .lbl {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-top: 4px;
  }
  .qubo-explain {
    margin-top: 16px;
    padding: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1.7;
  }
  .qubo-explain strong { color: var(--purple); }
  .qubo-explain ul { margin: 6px 0 0 18px; }

  /* ── Quantum Advantage Section ────────── */
  .qa-hero {
    background: linear-gradient(135deg, rgba(57,210,192,0.08), rgba(188,140,255,0.08));
    border: 1px solid var(--cyan);
    border-radius: 12px;
    margin-bottom: 20px;
    overflow: hidden;
  }
  .qa-hero .section-header {
    border-bottom: 1px solid rgba(57,210,192,0.25);
  }
  .qa-pipeline-flow {
    display: flex; align-items: center; gap: 0; flex-wrap: wrap;
    margin-bottom: 16px; padding: 10px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px;
  }
  .qa-pipe-step {
    font-size: 11px; font-weight: 600; color: var(--cyan);
    padding: 4px 10px; background: rgba(57,210,192,0.10);
    border-radius: 6px; white-space: nowrap;
  }
  .qa-pipe-arrow {
    color: var(--text-dim); font-size: 14px; margin: 0 4px;
  }
  .qa-makespan-compare {
    display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px;
    align-items: center; margin-bottom: 16px;
  }
  .qa-makespan-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; text-align: center;
  }
  .qa-makespan-box .val {
    font-size: 32px; font-weight: 800; line-height: 1.2;
  }
  .qa-makespan-box .lbl {
    font-size: 10px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.4px; margin-top: 4px;
  }
  .qa-arrow-col {
    text-align: center; font-size: 28px; color: var(--green);
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }
  .qa-arrow-col .reduction-badge {
    font-size: 14px; font-weight: 700; color: var(--green);
    background: rgba(63,185,80,0.12); padding: 2px 10px;
    border-radius: 10px;
  }

  .quantum-verdict-row {
    display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px;
  }
  .quantum-verdict-chip {
    font-size: 12px; font-weight: 600; padding: 4px 12px;
    border-radius: 6px; display: flex; gap: 6px; align-items: center;
  }

  /* ── Empty State ─────────────────────── */
  .empty {
    text-align: center;
    padding: 80px 20px;
    color: var(--text-dim);
  }
  .empty h2 {
    font-size: 22px;
    color: var(--text);
    margin-bottom: 8px;
  }
  .empty code {
    display: inline-block;
    margin-top: 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 20px;
    font-size: 14px;
    color: var(--green);
  }

  /* ── Responsive ──────────────────────── */
  @media (max-width: 768px) {
    .container { padding: 16px; }
    .topbar { padding: 12px 16px; }
    .phases-grid { grid-template-columns: 1fr; }
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
    .savings-row { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>

<div class="topbar">
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke="#58a6ff"/>
  </svg>
  <h1>Q-<span>Strainer</span></h1>
  <span class="version">v0.6.0</span>
  <div class="spacer"></div>
  <select id="runSelect" onchange="loadRun(this.value)">
    <option value="">Select a run...</option>
  </select>
</div>

<div class="container" id="app">
  <div class="empty" id="emptyState">
    <h2>No runs loaded</h2>
    <p>Run the demo first, then refresh this page.</p>
    <code>py tests/run_demo.py</code>
  </div>
</div>

<script>
const API = '/api';

async function fetchRuns() {
  const res = await fetch(`${API}/runs`);
  return res.json();
}

async function fetchRun(id) {
  const res = await fetch(`${API}/run?id=${encodeURIComponent(id)}`);
  return res.json();
}

function fmt(n, dec) {
  if (typeof n !== 'number' || isNaN(n)) return '—';
  if (n >= 1e15) return (n/1e15).toFixed(1) + ' PFLOP';
  if (n >= 1e12) return (n/1e12).toFixed(1) + ' TFLOP';
  if (n >= 1e9)  return (n/1e9).toFixed(1)  + ' GFLOP';
  if (n >= 1e6)  return (n/1e6).toFixed(1)  + 'M';
  if (n >= 1e3)  return (n/1e3).toFixed(1)  + 'K';
  if (typeof dec === 'number') return n.toFixed(dec);
  return n.toLocaleString();
}

function pct(n) {
  if (typeof n !== 'number') return '—';
  return (n * 100).toFixed(1) + '%';
}

const VERDICT_COLORS = {
  EXECUTE:     'var(--green)',
  APPROXIMATE: 'var(--yellow)',
  DEFER:       'var(--orange)',
  SKIP:        'var(--red)',
};

function verdictClass(v) {
  return 'verdict-badge verdict-' + (v || 'EXECUTE');
}

function zColor(z) {
  if (z >= 8) return 'var(--red)';
  if (z >= 5) return 'var(--orange)';
  if (z >= 3) return 'var(--yellow)';
  return 'var(--green)';
}

function renderDashboard(data) {
  const s = data.summary || {};
  const phases = data.phases || [];

  const p1 = phases.find(p => p.phase === 1) || {};
  const p3 = phases.find(p => p.phase === 3) || {};
  const p4 = phases.find(p => p.phase === 4) || {};
  const p6 = phases.find(p => p.phase === 6) || {};
  const p7 = phases.find(p => p.phase === 7) || {};

  const totalTasks = s.total_tasks || 0;
  const executed   = s.tasks_executed || 0;
  const strained   = s.tasks_strained || 0;
  const strainPct  = s.strain_ratio || (totalTasks ? strained / totalTasks : 0);
  const flopsSaved = s.total_flops_saved || 0;
  const timeSaved  = s.total_time_saved_s || 0;
  const costSaved  = s.total_cost_saved_usd || 0;
  const rawVd = s.verdict_distribution || {};
  // Merge quantum_* prefixed verdicts into base verdicts for the bar chart
  const vd = {};
  for (const [k, v] of Object.entries(rawVd)) {
    const base = k.startsWith('quantum_') ? k.slice(8) : k;
    vd[base] = (vd[base] || 0) + v;
  }

  const ts = new Date(data.timestamp);
  const timeStr = ts.toLocaleString();

  let html = '';

  // ── Run header
  html += `
  <div style="margin-bottom:8px;font-size:13px;color:var(--text-dim)">
    Run <b style="color:var(--text)">${data.run_id}</b> &mdash; ${timeStr}
  </div>`;

  // ── KPI Row
  html += `
  <div class="kpi-row">
    <div class="kpi accent">
      <div class="value">${fmt(totalTasks)}</div>
      <div class="label">Tasks Processed</div>
    </div>
    <div class="kpi green">
      <div class="value">${fmt(executed)}</div>
      <div class="label">Executed</div>
      <div class="sub">${pct(totalTasks ? executed / totalTasks : 0)}</div>
    </div>
    <div class="kpi red">
      <div class="value">${fmt(strained)}</div>
      <div class="label">Strained (Saved)</div>
      <div class="sub">${pct(strainPct)} strain ratio</div>
    </div>
    <div class="kpi cyan">
      <div class="value">${fmt(flopsSaved)}</div>
      <div class="label">FLOPs Saved</div>
    </div>
    <div class="kpi orange">
      <div class="value">${timeSaved.toFixed(1)}s</div>
      <div class="label">Time Saved</div>
    </div>
    <div class="kpi purple">
      <div class="value">$${costSaved.toFixed(4)}</div>
      <div class="label">Cost Saved</div>
    </div>
    <div class="kpi accent">
      <div class="value">${s.raw_features || 15} → ${s.derived_features || 60}</div>
      <div class="label">Feature Expansion</div>
    </div>
  </div>`;

  // ── Verdict Distribution Bar
  const verdictOrder = ['EXECUTE', 'APPROXIMATE', 'DEFER', 'SKIP'];
  const verdictTotal = Object.values(vd).reduce((a, b) => a + b, 0) || 1;

  html += `
  <div class="section">
    <div class="section-header">
      Verdict Distribution
      <span class="badge">${Object.values(vd).reduce((a, b) => a + b, 0)} decisions</span>
    </div>
    <div class="section-body">
      <div class="verdict-row">
        <div class="verdict-bar-container">
          <div class="verdict-bar-track">`;

  for (const v of verdictOrder) {
    const cnt = vd[v] || 0;
    const w = ((cnt / verdictTotal) * 100).toFixed(1);
    if (cnt > 0) {
      html += `<div class="verdict-bar-seg" style="width:${w}%;background:${VERDICT_COLORS[v]};color:#000">${cnt}</div>`;
    }
  }

  html += `</div>
          <div class="verdict-legend">`;

  for (const v of verdictOrder) {
    const cnt = vd[v] || 0;
    html += `
            <div class="verdict-legend-item">
              <div class="verdict-swatch" style="background:${VERDICT_COLORS[v]}"></div>
              <span>${v}</span>
              <span style="color:var(--text-dim)">${cnt} (${((cnt/verdictTotal)*100).toFixed(0)}%)</span>
            </div>`;
  }

  html += `</div></div></div></div></div>`;

  // ── Savings Highlight
  html += `
  <div class="section">
    <div class="section-header">
      Compute Savings
      <span class="badge">what the strainer saved</span>
    </div>
    <div class="section-body">
      <div class="savings-row">
        <div class="saving-card">
          <div class="big" style="color:var(--cyan)">${fmt(flopsSaved)}</div>
          <div class="desc">Total FLOPs Saved</div>
        </div>
        <div class="saving-card">
          <div class="big" style="color:var(--orange)">${timeSaved.toFixed(1)}s</div>
          <div class="desc">GPU Time Saved</div>
        </div>
        <div class="saving-card">
          <div class="big" style="color:var(--purple)">$${costSaved.toFixed(4)}</div>
          <div class="desc">Estimated Cost Saved</div>
        </div>
        <div class="saving-card">
          <div class="big" style="color:var(--green)">${pct(strainPct)}</div>
          <div class="desc">Strain Ratio</div>
        </div>
      </div>
    </div>
  </div>`;

  // ── Pipeline Phase Cards
  html += `
  <div class="section">
    <div class="section-header">
      Pipeline Phases <span class="badge">${phases.length} phases</span>
    </div>
    <div class="section-body">
      <div class="phases-grid">`;

  for (const p of phases) {
    html += `<div class="phase-card">
      <div class="phase-num">Phase ${p.phase}</div>
      <h3>${esc(p.name)}</h3>
      <div class="metrics">`;

    const skip = new Set(['name','phase','decision_details','dominant_signals']);
    for (const [k, v] of Object.entries(p)) {
      if (skip.has(k)) continue;
      const label = k.replace(/_/g, ' ');
      let val;
      if (typeof v === 'number') {
        if (k.includes('flops')) val = fmt(v);
        else if (k.includes('time_s') && k !== 'wall_time_s') val = v.toFixed(2) + 's';
        else val = fmt(v);
      } else if (Array.isArray(v)) {
        val = v.join(' × ');
      } else {
        val = String(v);
      }
      html += `<div class="metric"><span class="k">${esc(label)}</span><span class="v">${esc(val)}</span></div>`;
    }

    html += `</div></div>`;
  }

  html += `</div></div></div>`;

  // ── Straining Decisions Table (from Phase 3)
  const decisions = p3.decision_details || [];
  if (decisions.length > 0) {
    html += `
    <div class="section">
      <div class="section-header">
        Straining Decisions
        <span class="badge">${decisions.length} strained tasks</span>
      </div>
      <div class="section-body" style="padding:0;overflow-x:auto">
        <table class="dec-table">
          <thead><tr>
            <th style="width:110px">Verdict</th>
            <th style="width:110px">Redundancy</th>
            <th style="width:100px">Time Saved</th>
            <th>Reason</th>
          </tr></thead>
          <tbody>`;

    for (const d of decisions) {
      const reasons = (d.decisions || []).map(x => x.reason).join('; ') || '—';
      html += `<tr>
        <td><span class="${verdictClass(d.verdict)}">${esc(d.verdict)}</span></td>
        <td><span style="font-weight:600;font-variant-numeric:tabular-nums">${typeof d.redundancy_score === 'number' ? d.redundancy_score.toFixed(3) : '—'}</span></td>
        <td>${typeof d.time_saved_s === 'number' ? d.time_saved_s.toFixed(2) + 's' : '—'}</td>
        <td style="font-size:12px;color:var(--text-dim)">${esc(reasons)}</td>
      </tr>`;
    }

    html += `</tbody></table></div></div>`;
  }

  // ── Standalone Redundancy Decisions (Phase 6)
  const p6decs = p6.decision_details || [];
  if (p6decs.length > 0) {
    html += `
    <div class="section">
      <div class="section-header">
        Redundancy Strainer — Standalone Analysis
        <span class="badge">${p6decs.length} criteria flagged</span>
      </div>
      <div class="section-body" style="padding:0;overflow-x:auto">
        <table class="dec-table">
          <thead><tr>
            <th style="width:110px">Verdict</th>
            <th style="width:180px">Metric</th>
            <th>Reason</th>
          </tr></thead>
          <tbody>`;

    for (const d of p6decs) {
      html += `<tr>
        <td><span class="${verdictClass(d.verdict)}">${esc(d.verdict)}</span></td>
        <td><code style="color:var(--cyan);font-size:12px">${esc(d.metric || '')}</code></td>
        <td style="font-size:12px;color:var(--text-dim)">${esc(d.reason || '')}</td>
      </tr>`;
    }

    html += `</tbody></table></div></div>`;
  }

  // ── Quantum Scheduler Section
  const qs = s.quantum_scheduler || null;
  const p8 = phases.find(p => p.phase === 8) || {};
  if (qs) {
    html += `
    <div class="quantum-hero">
      <div class="section-header">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--purple)" stroke-width="2" stroke-linecap="round">
          <circle cx="12" cy="12" r="10"/><path d="M12 2v20M2 12h20"/>
          <circle cx="12" cy="12" r="4" fill="rgba(188,140,255,0.2)"/>
        </svg>
        Quantum Scheduler — QUBO Batch Optimisation
        <span class="badge" style="border-color:var(--purple);color:var(--purple)">${qs.solver || 'SA'}</span>
        <span class="badge">${qs.batch_size} tasks</span>
      </div>
      <div class="section-body">
        <div class="qubo-grid">
          <div class="qubo-metric"><div class="val" style="color:var(--purple)">${qs.qubo_dimensions}×${qs.qubo_dimensions}</div><div class="lbl">QUBO Matrix</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--accent)">${fmt(qs.pairwise_interactions)}</div><div class="lbl">Pairwise Interactions</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--cyan)">${typeof qs.qubo_energy === 'number' ? qs.qubo_energy.toFixed(2) : '—'}</div><div class="lbl">QUBO Energy</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--orange)">${typeof qs.solve_time_s === 'number' ? (qs.solve_time_s < 1 ? (qs.solve_time_s*1000).toFixed(0)+'ms' : qs.solve_time_s.toFixed(2)+'s') : '—'}</div><div class="lbl">Solve Time</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--green)">${fmt(qs.executed)}</div><div class="lbl">Executed</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--red)">${fmt(qs.strained)}</div><div class="lbl">Strained</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--cyan)">${fmt(qs.flops_saved)}</div><div class="lbl">FLOPs Saved</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--purple)">$${typeof qs.cost_saved_usd === 'number' ? qs.cost_saved_usd.toFixed(4) : '—'}</div><div class="lbl">Cost Saved</div></div>
        </div>`;

    // Quantum verdict chips
    const qvd = qs.verdict_distribution || {};
    const qvKeys = Object.keys(qvd);
    if (qvKeys.length) {
      html += `<div class="quantum-verdict-row">`;
      for (const v of qvKeys) {
        const bg = v === 'EXECUTE' ? 'rgba(63,185,80,0.15)' : v === 'SKIP' ? 'rgba(248,81,73,0.15)' : v === 'APPROXIMATE' ? 'rgba(210,153,34,0.15)' : 'rgba(240,136,62,0.15)';
        const fg = VERDICT_COLORS[v] || 'var(--text)';
        html += `<div class="quantum-verdict-chip" style="background:${bg};color:${fg}">${v}<span style="font-size:14px;font-weight:800">${qvd[v]}</span></div>`;
      }
      html += `</div>`;
    }

    html += `
        <div class="qubo-explain">
          <strong>Why quantum scheduling?</strong> The QUBO formulation captures <strong>task-task interactions</strong> that greedy per-task evaluation misses:
          <ul>
            <li><strong>Data similarity coupling</strong> — jointly redundant batches are strained together</li>
            <li><strong>Consecutive step anti-correlation</strong> — avoids long skip gaps that destabilise training</li>
            <li><strong>Cross-GPU fairness</strong> — balanced strain load across all GPUs</li>
            <li><strong>Strain rate constraint</strong> — soft cap ensures enough tasks still execute</li>
          </ul>
        </div>
      </div>
    </div>`;
  }

  // ── Quantum Advantage Pipeline (Phase 9)
  const qa = s.quantum_advantage || null;
  const p9 = phases.find(p => p.phase === 9) || {};
  if (qa) {
    const cg = qa.conflict_graph || {};
    const ising = qa.ising || {};
    const qaoa = qa.qaoa || {};
    const mk = qa.makespan || {};
    const par = qa.parallelism || {};
    const cv = qa.coloring_valid || {};
    const tm = qa.timing || {};
    const reductionPct = typeof mk.reduction === 'number' ? (mk.reduction * 100).toFixed(0) : '—';
    const dropPct = typeof cg.edge_drop_ratio === 'number' ? (cg.edge_drop_ratio * 100).toFixed(0) : '—';

    html += `
    <div class="qa-hero">
      <div class="section-header">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" stroke-width="2" stroke-linecap="round">
          <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5"/>
          <line x1="12" y1="2" x2="12" y2="22"/>
          <line x1="22" y1="8.5" x2="2" y2="15.5"/>
          <line x1="2" y1="8.5" x2="22" y2="15.5"/>
        </svg>
        Quantum Advantage — Conflict Graph Purification
        <span class="badge" style="border-color:var(--cyan);color:var(--cyan)">${qa.tasks} tasks</span>
        <span class="badge" style="border-color:var(--green);color:var(--green)">${reductionPct}% makespan reduction</span>
      </div>
      <div class="section-body">
        <div class="qa-pipeline-flow">
          <span class="qa-pipe-step">Conflict Graph</span><span class="qa-pipe-arrow">→</span>
          <span class="qa-pipe-step">QUBO</span><span class="qa-pipe-arrow">→</span>
          <span class="qa-pipe-step">Ising (h,J)</span><span class="qa-pipe-arrow">→</span>
          <span class="qa-pipe-step">QAOA Circuit</span><span class="qa-pipe-arrow">→</span>
          <span class="qa-pipe-step">Sampling</span><span class="qa-pipe-arrow">→</span>
          <span class="qa-pipe-step">Purification</span><span class="qa-pipe-arrow">→</span>
          <span class="qa-pipe-step">DSatur Coloring</span><span class="qa-pipe-arrow">→</span>
          <span class="qa-pipe-step" style="color:var(--green);background:rgba(63,185,80,0.12)">Schedule</span>
        </div>

        <div class="qa-makespan-compare">
          <div class="qa-makespan-box">
            <div class="val" style="color:var(--red)">${mk.original || '—'}</div>
            <div class="lbl">Original Makespan</div>
            <div style="font-size:10px;color:var(--text-dim);margin-top:4px">${cg.original_edges || '—'} conflict edges · density ${typeof cg.original_density === 'number' ? (cg.original_density * 100).toFixed(1) + '%' : '—'}</div>
          </div>
          <div class="qa-arrow-col">
            <span>→</span>
            <span class="reduction-badge">−${reductionPct}%</span>
          </div>
          <div class="qa-makespan-box" style="border-color:var(--green)">
            <div class="val" style="color:var(--green)">${mk.purified || '—'}</div>
            <div class="lbl">Purified Makespan</div>
            <div style="font-size:10px;color:var(--text-dim);margin-top:4px">${cg.purified_edges || '—'} edges · ${cg.edges_dropped || '—'} dropped (${dropPct}%)</div>
          </div>
        </div>

        <div class="qubo-grid">
          <div class="qubo-metric"><div class="val" style="color:var(--cyan)">${ising.qubo_size || '—'}×${ising.qubo_size || '—'}</div><div class="lbl">QUBO Size</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--purple)">${typeof ising.h_norm === 'number' ? ising.h_norm.toFixed(4) : '—'}</div><div class="lbl">||h|| Norm</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--accent)">${ising.j_nnz || '—'}</div><div class="lbl">J Non-Zero</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--purple)">${qaoa.p_layers || '—'}</div><div class="lbl">QAOA Layers</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--cyan)">${typeof qaoa.optimal_energy === 'number' ? qaoa.optimal_energy.toFixed(4) : '—'}</div><div class="lbl">QAOA Energy</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--orange)">${qaoa.n_samples || '—'}</div><div class="lbl">Samples</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--green)">${par.max || '—'}</div><div class="lbl">Max Parallelism</div></div>
          <div class="qubo-metric"><div class="val" style="color:var(--accent)">${typeof par.avg === 'number' ? par.avg.toFixed(1) : '—'}</div><div class="lbl">Avg Parallelism</div></div>
        </div>

        <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap">
          <div class="quantum-verdict-chip" style="background:rgba(63,185,80,0.12);color:var(--green)">
            Original coloring ${cv.original ? '✓' : '✗'}
          </div>
          <div class="quantum-verdict-chip" style="background:rgba(57,210,192,0.12);color:var(--cyan)">
            Purified coloring ${cv.purified ? '✓' : '✗'}
          </div>
          <div class="quantum-verdict-chip" style="background:rgba(188,140,255,0.12);color:var(--purple)">
            Total ${typeof tm.total_s === 'number' ? (tm.total_s < 1 ? (tm.total_s*1000).toFixed(0)+'ms' : tm.total_s.toFixed(2)+'s') : '—'}
          </div>
        </div>

        <div class="qubo-explain">
          <strong>How quantum advantage works:</strong> The QAOA circuit samples bitstrings from the Ising Hamiltonian, revealing which conflict edges are <strong>consistently resolved</strong> across quantum states.
          Edges with high resolution frequency are purged — the purified graph has <strong>fewer edges → fewer colours → smaller makespan</strong>.
          <ul>
            <li><strong>Conflict graph</strong> — GPU contention, data overlap, and memory pressure edges</li>
            <li><strong>QAOA sampling</strong> — ${qaoa.n_samples || '—'} bitstrings from ${qaoa.p_layers || '—'}-layer circuit</li>
            <li><strong>Purification</strong> — ${cg.edges_dropped || '—'} of ${cg.original_edges || '—'} edges dropped (${dropPct}% pruned)</li>
            <li><strong>DSatur coloring</strong> — makespan reduced from <strong>${mk.original || '—'}</strong> to <strong>${mk.purified || '—'}</strong> time slots</li>
          </ul>
        </div>
      </div>
    </div>`;
  }

  // ── Convergence Strainer Z-Score Signals (Phase 7)
  const signals = p7.dominant_signals || [];
  if (signals.length > 0) {
    const maxZ = Math.max(...signals.map(s => Math.abs(s.z_score)), 1);
    html += `
    <div class="section">
      <div class="section-header">
        Convergence Strainer — Z-Score Analysis
        <span class="badge">redundancy=${typeof p7.redundancy_score === 'number' ? p7.redundancy_score.toFixed(4) : '—'}</span>
      </div>
      <div class="section-body">
        <ul class="zscore-list">`;

    for (const sig of signals) {
      const absZ = Math.abs(sig.z_score);
      const pctBar = Math.min((absZ / maxZ) * 100, 100);
      html += `
          <li class="zscore-item">
            <span class="feat">${esc(sig.feature)}</span>
            <div class="zscore-bar-bg">
              <div class="zscore-bar" style="width:${pctBar}%;background:${zColor(absZ)}"></div>
            </div>
            <span class="zscore-val" style="color:${zColor(absZ)}">z=${sig.z_score}</span>
          </li>`;
    }

    html += `</ul></div></div>`;
  }

  document.getElementById('app').innerHTML = html;
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function loadRun(id) {
  if (!id) return;
  const data = await fetchRun(id);
  renderDashboard(data);
  document.getElementById('runSelect').value = id;
}

async function init() {
  const runs = await fetchRuns();
  const sel = document.getElementById('runSelect');
  sel.innerHTML = '<option value="">Select a run...</option>';
  for (const r of runs) {
    const opt = document.createElement('option');
    opt.value = r.id;
    opt.textContent = `${r.id}  —  ${new Date(r.timestamp).toLocaleString()}  —  ${r.total_tasks} tasks, ${r.strain_ratio}`;
    sel.appendChild(opt);
  }
  if (runs.length > 0) {
    await loadRun(runs[0].id);
  }
}

init();
</script>
</body>
</html>"""


# ── HTTP Handler ─────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the dashboard HTML and a JSON API for run data."""

    def log_message(self, format, *args):
        """Suppress noisy default access logs."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/runs":
            self._serve_runs_list()
        elif path == "/api/run":
            qs = parse_qs(parsed.query)
            run_id = qs.get("id", [None])[0]
            self._serve_run(run_id)
        else:
            self.send_error(404)

    def _serve_html(self):
        content = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_json(self, data):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_runs_list(self):
        """Return list of demo runs, newest first."""
        runs = []
        if RUNS_DIR.exists():
            for f in sorted(RUNS_DIR.glob("demo_*.json"), reverse=True):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    summary = data.get("summary", {})
                    runs.append({
                        "id": data.get("run_id", f.stem),
                        "timestamp": data.get("timestamp", ""),
                        "total_tasks": summary.get("total_tasks", 0),
                        "tasks_strained": summary.get("tasks_strained", 0),
                        "strain_ratio": f"{summary.get('strain_ratio', 0):.1%}",
                        "file": f.name,
                    })
                except Exception:
                    pass
        self._serve_json(runs)

    def _serve_run(self, run_id: str | None):
        """Return full JSON for one run."""
        if not run_id:
            self.send_error(400, "Missing ?id= parameter")
            return
        if RUNS_DIR.exists():
            for f in RUNS_DIR.glob("demo_*.json"):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    if data.get("run_id") == run_id:
                        self._serve_json(data)
                        return
                except Exception:
                    pass
        self.send_error(404, f"Run {run_id!r} not found")


# ── Main ─────────────────────────────────────────────────────

def main():
    port = PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    url = f"http://localhost:{port}"
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)

    print("=" * 52)
    print("  Q-Strainer Dashboard")
    print(f"  GPU Compute Workload Strainer")
    print(f"  Serving on {url}")
    print("  Press Ctrl+C to stop")
    print("=" * 52)

    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
