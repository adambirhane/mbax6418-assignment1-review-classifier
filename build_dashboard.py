"""Generate the results dashboard (single self-contained HTML file).

Usage:
    python3 build_dashboard.py --run step2_first100
    python3 build_dashboard.py --run step6_balanced3 -o dashboard.html

The generator recomputes every aggregate from the saved run's JSONL rows —
the saved output is the single source of truth, so numbers on the page
cannot drift from the recorded run. It also prints the same aggregates to
stdout so they can be cross-checked by hand.

Design: dark navy theme, fully recolorable through the CSS variables at the
top of the embedded stylesheet. Vanilla JS only — no libraries, no CDN,
works offline. Interactive filtering (assignment step 4) is built into the
review explorer.
"""

import argparse
import collections
import gzip
import json
import os
import sys

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gift Card Review Classifier — @@RUN@@</title>
<style>
/* ============================================================
   THEME — recolor the whole dashboard from this block.
   All surfaces are tints of navy; change --accent to re-skin.
   ============================================================ */
:root {
  --bg:            #0a0f1e;   /* page background        */
  --surface:       #0f1729;   /* card background        */
  --surface-2:     #16213c;   /* raised card / inputs   */
  --border:        #263353;   /* hairlines              */
  --text:          #e7ecf7;   /* body text              */
  --muted:         #8b9bbd;   /* secondary text         */
  --faint:         #5d6b8c;   /* tertiary text          */
  --accent:        #3d6df2;   /* navy blue accent       */
  --accent-bright: #6c95ff;   /* accent hover / glow    */
  --pos:   #3ddc97;           /* positive               */
  --neu:   #f2b84b;           /* neutral                */
  --neg:   #ff6b6b;           /* negative               */
  --good-bg:  rgba(61,220,151,.12);
  --bad-bg:   rgba(255,107,107,.12);
  --warn-bg:  rgba(242,184,75,.12);
  --radius: 14px;
  --font: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background:
    radial-gradient(1200px 500px at 20% -10%, rgba(61,109,242,.14), transparent 60%),
    radial-gradient(900px 400px at 90% 0%, rgba(28,52,120,.22), transparent 55%),
    var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 24px 64px; }

/* header */
.masthead { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; margin-bottom: 6px; }
.wordmark {
  font-size: 21px; font-weight: 700; letter-spacing: -.01em;
  background: linear-gradient(90deg, #a8c0ff, #6c95ff 45%, #3d6df2);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.wordmark .pre { font-weight: 600; opacity: .55; margin-right: 6px;
  background: none; -webkit-text-fill-color: currentColor; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 18px; }
.meta { display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0 4px; }
.chip {
  font-size: 11.5px; font-weight: 600; letter-spacing: .04em; color: var(--muted);
  border: 1px solid var(--border); background: var(--surface); border-radius: 999px;
  padding: 4px 11px; white-space: nowrap;
}
.chip b { color: var(--text); font-weight: 650; }

/* section headers */
.kicker {
  font-size: 11px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent-bright); margin: 34px 0 10px;
}
h2 { font-size: 17px; font-weight: 650; margin: 0 0 4px; letter-spacing: -.01em; }
.section-note { color: var(--muted); font-size: 12.5px; margin: 2px 0 14px; max-width: 72ch; }

/* KPI cards */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.kpi {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px 18px; position: relative; overflow: hidden;
}
.kpi::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
  background: var(--accent); }
.kpi.acc::before { background: var(--accent); }
.kpi.agree::before { background: var(--pos); }
.kpi.n::before { background: var(--nai, #8899cc); }
.kpi.baseline::before { background: var(--neu); }
.kpi .label { font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 6px; }
.kpi .value { font-size: 30px; font-weight: 700; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.kpi .note { font-size: 12px; color: var(--muted); margin-top: 4px; }
.kpi .value small { font-size: 15px; font-weight: 600; color: var(--muted); }

/* cards + grids */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px 22px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
@media (max-width: 900px) { .grid2, .grid3 { grid-template-columns: 1fr; } }

/* horizontal bar charts (all hand-rolled, no library) */
.hbar { display: flex; flex-direction: column; gap: 9px; }
.hbar .row { display: grid; grid-template-columns: 92px 1fr 88px; align-items: center; gap: 10px; }
.hbar .lbl { font-size: 12.5px; color: var(--muted); font-weight: 600; text-align: right;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hbar .track { background: rgba(255,255,255,.045); border-radius: 6px; height: 18px; position: relative; }
.hbar .fill { height: 100%; border-radius: 6px; min-width: 2px; /* never zero-width */ transition: width .35s ease; }
.hbar .val { font-size: 12.5px; color: var(--text); font-variant-numeric: tabular-nums; }
.hbar .val .pct { color: var(--muted); }
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 10px 0 14px; font-size: 12px; color: var(--muted); }
.legend .sw { display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 5px; }
.dual .caps { font-size: 10.5px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: var(--faint); margin: 2px 0 4px; }

/* confusion matrix */
.cm { display: grid; gap: 5px; max-width: 520px; }
.cm .cell { border-radius: 8px; padding: 10px 8px; text-align: center; min-height: 54px;
  display: flex; flex-direction: column; justify-content: center; gap: 1px; }
.cm .cell .n { font-size: 17px; font-weight: 700; font-variant-numeric: tabular-nums; }
.cm .cell .p { font-size: 10.5px; color: rgba(255,255,255,.55); font-variant-numeric: tabular-nums; }
.cm .hdr { color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; text-align: center; align-self: end; padding-bottom: 2px; }
.cm .corner { color: var(--faint); font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
  align-self: end; display: flex; align-items: center; justify-content: flex-end; }
.cmnote { font-size: 12px; color: var(--muted); margin-top: 10px; }

/* emotion rows */
.emo-row { display: grid; grid-template-columns: 86px 1fr 1fr 60px; gap: 10px; align-items: center; margin-bottom: 7px; font-size: 12.5px; }
.emo-row .lbl { color: var(--muted); font-weight: 600; text-transform: capitalize; text-align: right; }
.emo-row .track { background: rgba(255,255,255,.045); border-radius: 5px; height: 14px; }
.emo-row .fill { height: 100%; border-radius: 5px; min-width: 2px; }
.emo-row .n { color: var(--text); font-variant-numeric: tabular-nums; text-align: right; }

/* review table */
.toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }
.fgroup { display: flex; gap: 6px; flex-wrap: wrap; }
.fbtn {
  font: inherit; font-size: 12.5px; font-weight: 600; color: var(--muted); cursor: pointer;
  background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
  padding: 6px 13px; transition: all .15s ease;
}
.fbtn .c { color: var(--faint); font-variant-numeric: tabular-nums; }
.fbtn:hover { border-color: var(--accent); color: var(--text); }
.fbtn.on { background: var(--accent); border-color: var(--accent); color: #fff; }
.fbtn.on .c { color: rgba(255,255,255,.75); }
.fbtn.on.pos { background: var(--pos); border-color: var(--pos); color: #06281c; }
.fbtn.on.neu { background: var(--neu); border-color: var(--neu); color: #2c1d00; }
.fbtn.on.neg { background: var(--neg); border-color: var(--neg); color: #2d0505; }
.search {
  flex: 1; min-width: 200px; font: inherit; font-size: 13px; color: var(--text);
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px 13px; outline: none;
}
.search::placeholder { color: var(--faint); }
.search:focus { border-color: var(--accent); }
.countline { font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }
.countline b { color: var(--text); font-variant-numeric: tabular-nums; }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead th {
  text-align: left; font-size: 10.5px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--faint); padding: 8px 10px;
  border-bottom: 1px solid var(--border); white-space: nowrap;
}
tbody td { padding: 11px 10px; border-bottom: 1px solid rgba(38,51,83,.55); vertical-align: top; }
tbody tr { cursor: pointer; transition: background .12s ease; }
tbody tr:hover { background: rgba(61,109,242,.06); }
tbody tr.dim { opacity: .45; }
.stars { color: var(--accent-bright); letter-spacing: 1px; font-size: 12px; white-space: nowrap; }
.badge {
  display: inline-block; font-size: 10.5px; font-weight: 700; letter-spacing: .06em;
  border-radius: 6px; padding: 3px 8px; text-transform: uppercase; white-space: nowrap;
}
.badge.pos { background: var(--good-bg); color: var(--pos); }
.badge.neu { background: var(--warn-bg); color: var(--neu); }
.badge.neg { background: var(--bad-bg); color: var(--neg); }
.badge.llm { background: rgba(108,149,255,.14); color: var(--accent-bright); }
.badge.wl { background: rgba(139,155,189,.16); color: #b9c6dd; }
.badge.tiny { font-size: 9.5px; padding: 2px 6px; }
.badge.warn { outline: 1px solid var(--neg); color: var(--neg); background: rgba(255,107,107,.10); }
.title-cell { font-weight: 600; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.text-cell { color: var(--muted); max-width: 340px; font-size: 12.5px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.expand { background: rgba(255,255,255,.03); border-top: 1px solid var(--border); }
.expand td { padding: 14px 16px; }
.expand .full { color: var(--text); font-size: 13px; max-width: 900px; white-space: pre-wrap; }
.expand .raw { color: var(--faint); font-size: 11.5px; font-family: ui-monospace, "SF Mono", Menlo, monospace;
  margin-top: 8px; max-width: 900px; overflow-wrap: anywhere; }
.empty { color: var(--faint); text-align: center; padding: 26px 0; }

footer { margin-top: 40px; color: var(--faint); font-size: 12px; line-height: 1.7; }
footer b { color: var(--muted); }
@media (max-width: 720px) {
  .hbar .row { grid-template-columns: 76px 1fr 74px; }
  thead th:nth-child(4), tbody td:nth-child(4) { display: none; }
}
</style>
</head>
<body>
<div class="wrap" id="app"></div>

<script id="run-data" type="application/json">@@DATA@@</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("run-data").textContent);
const $ = (h) => { const t = document.createElement("template"); t.innerHTML = h.trim(); return t.content.firstChild; };
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (x) => (x == null ? "—" : x.toLocaleString("en-US"));
const pct1 = (x) => (x == null ? "—" : (100 * x).toFixed(1) + "%");

/* ---------- tiny helpers ---------- */
function barGroup(data, {colors, format}) {
  /* data: [{label, value, pct}] → returns html string */
  const max = Math.max(1, ...data.map(d => d.value));
  return data.map(d => {
    const w = Math.round(100 * d.value / max);
    return `<div class="row"><div class="lbl">${esc(d.label)}</div>
      <div class="track"><div class="fill" style="width:${Math.max(2, w)}%;background:${colors[d.label] || "#8899cc"}"></div></div>
      <div class="val">${fmt(d.value)} <span class="pct">${d.pct != null ? pct1(d.pct) : ""}</span></div></div>`;
  }).join("");
}
const CLS_COLOR = { "POSITIVE": "var(--pos)", "NEUTRAL": "var(--neu)", "NEGATIVE": "var(--neg)" };
const EMO_COLOR = { joy:"#f2b84b", trust:"#3ddc97", anticipation:"#6c95ff", surprise:"#f472b6",
  sadness:"#818cf8", anger:"#ff6b6b", fear:"#b388ff", disgust:"#a3b18a" };
const EMO_ORDER = ["joy","trust","anticipation","surprise","sadness","anger","fear","disgust"];
const NEG_EMOS = new Set(["anger","disgust","fear","sadness"]);
const POS_EMOS = new Set(["joy","trust","anticipation","surprise"]);
function contradictsSentiment(emotion, sentiment) {
  if (!emotion || !sentiment) return false;
  if (sentiment === "POSITIVE") return NEG_EMOS.has(emotion);
  if (sentiment === "NEGATIVE") return POS_EMOS.has(emotion);
  return false;
}

/* ---------- KPI row ---------- */
function kpis() {
  const s = D.summary;
  const acc = el("div", "kpi acc");
  acc.append(el("div","label","Accuracy vs rating"), el("div","value", pct1(s.accuracy)),
    el("div","note", `${fmt(s.agreement)} of ${fmt(s.n_parsed)} reviews agree with the star-rating label`));
  const agree = el("div", "kpi agree");
  const agreeVal = el("div","value"); agreeVal.innerHTML = `${s.agreement}<small> / ${fmt(s.n_parsed)}</small>`;
  agree.append(el("div","label","Agreement"), agreeVal,
    el("div","note","model label matches rating-derived label"));
  const n = el("div", "kpi n");
  n.append(el("div","label","Reviews scored"), el("div","value", fmt(s.n_parsed)),
    el("div","note", `${s.n_parse_fail} rows dropped for unparseable model output`));
  const base = el("div", "kpi baseline");
  const gain = s.accuracy - s.majority_share;
  base.append(el("div","label","Naive baseline"), el("div","value", pct1(s.majority_share)),
    el("div","note", `“always ${s.majority_tie ? "one class" : s.majority_class}” would score this — model ${gain >= 0 ? "+" : ""}${pct1(gain)} over it`));
  const wrap = el("div","kpis");
  wrap.append(acc, agree, n, base);
  return wrap;
}

/* ---------- charts ---------- */
function ratingDist() {
  const card = el("div", "card");
  card.append(el("div","kicker","Descriptive"), el("h2",null,"Star-rating distribution"),
    el("div","section-note",`This run's ${fmt(D.rows.length)} reviews versus the full population of ${fmt(D.pop_n)} — the balanced sample deliberately flattens the data's heavy ★4–5 skew.`));
  const legend = el("div","legend");
  legend.innerHTML = `<span><span class="sw" style="background:var(--accent)"></span>this run (share of sample)</span>
    <span><span class="sw" style="background:#8899cc;opacity:.55"></span>full population (share)</span>`;
  card.append(legend);
  const box = el("div","hbar dual");
  const rowsHtml = [];
  for (const r of D.summary.rating_dist) {
    const samplePct = 100 * r.n / D.rows.length;
    const popPct = D.pop_share[r.rating] != null ? 100 * D.pop_share[r.rating] : null;
    rowsHtml.push(`<div class="caps">★${r.rating}</div>
      <div class="row"><div class="lbl">sample</div>
        <div class="track"><div class="fill" style="width:${Math.max(2, samplePct)}%;background:var(--accent)"></div></div>
        <div class="val">${fmt(r.n)} <span class="pct">${samplePct.toFixed(1)}%</span></div></div>
      <div class="row" style="margin-bottom:10px"><div class="lbl">population</div>
        <div class="track"><div class="fill" style="width:${popPct != null ? Math.max(2, popPct) : 0}%;background:#8899cc;opacity:.55"></div></div>
        <div class="val">${popPct != null ? popPct.toFixed(1) + "%" : "—"}</div></div>`);
  }
  box.innerHTML = rowsHtml.join("");
  card.append(box); return card;
}

function classCompare() {
  const s = D.summary; const card = el("div", "card");
  card.append(el("div","kicker","Predictions"), el("h2",null,"Correct answer vs model prediction"),
    el("div","section-note","Per class: how many reviews the rating says are in the class, versus how many the model actually assigned to it."));
  const legend = el("div","legend");
  legend.innerHTML = `<span><span class="sw" style="background:var(--accent)"></span>rating-based (correct answer)</span>
    <span><span class="sw" style="background:#8899cc"></span>model prediction</span>`;
  card.append(legend);
  const out = [];
  for (const c of D.classes) {
    const correct = s.correct_class_dist[c] || 0, pred = s.pred_class_dist[c] || 0;
    const max = Math.max(1, correct, pred);
    out.push(`<div class="row"><div class="lbl">${esc(c)}</div>
      <div style="display:flex;flex-direction:column;gap:3px">
        <div class="track" style="height:9px"><div class="fill" style="width:${Math.max(2, Math.round(100*correct/max))}%;background:var(--accent)"></div></div>
        <div class="track" style="height:9px"><div class="fill" style="width:${Math.max(2, Math.round(100*pred/max))}%;background:#8899cc"></div></div>
      </div>
      <div class="val">${fmt(correct)} / ${fmt(pred)}</div></div>`);
  }
  const box = el("div","hbar"); box.innerHTML = out.join("");
  card.append(box); return card;
}

function recallChart() {
  const s = D.summary; const card = el("div","card");
  card.append(el("div","kicker","Per class"), el("h2",null,"How often each class is answered right"),
    el("div","section-note","Recall per class: share of a class's reviews the model labeled correctly."));
  const data = D.classes.map(c => ({label: c, value: s.per_class_recall[c] ? s.per_class_recall[c].recall * 100 : 0, pct: null}));
  const box = el("div","hbar");
  box.innerHTML = data.map(d => {
    const v = s.per_class_recall[d.label];
    return `<div class="row"><div class="lbl">${esc(d.label)}</div>
      <div class="track"><div class="fill" style="width:${Math.max(2, d.value)}%;background:${CLS_COLOR[d.label]}"></div></div>
      <div class="val">${v ? `${v.correct}/${v.n} · ${pct1(v.recall)}` : "—"}</div></div>`;
  }).join("");
  card.append(box); return card;
}

function confusionMatrix() {
  const s = D.summary; const card = el("div","card");
  card.append(el("div","kicker","Errors"), el("h2",null,"Confusion matrix"),
    el("div","section-note","Rows are the rating-based class; columns are what the model predicted. Diagonal = agreement."));
  const cs = D.classes;
  const legend = el("div","legend");
  legend.innerHTML = `<span><span class="sw" style="background:rgba(61,220,151,.5)"></span>agreement</span>
    <span><span class="sw" style="background:rgba(255,107,107,.5)"></span>error</span>`;
  card.append(legend);
  const grid = el("div","cm");
  grid.style.gridTemplateColumns = `auto repeat(${cs.length}, 1fr)`;
  grid.append(el("div","corner","correct ↓"));
  cs.forEach(c => grid.append(el("div","hdr", c)));
  cs.forEach(row => {
    grid.append(el("div","corner", row));
    const total = Math.max(1, (s.confusion[row] ? Object.values(s.confusion[row]).reduce((a,b)=>a+b,0) : 0));
    cs.forEach(col => {
      const n = (s.confusion[row] || {})[col] || 0;
      const cell = el("div","cell"); cell.style.background = n === 0 ? "rgba(255,255,255,.03)"
        : (row === col ? "rgba(61,220,151,.16)" : "rgba(255,107,107,.14)");
      cell.append(el("div","n", fmt(n))); cell.append(el("div","p", `${Math.round(100*n/total)}%`));
      grid.append(cell);
    });
  });
  card.append(grid);
  if (s.n_parse_fail) card.append(el("div","cmnote", `${fmt(s.n_parse_fail)} rows with unparseable output are excluded.`));
  return card;
}

/* ---------- emotions ---------- */
function emotionChart() {
  if (!D.has_emotions) return document.createDocumentFragment();
  const card = el("div","card");
  card.append(el("div","kicker","Emotions"), el("h2",null,"LLM vs word list: primary emotion"),
    el("div","section-note",`LLM judgement versus the NRC word-list derivation, over the ${fmt(D.emo_rows)} rows where both produced an emotion. They pick the same primary emotion ${pct1(D.emo_agreement)} of the time. ${D.emo_contradict_wl} word-list picks ${D.emo_contradict_wl === 1 ? "contradicts" : "contradict"} the review's own sentiment (e.g. joy on a ★1 complaint) vs ${D.emo_contradict_llm} for the LLM.`));
  const legend = el("div","legend");
  legend.innerHTML = `<span><span class="sw" style="background:var(--accent)"></span>LLM</span>
    <span><span class="sw" style="background:#8899cc"></span>NRC word list</span>`;
  card.append(legend);
  const max = Math.max(1, ...EMO_ORDER.map(e => Math.max(D.emo_llm[e] || 0, D.emo_wl[e] || 0)));
  const out = EMO_ORDER.map(e => {
    const a = D.emo_llm[e] || 0, b = D.emo_wl[e] || 0;
    return `<div class="emo-row"><div class="lbl">${e}</div>
      <div class="track"><div class="fill" style="width:${Math.max(2, Math.round(100*a/max))}%;background:${EMO_COLOR[e]};opacity:.85"></div></div>
      <div class="track"><div class="fill" style="width:${Math.max(2, Math.round(100*b/max))}%;background:${EMO_COLOR[e]};opacity:.4"></div></div>
      <div class="n">${fmt(a)} / ${fmt(b)}</div></div>`;
  }).join("");
  const box = el("div"); box.innerHTML = out; card.append(box); return card;
}

/* ---------- review table ---------- */
const FILTERS = { status: "all", cls: "all", q: "" };
function applyFilters() {
  const q = FILTERS.q.toLowerCase();
  return D.rows.filter(r => {
    if (FILTERS.status === "ok" && (r.correct_class !== r.sentiment_pred)) return false;
    if (FILTERS.status === "bad" && (r.correct_class === r.sentiment_pred)) return false;
    if (FILTERS.cls !== "all" && r.correct_class !== FILTERS.cls) return false;
    if (q && !((r.title || "") + " " + (r.text || "") + " " + (r.emotion_pred || "")).toLowerCase().includes(q)) return false;
    return true;
  });
}
const EMPTY = el("tr").append(el("td","empty", "No reviews match the current filters. Try widening them."));
function renderTable(rows) {
  const old = document.getElementById("rtbody");
  if (old) old.remove();
  const legacy = el("tr"); const legacyTd = el("td","empty", "No reviews match the current filters.");
  legacy.append(legacyTd);

  const tb = el("tbody"); tb.id = "rtbody";
  if (!rows.length) { tb.append(legacy); }
  let expandedIdx = -1;
  rows.forEach((r, i) => {
    const tr = el("tr"); tr.dataset.i = i;
    const ok = r.correct_class === r.sentiment_pred;
    tr.append(
      el("td","stars", "★".repeat(Math.round(r.rating))),
      el("td","title-cell", r.title || "(no title)"),
      el("td","text-cell", r.text || ""),
      (() => { const t = el("td"); t.append(el("span","badge " + r.correct_class.toLowerCase(), r.correct_class), el("span","badge tiny " + (ok ? "pos" : "neg"), ok ? "✓" : "✗")); return t; })(),
      (() => { const t = el("td"); t.append(el("span","badge llm", r.sentiment_pred || "—")); return t; })(),
    );
    if (D.has_emotions) {
      const t = el("td");
      const llmB = el("span","badge llm", r.emotion_pred || "—");
      if (contradictsSentiment(r.emotion_pred, r.sentiment_pred)) {
        llmB.classList.add("warn");
        llmB.title = "LLM emotion contradicts its own sentiment label";
      }
      t.append(llmB);
      if (r.wordlist_emotion) {
        const wlB = el("span","badge wl", r.wordlist_emotion);
        if (contradictsSentiment(r.wordlist_emotion, r.sentiment_pred)) {
          wlB.classList.add("warn");
          wlB.title = "word-list emotion contradicts the review's sentiment";
        }
        t.append(" ");
        t.append(wlB);
      }
      tr.append(t);
    }
    tr.addEventListener("click", () => {
      const existing = document.getElementById("exp" + i);
      if (existing) { existing.remove(); return; }
      const ex = el("tr","expand"); ex.id = "exp" + i;
      const td = el("td"); td.colSpan = D.has_emotions ? 6 : 5;
      td.append(el("div","full", r.text || "(no text)"));
      if (r.raw_model_output) td.append(el("div","raw", "model output: " + r.raw_model_output));
      if (r.emotion_missing) td.append(el("div","cmnote","LLM emotion missing/out of vocabulary for this row."));
      ex.append(td); tr.after(ex);
    });
    tb.append(tr);
  });
  document.getElementById("rexplorer").append(tb);
}
function tableSection() {
  const sec = el("div");
  sec.append(el("div","kicker","Explorer"), el("h2",null,"Review explorer"),
    el("div","section-note","Filter to a subset with a live count; click any row for the full text and the model's raw output."));
  const toolbar = el("div","toolbar");
  const fstatus = el("div","fgroup"); fstatus.id = "fstatus";
  const mk = (id, label, count) => { const b = el("button","fbtn"+ (id==="on"?" on":"")); b.dataset.k = id; b.innerHTML = `${label} <span class="c">${count}</span>`; return b; };
  const matches = D.rows.filter(r => r.correct_class === r.sentiment_pred).length;
  fstatus.append(mk("all","All", D.rows.length), mk("ok","Matched", matches), mk("bad","Mismatched", D.rows.length - matches));
  const fclass = el("div","fgroup"); fclass.id = "fclass";
  const counts = {}; D.rows.forEach(r => counts[r.correct_class] = (counts[r.correct_class] || 0) + 1);
  fclass.append(mk("all","Any class", D.rows.length));
  for (const c of D.classes) fclass.append(mk(c, c, counts[c] || 0));
  const search = el("input","search"); search.type = "text"; search.placeholder = "Search title, text, emotion…";
  toolbar.append(fstatus, fclass, search);
  sec.append(toolbar);
  const cl = el("div","countline"); cl.id = "cline";
  sec.append(cl);
  const table = el("table");
  const head = el("thead"); const hr = el("tr");
  hr.append(el("th",null,"Rating"), el("th",null,"Title"), el("th",null,"Review"), el("th",null,"Correct"), el("th",null,"Model"));
  if (D.has_emotions) hr.append(el("th",null,"Emotion (LLM / word list)"));
  head.append(hr); table.append(head);
  table.id = "rexplorer"; sec.append(table);
  const update = () => {
    const rows = applyFilters();
    document.getElementById("cline").innerHTML = `Showing <b>${fmt(rows.length)}</b> of <b>${fmt(D.rows.length)}</b> reviews`;
    fstatus.querySelectorAll(".fbtn").forEach(b => b.classList.toggle("on", b.dataset.k === FILTERS.status));
    fclass.querySelectorAll(".fbtn").forEach(b => b.classList.toggle("on", b.dataset.k === FILTERS.cls));
    renderTable(rows);
  };
  fstatus.addEventListener("click", e => { const b = e.target.closest(".fbtn"); if (!b) return; FILTERS.status = b.dataset.k; update(); });
  fclass.addEventListener("click", e => { const b = e.target.closest(".fbtn"); if (!b) return; FILTERS.cls = b.dataset.k; update(); });
  search.addEventListener("input", () => { FILTERS.q = search.value; update(); });
  // First render must wait until the section is attached to the document
  // (update() looks up #cline / #rexplorer via the document).
  sec.update = update;
  return sec;
}

/* ---------- compose ---------- */
const app = document.getElementById("app");
const m = document.createElement("div");
m.innerHTML = `<div class="masthead"><div class="wordmark"><span class="pre">MBAX 6418</span>Gift Card Review Classifier</div></div>
<p class="sub">Sentiment &amp; emotion classification of Amazon “Gift Cards” reviews (Amazon Reviews ’23, McAuley Lab) — scored against the reviewer’s star rating as ground truth.</p>
<div class="meta">
  <span class="chip">run <b>${esc(D.run)}</b></span>
  <span class="chip">sample <b>${esc(D.mode)}</b></span>
  <span class="chip">n <b>${fmt(D.rows.length)}</b></span>
  ${D.mode === "balanced" ? `<span class="chip">seed <b>${D.seed}</b> · ${D.per_class} per class</span>` : ""}
  <span class="chip">classes <b>${D.classes.join(" / ")}</b></span>
  ${D.has_emotions ? `<span class="chip">emotions <b>LLM + NRC</b></span>` : ""}
  <span class="chip">model <b>${esc(D.model)}</b></span>
</div>`;
app.append(...m.children);
app.append(kpis());
const row1 = el("div","grid2"); row1.append(ratingDist(), classCompare()); app.append(row1);
const row2 = el("div", D.has_emotions ? "grid3" : "grid2");
row2.append(recallChart(), confusionMatrix());
if (D.has_emotions) row2.append(emotionChart());
app.append(row2);
const ts = tableSection();
app.append(ts);
ts.update();
const foot = el("footer");
foot.innerHTML = `<b>Data:</b> Amazon Reviews ’23 “Gift Cards” category, McAuley Lab (UCSD),
  <a style="color:var(--accent-bright)" href="https://amazon-reviews-2023.github.io">amazon-reviews-2023.github.io</a> · <b>Ground truth:</b> star rating
  (${D.classes.length === 3 ? "4–5 positive, 3 neutral, 1–2 negative" : "4–5 positive, else negative"}), never shown to the model ·
  <b>Model:</b> ${esc(D.model)} (OpenAI-compatible endpoint) · <b>Emotions:</b> NRC emotion lexicon (Mohammad &amp; Turney).
  Every number on this page is recomputed from the saved run output (${esc(D.run)}.jsonl).`;
app.append(foot);
</script>
</body>
</html>
"""


def load_run(run_name):
    import os
    path = f"output/runs/{run_name}.jsonl"
    if not os.path.exists(path):
        sys.exit(f"run file not found: {path}")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    meta = {}
    summary_path = f"output/runs/{run_name}_summary.json"
    if os.path.exists(summary_path):
        meta = json.load(open(summary_path))
    return rows, meta


def load_population(path="data/Gift_Cards.jsonl.gz"):
    """Rating distribution of the FULL dataset (single pass over the .gz)."""
    if not os.path.exists(path):
        return None
    counts = collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            counts[int(json.loads(line)["rating"])] += 1  # int keys, like the sample side
    n = sum(counts.values())
    return {"n": n, "share": {k: v / n for k, v in counts.items()}}


def summarize(rows):
    """Recompute all aggregates from the raw rows (mirrors score.py)."""
    classes = list(rows[0]["classes_mode"] == "3-class" and ("POSITIVE", "NEUTRAL", "NEGATIVE")
                   or ("POSITIVE", "NEGATIVE"))
    parsed = [r for r in rows if not r["parse_fail"]]
    n = len(parsed)

    rating_dist = collections.Counter(int(r["rating"]) for r in parsed)
    correct_dist = collections.Counter(r["correct_class"] for r in parsed)
    pred_dist = collections.Counter(r["sentiment_pred"] for r in parsed)
    confusion = collections.defaultdict(collections.Counter)
    for r in parsed:
        confusion[r["correct_class"]][r["sentiment_pred"]] += 1

    per_class_recall = {}
    for c in classes:
        total = sum(confusion[c].values())
        per_class_recall[c] = {"n": total, "correct": confusion[c].get(c, 0),
                               "recall": (confusion[c].get(c, 0) / total) if total else None}

    agreement = sum(confusion[c].get(c, 0) for c in classes)
    accuracy = agreement / n if n else 0.0
    if correct_dist:
        counts = list(correct_dist.values())
        majority_class = correct_dist.most_common(1)[0][0]
        # Under balanced sampling every class ties; "always X" is misleading then.
        majority_tie = len(set(counts)) == 1
        majority_share = (correct_dist[majority_class] / n) if n else 0.0
    else:
        majority_class, majority_tie, majority_share = "—", False, 0.0

    has_emotions = any("emotion_pred" in r for r in rows)
    emo_llm = collections.Counter(r["emotion_pred"] for r in parsed if r.get("emotion_pred"))
    emo_wl = collections.Counter(r["wordlist_emotion"] for r in parsed if r.get("wordlist_emotion"))
    emo_both = [(r.get("emotion_pred"), r.get("wordlist_emotion"))
                for r in parsed if r.get("emotion_pred") and r.get("wordlist_emotion")]
    emo_agreement = (sum(1 for a, b in emo_both if a == b) / len(emo_both)) if emo_both else None

    negative_emos = {"anger", "disgust", "fear", "sadness"}
    positive_emos = {"joy", "trust", "anticipation", "surprise"}
    def contradicts(emo, sent):
        if not emo or not sent:
            return False
        return (sent == "POSITIVE" and emo in negative_emos
                or sent == "NEGATIVE" and emo in positive_emos)

    emo_contradict_wl = sum(1 for r in parsed if contradicts(r.get("wordlist_emotion"), r["sentiment_pred"]))
    emo_contradict_llm = sum(1 for r in parsed if contradicts(r.get("emotion_pred"), r["sentiment_pred"]))

    return {
        "n_total": len(rows), "n_parsed": n, "n_parse_fail": len(rows) - n,
        "accuracy": accuracy, "agreement": agreement,
        "confusion": {c: dict(confusion[c]) for c in classes},
        "per_class_recall": per_class_recall,
        "correct_class_dist": dict(correct_dist), "pred_class_dist": dict(pred_dist),
        "rating_dist": [{"rating": k, "n": rating_dist[k]} for k in sorted(rating_dist)],
        "majority_class": majority_class, "majority_share": majority_share,
        "majority_tie": majority_tie,
        "emo_llm": dict(emo_llm), "emo_wl": dict(emo_wl), "emo_agreement": emo_agreement,
        "emo_both_n": len(emo_both),
        "emo_contradict_wl": emo_contradict_wl, "emo_contradict_llm": emo_contradict_llm,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under output/runs/")
    ap.add_argument("-o", "--out", default="dashboard.html")
    args = ap.parse_args(argv)

    rows, meta = load_run(args.run)
    summary = summarize(rows)
    population = load_population()

    # Cross-check against the saved scoring summary when one exists.
    try:
        saved = json.load(open(f"output/runs/{args.run}_summary.json"))
        if "accuracy" in saved and abs(saved["accuracy"] - summary["accuracy"]) > 1e-9:
            print(f"WARNING: accuracy mismatch vs saved summary "
                  f"({saved['accuracy']} vs {summary['accuracy']})")
    except FileNotFoundError:
        pass

    row_payload = []
    for r in rows:
        row_payload.append({
            "rating": r["rating"], "title": r["title"], "text": (r["text"] or "")[:600],
            "correct_class": r["correct_class"], "sentiment_pred": r["sentiment_pred"],
            "emotion_pred": r.get("emotion_pred"), "wordlist_emotion": r.get("wordlist_emotion"),
            "emotion_missing": r.get("emotion_missing", False),
            "parse_fail": r.get("parse_fail", False),
            "raw_model_output": r.get("raw_model_output"),
            "verified_purchase": r.get("verified_purchase"),
            "helpful_vote": r.get("helpful_vote"),
        })

    data = {
        "run": args.run, "mode": meta.get("mode", "?"), "n": len(rows),
        "seed": meta.get("seed"), "per_class": meta.get("per_class"),
        "classes": [("POSITIVE", "NEUTRAL", "NEGATIVE") if rows and rows[0]["classes_mode"] == "3-class"
                    else ("POSITIVE", "NEGATIVE")][0],
        "has_emotions": summary["emo_both_n"] > 0 or any(r.get("emotion_pred") for r in rows),
        "model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
        "summary": summary, "rows": row_payload,
    }
    data["emo_agreement"] = summary["emo_agreement"]
    data["emo_llm"] = summary["emo_llm"]
    data["emo_wl"] = summary["emo_wl"]
    data["emo_rows"] = summary["emo_both_n"]
    data["emo_contradict_wl"] = summary["emo_contradict_wl"]
    data["emo_contradict_llm"] = summary["emo_contradict_llm"]
    data["pop_n"] = population["n"] if population else None
    data["pop_share"] = population["share"] if population else {}

    html = TEMPLATE.replace("@@RUN@@", args.run).replace("@@DATA@@", json.dumps(data))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"wrote {args.out} ({len(html)/1024:.0f} KB, {len(rows)} rows embedded)")
    print(f"accuracy {summary['accuracy']:.3f} ({summary['agreement']}/{summary['n_parsed']}), "
          f"baseline always-{summary['majority_class']} = {summary['majority_share']:.3f}")
    if summary["emo_agreement"] is not None:
        print(f"LLM/word-list emotion agreement {summary['emo_agreement']:.3f} "
              f"({summary['emo_both_n']} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
