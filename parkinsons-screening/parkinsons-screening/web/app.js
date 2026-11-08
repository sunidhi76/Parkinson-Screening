"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const EXAMPLE_ROW = "phon_R01_S01_1,119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,1,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654";

async function api(path, opts) {
  try {
    const r = await fetch(path, opts); let j = {};
    try { j = await r.json(); } catch (_) { /* non-JSON error body */ }
    if (!r.ok) return { error: j.detail || j.error || ("HTTP " + r.status) };
    return j;
  } catch (e) { return { error: "Could not reach the server. Is it running? (" + e.message + ")" }; }
}

function resultHTML(j) {
  if (j.error) return `<div class="errbox">${esc(typeof j.error === "string" ? j.error : JSON.stringify(j.error))}</div>`;
  let h = '<div class="res">';
  if (j.SYNTHETIC_MODEL_WARNING) h += `<div class="errbox"><strong>${esc(j.SYNTHETIC_MODEL_WARNING)}</strong></div>`;
  if (j.label) h += `<div class="warnbox"><strong>${esc(j.label)}</strong></div>`;
  if (j.probability === null || j.probability === undefined) h += "<p><strong>No score produced</strong> (input quality).</p>";
  else h += `<p>Model output: <strong>${(j.probability * 100).toFixed(1)}%</strong> — risk band: <strong class="band-${esc(j.risk_band)}">${esc(j.risk_band)}</strong></p>`;
  if (j.uncertain_please_retest) h += '<div class="warnbox">Uncertain / unavailable → please retest.</div>';
  const w = (j.quality && (j.quality.warnings || j.quality.out_of_training_range)) || [];
  if (w.length) h += '<div class="warnbox"><strong>Input-quality checks</strong><ul>' + w.map((x) => `<li>${esc(x)}</li>`).join("") + "</ul></div>";
  if (j.validity_note) h += `<p class="muted">${esc(j.validity_note)}</p>`;
  if (j.model_provenance) h += `<p class="muted">${esc(j.model_provenance)}</p>`;
  return h + `<p class="muted"><em>${esc(j.disclaimer || "")}</em></p></div>`;
}

(async () => {
  const h = await api("/health");
  $("health").textContent = h.error ? "Server unreachable: " + h.error
    : `Server OK · voice models: ${h.voice.loaded ? "loaded" : "missing"} · drawing model: ${h.drawing.loaded ? "loaded (trained on " + h.drawing.trained_on + ")" : "missing"}`;
  if (h.drawing && h.drawing.trained_on === "synthetic") $("drawOut").innerHTML = `<div class="errbox"><strong>${esc(h.drawing.warning)}</strong></div>`;
})();

$("exampleRow").onclick = () => { $("row").value = EXAMPLE_ROW; };
$("scoreRow").onclick = async () => {
  const model = $("vmodel").value; const body = { row: $("row").value }; if (model) body.model = model;
  $("rowOut").innerHTML = "…";
  const j = await api("/predict/voice-features", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if ($("row").value.trim().startsWith("phon_R01")) j.validity_note = "NOTE: this example is a TRAINING row, so the output is in-sample and not a test of the model. " + (j.validity_note || "");
  $("rowOut").innerHTML = resultHTML(j);
};

/* ---- audio ---- */
async function handleAudio(arrayBuffer) {
  $("audioOut").innerHTML = "Analysing…";
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ctx.decodeAudioData(arrayBuffer); const chans = [];
    for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
    const res = AudioFeatures.analyze(AudioFeatures.toMono(chans), buf.sampleRate);
    const q = res.quality; let html = `<table><tr><th>Quality check</th><th>Value</th></tr>
      <tr><td>duration</td><td>${q.duration_s.toFixed(1)} s</td></tr><tr><td>clipping</td><td>${(q.clipping_fraction * 100).toFixed(3)}%</td></tr>
      <tr><td>level</td><td>${q.rms_dbfs.toFixed(0)} dBFS</td></tr><tr><td>voiced fraction</td><td>${((q.voiced_fraction || 0) * 100).toFixed(0)}%</td></tr>
      <tr><td>periodicity</td><td>${(q.mean_periodicity || 0).toFixed(2)}</td></tr></table>`;
    for (const c of q.critical) html += `<div class="errbox">${esc(c)}</div>`;
    for (const w of q.warnings) html += `<div class="warnbox">${esc(w)}</div>`;
    if (res.features) {
      html += "<table><tr><th>Approximate feature</th><th>Value</th></tr>" + Object.entries(res.features).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${Number.isFinite(v) ? v.toPrecision(4) : "n/a"}</td></tr>`).join("") + "</table>";
      const ok = q.critical.length === 0 && Object.values(res.features).every(Number.isFinite);
      html += `<div class="row"><button id="scoreAudio" ${ok ? "" : "disabled"}>Score with unvalidated demo model</button></div><div id="audioScore"></div>`;
      $("audioOut").innerHTML = html;
      if (ok) $("scoreAudio").onclick = async () => {
        const j = await api("/predict/voice-features", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features: res.features }) });
        $("audioScore").innerHTML = resultHTML(j);
      };
    } else $("audioOut").innerHTML = html;
  } catch (e) { $("audioOut").innerHTML = `<div class="errbox">Could not decode the audio: ${esc(e.message)}</div>`; }
}
$("audioFile").onchange = async (e) => { if (e.target.files[0]) handleAudio(await e.target.files[0].arrayBuffer()); };
$("rec").onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const rec = new MediaRecorder(stream), chunks = []; rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = async () => { stream.getTracks().forEach((t) => t.stop()); $("rec").disabled = false; $("rec").textContent = "● Record 5 s (say “aaah”)"; handleAudio(await new Blob(chunks).arrayBuffer()); };
    $("rec").disabled = true; $("rec").textContent = "Recording…"; rec.start(); setTimeout(() => rec.stop(), 5000);
  } catch (e) { $("audioOut").innerHTML = `<div class="errbox">Microphone unavailable: ${esc(e.message)}</div>`; }
};

/* ---- drawing ---- */
async function sendImage(blob) {
  $("drawOut").innerHTML = "…"; const fd = new FormData(); fd.append("file", blob, "spiral.png");
  $("drawOut").innerHTML = resultHTML(await api("/predict/drawing", { method: "POST", body: fd }));
}
$("scoreImg").onclick = () => { const f = $("imgFile").files[0]; if (!f) { $("drawOut").innerHTML = '<div class="errbox">Choose an image first.</div>'; return; } sendImage(f); };
const pad = $("pad"), g = pad.getContext("2d"); let down = false;
function reset() { g.fillStyle = "#fff"; g.fillRect(0, 0, pad.width, pad.height); g.lineWidth = 3; g.lineCap = "round"; g.strokeStyle = "#000"; }
reset();
const pos = (e) => { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * pad.width / r.width, (e.clientY - r.top) * pad.height / r.height]; };
pad.onpointerdown = (e) => { down = true; g.beginPath(); g.moveTo(...pos(e)); pad.setPointerCapture(e.pointerId); };
pad.onpointermove = (e) => { if (down) { g.lineTo(...pos(e)); g.stroke(); } };
pad.onpointerup = () => { down = false; };
$("clearPad").onclick = reset;
$("scorePad").onclick = () => pad.toBlob(sendImage, "image/png");
