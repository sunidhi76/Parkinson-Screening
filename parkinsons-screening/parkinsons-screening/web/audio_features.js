/* Crude, browser-side voice feature extraction (sustained vowel). Runs entirely client-side: audio is never uploaded.
 * These are APPROXIMATIONS of seven UCI columns and are NOT computed like the UCI features (different software,
 * microphone, protocol). Any score derived from them is an "unvalidated demo, not comparable to the training data." */
(function (root) {
  "use strict";
  const F0_MIN = 70, F0_MAX = 400;

  function toMono(chans) {
    if (chans.length === 1) return chans[0];
    const n = chans[0].length, out = new Float32Array(n);
    for (const c of chans) for (let i = 0; i < n; i++) out[i] += c[i] / chans.length;
    return out;
  }

  /* normalised autocorrelation of x[start:start+len] over lags in [lo,hi]; returns best lag (refined) and its r */
  function bestLag(x, start, len, lo, hi) {
    let e0 = 0; for (let i = 0; i < len; i++) e0 += x[start + i] * x[start + i];
    if (e0 < 1e-12) return { lag: 0, r: 0 };
    const rs = new Float64Array(hi + 2);
    for (let lag = lo - 1; lag <= hi + 1; lag++) {
      let s = 0, e1 = 0;
      for (let i = 0; i + lag < len; i++) { s += x[start + i] * x[start + i + lag]; e1 += x[start + i + lag] * x[start + i + lag]; }
      rs[lag] = s / Math.sqrt(e0 * e1 + 1e-12);
    }
    let best = lo, br = -1;
    for (let lag = lo; lag <= hi; lag++) if (rs[lag] > br && rs[lag] >= rs[lag - 1] && rs[lag] >= rs[lag + 1]) { br = rs[lag]; best = lag; }
    if (br < 0) return { lag: 0, r: 0 };
    const a = rs[best - 1], b = rs[best], c = rs[best + 1], d = a - 2 * b + c;
    const off = Math.abs(d) > 1e-12 ? 0.5 * (a - c) / d : 0;
    return { lag: best + off, r: Math.min(Math.max(b, 0), 0.9999) };
  }

  function analyze(samples, fs) {
    const n = samples.length, dur = n / fs;
    const q = { duration_s: dur, warnings: [], critical: [] };
    let peak = 0, clip = 0, sq = 0;
    for (let i = 0; i < n; i++) { const a = Math.abs(samples[i]); if (a > peak) peak = a; if (a >= 0.99) clip++; sq += samples[i] * samples[i]; }
    q.clipping_fraction = clip / Math.max(n, 1);
    q.rms_dbfs = 20 * Math.log10(Math.sqrt(sq / Math.max(n, 1)) + 1e-12);
    if (dur < 1.0) q.critical.push("Recording too short (<1 s). Sustain a vowel such as 'aaah' for 3-5 s.");
    else if (dur < 3.0) q.warnings.push("Short recording (<3 s): estimates will be unstable.");
    if (q.clipping_fraction > 0.001) q.critical.push("Clipping detected (" + (100 * q.clipping_fraction).toFixed(2) + "% of samples). Lower the input gain / move back.");
    if (q.rms_dbfs < -45) q.critical.push("Signal very quiet (" + q.rms_dbfs.toFixed(0) + " dBFS). Move closer or raise the gain.");

    const win = Math.round(0.04 * fs), hop = Math.round(0.01 * fs);
    const lo = Math.floor(fs / F0_MAX), hi = Math.floor(fs / F0_MIN);
    const f0s = [], rs = [], frameRms = [];
    for (let s = 0; s + win + hi < n; s += hop) {
      let e = 0; for (let i = 0; i < win; i++) e += samples[s + i] * samples[s + i];
      const rms = Math.sqrt(e / win); frameRms.push(rms);
      if (20 * Math.log10(rms + 1e-12) < -50) continue;
      const bl = bestLag(samples, s, win, lo, hi);
      if (bl.r > 0.5 && bl.lag > 0) { f0s.push(fs / bl.lag); rs.push(bl.r); }
    }
    q.voiced_fraction = frameRms.length ? f0s.length / frameRms.length : 0;
    q.mean_periodicity = rs.length ? rs.reduce((a, b) => a + b, 0) / rs.length : 0;
    if (f0s.length < 20) { q.critical.push("Too little voiced signal found. Record a clear sustained vowel."); return { features: null, quality: q }; }
    if (q.voiced_fraction < 0.5) q.warnings.push("Less than half the recording is voiced.");
    if (q.mean_periodicity < 0.85) q.warnings.push("Low periodicity: recording is noisy, breathy or has background noise.");

    const sorted = f0s.slice().sort((a, b) => a - b), pct = p => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
    const Fo = f0s.reduce((a, b) => a + b, 0) / f0s.length, Fhi = pct(0.98), Flo = pct(0.02);

    /* cycle-to-cycle jitter/shimmer: track peaks one period apart on the median-f0 estimate */
    const T0 = fs / sorted[Math.floor(sorted.length / 2)], peaks = [], amps = [];
    let i = Math.floor(n * 0.1), end = Math.floor(n * 0.9);
    while (i < end) {
      let m = i, mv = -Infinity; const w = Math.floor(T0 * 0.5);
      for (let j = i; j < Math.min(i + w + Math.floor(T0 * 0.5), end); j++) if (samples[j] > mv) { mv = samples[j]; m = j; }
      peaks.push(m); amps.push(mv); i = m + Math.floor(T0 * 0.75);
    }
    const per = []; for (let k = 1; k < peaks.length; k++) per.push(peaks[k] - peaks[k - 1]);
    const goodP = per.filter(p => p > 0.6 * T0 && p < 1.4 * T0);
    let jit = NaN, shim = NaN;
    if (goodP.length > 10) {
      let d = 0; for (let k = 1; k < goodP.length; k++) d += Math.abs(goodP[k] - goodP[k - 1]);
      jit = 100 * (d / (goodP.length - 1)) / (goodP.reduce((a, b) => a + b, 0) / goodP.length);
      let ds = 0; for (let k = 1; k < amps.length; k++) ds += Math.abs(amps[k] - amps[k - 1]);
      shim = (ds / (amps.length - 1)) / (amps.reduce((a, b) => a + b, 0) / amps.length);
    } else q.warnings.push("Could not track enough glottal cycles for jitter/shimmer.");
    const nhrs = rs.map(r => (1 - r) / r), nhr = nhrs.reduce((a, b) => a + b, 0) / nhrs.length;
    const hnr = rs.map(r => 10 * Math.log10(r / (1 - r))).reduce((a, b) => a + b, 0) / rs.length;
    return { features: { "MDVP:Fo(Hz)": Fo, "MDVP:Fhi(Hz)": Fhi, "MDVP:Flo(Hz)": Flo, "MDVP:Jitter(%)": jit / 100 * 1,
                         "MDVP:Shimmer": shim, "NHR": nhr, "HNR": hnr }, quality: q,
             note: "MDVP:Jitter(%) is returned as a fraction (UCI stores e.g. 0.0078 for 0.78%)." };
  }

  const api = { analyze, toMono };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.AudioFeatures = api;
})(typeof self !== "undefined" ? self : this);
