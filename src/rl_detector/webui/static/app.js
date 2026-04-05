const els = {
  inputText: document.getElementById("inputText"),
  runBtn: document.getElementById("runBtn"),
  engineDot: document.getElementById("engineDot"),
  engineText: document.getElementById("engineText"),
  status: document.getElementById("status"),
  progressBar: document.getElementById("progressBar"),
  progressText: document.getElementById("progressText"),
  verdict: document.getElementById("verdict"),
  score: document.getElementById("score"),
  checkpointUsed: document.getElementById("checkpointUsed"),
  annotatedText: document.getElementById("annotatedText"),
  tellTpl: document.getElementById("tellTpl"),
};

let statusPollTimer = null;
let configPollTimer = null;
let engineReady = false;
let progressAnimTimer = null;
let progressDisplayPct = 0;
let progressBackendPct = 0;
let progressStageName = "Queued";
let progressStageStartedAt = 0;

const stagePlan = {
  Queued: { start: 0, end: 5, durationMs: 7500 },
  "Starting pipeline": { start: 5, end: 30, durationMs: 13500 },
  "Preparing prompt": { start: 30, end: 45, durationMs: 18000 },
  "Sampling one rollout": { start: 45, end: 65, durationMs: 66000 },
  "Parsed indicators": { start: 65, end: 80, durationMs: 13500 },
  "Scoring indicators": { start: 80, end: 98, durationMs: 54000 },
  Complete: { start: 98, end: 100, durationMs: 3600 },
  Failed: { start: 0, end: 100, durationMs: 1 },
};

function setStatus(text) {
  els.status.textContent = text;
}

function setProgress(pct) {
  const safePct = Math.max(0, Math.min(100, Number(pct) || 0));
  els.progressBar.style.width = `${safePct}%`;
  els.progressText.textContent = `${safePct}%`;
}

function stopProgressAnimation() {
  if (progressAnimTimer) {
    clearInterval(progressAnimTimer);
    progressAnimTimer = null;
  }
}

function startProgressAnimation() {
  stopProgressAnimation();
  progressAnimTimer = setInterval(() => {
    const now = Date.now();
    const p = stagePlan[progressStageName] || { start: progressBackendPct, end: progressBackendPct, durationMs: 1 };
    const dt = Math.max(0, now - progressStageStartedAt);
    const t = Math.min(1, dt / Math.max(1, p.durationMs));

    // slower log-shaped growth: intentionally pessimistic to avoid over-promising
    const logU = Math.log1p(9 * t) / Math.log1p(9);
    const slowU = Math.pow(logU, 1.8);
    let predicted = p.start + (p.end - p.start) * slowU;

    // keep a small headroom before stage boundary unless backend confirmed it
    if (progressBackendPct < p.end) {
      predicted = Math.min(predicted, p.end - 0.8);
    }

    const target = Math.max(progressBackendPct, predicted);
    progressDisplayPct = Math.max(progressDisplayPct, progressDisplayPct + (target - progressDisplayPct) * 0.08);
    setProgress(Math.floor(progressDisplayPct));
  }, 120);
}

function scoreSign(v) {
  if (v > 0) return "pos";
  if (v < 0) return "neg";
  return "neu";
}

function leaningLabel(v) {
  if (v > 0) return "AI-leaning";
  if (v < 0) return "Human-leaning";
  return "Neutral";
}

function renderSegments(segments) {
  els.annotatedText.innerHTML = "";
  if (!segments || segments.length === 0) {
    els.annotatedText.textContent = "No text to show.";
    return;
  }

  for (const seg of segments) {
    if (seg.type === "plain") {
      els.annotatedText.append(document.createTextNode(seg.text));
      continue;
    }

    const node = els.tellTpl.content.firstElementChild.cloneNode(true);
    node.dataset.scoreSign = scoreSign(seg.score);
    node.querySelector(".tell-text").textContent = seg.text;
    const why = seg.explanation || "No explanation provided.";
    const lean = leaningLabel(Number(seg.score) || 0);
    const scoreText = Number(seg.score).toFixed(2);
    node.querySelector(".tell-tip").textContent = `${lean} | score ${scoreText}: ${why}`;
    node.setAttribute("aria-label", `Highlighted tell: ${seg.text}. ${lean}. score ${scoreText}. ${why}`);
    els.annotatedText.append(node);
  }
}

async function pullConfig() {
  const res = await fetch("/api/config");
  const data = await res.json();
  const status = data.engine_status || "not_configured";
  engineReady = status === "ready";
  els.runBtn.disabled = !engineReady;

  els.engineDot.classList.remove("engine-red", "engine-green");
  if (engineReady) {
    els.engineDot.classList.add("engine-green");
    els.engineText.textContent = "Inference engine: ready";
    setStatus("Startup sampler ready.");
    if (configPollTimer) {
      clearInterval(configPollTimer);
      configPollTimer = null;
    }
    return;
  }

  els.engineDot.classList.add("engine-red");
  if (status === "loading") {
    els.engineText.textContent = "Inference engine: loading";
    setStatus("Engine is loading in background...");
  } else if (status === "error") {
    els.engineText.textContent = "Inference engine: failed";
    setStatus(data.engine_error || "Engine failed to load.");
  } else {
    els.engineText.textContent = "Inference engine: not configured";
    setStatus("No startup checkpoint configured.");
  }
}

async function runAnalysis() {
  const text = els.inputText.value.trim();
  if (!text) {
    setStatus("Please add text first.");
    return;
  }

  if (!engineReady) {
    setStatus("Inference engine is not ready yet.");
    return;
  }

  els.runBtn.disabled = true;
  const t0 = Date.now();
  progressDisplayPct = 0;
  progressBackendPct = 0;
  progressStageName = "Queued";
  progressStageStartedAt = Date.now();
  setProgress(0);
  startProgressAnimation();
  setStatus("Queued");

  const payload = { text };

  const startRes = await fetch("/api/analyze/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!startRes.ok) {
    stopProgressAnimation();
    setProgress(0);
    const maybeError = await startRes.json();
    setStatus(maybeError.detail || "Request failed.");
    els.runBtn.disabled = false;
    return;
  }

  const startData = await startRes.json();
  const jobId = startData.job_id;
  let cursor = 0;

  async function pollStatus() {
    const statusRes = await fetch(`/api/analyze/status/${jobId}?after=${cursor}`);
    if (!statusRes.ok) {
      return;
    }
    const data = await statusRes.json();
    cursor = data.next_offset;

    if (data.progress_stage && data.progress_stage !== progressStageName) {
      progressStageName = data.progress_stage;
      progressStageStartedAt = Date.now();
    }
    progressBackendPct = Math.max(progressBackendPct, Number(data.progress_pct) || 0);

    const elapsed = Math.floor((Date.now() - t0) / 1000);
    setStatus(`${data.progress_stage}; ${elapsed}s elapsed`);

    for (const line of data.logs || []) {
      console.log(`[pipeline] ${line}`);
    }

    if (data.status === "done") {
      clearInterval(statusPollTimer);
      statusPollTimer = null;
      progressStageName = "Complete";
      progressBackendPct = 100;
      progressStageStartedAt = Date.now();
      const out = data.result || {};
      els.verdict.textContent = out.verdict;
      els.score.textContent = Number(out.aggregate_score).toFixed(3);
      els.checkpointUsed.textContent = out.used_checkpoint || "-";
      renderSegments(out.segments || []);
      const total = Math.floor((Date.now() - t0) / 1000);
      setProgress(100);
      stopProgressAnimation();
      setStatus(`Done; ${total}s total`);
      els.runBtn.disabled = false;
      return;
    }

    if (data.status === "error") {
      clearInterval(statusPollTimer);
      statusPollTimer = null;
      progressStageName = "Failed";
      progressBackendPct = Math.max(progressBackendPct, Number(data.progress_pct) || 0);
      setProgress(progressBackendPct || 0);
      stopProgressAnimation();
      setStatus(data.error || "Pipeline failed.");
      els.runBtn.disabled = false;
    }
  }

  await pollStatus();
  statusPollTimer = setInterval(pollStatus, 700);
}

els.runBtn.addEventListener("click", runAnalysis);
pullConfig();
configPollTimer = setInterval(pullConfig, 2000);
