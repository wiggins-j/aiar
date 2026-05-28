let lastCallId = null;
let lastVerdict = null;

function $(id) { return document.getElementById(id); }

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function ratingBadge(rating) {
  const cls = rating === "good" ? "badge-good"
    : rating === "partial" ? "badge-partial"
    : "badge-bad";
  return { cls, text: rating || "?" };
}

// ---- Grounding Context dropdowns -------------------------------------------
//
// Save-on-change: every dropdown POSTs immediately, shows "Applied" inline,
// fades after a few seconds. Failures surface inline and do not silently
// revert — the user re-picks. Each control has its own status span so the
// user can see which field was just changed.

const STATUS_FADE_MS = 2500;
const _statusTimers = new Map();

function showStatus(id, text, mode) {
  const el = $(id);
  if (!el) return;
  const prior = _statusTimers.get(id);
  if (prior) { clearTimeout(prior); _statusTimers.delete(id); }
  el.textContent = text;
  el.classList.remove("gc-status-applied", "gc-status-fading",
                      "gc-status-error", "gc-status-pending");
  if (mode === "applied") {
    el.classList.add("gc-status-applied");
    const timer = setTimeout(() => {
      el.classList.add("gc-status-fading");
      const clearTimer = setTimeout(() => {
        el.textContent = "";
        el.classList.remove("gc-status-applied", "gc-status-fading");
      }, 600);
      _statusTimers.set(id, clearTimer);
    }, STATUS_FADE_MS);
    _statusTimers.set(id, timer);
  } else if (mode === "error") {
    el.classList.add("gc-status-error");
  } else if (mode === "pending") {
    el.classList.add("gc-status-pending");
  }
}

async function postJson(url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let data = {};
  try { data = await resp.json(); } catch (_) { /* empty body */ }
  if (!resp.ok) {
    const err = data.error || data.detail || ("HTTP " + resp.status);
    throw new Error(err);
  }
  return data;
}

// ---- Model dropdown --------------------------------------------------------

async function loadModelOptions(activeName) {
  const sel = $("gc-model-select");
  try {
    const data = await (await fetch("/api/models", { cache: "no-store" })).json();
    const models = data.models || [];
    if (!data.ollama_reachable) {
      sel.innerHTML = '<option value="">Ollama unreachable</option>';
      sel.disabled = true;
      showStatus("gc-model-status", "Ollama unreachable", "error");
      return;
    }
    if (!models.length) {
      sel.innerHTML = '<option value="">No installed models</option>';
      sel.disabled = true;
      return;
    }
    sel.disabled = false;
    const active = activeName || data.active;
    sel.innerHTML = models.map((m) => {
      const name = escapeHtml(m.name);
      const isActive = m.name === active;
      return `<option value="${name}"${isActive ? " selected" : ""}>${name}</option>`;
    }).join("");
  } catch (err) {
    sel.innerHTML = '<option value="">load failed</option>';
    showStatus("gc-model-status", "Failed to load models", "error");
  }
}

async function onModelChange() {
  const sel = $("gc-model-select");
  const model = sel.value;
  if (!model) return;
  showStatus("gc-model-status", "Applying...", "pending");
  try {
    await postJson("/api/models/active", { model });
    showStatus("gc-model-status", "Applied", "applied");
    await refreshGroundingSummary();
  } catch (err) {
    showStatus("gc-model-status", "Failed: " + err.message, "error");
  }
}

// ---- Corpus dropdown -------------------------------------------------------

async function loadCorpusOptions(activeName) {
  const sel = $("gc-corpus-select");
  try {
    const data = await (await fetch("/api/rag/instances", { cache: "no-store" })).json();
    const instances = data.instances || [];
    const active = activeName || data.active;
    const noneOption = data.no_rag_option || { name: "none", display_name: "No RAG" };
    const opts = [];
    instances.forEach((i) => {
      const name = escapeHtml(i.name);
      const disp = escapeHtml(i.display_name || i.name);
      const status = i.status && i.status !== "published" ? ` [${escapeHtml(i.status)}]` : "";
      opts.push(`<option value="${name}"${i.name === active ? " selected" : ""}>${disp}${status}</option>`);
    });
    const noneLabel = escapeHtml(noneOption.display_name || "No RAG");
    opts.push(`<option value="none"${active === "none" ? " selected" : ""}>${noneLabel}</option>`);
    sel.innerHTML = opts.join("");
    sel.disabled = false;
  } catch (err) {
    sel.innerHTML = '<option value="">load failed</option>';
    showStatus("gc-corpus-status", "Failed to load corpora", "error");
  }
}

async function onCorpusChange() {
  const sel = $("gc-corpus-select");
  const name = sel.value;
  if (!name) return;
  showStatus("gc-corpus-status", "Applying...", "pending");
  try {
    await postJson("/api/rag/active", { name });
    showStatus("gc-corpus-status", "Applied", "applied");
    await refreshGroundingSummary();
  } catch (err) {
    showStatus("gc-corpus-status", "Failed: " + err.message, "error");
  }
}

// ---- System prompt dropdown ------------------------------------------------
//
// Options: "Built-in default" (POST empty text => reset) plus any saved
// presets returned by /api/system-prompts. The CURRENT active prompt may not
// match either of those exactly (user might have set a custom prompt on the
// Settings page); we show that as a non-selectable "Custom override" entry so
// the user can see what's live without clobbering it on first selection.

let _systemPresets = [];

async function loadSystemPromptOptions(activeText, activeSource) {
  const sel = $("gc-system-prompt-select");
  try {
    const [presetsResp, activeResp] = await Promise.all([
      fetch("/api/system-prompts", { cache: "no-store" }).then((r) => r.json()),
      fetch("/api/system-prompt", { cache: "no-store" }).then((r) => r.json()),
    ]);
    _systemPresets = presetsResp.presets || [];
    const active = activeText !== undefined ? activeText : (activeResp.text || "");
    const source = activeSource || activeResp.source || "default";
    const matchedPreset = _systemPresets.find((p) => (p.text || "") === active);

    const opts = [];
    opts.push(`<option value="__default__"${source === "default" ? " selected" : ""}>Built-in default</option>`);
    _systemPresets.forEach((p) => {
      const n = escapeHtml(p.name);
      const isActive = matchedPreset && matchedPreset.name === p.name && source !== "default";
      opts.push(`<option value="preset:${n}"${isActive ? " selected" : ""}>${n}</option>`);
    });
    if (source !== "default" && !matchedPreset) {
      opts.push(`<option value="__custom__" selected disabled>Custom override</option>`);
    }
    sel.innerHTML = opts.join("");
    sel.disabled = false;
  } catch (err) {
    sel.innerHTML = '<option value="">load failed</option>';
    showStatus("gc-system-prompt-status", "Failed to load presets", "error");
  }
}

async function onSystemPromptChange() {
  const sel = $("gc-system-prompt-select");
  const value = sel.value;
  if (!value || value === "__custom__") return;
  showStatus("gc-system-prompt-status", "Applying...", "pending");
  try {
    let text = "";
    if (value === "__default__") {
      text = ""; // empty => reset to built-in default
    } else if (value.startsWith("preset:")) {
      const name = value.slice("preset:".length);
      const preset = _systemPresets.find((p) => p.name === name);
      if (!preset) throw new Error("preset not found");
      text = preset.text || "";
    }
    await postJson("/api/system-prompt", { text });
    showStatus("gc-system-prompt-status", "Applied", "applied");
    await refreshGroundingSummary();
  } catch (err) {
    showStatus("gc-system-prompt-status", "Failed: " + err.message, "error");
  }
}

// ---- Retrieval feature dropdowns ------------------------------------------

const INT_FEATURE_OPTIONS = {
  top_k: [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50],
  fetch_k: [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200],
};

function populateIntDropdown(id, value) {
  const sel = $(id);
  const featureKey = sel.dataset.feature;
  const choices = new Set(INT_FEATURE_OPTIONS[featureKey] || []);
  const v = Number(value);
  if (Number.isFinite(v) && v > 0) choices.add(v);
  const sorted = [...choices].sort((a, b) => a - b);
  sel.innerHTML = sorted.map((n) =>
    `<option value="${n}"${n === v ? " selected" : ""}>${n}</option>`).join("");
}

async function loadRetrievalControls(config) {
  let c = config;
  if (!c) {
    try {
      const data = await (await fetch("/api/retrieval", { cache: "no-store" })).json();
      c = data.config || {};
    } catch (err) {
      showStatus("gc-feat-hybrid-status", "Failed to load retrieval", "error");
      return;
    }
  }
  $("gc-feat-hybrid").value = c.hybrid ? "true" : "false";
  $("gc-feat-rerank").value = c.rerank ? "true" : "false";
  $("gc-feat-rewrite_mode").value = c.rewrite_mode || "off";
  $("gc-feat-grounding_reinjection").value = c.grounding_reinjection ? "true" : "false";
  populateIntDropdown("gc-feat-top_k", c.top_k ?? 3);
  populateIntDropdown("gc-feat-fetch_k", c.fetch_k ?? 20);
}

function _coerceFeatureValue(key, raw) {
  if (key === "top_k" || key === "fetch_k") return Number(raw);
  if (key === "rewrite_mode") return raw;
  return raw === "true"; // bool features
}

async function onRetrievalFeatureChange(event) {
  const sel = event.target;
  const key = sel.dataset.feature;
  if (!key) return;
  const statusId = `gc-feat-${key}-status`;
  showStatus(statusId, "Applying...", "pending");
  try {
    await postJson("/api/retrieval", { key, value: _coerceFeatureValue(key, sel.value) });
    showStatus(statusId, "Applied", "applied");
    await refreshGroundingSummary();
  } catch (err) {
    showStatus(statusId, "Failed: " + err.message, "error");
  }
}

// ---- Grounding summary (chunk count + live state refresh) ------------------

async function refreshGroundingSummary() {
  try {
    const resp = await fetch("/api/grounding-summary", { cache: "no-store" });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "load_failed");
    renderGroundingSummary(data);
  } catch (_err) {
    // Status lines already handle visible errors per-field; the chunk count
    // just shows a dash. Don't blow up the rest of the UI.
    const chunk = $("gs-chunk-count");
    if (chunk) chunk.textContent = "-";
  }
}

function renderGroundingSummary(summary) {
  const chunkCount = summary?.chunk_count;
  const chunkEl = $("gs-chunk-count");
  if (chunkEl) chunkEl.textContent = Number.isFinite(chunkCount) ? String(chunkCount) : "-";
  // Sync dropdowns to whatever the backend now reports (in case another tab
  // changed something, or the backend coerced the value).
  if (summary?.model_name) {
    const sel = $("gc-model-select");
    if (sel && [...sel.options].some((o) => o.value === summary.model_name)) {
      sel.value = summary.model_name;
    }
  }
  if (summary?.corpus_name) {
    const sel = $("gc-corpus-select");
    if (sel && [...sel.options].some((o) => o.value === summary.corpus_name)) {
      sel.value = summary.corpus_name;
    }
  }
}

// Initial composite load: corpus, model, system prompt, retrieval features.
async function loadGroundingControls() {
  await Promise.all([
    loadModelOptions(),
    loadCorpusOptions(),
    loadSystemPromptOptions(),
    loadRetrievalControls(),
    refreshGroundingSummary(),
  ]);
}

// Show which retrieval frameworks were active for this answer (A/B at a glance).
function renderRetrievalBadges(r) {
  const el = $("retrieval-badges");
  if (!el) return;
  if (!r) { el.innerHTML = ""; return; }
  const labels = [];
  if (r.rag) {
    if (r.hybrid) labels.push("Hybrid");
    if (r.rerank) labels.push("Rerank");
    if (r.rewrite_mode && r.rewrite_mode !== "off") {
      labels.push(r.rewrite_mode === "hyde" ? "HyDE" : "Rewrite");
    }
    labels.push("top-k " + Number(r.top_k || 0));
  }
  if (r.grounding_reinjection) labels.push("Grounding");
  el.innerHTML = labels
    .map((b) => `<span class="badge badge-neutral">${b}</span>`).join(" ");
}

async function runPrompt() {
  const prompt = $("prompt").value.trim();
  if (!prompt) { $("status").textContent = "Enter a prompt first."; return; }
  const runBtn = $("run");
  const judge = $("judge").checked;
  runBtn.disabled = true;
  $("status").textContent = judge
    ? "Running through the harness (retrieve → answer → judge)..."
    : "Running through the harness (retrieve → answer)...";
  $("mark-message").textContent = "";
  try {
    const resp = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        rag: $("rag").checked,
        reground: $("reground").checked,
        think: $("think").checked,
        judge,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || data.error || "harness_failed");
    renderResult(data);
    $("status").textContent = "";
  } catch (err) {
    $("status").textContent = "Failed: " + err + " — is Ollama running and a model pulled?";
  } finally {
    runBtn.disabled = false;
  }
}

function renderResult(data) {
  $("result-panel").classList.remove("hidden");
  lastCallId = data.call_id || null;

  if (data.reasoning) {
    $("reasoning-block").classList.remove("hidden");
    $("reasoning").textContent = data.reasoning;
  } else {
    $("reasoning-block").classList.add("hidden");
  }
  $("answer").textContent = data.answer || "(empty)";

  const verdict = data.verdict;
  lastVerdict = verdict || null;
  const vb = $("verdict-badge");
  const acceptBtn = $("accept-judge");
  if (verdict) {
    const b = ratingBadge(verdict.rating);
    vb.className = "badge " + b.cls;
    vb.textContent = "judge: " + b.text;
    $("verdict-reason").textContent = verdict.reason || "";
    // Only offer the accept button when there's a real verdict reason to
    // ground on — judge-could-not-run failures have no usable reason text.
    const tags = Array.isArray(verdict.failure_tags) ? verdict.failure_tags : [];
    const judgeFailed = tags.includes("judge_failed") ||
                        tags.includes("judge_timeout") ||
                        tags.includes("judge_unparseable");
    if (verdict.reason && !judgeFailed) {
      acceptBtn.hidden = false;
      acceptBtn.disabled = false;
    } else {
      acceptBtn.hidden = true;
    }
  } else {
    vb.className = "badge badge-neutral";
    vb.textContent = "judge: skipped";
    $("verdict-reason").textContent = "";
    acceptBtn.hidden = true;
  }

  const ragB = $("rag-badge");
  ragB.className = "badge " + (data.grounded ? "badge-good" : "badge-neutral");
  ragB.textContent = data.grounded ? "RAG: grounded" : "RAG: off";

  const rgB = $("reground-badge");
  if (data.reground_applied) {
    rgB.className = "badge badge-good";
    rgB.textContent = "Reground: applied";
    rgB.classList.remove("hidden");
  } else {
    rgB.classList.add("hidden");
  }

  renderRetrievalBadges(data.retrieval);
  renderGroundingSummary(data.grounding_summary);
}

async function markForEvaluation() {
  if (!lastCallId) return;
  const btn = $("mark");
  btn.disabled = true;
  $("mark-message").textContent = "Marking...";
  try {
    const resp = await fetch("/api/activity/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ call_id: lastCallId }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "queue_failed");
    $("mark-message").innerHTML = 'Marked. Open the <a href="/evaluation">Evaluation queue</a> to score it.';
  } catch (err) {
    $("mark-message").textContent = "Failed: " + err;
    btn.disabled = false;
  }
}

// Map judge rating -> score the same way the human scorer would. The grounding
// store re-derives rating from score, so this round-trip preserves the judge's
// classification while letting submit_verdict's reason-required guard pass.
function _ratingToScore(rating) {
  if (rating === "good") return 8;
  if (rating === "partial") return 5;
  return 2;
}

async function acceptJudgeEvaluation() {
  if (!lastCallId || !lastVerdict || !lastVerdict.reason) return;
  const btn = $("accept-judge");
  btn.disabled = true;
  $("mark-message").textContent = "Recording judge verdict...";
  try {
    const resp = await fetch("/api/evaluation/verdict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        call_id: lastCallId,
        score: _ratingToScore(lastVerdict.rating),
        correction: lastVerdict.reason,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "verdict_failed");
    $("mark-message").textContent = "Judge verdict accepted and added to grounding.";
    btn.hidden = true;
  } catch (err) {
    $("mark-message").textContent = "Failed: " + err;
    btn.disabled = false;
  }
}

// ---- wire-up --------------------------------------------------------------

$("run").addEventListener("click", runPrompt);
$("mark").addEventListener("click", markForEvaluation);
$("accept-judge").addEventListener("click", acceptJudgeEvaluation);
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runPrompt();
});

$("gc-model-select").addEventListener("change", onModelChange);
$("gc-corpus-select").addEventListener("change", onCorpusChange);
$("gc-system-prompt-select").addEventListener("change", onSystemPromptChange);
document.querySelectorAll(".gc-feature select[data-feature]").forEach((sel) => {
  sel.addEventListener("change", onRetrievalFeatureChange);
});

loadGroundingControls();
