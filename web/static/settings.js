// AIAR Settings page — model switch, RAG instance switch (incl. "No RAG"), and
// the harness answer system-prompt editor. Same vanilla idiom as the rest of
// the AIAR static GUI: escapeHtml, fetch({cache:"no-store"}), build POST -> ack
// -> re-fetch and re-render. No build step, no dependencies.

function $(id) { return document.getElementById(id); }

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

// ---- Model card ------------------------------------------------------------

async function loadModels() {
  const sel = $("settings-model-select");
  const msg = $("model-message");
  try {
    const resp = await fetch("/api/models", { cache: "no-store" });
    const data = await resp.json();
    const models = data.models || [];
    if (!data.ollama_reachable || !models.length) {
      sel.innerHTML = '<option value="">Ollama unreachable</option>';
      sel.disabled = true;
      $("apply-model").disabled = true;
      msg.textContent = "Ollama is unreachable — model switching is disabled.";
      return;
    }
    sel.disabled = false;
    $("apply-model").disabled = false;
    sel.innerHTML = models.map((m) => {
      const name = escapeHtml(m.name);
      const label = name + (m.active ? " (active)" : "");
      return `<option value="${name}"${m.active ? " selected" : ""}>${label}</option>`;
    }).join("");
    updateActiveSummary(data.active, null);
  } catch (err) {
    sel.innerHTML = '<option value="">load failed</option>';
    msg.textContent = "Failed to load models: " + err;
  }
}

async function applyModel() {
  const btn = $("apply-model");
  const msg = $("model-message");
  const model = $("settings-model-select").value;
  if (!model) return;
  btn.disabled = true;
  btn.textContent = "Applying...";
  try {
    const resp = await fetch("/api/models/active", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "apply_failed");
    msg.textContent = `Applied. Active model is now ${escapeHtml(data.active)} (was ${escapeHtml(data.previous)}).`;
    await loadModels();
  } catch (err) {
    msg.textContent = "Failed: " + err;
  } finally {
    btn.disabled = false;
    btn.textContent = "Apply model";
  }
}

// ---- RAG card --------------------------------------------------------------

async function loadRag() {
  const sel = $("settings-rag-select");
  const msg = $("rag-message");
  try {
    const resp = await fetch("/api/rag/instances", { cache: "no-store" });
    const data = await resp.json();
    const instances = data.instances || [];
    const active = data.active;
    // First-class "No RAG" option (value "none") + each instance.
    const options = [];
    instances.forEach((i) => {
      const name = escapeHtml(i.name);
      const disp = escapeHtml(i.display_name || i.name);
      const status = i.status && i.status !== "published" ? ` [${escapeHtml(i.status)}]` : "";
      const isActive = i.name === active;
      const label = `${disp}${status}` + (isActive ? " (active)" : "");
      options.push(`<option value="${name}"${isActive ? " selected" : ""}>${label}</option>`);
    });
    const noneActive = active === "none";
    options.push(`<option value="none"${noneActive ? " selected" : ""}>No RAG${noneActive ? " (active)" : ""}</option>`);
    sel.innerHTML = options.join("");
    updateActiveSummary(null, active);
  } catch (err) {
    sel.innerHTML = '<option value="">load failed</option>';
    msg.textContent = "Failed to load RAG instances: " + err;
  }
}

async function applyRag() {
  const btn = $("apply-rag");
  const msg = $("rag-message");
  const name = $("settings-rag-select").value;
  if (!name) return;
  btn.disabled = true;
  btn.textContent = "Applying...";
  try {
    const resp = await fetch("/api/rag/active", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "apply_failed");
    msg.textContent = `Applied. Active RAG is now ${escapeHtml(data.active)}.`;
    await loadRag();
  } catch (err) {
    msg.textContent = "Failed: " + err;
  } finally {
    btn.disabled = false;
    btn.textContent = "Apply RAG";
  }
}

// ---- System-prompt card ----------------------------------------------------

async function loadSystemPrompt() {
  const ta = $("settings-system-prompt");
  const msg = $("system-message");
  try {
    const resp = await fetch("/api/system-prompt", { cache: "no-store" });
    const data = await resp.json();
    ta.value = data.text || "";
    msg.textContent = data.source === "active"
      ? "Using a custom system prompt."
      : "Using the built-in default system prompt.";
  } catch (err) {
    msg.textContent = "Failed to load system prompt: " + err;
  }
}

async function postSystemPrompt(text) {
  const msg = $("system-message");
  try {
    const resp = await fetch("/api/system-prompt", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "save_failed");
    await loadSystemPrompt();
    msg.textContent = data.source === "active"
      ? "Saved custom system prompt."
      : "Reset to the built-in default.";
  } catch (err) {
    msg.textContent = "Failed: " + err;
  }
}

async function saveSystemPrompt() {
  const btn = $("save-system");
  btn.disabled = true;
  btn.textContent = "Saving...";
  await postSystemPrompt($("settings-system-prompt").value);
  btn.disabled = false;
  btn.textContent = "Save";
}

async function resetSystemPrompt() {
  const btn = $("reset-system");
  btn.disabled = true;
  await postSystemPrompt(""); // empty -> reset to default
  btn.disabled = false;
}

// ---- shared --------------------------------------------------------------

let _activeModel = null;
let _activeRag = null;
function updateActiveSummary(model, rag) {
  if (model !== null) _activeModel = model;
  if (rag !== null) _activeRag = rag;
  const parts = [];
  if (_activeModel) parts.push(`Active model: ${escapeHtml(_activeModel)}`);
  if (_activeRag) parts.push(`Active RAG: ${escapeHtml(_activeRag)}`);
  $("active-summary").textContent = parts.join(" · ");
}

$("apply-model").addEventListener("click", applyModel);
$("apply-rag").addEventListener("click", applyRag);
$("save-system").addEventListener("click", saveSystemPrompt);
$("reset-system").addEventListener("click", resetSystemPrompt);

loadModels();
loadRag();
loadSystemPrompt();
