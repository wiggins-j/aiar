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
    if (!data.ollama_reachable) {
      sel.innerHTML = '<option value="">Ollama unreachable</option>';
      sel.disabled = true;
      $("apply-model").disabled = true;
      msg.textContent = "Ollama is unreachable — model switching is disabled.";
      return;
    }
    if (!models.length) {
      sel.innerHTML = '<option value="">No matching models installed</option>';
      sel.disabled = true;
      $("apply-model").disabled = true;
      msg.textContent = "Ollama is reachable, but no installed models match the current filter.";
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
    const noneOption = data.no_rag_option || { name: "none", display_name: "No RAG" };
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
    const noneLabel = escapeHtml(noneOption.display_name || "No RAG");
    options.push(`<option value="none"${noneActive ? " selected" : ""}>${noneLabel}${noneActive ? " (active)" : ""}</option>`);
    sel.innerHTML = options.join("");
    updateDeleteState();
    updateActiveSummary(null, data.active_display_name || active);
  } catch (err) {
    sel.innerHTML = '<option value="">load failed</option>';
    msg.textContent = "Failed to load RAG instances: " + err;
  }
}

// The default instance and the "No RAG" sentinel are not deletable.
function updateDeleteState() {
  const name = $("settings-rag-select").value;
  $("delete-rag").disabled = !name || name === "none" || name === "default";
}

async function deleteRag() {
  const sel = $("settings-rag-select");
  const btn = $("delete-rag");
  const msg = $("rag-message");
  const name = sel.value;
  if (!name || name === "none" || name === "default") return;
  const label = sel.options[sel.selectedIndex]?.text || name;
  if (!window.confirm(`Delete RAG instance "${label}"? This removes its corpus permanently and cannot be undone.`)) return;
  btn.disabled = true;
  btn.textContent = "Deleting...";
  try {
    const resp = await fetch("/api/rag/delete", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "delete_failed");
    msg.textContent = `Deleted. Active RAG is now ${escapeHtml(data.active_display_name || data.active)}.`;
    await loadRag();
  } catch (err) {
    msg.textContent = "Failed: " + err;
  } finally {
    btn.textContent = "Delete RAG";
    updateDeleteState();
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
    msg.textContent = `Applied. Active RAG is now ${escapeHtml(data.active_display_name || data.active)}.`;
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

// ---- System-prompt presets (quick-save / load / delete) --------------------

let _presets = [];

function updatePresetDeleteState() {
  $("delete-preset").disabled = !$("system-preset-select").value;
}

async function loadSystemPresets() {
  const sel = $("system-preset-select");
  try {
    const resp = await fetch("/api/system-prompts", { cache: "no-store" });
    const data = await resp.json();
    _presets = data.presets || [];
    const limit = data.limit || 5;
    const opts = [`<option value="">— saved prompts (${_presets.length}/${limit}) —</option>`];
    _presets.forEach((p) => {
      const n = escapeHtml(p.name);
      opts.push(`<option value="${n}">${n}</option>`);
    });
    sel.innerHTML = opts.join("");
  } catch (err) {
    sel.innerHTML = '<option value="">load failed</option>';
  }
  updatePresetDeleteState();
}

function onPresetChange() {
  const name = $("system-preset-select").value;
  updatePresetDeleteState();
  if (!name) return;
  const preset = _presets.find((p) => p.name === name);
  if (!preset) return;
  $("settings-system-prompt").value = preset.text || "";
  $("system-preset-name").value = preset.name;
  $("system-preset-message").textContent = `Loaded "${preset.name}" — click Save to apply it.`;
}

async function saveSystemPreset() {
  const btn = $("save-preset");
  const msg = $("system-preset-message");
  const name = $("system-preset-name").value.trim();
  const text = $("settings-system-prompt").value;
  if (!name) { msg.textContent = "Enter a preset name first."; return; }
  if (!text.trim()) { msg.textContent = "The system prompt is empty — nothing to save."; return; }
  btn.disabled = true;
  btn.textContent = "Saving...";
  try {
    const resp = await fetch("/api/system-prompts/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, text }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      const why = data.error === "preset_limit" ? "limit reached (max 5) — delete one first"
        : data.error === "name_too_long" ? "name too long (max 60 chars)"
        : data.error || "save_failed";
      throw new Error(why);
    }
    msg.textContent = `Saved preset "${escapeHtml(name)}".`;
    await loadSystemPresets();
    $("system-preset-select").value = name;
    updatePresetDeleteState();
  } catch (err) {
    msg.textContent = "Failed: " + err;
  } finally {
    btn.disabled = false;
    btn.textContent = "Save preset";
  }
}

async function deleteSystemPreset() {
  const btn = $("delete-preset");
  const msg = $("system-preset-message");
  const name = $("system-preset-select").value;
  if (!name) return;
  if (!window.confirm(`Delete saved prompt "${name}"?`)) return;
  btn.disabled = true;
  btn.textContent = "Deleting...";
  try {
    const resp = await fetch("/api/system-prompts/delete", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "delete_failed");
    msg.textContent = `Deleted preset "${escapeHtml(name)}".`;
    await loadSystemPresets();
  } catch (err) {
    msg.textContent = "Failed: " + err;
  } finally {
    btn.textContent = "Delete preset";
    updatePresetDeleteState();
  }
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
$("delete-rag").addEventListener("click", deleteRag);
$("settings-rag-select").addEventListener("change", updateDeleteState);
$("save-system").addEventListener("click", saveSystemPrompt);
$("reset-system").addEventListener("click", resetSystemPrompt);
$("save-preset").addEventListener("click", saveSystemPreset);
$("delete-preset").addEventListener("click", deleteSystemPreset);
$("system-preset-select").addEventListener("change", onPresetChange);

loadModels();
loadRag();
loadSystemPrompt();
loadSystemPresets();
