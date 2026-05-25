const REFRESH_MS = 5000;
const expandedDetails = new Map();
let totalCount = 0;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function fmtTime(raw) {
  if (!raw) return "unknown";
  const d = new Date(raw);
  return Number.isNaN(d.getTime()) ? raw : d.toLocaleString();
}

function statusBadge(status) {
  const s = (status && status.status) || "none";
  const cls = s === "complete" ? "badge-good" : s === "pending" ? "badge-partial" : "badge-neutral";
  const label = (status && status.label) || "Not Queued";
  const score = status && status.score != null ? ` (${status.score})` : "";
  return `<span class="badge ${cls}">${escapeHtml(label)}${escapeHtml(score)}</span>`;
}

function metaLine(it) {
  const parts = [fmtTime(it.timestamp)];
  if (it.rag_state) parts.push(it.rag_state);
  if (it.model) parts.push(it.model);
  parts.push(`${escapeHtml(it.latency_ms || 0)}ms`);
  return parts.map((p) => escapeHtml(p)).join(" · ");
}

function render(items) {
  const root = document.getElementById("activity");
  if (!items.length) {
    root.innerHTML = '<p class="muted">No calls logged yet. Run a prompt on the Simulate page.</p>';
    return;
  }
  root.innerHTML = items.map((it) => `
    <article class="list-item" data-call-id="${escapeHtml(it.call_id)}">
      <div class="top">
        <div>
          <div class="activity-preview">
            <div><strong>Prompt:</strong> ${escapeHtml(it.prompt_preview || "")}</div>
            <div><strong>Response:</strong> ${escapeHtml(it.response_preview || "")}</div>
          </div>
          <div class="meta">${metaLine(it)}</div>
        </div>
        <div>${statusBadge(it.status)}</div>
      </div>
      <div class="eval-actions">
        <button class="secondary more-btn">${expandedDetails.has(it.call_id) ? "Less" : "More"}</button>
      </div>
      ${expandedDetails.has(it.call_id) ? renderDetails(expandedDetails.get(it.call_id)) : ""}
      <div class="eval-actions">
        <button class="secondary mark-btn"${(it.status && it.status.status !== "none") ? " disabled" : ""}>Mark for Evaluation</button>
        <span class="mark-msg muted"></span>
      </div>
    </article>
  `).join("");
}

function renderDetails(detail) {
  return `
    <div class="activity-detail">
      <div class="grid-2" style="margin-top:12px">
        <div class="block"><h3>Prompt</h3><pre>${escapeHtml(detail.prompt || "")}</pre></div>
        <div class="block"><h3>Response</h3><pre>${escapeHtml(detail.response || "")}</pre></div>
      </div>
    </div>`;
}

async function refresh() {
  const resp = await fetch("/api/activity", { cache: "no-store" });
  const data = await resp.json();
  render(data.items || []);
  totalCount = Number(data.count || 0);
  document.getElementById("refresh-label").textContent = `Updated ${fmtTime(data.generated_at)} · ${data.count} calls`;
  document.getElementById("clear-activity").disabled = totalCount === 0;
}

async function clearActivity() {
  const btn = document.getElementById("clear-activity");
  const msg = document.getElementById("clear-activity-message");
  if (totalCount === 0) { msg.textContent = "Nothing to clear."; return; }
  if (!window.confirm(`Clear all ${totalCount} logged call(s)? This cannot be undone.`)) return;
  btn.disabled = true;
  msg.textContent = "Clearing...";
  try {
    const resp = await fetch("/api/activity/clear", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "failed");
    expandedDetails.clear();
    msg.textContent = `Cleared ${data.cleared || 0} call(s).`;
    await refresh();
  } catch (err) {
    msg.textContent = "Failed: " + err;
    btn.disabled = false;
  }
}

document.getElementById("activity").addEventListener("click", async (e) => {
  const more = e.target.closest(".more-btn");
  if (more) {
    const card = more.closest(".list-item");
    const callId = card.dataset.callId;
    if (expandedDetails.has(callId)) {
      expandedDetails.delete(callId);
      await refresh();
      return;
    }
    more.disabled = true;
    try {
      const resp = await fetch(`/api/activity/detail?call_id=${encodeURIComponent(callId)}`, { cache: "no-store" });
      const data = await resp.json();
      if (!resp.ok || !data.found) throw new Error(data.error || "detail_failed");
      expandedDetails.set(callId, data);
      await refresh();
    } catch (err) {
      more.disabled = false;
      const msg = card.querySelector(".mark-msg");
      msg.textContent = "Failed to load details: " + err;
    }
    return;
  }

  const btn = e.target.closest(".mark-btn");
  if (!btn) return;
  const card = btn.closest(".list-item");
  const callId = card.dataset.callId;
  btn.disabled = true;
  const msg = card.querySelector(".mark-msg");
  msg.textContent = "Marking...";
  try {
    const resp = await fetch("/api/activity/evaluate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ call_id: callId }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "failed");
    msg.innerHTML = 'Marked — see the <a href="/evaluation">Evaluation queue</a>.';
  } catch (err) {
    msg.textContent = "Failed: " + err;
    btn.disabled = false;
  }
});

document.getElementById("clear-activity").addEventListener("click", clearActivity);

async function loop() {
  try { await refresh(); } catch (e) {
    document.getElementById("refresh-label").textContent = "Refresh failed: " + e;
  } finally { setTimeout(loop, REFRESH_MS); }
}
loop();
