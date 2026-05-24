const REFRESH_MS = 5000;

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
  const label = (status && status.label) || "Not queued";
  const score = status && status.score != null ? ` (${status.score})` : "";
  return `<span class="badge ${cls}">${escapeHtml(label)}${escapeHtml(score)}</span>`;
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
          <div class="title">${escapeHtml(it.summary)}</div>
          <div class="meta">${escapeHtml(fmtTime(it.timestamp))} · ${escapeHtml(it.endpoint || "")} · ${escapeHtml(it.model || "")} · ${escapeHtml(it.latency_ms || 0)}ms</div>
        </div>
        <div>${statusBadge(it.status)}</div>
      </div>
      <div class="eval-actions">
        <button class="secondary mark-btn"${(it.status && it.status.status !== "none") ? " disabled" : ""}>Mark for evaluation</button>
        <span class="mark-msg muted"></span>
      </div>
    </article>
  `).join("");
}

async function refresh() {
  const resp = await fetch("/api/activity", { cache: "no-store" });
  const data = await resp.json();
  render(data.items || []);
  document.getElementById("refresh-label").textContent = `Updated ${fmtTime(data.generated_at)} · ${data.count} calls`;
}

document.getElementById("activity").addEventListener("click", async (e) => {
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

async function loop() {
  try { await refresh(); } catch (e) {
    document.getElementById("refresh-label").textContent = "Refresh failed: " + e;
  } finally { setTimeout(loop, REFRESH_MS); }
}
loop();
