const REFRESH_MS = 6000;
let reasonThreshold = 7;
const drafts = new Map();

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
function draftFor(id) {
  if (!drafts.has(id)) drafts.set(id, { score: "10", correction: "" });
  return drafts.get(id);
}

function render(items) {
  const root = document.getElementById("queue");
  if (!items.length) {
    root.innerHTML = '<p class="muted">Queue is clear. Mark an answer on the Simulate or Activity page.</p>';
    return;
  }
  root.innerHTML = items.map((it) => {
    const d = draftFor(it.call_id);
    const needs = Number(d.score || 0) <= reasonThreshold;
    return `
      <article class="panel eval-card" data-call-id="${escapeHtml(it.call_id)}">
        <div class="top">
          <div>
            <div class="title">${escapeHtml(it.summary)}</div>
            <div class="meta">Queued ${escapeHtml(fmtTime(it.queued_at))} · ${escapeHtml(it.endpoint || "")}</div>
          </div>
          <span class="badge badge-partial">Pending</span>
        </div>
        <div class="grid-2" style="margin-top:12px">
          <div class="block"><h3>Prompt</h3><pre>${escapeHtml(it.prompt || "")}</pre></div>
          <div class="block"><h3>Response</h3><pre>${escapeHtml(it.response || "")}</pre></div>
        </div>
        <div class="controls" style="margin-top:14px">
          <label class="field" style="margin:0">
            <span>Score (1-10)</span>
            <select class="score">
              ${Array.from({ length: 10 }, (_, i) => {
                const s = String(i + 1);
                return `<option value="${s}"${d.score === s ? " selected" : ""}>${s}</option>`;
              }).join("")}
            </select>
          </label>
        </div>
        <label class="field correction-field${needs ? "" : " hidden"}">
          <span>Correction (what the answer should have been — fed back into grounding)</span>
          <textarea class="correction" rows="4" placeholder="The correct answer / guidance...">${escapeHtml(d.correction)}</textarea>
        </label>
        <div class="eval-actions">
          <button class="submit-btn">Submit + Reground</button>
          <span class="muted">Scores ${reasonThreshold} and below require a correction.</span>
        </div>
        <p class="eval-message muted"></p>
      </article>`;
  }).join("");
}

function captureDrafts() {
  for (const card of document.querySelectorAll(".eval-card")) {
    const id = card.dataset.callId;
    drafts.set(id, {
      score: card.querySelector(".score")?.value || "10",
      correction: card.querySelector(".correction")?.value || "",
    });
  }
}

async function refresh() {
  captureDrafts();
  const resp = await fetch("/api/evaluation/queue", { cache: "no-store" });
  const data = await resp.json();
  reasonThreshold = data.reason_threshold ?? 7;
  render(data.items || []);
  document.getElementById("queue-label").textContent =
    `${data.count} pending · updated ${fmtTime(data.generated_at)}`;
}

document.getElementById("queue").addEventListener("input", (e) => {
  const card = e.target.closest(".eval-card");
  if (!card) return;
  const id = card.dataset.callId;
  const score = card.querySelector(".score")?.value || "10";
  const correction = card.querySelector(".correction")?.value || "";
  drafts.set(id, { score, correction });
  card.querySelector(".correction-field")?.classList.toggle("hidden", Number(score) > reasonThreshold);
});

document.getElementById("queue").addEventListener("click", async (e) => {
  const btn = e.target.closest(".submit-btn");
  if (!btn) return;
  const card = btn.closest(".eval-card");
  const id = card.dataset.callId;
  const score = Number(card.querySelector(".score")?.value || 0);
  const correction = card.querySelector(".correction")?.value || "";
  const msg = card.querySelector(".eval-message");
  if (score <= reasonThreshold && !correction.trim()) {
    msg.textContent = `A correction is required for scores of ${reasonThreshold} or below.`;
    return;
  }
  btn.disabled = true;
  btn.textContent = "Submitting...";
  try {
    const resp = await fetch("/api/evaluation/verdict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ call_id: id, score, correction }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "failed");
    drafts.delete(id);
    card.classList.add("flash-improved");
    msg.textContent = "Regrounded. Re-run the same prompt with Reground on to verify it improved.";
    setTimeout(refresh, 900);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Submit + Reground";
    msg.textContent = "Failed: " + err;
  }
});

async function loop() {
  try { await refresh(); } catch (e) {
    document.getElementById("queue-label").textContent = "Refresh failed: " + e;
  } finally { setTimeout(loop, REFRESH_MS); }
}
loop();
