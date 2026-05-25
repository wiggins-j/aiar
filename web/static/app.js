let lastCallId = null;

function $(id) { return document.getElementById(id); }

function ratingBadge(rating) {
  const cls = rating === "good" ? "badge-good"
    : rating === "partial" ? "badge-partial"
    : "badge-bad";
  return { cls, text: rating || "?" };
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
  const vb = $("verdict-badge");
  if (verdict) {
    const b = ratingBadge(verdict.rating);
    vb.className = "badge " + b.cls;
    vb.textContent = "judge: " + b.text;
    $("verdict-reason").textContent = verdict.reason || "";
  } else {
    vb.className = "badge badge-neutral";
    vb.textContent = "judge: skipped";
    $("verdict-reason").textContent = "";
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

  $("latency").textContent = (data.latency_ms || 0) + " ms · call " + (lastCallId || "n/a");
  $("mark").disabled = !lastCallId;
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

$("run").addEventListener("click", runPrompt);
$("mark").addEventListener("click", markForEvaluation);
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runPrompt();
});
