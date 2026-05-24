# Template — "Build my AIAR RAG Collection Brief"

Copy everything in the fenced block below and paste it into any capable AI agent
(Claude, ChatGPT/Codex, Gemini, Cursor, Copilot Chat, …). It will **interview you**
and then write a complete **Collection Brief** `.md` — the kind of file you hand to
a (possibly different) AI agent so it builds your AIAR corpus for you. A full worked
example of the output is [`tesla-manual-expert-collection-brief.md`](tesla-manual-expert-collection-brief.md).

````text
You are helping me author a "RAG Collection Brief" for AIAR, an open-source,
model-agnostic RAG framework (https://github.com/ — local; ingest with
`python -m aiar.rag.ingest <folder> --instance <name>`). The brief is a single
Markdown file that a LATER AI agent (with web + file tools) will follow to FIND,
FETCH, and NORMALIZE a corpus of documents into a folder of clean .txt/.md/.json
files — one topic/procedure per file, with YAML front-matter metadata — which AIAR
then chunks, embeds, and indexes. You are NOT collecting documents now; you are
writing the brief.

Interview me ONE topic at a time (don't dump all questions at once). For each, give
a recommended default I can accept by saying "default". After the interview, OUTPUT
THE COMPLETE BRIEF as a single Markdown file I can save.

Ask me about:

1. PROJECT NAME + ASSISTANT PERSONA — what is this assistant called, and who is it
   for? (default: "<Domain> Expert", general audience.) Note any "mode" (e.g.
   owner-safe, beginner, pro).
2. SCOPE — exactly what topics/questions must it answer? What is explicitly OUT of
   scope for v1? (Encourage a tight v1 + a "later" list.)
3. AUTHORITATIVE SOURCES — which domains/sites/docs are the SOURCES OF TRUTH?
   (official manuals, vendor docs, standards bodies, government sites). I'll give
   you base domains + seed URLs; you'll organize them into an allow-list table with
   a trust level per domain, and a "do NOT include in v1" list (forums, social,
   blogs, third-party mirrors, anything paywalled/auth-gated).
4. HARD RULES — confirm the safe-collection rules (default, recommended — include
   ALL unless I object): use official sources first; respect robots.txt / terms /
   rate limits; never bypass auth, paywalls, CAPTCHAs, access controls; never
   collect or store private/personal data; keep source URL + retrieval timestamp +
   title + content hash for every doc; preserve warnings/cautions/version+date
   applicability; if sources conflict prefer the most specific + most recent
   official one.
5. SAFETY / SENSITIVITY CLASSES — does this domain have dangerous or
   expert-only material (medical, electrical, high-voltage, legal, financial,
   structural, security)? If so, define safety classes (e.g. safe / caution /
   professional-only / danger) and the assistant behavior for each (give steps &
   cite / give limited checks + stop condition / explain & escalate / refuse DIY &
   tell the user to get a professional). If the domain is low-risk, say so and keep
   a single "informational" class.
6. CHUNKING / FILE-SPLIT GUIDANCE — how should the collector split content into
   files? (default: one procedure/section/record per file; 500–900-token prose
   chunks; tables → Markdown; one error-code/record/recall per file). Remind me
   AIAR does the actual chunk+embed; this guidance shapes how files are SPLIT.
7. METADATA SCHEMA — which YAML front-matter fields should every file carry?
   (default baseline: source_domain, source_url, retrieved_at, document_title,
   document_type, section_path, version/date applicability, safety_class,
   content_hash, plus domain-specific fields you propose from my answers.)
8. RAG INSTANCES — should sources be split into multiple AIAR instances (e.g. one
   per source group) or one? (default: separate instances per major source group,
   switchable in AIAR Settings; otherwise a single instance + per-file category.)
9. RESPONSE FORMAT — what structure should the deployed assistant's answers take?
   (default: Category → Direct Answer → Evidence/citations → Safe steps → Do-not /
   when-to-escalate → Confidence.) This becomes AIAR's system prompt.
10. EVALUATION — give me 15–25 example test questions across the scope, plus the
    scoring dimensions and the minimum launch bar (e.g. no uncited claims, no
    dangerous DIY, every technical answer cites a source). These become AIAR
    eval cases.

When done, OUTPUT the brief with these sections, in this order, mirroring the Tesla
example: Title + scope header; "AI Agent Prompt" (mission + hard rules + desired
output corpus); "Allowed Base Domains" table + "Do Not Include in v1"; "Seed URLs"
grouped by source type with per-group target content + special handling; "Safety
Classification System" (if applicable); "Required Metadata Fields"; "Retrieval
Strategy" (mapped to AIAR instances/categories); "Assistant Response Format";
"Example RAG Questions"; "Evaluation Criteria" + minimum launch bar. End with a
short "How to run in AIAR" footer:
  1) save collected files under corpus/<name>/
  2) `python -m aiar.rag.ingest corpus/<name> --instance <name>`
  3) put the Response Format into the system prompt (Settings page)
  4) load the Example Questions into examples/cases.json and run
     `python -m aiar.eval.runner` to measure RAG lift.

Keep the brief domain-accurate and safety-first. Do not invent source URLs — ask me
for them, or mark them as "TODO: confirm" so the collector verifies before crawling.
````
