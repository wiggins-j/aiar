# EXAMPLE — AIAR RAG Collection Brief

> This is a **worked example** of a "RAG Collection Brief": a single `.md` file you
> hand to a capable AI agent (with web/file tools) so it finds, fetches, and
> normalizes a corpus *for* you, instead of you gathering documents by hand.
>
> **How it maps to AIAR:** the AI's job is **find → collect → normalize into a
> folder** of clean `.txt`/`.md`/`.json` files (one record per file, with the
> metadata below written as front-matter inside each file). **AIAR does the
> chunk + embed + index** when you then run
> `python -m aiar.rag.ingest <that folder> --instance <name>`.
>
> To generate your own brief like this one, hand
> [`collection-brief-builder-prompt.md`](collection-brief-builder-prompt.md) to your
> AI and answer its questions. See the repo README/PLAYBOOK section
> "Building your RAG corpus: two ways."

---

# Tesla Safety & Recall Expert — NHTSA-Sourced

Created: 2026-05-24 · Revised: 2026-05-25
Project: **Tesla Safety & Recall Expert — NHTSA-Sourced**
Primary vehicle scope: **Tesla Model 3 and Tesla Model Y**
Secondary scope: Model S, Model X, Cybertruck (after v1 is stable).

> **Why NHTSA-only?** Tesla's own sites are not usable by an automated collector:
> `www.tesla.com` returns **HTTP 403** to programmatic HTTP fetchers (owner
> manuals, support pages, and PDFs alike), and `service.tesla.com` is behind
> authentication. This brief therefore sources **only** the U.S. government's open
> NHTSA vehicle-safety data, which is authoritative, free, and designed for
> programmatic access. The trade-off: this corpus covers **recalls, investigations,
> complaints, manufacturer communications, safety ratings, and emergency-response
> guides** — it does **not** cover owner-manual operation or DIY maintenance steps
> (those live only on the blocked Tesla domains). If you need owner-manual content,
> collect it with a browser-capable agent or by manual download into a *separate*
> instance; do not mix it with this official NHTSA record set.

---

# AI Agent Prompt: Tesla NHTSA Safety-Record Collector

You are an AI research agent building a high-quality Retrieval-Augmented Generation
knowledge base for a local model called **Tesla Safety & Recall Expert**.

Your job is to find, collect, and normalize **official U.S. government NHTSA**
vehicle-safety records so a small local model (any Qwen served by Ollama, e.g.
`qwen2.5:7b`) can answer Tesla safety, recall, and complaint questions with precise
citations.

## Mission

Build a RAG corpus that lets the model answer questions about:

- Tesla recalls (campaign numbers, affected populations, conditions, remedies)
- whether a recall is remedied by an over-the-air (OTA) update or a service visit
- NHTSA defect investigations (openings, closings, scope)
- manufacturer communications and technical service bulletins (TSBs) on file with NHTSA
- owner-reported complaints (clearly labeled as non-authoritative)
- NCAP safety ratings (crash-test / rollover) where available
- emergency-response guidance for Tesla vehicles (NHTSA-hosted)
- how an owner should act on a possible recall (including VIN lookup)

## Hard Rules

1. Use official NHTSA sources only (the domains listed below).
2. Respect robots.txt, site terms, rate limits, and API usage norms.
3. Do not bypass authentication, paywalls, CAPTCHAs, or technical access controls.
   (This is why Tesla-owned domains are excluded — see "Why NHTSA-only?" above.)
4. Do not scrape private owner data; do not store VINs, license plates, names,
   addresses, phone numbers, or any partial VIN that may appear in complaint records.
5. Keep exact source URLs, retrieval timestamps, record IDs, and content hashes.
6. Preserve campaign numbers, component categories, model-year ranges, dates,
   consequence and remedy text, and revision notes verbatim.
7. Treat high-voltage, SRS/airbag, battery, braking, steering, and driver-assistance
   topics as safety-sensitive.
8. Label every owner-reported complaint as non-authoritative; prioritize recalls,
   investigations, and manufacturer communications over complaints.
9. The assistant must explain, cite, and escalate appropriately. It must not invent
   recalls, campaign numbers, torque specs, or repair procedures, and must not give
   DIY instructions for high-voltage, airbag, or structural work.
10. If records conflict, prefer the most specific and most recent official record for
    the user's stated model, model year, and region.

## Desired Output Corpus

Index as retrievable records with rich metadata — **one record per file**:

- one recall campaign per file
- one investigation (opening or closing resume) per file
- one manufacturer communication / TSB per file
- one complaint per file (or a small grouped summary clearly marked owner-reported)
- one safety-ratings record per model/year/variant
- one emergency-response guide per model/year

---

# Allowed Sources (NHTSA only)

Only collect from these official U.S. government domains.

| Source | Purpose | Trust |
|---|---|---|
| `https://api.nhtsa.gov` | Open JSON API: recalls, complaints, safety ratings (verified working) | Official U.S. government |
| `https://www.nhtsa.gov` | Safety-issue search, recalls, investigations, emergency-response-guide pages | Official U.S. government |
| `https://static.nhtsa.gov` | NHTSA-hosted PDFs: Part 573 recall reports, TSBs, investigation PDFs, ERGs | Official U.S. government |
| `https://vinrcl.safercar.gov` | NHTSA VIN recall lookup info and support | Official U.S. government / NHTSA |

## Do Not Include

- **`tesla.com` / `service.tesla.com`** — blocked (403) or auth-gated for automated
  agents; out of scope for this brief. (Collect via a browser agent or manual
  download into a *separate* instance if you need owner-manual content.)
- Reddit, Tesla Motors Club, YouTube, Facebook/X/Instagram/TikTok
- random blogs, repair-shop marketing pages, third-party PDF mirrors
- leaked or unauthorized service documents

---

# Seed Endpoints (verified working)

Use the open NHTSA API. Iterate over models and model years. Examples:

```text
# Recalls for a model + year (returns campaign number, component, summary,
# consequence, remedy, dates):
https://api.nhtsa.gov/recalls/recallsByVehicle?make=tesla&model=model 3&modelYear=2023
https://api.nhtsa.gov/recalls/recallsByVehicle?make=tesla&model=model y&modelYear=2024

# Owner-reported complaints for a model + year (label as non-authoritative):
https://api.nhtsa.gov/complaints/complaintsByVehicle?make=tesla&model=model 3&modelYear=2023

# Safety-ratings vehicle variants for a model + year (then resolve ratings by VehicleId):
https://api.nhtsa.gov/SafetyRatings/modelyear/2023/make/tesla/model/model 3
https://api.nhtsa.gov/SafetyRatings/VehicleId/<id>
```

Human-readable / PDF sources to supplement:

```text
https://www.nhtsa.gov/recalls
https://www.nhtsa.gov/search-safety-issues
https://www.nhtsa.gov/resources-investigations-recalls
https://www.nhtsa.gov/emergency-response-guides
https://vinrcl.safercar.gov/vin/
# Part 573 reports, TSBs, ERGs are served as PDFs under:
https://static.nhtsa.gov
```

Coverage target: Model 3 (2017–present) and Model Y (2020–present); expand to
Model S / X / Cybertruck once v1 is stable. Recalls alone run to 100+ records
across the lineup; complaints number in the hundreds per model-year.

---

# Safety Classification System

Tag every record with one safety class.

| Safety Class | Meaning | Assistant Behavior |
|---|---|---|
| `owner_safe` | Normal owner action (e.g. check VIN, install an OTA update) | Give steps and cite source |
| `owner_caution` | Owner may inspect, but should not repair | Give safe checks, clear stop condition |
| `service_center` | Requires Tesla service or a trained technician | Explain issue and escalation path |
| `first_responder` | Emergency-response guidance | Conservative emergency instructions only |
| `high_voltage_danger` | HV battery/cables, pyrotechnics, SRS/airbag, severe crash/fire/submersion | No DIY; instruct user to avoid and escalate |

---

# Required Metadata Fields

Each record file should carry consistent YAML front-matter:

```yaml
source_domain:        # nhtsa.gov | api.nhtsa.gov | static.nhtsa.gov | safercar.gov
source_url:
retrieved_at:
document_title:
document_type:        # recall | investigation | complaint | manufacturer_communication | tsb | safety_rating | emergency_response_guide
record_type:
nhtsa_campaign_number:
odi_number:           # for complaints/investigations
investigation_number:
make:
model:
model_year_range:
component:
summary:
consequence:
remedy:
affected_population:
ota_update_possible:
service_required:
authoritative:        # true for recalls/investigations/TSBs; false for complaints
safety_class:
owner_safe:
content_hash:
chunk_id:
```

Fields may be null if not applicable, but keep the schema consistent.

---

# Retrieval Strategy

Use separate AIAR **RAG instances** (or `--category` tags within one instance) for:

1. Recalls
2. Investigations
3. Manufacturer communications / TSBs
4. Complaints (owner-reported)
5. Safety ratings
6. Emergency-response guides

> **AIAR mapping:** ingest each group into its own instance, e.g.
> `python -m aiar.rag.ingest corpus/tesla-recalls --instance tesla-recalls`, and
> switch the active instance per question (or in the Settings page). Within one
> instance, `--category` plus the front-matter above gives per-record filtering.

Recommended retrieval flow:

1. Detect model, model year, and component/topic from the question.
2. Retrieve from authoritative records (recalls/investigations/TSBs) first.
3. Use complaints only as supporting, clearly-labeled owner-reported signal.
4. If high-voltage, airbag, or first-responder material is involved, switch to a
   safety-first response.
5. Cite every answer with record title, campaign/ODI number, URL, and retrieval date.
6. If no relevant record is retrieved, say so. Do not guess.

---

# Assistant Response Format

```text
Category:
Owner-safe / Owner-caution / Service-center / Emergency / High-voltage-danger

Direct Answer:
Brief answer to the user's question.

Evidence:
- Record title, campaign/ODI number, URL
- Applicable model / year

What You Can Safely Do:
1.
2.

When To Contact Tesla Service / NHTSA:
- Clear trigger conditions

Confidence:
High / Medium / Low, with reason
```

> **AIAR mapping:** put this response contract into the **system prompt** (Settings
> page → system-prompt editor, or `POST /api/system-prompt`). The AIAR judge then
> scores answers against it.

---

# Example RAG Questions

Use these as initial tests after indexing. (In AIAR: drop them into
`examples/cases.json` and run `python -m aiar.eval.runner` to measure RAG lift.)

1. What does NHTSA recall 23V-085 cover, and how is it remedied?
2. Which Tesla recalls were fixed by an over-the-air update versus a service visit?
3. What recall affects the 2024 Model Y front seats?
4. How should the assistant distinguish an NHTSA complaint from an official recall?
5. What should an owner do if a recall may apply but they only have the year and
   model and no VIN?
6. What components are most commonly reported in 2023 Model 3 complaints?
7. Was the pyrotechnic battery-disconnect issue (23V-434) an OTA fix or a service fix?

---

# Evaluation Criteria

Score answers 0–5 on each dimension.

| Dimension | Description |
|---|---|
| Source accuracy | Cites the right NHTSA record and campaign/ODI number |
| Model/year specificity | Correctly handles Model 3 vs Model Y and year ranges |
| Authority handling | Distinguishes recalls/investigations from owner complaints |
| Safety classification | Correctly labels owner-safe vs service-only vs dangerous |
| Non-hallucination | Does not invent recalls, campaign numbers, or remedies |
| Escalation judgment | Tells the user when to contact Tesla service or NHTSA |
| Clarity | A normal owner can understand the answer |

Minimum launch bar:

- No invented recalls or campaign numbers.
- Complaints always labeled as owner-reported, non-authoritative.
- No DIY instructions for HV battery, SRS/airbag, or structural repair.
- Every safety claim carries an NHTSA citation.
