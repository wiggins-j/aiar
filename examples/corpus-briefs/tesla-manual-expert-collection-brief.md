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

# Tesla Knowledge Base — Open-Data & Access-Respecting

Created: 2026-05-24 · Revised: 2026-05-25
Project: **Tesla Knowledge Base — Open-Data & Access-Respecting**
Primary vehicle scope: **Tesla Model 3 and Tesla Model Y**
Secondary scope: Model S, Model X, Cybertruck (after v1 is stable).

> **Access policy (read first).** Do **not** try to bypass Tesla's bot protection.
> Build the knowledge base around **open APIs, public data files, official
> PDFs/manuals where accessible, and licensed/permissioned sources.** Tesla's own
> web properties block automated HTTP fetchers — `www.tesla.com` returns **HTTP
> 403** to programmatic requests (owner manuals, support pages, and PDFs alike) and
> `service.tesla.com` is behind authentication — so they are **not** crawled by this
> brief. If you need owner-manual content, collect it with a browser-capable agent
> or by manual download into a *separate* instance; never defeat access controls.

The good news: the most authoritative facts (recalls, complaints, safety ratings,
VIN/spec metadata, range/efficiency, charging infrastructure, news) are available
from **open government APIs and public datasets** that need no scraping at all.

---

# AI Agent Prompt: Tesla Open-Data Collector

You are an AI research agent building a Retrieval-Augmented Generation knowledge
base for a local model (any Qwen served by Ollama, e.g. `qwen2.5:7b`) so it can
answer Tesla ownership, safety, spec, charging, and market questions with precise
citations.

## Mission

Collect and normalize official and reputable Tesla data so the model can answer:

- recalls, complaints, investigations, manufacturer communications, safety ratings
- VIN decoding and model/year/body/trim metadata
- EPA range, efficiency (MPGe), energy use, and annual energy cost
- public EV charging infrastructure (station locations, connector types, networks)
- crashworthiness (IIHS / Euro NCAP) where publicly available
- software-update timelines and feature/release-note history (clearly labeled)
- reviews, reliability, owner satisfaction, pricing/market value (licensed sources)
- common owner-reported issues and repair-cost ballparks (clearly labeled)
- corporate facts and production/delivery disclosures (SEC filings)
- market & sales context (third-party estimates; see "Market & Sales" below)

## Hard Rules

1. Use open APIs, public data files, official accessible PDFs, and licensed sources.
2. Respect robots.txt, site terms, rate limits, API-key terms, and licensing.
3. Do **not** bypass authentication, paywalls, CAPTCHAs, or bot protection. Tesla
   domains are excluded for exactly this reason (see Access policy above).
4. Do not scrape private owner data; never store VINs, plates, names, addresses,
   phone numbers, or account details (these can appear in complaint records).
5. Keep exact source URLs, retrieval timestamps, record IDs, and content hashes.
6. Preserve campaign numbers, components, model-year ranges, dates, consequence and
   remedy text, spec values, and units verbatim.
7. Treat high-voltage, SRS/airbag, battery, braking, steering, and driver-assistance
   topics as safety-sensitive.
8. Mark authority on every record: **authoritative** (regulators, official specs,
   SEC) vs **owner-reported/anecdotal** (complaints, forums, Reddit, YouTube) vs
   **editorial/licensed** (reviews). Never let anecdotes override regulators.
9. Do not invent recalls, campaign numbers, torque specs, ranges, prices, or sales
   figures. Cite, and say so when a source is an estimate.
10. If records conflict, prefer the most specific and most recent authoritative
    source for the user's stated model, year, and region.

---

# Source Tiers

Each row notes **access style** and a **verification status** from a live check on
2026-05-25: ✅ verified open (no auth), 🔑 free API key, 🧾 needs a declared
User-Agent, 📄 public pages/files (fetch politely), 🔒 licensed/paywalled (respect
terms), ⛔ blocked/auth — do not bypass, 🗣️ unofficial (label as such).

## Tier 1 — Best factual sources (open data)

| Source | Use it for | Access / status |
|---|---|---|
| `api.nhtsa.gov` | Recalls, complaints, safety ratings | ✅ Open JSON API (verified) |
| `static.nhtsa.gov` | Part 573 recall PDFs, TSBs, investigation PDFs, ERGs | 📄 Public PDFs |
| `vpic.nhtsa.dot.gov` | VIN decode; model/year/body/manufacturer metadata | ✅ Official vPIC API (verified) |
| `data.transportation.gov` | Structured ODI complaints/recalls (Socrata datasets) | 📄 Socrata API (DB-style, no scraping) |
| `fueleconomy.gov` | EPA range, MPGe, energy use, annual energy cost | ✅ Web service / XML (verified; use menu endpoints to find valid year/make/model) |
| `epa.gov` | Raw EPA vehicle test data | 📄 Downloadable data files |
| `developer.nrel.gov` (AFDC) | EV charging stations, connector types, networks | 🔑 Free API key (DEMO_KEY works for testing; verified) |
| `iihs.org` | Crashworthiness / safety ratings | 📄 Public model pages |
| `euroncap.com` | European crash/safety testing | 📄 Public results / downloadable reports |

## Tier 2 — Official Tesla sources (excluded here; access-gated)

| Source | Use it for | Status |
|---|---|---|
| `tesla.com/ownersmanual` | Owner manuals, charging, maintenance, alerts | ⛔ HTTP 403 to automated agents — browser/manual only |
| `service.tesla.com/docs` | Service manuals, torque specs, fluids, intervals | ⛔ Auth-gated |
| `developer.tesla.com` (Fleet API) | Authorized-owner/fleet telemetry & vehicle endpoints | ⛔ Requires auth tokens; not general public facts |
| `epc.tesla.com` | Parts catalog, assemblies, part numbers | ⛔ Account/terms apply |
| Tesla support/software pages | Official "how Tesla says it works" (OTA, app) | ⛔ Bot-blocked — browser/manual only |

> Collect Tier 2 only with a browser-capable agent or manual download, into a
> **separate** instance, and only in ways Tesla's terms permit. Do not bypass blocks.

## Tier 3 — Software updates, release notes, features

| Source | Use it for | Trust |
|---|---|---|
| `notateslaapp.com` | Software updates, feature explainers, release-note tracking | 🗣️ Unofficial — label |
| `teslafi.com` | Firmware rollout tracker | 🗣️ User-contributed fleet data |
| `teslascope.com` | Update tracking / release-note timelines | 🗣️ Unofficial |
| `github.com/teslamotors/vehicle-command` | Official vehicle command SDK | 📄 Official GitHub (use `gh`/API) |

## Tier 4 — Reviews, reliability, owner satisfaction (licensed)

| Source | Use it for | Caution |
|---|---|---|
| `consumerreports.org` | Reliability, owner satisfaction, road tests | 🔒 Subscription/licensing — do not scrape paid content |
| `edmunds.com` | Expert/owner reviews, pricing, trims, specs | 🔒 Check terms/API/licensing |
| `kbb.com` | Used values, owner/expert reviews | 🔒 Pricing/market value |
| `cars.com` | Owner reviews, inventory pricing, listings | 🔒 Real-world sentiment + market data |
| `jdpower.com` | Dependability, quality, ratings | 🔒 Licensed/high-authority |
| `caranddriver.com`, `motortrend.com`, `insideevs.com`, `greencarreports.com` | Road tests, long-term/EV reporting | 🔒 Editorial — separate review opinion from factual specs |
| `recurrentauto.com` | Battery health, used EV range context | 🔒 Treat proprietary scores as their methodology |

## Tier 5 — Issues, repairs, complaints, patterns

| Source | Use it for | Trust |
|---|---|---|
| NHTSA complaints (`api.nhtsa.gov`) | Structured owner-reported safety issues | ✅ verified — label as complaints, not verified defects |
| NHTSA recalls / investigations / manufacturer comms | Verified regulatory/defect history | ✅ Highest authority |
| `carcomplaints.com` | Complaint clustering by model/year/category | 🗣️ Pattern discovery — unverified |
| `repairpal.com` | Repair-cost / maintenance-cost estimates | 🗣️ Ballparks; sparse Tesla-specific data |
| `teslamotorsclub.com`, `teslaownersonline.com` | Owner repair threads, DIY, service experiences | 🗣️ High volume, unverified — "owner-reported" |
| Reddit (`r/TeslaModel3`, `r/TeslaModelY`, `r/teslamotors`, …) | Owner issues, tips, costs, buying advice | 🗣️ Use Reddit API / permitted JSON — label anecdotal |
| YouTube Data API | Comments/metadata from repair/review channels | 🗣️ Sentiment/FAQ mining — not factual unless corroborated |

## Tier 6 — News and monitoring

| Source | Use it for | Status |
|---|---|---|
| `api.gdeltproject.org` (GDELT) | Real-time news monitoring / broad Tesla coverage | ✅ Open JSON API (verified) |
| Reuters, AP, Bloomberg, WSJ, CNBC, The Verge, InsideEVs, Electrek | Recalls, lawsuits, production, policy news | 🔒/📄 Respect each outlet's terms |
| SEC EDGAR (`data.sec.gov`, `sec.gov`) | Corporate facts, risks, production/delivery disclosures | 🧾 Open, but requires a descriptive `User-Agent` header (403 without) |

---

# Verified Quick-Start Endpoints

These were confirmed working (open, no auth unless noted) on 2026-05-25:

```text
# NHTSA recalls / complaints / safety ratings:
https://api.nhtsa.gov/recalls/recallsByVehicle?make=tesla&model=model 3&modelYear=2023
https://api.nhtsa.gov/complaints/complaintsByVehicle?make=tesla&model=model 3&modelYear=2023
https://api.nhtsa.gov/SafetyRatings/modelyear/2023/make/tesla/model/model 3

# vPIC VIN decode + model list:
https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMake/tesla?format=json
https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/<VIN>?format=json

# EPA fuel economy (XML; use menu endpoints to discover valid year/make/model):
https://www.fueleconomy.gov/ws/rest/vehicle/menu/options?year=2023&make=Tesla&model=Model%203

# NREL AFDC charging stations (DEMO_KEY for testing; get a free key for real use):
https://developer.nrel.gov/api/alt-fuel-stations/v1.json?fuel_type=ELEC&state=CA&api_key=DEMO_KEY

# GDELT news monitoring:
https://api.gdeltproject.org/api/v2/doc/doc?query=tesla recall&mode=artlist&format=json

# SEC EDGAR (send a real User-Agent like "you@example.com"; 403 without it):
https://data.sec.gov/submissions/CIK0001318605.json   # Tesla, Inc.
```

---

# Market & Sales (estimates — handle with care)

Tesla does **not** publish U.S. sales broken out by model, and does not publish a
"used Tesla" sales count. So a "sales across the USA, new and used, all models"
view must be **assembled from third-party estimates** and clearly labeled:

- **New (U.S.) — third-party estimates** (they differ by methodology):
  - Cox Automotive / Kelley Blue Book EV Sales Reports (public PDFs) — ~633K Tesla
    U.S. units in 2024.
  - GoodCarBadCar (`goodcarbadcar.net`) — ~517K (2024), ~670K (2023);
    Cybertruck ~39K (2024); Model Y #1, Model 3 #2 EV.
  - Argonne National Lab (`anl.gov`) and AFDC (`afdc.energy.gov/data`) — U.S.
    plug-in EV monthly/annual sales (includes Tesla models).
- **Global production/deliveries (official)**: Tesla SEC filings (10-K/10-Q via
  `data.sec.gov`) — global, not U.S.-by-model.
- **Used (U.S.)**: no authoritative Tesla-specific count is published. Aggregate
  used-EV volume (Cox) and marketplace data (Edmunds, KBB, Cars.com, iSeeCars,
  Recurrent) give estimates only — label as estimates and cite the source/quarter.

> Always tag sales records `authoritative: false`, `record_type: market_estimate`,
> and store the source + methodology so the assistant can say "estimate, per X."

---

# Safety Classification System

Tag safety-relevant records with one class.

| Safety Class | Meaning | Assistant Behavior |
|---|---|---|
| `owner_safe` | Normal owner action (check VIN, install OTA update) | Give steps and cite |
| `owner_caution` | Owner may inspect, not repair | Safe checks, clear stop condition |
| `service_center` | Requires Tesla service / trained technician | Explain + escalation path |
| `first_responder` | Emergency-response guidance | Conservative emergency instructions only |
| `high_voltage_danger` | HV battery/cables, pyrotechnics, SRS/airbag, severe crash/fire | No DIY; instruct user to avoid and escalate |

---

# Required Metadata Fields

Consistent YAML front-matter per record file:

```yaml
source_domain:
source_url:
retrieved_at:
document_title:
document_type:        # recall | investigation | complaint | tsb | safety_rating |
                      # spec | charging_station | review | software_update | news |
                      # sec_filing | market_estimate | emergency_response_guide
record_type:
authoritative:        # true = regulator/official spec/SEC; false = anecdote/estimate
nhtsa_campaign_number:
odi_number:
make:
model:
model_year_range:
component:
summary:
consequence:
remedy:
safety_class:
owner_safe:
source_tier:          # 1..6 from this brief
content_hash:
chunk_id:
```

Fields may be null if not applicable, but keep the schema consistent.

---

# Recommended Ingestion Order

1. **NHTSA + vPIC + EPA** — recalls, complaints, investigations, VIN decoding,
   safety ratings, range, efficiency, model-year metadata. (All open APIs.)
2. **NREL AFDC** — charging infrastructure.
3. **IIHS + Euro NCAP** — crash/safety evidence.
4. **(Browser/manual, separate instance) Tesla owner & service manuals** — official
   "how the car works" and service procedures, only if collected without bypassing
   blocks.
5. **Consumer Reports / Edmunds / KBB / Cars.com / J.D. Power** — reviews, owner
   satisfaction, reliability, market context, subject to licensing.
6. **CarComplaints + RepairPal + forums + Reddit** — discover common owner-reported
   issues, then validate against NHTSA / recalls / multiple independent reports.
7. **Not a Tesla App + TeslaFi + Teslascope** — software-update timelines, release
   notes, rollout patterns (labeled unofficial).
8. **GDELT + SEC + market/sales estimates** — news monitoring, corporate facts,
   and U.S. sales context (estimates).

> **AIAR mapping:** ingest each group into its own **RAG instance**, e.g.
> `python -m aiar.rag.ingest corpus/tesla-recalls --instance tesla-recalls`, and
> switch the active instance per question (Settings page). Within one instance,
> `--category` plus the front-matter gives per-record filtering.

---

# Retrieval Strategy

1. Detect model, model year, region, and topic from the question.
2. Retrieve from authoritative sources first (NHTSA, vPIC/EPA specs, SEC).
3. Use anecdotal/estimate sources only as labeled, supporting signal.
4. For high-voltage, airbag, or first-responder topics, switch to safety-first mode.
5. Cite every answer (record title, ID/campaign number, URL, retrieval date).
6. If no relevant record is retrieved, say so. Do not guess.

---

# Assistant Response Format

```text
Category:
Owner-safe / Owner-caution / Service-center / Emergency / High-voltage-danger / Info

Direct Answer:
Brief answer to the user's question.

Evidence:
- Record title, ID/campaign number, URL, source tier
- Applicable model / year; note if the figure is an ESTIMATE

What You Can Safely Do:
1.

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

(In AIAR: drop these into `examples/cases.json` and run
`python -m aiar.eval.runner` to measure RAG lift.)

1. What does NHTSA recall 23V-085 cover, and how is it remedied?
2. Which Tesla recalls were fixed by an OTA update versus a service visit?
3. What is the EPA range and MPGe of a 2023 Model 3, and what trims exist (vPIC/EPA)?
4. How many DC fast-charging stations with CCS are near a given state (AFDC)?
5. How should the assistant distinguish an NHTSA complaint from an official recall?
6. What components are most commonly reported in 2023 Model 3 complaints?
7. Roughly how many Teslas were sold new in the U.S. in 2024, and by which source —
   and is a used-Tesla count officially published? (Answer: estimates only.)

---

# Evaluation Criteria

Score answers 0–5 on each dimension.

| Dimension | Description |
|---|---|
| Source accuracy | Cites the right record and ID/campaign number |
| Model/year specificity | Correctly handles Model 3 vs Model Y and year ranges |
| Authority handling | Separates regulators/official specs from anecdotes/estimates |
| Safety classification | Correctly labels owner-safe vs service-only vs dangerous |
| Non-hallucination | Does not invent recalls, specs, prices, or sales figures |
| Escalation judgment | Tells the user when to contact Tesla service or NHTSA |
| Clarity | A normal owner can understand the answer |

Minimum launch bar:

- No invented recalls, campaign numbers, specs, or sales figures.
- Complaints/forums/Reddit/sales estimates always labeled non-authoritative.
- No DIY instructions for HV battery, SRS/airbag, or structural repair.
- Every safety or spec claim carries a citation; every sales figure says "estimate."
- No source collected by bypassing authentication or bot protection.
