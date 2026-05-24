# EXAMPLE — AIAR RAG Collection Brief

> This is a **worked example** of a "RAG Collection Brief": a single `.md` file you
> hand to a capable AI agent (with web/file tools) so it finds, fetches, and
> normalizes a corpus *for* you, instead of you gathering documents by hand.
>
> **How it maps to AIAR:** the AI's job is **find → collect → normalize into a
> folder** of clean `.txt`/`.md`/`.json` files (one topic/procedure per file, with
> the metadata below written as front-matter inside each file). **AIAR does the
> chunk + embed + index** when you then run
> `python -m aiar.rag.ingest <that folder> --instance <name>`. So treat the
> "chunking / indexing / retrieval-strategy" sections below as guidance for how to
> *split files* and *tag content* — AIAR performs the actual vectorization.
>
> To generate your own brief like this one, hand
> [`collection-brief-builder-prompt.md`](collection-brief-builder-prompt.md) to your
> AI and answer its questions. See the repo README/PLAYBOOK section
> "Building your RAG corpus: two ways."

---

# Tesla Manual Expert RAG Collection Brief

Created: 2026-05-24
Project: **Tesla Manual Expert — Owner-Safe Mode**
Primary vehicle scope: **Tesla Model 3 and Tesla Model Y**
Secondary vehicle scope: Model S, Model X, Cybertruck, Roadster, Tesla Energy products only after v1 is stable.

---

# AI Agent Prompt: Tesla Documentation RAG Collector

You are an AI research and scraping agent responsible for building a high-quality Retrieval-Augmented Generation knowledge base for a local model called **Tesla Manual Expert — Owner-Safe Mode**.

Your job is to find, collect, normalize, chunk, and index official Tesla and official U.S. government vehicle-safety documentation so a small local model (any Qwen served by Ollama, e.g. `qwen2.5:7b`) can answer Tesla product-support questions with precise citations.

## Mission

Build a RAG corpus that allows the model to answer questions about:

- Tesla Model 3 and Model Y owner operation
- alerts, warnings, touchscreen messages, and common troubleshooting
- safe owner-level maintenance
- charging, towing, roadside, and transport guidance
- Service Mode concepts and diagnostic panels
- service-manual procedures and component references
- electrical references, connector references, schematics, and pinouts
- collision repair references
- emergency response guidance and rescue sheets
- recalls, service bulletins, investigations, and NHTSA records

## Hard Rules

1. Use official sources first.
2. Respect robots.txt, site terms, rate limits, and access controls.
3. Do not bypass authentication, paywalls, account requirements, CAPTCHAs, or technical access restrictions.
4. Do not scrape private owner data, VIN-specific data, account data, or service-center records.
5. Do not store user VINs, license plates, addresses, names, phone numbers, or Tesla account details.
6. Keep exact source URLs, crawl timestamps, document titles, section headers, and content hashes.
7. Preserve warning labels, cautions, torque specs, model-year ranges, software-version references, region references, and document revision notes.
8. Treat high-voltage, SRS/airbag, collision-structural, battery, charging-system, steering, braking, and driver-assistance procedures as safety-sensitive.
9. The resulting assistant must explain, cite, and escalate when appropriate. It must not encourage untrained users to perform dangerous service procedures.
10. If documentation conflicts, prefer the most specific and most recent official source for the user's stated vehicle model, model year, region, and software version.

## Desired Output Corpus

The final corpus should be indexed as retrievable chunks with rich metadata, not as giant unstructured PDFs. Each chunk should be small enough to retrieve accurately but large enough to preserve procedural context.

Recommended chunking:

- Owner manual chunks: 500-900 tokens
- Service manual procedure chunks: one procedure or major procedure section per chunk
- Warning/caution chunks: preserve with the procedure they govern and also index separately
- Tables: convert to structured Markdown or JSON-like records
- Error, DTC, and alert records: one alert or code per record when possible
- Recall, TSB, and investigation records: one recall, campaign, bulletin, or investigation per record
- Electrical connector records: one connector, pinout, or schematic reference per record

---

# Allowed Base Domains

Only crawl from these base domains unless explicitly approved later.

| Domain | Purpose | Trust Level |
|---|---|---|
| `https://www.tesla.com` | Owner manuals, support pages, DIY guides, recall pages, first-responder pages, emergency PDFs, video guide pages | Official Tesla |
| `https://service.tesla.com` | Service manuals, electrical references, collision repair docs, Service Mode docs | Official Tesla |
| `https://www.nhtsa.gov` | Recalls, investigations, complaints, manufacturer communications, emergency-response guide database, vehicle safety pages | Official U.S. government |
| `https://static.nhtsa.gov` | Official NHTSA-hosted PDFs, TSBs, recall reports, manufacturer communications, investigation PDFs | Official U.S. government |
| `https://vinrcl.safercar.gov` | NHTSA VIN recall lookup information and support pages | Official U.S. government / NHTSA |

## Do Not Include in v1

Do not crawl or index these sources in v1:

- Reddit
- Tesla Motors Club
- YouTube comments or unofficial transcripts
- Facebook, X, Instagram, TikTok
- random blog posts
- repair-shop marketing pages
- third-party PDF mirrors
- leaked or unauthorized service documents
- scraped copies of Tesla docs hosted on non-Tesla domains

These may be useful later as a separate **community-symptom index**, but they must not be mixed with official documentation.

---

# Seed URLs

Use these as starting points. Expand through same-domain internal links only.

## Tesla Owner Manuals

Primary seeds:

- `https://www.tesla.com/ownersmanual/`
- `https://www.tesla.com/ownersmanual/model3/en_us/`
- `https://www.tesla.com/ownersmanual/modely/en_us/`
- `https://www.tesla.com/ownersmanual/index-model-3-2017.html`
- `https://www.tesla.com/ownersmanual/index-model-3.html`
- `https://www.tesla.com/ownersmanual/index-model-y-2020.html`
- `https://www.tesla.com/ownersmanual/index-model-y.html`

Target owner-manual coverage:

- Model 3, 2017-2023
- Model 3, 2024+
- Model Y, 2020-2024
- Model Y, 2025+
- region-specific variants if easily available
- manual pages that mention release notes, software-version specificity, regional applicability, and in-car manual precedence

Important owner-manual sections to prioritize:

- Using This Owner's Manual
- Getting Started
- Opening and Closing
- Seating and Safety Restraints
- Driving
- Autopilot / Assisted Driving
- Traffic-Aware Cruise Control
- Autosteer
- Full Self-Driving / supervised assisted-driving features, if present
- Charging
- Maintenance
- Wheels and Tires
- Cleaning
- Cold Weather Best Practices
- Towing and Transport
- Roadside Assistance
- Troubleshooting Alerts
- Safety Information
- Specifications

---

## Tesla Service Portal

Primary seed:

- `https://www.tesla.com/support/service-portal`

Purpose:

Use this as a high-level source index for Tesla service resources.

Target resource categories:

- Do It Yourself Guides
- Service Manuals
- Parts Catalog references
- Wiring Diagrams / Electrical References
- Collision Repair
- Maintenance information
- Repair information

---

## Tesla Service Manuals

Primary seeds:

- `https://service.tesla.com/docs/Model3/ServiceManual/en-us/index.html`
- `https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/index.html`
- `https://service.tesla.com/docs/ModelY/ServiceManual/en-us/`
- `https://service.tesla.com/docs/ModelY/ServiceManual/2025/en-us/`

Secondary seeds for later expansion:

- `https://service.tesla.com/docs/ModelS/ServiceManual/en-us/`
- `https://service.tesla.com/docs/ModelX/ServiceManual/en-us/`
- `https://service.tesla.com/docs/Cybertruck/ServiceManual/en-us/`

Target Model 3 / Model Y service-manual coverage:

- Introduction
- Abbreviations and symbols
- How to use this manual
- Best practices
- safety precautions
- Airbag and SRS precautions
- high-voltage precautions
- lifting and jacking
- maintenance service intervals
- Vehicle Health Check
- firmware service
- Diagnostic Trouble Codes
- wheels and tires
- brakes
- suspension
- steering
- thermal system
- body exterior
- body interior
- closures
- electrical
- low-voltage battery / LV power
- high-voltage battery / HV system
- drive units
- charging system
- torque specifications
- remove-and-replace procedures

Special handling:

- Preserve all warnings and cautions.
- Tag procedures with safety class.
- Do not flatten steps that depend on previous conditions.
- Extract part names, torque values, fastener types, tool references, and model-year applicability.

---

## Tesla Electrical References / Wiring Diagrams

Primary seeds:

- `https://service.tesla.com/docs/Model3/ElectricalReference/`
- `https://service.tesla.com/docs/ModelY/ElectricalReference/`
- `https://service.tesla.com/docs/ModelY/ElectricalReference/2025/index-model-y-2025.html`

Target content:

- interactive schematics
- connector reference pages
- connector IDs
- Tesla part numbers
- connector manufacturer and part number
- pin/cavity data
- wire colors
- wire sizes
- wire destinations
- production-date filters
- factory filters
- schematic navigation notes

Metadata to capture:

```yaml
doc_type: electrical_reference
model:
model_year_range:
factory:
production_date_start:
production_date_end:
connector_id:
tesla_part_number:
connector_manufacturer:
connector_part_number:
wire_color:
wire_size:
cavity:
destination_designator:
destination_cavity:
source_url:
retrieved_at:
content_hash:
```

Special handling:

- Electrical references are technical and safety-sensitive.
- The final assistant should explain references but should not instruct untrained owners to probe HV circuits or modify wiring.

---

## Tesla Service Mode

Primary seeds:

- `https://service.tesla.com/service-mode`
- `https://service.tesla.com/docs/Public/ServiceMode/service_mode_user_guide.pdf`

Target content:

- what Service Mode is
- supported users / intended use
- how Service Mode behavior differs by model, year, software version, and hardware
- safety limitations
- diagnostic panels
- charging panels
- low-voltage battery information
- high-voltage system state
- ECU update status
- vehicle health checks
- thermal routines
- camera calibration
- software reinstall / firmware panels
- service actions that require trained personnel

Metadata to capture:

```yaml
doc_type: service_mode
source_software_version:
model_applicability:
panel_name:
function_category:
safety_class:
owner_safe:
source_url:
retrieved_at:
content_hash:
```

---

## Tesla Do It Yourself Guides

Primary seed:

- `https://www.tesla.com/support/do-it-yourself-guides`

Purpose:

This is one of the highest-value and safest document sets for the assistant.

Target DIY categories:

- wiper blade replacement
- cabin air filter replacement
- washer fluid
- tire pressure and tire care
- wheel covers / aero covers
- key card / phone key basics
- charging adapter basics
- simple resets and restarts
- software update basics
- owner-performable accessories
- model-specific owner maintenance procedures

Preferred chunk format:

```yaml
doc_type: diy_guide
model:
model_year_range:
procedure_name:
owner_safe: true
tools_required:
parts_required:
steps:
warnings:
source_url:
retrieved_at:
content_hash:
```

---

## Tesla Vehicle Maintenance

Primary seed:

- `https://www.tesla.com/support/vehicle-maintenance`

Target content:

- maintenance intervals
- brake service expectations
- tire rotation and tire care
- cabin air filter guidance
- HEPA filter guidance, if applicable
- brake fluid checks
- A/C service guidance
- winter care
- traditional gas-vehicle maintenance that does not apply to Tesla vehicles

---

## Tesla Recall Information

Primary seeds:

- `https://www.tesla.com/support/annual-and-recall-service`
- `https://www.tesla.com/support/recalls`
- `https://www.tesla.com/support/recall-battery-pack-contactor`

Target content:

- Tesla recall campaign pages
- Model 3 / Model Y recall pages
- software-update recalls
- service-center remedy recalls
- recall applicability statements
- owner action guidance
- VIN lookup instructions
- references to NHTSA VIN Recall Search

Preferred recall record format:

```yaml
doc_type: recall
source: tesla
recall_name:
nhtsa_campaign_number:
tesla_campaign_number:
model:
model_year_range:
affected_population:
condition:
risk:
remedy:
owner_action:
ota_update_possible:
service_required:
source_url:
retrieved_at:
content_hash:
```

---

## Tesla First Responder / Emergency Response Docs

Primary seeds:

- `https://www.tesla.com/firstresponders`
- `https://www.tesla.com/firstresponders/vehicles-charging`

Target documents:

- Emergency Response Guides
- Quick Response Sheets
- Rescue Sheets
- vehicle-specific first-responder PDFs
- high-voltage disable procedures
- fire, submersion, post-crash, and tow/storage emergency guidance
- Model 3 Emergency Response Guide
- Model Y Emergency Response Guide
- Model S / Model X / Cybertruck emergency docs for later expansion

Known Tesla-hosted PDF patterns may include:

- `https://www.tesla.com/sites/default/files/downloads/Model_3_Emergency_Response_Guide_en.pdf`
- `https://www.tesla.com/sites/default/files/downloads/2017_Model_3_Emergency_Response_Guide_en.pdf`
- `https://www.tesla.com/sites/default/files/downloads/2016_Model_S_Emergency_Response_Guide_en.pdf`

Special handling:

- These documents are intended for trained first responders.
- Tag as emergency / first-responder / high-voltage-danger.
- The assistant should not convert responder procedures into casual owner repair instructions.
- Emergency answers should be conservative: move away, call emergency services, avoid HV components, follow official responder guidance.

---

## Tesla Collision Repair Procedures

Primary seeds:

- `https://service.tesla.com/docs/BodyRepair/Body_Repair_Procedures/Model_3/HTML/en-us/index.html`
- `https://service.tesla.com/docs/BodyRepair/Body_Repair_Procedures/Model_Y/HTML/en-us/index.html`
- `https://service.tesla.com/docs/BodyRepair/Body_Repair_Procedures/Model_Y_2025/en_us/index.html`

Target content:

- structural repair procedures
- body structure materials
- allowed operations
- sectioning guidelines
- repairability guidelines
- approved welders
- adhesives and fasteners
- fascia repair guidelines
- glass replacement guidance
- wheel repairability
- post-repair operations
- Tesla collision repair contact/escalation guidance

Special handling:

- Tag as collision-shop.
- The assistant can explain what category of repair a document discusses.
- The assistant should not instruct an untrained user to perform structural repair.

---

## Tesla Video Guides

Primary seed:

- `https://www.tesla.com/support/videos`

Use only if clean transcripts are available from official Tesla pages.

Do not use YouTube comments. Do not use unofficial transcripts unless explicitly approved.

---

## NHTSA Recalls, Investigations, Complaints, and Manufacturer Communications

Primary seeds:

- `https://www.nhtsa.gov/search-safety-issues`
- `https://www.nhtsa.gov/recalls`
- `https://www.nhtsa.gov/resources-investigations-recalls`
- `https://www.nhtsa.gov/emergency-response-guides`
- `https://vinrcl.safercar.gov/vin/`

PDF source domain:

- `https://static.nhtsa.gov`

Target NHTSA content:

- Part 573 Safety Recall Reports
- recall acknowledgments
- defect investigation opening resumes
- defect investigation closing resumes
- recall query results
- manufacturer communications
- technical service bulletins
- owner notification letters
- remedy instructions
- emergency response guide entries
- complaint summaries only if needed and clearly labeled as owner-reported, non-authoritative

Preferred NHTSA record format:

```yaml
doc_type: nhtsa_record
record_type: recall | investigation | manufacturer_communication | tsb | complaint | emergency_response_guide
nhtsa_id:
campaign_number:
investigation_number:
manufacturer:
make:
model:
model_year:
component:
summary:
consequence:
remedy:
dates:
affected_population:
source_url:
retrieved_at:
content_hash:
```

Special handling:

- NHTSA complaints are not authoritative findings.
- Label complaints as owner-reported.
- Recalls, investigations, and manufacturer communications should be prioritized over complaints.

---

# Safety Classification System

Every chunk should receive one of the following safety classes.

| Safety Class | Meaning | Assistant Behavior |
|---|---|---|
| `owner_safe` | Normal owner operation or basic maintenance | Give steps and cite source |
| `owner_caution` | Owner may inspect or perform limited checks, but should avoid deeper repair | Give safe checks, clear stop condition |
| `service_center` | Requires Tesla service or trained technician | Explain issue and escalation path |
| `collision_shop` | Structural/body/collision repair | Explain category and refer to qualified collision repair |
| `first_responder` | Emergency response guidance | Conservative emergency instructions only |
| `high_voltage_danger` | HV battery, HV cables, HV interlock, pyrotechnics, SRS, airbag, severe crash/fire/submersion | Do not provide DIY instructions; instruct user to avoid and escalate |

---

# Required Metadata Fields

Each indexed chunk should include:

```yaml
source_domain:
source_url:
canonical_url:
retrieved_at:
document_title:
document_type:
section_path:
model:
model_year_range:
region:
language:
software_version:
revision_date:
safety_class:
owner_safe:
procedure_name:
alert_code:
dtc_code:
recall_number:
nhtsa_campaign_number:
component:
tools_required:
parts_required:
warnings:
content_hash:
chunk_id:
parent_document_id:
```

Fields may be null if not applicable, but the schema should be consistent.

---

# Retrieval Strategy

Use separate indexes or filters for:

1. Owner manuals
2. DIY guides
3. Service manuals
4. Electrical references
5. Service Mode
6. Collision repair
7. Emergency response
8. Recalls / NHTSA records
9. Troubleshooting alerts / DTCs

> **AIAR mapping:** "separate indexes" = separate **RAG instances**. Ingest each
> source group into its own instance, e.g.
> `python -m aiar.rag.ingest corpus/tesla-owner --instance tesla-owner`, and switch
> the active instance per question (or via the Settings page). Within one instance,
> the `--category` tag plus the front-matter above gives you per-doc filtering.

Recommended retrieval flow:

1. Detect model, model year, region, and software version from user question.
2. Detect problem category.
3. Retrieve from the safest relevant source first.
4. Escalate to technical documents only if needed.
5. If high-voltage, SRS, collision, or first-responder material is involved, switch to safety-first response mode.
6. Cite every answer with source title, section, URL, and retrieval timestamp.
7. If no relevant source is retrieved, say so. Do not guess.

---

# Assistant Response Format

The deployed assistant should answer in this structure:

```text
Category:
Owner-safe / Owner-caution / Service-center / Collision-shop / Emergency / High-voltage-danger

Direct Answer:
Brief answer to the user's question.

Evidence:
- Source title, section, URL
- Relevant model/year/software applicability

Steps You Can Safely Try:
1.
2.
3.

Do Not Attempt:
- Safety-sensitive or unsupported actions

When To Schedule Tesla Service:
- Clear trigger conditions

Confidence:
High / Medium / Low, with reason
```

> **AIAR mapping:** put this response contract into the **system prompt** (Settings
> page → system-prompt editor, or `POST /system-prompt`). The AIAR judge will then
> score answers against it.

---

# Example RAG Questions

Use these as initial tests after indexing. (In AIAR: drop them into
`examples/cases.json` and run `python -m aiar.eval.runner` to measure RAG lift.)

## Owner Manual

1. Can I tow a Model 3 with only the rear wheels lifted?
2. Where does Tesla say the most customized owner manual is located?
3. What does the Model 3 manual say about tire pressure after changing tire pressures?
4. What should I do if my Tesla key card is not recognized?
5. What charging warnings apply before using an adapter?
6. What does Tesla say about cleaning seat belts?
7. When does the Model Y manual say to use Transport Mode?

## DIY / Maintenance

1. How do I replace the cabin air filter on a Model 3, and what tools are listed?
2. What owner-performable maintenance does Tesla list?
3. Does a Tesla need oil changes?
4. What tire rotation guidance does Tesla provide?
5. What maintenance items are different for cold-weather use?

## Service Manual

1. Where does the service manual list lifting and jacking information?
2. What safety precautions are listed before SRS component work?
3. What service manual section covers Diagnostic Trouble Codes?
4. What does the manual say about firmware reinstall vs firmware update?
5. What service class should apply to high-voltage battery removal?

## Service Mode

1. What is Service Mode intended for?
2. Which guide software version is the public Service Mode guide based on?
3. Why might the Service Mode guide differ from the vehicle screen?
4. Which Service Mode panels relate to charging or low-voltage battery status?
5. What should the assistant say before telling a normal owner to run a service action?

## NHTSA / Recalls

1. What NHTSA records exist for a given Tesla Model 3 recall campaign?
2. Which affected models are listed in NHTSA recall 23V-085?
3. Was a specific recall resolved by OTA update or service visit?
4. How should the assistant distinguish a complaint from an official recall?
5. What should an owner do if a recall may apply but they only have the vehicle year/model and no VIN?

---

# Evaluation Criteria

Score answers from 0-5 on each dimension.

| Dimension | Description |
|---|---|
| Source accuracy | Answer cites the right document and section |
| Model/year specificity | Correctly handles Model 3 vs Model Y and year ranges |
| Safety classification | Correctly labels owner-safe vs service-only vs dangerous |
| Procedural accuracy | Steps match official instructions |
| Non-hallucination | Does not invent torque specs, codes, warnings, or recalls |
| Escalation judgment | Tells user when to stop and schedule service |
| Clarity | Normal owner can understand answer |
| Completeness | Includes enough detail without over-answering |

Minimum launch bar:

- No invented safety-critical procedures.
- No uncited recall claims.
- No DIY instructions for HV battery, SRS/airbag, pyrotechnic, or structural collision repair.
- Every technical answer includes source citations.
