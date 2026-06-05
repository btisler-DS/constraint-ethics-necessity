# Publication Corrections Changelog

This file records corrections to the public-facing publication record for
`btisler-DS/constraint-ethics-necessity`. New entries go at the top.

The canonical source of truth is [`publications.json`](publications.json).
Run [`scripts/check_publications.js`](scripts/check_publications.js) to
re-verify that registry against Zenodo at any time.

---

## 2026-06-05 — Protocol 6 canonical resolved; schema refinement

**Verification date:** 2026-06-05.

### Protocol 6 — current canonical citation selected (Version 2)

The previous entry (below) left Protocol 6 canonical status flagged as
`undetermined` because the README did not nominate a current DOI.
Bruce has now resolved this:

- **Current canonical citation (Version 2):** `10.5281/zenodo.20313340`,
  published 2026-05-20. Contains `Protocol6_Results_Paper.pdf` and
  `Protocol6_Confirmatory_Artifacts.zip`. Registry entry
  `p6-results-may20` updated to `repo_canonical: true`,
  `external_citation_status: "cite_this"`,
  `version_status: "current_version"`.
- **Historical (Version 1):** `10.5281/zenodo.19485185`, published
  2026-04-09. Retained on Zenodo. Registry entry `p6-results-apr09`
  updated to `repo_canonical: false`,
  `external_citation_status: "historical"`,
  `version_status: "previous_version"`.

Unlike the Protocol 4 and Protocol 5 April-9 / May-20 pairs, **Zenodo
DOES formally treat the Protocol 6 pair as Version 1 / Version 2 of a
single concept record.** Both records carry `conceptrecid: 19485184`
(concept DOI `10.5281/zenodo.19485184`); the API marks `19485185` as
`is_last:false` and `20313340` as `is_last:true`. That justifies setting
`version_status` to `previous_version` and `current_version` on this
pair, and populating the `replaces_doi` / `replaced_by_doi` fields
between them.

The README now has a `## Protocol 6 Results — Emergent Constraint
Fields` section for parity with Protocols 3, 4, and 5, citing the
Version 2 DOI as canonical and labeling the April-9 DOI as the
Historical Version 1 record.

### Schema refinement (bump to `1.1.0`)

To accommodate the version_status distinction cleanly, two schema
changes landed in the same commit:

1. **Field rename:** `zenodo_status` → `zenodo_formal_status`, for
   precision about what the field measures.
2. **New field:** `version_status` (allowed: `current_version`,
   `previous_version`, `only_version`). Captures Zenodo-formal
   versioning explicitly so it cannot be confused with the editorial
   `external_citation_status` field.

Every record now carries both fields. Only the Protocol 6 pair uses
`current_version` / `previous_version`; the rest of the registry uses
`only_version`. The Protocol 4 and Protocol 5 April-9 records remain
`only_version` (their concept records contain only one version each),
which preserves the distinction between Bruce's editorial "historical"
classification and Zenodo's "previous version" classification.

The checker script (`scripts/check_publications.js`) was updated to
validate the new vocabulary and to flag the inconsistency
`version_status: previous_version` paired with
`external_citation_status: cite_this`.

`publication-check-report.md` was regenerated — all 20 records pass.

---

## 2026-06-05 — Withdrawn DOI, Protocol 2/3 attribution, Protocol 4/5/6 review

**Verification date:** 2026-06-05 (via Zenodo records API).

### 1. `10.5281/zenodo.19485721` is withdrawn (HTTP 410 Gone)

- This DOI returns HTTP 410 Gone from both `doi.org` and the Zenodo
  records API. The underlying record was withdrawn from Zenodo and there is
  no replacement at this DOI.
- A public-facing list previously used this single DOI for two different
  papers: Virtue Theater (Protocol 2 results) and Protocol 3 Results.
  Both attributions were wrong.
- Registry treatment: retained as `internal_id: withdrawn-19485721` with
  `zenodo_status: "withdrawn"`, `repo_canonical: false`,
  `external_citation_status: "do_not_cite"`. Kept so the broken DOI does
  not silently re-enter public output.

### 2. Protocol 2 (Virtue Theater) — corrected DOI

- **Broken DOI on the public-facing list:** `10.5281/zenodo.19485721`
- **Corrected DOI:** `10.5281/zenodo.19485645`
- **Title (verified against Zenodo):** "Virtue Theater: Specification Gaming and Regulatory Constraint Failure in Multi-Agent Systems"
- **Publication date:** 2026-04-09
- Registry entry: `internal_id: p2-virtue-theater`,
  `zenodo_status: "active"`, `repo_canonical: true`,
  `external_citation_status: "cite_this"`.

### 3. Protocol 3 Results — corrected DOI

- **Broken DOI on the public-facing list:** `10.5281/zenodo.19485721` (the
  same withdrawn DOI mistakenly reused).
- **Corrected DOI:** `10.5281/zenodo.20312682`
- **Title (verified against Zenodo):** "Enforcement Opacity Increased Query Behavior in a Constrained MARL System: Protocol 3 Results"
- **Publication date:** 2026-05-20
- Registry entry: `internal_id: p3-results`, `zenodo_status: "active"`,
  `repo_canonical: true`, `external_citation_status: "cite_this"`.

### 4. Protocol 4 Results — canonical status reviewed (NOT formally superseded)

- The public-facing list cited `10.5281/zenodo.19485699` (April 9, 2026,
  "Ethics as Emergent Constraint Response… Protocol 4 Results"). That
  record is still live on Zenodo.
- The repo README, however, cites `10.5281/zenodo.20314828` (May 20, 2026,
  "Architectural Depth Increased Sacrifice-Like Behavior Without
  Ethical-Framework Alignment") as the current Protocol 4 results record.
- **Zenodo does NOT formally mark either as superseding the other.** They
  sit on independent Zenodo concept records (concept `19485698` for the
  April-9 record; concept `20314827` for the May-20 record). There is no
  `isSupersededBy` / `isNewVersionOf` relation between them.
- **Registry treatment:**
  - `p4-results-may20` (`20314828`) → `repo_canonical: true`,
    `external_citation_status: "cite_this"`.
  - `p4-results-apr09` (`19485699`) → `repo_canonical: false`,
    `external_citation_status: "historical"`. Not marked as
    `superseded`, because Zenodo's metadata does not say so.

### 5. Protocol 5 Results — canonical status reviewed (NOT formally superseded)

- Same pattern as Protocol 4. The April-9 record
  `10.5281/zenodo.19485713` ("The Optimization-Sacrifice Tension is
  Architecturally Invariant… Preregistered Null Result") was on the
  public-facing list.
- The repo README cites `10.5281/zenodo.20314078` (May 20, 2026,
  "Temporal Integration Span and Welfare Coupling Did Not Resolve the
  Optimization-Sacrifice Dissociation").
- Zenodo does not formally link them (independent concept records
  `19485712` and `20314077`).
- **Registry treatment:**
  - `p5-results-may20` (`20314078`) → `repo_canonical: true`,
    `external_citation_status: "cite_this"`.
  - `p5-results-apr09` (`19485713`) → `repo_canonical: false`,
    `external_citation_status: "historical"`.

### 6. Protocol 6 Results — canonical status UNRESOLVED

- The public-facing list cited `10.5281/zenodo.19485185` (April 9, 2026,
  "Emergent Constraint Fields Are Causally Active But Do Not Outperform
  Fixed External Rules…").
- An identically-titled May-20 record exists at
  `10.5281/zenodo.20313340`. It is referenced from inside the repo
  (`docs/paper_protocol4.md` lists it as a related identifier) but the
  README does NOT list a Protocol 6 results DOI, so the README cannot
  decide which record is currently canonical.
- **Registry treatment (pending editorial decision):**
  - `p6-results-apr09` (`19485185`) → `repo_canonical: null`,
    `external_citation_status: "undetermined"`.
  - `p6-results-may20` (`20313340`) → `repo_canonical: null`,
    `external_citation_status: "undetermined"`.
- **Action needed:** decide which Protocol 6 record is the current
  external-facing citation, then update `publications.json` accordingly
  (set one to `cite_this` and `repo_canonical: true`; set the other to
  `historical` and `repo_canonical: false`). Adding a Protocol 6 section
  to the README at the same time is recommended for parity with P3/P4/P5.

---

## Policy

**Earlier Zenodo records do not disappear when newer ones are deposited
on independent concept records.** Some April-9 results papers in this
project remain available as `historical` records — they were never
formally superseded by Zenodo, only deprioritized by the repo README.
The list of current canonical citations is always
[`publications.json`](publications.json). If you discover a stray DOI
in public-facing material that does not match the registry, treat the
registry as authoritative and open a correction.
