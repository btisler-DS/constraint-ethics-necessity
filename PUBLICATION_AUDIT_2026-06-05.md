# Publication Audit & Hygiene Policy — 2026-06-05

This document records the publication-registry cleanup performed on
2026-06-05 and establishes the pre-publication check rule that follows
from it. It is intended both as a historical record (so future
contributors understand WHY this registry exists) and as a forward-looking
policy (so the kind of confusion this audit cleaned up does not recur).

**The rule, in one sentence:** before any publication list, external
document, website page, or outreach material that mentions a project DOI
is sent or published, every DOI in it must be checked against
[`publications.json`](publications.json) in this repository, and only
records carrying `external_citation_status: "cite_this"` may be used as
current external citations.

---

## 1. Executive Summary

Before the audit, the public-facing list of project publications had
multiple errors that had been propagating silently:

- A withdrawn Zenodo DOI (`10.5281/zenodo.19485721`, HTTP 410 Gone) was
  cited as the live external DOI for two different papers simultaneously.
- Protocol 4 and Protocol 5 cited their April-9 records, while the repo
  README had since deposited and started citing newer May-20 records.
- Protocol 6 cited only its April-9 record, even though a May-20
  Version-2 deposit existed and was Zenodo-formally newer (same concept
  record).
- No machine-readable registry of project publications existed, so
  there was no programmatic way to detect drift, withdrawals, or
  attribution errors before they reached the public.

The audit landed five files in this repository as PR #1 (tag
[`publications-registry-v1.1.0`](https://github.com/btisler-DS/constraint-ethics-necessity/releases/tag/publications-registry-v1.1.0),
merge commit `03deecc77f966241d52733b1dbedbc3f77a2a9d3`):

| File | Purpose |
|---|---|
| `publications.json` | Canonical 20-record machine-readable registry |
| `scripts/check_publications.js` | DOI checker — schema + live doi.org + Zenodo API + title match |
| `publication-check-report.md` | Latest checker output (regenerate on every change) |
| `PUBLICATION_CORRECTIONS.md` | Append-only changelog of corrections |
| `README.md` | Updated to add Virtue Theater DOI, the Protocol 6 Results section, and pointers to the registry |

This document (`PUBLICATION_AUDIT_2026-06-05.md`) is the sixth artifact
of the cleanup. It is added in a follow-up PR after Bruce broadened the
scope of the pre-publication check rule.

---

## 2. What Went Wrong: Inventory of the Confusion

Each of the items below is a confirmed problem at the start of the audit
(2026-06-05). All are now fixed in the registry.

### 2.1 One withdrawn DOI, two misattributions

The public list cited `10.5281/zenodo.19485721` as the DOI for **both**:

- "Virtue Theater (Protocol 2 results)"
- "Protocol 3 Results"

Independently verified against the Zenodo records API: that DOI returns
**HTTP 410 Gone** — the underlying record was withdrawn from Zenodo and
no longer exists. So the public list pointed two different papers at
the same nonexistent record.

The correct DOIs:

- **Virtue Theater (P2 Results):** `10.5281/zenodo.19485645`,
  published 2026-04-09, title "Virtue Theater: Specification Gaming and
  Regulatory Constraint Failure in Multi-Agent Systems."
- **Protocol 3 Results:** `10.5281/zenodo.20312682`, published
  2026-05-20, title "Enforcement Opacity Increased Query Behavior in a
  Constrained MARL System: Protocol 3 Results."

The withdrawn DOI is retained in the registry as
`internal_id: withdrawn-19485721`, with
`zenodo_formal_status: "withdrawn"` and
`external_citation_status: "do_not_cite"`. It is kept so the broken
DOI is *visible* in the registry — if it ever resurfaces in a new
draft, the checker will surface it instead of letting it slip through.

### 2.2 Protocols 4, 5, 6 had unacknowledged later records

The April-9 records for these protocols still resolve on Zenodo, but
the repo README cites the newer May-20 records (deposited 2026-05-20)
as the current results. The public list was carrying the April-9 DOIs,
which had become stale relative to the README without being formally
"withdrawn."

The relationship between the April-9 and May-20 records differs by
protocol, in a way that matters for citation hygiene:

| Protocol | April-9 DOI | May-20 DOI | Zenodo-formal relationship |
|---|---|---|---|
| **P4** | `10.5281/zenodo.19485699` | `10.5281/zenodo.20314828` | Independent concept records (`19485698` vs `20314827`). Zenodo does NOT mark either as a version of the other. |
| **P5** | `10.5281/zenodo.19485713` | `10.5281/zenodo.20314078` | Independent concept records (`19485712` vs `20314077`). Same — Zenodo does NOT link them. |
| **P6** | `10.5281/zenodo.19485185` | `10.5281/zenodo.20313340` | **Same concept record** (`19485184`). Zenodo formally marks them as Version 1 (`is_last: false`) and Version 2 (`is_last: true`). |

For P4 and P5, the April-9 records are NOT formally superseded by the
May-20 records — they are independent Zenodo deposits. The registry
captures Bruce's editorial decision to prefer the May-20 records via
`external_citation_status: "cite_this"` on the May-20 records and
`"historical"` on the April-9 records. It does NOT misuse Zenodo's
"superseded" vocabulary.

For P6, the registry can and does use Zenodo's own versioning: the
May-20 record is `version_status: "current_version"`, the April-9
record is `version_status: "previous_version"`, and the
`replaces_doi` / `replaced_by_doi` fields point at each other. This
distinction matters because someone resolving the Zenodo concept DOI
`10.5281/zenodo.19485184` will land on the May-20 record automatically;
for P4 and P5, no such concept-DOI shortcut exists.

### 2.3 No registry → no programmatic check

Even after the errors above were fixed by hand, there was no
machine-readable canonical source — so the next time someone copy-pasted
a publication list, the same class of error could recur. The registry
and checker exist precisely to make that not happen.

---

## 3. What Was Done

### 3.1 Registry (PR #1, commits `3d54391` → `dd1dd94`)

`publications.json` was created with 20 records: 11 protocol records
(prereg + results for Protocols 2, 3, 4, 5, 6, 8 where applicable, plus
the P2 build report and the P2 Virtue Theater results paper), 4
standalone papers (DAS White Paper, Documentary Accountability Gap,
Formal Proof of the Minimal Interrogative Basis, Formal Framework for
Reasoning Stability, From Collective Computation to Developmental
Inquiry), and the withdrawn `19485721` marker.

Each record carries four independent status fields:

- `zenodo_formal_status` (`active` | `withdrawn`)
- `version_status` (`current_version` | `previous_version` | `only_version`)
- `repo_canonical` (`true` | `false` | `null`)
- `external_citation_status` (`cite_this` | `historical` | `do_not_cite` | `undetermined`)

These dimensions are deliberately separate. Collapsing them would
re-introduce ambiguity: "withdrawn on Zenodo," "Zenodo-formal Version
2," "currently cited by README," and "preferred for new external
citations" are four different questions, and the registry needs to
answer all four without conflating them.

### 3.2 Checker (commit `4970766`)

`scripts/check_publications.js` is a Node 18+ script that:

- Validates `publications.json` against the schema (vocabulary, required
  fields, internal consistency rules).
- Hits `https://doi.org/{doi}` for every record (HEAD, redirect-follow)
  and records HTTP status + final URL.
- Hits the Zenodo records API for every record and records HTTP status,
  title, and withdrawal state.
- Compares the registry title against the Zenodo title (token Jaccard
  ≥ 0.4) and flags mismatches.
- Cross-checks: a record with `zenodo_formal_status: "withdrawn"` must
  actually return HTTP 410; a record returning 200 must not be marked
  withdrawn; a `do_not_cite` record cannot be `repo_canonical: true`;
  a `previous_version` record cannot be `external_citation_status:
  "cite_this"`.
- Exits non-zero on any failure, so the script can be wired into CI if
  desired.

`publication-check-report.md` is the rendered output. **It is
regenerated on every checker run** — if `publications.json` changes,
the report must be regenerated and committed alongside the change.

### 3.3 P6 canonical resolution (PR #1 sixth commit, `9709e10`)

Initially the registry flagged Protocol 6 canonical status as
`undetermined` because the README did not list a P6 results DOI. Bruce
resolved this on 2026-06-05: the May-20 deposit (`20313340`) is the
current canonical record (it contains both
`Protocol6_Results_Paper.pdf` and
`Protocol6_Confirmatory_Artifacts.zip`); the April-9 record (`19485185`)
is retained as the historical Version 1.

The registry schema was bumped from `1.0.0` to `1.1.0` in the same
commit to accommodate the new `version_status` field and the rename
from `zenodo_status` to `zenodo_formal_status`.

### 3.4 Tag and merge

PR #1 was merged into `main` via merge commit, preserving all six
step-commits. The state of `main` at that point is tagged
`publications-registry-v1.1.0` and pushed to origin.

### 3.5 Cleanup audit (this document, PR #2 — `publication-hygiene-policy` branch)

This audit document and the broadened pre-publication check rule are
the cleanup pass that closes the loop on the initial registry work.

---

## 4. The Rule Going Forward

### 4.1 What the rule applies to

The pre-publication check applies to **any** material that mentions a
project DOI. Explicitly:

- README files in any Bruce-owned repository
- Project website pages — `quantuminquiry.org`, `das-demo-pi.vercel.app`,
  `tracestack-demo.vercel.app`, and any successor sites
- Outreach material — emails, blog posts, conference submissions, talk
  slides, social posts (LinkedIn, X, etc.)
- Funding and grant applications — FAR documents, NSF/NIH/DARPA
  submissions, internal/foundation proposals
- Co-authorship correspondence and pre-publication drafts
- Any other external citation list, including ones that appear "just
  this once"

The rule does **not** distinguish "important" from "casual" channels.
The original error was on a quick reuse-what-I-cited-last-time list;
that's exactly the class of situation where the check is most needed.

### 4.2 The pre-publication checklist

Before sending or publishing any material with a project DOI:

1. **Refresh the working clone.** Ensure `btisler-DS/constraint-ethics-necessity`
   is checked out at a recent commit on `main`. `git pull` if needed.
2. **Run the checker.** From the repo root:

   ```bash
   node scripts/check_publications.js
   ```

   It must exit with `All checks passed.` If it doesn't, fix the
   registry before continuing.

3. **For every DOI in the draft material, find its record in
   `publications.json`.** Search by the DOI string. If a draft DOI is
   not in the registry, STOP — add it to the registry first (see § 4.3).
4. **Confirm `external_citation_status: "cite_this"`.** Any other value
   means the DOI is not the preferred external citation:
   - `historical` — earlier related record; cite the current record
     instead.
   - `do_not_cite` — withdrawn or otherwise broken; never use.
   - `undetermined` — canonical status is pending; pause and resolve
     before publishing.
5. **Use the registry's `title` field for the citation text.** Drift
   between what the draft says and what the registry says is a smell
   worth fixing (it usually means either the draft is using a stale
   title or the registry needs updating).
6. **Check the verification date.** The registry's
   `last_full_verification` field tells you when the checker last ran
   against live Zenodo. If it's more than ~30 days stale, re-run the
   checker before relying on the registry.

### 4.3 Adding or correcting records

If a draft DOI is not yet in the registry, or the registry is wrong:

1. Branch from `main` in `btisler-DS/constraint-ethics-necessity`
   (e.g., `publication-add-<short-name>` or
   `publication-correct-<short-name>`).
2. Edit `publications.json`. Use the structure of an existing record as
   a template. Set all four status fields consciously; do not default
   to `cite_this` without reason.
3. Run `node scripts/check_publications.js`. It must pass.
4. Commit the registry change AND the regenerated
   `publication-check-report.md` together.
5. Add an entry to `PUBLICATION_CORRECTIONS.md` if the change fixes a
   prior error (not needed for routine additions of new papers).
6. Push the branch, open a PR for review.
7. Only after the PR merges should the new/corrected DOI appear in
   external-facing material.

### 4.4 What to do if you find a stray DOI in published material

If something is already public and uses a stale or wrong DOI:

1. **Don't quietly edit and pretend nothing happened.** Add a
   correction entry to `PUBLICATION_CORRECTIONS.md` describing the
   finding (where it appeared, what was wrong, what it should say).
2. Update the external material (website, document, etc.) to use the
   correct DOI from the registry.
3. If the wrong DOI is a withdrawn record, the urgency is highest —
   the link is dead, not just stale.

### 4.5 Authority hierarchy

When references disagree, follow this order:

1. **Zenodo records API** — authoritative for `zenodo_formal_status`
   and Zenodo-formal `version_status`.
2. **`publications.json` in this repo** — authoritative for the
   project's `repo_canonical` and `external_citation_status` decisions.
3. **The repo `README.md`** — should match the registry. If it
   doesn't, the registry wins and the README is a bug.
4. **Any external document** — never authoritative. Always verify
   against the registry.

---

## 5. Open Work (Cross-Repo Sweep — Deferred From PR #1)

PR #1 was deliberately scoped to `btisler-DS/constraint-ethics-necessity`
only. The same DOI-hygiene check still needs to happen across the rest
of Bruce's publication surface. None of the work below has started; all
of it should be done as separate, scoped PRs (one per location), each
of which:

1. greps the target for `zenodo\.\d+`,
2. cross-checks every hit against `publications.json`,
3. replaces stale or wrong DOIs with the canonical ones,
4. records the corrections in `PUBLICATION_CORRECTIONS.md` (in this repo).

Pending locations:

- [ ] `btisler-DS/tracestack-demo` — `public/topology.html` had a
      placeholder Zenodo DOI link for the DAS white paper as of
      2026-05-27. Should now point to
      `10.5281/zenodo.19369623`.
- [ ] `btisler-DS/dynamic-cross-origin-constraint` — parent study repo.
      Likely cites the constraint-ethics-necessity protocols somewhere.
- [ ] `quantuminquiry.org` — website pages, especially anything listing
      project publications.
- [ ] `das-demo-pi.vercel.app` — DAS pipeline demo site. If it has any
      citation panels, sweep them.
- [ ] `tracestack-demo.vercel.app` — production demo site. Likely
      references the DAS white paper and possibly other artifacts.
- [ ] Dropbox `Network Cloud Storage/Project/2025/Trinex HDR/Current Work/6-26/TraceStack/`
      — local working documents; sweep for stale DOIs before any of
      them are sent externally.
- [ ] FAR documents and any grant material citing project DOIs.
- [ ] Any LinkedIn or external-platform posts that linked to a Zenodo DOI.

A natural follow-up workflow: clone each target, do the grep + cross-check
in one pass, then open a single PR per repo (or a single tracked task
list for external sites that aren't in git).

---

## 6. References

- [`publications.json`](publications.json) — the canonical registry.
- [`scripts/check_publications.js`](scripts/check_publications.js) —
  run before any publication-touching work.
- [`publication-check-report.md`](publication-check-report.md) —
  most-recent checker output.
- [`PUBLICATION_CORRECTIONS.md`](PUBLICATION_CORRECTIONS.md) —
  append-only changelog of corrections (now including this audit's
  resolutions).
- Tag [`publications-registry-v1.1.0`](https://github.com/btisler-DS/constraint-ethics-necessity/releases/tag/publications-registry-v1.1.0) —
  the state of `main` immediately after the registry landed.
- PR #1 — `https://github.com/btisler-DS/constraint-ethics-necessity/pull/1`

---

*This document should be updated when the cross-repo sweep in § 5
makes meaningful progress, when the schema changes in a way that
affects the checklist, or when a new correction reveals a gap in the
process. Routine additions of new publications do not require updating
this document — they go in the registry and (if correcting a prior
error) in `PUBLICATION_CORRECTIONS.md`.*
