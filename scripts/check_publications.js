#!/usr/bin/env node
// Verifies every DOI in publications.json against Zenodo + doi.org.
// Writes publication-check-report.md. Exits non-zero on registry/world mismatch.
//
// Usage:
//   node scripts/check_publications.js
//   node scripts/check_publications.js --no-network   (skip live checks, validate schema only)
//
// Requires Node 18+ (uses global fetch).

const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const REGISTRY_PATH = path.join(ROOT, 'publications.json');
const REPORT_PATH = path.join(ROOT, 'publication-check-report.md');

const args = new Set(process.argv.slice(2));
const NO_NETWORK = args.has('--no-network');

const ALLOWED_DOC_TYPES = new Set([
  'preregistration', 'results_paper', 'white_paper', 'framework_paper',
  'proof', 'working_paper', 'build_report', 'withdrawn_record',
]);
const ALLOWED_ZENODO_FORMAL = new Set(['active', 'withdrawn']);
const ALLOWED_VERSION_STATUS = new Set(['current_version', 'previous_version', 'only_version']);
const ALLOWED_REPO_CANONICAL = new Set([true, false, null]);
const ALLOWED_EXT_CITATION = new Set(['cite_this', 'historical', 'do_not_cite', 'undetermined']);

function recIdFromDoi(doi) {
  const m = (doi || '').match(/zenodo\.(\d+)$/);
  return m ? m[1] : null;
}

function normalize(s) {
  return (s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function tokenSet(s) {
  return new Set(normalize(s).split(' ').filter(t => t.length > 3));
}

function jaccard(a, b) {
  if (a.size === 0 || b.size === 0) return 0;
  const inter = new Set([...a].filter(x => b.has(x))).size;
  const union = new Set([...a, ...b]).size;
  return inter / union;
}

function titleSimilar(registryTitle, zenodoTitle) {
  const A = tokenSet(registryTitle);
  const B = tokenSet(zenodoTitle);
  return jaccard(A, B) >= 0.4;
}

function validateSchema(pub, i) {
  const errs = [];
  const required = [
    'internal_id', 'title', 'short_title', 'document_type', 'doi',
    'zenodo_url', 'zenodo_formal_status', 'version_status', 'repo_canonical',
    'external_citation_status', 'publication_date', 'last_verified_date',
  ];
  for (const k of required) {
    if (!(k in pub)) errs.push(`missing field "${k}"`);
  }
  if (pub.document_type && !ALLOWED_DOC_TYPES.has(pub.document_type)) {
    errs.push(`document_type "${pub.document_type}" not in vocabulary`);
  }
  if (pub.zenodo_formal_status && !ALLOWED_ZENODO_FORMAL.has(pub.zenodo_formal_status)) {
    errs.push(`zenodo_formal_status "${pub.zenodo_formal_status}" invalid`);
  }
  if (pub.version_status && !ALLOWED_VERSION_STATUS.has(pub.version_status)) {
    errs.push(`version_status "${pub.version_status}" invalid`);
  }
  if (!ALLOWED_REPO_CANONICAL.has(pub.repo_canonical)) {
    errs.push(`repo_canonical "${pub.repo_canonical}" invalid (must be true/false/null)`);
  }
  if (pub.external_citation_status && !ALLOWED_EXT_CITATION.has(pub.external_citation_status)) {
    errs.push(`external_citation_status "${pub.external_citation_status}" invalid`);
  }
  if (pub.external_citation_status === 'do_not_cite' && pub.repo_canonical === true) {
    errs.push(`internal inconsistency: do_not_cite but repo_canonical=true`);
  }
  if (pub.zenodo_formal_status === 'withdrawn' && pub.external_citation_status === 'cite_this') {
    errs.push(`internal inconsistency: zenodo withdrawn but external_citation_status=cite_this`);
  }
  if (pub.version_status === 'previous_version' && pub.external_citation_status === 'cite_this') {
    errs.push(`internal inconsistency: previous_version should not be cite_this (use historical)`);
  }
  if (!recIdFromDoi(pub.doi)) {
    errs.push(`DOI "${pub.doi}" does not match zenodo.NNN pattern`);
  }
  return errs.map(e => `[${pub.internal_id || `index ${i}`}] ${e}`);
}

async function fetchSafe(url, opts = {}) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 15000);
  try {
    const r = await fetch(url, { ...opts, signal: controller.signal, redirect: 'follow' });
    return r;
  } finally {
    clearTimeout(t);
  }
}

async function checkOne(pub) {
  const result = {
    internal_id: pub.internal_id,
    doi: pub.doi,
    registry_title: pub.title,
    registry_zenodo_formal_status: pub.zenodo_formal_status,
    registry_version_status: pub.version_status,
    registry_external_citation: pub.external_citation_status,
    doi_resolve_status: null,
    doi_final_url: null,
    zenodo_api_status: null,
    zenodo_title: null,
    title_match: null,
    world_says_withdrawn: null,
    issues: [],
  };

  if (NO_NETWORK) return result;

  // 1. doi.org resolution
  try {
    const r = await fetchSafe(`https://doi.org/${pub.doi}`, { method: 'HEAD' });
    result.doi_resolve_status = r.status;
    result.doi_final_url = r.url;
  } catch (e) {
    result.doi_resolve_status = 'error';
    result.issues.push(`doi.org error: ${e.message}`);
  }

  // 2. Zenodo API record
  const recid = recIdFromDoi(pub.doi);
  if (recid) {
    try {
      const r = await fetchSafe(`https://zenodo.org/api/records/${recid}`);
      result.zenodo_api_status = r.status;
      if (r.status === 200) {
        const data = await r.json();
        result.zenodo_title = data?.metadata?.title ?? null;
        result.world_says_withdrawn = false;
      } else if (r.status === 410) {
        result.world_says_withdrawn = true;
      } else if (r.status === 404) {
        result.world_says_withdrawn = null;
        result.issues.push(`Zenodo API returned 404 — record not found`);
      } else {
        result.issues.push(`Zenodo API returned unexpected status ${r.status}`);
      }
    } catch (e) {
      result.zenodo_api_status = 'error';
      result.issues.push(`Zenodo API error: ${e.message}`);
    }
  }

  // 3. Title comparison
  if (result.zenodo_title && pub.title && !pub.title.startsWith('(unknown')) {
    result.title_match = titleSimilar(pub.title, result.zenodo_title);
    if (!result.title_match) {
      result.issues.push(
        `title mismatch — registry: "${pub.title.slice(0, 80)}…" vs Zenodo: "${result.zenodo_title.slice(0, 80)}…"`
      );
    }
  }

  // 4. Cross-check withdrawal claim
  if (result.world_says_withdrawn === true && pub.zenodo_formal_status !== 'withdrawn') {
    result.issues.push(`Zenodo says withdrawn but registry has zenodo_formal_status="${pub.zenodo_formal_status}"`);
  }
  if (result.world_says_withdrawn === false && pub.zenodo_formal_status === 'withdrawn') {
    result.issues.push(`Registry says withdrawn but Zenodo returns 200`);
  }

  return result;
}

function renderReport(registry, schemaIssues, results) {
  const date = registry.last_full_verification ?? '(unset)';
  const total = results.length;
  const withIssues = results.filter(r => r.issues.length > 0);
  const ok = total - withIssues.length;

  let md = '';
  md += `# Publication Check Report\n\n`;
  md += `Generated by \`scripts/check_publications.js\` against \`publications.json\`.\n`;
  md += `Registry \`last_full_verification\`: \`${date}\`.\n`;
  if (NO_NETWORK) md += `\n> Network checks were SKIPPED (\`--no-network\`). Only schema validation ran.\n`;
  md += `\n## Summary\n\n`;
  md += `- Schema validation errors: **${schemaIssues.length}**\n`;
  md += `- Records checked: **${total}**\n`;
  md += `- Records OK: **${ok}**\n`;
  md += `- Records with issues: **${withIssues.length}**\n\n`;

  if (schemaIssues.length > 0) {
    md += `## Schema validation\n\n`;
    for (const s of schemaIssues) md += `- ${s}\n`;
    md += `\n`;
  }

  md += `## Per-record results\n\n`;
  md += `| internal_id | DOI | doi.org | Zenodo API | Withdrawn? | Title match | Issues |\n`;
  md += `|---|---|---|---|---|---|---|\n`;
  for (const r of results) {
    const w = r.world_says_withdrawn === null ? '—' : (r.world_says_withdrawn ? 'yes' : 'no');
    const tm = r.title_match === null ? '—' : (r.title_match ? '✓' : '✗');
    const dRes = r.doi_resolve_status ?? '—';
    const zRes = r.zenodo_api_status ?? '—';
    md += `| \`${r.internal_id}\` | \`${r.doi}\` | ${dRes} | ${zRes} | ${w} | ${tm} | ${r.issues.length || '—'} |\n`;
  }

  md += `\n## Issues detail\n\n`;
  if (withIssues.length === 0) {
    md += `_No issues detected at the per-record level._\n`;
  } else {
    for (const r of withIssues) {
      md += `### \`${r.internal_id}\` — ${r.doi}\n\n`;
      for (const i of r.issues) md += `- ${i}\n`;
      md += `\n`;
    }
  }

  md += `\n## Method notes\n\n`;
  md += `- DOI resolution: HTTP HEAD against \`https://doi.org/{doi}\` with redirect following; the final URL is the resolved landing page.\n`;
  md += `- Zenodo record lookup: GET \`https://zenodo.org/api/records/{recid}\`. HTTP 410 indicates a withdrawn record. The registry must mark such records with \`zenodo_formal_status: "withdrawn"\` and \`external_citation_status: "do_not_cite"\`.\n`;
  md += `- Title comparison: token-set Jaccard ≥ 0.4 on lowercased, stop-word-filtered tokens. Coarse by design — flags substantive renames, tolerates punctuation drift.\n`;
  md += `- Schema validation runs even with \`--no-network\` and covers field presence, vocabulary, and a few internal consistency rules (e.g. \`do_not_cite\` records cannot also be \`repo_canonical: true\`).\n`;
  return md;
}

async function main() {
  if (!fs.existsSync(REGISTRY_PATH)) {
    console.error(`ERROR: ${REGISTRY_PATH} not found`);
    process.exit(2);
  }
  let registry;
  try {
    registry = JSON.parse(fs.readFileSync(REGISTRY_PATH, 'utf8'));
  } catch (e) {
    console.error(`ERROR: publications.json is not valid JSON: ${e.message}`);
    process.exit(2);
  }
  const pubs = registry.publications || [];

  // Schema pass
  const schemaIssues = [];
  pubs.forEach((p, i) => schemaIssues.push(...validateSchema(p, i)));
  if (schemaIssues.length > 0) {
    console.log(`Schema issues: ${schemaIssues.length}`);
    for (const s of schemaIssues) console.log(`  ${s}`);
  } else {
    console.log(`Schema OK (${pubs.length} records).`);
  }

  // Network checks
  const results = [];
  for (const pub of pubs) {
    process.stdout.write(`  ${pub.internal_id.padEnd(48)} ${pub.doi.padEnd(28)} `);
    const r = await checkOne(pub);
    results.push(r);
    if (NO_NETWORK) {
      console.log('(skipped)');
    } else {
      const label = r.issues.length === 0 ? 'OK' : `${r.issues.length} issue(s)`;
      console.log(label);
    }
    // be polite
    if (!NO_NETWORK) await new Promise(res => setTimeout(res, 150));
  }

  const md = renderReport(registry, schemaIssues, results);
  fs.writeFileSync(REPORT_PATH, md, 'utf8');
  console.log(`\nReport written: ${REPORT_PATH}`);

  const recordIssues = results.reduce((n, r) => n + r.issues.length, 0);
  const total = schemaIssues.length + recordIssues;
  if (total > 0) {
    console.log(`\nFAILED — ${schemaIssues.length} schema + ${recordIssues} record issues.`);
    process.exit(1);
  }
  console.log(`\nAll checks passed.`);
}

main().catch(e => {
  console.error(e);
  process.exit(2);
});
