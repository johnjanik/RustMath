export const meta = {
  name: 'handoff-followups-implement',
  description: 'Implement the 13 approved HANDOFF follow-up designs one unit at a time (integers, polynomials, finitefields, rings, matrix, symbolic, manifolds, combinatorics, REPL): design fix-ups where needed, implement, adversarial review with fix-ups, commit and push',
  phases: [
    { title: 'Design fix', detail: 'extra design rounds for C1, MF1, S1 at the current HEAD' },
    { title: 'Implement', detail: 'one unit at a time, tests with PARI/GP/GAP reference values' },
    { title: 'Review', detail: 'adversarial code review, up to 3 fix-up rounds' },
    { title: 'Commit', detail: 'commit and push each approved unit, or park an unapproved one as a patch' },
  ],
}
const REPO = '/home/user/RustMath'
// Pass args = { scratch: '<a scratch dir holding design/ and impl/ copied from this folder>' }.
// Units listed in args.done (default: the four committed on 2026-11-13) are skipped.
const S = (args && args.scratch) || 'SET_args.scratch'
const D = `${S}/design`
const I = `${S}/impl`
const BRANCH = 'claude/gracious-gates-e6b6l7'
const UNITS = [
  { key: 'I3-integers-arith', deps: [], fixDesign: false },
  { key: 'I1-integers-primality', deps: ['I3-integers-arith'], fixDesign: false },
  { key: 'I2-integers-factorization', deps: ['I1-integers-primality'], fixDesign: false },
  { key: 'P1-polynomials', deps: [], fixDesign: false },
  { key: 'F1-finitefields', deps: ['I2-integers-factorization'], fixDesign: false },
  { key: 'R1-rings-padics-category', deps: [], fixDesign: false },
  { key: 'M1a-matrix-rcf-bound', deps: [], fixDesign: false },
  { key: 'M1b-matrix-forms', deps: ['M1a-matrix-rcf-bound'], fixDesign: false },
  { key: 'S1-symbolic', deps: [], fixDesign: true, lastReview: 3 },
  { key: 'MF1-manifolds', deps: ['S1-symbolic'], fixDesign: true, lastReview: 3 },
  { key: 'C1-combinatorics', deps: [], fixDesign: true, lastReview: 3 },
  { key: 'J1a-repl-parser-and-small', deps: ['I2-integers-factorization'], fixDesign: false },
  { key: 'J1b-repl-values-and-limits', deps: ['J1a-repl-parser-and-small'], fixDesign: false },
]
const CONTEXT = `Repository ${REPO}, branch ${BRANCH}. The user asked: "Fix the open follow-ups in HANDOFF" (docs/HANDOFF.md, "Open follow-ups"). The list was split into 13 units by crate, designed and adversarially design-reviewed (designs in ${D}/<unit>.md, written against HEAD c79a582, the body as revised is authoritative; reviews in ${D}/<unit>.review<r>.md). Units are implemented one at a time in this order: I3, I1, I2, P1, F1, R1, M1a, M1b, S1, MF1, C1, J1a, J1b; each approved unit is committed before the next starts, so HEAD now contains the earlier units of this round (\`git -C ${REPO} log --oneline c79a582..HEAD\`). Implementation reports of earlier units: ${I}/<unit>.md; their "Handoffs" sections may hand items to later units. At c79a582 the workspace was test-green (14,173 passed, 0 failed, 330 ignored).`
const RULES = `Project rules (CLAUDE.md): honesty over coverage (no facades; missing functionality returns Err / documented None / unimplemented!; certifying routines distinguish Decided from Unresolved; a bounded search never decides); consume rustmath-core traits and existing crates (no new private num-* wrappers); cite the MAGMA chapter (digests in rustmath-lava/docs/) and the paper in new file headers; test against reference values from PARI/GP (/usr/bin/gp) or GAP (/usr/bin/gap) with the command in a comment. Never weaken an existing assertion; a test asserting buggy behaviour may be corrected only with a reference value proving it wrong (say so in the report). Zero unsafe. Keep crates build-green and test-green. Prose in docs and comments: no em-dashes, short plain sentences. Do not edit PROJECT_TRACKER.md, docs/*, docs/magma/* (the orchestrator updates trackers at the end). Do not commit, push, stash, reset or checkout (except the park step, which is told so). Never leave rustmath-misc/output.txt or .tmp_dir_* changes. Disk is limited (about 11 GB free; incremental builds are disabled in ~/.cargo/config.toml): never set CARGO_TARGET_DIR or create other target dirs; release builds only for the timing checks that need them, then delete ${REPO}/target/release. A single shell command may run at most 10 minutes in the foreground: run longer builds/tests in the background and wait for them. "Blocking waiting for file lock" means another build is running: wait. Do not run the full \`cargo test --workspace\` (the orchestrator runs it after all units); run the design's acceptance commands instead.`
function scope(u) { return `Unit ${u.key}. Design: ${D}/${u.key}.md (all "Revision" sections apply; the revised body is authoritative). Reviews: ${D}/${u.key}.review*.md; the last one approved the design: fold in its minor notes where they are correct and cheap, and list in your report the ones you skipped and why. Also read the earlier units' reports in ${I}/ for handoffs addressed to this unit, and adapt the design to the code as it is now at HEAD (earlier units of this round changed it; check the designs' assumptions about other units).` }
function testCmd(u) { return `the acceptance commands in the Plan section of ${D}/${u.key}.md (latest revision), except a full \`cargo test --workspace\`; always including \`cargo check --workspace --tests\`` }

const VERDICT = { type: 'object', properties: { approve: { type: 'boolean' }, issues: { type: 'array', items: { type: 'object', properties: { item: { type: 'string' }, severity: { type: 'string', enum: ['blocker', 'minor'] }, problem: { type: 'string' }, required_change: { type: 'string' } }, required: ['item', 'severity', 'problem', 'required_change'] } }, summary: { type: 'string' } }, required: ['approve', 'issues', 'summary'] }
const CREVIEW = { type: 'object', properties: { approve: { type: 'boolean' }, tests_pass: { type: 'boolean' }, issues: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, line: { type: 'integer' }, severity: { type: 'string', enum: ['blocker', 'minor'] }, problem: { type: 'string' }, required_change: { type: 'string' } }, required: ['file', 'severity', 'problem', 'required_change'] } }, summary: { type: 'string' } }, required: ['approve', 'tests_pass', 'issues', 'summary'] }

function dReviseP(u, rPrev) { return `${CONTEXT}
${RULES}
You revise the design for unit ${u.key}, ${D}/${u.key}.md. Its last independent review (${D}/${u.key}.review${rPrev}.md) did not approve it; read that review in full and fix every blocker and the right minor issues. The code at HEAD may differ from c79a582 because earlier units landed: re-check the design's facts against HEAD. Edit only the design file (add "Revision ${rPrev}" at the top listing each issue and the change, keep the body one consistent design). Do not edit repository files; experiments in ${D}/${u.key}/rev${rPrev}/. Return a short summary.` }
function dReviewP(u, r) { return `${CONTEXT}
${RULES}
You are an INDEPENDENT ADVERSARIAL REVIEWER of the design for unit ${u.key}: ${D}/${u.key}.md (read all of it, including every "Revision" section), at the current HEAD. Do not edit the design or any repository file; scratch dir ${D}/${u.key}/review${r}/. Check by reproducing: every claimed defect at HEAD; every HANDOFF item of the unit covered or genuinely out of scope; the algorithms correct (re-derive the key steps, check citations and the MAGMA digest semantics); Decided vs Unresolved where a search is bounded; no facade or silent truncation; every workspace caller found and behaviour changes handled; tests can fail on HEAD and use external reference values (run a sample of the gp/GAP commands); cost claims hold (measure a sample); the blockers of the previous review (${D}/${u.key}.review${r - 1}.md) are resolved. Write ${D}/${u.key}.review${r}.md. Approve only with no blockers. Return the structured result.` }
function implP(u) { return `${CONTEXT}
You are implementing ${scope(u)}
${RULES}
If uncommitted changes already exist (\`git -C ${REPO} status --short\`), they are from an interrupted earlier attempt at THIS unit (the container can restart): inspect them, keep what is right and continue. Implement the design in its Plan order. Add the design's tests with reference values (command in a comment). Run ${testCmd(u)} and make every target pass. Write a report to ${I}/${u.key}.md (mkdir -p): per item what changed (file:line), tests and reference sources, API and behaviour changes and the callers checked, deviations from the design with evidence, final test counts, anything left undone and why, and a "Handoffs" section for later units or the trackers. Return a short summary.` }
function reviewP(u, r) { return `${CONTEXT}
You are an adversarial code reviewer for ${scope(u)}
Implementation report: ${I}/${u.key}.md. Read \`git -C ${REPO} diff\` and \`git -C ${REPO} status --short\` (every uncommitted change belongs to this unit; read new files in full) and the surrounding code. Do not edit files, commit, stash or checkout; scratch dir ${I}/${u.key}.rev${r}/.
${RULES}
Check: every item of the design is implemented and fixes the defect for every input class (work examples; verify reference values yourself with gp/GAP); no path returns an undecided result as decided (a composite as prime, an unsplit block, a truncated argument, a guessed value); no new unbounded loop or allocation reachable from public APIs; no weakened or deleted assertions; tests can fail on the old code and use external reference values; behaviour changes and all workspace callers handled; cost claims hold (measure a sample); no unsafe; sources cited; docs accurate. Run ${testCmd(u)} yourself.
Write ${I}/${u.key}.review${r}.md. Approve only with no blockers and passing tests. Return the structured result.` }
function fixupP(u, rv, r) { return `${CONTEXT}
Continue ${scope(u)}
Report ${I}/${u.key}.md; the uncommitted changes are in place. The reviewer did not approve (round ${r}; ${I}/${u.key}.review${r}.md):
${JSON.stringify(rv, null, 1).slice(0, 14000)}
Fix every blocker and the right minor issues; explain disagreements with evidence in the report.
${RULES}
Run ${testCmd(u)}; append "Fix-up round ${r}" to the report. Return a short summary.` }
function commitP(u) { return `In ${REPO} (branch ${BRANCH}), commit the approved changes of unit ${u.key} and push. Units run one at a time, so every modified or new file in \`git -C ${REPO} status --short\` belongs to this unit (the report ${I}/${u.key}.md lists its files). Stage them with \`git add -- <paths>\` (never -A or .), except rustmath-misc/output.txt, .tmp_dir_* and obvious scratch files (list any you leave out); compare \`git diff --cached --stat\` with the report. If the unit spans several crates, you may make one commit per crate in dependency order, each building on its own.
Write each commit message to a file and commit with \`git commit -F <file>\`: a subject line under 72 characters naming the crate and the fix, then a BLANK LINE, then short bullet lines from the report (what was wrong, what changed, tests and reference sources, behaviour changes). No em-dashes. No model names. End with exactly these two lines after a blank line:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017iwyw8n2vtERdA6tWcBE9D
Push: \`git -C ${REPO} push -u origin ${BRANCH}\`; on network failure retry up to 4 times (2, 4, 8, 16 s). Never force-push. Return the commit hashes and the push result.` }
function parkP(u) { return `In ${REPO} (branch ${BRANCH}), unit ${u.key} was NOT approved after the review rounds. Park its changes so later units start from a clean HEAD: (1) \`git -C ${REPO} add -N\` every untracked file the unit created (not rustmath-misc/output.txt or .tmp_dir_*), then save \`git -C ${REPO} diff HEAD --binary > ${I}/${u.key}.parked.patch\` and check the file is non-empty and lists those files; (2) restore the tree: \`git -C ${REPO} reset -q\`, \`git -C ${REPO} checkout -- .\`, and delete exactly the untracked files you listed (nothing else); (3) confirm \`git -C ${REPO} status --short\` is empty and that \`git -C ${REPO} apply --check ${I}/${u.key}.parked.patch\` succeeds. Do not commit or push. Return the patch path and the file list.` }

const done = new Set((args && args.done) || ['I3-integers-arith', 'I1-integers-primality', 'I2-integers-factorization', 'P1-polynomials'])
const results = []
for (const u of UNITS) {
  if (done.has(u.key)) continue
  const missing = u.deps.filter(d => !done.has(d))
  if (missing.length) { log(`${u.key}: skipped, depends on unfinished ${missing.join(', ')}`); results.push({ unit: u.key, skipped: missing }); continue }
  if (u.fixDesign) {
    let ok = false, prev = u.lastReview
    for (let k = 0; k < 2; k++) {
      await agent(dReviseP(u, prev), { label: `drevise${prev}:${u.key}`, phase: 'Design fix' })
      const v = await agent(dReviewP(u, prev + 1), { label: `dreview${prev + 1}:${u.key}`, phase: 'Design fix', schema: VERDICT })
      prev = prev + 1
      if (v && v.approve) { ok = true; break }
    }
    if (!ok) { log(`${u.key}: design still not approved; skipped`); results.push({ unit: u.key, designApproved: false }); continue }
  }
  await agent(implP(u), { label: `impl:${u.key}`, phase: 'Implement' })
  let approved = false, last = null
  for (let r = 1; r <= 4; r++) {
    const rv = await agent(reviewP(u, r), { label: `review${r}:${u.key}`, phase: 'Review', schema: CREVIEW })
    last = rv
    if (rv && rv.approve) { approved = true; break }
    if (r === 4) break
    await agent(fixupP(u, rv, r), { label: `fixup${r}:${u.key}`, phase: 'Review' })
  }
  if (!approved) {
    const p = await agent(parkP(u), { label: `park:${u.key}`, phase: 'Commit', effort: 'low' })
    log(`${u.key}: NOT approved after 4 reviews; parked: ${String(p).slice(0, 160)}`)
    results.push({ unit: u.key, approved: false, parked: p, last })
    continue
  }
  const c = await agent(commitP(u), { label: `commit:${u.key}`, phase: 'Commit', effort: 'low' })
  done.add(u.key)
  log(`${u.key}: committed ${String(c).slice(0, 140)}`)
  results.push({ unit: u.key, approved: true, commit: c, minors: ((last && last.issues) || []).filter(i => i.severity === 'minor') })
}
return results