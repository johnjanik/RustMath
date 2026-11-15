export const meta = {
  name: 'handoff-followups-s1-redteam',
  description: 'Finish unit S1 (symbolic series engine): a revision for review 7, red-team rounds with four lenses before each formal review, then implement, adversarially review with fix-ups, and commit and push if approved',
  phases: [
    { title: 'Design fix', detail: 'revision 7 for review 7, fixes after each red-team round' },
    { title: 'Red team', detail: 'four attackers (numeric layer, memory and ownership, series arithmetic, input structure and recursion)' },
    { title: 'Design review', detail: 'independent formal review at the fixed bar' },
    { title: 'Implement', detail: 'port the prototype, tests with PARI/GP reference values' },
    { title: 'Review', detail: 'adversarial code review, up to 3 fix-up rounds' },
    { title: 'Commit', detail: 'commit and push if approved, or park as a patch' },
  ],
}
const REPO = '/home/user/RustMath'
const S = (args && args.scratch) || 'SET_args.scratch'
const D = `${S}/design`
const I = `${S}/impl`
const BRANCH = 'claude/gracious-gates-e6b6l7'
const CONTEXT = `Repository ${REPO}, branch ${BRANCH}. The user asked: "Fix the open follow-ups in HANDOFF" (docs/HANDOFF.md, "Open follow-ups"). The list was split into 13 units by crate, designed and adversarially design-reviewed (designs in ${D}/<unit>.md, written against HEAD c79a582, the body as revised is authoritative; reviews in ${D}/<unit>.review<r>.md). Units are implemented one at a time in this order: I3, I1, I2, P1, F1, R1, M1a, M1b, S1, MF1, C1, J1a, J1b; each approved unit is committed before the next starts, so HEAD now contains the earlier units of this round (\`git -C ${REPO} log --oneline c79a582..HEAD\`). Implementation reports of earlier units: ${I}/<unit>.md; their "Handoffs" sections may hand items to later units. At c79a582 the workspace was test-green (14,173 passed, 0 failed, 330 ignored), and at 922024b (14,386 passed, 0 failed, 333 ignored). Since then F1, R1, M1a, M1b, C1, J1a and J1b were committed (4b7f034..36c2170), the trackers updated (0c2f368), and MF1 committed (f29c492). S1 is the last unit. A full \`cargo test --workspace\` may be running in the main checkout (debug profile): a debug cargo command may block on the build lock for a while; wait, never kill it.`
const RULES = `Project rules (CLAUDE.md): honesty over coverage (no facades; missing functionality returns Err / documented None / unimplemented!; certifying routines distinguish Decided from Unresolved; a bounded search never decides); consume rustmath-core traits and existing crates (no new private num-* wrappers); cite the MAGMA chapter (digests in rustmath-lava/docs/) and the paper in new file headers; test against reference values from PARI/GP (/usr/bin/gp) or GAP (/usr/bin/gap) with the command in a comment. Never weaken an existing assertion; a test asserting buggy behaviour may be corrected only with a reference value proving it wrong (say so in the report). Zero unsafe. Keep crates build-green and test-green. Prose in docs and comments: no em-dashes, short plain sentences. Do not edit PROJECT_TRACKER.md, docs/*, docs/magma/* (the orchestrator updates trackers at the end). Do not commit, push, stash, reset or checkout (except the park step, which is told so). Never leave rustmath-misc/output.txt or .tmp_dir_* changes. Disk is limited (about 20 GB free; delete large binaries you build in scratch dirs when done; incremental builds are disabled in ~/.cargo/config.toml): never set CARGO_TARGET_DIR or create other target dirs; release builds only for the timing checks that need them. Several S1 design agents may run at the same time and share ${REPO}/target: during design work never delete target/release or other shared build output (the orchestrator cleans it); an implementer deletes ${REPO}/target/release when done with it. A single shell command may run at most 10 minutes in the foreground: run longer builds/tests in the background and wait for them. "Blocking waiting for file lock" means another build is running: wait. Do not run the full \`cargo test --workspace\` (the orchestrator runs it after all units); run the design's acceptance commands instead.`
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

const S1_BAR = `Review bar for unit S1 (the reviser, the attackers and the reviewer get the same text). A blocker is (a) a wrong, guessed or fabricated result, or an undecided answer reported as decided; (b) a regression against HEAD on an input HEAD handles (a crash, an abort or a stack overflow included); or (c) a budget hole: an input of at most 100 KB of text that, through \`taylor_with_budget\` with the REPL's budget (or another public entry point with its own budget, such as \`series_expansion\` or the \`limit_via_series\` path), runs longer than 20 s in release, uses more than 512 MB, or overflows the stack of an 8 MB thread in release or of a 2 MB thread in debug. Anything else (a constant factor, a stale figure, a rate slower than stated but within the bar) is a minor issue.`
const PROTO_RULE = `Keep the prototype in step with the design: the design's first lines must name the prototype directory that implements the current revision, its build script, and how to time one call through each public entry point (taylor_with_budget, series_expansion with to_expr, the limit_via_series path), with peak memory. Attackers and reviewers measure that prototype.`
const u = { key: 'S1-symbolic' }
const EXTRA7 = `Fix both blockers of review 7 as it requires (erf charged by an argument- and precision-dependent bound or replaced by a cheap rigorous enclosure for large arguments, with a scan of every enclosed function at 64, 256 and 1024 bits; SeriesExpansion keeps shared Arc coefficients, or every owned copy is charged by its size, measured through the public entry points) and its minors. Keep the "Invariants" and caps tables current. Re-run the reproduction families of reviews 3 to 7 (their scripts are under ${D}/${u.key}/review*/ and rev*/) on the updated prototype and record the outcomes in section 5.3. ${PROTO_RULE}
${S1_BAR}`
const LENSES = [
  { key: 'numeric', text: 'the rigorous numeric layer: every enclosed function (exp, log, sqrt, sin, cos, tan, atan, asin, acos, sinh, cosh, tanh, asinh, acosh, atanh, erf, powers, radicals) at every precision the driver reaches, over arguments near each guard, huge and tiny exponents, rationals with long mantissas and Ziv retries; the precision escalation of the zero test and its grouping rules; the cost charged against the measured cost' },
  { key: 'memory', text: 'memory and ownership across every boundary: copies from the engine\'s internal structures into SeriesExpansion, Expr, Display text, the limit_via_series path and the REPL taylor arm; the Arc sharing actually achieved; long names, long literals, many terms; peak memory (VmHWM) against the charged units, measured through the public entry points, not only the prototype\'s internal renderer' },
  { key: 'series', text: 'the series arithmetic: products (dense and sparse paths), division and inversion, composition, powers with large, negative or rational exponents, the exp/log/trig/erf recurrences, high orders up to TAYLOR_ORDER, high pole orders and wide exponent ranges, very sparse and very dense inputs, coefficient growth; the charged cost against the measured time of each operation' },
  { key: 'structure', text: 'input structure and recursion: deep, wide and DAG inputs, long monomials, many kernels, many radicals, nested kernels, huge literals, the parser, simplify, Drop, Display and PartialEq of results, every recursion of the engine against the 2 MB debug and 8 MB release stacks; and the caps table: is every cap checked before the work it bounds, and does every loop, allocation and recursion appear in the Invariants section' },
]
const HOLES = { type: 'object', properties: { holes: { type: 'array', items: { type: 'object', properties: { title: { type: 'string' }, input: { type: 'string' }, measured: { type: 'string' }, over_bar: { type: 'boolean' }, section: { type: 'string' }, fix: { type: 'string' } }, required: ['title', 'input', 'measured', 'over_bar', 'section', 'fix'] } }, held: { type: 'array', items: { type: 'string' } }, summary: { type: 'string' } }, required: ['holes', 'held', 'summary'] }

function attackP(l, round) { return `${CONTEXT}
${RULES}
You are a RED-TEAM attacker of the design for unit S1, ${D}/${u.key}.md (read all of it, including the Revision sections, the "Invariants" section and the caps table), and of its prototype (the directory the design names at the top). Your lens: ${l.text}.
Find inputs that break the design at the review bar below. Reproduce each one with the prototype (release build; the REPL's budget TAYLOR_WORK unless the entry point has its own; a timeout and an address-space cap on every run), measure time, charged units, peak memory and the outcome, and check correctness on a sample against gp (/usr/bin/gp). Report only holes you reproduced. For each: whether it is over the bar, the design section, and a fix that closes its class. Also list the families you tried that held. Stay within your lens; another attacker covers each other lens.
Do not edit the design or any repository file. Scratch dir ${D}/${u.key}/redteam${round}-${l.key}/. Write your findings to ${D}/${u.key}.redteam${round}-${l.key}.md. Return the structured result.
${S1_BAR}` }
function rtFixP(round, over, all) { return `${CONTEXT}
${RULES}
You revise the design for unit S1, ${D}/${u.key}.md, and its prototype after red-team round ${round}. The four attackers' reports are ${D}/${u.key}.redteam${round}-*.md. Holes over the bar (fix every one; close the class, not only the instance; then re-run each reproduction on the updated prototype and record the numbers in section 5.3):
${JSON.stringify(over, null, 1).slice(0, 16000)}
Holes below the bar (fix where cheap; otherwise record them as known costs or in section 7):
${JSON.stringify(all.filter(h => !h.over_bar).map(h => h.title + ': ' + h.measured), null, 1).slice(0, 4000)}
Add "Revision 7, red-team round ${round}" at the top listing each hole and the change, keep the body one consistent design, and update the Invariants and caps tables. Experiments in ${D}/${u.key}/rev7rt${round}/. Do not edit repository files. ${PROTO_RULE}
${S1_BAR}
Return a short summary.` }

async function redTeam(round) {
  const res = await parallel(LENSES.map(l => () => agent(attackP(l, round), { label: `redteam${round}:${l.key}`, phase: 'Red team', schema: HOLES })
    .then(r => ((r && r.holes) || []).map(h => ({ ...h, lens: l.key })))))
  return res.filter(Boolean).flat()
}
async function redTeamLoop(first, rounds) {
  let lastOver = null
  for (let round = first; round < first + rounds; round++) {
    const holes = await redTeam(round)
    const over = holes.filter(h => h.over_bar)
    log(`S1 red team round ${round}: ${holes.length} holes, ${over.length} over the bar`)
    lastOver = over
    if (!over.length) return { clean: true, round }
    await agent(rtFixP(round, over, holes), { label: `rtfix${round}:S1`, phase: 'Design fix' })
  }
  return { clean: false, lastOver: (lastOver || []).map(h => h.title) }
}

phase('Design fix')
await agent(`${dReviseP(u, 7)}\n${EXTRA7}`, { label: 'drevise7:S1', phase: 'Design fix' })
const rt1 = await redTeamLoop(1, 3)
let v = await agent(`${dReviewP(u, 8)}\nRed-team rounds ran between review 7 and this review; their reports are ${D}/${u.key}.redteam*.md, and the design's Revision 7 sections list the fixes. Check their holes are closed and look for others.\n${S1_BAR}`, { label: 'dreview8:S1', phase: 'Design review', schema: VERDICT })
let reviews = 8
if (!(v && v.approve)) {
  await agent(`${dReviseP(u, 8)}\n${PROTO_RULE}\n${S1_BAR}`, { label: 'drevise8:S1', phase: 'Design fix' })
  await redTeamLoop(4, 1)
  v = await agent(`${dReviewP(u, 9)}\nRed-team rounds ran between reviews; their reports are ${D}/${u.key}.redteam*.md, and the design's Revision sections list the fixes.\n${S1_BAR}`, { label: 'dreview9:S1', phase: 'Design review', schema: VERDICT })
  reviews = 9
}
if (!(v && v.approve)) {
  return { unit: u.key, designApproved: false, reviews, redTeam: rt1, blockers: ((v && v.issues) || []).filter(i => i.severity === 'blocker').map(i => i.item + ': ' + i.problem.slice(0, 400)) }
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
  return { unit: u.key, designApproved: true, reviews, approved: false, parked: p, last }
}
const c = await agent(commitP(u), { label: `commit:${u.key}`, phase: 'Commit', effort: 'low' })
return { unit: u.key, designApproved: true, reviews, approved: true, commit: c, minors: ((last && last.issues) || []).filter(x => x.severity === 'minor') }
