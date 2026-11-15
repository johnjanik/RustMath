# Review 1 of the Steiner labelling implementation (impl/SL.md)

Date: 2026-11-15. Reviewer: independent adversarial code review. Object: the uncommitted changes
in `/home/user/RustMath-steiner` (branch `claude/m23-steiner` on `d91f880`), 19 entries of
`git status --short` (10 modified files, 9 new files or directories), against the design
`design/SL.md` (with its review `design/SL.review1.md`) and the report `impl/SL.md`.
Scratch material of this review: `impl/SL.rev1/` (paths below relative to
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/sl/`).

No file of the worktree was edited. Before each cargo run I touched the `lib.rs` of the four
changed crates (mtime only, as the report's `cargo_sl.sh` does, because the target directory is
shared with the WP-F worktree). `git status --short` and `git diff --stat` are unchanged after the
review. `target/release` (created by my release run) was deleted afterwards.

## 0. Verdict

**Approve. No blocker, five minor issues (Section 8). All tests pass.**

I reproduced the six integers, the two rejected residues, the minimal-precision residues, the
second-prime values and all four degree 253 resolvents with my own PARI/GP code, written from
scratch. The duad resolvents were also reproduced by an exact resultant over Z, with no p-adic
arithmetic at all. The CC0 reference file was confirmed value by value. The decision logic has no
path that returns an undecided result as decided. The bounds are exact integer arithmetic and are
proven. Licence handling is correct. The measured times agree with the report.

## 1. Acceptance commands run

| Command | Result | Time |
|---|---|---|
| `cargo test -j 2 -p rustmath-finitefields` | all pass (lib 205, 12 integration files incl. `galois_ring_frobenius` 11, doc 8) | 32 s wall |
| `cargo test -j 2 -p rustmath-coding` | all pass (lib 67, `cyclic_golay` 7, doc 5) | 20 s |
| `cargo test -j 2 -p rustmath-combinatorics` | all pass (lib 1509, `designs_exact` 5, doc 189, 11 ignored) | 60 s |
| `cargo test -j 2 -p rustmath-galois` | all pass (lib 65, `steiner_labelling` 13 + 10 ignored, `descent` 12, others) | 75 s |
| total of the four crates | 2305 passed, 0 failed, 33 ignored | |
| `cargo test -j 2 --release -p rustmath-galois --test steiner_labelling --no-run` | built | 128 s |
| `... --release ... -- --ignored full_` | 2 passed | 1.62 s |
| `M23ISGALOIS=/home/user/external/m23isgalois ... --release ... -- --ignored env_ --test-threads 1` | 8 passed | 29.57 s |
| same without `M23ISGALOIS` | 8 FAILED, each "set M23ISGALOIS to a checkout of shaowuz/m23isgalois" | 0.05 s |
| fast set in release | 13 passed | 1.80 s |
| `cargo clippy -j 2 --no-deps -p <crate> --all-targets` (finitefields, coding, galois) and `-p rustmath-combinatorics --lib --test designs_exact -- --cap-lints warn` | no warning or error on a new or changed line (all warnings are on pre-existing lines: `galois_ring.rs:59`, `poly_factor.rs:95,121,814,985`, `designs.rs` lines below 958) | |
| `rustfmt --edition 2021 --check` on the eight new `.rs` files | clean | |

Debug fast set of `steiner_labelling`: 20.42 s of test time (report: 21.5 to 23.4 s). Logs:
`impl/SL.rev1/debug_*.log`, `rel_*.log`, `clippy.log`.

Times printed by the `env_` tests (release, one thread): heptad resolvent f2 2.00 s (N = 360),
f1 2.05 s (N = 395); duad resolvent f2 1.26 s (N = 267), f1 1.61 s (N = 302);
`subset_sum_resolvent` 0.68 s and 0.84 s. Report: 2.12, 2.32, 1.21, 1.61, 0.74, 0.82 s. The cost
claims hold.

## 2. Independent reproduction (PARI/GP 2.15.4, my own script)

`impl/SL.rev1/gp/myref.gp`, drivers `run_f.gp`, `run_res.gp`, `run_cc0.gp`, outputs `*.txt`. Run
with `gp -q -D parisizemax=4000000000 -f <driver> < /dev/null`. Method: heptads from the masks
`0xAE3`, `0xC75` by enumerating the 4096 words; the Frobenius image by Newton in
`Mod(Mod(., p^N), f)` with a Newton inverse; roots `alpha_{i+1} = alpha_i(phi(x))`; the checks
`f(y) = 0`, `y = x^p mod p`, `phi^23 = id`; **direct** sums and products over all 253 heptads (no
traces, no orbit compression, unlike the Rust code). Fujiwara's `best` by `sqrtnint`.

| Quantity | My gp | Rust (`env_` / fast tests) |
|---|---|---|
| f2 `best`, `B5`, `B_J5`, `B_J7` | 2157, 5778453176946, 7220058114, 97634149933 | same |
| f2 at 29, N = 30 | `T5- = -1059840`, `T5+ = -29622039743232607524836197957123800868867866` | same, Decided(-) |
| f2 J at N_J = 8 | `J5 = 61916`, `J7 = -83030` (the `+` system gives 110822631967, -170484025683, outside the bounds) | same |
| f2 N = 9 / 10 | `[-3209424281433, -1059840]` / `[-206309467943599, -1059840]` | Unresolved / Decided(-) |
| f2 at 223 | N = 19, `T5- = -1059840`, rejected 45 digits; N_J = 5, J5, J7 as at 29 | same |
| f1 `best`, `B5`, `B_J5`, `B_J7` | 3473, 62529476039478, 78129291179, 2738950710192 | same |
| f1 at 29, N = 31 | `T5- = 92747040`, `T5+ = -883496073094568376895161675988763338141335547` | same, Decided(-) |
| f1 J at N_J = 9 | `J5 = 524492`, `J7 = -2281968` | same |
| f1 N = 10 | `[-114831914638610, 92747040]` | Decided(-), same rejected |
| f1 at 73 | N = 24, `T5- = 92747040`; N_J = 7, J5, J7 as at 29 | same |
| f2 duad resolvent | exact `Res_x(f(x), f(Z - x)) / (2^23 f(Z/2))` is a square, its root equals `R253_duad` (2.9 s) | equal |
| f1 duad resolvent | same route, equal to `R253_duad` (3.3 s) | equal |
| f2 heptad resolvent | product of 253 linear factors at N = 360 (526-digit bound) equals `R253_heptad` (10.2 s) | equal |
| f1 heptad resolvent | same at N = 395 (578-digit bound), equal (11.3 s) | equal |
| CC0 monic associate | monic, integral, 1605-digit coefficients; raw `f` has content 1 and is irreducible over Q | |
| CC0 at 29, N = 270 | `T5-` and the rejected `T5+` equal the vendored file; `|T5+| > B5 >= |T5-|` | same |
| CC0 J at N_J = 344 | `J5`, `J7` equal the vendored file (3.7 s) | same |
| CC0 at 67, N = 216 | `T5+ = T5` of the file, rejected `T5-` equal | Decided(+) |
| CC0 signs at 73, 109, 389, 421, 439, 647, 811, 823, 859, 887 (N = 1) | exactly one match each; equal to the file and to the f1 rows of `frobenius_signs.json` | same (`cc0_frobenius_signs`, `full_cc0_frobenius_signs`) |

So the six integers and all four resolvents are reproduced by a third implementation, and the duad
resolvents by an exact method independent of every p-adic routine.

## 3. Mathematics and bounds

- **Fujiwara bound** (`RootBound::fujiwara`): `r_k` is the least integer with
  `r_k^k >= 256^k |a_{n-k}|` (floor root by `Integer::nth_root`, plus one unless exact), so
  `best / 256 >= mu` and `M = best / 128 >= 2 mu`. The proof in the doc is correct. It uses
  `|a_0|^{1/n}` (weaker than Fujiwara's `|a_0/2|^{1/n}`, still valid). Integer only.
- **Derived bounds**: `power_bound = floor(count (scale best)^e / 128^e) + 1` and
  `binomial_bound = floor((128 + scale best)^e / 128^e) + 1`, exact integer division of
  non-negative values. `B5 = 253 (7M)^5`, `B_J5 = 5313 M^5` (`J5 = sum_H e_5(alpha_H)`, 253 * 21
  terms), `B_J7 = 253 M^7`, `(1 + 7M)^253`, `(1 + 2M)^253`: all valid. My gp values agree.
- **Precision**: `precision_for` gives the least `N >= 1` with `p^N > 2 B 10^e`. J and the
  resolvents use `e = 0`; every value passes `bounded_integer` (refuses `m <= 2B`), and a value
  outside its bound is `Err(BoundExceeded)`. No float anywhere.
- **Decision** (`integrality_dichotomy` then the status map): exactly one pass is `Decided`, two
  passes is `Unresolved`, none is `NotInM23`. This is Proposition 5.1 of the paper (the paper's
  "the other residue exceeds the bound" is the same rule, since the true sign always passes). The
  `NotInM23` claim (a proof without the hypothesis) is right: any M23-conjugate containing the
  shift is `Aut(S+)` or `Aut(S-)` by `lem:twocodes`, whose premises (two distinct shift-invariant
  S(4,7,23)) are checked at run time by `golay_systems_checked`.
- **Orbit compression**: `phi^i(s_H) = s_{H+i}`; heptad and duad orbits have length 23; the trace
  of `w^j` is the power sum `P_j` by Newton's identities (no division, so `p = 2, 23` work). The
  duad representatives `{0, d}`, `d = 1..11`, cover the 253 pairs once.
- **Newton lift** (`FrobeniusMatrix::lift`): the order (u, then y with the new u, then u at the new
  y) doubles both accuracies; `lift_to` first lifts `u` alone to the current precision. The final
  checks `D(y) = 0`, `y = w^p mod p`, `phi^d(w) = w` are genuine.
- **Lazy product** (`mul_reduce`): top-down reduction with `c = r[k] mod m` is correct because
  `r[k] - c` is a multiple of `m`; the result is canonical. `with_modulus` normalization keeps the
  ideal (`D[d]` is a unit mod `p^a`); probe with `D[d] = -4` at `p = 5` stores `[12, 6, 1]` and the
  generator is a root of both (`SL.rev1/probe/probe.rs`).

## 4. No undecided result returned as decided

Walked every path and probed the edge cases (`probe/probe.rs`, run against the release rlibs):

| Case | Behaviour | Test |
|---|---|---|
| both within the bound | `Unresolved`; `t5()`, `sign()` `None`; `golay_invariants`, `heptad_resolvent`, `frobenius_sign` `Err(NotDecided)` | C23 (2, 3, with retries), f2 N = 9 |
| neither | `NotInM23`, same `Err(NotDecided)` | `x^23 - x - 1` at 23 and at 157 (N = 18) |
| not inert / not squarefree mod p | `Err(Frobenius(NotInertModP / NotSquarefreeModP))`, small and large `p` branches | CC0 at 11, `x^23 + 1` at 23, `refusals` |
| not monic, trailing zero | `NotMonic` | CC0 raw, probe |
| `p^N <= 2 B5` | `Err(Integrality(PrecisionTooLow))`, also for `N = 0` | C23 N = 39, f2 N = 8 |
| composite `p` | `Err(Frobenius(Prime))` | `p = 25` |
| false hypothesis, minimal margin | `Decided(-)`, documented in the module docs as relative to the trusted input | `false_hypothesis_can_decide`, probe with `extra_digits = 0` |
| `p` in `{0, 1}` or negative | **panic** in `precision_for` (m1) | probe |

The only defect found here is the panic of m1. It is not a facade.

## 5. Tests

- Fast tests: Golay systems (weights, `S(4,7,23)` exactly, 77/21/5, intersections, orbit
  representatives and heptads through `{0,1,2}` from golay.py), the exact design checker on four
  classical designs, the Frobenius lift against PARI/GP (`x^3 + x + 1` at `5^6`, `x^23 - x - 1` at
  `23^4`, `x^5 - x + 1` at `3^7`, two primes above `MAX_P`), the Teichmueller Frobenius as a second
  Frobenius, bounds (`x^23 - x - 1`, C23), the CC0 labelling with two references, `T5 = 120 J5 + S`
  (p-adic free). They assert exact external values, so they fail on wrong code. The sign
  assertions also catch an ordering bug (the design review showed `phi^{-1}` swaps the signs).
- `env_` tests: read f1, f2 at run time and check the files' fingerprint lines; they fail loudly
  without `M23ISGALOIS` (checked). They cover the six integers, the rejected sizes, the bounds,
  minimal precisions, all four resolvents (the duad one three ways), the second primes and the 80
  signs. The stability test is labelled as a cross-check only.
- No existing assertion was changed: the diff of existing files has no test hunk.

## 6. Licence and data

- `golay_resolvents_no_f.json`: re-derived here with the README command,
  sha256 `87123dfe...4f87a381`, original `dcf1e38c...70d13`; no `"f"` field. `frobenius_signs.json`
  `30ec4670...17a4f` and `lelay_specialization_t2_Z.json` `f775aa4c...45862` equal the sources
  (the latter is line 84 of Le Lay's `CHECKSUMS.sha256`). `cc0_t2_steiner_reference.json`
  `91532572...58dee` as stated; all its values confirmed by my gp (Section 2).
- All 158 distinct integers of at least six digits in `polynomial_f1.gp`, `polynomial_f2.gp` and
  `polynomial_F.m` were compared with all integers of at least six digits in every new or changed
  file (1759 distinct): no overlap. f1 and f2 are read only from `$M23ISGALOIS`.
- The README states that `R253_duad` still determines f1 and f2 (design review m4). Compliant with
  the brief, which allows vendoring the resolvents.
- `make_cc0_reference.py` imports the owner's `arith.py`; it does not copy it. Le Lay's CC BY-NC-SA
  paper and code are not used.

## 7. Other checks

- No `unsafe` in the new code. No em-dash in the new files or the added lines.
- Sources cited in every new file header: the paper sections, Fujiwara, Witt, MacWilliams-Sloane,
  McDonald, Wan, von zur Gathen-Gerhard, Stauduhar, Geissler-Klueners, and the MAGMA digests of
  ch. 38, 47, 48, 147, 152, which exist under `rustmath-lava/docs/`.
- No dependency cycle: `rustmath-coding` dev-depends on `rustmath-combinatorics`, which does not
  depend on `rustmath-coding`. Downstream users of `GaloisRing` (`rustmath-rings`, `-category`,
  `-algebras`) call only `GaloisRing::new`, whose behaviour is unchanged; the API only grew.
- Line numbers and counts quoted in the report match the files.
- The report's Section 7 (shared target hazard) is accurate; I worked around it the same way.

## 8. Issues

All minor. None blocks.

- **m1 (robustness): panic for `p < 2`.** `steiner_labelling` with `prime: Some(0)` or `Some(1)`,
  and `duad_resolvent(f, p)` with `p <= 1`, panic in `precision_for`'s assert
  (`steiner_labelling.rs:376`) before `FrobeniusRoots::new` can refuse the prime. Probe output:
  `precision_for: p = 0 < 2`, `p = 1 < 2`, `p = -23 < 2`. These public functions return `Result`,
  and their docs list `Frobenius` (prime not proven) as the error. Required change: check
  `p >= 2` (or prove `p` prime) before calling `precision_for` and return
  `Err(SteinerError::Frobenius(FrobeniusRootsError::Prime(..)))`; add the two cases to
  `refusals`.
- **m2 (cost): memory of `t_design_index` at its limit.** Measured (release): one block of 120
  points with `t = 4` (8,214,570 subsets, 12 times below `DESIGN_CHECK_LIMIT = 10^8`) takes 6.1 s
  and 1.13 GB peak RSS, about 140 bytes per subset (`HashMap<Vec<usize>, u64>`). At the limit that
  is about 14 GB, the whole RAM of this machine (15 GB). The doc states the insertions but not the
  memory. Required change: document the memory (about 140 bytes per distinct `t`-subset), and
  either lower the limit (for example `10^7`) or use a compact key (the rank of the subset, or
  `Vec<u32>`/a fixed array). The S(4,7,23) check (8855 subsets) is unaffected.
- **m3 (docs): costs of the public entry points.** The docs of `heptad_resolvent`,
  `duad_resolvent` (both forms) and `golay_invariants` do not say how the precision and the time
  scale with the root bound. For the CC0 associate the heptad resolvent needs N about 12,500 at 29,
  about 10 minutes by the design's estimate. Likewise `LabellingOptions::retries` (each retry adds
  10 digits to an always-Unresolved input such as C23), `extra_digits` and `precision` (`p^N` is
  built at once), and `search_limit` (a scan up to `fp_factor::MAX_P` with one factorization per
  prime). Required change: one cost line per item, using the design's Section 9 numbers (f1, f2:
  about 2 s per resolvent in release; CC0: N about 12,500).
- **m4 (reuse for WP-A): vacuous pass of an empty candidate.** `integrality_dichotomy` treats a
  candidate with an empty vector as passing (`all` over nothing). Probe:
  `[("empty", []), ("big", [400])]` gives `Decided { label: "empty", value: [] }`. Candidates of
  unequal lengths are also accepted. The labelling never builds such input, but WP-A A4 will pass
  coefficient vectors. Required change: refuse empty or unequal-length candidate vectors (a new
  `IntegralityError` variant) or document the behaviour, with a unit test.
- **m5 (nit, test comment).** In `orbit_polynomial_rejects_an_unstable_family` the comment
  "only the constant term fails... and w^2 is not constant" is muddled. Suggested: "(Z - w)(Z + w)
  = Z^2 - w^2: the constant term `-w^2` is not in Z/125."

## 9. Files of this review

`impl/SL.rev1/`: `gp/myref.gp` (the reference), `gp/run_f.gp`, `gp/run_res.gp`, `gp/run_cc0.gp`
and their outputs `run_f.txt`, `run_res.txt`, `run_cc0.txt`; `gp/vendored_res.gp` and
`gp/cc0_data.gp` (generated from the vendored files); `probe/probe.rs`, `probe/design_mem.rs`
(compiled with `rustc --edition 2021 --test -O` against the release rlibs, binaries deleted) and
`probe/probe_out.txt`; `hjlppz_ints.txt`, `new_ints.txt`, `cf.txt` (the licence grep);
`rederived.json` (the stripped file, re-derived); `run_debug.sh`, `run_release.sh` and the logs
`debug_*.log`, `rel_*.log`, `clippy.log`, `debug_summary.txt`, `rel_summary.txt`.
