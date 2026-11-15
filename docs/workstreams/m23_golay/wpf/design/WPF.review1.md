# Review 1 of the WP-F design (independent, adversarial)

Date: 2026-11-15. Object: `scratchpad/m23/design/WPF.md` (957 lines, no Revision section) and the
prototype `scratchpad/m23/design/WPF/proto`. Reviewer scratch: `scratchpad/m23/design/WPF/review1/`.
Nothing in the design, the prototype or the repository was edited. The prototype was copied to
`review1/proto` and extended with review commands (`pred`, `weil`, `tangent`). Its build directories
were deleted afterwards.

## Verdict

**Approve, with the decisions of Section 2 and the required changes of Section 3 (all minor).**

The mathematics is right. The node certificate is sufficient: I checked the proof step by step and
found no case where it decides wrongly on a correct input. The two headline corrections (the exact
counts at p = 5, and the corrected L row at p = 31) survive a check that the design does not use:
the counts at k = 5 and 6 are predicted exactly by the P computed from N_1..N_4, and the
reference values fail that check. One mathematical statement is false (`deg G <= 2`, Section 3.1
below), but its only effect is a refusal with a wrong message, never a wrong decision.

D1 is rejected for this unit (Section 2). That is a decision the design asked for, and the design
already specifies the fallback.

## 1. What I reproduced

All runs on the worktree crates at `0d277f0` (the prototype's path dependencies), release unless
stated, 4 cores shared with other agents (load 2 to 7).

| Claim | How I checked | Result |
|---|---|---|
| F and G are the same curve: `F(T, (967W-3)/(81-2267W)) (81-2267W)^23 = c G` | own gp script `mob_check3.gp`, F parsed directly from `polynomial_F.m`, G directly from Le Lay's JSON (not the designer's parses) | exact equality; `c = 2^23 3^23 131^23 > 0`; det `71526 = 2 3 7 13 131`; contents of F and G are 1 |
| Le Lay file is the CC0 one | `sha256sum`, `data/LICENSE`, `SOURCE_NOTES.md`, repo commit | `cfb4185b...`, CC0-1.0, commit `3f6ab4b` |
| designer's parses of F and calF | own regex parse of both `.m` files | identical to `work/coeffs_str.json` |
| X rows 7, 11, 13, 17, 19 (k <= 4) | `pred` on G7, G, G13 | equal to `zeta_X.json`; 62 nodes (13: 61 nodes + A2) |
| X rows 83, 107, 127 (k <= 3) | `pred` | equal to Table 7.1; 107 has 60 nodes + A3 with rational branches |
| p = 5: N = 7, 37, 169, 561, P Weil, mod 23 and mod 2 checks | `pred 5 6 G`, gp `l31.gp` | reproduced; see the k = 5, 6 check below |
| L row 31, s0 = 12: exact 43, 1017, 30265, 923729 | `pred 31 5 calF 12`, gp | reproduced; reference P fails mod 23 (`P(1) = 15`, `P(31) = 7`), exact P passes |
| anchor at 53 | `pred 53 2 calF s0`, s0 = 21, 36, 41 | N_1 = 52, 57, 45, each 62 nodes |
| refusals | `pred` at 3 (F, G), 23 (F, G), 131 (F), 13 (G), 7 (G) | 3 and 23: D = 0 identically for F and G; 131 (F) and 13 (G): fibre component; 7 (G): D = 0 |
| L at 5 and 11 degenerate | gp `idealprimedec` | (5, s - 3) is unramified of degree 1; (11, s - 2) is **ramified** (e = 2, f = 1). Index of Z[s] in O_L is 1, so s -> s0 is the residue map |
| small curves of 9.4 | `gp small_curves.gp`, `branch.gp`, prototype `small` | every PARI value reproduced, every prototype row OK |
| both fibrations and brute force | own gp brute force over P^1 x P^1(F_p) (`brute.gp`); `tfib` | plane counts 11, 15, 22, 17 at 7, 11, 17, 19 (F), 15 at 11 (G), all three methods agree; 133 at p = 11, k = 2 both fibrations |
| exact Weil test | `weil` command: 11 cases x 5 primes | all correct, including double roots on +-2 sqrt p, simple roots on the boundary, `4p +- 1` just inside and outside, a triple root, non-real roots, g = 1 and g = 2 |
| cost | `bench`, `oldff` | kernel 5.3 us per fibre (p = 31, k = 4, one thread); existing `FiniteField`/`FqPoly` 4.14 ms (k = 4) and 126 us (k = 1); p = 79, k = 4 in 17.9 s on 4 threads. Matches Section 8.1 |
| debug times of the fast set | debug build, one thread | G7 at 7 (k <= 4) 0.07 s, G at 11 (k <= 4) 0.31 s, G at 31 (k <= 3) 0.68 s, G13 at 13 (k <= 3) 0.15 s, G at 5 0.09 s, small curves 0.05 s. Matches Section 8.3 |

### The k = 5, 6 check (new evidence, not in the design)

If the counts N_1..N_4 are those of a smooth genus 4 curve, then P determines every N_k. So the
plane count at k = 5, 6 plus the branch corrections must equal `p^k + 1 - s_k(P)`. A wrong branch
rule breaks this. Results (`review1/proto`, command `pred`):

| row | rule | P from N_1..N_4 | N_5, N_6 predicted | N_5, N_6 counted | agree |
|---|---|---|---|---|---|
| X, p = 5 | reference (unibranch) | not Weil | 3268, 15090 | 3198, 15708 | no |
| X, p = 5 | exact (A3, conjugate) | Weil | 3197, 15709 | 3197, 15709 | **yes** |
| L, p = 31, s0 = 12 | reference | Weil (numerically and exactly) | 28598074 | 28604634 | no |
| L, p = 31, s0 = 12 | exact | Weil | 28604633 | 28604633 | **yes** |
| X, 7 (G7), 11 (G), 13 (G13) | both (equal) | Weil | k = 5, 6 | same | yes |
| X, 17, 19, 31, 41 | both (equal) | Weil | k = 5 | same | yes |

This is the strongest independent support for the two differences from the reference. It also
shows that the Weil test alone cannot catch the L-31 error: the reference P is a Weil polynomial.

## 2. Decisions

- **D1: rejected for this unit.** The scope states that a model with a non-nodal singular point
  gives its counts with status Unresolved, never Decided. Theorem 4.5 plus the exact classification
  is, in my reading, a valid proof. But the rule is explicit, and a reviewer cannot waive it. The
  delta-sum path also trusts a larger piece of code (the A_k and ordinary m-fold classifier) whose
  external tests cover only A2, A3 and one triple point. Use the design's own fallback:
  - `Certificate` has only `Nodal`. The delta-sum result becomes
    `Unresolved::DeltaSumCertified { delta_sum, points }`, with the reason text "non-nodal
    singularities classified exactly, delta-sum certified" and `BranchRule::Exact`.
  - Remove `CountOptions::allow_delta_certificate`. No option may produce Decided for a non-nodal
    model.
  - Keep all the computation and evidence. Add follow-up **F-9**: the owner decides whether
    `DeltaSumCertified` is promoted to Decided. Cite Theorem 4.5 and the k = 5, 6 check.
- **D2: accepted.** Vendor only Le Lay's CC0 G. Derive G7 and G13 at run time. Read F and calF only
  from `M23ISGALOIS`. The Moebius identity is verified exactly (Section 1).
- **D3: accepted.** Assert the reference value in reference mode and the corrected value in exact
  mode. Add the k = 5 evidence of Section 1 to the data README and to the report.
- **D4: accepted.** Kernel in `rustmath-finitefields` (commit 1, green on its own), the rest in
  `rustmath-curves` (commit 2), one branch.

## 3. Required changes

All are minor. None changes the architecture.

### 3.1 `deg G <= 2` is false (4.3 step 2)

`G = gcd(f, f_T, F_V(T, v0))` has degree up to 3. Degree 3 occurs when the fibre is
`c (T - tau)^4` and `F_V(T, v0)` vanishes to order 3 at tau: for example, a tacnode whose tangent
line is the fibre. Reproduced: `F = V^2 + (T^3 + 2T^4) V + 3T^4` at p = 11 has one A3 at (0, 0)
(PARI: the normalization `y^2 = 4T^4 + 4T^3 + T^2 - 12` has `a_p = 4`, so N_1 = 8, `review1/tangent.gp`).
The prototype refuses it with "3 singular points in one fibre", which is false: there is one
point. For a node, G has multiplicity exactly 1 at the node (if `F_TV = 0` then `F_TT != 0`), so
the Nodal certificate is unaffected. Degree 2 with zero discriminant, and degree 3, always mean a
non-nodal point.

Required: state `deg G <= 3`. Handle `deg G = 3` as one point (check `G = (T - tau)^3`) and pass it
to the local classifier. If anything else is left unhandled, return `Unresolved(Unclassified)`, not
an error with a false message. Add this curve to `biprojective_small_curves.rs` (expected N_1 = 8,
`P = T^2 - 4T + 11`, source: the gp script).

### 3.2 The genus is a trusted hypothesis

`normalization_point_counts(f, genus, ..)` decides relative to the `genus` argument. With a genus
below the true one, a bad prime can be certified. Example: curve A of 9.4 (smooth, genus 3) passed
with genus 2 at a prime where its reduction gets one node gives "Nodal, 1 = p_a - 2", which is
wrong. For X and the covers, g = 4 is a published theorem (HJLPPZ Prop. 3.10; Riemann-Hurwitz for
`(2A, 23A, 23B)`), so the M23 results stand.

Required: document it in the API doc. Record it in the status (for example
`Certificate::Nodal { nodes, genus_hypothesis }`). Cite the source of g at the M23 call sites.
Split `DeltaMismatch`: `found < p_a - g` contradicts the remark of 4.5 and means the genus input or
the code is wrong. Return `Err(CountError::GenusInconsistent { .. })` for it. `found > p_a - g`
stays `Unresolved` (bad reduction or bad model).

### 3.3 Add the k = g + 1, g + 2 consistency check

This is a cheap and strong test of the branch rules (Section 1). It is the only check besides
mod 23 that catches the L-31 error.

Required:
- a derived check `predicted_counts_agree(kmax)`, reported for every row where it is affordable;
- fast debug tests: X at p = 5 (G, k <= 6), p = 7 (G7, k <= 6), p = 11 (G, k <= 5). The exact rule
  agrees and the reference rule fails at 5;
- opt-in release tests: L at (31, 12) with k = 5 (14.5 s on 4 threads), X at 31 and 41 with k = 5
  (20 s and 88 s);
- the kernel and `normalization_point_counts` must support k up to 8, which they do by design.

### 3.4 Statuses and claims after D1

- Section 0 items 3 and 4, Section 7 (status column) and the 9.1 test expectations: 13, 31, 41, 73,
  5 and 107 for X, and L at (31, 12), (79, 15), (103, 93), become `Unresolved(DeltaSumCertified)`.
  Rephrase "p = 5 is decided" and "the D flags go away" accordingly. Keep the evidence.
- 3.3 item 1 (D flags): not removed under the rule. Record why. Item 1 in its "recount on a
  second model" sense is answered by F-2: the (8,23) model is already non-nodal over Q.
- 2-torsion order (4.10): `TwoTorsionOrder::Decided` depends on P only. The driver must report it as
  Decided only on a row whose count status is Decided. At 31 the order 15 then becomes conditional
  (the brief predicts 15 at 11, 31, 79). Rename the variant or document it, and propagate the
  status in the driver.

### 3.5 Inconsistencies in the text

- p = 23: Section 5 says "run and report", Section 7.1 and F-5 say "not run". Measured: F and G
  both have `D = 0` identically mod 23, so 23 is refused (`DiscriminantVanishes`). Section 0 item
  2 ("except 23 (not run)") and F-5 must say so. F-5 then needs another model.
- 4.3 step 5 says both two-point non-nodal cases "are covered by constructed tests". F-3 says the
  test over F_{p^2} is a follow-up. Pick one. Preferably add the constructed test now.
- 9.1 kernel tests: `p = 65521` with `K = 4` is impossible under the design's own bound
  (`65521^4 = 1.84e19 > 2^63`). Test K <= 3 for 65521 and add a `FieldTooLarge` test for K = 4.
  Test the const-generic path (degree <= 4) and the `Vec` path (degree 5, 6 and 23) separately.

### 3.6 Model selection in the M23 driver

"Reports the best status" needs an order: Nodal > DeltaSumCertified > NonNodal > Unclassified >
DeltaMismatch > refusal. When two models of the list both give counts at the same p, assert that
the counts are equal. They must be, since the models are isomorphic over Q and, at a good prime,
the counts are those of the reduction of X.

### 3.7 Small additions to the test plan

- One small curve with a node of residue degree 2: a (4,2) form with `b^2 - 4ac` divisible by
  `(T^2 - n)^2`, n a nonsquare, with the PARI value. This exercises the node branch rule
  `[2]` -> `[1,1]` over F_{p^2} outside the M23 data.
- The L row at 11: note in the README that (11, s - 2) is ramified (e = 2). The refusal is still
  right, since D vanishes identically there.
- Measure the debug test-harness compile of `rustmath-curves`. Its dependency tree includes
  symbolic, schemes, numberfields, numerical and rug. Report the time and the target size, since
  disk is limited.

## 4. Checked and found right

- **Algorithm against `curve.py` (`fibre="V"`).** Fibre over each v in P^1(F_q). Distinct roots
  in P^1(F_q), with a degree drop as one root at infinity. The fibre over V = infinity from the
  `V^23` coefficients. Frobenius orbits by minimal code, weighted by orbit length, the same code
  convention. Corrections from the certificate instead of the in-loop Hessian: equivalent, because
  the certificate finds every singular point (Lemma 4.3 plus the V = infinity place when
  `deg D < 138`, the T = infinity chart in every fibre) and the square class over F_q of a node of
  degree d is `+1` iff `chi = +1` or `k/d` is even. The formula
  `N_k = #X0(F_q) + sum_{d_P | k} d_P (b_P(k) - 1)` is right for nodes, A_k and ordinary points.
- **Node certificate (Theorem 4.5).** Each step holds: flatness of the Cartier divisor, finiteness
  of the normalization (R excellent), constancy of chi, R0 + S1 for `Y_p`, birationality of
  `Y_p -> X0_p`, `delta(Y_p) = 0`, smoothness, and connectedness by Zariski. The proof also
  shows that a reducible reduction (for example with a horizontal line) can never reach delta-sum
  exactly 62. The only way it decides wrongly is a wrong genus input (3.2). The completeness of the
  singular-point search, the residue degrees, the trace and norm square-class formulas
  (`Res(g, h)` with g monic: `factor_with_multiplicity` returns monic factors) and the two-point
  fibre logic are right.
- **Exact Weil test.** The reduction to `R` and `S(y) = E^2 - y O^2`, and the use of
  `count_real_roots_in` on (-1, 4p] for each square-free factor, are right. The boundary case is
  verified by computation.
- **Derived checks.** Mod 23 (same `chi(p)` at degree-one primes of L), mod 2 with all Q, and the
  2-torsion order (squarefree char poly implies semisimple, order = lcm of the orders of x;
  otherwise the bound `s * 2^ceil(log2 e_max)`).
- **Degenerate primes.** Every refusal I ran is a typed error. No bounded search decides anything:
  the classifier's precision cap gives Unclassified, and the nonsquare scan only finds a helper.
- **Licences.** F and calF are never copied. Tests read them from `M23ISGALOIS`, and the opt-in
  tests panic without it. G is CC0 with attribution, checksum and source commit. The JSON tables
  come from the owner's repository, as the brief allows. `work/F.gp` and `coeffs*.json` are scratch
  only. My own copy of F (`Fr.gp`) is deleted; `gen_inputs.py` regenerates it.
- **Tests can fail and use external values.** Reference values come from `zeta_X.json` and
  `zeta_L.json` (FLINT reference) and PARI/GP for the small curves (reproduced). The new rows are
  labelled new.
- **API.** No facade: missing cases are errors or Unresolved. Counts without a certificate carry
  `BranchRule` and an Unresolved status. With `unibranch_if_unresolved = false` the counts are
  empty. The plane-curve facades of `rustmath-curves` are not used.
- **Kernel justification.** The existing stack is about 780 times slower (measured). There is no
  word-size GF(p^k) in `rustmath-finitefields` today. The accumulation bounds (< 2^40 for the
  quartic path, a little more for degree 23, far below 2^64) hold.

## 5. Reproduction

In `scratchpad/m23/design/WPF/review1/`: `python3 -I gen_inputs.py`, then
`gp -q mob_check3.gp < /dev/null`, `gp -q l31.gp < /dev/null`, `gp -q brute.gp < /dev/null`,
`gp -q tangent.gp < /dev/null`, `gp -q small_curves.gp < /dev/null`. Then `cd proto && cargo build --release` (26 s) and
`./target/release/wpf_proto pred 5 6 G`, `pred 31 5 calF 12`, `pred 13 6 G13`, `weil`, `tangent`,
`small`, `bench 31 4` (with `RAYON_NUM_THREADS=1`), `oldff 31 4 200`. `pred_31_41.txt` holds the
k = 5 runs at 31 and 41. Delete `proto/target` afterwards (about 480 MB).
