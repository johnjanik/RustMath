# RustMath requests from the M23 project

Date: 2026-11-16. Written by the M23 session for the RustMath session.

This file supplements `docs/workstreams/m23_golay/HANDOFF.md`, which stays the specification of
the three work packages (WP-F, WP-P, WP-A). It holds three things the handoff does not:

1. new facts found since the handoff, some of which change its predictions (Section 1);
2. changes I would like in what already exists (Section 2);
3. what the exploratory work still needs from RustMath, in dependency order (Section 3).

Written against RustMath `main` at `28c4b69` and the M23 repository (`johnjanik/M23`) at `main`
`2d11038`. Two new scripts and one data file in the M23 repository back Section 1; they are on
its work branch `claude/blissful-knuth-qla7ym` (commit `2750a56`) and reach `main` when the owner
merges. Every
table they produce is copied into this file, so nothing here needs a fetch.

In this file a path that does not start with `rustmath-`, `docs/` or `PROJECT_TRACKER.md` is
relative to a checkout of the M23 repository, as in the handoff. The same text is kept there as
`docs/RUSTMATH_REQUESTS.md`.

Priorities: **P0** correctness of something already merged; **P1** blocks the next mathematical
result; **P2** wanted for exploration; **P3** stretch, do not start without asking.

## 0. Summary

| ID | Request | Priority | Size |
|---|---|---|---|
| C-1 | S1: refuse an argument that leaves a branch cut at the expansion point | P0 | small |
| C-2 | S1: bring `docs/HANDOFF.md`, the tracker and the `series.rs` conventions in line with what is merged | P0 | small |
| C-3 | S1: split the resource bar; correct doc comments that claim bounded consumers | P0 | small, needs the owner |
| C-4 | S1: run the crates downstream of `simplify.rs` and `expand.rs` | P0 | small |
| C-5 | `two_torsion_frobenius_order`: use the 2-rank of `Jac(F_p)` | P1 | small |
| C-6 | Upkeep of the M23 handoff (clone line, table 5.2, the "both block systems" case) | P1 | small |
| C-7 | JSON output of the Frobenius tables, one command | P1 | small |
| E-1 | Canonical model of X (handoff A0), on the CC0 form | P1 | medium |
| E-2 | The special fibre at 3 | P1 | medium |
| E-3 | Tritangent planes to `g15` and `g8` (handoff A1 to A5), with the sharper tests of Section 1 | P1 | large |
| E-4 | Check of Section 1.1 on the planes at p = 17 | P2 | small after E-3 |
| E-5 | WP-P: the 57 rows, then the twists | P1 | medium |
| E-6 | Degree 23 number fields: discriminant, decomposition, a specialization scan | P2 | medium |
| E-7 | Belyi maps of positive genus, with the test "recover the sextic field L" | P2 | large |
| E-8 | Periods and theta constants; the Schottky test for Question 10.2 | P3 | large |
| E-9 | A long trace table: `N_1` for all good p below 2^16 | P2 | small |
| E-10 | Point counts for fibre degree up to 8 and for curves in P^3 | P2 | medium |
| E-11 | Four branch points: braid orbits on Nielsen classes | P3 | large |
| E-12 | Functional equation of the L-function | P3 | not now |

## 1. New facts since the handoff

### 1.1 The image of Galois on Jac(X)[2] is the full stabilizer of W

Write `J[2] = Jac(X)[2]` and let `P` be the stabilizer of the Golay Lagrangian `W` in
`Sp(J[2]) = Sp_8(F_2)`. Then `P = R : GL(W)`, where `R` is the group of elements acting trivially
on `W` and on `J[2]/W`. With `J[2]/W = W^dual` through the Weil pairing, an element of `R` is
`1 + n` with `n` a symmetric tensor in `W (x) W`, so `R` is elementary abelian of order `2^10`.

**Fact.** The image of `Gal(Qbar/Q)` on `J[2]` is all of `P`, a group `2^10 : A8` of order
`2^10 * 20160 = 20643840`.

Proof from the existing table.

1. The image `G` maps onto `GL(W)` (paper, Theorem 4.8). `U = G n R` is normal in `G` and `R` is
   abelian, so `U` is a `GL(W)`-submodule of `R`.
2. At p = 17 the plane model is nodal (62 nodes), the counts are `21, 285, 4923, 82985`, and
   `P_17 mod 2 = (x+1)^2 (x^3+x+1)(x^3+x^2+1)`. The order of the Jacobian is
   `#Jac(X)(F_17) = P_17(1) = 98714 = 2 * 49357`. So `Jac(X)(F_17)[2]` has order 2: Frobenius
   minus 1 has a kernel of dimension 1 on the generalized eigenspace for the eigenvalue 1, which
   has dimension 2. That is one Jordan block of size 2.
3. Frobenius has order 7 on `W` and on `J[2]/W` (squarefree characteristic polynomials), and
   order 14 on `J[2]`. Its seventh power `u` lies in `U`, and `u - 1` has rank 1, so
   `u = 1 + w (x) w` for a nonzero `w` in `W`.
4. `GL(W)` is transitive on the nonzero `w`, and the fifteen tensors `w (x) w` span the symmetric
   tensors (`e_i (x) e_j + e_j (x) e_i = (e_i+e_j) (x) (e_i+e_j) - e_i (x) e_i - e_j (x) e_j`).
   So `U = R` and `G = P`.

The same holds at p = 73 and p = 127, where `#Jac(X)(F_p)` is again 2 times an odd number. The
rows at 17 and 73 were counted by both implementations.

Consequences, all new:

- `Q(J[2])` has degree `20643840` over Q and is an elementary abelian extension of degree `2^10`
  of the A8 field `K_W`.
- `W` is the only Galois-stable subspace of `J[2]` other than 0 and `J[2]`. In particular `J[2]`
  is not `W + W^dual` as a Galois module.
- For the handoff: in step A4 of WP-A exactly one of the two block systems `U`, `U'` is
  rational. The case "both, if `J[2]` splits" (5.3 A4 and the third item of 5.7) cannot happen.
- The 120 tritangent planes form one Galois orbit. The stabilizer of an octad acts on its 8
  planes as the affine group `AGL_3(2)` of order 1344, so each octad is an affine 3-space over
  `F_2`. The 16 even theta characteristics of `Theta_0` form one orbit, with group
  `AGL_4(2) = 2^4 : A8`; none of them is rational.

Status: a deduction from certified rows, checked at the level of linear algebra by
`scripts/jac2_frobenius_orbits.py`. It is not yet in the paper. RustMath can reproduce steps 2
and 3 today (request C-5) and test the consequence for the planes later (request E-4).

### 1.2 Sharper Frobenius predictions for `g8` and `g15`

`scripts/a8_frobenius_predictions.py` enumerates `GL_4(2)` in the model of
`data/pg32_a8_model.json` and writes `data/a8_frobenius_predictions.json`. The 14 classes:

| order | class size | characteristic polynomial on `W` | on 8 letters | on 15 points or hyperplanes | dim of fixed space |
|---:|---:|---|---|---|---:|
| 1 | 1 | `(x+1)^4` | `1^8` | `1^15` | 4 |
| 2 | 210 | `(x+1)^4` | `1^4 2^2` | `1^3 2^6` | 2 |
| 2 | 105 | `(x+1)^4` | `2^4` | `1^7 2^4` | 3 |
| 3 | 112 | `(x^2+x+1)^2` | `1^5 3` | `3^5` | 0 |
| 3 | 1120 | `(x+1)^2(x^2+x+1)` | `1^2 3^2` | `1^3 3^4` | 2 |
| 4 | 1260 | `(x+1)^4` | `4^2` | `1^3 2^2 4^2` | 2 |
| 4 | 2520 | `(x+1)^4` | `1^2 2 4` | `1 2 4^3` | 1 |
| 5 | 1344 | `x^4+x^3+x^2+x+1` | `1^3 5` | `5^3` | 0 |
| 6 | 1680 | `(x^2+x+1)^2` | `1 2^2 3` | `3 6^2` | 0 |
| 6 | 3360 | `(x+1)^2(x^2+x+1)` | `2 6` | `1 2 3^2 6` | 1 |
| 7 | 2880 | `(x+1)(x^3+x+1)` | `1 7` | `1 7^2` | 1 |
| 7 | 2880 | `(x+1)(x^3+x^2+1)` | `1 7` | `1 7^2` | 1 |
| 15 | 1344 | `x^4+x+1` | `3 5` | `15` | 0 |
| 15 | 1344 | `x^4+x^3+1` | `3 5` | `15` | 0 |

The cycle type on hyperplanes equals the one on points.

The rule used in 1.1 decides more. The 2-rank of `Jac(X)(F_p)` is the dimension of the kernel of
Frobenius minus 1 on `J[2]`, and it is at most `v = v_2(#Jac(X)(F_p)) = v_2(P_p(1))`. When
`v = 1` and `(x+1)^m` exactly divides `Q_p` with `m >= 1`, Frobenius has one Jordan block of size
`2m` for the eigenvalue 1 on `J[2]` and one of size `m` on `W` (the invariant subspaces of a
single block form a chain). That fixes the order on `J[2]` and, for `m >= 2`, the class on `W`.

All 30 primes of the two tables (this replaces the table in 5.2 of the handoff; p = 5 is new
there, and "RM" marks rows counted in RustMath only):

| p | `Q_p` up to reversal | `v` | order on `J[2]` | on 8 letters | on 15 | rows |
|---:|---|---:|---:|---|---|---|
| 5 | `(x+1)(x^3+x^2+1)` | 2 | 7 or 14 | `1 7` | `1 7^2` | both |
| 7 | `(x+1)(x^3+x^2+1)` | 5 | 7 or 14 | `1 7` | `1 7^2` | both |
| 11 | `x^4+x+1` | 0 | 15 | `3 5` | `15` | both |
| 13 | `(x+1)(x^3+x^2+1)` | 2 | 7 or 14 | `1 7` | `1 7^2` | both |
| 17 | `(x+1)(x^3+x^2+1)` | 1 | **14** | `1 7` | `1 7^2` | both |
| 19 | `(x+1)(x^3+x^2+1)` | 2 | 7 or 14 | `1 7` | `1 7^2` | both |
| 29 | `(x+1)(x^3+x^2+1)` | 3 | 7 or 14 | `1 7` | `1 7^2` | both |
| 31 | `x^4+x+1` | 0 | 15 | `3 5` | `15` | both |
| 37 | `(x+1)(x^3+x^2+1)` | 3 | 7 or 14 | `1 7` | `1 7^2` | both |
| 41 | `(x+1)^2(x^2+x+1)` | 2 | open | `1^2 3^2` or `2 6` | `1^3 3^4` or `1 2 3^2 6` | both |
| 43 | `(x+1)^2(x^2+x+1)` | 1 | **12** | **`2 6`** | **`1 2 3^2 6`** | both |
| 47 | `(x+1)(x^3+x^2+1)` | 2 | 7 or 14 | `1 7` | `1 7^2` | both |
| 53 | `(x+1)^2(x^2+x+1)` | 2 | open | `1^2 3^2` or `2 6` | `1^3 3^4` or `1 2 3^2 6` | both |
| 59 | `x^4+x^3+x^2+x+1` | 0 | 5 or 10 | `1^3 5` | `5^3` | both |
| 61 | `(x+1)^2(x^2+x+1)` | 3 | open | `1^2 3^2` or `2 6` | `1^3 3^4` or `1 2 3^2 6` | both |
| 67 | `x^4+x^3+x^2+x+1` | 0 | 5 or 10 | `1^3 5` | `5^3` | both |
| 71 | `x^4+x^3+x^2+x+1` | 0 | 5 or 10 | `1^3 5` | `5^3` | both |
| 73 | `(x+1)(x^3+x^2+1)` | 1 | **14** | `1 7` | `1 7^2` | both |
| 79 | `x^4+x+1` | 0 | 15 | `3 5` | `15` | both |
| 83 | `(x+1)(x^3+x^2+1)` | 2 | 7 or 14 | `1 7` | `1 7^2` | both |
| 89 | `(x+1)^4` | 1 | **8** | **`1^2 2 4`** | **`1 2 4^3`** | RM |
| 97 | `x^4+x+1` | 0 | 15 | `3 5` | `15` | both |
| 101 | `(x+1)^2(x^2+x+1)` | 4 | open | `1^2 3^2` or `2 6` | `1^3 3^4` or `1 2 3^2 6` | RM |
| 103 | `(x+1)^2(x^2+x+1)` | 1 | **12** | **`2 6`** | **`1 2 3^2 6`** | RM |
| 107 | `(x+1)^4` | 9 | open | one of the 5 unipotent types | | both |
| 109 | `(x+1)^4` | 6 | open | one of the 5 unipotent types | | RM |
| 113 | `(x+1)^2(x^2+x+1)` | 1 | **12** | **`2 6`** | **`1 2 3^2 6`** | RM |
| 127 | `(x+1)(x^3+x^2+1)` | 1 | **14** | `1 7` | `1 7^2` | RM |
| 131 | `(x+1)^2(x^2+x+1)` | 3 | open | `1^2 3^2` or `2 6` | `1^3 3^4` or `1 2 3^2 6` | RM |

Bold entries are decided by the rule above. `Q_p` is listed as the smaller of `Q` and `Q^rev`;
which of the two is the characteristic polynomial on `W` is still open (handoff 3.3 item 4).

### 1.3 Local facts at 3

The reduction of X at 3 is open (paper, Question 10.1). Two facts from the published M23 fields
of HJLPPZ, computed today with PARI/GP 2.15 (`nfdisc`, `idealprimedec`):

| Field | `t0` | `t0^2 + 23` | discriminant | 3 | 2 | 7 | 23 |
|---|---|---|---|---|---|---|---|
| `f1` | `2` | `3^3` | `2^36 3^18 23^30` | `e = 1, 2, 4, 8, 8`, all `f = 1` | `(e,f) = (3,1), (3,1), (1,1), (4,4)` | unramified | `e = 23` |
| `f2` | `-3703/947 = -7 * 23^2 / 947` | `2^11 3^6 23 / 947^2` | `2^44 7^8 23^24` | unramified, `f = 2, 7, 14` | `(e,f) = (3,1), (20,1)` | `(e,f) = (2,1), (2,7), (1,7)` | `e = 23` |

The branch points `0, sqrt(-23), -sqrt(-23)` stay distinct modulo 3, and `sqrt(-23)` lies in
`Q_3`. Both specializations are 3-adically close to a branch point of class 23A or 23B: one
factor of `t0^2 + 23` has valuation 3 for `f1` and 6 for `f2`. At a prime of good reduction of
the cover, the usual rule for specializations (Beckmann; read the exact hypotheses in the
source) gives inertia generated by the third or sixth power of a branch cycle of order 23, so
of order 23. What is observed: inertia at 3 is cyclic of order 8 for `f1` (cycle type
`1 2 4 8^2`, the class 8A) and trivial for `f2`. So the rule fails at 3 and the cover does not
have good reduction there in that sense, which is possible because 3 divides `|M23|`. This says
nothing yet about the curve X or its Jacobian; it is recorded so that E-2 is read with it.

### 1.4 The review of S1 at `28c4b69`

The two blockers of design review 9 are fixed, the 767 rows of the singular-point matrix pass,
and the whole `rustmath-symbolic` suite passes (874 passed, 0 failed, 32 ignored). One finding
remains, request C-1.

## 2. Changes wanted in what exists

### C-1 (P0) S1: an argument that leaves a branch cut

When the constant term of the argument lies on a branch cut and the argument has a coefficient
that takes it off the cut, the engine returns `Ok` with the formal series that gp gives. For a
real variable that series is the function on one side of the point, or on neither. Checked with
gp at 30 digits, `x0 = 1/100` and `x0 = -1/100`:

| Input at 0 | True value at `x0 = 1/100` | Series at `x0 = 1/100` | Error |
|---|---|---|---|
| `log(-1 - I*x^2)` | `5.0e-9 - 3.14149265 I` | `5.0e-9 + 3.14169265 I` | `2 pi I`, on both sides |
| `sqrt(-1 - I*x^2)` | `5.0e-5 - 1.0000000 I` | `-5.0e-5 + 1.0000000 I` | sign, on both sides |
| `log(-1 + I*x)` | `5.0e-5 + 3.13159299 I` | the same | none for `x0 > 0`; `2 pi I` at `x0 = -1/100` |
| `arctanh(2 + I*x^2)` | `0.54930614 + 1.57076299 I` | `0.54930614 - 1.57082966 I` | `pi I`, on both sides |
| `arcsin(3/2 + I*x^2)` | `1.57070688 + 0.96242366 I` | `1.57088577 - 0.96242366 I` | `1.92 I`, on both sides |

The control `log(-1 - x^2)` agrees to 25 digits: with real coefficients the argument stays on
the cut.

The design records the choice (`docs/workstreams/handoff-followups/design/S1-symbolic.md`,
lines 2226 to 2236). The conventions in the header of `rustmath-symbolic/src/series.rs` cover a
constant on a cut and do not say that the result can fail to represent the function. No gp-based
oracle can see this, because gp returns the same formal series.

Wanted:

1. At each application of a branch function (`log`, a non-integer power, `arcsin`, `arccos`,
   `arctanh`, `arccosh`, `arctan`, `arcsinh`): if the constant term `c0` is on an open cut and
   the engine cannot decide that every higher coefficient is a real multiple of the direction of
   the cut (real for the cuts on the real axis, purely imaginary for those on the imaginary
   axis), return `NotLaurent` with a message that says the argument leaves a branch cut at the
   point. This is the owner's rule 2 applied to cuts. A finer answer for the case where the
   argument leaves the cut to the same side for both signs of `x` (the expansion on the
   punctured neighbourhood, whose value at the point is not the principal one) can come later.
2. State the real-variable assumption and this rule in the conventions of `series.rs`.
3. A two-sided numeric oracle in the harness (`tests/data/s1_gp.py`): for every row with status
   `value` whose argument has a non-real coefficient, compare the truncated series at `x0` and
   at `-x0` with the scalar function evaluated by gp, with a tolerance from the truncation order.
   The five inputs above and their mirror images are the first rows.

Acceptance: the five inputs refuse; real-coefficient controls on a cut (`log(-1 - x^2)`,
`sqrt(-4 + x)`, `arcsin(2 + x)`, `arctanh(2 - x^2)`) still return a value and pass the two-sided
check; the matrix and the fuzz data still pass.

### C-2 (P0) S1: status and documents

- `docs/HANDOFF.md` lines 7 and 312 to 316 say that S1 is parked and has not landed.
  `4763f31` ("work in progress (not approved)") is in `main` through PR #584. Say what is
  merged, that the design is not approved, and what is open (C-1, C-3).
- `PROJECT_TRACKER.md`: rows for `series_engine.rs`, `series_const.rs`, `series_ball.rs` and the
  changed rows of `series.rs`, `simplify.rs`, `expand.rs`, `limits.rs`.
- Record that 39 matrix rows of the form `x + 0*f(...)` now refuse where the old code returned
  `x`, and that `sqrt(x^4)` and `abs(x)^2` refuse (conservative; a later relaxation is welcome).

### C-3 (P0, needs the owner's decision) S1: the resource bar

S1_BAR (c) asks that the consumers of a result (`simplify`, `expand`, `differentiate`) stay
bounded on adversarial inputs of at most 100 KB. The resources review found their cost to be
about the number of terms times the cost on the kernel source. My recommendation:

- the bar for S1 is that the engine itself is bounded (the `s1_depth`, `s1_payload` and memory
  tests) and that no consumer is slower or larger than before the change on the same inputs,
  measured;
- bounded `simplify`, `expand` and `differentiate` become a unit of their own;
- every doc comment that claims a bounded consumer is corrected now, whatever is decided.

### C-4 (P0) S1: downstream crates

`4763f31` adds 631 lines to `simplify.rs` and changes `expand.rs`, `expression.rs`, `parser.rs`
and `rustmath-jupyter/src/repl.rs`. I ran only `cargo test -p rustmath-symbolic`. Please run the
tests of every crate that depends on `rustmath-symbolic` (at least `rustmath-manifolds` and
`rustmath-jupyter`) and `cargo check --workspace`, and report the counts.

### C-5 (P1) `two_torsion_frobenius_order`: use the 2-rank

`rustmath-curves/src/weil.rs:398` reports `Unresolved (multiple of 7, divides 14)` at p = 17,
where the order is decided. Add the rule of Section 1.2: for each `k` it tries,
`dim ker(Frob^k - 1 on J[2]) <= v_2(#Jac(F_{p^k}))`, with `#Jac(F_{p^k})` the resultant, up to
sign, of the reversed L-polynomial and `x^k - 1`. With `k = 1` alone:

| p | expected order on `J[2]` | status |
|---:|---:|---|
| 17, 73, 127 | 14 | Decided |
| 43, 103, 113 | 12 | Decided |
| 89 | 8 | Decided |

Also return, when decided, the Jordan type for the eigenvalue 1 on `W`, so that the table of 1.2
can be regenerated in Rust. Apply the same rule to the rows over L. Report any prime where the
rule with `k > 1` decides more than the table of 1.2.

Acceptance: the seven values above, the four order 15 rows unchanged, and no row that was
Decided changes.

### C-6 (P1) Upkeep of the M23 handoff

- Opening paragraph: the default branch of `johnjanik/M23` is now `main`;
  `claude/blissful-knuth-qla7ym` is the work branch. The clone line should read
  `git clone https://github.com/johnjanik/M23 ../M23`, with the pin moved to the commit used.
- The three lines the Status block of Section 3 lists as out of date.
- 5.2: replace the table by the one in 1.2 (p = 5 added, primes to 131, the decided classes at
  43, 89, 103, 113).
- 5.3 A4 and 5.7: remove the case of two rational block systems (Section 1.1), and add the
  prediction of E-4.
- If PR #582 (`claude/m23-steiner` into `claude/m23-wpf`) is still open, close it as superseded
  by #583.

### C-7 (P1) Machine-readable tables

`data/rustmath_wpf_tables.json` in the M23 repository was transcribed by hand from
`wpf/impl/WPF.md`. Follow-up F-10 would remove that step. Wanted: a `--json <file>` option of
`cargo run --release -p rustmath-curves --example m23_zeta`, one object per row, with the keys
of `data/zeta_X.json` (`p`, `counts`, `P`, the singular points with their labels, the two mod 23
values, the factors of `P mod 2`, the `Q` candidates) plus `model`, `status`, the irreducibility
witness and the order on `J[2]` with its status. For L rows add `s0`. The M23 script
`scripts/crosscheck_rustmath.py` will then read that file directly.

## 3. What the exploratory work still needs

Each item says what I want, what it is for, and how I will know it is right. Items E-1 to E-3
and E-5 are in the handoff already; what is written here is only what changed.

### E-1 (P1) The canonical model of X (handoff 5.3, A0)

This is the gateway: E-2, E-3, E-10 and a second model for cross-checks all start from it.

- Run it on Le Lay's CC0 form `G`, which RustMath already vendors. The output is then derived
  from CC0 data and can be committed in both repositories. HJLPPZ's `F` has no licence; use it
  only as a cross-check through `$M23ISGALOIS`.
- Output as JSON: the four adjoint forms of bidegree at most (2, 21), the quadric (10
  coefficients), the cubic (20 coefficients), the change of basis used to reduce the
  coefficients, and the source commit. Reduce the basis (LLL) so that the model is small.
- Tests, as in the handoff: the quadric is unique with discriminant `4873` times a square; the
  model is smooth over Q; `#X(F_p)` on the model equals `N_1` of the table for the good primes
  at which the model is smooth; the two totally ramified points have vanishing sequence
  (0,1,2,3). Add: equality with Le Lay's `data/canonical_quadric_Q.json` up to a change of
  coordinates, and the primes at which the reduced model is singular, as a list.

### E-2 (P1) The special fibre at 3 (follow-up F-1)

- From E-1: make the ideal of the model saturated at 3 (a primitive quadric; the cubic made
  primitive modulo the quadric), reduce modulo 3, and compute the singular locus of the
  reduction over the algebraic closure, with the degree of its ideal.
- Statuses: `Decided: good reduction at 3` if the reduction is a smooth curve of genus 4.
  Otherwise `Unresolved: this model is singular at 3`, with the singular points and their local
  types. A singular model does not decide bad reduction, and a search over changes of
  coordinates that finds no smooth model never decides.
- A second way to decide, in the other direction: good reduction at 3 implies that 3 is
  unramified in `Q(J[2])`, hence in `K_W`. If 3 divides the discriminant of the field of `g8` or
  `g15` (E-3), X has bad reduction at 3, Decided.
- Read 1.3 first. It shows the cover is of bad reduction at 3; the question is about the curve.

Why: good reduction at 3 would cut the ramification of `K_W` down to `{2, 23}`.

### E-3 (P1) Tritangent planes, `g15`, `g8` (handoff 5.3, A1 to A5)

The specification stands. Changes:

- The acceptance table is the one in Section 1.2: 30 primes, with the classes at 43, 103, 113
  (`2 6`) and 89 (`1^2 2 4`) decided. These four are the strongest tests in the table, because
  an A8 field chosen at random matches each with probability 1/6 and 1/8.
- Exactly one block system is rational (Section 1.1), so the dichotomy of A4 has no third case.
- 5 is a good prime; the field discriminant test is `{2, 3, 23}`, and E-2 may remove 3.
- Deliver `g15` and `g8` as JSON with the working primes, the precision reached, and the
  status in the terms of 5.6 of the handoff.

### E-4 (P2) The image on `J[2]`, checked on the planes

After A1 works at one prime, run it at p = 17 (or 73, or 127). Prediction from Section 1.1,
checked by `scripts/jac2_frobenius_orbits.py`: the Frobenius orbits on the 120 tritangent planes
have lengths

- `1, 7^9, 14^4` (one rational plane, nine orbits of length 7, four of length 14), so 64 planes
  are defined over `F_{p^7}` and 56 only over `F_{p^14}`.

If Frobenius had order 7 on `J[2]` the lengths would be `1, 7^17`. For comparison, at a prime of
order 15 they are `15^8` (the prediction already in the handoff). First check that the canonical
quadric has rank 4 modulo the chosen prime; if it has rank 3 there (a vanishing theta-null of
the reduction), take another of the three primes.

Why: an independent confirmation of 1.1 that does not go through `#Jac(F_p)`.

### E-5 (P1) WP-P

- The 57 rows of `data/m23_three_point_nielsen.json` and the three dictionaries (handoff 4.2,
  4.3), as specified.
- Then the twists (handoff 4.4 item 1). There is no reference for these yet. I will compute one
  in Python (`m23golay/nielsen.py`) so that there are two implementations; until then report
  them as new and unverified.
- The parabolic cohomology over `F_2` per cover (handoff 4.3) is the piece I need most for the
  paper's last question: for each class vector, `dim H^1(P^1, j_* G0)` and whether the image in
  `Jac[2]` is Lagrangian.

### E-6 (P2) Number fields of degree 23

Wanted: for a monic integer polynomial of degree 23 and a prime `p`, the exponent of `p` in the
field discriminant and the list of `(e, f)` of the primes above `p`, exact, through round 2 or
the OM/Montes stack (`Discriminant(MaximalOrder(K))`, `Decomposition`, `pMaximalOrder` in the
handbook). The coefficients of `f1` have up to 14 digits; those of a specialization of `F`
start at about 60 digits before any reduction of the polynomial.

Acceptance: the table of 1.3, for `f1` at 2, 3, 23 and `f2` at 2, 3, 7, 23.

Then a specialization scan. For `t0` in a box of rationals:

- the primes that can ramify divide `2 * 3 * 23`, the numerator and denominator of `t0`, and the
  numerator of `t0^2 + 23` (the other factor of the discriminant of `F` in `V` is the square of
  `h62`, `data/h62.json`);
- the Galois group of `F(t0, V)` is M23 as soon as it is irreducible modulo one prime and has,
  modulo another, a cycle type other than `1^23`, `23` and `1 11^2`. The transitive subgroups of
  M23 in degree 23 are `C23`, `23:11` and M23, and the group of a specialization lies in M23
  (trusted input, recorded in the result as in the Steiner labelling). This is the positive
  counterpart of follow-up SL-9;
- output per `t0`: the discriminant exponents, the `(e, f)` lists at 2, 3, 23, and the status
  of the Galois group.

What it is for: the behaviour at 3 as a function of `v_3(t0^2 + 23)` (Section 1.3); M23 fields
of small discriminant; and Question 10.4 of the paper, since a rational point of the curves
`C_2` or `C_7` above `t0` makes `F(t0, V)` reducible, so irreducibility modulo one prime for
every `t0` of height up to `H` proves there is no such point of that height.

### E-7 (P2) Belyi maps of positive genus

The handoff (4.4 item 2 and 5.4) explains that the Belyi engine is genus 0 and written for one
passport. For the exploratory work the needed capability is: from a permutation triple of
degree 23 and genus 2 to 4, a numerical model of the cover good enough to recognize its field
of moduli.

End-to-end test with a known answer, before any new class vector is tried: take the seven
triples of the class `(2A, 23A, 23B)` (`m23golay.nielsen.nielsen_class()`), compute an invariant
of each of the seven covers, and recover that the Hurwitz scheme is one rational point and one
point with residue field `L = Q[x]/(x^6 - x^5 - 3x^4 - 3x^3 + x^2 + 5x + 4)`. For the rational
cover, recover the canonical model of E-1 up to a change of coordinates.

Then the genus 2 class vectors `(2A,14A,14B)`, `(2A,11A,11B)`, `(2A,15A,15B)` (17 covers in the
first, 3 of them real), and the fifteen covers of degree 46 of route B.

### E-8 (P3) Periods and theta constants

Question 10.2 of the paper asks whether the principally polarized fourfold `Jac(X)/W` is a
Jacobian. Write `Jac(X) = C^4 / (Z^4 + tau Z^4)` in a symplectic basis for which `W` is the
image of `(1/2) Z^4`. Then the period matrix of `Jac(X)/W` is `2 tau`. The Schottky-Igusa form
`2^4 * sum(theta_m^16) - (sum(theta_m^8))^2`, summed over the 136 even characteristics,
vanishes exactly on the closure of the Jacobian locus in genus 4. So:

- a period matrix `tau` of X in a homology basis read from the triple `(g1, g2, g3)`, with `W`
  located by the cocycle description of handoff 4.3;
- the 136 even theta constants at `2 tau`;
- status `Decided: not a Jacobian` only if the value is a ball that excludes 0 (this needs the
  certified enclosures the handoff lists as absent); a value that is 0 to the working precision
  is `Unresolved`, reported with the precision.

Take the statement about the Schottky-Igusa form and its normalization from the literature
(Igusa 1981) when this is built, not from this file. MAGMA's Riemann surface functions are the
interface reference; the chapter digests may not include them yet.

### E-9 (P2) A long trace table

`N_1` alone costs about `p` fibre evaluations, so the certificate and `#X(F_p)` for every prime
of good reduction below `2^16` (the limit of `SmallFq`) is about `2 * 10^8` quartic root
counts in total. Follow-up F-4 (a faster root count) helps here. Output: `p`, `N_1`, the
certificate label.

What it is for: the moments of `(p + 1 - N_1)/sqrt(p)` against those of `USp(8)`, as numerical
support for `End(Jac X) = Z`; take the expected moments from the literature (Kedlaya and
Sutherland), not from memory. It is also the input any later work on the L-function needs.

### E-10 (P2) More general point counts

- Fibre degree up to 8, so that Le Lay's bidegree (8, 23) model can be counted (follow-up F-2):
  an independent model at the primes with a cusp (13, 31, 41, 73).
- A curve in P^3 given by a quadric and a cubic, by enumeration of the quadric: needed for the
  tests of E-1.

### E-11 (P3) Four branch points

Braid orbits on Nielsen classes of length 4 for M23, with the genera of the reduced Hurwitz
curves (handoff 4.4 item 3). The published method is the BRAID algorithm of Magaard,
Shpectorov and Voelklein. Start only after E-5.

### E-12 (P3) Functional equation

Not now. The conductor of `Jac(X)` is `2^a 3^b 23^c` with unknown exponents, and a numerical
test of the functional equation of a degree 8 L-function needs on the order of the square root
of the conductor many coefficients. With `c` likely at least 8 that is beyond what naive
counting reaches. It is listed so that E-9 writes its table in a form that can be extended.

## 4. Cross-cutting requests

- Keep the statuses as they are now: `Decided` with the reason, `Unresolved` with what was
  tried, a refusal when the model is unusable, and the trusted inputs (the genus, `Gal = M23`)
  recorded in every result.
- A passed Weil test is weak evidence. The wrong row at (31, s - 12) passed it and failed the
  mod 23 congruence. Keep the mod 23 and mod 2 checks as the acceptance tests of any new row.
- Prefer CC0 inputs (Le Lay's `data/`) for anything whose output should be committed. Files of
  `shaowuz/m23isgalois` stay behind `$M23ISGALOIS`.
- Write results that I will compare as JSON, with the commit and the command in the file.
- For anything new and numerical, say in the report which of three classes it is: reproduced
  from reference, new and verified at two primes or by two methods, new and unverified.

## 5. Suggested order

1. C-1 to C-4 (S1), since that code is in `main`.
2. C-5, C-6, C-7: small, and they bring the tables in line with Section 1.
3. E-5, first part: the 57 rows.
4. E-1, then E-2.
5. E-3 through A1 at p = 79, then E-4 at p = 17, then A2 to A5.
6. E-9 and E-6 as fillers; they are independent of everything else.
7. E-10, E-7.
8. E-8, E-11, E-12 only after asking.

## 6. Notes for the owner

- C-1 and C-3 change decisions recorded in the S1 design. They need your word before the
  RustMath session acts on them.
- The RustMath data README notes that the vendored `golay_resolvents_no_f.json` still determines
  `f1` and `f2`. Both polynomials are printed in the HJLPPZ paper (Examples 1.2 and 3.17) and
  are reproduced with that citation in `m23golay/data.py`; the restriction we set concerns the
  files of the unlicensed repository. Your call whether anything should change.
- Section 1.1 is a strengthening of Theorem 4.8 of the paper and Section 1.2 sharpens its
  Frobenius table. Neither is in the manuscript yet.
