# RustMath build handoff --- algorithms the M23/Q campaign needs

> Audience: the session that **builds** RustMath (Rust). This session runs the M23-over-Q campaign and does **not** compile Rust. Everything below is grounded in four read-only inventories of the actual tree; every "present" claim carries a `path:line`, and every "absent" claim was verified by grep. Do not invent symbols. If a thing is marked missing, it is missing; if `exists-unwired`, the code is there and only needs a call site.

---

## 0. Orientation --- what the campaign is doing, so you know what each algorithm is FOR

**Endpoint (S98, the deleted-sheet realization).** The prize is a single Belyi-type map `phi = P/Q in Q(x)`, `deg 24`, with **M24 monodromy over P^1_Q**. From it, the *deleted-sheet resolvent*

```
R(X, u) = ( P(X)·Q(u) − P(u)·Q(X) ) / (X − u)   ∈ Q(u)[X],  deg_X = 23
```

has `Gal( R / Q(u) ) = M23` **regularly** over the rational function field. That regular M23/Q(u) statement is the true target. Getting one exact `phi = P/Q` over `P^1_Q` is the whole game.

**Two live routes to that `phi`, plus a closure step:**

- **Route A --- the `[2,12,5]` Belyi map.** Solve the genus-0 Belyi map with ramification `(2,12,5)` numerically via modular forms, snap the float coefficients to exact `Q` (or a small number field), then feed the exact `coeffs:&[Rational]` into the already-built DECIDE/CERTIFY half.
  - *Step 1 (S100)*: run the numerical modular-forms chart chain to get `(P,Q)` floats + a residual, **and persist them**.
  - *Step 2 (S100)*: **recognize** those floats to exact `Q`/number field (the single highest-leverage gap).
- **Route B --- the `(2A^4, 4B)` surface lift.** Take an `F_17` point of a 75-equation framed system, lift it to a bivariate `Z_17[[u,v]]` power-series surface (multivariate Hensel/Newton), slice, and reconstruct rational functions in `Q(u)` (Hermite-Padé + CRT/rational reconstruction).
- **Closure**: p-adic factorization / Newton-polygon / Montes-Ore analysis at `Q_2` and `Q_p`, framed-ideal Groebner/rank gates, and conic isotropy for descent obstructions.

**One trap to internalize:** two different "solve" code paths both carry the `[2,12,5]` name and must never be conflated:
- **(A)** the numerical **modular-forms chart chain** (`assemble_scaled_ami/recover_forms → forms_to_series → echelonize → coordinate_x → solve_belyi_map`) returning `(P,Q)` as `Vec<Complex>`. Fully implemented and unit-tested **only for `(5,3,3)` at `N=48`, in memory**; **never invoked for `[2,12,5]`, never persisted, hard-codes the `z_a` chart.**
- **(B)** a pinned polynomial **parameter-homotopy** route (`pipeline.rs:322 → rustmath-numerical/homotopy.rs`) that shells out to a Julia `HomotopyContinuation.jl` script writing `belyi_2_12_5_result.json`. **This is the source of the empty `{"solutions":[]}`** and is unrelated to `recover_forms/solve_belyi_map`. No `belyi_2_12_5_result.json` exists on disk yet.

---

## 1. Already built --- DO NOT rebuild; these are the CONSUMERS

Wire your new code **into** these. They all take exact `coeffs:&[Rational]` or an exactified `NumericalSolution` and must not be re-authored.

### 1a. DECIDE / CERTIFY half (exact-coefficient consumers) --- `rustmath-curves/src/belyi/`

| Symbol | Path:line | Signature (as found) | Role |
|---|---|---|---|
| `decide_nonprovisional` | `pipeline.rs:534` | `fn(name:&str, sol:&NumericalSolution, max_deg:usize, primes:&[i64], x0_candidates:&[Rational], l_descent:Option<(&LCover,&SigmaCorrespondence)>, bad_locus:Option<(&GenusZeroBelyiFactorizationQ,&P1PointQ)>) -> DecideReportNonProvisional` | **Sole producer** of the `M23QRealized` verdict. `exactify(sol)` → field of def; over Q runs `audit_m24` then (only if Confirmed) `audit_m23_residual`. |
| `DecideReportNonProvisional::is_m23_q_realized` | `pipeline.rs:495` | `fn(&self)->bool` | **THE payoff predicate.** True iff cover/Q ∧ `audit_m24==Confirmed(M24)` ∧ `audit_m23_residual==Confirmed(M23)` with the deg-23 witness. |
| `audit_m23_residual` | `audit.rs:409` | `fn(coeffs:&[Rational], x0_candidates:&[Rational], primes:&[i64]) -> M23Witness` | The real **1+23 deleted-sheet split**: `specialize_numerator`→`div_linear` off `(x−x0)`→primitive deg-23 residual→Zassenhaus irreducible→disc square-bit→Frobenius up `C23⊂F23⊂M23⊂A23`. |
| `audit_m24` | `audit.rs:330` | `fn(coeffs:&[Rational], primes:&[i64]) -> GroupVerdict` | Decisive M24 gate; prereq before the M23 residual is attempted. |
| `classify_m24` | `audit.rs:290` | `fn(observed:&[Vec<usize>]) -> GroupVerdict` | Confirmed(M24) iff blind cycle-type class matches exactly. |
| `specialize_numerator` | `audit.rs:221` | `fn(coeffs:&[Rational], t0:&Rational) -> Vec<Integer>` | Deg-24 numerator of `phi−t0` at a numeric rational `t0`. |
| `frobenius_types` | `audit.rs:268` | `fn(f:&[Integer], primes:&[i64], deg:usize) -> Vec<Vec<usize>>` | Chebotarev evidence collector. |
| `transitive23::classify` | `rustmath-groups/src/transitive23.rs:277` | `fn(observed:&[Vec<usize>], disc_is_square:bool) -> Option<Group23>` | The deg-23 certifier actually used (statistical; needs Chebotarev saturation). |
| `decide_from_solved_cover` | `pipeline.rs:381` | `fn<C:ExactBelyiCover>(...)->DecideReport` | Provisional/constructibility pipeline (distinct from the non-provisional M23 gate). |
| `verify_2_12_5_cover` / `trait ExactBelyiCover` | `verify.rs:59` (trait `:49`) | `fn<C:ExactBelyiCover>(cover:&C)->VerificationReport` | Hard constructibility gate: exact identity, branch locus `{0,1,∞}`, ram `2^8 1^8 / 12^2 / 5^4 1^4`, genus 0, independent monodromy. |
| `verify_identity` (candidate) | `candidate_verify.rs:48` | `fn(p_zero,p_pole,p_one:&PolyC, lambda,c:Complex64, tol:f64)->CandidateReport` | Numeric f64 identity check for candidate covers. |
| `certify_phi_sigma_over_L` | `descent.rs:329` | `fn(cover:&LCover, g_sigma:&Gl2Quad)->bool` | B3: exact gluing identity over `L=Q(√δ)`. |
| `descent_conic` | `descent.rs:207` | `fn(g_sigma:&Gl2Quad, bad_locus_clear:bool)->Verdict<ConicBrauerReport>` | Full quadratic descent → quaternion → conic → Brauer verdict. |
| `LCover` | `descent.rs:263` | `struct{field:QuadField, a,b,r,s:Vec<QuadElem>, lambda:QuadElem}` | Solved cover over `L`. |
| `loop_min_approach_hp` | `numerical_monodromy.rs:362` | `fn(s:&SolHp, base_fiber_f64:&[Complex64], center:Complex64, n_steps:usize)->f64` | Monodromy sheet-crush diagnostic. |

> **PHANTOM WARNING:** prior sessions referenced `deleted_sheet_family` / `deleted_sheet_resolvent` as *existing* code. They do **not** exist (zero grep hits across all of RustMath). The real *specialized* 1+23 split is `audit_m23_residual` above. The *symbolic* resolvent is genuinely **absent** and is a build target (see B1/B2).

### 1b. Numerical SOLVE chain (present, but only `(5,3,3)`-tested, in-memory, z_a-only) --- `rustmath-curves/src/belyi/`

| Symbol | Path:line | Signature | Note |
|---|---|---|---|
| `recover_forms` | `modular_forms_hp.rs:540` | `fn(tg64,tg,cg,k:i64,big_n:usize,q:usize,threshold_decimal:&str,tol_decimal:&str,rho_scale:f64)->Vec<Vec<Complex>>` | **Hard-codes `&tg.z_a`** at `:552`; assembles `(A−I)` fully in memory; runs `jacobi_svd`. |
| `forms_to_series` | `genus0_map.rs:164` | `fn(forms:&[Vec<Complex>], k:i64, prec:u32)->Vec<Vec<Complex>>` | i128→f64 binomial cast (`:171-172`) = precision cliff at large `k`. |
| `echelonize` | `genus0_map.rs:181` | `fn(rows:Vec<Vec<Complex>>, prec:u32, tol:f64)->Vec<Vec<Complex>>` | f64-tol valuation, not prec-scaled. |
| `coordinate_x` | `genus0_map.rs:234` | `fn(echelon:&[Vec<Complex>], m,e:usize, prec:u32, tol:f64)->Vec<Complex>` | Hauptmodul `x=h/(g+c·h)`; tested only for `(5,3,3) m=1,e=1`. |
| `solve_belyi_map` | `genus0_map.rs:17` | `fn(x:&[Complex], phi:&[Complex], d:usize, prec:u32)->(Vec<Complex>,Vec<Complex>)` | Returns `(P,Q)` up to scale; **no residual, no normalization, no persistence**; hard-codes `JacobiSvdOptions::new(prec,80,"1e-70","1e-40")` and `nrows=ncols+6`. |
| `assemble_scaled_ami` / `run_atlas_dump` | `modular_forms_hp.rs:130` / `:452` | `fn(...,ctr:&Complex)->(Vec<Vec<Complex>>,Float)` ; `fn(&AtlasDumpParams)->AtlasDumpResult` | `assemble_scaled_ami` **already accepts an arbitrary chart center** (`ctr`); `run_atlas_dump` dispatches `"a"/"b"/"c"/"a2"/"b2"/"c2"` where `"b"=z_b` is the order-12 chart --- but only **streams** to a write-only EXT limb file, never runs SVD. |
| `CosetGraph::build` / `compactify_with` | `coset_graph.rs:78` / `:168` | `fn(tg,&[usize],&[usize])->CosetGraph` | Combinatorial f64 layer. |
| `jacobi_svd` | `mp_svd.rs:234` | `fn(a:&MpMatrix, opt:&JacobiSvdOptions)->Result<JacobiSvdResult,SvdError>` | Arbitrary-prec one-sided complex Jacobi SVD; in-memory `MpMatrix` only. Provides `numerical_nullity_indices`, `right_nullspace_basis`. |

### 1c. Exact-recovery / lattice engines to build ON (present) --- number/matrix crates

| Symbol | Path:line | Signature | Note |
|---|---|---|---|
| `recognize_complex_algebraic` | `rustmath-numberfields/src/recognize.rs:259` | `fn(re:f64, im:f64, max_deg:usize)->Option<Vec<BigInt>>` | The Wave-2 recognizer. **f64-only, weight fixed 1e10, residual<1e-4** --- cannot exploit hp data. |
| `recognize_rational` | `recognize.rs:183` | `fn(a:&BigInt, m:&BigInt)->Option<Rational>` | p-adic/CRT residue → Q (modulus-based, not float). |
| `recognize_algebraic` | `recognize.rs:195` | `fn(a:&BigInt, m:&BigInt, max_deg:usize)->Option<Vec<BigInt>>` | Minpoly from a p-adic residue via LLL. |
| `lll_reduce` (exact) | `recognize.rs:85` | `fn(basis:Vec<Vec<BigInt>>)->Vec<Vec<BigInt>>` | Exact BigRational GSO LLL, δ=3/4. Engine behind all `recognize_*`. |
| `exactify` / `ExactifyOutcome` | `rustmath-numerical/src/exactify.rs:49` | `fn(sol:&NumericalSolution, system:&PolySystem, max_deg:usize)->ExactifyOutcome` | The only orchestrator that snaps a numeric candidate to exact; **downcasts to f64 at `:52-53`** --- drops all hp precision. |
| `lll_reduce` / `lll_reduced_basis` (matrix) | `rustmath-matrix/src/lll.rs:75` / `:133` | `fn(basis:&[Vec<Integer>])->(Vec<Vec<Integer>>,Vec<Vec<Integer>>)` (returns unimodular `U`) | Natural base for a real integer-relation/PSLQ builder. |
| Lattice LLL family | `rustmath-matrix/src/lattice.rs:298/376/192/477/510/545` | `lll_reduce_real`, `lll_reduce_rf`, `gram_schmidt_exact`, `Lattice::lll_reduced`, `short_vectors`, `shortest_vectors` | Building blocks for simultaneous relations; none do value-recognition. |
| `rational_reconstruct` / `ContinuedFraction` | `recognize.rs:160` ; `rustmath-rationals/src/continued_fraction.rs:13` | `fn(a:&BigInt,m:&BigInt)->Option<(BigInt,BigInt)>` ; `from_rational`, `all_convergents` | CF reconstructor is **modulus-driven**; `ContinuedFraction` has no `from_real`. |

### 1d. p-adic / algebraic infra (present, `exists-unwired`) --- Route B & closure

| Symbol | Path:line | Signature (as found) |
|---|---|---|
| `fp_factor::factor` (+ SFF/DDF/EDF, Berlekamp) | `rustmath-polynomials/src/fp_factor.rs:375`, `factorization.rs:558` | `fn(f:&[i64], p:i64)->Vec<Vec<i64>>` |
| `zp_hensel::hensel_lift` / `_all` | `rustmath-polynomials/src/zp_hensel.rs:143/:192` (`factorization.rs:976`) | univariate Hensel lift `f=g0·h0 mod p^k` |
| `panayi::roots_in_zp` / `count_roots_qp` / `hensel_lift_root` | `rustmath-numberfields/src/panayi.rs:112/154` ; `rustmath-rings/src/padics/padic_integer.rs:215` | Z_p / Q_p root enumeration + single-root refine |
| `om_factorization` (Montes/MacLane) | `rustmath-rings/src/padics/om_factorization.rs:244` | `fn(f:&UnivariatePolynomial<Rational>, p:i64, min_precision:u32)->Result<OmFactorization>` (per-factor `e`,`f`, MacLane leaf) |
| `padic_factor` / `ramification_type` / `local_decomposition` | `rustmath-polynomials/src/padic_factor.rs:125/160` ; `rustmath-numberfields/src/local_field.rs:114` | lighter Panayi-based local factorization |
| `NewtonPolygon::of_rational_polynomial` / `slopes` / `root_valuations` | `rustmath-rings/src/padics/newton_polygon.rs:122/162/183` ; `rustmath-polynomials/src/newton.rs:73` | prime-agnostic (p=2 fine); **two independent impls** |
| MacLane tower | `rustmath-rings/src/valuation/maclane.rs:625/1130` (+ `augmented_valuation.rs`, `inductive_valuation.rs`, `gauss_valuation.rs`, `limit_valuation.rs`) | `mac_lane_approximants`, `PAdicInductiveValuation`, `solve_mod_p` |
| Z_p/Q_p rings + extensions | `rustmath-rings/src/padics/factory.rs:119/245/347/423` (+ `capped_relative.rs`, `unramified.rs`, `eisenstein.rs`) | `PadicIntegerRing::new`, `PadicField::new/from_rational`, `Unramified/EisensteinExtension` |
| `MPowerSeries` container | `rustmath-rings/src/multi_power_series_ring{,_element}.rs:45/79/90…` | `with_precision`, `set_coefficient`, `truncate`, `derivative`, `integral`, `valuation` --- **container only, no solver** |
| `newton_puiseux` / `implicit_function` / `prime_field_roots` | `rustmath-powerseries/src/algebraic.rs:108/182/396` | univariate-base series roots (single series variable) |
| `PolySystem` (modular eval + Jacobian) | `rustmath-polynomials/src/poly_system.rs:62/136/153/186/195` | `from_terms`, `evaluate_mod`, `jacobian_mod`, `evaluate`, `is_exact_solution` --- **scalar** Newton-over-p^k |
| `chinese_remainder_theorem` / `crt_two` | `rustmath-integers/src/crt.rs:17/93` | multi-prime CRT |
| Groebner over any field | `rustmath-polynomials/src/groebner.rs:1043` (+ `ideal.rs`) | `groebner_basis_field<R:Field>(...)`, F4, `Ideal::dimension` |
| `PrimeField` (GF(p) Ring+Field) | `rustmath-finitefields/src/prime_field.rs:12/289` | instantiate any generic-over-Field routine at `p` |
| Matrix rank/kernel over a field | `rustmath-matrix/src/linear_solve.rs:21/129/200` | `row_echelon_form`, `rank`, `kernel` |
| Resultants (uni / bivariate-in-t) | `rustmath-polynomials/src/disc.rs:149`, `univariate.rs:501`, `bivariate.rs:91/208` | `resultant`, `resultant_in_t`, `resultant_q` |
| `hilbert_symbol` (all places) | `rustmath-quadraticforms/src/hilbert.rs:89/179` | `fn(a,b:&Rational, place:Place)->Result<i8,HilbertError>` |
| `DiagonalConicQ` (+ Brauer/quaternion verdict) | `rustmath-quadraticforms/src/conic.rs:120/173` | `new`, `from_ternary_form`, `quaternion_class`, `verdict(bad_locus_clear)` --- already consumed by the Belyi portal |
| `class_group` / `minkowski_bound` | `rustmath-numberfields/src/classgroup.rs:264/304/182` | closure-route obstruction checks |

> **No external CAS oracle exists.** Zero real hits for `python-flint`/`flint`/Sage bridge in the tree (`rustmath-rationals/src/sage_wrapper.rs` is a pure-Rust API-compat shim, not a Sage bridge). The only in-crate certificate is `PolySystem::is_exact_solution` (exact back-substitution). Plan cross-checks accordingly.

---

## 2. ALGORITHMS TO BUILD (prioritized by leverage)

Legend: **State** = grounded in tree (`missing` / `partial` / `exists-unwired`). **Build/Delegate** = author in RustMath vs hand to Sage.

---

### P0 --- Route A: the recognition-to-Q connector and the persist harness (unblocks a candidate M23/Q witness fastest)

---

#### **A1 --- HP precision-aware complex algebraic recognizer** `P0` · build-in-RustMath

- **Symbol:** `rustmath_numberfields::recognize::recognize_complex_algebraic_hp`
- **Signature:** `pub fn recognize_complex_algebraic_hp(re: &rug::Float, im: &rug::Float, prec_bits: u32, max_deg: usize) -> Option<Vec<BigInt>>`
- **Spec:** Same LLL relation search as `recognize_complex_algebraic` (`recognize.rs:259`) but ingest **arbitrary-precision** Floats; set the lattice weight/threshold from `prec_bits` (e.g. `W = 2^(prec_bits·frac)` rather than the fixed `1e10`), and verify the residual `|Σ c_i z^i|` in **hp arithmetic** rather than `f64 < 1e-4`. Build the degree-`d` lattice rows `e_i ++ [round(W·Re z^i), round(W·Im z^i)]`, run the exact-BigRational `lll_reduce` (`recognize.rs:85`), accept the shortest reduced vector with coeffs `< W^(1/(d+2))` and hp residual below a prec-scaled bound; `normalize_poly`.
- **In/out:** in `(&rug::Float, &rug::Float, u32, usize)` → out `Option<Vec<BigInt>>` (minpoly coeffs).
- **Campaign use:** **Route A Step 2 (S100)** --- the numberfields-side primitive the belyi connector (A2) calls.
- **State:** `missing`. `recognize.rs:259` signature is `(re:f64, im:f64, ...)`; module doc `:16-17` pins the f64 signature as the "Wave-2 interface contract". No hp overload (`grep 'rug::Float|BigFloat|prec_bits'` in `recognize.rs` = 0).
- **Deps:** exact `lll_reduce` (`recognize.rs:85`), `rug::Float`.
- **Acceptance:** on a known algebraic number rendered to 512 bits (e.g. root of a chosen deg-4 minpoly with height ~10^6), recover the **byte-identical** minpoly where the f64 `recognize_complex_algebraic` fails; residual verified in hp.

---

#### **A2 --- Belyi-side hp→exact connector (gauge-normalized, doubling-stable)** `P0` · build-in-RustMath

- **Symbol:** `rustmath_curves::belyi::exactify_hp::snap_hp_solution`
- **Signature:** `pub fn snap_hp_solution(report: &NewtonHpReport, gauge: &Gauge2_12_5, system: &PolySystem, max_deg: usize) -> ExactifyOutcome`
- **Spec:** Consume `newton_hp`'s 512-bit `solution_decimals: Vec<String>` and:
  1. **Gauge normalization** --- undo the frozen white/black-point placement (frozen indices `[0,1,48,49,50,51]`) to canonical coordinates so recognition is basis-independent.
  2. **HP recognition** --- call `recognize_complex_algebraic_hp` (A1) per complex coordinate (weight from working precision, **not** truncating to f64).
  3. **Stability-under-doubling gate** --- re-run `refine_hp` at `2×prec_bits`, recognize again, accept a coordinate only if its minpoly is **byte-identical** across the doubling.
  4. Route degree-1 coords to `CertifiedRational` via `system.is_exact_solution(&pt)`; higher-degree coords to a common number field.
- **In/out:** in `(&NewtonHpReport, &Gauge2_12_5, &PolySystem, usize)` → out `ExactifyOutcome` (the existing enum: `CertifiedRational(Vec<Rational>)` / `AlgebraicCoordinates(Vec<Vec<BigInt>>)` / `RecognitionFailed` / `SubstitutionFailed`).
- **Campaign use:** **Route A Step 2 (S100)** --- the single highest-leverage gap. Its `CertifiedRational` output is exactly what `decide_nonprovisional` (`pipeline.rs:534`) then `audit_m24`/`audit_m23_residual` consume to reach `is_m23_q_realized`.
- **State:** `missing`. `grep 'recognize_*|lll_|pslq'` over `rustmath-curves/src/belyi` = 0. `newton_hp.rs:250-251` `solution_decimals` is documented "for LLL/PSLQ recognition" but has **no consumers at all** outside its own definition in `newton_hp.rs` (no recognizer reads it, and not even the examples touch the field) --- which only strengthens the gap argument: the 512-bit precision is discarded outright. Today `decide_nonprovisional` (`pipeline.rs:534`) → `exactify` downcasts to f64 at `exactify.rs:52-53`, discarding all 512-bit precision.
- **Deps:** A1; `NewtonHpReport` (`newton_hp.rs:250`); `ExactifyOutcome` (`exactify.rs:49`); `PolySystem::is_exact_solution` (`poly_system.rs:195`). Note `exactify` lives in `rustmath-numerical`; this connector belongs in `rustmath-curves/belyi` and must NOT take the f64 `NumericalSolution` path.
- **Acceptance:** feed the doubling-stable snap into `decide_nonprovisional`; on the true `[2,12,5]` solution reach `CertifiedRational`, and downstream `is_m23_q_realized()==true`. Cross-check: `system.is_exact_solution` returns true on the recovered point.

---

#### **A3 --- `solve_and_persist` driver for `[2,12,5]` at ~100 digits** `P0` · build-in-RustMath

- **Symbol:** `rustmath_curves::belyi::genus0_map::run_and_persist_belyi`
- **Signature:** `pub fn run_and_persist_belyi(params: &AtlasDumpParams, d: usize, phi: &[Complex], out: &str) -> std::io::Result<()>`
- **Spec:** Drive the full numerical chain end-to-end for `[2,12,5]` at `prec ≥ 400` bits: `recover_forms → forms_to_series → echelonize → coordinate_x → solve_belyi_map`, then serialize `(P,Q)` (the `Vec<Complex>` float vectors) **plus a residual** to `belyi_2_12_5_result.json` as `{P:[..], Q:[..], residual, prec, N, chart}` with re/im decimal strings.
- **In/out:** in `(&AtlasDumpParams, usize, &[Complex], &str)` → out `io::Result<()>` writing the JSON.
- **Campaign use:** **Route A Step 1 (S100)** --- persist `P,Q` floats + residual for downstream recognition (A2).
- **State:** `missing`. All callers of `solve_belyi_map`/`recover_forms` are **tests only**: `genus0_map.rs:349` (`solve_belyi_map`), `:269/:321/:381` and `modular_forms_hp.rs:962` (`recover_forms`), all `abc=(5,3,3), N=48`. No example/bin drives `[2,12,5]`. **Do not confuse with** the Julia parameter-homotopy writer (`pipeline.rs:322-332` → `homotopy.rs:255-311`) that emits the empty `{"solutions":[]}` --- that is a *different route* and is unrelated to this chain.
- **Deps:** A4 (thread `ctr` for the z_b chart), A6 (residual return), the present chain functions in §1b.
- **Acceptance:** file `belyi_2_12_5_result.json` appears on disk with non-empty `P,Q` and a residual small relative to `prec`; the `(5,3,3)` chain still passes its unit tests. No `belyi_2_12_5_result.json` exists on disk today (background `find` over `/` found none).

---

#### **A4 --- Thread chart center through `recover_forms` (order-12 `z_b` chart)** `P1` · build-in-RustMath

- **Symbol:** `rustmath_curves::belyi::modular_forms_hp::recover_forms_centered`
- **Signature:** `pub fn recover_forms_centered(tg64, tg, cg, k:i64, big_n:usize, q:usize, threshold_decimal:&str, tol_decimal:&str, rho_scale:f64, ctr:&Complex) -> Vec<Vec<Complex>>` (or add a `ctr` param to `recover_forms`).
- **Spec:** Allow the in-memory recover path to expand about the order-12 vertex `z_b` (or satellite `b2`), instead of the hard-coded `z_a`. `assemble_scaled_ami` **already accepts `ctr`** --- just thread it in.
- **Campaign use:** **Route A Step 1 (S100)** requires the order-12 chart at `z_b` (the multi-chart Hauptmodul requirement).
- **State:** `partial`. `recover_forms` hard-codes `&tg.z_a` at `modular_forms_hp.rs:552` (same at `dim_s_k_svd:519`, `nullity_s_k:500`). Chart dispatch to `z_b` exists **only** in `run_atlas_dump:476` and the dump harness `:703,:794`, which do **not** call `recover_forms`/solve.
- **Deps:** `assemble_scaled_ami(ctr)` (`modular_forms_hp.rs:130`); `run_atlas_dump` center dispatch (`:475`).
- **Acceptance:** `recover_forms_centered(..., &tg.z_b)` reproduces the `z_a` result on a symmetric test and yields the correct `dim S_k` kernel at `z_b` for `(5,3,3)`.

---

#### **A6 --- Residual + gauge normalization for `solve_belyi_map`** `P1` · build-in-RustMath

- **Symbol:** extend `rustmath_curves::belyi::genus0_map::solve_belyi_map`
- **Signature:** `pub fn solve_belyi_map(x:&[Complex], phi:&[Complex], d:usize, prec:u32) -> (Vec<Complex>, Vec<Complex>, rug::Float)` (add residual; add canonical monic/gauge normalization of `P,Q`).
- **Spec:** Return the fit residual (smallest `sigma`, or `||phi·Q − P||`) and a canonical normalization so the persisted answer is reproducible/verifiable to ~100 digits. The null vector is only defined up to scale today.
- **Campaign use:** **Route A Step 1 (S100)** --- `S100` must persist a residual alongside `P,Q`.
- **State:** `partial`. `solve_belyi_map` returns just `(p,q)` from `svd.v.get(i,last)` (`genus0_map.rs:44-50`); `svd.sigma` (the residual signal) **is available** in `JacobiSvdResult` (`mp_svd.rs:204`) but discarded.
- **Deps:** `JacobiSvdResult.sigma` (`mp_svd.rs:204`).
- **Acceptance:** returned residual matches an independent `||phi·Q − P||` recomputation; normalization is idempotent (re-normalizing is a no-op).

---

#### **A7 --- N-vs-precision binding (`SolveParams` ctor)** `P1` · build-in-RustMath

- **Symbol:** `rustmath_curves::belyi::solve::SolveParams::new`
- **Signature:** `pub fn new(prec_bits: u32, n: usize, digits: usize) -> Result<SolveParams, ParamError>` enforcing `prec ≥ f(N, rho)` and deriving `threshold_decimal`/`tol_decimal` from `prec`.
- **Spec:** Bind `N`, `prec`, `threshold_decimal`, `tol_decimal` so truncation error `rho^N` and SVD tolerances are consistent for a 100-digit target. Today all four are independent free arguments with test-only literals.
- **Campaign use:** **Route A Step 1 (S100)** at `prec ≥ 400` bits / `N ~ 24000`, thresholds must scale with prec.
- **State:** `missing`. `recover_forms`/`dim_s_k_svd` take `big_n`, `threshold_decimal`, `tol_decimal` as unrelated args (`modular_forms_hp.rs:540-549`); `solve_belyi_map` hard-codes `JacobiSvdOptions::new(prec,80,"1e-70","1e-40")` (`genus0_map.rs:41`) and `nrows=ncols+6` (`:27`) regardless of `N`/`prec`. No coupling logic anywhere.
- **Deps:** feeds A3, A4, A6.
- **Acceptance:** for a chosen `(N, digits)` the ctor picks a `prec` and thresholds under which the `(5,3,3)` kernel is correctly ranked, and rejects an under-precisioned request.

---

#### **A8 --- EXT-dump → `MpMatrix` reader (or external-SVD bridge) for `N ~ 24000`** `P2` · build-in-RustMath

- **Symbol:** `rustmath_curves::belyi::modular_forms_hp::read_ext_matrix` + `recover_forms_from_matrix`
- **Signature:** `pub fn read_ext_matrix(path:&str) -> Result<MpMatrix, SvdError>` ; `pub fn recover_forms_from_matrix(m:&MpMatrix, rho:&rug::Float, threshold_decimal:&str, tol_decimal:&str) -> Vec<Vec<Complex>>`
- **Spec:** Reconstruct the streamed EXT limb dump back into an `MpMatrix` (or accept an external SVD result) so the `N ~ 24000` matrix can be solved without the in-memory OOM. Currently the EXT format is **write-only** from Rust.
- **Campaign use:** **Route A Step 1 (S100)** at `N ~ 24000`, which cannot go through the in-memory `assemble_scaled_ami`/`recover_forms`.
- **State:** `missing`. `dump_scaled_ami_streamed` doc (`modular_forms_hp.rs:220-233`) cites the ~1.6GB/`dim=3001` OOM of the rayon fold; grep finds **no** `read_ext`/`from_ext`/`load_matrix` consumer, only writers (`modular_forms_hp.rs:301-366, 681-743`).
- **Deps:** the streamed writer format (`dump_scaled_ami_streamed`, `:236`); `jacobi_svd` (`mp_svd.rs:234`).
- **Acceptance:** round-trip a small `assemble_scaled_ami` matrix through write→read and confirm bit-identical limbs; SVD on the reconstructed matrix matches the in-memory SVD kernel.

---

### P0/P1 --- The regular statement (symbolic deleted-sheet + Q(u) Galois certifier)

---

#### **B1 --- Symbolic deleted-sheet resolvent over Q(u)** `P1` · build-in-RustMath

- **Symbol:** `rustmath_curves::belyi::deleted_sheet_resolvent`
- **Signature:** `pub fn deleted_sheet_resolvent(p: &[Rational], q: &[Rational]) -> Vec<FunctionFieldElem>` (result is `Q(u)[X]`, `deg_X = 23`)
- **Spec:** From exact numerator `P` and denominator `Q` of the degree-24 Belyi map, form the **generic** (parameter-`u`) deg-23 resolvent `R(X,u) = (P(X)Q(u) − P(u)Q(X)) / (X−u)` in `Q(u)[X]` --- the **regular** statement, not a numeric specialization at a rational `t0`.
- **In/out:** in `(&[Rational], &[Rational])` → out `Vec<FunctionFieldElem>` (coefficients in `Q(u)`).
- **Campaign use:** **S98/S100** route to a *regular* `Gal = M23 / Q(u)` statement. Today `is_m23_q_realized()` rests only on Frobenius statistics of a *specialized* number field at one rational `t0/x0` (proves the group for that fibre, not the generic cover).
- **State:** `missing`. **No** function-field / `Q(u)` polynomial type is used anywhere in the DECIDE half: `specialize_numerator` (`audit.rs:221`) hard-substitutes a numeric `t0:&Rational`; every resolvent in the tree (`rustmath-polynomials/src/resolvent.rs`, `rustmath-galois/src/quintic.rs`, `rustmath-numberfields/src/panayi.rs`) is over `Integer`/`Q`, none over `Q(u)`. No `Q(u)[X]` constructor exists.
- **Deps:** a `FunctionFieldElem` / `Q(u)` polynomial type (new); `specialize_numerator` structure for the `A^2 B − t·λ R^5 S` shape (`audit.rs:221`).
- **Acceptance:** specializing `R(X,u)` at a rational `u=t0` reproduces the deg-23 primitive residual that `audit_m23_residual` (`audit.rs:409`) produces via `specialize_numerator`+`div_linear` --- i.e. the symbolic and numeric splits agree on many `t0`.

---

#### **B2 --- Regular Galois-over-Q(u) certifier (regular `Gal = M23`)** `P2` · Sage-assisted (delegate cross-check; author bookkeeping in RustMath)

- **Symbol:** `rustmath_curves::belyi::certify_regular_gal_m23_over_qu`
- **Signature:** `pub fn certify_regular_gal_m23_over_qu(resolvent_in_qu_X: &[FunctionFieldElem]) -> bool`
- **Spec:** Prove `Gal(R / Q(u)) = M23` **regularly** --- via specialization / Hilbert-irreducibility bookkeeping plus a *proven* (not merely Chebotarev-saturated) group identification. Upgrades `is_m23_q_realized()` from statistical single-fibre to a proven regular certificate --- the true S100 target.
- **Campaign use:** the endpoint proof-grade certifier.
- **State:** `missing`. The only deg-23 certifier is `transitive23::classify` (`transitive23.rs:277`) --- statistical, needs saturation, takes a *specialized* integer polynomial's Frobenius types. No `Q(u)`/function-field certifier exists. `pipeline.rs:530` explicitly defers to an external **OSCAR `galois_group`** cross-check ("OSCAR galois_group cross-check to follow"), confirming the in-crate path is not a proof.
- **Deps:** B1 (needs `R(X,u)` as input); `transitive23::classify` for the statistical prior; an external OSCAR/Sage `galois_group` oracle for the proof-grade cross-check (no in-tree CAS oracle exists --- see §1c note).
- **Recommendation:** **delegate the group-identity proof to Sage/OSCAR** (`galois_group` over `Q(u)` / Hilbert irreducibility), keep the specialization bookkeeping and the `bool` gate in RustMath. This matches the code's own deferred plan (`pipeline.rs:530`).
- **Acceptance:** on a known regular-M23 test resolvent, agree with OSCAR/Sage `galois_group`; reject a resolvent whose generic group is a proper subgroup.

> **Bridge note (not a build item):** the mechanical path *exact `P,Q` → `is_m23_q_realized()`* already **exists** via `decide_nonprovisional` (`pipeline.rs:534`) → `audit_m24` → `audit_m23_residual` (`pipeline.rs:552-572`). It is the *specialized/statistical* route. The remaining distance is exactly B1 + B2 (the regular `Q(u)` statement and a proof-grade certifier). Do **not** rebuild the numeric wiring.

---

### P1/P2 --- Route B: the p-adic surface-lift stack

---

#### **C1 --- Multivariate Hensel/Newton bivariate lift to `Z_17[[u,v]]`** `P1` · build-in-RustMath (biggest Route-B gap)

- **Symbol:** `rustmath_polynomials::poly_system::PolySystem::newton_lift_bivariate`
- **Signature:** `pub fn newton_lift_bivariate(&self, seed: &[MPowerSeries<PadicRational>], base: i64 /*=17*/, uv_order: usize, p_prec: u32) -> Result<Vec<MPowerSeries<PadicRational>>>`
- **Spec:** Given a mod-17 seed solution, run **coupled** multivariate Newton iteration solving `J·δ = −F` over the bivariate power-series ring in `(u,v)` with `Z_17` coefficients, doubling both the `(u,v)`-truncation order and the p-adic precision each step (quadratic convergence). Return `z_i(u,v)`.
- **In/out:** in `(&[MPowerSeries<PadicRational>], i64, usize, u32)` → out `Result<Vec<MPowerSeries<PadicRational>>>`.
- **Campaign use:** **Route B Step 1 (S100)** --- the surface lift `z_i(u,v)` out of the `F_17` point.
- **State:** `partial`. Scalar pieces exist: `PolySystem::jacobian_mod`/`evaluate_mod` (`poly_system.rs:136,153`) do Newton mod `p^k` for **integer** points only; `MPowerSeries` is a **container** with no solver (`multi_power_series_ring_element.rs:90`); matrix solve over a field exists (`linear_solve.rs`). **No driver couples them** (`grep newton.?lift|hensel.*system` = only scalar `poly_system` tests + univariate `poly_arith.rs:243`).
- **Deps:** `PolySystem::jacobian_mod` (`poly_system.rs:136`), `MPowerSeries` (`multi_power_series_ring_element.rs:90`), `Matrix::solve`/`kernel` over a field (`linear_solve.rs`), `PadicRational` (`padic_rational.rs`).
- **Acceptance:** lift a small known bivariate system (rank-full Jacobian) and confirm `F(z(u,v)) ≡ 0 mod (p^k, (u,v)^order)`; check the framed-ideal rank-75 gate (C6) passes at the seed.

---

#### **C2 --- Hermite-Padé / rational-function reconstruction** `P1` · Sage-attractive (or build in RustMath)

- **Symbol:** `rustmath_polynomials::pade::hermite_pade`
- **Signature:** `pub fn hermite_pade(series: &[Rational], num_deg: usize, den_deg: usize) -> Option<(Vec<Rational>, Vec<Rational>)>` (+ a simultaneous/vector variant)
- **Spec:** From a truncated power series `z_i(u)` recover the minimal `(p(u), q(u))` with `z_i = p/q + O(u^N)` (Padé / minimal approximant), to decide membership in `Q(u)`.
- **Campaign use:** **Route B Step 2 (S100)** --- certify each sliced coordinate `z_i(u)` (on slices `v = au+b`) is a rational function so the parametrization descends to `Q(u)`.
- **State:** `missing`. `grep 'hermite_pade|pade|minimal.*approximant'` across all crates = **0** code hits (only the number-analog `rational_reconstruct`, `recognize.rs:160`). `resultant_in_t` (`bivariate.rs:208`) and univariate half-gcd are reusable building blocks.
- **Deps:** univariate half-gcd / `resultant_in_t` (`bivariate.rs:208`).
- **Recommendation:** **Sage is genuinely attractive here** (`matrix.pade` / `berlekamp_massey` / `QQ['u']` rational reconstruction). This is the one Route-B piece where delegation clearly beats authoring, since RustMath has no Padé primitive at all. Otherwise build from half-gcd.
- **Acceptance:** on a known `p/q` truncated to `2·max(deg)+1` terms, recover `(p,q)` exactly; returns `None` when the series is not rational to the given order.

---

#### **C3 --- Multi-prime CRT + bounded-height rational reconstruction driver** `P1` · build-in-RustMath (thin wrapper)

- **Symbol:** `rustmath_integers::crt::crt_rational_reconstruct`
- **Signature:** `pub fn crt_rational_reconstruct(residues: &[(Integer, Integer)]) -> Option<Rational>`
- **Spec:** Combine per-prime residues via CRT, then run bounded-height rational reconstruction with early-exit once the modulus exceeds twice the coefficient-height bound.
- **Campaign use:** **S100 closure** --- reconstruct exact `Q` coefficients of the lifted/parametrized objects from many `F_p` images.
- **State:** `exists-unwired`. Both halves exist: `chinese_remainder_theorem` (`crt.rs:17`) and `rational_reconstruct` (`recognize.rs:160`); `disc.rs:7` comment describes exactly this CRT+Hadamard-bound pattern but only for a discriminant, not exposed as a reusable combinator. **No single function chains them.**
- **Deps:** `chinese_remainder_theorem` (`crt.rs:17`), `rational_reconstruct` (`recognize.rs:160`).
- **Acceptance:** reconstruct a chosen rational with numerator/denominator height `~10^9` from residues mod ~5 primes, with correct early-exit.

---

#### **C6 --- Framed-ideal rank-75 gate over F_p** `P1` · build-in-RustMath (wiring only)

- **Symbol:** wiring of `groebner_basis_field::<PrimeField>` + `Matrix::<PrimeField>::rank`
- **Signature (present):** `groebner_basis_field<R:Field>(gens, ordering, &GroebnerBudget) -> Result<...>` ; `Matrix<F:Field>::rank() -> Result<usize>`
- **Spec:** Reduce the framed ideal over `F_p` and/or take the rank of the 75-row Jacobian over `F_p` to certify the transversality / rank-75 condition at the `F_17` point.
- **Campaign use:** **Route B gate (S100)** --- the framed-ideal Jacobian must have rank 75 for the lift (C1) to be well-posed.
- **State:** `exists-unwired`. `groebner_basis_field` is generic over `R:Field` (`groebner.rs:1043`), `PrimeField:Field` exists (`prime_field.rs:289`), matrix rank over a field at `linear_solve.rs:129`. All present; needs instantiation at `p` and wiring to the framed Jacobian (entries from `PolySystem::jacobian_mod`, `poly_system.rs:136`).
- **Deps:** `PolySystem::jacobian_mod`, `PrimeField`, `Matrix::rank`.
- **Acceptance:** at the `F_17` seed the 75-row Jacobian returns `rank == 75`; a deliberately degenerate point returns `< 75`.

---

#### **C4 --- F_p and Q_p univariate factorization (closure)** `P2` · build-in-RustMath (wiring only)

- **Symbols (present):** `fp_factor::factor(&[i64], p) -> Vec<Vec<i64>>` (`fp_factor.rs:375`) ; `om_factorization(&UnivariatePolynomial<Rational>, p, prec) -> Result<OmFactorization>` (`om_factorization.rs:244`).
- **Spec/use:** **S96 closure** --- split the deg-442 factor, identify local factors / ramification in Route B.
- **State:** `exists-unwired`. Fully implemented (`fp_factor.rs:375` + `berlekamp_factor_gf:558`; `om_factorization.rs:244` + `padic_factor.rs:125`); only needs a call site in the campaign pipeline.
- **Acceptance:** factor a known deg-442-with-a-quadratic-times-... test over `F_17` and over `Q_2`; residue degrees/ramification match a hand computation.

---

#### **C5 --- Newton polygon / Montes-Ore over Q_2 (t=v/4 stratum + deg-442 closures)** `P2` · build-in-RustMath (wiring only)

- **Symbols (present):** `NewtonPolygon::of_rational_polynomial(&[Rational], p=&2) -> Result<NewtonPolygon>` (`newton_polygon.rs:122`, slopes `:162`, root_valuations `:183`) ; `om_factorization(f, 2, prec)` (`om_factorization.rs:244`) ; MacLane tower (`maclane.rs:625`).
- **Spec/use:** **S96** --- analyze the `t=v/4` (2-adic) stratum and close the deg-442 factor.
- **State:** `exists-unwired`. Prime-agnostic, `p=2` fine; two independent Newton-polygon impls (`newton_polygon.rs:122` vs `polynomials/src/newton.rs:73`). **Use the rings/padics + `om_factorization` path for exact p-adic work.** Not invoked by any campaign pipeline.
- **Acceptance:** slopes/root-valuations at `p=2` for a chosen Eisenstein-ish test match by hand; `om_factorization` leaf `e,f` consistent with the Newton polygon.

---

#### **C7 --- Conic isotropy via Hilbert symbols over Q (descent obstruction)** `P2` · build-in-RustMath (re-point only)

- **Symbols (present):** `DiagonalConicQ::verdict(bad_locus_clear:bool) -> Result<Verdict<ConicBrauerReport>, ConicError>` (`conic.rs:173`) ; `hilbert_symbol(&Rational, &Rational, Place) -> Result<i8, HilbertError>` (`hilbert.rs:89`).
- **Spec/use:** **closure/descent** --- decide whether the diagonal ternary conic arising from the field of moduli / descent datum is isotropic (product-of-Hilbert-symbols / Brauer class).
- **State:** `exists-unwired`. Already consumed by the Belyi portal (`portal.rs`, `bad_locus.rs`, `descent.rs` reference `conic`/`hilbert`), so wiring exists for the Belyi path; **may need re-pointing** at the S100 descent datum.
- **Acceptance:** a conic known isotropic returns "has a rational point"; a known-anisotropic conic returns the Brauer obstruction; agrees with an independent Hilbert-symbol product.

---

### P2 --- Recognition helpers (used across routes)

---

#### **D1 --- Real (multi-vector) PSLQ / simultaneous integer relation** `P2` · build-in-RustMath

- **Symbol:** `rustmath_matrix::lll::integer_relation`
- **Signature:** `pub fn integer_relation(x: &[rug::Float], prec_bits: u32) -> Option<Vec<BigInt>>` (or a true `pslq(x:&[Float], tol) -> Option<Vec<BigInt>>`).
- **Spec:** Given a real/complex vector, return a nonzero integer relation `Σ a_i x_i = 0` (or `None`). Snaps a **whole** coefficient tuple to one common number field simultaneously (shared minpoly / shared denominator) --- which per-number `recognize_complex_algebraic` cannot guarantee. Reuse the LLL that returns the unimodular `U` (`rustmath-matrix/src/lll.rs:75`) or `Lattice::short_vectors` (`lattice.rs:510`).
- **Campaign use:** when the Belyi map is defined over a number field `L` (not `Q`), recognize the 25-vector coefficients in **one** field with a consistent primitive element. `bridge.rs:58` explicitly defers "the common-field embedding + exact substitution over L" as follow-up.
- **State:** `partial`. Every "PSLQ" token in the tree is a comment or placeholder: `rustmath-calculus/src/minpoly.rs:259` ("placeholder for a full PSLQ/LLL implementation" --- brute-forces `a,b∈-10..=10` for `x^2−a/b`); `newton_hp.rs:250` comment only. No `fn pslq|integer_relation` exists (`geometry::positive_integer_relations` is an unrelated Hilbert-basis routine). Real LLL engines exist (`lll.rs:75`, `lattice.rs:298/376`) but none wrap value-recognition.
- **Deps:** `lll_reduce` returning `U` (`lll.rs:75`), `Lattice::short_vectors` (`lattice.rs:510`).
- **Acceptance:** recover a known linear integer relation among 6 reals to 400-bit precision; return `None` for a generically independent tuple.

---

#### **D2 --- Float→best-rational recognizer via continued fractions** `P2` · build-in-RustMath (cheap fallback)

- **Symbol:** `rustmath_rationals::continued_fraction::ContinuedFraction::from_real`
- **Signature:** `pub fn from_real(x: &rug::Float, max_denom: &Integer) -> Rational` (accept `f64` too).
- **Spec:** Recognize a rational from a real approximation via CF convergents with a denominator/precision bound --- the cheap 1-D fallback for coordinates known a priori to lie in `Q`, and an independent cross-check for the doubling-stability gate (A2).
- **Campaign use:** many Belyi coordinates are rational; faster/more transparent than LLL and gives a second opinion.
- **State:** `partial`. `ContinuedFraction` has `from_rational` (`continued_fraction.rs:27`) and `all_convergents` (`:82`) but **no** `from_real`/`from_float`. The only float→Q path is `Rational::from_f64` (`rational.rs:157`) --- exact IEEE decode, **not** best-approximation. `rational_reconstruct` (`recognize.rs:160`) is CF-based but modulus-driven.
- **Deps:** `ContinuedFraction::all_convergents` (`continued_fraction.rs:82`) (or `convergent(n)` at `:71`).
- **Acceptance:** recover `355/113` from `π`-truncation with `max_denom=200`; recover an exact rational from its 512-bit rendering within the height bound.

---

## 3. Minimal critical path --- smallest set that unblocks a candidate M23/Q witness via Route A

The fastest arrow to a `is_m23_q_realized()==true` candidate (statistical, single-fibre --- the honest current bar, not yet the regular proof) is **Route A**, and it needs only:

**A3** (`run_and_persist_belyi`) to actually run the modular-forms chain for `[2,12,5]` and persist `(P,Q)` + residual --- which requires **A4** (thread the `z_b` chart into `recover_forms`), **A6** (surface the residual `solve_belyi_map` already computes), and **A7** (couple `N`/`prec`/thresholds so the kernel is ranked correctly) --- then **A1 + A2** (the HP recognizer and the gauge-normalized, doubling-stable `snap_hp_solution`) to convert those floats into `ExactifyOutcome::CertifiedRational`. That `CertifiedRational` drops straight into the **already-built** `decide_nonprovisional` → `audit_m24` → `audit_m23_residual` → `is_m23_q_realized` (do not rebuild). 

That is the whole minimal path: **A1, A2, A3, A4, A6, A7** (six items, all `rustmath-curves`/`rustmath-numberfields`, no Sage). **A8** (EXT→MpMatrix reader) becomes necessary only if `N ~ 24000` exceeds the in-memory assembler; defer it until the in-memory path OOMs. Everything in Route B (C-series) and the regular-statement proof (B1/B2) is a **separate, higher-assurance** track and is not on the critical path to a first candidate witness --- B1+B2 upgrade that candidate from statistical to a proven **regular M23/Q(u)** certificate, which is the eventual endpoint but not the unblock.