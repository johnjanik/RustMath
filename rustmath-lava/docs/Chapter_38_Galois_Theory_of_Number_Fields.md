# Chapter 38 — Galois Theory of Number Fields

**Authors:** C. Fieker ∘ J. Klüners, K. Geißler
**Handbook part:** VI — Global Arithmetic Fields
**Handbook pages:** 961–995 (PDF pages 1094–1128)

---

## Scope and overview

The Galois theory of number fields, as implemented in Magma, addresses three algorithmically
*independent* problems, each treated by a different method despite their common theoretical
grounding in the main theorems of Galois theory:

1. **Automorphism groups** — the group of automorphisms of a (normal) number field, and the
   group of automorphisms of the normal closure. Computing automorphisms of normal extensions
   of **Q** (and abelian extensions of number fields) can be viewed as a special case of
   factoring polynomials over number fields: automorphisms correspond one-to-one to the roots
   of the defining equation lying in the field. The implementation uses a combinatorial
   variation of this idea. **These algorithms apply only to normal fields** — they cannot find
   non-trivial automorphisms of non-normal fields.

2. **Galois groups** — the Galois group of the normal closure of a number field or polynomial.
   Returned as a permutation group acting on the (p-adically approximated) roots of the
   defining polynomial; the splitting field itself is *not* directly part of the computation.
   The explicit action on the roots permits algebraic reconstruction of arbitrary subfields of
   the splitting field, up to and including the splitting field itself.

3. **Subfields** — computation of all subfields (or all subfields of a given degree) of a
   number field. This is independent of the Galois group computation, is mainly combinatorial,
   and is in fact usually the *first* step of the Galois group computation.

Applications also covered: subfield/subfield-tower construction of the splitting field;
solvability by radicals (when the Galois group is solvable); and basic Galois cohomology
(action of automorphisms on the ideal class group, the multiplicative group, and derived
objects).

---

## 38.1 Automorphism Groups

Computes automorphisms of an algebraic field, the group they form, and field invariants
related to that group (decomposition / ramification / inertia groups and their fixed fields).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Automorphisms(F)` | Returns the automorphisms of algebraic field `F` as a sequence of maps. | If `Abelian := true`, an efficient abelian-specific algorithm is used **[Klü97, AK99]**. If `F` is not normal, automorphisms are obtained by a variation of the polynomial-factorisation algorithm. |
| `AutomorphismGroup(F)` | For `F` a simple normal or simple abelian extension of **Q**, returns the automorphism group `G` as a permutation group of degree `n = deg F`, plus the power structure `Aut` of all automorphisms and the transfer map φ: G → Aut. Errors if `F` is not normal over **Q**. | `Abelian := true` selects the abelian algorithm **[Klü97, AK99]**; otherwise the factorisation-variation method. |
| `AutomorphismGroup(K, F)` | Group of `K`-automorphisms of `F` as a permutation group, with the list of automorphisms and the connecting map. Computes the automorphism group of `F` over **Q** first. | Reduction to the absolute case. |
| `DecompositionGroup(p)` | For a prime ideal `p` of the maximal order of an absolute normal field `F` with automorphism group `G`: the subgroup `U = { s ∈ G : s(p) = p }`. Errors if `F` not normal over **Q**. | Direct group-theoretic definition. |
| `RamificationGroup(p, i)` | The *i*-th ramification group `U = { s ∈ G : s(x) − x ∈ pⁱ⁺¹ for all x in M }`, `M` the maximal order. Errors if not normal over **Q**. | Direct definition. |
| `RamificationGroup(p)` | Abbreviation for `RamificationGroup(p, 1)`. | — |
| `InertiaGroup(p)` | Abbreviation for `RamificationGroup(p, 0)`. | — |
| `FixedField(K, U)` | For normal `K/Q` and a subgroup `U` of `AutomorphismGroup(K)`: the subfield `L` fixed by `U`. Inverse to `FixedGroup`. | Galois correspondence. |
| `FixedField(K, S)` | For algebraic `K` and a list `S` of automorphisms: the maximal subfield fixed by `S`. | Galois correspondence. |
| `FixedGroup(K, L)` | For normal `K/Q` and subfield `L` (or a sequence of field elements, or a single element `a`): the subgroup of `AutomorphismGroup(K)` fixing `L` (resp. the elements / `a`). Inverse to `FixedField`. Errors if not normal over **Q**. | Galois correspondence. |
| `DecompositionField(p)` | Abbreviation for `FixedField(K, DecompositionGroup(p))`. | — |
| `RamificationField(p, i)` / `RamificationField(p)` | Abbreviation for `FixedField(K, RamificationGroup(p, i))` (resp. `…(p)`). | — |
| `InertiaField(p)` | Abbreviation for `FixedField(K, InertiaGroup(p))`. | — |
| `FrobeniusElement(K, p)` | A Frobenius element at `p` in the Galois group of the Galois closure of `K`: a permutation of the roots of a defining polynomial of `K` (recoverable as `DefiningPolynomial(A)` for an Artin representation `A` of `K`). Well-defined up to conjugacy and modulo inertia. | p-adic Frobenius lifting. |

*Worked examples in the handbook:* H38E1 (automorphisms of `x⁴ − 4x² + 1`); H38E2 (ramification/inertia/decomposition groups and fields, splitting behaviour via double cosets); H38E3 (Frobenius elements of a D₅ polynomial at p = 2 and p = 5).

---

## 38.2 Galois Groups

**Problem and methods.** Finding Galois groups of normal closures is hard in general. The two
families of practical algorithms are the **absolute resolvent method [SM85]** and the
**method of Stauduhar [Sta73]**. Magma's implementation extends Stauduhar's method
**[GK00, Gei03]** together with recent work of Klüners and Fieker **[FK12]**.

**Capabilities.** Computes Galois groups for polynomials over **Z**, **Q**, number fields and
global function fields, and irreducible polynomials over function fields over **Q**, with **no
a-priori degree restriction** (degree > 50 may be infeasible in time/memory, but degree > 200
has succeeded). Unlike the absolute resolvent method, it provides the **explicit action on the
roots**. An older method, limited to degree ≤ 23, is still available on demand.

**Algorithmic idea.** Stauduhar's method traverses the lattice of transitive permutation
groups from the symmetric group `Sₙ` down to the actual Galois group, using **relative
resolvents** — polynomials whose splitting fields are subfields of the splitting field of `f`,
computed from p-adic (or complex) approximations of the roots of `f`.

- **Imprimitive case (the field has subfields):** the starting point in the subgroup lattice
  is changed to get as close as possible to the actual Galois group, by first computing
  subfields of a stem field of `f`. The Galois group is then found as a subgroup of an
  intersection of suitable wreath products (easily computed), which is a good starting point.
- **Primitive case (no subfields):** a combination of Stauduhar's method and the absolute
  resolvent method. The Frobenius automorphism of the underlying p-adic field (or complex
  conjugation, with complex approximations) already determines a subgroup of the Galois group,
  used to speed up the primitive case.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GaloisGroup(f)` | Galois group of a splitting field of polynomial `f` (over **Z**, **Q**, a number field, or an order): the subgroup of permutations of the roots corresponding to field automorphisms. **Not** a proven result — call `GaloisProof` if certainty is needed. Parameters: `Prime` (prime for splitting-field computation), `NextPrime`, `ShortOK`, `Ring`. | Variant of Stauduhar's algorithm **[GK00, Gei03, FK12]**, no dependency on the classification of transitive groups, hence no degree limit. |
| `GaloisGroup(K)` | Galois group of a normal closure of number field `K`, as an abstract permutation group on the roots of the defining polynomial in a suitable splitting field. **Not** proven (see `GaloisProof`). Parameters include `Prime`, `NextPrime`, `Current`, `Subfields` (default `true`), `Old` (use the ≤ degree-23 method), `Type` (default `"p-Adic"`), `Prec` (default 20), `Time`. | Same Stauduhar variant **[GK00, Gei03, FK12]**. |
| `GaloisProof(f, S)` / `GaloisProof(K, S)` | Given the (conditional) result `S` of a `GaloisGroup` computation for an irreducible polynomial over **Z** or an absolute number field, attempts to prove the conditional steps. | Shows a suitable **absolute resolvent** polynomial has a factor of a specific degree that distinguishes the candidate groups (absolute resolvent method **[SM85]**). |
| `GaloisRoot(f, i, S)` | The *i*-th root (in the Galois-process ordering) to a given precision. Precision via `Prec` (p-adic precision) or via `Bound` `B` (then `pᵏ > B`). | p-adic root approximation in the ordering fixed by `S`. |
| `GaloisRoot(i, S)` | The *i*-th conjugate of the primitive element used during the computation. Same precision controls. | As above. |
| `Stauduhar(G, H, S, B)` | A single step of the Stauduhar method: with `G` containing the Galois group, `H` a (maximal) subgroup, and `B` a bound on the absolute value of the complex roots, decides whether some `g ∈ G` gives `GaloisGroup ⊆ Hᵍ`. Returns: `1` (proven subgroup of `Hᵍ` up to precision), `-1` (proof it lies in a proper subgroup of `G`, maybe in `Hᵍ`), `-2` (maybe in `Hᵍ` but no proof of proper containment), `0` (not in any conjugate of `H`). Secondary returns: the conjugating `g`, a proven/heuristic flag, and the separating invariant. Parameters: `Coset` (a transversal of `G/H`), `PreCompInvar`, `AlwaysTransform`. | Core Stauduhar step **[Sta73, GK00]** evaluating a `G`-relative `H`-invariant. |
| `IsInt(x, B, S)` | For `x` in the splitting field of `S` and bound `B`: decides if there is `y ∈ Z` (or the relevant extension) with `y = x` up to the precision of `x` and `|y| < B`; returns `y` if so. | Bounded p-adic-to-algebraic recognition. |

*Worked examples:* H38E4 (`GaloisGroup` of `x⁶ − 108`, `x³² − x¹⁶ + 2`, and relative cases via `galpols` / `PolynomialWithGaloisGroup`).

### 38.2.1 Straight-line Polynomials

Invariants in computational Galois theory are multivariate polynomials invariant under a
permutation group. Rather than the general invariant-theory module (Chapter 110), Galois
theory uses **straight-line polynomials**: polynomials represented as branch-free programs.
Category `RngSLPol`, elements `RngSLPolElt`. This makes some operations (e.g. representing
`(a−b)¹⁰⁰⁰(a+b)¹⁰⁰⁰ − (a²−b²)¹⁰⁰⁰`) trivial and fast to evaluate, while others (e.g. expanding
into a standard polynomial to detect identity) are very difficult.

| Intrinsic | Description |
|-----------|-------------|
| `SLPolynomialRing(R, n)` | Ring of multivariate straight-line polynomials over `R` with `n` indeterminates. Parameter `Global`. |
| `Name(R, i)` / `R . i` | The *i*-th indeterminate. |
| `BaseRing(R)` / `CoefficientRing(R)` | The coefficient ring of `R`. |
| `Rank(R)` | Number of independent indeterminates over the coefficient ring. |
| `SetEvaluationComparison(R, F, n)` | Prepare probabilistic comparison of SL-polynomials, by evaluation at `n` random tuples from the finite field `F`. |
| `GetEvaluationComparison(R)` | The finite field and number of samples used; `(false, undefined)` if not set. |
| `x * y`, `x + y`, `x - y`, `- x` | Arithmetic on SL-polynomials. |
| `Derivative(x, i)` | The *i*-th partial derivative of the SL-polynomial `x`. |

### 38.2.2 Invariants

At the core of the Stauduhar step is deciding, for `G` and a maximal subgroup `U`, whether the
Galois group is a subgroup of `U` (given it is in `G`). This is done by evaluating a
**G-relative U-invariant** polynomial `f ∈ Z[x₁,…,xₙ]` (or `f ∈ F_q[t][x₁,…,xₙ]` in prime
characteristic), where `n` is the degree of `G < Sₙ`. Invariants are represented as
straight-line polynomials for compact storage and fast evaluation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GaloisGroupInvariant(G, H)` | For `H` maximal in transitive `G < Sₙ`: a `G`-relative `H`-invariant. Parameters: `DoCost` (returns an evaluation-cost estimate and a thunk, to compare invariants by complexity before committing), `Worklevel` (restrict to certain invariant types; higher = more expensive; `false` signals no invariant found at that level). | Compares group-theoretic properties of `(G, H)` to find easy-to-evaluate special invariants; falls back to generic invariants. |
| `RelativeInvariant(G, H)` | For `H < G` (not necessarily maximal): a `G`-relative `H`-invariant. Parameters: `IsMaximal` (assume `H` maximal, skip chain), `Risk` (use `GaloisGroupInvariant` at each level). | Three phases: (1) compute a chain `H = U₀ < U₁ < … < G` of maximal subgroups; (2) one invariant per maximal pair `Uᵢ < Uᵢ₊₁`; (3) combine into a `G`-relative `H`-invariant. By default uses generic invariants (orbit sums of monomials); `Risk` uses special invariants (faster but the `G`-stabiliser may be too large to guarantee correctness). |
| `CombineInvariants(G, H1, H2, H3)` | Given `G < Sₙ` and three maximal subgroups, two with known invariants, derive an invariant for `H3`. Inputs for `H1`, `H2` are tuples `(subgroup, invariant[, Tschirnhaus transform])`. | Typical use: `H1`, `H2` index-2 subgroups of `G`, giving a third index-2 subgroup `H3`. Requires `core(H1 ∩ H2) ⊆ H3`; useful only when the index of the core is small. |
| `IsInvariant(F, p)` | Probabilistic test whether SL-polynomial `F` is invariant under permutation `p` (`F^p = F`): evaluates `F` at random points in a large finite field, permutes by `p`, re-evaluates; agreement ⇒ "most likely invariant", disagreement ⇒ "definitely not". `Sign := true` tests `F^p = −F`. | Schwartz–Zippel-style probabilistic identity test. For a proof, convert to a standard multivariate polynomial (often infeasible due to degree/term count). |
| `Bound(I, B)` (integer `B`) | An integer `M` with `\|I(x₁,…,xₙ)\| ≤ M` whenever all `\|xᵢ\| ≤ B`. | Bound on the size of an evaluation of `I`. |
| `Bound(I, B)` (power series `B`) | A power series `M` such that whenever each `xᵢ` has coefficients bounded by those of `B`, the coefficients of `I(x₁,…,xₙ)` are bounded by those of `M`. | Power-series coefficient bound. |

### 38.2.3 Subfields and Subfield Towers

The result of a Galois group computation contains the abstract group *and* the explicit action
on the roots in a splitting field. Combined with group-pair invariants, this allows computing
arbitrary subfields of the splitting field.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GaloisSubgroup(K, U)` / `(S, U)` / `(f, U)` | Given a number field `K` / Galois data `S` / polynomial `f`, and a subgroup `U < G` (the Galois group): a defining polynomial for the subfield of the splitting field fixed by `U`. | Invariant evaluation on the explicit root action. |
| `GaloisQuotient(K, Q)` / `(f, Q)` / `(S, Q)` | All subfields of the splitting field whose Galois group is isomorphic to permutation group `Q`. | Finds all subgroups `U ≤ G` such that the action of `G` on cosets `G/U` is isomorphic to `Q`. |
| `GaloisSubfieldTower(S, L)` | For Galois data `S` (third return of `GaloisGroup`) and a subgroup chain `U₁ > U₂ > … > Uₛ`: the corresponding tower of fixed fields `Kᵢ`. **Currently only over Q / absolute extensions of Q.** Returns: largest field in the tower; a sequence of per-step data tuples `(invariant, Tschirnhaus transform, transversal Uᵢ/Uᵢ₊₁)`; a reconstruction function (p-adic conjugates → algebraic element); and a precision-bound function. Parameters: `Risk`, `MinBound`, `MaxBound`, `Inv` (supply `Uᵢ`-relative `Uᵢ₊₁`-invariants). | Iterated fixed-field construction via relative invariants; `Risk` uses the risky `RelativeInvariant` for non-maximal pairs. `MinBound`/`MaxBound` tune internal p-adic precision (estimates grow pessimistic as the chain lengthens). |
| `GaloisSplittingField(f)` | For `f` over **Z**, **Q**, or an absolute number field: the splitting field as a tower of fields. Returns the field, the roots (if `Roots := true`), the Galois group, and optionally the automorphisms. Parameters: `Galois` (pass in `GaloisGroup(f)` output), `Roots` (default `true`), `AllAuto`, `Stab` (default `true` ⇒ tower from the stabiliser chain of `{1}, {1,2}, …`), `Chain` (a subgroup chain, first element must be the full group `G`), `Inv`, `Name`. | Builds the tower from a subgroup chain (default: point-stabiliser chain) using the explicit root action and invariants. |

*Worked examples:* H38E5 (`GaloisSplittingField` of `x³ − 2` with various `Chain`/`Name` options; degree-10 `[2⁴]5` group `10T8`, `GaloisSubgroup`, `GaloisSubfieldTower` and p-adic reconstruction via `Bound`/`Bnd`/`GaloisRoot`/`Reco`).

### 38.2.4 Solvability by Radicals

For `f ∈ Z[t]` with **solvable** Galois group, the roots can be written as nested radicals; no
good general algorithm is known, so Magma uses the explicit permutation action of the Galois
group on the p-adic roots to construct such a representation.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SolveByRadicals(f)` | For `f ∈ Z[t]` with solvable Galois group: a splitting field as a tower of radical extensions, with algebraic representations of the roots; the third return value gives the non-trivial roots of unity used. Parameters: `Prime`, `Name`, `Galois`, `UseZeta_p` (if `true`, root expressions use pure radicals and roots of unity; otherwise radical expressions for the needed roots of unity are also computed), `MaxBound` (upper bound on internal p-adic precision). | Builds a radical tower from the explicit action of the (solvable) Galois group on the p-adic roots. |
| `CyclicToRadical(K, a, z)` | For `K/k` with cyclic automorphism group of order `n` generated by `K.1 ↦ a`, and `z` an `n`-th root of unity in `k`: returns a field `L ≅ K` that is a Kummer extension (defining polynomial `tⁿ − b`, `b ∈ k`), plus the roots of `f` in `L` and the roots of unity used. | Kummer theory / Lagrange resolvents. |

*Worked example:* H38E6 (`SolveByRadicals` of a degree-6 polynomial, showing each tower step is radical `xⁿ − a`).

### 38.2.5 Linear Relations

Finding all linear (additive) relations among the roots of an integral polynomial. Trivial if
the splitting field is built explicitly, but otherwise non-obvious; two algorithms find such
relations and a third verifies arbitrary ones.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LinearRelations(f)` | For monic integral `f`: a basis (as a matrix) for the **module of additive relations** between the roots of `f` (roots ordered as by `GaloisRoot` / the Galois group computation), plus the Galois data. Parameters: `Proof` (default `true`), `Galois`, `UseAction`, `UseLLL` (default `true`), `Power` (relations between `Power`-th powers of the roots), `kMax`, `LogLambdaMax`. | Algorithm of **[dGF07]** (finding integral linear dependencies of algebraic numbers); LLL-based by default. |
| `LinearRelations(f, I)` | As above, but for the elements of the splitting field `K = Q[x₁,…,xₙ]/J` represented by the multivariate polynomials in sequence `I` (in the roots `αᵢ`): a basis for the module of relations among those elements. Same parameters. | Algorithm of **[dGF07]**. |
| `VerifyRelation(f, F)` | For monic integral `f` and a polynomial `F` in the roots `αᵢ`: verifies whether `F` evaluated at the roots equals zero (i.e. `F` is a genuine relation). Parameters: `Galois`, `kMax`. | Algorithm of **[dGF07]**. |

*Worked example:* H38E7 (using `ShephardTodd(8)`, `InvariantRing`, `PrimaryInvariants`, a degree-24 resultant, `GaloisGroup` of order 192, then `LinearRelations` / `VerifyRelation`; based on **[BDE+]** on the conjugate dimension of algebraic numbers).

### 38.2.6 Other

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ConjugatesToPowerSums(I)` | For a sequence `I` (Galois conjugates of an algebraic number / roots of a polynomial): the power sums `Σ Iᵢʲ`, `j = 1,…,#I`. | Direct power-sum computation. |
| `PowerSumToElementarySymmetric(I)` | For `I` interpreted as power sums of an algebraic number: the elementary symmetric functions in the conjugates. Requires the characteristic of the ring to exceed the sequence length. | **Newton's identities** (Newton–Girard relations). |

---

## 38.3 Subfields

Computes all subfields of any number field, or all subfields of a given degree of a simple
absolute algebraic field or simple relative extension. **Independent of the Galois group
computation**, and with **no limit on the field degree**.

**Algorithms used:** Klüners's method **[Klü95, Klü97, KP97, Klü98]** and the newer method of
Klüners, van Hoeij and Novocin **[vHKN11]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Subfields(K, n)` | For a simple absolute algebraic field or simple relative extension `K` and integer `n > 1`: a sequence of pairs `(subfield of degree n, embedding into K)`. The sequence may contain isomorphic fields, but the embeddings are distinct. | Klüners / Klüners–van Hoeij **[Klü97, KP97, Klü98, vHKN11]**. |
| `Subfields(K)` | For an algebraic field `K`: a sequence of pairs `(subfield (except Q), embedding into K)`; may contain isomorphic fields. Parameters: `Al` (`"Default"`, `"Klueners"`, or `"KluenersvanHoeij"`), `Current`, `Proof` (default 1). | Extensions of a number field always use Klüners–van Hoeij **[vHKN11]**. For extensions of **Q**, `"Default"` chooses optimally: `"Klueners"` **[Klü95, Klü97, KP97, Klü98]** when the defining polynomial has large-degree factors over a residue field or has large coefficients; otherwise `"KluenersvanHoeij"` **[vHKN11]**, optimal when factors have small degree over several residue fields (i.e. many subfields). |

### 38.3.1 The Subfield Lattice

Subfields can be retrieved as a lattice exposing additional structure.

| Intrinsic | Description |
|-----------|-------------|
| `SubfieldLattice(K)` | The lattice of subfields of an absolute number field `K`. |
| `#L` | Number of fields in the lattice `L`. |
| `Representative(L)` / `Rep(L)` | A representative element. |
| `Bottom(L)` | The bottom element (corresponds to **Q**). |
| `Top(L)` | The top element (corresponds to the original field `K`). |
| `Random(L)` | A random element of `L`. |
| `L ! n` / `L[n]` | The *n*-th element of `L`. |
| `NumberField(e)` | The number field corresponding to lattice element `e`. |
| `EmbeddingMap(e)` | The embedding of `NumberField(e)` into the top field. |
| `Degree(e)` | The (absolute) degree of the field corresponding to `e`. |
| `e eq f` | Equality of lattice elements. |
| `e subset f` | True iff `e` is a subfield of `f`. |
| `e * f` | The smallest field containing both `e` and `f` (compositum). |
| `e meet f` | The intersection (largest common subfield) of `e` and `f`. |
| `&meet S` | Intersection of the subfields in sequence `S`. |
| `MaximalSubfields(e)` | Maximal lattice elements contained in `e`. |
| `MinimalOverfields(e)` | Minimal lattice elements containing `e`. |

*Worked example:* H38E8 (subfield lattice of `x⁸ − x⁴ + 1`, identification of √2, √−1, √3 subfields; `AbsoluteField`, `IsIsomorphic`, `OptimizedRepresentation`).

---

## 38.4 Galois Cohomology

Rudimentary functions for Galois cohomology of number fields (action of automorphisms on the
multiplicative group / S-units, ideal class group, etc.).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Hilbert90(a, M)` | For a number field `K`, automorphism `M : K → K` with fixed field `k` (so `M` generates `Gal(K/k)`, cyclic), and `a ∈ K` with `N_{K/k}(a) = 1`: find `b` with `a = b / M(b)`. Parameter `S` (sequence of prime ideals; `b` sought in the S-unit group). | **Hilbert's Theorem 90** (constructive), realised in the S-unit group. |
| `SUnitCohomologyProcess(S, U)` | For a normal number field `k` with abstract automorphism group `G`, a set `S` of primes closed under `U ≤ G`: a process for working with the cohomology of the multiplicative group of `k` (partially represented by an S-unit group). Parameters: `ClassGroup` (enlarge `S` to support current class-group generators), `Ramification` (include all ramified primes). `S` may grow during use. | S-unit group as a finitely generated `G`-module. |
| `IsGloballySplit(C, l)` | For a cohomology process `C` and a 2-cocycle `l : U × U → k` (a Magma function): decides whether `l` is split, i.e. whether a 1-cochain `m : U → k` exists with `δm = l`. Parameter `Sub` (a subgroup of the automorphism group; default the full group). | A fixed cocycle takes finitely many values ⇒ view it in a suitable S-unit group; minimise `S` (remove ideals from the support of `l`), then enlarge so `m` (if it exists) has values in the S′-unit group; the resulting finitely-generated-abelian-group problem is solved by Magma's general cohomology machinery. |
| `IsSplitAsIdealAt(I, l)` | For `U ≤ G = Aut(k)`, a 2-cocycle `l : U × U → k*`, and an ideal `I`: assuming each `l(u,v)` decomposes as `J^{x(u,v)} A(u,v)` (with `A` coprime to `J`) for all `J ∈ I^U`, define a cocycle valued in the finitely generated group `I^U`; decide whether it splits and, if so, return a 1-cochain valued in `I^U` (and `I^U`). Parameter `Sub` (default `U := G`). | Reduction to a cocycle valued in the finitely generated abelian group `I^U`. |

---

## 38.5 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[AK99]** | V. Acciaro and J. Klüners. *Computing Automorphisms of Abelian Number Fields.* Math. Comp. **68**(227):1179–1186, 1999. |
| **[BDE+]** | N. Berry, A. Dubickas, N. Elkies, B. Poonen, C. Smyth. *The conjugate dimension of algebraic numbers.* Quart. J. Math. **55**:237–252. |
| **[dGF07]** | W. de Graaf and C. Fieker. *Finding integral linear dependencies of algebraic numbers and algebraic Lie algebras.* LMS J. Comput. Math. **11**, 2007. |
| **[FK12]** | C. Fieker and J. Klüners. *Computational Galois Theory I: Invariants and Computations over Q.* Submitted, arXiv:1211.3588, 2012. |
| **[Gei03]** | K. Geißler. *Berechnung von Galoisgruppen über Zahl- und Funktionenkörpern.* PhD Thesis, TU Berlin, 2003. |
| **[GK00]** | K. Geißler and J. Klüners. *The determination of Galois Groups.* J. Symbolic Comp. **30**(6):653–674, 2000. |
| **[Klü95]** | J. Klüners. *Über die Berechnung von Teilkörpern algebraischer Zahlkörper.* Diplomarbeit, TU Berlin, 1995. |
| **[Klü97]** | J. Klüners. *Über die Berechnung von Automorphismen und Teilkörpern algebraischer Zahlkörper.* Dissertation, TU Berlin, 1997. |
| **[Klü98]** | J. Klüners. *On computing subfields. A detailed description of the algorithm.* J. Théor. Nombres Bordeaux **2**(10):243–271, 1998. |
| **[KP97]** | J. Klüners and M. E. Pohst. *On Computing Subfields.* J. Symbolic Comp. **24**(3):385–397, 1997. |
| **[SM85]** | L. H. Soicher and J. McKay. *Computing Galois Groups over the rationals.* J. Number Th. **20**:273–281, 1985. |
| **[Sta73]** | R. P. Stauduhar. *The determination of Galois Groups.* Math. Comp. **27**:981–996, 1973. |
| **[vHKN11]** | M. van Hoeij, J. Klüners, A. Novocin. *Generating subfields* (Klüners–van Hoeij–Novocin subfield algorithm), 2011. *(Cited in §38.3; not listed in the chapter's own bibliography.)* |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Polynomial-factorisation variation (automorphisms of non-normal/normal fields) | `Automorphisms`, `AutomorphismGroup` |
| Abelian-specific automorphism algorithm **[Klü97, AK99]** | `Automorphisms(:Abelian)`, `AutomorphismGroup(:Abelian)` |
| Stauduhar's method (+ extensions) **[Sta73, GK00, Gei03, FK12]** | `GaloisGroup`, `Stauduhar`, `GaloisSubgroup`, `GaloisQuotient`, `GaloisSubfieldTower`, `GaloisSplittingField` |
| Absolute resolvent method **[SM85]** | `GaloisProof` (and primitive-case `GaloisGroup`) |
| Invariant construction (group-pair, generic orbit sums) | `GaloisGroupInvariant`, `RelativeInvariant`, `CombineInvariants` |
| Probabilistic identity testing (Schwartz–Zippel style) | `IsInvariant`, `SetEvaluationComparison` |
| Integral linear dependencies (LLL-based) **[dGF07]** | `LinearRelations`, `VerifyRelation` |
| Kummer theory / radical towers | `SolveByRadicals`, `CyclicToRadical` |
| Newton's identities | `PowerSumToElementarySymmetric` |
| Subfield computation **[Klü95, Klü97, KP97, Klü98]** / **[vHKN11]** | `Subfields`, `SubfieldLattice` |
| Hilbert 90 / S-unit cohomology | `Hilbert90`, `SUnitCohomologyProcess`, `IsGloballySplit`, `IsSplitAsIdealAt` |
