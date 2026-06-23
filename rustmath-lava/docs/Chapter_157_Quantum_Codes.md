# Chapter 157 — Quantum Codes

**Handbook part:** XXI — Coding Theory
**Handbook pages:** 5233–5270 (PDF pages 5366–5405)

---

## Scope and overview

This chapter documents Magma's package for **quantum stabilizer codes** and a basic package
for **quantum Hilbert spaces**. The central result, due to Calderbank, Rains, Shor and Sloane
**[CRSS98]** (the major reference for the chapter), is that quantum stabilizer codes can be
represented in terms of *additive codes over finite fields* (see Chapter 156 for additive
codes). This reduces the construction of fault-tolerant encodings on a continuous Hilbert
space to the construction of certain discrete classical codes, so that the tools of classical
coding theory become available.

The Magma package deals **exclusively with the finite field representation** of stabilizer
codes. A quantum code is *represented* by a code over a finite field, but an actual quantum
code is a different object; many classical conventions can be ambiguous in the quantum
setting, so the handbook should be consulted before assuming classical definitions carry over.

**Representation.** Errors on N qubits are described by the *quantum error group*: combinations
of bit-flip operators X and phase-shift operators Z (plus an overall phase). Group elements
are length-2N binary vectors `(a|b)` — the **extended format** — or equivalently the length-N
vector `w = a + ωb` over GF(4) (`ω` a primitive element), the **compact format**, which is the
default in Magma. For non-binary codes over GF(q), the compact format over GF(q²) is
`w = a + λb`, where λ is returned by `QuantumBasisElement(GF(q²))`. Two errors commute iff
their finite field representations are orthogonal under the **symplectic inner product**; in
extended format `(a₁|b₁) * (a₂|b₂) = a₁·b₂ − a₂·b₁`, and in compact format over GF(4) it is
`Trace(w₁ · w̄₂)`.

**Stabilizer codes.** A quantum stabilizer code `Q` is defined by a symplectic self-orthogonal
additive code `S` (its *stabilizer code*). Undetectable errors are the words of `S⊥ \ S`
(`S⊥` the symplectic dual). The **minimum weight** of `Q` is the minimum weight of `S⊥ \ S`,
except for **self-dual** quantum codes (dimension 0, defined by symplectic self-dual `S`),
whose minimum weight is the classical minimum weight of `S`. An `[n, k]` symplectic
self-orthogonal linear code over GF(q²) generates an `[[n, n/2 − k]]` quantum stabilizer code;
a compact-format additive self-orthogonal code of N codewords gives "dimension" `log_q(N)`.

The final section (157.10) provides a basic **quantum Hilbert space** package for creating and
manipulating quantum states, including unitary transformations and measurement probabilities;
this is a first release expected to grow.

---

## 157.1 Introduction

Introductory only — establishes the qubit state-function formalism in a Hilbert space, Shor's
no-cloning observation and 1995 demonstration that quantum error-correction is possible
**[Sho94, Sho95]**, and the stabilizer-code / additive-code correspondence **[CRSS98]**. No
intrinsics.

---

## 157.2 Constructing Quantum Codes

A quantum code of length n over GF(q) is defined by a symplectic self-orthogonal stabilizer
code, supplied either as a length-n additive code over GF(q²) (compact format) or a length-2n
additive code over GF(q) (extended format). With compact generator matrix `G₁` and extended
generator matrix `G₂ = (A|B)`, the relation is `G₁ = A + λB`, where λ is
`QuantumBasisElement(GF(q²))`. The compact format is the default; the extended format is
selected with the `ExtendedFormat` parameter.

### 157.2.1 Construction of General Quantum Codes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuantumCode(S)` | Given an additive code `S` self-orthogonal with respect to the symplectic inner product, return the quantum code defined by `S`. Parameter `ExtendedFormat` (default `false`): when `false`, `S` is a length-n additive code over GF(q²) (compact); when `true`, `S` is a length-2n additive code over GF(q) (extended). | Stabilizer-code construction **[CRSS98]**. |
| `QuantumCode(M)` | Given a matrix `M` over GF(q²) whose GF(q)-additive row span is a symplectic self-orthogonal code `S`, return the quantum code defined by `S`. Parameter `ExtendedFormat` (default `false`): compact (length-n rows over GF(q²)) vs extended (length-2n rows over GF(q)). | Builds `S` from the additive span of the rows of `M`, then **[CRSS98]**. |
| `QuantumCode(G)` | Given a graph `G`, return the self-dual (dimension 0) quantum code defined by the adjacency matrix of `G`. | Graph-adjacency self-dual construction. |
| `RandomQuantumCode(F, n, k)` | For `F` a degree-2 extension of GF(q) and positive integers `n ≥ k`, return a random `[[n, k]]` quantum stabilizer code over `F`. `F` is assumed in compact format. | Random symplectic self-orthogonal additive code. |
| `Subcode(Q, k)` | Given a quantum code `Q` of dimension `k_Q ≥ k`, return a subcode of `Q` of dimension `k`. | — |

*Worked examples: H157E1 (even GF(4)-linear code is symplectic self-orthogonal → `[[6, 2, 1]]` code); H157E2 (same code in extended format via `ExtendedFormat := true`); H157E3 (rate-1/2 self-orthogonal code is self-dual; `[[6, 0, 4]]` extremal self-dual code); H157E4 (a randomly generated additive — neither linear nor even — symplectic self-orthogonal code gives `[[7, 4, 1]]`); H157E5 (`QuantumCode(M)` directly from a stabiliser matrix); H157E8 (`RandomQuantumCode(F, 10, 6)`).*

### 157.2.2 Construction of Special Quantum Codes

| Intrinsic | Description |
|-----------|-------------|
| `Hexacode()` | Return the `[[6, 0, 4]]` self-dual quantum hexacode. |
| `Dodecacode()` | Return the `[[12, 0, 6]]` self-dual quantum dodecacode. |

*Worked examples: H157E6 (hexacode from a 5-spoked-wheel graph via `QuantumCode(G)`); H157E7 (dodecacode from a graph with construction derived from Danielsen **[Dan05]**).*

### 157.2.3 CSS Codes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CSSCode(C1, C2)` / `CalderbankShorSteaneCode(C1, C2)` | Given two classical linear binary codes `C1` and `C2` of length n with `C2` a subcode of `C1`, form a quantum code using the Calderbank–Shor–Steane construction. | CSS construction **[CS96, Ste96a, Ste96b]**. |

*Worked example: H157E9 (the `[7,4,3]` Hamming code `C1` and its dual `C2` give a `[[7, 1, 3]]` CSS code).*

### 157.2.4 Cyclic Quantum Codes

Cyclic quantum codes are those having cyclic stabilizer codes; conditions on generating
polynomials giving symplectic self-orthogonal stabilizer codes are listed in **[CRSS98]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuantumCyclicCode(v)` / `QuantumCyclicCode(Q)` | Given a single vector `v` or a sequence of vectors `Q` over a finite field `F`, return the quantum code generated by the span of the cyclic shifts of the supplied vectors. The span must be symplectic self-orthogonal. Parameter `LinearSpan` (default `false`): additive span over the prime field, or, if `true`, the linear span. | Cyclic-shift span, symplectic self-orthogonal **[CRSS98]**. |
| `QuantumCyclicCode(n, f)` / `QuantumCyclicCode(n, Q)` | For positive integer `n` and a single polynomial `f` or a sequence `Q` of polynomials over a finite field `F`, return the length-n quantum code generated by the additive span of their cyclic shifts (must be symplectic self-orthogonal). Parameter `LinearSpan` (default `false`): additive span over prime field vs linear span if `true`. | Cyclic code from generating polynomials **[CRSS98]**. |
| `QuantumCyclicCode(v4, v2)` | For GF(2)-additive codes over GF(4): given `v4` over GF(4) and `v2` over GF(2), both of length n, return the length-n code generated by the additive span of their cyclic shifts. The span must be symplectic self-orthogonal. | Two-generator additive cyclic construction. |

*Worked examples: H157E10 (`[[15, 0, 6]]`, the best known binary self-dual code of length 15, from a single vector); H157E11 (`[[23, 12, 4]]` and `[[25, 0, 8]]` from generating polynomials); H157E12 (`[[21, 0, 8]]` and `[[21, 5, 6]]` from a GF(4) vector plus a GF(2) vector).*

### 157.2.5 Quasi-Cyclic Quantum Codes

Quasi-cyclic quantum codes are those having quasi-cyclic stabilizer codes.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `QuantumQuasiCyclicCode(n, Q)` | Given an integer `n` and a sequence `Q` of polynomials over a finite field `F`, let `S` be the quasi-cyclic classical code generated by the span of the vectors formed by concatenating cyclic blocks generated by the polynomials in `Q`. If `S` is symplectic self-orthogonal, return the quasi-cyclic quantum code with stabiliser code `S` (else an error). Parameter `LinearSpan` (default `false`): additive span over the prime field, or linear span if `true`. | Quasi-cyclic stabiliser construction. |
| `QuantumQuasiCyclicCode(Q)` | Given a sequence `Q` of vectors, return the quantum code whose additive stabiliser matrix is constructed from the length-n cyclic blocks generated by the cyclic shifts of the vectors in `Q`. Parameter `LinearSpan` (default `false`): linear span if `true`, additive span otherwise. | Quasi-cyclic stabiliser construction. |

*Worked example: H157E13 (`[[14, 0, 6]]` and `[[18, 6, 5]]`, best known quasi-cyclic binary codes).*

---

## 157.3 Access Functions

| Intrinsic | Description |
|-----------|-------------|
| `QuantumBasisElement(F)` | Given a degree-2 extension field `F = GF(q²)`, return the element λ ∈ F connecting the extended and compact formats: a vector `(a|b)` in extended format corresponds to `w = a + λb` in compact format. |
| `StabilizerCode(Q)` / `StabiliserCode(Q)` | The additive stabiliser code `S` defining the quantum code `Q`. Parameter `ExtendedFormat` (default `false`): compact length-n code over GF(q²), or extended length-2n code over GF(q) if `true`. |
| `StabilizerMatrix(Q)` / `StabiliserMatrix(Q)` | The additive stabiliser matrix `M` defining `Q`. Parameter `ExtendedFormat` (default `false`): compact vs extended as above. |
| `NormalizerCode(Q)` / `NormaliserCode(Q)` | The additive normalizer code `N` defining `Q`. Parameter `ExtendedFormat` (default `false`): compact vs extended. |
| `NormalizerMatrix(Q)` / `NormaliserMatrix(Q)` | The additive normalizer matrix `M` defining `Q`. Parameter `ExtendedFormat` (default `false`): compact vs extended. |

### 157.3.1 Quantum Error Group

For a p-ary N-qubit system (p prime) the error group is the extra-special group of order
`p^(2N+1)`: combinations of N bit-flip errors, N phase-flip errors and an overall phase shift.
All groups in this section use a polycyclic group representation.

| Intrinsic | Description |
|-----------|-------------|
| `QuantumErrorGroup(p, n)` | Return the abelian group representing all possible errors for a length-n p-ary qubit system: an extra-special group of order `p^(2n+1)` with `2n + 1` generators corresponding to the qubit-flip operators `X(i)`, phase-flip operators `Z(i)`, and overall phase multiplication `W` by the p-th root of unity. Generators appear in the order `X(1), Z(1), …, X(n), Z(n), W`. |
| `QuantumBinaryErrorGroup(n)` | Return the abelian group representing all possible errors on a length-n binary qubit system: an extra-special group of order `2^(2n-1)`. |
| `QuantumErrorGroup(Q)` | For a quantum code `Q` of length n, return the group of all errors on n qubits — the full error group / ambient space of all possible errors. |
| `StabilizerGroup(Q)` / `StabiliserGroup(Q)` | Return the abelian group of errors defining `Q`, a subgroup of `QuantumErrorGroup(Q)`. |
| `StabilizerGroup(Q, G)` / `StabiliserGroup(Q, G)` | Given `Q` with error group `G` (an extra-special group), return the abelian group of errors of `Q` as a subgroup of `G`. |

*Worked examples: H157E14 (mapping symplectic vectors to error-group elements; orthogonality ⇔ commutativity); H157E15 (stabilizer group of a GF(4) stabilizer code is abelian); H157E16 (intersecting stabilizer groups of two codes by building them as subgroups of a common error group).*

---

## 157.4 Inner Products and Duals

The functions in this section use the symplectic inner product defined for quantum codes.

| Intrinsic | Description |
|-----------|-------------|
| `SymplecticInnerProduct(v1, v2)` | For `v1, v2` in `K^(n)` (`K` a finite field), return their symplectic inner product. In extended format `(a\|b) * (c\|d) = ad − bc`; for binary codes whose compact format is over GF(4), it is `Trace(v1 · v̄2)`. Parameter `ExtendedFormat` (default `false`). |
| `SymplecticDual(C)` | The dual of the additive (or linear) code `C` with respect to the symplectic inner product. Parameter `ExtendedFormat` (default `false`): compact length-n code over GF(q²) vs extended length-2n code over GF(q). |
| `IsSymplecticSelfDual(C)` | Return `true` if `C` equals its symplectic dual. Parameter `ExtendedFormat` (default `false`). |
| `IsSymplecticSelfOrthogonal(C)` | Return `true` if `C` is contained in its symplectic dual. Parameter `ExtendedFormat` (default `false`). |

*Worked examples: H157E17 (symplectically orthogonal vectors build a self-orthogonal additive code → `[[5, 3]]` quantum code); H157E18 (every vector over GF(4) is symplectically orthogonal to itself).*

---

## 157.5 Weight Distribution and Minimum Weight

The weight distribution of a quantum code `Q` comprises three separate distributions: that of
the stabilizer code `S`, that of the symplectic dual `S⊥`, and that of the words in `S⊥ \ S`
(not a linear space). The weights of undetectable errors are the weights of `S⊥ \ S`. A code is
*pure* iff its minimum weight is ≤ the weight of its stabilizer code.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WeightDistribution(Q)` | Given a quantum code `Q` with stabiliser code `S`, return its weight distribution: three separate values — the weight distributions of `S`, `S⊥` and `S⊥ \ S`. Each distribution is a sequence of `<weight, number-of-words>` tuples. | Classical weight-distribution computation on `S`, `S⊥`. |
| `MinimumWeight(Q)` | For `Q` with stabiliser code `S`, return the minimum weight of `Q` (the minimum weight of `S⊥ \ S`; for self-dual / dimension-0 codes, the minimum weight of `S`). Parameters: `Method` (default `"Auto"`; set to `"Distribution"` to compute via the full weight distribution instead), `RankLowerBound` (default 0), `MaximumTime` (default ∞). | Default uses the classical minimum-weight algorithm of section 152.8.1; alternatively the full-weight-distribution method via `Method := "Distribution"`. |
| `IsPure(Q)` | Return `true` if `Q` is a pure quantum code, i.e. its minimum weight is ≤ the minimum weight of its stabilizer code. | Comparison of minimum weights. |

*Worked examples: H157E19 (the three weight distributions of `QECC(GF(4),6,3)`); H157E20 (verbose minimum-weight computation on a `[[40, 30]]` quasi-cyclic code); H157E21 (purity of best-known length-15 codes via `IsPure`).*

---

## 157.6 New Codes From Old

| Intrinsic | Description |
|-----------|-------------|
| `DirectSum(Q1, Q2)` | Given `[[n₁, k₁, d₁]]` and `[[n₂, k₂, d₂]]` quantum codes, return the `[[n₁+n₂, k₁+k₂, min{d₁,d₂}]]` quantum code which is their direct product. |
| `ExtendCode(Q)` | Given an `[[n, k, d]]` quantum code `Q`, return the extended `[[n+1, k, d]]` quantum code. |
| `ExtendCode(Q, m)` | Perform `m` extensions on the `[[n, k, d]]` quantum code `Q`, returning the extended `[[n+m, k, d]]` quantum code. |
| `PunctureCode(Q, i)` | Given a `[[n, k, d]]` quantum code `Q` and a coordinate position `i`, return the `[[n−1, k, d' >= d−1]]` quantum code produced by puncturing at position `i`. |
| `PunctureCode(Q, I)` | Given `[[n, k, d]]` quantum code `Q` and a set `I` of coordinate positions of size `s`, return the `[[n−s, k, d' >= d−s]]` quantum code produced by puncturing at the positions in `I`. |
| `ShortenCode(Q, i)` | Given `[[n, k, d]]` quantum code `Q` and a coordinate position `i`, return the `[[n−1, k' >= k−1, d' >= d]]` quantum code produced by shortening at position `i`. May fail to produce a valid (symplectic self-orthogonal) quantum code, in which case an error is given. |
| `ShortenCode(Q, I)` | Given `[[n, k, d]]` quantum code `Q` and a set `I` of coordinate positions of size `s`, return the `[[n−s, k' >= k−s, d' >= d]]` quantum code produced by shortening at the positions in `I`. May give an error if not valid. |

*Worked example: H157E22 (a `[[28, 8, 6]]` code built from `[[14, 8, 3]]` and `[[14, 0, 6]]` codes via a Plotkin sum, using `PlotkinSum`/`SymplecticDual`/`StabilizerCode` and Theorem 12 of **[CRSS98]**).*

---

## 157.7 Best Known Quantum Codes

An `[[n, k]]` quantum stabilizer code is a *best known* `[[n, k]]` quantum code (BKQC) if it has
the highest minimum weight among all known `[[n, k]]` codes. The acronym QECC (Quantum Error
Correcting Code) distinguishes these from the best known linear codes database (BKLC). Magma's
database is for binary quantum codes (alphabet GF(4), not GF(2)); the GF(4) database contains
constructions of all best known quantum codes of length up to 35, including self-dual
(dimension 0) codes up to length 35. Codes of length up to 12 are optimal (665 codes). The
database uses tables and constructions compiled by Markus Grassl **[Gra]** with results from
**[CRSS98]**, plus contributions by Eric Rains and Zlatko Varbanov. The verbose flag
`BestCode` displays the construction steps.

| Intrinsic | Description |
|-----------|-------------|
| `QECC(F, n, k)` / `BKQC(F, n, k)` / `BestKnownQuantumCode(F, n, k)` | Given a finite field `F` and positive integers `n`, `k` with `k ≤ n`, return an `[[n, k]]` quantum code over `F` with the largest minimum weight among all known `[[n, k]]` codes. A second boolean return signals whether the desired code exists in the database. The database currently exists for GF(4) (binary quantum codes) up to length 35. |

*Worked examples: H157E23 (weight distribution of a `[[25, 16, 3]]` best known code, verifying impurity); H157E24 (self-dual `[[8, 0, 4]]` code via `QECC(GF(4),8,0)`); H157E25 (`BestCode` verbose construction of a `[[25, 11, 4]]` code from a quasi-cyclic, shortened, then extended code).*

---

## 157.8 Best Known Bounds

A database of best known upper and lower bounds on the maximal possible minimum weights of
quantum codes. The lower bounds match the minimum weights of the best known quantum codes
database; the upper bounds are not currently known with much accuracy.

| Intrinsic | Description |
|-----------|-------------|
| `QECCLowerBound(F, n, k)` | Return the best known lower bound on the maximal minimum distance of `[[n, k]]` quantum codes over `F`. Available for binary quantum codes (`F = GF(4)`) up to length 35. |
| `QECCUpperBound(F, n, k)` | Return the best known upper bound on the minimum distance of `[[n, k]]` quantum codes over `F`. Available for binary quantum codes (`F = GF(4)`) up to length 35. |

*Worked example: H157E26 (comparing lower and upper bounds for `[[20, 10]]` (optimal) and `[[25, 13]]` (gap between bounds)).*

---

## 157.9 Automorphism Group

Automorphisms of a quantum code generalise those of its additive stabiliser code: they consist
of a permutation action on the columns combined with a monomial action on the individual column
values. The automorphism group of a length-n additive stabiliser code over **F₄** is a subgroup
of `Z₃ ≀ Sym(n)` of order `3·n!`; the automorphism group of the quantum code it generates is a
subgroup of `Sym(3) ≀ Sym(n)` of order `3!·n!` (more general action on the column values). In
Magma automorphisms are returned as permutations: length-3n permutations for the full monomial
action, or length-n permutations when restricted to the permutation action on the columns.

| Intrinsic | Description |
|-----------|-------------|
| `AutomorphismGroup(Q)` | The automorphism group of the quantum code `Q`. Currently only applies to binary quantum codes. |
| `PermutationGroup(Q)` | The subgroup of the automorphism group of `Q` consisting of those automorphisms which permute the coordinates of codewords. Currently only applies to binary quantum codes. |

*Worked examples: H157E27 (full automorphism group of the dodecacode, order 648, and its coordinate-permutation subgroup); H157E28 (the hexacode's automorphism group, order 2160, is larger than that of its stabilizer code, order 180).*

---

## 157.10 Hilbert Spaces

A basic (first-release) package for creating and computing with quantum Hilbert spaces. A
Hilbert space may be *densely* or *sparsely* represented; dense gives faster computation,
sparse uses less memory. Currently supports basic unitary transformations and manipulations of
quantum states; future versions will add more complex unitary transformations, measurements,
and encoding of states via quantum error correcting codes.

| Intrinsic | Description |
|-----------|-------------|
| `HilbertSpace(F, n)` | Given a complex field `F` and positive integer `n`, return the quantum Hilbert space on `n` qubits over `F`. Parameter `IsDense` (BoolElt): `true`/`false` force dense/sparse representation; if unset Magma decides automatically. |
| `Field(H)` | The complex field over which the coefficients of states of `H` are defined. |
| `NumberOfQubits(H)` / `Nqubits(H)` | The number of qubits comprising the space `H`. |
| `Dimension(H)` | The dimension of `H`, equal to `2ⁿ` where `n` is the number of qubits. |
| `IsDenselyRepresented(H)` | Return `true` if `H` uses a dense representation. |
| `H1 eq H2` / `H1 ne H2` | Return `true` if the Hilbert spaces are equal (resp. not equal). |

*Worked example: H157E29 (a 5-qubit space defaults to dense; a sparse copy is not considered equal).*

### 157.10.1 Creation of Quantum States

| Intrinsic | Description |
|-----------|-------------|
| `QuantumState(H, v)` | Given a Hilbert space `H` and coefficients `v` (a dense or sparse vector) of length equal to `Dimension(H)`, return the quantum state in `H` defined by `v`. |
| `H ! i` | Return the `i`-th quantum basis state of `H` — the basis state whose qubits give the binary representation of `i`. |
| `H ! s` | Given a sequence `s` of binary values of length equal to the number of qubits of `H`, return the corresponding quantum basis state. |
| `SetPrintKetsInteger(b)` | Boolean `b` controls a global variable for printing quantum states. `false` (default): basis kets printed as binary sequences (e.g. \|1010⟩). `true`: basis kets printed using integer values (e.g. \|5⟩). |

*Worked examples: H157E30 (state from a coefficient vector); H157E31 (states from combinations of basis states, integer or binary; effect of `SetPrintKetsInteger`).*

### 157.10.2 Manipulation of Quantum States

| Intrinsic | Description |
|-----------|-------------|
| `a * e` | Given a complex scalar `a`, multiply the coefficients of the quantum state `e` by `a`. |
| `-e` | Negate all coefficients of the quantum state `e`. |
| `e1 + e2` / `e1 - e2` | Addition and subtraction of the quantum states `e1` and `e2`. |
| `Normalisation(e)` / `Normalisation(~e)` / `Normalization(e)` / `Normalization(~e)` | Normalize the coefficients of `e`, giving an equivalent state whose normalization coefficient is 1. Available either as a procedure (`~e`) or a function. |
| `NormalisationCoefficient(e)` / `NormalizationCoefficient(e)` | Return the normalisation coefficient of the quantum state `e`. |
| `e1 eq e2` / `e1 ne e2` | Return `true` iff the states `e1`, `e2` are equal (resp. not equal). States are still considered equal if they have different normalizations (they occupy the same ray). |

*Worked example: H157E32 (a state and its normalisation are equal because they lie on the same ray).*

### 157.10.3 Inner Product and Probabilities of Quantum States

| Intrinsic | Description |
|-----------|-------------|
| `InnerProduct(e1, e2)` | Return the inner product of the quantum states `e1` and `e2`. |
| `ProbabilityDistribution(e)` | Return the probability distribution of the quantum state as a vector over the reals. |
| `Probability(e, i)` | Return the probability of basis state `i` being returned as the result of a measurement on `e`. |
| `Probability(e, v)` | Given a binary vector `v` of length equal to the number of qubits in `e`, return the probability of the basis state corresponding to `v` being returned from a measurement on `e`. |
| `PrintProbabilityDistribution(e)` | Print the probability distribution of the quantum state. |
| `PrintSortedProbabilityDistribution(e)` | Print the probability distribution in sorted order, most probable states first. Parameter `Max` (RngIntElt, default ∞): maximum number of basis states to print. Parameter `MinProbability` (RngIntElt, default 0): an integer 1–100 giving the minimum probability (percent) for a basis state to be printed. |

*Worked examples: H157E33 (`ProbabilityDistribution`, `Probability`, `PrintProbabilityDistribution` on a 3-qubit state); H157E34 (`PrintSortedProbabilityDistribution` with `Max` and `MinProbability`).*

### 157.10.4 Unitary Transformations on Quantum States

A small selection of unitary transformations; each is available as a function (`e`) and an
in-place procedure (`~e`).

| Intrinsic | Description |
|-----------|-------------|
| `BitFlip(e, k)` / `BitFlip(~e, k)` | Flip the value of the `k`-th qubit of the quantum state `e`. |
| `BitFlip(e, B)` / `BitFlip(~e, B)` | Given a set `B` of positive integers, flip the value of the qubits of `e` indexed by the entries in `B`. |
| `PhaseFlip(e, k)` / `PhaseFlip(~e, k)` | Flip the phase on the `k`-th qubit of `e`. |
| `PhaseFlip(e, B)` / `PhaseFlip(~e, B)` | Given a set `B` of positive integers, flip the phase on the qubits of `e` indexed by the entries in `B`. |
| `ControlledNot(e, B, k)` / `ControlledNot(~e, B, k)` | Flip the `k`-th bit of `e` if all bits contained in `B` are set to 1. |
| `HadamardTrasformation(e)` / `HadamardTrasformation(~e)` | Perform a Hadamard transformation on `e`, which must be densely represented. *(Spelling "Trasformation" as printed in the handbook.)* |

*Worked example: H157E35 (sequence of `PhaseFlip`, `ControlledNot`, `BitFlip` operations on a 4-qubit state).*

---

## 157.11 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[CRSS98]** | A. Robert Calderbank, Eric M. Rains, P. W. Shor, Neil J. A. Sloane. *Quantum error correction via codes over GF(4).* IEEE Trans. Inform. Theory **44**(4):1369–1387, 1998. |
| **[CS96]** | A. R. Calderbank and P. W. Shor. *Good quantum error-correcting codes exist.* Phys. Rev. A **54**:2551–2577, 1996. |
| **[Dan05]** | D. E. Danielsen. *On self-dual quantum codes, graphs, and Boolean functions.* Master's thesis, University of Bergen, 2005. |
| **[Gra]** | Markus Grassl. *Bounds on the minimum distance of quantum codes.* URL: http://iaks-www.ira.uka.de/home/grassl/QECC/. |
| **[Sho94]** | Peter W. Shor. *Algorithms for quantum computation: discrete logarithms and factoring.* In *35th Annual Symposium on Foundations of Computer Science (Santa Fe, NM, 1994)*, pages 124–134. IEEE Comput. Soc. Press, Los Alamitos, CA, 1994. |
| **[Sho95]** | P. W. Shor. *Scheme for reducing decoherence in quantum computer memory.* Phys. Rev. A **52**:2493–2496, 1995. |
| **[Ste96a]** | A. M. Steane. *Error correcting codes in quantum theory.* Phys. Rev. Lett. **77**(5):793–797, 1996. |
| **[Ste96b]** | Andrew Steane. *Multiple-particle interference and quantum error correction.* Proc. Roy. Soc. London Ser. A **452**(1954):2551–2577, 1996. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Stabilizer code ↔ additive code over GF(4) **[CRSS98]** | `QuantumCode`, `RandomQuantumCode`, `StabilizerCode`, `QuantumBasisElement` |
| Calderbank–Shor–Steane (CSS) construction **[CS96, Ste96a, Ste96b]** | `CSSCode` / `CalderbankShorSteaneCode` |
| Cyclic / quasi-cyclic stabiliser constructions **[CRSS98]** | `QuantumCyclicCode`, `QuantumQuasiCyclicCode` |
| Graph-adjacency self-dual construction **[Dan05]** | `QuantumCode(G)`, `Dodecacode`, `Hexacode` |
| Symplectic inner product / duality | `SymplecticInnerProduct`, `SymplecticDual`, `IsSymplecticSelfDual`, `IsSymplecticSelfOrthogonal` |
| Extra-special quantum error group (polycyclic) | `QuantumErrorGroup`, `QuantumBinaryErrorGroup`, `StabilizerGroup` |
| Classical minimum-weight / weight-distribution algorithms (§152.8.1) | `MinimumWeight`, `WeightDistribution`, `IsPure` |
| New-codes-from-old / Plotkin sum **[CRSS98]** | `DirectSum`, `ExtendCode`, `PunctureCode`, `ShortenCode` |
| Best known codes / bounds database **[Gra, CRSS98]** | `QECC` / `BKQC` / `BestKnownQuantumCode`, `QECCLowerBound`, `QECCUpperBound` |
| Quantum code automorphisms (monomial + permutation action) | `AutomorphismGroup`, `PermutationGroup` |
| Quantum Hilbert-space simulation | `HilbertSpace`, `QuantumState`, unitary transforms (`BitFlip`, `PhaseFlip`, `ControlledNot`, `HadamardTrasformation`) |
