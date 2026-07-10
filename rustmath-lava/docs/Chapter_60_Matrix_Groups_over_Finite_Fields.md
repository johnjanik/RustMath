# Chapter 60 — Matrix Groups over Finite Fields

**Authors:** Henrik Bäärnhielm, Derek Holt, C.R. Leedham-Green, E.A. O'Brien (package); additional code by Peter Brooksbank, Elliot Costi, Heiko Dietrich, Alice Niemeyer, Csaba Schneider
**Handbook part:** IX — Finite Groups
**Handbook pages:** 1711–1758 (PDF pages 1842–1891)

---

## Scope and overview

When a matrix group G is defined over a finite field and is not too large, a base-and-strong-generating-set (BSGS) representation can be built and the standard algorithms of Chapter 59 applied. For groups of moderately small dimension but very large order, however, a BSGS representation is infeasible. Chapter 60 describes techniques that operate without one.

The chapter covers two broad families of methods:

(a) **Aschbacher-reduction functions** — algorithms rooted in Aschbacher's classification theorem **[Asc84]** for maximal subgroups of the general linear group. Each Aschbacher category corresponds to a structural decomposition (imprimitive, semilinear, tensor product, tensor-induced, extraspecial-normaliser, subfield, almost-simple), and the chapter provides functions to test membership in each category and to make the decomposition explicit.

(b) **Monte Carlo and Las Vegas algorithms** — probabilistic algorithms for centralizers of involutions, normal closures, derived groups, and perfectness testing. These may return subgroups of the intended object; correctness is not guaranteed unless the Verify flag is used.

The **CompositionTree** package (§60.6) unifies both families into a recursive data structure that decomposes G via a tree of homomorphisms, with leaves that are cyclic, elementary abelian, or (nearly) simple groups. The **LMG** (large matrix group) functions (§60.7) wrap the CompositionTree package in a user-friendly interface that automatically decides whether to use BSGS or composition-tree methods based on orbit-length estimates.

A separate section covers **constructive recognition** of quasisimple classical groups (§60.5): finding standard generators, rewriting arbitrary elements as straight-line programs (SLPs) in those generators, and verifying the isomorphism via standard presentations. Finally, §60.8 covers **unipotent matrix groups**, for which power-conjugate presentations enable fast order computation and membership testing.

For surveys of the area, see **[O'B06, O'B11]**.

---

## 60.1 Introduction

*(No intrinsics. Motivating discussion; see Scope and overview above.)*

---

## 60.2 Finding Elements with Prescribed Properties

Random-search routines for elements satisfying order or involution conditions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `RandomElementOfOrder(G, n : parameters)` | Given a finite matrix group `G`, attempts to find an element `x` of order `n` by random search. Returns `true`, the element `x`, and an SLP for `x`; or `false`. Parameters: `Central` (BoolElt, default `false`) — if true, seek `x` of order `n` modulo the centre; `Proof` (BoolElt, default `true`) — if false, `x` may have order a multiple of `n`; `MaxTries` (RngIntElt, default 100) — maximum random elements tried; `Randomiser` (GrpRandProc) — random process (default `RandomProcessWithWords(G)`). The last return indicates whether the precise order is proven. | Random search with order checking. |
| `RandomElementOfNormalClosure(G, N)` | Given a group `G` and a subgroup `N`, returns a random element of the normal closure of `N` in `G`. Works for permutation or matrix groups. | Algorithm of Leedham-Green and O'Brien **[LGO02]**. |
| `InvolutionClassicalGroupEven(G : parameters)` | Let `G` be a quasisimple classical group in its natural representation in even characteristic. Returns an involution `I` of corank in `[d/4, …, d/2]`, its SLP in `WordGroup(G)`, and its corank. For types `Ω+` or `Ω−` the degree must be at least 4 over a field with at least 4 elements. Parameters: `SmallCorank` (BoolElt, default `false`) — accept involutions of small corank; `Case` (MonStgElt, default `"unknown"`) — one of `"SL"`, `"Sp"`, `"SU"`, `"Omega-"`, `"Omega+"`. Implemented by Heiko Dietrich. | Algorithm of Dietrich, Lübeck, Leedham-Green, O'Brien **[DLLGO13]**. |

---

## 60.3 Monte Carlo Algorithms for Subgroups

Probabilistic algorithms for centralizers, normal closures, derived groups, and perfectness. All are Monte Carlo and may return proper subgroups of the intended object.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CentraliserOfInvolution(G, g : parameters)` | Given an involution `g` in `G`, returns the centraliser `C` of `g` in `G` (Monte Carlo). Parameters: `Central` (BoolElt, default `false`) — construct projective centraliser (commute modulo centre); `NumberRandom` (RngIntElt, default 100) — max random elements; `CompletionCheck` (UserProgram) — user function `(G, C, g) -> Bool` to determine completion (default: stop at 20 generators or `NumberRandom` elements). | Bray's involution centraliser algorithm **[Bra00]**. |
| `CentraliserOfInvolution(G, g, w : parameters)` | As above, but also takes an SLP `w` for `g` and returns SLPs for the generators of `C`. Parameters add `Randomiser` (GrpRandProc, default `RandomProcessWithWords(G)`); `CompletionCheck` takes four arguments `(G, C, g, SLPs)`. The SLP `w` must lie in the word group of `Randomiser`. | Bray **[Bra00]**. |
| `AreInvolutionsConjugate(G, x, wx, y, wy : parameters)` | Monte Carlo algorithm to find `c ∈ G` conjugating involution `x` to involution `y`. `wx`, `wy` are SLPs for `x`, `y`. Returns `true`, `c`, and SLP for `c` if found; otherwise `false`. Parameters: `Randomiser` (GrpRandProc); `MaxTries` (RngIntElt, default 100). SLPs must lie in the word group of `Randomiser`. | Monte Carlo conjugacy search. |
| `NormalClosureMonteCarlo(G, H)` / `NormalClosureMonteCarlo(G, H : parameters)` | Constructs the normal closure `N` of `H` in `G` (Monte Carlo). If SLPs for generators of `H` are supplied via parameter `slpsH` (default `[]`), also returns SLPs for generators of `N`. Parameters: `ErrorProb` (FltRatElt, default 9/10) — upper bound on probability that `N` is a proper subgroup of the true normal closure; `SubgroupChainLength` (RngIntElt, default `Degree(H)`) — upper bound on subgroup chain length in `H`. | Monte Carlo normal closure construction. |
| `DerivedGroupMonteCarlo(G : parameters)` | Given a matrix group `G` over a finite field, returns the derived group of `G` and SLPs of its generators in the generators of `G` (Monte Carlo). Parameters: `Randomiser` (GrpRandProc, default `RandomProcessWithWords(G)`); `NumberGenerators` (RngIntElt, default 10) — minimum generators; `MaxGenerators` (RngIntElt, default 100) — maximum generators. | Monte Carlo derived group construction. |
| `IsProbablyPerfect(G : parameters)` | Attempts to prove that a matrix or permutation group `G` is perfect by verifying that its generators lie in `G'`. Monte Carlo — if returns `true`, `G` is perfect; if `false`, `G` might still be perfect. Parameter: `NumberRandom` (RngIntElt, default 100). Uses `NormalSubgroupRandomElement`. | Leedham-Green and O'Brien **[LGO02]**. |

*Worked examples: H60E1 (IsProbablyPerfect on a subgroup of GU(4,9) and on SO(7,5) / Ω(7,5)).*

---

## 60.4 Aschbacher Reduction

### 60.4.1 Introduction

Aschbacher's theorem **[Asc84]** classifies the maximal subgroups of GL(d, K) (K a finite field): every matrix group G acting on the natural module V over K satisfies at least one of:

1. G acts reducibly on V;
2. G acts semilinearly over an extension field of K;
3. G acts imprimitively on V;
4. G preserves a nontrivial tensor-product decomposition of V;
5. G has a normal subgroup N acting absolutely irreducibly on V that is an extraspecial p-group or 2-group of symplectic type;
6. G preserves a tensor-induced decomposition of V;
7. G acts (modulo scalars) linearly over a proper subfield of K;
8. G contains a classical group in its natural action over K;
9. G is almost simple modulo scalars.

The CompositionTree package includes functions for all categories. Category (i) is handled by Meataxe (R-modules chapter). The subsections below cover categories (ii)–(vii) and general decomposition search.

### 60.4.2 Primitivity

G acts imprimitively on V if there is a nontrivial direct-sum decomposition V = V₁ ⊕ ··· ⊕ Vᵣ with the Vᵢ permuted by G. Verbose flag: `SetVerbose("Smash", 1)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsPrimitive(G : parameters)` | Given a matrix group `G` over a finite field, returns `true` if primitive, `false` if not, or `"unknown"`. Parameter: `BlockSizes` ([RngIntElt], default `[]`) — restrict search to block sizes in this sequence. | Holt, Leedham-Green, O'Brien, Rees **[HLGOR96b]**. |
| `ImprimitiveBasis(G)` | For imprimitive `G`, returns the change-of-basis matrix exhibiting the block structure. | — |
| `Blocks(G)` | For imprimitive `G`, returns the blocks of imprimitivity. | — |
| `BlocksImage(G)` | For imprimitive `G`, returns the group induced by the action of `G` on the system of imprimitivity. | — |
| `ImprimitiveAction(G, g)` | For imprimitive `G` and element `g ∈ G`, returns the action of `g` on the blocks as a permutation. | — |

*Worked examples: H60E2 (wreath product GL(4,7) wr S₃; Blocks, BlocksImage, ImprimitiveAction).*

### 60.4.3 Semilinearity

G ≤ GL(d, q) is semilinear if a normal subgroup N embeds in GL(d/e, qᵉ) for some e > 1, with a centralising matrix C acting as multiplication by a field generator of Fqᵉ, and each generator of G corresponds to a field automorphism λ → λⁱ. Verbose flag: `SetVerbose("SemiLinear", 1)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsSemiLinear(G)` | Returns `true` if `G` is semilinear, `false` if not, or `"unknown"`. | Holt, Leedham-Green, O'Brien, Rees **[HLGOR96a]**. |
| `DegreeOfFieldExtension(G)` | For semilinear `G ≤ GL(d, q)`, returns the degree `e` of the extension field. | — |
| `CentralisingMatrix(G)` | For semilinear `G`, returns the matrix `C` that centralises the normal subgroup acting linearly over the extension field. | — |
| `FrobeniusAutomorphisms(G)` | For semilinear `G` with centralising matrix `C`, returns a sequence `S` where `S[i]` is the least positive integer such that `gᵢ⁻¹ C gᵢ = C^{S[i]}`. | — |
| `WriteOverLargerField(G)` | For semilinear `G ≤ GL(d, q)` with extension degree `e`, returns: (i) the normal subgroup `N` (kernel of `G → Cₑ`, acting linearly over the extension field, equal to the centraliser of `C` in `G`); (ii) a cyclic group `E` of order `e` isomorphic to `G/N`; (iii) a sequence of images of the generators of `G` in `E`. | — |

*Worked examples: H60E3 (semilinear subgroup of GL(6,3): IsSemiLinear, DegreeOfFieldExtension, CentralisingMatrix, FrobeniusAutomorphisms, WriteOverLargerField).*

### 60.4.4 Tensor Products

G ≤ GL(d, K) preserves a tensor decomposition of V as U ⊗ W if V ≅ U ⊗ W with the induced image of G lying in GL(U) ∘ GL(W). Verbose flag: `SetVerbose("Tensor", 1)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsTensor(G : parameters)` | Returns `true` if `G` preserves a nontrivial tensor decomposition, `false` if not, or `"unknown"`. Parameter: `Factors` ([SeqEnum], default `[]`) — sequence of `[u, w]` pairs restricting search to decompositions `U ⊗ W` with dim U = u, dim W = w. | Leedham-Green and O'Brien **[LGO97b, LGO97a]**. |
| `TensorBasis(G)` | For tensor-decomposable `G`, returns the change-of-basis matrix exhibiting the tensor decomposition. | — |
| `TensorFactors(G)` | For tensor-decomposable `G`, returns the two matrix groups that are the tensor factors of `G`. | — |
| `IsProportional(X, k)` | For tensor-decomposable `G`, returns `true` iff matrix `X` consists of k×k blocks differing only by scalars; if so, also returns the tensor decomposition of `X`. | — |

*Worked examples: H60E4 (subgroup of GL(6,3) with tensor decomposition; TensorBasis, IsProportional, TensorFactors). H60E5 (tensor-induced example using IsProportional at block sizes 2 and 4).*

### 60.4.5 Tensor-induced Groups

G ≤ GL(d, K), d = uʳ, is tensor-induced if it preserves a decomposition V = U₁ ⊗ U₂ ⊗ ··· ⊗ Uᵣ with each Uᵢ of dimension u > 1, r > 1, and the set {Uᵢ} permuted by G; this gives a homomorphism G → Sᵣ. Verbose flag: `SetVerbose("TensorInduced", 1)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsTensorInduced(G : parameters)` | Returns `true` if `G` is tensor-induced, `false` if not, or `"unknown"`. Parameter: `InducedDegree` (RngIntElt, default `"All"`) — if set to `r`, restricts search to homomorphisms into Sᵣ only. | Leedham-Green and O'Brien **[LGO02]**. |
| `TensorInducedBasis(G)` | For tensor-induced `G`, returns the change-of-basis matrix exhibiting that `G` is tensor-induced. | — |
| `TensorInducedPermutations(G)` | For tensor-induced `G`, returns a sequence whose i-th entry is the homomorphic image of `G.i` in Sᵣ. | — |
| `TensorInducedAction(G, g)` | For tensor-induced `G` and element `g ∈ G`, returns the tensor-induced action of `g`. | — |

*Worked examples: H60E5 (TensorWreathProduct of GL(2,3) with S₃; IsTensorInduced, TensorInducedPermutations, TensorInducedBasis, IsProportional).*

### 60.4.6 Normalisers of Extraspecial r-groups and Symplectic 2-groups

Let G ≤ GL(d, q) with d = rᵐ for a prime r. If G normalises an r-group R of order r^{2m+1} or 2^{2m+2}, then R is either extraspecial (first case) or a 2-group of symplectic type (central product of an extraspecial 2-group with Z/4). For d = r an odd prime, a Monte Carlo algorithm **[Nie05]** is used; otherwise `IsExtraSpecialNormaliser` may return `"unknown"`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsExtraSpecialNormaliser(G)` | Returns `true` if `G` normalises an extraspecial r-group or 2-group of symplectic type, `false` if known not to, or `"unknown"`. | Monte Carlo algorithm of Niemeyer **[Nie05]** for odd-prime degree; direct search otherwise. |
| `ExtraSpecialParameters(G)` | For `G` known to normalise such a group, returns `[r, n]` where the normalised subgroup `R` has order `rⁿ`. | — |
| `ExtraSpecialGroup(G)` | For `G` known to normalise such a group, returns the extraspecial or symplectic subgroup `R` normalised by `G`. | — |
| `ExtraSpecialNormaliser(G)` | For `G` known to normalise such a group, returns the action of each generator of `G` on `R` as a sequence of matrices of degree 2r. | — |
| `ExtraSpecialAction(G, g)` | For `G` known to normalise such a group, returns a matrix of degree 2r describing the action of element `g` on `R`. | — |
| `ExtraSpecialBasis(G)` | For the odd-prime-degree case, returns the change-of-basis matrix that conjugates the normal extraspecial subgroup into a "nice" representation (generated by a diagonal and a permutation matrix). | — |

*Worked examples: H60E6 (subgroup of GL(7,8) normalising an extraspecial r-group; IsExtraSpecialNormaliser, ExtraSpecialParameters, ExtraSpecialNormaliser, ExtraSpecialAction).*

### 60.4.7 Writing Representations over Subfields

Tests whether G (or G modulo scalars) can be realised over a proper subfield of its defining field, and provides change-of-basis matrices and images.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsOverSmallerField(G : parameters)` | For absolutely irreducible `G` over finite field `K`, decides if `G` has an equivalent representation over a subfield of `K`. Returns `true` and the representation over the smallest such subfield, or `false`. Parameters: `Scalars` (BoolElt, default `false`) — test `G` modulo scalars; `Algorithm` (MonStgElt, default `"GLO"`) — `"GLO"` for Glasby–Leedham-Green–O'Brien **[GLGO05]** or `"GH"` for Glasby–Howlett **[GH97]** (non-scalar case only). | Glasby, Leedham-Green, O'Brien **[GLGO05]**; or Glasby–Howlett **[GH97]**. |
| `IsOverSmallerField(G, k : parameters)` | As above, but tests for a specific subfield of degree `k` over the prime field (k a proper divisor of deg K). Same parameters. | Glasby, Leedham-Green, O'Brien **[GLGO05]**; or Glasby–Howlett **[GH97]**. |
| `SmallerField(G)` | For `G` representable over a proper subfield (possibly modulo scalars), returns that subfield. | — |
| `SmallerFieldBasis(G)` | For `G` representable over a proper subfield (possibly modulo scalars), returns the change-of-basis matrix. | — |
| `SmallerFieldImage(G, g)` | For `G` representable over a proper subfield, returns the image of `g ∈ G` in the group over the subfield. | — |
| `WriteOverSmallerField(G, F)` | Given a group `G` of d×d matrices over finite field `E` of degree `e` and a subfield `F` of degree `f`, writes `G` as (de/f)×(de/f) matrices over `F` and returns the group and the isomorphism. | — |

*Worked examples: H60E7 (GL(2, GF(3,2)) embedded in GL(2, GF(3,8)); IsOverSmallerField, Scalars:=true, SmallerField, SmallerFieldBasis, SmallerFieldImage). H60E8 (GL(2,4) rewritten over GF(2) as degree-4 group; WriteOverSmallerField).*

### 60.4.8 Decompositions with Respect to a Normal Subgroup

#### 60.4.8.1 Accessing the Decomposition Information

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SearchForDecomposition(G, S)` | Given a matrix group `G` over a finite field and a sequence `S` of elements of `G`, first constructs the normal closure `N` of `S` in `G`, then tests whether `G` (with respect to `N`) has a decomposition of Aschbacher type (ii)–(vi): semilinear over extension field (N linear over extension); imprimitive (N fixes blocks); tensor product (N scalar on one factor); extraspecial p-group or symplectic-type normal subgroup; or tensor-induced (N acts absolutely irreducibly, fixing factors). Reports type found and returns `true` if any decomposition is found, `false` otherwise. Answer is conclusive for types (ii)–(v); negative for (vi) may not be conclusive. Verbose flag: `SetVerbose("Smash", 1)`. | Holt, Leedham-Green, O'Brien, Rees **[HLGOR96a]**. |

*Worked examples: H60E9 (five examples: GL(4,5) — no decomposition; wreath product — imprimitive; semilinear subgroup of GL(6,3); tensor product GL(5,5)⊗GL(3,5); tensor-induced TensorWreathProduct(GL(3,GF(2)),S₃); normaliser of symplectic 2-group in GL(4,5)).*

The access functions for each decomposition type are those described in §60.4.2–§60.4.6.

---

## 60.5 Constructive Recognition for Simple Groups

For each finite non-abelian simple group S, a standard copy is designated with standard generators. Constructive recognition constructs an effective isomorphism φ: G → S by finding standard generators in G. A rewriting algorithm then expresses any g ∈ G as an SLP in those generators. Correctness is verified by evaluating a standard presentation for S on the discovered generators. For background see **[O'B11, LGO09]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ClassicalStandardGenerators(type, d, q)` | Produces the standard generators of Leedham-Green and O'Brien for the quasisimple classical group of specified `type` in dimension `d` over the field of size `q`. Type must be one of `"SL"`, `"Sp"`, `"SU"`, `"Omega"`, `"Omega-"`, `"Omega+"`. Standard generators defined in **[LGO09]** and **[DLLGO13]**. | Leedham-Green–O'Brien standard generators **[LGO09, DLLGO13]**. |
| `ClassicalConstructiveRecognition(G : parameters)` | `G` must be a matrix group over a finite field conjugate to a quasisimple classical group in its natural representation (dimension ≥ 2). Constructs standard generators `S` in `G` and SLPs for them in the defining generators `X`. Returns `true`, `S`, and SLPs if `G` is quasisimple classical; otherwise `false`. Parameters: `Case` (MonStgElt, default `"unknown"`) — supply the result of `ClassicalType(G)`; `Randomiser` (GrpRandProc, default `RandomProcessWithWords(G)`). Even characteristic implementation by Heiko Dietrich; odd characteristic by Eamonn O'Brien. | Constructive recognition; even char **[DLLGO13]**, odd char **[LGO09]**. |
| `ClassicalChangeOfBasis(G)` | `G` must be a classical group in its natural representation on which `ClassicalConstructiveRecognition` has been applied. Returns a change-of-basis matrix `CB` conjugating the generators returned by `ClassicalStandardGenerators` to those returned by `ClassicalConstructiveRecognition(G)`. | — |
| `ClassicalRewrite(G, gens, type, dim, q, g : parameters)` | `G` is a classical group of specified `type` with dimension `dim` over `Fq`, generated by `gens` satisfying `ClassicalStandardPresentation(type, dim, q)`. Given `g ∈ Generic(G)`: if `g ∈ G` returns `true` and an SLP; if `g ∉ G` attempts to find SLP `w` with `g · Evaluate(w, gens)⁻¹` centralising `G` and returns `false, w`; otherwise `false, false`. Method selection: (i) natural representation with `ClassicalStandardGenerators` → Costi's implementation **[Cos09]**; (ii) absolutely irreducible in defining characteristic → Schneider's implementation based on Costi **[Cos09]**; (iii) otherwise → black-box method of Schneider **[AMPS10]** (not yet for orthogonal groups). Parameter: `Method` (MonStgElt) — override with `"CharP"` or `"BB"`. Code prepared by Csaba Schneider. | Costi **[Cos09]** (natural/defining char); black-box **[AMPS10]**. |
| `ClassicalRewriteNatural(type, CB, g)` | Faster specialised version of `ClassicalRewrite` for classical groups in their natural representation. `type` is one of `"SL"`, `"Sp"`, `"SU"`, `"Omega"`, `"Omega-"`, `"Omega+"`. `CB` and `g` are elements of GL(d, q). If `g` is a member of the group generated by `ClassicalStandardGenerators(type, d, q)^CB`, returns `true` and SLP `w` such that `Evaluate(w, ClassicalStandardGenerators(type, d, q)^CB) = g`; otherwise `false, false`. Developed by Elliot Costi; prepared by Csaba Schneider. | Costi **[Cos09]**. |
| `ClassicalStandardPresentation(type, d, q : parameters)` | Constructs a presentation on the standard generators for the quasisimple group of type `type` (one of `"SL"`, `"Sp"`, `"SU"`, `"Omega"`, `"Omega-"`, `"Omega+"`), dimension `d`, field of size `q`. Returns relations as SLPs and the parent SLPGroup. Parameter: `Projective` (BoolElt, default `false`) — if true, gives a presentation for the corresponding projective group. Presentations described in **[LGO]**. | Leedham-Green–O'Brien standard presentations **[LGO]**. |

*Worked examples: H60E10 (standard generators for SL(6,5³); ClassicalConstructiveRecognition; ClassicalChangeOfBasis; ClassicalRewriteNatural for SL, Sp, Ω−; ClassicalRewrite with Method:="BB"; ClassicalStandardPresentation verification).*

---

## 60.6 Composition Trees for Matrix Groups

A composition tree for G is a recursive data structure presenting G in terms of its composition factors. Construction alternates between:

(i) **Reduction**: find an effective homomorphism φ: G → G₁ (G₁ "smaller" in degree or field), then recurse on G₁ and Ker(φ); and
(ii) **Leaf**: decide G is cyclic, elementary abelian, or (nearly) simple.

When G ≤ GL(d, q), Step (i) exploits Aschbacher's theorem **[Asc84]**; other homomorphisms (e.g. the determinant map) are also used. Leaves may be cyclic, elementary abelian, soluble or non-abelian simple primitive permutation groups, or absolutely irreducible matrix groups simple modulo centre.

Once the tree is built, a list of "nice generators" Y is stored; ⟨Y⟩ is the "nice group". Rewriting operates on Y; the results can be pulled back to SLPs in the original generators X. Verbose flag: `SetVerbose("CompositionTree", n)` for n = 1, …, 10.

The package was prepared by Bäärnhielm, Holt, Leedham-Green, and O'Brien. See **[LG01, O'B06, O'B11, NS06, BHLGO11]**.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CompositionTree(G : parameters)` | Given a matrix group `G` over a finite field, constructs and returns a composition tree for `G`. Parameters: `Verify` (BoolElt, default `false`) — verify correctness during construction; `KernelBatchSize` (RngIntElt, default 5) — normal generators per kernel step; `MandarinBatchSize` (RngIntElt, default 100) — random elements for Monte Carlo checks; `MaxHomFinderFails` (RngIntElt, default 1) — give up after this many Monte Carlo failures; `MaxQuotientOrder` (RngIntElt, default 10⁶) — leaves larger than this are not refined to composition factors; `FastTensorTest` (BoolElt, default `true`) — use only the fast tensor test; `MaxBSGSVerifyOrder` (RngIntElt, default 2000) — verify RandomSchreier calculations below this order; `AnalysePermGroups` (BoolElt, default `false`) — if false, treat permutation group as a leaf; `KnownLeaf` (BoolElt, default `false`); `NamingElements` (RngIntElt, default 200) — random elements for LieType/RecogniseClassical; `UnipotentBatchSize` (RngIntElt, default 100); `PresentationKernel` (BoolElt, default `true`) — use presentations to obtain kernels where possible. | Constructive Aschbacher reduction **[Asc84]** + constructive recognition **[LG01, O'B06, O'B11]**; new ideas from **[NS06]**; full description in **[BHLGO11]**. |
| `CompositionTreeFastVerification(G)` | For `G` with a composition tree, determines if correctness can be verified cheaply using presentations (i.e., presentations on nice generators are known for all leaves). Returns boolean. | — |
| `CompositionTreeVerify(G)` | For `G` with a composition tree, verifies correctness by constructing a presentation for `G` and checking it. Returns `true` and the relators (as SLPs) if the presentation is satisfied, otherwise `false`. Presentation is on `CompositionTreeNiceGroup(G)`. | Presentation-based verification. |
| `CompositionTreeNiceGroup(G)` | Returns the nice group for `G` (must have a composition tree). | — |
| `CompositionTreeSLPGroup(G)` | For `G` with a composition tree and associated nice group `H`, returns the word group `W` for `H` and the map `W → H`. | — |
| `DisplayCompTreeNodes(G : parameters)` | Displays information about nodes in the composition tree for `G`, traversing in-order. Parameters: `NonTrivial` (BoolElt, default `true`) — show only non-trivial nodes; `Leaves` (BoolElt, default `false`) — show only leaves. | — |
| `CompositionTreeNiceToUser(G)` | Returns the coercion map from SLPs in nice generators of `G` to SLPs in the user-supplied generators of `G`, plus the SLPs of the nice generators in terms of user generators. | — |
| `CompositionTreeOrder(G)` | Returns the order of `G` (must have a composition tree). | Derived from composition tree. |
| `CompositionTreeElementToWord(G, g)` | For `G` with a composition tree and element `g`, returns `true` and an SLP for `g` in the nice generators of `G`, or `false` if `g ∉ G`. | Rewriting via composition tree. |
| `CompositionTreeCBM(G)` | Returns a change-of-basis matrix exhibiting the Aschbacher reductions of `G` given by the composition tree. | — |
| `CompositionTreeReductionInfo(G, t)` | Returns a string description of the reduction at internal node `t` in the composition tree for `G`, plus the image and kernel of that reduction. | — |
| `CompositionTreeSeries(G)` | Returns: (1) a normal series 1 = G₀ < G₁ < ··· < Gₖ = G; (2) maps Gᵢ → Sᵢ (standard copy of Gᵢ/Gᵢ₋₁, possibly plus scalars Z); (3) maps Sᵢ → Gᵢ; (4) maps Sᵢ → WordGroup(Sᵢ); (5) boolean flag for true composition series; (6) sequence of leaf nodes for each factor. All maps are defined by rules (use `Function` to avoid built-in membership testing). | — |
| `CompositionTreeFactorNumber(G, g)` | Returns the minimal integer `i` such that `g` lies in the i-th term of the normal series returned by `CompositionTreeSeries`. | — |
| `HasCompositionTree(G)` | Returns `true` if `G` has a composition tree, `false` otherwise. | — |
| `CleanCompositionTree(G)` | Removes all composition tree data structures for `G`. | — |

*Worked examples: H60E11 (CGOPlus(4,5²): CompositionTree, DisplayCompTreeNodes, CompositionTreeFastVerification, CompositionTreeVerify, CompositionTreeOrder, CompositionTreeElementToWord, CompositionTreeNiceToUser, CompositionTreeSeries, CompositionTreeFactorNumber). H60E12 (maximal subgroup of SL(10,2⁸): CompositionTree, DisplayCompTreeNodes, CompositionTreeFastVerification, CompositionTreeVerify, CompositionTreeElementToWord, CompositionTreeNiceToUser, CompositionTreeSeries).*

---

## 60.7 The LMG Functions

The LMG (large matrix group) functions provide a user-friendly interface to the CompositionTree package for structural calculations in matrix groups too large for BSGS methods. On the first call to any LMG function on G, Magma tests whether all basic orbit lengths are at most `LMGSchreierBound` (default 40000): if so, BSGS is used; otherwise, `CompositionTree(G)` is called. The user can force a method via `LMGInitialise`. By default these functions have a small probability of failing or returning incorrect results; calling `LMGInitialise(G : Verify := true)` before other calls requests verified results. Verbose flag: `SetVerbose("LMG", n)` for n = 1, 2, 3.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetLMGSchreierBound(n)` | Sets the global constant `LMGSchreierBound` to `n`. | — |
| `LMGInitialize(G : parameters)` / `LMGInitialise(G : parameters)` | Initialises `G` for LMG computations. Parameters: `Al` (MonStgElt, default `""`) — force `"CompositionTree"` (or `"CT"`) or `"RandomSchreier"` (or `"RS"`); `Verify` (BoolElt, default `false`) — attempt to verify the BSGS or composition tree; `RandomSchreierBound` (RngIntElt, default `LMGSchreierBound`). Further calls after the first have no effect. | Selects BSGS or CompositionTree. |
| `LMGOrder(G)` | Returns the order of `G`. | Via BSGS or composition tree. |
| `LMGFactoredOrder(G)` | Returns the factored order of `G`. | Via BSGS or composition tree. |
| `LMGIsIn(G, x)` | Returns `true` if `x ∈ G`, `false` otherwise. | — |
| `LMGIsSubgroup(G, H)` | Returns `true` if `H ≤ G`, `false` otherwise. | — |
| `LMGEqual(G, H)` | For `G`, `H` in a common overgroup GL(n, q), returns `true` if `G = H`. | — |
| `LMGIndex(G, H)` | Returns the index `[G : H]`. | — |
| `LMGIsNormal(G, H)` | Returns `true` if `H` is normal in `G`. | — |
| `LMGNormalClosure(G, H)` | Returns the normal closure of `H` in `G`. | — |
| `LMGDerivedGroup(G)` | Returns the derived subgroup of `G`. | — |
| `LMGCommutatorSubgroup(G, H)` | Returns the commutator subgroup of `G` and `H` (both subgroups of GL(n, q)). | — |
| `LMGIsSoluble(G)` / `LMGIsSolvable(G)` | Returns `true` if `G` is soluble. | — |
| `LMGIsNilpotent(G)` | Returns `true` if `G` is nilpotent. | — |
| `LMGCompositionSeries(G)` | Returns a composition series for `G`. | — |
| `LMGCompositionFactors(G)` | Returns the composition factors of `G` (same format as `CompositionFactors` for finite groups). | — |
| `LMGChiefSeries(G)` | Returns a chief series for `G`. | — |
| `LMGChiefFactors(G)` | Returns the chief factors of `G` (same format as `ChiefFactors` for finite groups). | — |
| `LMGUnipotentRadical(G)` | Returns the unipotent radical `U` of `G`, a PC group `P`, and an isomorphism `U → P`. | — |
| `LMGSolubleRadical(G)` / `LMGSolvableRadical(G)` | Returns the soluble radical `S` of `G`, a PC group `P`, and an isomorphism `S → P`. | — |
| `LMGFittingSubgroup(G)` | Returns the Fitting subgroup `S` of `G`, a PC group `P`, and an isomorphism `S → P`. | — |
| `LMGCentre(G)` / `LMGCenter(G)` | Returns the centre of `G`. | — |
| `LMGSylow(G, p)` | Returns a Sylow p-subgroup of `G`. | — |
| `LMGSocleStar(G)` | Returns the inverse image in `G` of the socle of `G/S`, where `S` is the soluble radical of `G`. | — |
| `LMGSocleStarFactors(G)` | Returns the simple direct factors of `LMGSocleStar(G)/LMGSolubleRadical(G)` (possibly represented projectively for large classical groups), plus a list of maps from the factors to `G`. | — |
| `LMGSocleStarAction(G)` | Returns the map φ representing the conjugation action of `G` on the simple direct factors of `LMGSocleStar(G)/LMGSolubleRadical(G)`, plus the image and kernel of φ. | — |
| `LMGSocleStarActionKernel(G)` | Returns: the kernel of the conjugation action of `G` on the simple direct factors of `LMGSocleStar(G)/LMGSolubleRadical(G)`; a PC group `P` isomorphic to `LMGSocleStarActionKernel(G)/LMGSocleStar(G)`; and the epimorphism `G → P`. | — |
| `LMGSocleStarQuotient(G)` | Returns the quotient group `G/LMGSocleStar(G)` as a permutation group, with associated epimorphism and kernel. | — |
| `LMGRadicalQuotient(G)` | Returns a permutation group `P ≅ G/L` (where `L` is the soluble radical), an epimorphism `G → P`, and `L`. Required (implicitly) as a first step for the remaining functions. | — |
| `LMGCentraliser(G, g)` / `LMGCentralizer(G, g)` | Returns the centraliser of `g ∈ G` in `G`. Requires `G/L` to have a permutation representation of manageable degree. | Via radical quotient + lifting. |
| `LMGIsConjugate(G, g, h)` | Returns `true` if elements `g`, `h ∈ G` are conjugate; if so, also returns a conjugating element. | Via radical quotient + lifting. |
| `LMGClasses(G)` / `LMGConjugacyClasses(G)` | Returns the conjugacy classes of `G`. | Via radical quotient + lifting. |
| `LMGNormaliser(G, H)` / `LMGNormalizer(G, H)` | Returns the normaliser of subgroup `H` in `G`. | Via radical quotient + lifting. |
| `LMGIsConjugate(G, H, K)` | Returns `true` if subgroups `H`, `K ≤ G` are conjugate in `G`; if so, also returns a conjugating element. | Via radical quotient + lifting. |
| `LMGMaximalSubgroups(G)` | Returns the maximal subgroups of `G`. | Via radical quotient + lifting. |

*Worked examples: H60E13 (maximal subgroup C[4] of SL(12,5): LMGFactoredOrder, LMGChiefFactors, LMGDerivedGroup, LMGIndex, LMGEqual, LMGSolubleRadical, LMGIsSoluble, LMGIsNilpotent, LMGCentre, LMGFittingSubgroup, LMGSylow, LMGNormalClosure, LMGSocleStarFactors, LMGChiefFactors on factors and on their normal closures).*

---

## 60.8 Unipotent Matrix Groups

A power-conjugate (PC) presentation is a highly efficient representation for unipotent groups (see Chapter 63). The algorithm used is a straightforward echelonisation-like procedure.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `UnipotentMatrixGroup(G)` | Given a matrix group `G` over a finite field, constructs a known unipotent matrix group from `G`. Note: Magma does not verify that `G` is unipotent. | — |
| `WordMap(G)` | Given a unipotent matrix group `G`, constructs and returns the word map: a map from `G` to the SLP group on `n = Ngens(G)` generators. See Chapter 76 for SLP groups. | Echelonisation-based. |
| `PCPresentation(G)` | Given a unipotent matrix group `G`, constructs a PC-presentation. Returns a finite soluble group `H`, a map `G → H`, and a map `H → G`. | Echelonisation-like procedure. |
| `Order(G)` / `#G` | Returns the order of the unipotent matrix group `G` as an integer. Faster than the standard matrix group order intrinsic due to use of the PC-presentation. | Via PC-presentation. |
| `FactoredOrder(G)` | Returns the factored order of the unipotent matrix group `G`. Faster than standard due to PC-presentation. | Via PC-presentation. |
| `g in G` | Given a matrix `g` and a unipotent matrix group `G`, returns `true` if `g ∈ G`. Faster than standard membership testing due to PC-presentation. | Via PC-presentation. |

*Worked examples: H60E14 (unipotent subgroup of GL(4,5): UnipotentMatrixGroup, WordMap, membership). H60E15 (Sylow 7-subgroup of GL(9,7): PCPresentation, FactoredOrder).*

---

## 60.9 Bibliography

| Key | Reference |
|-----|-----------|
| **[AMPS10]** | S. Ambrose, S. H. Murray, C. E. Praeger, and C. Schneider. Constructive membership testing in black-box classical groups. In *Proceedings of The Third International Congress on Mathematical Software*, number 6327 in Lecture Notes in Computer Science, pages 54–57, Basel, 2010. Springer. |
| **[Asc84]** | M. Aschbacher. On the maximal subgroups of the finite classical groups. *Invent. Math.*, 76:469–514, 1984. |
| **[BHLGO11]** | H. Bäärnhielm, Derek Holt, C.R. Leedham-Green, and E.A. O'Brien. A practical model for computation with matrix groups. Preprint, 2011. |
| **[Bra00]** | J.N. Bray. An improved method of finding the centralizer of an involution. *Arch. Math. (Basel)*, 74(1):241–245, 2000. |
| **[Cos09]** | E. Costi. Constructive membership testing in classical groups. PhD thesis, Queen Mary, University of London, 2009. |
| **[DLLGO13]** | Heiko Dietrich, Frank Lübeck, C.R. Leedham-Green, and E.A. O'Brien. Constructive recognition of classical groups in even characteristic. *J. Algebra*, 2013. |
| **[GH97]** | S.P. Glasby and R.B. Howlett. Writing representations over minimal fields. *Comm. Algebra*, 25(6):1703–1711, 1997. |
| **[GLGO05]** | S.P. Glasby, C.R. Leedham-Green, and E.A. O'Brien. Writing projective representations over subfields. *J. Algebra*, 295:51–61, 2005. |
| **[HLGOR96a]** | Derek F. Holt, C.R. Leedham-Green, E.A. O'Brien, and Sarah Rees. Computing decompositions for modules with respect to a normal subgroup. *J. Algebra*, 184:818–838, 1996. |
| **[HLGOR96b]** | Derek F. Holt, C.R. Leedham-Green, E.A. O'Brien, and Sarah Rees. Testing matrix groups for primitivity. *J. Algebra*, 184:795–817, 1996. |
| **[LG01]** | Charles R. Leedham-Green. The computational matrix group project. In *Groups and computation, III (Columbus, OH, 1999)*, volume 8 of Ohio State Univ. Math. Res. Inst. Publ., pages 229–247. de Gruyter, Berlin, 2001. |
| **[LGO]** | C.R. Leedham-Green and E.A. O'Brien. Short presentations for classical groups. Preprint. |
| **[LGO97a]** | C.R. Leedham-Green and E.A. O'Brien. Recognising tensor products of matrix groups. *Internat. J. Algebra Comput.*, 7:541–559, 1997. |
| **[LGO97b]** | C.R. Leedham-Green and E.A. O'Brien. Tensor Products are Projective Geometries. *J. Algebra*, 189:514–528, 1997. |
| **[LGO02]** | C.R. Leedham-Green and E.A. O'Brien. Recognising tensor-induced matrix groups. *J. Algebra*, 253:14–30, 2002. |
| **[LGO09]** | C.R. Leedham-Green and E.A. O'Brien. Constructive recognition of classical groups in odd characteristic. *J. Algebra*, 322:833–881, 2009. |
| **[Nie05]** | Alice C. Niemeyer. Constructive recognition of normalisers of small extraspecial matrix groups. *Internat. J. Algebra Comput.*, 15:367–394, 2005. |
| **[NS06]** | Max Neunhöffer and Ákos Seress. A data structure for a uniform approach to computations with finite groups. In *ISSAC 2006*, pages 254–261. ACM, New York, 2006. |
| **[O'B06]** | E.A. O'Brien. Towards effective algorithms for linear groups. In *Finite Geometries, Groups and Computation*, pages 163–190. De Gruyter, 2006. |
| **[O'B11]** | E.A. O'Brien. Algorithms for matrix groups. In *Groups St Andrews (Bath)*, volume 388 of LMS Lecture Notes, pages 297–323. Cambridge University Press, 2011. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Aschbacher classification **[Asc84]** | `IsPrimitive`, `IsSemiLinear`, `IsTensor`, `IsTensorInduced`, `IsExtraSpecialNormaliser`, `IsOverSmallerField`, `SearchForDecomposition`, `CompositionTree` |
| Primitivity testing **[HLGOR96b]** | `IsPrimitive`, `ImprimitiveBasis`, `Blocks`, `BlocksImage`, `ImprimitiveAction` |
| Semilinearity **[HLGOR96a]** | `IsSemiLinear`, `DegreeOfFieldExtension`, `CentralisingMatrix`, `FrobeniusAutomorphisms`, `WriteOverLargerField` |
| Tensor product recognition **[LGO97b, LGO97a]** | `IsTensor`, `TensorBasis`, `TensorFactors`, `IsProportional` |
| Tensor induction recognition **[LGO02]** | `IsTensorInduced`, `TensorInducedBasis`, `TensorInducedPermutations`, `TensorInducedAction` |
| Extraspecial normaliser recognition **[Nie05]** | `IsExtraSpecialNormaliser`, `ExtraSpecialParameters`, `ExtraSpecialGroup`, `ExtraSpecialNormaliser`, `ExtraSpecialAction`, `ExtraSpecialBasis` |
| Subfield writing — Glasby–Leedham-Green–O'Brien **[GLGO05]** | `IsOverSmallerField`, `SmallerField`, `SmallerFieldBasis`, `SmallerFieldImage` |
| Subfield writing — Glasby–Howlett **[GH97]** | `IsOverSmallerField(:Algorithm:="GH")` |
| Decomposition w.r.t. normal subgroup **[HLGOR96a]** | `SearchForDecomposition` |
| Bray involution centraliser **[Bra00]** | `CentraliserOfInvolution` |
| Leedham-Green–O'Brien random methods **[LGO02]** | `RandomElementOfNormalClosure`, `IsProbablyPerfect` |
| Even-characteristic involution **[DLLGO13]** | `InvolutionClassicalGroupEven` |
| Constructive recognition (odd char) **[LGO09]** | `ClassicalConstructiveRecognition`, `ClassicalStandardGenerators`, `ClassicalChangeOfBasis`, `ClassicalRewriteNatural` |
| Constructive recognition (even char) **[DLLGO13]** | `ClassicalConstructiveRecognition` |
| Standard presentations **[LGO]** | `ClassicalStandardPresentation` |
| Classical rewriting (natural/char-p) **[Cos09]** | `ClassicalRewrite`, `ClassicalRewriteNatural` |
| Classical rewriting (black-box) **[AMPS10]** | `ClassicalRewrite(:Method:="BB")` |
| Composition tree **[LG01, O'B06, O'B11, NS06, BHLGO11]** | `CompositionTree`, `CompositionTreeVerify`, `CompositionTreeFastVerification`, `CompositionTreeNiceGroup`, `CompositionTreeSLPGroup`, `CompositionTreeOrder`, `CompositionTreeElementToWord`, `CompositionTreeCBM`, `CompositionTreeReductionInfo`, `CompositionTreeSeries`, `CompositionTreeFactorNumber`, `DisplayCompTreeNodes`, `CompositionTreeNiceToUser`, `HasCompositionTree`, `CleanCompositionTree` |
| LMG interface (BSGS or composition tree) | All `LMG*` functions, `SetLMGSchreierBound` |
| Unipotent PC-presentation | `UnipotentMatrixGroup`, `WordMap`, `PCPresentation`, `Order`/`#`, `FactoredOrder`, `in` (for GrpMatUnip) |
