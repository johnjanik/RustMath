# Chapter 121 — Elliptic Curves over Finite Fields

**Handbook part:** XVI — Arithmetic Geometry
**Handbook pages:** 3979–3999 (PDF pages 4110–4133)

---

## Scope and overview

This chapter describes the specialised facilities for elliptic curves defined over finite
fields. Construction, arithmetic, and basic properties are handled in Chapter 120; this
chapter focuses on the algorithmically richer computations that arise from the finite-field
setting, particularly in the context of Elliptic Curve Cryptography (ECC).

**Point counting.** The dominant tool is an efficient implementation of the
Schoof–Elkies–Atkin (SEA) algorithm, extended by Lercier to characteristic-2 fields, for
determining #E(Fq). For small characteristics (p < 1000) faster p-adic canonical lift
methods or Dwork's p-adic deformation method are used instead. Canonical lift uses
Lercier–Lubicz [LL03] when a Gaussian Normal Basis of type ≤ 2 exists, and the MSST
algorithm [Gau02] / Harley's recursive variant [Har] otherwise. For p < 10 and p = 13,
special modular polynomials arising from genus-0 modular curves X0(p^r) [Koh03] give
further speedups. The deformation method, due to Hubrechts [Hub07], applies where
canonical lift is not used and the extension degree is large enough.

**Pairings.** Magma implements the Weil, Tate (Tate–Lichtenbaum), Eta (T and q
variants, for supersingular curves), and Ate (T and q variants, for general curves)
pairings. All are based on Miller's algorithm evaluating the Miller function f_{n,P}. The
Weil pairing provides the basis for the MOV reduction of the discrete logarithm problem
(DLP) on a supersingular curve to a DLP in a finite field.

**Weil descent.** An implementation by Florian Heß of the GHS Weil descent attack for
ordinary elliptic curves in characteristic 2 [Gau00, Bla05] constructs a higher-genus curve
C/k together with a divisor-class homomorphism from E(K) to Jac(C)(k), transferring the
DLP to a smaller field where index-calculus applies.

**Discrete logarithms.** For a direct attack on the elliptic curve DLP, Magma provides a
parallel collision-search version of Pollard's rho algorithm incorporating Pohlig–Hellman
reduction, Teske's r-adding walks, and the Wiener–Zuccherato negation trick that halves
the effective search space.

---

## 121.1 Supersingular Curves

An elliptic curve E over a finite field is *supersingular* if and only if its trace of
Frobenius is divisible by the characteristic. The four intrinsics below detect or certify
supersingularity.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsSupersingular(E : parameters)` | Returns `false` if E is ordinary, otherwise proves E is supersingular and returns `true`. Parameter `Proof` (default `true`): if set to `false`, the function behaves like `IsProbablySupersingular` (non-deterministic). | Certifying test; non-deterministic path via `IsProbablySupersingular`. |
| `SupersingularPolynomial(p)` | Given a prime p, returns the separable monic polynomial over Fp whose roots are exactly the j-invariants of supersingular elliptic curves in characteristic p. Computed via a partial power-series expansion of a certain hypergeometric function reduced mod p; very fast for p of moderate size. | Hypergeometric partial-fraction/power-series formula; ignores factors for j = 0 and j = 1728. |
| `IsOrdinary(E)` | Returns `true` if E is ordinary (i.e. not supersingular); logical negation of `IsSupersingular`. | — |
| `IsProbablySupersingular(E)` | Returns `false` if E is proved ordinary, otherwise `true`. Non-deterministic; repeated calls are independent. | Probabilistic test. |

---

## 121.2 The Order of the Group of Points

### 121.2.1 Point Counting

Magma contains an efficient implementation of the Schoof–Elkies–Atkin (SEA) algorithm
with Lercier's extension to characteristic-2 base fields. For small characteristic (p < 1000)
faster p-adic canonical lift methods or the p-adic deformation method are preferred.
Calculations are performed in the smallest field over which the curve is defined and then
lifted to the original field. All group functions on elliptic curves actually apply to a
particular point set; the curve is identified with its base point set for these functions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#E` / `Order(E)` | The order of the group of K-rational points of E, where E is defined over the finite field K. | SEA algorithm for p ≥ 1000 (characteristic > 2 uses full SEA; characteristic 2 uses Lercier–SEA). For p < 1000: canonical lift (Lercier–Lubicz **[LL03]** when a GNB of type ≤ 2 exists; MSST **[Gau02]** / Harley recursion **[Har]** otherwise; special genus-0 modular polynomial method **[Koh03]** for p < 10 and p = 13); deformation (Hubrechts **[Hub07]**) for larger extension degree. |
| `FactoredOrder(E)` | Factorisation of the order of the K-rational point group of E. The factorisation is cached on E and reused by point-order computations. | Same as `Order(E)`. |
| `SEA(H : parameters)` / `SEA(E : parameters)` | Internal point-counting routine, exposed for direct use. Returns the group order, or 0 if the early-abort mechanism was triggered. Parameters: `Al` (`"Default"` / `"BSGS"` / `"Enumerate"`; `"BSGS"` currently equivalent to `"Default"`), `MaxSmooth` (upper bound on the smooth-order product that triggers abort; default ∞), `AbortLevel` (0 = abort if curve has smooth order; 1 = abort if both curve and twist have smooth order; 2 = abort if either has smooth order; default 0), `UseSEA` (force SEA even in small characteristic; default `false`). Unlike `Order`, does not cache the result back on E. | Schoof–Elkies–Atkin (full SEA) **[Schoof, Elkies, Atkin]**; Lercier extension for characteristic 2; canonical lift **[LL03, Gau02, Har, Koh03]**; deformation **[Hub07]**. |
| `SetVerbose("SEA", v)` | Sets verbose output level for the SEA point-counting algorithm. Legal values: `false`/0 (silent), `true`/1, or integers 2–5 (increasing verbosity). Replaces the deprecated `"ECPointCount"` flag. | — |
| `Order(E, r)` | Order of E over the degree-r extension field of the base field K. Computed from the order of E over K by lifting. | Order-lifting formula. |
| `Trace(E)` / `TraceOfFrobenius(E)` | Trace of the Frobenius endomorphism: q + 1 − n, where q = #K and n = #E(K). | Derived from `Order(E)`. |
| `Trace(E, r)` / `TraceOfFrobenius(E, r)` | Trace of the r-th power Frobenius endomorphism on E. | Derived from `Order(E, r)` by the recurrence for power-Frobenius traces. |

*Worked examples: H121E1 (characteristic-2 and characteristic-3 point counting illustrating GNB speed differences and the deformation method; early-abort SEA and prime-order search); H121E2 (SEA over GF(2^133), FactoredOrder, TraceOfFrobenius over extension fields); H121E3 (twists and quadratic twists over GF(101^2), trace-negation property).*

### 121.2.2 Zeta Functions

The zeta function of E over Fq is a rational function whose logarithmic derivative has the
power-series expansion sum_{n=1}^∞ |E(F_{q^n})| t^n. Its numerator encodes the trace of
Frobenius and is equivalent data to `Order(E)` and `Trace(E)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ZetaFunction(E)` | Given E over a finite field, returns the zeta function of E as a rational function in one variable. Equivalent to computing `Order(E)` / `Trace(E)`. | SEA (same invocation as `Order`). |

*Worked example: H121E4 (zeta function of E over GF(11), verifying the logarithmic-derivative power-series expansion against `Order(E, n)`).*

### 121.2.3 Cryptographic Elliptic Curve Domains

Functions for finding or validating a cryptographic Elliptic Curve Domain: a curve E over
a finite field together with a point P whose order is a large prime, and such that (E, P)
resists MOV and Anomalous attacks [JM00].

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `CryptographicCurve(F)` | Given a finite field F, searches for a cryptographic EC domain: generates random curves, computes #E, sieves by small primes, checks that the remaining large pseudoprime p satisfies p ≥ max(OrderBound, 4√(#F)), verifies the MOV/Anomalous security conditions for p, optionally proves primality, then finds a random point of order p. Returns E, P, p, #E/p. Parameters: `OrderBound` (default 2^160), `Proof` (default `true`), `UseSEA` (force SEA for point counting; not recommended; default `false`). Requires #F ≥ OrderBound (or 2*OrderBound in characteristic 2). | Random curve search + SEA / p-adic point counting; security checks from **[JM00]**. |
| `ValidateCryptographicCurve(E, P, ordP, h)` | Verifies that (E, P, ordP, h) is a valid cryptographic EC domain with ordP satisfying the security inequality. Parameter `Proof` (default `true`; if `false`, uses a strong pseudoprimality test instead of proving primality of ordP). | Security-condition check per **[JM00]**; primality proof or pseudoprimality test. |
| `SetVerbose("ECDom", v)` | Sets verbose output level for `CryptographicCurve` and `ValidateCryptographicCurve`. Legal values: `false`/0, `true`/1, 2. | — |

*Worked example: H121E5 (finding a cryptographic curve over GF(2^196) with a point of prime order > 2^160, then searching for one with order = 2 × prime; validating the result).*

---

## 121.3 Enumeration of Points

The following intrinsics operate on a point set H (which may be supplied directly or
derived as the base point set of an elliptic curve E).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Points(E)` / `Points(H)` / `RationalPoints(E)` / `RationalPoints(H)` | Returns the set of all rational points of the point set H (or the base point set of E), including the point at infinity. | Enumeration. |
| `Random(E)` / `Random(H)` | Returns a uniformly random rational point of the point set H or the base point set of E, including the point at infinity. | Uniform random selection. |

---

## 121.4 Abelian Group Structure

All group-structure intrinsics apply to a particular point set; the curve is identified with
its base point set for these functions.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AbelianGroup(E)` / `TorsionSubgroup(E)` | Returns an abelian group A isomorphic to the group of rational points of E over a finite field, together with an isomorphism map m : A → E. | Baby-step giant-step / order-factored structure computation. |
| `Generators(H)` / `Generators(E)` | Returns a sequence of generators for the group of rational points of E (or of the point set H). The i-th element corresponds to the i-th generator of the abelian group returned by `AbelianGroup`. | — |
| `NumberOfGenerators(H)` / `NumberOfGenerators(E)` / `Ngens(H)` / `Ngens(E)` | The number of generators of the rational-point group of E (or point set H); equal to the length of the sequence returned by `Generators(E)`. | — |

*Worked example: H121E6 (AbelianGroup and Generators for a curve over GF(1048583^2), verifying generators against the abstract group).*

---

## 121.5 Pairings on Elliptic Curves

Pairings on elliptic curves over finite fields have both destructive (MOV attack) and
constructive (pairing-based cryptography) applications. All pairings evaluate a Miller
function f_{n,P}: any function on E with divisor n(P) − ([n]P) − (n−1)∞.

### 121.5.1 Weil Pairing

The Weil pairing w_n(P, Q) is a non-degenerate bilinear map from E[n] × E[n] to µ_n
(the n-th roots of unity), computed as (−1)^n f_{n,P}(Q) / f_{n,Q}(P).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WeilPairing(P, Q, n)` | Given n-torsion points P and Q on E (in the same point set), returns the Weil pairing w_n(P, Q). | Miller's algorithm: evaluates f_{n,P}(Q) and f_{n,Q}(P), then divides (with sign). |

### 121.5.2 Tate Pairing

The Tate pairing (Tate–Lichtenbaum pairing) t_n(P, Q) = f_{n,P}(Q) is a non-degenerate
bilinear map from E[n] × E/nE into K*/(K*)^n, where K is a finite field containing the
n-th roots of unity. The reduced version applies the final exponentiation (power by
(#K* − 1)/n) to map into µ_n.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `TatePairing(P, Q, n)` | Given n-torsion point P and point Q on E (same point set), returns the Tate pairing t_n(P, Q) as a random representative of the coset in K*/(K*)^n. | Miller's algorithm evaluating f_{n,P}(Q). |
| `ReducedTatePairing(P, Q, n)` | Returns the reduced Tate pairing e_n(P, Q) = t_n(P, Q)^((#K−1)/n), mapping into µ_n(K). Requires n | #K − 1. | Miller's algorithm + final exponentiation. |

### 121.5.3 Eta Pairing

The Eta pairing is only defined on supersingular curves and is an optimised variant of the
Tate pairing. For a supersingular E/Fq with n | #E(Fq), let k be the security multiplier
(smallest positive k with q^k ≡ 1 mod n; k | 6 in the supersingular case). The Eta
pairing is defined on the product of the two Frobenius eigenspaces G1 × G2 of E[n],
where P ∈ G1 iff F(P) = P (i.e. P ∈ E(Fq)[n]) and Q ∈ G2 iff F(Q) = [q]Q.

The Eta pairing e_T(P, Q) = f_{T,P}(Q) with T = t − 1 (where #E(Fq) = q + 1 − t) or T = q.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `EtaTPairing(P, Q, n, q)` | Given a supersingular E/Fq, P ∈ G1, Q ∈ G2 (both in E(F_{q^k})), returns the unreduced Eta pairing with T = t − 1 (a random coset representative). `q` must be the size of the base field. Parameters: `CheckCurve` (default `false`), `CheckPoints` (default `false`). | Miller's algorithm with T = t − 1; no final exponentiation. |
| `ReducedEtaTPairing(P, Q, n, q)` | Reduced version of `EtaTPairing`: applies final exponentiation to map into µ_n. Parameters: `CheckCurve`, `CheckPoints` (both default `false`). | Miller's algorithm with T = t − 1 + final exponentiation. |
| `EtaqPairing(P, Q, n, q)` | Eta pairing with T = q; automatically reduced (maps directly into n-th roots of unity). Parameters: `CheckCurve`, `CheckPoints` (both default `false`). | Miller's algorithm with T = q; inherently reduced. |

### 121.5.4 Ate Pairing

The Ate pairing generalises the Eta pairing to all elliptic curves. It is defined on
G2 × G1 (arguments swapped relative to Eta). The Ate pairing a_T(Q, P) = f_{T,Q}(P)
with Q ∈ G2, P ∈ G1, and T = t − 1 or T = q.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AteTPairing(Q, P, n, q)` | Given Q ∈ G2, P ∈ G1 (F(Q) = [q]Q, F(P) = P, both in E(F_{q^k})), returns the unreduced Ate pairing with T = t − 1 (a random coset representative). `q` is the size of the base field. Parameter: `CheckPoints` (default `false`). | Miller's algorithm with T = t − 1; no final exponentiation. |
| `ReducedAteTPairing(Q, P, n, q)` | Reduced version of `AteTPairing`: applies final exponentiation to map into µ_n. Parameter: `CheckPoints` (default `false`). | Miller's algorithm with T = t − 1 + final exponentiation. |
| `AteqPairing(P, Q, m, q)` | Ate pairing with T = q; automatically reduced (maps directly into n-th roots of unity). Parameter: `CheckPoints` (default `false`). | Miller's algorithm with T = q; inherently reduced. |

*Worked examples: H121E7 (constructing a BN-curve of embedding degree 12 over a ~160-bit prime field; testing Weil and Tate pairing bilinearity; computing Ate pairings after projecting to G2 via the Frobenius trace; then testing Eta pairings on a supersingular curve over GF(2^163) of prime order); H121E8 (MOV reduction: discrete log on a supersingular curve over GF(p), p = NextPrime(2^131), transferred to a finite-field DLP via the Weil pairing over GF(p^2)).*

---

## 121.6 Weil Descent in Characteristic Two

One approach to attacking the elliptic curve DLP over finite fields is Weil descent: given
E/K and a subfield k of K, one seeks a higher-genus curve C/k and a non-trivial
homomorphism E(K) → Jac(C)(k), reducing the DLP to a smaller field where index
calculus applies.

Magma contains an implementation, due to Florian Heß, of the GHS Weil descent for
ordinary (j(E) ≠ 0) elliptic curves in characteristic 2 [Gau00, Bla05]. The descent
constructs C together with the divisor map from E to C using an Artin–Schreier extension
y^2 + y = c/x + a2(E) + bx (where r = sqrt(1/j(E)) and b = r/c). The choice of c
strongly affects the genus and degree of C; c ∈ k or b ∈ k yields hyperelliptic C, though
not necessarily with small genus.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `WeilDescent(E, k, c)` | E is an ordinary elliptic curve over a characteristic-2 finite field K, k is a subfield of K, and c is a non-zero element of K. Constructs a plane curve C/k and a divisor map from E(K) to k-divisors of C via GHS descent. If C is hyperelliptic (and `HyperellipticImage` is `true`, the default), C is returned as type `CrvHyp` and the map is the divisor-class homomorphism to Jac(C)(k). Otherwise C is a general plane curve and the map sends points to effective divisors (no Jacobian reduction). Parameter `HyperellipticImage` (default `true`). | GHS Weil descent **[Gau00, Bla05]** (Heß's implementation); function-field inclusion K(E) → K(C) followed by trace to k; Artin–Schreier extension parameterised by c. |
| `WeilDescentGenus(E, k, c)` | Returns the genus of the Weil descent curve C that `WeilDescent(E, k, c)` would produce. | GHS genus formula; uses the same Artin–Schreier parameter c. |
| `WeilDescentDegree(E, k, c)` | Returns the degree in the second variable of the plane Weil descent curve C that `WeilDescent(E, k, c)` would produce. | GHS degree formula. |

*Worked example: H121E9 (GHS descent from E/GF(2^155) to a hyperelliptic curve C of genus 31 over GF(2^5); mapping a point of E to a Jacobian point and verifying the order).*

---

## 121.7 Discrete Logarithms

Computing discrete logarithms on elliptic curves over finite fields is considered hard for
general curves; the best general algorithms run in exponential time. Magma's approach:

1. Compute the factorisation of the order of the base point Q (invoking SEA if needed).
2. Check that ord(Q) is a multiple of ord(P); if not, return −1.
3. Reduce to prime-power subproblems via the **Pohlig–Hellman algorithm**.
4. For small primes: exhaustive search; may detect non-existence early.
5. For larger prime-power factors: **parallel collision search** (Pollard rho) with
   Teske's r-adding walks [Teske], and the Wiener–Zuccherato negation trick [WZ]
   (treating P and −P as identical on curves y^2 = x^3 + ax + b, cutting the search
   space by sqrt(2)).

If no solution exists Magma may or may not detect it; the computation may run forever on
bad parameters, so a time limit is recommended for large instances.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Log(Q, P)` | Discrete logarithm of P to the base Q: integer n with 0 ≤ n < ord(Q) and n*Q = P. Q and P must be points on the same elliptic curve over a finite field. Returns −1 if it is determined that no solution exists. | Pohlig–Hellman decomposition; parallel Pollard rho with r-adding walks [Teske] and negation map [Wiener–Zuccherato] for large prime factors. |
| `Log(Q, P, t)` | As `Log(Q, P)` but with a time limit of t seconds (must be a small positive integer). Returns −1 if no solution, −2 if the time limit is exceeded before a solution is found. | Same as `Log(Q, P)`. |

*Worked example: H121E10 (random 40-bit prime field, random base point Q, random multiple P = m*Q; recovering m via Log and verifying m*Q − P = 0).*

---

## 121.8 Bibliography

| Key | Reference |
|-----|-----------|
| **[Bla05]** | I. Blake, G. Seroussi and N. Smart, editors. *Advances in Elliptic Curve Cryptography*, volume 317 of LMS LNS. Cambridge University Press, Cambridge, 2005. |
| **[Gau00]** | P. Gaudry, F. Heß and N. P. Smart. Constructive and destructive facets of Weil descent on elliptic curves. *J. Cryptology*, 15(1):19–46, 2000. |
| **[Gau02]** | P. Gaudry. A Comparison and a Combination of SST and AGM Algorithms for Counting Points on Elliptic Curves in Characteristic 2. In Y. Zheng, editor, *Advances in Cryptology — AsiaCrypt 2002*, volume 2501 of LNCS, pages 311–327. Springer-Verlag, 2002. |
| **[Har]** | R. Harley. Web posting. Under November 2002 entry at URL: http://listserv.nodak.edu/archives/nmbrthry.html |
| **[Hub07]** | Hendrik Hubrechts. Quasi-quadratic elliptic curve point counting using rigid cohomology. To appear in the *Journal of Symbolic Computation*, 2007. URL: http://wis.kuleuven.be/algebra/hubrechts/ |
| **[JM00]** | D. Johnson and A. Menezes. The Elliptic Curve Digital Signature Algorithm (ECDSA). Technical report, Univ. Waterloo, 2000. Available at URL: http://www.cacr.math.uwaterloo.ca/ |
| **[Koh03]** | David R. Kohel. The AGM-X0(N) Heegner Point Lifting Algorithm and Elliptic Curve Point Counting. In *Advances in Cryptology — AsiaCrypt 2003*, number 2894 in LNCS, Berlin, 2003. Springer. |
| **[LL03]** | R. Lercier and D. Lubicz. Counting Points on Elliptic Curves over Finite Fields of Small Characteristic in Quasi Quadratic Time. In *Advances in Cryptology — EuroCrypt 2003*, volume 2656 of LNCS, pages 360–373. Springer, 2003. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Schoof–Elkies–Atkin (SEA) point counting | `#E`, `Order(E)`, `FactoredOrder(E)`, `SEA(E)`, `Trace(E)`, `TraceOfFrobenius(E)`, `ZetaFunction(E)`, `Order(E,r)`, `Trace(E,r)` |
| Canonical lift — Lercier–Lubicz (GNB type ≤ 2) **[LL03]** | `#E`, `Order(E)`, `SEA(E)` (small characteristic) |
| Canonical lift — MSST / Harley recursion **[Gau02, Har]** | `#E`, `Order(E)`, `SEA(E)` (small characteristic, no suitable GNB) |
| Genus-0 modular curve lift (p < 10, p = 13) **[Koh03]** | `#E`, `Order(E)`, `SEA(E)` |
| p-adic deformation (Dwork/Lauder/Hubrechts) **[Hub07]** | `#E`, `Order(E)`, `SEA(E)` (small p, large extension degree) |
| Supersingularity detection / supersingular polynomial | `IsSupersingular`, `IsProbablySupersingular`, `IsOrdinary`, `SupersingularPolynomial` |
| MOV/Anomalous security conditions **[JM00]** | `CryptographicCurve`, `ValidateCryptographicCurve` |
| Miller's algorithm — Weil pairing | `WeilPairing` |
| Miller's algorithm — Tate pairing | `TatePairing`, `ReducedTatePairing` |
| Miller's algorithm — Eta pairing (supersingular curves) | `EtaTPairing`, `ReducedEtaTPairing`, `EtaqPairing` |
| Miller's algorithm — Ate pairing (general curves) | `AteTPairing`, `ReducedAteTPairing`, `AteqPairing` |
| GHS Weil descent **[Gau00, Bla05]** (Heß implementation) | `WeilDescent`, `WeilDescentGenus`, `WeilDescentDegree` |
| Pohlig–Hellman + parallel Pollard rho (r-adding walks, negation map) | `Log(Q,P)`, `Log(Q,P,t)` |
| Baby-step giant-step / abelian group structure | `AbelianGroup`, `TorsionSubgroup`, `Generators`, `NumberOfGenerators`, `Ngens` |
