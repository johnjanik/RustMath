# Chapter 32 — Quadratic Forms

**Handbook part:** V — Lattices and Quadratic Forms
**Handbook pages:** 745–750 (PDF pages 876–883)

---

## Scope and overview

Chapter 32 describes miscellaneous functionality for fairly general quadratic forms over the
rationals or the integers. Quadratic forms in Magma are represented either as multivariate
polynomials (homogeneous of degree 2) or as symmetric matrices; the chapter provides
conversions between these representations and from lattice objects.

The main computational feature is an implementation of **Simon's algorithm** for finding
isotropic subspaces of integral quadratic forms. The algorithm locates a maximal (or
near-maximal) totally isotropic subspace, and uses the Bosma–Stevenhagen algorithm for the
2-part of class groups of quadratic fields as a subroutine.

The chapter also provides the standard **local invariants** that characterise a rational
quadratic form up to rational equivalence: p-signatures, p-excesses, oddity, and
Witt (Hasse–Minkowski) invariants at all places. Definitions follow Conway–Sloane
**[JC98]**, Chapter 15.

---

## 32.2 Constructions and Conversions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SymmetricMatrix(f)` | Given a multivariate polynomial `f` that is homogeneous of degree 2, returns the symmetric matrix representing the same quadratic form. | — |
| `GramMatrix(L)` | The symmetric matrix giving the quadratic form on the lattice `L`. | — |
| `QuadraticForm(L)` | The quadratic form associated to the lattice `L`, as a multivariate polynomial. | — |
| `QuadraticForm(M)` | The quadratic form for a symmetric matrix `M`, as a multivariate polynomial. | — |

---

## 32.3 Local Invariants

These commands calculate the standard invariants that characterise a quadratic form over
the rationals. Definitions may be found in Conway–Sloane **[JC98]**, Chapter 15, Section 5.1.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `pSignature(f, p)` / `pSignature(M, p)` / `pSignature(L, p)` | The p-signature of the specified quadratic form over the rationals, where `p` is a prime number or `−1` (designating the real place). For odd primes `p`: diagonalize the form; add p-parts of diagonal entries to 4 times the number of anti-squares (mod p) among those entries (an "anti-square" mod p has odd valuation at p, and its prime-to-p part `u` satisfies the Kronecker symbol `(u/p) = −1`). At `p = 2`: sum of odd parts of diagonalized entries plus 4 times the number of anti-squares. At the real place: difference between the number of positive and negative eigenvalues. The result is defined modulo 8. | Diagonalization; **[JC98]** Ch. 15 §5.1 |
| `Oddity(f)` / `Oddity(L)` / `Oddity(M)` | Returns the 2-signature of the given quadratic form over the rationals. | Alias for `pSignature` at `p = 2`; **[JC98]** |
| `pExcess(f, p)` / `pExcess(M, p)` / `pExcess(L, p)` | The p-excess of the specified quadratic form over the rationals, where `p` is a prime or `−1`. The p-excess is the difference between the p-signature and the dimension for odd primes (including `−1`), and the negation of this for `p = 2`. The sum of p-excesses over all primes is 0 mod 8. | **[JC98]** Ch. 15 §5.1 |
| `WittInvariant(f, p)` / `WittInvariant(M, p)` / `WittInvariant(L, p)` | Calculates the Witt invariant (also called the Hasse–Minkowski invariant) over **Q**_p of the given quadratic form (must be defined over the rationals or integers; p-adic input is not allowed due to precision issues). Returns an element of `{−1, +1}`. Can also be called via `HasseMinkowskiInvariant`. | Diagonalize the form; take the product of Hilbert symbols of all `n choose 2` pairs of distinct nonzero diagonal entries; **[JC98]** Ch. 15 §5.3 |
| `HasseMinkowskiInvariant(f, p)` / `HasseMinkowskiInvariant(M, p)` / `HasseMinkowskiInvariant(L, p)` | Alias for `WittInvariant`. | As `WittInvariant`; **[JC98]** |
| `WittInvariants(f)` / `WittInvariants(M)` / `WittInvariants(L)` | Computes `WittInvariant(f, p)` for all bad primes `p`, and returns a sequence of tuples `⟨p, W_p(f)⟩`. The set of bad primes includes the real place, `p = 2`, and all primes dividing the numerator or denominator of the determinant of the symmetric matrix associated to `f`. Can also be called via `HasseMinkowskiInvariants`. | As `WittInvariant` applied at each bad prime; **[JC98]** |
| `HasseMinkowskiInvariants(f)` / `HasseMinkowskiInvariants(M)` / `HasseMinkowskiInvariants(L)` | Alias for `WittInvariants`. | As `WittInvariants`; **[JC98]** |

*Worked examples: H32E1 (computing `WittInvariant`, `pSignature`, `IsotropicSubspace` for a 4×4 integral matrix with determinant 1936 = 2⁴·11²; verification via Hilbert symbols).*

---

## 32.4 Isotropic Subspaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsotropicSubspace(f)` / `IsotropicSubspace(M)` | Returns an isotropic subspace for the given quadratic form (which must be integral or rational), given either as a multivariate polynomial `f` or as a symmetric matrix `M`. For a nonsingular form of signature `(r, s)` with `r ≥ s`, the dimension of the space returned is at least `min(r, s + 2) − 2`, which is maximal possible when `s ≤ r + 2`. Since version 2.18, an improvement typically enlarges the dimension by 1 when `r + s` is even and `s ≤ r + 2` (subject to a solvability criterion for a 4-dimensional subspace). There is no corresponding intrinsic for a lattice, since the associated form is definite and has no isotropic vectors. | Simon's algorithm **[Sim05]**, using the Bosma–Stevenhagen algorithm for the 2-part of class groups of quadratic fields **[BS96]**. |

*Worked examples: H32E1 (4×4 matrix, `IsotropicSubspace` returns a rank-2 subspace); H32E2 (random 20×20 form with 20 variables, `IsotropicSubspace` finds a dimension-8 isotropic subspace in 0.64 s; verification that all basis inner products vanish).*

---

## 32.5 Bibliography

| Key | Reference |
|-----|-----------|
| **[BS96]** | W. Bosma and P. Stevenhagen. *On the computation of quadratic 2-class groups.* Journal de théorie des nombres de Bordeaux, **8**(2):283–313, 1996. |
| **[JC98]** | N. J. A. Sloane, J. H. Conway. *Sphere Packings, Lattices and Groups*, volume 290 of Grundlehren der Mathematischen Wissenschaften. Springer, New York–Berlin–Heidelberg, 3rd edition, 1998. |
| **[Sim05]** | Denis Simon. *Quadratic equations in dimensions 4, 5 and more.* Preprint, URL: http://www.math.unicaen.fr/~simon/, 2005. |

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Diagonalization + Hilbert symbols (Witt / Hasse–Minkowski invariants) **[JC98]** | `WittInvariant`, `HasseMinkowskiInvariant`, `WittInvariants`, `HasseMinkowskiInvariants` |
| p-signature and p-excess computation **[JC98]** | `pSignature`, `Oddity`, `pExcess` |
| Simon's isotropic subspace algorithm **[Sim05]** | `IsotropicSubspace` |
| Bosma–Stevenhagen 2-class group algorithm **[BS96]** | `IsotropicSubspace` (subroutine) |
| Form / matrix / lattice conversions | `SymmetricMatrix`, `GramMatrix`, `QuadraticForm` |
