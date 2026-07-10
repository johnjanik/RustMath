//! Exact graph spectra over the integers (MAGMA Handbook Chapter 149,
//! §149.10 "Elementary Invariants of a Graph").
//!
//! MAGMA's `CharacteristicPolynomial(G)` is the characteristic polynomial *over
//! Z* of the adjacency matrix of `G`, and `Spectrum(G)` returns the roots of
//! that polynomial with multiplicity.  The pre-existing `spectra.rs` computes
//! these with an f64 QR iteration, which is inexact.  This module lives *beside*
//! it and provides the exact answers:
//!
//!   * `CharacteristicPolynomial(G)` -> [`characteristic_polynomial_integer`]
//!     as a `rustmath-polynomials::UnivariatePolynomial<Integer>`.
//!   * `Spectrum(G)` (integral part) -> [`integer_spectrum`] as
//!     `(Integer eigenvalue, multiplicity)` pairs.
//!
//! The characteristic polynomial is computed exactly by the division-free
//! Samuelson–Berkowitz algorithm ([`rustmath_matrix::charpoly_berkowitz`]),
//! which is valid over any commutative ring and in particular needs no
//! Faddeev–LeVerrier-style division by the step index `k` (that recurrence
//! requires a field, or at least that `k` be invertible, which fails e.g. over
//! `Z/pZ` for `k` a multiple of `p`; it happens to work over `Z` itself only
//! because every intermediate division is guaranteed exact by the
//! Faddeev–LeVerrier theorem). Berkowitz sidesteps that constraint entirely
//! and is the single canonical exact charpoly implementation shared across
//! RustMath; this module just plugs `Matrix<Integer>` into it.
//!
//! `integer_spectrum` extracts the *integer* eigenvalues exactly.  A graph all
//! of whose eigenvalues are integers is called *integral* (e.g. complete
//! graphs, complete bipartite graphs, hypercubes); for such graphs the returned
//! list is the full spectrum.  For non-integral graphs the irrational
//! eigenvalues are still captured exactly by the characteristic polynomial
//! itself.  (Extracting the irrational eigenvalues would require a
//! `RealField`-valued real-root isolator, deferred here.)
//!
//! [`spectral_radius_exact`], [`algebraic_connectivity_exact`] and
//! [`graph_energy_exact`] give exact `Integer` answers for the corresponding
//! `f64` functions in `crate::spectra` *when the graph (or its Laplacian) is
//! integral*; they return `None` otherwise, and the caller falls back to the
//! `f64` QR iteration in `crate::spectra` (which stays available for
//! non-integral graphs and for graphs too large to make exact root-search over
//! `[-Δ, Δ]` attractive). See `crate::spectra` module docs for that policy.

use crate::graph::Graph;
use crate::magma_matrices::{adjacency_matrix_integer, laplacian_matrix_integer};
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use rustmath_polynomials::UnivariatePolynomial;

/// Characteristic polynomial `det(xI - M)` of an integer matrix `M`, computed
/// exactly by the division-free Samuelson–Berkowitz algorithm (delegates to
/// [`rustmath_matrix::charpoly_berkowitz`], the canonical exact-charpoly
/// implementation, so this module carries no duplicate linear-algebra logic).
/// Returns a monic `UnivariatePolynomial<Integer>` of degree `n` (ascending
/// coefficients).
///
/// Panics if `M` is not square (Berkowitz already validates this; the
/// assertion below documents the precondition to callers).
pub fn char_poly_of_integer_matrix(m: &Matrix<Integer>) -> UnivariatePolynomial<Integer> {
    assert_eq!(m.rows(), m.cols(), "characteristic polynomial requires a square matrix");
    rustmath_matrix::charpoly_berkowitz(m).expect("square matrix: charpoly_berkowitz cannot fail")
}

/// `CharacteristicPolynomial(G)` — the exact characteristic polynomial over `Z`
/// of the adjacency matrix of `G`.
pub fn characteristic_polynomial_integer(g: &Graph) -> UnivariatePolynomial<Integer> {
    char_poly_of_integer_matrix(&adjacency_matrix_integer(g))
}

/// The exact characteristic polynomial over `Z` of the Laplacian matrix of `G`
/// (RustMath extension; the *Laplacian spectrum* underlies Kirchhoff's
/// spanning-tree theorem and algebraic connectivity).
pub fn laplacian_characteristic_polynomial_integer(g: &Graph) -> UnivariatePolynomial<Integer> {
    char_poly_of_integer_matrix(&laplacian_matrix_integer(g))
}

/// Divide out one factor `(x - root)` from a monic integer polynomial, returning
/// the quotient, provided `root` is actually a root.  `poly` is given as
/// ascending coefficients.
fn deflate_at_integer_root(coeffs: &[Integer], root: &Integer) -> Option<Vec<Integer>> {
    // Synthetic division by (x - root).  For ascending coeffs a_0..a_d,
    // process from the top down.
    let d = coeffs.len();
    if d == 0 {
        return None;
    }
    let mut quotient = vec![Integer::zero(); d - 1];
    let mut carry = Integer::zero();
    for i in (0..d).rev() {
        let cur = coeffs[i].clone() + carry.clone();
        if i == 0 {
            // remainder
            if !cur.is_zero() {
                return None;
            }
        } else {
            quotient[i - 1] = cur.clone();
            carry = cur * root.clone();
        }
    }
    Some(quotient)
}

/// Evaluate an ascending-coefficient integer polynomial at `x`.
fn eval_integer_poly(coeffs: &[Integer], x: &Integer) -> Integer {
    let mut acc = Integer::zero();
    for c in coeffs.iter().rev() {
        acc = acc * x.clone() + c.clone();
    }
    acc
}

/// Integer roots (with multiplicity) of a monic integer polynomial, searched
/// over the closed range `[lo, hi]`.  Returned sorted ascending by eigenvalue.
fn integer_roots_in_range(
    poly: &UnivariatePolynomial<Integer>,
    lo: i64,
    hi: i64,
) -> Vec<(Integer, usize)> {
    let mut coeffs: Vec<Integer> = poly.coefficients().to_vec();
    // Normalize away trailing zero (zero polynomial guard).
    if coeffs.is_empty() {
        return vec![];
    }
    let mut out = Vec::new();
    let mut cand = lo;
    while cand <= hi {
        let c = Integer::from(cand);
        let mut mult = 0usize;
        // Peel off repeated roots.
        while coeffs.len() > 1 && eval_integer_poly(&coeffs, &c).is_zero() {
            match deflate_at_integer_root(&coeffs, &c) {
                Some(q) => {
                    coeffs = q;
                    mult += 1;
                }
                None => break,
            }
        }
        if mult > 0 {
            out.push((c, mult));
        }
        cand += 1;
    }
    out
}

/// `Spectrum(G)` (integral part) — the integer eigenvalues of the adjacency
/// matrix of `G`, each paired with its multiplicity, sorted ascending.
///
/// All adjacency eigenvalues lie in `[-Δ, Δ]` where `Δ` is the maximum degree,
/// so the search range is exact and finite.  For an *integral* graph the sum of
/// the returned multiplicities equals the number of vertices.
pub fn integer_spectrum(g: &Graph) -> Vec<(Integer, usize)> {
    let n = g.num_vertices();
    if n == 0 {
        return vec![];
    }
    let max_deg = (0..n).map(|v| g.degree(v).unwrap_or(0)).max().unwrap_or(0) as i64;
    let poly = characteristic_polynomial_integer(g);
    integer_roots_in_range(&poly, -max_deg, max_deg)
}

/// The integer Laplacian eigenvalues of `G` with multiplicity, sorted ascending.
/// Laplacian eigenvalues lie in `[0, n]`.
pub fn laplacian_integer_spectrum(g: &Graph) -> Vec<(Integer, usize)> {
    let n = g.num_vertices();
    if n == 0 {
        return vec![];
    }
    let poly = laplacian_characteristic_polynomial_integer(g);
    integer_roots_in_range(&poly, 0, n as i64)
}

/// Whether `g` is an *integral graph*: every adjacency eigenvalue is an
/// integer. Decided exactly, by checking that [`integer_spectrum`] already
/// accounts for all `n` eigenvalues (with multiplicity) — no floating point
/// or root isolation involved.
pub fn is_integral_graph(g: &Graph) -> bool {
    let n = g.num_vertices();
    let total: usize = integer_spectrum(g).iter().map(|(_, m)| *m).sum();
    total == n
}

/// Whether the Laplacian spectrum of `g` is entirely integral. Decided
/// exactly, analogously to [`is_integral_graph`].
pub fn is_laplacian_integral_graph(g: &Graph) -> bool {
    let n = g.num_vertices();
    let total: usize = laplacian_integer_spectrum(g).iter().map(|(_, m)| *m).sum();
    total == n
}

/// Exact spectral radius `max_i |λ_i|` of the adjacency matrix of `g`, i.e.
/// MAGMA/Sage's `spectral_radius`, computed exactly rather than by the `f64`
/// QR iteration in [`crate::spectra::spectral_radius`].
///
/// The adjacency matrix is entrywise nonnegative, so by the Perron–Frobenius
/// theorem its eigenvalue of largest modulus is real, nonnegative, and equal
/// to the *largest* eigenvalue (not merely largest in absolute value) — this
/// holds for every graph, connected or not. Consequently, whenever the graph
/// is [`is_integral_graph`] the maximum entry of [`integer_spectrum`] *is* the
/// exact spectral radius, and this function returns it. Returns `None` when
/// `g` is not integral (fall back to `crate::spectra::spectral_radius` for an
/// `f64` approximation in that case) or when `g` has no vertices.
pub fn spectral_radius_exact(g: &Graph) -> Option<Integer> {
    let n = g.num_vertices();
    if n == 0 {
        return None;
    }
    let spec = integer_spectrum(g);
    let total: usize = spec.iter().map(|(_, m)| *m).sum();
    if total != n {
        return None;
    }
    spec.into_iter().map(|(v, _)| v).max()
}

/// Exact algebraic connectivity (Fiedler value): the second-smallest
/// Laplacian eigenvalue of `g`, counted with multiplicity, when the full
/// Laplacian spectrum is integral (see [`is_laplacian_integral_graph`]).
/// Laplacian eigenvalues are always real and `>= 0` (`L` is positive
/// semidefinite), and [`laplacian_integer_spectrum`] is already sorted
/// ascending, so the second entry (expanded by multiplicity) is exact.
/// Returns `None` for `n < 2` or when the Laplacian spectrum is not fully
/// integral; fall back to `crate::spectra::algebraic_connectivity` for an
/// `f64` approximation in that case.
pub fn algebraic_connectivity_exact(g: &Graph) -> Option<Integer> {
    let n = g.num_vertices();
    if n < 2 {
        return None;
    }
    let spec = laplacian_integer_spectrum(g);
    let total: usize = spec.iter().map(|(_, m)| *m).sum();
    if total != n {
        return None;
    }
    // Expand by multiplicity (spec is sorted ascending) and take index 1.
    let mut seen = 0usize;
    for (v, m) in spec {
        if seen + m > 1 {
            return Some(v);
        }
        seen += m;
    }
    None // unreachable when total == n and n >= 2
}

/// Exact graph energy `Σ_i |λ_i|` of the adjacency spectrum of `g`, when `g`
/// is integral (see [`is_integral_graph`]); `None` otherwise (fall back to
/// `crate::spectra::graph_energy`, which is `f64`-only).
pub fn graph_energy_exact(g: &Graph) -> Option<Integer> {
    let n = g.num_vertices();
    if n == 0 {
        return Some(Integer::zero());
    }
    let spec = integer_spectrum(g);
    let total: usize = spec.iter().map(|(_, m)| *m).sum();
    if total != n {
        return None;
    }
    Some(
        spec.into_iter()
            .map(|(v, m)| v.abs() * Integer::from(m as i64))
            .fold(Integer::zero(), |acc, x| acc + x),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j).unwrap();
            }
        }
        g
    }

    fn cycle(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            g.add_edge(i, (i + 1) % n).unwrap();
        }
        g
    }

    #[test]
    fn single_edge_char_poly_is_x2_minus_1() {
        // K2: adjacency = [[0,1],[1,0]], char poly = x^2 - 1.
        let g = complete(2);
        let p = characteristic_polynomial_integer(&g);
        let c = p.coefficients();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], Integer::from(-1)); // constant
        assert_eq!(c[1], Integer::zero());
        assert_eq!(c[2], Integer::one()); // monic x^2
    }

    #[test]
    fn triangle_char_poly() {
        // K3 char poly of adjacency: x^3 - 3x - 2 = (x-2)(x+1)^2.
        let g = complete(3);
        let p = characteristic_polynomial_integer(&g);
        let c = p.coefficients();
        assert_eq!(c[0], Integer::from(-2));
        assert_eq!(c[1], Integer::from(-3));
        assert_eq!(c[2], Integer::zero());
        assert_eq!(c[3], Integer::one());
    }

    #[test]
    fn complete_graph_spectrum_is_integral() {
        // K_n has eigenvalues n-1 (once) and -1 (n-1 times).
        let g = complete(4);
        let spec = integer_spectrum(&g);
        // Sorted ascending: (-1, 3), (3, 1).
        assert_eq!(spec, vec![(Integer::from(-1), 3), (Integer::from(3), 1)]);
        let total: usize = spec.iter().map(|(_, m)| m).sum();
        assert_eq!(total, 4); // integral graph: all eigenvalues accounted for.
    }

    #[test]
    fn c4_spectrum_is_integral() {
        // C4 has adjacency eigenvalues 2, 0, 0, -2.
        let g = cycle(4);
        let spec = integer_spectrum(&g);
        assert_eq!(
            spec,
            vec![
                (Integer::from(-2), 1),
                (Integer::from(0), 2),
                (Integer::from(2), 1)
            ]
        );
    }

    #[test]
    fn c5_spectrum_only_captures_integer_eigenvalue() {
        // C5 has one integer eigenvalue (2, the degree) and four irrational ones;
        // integer_spectrum returns just the integral part.
        let g = cycle(5);
        let spec = integer_spectrum(&g);
        assert_eq!(spec, vec![(Integer::from(2), 1)]);
    }

    #[test]
    fn laplacian_spectrum_complete_graph() {
        // K_n Laplacian eigenvalues: 0 (once) and n (n-1 times).
        let g = complete(4);
        let spec = laplacian_integer_spectrum(&g);
        assert_eq!(spec, vec![(Integer::from(0), 1), (Integer::from(4), 3)]);
    }

    #[test]
    fn laplacian_constant_term_is_zero() {
        // 0 is always a Laplacian eigenvalue -> constant coefficient is 0.
        let g = cycle(5);
        let p = laplacian_characteristic_polynomial_integer(&g);
        assert_eq!(p.coefficients()[0], Integer::zero());
    }

    // ---- Known-family exact spectra (independently verified with sympy
    // before writing these assertions; see the P2-E2 handoff notes). ----

    #[test]
    fn complete_graph_family_spectrum_n_2_to_6() {
        // K_n: adjacency spectrum {n-1 (mult 1), -1 (mult n-1)};
        // Laplacian spectrum {0 (mult 1), n (mult n-1)}; energy = 2(n-1);
        // algebraic connectivity = n. Cross-checked with sympy for n=2..6.
        for n in 2..=6usize {
            let g = complete(n);
            assert!(is_integral_graph(&g));
            assert!(is_laplacian_integral_graph(&g));

            let spec = integer_spectrum(&g);
            assert_eq!(
                spec,
                vec![(Integer::from(-1), n - 1), (Integer::from(n as i64 - 1), 1)]
            );

            let lspec = laplacian_integer_spectrum(&g);
            assert_eq!(
                lspec,
                vec![(Integer::from(0), 1), (Integer::from(n as i64), n - 1)]
            );

            assert_eq!(spectral_radius_exact(&g), Some(Integer::from(n as i64 - 1)));
            assert_eq!(algebraic_connectivity_exact(&g), Some(Integer::from(n as i64)));
            assert_eq!(
                graph_energy_exact(&g),
                Some(Integer::from(2 * (n as i64 - 1)))
            );
        }
    }

    #[test]
    fn petersen_graph_adjacency_spectrum() {
        // Petersen graph: srg(10,3,0,1), adjacency spectrum {3^1, 1^5, (-2)^4}
        // (verified independently with sympy: charpoly matches
        // x^10 - 15x^8 + 75x^6 - 24x^5 - 165x^4 + 120x^3 + 120x^2 - 160x + 48,
        // and A.eigenvals() == {3: 1, 1: 5, -2: 4}).
        let g = crate::generators::petersen_graph();
        assert!(is_integral_graph(&g));

        let spec = integer_spectrum(&g);
        assert_eq!(
            spec,
            vec![
                (Integer::from(-2), 4),
                (Integer::from(1), 5),
                (Integer::from(3), 1),
            ]
        );

        // Perron–Frobenius top eigenvalue == the (regular) degree, 3.
        assert_eq!(spectral_radius_exact(&g), Some(Integer::from(3)));
        // Known value: graph energy of the Petersen graph is 16.
        assert_eq!(graph_energy_exact(&g), Some(Integer::from(16)));
    }

    #[test]
    fn petersen_graph_laplacian_spectrum() {
        // Laplacian spectrum {0^1, 2^5, 5^4} (verified with sympy:
        // L.eigenvals() == {0: 1, 2: 5, 5: 4}); algebraic connectivity 2 is
        // the well-known Fiedler value of the Petersen graph.
        let g = crate::generators::petersen_graph();
        assert!(is_laplacian_integral_graph(&g));

        let lspec = laplacian_integer_spectrum(&g);
        assert_eq!(
            lspec,
            vec![
                (Integer::from(0), 1),
                (Integer::from(2), 5),
                (Integer::from(5), 4),
            ]
        );
        assert_eq!(algebraic_connectivity_exact(&g), Some(Integer::from(2)));
    }

    #[test]
    fn non_integral_graph_exact_helpers_report_none() {
        // C5 is not integral (one integer eigenvalue, four irrational); the
        // exact helpers must honestly report None rather than guess, so
        // callers fall back to crate::spectra's f64 QR iteration.
        let g = cycle(5);
        assert!(!is_integral_graph(&g));
        assert_eq!(spectral_radius_exact(&g), None);
        assert_eq!(graph_energy_exact(&g), None);
    }
}
