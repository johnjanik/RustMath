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
//! The characteristic polynomial is computed exactly with the
//! Faddeev–LeVerrier algorithm, which keeps every intermediate quantity in `Z`
//! (each division by `k` is exact by the theorem of Faddeev–LeVerrier).
//!
//! `integer_spectrum` extracts the *integer* eigenvalues exactly.  A graph all
//! of whose eigenvalues are integers is called *integral* (e.g. complete
//! graphs, complete bipartite graphs, hypercubes); for such graphs the returned
//! list is the full spectrum.  For non-integral graphs the irrational
//! eigenvalues are still captured exactly by the characteristic polynomial
//! itself.  (Extracting the irrational eigenvalues would require a
//! `RealField`-valued real-root isolator, deferred here.)

use crate::graph::Graph;
use crate::magma_matrices::{adjacency_matrix_integer, laplacian_matrix_integer};
use rustmath_core::EuclideanDomain;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use rustmath_polynomials::UnivariatePolynomial;

/// Characteristic polynomial `det(xI - M)` of an integer matrix `M`, computed
/// exactly by the Faddeev–LeVerrier recurrence.  Returns a monic
/// `UnivariatePolynomial<Integer>` of degree `n` (ascending coefficients).
///
/// Panics if `M` is not square.
pub fn char_poly_of_integer_matrix(m: &Matrix<Integer>) -> UnivariatePolynomial<Integer> {
    let n = m.rows();
    assert_eq!(n, m.cols(), "characteristic polynomial requires a square matrix");
    if n == 0 {
        // det of the empty matrix is 1 (the constant polynomial 1).
        return UnivariatePolynomial::from_coefficients(vec![Integer::one()]);
    }

    // coeffs[i] = coefficient of x^i, with coeffs[n] = 1 (monic).
    let mut coeffs = vec![Integer::zero(); n + 1];
    coeffs[n] = Integer::one();

    // Faddeev–LeVerrier:  M_1 = I;  for k = 1..n:
    //   AM   = A * M_k
    //   c_k  = -trace(AM) / k                (exact division in Z)
    //   coeff of x^{n-k} is c_k
    //   M_{k+1} = AM + c_k * I
    let mut mk: Matrix<Integer> = Matrix::identity(n); // M_1 = I
    for k in 1..=n {
        let am = m.mul(&mk).expect("square multiply");
        let tr = am.trace().expect("square trace");
        let (q, r) = tr
            .div_rem(&Integer::from(k as i64))
            .expect("k != 0");
        debug_assert!(r.is_zero(), "Faddeev–LeVerrier division must be exact");
        let c_k = Integer::zero() - q; // -trace/k
        coeffs[n - k] = c_k.clone();

        if k < n {
            // M_{k+1} = AM + c_k * I
            let mut next = am;
            for i in 0..n {
                let v = next.get(i, i).unwrap().clone() + c_k.clone();
                next.set(i, i, v).unwrap();
            }
            mk = next;
        }
    }

    UnivariatePolynomial::from_coefficients(coeffs)
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
}
