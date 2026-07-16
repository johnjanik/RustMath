//! IGP24 closure wiring (items C4–C7) — campaign-shaped call sites over
//! algorithms that already exist in the workspace. Nothing here reimplements
//! mathematics: each function adapts an existing, tested implementation to
//! the shape the M23/Q solve campaign consumes.
//!
//! * C4: `F_p` factorization (`rustmath_polynomials::fp_factor`) and `Q_p`
//!   OM/MacLane factorization (`rustmath_rings::padics::om_factorization`).
//! * C5: Newton polygon over `Q_2`
//!   (`rustmath_rings::padics::NewtonPolygon`), cross-certified against the
//!   OM leaves' `(e, f)` data.
//! * C6: the framed-ideal Jacobian rank gate over `F_p`
//!   (`rustmath_polynomials::PolySystem::jacobian_mod` + mod-p elimination),
//!   the Route-B well-posedness precondition for the C1 Newton lift.
//! * C7: conic isotropy re-point
//!   (`rustmath_quadraticforms::conic::DiagonalConicQ::verdict`) for the S100
//!   descent datum.
//!
//! This module deliberately lives OUTSIDE `src/belyi/`: the M23 solve_runner
//! `#[path]`-includes `belyi/mod.rs` and must not be forced to compile
//! `rustmath_rings`; modules registered only in this crate's `lib.rs` are
//! free to use the full dependency set.

use rustmath_integers::Integer;
use rustmath_polynomials::{fp_factor, PolySystem, UnivariatePolynomial};
use rustmath_quadraticforms::conic::{ConicBrauerReport, DiagonalConicQ, Verdict};
use rustmath_rationals::Rational;
use rustmath_rings::padics::{om_factorization, NewtonPolygon, OmFactorization};

pub use rustmath_quadraticforms::conic::VerdictKind;

// Primes must stay below 2^31 so i64 products of two reduced residues cannot
// overflow (both in fp_factor's arithmetic and the local elimination below).
const MAX_SMALL_PRIME: i64 = 1 << 31;

fn check_small_prime(p: i64, who: &str) -> Result<(), String> {
    if p < 2 || p >= MAX_SMALL_PRIME {
        return Err(format!("{who}: p = {p} out of range [2, 2^31)"));
    }
    if !Integer::from(p).is_prime() {
        return Err(format!("{who}: p = {p} is not prime"));
    }
    Ok(())
}

// --- C4a: F_p univariate factorization -----------------------------------

/// Irreducible factorization of `f mod p` in `F_p[x]`.
///
/// `f` is little-endian over `Z`; the result is the list of distinct monic
/// irreducible factors of the reduction, little-endian with coefficients in
/// `[0, p)`, sorted by (degree, coefficients) for stable consumption.
///
/// Honest caveats, inherited from [`fp_factor::factor`]: multiplicities are
/// NOT tracked — a repeated factor appears once. The reduction of the leading
/// coefficient must not vanish mod `p` (`Err` otherwise): a silent degree
/// drop would corrupt the campaign's degree bookkeeping.
pub fn factor_mod_p(f: &[Integer], p: i64) -> Result<Vec<Vec<i64>>, String> {
    check_small_prime(p, "factor_mod_p")?;
    let deg_z = match f.iter().rposition(|c| !c.is_zero()) {
        Some(d) => d,
        None => return Err("factor_mod_p: zero polynomial".to_string()),
    };
    if deg_z == 0 {
        return Err("factor_mod_p: constant polynomial has no factorization".to_string());
    }
    let p_int = Integer::from(p);
    let reduced: Vec<i64> = f[..=deg_z]
        .iter()
        .map(|c| c.modulo(&p_int).to_i64())
        .collect();
    if reduced[deg_z] == 0 {
        return Err(format!(
            "factor_mod_p: leading coefficient vanishes mod {p} (degree would drop)"
        ));
    }
    let mut factors = fp_factor::factor(&reduced, p);
    factors.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    Ok(factors)
}

// --- C4b: Q_p local factorization (OM / MacLane) --------------------------

/// Campaign-shaped summary of the OM factorization of `f` over `Q_p`.
#[derive(Clone, Debug)]
pub struct QpLocalFactorization {
    /// The prime.
    pub p: i64,
    /// Per-factor `(degree, e, f)` with `degree = e * f`, sorted. DECIDED
    /// data (certified by the MacLane tree's completeness check).
    pub shapes: Vec<(usize, u64, u64)>,
    /// Congruence certificate for the factor approximations:
    /// `prod_i phi_i ≡ f mod p^N`; `None` means the product is exactly `f`.
    pub congruence_precision: Option<u32>,
    /// The full OM object (per-factor approximations and MacLane leaves via
    /// `om.factors()[i].approximation() / .leaf()`).
    pub om: OmFactorization,
}

/// OM (Okutsu–Montes / MacLane) factorization of `f` over `Q_p`.
///
/// `f` is little-endian over `Q`; it must be monic, squarefree over `Q` and
/// p-integral (`Err` otherwise, from the MacLane gates). `prec` is the
/// minimum congruence precision requested for the factor approximations; the
/// `(e, f)` invariants are exact regardless.
pub fn qp_local_factorization(
    f: &[Rational],
    p: i64,
    prec: u32,
) -> Result<QpLocalFactorization, String> {
    check_small_prime(p, "qp_local_factorization")?;
    let poly = UnivariatePolynomial::new(f.to_vec());
    let om = om_factorization(&poly, p, prec)
        .map_err(|e| format!("qp_local_factorization: {e}"))?;
    let mut shapes: Vec<(usize, u64, u64)> = om
        .factors()
        .iter()
        .map(|g| (g.degree(), g.e(), g.f()))
        .collect();
    shapes.sort();
    Ok(QpLocalFactorization {
        p,
        shapes,
        congruence_precision: om.congruence_precision(),
        om,
    })
}

// --- C5: Newton polygon over Q_2 ------------------------------------------

/// Newton polygon of `f` (little-endian over `Q`) with respect to `p = 2`.
///
/// Returns `(slopes, root_valuations)`:
/// * `slopes`: `(slope, horizontal length)` per hull segment, slopes strictly
///   increasing left to right;
/// * `root_valuations`: `(valuation, count)` — `count` roots of `f` in an
///   algebraic closure of `Q_2` of 2-adic valuation `valuation` (with
///   multiplicity), by decreasing valuation. Roots equal to 0 are not listed.
pub fn newton_polygon_q2(
    f: &[Rational],
) -> Result<(Vec<(Rational, u64)>, Vec<(Rational, u64)>), String> {
    let polygon = NewtonPolygon::of_rational_polynomial(f, &Integer::from(2))
        .map_err(|e| format!("newton_polygon_q2: {e}"))?;
    let slopes = polygon
        .slopes()
        .into_iter()
        .map(|s| (s.slope, s.length))
        .collect();
    Ok((slopes, polygon.root_valuations()))
}

// --- C6: framed-ideal Jacobian rank gate over F_p --------------------------

/// Rank over `F_p` of the Jacobian of `system` at `point`.
///
/// The Jacobian is taken via [`PolySystem::jacobian_mod`] (entries reduced to
/// `[0, p)`), then row-reduced by a LOCAL dense mod-p Gaussian elimination
/// over `i64` rather than `Matrix<PrimeField>`: `PrimeField`'s
/// `Ring::zero()`/`Ring::one()` PANIC (frozen-crate bug B-01 — no modulus
/// context in the trait constructors), so `Matrix<PrimeField>` is only
/// accidentally panic-free on today's `rank()` code path and we refuse to
/// build a campaign gate on that accident. Entries are `< p < 2^31`, so i64
/// products cannot overflow. Pivot inverses reuse [`fp_factor::mod_inv`].
pub fn framed_jacobian_rank_mod_p(
    system: &PolySystem,
    point: &[Integer],
    p: i64,
) -> Result<usize, String> {
    check_small_prime(p, "framed_jacobian_rank_mod_p")?;
    if point.len() < system.num_variables() {
        return Err(format!(
            "framed_jacobian_rank_mod_p: point has {} coordinates, system has {} variables",
            point.len(),
            system.num_variables()
        ));
    }
    let jac = system.jacobian_mod(point, &Integer::from(p));
    let rows: Vec<Vec<i64>> = jac
        .iter()
        .map(|row| row.iter().map(|c| c.to_i64()).collect())
        .collect();
    Ok(rank_mod_p(rows, p))
}

/// Rank of an i64 matrix over `F_p` by forward Gaussian elimination.
/// Precondition (enforced by callers): `p` prime, `2 <= p < 2^31`, entries
/// reduced to `[0, p)`.
fn rank_mod_p(mut rows: Vec<Vec<i64>>, p: i64) -> usize {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let mut rank = 0;
    for col in 0..ncols {
        if rank == nrows {
            break;
        }
        let Some(pivot_row) = (rank..nrows).find(|&r| rows[r][col] % p != 0) else {
            continue;
        };
        rows.swap(rank, pivot_row);
        let inv = fp_factor::mod_inv(rows[rank][col], p)
            .expect("pivot is nonzero mod a prime, hence invertible");
        for j in col..ncols {
            rows[rank][j] = (rows[rank][j] % p * inv).rem_euclid(p);
        }
        for r in (rank + 1)..nrows {
            let factor = rows[r][col] % p;
            if factor != 0 {
                for j in col..ncols {
                    rows[r][j] = (rows[r][j] - factor * rows[rank][j]).rem_euclid(p);
                }
            }
        }
        rank += 1;
    }
    rank
}

/// Route-B well-posedness gate: does the framed Jacobian of `system` at
/// `point` have rank exactly `expected` over `F_p`?
///
/// Campaign role: `rank_gate(system, f17_point, 17, 75)` is the PRECONDITION
/// for the C1 Newton lift — a lift from the F_17 seed is well-posed only when
/// the framed Jacobian has the expected full rank 75 there. A deficient rank
/// means the seed sits on a degenerate stratum (duplicate constraint, seed
/// off the variety) and the lift must not be attempted.
pub fn rank_gate(
    system: &PolySystem,
    point: &[Integer],
    p: i64,
    expected: usize,
) -> Result<bool, String> {
    Ok(framed_jacobian_rank_mod_p(system, point, p)? == expected)
}

// --- C7: conic isotropy re-point --------------------------------------------

/// Hasse–Minkowski verdict for the diagonal ternary conic
/// `a x^2 + b y^2 + c z^2 = 0` over `Q`.
///
/// Thin adapter from a raw S100 descent datum `(a, b, c)` to the existing
/// [`DiagonalConicQ::verdict`] machinery (the belyi portal already consumes
/// it for the L-cover path). `bad_locus_clear` is the caller's certificate
/// that a produced rational point lies off the bad locus `Z_C`; a point on
/// `Z_C` does not realise `M23/Q`, so without the certificate an isotropic
/// conic yields `Unresolved`, not `Constructed`. Anisotropy (a nonempty
/// ramified-place set for the quaternion class `(-a/c, -b/c)`) yields
/// `LocallyEmpty` — a theorem-grade local obstruction.
pub fn descent_conic_verdict(
    a: &Rational,
    b: &Rational,
    c: &Rational,
    bad_locus_clear: bool,
) -> Result<Verdict<ConicBrauerReport>, String> {
    let conic = DiagonalConicQ::new(a.clone(), b.clone(), c.clone())
        .map_err(|e| format!("descent_conic_verdict: {e}"))?;
    conic
        .verdict(bad_locus_clear)
        .map_err(|e| format!("descent_conic_verdict: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_polynomials::poly_system::PolySystem;
    use rustmath_quadraticforms::hilbert::{
        candidate_places_for_quaternion, hilbert_symbol,
    };

    fn int(n: i64) -> Integer {
        Integer::from(n)
    }

    fn rat(n: i64, d: i64) -> Rational {
        Rational::new(int(n), int(d)).unwrap()
    }

    fn rats(coeffs: &[i64]) -> Vec<Rational> {
        coeffs.iter().map(|&c| Rational::from_i64(c)).collect()
    }

    // --- C4a ---------------------------------------------------------------

    #[test]
    fn c4_factor_mod_17_splits_x4_plus_1() {
        // Oracle (gp `factormod(x^4+1, 17)` + hand check 2^4 = 16 = -1 mod 17,
        // roots {2, 8, 9, 15}): x^4+1 = (x+2)(x+8)(x+9)(x+15) mod 17.
        let f: Vec<Integer> = [1, 0, 0, 0, 1].iter().map(|&c| int(c)).collect();
        let factors = factor_mod_p(&f, 17).unwrap();
        assert_eq!(
            factors,
            vec![vec![2, 1], vec![8, 1], vec![9, 1], vec![15, 1]]
        );
        // self-certification: the product of the factors reproduces f mod 17
        let mut product = vec![1i64];
        for g in &factors {
            product = fp_factor::mul(&product, g, 17);
        }
        assert_eq!(product, vec![1, 0, 0, 0, 1]);
    }

    #[test]
    fn c4_factor_mod_17_mixed_degrees() {
        // Oracle (gp + synthetic division by the cube root 8 of 2 mod 17):
        // x^3 - 2 = (x+9)(x^2+8x+13) mod 17, the quadratic irreducible since
        // its discriminant 64-52 = 12 is a non-residue mod 17.
        let f: Vec<Integer> = [-2, 0, 0, 1].iter().map(|&c| int(c)).collect();
        let factors = factor_mod_p(&f, 17).unwrap();
        assert_eq!(factors, vec![vec![9, 1], vec![13, 8, 1]]);
        let mut product = vec![1i64];
        for g in &factors {
            product = fp_factor::mul(&product, g, 17);
        }
        assert_eq!(product, vec![15, 0, 0, 1]); // x^3 - 2 ≡ x^3 + 15
    }

    #[test]
    fn c4_factor_mod_p_rejects_bad_input() {
        let f: Vec<Integer> = [1, 1].iter().map(|&c| int(c)).collect();
        assert!(factor_mod_p(&f, 15).is_err()); // composite p
        assert!(factor_mod_p(&[int(5)], 17).is_err()); // constant
        assert!(factor_mod_p(&[], 17).is_err()); // zero polynomial
        // leading coefficient 17 vanishes mod 17 -> degree would drop
        let g: Vec<Integer> = [1, 1, 17].iter().map(|&c| int(c)).collect();
        assert!(factor_mod_p(&g, 17).is_err());
    }

    // --- C4b ---------------------------------------------------------------

    #[test]
    fn c4_qp_local_factorization_over_q2() {
        // f = (x^2 - 2)(x^2 + x + 1) = x^4 + x^3 - x^2 - 2x - 2, monic and
        // squarefree over Q (gp: gcd(f, f') = 1).
        // Oracle (gp): factorpadic(f, 2, 30) gives two degree-2 factors;
        // idealprimedec(nfinit(y^2-2), 2)   -> one prime, e=2, f=1 (Eisenstein);
        // idealprimedec(nfinit(y^2+y+1), 2) -> one prime, e=1, f=2 (x^2+x+1
        // irreducible mod 2). Each field has a single prime above 2, so the
        // local factor <-> prime correspondence is unambiguous.
        let f = rats(&[-2, -2, -1, 1, 1]);
        let fac = qp_local_factorization(&f, 2, 12).unwrap();
        assert_eq!(fac.p, 2);
        assert_eq!(fac.shapes, vec![(2, 1, 2), (2, 2, 1)]);
        assert_eq!(fac.shapes.iter().map(|s| s.0).sum::<usize>(), 4);
        assert!(!fac.om.is_irreducible());
        // requested congruence certificate honored (None = exact product)
        assert!(fac.congruence_precision.map_or(true, |n| n >= 12));
    }

    // --- C5 ------------------------------------------------------------------

    #[test]
    fn c5_newton_polygon_q2_eisenstein_cubic_pairs_with_om() {
        // g = x^3 + 2x + 2, Eisenstein at 2 (2 | a_0, a_1, a_2; 4 ∤ a_0).
        // Polygon by hand: points (0,1), (1,1), (3,0); (1,1) lies above the
        // segment (0,1)-(3,0) (which passes through (1, 2/3)), so the hull is
        // the single segment of slope -1/3, length 3.
        // Oracle (gp): newtonpoly(g, 2) = [1/3, 1/3, 1/3];
        // idealprimedec(nfinit(y^3+2y+2), 2) -> one prime, e=3, f=1.
        let g = rats(&[2, 2, 0, 1]);
        let (slopes, root_vals) = newton_polygon_q2(&g).unwrap();
        assert_eq!(slopes, vec![(rat(-1, 3), 3)]);
        assert_eq!(root_vals, vec![(rat(1, 3), 3)]);

        // Cross-check against the OM leaves at p = 2: one factor, e=3, f=1.
        let fac = qp_local_factorization(&g, 2, 8).unwrap();
        assert_eq!(fac.shapes, vec![(3, 3, 1)]);
        // pairing: the 3 roots of valuation 1/3 (denominator 3) belong to the
        // unique factor, whose ramification index e = 3 matches the
        // denominator; segment length = e * f = factor degree.
        assert_eq!(root_vals[0].0.denominator(), &int(3));
        assert_eq!(root_vals[0].1, fac.shapes[0].1 * fac.shapes[0].2);
    }

    #[test]
    fn c5_newton_polygon_q2_two_slopes_pair_with_om() {
        // h = (x^2 - 2)(x - 4) = x^3 - 4x^2 - 2x + 8, monic squarefree.
        // Polygon by hand: points (0,3), (1,1), (2,2), (3,0); (2,2) lies above
        // the segment (1,1)-(3,0) (through (2, 1/2)), so the hull is
        // (0,3)-(1,1)-(3,0): slopes -2 (length 1) and -1/2 (length 2).
        // Oracle (gp): newtonpoly(h, 2) = [2, 1/2, 1/2];
        // factorpadic(h, 2, 30) -> degrees [1, 2].
        let h = rats(&[8, -2, -4, 1]);
        let (slopes, root_vals) = newton_polygon_q2(&h).unwrap();
        assert_eq!(slopes, vec![(rat(-2, 1), 1), (rat(-1, 2), 2)]);
        assert_eq!(root_vals, vec![(rat(2, 1), 1), (rat(1, 2), 2)]);

        // Cross-check against the OM leaves at p = 2 (same input, same test):
        // x - 4 is (1,1); x^2 - 2 is Eisenstein, (2,1).
        let fac = qp_local_factorization(&h, 2, 8).unwrap();
        assert_eq!(fac.shapes, vec![(1, 1, 1), (2, 2, 1)]);
        // pairing: total polygon roots = total factor degree;
        let polygon_roots: u64 = root_vals.iter().map(|rv| rv.1).sum();
        let factor_degree: usize = fac.shapes.iter().map(|s| s.0).sum();
        assert_eq!(polygon_roots as usize, factor_degree);
        // the 2 roots of half-integer valuation are exactly the roots of the
        // e = 2 factor (its degree is 2); the 1 integer-valuation root is the
        // e = 1, f = 1 factor's.
        assert_eq!(root_vals[1].0.denominator(), &int(2));
        assert_eq!(root_vals[1].1 as usize, fac.shapes[1].0);
        assert_eq!(fac.shapes[1].1, 2);
        assert_eq!(root_vals[0].0.denominator(), &int(1));
        assert_eq!(root_vals[0].1 as usize, fac.shapes[0].0);
        assert_eq!(fac.shapes[0].1, 1);
    }

    // --- C6 ------------------------------------------------------------------

    /// f1 = x^2 - y, f2 = y^2 - z, f3 = x + y + z (Jacobian rank 3 at (1,1,1)).
    fn full_rank_system() -> PolySystem {
        PolySystem::from_terms(
            3,
            &[
                vec![(vec![2, 0, 0], 1), (vec![0, 1, 0], -1)],
                vec![(vec![0, 2, 0], 1), (vec![0, 0, 1], -1)],
                vec![(vec![1, 0, 0], 1), (vec![0, 1, 0], 1), (vec![0, 0, 1], 1)],
            ],
        )
    }

    /// f1 = x^2 - yz, f2 = y^2 - xz, f3 = z^2 - xy (rank drops on x = y = z).
    fn stratified_system() -> PolySystem {
        PolySystem::from_terms(
            3,
            &[
                vec![(vec![2, 0, 0], 1), (vec![0, 1, 1], -1)],
                vec![(vec![0, 2, 0], 1), (vec![1, 0, 1], -1)],
                vec![(vec![0, 0, 2], 1), (vec![1, 1, 0], -1)],
            ],
        )
    }

    #[test]
    fn c6_full_rank_seed_passes_gate() {
        // Oracle (sympy over GF(17)): Matrix([[2,-1,0],[0,2,-1],[1,1,1]])
        // .rank() = 3 (det = 7 mod 17, hand-expanded).
        let system = full_rank_system();
        let seed = [int(1), int(1), int(1)];
        assert_eq!(framed_jacobian_rank_mod_p(&system, &seed, 17).unwrap(), 3);
        assert!(rank_gate(&system, &seed, 17, 3).unwrap());
        assert!(!rank_gate(&system, &seed, 17, 2).unwrap());
    }

    #[test]
    fn c6_duplicate_equation_is_rank_deficient() {
        // Same system with f3 replaced by a duplicate of f1.
        // Oracle (sympy over GF(17)): rank([[2,-1,0],[0,2,-1],[2,-1,0]]) = 2.
        let system = PolySystem::from_terms(
            3,
            &[
                vec![(vec![2, 0, 0], 1), (vec![0, 1, 0], -1)],
                vec![(vec![0, 2, 0], 1), (vec![0, 0, 1], -1)],
                vec![(vec![2, 0, 0], 1), (vec![0, 1, 0], -1)],
            ],
        );
        let seed = [int(1), int(1), int(1)];
        assert_eq!(framed_jacobian_rank_mod_p(&system, &seed, 17).unwrap(), 2);
        assert!(!rank_gate(&system, &seed, 17, 3).unwrap());
    }

    #[test]
    fn c6_degenerate_stratum_vs_good_seed() {
        // Oracle (sympy over GF(17)): at (1,1,1) the Jacobian
        // [[2,-1,-1],[-1,2,-1],[-1,-1,2]] has rank 2 (rows sum to zero);
        // at (1,2,3), [[2,-3,-2],[-3,4,-1],[-2,-1,6]] has rank 3
        // (det = -36 ≡ 15 mod 17, hand-expanded).
        let system = stratified_system();
        let bad_seed = [int(1), int(1), int(1)];
        let good_seed = [int(1), int(2), int(3)];
        assert_eq!(
            framed_jacobian_rank_mod_p(&system, &bad_seed, 17).unwrap(),
            2
        );
        assert!(!rank_gate(&system, &bad_seed, 17, 3).unwrap());
        assert_eq!(
            framed_jacobian_rank_mod_p(&system, &good_seed, 17).unwrap(),
            3
        );
        assert!(rank_gate(&system, &good_seed, 17, 3).unwrap());
    }

    #[test]
    fn c6_rank_gate_rejects_bad_input() {
        let system = full_rank_system();
        let seed = [int(1), int(1), int(1)];
        assert!(framed_jacobian_rank_mod_p(&system, &seed, 15).is_err()); // composite
        assert!(framed_jacobian_rank_mod_p(&system, &seed[..2], 17).is_err()); // arity
    }

    // --- C7 ------------------------------------------------------------------

    /// Recompute the Hilbert symbol of the report's quaternion class at every
    /// candidate place (the only places where it can ramify), check each
    /// symbol against the reported ramified set, and check Hilbert
    /// reciprocity: the product over all places is +1.
    fn certify_hilbert_product(report: &ConicBrauerReport) {
        let places = candidate_places_for_quaternion(&report.quaternion_a, &report.quaternion_b);
        let mut product = 1i8;
        for place in places {
            let s = hilbert_symbol(&report.quaternion_a, &report.quaternion_b, place).unwrap();
            assert_eq!(
                s == -1,
                report.ramified_places.contains(&place),
                "symbol/ramified mismatch at {place}"
            );
            product *= s;
        }
        assert_eq!(product, 1, "Hilbert reciprocity violated");
    }

    #[test]
    fn c7_isotropic_conic_x2_y2_minus_2z2() {
        // x^2 + y^2 - 2z^2 = 0 has the point (1,1,1): 1 + 1 - 2 = 0 (hand).
        // Quaternion class (-a/c, -b/c) = (1/2, 1/2); oracle (gp):
        // hilbert(1/2, 1/2, p) = +1 at p = 2, at the real place, and at every
        // odd prime — no ramified places, so a rational point exists.
        let (a, b) = (Rational::from_i64(1), Rational::from_i64(1));
        let c = Rational::from_i64(-2);

        let v = descent_conic_verdict(&a, &b, &c, true).unwrap();
        assert_eq!(v.kind, VerdictKind::Constructed);
        let report = v.value.expect("constructed verdict carries the report");
        assert!(report.has_rational_point);
        assert!(report.ramified_places.is_empty());
        certify_hilbert_product(&report);

        // without the Z_C-clearance certificate the point may sit on the bad
        // locus: honest Unresolved, never Constructed.
        let v = descent_conic_verdict(&a, &b, &c, false).unwrap();
        assert_eq!(v.kind, VerdictKind::Unresolved);
    }

    #[test]
    fn c7_anisotropic_conic_x2_y2_z2() {
        // x^2 + y^2 + z^2 = 0 over Q: quaternion class (-1,-1); oracle (gp +
        // hand via the (ε, ω) formulas): hilbert(-1,-1,2) = -1,
        // hilbert(-1,-1,oo) = -1, +1 at all odd primes. Ramified exactly at
        // {2, oo} (even count, reciprocity), so no rational point — a
        // theorem-grade local obstruction independent of bad_locus_clear.
        let one = Rational::from_i64(1);
        for clear in [true, false] {
            let v = descent_conic_verdict(&one, &one, &one, clear).unwrap();
            assert_eq!(v.kind, VerdictKind::LocallyEmpty);
            let report = v.value.expect("locally-empty verdict carries the report");
            assert!(!report.has_rational_point);
            assert_eq!(report.ramified_places.len(), 2);
            assert!(report
                .ramified_places
                .contains(&rustmath_quadraticforms::hilbert::Place::Real));
            assert!(report
                .ramified_places
                .contains(&rustmath_quadraticforms::hilbert::Place::Finite(2)));
            certify_hilbert_product(&report);
        }
    }

    #[test]
    fn c7_degenerate_conic_is_rejected() {
        let one = Rational::from_i64(1);
        let zero = Rational::from_i64(0);
        assert!(descent_conic_verdict(&one, &one, &zero, true).is_err());
    }
}
