//! Point counts on the normalization of a curve in `P^1 x P^1` given by a bihomogeneous form
//! over `Z`, with a certificate for the reduction mod p.
//!
//! Specification: MAGMA Handbook ch. 114 *Algebraic Curves* (digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`): 114.4.3 `IsNode`, `IsCusp`,
//! `IsOrdinarySingularity`, `Multiplicity`, `TangentCone`, `IsAnalyticallyIrreducible`, and
//! 114.8.1 `NumberOfPlacesOfDegreeOneECF`, `LPolynomial`, `ZetaFunction`; ch. 42 *Algebraic
//! Function Fields* for the same invariants of function fields. MAGMA computes these for any
//! curve by other methods. This module implements naive fibre counting, which is what a curve of
//! small genus needs (the genus 4 curve of the M23 workstream,
//! `docs/workstreams/m23_golay/HANDOFF.md` Section 3).
//!
//! This module does not use the plane-curve layer of this crate (`genus.rs`, `singularities.rs`,
//! `riemann_roch.rs`, `differentials.rs`), which the brief lists as facade or broken.
//!
//! # Method
//!
//! Let `F = sum F[i][j] X^i Y^j` have bidegree `(dx, dy)` and let `X0_p = V(F mod p)` in
//! `P^1 x P^1` over `F_p`.
//!
//! 1. Plane count ([`plane_point_count`]): `#X0_p(F_q)` is the sum over the base points of
//!    `P^1(F_q)` of the number of distinct roots of the fibre in `P^1(F_q)`, with Frobenius-orbit
//!    weighting (reference: `m23golay/curve.py`, `count_points`, in johnjanik/M23 at `d05475b`).
//! 2. Singular locus ([`singular_locus`]): every singular point lies over a root of the formal
//!    discriminant `D(Y) = disc_X F`; each irreducible factor of `D` mod p (and `Y = infinity`
//!    when `deg D` drops) is analysed in its residue field. Nodes are found by the Hessian
//!    discriminant (MAGMA `IsNode`); other points are classified exactly (`A_k` by Weierstrass
//!    preparation, ordinary m-fold points by the tangent cone).
//! 3. Certificate ([`certify`]). Theorem (WP-F design, Theorem 4.5; W. Fulton, *Algebraic curves*,
//!    delta invariants; R. Hartshorne, *Algebraic geometry*, III.9 and III.11; Q. Liu, *Algebraic
//!    geometry and arithmetic curves*, 7.5 and 10.1): let `F` be primitive and let `g` be the
//!    genus of the normalization of `V(F)` over `Q`. If `X0_p` has finitely many singular points
//!    and their delta invariants add up to `p_a - g`, `p_a = (dx - 1)(dy - 1)`, then the curve has
//!    good reduction at p and the normalization of `X0_p` is its reduction. The status is
//!    `Decided` only when every singular point is a node (the project rule: a count on a model
//!    with a non-nodal singular point is not Decided). A delta-sum below `p_a - g` is impossible
//!    for a correct genus input and is an error; above it the model (or the curve) is bad at p.
//! 4. Counts ([`normalization_point_counts`]): `N_k = #X0_p(F_q) + sum over singular points P
//!    with d_P | k of d_P (b_P(k) - 1)`, `b_P(k)` the number of `F_q`-rational branches.
//!
//! The genus `g` is a trusted input: with a genus below the true one a bad prime can be
//! certified. The certificate records it ([`Certificate::Nodal`] `genus_hypothesis`).
//!
//! Papers: \[HJLPPZ\] X. Huang, B. Jackson, K.-H. Lee, B. Poonen, R. Pries, S. Zhang, *The Mathieu
//! group M23 is a Galois group over Q*, arXiv:2608.08538v2 (the curve, 62 nodes, genus 4,
//! Prop. 3.10). G.-M. Greuel, C. Lossen, E. Shustin, *Introduction to singularities and
//! deformations*, I.2 (preparation and the `A_k` normal form).

mod certify;
mod fibres;
mod form;
mod local;
mod residue;
mod singular;

pub use certify::{
    certify, normalization_point_counts, BranchRule, Certificate, CountOptions,
    NormalizationCounts, ReductionStatus, Unresolved,
};
pub use fibres::{plane_point_count, Fibration};
pub use form::{BihomogeneousForm, FormModP};
pub use local::LocalType;
pub use singular::{singular_locus, SingularLocus, SingularPoint, YPlace};

use std::fmt;

/// Why a model is refused at a prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Degeneracy {
    /// The formal discriminant `disc_X F` vanishes identically mod p: every fibre has a
    /// multiple root ("model degenerate at this prime").
    DiscriminantVanishes,
    /// A fibre `Y = y0` (or `X = x0` in the other fibration) vanishes identically: the
    /// reduction contains a line of the ruling.
    FibreComponent { place: String },
}

/// Errors of the counting and certificate routines. None of them is a status: an `Err` means
/// that nothing was computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CountError {
    /// `p` is not prime.
    NotPrime(u64),
    /// `p = 2`: the Hessian test and completing the square need p odd.
    CharacteristicTwo,
    /// `p >= 2^16`, the bound of the word-size kernel.
    PrimeTooLarge(u64),
    /// `p^k >= 2^63` or `k > 8`, or a point count above `u64::MAX`.
    FieldTooLarge { p: u64, k: u32 },
    /// The form is zero mod p.
    ZeroForm,
    /// The model degenerates at p.
    DegenerateModel(Degeneracy),
    /// The fibre degree `dx` of the certificate is not in `2..=4` (transpose the form).
    FibreDegree(usize),
    /// A malformed coefficient array, or a bad genus input.
    Shape(String),
    /// The delta-sum is below `p_a - g`. This contradicts Theorem 4.5 for a correct genus input:
    /// the genus passed in, or the code, is wrong.
    GenusInconsistent { found: usize, expected: usize },
}

impl fmt::Display for CountError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CountError::NotPrime(p) => write!(f, "{p} is not prime"),
            CountError::CharacteristicTwo => write!(f, "characteristic 2 is not supported"),
            CountError::PrimeTooLarge(p) => write!(f, "p = {p} is not below 2^16"),
            CountError::FieldTooLarge { p, k } => {
                write!(f, "GF({p}^{k}) is too large for the word-size kernel")
            }
            CountError::ZeroForm => write!(f, "the form vanishes mod p"),
            CountError::DegenerateModel(Degeneracy::DiscriminantVanishes) => write!(
                f,
                "model degenerate at this prime: the discriminant vanishes identically mod p"
            ),
            CountError::DegenerateModel(Degeneracy::FibreComponent { place }) => write!(
                f,
                "model degenerate at this prime: the fibre at {place} vanishes identically"
            ),
            CountError::FibreDegree(d) => {
                write!(f, "fibre degree {d} is not in 2..=4 (transpose the form)")
            }
            CountError::Shape(s) => write!(f, "malformed input: {s}"),
            CountError::GenusInconsistent { found, expected } => write!(
                f,
                "delta-sum {found} is below p_a - g = {expected}: the genus input is wrong"
            ),
        }
    }
}

impl std::error::Error for CountError {}

/// Check that `p` is an odd prime below `2^16`.
pub(crate) fn check_prime(p: u64) -> Result<(), CountError> {
    if p == 2 {
        return Err(CountError::CharacteristicTwo);
    }
    if p >= 1 << 16 {
        return Err(CountError::PrimeTooLarge(p));
    }
    if !rustmath_integers::prime::is_prime_u64(p) {
        return Err(CountError::NotPrime(p));
    }
    Ok(())
}
