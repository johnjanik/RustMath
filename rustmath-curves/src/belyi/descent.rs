//! Quadratic Galois descent (S3b, Route A) — the descent cocycle → conic reader.
//!
//! Ported from `dessin_engine/src/descent.rs` in
//! `/home/john/inverse_galois/M23/dessin_engine`, retyped onto the RustMath
//! foundation: the reference `Rat`/`QuadField`/`QuadElem` become
//! [`rustmath_rationals::Rational`] and the Wave-1 number-field layer
//! [`rustmath_numberfields::quadratic`] (`Q(√δ)`), and the Brauer/Hasse read runs
//! through the Wave-1 [`rustmath_quadraticforms::conic`] reader.
//!
//! The source `X_C` of an `L`-defined cover (`L = Q(√δ)`) is a conic over `Q`;
//! its class in `Br(Q)[2]` is the descent obstruction. The gluing
//! `g_σ ∈ PGL₂(L)` relating the `L`-coordinate to its conjugate is a cocycle;
//! lifting to `ĝ_σ ∈ GL₂(L)`, the coboundary `ĝ_σ·σ(ĝ_σ) = β·I` gives
//! `β ∈ Q^×`, and the conic is the **quaternion `(δ, β)`** — fed straight to D4.
//!
//! `g_σ` itself is recovered from three labelled ramification-point
//! correspondences ([`mobius_from_three_pairs`]).

use rustmath_core::Ring; // brings Rational::is_zero into scope
use rustmath_numberfields::quadratic::{QuadElem, QuadField};
use rustmath_quadraticforms::conic::{ConicBrauerReport, ConicError, DiagonalConicQ, Verdict};
use rustmath_rationals::Rational;

/// A projective point `[y:z] ∈ P¹_L`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P1Quad {
    pub y: QuadElem,
    pub z: QuadElem,
}

/// A `2×2` matrix over `L` (a lift of a `PGL₂(L)` element).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gl2Quad {
    pub a: QuadElem,
    pub b: QuadElem,
    pub c: QuadElem,
    pub d: QuadElem,
}

impl Gl2Quad {
    pub fn det(&self) -> QuadElem {
        self.a.mul(&self.d).sub(&self.b.mul(&self.c))
    }
    pub fn is_invertible(&self) -> bool {
        !self.det().norm().is_zero()
    }
    /// Apply `σ` entry-wise.
    pub fn conjugate(&self) -> Self {
        Self {
            a: self.a.conjugate(),
            b: self.b.conjugate(),
            c: self.c.conjugate(),
            d: self.d.conjugate(),
        }
    }
    pub fn mul(&self, r: &Self) -> Self {
        Self {
            a: self.a.mul(&r.a).add(&self.b.mul(&r.c)),
            b: self.a.mul(&r.b).add(&self.b.mul(&r.d)),
            c: self.c.mul(&r.a).add(&self.d.mul(&r.c)),
            d: self.c.mul(&r.b).add(&self.d.mul(&r.d)),
        }
    }
    pub fn apply(&self, p: &P1Quad) -> P1Quad {
        P1Quad {
            y: self.a.mul(&p.y).add(&self.b.mul(&p.z)),
            z: self.c.mul(&p.y).add(&self.d.mul(&p.z)),
        }
    }
    /// If the matrix is a rational scalar `β·I`, return `β`.
    pub fn scalar_rational(&self) -> Option<Rational> {
        if !self.b.is_zero() || !self.c.is_zero() || self.a != self.d {
            return None;
        }
        self.a.as_rational()
    }
}

/// `[y:z] = [u:v]` ⇔ `y v − z u = 0`.
fn same_point(a: &P1Quad, b: &P1Quad) -> bool {
    a.y.mul(&b.z).sub(&a.z.mul(&b.y)).is_zero()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MobiusError {
    KernelNotOneDimensional,
    SingularMap,
    VerificationFailed,
}

fn det3(m: &[[QuadElem; 3]; 3]) -> QuadElem {
    let pos = m[0][0]
        .mul(&m[1][1].mul(&m[2][2]))
        .add(&m[0][1].mul(&m[1][2].mul(&m[2][0])))
        .add(&m[0][2].mul(&m[1][0].mul(&m[2][1])));
    let neg = m[0][2]
        .mul(&m[1][1].mul(&m[2][0]))
        .add(&m[0][0].mul(&m[1][2].mul(&m[2][1])))
        .add(&m[0][1].mul(&m[1][0].mul(&m[2][2])));
    pos.sub(&neg)
}

/// One-dimensional kernel of a `3×4` matrix (rank 3) via signed `3×3` minors;
/// empty if rank `< 3`.
fn nullspace_3x4(rows: &[[QuadElem; 4]; 3]) -> Vec<QuadElem> {
    let mut v = Vec::with_capacity(4);
    for j in 0..4 {
        // delete column j -> 3x3 minor
        let m: [[QuadElem; 3]; 3] = std::array::from_fn(|r| {
            let cols: Vec<usize> = (0..4).filter(|&c| c != j).collect();
            std::array::from_fn(|k| rows[r][cols[k]].clone())
        });
        let mut det = det3(&m);
        if j % 2 == 1 {
            det = det.neg();
        }
        v.push(det);
    }
    if v.iter().all(|x| x.is_zero()) {
        Vec::new()
    } else {
        v
    }
}

/// Recover a `GL₂(L)` lift of the Möbius map sending `src[i] ↦ dst[i]`, from
/// three labelled correspondences; certified on those three points.
pub fn mobius_from_three_pairs(
    field: &QuadField,
    src: [&P1Quad; 3],
    dst: [&P1Quad; 3],
) -> Result<Gl2Quad, MobiusError> {
    // rows in [a,b,c,d]:  v(a y + b z) - u(c y + d z) = 0
    let rows: [[QuadElem; 4]; 3] = std::array::from_fn(|i| {
        let (y, z, u, v) = (&src[i].y, &src[i].z, &dst[i].y, &dst[i].z);
        [v.mul(y), v.mul(z), u.neg().mul(y), u.neg().mul(z)]
    });
    let _ = field;
    let kernel = nullspace_3x4(&rows);
    if kernel.len() != 4 {
        return Err(MobiusError::KernelNotOneDimensional);
    }
    let g = Gl2Quad {
        a: kernel[0].clone(),
        b: kernel[1].clone(),
        c: kernel[2].clone(),
        d: kernel[3].clone(),
    };
    if !g.is_invertible() {
        return Err(MobiusError::SingularMap);
    }
    for i in 0..3 {
        if !same_point(&g.apply(src[i]), dst[i]) {
            return Err(MobiusError::VerificationFailed);
        }
    }
    Ok(g)
}

/// The quaternion class `(δ, β)` of the descent.
#[derive(Debug, Clone)]
pub struct QuaternionClassQ {
    pub delta: Rational,
    pub beta: Rational,
}

impl QuaternionClassQ {
    /// The conic `δ X² + β Y² − Z² = 0`, whose D4 quaternion class is `(δ, β)`.
    pub fn to_conic(&self) -> Result<DiagonalConicQ, ConicError> {
        DiagonalConicQ::new(
            self.delta.clone(),
            self.beta.clone(),
            Rational::from_i64(-1),
        )
    }
    /// Read the descent conic through D4. `bad_locus_clear` from S4.
    pub fn read(&self, bad_locus_clear: bool) -> Verdict<ConicBrauerReport> {
        match self.to_conic() {
            Ok(c) => c
                .verdict(bad_locus_clear)
                .unwrap_or_else(|e| Verdict::unresolved(format!("conic read failed: {e}"))),
            Err(e) => Verdict::unresolved(format!("degenerate descent conic: {e}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescentError {
    NotInvertible,
    CoboundaryNotScalar,
}

/// `β` from the coboundary `ĝ_σ · σ(ĝ_σ) = β·I`, giving the quaternion `(δ, β)`.
/// `β` is well-defined in `Q^× / (Q^×)²·N_{L/Q}(L^×)`; D4's Hilbert symbols are
/// invariant under squares, so no further canonicalization is needed.
pub fn quadratic_cocycle_quaternion(g_sigma: &Gl2Quad) -> Result<QuaternionClassQ, DescentError> {
    if !g_sigma.is_invertible() {
        return Err(DescentError::NotInvertible);
    }
    let delta = g_sigma.a.field.delta.clone();
    let b = g_sigma.mul(&g_sigma.conjugate());
    let beta = b.scalar_rational().ok_or(DescentError::CoboundaryNotScalar)?;
    Ok(QuaternionClassQ { delta, beta })
}

/// Full descent: cocycle `g_σ` → quaternion `(δ,β)` → conic → D4 verdict.
pub fn descent_conic(g_sigma: &Gl2Quad, bad_locus_clear: bool) -> Verdict<ConicBrauerReport> {
    match quadratic_cocycle_quaternion(g_sigma) {
        Ok(q) => q.read(bad_locus_clear),
        Err(e) => Verdict::unresolved(format!("descent failed: {e:?}")),
    }
}

// ===========================================================================
// B5/B3 — the descent from a *solved* cover over a quadratic `L = Q(√δ)`.
// ===========================================================================
//
// When the `[2,12,5]` cover exactifies to `AlgebraicCoordinates` over a quadratic
// field `L = Q(√δ)`, the descent obstruction is read from the gluing
// `g_σ ∈ PGL₂(L)` relating `φ` to its Galois conjugate `φ^σ` (σ = the nontrivial
// automorphism of `L`):
//
//   B5 — recover `g_σ` from three labelled ramification-point correspondences of
//        `φ` vs `φ^σ` (the two order-12 points `{0, ∞}`, one order-5 point, one
//        order-2 point, in a rational-over-`L` labelling);
//   B3 — CERTIFY the gluing exactly: `φ^σ = φ ∘ g_σ` as an identity of rational
//        maps over `L` (all numerator/denominator coefficients agree), not just on
//        the three fitting points;
//   then feed `g_σ` into [`descent_conic`] for the exact conic class `(δ, β)`.

/// The three labelled ramification-point correspondences that pin `g_σ`.
///
/// `src[i]` is a ramification point of `φ`; `dst[i]` is the corresponding point of
/// `φ^σ` (the point carrying the same combinatorial label). Three suffice to fix a
/// Möbius map; [`certify_phi_sigma_over_L`] then checks the *whole* identity.
#[derive(Debug, Clone)]
pub struct SigmaCorrespondence {
    pub field: QuadField,
    pub src: [P1Quad; 3],
    pub dst: [P1Quad; 3],
}

/// **B5** — recover the descent gluing `g_σ ∈ PGL₂(L)` from three labelled
/// ramification-point correspondences of the solved cover (`φ` vs `φ^σ`).
///
/// Certified on those three points by [`mobius_from_three_pairs`]; the exact
/// whole-map check is [`certify_phi_sigma_over_L`] (B3).
pub fn g_sigma_from_solved_cover(corr: &SigmaCorrespondence) -> Result<Gl2Quad, MobiusError> {
    mobius_from_three_pairs(
        &corr.field,
        [&corr.src[0], &corr.src[1], &corr.src[2]],
        [&corr.dst[0], &corr.dst[1], &corr.dst[2]],
    )
}

/// The solved Belyi cover with coefficients over `L = Q(√δ)`.
///
/// All four factors are stored as **ascending** `L`-coefficient vectors, monic in
/// the pinned frame: `A = x⁸ + …` (length 9), `B = x⁸ + …` (length 9),
/// `R = (x−1)(x³+…)` (length 5), `S = x⁴ + …` (length 5), and `λ ∈ L`. The Belyi
/// map is `φ = A²B / (λR⁵S)`.
#[derive(Debug, Clone)]
pub struct LCover {
    pub field: QuadField,
    pub a: Vec<QuadElem>,
    pub b: Vec<QuadElem>,
    pub r: Vec<QuadElem>,
    pub s: Vec<QuadElem>,
    pub lambda: QuadElem,
}

impl LCover {
    /// Build an `L`-cover whose coefficients are *rational* (σ-fixed), embedded in
    /// `Q(√δ)`. Layout of `coeffs` (length ≥ 24): `a₀..a₇, b₀..b₇, r₀..r₂,
    /// s₀..s₃, λ` (same as the exactified tuple, `c` ignored). This is the
    /// convenience constructor for exercising the B3 machinery; the general path
    /// takes genuinely-`L` coefficients from exactification.
    pub fn from_rational_coeffs(delta: Rational, coeffs: &[Rational]) -> Self {
        assert!(coeffs.len() >= 24, "need at least the 24 solving coefficients");
        let field = QuadField::new(delta);
        let one = field.from_rat(Rational::from_i64(1));
        let e = |r: &Rational| field.from_rat(r.clone());

        let mut a: Vec<QuadElem> = coeffs[0..8].iter().map(e).collect();
        a.push(one.clone());
        let mut b: Vec<QuadElem> = coeffs[8..16].iter().map(e).collect();
        b.push(one.clone());
        let mut s: Vec<QuadElem> = coeffs[19..23].iter().map(e).collect();
        s.push(one.clone());
        // cubic = x³ + r₂x² + r₁x + r₀ ; R = (x−1)·cubic
        let cubic = vec![e(&coeffs[16]), e(&coeffs[17]), e(&coeffs[18]), one.clone()];
        let x_minus_1 = vec![field.from_rat(Rational::from_i64(-1)), one];
        let r = qpoly_mul(&field, &x_minus_1, &cubic);
        let lambda = e(&coeffs[23]);

        LCover {
            field,
            a,
            b,
            r,
            s,
            lambda,
        }
    }

    /// The numerator `N = A²B` (ascending `L`-coefficients).
    pub fn numerator(&self) -> Vec<QuadElem> {
        let a2 = qpoly_mul(&self.field, &self.a, &self.a);
        qpoly_mul(&self.field, &a2, &self.b)
    }

    /// The denominator `D = λ·R⁵·S` (ascending `L`-coefficients).
    pub fn denominator(&self) -> Vec<QuadElem> {
        let r5 = qpoly_pow(&self.field, &self.r, 5);
        let r5s = qpoly_mul(&self.field, &r5, &self.s);
        r5s.iter().map(|c| self.lambda.mul(c)).collect()
    }
}

/// **B3** — certify the gluing `φ^σ = φ ∘ g_σ` as an EXACT identity of rational
/// maps over `L`.
///
/// With `N = A²B`, `D = λR⁵S`, homogenization degree `D° = max(deg N, deg D)`, and
/// `g_σ = [[a,b],[c,d]]`, the composition satisfies
/// `φ ∘ g_σ = Ñ / D̃` where `p̃(x) = Σ_k p_k (ax+b)^k (cx+d)^{D°−k}`. The identity
/// `N^σ/D^σ = Ñ/D̃` holds iff the cross-product `N^σ·D̃ − D^σ·Ñ` vanishes
/// identically — checked here across **all** coefficients over `L`, not just the
/// three fitting points.
pub fn certify_phi_sigma_over_L(cover: &LCover, g_sigma: &Gl2Quad) -> bool {
    let field = &cover.field;
    let n = cover.numerator();
    let d = cover.denominator();
    let hom = qdeg(&n).max(qdeg(&d)).max(0) as usize;

    let n_sigma = qpoly_conj(&n);
    let d_sigma = qpoly_conj(&d);
    let n_tilde = mobius_transform(field, &n, g_sigma, hom);
    let d_tilde = mobius_transform(field, &d, g_sigma, hom);

    let lhs = qpoly_mul(field, &n_sigma, &d_tilde);
    let rhs = qpoly_mul(field, &d_sigma, &n_tilde);
    let diff = qpoly_sub(field, &lhs, &rhs);
    diff.iter().all(|c| c.is_zero())
}

// --- dense polynomial helpers over L (ascending QuadElem coefficients) ------

fn qpoly_mul(field: &QuadField, a: &[QuadElem], b: &[QuadElem]) -> Vec<QuadElem> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut c = vec![field.zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            c[i + j] = c[i + j].add(&ai.mul(bj));
        }
    }
    c
}

fn qpoly_pow(field: &QuadField, a: &[QuadElem], e: u32) -> Vec<QuadElem> {
    let mut acc = vec![field.one()];
    for _ in 0..e {
        acc = qpoly_mul(field, &acc, a);
    }
    acc
}

fn qpoly_conj(a: &[QuadElem]) -> Vec<QuadElem> {
    a.iter().map(|c| c.conjugate()).collect()
}

fn qpoly_sub(field: &QuadField, a: &[QuadElem], b: &[QuadElem]) -> Vec<QuadElem> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let ai = a.get(i).cloned().unwrap_or_else(|| field.zero());
            let bi = b.get(i).cloned().unwrap_or_else(|| field.zero());
            ai.sub(&bi)
        })
        .collect()
}

/// Degree (index of the highest nonzero coefficient); `-1` for the zero poly.
fn qdeg(a: &[QuadElem]) -> i64 {
    for i in (0..a.len()).rev() {
        if !a[i].is_zero() {
            return i as i64;
        }
    }
    -1
}

/// `p̃(x) = Σ_k p_k · (ax+b)^k · (cx+d)^{hom−k}` — the homogeneous Möbius
/// substitution of `p` under `g = [[a,b],[c,d]]` at homogenization degree `hom`.
fn mobius_transform(
    field: &QuadField,
    poly: &[QuadElem],
    g: &Gl2Quad,
    hom: usize,
) -> Vec<QuadElem> {
    let axpb = vec![g.b.clone(), g.a.clone()]; // b + a·x
    let cxpd = vec![g.d.clone(), g.c.clone()]; // d + c·x
    let mut acc = vec![field.zero(); hom + 1];
    for k in 0..=hom {
        let pk = poly.get(k).cloned().unwrap_or_else(|| field.zero());
        if pk.is_zero() {
            continue;
        }
        let hi = qpoly_pow(field, &axpb, k as u32);
        let lo = qpoly_pow(field, &cxpd, (hom - k) as u32);
        let term = qpoly_mul(field, &hi, &lo); // degree exactly hom
        for (i, t) in term.iter().enumerate() {
            acc[i] = acc[i].add(&pk.mul(t));
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_quadraticforms::conic::VerdictKind;
    use rustmath_quadraticforms::hilbert::Place;

    fn ri(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    fn pt(l: &QuadField, y: i64, z: i64) -> P1Quad {
        P1Quad {
            y: l.from_rat(ri(y)),
            z: l.from_rat(ri(z)),
        }
    }

    #[test]
    fn mobius_recovers_known_map() {
        let l = QuadField::new(ri(-1));
        // g = [[2,1],[1,1]] over Q ⊂ L
        let g = Gl2Quad {
            a: l.from_rat(ri(2)),
            b: l.from_rat(ri(1)),
            c: l.from_rat(ri(1)),
            d: l.from_rat(ri(1)),
        };
        let src = [pt(&l, 0, 1), pt(&l, 1, 0), pt(&l, 1, 1)];
        let dst = [g.apply(&src[0]), g.apply(&src[1]), g.apply(&src[2])];
        let rec = mobius_from_three_pairs(
            &l,
            [&src[0], &src[1], &src[2]],
            [&dst[0], &dst[1], &dst[2]],
        )
        .unwrap();
        // recovered map agrees with g on a fresh point
        let fresh = pt(&l, 3, 2);
        assert!(super::same_point(&rec.apply(&fresh), &g.apply(&fresh)));
    }

    #[test]
    fn mueller_descent_gives_hamilton_conic() {
        // L = Q(i), δ = -1. A cocycle g_σ = [[0,1],[-1,0]] (rational, σ-fixed) has
        // g_σ·σ(g_σ) = g_σ^2 = -I, so β = -1: quaternion (-1,-1) -> x^2+y^2+z^2.
        let l = QuadField::new(ri(-1));
        let g = Gl2Quad {
            a: l.from_rat(ri(0)),
            b: l.from_rat(ri(1)),
            c: l.from_rat(ri(-1)),
            d: l.from_rat(ri(0)),
        };
        let q = quadratic_cocycle_quaternion(&g).unwrap();
        assert_eq!(q.delta, ri(-1));
        assert_eq!(q.beta, ri(-1));

        let v = descent_conic(&g, false);
        assert_eq!(v.kind, VerdictKind::LocallyEmpty); // anisotropic (-1,-1)
        let report = v.value.unwrap();
        assert!(report.ramified_places.contains(&Place::Finite(2)));
        assert!(report.ramified_places.contains(&Place::Real));
        assert_eq!(report.ramified_places.len(), 2);
    }

    // --- B5: g_sigma from three labelled ramification-point correspondences ---
    #[test]
    fn g_sigma_recovered_from_correspondences() {
        let l = QuadField::new(ri(-1));
        // A known gluing g = [[1,1],[0,1]] (x -> x+1), applied to three labelled
        // ramification points {0, ∞, 1}.
        let g = Gl2Quad {
            a: l.from_rat(ri(1)),
            b: l.from_rat(ri(1)),
            c: l.from_rat(ri(0)),
            d: l.from_rat(ri(1)),
        };
        let src = [pt(&l, 0, 1), pt(&l, 1, 0), pt(&l, 1, 1)];
        let dst = [g.apply(&src[0]), g.apply(&src[1]), g.apply(&src[2])];
        let corr = SigmaCorrespondence {
            field: l.clone(),
            src: [src[0].clone(), src[1].clone(), src[2].clone()],
            dst,
        };
        let rec = g_sigma_from_solved_cover(&corr).unwrap();
        let fresh = pt(&l, 5, 2);
        assert!(super::same_point(&rec.apply(&fresh), &g.apply(&fresh)));
    }

    // --- B3: certify φ^σ = φ ∘ g_σ as an exact identity over L -----------------
    #[test]
    fn certify_identity_holds_for_g_identity_and_fails_otherwise() {
        // A rational-coefficient cover embedded in L: φ^σ = φ, so the identity
        // holds exactly for g_σ = I and must FAIL for a non-stabilizing g.
        let mut coeffs: Vec<Rational> = Vec::new();
        coeffs.extend([1, -2, 3, 0, -1, 2, 1, -3].iter().map(|&n| ri(n))); // a
        coeffs.extend([2, 1, -1, 3, 0, -2, 1, 1].iter().map(|&n| ri(n))); // b
        coeffs.extend([-1, 2, 1].iter().map(|&n| ri(n))); // r
        coeffs.extend([3, -2, 1, 2].iter().map(|&n| ri(n))); // s
        coeffs.push(Rational::new(3, 2).unwrap()); // lambda
        coeffs.push(ri(1)); // c

        let cover = LCover::from_rational_coeffs(ri(-1), &coeffs);
        let l = cover.field.clone();
        let id = Gl2Quad {
            a: l.from_rat(ri(1)),
            b: l.from_rat(ri(0)),
            c: l.from_rat(ri(0)),
            d: l.from_rat(ri(1)),
        };
        assert!(
            certify_phi_sigma_over_L(&cover, &id),
            "φ^σ = φ ∘ I must certify exactly for a σ-fixed cover"
        );
        // A translation g = [[1,1],[0,1]] does not stabilize φ.
        let g = Gl2Quad {
            a: l.from_rat(ri(1)),
            b: l.from_rat(ri(1)),
            c: l.from_rat(ri(0)),
            d: l.from_rat(ri(1)),
        };
        assert!(
            !certify_phi_sigma_over_L(&cover, &g),
            "a non-stabilizing gluing must not certify"
        );
    }

    #[test]
    fn certify_low_degree_composition_matches() {
        // φ = x² / 1 over Q ⊂ L: φ^σ = x². φ ∘ I = x² certifies; φ ∘ (x+1) ≠ x².
        let l = QuadField::new(ri(-1));
        let cover = LCover {
            field: l.clone(),
            a: vec![l.from_rat(ri(0)), l.from_rat(ri(1))], // A = x  (so A² = x²)
            b: vec![l.from_rat(ri(1))],                    // B = 1
            r: vec![l.from_rat(ri(1))],                    // R = 1
            s: vec![l.from_rat(ri(1))],                    // S = 1
            lambda: l.from_rat(ri(1)),                     // λ = 1  (D = 1)
        };
        let id = Gl2Quad {
            a: l.from_rat(ri(1)),
            b: l.from_rat(ri(0)),
            c: l.from_rat(ri(0)),
            d: l.from_rat(ri(1)),
        };
        assert!(certify_phi_sigma_over_L(&cover, &id));
        let g = Gl2Quad {
            a: l.from_rat(ri(1)),
            b: l.from_rat(ri(1)),
            c: l.from_rat(ri(0)),
            d: l.from_rat(ri(1)),
        };
        assert!(!certify_phi_sigma_over_L(&cover, &g));
    }

    #[test]
    fn coboundary_gives_split_conic() {
        // g_σ = β'·I with β' rational is a coboundary -> g_σ σ(g_σ) = β'^2 I,
        // β = β'^2 a square -> conic (δ, square) splits.
        let l = QuadField::new(ri(-1));
        let g = Gl2Quad {
            a: l.from_rat(ri(3)),
            b: l.zero(),
            c: l.zero(),
            d: l.from_rat(ri(3)),
        };
        let q = quadratic_cocycle_quaternion(&g).unwrap();
        assert_eq!(q.beta, ri(9)); // 3*3
        let report = q.to_conic().unwrap().brauer_report().unwrap();
        assert!(report.has_rational_point); // (δ, 9) splits since 9 is a square
    }
}
