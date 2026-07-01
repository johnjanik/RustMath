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
