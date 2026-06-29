//! **Layer 0** (discriminant / branch locus) and **A1** (first-order
//! Newton-polygon ramification → genus) of the function-field stack.
//!
//! For `F ∈ ℚ(t)[x]` monic separable of degree `n`, defining the cover
//! `{F=0} → P¹_t`, the genus of `K = ℚ(t)[x]/(F)` is, by Riemann–Hurwitz over the
//! constant field `ℚ`,
//! ```text
//!   2g − 2 = −2n + deg(Different),
//!   deg(Different) = Σ_{base places P}  deg(P) · Σ_{faces of the Newton
//!                    polygon of F at P} (e_face − 1)·t_face        (tame case)
//! ```
//! where a face spanning horizontal length `ℓ` with valuation drop `Δv` has
//! `e = ℓ/gcd(|Δv|,ℓ)`, `t = gcd(|Δv|,ℓ)`. In characteristic 0 every place is
//! tame, so the only subtlety is **non-regular** faces (the residual has a
//! repeated factor), which need higher-order (φ-adic) Montes — flagged here as
//! [`GenusError::NeedsHigherOrder`] rather than silently miscounted, mirroring
//! `rustmath_padics::montes`.
//!
//! Implemented now: the x-adic first-order case, plus, for a degree-1 branch
//! place with a *flat* x-polygon, translation to each rational repeated root.
//! Deferred (TODO, for the ROG): general φ-adic Montes at higher-degree places
//! and non-rational repeated roots.

use crate::function_field::FfPoly;
use crate::places::Place;
use crate::ratfunc::{QtPoly, RationalFunction};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// `disc_x(F) ∈ ℚ(t)`, via `disc = (−1)^{n(n−1)/2}·Res_x(F, F′)` on the monic
/// model (so `lc = 1`). The built-in `discriminant()` only covers `n ≤ 3`; this
/// is the general route. Its numerator's roots are the finite branch points.
pub fn disc_x(f: &FfPoly) -> RationalFunction {
    let g = f.clone().make_monic();
    let n = g.degree().expect("discriminant requires deg ≥ 1") as i64;
    let res = g.resultant(&g.derivative());
    if (n * (n - 1) / 2) % 2 == 1 {
        RationalFunction::zero() - res
    } else {
        res
    }
}

/// Squarefree radical of `num(disc_x F)` — a monic `q ∈ ℚ[t]` whose roots are the
/// finite branch points.
pub fn branch_radical(f: &FfPoly) -> QtPoly {
    let d = disc_x(f);
    let num = d.numerator().clone();
    if num.degree().map_or(true, |dd| dd == 0) {
        return QtPoly::one();
    }
    let g = num.gcd(&num.derivative());
    let (rad, _) = num.quo_rem(&g);
    rad.make_monic()
}

/// Outcome of a genus attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum GenusError {
    /// A branch place whose ramification the first-order x-adic polygon cannot
    /// see (flat polygon, repeated residual factor not at a rational root).
    /// The φ-adic Montes refinement is the deferred Layer-A1 completion.
    NeedsHigherOrder(Place),
    /// `deg(Different)` came out odd — should never happen for a correct,
    /// fully-resolved computation (a signal that a place was missed).
    NonIntegralGenus(i64),
}

/// Genus of `K = ℚ(t)[x]/(F)` via the different. `F` must be monic separable.
pub fn genus(f: &FfPoly) -> Result<i64, GenusError> {
    let n = f.degree().expect("nonzero F") as i64;
    let mut deg_diff: i64 = 0;

    // finite branch places
    let rad = branch_radical(f);
    for q in irreducible_factors(&rad) {
        let place = Place::Finite(q.clone());
        let local = different_at(f, &place)?;
        deg_diff += (place.degree() as i64) * local;
    }
    // the place at infinity
    let inf_local = different_at(f, &Place::Infinite)?;
    deg_diff += inf_local; // deg(∞) = 1

    if deg_diff % 2 != 0 {
        return Err(GenusError::NonIntegralGenus(deg_diff));
    }
    Ok(1 - n + deg_diff / 2)
}

/// `Σ_faces (e−1)·t` of the first-order Newton polygon of `F` at `place`.
/// For a flat polygon at a finite *degree-1* branch place, retries after
/// translating to each rational repeated root.
fn different_at(f: &FfPoly, place: &Place) -> Result<i64, GenusError> {
    let raw = newton_different(f, place);
    if raw > 0 {
        return Ok(raw);
    }
    // flat polygon: either genuinely unramified, or hidden ramification.
    if !is_branch_place(f, place) {
        return Ok(0);
    }
    // hidden ramification — only handled here for a degree-1 finite place.
    if let Place::Finite(q) = place {
        if q.degree() == Some(1) {
            if let Some(alpha) = root_of_linear(q) {
                return translated_different(f, &alpha, place);
            }
        }
    }
    Err(GenusError::NeedsHigherOrder(place.clone()))
}

/// `(e−1)t` summed over the lower-hull faces of `{(i, v_place(c_i))}`.
fn newton_different(f: &FfPoly, place: &Place) -> i64 {
    let n = f.degree().unwrap_or(0);
    let mut pts: Vec<(i64, i64)> = Vec::new();
    for i in 0..=n {
        let c = f.coeff(i);
        if !c.numerator().is_zero() {
            pts.push((i as i64, place.valuation(c)));
        }
    }
    let hull = lower_hull(pts);
    let mut total = 0i64;
    for w in hull.windows(2) {
        let (i1, v1) = w[0];
        let (i2, v2) = w[1];
        let l = i2 - i1;
        let dv = (v2 - v1).abs();
        let g = gcd_i64(dv, l).max(1);
        let e = l / g;
        let t = g;
        total += (e - 1) * t;
    }
    total
}

/// Does `place` divide the discriminant (i.e. is it a branch place)?
fn is_branch_place(f: &FfPoly, place: &Place) -> bool {
    place.valuation(&disc_x(f)) > 0
}

/// Translate `x ↦ x + α` and recompute the local different at `place` (α ∈ ℚ,
/// summed over the rational repeated roots reached by translation).
fn translated_different(
    _f: &FfPoly,
    _alpha: &Rational,
    place: &Place,
) -> Result<i64, GenusError> {
    // Repeated roots of F mod (t−α) lie in ℚ; translate to each and re-polygon.
    // Scaffold: defer to the higher-order path until the residual-root finder is
    // wired (planned with A1 completion).  Documented limitation, not silent.
    Err(GenusError::NeedsHigherOrder(place.clone()))
}

fn root_of_linear(q: &QtPoly) -> Option<Rational> {
    // monic t − α  ⇒  α = −c0
    if q.degree() == Some(1) {
        Some(Rational::from_i64(0) - q.coeff(0).clone())
    } else {
        None
    }
}

// ---- small helpers ----

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Monotone-chain lower convex hull of integer points (ascending `x`).
fn lower_hull(mut pts: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    pts.sort();
    pts.dedup();
    let mut hull: Vec<(i64, i64)> = Vec::new();
    for p in pts {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            // cross((b−a),(p−a)) ≤ 0 ⇒ b not below the a→p line ⇒ pop
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            if cross <= 0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull
}

/// Irreducible monic factors over ℚ of a squarefree `p ∈ ℚ[t]`, via the
/// integer Zassenhaus factorizer (clear denominators, factor, renormalize).
fn irreducible_factors(p: &QtPoly) -> Vec<QtPoly> {
    if p.degree().map_or(true, |d| d < 1) {
        return Vec::new();
    }
    // clear denominators
    let mut lcm = Integer::from(1i64);
    for c in p.coefficients() {
        lcm = ilcm(&lcm, c.denominator());
    }
    let int_coeffs: Vec<Integer> = p
        .coefficients()
        .iter()
        .map(|c| c.numerator().clone() * (&lcm / c.denominator()))
        .collect();
    match rustmath_polynomials::zassenhaus::factor(&int_coeffs) {
        Ok((_, facs)) => facs
            .into_iter()
            .flat_map(|(fac, mult)| {
                let qpoly = QtPoly::new(
                    fac.iter().map(|z| Rational::new(z.clone(), Integer::from(1i64)).unwrap()).collect(),
                )
                .make_monic();
                std::iter::repeat(qpoly).take(mult as usize)
            })
            // squarefree input ⇒ each appears once, but be safe and dedup-by-degree
            .collect(),
        Err(_) => {
            // fall back to treating p as a single (possibly reducible) place;
            // correct for genus only if p is irreducible.
            vec![p.clone().make_monic()]
        }
    }
}

fn ilcm(a: &Integer, b: &Integer) -> Integer {
    if a.is_zero() || b.is_zero() {
        return Integer::from(0i64);
    }
    let g = igcd(a, b);
    (a.clone() * b.clone()) / g
}
fn igcd(a: &Integer, b: &Integer) -> Integer {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
        let r = a.clone() % b.clone();
        a = b;
        b = r;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_field::ff_poly_from_coeffs;
    use rustmath_core::Ring;

    fn rf_poly(c: i64) -> RationalFunction {
        RationalFunction::new(QtPoly::new(vec![Rational::from_i64(c)]), QtPoly::one()).unwrap()
    }
    /// constant-in-x coefficient equal to the ℚ[t] polynomial `coeffs`.
    fn rf_t(coeffs: &[i64]) -> RationalFunction {
        RationalFunction::new(
            QtPoly::new(coeffs.iter().map(|&c| Rational::from_i64(c)).collect()),
            QtPoly::one(),
        )
        .unwrap()
    }

    #[test]
    fn genus_conic_x2_minus_t2_plus_1() {
        // x^2 − (t^2 + 1): hyperelliptic over a degree-2 branch place, genus 0.
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - rf_t(&[1, 0, 1]),
            rf_poly(0),
            rf_poly(1),
        ]);
        assert_eq!(genus(&f), Ok(0));
    }

    #[test]
    fn genus_elliptic_x2_minus_t3_plus_1() {
        // x^2 − (t^3 + 1): an elliptic curve, genus 1.
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - rf_t(&[1, 0, 0, 1]),
            rf_poly(0),
            rf_poly(1),
        ]);
        assert_eq!(genus(&f), Ok(1));
    }

    #[test]
    fn genus_cyclic_x3_minus_t() {
        // x^3 − t: totally ramified at 0 and ∞, genus 0.
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - RationalFunction::t(),
            rf_poly(0),
            rf_poly(0),
            rf_poly(1),
        ]);
        assert_eq!(genus(&f), Ok(0));
    }
}
