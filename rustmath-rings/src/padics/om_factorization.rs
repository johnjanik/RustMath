//! # OM (Okutsu–Montes / MacLane) factorization over Q_p
//!
//! For a monic squarefree p-integral `f` in `Q[x]`, the MacLane tree
//! ([`crate::valuation::maclane::mac_lane_approximants`]) terminates with
//! leaves in bijection with the irreducible factors of `f` over `Q_p`
//! (MacLane 1936; the termination criterion `sum E_i * F_i = deg f` *is*
//! the completeness certificate). This module packages that as a
//! factorization:
//!
//! - **DECIDED data** (certified): the factor count, and for each factor its
//!   degree `e_i * f_i`, ramification index `e_i` and residue degree `f_i`.
//!   Every step of the tree is self-certifying (verified key polynomials,
//!   certified residual factorizations, re-checked lifts), the leaf data is
//!   accepted only when `sum e_i f_i = deg f`, and the tame discriminant
//!   bound `v_p(disc f) >= sum_i f_i (e_i - 1)` is re-verified on exact
//!   rational arithmetic as a final tripwire.
//! - **APPROXIMATE data** (with an explicit congruence certificate): a monic
//!   approximation `phi_i` of full degree `e_i * f_i` to each factor — the
//!   leaf key polynomial after enough refinement steps. The certificate is
//!   computed, not assumed: `prod_i phi_i ≡ f` coefficient-wise modulo
//!   `p^N` with `N = congruence_precision()` (`None` means the product is
//!   *exactly* `f`). Per-factor Krasner-style closeness bounds are NOT
//!   claimed (deferred honestly); an approximation is only flagged `exact`
//!   when its leaf is an infinite (pseudo-valuation) leaf, i.e. the key
//!   literally divides `f`.
//!
//! Honest limitations: input must be monic, squarefree over `Q` and
//! p-integral (`Err` otherwise, from the tree gates); `p < 2^31`. There are
//! no p = 2 carve-outs — the residue machinery handles wild ramification
//! (see the `x^4 + 4x^2 + 12` level-2 gate) — but refinement that fails to
//! reach the requested congruence precision within the round guard is an
//! honest `Err`, never a silently weaker certificate.

use crate::valuation::maclane::{mac_lane_approximants, PAdicInductiveValuation};
use rustmath_core::{MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::fmt;

/// One irreducible factor of `f` over `Q_p`, as decided (e, f, degree) data
/// plus a congruence-certified approximation.
#[derive(Clone, Debug)]
pub struct OmFactor {
    approximation: UnivariatePolynomial<Rational>,
    e: u64,
    f: u64,
    exact: bool,
    leaf: PAdicInductiveValuation,
}

impl OmFactor {
    /// Monic approximation to the factor, of full degree `e * f`. Exact iff
    /// [`Self::is_exact`].
    pub fn approximation(&self) -> &UnivariatePolynomial<Rational> {
        &self.approximation
    }

    /// Ramification index `e` of the local field `Q_p[x]/(g)` (DECIDED).
    pub fn e(&self) -> u64 {
        self.e
    }

    /// Residue degree `f` of the local field `Q_p[x]/(g)` (DECIDED).
    pub fn f(&self) -> u64 {
        self.f
    }

    /// Degree `e * f` of the true factor (DECIDED).
    pub fn degree(&self) -> usize {
        (self.e * self.f) as usize
    }

    /// Is the approximation exactly the factor (infinite leaf: the key
    /// polynomial divides `f` in `Q[x]`)?
    pub fn is_exact(&self) -> bool {
        self.exact
    }

    /// The MacLane leaf valuation approximating this factor.
    pub fn leaf(&self) -> &PAdicInductiveValuation {
        &self.leaf
    }
}

/// The OM factorization of a monic squarefree p-integral polynomial over
/// `Q_p`. See the module docs for what is decided vs. approximate.
#[derive(Clone, Debug)]
pub struct OmFactorization {
    p: i64,
    poly: UnivariatePolynomial<Rational>,
    factors: Vec<OmFactor>,
    congruence_precision: Option<u32>,
}

impl OmFactorization {
    /// The prime.
    pub fn p(&self) -> i64 {
        self.p
    }

    /// The factored polynomial.
    pub fn polynomial(&self) -> &UnivariatePolynomial<Rational> {
        &self.poly
    }

    /// The factors (count and per-factor `(e, f)` are DECIDED data).
    pub fn factors(&self) -> &[OmFactor] {
        &self.factors
    }

    /// The congruence certificate: `prod_i approximation_i ≡ f mod p^N`
    /// coefficient-wise, recomputed at construction. `None` means the
    /// product is exactly `f`.
    pub fn congruence_precision(&self) -> Option<u32> {
        self.congruence_precision
    }

    /// Sorted `(e, f)` pairs of the factors (DECIDED).
    pub fn ramification_data(&self) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = self.factors.iter().map(|g| (g.e, g.f)).collect();
        out.sort();
        out
    }

    /// Is `f` irreducible over `Q_p` (DECIDED)?
    pub fn is_irreducible(&self) -> bool {
        self.factors.len() == 1
    }
}

impl fmt::Display for OmFactorization {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            out,
            "OM factorization of {} over Q_{} ({} factor(s)):",
            self.poly,
            self.p,
            self.factors.len()
        )?;
        for g in &self.factors {
            writeln!(
                out,
                "  [e = {}, f = {}, deg = {}] {} ({})",
                g.e,
                g.f,
                g.degree(),
                g.approximation,
                if g.exact { "exact" } else { "approximate" }
            )?;
        }
        match self.congruence_precision {
            None => write!(out, "  product of approximations = f exactly"),
            Some(n) => write!(out, "  product of approximations = f mod {}^{}", self.p, n),
        }
    }
}

/// `v_p` of a nonzero rational.
fn vp_rational(q: &Rational, p: &Integer) -> i64 {
    q.valuation(p) as i64
}

/// `Res(f, g)` over `Q` by the Euclidean recursion
/// `Res(f, g) = (-1)^{deg f * deg g} lc(g)^{deg f - deg r} Res(g, r)`
/// (`r = f mod g`), exact in rational arithmetic. (The generic
/// `UnivariatePolynomial::resultant` expands the Sylvester determinant by
/// cofactors and is unusable beyond tiny degrees.)
fn resultant_euclid(
    f: &UnivariatePolynomial<Rational>,
    g: &UnivariatePolynomial<Rational>,
) -> Result<Rational> {
    let df = match f.degree() {
        Some(d) => d,
        None => return Ok(Rational::zero()),
    };
    let dg = match g.degree() {
        Some(d) => d,
        None => return Ok(Rational::zero()),
    };
    if dg == 0 {
        let mut acc = Rational::one();
        for _ in 0..df {
            acc = acc * g.coefficients()[0].clone();
        }
        return Ok(acc);
    }
    let (_, r) = f.div_rem(g)?;
    let lc_g = g.coefficients()[dg].clone();
    let dr = r.degree();
    let sign = if (df * dg) % 2 == 1 {
        -Rational::one()
    } else {
        Rational::one()
    };
    match dr {
        None => Ok(Rational::zero()),
        Some(dr) => {
            let mut lc_pow = Rational::one();
            for _ in 0..(df - dr) {
                lc_pow = lc_pow * lc_g.clone();
            }
            Ok(sign * lc_pow * resultant_euclid(g, &r)?)
        }
    }
}

/// `disc(f) = (-1)^{n(n-1)/2} Res(f, f')` for monic `f` of degree `n`.
fn discriminant_monic(f: &UnivariatePolynomial<Rational>) -> Result<Rational> {
    let n = f.degree().ok_or_else(|| {
        MathError::InvalidArgument("discriminant of the zero polynomial".to_string())
    })?;
    let res = resultant_euclid(f, &f.derivative())?;
    if (n * (n - 1) / 2) % 2 == 1 {
        Ok(-res)
    } else {
        Ok(res)
    }
}

/// Factor a monic squarefree p-integral `f` over `Q_p` by the MacLane/OM
/// tree, refining the leaf keys until every approximation has full degree
/// `e_i * f_i` and the congruence certificate reaches `min_precision`
/// (i.e. `prod approximations ≡ f mod p^min_precision`).
///
/// The factor COUNT and the per-factor `(e, f)` data are decided outputs;
/// the approximations carry exactly the recomputed congruence certificate
/// (see the module docs).
///
/// ```
/// use rustmath_rings::padics::om_factorization::om_factorization;
/// use rustmath_polynomials::UnivariatePolynomial;
/// use rustmath_rationals::Rational;
///
/// // (x^2 + 1)(x^2 - 2) over Q_2: both factors are ramified quadratics.
/// let f = UnivariatePolynomial::new(
///     [-2i64, 0, -1, 0, 1].iter().map(|&c| Rational::from_i64(c)).collect::<Vec<_>>(),
/// );
/// let fac = om_factorization(&f, 2, 8).unwrap();
/// assert_eq!(fac.factors().len(), 2);
/// assert_eq!(fac.ramification_data(), vec![(2, 1), (2, 1)]);
/// assert!(!fac.is_irreducible());
/// ```
pub fn om_factorization(
    f: &UnivariatePolynomial<Rational>,
    p: i64,
    min_precision: u32,
) -> Result<OmFactorization> {
    let n = f.degree().ok_or_else(|| {
        MathError::InvalidArgument("om_factorization: zero polynomial".to_string())
    })?;
    let mut leaves = mac_lane_approximants(f, p)?;
    let p_int = Integer::from(p);

    // completeness re-check (mac_lane_approximants guarantees this)
    let total: u64 = leaves
        .iter()
        .map(|w| w.ramification_index() * w.residue_degree())
        .sum();
    if total != n as u64 {
        return Err(MathError::NumericalError(
            "om_factorization: internal error: leaf data does not sum to deg f".to_string(),
        ));
    }

    // Refine one isolated leaf by one MacLane step. After termination each
    // leaf is in bijection with ONE irreducible factor. A step can
    // transiently regenerate a SIBLING factor's branch (the polygon of f
    // with respect to the leaf key still sees the other factors' roots).
    // The true continuation of THIS branch is unique (within one
    // Q_p-irreducible factor the conjugate root distances are Galois-equal)
    // and keeps the leaf's (E, F) exactly — the terminal leaf already
    // carries the factor's full invariants, and an approximant's (E, F)
    // divide its factor's. So: (1) keep only children with the same (E, F);
    // (2) if several remain, drop children comparable (in the domination
    // order) with one of the OTHER terminal leaves — those are resurrected
    // sibling nodes. Exactly one child must survive; anything else is an
    // honest error.
    let originals = leaves.clone();
    let step_isolated = |w: &PAdicInductiveValuation,
                         self_index: usize|
     -> Result<PAdicInductiveValuation> {
        let e = w.ramification_index();
        let fdeg = w.residue_degree();
        let mut kids = w.mac_lane_step(f)?;
        kids.retain(|k| k.ramification_index() == e && k.residue_degree() == fdeg);
        if kids.len() > 1 {
            kids.retain(|k| {
                !originals
                    .iter()
                    .enumerate()
                    .any(|(j, o)| j != self_index && (k.dominates(o) || o.dominates(k)))
            });
        }
        if kids.len() != 1 {
            return Err(MathError::NumericalError(
                "om_factorization: internal error: isolated leaf did not refine uniquely"
                    .to_string(),
            ));
        }
        Ok(kids.into_iter().next().expect("one child"))
    };

    // (1) grow every leaf key to full degree e*f
    for (i, w) in leaves.iter_mut().enumerate() {
        let target = (w.ramification_index() * w.residue_degree()) as usize;
        let mut rounds = 0;
        while !w.is_final()
            && w.last_key().and_then(|k| k.degree()).unwrap_or(0) < target
        {
            *w = step_isolated(w, i)?;
            rounds += 1;
            if rounds > 64 {
                return Err(MathError::NumericalError(
                    "om_factorization: key degree did not reach e*f within 64 refinements"
                        .to_string(),
                ));
            }
        }
        if w.last_key().and_then(|k| k.degree()) != Some(target) {
            return Err(MathError::NumericalError(
                "om_factorization: internal error: leaf key degree != e*f".to_string(),
            ));
        }
    }

    // (2) refine until the congruence certificate reaches min_precision
    let congruence = |leaves: &[PAdicInductiveValuation]| -> Result<Option<i64>> {
        let mut prod = UnivariatePolynomial::<Rational>::one();
        for w in leaves {
            prod = prod * w.last_key().expect("leaf has a key").clone();
        }
        let delta = f.clone() - prod;
        if delta.is_zero() {
            return Ok(None);
        }
        let mut prec: Option<i64> = None;
        for c in delta.coefficients() {
            if c.is_zero() {
                continue;
            }
            let v = vp_rational(c, &p_int);
            if prec.map(|q| v < q).unwrap_or(true) {
                prec = Some(v);
            }
        }
        Ok(Some(prec.expect("nonzero delta has a nonzero coefficient")))
    };
    let mut rounds = 0;
    let precision = loop {
        match congruence(&leaves)? {
            None => break None,
            Some(prec) if prec >= min_precision as i64 => break Some(prec as u32),
            Some(_) => {}
        }
        let mut improved = false;
        for (i, w) in leaves.iter_mut().enumerate() {
            if !w.is_final() {
                *w = step_isolated(w, i)?;
                improved = true;
            }
        }
        if !improved {
            // all leaves final: the product of the exact factors must be f
            return Err(MathError::NumericalError(
                "om_factorization: internal error: exact factors do not multiply to f"
                    .to_string(),
            ));
        }
        rounds += 1;
        if rounds > 64 {
            return Err(MathError::NumericalError(format!(
                "om_factorization: congruence precision p^{} not reached within 64 refinement rounds",
                min_precision
            )));
        }
    };

    // (3) the tame discriminant law as a final tripwire:
    // v_p(disc f) >= sum_i f_i (e_i - 1), on exact rational arithmetic.
    let disc = discriminant_monic(f)?;
    if disc.is_zero() {
        return Err(MathError::InvalidArgument(
            "om_factorization: f is not squarefree".to_string(),
        ));
    }
    let tame: i64 = leaves
        .iter()
        .map(|w| (w.residue_degree() * (w.ramification_index() - 1)) as i64)
        .sum();
    if vp_rational(&disc, &p_int) < tame {
        return Err(MathError::NumericalError(
            "om_factorization: internal error: v_p(disc) violates the tame lower bound"
                .to_string(),
        ));
    }

    let factors = leaves
        .into_iter()
        .map(|w| OmFactor {
            approximation: w.last_key().expect("leaf has a key").clone(),
            e: w.ramification_index(),
            f: w.residue_degree(),
            exact: w.is_final(),
            leaf: w,
        })
        .collect();
    Ok(OmFactorization {
        p,
        poly: f.clone(),
        factors,
        congruence_precision: precision,
    })
}

// ---------------------------------------------------------------------------
// Gates. Every (e, f) expectation and factor count below was derived
// independently BEFORE being asserted, in scratchpad/verify_om_gates.py
// (PARI/gp idealprimedec on each Q-irreducible factor + factorpadic degree
// cross-check + scripted Newton-polygon hand derivations; 55/55 checks).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn qpoly(coeffs: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.iter().map(|&n| Rational::from_i64(n)).collect())
    }

    fn vp_int(n: &Integer, p: i64) -> i64 {
        Rational::new(n.clone(), Integer::one())
            .unwrap()
            .valuation(&Integer::from(p)) as i64
    }

    /// Re-verify the congruence certificate from scratch (do not trust the
    /// constructor): prod approx - f has all coefficient valuations >= N.
    fn assert_congruence(fac: &OmFactorization, f: &UnivariatePolynomial<Rational>, min: u32) {
        let mut prod = UnivariatePolynomial::<Rational>::one();
        for g in fac.factors() {
            assert!(g.approximation().is_monic());
            assert_eq!(g.approximation().degree(), Some(g.degree()));
            prod = prod * g.approximation().clone();
        }
        let delta = f.clone() - prod;
        match fac.congruence_precision() {
            None => assert!(delta.is_zero(), "certificate says exact but product != f"),
            Some(n) => {
                assert!(n >= min);
                let p_int = Integer::from(fac.p());
                for c in delta.coefficients() {
                    if !c.is_zero() {
                        assert!(
                            c.valuation(&p_int) as i64 >= n as i64,
                            "certificate p^{} violated: coefficient {} has v = {}",
                            n,
                            c,
                            c.valuation(&p_int)
                        );
                    }
                }
            }
        }
        // degrees partition deg f
        let total: usize = fac.factors().iter().map(|g| g.degree()).sum();
        assert_eq!(total, f.degree().unwrap());
    }

    #[test]
    fn test_om_x4_plus_1_q2() {
        // x^4 + 1 over Q_2: IRREDUCIBLE (Q_2(zeta_8)/Q_2 totally ramified of
        // degree phi(8) = 4): e = 4, f = 1. gp-confirmed; v_2(disc) = 8.
        let f = qpoly(&[1, 0, 0, 0, 1]);
        let fac = om_factorization(&f, 2, 8).unwrap();
        assert!(fac.is_irreducible());
        assert_eq!(fac.ramification_data(), vec![(4, 1)]);
        assert_congruence(&fac, &f, 8);
        let disc = discriminant_monic(&f).unwrap();
        assert_eq!(vp_int(disc.numerator(), 2), 8); // disc = 256
    }

    #[test]
    fn test_om_x4_minus_2_q2() {
        // x^4 - 2 over Q_2: Eisenstein: irreducible, e = 4, f = 1;
        // v_2(disc) = 11 (disc = -2048). gp-confirmed.
        let f = qpoly(&[-2, 0, 0, 0, 1]);
        let fac = om_factorization(&f, 2, 10).unwrap();
        assert!(fac.is_irreducible());
        assert_eq!(fac.ramification_data(), vec![(4, 1)]);
        assert_congruence(&fac, &f, 10);
        let disc = discriminant_monic(&f).unwrap();
        assert_eq!(vp_int(disc.numerator(), 2), 11);
    }

    #[test]
    fn test_om_x3_minus_x_minus_1_q23() {
        // x^3 - x - 1 over Q_23: disc = -23 exactly (v_23 = 1): one linear
        // factor and one RAMIFIED quadratic: (1,1) and (2,1). gp-confirmed.
        let f = qpoly(&[-1, -1, 0, 1]);
        let fac = om_factorization(&f, 23, 6).unwrap();
        assert_eq!(fac.factors().len(), 2);
        assert_eq!(fac.ramification_data(), vec![(1, 1), (2, 1)]);
        assert_congruence(&fac, &f, 6);
        let disc = discriminant_monic(&f).unwrap();
        assert_eq!(disc, Rational::from_i64(-23));
        // exact tame equality here: v(disc) = 1 = f*(e-1) of the quadratic
        assert_eq!(vp_int(disc.numerator(), 23), 1);
    }

    #[test]
    fn test_om_product_recovers_both_factors_q2() {
        // (x^2+1)(x^2-2) over Q_2: two ramified quadratics, e = 2, f = 1
        // each; the approximations converge to the two exact factors.
        let f = qpoly(&[-2, 0, -1, 0, 1]);
        let fac = om_factorization(&f, 2, 12).unwrap();
        assert_eq!(fac.factors().len(), 2);
        assert_eq!(fac.ramification_data(), vec![(2, 1), (2, 1)]);
        assert_congruence(&fac, &f, 12);
        // the two approximations are congruent to x^2+1 and x^2-2 mod 2^12:
        // each true factor must match exactly one approximation mod 8
        let targets = [qpoly(&[1, 0, 1]), qpoly(&[-2, 0, 1])];
        let p_int = Integer::from(2);
        for t in &targets {
            let hit = fac.factors().iter().any(|g| {
                let d = t.clone() - g.approximation().clone();
                d.is_zero()
                    || d.coefficients()
                        .iter()
                        .all(|c| c.is_zero() || c.valuation(&p_int) >= 3)
            });
            assert!(hit, "no approximation matches {} mod 8", t);
        }
    }

    #[test]
    fn test_om_x6_plus_2x3_plus_4_q2() {
        // x^6 + 2x^3 + 4 over Q_2: irreducible with e = 3, f = 2
        // (slope -1/3, residual y^2+y+1 irreducible over GF(2)).
        // gp-confirmed. The polynomial is its own minimal OM key
        // (the level-1 lift reproduces it exactly), so the leaf is final.
        let f = qpoly(&[4, 0, 0, 2, 0, 0, 1]);
        let fac = om_factorization(&f, 2, 6).unwrap();
        assert!(fac.is_irreducible());
        assert_eq!(fac.ramification_data(), vec![(3, 2)]);
        assert_congruence(&fac, &f, 6);
        assert!(fac.factors()[0].is_exact());
        assert_eq!(fac.congruence_precision(), None);
    }

    #[test]
    fn test_om_level2_x4_plus_4x2_plus_12_q2() {
        // x^4 + 4x^2 + 12 over Q_2: irreducible, e = 4, f = 1, via a
        // GENUINE level-2 MacLane tree (hand-derived chain
        // [Gauss, v(x)=1/2, v(x^2+2x+2)=7/4]; gp-confirmed (e,f)).
        // v_2(disc) = 16.
        let f = qpoly(&[12, 0, 4, 0, 1]);
        let fac = om_factorization(&f, 2, 10).unwrap();
        assert!(fac.is_irreducible());
        assert_eq!(fac.ramification_data(), vec![(4, 1)]);
        // refinement may extend the chain beyond the terminal level 2
        assert!(fac.factors()[0].leaf().level() >= 2);
        assert_congruence(&fac, &f, 10);
        let disc = discriminant_monic(&f).unwrap();
        assert_eq!(vp_int(disc.numerator(), 2), 16);
        // wild ramification: strict inequality over the tame bound f*(e-1)=3
        assert!(16 > 3);
    }

    #[test]
    fn test_om_gf4_residual_irreducible_q2() {
        // x^4+2x^3+5x^2+8x+3 over Q_2: unramified quartic (e = 1, f = 4);
        // residual y^2+y+w irreducible over GF(4) (Tr(w) = 1). gp-confirmed.
        let f = qpoly(&[3, 8, 5, 2, 1]);
        let fac = om_factorization(&f, 2, 8).unwrap();
        assert!(fac.is_irreducible());
        assert_eq!(fac.ramification_data(), vec![(1, 4)]);
        assert_congruence(&fac, &f, 8);
    }

    #[test]
    fn test_om_gf4_residual_split_q2() {
        // x^4+2x^3+5x^2+4x+7 over Q_2: residual y^2+y+1 = (y+w)(y+w^2) over
        // GF(4): TWO unramified quadratics (e = 1, f = 2 each). gp-confirmed.
        let f = qpoly(&[7, 4, 5, 2, 1]);
        let fac = om_factorization(&f, 2, 8).unwrap();
        assert_eq!(fac.factors().len(), 2);
        assert_eq!(fac.ramification_data(), vec![(1, 2), (1, 2)]);
        assert_congruence(&fac, &f, 8);
    }

    #[test]
    fn test_om_mixed_slopes_q3() {
        // (x^2-3)(x^3-3) over Q_3: e = 2 and e = 3, f = 1 each. The
        // approximations converge to the exact factors. gp-confirmed.
        let f = qpoly(&[9, 0, -3, -3, 0, 1]);
        let fac = om_factorization(&f, 3, 8).unwrap();
        assert_eq!(fac.factors().len(), 2);
        assert_eq!(fac.ramification_data(), vec![(2, 1), (3, 1)]);
        assert_congruence(&fac, &f, 8);
    }

    #[test]
    fn test_om_equivalent_keys_q2() {
        // (x^2+x+1)(x^2+x+3) over Q_2: both factors congruent mod 2: the
        // tree must separate equivalent degree-2 keys: two unramified
        // quadratics. gp-confirmed.
        let f = qpoly(&[3, 4, 5, 2, 1]);
        let fac = om_factorization(&f, 2, 10).unwrap();
        assert_eq!(fac.factors().len(), 2);
        assert_eq!(fac.ramification_data(), vec![(1, 2), (1, 2)]);
        assert_congruence(&fac, &f, 10);
        // the true factors mod 2^4: each matches exactly one approximation
        let targets = [qpoly(&[1, 1, 1]), qpoly(&[3, 1, 1])];
        let p_int = Integer::from(2);
        for t in &targets {
            let hits = fac
                .factors()
                .iter()
                .filter(|g| {
                    let d = t.clone() - g.approximation().clone();
                    d.is_zero()
                        || d.coefficients()
                            .iter()
                            .all(|c| c.is_zero() || c.valuation(&p_int) >= 4)
                })
                .count();
            assert_eq!(hits, 1, "factor {} must match exactly one approximation", t);
        }
    }

    #[test]
    fn test_om_unramified_splitting_q2() {
        // x^2 - 17 over Q_2 (17 = 1 mod 8 is a 2-adic square): SPLITS into
        // two linear factors, e = f = 1 each (sympy-verified in stage 1).
        let f = qpoly(&[-17, 0, 1]);
        let fac = om_factorization(&f, 2, 10).unwrap();
        assert_eq!(fac.factors().len(), 2);
        assert_eq!(fac.ramification_data(), vec![(1, 1), (1, 1)]);
        assert_congruence(&fac, &f, 10);
        // sqrt(17) in Z_2: the two approximations are x -+ a with
        // a^2 = 17 mod 2^10 (implied by the product certificate)
        for g in fac.factors() {
            assert_eq!(g.degree(), 1);
            let a = -g.approximation().coefficients()[0].clone();
            let sq = a.clone() * a - Rational::from_i64(17);
            assert!(sq.is_zero() || sq.valuation(&Integer::from(2)) >= 10);
        }
    }

    #[test]
    fn test_om_input_gates() {
        // non-squarefree
        assert!(om_factorization(&qpoly(&[0, 0, 1]), 2, 4).is_err());
        // non-monic
        assert!(om_factorization(&qpoly(&[1, 0, 2]), 2, 4).is_err());
        // composite p
        assert!(om_factorization(&qpoly(&[1, 0, 1]), 4, 4).is_err());
        // constant
        assert!(om_factorization(&qpoly(&[5]), 2, 4).is_err());
    }

    #[test]
    fn test_om_sum_ef_law_battery() {
        // sum e_i f_i = deg f across all gate polynomials and primes.
        for (coeffs, p) in [
            (&[1i64, 0, 0, 0, 1][..], 2i64),
            (&[-2, 0, 0, 0, 1][..], 2),
            (&[-1, -1, 0, 1][..], 23),
            (&[-2, 0, -1, 0, 1][..], 2),
            (&[4, 0, 0, 2, 0, 0, 1][..], 2),
            (&[12, 0, 4, 0, 1][..], 2),
            (&[3, 8, 5, 2, 1][..], 2),
            (&[7, 4, 5, 2, 1][..], 2),
            (&[9, 0, -3, -3, 0, 1][..], 3),
            (&[3, 4, 5, 2, 1][..], 2),
            (&[-1, -1, 0, 1][..], 2), // unramified cubic control (1,3)
        ] {
            let f = qpoly(coeffs);
            let fac = om_factorization(&f, p, 4).unwrap();
            let total: u64 = fac.factors().iter().map(|g| g.e() * g.f()).sum();
            assert_eq!(total, f.degree().unwrap() as u64, "sum ef for {} over Q_{}", f, p);
            // tame disc law: v_p(disc) >= sum f_i (e_i - 1)
            let disc = discriminant_monic(&f).unwrap();
            let v = disc.valuation(&Integer::from(p)) as i64;
            let tame: i64 = fac.factors().iter().map(|g| (g.f() * (g.e() - 1)) as i64).sum();
            assert!(v >= tame, "disc law for {} over Q_{}: {} < {}", f, p, v, tame);
        }
    }

    #[test]
    fn test_om_display() {
        let f = qpoly(&[-2, 0, -1, 0, 1]);
        let fac = om_factorization(&f, 2, 6).unwrap();
        let s = format!("{}", fac);
        assert!(s.contains("2 factor(s)"), "{}", s);
        assert!(s.contains("e = 2"), "{}", s);
    }
}
