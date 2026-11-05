//! Divisors of K(x): the free abelian group on places.
//!
//! Provides principal divisors div(f) (with the exact-degree gate
//! deg div(f) = 0), effective/positive parts, and — because K(x) has genus 0
//! — exact Riemann-Roch spaces: for deg D >= 0,
//! dim L(D) = deg D + 1 with the explicit basis {x^i / g : 0 <= i <= deg D}
//! where g = prod p^{n_p} over the finite places of D; for deg D < 0,
//! L(D) = 0. (For D = n[inf] the basis is {1, x, ..., x^n}.)

use super::element::RationalFunction;
use super::factor::FactorableConstantField;
use super::place::Place;
use rustmath_core::{EuclideanDomain, Field, MathError, Result, Ring};
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::{Add, Neg, Sub};

/// An element of Div(K(x)): a finite formal Z-linear combination of places.
///
/// Invariant: multiplicities are nonzero and places are pairwise distinct.
#[derive(Clone, Debug)]
pub struct Divisor<K: Field + EuclideanDomain> {
    coeffs: Vec<(Place<K>, i64)>,
}

impl<K: Field + EuclideanDomain> Divisor<K> {
    /// The zero divisor.
    pub fn zero() -> Self {
        Divisor { coeffs: vec![] }
    }

    /// Build a divisor from (place, multiplicity) pairs (duplicates merged,
    /// zeros dropped).
    pub fn from_places(pairs: Vec<(Place<K>, i64)>) -> Self {
        let mut d = Divisor::zero();
        for (pl, m) in pairs {
            d.add_place(pl, m);
        }
        d
    }

    /// Add m * [place] to this divisor.
    pub fn add_place(&mut self, place: Place<K>, m: i64) {
        if m == 0 {
            return;
        }
        for (p, c) in self.coeffs.iter_mut() {
            if *p == place {
                *c += m;
                if *c == 0 {
                    self.coeffs.retain(|(_, c)| *c != 0);
                }
                return;
            }
        }
        self.coeffs.push((place, m));
    }

    /// The multiplicity n_P of a place in this divisor (0 if absent).
    pub fn multiplicity(&self, place: &Place<K>) -> i64 {
        self.coeffs
            .iter()
            .find(|(p, _)| p == place)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    }

    /// The support: places with nonzero multiplicity.
    pub fn support(&self) -> Vec<&Place<K>> {
        self.coeffs.iter().map(|(p, _)| p).collect()
    }

    /// All (place, multiplicity) pairs.
    pub fn coefficients(&self) -> &[(Place<K>, i64)] {
        &self.coeffs
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// deg D = sum n_P * deg P.
    pub fn degree(&self) -> i64 {
        self.coeffs
            .iter()
            .map(|(p, c)| c * p.degree() as i64)
            .sum()
    }

    /// D >= 0: every multiplicity non-negative.
    pub fn is_effective(&self) -> bool {
        self.coeffs.iter().all(|(_, c)| *c >= 0)
    }

    /// Split into effective parts: D = pos - neg with pos, neg >= 0 and
    /// disjoint supports. For D = div(f) these are the zero and pole divisors.
    pub fn effective_parts(&self) -> (Self, Self) {
        let mut pos = Divisor::zero();
        let mut neg = Divisor::zero();
        for (p, c) in &self.coeffs {
            if *c > 0 {
                pos.add_place(p.clone(), *c);
            } else {
                neg.add_place(p.clone(), -*c);
            }
        }
        (pos, neg)
    }

    /// The principal divisor div(f) = sum_P v_P(f) [P], for nonzero f.
    ///
    /// Both numerator and denominator are factored (certified) and the
    /// infinite place included, so deg div(f) = 0 holds exactly; this is
    /// asserted here as a self-check and any violation is an `Err`.
    pub fn principal(f: &RationalFunction<K>) -> Result<Self>
    where
        K: FactorableConstantField,
    {
        if f.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let mut d = Divisor::zero();
        let (_, num_factors) = K::factor_poly(f.numerator())?;
        for (p, e) in num_factors {
            d.add_place(Place::finite_unchecked(p), e as i64);
        }
        let (_, den_factors) = K::factor_poly(f.denominator())?;
        for (p, e) in den_factors {
            d.add_place(Place::finite_unchecked(p), -(e as i64));
        }
        let v_inf = f.denominator().degree().unwrap_or(0) as i64
            - f.numerator().degree().unwrap_or(0) as i64;
        d.add_place(Place::infinite(), v_inf);
        if d.degree() != 0 {
            return Err(MathError::NumericalError(format!(
                "principal divisor degree check failed: deg div(f) = {} != 0",
                d.degree()
            )));
        }
        Ok(d)
    }

    /// The exact Riemann-Roch space basis of L(D) = {f : div(f) + D >= 0} on
    /// K(x) (genus 0).
    ///
    /// With D = sum n_p [p] + n_inf [inf] and g = prod p^{n_p}, one has
    /// L(D) = { h/g : h in K[x], deg h <= deg D }, so the basis is
    /// { x^i / g : 0 <= i <= deg D } (empty when deg D < 0, by
    /// deg div(f) = 0). Returns `deg D + 1` elements for deg D >= 0.
    pub fn riemann_roch_basis(&self) -> Vec<RationalFunction<K>> {
        let deg = self.degree();
        if deg < 0 {
            return vec![];
        }
        // g = prod p^{n_p}: numerator collects positive powers, denominator
        // negative ones. Distinct monic irreducibles => already coprime and
        // monic; x^i * (1/g) then normalizes internally.
        let mut g_num = UnivariatePolynomial::<K>::one();
        let mut g_den = UnivariatePolynomial::<K>::one();
        for (pl, c) in &self.coeffs {
            if let Some(p) = pl.polynomial() {
                let e = c.unsigned_abs();
                for _ in 0..e {
                    if *c > 0 {
                        g_num = g_num * p.clone();
                    } else {
                        g_den = g_den * p.clone();
                    }
                }
            }
        }
        // basis_i = x^i / g = (x^i * g_den) / g_num.
        let mut basis = Vec::with_capacity((deg + 1) as usize);
        let mut xi = UnivariatePolynomial::<K>::one();
        let x = UnivariatePolynomial::new(vec![K::zero(), K::one()]);
        for _ in 0..=deg {
            let f = RationalFunction::new(xi.clone() * g_den.clone(), g_num.clone())
                .expect("g_num is a product of monic irreducibles, hence nonzero");
            basis.push(f);
            xi = xi * x.clone();
        }
        basis
    }

    /// dim_K L(D): `deg D + 1` for deg D >= 0, else 0 (Riemann-Roch on P^1,
    /// genus 0). Equals `riemann_roch_basis().len()`.
    pub fn riemann_roch_dimension(&self) -> usize {
        let deg = self.degree();
        if deg < 0 {
            0
        } else {
            (deg + 1) as usize
        }
    }
}

impl<K: Field + EuclideanDomain> PartialEq for Divisor<K> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs.len() == other.coeffs.len()
            && self
                .coeffs
                .iter()
                .all(|(p, c)| other.multiplicity(p) == *c)
    }
}

impl<K: Field + EuclideanDomain> Add for Divisor<K> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for (p, c) in rhs.coeffs {
            self.add_place(p, c);
        }
        self
    }
}

impl<K: Field + EuclideanDomain> Neg for Divisor<K> {
    type Output = Self;
    fn neg(mut self) -> Self {
        for (_, c) in self.coeffs.iter_mut() {
            *c = -*c;
        }
        self
    }
}

impl<K: Field + EuclideanDomain> Sub for Divisor<K> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<K: Field + EuclideanDomain> fmt::Display for Divisor<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coeffs.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (p, c) in &self.coeffs {
            if first {
                write!(f, "{}*{}", c, p)?;
                first = false;
            } else {
                write!(f, " + {}*{}", c, p)?;
            }
        }
        Ok(())
    }
}

/// Check membership f in L(D) directly from the definition:
/// f = 0, or div(f) + D >= 0. Used as an independent gate in tests.
pub fn is_in_riemann_roch_space<K: FactorableConstantField>(
    f: &RationalFunction<K>,
    d: &Divisor<K>,
) -> Result<bool> {
    if f.is_zero() {
        return Ok(true);
    }
    let df = Divisor::principal(f)?;
    Ok((df + d.clone()).is_effective())
}

#[cfg(test)]
mod tests {
    use super::super::gfp::GFp;
    use super::*;
    use rustmath_rationals::Rational;

    fn qpoly(coeffs: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| Rational::from_i64(c)).collect())
    }

    fn fpoly<const P: u64>(coeffs: &[i64]) -> UnivariatePolynomial<GFp<P>> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| GFp::<P>::new(c)).collect())
    }

    fn qrf(num: &[i64], den: &[i64]) -> RationalFunction<Rational> {
        RationalFunction::new(qpoly(num), qpoly(den)).unwrap()
    }

    fn frf<const P: u64>(num: &[i64], den: &[i64]) -> RationalFunction<GFp<P>> {
        RationalFunction::new(fpoly::<P>(num), fpoly::<P>(den)).unwrap()
    }

    /// Deterministic LCG for reproducible random batteries.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    fn random_coeffs(state: &mut u64) -> Vec<i64> {
        let deg = (lcg(state) % 5) as usize;
        (0..=deg).map(|_| (lcg(state) % 9) as i64 - 4).collect()
    }

    fn random_qrf(state: &mut u64) -> RationalFunction<Rational> {
        loop {
            let n = qpoly(&random_coeffs(state));
            let d = qpoly(&random_coeffs(state));
            if !n.is_zero() && !d.is_zero() {
                return RationalFunction::new(n, d).unwrap();
            }
        }
    }

    fn random_frf5(state: &mut u64) -> RationalFunction<GFp<5>> {
        loop {
            let n = fpoly::<5>(&random_coeffs(state));
            let d = fpoly::<5>(&random_coeffs(state));
            if !n.is_zero() && !d.is_zero() {
                return RationalFunction::new(n, d).unwrap();
            }
        }
    }

    #[test]
    fn test_principal_divisor_q_specific() {
        // sympy-verified: f = (x^2-1)/(x^3+2x) has
        // div(f) = [x-1] + [x+1] - [x] - [x^2+2] + 1*[inf], degree 0.
        let f = qrf(&[-1, 0, 1], &[0, 2, 0, 1]);
        let d = Divisor::principal(&f).unwrap();
        assert_eq!(d.multiplicity(&Place::finite_linear(Rational::from_i64(1))), 1);
        assert_eq!(d.multiplicity(&Place::finite_linear(Rational::from_i64(-1))), 1);
        assert_eq!(d.multiplicity(&Place::finite_linear(Rational::zero())), -1);
        assert_eq!(
            d.multiplicity(&Place::finite(qpoly(&[2, 0, 1])).unwrap()),
            -1
        );
        assert_eq!(d.multiplicity(&Place::infinite()), 1);
        assert_eq!(d.support().len(), 5);
        assert_eq!(d.degree(), 0);
    }

    #[test]
    fn test_principal_divisor_gf5_specific() {
        // sympy-verified over GF(5): x^2+1 = (x+2)(x+3), x^5-x+1 irreducible,
        // so div((x^2+1)/(x^5-x+1)) = [x+2] + [x+3] - [x^5-x+1] + 3*[inf].
        let f = frf::<5>(&[1, 0, 1], &[1, 4, 0, 0, 0, 1]);
        let d = Divisor::principal(&f).unwrap();
        assert_eq!(d.multiplicity(&Place::finite(fpoly::<5>(&[2, 1])).unwrap()), 1);
        assert_eq!(d.multiplicity(&Place::finite(fpoly::<5>(&[3, 1])).unwrap()), 1);
        assert_eq!(
            d.multiplicity(&Place::finite(fpoly::<5>(&[1, 4, 0, 0, 0, 1])).unwrap()),
            -1
        );
        assert_eq!(d.multiplicity(&Place::infinite()), 3);
        assert_eq!(d.degree(), 0);
    }

    #[test]
    fn test_degree_law_battery_q() {
        // Gate: sum over all places of v_P(f) * deg(P) = deg div(f) = 0,
        // for 25 pseudo-random nonzero f over Q. principal() itself certifies
        // the factorizations by reconstruction.
        let mut state = 0xC0FFEE_u64;
        for _ in 0..25 {
            let f = random_qrf(&mut state);
            let d = Divisor::principal(&f).expect("factorization certified");
            assert_eq!(d.degree(), 0, "deg div({}) != 0", f);
        }
    }

    #[test]
    fn test_degree_law_battery_gf5() {
        let mut state = 0xBEEF_u64;
        for _ in 0..25 {
            let f = random_frf5(&mut state);
            let d = Divisor::principal(&f).expect("factorization certified");
            assert_eq!(d.degree(), 0, "deg div({}) != 0", f);
        }
    }

    #[test]
    fn test_product_law_battery_q() {
        // v_P(fg) = v_P(f) + v_P(g) at every place of the combined support
        // (and the infinite place), 15 random pairs over Q.
        let mut state = 0xABCD_u64;
        for _ in 0..15 {
            let f = random_qrf(&mut state);
            let g = random_qrf(&mut state);
            let prod = f.clone() * g.clone();
            assert!(!prod.is_zero());
            let mut places: Vec<Place<Rational>> = vec![Place::infinite()];
            for pl in Divisor::principal(&f).unwrap().support() {
                if !places.contains(pl) {
                    places.push(pl.clone());
                }
            }
            for pl in Divisor::principal(&g).unwrap().support() {
                if !places.contains(pl) {
                    places.push(pl.clone());
                }
            }
            for pl in &places {
                assert_eq!(
                    pl.valuation(&prod).unwrap(),
                    pl.valuation(&f).unwrap() + pl.valuation(&g).unwrap(),
                    "v(fg) != v(f)+v(g) at {} for f={}, g={}",
                    pl,
                    f,
                    g
                );
            }
        }
    }

    #[test]
    fn test_product_law_battery_gf5() {
        let mut state = 0x5EED_u64;
        for _ in 0..15 {
            let f = random_frf5(&mut state);
            let g = random_frf5(&mut state);
            let prod = f.clone() * g.clone();
            let mut places: Vec<Place<GFp<5>>> = vec![Place::infinite()];
            for pl in Divisor::principal(&f).unwrap().support() {
                if !places.contains(pl) {
                    places.push(pl.clone());
                }
            }
            for pl in Divisor::principal(&g).unwrap().support() {
                if !places.contains(pl) {
                    places.push(pl.clone());
                }
            }
            for pl in &places {
                assert_eq!(
                    pl.valuation(&prod).unwrap(),
                    pl.valuation(&f).unwrap() + pl.valuation(&g).unwrap()
                );
            }
        }
    }

    #[test]
    fn test_ultrametric_battery() {
        // v(f+g) >= min(v(f), v(g)), with equality whenever v(f) != v(g).
        let mut state = 0xFACE_u64;
        for _ in 0..15 {
            let f = random_qrf(&mut state);
            let g = random_qrf(&mut state);
            let sum = f.clone() + g.clone();
            let mut places: Vec<Place<Rational>> = vec![Place::infinite()];
            for pl in Divisor::principal(&f).unwrap().support() {
                if !places.contains(pl) {
                    places.push(pl.clone());
                }
            }
            for pl in Divisor::principal(&g).unwrap().support() {
                if !places.contains(pl) {
                    places.push(pl.clone());
                }
            }
            for pl in &places {
                let vf = pl.valuation(&f).unwrap();
                let vg = pl.valuation(&g).unwrap();
                match pl.valuation(&sum) {
                    None => {
                        // f + g = 0: forces v(f) = v(g).
                        assert_eq!(vf, vg);
                    }
                    Some(vs) => {
                        assert!(vs >= vf.min(vg), "ultrametric fails at {}", pl);
                        if vf != vg {
                            assert_eq!(vs, vf.min(vg), "strictness fails at {}", pl);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_divisor_group_laws() {
        let p1 = Place::finite_linear(Rational::zero());
        let p2 = Place::<Rational>::finite(qpoly(&[2, 0, 1])).unwrap();
        let inf = Place::<Rational>::infinite();
        let d1 = Divisor::from_places(vec![(p1.clone(), 2), (p2.clone(), -1), (inf.clone(), 1)]);
        let d2 = Divisor::from_places(vec![(p1.clone(), -2), (inf.clone(), 3)]);
        // Degrees: d1 = 2 - 2 + 1 = 1; d2 = -2 + 3 = 1; additive.
        assert_eq!(d1.degree(), 1);
        assert_eq!(d2.degree(), 1);
        assert_eq!((d1.clone() + d2.clone()).degree(), 2);
        // D - D = 0.
        assert!((d1.clone() - d1.clone()).is_zero());
        // Cancellation in the sum: p1 coefficient 2 + (-2) = 0 drops out.
        let s = d1.clone() + d2.clone();
        assert_eq!(s.multiplicity(&p1), 0);
        assert_eq!(s.multiplicity(&p2), -1);
        assert_eq!(s.multiplicity(&inf), 4);
        // Effective parts.
        let (pos, neg) = d1.effective_parts();
        assert!(pos.is_effective() && neg.is_effective());
        assert_eq!(pos - neg, d1);
    }

    #[test]
    fn test_principal_zeros_and_poles() {
        // div(f) = zeros - poles with both effective; for f = (x^2-1)/(x^3+2x)
        // zeros = [x-1]+[x+1]+[inf], poles = [x]+[x^2+2].
        let f = qrf(&[-1, 0, 1], &[0, 2, 0, 1]);
        let d = Divisor::principal(&f).unwrap();
        let (zeros, poles) = d.effective_parts();
        assert_eq!(zeros.degree(), 3);
        assert_eq!(poles.degree(), 3);
        assert!(zeros.is_effective() && poles.is_effective());
    }

    #[test]
    fn test_riemann_roch_n_infinity() {
        // L(n*[inf]) has basis {1, x, ..., x^n}: the exact statement of the
        // task's gate for D = n*[inf].
        let d = Divisor::from_places(vec![(Place::<Rational>::infinite(), 3)]);
        assert_eq!(d.riemann_roch_dimension(), 4);
        let basis = d.riemann_roch_basis();
        assert_eq!(basis.len(), 4);
        for (i, f) in basis.iter().enumerate() {
            let mut expect = qpoly(&[1]);
            for _ in 0..i {
                expect = expect * qpoly(&[0, 1]);
            }
            assert_eq!(f, &RationalFunction::from_polynomial(expect));
            assert!(is_in_riemann_roch_space(f, &d).unwrap());
        }
    }

    #[test]
    fn test_riemann_roch_mixed_divisor() {
        // sympy-verified example: D = 2[x-1] - [x] + [inf], deg D = 2,
        // dim L(D) = 3; every basis element satisfies div(f) + D >= 0 and
        // f_i = x^i * f_0 (certifying the x^i/g form => linear independence).
        let d = Divisor::from_places(vec![
            (Place::finite_linear(Rational::from_i64(1)), 2),
            (Place::finite_linear(Rational::zero()), -1),
            (Place::<Rational>::infinite(), 1),
        ]);
        assert_eq!(d.degree(), 2);
        assert_eq!(d.riemann_roch_dimension(), 3);
        let basis = d.riemann_roch_basis();
        assert_eq!(basis.len(), 3);
        let x = RationalFunction::<Rational>::gen();
        for (i, f) in basis.iter().enumerate() {
            assert!(
                is_in_riemann_roch_space(f, &d).unwrap(),
                "basis element {} not in L(D)",
                f
            );
            let mut expect = basis[0].clone();
            for _ in 0..i {
                expect = expect * x.clone();
            }
            assert_eq!(f, &expect);
        }
    }

    #[test]
    fn test_riemann_roch_negative_degree() {
        // deg D < 0 => L(D) = 0 on P^1.
        let d = Divisor::from_places(vec![(Place::finite_linear(Rational::zero()), -1)]);
        assert_eq!(d.riemann_roch_dimension(), 0);
        assert!(d.riemann_roch_basis().is_empty());
        let d2 = Divisor::from_places(vec![
            (Place::finite_linear(Rational::zero()), 1),
            (Place::<Rational>::infinite(), -2),
        ]);
        assert_eq!(d2.degree(), -1);
        assert_eq!(d2.riemann_roch_dimension(), 0);
    }

    #[test]
    fn test_riemann_roch_battery() {
        // RR gate on P^1: dim L(D) = deg D + 1 for deg D >= 0, membership of
        // every basis element checked against the definition div(f) + D >= 0.
        let mut state = 0x12345_u64;
        let base_places = [
            Place::finite_linear(Rational::zero()),
            Place::finite_linear(Rational::from_i64(1)),
            Place::<Rational>::finite(qpoly(&[2, 0, 1])).unwrap(),
            Place::<Rational>::infinite(),
        ];
        for _ in 0..10 {
            let mut d = Divisor::zero();
            for pl in &base_places {
                let m = (lcg(&mut state) % 5) as i64 - 2;
                d.add_place(pl.clone(), m);
            }
            let deg = d.degree();
            let basis = d.riemann_roch_basis();
            if deg < 0 {
                assert!(basis.is_empty());
                assert_eq!(d.riemann_roch_dimension(), 0);
            } else {
                assert_eq!(basis.len() as i64, deg + 1);
                assert_eq!(d.riemann_roch_dimension() as i64, deg + 1);
                for f in &basis {
                    assert!(is_in_riemann_roch_space(f, &d).unwrap());
                }
            }
        }
    }

    #[test]
    fn test_riemann_roch_gf5() {
        // Same RR gate over GF(5): D = 2[x] + [x^2+2] - [inf], deg = 3.
        let d = Divisor::from_places(vec![
            (Place::finite_linear(GFp::<5>::new(0)), 2),
            (Place::finite(fpoly::<5>(&[2, 0, 1])).unwrap(), 1),
            (Place::<GFp<5>>::infinite(), -1),
        ]);
        assert_eq!(d.degree(), 3);
        let basis = d.riemann_roch_basis();
        assert_eq!(basis.len(), 4);
        for f in &basis {
            assert!(is_in_riemann_roch_space(f, &d).unwrap());
        }
    }

    #[test]
    fn test_principal_of_zero_is_err() {
        assert!(Divisor::principal(&RationalFunction::<Rational>::zero()).is_err());
    }

    #[test]
    fn test_principal_of_constant_is_zero_divisor() {
        let c = RationalFunction::constant(Rational::from_i64(7));
        let d = Divisor::principal(&c).unwrap();
        assert!(d.is_zero());
        assert_eq!(d.degree(), 0);
    }
}
