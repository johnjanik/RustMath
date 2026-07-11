//! Places of the rational function field K(x), with exact valuations,
//! uniformizers, and residue fields.
//!
//! The places of K(x) are:
//!
//! - **finite places**: one for each monic irreducible p(x) in K[x], with
//!   valuation v_p(f) = (multiplicity of p in num) - (multiplicity in den),
//!   uniformizer p(x), and residue field K[x]/(p) of degree deg p over K;
//! - **the infinite place**: v_inf(f) = deg(den) - deg(num), uniformizer 1/x,
//!   residue field K (degree 1).
//!
//! Residue fields of finite places are represented by the quotient ring
//! [`ResidueClass`] (K[x] mod p). For K = GF(p) and deg p = d this IS the
//! field GF(p^d); the tests enumerate GF(25) = GF(5)[x]/(x^2+2) and certify
//! field axioms (every nonzero element invertible, a^24 = 1).

use super::element::RationalFunction;
use super::factor::FactorableConstantField;
use rustmath_core::{EuclideanDomain, Field, MathError, Result, Ring};
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;

/// Internal representation; kept private so that the "finite place
/// polynomials are monic irreducible" invariant cannot be violated from
/// outside.
#[derive(Clone, Debug, PartialEq)]
enum PlaceKind<K: Field + EuclideanDomain> {
    /// A monic irreducible polynomial p(x).
    Finite(UnivariatePolynomial<K>),
    /// The place at infinity (the pole of x).
    Infinite,
}

/// A place of K(x).
#[derive(Clone, Debug, PartialEq)]
pub struct Place<K: Field + EuclideanDomain> {
    kind: PlaceKind<K>,
}

/// Extended Euclid for polynomials over a field: returns (g, s, t) with
/// g = s*a + t*b. `g` is not normalized to monic.
fn poly_extended_gcd<K: Field + EuclideanDomain>(
    a: &UnivariatePolynomial<K>,
    b: &UnivariatePolynomial<K>,
) -> (
    UnivariatePolynomial<K>,
    UnivariatePolynomial<K>,
    UnivariatePolynomial<K>,
) {
    let mut old_r = a.clone();
    let mut r = b.clone();
    let mut old_s = UnivariatePolynomial::one();
    let mut s = UnivariatePolynomial::zero();
    let mut old_t = UnivariatePolynomial::zero();
    let mut t = UnivariatePolynomial::one();
    while !r.is_zero() {
        let (q, rem) = old_r.div_rem(&r).expect("nonzero divisor");
        old_r = r;
        r = rem;
        let ns = old_s - q.clone() * s.clone();
        old_s = s;
        s = ns;
        let nt = old_t - q * t.clone();
        old_t = t;
        t = nt;
    }
    (old_r, old_s, old_t)
}

impl<K: Field + EuclideanDomain> Place<K> {
    /// The infinite place of K(x).
    pub fn infinite() -> Self {
        Place {
            kind: PlaceKind::Infinite,
        }
    }

    /// The degree-1 finite place x - a. Linear polynomials are always
    /// irreducible, so no factorization capability is needed.
    pub fn finite_linear(a: K) -> Self {
        Place {
            kind: PlaceKind::Finite(UnivariatePolynomial::new(vec![-a, K::one()])),
        }
    }

    /// The finite place of the monic irreducible polynomial `p`.
    ///
    /// `p` is made monic; irreducibility is *verified* via
    /// [`FactorableConstantField::factor_poly`] and a reducible or constant
    /// `p` is an `Err`.
    pub fn finite(p: UnivariatePolynomial<K>) -> Result<Self>
    where
        K: FactorableConstantField,
    {
        let d = p.degree().unwrap_or(0);
        if p.is_zero() || d == 0 {
            return Err(MathError::InvalidArgument(
                "a finite place needs a non-constant polynomial".to_string(),
            ));
        }
        let p = p.make_monic();
        let (_, factors) = K::factor_poly(&p)?;
        if factors.len() != 1 || factors[0].1 != 1 {
            return Err(MathError::InvalidArgument(format!(
                "polynomial {} is not irreducible; not a place",
                p
            )));
        }
        Ok(Place {
            kind: PlaceKind::Finite(p),
        })
    }

    /// Construct a finite place from a factor already known to be monic
    /// irreducible (crate-internal: used by divisor decomposition, where the
    /// factorization has just been certified).
    pub(crate) fn finite_unchecked(p: UnivariatePolynomial<K>) -> Self {
        debug_assert!(p.is_monic() && p.degree().unwrap_or(0) >= 1);
        Place {
            kind: PlaceKind::Finite(p),
        }
    }

    /// Is this the infinite place?
    pub fn is_infinite(&self) -> bool {
        matches!(self.kind, PlaceKind::Infinite)
    }

    /// The monic irreducible polynomial of a finite place (None for infinity).
    pub fn polynomial(&self) -> Option<&UnivariatePolynomial<K>> {
        match &self.kind {
            PlaceKind::Finite(p) => Some(p),
            PlaceKind::Infinite => None,
        }
    }

    /// The degree of the place: deg p for a finite place, 1 for infinity.
    /// This equals [K_P : K] where K_P is the residue field.
    pub fn degree(&self) -> usize {
        match &self.kind {
            PlaceKind::Finite(p) => p.degree().expect("place polynomial is non-constant"),
            PlaceKind::Infinite => 1,
        }
    }

    /// A uniformizer: an element t with v_P(t) = 1 (p(x) finitely, 1/x at
    /// infinity).
    pub fn uniformizer(&self) -> RationalFunction<K> {
        match &self.kind {
            PlaceKind::Finite(p) => RationalFunction::from_polynomial(p.clone()),
            PlaceKind::Infinite => RationalFunction::new(
                UnivariatePolynomial::one(),
                UnivariatePolynomial::new(vec![K::zero(), K::one()]),
            )
            .expect("x is nonzero"),
        }
    }

    /// The valuation v_P(f). Returns `None` for f = 0 (v_P(0) = +infinity).
    pub fn valuation(&self, f: &RationalFunction<K>) -> Option<i64> {
        if f.is_zero() {
            return None;
        }
        Some(match &self.kind {
            PlaceKind::Finite(p) => {
                poly_divide_out(f.numerator(), p).0 - poly_divide_out(f.denominator(), p).0
            }
            PlaceKind::Infinite => {
                f.denominator().degree().unwrap_or(0) as i64
                    - f.numerator().degree().unwrap_or(0) as i64
            }
        })
    }

    /// Is f in the valuation ring O_P = {f : v_P(f) >= 0}? (0 is in every O_P.)
    pub fn is_in_valuation_ring(&self, f: &RationalFunction<K>) -> bool {
        match self.valuation(f) {
            None => true,
            Some(v) => v >= 0,
        }
    }

    /// Reduce f modulo a **finite** place: the image of f in the residue
    /// field K[x]/(p).
    ///
    /// `Err` if f has a pole at the place (v_P(f) < 0) or if the place is
    /// infinite (use [`Place::residue_at_infinity`]).
    pub fn residue(&self, f: &RationalFunction<K>) -> Result<ResidueClass<K>> {
        let p = match &self.kind {
            PlaceKind::Finite(p) => p,
            PlaceKind::Infinite => {
                return Err(MathError::InvalidArgument(
                    "residue at infinity lives in K; use residue_at_infinity".to_string(),
                ))
            }
        };
        if let Some(v) = self.valuation(f) {
            if v < 0 {
                return Err(MathError::InvalidArgument(
                    "f has a pole at this place; no residue".to_string(),
                ));
            }
        }
        // gcd(num, den) = 1 and v_P(f) >= 0 imply p does not divide den, so
        // den is invertible mod p.
        let num_cls = ResidueClass::new(f.numerator().clone(), p.clone())?;
        let den_cls = ResidueClass::new(f.denominator().clone(), p.clone())?;
        num_cls.mul(&den_cls.inverse()?)
    }

    /// Reduce f at the infinite place: the residue field is K itself.
    ///
    /// v_inf(f) > 0 gives 0; v_inf(f) = 0 gives lc(num)/lc(den); a pole
    /// (v_inf(f) < 0) is an `Err`.
    pub fn residue_at_infinity(f: &RationalFunction<K>) -> Result<K> {
        if f.is_zero() {
            return Ok(K::zero());
        }
        let dn = f.numerator().degree().unwrap_or(0);
        let dd = f.denominator().degree().unwrap_or(0);
        if dn > dd {
            return Err(MathError::InvalidArgument(
                "f has a pole at infinity; no residue".to_string(),
            ));
        }
        if dn < dd {
            return Ok(K::zero());
        }
        let ln = f
            .numerator()
            .leading_coefficient()
            .expect("nonzero")
            .clone();
        let ld = f
            .denominator()
            .leading_coefficient()
            .expect("nonzero")
            .clone();
        Ok(ln * ld.inverse()?)
    }

    /// The degree of the residue field over K (same as [`Place::degree`]).
    pub fn residue_field_degree(&self) -> usize {
        self.degree()
    }
}

impl<K: Field + EuclideanDomain> fmt::Display for Place<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            PlaceKind::Finite(p) => write!(f, "({})", p),
            PlaceKind::Infinite => write!(f, "(1/x)"),
        }
    }
}

/// An element of the residue field K[x]/(p) of a finite place.
///
/// The representative has degree < deg p. For K = GF(q) and deg p = d this
/// quotient ring is the finite field GF(q^d).
#[derive(Clone, Debug, PartialEq)]
pub struct ResidueClass<K: Field + EuclideanDomain> {
    rep: UnivariatePolynomial<K>,
    modulus: UnivariatePolynomial<K>,
}

impl<K: Field + EuclideanDomain> ResidueClass<K> {
    /// The class of `rep` modulo `modulus` (a monic non-constant polynomial).
    pub fn new(rep: UnivariatePolynomial<K>, modulus: UnivariatePolynomial<K>) -> Result<Self> {
        if modulus.degree().unwrap_or(0) == 0 {
            return Err(MathError::InvalidArgument(
                "residue class modulus must be non-constant".to_string(),
            ));
        }
        let (_, r) = rep.div_rem(&modulus)?;
        Ok(ResidueClass {
            rep: r,
            modulus,
        })
    }

    /// The canonical representative (degree < deg modulus).
    pub fn representative(&self) -> &UnivariatePolynomial<K> {
        &self.rep
    }

    /// The modulus p(x).
    pub fn modulus(&self) -> &UnivariatePolynomial<K> {
        &self.modulus
    }

    pub fn is_zero(&self) -> bool {
        self.rep.is_zero()
    }

    fn check_same_modulus(&self, other: &Self) -> Result<()> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument(
                "residue classes from different residue fields".to_string(),
            ));
        }
        Ok(())
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.check_same_modulus(other)?;
        ResidueClass::new(self.rep.clone() + other.rep.clone(), self.modulus.clone())
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.check_same_modulus(other)?;
        ResidueClass::new(self.rep.clone() - other.rep.clone(), self.modulus.clone())
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.check_same_modulus(other)?;
        ResidueClass::new(self.rep.clone() * other.rep.clone(), self.modulus.clone())
    }

    /// Multiplicative inverse via the extended Euclidean algorithm.
    ///
    /// `Err(DivisionByZero)` for the zero class. When the modulus is
    /// irreducible (always true for classes produced by [`Place::residue`]),
    /// every nonzero class is invertible.
    pub fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let (g, s, _) = poly_extended_gcd(&self.rep, &self.modulus);
        if g.degree().unwrap_or(0) != 0 {
            return Err(MathError::InvalidArgument(
                "representative shares a factor with the modulus; not invertible \
                 (modulus not irreducible?)"
                    .to_string(),
            ));
        }
        // g is a nonzero constant c: s * rep = c (mod modulus), inverse = s/c.
        let c = g.leading_coefficient().expect("nonzero gcd").clone();
        let s = s.scalar_mul(&c.inverse()?);
        ResidueClass::new(s, self.modulus.clone())
    }

    /// Raise to a non-negative power (square-and-multiply on representatives).
    pub fn pow(&self, mut n: u64) -> Result<Self> {
        let mut base = self.clone();
        let mut acc = ResidueClass::new(UnivariatePolynomial::one(), self.modulus.clone())?;
        while n > 0 {
            if n & 1 == 1 {
                acc = acc.mul(&base)?;
            }
            base = base.mul(&base)?;
            n >>= 1;
        }
        Ok(acc)
    }
}

impl<K: Field + EuclideanDomain> fmt::Display for ResidueClass<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.rep, self.modulus)
    }
}

/// Exact division a/p^m within K[x]; used by divisor decomposition.
pub(crate) fn poly_divide_out<K: Field + EuclideanDomain>(
    a: &UnivariatePolynomial<K>,
    p: &UnivariatePolynomial<K>,
) -> (i64, UnivariatePolynomial<K>) {
    let mut m = 0i64;
    let mut work = a.clone();
    loop {
        if work.degree().unwrap_or(0) < p.degree().unwrap_or(0) {
            break;
        }
        let (q, r) = work.div_rem(p).expect("nonzero divisor");
        if !r.is_zero() {
            break;
        }
        m += 1;
        work = q;
    }
    (m, work)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn qpoly(coeffs: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| Rational::from_i64(c)).collect())
    }

    fn qrf(num: &[i64], den: &[i64]) -> RationalFunction<Rational> {
        RationalFunction::new(qpoly(num), qpoly(den)).unwrap()
    }

    #[test]
    fn test_finite_place_rejects_reducible() {
        // x^2-1 = (x-1)(x+1): not a place.
        assert!(Place::<Rational>::finite(qpoly(&[-1, 0, 1])).is_err());
        // x^2+2 is irreducible over Q (sympy-verified).
        let p = Place::<Rational>::finite(qpoly(&[2, 0, 1])).unwrap();
        assert_eq!(p.degree(), 2);
    }

    #[test]
    fn test_valuations_specific() {
        // f = (x^2-1)/(x^3+2x); sympy-verified factorizations:
        // num = (x-1)(x+1), den = x(x^2+2), v_inf = 1.
        let f = qrf(&[-1, 0, 1], &[0, 2, 0, 1]);
        let px = Place::finite_linear(Rational::zero()); // x
        let p1 = Place::finite_linear(Rational::from_i64(1)); // x-1
        let pm1 = Place::finite_linear(Rational::from_i64(-1)); // x+1
        let pq = Place::<Rational>::finite(qpoly(&[2, 0, 1])).unwrap(); // x^2+2
        let pinf = Place::<Rational>::infinite();
        assert_eq!(p1.valuation(&f), Some(1));
        assert_eq!(pm1.valuation(&f), Some(1));
        assert_eq!(px.valuation(&f), Some(-1));
        assert_eq!(pq.valuation(&f), Some(-1));
        assert_eq!(pinf.valuation(&f), Some(1));
        // Degree-weighted sum over ALL places (only these five are nonzero): 0.
        let s = 1 * 1 + 1 * 1 + (-1) * 1 + (-1) * 2 + 1 * 1;
        assert_eq!(s, 0);
    }

    #[test]
    fn test_valuation_of_zero_is_infinity() {
        let px = Place::finite_linear(Rational::zero());
        assert_eq!(px.valuation(&RationalFunction::zero()), None);
        assert!(px.is_in_valuation_ring(&RationalFunction::zero()));
    }

    #[test]
    fn test_uniformizer_has_valuation_one() {
        let places = [
            Place::finite_linear(Rational::from_i64(3)),
            Place::<Rational>::finite(qpoly(&[2, 0, 1])).unwrap(),
            Place::<Rational>::infinite(),
        ];
        for pl in &places {
            assert_eq!(pl.valuation(&pl.uniformizer()), Some(1));
        }
    }

    #[test]
    fn test_residue_at_quadratic_place() {
        // sympy-verified: residue of x/(x+1) mod (x^2+2) is (x+2)/3, i.e.
        // representative (2/3) + (1/3)x.
        let f = qrf(&[0, 1], &[1, 1]);
        let pq = Place::<Rational>::finite(qpoly(&[2, 0, 1])).unwrap();
        let r = pq.residue(&f).unwrap();
        let expected = UnivariatePolynomial::new(vec![
            Rational::new(2, 3).unwrap(),
            Rational::new(1, 3).unwrap(),
        ]);
        assert_eq!(r.representative(), &expected);
    }

    #[test]
    fn test_residue_pole_is_err() {
        // f = 1/x has a pole at the place x.
        let f = qrf(&[1], &[0, 1]);
        let px = Place::finite_linear(Rational::zero());
        assert!(px.residue(&f).is_err());
    }

    #[test]
    fn test_residue_degree_one_place_is_evaluation() {
        // At x - 2, the residue of f = (x^2+1)/(x+1) is f(2) = 5/3.
        let f = qrf(&[1, 0, 1], &[1, 1]);
        let p2 = Place::finite_linear(Rational::from_i64(2));
        let r = p2.residue(&f).unwrap();
        assert_eq!(
            r.representative(),
            &UnivariatePolynomial::constant(Rational::new(5, 3).unwrap())
        );
        assert_eq!(f.evaluate(&Rational::from_i64(2)).unwrap(), Rational::new(5, 3).unwrap());
    }

    #[test]
    fn test_residue_at_infinity() {
        // v_inf = 0: f = (2x^2+1)/(x^2-1) -> lc ratio 2.
        let f = qrf(&[1, 0, 2], &[-1, 0, 1]);
        assert_eq!(
            Place::residue_at_infinity(&f).unwrap(),
            Rational::from_i64(2)
        );
        // v_inf > 0: 1/x -> 0.
        let g = qrf(&[1], &[0, 1]);
        assert_eq!(Place::residue_at_infinity(&g).unwrap(), Rational::zero());
        // v_inf < 0: x has a pole at infinity.
        let h = qrf(&[0, 1], &[1]);
        assert!(Place::<Rational>::residue_at_infinity(&h).is_err());
    }

    #[test]
    fn test_valuation_ring_membership() {
        // v_(x-1)(x) = 0 -> x is in O_P for P = (x-1); 1/(x-1) is not.
        let p1 = Place::finite_linear(Rational::from_i64(1));
        let x = RationalFunction::<Rational>::gen();
        assert!(p1.is_in_valuation_ring(&x));
        let inv = qrf(&[1], &[-1, 1]);
        assert!(!p1.is_in_valuation_ring(&inv));
        assert_eq!(p1.valuation(&inv), Some(-1));
    }

    #[test]
    fn test_residue_field_gf25_is_a_field() {
        // sympy-verified: x^2+2 is irreducible over GF(5), so
        // GF(5)[x]/(x^2+2) = GF(25). Enumerate all 24 nonzero classes:
        // each is invertible with a * a^{-1} = 1 and a^24 = 1.
        use super::super::gfp::GFp;
        let mkpoly = |coeffs: &[i64]| {
            UnivariatePolynomial::new(coeffs.iter().map(|&c| GFp::<5>::new(c)).collect::<Vec<_>>())
        };
        let modulus = mkpoly(&[2, 0, 1]);
        let one = ResidueClass::new(mkpoly(&[1]), modulus.clone()).unwrap();
        let mut nonzero = 0;
        for a0 in 0..5i64 {
            for a1 in 0..5i64 {
                if a0 == 0 && a1 == 0 {
                    continue;
                }
                let a = ResidueClass::new(mkpoly(&[a0, a1]), modulus.clone()).unwrap();
                let inv = a.inverse().unwrap();
                assert_eq!(a.mul(&inv).unwrap(), one, "a = {}", a);
                assert_eq!(a.pow(24).unwrap(), one, "a^24 != 1 for a = {}", a);
                nonzero += 1;
            }
        }
        assert_eq!(nonzero, 24);
        // The zero class is honestly non-invertible.
        let zero = ResidueClass::new(mkpoly(&[0]), modulus).unwrap();
        assert!(zero.inverse().is_err());
    }

    #[test]
    fn test_residue_map_gf5_at_quadratic_place() {
        // sympy-verified: (x+1)^{-1} mod (x^2+2) over GF(5) is 3x + 2, so the
        // residue of f = 1/(x+1) at the place (x^2+2) has representative 3x+2.
        use super::super::gfp::GFp;
        let mkpoly = |coeffs: &[i64]| {
            UnivariatePolynomial::new(coeffs.iter().map(|&c| GFp::<5>::new(c)).collect::<Vec<_>>())
        };
        let f = RationalFunction::new(mkpoly(&[1]), mkpoly(&[1, 1])).unwrap();
        let place = Place::finite(mkpoly(&[2, 0, 1])).unwrap();
        assert_eq!(place.degree(), 2);
        assert_eq!(place.residue_field_degree(), 2);
        let r = place.residue(&f).unwrap();
        assert_eq!(r.representative(), &mkpoly(&[2, 3]));
    }

    #[test]
    fn test_uniformizer_and_valuation_gf5() {
        use super::super::gfp::GFp;
        let mkpoly = |coeffs: &[i64]| {
            UnivariatePolynomial::new(coeffs.iter().map(|&c| GFp::<5>::new(c)).collect::<Vec<_>>())
        };
        let places = [
            Place::finite_linear(GFp::<5>::new(2)),
            Place::finite(mkpoly(&[2, 0, 1])).unwrap(),
            Place::<GFp<5>>::infinite(),
        ];
        for pl in &places {
            assert_eq!(pl.valuation(&pl.uniformizer()), Some(1));
        }
    }
}
