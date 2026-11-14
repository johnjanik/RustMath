//! The ring predicates of a number field: core [`Parent`] and the three-valued
//! `rustmath_category` traits [`RingParentMethods`] and [`CommutativeRingParentMethods`].
//!
//! MAGMA Handbook Ch. 34 §34.2.1 (`NumberField(f)` with `Check := true`: f irreducible),
//! §34.3.4 (`Characteristic` 0), §34.3.6 (`IsCommutative`, `IsFinite` false, `IsField`);
//! Ch. 17 §17.4.2-3. Irreducibility over Q is decided by the complete Zassenhaus
//! factoriser of rustmath-polynomials (H. Zassenhaus, "On Hensel factorization I", J.
//! Number Theory 1 (1969); H. Cohen, *A Course in Computational Algebraic Number Theory*,
//! GTM 138, §3.5).
//!
//! K = Q\[x\]/(f), deg f >= 1. If f is irreducible over Q, (f) is maximal and K is a field
//! (so a domain, Euclidean, a PID and a UFD). If f = gh with deg g, deg h >= 1, then
//! g(α)h(α) = 0 with g(α), h(α) nonzero (degrees < deg f), so K is neither a domain nor a
//! field. The content of f is a unit of Q and does not change K.

use crate::{NumberField, NumberFieldElement};
use rustmath_category::{CommutativeRingParentMethods, RingParentMethods};
use rustmath_core::Parent;
use rustmath_integers::Integer;
use rustmath_polynomials::{is_irreducible_over_integers, UnivariatePolynomial};
use rustmath_rationals::Rational;

impl NumberField {
    /// Whether the defining polynomial is irreducible over Q (PARI `polisirreducible`;
    /// MAGMA's `Check` in `NumberField(f)`). [`NumberField::new`] does not check it. The
    /// polynomial is scaled by the lcm of its denominators to an integer polynomial (the
    /// content does not matter), then
    /// [`is_irreducible_over_integers`] (complete Zassenhaus factorisation) decides:
    /// `Ok(b)` gives `Some(b)`, `Err` (for example a recombination bound exceeded) gives
    /// `None`.
    pub fn defining_polynomial_is_irreducible(&self) -> Option<bool> {
        let coeffs = self.minimal_polynomial.coefficients();
        let den = coeffs
            .iter()
            .fold(Integer::one(), |l, c| l.lcm(c.denominator()));
        let scale = Rational::from_integer(den);
        let ints: Vec<Integer> = coeffs
            .iter()
            .map(|c| (c.clone() * scale.clone()).numerator().clone())
            .collect();
        is_irreducible_over_integers(&UnivariatePolynomial::new(ints)).ok()
    }
}

impl Parent for NumberField {
    type Element = NumberFieldElement;

    /// The element belongs to this field: the same defining polynomial.
    fn contains(&self, x: &NumberFieldElement) -> bool {
        x.min_poly == self.minimal_polynomial
    }

    fn zero(&self) -> Option<NumberFieldElement> {
        Some(NumberField::zero(self))
    }

    fn one(&self) -> Option<NumberFieldElement> {
        Some(NumberField::one(self))
    }

    /// Characteristic 0 and nonzero (deg f >= 1), so infinite (MAGMA §34.3.6: `IsFinite`
    /// false).
    fn is_finite(&self) -> Option<bool> {
        Some(false)
    }

    fn name(&self) -> String {
        format!("Number field defined by {}", self.minimal_polynomial)
    }
}

impl RingParentMethods for NumberField {
    /// Q ⊂ K.
    fn characteristic(&self) -> Option<u64> {
        Some(0)
    }
    /// A quotient of Q\[x\].
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        self.defining_polynomial_is_irreducible()
    }
    fn is_field(&self) -> Option<bool> {
        self.defining_polynomial_is_irreducible()
    }
}

impl CommutativeRingParentMethods for NumberField {}

#[cfg(test)]
mod tests {
    use super::*;

    fn nf(cs: &[(i64, i64)]) -> NumberField {
        NumberField::new(UnivariatePolynomial::new(
            cs.iter()
                .map(|&(n, d)| Rational::new(n, d).unwrap())
                .collect(),
        ))
    }
    fn kleene(k: &NumberField) -> [Option<bool>; 5] {
        [
            k.is_integral_domain(),
            RingParentMethods::is_field(k),
            k.is_pid(),
            k.is_ufd(),
            k.is_euclidean(),
        ]
    }

    // PARI: polisirreducible(x^2-2) -> 1, polisirreducible(x^2-1) -> 0,
    // polisirreducible(x-3) -> 1, polisirreducible(x^2/2-1) -> 1. GAP:
    // IsField(AlgebraicExtension(Rationals, x^2-2)) -> true;
    // IsIrreducibleRingElement(PolynomialRing(Rationals,1), x^2-1) -> false. MAGMA §34.3.6:
    // IsFinite false, Characteristic 0. At c79a582 NumberField had no Parent impl.
    #[test]
    fn number_field_category() {
        let k = nf(&[(-2, 1), (0, 1), (1, 1)]);
        assert_eq!(k.characteristic(), Some(0));
        assert_eq!(k.is_commutative(), Some(true));
        assert_eq!(kleene(&k), [Some(true); 5]);
        assert_eq!(Parent::is_finite(&k), Some(false));
        let split = nf(&[(-1, 1), (0, 1), (1, 1)]);
        assert_eq!(kleene(&split), [Some(false); 5]);
        assert_eq!(
            RingParentMethods::is_field(&nf(&[(-3, 1), (1, 1)])),
            Some(true)
        );
        assert_eq!(
            RingParentMethods::is_field(&nf(&[(-1, 1), (0, 1), (1, 2)])),
            Some(true)
        );
        let l = nf(&[(-3, 1), (0, 1), (1, 1)]);
        assert!(k.contains(&k.generator()));
        assert!(!k.contains(&l.generator()));
        assert_eq!(Parent::zero(&k), Some(k.zero()));
        assert_eq!(Parent::one(&k), Some(k.one()));
        assert!(k.name().contains("Number field"));
    }
}
