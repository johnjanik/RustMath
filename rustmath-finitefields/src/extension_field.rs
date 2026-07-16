//! Extension finite fields GF(p^n) — legacy per-element API
//!
//! # Canonicalization note (Wave 2)
//!
//! This is the **legacy, per-element** API for `GF(p^n)`: every
//! [`ExtensionField`] value is built from its own copy of the characteristic
//! and the defining irreducible, and there is no separate parent object in
//! the API. As of Wave 2 it is a **thin wrapper around the canonical types**
//! [`crate::FiniteField`] / [`crate::FiniteFieldElement`] (module
//! [`crate::finite_field`]): construction goes through
//! [`FiniteField::with_modulus`] (so the parent is shared via
//! `UniqueRepresentation`, coefficients are reduced mod `p` and mod the
//! irreducible on entry, and invalid parameters — non-prime `p`, non-monic or
//! reducible modulus — are now rejected instead of silently accepted), and
//! all arithmetic delegates to [`FiniteFieldElement`].
//!
//! New code should use [`crate::FiniteField`] directly; this wrapper only
//! preserves the historical constructor/accessor shape
//! (`UnivariatePolynomial<Integer>` in and out).

use rustmath_core::{CommutativeRing, Field, Result, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::finite_field::{FiniteField, FiniteFieldElement};

/// Element of an extension finite field GF(p^n)
///
/// Represents elements as polynomials modulo an irreducible polynomial.
/// Since Wave 2 this is a thin wrapper over the canonical
/// [`FiniteFieldElement`]; the polynomial representation returned by
/// [`ExtensionField::poly`] is always fully reduced (coefficients in
/// `[0, p)`, degree below the degree of the irreducible).
#[derive(Clone, Debug)]
pub struct ExtensionField {
    elem: FiniteFieldElement,
    /// Reduced polynomial representative, kept for the legacy `poly()` accessor.
    poly: UnivariatePolynomial<Integer>,
}

impl ExtensionField {
    fn from_elem(elem: FiniteFieldElement) -> Self {
        let poly = UnivariatePolynomial::new(elem.eltseq().to_vec());
        ExtensionField { elem, poly }
    }

    /// Create a new element in GF(p^n)
    ///
    /// # Arguments
    ///
    /// * `poly` - Polynomial with coefficients in GF(p)
    /// * `characteristic` - Prime p
    /// * `irreducible` - Monic irreducible polynomial of degree n over GF(p)
    ///
    /// The input polynomial is reduced modulo `p` and modulo the irreducible.
    /// Errors if `characteristic` is not prime or `irreducible` is not a
    /// monic irreducible of degree >= 1 (the pre-Wave-2 constructor accepted
    /// such inputs silently).
    pub fn new(
        poly: UnivariatePolynomial<Integer>,
        characteristic: Integer,
        irreducible: UnivariatePolynomial<Integer>,
    ) -> Result<Self> {
        let field =
            FiniteField::with_modulus(characteristic, irreducible.coefficients().to_vec())?;
        Ok(Self::from_elem(field.element(poly.coefficients().to_vec())))
    }

    /// Get the (reduced) polynomial representation
    pub fn poly(&self) -> &UnivariatePolynomial<Integer> {
        &self.poly
    }

    /// Get the characteristic
    pub fn characteristic(&self) -> &Integer {
        self.elem.field().characteristic()
    }

    /// Get the degree of the extension
    pub fn degree(&self) -> usize {
        self.elem.field().degree()
    }

    /// The canonical parent field this element lives in.
    pub fn field(&self) -> &FiniteField {
        self.elem.field()
    }

    /// The canonical element this wrapper delegates to.
    pub fn as_element(&self) -> &FiniteFieldElement {
        &self.elem
    }

    /// Unwrap into the canonical [`FiniteFieldElement`].
    pub fn into_element(self) -> FiniteFieldElement {
        self.elem
    }

    /// Wrap a canonical [`FiniteFieldElement`] in the legacy API.
    pub fn from_element(elem: FiniteFieldElement) -> Self {
        Self::from_elem(elem)
    }

    /// Compute the Frobenius endomorphism: x -> x^p
    ///
    /// This is the fundamental automorphism of finite fields.
    pub fn frobenius(&self) -> Self {
        Self::from_elem(self.elem.frobenius())
    }

    /// Compute the norm N(x) = x · x^p · x^(p^2) · ... · x^(p^(n-1)), an element of
    /// `F_p` (returned as its representative in `[0, p)`).
    pub fn norm(&self) -> Integer {
        self.elem.norm()
    }

    /// Compute the trace Tr(x) = x + x^p + x^(p^2) + ... + x^(p^(n-1)),
    /// an element of `F_p` (returned as its representative in `[0, p)`).
    pub fn trace(&self) -> Integer {
        self.elem.trace()
    }
}

impl fmt::Display for ExtensionField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.elem.is_bound() {
            // Unbound sentinel (Ring::zero()/one()): no parameters to show.
            return write!(f, "{} (unbound)", self.poly);
        }
        write!(
            f,
            "{} in GF({}^{})",
            self.poly,
            self.characteristic(),
            self.degree()
        )
    }
}

impl PartialEq for ExtensionField {
    /// Delegates to the canonical [`FiniteFieldElement`] equality, and so
    /// inherits its *coercing* semantics (see that impl's docs):
    ///
    /// * **bound vs bound**: same parent (characteristic + modulus) and same
    ///   reduced coefficients; different parents are simply unequal.
    /// * **unbound sentinel (`Ring::zero()`/`one()`) vs bound**: the unbound
    ///   constant is bound into the other operand's field before comparing,
    ///   so `ExtensionField::zero() == (bound zero)`.
    /// * **unbound vs unbound**: equality of the integer constants in Z.
    ///
    /// The transitivity caveat is the same as [`FiniteFieldElement`]'s: the
    /// unbound zero equals the bound zero of every field while bound zeros
    /// of different fields stay unequal — confined to the cross-field corner
    /// whose arithmetic already panics.
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}

impl Add for ExtensionField {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::from_elem(self.elem + other.elem)
    }
}

impl Sub for ExtensionField {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::from_elem(self.elem - other.elem)
    }
}

impl Mul for ExtensionField {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::from_elem(self.elem * other.elem)
    }
}

impl Neg for ExtensionField {
    type Output = Self;

    fn neg(self) -> Self {
        Self::from_elem(-self.elem)
    }
}

impl Div for ExtensionField {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::from_elem(self.elem / rhs.elem)
    }
}

impl Ring for ExtensionField {
    /// The additive identity, as an *unbound* element: this wrapper delegates
    /// to the canonical [`FiniteFieldElement`], whose `Ring::zero()` is the
    /// unbound integer-constant sentinel bound on first contact with a bound
    /// element (see the `FiniteFieldElement` type docs for the precise
    /// algebra). Accessors that need the parameters
    /// ([`Self::characteristic`], [`Self::degree`], [`Self::field`],
    /// [`Self::poly`] shape) panic on a still-unbound element with a precise
    /// message.
    fn zero() -> Self {
        ExtensionField::from_elem(FiniteFieldElement::zero())
    }

    /// The multiplicative identity, as an *unbound* element (see
    /// [`Ring::zero`] above).
    fn one() -> Self {
        ExtensionField::from_elem(FiniteFieldElement::one())
    }

    fn is_zero(&self) -> bool {
        self.elem.is_zero()
    }

    fn is_one(&self) -> bool {
        self.elem.is_one()
    }
}

impl CommutativeRing for ExtensionField {
    // Marker trait, no methods to implement
}

impl Field for ExtensionField {
    /// Multiplicative inverse of a nonzero element of GF(p^n) (delegates to
    /// the canonical [`FiniteFieldElement::inverse`]).
    fn inverse(&self) -> Result<Self> {
        Ok(Self::from_elem(self.elem.inverse()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let p = Integer::from(2);

        // Create GF(2^2) with irreducible polynomial x^2 + x + 1
        let irreducible = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        ]);

        let poly = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(0)]);

        let elem = ExtensionField::new(poly, p, irreducible).unwrap();

        assert_eq!(elem.degree(), 2);
    }

    #[test]
    fn ring_zero_one_are_unbound_sentinels() {
        // These two calls used to panic unconditionally.
        let z = <ExtensionField as Ring>::zero();
        let o = <ExtensionField as Ring>::one();
        assert!(z.is_zero());
        assert!(o.is_one());
        // They bind on contact with a bound element (delegating to the
        // canonical FiniteFieldElement sentinel semantics).
        let irr = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        ]);
        let alpha = ExtensionField::new(
            UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(1)]),
            Integer::from(2),
            irr,
        )
        .unwrap();
        assert_eq!(z + alpha.clone(), alpha);
        assert_eq!(o * alpha.clone(), alpha);
        let z2 = <ExtensionField as Ring>::zero() * alpha.clone();
        assert!(z2.is_zero());
        assert_eq!(z2.degree(), 2); // bound into GF(2^2)
    }

    #[test]
    fn rejects_invalid_parameters() {
        // Wave 2: the wrapper now validates through FiniteField::with_modulus.
        let x2_plus_1 = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        ]); // reducible over F_2: (x+1)^2
        let one = UnivariatePolynomial::new(vec![Integer::from(1)]);
        assert!(ExtensionField::new(one.clone(), Integer::from(2), x2_plus_1).is_err());
        // non-prime characteristic
        let x2_x_1 = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        ]);
        assert!(ExtensionField::new(one, Integer::from(4), x2_x_1).is_err());
    }

    fn gf(coeffs: &[i64], p: i64, irr: &[i64]) -> ExtensionField {
        ExtensionField::new(
            UnivariatePolynomial::new(coeffs.iter().map(|&c| Integer::from(c)).collect()),
            Integer::from(p),
            UnivariatePolynomial::new(irr.iter().map(|&c| Integer::from(c)).collect()),
        )
        .unwrap()
    }

    fn coeffs(e: &ExtensionField, n: usize) -> Vec<i64> {
        let mut c: Vec<i64> = e.poly().coefficients().iter().map(|x| x.to_i64()).collect();
        c.resize(n, 0);
        c
    }

    #[test]
    fn frobenius_correct_in_gf4() {
        // F_4 = F_2[α]/(α²+α+1). Frobenius α ↦ α² = α+1 (NOT the old buggy [1,0]).
        let irr = [1, 1, 1];
        let alpha = gf(&[0, 1], 2, &irr); // α
        let fr = alpha.frobenius();
        assert_eq!(coeffs(&fr, 2), vec![1, 1]); // α + 1
        // φ² = identity (Gal(F_4/F_2) ≅ C_2).
        assert_eq!(coeffs(&fr.frobenius(), 2), vec![0, 1]);
        // φ fixes F_2: φ(1) = 1.
        assert_eq!(coeffs(&gf(&[1], 2, &irr).frobenius(), 2), vec![1, 0]);
    }

    #[test]
    fn multiplication_reduces_mod_irreducible() {
        // In F_4: α·α = α² = α+1 (must reduce, not truncate to 0).
        let irr = [1, 1, 1];
        let alpha = gf(&[0, 1], 2, &irr);
        let prod = alpha.clone() * alpha.clone();
        assert_eq!(coeffs(&prod, 2), vec![1, 1]);
        // α³ = 1 (α is a generator of F_4* of order 3).
        let cube = prod * alpha;
        assert_eq!(coeffs(&cube, 2), vec![1, 0]);
    }

    #[test]
    fn construction_reduces_input() {
        // Wave 2: inputs are reduced mod p and mod the irreducible on entry.
        // 3α² + 5 over F_2[α]/(α²+α+1): 3α² = α² = α+1, 5 = 1 → α + (1+1) = α.
        let irr = [1, 1, 1];
        let e = gf(&[5, 0, 3], 2, &irr);
        assert_eq!(coeffs(&e, 2), vec![0, 1]);
        // And equality is now field-aware on reduced forms.
        assert_eq!(e, gf(&[0, 1], 2, &irr));
    }

    #[test]
    fn frobenius_order_in_gf8() {
        // F_8 = F_2[α]/(α³+α+1). Frobenius has order 3; φ³ = id, φ ≠ id on α.
        let irr = [1, 1, 0, 1]; // x³ + x + 1
        let alpha = gf(&[0, 1], 2, &irr);
        let f1 = alpha.frobenius();
        assert_ne!(coeffs(&f1, 3), coeffs(&alpha, 3));
        assert_eq!(coeffs(&f1.frobenius().frobenius(), 3), coeffs(&alpha, 3)); // φ³ = id
        // Norm of a generator α: N(α) = α^(1+2+4) = α^7 = 1 (|F_8*| = 7).
        assert_eq!(alpha.norm(), Integer::from(1));
    }

    #[test]
    fn norm_and_trace_land_in_fp() {
        // F_4: N(α) = α·α² = α³ = 1; Tr(α) = α + α² = α + (α+1) = 1.
        let irr = [1, 1, 1];
        let alpha = gf(&[0, 1], 2, &irr);
        assert_eq!(alpha.norm(), Integer::from(1));
        assert_eq!(alpha.trace(), Integer::from(1));
    }

    /// Enumerate all `p^n` elements of GF(p^n) for the given irreducible.
    fn all_elements(p: i64, n: usize, irr: &[i64]) -> Vec<ExtensionField> {
        let mut out = Vec::new();
        let mut idx = vec![0i64; n];
        loop {
            out.push(gf(&idx, p, irr));
            // Increment idx as a base-p counter.
            let mut i = 0;
            loop {
                if i == n {
                    return out;
                }
                idx[i] += 1;
                if idx[i] == p {
                    idx[i] = 0;
                    i += 1;
                } else {
                    break;
                }
            }
        }
    }

    #[test]
    fn inverse_correct_in_gf8() {
        // F_8 = F_2[α]/(α³+α+1).
        let irr = [1, 1, 0, 1];
        let one = gf(&[1, 0, 0], 2, &irr);
        for e in all_elements(2, 3, &irr) {
            if e.is_zero() {
                assert!(e.inverse().is_err());
                continue;
            }
            let inv = e.inverse().unwrap();
            assert_eq!(e.clone() * inv.clone(), one, "a * a^-1 != 1 for a = {}", e);
            assert_eq!(inv.clone() * e.clone(), one, "a^-1 * a != 1 for a = {}", e);
            assert_eq!(e.clone() / e.clone(), one, "a / a != 1 for a = {}", e);
        }
    }

    #[test]
    fn inverse_correct_in_gf9() {
        // F_9 = F_3[α]/(α²+1).
        let irr = [1, 0, 1];
        let one = gf(&[1, 0], 3, &irr);
        for e in all_elements(3, 2, &irr) {
            if e.is_zero() {
                assert!(e.inverse().is_err());
                continue;
            }
            let inv = e.inverse().unwrap();
            assert_eq!(e.clone() * inv.clone(), one, "a * a^-1 != 1 for a = {}", e);
            assert_eq!(inv.clone() * e.clone(), one, "a^-1 * a != 1 for a = {}", e);
            assert_eq!(e.clone() / e.clone(), one, "a / a != 1 for a = {}", e);
        }
    }

    #[test]
    fn division_matches_multiplication_by_inverse() {
        // In F_4, check a / b == a * b.inverse() for a spread of nonzero a, b.
        let irr = [1, 1, 1];
        let alpha = gf(&[0, 1], 2, &irr); // α
        let alpha2 = alpha.clone() * alpha.clone(); // α + 1
        let one = gf(&[1, 0], 2, &irr);

        for a in [one.clone(), alpha.clone(), alpha2.clone()] {
            for b in [one.clone(), alpha.clone(), alpha2.clone()] {
                let quot = a.clone() / b.clone();
                let via_inv = a.clone() * b.inverse().unwrap();
                assert_eq!(quot, via_inv);
            }
        }
    }

    #[test]
    fn wrapper_agrees_with_canonical_element() {
        // The wrapper must expose exactly the canonical FiniteFieldElement.
        let irr = [1, 1, 0, 1];
        let alpha = gf(&[0, 1], 2, &irr);
        let field = alpha.field().clone();
        assert_eq!(*alpha.as_element(), field.generator());
        let via_canonical =
            ExtensionField::from_element(field.generator().pow_int(&Integer::from(5)));
        let via_wrapper = alpha.clone() * alpha.clone() * alpha.clone() * alpha.clone() * alpha;
        assert_eq!(via_canonical, via_wrapper);
    }
}
