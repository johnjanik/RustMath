//! [`RingParentMethods`] and [`CommutativeRingParentMethods`] for the concrete ring parents
//! of the crates this one depends on: `Zmod`, `FiniteField`, `GaloisRing`
//! (rustmath-finitefields), `MatrixAlgebra<F>` (rustmath-matrix) and `BigFloatField`
//! (rustmath-reals).
//!
//! MAGMA Handbook Ch. 17 §17.4.2-3 (`Characteristic`, `IsCommutative`, `IsDomain`,
//! `IsField`, `IsEuclideanDomain`, `IsPID`, `IsUFD`), Ch. 19 (Z/mZ), Ch. 21 (GF(q)), Ch. 25
//! §25.3.2-3 (real fields), Ch. 48 (Galois rings), Ch. 83 (matrix algebras). Every
//! `Some(_)` is a theorem; the `CommutativeRingParentMethods` predicates keep their
//! defaults (field ⇒ Euclidean ⇒ PID ⇒ UFD; a non-domain is none of them).
//!
//! Inherent methods with the same names (`Zmod::characteristic`, `GaloisRing::is_field`)
//! are called by explicit path; at a concrete call site they win over the trait.

use crate::ring_category::{proven_primality, CommutativeRingParentMethods, RingParentMethods};
use rustmath_core::Field;
use rustmath_finitefields::{FiniteField, GaloisRing, Zmod};
use rustmath_matrix::MatrixAlgebra;
use rustmath_reals::BigFloatField;

/// `Some(true)` when `b` is `Some(true)`, else `None`.
fn only_if_proven(b: Option<bool>) -> Option<bool> {
    (b == Some(true)).then_some(true)
}

/// Z/m (m > 1): commutative; a domain iff a field iff m is prime (proven; an unresolved
/// proof gives `None`). The characteristic is `None` above u64.
impl RingParentMethods for Zmod {
    fn characteristic(&self) -> Option<u64> {
        u64::try_from(&Zmod::characteristic(self)).ok()
    }
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        proven_primality(self.modulus())
    }
    fn is_field(&self) -> Option<bool> {
        proven_primality(self.modulus())
    }
}
impl CommutativeRingParentMethods for Zmod {}

/// GF(q): the constructors prove the characteristic prime; the rows read that proof again
/// (cached above psi_12 by rustmath-integers) and claim a field only when it holds.
impl RingParentMethods for FiniteField {
    fn characteristic(&self) -> Option<u64> {
        u64::try_from(FiniteField::characteristic(self)).ok()
    }
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        only_if_proven(proven_primality(FiniteField::characteristic(self)))
    }
    fn is_field(&self) -> Option<bool> {
        only_if_proven(proven_primality(FiniteField::characteristic(self)))
    }
}
impl CommutativeRingParentMethods for FiniteField {}

/// GR(p^a, d): commutative of characteristic p^a; for a > 1 the element p is nilpotent and
/// nonzero (not a domain); GR(p, d) = GF(p^d).
impl RingParentMethods for GaloisRing {
    fn characteristic(&self) -> Option<u64> {
        u64::try_from(GaloisRing::characteristic(self)).ok()
    }
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        if self.exponent() > 1 {
            Some(false)
        } else {
            only_if_proven(proven_primality(self.prime()))
        }
    }
    fn is_field(&self) -> Option<bool> {
        RingParentMethods::is_integral_domain(self)
    }
}
impl CommutativeRingParentMethods for GaloisRing {}

/// A matrix algebra A ⊆ M_n(F) (MAGMA Ch. 83). Decided only from structure:
///
/// | | char | commutative | domain, field |
/// |---|---|---|---|
/// | n = 0 (the zero algebra) | 1 | yes | no |
/// | full, n = 1 (A = F) | `F::static_characteristic()` | yes | yes |
/// | full, n >= 2 | `F::static_characteristic()` | no: E_12 E_21 = E_11 != E_22 = E_21 E_12 | no: E_11 E_22 = 0 |
/// | a generated subalgebra | `F::static_characteristic()` (F·I ⊆ A) | `None` | `None` |
///
/// "Full" is [`MatrixAlgebra::is_full`] (built by `MatrixAlgebra::full`), never a computed
/// dimension: over an inexact F (f64, RealMPFR) rounding can change the dimension that
/// `generated` computes, and a proper subalgebra can be commutative or a field (Q(i) inside
/// M_2(Q)). The full n = 1 row relies on the `Field` impl of F being true for every value
/// (for `PadicRational` with a composite p it is not; see the HANDOFF).
impl<F: Field> RingParentMethods for MatrixAlgebra<F> {
    fn characteristic(&self) -> Option<u64> {
        if self.degree() == 0 {
            Some(1)
        } else {
            F::static_characteristic()
        }
    }
    fn is_commutative(&self) -> Option<bool> {
        match (self.degree(), self.is_full()) {
            (0, _) => Some(true),
            (1, true) => Some(true),
            (_, true) => Some(false),
            (_, false) => None,
        }
    }
    fn is_integral_domain(&self) -> Option<bool> {
        match (self.degree(), self.is_full()) {
            (0, _) => Some(false),
            (1, true) => Some(true),
            (_, true) => Some(false),
            (_, false) => None,
        }
    }
    fn is_field(&self) -> Option<bool> {
        RingParentMethods::is_integral_domain(self)
    }
}
impl<F: Field> CommutativeRingParentMethods for MatrixAlgebra<F> {}

/// `RealField(prec)` (MAGMA Ch. 25 §25.3.2-3: `Characteristic` 0, `IsCommutative`,
/// `IsDomain`, `IsField` true; Sage `RealField(53).is_field()` is True).
impl RingParentMethods for BigFloatField {
    fn characteristic(&self) -> Option<u64> {
        Some(0)
    }
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        Some(true)
    }
    fn is_field(&self) -> Option<bool> {
        Some(true)
    }
}
impl CommutativeRingParentMethods for BigFloatField {}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_finitefields::FiniteFieldElement;
    use rustmath_integers::Integer;
    use rustmath_matrix::Matrix;
    use rustmath_rationals::Rational;

    fn kleene<P: CommutativeRingParentMethods>(r: &P) -> [Option<bool>; 5] {
        [
            r.is_integral_domain(),
            r.is_field(),
            r.is_pid(),
            r.is_ufd(),
            r.is_euclidean(),
        ]
    }

    // GAP: IsField(ZmodnZ(6)) -> false, IsField(ZmodnZ(7)) -> true.
    #[test]
    fn zmod_category() {
        let z6 = Zmod::new(Integer::from(6)).unwrap();
        assert_eq!(kleene(&z6), [Some(false); 5]);
        assert_eq!(RingParentMethods::characteristic(&z6), Some(6));
        let z7 = Zmod::new(Integer::from(7)).unwrap();
        assert_eq!(kleene(&z7), [Some(true); 5]);
        assert_eq!(z7.is_commutative(), Some(true));
    }

    // GAP: [IsField, IsEuclideanRing, IsUniqueFactorizationRing](GF(25)) -> all true.
    #[test]
    fn finite_field_category() {
        let f25 = FiniteField::new(Integer::from(5), 2).unwrap();
        assert_eq!(kleene(&f25), [Some(true); 5]);
        assert_eq!(RingParentMethods::characteristic(&f25), Some(5));
    }

    // GR(8, 2): PARI Mod(2,8)^3 -> Mod(0, 8), not a domain; GR(5, 3) = GF(125).
    #[test]
    fn galois_ring_category() {
        let gr = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        assert_eq!(RingParentMethods::is_field(&gr), Some(false));
        assert_eq!(gr.is_pid(), Some(false));
        assert_eq!(RingParentMethods::characteristic(&gr), Some(8));
        let gr53 = GaloisRing::new(Integer::from(5), 1, 3).unwrap();
        assert_eq!(RingParentMethods::is_field(&gr53), Some(true));
    }

    // GAP: [Characteristic, IsCommutative](FullMatrixAlgebra(Rationals, 2)) -> [0, false];
    // Basis(A)[1]*Basis(A)[4] (E11*E22) -> the zero matrix;
    // IsCommutative(FullMatrixAlgebra(Rationals, 1)) -> true;
    // [IsCommutative, Characteristic](FullMatrixAlgebra(GF(5), 2)) -> [false, 5] (this
    // parent cannot see p: FiniteFieldElement has no static characteristic).
    #[test]
    fn matrix_algebra_category() {
        let m2 = MatrixAlgebra::<Rational>::full(2);
        assert_eq!(m2.characteristic(), Some(0));
        assert_eq!(m2.is_commutative(), Some(false));
        assert_eq!(kleene(&m2), [Some(false); 5]);
        let m1 = MatrixAlgebra::<Rational>::full(1);
        assert_eq!(m1.is_commutative(), Some(true));
        assert_eq!(m1.is_integral_domain(), Some(true));
        assert_eq!(m1.is_field(), Some(true));
        let m0 = MatrixAlgebra::<Rational>::full(0);
        assert_eq!(m0.characteristic(), Some(1));
        assert_eq!(m0.is_integral_domain(), Some(false));
        let f2 = MatrixAlgebra::<FiniteFieldElement>::full(2);
        assert_eq!(f2.characteristic(), None);
        assert_eq!(f2.is_commutative(), Some(false));
        let r = |n: i64| Rational::from_integer(n);
        let diag = Matrix::from_vec(2, 2, vec![r(1), r(0), r(0), r(2)]).unwrap();
        let d = MatrixAlgebra::generated(2, vec![diag]).unwrap();
        assert_eq!(d.dimension(), 2);
        assert_eq!(d.characteristic(), Some(0));
        assert_eq!(d.is_commutative(), None);
        assert_eq!(d.is_field(), None);
        // generated(2, []) is Q·I (dimension 1); not structurally full, so no claim.
        let scalars = MatrixAlgebra::<Rational>::generated(2, vec![]).unwrap();
        assert_eq!(scalars.dimension(), 1);
        assert!(!scalars.is_full());
        assert_eq!(scalars.is_field(), None);
    }

    // MAGMA Ch. 25 §25.3.2-3: Characteristic 0, IsCommutative, IsField, IsDomain,
    // IsEuclideanDomain, IsPrincipalIdealRing all true for a real field.
    #[test]
    fn big_float_field_category() {
        let r = BigFloatField::new(53);
        assert_eq!(r.characteristic(), Some(0));
        assert_eq!(r.is_commutative(), Some(true));
        assert_eq!(kleene(&r), [Some(true); 5]);
    }
}
