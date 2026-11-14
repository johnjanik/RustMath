//! Real parent types for the three-valued ring predicates of
//! [`RingParentMethods`](crate::category_methods::RingParentMethods).
//!
//! MAGMA Handbook Ch. 17 §17.4.2-3 (`Characteristic`, `IsCommutative`, `IsFinite`,
//! `IsField`, `IsDomain`), Ch. 18 (Z), Ch. 19 (Z/mZ), Ch. 20 (Q), Ch. 21 (GF(q)), Ch. 23
//! (R\[x\]), Ch. 47 (p-adic rings and their extensions), Ch. 48 (Galois rings).
//! Algebra used: M. F. Atiyah and I. G. Macdonald, *Introduction to Commutative Algebra*
//! (Thm 7.5 Hilbert basis, Prop. 6.5, Prop. 7.1, Thm 8.5, Ex. 1.2, 5.9, 11.7); H. Matsumura,
//! *Commutative Ring Theory*, Thm 15.4; J.-P. Serre, *Local Fields*, Ch. II §1 (Z_p is a
//! complete DVR).
//!
//! Each impl overrides only what it decides; every other predicate keeps the trait default.
//! Calls go to the inherent methods by explicit path (`Zmod::characteristic(self)`): the
//! inherent methods share names with the trait (`characteristic`, `order`, `is_field`) and
//! win at concrete call sites, while generic code reaches the trait.
//!
//! Every `Some(_)` below is a theorem about the ring. The p-adic rows hold for every p > 1
//! (the constructors accept composite p): Z/p^n ≅ ∏_{q | p} Z/q^(a_q n) (CRT), and inverse
//! limits commute with finite products, so Z_p ≅ ∏_{q | p} Z_q and Q_p ≅ ∏_{q | p} Q_q.

use crate::category_methods::{padic_base_is_prime_power, proven_primality, RingParentMethods};
use crate::padics::{
    PadicExtension, PadicExtensionElement, PadicField, PadicInteger, PadicIntegerRing,
    PadicRational,
};
use rustmath_finitefields::{
    FiniteField, FiniteFieldElement, GaloisRing, GaloisRingElement, IntegerMod, Zmod,
};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

/// Z (MAGMA `Integers()`, Ch. 18).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntegerRingParent;

/// Q (MAGMA `Rationals()`, Ch. 20).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RationalFieldParent;

/// R\[x\] over a base parent (MAGMA `PolynomialRing(R)`, Ch. 23), with elements
/// `UnivariatePolynomial<P::Element>`.
#[derive(Clone, Debug)]
pub struct PolynomialRingParent<P: RingParentMethods> {
    base: P,
}

impl<P: RingParentMethods> PolynomialRingParent<P> {
    /// R\[x\] for the base R.
    pub fn new(base: P) -> Self {
        PolynomialRingParent { base }
    }

    /// The coefficient ring R.
    pub fn base(&self) -> &P {
        &self.base
    }
}

/// `Some(true)` when `b` is `Some(true)`, else `None` (a proven prime p decides, an
/// unresolved proof makes no claim).
fn only_if_proven(b: Option<bool>) -> Option<bool> {
    (b == Some(true)).then_some(true)
}

/// Z: a PID, not a field (2 has no inverse), Noetherian and integrally closed (a UFD),
/// Krull dimension 1 (the chains 0 ⊂ (p)). GAP: `[Characteristic, IsCommutative,
/// IsIntegralRing, IsField, IsFinite](Integers)` = `[0, true, true, false, false]`.
impl RingParentMethods for IntegerRingParent {
    type Element = Integer;
    fn characteristic(&self) -> Option<u64> {
        Some(0)
    }
    fn is_zero_ring(&self) -> Option<bool> {
        Some(false)
    }
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        Some(true)
    }
    fn is_field(&self) -> Option<bool> {
        Some(false)
    }
    fn is_integrally_closed(&self) -> Option<bool> {
        Some(true)
    }
    fn is_noetherian(&self) -> Option<bool> {
        Some(true)
    }
    fn krull_dimension(&self) -> Option<usize> {
        Some(1)
    }
}

/// Q: a field, the prime field of characteristic 0, dimension 0.
impl RingParentMethods for RationalFieldParent {
    type Element = Rational;
    fn characteristic(&self) -> Option<u64> {
        Some(0)
    }
    fn is_zero_ring(&self) -> Option<bool> {
        Some(false)
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
    fn is_prime_field(&self) -> Option<bool> {
        Some(true)
    }
    fn krull_dimension(&self) -> Option<usize> {
        Some(0)
    }
}

/// Z/m (m > 1, `Zmod::new`): finite and nonzero, so Artinian of dimension 0 (A-M Thm 8.5);
/// a domain iff a field iff m is prime (proven, MAGMA `IsPrime(m : Proof := true)`; an
/// unresolved proof gives `None`); then it is its own prime field. `characteristic`/`order`
/// are `None` above u64/usize; `is_zero_ring` and `is_finite` read the exact modulus.
impl RingParentMethods for Zmod {
    type Element = IntegerMod;
    fn order(&self) -> Option<usize> {
        usize::try_from(Zmod::order(self)).ok()
    }
    fn characteristic(&self) -> Option<u64> {
        u64::try_from(&Zmod::characteristic(self)).ok()
    }
    fn is_zero_ring(&self) -> Option<bool> {
        Some(false)
    }
    fn is_finite(&self) -> Option<bool> {
        Some(true)
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
    fn is_prime_field(&self) -> Option<bool> {
        proven_primality(self.modulus())
    }
    fn is_integrally_closed(&self) -> Option<bool> {
        only_if_proven(proven_primality(self.modulus()))
    }
    fn is_noetherian(&self) -> Option<bool> {
        Some(true)
    }
    fn krull_dimension(&self) -> Option<usize> {
        Some(0)
    }
}

/// GF(q), q = p^n: `FiniteField::new`/`with_modulus` prove p prime and the modulus
/// irreducible, so these rows read the proof of p again (cached by rustmath-integers above
/// psi_12): `Some(true)` only when p is proven. The prime field iff n = 1.
impl RingParentMethods for FiniteField {
    type Element = FiniteFieldElement;
    fn order(&self) -> Option<usize> {
        usize::try_from(&FiniteField::order(self)).ok()
    }
    fn characteristic(&self) -> Option<u64> {
        u64::try_from(FiniteField::characteristic(self)).ok()
    }
    fn is_zero_ring(&self) -> Option<bool> {
        Some(false)
    }
    fn is_finite(&self) -> Option<bool> {
        Some(true)
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
    fn is_prime_field(&self) -> Option<bool> {
        only_if_proven(proven_primality(FiniteField::characteristic(self)))
            .map(|_| self.degree() == 1)
    }
    fn is_integrally_closed(&self) -> Option<bool> {
        only_if_proven(proven_primality(FiniteField::characteristic(self)))
    }
    fn is_noetherian(&self) -> Option<bool> {
        Some(true)
    }
    fn krull_dimension(&self) -> Option<usize> {
        Some(0)
    }
}

/// GR(p^a, d) (MAGMA Ch. 48): finite of order p^(ad) and characteristic p^a. For a > 1, p is
/// nilpotent and nonzero (p^a = 0, p^(a-1) != 0), so it is not a domain; GR(p, d) = GF(p^d).
/// `GaloisRing::new` proves p prime; these rows read the proof again, as for GF(q).
impl RingParentMethods for GaloisRing {
    type Element = GaloisRingElement;
    fn order(&self) -> Option<usize> {
        usize::try_from(&self.cardinality()).ok()
    }
    fn characteristic(&self) -> Option<u64> {
        u64::try_from(GaloisRing::characteristic(self)).ok()
    }
    fn is_zero_ring(&self) -> Option<bool> {
        Some(false)
    }
    fn is_finite(&self) -> Option<bool> {
        Some(true)
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
    fn is_prime_field(&self) -> Option<bool> {
        if self.exponent() > 1 {
            Some(false)
        } else {
            only_if_proven(proven_primality(self.prime())).map(|_| self.degree() == 1)
        }
    }
    fn is_integrally_closed(&self) -> Option<bool> {
        if self.exponent() > 1 {
            None
        } else {
            only_if_proven(proven_primality(self.prime()))
        }
    }
    fn is_noetherian(&self) -> Option<bool> {
        Some(true)
    }
    fn krull_dimension(&self) -> Option<usize> {
        Some(0)
    }
}

/// Z_p for every p > 1: Z_p ≅ ∏_{q | p} Z_q, a finite product of complete DVRs (Serre II
/// §1). So: characteristic 0 (Sage `characteristic()` is 0, also for the fixed-mod model),
/// infinite, nonzero, commutative, reduced, Noetherian (a finite product of Noetherian
/// rings), Krull dimension 1 (Spec of a finite product is the disjoint union of the factors'
/// spectra), not a field (p is a nonzero non-unit). A domain iff exactly one prime divides
/// p, decided by the minimal root of p; then it is a DVR, integrally closed. For a
/// non-domain no claim on integral closure (the notion is for domains).
impl RingParentMethods for PadicIntegerRing {
    type Element = PadicInteger;
    fn characteristic(&self) -> Option<u64> {
        Some(0)
    }
    fn is_zero_ring(&self) -> Option<bool> {
        Some(false)
    }
    fn is_finite(&self) -> Option<bool> {
        Some(false)
    }
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        padic_base_is_prime_power(self.prime())
    }
    fn is_field(&self) -> Option<bool> {
        Some(false)
    }
    fn is_integrally_closed(&self) -> Option<bool> {
        only_if_proven(padic_base_is_prime_power(self.prime()))
    }
    fn is_noetherian(&self) -> Option<bool> {
        Some(true)
    }
    fn krull_dimension(&self) -> Option<usize> {
        Some(1)
    }
}

/// Q_p = Z_p\[1/p\] ≅ ∏_{q | p} Q_q: a field iff p is a prime power; Noetherian of dimension
/// 0; never Q (uncountable when a field), so never a prime field.
impl RingParentMethods for PadicField {
    type Element = PadicRational;
    fn characteristic(&self) -> Option<u64> {
        Some(0)
    }
    fn is_zero_ring(&self) -> Option<bool> {
        Some(false)
    }
    fn is_finite(&self) -> Option<bool> {
        Some(false)
    }
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        padic_base_is_prime_power(self.prime())
    }
    fn is_field(&self) -> Option<bool> {
        padic_base_is_prime_power(self.prime())
    }
    fn is_prime_field(&self) -> Option<bool> {
        Some(false)
    }
    fn is_noetherian(&self) -> Option<bool> {
        Some(true)
    }
    fn krull_dimension(&self) -> Option<usize> {
        Some(0)
    }
}

/// K = Q_p\[x\]/(f): a finitely generated module over the Artinian ring Q_p (a finite product
/// of fields), so an Artinian ring (A-M Prop. 6.5), hence Noetherian of dimension 0 (A-M Thm
/// 8.5), for every f. [`PadicExtension::new`] refuses a constant f, so deg f >= 1 and Q_p
/// embeds in K (a nonzero constant is not a multiple of f): K is nonzero, infinite and of
/// characteristic 0. A field (and a domain) iff [`PadicExtension::field_status`]; not Q, so
/// not a prime field.
impl RingParentMethods for PadicExtension {
    type Element = PadicExtensionElement;
    fn characteristic(&self) -> Option<u64> {
        Some(0)
    }
    fn is_zero_ring(&self) -> Option<bool> {
        Some(false)
    }
    fn is_finite(&self) -> Option<bool> {
        Some(false)
    }
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }
    fn is_integral_domain(&self) -> Option<bool> {
        self.field_status()
    }
    fn is_field(&self) -> Option<bool> {
        self.field_status()
    }
    fn is_prime_field(&self) -> Option<bool> {
        Some(false)
    }
    fn is_noetherian(&self) -> Option<bool> {
        Some(true)
    }
    fn krull_dimension(&self) -> Option<usize> {
        Some(0)
    }
}

/// R\[x\]: R ⊂ R\[x\] gives the characteristic and, with x central, commutativity iff R is;
/// R\[x\] = 0 iff R = 0, and R\[x\] is finite iff R = 0; R\[x\] is a domain iff R is (leading
/// coefficients); x is not a unit and R = 0 gives R\[x\] = 0, so R\[x\] is never a field;
/// Noetherian iff R is (Hilbert basis theorem, A-M Thm 7.5; R ≅ R\[x\]/(x), A-M Prop. 7.1);
/// for a domain R, R\[x\] is integrally closed iff R is (A-M Ex. 5.9, and R = R\[x\] ∩
/// Frac(R)); for a commutative Noetherian R, dim R\[x\] = dim R + 1 (A-M Ex. 11.7, Matsumura
/// Thm 15.4; both are commutative statements). Without Noetherianity the dimension is only
/// bounded (Seidenberg), so `None`. These methods read only the base's predicates.
impl<P: RingParentMethods> RingParentMethods for PolynomialRingParent<P> {
    type Element = UnivariatePolynomial<P::Element>;
    fn order(&self) -> Option<usize> {
        (self.base.is_zero_ring() == Some(true)).then_some(1)
    }
    fn characteristic(&self) -> Option<u64> {
        self.base.characteristic()
    }
    fn is_zero_ring(&self) -> Option<bool> {
        self.base.is_zero_ring()
    }
    fn is_finite(&self) -> Option<bool> {
        self.base.is_zero_ring()
    }
    fn is_commutative(&self) -> Option<bool> {
        self.base.is_commutative()
    }
    fn is_integral_domain(&self) -> Option<bool> {
        self.base.is_integral_domain()
    }
    fn is_field(&self) -> Option<bool> {
        Some(false)
    }
    fn is_integrally_closed(&self) -> Option<bool> {
        if self.base.is_integral_domain() == Some(true) {
            self.base.is_integrally_closed()
        } else {
            None
        }
    }
    fn is_noetherian(&self) -> Option<bool> {
        self.base.is_noetherian()
    }
    fn krull_dimension(&self) -> Option<usize> {
        if self.base.is_noetherian() == Some(true)
            && self.base.is_zero_ring() == Some(false)
            && self.base.is_commutative() == Some(true)
        {
            self.base.krull_dimension().map(|d| d + 1)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::padics::{ExtensionType, PrecisionModel};

    /// The tuple (char, zero ring, commutative, domain, field, prime field, finite,
    /// Noetherian, integrally closed, Krull) of a parent.
    type Profile = (
        Option<u64>,
        Option<bool>,
        Option<bool>,
        Option<bool>,
        Option<bool>,
        Option<bool>,
        Option<bool>,
        Option<bool>,
        Option<bool>,
        Option<usize>,
    );
    fn profile<P: RingParentMethods>(r: &P) -> Profile {
        (
            r.characteristic(),
            r.is_zero_ring(),
            r.is_commutative(),
            r.is_integral_domain(),
            r.is_field(),
            r.is_prime_field(),
            r.is_finite(),
            r.is_noetherian(),
            r.is_integrally_closed(),
            r.krull_dimension(),
        )
    }
    fn big_prime() -> Integer {
        Integer::from(2u64).pow(64) + Integer::from(13) // PARI isprime(2^64+13) -> 1
    }

    // GAP: [Characteristic, IsCommutative, IsIntegralRing, IsField, IsFinite](Integers)
    // -> [0, true, true, false, false]. At c79a582 no parent type for Z existed.
    #[test]
    fn integer_ring_parent() {
        let z = IntegerRingParent;
        assert_eq!(
            profile(&z),
            (
                Some(0),
                Some(false),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(1)
            )
        );
    }

    // GAP: [Characteristic, IsCommutative, IsIntegralRing, IsField, IsFinite](Rationals)
    // -> [0, true, true, true, false]; IsPrimeField(Rationals) -> true.
    #[test]
    fn rational_field_parent() {
        let q = RationalFieldParent;
        assert_eq!(
            profile(&q),
            (
                Some(0),
                Some(false),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                Some(0)
            )
        );
    }

    // GAP: [Size, IsField](ZmodnZ(n)) for n = 2, 3, 4, 6, 7, 8 -> [2, true], [3, true],
    // [4, false], [6, false], [7, true], [8, false]; IsField(ZmodnZ(2^64+13)) -> true,
    // IsPrime(2^64+13) -> true; IsField(ZmodnZ(3*2^64)) -> false.
    #[test]
    fn zmod_parents() {
        for (n, field) in [
            (2u64, true),
            (3, true),
            (4, false),
            (6, false),
            (7, true),
            (8, false),
        ] {
            let r = Zmod::new(Integer::from(n)).unwrap();
            assert_eq!(RingParentMethods::characteristic(&r), Some(n));
            assert_eq!(RingParentMethods::order(&r), Some(n as usize));
            assert_eq!(RingParentMethods::is_field(&r), Some(field), "Z/{n}");
            assert_eq!(r.is_integral_domain(), Some(field), "Z/{n}");
            assert_eq!(r.is_prime_field(), Some(field), "Z/{n}");
            assert_eq!(r.krull_dimension(), Some(0));
            assert_eq!(r.is_commutative(), Some(true));
        }
        let big = Zmod::new(big_prime()).unwrap();
        assert_eq!(RingParentMethods::characteristic(&big), None);
        assert_eq!(RingParentMethods::order(&big), None);
        assert_eq!(big.is_zero_ring(), Some(false));
        assert_eq!(big.is_finite(), Some(true));
        assert_eq!(RingParentMethods::is_field(&big), Some(true));
        let composite = Zmod::new(Integer::from(3u64) * Integer::from(2u64).pow(64)).unwrap();
        assert_eq!(RingParentMethods::is_field(&composite), Some(false));
        assert_eq!(composite.krull_dimension(), Some(0));
    }

    // GAP: [Characteristic, IsCommutative, IsIntegralRing, IsField, IsFinite](GF(25))
    // -> [5, true, true, true, true]; IsPrimeField(GF(25)) -> false, IsPrimeField(GF(7))
    // -> true.
    #[test]
    fn finite_field_parents() {
        let f25 = FiniteField::new(Integer::from(5), 2).unwrap();
        assert_eq!(
            profile(&f25),
            (
                Some(5),
                Some(false),
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                Some(true),
                Some(0)
            )
        );
        assert_eq!(RingParentMethods::order(&f25), Some(25));
        let f7 = FiniteField::new(Integer::from(7), 1).unwrap();
        assert_eq!(RingParentMethods::is_prime_field(&f7), Some(true));
        let big = FiniteField::new(big_prime(), 1).unwrap();
        assert_eq!(RingParentMethods::characteristic(&big), None);
        assert_eq!(RingParentMethods::is_field(&big), Some(true));
        assert_eq!(RingParentMethods::is_prime_field(&big), Some(true));
        assert_eq!(big.is_zero_ring(), Some(false));
    }

    // GR(8, 2) (MAGMA H48E1): char 8, order 64, not a domain (PARI Mod(2,8)^3 == 0);
    // GR(5, 3) = GF(125), GR(7, 1) = GF(7).
    #[test]
    fn galois_ring_parents() {
        let gr = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        assert_eq!(RingParentMethods::characteristic(&gr), Some(8));
        assert_eq!(RingParentMethods::order(&gr), Some(64));
        assert_eq!(gr.is_integral_domain(), Some(false));
        assert_eq!(RingParentMethods::is_field(&gr), Some(false));
        assert_eq!(gr.is_noetherian(), Some(true));
        assert_eq!(gr.krull_dimension(), Some(0));
        assert_eq!(gr.is_integrally_closed(), None);
        let gr53 = GaloisRing::new(Integer::from(5), 1, 3).unwrap();
        assert_eq!(RingParentMethods::is_field(&gr53), Some(true));
        assert_eq!(gr53.is_prime_field(), Some(false));
        let gr71 = GaloisRing::new(Integer::from(7), 1, 1).unwrap();
        assert_eq!(gr71.is_prime_field(), Some(true));
    }

    // Sage: Zp(3,10,'fixed-mod').characteristic() -> 0. Z_6 is not a domain: gp
    // e = lift(chinese(Mod(1,2^10),Mod(0,3^10))) = 24151041 has e*(1-e) % 6^10 = 0, with
    // e ≡ 3 and 1 - e ≡ 4 mod 6, so e and 1 - e are nonzero zero divisors. Z_4 = Z_2
    // (gp isprimepower(4) -> 2).
    #[test]
    fn padic_parents() {
        for model in [PrecisionModel::CappedRelative, PrecisionModel::FixedModulus] {
            let z5 = PadicIntegerRing::new(Integer::from(5), 20, model).unwrap();
            assert_eq!(
                profile(&z5),
                (
                    Some(0),
                    Some(false),
                    Some(true),
                    Some(true),
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(true),
                    Some(true),
                    Some(1)
                )
            );
        }
        let q5 = PadicField::new(Integer::from(5), 20, PrecisionModel::CappedRelative).unwrap();
        assert_eq!(q5.is_field(), Some(true));
        assert_eq!(q5.is_prime_field(), Some(false));
        assert_eq!(q5.krull_dimension(), Some(0));

        let z6 =
            PadicIntegerRing::new(Integer::from(6), 10, PrecisionModel::CappedRelative).unwrap();
        assert_eq!(z6.is_integral_domain(), Some(false));
        assert_eq!(z6.is_field(), Some(false));
        assert_eq!(z6.is_noetherian(), Some(true));
        assert_eq!(z6.krull_dimension(), Some(1));
        assert_eq!(z6.is_integrally_closed(), None);
        let q6 = PadicField::new(Integer::from(6), 10, PrecisionModel::CappedRelative).unwrap();
        assert_eq!(q6.is_field(), Some(false));
        assert_eq!(q6.is_integral_domain(), Some(false));
        assert_eq!(q6.krull_dimension(), Some(0));
        assert_eq!(q6.is_prime_field(), Some(false));

        let z4 =
            PadicIntegerRing::new(Integer::from(4), 10, PrecisionModel::CappedRelative).unwrap();
        assert_eq!(z4.is_integral_domain(), Some(true));
        assert_eq!(z4.is_integrally_closed(), Some(true));
        let q4 = PadicField::new(Integer::from(4), 10, PrecisionModel::CappedRelative).unwrap();
        assert_eq!(q4.is_field(), Some(true));
    }

    // gp: #factorpadic(x^2-1,5,10)[,1] -> 2; #factorpadic(x^2-125,5,10)[,1] -> 1;
    // issquarefree((x-1)^2) -> 0; #factorpadic(x^2-5,2,10)[,1] -> 1.
    #[test]
    fn padic_extension_parents() {
        let poly =
            |cs: &[i64]| UnivariatePolynomial::new(cs.iter().map(|&c| Integer::from(c)).collect());
        let new = |p: i64, cs: &[i64]| {
            PadicExtension::new(Integer::from(p), poly(cs), 10, ExtensionType::Unramified).unwrap()
        };
        let k = PadicExtension::eisenstein(Integer::from(5), poly(&[-5, 0, 1]), 10).unwrap();
        assert_eq!(k.is_field(), Some(true));
        assert_eq!(k.is_noetherian(), Some(true));
        assert_eq!(k.krull_dimension(), Some(0));
        assert_eq!(k.is_prime_field(), Some(false));
        assert_eq!(k.characteristic(), Some(0));
        assert_eq!(new(5, &[-1, 0, 1]).is_field(), Some(false));
        let sq = new(5, &[1, -2, 1]);
        assert_eq!(sq.is_field(), Some(false));
        assert_eq!(sq.is_noetherian(), Some(true));
        assert_eq!(sq.krull_dimension(), Some(0));
        assert_eq!(new(5, &[-125, 0, 1]).is_field(), Some(true));
        assert_eq!(new(4, &[-5, 0, 1]).is_field(), Some(true));
    }

    /// The zero ring as a base (characteristic 1).
    #[derive(Clone, Debug)]
    struct TestZeroRing;
    impl RingParentMethods for TestZeroRing {
        type Element = IntegerMod;
        fn characteristic(&self) -> Option<u64> {
            Some(1)
        }
    }
    /// A base that is Noetherian of dimension 0 but of unknown commutativity.
    #[derive(Clone, Debug)]
    struct NoetherianUnknownCommutativity;
    impl RingParentMethods for NoetherianUnknownCommutativity {
        type Element = IntegerMod;
        fn characteristic(&self) -> Option<u64> {
            Some(5)
        }
        fn is_noetherian(&self) -> Option<bool> {
            Some(true)
        }
        fn krull_dimension(&self) -> Option<usize> {
            Some(0)
        }
    }

    // GAP: [Characteristic, IsCommutative, IsField, IsFinite](PolynomialRing(Integers, 1))
    // -> [0, true, false, false]; ... (PolynomialRing(GF(4), 1)) -> [2, true, false, false];
    // ... (PolynomialRing(Rationals, 2)) -> [0, true, false, false];
    // IsFinite(PolynomialRing(ZmodnZ(6), 1)) -> false; IsFinite(PolynomialRing(ZmodnZ(2^64+13),
    // 1)) -> false. PARI: Mod(2,6)*Mod(3,6) -> Mod(0, 6).
    #[test]
    fn polynomial_ring_parents() {
        let zx = PolynomialRingParent::new(IntegerRingParent);
        assert_eq!(
            profile(&zx),
            (
                Some(0),
                Some(false),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(2)
            )
        );
        let f4x = PolynomialRingParent::new(FiniteField::new(Integer::from(2), 2).unwrap());
        assert_eq!(f4x.characteristic(), Some(2));
        assert_eq!(f4x.is_commutative(), Some(true));
        assert_eq!(f4x.is_field(), Some(false));
        assert_eq!(f4x.is_finite(), Some(false));
        assert_eq!(f4x.is_integral_domain(), Some(true));
        assert_eq!(f4x.krull_dimension(), Some(1));
        let qxy = PolynomialRingParent::new(PolynomialRingParent::new(RationalFieldParent));
        assert_eq!(qxy.krull_dimension(), Some(2));
        assert_eq!(qxy.is_integral_domain(), Some(true));
        assert_eq!(qxy.is_finite(), Some(false));
        let z6x = PolynomialRingParent::new(Zmod::new(Integer::from(6)).unwrap());
        assert_eq!(z6x.characteristic(), Some(6));
        assert_eq!(z6x.is_finite(), Some(false));
        assert_eq!(z6x.is_integral_domain(), Some(false));
        assert_eq!(z6x.krull_dimension(), Some(1));
        let bigx = PolynomialRingParent::new(Zmod::new(big_prime()).unwrap());
        assert_eq!(bigx.is_zero_ring(), Some(false));
        assert_eq!(bigx.is_finite(), Some(false));
        assert_eq!(bigx.is_integral_domain(), Some(true));
        assert_eq!(bigx.krull_dimension(), Some(1));
        let zero_x = PolynomialRingParent::new(TestZeroRing);
        assert_eq!(zero_x.order(), Some(1));
        assert_eq!(zero_x.is_finite(), Some(true));
        assert_eq!(zero_x.is_field(), Some(false));
        assert_eq!(zero_x.is_integral_domain(), Some(false));
        assert_eq!(zero_x.krull_dimension(), None);
        let unknown_x = PolynomialRingParent::new(NoetherianUnknownCommutativity);
        assert_eq!(unknown_x.krull_dimension(), None);
    }
}
