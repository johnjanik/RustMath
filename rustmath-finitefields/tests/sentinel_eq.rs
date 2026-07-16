//! Integration probes for the coercing (binding) equality of the modulus-0 /
//! unbound sentinels (`Ring::zero()`/`Ring::one()`).
//!
//! History: the sentinel design originally shipped with STRICT `PartialEq`
//! (value AND modulus), which made every `x == F::zero()` in Field-generic
//! consumer code a false-negative on bound elements. The concrete end-to-end
//! failure was `rustmath_matrix::berlekamp_massey`:
//!
//! * `berlekamp_massey_verify::<PrimeField>` on `[1, 2]` over GF(7) returned
//!   `Ok((poly, verified = false))` — a silent lie (the poly is correct and
//!   does generate the sequence).
//! * `berlekamp_massey::<PrimeField>` on a genuine LFSR sequence (Fibonacci
//!   mod 7) hit the wrong branch on a zero discrepancy (`d == F::zero()` was
//!   false for the *bound* zero) and died with `Err(DivisionByZero)` when it
//!   later tried to invert that zero.
//!
//! The fix: `PartialEq` now binds on compare, exactly like the arithmetic —
//! `unbound(v) == bound(w mod m)` iff `v mod m == w`; unbound vs unbound
//! compares in Z; bound vs bound is unchanged (value AND modulus). These
//! tests pin both the equality semantics (for all four element types) and
//! the berlekamp_massey end-to-end behavior.

use rustmath_core::Ring;
use rustmath_finitefields::{
    ExtensionField, FiniteField, FiniteFieldElement, IntegerMod, PrimeField,
};
use rustmath_integers::Integer;
use rustmath_matrix::berlekamp_massey::{berlekamp_massey, berlekamp_massey_verify};
use rustmath_polynomials::UnivariatePolynomial;

fn gf7(v: i64) -> PrimeField {
    PrimeField::new(Integer::from(v), Integer::from(7)).unwrap()
}

fn zmod7(v: i64) -> IntegerMod {
    IntegerMod::new(Integer::from(v), Integer::from(7)).unwrap()
}

/// The unbound PrimeField sentinel with value `v` (built through sentinel
/// arithmetic only, never touching a modulus).
fn unbound_pf(v: i64) -> PrimeField {
    let mut x = PrimeField::zero();
    let step = if v >= 0 {
        PrimeField::one()
    } else {
        -PrimeField::one()
    };
    for _ in 0..v.abs() {
        x = x + step.clone();
    }
    x
}

fn unbound_im(v: i64) -> IntegerMod {
    let mut x = <IntegerMod as Ring>::zero();
    let step = if v >= 0 {
        <IntegerMod as Ring>::one()
    } else {
        -<IntegerMod as Ring>::one()
    };
    for _ in 0..v.abs() {
        x = x + step.clone();
    }
    x
}

fn unbound_ffe(v: i64) -> FiniteFieldElement {
    let mut x = FiniteFieldElement::zero();
    let step = if v >= 0 {
        FiniteFieldElement::one()
    } else {
        -FiniteFieldElement::one()
    };
    for _ in 0..v.abs() {
        x = x + step.clone();
    }
    x
}

fn unbound_ext(v: i64) -> ExtensionField {
    let mut x = <ExtensionField as Ring>::zero();
    let step = if v >= 0 {
        <ExtensionField as Ring>::one()
    } else {
        -<ExtensionField as Ring>::one()
    };
    for _ in 0..v.abs() {
        x = x + step.clone();
    }
    x
}

// ---------------------------------------------------------------------------
// Gate 1a: berlekamp_massey_verify over GF(7) on [1, 2].
//
// The minimal LFSR for [1, 2] is s_i = 2 s_{i-1}: connection polynomial
// 1 - 2x = 1 + 5x over GF(7) (convention: c_0 s_i + c_1 s_{i-1} = 0, c_0 = 1).
// Pre-fix this returned verified = false — a silent lie.
// ---------------------------------------------------------------------------
#[test]
fn berlekamp_massey_verify_gf7_no_silent_lie() {
    let (poly, verified) = berlekamp_massey_verify::<PrimeField>(vec![gf7(1), gf7(2)]).unwrap();
    // Independently derived: 1 + 5x (i.e. 1 - 2x mod 7).
    assert_eq!(poly.degree(), Some(1), "minimal LFSR for [1,2] has degree 1");
    assert_eq!(
        poly,
        UnivariatePolynomial::new(vec![gf7(1), gf7(5)]),
        "connection polynomial must be 1 + 5x over GF(7), got {poly}"
    );
    assert!(
        verified,
        "the returned polynomial genuinely generates the sequence; verified=false is a lie"
    );
}

// ---------------------------------------------------------------------------
// Gate 1b: berlekamp_massey over GF(7) on Fibonacci mod 7.
//
// s = [1, 1, 2, 3, 5, 8, 13, 21] mod 7 = [1, 1, 2, 3, 5, 1, 6, 0], satisfying
// s_n = s_{n-1} + s_{n-2}. With this implementation's convention
// (c_0 s_i + c_1 s_{i-1} + c_2 s_{i-2} = 0, c_0 = 1) the minimal connection
// polynomial is 1 - x - x^2 = 1 + 6x + 6x^2 over GF(7).
// Pre-fix this returned Err(DivisionByZero) (wrong branch on a bound zero
// discrepancy at line 96, then inverted that zero at line 122).
// ---------------------------------------------------------------------------
#[test]
fn berlekamp_massey_gf7_fibonacci_recovers_true_lfsr() {
    let seq: Vec<PrimeField> = [1, 1, 2, 3, 5, 1, 6, 0].iter().map(|&v| gf7(v)).collect();
    let poly = berlekamp_massey(seq.clone())
        .expect("genuine LFSR sequence must not error (pre-fix: DivisionByZero)");
    assert_eq!(poly.degree(), Some(2), "Fibonacci needs a degree-2 LFSR");
    assert_eq!(
        poly,
        UnivariatePolynomial::new(vec![gf7(1), gf7(6), gf7(6)]),
        "connection polynomial must be 1 + 6x + 6x^2 (= 1 - x - x^2) over GF(7), got {poly}"
    );
    // And the verifying variant agrees end-to-end.
    let (poly2, verified) = berlekamp_massey_verify(seq).unwrap();
    assert_eq!(poly2, poly);
    assert!(verified);
}

// ---------------------------------------------------------------------------
// Gate 2: the equality battery, for all four element types.
// ---------------------------------------------------------------------------

#[test]
fn eq_battery_prime_field() {
    // bound zero == sentinel zero, both directions
    assert_eq!(gf7(0), PrimeField::zero());
    assert_eq!(PrimeField::zero(), gf7(0));
    // bound nonzero != sentinel zero
    assert_ne!(gf7(3), PrimeField::zero());
    assert_ne!(PrimeField::zero(), gf7(3));
    // unbound binds on compare: 7 = 0, 8 = 1 (mod 7)
    assert_eq!(unbound_pf(7), gf7(0));
    assert_eq!(gf7(0), unbound_pf(7));
    assert_eq!(unbound_pf(8), gf7(1));
    assert_eq!(gf7(1), unbound_pf(8));
    // negative unbound values reduce Euclidean-style: -1 = 6 (mod 7)
    assert_eq!(unbound_pf(-1), gf7(6));
    // unbound vs unbound: equality in Z
    assert_eq!(unbound_pf(1), PrimeField::one());
    assert_ne!(unbound_pf(8), PrimeField::one()); // 8 != 1 in Z
    assert_ne!(unbound_pf(7), PrimeField::zero()); // 7 != 0 in Z
    // bound vs bound cross-modulus: unchanged, still false
    let gf5_0 = PrimeField::new(Integer::from(0), Integer::from(5)).unwrap();
    assert_ne!(gf7(0), gf5_0);
    assert_ne!(gf7(1), PrimeField::new(Integer::from(1), Integer::from(5)).unwrap());
    // is_zero / is_one agree with ==
    assert_eq!(gf7(0).is_zero(), gf7(0) == PrimeField::zero());
    assert_eq!(gf7(1).is_one(), gf7(1) == PrimeField::one());
    assert_eq!(gf7(3).is_zero(), gf7(3) == PrimeField::zero());
    assert_eq!(unbound_pf(7).is_zero(), unbound_pf(7) == PrimeField::zero());
    assert!(PrimeField::zero().is_zero());
    assert!(PrimeField::one().is_one());
    // Neg/Sub paths that construct bound-vs-sentinel compares
    assert_eq!(-PrimeField::zero(), gf7(0)); // Neg keeps the sentinel unbound
    assert_eq!(gf7(3) - gf7(3), PrimeField::zero()); // bound zero from Sub
    assert_eq!(gf7(3) + (-gf7(3)), PrimeField::zero()); // bound zero via Neg
    assert_eq!(gf7(2) - PrimeField::one(), gf7(1)); // sentinel binds in Sub
}

#[test]
fn eq_battery_integer_mod() {
    let zero = <IntegerMod as Ring>::zero();
    let one = <IntegerMod as Ring>::one();
    assert_eq!(zmod7(0), zero);
    assert_eq!(zero, zmod7(0));
    assert_ne!(zmod7(3), zero);
    assert_ne!(zero, zmod7(3));
    assert_eq!(unbound_im(7), zmod7(0));
    assert_eq!(zmod7(0), unbound_im(7));
    assert_eq!(unbound_im(8), zmod7(1));
    assert_eq!(zmod7(1), unbound_im(8));
    assert_eq!(unbound_im(-1), zmod7(6));
    assert_eq!(unbound_im(1), one);
    assert_ne!(unbound_im(8), one);
    assert_ne!(unbound_im(7), zero);
    // bound vs bound cross-modulus unchanged
    let z10_3 = IntegerMod::new(Integer::from(3), Integer::from(10)).unwrap();
    assert_ne!(zmod7(3), z10_3);
    // is_zero / is_one agree with ==
    assert_eq!(zmod7(0).is_zero(), zmod7(0) == zero);
    assert_eq!(zmod7(1).is_one(), zmod7(1) == one);
    assert_eq!(zmod7(3).is_zero(), zmod7(3) == zero);
    assert_eq!(unbound_im(7).is_zero(), unbound_im(7) == zero);
    assert!(zero.is_zero());
    assert!(one.is_one());
    // Neg/Sub paths that construct bound-vs-sentinel compares
    assert_eq!(-zero.clone(), zmod7(0));
    assert_eq!(zmod7(3) - zmod7(3), zero);
    assert_eq!(zmod7(3) + (-zmod7(3)), zero);
    assert_eq!(zmod7(2) - one.clone(), zmod7(1));
}

#[test]
fn eq_battery_finite_field_element() {
    let f4 = FiniteField::new(Integer::from(2), 2).unwrap(); // GF(4)
    let f7 = FiniteField::new(Integer::from(7), 1).unwrap(); // GF(7)

    // bound zero == sentinel zero, both directions
    assert_eq!(f4.zero(), FiniteFieldElement::zero());
    assert_eq!(FiniteFieldElement::zero(), f4.zero());
    assert_eq!(f7.zero(), FiniteFieldElement::zero());
    // bound nonzero != sentinel zero
    assert_ne!(f4.generator(), FiniteFieldElement::zero());
    assert_ne!(FiniteFieldElement::zero(), f4.generator());
    // unbound binds on compare via Z -> GF(p^n): 7 = 0, 8 = 1 in GF(7)
    assert_eq!(unbound_ffe(7), f7.zero());
    assert_eq!(f7.zero(), unbound_ffe(7));
    assert_eq!(unbound_ffe(8), f7.one());
    assert_eq!(f7.one(), unbound_ffe(8));
    // ... and in GF(4) (characteristic 2): 7 = 1, 8 = 0
    assert_eq!(unbound_ffe(7), f4.one());
    assert_eq!(unbound_ffe(8), f4.zero());
    assert_eq!(unbound_ffe(-1), f7.from_int(Integer::from(6)));
    // unbound vs unbound: equality in Z
    assert_eq!(unbound_ffe(1), FiniteFieldElement::one());
    assert_ne!(unbound_ffe(8), FiniteFieldElement::one());
    assert_ne!(unbound_ffe(7), FiniteFieldElement::zero());
    // bound vs bound across different fields: unchanged, still false
    assert_ne!(f4.zero(), f7.zero());
    assert_ne!(f4.one(), f7.one());
    // is_zero / is_one agree with ==
    assert_eq!(f4.zero().is_zero(), f4.zero() == FiniteFieldElement::zero());
    assert_eq!(f4.one().is_one(), f4.one() == FiniteFieldElement::one());
    assert_eq!(
        f4.generator().is_zero(),
        f4.generator() == FiniteFieldElement::zero()
    );
    assert_eq!(
        unbound_ffe(7).is_zero(),
        unbound_ffe(7) == FiniteFieldElement::zero()
    );
    assert!(FiniteFieldElement::zero().is_zero());
    assert!(FiniteFieldElement::one().is_one());
    // Neg/Sub paths that construct bound-vs-sentinel compares
    assert_eq!(-FiniteFieldElement::zero(), f7.zero());
    assert_eq!(f7.one() - f7.one(), FiniteFieldElement::zero());
    assert_eq!(f4.generator() - f4.generator(), FiniteFieldElement::zero());
    assert_eq!(
        f7.from_int(Integer::from(2)) - FiniteFieldElement::one(),
        f7.one()
    );
}

#[test]
fn eq_battery_extension_field() {
    // GF(4) = F_2[x]/(x^2 + x + 1)
    let irr = UnivariatePolynomial::new(vec![
        Integer::from(1),
        Integer::from(1),
        Integer::from(1),
    ]);
    let el = |coeffs: &[i64]| -> ExtensionField {
        ExtensionField::new(
            UnivariatePolynomial::new(coeffs.iter().map(|&c| Integer::from(c)).collect()),
            Integer::from(2),
            irr.clone(),
        )
        .unwrap()
    };
    let zero4 = el(&[0]);
    let one4 = el(&[1]);
    let alpha = el(&[0, 1]);

    // bound zero == sentinel zero, both directions
    assert_eq!(zero4, <ExtensionField as Ring>::zero());
    assert_eq!(<ExtensionField as Ring>::zero(), zero4);
    // bound nonzero != sentinel zero
    assert_ne!(alpha, <ExtensionField as Ring>::zero());
    assert_ne!(<ExtensionField as Ring>::zero(), alpha);
    // unbound binds on compare (characteristic 2): 7 = 1, 8 = 0
    assert_eq!(unbound_ext(7), one4);
    assert_eq!(one4, unbound_ext(7));
    assert_eq!(unbound_ext(8), zero4);
    assert_eq!(zero4, unbound_ext(8));
    assert_eq!(unbound_ext(-1), one4); // -1 = 1 in characteristic 2
    // unbound vs unbound: equality in Z
    assert_eq!(unbound_ext(1), <ExtensionField as Ring>::one());
    assert_ne!(unbound_ext(7), <ExtensionField as Ring>::one());
    assert_ne!(unbound_ext(8), <ExtensionField as Ring>::zero());
    // bound vs bound across different fields: unchanged, still false
    let irr9 = UnivariatePolynomial::new(vec![
        Integer::from(1),
        Integer::from(0),
        Integer::from(1),
    ]); // x^2 + 1, irreducible over F_3
    let zero9 = ExtensionField::new(
        UnivariatePolynomial::new(vec![Integer::from(0)]),
        Integer::from(3),
        irr9,
    )
    .unwrap();
    assert_ne!(zero4, zero9);
    // is_zero / is_one agree with ==
    assert_eq!(zero4.is_zero(), zero4 == <ExtensionField as Ring>::zero());
    assert_eq!(one4.is_one(), one4 == <ExtensionField as Ring>::one());
    assert_eq!(alpha.is_zero(), alpha == <ExtensionField as Ring>::zero());
    assert!(<ExtensionField as Ring>::zero().is_zero());
    assert!(<ExtensionField as Ring>::one().is_one());
    // Neg/Sub paths that construct bound-vs-sentinel compares
    assert_eq!(-<ExtensionField as Ring>::zero(), zero4);
    assert_eq!(alpha.clone() - alpha.clone(), <ExtensionField as Ring>::zero());
    assert_eq!(one4.clone() - <ExtensionField as Ring>::one(), zero4);
}

// ---------------------------------------------------------------------------
// The documented transitivity corner (the reason Eq stays a marker with a
// caveat): unbound(0) equals the bound zero of EVERY modulus, while bound
// zeros of different moduli stay unequal. This is the one PartialEq-law
// violation, confined to a zone whose arithmetic already panics.
// ---------------------------------------------------------------------------
#[test]
fn eq_transitivity_corner_is_exactly_as_documented() {
    let gf5_0 = PrimeField::new(Integer::from(0), Integer::from(5)).unwrap();
    assert_eq!(PrimeField::zero(), gf7(0));
    assert_eq!(PrimeField::zero(), gf5_0);
    assert_ne!(gf7(0), gf5_0); // transitivity fails exactly here

    let z5_0 = IntegerMod::new(Integer::from(0), Integer::from(5)).unwrap();
    assert_eq!(<IntegerMod as Ring>::zero(), zmod7(0));
    assert_eq!(<IntegerMod as Ring>::zero(), z5_0);
    assert_ne!(zmod7(0), z5_0);

    let f4 = FiniteField::new(Integer::from(2), 2).unwrap();
    let f7 = FiniteField::new(Integer::from(7), 1).unwrap();
    assert_eq!(FiniteFieldElement::zero(), f4.zero());
    assert_eq!(FiniteFieldElement::zero(), f7.zero());
    assert_ne!(f4.zero(), f7.zero());
}
