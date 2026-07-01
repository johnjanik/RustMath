//! Core `rustmath-core` trait tower for [`NCSF`] (noncommutative symmetric functions).
//!
//! NCSF is a **noncommutative** `Q`-algebra (dual to QSym), companion to the
//! symmetric functions of MAGMA ch 146. This module supplies the `std::ops`
//! operators and the `Ring` / `Module<Rational> -> VectorSpace -> Algebra<Rational>`
//! tower. `Ring` (not `CommutativeRing`) is implemented because the product is
//! concatenation of compositions, which is noncommutative.
//!
//! Ring operations are routed through the Complete (`S`) basis, where the product is
//! concatenation (`crate::ncsf::product_complete`). Cross-basis arithmetic is
//! supported for the Complete/Elementary/Monomial bases (which have change-of-basis
//! maps in `crate::ncsf`); same-basis arithmetic works for every basis. Arithmetic
//! mixing the Ribbon/Phi/Psi bases with a different basis is not yet wired.

use crate::ncsf::{
    complete_to_elementary, complete_to_monomial, elementary_to_complete, monomial_to_complete,
    product_complete, NCSFBasis, NCSF,
};
use rustmath_combinatorics::Composition;
use rustmath_core::{Algebra, Module, Ring, VectorSpace};
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

fn to_complete(x: &NCSF) -> NCSF {
    match x.basis {
        NCSFBasis::Complete => x.clone(),
        NCSFBasis::Elementary => elementary_to_complete(x),
        NCSFBasis::Monomial => monomial_to_complete(x),
        other => panic!(
            "NCSF: conversion from {:?} basis to Complete is not implemented",
            other
        ),
    }
}

fn complete_to(x: &NCSF, target: NCSFBasis) -> NCSF {
    debug_assert_eq!(x.basis, NCSFBasis::Complete);
    match target {
        NCSFBasis::Complete => x.clone(),
        NCSFBasis::Elementary => complete_to_elementary(x),
        NCSFBasis::Monomial => complete_to_monomial(x),
        other => panic!(
            "NCSF: conversion from Complete basis to {:?} is not implemented",
            other
        ),
    }
}

fn convert(x: &NCSF, target: NCSFBasis) -> NCSF {
    if x.basis == target {
        return x.clone();
    }
    complete_to(&to_complete(x), target)
}

fn ncsf_add(a: &NCSF, b: &NCSF) -> NCSF {
    let rhs = convert(b, a.basis);
    let mut out = a.clone();
    for (comp, c) in &rhs.coeffs {
        out.add_term(comp.clone(), c.clone());
    }
    out.coeffs.retain(|_, c| !c.is_zero());
    out
}

fn ncsf_neg(a: &NCSF) -> NCSF {
    let mut out = a.clone();
    for c in out.coeffs.values_mut() {
        *c = -c.clone();
    }
    out
}

fn ncsf_mul(a: &NCSF, b: &NCSF) -> NCSF {
    let ac = to_complete(a);
    let bc = to_complete(b);
    let prod = product_complete(&ac, &bc);
    convert(&prod, a.basis)
}

impl fmt::Display for NCSF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms: Vec<(&Composition, &Rational)> =
            self.coeffs.iter().filter(|(_, c)| !c.is_zero()).collect();
        terms.sort_by(|a, b| {
            a.0.sum()
                .cmp(&b.0.sum())
                .then_with(|| a.0.parts().cmp(b.0.parts()))
        });
        if terms.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (comp, coeff) in terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            write!(f, "{}*{}{:?}", coeff, self.basis, comp.parts())?;
        }
        Ok(())
    }
}

impl Add for NCSF {
    type Output = NCSF;
    fn add(self, other: NCSF) -> NCSF {
        ncsf_add(&self, &other)
    }
}

impl Sub for NCSF {
    type Output = NCSF;
    fn sub(self, other: NCSF) -> NCSF {
        ncsf_add(&self, &ncsf_neg(&other))
    }
}

impl Neg for NCSF {
    type Output = NCSF;
    fn neg(self) -> NCSF {
        ncsf_neg(&self)
    }
}

impl Mul for NCSF {
    type Output = NCSF;
    fn mul(self, other: NCSF) -> NCSF {
        ncsf_mul(&self, &other)
    }
}

impl Ring for NCSF {
    fn zero() -> Self {
        NCSF::new(NCSFBasis::Complete)
    }

    fn one() -> Self {
        NCSF::one(NCSFBasis::Complete)
    }

    fn is_zero(&self) -> bool {
        self.coeffs.values().all(|c| c.is_zero())
    }

    fn is_one(&self) -> bool {
        let c = to_complete(self);
        let empty = Composition::new(vec![]).unwrap();
        c.coeffs.iter().filter(|(_, v)| !v.is_zero()).count() == 1
            && c.coeffs.get(&empty).map(|v| v.is_one()).unwrap_or(false)
    }
}

impl Module<Rational> for NCSF {
    fn scalar_mul(&self, scalar: &Rational) -> Self {
        let mut out = self.scale(scalar);
        out.coeffs.retain(|_, c| !c.is_zero());
        out
    }

    fn zero() -> Self {
        <NCSF as Ring>::zero()
    }

    fn is_zero(&self) -> bool {
        <NCSF as Ring>::is_zero(self)
    }
}

impl VectorSpace<Rational> for NCSF {
    fn dimension() -> Option<usize> {
        None
    }
}

impl Algebra<Rational> for NCSF {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ncsf::{complete, elementary};

    fn comp(v: Vec<usize>) -> Composition {
        Composition::new(v).unwrap()
    }

    #[test]
    fn test_ncsf_ring_identities() {
        let z = <NCSF as Ring>::zero();
        assert!(<NCSF as Ring>::is_zero(&z));
        let o = <NCSF as Ring>::one();
        assert!(<NCSF as Ring>::is_one(&o));
    }

    #[test]
    fn test_ncsf_complete_product_is_concatenation() {
        // S_(2) * S_(1) = S_(2,1) and S_(1) * S_(2) = S_(1,2) (noncommutative).
        let s2 = complete(comp(vec![2]));
        let s1 = complete(comp(vec![1]));
        let left = s2.clone() * s1.clone();
        let right = s1.clone() * s2.clone();
        assert_eq!(left.coeff(&comp(vec![2, 1])), Rational::one());
        assert_eq!(right.coeff(&comp(vec![1, 2])), Rational::one());
        assert_ne!(left, right);
    }

    #[test]
    fn test_ncsf_one_is_neutral() {
        let s = complete(comp(vec![2, 1]));
        let prod = s.clone() * <NCSF as Ring>::one();
        assert_eq!(prod.coeff(&comp(vec![2, 1])), Rational::one());
        assert_eq!(prod.coeffs.values().filter(|c| !c.is_zero()).count(), 1);
    }

    #[test]
    fn test_ncsf_same_basis_add() {
        let a = elementary(comp(vec![2]));
        let b = elementary(comp(vec![2]));
        let sum = a + b;
        assert_eq!(sum.coeff(&comp(vec![2])), Rational::from(2));
    }

    #[test]
    fn test_ncsf_distributive() {
        let s1 = complete(comp(vec![1]));
        let s2 = complete(comp(vec![2]));
        let s3 = complete(comp(vec![3]));
        let lhs = s1.clone() * (s2.clone() + s3.clone());
        let rhs = (s1.clone() * s2.clone()) + (s1.clone() * s3.clone());
        assert_eq!(lhs, rhs);
    }
}
