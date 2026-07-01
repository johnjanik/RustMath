//! Core `rustmath-core` trait tower for [`FQSym`] (free quasi-symmetric functions).
//!
//! `FQSym` already carries `std::ops` operators (its default product is the shuffle
//! product); this module adds the `Ring` and
//! `Module<Rational> -> VectorSpace<Rational> -> Algebra<Rational>` structure plus a
//! `Display` impl, so it is a first-class algebra element for generic code. Companion
//! algebra to the symmetric functions of MAGMA ch 146. `Ring` (not `CommutativeRing`)
//! is implemented, matching the general (noncommutative) FQSym convention.

use crate::fqsym::FQSym;
use rustmath_combinatorics::Composition;
use rustmath_core::{Algebra, Module, Ring, VectorSpace};
use rustmath_rationals::Rational;
use std::fmt;

impl fmt::Display for FQSym {
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
            write!(f, "{}*F{:?}", coeff, comp.parts())?;
        }
        Ok(())
    }
}

impl Ring for FQSym {
    fn zero() -> Self {
        FQSym::zero()
    }

    fn one() -> Self {
        FQSym::one()
    }

    fn is_zero(&self) -> bool {
        self.coeffs.values().all(|c| c.is_zero())
    }

    fn is_one(&self) -> bool {
        let empty = Composition::new(vec![]).unwrap();
        self.coeffs.iter().filter(|(_, c)| !c.is_zero()).count() == 1
            && self.coeffs.get(&empty).map(|c| c.is_one()).unwrap_or(false)
    }
}

impl Module<Rational> for FQSym {
    fn scalar_mul(&self, scalar: &Rational) -> Self {
        let mut out = self.scale(scalar);
        out.coeffs.retain(|_, c| !c.is_zero());
        out
    }

    fn zero() -> Self {
        FQSym::zero()
    }

    fn is_zero(&self) -> bool {
        <FQSym as Ring>::is_zero(self)
    }
}

impl VectorSpace<Rational> for FQSym {
    fn dimension() -> Option<usize> {
        None
    }
}

impl Algebra<Rational> for FQSym {}

#[cfg(test)]
mod tests {
    use super::*;

    fn comp(v: Vec<usize>) -> Composition {
        Composition::new(v).unwrap()
    }

    #[test]
    fn test_fqsym_ring_identities() {
        let z = <FQSym as Ring>::zero();
        assert!(<FQSym as Ring>::is_zero(&z));
        let o = <FQSym as Ring>::one();
        assert!(<FQSym as Ring>::is_one(&o));
        assert!(!<FQSym as Ring>::is_one(&z));
    }

    #[test]
    fn test_fqsym_unit_neutral() {
        let f = FQSym::f_basis(comp(vec![2, 1]));
        let prod = f.clone() * <FQSym as Ring>::one();
        assert_eq!(prod.coeff(&comp(vec![2, 1])), Rational::one());
    }

    #[test]
    fn test_fqsym_distributive() {
        let a = FQSym::f_basis(comp(vec![1]));
        let b = FQSym::f_basis(comp(vec![2]));
        let c = FQSym::f_basis(comp(vec![1, 1]));
        let lhs = a.clone() * (b.clone() + c.clone());
        let rhs = (a.clone() * b.clone()) + (a.clone() * c.clone());
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_fqsym_scalar_mul() {
        let f = FQSym::f_basis(comp(vec![2, 1]));
        let scaled = Module::scalar_mul(&f, &Rational::from(3));
        assert_eq!(scaled.coeff(&comp(vec![2, 1])), Rational::from(3));
        assert_eq!(<FQSym as VectorSpace<Rational>>::dimension(), None);
    }
}
