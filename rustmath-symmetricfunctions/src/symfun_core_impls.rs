//! Core `rustmath-core` trait tower for [`SymFun`] (MAGMA Handbook ch 146).
//!
//! The algebra of symmetric functions `Lambda` is a commutative `Q`-algebra. This
//! module gives `SymFun` its `std::ops` operators plus the
//! `Ring -> CommutativeRing -> IntegralDomain` and
//! `Module<Rational> -> VectorSpace<Rational> -> Algebra<Rational>` structure, so
//! symmetric functions are visible to generic algebra code (previously only ad-hoc
//! `add`/`scale`/`coeff` methods existed). Ring operations are routed through the
//! power-sum hub in [`crate::classical_bases`]; `zero`/`one` are canonically returned
//! in the power-sum basis.

use crate::classical_bases::{change_basis, multiply};
use crate::{SymFun, SymmetricFunctionBasis};
use rustmath_core::{Algebra, CommutativeRing, IntegralDomain, Module, Ring, VectorSpace};
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

impl fmt::Display for SymFun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self.basis {
            SymmetricFunctionBasis::Monomial => "m",
            SymmetricFunctionBasis::Elementary => "e",
            SymmetricFunctionBasis::PowerSum => "p",
            SymmetricFunctionBasis::Schur => "s",
        };
        // Deterministic order: by weight then lexicographically on parts.
        let mut terms: Vec<(&rustmath_combinatorics::Partition, &Rational)> = self
            .coeffs
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .collect();
        terms.sort_by(|a, b| {
            a.0.sum()
                .cmp(&b.0.sum())
                .then_with(|| a.0.parts().cmp(b.0.parts()))
        });
        if terms.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (part, coeff) in terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            write!(f, "{}*{}{:?}", coeff, symbol, part.parts())?;
        }
        Ok(())
    }
}

impl SymFun {
    /// Add two symmetric functions, converting `other` into `self`'s basis first.
    fn add_converted(&self, other: &SymFun) -> SymFun {
        let rhs = if self.basis == other.basis {
            other.clone()
        } else {
            change_basis(other, self.basis)
        };
        let mut result = self.clone();
        for (part, coeff) in &rhs.coeffs {
            result.add_term(part.clone(), coeff.clone());
        }
        result.coeffs.retain(|_, c| !c.is_zero());
        result
    }

    /// Negate all coefficients.
    fn negated(&self) -> SymFun {
        let mut result = self.clone();
        for c in result.coeffs.values_mut() {
            *c = -c.clone();
        }
        result
    }
}

impl Add for SymFun {
    type Output = SymFun;
    fn add(self, other: SymFun) -> SymFun {
        self.add_converted(&other)
    }
}

impl Sub for SymFun {
    type Output = SymFun;
    fn sub(self, other: SymFun) -> SymFun {
        self.add_converted(&other.negated())
    }
}

impl Neg for SymFun {
    type Output = SymFun;
    fn neg(self) -> SymFun {
        self.negated()
    }
}

impl Mul for SymFun {
    type Output = SymFun;
    fn mul(self, other: SymFun) -> SymFun {
        multiply(&self, &other)
    }
}

impl Ring for SymFun {
    fn zero() -> Self {
        SymFun::new(SymmetricFunctionBasis::PowerSum)
    }

    fn one() -> Self {
        let mut sf = SymFun::new(SymmetricFunctionBasis::PowerSum);
        sf.add_term(rustmath_combinatorics::Partition::new(vec![]), Rational::one());
        sf
    }

    fn is_zero(&self) -> bool {
        self.coeffs.values().all(|c| c.is_zero())
    }

    fn is_one(&self) -> bool {
        // 1 = p_[] with coefficient 1 and nothing else.
        let pm = crate::classical_bases::to_powersum(self);
        let empty = rustmath_combinatorics::Partition::new(vec![]);
        pm.len() == 1 && pm.get(&empty).map(|c| c.is_one()).unwrap_or(false)
    }
}

impl CommutativeRing for SymFun {}
impl IntegralDomain for SymFun {}

impl Module<Rational> for SymFun {
    fn scalar_mul(&self, scalar: &Rational) -> Self {
        self.scale(scalar)
    }

    fn zero() -> Self {
        <SymFun as Ring>::zero()
    }

    fn is_zero(&self) -> bool {
        <SymFun as Ring>::is_zero(self)
    }
}

impl VectorSpace<Rational> for SymFun {
    fn dimension() -> Option<usize> {
        // Lambda is a free module of infinite rank (one generator per partition).
        None
    }
}

impl Algebra<Rational> for SymFun {
    fn scalar_mul(&self, scalar: &Rational) -> Self {
        Module::scalar_mul(self, scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{power_sum_symmetric, schur_function};
    use rustmath_combinatorics::Partition;

    fn part(v: Vec<usize>) -> Partition {
        Partition::new(v)
    }

    #[test]
    fn test_ring_zero_one() {
        let z = <SymFun as Ring>::zero();
        assert!(<SymFun as Ring>::is_zero(&z));
        let o = <SymFun as Ring>::one();
        assert!(<SymFun as Ring>::is_one(&o));
        assert!(!<SymFun as Ring>::is_zero(&o));
    }

    #[test]
    fn test_additive_identity() {
        let s = schur_function(part(vec![2, 1]));
        let sum = s.clone() + <SymFun as Ring>::zero();
        assert_eq!(sum.coeff(&part(vec![2, 1])), Rational::one());
        assert_eq!(sum.support().len(), 1);
    }

    #[test]
    fn test_multiplicative_identity() {
        let s = schur_function(part(vec![2, 1]));
        let prod = s.clone() * <SymFun as Ring>::one();
        // multiply returns in self's basis (Schur here).
        assert_eq!(prod.basis, SymmetricFunctionBasis::Schur);
        assert_eq!(prod.coeff(&part(vec![2, 1])), Rational::one());
        assert_eq!(prod.support().len(), 1);
    }

    #[test]
    fn test_distributivity() {
        // p1 * (p2 + p3) == p1*p2 + p1*p3 in the power-sum basis.
        let p1 = power_sum_symmetric(part(vec![1]));
        let p2 = power_sum_symmetric(part(vec![2]));
        let p3 = power_sum_symmetric(part(vec![3]));
        let lhs = p1.clone() * (p2.clone() + p3.clone());
        let rhs = (p1.clone() * p2.clone()) + (p1.clone() * p3.clone());
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_negation() {
        let s = schur_function(part(vec![2]));
        let d = s.clone() - s.clone();
        assert!(<SymFun as Ring>::is_zero(&d));
    }

    #[test]
    fn test_commutativity_of_product() {
        let a = schur_function(part(vec![2, 1]));
        let b = schur_function(part(vec![1, 1]));
        let ab = a.clone() * b.clone();
        let ba = b.clone() * a.clone();
        // Both land in the same (Schur) basis and are equal as symmetric functions.
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_display_nonempty() {
        let s = schur_function(part(vec![2, 1]));
        let text = format!("{}", s);
        assert!(text.contains('s'));
    }

    #[test]
    fn test_algebra_scalar_mul() {
        let s = schur_function(part(vec![2, 1]));
        let scaled = Module::scalar_mul(&s, &Rational::from(3));
        assert_eq!(scaled.coeff(&part(vec![2, 1])), Rational::from(3));
        assert_eq!(<SymFun as VectorSpace<Rational>>::dimension(), None);
    }
}
