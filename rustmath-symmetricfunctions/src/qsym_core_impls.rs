//! Core `rustmath-core` trait tower for [`QSym`] (quasi-symmetric functions).
//!
//! QSym is a commutative `Q`-Hopf-algebra, dual to NCSF; MAGMA ch 146 is the
//! symmetric-function reference and QSym is its standard companion (Gessel). This
//! module gives `QSym` the `std::ops` operators and the
//! `Ring -> CommutativeRing -> IntegralDomain` /
//! `Module<Rational> -> VectorSpace<Rational> -> Algebra<Rational>` tower. Ring
//! operations are routed through the monomial basis; `F_alpha -> M` uses
//! `F_alpha = sum_{beta refines alpha} M_beta` and its Möbius inverse.

use crate::qsym::{product_monomial, refinements, QSym, QSymBasis};
use rustmath_combinatorics::Composition;
use rustmath_core::{Algebra, CommutativeRing, IntegralDomain, Module, Ring, VectorSpace};
use rustmath_rationals::Rational;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

fn add_term(q: &mut QSym, comp: Composition, c: Rational) {
    if c.is_zero() {
        return;
    }
    let entry = q.coeffs.entry(comp.clone()).or_insert_with(Rational::zero);
    *entry = entry.clone() + c;
    if entry.is_zero() {
        q.coeffs.remove(&comp);
    }
}

fn to_monomial(q: &QSym) -> QSym {
    match q.basis {
        QSymBasis::Monomial => q.clone(),
        QSymBasis::Fundamental => {
            let mut out = QSym {
                basis: QSymBasis::Monomial,
                coeffs: HashMap::new(),
            };
            for (alpha, c) in &q.coeffs {
                let mut seen = HashSet::new();
                for beta in refinements(alpha) {
                    if seen.insert(beta.clone()) {
                        add_term(&mut out, beta, c.clone());
                    }
                }
            }
            out
        }
    }
}

fn monomial_to_fundamental(q: &QSym) -> QSym {
    // q is assumed to be in the Monomial basis.
    // M_alpha = sum_{beta refines alpha} (-1)^{l(beta)-l(alpha)} F_beta.
    let mut out = QSym {
        basis: QSymBasis::Fundamental,
        coeffs: HashMap::new(),
    };
    for (alpha, c) in &q.coeffs {
        let la = alpha.length();
        let mut seen = HashSet::new();
        for beta in refinements(alpha) {
            if seen.insert(beta.clone()) {
                let sign = if (beta.length() - la) % 2 == 0 { 1i64 } else { -1 };
                add_term(&mut out, beta, c.clone() * Rational::from(sign));
            }
        }
    }
    out
}

fn convert(q: &QSym, target: QSymBasis) -> QSym {
    if q.basis == target {
        return q.clone();
    }
    match target {
        QSymBasis::Monomial => to_monomial(q),
        QSymBasis::Fundamental => monomial_to_fundamental(&to_monomial(q)),
    }
}

fn qsym_add(a: &QSym, b: &QSym) -> QSym {
    let rhs = convert(b, a.basis);
    let mut out = a.clone();
    for (comp, c) in &rhs.coeffs {
        add_term(&mut out, comp.clone(), c.clone());
    }
    out
}

fn qsym_neg(a: &QSym) -> QSym {
    let mut out = a.clone();
    for c in out.coeffs.values_mut() {
        *c = -c.clone();
    }
    out
}

fn qsym_mul(a: &QSym, b: &QSym) -> QSym {
    let am = to_monomial(a);
    let bm = to_monomial(b);
    let prod = product_monomial(&am, &bm);
    convert(&prod, a.basis)
}

impl fmt::Display for QSym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self.basis {
            QSymBasis::Monomial => "M",
            QSymBasis::Fundamental => "F",
        };
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
            write!(f, "{}*{}{:?}", coeff, symbol, comp.parts())?;
        }
        Ok(())
    }
}

impl Add for QSym {
    type Output = QSym;
    fn add(self, other: QSym) -> QSym {
        qsym_add(&self, &other)
    }
}

impl Sub for QSym {
    type Output = QSym;
    fn sub(self, other: QSym) -> QSym {
        qsym_add(&self, &qsym_neg(&other))
    }
}

impl Neg for QSym {
    type Output = QSym;
    fn neg(self) -> QSym {
        qsym_neg(&self)
    }
}

impl Mul for QSym {
    type Output = QSym;
    fn mul(self, other: QSym) -> QSym {
        qsym_mul(&self, &other)
    }
}

impl Ring for QSym {
    fn zero() -> Self {
        QSym {
            basis: QSymBasis::Monomial,
            coeffs: HashMap::new(),
        }
    }

    fn one() -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(Composition::new(vec![]).unwrap(), Rational::one());
        QSym {
            basis: QSymBasis::Monomial,
            coeffs,
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.values().all(|c| c.is_zero())
    }

    fn is_one(&self) -> bool {
        let m = to_monomial(self);
        let empty = Composition::new(vec![]).unwrap();
        m.coeffs.iter().filter(|(_, c)| !c.is_zero()).count() == 1
            && m.coeffs.get(&empty).map(|c| c.is_one()).unwrap_or(false)
    }
}

impl CommutativeRing for QSym {}
impl IntegralDomain for QSym {}

impl Module<Rational> for QSym {
    fn scalar_mul(&self, scalar: &Rational) -> Self {
        let mut out = self.clone();
        for c in out.coeffs.values_mut() {
            *c = c.clone() * scalar.clone();
        }
        out.coeffs.retain(|_, c| !c.is_zero());
        out
    }

    fn zero() -> Self {
        <QSym as Ring>::zero()
    }

    fn is_zero(&self) -> bool {
        <QSym as Ring>::is_zero(self)
    }
}

impl VectorSpace<Rational> for QSym {
    fn dimension() -> Option<usize> {
        None
    }
}

impl Algebra<Rational> for QSym {
    fn scalar_mul(&self, scalar: &Rational) -> Self {
        Module::scalar_mul(self, scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qsym::{fundamental, monomial};

    fn comp(v: Vec<usize>) -> Composition {
        Composition::new(v).unwrap()
    }

    #[test]
    fn test_qsym_ring_identities() {
        let z = <QSym as Ring>::zero();
        assert!(<QSym as Ring>::is_zero(&z));
        let o = <QSym as Ring>::one();
        assert!(<QSym as Ring>::is_one(&o));
    }

    #[test]
    fn test_qsym_monomial_product() {
        // M_(1) * M_(1) = 2 M_(1,1) + M_(2) (quasi-shuffle).
        let m1 = monomial(comp(vec![1]));
        let prod = m1.clone() * m1.clone();
        assert_eq!(prod.coeff(&comp(vec![1, 1])), Rational::from(2));
        assert_eq!(prod.coeff(&comp(vec![2])), Rational::one());
    }

    #[test]
    fn test_fundamental_to_monomial() {
        // F_(2) = M_(2) + M_(1,1);  F_(1,1) = M_(1,1).
        let f2 = fundamental(comp(vec![2]));
        let m = to_monomial(&f2);
        assert_eq!(m.coeff(&comp(vec![2])), Rational::one());
        assert_eq!(m.coeff(&comp(vec![1, 1])), Rational::one());

        let f11 = fundamental(comp(vec![1, 1]));
        let m11 = to_monomial(&f11);
        assert_eq!(m11.coeff(&comp(vec![1, 1])), Rational::one());
        assert_eq!(m11.coeff(&comp(vec![2])), Rational::zero());
    }

    #[test]
    fn test_fundamental_roundtrip() {
        for parts in [vec![2, 1], vec![1, 1, 1], vec![3], vec![1, 2]] {
            let f = fundamental(comp(parts.clone()));
            let back = convert(&to_monomial(&f), QSymBasis::Fundamental);
            assert_eq!(back.coeff(&comp(parts)), Rational::one());
            assert_eq!(back.coeffs.values().filter(|c| !c.is_zero()).count(), 1);
        }
    }

    #[test]
    fn test_fundamental_product_consistent() {
        // F_(1) = M_(1); so F_(1)^2 must equal M_(1)^2 = 2M_(1,1)+M_(2).
        let f1 = fundamental(comp(vec![1]));
        let prod = f1.clone() * f1.clone();
        let m = to_monomial(&prod);
        assert_eq!(m.coeff(&comp(vec![1, 1])), Rational::from(2));
        assert_eq!(m.coeff(&comp(vec![2])), Rational::one());
    }

    #[test]
    fn test_qsym_distributive() {
        let m1 = monomial(comp(vec![1]));
        let m2 = monomial(comp(vec![2]));
        let m11 = monomial(comp(vec![1, 1]));
        let lhs = m1.clone() * (m2.clone() + m11.clone());
        let rhs = (m1.clone() * m2.clone()) + (m1.clone() * m11.clone());
        assert_eq!(lhs, rhs);
    }
}
