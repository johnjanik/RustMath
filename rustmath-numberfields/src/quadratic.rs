//! Quadratic field arithmetic `L = Q(√δ)`.
//!
//! Ported from dessin_engine `src/quad_field.rs`. Lightweight sugar over the
//! `x² − δ` convention: a `QuadElem` is `a + b·w` with `w² = δ`. The nontrivial
//! automorphism is `σ(√δ) = −√δ` ([`QuadElem::conjugate`]).
//!
//! This is a *working* element layer that lives beside the crate's toy OO
//! `NumberField` in `lib.rs`; it uses the RustMath foundation type
//! [`rustmath_rationals::Rational`] rather than dessin_engine's private `Rat`.

use rustmath_core::Ring;
use rustmath_rationals::Rational;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadField {
    /// `L = Q(w)`, `w² = δ`.
    pub delta: Rational,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadElem {
    pub field: QuadField,
    /// `a + b·w`.
    pub a: Rational,
    pub b: Rational,
}

impl QuadField {
    pub fn new(delta: Rational) -> Self {
        Self { delta }
    }
    pub fn elem(&self, a: Rational, b: Rational) -> QuadElem {
        QuadElem {
            field: self.clone(),
            a,
            b,
        }
    }
    pub fn zero(&self) -> QuadElem {
        self.elem(Rational::zero(), Rational::zero())
    }
    pub fn one(&self) -> QuadElem {
        self.elem(Rational::one(), Rational::zero())
    }
    /// The generator `w` with `w² = δ`.
    pub fn w(&self) -> QuadElem {
        self.elem(Rational::zero(), Rational::one())
    }
    pub fn from_rat(&self, r: Rational) -> QuadElem {
        self.elem(r, Rational::zero())
    }
}

impl QuadElem {
    fn check(&self, rhs: &Self) {
        debug_assert_eq!(self.field.delta, rhs.field.delta);
    }
    pub fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }
    /// `σ(a + b w) = a − b w`.
    pub fn conjugate(&self) -> Self {
        self.field.elem(self.a.clone(), -self.b.clone())
    }
    pub fn add(&self, rhs: &Self) -> Self {
        self.check(rhs);
        self.field
            .elem(&self.a + &rhs.a, &self.b + &rhs.b)
    }
    pub fn neg(&self) -> Self {
        self.field.elem(-&self.a, -&self.b)
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }
    pub fn mul(&self, rhs: &Self) -> Self {
        self.check(rhs);
        let delta = &self.field.delta;
        let a = &(&self.a * &rhs.a) + &(delta * &(&self.b * &rhs.b));
        let b = &(&self.a * &rhs.b) + &(&self.b * &rhs.a);
        self.field.elem(a, b)
    }
    /// `N(a + b w) = a² − δ b²`.
    pub fn norm(&self) -> Rational {
        &(&self.a * &self.a) - &(&self.field.delta * &(&self.b * &self.b))
    }
    pub fn inv(&self) -> Option<Self> {
        let n = self.norm();
        if n.is_zero() {
            return None;
        }
        let c = self.conjugate();
        Some(self.field.elem(&c.a / &n, &c.b / &n))
    }
    pub fn div(&self, rhs: &Self) -> Option<Self> {
        Some(self.mul(&rhs.inv()?))
    }
    pub fn is_rational(&self) -> bool {
        self.b.is_zero()
    }
    pub fn as_rational(&self) -> Option<Rational> {
        if self.is_rational() {
            Some(self.a.clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ri(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn gaussian_integers_arithmetic() {
        // L = Q(i), w = i, w^2 = -1
        let l = QuadField::new(ri(-1));
        let w = l.w();
        assert_eq!(w.mul(&w), l.from_rat(ri(-1))); // i^2 = -1
                                                    // (1+i)(1-i) = 1 - i^2 = 2
        let onep = l.elem(ri(1), ri(1));
        let onem = l.elem(ri(1), ri(-1));
        assert_eq!(onep.mul(&onem), l.from_rat(ri(2)));
        // norm(a+bi) = a^2 + b^2
        assert_eq!(l.elem(ri(3), ri(4)).norm(), ri(25));
    }

    #[test]
    fn inverse_and_conjugate() {
        let l = QuadField::new(ri(2)); // Q(sqrt 2)
        let x = l.elem(ri(1), ri(1)); // 1 + sqrt2
        let inv = x.inv().unwrap();
        assert_eq!(x.mul(&inv), l.one());
        // conjugate of 1+sqrt2 is 1-sqrt2; product = 1 - 2 = -1 = norm
        assert_eq!(x.mul(&x.conjugate()), l.from_rat(x.norm()));
        assert_eq!(x.norm(), ri(-1));
    }
}
