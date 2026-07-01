//! Minimal number-field element arithmetic `L = Q[x]/(m)`.
//!
//! Ported from dessin_engine `src/number_field.rs`. A dense `Q[x]` with division
//! and extended gcd, a monic modulus, and element arithmetic (including inverse
//! via xgcd). This is a *working* element layer that lives beside the crate's toy
//! OO `NumberField` in `lib.rs` (whose `inv` is a stub for non-rational
//! elements). It uses the RustMath foundation type
//! [`rustmath_rationals::Rational`] rather than dessin_engine's private `Rat`.
//!
//! The types here (`PolyQ`, `NumberField`, `NfElem`) are intentionally scoped to
//! this module and are *not* re-exported from the crate root, so they do not
//! collide with the crate-root `NumberField`.

use rustmath_core::Ring;
use rustmath_rationals::Rational;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NfError {
    #[error("defining polynomial must be monic")]
    NonMonic,
    #[error("division by zero")]
    DivisionByZero,
    #[error("element is not invertible (modulus not irreducible, or zero)")]
    NoInverse,
    #[error("elements live in different fields")]
    DegreeMismatch,
}

/// Dense polynomial over `Q`, low-to-high: `c[0] + c[1] x + … + c[n] x^n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyQ {
    pub c: Vec<Rational>,
}

impl PolyQ {
    pub fn zero() -> Self {
        Self { c: vec![] }
    }
    pub fn one() -> Self {
        Self {
            c: vec![Rational::one()],
        }
    }
    pub fn degree(&self) -> Option<usize> {
        self.c.iter().rposition(|x| !x.is_zero())
    }
    pub fn trim(&mut self) {
        while self.c.last().is_some_and(|x| x.is_zero()) {
            self.c.pop();
        }
    }
    pub fn is_zero(&self) -> bool {
        self.degree().is_none()
    }
    pub fn monomial(coeff: Rational, deg: usize) -> Self {
        let mut c = vec![Rational::zero(); deg + 1];
        c[deg] = coeff;
        let mut p = Self { c };
        p.trim();
        p
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let n = self.c.len().max(rhs.c.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.c.get(i).cloned().unwrap_or_else(Rational::zero);
            let b = rhs.c.get(i).cloned().unwrap_or_else(Rational::zero);
            out.push(a + b);
        }
        let mut p = Self { c: out };
        p.trim();
        p
    }
    pub fn neg(&self) -> Self {
        Self {
            c: self.c.iter().map(|x| -x.clone()).collect(),
        }
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }
    pub fn mul(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        let mut out = vec![Rational::zero(); self.c.len() + rhs.c.len() - 1];
        for (i, a) in self.c.iter().enumerate() {
            for (j, b) in rhs.c.iter().enumerate() {
                out[i + j] = out[i + j].clone() + a.clone() * b.clone();
            }
        }
        let mut p = Self { c: out };
        p.trim();
        p
    }
    fn scale(&self, s: &Rational) -> Self {
        let mut p = Self {
            c: self.c.iter().map(|x| x.clone() * s.clone()).collect(),
        };
        p.trim();
        p
    }

    /// General division `self = q·b + r`, `deg r < deg b`.
    pub fn div_rem(&self, b: &Self) -> Result<(Self, Self), NfError> {
        let bd = b.degree().ok_or(NfError::DivisionByZero)?;
        let blc = b.c[bd].clone();
        let mut r = self.clone();
        r.trim();
        let mut q = PolyQ::zero();
        while let Some(rd) = r.degree() {
            if rd < bd {
                break;
            }
            let coeff = r.c[rd].clone() / blc.clone();
            let term = PolyQ::monomial(coeff, rd - bd);
            q = q.add(&term);
            r = r.sub(&term.mul(b));
            r.trim();
        }
        Ok((q, r))
    }

    pub fn rem(&self, b: &Self) -> Result<Self, NfError> {
        Ok(self.div_rem(b)?.1)
    }
}

/// Extended gcd over `Q[x]`: returns `(g, s, t)` with `s·a + t·b = g`, `g` monic.
pub fn xgcd(a: &PolyQ, b: &PolyQ) -> Result<(PolyQ, PolyQ, PolyQ), NfError> {
    let (mut r0, mut r1) = (a.clone(), b.clone());
    let (mut s0, mut s1) = (PolyQ::one(), PolyQ::zero());
    let (mut t0, mut t1) = (PolyQ::zero(), PolyQ::one());
    while !r1.is_zero() {
        let (q, r2) = r0.div_rem(&r1)?;
        r0 = std::mem::replace(&mut r1, r2);
        let s2 = s0.sub(&q.mul(&s1));
        s0 = std::mem::replace(&mut s1, s2);
        let t2 = t0.sub(&q.mul(&t1));
        t0 = std::mem::replace(&mut t1, t2);
    }
    if let Some(d) = r0.degree() {
        let inv = Rational::one() / r0.c[d].clone();
        r0 = r0.scale(&inv);
        s0 = s0.scale(&inv);
        t0 = t0.scale(&inv);
    }
    Ok((r0, s0, t0))
}

/// A number field `Q[x]/(m)`, `m` monic irreducible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumberField {
    pub modulus: PolyQ,
    pub degree: usize,
}

impl NumberField {
    pub fn new(modulus: PolyQ) -> Result<Arc<Self>, NfError> {
        let d = modulus.degree().ok_or(NfError::NonMonic)?;
        if modulus.c[d] != Rational::one() {
            return Err(NfError::NonMonic);
        }
        Ok(Arc::new(Self { modulus, degree: d }))
    }

    pub fn elem(self: &Arc<Self>, coeffs: Vec<Rational>) -> Result<NfElem, NfError> {
        let reduced = PolyQ { c: coeffs }.rem(&self.modulus)?;
        let mut c = reduced.c;
        c.resize(self.degree, Rational::zero());
        Ok(NfElem {
            field: self.clone(),
            c,
        })
    }

    pub fn zero(self: &Arc<Self>) -> NfElem {
        NfElem {
            field: self.clone(),
            c: vec![Rational::zero(); self.degree],
        }
    }
    pub fn one(self: &Arc<Self>) -> NfElem {
        let mut c = vec![Rational::zero(); self.degree];
        c[0] = Rational::one();
        NfElem {
            field: self.clone(),
            c,
        }
    }
    /// The generator `x = α`.
    pub fn gen(self: &Arc<Self>) -> NfElem {
        let mut c = vec![Rational::zero(); self.degree];
        if self.degree > 1 {
            c[1] = Rational::one();
        }
        NfElem {
            field: self.clone(),
            c,
        }
    }
}

/// An element of `L`, stored as a length-`degree` coefficient vector in `α`.
#[derive(Debug, Clone)]
pub struct NfElem {
    pub field: Arc<NumberField>,
    pub c: Vec<Rational>,
}

impl NfElem {
    fn to_poly(&self) -> PolyQ {
        PolyQ { c: self.c.clone() }
    }
    pub fn is_zero(&self) -> bool {
        self.c.iter().all(|x| x.is_zero())
    }

    fn same_field(&self, rhs: &Self) -> Result<(), NfError> {
        if Arc::ptr_eq(&self.field, &rhs.field) || self.field.modulus == rhs.field.modulus {
            Ok(())
        } else {
            Err(NfError::DegreeMismatch)
        }
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, NfError> {
        self.same_field(rhs)?;
        let c = self
            .c
            .iter()
            .zip(&rhs.c)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Ok(Self {
            field: self.field.clone(),
            c,
        })
    }
    pub fn neg(&self) -> Self {
        Self {
            field: self.field.clone(),
            c: self.c.iter().map(|x| -x.clone()).collect(),
        }
    }
    pub fn sub(&self, rhs: &Self) -> Result<Self, NfError> {
        self.add(&rhs.neg())
    }
    pub fn mul(&self, rhs: &Self) -> Result<Self, NfError> {
        self.same_field(rhs)?;
        self.field.elem(self.to_poly().mul(&rhs.to_poly()).c)
    }
    pub fn pow(&self, e: u32) -> Result<Self, NfError> {
        let mut acc = self.field.one();
        for _ in 0..e {
            acc = acc.mul(self)?;
        }
        Ok(acc)
    }
    pub fn inv(&self) -> Result<Self, NfError> {
        if self.is_zero() {
            return Err(NfError::DivisionByZero);
        }
        let (g, s, _) = xgcd(&self.to_poly(), &self.field.modulus)?;
        if g.degree() != Some(0) {
            return Err(NfError::NoInverse); // common factor ⇒ modulus reducible
        }
        // g is the constant 1 (xgcd returns monic g); s is the inverse mod m
        self.field.elem(s.c)
    }
    pub fn div(&self, rhs: &Self) -> Result<Self, NfError> {
        self.mul(&rhs.inv()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }
    fn ri(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn polyq_div_rem_and_xgcd() {
        // (x^2 - 1) = (x+1)(x-1)
        let x2m1 = PolyQ {
            c: vec![ri(-1), ri(0), ri(1)],
        };
        let xp1 = PolyQ {
            c: vec![ri(1), ri(1)],
        };
        let (q, r) = x2m1.div_rem(&xp1).unwrap();
        assert!(r.is_zero());
        assert_eq!(
            q,
            PolyQ {
                c: vec![ri(-1), ri(1)]
            }
        ); // x - 1
           // gcd(x^2-1, x-1) = x-1 (monic)
        let xm1 = PolyQ {
            c: vec![ri(-1), ri(1)],
        };
        let (g, _, _) = xgcd(&x2m1, &xm1).unwrap();
        assert_eq!(g, xm1);
    }

    #[test]
    fn quadratic_field_sqrt2_arithmetic() {
        // L = Q[x]/(x^2 - 2), α = sqrt(2)
        let m = PolyQ {
            c: vec![ri(-2), ri(0), ri(1)],
        };
        let l = NumberField::new(m).unwrap();
        let a = l.gen(); // sqrt2
                         // α^2 = 2
        assert_eq!(a.mul(&a).unwrap().c, vec![ri(2), ri(0)]);
        // (1 + α)(1 - α) = 1 - α^2 = -1
        let onep = l.elem(vec![ri(1), ri(1)]).unwrap();
        let onem = l.elem(vec![ri(1), ri(-1)]).unwrap();
        assert_eq!(onep.mul(&onem).unwrap().c, vec![ri(-1), ri(0)]);
        // inverse of α is α/2
        let inv = a.inv().unwrap();
        assert_eq!(inv.c, vec![ri(0), rat(1, 2)]);
        assert!(a.mul(&inv).unwrap().sub(&l.one()).unwrap().is_zero());
    }

    #[test]
    fn quartic_field_inverse_roundtrips() {
        // L = Q[x]/(x^4 - x - 1), α a root; check α · α^{-1} = 1.
        let m = PolyQ {
            c: vec![ri(-1), ri(-1), ri(0), ri(0), ri(1)],
        };
        let l = NumberField::new(m).unwrap();
        let a = l.elem(vec![ri(1), ri(2), ri(-1), ri(1)]).unwrap(); // generic element
        let inv = a.inv().unwrap();
        assert!(a.mul(&inv).unwrap().sub(&l.one()).unwrap().is_zero());
    }
}
