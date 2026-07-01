//! Finite fields `GF(p^n)` as a shared Parent (MAGMA Handbook ch. 21).
//!
//! MAGMA source: Chapter 21 — Finite Fields (`FiniteField`, `GF`, `ext<>`,
//! `Generator`, `Norm`, `Trace`, `Frobenius`, `MinimalPolynomial`, `Order`,
//! `Log`/discrete logarithm, `DefiningPolynomial`).
//!
//! This module fixes the two structural problems the port survey flagged in the
//! legacy [`crate::ExtensionField`]:
//!   1. it carries the modulus/irreducible via a **shared [`Parent`]**
//!      ([`FiniteField`], an `Arc`-backed [`UniqueRepresentation`]) rather than
//!      one copy per element, so cross-field equality/coercion is checked; and
//!   2. every element **reduces its coefficients modulo `p` and modulo the
//!      defining irreducible on construction** (the legacy `ExtensionField::new`
//!      stored raw coefficients without reducing).
//!
//! Default fields `GF(p, n)` use the [`crate::conway`] polynomial when tabulated
//! (guaranteeing compatible / Conway-consistent embeddings for the small fields
//! in the table); otherwise a monic irreducible is searched for
//! ([`crate::poly_factor`]). Elements are represented as little-endian
//! coefficient vectors of length `n` over `F_p`. The prime field `GF(p) = GF(p,1)`
//! is the same type with `n = 1`.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use rustmath_core::{
    CommutativeRing, Field, MathError, NumericConversion, Parent, Result, Ring, UniqueCache,
    UniqueRepresentation,
};
use rustmath_integers::prime::factor;
use rustmath_integers::Integer;

use crate::conway::conway_polynomial;

// ---------------------------------------------------------------------------
// F_p coefficient helpers (little-endian Vec<Integer>, coeffs kept in [0, p))
// ---------------------------------------------------------------------------

fn redp(x: Integer, p: &Integer) -> Integer {
    let r = x % p.clone();
    if r.signum() < 0 {
        r + p.clone()
    } else {
        r
    }
}

/// Reduce `poly` (arbitrary length) modulo the monic irreducible `modulus`
/// (little-endian, `modulus[n] == 1`, length `n + 1`), returning length-`n`
/// coefficients in `[0, p)`.
fn reduce_mod_irr(mut poly: Vec<Integer>, modulus: &[Integer], p: &Integer) -> Vec<Integer> {
    let n = modulus.len() - 1;
    for c in poly.iter_mut() {
        *c = redp(c.clone(), p);
    }
    // x^k = -Σ_{i<n} modulus[i] x^{k-n+i} for k >= n.
    for k in (n..poly.len()).rev() {
        let lead = poly[k].clone();
        if lead.is_zero() {
            continue;
        }
        for i in 0..n {
            poly[k - n + i] = redp(poly[k - n + i].clone() - lead.clone() * modulus[i].clone(), p);
        }
        poly[k] = Integer::zero();
    }
    poly.truncate(n);
    poly.resize(n, Integer::zero());
    poly
}

fn poly_mul_mod(a: &[Integer], b: &[Integer], modulus: &[Integer], p: &Integer) -> Vec<Integer> {
    let n = modulus.len() - 1;
    if a.is_empty() || b.is_empty() {
        return vec![Integer::zero(); n];
    }
    let mut prod = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            prod[i + j] = redp(prod[i + j].clone() + ai.clone() * bj.clone(), p);
        }
    }
    reduce_mod_irr(prod, modulus, p)
}

// ---------------------------------------------------------------------------
// Parent: FiniteField (GF(p^n))
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FiniteFieldData {
    p: Integer,
    n: usize,
    /// Defining monic irreducible, little-endian, length `n + 1`, coeffs in `[0, p)`.
    modulus: Vec<Integer>,
}

/// A finite field `GF(p^n)`, carried as a shared [`Parent`].
///
/// Cloning is cheap (an `Arc` bump). Two `FiniteField`s are equal iff they have
/// the same characteristic, degree, and defining polynomial.
#[derive(Clone)]
pub struct FiniteField {
    data: Arc<FiniteFieldData>,
}

static FF_CACHE: Lazy<UniqueCache<(Integer, usize, Vec<Integer>), FiniteFieldData>> =
    Lazy::new(UniqueCache::new);

impl UniqueRepresentation for FiniteFieldData {
    type Key = (Integer, usize, Vec<Integer>);
    fn key(&self) -> Self::Key {
        (self.p.clone(), self.n, self.modulus.clone())
    }
    fn cache() -> &'static UniqueCache<Self::Key, Self> {
        &FF_CACHE
    }
}

impl FiniteField {
    /// Construct `GF(p^n)`, using the Conway polynomial if one is tabulated for
    /// `(p, n)`, otherwise searching for a monic irreducible of degree `n`.
    ///
    /// `p` must be prime and `n >= 1`.
    pub fn new(p: Integer, n: usize) -> Result<Self> {
        if !p.is_prime() {
            return Err(MathError::InvalidArgument(format!(
                "characteristic {p} is not prime"
            )));
        }
        if n == 0 {
            return Err(MathError::InvalidArgument("degree must be >= 1".into()));
        }
        if n == 1 {
            // GF(p): F_p[x]/(x).
            return Ok(Self::from_data(FiniteFieldData {
                p: p.clone(),
                n: 1,
                modulus: vec![Integer::zero(), Integer::one()],
            }));
        }
        // Conway polynomial if available.
        if let Some(cp) = p
            .to_u64()
            .and_then(|v| u32::try_from(v).ok())
            .and_then(|pu| conway_polynomial(pu, n))
        {
            let modulus: Vec<Integer> = {
                let mut c: Vec<Integer> = cp.coefficients().to_vec();
                c.resize(n + 1, Integer::zero());
                c.iter().map(|x| redp(x.clone(), &p)).collect()
            };
            return Ok(Self::from_data(FiniteFieldData { p, n, modulus }));
        }
        // Otherwise search for an irreducible of degree n.
        let modulus = crate::poly_factor::find_irreducible(&p, n)?;
        Ok(Self::from_data(FiniteFieldData { p, n, modulus }))
    }

    /// Construct `GF(p^n)` with a caller-supplied monic irreducible polynomial,
    /// given little-endian over `Z` (reduced mod `p` on entry). The polynomial's
    /// degree fixes `n`; it must be monic and irreducible over `F_p`.
    pub fn with_modulus(p: Integer, modulus: Vec<Integer>) -> Result<Self> {
        if !p.is_prime() {
            return Err(MathError::InvalidArgument(format!(
                "characteristic {p} is not prime"
            )));
        }
        if modulus.len() < 2 {
            return Err(MathError::InvalidArgument(
                "defining polynomial must have degree >= 1".into(),
            ));
        }
        let n = modulus.len() - 1;
        let reduced: Vec<Integer> = modulus.iter().map(|c| redp(c.clone(), &p)).collect();
        if !reduced[n].is_one() {
            return Err(MathError::InvalidArgument(
                "defining polynomial must be monic".into(),
            ));
        }
        if n >= 2 && !crate::poly_factor::is_irreducible_fp(&reduced, &p) {
            return Err(MathError::InvalidArgument(
                "defining polynomial is not irreducible over F_p".into(),
            ));
        }
        Ok(Self::from_data(FiniteFieldData {
            p,
            n,
            modulus: reduced,
        }))
    }

    fn from_data(d: FiniteFieldData) -> Self {
        let key = d.key();
        FiniteField {
            data: FiniteFieldData::get_unique(key, || d),
        }
    }

    /// Characteristic `p`.
    pub fn characteristic(&self) -> &Integer {
        &self.data.p
    }

    /// Degree `n` (so the field is `GF(p^n)`).
    pub fn degree(&self) -> usize {
        self.data.n
    }

    /// Cardinality `p^n` as an [`Integer`].
    pub fn order(&self) -> Integer {
        self.data.p.pow(self.data.n as u32)
    }

    /// Defining polynomial, little-endian coefficients over `[0, p)`.
    pub fn defining_polynomial(&self) -> &[Integer] {
        &self.data.modulus
    }

    /// Whether the defining polynomial equals the tabulated Conway polynomial
    /// for `(p, n)` (derived, so it is independent of how the field was built).
    pub fn is_conway(&self) -> bool {
        match self
            .data
            .p
            .to_u64()
            .and_then(|v| u32::try_from(v).ok())
            .and_then(|pu| conway_polynomial(pu, self.data.n))
        {
            Some(cp) => {
                let mut c: Vec<Integer> = cp.coefficients().to_vec();
                c.resize(self.data.n + 1, Integer::zero());
                let c: Vec<Integer> = c.iter().map(|x| redp(x.clone(), &self.data.p)).collect();
                c == self.data.modulus
            }
            None => false,
        }
    }

    /// Whether two fields share the same modulus/parameters.
    pub fn same_field(&self, other: &FiniteField) -> bool {
        self.data == other.data
    }

    /// The prime subfield `GF(p)`.
    pub fn prime_field(&self) -> FiniteField {
        FiniteField::new(self.data.p.clone(), 1).unwrap()
    }

    /// Whether this is a prime field (`n == 1`).
    pub fn is_prime_field(&self) -> bool {
        self.data.n == 1
    }

    /// Whether `GF(p^m)` embeds in this field, i.e. `m | n`. (Compatible /
    /// Conway embeddings in the sense of BCS97 are only guaranteed for
    /// Conway-defined fields; the full embedding lattice is a later effort.)
    pub fn has_subfield_degree(&self, m: usize) -> bool {
        m >= 1 && self.data.n % m == 0
    }

    // -- element constructors -------------------------------------------------

    /// The zero element.
    pub fn zero(&self) -> FiniteFieldElement {
        FiniteFieldElement {
            coeffs: vec![Integer::zero(); self.data.n],
            field: self.clone(),
        }
    }

    /// The one element.
    pub fn one(&self) -> FiniteFieldElement {
        let mut c = vec![Integer::zero(); self.data.n];
        c[0] = Integer::one();
        FiniteFieldElement {
            coeffs: c,
            field: self.clone(),
        }
    }

    /// The generator `x` (the class of the polynomial variable, `F.1`).
    pub fn generator(&self) -> FiniteFieldElement {
        let mut c = vec![Integer::zero(); self.data.n];
        if self.data.n >= 2 {
            c[1] = Integer::one();
        }
        FiniteFieldElement {
            coeffs: c,
            field: self.clone(),
        }
    }

    /// Build an element from little-endian coefficients over `Z`, reducing mod
    /// `p` and mod the defining polynomial.
    pub fn element(&self, coeffs: Vec<Integer>) -> FiniteFieldElement {
        let reduced = reduce_mod_irr(coeffs, &self.data.modulus, &self.data.p);
        FiniteFieldElement {
            coeffs: reduced,
            field: self.clone(),
        }
    }

    /// Coerce an integer `k` into the field (as `k mod p` in the prime subfield).
    pub fn from_int(&self, k: Integer) -> FiniteFieldElement {
        self.element(vec![k])
    }
}

impl fmt::Debug for FiniteField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GF({}^{})", self.data.p, self.data.n)
    }
}

impl fmt::Display for FiniteField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.n == 1 {
            write!(f, "GF({})", self.data.p)
        } else {
            write!(f, "GF({}^{})", self.data.p, self.data.n)
        }
    }
}

impl PartialEq for FiniteField {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl Eq for FiniteField {}

impl std::hash::Hash for FiniteField {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Parent for FiniteField {
    type Element = FiniteFieldElement;

    fn contains(&self, element: &Self::Element) -> bool {
        element.field.data == self.data
    }
    fn zero(&self) -> Option<Self::Element> {
        Some(FiniteField::zero(self))
    }
    fn one(&self) -> Option<Self::Element> {
        Some(FiniteField::one(self))
    }
    fn cardinality(&self) -> Option<usize> {
        self.order().to_usize()
    }
    fn name(&self) -> String {
        format!("{self}")
    }
}

// ---------------------------------------------------------------------------
// Element: FiniteFieldElement
// ---------------------------------------------------------------------------

/// An element of a [`FiniteField`] `GF(p^n)`.
///
/// The parent field is carried by (cheap, `Arc`-backed) value, so arithmetic
/// checks that operands live in the same field.
#[derive(Clone)]
pub struct FiniteFieldElement {
    coeffs: Vec<Integer>,
    field: FiniteField,
}

impl FiniteFieldElement {
    /// The parent field.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Little-endian coefficient sequence over `F_p` (length `n`) — MAGMA `Eltseq`.
    pub fn eltseq(&self) -> &[Integer] {
        &self.coeffs
    }

    fn p(&self) -> &Integer {
        &self.field.data.p
    }
    fn modulus(&self) -> &[Integer] {
        &self.field.data.modulus
    }
    fn n(&self) -> usize {
        self.field.data.n
    }

    fn assert_same(&self, other: &Self) {
        assert!(
            self.field.data == other.field.data,
            "operands live in different finite fields"
        );
    }

    /// `self^e` for a non-negative integer exponent (square-and-multiply).
    pub fn pow_int(&self, e: &Integer) -> FiniteFieldElement {
        let mut result = self.field.one();
        let mut base = self.clone();
        let mut exp = e.clone();
        let two = Integer::from(2);
        while exp > Integer::zero() {
            if redp(exp.clone(), &two).is_one() {
                result = result * base.clone();
            }
            base = base.clone() * base.clone();
            exp = exp / two.clone();
        }
        result
    }

    /// The Frobenius endomorphism `a -> a^p`.
    pub fn frobenius(&self) -> FiniteFieldElement {
        self.pow_int(&self.p().clone())
    }

    /// The `r`-fold Frobenius `a -> a^{p^r}`.
    pub fn frobenius_pow(&self, r: usize) -> FiniteFieldElement {
        let mut cur = self.clone();
        for _ in 0..r {
            cur = cur.frobenius();
        }
        cur
    }

    /// Absolute trace `Tr_{GF(p^n)/GF(p)}(a) = a + a^p + ... + a^{p^{n-1}}`,
    /// returned as its representative in `F_p`.
    pub fn trace(&self) -> Integer {
        let mut sum = self.field.zero();
        let mut cur = self.clone();
        for _ in 0..self.n() {
            sum = sum + cur.clone();
            cur = cur.frobenius();
        }
        sum.coeffs[0].clone()
    }

    /// Absolute norm `N_{GF(p^n)/GF(p)}(a) = a^{(p^n-1)/(p-1)}`, in `F_p`.
    pub fn norm(&self) -> Integer {
        if self.is_zero() {
            return Integer::zero();
        }
        let q = self.field.order();
        let exp = (q - Integer::one()) / (self.p().clone() - Integer::one());
        self.pow_int(&exp).coeffs[0].clone()
    }

    /// The minimal polynomial of `a` over `F_p`, little-endian and monic.
    pub fn minimal_polynomial(&self) -> Vec<Integer> {
        // Product of (x - a^{p^i}) over the distinct conjugates.
        let p = self.p().clone();
        let mut conjugates: Vec<Vec<Integer>> = Vec::new();
        let mut cur = self.clone();
        loop {
            let c = cur.coeffs.clone();
            if conjugates.iter().any(|q| *q == c) {
                break;
            }
            conjugates.push(c);
            cur = cur.frobenius();
        }
        // Multiply linear factors (x - r) with coefficients in the field; the
        // product lands in F_p (constant polynomials).
        let mut poly = vec![self.field.one()]; // start with 1
        for r in &conjugates {
            let root = FiniteFieldElement {
                coeffs: r.clone(),
                field: self.field.clone(),
            };
            // multiply poly by (x - root)
            let mut next = vec![self.field.zero(); poly.len() + 1];
            for (i, coef) in poly.iter().enumerate() {
                next[i + 1] = next[i + 1].clone() + coef.clone(); // * x
                next[i] = next[i].clone() - coef.clone() * root.clone(); // * (-root)
            }
            poly = next;
        }
        // Extract F_p representatives (constant term of each field element).
        poly.iter()
            .map(|c| redp(c.coeffs[0].clone(), &p))
            .collect()
    }

    /// The multiplicative order of a non-zero element (divides `p^n - 1`).
    pub fn multiplicative_order(&self) -> Option<Integer> {
        if self.is_zero() {
            return None;
        }
        let group_order = self.field.order() - Integer::one();
        // Start from the full group order and strip prime factors.
        let mut order = group_order.clone();
        for (prime, mult) in factor(&group_order) {
            for _ in 0..mult {
                let candidate = order.clone() / prime.clone();
                if self.pow_int(&candidate).is_one() {
                    order = candidate;
                } else {
                    break;
                }
            }
        }
        Some(order)
    }

    /// Whether the element generates `GF(p^n)^*` (a primitive element).
    pub fn is_primitive(&self) -> bool {
        match self.multiplicative_order() {
            Some(o) => o == self.field.order() - Integer::one(),
            None => false,
        }
    }

    /// Discrete logarithm by baby-step / giant-step: return `x` with
    /// `base^x = self`, if it lies in the cyclic group generated by `base`.
    ///
    /// MAGMA `Log(b, x)`. Runs in `O(sqrt(N))` time and space, `N = p^n - 1`.
    pub fn discrete_log(base: &FiniteFieldElement, target: &FiniteFieldElement) -> Result<Integer> {
        base.assert_same(target);
        if base.is_zero() || base.is_one() {
            return Err(MathError::InvalidArgument(
                "base must be neither 0 nor 1".into(),
            ));
        }
        if target.is_zero() {
            return Err(MathError::InvalidArgument("log of zero is undefined".into()));
        }
        if target.is_one() {
            return Ok(Integer::zero());
        }
        let group_order = base.field.order() - Integer::one();
        let m = group_order.sqrt()? + Integer::one();
        let m_usize = m.to_usize().ok_or_else(|| {
            MathError::NumericalError("group too large for baby-step/giant-step".into())
        })?;

        let mut baby: HashMap<Vec<Integer>, usize> = HashMap::new();
        let mut power = base.field.one();
        for j in 0..m_usize {
            baby.entry(power.coeffs.clone()).or_insert(j);
            power = power * base.clone();
        }
        // giant step multiplier = base^{-m}
        let base_m = base.pow_int(&m);
        let mult = base_m.inverse()?;
        let mut gamma = target.clone();
        for i in 0..m_usize {
            if let Some(&j) = baby.get(&gamma.coeffs) {
                let x = Integer::from(i as i64) * m.clone() + Integer::from(j as i64);
                return Ok(redp(x, &group_order));
            }
            gamma = gamma * mult.clone();
        }
        Err(MathError::InvalidArgument(
            "discrete logarithm not found (target not in <base>)".into(),
        ))
    }
}

impl fmt::Debug for FiniteFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl fmt::Display for FiniteFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // print as a polynomial in the generator w
        let mut terms: Vec<String> = Vec::new();
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            match i {
                0 => terms.push(format!("{c}")),
                1 => terms.push(format!("{c}*w")),
                _ => terms.push(format!("{c}*w^{i}")),
            }
        }
        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

impl PartialEq for FiniteFieldElement {
    fn eq(&self, other: &Self) -> bool {
        self.field.data == other.field.data && self.coeffs == other.coeffs
    }
}
impl Eq for FiniteFieldElement {}

impl Add for FiniteFieldElement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.assert_same(&other);
        let p = self.p().clone();
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| redp(a.clone() + b.clone(), &p))
            .collect();
        FiniteFieldElement {
            coeffs,
            field: self.field,
        }
    }
}

impl Sub for FiniteFieldElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.assert_same(&other);
        let p = self.p().clone();
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| redp(a.clone() - b.clone(), &p))
            .collect();
        FiniteFieldElement {
            coeffs,
            field: self.field,
        }
    }
}

impl Mul for FiniteFieldElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.assert_same(&other);
        let coeffs = poly_mul_mod(&self.coeffs, &other.coeffs, self.modulus(), self.p());
        FiniteFieldElement {
            coeffs,
            field: self.field,
        }
    }
}

impl Neg for FiniteFieldElement {
    type Output = Self;
    fn neg(self) -> Self {
        let p = self.p().clone();
        let coeffs = self
            .coeffs
            .iter()
            .map(|c| redp(-c.clone(), &p))
            .collect();
        FiniteFieldElement {
            coeffs,
            field: self.field,
        }
    }
}

impl Div for FiniteFieldElement {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.assert_same(&other);
        let inv = other.inverse().expect("division by non-invertible element");
        self * inv
    }
}

impl Ring for FiniteFieldElement {
    fn zero() -> Self {
        panic!("FiniteFieldElement::zero() needs a parent field; use FiniteField::zero()");
    }
    fn one() -> Self {
        panic!("FiniteFieldElement::one() needs a parent field; use FiniteField::one()");
    }
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }
    fn is_one(&self) -> bool {
        self.coeffs[0].is_one() && self.coeffs[1..].iter().all(|c| c.is_zero())
    }
}

impl CommutativeRing for FiniteFieldElement {}

impl Field for FiniteFieldElement {
    fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        // Fermat: a^{p^n - 2} = a^{-1} (a in GF(p^n)^*, order p^n - 1).
        let exp = self.field.order() - Integer::from(2);
        Ok(self.pow_int(&exp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gf4_construction_and_reduction() {
        // GF(2^2) via Conway x^2 + x + 1.
        let f = FiniteField::new(Integer::from(2), 2).unwrap();
        assert_eq!(f.degree(), 2);
        assert_eq!(f.order(), Integer::from(4));
        assert!(f.is_conway());
        // alpha = generator
        let a = f.generator();
        // alpha^2 must reduce to alpha + 1, NOT 0 (the legacy bug).
        let a2 = a.clone() * a.clone();
        assert_eq!(a2, f.element(vec![Integer::from(1), Integer::from(1)]));
        // alpha^3 = 1
        let a3 = a2 * a;
        assert!(a3.is_one());
    }

    #[test]
    fn gf_p_prime_field() {
        let f = FiniteField::new(Integer::from(7), 1).unwrap();
        assert!(f.is_prime_field());
        let a = f.from_int(Integer::from(3));
        let b = f.from_int(Integer::from(5));
        assert_eq!((a.clone() + b.clone()), f.from_int(Integer::from(1))); // 8 = 1
        assert_eq!((a.clone() * b.clone()), f.from_int(Integer::from(1))); // 15 = 1
        assert_eq!(a.inverse().unwrap(), f.from_int(Integer::from(5))); // 3*5=1
    }

    #[test]
    fn inverse_in_gf8() {
        let f = FiniteField::new(Integer::from(2), 3).unwrap();
        let a = f.generator();
        let inv = a.inverse().unwrap();
        assert!((a * inv).is_one());
    }

    #[test]
    fn norm_trace_in_fp() {
        // GF(4): N(alpha) = 1, Tr(alpha) = 1.
        let f = FiniteField::new(Integer::from(2), 2).unwrap();
        let a = f.generator();
        assert_eq!(a.norm(), Integer::from(1));
        assert_eq!(a.trace(), Integer::from(1));
    }

    #[test]
    fn frobenius_has_right_order() {
        // GF(8): Frobenius has order 3.
        let f = FiniteField::new(Integer::from(2), 3).unwrap();
        let a = f.generator();
        assert_ne!(a.frobenius(), a);
        assert_eq!(a.frobenius_pow(3), a);
    }

    #[test]
    fn minimal_polynomial_of_generator() {
        // In GF(4), min poly of alpha is x^2 + x + 1 -> [1,1,1].
        let f = FiniteField::new(Integer::from(2), 2).unwrap();
        let a = f.generator();
        let mp = a.minimal_polynomial();
        assert_eq!(
            mp,
            vec![Integer::from(1), Integer::from(1), Integer::from(1)]
        );
        // constant 1 has min poly x - 1 -> [-1 mod2, 1] = [1,1]
        let one = f.one();
        assert_eq!(one.minimal_polynomial(), vec![Integer::from(1), Integer::from(1)]);
    }

    #[test]
    fn multiplicative_order_and_primitive() {
        // In GF(7), 3 is primitive (order 6), 2 has order 3.
        let f = FiniteField::new(Integer::from(7), 1).unwrap();
        assert_eq!(
            f.from_int(Integer::from(3)).multiplicative_order(),
            Some(Integer::from(6))
        );
        assert!(f.from_int(Integer::from(3)).is_primitive());
        assert_eq!(
            f.from_int(Integer::from(2)).multiplicative_order(),
            Some(Integer::from(3))
        );
        // alpha in GF(8) is primitive (order 7).
        let g = FiniteField::new(Integer::from(2), 3).unwrap();
        assert_eq!(g.generator().multiplicative_order(), Some(Integer::from(7)));
    }

    #[test]
    fn discrete_log_gf8() {
        let f = FiniteField::new(Integer::from(2), 3).unwrap();
        let g = f.generator(); // primitive
        let target = g.pow_int(&Integer::from(5));
        let x = FiniteFieldElement::discrete_log(&g, &target).unwrap();
        assert_eq!(g.pow_int(&x), target);
    }

    #[test]
    fn discrete_log_prime_field() {
        // 2 is primitive mod 11; log_2(7) with 2^7 = 128 = 7 mod 11.
        let f = FiniteField::new(Integer::from(11), 1).unwrap();
        let g = f.from_int(Integer::from(2));
        let target = f.from_int(Integer::from(7));
        let x = FiniteFieldElement::discrete_log(&g, &target).unwrap();
        assert_eq!(g.pow_int(&x), target);
    }

    #[test]
    fn unique_parent_shared() {
        let f1 = FiniteField::new(Integer::from(3), 2).unwrap();
        let f2 = FiniteField::new(Integer::from(3), 2).unwrap();
        assert!(f1.same_field(&f2));
        // Arc is shared by UniqueRepresentation.
        assert!(Arc::ptr_eq(&f1.data, &f2.data));
    }

    #[test]
    fn custom_modulus_rejects_reducible() {
        // x^2 + 1 is reducible over F_2 (= (x+1)^2), so with_modulus must reject.
        let bad = FiniteField::with_modulus(
            Integer::from(2),
            vec![Integer::from(1), Integer::from(0), Integer::from(1)],
        );
        assert!(bad.is_err());
        // x^2 + x + 1 is irreducible over F_2.
        let good = FiniteField::with_modulus(
            Integer::from(2),
            vec![Integer::from(1), Integer::from(1), Integer::from(1)],
        );
        assert!(good.is_ok());
    }
}
