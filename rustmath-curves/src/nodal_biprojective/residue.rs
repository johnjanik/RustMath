//! Residue fields of closed points, for the certificate and the local classifier (rare, small
//! computations). `K = F_p[Y]/(g)` on `fp_factor` polynomials, and the quadratic extension
//! `K[X]/(X^2 + g1 X + g0)` for a fibre with two conjugate singular points.
//!
//! Specification: MAGMA Handbook ch. 114 *Algebraic Curves* (digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`): 114.4.1 points with coordinates in an
//! extension of the base field, 114.5.1 `HasSingularPointsOverExtension`; and ch. 21 *Finite
//! Fields* for `Norm` and `IsSquare` in these fields.
//!
//! These are small private helpers built on `rustmath_polynomials::fp_factor`, not a new
//! numeric type. Squares: `a != 0` is a square iff `N(a)` is a square in `F_p` (the norm of a
//! finite field extension is onto and `a^((q-1)/2) = N(a)^((p-1)/2)`; R. Lidl, H. Niederreiter,
//! *Finite fields*, Thm. 2.28). Norms: `N_{K/F_p}(h(y0)) = Res(g, h)` for monic `g`.

use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor as fp;
use std::fmt::Debug;

pub(crate) type Poly = Vec<i64>;

pub(crate) fn powm(b: i64, mut e: u64, p: i64) -> i64 {
    let mut r = 1i64;
    let mut b = b.rem_euclid(p);
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % p;
        }
        b = b * b % p;
        e >>= 1;
    }
    r
}

/// The Legendre symbol `(a/p)`, `p` an odd prime.
pub(crate) fn legendre(a: i64, p: i64) -> i8 {
    let a = a.rem_euclid(p);
    if a == 0 {
        0
    } else if powm(a, ((p - 1) / 2) as u64, p) == 1 {
        1
    } else {
        -1
    }
}

/// `Res(a, b)` over `F_p` (Euclid: `Res(a, b) = (-1)^(da db) lc(b)^(da - dr) Res(b, a mod b)`).
pub(crate) fn resultant(a: &[i64], b: &[i64], p: i64) -> i64 {
    let da = fp::degree(a);
    let db = fp::degree(b);
    if da < 0 || db < 0 {
        return 0;
    }
    if db == 0 {
        return powm(fp::trim(b)[0], da as u64, p);
    }
    if da == 0 {
        return powm(fp::trim(a)[0], db as u64, p);
    }
    let r = fp::div_mod(a, b, p).1;
    let dr = fp::degree(&r);
    if dr < 0 {
        return 0;
    }
    let lcb = *fp::trim(b).last().expect("nonzero");
    let sign = if (da * db) % 2 == 1 { p - 1 } else { 1 };
    let t = powm(lcb, (da - dr) as u64, p);
    (sign * t % p) * resultant(b, &r, p) % p
}

/// A finite field of characteristic `p`, for the rare computations at singular points.
pub(crate) trait ResField: Clone + Debug {
    type E: Clone + PartialEq + Debug;
    fn p(&self) -> i64;
    /// The degree over `F_p`.
    fn degree(&self) -> usize;
    fn zero(&self) -> Self::E;
    fn one(&self) -> Self::E;
    fn constant(&self, c: i64) -> Self::E;
    fn add(&self, a: &Self::E, b: &Self::E) -> Self::E;
    fn sub(&self, a: &Self::E, b: &Self::E) -> Self::E;
    fn mul(&self, a: &Self::E, b: &Self::E) -> Self::E;
    fn is_zero(&self, a: &Self::E) -> bool;
    fn inv(&self, a: &Self::E) -> Option<Self::E>;
    /// `N_{kappa/F_p}(a)` in `[0, p)`.
    fn norm(&self, a: &Self::E) -> i64;
    /// The element with base-`p` digits `n` in some fixed `F_p`-basis (an enumeration).
    fn element(&self, n: u64) -> Self::E;
    /// `Some(c)` if `a` lies in the prime field, as `c` in `[0, p)`.
    fn to_fp(&self, a: &Self::E) -> Option<i64>;

    /// The characteristic polynomial over `F_p` of `a` (monic, ascending, degree
    /// `self.degree()`): the product of `T - a^(p^i)`, `0 <= i < degree`.
    fn charpoly(&self, a: &Self::E) -> Vec<i64> {
        let pz = Integer::from(self.p());
        let mut prod: Vec<Self::E> = vec![self.one()];
        let mut c = a.clone();
        for _ in 0..self.degree() {
            prod = rmul(self, &prod, &[self.neg(&c), self.one()]);
            c = self.pow(&c, &pz);
        }
        prod.iter()
            .map(|x| {
                self.to_fp(x)
                    .expect("the characteristic polynomial lies over F_p")
            })
            .collect()
    }

    fn neg(&self, a: &Self::E) -> Self::E {
        self.sub(&self.zero(), a)
    }
    fn scale(&self, a: &Self::E, c: i64) -> Self::E {
        self.mul(a, &self.constant(c))
    }
    /// True iff `a` is a square (0 included).
    fn is_square(&self, a: &Self::E) -> bool {
        self.is_zero(a) || legendre(self.norm(a), self.p()) == 1
    }
    /// `|kappa| = p^degree`.
    fn order(&self) -> Integer {
        Integer::from(self.p()).pow(self.degree() as u32)
    }
    fn pow(&self, a: &Self::E, e: &Integer) -> Self::E {
        let two = Integer::from(2);
        let mut bits = Vec::new();
        let mut x = e.clone();
        while x.signum() > 0 {
            bits.push(!(&x % &two).is_zero());
            x = x / two.clone();
        }
        let mut r = self.one();
        for &b in bits.iter().rev() {
            r = self.mul(&r, &r);
            if b {
                r = self.mul(&r, a);
            }
        }
        r
    }
    /// A square root of `a` (Tonelli-Shanks), or `None` if `a` is not a square. The auxiliary
    /// nonsquare is found by enumerating elements; half of the nonzero elements qualify, so the
    /// scan ends after at most `(q + 1) / 2` steps. It decides nothing: the root is checked.
    fn sqrt(&self, a: &Self::E) -> Option<Self::E> {
        if self.is_zero(a) {
            return Some(self.zero());
        }
        if !self.is_square(a) {
            return None;
        }
        let one = Integer::one();
        let qm1 = self.order() - one.clone();
        let two = Integer::from(2);
        // q - 1 = 2^s t, t odd
        let mut t = qm1.clone();
        let mut s = 0u32;
        while (&t % &two).is_zero() {
            t = t / two.clone();
            s += 1;
        }
        let mut n = 1u64;
        let z = loop {
            let c = self.element(n);
            if !self.is_zero(&c) && !self.is_square(&c) {
                break c;
            }
            n += 1;
        };
        let mut m = s;
        let mut c = self.pow(&z, &t);
        let mut tt = self.pow(a, &t);
        let mut r = self.pow(a, &((t.clone() + one) / two));
        while tt != self.one() {
            // least i with tt^(2^i) = 1
            let mut i = 0u32;
            let mut x = tt.clone();
            while x != self.one() {
                x = self.mul(&x, &x);
                i += 1;
            }
            let mut b = c.clone();
            for _ in 0..(m - i - 1) {
                b = self.mul(&b, &b);
            }
            m = i;
            c = self.mul(&b, &b);
            tt = self.mul(&tt, &c);
            r = self.mul(&r, &b);
        }
        debug_assert_eq!(self.mul(&r, &r), *a);
        if self.mul(&r, &r) == *a {
            Some(r)
        } else {
            None
        }
    }
}

/// `K = F_p[Y]/(g)` for a monic irreducible `g` over `F_p`. Elements are trimmed `fp_factor`
/// polynomials of degree `< deg g`.
#[derive(Clone, Debug)]
pub(crate) struct PolyField {
    p: i64,
    g: Poly,
    d: usize,
}

impl PolyField {
    /// `g` must be monic irreducible over `F_p` (the irreducible factors returned by
    /// `fp_factor::factor_with_multiplicity` are).
    pub(crate) fn new(p: i64, g: &[i64]) -> Self {
        let g = fp::make_monic(&fp::trim(g), p);
        let d = fp::degree(&g) as usize;
        assert!(d >= 1);
        PolyField { p, g, d }
    }
    fn red(&self, a: &[i64]) -> Poly {
        fp::div_mod(a, &self.g, self.p).1
    }
    /// The class of `Y`.
    pub(crate) fn generator(&self) -> Poly {
        self.red(&[0, 1])
    }
    /// For `d = 1`: the element as an integer in `[0, p)`.
    pub(crate) fn as_fp(&self, a: &Poly) -> Option<i64> {
        let t = fp::trim(a);
        if t.len() == 1 {
            Some(t[0])
        } else {
            None
        }
    }
}

impl ResField for PolyField {
    type E = Poly;
    fn p(&self) -> i64 {
        self.p
    }
    fn degree(&self) -> usize {
        self.d
    }
    fn zero(&self) -> Poly {
        vec![0]
    }
    fn one(&self) -> Poly {
        vec![1]
    }
    fn constant(&self, c: i64) -> Poly {
        vec![c.rem_euclid(self.p)]
    }
    fn add(&self, a: &Poly, b: &Poly) -> Poly {
        fp::add(a, b, self.p)
    }
    fn sub(&self, a: &Poly, b: &Poly) -> Poly {
        fp::sub(a, b, self.p)
    }
    fn mul(&self, a: &Poly, b: &Poly) -> Poly {
        if self.d == 1 {
            return vec![(fp::trim(a)[0] * fp::trim(b)[0]) % self.p];
        }
        self.red(&fp::mul(a, b, self.p))
    }
    fn is_zero(&self, a: &Poly) -> bool {
        fp::is_zero(a)
    }
    fn inv(&self, a: &Poly) -> Option<Poly> {
        if fp::is_zero(a) {
            return None;
        }
        let (d, s, _t) = fp::extended_gcd(a, &self.g, self.p);
        if fp::degree(&d) != 0 {
            return None;
        }
        let inv = fp::mod_inv(fp::trim(&d)[0], self.p)?;
        Some(self.red(&fp::mul(&s, &[inv], self.p)))
    }
    fn norm(&self, a: &Poly) -> i64 {
        resultant(&self.g, a, self.p).rem_euclid(self.p)
    }
    fn to_fp(&self, a: &Poly) -> Option<i64> {
        self.as_fp(a)
    }
    fn element(&self, mut n: u64) -> Poly {
        let p = self.p as u64;
        let mut c = vec![0i64; self.d];
        for ci in c.iter_mut() {
            *ci = (n % p) as i64;
            n /= p;
        }
        fp::trim(&c)
    }
}

/// `E = K[X]/(X^2 + g1 X + g0)` for an irreducible quadratic over `K`; elements `a0 + a1 X`.
#[derive(Clone, Debug)]
pub(crate) struct QuadExt {
    pub(crate) base: PolyField,
    g1: Poly,
    g0: Poly,
}

impl QuadExt {
    pub(crate) fn new(base: PolyField, g1: Poly, g0: Poly) -> Self {
        QuadExt { base, g1, g0 }
    }
    /// The class of `X`.
    pub(crate) fn generator(&self) -> (Poly, Poly) {
        (self.base.zero(), self.base.one())
    }
    pub(crate) fn embed(&self, a: Poly) -> (Poly, Poly) {
        (a, self.base.zero())
    }
}

impl ResField for QuadExt {
    type E = (Poly, Poly);
    fn p(&self) -> i64 {
        self.base.p
    }
    fn degree(&self) -> usize {
        2 * self.base.d
    }
    fn zero(&self) -> (Poly, Poly) {
        (self.base.zero(), self.base.zero())
    }
    fn one(&self) -> (Poly, Poly) {
        (self.base.one(), self.base.zero())
    }
    fn constant(&self, c: i64) -> (Poly, Poly) {
        (self.base.constant(c), self.base.zero())
    }
    fn add(&self, a: &Self::E, b: &Self::E) -> Self::E {
        (self.base.add(&a.0, &b.0), self.base.add(&a.1, &b.1))
    }
    fn sub(&self, a: &Self::E, b: &Self::E) -> Self::E {
        (self.base.sub(&a.0, &b.0), self.base.sub(&a.1, &b.1))
    }
    fn mul(&self, a: &Self::E, b: &Self::E) -> Self::E {
        let k = &self.base;
        let c0 = k.mul(&a.0, &b.0);
        let c1 = k.add(&k.mul(&a.0, &b.1), &k.mul(&a.1, &b.0));
        let c2 = k.mul(&a.1, &b.1);
        // X^2 = -g1 X - g0
        (
            k.sub(&c0, &k.mul(&c2, &self.g0)),
            k.sub(&c1, &k.mul(&c2, &self.g1)),
        )
    }
    fn is_zero(&self, a: &Self::E) -> bool {
        self.base.is_zero(&a.0) && self.base.is_zero(&a.1)
    }
    fn inv(&self, a: &Self::E) -> Option<Self::E> {
        let k = &self.base;
        // conj = (a0 - g1 a1) - a1 X; a conj = N = a0^2 - g1 a0 a1 + g0 a1^2
        let n = self.norm_to_base(a);
        let ni = k.inv(&n)?;
        let c0 = k.sub(&a.0, &k.mul(&self.g1, &a.1));
        let c1 = k.neg(&a.1);
        Some((k.mul(&c0, &ni), k.mul(&c1, &ni)))
    }
    fn norm(&self, a: &Self::E) -> i64 {
        self.base.norm(&self.norm_to_base(a))
    }
    fn to_fp(&self, a: &Self::E) -> Option<i64> {
        if self.base.is_zero(&a.1) {
            self.base.as_fp(&a.0)
        } else {
            None
        }
    }
    fn element(&self, n: u64) -> Self::E {
        let q = self.base.order().to_i64().unwrap_or(i64::MAX) as u64;
        (self.base.element(n % q), self.base.element(n / q))
    }
}

impl QuadExt {
    fn norm_to_base(&self, a: &(Poly, Poly)) -> Poly {
        let k = &self.base;
        let t0 = k.mul(&a.0, &a.0);
        let t1 = k.mul(&k.mul(&self.g1, &a.0), &a.1);
        let t2 = k.mul(&k.mul(&self.g0, &a.1), &a.1);
        k.add(&k.sub(&t0, &t1), &t2)
    }
}

// ------------------------------------------------------------------------------------------
// Polynomials over a ResField (coefficient vectors, little-endian, trimmed on demand).
// ------------------------------------------------------------------------------------------

pub(crate) fn rtrim<F: ResField>(f: &F, mut a: Vec<F::E>) -> Vec<F::E> {
    while a.last().is_some_and(|c| f.is_zero(c)) {
        a.pop();
    }
    a
}

/// Degree, `-1` for zero.
pub(crate) fn rdeg<F: ResField>(f: &F, a: &[F::E]) -> isize {
    let mut d = a.len() as isize - 1;
    while d >= 0 && f.is_zero(&a[d as usize]) {
        d -= 1;
    }
    d
}

pub(crate) fn rmonic<F: ResField>(f: &F, a: Vec<F::E>) -> Vec<F::E> {
    let a = rtrim(f, a);
    if a.is_empty() {
        return a;
    }
    let inv = f.inv(a.last().expect("nonzero")).expect("field");
    a.iter().map(|c| f.mul(c, &inv)).collect()
}

pub(crate) fn rmul<F: ResField>(f: &F, a: &[F::E], b: &[F::E]) -> Vec<F::E> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![f.zero(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if f.is_zero(x) {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            r[i + j] = f.add(&r[i + j], &f.mul(x, y));
        }
    }
    rtrim(f, r)
}

/// `a mod b`, `b` nonzero.
pub(crate) fn rrem<F: ResField>(f: &F, a: &[F::E], b: &[F::E]) -> Vec<F::E> {
    let mut a = rtrim(f, a.to_vec());
    let b = rtrim(f, b.to_vec());
    assert!(!b.is_empty(), "rrem by zero");
    let inv = f.inv(b.last().expect("nonzero")).expect("field");
    while a.len() >= b.len() {
        let c = f.mul(a.last().expect("nonzero"), &inv);
        let s = a.len() - b.len();
        for (j, bj) in b.iter().enumerate() {
            a[s + j] = f.sub(&a[s + j], &f.mul(&c, bj));
        }
        a = rtrim(f, a);
    }
    a
}

/// Monic gcd (empty for `gcd(0, 0)`).
pub(crate) fn rgcd<F: ResField>(f: &F, a: &[F::E], b: &[F::E]) -> Vec<F::E> {
    let mut a = rtrim(f, a.to_vec());
    let mut b = rtrim(f, b.to_vec());
    while !b.is_empty() {
        let r = rrem(f, &a, &b);
        a = b;
        b = r;
    }
    rmonic(f, a)
}

pub(crate) fn rderiv<F: ResField>(f: &F, a: &[F::E]) -> Vec<F::E> {
    let d: Vec<F::E> = a
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| f.scale(c, i as i64))
        .collect();
    rtrim(f, d)
}

/// `base^e mod m` over `F` (square-and-multiply).
pub(crate) fn rpow_mod<F: ResField>(f: &F, base: &[F::E], e: &Integer, m: &[F::E]) -> Vec<F::E> {
    let two = Integer::from(2);
    let mut bits = Vec::new();
    let mut x = e.clone();
    while x.signum() > 0 {
        bits.push(!(&x % &two).is_zero());
        x = x / two.clone();
    }
    let b = rrem(f, base, m);
    let mut r = rrem(f, &[f.one()], m);
    for &bit in bits.iter().rev() {
        r = rrem(f, &rmul(f, &r, &r), m);
        if bit {
            r = rrem(f, &rmul(f, &r, &b), m);
        }
    }
    r
}

/// The degrees of the irreducible factors of a squarefree `a` over `F` (distinct-degree
/// factorization: `gcd(a, x^(q^i) - x)`, J. von zur Gathen, J. Gerhard, *Modern computer
/// algebra*, Algorithm 14.3). `None` if `a` is not squarefree.
pub(crate) fn factor_degrees<F: ResField>(f: &F, a: &[F::E]) -> Option<Vec<usize>> {
    let a = rmonic(f, a.to_vec());
    let d = rdeg(f, &a);
    if d <= 0 {
        return Some(Vec::new());
    }
    let g = rgcd(f, &a, &rderiv(f, &a));
    if rdeg(f, &g) > 0 {
        return None;
    }
    let q = f.order();
    let mut rest = a;
    let mut h = rrem(f, &[f.zero(), f.one()], &rest);
    let mut out = Vec::new();
    let mut i = 1usize;
    while rdeg(f, &rest) > 0 {
        if 2 * i > rdeg(f, &rest) as usize {
            out.push(rdeg(f, &rest) as usize);
            break;
        }
        // h = x^(q^i) mod rest
        h = rpow_mod(f, &h, &q, &rest);
        let mut hx = h.clone();
        while hx.len() < 2 {
            hx.push(f.zero());
        }
        hx[1] = f.sub(&hx[1], &f.one());
        let gi = rgcd(f, &rest, &hx);
        let dg = rdeg(f, &gi);
        if dg > 0 {
            for _ in 0..(dg as usize / i) {
                out.push(i);
            }
            // rest = rest / gi
            rest = rdiv_exact(f, &rest, &gi);
            h = rrem(f, &h, &rest);
        }
        i += 1;
    }
    out.sort();
    Some(out)
}

fn rdiv_exact<F: ResField>(f: &F, a: &[F::E], b: &[F::E]) -> Vec<F::E> {
    let mut a = rtrim(f, a.to_vec());
    let b = rtrim(f, b.to_vec());
    let inv = f.inv(b.last().expect("nonzero")).expect("field");
    let mut q = vec![f.zero(); a.len().saturating_sub(b.len()) + 1];
    while a.len() >= b.len() && !a.is_empty() {
        let c = f.mul(a.last().expect("nonzero"), &inv);
        let s = a.len() - b.len();
        q[s] = c.clone();
        for (j, bj) in b.iter().enumerate() {
            a[s + j] = f.sub(&a[s + j], &f.mul(&c, bj));
        }
        a = rtrim(f, a);
    }
    debug_assert!(a.is_empty(), "rdiv_exact: not exact");
    rtrim(f, q)
}

/// `sum c_j y0^j` for a polynomial over `F_p` (coefficients in `[0, p)`) at `y0` in `F`.
pub(crate) fn eval_fp_poly<F: ResField>(f: &F, c: &[i64], y0: &F::E) -> F::E {
    let mut r = f.zero();
    for &cj in c.iter().rev() {
        r = f.add(&f.mul(&r, y0), &f.constant(cj));
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residue_field_arithmetic() {
        // K = F_7[Y]/(Y^2 + 1) (-1 is a nonsquare mod 7).
        let k = PolyField::new(7, &[1, 0, 1]);
        let y = k.generator();
        assert_eq!(k.mul(&y, &y), vec![6]);
        assert_eq!(k.norm(&y), 1); // N(i) = i * (-i) = 1
        let a = vec![3, 2];
        let ai = k.inv(&a).unwrap();
        assert_eq!(k.mul(&a, &ai), vec![1]);
        // sqrt of -1 = Y^2 in K is +-Y
        let r = k.sqrt(&vec![6]).unwrap();
        assert_eq!(k.mul(&r, &r), vec![6]);
        // squares in GF(49): exactly 24 nonzero ones
        let sq = (1..49).filter(|&n| k.is_square(&k.element(n))).count();
        assert_eq!(sq, 24);
        // E = K[X]/(X^2 - c) for a nonsquare c of K: N_E(X) = -(-c) ... check E is a field of
        // order 7^4 with (7^4 - 1)/2 nonzero squares.
        let c = (1..49)
            .map(|n| k.element(n))
            .find(|c| !k.is_square(c))
            .unwrap();
        let e = QuadExt::new(k.clone(), vec![0], k.neg(&c));
        let x = e.generator();
        assert_eq!(e.mul(&x, &x), e.embed(c.clone()));
        let xi = e.inv(&x).unwrap();
        assert_eq!(e.mul(&x, &xi), e.one());
        assert!(!e.is_square(&x));
        assert!(e.is_square(&e.embed(c)));
        // factor degrees over K of x^2 + 1 (splits: i in K) and x^2 - 3 (3 nonsquare mod 7,
        // N(3) = 9 = 2 is a square mod 7, so 3 is a square in GF(49)).
        assert_eq!(
            factor_degrees(&k, &[vec![1], vec![0], vec![1]]),
            Some(vec![1, 1])
        );
        assert_eq!(
            factor_degrees(&k, &[vec![4], vec![0], vec![1]]),
            Some(vec![1, 1])
        );
        // x^3 - 2 over F_7: 2 is a cube mod 7? cubes mod 7 are {0, 1, 6}: irreducible over
        // F_7; over GF(49) still irreducible (3 does not divide 2).
        let f7 = PolyField::new(7, &[0, 1]);
        assert_eq!(
            factor_degrees(&f7, &[vec![5], vec![0], vec![0], vec![1]]),
            Some(vec![3])
        );
        assert_eq!(
            factor_degrees(&k, &[vec![5], vec![0], vec![0], vec![1]]),
            Some(vec![3])
        );
        assert_eq!(factor_degrees(&f7, &[vec![1], vec![2], vec![1]]), None);
    }

    #[test]
    fn resultants() {
        // Res(x^2 + 1, x + 2) over F_7: (-2)^2 + 1 = 5
        assert_eq!(resultant(&[1, 0, 1], &[2, 1], 7), 5);
        assert_eq!(resultant(&[2, 1], &[1, 0, 1], 7), 5);
    }
}
