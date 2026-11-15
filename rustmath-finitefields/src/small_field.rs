//! `GF(p^K)` on machine words, with a distinct-root count for polynomials over it.
//!
//! Specification: MAGMA Handbook ch. 21 *Finite Fields* (digest
//! `rustmath-lava/docs/Chapter_21_Finite_Fields.md`): `FiniteField`, `Frobenius` (21.4.6),
//! `Norm` (21.4.6), `IsSquare` (21.4.4), and ch. 23 `Roots` (the root count is the degree of
//! `gcd(f, x^q - x)`). This is a specialised kernel, not a new general field type: the canonical
//! `GF(p^n)` of this crate is [`crate::FiniteField`].
//!
//! Why it exists. [`crate::FiniteField`] stores coefficients as heap
//! [`Integer`](rustmath_integers::Integer)s. Counting the
//! roots of one quartic over `GF(31^4)` with [`crate::FqPoly::pow_mod`] and [`crate::FqPoly::gcd`]
//! costs about 4.0 ms (release, one thread), against about 5 us here (WP-F design, Section 8.1,
//! `docs/workstreams/m23_golay/HANDOFF.md` Section 3). The point counts of the M23 curve need
//! about `p^4 / 4` such root counts per prime (9.7 million at p = 79).
//!
//! Representation. `p` is a proved prime below `2^16`, `K <= 8` and `p^K < 2^63`. An element is
//! `[u64; K]`, the little-endian coefficients in `[0, p)` of a polynomial of degree `< K` in
//! `F_p[x]/(m(x))`, `m` monic irreducible of degree `K`. Products are accumulated in `u64` without
//! reduction and reduced once per coefficient (Barrett reduction modulo `p`). Every accumulator
//! stays below `2^48`: a product of two residues is below `2^32`, and no accumulator receives more
//! than `2^16` of them.
//!
//! Algorithms:
//! - Frobenius `a -> a^p` is F_p-linear: a precomputed `K x K` matrix (the images of `x^i`).
//! - Norm and inverse: `N(a) = a a^p ... a^(p^(K-1))` lies in `F_p`, and
//!   `a^(-1) = (a^p ... a^(p^(K-1))) / N(a)` (\[LN97\] R. Lidl, H. Niederreiter, *Finite fields*,
//!   Thm. 2.28 and Def. 2.22).
//! - Squares: for `a != 0`, `a` is a square in `GF(q)` iff `N(a)` is a square in `F_p` (the norm
//!   is onto and `a^((q-1)/2) = N(a)^((p-1)/2)`).
//! - Root count: `#{roots of f in GF(q)} = deg gcd(f, x^q - x)` (\[Rab80\] M. O. Rabin, *Probabilistic
//!   algorithms in finite fields*, SIAM J. Comput. 9 (1980): `x^q - x` is the product of the
//!   monic linear polynomials). For `deg f <= 4`, `x^p mod f` by square-and-multiply and then
//!   `x^q` by the semilinear map `g^p = sum sigma(g_j) (x^p)^j mod f` applied `K - 1` times
//!   (\[vzGG\] J. von zur Gathen, J. Gerhard, *Modern computer algebra*, Section 14.7, the
//!   Frobenius as a linear map); fixed-size arrays. Any degree: the same on vectors, with `x^q`
//!   by square-and-multiply.

// Index loops over the fixed-size coefficient arrays mirror the formulas; keep them.
#![allow(clippy::needless_range_loop)]

use rustmath_core::{MathError, Result};
use rustmath_integers::prime::is_prime_u64;
use rustmath_polynomials::fp_factor;

/// Largest prime (exclusive bound) accepted by [`SmallFq`]: `p < 2^16`.
pub const SMALL_FIELD_P_BOUND: u64 = 1 << 16;
/// Largest extension degree accepted by [`SmallFq`].
pub const SMALL_FIELD_MAX_DEGREE: usize = 8;

/// An element of a [`SmallFq<K>`]: little-endian coefficients in `[0, p)`.
pub type SmallFqElt<const K: usize> = [u64; K];

/// `GF(p^K)` on machine words: `p < 2^16` (proved prime), `1 <= K <= 8`, `p^K < 2^63`.
#[derive(Clone, Debug)]
pub struct SmallFq<const K: usize> {
    p: u64,
    /// `floor((2^64 - 1) / p)`, the Barrett constant.
    bar: u64,
    order: u64,
    /// The defining polynomial, monic, little-endian, length `K + 1`.
    modulus: Vec<u64>,
    /// `x^K = sum negm[j] x^j` in the field.
    negm: [u64; K],
    /// `frob[i]` = the image of `x^i` under `a -> a^p`.
    frob: [[u64; K]; K],
    /// Inverses modulo `p` (`inv_p[0] = 0`).
    inv_p: Vec<u32>,
    /// `square_p[a]` is true iff `a` is a square modulo `p` (0 included).
    square_p: Vec<bool>,
}

fn check_prime(p: u64) -> Result<()> {
    if p >= SMALL_FIELD_P_BOUND {
        return Err(MathError::InvalidArgument(format!(
            "SmallFq: p = {p} is not below 2^16"
        )));
    }
    if !is_prime_u64(p) {
        return Err(MathError::InvalidArgument(format!(
            "SmallFq: p = {p} is not prime"
        )));
    }
    Ok(())
}

fn checked_order(p: u64, k: usize) -> Result<u64> {
    if k == 0 || k > SMALL_FIELD_MAX_DEGREE {
        return Err(MathError::InvalidArgument(format!(
            "SmallFq: degree {k} is not in 1..=8"
        )));
    }
    let mut q: u64 = 1;
    for _ in 0..k {
        q = q
            .checked_mul(p)
            .filter(|&v| v < (1u64 << 63))
            .ok_or_else(|| {
                MathError::InvalidArgument(format!("SmallFq: {p}^{k} is not below 2^63"))
            })?;
    }
    Ok(q)
}

/// The least monic irreducible of degree `k` over `F_p` in base-`p` counter order on
/// `(m_0, ..., m_(k-1))` (`m_0` the least significant digit). `p` must be a proved prime.
fn least_irreducible(p: u64, k: usize) -> Vec<u64> {
    if k == 1 {
        return vec![0, 1];
    }
    let pi = p as i64;
    let mut code: u64 = 0;
    loop {
        let mut c = vec![0i64; k + 1];
        let mut n = code;
        for ci in c.iter_mut().take(k) {
            *ci = (n % p) as i64;
            n /= p;
        }
        c[k] = 1;
        code += 1;
        if c[0] == 0 {
            continue;
        }
        let f = fp_factor::factor_with_multiplicity(&c, pi);
        if f.len() == 1 && f[0].1 == 1 && fp_factor::degree(&f[0].0) == k as i64 {
            return c.into_iter().map(|x| x as u64).collect();
        }
        // An irreducible of degree k exists (Gauss), so the loop ends below p^k.
        debug_assert!(code < p.saturating_pow(k as u32));
    }
}

impl<const K: usize> SmallFq<K> {
    /// `GF(p^K)` with the Conway polynomial when [`crate::conway`] tabulates `(p, K)`, otherwise
    /// the least monic irreducible of degree `K` in base-`p` counter order.
    ///
    /// Errors: `p` not a prime below `2^16`, `K` not in `1..=8`, `p^K >= 2^63`.
    pub fn new(p: u64) -> Result<Self> {
        check_prime(p)?;
        checked_order(p, K)?;
        let modulus = match crate::conway::conway_polynomial(p as u32, K) {
            Some(cp) if K >= 2 => {
                let mut c: Vec<u64> = cp
                    .coefficients()
                    .iter()
                    .map(|x| {
                        let v = x.to_i64().expect("Conway coefficient fits i64");
                        v.rem_euclid(p as i64) as u64
                    })
                    .collect();
                c.resize(K + 1, 0);
                c
            }
            _ => least_irreducible(p, K),
        };
        Self::with_modulus(p, &modulus)
    }

    /// `GF(p^K) = F_p[x]/(m)` for a monic irreducible `m` of degree `K` (little-endian,
    /// coefficients read modulo `p`; irreducibility is checked with
    /// [`fp_factor::factor_with_multiplicity`]).
    pub fn with_modulus(p: u64, modulus: &[u64]) -> Result<Self> {
        check_prime(p)?;
        let order = checked_order(p, K)?;
        if modulus.len() != K + 1 {
            return Err(MathError::InvalidArgument(format!(
                "SmallFq: modulus has {} coefficients, expected {}",
                modulus.len(),
                K + 1
            )));
        }
        let m: Vec<u64> = modulus.iter().map(|&c| c % p).collect();
        if m[K] != 1 {
            return Err(MathError::InvalidArgument(
                "SmallFq: modulus is not monic".into(),
            ));
        }
        if K >= 2 {
            let mi: Vec<i64> = m.iter().map(|&c| c as i64).collect();
            let f = fp_factor::factor_with_multiplicity(&mi, p as i64);
            if !(f.len() == 1 && f[0].1 == 1 && fp_factor::degree(&f[0].0) == K as i64) {
                return Err(MathError::InvalidArgument(
                    "SmallFq: modulus is not irreducible over F_p".into(),
                ));
            }
        }
        let mut negm = [0u64; K];
        for j in 0..K {
            negm[j] = (p - m[j]) % p;
        }
        let mut inv_p = vec![0u32; p as usize];
        for a in 1..p {
            inv_p[a as usize] = fp_factor::mod_inv(a as i64, p as i64).expect("p is prime") as u32;
        }
        let mut square_p = vec![false; p as usize];
        for a in 0..p {
            square_p[((a * a) % p) as usize] = true;
        }
        let mut f = SmallFq {
            p,
            bar: u64::MAX / p,
            order,
            modulus: m,
            negm,
            frob: [[0u64; K]; K],
            inv_p,
            square_p,
        };
        if K >= 2 {
            let mut x = [0u64; K];
            x[1] = 1;
            let xp = f.pow(&x, p);
            let mut cur = f.one();
            for i in 0..K {
                f.frob[i] = cur;
                cur = f.mul(&cur, &xp);
            }
        } else {
            f.frob[0][0] = 1;
        }
        Ok(f)
    }

    /// The characteristic `p`.
    pub fn characteristic(&self) -> u64 {
        self.p
    }

    /// The order `q = p^K`.
    pub fn order(&self) -> u64 {
        self.order
    }

    /// The degree `K` over `F_p`.
    pub fn degree(&self) -> usize {
        K
    }

    /// The defining polynomial, monic, little-endian.
    pub fn modulus(&self) -> &[u64] {
        &self.modulus
    }

    /// `x mod p` for `x < 2^64` (Barrett reduction).
    #[inline(always)]
    pub fn reduce(&self, x: u64) -> u64 {
        let q = ((x as u128 * self.bar as u128) >> 64) as u64;
        let mut r = x - q * self.p;
        while r >= self.p {
            r -= self.p;
        }
        r
    }

    /// The zero element.
    #[inline(always)]
    pub fn zero(&self) -> SmallFqElt<K> {
        [0u64; K]
    }

    /// The one element.
    #[inline(always)]
    pub fn one(&self) -> SmallFqElt<K> {
        let mut a = [0u64; K];
        a[0] = 1;
        a
    }

    /// The image of the integer `c` (read modulo `p`) in the prime field.
    #[inline(always)]
    pub fn from_fp(&self, c: u64) -> SmallFqElt<K> {
        let mut a = [0u64; K];
        a[0] = c % self.p;
        a
    }

    /// True for the zero element.
    #[inline(always)]
    pub fn is_zero(&self, a: &SmallFqElt<K>) -> bool {
        a.iter().all(|&c| c == 0)
    }

    /// `a + b`.
    #[inline(always)]
    pub fn add(&self, a: &SmallFqElt<K>, b: &SmallFqElt<K>) -> SmallFqElt<K> {
        let mut r = [0u64; K];
        for i in 0..K {
            let s = a[i] + b[i];
            r[i] = if s >= self.p { s - self.p } else { s };
        }
        r
    }

    /// `a - b`.
    #[inline(always)]
    pub fn sub(&self, a: &SmallFqElt<K>, b: &SmallFqElt<K>) -> SmallFqElt<K> {
        let mut r = [0u64; K];
        for i in 0..K {
            let s = a[i] + self.p - b[i];
            r[i] = if s >= self.p { s - self.p } else { s };
        }
        r
    }

    /// `-a`.
    #[inline(always)]
    pub fn neg(&self, a: &SmallFqElt<K>) -> SmallFqElt<K> {
        let mut r = [0u64; K];
        for i in 0..K {
            r[i] = if a[i] == 0 { 0 } else { self.p - a[i] };
        }
        r
    }

    /// `c a` for `c` in `[0, p)`.
    #[inline(always)]
    pub fn scale(&self, a: &SmallFqElt<K>, c: u64) -> SmallFqElt<K> {
        let mut r = [0u64; K];
        for i in 0..K {
            r[i] = self.reduce(a[i] * c);
        }
        r
    }

    /// Reduce a raw product (coefficients `t[0..2K-1]`, unreduced) modulo `m` and `p`.
    #[inline(always)]
    pub(crate) fn reduce_raw(&self, t: &mut [u64; 16]) -> SmallFqElt<K> {
        if K >= 2 {
            for i in (K..2 * K - 1).rev() {
                let c = self.reduce(t[i]);
                if c != 0 {
                    for j in 0..K {
                        t[i - K + j] += c * self.negm[j];
                    }
                }
            }
        }
        let mut r = [0u64; K];
        for j in 0..K {
            r[j] = self.reduce(t[j]);
        }
        r
    }

    /// `a b`.
    #[inline(always)]
    pub fn mul(&self, a: &SmallFqElt<K>, b: &SmallFqElt<K>) -> SmallFqElt<K> {
        let mut t = [0u64; 16];
        for i in 0..K {
            if a[i] == 0 {
                continue;
            }
            for j in 0..K {
                t[i + j] += a[i] * b[j];
            }
        }
        self.reduce_raw(&mut t)
    }

    /// `a^e`.
    pub fn pow(&self, a: &SmallFqElt<K>, mut e: u64) -> SmallFqElt<K> {
        let mut r = self.one();
        let mut b = *a;
        while e > 0 {
            if e & 1 == 1 {
                r = self.mul(&r, &b);
            }
            b = self.mul(&b, &b);
            e >>= 1;
        }
        r
    }

    /// The Frobenius `a^p` (MAGMA `Frobenius(a)`).
    #[inline(always)]
    pub fn frobenius(&self, a: &SmallFqElt<K>) -> SmallFqElt<K> {
        if K == 1 {
            return *a;
        }
        let mut t = [0u64; K];
        for i in 0..K {
            if a[i] == 0 {
                continue;
            }
            for j in 0..K {
                t[j] += a[i] * self.frob[i][j];
            }
        }
        let mut r = [0u64; K];
        for j in 0..K {
            r[j] = self.reduce(t[j]);
        }
        r
    }

    /// `(N(a), a^p a^(p^2) ... a^(p^(K-1)))`.
    #[inline(always)]
    fn norm_cofactor(&self, a: &SmallFqElt<K>) -> (u64, SmallFqElt<K>) {
        if K == 1 {
            return (a[0], self.one());
        }
        let mut c = self.frobenius(a);
        let mut prod = c;
        for _ in 2..K {
            c = self.frobenius(&c);
            prod = self.mul(&prod, &c);
        }
        let n = self.mul(a, &prod);
        debug_assert!(n[1..].iter().all(|&x| x == 0), "the norm lies in F_p");
        (n[0], prod)
    }

    /// The norm `N_{GF(q)/F_p}(a)` in `[0, p)` (MAGMA `Norm(a)`).
    pub fn norm(&self, a: &SmallFqElt<K>) -> u64 {
        self.norm_cofactor(a).0
    }

    /// `a^(-1)`, or `None` for `a = 0`.
    #[inline(always)]
    pub fn inverse(&self, a: &SmallFqElt<K>) -> Option<SmallFqElt<K>> {
        let (n, cof) = self.norm_cofactor(a);
        if n == 0 {
            return None;
        }
        Some(self.scale(&cof, self.inv_p[n as usize] as u64))
    }

    /// The inverse of a nonzero element (no check in release builds).
    #[inline(always)]
    fn inv_nonzero(&self, a: &SmallFqElt<K>) -> SmallFqElt<K> {
        let (n, cof) = self.norm_cofactor(a);
        debug_assert!(n != 0);
        self.scale(&cof, self.inv_p[n as usize] as u64)
    }

    /// True iff `a` is a square in `GF(q)` (0 is a square; MAGMA `IsSquare(a)`).
    pub fn is_square(&self, a: &SmallFqElt<K>) -> bool {
        self.square_p[self.norm(a) as usize]
    }

    /// The code `sum a_i p^i` of `a`, in `[0, q)`.
    #[inline(always)]
    pub fn code(&self, a: &SmallFqElt<K>) -> u64 {
        let mut c = 0u64;
        for i in (0..K).rev() {
            c = c * self.p + a[i];
        }
        c
    }

    /// The element with code `n mod q` (base-`p` digits, least significant first).
    #[inline(always)]
    pub fn from_code(&self, n: u64) -> SmallFqElt<K> {
        let mut n = n % self.order;
        let mut r = [0u64; K];
        for ri in r.iter_mut() {
            *ri = n % self.p;
            n /= self.p;
        }
        r
    }
}

// ------------------------------------------------------------------------------------------
// Distinct-root counts.
// ------------------------------------------------------------------------------------------

/// Arithmetic in `GF(q)[x]/(f)` for a monic `f` of degree `N` in `2..=4`; `x^N = sum negf[j] x^j`.
struct ModF<'a, const K: usize, const N: usize> {
    fq: &'a SmallFq<K>,
    negf: [SmallFqElt<K>; N],
}

impl<'a, const K: usize, const N: usize> ModF<'a, K, N> {
    fn new(fq: &'a SmallFq<K>, f_low: &[SmallFqElt<K>]) -> Self {
        let mut negf = [[0u64; K]; N];
        for j in 0..N {
            negf[j] = fq.neg(&f_low[j]);
        }
        ModF { fq, negf }
    }

    #[inline(always)]
    fn conv_add(acc: &mut [u64; 16], a: &SmallFqElt<K>, b: &SmallFqElt<K>) {
        for u in 0..K {
            if a[u] == 0 {
                continue;
            }
            for w in 0..K {
                acc[u + w] += a[u] * b[w];
            }
        }
    }

    #[inline(always)]
    fn mulmod(&self, a: &[SmallFqElt<K>; N], b: &[SmallFqElt<K>; N]) -> [SmallFqElt<K>; N] {
        let mut raw = [[0u64; 16]; 8];
        for i in 0..N {
            for j in 0..N {
                Self::conv_add(&mut raw[i + j], &a[i], &b[j]);
            }
        }
        for m in (N..2 * N - 1).rev() {
            let c = self.fq.reduce_raw(&mut raw[m]);
            if self.fq.is_zero(&c) {
                continue;
            }
            for j in 0..N {
                Self::conv_add(&mut raw[m - N + j], &c, &self.negf[j]);
            }
        }
        let mut r = [[0u64; K]; N];
        for j in 0..N {
            r[j] = self.fq.reduce_raw(&mut raw[j]);
        }
        r
    }

    /// `x a mod f`.
    #[inline(always)]
    fn mulx(&self, a: &[SmallFqElt<K>; N]) -> [SmallFqElt<K>; N] {
        let top = a[N - 1];
        let mut r = [[0u64; K]; N];
        for j in (1..N).rev() {
            r[j] = a[j - 1];
        }
        if !self.fq.is_zero(&top) {
            for j in 0..N {
                let t = self.fq.mul(&top, &self.negf[j]);
                r[j] = self.fq.add(&r[j], &t);
            }
        }
        r
    }

    /// `x^e mod f` (left-to-right square-and-multiply).
    fn xpow(&self, e: u64) -> [SmallFqElt<K>; N] {
        let mut r = [[0u64; K]; N];
        r[0] = self.fq.one();
        let bits = 64 - e.leading_zeros();
        for b in (0..bits).rev() {
            r = self.mulmod(&r, &r);
            if (e >> b) & 1 == 1 {
                r = self.mulx(&r);
            }
        }
        r
    }

    /// `x^q mod f` from `h1 = x^p mod f` and `K - 1` applications of the semilinear p-power map.
    fn x_q(&self) -> [SmallFqElt<K>; N] {
        let h1 = self.xpow(self.fq.p);
        if K == 1 {
            return h1;
        }
        let mut pw = [[[0u64; K]; N]; N];
        pw[0][0] = self.fq.one();
        pw[1] = h1;
        for j in 2..N {
            pw[j] = self.mulmod(&pw[j - 1], &h1);
        }
        let mut h = h1;
        for _ in 1..K {
            let mut raw = [[0u64; 16]; 8];
            for j in 0..N {
                let s = self.fq.frobenius(&h[j]);
                if self.fq.is_zero(&s) {
                    continue;
                }
                for l in 0..N {
                    Self::conv_add(&mut raw[l], &s, &pw[j][l]);
                }
            }
            let mut r = [[0u64; K]; N];
            for l in 0..N {
                r[l] = self.fq.reduce_raw(&mut raw[l]);
            }
            h = r;
        }
        h
    }
}

/// Degree of `gcd(a, b)` for polynomials over `GF(q)`; `-1` when both are zero.
fn gcd_degree<const K: usize>(
    fq: &SmallFq<K>,
    a0: &[SmallFqElt<K>],
    b0: &[SmallFqElt<K>],
) -> isize {
    let mut a: Vec<SmallFqElt<K>> = a0.to_vec();
    let mut b: Vec<SmallFqElt<K>> = b0.to_vec();
    let deg = |v: &Vec<SmallFqElt<K>>| -> isize {
        let mut d = v.len() as isize - 1;
        while d >= 0 && fq.is_zero(&v[d as usize]) {
            d -= 1;
        }
        d
    };
    let mut da = deg(&a);
    let mut db = deg(&b);
    loop {
        if db < 0 {
            return da;
        }
        let inv = fq.inv_nonzero(&b[db as usize]);
        while da >= db {
            let c = fq.mul(&a[da as usize], &inv);
            let s = (da - db) as usize;
            for j in 0..=(db as usize) {
                let t = fq.mul(&c, &b[j]);
                a[s + j] = fq.sub(&a[s + j], &t);
            }
            da = deg(&a);
        }
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut da, &mut db);
    }
}

/// Distinct roots of the monic `x^N + sum f_low[j] x^j`, `N` in `2..=4`.
#[inline(always)]
fn count_monic<const K: usize, const N: usize>(fq: &SmallFq<K>, f_low: &[SmallFqElt<K>]) -> usize {
    let m = ModF::<K, N>::new(fq, f_low);
    let h = m.x_q();
    let mut hx: Vec<SmallFqElt<K>> = h.to_vec();
    hx[1] = fq.sub(&hx[1], &fq.one());
    let mut f: Vec<SmallFqElt<K>> = f_low[..N].to_vec();
    f.push(fq.one());
    // x^q - x = 0 mod f means f | x^q - x: then gcd = f and gcd_degree returns N.
    gcd_degree(fq, &f, &hx).max(0) as usize
}

/// Degree of `c` after trimming; `-1` for the zero polynomial.
#[inline(always)]
fn trimmed_degree<const K: usize>(fq: &SmallFq<K>, c: &[SmallFqElt<K>]) -> isize {
    let mut d = c.len() as isize - 1;
    while d >= 0 && fq.is_zero(&c[d as usize]) {
        d -= 1;
    }
    d
}

/// The number of distinct roots in `GF(q)` of the polynomial `f = sum f[i] x^i` (little-endian;
/// trailing zeros allowed), or `None` for the zero polynomial (every element is a root).
///
/// Degrees `<= 4` use fixed-size arithmetic modulo `f` and the semilinear Frobenius; larger
/// degrees use [`count_distinct_roots_any_degree`]. Exact: the result is `deg gcd(f, x^q - x)`.
///
/// # Panics
///
/// Panics if the degree of `f` (after trimming) is above 4096. Degrees above 4 take the vector
/// path [`count_distinct_roots_any_degree`], which has this bound.
#[inline]
pub fn count_distinct_roots<const K: usize>(fq: &SmallFq<K>, f: &[SmallFqElt<K>]) -> Option<usize> {
    let d = trimmed_degree(fq, f);
    if d < 0 {
        return None;
    }
    if d > 4 {
        return count_distinct_roots_any_degree(fq, f);
    }
    if d == 0 {
        return Some(0);
    }
    if d == 1 {
        return Some(1);
    }
    let inv = fq.inv_nonzero(&f[d as usize]);
    let mut low = [[0u64; K]; 4];
    for j in 0..(d as usize) {
        low[j] = fq.mul(&f[j], &inv);
    }
    Some(match d {
        2 => count_monic::<K, 2>(fq, &low),
        3 => count_monic::<K, 3>(fq, &low),
        _ => count_monic::<K, 4>(fq, &low),
    })
}

/// `a b mod f` for vectors; `negf` has `n = deg f` entries (`x^n = sum negf[j] x^j`).
fn vmulmod<const K: usize>(
    fq: &SmallFq<K>,
    a: &[SmallFqElt<K>],
    b: &[SmallFqElt<K>],
    negf: &[SmallFqElt<K>],
) -> Vec<SmallFqElt<K>> {
    let n = negf.len();
    let top = a.len() + b.len() - 1;
    let mut raw = vec![[0u64; 16]; top.max(n)];
    for (i, ai) in a.iter().enumerate() {
        if fq.is_zero(ai) {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            for u in 0..K {
                if ai[u] == 0 {
                    continue;
                }
                for w in 0..K {
                    raw[i + j][u + w] += ai[u] * bj[w];
                }
            }
        }
    }
    for m in (n..top).rev() {
        let c = fq.reduce_raw(&mut raw[m]);
        if fq.is_zero(&c) {
            continue;
        }
        for j in 0..n {
            for u in 0..K {
                if c[u] == 0 {
                    continue;
                }
                for w in 0..K {
                    raw[m - n + j][u + w] += c[u] * negf[j][w];
                }
            }
        }
    }
    let mut r = vec![[0u64; K]; n];
    for j in 0..n {
        r[j] = fq.reduce_raw(&mut raw[j]);
    }
    r
}

/// [`count_distinct_roots`] on vectors, for any degree (the path used above degree 4; public so
/// that both paths can be tested against each other). Degree `<= 2^12` (accumulator bound).
///
/// # Panics
///
/// Panics if the degree of `f` (after trimming) is above 4096. Above it, the bound `2^48` on
/// the lazy `u64` accumulators of the products modulo `f` (module doc) no longer holds.
pub fn count_distinct_roots_any_degree<const K: usize>(
    fq: &SmallFq<K>,
    f: &[SmallFqElt<K>],
) -> Option<usize> {
    let d = trimmed_degree(fq, f);
    if d < 0 {
        return None;
    }
    let d = d as usize;
    assert!(
        d <= 4096,
        "count_distinct_roots_any_degree: degree {d} above 4096"
    );
    if d == 0 {
        return Some(0);
    }
    if d == 1 {
        return Some(1);
    }
    let inv = fq.inv_nonzero(&f[d]);
    let low: Vec<SmallFqElt<K>> = (0..d).map(|j| fq.mul(&f[j], &inv)).collect();
    let negf: Vec<SmallFqElt<K>> = low.iter().map(|x| fq.neg(x)).collect();
    let q = fq.order;
    let mut r = vec![[0u64; K]; d];
    r[0] = fq.one();
    let mut x = vec![[0u64; K]; d];
    x[1] = fq.one();
    let bits = 64 - q.leading_zeros();
    for b in (0..bits).rev() {
        r = vmulmod(fq, &r, &r, &negf);
        if (q >> b) & 1 == 1 {
            r = vmulmod(fq, &r, &x, &negf);
        }
    }
    r[1] = fq.sub(&r[1], &fq.one());
    let mut fm = low;
    fm.push(fq.one());
    Some(gcd_degree(fq, &fm, &r).max(0) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals() {
        assert!(SmallFq::<1>::new(1).is_err());
        assert!(SmallFq::<2>::new(9).is_err());
        assert!(SmallFq::<1>::new(65537).is_err()); // prime, but not below 2^16
        assert!(SmallFq::<4>::new(65521).is_err()); // 65521^4 > 2^63
        assert!(SmallFq::<3>::new(65521).is_ok());
        assert!(SmallFq::<9>::new(3).is_err());
        // x^2 + 1 = (x + 2)(x + 3) over F_5: reducible.
        assert!(SmallFq::<2>::with_modulus(5, &[1, 0, 1]).is_err());
        // not monic
        assert!(SmallFq::<2>::with_modulus(5, &[2, 0, 2]).is_err());
        // x^2 + 2 is irreducible over F_5 (2 is a nonsquare mod 5)
        assert!(SmallFq::<2>::with_modulus(5, &[2, 0, 1]).is_ok());
    }

    #[test]
    fn small_field_order_and_frobenius_order() {
        let fq = SmallFq::<3>::new(7).unwrap();
        assert_eq!(fq.order(), 343);
        // Frobenius has order 3: a^(p^3) = a for every a.
        for n in 0..343 {
            let a = fq.from_code(n);
            let b = fq.frobenius(&fq.frobenius(&fq.frobenius(&a)));
            assert_eq!(a, b);
            assert_eq!(fq.code(&a), n);
            assert_eq!(fq.frobenius(&a), fq.pow(&a, 7));
        }
    }

    #[test]
    fn roots_of_x_q_minus_x_factors() {
        // x^2 - x over GF(5^2): 2 roots; (x - 1)^3 (x - 2): 2 distinct roots.
        let fq = SmallFq::<2>::new(5).unwrap();
        let f = vec![fq.zero(), fq.from_fp(4), fq.one()];
        assert_eq!(count_distinct_roots(&fq, &f), Some(2));
        // (x-1)^3 (x-2) = x^4 - 5x^3 + 9x^2 - 7x + 2 = x^4 + 4x^2 + 3x + 2 mod 5
        let g = vec![
            fq.from_fp(2),
            fq.from_fp(3),
            fq.from_fp(4),
            fq.zero(),
            fq.one(),
        ];
        assert_eq!(count_distinct_roots(&fq, &g), Some(2));
        assert_eq!(count_distinct_roots_any_degree(&fq, &g), Some(2));
        assert_eq!(count_distinct_roots(&fq, &[fq.zero(), fq.zero()]), None);
        assert_eq!(count_distinct_roots(&fq, &[fq.from_fp(3)]), Some(0));
    }
}
