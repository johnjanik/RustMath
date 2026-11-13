//! Berlekamp factorization over GF(p), p any prime.
//!
//! Specification: MAGMA Handbook ch. 23, *Univariate Polynomial Rings* (digest
//! `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md`), §23.8.1 `Factorization`
//! over a finite field (parameter `Al := "BerlekampSmall"` / `"BerlekampLarge"`; the MAGMA
//! V2.3 handbook gives the default "BerlekampSmall if q < 25, BerlekampLarge otherwise", and
//! the same threshold is used here), `SquarefreeFactorization` and `QMatrix`.
//!
//! Representation, as in [`crate::zx`] and [`crate::fp_factor`]: little-endian `Vec<Integer>`
//! (index = power of x), every coefficient in `[0, p)`, trimmed, `[]` is the zero polynomial.
//! The prime is an arbitrary-size [`Integer`], so this complements `fp_factor` (i64 arithmetic,
//! p <= `fp_factor::MAX_P` = 3037000499, Cantor-Zassenhaus).
//!
//! Algorithm ([`factor`]):
//! 1. Square-free decomposition over GF(p) (Yun's gcd recursion with the p-th-root step for
//!    `f' = 0`; Knuth §4.6.2; Cohen §3.4, square-free factorization):
//!    `f = lc(f) · ∏ s_i^m_i`.
//! 2. For each square-free part s of degree n: the Berlekamp matrix Q (row k = x^(kp) mod s,
//!    computed from x^p mod s once and then by successive multiplication), and a basis of
//!    the null space {v : v(Q - I) = 0}, i.e. of the polynomials v with v^p ≡ v (mod s)
//!    (reduced row echelon form, the row form of Knuth's Algorithm N). Its dimension r is the
//!    number of irreducible factors of s (Berlekamp 1967).
//! 3. Splitting. p < 25: Berlekamp's deterministic split, every w replaced by the nontrivial
//!    members of {gcd(w, v - t) : t ∈ GF(p)} for each basis vector v (Knuth §4.6.2; Cohen
//!    §3.4, Berlekamp's algorithm for small p). p >= 25: the probabilistic split of Rabin
//!    (1980) on random elements of the Berlekamp subalgebra (Knuth §4.6.2, the remarks on
//!    large p; the splitting step of Cantor and Zassenhaus 1981): v random in the null
//!    space, gcd(w, (v mod w)^((p-1)/2) - 1).
//!    This is not Berlekamp's deterministic 1970 method for large p. Both stop exactly when r
//!    factors are found, so the count is never guessed.
//!
//! References:
//! - E. R. Berlekamp, *Factoring polynomials over finite fields*, Bell System Tech. J. 46
//!   (1967) 1853-1859.
//! - E. R. Berlekamp, *Factoring polynomials over large finite fields*, Math. Comp. 24 (1970)
//!   713-735.
//! - M. O. Rabin, *Probabilistic algorithms in finite fields*, SIAM J. Comput. 9 (1980)
//!   273-280.
//! - D. G. Cantor, H. Zassenhaus, *A new algorithm for factoring polynomials over finite
//!   fields*, Math. Comp. 36 (1981) 587-592.
//! - D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed. (1997), §4.6.2
//!   (Berlekamp's procedure, the null-space Algorithm N, square-free factorization, large p).
//! - H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138 (1993), §3.4
//!   (square-free factorization modulo p; Berlekamp's algorithm). The algorithm numbers
//!   within §3.4 are not cited because they were not checked against the book.

use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use std::cmp::Ordering;

/// The splitting is deterministic (Berlekamp) for p below this bound and probabilistic above
/// (MAGMA V2.3: "BerlekampSmall" if q < 25).
const SMALL_P: i64 = 25;

/// Rounds of the probabilistic split allowed per irreducible factor before giving up with an
/// honest `Err` (never observed above 16 in total on the reference inputs).
const ROUNDS_PER_FACTOR: usize = 64;

/// A nonzero coefficient without an inverse modulo `p`. Every entry checks first that `p` is
/// a proven prime ([`check_prime`]), so this is a bug, not a composite modulus.
fn no_inverse(p: &Integer) -> MathError {
    internal(&format!(
        "a nonzero coefficient has no inverse modulo the proven prime {p}"
    ))
}

fn internal(what: &str) -> MathError {
    MathError::InvalidOperation(format!(
        "berlekamp: internal check failed ({what}); the modulus is a proven prime, so this \
         is a bug"
    ))
}

/// `Ok(())` iff `p` is a proven prime ([`rustmath_integers::prime::try_is_prime`]: a proof
/// for every size). `p < 2` or `p` composite: `Err(InvalidArgument("berlekamp: the modulus
/// {p} is not a prime"))`. A probable prime whose ECPP proof is unresolved:
/// `Err(NotSupported("berlekamp: the primality of the modulus {p} is unresolved (..)"))`.
fn check_prime(p: &Integer) -> Result<()> {
    if *p < Integer::from(2i64) {
        return Err(MathError::InvalidArgument(format!(
            "berlekamp: the modulus {p} is not a prime"
        )));
    }
    match rustmath_integers::prime::try_is_prime(p) {
        Ok(true) => Ok(()),
        Ok(false) => Err(MathError::InvalidArgument(format!(
            "berlekamp: the modulus {p} is not a prime"
        ))),
        Err(e) => Err(MathError::NotSupported(format!(
            "berlekamp: the primality of the modulus {p} is unresolved ({e})"
        ))),
    }
}

fn trim(mut v: Vec<Integer>) -> Vec<Integer> {
    while v.last().is_some_and(|c| c.is_zero()) {
        v.pop();
    }
    v
}

/// `f` with every coefficient reduced to `[0, p)`, trimmed.
fn reduce(f: &[Integer], p: &Integer) -> Vec<Integer> {
    trim(f.iter().map(|c| c.modulo(p)).collect())
}

fn sub(a: &[Integer], b: &[Integer], p: &Integer) -> Vec<Integer> {
    let n = a.len().max(b.len());
    let zero = Integer::zero();
    trim(
        (0..n)
            .map(|i| (a.get(i).unwrap_or(&zero) - b.get(i).unwrap_or(&zero)).modulo(p))
            .collect(),
    )
}

fn mul(a: &[Integer], b: &[Integer], p: &Integer) -> Vec<Integer> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            out[i + j] = &out[i + j] + &(x * y);
        }
    }
    trim(out.into_iter().map(|c| c.modulo(p)).collect())
}

/// `(q, r)` with `a = q·b + r`, `deg r < deg b`. `b` is nonzero; its leading coefficient must
/// be invertible modulo p (always for p prime).
fn divrem(a: &[Integer], b: &[Integer], p: &Integer) -> Result<(Vec<Integer>, Vec<Integer>)> {
    let lb = b.last().ok_or(MathError::DivisionByZero)?;
    if a.len() < b.len() {
        return Ok((Vec::new(), a.to_vec()));
    }
    let inv = lb.mod_inverse(p).ok_or_else(|| no_inverse(p))?;
    let mut q = vec![Integer::zero(); a.len() - b.len() + 1];
    let mut r = a.to_vec();
    while r.len() >= b.len() {
        let s = r.len() - b.len();
        let c = (r.last().expect("nonempty") * &inv).modulo(p);
        for (j, y) in b.iter().enumerate() {
            r[s + j] = (&r[s + j] - &(&c * y)).modulo(p);
        }
        // c·lc(b) ≡ lc(r), so the top coefficient is now 0.
        debug_assert!(r.last().is_some_and(|x| x.is_zero()));
        r.pop();
        q[s] = c;
        r = trim(r);
    }
    Ok((trim(q), r))
}

fn rem(a: &[Integer], b: &[Integer], p: &Integer) -> Result<Vec<Integer>> {
    Ok(divrem(a, b, p)?.1)
}

/// `a / lc(a)`; `[]` stays `[]`.
fn monic(a: &[Integer], p: &Integer) -> Result<Vec<Integer>> {
    let Some(l) = a.last() else {
        return Ok(Vec::new());
    };
    if l.is_one() {
        return Ok(a.to_vec());
    }
    let inv = l.mod_inverse(p).ok_or_else(|| no_inverse(p))?;
    Ok(trim(a.iter().map(|c| (c * &inv).modulo(p)).collect()))
}

/// Monic gcd (Euclid); `gcd(0, 0) = 0`.
fn gcd(a: &[Integer], b: &[Integer], p: &Integer) -> Result<Vec<Integer>> {
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    while !b.is_empty() {
        let r = rem(&a, &b, p)?;
        a = std::mem::replace(&mut b, r);
    }
    monic(&a, p)
}

fn derivative(a: &[Integer], p: &Integer) -> Vec<Integer> {
    trim(
        a.iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| (c * &Integer::from(i as u64)).modulo(p))
            .collect(),
    )
}

fn mulmod(a: &[Integer], b: &[Integer], m: &[Integer], p: &Integer) -> Result<Vec<Integer>> {
    rem(&mul(a, b, p), m, p)
}

/// `a^e mod m` (square and multiply, Knuth §4.6.3); `deg m >= 1`.
fn powmod(a: &[Integer], e: &Integer, m: &[Integer], p: &Integer) -> Result<Vec<Integer>> {
    let two = Integer::from(2i64);
    let mut base = rem(a, m, p)?;
    let mut acc = vec![Integer::one()];
    let mut e = e.clone();
    while e.signum() > 0 {
        if e.is_odd() {
            acc = mulmod(&acc, &base, m, p)?;
        }
        e = &e / &two;
        if e.signum() > 0 {
            base = mulmod(&base, &base, m, p)?;
        }
    }
    Ok(acc)
}

/// Order of the factors: by degree, then by coefficients from the top down (the order PARI's
/// `factormod` prints on every reference input; MAGMA's order is not documented).
fn factor_order(a: &[Integer], b: &[Integer]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

/// Deterministic 64-bit LCG (Knuth MMIX constants), the source of the probabilistic split.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// An element of `[0, m)` from enough 31-bit draws to exceed `m` by 31 bits.
    fn below(&mut self, m: &Integer) -> Integer {
        let chunks = m.bit_length().div_ceil(31) + 1;
        let base = Integer::from(1u64 << 31);
        let mut x = Integer::zero();
        for _ in 0..chunks {
            x = &(&x * &base) + &Integer::from(self.next());
        }
        x.modulo(m)
    }
}

/// A list of `(factor, multiplicity)` pairs, factors as little-endian coefficient vectors.
pub type FactorList = Vec<(Vec<Integer>, u32)>;

/// Validate `p`, reduce `f` modulo p, reject 0.
fn prepare(f: &[Integer], p: &Integer, who: &str) -> Result<Vec<Integer>> {
    check_prime(p)?;
    let f = reduce(f, p);
    if f.is_empty() {
        return Err(MathError::InvalidArgument(format!(
            "berlekamp::{who}: f ≡ 0 (mod {p}) has no factorization"
        )));
    }
    Ok(f)
}

/// MAGMA `Factorization(f)` over GF(p) (Handbook ch. 23 §23.8.1, `Al := "BerlekampSmall"` for
/// p < 25, `"BerlekampLarge"` otherwise): `(u, [(g_1, e_1), ..., (g_k, e_k)])` with
/// `f ≡ u · ∏ g_i^e_i (mod p)`, `u = lc(f) mod p` (MAGMA: "over a field [the unit] will be
/// such that ∏ q_i^k_i is monic"), every `g_i` monic irreducible over GF(p), pairwise
/// distinct, sorted by degree and then by coefficients from the top down (the order of PARI
/// `factormod` on all reference inputs; a RustMath convention, MAGMA's order is not
/// documented).
///
/// * `f` is read modulo p (any integer coefficients; a leading coefficient divisible by p
///   drops out).
/// * `p < 2` or p not prime: `Err(InvalidArgument)`. Primality is
///   [`rustmath_integers::prime::try_is_prime`], a proof for every size (Baillie-PSW and ECPP
///   above 2^64, cached per process); a probable prime whose ECPP proof is unresolved gives
///   `Err(NotSupported)`.
/// * `f ≡ 0 (mod p)`: `Err(InvalidArgument)` (no factorization of 0; PARI returns `[0, 1]`).
/// * `f ≡ c ≠ 0`: `Ok((c, vec![]))`.
/// * Otherwise, for prime `p`, the complete factorization. The number of irreducible
///   factors of each square-free part is `dim ker(Q - I)`, computed exactly, and the
///   splitting stops exactly when that many factors are found, so the result is certified
///   for prime `p`. For p >= 25 the splitting is randomized (seeded from the degree and p,
///   so reproducible); if it has not finished after `64 · r` rounds (r factors; at most 16
///   rounds were needed on the reference inputs, r up to 64) the result is
///   `Err(NotSupported)` naming the undecided factorization, never a partial answer.
pub fn factor(f: &[Integer], p: &Integer) -> Result<(Integer, FactorList)> {
    let f = prepare(f, p, "factor")?;
    let u = f.last().expect("nonzero").clone();
    if f.len() == 1 {
        return Ok((u, Vec::new()));
    }
    let f = monic(&f, p)?;
    let mut out = Vec::new();
    for (s, m) in squarefree_parts(f, p)? {
        for g in factor_squarefree(&s, p)? {
            out.push((g, m));
        }
    }
    out.sort_by(|a, b| factor_order(&a.0, &b.0));
    Ok((u, out))
}

/// MAGMA `SquarefreeFactorization(f)` over GF(p) (Handbook ch. 23 §23.8.1; Knuth §4.6.2;
/// Cohen §3.4, square-free factorization): `[(s_1, m_1), ...]` with every `s_i` monic,
/// square-free, of positive degree, pairwise coprime, the `m_i` distinct and increasing, and
/// `f ≡ lc(f) · ∏ s_i^m_i (mod p)`. Multiplicities divisible by p are found through the
/// p-th root of `f` when `f' = 0` (over GF(p), `g(x^p) = g(x)^p`). Same errors as
/// [`factor`]; `Ok(vec![])` for a nonzero constant.
pub fn squarefree_decomposition(f: &[Integer], p: &Integer) -> Result<FactorList> {
    let f = prepare(f, p, "squarefree_decomposition")?;
    if f.len() == 1 {
        return Ok(Vec::new());
    }
    let f = monic(&f, p)?;
    squarefree_parts(f, p)
}

/// The Berlekamp matrix of `f` (degree n >= 1) over GF(p): the n × n matrix whose row k holds
/// the coefficients of x^(kp) mod f, k = 0..n-1 (Knuth §4.6.2), little-endian within a row,
/// entries in `[0, p)`. `Err` as for [`factor`], and `Err(InvalidArgument)` for deg f < 1.
///
/// MAGMA's `QMatrix(f)` is documented as "an element of the degree d - 1 matrix algebra over
/// F" (V2.3 handbook; the V2.19 handbook, §23.8.1, has the same text). This module returns
/// the d × d matrix, for three reasons. On the same handbook page `CompanionMatrix(f)` is
/// also "an element of the full matrix algebra of degree d - 1", while the matrix shown there
/// has d rows and the d entries `-a_0 … -a_(d-1)` in its last row, so "degree d - 1" there is
/// not the matrix size (elsewhere MAGMA's "degree n" is n × n: "Full Matrix Algebra of degree
/// 3 over GF(23)", ch. 26). MAGMA cites Knuth §4.6.2 for Berlekamp's algorithm (§23.8.1,
/// `Factorization`), and Knuth's Q is d × d with row k the coefficients of x^(kq) mod f. Which
/// convention MAGMA's `QMatrix` prints is still unverified (no MAGMA was available).
pub fn q_matrix(f: &[Integer], p: &Integer) -> Result<Vec<Vec<Integer>>> {
    let f = prepare(f, p, "q_matrix")?;
    if f.len() < 2 {
        return Err(MathError::InvalidArgument(
            "berlekamp::q_matrix: f must have degree >= 1 modulo p".to_string(),
        ));
    }
    let f = monic(&f, p)?;
    q_rows(&f, p)
}

/// MAGMA `IsIrreducible(f)` over GF(p) (Handbook ch. 23 §23.8.1; the V2.3 handbook allows
/// R = Z/pZ, p prime): whether `f mod p` is irreducible in `GF(p)[x]`. Decided without
/// splitting, by Berlekamp's criterion (Berlekamp 1967; Knuth §4.6.2): for a monic
/// square-free `f` of degree n >= 1, `dim ker(Q - I)` (the polynomials v of degree < n with
/// v^p ≡ v mod f) is the number of irreducible factors of `f`. So `f` of degree >= 2 is
/// irreducible iff it is square-free (`f' ≠ 0` and `gcd(f, f') = 1`; `f' = 0` means
/// `f = g(x^p) = g^p`) and that dimension is 1. Deterministic, no randomness.
///
/// * `f` is read modulo p, like [`factor`].
/// * `f ≡ 0` and nonzero constants: `Ok(false)` (0 and units are not irreducible; PARI
///   `polisirreducible(Mod(0,5)) = polisirreducible(Mod(2,5)) = 0`). Degree 1: `Ok(true)`.
/// * `p < 2` or p not prime: `Err(InvalidArgument)`; an unresolved primality proof:
///   `Err(NotSupported)` (as for [`factor`], `try_is_prime` is a proof for every size).
pub fn is_irreducible(f: &[Integer], p: &Integer) -> Result<bool> {
    check_prime(p)?;
    let f = reduce(f, p);
    if f.len() <= 1 {
        return Ok(false);
    }
    if f.len() == 2 {
        return Ok(true);
    }
    let f = monic(&f, p)?;
    let fp = derivative(&f, p);
    if fp.is_empty() || gcd(&f, &fp, p)?.len() > 1 {
        return Ok(false);
    }
    Ok(kernel_basis(&q_rows(&f, p)?, p)?.len() == 1)
}

/// Rows of Q for a monic `f` of degree n >= 1.
fn q_rows(f: &[Integer], p: &Integer) -> Result<Vec<Vec<Integer>>> {
    let n = f.len() - 1;
    let x = [Integer::zero(), Integer::one()];
    let xp = powmod(&x, p, f, p)?;
    let mut rows = Vec::with_capacity(n);
    let mut cur = vec![Integer::one()];
    for k in 0..n {
        let mut row = vec![Integer::zero(); n];
        for (j, c) in cur.iter().enumerate() {
            row[j] = c.clone();
        }
        rows.push(row);
        if k + 1 < n {
            cur = mulmod(&cur, &xp, f, p)?;
        }
    }
    Ok(rows)
}

/// A basis of `{v : v (Q - I) = 0}` over GF(p), one vector per free column of the reduced
/// row echelon form of `(Q - I)^T`. Column 0 is always free (row 0 of Q is e_0), so the first
/// vector is the constant polynomial 1.
fn kernel_basis(q: &[Vec<Integer>], p: &Integer) -> Result<Vec<Vec<Integer>>> {
    let n = q.len();
    let one = Integer::one();
    let mut a: Vec<Vec<Integer>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        (&q[j][i] - &one).modulo(p)
                    } else {
                        q[j][i].clone()
                    }
                })
                .collect()
        })
        .collect();
    let mut pivots: Vec<usize> = Vec::new();
    for c in 0..n {
        let r = pivots.len();
        if r == n {
            break;
        }
        let Some(pr) = (r..n).find(|&i| !a[i][c].is_zero()) else {
            continue;
        };
        a.swap(r, pr);
        // Row r is zero left of column c (eliminated pivot columns, or columns with no
        // pivot in rows >= r).
        let inv = a[r][c].mod_inverse(p).ok_or_else(|| no_inverse(p))?;
        for x in a[r][c..].iter_mut() {
            *x = (&*x * &inv).modulo(p);
        }
        let pivot_row = a[r].clone();
        for (i, row) in a.iter_mut().enumerate() {
            if i == r || row[c].is_zero() {
                continue;
            }
            let t = row[c].clone();
            for (x, y) in row[c..].iter_mut().zip(&pivot_row[c..]) {
                *x = (&*x - &(&t * y)).modulo(p);
            }
        }
        pivots.push(c);
    }
    let mut is_pivot = vec![false; n];
    for &c in &pivots {
        is_pivot[c] = true;
    }
    let mut basis = Vec::new();
    for fc in (0..n).filter(|&c| !is_pivot[c]) {
        let mut v = vec![Integer::zero(); n];
        v[fc] = Integer::one();
        for (i, &pc) in pivots.iter().enumerate() {
            v[pc] = (-&a[i][fc]).modulo(p);
        }
        basis.push(v);
    }
    Ok(basis)
}

/// Square-free decomposition of a monic `f` of positive degree (see
/// [`squarefree_decomposition`]), sorted by multiplicity.
fn squarefree_parts(f: Vec<Integer>, p: &Integer) -> Result<FactorList> {
    let mut out = Vec::new();
    squarefree_rec(f, 1, p, &mut out)?;
    out.sort_by_key(|(_, m)| *m);
    Ok(out)
}

/// Yun's recursion with the p-th-root step. For monic `f = ∏ P^e`, `c = gcd(f, f')` is
/// `∏_{p ∤ e} P^(e-1) · ∏_{p | e} P^e`; the loop peels off `∏_{p ∤ e, e = i} P` for
/// i = 1, 2, ..., and what is left of `c` is `∏_{p | e} P^e`, a p-th power, handled by the
/// recursion with the multiplier `mult · p`. An irreducible P of multiplicity p^k·e'
/// (p ∤ e') is emitted once, at depth k with i = e'.
fn squarefree_rec(f: Vec<Integer>, mult: u32, p: &Integer, out: &mut FactorList) -> Result<()> {
    if f.len() <= 1 {
        return Ok(());
    }
    let overflow = || MathError::NotSupported("berlekamp: multiplicity overflows u32".into());
    let fp = derivative(&f, p);
    if fp.is_empty() {
        // f = g(x^p) = g^p (every element of GF(p) is its own p-th root). This needs
        // p <= deg f, so p fits in usize.
        let deg = f.len() - 1;
        if *p > Integer::from(deg as u64) {
            return Err(internal("f' = 0 with deg f < p"));
        }
        let pu = usize::try_from(p).expect("berlekamp: p <= deg f (checked above) fits in usize");
        let g: Vec<Integer> = f.iter().step_by(pu).cloned().collect();
        let m = mult.checked_mul(pu as u32).ok_or_else(overflow)?;
        return squarefree_rec(g, m, p, out);
    }
    let mut c = gcd(&f, &fp, p)?;
    let mut w = divrem(&f, &c, p)?.0;
    let mut i: u32 = 1;
    while w.len() > 1 {
        let y = gcd(&w, &c, p)?;
        let z = divrem(&w, &y, p)?.0;
        if z.len() > 1 {
            out.push((z, i.checked_mul(mult).ok_or_else(overflow)?));
        }
        c = divrem(&c, &y, p)?.0;
        w = y;
        i += 1;
    }
    if c.len() > 1 {
        squarefree_rec(c, mult, p, out)?;
    }
    Ok(())
}

/// The monic irreducible factors of a monic square-free `s` of positive degree.
fn factor_squarefree(s: &[Integer], p: &Integer) -> Result<Vec<Vec<Integer>>> {
    if s.len() == 2 {
        return Ok(vec![s.to_vec()]);
    }
    let basis = kernel_basis(&q_rows(s, p)?, p)?;
    let r = basis.len();
    if r == 0 {
        return Err(internal("empty Berlekamp null space"));
    }
    if r == 1 {
        return Ok(vec![s.to_vec()]);
    }
    if *p < Integer::from(SMALL_P) {
        split_small(s, &basis, p)
    } else {
        split_random(s, &basis, p)
    }
}

/// Berlekamp's deterministic split (p < 25): for each non-constant basis vector v, every w of
/// degree >= 2 is replaced by the nontrivial members of {gcd(w, v - t) : t = 0..p-1}, which
/// are pairwise coprime with product w (w | v^p - v = ∏_t (v - t)). The full basis separates
/// every pair of irreducible factors, so this ends with exactly r factors.
fn split_small(s: &[Integer], basis: &[Vec<Integer>], p: &Integer) -> Result<Vec<Vec<Integer>>> {
    let r = basis.len();
    let p_small = i64::try_from(p).expect("split_small runs only for p < SMALL_P");
    let mut factors = vec![s.to_vec()];
    for v in basis {
        if factors.len() == r {
            break;
        }
        let v = trim(v.clone());
        if v.len() <= 1 {
            continue; // the constant vector separates nothing
        }
        let mut next = Vec::with_capacity(r);
        for w in factors {
            if w.len() <= 2 {
                next.push(w);
                continue;
            }
            let dw = w.len() - 1;
            let mut found = 0usize;
            for t in 0..p_small {
                let g = gcd(&w, &sub(&v, &[Integer::from(t)], p), p)?;
                if g.len() > 1 {
                    found += g.len() - 1;
                    next.push(g);
                }
                if found == dw {
                    break;
                }
            }
            if found != dw {
                return Err(internal("Berlekamp split lost a factor"));
            }
        }
        factors = next;
    }
    if factors.len() != r {
        return Err(internal("Berlekamp split count"));
    }
    Ok(factors)
}

/// Probabilistic split (p >= 25, odd): v = Σ c_j b_j with c_j uniform in GF(p) is a uniform
/// random element of the Berlekamp subalgebra, so v ≡ t_i (mod P_i) with independent uniform
/// t_i; gcd(w, (v mod w)^((p-1)/2) - 1) collects the P_i | w with t_i a nonzero square, and
/// splits w with probability about 1/2 for every pair of its factors (Rabin 1980).
fn split_random(s: &[Integer], basis: &[Vec<Integer>], p: &Integer) -> Result<Vec<Vec<Integer>>> {
    let r = basis.len();
    let n = s.len() - 1;
    let low32 = u64::try_from(&p.modulo(&Integer::from(1u64 << 32)))
        .expect("a residue mod 2^32 fits in u64");
    let mut rng = Lcg(0x5EED ^ (n as u64) ^ low32);
    let e = &(p - &Integer::one()) / &Integer::from(2i64);
    let one = [Integer::one()];
    let mut factors = vec![s.to_vec()];
    let mut rounds = 0usize;
    while factors.len() < r {
        if rounds == ROUNDS_PER_FACTOR * r {
            return Err(MathError::NotSupported(format!(
                "berlekamp: factorization undecided: the randomized split found {} of {r} \
                 factors in {rounds} rounds",
                factors.len()
            )));
        }
        rounds += 1;
        let mut v = vec![Integer::zero(); n];
        for b in basis {
            let c = rng.below(p);
            if c.is_zero() {
                continue;
            }
            for (vi, bi) in v.iter_mut().zip(b) {
                *vi = (&*vi + &(&c * bi)).modulo(p);
            }
        }
        let v = trim(v);
        let mut next = Vec::with_capacity(r);
        for w in factors {
            if w.len() <= 2 {
                next.push(w);
                continue;
            }
            let a = rem(&v, &w, p)?;
            let h = powmod(&a, &e, &w, p)?;
            let g = gcd(&w, &sub(&h, &one, p), p)?;
            if g.len() > 1 && g.len() < w.len() {
                let q = divrem(&w, &g, p)?.0;
                next.push(g);
                next.push(q);
            } else {
                next.push(w);
            }
        }
        factors = next;
    }
    Ok(factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(c: &[i64]) -> Vec<Integer> {
        c.iter().map(|&x| Integer::from(x)).collect()
    }

    fn m(rows: &[&[i64]]) -> Vec<Vec<Integer>> {
        rows.iter().map(|r| v(r)).collect()
    }

    fn int(x: i64) -> Integer {
        Integer::from(x)
    }

    // gp: Qm(f,p) = matrix(n,n,i,j, lift(polcoef(lift(Mod(x, f*Mod(1,p))^((i-1)*p)), j-1)))
    //   Qm(x^2+2,3)   -> [1, 0; 0, 1]
    //   Qm(x^3+1,2)   -> [1, 0, 0; 0, 0, 1; 0, 1, 0]
    //   Qm(x^4+1,3)   -> [1, 0, 0, 0; 0, 0, 0, 1; 0, 0, 2, 0; 0, 1, 0, 0]
    //   Qm(x^5+x+3,7) -> [1, 0, 0, 0, 0; 0, 0, 4, 6, 0; 3, 5, 6, 0, 2; 4, 3, 3, 5, 5;
    //                     5, 1, 2, 6, 2]
    // (the old code built row i as x^(i(p+1)) mod f: [[1,0],[1,0]] for x^2+2 over GF(3))
    #[test]
    fn q_matrix_matches_pari() {
        assert_eq!(
            q_matrix(&v(&[2, 0, 1]), &int(3)).unwrap(),
            m(&[&[1, 0], &[0, 1]])
        );
        assert_eq!(
            q_matrix(&v(&[1, 0, 0, 1]), &int(2)).unwrap(),
            m(&[&[1, 0, 0], &[0, 0, 1], &[0, 1, 0]])
        );
        assert_eq!(
            q_matrix(&v(&[1, 0, 0, 0, 1]), &int(3)).unwrap(),
            m(&[&[1, 0, 0, 0], &[0, 0, 0, 1], &[0, 0, 2, 0], &[0, 1, 0, 0]])
        );
        assert_eq!(
            q_matrix(&v(&[3, 1, 0, 0, 0, 1]), &int(7)).unwrap(),
            m(&[
                &[1, 0, 0, 0, 0],
                &[0, 0, 4, 6, 0],
                &[3, 5, 6, 0, 2],
                &[4, 3, 3, 5, 5],
                &[5, 1, 2, 6, 2]
            ])
        );
        // Read modulo p and made monic first: 2x^2 + 4 ≡ 2(x^2 + 2) over GF(3).
        assert_eq!(
            q_matrix(&v(&[4, 0, 2]), &int(3)).unwrap(),
            m(&[&[1, 0], &[0, 1]])
        );
    }

    // gp: #matker(Mod(Qm(f,p),p)~ - 1) -> 2, 2, 2, 1, and 9 for f18 mod 3
    // (f18 = x^18+x^17+2*x^16+2*x^15+2*x^11+2*x^10+x^9+x^8+x^7+x^3+x).
    #[test]
    fn kernel_dimension_matches_pari() {
        let dim = |f: &[i64], p: i64| {
            let q = q_matrix(&v(f), &int(p)).unwrap();
            let b = kernel_basis(&q, &int(p)).unwrap();
            // Every basis vector satisfies v Q = v, and the first is the constant 1.
            assert_eq!(trim(b[0].clone()), v(&[1]));
            for w in &b {
                let n = w.len();
                for j in 0..n {
                    let s = (0..n).fold(Integer::zero(), |acc, k| &acc + &(&w[k] * &q[k][j]));
                    assert_eq!(s.modulo(&int(p)), w[j]);
                }
            }
            b.len()
        };
        assert_eq!(dim(&[2, 0, 1], 3), 2);
        assert_eq!(dim(&[1, 0, 0, 1], 2), 2);
        assert_eq!(dim(&[1, 0, 0, 0, 1], 3), 2);
        assert_eq!(dim(&[3, 1, 0, 0, 0, 1], 7), 1);
        let f18 = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 2, 2, 1, 1];
        assert_eq!(dim(&f18, 3), 9);
    }

    // gp: factormod(x^9+2*x^6+x^3+2,3) -> [x + 2, 3; x^2 + 1, 3]
    //     factormod(x^27-x^9+1,3) -> [x^3 + 2*x + 1, 9]
    //     factormod((x+1)^3*(x+2),3) -> [x + 1, 3; x + 2, 1]
    #[test]
    fn squarefree_decomposition_with_pth_roots() {
        // (x+2)(x^2+1) = x^3 + 2x^2 + x + 2
        assert_eq!(
            squarefree_decomposition(&v(&[2, 0, 0, 1, 0, 0, 2, 0, 0, 1]), &int(3)).unwrap(),
            vec![(v(&[2, 1, 2, 1]), 3)]
        );
        let mut f = vec![0i64; 28];
        f[0] = 1;
        f[9] = -1;
        f[27] = 1;
        assert_eq!(
            squarefree_decomposition(&v(&f), &int(3)).unwrap(),
            vec![(v(&[1, 2, 0, 1]), 9)]
        );
        assert_eq!(
            squarefree_decomposition(&v(&[2, 1, 0, 2, 1]), &int(3)).unwrap(),
            vec![(v(&[2, 1]), 1), (v(&[1, 1]), 3)]
        );
        assert_eq!(squarefree_decomposition(&v(&[7]), &int(5)).unwrap(), vec![]);
    }

    #[test]
    fn invalid_inputs_are_errors() {
        for p in [0, 1, 4, 9, -7] {
            assert!(
                matches!(
                    factor(&v(&[1, 1]), &int(p)),
                    Err(MathError::InvalidArgument(_))
                ),
                "p = {p}"
            );
            assert!(squarefree_decomposition(&v(&[1, 1]), &int(p)).is_err());
            assert!(q_matrix(&v(&[1, 1]), &int(p)).is_err());
        }
        // f ≡ 0 (mod p): gp factormod(3*x+3,3) -> [0, 1]; here no factorization.
        assert!(matches!(
            factor(&[], &int(3)),
            Err(MathError::InvalidArgument(_))
        ));
        assert!(matches!(
            factor(&v(&[3, 3]), &int(3)),
            Err(MathError::InvalidArgument(_))
        ));
        assert!(q_matrix(&v(&[5]), &int(3)).is_err());
    }

    // gp: polisirreducible(Mod(1,p)*f) for each (f, p) below; factormod(x^4+1,3) =
    // (x^2+x+2)(x^2+2x+2) is square-free with no root, so only the kernel dimension (2)
    // shows it is reducible; x^5+x+3 mod 7 has kernel dimension 1.
    #[test]
    fn is_irreducible_matches_pari() {
        let cases: [(&[i64], i64, bool); 12] = [
            (&[1, 0, 1], 3, true),
            (&[1, 0, 1], 5, false),
            (&[1, 0, 1], 2, false),
            (&[1, 0, 0, 0, 1], 3, false),
            (&[3, 1, 0, 0, 0, 1], 7, true),
            (&[1, 0, 0, 1], 2, false),
            (&[2, 7, 9, 5, 1], 3, false),
            (&[2, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, false),
            (&[1, 4], 3, true),
            (&[3], 3, false),
            (&[7], 3, false),
            (&[], 3, false),
        ];
        for (f, p, want) in cases {
            assert_eq!(
                is_irreducible(&v(f), &int(p)).unwrap(),
                want,
                "f = {f:?}, p = {p}"
            );
        }
        for p in [0, 1, 4, 9, -7] {
            assert!(matches!(
                is_irreducible(&v(&[1, 0, 1]), &int(p)),
                Err(MathError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn lcg_below_is_in_range() {
        let mut rng = Lcg(1);
        let m = Integer::from_decimal_str("2305843009213693951").unwrap();
        for _ in 0..100 {
            let x = rng.below(&m);
            assert!(x.signum() >= 0 && x < m);
        }
    }
}
