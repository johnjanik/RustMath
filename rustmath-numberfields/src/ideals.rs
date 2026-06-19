//! Prime-ideal factorization in a number field `K = ℚ[x]/(f)` — the `idealprimedec`
//! analogue, via the Dedekind–Kummer theorem.
//!
//! For a rational prime `p`, factor `f mod p = ∏ ḡᵢ^{eᵢ}` over `F_p`. **Dedekind's
//! criterion** decides whether `p ∤ [O_K : ℤ[θ]]`: with `ḡ = ∏ ḡᵢ` (radical),
//! `h̄ = f̄/ḡ`, and `T = (f − g·h)/p` (integer lifts), `p ∤ index` iff
//! `gcd(T̄, ḡ, h̄) = 1`. When it holds, the primes above `p` are `𝔭ᵢ = (p, gᵢ(θ))`
//! with ramification `eᵢ` and residue degree `deg ḡᵢ` — exactly the `(e, f)` data of
//! `idealprimedec`. When `p | index` the result is flagged; the maximal-order method
//! (Round 2, [`crate::round2`]) is then required.

use rustmath_integers::Integer;
use rustmath_polynomials::{fp_factor, zx};

/// One prime ideal above `p`: ramification `e`, residue degree `f`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimeIdeal {
    pub e: usize,
    pub f: usize,
}

/// The factorization of `(p)` in `O_K`.
#[derive(Debug, Clone)]
pub struct Factorization {
    pub p: i64,
    pub primes: Vec<PrimeIdeal>,
    /// `true` if `p | [O_K : ℤ[θ]]`, in which case `primes` is **not** the true
    /// decomposition and the maximal-order method is required.
    pub p_divides_index: bool,
}

impl Factorization {
    /// Sorted `(e, f)` multiset (the true decomposition iff `!p_divides_index`).
    pub fn ef(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = self.primes.iter().map(|pr| (pr.e, pr.f)).collect();
        v.sort_unstable();
        v
    }
}

fn reduce_mod_p(f: &[Integer], p: i64) -> Vec<i64> {
    let pi = Integer::from(p);
    let v: Vec<i64> = f
        .iter()
        .map(|c| {
            let r = (c.clone() % pi.clone()).to_i64();
            ((r % p) + p) % p
        })
        .collect();
    fp_factor::trim(&v)
}

/// `p`-th root of a polynomial that is a `p`-th power in `F_p[x]`: if `f = S(x)^p`
/// then `f = S(x^p)` (Frobenius), so `S` reads off the coefficients at multiples of
/// `p` (and `c^{1/p} = c` in `F_p`).
fn pth_root(f: &[i64], p: i64) -> Vec<i64> {
    let d = fp_factor::degree(f);
    if d <= 0 {
        return f.to_vec();
    }
    let pu = p as usize;
    let mut out = vec![0i64; (d as usize) / pu + 1];
    for (j, slot) in out.iter_mut().enumerate() {
        *slot = f[j * pu];
    }
    fp_factor::trim(&out)
}

/// Square-free decomposition over `F_p`: `(Aᵢ, i)` with `f = ∏ Aᵢ^i`, each `Aᵢ`
/// square-free and pairwise coprime. Handles `char p` via the `p`-th-root tail.
fn squarefree_decomp(f: &[i64], p: i64) -> Vec<(Vec<i64>, usize)> {
    let f = fp_factor::make_monic(f, p);
    if fp_factor::degree(&f) <= 0 {
        return Vec::new();
    }
    let fp = fp_factor::derivative_of(&f, p);
    if fp_factor::is_zero(&fp) {
        // f = r(x)^p  →  multiplicities scale by p
        return squarefree_decomp(&pth_root(&f, p), p)
            .into_iter()
            .map(|(a, m)| (a, m * p as usize))
            .collect();
    }
    let mut out = Vec::new();
    let mut g = fp_factor::gcd(&f, &fp, p);
    let mut w = fp_factor::div_mod(&f, &g, p).0; // separable radical (each p∤e factor once)
    let mut i = 1usize;
    while fp_factor::degree(&w) > 0 {
        let y = fp_factor::gcd(&w, &g, p);
        let a = fp_factor::div_mod(&w, &y, p).0; // factors of multiplicity exactly i
        if fp_factor::degree(&a) > 0 {
            out.push((a, i));
        }
        g = fp_factor::div_mod(&g, &y, p).0; // remove these factors from g
        w = y;
        i += 1;
    }
    // remaining g (if any) is a p-th power: the inseparable (p | e) factors
    if fp_factor::degree(&g) > 0 {
        for (a, m) in squarefree_decomp(&pth_root(&g, p), p) {
            out.push((a, m * p as usize));
        }
    }
    out
}

/// All distinct monic irreducible factors of `f mod p` with their multiplicities.
fn factor_with_mult(fbar: &[i64], p: i64) -> Vec<(Vec<i64>, usize)> {
    let mut out = Vec::new();
    for (sqfree, mult) in squarefree_decomp(fbar, p) {
        for g in fp_factor::factor(&sqfree, p) {
            out.push((g, mult));
        }
    }
    out
}

/// Prime-ideal factorization of `(p)` in `K = ℚ[x]/(f)` (f monic irreducible).
pub fn prime_decomposition(f: &[Integer], p: i64) -> Factorization {
    let fbar = reduce_mod_p(f, p);
    let factors = factor_with_mult(&fbar, p);

    let mut primes = Vec::new();
    let mut gbar = vec![1i64]; // ∏ distinct ḡ_i (radical)
    for (g, e) in &factors {
        primes.push(PrimeIdeal { e: *e, f: (g.len() - 1).max(0) });
        gbar = fp_factor::mul(&gbar, g, p);
    }
    // Dedekind's criterion: h̄ = f̄/ḡ ; T = (f − g·h)/p ; p∤index iff gcd(T̄,ḡ,h̄)=1
    let hbar = fp_factor::div_mod(&fbar, &gbar, p).0;
    let g_lift: Vec<Integer> = gbar.iter().map(|&c| Integer::from(c)).collect();
    let h_lift: Vec<Integer> = hbar.iter().map(|&c| Integer::from(c)).collect();
    let diff = zx::sub(f, &zx::mul(&g_lift, &h_lift)); // f − g·h, divisible by p
    let pi = Integer::from(p);
    let t_over_p: Vec<Integer> = diff.iter().map(|c| c.clone() / pi.clone()).collect();
    let tbar = reduce_mod_p(&t_over_p, p);
    let common = fp_factor::gcd(&fp_factor::gcd(&tbar, &gbar, p), &hbar, p);
    let p_divides_index = fp_factor::degree(&common) > 0;

    Factorization { p, primes, p_divides_index }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn squarefree_handles_repeats_and_wild() {
        // (x-1)^2(x-2) mod 5 = x^3 + x^2 + 3
        let d = factor_with_mult(&[3, 0, 1, 1], 5);
        let mut got: Vec<(Vec<i64>, usize)> = d;
        got.sort();
        // (x-1)=(x+4): [4,1] mult 2 ; (x-2)=(x+3): [3,1] mult 1
        assert!(got.contains(&(vec![4, 1], 2)));
        assert!(got.contains(&(vec![3, 1], 1)));
        // wild: x^2+1 mod 2 = (x+1)^2
        let w = factor_with_mult(&[1, 0, 1], 2);
        assert_eq!(w, vec![(vec![1, 1], 2)]);
    }

    #[test]
    fn quadratic_split_inert_ramified() {
        let f = iz(&[1, 0, 1]); // Q(i)
        assert_eq!(prime_decomposition(&f, 5).ef(), vec![(1, 1), (1, 1)]); // split
        assert_eq!(prime_decomposition(&f, 7).ef(), vec![(1, 2)]); // inert
        assert_eq!(prime_decomposition(&f, 2).ef(), vec![(2, 1)]); // ramified
        assert!(!prime_decomposition(&f, 5).p_divides_index);
    }

    #[test]
    fn dedekind_detects_index_divisor() {
        // Dedekind's cubic x^3 - x^2 - 2x - 8: 2 divides the index.
        let f = iz(&[-8, -2, -1, 1]);
        assert!(prime_decomposition(&f, 2).p_divides_index);
        let d5 = prime_decomposition(&f, 5);
        assert!(!d5.p_divides_index);
        // sum of e·f equals the degree
        assert_eq!(d5.ef().iter().map(|(e, f)| e * f).sum::<usize>(), 3);
    }
}
