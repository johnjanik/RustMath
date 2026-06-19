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
pub(crate) fn factor_with_mult(fbar: &[i64], p: i64) -> Vec<(Vec<i64>, usize)> {
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

// --------------------------------------------------------------------------- //
// Ideal arithmetic (HNF Z-basis in integral-basis coordinates)
// --------------------------------------------------------------------------- //
use crate::round2::{bareiss_det, hnf_basis, maximal_order_data, OrderData};

/// A nonzero ideal of `O_K`, as an `n×n` HNF integer matrix whose columns are a
/// `ℤ`-basis in integral-basis coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ideal {
    pub basis: Vec<Vec<Integer>>, // n columns, each length n
    pub n: usize,
}

fn floor_div(a: &Integer, b: &Integer) -> Integer {
    let q = a.clone() / b.clone();
    let r = a.clone() - q.clone() * b.clone();
    if !r.is_zero() && (r.signum() as i64) * (b.signum() as i64) < 0 {
        q - Integer::one()
    } else {
        q
    }
}

/// Canonical HNF: `hnf_basis` gives a lower-triangular basis (column `r` has its
/// pivot at row `r`, zeros above); reduce each below-pivot entry into `[0, pivot)`
/// so equal lattices get identical bases (needed for ideal equality).
fn canonicalize(mut basis: Vec<Vec<Integer>>, n: usize) -> Vec<Vec<Integer>> {
    for i in 0..n {
        let piv = basis[i][i].clone();
        if piv.is_zero() {
            continue;
        }
        for j in 0..i {
            let q = floor_div(&basis[j][i], &piv);
            if !q.is_zero() {
                for r in 0..n {
                    basis[j][r] = basis[j][r].clone() - q.clone() * basis[i][r].clone();
                }
            }
        }
    }
    basis
}

fn make_ideal(cols: &[Vec<Integer>], n: usize) -> Ideal {
    Ideal { basis: canonicalize(hnf_basis(cols, n), n), n }
}

/// The `O_K`-ideal generated by `gens` (each an element in integral-basis coords):
/// the HNF of the `ℤ`-span of `{ωᵢ·g}`.
pub fn ideal_from_generators(ord: &OrderData, gens: &[Vec<Integer>]) -> Ideal {
    let n = ord.n;
    let mut cols: Vec<Vec<Integer>> = Vec::new();
    for g in gens {
        for i in 0..n {
            let mut ei = vec![Integer::zero(); n];
            ei[i] = Integer::one(); // ω_i in O_K coords
            cols.push(ord.mul(&ei, g));
        }
    }
    make_ideal(&cols, n)
}

/// Absolute norm `N(I) = [O_K : I] = |det(basis)|`.
pub fn ideal_norm(i: &Ideal) -> Integer {
    let n = i.n;
    let rows: Vec<Vec<Integer>> = (0..n).map(|r| (0..n).map(|c| i.basis[c][r].clone()).collect()).collect();
    bareiss_det(&rows).abs()
}

/// Product of two ideals: HNF of the `ℤ`-span of all `bᵢ·cⱼ`.
pub fn ideal_mul(ord: &OrderData, i: &Ideal, j: &Ideal) -> Ideal {
    let n = ord.n;
    let mut cols: Vec<Vec<Integer>> = Vec::new();
    for a in &i.basis {
        for b in &j.basis {
            cols.push(ord.mul(a, b));
        }
    }
    make_ideal(&cols, n)
}

/// The unit ideal `O_K` (identity basis).
pub fn one_ideal(ord: &OrderData) -> Ideal {
    ideal_from_generators(ord, &[ord.one()])
}

/// The ideal `(p) = p·O_K`.
pub fn rational_prime_ideal(ord: &OrderData, p: i64) -> Ideal {
    let p_one: Vec<Integer> = ord.one().iter().map(|c| c.clone() * Integer::from(p)).collect();
    ideal_from_generators(ord, &[p_one])
}

/// The prime ideals above `p` as actual ideals, with `(e, f)`:
/// `𝔭ᵢ = (p, gᵢ(θ))` for each mod-`p` factor `gᵢ`. (Exact when `p ∤ index`; for
/// `p | index` use the maximal-order routine — these generators may not be prime.)
pub fn prime_ideals(f: &[Integer], p: i64) -> (OrderData, Vec<(Ideal, usize, usize)>) {
    let ord = maximal_order_data(f);
    let fbar = reduce_mod_p(f, p);
    let factors = factor_with_mult(&fbar, p);
    let p_one: Vec<Integer> = ord.one().iter().map(|c| c.clone() * Integer::from(p)).collect();
    let mut out = Vec::new();
    for (g, e) in factors {
        let deg_g = (g.len() - 1).max(0);
        // g(θ) in power coordinates (degree < n), then to integral-basis coords
        let mut g_pow = vec![Integer::zero(); ord.n];
        for (i, &c) in g.iter().enumerate() {
            if i < ord.n {
                g_pow[i] = Integer::from(c);
            }
        }
        let g_ord = ord.power_to_order(&g_pow);
        let ideal = ideal_from_generators(&ord, &[p_one.clone(), g_ord]);
        out.push((ideal, e, deg_g));
    }
    (ord, out)
}

#[cfg(test)]
mod ideal_arith_tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn prime_ideal_norms_and_factorization_qi() {
        // K = Q(i), f = x^2 + 1.
        // p = 5 splits: two primes, each norm 5; product = (5), norm 25.
        let f = iz(&[1, 0, 1]);
        let (ord, primes) = prime_ideals(&f, 5);
        assert_eq!(primes.len(), 2);
        for (id, e, fdeg) in &primes {
            assert_eq!(*e, 1);
            assert_eq!(*fdeg, 1);
            assert_eq!(ideal_norm(id), Integer::from(5)); // N(𝔭) = p^f = 5
        }
        let prod = ideal_mul(&ord, &primes[0].0, &primes[1].0);
        assert_eq!(ideal_norm(&prod), Integer::from(25)); // N((5)) = 25
        assert_eq!(prod, rational_prime_ideal(&ord, 5)); // 𝔭_1·𝔭_2 = (5)
    }

    #[test]
    fn ramified_prime_squared_is_p() {
        // p = 2 ramifies in Q(i): 𝔭 = (2, i+1), 𝔭^2 = (2).
        let f = iz(&[1, 0, 1]);
        let (ord, primes) = prime_ideals(&f, 2);
        assert_eq!(primes.len(), 1);
        let (p2, e, _) = &primes[0];
        assert_eq!(*e, 2);
        assert_eq!(ideal_norm(p2), Integer::from(2));
        let sq = ideal_mul(&ord, p2, p2);
        assert_eq!(sq, rational_prime_ideal(&ord, 2));
        assert_eq!(ideal_norm(&sq), Integer::from(4));
    }

    #[test]
    fn norm_is_multiplicative() {
        // In Q(sqrt(-5)) (f = x^2 + 5), N(I·J) = N(I)·N(J). Use the split prime at 29
        // (or any two ideals). Here check N(𝔭·𝔮) = N(𝔭)N(𝔮) over p=3,7 primes.
        let f = iz(&[5, 0, 1]);
        let (ord3, p3) = prime_ideals(&f, 3); // 3 splits? -5 mod 3 = 1 (QR) → splits
        let (_o7, p7) = prime_ideals(&f, 7);
        let i = &p3[0].0;
        let j = &p7[0].0;
        let ni = ideal_norm(i);
        let nj = ideal_norm(j);
        let prod = ideal_mul(&ord3, i, j);
        assert_eq!(ideal_norm(&prod), ni * nj);
    }
}
