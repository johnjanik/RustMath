//! **Layer 6b** — genus from branch cycles, and branch-cycle recovery mod `p`.
//!
//! For a degree-`n` cover `X → P¹_t` with monodromy generators `g₁,…,g_r`
//! (the inertia/branch cycles, as partitions of `n`), Riemann–Hurwitz gives the
//! genus *without any function-field computation*:
//! ```text
//!   2g − 2 = −2n + Σ_i (n − #cycles(g_i)).
//! ```
//! This sidesteps the discriminant/integral-basis bottleneck entirely whenever
//! the branch cycles are known. [`genus_from_branch_cycles`] is exact
//! (`Algorithm`).
//!
//! When only `F(x,t)` is given, [`finite_branch_cycles_mod_p`] *recovers* the
//! finite branch cycles by reduction mod a prime `p`: at each `t₀ ∈ 𝔽_p` it
//! specialises `F(x,t₀)`, and in the tame case the inertia cycle type is the
//! multiplicity partition of the fibre (a residue-degree-`d`, multiplicity-`j`
//! factor contributes `d` parts of size `j`). Branch points not rational over
//! `𝔽_p` are invisible to the `𝔽_p` sweep, so [`genus_via_branch_cycles`] is a
//! `CertifyingSemialgorithm`: trustworthy only when completeness is confirmed
//! (e.g. by agreement across primes, or a discriminant-degree cross-check).

use crate::function_field::FfPoly;
use crate::genus::infinity_different;
use crate::ratfunc::{QtPoly, RationalFunction};
use crate::status::Verdict;
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor as fp;

/// `2g − 2 = −2n + Σ (n − #cycles)` ⇒ genus, from explicit branch cycles
/// (each a partition of `n`). Exact.
pub fn genus_from_branch_cycles(n: usize, cycles: &[Vec<usize>]) -> i64 {
    let n = n as i64;
    let ram: i64 = cycles.iter().map(|c| n - c.len() as i64).sum();
    1 - n + ram / 2
}

// ---- modular evaluation of ℚ(t) coefficients ----

fn imod(z: &Integer, p: i64) -> i64 {
    let r = (z.clone() % Integer::from(p)).to_i64();
    ((r % p) + p) % p
}
fn mul_mod(a: i64, b: i64, p: i64) -> i64 {
    (((a as i128) * (b as i128)).rem_euclid(p as i128)) as i64
}
fn eval_qt(poly: &QtPoly, t0: i64, p: i64) -> i64 {
    let mut acc = 0i64;
    for c in poly.coefficients().iter().rev() {
        let num = imod(c.numerator(), p);
        let den = imod(c.denominator(), p);
        let cm = match fp::mod_inv(den, p) {
            Some(di) => mul_mod(num, di, p),
            None => 0,
        };
        acc = (mul_mod(acc, t0, p) + cm).rem_euclid(p);
    }
    acc
}
fn eval_rf(rf: &RationalFunction, t0: i64, p: i64) -> Option<i64> {
    let num = eval_qt(rf.numerator(), t0, p);
    let den = eval_qt(rf.denominator(), t0, p);
    fp::mod_inv(den, p).map(|di| mul_mod(num, di, p))
}

/// `F(x, t₀) mod p` as a coefficient vector in `𝔽_p[x]` (ascending), or `None`
/// if a coefficient denominator vanishes mod `p`.
fn fibre(f: &FfPoly, t0: i64, p: i64) -> Option<Vec<i64>> {
    let n = f.degree()?;
    let mut out = vec![0i64; n + 1];
    for i in 0..=n {
        out[i] = eval_rf(f.coeff(i), t0, p)?;
    }
    Some(out)
}

// ---- Yun squarefree decomposition over 𝔽_p (p > n, so char-safe) ----

/// `F = Π s_j^j`; returns `(j, s_j)` with `s_j` squarefree, monic.
fn squarefree_decomp(f: &[i64], p: i64) -> Vec<(usize, Vec<i64>)> {
    let f = fp::make_monic(f, p);
    if fp::degree(&f) < 1 {
        return vec![];
    }
    let df = fp::derivative_of(&f, p);
    let a = fp::gcd(&f, &df, p);
    let (mut b, _) = fp::div_mod(&f, &a, p);
    let (mut c, _) = fp::div_mod(&df, &a, p);
    let mut d = fp::sub(&c, &fp::derivative_of(&b, p), p);
    let mut out = Vec::new();
    let mut j = 1usize;
    let bound = fp::degree(&f) as usize + 2;
    while fp::degree(&b) > 0 && j <= bound {
        let s = fp::gcd(&b, &d, p);
        if fp::degree(&s) > 0 {
            out.push((j, fp::make_monic(&s, p)));
        }
        let (b2, _) = fp::div_mod(&b, &s, p);
        let (c2, _) = fp::div_mod(&d, &s, p);
        b = b2;
        c = c2;
        d = fp::sub(&c, &fp::derivative_of(&b, p), p);
        j += 1;
    }
    out
}

/// The inertia cycle type at `t₀` (tame): a multiplicity-`j`, degree-`d`
/// factor contributes `d` parts of size `j`. Returns `None` on a degenerate
/// fibre (degree drop / pole).
pub fn cycle_type_at(f: &FfPoly, t0: i64, p: i64) -> Option<Vec<usize>> {
    let n = f.degree()?;
    let poly = fibre(f, t0, p)?;
    if fp::degree(&poly) != n as i64 {
        return None;
    }
    let mut ct = Vec::new();
    for (j, s) in squarefree_decomp(&poly, p) {
        for _ in 0..(fp::degree(&s) as usize) {
            ct.push(j);
        }
    }
    ct.sort_unstable_by(|a, b| b.cmp(a));
    Some(ct)
}

/// Recover the finite branch cycles (cycle types at the ramified `𝔽_p`-fibres).
/// Returns `(cycles, n_degenerate_fibres)`.
pub fn finite_branch_cycles_mod_p(f: &FfPoly, p: i64) -> (Vec<Vec<usize>>, usize) {
    let n = f.degree().unwrap_or(0);
    let mut cycles = Vec::new();
    let mut bad = 0usize;
    for t0 in 0..p {
        match cycle_type_at(f, t0, p) {
            Some(ct) if ct.len() < n => cycles.push(ct),
            Some(_) => {}
            None => bad += 1,
        }
    }
    (cycles, bad)
}

/// Genus from one prime's recovered finite cycles + the ∞ contribution. Returns
/// `None` when the total different is odd (a sure sign some branch point is not
/// `𝔽_p`-rational, hence missed by the `𝔽_p` sweep).
fn genus_one_prime(f: &FfPoly, p: i64) -> Option<i64> {
    let n = f.degree()? as i64;
    let (finite, _bad) = finite_branch_cycles_mod_p(f, p);
    let finite_ram: i64 = finite.iter().map(|c| n - c.len() as i64).sum();
    let total = finite_ram + infinity_different(f);
    if total % 2 != 0 {
        return None;
    }
    Some(1 - n + total / 2)
}

/// Genus via branch cycles recovered mod several primes, with **cross-prime
/// consistency** as the completeness check: the `𝔽_p` sweep only sees branch
/// points rational over `𝔽_p`, so a higher-degree branch place is invisible at
/// some primes — different `p` then disagree. We certify a genus only when the
/// primes agree (strong evidence every branch point was seen); otherwise the
/// honest output is `UNRESOLVED` (the cover needs an `𝔽_{p^k}` extension sweep).
/// `CertifyingSemialgorithm`.
pub fn genus_via_branch_cycles(f: &FfPoly, primes: &[i64]) -> Verdict<i64> {
    if f.degree().is_none() {
        return Verdict::unresolved("zero polynomial");
    }
    let gs: Vec<i64> = primes.iter().filter_map(|&p| genus_one_prime(f, p)).collect();
    if gs.is_empty() {
        return Verdict::unresolved("no prime gave an even different — branch points not 𝔽_p-rational");
    }
    if gs.iter().all(|&g| g == gs[0]) {
        Verdict::certifying(
            gs[0],
            format!("consistent across {} prime(s); all branch points 𝔽_p-rational", gs.len()),
        )
    } else {
        Verdict::unresolved(
            "primes disagree — some branch point is not 𝔽_p-rational; an 𝔽_{p^k} sweep is needed",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_field::ff_poly_from_coeffs;
    use rustmath_core::Ring;
    use rustmath_rationals::Rational;

    #[test]
    fn layer6b_mueller_genus_zero() {
        // Müller's M24-cover branch cycles on 24 points: (2^8 1^8) thrice + 12^2.
        let invol: Vec<usize> = std::iter::repeat(2).take(8).chain(std::iter::repeat(1).take(8)).collect();
        let inf = vec![12, 12];
        let cycles = vec![invol.clone(), invol.clone(), invol, inf];
        assert_eq!(genus_from_branch_cycles(24, &cycles), 0);
    }

    #[test]
    fn layer6b_hyperelliptic_genus_one() {
        // y^2 = t^3+1: four simple (transposition) branch points, each [2] on 2 sheets.
        let cycles = vec![vec![2], vec![2], vec![2], vec![2]];
        assert_eq!(genus_from_branch_cycles(2, &cycles), 1);
    }

    fn rf_t(coeffs: &[i64]) -> RationalFunction {
        RationalFunction::new(
            QtPoly::new(coeffs.iter().map(|&c| Rational::from_i64(c)).collect()),
            QtPoly::one(),
        )
        .unwrap()
    }
    fn rf(c: i64) -> RationalFunction {
        RationalFunction::new(QtPoly::new(vec![Rational::from_i64(c)]), QtPoly::one()).unwrap()
    }

    #[test]
    fn recover_branch_cycles_split_quadratic() {
        // x^2 - (t-1)(t-2) = x^2 - (t^2 - 3t + 2): branch points t=1,2 (𝔽_p-rational),
        // each fibre x^2 -> cycle type [2]; with the two ∞ sheets unramified -> genus 0.
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - rf_t(&[2, -3, 1]),
            rf(0),
            rf(1),
        ]);
        let (cycles, bad) = finite_branch_cycles_mod_p(&f, 101);
        assert_eq!(bad, 0);
        assert_eq!(cycles.len(), 2);
        assert!(cycles.iter().all(|c| c == &vec![2usize]));
        let v = genus_via_branch_cycles(&f, &[101, 103, 107]);
        assert_eq!(v.value, Some(0));
    }

    #[test]
    fn recover_elliptic_genus_one() {
        // x^2 - (t^3 + 1): branch points are the roots of t^3+1; pick p=103 where
        // t^3+1 splits (103 ≡ 1 mod 3) so all three are 𝔽_p-rational.
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - rf_t(&[1, 0, 0, 1]),
            rf(0),
            rf(1),
        ]);
        // primes ≡ 1 mod 3 so t^3+1 splits and all three branch points are 𝔽_p-rational.
        let v = genus_via_branch_cycles(&f, &[103, 109, 127]);
        assert_eq!(v.value, Some(1));
    }
}
