//! Factorization over ℤ[x] by the Zassenhaus method, over `Vec<Integer>`.
//!
//! Pipeline: content + [`zx::squarefree_decomposition`] → for each square-free
//! primitive factor, reduce to the monic case, pick a prime `p` keeping the
//! reduction square-free and yielding **few** factors mod `p` (a Chebotarev-cheap
//! choice — a prime where Frobenius is a long cycle proves irreducibility outright),
//! factor mod `p` ([`fp_factor`]), Hensel-lift to `p^k` past the **Landau–Mignotte**
//! coefficient bound ([`zp_hensel`]), then **recombine** lifted factors by subset
//! trial division.
//!
//! This replaces the rational-roots-only `factorization::factor_over_integers` with
//! a complete factorization, unblocking reliable irreducibility/factor testing for
//! the IGP24 degree-24 work.

use crate::{fp_factor, zp_hensel, zx};
use rustmath_integers::Integer;

/// Cap on how many valid primes to scan before committing to the one with the
/// fewest mod-`p` factors. More scanning ⇒ better chance of a low-factor prime
/// (cheaper recombination), at linear cost.
const MAX_PRIME_SCAN: usize = 30;

/// Hard cap on subsets tested during recombination, a guard against the classic
/// Zassenhaus exponential blow-up so a pathological input degrades to an error
/// rather than a hang.
const MAX_RECOMBINE: u64 = 1 << 22;

fn is_small_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2i64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// Reduce an integer polynomial mod `p` to `Vec<i64>` in `[0, p)`.
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

/// Landau–Mignotte: a bound `B` such that every coefficient of every factor of the
/// monic square-free `f` (degree `n`) has absolute value `≤ B`. Uses
/// `|b_i| ≤ binom(deg g, i)·‖f‖₂ ≤ 2^n·‖f‖₂`.
fn mignotte_bound(f: &[Integer]) -> Integer {
    let n = zx::degree(f).max(0) as u32;
    let mut sumsq = Integer::zero();
    for c in f {
        sumsq = sumsq + c.clone() * c.clone();
    }
    // ceil(sqrt(sumsq))
    let mut norm = sumsq.sqrt().unwrap_or_else(|_| Integer::zero());
    if norm.clone() * norm.clone() < sumsq {
        norm = norm + Integer::one();
    }
    Integer::from(2).pow(n) * norm
}

/// Reduce each coefficient of `f` into `[0, m)`.
fn reduce_poly(f: &[Integer], m: &Integer) -> Vec<Integer> {
    zp_hensel::reduce(f, m)
}

/// Balanced representative mod `m`: coefficients in `(-m/2, m/2]`.
fn balanced(f: &[Integer], m: &Integer) -> Vec<Integer> {
    let half = m.clone() / Integer::from(2);
    let r = reduce_poly(f, m);
    let out: Vec<Integer> = r
        .into_iter()
        .map(|c| if c > half { c - m.clone() } else { c })
        .collect();
    zx::trim(&out)
}

/// Multiply two polynomials and reduce coefficients into `[0, m)`.
fn mul_mod(a: &[Integer], b: &[Integer], m: &Integer) -> Vec<Integer> {
    reduce_poly(&zx::mul(a, b), m)
}

/// Choose a factoring prime. Returns `(p, factors_mod_p)` for the scanned prime
/// giving the fewest mod-`p` factors (a 1-factor prime proves irreducibility).
/// `None` if no suitable prime was found within the scan budget.
fn choose_prime(f: &[Integer]) -> Option<(i64, Vec<Vec<i64>>)> {
    let n = zx::degree(f);
    let mut best: Option<(i64, Vec<Vec<i64>>)> = None;
    let mut scanned = 0usize;
    let mut p = 2i64;
    while scanned < MAX_PRIME_SCAN && p < 100_000 {
        if !is_small_prime(p) {
            p += 1;
            continue;
        }
        let fp = reduce_mod_p(f, p);
        if fp_factor::degree(&fp) != n {
            p += 1; // p divides the leading coefficient
            continue;
        }
        // square-free mod p ?  gcd(fp, fp') constant
        let g = fp_factor::gcd(&fp, &fp_factor::derivative_of(&fp, p), p);
        if fp_factor::degree(&g) != 0 {
            p += 1;
            continue;
        }
        let factors = fp_factor::factor(&fp, p);
        scanned += 1;
        if factors.len() == 1 {
            return Some((p, factors)); // irreducible — stop immediately
        }
        match &best {
            Some((_, bf)) if bf.len() <= factors.len() => {}
            _ => best = Some((p, factors)),
        }
        p += 1;
    }
    best
}

/// Recombine Hensel-lifted factors of a monic `f` (mod `pk`) into the true ℤ
/// factors by subset trial division. Returns the irreducible monic factors.
fn recombine(f: &[Integer], lifted: Vec<Vec<Integer>>, pk: &Integer) -> Result<Vec<Vec<Integer>>, ()> {
    let mut pool: Vec<Vec<Integer>> = lifted.iter().map(|g| reduce_poly(g, pk)).collect();
    let mut factors: Vec<Vec<Integer>> = Vec::new();
    let mut remaining = zx::trim(f);
    let mut tested: u64 = 0;

    let mut s = 1usize;
    while 2 * s <= pool.len() {
        let mut found: Option<Vec<usize>> = None;
        let mut combo: Vec<usize> = (0..s).collect();
        loop {
            tested += 1;
            if tested > MAX_RECOMBINE {
                return Err(());
            }
            // product of the chosen lifted factors mod pk
            let mut prod = vec![Integer::one()];
            for &i in &combo {
                prod = mul_mod(&prod, &pool[i], pk);
            }
            let cand = balanced(&prod, pk);
            if zx::degree(&cand) >= 1 {
                if let Some(q) = zx::try_divide(&remaining, &cand) {
                    factors.push(cand);
                    remaining = q;
                    found = Some(combo.clone());
                    break;
                }
            }
            if !next_combination(&mut combo, pool.len()) {
                break;
            }
        }
        match found {
            Some(used) => {
                // drop used indices, restart from the smallest subset size
                for &i in used.iter().rev() {
                    pool.remove(i);
                }
                s = 1;
            }
            None => s += 1,
        }
    }
    if zx::degree(&remaining) >= 1 {
        factors.push(remaining);
    }
    Ok(factors)
}

/// Advance `combo` (a strictly increasing index list of fixed length) to the next
/// combination of `{0..n}` in lexicographic order. Returns false when exhausted.
fn next_combination(combo: &mut [usize], n: usize) -> bool {
    let k = combo.len();
    if k == 0 {
        return false;
    }
    let mut i = k;
    while i > 0 {
        i -= 1;
        if combo[i] != i + n - k {
            combo[i] += 1;
            for j in i + 1..k {
                combo[j] = combo[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// Factor a monic square-free `f` (degree ≥ 2) into monic irreducibles over ℤ.
fn factor_monic_squarefree(f: &[Integer]) -> Result<Vec<Vec<Integer>>, ()> {
    let (p, fp_factors) = choose_prime(f).ok_or(())?;
    if fp_factors.len() == 1 {
        return Ok(vec![zx::trim(f)]);
    }
    // smallest k with p^k >= 2·B + 1
    let bound = mignotte_bound(f) * Integer::from(2) + Integer::one();
    let pi = Integer::from(p);
    let mut k = 1u32;
    let mut pk = pi.clone();
    while pk < bound {
        pk = pk * pi.clone();
        k += 1;
    }
    let lifted = zp_hensel::hensel_lift_all(f, &fp_factors, p, k).ok_or(())?;
    recombine(f, lifted, &pk)
}

/// Scale the variable: return `g(c·x)`  (coefficient `i` ↦ `g_i · c^i`).
fn scale_var(g: &[Integer], c: &Integer) -> Vec<Integer> {
    let mut cp = Integer::one();
    let mut out = Vec::with_capacity(g.len());
    for gi in g {
        out.push(gi.clone() * cp.clone());
        cp = cp * c.clone();
    }
    zx::trim(&out)
}

/// Factor a primitive square-free `f` into primitive irreducibles over ℤ,
/// handling a non-monic leading coefficient by the monic substitution
/// `g(x) = lc^{n-1} f(x/lc)`.
fn factor_primitive_squarefree(f: &[Integer]) -> Result<Vec<Vec<Integer>>, ()> {
    let f = zx::normalize(f); // primitive, positive leading coeff
    let n = zx::degree(&f);
    if n <= 1 {
        return Ok(vec![f]);
    }
    let lc = f[f.len() - 1].clone();
    if lc.is_one() {
        return factor_monic_squarefree(&f);
    }
    // g(x) = lc^{n-1} f(x/lc):  g_i = f_i · lc^{n-1-i}  (and g_n = 1)
    let nu = n as u32;
    let mut g = vec![Integer::zero(); f.len()];
    for i in 0..f.len() {
        let e = (nu - 1).saturating_sub(i as u32);
        if (i as i64) < n {
            g[i] = f[i].clone() * lc.pow(e);
        } else {
            g[i] = Integer::one();
        }
    }
    let gfactors = factor_monic_squarefree(&zx::trim(&g))?;
    // back-substitute: factor of f = primitive_part( G(lc·x) )
    let mut out = Vec::with_capacity(gfactors.len());
    for gf in gfactors {
        out.push(zx::normalize(&zx::primitive_part(&scale_var(&gf, &lc))));
    }
    Ok(out)
}

/// Full factorization of `f ∈ ℤ[x]` into irreducible factors with multiplicity.
///
/// Returns `(content, factors)` where `content` is the (sign-carrying) integer
/// content and `factors` are primitive irreducibles with positive leading
/// coefficient: `f = content · ∏ gᵢ^{eᵢ}`. Returns `Err(())` only if recombination
/// exceeds [`MAX_RECOMBINE`] (pathological factor count) — never hangs.
pub fn factor(f: &[Integer]) -> Result<(Integer, Vec<(Vec<Integer>, u32)>), ()> {
    let f = zx::trim(f);
    if f.is_empty() {
        return Ok((Integer::zero(), Vec::new()));
    }
    if zx::degree(&f) == 0 {
        return Ok((f[0].clone(), Vec::new()));
    }
    // content carries the sign of the leading coefficient so factors stay positive-leading
    let mut cont = zx::content(&f);
    if f[f.len() - 1].signum() < 0 {
        cont = -cont;
    }
    let prim = zx::scalar_div_exact(&f, &cont);

    let mut out: Vec<(Vec<Integer>, u32)> = Vec::new();
    for (sf, mult) in zx::squarefree_decomposition(&prim) {
        for irr in factor_primitive_squarefree(&sf)? {
            out.push((irr, mult));
        }
    }
    Ok((cont, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    fn factors_only(f: &[Integer]) -> Vec<Vec<Integer>> {
        let (_, fs) = factor(f).unwrap();
        let mut v: Vec<Vec<Integer>> = fs.into_iter().map(|(g, _)| g).collect();
        v.sort();
        v
    }

    #[test]
    fn test_irreducible_quadratic() {
        // x^2 + 1 is irreducible over ℤ (the old factorizer's blind spot)
        let fs = factors_only(&p(&[1, 0, 1]));
        assert_eq!(fs, vec![p(&[1, 0, 1])]);
    }

    #[test]
    fn test_split_quadratic() {
        // x^2 - 1 = (x-1)(x+1)
        let mut expect = vec![p(&[-1, 1]), p(&[1, 1])];
        expect.sort();
        assert_eq!(factors_only(&p(&[-1, 0, 1])), expect);
    }

    #[test]
    fn test_product_of_irreducibles() {
        // (x^2+1)(x^2+2)(x^3+x+1)
        let f = zx::mul(&zx::mul(&p(&[1, 0, 1]), &p(&[2, 0, 1])), &p(&[1, 1, 0, 1]));
        let fs = factors_only(&f);
        let mut expect = vec![p(&[1, 0, 1]), p(&[2, 0, 1]), p(&[1, 1, 0, 1])];
        expect.sort();
        assert_eq!(fs, expect);
    }

    #[test]
    fn test_repeated_factor_multiplicity() {
        // (x-1)^2 (x+2)
        let f = zx::mul(&zx::mul(&p(&[-1, 1]), &p(&[-1, 1])), &p(&[2, 1]));
        let (_, fs) = factor(&f).unwrap();
        // x-1 with multiplicity 2, x+2 with multiplicity 1
        let mut got: Vec<(Vec<Integer>, u32)> = fs;
        got.sort();
        assert!(got.contains(&(p(&[-1, 1]), 2)));
        assert!(got.contains(&(p(&[2, 1]), 1)));
    }

    #[test]
    fn test_content_extraction() {
        // 6x^2 - 6 = 6(x-1)(x+1)
        let (cont, fs) = factor(&p(&[-6, 0, 6])).unwrap();
        assert_eq!(cont, Integer::from(6));
        let mut v: Vec<Vec<Integer>> = fs.into_iter().map(|(g, _)| g).collect();
        v.sort();
        let mut expect = vec![p(&[-1, 1]), p(&[1, 1])];
        expect.sort();
        assert_eq!(v, expect);
    }

    #[test]
    fn test_nonmonic_primitive() {
        // 2x^2 + 3x + 1 = (2x+1)(x+1)
        let fs = factors_only(&p(&[1, 3, 2]));
        let mut expect = vec![p(&[1, 1]), p(&[1, 2])];
        expect.sort();
        assert_eq!(fs, expect);
    }

    #[test]
    fn test_cyclotomic_15() {
        // Φ_15 = x^8 - x^7 + x^5 - x^4 + x^3 - x + 1 is irreducible
        let phi15 = p(&[1, -1, 0, 1, -1, 1, 0, -1, 1]);
        assert_eq!(factors_only(&phi15), vec![phi15]);
    }

    #[test]
    fn test_swinnerton_dyer_2_3() {
        // (x^2-2)(x^2-3) ... min poly of √2+√3 is x^4-10x^2+1, irreducible,
        // but splits into 4 linear factors mod every prime (Zassenhaus stress test)
        let sd = p(&[1, 0, -10, 0, 1]);
        assert_eq!(factors_only(&sd), vec![sd]);
    }
}
