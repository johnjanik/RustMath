//! # Dimensions of spaces of modular forms for Gamma0(N)
//!
//! Everything here is computed from the exact classical formulas; nothing is
//! read out of a table and nothing is approximated.
//!
//! ## The Gamma0(N) invariants
//!
//! With `psi(N) = [SL2(Z) : Gamma0(N)] = N prod_{p | N} (1 + 1/p)`,
//! `nu2 = #{x mod N : x^2 + 1 = 0}`, `nu3 = #{x mod N : x^2 + x + 1 = 0}` the
//! numbers of elliptic points of order 2 and 3, and
//! `eps_inf = sum_{d | N} phi(gcd(d, N/d))` the number of cusps, the genus of
//! X_0(N) is
//!
//! ```text
//!     g = 1 + psi/12 - nu2/4 - nu3/3 - eps_inf/2.
//! ```
//!
//! ## The dimension formula (Diamond-Shurman, Theorem 3.5.1)
//!
//! For even `k >= 2` (odd `k` gives the zero space because -I lies in
//! Gamma0(N) and acts by (-1)^k):
//!
//! ```text
//!     dim M_k = (k-1)(g-1) + floor(k/4) nu2 + floor(k/3) nu3 + (k/2) eps_inf
//!     dim S_k = dim M_k - eps_inf            (k >= 4)
//!     dim S_2 = dim M_2 - eps_inf + 1 = g    (the k = 2 correction)
//! ```
//! and `dim M_0 = 1`, `dim S_0 = 0`.
//!
//! ## New dimensions by Mobius inversion
//!
//! Strong multiplicity one gives the multiplicity of a level-M newform inside
//! level N as the number of `t | N/M`, i.e. `sigma_0(N/M)`:
//!
//! ```text
//!     dim S_k(Gamma0(N)) = sum_{M | N} sigma_0(N/M) * dim S_k^new(M).
//! ```
//!
//! As Dirichlet series this reads `dim S_k = sigma_0 * dim S_k^new`, and
//! `sigma_0 = 1 * 1` (the constant-one function convolved with itself), so the
//! Dirichlet inverse of `sigma_0` is `beta = mu * mu`, i.e.
//!
//! ```text
//!     beta(n) = sum_{d | n} mu(d) mu(n/d),
//!     dim S_k^new(N) = sum_{M | N} beta(N/M) * dim S_k(Gamma0(M)).
//! ```
//!
//! `beta` is multiplicative with `beta(1) = 1`, `beta(p) = -2`,
//! `beta(p^2) = 1` and `beta(p^e) = 0` for `e >= 3`; the identity
//! `sum_{d | n} sigma_0(d) beta(n/d) = [n = 1]` that makes the inversion valid
//! is re-verified for n = 1..=200 in the tests, and the resulting new
//! dimensions are cross-checked against the modular symbol spaces themselves
//! in `modsym::degeneracy`.
//!
//! ## Honest failure
//!
//! The `Integer`-returning functions cannot report an error, so each has a
//! `try_`-prefixed sibling returning `Result`; the plain function delegates and
//! PANICS with a precise message rather than inventing a plausible dimension.
//! The only inputs that fail are `N <= 0` and `N` too large for `u64` (the
//! weight is never a failure: odd and negative weights give the zero space
//! exactly, and every even weight >= 0 is covered by the formula above).
//!
//! Corresponds to SageMath's `sage.modular.dims` restricted to trivial
//! character.  The character-indexed Cohen-Oesterle quantities
//! (`CO_delta(r, p, N, eps)`, `CO_nu`, `CohenOesterle`) are NOT implemented:
//! the single-argument stubs that used to live here did not compute them and
//! were used only to fake the Gamma0 dimensions above, which are now exact.

use rustmath_core::NumericConversion;
use rustmath_integers::Integer;

/// The largest level these formulas will accept.
///
/// Everything below factors and enumerates divisors by TRIAL DIVISION, which is
/// O(sqrt(N)); at the cap that is at most 10^6 iterations.  Without a cap a level
/// near `u64::MAX` -- which `level_u64` would otherwise happily accept -- would
/// spin for ~2^32 iterations per factorization, and the `p * p <= n` guard would
/// itself overflow `u64` once `p` passed 2^32 (the loop is written as
/// `p <= n / p` below so that it cannot, but the running time stands).  Refusing
/// is the honest answer; silently taking hours is not.
const MAX_LEVEL: u64 = 1_000_000_000_000;

/// Factorization of n >= 1 by trial division, as (prime, exponent) pairs.
///
/// The caller must have bounded n by [`MAX_LEVEL`]; the `p <= n / p` test keeps
/// the loop guard itself free of overflow at any n.
fn factor_u64(mut n: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    let mut p = 2u64;
    while p <= n / p {
        if n.is_multiple_of(p) {
            let mut e = 0u32;
            while n.is_multiple_of(p) {
                n /= p;
                e += 1;
            }
            out.push((p, e));
        }
        p += 1;
    }
    if n > 1 {
        out.push((n, 1));
    }
    out
}

/// Euler's totient of p^e.
fn phi_prime_power(p: u64, e: u32) -> u128 {
    if e == 0 {
        1
    } else {
        (p as u128).pow(e - 1) * ((p - 1) as u128)
    }
}

/// The four invariants of Gamma0(N): the index psi(N), the numbers nu2, nu3 of
/// elliptic points of order 2 and 3, and the number of cusps.  All computed
/// multiplicatively from the factorization; the tests re-derive nu2, nu3 and
/// the cusp count by brute-force counting and compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gamma0Invariants {
    /// psi(N) = [SL2(Z) : Gamma0(N)].
    pub index: u128,
    /// Number of elliptic points of order 2.
    pub nu2: u128,
    /// Number of elliptic points of order 3.
    pub nu3: u128,
    /// Number of cusps of Gamma0(N).
    pub cusps: u128,
    /// Genus of X_0(N).
    pub genus: u128,
}

/// The Gamma0(N) invariants for 1 <= N <= [`MAX_LEVEL`].
pub fn gamma0_invariants(n: u64) -> Result<Gamma0Invariants, String> {
    if n == 0 {
        return Err("Gamma0(0) is not a group: the level must be >= 1".to_string());
    }
    if n > MAX_LEVEL {
        return Err(format!(
            "level {n} exceeds the cap {MAX_LEVEL}: these formulas factor and enumerate \
             divisors by trial division, which is O(sqrt(N)) and would spin for ~2^32 \
             iterations at a level near u64::MAX"
        ));
    }
    let f = factor_u64(n);

    // psi(N) = prod p^(e-1) (p + 1)
    let mut index: u128 = 1;
    for &(p, e) in &f {
        index *= (p as u128).pow(e - 1) * ((p + 1) as u128);
    }

    // nu2 = 0 if 4 | N, else prod_{p | N} (1 + (-1/p)); the Legendre symbol
    // (-1/p) is 1 for p = 1 mod 4, -1 for p = 3 mod 4, and the p = 2 factor is
    // 1 (there is exactly one square root of -1 mod 2).
    let nu2: u128 = if n.is_multiple_of(4) {
        0
    } else {
        f.iter()
            .map(|&(p, _)| match p % 4 {
                _ if p == 2 => 1u128,
                1 => 2,
                _ => 0,
            })
            .product()
    };

    // nu3 = 0 if 9 | N, else prod_{p | N} (1 + (-3/p)): 1 for p = 1 mod 3,
    // -1 for p = 2 mod 3, and the p = 3 factor is 1.
    let nu3: u128 = if n.is_multiple_of(9) {
        0
    } else {
        f.iter()
            .map(|&(p, _)| match p % 3 {
                _ if p == 3 => 1u128,
                1 => 2,
                _ => 0,
            })
            .product()
    };

    // #cusps = sum_{d | N} phi(gcd(d, N/d)), multiplicative with local factor
    // sum_{i=0}^{e} phi(p^min(i, e-i)).
    let cusps: u128 = f
        .iter()
        .map(|&(p, e)| {
            (0..=e)
                .map(|i| phi_prime_power(p, i.min(e - i)))
                .sum::<u128>()
        })
        .product();

    // g = 1 + (psi - 3 nu2 - 4 nu3 - 6 cusps) / 12
    let num = index as i128 - 3 * nu2 as i128 - 4 * nu3 as i128 - 6 * cusps as i128;
    if num % 12 != 0 {
        return Err(format!(
            "genus numerator {num} for X_0({n}) is not divisible by 12"
        ));
    }
    let g = 1 + num / 12;
    if g < 0 {
        return Err(format!("negative genus {g} for X_0({n})"));
    }

    Ok(Gamma0Invariants {
        index,
        nu2,
        nu3,
        cusps,
        genus: g as u128,
    })
}

/// The Mobius function of n >= 1.
fn mobius_u64(n: u64) -> i64 {
    let f = factor_u64(n);
    if f.iter().any(|&(_, e)| e > 1) {
        0
    } else if f.len().is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// The divisors of n >= 1, unsorted.
///
/// As with [`factor_u64`], the caller must have bounded n by [`MAX_LEVEL`]; the
/// `d <= n / d` test keeps the loop guard free of overflow at any n.
fn divisors_u64(n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut d = 1u64;
    while d <= n / d {
        if n.is_multiple_of(d) {
            out.push(d);
            if d != n / d {
                out.push(n / d);
            }
        }
        d += 1;
    }
    out
}

/// `beta = mu * mu`, the Dirichlet inverse of `sigma_0` (the number-of-divisors
/// function): `beta(n) = sum_{d | n} mu(d) mu(n/d)`.  Because
/// `sigma_0 = 1 * 1`, this is exactly the kernel that inverts the
/// old/new multiplicity law; see the module docs.
fn beta_dirichlet_inverse_of_sigma0(n: u64) -> i64 {
    divisors_u64(n)
        .into_iter()
        .map(|d| mobius_u64(d) * mobius_u64(n / d))
        .sum()
}

/// Number of divisors of n >= 1 (used to re-verify the multiplicity law that
/// the Mobius inversion above inverts).
#[cfg(test)]
fn sigma0_u64(n: u64) -> u64 {
    factor_u64(n)
        .iter()
        .map(|&(_, e)| (e + 1) as u64)
        .product()
}

/// Turn a level `Integer` into the `u64` the formulas need, or say why not.
fn level_u64(n: &Integer) -> Result<u64, String> {
    if n <= &Integer::zero() {
        return Err(format!("the level must be positive, got {n}"));
    }
    let n = n
        .to_u64()
        .ok_or_else(|| format!("level {n} does not fit in u64; the formulas are not applicable"))?;
    if n > MAX_LEVEL {
        return Err(format!(
            "level {n} exceeds the cap {MAX_LEVEL}: these formulas factor and enumerate \
             divisors by trial division, which is O(sqrt(N))"
        ));
    }
    Ok(n)
}

/// `dim S_k(Gamma0(N))` and `dim M_k(Gamma0(N))` for the invariants of a level.
///
/// Returns `(dim S_k, dim M_k)`.  Odd or negative `k` gives `(0, 0)`
/// (odd: -I in Gamma0(N) acts by (-1)^k, so the space is zero); `k = 0` gives
/// `(0, 1)`.
fn cusp_and_full_dim(inv: &Gamma0Invariants, k: i64) -> (i128, i128) {
    if k < 0 || k % 2 != 0 {
        return (0, 0);
    }
    if k == 0 {
        return (0, 1);
    }
    let (g, e2, e3, ei) = (
        inv.genus as i128,
        inv.nu2 as i128,
        inv.nu3 as i128,
        inv.cusps as i128,
    );
    let k = k as i128;
    let m = (k - 1) * (g - 1) + (k / 4) * e2 + (k / 3) * e3 + (k / 2) * ei;
    let s = m - ei + if k == 2 { 1 } else { 0 };
    (s, m)
}

/// Turn a nonnegative i128 dimension into an `Integer`.
fn dim_integer(d: i128, what: &str) -> Result<Integer, String> {
    if d < 0 {
        return Err(format!("{what} came out negative ({d})"));
    }
    i64::try_from(d)
        .map(Integer::from)
        .map_err(|_| format!("{what} = {d} does not fit in i64"))
}

/// `dim S_k(Gamma0(N))`, or an honest error.
pub fn try_dimension_cusp_forms(n: &Integer, k: i64) -> Result<Integer, String> {
    let n = level_u64(n)?;
    let inv = gamma0_invariants(n)?;
    dim_integer(cusp_and_full_dim(&inv, k).0, &format!("dim S_{k}(Gamma0({n}))"))
}

/// `dim M_k(Gamma0(N))`, or an honest error.
pub fn try_dimension_modular_forms(n: &Integer, k: i64) -> Result<Integer, String> {
    let n = level_u64(n)?;
    let inv = gamma0_invariants(n)?;
    dim_integer(cusp_and_full_dim(&inv, k).1, &format!("dim M_{k}(Gamma0({n}))"))
}

/// `dim E_k(Gamma0(N)) = dim M_k - dim S_k`, or an honest error.
pub fn try_dimension_eis(n: &Integer, k: i64) -> Result<Integer, String> {
    let n = level_u64(n)?;
    let inv = gamma0_invariants(n)?;
    let (s, m) = cusp_and_full_dim(&inv, k);
    dim_integer(m - s, &format!("dim E_{k}(Gamma0({n}))"))
}

/// `dim S_k^new(Gamma0(N))` by Mobius inversion of the multiplicity law
/// `dim S_k(N) = sum_{M | N} sigma_0(N/M) dim S_k^new(M)`, i.e.
/// `dim S_k^new(N) = sum_{M | N} beta(N/M) dim S_k(Gamma0(M))` with
/// `beta = mu * mu`.  Returns an honest error rather than a plausible integer.
pub fn try_dimension_new_cusp_forms(n: &Integer, k: i64) -> Result<Integer, String> {
    let n = level_u64(n)?;
    if k < 0 || k % 2 != 0 || k == 0 {
        // The whole cuspidal space is zero, hence so is its new part.
        return Ok(Integer::zero());
    }
    let mut total: i128 = 0;
    for m in divisors_u64(n) {
        let b = beta_dirichlet_inverse_of_sigma0(n / m) as i128;
        if b == 0 {
            continue;
        }
        let inv = gamma0_invariants(m)?;
        total += b * cusp_and_full_dim(&inv, k).0;
    }
    dim_integer(total, &format!("dim S_{k}^new(Gamma0({n}))"))
}

/// Number of Eisenstein series of weight k for Gamma0(N).
///
/// PANICS on a level that is not a positive `u64`; see [`try_dimension_eis`].
pub fn eisen(n: &Integer, k: i64) -> Integer {
    try_dimension_eis(n, k).expect("eisen: invalid level")
}

/// Dimension of the space of new cusp forms S_k^new(Gamma0(N)).
///
/// PANICS on a level that is not a positive `u64`; see
/// [`try_dimension_new_cusp_forms`] for the fallible form.  Every weight is
/// supported: odd and negative weights give 0 exactly (as -I in Gamma0(N) acts
/// by (-1)^k), and every even weight is covered by the exact formula.
pub fn dimension_new_cusp_forms(n: &Integer, k: i64) -> Integer {
    try_dimension_new_cusp_forms(n, k).expect("dimension_new_cusp_forms: invalid level")
}

/// Dimension of the space of cusp forms S_k(Gamma0(N)).
///
/// PANICS on a level that is not a positive `u64`; see
/// [`try_dimension_cusp_forms`].
pub fn dimension_cusp_forms(n: &Integer, k: i64) -> Integer {
    try_dimension_cusp_forms(n, k).expect("dimension_cusp_forms: invalid level")
}

/// Dimension of the space of Eisenstein series E_k(Gamma0(N)).
///
/// PANICS on a level that is not a positive `u64`; see [`try_dimension_eis`].
pub fn dimension_eis(n: &Integer, k: i64) -> Integer {
    eisen(n, k)
}

/// Dimension of the space of modular forms M_k(Gamma0(N)).
///
/// PANICS on a level that is not a positive `u64`; see
/// [`try_dimension_modular_forms`].
pub fn dimension_modular_forms(n: &Integer, k: i64) -> Integer {
    try_dimension_modular_forms(n, k).expect("dimension_modular_forms: invalid level")
}

/// The Sturm bound `B = floor(k psi(N) / 12)` for Gamma0(N), where
/// `psi(N) = [SL2(Z) : Gamma0(N)]`.
///
/// The theorem behind it is the valence formula: a nonzero `f` in
/// `M_k(Gamma0(N))` has `ord_inf(f) <= k psi(N) / 12`, because the orders of
/// vanishing of `f` over the whole fundamental domain are nonnegative and sum to
/// `k psi(N) / 12`, and the cusp `infinity` of Gamma0(N) has width 1 (so
/// `ord_inf` is exactly the index of the first nonzero q-coefficient).  Hence
/// `ord_inf(f) > B` forces `f = 0`, and `B` is the sharpest such bound because
/// `B + 1 > k psi(N) / 12` while `B` need not be.
///
/// The two usable forms differ in whether `a_0` is part of the count:
///
/// ```text
///     f in S_k(Gamma0(N)):   a_1 = ... = a_B = 0        ==>  f = 0
///     f in M_k(Gamma0(N)):   a_0 = a_1 = ... = a_B = 0  ==>  f = 0
/// ```
///
/// For a CUSP form `a_0 = 0` holds automatically, so `ord_inf(f) >= 1` and the
/// vanishing of `a_1, ..., a_B` already gives `ord_inf(f) > B`.  For the FULL
/// space `M_k` the count of the order of vanishing starts at `a_0`, and `a_0`
/// MUST be included in the hypothesis.
///
/// Dropping `a_0` from the `M_k` statement makes it FALSE, and not merely at the
/// margin: at `N = 2, k = 2` we have `psi(2) = 3` and `B = floor(6/12) = 0`, so
/// "a_1, ..., a_B all vanish" is a vacuous hypothesis, while
/// `dim M_2(Gamma0(2)) = 1` -- the space is spanned by the Eisenstein series
/// `2 E_2(2 tau) - E_2(tau) = 1 + 24q + 24q^2 + 96q^3 + ...`, whose `a_0 = 1`
/// is precisely the coefficient the vacuous hypothesis fails to constrain.
/// (`dim S_2(Gamma0(2)) = 0`, so the cusp-form form of the statement is fine
/// here.)  Equivalently, the injections these bounds encode are
///
/// ```text
///     dim S_k(Gamma0(N)) <= B        and        dim M_k(Gamma0(N)) <= B + 1,
/// ```
///
/// and the second cannot be sharpened to `<= B`.  Both are asserted, over a
/// range of (N, k) and at the (2, 2) witness, in `test_sturm_bound`.
///
/// PANICS on a level that is not a positive `u64`.
pub fn sturm_bound(n: &Integer, k: i64) -> Integer {
    try_sturm_bound(n, k).expect("sturm_bound: invalid level")
}

/// The Sturm bound, or an honest error.  See [`sturm_bound`] for the theorem it
/// satisfies -- in particular for why the `M_k` form of the statement must
/// include `a_0`.
pub fn try_sturm_bound(n: &Integer, k: i64) -> Result<Integer, String> {
    let n = level_u64(n)?;
    if k < 0 {
        return Err(format!("negative weight {k} has no Sturm bound"));
    }
    let inv = gamma0_invariants(n)?;
    dim_integer(
        (k as i128) * (inv.index as i128) / 12,
        &format!("Sturm bound for Gamma0({n}) in weight {k}"),
    )
}

/// The number of q-expansion coefficients that determine a form, i.e. the number
/// of coefficients whose vanishing forces the form to be zero:
/// `B` of them (`a_1, ..., a_B`) for `S_k`, and `B + 1` of them
/// (`a_0, ..., a_B`) for `M_k`, with `B = ` [`sturm_bound`].
///
/// This is the form of the bound one actually uses to compare or certify forms,
/// and it keeps the `a_0` asymmetry between the two spaces explicit rather than
/// leaving it to the caller to remember.
pub fn try_sturm_bound_coefficients(
    n: &Integer,
    k: i64,
    cuspidal: bool,
) -> Result<Integer, String> {
    let b = try_sturm_bound(n, k)?;
    Ok(if cuspidal { b } else { b + Integer::one() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: u64) -> Integer {
        Integer::from(n)
    }

    /// Brute-force counting definitions of the elliptic point counts and the
    /// cusp count, used only to certify the multiplicative formulas above.
    fn brute_nu2(n: u64) -> u128 {
        (0..n)
            .filter(|&x| ((x as u128 * x as u128) + 1).is_multiple_of(n as u128))
            .count() as u128
    }

    fn brute_nu3(n: u64) -> u128 {
        (0..n)
            .filter(|&x| ((x as u128 * x as u128) + x as u128 + 1).is_multiple_of(n as u128))
            .count() as u128
    }

    fn brute_cusps(n: u64) -> u128 {
        fn gcd(a: u64, b: u64) -> u64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        fn phi(mut m: u64) -> u128 {
            let mut r = m as u128;
            let mut p = 2u64;
            while p * p <= m {
                if m.is_multiple_of(p) {
                    while m.is_multiple_of(p) {
                        m /= p;
                    }
                    r -= r / p as u128;
                }
                p += 1;
            }
            if m > 1 {
                r -= r / m as u128;
            }
            r
        }
        divisors_u64(n)
            .into_iter()
            .map(|d| phi(gcd(d, n / d)))
            .sum()
    }

    /// psi(N) = #P^1(Z/N): pairs (c, d) mod N with gcd(c, d, N) = 1, modulo
    /// scaling by the units.  O(N^3), so only for small N.
    fn brute_index(n: u64) -> u128 {
        fn gcd(a: u64, b: u64) -> u64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        let mut classes: Vec<(u64, u64)> = Vec::new();
        for c in 0..n {
            for d in 0..n {
                if gcd(gcd(c, d), n) != 1 {
                    continue;
                }
                let known = classes.iter().any(|&(a, b)| {
                    (0..n).any(|u| gcd(u, n) == 1 && (u * a) % n == c && (u * b) % n == d)
                });
                if !known {
                    classes.push((c, d));
                }
            }
        }
        classes.len() as u128
    }

    /// GATE: the multiplicative formulas for the Gamma0(N) invariants agree
    /// with their brute-force counting definitions.  (The index is checked
    /// against #P^1(Z/N) only for small N, the counting being O(N^3).)
    #[test]
    fn test_gamma0_invariants_against_brute_force() {
        for n in 1..=200u64 {
            let inv = gamma0_invariants(n).unwrap();
            assert_eq!(inv.nu2, brute_nu2(n), "nu2({n})");
            assert_eq!(inv.nu3, brute_nu3(n), "nu3({n})");
            assert_eq!(inv.cusps, brute_cusps(n), "#cusps({n})");
        }
        for n in 1..=24u64 {
            assert_eq!(gamma0_invariants(n).unwrap().index, brute_index(n), "psi({n})");
        }
        assert!(gamma0_invariants(0).is_err());
    }

    /// GATE: beta really IS the Dirichlet inverse of sigma_0, i.e.
    /// sum_{d | n} sigma_0(d) beta(n/d) = [n = 1].  This is the identity that
    /// makes the Mobius inversion of the old/new multiplicity law valid, and
    /// it is checked here rather than assumed.
    #[test]
    fn test_beta_is_the_dirichlet_inverse_of_sigma0() {
        for n in 1..=200u64 {
            let conv: i64 = divisors_u64(n)
                .into_iter()
                .map(|d| sigma0_u64(d) as i64 * beta_dirichlet_inverse_of_sigma0(n / d))
                .sum();
            assert_eq!(conv, i64::from(n == 1), "(sigma_0 * beta)({n})");
        }
        // the closed form: multiplicative, beta(p) = -2, beta(p^2) = 1,
        // beta(p^e) = 0 for e >= 3
        assert_eq!(beta_dirichlet_inverse_of_sigma0(1), 1);
        for p in [2u64, 3, 5, 7, 11, 13] {
            assert_eq!(beta_dirichlet_inverse_of_sigma0(p), -2, "beta({p})");
            assert_eq!(beta_dirichlet_inverse_of_sigma0(p * p), 1, "beta({p}^2)");
            for e in 3..=5u32 {
                assert_eq!(beta_dirichlet_inverse_of_sigma0(p.pow(e)), 0, "beta({p}^{e})");
            }
        }
        assert_eq!(beta_dirichlet_inverse_of_sigma0(6), 4);
        assert_eq!(beta_dirichlet_inverse_of_sigma0(15), 4);
        assert_eq!(beta_dirichlet_inverse_of_sigma0(12), -2);
        assert_eq!(beta_dirichlet_inverse_of_sigma0(30), -8);
    }

    /// GATE: the forward multiplicity law that the inversion undoes:
    ///   dim S_k(Gamma0(N)) = sum_{M | N} sigma_0(N/M) dim S_k^new(M),
    /// for every N <= 200 and every weight in 2..=20.
    ///
    /// WHAT THIS DOES AND DOES NOT CERTIFY.  It certifies that
    /// `try_dimension_new_cusp_forms` really is the Mobius inversion of
    /// `try_dimension_cusp_forms` against sigma_0 -- i.e. that the two functions
    /// are mutually consistent, and, together with the nonnegativity assertion
    /// below, that the inversion of the true cusp dimensions lands in the
    /// nonnegative integers (which a wrong dimension formula generally would
    /// not).
    ///
    /// It does NOT certify the dimension formula itself.  Because beta is the
    /// exact Dirichlet inverse of sigma_0 (proved in
    /// `test_beta_is_the_dirichlet_inverse_of_sigma0`), substituting the
    /// definition of dim S_k^new collapses the identity to
    /// `sigma_0 * (beta * D) = D` for ANY function D in place of dim S_k -- so
    /// as an equation it holds no matter what `try_dimension_cusp_forms`
    /// returns.  The dimension formula is certified elsewhere: against
    /// brute-force counts of the Gamma0 invariants in
    /// `test_gamma0_invariants_against_brute_force`, and against PARI/GP in
    /// `test_landmark_dimensions` and
    /// `test_new_cusp_form_dimensions_against_pari`.
    #[test]
    fn test_old_new_multiplicity_law() {
        for n in 1..=200u64 {
            for k in [2i64, 4, 6, 8, 10, 12, 16, 20] {
                let lhs = try_dimension_cusp_forms(&int(n), k).unwrap();
                let mut rhs = Integer::zero();
                for m in divisors_u64(n) {
                    let mult = Integer::from(sigma0_u64(n / m));
                    let new = try_dimension_new_cusp_forms(&int(m), k).unwrap();
                    assert!(new >= Integer::zero(), "dim new S_{k}({m}) must be >= 0");
                    rhs = rhs + mult * new;
                }
                assert_eq!(lhs, rhs, "multiplicity law at N = {n}, k = {k}");
            }
        }
    }

    /// Landmark dimensions, each derived independently (PARI/GP `mfdim`, and
    /// the classical facts: dim S_12(SL2Z) = 1 generated by Delta;
    /// dim S_2(Gamma0(11)) = genus X_0(11) = 1; dim S_2(Gamma0(37)) = 2 with
    /// both newforms new; X_0(22) has genus 2 but NO newforms, its whole
    /// cuspidal space being old from level 11).
    #[test]
    fn test_landmark_dimensions() {
        // level 1
        assert_eq!(dimension_cusp_forms(&int(1), 12), int(1));
        assert_eq!(dimension_new_cusp_forms(&int(1), 12), int(1));
        assert_eq!(dimension_modular_forms(&int(1), 12), int(2));
        assert_eq!(dimension_cusp_forms(&int(1), 4), int(0));
        assert_eq!(dimension_modular_forms(&int(1), 4), int(1));
        assert_eq!(dimension_modular_forms(&int(1), 2), int(0));
        assert_eq!(dimension_cusp_forms(&int(1), 10), int(0));
        assert_eq!(dimension_cusp_forms(&int(1), 24), int(2));
        // weight 2 = genus of X_0(N)
        for (n, g) in [(11u64, 1u64), (22, 2), (23, 2), (33, 3), (37, 2), (44, 4), (100, 7)] {
            assert_eq!(dimension_cusp_forms(&int(n), 2), int(g), "genus X_0({n})");
        }
        // new dimensions in weight 2 (LMFDB/PARI: 11a; none at 22; 23a is a
        // 2-dim Galois orbit; 33a; 37a and 37b; 44a)
        for (n, d) in [(11u64, 1u64), (22, 0), (23, 2), (33, 1), (37, 2), (44, 1), (100, 1)] {
            assert_eq!(
                dimension_new_cusp_forms(&int(n), 2),
                int(d),
                "dim S_2^new(Gamma0({n}))"
            );
        }
        // a nontrivial higher weight: dim S_4(Gamma0(5)) = 1, dim S_6(Gamma0(3)) = 1
        assert_eq!(dimension_cusp_forms(&int(5), 4), int(1));
        assert_eq!(dimension_cusp_forms(&int(3), 6), int(1));
        assert_eq!(dimension_new_cusp_forms(&int(12), 6), int(0));
        // odd and negative weights: the space is exactly zero (-I acts by (-1)^k)
        assert_eq!(dimension_cusp_forms(&int(11), 3), int(0));
        assert_eq!(dimension_new_cusp_forms(&int(11), 3), int(0));
        assert_eq!(dimension_cusp_forms(&int(11), -2), int(0));
        // weight 0: only the constants
        assert_eq!(dimension_modular_forms(&int(11), 0), int(1));
        assert_eq!(dimension_cusp_forms(&int(11), 0), int(0));
    }

    /// GATE (non-tautological): dim S_k^new(Gamma0(N)) against externally
    /// derived constants, for every N = 1..=40 and k in {2, 4, 6, 8}.
    ///
    /// Unlike `test_old_new_multiplicity_law` -- which is an internal
    /// consistency check and cannot see an error in the dimension formula -- this
    /// table comes from OUTSIDE the crate: PARI/GP
    /// `mfdim(mfinit([N, k], 0))`, the dimension of the new subspace.  A wrong
    /// genus, a wrong cusp or elliptic-point count, a wrong Diamond-Shurman
    /// formula, or a wrong beta all show up here.
    #[test]
    fn test_new_cusp_form_dimensions_against_pari() {
        // PARI/GP:  for(N=1,40, print1(mfdim(mfinit([N,k],0))))
        let pari: [(i64, [u64; 40]); 4] = [
            (
                2,
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 2, 1, 0, 2,
                    1, 0, 2, 1, 2, 1, 1, 1, 3, 1, 2, 2, 3, 1,
                ],
            ),
            (
                4,
                [
                    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 1, 4, 1, 4, 1, 4, 3, 5, 1, 3, 3,
                    4, 2, 7, 2, 7, 3, 6, 4, 6, 1, 9, 5, 6, 3,
                ],
            ),
            (
                6,
                [
                    0, 0, 1, 1, 1, 1, 3, 1, 1, 3, 4, 0, 5, 2, 4, 2, 6, 3, 8, 1, 4, 5, 9, 3, 7, 5,
                    7, 2, 11, 2, 13, 5, 8, 8, 10, 2, 15, 7, 10, 5,
                ],
            ),
            (
                8,
                [
                    0, 1, 1, 0, 3, 1, 3, 2, 3, 1, 6, 2, 7, 4, 4, 3, 10, 2, 10, 3, 8, 5, 13, 3, 9,
                    7, 9, 4, 17, 6, 17, 7, 12, 8, 14, 3, 21, 11, 14, 7,
                ],
            ),
        ];
        for (k, row) in pari {
            for (i, &d) in row.iter().enumerate() {
                let n = (i + 1) as u64;
                assert_eq!(
                    dimension_new_cusp_forms(&int(n), k),
                    int(d),
                    "dim S_{k}^new(Gamma0({n})) against PARI mfdim(mfinit([{n},{k}],0))"
                );
            }
        }
    }

    /// Honest refusal on a level too large to factor by trial division, rather
    /// than an unbounded spin.  (`level_u64` used to accept anything up to
    /// `u64::MAX`, at which the trial division would run for ~2^32 iterations.)
    #[test]
    fn test_absurd_level_is_refused_not_spun_on() {
        let huge = Integer::from(u64::MAX);
        assert!(try_dimension_cusp_forms(&huge, 2).is_err());
        assert!(try_dimension_modular_forms(&huge, 2).is_err());
        assert!(try_dimension_new_cusp_forms(&huge, 2).is_err());
        assert!(try_sturm_bound(&huge, 2).is_err());
        assert!(gamma0_invariants(u64::MAX).is_err());
        // the cap itself is still accepted
        assert!(gamma0_invariants(MAX_LEVEL).is_ok());
        assert!(gamma0_invariants(MAX_LEVEL + 1).is_err());
    }

    /// The Eisenstein dimension is #cusps for k >= 4 and #cusps - 1 for k = 2.
    #[test]
    fn test_eisenstein_dimension() {
        for n in 1..=100u64 {
            let inv = gamma0_invariants(n).unwrap();
            let c = Integer::from(u64::try_from(inv.cusps).unwrap());
            assert_eq!(dimension_eis(&int(n), 4), c.clone(), "dim E_4(Gamma0({n}))");
            assert_eq!(dimension_eis(&int(n), 6), c.clone(), "dim E_6(Gamma0({n}))");
            assert_eq!(
                dimension_eis(&int(n), 2),
                c - Integer::one(),
                "dim E_2(Gamma0({n}))"
            );
            // M = S + E at every weight
            for k in [2i64, 4, 6, 12] {
                assert_eq!(
                    dimension_modular_forms(&int(n), k),
                    dimension_cusp_forms(&int(n), k) + dimension_eis(&int(n), k),
                    "M = S + E at N = {n}, k = {k}"
                );
            }
        }
        assert_eq!(eisen(&int(1), 4), int(1));
        assert_eq!(eisen(&int(1), 6), int(1));
        assert_eq!(eisen(&int(1), 3), int(0));
        assert_eq!(eisen(&int(11), 2), int(1));
    }

    /// GATE: the Sturm bound as a theorem that is TRUE of the code, in BOTH of
    /// its forms, and the witness that separates them.
    ///
    /// The bound B = floor(k psi(N)/12) encodes two injections (see
    /// [`sturm_bound`]): f |-> (a_1, ..., a_B) on S_k and f |-> (a_0, ..., a_B)
    /// on M_k.  So
    ///
    /// ```text
    ///     dim S_k <= B                dim M_k <= B + 1
    /// ```
    ///
    /// and the second is NOT sharpenable to dim M_k <= B.  The `saw_a0_matters`
    /// witness below is exactly the off-by-one: it collects the (N, k) at which
    /// dim M_k > B, i.e. at which the a_0-free statement "a_1..a_B all vanish
    /// implies f = 0 in M_k" is FALSE.  The doc used to assert precisely that
    /// statement, and (N, k) = (2, 2) refutes it: B = 0 there, so the hypothesis
    /// is vacuous, yet dim M_2(Gamma0(2)) = 1.
    ///
    /// A test asserting only `B >= dim S_k` (as this one used to) is blind to
    /// that: the cusp-form form of the theorem is true, and it was only the
    /// full-space form that was wrong.
    #[test]
    fn test_sturm_bound() {
        // psi(1) = 1, so the level-1 weight-12 bound is floor(12/12) = 1.
        assert_eq!(sturm_bound(&int(1), 12), int(1));
        // psi(11) = 12, so the weight-2 bound at level 11 is floor(24/12) = 2.
        assert_eq!(sturm_bound(&int(11), 2), int(2));
        // psi(37) = 38: floor(2 * 38 / 12) = 6.
        assert_eq!(sturm_bound(&int(37), 2), int(6));

        let mut saw_a0_matters = false;
        for n in 1..=120u64 {
            for k in [2i64, 4, 6, 8, 10, 12, 14, 16, 18, 20] {
                let b = sturm_bound(&int(n), k);
                let s = dimension_cusp_forms(&int(n), k);
                let m = dimension_modular_forms(&int(n), k);

                // S_k: a_1, ..., a_B determine the form.
                assert!(
                    b >= s,
                    "dim S_{k}(Gamma0({n})) = {s} exceeds the Sturm bound {b}"
                );
                // M_k: a_0, ..., a_B determine the form -- B + 1 coefficients.
                assert!(
                    b.clone() + Integer::one() >= m,
                    "dim M_{k}(Gamma0({n})) = {m} exceeds the Sturm bound + 1 = {} \
                     (the a_0-inclusive count)",
                    b.clone() + Integer::one()
                );
                // ... and B alone does NOT suffice for M_k.
                if m > b {
                    saw_a0_matters = true;
                }

                assert_eq!(try_sturm_bound_coefficients(&int(n), k, true).unwrap(), b);
                assert_eq!(
                    try_sturm_bound_coefficients(&int(n), k, false).unwrap(),
                    b + Integer::one()
                );
            }
        }
        assert!(
            saw_a0_matters,
            "no (N, k) in range has dim M_k > B, so this test cannot see the \
             a_0 off-by-one it exists to catch"
        );

        // THE COUNTEREXAMPLE, explicitly.  psi(2) = 3 and B = floor(2*3/12) = 0,
        // so "a_1, ..., a_B all vanish" is a VACUOUS hypothesis; if that sufficed
        // for M_k we would get M_2(Gamma0(2)) = 0.  It is 1-dimensional, spanned
        // by the Eisenstein series
        //   2 E_2(2 tau) - E_2(tau) = 1 + 24q + 24q^2 + 96q^3 + 24q^4 + ...
        // (the classical level-raising of the quasi-modular E_2:
        //  N E_2(N tau) - E_2(tau) lies in M_2(Gamma0(N)) -- Diamond-Shurman
        //  Exercise 1.2.8), whose a_0 = 1 is exactly the coefficient the vacuous
        // hypothesis leaves unconstrained.  Its a_1, ..., a_B is the empty list.
        assert_eq!(gamma0_invariants(2).unwrap().index, 3, "psi(2)");
        assert_eq!(sturm_bound(&int(2), 2), int(0));
        assert_eq!(dimension_modular_forms(&int(2), 2), int(1));
        assert_eq!(dimension_cusp_forms(&int(2), 2), int(0));
        // the a_0-inclusive count is what actually bounds it
        assert_eq!(
            try_sturm_bound_coefficients(&int(2), 2, false).unwrap(),
            int(1)
        );

        assert!(try_sturm_bound(&int(11), -2).is_err());
    }

    /// Honest refusal, not a plausible integer, on an impossible level.
    #[test]
    fn test_invalid_level_is_an_honest_error() {
        assert!(try_dimension_cusp_forms(&Integer::zero(), 2).is_err());
        assert!(try_dimension_new_cusp_forms(&Integer::from(-11), 2).is_err());
        assert!(try_dimension_modular_forms(&Integer::zero(), 4).is_err());
        assert!(try_dimension_eis(&Integer::zero(), 4).is_err());
    }

    #[test]
    #[should_panic(expected = "dimension_cusp_forms: invalid level")]
    fn test_zero_level_panics_rather_than_faking_a_dimension() {
        let _ = dimension_cusp_forms(&Integer::zero(), 2);
    }
}
