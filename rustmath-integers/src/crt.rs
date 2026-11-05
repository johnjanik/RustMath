//! Chinese Remainder Theorem implementation

use crate::Integer;
use rustmath_core::{MathError, Result};

/// Solve the Chinese Remainder Theorem
///
/// Given congruences:
/// x ≡ a₁ (mod m₁)
/// x ≡ a₂ (mod m₂)
/// ...
/// x ≡ aₙ (mod mₙ)
///
/// Returns x (mod M) where M = m₁ × m₂ × ... × mₙ
///
/// The moduli must be pairwise coprime (gcd(mᵢ, mⱼ) = 1 for i ≠ j)
pub fn chinese_remainder_theorem(remainders: &[Integer], moduli: &[Integer]) -> Result<Integer> {
    if remainders.len() != moduli.len() {
        return Err(MathError::InvalidArgument(
            "Number of remainders must match number of moduli".to_string(),
        ));
    }

    if remainders.is_empty() {
        return Err(MathError::InvalidArgument(
            "Need at least one congruence".to_string(),
        ));
    }

    // Check that all moduli are positive
    for m in moduli {
        if m.signum() <= 0 {
            return Err(MathError::InvalidArgument(
                "All moduli must be positive".to_string(),
            ));
        }
    }

    // Check that moduli are pairwise coprime
    for i in 0..moduli.len() {
        for j in (i + 1)..moduli.len() {
            let gcd = moduli[i].gcd(&moduli[j]);
            if !gcd.is_one() {
                return Err(MathError::InvalidArgument(format!(
                    "Moduli must be pairwise coprime: gcd({}, {}) = {}",
                    moduli[i], moduli[j], gcd
                )));
            }
        }
    }

    // Compute M = m₁ × m₂ × ... × mₙ
    let mut big_m = Integer::one();
    for m in moduli {
        big_m = big_m * m.clone();
    }

    // Apply CRT formula
    let mut result = Integer::zero();

    for (a, m) in remainders.iter().zip(moduli.iter()) {
        // Mᵢ = M / mᵢ
        let big_m_i = big_m.clone() / m.clone();

        // Find yᵢ such that Mᵢ × yᵢ ≡ 1 (mod mᵢ)
        // Using extended GCD: gcd(Mᵢ, mᵢ) = 1 = s×Mᵢ + t×mᵢ
        // So s×Mᵢ ≡ 1 (mod mᵢ), thus yᵢ = s
        let (gcd, y_i, _) = big_m_i.extended_gcd(m);

        if !gcd.is_one() {
            return Err(MathError::InvalidArgument(format!(
                "Internal error: gcd should be 1, got {}",
                gcd
            )));
        }

        // Add aᵢ × Mᵢ × yᵢ to result
        result = result + a.clone() * big_m_i * y_i;
    }

    // Reduce result modulo M
    result = result % big_m.clone();

    // Ensure result is in range [0, M)
    if result.signum() < 0 {
        result = result + big_m.clone();
    }

    Ok(result)
}

/// Solve CRT for two congruences (simpler interface)
pub fn crt_two(a1: &Integer, m1: &Integer, a2: &Integer, m2: &Integer) -> Result<Integer> {
    chinese_remainder_theorem(&[a1.clone(), a2.clone()], &[m1.clone(), m2.clone()])
}

/// Symmetric height bound for bounded-height rational reconstruction modulo `m`.
///
/// Returns `N = floor(sqrt((m-1)/2))`. Rational reconstruction of a value modulo
/// `m` uniquely recovers any rational `n/d` in lowest terms with `|n| <= N` and
/// `0 < d <= N` (which forces `2*|n|*d < m`). A multi-prime CRT caller can
/// therefore stop adding primes as soon as this bound exceeds the largest
/// numerator/denominator it expects (the "height" `H`): once `N >= H` the
/// reconstruction is uniquely determined. Returns `0` for `m <= 0`.
pub fn rational_reconstruct_bound(m: &Integer) -> Integer {
    if m.signum() <= 0 {
        return Integer::zero();
    }
    // floor(sqrt((m-1)/2)); Integer::sqrt is a floor square root on non-negatives.
    let half = (m.clone() - Integer::one()) / Integer::from(2i64);
    half.sqrt().unwrap_or_else(|_| Integer::zero())
}

/// Bounded-height (Wang / half-GCD) rational reconstruction.
///
/// Given a residue `a` and a positive modulus `m`, find the unique rational
/// `n/d` in lowest terms with `|n|, d <= floor(sqrt((m-1)/2))`, `d > 0`, and
/// `n ≡ a * d (mod m)`. Returns `Some((n, d))`, or `None` when no such rational
/// exists within the height bound (i.e. `m` is too small to determine the
/// value, or the moduli/inputs are degenerate).
///
/// The algorithm is the standard truncated extended-Euclidean algorithm on the
/// pair `(m, a)`: it maintains rows `(r_i, t_i)` with the invariant
/// `r_i ≡ t_i * a (mod m)` and stops at the first remainder `r_i` that drops to
/// or below the height bound; the candidate is then `n/d = r_i / t_i`, accepted
/// only if `d` is also within the bound and `gcd(n, d) = 1`.
///
/// It is deliberately self-contained (only `Integer` arithmetic) so that
/// `rustmath-integers` need not depend on `rustmath-rationals`.
pub fn rational_reconstruct(a: &Integer, m: &Integer) -> Option<(Integer, Integer)> {
    if m.signum() <= 0 {
        return None;
    }
    let bound = rational_reconstruct_bound(m);

    // Reduce a into [0, m).
    let mut r1 = a.clone() % m.clone();
    if r1.signum() < 0 {
        r1 = r1 + m.clone();
    }
    let mut r0 = m.clone();
    let mut t0 = Integer::zero();
    let mut t1 = Integer::one();

    // Truncated Euclid: r1 strictly decreases each step, so this terminates.
    while r1 > bound {
        let q = r0.clone() / r1.clone();
        let r2 = r0 - q.clone() * r1.clone();
        let t2 = t0 - q * t1.clone();
        r0 = r1;
        r1 = r2;
        t0 = t1;
        t1 = t2;
    }

    // Candidate n/d = r1/t1. r1 >= 0 by construction; normalize d > 0.
    let mut n = r1;
    let mut d = t1;
    if d.signum() < 0 {
        n = -n;
        d = -d;
    }
    if d.is_zero() || d > bound {
        // No denominator within the height bound: value not determined.
        return None;
    }
    if !n.gcd(&d).is_one() {
        return None;
    }
    Some((n, d))
}

/// Multi-prime CRT combine followed by bounded-height rational reconstruction.
///
/// `residues` is a list of `(a_i, p_i)` meaning the unknown value is
/// `≡ a_i (mod p_i)`. The moduli `p_i` must be positive and pairwise coprime.
/// This CRT-combines them into `(a mod M)` with `M = ∏ p_i`, then runs
/// bounded-height rational reconstruction (Wang / half-GCD) to recover the
/// rational `n/d` the residues encode.
///
/// Returns `Some((numerator, denominator))` in lowest terms with
/// `denominator > 0`, or `None` if:
/// * `residues` is empty,
/// * the moduli are not positive or not pairwise coprime (CRT is undefined), or
/// * `M` is too small to determine a rational within the height bound
///   `floor(sqrt((M-1)/2))` — the reconstruction is not uniquely decided.
///
/// Early-exit: reconstructing a rational of height `H = max(|n|, d)` is
/// guaranteed once `2*H^2 < M`, equivalently
/// `rational_reconstruct_bound(&M) >= H`. A caller adding primes on the fly can
/// stop as soon as the running product `M` satisfies that bound.
///
/// The return type is `(numerator, denominator)` rather than a `Rational`
/// because `rustmath-rationals` depends on `rustmath-integers`; returning that
/// type here would introduce a dependency cycle. Callers build their own
/// rational from the pair (`Rational::new(n, d)`).
///
/// ```
/// use rustmath_integers::crt::crt_rational_reconstruct;
/// use rustmath_integers::Integer;
///
/// // The value 3/4, presented modulo three small primes.
/// let residues = [
///     (Integer::from(26i64), Integer::from(101i64)),
///     (Integer::from(78i64), Integer::from(103i64)),
///     (Integer::from(81i64), Integer::from(107i64)),
/// ];
/// let (n, d) = crt_rational_reconstruct(&residues).unwrap();
/// assert_eq!(n, Integer::from(3i64));
/// assert_eq!(d, Integer::from(4i64));
/// ```
pub fn crt_rational_reconstruct(residues: &[(Integer, Integer)]) -> Option<(Integer, Integer)> {
    if residues.is_empty() {
        return None;
    }

    let remainders: Vec<Integer> = residues.iter().map(|(a, _)| a.clone()).collect();
    let moduli: Vec<Integer> = residues.iter().map(|(_, p)| p.clone()).collect();

    // CRT-combine; an honest None on non-coprime / non-positive moduli.
    let a = chinese_remainder_theorem(&remainders, &moduli).ok()?;

    let mut big_m = Integer::one();
    for p in &moduli {
        big_m = big_m * p.clone();
    }

    rational_reconstruct(&a, &big_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crt_basic() {
        // x ≡ 2 (mod 3)
        // x ≡ 3 (mod 5)
        // x ≡ 2 (mod 7)
        // Solution: x = 23 (mod 105)
        let remainders = vec![Integer::from(2), Integer::from(3), Integer::from(2)];
        let moduli = vec![Integer::from(3), Integer::from(5), Integer::from(7)];

        let result = chinese_remainder_theorem(&remainders, &moduli).unwrap();
        assert_eq!(result, Integer::from(23));

        // Verify the solution
        assert_eq!(result.clone() % Integer::from(3), Integer::from(2));
        assert_eq!(result.clone() % Integer::from(5), Integer::from(3));
        assert_eq!(result % Integer::from(7), Integer::from(2));
    }

    #[test]
    fn test_crt_two() {
        // x ≡ 2 (mod 3)
        // x ≡ 3 (mod 5)
        // Solution: x = 8 (mod 15)
        let result = crt_two(
            &Integer::from(2),
            &Integer::from(3),
            &Integer::from(3),
            &Integer::from(5),
        )
        .unwrap();

        assert_eq!(result, Integer::from(8));
        assert_eq!(result.clone() % Integer::from(3), Integer::from(2));
        assert_eq!(result % Integer::from(5), Integer::from(3));
    }

    #[test]
    fn test_crt_not_coprime() {
        // Moduli 6 and 9 are not coprime (gcd = 3)
        let result = crt_two(
            &Integer::from(1),
            &Integer::from(6),
            &Integer::from(2),
            &Integer::from(9),
        );

        assert!(result.is_err());
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }

    #[test]
    fn test_crt_large() {
        // Test with larger numbers
        let remainders = vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        ];
        let moduli = vec![
            Integer::from(5),
            Integer::from(7),
            Integer::from(9),
            Integer::from(11),
        ];

        let result = chinese_remainder_theorem(&remainders, &moduli).unwrap();

        // Verify the solution
        for (a, m) in remainders.iter().zip(moduli.iter()) {
            assert_eq!(result.clone() % m.clone(), a.clone());
        }
    }

    // Independent oracle for the following tests (python3 + sympy):
    //   n=987654321, d=1000000007 (prime, so gcd=1); primes ~1e6:
    //     [1000003,1000033,1000037,1000039,1000081].
    //   residue a_i = (n * d^{-1}) mod p_i. M = prod p_i ~1e30 (100 bits),
    //   comfortably above 2*|n|*d ~ 1.98e18, so reconstruction is decided.
    //   Verified crt_rational_reconstruct recovers exactly (n, d).

    #[test]
    fn test_crt_rational_reconstruct_recovers_large_height() {
        // 987654321 / 1000000007
        let residues = [
            (Integer::from(297812i64), Integer::from(1000003i64)),
            (Integer::from(240919i64), Integer::from(1000033i64)),
            (Integer::from(830928i64), Integer::from(1000037i64)),
            (Integer::from(257938i64), Integer::from(1000039i64)),
            (Integer::from(623480i64), Integer::from(1000081i64)),
        ];
        let (n, d) = crt_rational_reconstruct(&residues).expect("should reconstruct");
        assert_eq!(n, Integer::from(987654321i64));
        assert_eq!(d, Integer::from(1000000007i64));
    }

    #[test]
    fn test_crt_rational_reconstruct_negative_numerator() {
        // -876543219 / 999999937 (prime denominator)
        let residues = [
            (Integer::from(857182i64), Integer::from(1000003i64)),
            (Integer::from(452621i64), Integer::from(1000033i64)),
            (Integer::from(742183i64), Integer::from(1000037i64)),
            (Integer::from(685855i64), Integer::from(1000039i64)),
            (Integer::from(299945i64), Integer::from(1000081i64)),
        ];
        let (n, d) = crt_rational_reconstruct(&residues).expect("should reconstruct");
        assert_eq!(n, Integer::from(-876543219i64));
        assert_eq!(d, Integer::from(999999937i64));
    }

    #[test]
    fn test_crt_rational_reconstruct_none_when_modulus_too_small() {
        // Same value 987654321/1000000007 but only two primes: M ~1e12, far
        // below 2*|n|*d ~1.98e18. The height bound floor(sqrt((M-1)/2)) ~7.1e5
        // cannot cover a ~1e9 numerator/denominator, so the reconstruction is
        // not determined. Oracle (sympy mirror of the same algorithm): None.
        let residues = [
            (Integer::from(297812i64), Integer::from(1000003i64)),
            (Integer::from(240919i64), Integer::from(1000033i64)),
        ];
        assert_eq!(crt_rational_reconstruct(&residues), None);
    }

    #[test]
    fn test_crt_rational_reconstruct_integer_and_zero() {
        // Integer value 5 == 5/1 over the 5-prime CRT.
        let five = [
            (Integer::from(5i64), Integer::from(1000003i64)),
            (Integer::from(5i64), Integer::from(1000033i64)),
            (Integer::from(5i64), Integer::from(1000037i64)),
            (Integer::from(5i64), Integer::from(1000039i64)),
            (Integer::from(5i64), Integer::from(1000081i64)),
        ];
        assert_eq!(
            crt_rational_reconstruct(&five),
            Some((Integer::from(5i64), Integer::from(1i64)))
        );

        // Zero == 0/1.
        let zero = [
            (Integer::from(0i64), Integer::from(1000003i64)),
            (Integer::from(0i64), Integer::from(1000033i64)),
            (Integer::from(0i64), Integer::from(1000037i64)),
            (Integer::from(0i64), Integer::from(1000039i64)),
            (Integer::from(0i64), Integer::from(1000081i64)),
        ];
        assert_eq!(
            crt_rational_reconstruct(&zero),
            Some((Integer::from(0i64), Integer::from(1i64)))
        );
    }

    #[test]
    fn test_crt_rational_reconstruct_non_coprime_moduli_is_none() {
        // Moduli 6 and 9 share the factor 3; CRT is undefined -> honest None.
        let residues = [
            (Integer::from(1i64), Integer::from(6i64)),
            (Integer::from(2i64), Integer::from(9i64)),
        ];
        assert_eq!(crt_rational_reconstruct(&residues), None);
    }

    #[test]
    fn test_crt_rational_reconstruct_empty_is_none() {
        assert_eq!(crt_rational_reconstruct(&[]), None);
    }

    #[test]
    fn test_rational_reconstruct_bound_matches_floor_sqrt() {
        // M = 101*103*107 = 1113121; floor(sqrt((M-1)/2)) = 746 (sympy).
        let m = Integer::from(1113121i64);
        assert_eq!(rational_reconstruct_bound(&m), Integer::from(746i64));
    }

    #[test]
    fn test_rational_reconstruct_direct_and_round_trip() {
        // Direct reconstruction of 3/4 modulo M = 101*103*107.
        // a = 3 * 4^{-1} mod M; recompute a independently via CRT of the parts.
        let m = Integer::from(1113121i64);
        // 4^{-1} mod M then times 3, computed here from scratch:
        let four_inv = Integer::from(4i64)
            .mod_inverse(&m)
            .expect("4 invertible mod odd M");
        let a = (Integer::from(3i64) * four_inv) % m.clone();
        assert_eq!(
            rational_reconstruct(&a, &m),
            Some((Integer::from(3i64), Integer::from(4i64)))
        );

        // A rational whose height exceeds the bound (746) is not determined:
        // 1000/999 has denominator 999 > 746, so None.
        let big_inv = Integer::from(999i64).mod_inverse(&m).unwrap();
        let a2 = (Integer::from(1000i64) * big_inv) % m.clone();
        assert_eq!(rational_reconstruct(&a2, &m), None);
    }
}
