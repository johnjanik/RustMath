//! Chinese Remainder Theorem implementation

use crate::Integer;
use rustmath_core::{MathError, Result, Ring};

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
}
