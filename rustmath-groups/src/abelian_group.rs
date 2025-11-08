//! Abelian groups - groups where all elements commute
//!
//! This module implements finite abelian groups and their structure theory

use std::collections::HashMap;
use std::fmt;

/// A finitely generated abelian group
///
/// By the fundamental theorem, every finitely generated abelian group
/// is isomorphic to Z^r × Z/n₁Z × Z/n₂Z × ... × Z/nₖZ
/// where n₁ | n₂ | ... | nₖ (invariant factors)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbelianGroup {
    /// Free rank (number of Z factors)
    free_rank: usize,
    /// Invariant factors (torsion part)
    invariant_factors: Vec<usize>,
}

impl AbelianGroup {
    /// Create a new abelian group from invariant factors
    ///
    /// # Arguments
    /// * `free_rank` - Number of free Z factors
    /// * `invariant_factors` - Torsion part, must satisfy n₁ | n₂ | ... | nₖ
    pub fn new(free_rank: usize, invariant_factors: Vec<usize>) -> Result<Self, String> {
        // Validate divisibility condition
        for i in 1..invariant_factors.len() {
            if invariant_factors[i] % invariant_factors[i - 1] != 0 {
                return Err(format!(
                    "Invariant factors must be divisible: {} does not divide {}",
                    invariant_factors[i - 1],
                    invariant_factors[i]
                ));
            }
        }

        Ok(AbelianGroup {
            free_rank,
            invariant_factors,
        })
    }

    /// Create a cyclic group Z/nZ
    pub fn cyclic(n: usize) -> Self {
        if n == 0 {
            // Z (infinite cyclic)
            AbelianGroup {
                free_rank: 1,
                invariant_factors: vec![],
            }
        } else {
            AbelianGroup {
                free_rank: 0,
                invariant_factors: vec![n],
            }
        }
    }

    /// Create a free abelian group Z^n
    pub fn free(rank: usize) -> Self {
        AbelianGroup {
            free_rank: rank,
            invariant_factors: vec![],
        }
    }

    /// Create from elementary divisors (prime power decomposition)
    ///
    /// Elementary divisors are the prime powers p^k that appear
    pub fn from_elementary_divisors(free_rank: usize, elementary: Vec<usize>) -> Self {
        // Convert elementary divisors to invariant factors
        let invariant_factors = elementary_to_invariant(&elementary);

        AbelianGroup {
            free_rank,
            invariant_factors,
        }
    }

    /// Get the free rank
    pub fn free_rank(&self) -> usize {
        self.free_rank
    }

    /// Get the invariant factors
    pub fn invariant_factors(&self) -> &[usize] {
        &self.invariant_factors
    }

    /// Get the torsion rank (number of invariant factors)
    pub fn torsion_rank(&self) -> usize {
        self.invariant_factors.len()
    }

    /// Get the total rank (free + torsion)
    pub fn rank(&self) -> usize {
        self.free_rank + self.torsion_rank()
    }

    /// Get the order of the torsion subgroup
    ///
    /// Returns None if infinite, Some(n) if finite
    pub fn torsion_order(&self) -> Option<usize> {
        if self.invariant_factors.is_empty() {
            return Some(1);
        }

        let mut order = 1usize;
        for &n in &self.invariant_factors {
            order = order.checked_mul(n)?;
        }

        Some(order)
    }

    /// Check if the group is finite
    pub fn is_finite(&self) -> bool {
        self.free_rank == 0
    }

    /// Check if the group is torsion-free
    pub fn is_torsion_free(&self) -> bool {
        self.invariant_factors.is_empty()
    }

    /// Check if the group is cyclic
    pub fn is_cyclic(&self) -> bool {
        self.free_rank + self.invariant_factors.len() <= 1
    }

    /// Get the elementary divisors
    pub fn elementary_divisors(&self) -> Vec<usize> {
        invariant_to_elementary(&self.invariant_factors)
    }

    /// Compute the direct sum with another abelian group
    pub fn direct_sum(&self, other: &AbelianGroup) -> AbelianGroup {
        let mut factors = self.invariant_factors.clone();
        factors.extend(other.invariant_factors.clone());

        // Sort and combine to get new invariant factors
        let elementary = invariant_to_elementary(&factors);
        AbelianGroup::from_elementary_divisors(
            self.free_rank + other.free_rank,
            elementary,
        )
    }

    /// Get a string representation following the structure theorem
    pub fn structure_string(&self) -> String {
        let mut parts = Vec::new();

        // Add free part
        if self.free_rank > 0 {
            if self.free_rank == 1 {
                parts.push("Z".to_string());
            } else {
                parts.push(format!("Z^{}", self.free_rank));
            }
        }

        // Add torsion part
        for &n in &self.invariant_factors {
            parts.push(format!("Z/{}", n));
        }

        if parts.is_empty() {
            "{0}".to_string()
        } else {
            parts.join(" × ")
        }
    }
}

impl fmt::Display for AbelianGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.structure_string())
    }
}

/// Convert elementary divisors to invariant factors
fn elementary_to_invariant(elementary: &[usize]) -> Vec<usize> {
    if elementary.is_empty() {
        return vec![];
    }

    // Group by prime
    let mut prime_powers: HashMap<usize, Vec<usize>> = HashMap::new();

    for &n in elementary {
        if n <= 1 {
            continue;
        }

        // Factor into prime powers
        let factors = prime_factorization(n);
        for (p, k) in factors {
            prime_powers
                .entry(p)
                .or_insert_with(Vec::new)
                .push(p.pow(k as u32));
        }
    }

    // Sort powers for each prime
    for powers in prime_powers.values_mut() {
        powers.sort_unstable();
    }

    // Compute invariant factors
    let max_len = prime_powers.values().map(|v| v.len()).max().unwrap_or(0);
    let mut invariants = vec![1; max_len];

    for powers in prime_powers.values() {
        for (i, &p_power) in powers.iter().enumerate() {
            invariants[i] *= p_power;
        }
    }

    invariants.retain(|&n| n > 1);
    invariants
}

/// Convert invariant factors to elementary divisors
fn invariant_to_elementary(invariant: &[usize]) -> Vec<usize> {
    let mut elementary = Vec::new();

    for &n in invariant {
        if n <= 1 {
            continue;
        }

        let factors = prime_factorization(n);
        for (p, k) in factors {
            elementary.push(p.pow(k as u32));
        }
    }

    elementary.sort_unstable();
    elementary
}

/// Simple prime factorization
fn prime_factorization(mut n: usize) -> Vec<(usize, usize)> {
    let mut factors = Vec::new();
    let mut d = 2;

    while d * d <= n {
        let mut exp = 0;
        while n % d == 0 {
            n /= d;
            exp += 1;
        }
        if exp > 0 {
            factors.push((d, exp));
        }
        d += 1;
    }

    if n > 1 {
        factors.push((n, 1));
    }

    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyclic_group() {
        let z5 = AbelianGroup::cyclic(5);
        assert_eq!(z5.free_rank(), 0);
        assert_eq!(z5.invariant_factors(), &[5]);
        assert!(z5.is_finite());
        assert!(z5.is_cyclic());
    }

    #[test]
    fn test_free_group() {
        let z3 = AbelianGroup::free(3);
        assert_eq!(z3.free_rank(), 3);
        assert_eq!(z3.torsion_rank(), 0);
        assert!(!z3.is_finite());
        assert!(z3.is_torsion_free());
    }

    #[test]
    fn test_structure_string() {
        let g1 = AbelianGroup::cyclic(6);
        assert_eq!(g1.structure_string(), "Z/6");

        let g2 = AbelianGroup::free(2);
        assert_eq!(g2.structure_string(), "Z^2");

        let g3 = AbelianGroup::new(1, vec![2, 4]).unwrap();
        assert_eq!(g3.structure_string(), "Z × Z/2 × Z/4");
    }

    #[test]
    fn test_torsion_order() {
        let z6 = AbelianGroup::cyclic(6);
        assert_eq!(z6.torsion_order(), Some(6));

        let z2_z3 = AbelianGroup::new(0, vec![2, 6]).unwrap();
        assert_eq!(z2_z3.torsion_order(), Some(12));

        let z_free = AbelianGroup::free(1);
        assert_eq!(z_free.torsion_order(), Some(1));
    }

    #[test]
    fn test_direct_sum() {
        let z2 = AbelianGroup::cyclic(2);
        let z3 = AbelianGroup::cyclic(3);

        let sum = z2.direct_sum(&z3);
        // Z/2 ⊕ Z/3 ≅ Z/6 (since gcd(2,3)=1)
        assert_eq!(sum.torsion_order(), Some(6));
    }

    #[test]
    fn test_elementary_divisors() {
        let g = AbelianGroup::new(0, vec![6, 12]).unwrap();
        let elem = g.elementary_divisors();

        // 6 = 2·3, 12 = 4·3
        // Elementary divisors should be [2, 3, 4, 3] = [2, 3, 3, 4] sorted
        assert!(elem.contains(&2));
        assert!(elem.contains(&4));
        assert_eq!(elem.iter().filter(|&&x| x == 3).count(), 2);
    }

    #[test]
    fn test_is_cyclic() {
        let z6 = AbelianGroup::cyclic(6);
        assert!(z6.is_cyclic());

        let z2_z2 = AbelianGroup::new(0, vec![2, 2]).unwrap();
        assert!(!z2_z2.is_cyclic());

        let z = AbelianGroup::free(1);
        assert!(z.is_cyclic());
    }

    #[test]
    fn test_elementary_divisors_conversion() {
        // Test that elementary_divisors() gives back valid results
        let g = AbelianGroup::new(0, vec![6, 12]).unwrap();
        let elem = g.elementary_divisors();

        // Should contain prime powers
        assert!(!elem.is_empty());
        // All should be prime powers
        for &e in &elem {
            assert!(e > 1);
        }
    }

    #[test]
    fn test_rank() {
        let g = AbelianGroup::new(2, vec![3, 9]).unwrap();
        assert_eq!(g.free_rank(), 2);
        assert_eq!(g.torsion_rank(), 2);
        assert_eq!(g.rank(), 4);
    }
}
