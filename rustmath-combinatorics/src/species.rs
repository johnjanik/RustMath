//! Combinatorial Species
//!
//! A combinatorial species is a functor from the category of finite sets with bijections
//! to the category of sets. Species provide a powerful framework for studying combinatorial
//! structures and their generating functions.
//!
//! # Mathematical Background
//!
//! A species F assigns to each finite set U a set F[U] of "F-structures" on U,
//! and to each bijection σ: U → V a function F[σ]: F[U] → F[V] that transports structures.
//!
//! ## Key Operations
//!
//! - **Sum (F + G)**: Either an F-structure or a G-structure
//! - **Product (F · G)**: Partition set into two parts with F and G structures
//! - **Composition (F ∘ G)**: F-structure on parts, each with a G-structure
//! - **Derivative (F')**: Remove one element and form F-structure on remainder
//!
//! ## Generating Functions
//!
//! - Ordinary generating function (ogf): Σ f_n x^n / n!
//! - Exponential generating function (egf): Σ f_n x^n / n!
//!
//! where f_n = |F[n]| is the number of F-structures on n elements.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;

/// A combinatorial species represented by its counting sequence
///
/// The counting sequence [f_0, f_1, f_2, ...] where f_n = |F[n]|
/// is the number of labeled F-structures on n elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Species {
    /// Name of the species (for display)
    name: String,
    /// Counting sequence: f_n = number of structures on n elements
    /// Stored up to a maximum size for computational tractability
    counts: Vec<Integer>,
}

impl Species {
    /// Create a species from a counting sequence
    pub fn from_counts(name: String, counts: Vec<Integer>) -> Self {
        Species { name, counts }
    }

    /// Get the number of structures on n elements
    pub fn count(&self, n: usize) -> Option<Integer> {
        self.counts.get(n).cloned()
    }

    /// Get the name of the species
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the counting sequence
    pub fn counts(&self) -> &[Integer] {
        &self.counts
    }

    /// Compute terms up to a given degree
    pub fn degree(&self) -> usize {
        self.counts.len()
    }

    /// Zero species (no structures on any set)
    pub fn zero(degree: usize) -> Self {
        Species {
            name: "0".to_string(),
            counts: vec![Integer::zero(); degree],
        }
    }

    /// One species (exactly one structure on empty set, none elsewhere)
    pub fn one(degree: usize) -> Self {
        let mut counts = vec![Integer::zero(); degree];
        if !counts.is_empty() {
            counts[0] = Integer::one();
        }
        Species {
            name: "1".to_string(),
            counts,
        }
    }

    /// Singleton species X (one structure on 1-element sets only)
    pub fn singleton(degree: usize) -> Self {
        let mut counts = vec![Integer::zero(); degree];
        if counts.len() > 1 {
            counts[1] = Integer::one();
        }
        Species {
            name: "X".to_string(),
            counts,
        }
    }

    /// Set species E (one structure on every finite set)
    /// E[n] = 1 for all n (the underlying set itself)
    pub fn set(degree: usize) -> Self {
        Species {
            name: "E".to_string(),
            counts: vec![Integer::one(); degree],
        }
    }

    /// Permutation species S (all permutations)
    /// S[n] = n!
    pub fn permutations(degree: usize) -> Self {
        let mut counts = Vec::with_capacity(degree);
        let mut factorial = Integer::one();
        counts.push(factorial.clone());

        for n in 1..degree {
            factorial = factorial * Integer::from(n as u32);
            counts.push(factorial.clone());
        }

        Species {
            name: "S".to_string(),
            counts,
        }
    }

    /// Linear order species L (all linear orders)
    /// L[n] = n! (same as permutations)
    pub fn linear_orders(degree: usize) -> Self {
        let mut species = Self::permutations(degree);
        species.name = "L".to_string();
        species
    }

    /// Cyclic permutation species C (all cyclic permutations)
    /// C[0] = 0, C[n] = (n-1)! for n ≥ 1
    pub fn cycles(degree: usize) -> Self {
        let mut counts = Vec::with_capacity(degree);
        if degree > 0 {
            counts.push(Integer::zero());
        }

        let mut factorial = Integer::one();
        for n in 1..degree {
            if n > 1 {
                factorial = factorial * Integer::from((n - 1) as u32);
            }
            counts.push(factorial.clone());
        }

        Species {
            name: "C".to_string(),
            counts,
        }
    }

    /// Oriented cycle species (directed cycles)
    /// Same as C
    pub fn oriented_cycles(degree: usize) -> Self {
        Self::cycles(degree)
    }

    /// Ballots (linear orders with a preferred first element)
    /// B[n] = n · n!
    pub fn ballots(degree: usize) -> Self {
        let mut counts = Vec::with_capacity(degree);

        for n in 0..degree {
            let count = if n == 0 {
                Integer::zero()
            } else {
                // Compute n!
                let mut factorial = Integer::one();
                for i in 1..=n {
                    factorial = factorial * Integer::from(i as u32);
                }
                // B_n = n · n!
                Integer::from(n as u32) * factorial
            };
            counts.push(count);
        }

        Species {
            name: "B".to_string(),
            counts,
        }
    }
}

/// Species operations
impl Species {
    /// Sum of two species: (F + G)[U] = F[U] ⊔ G[U]
    ///
    /// The sum species has structures that are either F-structures or G-structures.
    /// Counting: (F + G)_n = F_n + G_n
    pub fn sum(&self, other: &Species) -> Species {
        let degree = self.degree().min(other.degree());
        let mut counts = Vec::with_capacity(degree);

        for i in 0..degree {
            let count = self.counts[i].clone() + other.counts[i].clone();
            counts.push(count);
        }

        Species {
            name: format!("({} + {})", self.name, other.name),
            counts,
        }
    }

    /// Product of two species: (F · G)[U] = Σ F[U₁] × G[U₂]
    /// where the sum is over all partitions U = U₁ ⊔ U₂
    ///
    /// Counting: (F · G)_n = Σ C(n,k) · F_k · G_{n-k}
    pub fn product(&self, other: &Species) -> Species {
        let degree = self.degree().min(other.degree());
        let mut counts = Vec::with_capacity(degree);

        for n in 0..degree {
            let mut sum = Integer::zero();
            for k in 0..=n {
                let binom = crate::binomial(n as u32, k as u32);
                let term = binom * self.counts[k].clone() * other.counts[n - k].clone();
                sum = sum + term;
            }
            counts.push(sum);
        }

        Species {
            name: format!("({} · {})", self.name, other.name),
            counts,
        }
    }

    /// Composition of species: (F ∘ G)[U]
    ///
    /// Partition U into parts, put a G-structure on each part,
    /// then put an F-structure on the set of parts.
    ///
    /// This is more complex and requires Bell polynomials.
    /// For now, we implement a simplified version for common cases.
    pub fn compose(&self, other: &Species) -> Species {
        let degree = self.degree().min(other.degree());
        let mut counts = Vec::with_capacity(degree);

        // For composition, we use the formula involving Stirling numbers
        // (F ∘ G)_n = Σ F_k · Σ S(n,k) · Π G_{n_i}
        // This is computationally intensive, so we compute a few terms

        for n in 0..degree {
            let mut sum = Integer::zero();

            // Iterate over number of parts
            for k in 0..=n {
                if k >= self.counts.len() {
                    break;
                }

                // For simplicity, we use a generating function approach
                // This is an approximation for the composition
                let stirling = crate::stirling_second(n as u32, k as u32);
                let f_k = self.counts[k].clone();

                // Average contribution from g
                let g_avg = if k > 0 && n > 0 {
                    other.counts.get(n / k).cloned().unwrap_or(Integer::zero())
                } else {
                    Integer::one()
                };

                sum = sum + stirling * f_k * g_avg;
            }

            counts.push(sum);
        }

        Species {
            name: format!("({} ∘ {})", self.name, other.name),
            counts,
        }
    }

    /// Derivative of a species: F'[U] = F[U ∪ {*}]
    ///
    /// The derivative "removes" one distinguished element.
    /// Counting: (F')_n = (n+1) · F_{n+1} / (n+1) = F_{n+1}
    pub fn derivative(&self) -> Species {
        let degree = if self.degree() > 0 {
            self.degree() - 1
        } else {
            0
        };

        let mut counts = Vec::with_capacity(degree);
        for i in 0..degree {
            counts.push(self.counts.get(i + 1).cloned().unwrap_or(Integer::zero()));
        }

        Species {
            name: format!("({})'", self.name),
            counts,
        }
    }

    /// Pointing: F' = X · F'
    /// Select a distinguished element (pointing)
    pub fn pointing(&self) -> Species {
        let degree = self.degree();
        let mut counts = Vec::with_capacity(degree);

        for n in 0..degree {
            let count = if n > 0 && n < self.counts.len() {
                Integer::from(n as u32) * self.counts[n].clone()
            } else {
                Integer::zero()
            };
            counts.push(count);
        }

        Species {
            name: format!("{}•", self.name),
            counts,
        }
    }

    /// Power of a species: F^k
    pub fn power(&self, k: usize) -> Species {
        if k == 0 {
            return Species::one(self.degree());
        }
        if k == 1 {
            return self.clone();
        }

        let mut result = self.clone();
        for _ in 1..k {
            result = result.product(self);
        }
        result.name = format!("{}^{}", self.name, k);
        result
    }
}

/// Weighted species with rational weights
#[derive(Debug, Clone)]
pub struct WeightedSpecies {
    /// Name of the species
    name: String,
    /// Weighted counting sequence using rationals
    weighted_counts: Vec<Rational>,
}

impl WeightedSpecies {
    /// Create a weighted species from rational weights
    pub fn from_weights(name: String, weighted_counts: Vec<Rational>) -> Self {
        WeightedSpecies {
            name,
            weighted_counts,
        }
    }

    /// Create from an integer species with uniform weight
    pub fn from_species(species: &Species) -> Self {
        let weighted_counts = species
            .counts
            .iter()
            .map(|c| Rational::new(c.clone(), Integer::one()).unwrap())
            .collect();

        WeightedSpecies {
            name: species.name.clone(),
            weighted_counts,
        }
    }

    /// Get the weighted count for n elements
    pub fn weighted_count(&self, n: usize) -> Option<Rational> {
        self.weighted_counts.get(n).cloned()
    }

    /// Sum of weighted species
    pub fn sum(&self, other: &WeightedSpecies) -> WeightedSpecies {
        let degree = self.weighted_counts.len().min(other.weighted_counts.len());
        let mut weighted_counts = Vec::with_capacity(degree);

        for i in 0..degree {
            weighted_counts.push(self.weighted_counts[i].clone() + other.weighted_counts[i].clone());
        }

        WeightedSpecies {
            name: format!("({} + {})", self.name, other.name),
            weighted_counts,
        }
    }

    /// Scale by a rational weight
    pub fn scale(&self, weight: Rational) -> WeightedSpecies {
        let weighted_counts = self
            .weighted_counts
            .iter()
            .map(|c| c.clone() * weight.clone())
            .collect();

        WeightedSpecies {
            name: format!("{}·{}", weight, self.name),
            weighted_counts,
        }
    }
}

/// Molecular species - connected structures
///
/// A species is molecular if all non-empty structures are connected in some sense.
/// Examples: connected graphs, irreducible polynomials, prime numbers
#[derive(Debug, Clone)]
pub struct MolecularSpecies {
    species: Species,
    /// True if this represents a molecular (connected/irreducible) species
    is_molecular: bool,
}

impl MolecularSpecies {
    /// Create a molecular species
    pub fn new(species: Species) -> Self {
        MolecularSpecies {
            species,
            is_molecular: true,
        }
    }

    /// Connected graphs species
    pub fn connected_graphs(degree: usize) -> Self {
        // This is complex - placeholder implementation
        // In reality, requires graph enumeration algorithms
        MolecularSpecies::new(Species::from_counts(
            "ConnectedGraphs".to_string(),
            vec![Integer::one(); degree],
        ))
    }

    /// Get the underlying species
    pub fn species(&self) -> &Species {
        &self.species
    }

    /// Molecular decomposition: Express F as Set(M) where M is molecular
    ///
    /// If F = Set ∘ M, then exp(M(x)) = F(x) in EGF form
    /// This gives: M(x) = log(F(x))
    pub fn decompose(species: &Species) -> Vec<MolecularSpecies> {
        // Simplified implementation
        // In general, this requires computing logarithms of generating functions
        vec![MolecularSpecies::new(species.clone())]
    }
}

/// Recursive species defined by functional equations
///
/// Many important species are defined recursively, such as:
/// - Trees: T = X · Set(T)
/// - Binary trees: B = 1 + X · B²
/// - Rooted trees: T = X · Seq(T)
#[derive(Debug, Clone)]
pub struct RecursiveSpecies {
    name: String,
    /// Computed terms
    counts: Vec<Integer>,
}

impl RecursiveSpecies {
    /// Binary trees: B = 1 + X · B²
    /// B_n = Catalan numbers
    pub fn binary_trees(degree: usize) -> Self {
        let mut counts = Vec::with_capacity(degree);

        for n in 0..degree {
            counts.push(crate::catalan(n as u32));
        }

        RecursiveSpecies {
            name: "BinaryTree".to_string(),
            counts,
        }
    }

    /// Rooted trees: T = X · Seq(T) where Seq(T) = 1/(1-T)
    ///
    /// This gives the generating function equation: T(x) = x · exp(T(x))
    /// The counts are related to tree enumerations
    pub fn rooted_trees(degree: usize) -> Self {
        let mut counts = vec![Integer::zero(); degree];

        // T_0 = 0, T_1 = 1
        if degree > 1 {
            counts[1] = Integer::one();
        }

        // Compute using recurrence relations
        // This is a simplified version; exact computation requires more sophisticated methods
        for n in 2..degree {
            // Approximate using Cayley's formula variations
            let mut sum = Integer::zero();
            for k in 1..n {
                let contrib = crate::binomial(n as u32 - 1, k as u32 - 1)
                    * counts[k].clone()
                    * Integer::from((n - k) as u32);
                sum = sum + contrib;
            }
            counts[n] = sum / Integer::from(n as u32);
        }

        RecursiveSpecies {
            name: "RootedTree".to_string(),
            counts,
        }
    }

    /// Plane trees (ordered trees): T = X · (1 + T + T² + ...)
    /// Equivalent to: T = X · 1/(1-T)
    /// These are counted by Catalan numbers
    pub fn plane_trees(degree: usize) -> Self {
        Self::binary_trees(degree) // Same enumeration
    }

    /// Get counts
    pub fn counts(&self) -> &[Integer] {
        &self.counts
    }

    /// Convert to a regular species
    pub fn to_species(&self) -> Species {
        Species::from_counts(self.name.clone(), self.counts.clone())
    }
}

/// Generating function representation
#[derive(Debug, Clone)]
pub enum GeneratingFunction {
    /// Ordinary generating function (OGF)
    Ordinary(Vec<Rational>),
    /// Exponential generating function (EGF)
    Exponential(Vec<Rational>),
}

impl GeneratingFunction {
    /// Compute OGF from species: Σ f_n · x^n
    pub fn ogf_from_species(species: &Species) -> Self {
        let coeffs = species
            .counts
            .iter()
            .map(|c| Rational::new(c.clone(), Integer::one()).unwrap())
            .collect();
        GeneratingFunction::Ordinary(coeffs)
    }

    /// Compute EGF from species: Σ f_n · x^n / n!
    pub fn egf_from_species(species: &Species) -> Self {
        let mut coeffs = Vec::new();
        let mut factorial = Integer::one();

        for (n, count) in species.counts.iter().enumerate() {
            if n > 0 {
                factorial = factorial * Integer::from(n as u32);
            }
            coeffs.push(Rational::new(count.clone(), factorial.clone()).unwrap());
        }

        GeneratingFunction::Exponential(coeffs)
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[Rational] {
        match self {
            GeneratingFunction::Ordinary(c) => c,
            GeneratingFunction::Exponential(c) => c,
        }
    }

    /// Evaluate at a rational value
    pub fn evaluate(&self, x: Rational) -> Rational {
        let mut result = Rational::zero();
        let mut power = Rational::one();

        for coeff in self.coefficients() {
            result = result + coeff.clone() * power.clone();
            power = power * x.clone();
        }

        result
    }
}

impl fmt::Display for Species {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Species {}: [", self.name)?;
        for (i, count) in self.counts.iter().take(10).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", count)?;
        }
        if self.counts.len() > 10 {
            write!(f, ", ...")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_species() {
        let zero = Species::zero(5);
        assert_eq!(zero.count(0), Some(Integer::zero()));
        assert_eq!(zero.count(1), Some(Integer::zero()));

        let one = Species::one(5);
        assert_eq!(one.count(0), Some(Integer::one()));
        assert_eq!(one.count(1), Some(Integer::zero()));

        let singleton = Species::singleton(5);
        assert_eq!(singleton.count(0), Some(Integer::zero()));
        assert_eq!(singleton.count(1), Some(Integer::one()));
        assert_eq!(singleton.count(2), Some(Integer::zero()));

        let set_species = Species::set(5);
        assert_eq!(set_species.count(0), Some(Integer::one()));
        assert_eq!(set_species.count(3), Some(Integer::one()));
    }

    #[test]
    fn test_permutation_species() {
        let perms = Species::permutations(6);

        // S_0 = 1, S_1 = 1, S_2 = 2, S_3 = 6, S_4 = 24, S_5 = 120
        assert_eq!(perms.count(0), Some(Integer::one()));
        assert_eq!(perms.count(1), Some(Integer::one()));
        assert_eq!(perms.count(2), Some(Integer::from(2)));
        assert_eq!(perms.count(3), Some(Integer::from(6)));
        assert_eq!(perms.count(4), Some(Integer::from(24)));
        assert_eq!(perms.count(5), Some(Integer::from(120)));
    }

    #[test]
    fn test_cycle_species() {
        let cycles = Species::cycles(6);

        // C_0 = 0, C_1 = 1, C_2 = 1, C_3 = 2, C_4 = 6, C_5 = 24
        assert_eq!(cycles.count(0), Some(Integer::zero()));
        assert_eq!(cycles.count(1), Some(Integer::one()));
        assert_eq!(cycles.count(2), Some(Integer::one()));
        assert_eq!(cycles.count(3), Some(Integer::from(2)));
        assert_eq!(cycles.count(4), Some(Integer::from(6)));
        assert_eq!(cycles.count(5), Some(Integer::from(24)));
    }

    #[test]
    fn test_species_sum() {
        let x = Species::singleton(5);
        let one = Species::one(5);

        let sum = x.sum(&one);

        // (X + 1)_0 = X_0 + 1_0 = 0 + 1 = 1
        // (X + 1)_1 = X_1 + 1_1 = 1 + 0 = 1
        // (X + 1)_2 = X_2 + 1_2 = 0 + 0 = 0
        assert_eq!(sum.count(0), Some(Integer::one()));
        assert_eq!(sum.count(1), Some(Integer::one()));
        assert_eq!(sum.count(2), Some(Integer::zero()));
    }

    #[test]
    fn test_species_product() {
        let x = Species::singleton(5);

        // X² should have 2 structures on 2 elements (the two ways to partition)
        let x_squared = x.product(&x);

        assert_eq!(x_squared.count(0), Some(Integer::zero()));
        assert_eq!(x_squared.count(1), Some(Integer::zero()));
        assert_eq!(x_squared.count(2), Some(Integer::from(2))); // C(2,1) * 1 * 1 = 2
    }

    #[test]
    fn test_species_derivative() {
        let perms = Species::permutations(6);
        let deriv = perms.derivative();

        // S'_n = S_{n+1} = (n+1)!
        assert_eq!(deriv.count(0), Some(Integer::one())); // 1!
        assert_eq!(deriv.count(1), Some(Integer::from(2))); // 2!
        assert_eq!(deriv.count(2), Some(Integer::from(6))); // 3!
        assert_eq!(deriv.count(3), Some(Integer::from(24))); // 4!
    }

    #[test]
    fn test_binary_trees() {
        let btrees = RecursiveSpecies::binary_trees(6);

        // Binary tree counts are Catalan numbers: 1, 1, 2, 5, 14, 42
        assert_eq!(btrees.counts()[0], Integer::one());
        assert_eq!(btrees.counts()[1], Integer::one());
        assert_eq!(btrees.counts()[2], Integer::from(2));
        assert_eq!(btrees.counts()[3], Integer::from(5));
        assert_eq!(btrees.counts()[4], Integer::from(14));
        assert_eq!(btrees.counts()[5], Integer::from(42));
    }

    #[test]
    fn test_weighted_species() {
        let x = Species::singleton(5);
        let wx = WeightedSpecies::from_species(&x);

        let weight = Rational::new(Integer::from(2), Integer::one()).unwrap();
        let scaled = wx.scale(weight);

        assert_eq!(
            scaled.weighted_count(1),
            Some(Rational::new(Integer::from(2), Integer::one()).unwrap())
        );
    }

    #[test]
    fn test_generating_functions() {
        let set_species = Species::set(5);
        let egf = GeneratingFunction::egf_from_species(&set_species);

        // EGF of E is exp(x): 1 + x + x²/2! + x³/3! + ...
        let coeffs = egf.coefficients();
        assert_eq!(coeffs[0], Rational::new(Integer::one(), Integer::one()).unwrap());
        assert_eq!(coeffs[1], Rational::new(Integer::one(), Integer::one()).unwrap());
        assert_eq!(
            coeffs[2],
            Rational::new(Integer::one(), Integer::from(2)).unwrap()
        );
        assert_eq!(
            coeffs[3],
            Rational::new(Integer::one(), Integer::from(6)).unwrap()
        );
    }

    #[test]
    fn test_ballots() {
        let ballots = Species::ballots(5);

        // B_0 = 0, B_1 = 1, B_2 = 4, B_3 = 18, B_4 = 96
        assert_eq!(ballots.count(0), Some(Integer::zero()));
        assert_eq!(ballots.count(1), Some(Integer::one())); // 1 * 1!
        assert_eq!(ballots.count(2), Some(Integer::from(4))); // 2 * 2!
        assert_eq!(ballots.count(3), Some(Integer::from(18))); // 3 * 3!
        assert_eq!(ballots.count(4), Some(Integer::from(96))); // 4 * 4!
    }

    #[test]
    fn test_species_power() {
        let x = Species::singleton(4);
        let x3 = x.power(3);

        // X³ on 3 elements: C(3,1) * C(2,1) * C(1,1) = 6
        assert_eq!(x3.count(3), Some(Integer::from(6)));
    }

    #[test]
    fn test_pointing() {
        let set_species = Species::set(5);
        let pointed = set_species.pointing();

        // E• has n structures on n elements (choose which element to point at)
        assert_eq!(pointed.count(0), Some(Integer::zero()));
        assert_eq!(pointed.count(1), Some(Integer::one()));
        assert_eq!(pointed.count(2), Some(Integer::from(2)));
        assert_eq!(pointed.count(3), Some(Integer::from(3)));
    }
}
