//! Free Groups
//!
//! A free group F_n is the group with n generators and no relations other than
//! the group axioms. Elements are represented as reduced words in the generators
//! and their inverses.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::FreeGroup;
//!
//! // Create free group on 2 generators
//! let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
//!
//! // Create elements
//! let a = f2.generator(0).unwrap();
//! let b = f2.generator(1).unwrap();
//!
//! // Compute ab * a^-1 = b
//! let ab = a.mul(&b);
//! let a_inv = a.inverse();
//! let result = ab.mul(&a_inv);
//!
//! assert_eq!(result, b);
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};

/// A free group on n generators
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreeGroup {
    /// Number of generators
    rank: usize,
    /// Names of generators
    generator_names: Vec<String>,
}

impl FreeGroup {
    /// Create a new free group on n generators
    ///
    /// # Arguments
    /// * `rank` - Number of generators
    /// * `names` - Names for the generators (must have length equal to rank)
    ///
    /// # Examples
    /// ```
    /// use rustmath_groups::FreeGroup;
    ///
    /// let f2 = FreeGroup::new(2, vec!["x".to_string(), "y".to_string()]);
    /// assert_eq!(f2.rank(), 2);
    /// ```
    pub fn new(rank: usize, names: Vec<String>) -> Self {
        assert_eq!(rank, names.len(), "Number of names must equal rank");
        FreeGroup {
            rank,
            generator_names: names,
        }
    }

    /// Create a free group with default generator names (x_0, x_1, ...)
    pub fn with_default_names(rank: usize) -> Self {
        let names: Vec<String> = (0..rank).map(|i| format!("x_{}", i)).collect();
        FreeGroup {
            rank,
            generator_names: names,
        }
    }

    /// Get the rank (number of generators) of the free group
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the generator names
    pub fn generator_names(&self) -> &[String] {
        &self.generator_names
    }

    /// Get the i-th generator
    pub fn generator(&self, i: usize) -> Option<FreeGroupElement> {
        if i >= self.rank {
            return None;
        }
        Some(FreeGroupElement {
            group: self.clone(),
            word: vec![(i as isize, 1)],
        })
    }

    /// Get the identity element
    pub fn identity(&self) -> FreeGroupElement {
        FreeGroupElement {
            group: self.clone(),
            word: vec![],
        }
    }

    /// Create an element from a Tietze representation
    ///
    /// Tietze representation uses integers: 1, 2, ... for generators
    /// and -1, -2, ... for their inverses
    pub fn from_tietze(&self, tietze: Vec<isize>) -> FreeGroupElement {
        let mut word = Vec::new();

        for &t in &tietze {
            if t == 0 {
                continue;
            }

            let gen = if t > 0 {
                (t - 1) as isize
            } else {
                (-t - 1) as isize
            };

            let exp = if t > 0 { 1 } else { -1 };

            if gen < 0 || gen >= self.rank as isize {
                continue; // Skip invalid generators
            }

            word.push((gen, exp));
        }

        let mut elem = FreeGroupElement {
            group: self.clone(),
            word,
        };
        elem.reduce();
        elem
    }

    /// Check if this is the trivial group (rank 0)
    pub fn is_trivial(&self) -> bool {
        self.rank == 0
    }
}

impl fmt::Display for FreeGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.rank == 0 {
            write!(f, "Free Group on 0 generators")
        } else {
            write!(
                f,
                "Free Group on generators {{{}}}",
                self.generator_names.join(", ")
            )
        }
    }
}

/// An element of a free group, represented as a reduced word
///
/// The word is stored as a list of (generator_index, exponent) pairs.
/// The word is always kept in reduced form (no adjacent pairs with the same generator).
#[derive(Clone, Debug)]
pub struct FreeGroupElement {
    /// The parent free group
    group: FreeGroup,
    /// The word as (generator_index, exponent) pairs
    /// Invariant: No adjacent pairs have the same generator, and no exponent is 0
    word: Vec<(isize, isize)>,
}

impl FreeGroupElement {
    /// Create a generator element with a given index and exponent
    ///
    /// This creates a minimal free group element without needing the full group structure.
    /// Useful for constructing words programmatically.
    ///
    /// # Arguments
    /// * `gen_index` - The generator index (0-based)
    /// * `exponent` - The exponent for the generator
    pub fn generator(gen_index: i32, exponent: isize) -> Self {
        // Create a minimal free group for this purpose
        let group = FreeGroup::with_default_names((gen_index.abs() + 1) as usize);

        if exponent == 0 {
            return FreeGroupElement {
                group,
                word: vec![],
            };
        }

        FreeGroupElement {
            group,
            word: vec![(gen_index as isize, exponent)],
        }
    }

    /// Create the identity element of a free group
    ///
    /// Creates a minimal free group with default structure
    pub fn identity() -> Self {
        let group = FreeGroup::with_default_names(1);
        FreeGroupElement {
            group,
            word: vec![],
        }
    }

    /// Alias for identity() to match common usage
    pub fn zero() -> Self {
        Self::identity()
    }

    /// Get the parent free group
    pub fn group(&self) -> &FreeGroup {
        &self.group
    }

    /// Get the word representation
    pub fn word(&self) -> &[(isize, isize)] {
        &self.word
    }

    /// Get the Tietze representation
    ///
    /// Returns a list of integers where positive integers represent generators
    /// and negative integers represent their inverses
    pub fn tietze(&self) -> Vec<isize> {
        let mut result = Vec::new();

        for &(gen, exp) in &self.word {
            let base = if exp > 0 {
                gen + 1
            } else {
                -(gen + 1)
            };

            for _ in 0..exp.abs() {
                result.push(base);
            }
        }

        result
    }

    /// Get the syllable decomposition
    ///
    /// Returns the word as (generator_index, exponent) pairs
    pub fn syllables(&self) -> Vec<(usize, isize)> {
        self.word.iter().map(|&(g, e)| (g as usize, e)).collect()
    }

    /// Get the length of the word (number of syllables)
    pub fn length(&self) -> usize {
        self.word.len()
    }

    /// Get the total length (sum of absolute values of exponents)
    pub fn total_length(&self) -> usize {
        self.word.iter().map(|(_, e)| e.abs() as usize).sum()
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.word.is_empty()
    }

    /// Get the unique letters (generator indices) that appear in this element
    ///
    /// Returns a vector of unique generator indices in the order they first appear.
    ///
    /// # Examples
    /// ```
    /// use rustmath_groups::FreeGroup;
    ///
    /// let f3 = FreeGroup::with_default_names(3);
    /// let x0 = f3.generator(0).unwrap();
    /// let x1 = f3.generator(1).unwrap();
    /// let x2 = f3.generator(2).unwrap();
    ///
    /// // x0 * x1 * x0^2 * x2 has letters [0, 1, 2]
    /// let elem = x0.mul(&x1).mul(&x0.pow(2)).mul(&x2);
    /// let letters = elem.letters();
    /// assert_eq!(letters, vec![0, 1, 2]);
    /// ```
    pub fn letters(&self) -> Vec<usize> {
        let mut seen = Vec::new();
        for &(gen, _) in &self.word {
            let gen_idx = gen as usize;
            if !seen.contains(&gen_idx) {
                seen.push(gen_idx);
            }
        }
        seen
    }

    /// Get the sum of exponents for a given generator
    ///
    /// Returns the sum of all exponents for the specified generator in this element.
    ///
    /// # Examples
    /// ```
    /// use rustmath_groups::FreeGroup;
    ///
    /// let f2 = FreeGroup::with_default_names(2);
    /// let x0 = f2.generator(0).unwrap();
    /// let x1 = f2.generator(1).unwrap();
    ///
    /// // x0^3 * x1 * x0^-1 has exponent sum 2 for generator 0
    /// let elem = x0.pow(3).mul(&x1).mul(&x0.pow(-1));
    /// assert_eq!(elem.exponent_sum(0), 2);
    /// assert_eq!(elem.exponent_sum(1), 1);
    /// ```
    pub fn exponent_sum(&self, generator: usize) -> isize {
        self.word
            .iter()
            .filter(|(gen, _)| *gen == generator as isize)
            .map(|(_, exp)| exp)
            .sum()
    }

    /// Multiply this element by another
    pub fn mul(&self, other: &FreeGroupElement) -> FreeGroupElement {
        // Try to unify groups if they're compatible
        let target_group = if self.group.rank() >= other.group.rank() {
            self.group.clone()
        } else {
            other.group.clone()
        };

        let mut word = self.word.clone();
        word.extend_from_slice(&other.word);

        let mut result = FreeGroupElement {
            group: target_group,
            word,
        };
        result.reduce();
        result
    }

    /// Alias for mul() to match SageMath naming convention
    pub fn multiply(&self, other: &FreeGroupElement) -> FreeGroupElement {
        self.mul(other)
    }

    /// Compute the inverse of this element
    pub fn inverse(&self) -> FreeGroupElement {
        let mut word: Vec<(isize, isize)> = self
            .word
            .iter()
            .rev()
            .map(|&(g, e)| (g, -e))
            .collect();

        FreeGroupElement {
            group: self.group.clone(),
            word,
        }
    }

    /// Raise this element to a power
    pub fn pow(&self, n: isize) -> FreeGroupElement {
        if n == 0 {
            return self.group.identity();
        }

        if n < 0 {
            return self.inverse().pow(-n);
        }

        let mut word = Vec::new();
        for _ in 0..n {
            word.extend_from_slice(&self.word);
        }

        let mut result = FreeGroupElement {
            group: self.group.clone(),
            word,
        };
        result.reduce();
        result
    }

    /// Compute the commutator [self, other] = self * other * self^-1 * other^-1
    pub fn commutator(&self, other: &FreeGroupElement) -> FreeGroupElement {
        self.mul(other)
            .mul(&self.inverse())
            .mul(&other.inverse())
    }

    /// Compute the conjugate by another element: other * self * other^-1
    pub fn conjugate_by(&self, other: &FreeGroupElement) -> FreeGroupElement {
        other.mul(self).mul(&other.inverse())
    }

    /// Reduce the word to canonical form
    ///
    /// This combines adjacent syllables with the same generator and removes
    /// syllables with exponent 0.
    fn reduce(&mut self) {
        if self.word.is_empty() {
            return;
        }

        let mut reduced = Vec::new();
        let mut current_gen = self.word[0].0;
        let mut current_exp = self.word[0].1;

        for &(gen, exp) in &self.word[1..] {
            if gen == current_gen {
                current_exp += exp;
            } else {
                if current_exp != 0 {
                    reduced.push((current_gen, current_exp));
                }
                current_gen = gen;
                current_exp = exp;
            }
        }

        if current_exp != 0 {
            reduced.push((current_gen, current_exp));
        }

        self.word = reduced;
    }

    /// Compute the Fox derivative with respect to a generator
    ///
    /// The Fox derivative is a free differential calculus operation.
    /// Returns a formal sum represented as a vector of group elements.
    pub fn fox_derivative(&self, gen_index: usize) -> Vec<FreeGroupElement> {
        let mut result = Vec::new();

        let mut prefix = self.group.identity();

        for &(g, e) in &self.word {
            if g as usize == gen_index {
                // Add terms for each occurrence of the generator
                if e > 0 {
                    for i in 0..e {
                        let term = prefix.mul(&self.group.generator(gen_index).unwrap().pow(i));
                        result.push(term);
                    }
                } else {
                    let gen_elem = self.group.generator(gen_index).unwrap();
                    for i in 0..(-e) {
                        let term = prefix.mul(&gen_elem.pow(-i - 1));
                        result.push(term.inverse());
                    }
                }
            }

            // Update prefix for next iteration
            prefix = prefix.mul(&FreeGroupElement {
                group: self.group.clone(),
                word: vec![(g, e)],
            });
        }

        result
    }

    /// Evaluate the element by replacing generators with values
    ///
    /// This is useful for homomorphisms from the free group to other groups
    pub fn evaluate<T, F>(&self, mut f: F) -> T
    where
        T: Clone,
        F: FnMut(usize, isize) -> T,
    {
        if self.word.is_empty() {
            return f(0, 0); // Identity - need special handling
        }

        let mut result = f(self.word[0].0 as usize, self.word[0].1);

        for &(gen, exp) in &self.word[1..] {
            // This is a simplified version - actual implementation would need
            // proper multiplication operation for type T
            result = f(gen as usize, exp);
        }

        result
    }
}

impl PartialEq for FreeGroupElement {
    fn eq(&self, other: &Self) -> bool {
        self.group == other.group && self.word == other.word
    }
}

impl Eq for FreeGroupElement {}

impl Hash for FreeGroupElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.word.hash(state);
    }
}

impl fmt::Display for FreeGroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.word.is_empty() {
            return write!(f, "1");
        }

        let mut parts = Vec::new();
        for &(gen, exp) in &self.word {
            let name = &self.group.generator_names[gen as usize];
            if exp == 1 {
                parts.push(name.clone());
            } else if exp == -1 {
                parts.push(format!("{}^-1", name));
            } else {
                parts.push(format!("{}^{}", name, exp));
            }
        }

        write!(f, "{}", parts.join("*"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_group_creation() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(f2.rank(), 2);
        assert_eq!(f2.generator_names(), &["a", "b"]);
        assert!(!f2.is_trivial());
    }

    #[test]
    fn test_default_names() {
        let f3 = FreeGroup::with_default_names(3);
        assert_eq!(f3.rank(), 3);
        assert_eq!(f3.generator_names(), &["x_0", "x_1", "x_2"]);
    }

    #[test]
    fn test_identity() {
        let f2 = FreeGroup::with_default_names(2);
        let e = f2.identity();
        assert!(e.is_identity());
        assert_eq!(e.length(), 0);
    }

    #[test]
    fn test_generator() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();
        let b = f2.generator(1).unwrap();

        assert_eq!(a.length(), 1);
        assert_eq!(b.length(), 1);
        assert!(!a.is_identity());
        assert!(!b.is_identity());
    }

    #[test]
    fn test_multiplication() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();
        let b = f2.generator(1).unwrap();

        let ab = a.mul(&b);
        assert_eq!(ab.length(), 2);
        assert_eq!(ab.word(), &[(0, 1), (1, 1)]);
    }

    #[test]
    fn test_inverse() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();

        let a_inv = a.inverse();
        let prod = a.mul(&a_inv);
        assert!(prod.is_identity());
    }

    #[test]
    fn test_reduction() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();
        let b = f2.generator(1).unwrap();

        // Compute ab * a^-1, should reduce to b
        let ab = a.mul(&b);
        let a_inv = a.inverse();
        let result = ab.mul(&a_inv);

        assert_eq!(result, b);
    }

    #[test]
    fn test_power() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();

        let a3 = a.pow(3);
        assert_eq!(a3.word(), &[(0, 3)]);

        let a_neg2 = a.pow(-2);
        assert_eq!(a_neg2.word(), &[(0, -2)]);

        let a0 = a.pow(0);
        assert!(a0.is_identity());
    }

    #[test]
    fn test_commutator() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();
        let b = f2.generator(1).unwrap();

        let comm = a.commutator(&b);
        // [a, b] = aba^-1b^-1
        assert_eq!(comm.length(), 4);
    }

    #[test]
    fn test_tietze_representation() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();
        let b = f2.generator(1).unwrap();

        let ab = a.mul(&b);
        assert_eq!(ab.tietze(), vec![1, 2]); // a is 1, b is 2

        let a_inv = a.inverse();
        assert_eq!(a_inv.tietze(), vec![-1]); // a^-1 is -1
    }

    #[test]
    fn test_from_tietze() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);

        let elem = f2.from_tietze(vec![1, 2, -1]);
        // This is a * b * a^-1 = b (after reduction)
        let b = f2.generator(1).unwrap();
        assert_eq!(elem, b);
    }

    #[test]
    fn test_syllables() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();

        let a3 = a.pow(3);
        let syllables = a3.syllables();
        assert_eq!(syllables, vec![(0, 3)]);
    }

    #[test]
    fn test_total_length() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();
        let b = f2.generator(1).unwrap();

        let word = a.pow(3).mul(&b.pow(-2));
        assert_eq!(word.total_length(), 5); // |3| + |-2| = 5
        assert_eq!(word.length(), 2); // Two syllables
    }

    #[test]
    fn test_conjugate() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();
        let b = f2.generator(1).unwrap();

        let conjugate = a.conjugate_by(&b);
        // b * a * b^-1
        assert_eq!(conjugate.length(), 3);
    }

    #[test]
    fn test_fox_derivative() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a = f2.generator(0).unwrap();

        // Derivative of a^3 with respect to a should give [1, a, a^2]
        let a3 = a.pow(3);
        let deriv = a3.fox_derivative(0);
        assert_eq!(deriv.len(), 3);
    }

    #[test]
    fn test_display() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(f2.to_string(), "Free Group on generators {a, b}");

        let a = f2.generator(0).unwrap();
        let b = f2.generator(1).unwrap();
        assert_eq!(a.to_string(), "a");

        let ab = a.mul(&b);
        assert_eq!(ab.to_string(), "a*b");

        let a_inv = a.inverse();
        assert_eq!(a_inv.to_string(), "a^-1");
    }

    #[test]
    fn test_equality_and_hash() {
        let f2 = FreeGroup::new(2, vec!["a".to_string(), "b".to_string()]);
        let a1 = f2.generator(0).unwrap();
        let a2 = f2.generator(0).unwrap();

        assert_eq!(a1, a2);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a1.clone());
        assert!(set.contains(&a2));
    }

    #[test]
    fn test_complex_reduction() {
        let f3 = FreeGroup::with_default_names(3);

        // Create x_0 * x_1 * x_0^-1 * x_1^-1 (commutator)
        let x0 = f3.generator(0).unwrap();
        let x1 = f3.generator(1).unwrap();

        let comm = x0.mul(&x1).mul(&x0.inverse()).mul(&x1.inverse());
        // Should not reduce further
        assert_eq!(comm.length(), 4);
    }
}
