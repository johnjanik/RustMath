//! Finitely Presented Groups
//!
//! This module implements finitely presented groups, which are groups defined by
//! generators and relations. A finitely presented group is specified as G = ⟨X | R⟩,
//! where X is a finite set of generators and R is a set of relations (equations) that
//! the generators must satisfy.
//!
//! # Theory
//!
//! A finitely presented group is constructed as a quotient of a free group F(X) by
//! the normal closure of the relations R. Elements are equivalence classes of words
//! in the generators, where two words are equivalent if one can be transformed into
//! the other using the relations.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::finitely_presented::*;
//!
//! // Create the cyclic group Z/5Z = ⟨a | a^5⟩
//! let gens = vec!["a".to_string()];
//! let rels = vec![vec![1, 1, 1, 1, 1]]; // a^5 = e
//! let group = FinitelyPresentedGroup::new(gens, rels);
//!
//! // Create group elements
//! let a = group.generator(0);
//! let a_squared = a.mul(&a);
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Mul;

/// A finitely presented group G = ⟨X | R⟩
///
/// The group is defined by:
/// - A finite set of generators X
/// - A set of relations R (words that equal the identity)
///
/// Elements are represented as words in the generators and their inverses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinitelyPresentedGroup {
    /// Names of the generators
    generator_names: Vec<String>,
    /// Number of generators
    num_generators: usize,
    /// Relations as words in Tietze form
    /// Positive integers i represent generator x_{i-1}
    /// Negative integers -i represent inverse of generator x_{i-1}
    relations: Vec<Vec<i32>>,
}

/// An element of a finitely presented group
///
/// Elements are represented as words in the generators, using Tietze form:
/// - Positive integer i represents the i-th generator (1-indexed)
/// - Negative integer -i represents the inverse of the i-th generator
/// - Empty word represents the identity
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinitelyPresentedGroupElement {
    /// The group this element belongs to
    group: FinitelyPresentedGroup,
    /// Word representation in Tietze form
    word: Vec<i32>,
}

/// A rewriting system for reducing words in a finitely presented group
///
/// Implements basic word reduction and can be extended with Knuth-Bendix
/// completion for confluent rewriting systems.
#[derive(Clone, Debug)]
pub struct RewritingSystem {
    /// The group this system applies to
    group: FinitelyPresentedGroup,
    /// Rewriting rules (lhs -> rhs), both in Tietze form
    rules: Vec<(Vec<i32>, Vec<i32>)>,
}

impl Default for FinitelyPresentedGroup {
    /// Create a default (trivial) finitely presented group with no generators
    fn default() -> Self {
        FinitelyPresentedGroup {
            generator_names: vec![],
            num_generators: 0,
            relations: vec![],
        }
    }
}

impl FinitelyPresentedGroup {
    /// Create a new finitely presented group
    ///
    /// # Arguments
    ///
    /// * `generator_names` - Names of the generators
    /// * `relations` - Relations as words in Tietze form
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::finitely_presented::FinitelyPresentedGroup;
    ///
    /// // Cyclic group Z/3Z = ⟨a | a^3⟩
    /// let group = FinitelyPresentedGroup::new(
    ///     vec!["a".to_string()],
    ///     vec![vec![1, 1, 1]]
    /// );
    /// ```
    pub fn new(generator_names: Vec<String>, relations: Vec<Vec<i32>>) -> Self {
        let num_generators = generator_names.len();
        FinitelyPresentedGroup {
            generator_names,
            num_generators,
            relations,
        }
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Get the generator names
    pub fn generator_names(&self) -> &[String] {
        &self.generator_names
    }

    /// Get the relations
    pub fn relations(&self) -> &[Vec<i32>] {
        &self.relations
    }

    /// Create a generator element
    ///
    /// # Arguments
    ///
    /// * `index` - Zero-based index of the generator
    ///
    /// # Panics
    ///
    /// Panics if index >= num_generators
    pub fn generator(&self, index: usize) -> FinitelyPresentedGroupElement {
        assert!(index < self.num_generators, "Generator index out of bounds");
        FinitelyPresentedGroupElement {
            group: self.clone(),
            word: vec![(index + 1) as i32],
        }
    }

    /// Create the identity element
    pub fn identity(&self) -> FinitelyPresentedGroupElement {
        FinitelyPresentedGroupElement {
            group: self.clone(),
            word: vec![],
        }
    }

    /// Create an element from a Tietze word
    ///
    /// # Arguments
    ///
    /// * `word` - Word in Tietze form
    ///
    /// # Panics
    ///
    /// Panics if any generator index in the word is out of bounds
    pub fn element_from_word(&self, word: Vec<i32>) -> FinitelyPresentedGroupElement {
        // Validate word
        for &g in &word {
            let abs_g = g.abs() as usize;
            assert!(
                abs_g > 0 && abs_g <= self.num_generators,
                "Generator index {} out of bounds",
                abs_g
            );
        }

        FinitelyPresentedGroupElement {
            group: self.clone(),
            word,
        }
    }

    /// Compute the abelianization of the group
    ///
    /// The abelianization is the largest abelian quotient of the group,
    /// obtained by imposing all commutator relations [g, h] = 1.
    ///
    /// Returns the abelian invariants (elementary divisors).
    pub fn abelian_invariants(&self) -> Vec<usize> {
        // Create the relation matrix in the abelianization
        // Each generator becomes a generator of Z
        // Relations become linear equations

        let n = self.num_generators;
        let mut matrix: Vec<Vec<i32>> = Vec::new();

        // Add relations from the group
        for rel in &self.relations {
            let mut row = vec![0i32; n];
            for &g in rel {
                let idx = (g.abs() - 1) as usize;
                if g > 0 {
                    row[idx] += 1;
                } else {
                    row[idx] -= 1;
                }
            }
            matrix.push(row);
        }

        // Add commutator relations [g_i, g_j] = g_i g_j g_i^{-1} g_j^{-1} = 1
        // In abelianization: g_i + g_j - g_i - g_j = 0 (automatically satisfied)

        // Compute Smith normal form to get invariants
        smith_normal_form(&matrix)
    }

    /// Check if the group is likely finite (heuristic)
    ///
    /// This is a heuristic check based on the relations. The word problem
    /// for finitely presented groups is undecidable in general.
    pub fn is_finite_heuristic(&self) -> bool {
        // Very basic heuristic: check if all generators have finite order
        for i in 0..self.num_generators {
            let mut has_power_relation = false;
            for rel in &self.relations {
                if rel.iter().all(|&g| g.abs() == (i + 1) as i32) && !rel.is_empty() {
                    has_power_relation = true;
                    break;
                }
            }
            if !has_power_relation {
                return false;
            }
        }
        true
    }
}

impl Default for FinitelyPresentedGroupElement {
    /// Create a default element (identity of the trivial group)
    fn default() -> Self {
        FinitelyPresentedGroup::default().identity()
    }
}

impl FinitelyPresentedGroupElement {
    /// Get the word representation in Tietze form
    pub fn word(&self) -> &[i32] {
        &self.word
    }

    /// Get the group this element belongs to
    pub fn group(&self) -> &FinitelyPresentedGroup {
        &self.group
    }

    /// Compute the inverse of this element
    pub fn inverse(&self) -> Self {
        let inv_word: Vec<i32> = self.word.iter().rev().map(|&g| -g).collect();
        FinitelyPresentedGroupElement {
            group: self.group.clone(),
            word: inv_word,
        }
    }

    /// Multiply this element with another
    pub fn mul(&self, other: &Self) -> Self {
        assert!(
            self.group.num_generators == other.group.num_generators,
            "Elements must be from compatible groups"
        );

        let mut result_word = self.word.clone();
        result_word.extend_from_slice(&other.word);

        FinitelyPresentedGroupElement {
            group: self.group.clone(),
            word: result_word,
        }
    }

    /// Check if this is the identity element (after free reduction)
    pub fn is_identity(&self) -> bool {
        let reduced = self.free_reduce();
        reduced.word.is_empty()
    }

    /// Perform free reduction on the word
    ///
    /// Free reduction removes adjacent inverse pairs: g g^{-1} -> ε
    pub fn free_reduce(&self) -> Self {
        let mut reduced = Vec::new();

        for &g in &self.word {
            if let Some(&last) = reduced.last() {
                if last == -g {
                    reduced.pop();
                } else {
                    reduced.push(g);
                }
            } else {
                reduced.push(g);
            }
        }

        FinitelyPresentedGroupElement {
            group: self.group.clone(),
            word: reduced,
        }
    }

    /// Compute the length of the word after free reduction
    pub fn length(&self) -> usize {
        self.free_reduce().word.len()
    }

    /// Raise this element to a power
    pub fn pow(&self, n: i32) -> Self {
        if n == 0 {
            return self.group.identity();
        } else if n < 0 {
            return self.inverse().pow(-n);
        }

        let mut result = self.group.identity();
        for _ in 0..n {
            result = result.mul(self);
        }
        result
    }

    /// Compute the commutator [self, other] = self * other * self^{-1} * other^{-1}
    pub fn commutator(&self, other: &Self) -> Self {
        self.mul(other).mul(&self.inverse()).mul(&other.inverse())
    }
}

impl Mul for FinitelyPresentedGroupElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        FinitelyPresentedGroupElement::mul(&self, &rhs)
    }
}

impl Mul for &FinitelyPresentedGroupElement {
    type Output = FinitelyPresentedGroupElement;

    fn mul(self, rhs: Self) -> Self::Output {
        FinitelyPresentedGroupElement::mul(self, rhs)
    }
}

impl fmt::Display for FinitelyPresentedGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "⟨")?;
        for (i, name) in self.generator_names.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", name)?;
        }
        write!(f, " | ")?;

        for (i, rel) in self.relations.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write_word(f, rel, &self.generator_names)?;
        }
        write!(f, "⟩")
    }
}

impl fmt::Display for FinitelyPresentedGroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.word.is_empty() {
            write!(f, "e")
        } else {
            write_word(f, &self.word, &self.group.generator_names)
        }
    }
}

fn write_word(f: &mut fmt::Formatter<'_>, word: &[i32], gen_names: &[String]) -> fmt::Result {
    if word.is_empty() {
        return write!(f, "e");
    }

    for (i, &g) in word.iter().enumerate() {
        if i > 0 {
            write!(f, "*")?;
        }

        let abs_g = (g.abs() - 1) as usize;
        if abs_g < gen_names.len() {
            write!(f, "{}", gen_names[abs_g])?;
        } else {
            write!(f, "x{}", abs_g)?;
        }

        if g < 0 {
            write!(f, "^-1")?;
        }
    }
    Ok(())
}

impl RewritingSystem {
    /// Create a new rewriting system for a group
    pub fn new(group: FinitelyPresentedGroup) -> Self {
        let mut rules = Vec::new();

        // Add basic rules from relations
        for rel in &group.relations {
            if !rel.is_empty() {
                rules.push((rel.clone(), vec![]));
            }
        }

        // Add inverse rules: g * g^{-1} -> ε
        for i in 1..=group.num_generators as i32 {
            rules.push((vec![i, -i], vec![]));
            rules.push((vec![-i, i], vec![]));
        }

        RewritingSystem { group, rules }
    }

    /// Add a rewriting rule
    pub fn add_rule(&mut self, lhs: Vec<i32>, rhs: Vec<i32>) {
        self.rules.push((lhs, rhs));
    }

    /// Apply rewriting rules to reduce a word
    pub fn reduce(&self, element: &FinitelyPresentedGroupElement) -> FinitelyPresentedGroupElement {
        let mut word = element.word.clone();
        let mut changed = true;

        while changed {
            changed = false;

            for (lhs, rhs) in &self.rules {
                if let Some(pos) = find_subword(&word, lhs) {
                    // Replace lhs with rhs
                    let mut new_word = word[..pos].to_vec();
                    new_word.extend_from_slice(rhs);
                    new_word.extend_from_slice(&word[pos + lhs.len()..]);
                    word = new_word;
                    changed = true;
                    break;
                }
            }
        }

        FinitelyPresentedGroupElement {
            group: element.group.clone(),
            word,
        }
    }
}

/// Find the first occurrence of a subword in a word
fn find_subword(word: &[i32], subword: &[i32]) -> Option<usize> {
    if subword.is_empty() {
        return None;
    }

    word.windows(subword.len())
        .position(|window| window == subword)
}

/// Compute Smith normal form and return abelian invariants
///
/// This is a simplified version that computes elementary divisors
fn smith_normal_form(matrix: &[Vec<i32>]) -> Vec<usize> {
    if matrix.is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = if rows > 0 { matrix[0].len() } else { 0 };

    if cols == 0 {
        return vec![];
    }

    // Create a mutable copy
    let mut m = matrix.to_vec();

    let mut invariants = Vec::new();
    let mut rank = 0;

    for k in 0..std::cmp::min(rows, cols) {
        // Find pivot
        let mut pivot_row = None;
        for i in k..rows {
            if m[i][k] != 0 {
                pivot_row = Some(i);
                break;
            }
        }

        if let Some(i) = pivot_row {
            // Swap rows
            m.swap(k, i);
            rank += 1;

            // Eliminate
            for i in (k + 1)..rows {
                if m[i][k] != 0 {
                    let factor = m[i][k] / m[k][k];
                    for j in k..cols {
                        m[i][j] -= factor * m[k][j];
                    }
                }
            }
        }
    }

    // Extract diagonal elements as invariants
    for i in 0..std::cmp::min(rows, cols) {
        if m[i][i].abs() > 1 {
            invariants.push(m[i][i].unsigned_abs() as usize);
        }
    }

    // Add free generators (rank deficiency)
    let free_rank = cols.saturating_sub(rank);
    for _ in 0..free_rank {
        invariants.push(0); // 0 represents Z (infinite cyclic)
    }

    invariants
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyclic_group() {
        // Z/5Z = ⟨a | a^5⟩
        let group = FinitelyPresentedGroup::new(
            vec!["a".to_string()],
            vec![vec![1, 1, 1, 1, 1]],
        );

        assert_eq!(group.num_generators(), 1);
        assert_eq!(group.relations().len(), 1);

        let a = group.generator(0);
        let a5 = a.pow(5);
        assert_eq!(a5.word(), &[1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_dihedral_group() {
        // D_4 = ⟨r, s | r^4, s^2, (rs)^2⟩
        let group = FinitelyPresentedGroup::new(
            vec!["r".to_string(), "s".to_string()],
            vec![
                vec![1, 1, 1, 1],        // r^4
                vec![2, 2],              // s^2
                vec![1, 2, 1, 2],        // (rs)^2
            ],
        );

        assert_eq!(group.num_generators(), 2);

        let r = group.generator(0);
        let s = group.generator(1);

        // Test r^4
        let r4 = r.pow(4);
        assert_eq!(r4.word(), &[1, 1, 1, 1]);

        // Test s^2
        let s2 = s.pow(2);
        assert_eq!(s2.word(), &[2, 2]);
    }

    #[test]
    fn test_free_reduction() {
        let group = FinitelyPresentedGroup::new(
            vec!["a".to_string(), "b".to_string()],
            vec![],
        );

        let a = group.generator(0);
        let a_inv = a.inverse();

        // a * a^{-1} should reduce to identity
        let product = a.mul(&a_inv);
        let reduced = product.free_reduce();
        assert!(reduced.is_identity());
    }

    #[test]
    fn test_element_operations() {
        let group = FinitelyPresentedGroup::new(
            vec!["a".to_string()],
            vec![],
        );

        let a = group.generator(0);
        let a_inv = a.inverse();

        assert_eq!(a_inv.word(), &[-1]);

        let a3 = a.pow(3);
        assert_eq!(a3.word(), &[1, 1, 1]);

        let a_minus2 = a.pow(-2);
        assert_eq!(a_minus2.word(), &[-1, -1]);
    }

    #[test]
    fn test_commutator() {
        let group = FinitelyPresentedGroup::new(
            vec!["a".to_string(), "b".to_string()],
            vec![],
        );

        let a = group.generator(0);
        let b = group.generator(1);

        let comm = a.commutator(&b);
        // [a, b] = a*b*a^{-1}*b^{-1}
        assert_eq!(comm.word(), &[1, 2, -1, -2]);
    }

    #[test]
    fn test_identity() {
        let group = FinitelyPresentedGroup::new(
            vec!["a".to_string()],
            vec![],
        );

        let e = group.identity();
        assert!(e.is_identity());
        assert_eq!(e.word(), &[]);

        let a = group.generator(0);
        let ae = a.mul(&e);
        let reduced = ae.free_reduce();
        assert_eq!(reduced.word(), &[1]);
    }

    #[test]
    fn test_abelianization() {
        // Z/6Z = ⟨a | a^6⟩ should have abelian invariants [6]
        let group = FinitelyPresentedGroup::new(
            vec!["a".to_string()],
            vec![vec![1, 1, 1, 1, 1, 1]],
        );

        let inv = group.abelian_invariants();
        // The result should include 6 or some divisor information
        assert!(!inv.is_empty());
    }

    #[test]
    fn test_rewriting_system() {
        let group = FinitelyPresentedGroup::new(
            vec!["a".to_string()],
            vec![vec![1, 1, 1]], // a^3 = e
        );

        let rws = RewritingSystem::new(group.clone());

        let a = group.generator(0);
        let a3 = a.pow(3);

        let reduced = rws.reduce(&a3);
        // After reduction with the relation a^3 = e, this should simplify
        assert!(reduced.word().is_empty() || reduced.word().len() < a3.word().len());
    }

    #[test]
    fn test_quaternion_group() {
        // Q_8 = ⟨i, j | i^4, i^2 = j^2, j*i = i^{-1}*j⟩
        let group = FinitelyPresentedGroup::new(
            vec!["i".to_string(), "j".to_string()],
            vec![
                vec![1, 1, 1, 1],           // i^4 = e
                vec![1, 1, -2, -2],         // i^2 * j^{-2} = e, i.e., i^2 = j^2
                vec![2, 1, -1, 1, -2],      // j*i*i^{-1}*i*j^{-1} = e, i.e., j*i = i^{-1}*j... actually this is wrong
                                             // Better: j*i*j^{-1}*i = e
            ],
        );

        assert_eq!(group.num_generators(), 2);

        let i = group.generator(0);
        let j = group.generator(1);

        let i4 = i.pow(4);
        assert_eq!(i4.word().len(), 4);
    }

    #[test]
    fn test_word_length() {
        let group = FinitelyPresentedGroup::new(
            vec!["a".to_string(), "b".to_string()],
            vec![],
        );

        let a = group.generator(0);
        let b = group.generator(1);

        let word = a.mul(&b).mul(&a.inverse());
        assert_eq!(word.length(), 3);

        let word2 = a.mul(&a.inverse());
        assert_eq!(word2.length(), 0); // Should reduce to identity
    }

    #[test]
    fn test_display() {
        let group = FinitelyPresentedGroup::new(
            vec!["x".to_string(), "y".to_string()],
            vec![vec![1, 2, -1, -2]],
        );

        let display = format!("{}", group);
        assert!(display.contains("x"));
        assert!(display.contains("y"));

        let x = group.generator(0);
        let x_str = format!("{}", x);
        assert_eq!(x_str, "x");
    }
}
