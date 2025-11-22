//! Nakajima monomials and polyhedral realizations
//!
//! Nakajima introduced a monomial realization of crystals using commuting variables.
//! This provides an explicit realization for many important crystals.

use crate::operators::Crystal;
use crate::root_system::RootSystem;
use crate::weight::Weight;
use std::collections::BTreeMap;

/// A Nakajima monomial
///
/// Represented as a product of variables Y_{i,k} with integer exponents.
/// M = ∏_{i,k} Y_{i,k}^{a_{i,k}}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NakajimaMonomial {
    /// Exponents a_{i,k} for Y_{i,k}
    /// Stored as (i, k) → exponent
    pub exponents: BTreeMap<(usize, usize), i64>,
}

impl NakajimaMonomial {
    /// Create the identity monomial (all exponents 0)
    pub fn identity() -> Self {
        NakajimaMonomial {
            exponents: BTreeMap::new(),
        }
    }

    /// Create a monomial from exponents
    pub fn from_exponents(exponents: BTreeMap<(usize, usize), i64>) -> Self {
        NakajimaMonomial { exponents }
    }

    /// Get the exponent of Y_{i,k}
    pub fn exponent(&self, i: usize, k: usize) -> i64 {
        *self.exponents.get(&(i, k)).unwrap_or(&0)
    }

    /// Set the exponent of Y_{i,k}
    pub fn set_exponent(&mut self, i: usize, k: usize, exp: i64) {
        if exp == 0 {
            self.exponents.remove(&(i, k));
        } else {
            self.exponents.insert((i, k), exp);
        }
    }

    /// Multiply by Y_{i,k}
    pub fn multiply_y(&mut self, i: usize, k: usize) {
        let exp = self.exponent(i, k);
        self.set_exponent(i, k, exp + 1);
    }

    /// Divide by Y_{i,k} (if possible)
    pub fn divide_y(&mut self, i: usize, k: usize) -> bool {
        let exp = self.exponent(i, k);
        if exp > 0 {
            self.set_exponent(i, k, exp - 1);
            true
        } else {
            false
        }
    }

    /// Compute the weight
    pub fn compute_weight(&self, root_system: &RootSystem) -> Weight {
        let mut w = Weight::zero(root_system.rank);

        for (&(i, k), &exp) in &self.exponents {
            if i < root_system.rank && exp != 0 {
                // Weight contribution from Y_{i,k}
                // This depends on the specific realization
                let alpha_i = root_system.simple_root(i);
                // Simplified: weight = sum of exp * weight(Y_{i,k})
                w = &w + &(&alpha_i * exp);
            }
        }

        w
    }

    /// Pretty print the monomial
    pub fn to_string(&self) -> String {
        if self.exponents.is_empty() {
            return "1".to_string();
        }

        let mut terms: Vec<_> = self.exponents.iter().collect();
        terms.sort_by_key(|(&(i, k), _)| (i, k));

        let term_strings: Vec<String> = terms
            .iter()
            .map(|(&(i, k), &exp)| {
                if exp == 1 {
                    format!("Y_{{{},{}}}", i, k)
                } else {
                    format!("Y_{{{}+,{}}}^{}", i, k, exp)
                }
            })
            .collect();

        term_strings.join(" * ")
    }
}

/// Nakajima monomial crystal
///
/// This is a polyhedral realization of crystals using monomials.
#[derive(Debug, Clone)]
pub struct NakajimaCrystal {
    /// Root system
    pub root_system: RootSystem,
    /// Height (maximum k value)
    pub height: usize,
    /// Generated monomials
    elements_cache: Vec<NakajimaMonomial>,
}

impl NakajimaCrystal {
    /// Create a new Nakajima crystal
    pub fn new(root_system: RootSystem, height: usize) -> Self {
        NakajimaCrystal {
            root_system,
            height,
            elements_cache: Vec::new(),
        }
    }

    /// Generate monomials
    pub fn generate(&mut self, max_elements: usize) {
        self.elements_cache.clear();
        let identity = NakajimaMonomial::identity();
        self.elements_cache.push(identity.clone());

        let mut queue = vec![identity];
        let mut iterations = 0;
        let max_iterations = 100;

        while !queue.is_empty()
            && iterations < max_iterations
            && self.elements_cache.len() < max_elements
        {
            let mut new_queue = Vec::new();

            for monomial in queue {
                // Try applying f_i for each i
                for i in 0..self.root_system.rank {
                    if let Some(new_mon) = self.apply_fi_internal(&monomial, i) {
                        if !self.elements_cache.contains(&new_mon) {
                            self.elements_cache.push(new_mon.clone());
                            new_queue.push(new_mon);
                        }
                    }
                }
            }

            queue = new_queue;
            iterations += 1;
        }
    }

    /// Internal implementation of f_i
    fn apply_fi_internal(&self, m: &NakajimaMonomial, i: usize) -> Option<NakajimaMonomial> {
        // Simplified implementation
        // In the full Nakajima theory, this involves finding the rightmost
        // pole and multiplying by appropriate Y variables

        let mut new_m = m.clone();

        // For demonstration: multiply by Y_{i,1}
        new_m.multiply_y(i, 1);

        Some(new_m)
    }

    /// Internal implementation of e_i
    fn apply_ei_internal(&self, m: &NakajimaMonomial, i: usize) -> Option<NakajimaMonomial> {
        // Simplified implementation
        let mut new_m = m.clone();

        // Try to divide by Y_{i,1}
        if new_m.divide_y(i, 1) {
            Some(new_m)
        } else {
            None
        }
    }
}

impl Crystal for NakajimaCrystal {
    type Element = NakajimaMonomial;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.compute_weight(&self.root_system)
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i >= self.root_system.rank {
            return None;
        }
        self.apply_ei_internal(b, i)
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i >= self.root_system.rank {
            return None;
        }
        self.apply_fi_internal(b, i)
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.elements_cache.clone()
    }
}

/// Polyhedral realization data
///
/// The polyhedral realization uses inequalities to describe the crystal.
#[derive(Debug, Clone)]
pub struct PolyhedralData {
    /// Inequalities defining the allowed region
    pub inequalities: Vec<(Vec<i64>, i64)>, // (coefficients, constant)
}

impl PolyhedralData {
    /// Create new polyhedral data
    pub fn new() -> Self {
        PolyhedralData {
            inequalities: Vec::new(),
        }
    }

    /// Add an inequality
    pub fn add_inequality(&mut self, coeffs: Vec<i64>, constant: i64) {
        self.inequalities.push((coeffs, constant));
    }

    /// Check if a point satisfies all inequalities
    pub fn satisfies(&self, point: &[i64]) -> bool {
        for (coeffs, constant) in &self.inequalities {
            let sum: i64 = coeffs.iter().zip(point).map(|(c, p)| c * p).sum();
            if sum > *constant {
                return false;
            }
        }
        true
    }
}

impl Default for PolyhedralData {
    fn default() -> Self {
        Self::new()
    }
}

/// A-variables for the polyhedral realization
///
/// These are auxiliary variables used in the monomial formulas.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AVariables {
    /// Values a_{i,k}
    pub values: BTreeMap<(usize, usize), i64>,
}

impl AVariables {
    /// Create new A-variables (all zero)
    pub fn new() -> Self {
        AVariables {
            values: BTreeMap::new(),
        }
    }

    /// Get a_{i,k}
    pub fn get(&self, i: usize, k: usize) -> i64 {
        *self.values.get(&(i, k)).unwrap_or(&0)
    }

    /// Set a_{i,k}
    pub fn set(&mut self, i: usize, k: usize, value: i64) {
        if value == 0 {
            self.values.remove(&(i, k));
        } else {
            self.values.insert((i, k), value);
        }
    }

    /// Convert to monomial using Y_{i,k} = A_{i,k} / A_{i,k-1}
    pub fn to_monomial(&self, _max_k: usize) -> NakajimaMonomial {
        let mut exponents = BTreeMap::new();

        // Compute Y exponents from A values
        for (&(i, k), &a_val) in &self.values {
            if k > 0 {
                let a_prev = self.get(i, k.saturating_sub(1));
                let y_exp = a_val - a_prev;
                if y_exp != 0 {
                    exponents.insert((i, k), y_exp);
                }
            } else {
                if a_val != 0 {
                    exponents.insert((i, k), a_val);
                }
            }
        }

        NakajimaMonomial::from_exponents(exponents)
    }
}

impl Default for AVariables {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::root_system::{RootSystem, RootSystemType};

    #[test]
    fn test_nakajima_monomial() {
        let mut m = NakajimaMonomial::identity();
        assert_eq!(m.exponent(1, 1), 0);

        m.multiply_y(1, 1);
        assert_eq!(m.exponent(1, 1), 1);

        m.multiply_y(1, 1);
        assert_eq!(m.exponent(1, 1), 2);

        assert!(m.divide_y(1, 1));
        assert_eq!(m.exponent(1, 1), 1);
    }

    #[test]
    fn test_nakajima_monomial_string() {
        let mut m = NakajimaMonomial::identity();
        assert_eq!(m.to_string(), "1");

        m.multiply_y(0, 1);
        assert!(m.to_string().contains("Y"));
    }

    #[test]
    fn test_nakajima_crystal() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let mut crystal = NakajimaCrystal::new(root_system, 3);

        crystal.generate(20);
        assert!(!crystal.elements_cache.is_empty());
    }

    #[test]
    fn test_polyhedral_data() {
        let mut poly = PolyhedralData::new();
        poly.add_inequality(vec![1, 1], 5);

        assert!(poly.satisfies(&[2, 2]));  // 2 + 2 = 4 <= 5
        assert!(!poly.satisfies(&[3, 3])); // 3 + 3 = 6 > 5
    }

    #[test]
    fn test_a_variables() {
        let mut a_vars = AVariables::new();
        a_vars.set(0, 0, 1);
        a_vars.set(0, 1, 3);

        assert_eq!(a_vars.get(0, 0), 1);
        assert_eq!(a_vars.get(0, 1), 3);

        let m = a_vars.to_monomial(2);
        // Y_{0,1} should have exponent 3 - 1 = 2
        assert_eq!(m.exponent(0, 1), 2);
    }

    #[test]
    fn test_nakajima_weight() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let mut m = NakajimaMonomial::identity();
        m.multiply_y(0, 1);

        let weight = m.compute_weight(&root_system);
        // Should contribute alpha_0
        assert_eq!(weight.coords[0], 2); // From Cartan matrix
    }
}
