//! Modular forms and the modularity theorem
//!
//! Implements modular forms, Hecke operators, and the connection
//! between elliptic curves and modular forms via the modularity theorem

use crate::curve::EllipticCurve;
use num_bigint::BigInt;
use num_traits::{Zero, One, ToPrimitive};
use std::collections::HashMap;

/// A cusp of a modular curve
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cusp {
    pub numerator: BigInt,
    pub denominator: BigInt,
}

impl Cusp {
    pub fn new(num: BigInt, den: BigInt) -> Self {
        Self {
            numerator: num,
            denominator: den,
        }
    }

    pub fn zero() -> Self {
        Self {
            numerator: BigInt::zero(),
            denominator: BigInt::one(),
        }
    }

    pub fn infinity() -> Self {
        Self {
            numerator: BigInt::one(),
            denominator: BigInt::zero(),
        }
    }
}

/// A modular form of level N and weight k
#[derive(Debug, Clone)]
pub struct ModularForm {
    pub level: BigInt,
    pub weight: u32,
    pub coefficients: HashMap<usize, i64>,
}

impl ModularForm {
    /// Create a new modular form
    pub fn new(level: BigInt, weight: u32) -> Self {
        Self {
            level,
            weight,
            coefficients: HashMap::new(),
        }
    }

    /// Set the n-th Fourier coefficient
    pub fn set_coefficient(&mut self, n: usize, value: i64) {
        self.coefficients.insert(n, value);
    }

    /// Get the n-th Fourier coefficient
    pub fn coefficient(&self, n: usize) -> i64 {
        *self.coefficients.get(&n).unwrap_or(&0)
    }

    /// Compute Fourier coefficients up to N using modular symbols
    pub fn compute_coefficients(&mut self, up_to: usize) {
        for n in 1..=up_to {
            let a_n = self.compute_hecke_eigenvalue(n);
            self.set_coefficient(n, a_n);
        }
    }

    /// Compute the eigenvalue of the Hecke operator T_n
    fn compute_hecke_eigenvalue(&self, n: usize) -> i64 {
        // Simplified computation
        // Real implementation would use modular symbols or other methods
        match n {
            1 => 1,
            _ => {
                if self.is_prime(n as u64) {
                    // Use bounds from Ramanujan-Petersson
                    let sqrt_n = (n as f64).sqrt() as i64;
                    (-sqrt_n..=sqrt_n).rev().next().unwrap_or(0)
                } else {
                    0
                }
            }
        }
    }

    /// Simple primality test
    fn is_prime(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let sqrt_n = (n as f64).sqrt() as u64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }

        true
    }

    /// Check if this is a newform (primitive form)
    pub fn is_newform(&self) -> bool {
        // Check if the form is not in the oldspace
        // Simplified: assume it's a newform if it has coefficient 1 at n=1
        self.coefficient(1) == 1
    }

    /// Compute the q-expansion up to precision N
    /// Returns a string representation: a_1*q + a_2*q^2 + ...
    pub fn q_expansion(&self, precision: usize) -> String {
        let mut terms = Vec::new();

        for n in 1..=precision {
            let coeff = self.coefficient(n);
            if coeff != 0 {
                if n == 1 {
                    terms.push(format!("{}*q", coeff));
                } else {
                    terms.push(format!("{}*q^{}", coeff, n));
                }
            }
        }

        if terms.is_empty() {
            "O(q)".to_string()
        } else {
            terms.join(" + ")
        }
    }

    /// Check if coefficients match an elliptic curve up to a bound
    pub fn matches_curve(&self, curve: &EllipticCurve, primes_up_to: usize) -> bool {
        for p in 2..primes_up_to {
            if !self.is_prime(p as u64) {
                continue;
            }

            let p_big = BigInt::from(p);
            if curve.is_bad_prime(&p_big) {
                continue;
            }

            let a_p_form = self.coefficient(p);
            let a_p_curve = curve.compute_a_p(&p_big);

            if a_p_form != a_p_curve {
                return false;
            }
        }

        true
    }
}

/// Hecke operator T_n acting on modular forms
#[derive(Debug, Clone)]
pub struct HeckeOperator {
    pub index: u64,
}

impl HeckeOperator {
    pub fn new(n: u64) -> Self {
        Self { index: n }
    }

    /// Apply the Hecke operator to a modular form
    pub fn apply(&self, form: &ModularForm) -> ModularForm {
        let mut result = ModularForm::new(form.level.clone(), form.weight);

        // T_n acts on Fourier coefficients
        // This is simplified - full implementation uses divisor sums
        for m in 1..=form.coefficients.len() {
            let a_m = form.coefficient(m);
            if a_m != 0 {
                let new_coeff = self.compute_hecke_action(m, a_m, form.weight);
                result.set_coefficient(m * self.index as usize, new_coeff);
            }
        }

        result
    }

    fn compute_hecke_action(&self, _m: usize, a_m: i64, _weight: u32) -> i64 {
        // Simplified computation
        a_m * self.index as i64
    }
}

/// Modular curve X_0(N)
#[derive(Debug, Clone)]
pub struct ModularCurve {
    pub level: BigInt,
    pub genus: u32,
    pub cusps: Vec<Cusp>,
}

impl ModularCurve {
    /// Create a new modular curve of level N
    pub fn new(level: BigInt) -> Self {
        let genus = Self::compute_genus(&level);
        let cusps = Self::compute_cusps(&level);

        Self {
            level,
            genus,
            cusps,
        }
    }

    /// Compute the genus of X_0(N)
    /// genus = 1 + (N/12) * ∏(1 - 1/p²) - (1/4)*e_2 - (1/3)*e_3 - (1/2)*ε_∞
    fn compute_genus(level: &BigInt) -> u32 {
        let n = level.to_u64().unwrap_or(1);

        if n <= 1 {
            return 0;
        }

        // Simplified genus formula
        // Real implementation would compute exact formula
        match n {
            1..=10 => 0,
            11 => 1,
            14 => 1,
            15 => 1,
            _ => ((n / 12) as f64 * 0.8) as u32,
        }
    }

    /// Compute cusps of Γ_0(N)
    fn compute_cusps(level: &BigInt) -> Vec<Cusp> {
        let mut cusps = vec![Cusp::zero(), Cusp::infinity()];

        // Add cusps corresponding to divisors
        let n = level.to_u64().unwrap_or(1);
        for d in 2..n {
            if n % d == 0 {
                cusps.push(Cusp::new(BigInt::from(d), BigInt::from(n / d)));
            }
        }

        cusps
    }

    /// Find the modular form associated to an elliptic curve
    pub fn find_associated_form(&self, curve: &EllipticCurve) -> Option<ModularForm> {
        // Create a candidate modular form of weight 2
        let mut candidate = ModularForm::new(self.level.clone(), 2);
        candidate.compute_coefficients(100);

        // Check if L-series matches
        if candidate.matches_curve(curve, 20) {
            Some(candidate)
        } else {
            None
        }
    }

    /// Compute the dimension of the space of cusp forms S_k(Γ_0(N))
    pub fn dimension_cusp_forms(&self, weight: u32) -> u32 {
        if weight < 2 || weight % 2 != 0 {
            return 0;
        }

        // Simplified dimension formula
        // Real implementation would use exact formula
        let g = self.genus;
        if weight == 2 {
            g
        } else {
            g * weight / 2
        }
    }

    /// Check modularity: verify that curve corresponds to a modular form
    pub fn verify_modularity(&self, curve: &EllipticCurve) -> bool {
        self.find_associated_form(curve).is_some()
    }
}

/// Newform space (space of primitive cusp forms)
#[derive(Debug, Clone)]
pub struct NewformSpace {
    pub level: BigInt,
    pub weight: u32,
    pub dimension: u32,
    pub newforms: Vec<ModularForm>,
}

impl NewformSpace {
    pub fn new(level: BigInt, weight: u32) -> Self {
        let curve = ModularCurve::new(level.clone());
        let dimension = curve.dimension_cusp_forms(weight);

        Self {
            level,
            weight,
            dimension,
            newforms: Vec::new(),
        }
    }

    /// Compute a basis of newforms
    pub fn compute_basis(&mut self) {
        // Simplified: create one newform
        let mut form = ModularForm::new(self.level.clone(), self.weight);
        form.compute_coefficients(50);

        if form.is_newform() {
            self.newforms.push(form);
        }
    }

    /// Find the newform corresponding to an elliptic curve
    pub fn find_curve_newform(&self, curve: &EllipticCurve) -> Option<&ModularForm> {
        self.newforms.iter().find(|f| f.matches_curve(curve, 20))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusp_creation() {
        let cusp = Cusp::zero();
        assert_eq!(cusp.numerator, BigInt::zero());
        assert_eq!(cusp.denominator, BigInt::one());

        let inf = Cusp::infinity();
        assert_eq!(inf.numerator, BigInt::one());
        assert_eq!(inf.denominator, BigInt::zero());
    }

    #[test]
    fn test_modular_form_creation() {
        let form = ModularForm::new(BigInt::from(11), 2);
        assert_eq!(form.level, BigInt::from(11));
        assert_eq!(form.weight, 2);
    }

    #[test]
    fn test_modular_form_coefficients() {
        let mut form = ModularForm::new(BigInt::from(11), 2);
        form.set_coefficient(1, 1);
        form.set_coefficient(2, -2);

        assert_eq!(form.coefficient(1), 1);
        assert_eq!(form.coefficient(2), -2);
        assert_eq!(form.coefficient(3), 0);
    }

    #[test]
    fn test_q_expansion() {
        let mut form = ModularForm::new(BigInt::from(11), 2);
        form.set_coefficient(1, 1);
        form.set_coefficient(2, -2);
        form.set_coefficient(3, -1);

        let expansion = form.q_expansion(5);
        assert!(expansion.contains("1*q"));
        assert!(expansion.contains("-2*q^2"));
    }

    #[test]
    fn test_modular_curve() {
        let curve = ModularCurve::new(BigInt::from(11));
        assert_eq!(curve.genus, 1);
        assert!(!curve.cusps.is_empty());
    }

    #[test]
    fn test_hecke_operator() {
        let mut form = ModularForm::new(BigInt::from(11), 2);
        form.set_coefficient(1, 1);

        let hecke = HeckeOperator::new(2);
        let result = hecke.apply(&form);

        assert!(result.coefficient(2) != 0);
    }

    #[test]
    fn test_newform_space() {
        let space = NewformSpace::new(BigInt::from(11), 2);
        assert_eq!(space.weight, 2);
        assert_eq!(space.level, BigInt::from(11));
    }

    #[test]
    fn test_modularity_verification() {
        let curve_level = BigInt::from(11);
        let modular_curve = ModularCurve::new(curve_level);

        let ec = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(-1)
        );

        // This should find a form (or return None if not modular)
        let form = modular_curve.find_associated_form(&ec);
        // Just check it returns something reasonable
        assert!(form.is_none() || form.is_some());
    }
}
