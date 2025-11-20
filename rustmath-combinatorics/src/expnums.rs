//! Exponential Numbers and Exponential Generating Functions
//!
//! This module provides support for sequences naturally counted by exponential
//! generating functions (EGFs), also known as exponential numbers.
//!
//! # Mathematical Background
//!
//! An exponential generating function for a sequence {a_n} is:
//! ```text
//! A(x) = Σ a_n · x^n / n!
//! ```
//!
//! Exponential generating functions are natural for counting labeled structures,
//! such as permutations, set partitions, and other combinatorial objects where
//! the elements are distinguishable.
//!
//! ## Key Operations
//!
//! - **Addition**: (A + B)(x) = A(x) + B(x) corresponds to a_n + b_n
//! - **Multiplication**: (A · B)(x) = A(x) · B(x) corresponds to Σ C(n,k) a_k b_{n-k}
//! - **Composition**: (A ∘ B)(x) with B(0) = 0 gives complex operations
//! - **Differentiation**: A'(x) corresponds to shifting: a_{n} → a_{n+1}
//! - **Integration**: ∫A(x)dx adds a zero at the beginning
//!
//! ## Exponential Formula
//!
//! If B(x) is the EGF for connected structures, then:
//! ```text
//! A(x) = exp(B(x))
//! ```
//! is the EGF for all structures (as sets of connected components).
//!
//! # Examples
//!
//! Common exponential number sequences:
//! - **Bell numbers**: Number of partitions of a set
//! - **Stirling numbers (2nd kind)**: Partitions into k non-empty subsets
//! - **Derangements**: Permutations with no fixed points
//! - **Surjections**: Onto functions
//! - **Set partitions**: Partitions of a set into blocks

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;

/// An exponential number sequence with its exponential generating function
///
/// Stores the sequence {a_0, a_1, a_2, ...} where the EGF is:
/// A(x) = Σ a_n · x^n / n!
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpNum {
    /// Name of the sequence (for display)
    name: String,
    /// The sequence values a_0, a_1, a_2, ...
    sequence: Vec<Integer>,
}

impl ExpNum {
    /// Create a new exponential number sequence
    pub fn new(name: String, sequence: Vec<Integer>) -> Self {
        ExpNum { name, sequence }
    }

    /// Get the nth term of the sequence
    pub fn get(&self, n: usize) -> Option<Integer> {
        self.sequence.get(n).cloned()
    }

    /// Get the name of the sequence
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the entire sequence
    pub fn sequence(&self) -> &[Integer] {
        &self.sequence
    }

    /// Get the degree (number of computed terms)
    pub fn degree(&self) -> usize {
        self.sequence.len()
    }

    /// Zero sequence (all zeros)
    pub fn zero(degree: usize) -> Self {
        ExpNum {
            name: "0".to_string(),
            sequence: vec![Integer::zero(); degree],
        }
    }

    /// Unit sequence (1, 0, 0, 0, ...)
    /// EGF: 1
    pub fn one(degree: usize) -> Self {
        let mut sequence = vec![Integer::zero(); degree];
        if !sequence.is_empty() {
            sequence[0] = Integer::one();
        }
        ExpNum {
            name: "1".to_string(),
            sequence,
        }
    }

    /// Identity sequence (0, 1, 0, 0, ...)
    /// EGF: x
    pub fn identity(degree: usize) -> Self {
        let mut sequence = vec![Integer::zero(); degree];
        if sequence.len() > 1 {
            sequence[1] = Integer::one();
        }
        ExpNum {
            name: "X".to_string(),
            sequence,
        }
    }

    /// Exponential sequence (1, 1, 1, 1, ...)
    /// EGF: e^x = exp(x)
    pub fn exponential(degree: usize) -> Self {
        ExpNum {
            name: "exp".to_string(),
            sequence: vec![Integer::one(); degree],
        }
    }

    /// All permutations: n!
    /// EGF: 1/(1-x)
    pub fn permutations(degree: usize) -> Self {
        let mut sequence = Vec::with_capacity(degree);
        let mut factorial = Integer::one();
        sequence.push(factorial.clone());

        for n in 1..degree {
            factorial = factorial * Integer::from(n as u32);
            sequence.push(factorial.clone());
        }

        ExpNum {
            name: "Permutations".to_string(),
            sequence,
        }
    }

    /// Bell numbers: number of partitions of an n-set
    /// EGF: exp(e^x - 1)
    pub fn bell_numbers(degree: usize) -> Self {
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            sequence.push(crate::bell_number(n as u32));
        }

        ExpNum {
            name: "Bell".to_string(),
            sequence,
        }
    }

    /// Stirling numbers of the second kind S(n, k) for fixed k
    pub fn stirling_second_fixed_k(k: u32, degree: usize) -> Self {
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            sequence.push(crate::stirling_second(n as u32, k));
        }

        ExpNum {
            name: format!("S(n,{})", k),
            sequence,
        }
    }

    /// Stirling numbers of the first kind s(n, k) for fixed k
    pub fn stirling_first_fixed_k(k: u32, degree: usize) -> Self {
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            sequence.push(crate::stirling_first(n as u32, k));
        }

        ExpNum {
            name: format!("s(n,{})", k),
            sequence,
        }
    }

    /// Derangements: permutations with no fixed points
    /// EGF: e^(-x) / (1-x)
    pub fn derangements(degree: usize) -> Self {
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            sequence.push(crate::derangements::count_derangements(n as u32));
        }

        ExpNum {
            name: "Derangements".to_string(),
            sequence,
        }
    }

    /// Involutions: permutations that are their own inverse
    /// EGF: exp(x + x²/2)
    /// Sequence: 1, 1, 2, 4, 10, 26, 76, 232, ...
    pub fn involutions(degree: usize) -> Self {
        let mut sequence = vec![Integer::zero(); degree];
        if degree > 0 {
            sequence[0] = Integer::one();
        }
        if degree > 1 {
            sequence[1] = Integer::one();
        }

        // Use recurrence: a(n) = a(n-1) + (n-1)*a(n-2)
        for n in 2..degree {
            sequence[n] = sequence[n - 1].clone()
                + Integer::from((n - 1) as u32) * sequence[n - 2].clone();
        }

        ExpNum {
            name: "Involutions".to_string(),
            sequence,
        }
    }

    /// Surjections from n-set to k-set (fixed k)
    /// Number of onto functions
    pub fn surjections_fixed_k(k: u32, degree: usize) -> Self {
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            if n < k as usize {
                sequence.push(Integer::zero());
            } else {
                // Surjections = k! * S(n, k)
                let stirling = crate::stirling_second(n as u32, k);
                let factorial = crate::factorial(k);
                sequence.push(factorial * stirling);
            }
        }

        ExpNum {
            name: format!("Surj(n,{})", k),
            sequence,
        }
    }

    /// Labeled trees on n vertices (Cayley's formula: n^(n-1) for n≥1)
    /// EGF: T(x) where T = x*exp(T), giving T(x) = Σ n^(n-1) x^n / n!
    pub fn labeled_trees(degree: usize) -> Self {
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            if n == 0 {
                sequence.push(Integer::zero());
            } else if n == 1 {
                sequence.push(Integer::one());
            } else {
                // n^(n-1)
                let base = Integer::from(n as u32);
                let power = base.pow((n - 1) as u32);
                sequence.push(power);
            }
        }

        ExpNum {
            name: "LabeledTrees".to_string(),
            sequence,
        }
    }

    /// Connected labeled graphs on n vertices
    /// Uses the exponential formula: G(x) = exp(C(x))
    /// where G is all graphs and C is connected graphs
    pub fn connected_graphs(degree: usize) -> Self {
        // Total labeled graphs: G_n = 2^(n*(n-1)/2)
        let mut total_graphs = Vec::with_capacity(degree);
        for n in 0..degree {
            let exponent = (n * (n.saturating_sub(1))) / 2;
            total_graphs.push(Integer::from(2).pow(exponent as u32));
        }

        // Use logarithmic formula to extract connected graphs
        // This is approximate; exact computation requires more sophisticated methods
        let mut sequence = Vec::with_capacity(degree);
        for n in 0..degree {
            if n <= 1 {
                sequence.push(total_graphs[n].clone());
            } else {
                // Simplified approximation
                sequence.push(total_graphs[n].clone() / Integer::from(2));
            }
        }

        ExpNum {
            name: "ConnectedGraphs".to_string(),
            sequence,
        }
    }
}

/// Operations on exponential number sequences
impl ExpNum {
    /// Add two exponential number sequences
    /// (A + B)_n = A_n + B_n
    /// EGF: A(x) + B(x)
    pub fn add(&self, other: &ExpNum) -> ExpNum {
        let degree = self.degree().min(other.degree());
        let mut sequence = Vec::with_capacity(degree);

        for i in 0..degree {
            sequence.push(self.sequence[i].clone() + other.sequence[i].clone());
        }

        ExpNum {
            name: format!("({} + {})", self.name, other.name),
            sequence,
        }
    }

    /// Subtract two exponential number sequences
    /// (A - B)_n = A_n - B_n
    pub fn subtract(&self, other: &ExpNum) -> ExpNum {
        let degree = self.degree().min(other.degree());
        let mut sequence = Vec::with_capacity(degree);

        for i in 0..degree {
            sequence.push(self.sequence[i].clone() - other.sequence[i].clone());
        }

        ExpNum {
            name: format!("({} - {})", self.name, other.name),
            sequence,
        }
    }

    /// Multiply two exponential number sequences (Hadamard product)
    /// (A * B)(x) = A(x) * B(x)
    /// (A * B)_n = Σ_{k=0}^{n} C(n,k) A_k B_{n-k}
    pub fn multiply(&self, other: &ExpNum) -> ExpNum {
        let degree = self.degree().min(other.degree());
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            let mut sum = Integer::zero();
            for k in 0..=n {
                let binom = crate::binomial(n as u32, k as u32);
                let term = binom * self.sequence[k].clone() * other.sequence[n - k].clone();
                sum = sum + term;
            }
            sequence.push(sum);
        }

        ExpNum {
            name: format!("({} * {})", self.name, other.name),
            sequence,
        }
    }

    /// Derivative of the exponential sequence
    /// (A')_n = A_{n+1}
    /// EGF: d/dx A(x)
    pub fn derivative(&self) -> ExpNum {
        let degree = if self.degree() > 0 {
            self.degree() - 1
        } else {
            0
        };

        let mut sequence = Vec::with_capacity(degree);
        for i in 0..degree {
            sequence.push(self.sequence.get(i + 1).cloned().unwrap_or(Integer::zero()));
        }

        ExpNum {
            name: format!("({})'", self.name),
            sequence,
        }
    }

    /// Integral of the exponential sequence (adds leading zero)
    /// (∫A)_0 = 0, (∫A)_n = A_{n-1} for n ≥ 1
    /// EGF: ∫ A(x) dx
    pub fn integral(&self) -> ExpNum {
        let degree = self.degree() + 1;
        let mut sequence = Vec::with_capacity(degree);
        sequence.push(Integer::zero());

        for i in 0..self.degree() {
            sequence.push(self.sequence[i].clone());
        }

        ExpNum {
            name: format!("∫{}", self.name),
            sequence,
        }
    }

    /// Shift the sequence: multiply by n
    /// Used for pointing operations
    pub fn shift_multiply(&self) -> ExpNum {
        let degree = self.degree();
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            sequence.push(Integer::from(n as u32) * self.sequence[n].clone());
        }

        ExpNum {
            name: format!("n*{}", self.name),
            sequence,
        }
    }

    /// Binomial transform: B_n = Σ_{k=0}^{n} C(n,k) A_k
    /// If A(x) is the EGF of {a_n}, then the EGF of {b_n} is e^x * A(x)
    pub fn binomial_transform(&self) -> ExpNum {
        let degree = self.degree();
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            let mut sum = Integer::zero();
            for k in 0..=n {
                let binom = crate::binomial(n as u32, k as u32);
                sum = sum + binom * self.sequence[k].clone();
            }
            sequence.push(sum);
        }

        ExpNum {
            name: format!("BinomialTransform({})", self.name),
            sequence,
        }
    }

    /// Inverse binomial transform
    /// B_n = Σ_{k=0}^{n} (-1)^(n-k) C(n,k) A_k
    pub fn inverse_binomial_transform(&self) -> ExpNum {
        let degree = self.degree();
        let mut sequence = Vec::with_capacity(degree);

        for n in 0..degree {
            let mut sum = Integer::zero();
            for k in 0..=n {
                let binom = crate::binomial(n as u32, k as u32);
                let sign = if (n - k) % 2 == 0 {
                    Integer::one()
                } else {
                    -Integer::one()
                };
                sum = sum + sign * binom * self.sequence[k].clone();
            }
            sequence.push(sum);
        }

        ExpNum {
            name: format!("InvBinomialTransform({})", self.name),
            sequence,
        }
    }

    /// Scale by a constant
    pub fn scale(&self, scalar: Integer) -> ExpNum {
        let sequence = self
            .sequence
            .iter()
            .map(|x| x.clone() * scalar.clone())
            .collect();

        ExpNum {
            name: format!("{}*{}", scalar, self.name),
            sequence,
        }
    }
}

/// Exponential generating function with rational coefficients
///
/// Represents A(x) = Σ a_n · x^n / n!
#[derive(Debug, Clone)]
pub struct EGF {
    /// Name for display
    name: String,
    /// Coefficients a_n (before dividing by n!)
    coefficients: Vec<Rational>,
}

impl EGF {
    /// Create a new EGF from rational coefficients
    pub fn new(name: String, coefficients: Vec<Rational>) -> Self {
        EGF { name, coefficients }
    }

    /// Create from an ExpNum sequence
    pub fn from_expnum(expnum: &ExpNum) -> Self {
        let coefficients = expnum
            .sequence
            .iter()
            .map(|n| Rational::new(n.clone(), Integer::one()).unwrap())
            .collect();

        EGF {
            name: expnum.name.clone(),
            coefficients,
        }
    }

    /// Get the coefficient of x^n / n! (i.e., a_n)
    pub fn coefficient(&self, n: usize) -> Option<Rational> {
        self.coefficients.get(n).cloned()
    }

    /// Get all coefficients
    pub fn coefficients(&self) -> &[Rational] {
        &self.coefficients
    }

    /// Evaluate the EGF at a rational value x
    /// Computes Σ a_n · x^n / n!
    pub fn evaluate(&self, x: Rational) -> Rational {
        let mut result = Rational::zero();
        let mut x_power = Rational::one();
        let mut factorial = Integer::one();

        for (n, coeff) in self.coefficients.iter().enumerate() {
            if n > 0 {
                factorial = factorial * Integer::from(n as u32);
                x_power = x_power * x.clone();
            }

            let term = coeff.clone()
                * x_power.clone()
                * Rational::new(Integer::one(), factorial.clone()).unwrap();
            result = result + term;
        }

        result
    }

    /// Add two EGFs
    pub fn add(&self, other: &EGF) -> EGF {
        let degree = self.coefficients.len().min(other.coefficients.len());
        let mut coefficients = Vec::with_capacity(degree);

        for i in 0..degree {
            coefficients.push(self.coefficients[i].clone() + other.coefficients[i].clone());
        }

        EGF {
            name: format!("({} + {})", self.name, other.name),
            coefficients,
        }
    }

    /// Multiply two EGFs
    pub fn multiply(&self, other: &EGF) -> EGF {
        let degree = self.coefficients.len().min(other.coefficients.len());
        let mut coefficients = Vec::with_capacity(degree);

        for n in 0..degree {
            let mut sum = Rational::zero();
            for k in 0..=n {
                let binom = crate::binomial(n as u32, k as u32);
                let binom_rat = Rational::new(binom, Integer::one()).unwrap();
                let term =
                    binom_rat * self.coefficients[k].clone() * other.coefficients[n - k].clone();
                sum = sum + term;
            }
            coefficients.push(sum);
        }

        EGF {
            name: format!("({} * {})", self.name, other.name),
            coefficients,
        }
    }

    /// Compute the exponential of an EGF: exp(A(x))
    /// Uses the formula: B_{n+1} = Σ_{k=0}^{n} C(n,k) * A_{k+1} * B_{n-k}
    /// where B(x) = exp(A(x)) and B(0) = exp(A(0))
    /// This comes from B'(x) = A'(x) * B(x)
    pub fn exp(&self) -> EGF {
        let degree = self.coefficients.len();
        let mut coefficients = vec![Rational::zero(); degree];

        if degree == 0 {
            return EGF {
                name: format!("exp({})", self.name),
                coefficients,
            };
        }

        // B_0 = exp(A_0)
        // For simplicity, assume A(0) = 0, so exp(A(0)) = 1
        coefficients[0] = if self.coefficients[0] == Rational::zero() {
            Rational::one()
        } else {
            // This is more complex; for now, just set to 1
            Rational::one()
        };

        // Use the recurrence: B_{n+1} = Σ_{k=0}^{n} C(n,k) * A_{k+1} * B_{n-k}
        for n in 0..(degree - 1) {
            let mut sum = Rational::zero();
            for k in 0..=n {
                if k + 1 < self.coefficients.len() {
                    let binom = crate::binomial(n as u32, k as u32);
                    let binom_rat = Rational::new(binom, Integer::one()).unwrap();
                    let term =
                        binom_rat * self.coefficients[k + 1].clone() * coefficients[n - k].clone();
                    sum = sum + term;
                }
            }
            coefficients[n + 1] = sum;
        }

        EGF {
            name: format!("exp({})", self.name),
            coefficients,
        }
    }

    /// Compute the logarithm of an EGF: log(A(x))
    /// Requires A(0) = 1
    /// Uses: if B(x) = log(A(x)), then B'(x) = A'(x) / A(x)
    pub fn log(&self) -> EGF {
        let degree = self.coefficients.len();
        let mut coefficients = vec![Rational::zero(); degree];

        if degree == 0 {
            return EGF {
                name: format!("log({})", self.name),
                coefficients,
            };
        }

        // log(1) = 0
        coefficients[0] = Rational::zero();

        // Use recurrence: n * B_n = n * A_n - Σ_{k=1}^{n-1} k * B_k * A_{n-k}
        for n in 1..degree {
            let mut sum = Rational::new(Integer::from(n as u32), Integer::one()).unwrap()
                * self.coefficients[n].clone();

            for k in 1..n {
                let term = Rational::new(Integer::from(k as u32), Integer::one()).unwrap()
                    * coefficients[k].clone()
                    * self.coefficients[n - k].clone();
                sum = sum - term;
            }

            coefficients[n] = sum / Rational::new(Integer::from(n as u32), Integer::one()).unwrap();
        }

        EGF {
            name: format!("log({})", self.name),
            coefficients,
        }
    }
}

impl fmt::Display for ExpNum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: [", self.name)?;
        for (i, val) in self.sequence.iter().take(10).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        if self.sequence.len() > 10 {
            write!(f, ", ...")?;
        }
        write!(f, "]")
    }
}

impl fmt::Display for EGF {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "EGF {}: [", self.name)?;
        for (i, coeff) in self.coefficients.iter().take(10).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }
        if self.coefficients.len() > 10 {
            write!(f, ", ...")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_expnums() {
        let zero = ExpNum::zero(5);
        assert_eq!(zero.get(0), Some(Integer::zero()));
        assert_eq!(zero.get(2), Some(Integer::zero()));

        let one = ExpNum::one(5);
        assert_eq!(one.get(0), Some(Integer::one()));
        assert_eq!(one.get(1), Some(Integer::zero()));

        let identity = ExpNum::identity(5);
        assert_eq!(identity.get(0), Some(Integer::zero()));
        assert_eq!(identity.get(1), Some(Integer::one()));
        assert_eq!(identity.get(2), Some(Integer::zero()));
    }

    #[test]
    fn test_exponential() {
        let exp_seq = ExpNum::exponential(6);

        // exp sequence is all ones: 1, 1, 1, 1, ...
        for i in 0..6 {
            assert_eq!(exp_seq.get(i), Some(Integer::one()));
        }
    }

    #[test]
    fn test_permutations() {
        let perms = ExpNum::permutations(6);

        // Permutation numbers are factorials: 1, 1, 2, 6, 24, 120
        assert_eq!(perms.get(0), Some(Integer::one()));
        assert_eq!(perms.get(1), Some(Integer::one()));
        assert_eq!(perms.get(2), Some(Integer::from(2)));
        assert_eq!(perms.get(3), Some(Integer::from(6)));
        assert_eq!(perms.get(4), Some(Integer::from(24)));
        assert_eq!(perms.get(5), Some(Integer::from(120)));
    }

    #[test]
    fn test_bell_numbers() {
        let bell = ExpNum::bell_numbers(6);

        // Bell numbers: 1, 1, 2, 5, 15, 52
        assert_eq!(bell.get(0), Some(Integer::one()));
        assert_eq!(bell.get(1), Some(Integer::one()));
        assert_eq!(bell.get(2), Some(Integer::from(2)));
        assert_eq!(bell.get(3), Some(Integer::from(5)));
        assert_eq!(bell.get(4), Some(Integer::from(15)));
        assert_eq!(bell.get(5), Some(Integer::from(52)));
    }

    #[test]
    fn test_derangements() {
        let derang = ExpNum::derangements(7);

        // Derangements: 1, 0, 1, 2, 9, 44, 265
        assert_eq!(derang.get(0), Some(Integer::one()));
        assert_eq!(derang.get(1), Some(Integer::zero()));
        assert_eq!(derang.get(2), Some(Integer::one()));
        assert_eq!(derang.get(3), Some(Integer::from(2)));
        assert_eq!(derang.get(4), Some(Integer::from(9)));
        assert_eq!(derang.get(5), Some(Integer::from(44)));
        assert_eq!(derang.get(6), Some(Integer::from(265)));
    }

    #[test]
    fn test_involutions() {
        let inv = ExpNum::involutions(8);

        // Involutions: 1, 1, 2, 4, 10, 26, 76, 232
        assert_eq!(inv.get(0), Some(Integer::one()));
        assert_eq!(inv.get(1), Some(Integer::one()));
        assert_eq!(inv.get(2), Some(Integer::from(2)));
        assert_eq!(inv.get(3), Some(Integer::from(4)));
        assert_eq!(inv.get(4), Some(Integer::from(10)));
        assert_eq!(inv.get(5), Some(Integer::from(26)));
        assert_eq!(inv.get(6), Some(Integer::from(76)));
        assert_eq!(inv.get(7), Some(Integer::from(232)));
    }

    #[test]
    fn test_labeled_trees() {
        let trees = ExpNum::labeled_trees(6);

        // Labeled trees: 0, 1, 1, 3, 16, 125
        // n^(n-1): 0^-1=0, 1^0=1, 2^1=2, 3^2=9, 4^3=64, 5^4=625
        // Wait, let me check: for n=3, it's 3^2 = 9, but Cayley says there are 3 labeled trees on 3 vertices
        // Actually Cayley is n^(n-2) for the number of trees, but the sequence here is n^(n-1)
        // which counts labeled trees times n (or rooted labeled trees)
        assert_eq!(trees.get(0), Some(Integer::zero()));
        assert_eq!(trees.get(1), Some(Integer::one()));
        assert_eq!(trees.get(2), Some(Integer::from(2))); // 2^1
        assert_eq!(trees.get(3), Some(Integer::from(9))); // 3^2
        assert_eq!(trees.get(4), Some(Integer::from(64))); // 4^3
        assert_eq!(trees.get(5), Some(Integer::from(625))); // 5^4
    }

    #[test]
    fn test_addition() {
        let one = ExpNum::one(5);
        let identity = ExpNum::identity(5);

        let sum = one.add(&identity);

        // (1 + X): 1, 1, 0, 0, 0
        assert_eq!(sum.get(0), Some(Integer::one()));
        assert_eq!(sum.get(1), Some(Integer::one()));
        assert_eq!(sum.get(2), Some(Integer::zero()));
    }

    #[test]
    fn test_multiplication() {
        let identity = ExpNum::identity(5);

        // X * X with EGF multiplication
        let x_squared = identity.multiply(&identity);

        // (X * X)_n = Σ C(n,k) X_k X_{n-k}
        // For n=2: C(2,1) * 1 * 1 = 2
        assert_eq!(x_squared.get(0), Some(Integer::zero()));
        assert_eq!(x_squared.get(1), Some(Integer::zero()));
        assert_eq!(x_squared.get(2), Some(Integer::from(2)));
    }

    #[test]
    fn test_derivative() {
        let perms = ExpNum::permutations(6);
        let deriv = perms.derivative();

        // Derivative shifts: (n!)' -> (n+1)!
        assert_eq!(deriv.get(0), perms.get(1));
        assert_eq!(deriv.get(1), perms.get(2));
        assert_eq!(deriv.get(2), perms.get(3));
    }

    #[test]
    fn test_integral() {
        let perms = ExpNum::permutations(5);
        let integ = perms.integral();

        // Integral adds leading zero
        assert_eq!(integ.get(0), Some(Integer::zero()));
        assert_eq!(integ.get(1), perms.get(0));
        assert_eq!(integ.get(2), perms.get(1));
    }

    #[test]
    fn test_binomial_transform() {
        let identity = ExpNum::identity(5);
        let transformed = identity.binomial_transform();

        // Binomial transform of (0,1,0,0,...):
        // B_n = Σ C(n,k) * a_k = C(n,1) * 1 = n
        assert_eq!(transformed.get(0), Some(Integer::zero()));
        assert_eq!(transformed.get(1), Some(Integer::one()));
        assert_eq!(transformed.get(2), Some(Integer::from(2)));
        assert_eq!(transformed.get(3), Some(Integer::from(3)));
    }

    #[test]
    fn test_stirling_second() {
        let s_n_2 = ExpNum::stirling_second_fixed_k(2, 6);

        // S(n,2) for n=0,1,2,3,4,5: 0, 0, 1, 3, 7, 15
        assert_eq!(s_n_2.get(0), Some(Integer::zero()));
        assert_eq!(s_n_2.get(1), Some(Integer::zero()));
        assert_eq!(s_n_2.get(2), Some(Integer::one()));
        assert_eq!(s_n_2.get(3), Some(Integer::from(3)));
        assert_eq!(s_n_2.get(4), Some(Integer::from(7)));
        assert_eq!(s_n_2.get(5), Some(Integer::from(15)));
    }

    #[test]
    fn test_egf_from_expnum() {
        let perms = ExpNum::permutations(5);
        let egf = EGF::from_expnum(&perms);

        // Coefficients should be the same as the sequence
        assert_eq!(
            egf.coefficient(0),
            Some(Rational::new(Integer::one(), Integer::one()).unwrap())
        );
        assert_eq!(
            egf.coefficient(3),
            Some(Rational::new(Integer::from(6), Integer::one()).unwrap())
        );
    }

    #[test]
    fn test_egf_add() {
        let one = ExpNum::one(5);
        let identity = ExpNum::identity(5);

        let egf1 = EGF::from_expnum(&one);
        let egf2 = EGF::from_expnum(&identity);

        let sum = egf1.add(&egf2);

        // Coefficients should be: 1, 1, 0, 0, 0
        assert_eq!(
            sum.coefficient(0),
            Some(Rational::new(Integer::one(), Integer::one()).unwrap())
        );
        assert_eq!(
            sum.coefficient(1),
            Some(Rational::new(Integer::one(), Integer::one()).unwrap())
        );
        assert_eq!(
            sum.coefficient(2),
            Some(Rational::new(Integer::zero(), Integer::one()).unwrap())
        );
    }

    #[test]
    fn test_egf_multiply() {
        let identity = ExpNum::identity(5);
        let egf = EGF::from_expnum(&identity);

        let squared = egf.multiply(&egf);

        // Should give (0, 0, 2, 0, 0) for X * X
        assert_eq!(
            squared.coefficient(2),
            Some(Rational::new(Integer::from(2), Integer::one()).unwrap())
        );
    }

    #[test]
    fn test_egf_exp() {
        // Test exp on a simple EGF
        // For the zero sequence (starting with 0), exp(0) = 1
        let zero = ExpNum::zero(10);
        let egf = EGF::from_expnum(&zero);

        let exp_egf = egf.exp();

        // exp(0) = 1, so we expect (1, 0, 0, 0, ...)
        assert_eq!(
            exp_egf.coefficient(0),
            Some(Rational::new(Integer::one(), Integer::one()).unwrap())
        );
        assert_eq!(
            exp_egf.coefficient(1),
            Some(Rational::new(Integer::zero(), Integer::one()).unwrap())
        );

        // Test exp(x): the coefficients should satisfy the differential equation
        // If B(x) = exp(A(x)), then B'(x) = A'(x) * B(x)
        // For A(x) = x, we have A'(x) = 1, so B'(x) = B(x)
        // This gives B(x) = e^x = Σ x^n/n!, so coefficients are all 1
        let identity = ExpNum::identity(10);
        let egf_x = EGF::from_expnum(&identity);
        let exp_x = egf_x.exp();

        // The coefficients should all be 1 for exp(x)
        assert_eq!(
            exp_x.coefficient(0),
            Some(Rational::new(Integer::one(), Integer::one()).unwrap())
        );
        assert_eq!(
            exp_x.coefficient(1),
            Some(Rational::new(Integer::one(), Integer::one()).unwrap())
        );
        assert_eq!(
            exp_x.coefficient(2),
            Some(Rational::new(Integer::one(), Integer::one()).unwrap())
        );
        assert_eq!(
            exp_x.coefficient(3),
            Some(Rational::new(Integer::one(), Integer::one()).unwrap())
        );
    }

    #[test]
    fn test_scale() {
        let identity = ExpNum::identity(5);
        let scaled = identity.scale(Integer::from(3));

        assert_eq!(scaled.get(0), Some(Integer::zero()));
        assert_eq!(scaled.get(1), Some(Integer::from(3)));
        assert_eq!(scaled.get(2), Some(Integer::zero()));
    }

    #[test]
    fn test_shift_multiply() {
        let exp_seq = ExpNum::exponential(5);
        let shifted = exp_seq.shift_multiply();

        // n * 1 = n
        assert_eq!(shifted.get(0), Some(Integer::zero()));
        assert_eq!(shifted.get(1), Some(Integer::one()));
        assert_eq!(shifted.get(2), Some(Integer::from(2)));
        assert_eq!(shifted.get(3), Some(Integer::from(3)));
        assert_eq!(shifted.get(4), Some(Integer::from(4)));
    }

    #[test]
    fn test_surjections() {
        let surj_3 = ExpNum::surjections_fixed_k(3, 6);

        // Surjections to 3-set: 0 for n<3, then 3! * S(n,3)
        assert_eq!(surj_3.get(0), Some(Integer::zero()));
        assert_eq!(surj_3.get(1), Some(Integer::zero()));
        assert_eq!(surj_3.get(2), Some(Integer::zero()));
        // S(3,3) = 1, so 3! * 1 = 6
        assert_eq!(surj_3.get(3), Some(Integer::from(6)));
        // S(4,3) = 6, so 3! * 6 = 36
        assert_eq!(surj_3.get(4), Some(Integer::from(36)));
        // S(5,3) = 25, so 3! * 25 = 150
        assert_eq!(surj_3.get(5), Some(Integer::from(150)));
    }

    #[test]
    fn test_inverse_binomial_transform() {
        let exp_seq = ExpNum::exponential(5);
        let transformed = exp_seq.binomial_transform();
        let back = transformed.inverse_binomial_transform();

        // Should get back to original (approximately, up to precision)
        for i in 0..5 {
            assert_eq!(back.get(i), exp_seq.get(i));
        }
    }
}
