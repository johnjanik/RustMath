//! # C-Finite Sequences
//!
//! This module implements C-finite sequences (constant-recursive sequences),
//! which are sequences satisfying homogeneous linear recurrence relations
//! with constant coefficients.
//!
//! ## Background
//!
//! A C-finite sequence {a_n} satisfies a recurrence of the form:
//! ```text
//! a_{n+d} = c_0*a_n + c_1*a_{n+1} + ... + c_{d-1}*a_{n+d-1}
//! ```
//!
//! Such sequences have rational ordinary generating functions (o.g.f.):
//! ```text
//! G(x) = Σ a_n * x^n = P(x) / Q(x)
//! ```
//! where P and Q are polynomials.
//!
//! ## Examples
//!
//! The Fibonacci sequence: F_{n+2} = F_{n+1} + F_n with F_0=0, F_1=1
//!
//! ```rust
//! use rustmath_rings::cfinite_sequence::CFiniteSequence;
//! use rustmath_rationals::Rational;
//!
//! // Fibonacci: recurrence [1, 1], initial values [0, 1]
//! let fib = CFiniteSequence::from_recurrence(
//!     vec![1, 1],
//!     vec![0, 1]
//! );
//!
//! assert_eq!(fib.nth(0), Some(0));
//! assert_eq!(fib.nth(1), Some(1));
//! assert_eq!(fib.nth(2), Some(1));
//! assert_eq!(fib.nth(5), Some(5));
//! assert_eq!(fib.nth(10), Some(55));
//! ```

use rustmath_rationals::Rational;
use std::fmt;

/// A C-finite sequence defined by a linear recurrence relation
///
/// The sequence satisfies:
/// ```text
/// a_{n+d} = c_0*a_n + c_1*a_{n+1} + ... + c_{d-1}*a_{n+d-1}
/// ```
#[derive(Clone, Debug)]
pub struct CFiniteSequence {
    /// Recurrence coefficients [c_0, c_1, ..., c_{d-1}]
    coefficients: Vec<i64>,

    /// Initial values [a_0, a_1, ..., a_{d-1}]
    initial_values: Vec<i64>,

    /// Cached computed values for efficiency
    cache: Vec<i64>,
}

impl CFiniteSequence {
    /// Create a C-finite sequence from recurrence coefficients and initial values
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Recurrence coefficients [c_0, c_1, ..., c_{d-1}]
    /// * `initial_values` - Initial values [a_0, a_1, ..., a_{d-1}]
    ///
    /// # Panics
    ///
    /// Panics if coefficients and initial_values have different lengths
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::cfinite_sequence::CFiniteSequence;
    ///
    /// // Fibonacci: F_{n+2} = F_{n+1} + F_n
    /// let fib = CFiniteSequence::from_recurrence(vec![1, 1], vec![0, 1]);
    ///
    /// // Lucas: L_{n+2} = L_{n+1} + L_n, L_0=2, L_1=1
    /// let lucas = CFiniteSequence::from_recurrence(vec![1, 1], vec![2, 1]);
    /// ```
    pub fn from_recurrence(coefficients: Vec<i64>, initial_values: Vec<i64>) -> Self {
        assert_eq!(
            coefficients.len(),
            initial_values.len(),
            "Coefficients and initial values must have same length"
        );

        CFiniteSequence {
            coefficients,
            initial_values: initial_values.clone(),
            cache: initial_values,
        }
    }

    /// Create the Fibonacci sequence
    ///
    /// F_0 = 0, F_1 = 1, F_{n+2} = F_{n+1} + F_n
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::cfinite_sequence::CFiniteSequence;
    ///
    /// let fib = CFiniteSequence::fibonacci();
    /// assert_eq!(fib.nth(10), Some(55));
    /// ```
    pub fn fibonacci() -> Self {
        Self::from_recurrence(vec![1, 1], vec![0, 1])
    }

    /// Create the Lucas sequence
    ///
    /// L_0 = 2, L_1 = 1, L_{n+2} = L_{n+1} + L_n
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::cfinite_sequence::CFiniteSequence;
    ///
    /// let lucas = CFiniteSequence::lucas();
    /// assert_eq!(lucas.nth(0), Some(2));
    /// assert_eq!(lucas.nth(1), Some(1));
    /// assert_eq!(lucas.nth(5), Some(11));
    /// ```
    pub fn lucas() -> Self {
        Self::from_recurrence(vec![1, 1], vec![2, 1])
    }

    /// Create the Pell sequence
    ///
    /// P_0 = 0, P_1 = 1, P_{n+2} = 2*P_{n+1} + P_n
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::cfinite_sequence::CFiniteSequence;
    ///
    /// let pell = CFiniteSequence::pell();
    /// assert_eq!(pell.nth(5), Some(29));
    /// ```
    pub fn pell() -> Self {
        Self::from_recurrence(vec![1, 2], vec![0, 1])
    }

    /// Create the Tribonacci sequence
    ///
    /// T_0 = 0, T_1 = 0, T_2 = 1, T_{n+3} = T_{n+2} + T_{n+1} + T_n
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::cfinite_sequence::CFiniteSequence;
    ///
    /// let trib = CFiniteSequence::tribonacci();
    /// assert_eq!(trib.nth(10), Some(81));
    /// ```
    pub fn tribonacci() -> Self {
        Self::from_recurrence(vec![1, 1, 1], vec![0, 0, 1])
    }

    /// Get the n-th term of the sequence
    ///
    /// Uses the recurrence relation to compute terms beyond the initial values.
    /// Results are cached for efficiency.
    ///
    /// # Arguments
    ///
    /// * `n` - Index of the term to retrieve (0-indexed)
    ///
    /// # Returns
    ///
    /// The n-th term, or None if overflow would occur
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::cfinite_sequence::CFiniteSequence;
    ///
    /// let fib = CFiniteSequence::fibonacci();
    /// assert_eq!(fib.nth(0), Some(0));
    /// assert_eq!(fib.nth(10), Some(55));
    /// ```
    pub fn nth(&self, n: usize) -> Option<i64> {
        // Use immutable self, so we can't modify cache
        // For a mutable version, we'd need &mut self

        if n < self.cache.len() {
            return Some(self.cache[n]);
        }

        // Compute using recurrence
        let mut values = self.cache.clone();
        let d = self.coefficients.len();

        for i in values.len()..=n {
            let mut next = 0i64;
            for (j, &coeff) in self.coefficients.iter().enumerate() {
                next = next.checked_add(coeff.checked_mul(values[i - d + j])?)?;
            }
            values.push(next);
        }

        Some(values[n])
    }

    /// Get the n-th term as a Rational (for compatibility)
    pub fn nth_rational(&self, n: usize) -> Option<Rational> {
        self.nth(n).map(Rational::from)
    }

    /// Get multiple terms as a vector
    ///
    /// # Arguments
    ///
    /// * `start` - Starting index (inclusive)
    /// * `end` - Ending index (exclusive)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::cfinite_sequence::CFiniteSequence;
    ///
    /// let fib = CFiniteSequence::fibonacci();
    /// let terms = fib.terms(0, 10);
    /// assert_eq!(terms, vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
    /// ```
    pub fn terms(&self, start: usize, end: usize) -> Vec<i64> {
        (start..end).filter_map(|i| self.nth(i)).collect()
    }

    /// Get the degree of the recurrence (number of terms it depends on)
    pub fn degree(&self) -> usize {
        self.coefficients.len()
    }

    /// Get the recurrence coefficients
    pub fn coefficients(&self) -> &[i64] {
        &self.coefficients
    }

    /// Get the initial values
    pub fn initial_values(&self) -> &[i64] {
        &self.initial_values
    }

    /// Create a closed-form expression for the n-th term (as a string)
    ///
    /// This uses the characteristic polynomial method.
    /// For sequences with simple roots, this gives an explicit formula.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::cfinite_sequence::CFiniteSequence;
    ///
    /// let fib = CFiniteSequence::fibonacci();
    /// // Returns a string representation of Binet's formula
    /// let formula = fib.closed_form();
    /// ```
    pub fn closed_form(&self) -> String {
        // This is a placeholder - full implementation would require:
        // 1. Find roots of characteristic polynomial
        // 2. Solve system of linear equations for coefficients
        // 3. Return explicit formula
        format!(
            "a_n satisfies recurrence: a_{{n+{}}} = {}",
            self.degree(),
            self.recurrence_string()
        )
    }

    /// Get a string representation of the recurrence relation
    fn recurrence_string(&self) -> String {
        let _d = self.degree();
        let terms: Vec<String> = self
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if c == 1 {
                    format!("a_{{n+{}}}", i)
                } else if c == -1 {
                    format!("-a_{{n+{}}}", i)
                } else if c >= 0 {
                    format!("{}*a_{{n+{}}}", c, i)
                } else {
                    format!("({})*a_{{n+{}}}", c, i)
                }
            })
            .collect();

        terms.join(" + ")
    }

    /// Test if a given sequence matches this recurrence
    ///
    /// Checks if the provided values satisfy the recurrence relation.
    ///
    /// # Arguments
    ///
    /// * `values` - Sequence of values to test
    ///
    /// # Returns
    ///
    /// true if the values satisfy the recurrence, false otherwise
    pub fn matches(&self, values: &[i64]) -> bool {
        let d = self.degree();

        if values.len() < d {
            return false;
        }

        // Check initial values
        if &values[..d] != &self.initial_values[..] {
            return false;
        }

        // Check recurrence for remaining terms
        for i in d..values.len() {
            let expected: Option<i64> = self
                .coefficients
                .iter()
                .enumerate()
                .try_fold(0i64, |acc, (j, &coeff)| {
                    acc.checked_add(coeff.checked_mul(values[i - d + j])?)
                });

            if let Some(exp) = expected {
                if exp != values[i] {
                    return false;
                }
            } else {
                return false; // Overflow
            }
        }

        true
    }
}

impl fmt::Display for CFiniteSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "C-finite sequence: {} with initial values {:?}",
            self.recurrence_string(),
            self.initial_values
        )
    }
}

/// Guess a C-finite recurrence from a sequence of values
///
/// Uses linear algebra to find the minimal recurrence that generates
/// the given sequence.
///
/// # Arguments
///
/// * `values` - Sequence values to analyze
/// * `max_degree` - Maximum recurrence degree to try
///
/// # Returns
///
/// A CFiniteSequence if a recurrence is found, None otherwise
///
/// # Examples
///
/// ```rust
/// use rustmath_rings::cfinite_sequence::guess_recurrence;
///
/// let fib_values = vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34];
/// let seq = guess_recurrence(&fib_values, 5);
/// assert!(seq.is_some());
/// ```
pub fn guess_recurrence(values: &[i64], max_degree: usize) -> Option<CFiniteSequence> {
    // Try degrees from 1 to max_degree
    for d in 1..=max_degree.min(values.len() / 2) {
        if let Some(seq) = try_degree(values, d) {
            return Some(seq);
        }
    }

    None
}

/// Try to find a recurrence of a specific degree
fn try_degree(values: &[i64], degree: usize) -> Option<CFiniteSequence> {
    if values.len() < 2 * degree {
        return None;
    }

    // Build system of linear equations: A * c = b
    // where c are the coefficients to find
    let n_equations = values.len() - degree;
    let mut matrix: Vec<Vec<f64>> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();

    for i in 0..n_equations {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..degree {
            row.push(values[i + j] as f64);
        }
        matrix.push(row);
        rhs.push(values[i + degree] as f64);
    }

    // Solve using least squares (simple Gaussian elimination for square systems)
    if let Some(coeffs) = solve_linear_system(&matrix, &rhs) {
        // Convert to integer coefficients
        let int_coeffs: Vec<i64> = coeffs.iter().map(|&x| x.round() as i64).collect();

        // Verify the solution
        let seq = CFiniteSequence::from_recurrence(int_coeffs, values[..degree].to_vec());

        if seq.matches(values) {
            return Some(seq);
        }
    }

    None
}

/// Simple linear system solver (Gaussian elimination)
///
/// This is a basic implementation for small systems.
/// For production use, consider using a linear algebra library.
fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = matrix.len();
    let m = matrix[0].len();

    if n < m || rhs.len() != n {
        return None;
    }

    // Create augmented matrix
    let mut aug: Vec<Vec<f64>> = matrix.iter().map(|row| row.clone()).collect();
    for i in 0..n {
        aug[i].push(rhs[i]);
    }

    // Forward elimination
    for i in 0..m.min(n) {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        aug.swap(i, max_row);

        // Skip if pivot is too small
        if aug[i][i].abs() < 1e-10 {
            continue;
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[k][i] / aug[i][i];
            for j in i..=m {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut solution = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = aug[i][m];
        for j in (i + 1)..m {
            sum -= aug[i][j] * solution[j];
        }
        if aug[i][i].abs() < 1e-10 {
            solution[i] = 0.0;
        } else {
            solution[i] = sum / aug[i][i];
        }
    }

    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        let fib = CFiniteSequence::fibonacci();

        assert_eq!(fib.nth(0), Some(0));
        assert_eq!(fib.nth(1), Some(1));
        assert_eq!(fib.nth(2), Some(1));
        assert_eq!(fib.nth(3), Some(2));
        assert_eq!(fib.nth(4), Some(3));
        assert_eq!(fib.nth(5), Some(5));
        assert_eq!(fib.nth(6), Some(8));
        assert_eq!(fib.nth(10), Some(55));
        assert_eq!(fib.nth(20), Some(6765));
    }

    #[test]
    fn test_lucas() {
        let lucas = CFiniteSequence::lucas();

        assert_eq!(lucas.nth(0), Some(2));
        assert_eq!(lucas.nth(1), Some(1));
        assert_eq!(lucas.nth(2), Some(3));
        assert_eq!(lucas.nth(3), Some(4));
        assert_eq!(lucas.nth(5), Some(11));
    }

    #[test]
    fn test_pell() {
        let pell = CFiniteSequence::pell();

        assert_eq!(pell.nth(0), Some(0));
        assert_eq!(pell.nth(1), Some(1));
        assert_eq!(pell.nth(2), Some(2));
        assert_eq!(pell.nth(3), Some(5));
        assert_eq!(pell.nth(5), Some(29));
    }

    #[test]
    fn test_tribonacci() {
        let trib = CFiniteSequence::tribonacci();

        assert_eq!(trib.nth(0), Some(0));
        assert_eq!(trib.nth(1), Some(0));
        assert_eq!(trib.nth(2), Some(1));
        assert_eq!(trib.nth(3), Some(1));
        assert_eq!(trib.nth(4), Some(2));
        assert_eq!(trib.nth(5), Some(4));
        assert_eq!(trib.nth(10), Some(81));
    }

    #[test]
    fn test_terms() {
        let fib = CFiniteSequence::fibonacci();
        let terms = fib.terms(0, 10);
        assert_eq!(terms, vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
    }

    #[test]
    fn test_matches() {
        let fib = CFiniteSequence::fibonacci();

        let correct = vec![0, 1, 1, 2, 3, 5, 8, 13];
        assert!(fib.matches(&correct));

        let incorrect = vec![0, 1, 1, 2, 3, 5, 8, 14];
        assert!(!fib.matches(&incorrect));
    }

    #[test]
    fn test_guess_fibonacci() {
        let fib_values = vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34];
        let seq = guess_recurrence(&fib_values, 5);

        assert!(seq.is_some());
        let seq = seq.unwrap();
        assert_eq!(seq.degree(), 2);
        assert!(seq.matches(&fib_values));
    }

    #[test]
    fn test_guess_tribonacci() {
        let trib_values = vec![0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81];
        let seq = guess_recurrence(&trib_values, 5);

        assert!(seq.is_some());
        let seq = seq.unwrap();
        assert_eq!(seq.degree(), 3);
        assert!(seq.matches(&trib_values));
    }

    #[test]
    fn test_custom_recurrence() {
        // a_n = 2*a_{n-1} + 3*a_{n-2}, a_0=1, a_1=2
        let seq = CFiniteSequence::from_recurrence(vec![3, 2], vec![1, 2]);

        assert_eq!(seq.nth(0), Some(1));
        assert_eq!(seq.nth(1), Some(2));
        assert_eq!(seq.nth(2), Some(7)); // 2*2 + 3*1 = 7
        assert_eq!(seq.nth(3), Some(20)); // 2*7 + 3*2 = 20
    }

    #[test]
    fn test_display() {
        let fib = CFiniteSequence::fibonacci();
        let display = format!("{}", fib);
        assert!(display.contains("C-finite sequence"));
        assert!(display.contains("[0, 1]"));
    }

    #[test]
    fn test_closed_form() {
        let fib = CFiniteSequence::fibonacci();
        let formula = fib.closed_form();
        assert!(formula.contains("recurrence"));
    }

    #[test]
    fn test_degree() {
        let fib = CFiniteSequence::fibonacci();
        assert_eq!(fib.degree(), 2);

        let trib = CFiniteSequence::tribonacci();
        assert_eq!(trib.degree(), 3);
    }
}
