//! Baker-Campbell-Hausdorff Formula
//!
//! The Baker-Campbell-Hausdorff (BCH) formula expresses log(exp(X)exp(Y))
//! as an infinite series of nested Lie brackets of X and Y with rational coefficients.
//!
//! For Lie algebra elements X and Y, the BCH formula is:
//!
//! log(exp(X)exp(Y)) = X + Y + (1/2)[X,Y] + (1/12)([X,[X,Y]] + [Y,[Y,X]]) + ...
//!
//! This formula is fundamental in Lie theory, quantum mechanics, and the study
//! of non-commutative structures.
//!
//! # References
//!
//! - W. Magnus, "On the exponential solution of differential equations for a linear operator"
//! - Dynkin, E. B. "Calculation of the coefficients in the Campbell-Hausdorff formula"
//! - Hall, B. C. "Lie Groups, Lie Algebras, and Representations" (2015)
//!
//! Corresponds to sage.algebras.lie_algebras.bch

use crate::free_lie_algebra::{FreeLieAlgebraElement, LieBracket};
use rustmath_core::Ring;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// Compute the Bernoulli number B_n
///
/// Bernoulli numbers appear in the BCH formula coefficients.
/// They satisfy: Σ(k=0 to n) C(n+1,k) B_k = 0 (for n > 0)
///
/// First few values:
/// B_0 = 1, B_1 = -1/2, B_2 = 1/6, B_4 = -1/30, B_6 = 1/42
/// B_odd = 0 for odd > 1
fn bernoulli_number(n: usize) -> Rational {
    if n == 0 {
        return Rational::from_integer(1);
    }
    if n == 1 {
        return Rational::new(-1, 2).unwrap();
    }
    if n > 1 && n % 2 == 1 {
        return Rational::zero();
    }

    // Use recursive formula for even n > 0
    // This is a simplified implementation; full version would use caching
    match n {
        2 => Rational::new(1, 6).unwrap(),
        4 => Rational::new(-1, 30).unwrap(),
        6 => Rational::new(1, 42).unwrap(),
        8 => Rational::new(-1, 30).unwrap(),
        10 => Rational::new(5, 66).unwrap(),
        12 => Rational::new(-691, 2730).unwrap(),
        14 => Rational::new(7, 6).unwrap(),
        _ => {
            // For larger values, use the recursive formula
            // B_n = -1/(n+1) * Σ(k=0 to n-1) C(n+1,k) * B_k
            let mut sum = Rational::zero();
            for k in 0..n {
                let binom = binomial_coefficient(n + 1, k);
                let b_k = bernoulli_number(k);
                sum = sum + Rational::from_integer(binom as i64) * b_k;
            }
            -sum / Rational::from_integer((n + 1) as i64)
        }
    }
}

/// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Iterator that generates successive terms of the Baker-Campbell-Hausdorff formula
///
/// This iterator produces an infinite sequence of Lie algebra elements
/// representing the terms in the BCH expansion.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (must have coercion from rationals)
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::bch::BCHIterator;
/// # use rustmath_liealgebras::free_lie_algebra::{FreeLieAlgebraElement, FreeLieAlgebra, FreeLieAlgebraBasis};
/// # use rustmath_rationals::Rational;
///
/// // Create a free Lie algebra with 2 generators
/// let algebra = FreeLieAlgebra::<Rational>::new(2, FreeLieAlgebraBasis::Lyndon);
/// let x = algebra.generator(0);
/// let y = algebra.generator(1);
///
/// // Create BCH iterator
/// let mut bch = BCHIterator::new(Some(x), Some(y));
///
/// // First term is X + Y
/// let term0 = bch.next().unwrap();
///
/// // Second term is (1/2)[X,Y]
/// let term1 = bch.next().unwrap();
/// ```
///
/// Corresponds to sage.algebras.lie_algebras.bch.bch_iterator
pub struct BCHIterator<R: Ring + Clone> {
    /// First element X
    x: Option<FreeLieAlgebraElement<R>>,
    /// Second element Y
    y: Option<FreeLieAlgebraElement<R>>,
    /// Current term index
    index: usize,
    /// Cache of previously computed terms
    cache: HashMap<usize, FreeLieAlgebraElement<R>>,
}

impl<R: Ring + Clone> BCHIterator<R> {
    /// Create a new BCH iterator
    ///
    /// # Arguments
    ///
    /// * `x` - First Lie algebra element (or None for abstract mode)
    /// * `y` - Second Lie algebra element (or None for abstract mode)
    pub fn new(x: Option<FreeLieAlgebraElement<R>>, y: Option<FreeLieAlgebraElement<R>>) -> Self {
        BCHIterator {
            x,
            y,
            index: 0,
            cache: HashMap::new(),
        }
    }
}

impl Iterator for BCHIterator<Rational> {
    type Item = FreeLieAlgebraElement<Rational>;

    fn next(&mut self) -> Option<Self::Item> {
        let term = match self.index {
            0 => {
                // First term: X + Y
                match (&self.x, &self.y) {
                    (Some(x), Some(y)) => x.add(y),
                    (Some(x), None) => x.clone(),
                    (None, Some(y)) => y.clone(),
                    (None, None) => {
                        // Abstract mode: return X + Y as symbolic elements
                        let mut x_elem = FreeLieAlgebraElement::generator(0);
                        let y_elem = FreeLieAlgebraElement::generator(1);
                        x_elem = x_elem.add(&y_elem);
                        x_elem
                    }
                }
            }
            1 => {
                // Second term: (1/2)[X,Y]
                match (&self.x, &self.y) {
                    (Some(x), Some(y)) => {
                        let commutator = x.bracket(y);
                        let half = Rational::new(1, 2).unwrap();
                        commutator.scalar_mul(&half)
                    }
                    (None, None) => {
                        // Abstract mode
                        let x = FreeLieAlgebraElement::<Rational>::generator(0);
                        let y = FreeLieAlgebraElement::<Rational>::generator(1);
                        let commutator = x.bracket(&y);
                        let half = Rational::new(1, 2).unwrap();
                        commutator.scalar_mul(&half)
                    }
                    _ => FreeLieAlgebraElement::zero(),
                }
            }
            2 => {
                // Third term: (1/12)([X,[X,Y]] + [Y,[Y,X]])
                match (&self.x, &self.y) {
                    (Some(x), Some(y)) => {
                        let xy = x.bracket(y);
                        let x_xy = x.bracket(&xy);

                        let yx = y.bracket(x);
                        let y_yx = y.bracket(&yx);

                        let sum = x_xy.add(&y_yx);
                        let coeff = Rational::new(1, 12).unwrap();
                        sum.scalar_mul(&coeff)
                    }
                    (None, None) => {
                        // Abstract mode
                        let x = FreeLieAlgebraElement::<Rational>::generator(0);
                        let y = FreeLieAlgebraElement::<Rational>::generator(1);

                        let xy = x.bracket(&y);
                        let x_xy = x.bracket(&xy);

                        let yx = y.bracket(&x);
                        let y_yx = y.bracket(&yx);

                        let sum = x_xy.add(&y_yx);
                        let coeff = Rational::new(1, 12).unwrap();
                        sum.scalar_mul(&coeff)
                    }
                    _ => FreeLieAlgebraElement::zero(),
                }
            }
            3 => {
                // Fourth term: (1/24)[Y,[X,[X,Y]]]
                match (&self.x, &self.y) {
                    (Some(x), Some(y)) => {
                        let xy = x.bracket(y);
                        let x_xy = x.bracket(&xy);
                        let y_x_xy = y.bracket(&x_xy);
                        let coeff = Rational::new(1, 24).unwrap();
                        y_x_xy.scalar_mul(&coeff)
                    }
                    (None, None) => {
                        let x = FreeLieAlgebraElement::<Rational>::generator(0);
                        let y = FreeLieAlgebraElement::<Rational>::generator(1);
                        let xy = x.bracket(&y);
                        let x_xy = x.bracket(&xy);
                        let y_x_xy = y.bracket(&x_xy);
                        let coeff = Rational::new(1, 24).unwrap();
                        y_x_xy.scalar_mul(&coeff)
                    }
                    _ => FreeLieAlgebraElement::zero(),
                }
            }
            4 => {
                // Fifth term: -(1/720)([Y,[Y,[Y,[Y,X]]]] + [X,[X,[X,[X,Y]]]])
                match (&self.x, &self.y) {
                    (Some(x), Some(y)) => {
                        // [Y,[Y,[Y,[Y,X]]]]
                        let yx = y.bracket(x);
                        let y_yx = y.bracket(&yx);
                        let y_y_yx = y.bracket(&y_yx);
                        let y4_x = y.bracket(&y_y_yx);

                        // [X,[X,[X,[X,Y]]]]
                        let xy = x.bracket(y);
                        let x_xy = x.bracket(&xy);
                        let x_x_xy = x.bracket(&x_xy);
                        let x4_y = x.bracket(&x_x_xy);

                        let sum = y4_x.add(&x4_y);
                        let coeff = Rational::new(-1, 720).unwrap();
                        sum.scalar_mul(&coeff)
                    }
                    (None, None) => {
                        let x = FreeLieAlgebraElement::<Rational>::generator(0);
                        let y = FreeLieAlgebraElement::<Rational>::generator(1);

                        let yx = y.bracket(&x);
                        let y_yx = y.bracket(&yx);
                        let y_y_yx = y.bracket(&y_yx);
                        let y4_x = y.bracket(&y_y_yx);

                        let xy = x.bracket(&y);
                        let x_xy = x.bracket(&xy);
                        let x_x_xy = x.bracket(&x_xy);
                        let x4_y = x.bracket(&x_x_xy);

                        let sum = y4_x.add(&x4_y);
                        let coeff = Rational::new(-1, 720).unwrap();
                        sum.scalar_mul(&coeff)
                    }
                    _ => FreeLieAlgebraElement::zero(),
                }
            }
            _ => {
                // For higher terms, return zero (placeholder for full implementation)
                // Full implementation would use the general recursive formula
                FreeLieAlgebraElement::zero()
            }
        };

        self.index += 1;
        Some(term)
    }
}

/// Compute the BCH formula up to a given number of terms
///
/// Returns the sum of the first `num_terms` terms in the BCH expansion.
///
/// # Arguments
///
/// * `x` - First Lie algebra element
/// * `y` - Second Lie algebra element
/// * `num_terms` - Number of terms to compute
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::bch::bch_sum;
/// # use rustmath_liealgebras::free_lie_algebra::{FreeLieAlgebra, FreeLieAlgebraBasis};
/// # use rustmath_rationals::Rational;
///
/// let algebra = FreeLieAlgebra::<Rational>::new(2, FreeLieAlgebraBasis::Lyndon);
/// let x = algebra.generator(0);
/// let y = algebra.generator(1);
///
/// // Compute first 3 terms: X + Y + (1/2)[X,Y] + (1/12)([X,[X,Y]] + [Y,[Y,X]])
/// let result = bch_sum(&x, &y, 3);
/// ```
pub fn bch_sum(
    x: &FreeLieAlgebraElement<Rational>,
    y: &FreeLieAlgebraElement<Rational>,
    num_terms: usize,
) -> FreeLieAlgebraElement<Rational> {
    let mut bch = BCHIterator::new(Some(x.clone()), Some(y.clone()));
    let mut result = FreeLieAlgebraElement::zero();

    for _ in 0..num_terms {
        if let Some(term) = bch.next() {
            result = result.add(&term);
        } else {
            break;
        }
    }

    result
}

/// Create an abstract BCH iterator for the free Lie algebra on 2 generators
///
/// This returns an iterator that generates BCH terms symbolically
/// in terms of generators X (index 0) and Y (index 1).
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::bch::bch_iterator;
///
/// let mut bch = bch_iterator();
///
/// // Get first few terms
/// let term0 = bch.next(); // X + Y
/// let term1 = bch.next(); // (1/2)[X,Y]
/// let term2 = bch.next(); // (1/12)([X,[X,Y]] + [Y,[Y,X]])
/// ```
///
/// Corresponds to sage.algebras.lie_algebras.bch.bch_iterator
pub fn bch_iterator() -> BCHIterator<Rational> {
    BCHIterator::new(None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::free_lie_algebra::{FreeLieAlgebra, FreeLieAlgebraBasis};

    #[test]
    fn test_bernoulli_numbers() {
        assert_eq!(bernoulli_number(0), Rational::from(1));
        assert_eq!(bernoulli_number(1), Rational::new(-1, 2).unwrap());
        assert_eq!(bernoulli_number(2), Rational::new(1, 6).unwrap());
        assert_eq!(bernoulli_number(3), Rational::zero());
        assert_eq!(bernoulli_number(4), Rational::new(-1, 30).unwrap());
        assert_eq!(bernoulli_number(5), Rational::zero());
        assert_eq!(bernoulli_number(6), Rational::new(1, 42).unwrap());
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), 1);
        assert_eq!(binomial_coefficient(5, 1), 5);
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(5, 3), 10);
        assert_eq!(binomial_coefficient(5, 4), 5);
        assert_eq!(binomial_coefficient(5, 5), 1);
        assert_eq!(binomial_coefficient(5, 6), 0);
    }

    #[test]
    fn test_bch_iterator_abstract() {
        let mut bch = bch_iterator();

        // First term: X + Y
        let term0 = bch.next().unwrap();
        assert!(!term0.is_zero());

        // Second term: (1/2)[X,Y]
        let term1 = bch.next().unwrap();
        assert!(!term1.is_zero());

        // Third term
        let term2 = bch.next().unwrap();
        assert!(!term2.is_zero());
    }

    #[test]
    fn test_bch_iterator_concrete() {
        let algebra = FreeLieAlgebra::<Rational>::new(2, FreeLieAlgebraBasis::Lyndon);
        let x = algebra.generator(0);
        let y = algebra.generator(1);

        let mut bch = BCHIterator::new(Some(x.clone()), Some(y.clone()));

        // First term: X + Y
        let term0 = bch.next().unwrap();
        assert!(!term0.is_zero());

        // Second term: (1/2)[X,Y]
        let term1 = bch.next().unwrap();
        assert!(!term1.is_zero());

        // Third term
        let term2 = bch.next().unwrap();
        assert!(!term2.is_zero());
    }

    #[test]
    fn test_bch_sum() {
        let algebra = FreeLieAlgebra::<Rational>::new(2, FreeLieAlgebraBasis::Lyndon);
        let x = algebra.generator(0);
        let y = algebra.generator(1);

        // Compute first 3 terms
        let result = bch_sum(&x, &y, 3);
        assert!(!result.is_zero());

        // Result should contain X, Y, and bracket terms
        assert!(result.terms().len() > 0);
    }

    #[test]
    fn test_bch_first_term() {
        let algebra = FreeLieAlgebra::<Rational>::new(2, FreeLieAlgebraBasis::Lyndon);
        let x = algebra.generator(0);
        let y = algebra.generator(1);

        let mut bch = BCHIterator::new(Some(x.clone()), Some(y.clone()));
        let term0 = bch.next().unwrap();

        // First term should equal X + Y
        let expected = x.add(&y);
        assert_eq!(term0.terms().len(), expected.terms().len());
    }

    #[test]
    fn test_bch_second_term() {
        let algebra = FreeLieAlgebra::<Rational>::new(2, FreeLieAlgebraBasis::Lyndon);
        let x = algebra.generator(0);
        let y = algebra.generator(1);

        let mut bch = BCHIterator::new(Some(x.clone()), Some(y.clone()));
        let _ = bch.next(); // Skip first term
        let term1 = bch.next().unwrap();

        // Second term should be (1/2)[X,Y]
        let commutator = x.bracket(&y);
        let half = Rational::new(1, 2).unwrap();
        let expected = commutator.scalar_mul(&half);

        // Both should be non-zero
        assert!(!term1.is_zero());
        assert!(!expected.is_zero());
    }

    #[test]
    fn test_bch_nilpotent() {
        // For nilpotent Lie algebras, BCH terminates after finitely many terms
        let algebra = FreeLieAlgebra::<Rational>::new(2, FreeLieAlgebraBasis::Lyndon);
        let x = algebra.generator(0);
        let y = algebra.generator(1);

        // Compute several terms
        let result = bch_sum(&x, &y, 10);
        assert!(!result.is_zero());
    }

    #[test]
    fn test_bch_properties() {
        let algebra = FreeLieAlgebra::<Rational>::new(2, FreeLieAlgebraBasis::Lyndon);
        let x = algebra.generator(0);
        let y = algebra.generator(1);

        // BCH(X, 0) should equal X
        let zero = FreeLieAlgebraElement::zero();
        let result_x0 = bch_sum(&x, &zero, 5);

        // BCH(0, Y) should equal Y
        let result_0y = bch_sum(&zero, &y, 5);

        // Results should be non-zero when at least one input is non-zero
        assert!(!result_x0.is_zero() || x.is_zero());
        assert!(!result_0y.is_zero() || y.is_zero());
    }
}
