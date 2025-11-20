//! t-Sequences - t-analogues of integer sequences
//!
//! This module provides t-analogues of classical integer sequences, including:
//! - t-integers (also called t-numbers)
//! - t-factorials
//! - t-binomial coefficients
//! - t-Catalan numbers
//! - t-Fibonacci numbers
//! - t-Lucas numbers
//! - t-Stirling numbers
//! - t-Bell numbers
//! - t-Eulerian numbers
//!
//! # Mathematical Background
//!
//! The t-analogue is a generalization of integer sequences that interpolates
//! between classical sequences and their q-analogues. Like q-analogues, t-analogues
//! provide a one-parameter family of polynomials that reduce to the classical
//! sequences when t = 1.
//!
//! The t-analogue of an integer n is defined as:
//! ```text
//! [n]_t = (1 - t^n) / (1 - t) = 1 + t + t^2 + ... + t^{n-1}
//! ```
//!
//! When t = 1, we have [n]_t = n, recovering the classical integer.
//!
//! The t-factorial is:
//! ```text
//! [n]_t! = [1]_t · [2]_t · ... · [n]_t
//! ```
//!
//! The t-binomial coefficient is:
//! ```text
//! [n choose k]_t = [n]_t! / ([k]_t! · [n-k]_t!)
//! ```

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_polynomials::{Polynomial, UnivariatePolynomial};

/// Compute the t-integer (t-number) [n]_t = 1 + t + t^2 + ... + t^{n-1}
///
/// This is a polynomial in t with integer coefficients.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_integer;
///
/// // [0]_t = 0
/// let t0 = t_integer(0);
///
/// // [1]_t = 1
/// let t1 = t_integer(1);
///
/// // [3]_t = 1 + t + t^2
/// let t3 = t_integer(3);
/// ```
///
/// # Mathematical Definition
///
/// [n]_t = (1 - t^n) / (1 - t) = 1 + t + t^2 + ... + t^{n-1}
///
/// When t = 1, this reduces to n (the classical integer).
pub fn t_integer(n: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    // [n]_t = 1 + t + t^2 + ... + t^{n-1}
    // Coefficients: [1, 1, 1, ..., 1] (n times)
    let coeffs = vec![Integer::one(); n as usize];
    UnivariatePolynomial::new(coeffs)
}

/// Compute the t-factorial [n]_t! = [1]_t · [2]_t · ... · [n]_t
///
/// This is a polynomial in t with integer coefficients.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_factorial;
///
/// // [0]_t! = 1
/// let tf0 = t_factorial(0);
///
/// // [1]_t! = [1]_t = 1
/// let tf1 = t_factorial(1);
///
/// // [3]_t! = [1]_t · [2]_t · [3]_t = 1 · (1+t) · (1+t+t^2)
/// let tf3 = t_factorial(3);
/// ```
pub fn t_factorial(n: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    let mut result = UnivariatePolynomial::new(vec![Integer::one()]);

    for k in 1..=n {
        let t_k = t_integer(k);
        result = result * t_k;
    }

    result
}

/// Compute the t-binomial coefficient
///
/// [n choose k]_t = [n]_t! / ([k]_t! · [n-k]_t!)
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_binomial;
///
/// // [n choose 0]_t = 1
/// let tb = t_binomial(5, 0);
///
/// // [n choose n]_t = 1
/// let tb = t_binomial(5, 5);
///
/// // [5 choose 2]_t
/// let tb = t_binomial(5, 2);
/// ```
///
/// # Properties
///
/// - [n choose 0]_t = [n choose n]_t = 1
/// - [n choose k]_t = 0 when k > n
/// - When t = 1, recovers the classical binomial coefficient
/// - Symmetric: [n choose k]_t = [n choose n-k]_t
pub fn t_binomial(n: u32, k: u32) -> UnivariatePolynomial<Integer> {
    if k > n {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    if k == 0 || k == n {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    // Use symmetry to optimize computation
    let k = k.min(n - k);

    // Use the recurrence relation:
    // [n choose k]_t = t^k · [n-1 choose k]_t + [n-1 choose k-1]_t

    // Initialize DP table
    let mut dp: Vec<Vec<UnivariatePolynomial<Integer>>> = vec![vec![]; (n + 1) as usize];

    for i in 0..=n as usize {
        dp[i] = vec![UnivariatePolynomial::new(vec![Integer::zero()]); (k + 1) as usize];
        dp[i][0] = UnivariatePolynomial::new(vec![Integer::one()]);
    }

    for i in 1..=n as usize {
        for j in 1..=k.min(i as u32) as usize {
            if j == i {
                dp[i][j] = UnivariatePolynomial::new(vec![Integer::one()]);
            } else {
                // [i choose j]_t = t^j · [i-1 choose j]_t + [i-1 choose j-1]_t
                let t_power_j = if j > 0 {
                    let mut coeffs = vec![Integer::zero(); j + 1];
                    coeffs[j] = Integer::one();
                    UnivariatePolynomial::new(coeffs)
                } else {
                    UnivariatePolynomial::new(vec![Integer::one()])
                };

                let term1 = dp[i - 1][j].clone() * t_power_j;
                let term2 = dp[i - 1][j - 1].clone();
                dp[i][j] = term1 + term2;
            }
        }
    }

    dp[n as usize][k as usize].clone()
}

/// Evaluate a t-binomial coefficient at a specific value of t
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_binomial_eval;
/// use rustmath_integers::Integer;
///
/// // Evaluate [5 choose 2]_t at t = 2
/// let result = t_binomial_eval(5, 2, &Integer::from(2));
/// ```
pub fn t_binomial_eval(n: u32, k: u32, t_value: &Integer) -> Integer {
    let poly = t_binomial(n, k);
    poly.eval(t_value)
}

/// Compute the t-Catalan number C_n(t)
///
/// The t-Catalan numbers are t-analogues of the Catalan numbers.
/// They can be defined as:
///
/// C_n(t) = (1/[n+1]_t) * [2n choose n]_t
///
/// Alternatively, using the recurrence:
/// C_n(t) = Sum_{k=0}^{n-1} t^k * C_k(t) * C_{n-1-k}(t)
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_catalan;
///
/// // C_0(t) = 1
/// let tc0 = t_catalan(0);
///
/// // C_3(t)
/// let tc3 = t_catalan(3);
/// ```
///
/// # Properties
///
/// - C_0(t) = 1
/// - When t = 1, reduces to the classical Catalan number
pub fn t_catalan(n: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    // Use the recurrence relation:
    // C_n(t) = Sum_{k=0}^{n-1} t^k * C_k(t) * C_{n-1-k}(t)
    let mut c = vec![UnivariatePolynomial::new(vec![Integer::zero()]); (n + 1) as usize];
    c[0] = UnivariatePolynomial::new(vec![Integer::one()]);

    for i in 1..=n as usize {
        let mut sum = UnivariatePolynomial::new(vec![Integer::zero()]);

        for k in 0..i {
            // t^k
            let mut t_power_coeffs = vec![Integer::zero(); k + 1];
            t_power_coeffs[k] = Integer::one();
            let t_power = UnivariatePolynomial::new(t_power_coeffs);

            let term = t_power * c[k].clone() * c[i - 1 - k].clone();
            sum = sum + term;
        }

        c[i] = sum;
    }

    c[n as usize].clone()
}

/// Compute the t-Fibonacci number F_n(t)
///
/// The t-Fibonacci numbers are a t-analogue of the Fibonacci sequence.
/// They satisfy the recurrence:
///
/// F_0(t) = 0, F_1(t) = 1
/// F_n(t) = F_{n-1}(t) + t^{n-2} * F_{n-2}(t)
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_fibonacci;
///
/// // F_0(t) = 0
/// let tf0 = t_fibonacci(0);
///
/// // F_1(t) = 1
/// let tf1 = t_fibonacci(1);
///
/// // F_5(t)
/// let tf5 = t_fibonacci(5);
/// ```
///
/// # Properties
///
/// - When t = 1, reduces to the classical Fibonacci sequence
pub fn t_fibonacci(n: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }
    if n == 1 {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    let mut f_prev2 = UnivariatePolynomial::new(vec![Integer::zero()]);
    let mut f_prev1 = UnivariatePolynomial::new(vec![Integer::one()]);

    for i in 2..=n {
        // F_i(t) = F_{i-1}(t) + t^{i-2} * F_{i-2}(t)
        let power = (i - 2) as usize;
        let mut t_power_coeffs = vec![Integer::zero(); power + 1];
        t_power_coeffs[power] = Integer::one();
        let t_power = UnivariatePolynomial::new(t_power_coeffs);

        let f_current = f_prev1.clone() + (t_power * f_prev2.clone());

        f_prev2 = f_prev1;
        f_prev1 = f_current;
    }

    f_prev1
}

/// Compute the t-Lucas number L_n(t)
///
/// The t-Lucas numbers are a t-analogue of the Lucas sequence.
/// They satisfy the recurrence:
///
/// L_0(t) = 2, L_1(t) = 1
/// L_n(t) = L_{n-1}(t) + t^{n-2} * L_{n-2}(t)
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_lucas;
///
/// // L_0(t) = 2
/// let tl0 = t_lucas(0);
///
/// // L_1(t) = 1
/// let tl1 = t_lucas(1);
///
/// // L_5(t)
/// let tl5 = t_lucas(5);
/// ```
///
/// # Properties
///
/// - When t = 1, reduces to the classical Lucas sequence
pub fn t_lucas(n: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 {
        return UnivariatePolynomial::new(vec![Integer::from(2)]);
    }
    if n == 1 {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    let mut l_prev2 = UnivariatePolynomial::new(vec![Integer::from(2)]);
    let mut l_prev1 = UnivariatePolynomial::new(vec![Integer::one()]);

    for i in 2..=n {
        // L_i(t) = L_{i-1}(t) + t^{i-2} * L_{i-2}(t)
        let power = (i - 2) as usize;
        let mut t_power_coeffs = vec![Integer::zero(); power + 1];
        t_power_coeffs[power] = Integer::one();
        let t_power = UnivariatePolynomial::new(t_power_coeffs);

        let l_current = l_prev1.clone() + (t_power * l_prev2.clone());

        l_prev2 = l_prev1;
        l_prev1 = l_current;
    }

    l_prev1
}

/// Compute the t-Stirling number of the second kind S(n, k; t)
///
/// The t-Stirling numbers of the second kind are a t-analogue of the
/// Stirling numbers of the second kind. They satisfy the recurrence:
///
/// S(n, k; t) = t*k * S(n-1, k; t) + S(n-1, k-1; t)
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_stirling_second;
///
/// // S(5, 2; t)
/// let ts = t_stirling_second(5, 2);
/// ```
///
/// # Properties
///
/// - S(n, 0; t) = 0 for n > 0
/// - S(0, 0; t) = 1
/// - S(n, n; t) = 1
/// - When t = 1, reduces to the classical Stirling numbers of the second kind
pub fn t_stirling_second(n: u32, k: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 && k == 0 {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }
    if n == 0 || k == 0 || k > n {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }
    if k == 1 || k == n {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    // Use recurrence: S(n, k; t) = t*k * S(n-1, k; t) + S(n-1, k-1; t)
    let mut dp: Vec<Vec<UnivariatePolynomial<Integer>>> = vec![vec![]; (n + 1) as usize];

    for i in 0..=n as usize {
        dp[i] = vec![UnivariatePolynomial::new(vec![Integer::zero()]); (k + 1) as usize];
        if i == 0 {
            dp[i][0] = UnivariatePolynomial::new(vec![Integer::one()]);
        } else {
            dp[i][0] = UnivariatePolynomial::new(vec![Integer::zero()]);
        }
    }

    for i in 1..=n as usize {
        for j in 1..=k.min(i as u32) as usize {
            if j == 1 {
                dp[i][j] = UnivariatePolynomial::new(vec![Integer::one()]);
            } else if j == i {
                dp[i][j] = UnivariatePolynomial::new(vec![Integer::one()]);
            } else {
                // S(i, j; t) = t*j * S(i-1, j; t) + S(i-1, j-1; t)
                // t*j is the polynomial with coefficients [0, j]
                let t_times_j = UnivariatePolynomial::new(vec![Integer::zero(), Integer::from(j as u32)]);

                let term1 = t_times_j * dp[i - 1][j].clone();
                let term2 = dp[i - 1][j - 1].clone();
                dp[i][j] = term1 + term2;
            }
        }
    }

    dp[n as usize][k as usize].clone()
}

/// Compute the t-Bell number B_n(t)
///
/// The t-Bell numbers are a t-analogue of the Bell numbers.
/// They are defined as:
///
/// B_n(t) = Sum_{k=0}^{n} S(n, k; t)
///
/// where S(n, k; t) are the t-Stirling numbers of the second kind.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_bell;
///
/// // B_0(t) = 1
/// let tb0 = t_bell(0);
///
/// // B_5(t)
/// let tb5 = t_bell(5);
/// ```
///
/// # Properties
///
/// - When t = 1, reduces to the classical Bell numbers
pub fn t_bell(n: u32) -> UnivariatePolynomial<Integer> {
    let mut sum = UnivariatePolynomial::new(vec![Integer::zero()]);

    for k in 0..=n {
        sum = sum + t_stirling_second(n, k);
    }

    sum
}

/// Compute the t-Eulerian number A(n, k; t)
///
/// The t-Eulerian numbers are a t-analogue of the Eulerian numbers.
/// They can be defined using the recurrence:
///
/// A(n, k; t) = (k+1) * A(n-1, k; t) + t * (n-k) * A(n-1, k-1; t)
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_eulerian;
///
/// // A(5, 2; t)
/// let te = t_eulerian(5, 2);
/// ```
///
/// # Properties
///
/// - A(n, k; 1) equals the classical Eulerian number
/// - A(n, 0; t) = 1 for all n
pub fn t_eulerian(n: u32, k: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 && k == 0 {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }
    if k >= n {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }
    if k == 0 {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    // Use dynamic programming
    let mut dp: Vec<Vec<UnivariatePolynomial<Integer>>> = vec![vec![]; (n + 1) as usize];
    dp[0] = vec![UnivariatePolynomial::new(vec![Integer::one()])];

    for i in 1..=n as usize {
        dp[i] = vec![UnivariatePolynomial::new(vec![Integer::zero()]); (k + 1) as usize];

        for j in 0..=k.min((i - 1) as u32) as usize {
            if j == 0 {
                dp[i][j] = UnivariatePolynomial::new(vec![Integer::one()]);
            } else {
                // A(i, j; t) = (j+1) * A(i-1, j; t) + t * (i-j) * A(i-1, j-1; t)
                let term1 = dp[i - 1][j].clone() * UnivariatePolynomial::new(vec![Integer::from((j + 1) as u32)]);

                // t * (i-j)
                let coeff = Integer::from((i as u32) - (j as u32));
                let t_times_coeff = UnivariatePolynomial::new(vec![Integer::zero(), coeff]);
                let term2 = t_times_coeff * dp[i - 1][j - 1].clone();

                dp[i][j] = term1 + term2;
            }
        }
    }

    dp[n as usize][k as usize].clone()
}

/// Compute the t-multinomial coefficient
///
/// [n; k1, k2, ..., km]_t = [n]_t! / ([k1]_t! · [k2]_t! · ... · [km]_t!)
///
/// where k1 + k2 + ... + km = n
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::t_multinomial;
///
/// // [5; 2, 2, 1]_t
/// let tm = t_multinomial(5, &[2, 2, 1]);
/// ```
pub fn t_multinomial(n: u32, ks: &[u32]) -> UnivariatePolynomial<Integer> {
    let sum: u32 = ks.iter().sum();
    if sum != n {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    let mut result = t_factorial(n);

    for &k in ks {
        let t_k_factorial = t_factorial(k);
        result = divide_t_factorial_polynomials(result, t_k_factorial);
    }

    result
}

/// Helper function to divide two t-factorial polynomials
///
/// This function divides polynomial p by polynomial q, assuming the division
/// is exact (no remainder).
fn divide_t_factorial_polynomials(
    p: UnivariatePolynomial<Integer>,
    q: UnivariatePolynomial<Integer>,
) -> UnivariatePolynomial<Integer> {
    if q.is_zero() {
        panic!("Division by zero polynomial");
    }

    if p.is_zero() {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    let p_deg = p.degree().unwrap_or(0);
    let q_deg = q.degree().unwrap_or(0);

    if p_deg < q_deg {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    let mut remainder = p.clone();
    let mut quotient_coeffs = vec![Integer::zero(); p_deg - q_deg + 1];

    let q_lead = q.coefficients()[q_deg].clone();

    while let Some(r_deg) = remainder.degree() {
        if r_deg < q_deg {
            break;
        }

        let r_lead = remainder.coefficients()[r_deg].clone();

        // For exact division with integers, we need r_lead to be divisible by q_lead
        if r_lead.clone() % q_lead.clone() != Integer::zero() {
            break;
        }

        let coeff = r_lead / q_lead.clone();
        let power = r_deg - q_deg;

        quotient_coeffs[power] = coeff.clone();

        // Subtract coeff * q * t^power from remainder
        let mut term_coeffs = vec![Integer::zero(); power];
        for (i, c) in q.coefficients().iter().enumerate() {
            term_coeffs.push(c.clone() * coeff.clone());
        }
        let term = UnivariatePolynomial::new(term_coeffs);
        remainder = remainder - term;
    }

    UnivariatePolynomial::new(quotient_coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring;
    use rustmath_polynomials::Polynomial;

    #[test]
    fn test_t_integer() {
        // [0]_t = 0
        let t0 = t_integer(0);
        assert_eq!(t0.coefficients(), &[Integer::zero()]);

        // [1]_t = 1
        let t1 = t_integer(1);
        assert_eq!(t1.coefficients(), &[Integer::one()]);

        // [2]_t = 1 + t
        let t2 = t_integer(2);
        assert_eq!(t2.coefficients(), &[Integer::one(), Integer::one()]);

        // [3]_t = 1 + t + t^2
        let t3 = t_integer(3);
        assert_eq!(
            t3.coefficients(),
            &[Integer::one(), Integer::one(), Integer::one()]
        );

        // Test evaluation at t = 1
        for n in 0..10 {
            let tn = t_integer(n);
            let result = tn.eval(&Integer::one());
            assert_eq!(result, Integer::from(n));
        }
    }

    #[test]
    fn test_t_factorial() {
        // [0]_t! = 1
        let tf0 = t_factorial(0);
        assert_eq!(tf0.coefficients(), &[Integer::one()]);

        // [1]_t! = 1
        let tf1 = t_factorial(1);
        assert_eq!(tf1.coefficients(), &[Integer::one()]);

        // [2]_t! = [1]_t · [2]_t = 1 · (1 + t) = 1 + t
        let tf2 = t_factorial(2);
        assert_eq!(tf2.coefficients(), &[Integer::one(), Integer::one()]);

        // Test evaluation at t = 1
        let factorials = [1, 1, 2, 6, 24, 120];
        for (n, expected) in factorials.iter().enumerate() {
            let tfn = t_factorial(n as u32);
            let result = tfn.eval(&Integer::one());
            assert_eq!(result, Integer::from(*expected));
        }
    }

    #[test]
    fn test_t_binomial() {
        // [n choose 0]_t = 1
        for n in 0..10 {
            let tb = t_binomial(n, 0);
            assert_eq!(tb.coefficients(), &[Integer::one()]);
        }

        // [n choose n]_t = 1
        for n in 0..10 {
            let tb = t_binomial(n, n);
            assert_eq!(tb.coefficients(), &[Integer::one()]);
        }

        // [n choose k]_t = 0 when k > n
        let tb = t_binomial(5, 6);
        assert_eq!(tb.coefficients(), &[Integer::zero()]);

        // Test evaluation at t = 1
        let test_cases = [(5, 2, 10), (6, 3, 20), (7, 2, 21)];
        for (n, k, expected) in test_cases.iter() {
            let tb = t_binomial(*n, *k);
            let result = tb.eval(&Integer::one());
            assert_eq!(result, Integer::from(*expected));
        }
    }

    #[test]
    fn test_t_catalan() {
        // C_0(t) = 1
        let tc0 = t_catalan(0);
        assert_eq!(tc0.coefficients(), &[Integer::one()]);

        // Test evaluation at t = 1 (should give Catalan numbers)
        let catalan_numbers = [1, 1, 2, 5, 14, 42];
        for (n, expected) in catalan_numbers.iter().enumerate() {
            let tcn = t_catalan(n as u32);
            let result = tcn.eval(&Integer::one());
            assert_eq!(result, Integer::from(*expected));
        }
    }

    #[test]
    fn test_t_fibonacci() {
        // F_0(t) = 0
        let tf0 = t_fibonacci(0);
        assert_eq!(tf0.coefficients(), &[Integer::zero()]);

        // F_1(t) = 1
        let tf1 = t_fibonacci(1);
        assert_eq!(tf1.coefficients(), &[Integer::one()]);

        // Test evaluation at t = 1 (should give Fibonacci numbers)
        let fib_numbers = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34];
        for (n, expected) in fib_numbers.iter().enumerate() {
            let tfn = t_fibonacci(n as u32);
            let result = tfn.eval(&Integer::one());
            assert_eq!(result, Integer::from(*expected));
        }
    }

    #[test]
    fn test_t_lucas() {
        // L_0(t) = 2
        let tl0 = t_lucas(0);
        assert_eq!(tl0.coefficients(), &[Integer::from(2)]);

        // L_1(t) = 1
        let tl1 = t_lucas(1);
        assert_eq!(tl1.coefficients(), &[Integer::one()]);

        // Test evaluation at t = 1 (should give Lucas numbers)
        let lucas_numbers = [2, 1, 3, 4, 7, 11, 18, 29];
        for (n, expected) in lucas_numbers.iter().enumerate() {
            let tln = t_lucas(n as u32);
            let result = tln.eval(&Integer::one());
            assert_eq!(result, Integer::from(*expected));
        }
    }

    #[test]
    fn test_t_stirling_second() {
        // S(n, 0; t) = 0 for n > 0
        assert_eq!(
            t_stirling_second(5, 0).coefficients(),
            &[Integer::zero()]
        );

        // S(0, 0; t) = 1
        assert_eq!(
            t_stirling_second(0, 0).coefficients(),
            &[Integer::one()]
        );

        // S(n, n; t) = 1
        assert_eq!(
            t_stirling_second(5, 5).coefficients(),
            &[Integer::one()]
        );

        // Test evaluation at t = 1 (should give Stirling numbers)
        assert_eq!(
            t_stirling_second(5, 2).eval(&Integer::one()),
            Integer::from(15)
        );
        assert_eq!(
            t_stirling_second(5, 3).eval(&Integer::one()),
            Integer::from(25)
        );
    }

    #[test]
    fn test_t_bell() {
        // B_0(t) = 1
        let tb0 = t_bell(0);
        assert_eq!(tb0.eval(&Integer::one()), Integer::from(1));

        // Test evaluation at t = 1 (should give Bell numbers)
        let bell_numbers = [1, 1, 2, 5, 15, 52];
        for (n, expected) in bell_numbers.iter().enumerate() {
            let tbn = t_bell(n as u32);
            let result = tbn.eval(&Integer::one());
            assert_eq!(result, Integer::from(*expected));
        }
    }

    #[test]
    fn test_t_eulerian() {
        // A(n, 0; t) = 1
        for n in 1..10 {
            let te = t_eulerian(n, 0);
            assert_eq!(te.eval(&Integer::one()), Integer::from(1));
        }

        // A(n, k; t) = 0 for k >= n
        assert_eq!(
            t_eulerian(5, 5).coefficients(),
            &[Integer::zero()]
        );

        // Test evaluation at t = 1 (should give Eulerian numbers)
        assert_eq!(t_eulerian(3, 0).eval(&Integer::one()), Integer::from(1));
        assert_eq!(t_eulerian(3, 1).eval(&Integer::one()), Integer::from(4));
        assert_eq!(t_eulerian(3, 2).eval(&Integer::one()), Integer::from(1));
    }

    #[test]
    fn test_t_binomial_eval() {
        // Test that evaluation works correctly
        let result = t_binomial_eval(5, 2, &Integer::from(2));
        // Should be the same as evaluating the polynomial
        let poly = t_binomial(5, 2);
        assert_eq!(result, poly.eval(&Integer::from(2)));
    }

    #[test]
    fn test_t_multinomial() {
        // [3; 1, 1, 1]_t should equal [3]_t!
        let tm = t_multinomial(3, &[1, 1, 1]);
        let tf3 = t_factorial(3);
        assert_eq!(
            tm.eval(&Integer::one()),
            tf3.eval(&Integer::one())
        );

        // Invalid case: sum doesn't equal n
        let tm_invalid = t_multinomial(5, &[2, 2]);
        assert_eq!(tm_invalid.coefficients(), &[Integer::zero()]);
    }

    #[test]
    fn test_evaluation_at_zero() {
        // Test evaluation at t = 0
        // [n]_0 = 0 for n > 1, [1]_0 = 1, [0]_0 = 0
        assert_eq!(t_integer(0).eval(&Integer::zero()), Integer::zero());
        assert_eq!(t_integer(1).eval(&Integer::zero()), Integer::one());
        assert_eq!(t_integer(5).eval(&Integer::zero()), Integer::one());

        // F_n(0) should give the n-th Fibonacci number with alternating signs
        assert_eq!(t_fibonacci(2).eval(&Integer::zero()), Integer::one());
    }

    #[test]
    fn test_symmetry() {
        // Test that [n choose k]_t = [n choose n-k]_t when evaluated
        for n in 2..8 {
            for k in 0..=n {
                let tb1 = t_binomial(n, k);
                let tb2 = t_binomial(n, n - k);

                assert_eq!(
                    tb1.eval(&Integer::one()),
                    tb2.eval(&Integer::one())
                );
                assert_eq!(
                    tb1.eval(&Integer::from(2)),
                    tb2.eval(&Integer::from(2))
                );
            }
        }
    }
}
