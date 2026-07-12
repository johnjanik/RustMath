//! Modular forms and cusp forms
//!
//! This module implements basic structures for modular forms on congruence subgroups.
//!
//! ## Exact q-expansions
//!
//! The level-1 forms in this module ([`EisensteinSeries`], [`ModularDiscriminant`],
//! [`JInvariant`]) have their q-expansions computed EXACTLY, in `Integer` /
//! `Rational` arithmetic, from their defining product/series formulas:
//!
//! ```text
//!     E_k    = 1 - (2k / B_k) sum_{n >= 1} sigma_{k-1}(n) q^n      (k >= 4 even)
//!     Delta  = q prod_{m >= 1} (1 - q^m)^24 = sum_{n >= 1} tau(n) q^n
//!     j      = E_4^3 / Delta = 1/q + 744 + 196884 q + ...
//! ```
//!
//! `prod (1 - q^m)^24` is `(prod (1 - q^m)^3)^8`, and `prod (1 - q^m)^3` is the
//! SPARSE Jacobi series `sum_{n >= 0} (-1)^n (2n+1) q^{n(n+1)/2}`, so the
//! expansion costs eight sparse multiplications rather than a dense 24th power.
//! `j` is then the exact power-series quotient `E_4^3 * P^{-1}` shifted by one,
//! where `P = Delta / q` has constant term 1 and is therefore invertible over Z.
//!
//! These are cross-certified in the tests by the classical identity
//! `E_4^3 - E_6^2 = 1728 Delta` (which ties the Eisenstein series and the eta
//! product together with no baked-in constants), by the multiplicativity of tau
//! and the Ramanujan congruence `tau(n) = sigma_11(n) mod 691`, and against
//! PARI/GP's `ramanujantau` and `ellj`.
//!
//! ## Honest failure
//!
//! Expanding to precision n costs O(n^1.5) bignum operations and O(n) memory
//! (O(n^2) for `j`, which needs a dense series inversion), so the expansions are
//! capped; past the cap the `try_`-prefixed methods return an `Err` and the
//! infallible ones PANIC with a precise message rather than inventing a
//! coefficient.

use crate::arithgroup::ArithmeticSubgroup;
use rustmath_complex::Complex;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::{bernoulli, Rational};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Weight of a modular form
pub type Weight = i32;

/// Largest n for which `Delta`'s coefficient tau(n) will be expanded.
///
/// The expansion is O(n^1.5) bignum operations and O(n) bignum memory; a single
/// tau(n) for larger n needs a genuinely different algorithm (Hecke recursion
/// off the tau(p) for p | n, or a point count), which is not implemented here.
pub const MAX_DELTA_PRECISION: u64 = 10_000;

/// Largest n for which `j`'s coefficient c(n) will be expanded.
///
/// Lower than [`MAX_DELTA_PRECISION`] because the series inversion behind `j` is
/// O(n^2) and the coefficients themselves grow like exp(4 pi sqrt(n)).
pub const MAX_J_PRECISION: u64 = 1_000;

/// `sigma_m(n)` = sum of the m-th powers of the divisors of n, exactly.
fn sigma_exact(n: u64, m: u32) -> Integer {
    let mut sum = Integer::zero();
    let mut d = 1u64;
    while d <= n / d {
        if n.is_multiple_of(d) {
            sum = sum + Integer::from(d).pow(m);
            let e = n / d;
            if e != d {
                sum = sum + Integer::from(e).pow(m);
            }
        }
        d += 1;
    }
    sum
}

/// Jacobi's identity as a SPARSE series (ascending in the exponent):
///
/// ```text
///     prod_{m >= 1} (1 - q^m)^3 = sum_{n >= 0} (-1)^n (2n + 1) q^{n(n+1)/2}.
/// ```
fn jacobi_cube_sparse(prec: usize) -> Vec<(usize, i64)> {
    let mut out = Vec::new();
    let mut n = 0usize;
    loop {
        let e = n * (n + 1) / 2;
        if e > prec {
            break;
        }
        let c = (2 * n + 1) as i64;
        out.push((e, if n.is_multiple_of(2) { c } else { -c }));
        n += 1;
    }
    out
}

/// `dense * sparse`, truncated at q^prec.  `sparse` must be ascending in the
/// exponent (as [`jacobi_cube_sparse`] is), which is what makes the inner
/// `break` correct.
fn mul_sparse(dense: &[Integer], sparse: &[(usize, i64)], prec: usize) -> Vec<Integer> {
    let mut out = vec![Integer::zero(); prec + 1];
    for (i, a) in dense.iter().enumerate() {
        if i > prec || a.is_zero() {
            continue;
        }
        for &(e, c) in sparse {
            if i + e > prec {
                break;
            }
            out[i + e] = out[i + e].clone() + a * &Integer::from(c);
        }
    }
    out
}

/// `a * b`, truncated at q^prec.
fn mul_trunc(a: &[Integer], b: &[Integer], prec: usize) -> Vec<Integer> {
    let mut out = vec![Integer::zero(); prec + 1];
    for (i, x) in a.iter().enumerate() {
        if i > prec || x.is_zero() {
            continue;
        }
        for (jj, y) in b.iter().enumerate() {
            if i + jj > prec {
                break;
            }
            if y.is_zero() {
                continue;
            }
            out[i + jj] = out[i + jj].clone() + x * y;
        }
    }
    out
}

/// The inverse of a power series with constant term 1, exactly over Z:
/// `inv[0] = 1`, `inv[m] = -sum_{i=1}^{m} a[i] inv[m-i]`.
fn invert_unit_series(a: &[Integer], prec: usize) -> Vec<Integer> {
    assert!(
        a.first().map(|c| c.is_one()).unwrap_or(false),
        "invert_unit_series needs constant term 1"
    );
    let mut inv = vec![Integer::zero(); prec + 1];
    inv[0] = Integer::one();
    for m in 1..=prec {
        let mut s = Integer::zero();
        for i in 1..=m {
            if i < a.len() && !a[i].is_zero() {
                s = s + &a[i] * &inv[m - i];
            }
        }
        inv[m] = -s;
    }
    inv
}

/// Coefficients of `P = prod_{m >= 1} (1 - q^m)^24 = (prod (1 - q^m)^3)^8` up to
/// q^prec, exactly.  `P` has constant term 1 and `Delta = q P`.
fn eta_product_24(prec: usize) -> Vec<Integer> {
    let s = jacobi_cube_sparse(prec);
    let mut acc = vec![Integer::zero(); prec + 1];
    acc[0] = Integer::one();
    for _ in 0..8 {
        acc = mul_sparse(&acc, &s, prec);
    }
    acc
}

/// Coefficients a_0, ..., a_prec of `Delta = q prod (1 - q^m)^24`, so index n
/// holds tau(n) (and index 0 holds 0).
fn delta_expansion(prec: usize) -> Vec<Integer> {
    let mut out = vec![Integer::zero(); prec + 1];
    if prec == 0 {
        return out;
    }
    let p = eta_product_24(prec - 1);
    for n in 1..=prec {
        out[n] = p[n - 1].clone();
    }
    out
}

/// Coefficients of `E_4 = 1 + 240 sum_{n >= 1} sigma_3(n) q^n` up to q^prec.
fn e4_expansion(prec: usize) -> Vec<Integer> {
    let mut out = vec![Integer::zero(); prec + 1];
    out[0] = Integer::one();
    let c = Integer::from(240);
    for n in 1..=prec {
        out[n] = &c * &sigma_exact(n as u64, 3);
    }
    out
}

/// The coefficients c(-1), c(0), c(1), ..., c(prec) of
/// `j = E_4^3 / Delta = 1/q + 744 + 196884 q + ...`, at index 0, 1, 2, ...
///
/// `Delta = q P` with `P` of constant term 1, so `q j = E_4^3 P^{-1}` and
/// `c(n) = [q^{n+1}] (E_4^3 P^{-1})`.  Everything is exact over Z.
fn j_expansion(prec: usize) -> Vec<Integer> {
    let m = prec + 1;
    let e4 = e4_expansion(m);
    let e4_squared = mul_trunc(&e4, &e4, m);
    let e4_cubed = mul_trunc(&e4_squared, &e4, m);
    let p = eta_product_24(m);
    let p_inv = invert_unit_series(&p, m);
    mul_trunc(&e4_cubed, &p_inv, m)
}

fn delta_cache() -> &'static Mutex<Vec<Integer>> {
    static CACHE: OnceLock<Mutex<Vec<Integer>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(vec![Integer::zero()]))
}

fn j_cache() -> &'static Mutex<Vec<Integer>> {
    static CACHE: OnceLock<Mutex<Vec<Integer>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

/// tau(n) exactly, cached, or an honest error past the precision cap.
fn tau_exact(n: u64) -> Result<Integer, String> {
    if n > MAX_DELTA_PRECISION {
        return Err(format!(
            "tau({n}): Delta's q-expansion is obtained by expanding \
             prod (1 - q^m)^24 out to q^n, which is O(n^1.5) bignum operations and \
             O(n) bignum memory, and is capped here at n <= {MAX_DELTA_PRECISION}. \
             A single tau(n) for larger n needs a different algorithm (a Hecke \
             recursion off the tau(p) for p | n, or a point count), which is not \
             implemented."
        ));
    }
    let idx = n as usize;
    let mut cache = delta_cache()
        .lock()
        .map_err(|_| "the Delta q-expansion cache is poisoned".to_string())?;
    if cache.len() <= idx {
        let prec = idx
            .max(32)
            .max(2 * cache.len())
            .min(MAX_DELTA_PRECISION as usize)
            .max(idx);
        *cache = delta_expansion(prec);
    }
    Ok(cache[idx].clone())
}

/// c(n) exactly, cached, or an honest error past the precision cap.
fn j_coefficient_exact(n: i64) -> Result<Integer, String> {
    // j is holomorphic on the upper half-plane with a SIMPLE pole at the cusp,
    // so the principal part is exactly c(-1)/q: every c(n) with n <= -2 is zero.
    // This zero is a theorem, not a placeholder.
    if n < -1 {
        return Ok(Integer::zero());
    }
    if n > MAX_J_PRECISION as i64 {
        return Err(format!(
            "j coefficient c({n}): the q-expansion of j = E_4^3 / Delta needs a dense \
             power-series inversion, which is O(n^2) bignum operations, and its \
             coefficients grow like exp(4 pi sqrt(n)); this is capped here at \
             n <= {MAX_J_PRECISION}."
        ));
    }
    let idx = (n + 1) as usize;
    let need = n.max(0) as usize;
    let mut cache = j_cache()
        .lock()
        .map_err(|_| "the j q-expansion cache is poisoned".to_string())?;
    if cache.len() <= idx {
        let prec = need
            .max(32)
            .max(2 * cache.len())
            .min(MAX_J_PRECISION as usize)
            .max(need);
        *cache = j_expansion(prec);
    }
    Ok(cache[idx].clone())
}

/// A modular form is a holomorphic function on the upper half-plane
/// satisfying transformation properties under a congruence subgroup
#[derive(Debug, Clone)]
pub struct ModularForm {
    /// Weight of the modular form
    weight: Weight,
    /// Level (for congruence subgroups)
    level: u64,
    /// q-expansion coefficients a(n) where f(q) = sum a(n) q^n
    /// We store coefficients up to some precision
    q_expansion: HashMap<u64, Rational>,
    /// Maximum n for which we have computed a(n)
    precision: u64,
}

impl ModularForm {
    /// Create a new modular form with given weight and level
    pub fn new(weight: Weight, level: u64) -> Self {
        ModularForm {
            weight,
            level,
            q_expansion: HashMap::new(),
            precision: 0,
        }
    }

    /// Get the weight
    pub fn weight(&self) -> Weight {
        self.weight
    }

    /// Get the level
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Get the q-expansion coefficient a(n)
    pub fn coefficient(&self, n: u64) -> Option<&Rational> {
        self.q_expansion.get(&n)
    }

    /// Set a q-expansion coefficient
    pub fn set_coefficient(&mut self, n: u64, value: Rational) {
        self.q_expansion.insert(n, value);
        if n > self.precision {
            self.precision = n;
        }
    }

    /// Get all coefficients up to precision
    pub fn coefficients(&self, max_n: u64) -> Vec<Option<Rational>> {
        (0..=max_n)
            .map(|n| self.q_expansion.get(&n).cloned())
            .collect()
    }

    /// Check if this is a cusp form (a(0) = 0)
    pub fn is_cusp_form(&self) -> bool {
        self.q_expansion
            .get(&0)
            .map(|c| c.is_zero())
            .unwrap_or(true)
    }

    /// Evaluate at a point in the upper half-plane (approximately)
    /// q = exp(2πiτ), so we compute sum a(n) q^n
    pub fn evaluate_approx(&self, tau: Complex, terms: usize) -> Complex {
        use std::f64::consts::PI;

        // q = exp(2πiτ)
        let q = (Complex::new(0.0, 2.0 * PI) * tau).exp();

        let mut result = Complex::new(0.0, 0.0);
        for n in 0..terms.min(self.precision as usize + 1) {
            if let Some(coeff) = self.q_expansion.get(&(n as u64)) {
                let c = coeff.numerator().to_string().parse::<f64>().unwrap_or(0.0)
                    / coeff.denominator().to_string().parse::<f64>().unwrap_or(1.0);
                result = result + Complex::new(c, 0.0) * q.pow(&Complex::new(n as f64, 0.0));
            }
        }
        result
    }

    /// Add two modular forms (must have same weight and level)
    pub fn add(&self, other: &ModularForm) -> Option<ModularForm> {
        if self.weight != other.weight || self.level != other.level {
            return None;
        }

        let mut result = ModularForm::new(self.weight, self.level);
        let max_prec = self.precision.max(other.precision);

        for n in 0..=max_prec {
            let a_n = self.q_expansion.get(&n).cloned().unwrap_or(Rational::zero());
            let b_n = other.q_expansion.get(&n).cloned().unwrap_or(Rational::zero());
            result.set_coefficient(n, a_n + b_n);
        }

        Some(result)
    }

    /// Multiply by a scalar
    pub fn scalar_mul(&self, scalar: &Rational) -> ModularForm {
        let mut result = ModularForm::new(self.weight, self.level);
        for (&n, coeff) in &self.q_expansion {
            result.set_coefficient(n, coeff * scalar);
        }
        result
    }
}

/// A cusp form (modular form vanishing at all cusps)
#[derive(Debug, Clone)]
pub struct CuspForm {
    /// Underlying modular form
    form: ModularForm,
}

impl CuspForm {
    /// Create a new cusp form
    pub fn new(weight: Weight, level: u64) -> Self {
        let mut form = ModularForm::new(weight, level);
        // Ensure a(0) = 0
        form.set_coefficient(0, Rational::zero());
        CuspForm { form }
    }

    /// Get the underlying modular form
    pub fn modular_form(&self) -> &ModularForm {
        &self.form
    }

    /// Get the weight
    pub fn weight(&self) -> Weight {
        self.form.weight()
    }

    /// Get the level
    pub fn level(&self) -> u64 {
        self.form.level()
    }

    /// Get coefficient
    pub fn coefficient(&self, n: u64) -> Option<&Rational> {
        self.form.coefficient(n)
    }

    /// Set coefficient (n > 0 only for cusp forms)
    pub fn set_coefficient(&mut self, n: u64, value: Rational) {
        if n > 0 {
            self.form.set_coefficient(n, value);
        }
    }
}

/// Eisenstein series E_k (for even k >= 4)
pub struct EisensteinSeries {
    weight: Weight,
}

impl EisensteinSeries {
    /// Create Eisenstein series of given weight
    pub fn new(weight: Weight) -> Option<Self> {
        if weight < 4 || weight % 2 != 0 {
            None
        } else {
            Some(EisensteinSeries { weight })
        }
    }

    /// The q-expansion coefficient a_n of the NORMALIZED Eisenstein series
    ///
    /// ```text
    ///     E_k = 1 - (2k / B_k) sum_{n >= 1} sigma_{k-1}(n) q^n,
    /// ```
    ///
    /// exactly: a_0 = 1 and a_n = -(2k / B_k) sigma_{k-1}(n) for n >= 1, with
    /// `B_k` the k-th Bernoulli number.  This is the normalization for which E_k
    /// is the weight-k level-1 Eisenstein series with constant term 1 (so
    /// E_4 = 1 + 240 q + 2160 q^2 + ..., E_6 = 1 - 504 q - 16632 q^2 - ...), and
    /// it is the normalization under which E_4^3 - E_6^2 = 1728 Delta holds; the
    /// tests check exactly that.  (The factor -2k/B_k was previously omitted, so
    /// this returned the bare sigma_{k-1}(n).)
    pub fn coefficient(&self, n: u64) -> Rational {
        if n == 0 {
            return Rational::one();
        }
        let k = self.weight as u32;
        // `new` admits only even k >= 4, so B_k is nonzero and the quotient exists.
        let b_k = bernoulli(k).expect("Bernoulli number of an even weight >= 4");
        let factor = -(Rational::from_integer(Integer::from(2 * k)) / b_k);
        factor * Rational::from_integer(sigma_exact(n, k - 1))
    }

    /// Get weight
    pub fn weight(&self) -> Weight {
        self.weight
    }
}

/// The modular discriminant Delta (unique normalized cusp form of weight 12 and level 1)
pub struct ModularDiscriminant;

impl ModularDiscriminant {
    pub fn new() -> Self {
        ModularDiscriminant
    }

    /// The q-expansion coefficient a_n(Delta) = tau(n), Ramanujan's tau function.
    ///
    /// Computed exactly from Delta = q prod_{m >= 1} (1 - q^m)^24, with the
    /// product expanded as `(prod (1 - q^m)^3)^8` off the sparse Jacobi series
    /// (see the module docs).  a_0 = 0, a_1 = 1, a_2 = tau(2) = -24, ...
    ///
    /// PANICS past [`MAX_DELTA_PRECISION`]; see [`Self::try_coefficient`].
    pub fn coefficient(&self, n: u64) -> Integer {
        self.try_coefficient(n)
            .expect("ModularDiscriminant::coefficient")
    }

    /// tau(n), or an honest error past the precision cap.
    pub fn try_coefficient(&self, n: u64) -> Result<Integer, String> {
        tau_exact(n)
    }

    /// Ramanujan's tau function, tau(n) = a_n(Delta).  Alias of
    /// [`Self::coefficient`].
    pub fn tau(&self, n: u64) -> Integer {
        self.coefficient(n)
    }

    /// The coefficients a_0, ..., a_prec of Delta (so index n is tau(n)).
    pub fn q_expansion(&self, prec: u64) -> Result<Vec<Integer>, String> {
        // Reuse the cache rather than re-expanding: warm it to `prec` first.
        tau_exact(prec)?;
        Ok((0..=prec)
            .map(|n| tau_exact(n).expect("cache already warm to prec"))
            .collect())
    }

    pub fn weight(&self) -> Weight {
        12
    }

    pub fn level(&self) -> u64 {
        1
    }
}

impl Default for ModularDiscriminant {
    fn default() -> Self {
        Self::new()
    }
}

/// The j-invariant (modular function for SL(2,Z))
pub struct JInvariant;

impl JInvariant {
    pub fn new() -> Self {
        JInvariant
    }

    /// The q-expansion coefficient c(n) of
    /// `j = E_4^3 / Delta = 1/q + 744 + 196884 q + 21493760 q^2 + ...`
    ///
    /// Computed exactly as the power-series quotient `E_4^3 * (Delta/q)^{-1}`
    /// over Z, shifted by one (see the module docs).  Every c(n) with n <= -2 is
    /// zero because j has a SIMPLE pole at the cusp -- that zero is a theorem,
    /// not a fallback.  (This used to be four hard-coded values with a fabricated
    /// zero for every n >= 3.)
    ///
    /// PANICS past [`MAX_J_PRECISION`]; see [`Self::try_coefficient`].
    pub fn coefficient(&self, n: i64) -> Integer {
        self.try_coefficient(n).expect("JInvariant::coefficient")
    }

    /// c(n), or an honest error past the precision cap.
    pub fn try_coefficient(&self, n: i64) -> Result<Integer, String> {
        j_coefficient_exact(n)
    }

    /// The coefficients c(-1), c(0), ..., c(prec) of j, at index 0, 1, ...
    pub fn q_expansion(&self, prec: u64) -> Result<Vec<Integer>, String> {
        if prec > MAX_J_PRECISION {
            return Err(format!(
                "j q-expansion to precision {prec} exceeds the cap {MAX_J_PRECISION}"
            ));
        }
        Ok(j_expansion(prec as usize))
    }
}

impl Default for JInvariant {
    fn default() -> Self {
        Self::new()
    }
}

/// Dimension formulas for spaces of modular forms
pub mod dimensions {
    use super::*;

    /// Dimension of M_k(Gamma0(N)) (modular forms of weight k and level N)
    pub fn modular_forms_gamma0(weight: Weight, level: u64) -> u64 {
        use rustmath_core::NumericConversion;

        if weight < 0 {
            return 0;
        }
        if weight == 0 {
            // M_0(Gamma0(N)) = constants, dimension 1.
            return 1;
        }

        // Delegate to `dims`, which computes the exact Diamond-Shurman
        // dimension from the genus, the elliptic point counts and the cusp
        // count. The old `((k-1)*index)/12 + 1` approximation was wrong (e.g.
        // it gave dim M_12(SL2Z) = 1 instead of the correct 2).
        crate::dims::dimension_modular_forms(&Integer::from(level), weight as i64)
            .to_u64()
            .unwrap_or(0)
    }

    /// Dimension of S_k(Gamma0(N)) (cusp forms of weight k and level N)
    pub fn cusp_forms_gamma0(weight: Weight, level: u64) -> u64 {
        use rustmath_core::NumericConversion;

        if weight < 2 {
            return 0;
        }

        // Delegate to the exact cusp-form dimension in `dims`. The old
        // `M_k - 1` shortcut assumed a single Eisenstein series and gave
        // dim S_12(SL2Z) = 0 instead of the correct 1 (generated by Delta).
        crate::dims::dimension_cusp_forms(&Integer::from(level), weight as i64)
            .to_u64()
            .unwrap_or(0)
    }

    /// Dimension of M_k(Gamma1(N))
    pub fn modular_forms_gamma1(weight: Weight, level: u64) -> u64 {
        if weight < 0 {
            return 0;
        }

        let index = crate::arithgroup::Gamma1::new(level).index().unwrap_or(1);
        let k = weight as u64;

        if k >= 2 {
            ((k - 1) * index) / 12 + 1
        } else {
            0
        }
    }

    /// Dimension of S_k(Gamma1(N))
    pub fn cusp_forms_gamma1(weight: Weight, level: u64) -> u64 {
        if weight < 2 {
            return 0;
        }

        let total_dim = modular_forms_gamma1(weight, level);
        if total_dim == 0 {
            0
        } else {
            total_dim.saturating_sub(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_form_creation() {
        let f = ModularForm::new(2, 1);
        assert_eq!(f.weight(), 2);
        assert_eq!(f.level(), 1);
        assert!(f.is_cusp_form()); // No coefficient set, so a(0) is implicitly 0
    }

    #[test]
    fn test_cusp_form() {
        let mut f = CuspForm::new(12, 1);
        f.set_coefficient(1, Rational::one());
        assert_eq!(f.weight(), 12);
        assert_eq!(f.coefficient(0), Some(&Rational::zero()));
        assert_eq!(f.coefficient(1), Some(&Rational::one()));
    }

    #[test]
    fn test_modular_form_addition() {
        let mut f1 = ModularForm::new(2, 1);
        f1.set_coefficient(1, Rational::from_integer(Integer::from(1)));

        let mut f2 = ModularForm::new(2, 1);
        f2.set_coefficient(1, Rational::from_integer(Integer::from(2)));

        let f3 = f1.add(&f2).unwrap();
        assert_eq!(
            f3.coefficient(1),
            Some(&Rational::from_integer(Integer::from(3)))
        );
    }

    /// `E_6 = 1 - 504 sum sigma_5(n) q^n`, built independently of
    /// `EisensteinSeries` so that the E_4^3 - E_6^2 = 1728 Delta gate below is a
    /// check of the Bernoulli normalization rather than a restatement of it.
    fn e6_expansion(prec: usize) -> Vec<Integer> {
        let mut out = vec![Integer::zero(); prec + 1];
        out[0] = Integer::one();
        for n in 1..=prec {
            out[n] = Integer::from(-504) * sigma_exact(n as u64, 5);
        }
        out
    }

    /// The normalized Eisenstein series: PARI/GP
    /// `1 + 240*sum(n=1,10,sigma(n,3)*q^n)` gives
    /// E_4 = 1 + 240q + 2160q^2 + 6720q^3 + 17520q^4 + ..., and the
    /// -2k/B_k normalization must reproduce exactly that (it previously
    /// returned the bare sigma_{k-1}(n), i.e. 1, 9, 28, ...).
    #[test]
    fn test_eisenstein_series() {
        let e4 = EisensteinSeries::new(4).unwrap();
        assert_eq!(e4.weight(), 4);
        assert_eq!(e4.coefficient(0), Rational::one());

        let expect_e4 = [240i64, 2160, 6720, 17520, 30240, 60480, 82560];
        for (i, &a) in expect_e4.iter().enumerate() {
            let n = (i + 1) as u64;
            assert_eq!(
                e4.coefficient(n),
                Rational::from_integer(Integer::from(a)),
                "a_{n}(E_4)"
            );
        }

        // -2k/B_k for k = 6 is -504, for k = 8 is 480, for k = 12 is 65520/691.
        let e6 = EisensteinSeries::new(6).unwrap();
        assert_eq!(
            e6.coefficient(1),
            Rational::from_integer(Integer::from(-504))
        );
        let e8 = EisensteinSeries::new(8).unwrap();
        assert_eq!(e8.coefficient(1), Rational::from_integer(Integer::from(480)));
        let e12 = EisensteinSeries::new(12).unwrap();
        assert_eq!(
            e12.coefficient(1),
            Rational::new(65520, 691).unwrap(),
            "the 691 in the denominator of E_12 is the Ramanujan-congruence prime"
        );
    }

    /// GATE: the classical identity `E_4^3 - E_6^2 = 1728 Delta`, as an exact
    /// identity of integer q-expansions to precision 60.
    ///
    /// This is the strongest self-certification available here and it uses NO
    /// baked-in constant: it ties the Bernoulli-normalized Eisenstein series
    /// (built from divisor sums) to the eta product (built from the sparse Jacobi
    /// series) to Delta.  An error in the -2k/B_k normalization, in the Jacobi
    /// series, in the 8th power, or in the truncated multiplication breaks it.
    #[test]
    fn test_e4_cubed_minus_e6_squared_is_1728_delta() {
        let prec = 60usize;
        let e4 = e4_expansion(prec);
        let e6 = e6_expansion(prec);
        let e4_cubed = mul_trunc(&mul_trunc(&e4, &e4, prec), &e4, prec);
        let e6_squared = mul_trunc(&e6, &e6, prec);
        let delta = delta_expansion(prec);

        // The E_4 built from `EisensteinSeries` (via Bernoulli) must agree with
        // the 240*sigma_3 series used inside `j`.
        let es4 = EisensteinSeries::new(4).unwrap();
        for (n, c) in e4.iter().enumerate() {
            assert_eq!(
                es4.coefficient(n as u64),
                Rational::from_integer(c.clone()),
                "EisensteinSeries(4) vs the 240*sigma_3 series at n = {n}"
            );
        }

        for n in 0..=prec {
            let lhs = e4_cubed[n].clone() - e6_squared[n].clone();
            let rhs = Integer::from(1728) * delta[n].clone();
            assert_eq!(lhs, rhs, "E_4^3 - E_6^2 = 1728 Delta at q^{n}");
        }
    }

    /// tau(1..12), from PARI/GP `ramanujantau(n)`.
    #[test]
    fn test_modular_discriminant() {
        let delta = ModularDiscriminant::new();
        assert_eq!(delta.weight(), 12);
        assert_eq!(delta.level(), 1);
        assert_eq!(delta.coefficient(0), Integer::zero());

        let tau = [
            1i64, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920, 534612, -370944,
        ];
        for (i, &t) in tau.iter().enumerate() {
            let n = (i + 1) as u64;
            assert_eq!(delta.coefficient(n), Integer::from(t), "tau({n})");
        }
        // the facade returned Integer::one() for every n >= 1
        assert_eq!(delta.tau(2), Integer::from(-24));
        assert_ne!(delta.tau(2), Integer::one());

        let qexp = delta.q_expansion(12).unwrap();
        assert_eq!(qexp.len(), 13);
        assert_eq!(qexp[0], Integer::zero());
        assert_eq!(qexp[12], Integer::from(-370944));
    }

    /// GATE: tau is multiplicative, and satisfies Ramanujan's congruence
    /// `tau(n) = sigma_11(n) mod 691`.  Both are properties of the true tau that
    /// no fabricated sequence satisfies, and both are checked here against the
    /// expansion rather than against a table.
    #[test]
    fn test_tau_multiplicativity_and_ramanujan_congruence() {
        let delta = ModularDiscriminant::new();
        fn gcd(a: u64, b: u64) -> u64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        for m in 1..=14u64 {
            for n in 1..=14u64 {
                if gcd(m, n) == 1 {
                    assert_eq!(
                        delta.coefficient(m * n),
                        delta.coefficient(m) * delta.coefficient(n),
                        "tau({m} * {n}) = tau({m}) tau({n})"
                    );
                }
            }
        }
        let p691 = Integer::from(691);
        for n in 1..=80u64 {
            let diff = delta.coefficient(n) - sigma_exact(n, 11);
            assert!(
                diff.modulo(&p691).is_zero(),
                "tau({n}) = sigma_11({n}) mod 691"
            );
        }
    }

    /// The q-expansion of j, from PARI/GP `ellj(q + O(q^12))`:
    /// 1/q + 744 + 196884q + 21493760q^2 + 864299970q^3 + 20245856256q^4
    ///     + 333202640600q^5 + 4252023300096q^6 + 44656994071935q^7 + ...
    ///
    /// The facade had only the first four and returned a fabricated 0 from n = 3.
    #[test]
    fn test_j_invariant() {
        let j = JInvariant::new();
        let c = [
            1i64,
            744,
            196884,
            21493760,
            864299970,
            20245856256,
            333202640600,
            4252023300096,
            44656994071935,
        ];
        for (i, &v) in c.iter().enumerate() {
            let n = i as i64 - 1;
            assert_eq!(j.coefficient(n), Integer::from(v), "c({n}) of j");
        }
        // the fabricated zero used to start here
        assert_ne!(j.coefficient(3), Integer::zero());
        // j has a SIMPLE pole: these zeros are the theorem, not a fallback
        assert_eq!(j.coefficient(-2), Integer::zero());
        assert_eq!(j.coefficient(-17), Integer::zero());
    }

    /// GATE: j is defined by `j Delta = E_4^3`, so re-multiplying must return
    /// E_4^3 exactly.  Certifies the series inversion behind `j`.
    #[test]
    fn test_j_times_delta_is_e4_cubed() {
        let prec = 40usize;
        // q * j has coefficients c(-1), c(0), ... at q^0, q^1, ...
        let qj = j_expansion(prec);
        let p = eta_product_24(prec + 1); // Delta / q
        let product = mul_trunc(&qj, &p, prec);
        let e4 = e4_expansion(prec);
        let e4_cubed = mul_trunc(&mul_trunc(&e4, &e4, prec), &e4, prec);
        for n in 0..=prec {
            assert_eq!(product[n], e4_cubed[n], "(q j)(Delta / q) = E_4^3 at q^{n}");
        }
    }

    /// Honest refusal past the precision cap, not a fabricated coefficient.
    #[test]
    fn test_qexpansion_precision_cap_is_an_honest_error() {
        let delta = ModularDiscriminant::new();
        assert!(delta.try_coefficient(MAX_DELTA_PRECISION).is_ok());
        assert!(delta.try_coefficient(MAX_DELTA_PRECISION + 1).is_err());
        let j = JInvariant::new();
        assert!(j.try_coefficient(MAX_J_PRECISION as i64 + 1).is_err());
    }

    #[test]
    #[should_panic(expected = "ModularDiscriminant::coefficient")]
    fn test_delta_past_the_cap_panics_rather_than_faking_tau() {
        let _ = ModularDiscriminant::new().coefficient(MAX_DELTA_PRECISION + 1);
    }

    #[test]
    fn test_dimension_formulas() {
        use dimensions::*;

        // Exact values, verified against SageMath and the standard
        // genus/cusp dimension formulas:
        //   dim S_12(SL(2,Z))   = 1   (generated by Delta)
        //   dim M_12(SL(2,Z))   = 2   (Delta together with E_12)
        //   dim M_2(SL(2,Z))    = 0   (both S_2 and E_2 vanish)
        //   dim S_2(Gamma0(11)) = 1   (the genus of X_0(11))
        //   dim M_2(Gamma0(11)) = 2   (S_2 dim 1 + E_2 dim 1 = #cusps - 1)
        assert_eq!(cusp_forms_gamma0(12, 1), 1);
        assert_eq!(modular_forms_gamma0(12, 1), 2);
        assert_eq!(modular_forms_gamma0(2, 1), 0);
        assert_eq!(cusp_forms_gamma0(2, 11), 1);
        assert_eq!(modular_forms_gamma0(2, 11), 2);
    }
}
