//! # Eta Products
//!
//! This module provides eta products and eta quotients,
//! corresponding to SageMath's sage.modular.etaproducts module.
//!
//! The Dedekind eta function is η(τ) = q^(1/24) * ∏(1 - q^n) where q = e^(2πiτ).
//! An eta product is a product of powers of η(d*τ) for various divisors d of the level.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// An element of the eta group
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EtaGroupElement {
    /// The level N
    level: Integer,
    /// Map from divisor d of N to the power r_d in η(d*τ)^{r_d}
    powers: HashMap<Integer, i64>,
}

impl EtaGroupElement {
    /// Create a new eta group element
    ///
    /// # Arguments
    /// * `level` - The level N
    /// * `powers` - Map from divisors to powers
    pub fn new(level: Integer, powers: HashMap<Integer, i64>) -> Self {
        EtaGroupElement { level, powers }
    }

    /// Get the level
    pub fn level(&self) -> &Integer {
        &self.level
    }

    /// Get the powers
    pub fn powers(&self) -> &HashMap<Integer, i64> {
        &self.powers
    }

    /// Get the power for a specific divisor
    pub fn get_power(&self, divisor: &Integer) -> i64 {
        *self.powers.get(divisor).unwrap_or(&0)
    }

    /// Set the power for a divisor
    pub fn set_power(&mut self, divisor: Integer, power: i64) {
        if power == 0 {
            self.powers.remove(&divisor);
        } else {
            self.powers.insert(divisor, power);
        }
    }

    /// Compute the order at infinity (q-expansion order)
    pub fn order_at_infinity(&self) -> Rational {
        let mut order = Rational::zero();

        for (d, &r) in &self.powers {
            // Each η(d*τ) contributes d/24 to the order
            let contribution = Rational::new(
                d.clone() * Integer::from(r),
                Integer::from(24),
            )
            .expect("denominator 24 is nonzero");
            order = order + contribution;
        }

        order
    }

    /// Compute the weight
    pub fn weight(&self) -> i64 {
        let mut w = 0i64;
        for &r in self.powers.values() {
            w += r;
        }
        w / 2
    }

    /// Check if this is a valid eta product (satisfies certain conditions)
    pub fn is_valid(&self) -> bool {
        // Check that all divisors actually divide the level
        for d in self.powers.keys() {
            if !(&self.level % d).is_zero() {
                return false;
            }
        }

        // Weight must be integral
        let total: i64 = self.powers.values().sum();
        if total % 2 != 0 {
            return false;
        }

        true
    }

    /// Multiply two eta products
    pub fn mul(&self, other: &EtaGroupElement) -> Option<EtaGroupElement> {
        if self.level != other.level {
            return None;
        }

        let mut new_powers = self.powers.clone();
        for (d, &r) in &other.powers {
            *new_powers.entry(d.clone()).or_insert(0) += r;
        }

        // Remove zero powers
        new_powers.retain(|_, &mut v| v != 0);

        Some(EtaGroupElement::new(self.level.clone(), new_powers))
    }

    /// Compute the inverse
    pub fn inverse(&self) -> EtaGroupElement {
        let mut new_powers = HashMap::new();
        for (d, &r) in &self.powers {
            new_powers.insert(d.clone(), -r);
        }
        EtaGroupElement::new(self.level.clone(), new_powers)
    }

    /// Check the LIGOZAT CONDITIONS for `prod_{d|N} eta(d tau)^{r_d}` to be a
    /// modular FUNCTION on `Gamma_0(N)` with trivial character -- i.e. an element
    /// of the eta group of level `N` -- returning a precise reason if it is not:
    ///
    /// 1. every `d` with `r_d != 0` divides `N`;
    /// 2. `sum_d r_d = 0`                      (weight 0);
    /// 3. `sum_d d r_d = 0 (mod 24)`           (integral order at infinity);
    /// 4. `sum_d (N/d) r_d = 0 (mod 24)`       (integral order at 0);
    /// 5. `prod_d d^{r_d}` is the square of a rational   (trivial character).
    ///
    /// These are exactly the hypotheses under which [`Self::order_at_cusp`] and
    /// [`eta_poly_relations`] are valid: such an `f` is holomorphic and
    /// non-vanishing on the upper half-plane (`eta` has no zeros there), so
    /// `div(f)` is supported on the cusps and has degree 0.
    pub fn eta_group_conditions(&self) -> Result<(), String> {
        let n = &self.level;
        for (d, &r) in &self.powers {
            if r != 0 && !(n % d).is_zero() {
                return Err(format!("exponent r_{d} = {r} but {d} does not divide N = {n}"));
            }
        }

        let weight_sum: i64 = self.powers.values().sum();
        if weight_sum != 0 {
            return Err(format!(
                "sum of exponents is {weight_sum}, not 0: this is an eta product of weight \
                 {}/2, not a modular function (the eta group is the weight-0 group)",
                weight_sum
            ));
        }

        let n_i64 = n
            .to_string()
            .parse::<i64>()
            .map_err(|_| format!("level {n} does not fit in i64"))?;

        let mut s_inf = 0i64; // sum d r_d
        let mut s_zero = 0i64; // sum (N/d) r_d
        for (d, &r) in &self.powers {
            let d_i64 = d
                .to_string()
                .parse::<i64>()
                .map_err(|_| format!("divisor {d} does not fit in i64"))?;
            s_inf += d_i64 * r;
            s_zero += (n_i64 / d_i64) * r;
        }
        if s_inf.rem_euclid(24) != 0 {
            return Err(format!(
                "sum_d d r_d = {s_inf} is not divisible by 24: the order at infinity, \
                 {s_inf}/24, is not an integer"
            ));
        }
        if s_zero.rem_euclid(24) != 0 {
            return Err(format!(
                "sum_d (N/d) r_d = {s_zero} is not divisible by 24: the order at the cusp 0 \
                 is not an integer"
            ));
        }

        // prod d^{r_d} must be a rational square: collect the exponent of each
        // prime and require all of them even.
        let mut prime_exponents: HashMap<u64, i64> = HashMap::new();
        for (d, &r) in &self.powers {
            let mut m = d
                .to_string()
                .parse::<u64>()
                .map_err(|_| format!("divisor {d} does not fit in u64"))?;
            let mut p = 2u64;
            while p * p <= m {
                while m % p == 0 {
                    *prime_exponents.entry(p).or_insert(0) += r;
                    m /= p;
                }
                p += 1;
            }
            if m > 1 {
                *prime_exponents.entry(m).or_insert(0) += r;
            }
        }
        for (p, e) in &prime_exponents {
            if e.rem_euclid(2) != 0 {
                return Err(format!(
                    "prod_d d^(r_d) has odd exponent {e} at the prime {p}, so it is not a \
                     rational square: the eta quotient has a nontrivial character"
                ));
            }
        }

        Ok(())
    }

    /// The order of vanishing of the eta quotient at the cusp `a/d` of
    /// `Gamma_0(N)` (`d | N`), in the local uniformizer of `X_0(N)` at that cusp.
    ///
    /// LIGOZAT's formula (Ono, *Web of Modularity*, Thm 1.64):
    ///
    /// ```text
    ///     ord_{a/d}(f) = (N/24) sum_{delta | N} gcd(d, delta)^2 r_delta
    ///                                          / (gcd(d, N/d) * d * delta)
    /// ```
    ///
    /// It depends only on `d`, not on `a`.  The cusp `d = N` is the cusp
    /// `infinity` (`1/N ~ infinity` under `[[1,0],[N,1]]`), and there the formula
    /// returns the q-valuation `(1/24) sum_d d r_d`.
    ///
    /// The tests check the theorem that certifies this: for an eta-group element,
    /// `sum_{d | N} phi(gcd(d, N/d)) ord_{a/d}(f) = 0` -- the degree of the divisor
    /// of a function on a curve.
    pub fn order_at_cusp(&self, d: &Integer) -> Result<Rational, String> {
        let n = &self.level;
        if (n % d).is_zero() && !d.is_zero() {
            // ok
        } else {
            return Err(format!("{d} is not a divisor of the level {n}"));
        }

        let n_over_d = n / d;
        let g = d.gcd(&n_over_d);

        let mut sum = Rational::zero();
        for (delta, &r) in &self.powers {
            if r == 0 {
                continue;
            }
            if !(n % delta).is_zero() {
                return Err(format!("exponent at {delta}, which does not divide N = {n}"));
            }
            let gcd_d_delta = d.gcd(delta);
            let num = gcd_d_delta.clone() * gcd_d_delta * Integer::from(r);
            let den = g.clone() * d.clone() * delta.clone();
            sum = sum + Rational::new(num, den).map_err(|e| format!("{e:?}"))?;
        }

        let scale = Rational::new(n.clone(), Integer::from(24)).map_err(|e| format!("{e:?}"))?;
        Ok(scale * sum)
    }

    /// The q-expansion of the eta quotient at infinity:
    /// `f = q^v prod_{d} prod_{n >= 1} (1 - q^{dn})^{r_d}` with
    /// `v = (1/24) sum_d d r_d`, returned with `prec` coefficients of the unit
    /// part (so the coefficients of `q^v, ..., q^{v + prec - 1}`).
    ///
    /// The coefficients are EXACT integers: the product/quotient by each
    /// `(1 - q^m)` is a finite integer recurrence.
    ///
    /// Errors if `24` does not divide `sum_d d r_d` (the valuation would not be an
    /// integer, so there is no q-expansion in integer powers of q).
    pub fn qexp(&self, prec: usize) -> Result<LaurentQExpansion, String> {
        let mut s_inf = 0i64;
        let mut divisors: Vec<(u64, i64)> = Vec::new();
        for (d, &r) in &self.powers {
            if r == 0 {
                continue;
            }
            let d_u64 = d
                .to_string()
                .parse::<u64>()
                .map_err(|_| format!("divisor {d} does not fit in u64"))?;
            s_inf += (d_u64 as i64) * r;
            divisors.push((d_u64, r));
        }
        if s_inf.rem_euclid(24) != 0 {
            return Err(format!(
                "sum_d d r_d = {s_inf} is not divisible by 24, so the q-expansion of this eta \
                 quotient has the non-integral valuation {s_inf}/24 and is not a Laurent \
                 series in q"
            ));
        }
        divisors.sort_unstable();

        let mut coeffs = vec![Integer::zero(); prec];
        if prec > 0 {
            coeffs[0] = Integer::one();
        }

        for (d, r) in divisors {
            let mut m = d as usize;
            while m < prec.max(1) && m < prec {
                for _ in 0..r.abs() {
                    if r > 0 {
                        // multiply by (1 - q^m)
                        for k in (m..prec).rev() {
                            coeffs[k] = coeffs[k].clone() - coeffs[k - m].clone();
                        }
                    } else {
                        // divide by (1 - q^m), i.e. multiply by sum_k q^{mk}
                        for k in m..prec {
                            coeffs[k] = coeffs[k].clone() + coeffs[k - m].clone();
                        }
                    }
                }
                m += d as usize;
            }
        }

        Ok(LaurentQExpansion {
            valuation: s_inf / 24,
            coefficients: coeffs,
        })
    }
}

/// A Laurent q-expansion `q^v (c_0 + c_1 q + ... + c_{prec-1} q^{prec-1}) + O(q^{v + prec})`
/// with exact integer coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaurentQExpansion {
    valuation: i64,
    coefficients: Vec<Integer>,
}

impl LaurentQExpansion {
    /// The constant Laurent series 1, with `prec` coefficients.
    pub fn one(prec: usize) -> Self {
        let mut coefficients = vec![Integer::zero(); prec];
        if prec > 0 {
            coefficients[0] = Integer::one();
        }
        LaurentQExpansion {
            valuation: 0,
            coefficients,
        }
    }

    /// The exponent `v` of the leading term (which may be zero: this is the
    /// nominal valuation `sum_d d r_d / 24`, not a recomputed one).
    pub fn valuation(&self) -> i64 {
        self.valuation
    }

    /// The coefficients of `q^v, q^{v+1}, ...`.
    pub fn coefficients(&self) -> &[Integer] {
        &self.coefficients
    }

    /// The coefficient of `q^e`: zero below the valuation, `None` beyond the
    /// computed precision.
    pub fn coefficient_of(&self, e: i64) -> Option<Integer> {
        let k = e - self.valuation;
        if k < 0 {
            return Some(Integer::zero());
        }
        self.coefficients.get(k as usize).cloned()
    }

    /// Truncated product, keeping `prec` coefficients.
    pub fn mul(&self, other: &LaurentQExpansion, prec: usize) -> LaurentQExpansion {
        let mut coefficients = vec![Integer::zero(); prec];
        for (i, a) in self.coefficients.iter().enumerate() {
            if a.is_zero() || i >= prec {
                continue;
            }
            for (j, b) in other.coefficients.iter().enumerate() {
                if i + j >= prec {
                    break;
                }
                if b.is_zero() {
                    continue;
                }
                coefficients[i + j] = coefficients[i + j].clone() + a.clone() * b.clone();
            }
        }
        LaurentQExpansion {
            valuation: self.valuation + other.valuation,
            coefficients,
        }
    }
}

/// The eta group for level N
#[derive(Debug, Clone)]
pub struct EtaGroup {
    /// The level
    level: Integer,
    /// Divisors of the level
    divisors: Vec<Integer>,
}

impl EtaGroup {
    /// Create a new eta group
    ///
    /// # Arguments
    /// * `level` - The level N
    pub fn new(level: Integer) -> Self {
        let divisors = compute_divisors(&level);
        EtaGroup { level, divisors }
    }

    /// Get the level
    pub fn level(&self) -> &Integer {
        &self.level
    }

    /// Get the divisors
    pub fn divisors(&self) -> &[Integer] {
        &self.divisors
    }

    /// Create the identity element
    pub fn identity(&self) -> EtaGroupElement {
        EtaGroupElement::new(self.level.clone(), HashMap::new())
    }

    /// Create an eta product from powers
    pub fn element(&self, powers: HashMap<Integer, i64>) -> EtaGroupElement {
        EtaGroupElement::new(self.level.clone(), powers)
    }
}

/// Create an eta group for level N
pub fn eta_group_class(N: Integer) -> EtaGroup {
    EtaGroup::new(N)
}

/// Create an eta product
///
/// # Arguments
/// * `N` - The level
/// * `powers` - Map from divisors to powers
pub fn eta_product(N: Integer, powers: HashMap<Integer, i64>) -> EtaGroupElement {
    EtaGroupElement::new(N, powers)
}

/// Compute divisors of n
fn compute_divisors(n: &Integer) -> Vec<Integer> {
    if n <= &Integer::zero() {
        return vec![];
    }

    let mut divisors = Vec::new();
    let mut i = Integer::one();
    let sqrt_n = n.sqrt().expect("n > 0 checked above");

    while &i <= &sqrt_n {
        if (n % &i).is_zero() {
            divisors.push(i.clone());
            let other = n / &i;
            if i != other {
                divisors.push(other);
            }
        }
        i = i + Integer::one();
    }

    divisors.sort();
    divisors
}

/// A family of cusps
#[derive(Debug, Clone)]
pub struct CuspFamily {
    /// The level
    level: Integer,
    /// Width of cusps in this family
    width: Integer,
    /// Cusps in this family
    cusps: Vec<(Integer, Integer)>, // (numerator, denominator) pairs
}

impl CuspFamily {
    /// Create a new cusp family
    pub fn new(level: Integer, width: Integer) -> Self {
        CuspFamily {
            level,
            width,
            cusps: Vec::new(),
        }
    }

    /// Get the level
    pub fn level(&self) -> &Integer {
        &self.level
    }

    /// Get the width
    pub fn width(&self) -> &Integer {
        &self.width
    }

    /// Get the cusps
    pub fn cusps(&self) -> &[(Integer, Integer)] {
        &self.cusps
    }

    /// Add a cusp to the family
    pub fn add_cusp(&mut self, numerator: Integer, denominator: Integer) {
        self.cusps.push((numerator, denominator));
    }

    /// Number of cusps in the family
    pub fn len(&self) -> usize {
        self.cusps.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.cusps.is_empty()
    }
}

/// Euler's totient phi(n), for the small arguments used here.
fn euler_phi(n: &Integer) -> u64 {
    let mut m: u64 = n.to_string().parse().expect("phi: argument must fit in u64");
    if m == 0 {
        return 0;
    }
    let mut result = m;
    let mut p = 2u64;
    while p * p <= m {
        if m % p == 0 {
            while m % p == 0 {
                m /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if m > 1 {
        result -= result / m;
    }
    result
}

/// ONE REPRESENTATIVE `(a, d)` -- the cusp `a/d` -- for each cusp of `Gamma_0(N)`.
///
/// The cusps of `Gamma_0(N)` are indexed by pairs `(d, a)` with `d | N` and `a` a
/// unit mod `gcd(d, N/d)`: `a/d` and `a'/d` are `Gamma_0(N)`-equivalent exactly
/// when `a = a' mod gcd(d, N/d)`, and cusps with different denominators `d | N`
/// are never equivalent.  So there are `phi(gcd(d, N/d))` cusps of denominator
/// `d`, and `sum_{d | N} phi(gcd(d, N/d))` in all -- which is the count
/// `dims::gamma0_invariants` gives, and the tests check exactly that.
///
/// The cusp `infinity` is the one with `d = N` (`1/N ~ infinity` under
/// `[[1, 0], [N, 1]]` in `Gamma_0(N)`); it is returned as `(1, N)`, not as a
/// separate `1/0`.
///
/// (This used to return EVERY `a/c` with `c | N`, `0 <= a < c`, `gcd(a, c) = 1`
/// plus a `1/0`, which massively over-counts -- 5 "cusps" for `Gamma_0(4)`, which
/// has 3, and 7 for `Gamma_0(9)`, which has 4 -- because it never identified
/// `a/d ~ a'/d` mod `gcd(d, N/d)`, and listed infinity twice.)
pub fn all_cusps(N: Integer) -> Vec<(Integer, Integer)> {
    let mut cusps = Vec::new();

    for d in compute_divisors(&N) {
        let n_over_d = &N / &d;
        let g = d.gcd(&n_over_d);

        // one a per class mod g, chosen coprime to d (so that a/d is in lowest
        // terms); a = 0 is a legal class representative only when g = 1, and then
        // gcd(0, d) = d = 1 forces d = 1, the cusp 0/1 = 0.
        let mut seen_classes: Vec<Integer> = Vec::new();
        let mut a = Integer::zero();
        while &a < &d {
            if a.gcd(&d).is_one() {
                let class = &a % &g;
                if !seen_classes.contains(&class) {
                    seen_classes.push(class);
                    cusps.push((a.clone(), d.clone()));
                }
            }
            a = a + Integer::one();
        }
    }

    cusps
}

/// The number of cusps of `Gamma_0(N)` of the given width.
///
/// The cusp `a/d` (`d | N`) has width `N / gcd(d^2, N)`, and there are
/// `phi(gcd(d, N/d))` cusps of denominator `d`, so this is
///
/// ```text
///     #{cusps of width w} = sum_{d | N, N/gcd(d^2, N) = w} phi(gcd(d, N/d)).
/// ```
///
/// (It used to count the DIVISORS of `N` equal to `width`, i.e. return 1 for every
/// divisor and 0 otherwise -- unrelated to the number of cusps of that width.
/// For `Gamma_0(4)` the true answer is: one cusp of width 4 (the cusp 0) and two
/// of width 1; the old code said 1 for each of the widths 1, 2, 4.)
///
/// The tests gate the two identities that pin this down: the widths sum to the
/// index `[SL(2,Z) : Gamma_0(N)]`, and the counts sum to the number of cusps.
pub fn num_cusps_of_width(N: &Integer, width: &Integer) -> usize {
    if N <= &Integer::zero() || width <= &Integer::zero() {
        return 0;
    }

    let mut count = 0usize;
    for d in compute_divisors(N) {
        let d_squared = d.clone() * d.clone();
        let w = N / &d_squared.gcd(N);
        if &w == width {
            let n_over_d = N / &d;
            count += euler_phi(&d.gcd(&n_over_d)) as usize;
        }
    }
    count
}

/// Compute the q-expansion of the eta function
///
/// # Arguments
/// * `prec` - Precision (number of terms)
///
/// # Returns
/// Coefficients of q^n for n = 1/24, 25/24, 49/24, ...
pub fn qexp_eta(prec: usize) -> Vec<i64> {
    // η(τ) = q^(1/24) * ∏(1 - q^n)
    // This computes coefficients for the product part
    let mut coeffs = vec![0i64; prec];

    if prec > 0 {
        coeffs[0] = 1;
    }

    // Compute product (1 - q)(1 - q^2)(1 - q^3)...
    for n in 1..prec {
        // Multiply by (1 - q^n)
        for k in (n..prec).rev() {
            coeffs[k] -= coeffs[k - n];
        }
    }

    coeffs
}

/// A polynomial relation `sum c_{ij} x1^i x2^j = 0` between two eta products,
/// stored as its nonzero terms `((i, j), c)` with primitive integer coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EtaRelation {
    terms: Vec<((usize, usize), Integer)>,
}

impl EtaRelation {
    /// The nonzero terms `((i, j), c)`, sorted by `(i, j)`.
    pub fn terms(&self) -> &[((usize, usize), Integer)] {
        &self.terms
    }

    /// The coefficient of `x1^i x2^j`.
    pub fn coefficient(&self, i: usize, j: usize) -> Integer {
        self.terms
            .iter()
            .find(|(ij, _)| *ij == (i, j))
            .map(|(_, c)| c.clone())
            .unwrap_or_else(Integer::zero)
    }
}

impl std::fmt::Display for EtaRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for ((i, j), c) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            write!(f, "{c}")?;
            if *i > 0 {
                write!(f, "*x1^{i}")?;
            }
            if *j > 0 {
                write!(f, "*x2^{j}")?;
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

/// ALL polynomial relations `sum_{0 <= i, j <= degree} c_{ij} f^i g^j = 0` between
/// two elements `f`, `g` of the eta group of level `N`, as a basis of the
/// `Q`-vector space of such relations.
///
/// # This is exact, and CERTIFIED -- an empty result is a theorem
///
/// `f` and `g` must satisfy [`EtaGroupElement::eta_group_conditions`] (weight 0,
/// the two `24`-divisibility conditions, and the square condition); otherwise the
/// monomials `f^i g^j` are not modular functions on `X_0(N)` of one and the same
/// weight and character, the argument below collapses, and this returns `Err`
/// rather than a guess.
///
/// Given that, `eta` has no zeros on the upper half-plane, so every monomial
/// `f^i g^j` -- and hence every `P = sum c_{ij} f^i g^j` -- is a modular function
/// on `X_0(N)` whose divisor is supported on the CUSPS.  If `P != 0` then
/// `deg div(P) = 0`, and at each cusp `c`,
/// `ord_c(P) >= M_c := min_{i,j} ord_c(f^i g^j)` (a bound independent of the
/// unknown `c_{ij}`, computed from Ligozat's formula).  Therefore
///
/// ```text
///     ord_inf(P)  =  - sum_{c != inf} ord_c(P)  <=  - sum_{c != inf} M_c  =:  B
/// ```
///
/// (each cusp counted with its multiplicity `phi(gcd(d, N/d))`).  So:
///
/// > if the q-expansion of `P` at infinity vanishes through `q^B`, then `P = 0`.
///
/// The `c_{ij}` for which that happens are exactly the kernel of the (finite,
/// exact, integer) matrix of q-expansion coefficients of the monomials in the
/// exponent range `[v_min, B]`, and that kernel is what is returned.  Both
/// directions hold, so the returned list is complete: if it is EMPTY, there is
/// provably no relation in that monomial box.
///
/// The result is a basis of the relation SPACE inside the given monomial box (not
/// a minimal generating set of the relation ideal: at a large enough `degree` it
/// will contain multiples `x1 * R`, `x2 * R` of a relation `R`).
///
/// (This function used to return `vec![]` unconditionally, with both arguments
/// unused -- i.e. it claimed "there are no relations" for every input.)
pub fn eta_poly_relations(
    f: &EtaGroupElement,
    g: &EtaGroupElement,
    degree: usize,
) -> Result<Vec<EtaRelation>, String> {
    if f.level() != g.level() {
        return Err(format!(
            "the two eta products have different levels, {} and {}: they are not modular on a \
             common Gamma_0(N)",
            f.level(),
            g.level()
        ));
    }
    f.eta_group_conditions()
        .map_err(|e| format!("the first eta product is not in the eta group of level {}: {e}", f.level()))?;
    g.eta_group_conditions()
        .map_err(|e| format!("the second eta product is not in the eta group of level {}: {e}", g.level()))?;

    let n = f.level().clone();
    let monomials: Vec<(usize, usize)> = (0..=degree)
        .flat_map(|i| (0..=degree).map(move |j| (i, j)))
        .collect();

    // B = - sum over cusps c != infinity of  min_{i,j} ord_c(f^i g^j),
    // each cusp counted phi(gcd(d, N/d)) times.  (d = N is the cusp infinity.)
    let mut bound = Rational::zero();
    for d in compute_divisors(&n) {
        if d == n {
            continue; // the cusp infinity
        }
        let of = f.order_at_cusp(&d)?;
        let og = g.order_at_cusp(&d)?;
        let mut min_ord: Option<Rational> = None;
        for &(i, j) in &monomials {
            let o = of.clone() * Rational::new(Integer::from(i as u64), Integer::one()).unwrap()
                + og.clone() * Rational::new(Integer::from(j as u64), Integer::one()).unwrap();
            min_ord = Some(match min_ord {
                None => o,
                Some(m) => {
                    if o < m {
                        o
                    } else {
                        m
                    }
                }
            });
        }
        let n_over_d = &n / &d;
        let mult = euler_phi(&d.gcd(&n_over_d));
        let mult = Rational::new(Integer::from(mult), Integer::one()).unwrap();
        bound = bound - mult * min_ord.expect("the monomial box is nonempty");
    }
    if !bound.denominator().is_one() {
        return Err(format!(
            "the valence bound came out as {bound}, which is not an integer: the eta quotients \
             do not have integral orders at the cusps of Gamma_0({n})"
        ));
    }
    let bound: i64 = bound
        .numerator()
        .to_string()
        .parse()
        .map_err(|_| format!("valence bound {bound} does not fit in i64"))?;

    // q-expansions of the monomials.  v_min <= 0 always (the monomial 1 is in the
    // box), and B >= 0 >= v_min likewise, so the exponent window is nonempty.
    let v_f = f.qexp(1)?.valuation();
    let v_g = g.qexp(1)?.valuation();
    let v_min = monomials
        .iter()
        .map(|&(i, j)| (i as i64) * v_f + (j as i64) * v_g)
        .min()
        .expect("the monomial box is nonempty");
    let width = bound - v_min + 1;
    if width <= 0 {
        return Err(format!(
            "empty exponent window [{v_min}, {bound}]: this cannot happen for a box containing \
             the constant monomial"
        ));
    }
    let prec = width as usize;

    let fq = f.qexp(prec)?;
    let gq = g.qexp(prec)?;
    let mut expansions: HashMap<(usize, usize), LaurentQExpansion> = HashMap::new();
    for &(i, j) in &monomials {
        let mut m = LaurentQExpansion::one(prec);
        for _ in 0..i {
            m = m.mul(&fq, prec);
        }
        for _ in 0..j {
            m = m.mul(&gq, prec);
        }
        expansions.insert((i, j), m);
    }

    // rows = exponents v_min ..= B, columns = monomials
    let mut matrix: Vec<Vec<Rational>> = Vec::with_capacity(prec);
    for e in v_min..=bound {
        let row = monomials
            .iter()
            .map(|ij| {
                let c = expansions[ij]
                    .coefficient_of(e)
                    .expect("prec was chosen to cover the window");
                Rational::new(c, Integer::one()).expect("denominator 1 is nonzero")
            })
            .collect();
        matrix.push(row);
    }

    let kernel = kernel_basis(&matrix, monomials.len());

    let mut relations = Vec::new();
    for v in kernel {
        let terms = primitive_integer_vector(&v)
            .into_iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(k, c)| (monomials[k], c))
            .collect();
        relations.push(EtaRelation { terms });
    }

    Ok(relations)
}

/// A basis of the kernel `{x : A x = 0}` of an exact rational matrix, by
/// Gauss-Jordan elimination.  `matrix` is a list of rows; `cols` is their length.
fn kernel_basis(matrix: &[Vec<Rational>], cols: usize) -> Vec<Vec<Rational>> {
    let mut a: Vec<Vec<Rational>> = matrix.to_vec();
    let rows = a.len();

    // reduced row echelon form
    let mut pivot_of_col: Vec<Option<usize>> = vec![None; cols];
    let mut r = 0usize;
    for c in 0..cols {
        let Some(p) = (r..rows).find(|&i| !a[i][c].is_zero()) else {
            continue;
        };
        a.swap(r, p);
        let inv = a[r][c].clone();
        for j in 0..cols {
            a[r][j] = a[r][j].clone() / inv.clone();
        }
        for i in 0..rows {
            if i != r && !a[i][c].is_zero() {
                let factor = a[i][c].clone();
                for j in 0..cols {
                    a[i][j] = a[i][j].clone() - factor.clone() * a[r][j].clone();
                }
            }
        }
        pivot_of_col[c] = Some(r);
        r += 1;
        if r == rows {
            break;
        }
    }

    // one basis vector per free column
    let mut basis = Vec::new();
    for c in 0..cols {
        if pivot_of_col[c].is_some() {
            continue;
        }
        let mut v = vec![Rational::zero(); cols];
        v[c] = Rational::one();
        for (pc, pr) in pivot_of_col.iter().enumerate() {
            if let Some(pr) = pr {
                v[pc] = Rational::zero() - a[*pr][c].clone();
            }
        }
        basis.push(v);
    }

    basis
}

/// Scale a rational vector to primitive integers (content 1), with a positive
/// leading nonzero entry.
fn primitive_integer_vector(v: &[Rational]) -> Vec<Integer> {
    let mut den = Integer::one();
    for x in v {
        let d = x.denominator().clone();
        let g = den.gcd(&d);
        den = den.clone() / g * d;
    }
    let mut ints: Vec<Integer> = v
        .iter()
        .map(|x| x.numerator().clone() * (den.clone() / x.denominator().clone()))
        .collect();

    let mut content = Integer::zero();
    for x in &ints {
        content = content.gcd(x);
    }
    if !content.is_zero() && !content.is_one() {
        ints = ints.into_iter().map(|x| x / content.clone()).collect();
    }
    if let Some(first) = ints.iter().find(|x| !x.is_zero()) {
        if first.signum() < 0 {
            ints = ints.into_iter().map(|x| Integer::zero() - x).collect();
        }
    }
    ints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eta_group_element() {
        let mut powers = HashMap::new();
        powers.insert(Integer::one(), 24);

        let eta = EtaGroupElement::new(Integer::one(), powers);
        assert_eq!(eta.level(), &Integer::one());
        assert_eq!(eta.get_power(&Integer::one()), 24);
    }

    #[test]
    fn test_eta_group() {
        let G = EtaGroup::new(Integer::from(12));
        assert_eq!(G.level(), &Integer::from(12));
        assert!(!G.divisors().is_empty());
    }

    #[test]
    fn test_EtaGroup_class() {
        let G = eta_group_class(Integer::from(6));
        assert_eq!(G.level(), &Integer::from(6));
    }

    #[test]
    fn test_compute_divisors() {
        let divs = compute_divisors(&Integer::from(12));
        assert!(divs.contains(&Integer::one()));
        assert!(divs.contains(&Integer::from(12)));
        assert!(divs.contains(&Integer::from(2)));
        assert!(divs.contains(&Integer::from(3)));
        assert!(divs.contains(&Integer::from(4)));
        assert!(divs.contains(&Integer::from(6)));
    }

    #[test]
    fn test_eta_product() {
        // eta(tau)^24 = Delta(tau), the weight-12 modular discriminant: a
        // genuine integral-weight eta product (sum of exponents = 24, even).
        // The old test used eta^1, which has half-integral weight 1/2
        // (odd exponent sum) and is correctly rejected by `is_valid`.
        let mut powers = HashMap::new();
        powers.insert(Integer::one(), 24);

        let eta = eta_product(Integer::one(), powers);
        assert!(eta.is_valid());
    }

    fn eta_elt(level: u64, powers: &[(u64, i64)]) -> EtaGroupElement {
        let mut m = HashMap::new();
        for &(d, r) in powers {
            m.insert(Integer::from(d), r);
        }
        EtaGroupElement::new(Integer::from(level), m)
    }

    /// GATE: the eta q-expansion against Ramanujan's tau, from PARI
    /// (`vector(10, n, ramanujantau(n))`): eta(tau)^24 = Delta = sum tau(n) q^n,
    /// valuation 1.
    #[test]
    fn test_qexp_against_ramanujan_tau() {
        let delta = eta_elt(1, &[(1, 24)]);
        let q = delta.qexp(11).unwrap();
        assert_eq!(q.valuation(), 1, "Delta = q - 24q^2 + ... has valuation 1");

        let tau: [i64; 10] = [1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920];
        for (n, &t) in tau.iter().enumerate() {
            assert_eq!(
                q.coefficient_of(n as i64 + 1).unwrap(),
                Integer::from(t),
                "tau({})",
                n + 1
            );
        }
    }

    /// GATE: the Ligozat orders at the cusps, against the theorem that certifies
    /// them -- for an eta-group element (a modular FUNCTION on X_0(N)), the degree
    /// of the divisor is zero:
    ///   sum_{d | N} phi(gcd(d, N/d)) ord_{a/d}(f) = 0,
    /// and the order at the cusp d = N is the q-valuation.
    #[test]
    fn test_ligozat_orders_have_degree_zero() {
        let cases: Vec<(u64, Vec<(u64, i64)>)> = vec![
            (2, vec![(1, -24), (2, 24)]),
            (4, vec![(1, -8), (4, 8)]),
            (9, vec![(1, -3), (9, 3)]),
            (26, vec![(1, -2), (2, 2), (13, 2), (26, -2)]),
            (26, vec![(1, -2), (2, 4), (13, 2), (26, -4)]),
            (12, vec![(1, -12), (2, 12), (3, 12), (6, -12)]),
        ];

        for (n, powers) in cases {
            let f = eta_elt(n, &powers);
            f.eta_group_conditions()
                .unwrap_or_else(|e| panic!("level {n}: {e}"));

            let level = Integer::from(n);
            let mut degree = Rational::zero();
            for d in compute_divisors(&level) {
                let ord = f.order_at_cusp(&d).unwrap();
                assert!(
                    ord.denominator().is_one(),
                    "level {n}: ord at d = {d} is {ord}, not an integer"
                );
                let mult = Integer::from(euler_phi(&d.gcd(&(&level / &d))));
                degree = degree + Rational::new(mult, Integer::one()).unwrap() * ord;
            }
            assert_eq!(
                degree,
                Rational::zero(),
                "level {n}: deg div(f) must be 0 (valence formula)"
            );

            // the order at the cusp d = N is the q-valuation
            let v = f.qexp(1).unwrap().valuation();
            let ord_inf = f.order_at_cusp(&level).unwrap();
            assert_eq!(
                ord_inf,
                Rational::new(Integer::from(v), Integer::one()).unwrap(),
                "level {n}: ord at infinity (d = N) must be the q-valuation"
            );
        }
    }

    /// The eta-group conditions REFUSE what they must refuse.
    #[test]
    fn test_eta_group_conditions() {
        // Delta = eta^24 is weight 12, not a modular function
        assert!(eta_elt(1, &[(1, 24)]).eta_group_conditions().is_err());
        // 5 does not divide 4
        assert!(eta_elt(4, &[(1, -24), (5, 24)]).eta_group_conditions().is_err());
        // weight 0 but sum d r_d = 1*(-1) + 2*1 = 1, not divisible by 24
        assert!(eta_elt(2, &[(1, -1), (2, 1)]).eta_group_conditions().is_err());
        // genuine elements
        assert!(eta_elt(2, &[(1, -24), (2, 24)]).eta_group_conditions().is_ok());
        assert!(
            eta_elt(26, &[(1, -2), (2, 2), (13, 2), (26, -2)])
                .eta_group_conditions()
                .is_ok()
        );
    }

    /// GATE: `eta_poly_relations` on the level-26 pair
    ///   t = eta(2t)^2 eta(13t)^2 / (eta(t)^2 eta(26t)^2),
    ///   u = eta(2t)^4 eta(13t)^2 / (eta(t)^2 eta(26t)^4).
    ///
    /// Derived INDEPENDENTLY in python (sympy) from the definitions -- q-expansions
    /// straight from eta = q^(1/24) prod (1 - q^n), Ligozat orders, and an exact
    /// nullspace -- and then re-verified there to O(q^400), far beyond the
    /// certificate's 16 coefficients:
    ///
    ///   box i,j <= 1: NO relation      (certified: B = 1)
    ///   box i,j <= 2: NO relation      (certified: B = 2)
    ///   box i,j <= 3: a 2-dimensional relation space, containing
    ///        x1^3 x2 - 13 x1^3 - 4 x1^2 x2 - 4 x1 x2 - x2^2 + x2 = 0
    ///
    /// `eta_poly_relations` used to return `vec![]` for every input, i.e. it
    /// claimed the last of these had no relations either.
    #[test]
    fn test_eta_poly_relations_level_26() {
        let t = eta_elt(26, &[(1, -2), (2, 2), (13, 2), (26, -2)]);
        let u = eta_elt(26, &[(1, -2), (2, 4), (13, 2), (26, -4)]);

        // q-expansions, pinned against the python computation
        let tq = t.qexp(8).unwrap();
        assert_eq!(tq.valuation(), -1);
        assert_eq!(
            tq.coefficients(),
            &[1, 2, 3, 6, 9, 14, 22, 32].map(Integer::from)
        );
        let uq = u.qexp(8).unwrap();
        assert_eq!(uq.valuation(), -3);
        assert_eq!(
            uq.coefficients(),
            &[1, 2, 1, 2, 2, 0, 3, 2].map(Integer::from)
        );

        // no relations in the small boxes -- this is a certified NEGATIVE
        assert!(eta_poly_relations(&t, &u, 1).unwrap().is_empty());
        assert!(eta_poly_relations(&t, &u, 2).unwrap().is_empty());

        // and a 2-dimensional space of them at degree 3
        let rels = eta_poly_relations(&t, &u, 3).unwrap();
        assert_eq!(rels.len(), 2, "relations found: {rels:?}");

        // every returned relation really does vanish, checked on q-expansions to
        // precision 200 -- an order of magnitude past the certificate
        for r in &rels {
            assert!(
                relation_vanishes(&t, &u, r, 200),
                "returned relation {r} does not actually vanish"
            );
        }

        // the python-derived relation is in the space (it vanishes, and the space
        // is exactly the vanishing locus)
        let known = EtaRelation {
            terms: vec![
                ((0, 1), Integer::from(1)),
                ((0, 2), Integer::from(-1)),
                ((1, 1), Integer::from(-4)),
                ((2, 1), Integer::from(-4)),
                ((3, 0), Integer::from(-13)),
                ((3, 1), Integer::from(1)),
            ],
        };
        assert!(
            relation_vanishes(&t, &u, &known, 200),
            "the independently derived relation must vanish on our q-expansions"
        );
    }

    /// A relation that is NOT a relation must not vanish (the check above has
    /// teeth), and f, f^2 must produce x2 = x1^2.
    #[test]
    fn test_eta_poly_relations_self_check() {
        let t = eta_elt(26, &[(1, -2), (2, 2), (13, 2), (26, -2)]);
        let u = eta_elt(26, &[(1, -2), (2, 4), (13, 2), (26, -4)]);

        // perturb the known relation: it must stop vanishing
        let bogus = EtaRelation {
            terms: vec![
                ((0, 1), Integer::from(1)),
                ((0, 2), Integer::from(-1)),
                ((1, 1), Integer::from(-4)),
                ((2, 1), Integer::from(-4)),
                ((3, 0), Integer::from(-12)), // was -13
                ((3, 1), Integer::from(1)),
            ],
        };
        assert!(!relation_vanishes(&t, &u, &bogus, 200));

        // t and t^2 satisfy x2 - x1^2 = 0, and nothing else in the box i,j <= 1
        let t2 = eta_elt(26, &[(1, -4), (2, 4), (13, 4), (26, -4)]);
        let rels = eta_poly_relations(&t, &t2, 1).unwrap();
        assert!(rels.is_empty(), "x2 - x1^2 is not in the box i,j <= 1");

        let rels = eta_poly_relations(&t, &t2, 2).unwrap();
        assert!(!rels.is_empty());
        for r in &rels {
            assert!(relation_vanishes(&t, &t2, r, 200), "{r}");
        }
        // x2 - x1^2 itself is one of them
        let sq = EtaRelation {
            terms: vec![((0, 1), Integer::from(1)), ((2, 0), Integer::from(-1))],
        };
        assert!(rels.contains(&sq), "expected x2 - x1^2 among {rels:?}");
    }

    /// Two eta products whose relation-finding hypotheses fail are REFUSED, not
    /// silently answered with an empty list.
    #[test]
    fn test_eta_poly_relations_refuses_bad_input() {
        let t = eta_elt(26, &[(1, -2), (2, 2), (13, 2), (26, -2)]);
        // different level
        let other = eta_elt(4, &[(1, -8), (4, 8)]);
        assert!(eta_poly_relations(&t, &other, 2).is_err());
        // weight 12, not a modular function
        let delta = eta_elt(26, &[(1, 24)]);
        assert!(eta_poly_relations(&t, &delta, 2).is_err());
        assert!(eta_poly_relations(&delta, &t, 2).is_err());
    }

    /// Evaluate a relation on the q-expansions of `f` and `g` to `prec` terms and
    /// report whether every coefficient vanishes.
    fn relation_vanishes(
        f: &EtaGroupElement,
        g: &EtaGroupElement,
        rel: &EtaRelation,
        prec: usize,
    ) -> bool {
        let fq = f.qexp(prec).unwrap();
        let gq = g.qexp(prec).unwrap();

        let mut terms: Vec<(Integer, LaurentQExpansion)> = Vec::new();
        for ((i, j), c) in rel.terms() {
            let mut m = LaurentQExpansion::one(prec);
            for _ in 0..*i {
                m = m.mul(&fq, prec);
            }
            for _ in 0..*j {
                m = m.mul(&gq, prec);
            }
            terms.push((c.clone(), m));
        }

        let v_min = terms.iter().map(|(_, m)| m.valuation()).min().unwrap();
        let v_max = terms
            .iter()
            .map(|(_, m)| m.valuation() + prec as i64 - 1)
            .min()
            .unwrap();

        for e in v_min..=v_max {
            let mut sum = Integer::zero();
            for (c, m) in &terms {
                sum = sum + c.clone() * m.coefficient_of(e).unwrap();
            }
            if !sum.is_zero() {
                return false;
            }
        }
        true
    }

    /// GATE: `all_cusps` against the certified cusp count in `dims`, and against
    /// the definition (a/d ~ a'/d iff a = a' mod gcd(d, N/d)).  It used to list
    /// every a/c with c | N, gcd(a, c) = 1 -- 5 "cusps" for Gamma_0(4), which has
    /// 3 -- plus a duplicate infinity.
    #[test]
    fn test_all_cusps() {
        for n in 1..=40u64 {
            let cusps = all_cusps(Integer::from(n));
            let expected = crate::dims::gamma0_invariants(n).unwrap().cusps;
            assert_eq!(
                cusps.len() as u128,
                expected,
                "#cusps(Gamma0({n})): got {cusps:?}"
            );

            // representatives are pairwise inequivalent and in lowest terms
            let level = Integer::from(n);
            let mut classes = Vec::new();
            for (a, d) in &cusps {
                assert!((&level % d).is_zero(), "denominator {d} must divide {n}");
                assert!(
                    a.gcd(d).is_one() || (d.is_one() && a.is_zero()),
                    "{a}/{d} is not in lowest terms"
                );
                let g = d.gcd(&(&level / d));
                let key = (d.clone(), a % &g);
                assert!(!classes.contains(&key), "{a}/{d} listed twice");
                classes.push(key);
            }
        }

        // Gamma_0(4) has exactly 3 cusps: 0/1, 1/2, 1/4 (= infinity)
        let c4 = all_cusps(Integer::from(4));
        assert_eq!(c4.len(), 3);
        assert!(c4.contains(&(Integer::zero(), Integer::one())));
        assert!(c4.contains(&(Integer::one(), Integer::from(4))));
    }

    /// GATE: `num_cusps_of_width` against the two identities that pin it down --
    /// the widths sum to the index, and the counts sum to the number of cusps.
    /// It used to return 1 for every divisor of N and 0 otherwise.
    #[test]
    fn test_num_cusps_of_width() {
        for n in 1..=40u64 {
            let level = Integer::from(n);
            let inv = crate::dims::gamma0_invariants(n).unwrap();

            let mut total_cusps = 0u128;
            let mut total_width = 0u128;
            for w in compute_divisors(&level) {
                let c = num_cusps_of_width(&level, &w) as u128;
                let w_u128: u128 = w.to_string().parse().unwrap();
                total_cusps += c;
                total_width += c * w_u128;
            }
            assert_eq!(total_cusps, inv.cusps, "sum of cusp counts, N = {n}");
            assert_eq!(
                total_width, inv.index,
                "sum of cusp widths = [SL(2,Z) : Gamma_0({n})]"
            );
        }

        // Gamma_0(4): one cusp of width 4 (the cusp 0), two of width 1 (1/2 and
        // infinity), none of width 2.  PARI: fordiv(4, d, [d, 4/gcd(d^2,4)]).
        let n4 = Integer::from(4);
        assert_eq!(num_cusps_of_width(&n4, &Integer::from(4)), 1);
        assert_eq!(num_cusps_of_width(&n4, &Integer::from(2)), 0);
        assert_eq!(num_cusps_of_width(&n4, &Integer::one()), 2);
    }

    #[test]
    fn test_cusp_family() {
        let mut family = CuspFamily::new(Integer::from(12), Integer::from(4));
        family.add_cusp(Integer::one(), Integer::from(4));
        assert_eq!(family.len(), 1);
        assert!(!family.is_empty());
    }

    #[test]
    fn test_qexp_eta() {
        let coeffs = qexp_eta(10);
        assert_eq!(coeffs.len(), 10);
        assert_eq!(coeffs[0], 1);
    }

    #[test]
    fn test_order_at_infinity() {
        let mut powers = HashMap::new();
        powers.insert(Integer::one(), 24);

        let eta = EtaGroupElement::new(Integer::one(), powers);
        let order = eta.order_at_infinity();
        assert_eq!(order, Rational::one());
    }

    #[test]
    fn test_eta_multiply() {
        let mut powers1 = HashMap::new();
        powers1.insert(Integer::one(), 1);

        let mut powers2 = HashMap::new();
        powers2.insert(Integer::one(), 2);

        let eta1 = EtaGroupElement::new(Integer::one(), powers1);
        let eta2 = EtaGroupElement::new(Integer::one(), powers2);

        let product = eta1.mul(&eta2).unwrap();
        assert_eq!(product.get_power(&Integer::one()), 3);
    }
}
