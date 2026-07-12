//! Multivariate polynomials (polynomials in multiple variables)

use rustmath_core::Ring;
use std::collections::BTreeMap;
use std::fmt;

/// A monomial in multiple variables
///
/// Represents x₀^e₀ × x₁^e₁ × ... × xₙ^eₙ
/// Stored as a map from variable index to exponent
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Monomial {
    /// Map from variable index to exponent
    /// Only non-zero exponents are stored
    exponents: BTreeMap<usize, u32>,
}

impl Monomial {
    /// Create a new monomial
    pub fn new() -> Self {
        Monomial {
            exponents: BTreeMap::new(),
        }
    }

    /// Create a monomial from exponents
    pub fn from_exponents(exponents: BTreeMap<usize, u32>) -> Self {
        let mut filtered = BTreeMap::new();
        for (var, exp) in exponents {
            if exp > 0 {
                filtered.insert(var, exp);
            }
        }
        Monomial {
            exponents: filtered,
        }
    }

    /// Create a monomial representing a single variable to a power
    pub fn variable(var: usize, power: u32) -> Self {
        let mut exponents = BTreeMap::new();
        if power > 0 {
            exponents.insert(var, power);
        }
        Monomial { exponents }
    }

    /// Get the exponent for a variable
    pub fn exponent(&self, var: usize) -> u32 {
        *self.exponents.get(&var).unwrap_or(&0)
    }

    /// Get the total degree (sum of all exponents)
    pub fn degree(&self) -> u32 {
        self.exponents.values().sum()
    }

    /// Multiply two monomials
    pub fn mul(&self, other: &Monomial) -> Monomial {
        let mut exponents = self.exponents.clone();
        for (var, exp) in &other.exponents {
            *exponents.entry(*var).or_insert(0) += exp;
        }
        Monomial { exponents }
    }

    /// Divide two monomials (returns None if not divisible)
    pub fn div(&self, other: &Monomial) -> Option<Monomial> {
        let mut exponents = BTreeMap::new();

        // Check if other divides self
        for (var, exp) in &other.exponents {
            let self_exp = self.exponent(*var);
            if self_exp < *exp {
                return None; // Not divisible
            }
            let diff = self_exp - exp;
            if diff > 0 {
                exponents.insert(*var, diff);
            }
        }

        // Add remaining variables from self
        for (var, exp) in &self.exponents {
            if !other.exponents.contains_key(var) {
                exponents.insert(*var, *exp);
            }
        }

        Some(Monomial { exponents })
    }

    /// Compute LCM (least common multiple) of two monomials
    pub fn lcm(&self, other: &Monomial) -> Monomial {
        let mut exponents = BTreeMap::new();

        // Get all variables from both monomials
        let mut all_vars = self.exponents.keys().copied().collect::<Vec<_>>();
        for var in other.exponents.keys() {
            if !all_vars.contains(var) {
                all_vars.push(*var);
            }
        }

        for var in all_vars {
            let exp = self.exponent(var).max(other.exponent(var));
            if exp > 0 {
                exponents.insert(var, exp);
            }
        }

        Monomial { exponents }
    }

    /// Check if this is the constant monomial (1)
    pub fn is_one(&self) -> bool {
        self.exponents.is_empty()
    }

    /// Get all variables that appear in this monomial
    pub fn variables(&self) -> Vec<usize> {
        self.exponents.keys().copied().collect()
    }

    /// Get an iterator over the (variable, exponent) pairs
    pub fn exponents(&self) -> impl Iterator<Item = (&usize, &u32)> {
        self.exponents.iter()
    }

    /// Iterate over the (variable, exponent) pairs
    pub fn iter_exponents(&self) -> impl Iterator<Item = (&usize, &u32)> {
        self.exponents.iter()
    }

    /// Compare monomials using lexicographic ordering
    pub fn cmp_lex(&self, other: &Monomial) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // Get all variables
        let mut all_vars = self.exponents.keys().copied().collect::<Vec<_>>();
        for var in other.exponents.keys() {
            if !all_vars.contains(var) {
                all_vars.push(*var);
            }
        }
        all_vars.sort_unstable();

        // Compare from left to right
        for var in all_vars {
            let cmp = self.exponent(var).cmp(&other.exponent(var));
            if cmp != Ordering::Equal {
                return cmp;
            }
        }

        Ordering::Equal
    }

    /// Total degree restricted to the variables with index `< block`.
    pub fn block_degree(&self, block: usize) -> u32 {
        self.exponents
            .iter()
            .filter(|(var, _)| **var < block)
            .map(|(_, exp)| *exp)
            .sum()
    }

    /// Compare monomials using the block elimination order for the first `block` variables.
    ///
    /// Any monomial that involves one of the variables `x_0, ..., x_{block-1}` is greater
    /// than any monomial that involves none of them; among monomials with the same
    /// "block degree" the comparison falls back to Grevlex on the full monomial.
    ///
    /// This is the *elimination property*: if the leading monomial of `g` involves no
    /// variable of the block, then no monomial of `g` does, because a monomial of positive
    /// block degree would have outranked it.
    pub fn cmp_elimination(&self, other: &Monomial, block: usize) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        let cmp = self.block_degree(block).cmp(&other.block_degree(block));
        if cmp != Ordering::Equal {
            return cmp;
        }

        self.cmp_grevlex(other)
    }

    /// Apply a variable renaming to this monomial.
    ///
    /// `map(v)` gives the new index of variable `v`. The caller is responsible for
    /// `map` being injective on the variables that actually occur; otherwise exponents
    /// of distinct variables would be silently merged, so this is checked and reported.
    pub fn relabel<F>(&self, map: F) -> Result<Monomial, String>
    where
        F: Fn(usize) -> usize,
    {
        let mut exponents = BTreeMap::new();
        for (var, exp) in &self.exponents {
            let new_var = map(*var);
            if exponents.insert(new_var, *exp).is_some() {
                return Err(format!(
                    "relabel: variable map is not injective (two variables both map to x{})",
                    new_var
                ));
            }
        }
        Ok(Monomial { exponents })
    }

    /// Compare monomials using graded lexicographic ordering
    pub fn cmp_grlex(&self, other: &Monomial) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // First compare total degree
        let deg_cmp = self.degree().cmp(&other.degree());
        if deg_cmp != Ordering::Equal {
            return deg_cmp;
        }

        // Then use lex
        self.cmp_lex(other)
    }

    /// Compare monomials using graded reverse lexicographic ordering
    pub fn cmp_grevlex(&self, other: &Monomial) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // First compare total degree
        let deg_cmp = self.degree().cmp(&other.degree());
        if deg_cmp != Ordering::Equal {
            return deg_cmp;
        }

        // Then use reverse lex (compare from right to left, with reversed comparison)
        let mut all_vars = self.exponents.keys().copied().collect::<Vec<_>>();
        for var in other.exponents.keys() {
            if !all_vars.contains(var) {
                all_vars.push(*var);
            }
        }
        all_vars.sort_unstable();
        all_vars.reverse();

        for var in all_vars {
            let cmp = other.exponent(var).cmp(&self.exponent(var)); // Note: reversed
            if cmp != Ordering::Equal {
                return cmp;
            }
        }

        Ordering::Equal
    }
}

impl Default for Monomial {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_one() {
            write!(f, "1")?;
            return Ok(());
        }

        let mut first = true;
        for (var, exp) in &self.exponents {
            if !first {
                write!(f, "*")?;
            }
            first = false;

            write!(f, "x{}", var)?;
            if *exp > 1 {
                write!(f, "^{}", exp)?;
            }
        }
        Ok(())
    }
}

/// A multivariate polynomial over a ring R
///
/// Represented as a map from monomials to coefficients
#[derive(Clone, Debug, PartialEq)]
pub struct MultivariatePolynomial<R: Ring> {
    /// Map from monomial to coefficient
    /// Only non-zero coefficients are stored
    terms: BTreeMap<Monomial, R>,
}

/// Type alias for convenience
pub type MultiPoly<R> = MultivariatePolynomial<R>;

impl<R: Ring> MultivariatePolynomial<R> {
    /// Create a new zero polynomial
    pub fn zero() -> Self {
        MultivariatePolynomial {
            terms: BTreeMap::new(),
        }
    }

    /// Create a constant polynomial
    pub fn constant(c: R) -> Self {
        if c.is_zero() {
            return Self::zero();
        }

        let mut terms = BTreeMap::new();
        terms.insert(Monomial::new(), c);
        MultivariatePolynomial { terms }
    }

    /// Create a polynomial representing a single variable
    pub fn variable(var: usize) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(Monomial::variable(var, 1), R::one());
        MultivariatePolynomial { terms }
    }

    /// Add a term to the polynomial
    pub fn add_term(&mut self, monomial: Monomial, coeff: R) {
        if coeff.is_zero() {
            return;
        }

        // Clone monomial for potential removal later
        let monomial_key = monomial.clone();
        let entry = self.terms.entry(monomial).or_insert_with(|| R::zero());
        *entry = entry.clone() + coeff;

        // Remove if it became zero
        if entry.is_zero() {
            self.terms.remove(&monomial_key);
        }
    }

    /// Get the coefficient of a monomial
    pub fn coefficient(&self, monomial: &Monomial) -> R {
        self.terms.get(monomial).cloned().unwrap_or_else(|| R::zero())
    }

    /// Get the total degree of the polynomial
    pub fn degree(&self) -> Option<u32> {
        self.terms.keys().map(|m| m.degree()).max()
    }

    /// Get the degree in a specific variable
    pub fn degree_in(&self, var: usize) -> Option<u32> {
        self.terms.keys().map(|m| m.exponent(var)).max()
    }

    /// Check if the polynomial is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if the polynomial is a constant
    pub fn is_constant(&self) -> bool {
        self.terms.len() <= 1 && self.terms.keys().all(|m| m.is_one())
    }

    /// Get an iterator over the terms (monomial, coefficient pairs)
    pub fn terms(&self) -> impl Iterator<Item = (&Monomial, &R)> {
        self.terms.iter()
    }

    /// Get all variables that appear in the polynomial
    pub fn variables(&self) -> Vec<usize> {
        let mut vars = std::collections::HashSet::new();
        for monomial in self.terms.keys() {
            for var in monomial.variables() {
                vars.insert(var);
            }
        }
        let mut result: Vec<_> = vars.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Number of terms in the polynomial
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Apply a variable renaming to every monomial of this polynomial.
    ///
    /// `map(v)` gives the new index of variable `v`. Errors if `map` is not injective
    /// on the variables actually occurring in this polynomial.
    pub fn relabel<F>(&self, map: F) -> Result<Self, String>
    where
        F: Fn(usize) -> usize + Copy,
    {
        let mut result = Self::zero();
        for (monomial, coeff) in &self.terms {
            result.add_term(monomial.relabel(map)?, coeff.clone());
        }
        Ok(result)
    }

    /// The largest variable index occurring in this polynomial, if any.
    pub fn max_variable(&self) -> Option<usize> {
        self.terms.keys().filter_map(|m| m.variables().last().copied()).max()
    }

    /// Substitute polynomials for variables: the ring homomorphism `x_v ↦ images[v]`.
    ///
    /// A variable absent from `images` is left alone (it maps to itself), so this is the
    /// unique `R`-algebra map determined by the given assignments. Unlike [`evaluate`],
    /// which lands in the coefficient ring, the images here are polynomials, so the result
    /// is again a polynomial.
    ///
    /// [`evaluate`]: Self::evaluate
    pub fn substitute(&self, images: &BTreeMap<usize, Self>) -> Self {
        let mut result = Self::zero();

        for (monomial, coeff) in &self.terms {
            let mut value = Self::constant(coeff.clone());
            for (var, exp) in monomial.iter_exponents() {
                let base = match images.get(var) {
                    Some(image) => image.clone(),
                    None => Self::variable(*var),
                };
                // Binary exponentiation: monomial exponents are small but the images can
                // be large, so squaring is worth it.
                let mut acc = Self::constant(R::one());
                let mut sq = base;
                let mut e = *exp;
                while e > 0 {
                    if e & 1 == 1 {
                        acc = acc * sq.clone();
                    }
                    e >>= 1;
                    if e > 0 {
                        sq = sq.clone() * sq;
                    }
                }
                value = value * acc;
            }
            result = result + value;
        }

        result
    }

    /// The order of vanishing of `self` at the origin: the least total degree of a term.
    ///
    /// For a hypersurface `V(f)` this is the *multiplicity* of `V(f)` at the origin. The
    /// zero polynomial has no least degree, so it returns `None`.
    pub fn order_at_origin(&self) -> Option<u32> {
        self.terms.keys().map(|m| m.degree()).min()
    }

    /// Iterate over the terms (monomial, coefficient) pairs
    pub fn iter_terms(&self) -> impl Iterator<Item = (&Monomial, &R)> {
        self.terms.iter()
    }

    /// Get the leading monomial using a specific monomial ordering
    pub fn leading_monomial<F>(&self, cmp: F) -> Option<Monomial>
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering,
    {
        if self.is_zero() {
            return None;
        }

        let mut max_monomial = None;
        for monomial in self.terms.keys() {
            match &max_monomial {
                None => max_monomial = Some(monomial.clone()),
                Some(current_max) => {
                    if cmp(monomial, current_max) == std::cmp::Ordering::Greater {
                        max_monomial = Some(monomial.clone());
                    }
                }
            }
        }

        max_monomial
    }

    /// Get the leading coefficient using a specific monomial ordering
    pub fn leading_coefficient<F>(&self, cmp: F) -> Option<R>
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering,
    {
        self.leading_monomial(cmp).map(|m| self.coefficient(&m))
    }

    /// Get the leading term (monomial, coefficient) using a specific monomial ordering
    pub fn leading_term<F>(&self, cmp: F) -> Option<(Monomial, R)>
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering,
    {
        let lm = self.leading_monomial(cmp)?;
        let lc = self.coefficient(&lm);
        Some((lm, lc))
    }

    /// Multiply by a scalar
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return Self::zero();
        }

        let mut result = Self::zero();
        for (monomial, coeff) in &self.terms {
            result.add_term(monomial.clone(), coeff.clone() * scalar.clone());
        }
        result
    }

    /// Multiply by a monomial
    pub fn monomial_mul(&self, monomial: &Monomial, coeff: &R) -> Self {
        let mut result = Self::zero();
        for (m, c) in &self.terms {
            result.add_term(m.mul(monomial), c.clone() * coeff.clone());
        }
        result
    }

    /// The coefficient `q` with `q · d = r`, when one can be *exhibited* without dividing.
    ///
    /// # Why this is not simply `r / d`
    ///
    /// `R: Ring` has no division: [`rustmath_core::Ring`] provides `+ - * zero one`, and
    /// nothing else. So a division step in the coefficient ring is only available when we
    /// can produce a candidate quotient and *verify* it by multiplying back. The four
    /// candidates below cover every case where the answer is forced by the ring axioms
    /// alone:
    ///
    /// - `d = 1`   ⇒ `q = r`   (the monic case — this is the one that matters);
    /// - `d = −1`  ⇒ `q = −r`;
    /// - `r = d`   ⇒ `q = 1`;
    /// - `r = −d`  ⇒ `q = −1`.
    ///
    /// Every candidate is checked with `q · d == r` before it is returned, so a `Some`
    /// is always a genuine exact division. `None` means *this ring cannot be divided in
    /// here*, not that no quotient exists mathematically (over `Q`, `d = 3, r = 2` has the
    /// quotient `2/3`, which we deliberately do not use — the caller is `R: Ring`, and a
    /// caller who wants real coefficient division must use the `R: Field` engine in
    /// [`crate::groebner`], which does).
    ///
    /// The previous code multiplied `r · d` and called it "a placeholder", then used the
    /// coefficient `1` regardless. For non-monic divisors the leading term therefore never
    /// cancelled, and the division loop spun forever.
    fn exact_quotient_coefficient(r: &R, d: &R) -> Option<R> {
        if d.is_zero() {
            return None;
        }
        if d.is_one() {
            return Some(r.clone());
        }
        let candidates = [r.clone(), -r.clone(), R::one(), -R::one()];
        candidates
            .into_iter()
            .find(|q| q.clone() * d.clone() == *r)
    }

    /// Divide this polynomial by a divisor, returning `(quotient, remainder)`.
    ///
    /// Multivariate division with respect to a monomial ordering. Guarantees
    /// `self == quotient · divisor + remainder`, and every step is an *exact* coefficient
    /// division (see [`Self::exact_quotient_coefficient`]).
    ///
    /// # Limits over a general ring
    ///
    /// A term of `remainder` may still be divisible by the leading monomial of `divisor`:
    /// that happens exactly when the leading coefficient of the divisor cannot be divided
    /// into it in `R`. Such a term is left in the remainder — "this divisor does not
    /// apply" — rather than being cancelled with a made-up coefficient. Over a field
    /// (where every non-zero coefficient divides) the remainder is the true normal form.
    ///
    /// Terminates: every iteration removes the current leading monomial of the working
    /// polynomial and introduces only strictly smaller ones, and the ordering is a
    /// well-ordering.
    pub fn divide<F>(&self, divisor: &Self, cmp: F) -> (Self, Self)
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering + Copy,
    {
        let mut quotient = Self::zero();
        let mut remainder = Self::zero();
        let mut work = self.clone();

        let Some((d_lm, d_lc)) = divisor.leading_term(cmp) else {
            // Division by the zero polynomial: nothing is divided, everything remains.
            return (Self::zero(), self.clone());
        };

        while !work.is_zero() {
            let Some((w_lm, w_lc)) = work.leading_term(cmp) else {
                break;
            };

            let step = w_lm.div(&d_lm).and_then(|q_mono| {
                Self::exact_quotient_coefficient(&w_lc, &d_lc).map(|q_coeff| (q_mono, q_coeff))
            });

            match step {
                Some((q_mono, q_coeff)) => {
                    quotient.add_term(q_mono.clone(), q_coeff.clone());
                    work = work - divisor.monomial_mul(&q_mono, &q_coeff);
                }
                None => {
                    // Not reducible by this divisor: retire the term to the remainder.
                    remainder.add_term(w_lm.clone(), w_lc.clone());
                    work.add_term(w_lm, -w_lc);
                }
            }
        }

        (quotient, remainder)
    }

    /// Divide by several divisors, returning `(quotients, remainder)`.
    ///
    /// The multivariate division algorithm: guarantees
    /// `self == Σ quotients[i] · divisors[i] + remainder`.
    ///
    /// The same two honesty constraints as [`Self::divide`] apply: a coefficient is only
    /// divided when the quotient can be exhibited and verified in `R`, and a term that no
    /// divisor applies to is *moved into the remainder* — it is never dropped, and the
    /// loop never spins on it.
    ///
    /// The previous version had two defects that this fixes:
    /// 1. it used the coefficient `1` for every division step, so a non-monic divisor
    ///    never cancelled the leading term and the loop ran forever;
    /// 2. when no divisor applied it *deleted* the leading term and stopped, so the
    ///    identity `self = Σ qᵢ·gᵢ + r` failed and, e.g., `x² + y` "reduced" to `0`
    ///    modulo `{x}` — which made ideal membership answer `true` for non-members.
    pub fn divide_multiple<F>(&self, divisors: &[Self], cmp: F) -> (Vec<Self>, Self)
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering + Copy,
    {
        let mut quotients = vec![Self::zero(); divisors.len()];
        let mut remainder = Self::zero();
        let mut work = self.clone();

        // (index, leading monomial, leading coefficient) of each usable divisor.
        let leads: Vec<(usize, Monomial, R)> = divisors
            .iter()
            .enumerate()
            .filter_map(|(i, d)| d.leading_term(cmp).map(|(lm, lc)| (i, lm, lc)))
            .collect();

        while !work.is_zero() {
            let Some((w_lm, w_lc)) = work.leading_term(cmp) else {
                break;
            };

            let mut step = None;
            for (i, d_lm, d_lc) in &leads {
                let Some(q_mono) = w_lm.div(d_lm) else {
                    continue;
                };
                let Some(q_coeff) = Self::exact_quotient_coefficient(&w_lc, d_lc) else {
                    // The monomial divides but the coefficient does not: this divisor
                    // does not apply. Try the next one.
                    continue;
                };
                step = Some((*i, q_mono, q_coeff));
                break;
            }

            match step {
                Some((i, q_mono, q_coeff)) => {
                    quotients[i].add_term(q_mono.clone(), q_coeff.clone());
                    work = work - divisors[i].monomial_mul(&q_mono, &q_coeff);
                }
                None => {
                    remainder.add_term(w_lm.clone(), w_lc.clone());
                    work.add_term(w_lm, -w_lc);
                }
            }
        }

        (quotients, remainder)
    }

    /// Evaluate the polynomial at a given point
    ///
    /// Substitutes the variables with the given values and returns the result
    pub fn evaluate(&self, point: &[R]) -> R {
        let mut result = R::zero();

        for (monomial, coeff) in &self.terms {
            // Evaluate the monomial at the point
            let mut monomial_value = R::one();
            for (var, exp) in monomial.iter_exponents() {
                if *var < point.len() {
                    // Compute point[var]^exp
                    let mut power = R::one();
                    for _ in 0..*exp {
                        power = power * point[*var].clone();
                    }
                    monomial_value = monomial_value * power;
                } else {
                    // Variable not provided, treat as 0
                    if *exp > 0 {
                        monomial_value = R::zero();
                        break;
                    }
                }
            }

            result = result + coeff.clone() * monomial_value;
        }

        result
    }

    /// Compute the partial derivative with respect to a variable
    ///
    /// Returns ∂f/∂xᵢ where i is the variable index
    pub fn partial_derivative(&self, var: usize) -> Self {
        let mut result = Self::zero();

        for (monomial, coeff) in &self.terms {
            let exp = monomial.exponent(var);

            if exp == 0 {
                // Derivative is zero for terms not containing this variable
                continue;
            }

            // Derivative: d/dx(c*x^n) = c*n*x^(n-1)
            // Multiply coefficient by exp using repeated addition
            let mut new_coeff = R::zero();
            for _ in 0..exp {
                new_coeff = new_coeff + coeff.clone();
            }

            // Create new monomial with exponent decreased by 1
            let mut new_exponents = BTreeMap::new();
            for (v, e) in monomial.iter_exponents() {
                if *v == var {
                    if *e > 1 {
                        new_exponents.insert(*v, e - 1);
                    }
                } else {
                    new_exponents.insert(*v, *e);
                }
            }

            result.add_term(Monomial::from_exponents(new_exponents), new_coeff);
        }

        result
    }

    /// Get the total degree of the polynomial (same as degree)
    pub fn total_degree(&self) -> usize {
        self.degree().unwrap_or(0) as usize
    }
}

impl<R: Ring> std::ops::Add for MultivariatePolynomial<R> {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        for (monomial, coeff) in other.terms {
            self.add_term(monomial, coeff);
        }
        self
    }
}

impl<R: Ring> std::ops::Sub for MultivariatePolynomial<R> {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        for (monomial, coeff) in other.terms {
            self.add_term(monomial, -coeff);
        }
        self
    }
}

impl<R: Ring> std::ops::Mul for MultivariatePolynomial<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let mut result = Self::zero();

        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                let new_monomial = m1.mul(m2);
                let new_coeff = c1.clone() * c2.clone();
                result.add_term(new_monomial, new_coeff);
            }
        }

        result
    }
}

impl<R: Ring> std::ops::Neg for MultivariatePolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut result = Self::zero();
        for (monomial, coeff) in self.terms {
            result.add_term(monomial, -coeff);
        }
        result
    }
}

impl<R: Ring> fmt::Display for MultivariatePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (monomial, coeff) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if monomial.is_one() {
                write!(f, "{}", coeff)?;
            } else if coeff.is_one() {
                write!(f, "{}", monomial)?;
            } else {
                write!(f, "{}*{}", coeff, monomial)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitute_is_a_ring_homomorphism() {
        // Work over Q so the coefficients are exact.
        use rustmath_integers::Integer;
        use rustmath_rationals::Rational;
        type P = MultivariatePolynomial<Rational>;

        let q = |n: i64| Rational::from_integer(Integer::from(n));
        let x: P = P::variable(0);
        let y: P = P::variable(1);

        // f = y^2 - x^3 (the cusp). Substitute the blow-up chart x -> u0, y -> u0*u1.
        let f = y.clone() * y.clone() - x.clone() * x.clone() * x.clone();

        let u0: P = P::variable(0);
        let u1: P = P::variable(1);
        let mut images = BTreeMap::new();
        images.insert(0, u0.clone());
        images.insert(1, u0.clone() * u1.clone());

        // Expected: (u0*u1)^2 - u0^3 = u0^2*u1^2 - u0^3.
        let expected = u0.clone() * u0.clone() * u1.clone() * u1.clone()
            - u0.clone() * u0.clone() * u0.clone();
        assert_eq!(f.substitute(&images), expected);

        // A variable absent from the map is left alone.
        let mut only_x = BTreeMap::new();
        only_x.insert(0, P::constant(q(2)));
        // f(2, y) = y^2 - 8
        let got = f.substitute(&only_x);
        assert_eq!(got, y.clone() * y.clone() - P::constant(q(8)));

        // Homomorphism property: (a*b)(σ) = a(σ) * b(σ), and likewise for +.
        let a = x.clone() + P::constant(q(1));
        let b = y.clone() - P::constant(q(3));
        assert_eq!(
            (a.clone() * b.clone()).substitute(&images),
            a.substitute(&images) * b.substitute(&images)
        );
        assert_eq!(
            (a.clone() + b.clone()).substitute(&images),
            a.substitute(&images) + b.substitute(&images)
        );

        // An empty substitution is the identity.
        assert_eq!(f.substitute(&BTreeMap::new()), f);
    }

    #[test]
    fn test_order_at_origin() {
        use rustmath_integers::Integer;
        use rustmath_rationals::Rational;
        type P = MultivariatePolynomial<Rational>;

        let q = |n: i64| Rational::from_integer(Integer::from(n));
        let x: P = P::variable(0);
        let y: P = P::variable(1);

        // y^2 - x^2 - x^3: lowest total degree 2 (the node's multiplicity).
        let node = y.clone() * y.clone() - x.clone() * x.clone() - x.clone() * x.clone() * x.clone();
        assert_eq!(node.order_at_origin(), Some(2));

        // y^3 - x^3: an ordinary triple point.
        let triple = y.clone() * y.clone() * y.clone() - x.clone() * x.clone() * x.clone();
        assert_eq!(triple.order_at_origin(), Some(3));

        // A unit does not vanish at the origin at all.
        assert_eq!(P::constant(q(5)).order_at_origin(), Some(0));

        // A smooth point: order 1.
        assert_eq!((x.clone() + y.clone()).order_at_origin(), Some(1));

        // The zero polynomial has no least degree.
        assert_eq!(P::zero().order_at_origin(), None);
    }

    #[test]
    fn test_monomial_creation() {
        let m = Monomial::variable(0, 2);
        assert_eq!(m.exponent(0), 2);
        assert_eq!(m.degree(), 2);
    }

    #[test]
    fn test_monomial_multiplication() {
        let m1 = Monomial::variable(0, 2); // x₀²
        let m2 = Monomial::variable(1, 3); // x₁³

        let m3 = m1.mul(&m2); // x₀²x₁³
        assert_eq!(m3.exponent(0), 2);
        assert_eq!(m3.exponent(1), 3);
        assert_eq!(m3.degree(), 5);
    }

    #[test]
    fn test_polynomial_creation() {
        let p: MultivariatePolynomial<i32> = MultivariatePolynomial::constant(5);
        assert!(p.is_constant());
        assert_eq!(p.coefficient(&Monomial::new()), 5);
    }

    #[test]
    fn test_polynomial_variable() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        assert!(!x.is_constant());
        assert!(!y.is_constant());
        assert_eq!(x.degree(), Some(1));
        assert_eq!(y.degree(), Some(1));
    }

    #[test]
    fn test_polynomial_addition() {
        // x + y
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let sum = x + y;
        assert_eq!(sum.num_terms(), 2);
        assert_eq!(sum.degree(), Some(1));
    }

    #[test]
    fn test_polynomial_multiplication() {
        // (x + 1) * (y + 2) = xy + 2x + y + 2
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);
        let one = MultivariatePolynomial::constant(1);
        let two = MultivariatePolynomial::constant(2);

        let p1 = x.clone() + one;
        let p2 = y + two;
        let product = p1 * p2;

        assert_eq!(product.num_terms(), 4);
        assert_eq!(product.degree(), Some(2)); // xy has degree 2
    }

    #[test]
    fn test_polynomial_display() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let p = x + y;
        let display = format!("{}", p);
        assert!(display.contains("x0") || display.contains("x1"));
    }
}
