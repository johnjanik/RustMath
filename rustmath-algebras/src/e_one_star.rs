//! E_1^* Hopf Algebra Structure
//!
//! This module implements the E_1^* page of a spectral sequence as a bigraded
//! Hopf algebra with a differential operator.
//!
//! # Mathematical Background
//!
//! The E_1 page (E_1^{*,*}) of a spectral sequence is a bigraded module
//! E_1 = ⊕_{p,q} E_1^{p,q} equipped with:
//! - Bigrading: (p, q) where p is the filtration degree and q is the total degree
//! - Differential d_1: E_1^{p,q} → E_1^{p+1,q} (or E_1^{p-1,q+r} depending on convention)
//! - Hopf algebra structure: product, coproduct, counit, antipode
//!
//! The differential satisfies:
//! - d_1² = 0 (nilpotence)
//! - d_1(xy) = d_1(x)y + (-1)^{|x|} x d_1(y) (graded Leibniz/derivation rule)
//!
//! The E_2 page is obtained as cohomology: E_2^{p,q} = H^{p,q}(E_1, d_1)
//!
//! # References
//!
//! - McCleary, J. "A User's Guide to Spectral Sequences" (2001)
//! - Weibel, C. "An Introduction to Homological Algebra" (1994)
//! - Ravenel, D. "Complex Cobordism and Stable Homotopy Groups of Spheres" (1986)
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::e_one_star::*;
//!
//! // Create an E_1 algebra over Z
//! let e1 = E1Algebra::<i32>::new(vec!["x", "y"]);
//!
//! // Create a basis element at bidegree (1, 2)
//! let elem = E1Element::from_bidegree(BiDegree::new(1, 2), 1);
//!
//! // Compute the coproduct
//! let coproduct = e1.coproduct_element(&elem);
//! ```

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};

/// Bidegree (p, q) for spectral sequence grading
///
/// In the E_1 page of a spectral sequence:
/// - p is the filtration degree
/// - q is the total degree (or complementary degree)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BiDegree {
    /// Filtration degree p
    pub p: i64,
    /// Total degree q
    pub q: i64,
}

impl BiDegree {
    /// Create a new bidegree (p, q)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::e_one_star::BiDegree;
    ///
    /// let bideg = BiDegree::new(2, 3);
    /// assert_eq!(bideg.p, 2);
    /// assert_eq!(bideg.q, 3);
    /// ```
    pub fn new(p: i64, q: i64) -> Self {
        BiDegree { p, q }
    }

    /// Create the zero bidegree (0, 0)
    pub fn zero() -> Self {
        BiDegree { p: 0, q: 0 }
    }

    /// Total degree (p + q)
    pub fn total(&self) -> i64 {
        self.p + self.q
    }

    /// Check if this is the zero bidegree
    pub fn is_zero(&self) -> bool {
        self.p == 0 && self.q == 0
    }

    /// Add two bidegrees
    pub fn add(&self, other: &BiDegree) -> BiDegree {
        BiDegree {
            p: self.p + other.p,
            q: self.q + other.q,
        }
    }

    /// Bidegree of d_1 differential (convention: d_1 has bidegree (1, 0))
    pub fn d1_shift() -> BiDegree {
        BiDegree { p: 1, q: 0 }
    }
}

impl Display for BiDegree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.p, self.q)
    }
}

/// Generator for the E_1 algebra
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct E1Generator {
    /// Name of the generator
    pub name: String,
    /// Bidegree of the generator
    pub bidegree: BiDegree,
}

impl E1Generator {
    /// Create a new generator
    pub fn new(name: String, bidegree: BiDegree) -> Self {
        E1Generator { name, bidegree }
    }

    /// Check if this generator has odd total degree (for Koszul signs)
    pub fn is_odd(&self) -> bool {
        self.bidegree.total() % 2 == 1
    }
}

impl Display for E1Generator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Basis element in E_1 algebra
///
/// A basis element is represented as a monomial in the generators
pub type E1Monomial = Vec<String>;

/// Element of the E_1 algebra
///
/// Elements are represented as linear combinations of basis monomials
/// with coefficients from the ring R.
#[derive(Debug, Clone)]
pub struct E1Element<R: Ring> {
    /// Coefficients for each monomial basis element
    coefficients: HashMap<E1Monomial, R>,
    /// The bidegree of this element (None if not homogeneous)
    bidegree: Option<BiDegree>,
}

impl<R: Ring> E1Element<R> {
    /// Create the zero element
    pub fn zero() -> Self {
        E1Element {
            coefficients: HashMap::new(),
            bidegree: Some(BiDegree::zero()),
        }
    }

    /// Create an element from a single monomial
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::e_one_star::*;
    ///
    /// let elem = E1Element::from_monomial(vec!["x".to_string()], 5);
    /// ```
    pub fn from_monomial(monomial: E1Monomial, coeff: R) -> Self {
        let mut coefficients = HashMap::new();
        if !coeff.is_zero() {
            coefficients.insert(monomial, coeff);
        }
        E1Element {
            coefficients,
            bidegree: None, // Bidegree needs to be computed from algebra structure
        }
    }

    /// Create an element from a bidegree (typically a generator)
    pub fn from_bidegree(bidegree: BiDegree, coeff: R) -> Self {
        let mut coefficients = HashMap::new();
        if !coeff.is_zero() {
            // Use a canonical monomial representation for this bidegree
            coefficients.insert(vec![format!("e_{{{},{}}}", bidegree.p, bidegree.q)], coeff);
        }
        E1Element {
            coefficients,
            bidegree: Some(bidegree),
        }
    }

    /// Create the unit element (multiplicative identity)
    pub fn one() -> Self
    where
        R: From<i32>,
    {
        E1Element::from_monomial(vec![], R::from(1))
    }

    /// Check if the element is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Check if the element is homogeneous (single bidegree)
    pub fn is_homogeneous(&self) -> bool {
        self.bidegree.is_some()
    }

    /// Get the bidegree if homogeneous
    pub fn bidegree(&self) -> Option<&BiDegree> {
        self.bidegree.as_ref()
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &HashMap<E1Monomial, R> {
        &self.coefficients
    }

    /// Add a term to this element
    pub fn add_term(&mut self, monomial: E1Monomial, coeff: R)
    where
        R: Clone,
    {
        if coeff.is_zero() {
            return;
        }

        let entry = self.coefficients.entry(monomial).or_insert_with(|| R::zero());
        *entry = entry.clone() + coeff;

        // Mark as non-homogeneous if we're adding different terms
        if self.coefficients.len() > 1 {
            self.bidegree = None;
        }
    }

    /// Scale the element by a scalar
    pub fn scale(&self, scalar: &R) -> Self
    where
        R: Clone,
    {
        let mut new_coeffs = HashMap::new();
        for (monomial, coeff) in &self.coefficients {
            let new_coeff = coeff.clone() * scalar.clone();
            if !new_coeff.is_zero() {
                new_coeffs.insert(monomial.clone(), new_coeff);
            }
        }
        E1Element {
            coefficients: new_coeffs,
            bidegree: self.bidegree.clone(),
        }
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for E1Element<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.coefficients.len() != other.coefficients.len() {
            return false;
        }
        for (monomial, coeff) in &self.coefficients {
            match other.coefficients.get(monomial) {
                Some(other_coeff) if coeff == other_coeff => continue,
                _ => return false,
            }
        }
        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for E1Element<R> {}

impl<R: Ring + Display> Display for E1Element<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms: Vec<_> = self.coefficients.iter().collect();
        terms.sort_by_key(|(k, _)| k.clone());

        for (i, (monomial, coeff)) in terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            if monomial.is_empty() {
                write!(f, "{}", coeff)?;
            } else {
                write!(f, "{}*{}", coeff, monomial.join(""))?;
            }
        }
        Ok(())
    }
}

/// The E_1^* Hopf Algebra
///
/// The E_1 page of a spectral sequence, viewed as a bigraded Hopf algebra
/// with a differential operator d_1.
#[derive(Debug, Clone)]
pub struct E1Algebra<R: Ring> {
    /// Generators of the algebra
    generators: Vec<E1Generator>,
    /// Differential d_1 on generators
    d1_on_generators: HashMap<String, E1Element<R>>,
    /// Phantom data for ring
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring + Clone> E1Algebra<R> {
    /// Create a new E_1 algebra with given generator names
    ///
    /// Generators are given default bidegrees (i, 0) for the i-th generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::e_one_star::E1Algebra;
    ///
    /// let e1 = E1Algebra::<i32>::new(vec!["x", "y", "z"]);
    /// assert_eq!(e1.num_generators(), 3);
    /// ```
    pub fn new(generator_names: Vec<&str>) -> Self {
        let generators = generator_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                E1Generator::new(
                    name.to_string(),
                    BiDegree::new(i as i64, 0),
                )
            })
            .collect();

        E1Algebra {
            generators,
            d1_on_generators: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create an E_1 algebra with explicit generators
    pub fn with_generators(generators: Vec<E1Generator>) -> Self {
        E1Algebra {
            generators,
            d1_on_generators: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the differential d_1 on a generator
    ///
    /// This defines how d_1 acts on generators, which extends to all elements
    /// by the derivation property.
    pub fn set_d1_on_generator(&mut self, generator_name: &str, image: E1Element<R>) {
        self.d1_on_generators.insert(generator_name.to_string(), image);
    }

    /// Get the generators
    pub fn generators(&self) -> &[E1Generator] {
        &self.generators
    }

    /// Get number of generators
    pub fn num_generators(&self) -> usize {
        self.generators.len()
    }

    /// Get a generator by name
    pub fn generator(&self, name: &str) -> Option<&E1Generator> {
        self.generators.iter().find(|g| g.name == name)
    }

    /// Get the unit element
    pub fn one(&self) -> E1Element<R>
    where
        R: From<i32>,
    {
        E1Element::one()
    }

    /// Get the zero element
    pub fn zero(&self) -> E1Element<R> {
        E1Element::zero()
    }

    /// Compute the product of two monomials
    ///
    /// In a commutative algebra, this is just concatenation and sorting.
    /// For a graded commutative algebra, we would apply Koszul signs.
    pub fn product_monomials(&self, m1: &E1Monomial, m2: &E1Monomial) -> E1Monomial {
        let mut result = m1.clone();
        result.extend(m2.clone());
        result.sort(); // Commutative multiplication
        result
    }

    /// Compute the product of two elements
    pub fn product(&self, a: &E1Element<R>, b: &E1Element<R>) -> E1Element<R>
    where
        R: Clone,
    {
        let mut result = E1Element::zero();

        for (m1, c1) in &a.coefficients {
            for (m2, c2) in &b.coefficients {
                let prod_monomial = self.product_monomials(m1, m2);
                let prod_coeff = c1.clone() * c2.clone();
                result.add_term(prod_monomial, prod_coeff);
            }
        }

        // Update bidegree if both inputs are homogeneous
        if let (Some(deg_a), Some(deg_b)) = (a.bidegree(), b.bidegree()) {
            result.bidegree = Some(deg_a.add(deg_b));
        }

        result
    }

    /// Compute the coproduct of a monomial
    ///
    /// The coproduct (deconcatenation) is defined by:
    /// Δ(m) = Σ_{i} m[0..i] ⊗ m[i..n]
    ///
    /// This is the same as in shuffle algebras.
    pub fn coproduct_monomial(&self, monomial: &E1Monomial) -> Vec<(E1Monomial, E1Monomial)> {
        let n = monomial.len();
        let mut result = Vec::new();

        for i in 0..=n {
            let left = monomial[0..i].to_vec();
            let right = monomial[i..n].to_vec();
            result.push((left, right));
        }

        result
    }

    /// Compute the coproduct of an element
    ///
    /// Returns pairs of elements representing the tensor product.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::e_one_star::*;
    ///
    /// let e1 = E1Algebra::<i32>::new(vec!["x"]);
    /// let elem = E1Element::from_monomial(vec!["x".to_string()], 1);
    /// let coproduct = e1.coproduct_element(&elem);
    /// assert_eq!(coproduct.len(), 2); // 1⊗x + x⊗1
    /// ```
    pub fn coproduct_element(&self, element: &E1Element<R>) -> Vec<(E1Element<R>, E1Element<R>)>
    where
        R: Clone + From<i32>,
    {
        let mut result = Vec::new();

        for (monomial, coeff) in &element.coefficients {
            let coproduct_terms = self.coproduct_monomial(monomial);

            for (left, right) in coproduct_terms {
                let left_elem = E1Element::from_monomial(left, coeff.clone());
                let right_elem = E1Element::from_monomial(right, R::from(1));
                result.push((left_elem, right_elem));
            }
        }

        result
    }

    /// Compute the counit (augmentation) of an element
    ///
    /// The counit ε: E_1 → R is defined by ε(1) = 1 and ε(g) = 0 for generators g.
    /// This projects onto the coefficient of the empty monomial.
    pub fn counit(&self, element: &E1Element<R>) -> R
    where
        R: Clone,
    {
        element
            .coefficients
            .get(&vec![])
            .cloned()
            .unwrap_or_else(R::zero)
    }

    /// Compute the antipode of a monomial
    ///
    /// The antipode S(m) = (-1)^|m| * reverse(m)
    ///
    /// This is the same formula as in shuffle algebras.
    pub fn antipode_monomial(&self, monomial: &E1Monomial) -> (i32, E1Monomial) {
        let sign = if monomial.len() % 2 == 0 { 1 } else { -1 };
        let mut reversed = monomial.clone();
        reversed.reverse();
        (sign, reversed)
    }

    /// Compute the antipode of an element
    pub fn antipode(&self, element: &E1Element<R>) -> E1Element<R>
    where
        R: Clone + From<i32>,
    {
        let mut result = E1Element::zero();

        for (monomial, coeff) in &element.coefficients {
            let (sign, reversed) = self.antipode_monomial(monomial);
            let sign_coeff = R::from(sign);
            let new_coeff = coeff.clone() * sign_coeff;
            result.add_term(reversed, new_coeff);
        }

        result
    }

    /// Apply the differential d_1 to an element
    ///
    /// The differential d_1: E_1^{p,q} → E_1^{p+1,q} satisfies:
    /// - d_1² = 0
    /// - d_1(xy) = d_1(x)y + (-1)^{|x|} x d_1(y) (graded derivation)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::e_one_star::*;
    ///
    /// let mut e1 = E1Algebra::<i32>::new(vec!["x"]);
    /// let elem = E1Element::from_monomial(vec!["x".to_string()], 1);
    ///
    /// // Set d_1(x) = 0 for a cocycle
    /// e1.set_d1_on_generator("x", E1Element::zero());
    /// let d_elem = e1.apply_d1(&elem);
    /// assert!(d_elem.is_zero());
    /// ```
    pub fn apply_d1(&self, element: &E1Element<R>) -> E1Element<R>
    where
        R: Clone + From<i32>,
    {
        let mut result = E1Element::zero();

        for (monomial, coeff) in &element.coefficients {
            // Apply derivation rule: d(m1*m2*...) = d(m1)*m2*... + (-1)^|m1| m1*d(m2)*... + ...
            let d_monomial = self.apply_d1_to_monomial(monomial);
            for term in d_monomial {
                result.add_term(term.0, coeff.clone() * term.1);
            }
        }

        result
    }

    /// Apply d_1 to a monomial using the derivation rule
    fn apply_d1_to_monomial(&self, monomial: &E1Monomial) -> Vec<(E1Monomial, R)>
    where
        R: Clone + From<i32>,
    {
        let mut result = Vec::new();

        if monomial.is_empty() {
            return result; // d(1) = 0
        }

        // Apply derivation: d(x₁x₂...xₙ) = Σᵢ (-1)^{|x₁...xᵢ₋₁|} x₁...xᵢ₋₁ d(xᵢ) xᵢ₊₁...xₙ
        for i in 0..monomial.len() {
            if let Some(d_gen) = self.d1_on_generators.get(&monomial[i]) {
                // Compute sign from Koszul rule
                let sign = if i % 2 == 0 { 1 } else { -1 }; // Simplified sign calculation

                // Build the result monomial: prefix + d(gen) + suffix
                for (d_mon, d_coeff) in &d_gen.coefficients {
                    let mut full_monomial = monomial[0..i].to_vec();
                    full_monomial.extend(d_mon.clone());
                    full_monomial.extend(monomial[i+1..].to_vec());

                    let coeff = d_coeff.clone() * R::from(sign);
                    result.push((full_monomial, coeff));
                }
            }
        }

        result
    }

    /// Check if an element is a cocycle (d_1(x) = 0)
    pub fn is_cocycle(&self, element: &E1Element<R>) -> bool
    where
        R: Clone + From<i32>,
    {
        self.apply_d1(element).is_zero()
    }

    /// Verify that d_1² = 0 for all generators
    pub fn verify_d1_nilpotent(&self) -> bool
    where
        R: Clone + From<i32> + PartialEq,
    {
        for gen in &self.generators {
            if let Some(d_gen) = self.d1_on_generators.get(&gen.name) {
                let d2_gen = self.apply_d1(d_gen);
                if !d2_gen.is_zero() {
                    return false;
                }
            }
        }
        true
    }
}

impl<R: Ring> Display for E1Algebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "E₁ Hopf algebra with generators: ")?;
        for (i, gen) in self.generators.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} at {}", gen.name, gen.bidegree)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bidegree_creation() {
        let bideg = BiDegree::new(2, 3);
        assert_eq!(bideg.p, 2);
        assert_eq!(bideg.q, 3);
        assert_eq!(bideg.total(), 5);
    }

    #[test]
    fn test_bidegree_addition() {
        let b1 = BiDegree::new(1, 2);
        let b2 = BiDegree::new(3, 4);
        let sum = b1.add(&b2);
        assert_eq!(sum.p, 4);
        assert_eq!(sum.q, 6);
    }

    #[test]
    fn test_bidegree_zero() {
        let zero = BiDegree::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.total(), 0);
    }

    #[test]
    fn test_e1_element_zero() {
        let elem: E1Element<i32> = E1Element::zero();
        assert!(elem.is_zero());
    }

    #[test]
    fn test_e1_element_one() {
        let elem: E1Element<i32> = E1Element::one();
        assert!(!elem.is_zero());
        assert_eq!(elem.coefficients().get(&vec![]), Some(&1));
    }

    #[test]
    fn test_e1_element_from_monomial() {
        let monomial = vec!["x".to_string(), "y".to_string()];
        let elem = E1Element::from_monomial(monomial.clone(), 5);
        assert!(!elem.is_zero());
        assert_eq!(elem.coefficients().get(&monomial), Some(&5));
    }

    #[test]
    fn test_e1_element_from_bidegree() {
        let bideg = BiDegree::new(2, 3);
        let elem = E1Element::<i32>::from_bidegree(bideg.clone(), 7);
        assert_eq!(elem.bidegree(), Some(&bideg));
        assert!(!elem.is_zero());
    }

    #[test]
    fn test_e1_algebra_creation() {
        let e1 = E1Algebra::<i32>::new(vec!["x", "y", "z"]);
        assert_eq!(e1.num_generators(), 3);
        assert!(e1.generator("x").is_some());
        assert!(e1.generator("w").is_none());
    }

    #[test]
    fn test_e1_algebra_product() {
        let e1 = E1Algebra::<i32>::new(vec!["x", "y"]);

        let elem1 = E1Element::from_monomial(vec!["x".to_string()], 2);
        let elem2 = E1Element::from_monomial(vec!["y".to_string()], 3);

        let product = e1.product(&elem1, &elem2);
        assert!(!product.is_zero());
    }

    #[test]
    fn test_e1_algebra_unit() {
        let e1 = E1Algebra::<i32>::new(vec!["x"]);
        let unit = e1.one();

        let elem = E1Element::from_monomial(vec!["x".to_string()], 5);
        let product = e1.product(&unit, &elem);

        // 1 * x = x (up to representation)
        assert!(!product.is_zero());
    }

    #[test]
    fn test_coproduct_monomial() {
        let e1 = E1Algebra::<i32>::new(vec!["x", "y"]);
        let monomial = vec!["x".to_string(), "y".to_string()];

        let coproduct = e1.coproduct_monomial(&monomial);

        // Should have 3 terms: 1⊗xy, x⊗y, xy⊗1
        assert_eq!(coproduct.len(), 3);
        assert!(coproduct.contains(&(vec![], vec!["x".to_string(), "y".to_string()])));
        assert!(coproduct.contains(&(vec!["x".to_string()], vec!["y".to_string()])));
        assert!(coproduct.contains(&(vec!["x".to_string(), "y".to_string()], vec![])));
    }

    #[test]
    fn test_counit() {
        let e1 = E1Algebra::<i32>::new(vec!["x"]);

        // ε(1) = 1
        let unit = E1Element::one();
        assert_eq!(e1.counit(&unit), 1);

        // ε(x) = 0
        let elem = E1Element::from_monomial(vec!["x".to_string()], 5);
        assert_eq!(e1.counit(&elem), 0);
    }

    #[test]
    fn test_antipode_monomial() {
        let e1 = E1Algebra::<i32>::new(vec!["x", "y"]);

        // Even length: sign = 1
        let monomial = vec!["x".to_string(), "y".to_string()];
        let (sign, reversed) = e1.antipode_monomial(&monomial);
        assert_eq!(sign, 1);
        assert_eq!(reversed, vec!["y".to_string(), "x".to_string()]);

        // Odd length: sign = -1
        let monomial2 = vec!["x".to_string()];
        let (sign2, reversed2) = e1.antipode_monomial(&monomial2);
        assert_eq!(sign2, -1);
        assert_eq!(reversed2, vec!["x".to_string()]);
    }

    #[test]
    fn test_differential_on_zero() {
        let e1 = E1Algebra::<i32>::new(vec!["x"]);
        let zero = E1Element::zero();
        let d_zero = e1.apply_d1(&zero);
        assert!(d_zero.is_zero());
    }

    #[test]
    fn test_differential_with_zero_image() {
        let mut e1 = E1Algebra::<i32>::new(vec!["x"]);

        // Set d_1(x) = 0
        e1.set_d1_on_generator("x", E1Element::zero());

        let elem = E1Element::from_monomial(vec!["x".to_string()], 1);
        let d_elem = e1.apply_d1(&elem);

        // d(x) = 0, so x is a cocycle
        assert!(d_elem.is_zero());
        assert!(e1.is_cocycle(&elem));
    }

    #[test]
    fn test_verify_d1_nilpotent() {
        let mut e1 = E1Algebra::<i32>::new(vec!["x", "y"]);

        // Set d_1(x) = 0 and d_1(y) = 0
        e1.set_d1_on_generator("x", E1Element::zero());
        e1.set_d1_on_generator("y", E1Element::zero());

        // Should satisfy d² = 0
        assert!(e1.verify_d1_nilpotent());
    }

    #[test]
    fn test_e1_element_equality() {
        let elem1 = E1Element::from_monomial(vec!["x".to_string()], 5);
        let elem2 = E1Element::from_monomial(vec!["x".to_string()], 5);
        let elem3 = E1Element::from_monomial(vec!["x".to_string()], 3);

        assert_eq!(elem1, elem2);
        assert_ne!(elem1, elem3);
    }

    #[test]
    fn test_e1_element_scale() {
        let elem = E1Element::from_monomial(vec!["x".to_string()], 3);
        let scaled = elem.scale(&2);

        assert_eq!(scaled.coefficients().get(&vec!["x".to_string()]), Some(&6));
    }

    #[test]
    fn test_generator_odd_check() {
        let gen_even = E1Generator::new("x".to_string(), BiDegree::new(2, 2));
        let gen_odd = E1Generator::new("y".to_string(), BiDegree::new(1, 2));

        assert!(!gen_even.is_odd());
        assert!(gen_odd.is_odd());
    }

    #[test]
    fn test_bidegree_display() {
        let bideg = BiDegree::new(2, 3);
        let display = format!("{}", bideg);
        assert_eq!(display, "(2, 3)");
    }

    #[test]
    fn test_algebra_display() {
        let e1 = E1Algebra::<i32>::new(vec!["x", "y"]);
        let display = format!("{}", e1);
        assert!(display.contains("E₁"));
        assert!(display.contains("x"));
        assert!(display.contains("y"));
    }

    #[test]
    fn test_element_display() {
        let elem = E1Element::from_monomial(vec!["x".to_string()], 3);
        let display = format!("{}", elem);
        assert!(display.contains("3"));
        assert!(display.contains("x"));

        let zero: E1Element<i32> = E1Element::zero();
        assert_eq!(format!("{}", zero), "0");
    }

    #[test]
    fn test_d1_bidegree_shift() {
        let shift = BiDegree::d1_shift();
        assert_eq!(shift.p, 1);
        assert_eq!(shift.q, 0);
    }

    #[test]
    fn test_homogeneous_product() {
        let e1 = E1Algebra::<i32>::new(vec!["x", "y"]);

        let elem1 = E1Element::<i32>::from_bidegree(BiDegree::new(1, 0), 1);
        let elem2 = E1Element::<i32>::from_bidegree(BiDegree::new(0, 1), 1);

        let product = e1.product(&elem1, &elem2);

        // Product of homogeneous elements should be homogeneous
        assert!(product.is_homogeneous());
        assert_eq!(product.bidegree(), Some(&BiDegree::new(1, 1)));
    }
}
