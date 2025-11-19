//! Field Category - ElementMethods and ParentMethods
//!
//! This module implements the SageMath-style category methods for fields:
//! - FieldElementMethods: Methods that apply to field elements
//! - FieldParentMethods: Methods that apply to field structures
//!
//! Corresponds to sage.categories.fields.ElementMethods and sage.categories.fields.ParentMethods

use rustmath_core::{Field, Ring, Parent, MathError, Result};

/// Methods for field elements (corresponds to sage.categories.fields.ElementMethods)
///
/// This trait provides additional methods that are available on all field elements,
/// extending the basic Field trait with category-theoretic operations.
///
/// # Examples
///
/// ```
/// use rustmath_core::Field;
/// use rustmath_category::FieldElementMethods;
///
/// // Any field element can use these methods
/// // let x = Rational::new(3, 4);
/// // assert!(x.is_unit());
/// // assert_eq!(x.euclidean_degree(), 0);
/// ```
pub trait FieldElementMethods: Field {
    /// Return the Euclidean degree of this element.
    ///
    /// For field elements, this is 0 for all nonzero elements.
    /// Zero element raises an error.
    ///
    /// # Returns
    ///
    /// - `Ok(0)` for nonzero elements
    /// - `Err(MathError::DivisionByZero)` for zero element
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldElementMethods;
    /// let x = Rational::new(3, 4);
    /// assert_eq!(x.euclidean_degree(), Ok(0));
    ///
    /// let zero = Rational::zero();
    /// assert!(zero.euclidean_degree().is_err());
    /// ```
    fn euclidean_degree(&self) -> Result<u64> {
        if self.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(0)
        }
    }

    /// Return quotient and remainder of division.
    ///
    /// For fields, the remainder is always zero and the quotient is self/other.
    ///
    /// # Arguments
    ///
    /// * `other` - The divisor
    ///
    /// # Returns
    ///
    /// - `Ok((quotient, zero))` where quotient = self/other
    /// - `Err(MathError::DivisionByZero)` if other is zero
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldElementMethods;
    /// let x = Rational::new(3, 4);
    /// let y = Rational::new(5, 6);
    /// let (q, r) = x.quo_rem(&y).unwrap();
    /// assert_eq!(q, Rational::new(9, 10));
    /// assert!(r.is_zero());
    /// ```
    fn quo_rem(&self, other: &Self) -> Result<(Self, Self)> {
        Ok((self.divide(other)?, Self::zero()))
    }

    /// Check if this element is a unit (invertible).
    ///
    /// For fields, all nonzero elements are units.
    ///
    /// # Returns
    ///
    /// `true` if element is nonzero, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldElementMethods;
    /// let x = Rational::new(3, 4);
    /// assert!(x.is_unit());
    ///
    /// let zero = Rational::zero();
    /// assert!(!zero.is_unit());
    /// ```
    fn is_unit(&self) -> bool {
        !self.is_zero()
    }

    /// Return the greatest common divisor with another element.
    ///
    /// For fields:
    /// - gcd(0, 0) = 0
    /// - gcd(x, y) = 1 for any nonzero x or y
    ///
    /// # Arguments
    ///
    /// * `other` - The other element
    ///
    /// # Returns
    ///
    /// - `Self::zero()` if both elements are zero
    /// - `Self::one()` if at least one element is nonzero
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldElementMethods;
    /// let x = Rational::new(3, 4);
    /// let y = Rational::new(5, 6);
    /// assert_eq!(x.gcd_field(&y), Rational::one());
    ///
    /// let zero = Rational::zero();
    /// assert_eq!(zero.gcd_field(&zero), Rational::zero());
    /// ```
    fn gcd_field(&self, other: &Self) -> Self {
        if self.is_zero() && other.is_zero() {
            Self::zero()
        } else {
            Self::one()
        }
    }

    /// Return the least common multiple with another element.
    ///
    /// For fields:
    /// - lcm(0, y) = lcm(x, 0) = 0
    /// - lcm(x, y) = 1 for nonzero x and y
    ///
    /// # Arguments
    ///
    /// * `other` - The other element
    ///
    /// # Returns
    ///
    /// - `Self::zero()` if either element is zero
    /// - `Self::one()` if both elements are nonzero
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldElementMethods;
    /// let x = Rational::new(3, 4);
    /// let y = Rational::new(5, 6);
    /// assert_eq!(x.lcm_field(&y), Rational::one());
    ///
    /// let zero = Rational::zero();
    /// assert_eq!(x.lcm_field(&zero), Rational::zero());
    /// ```
    fn lcm_field(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            Self::one()
        }
    }

    /// Return extended GCD: (gcd, s, t) where gcd = s*self + t*other.
    ///
    /// For fields:
    /// - If self is nonzero: returns (1, inverse, 0)
    /// - If both are zero: returns (0, 0, 0)
    /// - If self is zero but other is nonzero: returns (1, 0, inverse of other)
    ///
    /// # Arguments
    ///
    /// * `other` - The other element
    ///
    /// # Returns
    ///
    /// Tuple (gcd, s, t) where gcd = s*self + t*other
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldElementMethods;
    /// let x = Rational::new(3, 4);
    /// let y = Rational::new(5, 6);
    /// let (g, s, t) = x.xgcd_field(&y).unwrap();
    /// assert_eq!(g, Rational::one());
    /// assert_eq!(s, x.inverse().unwrap());
    /// assert!(t.is_zero());
    /// ```
    fn xgcd_field(&self, other: &Self) -> Result<(Self, Self, Self)> {
        if self.is_zero() && other.is_zero() {
            Ok((Self::zero(), Self::zero(), Self::zero()))
        } else if !self.is_zero() {
            // gcd = 1 = (1/self) * self + 0 * other
            Ok((Self::one(), self.inverse()?, Self::zero()))
        } else {
            // self is zero, other is nonzero
            // gcd = 1 = 0 * self + (1/other) * other
            Ok((Self::one(), Self::zero(), other.inverse()?))
        }
    }

    /// Return the inverse of this unit.
    ///
    /// This is equivalent to calling `inverse()`, provided for API compatibility.
    ///
    /// # Returns
    ///
    /// - `Ok(inverse)` if element is nonzero
    /// - `Err(MathError::DivisionByZero)` if element is zero
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldElementMethods;
    /// let x = Rational::new(3, 4);
    /// assert_eq!(x.inverse_of_unit().unwrap(), Rational::new(4, 3));
    /// ```
    fn inverse_of_unit(&self) -> Result<Self> {
        self.inverse()
    }

    /// Return the factorization of this element.
    ///
    /// For fields, this returns a trivial factorization: the element itself as a unit
    /// with no prime factors.
    ///
    /// # Returns
    ///
    /// - `Ok((unit, factors))` where unit is self and factors is empty
    /// - `Err(MathError::DivisionByZero)` for zero element
    ///
    /// Note: Returns a tuple (unit, empty vector) to represent the factorization.
    /// The actual Factorization type would be defined elsewhere if needed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldElementMethods;
    /// let x = Rational::new(3, 4);
    /// let (unit, factors) = x.factor_field().unwrap();
    /// assert_eq!(unit, x);
    /// assert!(factors.is_empty());
    /// ```
    fn factor_field(&self) -> Result<(Self, Vec<(Self, u32)>)> {
        if self.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            // Trivial factorization: just the unit itself with no factors
            Ok((self.clone(), Vec::new()))
        }
    }
}

/// Blanket implementation: all Fields get FieldElementMethods
impl<F: Field> FieldElementMethods for F {}

/// Methods for field parents (corresponds to sage.categories.fields.ParentMethods)
///
/// This trait provides methods that apply to field structures (parents) rather than
/// individual elements.
///
/// # Examples
///
/// ```ignore
/// use rustmath_core::Parent;
/// use rustmath_category::FieldParentMethods;
///
/// // A field structure would implement this
/// // let field = FiniteField::new(7);
/// // assert_eq!(field.krull_dimension(), 0);
/// // assert!(field.is_field(true));
/// ```
pub trait FieldParentMethods: Parent {
    /// Return the Krull dimension of this field.
    ///
    /// For all fields, the Krull dimension is 0.
    ///
    /// # Returns
    ///
    /// Always returns 0
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_finitefields::PrimeField;
    /// # use rustmath_category::FieldParentMethods;
    /// let field = PrimeField::new(7);
    /// assert_eq!(field.krull_dimension(), 0);
    /// ```
    fn krull_dimension(&self) -> u32 {
        0
    }

    /// Check if this parent is a field.
    ///
    /// For types implementing FieldParentMethods, this always returns true.
    ///
    /// # Arguments
    ///
    /// * `_proof` - If true, require a mathematical proof (not used in Rust implementation)
    ///
    /// # Returns
    ///
    /// Always returns `true`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_finitefields::PrimeField;
    /// # use rustmath_category::FieldParentMethods;
    /// let field = PrimeField::new(7);
    /// assert!(field.is_field(true));
    /// ```
    fn is_field(&self, _proof: bool) -> bool {
        true
    }

    /// Check if this field is integrally closed.
    ///
    /// All fields are integrally closed in their field of fractions (which is themselves).
    ///
    /// # Returns
    ///
    /// Always returns `true`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_finitefields::PrimeField;
    /// # use rustmath_category::FieldParentMethods;
    /// let field = PrimeField::new(7);
    /// assert!(field.is_integrally_closed());
    /// ```
    fn is_integrally_closed(&self) -> bool {
        true
    }

    /// Return the integral closure of this field.
    ///
    /// For fields, the integral closure is the field itself.
    ///
    /// # Returns
    ///
    /// A clone of self
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_finitefields::PrimeField;
    /// # use rustmath_category::FieldParentMethods;
    /// let field = PrimeField::new(7);
    /// let closure = field.integral_closure();
    /// // closure == field
    /// ```
    fn integral_closure(&self) -> Self
    where
        Self: Sized + Clone,
    {
        self.clone()
    }

    /// Return the fraction field of this field.
    ///
    /// For fields, the fraction field is the field itself.
    ///
    /// # Returns
    ///
    /// A clone of self
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_finitefields::PrimeField;
    /// # use rustmath_category::FieldParentMethods;
    /// let field = PrimeField::new(7);
    /// let frac_field = field.fraction_field();
    /// // frac_field == field
    /// ```
    fn fraction_field(&self) -> Self
    where
        Self: Sized + Clone,
    {
        self.clone()
    }

    /// Return the pseudo-fraction field of this field.
    ///
    /// This is provided for compatibility with rings. For fields, it returns self.
    ///
    /// # Returns
    ///
    /// A clone of self
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_finitefields::PrimeField;
    /// # use rustmath_category::FieldParentMethods;
    /// let field = PrimeField::new(7);
    /// let pseudo_frac = field.pseudo_fraction_field();
    /// // pseudo_frac == field
    /// ```
    fn pseudo_fraction_field(&self) -> Self
    where
        Self: Sized + Clone,
    {
        self.clone()
    }

    /// Check if x divides y in this field.
    ///
    /// In a field, every nonzero element divides every element.
    ///
    /// # Arguments
    ///
    /// * `x` - The potential divisor
    /// * `y` - The potential dividend
    ///
    /// # Returns
    ///
    /// - `true` if x is nonzero (x divides y)
    /// - `false` if x is zero and y is nonzero
    /// - `true` if both are zero (by convention, 0 divides 0)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_finitefields::PrimeField;
    /// # use rustmath_category::FieldParentMethods;
    /// let field = PrimeField::new(7);
    /// let x = field.element(3);
    /// let y = field.element(5);
    /// assert!(field.divides(&x, &y));
    /// ```
    fn divides(&self, x: &Self::Element, y: &Self::Element) -> bool
    where
        Self::Element: Ring,
    {
        !x.is_zero() || y.is_zero()
    }

    /// Check if this field is perfect.
    ///
    /// A field of characteristic 0 is always perfect.
    /// For positive characteristic, this would need to check if F^p = F.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` for characteristic 0 fields
    /// - `Err(MathError::NotImplemented)` for positive characteristic
    ///
    /// Note: Full implementation would require characteristic information
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_rationals::Rational;
    /// # use rustmath_category::FieldParentMethods;
    /// // Rationals have characteristic 0, so they are perfect
    /// // let qq = RationalField::new();
    /// // assert!(qq.is_perfect().unwrap());
    /// ```
    fn is_perfect(&self) -> Result<bool> {
        // For characteristic 0, always perfect
        // For char p > 0, need to check if F^p = F
        // This is a simplified implementation
        Err(MathError::NotImplemented(
            "is_perfect requires characteristic information".to_string(),
        ))
    }

    /// Create an ideal from generators.
    ///
    /// In a field, every nonzero element generates the unit ideal (the whole field),
    /// and zero generates the zero ideal.
    ///
    /// # Arguments
    ///
    /// * `generators` - The generators of the ideal
    ///
    /// # Returns
    ///
    /// - `UnitIdeal` if any generator is nonzero
    /// - `ZeroIdeal` if all generators are zero
    ///
    /// Note: Returns a simple boolean indicator (true = unit ideal, false = zero ideal)
    /// since we don't have a full Ideal type defined yet.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use rustmath_finitefields::PrimeField;
    /// # use rustmath_category::FieldParentMethods;
    /// let field = PrimeField::new(7);
    /// let x = field.element(3);
    /// assert!(field.ideal(&[x])); // Unit ideal
    ///
    /// let zero = field.zero();
    /// assert!(!field.ideal(&[zero])); // Zero ideal
    /// ```
    fn ideal(&self, generators: &[Self::Element]) -> bool
    where
        Self::Element: Ring,
    {
        // Return true for unit ideal (any nonzero generator), false for zero ideal
        generators.iter().any(|g| !g.is_zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::{Field, Ring, CommutativeRing, IntegralDomain};

    // Simple test field: Rationals modulo (using f64 for testing)
    #[derive(Debug, Clone, PartialEq)]
    struct TestField(f64);

    impl std::fmt::Display for TestField {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::ops::Add for TestField {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            TestField(self.0 + other.0)
        }
    }

    impl std::ops::Sub for TestField {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            TestField(self.0 - other.0)
        }
    }

    impl std::ops::Mul for TestField {
        type Output = Self;
        fn mul(self, other: Self) -> Self {
            TestField(self.0 * other.0)
        }
    }

    impl std::ops::Neg for TestField {
        type Output = Self;
        fn neg(self) -> Self {
            TestField(-self.0)
        }
    }

    impl std::ops::Div for TestField {
        type Output = Self;
        fn div(self, other: Self) -> Self {
            TestField(self.0 / other.0)
        }
    }

    impl Ring for TestField {
        fn zero() -> Self {
            TestField(0.0)
        }

        fn one() -> Self {
            TestField(1.0)
        }

        fn is_zero(&self) -> bool {
            self.0.abs() < 1e-10
        }

        fn is_one(&self) -> bool {
            (self.0 - 1.0).abs() < 1e-10
        }
    }

    impl CommutativeRing for TestField {}
    impl IntegralDomain for TestField {}

    impl Field for TestField {
        fn inverse(&self) -> Result<Self> {
            if self.is_zero() {
                Err(MathError::DivisionByZero)
            } else {
                Ok(TestField(1.0 / self.0))
            }
        }
    }

    #[test]
    fn test_euclidean_degree() {
        let x = TestField(3.0);
        assert_eq!(x.euclidean_degree().unwrap(), 0);

        let zero = TestField::zero();
        assert!(zero.euclidean_degree().is_err());
    }

    #[test]
    fn test_quo_rem() {
        let x = TestField(6.0);
        let y = TestField(2.0);
        let (q, r) = x.quo_rem(&y).unwrap();
        assert!((q.0 - 3.0).abs() < 1e-10);
        assert!(r.is_zero());

        let zero = TestField::zero();
        assert!(x.quo_rem(&zero).is_err());
    }

    #[test]
    fn test_is_unit() {
        let x = TestField(3.0);
        assert!(x.is_unit());

        let zero = TestField::zero();
        assert!(!zero.is_unit());
    }

    #[test]
    fn test_gcd_field() {
        let x = TestField(3.0);
        let y = TestField(5.0);
        assert_eq!(x.gcd_field(&y), TestField::one());

        let zero = TestField::zero();
        assert_eq!(zero.gcd_field(&zero), TestField::zero());
        assert_eq!(x.gcd_field(&zero), TestField::one());
    }

    #[test]
    fn test_lcm_field() {
        let x = TestField(3.0);
        let y = TestField(5.0);
        assert_eq!(x.lcm_field(&y), TestField::one());

        let zero = TestField::zero();
        assert_eq!(x.lcm_field(&zero), TestField::zero());
        assert_eq!(zero.lcm_field(&y), TestField::zero());
    }

    #[test]
    fn test_xgcd_field() {
        let x = TestField(3.0);
        let y = TestField(5.0);
        let (g, s, t) = x.xgcd_field(&y).unwrap();

        assert!(g.is_one());
        assert!((s.0 - 1.0 / 3.0).abs() < 1e-10);
        assert!(t.is_zero());

        let zero = TestField::zero();
        let (g, s, t) = zero.xgcd_field(&zero).unwrap();
        assert!(g.is_zero());
        assert!(s.is_zero());
        assert!(t.is_zero());

        let (g, s, t) = zero.xgcd_field(&y).unwrap();
        assert!(g.is_one());
        assert!(s.is_zero());
        assert!((t.0 - 1.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_of_unit() {
        let x = TestField(4.0);
        let inv = x.inverse_of_unit().unwrap();
        assert!((inv.0 - 0.25).abs() < 1e-10);

        let zero = TestField::zero();
        assert!(zero.inverse_of_unit().is_err());
    }

    #[test]
    fn test_factor_field() {
        let x = TestField(3.0);
        let (unit, factors) = x.factor_field().unwrap();
        assert_eq!(unit, x);
        assert!(factors.is_empty());

        let zero = TestField::zero();
        assert!(zero.factor_field().is_err());
    }

    // Parent tests
    #[derive(Debug, Clone)]
    struct TestFieldParent;

    impl Parent for TestFieldParent {
        type Element = TestField;

        fn contains(&self, _element: &Self::Element) -> bool {
            true
        }

        fn zero(&self) -> Option<Self::Element> {
            Some(TestField::zero())
        }

        fn one(&self) -> Option<Self::Element> {
            Some(TestField::one())
        }
    }

    #[test]
    fn test_krull_dimension() {
        let field = TestFieldParent;
        assert_eq!(field.krull_dimension(), 0);
    }

    #[test]
    fn test_is_field() {
        let field = TestFieldParent;
        assert!(field.is_field(true));
        assert!(field.is_field(false));
    }

    #[test]
    fn test_is_integrally_closed() {
        let field = TestFieldParent;
        assert!(field.is_integrally_closed());
    }

    #[test]
    fn test_integral_closure() {
        let field = TestFieldParent;
        let closure = field.integral_closure();
        // Should return self
        assert_eq!(format!("{:?}", field), format!("{:?}", closure));
    }

    #[test]
    fn test_fraction_field() {
        let field = TestFieldParent;
        let frac = field.fraction_field();
        // Should return self
        assert_eq!(format!("{:?}", field), format!("{:?}", frac));
    }

    #[test]
    fn test_pseudo_fraction_field() {
        let field = TestFieldParent;
        let pseudo = field.pseudo_fraction_field();
        // Should return self
        assert_eq!(format!("{:?}", field), format!("{:?}", pseudo));
    }

    #[test]
    fn test_divides() {
        let field = TestFieldParent;
        let x = TestField(3.0);
        let y = TestField(5.0);
        let zero = TestField::zero();

        assert!(field.divides(&x, &y));
        assert!(!field.divides(&zero, &y));
        assert!(field.divides(&zero, &zero));
        assert!(field.divides(&x, &zero));
    }

    #[test]
    fn test_ideal() {
        let field = TestFieldParent;
        let x = TestField(3.0);
        let y = TestField(5.0);
        let zero = TestField::zero();

        // Nonzero generator creates unit ideal (true)
        assert!(field.ideal(&[x.clone()]));
        assert!(field.ideal(&[x.clone(), y.clone()]));

        // Zero generators create zero ideal (false)
        assert!(!field.ideal(&[zero.clone()]));
        assert!(!field.ideal(&[zero.clone(), zero.clone()]));

        // Mix: any nonzero creates unit ideal
        assert!(field.ideal(&[zero.clone(), x.clone()]));
    }

    #[test]
    fn test_is_perfect() {
        let field = TestFieldParent;
        // Not implemented yet, should return error
        assert!(field.is_perfect().is_err());
    }
}
