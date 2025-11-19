//! Complex number implementation

use rustmath_core::{CommutativeRing, Field, MathError, NumericConversion, Result, Ring};
use rustmath_reals::{Real, RealField};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Complex number with real and imaginary parts
///
/// Represents a + bi where a, b are real numbers and i² = -1
///
/// # Precision
///
/// Complex numbers inherit their precision from their Real components.
/// The precision parameter is currently for f64 compatibility (53 bits)
/// but is designed for future arbitrary precision support.
#[derive(Clone, Debug)]
pub struct Complex {
    real: Real,
    imag: Real,
}

/// Complex number field with specified precision
///
/// Factory for creating Complex numbers with a specific precision.
/// This mirrors SageMath's ComplexField(prec) constructor.
///
/// # Example
///
/// ```
/// use rustmath_complex::ComplexField;
///
/// // Create a complex field with 100 bits of precision
/// let cf = ComplexField::new(100);
/// let z = cf.make_complex(3.0, 4.0);
/// ```
#[derive(Clone, Debug)]
pub struct ComplexField {
    real_field: RealField,
}

impl ComplexField {
    /// Create a new ComplexField with specified precision
    ///
    /// # Arguments
    ///
    /// * `precision` - Number of bits of precision for the mantissa
    ///                 (currently must be 53 for f64, but parameter is accepted
    ///                 for future compatibility)
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_complex::ComplexField;
    ///
    /// let cf = ComplexField::new(53);  // Standard f64 precision
    /// let high_prec = ComplexField::new(256);  // Future: 256-bit precision
    /// ```
    pub fn new(precision: u32) -> Self {
        ComplexField {
            real_field: RealField::new(precision),
        }
    }

    /// Create a default ComplexField (53-bit precision, equivalent to f64)
    pub fn default() -> Self {
        ComplexField {
            real_field: RealField::default(),
        }
    }

    /// Get the precision of this field
    pub fn precision(&self) -> u32 {
        self.real_field.precision()
    }

    /// Create a Complex number in this field from real and imaginary parts
    pub fn make_complex(&self, real: f64, imag: f64) -> Complex {
        Complex {
            real: self.real_field.make_real(real),
            imag: self.real_field.make_real(imag),
        }
    }

    /// Create a purely real complex number
    pub fn from_real(&self, real: f64) -> Complex {
        Complex {
            real: self.real_field.make_real(real),
            imag: self.real_field.zero(),
        }
    }

    /// Create a purely imaginary complex number
    pub fn from_imag(&self, imag: f64) -> Complex {
        Complex {
            real: self.real_field.zero(),
            imag: self.real_field.make_real(imag),
        }
    }

    /// Create zero
    pub fn zero(&self) -> Complex {
        Complex {
            real: self.real_field.zero(),
            imag: self.real_field.zero(),
        }
    }

    /// Create one
    pub fn one(&self) -> Complex {
        Complex {
            real: self.real_field.one(),
            imag: self.real_field.zero(),
        }
    }

    /// Create the imaginary unit i
    pub fn i(&self) -> Complex {
        Complex {
            real: self.real_field.zero(),
            imag: self.real_field.one(),
        }
    }
}

impl Complex {
    /// Create a new complex number from real and imaginary parts
    pub fn new(real: f64, imag: f64) -> Self {
        Complex {
            real: Real::from(real),
            imag: Real::from(imag),
        }
    }

    /// Create the zero complex number (0 + 0i)
    pub fn zero() -> Self {
        Complex {
            real: Real::from(0.0),
            imag: Real::from(0.0),
        }
    }

    /// Create the one complex number (1 + 0i)
    pub fn one() -> Self {
        Complex {
            real: Real::from(1.0),
            imag: Real::from(0.0),
        }
    }

    /// Create the imaginary unit (0 + 1i)
    pub fn i() -> Self {
        Complex {
            real: Real::from(0.0),
            imag: Real::from(1.0),
        }
    }

    /// Create a complex number from Real parts
    pub fn from_reals(real: Real, imag: Real) -> Self {
        Complex { real, imag }
    }

    /// Create a purely real complex number (imaginary part = 0)
    pub fn from_real(real: f64) -> Self {
        Complex {
            real: Real::from(real),
            imag: Real::from(0.0),
        }
    }

    /// Create a purely imaginary complex number (real part = 0)
    pub fn from_imag(imag: f64) -> Self {
        Complex {
            real: Real::from(0.0),
            imag: Real::from(imag),
        }
    }

    /// Get the real part
    pub fn real(&self) -> f64 {
        self.real.to_f64()
    }

    /// Get the imaginary part
    pub fn imag(&self) -> f64 {
        self.imag.to_f64()
    }

    /// Get the real part as Real
    pub fn real_part(&self) -> &Real {
        &self.real
    }

    /// Get the imaginary part as Real
    pub fn imag_part(&self) -> &Real {
        &self.imag
    }

    /// Compute the modulus (absolute value) |z| = √(a² + b²)
    pub fn abs(&self) -> f64 {
        let r_sq = self.real.clone() * self.real.clone();
        let i_sq = self.imag.clone() * self.imag.clone();
        (r_sq + i_sq).sqrt().to_f64()
    }

    /// Compute the squared modulus |z|² = a² + b²
    pub fn abs_sq(&self) -> f64 {
        let r_sq = self.real.clone() * self.real.clone();
        let i_sq = self.imag.clone() * self.imag.clone();
        (r_sq + i_sq).to_f64()
    }

    /// Compute the argument (phase angle) in radians
    ///
    /// Returns angle θ where z = r·e^(iθ), θ ∈ (-π, π]
    pub fn arg(&self) -> f64 {
        self.imag.atan2(&self.real).to_f64()
    }

    /// Compute the complex conjugate z̄ = a - bi
    pub fn conjugate(&self) -> Self {
        Complex {
            real: self.real.clone(),
            imag: -self.imag.clone(),
        }
    }

    /// Add two complex numbers
    pub fn add(&self, other: &Self) -> Self {
        Complex {
            real: self.real.clone() + other.real.clone(),
            imag: self.imag.clone() + other.imag.clone(),
        }
    }

    /// Subtract two complex numbers
    pub fn sub(&self, other: &Self) -> Self {
        Complex {
            real: self.real.clone() - other.real.clone(),
            imag: self.imag.clone() - other.imag.clone(),
        }
    }

    /// Multiply two complex numbers
    pub fn mul(&self, other: &Self) -> Self {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let ac = self.real.clone() * other.real.clone();
        let bd = self.imag.clone() * other.imag.clone();
        let ad = self.real.clone() * other.imag.clone();
        let bc = self.imag.clone() * other.real.clone();

        Complex {
            real: ac - bd,
            imag: ad + bc,
        }
    }

    /// Divide two complex numbers
    pub fn div(&self, other: &Self) -> Result<Self> {
        let recip = other.reciprocal()?;
        Ok(self.mul(&recip))
    }

    /// Compute the reciprocal 1/z
    pub fn reciprocal(&self) -> Result<Self> {
        let abs_sq = self.abs_sq();
        if abs_sq == 0.0 {
            return Err(MathError::DivisionByZero);
        }

        let conj = self.conjugate();
        let scale = Real::from(abs_sq);

        Ok(Complex {
            real: conj.real / scale.clone(),
            imag: conj.imag / scale,
        })
    }

    /// Compute the exponential e^z
    ///
    /// e^(a+bi) = e^a · (cos(b) + i·sin(b))
    pub fn exp(&self) -> Self {
        let exp_real = self.real.exp();
        let cos_imag = self.imag.cos();
        let sin_imag = self.imag.sin();

        Complex {
            real: exp_real.clone() * cos_imag,
            imag: exp_real * sin_imag,
        }
    }

    /// Compute the natural logarithm ln(z)
    ///
    /// ln(z) = ln|z| + i·arg(z)
    pub fn ln(&self) -> Self {
        let abs = Real::from(self.abs());
        let arg = Real::from(self.arg());

        Complex {
            real: abs.ln(),
            imag: arg,
        }
    }

    /// Compute z raised to power w: z^w
    ///
    /// z^w = e^(w·ln(z))
    pub fn pow(&self, w: &Self) -> Self {
        let ln_z = self.ln();
        let w_ln_z = w.clone() * ln_z;
        w_ln_z.exp()
    }

    /// Compute the square root
    ///
    /// Returns the principal square root (with positive real part, or positive imaginary if real part is zero)
    pub fn sqrt(&self) -> Self {
        let r = self.abs();
        let theta = self.arg();

        let sqrt_r = r.sqrt();
        let half_theta = theta / 2.0;

        Complex::new(
            sqrt_r * half_theta.cos(),
            sqrt_r * half_theta.sin(),
        )
    }

    /// Compute sine: sin(z) = (e^(iz) - e^(-iz)) / (2i)
    pub fn sin(&self) -> Self {
        let i = Complex::new(0.0, 1.0);
        let iz = i.clone() * self.clone();
        let neg_iz = -iz.clone();

        let e_iz = iz.exp();
        let e_neg_iz = neg_iz.exp();

        let two_i = Complex::new(0.0, 2.0);

        (e_iz - e_neg_iz) / two_i
    }

    /// Compute cosine: cos(z) = (e^(iz) + e^(-iz)) / 2
    pub fn cos(&self) -> Self {
        let i = Complex::new(0.0, 1.0);
        let iz = i * self.clone();
        let neg_iz = -iz.clone();

        let e_iz = iz.exp();
        let e_neg_iz = neg_iz.exp();

        let two = Complex::new(2.0, 0.0);

        (e_iz + e_neg_iz) / two
    }

    /// Compute tangent: tan(z) = sin(z) / cos(z)
    pub fn tan(&self) -> Self {
        self.sin() / self.cos()
    }

    /// Compute hyperbolic sine: sinh(z) = (e^z - e^(-z)) / 2
    pub fn sinh(&self) -> Self {
        let e_z = self.exp();
        let e_neg_z = (-self.clone()).exp();
        let two = Complex::new(2.0, 0.0);

        (e_z - e_neg_z) / two
    }

    /// Compute hyperbolic cosine: cosh(z) = (e^z + e^(-z)) / 2
    pub fn cosh(&self) -> Self {
        let e_z = self.exp();
        let e_neg_z = (-self.clone()).exp();
        let two = Complex::new(2.0, 0.0);

        (e_z + e_neg_z) / two
    }

    /// Compute hyperbolic tangent: tanh(z) = sinh(z) / cosh(z)
    pub fn tanh(&self) -> Self {
        self.sinh() / self.cosh()
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.real();
        let i = self.imag();

        if i >= 0.0 {
            write!(f, "{} + {}i", r, i)
        } else {
            write!(f, "{} - {}i", r, -i)
        }
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Complex {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let ac = self.real.clone() * other.real.clone();
        let bd = self.imag.clone() * other.imag.clone();
        let ad = self.real * other.imag.clone();
        let bc = self.imag * other.real;

        Complex {
            real: ac - bd,
            imag: ad + bc,
        }
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        // z1 / z2 = z1 * (1/z2)
        let recip = other.reciprocal().unwrap_or(Complex::new(f64::NAN, f64::NAN));
        self * recip
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Complex {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real && self.imag == other.imag
    }
}

impl Ring for Complex {
    fn zero() -> Self {
        Complex {
            real: Real::zero(),
            imag: Real::zero(),
        }
    }

    fn one() -> Self {
        Complex {
            real: Real::one(),
            imag: Real::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imag.is_zero()
    }

    fn is_one(&self) -> bool {
        self.real.is_one() && self.imag.is_zero()
    }
}

impl CommutativeRing for Complex {}

impl Field for Complex {
    fn inverse(&self) -> Result<Self> {
        self.reciprocal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let z1 = Complex::new(3.0, 4.0);
        let z2 = Complex::new(1.0, 2.0);

        let sum = z1.clone() + z2.clone();
        assert!((sum.real() - 4.0).abs() < 1e-10);
        assert!((sum.imag() - 6.0).abs() < 1e-10);

        let diff = z1.clone() - z2.clone();
        assert!((diff.real() - 2.0).abs() < 1e-10);
        assert!((diff.imag() - 2.0).abs() < 1e-10);

        let prod = z1.clone() * z2.clone();
        // (3+4i)(1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i
        assert!((prod.real() - (-5.0)).abs() < 1e-10);
        assert!((prod.imag() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_abs_arg() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.abs() - 5.0).abs() < 1e-10);

        let i = Complex::new(0.0, 1.0);
        assert!((i.arg() - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_conjugate() {
        let z = Complex::new(3.0, 4.0);
        let conj = z.conjugate();

        assert!((conj.real() - 3.0).abs() < 1e-10);
        assert!((conj.imag() - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_exp_ln() {
        let z = Complex::new(1.0, 1.0);
        let exp_z = z.exp();
        let ln_exp_z = exp_z.ln();

        // ln(e^z) = z (up to branch cut considerations)
        assert!((ln_exp_z.real() - z.real()).abs() < 1e-10);
        assert!((ln_exp_z.imag() - z.imag()).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let z = Complex::new(0.0, 4.0); // 4i
        let sqrt_z = z.sqrt();

        // √(4i) = √2 + √2·i
        let expected = 2.0_f64.sqrt();
        assert!((sqrt_z.real() - expected).abs() < 1e-10);
        assert!((sqrt_z.imag() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_trig() {
        let z = Complex::from_real(0.0);

        let sin_z = z.sin();
        let cos_z = z.cos();

        // sin(0) = 0, cos(0) = 1
        assert!(sin_z.abs() < 1e-10);
        assert!((cos_z.real() - 1.0).abs() < 1e-10);
        assert!(cos_z.imag().abs() < 1e-10);
    }
}
