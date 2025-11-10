//! Python bindings for Rational type

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use rustmath_rationals::Rational;
use rustmath_integers::Integer;
use rustmath_core::{Ring, Field};
use crate::integers::PyInteger;

/// Python wrapper for RustMath Rational (exact fractions)
#[pyclass]
#[derive(Clone)]
pub struct PyRational {
    pub(crate) inner: Rational,
}

#[pymethods]
impl PyRational {
    /// Create a new rational from numerator and denominator
    #[new]
    fn new(numerator: i64, denominator: i64) -> PyResult<Self> {
        if denominator == 0 {
            return Err(PyZeroDivisionError::new_err("Denominator cannot be zero"));
        }
        Rational::new(Integer::from(numerator), Integer::from(denominator))
            .map(|inner| PyRational { inner })
            .map_err(|e| PyValueError::new_err(format!("Error creating rational: {:?}", e)))
    }

    /// Create a rational from an integer
    #[staticmethod]
    fn from_integer(n: i64) -> Self {
        PyRational {
            inner: Rational::from_integer(Integer::from(n)),
        }
    }

    /// Create from a PyInteger
    #[staticmethod]
    fn from_pyinteger(n: &PyInteger) -> Self {
        PyRational {
            inner: Rational::from_integer(n.inner.clone()),
        }
    }

    /// Create from string like "3/4"
    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Self> {
        let parts: Vec<&str> = s.split('/').collect();
        match parts.len() {
            1 => {
                let n: i64 = parts[0].trim().parse()
                    .map_err(|_| PyValueError::new_err(format!("Invalid integer: {}", parts[0])))?;
                Ok(PyRational::from_integer(n))
            },
            2 => {
                let n: i64 = parts[0].trim().parse()
                    .map_err(|_| PyValueError::new_err(format!("Invalid numerator: {}", parts[0])))?;
                let d: i64 = parts[1].trim().parse()
                    .map_err(|_| PyValueError::new_err(format!("Invalid denominator: {}", parts[1])))?;
                PyRational::new(n, d)
            },
            _ => Err(PyValueError::new_err(format!("Invalid rational format: {}", s))),
        }
    }

    // ========== Accessors ==========

    /// Get the numerator
    fn numerator(&self) -> PyInteger {
        PyInteger {
            inner: self.inner.numerator().clone(),
        }
    }

    /// Get the denominator
    fn denominator(&self) -> PyInteger {
        PyInteger {
            inner: self.inner.denominator().clone(),
        }
    }

    // ========== Arithmetic Operations ==========

    fn __add__(&self, other: &PyRational) -> PyRational {
        PyRational {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    fn __sub__(&self, other: &PyRational) -> PyRational {
        PyRational {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    fn __mul__(&self, other: &PyRational) -> PyRational {
        PyRational {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }

    fn __truediv__(&self, other: &PyRational) -> PyResult<PyRational> {
        if other.inner.is_zero() {
            return Err(PyZeroDivisionError::new_err("Division by zero"));
        }
        Ok(PyRational {
            inner: self.inner.clone() / other.inner.clone(),
        })
    }

    fn __neg__(&self) -> PyRational {
        PyRational {
            inner: -self.inner.clone(),
        }
    }

    fn __abs__(&self) -> PyRational {
        PyRational {
            inner: self.inner.abs(),
        }
    }

    // ========== Comparison Operations ==========

    fn __eq__(&self, other: &PyRational) -> bool {
        self.inner == other.inner
    }

    fn __ne__(&self, other: &PyRational) -> bool {
        self.inner != other.inner
    }

    fn __lt__(&self, other: &PyRational) -> bool {
        self.inner < other.inner
    }

    fn __le__(&self, other: &PyRational) -> bool {
        self.inner <= other.inner
    }

    fn __gt__(&self, other: &PyRational) -> bool {
        self.inner > other.inner
    }

    fn __ge__(&self, other: &PyRational) -> bool {
        self.inner >= other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    // ========== Operations ==========

    /// Compute reciprocal (1/self)
    fn reciprocal(&self) -> PyResult<PyRational> {
        self.inner.reciprocal()
            .map(|r| PyRational { inner: r })
            .map_err(|e| PyZeroDivisionError::new_err(format!("Error: {:?}", e)))
    }

    /// Compute floor (largest integer <= self)
    fn floor(&self) -> PyInteger {
        PyInteger {
            inner: self.inner.floor(),
        }
    }

    /// Compute ceiling (smallest integer >= self)
    fn ceil(&self) -> PyInteger {
        PyInteger {
            inner: self.inner.ceil(),
        }
    }

    /// Round to nearest integer
    fn round(&self) -> PyInteger {
        PyInteger {
            inner: self.inner.round(),
        }
    }

    /// Raise to an integer power
    fn pow(&self, exp: i32) -> PyResult<PyRational> {
        // Convert signed to unsigned, handle negative by taking reciprocal
        if exp >= 0 {
            let result = self.inner.pow(exp as u32);
            Ok(PyRational { inner: result })
        } else {
            let result = self.inner.pow((-exp) as u32);
            result.reciprocal()
                .map(|r| PyRational { inner: r })
                .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
        }
    }

    // ========== Properties ==========

    /// Check if zero
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Check if one
    fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    /// Check if this is an integer (denominator = 1)
    fn is_integer(&self) -> bool {
        self.inner.denominator().is_one()
    }

    // ========== Conversions ==========

    /// Convert to float (may lose precision)
    fn to_float(&self) -> f64 {
        // Manual conversion: numerator / denominator
        let num = self.inner.numerator();
        let den = self.inner.denominator();

        // Convert to i64 if possible, otherwise use approximation
        use rustmath_core::NumericConversion;
        match (num.to_i64(), den.to_i64()) {
            (Some(n), Some(d)) => (n as f64) / (d as f64),
            _ => f64::NAN, // Number too large for f64
        }
    }

    // ========== String Representation ==========

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("Rational({}, {})", self.inner.numerator(), self.inner.denominator())
    }

    fn __float__(&self) -> f64 {
        self.to_float()
    }
}
