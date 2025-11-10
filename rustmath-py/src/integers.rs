//! Python bindings for Integer type

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use rustmath_integers::{Integer, prime};
use rustmath_core::Ring;

/// Python wrapper for RustMath Integer (arbitrary precision)
#[pyclass]
#[derive(Clone)]
pub struct PyInteger {
    pub(crate) inner: Integer,
}

#[pymethods]
impl PyInteger {
    /// Create a new integer from a Python int
    #[new]
    fn new(value: i64) -> Self {
        PyInteger {
            inner: Integer::from(value),
        }
    }

    /// Create from string representation
    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Self> {
        let value: i64 = s.parse()
            .map_err(|_| PyValueError::new_err(format!("Invalid integer: {}", s)))?;
        Ok(PyInteger {
            inner: Integer::from(value),
        })
    }

    // ========== Arithmetic Operations ==========

    fn __add__(&self, other: &PyInteger) -> PyInteger {
        PyInteger {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    fn __sub__(&self, other: &PyInteger) -> PyInteger {
        PyInteger {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    fn __mul__(&self, other: &PyInteger) -> PyInteger {
        PyInteger {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }

    fn __truediv__(&self, other: &PyInteger) -> PyResult<PyInteger> {
        if other.inner.is_zero() {
            return Err(PyZeroDivisionError::new_err("Division by zero"));
        }
        Ok(PyInteger {
            inner: self.inner.clone() / other.inner.clone(),
        })
    }

    fn __mod__(&self, other: &PyInteger) -> PyResult<PyInteger> {
        if other.inner.is_zero() {
            return Err(PyZeroDivisionError::new_err("Modulo by zero"));
        }
        Ok(PyInteger {
            inner: self.inner.clone() % other.inner.clone(),
        })
    }

    fn __pow__(&self, exponent: u32, _modulo: Option<u32>) -> PyInteger {
        PyInteger {
            inner: self.inner.pow(exponent),
        }
    }

    fn __neg__(&self) -> PyInteger {
        PyInteger {
            inner: -self.inner.clone(),
        }
    }

    fn __abs__(&self) -> PyInteger {
        PyInteger {
            inner: self.inner.abs(),
        }
    }

    // ========== Comparison Operations ==========

    fn __eq__(&self, other: &PyInteger) -> bool {
        self.inner == other.inner
    }

    fn __ne__(&self, other: &PyInteger) -> bool {
        self.inner != other.inner
    }

    fn __lt__(&self, other: &PyInteger) -> bool {
        self.inner < other.inner
    }

    fn __le__(&self, other: &PyInteger) -> bool {
        self.inner <= other.inner
    }

    fn __gt__(&self, other: &PyInteger) -> bool {
        self.inner > other.inner
    }

    fn __ge__(&self, other: &PyInteger) -> bool {
        self.inner >= other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    // ========== Number Theory Operations ==========

    /// Check if the number is prime using Miller-Rabin test
    fn is_prime(&self) -> bool {
        prime::is_prime(&self.inner)
    }

    /// Get the next prime number after this one
    fn next_prime(&self) -> PyInteger {
        PyInteger {
            inner: prime::next_prime(&self.inner),
        }
    }

    /// Get the previous prime number before this one
    fn previous_prime(&self) -> PyResult<PyInteger> {
        match prime::previous_prime(&self.inner) {
            Some(p) => Ok(PyInteger { inner: p }),
            None => Err(PyValueError::new_err("No prime less than 2")),
        }
    }

    /// Compute prime factorization
    /// Returns list of (prime, exponent) tuples
    fn factor(&self) -> Vec<(PyInteger, u32)> {
        prime::factor(&self.inner)
            .into_iter()
            .map(|(p, e)| (PyInteger { inner: p }, e))
            .collect()
    }

    /// Get all positive divisors
    fn divisors(&self) -> PyResult<Vec<PyInteger>> {
        self.inner.divisors()
            .map(|divs| divs.into_iter().map(|d| PyInteger { inner: d }).collect())
            .map_err(|e| PyValueError::new_err(format!("Error computing divisors: {:?}", e)))
    }

    /// Count the number of divisors (tau function)
    fn num_divisors(&self) -> PyResult<PyInteger> {
        self.inner.num_divisors()
            .map(|n| PyInteger { inner: n })
            .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
    }

    /// Sum of divisors (sigma function)
    fn sum_divisors(&self) -> PyResult<PyInteger> {
        self.inner.sum_divisors()
            .map(|n| PyInteger { inner: n })
            .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
    }

    /// Compute GCD with another integer
    fn gcd(&self, other: &PyInteger) -> PyInteger {
        PyInteger {
            inner: self.inner.gcd(&other.inner),
        }
    }

    /// Compute LCM with another integer
    fn lcm(&self, other: &PyInteger) -> PyInteger {
        PyInteger {
            inner: self.inner.lcm(&other.inner),
        }
    }

    /// Extended GCD: returns (gcd, s, t) where gcd = s*self + t*other
    fn extended_gcd(&self, other: &PyInteger) -> (PyInteger, PyInteger, PyInteger) {
        let (g, s, t) = self.inner.extended_gcd(&other.inner);
        (
            PyInteger { inner: g },
            PyInteger { inner: s },
            PyInteger { inner: t },
        )
    }

    /// Compute modular inverse: self^-1 (mod m)
    fn mod_inverse(&self, modulus: &PyInteger) -> PyResult<PyInteger> {
        let (gcd, s, _) = self.inner.extended_gcd(&modulus.inner);
        if !gcd.is_one() {
            return Err(PyValueError::new_err(
                format!("No modular inverse: gcd({}, {}) != 1", self.inner, modulus.inner)
            ));
        }
        // Ensure the result is positive
        let mut result = s % modulus.inner.clone();
        if result.signum() < 0 {
            result = result + modulus.inner.clone();
        }
        Ok(PyInteger { inner: result })
    }

    /// Modular exponentiation: self^exp (mod m)
    fn mod_pow(&self, exp: &PyInteger, modulus: &PyInteger) -> PyResult<PyInteger> {
        self.inner.mod_pow(&exp.inner, &modulus.inner)
            .map(|r| PyInteger { inner: r })
            .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
    }

    // ========== Basic Properties ==========

    /// Check if even
    fn is_even(&self) -> bool {
        self.inner.is_even()
    }

    /// Check if odd
    fn is_odd(&self) -> bool {
        self.inner.is_odd()
    }

    /// Check if zero
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Check if one
    fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    /// Get the sign: -1, 0, or 1
    fn signum(&self) -> i8 {
        self.inner.signum()
    }

    /// Get bit length
    fn bit_length(&self) -> u64 {
        self.inner.bit_length()
    }

    // ========== Root Operations ==========

    /// Compute integer square root (floor(sqrt(self)))
    fn sqrt(&self) -> PyResult<PyInteger> {
        self.inner.sqrt()
            .map(|r| PyInteger { inner: r })
            .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
    }

    /// Compute nth root (floor(nth_root(self)))
    fn nth_root(&self, n: u32) -> PyResult<PyInteger> {
        self.inner.nth_root(n)
            .map(|r| PyInteger { inner: r })
            .map_err(|e| PyValueError::new_err(format!("Error: {:?}", e)))
    }

    // ========== String Representation ==========

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("Integer({})", self.inner)
    }

    fn __int__(&self) -> PyResult<i64> {
        use rustmath_core::NumericConversion;
        self.inner.to_i64()
            .ok_or_else(|| PyValueError::new_err("Integer too large to convert to i64"))
    }
}
