//! PyO3 Python bindings for RustMath
//!
//! This crate provides Python bindings for RustMath, allowing direct
//! integration with SageMath for testing and validation.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

mod integers;
mod rationals;
mod matrix;

pub use integers::PyInteger;
pub use rationals::PyRational;
pub use matrix::PyMatrix;

/// Main module initialization
#[pymodule]
fn rustmath(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyInteger>()?;
    m.add_class::<PyRational>()?;
    m.add_class::<PyMatrix>()?;

    // Module-level functions
    m.add_function(wrap_pyfunction!(gcd_many, m)?)?;
    m.add_function(wrap_pyfunction!(lcm_many, m)?)?;

    Ok(())
}

/// Compute GCD of multiple integers
#[pyfunction]
fn gcd_many(values: Vec<PyInteger>) -> PyResult<PyInteger> {
    if values.is_empty() {
        return Err(PyValueError::new_err("Need at least one value"));
    }

    let mut result = values[0].inner.clone();
    for val in &values[1..] {
        result = result.gcd(&val.inner);
    }

    Ok(PyInteger { inner: result })
}

/// Compute LCM of multiple integers
#[pyfunction]
fn lcm_many(values: Vec<PyInteger>) -> PyResult<PyInteger> {
    if values.is_empty() {
        return Err(PyValueError::new_err("Need at least one value"));
    }

    let mut result = values[0].inner.clone();
    for val in &values[1..] {
        result = result.lcm(&val.inner);
    }

    Ok(PyInteger { inner: result })
}
