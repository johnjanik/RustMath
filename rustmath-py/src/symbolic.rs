//! Python bindings for symbolic expressions

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyTypeError};
use pyo3::types::PyString;
use rustmath_symbolic::{Expr as RustExpr, Symbol as RustSymbol, parse as rust_parse};
use std::sync::Arc;

/// Python wrapper for Symbol
#[pyclass(name = "Symbol")]
#[derive(Clone)]
pub struct PySymbol {
    pub inner: RustSymbol,
}

#[pymethods]
impl PySymbol {
    /// Create a new symbol
    #[new]
    fn new(name: &str) -> Self {
        PySymbol {
            inner: RustSymbol::new(name),
        }
    }

    /// Get the symbol name
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Symbol('{}')", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.inner.name().to_string()
    }

    /// Convert to expression
    fn to_expr(&self) -> PyExpr {
        PyExpr {
            inner: RustExpr::Symbol(self.inner.clone()),
        }
    }
}

/// Python wrapper for Expr
#[pyclass(name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: RustExpr,
}

#[pymethods]
impl PyExpr {
    /// Addition
    fn __add__(&self, other: &PyExpr) -> PyExpr {
        PyExpr {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    /// Subtraction
    fn __sub__(&self, other: &PyExpr) -> PyExpr {
        PyExpr {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    /// Multiplication
    fn __mul__(&self, other: &PyExpr) -> PyExpr {
        PyExpr {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }

    /// Division
    fn __truediv__(&self, other: &PyExpr) -> PyExpr {
        PyExpr {
            inner: self.inner.clone() / other.inner.clone(),
        }
    }

    /// Power (x ** y)
    fn __pow__(&self, other: &PyExpr, _modulo: Option<PyObject>) -> PyExpr {
        PyExpr {
            inner: self.inner.clone().pow(other.inner.clone()),
        }
    }

    /// Negation
    fn __neg__(&self) -> PyExpr {
        PyExpr {
            inner: -self.inner.clone(),
        }
    }

    /// String representation (debug format)
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// String representation (display format)
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// Differentiate with respect to a symbol
    fn diff(&self, symbol: &PySymbol) -> PyExpr {
        PyExpr {
            inner: self.inner.differentiate(&symbol.inner),
        }
    }

    /// Integrate with respect to a symbol
    fn integrate(&self, symbol: &PySymbol) -> PyResult<PyExpr> {
        match self.inner.integrate(&symbol.inner) {
            Some(result) => Ok(PyExpr { inner: result }),
            None => Err(PyValueError::new_err("Integration failed")),
        }
    }

    /// Simplify the expression
    fn simplify(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.simplify(),
        }
    }

    /// Expand the expression
    fn expand(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.expand(),
        }
    }

    /// Factor the expression
    fn factor(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.factor(),
        }
    }

    /// Substitute a symbol with an expression
    fn subs(&self, symbol: &PySymbol, value: &PyExpr) -> PyExpr {
        PyExpr {
            inner: self.inner.substitute(&symbol.inner, &value.inner),
        }
    }

    /// Check if the expression is constant
    fn is_constant(&self) -> bool {
        self.inner.is_constant()
    }

    /// Mathematical functions
    fn sin(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.clone().sin(),
        }
    }

    fn cos(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.clone().cos(),
        }
    }

    fn tan(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.clone().tan(),
        }
    }

    fn exp(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.clone().exp(),
        }
    }

    fn log(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.clone().log(),
        }
    }

    fn sqrt(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.clone().sqrt(),
        }
    }

    fn abs(&self) -> PyExpr {
        PyExpr {
            inner: self.inner.clone().abs(),
        }
    }
}

/// Parse a string into an expression
#[pyfunction]
fn parse(input: &str) -> PyResult<PyExpr> {
    match rust_parse(input) {
        Ok(expr) => Ok(PyExpr { inner: expr }),
        Err(e) => Err(PyValueError::new_err(format!("Parse error: {}", e))),
    }
}

/// Create a symbol (convenience function)
#[pyfunction]
fn symbols(names: &str) -> PyResult<Vec<PySymbol>> {
    let symbol_names: Vec<&str> = names.split_whitespace().collect();
    Ok(symbol_names
        .into_iter()
        .map(|name| PySymbol {
            inner: RustSymbol::new(name),
        })
        .collect())
}

/// Mathematical constants
#[pyfunction]
fn pi() -> PyExpr {
    PyExpr {
        inner: RustExpr::symbol("pi"),
    }
}

#[pyfunction]
fn e() -> PyExpr {
    PyExpr {
        inner: RustExpr::symbol("e"),
    }
}

#[pyfunction]
fn I() -> PyExpr {
    PyExpr {
        inner: RustExpr::symbol("I"),
    }
}

#[pyfunction]
fn oo() -> PyExpr {
    PyExpr {
        inner: RustExpr::symbol("oo"),
    }
}

/// Top-level functions for calculus
#[pyfunction]
fn diff(expr: &PyExpr, symbol: &PySymbol) -> PyExpr {
    expr.diff(symbol)
}

#[pyfunction]
fn integrate_fn(expr: &PyExpr, symbol: &PySymbol) -> PyResult<PyExpr> {
    expr.integrate(symbol)
}

#[pyfunction]
fn simplify(expr: &PyExpr) -> PyExpr {
    expr.simplify()
}

#[pyfunction]
fn expand(expr: &PyExpr) -> PyExpr {
    expr.expand()
}

#[pyfunction]
fn factor(expr: &PyExpr) -> PyExpr {
    expr.factor()
}

/// Mathematical functions
#[pyfunction]
fn sin(expr: &PyExpr) -> PyExpr {
    expr.sin()
}

#[pyfunction]
fn cos(expr: &PyExpr) -> PyExpr {
    expr.cos()
}

#[pyfunction]
fn tan(expr: &PyExpr) -> PyExpr {
    expr.tan()
}

#[pyfunction]
fn exp_fn(expr: &PyExpr) -> PyExpr {
    expr.exp()
}

#[pyfunction]
fn log(expr: &PyExpr) -> PyExpr {
    expr.log()
}

#[pyfunction]
fn sqrt(expr: &PyExpr) -> PyExpr {
    expr.sqrt()
}

#[pyfunction]
fn abs_fn(expr: &PyExpr) -> PyExpr {
    expr.abs()
}

/// Register symbolic module with Python
pub fn register_symbolic_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<PySymbol>()?;
    m.add_class::<PyExpr>()?;

    // Parsing
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(symbols, m)?)?;

    // Constants
    m.add_function(wrap_pyfunction!(pi, m)?)?;
    m.add_function(wrap_pyfunction!(e, m)?)?;
    m.add_function(wrap_pyfunction!(I, m)?)?;
    m.add_function(wrap_pyfunction!(oo, m)?)?;

    // Calculus operations
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_fn, m)?)?;
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    m.add_function(wrap_pyfunction!(expand, m)?)?;
    m.add_function(wrap_pyfunction!(factor, m)?)?;

    // Mathematical functions
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(exp_fn, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(abs_fn, m)?)?;

    Ok(())
}
