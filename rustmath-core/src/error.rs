//! Error types for mathematical operations

use thiserror::Error;

/// Result type for mathematical operations
pub type Result<T> = std::result::Result<T, MathError>;

/// Errors that can occur during mathematical operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MathError {
    /// Division by zero
    #[error("Division by zero")]
    DivisionByZero,

    /// Operation on incompatible types
    #[error("Type error: {0}")]
    TypeError(String),

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Numerical error (overflow, underflow, etc.)
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Domain error (e.g., sqrt of negative number in reals)
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Not invertible (e.g., singular matrix)
    #[error("Element is not invertible")]
    NotInvertible,

    /// Not yet implemented
    #[error("Not yet implemented: {0}")]
    NotImplemented(String),
}
