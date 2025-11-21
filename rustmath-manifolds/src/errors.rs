//! Error types for manifold operations

use thiserror::Error;
use rustmath_core::MathError;

/// Errors that can occur when working with manifolds
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ManifoldError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid chart: {0}")]
    InvalidChart(String),

    #[error("Point not in domain: {0}")]
    PointNotInDomain(String),

    #[error("Chart not found: {0}")]
    ChartNotFound(String),

    #[error("Invalid coordinate: {0}")]
    InvalidCoordinate(String),

    #[error("Manifold operation failed: {0}")]
    OperationFailed(String),

    #[error("Incompatible manifolds: {0}")]
    IncompatibleManifolds(String),

    #[error("No components defined in this chart")]
    NoComponentsInChart,

    #[error("No expression defined in this chart")]
    NoExpressionInChart,

    #[error("Invalid index: {0}")]
    InvalidIndex(String),

    #[error("Different manifolds")]
    DifferentManifolds,

    #[error("Different base points")]
    DifferentBasePoints,

    #[error("Invalid tensor rank")]
    InvalidTensorRank,

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("No chart available")]
    NoChart,

    #[error("Invalid degree: expected {expected}, got {actual}")]
    InvalidDegree { expected: usize, actual: usize },

    #[error("Invalid components")]
    InvalidComponents,

    #[error("Unsupported dimension: {0}")]
    UnsupportedDimension(usize),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    // Added to fix compilation errors - Phase 2
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Invalid structure: {0}")]
    InvalidStructure(String),

    #[error("Invalid point: {0}")]
    InvalidPoint(String),

    #[error("Math error: {0}")]
    MathError(#[from] MathError),
}

pub type Result<T> = std::result::Result<T, ManifoldError>;
