//! Error types for manifold operations

use thiserror::Error;

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
}

pub type Result<T> = std::result::Result<T, ManifoldError>;
