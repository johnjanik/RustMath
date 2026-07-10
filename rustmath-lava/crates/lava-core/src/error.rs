use std::path::PathBuf;
use thiserror::Error;

/// Source of the active query string, for diagnostics.
#[derive(Debug, Clone)]
pub enum QuerySource {
    /// Compiled into the binary.
    Embedded,
    /// Path on disk (CWD walk-up, XDG, or explicit --query).
    Path(PathBuf),
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("parse failed in {path}: {message}")]
    Parse { path: String, message: String },

    #[error("query error from {from:?}: {message}")]
    Query { from: QuerySource, message: String },

    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("formatter error: {0}")]
    Topiary(#[from] topiary_core::FormatterError),
}

pub type Result<T> = std::result::Result<T, Error>;
