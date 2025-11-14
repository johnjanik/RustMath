//! Generic backend for semidefinite programming (SDP)

/// Generic SDP backend
#[derive(Debug)]
pub struct GenericSDPBackend {
    matrix_size: usize,
}

impl GenericSDPBackend {
    /// Create a new generic SDP backend
    pub fn new(matrix_size: usize) -> Self {
        Self { matrix_size }
    }

    /// Get the matrix size
    pub fn matrix_size(&self) -> usize {
        self.matrix_size
    }
}
