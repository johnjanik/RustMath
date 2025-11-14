//! CVXOPT SDP backend

/// CVXOPT SDP backend
#[derive(Debug)]
pub struct CVXOPTSDPBackend {
    matrix_size: usize,
}

impl CVXOPTSDPBackend {
    /// Create a new CVXOPT SDP backend
    pub fn new(matrix_size: usize) -> Self {
        Self { matrix_size }
    }

    /// Get the matrix size
    pub fn matrix_size(&self) -> usize {
        self.matrix_size
    }
}
