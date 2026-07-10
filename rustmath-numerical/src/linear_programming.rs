//! Linear programming using the Simplex method

/// Linear programming result
#[derive(Clone, Debug)]
pub struct SimplexResult {
    pub optimal_value: f64,
    pub solution: Vec<f64>,
    pub converged: bool,
}

/// Solve a linear program using the Simplex method
///
/// Maximize: c^T x
/// Subject to: Ax <= b, x >= 0
///
/// # Facade warning
///
/// This function is currently a facade: it does not implement the Simplex
/// method and ignores the constraint vector `b`. Calling it panics loudly
/// rather than silently returning an incorrect all-zero solution.
pub fn simplex(
    c: &[f64],
    a: &[Vec<f64>],
    _b: &[f64],
    _max_iter: usize,
) -> Option<SimplexResult> {
    if a.is_empty() || c.is_empty() {
        return None;
    }

    unimplemented!(
        "LP/MIP/SDP solve not yet implemented (facade); planned as rustmath-optimization"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "LP facade -> unimplemented; needs real simplex (Phase 4)"]
    fn test_simplex_basic() {
        let c = vec![3.0, 2.0];
        let a = vec![vec![1.0, 1.0], vec![2.0, 1.0]];
        let b = vec![4.0, 5.0];

        let result = simplex(&c, &a, &b, 100);
        assert!(result.is_some());
    }
}
