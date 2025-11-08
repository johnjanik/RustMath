//! Interpolation methods

/// Lagrange interpolation
pub fn lagrange_interpolate(x_data: &[f64], y_data: &[f64], x: f64) -> Option<f64> {
    if x_data.len() != y_data.len() || x_data.is_empty() {
        return None;
    }

    let n = x_data.len();
    let mut result = 0.0;

    for i in 0..n {
        let mut term = y_data[i];

        for j in 0..n {
            if i != j {
                term *= (x - x_data[j]) / (x_data[i] - x_data[j]);
            }
        }

        result += term;
    }

    Some(result)
}

/// Linear spline interpolation
pub fn spline_interpolate(x_data: &[f64], y_data: &[f64], x: f64) -> Option<f64> {
    if x_data.len() != y_data.len() || x_data.len() < 2 {
        return None;
    }

    // Find the interval
    let n = x_data.len();

    if x <= x_data[0] {
        return Some(y_data[0]);
    }
    if x >= x_data[n - 1] {
        return Some(y_data[n - 1]);
    }

    for i in 0..n - 1 {
        if x >= x_data[i] && x <= x_data[i + 1] {
            // Linear interpolation
            let t = (x - x_data[i]) / (x_data[i + 1] - x_data[i]);
            return Some(y_data[i] * (1.0 - t) + y_data[i + 1] * t);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagrange() {
        let x_data = vec![0.0, 1.0, 2.0];
        let y_data = vec![0.0, 1.0, 4.0]; // y = x^2

        let result = lagrange_interpolate(&x_data, &y_data, 1.5).unwrap();
        assert!((result - 2.25).abs() < 0.01);
    }

    #[test]
    fn test_spline() {
        let x_data = vec![0.0, 1.0, 2.0];
        let y_data = vec![0.0, 1.0, 2.0];

        let result = spline_interpolate(&x_data, &y_data, 0.5).unwrap();
        assert!((result - 0.5).abs() < 0.01);
    }
}
