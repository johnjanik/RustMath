//! Fast Fourier Transform

use rustmath_complex::Complex;
use std::f64::consts::PI;

/// Fast Fourier Transform
pub fn fft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();

    if n <= 1 {
        return x.to_vec();
    }

    if n % 2 != 0 {
        // Use DFT for non-power-of-2 lengths
        return dft(x);
    }

    // Divide
    let even: Vec<Complex> = x.iter().step_by(2).cloned().collect();
    let odd: Vec<Complex> = x.iter().skip(1).step_by(2).cloned().collect();

    // Conquer
    let fft_even = fft(&even);
    let fft_odd = fft(&odd);

    // Combine
    let mut result = vec![Complex::new(0.0, 0.0); n];

    for k in 0..n / 2 {
        let angle = -2.0 * PI * k as f64 / n as f64;
        let w = Complex::new(angle.cos(), angle.sin());
        let t = w * fft_odd[k].clone();

        result[k] = fft_even[k].clone() + t.clone();
        result[k + n / 2] = fft_even[k].clone() - t;
    }

    result
}

/// Inverse FFT
pub fn ifft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();

    // Take complex conjugate
    let x_conj: Vec<Complex> = x.iter().map(|c| c.conjugate()).collect();

    // Apply FFT
    let fft_result = fft(&x_conj);

    // Conjugate and scale
    fft_result
        .iter()
        .map(|c| c.conjugate() / Complex::new(n as f64, 0.0))
        .collect()
}

/// Direct DFT for non-power-of-2 lengths
fn dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    let mut result = vec![Complex::new(0.0, 0.0); n];

    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (j, xj) in x.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
            let w = Complex::new(angle.cos(), angle.sin());
            sum = sum + w * xj.clone();
        }
        result[k] = sum;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_simple() {
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let result = fft(&input);

        // DC component should be 4.0
        assert!((result[0].real() - 4.0).abs() < 1e-10);
        assert!(result[0].imag().abs() < 1e-10);
    }

    #[test]
    fn test_ifft() {
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];

        let fft_result = fft(&input);
        let ifft_result = ifft(&fft_result);

        for (i, expected) in input.iter().enumerate() {
            assert!((ifft_result[i].real() - expected.real()).abs() < 1e-10);
            assert!((ifft_result[i].imag() - expected.imag()).abs() < 1e-10);
        }
    }
}
