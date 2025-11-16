//! Fast Fourier Transform (FFT)
//!
//! This module provides efficient FFT implementations using the Cooley-Tukey
//! algorithm. The FFT is an O(N log N) algorithm for computing the DFT.

use num_complex::Complex64 as Complex;
use std::f64::consts::PI;

/// Base trait for Fourier Transform implementations.
pub trait FastFourierTransformBase {
    /// Performs forward transform.
    fn forward(&self, input: &[Complex]) -> Vec<Complex>;

    /// Performs inverse transform.
    fn inverse(&self, input: &[Complex]) -> Vec<Complex>;

    /// Returns the size of the transform.
    fn size(&self) -> usize;
}

/// Fast Fourier Transform for complex-valued signals.
///
/// Uses the Cooley-Tukey radix-2 decimation-in-time algorithm.
#[derive(Debug, Clone)]
pub struct FastFourierTransformComplex {
    /// Size of the FFT (must be a power of 2)
    size: usize,
    /// Precomputed twiddle factors
    twiddles: Vec<Complex>,
}

impl FastFourierTransformComplex {
    /// Creates a new FFT for the given size.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the FFT (must be a power of 2)
    ///
    /// # Panics
    ///
    /// Panics if size is not a power of 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::transforms::fft::FastFourierTransformComplex;
    ///
    /// let fft = FastFourierTransformComplex::new(8);
    /// ```
    pub fn new(size: usize) -> Self {
        assert!(
            size > 0 && (size & (size - 1)) == 0,
            "FFT size must be a power of 2"
        );

        // Precompute twiddle factors
        let mut twiddles = Vec::with_capacity(size / 2);
        for k in 0..size / 2 {
            let angle = -2.0 * PI * (k as f64) / (size as f64);
            twiddles.push(Complex::new(angle.cos(), angle.sin()));
        }

        FastFourierTransformComplex { size, twiddles }
    }

    /// Computes FFT using Cooley-Tukey radix-2 algorithm.
    fn fft_recursive(&self, input: &[Complex]) -> Vec<Complex> {
        let n = input.len();

        if n <= 1 {
            return input.to_vec();
        }

        if n % 2 != 0 {
            // Fall back to DFT for non-power-of-2
            return self.dft_fallback(input);
        }

        // Divide: split into even and odd indices
        let mut even = Vec::with_capacity(n / 2);
        let mut odd = Vec::with_capacity(n / 2);

        for (i, val) in input.iter().enumerate() {
            if i % 2 == 0 {
                even.push(*val);
            } else {
                odd.push(*val);
            }
        }

        // Conquer: recursively compute FFT of even and odd parts
        let fft_even = self.fft_recursive(&even);
        let fft_odd = self.fft_recursive(&odd);

        // Combine: merge results
        let mut result = vec![Complex::new(0.0, 0.0); n];

        for k in 0..n / 2 {
            let twiddle_idx = k * (self.size / n);
            let twiddle = if twiddle_idx < self.twiddles.len() {
                self.twiddles[twiddle_idx]
            } else {
                let angle = -2.0 * PI * (k as f64) / (n as f64);
                Complex::new(angle.cos(), angle.sin())
            };

            let t = twiddle * fft_odd[k];
            result[k] = fft_even[k] + t;
            result[k + n / 2] = fft_even[k] - t;
        }

        result
    }

    /// Fallback to DFT for non-power-of-2 sizes.
    fn dft_fallback(&self, input: &[Complex]) -> Vec<Complex> {
        let n = input.len();
        let mut output = vec![Complex::new(0.0, 0.0); n];

        for k in 0..n {
            let mut sum = Complex::new(0.0, 0.0);
            for (j, &x) in input.iter().enumerate() {
                let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
                let twiddle = Complex::new(angle.cos(), angle.sin());
                sum = sum + x * twiddle;
            }
            output[k] = sum;
        }

        output
    }

    /// In-place FFT using iterative Cooley-Tukey algorithm.
    fn fft_iterative(&self, input: &[Complex]) -> Vec<Complex> {
        let n = input.len();
        let mut output = input.to_vec();

        // Bit-reversal permutation
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if i < j {
                output.swap(i, j);
            }
        }

        // Iterative FFT
        let mut size = 2;
        while size <= n {
            let half_size = size / 2;
            let table_step = self.size / size;

            for i in (0..n).step_by(size) {
                for j in 0..half_size {
                    let twiddle_idx = j * table_step;
                    let twiddle = self.twiddles[twiddle_idx];

                    let even = output[i + j];
                    let odd = output[i + j + half_size];

                    let t = twiddle * odd;
                    output[i + j] = even + t;
                    output[i + j + half_size] = even - t;
                }
            }

            size *= 2;
        }

        output
    }
}

impl FastFourierTransformBase for FastFourierTransformComplex {
    fn forward(&self, input: &[Complex]) -> Vec<Complex> {
        assert_eq!(
            input.len(),
            self.size,
            "Input size must match FFT size"
        );
        self.fft_iterative(input)
    }

    fn inverse(&self, input: &[Complex]) -> Vec<Complex> {
        assert_eq!(
            input.len(),
            self.size,
            "Input size must match FFT size"
        );

        // Conjugate input
        let conjugated: Vec<Complex> = input.iter().map(|c| c.conj()).collect();

        // Forward FFT
        let result = self.fft_iterative(&conjugated);

        // Conjugate and scale output
        result
            .iter()
            .map(|c| c.conj() / Complex::new(self.size as f64, 0.0))
            .collect()
    }

    fn size(&self) -> usize {
        self.size
    }
}

/// Fast Fourier Transform for real-valued signals.
///
/// Optimized FFT that exploits the conjugate symmetry of real signals.
#[derive(Debug, Clone)]
pub struct FourierTransformReal {
    /// Underlying complex FFT (half size)
    complex_fft: FastFourierTransformComplex,
    /// Full size
    size: usize,
}

impl FourierTransformReal {
    /// Creates a new real-valued FFT.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the real signal (must be a power of 2)
    pub fn new(size: usize) -> Self {
        assert!(
            size > 0 && (size & (size - 1)) == 0,
            "FFT size must be a power of 2"
        );

        FourierTransformReal {
            complex_fft: FastFourierTransformComplex::new(size / 2),
            size,
        }
    }

    /// Computes FFT of a real signal.
    ///
    /// Returns only the first N/2+1 coefficients (the rest are conjugate symmetric).
    pub fn forward_real(&self, input: &[f64]) -> Vec<Complex> {
        assert_eq!(input.len(), self.size, "Input size must match FFT size");

        // Pack real values into complex (even indices = real part, odd indices = imag part)
        let mut packed = Vec::with_capacity(self.size / 2);
        for i in 0..self.size / 2 {
            packed.push(Complex::new(input[2 * i], input[2 * i + 1]));
        }

        // Compute FFT
        let fft_packed = self.complex_fft.forward(&packed);

        // Unpack to get full spectrum (conjugate symmetric)
        let mut result = Vec::with_capacity(self.size / 2 + 1);
        result.push(Complex::new(
            fft_packed[0].re + fft_packed[0].im,
            0.0,
        ));

        for k in 1..self.size / 2 {
            let h1k = fft_packed[k];
            let h2k = fft_packed[self.size / 2 - k].conj();

            let f_even = (h1k + h2k) * Complex::new(0.5, 0.0);
            let f_odd = (h1k - h2k) * Complex::new(0.0, -0.5);

            let angle = -2.0 * PI * (k as f64) / (self.size as f64);
            let w = Complex::new(angle.cos(), angle.sin());

            result.push(f_even + w * f_odd);
        }

        result.push(Complex::new(
            fft_packed[0].re - fft_packed[0].im,
            0.0,
        ));

        result
    }

    /// Computes inverse FFT to get real signal.
    pub fn inverse_real(&self, input: &[Complex]) -> Vec<f64> {
        assert_eq!(
            input.len(),
            self.size / 2 + 1,
            "Input size must be N/2+1"
        );

        // Pack into complex FFT input
        let mut packed = vec![Complex::new(0.0, 0.0); self.size / 2];

        packed[0] = Complex::new(
            (input[0].re + input[self.size / 2].re) * 0.5,
            (input[0].re - input[self.size / 2].re) * 0.5,
        );

        for k in 1..self.size / 2 {
            let fk = input[k];
            let angle = 2.0 * PI * (k as f64) / (self.size as f64);
            let w = Complex::new(angle.cos(), angle.sin());

            let f_even = fk;
            let f_odd = input[self.size / 2 - k].conj();

            packed[k] = f_even + w * f_odd;
        }

        // Inverse FFT
        let ifft_packed = self.complex_fft.inverse(&packed);

        // Unpack real values
        let mut result = Vec::with_capacity(self.size);
        for c in ifft_packed {
            result.push(c.re);
            result.push(c.im);
        }

        result
    }
}

/// Complex Fourier Transform wrapper (for compatibility).
pub type FourierTransformComplex = FastFourierTransformComplex;

/// Convenience function to compute FFT.
///
/// # Arguments
///
/// * `input` - Input signal (complex)
///
/// # Returns
///
/// FFT of the input.
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let fft = FastFourierTransformComplex::new(n);
    fft.forward(input)
}

/// Convenience function to compute inverse FFT.
///
/// # Arguments
///
/// * `input` - Input spectrum (complex)
///
/// # Returns
///
/// Inverse FFT of the input.
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let fft = FastFourierTransformComplex::new(n);
    fft.inverse(input)
}

/// Convenience function to compute FFT of real signal.
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    let n = input.len();
    let fft = FourierTransformReal::new(n);
    fft.forward_real(input)
}

/// Convenience function to compute inverse FFT to real signal.
pub fn irfft(input: &[Complex]) -> Vec<f64> {
    let n = (input.len() - 1) * 2;
    let fft = FourierTransformReal::new(n);
    fft.inverse_real(input)
}

/// Alias for FFT (matching SageMath naming).
pub fn fast_fourier_transform(input: &[Complex]) -> Vec<Complex> {
    fft(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_size_validation() {
        // Should succeed for powers of 2
        let _fft2 = FastFourierTransformComplex::new(2);
        let _fft4 = FastFourierTransformComplex::new(4);
        let _fft8 = FastFourierTransformComplex::new(8);
    }

    #[test]
    #[should_panic(expected = "must be a power of 2")]
    fn test_fft_invalid_size() {
        // Should panic for non-power-of-2
        let _fft = FastFourierTransformComplex::new(3);
    }

    #[test]
    fn test_fft_dc_signal() {
        // DC signal should have all energy in first bin
        let input = vec![Complex::new(1.0, 0.0); 4];
        let output = fft(&input);

        // First bin should be 4.0
        assert!((output[0].re - 4.0).abs() < 1e-10);
        assert!(output[0].im.abs() < 1e-10);

        // Other bins should be near zero
        for i in 1..4 {
            assert!(output[i].norm() < 1e-10);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.5),
            Complex::new(3.0, -0.5),
            Complex::new(4.0, 1.0),
        ];

        let transformed = fft(&input);
        let recovered = ifft(&transformed);

        for (orig, recov) in input.iter().zip(recovered.iter()) {
            assert!((orig.re - recov.re).abs() < 1e-10);
            assert!((orig.im - recov.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_impulse() {
        // Impulse should transform to constant spectrum
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let output = fft(&input);

        // All bins should be 1.0
        for bin in &output {
            assert!((bin.re - 1.0).abs() < 1e-10);
            assert!(bin.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_rfft_real_signal() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = rfft(&input);

        // Output should have N/2+1 = 3 values
        assert_eq!(output.len(), 3);

        // DC component should be sum of inputs
        assert!((output[0].re - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_rfft_irfft_roundtrip() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let spectrum = rfft(&input);
        let recovered = irfft(&spectrum);

        assert_eq!(recovered.len(), input.len());

        for (orig, recov) in input.iter().zip(recovered.iter()) {
            assert!((orig - recov).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fft_larger_size() {
        let input: Vec<Complex> = (0..16).map(|i| Complex::new(i as f64, 0.0)).collect();

        let transformed = fft(&input);
        let recovered = ifft(&transformed);

        for (orig, recov) in input.iter().zip(recovered.iter()) {
            assert!((orig.re - recov.re).abs() < 1e-9);
            assert!((orig.im - recov.im).abs() < 1e-9);
        }
    }

    #[test]
    fn test_fast_fourier_transform_alias() {
        let input = vec![Complex::new(1.0, 0.0); 4];
        let output1 = fft(&input);
        let output2 = fast_fourier_transform(&input);

        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a.re - b.re).abs() < 1e-10);
            assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_base_trait() {
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];

        let fft = FastFourierTransformComplex::new(4);

        let forward = fft.forward(&input);
        let inverse = fft.inverse(&forward);

        assert_eq!(fft.size(), 4);

        for (orig, recov) in input.iter().zip(inverse.iter()) {
            assert!((orig.re - recov.re).abs() < 1e-10);
            assert!((orig.im - recov.im).abs() < 1e-10);
        }
    }
}
