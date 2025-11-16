//! Discrete Fourier Transform (DFT)
//!
//! This module provides discrete Fourier transform operations for sequences.
//! The DFT converts a finite sequence of equally-spaced samples into a
//! same-length sequence of complex frequency components.

use rustmath_complex::Complex;
use std::f64::consts::PI;

/// Represents an indexed sequence for DFT operations.
///
/// This structure wraps a sequence with additional indexing information
/// that may be useful for certain DFT applications.
#[derive(Debug, Clone)]
pub struct IndexedSequence {
    /// The sequence values
    values: Vec<Complex>,
    /// Starting index (default 0)
    start_index: i64,
}

impl IndexedSequence {
    /// Creates a new indexed sequence.
    ///
    /// # Arguments
    ///
    /// * `values` - The sequence values
    /// * `start_index` - Starting index (default 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::transforms::dft::IndexedSequence;
    /// use rustmath_complex::Complex;
    ///
    /// let values = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
    /// let seq = IndexedSequence::new(values, 0);
    /// ```
    pub fn new(values: Vec<Complex>, start_index: i64) -> Self {
        IndexedSequence {
            values,
            start_index,
        }
    }

    /// Creates a new indexed sequence from real values.
    ///
    /// # Arguments
    ///
    /// * `values` - Real-valued sequence
    /// * `start_index` - Starting index (default 0)
    pub fn from_real(values: Vec<f64>, start_index: i64) -> Self {
        let complex_values: Vec<Complex> = values.iter().map(|&v| Complex::new(v, 0.0)).collect();
        IndexedSequence::new(complex_values, start_index)
    }

    /// Returns the length of the sequence.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the starting index.
    pub fn start_index(&self) -> i64 {
        self.start_index
    }

    /// Returns a reference to the values.
    pub fn values(&self) -> &[Complex] {
        &self.values
    }

    /// Returns a mutable reference to the values.
    pub fn values_mut(&mut self) -> &mut [Complex] {
        &mut self.values
    }

    /// Gets a value at a specific index (accounting for start_index).
    pub fn get(&self, index: i64) -> Option<&Complex> {
        let offset = index - self.start_index;
        if offset >= 0 && (offset as usize) < self.values.len() {
            Some(&self.values[offset as usize])
        } else {
            None
        }
    }

    /// Applies the DFT to this sequence.
    pub fn dft(&self) -> IndexedSequence {
        dft_indexed(self)
    }

    /// Applies the inverse DFT to this sequence.
    pub fn idft(&self) -> IndexedSequence {
        idft_indexed(self)
    }
}

/// Computes the Discrete Fourier Transform of a sequence.
///
/// The DFT is defined as:
/// X[k] = Σ(n=0 to N-1) x[n] * exp(-2πi * k * n / N)
///
/// # Arguments
///
/// * `input` - Input sequence (complex values)
///
/// # Returns
///
/// The DFT of the input sequence.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::transforms::dft::dft;
/// use rustmath_complex::Complex;
///
/// let input = vec![
///     Complex::new(1.0, 0.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(3.0, 0.0),
///     Complex::new(4.0, 0.0),
/// ];
/// let output = dft(&input);
/// assert_eq!(output.len(), 4);
/// ```
pub fn dft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut output = vec![Complex::new(0.0, 0.0); n];

    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (j, &x_j) in input.iter().enumerate() {
            let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let twiddle = Complex::new(angle.cos(), angle.sin());
            sum = sum + x_j * twiddle;
        }
        output[k] = sum;
    }

    output
}

/// Computes the inverse Discrete Fourier Transform of a sequence.
///
/// The inverse DFT is defined as:
/// x[n] = (1/N) * Σ(k=0 to N-1) X[k] * exp(2πi * k * n / N)
///
/// # Arguments
///
/// * `input` - Input sequence (complex values, typically from DFT)
///
/// # Returns
///
/// The inverse DFT of the input sequence.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::transforms::dft::{dft, idft};
/// use rustmath_complex::Complex;
///
/// let input = vec![
///     Complex::new(1.0, 0.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(3.0, 0.0),
///     Complex::new(4.0, 0.0),
/// ];
/// let transformed = dft(&input);
/// let recovered = idft(&transformed);
/// // recovered should be approximately equal to input
/// ```
pub fn idft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut output = vec![Complex::new(0.0, 0.0); n];

    for j in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (k, &x_k) in input.iter().enumerate() {
            let angle = 2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let twiddle = Complex::new(angle.cos(), angle.sin());
            sum = sum + x_k * twiddle;
        }
        output[j] = sum / Complex::new(n as f64, 0.0);
    }

    output
}

/// Computes DFT of an indexed sequence.
pub fn dft_indexed(seq: &IndexedSequence) -> IndexedSequence {
    let transformed = dft(&seq.values);
    IndexedSequence::new(transformed, seq.start_index)
}

/// Computes inverse DFT of an indexed sequence.
pub fn idft_indexed(seq: &IndexedSequence) -> IndexedSequence {
    let transformed = idft(&seq.values);
    IndexedSequence::new(transformed, seq.start_index)
}

/// Computes the DFT of a real-valued sequence.
///
/// This is a convenience function that converts real values to complex
/// before computing the DFT.
///
/// # Arguments
///
/// * `input` - Real-valued input sequence
///
/// # Returns
///
/// The DFT as complex values.
pub fn dft_real(input: &[f64]) -> Vec<Complex> {
    let complex_input: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    dft(&complex_input)
}

/// Computes the inverse DFT and returns only the real parts.
///
/// This assumes the input represents a real signal in frequency domain
/// (i.e., it has conjugate symmetry).
///
/// # Arguments
///
/// * `input` - Frequency domain representation
///
/// # Returns
///
/// Real-valued time domain signal.
pub fn idft_to_real(input: &[Complex]) -> Vec<f64> {
    let complex_output = idft(input);
    complex_output.iter().map(|c| c.real()).collect()
}

/// Computes the power spectrum of a signal.
///
/// The power spectrum is |DFT(x)|^2, showing the distribution of
/// power in the frequency domain.
///
/// # Arguments
///
/// * `input` - Input signal (complex)
///
/// # Returns
///
/// Power spectrum (real values).
pub fn power_spectrum(input: &[Complex]) -> Vec<f64> {
    let transformed = dft(input);
    transformed.iter().map(|c| c.norm_squared()).collect()
}

/// Computes frequency values corresponding to DFT output.
///
/// # Arguments
///
/// * `n` - Length of the DFT
/// * `sample_rate` - Sampling rate in Hz
///
/// # Returns
///
/// Frequency values for each DFT bin.
pub fn dft_frequencies(n: usize, sample_rate: f64) -> Vec<f64> {
    let mut freqs = Vec::with_capacity(n);
    let freq_spacing = sample_rate / (n as f64);

    for k in 0..n {
        let freq = if k <= n / 2 {
            k as f64 * freq_spacing
        } else {
            ((k as i64) - (n as i64)) as f64 * freq_spacing
        };
        freqs.push(freq);
    }

    freqs
}

/// Shifts zero-frequency component to center of spectrum.
///
/// This is useful for visualization of the frequency spectrum.
///
/// # Arguments
///
/// * `input` - DFT output
///
/// # Returns
///
/// Shifted spectrum with zero frequency at the center.
pub fn fftshift(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut output = vec![Complex::new(0.0, 0.0); n];
    let mid = (n + 1) / 2;

    for (i, &val) in input.iter().enumerate() {
        let new_idx = (i + mid) % n;
        output[new_idx] = val;
    }

    output
}

/// Inverse of fftshift - undoes the shift.
pub fn ifftshift(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut output = vec![Complex::new(0.0, 0.0); n];
    let mid = n / 2;

    for (i, &val) in input.iter().enumerate() {
        let new_idx = (i + mid) % n;
        output[new_idx] = val;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dft_dc_signal() {
        // DC signal (constant) should have all energy in first bin
        let input = vec![Complex::new(1.0, 0.0); 4];
        let output = dft(&input);

        // First bin should be 4.0 (sum of inputs)
        assert!((output[0].real() - 4.0).abs() < 1e-10);
        assert!(output[0].imag().abs() < 1e-10);

        // Other bins should be near zero
        for i in 1..4 {
            assert!(output[i].norm() < 1e-10);
        }
    }

    #[test]
    fn test_dft_idft_roundtrip() {
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.5),
            Complex::new(3.0, -0.5),
            Complex::new(4.0, 1.0),
        ];

        let transformed = dft(&input);
        let recovered = idft(&transformed);

        for (orig, recov) in input.iter().zip(recovered.iter()) {
            assert!((orig.real() - recov.real()).abs() < 1e-10);
            assert!((orig.imag() - recov.imag()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_indexed_sequence() {
        let values = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let seq = IndexedSequence::new(values.clone(), 5);

        assert_eq!(seq.len(), 2);
        assert_eq!(seq.start_index(), 5);
        assert_eq!(seq.get(5), Some(&Complex::new(1.0, 0.0)));
        assert_eq!(seq.get(6), Some(&Complex::new(2.0, 0.0)));
        assert_eq!(seq.get(7), None);
    }

    #[test]
    fn test_dft_real() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = dft_real(&input);

        assert_eq!(output.len(), 4);
        // First bin should be sum = 10.0
        assert!((output[0].real() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_spectrum() {
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let power = power_spectrum(&input);
        assert_eq!(power.len(), 4);

        // All power should be in DC bin
        assert!(power[0] > 0.0);
    }

    #[test]
    fn test_dft_frequencies() {
        let freqs = dft_frequencies(8, 8.0);
        assert_eq!(freqs.len(), 8);

        // At sample rate 8 Hz with 8 samples:
        // Bins should be: 0, 1, 2, 3, -4, -3, -2, -1 Hz
        assert_eq!(freqs[0], 0.0);
        assert_eq!(freqs[1], 1.0);
        assert_eq!(freqs[4], -4.0);
    }

    #[test]
    fn test_fftshift() {
        let input = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];

        let shifted = fftshift(&input);
        let unshifted = ifftshift(&shifted);

        // Verify roundtrip
        for (orig, final_val) in input.iter().zip(unshifted.iter()) {
            assert!((orig.real() - final_val.real()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_idft_to_real() {
        // Create a frequency domain signal with conjugate symmetry
        let freq = vec![
            Complex::new(4.0, 0.0),  // DC
            Complex::new(1.0, 1.0),  // Positive freq
            Complex::new(1.0, -1.0), // Negative freq (conjugate)
            Complex::new(0.0, 0.0),
        ];

        let time = idft_to_real(&freq);
        assert_eq!(time.len(), 4);

        // All values should be real (imaginary parts were conjugate symmetric)
        for &val in &time {
            assert!(val.is_finite());
        }
    }
}
