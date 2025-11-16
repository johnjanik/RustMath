//! Discrete Wavelet Transform (DWT)
//!
//! This module provides discrete wavelet transform operations using various
//! wavelet families (Haar, Daubechies, etc.). The DWT decomposes a signal
//! into wavelet coefficients at different scales and positions.

/// Wavelet family type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletFamily {
    /// Haar wavelet (simplest wavelet)
    Haar,
    /// Daubechies wavelets (db2, db4, db6, etc.)
    Daubechies(usize),
    /// Symlet wavelets
    Symlet(usize),
    /// Coiflet wavelets
    Coiflet(usize),
}

impl WaveletFamily {
    /// Returns the filter coefficients for this wavelet.
    pub fn get_filters(&self) -> (Vec<f64>, Vec<f64>) {
        match self {
            WaveletFamily::Haar => haar_filters(),
            WaveletFamily::Daubechies(n) => daubechies_filters(*n),
            WaveletFamily::Symlet(n) => symlet_filters(*n),
            WaveletFamily::Coiflet(n) => coiflet_filters(*n),
        }
    }
}

/// Discrete Wavelet Transform structure
#[derive(Debug, Clone)]
pub struct DiscreteWaveletTransform {
    /// Wavelet family
    wavelet: WaveletFamily,
    /// Low-pass decomposition filter
    dec_lo: Vec<f64>,
    /// High-pass decomposition filter
    dec_hi: Vec<f64>,
    /// Low-pass reconstruction filter
    rec_lo: Vec<f64>,
    /// High-pass reconstruction filter
    rec_hi: Vec<f64>,
}

impl DiscreteWaveletTransform {
    /// Creates a new DWT with the specified wavelet.
    ///
    /// # Arguments
    ///
    /// * `wavelet` - Wavelet family to use
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::transforms::dwt::{DiscreteWaveletTransform, WaveletFamily};
    ///
    /// let dwt = DiscreteWaveletTransform::new(WaveletFamily::Haar);
    /// ```
    pub fn new(wavelet: WaveletFamily) -> Self {
        let (dec_lo, dec_hi) = wavelet.get_filters();
        let (rec_lo, rec_hi) = reconstruction_filters(&dec_lo, &dec_hi);

        DiscreteWaveletTransform {
            wavelet,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
        }
    }

    /// Performs single-level wavelet decomposition.
    ///
    /// Returns (approximation coefficients, detail coefficients).
    pub fn decompose(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if signal.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let approx = convolve_downsample(signal, &self.dec_lo);
        let detail = convolve_downsample(signal, &self.dec_hi);

        (approx, detail)
    }

    /// Performs single-level wavelet reconstruction.
    ///
    /// Reconstructs signal from approximation and detail coefficients.
    pub fn reconstruct(&self, approx: &[f64], detail: &[f64]) -> Vec<f64> {
        if approx.is_empty() && detail.is_empty() {
            return Vec::new();
        }

        let approx_up = upsample_convolve(approx, &self.rec_lo);
        let detail_up = upsample_convolve(detail, &self.rec_hi);

        // Add the two contributions
        let len = approx_up.len().max(detail_up.len());
        let mut result = vec![0.0; len];

        for i in 0..len {
            if i < approx_up.len() {
                result[i] += approx_up[i];
            }
            if i < detail_up.len() {
                result[i] += detail_up[i];
            }
        }

        result
    }

    /// Performs multi-level wavelet decomposition.
    ///
    /// Returns approximation coefficients and detail coefficients at each level.
    pub fn wavedec(&self, signal: &[f64], level: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        if signal.is_empty() || level == 0 {
            return (signal.to_vec(), Vec::new());
        }

        let mut approx = signal.to_vec();
        let mut details = Vec::new();

        for _ in 0..level {
            let (new_approx, new_detail) = self.decompose(&approx);
            approx = new_approx;
            details.push(new_detail);
        }

        // Reverse details so they're in order from coarsest to finest
        details.reverse();

        (approx, details)
    }

    /// Performs multi-level wavelet reconstruction.
    ///
    /// Reconstructs signal from multi-level decomposition.
    pub fn waverec(&self, approx: &[f64], details: &[Vec<f64>]) -> Vec<f64> {
        if details.is_empty() {
            return approx.to_vec();
        }

        let mut signal = approx.to_vec();

        // Reconstruct from coarsest to finest (reverse order)
        for detail in details.iter().rev() {
            signal = self.reconstruct(&signal, detail);
        }

        signal
    }
}

/// Convenience function to perform DWT.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Wavelet family to use
/// * `level` - Number of decomposition levels
///
/// # Returns
///
/// (approximation coefficients, detail coefficients at each level)
pub fn dwt(signal: &[f64], wavelet: WaveletFamily, level: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let transform = DiscreteWaveletTransform::new(wavelet);
    transform.wavedec(signal, level)
}

/// Alias for dwt function (WaveletTransform name from SageMath).
pub fn wavelet_transform(
    signal: &[f64],
    wavelet: WaveletFamily,
    level: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    dwt(signal, wavelet, level)
}

/// Alias for DWT function.
pub fn dft_alias(
    signal: &[f64],
    wavelet: WaveletFamily,
    level: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    dwt(signal, wavelet, level)
}

/// Checks if a number is a power of 2.
///
/// # Arguments
///
/// * `n` - Number to check
///
/// # Returns
///
/// `true` if n is a power of 2, `false` otherwise.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::transforms::dwt::is2pow;
///
/// assert!(is2pow(1));
/// assert!(is2pow(2));
/// assert!(is2pow(4));
/// assert!(is2pow(8));
/// assert!(!is2pow(3));
/// assert!(!is2pow(5));
/// ```
pub fn is2pow(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

// Filter coefficient functions

/// Returns Haar wavelet filter coefficients.
fn haar_filters() -> (Vec<f64>, Vec<f64>) {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let dec_lo = vec![inv_sqrt2, inv_sqrt2];
    let dec_hi = vec![inv_sqrt2, -inv_sqrt2];
    (dec_lo, dec_hi)
}

/// Returns Daubechies wavelet filter coefficients.
fn daubechies_filters(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        2 => {
            // db2 (same as Haar)
            haar_filters()
        }
        4 => {
            // db4 coefficients
            let dec_lo = vec![
                0.6830127,
                1.1830127,
                0.3169873,
                -0.1830127,
            ];
            let dec_hi = vec![
                -0.1830127,
                -0.3169873,
                1.1830127,
                -0.6830127,
            ];
            (normalize_filter(&dec_lo), normalize_filter(&dec_hi))
        }
        6 => {
            // db6 coefficients
            let dec_lo = vec![
                0.47046721,
                1.14111692,
                0.650365,
                -0.19093442,
                -0.12083221,
                0.0498175,
            ];
            let dec_hi = vec![
                0.0498175,
                0.12083221,
                -0.19093442,
                -0.650365,
                1.14111692,
                -0.47046721,
            ];
            (normalize_filter(&dec_lo), normalize_filter(&dec_hi))
        }
        _ => {
            // Default to Haar for unsupported orders
            haar_filters()
        }
    }
}

/// Returns Symlet wavelet filter coefficients.
fn symlet_filters(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Symlets are nearly symmetric versions of Daubechies wavelets
    // For now, use Daubechies as approximation
    daubechies_filters(n)
}

/// Returns Coiflet wavelet filter coefficients.
fn coiflet_filters(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Coiflets have vanishing moments for both scaling and wavelet functions
    // For now, use Daubechies as approximation
    daubechies_filters(n)
}

/// Normalizes a filter to have unit norm.
fn normalize_filter(filter: &[f64]) -> Vec<f64> {
    let norm: f64 = filter.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return filter.to_vec();
    }
    filter.iter().map(|x| x / norm).collect()
}

/// Constructs reconstruction filters from decomposition filters.
fn reconstruction_filters(dec_lo: &[f64], dec_hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Reconstruction filters are time-reversed and alternating-sign versions
    let rec_lo: Vec<f64> = dec_lo.iter().rev().copied().collect();

    let rec_hi: Vec<f64> = dec_hi
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &x)| if i % 2 == 0 { x } else { -x })
        .collect();

    (rec_lo, rec_hi)
}

// Convolution and sampling operations

/// Convolves signal with filter and downsamples by 2.
fn convolve_downsample(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    if signal.is_empty() || filter.is_empty() {
        return Vec::new();
    }

    let n = signal.len();
    let m = filter.len();
    let out_len = (n + 1) / 2;

    let mut result = Vec::with_capacity(out_len);

    for i in (0..n).step_by(2) {
        let mut sum = 0.0;
        for (j, &f) in filter.iter().enumerate() {
            let idx = if i + j < m - 1 {
                // Wrap around (periodic extension)
                (n + i + j - m + 1) % n
            } else {
                (i + j - m + 1) % n
            };
            sum += signal[idx] * f;
        }
        result.push(sum);
    }

    result
}

/// Upsamples by 2 and convolves with filter.
fn upsample_convolve(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    if signal.is_empty() || filter.is_empty() {
        return Vec::new();
    }

    // Upsample by inserting zeros
    let upsampled_len = signal.len() * 2;
    let mut upsampled = vec![0.0; upsampled_len];

    for (i, &val) in signal.iter().enumerate() {
        upsampled[i * 2] = val;
    }

    // Convolve with filter
    convolve(&upsampled, filter)
}

/// Simple convolution.
fn convolve(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    if signal.is_empty() || filter.is_empty() {
        return Vec::new();
    }

    let n = signal.len();
    let m = filter.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for (j, &f) in filter.iter().enumerate() {
            let idx = (i + n - j) % n;
            sum += signal[idx] * f;
        }
        result[i] = sum;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is2pow() {
        assert!(is2pow(1));
        assert!(is2pow(2));
        assert!(is2pow(4));
        assert!(is2pow(8));
        assert!(is2pow(16));
        assert!(is2pow(1024));

        assert!(!is2pow(0));
        assert!(!is2pow(3));
        assert!(!is2pow(5));
        assert!(!is2pow(6));
        assert!(!is2pow(7));
    }

    #[test]
    fn test_haar_filters() {
        let (lo, hi) = haar_filters();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        assert!((lo[0] - inv_sqrt2).abs() < 1e-10);
        assert!((lo[1] - inv_sqrt2).abs() < 1e-10);
        assert!((hi[0] - inv_sqrt2).abs() < 1e-10);
        assert!((hi[1] - (-inv_sqrt2)).abs() < 1e-10);
    }

    #[test]
    fn test_dwt_haar() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let dwt = DiscreteWaveletTransform::new(WaveletFamily::Haar);

        let (approx, detail) = dwt.decompose(&signal);

        assert_eq!(approx.len(), 2);
        assert_eq!(detail.len(), 2);
    }

    #[test]
    fn test_dwt_reconstruction() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dwt = DiscreteWaveletTransform::new(WaveletFamily::Haar);

        let (approx, detail) = dwt.decompose(&signal);
        let reconstructed = dwt.reconstruct(&approx, &detail);

        // Reconstructed should be close to original
        assert_eq!(reconstructed.len(), signal.len());

        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multilevel_dwt() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dwt = DiscreteWaveletTransform::new(WaveletFamily::Haar);

        let (approx, details) = dwt.wavedec(&signal, 2);

        assert_eq!(details.len(), 2);
        assert_eq!(approx.len(), 2); // After 2 levels, length is 8/2/2 = 2
    }

    #[test]
    fn test_multilevel_reconstruction() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dwt = DiscreteWaveletTransform::new(WaveletFamily::Haar);

        let (approx, details) = dwt.wavedec(&signal, 2);
        let reconstructed = dwt.waverec(&approx, &details);

        // Reconstructed should be close to original
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dwt_function() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let (approx, details) = dwt(&signal, WaveletFamily::Haar, 1);

        assert_eq!(details.len(), 1);
        assert_eq!(approx.len(), 2);
    }

    #[test]
    fn test_daubechies_filters() {
        let (lo, hi) = daubechies_filters(4);

        // db4 should have 4 coefficients
        assert_eq!(lo.len(), 4);
        assert_eq!(hi.len(), 4);

        // Filters should be normalized
        let lo_norm: f64 = lo.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((lo_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wavelet_family() {
        let haar = WaveletFamily::Haar;
        let db4 = WaveletFamily::Daubechies(4);

        let (haar_lo, _) = haar.get_filters();
        let (db4_lo, _) = db4.get_filters();

        assert_eq!(haar_lo.len(), 2);
        assert_eq!(db4_lo.len(), 4);
    }
}
