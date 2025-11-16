//! Transform modules
//!
//! This module provides various signal transforms including:
//! - DFT (Discrete Fourier Transform)
//! - FFT (Fast Fourier Transform)
//! - DWT (Discrete Wavelet Transform)

pub mod dft;
pub mod dwt;
pub mod fft;

// Re-export commonly used items
pub use dft::{dft, dft_real, idft, power_spectrum, IndexedSequence};
pub use dwt::{
    dwt as discrete_wavelet_transform, is2pow, wavelet_transform, DiscreteWaveletTransform,
    WaveletFamily,
};
pub use fft::{
    fft as fast_fourier_transform, ifft, rfft, irfft, FastFourierTransformBase,
    FastFourierTransformComplex, FourierTransformComplex, FourierTransformReal,
};
