//! Complex numbers with real and imaginary parts
//!
//! Provides complex number arithmetic over various real number types.

pub mod complex;

pub use complex::Complex;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_complex() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.real() - 3.0).abs() < 1e-10);
        assert!((z.imag() - 4.0).abs() < 1e-10);
        assert!((z.abs() - 5.0).abs() < 1e-10);
    }
}
