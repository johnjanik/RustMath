//! p-adic numbers Qp and Zp
//!
//! Provides arithmetic in p-adic number fields and rings.
//! - Qp: p-adic field (field of fractions of Zp)
//! - Zp: p-adic integers (ring)

pub mod padic_integer;
pub mod padic_rational;

pub use padic_integer::PadicInteger;
pub use padic_rational::PadicRational;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn basic_padic() {
        let p = Integer::from(5);
        let precision = 10;

        let a = PadicInteger::from_integer(Integer::from(7), p.clone(), precision).unwrap();

        assert_eq!(a.residue(), &Integer::from(7));
    }
}
