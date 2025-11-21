//! p-adic numbers qp and zp
//!
//! Provides arithmetic in p-adic number fields and rings.
//! - qp: p-adic field (field of fractions of zp)
//! - zp: p-adic integers (ring)

pub mod padic_integer;
pub mod padic_rational;

pub use padic_integer::{hensel_lift_root, PadicInteger};
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
