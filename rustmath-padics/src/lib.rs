//! p-adic numbers qp and zp — compatibility shim
//!
//! The p-adic machinery now lives in `rustmath_rings::padics` (its canonical
//! home, alongside capped-relative elements, extensions, and the factory
//! functions `zp`/`qp`/`zq`/`qq`). This crate is kept as a thin re-export
//! shim so existing dependents (e.g. `rustmath-schemes`) keep working
//! unchanged.
//!
//! - qp: p-adic field (field of fractions of zp)
//! - zp: p-adic integers (ring)

pub use rustmath_rings::padics::padic_integer;
pub use rustmath_rings::padics::padic_rational;

pub use rustmath_rings::padics::{hensel_lift_root, PadicInteger, PadicRational};

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn basic_padic() {
        let p = Integer::from(5);
        let precision = 10;

        let a = PadicInteger::from_integer(Integer::from(7), p.clone(), precision).unwrap();

        // value() is 7 mod p^precision = 7 (since p^precision = 5^10 >> 7)
        assert_eq!(a.value(), &Integer::from(7));
        // residue() is the image in the residue field Z/pZ: 7 mod 5 = 2
        assert_eq!(a.residue(), Integer::from(2));
    }
}
