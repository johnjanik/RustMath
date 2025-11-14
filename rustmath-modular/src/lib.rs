//! # RustMath Modular Forms
//!
//! This crate provides functionality for modular forms and related structures,
//! corresponding to SageMath's sage.modular module.
//!
//! ## Modules
//!
//! - `arithgroup`: Arithmetic subgroups of SL(2, Z) (Gamma0, Gamma1, GammaH, SL2Z)
//! - `modform`: Modular forms and cusp forms
//! - `modsym`: Modular symbols
//! - `hecke`: Hecke operators and Hecke modules
//! - `abvar`: Modular abelian varieties
//! - `cusps`: Cusps of modular curves

pub mod arithgroup;
pub mod modform;
pub mod modsym;
pub mod hecke;
pub mod abvar;
pub mod cusps;

// Re-export commonly used types
pub use arithgroup::{
    ArithmeticSubgroup, SL2Z, Gamma0, Gamma1, GammaH,
    ArithmeticSubgroupElement, CongruenceSubgroup
};
pub use modform::{ModularForm, CuspForm};
pub use cusps::Cusp;

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{Zero, One};

    #[test]
    fn test_basic_modular_imports() {
        // Just verify that the modules exist
        let _sl2z = SL2Z::new();
        let _gamma0 = Gamma0::new(2);
    }
}
