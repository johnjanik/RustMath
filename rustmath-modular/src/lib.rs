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
//! - `cusps_nf`: Cusps for number fields
//! - `dims`: Dimensions of spaces of modular forms
//! - `dirichlet`: Dirichlet characters
//! - `etaproducts`: Eta products and eta quotients
//! - `btquotients`: Bruhat-Tits tree quotients
//! - `drinfeld_modform`: Drinfeld modular forms
//! - `local_comp`: Local components of modular forms
//! - `modform_hecketriangle`: Modular forms for Hecke triangle groups
//! - `overconvergent`: Overconvergent modular forms
//! - `pollack_stevens`: Pollack-Stevens modular symbols
//! - `quasimodform`: Quasi-modular forms
//! - `quatalg`: Quaternion algebras
//! - `ssmod`: Supersingular modules
//! - `buzzard`: Buzzard's algorithm
//! - `hypergeometric_motive`: Hypergeometric motives
//! - `multiple_zeta`: Multiple zeta values

pub mod arithgroup;
pub mod modform;
pub mod modsym;
pub mod hecke;
pub mod abvar;
pub mod cusps;
pub mod cusps_nf;
pub mod dims;
pub mod dirichlet;
pub mod etaproducts;
pub mod btquotients;
pub mod drinfeld_modform;
pub mod local_comp;
pub mod modform_hecketriangle;
pub mod overconvergent;
pub mod pollack_stevens;
pub mod quasimodform;
pub mod quatalg;
pub mod ssmod;
pub mod buzzard;
pub mod hypergeometric_motive;
pub mod multiple_zeta;

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
