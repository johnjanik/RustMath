//! RustMath Lie Algebras - Lie algebras, root systems, and Weyl groups
//!
//! This crate provides infrastructure for working with:
//! - Cartan types (classification of root systems)
//! - Root systems (configurations of roots in Euclidean space)
//! - Weyl groups (reflection groups associated with root systems)
//! - Lie algebras (coming soon)
//!
//! Corresponds to sage.algebras.lie_algebras and sage.combinat.root_system

pub mod cartan_type;
pub mod root_system;
pub mod weyl_group;

pub use cartan_type::{CartanLetter, CartanType};
pub use root_system::{Root, RootSystem};
pub use weyl_group::{WeylGroup, WeylGroupElement};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        // Create a Cartan type
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();

        // Create the root system
        let rs = RootSystem::new(ct);
        assert_eq!(rs.rank(), 2);

        // Create the Weyl group
        let W = WeylGroup::new(ct);
        assert_eq!(W.order(), 6); // |S_3| = 6
    }
}
