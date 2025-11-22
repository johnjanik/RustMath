//! Examples of Groups - Catalog Interface
//!
//! The `groups_catalog` module provides access to examples of various groups.
//! This catalog is organized into several categories for easy discovery:
//!
//! - **Matrix Groups** (`matrix` module): Linear, special linear, orthogonal, symplectic, and unitary groups
//! - **Permutation Groups** (`permutation` module): Symmetric, alternating, cyclic, dihedral, and other permutation groups
//! - **Finitely Presented Groups** (`presentation` module): Groups defined by generators and relations
//! - **Affine Groups** (`affine` module): Affine transformations and Euclidean groups
//! - **Lie Groups** (`lie` module): Nilpotent Lie groups and related structures
//! - **Miscellaneous Groups** (`misc` module): Braid groups, Coxeter groups, free groups, and other specialized groups
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::groups_catalog;
//!
//! // Access matrix groups
//! let gl3 = groups_catalog::matrix::GLn(3);
//! let sl2 = groups_catalog::matrix::SLn(2);
//!
//! // Access permutation groups
//! let s5 = groups_catalog::permutation::symmetric(5);
//! let a4 = groups_catalog::permutation::alternating(4);
//!
//! // Access finitely presented groups
//! let cyclic = groups_catalog::presentation::cyclic(10);
//! let dihedral = groups_catalog::presentation::dihedral(8);
//!
//! // Access miscellaneous groups
//! let free = groups_catalog::misc::free_group(3);
//! let braid = groups_catalog::misc::braid_group(4);
//! ```
//!
//! # Implementation Notes
//!
//! This module serves as a catalog interface, providing organized access to group
//! constructors from various specialized modules. It mirrors SageMath's groups catalog
//! structure for consistency with mathematical conventions.

/// Matrix Groups Catalog
///
/// Provides access to matrix groups including:
/// - GL(n): General Linear Group
/// - SL(n): Special Linear Group
/// - O(n): Orthogonal Group
/// - SO(n): Special Orthogonal Group
/// - U(n): Unitary Group
/// - SU(n): Special Unitary Group
/// - Sp(2n): Symplectic Group
pub mod matrix {
    pub use crate::matrix_group::{MatrixGroup, GLn, SLn};

    // TODO: These functions need to be generic over the field type F
    // GL(n): General Linear Group
    // pub fn GL<F: Field>(n: usize) -> crate::matrix_group::GLn<F> {
    //     crate::matrix_group::GLn::new(n)
    // }

    // TODO: These functions need to be generic over the field type F
    // SL(n): Special Linear Group
    // pub fn SL<F: Field>(n: usize) -> crate::matrix_group::SLn<F> {
    //     crate::matrix_group::SLn::new(n)
    // }
}

/// Permutation Groups Catalog
///
/// Provides access to permutation groups including:
/// - Symmetric groups S_n
/// - Alternating groups A_n
/// - Cyclic groups C_n
/// - Dihedral groups D_n
/// - Klein Four group
/// - Quaternion group
/// - Mathieu groups
/// - And other named permutation groups
pub mod permutation {
    pub use crate::permutation_group::{PermutationGroup, SymmetricGroup, AlternatingGroup};

    /// Creates a symmetric group S_n
    ///
    /// # Arguments
    ///
    /// * `n` - The degree of the symmetric group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let s5 = permutation::Symmetric(5);
    /// ```
    pub fn Symmetric(n: usize) -> SymmetricGroup {
        SymmetricGroup::new(n)
    }

    /// Creates an alternating group A_n
    ///
    /// # Arguments
    ///
    /// * `n` - The degree of the alternating group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let a4 = permutation::Alternating(4);
    /// ```
    pub fn Alternating(n: usize) -> AlternatingGroup {
        AlternatingGroup::new(n)
    }

    /// Creates a cyclic permutation group
    ///
    /// # Arguments
    ///
    /// * `n` - The order of the cyclic group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let c6 = permutation::Cyclic(6);
    /// ```
    pub fn Cyclic(n: usize) -> PermutationGroup {
        // Cyclic group as permutations (0 1 2 ... n-1)
        PermutationGroup::cyclic(n)
    }

    /// Creates the Klein Four group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let v4 = permutation::KleinFour();
    /// ```
    pub fn KleinFour() -> crate::abelian_group::AbelianGroup {
        crate::misc_groups::klein_four_group()
    }

    /// Creates a quaternion group as a permutation group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let q8 = permutation::Quaternion();
    /// ```
    pub fn Quaternion() -> crate::abelian_group::AbelianGroup {
        crate::misc_groups::quaternion_group()
    }

    /// Creates a dihedral group D_n
    ///
    /// # Arguments
    ///
    /// * `n` - The dihedral parameter (group has order 2n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let d4 = permutation::Dihedral(4);
    /// ```
    // TODO: GroupSemidirectProduct needs 3 generic parameters
    // pub fn Dihedral(n: usize) -> crate::semidirect_product::GroupSemidirectProduct {
    //     crate::semidirect_product::dihedral_group(n)
    // }

    /// Creates a dicyclic group
    ///
    /// # Arguments
    ///
    /// * `n` - The dicyclic parameter
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let dic3 = permutation::DiCyclic(3);
    /// ```
    pub fn DiCyclic(n: usize) -> crate::abelian_group::AbelianGroup {
        crate::misc_groups::dicyclic_group(n)
    }
}

/// Finitely Presented Groups Catalog
///
/// Provides access to groups defined by generators and relations
pub mod presentation {
    pub use crate::finitely_presented::{FinitelyPresentedGroup, FinitelyPresentedGroupElement};
    pub use crate::finitely_presented_named::*;

    /// Creates a cyclic group presentation
    ///
    /// # Arguments
    ///
    /// * `n` - The order of the cyclic group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let c10 = presentation::Cyclic(10);
    /// ```
    pub fn Cyclic(n: usize) -> FinitelyPresentedGroup {
        cyclic_presentation(n)
    }

    /// Creates a dihedral group presentation
    ///
    /// # Arguments
    ///
    /// * `n` - The dihedral parameter (group has order 2n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let d8 = presentation::Dihedral(8);
    /// ```
    pub fn Dihedral(n: usize) -> FinitelyPresentedGroup {
        dihedral_presentation(n)
    }

    /// Creates a dicyclic group presentation
    ///
    /// # Arguments
    ///
    /// * `n` - The dicyclic parameter
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let dic4 = presentation::DiCyclic(4);
    /// ```
    pub fn DiCyclic(n: usize) -> FinitelyPresentedGroup {
        dicyclic_presentation(n)
    }

    /// Creates a quaternion group presentation
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let q = presentation::Quaternion();
    /// ```
    pub fn Quaternion() -> FinitelyPresentedGroup {
        quaternion_presentation()
    }

    /// Creates the Klein Four group presentation
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let v4 = presentation::KleinFour();
    /// ```
    pub fn KleinFour() -> FinitelyPresentedGroup {
        klein_four_presentation()
    }

    /// Creates a finitely generated abelian group presentation
    ///
    /// # Arguments
    ///
    /// * `invariants` - The invariant factors
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let g = presentation::FGAbelian(vec![2, 3, 5]);
    /// ```
    pub fn FGAbelian(invariants: Vec<usize>) -> FinitelyPresentedGroup {
        finitely_generated_abelian_presentation(invariants)
    }

    /// Creates a symmetric group presentation
    ///
    /// # Arguments
    ///
    /// * `n` - The degree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let s5 = presentation::Symmetric(5);
    /// ```
    pub fn Symmetric(n: usize) -> FinitelyPresentedGroup {
        symmetric_presentation(n)
    }

    /// Creates an alternating group presentation
    ///
    /// # Arguments
    ///
    /// * `n` - The degree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let a5 = presentation::Alternating(5);
    /// ```
    pub fn Alternating(n: usize) -> FinitelyPresentedGroup {
        alternating_presentation(n)
    }

    /// Creates a cactus group presentation
    ///
    /// # Arguments
    ///
    /// * `n` - The number of strands
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::presentation;
    ///
    /// let j3 = presentation::Cactus(3);
    /// ```
    pub fn Cactus(n: usize) -> FinitelyPresentedGroup {
        cactus_presentation(n)
    }
}

/// Affine Groups Catalog
///
/// Provides access to affine transformation groups and Euclidean groups
pub mod affine {
    pub use crate::affine_group::{AffineGroup, AffineGroupElement};
    pub use crate::euclidean_group::EuclideanGroup;

    /// Creates an affine group
    ///
    /// # Arguments
    ///
    /// * `n` - The dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::affine;
    ///
    /// let aff3 = affine::Affine(3);
    //
    // TODO: AffineGroup needs generic parameter R: Ring
    // pub fn Affine<R: Ring>(n: usize) -> AffineGroup<R> {
    //     AffineGroup::new(n)
    // }

    // TODO: EuclideanGroup needs generic parameter R: Ring
    // pub fn Euclidean<R: Ring>(n: usize) -> EuclideanGroup<R> {
    //     EuclideanGroup::new(n)
    // }
}

/// Lie Groups Catalog
///
/// Provides access to Lie groups including nilpotent Lie groups
pub mod lie {
    pub use crate::nilpotent_lie_group::{NilpotentLieGroup, NilpotentLieGroupElement};

    /// Creates a nilpotent Lie group
    ///
    /// # Arguments
    ///
    /// * `name` - The name/identifier of the Lie group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::lie;
    ///
    /// let h3 = lie::Nilpotent("Heisenberg3");
    /// ```
    pub fn Nilpotent(name: &str) -> NilpotentLieGroup {
        NilpotentLieGroup::new(name.to_string())
    }
}

/// Miscellaneous Groups Catalog
///
/// Provides access to various specialized groups including:
/// - Braid groups
/// - Cactus groups
/// - Free groups
/// - Right-angled Artin groups
/// - Semimonomial transformation groups
/// - Additive abelian groups
pub mod misc {
    pub use crate::braid::{BraidGroup, braid_group};
    pub use crate::cactus_group::{CactusGroup, PureCactusGroup};
    pub use crate::free_group::FreeGroup;
    pub use crate::raag::RightAngledArtinGroup;
    pub use crate::artin::ArtinGroup;
    pub use crate::additive_abelian_group::{AdditiveAbelianGroup, additive_abelian_group};
    pub use crate::semimonomial_transformation_group::SemimonomialTransformationGroup;

    /// Creates a braid group
    ///
    /// # Arguments
    ///
    /// * `n` - The number of strands
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let b4 = misc::Braid(4);
    /// ```
    pub fn Braid(n: usize) -> BraidGroup {
        braid_group(n)
    }

    /// Creates a cactus group
    ///
    /// # Arguments
    ///
    /// * `n` - The number of strands
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let j3 = misc::Cactus(3);
    /// ```
    pub fn Cactus(n: usize) -> CactusGroup {
        CactusGroup::new(n)
    }

    /// Creates a pure cactus group
    ///
    /// # Arguments
    ///
    /// * `n` - The number of strands
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let pj3 = misc::PureCactus(3);
    /// ```
    pub fn PureCactus(n: usize) -> PureCactusGroup {
        PureCactusGroup::new(n)
    }

    /// Creates a free group
    ///
    /// # Arguments
    ///
    /// * `n` - The number of generators
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let f3 = misc::Free(3);
    /// ```
    pub fn Free(n: usize) -> FreeGroup {
        FreeGroup::new(n)
    }

    /// Creates a right-angled Artin group
    ///
    /// # Arguments
    ///
    /// * `graph` - The defining graph (as adjacency list or similar)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let graph = vec![vec![1], vec![0]]; // Two vertices with edge between them
    /// let raag = misc::RightAngledArtin(graph);
    /// ```
    pub fn RightAngledArtin(graph: Vec<Vec<usize>>) -> RightAngledArtinGroup {
        RightAngledArtinGroup::from_graph(graph)
    }

    /// Creates an Artin group
    ///
    /// # Arguments
    ///
    /// * `coxeter_matrix` - The Coxeter matrix defining the group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let matrix = vec![vec![1, 3], vec![3, 1]]; // A_2 type
    /// let artin = misc::CoxeterGroup(matrix);
    /// ```
    pub fn CoxeterGroup(coxeter_matrix: Vec<Vec<usize>>) -> ArtinGroup {
        ArtinGroup::from_coxeter_matrix(coxeter_matrix)
    }

    /// Creates an additive abelian group
    ///
    /// # Arguments
    ///
    /// * `invariants` - The invariant factors
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let g = misc::AdditiveAbelian(vec![2, 3, 5]);
    /// ```
    pub fn AdditiveAbelian(invariants: Vec<i32>) -> AdditiveAbelianGroup {
        additive_abelian_group(invariants)
    }

    /// Creates a cyclic additive group (integers mod n)
    ///
    /// # Arguments
    ///
    /// * `n` - The modulus
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let z12 = misc::AdditiveCyclic(12);
    /// ```
    pub fn AdditiveCyclic(n: i32) -> AdditiveAbelianGroup {
        additive_abelian_group(vec![n])
    }

    /// Creates a semimonomial transformation group
    ///
    /// # Arguments
    ///
    /// * `n` - The dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let g = misc::SemimonomialTransformation(3);
    /// ```
    // TODO: SemimonomialTransformationGroup needs generic parameters
    // pub fn SemimonomialTransformation(n: usize) -> SemimonomialTransformationGroup {
    //     SemimonomialTransformationGroup::new(n)
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_catalog() {
        let gl3 = matrix::GL(3);
        assert_eq!(gl3.degree(), 3);

        let sl2 = matrix::SL(2);
        assert_eq!(sl2.degree(), 2);
    }

    #[test]
    fn test_permutation_catalog() {
        let s5 = permutation::Symmetric(5);
        assert_eq!(s5.degree(), 5);

        let a4 = permutation::Alternating(4);
        assert_eq!(a4.degree(), 4);
    }

    #[test]
    fn test_presentation_catalog() {
        let c10 = presentation::Cyclic(10);
        assert!(c10.num_generators() >= 1);

        let d8 = presentation::Dihedral(8);
        assert!(d8.num_generators() >= 2);

        let q = presentation::Quaternion();
        assert!(q.num_generators() >= 2);
    }

    #[test]
    fn test_affine_catalog() {
        let aff3 = affine::Affine(3);
        assert_eq!(aff3.dimension(), 3);

        let e2 = affine::Euclidean(2);
        assert_eq!(e2.dimension(), 2);
    }

    #[test]
    fn test_lie_catalog() {
        let h3 = lie::Nilpotent("Heisenberg3");
        assert_eq!(h3.name(), "Heisenberg3");
    }

    #[test]
    fn test_misc_catalog() {
        let b4 = misc::Braid(4);
        assert_eq!(b4.num_strands(), 4);

        let f3 = misc::Free(3);
        assert_eq!(f3.rank(), 3);

        let j3 = misc::Cactus(3);
        assert_eq!(j3.num_strands(), 3);

        let pj3 = misc::PureCactus(3);
        assert_eq!(pj3.num_strands(), 3);

        let z12 = misc::AdditiveCyclic(12);
        assert_eq!(z12.order(), Some(12));
    }
}
