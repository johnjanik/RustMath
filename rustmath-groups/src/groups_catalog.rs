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
//! use rustmath_integers::Integer;
//! use rustmath_rationals::Rational;
//!
//! // Matrix groups (generic over the base field)
//! let gl3 = groups_catalog::matrix::GLn::<Rational>::new(3);
//! assert_eq!(gl3.dimension(), 3);
//! let sl2 = groups_catalog::matrix::SLn::<Rational>::new(2);
//! assert_eq!(sl2.dimension(), 2);
//!
//! // Permutation groups: orders computed by Schreier-Sims from the generators
//! let s5 = groups_catalog::permutation::Symmetric(5);
//! assert_eq!(s5.group().order(), Integer::from(120));
//! let a4 = groups_catalog::permutation::Alternating(4);
//! assert_eq!(a4.group().order(), Integer::from(12));
//! let d4 = groups_catalog::permutation::Dihedral(4);
//! assert_eq!(d4.order(), Integer::from(8));
//!
//! // Finitely presented groups
//! let c10 = groups_catalog::presentation::Cyclic(10);
//! assert_eq!(c10.relations(), &[vec![1; 10]]); // <a | a^10>
//! let d8 = groups_catalog::presentation::Dihedral(8);
//! assert_eq!(d8.num_generators(), 2);
//!
//! // Miscellaneous groups
//! let free = groups_catalog::misc::Free(3);
//! assert_eq!(free.rank(), 3);
//! let braid = groups_catalog::misc::Braid(4);
//! assert_eq!(braid.strands(), 4);
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
    use crate::FinitelyPresentedGroup;

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

    /// Creates the Klein Four group (as a finitely presented group)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let v4 = permutation::KleinFour();
    /// ```
    pub fn KleinFour() -> FinitelyPresentedGroup {
        crate::misc_groups::klein_four_group()
    }

    /// Creates the quaternion group Q_8 (as a finitely presented group)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let q8 = permutation::Quaternion();
    /// ```
    pub fn Quaternion() -> FinitelyPresentedGroup {
        crate::misc_groups::quaternion_group()
    }

    /// Creates the dihedral group D_n of order 2n as a permutation group
    ///
    /// Same convention as GAP `DihedralGroup(IsPermGroup, 2n)` and Sage
    /// `groups.permutation.Dihedral(n)`; see [`PermutationGroup::dihedral`].
    ///
    /// # Arguments
    ///
    /// * `n` - The dihedral parameter (group has order 2n)
    ///
    /// # Panics
    ///
    /// If `n == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    /// use rustmath_integers::Integer;
    ///
    /// let d4 = permutation::Dihedral(4); // symmetries of the square
    /// assert_eq!(d4.degree(), 4);
    /// assert_eq!(d4.order(), Integer::from(8));
    /// ```
    pub fn Dihedral(n: usize) -> PermutationGroup {
        PermutationGroup::dihedral(n)
    }

    /// Creates the dicyclic group Dic_n of order 4n (as a finitely presented
    /// group ⟨a, x | a^(2n), x² a^(−n), x⁻¹ a x a⟩)
    ///
    /// # Arguments
    ///
    /// * `n` - The dicyclic parameter (n ≥ 2; the group has order 4n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::permutation;
    ///
    /// let dic3 = permutation::DiCyclic(3); // order 12
    /// assert_eq!(dic3.num_generators(), 2);
    /// assert_eq!(dic3.relations()[0], vec![1; 6]); // a^6 = 1
    /// ```
    pub fn DiCyclic(n: usize) -> FinitelyPresentedGroup {
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
        finitely_generated_abelian_presentation(&invariants)
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

    // TODO: Creates an affine group
    // Arguments:
    // * `n` - The dimension
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

    /// Creates a named nilpotent Lie group
    ///
    /// # Arguments
    ///
    /// * `name` - `"Heisenberg"` (the 3-dimensional Heisenberg group) or
    ///   `"Heisenberg<d>"` for odd `d ≥ 3` (the Heisenberg group H_d of
    ///   dimension d = 2n + 1, see [`NilpotentLieGroup::heisenberg`]).
    ///
    /// # Panics
    ///
    /// For any other name: no other named nilpotent Lie group is implemented,
    /// and a different group is never substituted.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::lie;
    ///
    /// let h3 = lie::Nilpotent("Heisenberg3");
    /// assert_eq!(h3.dimension(), 3);
    /// assert_eq!(h3.nilpotency_class(), 2);
    /// assert_eq!(lie::Nilpotent("Heisenberg5").dimension(), 5);
    /// ```
    pub fn Nilpotent(name: &str) -> NilpotentLieGroup {
        if let Some(rest) = name.strip_prefix("Heisenberg") {
            if rest.is_empty() {
                return NilpotentLieGroup::heisenberg(1);
            }
            if let Ok(d) = rest.parse::<usize>() {
                if d >= 3 && d % 2 == 1 {
                    return NilpotentLieGroup::heisenberg((d - 1) / 2);
                }
            }
        }
        unimplemented!(
            "lie::Nilpotent({:?}): only \"Heisenberg\" and \"Heisenberg<d>\" (odd d >= 3) are implemented",
            name
        )
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
    pub use crate::artin::{ArtinGroup, CoxeterMatrix};
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

    // Creates a pure cactus group
    //
    // Arguments:
    // * `n` - The number of strands
    //
    // TODO: PureCactusGroup not yet implemented
    // pub fn PureCactus(n: usize) -> PureCactusGroup {
    //     PureCactusGroup::new(n)
    // }

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
        let names: Vec<String> = (0..n).map(|i| format!("x{}", i)).collect();
        FreeGroup::new(n, names)
    }

    /// Creates a right-angled Artin group
    ///
    /// Generators `g_i` correspond to the vertices of the defining graph, and
    /// `g_i`, `g_j` commute exactly when `{i, j}` is an edge.
    ///
    /// # Arguments
    ///
    /// * `adjacency` - The defining simple graph as an adjacency list:
    ///   `adjacency[i]` lists the neighbours of vertex `i`. The graph is
    ///   undirected, so an edge may be listed from either endpoint or both.
    ///
    /// # Panics
    ///
    /// If a neighbour index is out of range or a vertex lists itself (the
    /// defining graph has no loops).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let graph = vec![vec![1], vec![0]]; // Two vertices with edge between them
    /// let raag = misc::RightAngledArtin(graph); // the free abelian group Z^2
    /// assert_eq!(raag.ngens(), 2);
    /// assert!(raag.commute(0, 1));
    /// let (a, b) = (raag.gen(0), raag.gen(1));
    /// assert_eq!(a.clone() * b.clone(), b * a);
    /// ```
    pub fn RightAngledArtin(adjacency: Vec<Vec<usize>>) -> RightAngledArtinGroup {
        let n = adjacency.len();
        let mut graph = rustmath_graphs::Graph::new(n);
        for (u, neighbours) in adjacency.iter().enumerate() {
            for &v in neighbours {
                assert!(v != u, "RAAG defining graph must not have a loop (vertex {})", u);
                assert!(v < n, "RAAG adjacency list: neighbour {} of vertex {} is out of range", v, u);
                graph.add_edge(u, v).expect("both endpoints are in range");
            }
        }
        RightAngledArtinGroup::new(graph)
    }

    /// Creates the Artin group of a Coxeter matrix
    ///
    /// The group ⟨s_0, …, s_{n−1} | s_i s_j s_i ⋯ = s_j s_i s_j ⋯ (m_ij factors
    /// each side)⟩, as in Sage `groups.misc.Artin`. (The Coxeter group of the
    /// same matrix is the quotient by s_i² = 1; it is not constructed here.)
    ///
    /// # Arguments
    ///
    /// * `coxeter_matrix` - Symmetric matrix with `m_ii = 1` and `m_ij ≥ 2`,
    ///   or `0` for m_ij = ∞ (no relation)
    ///
    /// # Panics
    ///
    /// If the matrix is not a valid Coxeter matrix (see [`CoxeterMatrix::new`]):
    /// empty, not square, not symmetric, a diagonal entry other than 1, or an
    /// off-diagonal entry m_ij = 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// let matrix = vec![vec![1, 3], vec![3, 1]]; // A_2 type: the braid group B_3
    /// let artin = misc::Artin(matrix);
    /// assert_eq!(artin.rank(), 2);
    /// // the single relation s0 s1 s0 = s1 s0 s1, as the Tietze word s0 s1 s0 s1^-1 s0^-1 s1^-1
    /// assert_eq!(
    ///     artin.as_finitely_presented_group().relations(),
    ///     &[vec![1, 2, 1, -2, -1, -2]]
    /// );
    /// ```
    pub fn Artin(coxeter_matrix: Vec<Vec<usize>>) -> ArtinGroup {
        ArtinGroup::new(CoxeterMatrix::new(coxeter_matrix))
    }

    /// Creates the additive abelian group Z/m_1 ⊕ … ⊕ Z/m_k
    ///
    /// The orders need not form a divisibility chain (as for MAGMA
    /// `AbelianGroup` and Sage `AdditiveAbelianGroup`): the group is returned
    /// in invariant-factor form, see
    /// [`AdditiveAbelianGroup::from_cyclic_orders`]. An order 0 is an infinite
    /// cyclic factor Z; a negative order m is read as |m| (Z/mZ = Z/(−m)Z).
    ///
    /// # Arguments
    ///
    /// * `invariants` - The orders m_i of the cyclic factors
    ///
    /// # Panics
    ///
    /// If an invariant factor (an lcm of the orders) overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::groups_catalog::misc;
    ///
    /// // Z/2 + Z/3 + Z/5 is cyclic of order 30
    /// let g = misc::AdditiveAbelian(vec![2, 3, 5]);
    /// assert_eq!(g.invariant_factors(), &[30]);
    /// assert_eq!(g.order(), Some(30));
    /// assert!(g.is_cyclic());
    /// ```
    pub fn AdditiveAbelian(invariants: Vec<i32>) -> AdditiveAbelianGroup {
        let orders: Vec<usize> = invariants.iter().map(|&x| x.unsigned_abs() as usize).collect();
        AdditiveAbelianGroup::from_cyclic_orders(&orders)
            .expect("invariant factor of the additive abelian group overflows usize")
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
        additive_abelian_group(vec![n.abs() as usize]).unwrap()
    }

    // TODO: Creates a semimonomial transformation group
    // Arguments:
    // * `n` - The dimension
    // Examples:
    // use rustmath_groups::groups_catalog::misc;
    // let g = misc::SemimonomialTransformation(3);
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
        // The catalog GL/SL helpers are field-generic and not yet exposed as
        // non-generic constructors, so build the re-exported types directly
        // over the rationals.
        use rustmath_rationals::Rational;

        let gl3 = matrix::GLn::<Rational>::new(3);
        assert_eq!(gl3.dimension(), 3);

        let sl2 = matrix::SLn::<Rational>::new(2);
        assert_eq!(sl2.dimension(), 2);
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
        // The catalog Affine/Euclidean helpers are ring-generic and not yet
        // exposed as non-generic constructors, so build the re-exported types
        // directly over the integers.
        use rustmath_integers::Integer;

        let aff3 = affine::AffineGroup::<Integer>::new(3);
        assert_eq!(aff3.dimension(), 3);

        let e2 = affine::EuclideanGroup::<Integer>::new(2);
        assert_eq!(e2.dimension(), 2);
    }

    #[test]
    fn test_lie_catalog() {
        // "Heisenberg3" is the 3-dimensional Heisenberg group (2-step nilpotent);
        // unknown names are refused (test_lie_unknown_name_is_refused).
        let h3 = lie::Nilpotent("Heisenberg3");
        assert_eq!(h3.dimension(), 3);
        assert_eq!(h3.nilpotency_class(), 2);
    }

    #[test]
    fn test_misc_catalog() {
        let b4 = misc::Braid(4);
        assert_eq!(b4.strands(), 4);

        let f3 = misc::Free(3);
        assert_eq!(f3.rank(), 3);

        let j3 = misc::Cactus(3);
        assert_eq!(j3.num_fruits(), 3);

        // Note: the catalog `misc::PureCactus` constructor is still commented
        // out (it wraps a KernelSubgroup type alias), so it is not exercised.

        let z12 = misc::AdditiveCyclic(12);
        assert_eq!(z12.order(), Some(12));
    }

    #[test]
    fn test_permutation_dihedral_catalog() {
        use rustmath_integers::Integer;
        // GAP: Size(DihedralGroup(IsPermGroup, 2n)) = 2n.
        for n in 1..=6usize {
            assert_eq!(permutation::Dihedral(n).order(), Integer::from(2 * n as u64));
        }
    }

    #[test]
    fn test_misc_raag_catalog() {
        // Path 0 - 1 - 2, the edge 1-2 listed from one endpoint only.
        let raag = misc::RightAngledArtin(vec![vec![1], vec![0, 2], vec![]]);
        assert_eq!(raag.ngens(), 3);
        assert!(raag.commute(0, 1));
        assert!(raag.commute(2, 1));
        assert!(!raag.commute(0, 2));
    }

    #[test]
    #[should_panic(expected = "must not have a loop")]
    fn test_misc_raag_rejects_loops() {
        misc::RightAngledArtin(vec![vec![0]]);
    }

    #[test]
    fn test_misc_artin_catalog() {
        // B_2 Coxeter matrix (m_01 = 4): one relation of length 8.
        let artin = misc::Artin(vec![vec![1, 4], vec![4, 1]]);
        assert_eq!(artin.rank(), 2);
        assert_eq!(
            artin.as_finitely_presented_group().relations(),
            &[vec![1, 2, 1, 2, -1, -2, -1, -2]]
        );
    }

    #[test]
    #[should_panic(expected = "Off-diagonal Coxeter matrix entries must be >= 2 or 0")]
    fn test_misc_artin_rejects_off_diagonal_one() {
        // m_01 = 1 is not a Coxeter matrix entry: refused, not silently read as
        // "no relation" (the free group of m_01 = 0 = infinity).
        misc::Artin(vec![vec![1, 1], vec![1, 1]]);
    }

    #[test]
    fn test_misc_additive_abelian_catalog() {
        // Z/2 + Z/4 + Z/6 + Z/9 = Z/2 + Z/6 + Z/36 (GAP AbelianInvariants
        // [2, 2, 3, 4, 9], Size 432); an order 0 becomes free rank.
        let g = misc::AdditiveAbelian(vec![2, 4, 6, 9]);
        assert_eq!(g.invariant_factors(), &[2, 6, 36]);
        assert_eq!(g.order(), Some(432));
        let h = misc::AdditiveAbelian(vec![0, -3]);
        assert_eq!(h.free_rank(), 1);
        assert_eq!(h.invariant_factors(), &[3]);
    }

    #[test]
    fn test_lie_heisenberg_names() {
        assert_eq!(lie::Nilpotent("Heisenberg").dimension(), 3);
        assert_eq!(lie::Nilpotent("Heisenberg7").dimension(), 7);
    }

    #[test]
    #[should_panic(expected = "only \"Heisenberg\"")]
    fn test_lie_unknown_name_is_refused() {
        // An unknown name must not silently return the Heisenberg group.
        lie::Nilpotent("Engel4");
    }

    #[test]
    #[should_panic(expected = "only \"Heisenberg\"")]
    fn test_lie_even_heisenberg_dimension_is_refused() {
        lie::Nilpotent("Heisenberg4");
    }
}
