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
pub mod lie_algebra;
pub mod lie_algebra_element;
pub mod poincare_birkhoff_witt;
pub mod representation;
pub mod abelian;
pub mod classical;
pub mod exceptional;
pub mod heisenberg;
pub mod free_lie_algebra;
pub mod nilpotent;
pub mod virasoro;
pub mod two_dimensional;
pub mod three_dimensional;
pub mod examples;
pub mod bch;
pub mod affine_lie_algebra;
pub mod bgg_dual_module;
pub mod bgg_resolution;
pub mod verma_module;
pub mod witt;
pub mod pwitt;
pub mod onsager;
pub mod structure_coefficients;
pub mod morphism;
pub mod subalgebra;
pub mod quotient;
pub mod chevalley_basis;
pub mod compact_real_form;

pub use cartan_type::{Affinity, CartanLetter, CartanType};
pub use root_system::{Root, RootSystem};
pub use weyl_group::{WeylGroup, WeylGroupElement};
pub use lie_algebra::{
    LieAlgebraElement, LieAlgebraBase, LieAlgebraWithGenerators,
    FinitelyGeneratedLieAlgebra, InfinitelyGeneratedLieAlgebra,
    LieAlgebraFromAssociative, MatrixLieAlgebraFromAssociative,
    AssociativeLieElement, MatrixLieElement, LiftMorphismToAssociative,
    LieAlgebraElementWrapper,
};
pub use lie_algebra_element::{
    LieObject, LieGenerator, LieBracket, GradedLieBracket, LyndonBracket,
    FreeLieAlgebraElement, StructureCoefficientsElement as StructureCoefficientsElt,
    UntwistedAffineLieAlgebraElement,
};
pub use poincare_birkhoff_witt::{
    PBWMonomial, PBWElement, PoincareBirkhoffWittBasis, PoincareBirkhoffWittBasisSemisimple,
};
pub use representation::{
    Representation, RepresentationElement, RepresentationByMorphism,
    TrivialRepresentation, FaithfulRepresentationNilpotentPBW, FaithfulRepresentationPBWPosChar,
};
pub use abelian::{
    AbelianLieAlgebra, AbelianLieAlgebraElement, InfiniteDimensionalAbelianLieAlgebra, LieAlgebra,
};
pub use classical::{
    GeneralLinearLieAlgebra, SpecialLinearLieAlgebra, SpecialOrthogonalLieAlgebra,
    SymplecticLieAlgebra,
};
pub use exceptional::{
    E6LieAlgebra, E7LieAlgebra, E8LieAlgebra, F4LieAlgebra, G2LieAlgebra,
    ExceptionalMatrixLieAlgebra,
};
pub use heisenberg::{
    HeisenbergAlgebra, HeisenbergAlgebraElement, HeisenbergAlgebraMatrix,
    InfiniteHeisenbergAlgebra,
};
pub use free_lie_algebra::{
    FreeLieAlgebra, FreeLieAlgebraElement, FreeLieAlgebraBasis,
    LieBracket, LyndonWord, is_lyndon,
};
pub use nilpotent::{
    NilpotentLieAlgebra, NilpotentLieAlgebraElement, FreeNilpotentLieAlgebra,
};
pub use virasoro::{
    VirasoroAlgebra, VirasoroElement, VirasoroGenerator,
    RankTwoHeisenbergVirasoro, RankTwoHeisenbergVirasoroElement, RankTwoGenerator,
};
pub use two_dimensional::{
    TwoDimensionalLieAlgebra, TwoDimensionalLieAlgebraElement,
};
pub use three_dimensional::{
    ThreeDimensionalLieAlgebra, ThreeDimensionalLieAlgebraElement,
};
pub use bch::{
    BCHIterator, bch_iterator, bch_sum,
};
pub use affine_lie_algebra::{
    AffineLieAlgebra, UntwistedAffineLieAlgebra, TwistedAffineLieAlgebra,
    UntwistedAffineElement, TwistedAffineIndices, AffineLieAlgebraElement,
};
pub use bgg_dual_module::{
    BGGDualModule, SimpleModule, FiniteDimensionalSimpleModule,
    SimpleModuleElement, SimpleModuleIndices,
};
pub use bgg_resolution::{
    BGGResolution, DifferentialMap, build_differential, dot_action,
};
pub use verma_module::{
    VermaModule, VermaModuleElement, VermaModuleBasisElement, Weight,
    VermaModuleHomomorphism, VermaModuleHomset, ContravariantForm,
};
pub use witt::{
    WittAlgebra, WittElement,
};
pub use pwitt::{
    PolynomialWittAlgebra, PolynomialWittElement,
};
pub use onsager::{
    OnsagerAlgebra, OnsagerElement, OnsagerGenerator, OnsagerGeneratorType,
    OnsagerAlgebraACE, ACEElement, ACEGenerator, ACEGeneratorType,
};
pub use structure_coefficients::{
    LieAlgebraWithStructureCoefficients, StructureCoefficientsElement,
};
pub use morphism::{
    LieAlgebraHomomorphism, LieAlgebraHomset, IdentityMorphism, ZeroMorphism,
    IsZero, EvaluateMorphism, PreservesBracket, ComposeMorphism, Kernel, Image,
    MorphismProperties,
};
pub use subalgebra::{
    LieSubalgebra, Center, Normalizer, Centralizer,
    BracketClosure, CheckIdeal, Reduce, Contains,
    DerivedSeries, LowerCentralSeries,
};
pub use quotient::{
    LieQuotient, QuotientElement, ProjectionMorphism,
    NaturalProjection, StructureCoefficients,
};
pub use chevalley_basis::{
    LieAlgebraChevalleyBasis, LieAlgebraChevalleyBasisSimplyLaced,
    ChevalleyBasisElement,
};
pub use compact_real_form::{
    MatrixCompactRealForm, CompactRealFormElement,
};

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
