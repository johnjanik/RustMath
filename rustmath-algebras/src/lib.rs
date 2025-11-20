//! RustMath Algebras - Advanced algebraic structures
//!
//! This crate provides implementations of various algebraic structures including:
//! - Free algebras
//! - Quotient algebras
//! - Clifford algebras
//! - Finite dimensional algebras
//! - Group algebras
//! - Hecke algebras
//! - Lie algebras
//!
//! Corresponds to the sage.algebras module from SageMath.

pub mod free_algebra;
pub mod finite_dimensional_algebra;
pub mod finite_dimensional_algebra_ideal;
pub mod finite_dimensional_algebra_morphism;
pub mod quotient_algebra;
pub mod clifford_algebra;
pub mod group_algebra;
pub mod symmetric_group_algebra;
pub mod traits;
pub mod affine_nil_temperley_lieb;
pub mod blob_algebra;
pub mod down_up_algebra;
pub mod algebra_morphism;
pub mod algebra_with_parent;
pub mod cached_algebra;
pub mod askey_wilson;
pub mod associated_graded;
pub mod cellular_basis;
pub mod cluster_algebra;
pub mod commutative_dga;
pub mod finite_gca;
pub mod free_zinbiel_algebra;
pub mod hall_algebra;
pub mod jordan_algebra;
pub mod ariki_koike_algebra;
pub mod cubic_hecke_algebra;
pub mod cubic_hecke_base_ring;
pub mod cubic_hecke_matrix_rep;
pub mod iwahori_hecke_algebra;
pub mod nil_coxeter_algebra;
pub mod catalog;
pub mod poly_tup_engine;
pub mod fusion_ring;
pub mod fusion_double;
pub mod f_matrix;
pub mod fast_parallel_fmats_methods;
pub mod fast_parallel_fusion_ring_braid_repn;
pub mod shm_managers;
pub mod ariki_koike_specht_modules;
pub mod octonion_algebra;
pub mod q_system;
pub mod quantum_oscillator;
pub mod quantum_clifford;
pub mod ace_quantum_onsager;
pub mod quantum_matrix_coordinate;
pub mod quaternion_algebra;
pub mod orlik_solomon;
pub mod orlik_terao;
pub mod rational_cherednik_algebra;
pub mod schur_algebra;
pub mod shuffle_algebra;
pub mod splitting_algebra;
pub mod quantum_group_gap;
pub mod quantum_group_representations;
pub mod steenrod_algebra;
pub mod tensor_algebra;
pub mod weyl_algebra;
pub mod yangian;
pub mod yokonuma_hecke_algebra;
pub mod partition_algebra;

pub use free_algebra::*;
pub use finite_dimensional_algebra::*;
pub use finite_dimensional_algebra_ideal::FiniteDimensionalAlgebraIdeal;
pub use finite_dimensional_algebra_morphism::{
    FiniteDimensionalAlgebraMorphism, FiniteDimensionalAlgebraHomset,
};
pub use quotient_algebra::{
    QuotientAlgebra, QuotientAlgebraElement, FreeAlgebraIdeal,
    hamilton_quatalg, is_FreeAlgebraQuotientElement,
};
pub use clifford_algebra::{
    CliffordAlgebra, CliffordAlgebraElement, CliffordBasisElement, CliffordAlgebraIndices,
    ExteriorAlgebra, ExteriorAlgebraDifferential, ExteriorAlgebraBoundary,
    ExteriorAlgebraCoboundary, ExteriorAlgebraIdeal, StructureCoefficients,
};
pub use group_algebra::*;
pub use symmetric_group_algebra::{
    SymmetricGroupElement, SymmetricGroupAlgebraBuilder,
    ZSymmetricGroupAlgebra, ZSymmetricGroupElement,
    QSymmetricGroupAlgebra, QSymmetricGroupElement,
    verify_coxeter_relations,
};
pub use traits::*;
pub use affine_nil_temperley_lieb::{AffineNilTemperleyLiebTypeA, Element as AffineNilTemperleyLiebElement};
pub use blob_algebra::{BlobAlgebra, BlobElement, BlobBasisElement};
pub use algebra_morphism::{AlgebraMorphism, AlgebraEndomorphism, AlgebraAutomorphism};
pub use askey_wilson::{AskeyWilsonAlgebra, AskeyWilsonIndex};
pub use associated_graded::AssociatedGradedAlgebra;
pub use cellular_basis::{CellularBasis, CellularIndex};
pub use cluster_algebra::{
    ClusterAlgebra, ClusterAlgebraElement, ClusterAlgebraSeed,
    PrincipalClusterAlgebraElement, GVector, DVector, ExchangeMatrix,
};
pub use commutative_dga::{
    GCAlgebra, GCAlgebraMultigraded, Differential, DifferentialMultigraded,
    DifferentialGCAlgebra, DifferentialGCAlgebraMultigraded,
    CohomologyClass, GCAlgebraHomset, GCAlgebraMorphism,
    Degree, Generator, GCAlgebraElement,
};
pub use finite_gca::{FiniteGCAlgebra, FiniteGCABasisElement, FiniteGCAlgebraElement};
pub use free_zinbiel_algebra::{
    FreeZinbielAlgebra, FreeZinbielAlgebraElement, ZinbielWord, ZinbielFunctor,
};
pub use hall_algebra::{
    HallAlgebra, HallAlgebraMonomials, HallAlgebraElement, Partition as HallPartition, transpose_cmp,
};
pub use jordan_algebra::{
    JordanAlgebra, JordanAlgebraElement, JordanAlgebraType, JordanAlgebraTrait,
    SpecialJordanAlgebra, JordanAlgebraSymmetricBilinear, ExceptionalJordanAlgebra,
};
pub use ariki_koike_algebra::{
    ArikiKoikeAlgebra, ArikiKoikeElement, ArikiKoikeWord, ArikiKoikeBasis,
};
pub use cubic_hecke_algebra::{
    CubicHeckeAlgebra, CubicHeckeElement, BraidWord,
};
pub use cubic_hecke_base_ring::{
    CubicHeckeRingOfDefinition, CubicHeckeExtensionRing, GaloisGroupAction,
    MarkovTraceVersion, normalize_names_markov, register_ring_hom, RingHomomorphism,
    embedding_to_extension,
};
pub use cubic_hecke_matrix_rep::{
    CubicHeckeMatrixRep, CubicHeckeMatrixSpace, GenSign, RepresentationType,
    AbsIrreducibleRep,
};
pub use iwahori_hecke_algebra::{
    IwahoriHeckeAlgebra, IwahoriHeckeElement, HeckeBasisType,
    index_cmp, normalized_laurent_polynomial,
};
pub use nil_coxeter_algebra::{
    NilCoxeterAlgebra, NilCoxeterElement,
};
pub use poly_tup_engine::{
    PolyTuple, poly_to_tup, constant_coeff, variables, get_variables_degrees,
    resize, apply_coeff_map, poly_tup_sortkey, compute_known_powers, tup_to_univ_poly,
};
pub use fusion_ring::{FusionRing, FusionRingElement, Weight};
pub use fusion_double::{FusionDouble, FusionDoubleElement, FusionDoubleIndex};
pub use f_matrix::FMatrix;
pub use fast_parallel_fmats_methods::{FMatrixExecutor, ParallelTask};
pub use fast_parallel_fusion_ring_braid_repn::BraidRepnExecutor;
pub use shm_managers::{FvarsHandler, KSHandler, make_fvars_handler, make_ks_handler};
pub use ariki_koike_specht_modules::{SpechtModule, SpechtModuleElement, Multipartition, Partition};
pub use octonion_algebra::{OctonionAlgebra, Octonion};
pub use q_system::{QSystem, QElement, QIndex, QMonomial, is_tamely_laced};
pub use quantum_oscillator::{
    QuantumOscillatorAlgebra, OscillatorElement, OscillatorIndex, Generator,
    FockSpaceRepresentation, FockSpaceElement,
};
pub use quantum_clifford::{
    QuantumCliffordAlgebra, CliffordElement, CliffordIndex, CliffordGenerator,
    FermionIndex,
};
pub use ace_quantum_onsager::{
    ACEQuantumOnsagerAlgebra, ACEOnsagerElement, OnsagerMonomial, Generator as OnsagerGenerator,
};
pub use quantum_matrix_coordinate::{
    QuantumMatrixCoordinateAlgebra, QuantumGL, QuantumMatrixElement,
    MatrixIndex, QuantumMatrixMonomial,
};
pub use quaternion_algebra::{
    QuaternionAlgebra, Quaternion, QuaternionOrder, QuaternionFractionalIdeal,
    QuaternionFractionalIdealRational, basis_for_quaternion_lattice,
    intersection_of_row_modules_over_zz, is_quaternion_algebra,
    normalize_basis_at_p, maxord_solve_aux_eq,
};
pub use orlik_solomon::{
    Matroid, OrlikSolomonAlgebra, OrlikSolomonElement, OrlikSolomonInvariantAlgebra,
};
pub use orlik_terao::{
    OrlikTeraoAlgebra, OrlikTeraoElement, OrlikTeraoInvariantAlgebra,
};
pub use rational_cherednik_algebra::{
    RationalCherednikAlgebra, RationalCherednikElement, CartanType,
};
pub use schur_algebra::{
    SchurAlgebra, SchurElement, SchurTensorModule,
    schur_representative_indices, schur_representative_from_index,
    gl_irreducible_character,
};
pub use shuffle_algebra::{
    ShuffleAlgebra, ShuffleElement, DualPBWBasis, Word, shuffle_product,
};
pub use splitting_algebra::{
    SplittingAlgebra, SplittingAlgebraElement, solve_with_extension,
};
pub use quantum_group_gap::{
    QuantumGroup, QuantumGroupModule, HighestWeightModule, HighestWeightSubmodule,
    TensorProductOfHighestWeightModules, LowerHalfQuantumGroup,
    QuaGroupModuleElement, QuaGroupRepresentationElement, CrystalGraphVertex,
    QuantumGroupMorphism, QuantumGroupHomset, projection_lower_half,
};
pub use quantum_group_representations::{
    QuantumGroupRepresentation, CyclicRepresentation, AdjointRepresentation,
    MinusculeRepresentation, CrystalElement,
};
pub use steenrod_algebra::{
    SteenrodAlgebra, SteenrodElement, SteenrodMonomial, SteenrodSquare, SteenrodPrime,
};
pub use tensor_algebra::{
    TensorAlgebra, TensorElement, TensorMonomial, TensorAlgebraFunctor,
};
pub use weyl_algebra::{
    WeylAlgebra, WeylElement, WeylMonomial,
};
pub use yangian::{
    Yangian, YangianElement, YangianMonomial, YangianIndex, YangianLevel, GradedYangianBase,
};
pub use yokonuma_hecke_algebra::{
    YokonumaHeckeAlgebra, YokonumaElement, YokonumaWord, YokonumaGenerator,
};
pub use partition_algebra::{
    PartitionAlgebra, PartitionAlgebraElement, PartitionDiagram,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_algebra_operations() {
        // Basic smoke tests will be added as we implement structures
        assert!(true);
    }
}
