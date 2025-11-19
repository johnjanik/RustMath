//! # RustMath Modules
//!
//! This crate provides functionality for modules over rings, corresponding to
//! SageMath's sage.modules module.
//!
//! ## Main Concepts
//!
//! - **Free Modules**: Modules with a free basis over a ring
//! - **Finitely Generated Modules**: Modules over PIDs (Principal Ideal Domains)
//! - **Graded Modules**: Modules with a grading structure
//! - **Vector Spaces**: Modules over fields
//! - **Quadratic Modules**: Modules with quadratic forms
//! - **Morphisms**: Homomorphisms between modules

pub mod module;
pub mod free_module;
pub mod free_module_element;
pub mod free_module_homspace;
pub mod free_module_morphism;
pub mod free_module_integer;
pub mod free_module_pseudohomspace;
pub mod free_module_pseudomorphism;
pub mod submodule;
pub mod quotient_module;
pub mod matrix_morphism;
pub mod vector_space_homspace;
pub mod vector_space_morphism;
pub mod misc;
pub mod tensor_operations;
pub mod tutorial_free_modules;
pub mod combinatorial_free_module;
pub mod module_morphism;

// Tensor modules - comprehensive tensor algebra for free modules
pub mod tensor;

// Finitely generated modules over PIDs
pub mod fg_pid {
    pub mod fgp_module;
    pub mod fgp_element;
    pub mod fgp_morphism;

    pub use fgp_module::FGPModule;
    pub use fgp_element::FGPElement;
    pub use fgp_morphism::FGPMorphism;
}

// Graded modules
pub mod fp_graded {
    pub mod module;
    pub mod element;
    pub mod morphism;
    pub mod homspace;
    pub mod free_module;
    pub mod free_element;
    pub mod free_morphism;
    pub mod free_homspace;

    // Steenrod algebra modules
    pub mod steenrod {
        pub mod module;
        pub mod morphism;

        pub use self::module::SteenrodModule;
        pub use self::morphism::SteenrodMorphism;
    }

    pub use self::module::FPGradedModule;
    pub use element::FPGradedElement;
    pub use morphism::FPGradedMorphism;
    pub use homspace::FPGradedHomspace;
}

// Modules with basis - comprehensive category implementation
#[path = "with_basis/lib.rs"]
pub mod with_basis;

// Re-export key with_basis types at module level for convenience
pub use with_basis::{
    ModuleWithBasisElement,
    ModuleWithBasis,
    ModuleWithBasisParentMethods,
    FreeModuleWithBasis,
    ModuleWithBasisMorphism,
    CartesianProduct,
    DualModule,
    HomSpace,
    TensorProduct,
};

// Ore modules (modules over Ore algebras)
pub mod ore_module;
pub mod ore_module_element;
pub mod ore_module_homspace;
pub mod ore_module_morphism;

// Quadratic modules
pub mod free_quadratic_module;
pub mod free_quadratic_module_integer_symmetric;
pub mod torsion_quadratic_module;

// Filtered vector spaces
pub mod filtered_vector_space;
pub mod multi_filtered_vector_space;

// Vector implementations (dense and sparse)
pub mod vector_integer_dense;
pub mod vector_integer_sparse;
pub mod vector_rational_dense;
pub mod vector_rational_sparse;
pub mod vector_double_dense;
pub mod vector_real_double_dense;
pub mod vector_complex_double_dense;
pub mod vector_mod2_dense;
pub mod vector_modn_dense;
pub mod vector_modn_sparse;
pub mod vector_symbolic_dense;
pub mod vector_symbolic_sparse;
pub mod vector_callable_symbolic_dense;

// NumPy-compatible vectors
pub mod vector_numpy_dense;
pub mod vector_numpy_integer_dense;

// Other utilities
pub mod diamond_cutting;
pub mod finite_submodule_iter;
pub mod complex_double_vector;
pub mod real_double_vector;

// Re-export commonly used types
pub use module::Module;
pub use free_module::FreeModule;
pub use free_module_element::FreeModuleElement;
pub use free_module_morphism::FreeModuleMorphism;
pub use quotient_module::QuotientModule;
pub use combinatorial_free_module::{CombinatorialFreeModule, CombinatorialFreeModuleElement};
pub use module_morphism::{ModuleMorphismByLinearity, ModuleEndomorphism, compose_morphisms};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_module_imports() {
        // Compilation test - verify tensor modules exist
        use crate::tensor::Components;
        let _: Components<i32> = Components::new(2, vec![2, 2]);
    }
}
