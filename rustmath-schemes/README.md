# rustmath-schemes

Comprehensive support for schemes in algebraic geometry - the fundamental objects of modern algebraic geometry.

## Overview

This crate implements a complete framework for working with schemes, including both affine and projective varieties, as well as specialized structures like elliptic curves. The design follows a trait-based approach that unifies different types of schemes under a common interface.

### Module Organization

- **`generic`**: Core scheme infrastructure and trait definitions
- **`affine`**: Affine schemes (Spec(R)) and affine space
- **`projective`**: Projective schemes, graded rings, and the Proj construction
- **`elliptic_curves`**: Elliptic curves with Weierstrass models and isogenies

### Core Features

#### Generic Scheme Framework

- **Scheme Trait**: Universal abstraction for all scheme types
- **Morphisms**: Structure-preserving maps between schemes
- **Dimension Theory**: Krull dimension and dimension computations
- **Algebraic Properties**: Smoothness, regularity, normality checks

#### Affine Schemes

- **Spec Construction**: Spec(R) for commutative rings R
- **Affine Space**: 𝔸ⁿ as Spec(k[x₁, ..., xₙ])
- **Closed Subschemes**: Varieties defined by ideals V(I)
- **Distinguished Opens**: D(f) basic open sets

#### Projective Schemes

- **Graded Rings**: Foundation for the Proj construction
- **Proj Construction**: Building schemes from graded rings
- **Projective Spaces**: ℙⁿ with homogeneous coordinates
- **Veronese Embeddings**: Maps ℙⁿ → ℙᴺ via degree d monomials
- **Segre Embeddings**: Product embeddings ℙⁿ × ℙᵐ → ℙᴺ
- **Projective Morphisms**: Morphisms between projective schemes
- **Line Bundles**: Locally free sheaves of rank 1
- **Ample Line Bundles**: Line bundles that embed into projective space

#### Elliptic Curves

- **Weierstrass Models**: Both general and short Weierstrass forms
- **Group Law**: Abelian group structure on rational points
- **Isogenies**: Morphisms between elliptic curves
- **Torsion Points**: Points of finite order E[n]
- **Invariants**: j-invariant and discriminant Δ

## Examples

### Projective Space

```rust
use rustmath_schemes::projective_space::{ProjectiveSpace, ProjectivePoint};

// Create ℙ² (projective plane)
let p2: ProjectiveSpace<i32> = ProjectiveSpace::new(2);
assert_eq!(p2.dimension(), 2);

// Create a point [1 : 2 : 3]
let point = ProjectivePoint::new(vec![1, 2, 3]).unwrap();
assert!(p2.contains_point(&point));
```

### Veronese Embedding

The twisted cubic curve ν₃: ℙ¹ → ℙ³:

```rust
use rustmath_schemes::veronese::VeroneseEmbedding;
use rustmath_schemes::projective_space::ProjectivePoint;

let twisted_cubic = VeroneseEmbedding::<i32>::twisted_cubic();

// Map [2:1] to [8:4:2:1] (corresponds to [s³:s²t:st²:t³])
let point = ProjectivePoint::new(vec![2, 1]).unwrap();
let image = twisted_cubic.apply(&point).unwrap();
assert_eq!(image.coordinates(), &[8, 4, 2, 1]);
```

### Segre Embedding

Products of projective spaces σ: ℙ¹ × ℙ¹ → ℙ³:

```rust
use rustmath_schemes::segre::SegreEmbedding;
use rustmath_schemes::projective_space::ProjectivePoint;

let segre = SegreEmbedding::<i32>::p1_times_p1();

let p1 = ProjectivePoint::new(vec![1, 2]).unwrap();
let p2 = ProjectivePoint::new(vec![3, 4]).unwrap();
let image = segre.apply(&p1, &p2).unwrap();
assert_eq!(image.coordinates(), &[3, 4, 6, 8]);
```

### Line Bundles

The twisting sheaves 𝒪(d) on projective space:

```rust
use rustmath_schemes::line_bundle::LineBundle;
use rustmath_schemes::proj;

let p2 = proj::projective_space::<i32>(2);

// Create 𝒪(3) on ℙ²
let o3 = LineBundle::twisting_sheaf(p2, 3);
assert!(o3.is_ample()); // 𝒪(d) is ample for d > 0
assert_eq!(o3.h0(), 10); // dim H⁰(ℙ², 𝒪(3)) = 10
```

### Canonical Bundle

The canonical bundle K_X on projective space:

```rust
use rustmath_schemes::line_bundle::CanonicalBundle;

// K_{ℙ²} = 𝒪(-3)
let k_p2 = CanonicalBundle::<i32>::of_projective_space(2);
assert_eq!(k_p2.line_bundle().degree(), -3);
assert!(k_p2.is_fano()); // ℙ² is Fano
```

## Features

### Graded Rings

- Direct sum decomposition R = ⊕ Rₙ
- Homogeneous elements and ideals
- Hilbert function computation
- Polynomial rings graded by degree

### Proj Construction

- Proj(R) for graded rings
- Standard affine charts
- Twisting sheaves 𝒪(n)
- Serre's theorem on cohomology

### Projective Morphisms

- Morphisms defined by homogeneous polynomials
- Composition of morphisms
- Base locus computation
- Properties: proper, finite, birational

### Veronese and Segre Embeddings

- Veronese embedding νₐ: ℙⁿ → ℙᴺ
- Twisted cubic (ν₃: ℙ¹ → ℙ³)
- Veronese surface (ν₂: ℙ² → ℙ⁵)
- Segre embedding σ: ℙⁿ × ℙᵐ → ℙᴺ
- Multi-factor Segre embeddings

### Line Bundles

- Twisting sheaves 𝒪(n)
- Tensor products and duals
- Ample and very ample bundles
- Global sections H⁰(X, L)
- Picard group Pic(X)
- Canonical bundle K_X
- Fano, Calabi-Yau, and general type classification

## Testing

All functionality is thoroughly tested:

```bash
cargo test -p rustmath-schemes
```

Current test coverage: 89 tests, all passing.

## Architectural Decisions

### Design Philosophy

This crate follows several key architectural principles:

#### 1. Trait-Based Unification

All schemes (affine, projective, elliptic curves) implement a common `Scheme` trait defined in the `generic` module. This provides:

- **Polymorphism**: Write generic algorithms that work over any scheme type
- **Consistency**: Uniform interface for dimension, irreducibility, and other properties
- **Extensibility**: Easy to add new scheme types by implementing the trait

```rust
pub trait Scheme {
    type BaseRing: Ring;
    fn base_ring(&self) -> &Self::BaseRing;
    fn dimension(&self) -> Option<usize>;
    fn is_affine(&self) -> bool;
    fn is_projective(&self) -> bool;
    // ... other common operations
}
```

#### 2. Type Safety Through Generics

The crate uses Rust's type system to enforce mathematical correctness:

- **Ring-Parametric Types**: `ProjectiveSpace<R: Ring>` works over any ring
- **Field Constraints**: `EllipticCurve<F: Field>` requires a field, not just a ring
- **Compile-Time Checks**: Invalid operations are caught at compile time

#### 3. Module Organization

The crate is organized into four main modules:

1. **`generic/`**: Abstract infrastructure and traits
   - Defines `Scheme`, `SchemeMorphism`, `SchemePoint`
   - Provides algebraic property traits (`AlgebraicScheme`, `Separated`)
   - Implements dimension theory

2. **`affine/`**: Affine-specific implementations
   - `AffineScheme<R>` for Spec(R)
   - `AffineSpace<R>` for 𝔸ⁿ
   - Closed subschemes and distinguished opens

3. **`projective/`**: Projective-specific implementations
   - Re-exports existing projective code
   - Implements `Scheme` trait for projective types
   - Organizes graded rings, Proj, line bundles, embeddings

4. **`elliptic_curves/`**: Specialized elliptic curve theory
   - Weierstrass models with group law
   - Isogenies and torsion subgroups
   - Integration with generic scheme framework

#### 4. Separation of Concerns

- **Generic vs Specific**: Common operations live in `generic`, specialized operations in specific modules
- **Data vs Behavior**: Struct definitions separate from trait implementations
- **Pure Functions**: Most operations are pure transformations without side effects

#### 5. Future-Proof Design

The architecture is designed for extensibility:

- **New Scheme Types**: Can add surfaces, higher-dimensional varieties, etc.
- **Additional Properties**: Easy to add traits for new mathematical properties
- **Computational Backends**: Structure sheaves and ideals can use different implementations

### Key Implementation Details

#### Scheme Point Representation

- Affine points use coordinate vectors
- Projective points use homogeneous coordinates with equivalence relations
- Elliptic curve points distinguish affine points from the point at infinity

#### Morphism Encoding

- Affine morphisms induced by ring homomorphisms
- Projective morphisms defined by homogeneous polynomials
- Isogenies as special morphisms between elliptic curves

#### Dimension Computation

- Cached where possible to avoid recomputation
- Optional values (`Option<usize>`) handle infinite-dimensional cases
- Krull dimension serves as the fundamental notion

### Dependencies

The crate depends on:

- **`rustmath-core`**: Core trait definitions (Ring, Field, etc.)
- **`rustmath-rings`**: Ring structures and operations
- **`rustmath-integers`**: Integer arithmetic
- **`rustmath-rationals`**: Rational arithmetic
- **`rustmath-polynomials`**: Polynomial rings and ideals
- **`rustmath-matrix`**: Linear algebra operations

### Testing Strategy

Tests are organized per-module:

- Unit tests in each module verify core functionality
- Integration tests (future) will test interactions between modules
- Doc tests serve as both tests and examples

### Performance Considerations

- **Lazy Evaluation**: Properties like discriminant are computed only when needed
- **Caching**: Computed values (dimension, invariants) are cached when possible
- **Generic Specialization**: Concrete types can provide optimized implementations

## Mathematical Background

This implementation follows standard conventions from algebraic geometry:

- **Hartshorne**: "Algebraic Geometry" (Chapters II-III)
- **Harris**: "Algebraic Geometry: A First Course"
- **Shafarevich**: "Basic Algebraic Geometry"
- **Silverman**: "The Arithmetic of Elliptic Curves" (for elliptic curves)
- **Liu**: "Algebraic Geometry and Arithmetic Curves"

## License

GPL-2.0-or-later
