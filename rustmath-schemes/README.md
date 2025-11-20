# rustmath-schemes

Comprehensive support for projective schemes in algebraic geometry.

## Overview

This crate implements fundamental concepts from algebraic geometry related to projective schemes:

- **Graded Rings**: Foundation for the Proj construction
- **Proj Construction**: Building schemes from graded rings
- **Projective Spaces**: ℙⁿ with homogeneous coordinates
- **Veronese Embeddings**: Maps ℙⁿ → ℙᴺ via degree d monomials
- **Segre Embeddings**: Product embeddings ℙⁿ × ℙᵐ → ℙᴺ
- **Projective Morphisms**: Morphisms between projective schemes
- **Line Bundles**: Locally free sheaves of rank 1
- **Ample Line Bundles**: Line bundles that embed into projective space

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

## Mathematical Background

This implementation follows standard conventions from algebraic geometry:

- **Hartshorne**: "Algebraic Geometry" (Chapters II-III)
- **Harris**: "Algebraic Geometry: A First Course"
- **Shafarevich**: "Basic Algebraic Geometry"

## License

GPL-2.0-or-later
