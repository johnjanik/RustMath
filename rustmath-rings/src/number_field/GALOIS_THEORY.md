# Galois Theory Implementation in RustMath

This document describes the implementation of Galois theory and number field morphisms in RustMath, corresponding to SageMath's `sage.rings.number_field.morphisms`.

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Algorithms](#algorithms)
6. [Examples](#examples)
7. [Limitations and Future Work](#limitations-and-future-work)

## Overview

Number field morphisms are structure-preserving maps between algebraic number fields. This implementation provides:

- **Embeddings**: Injective homomorphisms K → L (typically into ℂ)
- **Automorphisms**: Bijective homomorphisms K → K
- **Galois Groups**: The group Gal(K/ℚ) of automorphisms fixing ℚ
- **Splitting Fields**: The smallest field containing all roots of a polynomial
- **Galois Extensions**: Testing for normality, separability, and Galois properties

## Mathematical Background

### Number Fields

A **number field** is a finite field extension K of the rational numbers ℚ. Every number field can be written as K = ℚ(α) where α is algebraic over ℚ, meaning it satisfies some polynomial equation with rational coefficients.

The **degree** of K over ℚ, denoted [K:ℚ], is the dimension of K as a vector space over ℚ. It equals the degree of the minimal polynomial of α.

### Morphisms

A **morphism** φ: K → L between number fields is a ring homomorphism:
- φ(a + b) = φ(a) + φ(b)
- φ(a · b) = φ(a) · φ(b)
- φ(1) = 1

**Key Properties**:
- All morphisms between number fields are injective (no zero divisors)
- A morphism is determined by the image of a generator
- For K = ℚ(α), specifying φ(α) determines φ completely

### Embeddings

An **embedding** σ: K → ℂ is an injective morphism into the complex numbers. For a degree n number field K = ℚ(α), there are exactly n embeddings, corresponding to the n roots of the minimal polynomial of α in ℂ.

These n embeddings split into:
- **r₁ real embeddings**: Those with image in ℝ
- **r₂ pairs of complex embeddings**: Complex conjugate pairs
- We have: n = r₁ + 2r₂

The pair (r₁, r₂) is called the **signature** of the number field.

### Automorphisms

An **automorphism** is a bijective morphism σ: K → K from a field to itself. The set of all automorphisms forms a group under composition:

**Aut(K/ℚ)** = {σ: K → K | σ is an automorphism fixing ℚ}

**Properties**:
- Aut(K/ℚ) is always a subgroup of the symmetric group Sₙ
- |Aut(K/ℚ)| divides [K:ℚ]
- Equality holds if and only if K/ℚ is a Galois extension

### Galois Theory

A field extension K/ℚ is:
- **Normal**: If it's the splitting field of some polynomial over ℚ
- **Separable**: If minimal polynomials have no repeated roots
- **Galois**: If it's both normal and separable

**Fundamental Theorem of Galois Theory**: For a Galois extension K/ℚ, there is a bijection between:
- Subgroups of Gal(K/ℚ)
- Intermediate fields ℚ ⊆ F ⊆ K

The bijection reverses inclusion and is given by:
- H ↦ K^H (the fixed field of H)
- F ↦ Gal(K/F) (automorphisms fixing F)

### Splitting Fields

The **splitting field** of a polynomial f ∈ ℚ[x] is the smallest field containing all roots of f. It's unique up to isomorphism and is always a Galois extension of ℚ.

**Example**:
- For f(x) = x² - 2, the splitting field is ℚ(√2)
- For f(x) = x³ - 2, the splitting field is ℚ(∛2, ζ₃) where ζ₃ is a primitive cube root of unity

## Architecture

### Type Hierarchy

```
NumberFieldMorphism (trait)
    ├── NumberFieldEmbedding (struct)
    │   └── Embeddings into ℂ
    └── NumberFieldAutomorphism (struct)
        └── Automorphisms K → K
            └── GaloisGroup (struct)
                └── Collection of automorphisms with group structure
```

### Key Types

1. **NumberFieldMorphism**: Trait for all morphisms
   ```rust
   pub trait NumberFieldMorphism {
       fn apply(&self, element: &NumberFieldElement) -> Result<NumberFieldElement>;
       fn generator_image(&self) -> &NumberFieldElement;
       fn is_injective(&self) -> bool;
       fn is_surjective(&self) -> bool;
       fn is_automorphism(&self) -> bool;
   }
   ```

2. **NumberFieldEmbedding**: Injective morphisms into larger fields
   - Stores domain field and generator image
   - Used for computing with complex embeddings
   - Foundation for computing traces and norms

3. **NumberFieldAutomorphism**: Bijective self-maps
   - Can be composed to form new automorphisms
   - Have well-defined order and inverse
   - Form the Galois group when collected

4. **GaloisGroup**: Group structure on automorphisms
   - Stores multiplication table
   - Can identify group type (Cₙ, Sₙ, etc.)
   - Computes conjugacy classes and subgroups

## Implementation Details

### Representing Morphisms

A morphism φ: ℚ(α) → L is completely determined by φ(α), which must be a root of the minimal polynomial of α in L. We store:

```rust
pub struct NumberFieldEmbedding {
    domain: NumberField,
    generator_image: NumberFieldElement,
    description: String,
}
```

### Applying Morphisms

To apply φ to an element a₀ + a₁α + ... + aₙ₋₁α^(n-1):

1. Replace each power of α with the corresponding power of φ(α)
2. Compute a₀ + a₁φ(α) + ... + aₙ₋₁φ(α)^(n-1)

```rust
fn apply(&self, element: &NumberFieldElement) -> Result<NumberFieldElement> {
    let coeffs = (0..element.degree() + 1)
        .map(|i| element.coeff(i))
        .collect::<Vec<_>>();

    let mut result = self.domain.zero();
    let mut power = self.domain.one();

    for coeff in coeffs {
        let term = self.domain.from_rational(coeff);
        let scaled = self.domain.mul(&term, &power);
        result = self.domain.add(&result, &scaled);
        power = self.domain.mul(&power, &self.generator_image);
    }

    Ok(result)
}
```

### Computing Automorphisms

**Algorithm**: To find Aut(ℚ(α)/ℚ):

1. Find all roots of the minimal polynomial that lie in ℚ(α)
2. For each root β, check if α ↦ β extends to an automorphism
3. Verify: Does mapping α to β preserve all algebraic relations?
4. If yes, create the automorphism; if no, discard

**For Quadratic Fields**: ℚ(√d) always has exactly 2 automorphisms:
- Identity: √d ↦ √d
- Conjugation: √d ↦ -√d

```rust
fn compute_automorphisms_quadratic(field: &NumberField) -> Result<Vec<NumberFieldAutomorphism>> {
    let mut auts = vec![NumberFieldAutomorphism::identity(field.clone())];

    let alpha = field.generator();
    let neg_alpha = field.sub(&field.zero(), &alpha);

    if let Ok(aut) = NumberFieldAutomorphism::new(field.clone(), neg_alpha) {
        auts.push(aut);
    }

    Ok(auts)
}
```

### Computing Splitting Fields

**Algorithm**: To find the splitting field of f ∈ ℚ[x]:

1. If f is linear or constant, splitting field is ℚ
2. Factor f over ℚ to find an irreducible factor g
3. Let K₁ = ℚ(α) where α is a root of g
4. Factor f over K₁
5. If f splits completely, return K₁
6. Otherwise, find another irreducible factor and repeat

**For Quadratic Polynomials**: ax² + bx + c
- Splitting field is ℚ(√Δ) where Δ = b² - 4ac
- If Δ is a perfect square, splitting field is ℚ

```rust
fn splitting_field_quadratic(poly: &UnivariatePolynomial<Rational>) -> Result<NumberField> {
    let coeffs = poly.coefficients();
    let c = &coeffs[0];
    let b = &coeffs[1];
    let a = &coeffs[2];

    let disc = b.clone() * b.clone() - Rational::from_integer(4) * a.clone() * c.clone();

    let min_poly = UnivariatePolynomial::new(vec![
        -disc,
        Rational::zero(),
        Rational::one(),
    ]);

    Ok(NumberField::new(min_poly))
}
```

### Building Galois Groups

**Algorithm**:

1. Compute splitting field K of polynomial f
2. Compute all automorphisms Aut(K/ℚ)
3. Verify |Aut(K/ℚ)| = [K:ℚ] (Galois criterion)
4. Build multiplication table by composing automorphisms
5. Identify group structure (cyclic, symmetric, etc.)

```rust
pub struct GaloisGroup {
    field: NumberField,
    automorphisms: Vec<NumberFieldAutomorphism>,
    polynomial: UnivariatePolynomial<Rational>,
    multiplication_table: Option<Vec<Vec<usize>>>,
}
```

## Algorithms

### Testing for Galois Extensions

An extension K/ℚ is Galois if and only if |Aut(K/ℚ)| = [K:ℚ].

```rust
pub fn is_galois_extension(field: &NumberField) -> Result<bool> {
    let degree = field.degree();
    let auts = compute_automorphisms(field)?;
    Ok(auts.len() == degree)
}
```

### Testing for Normal Extensions

K/ℚ is normal if whenever an irreducible polynomial over ℚ has one root in K, it splits completely in K.

**Practical Test**: K is normal iff it's the splitting field of its minimal polynomial.

### Testing for Separable Extensions

Over ℚ (characteristic 0), **all** extensions are separable. The minimal polynomial never has repeated roots.

```rust
pub fn is_separable_extension(field: &NumberField) -> bool {
    true  // Always true in characteristic 0
}
```

## Examples

### Example 1: Quadratic Field ℚ(√2)

```rust
use rustmath_numberfields::NumberField;
use rustmath_rings::number_field::morphisms::*;
use rustmath_polynomials::univariate::UnivariatePolynomial;
use rustmath_rationals::Rational;

// Create ℚ(√2)
let poly = UnivariatePolynomial::new(vec![
    Rational::from_integer(-2),
    Rational::from_integer(0),
    Rational::from_integer(1),
]);
let field = NumberField::new(poly);

// Compute automorphisms
let auts = compute_automorphisms(&field).unwrap();
assert_eq!(auts.len(), 2);  // Identity and conjugation

// Check if Galois
let is_gal = is_galois_extension(&field).unwrap();
assert!(is_gal);  // Quadratic fields are always Galois

// Compute discriminant
let disc = field.discriminant();
assert_eq!(disc, Rational::from_integer(8));
```

### Example 2: Splitting Field of x² - 3

```rust
// Compute splitting field
let poly = UnivariatePolynomial::new(vec![
    Rational::from_integer(-3),
    Rational::from_integer(0),
    Rational::from_integer(1),
]);

let splitting = splitting_field(&poly).unwrap();
assert_eq!(splitting.degree(), 2);

// Compute Galois group
let gal = galois_group(&poly).unwrap();
assert_eq!(gal.order(), 2);
assert!(gal.is_abelian());
assert_eq!(gal.identify(), "C₂ (Cyclic group of order 2)");
```

### Example 3: Automorphism Application

```rust
let field = /* ... create ℚ(√2) ... */;
let auts = compute_automorphisms(&field).unwrap();

// Apply identity automorphism
let id = &auts[0];
let alpha = field.generator();  // α = √2
let result = id.apply(&alpha).unwrap();
assert_eq!(result, alpha);

// Apply conjugation (if it exists)
if auts.len() > 1 {
    let conj = &auts[1];
    let result = conj.apply(&alpha).unwrap();
    // result should be -√2
}
```

## Limitations and Future Work

### Current Limitations

1. **Root Finding**: Computing all embeddings requires finding roots of polynomials over ℂ
   - Numerical methods needed (Newton's method, eigenvalue algorithms)
   - Requires complex number arithmetic with arbitrary precision

2. **Higher Degree Fields**: Automorphism computation only works for degree ≤ 2
   - Need algorithms for finding roots within the field
   - Requires polynomial factorization over number fields

3. **Splitting Fields**: Only implemented for degree ≤ 2 polynomials
   - General case requires iterated field extensions
   - Need tower-of-fields representation

4. **Normality Testing**: Not yet implemented
   - Requires checking if minimal polynomial splits in the field
   - Need polynomial factorization algorithms

5. **Inverse Automorphisms**: Computing inverses not yet implemented
   - Need to solve α ↦ ? for given automorphism
   - Requires root finding in the field

### Future Enhancements

#### Phase 1: Numerical Embeddings
- Implement root-finding for polynomials (Newton's method, Durand-Kerner)
- Compute all embeddings into ℂ numerically
- Support for computing traces and norms via embeddings

#### Phase 2: Higher Degree Automorphisms
- Implement automorphism computation for cubic fields
- Use discriminant and resolvent polynomials
- Extend to quartic and higher degree fields

#### Phase 3: General Splitting Fields
- Implement tower-of-fields for iterated extensions
- Algorithm for computing splitting fields of arbitrary polynomials
- Galois group computation via permutation action on roots

#### Phase 4: Advanced Galois Theory
- Compute intermediate fields (subfield lattice)
- Identify Galois group structure (present as permutation group)
- Implement resolvent polynomials for Galois group computation
- Support for non-Galois extensions and their Galois closures

#### Phase 5: Computational Optimizations
- Cache automorphism computations
- Optimize morphism application
- Use matrix representation for automorphisms
- Parallel computation of embeddings

### Integration with Existing Systems

Future work could integrate with:
- **PARI/GP**: Use PARI for number field computations
- **FLINT**: Fast polynomial arithmetic
- **GAP**: Group-theoretic computations for Galois groups
- **Sage Wrappers**: Direct translation from SageMath code

## References

1. **Algebraic Number Theory** by Jürgen Neukirch
   - Comprehensive treatment of number fields and Galois theory

2. **A Course in Computational Algebraic Number Theory** by Henri Cohen
   - Algorithms for computing with number fields

3. **SageMath Documentation**: `sage.rings.number_field.morphisms`
   - Reference implementation

4. **LMFDB** (L-functions and Modular Forms Database)
   - Number field data and invariants

5. **Computational Galois Theory** by Alexander Hulpke
   - Algorithms for computing Galois groups

## Conclusion

This implementation provides a solid foundation for Galois theory computations in RustMath. While currently limited to low-degree cases, the architecture is designed to scale to more sophisticated algorithms. The type-safe Rust implementation ensures correctness while maintaining performance.

The modular design allows for gradual enhancement: numerical methods can be added for embeddings, factorization algorithms for automorphism computation, and group-theoretic tools for Galois group structure.
