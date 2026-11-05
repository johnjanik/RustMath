# RustMath Trait Hierarchy - Visual Guide

## Core Algebraic Trait Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALGEBRAIC STRUCTURES                          │
└─────────────────────────────────────────────────────────────────┘

                          GROUP HIERARCHY
                          ───────────────

    Magma (Binary Op)
         │
         ▼
    Semigroup (Associative)
         │
         ▼
    Monoid (+ Identity)
         │
         ▼
    Group (+ Inverses)
         │
         ▼
    AbelianGroup (Commutative)


                          RING HIERARCHY
                          ──────────────

    Ring (Add + Sub + Mul + Neg)
    └─ CommutativeRing (Commutative Mul)
       └─ IntegralDomain (No Zero Divisors)
          └─ EuclideanDomain (Division Algorithm)
             │
             └─ [Algorithms: gcd, lcm, extended_gcd]

    Ring (Also Extends)
    └─ Field (+ Multiplicative Inverses + Div)


                       MODULE/ALGEBRA HIERARCHY
                       ───────────────────────

    Module<R: Ring>
    └─ VectorSpace<F: Field>
       └─ Algebra<F: Field> (VectorSpace + Ring)


                         PARENT HIERARCHY
                         ────────────────

    Parent (Element Container)
    ├─ ParentWithBasis (Indexed Basis)
    ├─ ParentWithGenerators (Generator Set)
    ├─ RingParent (Has base_ring)
    ├─ ParentAsRing (Parent + Ring)
    └─ ParentEq (Equality)

```

---

## Category Theory Trait Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                   CATEGORY THEORY STRUCTURES                     │
└─────────────────────────────────────────────────────────────────┘

                         CATEGORIES
                         ──────────

    Category (Trait)
    │ name(), super_categories(), is_subcategory_of(), axioms()
    │
    ├─ FiniteCategory (All objects finite)
    ├─ CommutativeCategory (Commutative operations)
    ├─ TopologicalCategory (Has topology)
    └─ CartesianProductsCategory (Supports products)

    Concrete Implementations:
    ├─ GroupCategory (Objects: groups, Morphisms: homomorphisms)
    ├─ ModuleCategory<R: Ring> (Objects: R-modules, Morphisms: linear maps)
    └─ FieldCategory (Objects: fields, Morphisms: field homomorphisms)


                         MORPHISMS
                         ─────────

    Morphism (Trait)
    │ source(), target(), compose(), is_identity()
    │
    ├─ IdentityMorphism<T> (Maps T to itself)
    ├─ SetMorphism<T, F> (Function-based morphism)
    ├─ SetIsomorphism<T, F, G> (Bijection with inverse)
    ├─ FormalCoercionMorphism<S, T> (Type coercion)
    └─ CallMorphism<F> (Wraps callable)

    Isomorphism (Trait)
    │ inverse(), is_isomorphism()
    │
    └─ [Subtrait of Morphism]


                          FUNCTORS
                          ────────

    Functor (Trait)
    │ map_object(), map_morphism()
    │ [Maps objects and morphisms between categories]
    │
    ├─ IdentityFunctor<Obj, Morph> (Id_C: C → C)
    ├─ ForgetfulFunctor<From, To> (Forgets structure)
    └─ ComposedFunctor<F, G> (G ∘ F composition)


                   NATURAL TRANSFORMATIONS
                   ──────────────────────

    NaturalTransformation<Obj, Morph>
    │ components: HashMap<Obj, Morph>
    │ [Family of morphisms η_A: F(A) → G(A)]
    │
    ├─ vertical_composition() (θ ∘ η: F ⇒ H)
    └─ IdentityNaturalTransformation (Id_F: F ⇒ F)

```

---

## Integration Map: Traits Used Across Crates

```
┌─────────────────────────────────────────────────────────────────┐
│              TRAIT USAGE DEPENDENCY GRAPH                        │
└─────────────────────────────────────────────────────────────────┘

                        rustmath-core
                    [Foundation: Ring, Field,
                    Module, Parent traits]
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
           rustmath-      rustmath-   rustmath-
           integers       matrix      polynomials
                │            │            │
                │    ┌───────┴────────┬───┘
                │    ▼               ▼
                │  rustmath-algebras  rustmath-category
                │    (Algebras,      (Category, Functor,
                │     Modules)       Morphism, NatTrans)
                │                    │
                └────────┬───────────┘
                         ▼
                  [Category-aware
                   type hierarchy]

             Optional Integration Points
             ───────────────────────────

    rustmath-groups
      └─ group_traits.rs (Independent Group trait)
         [Could implement core Ring/Module traits]

    rustmath-rings
      └─ 85+ ring implementations
         [Not all use core Ring trait consistently]

    Domain-Specific Morphisms
      ├─ GroupMorphism (in rustmath-groups)
      ├─ ModuleMorphism (in rustmath-modules)
      └─ [Could use category.Morphism as base]

```

---

## Key Trait Composition Patterns Used

### Pattern 1: Generic Algorithms Over Rings
```rust
fn determinant<R: Ring>(matrix: &Matrix<R>) -> R { ... }
fn polynomial_gcd<R: EuclideanDomain>(p: Poly<R>, q: Poly<R>) -> Poly<R> { ... }
```

### Pattern 2: Parent Trait for Structure Containers
```rust
impl<R: Ring> Parent for PolynomialRing<R> {
    type Element = Polynomial<R>;
    fn contains(&self, poly: &Polynomial<R>) -> bool { ... }
}
```

### Pattern 3: Module Objects
```rust
impl<R: Ring> Module<R> for FreeModule<R> {
    type Element = FreeModuleElement<R>;
    fn scalar_mul(&self, r: &R, elem: &Self::Element) -> Self::Element { ... }
}
```

### Pattern 4: Category-Aware Type System
```rust
// GroupCategory tracks mathematical properties
impl GroupCategory {
    fn axioms(&self) -> Vec<&str> {
        vec!["closure", "associativity", "identity", "inverse"]
    }
}
```

---

## Crate Organization by Functionality Tier

```
┌─ TIER 0: Foundation
│  └─ rustmath-core (traits)
│
├─ TIER 1: Number Systems
│  ├─ rustmath-integers, rustmath-rationals, rustmath-reals
│  ├─ rustmath-complex, rustmath-finitefields, rustmath-padics
│  └─ [All implement core Ring/Field traits]
│
├─ TIER 2: Polynomials & Algebraic Structures
│  ├─ rustmath-polynomials<R: Ring>
│  ├─ rustmath-powerseries<R: Ring>
│  ├─ rustmath-modular<R: Ring>
│  └─ [Generic over Ring implementations from Tier 1]
│
├─ TIER 3: Linear & Module Algebra
│  ├─ rustmath-matrix<R: Ring>
│  ├─ rustmath-modules
│  └─ rustmath-algebras [Parent trait implementations]
│
├─ TIER 4: Algebraic Objects
│  ├─ rustmath-groups [Independent group_traits]
│  ├─ rustmath-rings [Ring implementations]
│  ├─ rustmath-monoids
│  ├─ rustmath-liealgebras
│  └─ rustmath-quivers
│
├─ TIER 5: Topology & Geometry
│  ├─ rustmath-topology, rustmath-manifolds
│  └─ rustmath-geometry, rustmath-sets
│
├─ TIER 6: Category Theory [NEW!]
│  ├─ rustmath-category [Category, Functor, Morphism, NatTrans]
│  └─ rustmath-homology [Homology theory emerging]
│
└─ TIER 7-9: Analysis, Applied Math, Visualization
   ├─ rustmath-calculus, rustmath-symbolic
   ├─ rustmath-numerics, rustmath-stats
   ├─ rustmath-graphs, rustmath-combinatorics
   └─ rustmath-plot*, rustmath-colors
```

---

## Integration Status Matrix

```
                    │ Uses Core  │ Uses Parent │ Uses Category │ Independent
                    │   Traits   │   Traits    │   Traits      │   Traits
────────────────────┼────────────┼─────────────┼───────────────┼───────────
Integers            │     ✓      │      -      │       -       │     -
Rationals           │     ✓      │      -      │       -       │     -
Complex             │     ✓      │      -      │       -       │     -
Finite Fields       │     ✓      │      -      │       -       │     -
Polynomials         │     ✓      │      -      │       -       │     -
Matrices<R>         │     ✓      │      -      │       -       │     -
Modules             │     ✓      │      ✓      │       -       │     -
Algebras            │     ✓      │      ✓      │       -       │     -
Groups              │     -      │      -      │       ✓       │     ✓
Rings (ring/)       │     ~      │      -      │       -       │     ~
Categories          │     -      │      -      │       ✓       │     -
────────────────────┴────────────┴─────────────┴───────────────┴───────────

Legend:
  ✓ = Fully integrated
  ~ = Partially integrated
  - = Not used
```

---

## Future Integration Opportunities

### High Priority
1. **Unify Group & Ring Traits** - Make rustmath-groups::Group implement core Ring
2. **Domain-Specific Morphisms** - GroupMorphism, ModuleMorphism inherit from category::Morphism
3. **Extend Categories** - Add PolynomialCategory, MatrixAlgebraCategory, FiniteFieldCategory

### Medium Priority
4. **Ring Implementations** - Standardize rustmath-rings/* to use core Ring trait
5. **Homology Module** - Build on NaturalTransformation for chain complexes
6. **Adjoint Functors** - Implement universal properties via adjoint pairs

### Lower Priority
7. **Derived Categories** - Triangulated structures for homological algebra
8. **Topos Theory** - Categorical logic foundations
9. **Computational Tactics** - Category-aware theorem proving

