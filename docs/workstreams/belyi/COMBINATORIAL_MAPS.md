# Combinatorial Maps System

## Overview

The combinatorial maps system in `rustmath-combinatorics` provides a powerful, type-safe infrastructure for registering and working with maps (functions) between combinatorial objects. The system features:

- **Decorator-based registration**: Clean, declarative syntax using macros
- **Bidirectional bijections**: Automatic inverse map registration
- **Type safety**: Full Rust type checking with no runtime type errors
- **Global registry**: Thread-safe global map registry for easy access
- **Metadata tracking**: Rich metadata about each map (description, types, bijection status)
- **Query capabilities**: Find maps by name or by source/target types

## Key Features

### 1. Simple Map Registration

Register a one-way map between combinatorial objects:

```rust
use rustmath_combinatorics::combinatorial_map::*;
use rustmath_combinatorics::Permutation;

register_map::<_, Permutation, Vec<Vec<usize>>>(
    "permutation_to_cycles",
    "Convert a permutation to its cycle representation",
    false,  // Not a bijection
    None,   // No inverse
    |perm: &Permutation| Some(perm.cycles()),
);
```

### 2. Bidirectional Bijections

Register a bijection with automatic inverse handling:

```rust
use rustmath_combinatorics::combinatorial_map::*;
use rustmath_combinatorics::Partition;

register_map::<_, Partition, Partition>(
    "partition_conjugate",
    "Conjugate (transpose) a partition",
    true,  // This is a bijection
    Some("partition_conjugate_inverse".to_string()),
    |p: &Partition| Some(p.conjugate()),
);

// Register the inverse
register_map::<_, Partition, Partition>(
    "partition_conjugate_inverse",
    "Inverse of partition conjugation",
    true,
    Some("partition_conjugate".to_string()),
    |p: &Partition| Some(p.conjugate()),
);
```

### 3. Declarative Macro Registration

Use the `register_combinatorial_map!` macro for cleaner syntax:

```rust
use rustmath_combinatorics::register_combinatorial_map;

// One-way map
register_combinatorial_map! {
    name: "permutation_to_cycles",
    description: "Convert a permutation to its cycle representation",
    from: Permutation => to: Vec<Vec<usize>>,
    bijection: false,
    forward: |perm| Some(perm.cycles())
}

// Bijection with automatic inverse registration
register_combinatorial_map! {
    name: "partition_conjugate",
    description: "Conjugate (transpose) a partition",
    from: Partition => to: Partition,
    bijection: true,
    forward: |p| Some(p.conjugate()),
    inverse: |p| Some(p.conjugate())
}
```

### 4. Applying Maps

Apply a registered map by name:

```rust
use rustmath_combinatorics::combinatorial_map::*;
use rustmath_combinatorics::{Permutation, Partition};

// Apply a one-way map
let perm = Permutation::from_vec(vec![2, 0, 1]).unwrap();
let cycles: Option<Vec<Vec<usize>>> = apply_map("permutation_to_cycles", &perm);

// Apply a bijection
let partition = Partition::new(vec![4, 3, 1]);
let conjugate: Option<Partition> = apply_map("partition_conjugate", &partition);

// Apply the inverse to get back the original
let back: Option<Partition> = apply_map("partition_conjugate_inverse", &conjugate.unwrap());
assert_eq!(back.unwrap(), partition);
```

### 5. Querying the Registry

Find maps by type:

```rust
use rustmath_combinatorics::combinatorial_map::*;
use rustmath_combinatorics::{Permutation, Partition};

// Find all maps from Permutation to Permutation
let maps = find_maps_between::<Permutation, Permutation>();
for map in maps {
    println!("{}: {}", map.name, map.description);
}

// Get metadata for a specific map
let metadata = get_map_metadata("partition_conjugate");
if let Some(meta) = metadata {
    println!("Map: {}", meta.name);
    println!("Description: {}", meta.description);
    println!("Is bijection: {}", meta.is_bijection);
    println!("Inverse: {:?}", meta.inverse_name);
}

// List all registered maps
let all_maps = list_all_maps();
for map in all_maps {
    println!("{} ({} -> {})",
             map.name,
             map.source_type_name,
             map.target_type_name);
}
```

## Architecture

### Core Components

1. **`CombinatorialMap` trait**: Marker trait for types that can be used in maps (automatically implemented for all `Clone + 'static` types)

2. **`MapRegistry`**: The core registry that stores maps and metadata
   - Stores type-erased map functions
   - Maintains metadata for each map
   - Provides type-safe querying and application

3. **`MapMetadata`**: Rich metadata about each map
   - Name and description
   - Source and target types (TypeId and name)
   - Whether it's a bijection
   - Name of inverse map (if applicable)

4. **Macros**:
   - `register_combinatorial_map!`: Declarative map registration
   - `define_bijection!`: Helper for defining bijection pairs

### Thread Safety

The global registry uses `Arc<RwLock<MapRegistry>>` for thread-safe access:
- Multiple concurrent readers (applying maps, querying metadata)
- Exclusive writer access for registration
- No performance penalty for read-heavy workloads

### Type Erasure

Maps are stored as type-erased `Box<dyn Fn(&dyn Any) -> Option<Box<dyn Any>>>`, but the public API maintains full type safety:
- Type checking at registration time
- Runtime type checking when applying maps
- Clear error handling with `Option` return types

## Example Maps

The `combinatorial_map_examples` module provides several example maps:

### 1. Permutation to Cycles
```rust
Permutation::from_vec(vec![2, 0, 1])
  → [[0, 2, 1]]
```

### 2. Partition Conjugation (Bijection)
```rust
Partition::new(vec![4, 3, 1])
  ↔ Partition::new(vec![3, 2, 2, 1])
```

### 3. Permutation Inverse (Bijection)
```rust
Permutation::from_vec(vec![2, 0, 1])
  ↔ Permutation::from_vec(vec![1, 2, 0])
```

### 4. Robinson-Schensted Correspondence (Planned)
```rust
Permutation → (Tableau, Tableau)  // Bijection between permutations and tableau pairs
```

## Usage Patterns

### Pattern 1: Simple Function Wrapper

Wrap an existing function as a map:

```rust
use rustmath_combinatorics::combinatorial_map::*;

register_map::<_, MyType, MyResult>(
    "my_function",
    "Description",
    false,
    None,
    |x: &MyType| Some(existing_function(x)),
);
```

### Pattern 2: Involution (Self-Inverse Bijection)

For operations where `f(f(x)) = x`:

```rust
// Conjugation, inverse, complement, etc.
register_combinatorial_map! {
    name: "conjugate",
    description: "Conjugate a partition",
    from: Partition => to: Partition,
    bijection: true,
    forward: |p| Some(p.conjugate()),
    inverse: |p| Some(p.conjugate())  // Same function!
}
```

### Pattern 3: Composition of Maps

Chain multiple maps together:

```rust
let step1: Option<TypeB> = apply_map("map1", &input);
let step2: Option<TypeC> = step1.and_then(|x| apply_map("map2", &x));
let step3: Option<TypeD> = step2.and_then(|x| apply_map("map3", &x));
```

### Pattern 4: Map Discovery

Find maps dynamically:

```rust
// Find all bijections from A to B
let bijections = find_maps_between::<TypeA, TypeB>()
    .into_iter()
    .filter(|m| m.is_bijection)
    .collect::<Vec<_>>();

// Apply the first available bijection
if let Some(map) = bijections.first() {
    let result = apply_map::<TypeA, TypeB>(&map.name, &input);
}
```

## Testing

Run the combinatorial map tests:

```bash
# Run all combinatorial_map tests
cargo test -p rustmath-combinatorics combinatorial_map

# Run specific test
cargo test -p rustmath-combinatorics test_bijection_property
```

All tests include:
- Basic map registration
- Bijection registration and inversion
- Finding maps by type
- Metadata queries
- Global registry operations
- Bijection properties (round-trip testing)

## Implementation Details

### Why Type Erasure?

The registry uses type erasure (`dyn Any`) to store maps of different types in a single collection. This allows:
- Heterogeneous collection of maps
- Runtime type checking
- Compile-time type safety in the public API

### Performance Considerations

- **Registration**: O(1) insertion with HashMap
- **Application**: O(1) lookup + function call + type downcasting
- **Querying**: O(1) for type-based queries (using TypeId index)
- **Thread safety**: Lock contention only during registration (typically at startup)

### Memory Layout

Each registered map consumes:
- Function pointer: ~16 bytes
- Metadata: ~200 bytes (strings, TypeIds, etc.)
- Index entries: ~32 bytes

Typical registry with 50 maps: ~10-15 KB

## Future Enhancements

Potential improvements:
1. **Automatic inverse detection**: Detect involutions automatically
2. **Map composition**: Define new maps as compositions of existing ones
3. **Category theory integration**: Verify composition laws
4. **Serialization**: Save/load map definitions
5. **JIT compilation**: Optimize frequently-used map chains
6. **Visualization**: Generate diagrams of map relationships

## References

### Classic Combinatorial Bijections

- Robinson-Schensted correspondence (permutations ↔ tableau pairs)
- RSK correspondence variants
- Partition conjugation
- Dyck path bijections
- Stanley's theory of combinatorial bijections

### Related Work

- SageMath's `CombinatorialMap` system
- Haskell's `Iso` type
- Lens/Prism abstractions in functional programming

## License

Part of the RustMath project, licensed under the same terms.
