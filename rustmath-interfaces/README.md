# RustMath Interfaces

External system interfaces for mathematical computation.

## Overview

This crate provides interfaces to external computational systems, allowing RustMath to leverage their powerful algorithms and databases.

## Features

### GAP Interface

GAP (Groups, Algorithms, Programming) is a system for computational discrete algebra, with particular emphasis on computational group theory.

**Capabilities:**
- Process management: Spawn and manage GAP processes
- Command translation: Convert Rust operations to GAP syntax
- Result parsing: Parse GAP output back to Rust structures
- Group operations: Use GAP for advanced group computations
- Permutation algorithms: Leverage GAP's Schreier-Sims and other algorithms

**Example:**

```rust
use rustmath_interfaces::gap::GapInterface;
use rustmath_interfaces::gap_permutation::GapPermutationGroup;

// Create a GAP interface
let gap = GapInterface::new()?;

// Execute GAP commands
let order = gap.group_order("SymmetricGroup(5)")?;
assert_eq!(order, 120);

// Use high-level permutation group interface
let group = GapPermutationGroup::symmetric(5)?;
assert_eq!(group.order()?, 120);
assert!(group.is_transitive()?);

// Compute advanced properties
let (base, sgs) = group.base_and_strong_generators()?;
let orbit = group.orbit(1)?;
let stabilizer = group.stabilizer(1)?;
```

## Installation

### GAP Installation

To use the GAP interface, you need to have GAP installed:

- **Ubuntu/Debian**: `sudo apt-get install gap`
- **macOS**: `brew install gap`
- **Windows**: Download from https://www.gap-system.org/

The `gap` command must be available in your PATH.

### Cargo Features

```toml
[dependencies]
rustmath-interfaces = "0.1"
```

Features:
- `gap`: Enable GAP interface (requires GAP installation)

## Architecture

### Process Management

Each `GapInterface` instance spawns its own GAP process, preventing state conflicts. Processes are automatically cleaned up when dropped.

### Thread Safety

The GAP interface uses `Arc<Mutex<>>` for thread-safe concurrent access to the GAP process.

### Error Handling

Comprehensive error types:
- `ProcessStartError`: Failed to spawn GAP
- `CommandExecutionError`: Failed to execute command
- `ParseError`: Failed to parse GAP output
- `GapRuntimeError`: GAP returned an error

### Performance

- **Process Overhead**: Reuse `GapInterface` instances
- **IPC Latency**: Batch commands when possible with `execute_batch()`
- **Memory**: Terminate processes when done with `terminate()`

## Module Structure

- `gap`: Core GAP interface and process management
- `gap_parser`: Parse GAP output into Rust structures
- `gap_permutation`: High-level permutation group algorithms

## Testing

Tests that require GAP are marked with `#[ignore]` to allow compilation without GAP:

```bash
# Run all tests (requires GAP)
cargo test -p rustmath-interfaces -- --ignored

# Run tests that don't require GAP
cargo test -p rustmath-interfaces
```

## Future Enhancements

Planned interfaces:
- **PARI/GP**: Number theory computations
- **Singular**: Algebraic geometry
- **FLINT**: Fast integer arithmetic
- **GMP/MPFR**: Arbitrary precision
- **SageMath**: Full integration

## References

- GAP: https://www.gap-system.org/
- GAP Reference Manual: https://www.gap-system.org/Manuals/doc/ref/chap0.html
- GAP Tutorial: https://www.gap-system.org/Manuals/doc/tut/chap0.html

## License

GPL-2.0-or-later
