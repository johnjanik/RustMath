//! RustMath Interfaces - External system interfaces for mathematical computation
//!
//! This crate provides interfaces to external computational systems,
//! allowing RustMath to leverage their powerful algorithms and databases.
//!
//! # Supported Interfaces
//!
//! ## GAP (Groups, Algorithms, Programming)
//!
//! GAP is a system for computational discrete algebra, with particular
//! emphasis on computational group theory. The GAP interface provides:
//!
//! - **Process Management**: Spawn and manage GAP processes
//! - **Command Translation**: Convert Rust operations to GAP syntax
//! - **Result Parsing**: Parse GAP output back to Rust structures
//! - **Group Operations**: Use GAP for advanced group computations
//! - **Permutation Algorithms**: Leverage GAP's Schreier-Sims and other algorithms
//!
//! # Features
//!
//! - `gap`: Enable GAP interface (requires GAP installation)
//!
//! # Examples
//!
//! ## Using GAP for Group Computations
//!
//! ```rust,ignore
//! use rustmath_interfaces::gap::GapInterface;
//! use rustmath_interfaces::gap_permutation::GapPermutationGroup;
//!
//! // Create a GAP interface
//! let gap = GapInterface::new()?;
//!
//! // Execute GAP commands
//! let s5 = gap.symmetric_group(5)?;
//! let order = gap.group_order("SymmetricGroup(5)")?;
//! assert_eq!(order, 120);
//!
//! // Use high-level permutation group interface
//! let group = GapPermutationGroup::symmetric(5)?;
//! assert_eq!(group.order()?, 120);
//! assert!(group.is_transitive()?);
//! ```
//!
//! ## Parsing GAP Output
//!
//! ```rust
//! use rustmath_interfaces::gap_parser::*;
//!
//! // Parse permutations
//! let perm = parse_permutation("(1,2,3)(4,5)").unwrap();
//! assert_eq!(perm.cycles.len(), 2);
//! assert_eq!(perm.order(), 6);
//!
//! // Parse lists
//! let list = parse_integer_list("[ 1, 2, 3, 4 ]").unwrap();
//! assert_eq!(list, vec![1, 2, 3, 4]);
//!
//! // Parse records
//! let record = parse_record("rec( order := 120 )").unwrap();
//! assert_eq!(record.get("order"), Some(&"120".to_string()));
//! ```
//!
//! # Installation Requirements
//!
//! To use the GAP interface, you need to have GAP installed on your system:
//!
//! - **Ubuntu/Debian**: `sudo apt-get install gap`
//! - **macOS**: `brew install gap`
//! - **Windows**: Download from <https://www.gap-system.org/>
//!
//! The `gap` command must be available in your PATH.
//!
//! # Architecture
//!
//! The GAP interface is designed with the following principles:
//!
//! 1. **Process Isolation**: Each GAP interface spawns its own GAP process,
//!    preventing state conflicts
//!
//! 2. **Thread Safety**: The GAP interface uses `Arc<Mutex<>>` for thread-safe
//!    concurrent access
//!
//! 3. **Error Handling**: Comprehensive error types for process, parsing, and
//!    runtime errors
//!
//! 4. **Graceful Degradation**: Tests are marked with `#[ignore]` to allow
//!    compilation without GAP installed
//!
//! # Performance Considerations
//!
//! - **Process Overhead**: Spawning GAP processes has overhead; reuse
//!   `GapInterface` instances when possible
//!
//! - **IPC Latency**: Communication with GAP happens through pipes, which
//!   has latency; batch commands when possible
//!
//! - **Memory**: GAP processes consume memory; terminate them when done
//!
//! # Future Enhancements
//!
//! Planned additions to the interfaces crate:
//!
//! - **PARI/GP**: Number theory computations
//! - **Singular**: Algebraic geometry and commutative algebra
//! - **FLINT**: Fast integer and polynomial arithmetic
//! - **GMP/MPFR**: Arbitrary precision arithmetic
//! - **SageMath**: Full SageMath integration
//!
//! # See Also
//!
//! - GAP: <https://www.gap-system.org/>
//! - GAP Reference Manual: <https://www.gap-system.org/Manuals/doc/ref/chap0.html>

pub mod gap;
pub mod gap_parser;
pub mod gap_permutation;

// Re-export main types for convenience
pub use gap::{GapError, GapInterface, GapProcess};
pub use gap_parser::{
    parse_boolean, parse_group, parse_integer, parse_integer_list, parse_list,
    parse_permutation, parse_record, GroupInfo, ParseError, Permutation,
};
pub use gap_permutation::GapPermutationGroup;

/// Version information for the interfaces crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if GAP is available on the system
///
/// This function attempts to spawn a GAP process to verify it's installed
/// and accessible.
///
/// # Returns
///
/// `true` if GAP is available, `false` otherwise
pub fn is_gap_available() -> bool {
    std::process::Command::new("gap")
        .arg("-q")
        .arg("-c")
        .arg("quit;")
        .output()
        .is_ok()
}

/// Get the GAP version string
///
/// Returns the version of GAP installed on the system, if available.
pub fn gap_version() -> Option<String> {
    let output = std::process::Command::new("gap")
        .arg("-q")
        .arg("-c")
        .arg("Print(GAPInfo.Version);")
        .output()
        .ok()?;

    String::from_utf8(output.stdout).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_gap_availability_check() {
        // This test doesn't require GAP, just checks the function works
        let _available = is_gap_available();
        // Don't assert - GAP may or may not be installed
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_gap_version() {
        let version = gap_version();
        assert!(version.is_some());
        println!("GAP version: {}", version.unwrap());
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_gap_is_available() {
        assert!(is_gap_available());
    }
}
