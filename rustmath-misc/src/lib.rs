//! # RustMath Miscellaneous Utilities
//!
//! This crate provides miscellaneous utility functions and structures,
//! corresponding to SageMath's sage.misc module.
//!
//! ## Modules
//!
//! - `sagedoc`: Documentation utilities
//! - `sagedoc_conf`: Documentation configuration
//! - `sageinspect`: Inspection utilities
//! - `search`: Search functionality
//! - `session`: Session management
//! - `sh`: Shell utilities
//! - `sphinxify`: Sphinx documentation generation
//! - `stopgap`: Stopgap warnings for incomplete features
//! - `superseded`: Deprecation warnings
//! - `table`: Table formatting
//! - `temporary_file`: Temporary file management
//! - `test_nested_class`: Testing utilities for nested classes
//! - `trace`: Tracing and debugging utilities
//! - `unknown`: Unknown object handling
//! - `verbose`: Verbose output control
//! - `viewer`: Viewing utilities
//! - `weak_dict`: Weak reference dictionaries
//! - `mrange`: Multi-dimensional range iterators and Cartesian products

pub mod mrange;
pub mod sagedoc;
pub mod sagedoc_conf;
pub mod sageinspect;
pub mod search;
pub mod session;
pub mod sh;
pub mod sphinxify;
pub mod stopgap;
pub mod superseded;
pub mod table;
pub mod temporary_file;
pub mod test_nested_class;
pub mod trace;
pub mod unknown;
pub mod verbose;
pub mod viewer;
pub mod weak_dict;

// Re-export commonly used utilities
pub use table::Table;
pub use verbose::{set_verbose, get_verbose};
pub use temporary_file::{TemporaryFile, TemporaryDir, tmp_filename, tmp_dir, atomic_write, atomic_dir};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_misc_imports() {
        // Compilation test - verify modules exist
    }
}
