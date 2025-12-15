//! RustMath Jupyter Kernel
//!
//! A native Rust implementation of the Jupyter kernel protocol for RustMath.
//! This kernel allows interactive mathematical computation in Jupyter notebooks.

pub mod protocol;
pub mod connection;
pub mod kernel;
pub mod handlers;
pub mod repl;
pub mod install;

pub use kernel::RustMathKernel;
pub use install::install_kernel;
