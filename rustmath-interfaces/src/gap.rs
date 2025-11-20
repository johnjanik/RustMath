//! GAP Interface - Interface to the GAP system for computational group theory
//!
//! This module provides a bridge between RustMath and GAP (Groups, Algorithms, Programming),
//! allowing RustMath to leverage GAP's powerful algorithms for group theory computations.
//!
//! # Overview
//!
//! The GAP interface consists of:
//! - Process management: Spawning and managing GAP processes
//! - Command translation: Converting Rust operations to GAP syntax
//! - Result parsing: Parsing GAP output back to Rust structures
//! - Group operations: Using GAP for group computations
//! - Permutation algorithms: Leveraging GAP's permutation group algorithms
//!
//! # Example
//!
//! ```rust,ignore
//! use rustmath_interfaces::gap::GapInterface;
//!
//! // Create a GAP interface
//! let gap = GapInterface::new()?;
//!
//! // Execute a GAP command
//! let result = gap.execute("SymmetricGroup(5)")?;
//!
//! // Get group order
//! let order = gap.group_order(&result)?;
//! assert_eq!(order, 120);
//! ```

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Errors that can occur when interfacing with GAP
#[derive(Error, Debug)]
pub enum GapError {
    #[error("Failed to start GAP process: {0}")]
    ProcessStartError(String),

    #[error("Failed to execute GAP command: {0}")]
    CommandExecutionError(String),

    #[error("Failed to parse GAP output: {0}")]
    ParseError(String),

    #[error("GAP process is not running")]
    ProcessNotRunning,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("GAP command timed out")]
    Timeout,

    #[error("GAP returned an error: {0}")]
    GapRuntimeError(String),
}

pub type Result<T> = std::result::Result<T, GapError>;

/// A handle to a GAP process
///
/// This struct manages a running GAP process, allowing commands to be sent
/// and results to be retrieved.
pub struct GapProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    command_counter: usize,
}

impl GapProcess {
    /// Start a new GAP process
    ///
    /// This spawns a GAP process in interactive mode, ready to accept commands.
    pub fn new() -> Result<Self> {
        Self::with_args(&["-q"]) // -q for quiet mode (no banner)
    }

    /// Start a GAP process with custom arguments
    pub fn with_args(args: &[&str]) -> Result<Self> {
        let mut child = Command::new("gap")
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| GapError::ProcessStartError(e.to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| GapError::ProcessStartError("Failed to capture stdin".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| GapError::ProcessStartError("Failed to capture stdout".to_string()))?;

        let stdout = BufReader::new(stdout);

        Ok(Self {
            child,
            stdin,
            stdout,
            command_counter: 0,
        })
    }

    /// Execute a GAP command and return the result
    ///
    /// # Arguments
    ///
    /// * `command` - The GAP command to execute
    ///
    /// # Returns
    ///
    /// The output from GAP as a string
    pub fn execute(&mut self, command: &str) -> Result<String> {
        // Send command to GAP
        writeln!(self.stdin, "{}", command)?;
        self.stdin.flush()?;

        // Add a marker to detect end of output
        let marker = format!("__RUSTMATH_MARKER_{}__", self.command_counter);
        self.command_counter += 1;
        writeln!(self.stdin, "Print(\"{}\");", marker)?;
        self.stdin.flush()?;

        // Read output until we see the marker
        let mut output = String::new();
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = self.stdout.read_line(&mut line)?;

            if bytes_read == 0 {
                return Err(GapError::ProcessNotRunning);
            }

            if line.contains(&marker) {
                break;
            }

            // Filter out GAP prompts (gap> or >)
            let trimmed = line.trim();
            if !trimmed.starts_with("gap>") && !trimmed.starts_with('>') {
                output.push_str(&line);
            }
        }

        // Check for error messages in output
        if output.contains("Error,") || output.contains("Syntax error") {
            return Err(GapError::GapRuntimeError(output.trim().to_string()));
        }

        Ok(output.trim().to_string())
    }

    /// Execute multiple GAP commands
    pub fn execute_batch(&mut self, commands: &[&str]) -> Result<Vec<String>> {
        commands.iter().map(|cmd| self.execute(cmd)).collect()
    }

    /// Check if the GAP process is still running
    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().ok().flatten().is_none()
    }

    /// Terminate the GAP process
    pub fn terminate(&mut self) -> Result<()> {
        writeln!(self.stdin, "quit;")?;
        self.stdin.flush()?;
        self.child.wait()?;
        Ok(())
    }
}

impl Drop for GapProcess {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}

/// High-level interface to GAP for group computations
///
/// This provides a convenient API for using GAP to perform group theory
/// computations. It manages a GAP process internally.
pub struct GapInterface {
    process: Arc<Mutex<GapProcess>>,
}

impl GapInterface {
    /// Create a new GAP interface
    pub fn new() -> Result<Self> {
        let process = GapProcess::new()?;
        Ok(Self {
            process: Arc::new(Mutex::new(process)),
        })
    }

    /// Execute a GAP command
    pub fn execute(&self, command: &str) -> Result<String> {
        let mut process = self
            .process
            .lock()
            .map_err(|e| GapError::CommandExecutionError(format!("Lock error: {}", e)))?;
        process.execute(command)
    }

    /// Execute multiple GAP commands
    pub fn execute_batch(&self, commands: &[&str]) -> Result<Vec<String>> {
        let mut process = self
            .process
            .lock()
            .map_err(|e| GapError::CommandExecutionError(format!("Lock error: {}", e)))?;
        process.execute_batch(commands)
    }

    /// Create a symmetric group S_n in GAP
    pub fn symmetric_group(&self, n: usize) -> Result<String> {
        self.execute(&format!("SymmetricGroup({})", n))
    }

    /// Create an alternating group A_n in GAP
    pub fn alternating_group(&self, n: usize) -> Result<String> {
        self.execute(&format!("AlternatingGroup({})", n))
    }

    /// Create a cyclic group Z_n in GAP
    pub fn cyclic_group(&self, n: usize) -> Result<String> {
        self.execute(&format!("CyclicGroup({})", n))
    }

    /// Create a dihedral group D_n in GAP
    pub fn dihedral_group(&self, n: usize) -> Result<String> {
        self.execute(&format!("DihedralGroup({})", n))
    }

    /// Get the order of a group
    pub fn group_order(&self, group: &str) -> Result<usize> {
        let result = self.execute(&format!("Order({})", group))?;
        result
            .trim()
            .parse()
            .map_err(|e| GapError::ParseError(format!("Failed to parse order: {}", e)))
    }

    /// Get the generators of a group
    pub fn generators(&self, group: &str) -> Result<String> {
        self.execute(&format!("GeneratorsOfGroup({})", group))
    }

    /// Check if a group is abelian
    pub fn is_abelian(&self, group: &str) -> Result<bool> {
        let result = self.execute(&format!("IsAbelian({})", group))?;
        Ok(result.trim() == "true")
    }

    /// Check if a group is simple
    pub fn is_simple(&self, group: &str) -> Result<bool> {
        let result = self.execute(&format!("IsSimpleGroup({})", group))?;
        Ok(result.trim() == "true")
    }

    /// Check if a group is solvable
    pub fn is_solvable(&self, group: &str) -> Result<bool> {
        let result = self.execute(&format!("IsSolvableGroup({})", group))?;
        Ok(result.trim() == "true")
    }

    /// Get the center of a group
    pub fn center(&self, group: &str) -> Result<String> {
        self.execute(&format!("Center({})", group))
    }

    /// Get the derived subgroup (commutator subgroup)
    pub fn derived_subgroup(&self, group: &str) -> Result<String> {
        self.execute(&format!("DerivedSubgroup({})", group))
    }

    /// Compute conjugacy classes
    pub fn conjugacy_classes(&self, group: &str) -> Result<String> {
        self.execute(&format!("ConjugacyClasses({})", group))
    }

    /// Get the number of conjugacy classes
    pub fn num_conjugacy_classes(&self, group: &str) -> Result<usize> {
        let result = self.execute(&format!("Size(ConjugacyClasses({}))", group))?;
        result
            .trim()
            .parse()
            .map_err(|e| GapError::ParseError(format!("Failed to parse size: {}", e)))
    }

    /// Compute the character table
    pub fn character_table(&self, group: &str) -> Result<String> {
        self.execute(&format!("CharacterTable({})", group))
    }

    /// Check if the GAP process is running
    pub fn is_running(&self) -> bool {
        if let Ok(mut process) = self.process.lock() {
            process.is_running()
        } else {
            false
        }
    }

    /// Terminate the GAP process
    pub fn terminate(&self) -> Result<()> {
        let mut process = self
            .process
            .lock()
            .map_err(|e| GapError::CommandExecutionError(format!("Lock error: {}", e)))?;
        process.terminate()
    }
}

impl Default for GapInterface {
    fn default() -> Self {
        Self::new().expect("Failed to create GAP interface")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GAP to be installed
    fn test_gap_process_creation() {
        let result = GapProcess::new();
        assert!(result.is_ok(), "Failed to create GAP process");
    }

    #[test]
    #[ignore] // Requires GAP to be installed
    fn test_basic_arithmetic() {
        let mut gap = GapProcess::new().unwrap();
        let result = gap.execute("2 + 2;").unwrap();
        assert_eq!(result.trim(), "4");
    }

    #[test]
    #[ignore] // Requires GAP to be installed
    fn test_symmetric_group() {
        let gap = GapInterface::new().unwrap();
        let s5 = gap.symmetric_group(5).unwrap();
        assert!(!s5.is_empty());

        let order = gap.group_order("SymmetricGroup(5)").unwrap();
        assert_eq!(order, 120);
    }

    #[test]
    #[ignore] // Requires GAP to be installed
    fn test_alternating_group() {
        let gap = GapInterface::new().unwrap();
        let a5 = gap.alternating_group(5).unwrap();
        assert!(!a5.is_empty());

        let order = gap.group_order("AlternatingGroup(5)").unwrap();
        assert_eq!(order, 60);
    }

    #[test]
    #[ignore] // Requires GAP to be installed
    fn test_group_properties() {
        let gap = GapInterface::new().unwrap();

        // Test abelian property
        assert!(gap.is_abelian("CyclicGroup(5)").unwrap());
        assert!(!gap.is_abelian("SymmetricGroup(3)").unwrap());

        // Test simple property
        assert!(gap.is_simple("AlternatingGroup(5)").unwrap());
        assert!(!gap.is_simple("SymmetricGroup(4)").unwrap());
    }

    #[test]
    #[ignore] // Requires GAP to be installed
    fn test_conjugacy_classes() {
        let gap = GapInterface::new().unwrap();
        let num_classes = gap.num_conjugacy_classes("SymmetricGroup(4)").unwrap();
        assert_eq!(num_classes, 5); // S4 has 5 conjugacy classes
    }
}
