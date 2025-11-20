//! GAP Interface Tests
//!
//! This module provides test functions for the GAP interface, including
//! tests for file I/O, process management, and basic functionality.
//!
//! # Overview
//!
//! These tests verify that:
//! - GAP processes can be spawned and managed
//! - Commands can be executed and results retrieved
//! - File I/O works correctly with GAP
//! - Error handling is robust
//!
//! # Note
//!
//! Most tests are marked with `#[ignore]` because they require GAP to be
//! installed. Run them explicitly with `cargo test -- --ignored`.

use crate::gap::{GapError, GapInterface, Result};
use std::fs;
use std::io::Write;

/// Test writing to a file from GAP
///
/// This test verifies that GAP can write to files and that the
/// RustMath interface can read those files correctly.
///
/// # Example
///
/// ```rust,ignore
/// use rustmath_interfaces::test::test_write_to_file;
///
/// test_write_to_file().unwrap();
/// ```
pub fn test_write_to_file() -> Result<()> {
    let gap = GapInterface::new()?;

    // Create a temporary file path
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("rustmath_gap_test.txt");
    let file_path_str = file_path.to_string_lossy();

    // Write to file from GAP
    gap.execute(&format!(
        "PrintTo(\"{}\", \"Hello from GAP!\\n\");",
        file_path_str
    ))?;

    // Read the file back
    let content = fs::read_to_string(&file_path)
        .map_err(|e| GapError::IoError(e))?;

    // Clean up
    let _ = fs::remove_file(&file_path);

    // Verify content
    if content.contains("Hello from GAP!") {
        Ok(())
    } else {
        Err(GapError::GapRuntimeError(
            format!("File content mismatch. Got: {}", content)
        ))
    }
}

/// Test appending to a file from GAP
pub fn test_append_to_file() -> Result<()> {
    let gap = GapInterface::new()?;

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("rustmath_gap_append_test.txt");
    let file_path_str = file_path.to_string_lossy();

    // Write initial content
    gap.execute(&format!(
        "PrintTo(\"{}\", \"Line 1\\n\");",
        file_path_str
    ))?;

    // Append content
    gap.execute(&format!(
        "AppendTo(\"{}\", \"Line 2\\n\");",
        file_path_str
    ))?;

    // Read back
    let content = fs::read_to_string(&file_path)
        .map_err(|e| GapError::IoError(e))?;

    // Clean up
    let _ = fs::remove_file(&file_path);

    // Verify
    if content.contains("Line 1") && content.contains("Line 2") {
        Ok(())
    } else {
        Err(GapError::GapRuntimeError(
            format!("File content mismatch. Got: {}", content)
        ))
    }
}

/// Test reading from a file in GAP
pub fn test_read_from_file() -> Result<()> {
    let gap = GapInterface::new()?;

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("rustmath_gap_read_test.g");
    let file_path_str = file_path.to_string_lossy();

    // Create a GAP script
    let mut file = fs::File::create(&file_path)
        .map_err(|e| GapError::IoError(e))?;
    writeln!(file, "test_var := 42;")
        .map_err(|e| GapError::IoError(e))?;

    // Read the file in GAP
    gap.execute(&format!("Read(\"{}\");", file_path_str))?;

    // Check if variable was set
    let result = gap.execute("test_var")?;

    // Clean up
    let _ = fs::remove_file(&file_path);

    // Verify
    if result.trim() == "42" {
        Ok(())
    } else {
        Err(GapError::GapRuntimeError(
            format!("Variable not set correctly. Got: {}", result)
        ))
    }
}

/// Test GAP process lifecycle
pub fn test_process_lifecycle() -> Result<()> {
    // Create a GAP process
    let gap = GapInterface::new()?;

    // Verify it's running
    if !gap.is_running() {
        return Err(GapError::ProcessNotRunning);
    }

    // Execute a command
    let result = gap.execute("2 + 2")?;
    if result.trim() != "4" {
        return Err(GapError::GapRuntimeError(
            format!("Unexpected result: {}", result)
        ));
    }

    // Terminate
    gap.terminate()?;

    Ok(())
}

/// Test error handling
pub fn test_error_handling() -> Result<()> {
    let gap = GapInterface::new()?;

    // Try to execute invalid GAP code
    let result = gap.execute("invalid_function_xyz()");

    // Should return an error
    match result {
        Err(GapError::GapRuntimeError(_)) => Ok(()),
        Err(e) => Err(e),
        Ok(_) => Err(GapError::GapRuntimeError(
            "Expected error but got success".to_string()
        )),
    }
}

/// Test batch execution
pub fn test_batch_execution() -> Result<()> {
    let gap = GapInterface::new()?;

    let commands = vec![
        "a := 10;",
        "b := 20;",
        "c := a + b;",
    ];

    gap.execute_batch(&commands)?;

    let result = gap.execute("c")?;

    if result.trim() == "30" {
        Ok(())
    } else {
        Err(GapError::GapRuntimeError(
            format!("Unexpected result: {}", result)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GAP
    fn test_write_file() {
        test_write_to_file().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_append_file() {
        test_append_to_file().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_read_file() {
        test_read_from_file().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_lifecycle() {
        test_process_lifecycle().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_errors() {
        test_error_handling().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_batch() {
        test_batch_execution().unwrap();
    }
}
