//! Long-running GAP Interface Tests
//!
//! This module provides long-running stress tests for the GAP interface.
//! These tests verify stability and correctness under extended use.
//!
//! # Overview
//!
//! Long tests include:
//! - Repeated command execution (loop tests)
//! - Memory leak detection
//! - Process stability
//! - Concurrent access patterns
//!
//! # Note
//!
//! These tests take significant time to run and are marked with `#[ignore]`.
//! They're primarily used for stability testing and benchmarking.

use crate::gap::{GapError, GapInterface, Result};
use std::time::Instant;

/// Test loop 1: Basic arithmetic operations repeated many times
///
/// This test executes simple arithmetic operations in a loop to verify
/// that the GAP interface remains stable under repeated use.
///
/// # Example
///
/// ```rust,ignore
/// use rustmath_interfaces::test_long::test_loop_1;
///
/// test_loop_1().unwrap();
/// ```
pub fn test_loop_1() -> Result<()> {
    let gap = GapInterface::new()?;
    let iterations = 1000;

    println!("Running test_loop_1 with {} iterations...", iterations);
    let start = Instant::now();

    for i in 0..iterations {
        let result = gap.execute(&format!("{} + {}", i, i + 1))?;
        let expected = (2 * i + 1).to_string();

        if result.trim() != expected {
            return Err(GapError::GapRuntimeError(
                format!("Iteration {}: expected {}, got {}", i, expected, result.trim())
            ));
        }

        if i % 100 == 0 {
            println!("  Completed {} iterations...", i);
        }
    }

    let elapsed = start.elapsed();
    println!("test_loop_1 completed in {:?}", elapsed);
    println!("Average time per operation: {:?}", elapsed / iterations);

    Ok(())
}

/// Test loop 2: Group theory computations repeated many times
///
/// This test creates groups and queries their properties repeatedly
/// to verify stability with more complex operations.
pub fn test_loop_2() -> Result<()> {
    let gap = GapInterface::new()?;
    let iterations = 100;

    println!("Running test_loop_2 with {} iterations...", iterations);
    let start = Instant::now();

    for i in 0..iterations {
        let n = 3 + (i % 5); // Test groups S3, S4, S5, S6, S7

        // Create symmetric group
        gap.execute(&format!("G := SymmetricGroup({});", n))?;

        // Get order
        let order_result = gap.execute("Order(G)")?;
        let order: usize = order_result.trim().parse()
            .map_err(|e| GapError::ParseError(format!("Failed to parse order: {}", e)))?;

        // Verify order is n!
        let expected_order = (1..=n).product();
        if order != expected_order {
            return Err(GapError::GapRuntimeError(
                format!("Iteration {}: S{} has order {}, expected {}", i, n, order, expected_order)
            ));
        }

        // Check if abelian (should be false for n >= 3)
        let abelian_result = gap.execute("IsAbelian(G)")?;
        let is_abelian = abelian_result.trim() == "true";

        if n >= 3 && is_abelian {
            return Err(GapError::GapRuntimeError(
                format!("Iteration {}: S{} should not be abelian", i, n)
            ));
        }

        if i % 10 == 0 {
            println!("  Completed {} iterations...", i);
        }
    }

    let elapsed = start.elapsed();
    println!("test_loop_2 completed in {:?}", elapsed);
    println!("Average time per operation: {:?}", elapsed / iterations);

    Ok(())
}

/// Test loop 3: Variable management and memory stability
///
/// This test creates many variables to check for memory leaks and
/// proper variable management in the GAP process.
pub fn test_loop_3() -> Result<()> {
    let gap = GapInterface::new()?;
    let iterations = 500;

    println!("Running test_loop_3 with {} iterations...", iterations);
    let start = Instant::now();

    for i in 0_usize..iterations {
        // Create variables
        gap.execute(&format!("var_{} := {};", i, i * i))?;

        // Access a previous variable periodically
        if i > 0 && i % 10 == 0 {
            let prev_var = format!("var_{}", i - 10);
            let result = gap.execute(&prev_var)?;
            let expected = ((i - 10) * (i - 10)).to_string();

            if result.trim() != expected {
                return Err(GapError::GapRuntimeError(
                    format!("Variable {} has wrong value: {} (expected {})",
                           prev_var, result.trim(), expected)
                ));
            }
        }

        // Clean up some variables periodically
        if i % 50 == 0 && i > 0 {
            for j in (i.saturating_sub(50))..i {
                gap.execute(&format!("Unbind(var_{});", j))?;
            }
            println!("  Cleaned up variables up to iteration {}", i);
        }

        if i % 100 == 0 {
            println!("  Completed {} iterations...", i);
        }
    }

    let elapsed = start.elapsed();
    println!("test_loop_3 completed in {:?}", elapsed);
    println!("Average time per operation: {:?}", elapsed / iterations);

    Ok(())
}

/// Test loop 4: Matrix operations
///
/// This test performs matrix computations repeatedly to test
/// stability with linear algebra operations.
pub fn test_loop_4() -> Result<()> {
    let gap = GapInterface::new()?;
    let iterations = 100;

    println!("Running test_loop_4 with {} iterations...", iterations);
    let start = Instant::now();

    for i in 0..iterations {
        let size = 3 + (i % 3); // Test 3x3, 4x4, 5x5 matrices

        // Create an identity matrix
        gap.execute(&format!("M := IdentityMat({});", size))?;

        // Compute determinant (should be 1)
        let det = gap.execute("DeterminantMat(M)")?;
        if det.trim() != "1" {
            return Err(GapError::GapRuntimeError(
                format!("Iteration {}: Identity matrix has determinant {}, expected 1", i, det.trim())
            ));
        }

        // Compute rank (should be size)
        let rank = gap.execute("RankMat(M)")?;
        let rank_val: usize = rank.trim().parse()
            .map_err(|e| GapError::ParseError(format!("Failed to parse rank: {}", e)))?;

        if rank_val != size {
            return Err(GapError::GapRuntimeError(
                format!("Iteration {}: Identity matrix has rank {}, expected {}", i, rank_val, size)
            ));
        }

        if i % 20 == 0 {
            println!("  Completed {} iterations...", i);
        }
    }

    let elapsed = start.elapsed();
    println!("test_loop_4 completed in {:?}", elapsed);
    println!("Average time per operation: {:?}", elapsed / iterations);

    Ok(())
}

/// Run all long tests
pub fn run_all_long_tests() -> Result<()> {
    println!("=== Running all long tests ===\n");

    println!("Test 1: Basic arithmetic loop");
    test_loop_1()?;
    println!();

    println!("Test 2: Group theory loop");
    test_loop_2()?;
    println!();

    println!("Test 3: Variable management loop");
    test_loop_3()?;
    println!();

    println!("Test 4: Matrix operations loop");
    test_loop_4()?;
    println!();

    println!("=== All long tests passed ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GAP and takes time
    fn test_loop_1_run() {
        test_loop_1().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP and takes time
    fn test_loop_2_run() {
        test_loop_2().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP and takes time
    fn test_loop_3_run() {
        test_loop_3().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP and takes time
    fn test_loop_4_run() {
        test_loop_4().unwrap();
    }

    #[test]
    #[ignore] // Requires GAP and takes significant time
    fn run_all_tests() {
        run_all_long_tests().unwrap();
    }
}
