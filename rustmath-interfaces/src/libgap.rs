//! LibGAP Interface - High-level interface to GAP system
//!
//! This module provides a high-level interface to GAP that mimics SageMath's
//! libgap interface. While SageMath uses libGAP (a library interface), RustMath
//! uses process-based communication for better isolation and portability.
//!
//! # Overview
//!
//! The `Gap` struct provides a convenient singleton-like interface to GAP,
//! managing the GAP process lifecycle and providing methods for:
//! - Creating GAP elements
//! - Executing GAP code
//! - Accessing common GAP functions
//! - Managing GAP variables
//!
//! # Example
//!
//! ```rust,ignore
//! use rustmath_interfaces::libgap::Gap;
//!
//! let gap = Gap::new()?;
//!
//! // Execute GAP code
//! let result = gap.eval("2 + 2")?;
//!
//! // Create groups
//! let s5 = gap.SymmetricGroup(5)?;
//! let order = gap.Order(&s5)?;
//! ```

use crate::gap::{GapError, GapInterface, Result};
use crate::gap_element::*;
use std::sync::{Arc, Mutex};

/// The main Gap interface - a high-level interface to GAP
///
/// This struct provides a convenient way to interact with GAP,
/// similar to SageMath's libgap interface.
pub struct Gap {
    interface: Arc<Mutex<GapInterface>>,
}

impl Gap {
    /// Create a new Gap interface
    ///
    /// This spawns a GAP process and prepares it for use.
    pub fn new() -> Result<Self> {
        let interface = GapInterface::new()?;
        Ok(Self {
            interface: Arc::new(Mutex::new(interface)),
        })
    }

    /// Evaluate a GAP expression and return the result as a string
    ///
    /// # Arguments
    ///
    /// * `code` - GAP code to execute
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = gap.eval("2 + 2")?;
    /// assert_eq!(result, "4");
    /// ```
    pub fn eval(&self, code: &str) -> Result<String> {
        let interface = self.interface.lock()
            .map_err(|e| GapError::CommandExecutionError(format!("Lock error: {}", e)))?;
        interface.execute(code)
    }

    /// Get the underlying GapInterface
    fn interface(&self) -> Result<GapInterface> {
        let interface = self.interface.lock()
            .map_err(|e| GapError::CommandExecutionError(format!("Lock error: {}", e)))?;
        // Clone the Arc inside to get a new reference
        Ok(GapInterface::new()?)
    }

    // ========================================================================
    // Group Construction Functions
    // ========================================================================

    /// Create a symmetric group S_n
    pub fn SymmetricGroup(&self, n: usize) -> Result<String> {
        self.eval(&format!("SymmetricGroup({})", n))
    }

    /// Create an alternating group A_n
    pub fn AlternatingGroup(&self, n: usize) -> Result<String> {
        self.eval(&format!("AlternatingGroup({})", n))
    }

    /// Create a cyclic group Z_n
    pub fn CyclicGroup(&self, n: usize) -> Result<String> {
        self.eval(&format!("CyclicGroup({})", n))
    }

    /// Create a dihedral group D_n
    pub fn DihedralGroup(&self, n: usize) -> Result<String> {
        self.eval(&format!("DihedralGroup({})", n))
    }

    /// Create a group from generators
    pub fn Group(&self, generators: &[&str]) -> Result<String> {
        let gens = generators.join(", ");
        self.eval(&format!("Group({})", gens))
    }

    /// Create a free group
    pub fn FreeGroup(&self, n: usize) -> Result<String> {
        self.eval(&format!("FreeGroup({})", n))
    }

    /// Create a free abelian group
    pub fn FreeAbelianGroup(&self, n: usize) -> Result<String> {
        self.eval(&format!("AbelianGroup([0, 0, ...])", ))
    }

    // ========================================================================
    // Number Theory Functions
    // ========================================================================

    /// Compute factorial
    pub fn Factorial(&self, n: usize) -> Result<String> {
        self.eval(&format!("Factorial({})", n))
    }

    /// Check if prime
    pub fn IsPrime(&self, n: i64) -> Result<bool> {
        let result = self.eval(&format!("IsPrimeInt({})", n))?;
        Ok(result.trim() == "true")
    }

    /// Compute GCD
    pub fn Gcd(&self, a: i64, b: i64) -> Result<i64> {
        let result = self.eval(&format!("Gcd({}, {})", a, b))?;
        result.trim().parse().map_err(|e| GapError::ParseError(format!("Failed to parse GCD: {}", e)))
    }

    /// Compute LCM
    pub fn Lcm(&self, a: i64, b: i64) -> Result<i64> {
        let result = self.eval(&format!("Lcm({}, {})", a, b))?;
        result.trim().parse().map_err(|e| GapError::ParseError(format!("Failed to parse LCM: {}", e)))
    }

    /// Factor an integer
    pub fn FactorsInt(&self, n: i64) -> Result<String> {
        self.eval(&format!("FactorsInt({})", n))
    }

    /// Compute Euler's totient function
    pub fn Phi(&self, n: i64) -> Result<i64> {
        let result = self.eval(&format!("Phi({})", n))?;
        result.trim().parse().map_err(|e| GapError::ParseError(format!("Failed to parse Phi: {}", e)))
    }

    // ========================================================================
    // Combinatorics Functions
    // ========================================================================

    /// Compute binomial coefficient
    pub fn Binomial(&self, n: i64, k: i64) -> Result<i64> {
        let result = self.eval(&format!("Binomial({}, {})", n, k))?;
        result.trim().parse().map_err(|e| GapError::ParseError(format!("Failed to parse binomial: {}", e)))
    }

    /// Get partitions of n
    pub fn Partitions(&self, n: usize) -> Result<String> {
        self.eval(&format!("Partitions({})", n))
    }

    /// Count partitions of n
    pub fn NrPartitions(&self, n: usize) -> Result<i64> {
        let result = self.eval(&format!("NrPartitions({})", n))?;
        result.trim().parse().map_err(|e| GapError::ParseError(format!("Failed to parse partition count: {}", e)))
    }

    /// Get all permutations
    pub fn Permutations(&self, list: &str) -> Result<String> {
        self.eval(&format!("Permutations({})", list))
    }

    /// Get combinations
    pub fn Combinations(&self, list: &str, k: usize) -> Result<String> {
        self.eval(&format!("Combinations({}, {})", list, k))
    }

    // ========================================================================
    // Group Theory Functions
    // ========================================================================

    /// Get the order of a group or element
    pub fn Order(&self, obj: &str) -> Result<usize> {
        let result = self.eval(&format!("Order({})", obj))?;
        result.trim().parse().map_err(|e| GapError::ParseError(format!("Failed to parse order: {}", e)))
    }

    /// Check if a group is abelian
    pub fn IsAbelian(&self, group: &str) -> Result<bool> {
        let result = self.eval(&format!("IsAbelian({})", group))?;
        Ok(result.trim() == "true")
    }

    /// Check if a group is simple
    pub fn IsSimpleGroup(&self, group: &str) -> Result<bool> {
        let result = self.eval(&format!("IsSimpleGroup({})", group))?;
        Ok(result.trim() == "true")
    }

    /// Check if a group is solvable
    pub fn IsSolvableGroup(&self, group: &str) -> Result<bool> {
        let result = self.eval(&format!("IsSolvableGroup({})", group))?;
        Ok(result.trim() == "true")
    }

    /// Get the center of a group
    pub fn Center(&self, group: &str) -> Result<String> {
        self.eval(&format!("Center({})", group))
    }

    /// Get the derived subgroup
    pub fn DerivedSubgroup(&self, group: &str) -> Result<String> {
        self.eval(&format!("DerivedSubgroup({})", group))
    }

    /// Get generators of a group
    pub fn GeneratorsOfGroup(&self, group: &str) -> Result<String> {
        self.eval(&format!("GeneratorsOfGroup({})", group))
    }

    /// Compute conjugacy classes
    pub fn ConjugacyClasses(&self, group: &str) -> Result<String> {
        self.eval(&format!("ConjugacyClasses({})", group))
    }

    /// Get the character table
    pub fn CharacterTable(&self, group: &str) -> Result<String> {
        self.eval(&format!("CharacterTable({})", group))
    }

    /// Compute normalizer
    pub fn Normalizer(&self, group: &str, subgroup: &str) -> Result<String> {
        self.eval(&format!("Normalizer({}, {})", group, subgroup))
    }

    /// Compute centralizer
    pub fn Centralizer(&self, group: &str, element: &str) -> Result<String> {
        self.eval(&format!("Centralizer({}, {})", group, element))
    }

    /// Compute orbit
    pub fn Orbit(&self, group: &str, point: usize) -> Result<String> {
        self.eval(&format!("Orbit({}, {})", group, point))
    }

    /// Compute stabilizer
    pub fn Stabilizer(&self, group: &str, point: usize) -> Result<String> {
        self.eval(&format!("Stabilizer({}, {})", group, point))
    }

    // ========================================================================
    // Matrix Functions
    // ========================================================================

    /// Create a matrix
    pub fn Matrix(&self, entries: &str) -> Result<String> {
        self.eval(&format!("{}", entries))
    }

    /// Compute determinant
    pub fn DeterminantMat(&self, matrix: &str) -> Result<String> {
        self.eval(&format!("DeterminantMat({})", matrix))
    }

    /// Compute rank
    pub fn RankMat(&self, matrix: &str) -> Result<usize> {
        let result = self.eval(&format!("RankMat({})", matrix))?;
        result.trim().parse().map_err(|e| GapError::ParseError(format!("Failed to parse rank: {}", e)))
    }

    /// Transpose matrix
    pub fn TransposedMat(&self, matrix: &str) -> Result<String> {
        self.eval(&format!("TransposedMat({})", matrix))
    }

    /// Compute inverse
    pub fn Inverse(&self, matrix: &str) -> Result<String> {
        self.eval(&format!("Inverse({})", matrix))
    }

    // ========================================================================
    // List and Utility Functions
    // ========================================================================

    /// Get length
    pub fn Length(&self, obj: &str) -> Result<usize> {
        let result = self.eval(&format!("Length({})", obj))?;
        result.trim().parse().map_err(|e| GapError::ParseError(format!("Failed to parse length: {}", e)))
    }

    /// Sum of list
    pub fn Sum(&self, list: &str) -> Result<String> {
        self.eval(&format!("Sum({})", list))
    }

    /// Product of list
    pub fn Product(&self, list: &str) -> Result<String> {
        self.eval(&format!("Product({})", list))
    }

    /// Sort a list
    pub fn Sort(&self, list: &str) -> Result<()> {
        self.eval(&format!("Sort({});", list))?;
        Ok(())
    }

    /// Get unique elements
    pub fn Set(&self, list: &str) -> Result<String> {
        self.eval(&format!("Set({})", list))
    }

    // ========================================================================
    // Type Checking Functions
    // ========================================================================

    /// Check if object is an integer
    pub fn IsInt(&self, obj: &str) -> Result<bool> {
        let result = self.eval(&format!("IsInt({})", obj))?;
        Ok(result.trim() == "true")
    }

    /// Check if object is a list
    pub fn IsList(&self, obj: &str) -> Result<bool> {
        let result = self.eval(&format!("IsList({})", obj))?;
        Ok(result.trim() == "true")
    }

    /// Check if object is a permutation
    pub fn IsPerm(&self, obj: &str) -> Result<bool> {
        let result = self.eval(&format!("IsPerm({})", obj))?;
        Ok(result.trim() == "true")
    }

    /// Check if object is a matrix
    pub fn IsMatrix(&self, obj: &str) -> Result<bool> {
        let result = self.eval(&format!("IsMatrix({})", obj))?;
        Ok(result.trim() == "true")
    }

    // ========================================================================
    // Help and Information
    // ========================================================================

    /// Get help for a GAP function
    pub fn help(&self, topic: &str) -> Result<String> {
        self.eval(&format!("?{}", topic))
    }

    /// Get the version of GAP
    pub fn version(&self) -> Result<String> {
        self.eval("GAPInfo.Version")
    }

    /// Check if GAP is running
    pub fn is_running(&self) -> bool {
        if let Ok(interface) = self.interface.lock() {
            interface.is_running()
        } else {
            false
        }
    }

    /// Terminate the GAP process
    pub fn quit(&self) -> Result<()> {
        let interface = self.interface.lock()
            .map_err(|e| GapError::CommandExecutionError(format!("Lock error: {}", e)))?;
        interface.terminate()
    }
}

impl Default for Gap {
    fn default() -> Self {
        Self::new().expect("Failed to create Gap interface")
    }
}

impl Drop for Gap {
    fn drop(&mut self) {
        let _ = self.quit();
    }
}

/// Global singleton instance (optional convenience)
static mut GLOBAL_GAP: Option<Gap> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Get the global Gap instance
///
/// This provides a singleton Gap instance for convenience.
/// Note: This is not thread-safe for initialization in some contexts.
pub fn gap() -> &'static Gap {
    unsafe {
        INIT.call_once(|| {
            GLOBAL_GAP = Some(Gap::new().expect("Failed to initialize global GAP instance"));
        });
        GLOBAL_GAP.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GAP
    fn test_gap_creation() {
        let gap = Gap::new().unwrap();
        assert!(gap.is_running());
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_eval() {
        let gap = Gap::new().unwrap();
        let result = gap.eval("2 + 2").unwrap();
        assert_eq!(result.trim(), "4");
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_symmetric_group() {
        let gap = Gap::new().unwrap();
        let s5 = gap.SymmetricGroup(5).unwrap();
        assert!(!s5.is_empty());

        let order = gap.Order("SymmetricGroup(5)").unwrap();
        assert_eq!(order, 120);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_number_theory() {
        let gap = Gap::new().unwrap();
        assert_eq!(gap.Gcd(12, 18).unwrap(), 6);
        assert_eq!(gap.Lcm(12, 18).unwrap(), 36);
        assert_eq!(gap.IsPrime(17).unwrap(), true);
        assert_eq!(gap.IsPrime(18).unwrap(), false);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_combinatorics() {
        let gap = Gap::new().unwrap();
        assert_eq!(gap.Binomial(10, 5).unwrap(), 252);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_group_properties() {
        let gap = Gap::new().unwrap();
        assert_eq!(gap.IsAbelian("CyclicGroup(5)").unwrap(), true);
        assert_eq!(gap.IsAbelian("SymmetricGroup(3)").unwrap(), false);
    }
}
