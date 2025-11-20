//! GAP Utility Functions
//!
//! This module provides utility functions for managing GAP objects,
//! tracking object lifecycle, and handling memory management.
//!
//! # Overview
//!
//! The utility module provides:
//! - `ObjWrapper`: A wrapper for GAP objects that tracks their lifecycle
//! - Object tracking: Keep track of all GAP objects created from Rust
//! - Memory management: Utilities for cleaning up GAP variables
//! - Debugging: Functions to inspect the current GAP state
//!
//! # Example
//!
//! ```rust,ignore
//! use rustmath_interfaces::util::*;
//! use rustmath_interfaces::gap::GapInterface;
//!
//! let gap = GapInterface::new()?;
//! let obj = ObjWrapper::new(&gap, "SymmetricGroup(5)")?;
//!
//! // Object is automatically tracked
//! let owned = get_owned_objects();
//! println!("Currently tracking {} objects", owned.len());
//! ```

use crate::gap::{GapError, GapInterface, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Global registry of owned GAP objects
static OWNED_OBJECTS: Mutex<Option<HashMap<String, ObjInfo>>> = Mutex::new(None);

/// Information about an owned GAP object
#[derive(Debug, Clone)]
pub struct ObjInfo {
    /// Variable name in GAP
    pub var_name: String,
    /// Type description (if known)
    pub type_desc: Option<String>,
    /// Whether the object is still valid
    pub valid: bool,
}

/// Initialize the object registry
fn ensure_registry() {
    let mut registry = OWNED_OBJECTS.lock().unwrap();
    if registry.is_none() {
        *registry = Some(HashMap::new());
    }
}

/// Register a GAP object in the tracking system
fn register_object(var_name: String, type_desc: Option<String>) {
    ensure_registry();
    let mut registry = OWNED_OBJECTS.lock().unwrap();
    if let Some(ref mut map) = *registry {
        map.insert(var_name.clone(), ObjInfo {
            var_name,
            type_desc,
            valid: true,
        });
    }
}

/// Unregister a GAP object from the tracking system
fn unregister_object(var_name: &str) {
    let mut registry = OWNED_OBJECTS.lock().unwrap();
    if let Some(ref mut map) = *registry {
        map.remove(var_name);
    }
}

/// Mark an object as invalid (but keep it in the registry)
fn invalidate_object(var_name: &str) {
    let mut registry = OWNED_OBJECTS.lock().unwrap();
    if let Some(ref mut map) = *registry {
        if let Some(obj) = map.get_mut(var_name) {
            obj.valid = false;
        }
    }
}

/// Get all currently tracked objects
///
/// This returns a list of all GAP objects that have been created and are
/// being tracked by the Rust interface.
///
/// # Example
///
/// ```rust,ignore
/// use rustmath_interfaces::util::get_owned_objects;
///
/// let objects = get_owned_objects();
/// for obj in objects {
///     println!("Object: {} (valid: {})", obj.var_name, obj.valid);
/// }
/// ```
pub fn get_owned_objects() -> Vec<ObjInfo> {
    ensure_registry();
    let registry = OWNED_OBJECTS.lock().unwrap();
    if let Some(ref map) = *registry {
        map.values().cloned().collect()
    } else {
        Vec::new()
    }
}

/// Get the number of tracked objects
pub fn num_owned_objects() -> usize {
    ensure_registry();
    let registry = OWNED_OBJECTS.lock().unwrap();
    if let Some(ref map) = *registry {
        map.len()
    } else {
        0
    }
}

/// Clear all tracked objects
///
/// This removes all objects from the tracking system but does not
/// delete them from GAP.
pub fn clear_object_registry() {
    let mut registry = OWNED_OBJECTS.lock().unwrap();
    if let Some(ref mut map) = *registry {
        map.clear();
    }
}

/// A wrapper for GAP objects that manages their lifecycle
///
/// `ObjWrapper` wraps a GAP variable name and provides automatic
/// cleanup when dropped. It also integrates with the object tracking
/// system.
///
/// # Example
///
/// ```rust,ignore
/// use rustmath_interfaces::util::ObjWrapper;
/// use rustmath_interfaces::gap::GapInterface;
///
/// let gap = GapInterface::new()?;
/// {
///     let obj = ObjWrapper::new(&gap, "SymmetricGroup(5)")?;
///     // Use obj...
/// } // obj is automatically cleaned up here
/// ```
pub struct ObjWrapper {
    gap: Arc<GapInterface>,
    var_name: String,
    auto_cleanup: bool,
}

impl ObjWrapper {
    /// Create a new ObjWrapper by executing GAP code
    ///
    /// # Arguments
    ///
    /// * `gap` - The GAP interface
    /// * `code` - GAP code to execute
    ///
    /// # Returns
    ///
    /// A new ObjWrapper that references the result
    pub fn new(gap: &GapInterface, code: &str) -> Result<Self> {
        // Generate a unique variable name
        let var_name = format!("__rustmath_obj_{}__",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos());

        // Execute the code and assign to variable
        gap.execute(&format!("{} := {};", var_name, code))?;

        // Register the object
        register_object(var_name.clone(), None);

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
            auto_cleanup: true,
        })
    }

    /// Create an ObjWrapper from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        register_object(var_name.clone(), None);

        Self {
            gap: Arc::new(gap.clone()),
            var_name,
            auto_cleanup: false, // Don't delete existing variables
        }
    }

    /// Get the variable name
    pub fn var_name(&self) -> &str {
        &self.var_name
    }

    /// Get the GAP interface
    pub fn gap(&self) -> &GapInterface {
        &self.gap
    }

    /// Execute a method on this object
    pub fn method(&self, method_name: &str, args: &[&str]) -> Result<String> {
        let args_str = if args.is_empty() {
            self.var_name.clone()
        } else {
            let mut all_args = vec![self.var_name.as_str()];
            all_args.extend_from_slice(args);
            all_args.join(", ")
        };

        self.gap.execute(&format!("{}({})", method_name, args_str))
    }

    /// Get the value as a string
    pub fn value(&self) -> Result<String> {
        self.gap.execute(&self.var_name)
    }

    /// Get the type of this object
    pub fn gap_type(&self) -> Result<String> {
        self.gap.execute(&format!("TypeObj({})", self.var_name))
    }

    /// Check if this object is bound in GAP
    pub fn is_bound(&self) -> Result<bool> {
        let result = self.gap.execute(&format!("IsBound({})", self.var_name))?;
        Ok(result.trim() == "true")
    }

    /// Unbind this object from GAP (delete it)
    pub fn unbind(&mut self) -> Result<()> {
        self.gap.execute(&format!("Unbind({});", self.var_name))?;
        invalidate_object(&self.var_name);
        Ok(())
    }

    /// Disable automatic cleanup
    pub fn disable_auto_cleanup(&mut self) {
        self.auto_cleanup = false;
    }

    /// Enable automatic cleanup
    pub fn enable_auto_cleanup(&mut self) {
        self.auto_cleanup = true;
    }
}

impl Drop for ObjWrapper {
    fn drop(&mut self) {
        if self.auto_cleanup {
            let _ = self.unbind();
        }
        unregister_object(&self.var_name);
    }
}

impl Clone for ObjWrapper {
    fn clone(&self) -> Self {
        // Cloning creates a reference to the same GAP object
        // Don't auto-cleanup clones to avoid double-free
        Self {
            gap: self.gap.clone(),
            var_name: self.var_name.clone(),
            auto_cleanup: false,
        }
    }
}

impl std::fmt::Debug for ObjWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjWrapper({})", self.var_name)
    }
}

/// Clean up all tracked objects in GAP
///
/// This unbinds all tracked GAP variables, freeing memory.
/// Use with caution as this will invalidate all existing ObjWrappers.
pub fn cleanup_all_objects(gap: &GapInterface) -> Result<()> {
    let objects = get_owned_objects();

    for obj in objects {
        if obj.valid {
            let _ = gap.execute(&format!("Unbind({});", obj.var_name));
        }
    }

    clear_object_registry();
    Ok(())
}

/// Get statistics about tracked objects
pub fn object_stats() -> HashMap<String, usize> {
    let objects = get_owned_objects();
    let mut stats = HashMap::new();

    stats.insert("total".to_string(), objects.len());
    stats.insert("valid".to_string(), objects.iter().filter(|o| o.valid).count());
    stats.insert("invalid".to_string(), objects.iter().filter(|o| !o.valid).count());

    stats
}

/// Print debug information about all tracked objects
pub fn debug_print_objects() {
    let objects = get_owned_objects();
    println!("=== Tracked GAP Objects ===");
    println!("Total: {}", objects.len());

    for (i, obj) in objects.iter().enumerate() {
        println!("  {}: {} (valid: {}, type: {:?})",
                 i + 1,
                 obj.var_name,
                 obj.valid,
                 obj.type_desc);
    }

    let stats = object_stats();
    println!("Statistics:");
    println!("  Valid:   {}", stats.get("valid").unwrap_or(&0));
    println!("  Invalid: {}", stats.get("invalid").unwrap_or(&0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_registry() {
        clear_object_registry();

        register_object("test_var_1".to_string(), Some("Group".to_string()));
        register_object("test_var_2".to_string(), None);

        let objects = get_owned_objects();
        assert_eq!(objects.len(), 2);

        unregister_object("test_var_1");
        let objects = get_owned_objects();
        assert_eq!(objects.len(), 1);

        clear_object_registry();
        let objects = get_owned_objects();
        assert_eq!(objects.len(), 0);
    }

    #[test]
    fn test_object_stats() {
        clear_object_registry();

        register_object("var1".to_string(), None);
        register_object("var2".to_string(), None);
        invalidate_object("var2");

        let stats = object_stats();
        assert_eq!(stats.get("total"), Some(&2));
        assert_eq!(stats.get("valid"), Some(&1));
        assert_eq!(stats.get("invalid"), Some(&1));

        clear_object_registry();
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_obj_wrapper() {
        let gap = GapInterface::new().unwrap();

        {
            let obj = ObjWrapper::new(&gap, "SymmetricGroup(5)").unwrap();
            assert!(obj.is_bound().unwrap());

            let value = obj.value().unwrap();
            assert!(!value.is_empty());
        }

        // Object should be cleaned up now
        assert_eq!(num_owned_objects(), 0);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_obj_wrapper_method() {
        let gap = GapInterface::new().unwrap();
        let obj = ObjWrapper::new(&gap, "SymmetricGroup(5)").unwrap();

        let order = obj.method("Order", &[]).unwrap();
        assert_eq!(order.trim(), "120");
    }
}
