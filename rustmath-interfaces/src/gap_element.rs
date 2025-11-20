//! GAP Element Types - Type-safe wrappers around GAP objects
//!
//! This module provides Rust types that wrap GAP elements, providing type-safe
//! access to GAP's computational objects. Each type corresponds to a GAP data type
//! and provides appropriate operations.
//!
//! # Overview
//!
//! GAP has a rich type system including:
//! - Basic types: Boolean, Integer, Rational, Float, String
//! - Algebraic types: FiniteField, IntegerMod, Cyclotomic
//! - Container types: List, Record
//! - Computational types: Permutation, Function
//!
//! This module wraps these types in Rust structs that maintain a reference
//! to the GAP process and the GAP variable name.
//!
//! # Architecture
//!
//! Since RustMath uses process-based GAP communication (not libGAP), each
//! GapElement maintains:
//! - A reference to the GAP interface
//! - A unique variable name in the GAP session
//! - Type-specific parsing and conversion methods
//!
//! # Example
//!
//! ```rust,ignore
//! use rustmath_interfaces::gap_element::*;
//! use rustmath_interfaces::gap::GapInterface;
//!
//! let gap = GapInterface::new()?;
//!
//! // Create an integer element
//! let x = GapElement_Integer::new(&gap, 42)?;
//! assert_eq!(x.value()?, 42);
//!
//! // Create a list element
//! let list = GapElement_List::from_vec(&gap, vec![1, 2, 3])?;
//! assert_eq!(list.length()?, 3);
//! ```

use crate::gap::{GapError, GapInterface, Result};
use crate::gap_parser::{parse_boolean, parse_integer, parse_list, parse_permutation, parse_record, Permutation};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Counter for generating unique GAP variable names
static VAR_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Generate a unique GAP variable name
fn next_var_name() -> String {
    let id = VAR_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!("__rustmath_var_{}__", id)
}

/// Base trait for all GAP elements
///
/// This trait provides common functionality for all GAP element types,
/// including variable management and basic operations.
pub trait GapElement: Sized {
    /// Get the GAP variable name for this element
    fn var_name(&self) -> &str;

    /// Get a reference to the GAP interface
    fn gap(&self) -> &GapInterface;

    /// Get the GAP representation as a string
    fn gap_repr(&self) -> Result<String> {
        self.gap().execute(self.var_name())
    }

    /// Check if this element is equal to another in GAP
    fn gap_equals(&self, other: &Self) -> Result<bool> {
        let result = self.gap().execute(&format!("{} = {}", self.var_name(), other.var_name()))?;
        parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Delete the GAP variable (free memory)
    fn unbind(&self) -> Result<()> {
        self.gap().execute(&format!("Unbind({});", self.var_name()))?;
        Ok(())
    }
}

/// GAP Boolean element
#[derive(Clone)]
pub struct GapElement_Boolean {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Boolean {
    /// Create a new boolean element from a Rust bool
    pub fn new(gap: &GapInterface, value: bool) -> Result<Self> {
        let var_name = next_var_name();
        let gap_value = if value { "true" } else { "false" };
        gap.execute(&format!("{} := {};", var_name, gap_value))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create a boolean element from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Get the boolean value
    pub fn value(&self) -> Result<bool> {
        let result = self.gap.execute(&self.var_name)?;
        parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Logical NOT
    pub fn not(&self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := not {};", var_name, self.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Logical AND
    pub fn and(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} and {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Logical OR
    pub fn or(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} or {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }
}

impl GapElement for GapElement_Boolean {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Boolean {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Boolean({})", self.var_name)
    }
}

/// GAP Integer element
#[derive(Clone)]
pub struct GapElement_Integer {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Integer {
    /// Create a new integer element from a Rust i64
    pub fn new(gap: &GapInterface, value: i64) -> Result<Self> {
        let var_name = next_var_name();
        gap.execute(&format!("{} := {};", var_name, value))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Get the integer value
    pub fn value(&self) -> Result<i64> {
        let result = self.gap.execute(&self.var_name)?;
        parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Add two integers
    pub fn add(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} + {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Subtract two integers
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} - {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Multiply two integers
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} * {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Integer division (quotient)
    pub fn div(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := QuoInt({}, {});", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Modulo operation
    pub fn modulo(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := RemInt({}, {});", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Check if prime
    pub fn is_prime(&self) -> Result<bool> {
        let result = self.gap.execute(&format!("IsPrimeInt({})", self.var_name))?;
        parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Factorial
    pub fn factorial(&self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := Factorial({});", var_name, self.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }
}

impl GapElement for GapElement_Integer {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Integer({})", self.var_name)
    }
}

/// GAP Rational element (exact fractions)
#[derive(Clone)]
pub struct GapElement_Rational {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Rational {
    /// Create a new rational from numerator and denominator
    pub fn new(gap: &GapInterface, numerator: i64, denominator: i64) -> Result<Self> {
        if denominator == 0 {
            return Err(GapError::GapRuntimeError("Denominator cannot be zero".to_string()));
        }

        let var_name = next_var_name();
        gap.execute(&format!("{} := {}/{};", var_name, numerator, denominator))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Get numerator
    pub fn numerator(&self) -> Result<i64> {
        let result = self.gap.execute(&format!("NumeratorRat({})", self.var_name))?;
        parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Get denominator
    pub fn denominator(&self) -> Result<i64> {
        let result = self.gap.execute(&format!("DenominatorRat({})", self.var_name))?;
        parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Add two rationals
    pub fn add(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} + {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Multiply two rationals
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} * {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }
}

impl GapElement for GapElement_Rational {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Rational({})", self.var_name)
    }
}

/// GAP Float element (floating-point numbers)
#[derive(Clone)]
pub struct GapElement_Float {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Float {
    /// Create a new float element from a Rust f64
    pub fn new(gap: &GapInterface, value: f64) -> Result<Self> {
        let var_name = next_var_name();
        gap.execute(&format!("{} := Float({});", var_name, value))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Get the float value as a string (GAP representation)
    pub fn value(&self) -> Result<String> {
        self.gap.execute(&self.var_name)
    }
}

impl GapElement for GapElement_Float {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Float({})", self.var_name)
    }
}

/// GAP String element
#[derive(Clone)]
pub struct GapElement_String {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_String {
    /// Create a new string element from a Rust string
    pub fn new(gap: &GapInterface, value: &str) -> Result<Self> {
        let var_name = next_var_name();
        // Escape quotes in the string
        let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
        gap.execute(&format!("{} := \"{}\";", var_name, escaped))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Get the string value
    pub fn value(&self) -> Result<String> {
        let result = self.gap.execute(&self.var_name)?;
        // Remove quotes if present
        let trimmed = result.trim();
        if trimmed.starts_with('"') && trimmed.ends_with('"') {
            Ok(trimmed[1..trimmed.len()-1].to_string())
        } else {
            Ok(trimmed.to_string())
        }
    }

    /// Get the length of the string
    pub fn length(&self) -> Result<usize> {
        let result = self.gap.execute(&format!("Length({})", self.var_name))?;
        parse_integer(&result)
            .map(|n| n as usize)
            .map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Concatenate with another string
    pub fn concat(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := Concatenation({}, {});", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }
}

impl GapElement for GapElement_String {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_String {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_String({})", self.var_name)
    }
}

/// GAP List element (arrays/lists)
#[derive(Clone)]
pub struct GapElement_List {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_List {
    /// Create a new empty list
    pub fn new(gap: &GapInterface) -> Result<Self> {
        let var_name = next_var_name();
        gap.execute(&format!("{} := [];", var_name))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create a list from a vector of integers
    pub fn from_vec(gap: &GapInterface, values: Vec<i64>) -> Result<Self> {
        let var_name = next_var_name();
        let values_str = values.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        gap.execute(&format!("{} := [ {} ];", var_name, values_str))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Get the length of the list
    pub fn length(&self) -> Result<usize> {
        let result = self.gap.execute(&format!("Length({})", self.var_name))?;
        parse_integer(&result)
            .map(|n| n as usize)
            .map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Get an element at an index (1-indexed, GAP style)
    pub fn get(&self, index: usize) -> Result<String> {
        self.gap.execute(&format!("{}[{}]", self.var_name, index))
    }

    /// Set an element at an index (1-indexed, GAP style)
    pub fn set(&self, index: usize, value: &str) -> Result<()> {
        self.gap.execute(&format!("{}[{}] := {};", self.var_name, index, value))?;
        Ok(())
    }

    /// Append an element to the list
    pub fn append(&self, value: &str) -> Result<()> {
        self.gap.execute(&format!("Add({}, {});", self.var_name, value))?;
        Ok(())
    }

    /// Get the list as a vector of strings
    pub fn to_vec(&self) -> Result<Vec<String>> {
        let result = self.gap.execute(&self.var_name)?;
        parse_list(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }
}

impl GapElement for GapElement_List {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_List({})", self.var_name)
    }
}

/// GAP Record element (associative arrays/dictionaries)
#[derive(Clone)]
pub struct GapElement_Record {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Record {
    /// Create a new empty record
    pub fn new(gap: &GapInterface) -> Result<Self> {
        let var_name = next_var_name();
        gap.execute(&format!("{} := rec();", var_name))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Get a field value
    pub fn get(&self, field: &str) -> Result<String> {
        self.gap.execute(&format!("{}.{}", self.var_name, field))
    }

    /// Set a field value
    pub fn set(&self, field: &str, value: &str) -> Result<()> {
        self.gap.execute(&format!("{}.{} := {};", self.var_name, field, value))?;
        Ok(())
    }

    /// Get all field names
    pub fn fields(&self) -> Result<Vec<String>> {
        let result = self.gap.execute(&format!("RecNames({})", self.var_name))?;
        parse_list(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Convert to a HashMap
    pub fn to_hashmap(&self) -> Result<HashMap<String, String>> {
        let result = self.gap.execute(&self.var_name)?;
        parse_record(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }
}

impl GapElement for GapElement_Record {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Record {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Record({})", self.var_name)
    }
}

/// Iterator for record fields
pub struct GapElement_RecordIterator {
    fields: Vec<String>,
    current: usize,
}

impl GapElement_RecordIterator {
    /// Create a new record iterator
    pub fn new(record: &GapElement_Record) -> Result<Self> {
        let fields = record.fields()?;
        Ok(Self { fields, current: 0 })
    }
}

impl Iterator for GapElement_RecordIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.fields.len() {
            let field = self.fields[self.current].clone();
            self.current += 1;
            Some(field)
        } else {
            None
        }
    }
}

/// GAP Permutation element
#[derive(Clone)]
pub struct GapElement_Permutation {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Permutation {
    /// Create a new permutation from cycle notation
    pub fn from_cycles(gap: &GapInterface, cycles: Vec<Vec<usize>>) -> Result<Self> {
        let var_name = next_var_name();

        // Convert to GAP cycle notation
        let cycles_str = cycles.iter()
            .map(|cycle| {
                let nums = cycle.iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!("({})", nums)
            })
            .collect::<Vec<_>>()
            .join("");

        gap.execute(&format!("{} := {};", var_name, cycles_str))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Parse the permutation into a Permutation struct
    pub fn to_permutation(&self) -> Result<Permutation> {
        let result = self.gap.execute(&self.var_name)?;
        parse_permutation(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Get the order of the permutation
    pub fn order(&self) -> Result<usize> {
        let result = self.gap.execute(&format!("Order({})", self.var_name))?;
        parse_integer(&result)
            .map(|n| n as usize)
            .map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Compose two permutations
    pub fn compose(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} * {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Get the inverse permutation
    pub fn inverse(&self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {}^-1;", var_name, self.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }
}

impl GapElement for GapElement_Permutation {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Permutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Permutation({})", self.var_name)
    }
}

/// GAP Function element
#[derive(Clone)]
pub struct GapElement_Function {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Function {
    /// Create from an existing GAP variable or function name
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Call the function with arguments
    pub fn call(&self, args: &[&str]) -> Result<String> {
        let args_str = args.join(", ");
        self.gap.execute(&format!("{}({})", self.var_name, args_str))
    }

    /// Call the function with no arguments
    pub fn call_no_args(&self) -> Result<String> {
        self.gap.execute(&format!("{}()", self.var_name))
    }
}

impl GapElement for GapElement_Function {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Function({})", self.var_name)
    }
}

/// GAP Function Method Proxy (for accessing object methods)
#[derive(Clone)]
pub struct GapElement_MethodProxy {
    function: GapElement_Function,
    object_var: String,
}

impl GapElement_MethodProxy {
    /// Create a method proxy for an object
    pub fn new(gap: &GapInterface, object_var: String, method_name: String) -> Self {
        Self {
            function: GapElement_Function::from_var(gap, method_name),
            object_var,
        }
    }

    /// Call the method on the object
    pub fn call(&self, args: &[&str]) -> Result<String> {
        let mut full_args = vec![self.object_var.as_str()];
        full_args.extend_from_slice(args);
        self.function.call(&full_args)
    }
}

impl fmt::Debug for GapElement_MethodProxy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_MethodProxy({}.{})", self.object_var, self.function.var_name())
    }
}

/// GAP FiniteField element (elements of GF(p^n))
#[derive(Clone)]
pub struct GapElement_FiniteField {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_FiniteField {
    /// Create a finite field element
    pub fn new(gap: &GapInterface, field: &str, value: &str) -> Result<Self> {
        let var_name = next_var_name();
        gap.execute(&format!("{} := {} * One({});", var_name, value, field))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Add two finite field elements
    pub fn add(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} + {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Multiply two finite field elements
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} * {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Get multiplicative inverse
    pub fn inverse(&self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {}^-1;", var_name, self.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }
}

impl GapElement for GapElement_FiniteField {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_FiniteField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_FiniteField({})", self.var_name)
    }
}

/// GAP IntegerMod element (elements of Z/nZ)
#[derive(Clone)]
pub struct GapElement_IntegerMod {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_IntegerMod {
    /// Create an integer mod n element
    pub fn new(gap: &GapInterface, value: i64, modulus: i64) -> Result<Self> {
        let var_name = next_var_name();
        gap.execute(&format!("{} := ZmodnZObj({}, {});", var_name, value, modulus))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} + {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Multiply two elements
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} * {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }
}

impl GapElement for GapElement_IntegerMod {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_IntegerMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_IntegerMod({})", self.var_name)
    }
}

/// GAP Cyclotomic element (cyclotomic numbers)
#[derive(Clone)]
pub struct GapElement_Cyclotomic {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Cyclotomic {
    /// Create a cyclotomic element using E(n) (primitive nth root of unity)
    pub fn from_root_of_unity(gap: &GapInterface, n: usize, coefficient: i64) -> Result<Self> {
        let var_name = next_var_name();
        gap.execute(&format!("{} := {} * E({});", var_name, coefficient, n))?;

        Ok(Self {
            gap: Arc::new(gap.clone()),
            var_name,
        })
    }

    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Add two cyclotomic elements
    pub fn add(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} + {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Multiply two cyclotomic elements
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} * {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Get the complex conjugate
    pub fn conjugate(&self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := ComplexConjugate({});", var_name, self.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }
}

impl GapElement for GapElement_Cyclotomic {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Cyclotomic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Cyclotomic({})", self.var_name)
    }
}

/// GAP Ring element (general ring elements)
#[derive(Clone)]
pub struct GapElement_Ring {
    gap: Arc<GapInterface>,
    var_name: String,
}

impl GapElement_Ring {
    /// Create from an existing GAP variable
    pub fn from_var(gap: &GapInterface, var_name: String) -> Self {
        Self {
            gap: Arc::new(gap.clone()),
            var_name,
        }
    }

    /// Add two ring elements
    pub fn add(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} + {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Multiply two ring elements
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let var_name = next_var_name();
        self.gap.execute(&format!("{} := {} * {};", var_name, self.var_name, other.var_name))?;
        Ok(Self::from_var(&self.gap, var_name))
    }

    /// Check if element is zero
    pub fn is_zero(&self) -> Result<bool> {
        let result = self.gap.execute(&format!("IsZero({})", self.var_name))?;
        parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Check if element is one
    pub fn is_one(&self) -> Result<bool> {
        let result = self.gap.execute(&format!("IsOne({})", self.var_name))?;
        parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
    }
}

impl GapElement for GapElement_Ring {
    fn var_name(&self) -> &str {
        &self.var_name
    }

    fn gap(&self) -> &GapInterface {
        &self.gap
    }
}

impl fmt::Debug for GapElement_Ring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GapElement_Ring({})", self.var_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GAP
    fn test_boolean_element() {
        let gap = GapInterface::new().unwrap();
        let t = GapElement_Boolean::new(&gap, true).unwrap();
        assert_eq!(t.value().unwrap(), true);

        let f = GapElement_Boolean::new(&gap, false).unwrap();
        assert_eq!(f.value().unwrap(), false);

        let result = t.and(&f).unwrap();
        assert_eq!(result.value().unwrap(), false);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_integer_element() {
        let gap = GapInterface::new().unwrap();
        let x = GapElement_Integer::new(&gap, 42).unwrap();
        assert_eq!(x.value().unwrap(), 42);

        let y = GapElement_Integer::new(&gap, 8).unwrap();
        let sum = x.add(&y).unwrap();
        assert_eq!(sum.value().unwrap(), 50);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_list_element() {
        let gap = GapInterface::new().unwrap();
        let list = GapElement_List::from_vec(&gap, vec![1, 2, 3]).unwrap();
        assert_eq!(list.length().unwrap(), 3);

        list.append("4").unwrap();
        assert_eq!(list.length().unwrap(), 4);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_permutation_element() {
        let gap = GapInterface::new().unwrap();
        let perm = GapElement_Permutation::from_cycles(&gap, vec![vec![1, 2, 3]]).unwrap();
        assert_eq!(perm.order().unwrap(), 3);
    }
}
