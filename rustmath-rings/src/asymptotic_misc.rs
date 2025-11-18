//! # Miscellaneous Utilities for Asymptotic Analysis
//!
//! This module provides various utility functions and classes that support asymptotic
//! analysis operations. These include:
//! - String parsing and representation utilities
//! - Error handling for special cases (O-zero, B-zero)
//! - Merging algorithms for sorted sequences
//! - Local variable management for nested scopes
//!
//! ## Key Components
//!
//! - **Locals**: Dictionary-like structure for managing local variables
//! - **WithLocals**: Trait for objects that maintain local variable scopes
//! - **Merging utilities**: Efficient algorithms for combining sorted sequences
//! - **String utilities**: Parsing and formatting for asymptotic expressions
//! - **Error types**: Special errors for asymptotic edge cases
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::asymptotic_misc::{Locals, bidirectional_merge_sorted};
//!
//! // Managing local variables
//! let mut locals = Locals::new();
//! locals.insert("x".to_string(), "value".to_string());
//!
//! // Merging sorted sequences
//! let seq1 = vec![1, 3, 5];
//! let seq2 = vec![2, 4, 6];
//! let merged = bidirectional_merge_sorted(&seq1, &seq2);
//! ```

use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::cmp::Ordering;

// ======================================================================================
// LOCALS (Local Variable Dictionary)
// ======================================================================================

/// A dictionary-like structure for managing local variables in asymptotic expressions.
///
/// This is used to maintain a mapping of variable names to their values or properties
/// within specific scopes during asymptotic computations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Locals {
    /// The underlying map from variable names to values
    data: HashMap<String, String>,
}

impl Locals {
    /// Creates a new empty Locals dictionary.
    pub fn new() -> Self {
        Locals {
            data: HashMap::new(),
        }
    }

    /// Creates a Locals dictionary from an existing HashMap.
    pub fn from_map(data: HashMap<String, String>) -> Self {
        Locals { data }
    }

    /// Inserts a key-value pair.
    pub fn insert(&mut self, key: String, value: String) -> Option<String> {
        self.data.insert(key, value)
    }

    /// Gets a value by key.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }

    /// Removes a key-value pair.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    /// Checks if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns an iterator over the key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.data.iter()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Merges another Locals into this one.
    pub fn merge(&mut self, other: &Locals) {
        for (k, v) in &other.data {
            self.data.insert(k.clone(), v.clone());
        }
    }
}

impl Default for Locals {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Locals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Locals{{")?;
        let mut first = true;
        for (k, v) in &self.data {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", k, v)?;
            first = false;
        }
        write!(f, "}}")
    }
}

// ======================================================================================
// WITH LOCALS (Trait for objects with local variable support)
// ======================================================================================

/// Trait for objects that maintain local variable scopes.
///
/// Objects implementing this trait can store and retrieve local variables,
/// which is useful for managing nested scopes in asymptotic expressions.
pub trait WithLocals {
    /// Returns a reference to the locals.
    fn locals(&self) -> &Locals;

    /// Returns a mutable reference to the locals.
    fn locals_mut(&mut self) -> &mut Locals;

    /// Sets a local variable.
    fn set_local(&mut self, key: String, value: String) {
        self.locals_mut().insert(key, value);
    }

    /// Gets a local variable.
    fn get_local(&self, key: &str) -> Option<&String> {
        self.locals().get(key)
    }

    /// Clears all local variables.
    fn clear_locals(&mut self) {
        self.locals_mut().clear();
    }
}

// ======================================================================================
// ERROR TYPES
// ======================================================================================

/// Error indicating that an operation resulted in O(0) (undefined behavior).
///
/// In asymptotic analysis, O(0) represents a function that is identically zero,
/// which can arise in certain operations where the result is not well-defined.
#[derive(Debug, Clone)]
pub struct NotImplementedOZero {
    message: String,
}

impl NotImplementedOZero {
    /// Creates a new NotImplementedOZero error.
    pub fn new(message: String) -> Self {
        NotImplementedOZero { message }
    }

    /// Returns the error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for NotImplementedOZero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NotImplementedOZero: {}", self.message)
    }
}

impl std::error::Error for NotImplementedOZero {}

/// Error indicating that an operation resulted in B(0) (undefined behavior).
///
/// B(0) represents the empty set in asymptotic notation, indicating that
/// certain operations cannot produce valid results.
#[derive(Debug, Clone)]
pub struct NotImplementedBZero {
    message: String,
}

impl NotImplementedBZero {
    /// Creates a new NotImplementedBZero error.
    pub fn new(message: String) -> Self {
        NotImplementedBZero { message }
    }

    /// Returns the error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for NotImplementedBZero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NotImplementedBZero: {}", self.message)
    }
}

impl std::error::Error for NotImplementedBZero {}

// ======================================================================================
// STRING UTILITIES
// ======================================================================================

/// Converts a parent object to its short string representation.
///
/// This is used for displaying growth groups, rings, and other algebraic
/// structures in a compact form.
///
/// # Arguments
/// * `parent_repr` - Full string representation of the parent
///
/// # Returns
/// A shortened representation suitable for display
pub fn parent_to_repr_short(parent_repr: &str) -> String {
    // Simplified: just take the first 50 characters or find a shorter representation
    if parent_repr.len() <= 50 {
        parent_repr.to_string()
    } else {
        // Extract the main type name if it follows a pattern like "SomeType(...)"
        if let Some(idx) = parent_repr.find('(') {
            parent_repr[..idx].to_string()
        } else {
            parent_repr[..50].to_string() + "..."
        }
    }
}

/// Converts a short representation back to a parent object reference.
///
/// This is the inverse operation of `parent_to_repr_short`.
///
/// # Arguments
/// * `repr_short` - The shortened representation
///
/// # Returns
/// The full parent representation (if recoverable)
pub fn repr_short_to_parent(repr_short: &str) -> String {
    // Simplified: just return the short representation
    // In a full implementation, this would use a registry of parent objects
    repr_short.to_string()
}

/// Splits a string by an operator while respecting parentheses.
///
/// This is used for parsing mathematical expressions.
///
/// # Arguments
/// * `s` - The string to split
/// * `op` - The operator to split on (e.g., "+", "*")
///
/// # Returns
/// A vector of substrings split by the operator
pub fn split_str_by_op(s: &str, op: char) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for c in s.chars() {
        match c {
            '(' | '[' | '{' => {
                current.push(c);
                depth += 1;
            }
            ')' | ']' | '}' => {
                current.push(c);
                depth -= 1;
            }
            _ if c == op && depth == 0 => {
                if !current.is_empty() {
                    result.push(current.trim().to_string());
                    current.clear();
                }
            }
            _ => current.push(c),
        }
    }

    if !current.is_empty() {
        result.push(current.trim().to_string());
    }

    result
}

/// Formats an operation for display.
///
/// # Arguments
/// * `left` - Left operand
/// * `op` - Operator symbol
/// * `right` - Right operand
///
/// # Returns
/// Formatted string representation
pub fn repr_op(left: &str, op: &str, right: &str) -> String {
    format!("({} {} {})", left, op, right)
}

/// Combines multiple exceptions into a single error message.
///
/// This is useful when multiple errors occur during a computation and
/// need to be reported together.
///
/// # Arguments
/// * `exceptions` - Vector of error messages
///
/// # Returns
/// A combined error message
pub fn combine_exceptions(exceptions: &[String]) -> String {
    if exceptions.is_empty() {
        return "No exceptions".to_string();
    }

    if exceptions.len() == 1 {
        return exceptions[0].clone();
    }

    let mut result = "Multiple exceptions:\n".to_string();
    for (i, exc) in exceptions.iter().enumerate() {
        result.push_str(&format!("  {}: {}\n", i + 1, exc));
    }
    result
}

/// Substitutes a value and raises an exception if the result is invalid.
///
/// This is used during symbolic substitution to catch undefined results.
///
/// # Arguments
/// * `expr` - The expression being substituted into
/// * `var` - The variable being substituted
/// * `value` - The value being substituted
///
/// # Returns
/// Result of the substitution or an error
pub fn substitute_raise_exception(
    expr: &str,
    var: &str,
    value: &str,
) -> Result<String, String> {
    // Simplified implementation: just perform basic substitution
    // In a full implementation, this would evaluate the expression
    let result = expr.replace(var, value);

    // Check for obviously invalid results
    if result.contains("1/0") || result.contains("0/0") {
        Err(format!("Invalid substitution: {} -> {}", var, value))
    } else {
        Ok(result)
    }
}

// ======================================================================================
// MERGING UTILITIES
// ======================================================================================

/// Merges two sorted sequences in a bidirectional manner.
///
/// This efficiently combines two sorted sequences into a single sorted sequence,
/// useful for maintaining sorted lists of asymptotic terms.
///
/// # Arguments
/// * `seq1` - First sorted sequence
/// * `seq2` - Second sorted sequence
///
/// # Returns
/// A merged sorted sequence
pub fn bidirectional_merge_sorted<T: Ord + Clone>(seq1: &[T], seq2: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(seq1.len() + seq2.len());
    let mut i = 0;
    let mut j = 0;

    while i < seq1.len() && j < seq2.len() {
        if seq1[i] <= seq2[j] {
            result.push(seq1[i].clone());
            i += 1;
        } else {
            result.push(seq2[j].clone());
            j += 1;
        }
    }

    // Add remaining elements
    while i < seq1.len() {
        result.push(seq1[i].clone());
        i += 1;
    }

    while j < seq2.len() {
        result.push(seq2[j].clone());
        j += 1;
    }

    result
}

/// Merges overlapping sorted sequences bidirectionally.
///
/// Similar to `bidirectional_merge_sorted` but handles overlapping ranges
/// by combining elements that compare as equal.
///
/// # Arguments
/// * `seq1` - First sorted sequence
/// * `seq2` - Second sorted sequence
/// * `combine_fn` - Function to combine equal elements
///
/// # Returns
/// A merged sorted sequence with overlaps combined
pub fn bidirectional_merge_overlapping<T: Ord + Clone, F>(
    seq1: &[T],
    seq2: &[T],
    combine_fn: F,
) -> Vec<T>
where
    F: Fn(&T, &T) -> T,
{
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < seq1.len() && j < seq2.len() {
        match seq1[i].cmp(&seq2[j]) {
            Ordering::Less => {
                result.push(seq1[i].clone());
                i += 1;
            }
            Ordering::Greater => {
                result.push(seq2[j].clone());
                j += 1;
            }
            Ordering::Equal => {
                result.push(combine_fn(&seq1[i], &seq2[j]));
                i += 1;
                j += 1;
            }
        }
    }

    // Add remaining elements
    while i < seq1.len() {
        result.push(seq1[i].clone());
        i += 1;
    }

    while j < seq2.len() {
        result.push(seq2[j].clone());
        j += 1;
    }

    result
}

// ======================================================================================
// LOGARITHM AND CATEGORY UTILITIES
// ======================================================================================

/// Converts a logarithm expression to string form.
///
/// # Arguments
/// * `base` - The logarithm base
/// * `arg` - The argument
///
/// # Returns
/// String representation like "log_base(arg)"
pub fn log_string(base: &str, arg: &str) -> String {
    if base == "e" {
        format!("log({})", arg)
    } else {
        format!("log_{}({})", base, arg)
    }
}

/// Strips symbolic wrapper from an expression.
///
/// This removes symbolic type information to get the underlying value.
///
/// # Arguments
/// * `expr` - Expression possibly wrapped in symbolic type
///
/// # Returns
/// The underlying expression
pub fn strip_symbolic(expr: &str) -> String {
    // Remove common symbolic wrappers
    let trimmed = expr.trim();

    if trimmed.starts_with("SR(") && trimmed.ends_with(')') {
        trimmed[3..trimmed.len() - 1].to_string()
    } else if trimmed.starts_with("Symbolic(") && trimmed.ends_with(')') {
        trimmed[9..trimmed.len() - 1].to_string()
    } else {
        expr.to_string()
    }
}

/// Transforms a category by applying a transformation function.
///
/// This is used in category theory operations on asymptotic structures.
///
/// # Arguments
/// * `category` - The category name
/// * `transform` - Transformation to apply
///
/// # Returns
/// The transformed category
pub fn transform_category(category: &str, transform: &str) -> String {
    format!("{}({})", transform, category)
}

// ======================================================================================
// TESTS
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locals_creation() {
        let locals = Locals::new();
        assert!(locals.is_empty());
        assert_eq!(locals.len(), 0);
    }

    #[test]
    fn test_locals_insert_get() {
        let mut locals = Locals::new();
        locals.insert("x".to_string(), "value1".to_string());
        locals.insert("y".to_string(), "value2".to_string());

        assert_eq!(locals.len(), 2);
        assert_eq!(locals.get("x"), Some(&"value1".to_string()));
        assert_eq!(locals.get("y"), Some(&"value2".to_string()));
        assert_eq!(locals.get("z"), None);
    }

    #[test]
    fn test_locals_remove() {
        let mut locals = Locals::new();
        locals.insert("x".to_string(), "value".to_string());

        assert_eq!(locals.remove("x"), Some("value".to_string()));
        assert!(locals.is_empty());
    }

    #[test]
    fn test_locals_contains_key() {
        let mut locals = Locals::new();
        locals.insert("x".to_string(), "value".to_string());

        assert!(locals.contains_key("x"));
        assert!(!locals.contains_key("y"));
    }

    #[test]
    fn test_locals_merge() {
        let mut locals1 = Locals::new();
        locals1.insert("x".to_string(), "value1".to_string());

        let mut locals2 = Locals::new();
        locals2.insert("y".to_string(), "value2".to_string());

        locals1.merge(&locals2);
        assert_eq!(locals1.len(), 2);
        assert!(locals1.contains_key("x"));
        assert!(locals1.contains_key("y"));
    }

    #[test]
    fn test_not_implemented_ozero() {
        let err = NotImplementedOZero::new("test error".to_string());
        assert_eq!(err.message(), "test error");
        assert!(format!("{}", err).contains("NotImplementedOZero"));
    }

    #[test]
    fn test_not_implemented_bzero() {
        let err = NotImplementedBZero::new("test error".to_string());
        assert_eq!(err.message(), "test error");
        assert!(format!("{}", err).contains("NotImplementedBZero"));
    }

    #[test]
    fn test_parent_to_repr_short() {
        let long_repr = "VeryLongParentTypeNameWithManyCharactersThatExceedsFiftyCharactersInTotal";
        let short = parent_to_repr_short(long_repr);
        assert!(short.len() <= 53); // 50 + "..."

        let short_repr = "ShortName";
        let result = parent_to_repr_short(short_repr);
        assert_eq!(result, "ShortName");
    }

    #[test]
    fn test_split_str_by_op() {
        let s = "a+b+c";
        let parts = split_str_by_op(s, '+');
        assert_eq!(parts, vec!["a", "b", "c"]);

        let s2 = "(a+b)+c";
        let parts2 = split_str_by_op(s2, '+');
        assert_eq!(parts2, vec!["(a+b)", "c"]);
    }

    #[test]
    fn test_repr_op() {
        let result = repr_op("x", "+", "y");
        assert_eq!(result, "(x + y)");
    }

    #[test]
    fn test_combine_exceptions() {
        let exceptions = vec!["Error 1".to_string(), "Error 2".to_string()];
        let combined = combine_exceptions(&exceptions);
        assert!(combined.contains("Error 1"));
        assert!(combined.contains("Error 2"));

        let single = vec!["Single error".to_string()];
        assert_eq!(combine_exceptions(&single), "Single error");

        let empty: Vec<String> = vec![];
        assert_eq!(combine_exceptions(&empty), "No exceptions");
    }

    #[test]
    fn test_substitute_raise_exception() {
        let result = substitute_raise_exception("x + 1", "x", "2");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "2 + 1");

        let invalid = substitute_raise_exception("1/x", "x", "0");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_bidirectional_merge_sorted() {
        let seq1 = vec![1, 3, 5, 7];
        let seq2 = vec![2, 4, 6, 8];
        let merged = bidirectional_merge_sorted(&seq1, &seq2);
        assert_eq!(merged, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let seq3 = vec![1, 2, 3];
        let seq4 = vec![4, 5, 6];
        let merged2 = bidirectional_merge_sorted(&seq3, &seq4);
        assert_eq!(merged2, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_bidirectional_merge_overlapping() {
        let seq1 = vec![1, 2, 3];
        let seq2 = vec![2, 3, 4];
        let merged = bidirectional_merge_overlapping(&seq1, &seq2, |a, b| *a.max(b));
        assert_eq!(merged, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_log_string() {
        assert_eq!(log_string("e", "x"), "log(x)");
        assert_eq!(log_string("2", "x"), "log_2(x)");
        assert_eq!(log_string("10", "n"), "log_10(n)");
    }

    #[test]
    fn test_strip_symbolic() {
        assert_eq!(strip_symbolic("SR(x+1)"), "x+1");
        assert_eq!(strip_symbolic("Symbolic(y)"), "y");
        assert_eq!(strip_symbolic("plain"), "plain");
    }

    #[test]
    fn test_transform_category() {
        let result = transform_category("Fields", "Infinite");
        assert_eq!(result, "Infinite(Fields)");
    }

    #[test]
    fn test_locals_display() {
        let mut locals = Locals::new();
        locals.insert("x".to_string(), "1".to_string());
        locals.insert("y".to_string(), "2".to_string());

        let display = format!("{}", locals);
        assert!(display.contains("Locals{"));
        assert!(display.contains("x"));
        assert!(display.contains("y"));
    }
}
