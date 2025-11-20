//! GAP Output Parser - Parse GAP output into Rust structures
//!
//! This module provides parsers for converting GAP's text output into
//! Rust data structures that can be used by RustMath.
//!
//! # Overview
//!
//! GAP outputs results in various formats:
//! - Permutations: `(1,2,3)(4,5)`
//! - Lists: `[ 1, 2, 3, 4 ]`
//! - Records: `rec( order := 120, name := "S5" )`
//! - Groups: `Group( (1,2), (1,2,3,4,5) )`
//!
//! This module parses these formats into appropriate Rust types.
//!
//! # Example
//!
//! ```rust
//! use rustmath_interfaces::gap_parser::*;
//!
//! // Parse a permutation
//! let perm = parse_permutation("(1,2,3)(4,5)").unwrap();
//!
//! // Parse a list
//! let list = parse_list("[ 1, 2, 3, 4 ]").unwrap();
//! ```

use regex::Regex;
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur when parsing GAP output
#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Invalid permutation format: {0}")]
    InvalidPermutation(String),

    #[error("Invalid list format: {0}")]
    InvalidList(String),

    #[error("Invalid record format: {0}")]
    InvalidRecord(String),

    #[error("Invalid integer: {0}")]
    InvalidInteger(String),

    #[error("Invalid boolean: {0}")]
    InvalidBoolean(String),

    #[error("Unexpected format: {0}")]
    UnexpectedFormat(String),
}

pub type Result<T> = std::result::Result<T, ParseError>;

/// A permutation represented as a list of cycles
///
/// For example, (1,2,3)(4,5) is represented as vec![vec![1,2,3], vec![4,5]]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    pub cycles: Vec<Vec<usize>>,
    pub degree: usize,
}

impl Permutation {
    /// Create a new permutation from cycles
    pub fn new(cycles: Vec<Vec<usize>>) -> Self {
        let degree = cycles
            .iter()
            .flat_map(|cycle| cycle.iter())
            .max()
            .copied()
            .unwrap_or(0);

        Self { cycles, degree }
    }

    /// Create the identity permutation
    pub fn identity(degree: usize) -> Self {
        Self {
            cycles: vec![],
            degree,
        }
    }

    /// Convert to image representation
    ///
    /// Returns a vector where the i-th element is the image of i
    pub fn to_images(&self) -> Vec<usize> {
        let mut images: Vec<usize> = (0..=self.degree).collect();

        for cycle in &self.cycles {
            if cycle.len() <= 1 {
                continue;
            }

            for i in 0..cycle.len() {
                let from = cycle[i];
                let to = cycle[(i + 1) % cycle.len()];
                images[from] = to;
            }
        }

        images
    }

    /// Get the order of this permutation
    pub fn order(&self) -> usize {
        if self.cycles.is_empty() {
            return 1;
        }

        // Order is LCM of cycle lengths
        self.cycles
            .iter()
            .map(|cycle| cycle.len())
            .fold(1, |acc, len| lcm(acc, len))
    }
}

/// Parse a GAP permutation string into a Permutation struct
///
/// # Examples
///
/// ```rust
/// use rustmath_interfaces::gap_parser::parse_permutation;
///
/// let perm = parse_permutation("(1,2,3)(4,5)").unwrap();
/// assert_eq!(perm.cycles.len(), 2);
/// assert_eq!(perm.cycles[0], vec![1, 2, 3]);
/// assert_eq!(perm.cycles[1], vec![4, 5]);
/// ```
pub fn parse_permutation(s: &str) -> Result<Permutation> {
    let s = s.trim();

    // Handle identity permutation
    if s == "()" || s.is_empty() {
        return Ok(Permutation::identity(0));
    }

    // Find all cycles using regex: (digits,digits,...)
    let re = Regex::new(r"\(([0-9,\s]+)\)").unwrap();
    let mut cycles = Vec::new();

    for cap in re.captures_iter(s) {
        let cycle_str = &cap[1];
        let cycle: Result<Vec<usize>> = cycle_str
            .split(',')
            .map(|num| {
                num.trim()
                    .parse()
                    .map_err(|_| ParseError::InvalidPermutation(s.to_string()))
            })
            .collect();

        cycles.push(cycle?);
    }

    if cycles.is_empty() {
        return Err(ParseError::InvalidPermutation(s.to_string()));
    }

    Ok(Permutation::new(cycles))
}

/// Parse a GAP list into a vector of strings
///
/// # Examples
///
/// ```rust
/// use rustmath_interfaces::gap_parser::parse_list;
///
/// let list = parse_list("[ 1, 2, 3, 4 ]").unwrap();
/// assert_eq!(list, vec!["1", "2", "3", "4"]);
/// ```
pub fn parse_list(s: &str) -> Result<Vec<String>> {
    let s = s.trim();

    // Check for list brackets
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(ParseError::InvalidList(s.to_string()));
    }

    // Extract content between brackets
    let content = &s[1..s.len() - 1];

    if content.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Split by commas (simple version - doesn't handle nested lists)
    let elements: Vec<String> = content
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    Ok(elements)
}

/// Parse a GAP list of integers
pub fn parse_integer_list(s: &str) -> Result<Vec<i64>> {
    let strings = parse_list(s)?;
    strings
        .iter()
        .map(|s| {
            s.parse()
                .map_err(|_| ParseError::InvalidInteger(s.clone()))
        })
        .collect()
}

/// Parse a GAP record into a HashMap
///
/// # Examples
///
/// ```rust
/// use rustmath_interfaces::gap_parser::parse_record;
///
/// let record = parse_record("rec( order := 120, name := \"S5\" )").unwrap();
/// assert_eq!(record.get("order"), Some(&"120".to_string()));
/// assert_eq!(record.get("name"), Some(&"\"S5\"".to_string()));
/// ```
pub fn parse_record(s: &str) -> Result<HashMap<String, String>> {
    let s = s.trim();

    // Check for record format
    if !s.starts_with("rec(") || !s.ends_with(')') {
        return Err(ParseError::InvalidRecord(s.to_string()));
    }

    let content = &s[4..s.len() - 1].trim();
    let mut record = HashMap::new();

    if content.is_empty() {
        return Ok(record);
    }

    // Split by commas
    for pair in content.split(',') {
        let parts: Vec<&str> = pair.splitn(2, ":=").collect();

        if parts.len() != 2 {
            return Err(ParseError::InvalidRecord(s.to_string()));
        }

        let key = parts[0].trim().to_string();
        let value = parts[1].trim().to_string();

        record.insert(key, value);
    }

    Ok(record)
}

/// Parse a GAP boolean value
pub fn parse_boolean(s: &str) -> Result<bool> {
    match s.trim().to_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ParseError::InvalidBoolean(s.to_string())),
    }
}

/// Parse a GAP integer
pub fn parse_integer(s: &str) -> Result<i64> {
    s.trim()
        .parse()
        .map_err(|_| ParseError::InvalidInteger(s.to_string()))
}

/// Parse a GAP group description
///
/// Extracts information about a group from its GAP representation
#[derive(Debug, Clone)]
pub struct GroupInfo {
    pub type_name: String,
    pub degree: Option<usize>,
    pub generators: Vec<String>,
}

/// Parse a GAP group object description
pub fn parse_group(s: &str) -> Result<GroupInfo> {
    let s = s.trim();

    // Try to extract group type
    let type_name = if s.contains("SymmetricGroup") {
        "SymmetricGroup"
    } else if s.contains("AlternatingGroup") {
        "AlternatingGroup"
    } else if s.contains("CyclicGroup") {
        "CyclicGroup"
    } else if s.contains("DihedralGroup") {
        "DihedralGroup"
    } else if s.contains("Group") {
        "Group"
    } else {
        return Err(ParseError::UnexpectedFormat(s.to_string()));
    };

    // Try to extract degree (for Sn, An, etc.)
    let degree_re = Regex::new(r"Group\((\d+)\)").unwrap();
    let degree = degree_re
        .captures(s)
        .and_then(|cap| cap[1].parse().ok());

    // Try to extract generators if in Group(...) format
    let mut generators = Vec::new();
    if s.starts_with("Group(") && s.ends_with(')') {
        let content = &s[6..s.len() - 1];
        generators = content.split(',').map(|s| s.trim().to_string()).collect();
    }

    Ok(GroupInfo {
        type_name: type_name.to_string(),
        degree,
        generators,
    })
}

/// Compute LCM of two numbers
fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    a * b / gcd(a, b)
}

/// Compute GCD of two numbers
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_permutation() {
        let perm = parse_permutation("(1,2,3)(4,5)").unwrap();
        assert_eq!(perm.cycles.len(), 2);
        assert_eq!(perm.cycles[0], vec![1, 2, 3]);
        assert_eq!(perm.cycles[1], vec![4, 5]);
        assert_eq!(perm.degree, 5);
    }

    #[test]
    fn test_parse_identity_permutation() {
        let perm = parse_permutation("()").unwrap();
        assert_eq!(perm.cycles.len(), 0);
    }

    #[test]
    fn test_permutation_order() {
        let perm = parse_permutation("(1,2,3)(4,5)").unwrap();
        assert_eq!(perm.order(), 6); // LCM(3, 2) = 6
    }

    #[test]
    fn test_permutation_to_images() {
        let perm = parse_permutation("(1,2,3)").unwrap();
        let images = perm.to_images();
        assert_eq!(images[1], 2);
        assert_eq!(images[2], 3);
        assert_eq!(images[3], 1);
    }

    #[test]
    fn test_parse_list() {
        let list = parse_list("[ 1, 2, 3, 4 ]").unwrap();
        assert_eq!(list, vec!["1", "2", "3", "4"]);
    }

    #[test]
    fn test_parse_empty_list() {
        let list = parse_list("[ ]").unwrap();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_parse_integer_list() {
        let list = parse_integer_list("[ 1, 2, 3, 4 ]").unwrap();
        assert_eq!(list, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_parse_record() {
        let record = parse_record("rec( order := 120, name := \"S5\" )").unwrap();
        assert_eq!(record.get("order"), Some(&"120".to_string()));
        assert_eq!(record.get("name"), Some(&"\"S5\"".to_string()));
    }

    #[test]
    fn test_parse_boolean() {
        assert_eq!(parse_boolean("true").unwrap(), true);
        assert_eq!(parse_boolean("false").unwrap(), false);
        assert_eq!(parse_boolean("True").unwrap(), true);
        assert_eq!(parse_boolean("FALSE").unwrap(), false);
    }

    #[test]
    fn test_parse_integer() {
        assert_eq!(parse_integer("42").unwrap(), 42);
        assert_eq!(parse_integer(" 123 ").unwrap(), 123);
        assert_eq!(parse_integer("-5").unwrap(), -5);
    }

    #[test]
    fn test_parse_group() {
        let info = parse_group("SymmetricGroup(5)").unwrap();
        assert_eq!(info.type_name, "SymmetricGroup");
        assert_eq!(info.degree, Some(5));
    }
}
