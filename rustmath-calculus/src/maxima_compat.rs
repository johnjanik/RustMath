//! Maxima compatibility layer
//!
//! This module provides compatibility functions for interfacing with Maxima-style
//! options and configurations. In SageMath, these functions help manage options
//! passed to Maxima for symbolic computation.

use std::collections::HashMap;

/// Represents Maxima options as key-value pairs.
pub type MaximaOptions = HashMap<String, MaximaValue>;

/// Represents possible values for Maxima options.
#[derive(Debug, Clone, PartialEq)]
pub enum MaximaValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Symbol value
    Symbol(String),
    /// List of values
    List(Vec<MaximaValue>),
}

impl From<bool> for MaximaValue {
    fn from(b: bool) -> Self {
        MaximaValue::Bool(b)
    }
}

impl From<i64> for MaximaValue {
    fn from(i: i64) -> Self {
        MaximaValue::Int(i)
    }
}

impl From<f64> for MaximaValue {
    fn from(f: f64) -> Self {
        MaximaValue::Float(f)
    }
}

impl From<String> for MaximaValue {
    fn from(s: String) -> Self {
        MaximaValue::String(s)
    }
}

impl From<&str> for MaximaValue {
    fn from(s: &str) -> Self {
        MaximaValue::String(s.to_string())
    }
}

/// Creates a mapping of option names to their Maxima equivalents.
///
/// This function maps user-friendly option names to the actual option names
/// used by Maxima. This provides a compatibility layer for SageMath-style
/// option handling.
///
/// # Arguments
///
/// * `options` - User-provided options
///
/// # Returns
///
/// A mapping from user option names to Maxima option names.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::maxima_compat::mapped_opts;
/// use std::collections::HashMap;
///
/// let mut options = HashMap::new();
/// options.insert("simplify".to_string(), "true".to_string());
/// let mapped = mapped_opts(&options);
/// // Returns mapping of option names
/// ```
pub fn mapped_opts(options: &HashMap<String, String>) -> HashMap<String, String> {
    let mut result = HashMap::new();

    // Define mappings from user-friendly names to Maxima option names
    let mappings = get_option_mappings();

    for (key, value) in options {
        if let Some(maxima_key) = mappings.get(key) {
            result.insert(maxima_key.clone(), value.clone());
        } else {
            // If no mapping exists, use the key as-is
            result.insert(key.clone(), value.clone());
        }
    }

    result
}

/// Returns the standard option name mappings.
fn get_option_mappings() -> HashMap<String, String> {
    let mut mappings = HashMap::new();

    // Common option mappings
    mappings.insert("simplify".to_string(), "simpsum".to_string());
    mappings.insert("expand".to_string(), "expop".to_string());
    mappings.insert("factor".to_string(), "factorflag".to_string());
    mappings.insert("ratsimp".to_string(), "ratsimp".to_string());
    mappings.insert("trigsimp".to_string(), "trigsign".to_string());
    mappings.insert("radexpand".to_string(), "radexpand".to_string());
    mappings.insert("logexpand".to_string(), "logexpand".to_string());
    mappings.insert("domain".to_string(), "domain".to_string());

    mappings
}

/// Configures Maxima options for symbolic computation.
///
/// This function sets various Maxima options that control symbolic computation
/// behavior such as simplification, expansion, and domain assumptions.
///
/// # Arguments
///
/// * `options` - Options to set, as key-value pairs
///
/// # Returns
///
/// The configured Maxima options.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::maxima_compat::{maxima_options, MaximaOptions, MaximaValue};
///
/// let mut opts = MaximaOptions::new();
/// opts.insert("domain".to_string(), MaximaValue::Symbol("complex".to_string()));
/// opts.insert("simplify".to_string(), MaximaValue::Bool(true));
///
/// let configured = maxima_options(opts);
/// // Returns validated and configured options
/// ```
pub fn maxima_options(options: MaximaOptions) -> MaximaOptions {
    let mut result = options.clone();

    // Apply default options if not specified
    apply_defaults(&mut result);

    // Validate options
    validate_options(&mut result);

    result
}

/// Applies default values for common Maxima options.
fn apply_defaults(options: &mut MaximaOptions) {
    // Set defaults if not already present
    options
        .entry("domain".to_string())
        .or_insert(MaximaValue::Symbol("complex".to_string()));

    options
        .entry("simpsum".to_string())
        .or_insert(MaximaValue::Bool(false));

    options
        .entry("ratsimp".to_string())
        .or_insert(MaximaValue::Bool(false));

    options
        .entry("logexpand".to_string())
        .or_insert(MaximaValue::Bool(true));
}

/// Validates Maxima options and corrects invalid values.
fn validate_options(options: &mut MaximaOptions) {
    // Validate domain option
    if let Some(MaximaValue::Symbol(domain)) = options.get("domain") {
        match domain.as_str() {
            "real" | "complex" | "integer" | "rational" => {
                // Valid domain
            }
            _ => {
                // Invalid domain - reset to default
                options.insert(
                    "domain".to_string(),
                    MaximaValue::Symbol("complex".to_string()),
                );
            }
        }
    }

    // Validate boolean options
    for key in &["simpsum", "ratsimp", "logexpand", "radexpand"] {
        if let Some(value) = options.get(*key) {
            if !matches!(value, MaximaValue::Bool(_)) {
                // Convert to boolean if possible
                let bool_val = match value {
                    MaximaValue::String(s) => s == "true" || s == "t",
                    MaximaValue::Symbol(s) => s == "true" || s == "t",
                    MaximaValue::Int(i) => *i != 0,
                    _ => false,
                };
                options.insert(key.to_string(), MaximaValue::Bool(bool_val));
            }
        }
    }
}

/// Converts RustMath options to Maxima format.
///
/// This function translates options from RustMath's internal format to
/// the format expected by Maxima, handling type conversions and formatting.
///
/// # Arguments
///
/// * `options` - RustMath options
///
/// # Returns
///
/// Maxima-formatted option string.
pub fn format_maxima_options(options: &MaximaOptions) -> String {
    let mut result = Vec::new();

    for (key, value) in options {
        let formatted = match value {
            MaximaValue::Bool(true) => format!("{}:true", key),
            MaximaValue::Bool(false) => format!("{}:false", key),
            MaximaValue::Int(i) => format!("{}:{}", key, i),
            MaximaValue::Float(f) => format!("{}:{}", key, f),
            MaximaValue::String(s) => format!("{}:\"{}\"", key, s),
            MaximaValue::Symbol(s) => format!("{}:{}", key, s),
            MaximaValue::List(items) => {
                let items_str: Vec<String> = items
                    .iter()
                    .map(|item| format_maxima_value(item))
                    .collect();
                format!("{}:[{}]", key, items_str.join(","))
            }
        };
        result.push(formatted);
    }

    result.join(",")
}

/// Formats a single Maxima value.
fn format_maxima_value(value: &MaximaValue) -> String {
    match value {
        MaximaValue::Bool(true) => "true".to_string(),
        MaximaValue::Bool(false) => "false".to_string(),
        MaximaValue::Int(i) => i.to_string(),
        MaximaValue::Float(f) => f.to_string(),
        MaximaValue::String(s) => format!("\"{}\"", s),
        MaximaValue::Symbol(s) => s.clone(),
        MaximaValue::List(items) => {
            let items_str: Vec<String> = items.iter().map(|i| format_maxima_value(i)).collect();
            format!("[{}]", items_str.join(","))
        }
    }
}

/// Parses Maxima options from a string.
///
/// # Arguments
///
/// * `options_str` - String containing Maxima options
///
/// # Returns
///
/// Parsed options as a HashMap.
pub fn parse_maxima_options(options_str: &str) -> MaximaOptions {
    let mut options = MaximaOptions::new();

    // Simple parsing - split by comma and parse key:value pairs
    for pair in options_str.split(',') {
        let parts: Vec<&str> = pair.split(':').collect();
        if parts.len() == 2 {
            let key = parts[0].trim().to_string();
            let value_str = parts[1].trim();

            // Parse the value
            let value = if value_str == "true" || value_str == "t" {
                MaximaValue::Bool(true)
            } else if value_str == "false" || value_str == "nil" {
                MaximaValue::Bool(false)
            } else if let Ok(i) = value_str.parse::<i64>() {
                MaximaValue::Int(i)
            } else if let Ok(f) = value_str.parse::<f64>() {
                MaximaValue::Float(f)
            } else if value_str.starts_with('"') && value_str.ends_with('"') {
                MaximaValue::String(value_str[1..value_str.len() - 1].to_string())
            } else {
                MaximaValue::Symbol(value_str.to_string())
            };

            options.insert(key, value);
        }
    }

    options
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapped_opts() {
        let mut options = HashMap::new();
        options.insert("simplify".to_string(), "true".to_string());
        options.insert("expand".to_string(), "false".to_string());

        let mapped = mapped_opts(&options);

        assert_eq!(mapped.get("simpsum"), Some(&"true".to_string()));
        assert_eq!(mapped.get("expop"), Some(&"false".to_string()));
    }

    #[test]
    fn test_maxima_options_defaults() {
        let options = MaximaOptions::new();
        let configured = maxima_options(options);

        assert!(configured.contains_key("domain"));
        assert!(configured.contains_key("simpsum"));
    }

    #[test]
    fn test_maxima_options_validation() {
        let mut options = MaximaOptions::new();
        options.insert(
            "domain".to_string(),
            MaximaValue::Symbol("invalid".to_string()),
        );

        let configured = maxima_options(options);

        // Should be reset to default "complex"
        assert_eq!(
            configured.get("domain"),
            Some(&MaximaValue::Symbol("complex".to_string()))
        );
    }

    #[test]
    fn test_format_maxima_options() {
        let mut options = MaximaOptions::new();
        options.insert("domain".to_string(), MaximaValue::Symbol("real".to_string()));
        options.insert("simpsum".to_string(), MaximaValue::Bool(true));
        options.insert("depth".to_string(), MaximaValue::Int(10));

        let formatted = format_maxima_options(&options);

        // Order may vary, so check that each option is present
        assert!(formatted.contains("domain:real"));
        assert!(formatted.contains("simpsum:true"));
        assert!(formatted.contains("depth:10"));
    }

    #[test]
    fn test_parse_maxima_options() {
        let options_str = "domain:real,simpsum:true,depth:10";
        let parsed = parse_maxima_options(options_str);

        assert_eq!(
            parsed.get("domain"),
            Some(&MaximaValue::Symbol("real".to_string()))
        );
        assert_eq!(parsed.get("simpsum"), Some(&MaximaValue::Bool(true)));
        assert_eq!(parsed.get("depth"), Some(&MaximaValue::Int(10)));
    }

    #[test]
    fn test_maxima_value_conversions() {
        let bool_val: MaximaValue = true.into();
        assert_eq!(bool_val, MaximaValue::Bool(true));

        let int_val: MaximaValue = 42i64.into();
        assert_eq!(int_val, MaximaValue::Int(42));

        let float_val: MaximaValue = 3.14.into();
        assert_eq!(float_val, MaximaValue::Float(3.14));

        let str_val: MaximaValue = "test".into();
        assert_eq!(str_val, MaximaValue::String("test".to_string()));
    }

    #[test]
    fn test_format_maxima_value_list() {
        let list = MaximaValue::List(vec![
            MaximaValue::Int(1),
            MaximaValue::Int(2),
            MaximaValue::Int(3),
        ]);

        let formatted = format_maxima_value(&list);
        assert_eq!(formatted, "[1,2,3]");
    }
}
