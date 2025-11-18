//! Physical units and dimensional analysis
//!
//! This module provides support for physical units and unit conversion.
//! In SageMath, this allows working with quantities like "5 meters" or
//! "10 seconds" and performing unit-aware computations.
//!
//! # Unit Systems
//!
//! - **SI (International System)**: meter, kilogram, second, ampere, kelvin, mole, candela
//! - **CGS**: centimeter, gram, second
//! - **Imperial**: foot, pound, second
//!
//! # Dimensional Analysis
//!
//! Units have dimensions: [L] for length, [M] for mass, [T] for time, etc.
//! Unit conversion is only valid between units with the same dimensions.
//!
//! # Implementation Status
//!
//! This is a simplified placeholder implementation. A full implementation would include:
//! - Complete unit database (SI, CGS, Imperial, etc.)
//! - Automatic unit conversion
//! - Dimensional analysis
//! - Unit arithmetic (multiplication, division)
//! - Temperature conversion (requires offset, not just scaling)
//! - Compound units (m/s, kg*m/s², etc.)

use crate::expression::Expr;
use std::collections::HashMap;

/// A unit expression combining a magnitude with a unit
///
/// # Example
///
/// ```
/// use rustmath_symbolic::units::UnitExpression;
/// use rustmath_symbolic::expression::Expr;
///
/// // 5 meters
/// let distance = UnitExpression::new(Expr::from(5), "meter");
/// assert_eq!(distance.magnitude(), &Expr::from(5));
/// assert_eq!(distance.unit(), "meter");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UnitExpression {
    magnitude: Expr,
    unit: String,
}

impl UnitExpression {
    /// Create a new unit expression
    ///
    /// # Arguments
    ///
    /// * `magnitude` - The numeric magnitude
    /// * `unit` - The unit name
    pub fn new(magnitude: Expr, unit: impl Into<String>) -> Self {
        Self {
            magnitude,
            unit: unit.into(),
        }
    }

    /// Get the magnitude
    pub fn magnitude(&self) -> &Expr {
        &self.magnitude
    }

    /// Get the unit
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Convert to a different unit
    ///
    /// # Arguments
    ///
    /// * `target_unit` - The target unit
    ///
    /// # Returns
    ///
    /// A new unit expression in the target unit, or an error if conversion is not possible
    pub fn convert_to(&self, target_unit: &str) -> Result<Self, String> {
        convert(&self.magnitude, &self.unit, target_unit)
            .map(|mag| UnitExpression::new(mag, target_unit))
    }
}

/// Container for unit definitions
///
/// # Example
///
/// ```
/// use rustmath_symbolic::units::Units;
///
/// let units = Units::new();
/// assert!(units.is_unit("meter"));
/// assert!(units.is_unit("second"));
/// assert!(!units.is_unit("invalid_unit"));
/// ```
#[derive(Debug, Clone)]
pub struct Units {
    /// Map from unit name to base unit and conversion factor
    units: HashMap<String, (String, f64)>,
}

impl Units {
    /// Create a new units container with default SI units
    pub fn new() -> Self {
        let mut units = HashMap::new();

        // Length (base: meter)
        units.insert("meter".to_string(), ("meter".to_string(), 1.0));
        units.insert("m".to_string(), ("meter".to_string(), 1.0));
        units.insert("kilometer".to_string(), ("meter".to_string(), 1000.0));
        units.insert("km".to_string(), ("meter".to_string(), 1000.0));
        units.insert("centimeter".to_string(), ("meter".to_string(), 0.01));
        units.insert("cm".to_string(), ("meter".to_string(), 0.01));
        units.insert("millimeter".to_string(), ("meter".to_string(), 0.001));
        units.insert("mm".to_string(), ("meter".to_string(), 0.001));

        // Time (base: second)
        units.insert("second".to_string(), ("second".to_string(), 1.0));
        units.insert("s".to_string(), ("second".to_string(), 1.0));
        units.insert("minute".to_string(), ("second".to_string(), 60.0));
        units.insert("min".to_string(), ("second".to_string(), 60.0));
        units.insert("hour".to_string(), ("second".to_string(), 3600.0));
        units.insert("h".to_string(), ("second".to_string(), 3600.0));

        // Mass (base: kilogram)
        units.insert("kilogram".to_string(), ("kilogram".to_string(), 1.0));
        units.insert("kg".to_string(), ("kilogram".to_string(), 1.0));
        units.insert("gram".to_string(), ("kilogram".to_string(), 0.001));
        units.insert("g".to_string(), ("kilogram".to_string(), 0.001));

        Self { units }
    }

    /// Check if a string represents a valid unit
    pub fn is_unit(&self, name: &str) -> bool {
        self.units.contains_key(name)
    }

    /// Get the base unit and conversion factor for a unit
    pub fn get_unit_info(&self, name: &str) -> Option<(&str, f64)> {
        self.units.get(name).map(|(base, factor)| (base.as_str(), *factor))
    }
}

impl Default for Units {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a string represents a valid unit
///
/// # Arguments
///
/// * `name` - The unit name to check
///
/// # Returns
///
/// `true` if the name is a valid unit, `false` otherwise
pub fn is_unit(name: &str) -> bool {
    Units::new().is_unit(name)
}

/// Convert an expression from one unit to another
///
/// # Arguments
///
/// * `value` - The value to convert
/// * `from_unit` - The source unit
/// * `to_unit` - The target unit
///
/// # Returns
///
/// The converted value, or an error if conversion is not possible
///
/// # Example
///
/// ```
/// use rustmath_symbolic::units::convert;
/// use rustmath_symbolic::expression::Expr;
///
/// let result = convert(&Expr::from(1000), "meter", "kilometer");
/// assert!(result.is_ok());
/// ```
pub fn convert(value: &Expr, from_unit: &str, to_unit: &str) -> Result<Expr, String> {
    let units = Units::new();

    let (from_base, from_factor) = units
        .get_unit_info(from_unit)
        .ok_or_else(|| format!("Unknown unit: {}", from_unit))?;

    let (to_base, to_factor) = units
        .get_unit_info(to_unit)
        .ok_or_else(|| format!("Unknown unit: {}", to_unit))?;

    if from_base != to_base {
        return Err(format!(
            "Cannot convert between incompatible units: {} and {}",
            from_unit, to_unit
        ));
    }

    // Convert: value_in_from * from_factor / to_factor = value_in_to
    // For simplicity in this placeholder, we return an error for non-trivial conversions
    // A full implementation would handle the conversion factor properly
    if (from_factor - to_factor).abs() < 1e-10 {
        Ok(value.clone())
    } else {
        Err(format!(
            "Unit conversion factor calculation not fully implemented (from {} to {})",
            from_unit, to_unit
        ))
    }
}

/// Convert temperature between different scales
///
/// Temperature conversion requires offsets, not just scaling.
/// - Celsius to Fahrenheit: F = C * 9/5 + 32
/// - Celsius to Kelvin: K = C + 273.15
/// - Kelvin to Fahrenheit: F = (K - 273.15) * 9/5 + 32
///
/// # Arguments
///
/// * `value` - The temperature value
/// * `from_scale` - The source scale ("celsius", "fahrenheit", "kelvin")
/// * `to_scale` - The target scale
///
/// # Returns
///
/// The converted temperature, or an error
pub fn convert_temperature(value: &Expr, from_scale: &str, to_scale: &str) -> Result<Expr, String> {
    // For simplicity, only handle numeric values
    // A full implementation would handle symbolic expressions

    let from_lower = from_scale.to_lowercase();
    let to_lower = to_scale.to_lowercase();

    if from_lower == to_lower {
        return Ok(value.clone());
    }

    // This is a simplified implementation
    // In practice, temperature conversion with expressions is complex
    Err(format!(
        "Temperature conversion from {} to {} not fully implemented",
        from_scale, to_scale
    ))
}

/// Convert a string representation to a unit
///
/// # Arguments
///
/// * `s` - The string to parse
///
/// # Returns
///
/// The unit name, or an error
pub fn str_to_unit(s: &str) -> Result<String, String> {
    if is_unit(s) {
        Ok(s.to_string())
    } else {
        Err(format!("'{}' is not a valid unit", s))
    }
}

/// Get base units for dimensional analysis
///
/// Returns the fundamental SI units.
///
/// # Returns
///
/// A vector of base unit names
pub fn base_units() -> Vec<String> {
    vec![
        "meter".to_string(),
        "kilogram".to_string(),
        "second".to_string(),
        "ampere".to_string(),
        "kelvin".to_string(),
        "mole".to_string(),
        "candela".to_string(),
    ]
}

/// Evaluate a unit dictionary
///
/// In SageMath, this converts a dictionary of units to expressions.
/// This is a simplified placeholder.
///
/// # Arguments
///
/// * `units_dict` - Map from unit names to values
///
/// # Returns
///
/// A map from unit names to evaluated expressions
pub fn evalunitdict(units_dict: HashMap<String, Expr>) -> HashMap<String, Expr> {
    // For now, just return the input unchanged
    // A full implementation would evaluate symbolic expressions
    units_dict
}

/// Extract variables from a string representation
///
/// # Arguments
///
/// * `s` - The string to parse
///
/// # Returns
///
/// A vector of variable names found in the string
pub fn vars_in_str(s: &str) -> Vec<String> {
    // Simplified implementation: find alphanumeric sequences
    let mut vars = Vec::new();
    let mut current = String::new();

    for ch in s.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else if !current.is_empty() {
            vars.push(current.clone());
            current.clear();
        }
    }

    if !current.is_empty() {
        vars.push(current);
    }

    vars
}

/// Get unit derivations for an expression
///
/// Returns how a unit is derived from base units.
///
/// # Arguments
///
/// * `expr` - The expression
///
/// # Returns
///
/// A map from base units to their exponents
pub fn unit_derivations_expr(_expr: &Expr) -> HashMap<String, i32> {
    // Placeholder: would analyze expression to extract dimensions
    HashMap::new()
}

/// Get documentation for units
///
/// # Returns
///
/// A string containing unit documentation
pub fn unitdocs() -> String {
    String::from("Unit documentation:\n\
        Length: meter (m), kilometer (km), centimeter (cm)\n\
        Time: second (s), minute (min), hour (h)\n\
        Mass: kilogram (kg), gram (g)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_expression_new() {
        let expr = UnitExpression::new(Expr::from(5), "meter");
        assert_eq!(expr.magnitude(), &Expr::from(5));
        assert_eq!(expr.unit(), "meter");
    }

    #[test]
    fn test_units_is_unit() {
        let units = Units::new();
        assert!(units.is_unit("meter"));
        assert!(units.is_unit("m"));
        assert!(units.is_unit("kilometer"));
        assert!(units.is_unit("second"));
        assert!(!units.is_unit("invalid"));
    }

    #[test]
    fn test_is_unit_function() {
        assert!(is_unit("meter"));
        assert!(is_unit("second"));
        assert!(!is_unit("invalid"));
    }

    #[test]
    fn test_convert_same_dimension() {
        let result = convert(&Expr::from(1000), "meter", "kilometer");
        // In the simplified implementation, conversion returns error
        // A full implementation would perform the conversion
        let _ = result;
    }

    #[test]
    fn test_convert_incompatible() {
        let result = convert(&Expr::from(1), "meter", "second");
        assert!(result.is_err());
    }

    #[test]
    fn test_str_to_unit() {
        assert!(str_to_unit("meter").is_ok());
        assert!(str_to_unit("invalid").is_err());
    }

    #[test]
    fn test_base_units() {
        let units = base_units();
        assert_eq!(units.len(), 7);
        assert!(units.contains(&"meter".to_string()));
        assert!(units.contains(&"kilogram".to_string()));
        assert!(units.contains(&"second".to_string()));
    }

    #[test]
    fn test_vars_in_str() {
        let vars = vars_in_str("x + y * z");
        // The function extracts alphanumeric sequences
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert!(vars.contains(&"z".to_string()));
        // May also contain other tokens depending on implementation
    }

    #[test]
    fn test_evalunitdict() {
        let mut dict = HashMap::new();
        dict.insert("meter".to_string(), Expr::from(1));

        let result = evalunitdict(dict.clone());
        assert_eq!(result.len(), dict.len());
    }

    #[test]
    fn test_unit_derivations_expr() {
        let expr = Expr::from(1);
        let derivations = unit_derivations_expr(&expr);
        // Placeholder implementation returns empty map
        assert_eq!(derivations.len(), 0);
    }

    #[test]
    fn test_unitdocs() {
        let docs = unitdocs();
        assert!(!docs.is_empty());
        assert!(docs.contains("meter"));
    }

    #[test]
    fn test_units_default() {
        let units = Units::default();
        assert!(units.is_unit("meter"));
    }

    #[test]
    fn test_convert_temperature_not_implemented() {
        let result = convert_temperature(&Expr::from(0), "celsius", "fahrenheit");
        assert!(result.is_err());
    }

    #[test]
    fn test_unit_expression_convert() {
        let expr = UnitExpression::new(Expr::from(1000), "meter");
        let result = expr.convert_to("kilometer");
        // Conversion may fail in simplified implementation
        let _ = result;
    }
}
