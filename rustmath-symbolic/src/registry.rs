//! Symbolic Registry System
//!
//! This module provides a registry for tracking the relationship between symbolic
//! variables and their context (e.g., coordinate charts in differential geometry).
//! This is essential for coordinate transformations and chart transitions.

use crate::symbol::Symbol;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};

/// A unique identifier for a chart
pub type ChartId = String;

/// Registry for tracking coordinate symbols and their associated charts
///
/// In differential geometry, coordinate symbols (like x, y, z, r, θ, φ) are
/// associated with specific coordinate charts. This registry maintains these
/// relationships to enable proper coordinate transformations.
///
/// # Thread Safety
///
/// The registry uses `RwLock` for interior mutability and thread-safe access.
/// Multiple readers can access the registry concurrently, but writes are exclusive.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::registry::CoordinateRegistry;
/// use rustmath_symbolic::Symbol;
///
/// let registry = CoordinateRegistry::new();
/// let x = Symbol::new("x");
/// let y = Symbol::new("y");
///
/// // Register symbols for a chart
/// registry.register_symbols("cartesian", vec![x.clone(), y.clone()]);
///
/// // Look up which chart a symbol belongs to
/// let chart_id = registry.get_chart_for_symbol(&x);
/// assert_eq!(chart_id, Some("cartesian".to_string()));
/// ```
pub struct CoordinateRegistry {
    /// Map from symbol to chart ID (using weak references to avoid keeping charts alive)
    symbol_to_chart: RwLock<HashMap<Symbol, ChartId>>,
    /// Map from chart ID to its coordinate symbols
    chart_to_symbols: RwLock<HashMap<ChartId, Vec<Symbol>>>,
}

impl CoordinateRegistry {
    /// Create a new empty coordinate registry
    pub fn new() -> Self {
        Self {
            symbol_to_chart: RwLock::new(HashMap::new()),
            chart_to_symbols: RwLock::new(HashMap::new()),
        }
    }

    /// Register a list of coordinate symbols for a chart
    ///
    /// # Arguments
    ///
    /// * `chart_id` - Unique identifier for the chart
    /// * `symbols` - Vector of coordinate symbols (ordered by coordinate index)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::registry::CoordinateRegistry;
    /// use rustmath_symbolic::Symbol;
    ///
    /// let registry = CoordinateRegistry::new();
    /// let x = Symbol::new("x");
    /// let y = Symbol::new("y");
    /// let z = Symbol::new("z");
    ///
    /// registry.register_symbols("cartesian3d", vec![x, y, z]);
    /// assert_eq!(registry.num_charts(), 1);
    /// ```
    pub fn register_symbols(&self, chart_id: impl Into<String>, symbols: Vec<Symbol>) {
        let chart_id = chart_id.into();

        // Register forward mapping (chart -> symbols)
        {
            let mut chart_map = self.chart_to_symbols.write().unwrap();
            chart_map.insert(chart_id.clone(), symbols.clone());
        }

        // Register reverse mapping (symbol -> chart)
        {
            let mut symbol_map = self.symbol_to_chart.write().unwrap();
            for symbol in symbols {
                symbol_map.insert(symbol, chart_id.clone());
            }
        }
    }

    /// Get the chart ID associated with a symbol
    ///
    /// Returns `None` if the symbol is not registered to any chart.
    pub fn get_chart_for_symbol(&self, symbol: &Symbol) -> Option<ChartId> {
        let symbol_map = self.symbol_to_chart.read().unwrap();
        symbol_map.get(symbol).cloned()
    }

    /// Get the coordinate symbols for a chart
    ///
    /// Returns `None` if the chart is not registered.
    pub fn get_symbols_for_chart(&self, chart_id: &str) -> Option<Vec<Symbol>> {
        let chart_map = self.chart_to_symbols.read().unwrap();
        chart_map.get(chart_id).cloned()
    }

    /// Check if a symbol is registered
    pub fn is_symbol_registered(&self, symbol: &Symbol) -> bool {
        let symbol_map = self.symbol_to_chart.read().unwrap();
        symbol_map.contains_key(symbol)
    }

    /// Check if a chart is registered
    pub fn is_chart_registered(&self, chart_id: &str) -> bool {
        let chart_map = self.chart_to_symbols.read().unwrap();
        chart_map.contains_key(chart_id)
    }

    /// Unregister all symbols for a chart
    ///
    /// Removes the chart and all its coordinate symbols from the registry.
    pub fn unregister_chart(&self, chart_id: &str) {
        // Get the symbols first
        let symbols = {
            let mut chart_map = self.chart_to_symbols.write().unwrap();
            chart_map.remove(chart_id)
        };

        // Remove the reverse mappings
        if let Some(symbols) = symbols {
            let mut symbol_map = self.symbol_to_chart.write().unwrap();
            for symbol in symbols {
                symbol_map.remove(&symbol);
            }
        }
    }

    /// Get the number of registered charts
    pub fn num_charts(&self) -> usize {
        let chart_map = self.chart_to_symbols.read().unwrap();
        chart_map.len()
    }

    /// Get the number of registered symbols
    pub fn num_symbols(&self) -> usize {
        let symbol_map = self.symbol_to_chart.read().unwrap();
        symbol_map.len()
    }

    /// Get all registered chart IDs
    pub fn all_chart_ids(&self) -> Vec<ChartId> {
        let chart_map = self.chart_to_symbols.read().unwrap();
        chart_map.keys().cloned().collect()
    }

    /// Clear all registrations
    pub fn clear(&self) {
        let mut symbol_map = self.symbol_to_chart.write().unwrap();
        let mut chart_map = self.chart_to_symbols.write().unwrap();
        symbol_map.clear();
        chart_map.clear();
    }

    /// Get the dimension (number of coordinates) of a chart
    ///
    /// Returns `None` if the chart is not registered.
    pub fn get_chart_dimension(&self, chart_id: &str) -> Option<usize> {
        let chart_map = self.chart_to_symbols.read().unwrap();
        chart_map.get(chart_id).map(|symbols| symbols.len())
    }

    /// Check if two symbols belong to the same chart
    pub fn same_chart(&self, symbol1: &Symbol, symbol2: &Symbol) -> bool {
        let symbol_map = self.symbol_to_chart.read().unwrap();
        match (symbol_map.get(symbol1), symbol_map.get(symbol2)) {
            (Some(chart1), Some(chart2)) => chart1 == chart2,
            _ => false,
        }
    }

    /// Get the index of a symbol within its chart's coordinate system
    ///
    /// Returns `None` if the symbol is not registered or not found in its chart.
    pub fn get_symbol_index(&self, symbol: &Symbol) -> Option<usize> {
        let symbol_map = self.symbol_to_chart.read().unwrap();
        let chart_id = symbol_map.get(symbol)?;

        let chart_map = self.chart_to_symbols.read().unwrap();
        let symbols = chart_map.get(chart_id)?;

        symbols.iter().position(|s| s == symbol)
    }
}

impl Default for CoordinateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global coordinate registry instance
///
/// This provides a globally accessible registry for coordinate symbols.
/// Use this when you need to share coordinate information across different
/// parts of the codebase.
static GLOBAL_REGISTRY: once_cell::sync::Lazy<Arc<CoordinateRegistry>> =
    once_cell::sync::Lazy::new(|| Arc::new(CoordinateRegistry::new()));

/// Get a reference to the global coordinate registry
pub fn global_registry() -> Arc<CoordinateRegistry> {
    Arc::clone(&GLOBAL_REGISTRY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = CoordinateRegistry::new();
        assert_eq!(registry.num_charts(), 0);
        assert_eq!(registry.num_symbols(), 0);
    }

    #[test]
    fn test_register_symbols() {
        let registry = CoordinateRegistry::new();
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        registry.register_symbols("cartesian", vec![x.clone(), y.clone()]);

        assert_eq!(registry.num_charts(), 1);
        assert_eq!(registry.num_symbols(), 2);
        assert!(registry.is_symbol_registered(&x));
        assert!(registry.is_symbol_registered(&y));
    }

    #[test]
    fn test_get_chart_for_symbol() {
        let registry = CoordinateRegistry::new();
        let r = Symbol::new("r");
        let theta = Symbol::new("theta");

        registry.register_symbols("polar", vec![r.clone(), theta.clone()]);

        assert_eq!(registry.get_chart_for_symbol(&r), Some("polar".to_string()));
        assert_eq!(registry.get_chart_for_symbol(&theta), Some("polar".to_string()));

        let z = Symbol::new("z");
        assert_eq!(registry.get_chart_for_symbol(&z), None);
    }

    #[test]
    fn test_get_symbols_for_chart() {
        let registry = CoordinateRegistry::new();
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let z = Symbol::new("z");

        registry.register_symbols("3d", vec![x.clone(), y.clone(), z.clone()]);

        let symbols = registry.get_symbols_for_chart("3d").unwrap();
        assert_eq!(symbols.len(), 3);

        assert!(registry.get_symbols_for_chart("nonexistent").is_none());
    }

    #[test]
    fn test_unregister_chart() {
        let registry = CoordinateRegistry::new();
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        registry.register_symbols("xy", vec![x.clone(), y.clone()]);
        assert_eq!(registry.num_charts(), 1);
        assert_eq!(registry.num_symbols(), 2);

        registry.unregister_chart("xy");
        assert_eq!(registry.num_charts(), 0);
        assert_eq!(registry.num_symbols(), 0);
        assert!(!registry.is_symbol_registered(&x));
    }

    #[test]
    fn test_multiple_charts() {
        let registry = CoordinateRegistry::new();

        let x = Symbol::new("x");
        let y = Symbol::new("y");
        registry.register_symbols("cartesian", vec![x.clone(), y.clone()]);

        let r = Symbol::new("r");
        let theta = Symbol::new("theta");
        registry.register_symbols("polar", vec![r.clone(), theta.clone()]);

        assert_eq!(registry.num_charts(), 2);
        assert_eq!(registry.num_symbols(), 4);

        assert_eq!(registry.get_chart_for_symbol(&x), Some("cartesian".to_string()));
        assert_eq!(registry.get_chart_for_symbol(&r), Some("polar".to_string()));
    }

    #[test]
    fn test_chart_dimension() {
        let registry = CoordinateRegistry::new();

        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let z = Symbol::new("z");
        registry.register_symbols("3d", vec![x, y, z]);

        assert_eq!(registry.get_chart_dimension("3d"), Some(3));
        assert_eq!(registry.get_chart_dimension("nonexistent"), None);
    }

    #[test]
    fn test_same_chart() {
        let registry = CoordinateRegistry::new();

        let x = Symbol::new("x");
        let y = Symbol::new("y");
        registry.register_symbols("xy", vec![x.clone(), y.clone()]);

        let r = Symbol::new("r");
        registry.register_symbols("r", vec![r.clone()]);

        assert!(registry.same_chart(&x, &y));
        assert!(!registry.same_chart(&x, &r));

        let z = Symbol::new("z");
        assert!(!registry.same_chart(&x, &z));
    }

    #[test]
    fn test_symbol_index() {
        let registry = CoordinateRegistry::new();

        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let z = Symbol::new("z");
        registry.register_symbols("xyz", vec![x.clone(), y.clone(), z.clone()]);

        assert_eq!(registry.get_symbol_index(&x), Some(0));
        assert_eq!(registry.get_symbol_index(&y), Some(1));
        assert_eq!(registry.get_symbol_index(&z), Some(2));

        let w = Symbol::new("w");
        assert_eq!(registry.get_symbol_index(&w), None);
    }

    #[test]
    fn test_all_chart_ids() {
        let registry = CoordinateRegistry::new();

        registry.register_symbols("cart", vec![Symbol::new("x")]);
        registry.register_symbols("polar", vec![Symbol::new("r")]);

        let ids = registry.all_chart_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"cart".to_string()));
        assert!(ids.contains(&"polar".to_string()));
    }

    #[test]
    fn test_clear() {
        let registry = CoordinateRegistry::new();

        registry.register_symbols("chart1", vec![Symbol::new("x")]);
        registry.register_symbols("chart2", vec![Symbol::new("y")]);

        assert_eq!(registry.num_charts(), 2);
        assert_eq!(registry.num_symbols(), 2);

        registry.clear();

        assert_eq!(registry.num_charts(), 0);
        assert_eq!(registry.num_symbols(), 0);
    }

    #[test]
    fn test_global_registry() {
        let registry1 = global_registry();
        let registry2 = global_registry();

        // Should be the same instance
        assert!(Arc::ptr_eq(&registry1, &registry2));
    }
}
