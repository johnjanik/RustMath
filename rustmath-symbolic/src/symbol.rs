//! Symbolic variables

use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, OnceLock};

/// Global interning registry mapping symbol names to stable ids.
///
/// Standard CAS semantics: two symbols with the same name are THE SAME
/// symbol. `Symbol::new("x")` therefore returns a symbol whose id is
/// looked up (or created) by name, so every `Symbol::new("x")` call yields
/// an equal symbol with the same id. This keeps equality, hashing,
/// ordering, and the id-keyed assumptions database consistent across
/// independently created instances.
static SYMBOL_REGISTRY: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();

/// A symbolic variable
///
/// Symbols are identified by name: two `Symbol`s with the same name are
/// equal (and share the same id), no matter which `Symbol::new` call
/// created them.
///
/// ```
/// use rustmath_symbolic::Symbol;
///
/// assert_eq!(Symbol::new("x"), Symbol::new("x"));
/// assert_ne!(Symbol::new("x"), Symbol::new("y"));
/// ```
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol {
    name: String,
    id: usize,
}

impl Symbol {
    /// Create a symbol with a given name
    ///
    /// The id is interned by name: the first call with a given name mints
    /// a fresh id, and every later call with the same name reuses it.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let registry = SYMBOL_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = registry.lock().unwrap();
        let next_id = map.len();
        let id = *map.entry(name.clone()).or_insert(next_id);
        Symbol { name, id }
    }

    /// Get the symbol name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the symbol id
    ///
    /// The id is determined by the name: equal names always yield equal ids.
    pub fn id(&self) -> usize {
        self.id
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol({}:{})", self.name, self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_name_symbols_are_equal() {
        // Regression test: Symbol::new used to mint a fresh id per call,
        // making two Symbol::new("x") unequal and breaking differentiation.
        let a = Symbol::new("x");
        let b = Symbol::new("x");
        assert_eq!(a, b);
        assert_eq!(a.id(), b.id());

        let c = Symbol::new("y");
        assert_ne!(a, c);
        assert_ne!(a.id(), c.id());
    }

    #[test]
    fn test_id_stable_across_calls() {
        let first = Symbol::new("stable_id_test_sym").id();
        let second = Symbol::new("stable_id_test_sym").id();
        let third = Symbol::new("stable_id_test_sym").id();
        assert_eq!(first, second);
        assert_eq!(second, third);
    }

    #[test]
    fn test_hashmap_keyed_by_symbol() {
        use std::collections::HashMap;

        let mut map: HashMap<Symbol, i32> = HashMap::new();
        map.insert(Symbol::new("hm_x"), 1);
        // Same name from a separate call is the same key: overwrite, not insert.
        map.insert(Symbol::new("hm_x"), 2);
        assert_eq!(map.len(), 1);
        assert_eq!(map[&Symbol::new("hm_x")], 2);

        map.insert(Symbol::new("hm_y"), 3);
        assert_eq!(map.len(), 2);
        assert_eq!(map[&Symbol::new("hm_y")], 3);
    }

    #[test]
    fn test_ordering_consistent_with_equality() {
        use std::cmp::Ordering;
        let a = Symbol::new("ord_a");
        let a2 = Symbol::new("ord_a");
        assert_eq!(a.cmp(&a2), Ordering::Equal);

        let b = Symbol::new("ord_b");
        assert_ne!(a.cmp(&b), Ordering::Equal);
    }
}
