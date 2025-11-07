//! Symbolic variables

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

static SYMBOL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A symbolic variable
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol {
    name: String,
    id: usize,
}

impl Symbol {
    /// Create a new symbol with a given name
    pub fn new(name: impl Into<String>) -> Self {
        let id = SYMBOL_COUNTER.fetch_add(1, Ordering::Relaxed);
        Symbol {
            name: name.into(),
            id,
        }
    }

    /// Get the symbol name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the symbol id
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
