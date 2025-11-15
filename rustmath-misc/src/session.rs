//! Session management for saving/loading computation state

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A computation session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    variables: HashMap<String, String>,
    history: Vec<String>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            history: Vec::new(),
        }
    }

    pub fn set_variable(&mut self, name: String, value: String) {
        self.variables.insert(name, value);
    }

    pub fn get_variable(&self, name: &str) -> Option<&String> {
        self.variables.get(name)
    }

    pub fn add_history(&mut self, command: String) {
        self.history.push(command);
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    pub fn clear(&mut self) {
        self.variables.clear();
        self.history.clear();
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
