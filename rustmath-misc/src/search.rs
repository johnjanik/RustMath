//! Search functionality for mathematical objects

use std::collections::HashMap;

/// Search index for mathematical objects
pub struct SearchIndex {
    objects: HashMap<String, Vec<String>>,
}

impl SearchIndex {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    pub fn index(&mut self, name: String, keywords: Vec<String>) {
        self.objects.insert(name, keywords);
    }

    pub fn search(&self, query: &str) -> Vec<&String> {
        let query_lower = query.to_lowercase();
        self.objects
            .iter()
            .filter(|(name, keywords)| {
                name.to_lowercase().contains(&query_lower) ||
                keywords.iter().any(|k| k.to_lowercase().contains(&query_lower))
            })
            .map(|(name, _)| name)
            .collect()
    }

    pub fn clear(&mut self) {
        self.objects.clear();
    }
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}
