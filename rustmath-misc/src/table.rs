//! Table formatting utilities

/// A table for displaying data
#[derive(Clone, Debug)]
pub struct Table {
    rows: Vec<Vec<String>>,
    #[allow(dead_code)]
    headers: Option<Vec<String>>,
}

impl Table {
    /// Create a new empty table
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            headers: None,
        }
    }

    /// Create a table with headers
    pub fn with_headers(headers: Vec<String>) -> Self {
        Self {
            rows: Vec::new(),
            headers: Some(headers),
        }
    }

    /// Add a row to the table
    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }
}

impl Default for Table {
    fn default() -> Self {
        Self::new()
    }
}
