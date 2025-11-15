//! Documentation utilities

/// Documentation formatter
pub struct DocFormatter {
    show_source: bool,
}

impl DocFormatter {
    pub fn new() -> Self {
        Self { show_source: false }
    }

    pub fn with_source(mut self, show_source: bool) -> Self {
        self.show_source = show_source;
        self
    }

    pub fn format(&self, docstring: &str) -> String {
        // Basic formatting: strip leading/trailing whitespace
        docstring.trim().to_string()
    }

    pub fn format_latex(&self, text: &str) -> String {
        // Convert inline math: $...$ to formatted output
        text.replace("$", "")
    }
}

impl Default for DocFormatter {
    fn default() -> Self {
        Self::new()
    }
}
