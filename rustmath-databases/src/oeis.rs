//! OEIS (Online Encyclopedia of Integer Sequences) interface
//!
//! Provides access to the OEIS database through its JSON API.
//!
//! # Example
//!
//! ```no_run
//! use rustmath_databases::oeis::OEISClient;
//!
//! let client = OEISClient::new();
//!
//! // Look up a sequence by its A-number
//! if let Ok(Some(seq)) = client.lookup("A000045") {
//!     println!("Sequence: {}", seq.name);
//!     println!("Data: {:?}", seq.data);
//! }
//!
//! // Search for sequences matching a pattern
//! if let Ok(results) = client.search_by_terms(&[1, 1, 2, 3, 5, 8]) {
//!     for seq in results.iter().take(5) {
//!         println!("{}: {}", seq.number, seq.name);
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Error type for OEIS operations
#[derive(Debug)]
pub enum OEISError {
    /// Network error during API request
    NetworkError(String),
    /// JSON parsing error
    ParseError(String),
    /// Sequence not found
    NotFound(String),
    /// Invalid sequence number format
    InvalidNumber(String),
}

impl fmt::Display for OEISError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OEISError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            OEISError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            OEISError::NotFound(msg) => write!(f, "Not found: {}", msg),
            OEISError::InvalidNumber(msg) => write!(f, "Invalid sequence number: {}", msg),
        }
    }
}

impl Error for OEISError {}

/// Result type for OEIS operations
pub type Result<T> = std::result::Result<T, OEISError>;

/// Represents an OEIS sequence
///
/// Contains information about a sequence from the OEIS database,
/// including its A-number, name, data, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OEISSequence {
    /// Sequence number (e.g., "A000045" for Fibonacci)
    pub number: String,

    /// Sequence name/description
    pub name: String,

    /// The sequence data (first several terms)
    #[serde(default)]
    pub data: Vec<i64>,

    /// Keywords describing the sequence
    #[serde(default)]
    pub keyword: Vec<String>,

    /// Offset (index of first term)
    #[serde(default)]
    pub offset: Vec<i32>,

    /// Comments about the sequence
    #[serde(default)]
    pub comment: Vec<String>,

    /// References
    #[serde(default)]
    pub reference: Vec<String>,

    /// Links to related resources
    #[serde(default)]
    pub link: Vec<String>,

    /// Formulas for the sequence
    #[serde(default)]
    pub formula: Vec<String>,

    /// Example terms with explanations
    #[serde(default)]
    pub example: Vec<String>,

    /// Author information
    #[serde(default)]
    pub author: String,
}

/// Internal structure for OEIS JSON API response
#[derive(Debug, Deserialize)]
struct OEISResponse {
    #[serde(default)]
    results: Vec<OEISResult>,
    #[serde(default)]
    count: i32,
}

/// Internal structure for individual OEIS result
#[derive(Debug, Deserialize)]
struct OEISResult {
    number: i32,
    #[serde(default)]
    data: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    keyword: String,
    #[serde(default)]
    offset: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    comment: Vec<String>,
    #[serde(default)]
    reference: Vec<String>,
    #[serde(default)]
    link: Vec<String>,
    #[serde(default)]
    formula: Vec<String>,
    #[serde(default)]
    example: Vec<String>,
}

impl OEISResult {
    /// Convert to OEISSequence
    fn to_sequence(self) -> OEISSequence {
        // Parse the data string (comma-separated integers)
        let data = parse_data_string(&self.data);

        // Parse keywords
        let keyword = self.keyword
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Parse offset
        let offset = self.offset
            .split(',')
            .filter_map(|s| s.trim().parse::<i32>().ok())
            .collect();

        OEISSequence {
            number: format!("A{:06}", self.number),
            name: self.name,
            data,
            keyword,
            offset,
            comment: self.comment,
            reference: self.reference,
            link: self.link,
            formula: self.formula,
            example: self.example,
            author: self.author,
        }
    }
}

/// Parse OEIS data string into vector of integers
fn parse_data_string(data: &str) -> Vec<i64> {
    data.split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect()
}

/// Client for accessing the OEIS database
///
/// Provides methods to lookup sequences by A-number or search for sequences
/// matching specific criteria.
///
/// # Example
///
/// ```no_run
/// use rustmath_databases::oeis::OEISClient;
///
/// let client = OEISClient::new();
/// let fibonacci = client.lookup("A000045").unwrap();
/// ```
pub struct OEISClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl OEISClient {
    /// Create a new OEIS client
    ///
    /// Uses the default OEIS API endpoint.
    pub fn new() -> Self {
        OEISClient {
            base_url: "https://oeis.org".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create a new OEIS client with custom base URL
    ///
    /// Useful for testing or using mirrors.
    pub fn with_base_url(base_url: String) -> Self {
        OEISClient {
            base_url,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Look up a sequence by its A-number
    ///
    /// # Arguments
    ///
    /// * `number` - Sequence number (e.g., "A000045" or "000045" or "45")
    ///
    /// # Returns
    ///
    /// * `Ok(Some(sequence))` - Sequence found
    /// * `Ok(None)` - Sequence not found
    /// * `Err(...)` - Error occurred
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustmath_databases::oeis::OEISClient;
    /// let client = OEISClient::new();
    /// let fibonacci = client.lookup("A000045")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn lookup(&self, number: &str) -> Result<Option<OEISSequence>> {
        // Normalize the A-number
        let a_number = normalize_a_number(number)?;

        // Query the API
        let url = format!("{}/search?q=id:{}&fmt=json", self.base_url, a_number);

        let response = self.client
            .get(&url)
            .send()
            .map_err(|e| OEISError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(OEISError::NetworkError(format!("HTTP {}", response.status())));
        }

        let oeis_response: OEISResponse = response
            .json()
            .map_err(|e| OEISError::ParseError(e.to_string()))?;

        Ok(oeis_response.results.into_iter().next().map(|r| r.to_sequence()))
    }

    /// Search for sequences matching a list of terms
    ///
    /// # Arguments
    ///
    /// * `terms` - Slice of integer terms to search for
    ///
    /// # Returns
    ///
    /// Vector of matching sequences (may be empty)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustmath_databases::oeis::OEISClient;
    /// let client = OEISClient::new();
    /// // Search for sequences containing 1, 1, 2, 3, 5, 8
    /// let results = client.search_by_terms(&[1, 1, 2, 3, 5, 8])?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search_by_terms(&self, terms: &[i64]) -> Result<Vec<OEISSequence>> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Build search query
        let query: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
        let query_str = query.join(",");

        self.search(&query_str)
    }

    /// Search for sequences matching a text query
    ///
    /// # Arguments
    ///
    /// * `query` - Search query string
    ///
    /// # Returns
    ///
    /// Vector of matching sequences (may be empty)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustmath_databases::oeis::OEISClient;
    /// let client = OEISClient::new();
    /// let results = client.search("fibonacci")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn search(&self, query: &str) -> Result<Vec<OEISSequence>> {
        let url = format!("{}/search?q={}&fmt=json", self.base_url,
                         urlencoding::encode(query));

        let response = self.client
            .get(&url)
            .send()
            .map_err(|e| OEISError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(OEISError::NetworkError(format!("HTTP {}", response.status())));
        }

        let oeis_response: OEISResponse = response
            .json()
            .map_err(|e| OEISError::ParseError(e.to_string()))?;

        Ok(oeis_response.results.into_iter().map(|r| r.to_sequence()).collect())
    }

    /// Get the first n terms of a sequence
    ///
    /// # Arguments
    ///
    /// * `number` - Sequence A-number
    /// * `n` - Number of terms to retrieve
    ///
    /// # Returns
    ///
    /// Vector of the first n terms (or fewer if sequence has fewer terms available)
    pub fn get_terms(&self, number: &str, n: usize) -> Result<Vec<i64>> {
        let seq = self.lookup(number)?
            .ok_or_else(|| OEISError::NotFound(format!("Sequence {} not found", number)))?;

        Ok(seq.data.into_iter().take(n).collect())
    }
}

impl Default for OEISClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize an A-number to standard format (A000000)
///
/// Accepts:
/// - "A000045" (returns as-is)
/// - "000045" (adds A prefix)
/// - "45" (adds A prefix and pads to 6 digits)
fn normalize_a_number(number: &str) -> Result<String> {
    let number = number.trim();

    // Remove 'A' prefix if present
    let digits = if number.starts_with('A') || number.starts_with('a') {
        &number[1..]
    } else {
        number
    };

    // Parse as number
    let num: u32 = digits.parse()
        .map_err(|_| OEISError::InvalidNumber(format!("Invalid A-number: {}", number)))?;

    Ok(format!("A{:06}", num))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_a_number() {
        assert_eq!(normalize_a_number("A000045").unwrap(), "A000045");
        assert_eq!(normalize_a_number("000045").unwrap(), "A000045");
        assert_eq!(normalize_a_number("45").unwrap(), "A000045");
        assert_eq!(normalize_a_number("a000045").unwrap(), "A000045");
    }

    #[test]
    fn test_parse_data_string() {
        assert_eq!(parse_data_string("1,1,2,3,5,8"), vec![1, 1, 2, 3, 5, 8]);
        assert_eq!(parse_data_string("1, 2, 3"), vec![1, 2, 3]);
        assert_eq!(parse_data_string(""), Vec::<i64>::new());
    }

    #[test]
    #[ignore = "Requires network access to OEIS"]
    fn test_lookup_fibonacci() {
        let client = OEISClient::new();
        let seq = client.lookup("A000045").unwrap();

        assert!(seq.is_some());
        let seq = seq.unwrap();
        assert_eq!(seq.number, "A000045");
        assert!(seq.name.to_lowercase().contains("fibonacci"));
        assert!(seq.data.starts_with(&[0, 1, 1, 2, 3, 5, 8]));
    }

    #[test]
    #[ignore = "Requires network access to OEIS"]
    fn test_search_by_terms() {
        let client = OEISClient::new();
        let results = client.search_by_terms(&[1, 1, 2, 3, 5, 8]).unwrap();

        assert!(!results.is_empty());
        // Fibonacci should be in the results
        assert!(results.iter().any(|s| s.number == "A000045"));
    }

    #[test]
    #[ignore = "Requires network access to OEIS"]
    fn test_search_text() {
        let client = OEISClient::new();
        let results = client.search("prime numbers").unwrap();

        assert!(!results.is_empty());
    }
}
