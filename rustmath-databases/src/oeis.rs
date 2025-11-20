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

/// Sequence type classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceType {
    /// Arithmetic progression (constant difference)
    Arithmetic,
    /// Geometric progression (constant ratio)
    Geometric,
    /// Polynomial sequence
    Polynomial(usize),  // degree
    /// Fibonacci-like (linear recurrence)
    Recurrence,
    /// Other/unknown pattern
    Unknown,
}

/// Sequence analysis utilities
pub struct SequenceAnalyzer;

impl SequenceAnalyzer {
    /// Compute differences of a sequence
    ///
    /// Returns successive difference sequences until constant or empty
    pub fn differences(seq: &[i64]) -> Vec<Vec<i64>> {
        if seq.len() < 2 {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current = seq.to_vec();

        for _ in 0..seq.len() - 1 {
            let diff: Vec<i64> = current.windows(2)
                .map(|w| w[1] - w[0])
                .collect();

            if diff.is_empty() {
                break;
            }

            // Check if constant
            let is_constant = diff.windows(2).all(|w| w[0] == w[1]);

            result.push(diff.clone());

            if is_constant || diff.len() < 2 {
                break;
            }

            current = diff;
        }

        result
    }

    /// Compute ratios of consecutive terms
    ///
    /// Useful for detecting geometric sequences
    pub fn ratios(seq: &[i64]) -> Vec<f64> {
        seq.windows(2)
            .filter(|w| w[0] != 0)
            .map(|w| w[1] as f64 / w[0] as f64)
            .collect()
    }

    /// Detect sequence type
    ///
    /// Analyzes a sequence and attempts to classify its pattern
    pub fn detect_type(seq: &[i64]) -> SequenceType {
        if seq.len() < 3 {
            return SequenceType::Unknown;
        }

        // Check for Fibonacci-like recurrence first (a_n = a_{n-1} + a_{n-2})
        // This is a more specific pattern than polynomial
        if seq.len() >= 5 {
            let mut is_fibonacci = true;
            for i in 2..seq.len().min(10) {
                if seq[i] != seq[i - 1] + seq[i - 2] {
                    is_fibonacci = false;
                    break;
                }
            }
            if is_fibonacci {
                return SequenceType::Recurrence;
            }
        }

        // Check for geometric sequence (constant ratios)
        // This should be checked before polynomial
        let ratios = Self::ratios(seq);
        if ratios.len() >= 2 {
            let first_ratio = ratios[0];
            let is_geometric = ratios.iter()
                .all(|&r| (r - first_ratio).abs() < 1e-6);
            if is_geometric && first_ratio != 0.0 && first_ratio != 1.0 {
                return SequenceType::Geometric;
            }
        }

        // Check for arithmetic sequence (constant differences)
        let diffs = Self::differences(seq);
        if !diffs.is_empty() {
            let first_diff = &diffs[0];
            if first_diff.len() >= 2 && first_diff.windows(2).all(|w| w[0] == w[1]) {
                return SequenceType::Arithmetic;
            }

            // Check for polynomial (eventually constant differences)
            for (degree, diff) in diffs.iter().enumerate() {
                if diff.len() >= 2 && diff.windows(2).all(|w| w[0] == w[1]) {
                    return SequenceType::Polynomial(degree + 1);
                }
            }
        }

        SequenceType::Unknown
    }

    /// Predict next terms of a sequence
    ///
    /// Attempts to predict the next n terms based on detected pattern
    pub fn predict_next(seq: &[i64], n: usize) -> Vec<i64> {
        if seq.is_empty() {
            return Vec::new();
        }

        let seq_type = Self::detect_type(seq);
        let mut result = Vec::new();
        let mut extended = seq.to_vec();

        match seq_type {
            SequenceType::Arithmetic => {
                if seq.len() >= 2 {
                    let diff = seq[1] - seq[0];
                    for _ in 0..n {
                        let next = extended.last().unwrap() + diff;
                        result.push(next);
                        extended.push(next);
                    }
                }
            }
            SequenceType::Geometric => {
                if seq.len() >= 2 && seq[0] != 0 {
                    let ratio = seq[1] as f64 / seq[0] as f64;
                    for _ in 0..n {
                        let next = (*extended.last().unwrap() as f64 * ratio).round() as i64;
                        result.push(next);
                        extended.push(next);
                    }
                }
            }
            SequenceType::Polynomial(degree) => {
                // Use differences to extend
                let diffs = Self::differences(seq);
                if degree > 0 && degree <= diffs.len() + 1 {
                    let mut current_diffs = diffs;
                    for _ in 0..n {
                        // Extend from the constant difference level up
                        let mut level = current_diffs.len() - 1;
                        loop {
                            if level == 0 {
                                break;
                            }
                            let constant = current_diffs[level][0];
                            let new_val = current_diffs[level - 1].last().unwrap() + constant;
                            current_diffs[level - 1].push(new_val);
                            level -= 1;
                        }

                        let next = extended.last().unwrap() + current_diffs[0].last().unwrap();
                        result.push(next);
                        extended.push(next);
                    }
                }
            }
            SequenceType::Recurrence => {
                // Fibonacci-like: a_n = a_{n-1} + a_{n-2}
                for _ in 0..n {
                    let len = extended.len();
                    if len >= 2 {
                        let next = extended[len - 1] + extended[len - 2];
                        result.push(next);
                        extended.push(next);
                    }
                }
            }
            SequenceType::Unknown => {
                // No prediction possible
            }
        }

        result
    }

    /// Generate sequence from formula
    ///
    /// Generates terms a_0, a_1, ..., a_{n-1} from a simple formula
    pub fn from_formula<F>(n: usize, formula: F) -> Vec<i64>
    where
        F: Fn(usize) -> i64,
    {
        (0..n).map(formula).collect()
    }
}

impl OEISSequence {
    /// Analyze the sequence
    ///
    /// Returns the detected sequence type
    pub fn analyze_type(&self) -> SequenceType {
        SequenceAnalyzer::detect_type(&self.data)
    }

    /// Get the differences of this sequence
    pub fn differences(&self) -> Vec<Vec<i64>> {
        SequenceAnalyzer::differences(&self.data)
    }

    /// Get the ratios of consecutive terms
    pub fn ratios(&self) -> Vec<f64> {
        SequenceAnalyzer::ratios(&self.data)
    }

    /// Predict the next n terms
    pub fn predict_next(&self, n: usize) -> Vec<i64> {
        SequenceAnalyzer::predict_next(&self.data, n)
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

    #[test]
    fn test_sequence_analysis_arithmetic() {
        let seq = vec![2, 5, 8, 11, 14, 17];  // Arithmetic: +3
        let seq_type = SequenceAnalyzer::detect_type(&seq);
        assert_eq!(seq_type, SequenceType::Arithmetic);

        let next = SequenceAnalyzer::predict_next(&seq, 3);
        assert_eq!(next, vec![20, 23, 26]);
    }

    #[test]
    fn test_sequence_analysis_geometric() {
        let seq = vec![2, 6, 18, 54, 162];  // Geometric: ×3
        let seq_type = SequenceAnalyzer::detect_type(&seq);
        assert_eq!(seq_type, SequenceType::Geometric);

        let next = SequenceAnalyzer::predict_next(&seq, 2);
        assert_eq!(next, vec![486, 1458]);
    }

    #[test]
    fn test_sequence_analysis_polynomial() {
        let seq = vec![1, 4, 9, 16, 25, 36];  // Squares: n^2
        let seq_type = SequenceAnalyzer::detect_type(&seq);
        assert_eq!(seq_type, SequenceType::Polynomial(2));

        let next = SequenceAnalyzer::predict_next(&seq, 2);
        assert_eq!(next, vec![49, 64]);
    }

    #[test]
    fn test_sequence_analysis_fibonacci() {
        let seq = vec![0, 1, 1, 2, 3, 5, 8, 13];  // Fibonacci
        let seq_type = SequenceAnalyzer::detect_type(&seq);
        assert_eq!(seq_type, SequenceType::Recurrence);

        let next = SequenceAnalyzer::predict_next(&seq, 3);
        assert_eq!(next, vec![21, 34, 55]);
    }

    #[test]
    fn test_sequence_differences() {
        let seq = vec![1, 4, 9, 16, 25];
        let diffs = SequenceAnalyzer::differences(&seq);

        assert_eq!(diffs.len(), 2);
        assert_eq!(diffs[0], vec![3, 5, 7, 9]);  // First difference
        assert_eq!(diffs[1], vec![2, 2, 2]);     // Second difference (constant)
    }

    #[test]
    fn test_sequence_ratios() {
        let seq = vec![1, 2, 4, 8, 16];
        let ratios = SequenceAnalyzer::ratios(&seq);

        assert_eq!(ratios.len(), 4);
        assert!((ratios[0] - 2.0).abs() < 1e-10);
        assert!((ratios[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_formula() {
        let squares = SequenceAnalyzer::from_formula(10, |n| (n * n) as i64);
        assert_eq!(squares, vec![0, 1, 4, 9, 16, 25, 36, 49, 64, 81]);

        let cubes = SequenceAnalyzer::from_formula(5, |n| (n * n * n) as i64);
        assert_eq!(cubes, vec![0, 1, 8, 27, 64]);
    }
}
