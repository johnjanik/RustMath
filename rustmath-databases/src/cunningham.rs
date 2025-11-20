//! Cunningham Tables interface
//!
//! Provides access to factorizations of numbers of the form b^n ± 1.
//! The Cunningham Project maintains tables of factorizations for bases
//! b = 2, 3, 5, 6, 7, 10, 11, 12.
//!
//! # Example
//!
//! ```
//! use rustmath_databases::cunningham::{CunninghamTables, CunninghamNumber};
//!
//! let tables = CunninghamTables::new();
//!
//! // Look up 2^10 - 1 = 1023 = 3 × 11 × 31
//! let num = CunninghamNumber::new(2, 10, false);
//! println!("Factorization of 2^10 - 1: {:?}", tables.factorization(&num));
//! ```

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;

/// Error type for Cunningham table operations
#[derive(Debug)]
pub enum CunninghamError {
    /// File I/O error
    IoError(String),
    /// Parse error
    ParseError(String),
    /// Factorization not found in tables
    NotFound(String),
    /// Invalid base (must be 2, 3, 5, 6, 7, 10, 11, or 12)
    InvalidBase(u32),
}

impl fmt::Display for CunninghamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CunninghamError::IoError(msg) => write!(f, "I/O error: {}", msg),
            CunninghamError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            CunninghamError::NotFound(msg) => write!(f, "Not found: {}", msg),
            CunninghamError::InvalidBase(b) => write!(f, "Invalid base: {}", b),
        }
    }
}

impl Error for CunninghamError {}

/// Result type for Cunningham operations
pub type Result<T> = std::result::Result<T, CunninghamError>;

/// Represents a Cunningham number b^n ± 1
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CunninghamNumber {
    /// Base (2, 3, 5, 6, 7, 10, 11, or 12)
    pub base: u32,
    /// Exponent
    pub exponent: u32,
    /// True for b^n + 1, false for b^n - 1
    pub is_plus: bool,
}

impl CunninghamNumber {
    /// Create a new Cunningham number
    ///
    /// # Arguments
    ///
    /// * `base` - Base value (2, 3, 5, 6, 7, 10, 11, or 12)
    /// * `exponent` - Exponent value
    /// * `is_plus` - true for b^n + 1, false for b^n - 1
    pub fn new(base: u32, exponent: u32, is_plus: bool) -> Self {
        CunninghamNumber {
            base,
            exponent,
            is_plus,
        }
    }

    /// Get the standard notation (e.g., "2,10-" for 2^10 - 1)
    pub fn notation(&self) -> String {
        format!("{},{}{}",
                self.base,
                self.exponent,
                if self.is_plus { "+" } else { "-" })
    }

    /// Validate that the base is supported
    pub fn validate(&self) -> Result<()> {
        const VALID_BASES: &[u32] = &[2, 3, 5, 6, 7, 10, 11, 12];
        if VALID_BASES.contains(&self.base) {
            Ok(())
        } else {
            Err(CunninghamError::InvalidBase(self.base))
        }
    }
}

impl fmt::Display for CunninghamNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}^{} {} 1",
               self.base,
               self.exponent,
               if self.is_plus { "+" } else { "-" })
    }
}

/// Represents a prime factor
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Factor {
    /// The factor value (as string to handle large numbers)
    pub value: String,
    /// Exponent (multiplicity)
    pub exponent: u32,
    /// True if this is a known prime, false if composite or unknown
    pub is_prime: bool,
}

impl Factor {
    /// Create a new prime factor
    pub fn new_prime(value: String) -> Self {
        Factor {
            value,
            exponent: 1,
            is_prime: true,
        }
    }

    /// Create a new composite factor
    pub fn new_composite(value: String) -> Self {
        Factor {
            value,
            exponent: 1,
            is_prime: false,
        }
    }

    /// Create a factor with given exponent
    pub fn with_exponent(mut self, exp: u32) -> Self {
        self.exponent = exp;
        self
    }
}

/// Represents a factorization of a Cunningham number
#[derive(Debug, Clone)]
pub struct Factorization {
    /// The Cunningham number being factored
    pub number: CunninghamNumber,
    /// List of factors
    pub factors: Vec<Factor>,
    /// True if factorization is complete (fully factored)
    pub is_complete: bool,
}

impl Factorization {
    /// Create a new factorization
    pub fn new(number: CunninghamNumber) -> Self {
        Factorization {
            number,
            factors: Vec::new(),
            is_complete: false,
        }
    }

    /// Add a factor
    pub fn add_factor(&mut self, factor: Factor) {
        self.factors.push(factor);
    }

    /// Check if the factorization has any composite factors
    pub fn has_composite_factors(&self) -> bool {
        self.factors.iter().any(|f| !f.is_prime)
    }
}

impl fmt::Display for Factorization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = ", self.number)?;
        for (i, factor) in self.factors.iter().enumerate() {
            if i > 0 {
                write!(f, " × ")?;
            }
            if factor.exponent > 1 {
                write!(f, "{}^{}", factor.value, factor.exponent)?;
            } else {
                write!(f, "{}", factor.value)?;
            }
            if !factor.is_prime {
                write!(f, " (composite)")?;
            }
        }
        Ok(())
    }
}

/// Client for accessing Cunningham tables
///
/// The Cunningham tables contain factorizations of numbers of the form
/// b^n ± 1 for small bases and exponents.
pub struct CunninghamTables {
    /// In-memory cache of factorizations
    factorizations: HashMap<CunninghamNumber, Factorization>,
    /// Path to data directory (if using local files)
    data_path: Option<String>,
}

impl CunninghamTables {
    /// Create a new Cunningham tables client with built-in data
    pub fn new() -> Self {
        let mut tables = CunninghamTables {
            factorizations: HashMap::new(),
            data_path: None,
        };

        // Initialize with some well-known factorizations
        tables.load_builtin_factorizations();
        tables
    }

    /// Create a client that loads data from a directory
    ///
    /// # Arguments
    ///
    /// * `data_path` - Path to directory containing Cunningham table files
    pub fn from_directory<P: AsRef<Path>>(data_path: P) -> Result<Self> {
        let mut tables = CunninghamTables {
            factorizations: HashMap::new(),
            data_path: Some(data_path.as_ref().to_string_lossy().to_string()),
        };

        tables.load_builtin_factorizations();
        tables.load_from_directory(&data_path)?;
        Ok(tables)
    }

    /// Load built-in factorizations for small cases
    fn load_builtin_factorizations(&mut self) {
        // Some well-known factorizations
        self.add_builtin(2, 2, false, vec!["3"]);  // 2^2 - 1 = 3
        self.add_builtin(2, 3, false, vec!["7"]);  // 2^3 - 1 = 7
        self.add_builtin(2, 4, false, vec!["3", "5"]);  // 2^4 - 1 = 15 = 3 × 5
        self.add_builtin(2, 5, false, vec!["31"]);  // 2^5 - 1 = 31
        self.add_builtin(2, 6, false, vec!["3", "3", "7"]);  // 2^6 - 1 = 63 = 3^2 × 7
        self.add_builtin(2, 7, false, vec!["127"]);  // 2^7 - 1 = 127
        self.add_builtin(2, 8, false, vec!["3", "5", "17"]);  // 2^8 - 1 = 255
        self.add_builtin(2, 9, false, vec!["7", "73"]);  // 2^9 - 1 = 511
        self.add_builtin(2, 10, false, vec!["3", "11", "31"]);  // 2^10 - 1 = 1023

        self.add_builtin(2, 2, true, vec!["5"]);  // 2^2 + 1 = 5
        self.add_builtin(2, 3, true, vec!["3", "3"]);  // 2^3 + 1 = 9 = 3^2
        self.add_builtin(2, 4, true, vec!["17"]);  // 2^4 + 1 = 17 (Fermat prime)
        self.add_builtin(2, 5, true, vec!["3", "11"]);  // 2^5 + 1 = 33
        self.add_builtin(2, 6, true, vec!["5", "13"]);  // 2^6 + 1 = 65
        self.add_builtin(2, 7, true, vec!["3", "43"]);  // 2^7 + 1 = 129
        self.add_builtin(2, 8, true, vec!["257"]);  // 2^8 + 1 = 257 (Fermat prime)

        // Some base 3 factorizations
        self.add_builtin(3, 2, false, vec!["2", "2", "2"]);  // 3^2 - 1 = 8 = 2^3
        self.add_builtin(3, 3, false, vec!["2", "13"]);  // 3^3 - 1 = 26
        self.add_builtin(3, 4, false, vec!["2", "2", "2", "2", "5"]);  // 3^4 - 1 = 80
        self.add_builtin(3, 5, false, vec!["2", "11", "11"]);  // 3^5 - 1 = 242

        self.add_builtin(3, 2, true, vec!["2", "5"]);  // 3^2 + 1 = 10
        self.add_builtin(3, 3, true, vec!["2", "2", "7"]);  // 3^3 + 1 = 28
        self.add_builtin(3, 4, true, vec!["2", "41"]);  // 3^4 + 1 = 82

        // Base 5
        self.add_builtin(5, 2, false, vec!["2", "2", "2", "3"]);  // 5^2 - 1 = 24
        self.add_builtin(5, 3, false, vec!["2", "2", "31"]);  // 5^3 - 1 = 124

        self.add_builtin(5, 2, true, vec!["2", "13"]);  // 5^2 + 1 = 26
        self.add_builtin(5, 3, true, vec!["2", "3", "3", "7"]);  // 5^3 + 1 = 126

        // Base 6
        self.add_builtin(6, 2, false, vec!["5", "7"]);  // 6^2 - 1 = 35
        self.add_builtin(6, 3, false, vec!["5", "43"]);  // 6^3 - 1 = 215
        self.add_builtin(6, 4, false, vec!["5", "7", "37"]);  // 6^4 - 1 = 1295
        self.add_builtin(6, 5, false, vec!["5", "11", "71"]);  // 6^5 - 1 = 7775

        self.add_builtin(6, 2, true, vec!["37"]);  // 6^2 + 1 = 37
        self.add_builtin(6, 3, true, vec!["7", "31"]);  // 6^3 + 1 = 217
        self.add_builtin(6, 4, true, vec!["13", "101"]);  // 6^4 + 1 = 1297

        // Base 7
        self.add_builtin(7, 2, false, vec!["2", "2", "2", "2", "3"]);  // 7^2 - 1 = 48
        self.add_builtin(7, 3, false, vec!["2", "3", "3", "19"]);  // 7^3 - 1 = 342
        self.add_builtin(7, 4, false, vec!["2", "2", "2", "2", "3", "5", "5"]);  // 7^4 - 1 = 2400
        self.add_builtin(7, 5, false, vec!["2", "3", "2801"]);  // 7^5 - 1 = 16806

        self.add_builtin(7, 2, true, vec!["2", "5", "5"]);  // 7^2 + 1 = 50
        self.add_builtin(7, 3, true, vec!["2", "2", "2", "43"]);  // 7^3 + 1 = 344
        self.add_builtin(7, 4, true, vec!["2", "5", "13", "37"]);  // 7^4 + 1 = 2402

        // Base 10
        self.add_builtin(10, 2, false, vec!["3", "3", "11"]);  // 10^2 - 1 = 99
        self.add_builtin(10, 3, false, vec!["3", "3", "3", "37"]);  // 10^3 - 1 = 999
        self.add_builtin(10, 4, false, vec!["3", "3", "11", "101"]);  // 10^4 - 1 = 9999
        self.add_builtin(10, 5, false, vec!["3", "3", "41", "271"]);  // 10^5 - 1 = 99999
        self.add_builtin(10, 6, false, vec!["3", "3", "3", "7", "11", "13", "37"]);  // 10^6 - 1 = 999999

        self.add_builtin(10, 2, true, vec!["101"]);  // 10^2 + 1 = 101
        self.add_builtin(10, 3, true, vec!["7", "11", "13"]);  // 10^3 + 1 = 1001
        self.add_builtin(10, 4, true, vec!["73", "137"]);  // 10^4 + 1 = 10001

        // Base 11
        self.add_builtin(11, 2, false, vec!["2", "2", "2", "3", "5"]);  // 11^2 - 1 = 120
        self.add_builtin(11, 3, false, vec!["2", "2", "2", "3", "5", "11"]);  // 11^3 - 1 = 1320
        self.add_builtin(11, 4, false, vec!["2", "2", "2", "3", "5", "61"]);  // 11^4 - 1 = 14640
        self.add_builtin(11, 5, false, vec!["2", "2", "2", "3", "5", "11", "61"]);  // 11^5 - 1 = 161050

        self.add_builtin(11, 2, true, vec!["2", "61"]);  // 11^2 + 1 = 122
        self.add_builtin(11, 3, true, vec!["2", "2", "331"]);  // 11^3 + 1 = 1332
        self.add_builtin(11, 4, true, vec!["2", "7321"]);  // 11^4 + 1 = 14642

        // Base 12
        self.add_builtin(12, 2, false, vec!["11", "13"]);  // 12^2 - 1 = 143
        self.add_builtin(12, 3, false, vec!["11", "157"]);  // 12^3 - 1 = 1727
        self.add_builtin(12, 4, false, vec!["11", "13", "127"]);  // 12^4 - 1 = 20735
        self.add_builtin(12, 5, false, vec!["11", "11", "23", "89"]);  // 12^5 - 1 = 248831

        self.add_builtin(12, 2, true, vec!["5", "29"]);  // 12^2 + 1 = 145
        self.add_builtin(12, 3, true, vec!["13", "133"]);  // 12^3 + 1 = 1729
        self.add_builtin(12, 4, true, vec!["17", "1217"]);  // 12^4 + 1 = 20737
    }

    /// Add a built-in factorization
    fn add_builtin(&mut self, base: u32, exp: u32, is_plus: bool, factors: Vec<&str>) {
        let number = CunninghamNumber::new(base, exp, is_plus);
        let mut factorization = Factorization::new(number.clone());

        // Group consecutive equal factors
        let mut factor_counts: HashMap<String, u32> = HashMap::new();
        for factor in factors {
            *factor_counts.entry(factor.to_string()).or_insert(0) += 1;
        }

        for (value, count) in factor_counts {
            let factor = Factor::new_prime(value).with_exponent(count);
            factorization.add_factor(factor);
        }

        factorization.is_complete = true;
        self.factorizations.insert(number, factorization);
    }

    /// Load factorizations from a directory
    fn load_from_directory<P: AsRef<Path>>(&mut self, _path: P) -> Result<()> {
        // Placeholder for loading from actual Cunningham table files
        // Real implementation would parse files like "t2m", "t2p", etc.
        Ok(())
    }

    /// Look up the factorization of a Cunningham number
    ///
    /// # Arguments
    ///
    /// * `number` - The Cunningham number to factor
    ///
    /// # Returns
    ///
    /// The factorization if available in the tables
    pub fn factorization(&self, number: &CunninghamNumber) -> Result<&Factorization> {
        number.validate()?;

        self.factorizations.get(number)
            .ok_or_else(|| CunninghamError::NotFound(
                format!("Factorization of {} not in tables", number.notation())
            ))
    }

    /// Get all available factorizations for a given base
    ///
    /// # Arguments
    ///
    /// * `base` - The base to query
    ///
    /// # Returns
    ///
    /// Vector of all factorizations for this base
    pub fn factorizations_for_base(&self, base: u32) -> Vec<&Factorization> {
        self.factorizations
            .iter()
            .filter(|(k, _)| k.base == base)
            .map(|(_, v)| v)
            .collect()
    }

    /// Add a factorization to the tables
    ///
    /// Useful for extending the built-in tables with custom factorizations
    pub fn add_factorization(&mut self, factorization: Factorization) -> Result<()> {
        factorization.number.validate()?;
        self.factorizations.insert(factorization.number.clone(), factorization);
        Ok(())
    }

    /// Check if a factorization is available
    pub fn has_factorization(&self, number: &CunninghamNumber) -> bool {
        self.factorizations.contains_key(number)
    }
}

impl Default for CunninghamTables {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cunningham_number_notation() {
        let num = CunninghamNumber::new(2, 10, false);
        assert_eq!(num.notation(), "2,10-");

        let num2 = CunninghamNumber::new(3, 5, true);
        assert_eq!(num2.notation(), "3,5+");
    }

    #[test]
    fn test_cunningham_number_display() {
        let num = CunninghamNumber::new(2, 10, false);
        assert_eq!(format!("{}", num), "2^10 - 1");

        let num2 = CunninghamNumber::new(3, 5, true);
        assert_eq!(format!("{}", num2), "3^5 + 1");
    }

    #[test]
    fn test_base_validation() {
        let valid = CunninghamNumber::new(2, 10, false);
        assert!(valid.validate().is_ok());

        let invalid = CunninghamNumber::new(13, 10, false);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_builtin_factorizations() {
        let tables = CunninghamTables::new();

        // Test 2^10 - 1 = 1023 = 3 × 11 × 31
        let num = CunninghamNumber::new(2, 10, false);
        let fact = tables.factorization(&num).unwrap();
        assert_eq!(fact.factors.len(), 3);
        assert!(fact.is_complete);
    }

    #[test]
    fn test_factorization_display() {
        let tables = CunninghamTables::new();
        let num = CunninghamNumber::new(2, 4, false);
        let fact = tables.factorization(&num).unwrap();
        let display = format!("{}", fact);
        assert!(display.contains("2^4 - 1"));
        assert!(display.contains("3"));
        assert!(display.contains("5"));
    }

    #[test]
    fn test_factorizations_for_base() {
        let tables = CunninghamTables::new();
        let base2_facts = tables.factorizations_for_base(2);
        assert!(!base2_facts.is_empty());

        for fact in base2_facts {
            assert_eq!(fact.number.base, 2);
        }
    }

    #[test]
    fn test_add_custom_factorization() {
        let mut tables = CunninghamTables::new();

        let num = CunninghamNumber::new(2, 11, false);
        let mut fact = Factorization::new(num.clone());
        fact.add_factor(Factor::new_prime("23".to_string()));
        fact.add_factor(Factor::new_prime("89".to_string()));
        fact.is_complete = true;

        tables.add_factorization(fact).unwrap();

        assert!(tables.has_factorization(&num));
        let retrieved = tables.factorization(&num).unwrap();
        assert_eq!(retrieved.factors.len(), 2);
    }

    #[test]
    fn test_mersenne_primes() {
        let tables = CunninghamTables::new();

        // 2^3 - 1 = 7 (Mersenne prime)
        let m3 = CunninghamNumber::new(2, 3, false);
        let fact = tables.factorization(&m3).unwrap();
        assert_eq!(fact.factors.len(), 1);
        assert_eq!(fact.factors[0].value, "7");

        // 2^5 - 1 = 31 (Mersenne prime)
        let m5 = CunninghamNumber::new(2, 5, false);
        let fact = tables.factorization(&m5).unwrap();
        assert_eq!(fact.factors.len(), 1);
        assert_eq!(fact.factors[0].value, "31");
    }

    #[test]
    fn test_fermat_numbers() {
        let tables = CunninghamTables::new();

        // 2^4 + 1 = 17 (Fermat prime F2)
        let f2 = CunninghamNumber::new(2, 4, true);
        let fact = tables.factorization(&f2).unwrap();
        assert_eq!(fact.factors.len(), 1);
        assert_eq!(fact.factors[0].value, "17");

        // 2^8 + 1 = 257 (Fermat prime F3)
        let f3 = CunninghamNumber::new(2, 8, true);
        let fact = tables.factorization(&f3).unwrap();
        assert_eq!(fact.factors.len(), 1);
        assert_eq!(fact.factors[0].value, "257");
    }

    #[test]
    fn test_base_6_factorizations() {
        let tables = CunninghamTables::new();

        // 6^2 - 1 = 35 = 5 × 7
        let num = CunninghamNumber::new(6, 2, false);
        let fact = tables.factorization(&num).unwrap();
        assert!(fact.is_complete);

        // 6^2 + 1 = 37 (prime)
        let num2 = CunninghamNumber::new(6, 2, true);
        let fact2 = tables.factorization(&num2).unwrap();
        assert_eq!(fact2.factors.len(), 1);
    }

    #[test]
    fn test_base_10_factorizations() {
        let tables = CunninghamTables::new();

        // 10^2 - 1 = 99 = 3^2 × 11
        let num = CunninghamNumber::new(10, 2, false);
        let fact = tables.factorization(&num).unwrap();
        assert!(fact.is_complete);

        // 10^2 + 1 = 101 (prime)
        let num2 = CunninghamNumber::new(10, 2, true);
        let fact2 = tables.factorization(&num2).unwrap();
        assert_eq!(fact2.factors.len(), 1);
        assert_eq!(fact2.factors[0].value, "101");
    }

    #[test]
    fn test_all_valid_bases() {
        let tables = CunninghamTables::new();
        let valid_bases = vec![2, 3, 5, 6, 7, 10, 11, 12];

        for base in valid_bases {
            let base_facts = tables.factorizations_for_base(base);
            assert!(!base_facts.is_empty(), "Base {} should have factorizations", base);
        }
    }
}
