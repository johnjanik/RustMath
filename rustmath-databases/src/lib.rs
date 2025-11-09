//! Database interfaces for mathematical sequences and tables
//!
//! Provides access to external mathematical databases including:
//! - OEIS (Online Encyclopedia of Integer Sequences)
//! - Cunningham tables (planned)
//! - Elliptic curve databases (planned)
//!
//! # Example: OEIS Interface
//!
//! ```no_run
//! use rustmath_databases::oeis::OEISClient;
//!
//! let client = OEISClient::new();
//!
//! // Look up Fibonacci sequence
//! if let Ok(Some(seq)) = client.lookup("A000045") {
//!     println!("Fibonacci: {}", seq.name);
//!     println!("First terms: {:?}", &seq.data[..10]);
//! }
//!
//! // Search for sequences
//! if let Ok(results) = client.search_by_terms(&[1, 1, 2, 3, 5, 8, 13]) {
//!     for seq in results.iter().take(3) {
//!         println!("{}: {}", seq.number, seq.name);
//!     }
//! }
//! ```

pub mod oeis;

pub use oeis::{OEISClient, OEISSequence, OEISError};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_client_creation() {
        let _client = OEISClient::new();
        let _client2 = OEISClient::default();
    }
}
