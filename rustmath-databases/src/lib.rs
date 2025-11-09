//! Database interfaces for mathematical sequences and tables
//!
//! Provides access to external mathematical databases including:
//! - OEIS (Online Encyclopedia of Integer Sequences)
//! - Cunningham tables (factorizations of b^n ± 1)
//! - Cremona elliptic curve database
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
//!
//! # Example: Cunningham Tables
//!
//! ```
//! use rustmath_databases::cunningham::{CunninghamTables, CunninghamNumber};
//!
//! let tables = CunninghamTables::new();
//!
//! // Look up 2^10 - 1 = 1023 = 3 × 11 × 31
//! let num = CunninghamNumber::new(2, 10, false);
//! if let Ok(fact) = tables.factorization(&num) {
//!     println!("{}", fact);
//! }
//! ```
//!
//! # Example: Cremona Database
//!
//! ```
//! use rustmath_databases::cremona::CremonaDatabase;
//!
//! let db = CremonaDatabase::new();
//!
//! // Look up curve 11a1
//! if let Some(curve) = db.lookup_curve("11a1") {
//!     println!("Curve: {}", curve.label);
//!     println!("Rank: {}", curve.rank);
//! }
//! ```

pub mod oeis;
pub mod cunningham;
pub mod cremona;

pub use oeis::{OEISClient, OEISSequence, OEISError};
pub use cunningham::{CunninghamTables, CunninghamNumber, Factorization, Factor, CunninghamError};
pub use cremona::{CremonaDatabase, EllipticCurve, CurveLabel, WeierstrassEquation, Point, CremonaError};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_client_creation() {
        let _client = OEISClient::new();
        let _client2 = OEISClient::default();
    }
}
