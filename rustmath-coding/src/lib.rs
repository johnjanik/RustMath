//! # RustMath Coding Theory
//!
//! Implementation of coding theory, including linear codes, error-correcting codes,
//! and various specific code constructions (Hamming, Reed-Solomon, BCH, Golay).
//!
//! ## Features
//!
//! - **Linear Codes**: Generator and parity check matrices (`u64` GF(p) and
//!   fully generic over any `Field`, including extension fields GF(p^n))
//! - **Encoding/Decoding**: Message encoding and syndrome decoding
//! - **Weight enumerators & MacWilliams identity**: exhaustive enumeration
//!   under an honest budget, with the exact-integer MacWilliams transform
//! - **Hamming Codes**: Perfect single-error-correcting codes
//! - **Reed-Solomon Codes**: MDS codes with a real algebraic decoder
//!   (Berlekamp-Massey + Chien search + Forney's formula)
//! - **BCH Codes**: Binary BCH codes for error correction
//! - **Golay Codes**: Binary and ternary perfect codes

pub mod linear_code;
pub mod generic_code;
pub mod hamming;
pub mod reed_solomon;
pub mod reed_solomon_generic;
pub mod bch;
pub mod golay;
pub mod syndrome;

pub use linear_code::LinearCode;
pub use generic_code::{
    finite_field_elements, macwilliams_transform, prime_field_elements, GenericLinearCode,
};
pub use hamming::HammingCode;
pub use reed_solomon::ReedSolomonCode;
pub use reed_solomon_generic::{
    find_primitive_element, find_primitive_root_gfp, GenericReedSolomonCode,
};
pub use bch::BCHCode;
pub use golay::{BinaryGolayCode, TernaryGolayCode};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_code_basics() {
        // [7,4,3] Hamming code generator matrix
        let g = vec![
            vec![1, 0, 0, 0, 1, 1, 0],
            vec![0, 1, 0, 0, 1, 0, 1],
            vec![0, 0, 1, 0, 0, 1, 1],
            vec![0, 0, 0, 1, 1, 1, 1],
        ];

        let code = LinearCode::from_generator_matrix(g, 2);
        assert_eq!(code.length(), 7);
        assert_eq!(code.dimension(), 4);
    }

    #[test]
    fn test_hamming_code() {
        let ham = HammingCode::new(3); // [7,4,3] Hamming code
        assert_eq!(ham.length(), 7);
        assert_eq!(ham.dimension(), 4);
        assert_eq!(ham.minimum_distance(), 3);
    }

    #[test]
    fn test_encoding_decoding() {
        let ham = HammingCode::new(3);
        let message = vec![1, 0, 1, 1];
        let codeword = ham.encode(&message).unwrap();
        assert_eq!(codeword.len(), 7);

        // No errors
        let decoded = ham.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_error_correction() {
        let ham = HammingCode::new(3);
        let message = vec![1, 0, 1, 1];
        let mut codeword = ham.encode(&message).unwrap();

        // Introduce single-bit error
        codeword[2] ^= 1;

        // Should correct the error
        let decoded = ham.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_golay_code() {
        let golay = BinaryGolayCode::new();
        assert_eq!(golay.length(), 23);
        assert_eq!(golay.dimension(), 12);
        assert_eq!(golay.minimum_distance(), 7);
    }
}
