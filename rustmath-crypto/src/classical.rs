//! Classical cryptography - Caesar, Vigenère, Substitution, and Hill ciphers

use rustmath_core::{MathError, Result};
use rustmath_matrix::Matrix;
use rustmath_integers::Integer;
use std::collections::HashMap;

/// Caesar cipher - shift each letter by a fixed amount
pub fn caesar_encrypt(text: &str, shift: u8) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_uppercase() {
                let shifted = ((c as u8 - b'A' + shift) % 26) + b'A';
                shifted as char
            } else if c.is_ascii_lowercase() {
                let shifted = ((c as u8 - b'a' + shift) % 26) + b'a';
                shifted as char
            } else {
                c // Leave non-alphabetic characters unchanged
            }
        })
        .collect()
}

/// Caesar cipher decryption
pub fn caesar_decrypt(text: &str, shift: u8) -> String {
    caesar_encrypt(text, 26 - (shift % 26))
}

/// Vigenère cipher encryption
pub fn vigenere_encrypt(text: &str, key: &str) -> Result<String> {
    if key.is_empty() {
        return Err(MathError::InvalidArgument(
            "Key must not be empty".to_string(),
        ));
    }

    let key_upper: String = key.to_uppercase();
    let key_bytes: Vec<u8> = key_upper.bytes().collect();

    let result: String = text
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if c.is_ascii_uppercase() {
                let shift = key_bytes[i % key_bytes.len()] - b'A';
                let shifted = ((c as u8 - b'A' + shift) % 26) + b'A';
                shifted as char
            } else if c.is_ascii_lowercase() {
                let shift = key_bytes[i % key_bytes.len()] - b'A';
                let shifted = ((c as u8 - b'a' + shift) % 26) + b'a';
                shifted as char
            } else {
                c
            }
        })
        .collect();

    Ok(result)
}

/// Vigenère cipher decryption
pub fn vigenere_decrypt(text: &str, key: &str) -> Result<String> {
    if key.is_empty() {
        return Err(MathError::InvalidArgument(
            "Key must not be empty".to_string(),
        ));
    }

    let key_upper: String = key.to_uppercase();
    let key_bytes: Vec<u8> = key_upper.bytes().collect();

    let result: String = text
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if c.is_ascii_uppercase() {
                let shift = key_bytes[i % key_bytes.len()] - b'A';
                let shifted = ((c as u8 - b'A' + 26 - shift) % 26) + b'A';
                shifted as char
            } else if c.is_ascii_lowercase() {
                let shift = key_bytes[i % key_bytes.len()] - b'A';
                let shifted = ((c as u8 - b'a' + 26 - shift) % 26) + b'a';
                shifted as char
            } else {
                c
            }
        })
        .collect();

    Ok(result)
}

/// Substitution cipher using a key mapping
pub fn substitution_encrypt(text: &str, key_map: &HashMap<char, char>) -> String {
    text.chars()
        .map(|c| *key_map.get(&c).unwrap_or(&c))
        .collect()
}

/// Substitution cipher decryption - reverse the key mapping
pub fn substitution_decrypt(text: &str, key_map: &HashMap<char, char>) -> String {
    // Create reverse mapping
    let reverse_map: HashMap<char, char> = key_map.iter().map(|(k, v)| (*v, *k)).collect();
    text.chars()
        .map(|c| *reverse_map.get(&c).unwrap_or(&c))
        .collect()
}

/// Generate a simple substitution key from a keyword
pub fn generate_substitution_key(keyword: &str) -> HashMap<char, char> {
    let mut key_map = HashMap::new();
    let keyword_upper: String = keyword.to_uppercase();

    // Add unique letters from keyword
    let mut used_letters = std::collections::HashSet::new();
    let mut cipher_alphabet = String::new();

    for c in keyword_upper.chars() {
        if c.is_ascii_uppercase() && !used_letters.contains(&c) {
            used_letters.insert(c);
            cipher_alphabet.push(c);
        }
    }

    // Add remaining letters
    for c in b'A'..=b'Z' {
        let ch = c as char;
        if !used_letters.contains(&ch) {
            cipher_alphabet.push(ch);
        }
    }

    // Create mapping
    for (i, plain_char) in (b'A'..=b'Z').enumerate() {
        key_map.insert(
            plain_char as char,
            cipher_alphabet.chars().nth(i).unwrap(),
        );
    }

    key_map
}

/// Hill cipher encryption using matrix multiplication
/// Text must be in blocks matching matrix size
pub fn hill_encrypt(text: &str, key_matrix: &Matrix<Integer>) -> Result<String> {
    if key_matrix.rows() != key_matrix.cols() {
        return Err(MathError::InvalidArgument(
            "Key matrix must be square".to_string(),
        ));
    }

    let n = key_matrix.rows();
    let text_upper: String = text.to_uppercase().chars().filter(|c| c.is_ascii_uppercase()).collect();

    // Pad text if necessary
    let mut padded = text_upper.clone();
    while padded.len() % n != 0 {
        padded.push('X');
    }

    let modulus = Integer::from(26);
    let mut result = String::new();

    // Process each block
    for chunk in padded.as_bytes().chunks(n) {
        // Convert letters to numbers (A=0, B=1, etc.) as exact Integers
        let plain_vec: Vec<Integer> = chunk
            .iter()
            .map(|&b| Integer::from((b - b'A') as i64))
            .collect();

        // Multiply by key matrix (mod 26), staying in exact Integer arithmetic
        // throughout (never through string round-tripping).
        let mut cipher_vec: Vec<u8> = Vec::with_capacity(n);
        for i in 0..n {
            let mut sum = Integer::zero();
            for j in 0..n {
                let key_elem = key_matrix.get(i, j).unwrap();
                sum = sum + key_elem.clone() * plain_vec[j].clone();
            }
            let mut residue = sum % modulus.clone();
            if residue.signum() < 0 {
                residue = residue + modulus.clone();
            }
            // residue is exactly in [0, 26), so this always fits in i64;
            // Integer::to_i64 panics (rather than silently truncating) if
            // it ever didn't, which would indicate a real bug upstream.
            cipher_vec.push(residue.to_i64() as u8);
        }

        // Convert back to letters
        for val in cipher_vec {
            let letter = (val + b'A') as char;
            result.push(letter);
        }
    }

    Ok(result)
}

/// Hill cipher decryption using inverse matrix
pub fn hill_decrypt(text: &str, key_matrix: &Matrix<Integer>) -> Result<String> {
    // For decryption, we need the inverse of the key matrix (mod 26)
    // This is complex, so we'll provide a simplified version
    // In practice, you'd compute the modular inverse of the matrix

    // For now, return an error indicating this needs the inverse
    Err(MathError::NotImplemented(
        "Hill cipher decryption requires modular matrix inversion".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caesar_cipher() {
        let plaintext = "HELLO";
        let shift = 3;

        let encrypted = caesar_encrypt(plaintext, shift);
        assert_eq!(encrypted, "KHOOR");

        let decrypted = caesar_decrypt(&encrypted, shift);
        assert_eq!(decrypted, "HELLO");
    }

    #[test]
    fn test_caesar_lowercase() {
        let plaintext = "hello world";
        let shift = 5;

        let encrypted = caesar_encrypt(plaintext, shift);
        assert_eq!(encrypted, "mjqqt btwqi");

        let decrypted = caesar_decrypt(&encrypted, shift);
        assert_eq!(decrypted, "hello world");
    }

    #[test]
    fn test_vigenere_cipher() {
        let plaintext = "ATTACKATDAWN";
        let key = "LEMON";

        let encrypted = vigenere_encrypt(plaintext, key).unwrap();
        assert_eq!(encrypted, "LXFOPVEFRNHR");

        let decrypted = vigenere_decrypt(&encrypted, key).unwrap();
        assert_eq!(decrypted, "ATTACKATDAWN");
    }

    #[test]
    fn test_vigenere_lowercase() {
        let plaintext = "hello";
        let key = "KEY";

        let encrypted = vigenere_encrypt(plaintext, key).unwrap();
        let decrypted = vigenere_decrypt(&encrypted, key).unwrap();

        assert_eq!(decrypted, "hello");
    }

    #[test]
    fn test_substitution_cipher() {
        let mut key_map = HashMap::new();
        key_map.insert('A', 'Z');
        key_map.insert('B', 'Y');
        key_map.insert('C', 'X');

        let plaintext = "ABC";
        let encrypted = substitution_encrypt(plaintext, &key_map);
        assert_eq!(encrypted, "ZYX");

        let decrypted = substitution_decrypt(&encrypted, &key_map);
        assert_eq!(decrypted, "ABC");
    }

    #[test]
    fn test_generate_substitution_key() {
        let keyword = "ZEBRA";
        let key_map = generate_substitution_key(keyword);

        // First 5 letters should be ZEBRA (unique letters)
        assert_eq!(key_map.get(&'A'), Some(&'Z'));
        assert_eq!(key_map.get(&'B'), Some(&'E'));
        assert_eq!(key_map.get(&'C'), Some(&'B'));
        assert_eq!(key_map.get(&'D'), Some(&'R'));
        assert_eq!(key_map.get(&'E'), Some(&'A'));
    }

    #[test]
    fn test_hill_cipher_2x2() {
        // Simple 2x2 key matrix
        let key_matrix: Matrix<Integer> = Matrix::from_vec(
            2,
            2,
            vec![
                Integer::from(3), Integer::from(3),
                Integer::from(2), Integer::from(5),
            ]
        )
        .unwrap();

        let plaintext = "HELP";
        let encrypted = hill_encrypt(plaintext, &key_matrix).unwrap();

        // Verify it encrypted to something different
        assert_ne!(encrypted, plaintext);
        assert_eq!(encrypted.len(), 4); // Same length (or padded)
    }

    #[test]
    fn test_hill_cipher_known_roundtrip() {
        // Known Hill cipher example: key = [[3,3],[2,5]], det = 9, which is
        // invertible mod 26 (gcd(9,26)=1), so this key is usable.
        let key_matrix: Matrix<Integer> = Matrix::from_vec(
            2,
            2,
            vec![
                Integer::from(3), Integer::from(3),
                Integer::from(2), Integer::from(5),
            ],
        )
        .unwrap();

        // Independently verified (by hand and with numpy): HELP -> HIAT.
        let plaintext = "HELP";
        let encrypted = hill_encrypt(plaintext, &key_matrix).unwrap();
        assert_eq!(encrypted, "HIAT");

        // Round-trip via the matrix inverse mod 26: key^{-1} mod 26 =
        // [[15,17],[20,9]] (verified key * key^{-1} == I mod 26 by hand).
        // Hill decryption is mathematically encryption with the inverse
        // key, so this exercises the full encrypt/decrypt round trip even
        // though `hill_decrypt` itself is not yet implemented.
        let inverse_key_matrix: Matrix<Integer> = Matrix::from_vec(
            2,
            2,
            vec![
                Integer::from(15), Integer::from(17),
                Integer::from(20), Integer::from(9),
            ],
        )
        .unwrap();
        let decrypted = hill_encrypt(&encrypted, &inverse_key_matrix).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_hill_cipher_large_matrix_entries() {
        // Entries deliberately larger than 26 (and one negative) to exercise
        // the exact Integer arithmetic path instead of relying on values
        // that happen to already fit trivially in i64/string round-trips.
        let key_matrix: Matrix<Integer> = Matrix::from_vec(
            2,
            2,
            vec![
                Integer::from(29), Integer::from(-23),
                Integer::from(54), Integer::from(31),
            ],
        )
        .unwrap();
        // 29 mod 26 = 3, -23 mod 26 = 3, 54 mod 26 = 2, 31 mod 26 = 5, so
        // this key is congruent mod 26 to the [[3,3],[2,5]] key above and
        // must encrypt identically.
        let encrypted = hill_encrypt("HELP", &key_matrix).unwrap();
        assert_eq!(encrypted, "HIAT");
    }

    #[test]
    fn test_caesar_wrap_around() {
        let encrypted = caesar_encrypt("XYZ", 3);
        assert_eq!(encrypted, "ABC");
    }

    #[test]
    fn test_vigenere_empty_key() {
        let result = vigenere_encrypt("HELLO", "");
        assert!(result.is_err());
    }
}
