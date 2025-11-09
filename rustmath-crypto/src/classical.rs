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

    let mut result = String::new();

    // Process each block
    for chunk in padded.as_bytes().chunks(n) {
        // Convert letters to numbers (A=0, B=1, etc.)
        let plain_vec: Vec<i64> = chunk
            .iter()
            .map(|&b| (b - b'A') as i64)
            .collect();

        // Multiply by key matrix (mod 26)
        // We need to convert Integer to i64 for arithmetic
        let mut cipher_vec = vec![0i64; n];
        for i in 0..n {
            let mut sum = 0i64;
            for j in 0..n {
                // For the Hill cipher to work, key matrix entries should be small integers
                // We'll assume the Integer values fit in i64
                let key_elem = key_matrix.get(i, j).unwrap();
                // Since Integer doesn't have to_i64, we'll work with the underlying BigInt
                // For now, let's just use the % operator which Integer does have
                let temp = key_elem.clone() * Integer::from(plain_vec[j]);
                // The result of multiplication might be large, but we take mod 26
                // We'll convert via string as a workaround
                let temp_mod = temp % Integer::from(26);
                // Use signum and abs to get a positive result
                let temp_val = if temp_mod.signum() >= 0 {
                    temp_mod
                } else {
                    temp_mod + Integer::from(26)
                };
                // This is hacky but works for small numbers
                // Convert to string, parse as i64
                sum += temp_val.to_string().parse::<i64>().unwrap_or(0);
            }
            cipher_vec[i] = sum % 26;
        }

        // Convert back to letters
        for val in cipher_vec {
            let letter = ((val % 26) as u8 + b'A') as char;
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
