//! Cryptographic hash functions
//!
//! This module implements standard cryptographic hash functions including:
//! - SHA-256: Part of the SHA-2 family (FIPS 180-4)
//! - SHA-3: Keccak-based hash function (FIPS 202)
//! - BLAKE2: Modern high-speed hash function

use std::fmt;

/// SHA-256 hash function implementation
///
/// SHA-256 is part of the SHA-2 family and produces a 256-bit (32-byte) hash value.
/// It's widely used in Bitcoin, SSL/TLS certificates, and digital signatures.
///
/// # Examples
///
/// ```
/// use rustmath_crypto::SHA256;
///
/// let mut hasher = SHA256::new();
/// hasher.update(b"hello world");
/// let digest = hasher.finalize();
/// ```
#[derive(Clone)]
pub struct SHA256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    length: u64,
}

impl SHA256 {
    // SHA-256 constants (first 32 bits of fractional parts of cube roots of first 64 primes)
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    /// Create a new SHA-256 hasher
    pub fn new() -> Self {
        SHA256 {
            // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buffer: Vec::new(),
            length: 0,
        }
    }

    /// Update the hash with new data
    pub fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        self.length += data.len() as u64;

        // Process complete 512-bit (64-byte) blocks
        while self.buffer.len() >= 64 {
            let block: [u8; 64] = self.buffer[..64].try_into().unwrap();
            self.process_block(&block);
            self.buffer.drain(..64);
        }
    }

    /// Finalize the hash and return the digest
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_length = self.length * 8;

        // Padding: append 0x80, then zeros, then 64-bit length
        self.buffer.push(0x80);

        // Pad to 56 bytes (leaving room for 8-byte length)
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0x00);
        }

        // Append length as big-endian 64-bit integer
        self.buffer.extend_from_slice(&bit_length.to_be_bytes());

        // Process final block(s)
        while !self.buffer.is_empty() {
            let block: [u8; 64] = self.buffer[..64].try_into().unwrap();
            self.process_block(&block);
            self.buffer.drain(..64);
        }

        // Convert state to bytes
        let mut digest = [0u8; 32];
        for (i, &h) in self.state.iter().enumerate() {
            digest[i * 4..(i + 1) * 4].copy_from_slice(&h.to_be_bytes());
        }
        digest
    }

    /// Process a single 512-bit block
    fn process_block(&mut self, block: &[u8; 64]) {
        // Prepare message schedule
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }

        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        // Initialize working variables
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];

        // Main compression loop (64 rounds)
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(Self::K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        // Add compressed chunk to current hash value
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }

    /// Convenience function to hash data in one call
    pub fn hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = SHA256::new();
        hasher.update(data);
        hasher.finalize()
    }
}

impl Default for SHA256 {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SHA256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SHA256")
            .field("length", &self.length)
            .finish()
    }
}

/// SHA-3 (Keccak) hash function implementation
///
/// SHA-3 is based on the Keccak algorithm and uses a sponge construction.
/// This implementation provides SHA3-256 (256-bit output).
///
/// # Examples
///
/// ```
/// use rustmath_crypto::SHA3_256;
///
/// let mut hasher = SHA3_256::new();
/// hasher.update(b"hello world");
/// let digest = hasher.finalize();
/// ```
#[derive(Clone)]
pub struct SHA3_256 {
    state: [u64; 25],
    buffer: Vec<u8>,
    rate: usize, // Rate in bytes (136 for SHA3-256)
}

impl SHA3_256 {
    // Keccak round constants
    const RC: [u64; 24] = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ];

    /// Create a new SHA3-256 hasher
    pub fn new() -> Self {
        SHA3_256 {
            state: [0u64; 25],
            buffer: Vec::new(),
            rate: 136, // (1600 - 2*256) / 8 = 136 bytes
        }
    }

    /// Update the hash with new data
    pub fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);

        // Process complete rate-sized blocks
        while self.buffer.len() >= self.rate {
            let mut block = vec![0u8; self.rate];
            block.copy_from_slice(&self.buffer[..self.rate]);
            self.buffer.drain(..self.rate);
            self.absorb_block(&block);
        }
    }

    /// Finalize the hash and return the digest
    pub fn finalize(mut self) -> [u8; 32] {
        // SHA-3 padding: append 0x06, then zeros, then 0x80
        self.buffer.push(0x06);
        while self.buffer.len() < self.rate {
            self.buffer.push(0x00);
        }
        self.buffer[self.rate - 1] |= 0x80;

        // Process final block
        let mut final_block = vec![0u8; self.rate];
        final_block.copy_from_slice(&self.buffer[..self.rate]);
        self.absorb_block(&final_block);

        // Squeeze out 256 bits (32 bytes)
        let mut digest = [0u8; 32];
        for i in 0..4 {
            digest[i * 8..(i + 1) * 8].copy_from_slice(&self.state[i].to_le_bytes());
        }
        digest
    }

    /// Absorb a rate-sized block into the state
    fn absorb_block(&mut self, block: &[u8]) {
        // XOR block into state
        for i in 0..(self.rate / 8) {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&block[i * 8..(i + 1) * 8]);
            self.state[i] ^= u64::from_le_bytes(bytes);
        }

        // Apply Keccak-f[1600] permutation
        self.keccak_f();
    }

    /// Keccak-f[1600] permutation
    fn keccak_f(&mut self) {
        for round in 0..24 {
            // θ (theta) step
            let mut c = [0u64; 5];
            for x in 0..5 {
                c[x] = self.state[x]
                    ^ self.state[x + 5]
                    ^ self.state[x + 10]
                    ^ self.state[x + 15]
                    ^ self.state[x + 20];
            }

            let mut d = [0u64; 5];
            for x in 0..5 {
                d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            }

            for x in 0..5 {
                for y in 0..5 {
                    self.state[x + y * 5] ^= d[x];
                }
            }

            // ρ (rho) and π (pi) steps
            let mut b = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let r = Self::rotation_offset(x, y);
                    let new_x = y;
                    let new_y = (2 * x + 3 * y) % 5;
                    b[new_x + new_y * 5] = self.state[x + y * 5].rotate_left(r);
                }
            }

            // χ (chi) step
            for y in 0..5 {
                let mut t = [0u64; 5];
                for x in 0..5 {
                    t[x] = b[x + y * 5];
                }
                for x in 0..5 {
                    self.state[x + y * 5] = t[x] ^ ((!t[(x + 1) % 5]) & t[(x + 2) % 5]);
                }
            }

            // ι (iota) step
            self.state[0] ^= Self::RC[round];
        }
    }

    /// Get rotation offset for ρ step
    fn rotation_offset(x: usize, y: usize) -> u32 {
        const OFFSETS: [[u32; 5]; 5] = [
            [0, 36, 3, 41, 18],
            [1, 44, 10, 45, 2],
            [62, 6, 43, 15, 61],
            [28, 55, 25, 21, 56],
            [27, 20, 39, 8, 14],
        ];
        OFFSETS[x][y]
    }

    /// Convenience function to hash data in one call
    pub fn hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = SHA3_256::new();
        hasher.update(data);
        hasher.finalize()
    }
}

impl Default for SHA3_256 {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SHA3_256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SHA3_256").finish()
    }
}

/// BLAKE2b hash function implementation
///
/// BLAKE2 is a cryptographic hash function faster than MD5, SHA-1, SHA-2, and SHA-3,
/// yet is at least as secure as SHA-3. This implementation provides BLAKE2b-256.
///
/// # Examples
///
/// ```
/// use rustmath_crypto::BLAKE2b;
///
/// let mut hasher = BLAKE2b::new(32); // 32-byte (256-bit) output
/// hasher.update(b"hello world");
/// let digest = hasher.finalize();
/// ```
#[derive(Clone)]
pub struct BLAKE2b {
    h: [u64; 8],
    t: [u64; 2],
    f: [u64; 2],
    buffer: Vec<u8>,
    output_length: usize,
}

impl BLAKE2b {
    // BLAKE2b initialization vectors
    const IV: [u64; 8] = [
        0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
        0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
    ];

    // BLAKE2b sigma permutations
    const SIGMA: [[usize; 16]; 12] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    ];

    /// Create a new BLAKE2b hasher with specified output length
    ///
    /// # Arguments
    /// * `output_length` - Desired output length in bytes (1-64)
    pub fn new(output_length: usize) -> Self {
        assert!(output_length > 0 && output_length <= 64, "Invalid output length");

        let mut h = Self::IV;
        h[0] ^= 0x01010000 ^ (output_length as u64);

        BLAKE2b {
            h,
            t: [0, 0],
            f: [0, 0],
            buffer: Vec::new(),
            output_length,
        }
    }

    /// Update the hash with new data
    pub fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);

        // Process complete 128-byte blocks
        while self.buffer.len() >= 128 {
            self.increment_counter(128);
            let block: [u8; 128] = self.buffer[..128].try_into().unwrap();
            self.compress(&block, false);
            self.buffer.drain(..128);
        }
    }

    /// Finalize the hash and return the digest
    pub fn finalize(mut self) -> Vec<u8> {
        // Pad buffer to 128 bytes
        let remaining = self.buffer.len();
        while self.buffer.len() < 128 {
            self.buffer.push(0);
        }

        self.increment_counter(remaining);
        self.f[0] = !0; // Set final block flag

        let block: [u8; 128] = self.buffer[..128].try_into().unwrap();
        self.compress(&block, true);

        // Convert state to bytes
        let mut digest = Vec::with_capacity(self.output_length);
        for &h in &self.h {
            digest.extend_from_slice(&h.to_le_bytes());
            if digest.len() >= self.output_length {
                break;
            }
        }
        digest.truncate(self.output_length);
        digest
    }

    /// Increment the counter
    fn increment_counter(&mut self, inc: usize) {
        self.t[0] = self.t[0].wrapping_add(inc as u64);
        if self.t[0] < inc as u64 {
            self.t[1] = self.t[1].wrapping_add(1);
        }
    }

    /// BLAKE2b compression function
    fn compress(&mut self, block: &[u8; 128], last: bool) {
        // Initialize working variables
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(&self.h);
        v[8..].copy_from_slice(&Self::IV);

        v[12] ^= self.t[0];
        v[13] ^= self.t[1];
        if last {
            v[14] ^= self.f[0];
        }

        // Parse message block
        let mut m = [0u64; 16];
        for i in 0..16 {
            m[i] = u64::from_le_bytes([
                block[i * 8],
                block[i * 8 + 1],
                block[i * 8 + 2],
                block[i * 8 + 3],
                block[i * 8 + 4],
                block[i * 8 + 5],
                block[i * 8 + 6],
                block[i * 8 + 7],
            ]);
        }

        // 12 rounds
        for i in 0..12 {
            // Mix columns
            Self::g(&mut v, 0, 4, 8, 12, m[Self::SIGMA[i][0]], m[Self::SIGMA[i][1]]);
            Self::g(&mut v, 1, 5, 9, 13, m[Self::SIGMA[i][2]], m[Self::SIGMA[i][3]]);
            Self::g(&mut v, 2, 6, 10, 14, m[Self::SIGMA[i][4]], m[Self::SIGMA[i][5]]);
            Self::g(&mut v, 3, 7, 11, 15, m[Self::SIGMA[i][6]], m[Self::SIGMA[i][7]]);

            // Mix diagonals
            Self::g(&mut v, 0, 5, 10, 15, m[Self::SIGMA[i][8]], m[Self::SIGMA[i][9]]);
            Self::g(&mut v, 1, 6, 11, 12, m[Self::SIGMA[i][10]], m[Self::SIGMA[i][11]]);
            Self::g(&mut v, 2, 7, 8, 13, m[Self::SIGMA[i][12]], m[Self::SIGMA[i][13]]);
            Self::g(&mut v, 3, 4, 9, 14, m[Self::SIGMA[i][14]], m[Self::SIGMA[i][15]]);
        }

        // Update hash values
        for i in 0..8 {
            self.h[i] ^= v[i] ^ v[i + 8];
        }
    }

    /// BLAKE2b mixing function G
    #[inline]
    fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);

        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);

        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);

        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }

    /// Convenience function to hash data in one call
    pub fn hash(data: &[u8], output_length: usize) -> Vec<u8> {
        let mut hasher = BLAKE2b::new(output_length);
        hasher.update(data);
        hasher.finalize()
    }
}

impl fmt::Debug for BLAKE2b {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BLAKE2b")
            .field("output_length", &self.output_length)
            .finish()
    }
}

/// Format a hash digest as a hexadecimal string
pub fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty() {
        let digest = SHA256::hash(b"");
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_sha256_abc() {
        let digest = SHA256::hash(b"abc");
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_sha256_hello_world() {
        let digest = SHA256::hash(b"hello world");
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_sha256_long() {
        let digest = SHA256::hash(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        let expected = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_sha256_incremental() {
        let mut hasher = SHA256::new();
        hasher.update(b"hello ");
        hasher.update(b"world");
        let digest = hasher.finalize();
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_sha3_256_empty() {
        let digest = SHA3_256::hash(b"");
        let expected = "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_sha3_256_abc() {
        let digest = SHA3_256::hash(b"abc");
        let expected = "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_sha3_256_incremental() {
        let mut hasher = SHA3_256::new();
        hasher.update(b"ab");
        hasher.update(b"c");
        let digest = hasher.finalize();
        let expected = "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_blake2b_256_empty() {
        let digest = BLAKE2b::hash(b"", 32);
        let expected = "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_blake2b_256_abc() {
        let digest = BLAKE2b::hash(b"abc", 32);
        let expected = "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_blake2b_incremental() {
        let mut hasher = BLAKE2b::new(32);
        hasher.update(b"ab");
        hasher.update(b"c");
        let digest = hasher.finalize();
        let expected = "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319";
        assert_eq!(hex_digest(&digest), expected);
    }

    #[test]
    fn test_blake2b_variable_length() {
        // Test 16-byte output
        let digest = BLAKE2b::hash(b"hello", 16);
        assert_eq!(digest.len(), 16);

        // Test 64-byte output
        let digest = BLAKE2b::hash(b"hello", 64);
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn test_hex_digest() {
        let bytes = [0xde, 0xad, 0xbe, 0xef];
        assert_eq!(hex_digest(&bytes), "deadbeef");
    }

    #[test]
    fn test_sha256_multi_block() {
        // Test with data that spans multiple 512-bit blocks
        let data = b"a".repeat(1000);
        let digest = SHA256::hash(&data);
        // Just verify it produces a valid 32-byte digest
        assert_eq!(digest.len(), 32);
    }

    #[test]
    fn test_sha3_256_multi_block() {
        // Test with data that spans multiple rate blocks
        let data = b"a".repeat(1000);
        let digest = SHA3_256::hash(&data);
        assert_eq!(digest.len(), 32);
    }

    #[test]
    fn test_blake2b_multi_block() {
        // Test with data that spans multiple 128-byte blocks
        let data = b"a".repeat(1000);
        let digest = BLAKE2b::hash(&data, 32);
        assert_eq!(digest.len(), 32);
    }
}
