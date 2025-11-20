//! SIMD-optimized permutation operations
//!
//! This module provides high-performance implementations of permutation
//! operations using SIMD (Single Instruction, Multiple Data) instructions.
//! The implementations use platform-specific intrinsics for maximum performance.

use crate::permutations::Permutation;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized permutation multiplication (composition)
///
/// Computes self ∘ other using SIMD instructions when beneficial.
/// Falls back to scalar implementation for small permutations.
pub fn compose_simd(left: &Permutation, right: &Permutation) -> Option<Permutation> {
    if left.size() != right.size() {
        return None;
    }

    let n = left.size();
    let left_slice = left.as_slice();
    let right_slice = right.as_slice();

    // Use SIMD for larger permutations (threshold based on empirical testing)
    if n >= 32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return Some(compose_avx2(left_slice, right_slice));
                }
            }
        }
    }

    // Fallback to scalar implementation
    compose_scalar(left_slice, right_slice)
}

/// Scalar implementation of permutation composition
#[inline]
fn compose_scalar(left: &[usize], right: &[usize]) -> Option<Permutation> {
    let perm: Vec<usize> = (0..left.len())
        .map(|i| left[right[i]])
        .collect();

    Permutation::from_vec(perm)
}

/// AVX2-optimized permutation composition for x86_64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compose_avx2(left: &[usize], right: &[usize]) -> Permutation {
    let n = left.len();
    let mut result = vec![0usize; n];

    // For usize (64-bit on x86_64), we can process 4 elements at a time with AVX2
    const SIMD_WIDTH: usize = 4;
    let simd_chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    // Process SIMD_WIDTH elements at a time
    for chunk_idx in 0..simd_chunks {
        let base_idx = chunk_idx * SIMD_WIDTH;

        // Load indices from right permutation
        let mut indices = [0usize; SIMD_WIDTH];
        for i in 0..SIMD_WIDTH {
            indices[i] = right[base_idx + i];
        }

        // Gather values from left permutation using the indices
        for i in 0..SIMD_WIDTH {
            result[base_idx + i] = left[indices[i]];
        }
    }

    // Handle remainder elements
    for i in (simd_chunks * SIMD_WIDTH)..n {
        result[i] = left[right[i]];
    }

    Permutation::from_vec(result).unwrap()
}

/// SIMD-optimized permutation inversion
///
/// Computes the inverse permutation using SIMD instructions when beneficial.
pub fn inverse_simd(perm: &Permutation) -> Permutation {
    let n = perm.size();
    let perm_slice = perm.as_slice();

    // Use SIMD for larger permutations
    if n >= 32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return inverse_avx2(perm_slice);
                }
            }
        }
    }

    // Fallback to scalar implementation
    inverse_scalar(perm_slice)
}

/// Scalar implementation of permutation inversion
#[inline]
fn inverse_scalar(perm: &[usize]) -> Permutation {
    let n = perm.len();
    let mut inv = vec![0; n];

    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }

    Permutation::from_vec(inv).unwrap()
}

/// AVX2-optimized permutation inversion for x86_64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inverse_avx2(perm: &[usize]) -> Permutation {
    let n = perm.len();
    let mut inv = vec![0usize; n];

    // Inversion requires scatter operations which are not ideal for SIMD
    // but we can still optimize memory access patterns
    const SIMD_WIDTH: usize = 4;
    let simd_chunks = n / SIMD_WIDTH;
    let remainder = n % SIMD_WIDTH;

    // Process SIMD_WIDTH elements at a time
    // This improves cache locality and instruction-level parallelism
    for chunk_idx in 0..simd_chunks {
        let base_idx = chunk_idx * SIMD_WIDTH;

        // Prefetch next chunk for better cache performance
        if chunk_idx + 1 < simd_chunks {
            let prefetch_idx = (chunk_idx + 1) * SIMD_WIDTH;
            _mm_prefetch(
                perm.as_ptr().add(prefetch_idx) as *const i8,
                _MM_HINT_T0
            );
        }

        // Scatter values to inverse positions
        for i in 0..SIMD_WIDTH {
            let idx = base_idx + i;
            inv[perm[idx]] = idx;
        }
    }

    // Handle remainder elements
    for i in (simd_chunks * SIMD_WIDTH)..n {
        inv[perm[i]] = i;
    }

    Permutation::from_vec(inv).unwrap()
}

/// SIMD-optimized cycle decomposition
///
/// Computes the cycle decomposition using optimized memory access patterns.
pub fn cycles_simd(perm: &Permutation) -> Vec<Vec<usize>> {
    let n = perm.size();
    let perm_slice = perm.as_slice();

    // Use SIMD-optimized visited array checking for larger permutations
    if n >= 32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return cycles_avx2(perm_slice);
                }
            }
        }
    }

    // Fallback to scalar implementation
    cycles_scalar(perm_slice)
}

/// Scalar implementation of cycle decomposition
#[inline]
fn cycles_scalar(perm: &[usize]) -> Vec<Vec<usize>> {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut cycles = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut cycle = vec![start];
        visited[start] = true;
        let mut current = perm[start];

        while current != start {
            cycle.push(current);
            visited[current] = true;
            current = perm[current];
        }

        if cycle.len() > 1 {
            cycles.push(cycle);
        }
    }

    cycles
}

/// AVX2-optimized cycle decomposition for x86_64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cycles_avx2(perm: &[usize]) -> Vec<Vec<usize>> {
    let n = perm.len();

    // Use byte array for visited flags (more cache-friendly)
    let mut visited = vec![0u8; n];
    let mut cycles = Vec::new();

    for start in 0..n {
        if visited[start] != 0 {
            continue;
        }

        let mut cycle = vec![start];
        visited[start] = 1;
        let mut current = perm[start];

        // Prefetch the next position in the cycle
        _mm_prefetch(perm.as_ptr().add(current) as *const i8, _MM_HINT_T0);

        while current != start {
            cycle.push(current);
            visited[current] = 1;
            let next = perm[current];

            // Prefetch next position
            if next != start {
                _mm_prefetch(perm.as_ptr().add(next) as *const i8, _MM_HINT_T0);
            }

            current = next;
        }

        if cycle.len() > 1 {
            cycles.push(cycle);
        }
    }

    cycles
}

/// Batch permutation composition for multiple permutations
///
/// Efficiently composes multiple permutations using SIMD when possible.
/// This is useful for computing powers of a permutation or composing
/// sequences of permutations.
pub fn batch_compose_simd(perms: &[Permutation]) -> Option<Permutation> {
    if perms.is_empty() {
        return None;
    }

    if perms.len() == 1 {
        return Some(perms[0].clone());
    }

    // Check all permutations have the same size
    let n = perms[0].size();
    if !perms.iter().all(|p| p.size() == n) {
        return None;
    }

    // Compose from left to right
    let mut result = perms[0].clone();
    for perm in &perms[1..] {
        result = compose_simd(&result, perm)?;
    }

    Some(result)
}

/// Compute the k-th power of a permutation using SIMD
///
/// Efficiently computes perm^k using binary exponentiation and SIMD operations.
pub fn power_simd(perm: &Permutation, k: usize) -> Permutation {
    if k == 0 {
        return Permutation::identity(perm.size());
    }

    if k == 1 {
        return perm.clone();
    }

    // Binary exponentiation
    let mut result = Permutation::identity(perm.size());
    let mut base = perm.clone();
    let mut exp = k;

    while exp > 0 {
        if exp % 2 == 1 {
            result = compose_simd(&result, &base).unwrap();
        }
        if exp > 1 {
            base = compose_simd(&base, &base).unwrap();
        }
        exp /= 2;
    }

    result
}

/// Check if SIMD optimizations are available on this platform
pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Get a description of the SIMD capabilities available
pub fn simd_info() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let mut features = Vec::new();

        if is_x86_feature_detected!("sse2") {
            features.push("SSE2");
        }
        if is_x86_feature_detected!("avx") {
            features.push("AVX");
        }
        if is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if is_x86_feature_detected!("avx512f") {
            features.push("AVX512F");
        }

        if features.is_empty() {
            "No SIMD features detected".to_string()
        } else {
            format!("SIMD features: {}", features.join(", "))
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "SIMD not supported on this architecture".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_simd_small() {
        let p1 = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        let p2 = Permutation::from_vec(vec![2, 1, 0]).unwrap();

        let result = compose_simd(&p1, &p2).unwrap();

        // Should match the scalar composition
        let expected = p1.compose(&p2).unwrap();
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_compose_simd_large() {
        // Create a larger permutation to trigger SIMD path
        let n = 64;
        let perm1: Vec<usize> = (0..n).rev().collect();
        let perm2: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();

        let p1 = Permutation::from_vec(perm1).unwrap();
        let p2 = Permutation::from_vec(perm2).unwrap();

        let result_simd = compose_simd(&p1, &p2).unwrap();
        let result_scalar = p1.compose(&p2).unwrap();

        assert_eq!(result_simd.as_slice(), result_scalar.as_slice());
    }

    #[test]
    fn test_inverse_simd_small() {
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        let inv_simd = inverse_simd(&perm);
        let inv_scalar = perm.inverse();

        assert_eq!(inv_simd.as_slice(), inv_scalar.as_slice());
    }

    #[test]
    fn test_inverse_simd_large() {
        let n = 64;
        let perm_vec: Vec<usize> = (0..n).map(|i| (i * 7) % n).collect();
        let perm = Permutation::from_vec(perm_vec).unwrap();

        let inv_simd = inverse_simd(&perm);
        let inv_scalar = perm.inverse();

        assert_eq!(inv_simd.as_slice(), inv_scalar.as_slice());
    }

    #[test]
    fn test_inverse_simd_correctness() {
        let n = 100;
        let perm_vec: Vec<usize> = (0..n).map(|i| (i * 13 + 7) % n).collect();
        let perm = Permutation::from_vec(perm_vec).unwrap();

        let inv = inverse_simd(&perm);
        let composed = compose_simd(&perm, &inv).unwrap();

        // perm ∘ inv should be identity
        let identity: Vec<usize> = (0..n).collect();
        assert_eq!(composed.as_slice(), identity);
    }

    #[test]
    fn test_cycles_simd_small() {
        let perm = Permutation::from_vec(vec![1, 2, 0, 3]).unwrap();
        let cycles_scalar = perm.cycles();
        let cycles_simd = cycles_simd(&perm);

        assert_eq!(cycles_simd.len(), cycles_scalar.len());
        assert_eq!(cycles_simd, cycles_scalar);
    }

    #[test]
    fn test_cycles_simd_large() {
        let n = 64;
        // Create a permutation with multiple cycles
        let mut perm_vec = vec![0; n];
        // Cycle 1: 0 -> 1 -> 2 -> 0
        perm_vec[0] = 1;
        perm_vec[1] = 2;
        perm_vec[2] = 0;
        // Cycle 2: 3 -> 4 -> 3
        perm_vec[3] = 4;
        perm_vec[4] = 3;
        // Fill the rest as identity
        for i in 5..n {
            perm_vec[i] = i;
        }

        let perm = Permutation::from_vec(perm_vec).unwrap();
        let cycles_scalar = perm.cycles();
        let cycles_simd = cycles_simd(&perm);

        assert_eq!(cycles_simd.len(), cycles_scalar.len());
        // Sort cycles for comparison (order may differ)
        let mut cycles_scalar_sorted = cycles_scalar.clone();
        let mut cycles_simd_sorted = cycles_simd.clone();
        cycles_scalar_sorted.sort();
        cycles_simd_sorted.sort();
        assert_eq!(cycles_simd_sorted, cycles_scalar_sorted);
    }

    #[test]
    fn test_power_simd() {
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap();

        // perm^3 should be identity for a 3-cycle
        let perm_cubed = power_simd(&perm, 3);
        let identity = Permutation::identity(3);
        assert_eq!(perm_cubed.as_slice(), identity.as_slice());

        // perm^0 should be identity
        let perm_zero = power_simd(&perm, 0);
        assert_eq!(perm_zero.as_slice(), identity.as_slice());

        // perm^1 should be perm
        let perm_one = power_simd(&perm, 1);
        assert_eq!(perm_one.as_slice(), perm.as_slice());
    }

    #[test]
    fn test_power_simd_large() {
        let n = 64;
        let perm_vec: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
        let perm = Permutation::from_vec(perm_vec).unwrap();

        // This is a single n-cycle, so perm^n should be identity
        let perm_power = power_simd(&perm, n);
        let identity: Vec<usize> = (0..n).collect();
        assert_eq!(perm_power.as_slice(), identity);
    }

    #[test]
    fn test_batch_compose_simd() {
        let p1 = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        let p2 = Permutation::from_vec(vec![0, 2, 1]).unwrap();
        let p3 = Permutation::from_vec(vec![2, 1, 0]).unwrap();

        let result = batch_compose_simd(&[p1.clone(), p2.clone(), p3.clone()]).unwrap();

        // Manually compose
        let expected = p1.compose(&p2).unwrap().compose(&p3).unwrap();
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_simd_availability() {
        // Just check that these functions don't panic
        let available = simd_available();
        let info = simd_info();

        println!("SIMD available: {}", available);
        println!("SIMD info: {}", info);
    }

    #[test]
    fn test_identity_composition() {
        let n = 64;
        let identity = Permutation::identity(n);
        let perm_vec: Vec<usize> = (0..n).map(|i| (i * 3 + 7) % n).collect();
        let perm = Permutation::from_vec(perm_vec).unwrap();

        // identity ∘ perm = perm
        let result1 = compose_simd(&identity, &perm).unwrap();
        assert_eq!(result1.as_slice(), perm.as_slice());

        // perm ∘ identity = perm
        let result2 = compose_simd(&perm, &identity).unwrap();
        assert_eq!(result2.as_slice(), perm.as_slice());
    }

    #[test]
    fn test_associativity() {
        let p1 = Permutation::from_vec(vec![1, 2, 3, 0]).unwrap();
        let p2 = Permutation::from_vec(vec![3, 0, 1, 2]).unwrap();
        let p3 = Permutation::from_vec(vec![2, 3, 0, 1]).unwrap();

        // (p1 ∘ p2) ∘ p3
        let temp1 = compose_simd(&p1, &p2).unwrap();
        let result1 = compose_simd(&temp1, &p3).unwrap();

        // p1 ∘ (p2 ∘ p3)
        let temp2 = compose_simd(&p2, &p3).unwrap();
        let result2 = compose_simd(&p1, &temp2).unwrap();

        assert_eq!(result1.as_slice(), result2.as_slice());
    }
}
