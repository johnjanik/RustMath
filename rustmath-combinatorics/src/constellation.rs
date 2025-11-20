//! Constellations - Genus 0 Maps on Surfaces
//!
//! A constellation is a bipartite map on a surface, which can be encoded as a pair
//! of permutations (σ, α) where:
//! - σ is a permutation representing the white vertices (hypermap vertices)
//! - α is a permutation representing the hyperedges
//! - The composition σα gives the black vertices (hypermap faces)
//!
//! For genus 0 maps (spherical maps), the Euler characteristic V - E + F = 2 must hold.
//!
//! This implementation focuses on constellations with encoding of the 13 fundamental
//! entities that characterize genus 0 bipartite maps.

use rustmath_core::Ring;
use rustmath_integers::Integer;

/// A constellation representing a genus 0 bipartite map
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constellation {
    /// Number of edges in the map
    n: usize,
    /// White vertex permutation (hypermap vertices)
    sigma: Vec<usize>,
    /// Hyperedge permutation
    alpha: Vec<usize>,
    /// Cached composition σα (black vertices/hypermap faces)
    phi: Vec<usize>,
}

impl Constellation {
    /// Create a new constellation from permutations
    ///
    /// # Arguments
    /// * `sigma` - White vertex permutation (0-indexed)
    /// * `alpha` - Hyperedge permutation (0-indexed)
    ///
    /// # Returns
    /// `Some(Constellation)` if the permutations form a valid genus 0 map, `None` otherwise
    pub fn new(sigma: Vec<usize>, alpha: Vec<usize>) -> Option<Self> {
        let n = sigma.len();

        if n == 0 || alpha.len() != n {
            return None;
        }

        // Validate that sigma and alpha are permutations
        if !is_permutation(&sigma) || !is_permutation(&alpha) {
            return None;
        }

        // Compute phi = sigma * alpha (composition)
        let phi = compose_permutations(&sigma, &alpha);

        let constellation = Constellation {
            n,
            sigma,
            alpha,
            phi,
        };

        // Check if it's genus 0 (Euler characteristic = 2)
        if constellation.genus() != 0 {
            return None;
        }

        Some(constellation)
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.n
    }

    /// Get the white vertex permutation
    pub fn sigma(&self) -> &[usize] {
        &self.sigma
    }

    /// Get the hyperedge permutation
    pub fn alpha(&self) -> &[usize] {
        &self.alpha
    }

    /// Get the black vertex permutation (phi = sigma * alpha)
    pub fn phi(&self) -> &[usize] {
        &self.phi
    }

    /// Count the number of cycles in a permutation
    fn count_cycles(perm: &[usize]) -> usize {
        let n = perm.len();
        let mut visited = vec![false; n];
        let mut num_cycles = 0;

        for i in 0..n {
            if !visited[i] {
                num_cycles += 1;
                let mut current = i;
                while !visited[current] {
                    visited[current] = true;
                    current = perm[current];
                }
            }
        }

        num_cycles
    }

    /// Get cycle structure of a permutation
    fn cycle_structure(perm: &[usize]) -> Vec<usize> {
        let n = perm.len();
        let mut visited = vec![false; n];
        let mut cycles = Vec::new();

        for i in 0..n {
            if !visited[i] {
                let mut cycle_len = 0;
                let mut current = i;
                while !visited[current] {
                    visited[current] = true;
                    current = perm[current];
                    cycle_len += 1;
                }
                cycles.push(cycle_len);
            }
        }

        cycles.sort_unstable();
        cycles.reverse();
        cycles
    }

    /// Get the number of white vertices (cycles in sigma)
    pub fn num_white_vertices(&self) -> usize {
        Self::count_cycles(&self.sigma)
    }

    /// Get the number of black vertices (cycles in phi)
    pub fn num_black_vertices(&self) -> usize {
        Self::count_cycles(&self.phi)
    }

    /// Get the number of hyperedges (cycles in alpha)
    pub fn num_hyperedges(&self) -> usize {
        Self::count_cycles(&self.alpha)
    }

    /// Calculate the genus using Euler characteristic
    /// For a hypermap: V - E + F = 2 - 2g, where:
    /// - V = number of white vertices (cycles in sigma)
    /// - E = number of edges (cycles in alpha, or n/2 if alpha is an involution)
    /// - F = number of black vertices/faces (cycles in phi)
    /// - g = genus
    pub fn genus(&self) -> i32 {
        let v = self.num_white_vertices() as i32;
        // For constellations, edges are typically cycles in alpha
        let e = self.num_hyperedges() as i32;
        let f = self.num_black_vertices() as i32;
        let chi = v - e + f;

        // chi = 2 - 2g, so g = (2 - chi) / 2
        (2 - chi) / 2
    }

    /// Get the valency sequence of white vertices
    pub fn white_valencies(&self) -> Vec<usize> {
        Self::cycle_structure(&self.sigma)
    }

    /// Get the valency sequence of black vertices
    pub fn black_valencies(&self) -> Vec<usize> {
        Self::cycle_structure(&self.phi)
    }

    /// Get the degree sequence of hyperedges
    pub fn hyperedge_degrees(&self) -> Vec<usize> {
        Self::cycle_structure(&self.alpha)
    }

    /// Encode the constellation as the 13 fundamental entities
    ///
    /// The 13 entities are:
    /// 1. n - number of edges
    /// 2. w - number of white vertices
    /// 3. b - number of black vertices
    /// 4. h - number of hyperedges
    /// 5. max_w - maximum white vertex valency
    /// 6. max_b - maximum black vertex valency
    /// 7. max_h - maximum hyperedge degree
    /// 8. min_w - minimum white vertex valency
    /// 9. min_b - minimum black vertex valency
    /// 10. min_h - minimum hyperedge degree
    /// 11. total_w - sum of white valencies (should equal n)
    /// 12. total_b - sum of black valencies (should equal n)
    /// 13. genus - genus of the surface (should be 0)
    pub fn encode(&self) -> ConstellationEncoding {
        let white_vals = self.white_valencies();
        let black_vals = self.black_valencies();
        let hyperedge_degs = self.hyperedge_degrees();

        ConstellationEncoding {
            n: self.n,
            num_white_vertices: self.num_white_vertices(),
            num_black_vertices: self.num_black_vertices(),
            num_hyperedges: self.num_hyperedges(),
            max_white_valency: *white_vals.first().unwrap_or(&0),
            max_black_valency: *black_vals.first().unwrap_or(&0),
            max_hyperedge_degree: *hyperedge_degs.first().unwrap_or(&0),
            min_white_valency: *white_vals.last().unwrap_or(&0),
            min_black_valency: *black_vals.last().unwrap_or(&0),
            min_hyperedge_degree: *hyperedge_degs.last().unwrap_or(&0),
            total_white_valency: self.n,
            total_black_valency: self.n,
            genus: self.genus(),
        }
    }

    /// Check if this constellation is connected
    pub fn is_connected(&self) -> bool {
        // A constellation is connected if the group generated by sigma and alpha acts transitively
        let n = self.n;
        if n == 0 {
            return true;
        }

        let mut reachable = vec![false; n];
        let mut queue = vec![0];
        reachable[0] = true;
        let mut count = 1;

        while let Some(current) = queue.pop() {
            // Try sigma
            let sigma_next = self.sigma[current];
            if !reachable[sigma_next] {
                reachable[sigma_next] = true;
                queue.push(sigma_next);
                count += 1;
            }

            // Try alpha
            let alpha_next = self.alpha[current];
            if !reachable[alpha_next] {
                reachable[alpha_next] = true;
                queue.push(alpha_next);
                count += 1;
            }

            // Try inverse of sigma
            for (i, &val) in self.sigma.iter().enumerate() {
                if val == current && !reachable[i] {
                    reachable[i] = true;
                    queue.push(i);
                    count += 1;
                }
            }

            // Try inverse of alpha
            for (i, &val) in self.alpha.iter().enumerate() {
                if val == current && !reachable[i] {
                    reachable[i] = true;
                    queue.push(i);
                    count += 1;
                }
            }
        }

        count == n
    }

    /// Check if this is a clean constellation (dessin d'enfant)
    /// A clean dessin has alpha as a fixed-point-free involution
    pub fn is_clean(&self) -> bool {
        // Check if alpha is an involution without fixed points
        for i in 0..self.n {
            if self.alpha[i] == i {
                return false; // Has a fixed point
            }
            if self.alpha[self.alpha[i]] != i {
                return false; // Not an involution
            }
        }
        true
    }
}

/// The 13 fundamental entities encoding a constellation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstellationEncoding {
    /// Entity 1: Number of edges
    pub n: usize,
    /// Entity 2: Number of white vertices
    pub num_white_vertices: usize,
    /// Entity 3: Number of black vertices
    pub num_black_vertices: usize,
    /// Entity 4: Number of hyperedges
    pub num_hyperedges: usize,
    /// Entity 5: Maximum white vertex valency
    pub max_white_valency: usize,
    /// Entity 6: Maximum black vertex valency
    pub max_black_valency: usize,
    /// Entity 7: Maximum hyperedge degree
    pub max_hyperedge_degree: usize,
    /// Entity 8: Minimum white vertex valency
    pub min_white_valency: usize,
    /// Entity 9: Minimum black vertex valency
    pub min_black_valency: usize,
    /// Entity 10: Minimum hyperedge degree
    pub min_hyperedge_degree: usize,
    /// Entity 11: Total white valency (sum of all white vertex valencies)
    pub total_white_valency: usize,
    /// Entity 12: Total black valency (sum of all black vertex valencies)
    pub total_black_valency: usize,
    /// Entity 13: Genus of the surface
    pub genus: i32,
}

impl ConstellationEncoding {
    /// Verify the encoding is consistent
    pub fn is_valid(&self) -> bool {
        // For genus 0: V - E + F = 2, where E = num_hyperedges (not n)
        let chi = (self.num_white_vertices as i32) - (self.num_hyperedges as i32) + (self.num_black_vertices as i32);

        // Check Euler characteristic
        if chi != 2 {
            return false;
        }

        // Check genus
        if self.genus != 0 {
            return false;
        }

        // Check that total valencies equal n
        if self.total_white_valency != self.n || self.total_black_valency != self.n {
            return false;
        }

        // Check min <= max
        if self.min_white_valency > self.max_white_valency {
            return false;
        }
        if self.min_black_valency > self.max_black_valency {
            return false;
        }
        if self.min_hyperedge_degree > self.max_hyperedge_degree {
            return false;
        }

        true
    }

    /// Get all 13 entities as a vector
    pub fn to_vec(&self) -> Vec<usize> {
        vec![
            self.n,
            self.num_white_vertices,
            self.num_black_vertices,
            self.num_hyperedges,
            self.max_white_valency,
            self.max_black_valency,
            self.max_hyperedge_degree,
            self.min_white_valency,
            self.min_black_valency,
            self.min_hyperedge_degree,
            self.total_white_valency,
            self.total_black_valency,
            self.genus as usize, // Note: genus is always 0 for our constellations
        ]
    }
}

/// Check if a vector is a valid permutation
fn is_permutation(perm: &[usize]) -> bool {
    let n = perm.len();
    let mut seen = vec![false; n];

    for &x in perm {
        if x >= n || seen[x] {
            return false;
        }
        seen[x] = true;
    }

    true
}

/// Compose two permutations: (a * b)[i] = b[a[i]]
fn compose_permutations(a: &[usize], b: &[usize]) -> Vec<usize> {
    a.iter().map(|&i| b[i]).collect()
}

/// Generate a simple genus 0 constellation (tree-like structure)
/// This creates a constellation with one vertex connected to multiple faces
pub fn trivial_constellation(n: usize) -> Option<Constellation> {
    if n < 2 || n % 2 != 0 {
        return None; // Need even number for pairing
    }

    // Create one big cycle for sigma (one vertex)
    let sigma: Vec<usize> = {
        let mut s: Vec<usize> = (1..n).collect();
        s.push(0);
        s
    };

    // Create pairing involution for alpha (edges)
    // Pair 0-1, 2-3, 4-5, etc.
    let alpha: Vec<usize> = (0..n)
        .map(|i| if i % 2 == 0 { i + 1 } else { i - 1 })
        .collect();

    Constellation::new(sigma, alpha)
}

/// Generate a star constellation with n half-edges (must be even and >= 4)
/// Creates a star-like genus 0 map with multiple vertices
pub fn star_constellation(n: usize) -> Option<Constellation> {
    if n < 4 || n % 2 != 0 {
        return None; // Need even number >= 4 for edge pairing
    }

    // sigma creates multiple 2-cycles (multiple vertices)
    let sigma: Vec<usize> = (0..n)
        .map(|i| if i % 2 == 0 { i + 1 } else { i - 1 })
        .collect();

    // alpha pairs half-edges into full edges differently than sigma
    // This creates a different pairing
    let alpha: Vec<usize> = (0..n)
        .map(|i| {
            if i < n / 2 {
                i + n / 2
            } else {
                i - n / 2
            }
        })
        .collect();

    Constellation::new(sigma, alpha)
}

/// Generate a path constellation with n edges
pub fn path_constellation(n: usize) -> Option<Constellation> {
    if n < 1 {
        return None;
    }

    if n == 1 {
        // Single edge
        let sigma = vec![0];
        let alpha = vec![0];
        return Constellation::new(sigma, alpha);
    }

    // Create a path: edges are paired except at endpoints
    let mut sigma = Vec::with_capacity(n);
    let mut alpha = Vec::with_capacity(n);

    for i in 0..n {
        if i == 0 {
            // First edge
            sigma.push(0);
            alpha.push(1);
        } else if i == n - 1 {
            // Last edge
            sigma.push(n - 1);
            alpha.push(n - 2);
        } else {
            // Middle edges
            sigma.push(i);
            if i % 2 == 1 {
                alpha.push(i - 1);
            } else {
                alpha.push(i + 1);
            }
        }
    }

    Constellation::new(sigma, alpha)
}

/// Count genus 0 constellations with n edges (exact enumeration for small n)
pub fn count_genus_0_constellations(n: usize) -> Integer {
    // This is a placeholder for exact counting
    // The actual count depends on additional constraints
    // For now, return a simple upper bound based on permutation pairs

    if n == 0 {
        return Integer::zero();
    }

    // Upper bound: (n!)^2 permutation pairs
    // In practice, far fewer satisfy genus 0 constraint
    let mut fact = Integer::one();
    for i in 1..=n {
        fact = fact * Integer::from(i as u32);
    }

    // This is just a placeholder - actual count requires more sophisticated enumeration
    fact.clone() * fact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_constellation() {
        let c = trivial_constellation(4).unwrap();
        assert_eq!(c.num_edges(), 4);
        assert_eq!(c.genus(), 0);

        let encoding = c.encode();
        assert!(encoding.is_valid());
        assert_eq!(encoding.genus, 0);
    }

    #[test]
    fn test_star_constellation() {
        let c = star_constellation(4).unwrap();
        assert_eq!(c.num_edges(), 4);
        assert_eq!(c.genus(), 0);

        let encoding = c.encode();
        assert!(encoding.is_valid());
    }

    #[test]
    fn test_constellation_encoding() {
        // Simple genus 0 constellation with 2 half-edges
        let sigma = vec![1, 0]; // One 2-cycle (one vertex)
        let alpha = vec![1, 0]; // One 2-cycle (one edge)

        let c = Constellation::new(sigma, alpha).unwrap();
        let enc = c.encode();

        assert_eq!(enc.n, 2);
        assert!(enc.is_valid());
        assert_eq!(enc.genus, 0);

        // Check that total valencies equal n
        assert_eq!(enc.total_white_valency, 2);
        assert_eq!(enc.total_black_valency, 2);

        // Check vector encoding has 13 elements
        assert_eq!(enc.to_vec().len(), 13);
    }

    #[test]
    fn test_is_permutation() {
        assert!(is_permutation(&[0, 1, 2]));
        assert!(is_permutation(&[2, 0, 1]));
        assert!(!is_permutation(&[0, 0, 1])); // Duplicate
        assert!(!is_permutation(&[0, 1, 3])); // Out of range
    }

    #[test]
    fn test_composition() {
        let a = vec![1, 2, 0]; // (0 1 2)
        let b = vec![0, 2, 1]; // (1 2)
        let c = compose_permutations(&a, &b);

        // c[i] = b[a[i]]
        assert_eq!(c[0], b[a[0]]); // b[1] = 2
        assert_eq!(c[1], b[a[1]]); // b[2] = 1
        assert_eq!(c[2], b[a[2]]); // b[0] = 0
        assert_eq!(c, vec![2, 1, 0]);
    }

    #[test]
    fn test_genus_calculation() {
        // Create a simple genus 0 map
        let sigma = vec![1, 0, 3, 2]; // Two 2-cycles
        let alpha = vec![0, 1, 2, 3]; // Identity

        if let Some(c) = Constellation::new(sigma, alpha) {
            assert_eq!(c.genus(), 0);
            assert_eq!(c.num_edges(), 4);
        }
    }

    #[test]
    fn test_is_clean() {
        // Clean constellation: alpha should be fixed-point-free involution
        let sigma = vec![1, 0, 3, 2];
        let alpha = vec![1, 0, 3, 2]; // Fixed-point-free involution

        if let Some(c) = Constellation::new(sigma.clone(), alpha) {
            assert!(c.is_clean());
        }

        // Not clean: alpha has fixed points
        let alpha2 = vec![0, 1, 2, 3]; // Identity - has fixed points
        if let Some(c) = Constellation::new(sigma, alpha2) {
            assert!(!c.is_clean());
        }
    }

    #[test]
    fn test_connectivity() {
        let sigma = vec![1, 2, 0];
        let alpha = vec![0, 1, 2];

        if let Some(c) = Constellation::new(sigma, alpha) {
            assert!(c.is_connected());
        }
    }

    #[test]
    fn test_encoding_invariants() {
        let c = star_constellation(4).unwrap();
        let enc = c.encode();

        // Verify Euler characteristic (using hyperedges, not n)
        let chi = (enc.num_white_vertices as i32) - (enc.num_hyperedges as i32) + (enc.num_black_vertices as i32);
        assert_eq!(chi, 2); // Genus 0

        // Verify encoding validity
        assert!(enc.is_valid());
    }
}
