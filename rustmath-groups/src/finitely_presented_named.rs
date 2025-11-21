//! Named Finitely Presented Groups
//!
//! This module provides constructors for various well-known groups as finitely presented groups.
//! These include classical groups like cyclic, dihedral, quaternion, and more exotic presentations.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::finitely_presented_named::{cyclic_presentation, dihedral_presentation};
//!
//! // Create the cyclic group of order 5
//! let c5 = cyclic_presentation(5);
//!
//! // Create the dihedral group D_4 (symmetries of a square)
//! let d4 = dihedral_presentation(4);
//! ```

use crate::finitely_presented::FinitelyPresentedGroup;
use crate::free_group::FreeGroupElement;

/// Create a presentation of the cyclic group of order n
///
/// The cyclic group C_n has one generator `a` with the relation `a^n = 1`.
///
/// # Arguments
///
/// * `n` - The order of the cyclic group (must be positive)
///
/// # Examples
///
/// ```
/// use rustmath_groups::finitely_presented_named::cyclic_presentation;
///
/// // Create C_5
/// let c5 = cyclic_presentation(5);
/// ```
pub fn cyclic_presentation(n: usize) -> FinitelyPresentedGroup {
    assert!(n > 0, "Order must be positive");

    let generator_names = vec!["a".to_string()];
    let relations = vec![
        FreeGroupElement::generator(0, n as isize)
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect()
    ];

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of a finitely generated abelian group
///
/// Given a list of invariant factors [n_1, n_2, ..., n_k], creates the direct product
/// Z_{n_1} × Z_{n_2} × ... × Z_{n_k} where 0 means infinite cyclic.
///
/// # Arguments
///
/// * `invariants` - List of invariant factors (0 for infinite cyclic, positive for finite)
///
/// # Examples
///
/// ```
/// use rustmath_groups::finitely_presented_named::finitely_generated_abelian_presentation;
///
/// // Create Z_2 × Z_3
/// let g = finitely_generated_abelian_presentation(&[2, 3]);
///
/// // Create Z × Z_4
/// let g2 = finitely_generated_abelian_presentation(&[0, 4]);
/// ```
pub fn finitely_generated_abelian_presentation(
    invariants: &[usize],
) -> FinitelyPresentedGroup {
    assert!(!invariants.is_empty(), "Need at least one generator");

    let n_gens = invariants.len();
    let generator_names: Vec<String> = (0..n_gens)
        .map(|i| format!("a_{}", i))
        .collect();
    let mut relations = Vec::new();

    // Add power relations for each generator
    for (i, &inv) in invariants.iter().enumerate() {
        if inv > 0 {
            // a_i^{inv} = 1
            let rel = FreeGroupElement::generator(i as isize, inv as isize)
                .tietze()
                .into_iter()
                .map(|x| x as i32)
                .collect();
            relations.push(rel);
        }
    }

    // Add commutator relations: [a_i, a_j] = 1 for all i != j
    for i in 0..n_gens {
        for j in (i + 1)..n_gens {
            // a_i a_j a_i^{-1} a_j^{-1} = 1
            let ai = FreeGroupElement::generator(i as isize, 1);
            let aj = FreeGroupElement::generator(j as isize, 1);
            let ai_inv = FreeGroupElement::generator(i as isize, -1);
            let aj_inv = FreeGroupElement::generator(j as isize, -1);

            let comm = ai.multiply(&aj).multiply(&ai_inv).multiply(&aj_inv);
            let rel = comm
                .tietze()
                .into_iter()
                .map(|x| x as i32)
                .collect();
            relations.push(rel);
        }
    }

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the Heisenberg group
///
/// The Heisenberg group H_n has generators x_1, ..., x_n, y_1, ..., y_n, z
/// with relations:
/// - [x_i, y_i] = z for all i
/// - [z, x_i] = 1, [z, y_i] = 1 for all i
/// - [x_i, y_j] = 1 for i ≠ j
/// - [x_i, x_j] = 1, [y_i, y_j] = 1 for all i, j
///
/// If p > 0, also adds z^p = 1 (for finite field version).
///
/// # Arguments
///
/// * `n` - The degree of the Heisenberg group (default 1)
/// * `p` - Prime for finite field version, or 0 for integers (default 0)
pub fn finitely_generated_heisenberg_presentation(n: usize, p: usize) -> FinitelyPresentedGroup {
    assert!(n >= 1, "Degree must be at least 1");
    if p > 0 {
        // Should verify p is prime, but we'll trust the caller
    }

    // Generators: x_0, ..., x_{n-1}, y_0, ..., y_{n-1}, z
    let z_idx = 2 * n; // z is the last generator

    let generator_names: Vec<String> = (0..n)
        .map(|i| format!("x_{}", i))
        .chain((0..n).map(|i| format!("y_{}", i)))
        .chain(std::iter::once("z".to_string()))
        .collect();

    let mut relations = Vec::new();

    // [x_i, y_i] = z for all i
    for i in 0..n {
        let x_i = FreeGroupElement::generator(i as isize, 1);
        let y_i = FreeGroupElement::generator((n + i) as isize, 1);
        let x_i_inv = FreeGroupElement::generator(i as isize, -1);
        let y_i_inv = FreeGroupElement::generator((n + i) as isize, -1);
        let z_inv = FreeGroupElement::generator(z_idx as isize, -1);

        // x_i y_i x_i^{-1} y_i^{-1} z^{-1} = 1
        let comm = x_i
            .multiply(&y_i)
            .multiply(&x_i_inv)
            .multiply(&y_i_inv)
            .multiply(&z_inv);
        let rel = comm
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    // [z, x_i] = 1 for all i
    for i in 0..n {
        let z = FreeGroupElement::generator(z_idx as isize, 1);
        let x_i = FreeGroupElement::generator(i as isize, 1);
        let z_inv = FreeGroupElement::generator(z_idx as isize, -1);
        let x_i_inv = FreeGroupElement::generator(i as isize, -1);

        // z x_i z^{-1} x_i^{-1} = 1
        let comm = z.multiply(&x_i).multiply(&z_inv).multiply(&x_i_inv);
        let rel = comm
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    // [z, y_i] = 1 for all i
    for i in 0..n {
        let z = FreeGroupElement::generator(z_idx as isize, 1);
        let y_i = FreeGroupElement::generator((n + i) as isize, 1);
        let z_inv = FreeGroupElement::generator(z_idx as isize, -1);
        let y_i_inv = FreeGroupElement::generator((n + i) as isize, -1);

        // z y_i z^{-1} y_i^{-1} = 1
        let comm = z.multiply(&y_i).multiply(&z_inv).multiply(&y_i_inv);
        let rel = comm
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    // [x_i, y_j] = 1 for i ≠ j
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let x_i = FreeGroupElement::generator(i as isize, 1);
                let y_j = FreeGroupElement::generator((n + j) as isize, 1);
                let x_i_inv = FreeGroupElement::generator(i as isize, -1);
                let y_j_inv = FreeGroupElement::generator((n + j) as isize, -1);

                let comm = x_i.multiply(&y_j).multiply(&x_i_inv).multiply(&y_j_inv);
                let rel = comm
                    .tietze()
                    .into_iter()
                    .map(|x| x as i32)
                    .collect();
                relations.push(rel);
            }
        }
    }

    // [x_i, x_j] = 1 for all i, j
    for i in 0..n {
        for j in (i + 1)..n {
            let x_i = FreeGroupElement::generator(i as isize, 1);
            let x_j = FreeGroupElement::generator(j as isize, 1);
            let x_i_inv = FreeGroupElement::generator(i as isize, -1);
            let x_j_inv = FreeGroupElement::generator(j as isize, -1);

            let comm = x_i.multiply(&x_j).multiply(&x_i_inv).multiply(&x_j_inv);
            let rel = comm
                .tietze()
                .into_iter()
                .map(|x| x as i32)
                .collect();
            relations.push(rel);
        }
    }

    // [y_i, y_j] = 1 for all i, j
    for i in 0..n {
        for j in (i + 1)..n {
            let y_i = FreeGroupElement::generator((n + i) as isize, 1);
            let y_j = FreeGroupElement::generator((n + j) as isize, 1);
            let y_i_inv = FreeGroupElement::generator((n + i) as isize, -1);
            let y_j_inv = FreeGroupElement::generator((n + j) as isize, -1);

            let comm = y_i.multiply(&y_j).multiply(&y_i_inv).multiply(&y_j_inv);
            let rel = comm
                .tietze()
                .into_iter()
                .map(|x| x as i32)
                .collect();
            relations.push(rel);
        }
    }

    // If p > 0, add z^p = 1
    if p > 0 {
        let rel = FreeGroupElement::generator(z_idx as isize, p as isize)
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the dihedral group D_n
///
/// The dihedral group D_n of order 2n has two generators `a` and `b` with relations:
/// - a^n = 1 (rotation)
/// - b^2 = 1 (reflection)
/// - (ab)^2 = 1 (braid relation)
///
/// # Arguments
///
/// * `n` - The order of rotation (group has order 2n)
pub fn dihedral_presentation(n: usize) -> FinitelyPresentedGroup {
    assert!(n >= 2, "Dihedral group requires n >= 2");

    let generator_names = vec!["a".to_string(), "b".to_string()];
    let mut relations = Vec::new();

    // a^n = 1
    let rel = FreeGroupElement::generator(0, n as isize)
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // b^2 = 1
    let rel = FreeGroupElement::generator(1, 2)
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // (ab)^2 = 1, i.e., abab = 1
    let a = FreeGroupElement::generator(0, 1);
    let b = FreeGroupElement::generator(1, 1);
    let rel_elem = a.multiply(&b).multiply(&a).multiply(&b);
    let rel = rel_elem
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the dicyclic group Dic_n
///
/// The dicyclic group of order 4n has generators `a` and `b` with relations:
/// - a^(2n) = 1
/// - b^2 = a^n
/// - b^{-1} a b = a^{-1}
///
/// # Arguments
///
/// * `n` - Parameter (group has order 4n, requires n >= 2)
pub fn dicyclic_presentation(n: usize) -> FinitelyPresentedGroup {
    assert!(n >= 2, "Dicyclic group requires n >= 2");

    let generator_names = vec!["a".to_string(), "b".to_string()];
    let mut relations = Vec::new();

    // a^(2n) = 1
    let rel = FreeGroupElement::generator(0, (2 * n) as isize)
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // b^2 a^{-n} = 1
    let b2 = FreeGroupElement::generator(1, 2);
    let a_neg_n = FreeGroupElement::generator(0, -(n as isize));
    let rel_elem = b2.multiply(&a_neg_n);
    let rel = rel_elem
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // b^{-1} a b a = 1
    let b_inv = FreeGroupElement::generator(1, -1);
    let a = FreeGroupElement::generator(0, 1);
    let b = FreeGroupElement::generator(1, 1);
    let rel_elem = b_inv.multiply(&a).multiply(&b).multiply(&a);
    let rel = rel_elem
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the quaternion group Q_8
///
/// The quaternion group of order 8 has generators `a` and `b` with relations:
/// - a^4 = 1
/// - b^2 = a^2
/// - bab^{-1} = a^{-1}
pub fn quaternion_presentation() -> FinitelyPresentedGroup {
    let generator_names = vec!["a".to_string(), "b".to_string()];
    let mut relations = Vec::new();

    // a^4 = 1
    let rel = FreeGroupElement::generator(0, 4)
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // b^2 a^{-2} = 1
    let b2 = FreeGroupElement::generator(1, 2);
    let a_neg2 = FreeGroupElement::generator(0, -2);
    let rel_elem = b2.multiply(&a_neg2);
    let rel = rel_elem
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // a b a b^{-1} = 1
    let a = FreeGroupElement::generator(0, 1);
    let b = FreeGroupElement::generator(1, 1);
    let b_inv = FreeGroupElement::generator(1, -1);
    let rel_elem = a.multiply(&b).multiply(&a).multiply(&b_inv);
    let rel = rel_elem
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the Klein four group
///
/// The Klein four group has generators `a` and `b` with relations:
/// - a^2 = 1
/// - b^2 = 1
/// - ab = ba (commutator)
pub fn klein_four_presentation() -> FinitelyPresentedGroup {
    let generator_names = vec!["a".to_string(), "b".to_string()];
    let mut relations = Vec::new();

    // a^2 = 1
    let rel = FreeGroupElement::generator(0, 2)
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // b^2 = 1
    let rel = FreeGroupElement::generator(1, 2)
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // a b a^{-1} b^{-1} = 1
    let a = FreeGroupElement::generator(0, 1);
    let b = FreeGroupElement::generator(1, 1);
    let a_inv = FreeGroupElement::generator(0, -1);
    let b_inv = FreeGroupElement::generator(1, -1);
    let comm = a.multiply(&b).multiply(&a_inv).multiply(&b_inv);
    let rel = comm
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the binary dihedral group
///
/// The binary dihedral group of order 4n has generators `x`, `y`, `z` with relations:
/// - x^{-2} y^2 = 1
/// - x^{-2} z^n = 1
/// - x^{-2} xyz = 1
///
/// # Arguments
///
/// * `n` - Parameter (group has order 4n)
pub fn binary_dihedral_presentation(n: usize) -> FinitelyPresentedGroup {
    assert!(n >= 1, "Binary dihedral group requires n >= 1");

    let generator_names = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let mut relations = Vec::new();

    // x^{-2} y^2 = 1
    let x_neg2 = FreeGroupElement::generator(0, -2);
    let y2 = FreeGroupElement::generator(1, 2);
    let rel_elem = x_neg2.multiply(&y2);
    let rel = rel_elem
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // x^{-2} z^n = 1
    let x_neg2_again = FreeGroupElement::generator(0, -2);
    let zn = FreeGroupElement::generator(2, n as isize);
    let rel_elem = x_neg2_again.multiply(&zn);
    let rel = rel_elem
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    // x^{-2} xyz = 1
    let x_neg2_third = FreeGroupElement::generator(0, -2);
    let x = FreeGroupElement::generator(0, 1);
    let y = FreeGroupElement::generator(1, 1);
    let z = FreeGroupElement::generator(2, 1);
    let rel_elem = x_neg2_third.multiply(&x).multiply(&y).multiply(&z);
    let rel = rel_elem
        .tietze()
        .into_iter()
        .map(|x| x as i32)
        .collect();
    relations.push(rel);

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the cactus group J_n
///
/// The cactus group on n fruits is defined with specific generators and relations
/// based on the structure of an n-fruit cactus graph. This is a simplified version
/// that creates a presentation with the appropriate number of generators.
///
/// # Arguments
///
/// * `n` - Number of fruits (must be positive)
///
/// # Note
///
/// This is a simplified implementation. The full cactus group has a more complex
/// structure with generators s_{ij} indexed by graph edges.
pub fn cactus_presentation(n: usize) -> FinitelyPresentedGroup {
    assert!(n >= 1, "Cactus group requires n >= 1");

    let generator_names: Vec<String> = (0..n)
        .map(|i| format!("s_{}", i))
        .collect();

    // For a cactus group on n fruits, we have n generators
    // with involution relations s_i^2 = 1 and braid-type relations
    let mut relations = Vec::new();

    // All generators are involutions
    for i in 0..n {
        let rel = FreeGroupElement::generator(i as isize, 2)
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    // Add braid-type relations for adjacent generators
    for i in 0..(n - 1) {
        let si = FreeGroupElement::generator(i as isize, 1);
        let si1 = FreeGroupElement::generator((i + 1) as isize, 1);
        // s_i s_{i+1} s_i = s_{i+1} s_i s_{i+1}
        // Which is equivalent to: s_i s_{i+1} s_i s_{i+1}^{-1} s_i^{-1} s_{i+1}^{-1} = 1
        let si_inv = FreeGroupElement::generator(i as isize, -1);
        let si1_inv = FreeGroupElement::generator((i + 1) as isize, -1);

        let rel_elem = si
            .multiply(&si1)
            .multiply(&si)
            .multiply(&si1_inv)
            .multiply(&si_inv)
            .multiply(&si1_inv);
        let rel = rel_elem
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the symmetric group S_n
///
/// This creates a presentation of the symmetric group using adjacent transpositions
/// as generators with standard braid relations.
///
/// # Arguments
///
/// * `n` - The degree of the symmetric group
///
/// # Note
///
/// This uses the Coxeter presentation with generators s_1, ..., s_{n-1}
/// representing adjacent transpositions (i, i+1).
pub fn symmetric_presentation(n: usize) -> FinitelyPresentedGroup {
    assert!(n >= 2, "Symmetric group requires n >= 2");

    let n_gens = n - 1;
    let generator_names: Vec<String> = (0..n_gens)
        .map(|i| format!("s_{}", i))
        .collect();
    let mut relations = Vec::new();

    // All generators have order 2: s_i^2 = 1
    for i in 0..n_gens {
        let rel = FreeGroupElement::generator(i as isize, 2)
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    // Braid relations: s_i s_{i+1} s_i = s_{i+1} s_i s_{i+1}
    for i in 0..(n_gens - 1) {
        let si = FreeGroupElement::generator(i as isize, 1);
        let si1 = FreeGroupElement::generator((i + 1) as isize, 1);
        let si_inv = FreeGroupElement::generator(i as isize, -1);
        let si1_inv = FreeGroupElement::generator((i + 1) as isize, -1);

        let rel_elem = si
            .multiply(&si1)
            .multiply(&si)
            .multiply(&si1_inv)
            .multiply(&si_inv)
            .multiply(&si1_inv);
        let rel = rel_elem
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    // Commuting relations: s_i s_j = s_j s_i for |i - j| >= 2
    for i in 0..n_gens {
        for j in (i + 2)..n_gens {
            let si = FreeGroupElement::generator(i as isize, 1);
            let sj = FreeGroupElement::generator(j as isize, 1);
            let si_inv = FreeGroupElement::generator(i as isize, -1);
            let sj_inv = FreeGroupElement::generator(j as isize, -1);

            let rel_elem = si.multiply(&sj).multiply(&si_inv).multiply(&sj_inv);
            let rel = rel_elem
                .tietze()
                .into_iter()
                .map(|x| x as i32)
                .collect();
            relations.push(rel);
        }
    }

    FinitelyPresentedGroup::new(generator_names, relations)
}

/// Create a presentation of the alternating group A_n
///
/// This creates a presentation of the alternating group using specific generators
/// and relations.
///
/// # Arguments
///
/// * `n` - The degree of the alternating group
///
/// # Note
///
/// For n >= 3, this uses 3-cycles as generators.
pub fn alternating_presentation(n: usize) -> FinitelyPresentedGroup {
    assert!(n >= 3, "Alternating group requires n >= 3");

    if n == 3 {
        // A_3 is cyclic of order 3
        return cyclic_presentation(3);
    }

    // For A_4 and larger, use a more complex presentation
    // This is a simplified version - full implementation would be more involved
    let n_gens = n - 2;
    let generator_names: Vec<String> = (0..n_gens)
        .map(|i| format!("s_{}", i))
        .collect();
    let mut relations = Vec::new();

    // Each generator has order 3 (they are 3-cycles)
    for i in 0..n_gens {
        let rel = FreeGroupElement::generator(i as isize, 3)
            .tietze()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        relations.push(rel);
    }

    // Add some commutator relations
    // This is a simplified presentation
    for i in 0..n_gens {
        for j in (i + 2)..n_gens {
            let si = FreeGroupElement::generator(i as isize, 1);
            let sj = FreeGroupElement::generator(j as isize, 1);
            let si_inv = FreeGroupElement::generator(i as isize, -1);
            let sj_inv = FreeGroupElement::generator(j as isize, -1);

            let rel_elem = si.multiply(&sj).multiply(&si_inv).multiply(&sj_inv);
            let rel = rel_elem
                .tietze()
                .into_iter()
                .map(|x| x as i32)
                .collect();
            relations.push(rel);
        }
    }

    FinitelyPresentedGroup::new(generator_names, relations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyclic_presentation() {
        let c5 = cyclic_presentation(5);
        // Should have 1 generator
        assert_eq!(c5.num_generators(), 1);
    }

    #[test]
    fn test_dihedral_presentation() {
        let d4 = dihedral_presentation(4);
        assert_eq!(d4.num_generators(), 2);
    }

    #[test]
    fn test_quaternion_presentation() {
        let q8 = quaternion_presentation();
        assert_eq!(q8.num_generators(), 2);
    }

    #[test]
    fn test_klein_four_presentation() {
        let v4 = klein_four_presentation();
        assert_eq!(v4.num_generators(), 2);
    }

    #[test]
    fn test_finitely_generated_abelian() {
        let g = finitely_generated_abelian_presentation(&[2, 3]);
        assert_eq!(g.num_generators(), 2);

        let g2 = finitely_generated_abelian_presentation(&[2, 3, 5]);
        assert_eq!(g2.num_generators(), 3);
    }

    #[test]
    fn test_heisenberg_presentation() {
        let h1 = finitely_generated_heisenberg_presentation(1, 0);
        assert_eq!(h1.num_generators(), 3); // x, y, z

        let h2 = finitely_generated_heisenberg_presentation(2, 0);
        assert_eq!(h2.num_generators(), 5); // x_0, x_1, y_0, y_1, z
    }

    #[test]
    fn test_dicyclic_presentation() {
        let dic3 = dicyclic_presentation(3);
        assert_eq!(dic3.num_generators(), 2);
    }

    #[test]
    fn test_binary_dihedral_presentation() {
        let bd4 = binary_dihedral_presentation(4);
        assert_eq!(bd4.num_generators(), 3);
    }

    #[test]
    fn test_cactus_presentation() {
        let j3 = cactus_presentation(3);
        assert_eq!(j3.num_generators(), 3);
    }

    #[test]
    fn test_symmetric_presentation() {
        let s3 = symmetric_presentation(3);
        assert_eq!(s3.num_generators(), 2);

        let s4 = symmetric_presentation(4);
        assert_eq!(s4.num_generators(), 3);
    }

    #[test]
    fn test_alternating_presentation() {
        let a3 = alternating_presentation(3);
        assert_eq!(a3.num_generators(), 1); // A_3 is cyclic

        let a4 = alternating_presentation(4);
        assert_eq!(a4.num_generators(), 2);
    }

    #[test]
    #[should_panic(expected = "Order must be positive")]
    fn test_cyclic_invalid() {
        cyclic_presentation(0);
    }

    #[test]
    #[should_panic(expected = "Dihedral group requires n >= 2")]
    fn test_dihedral_invalid() {
        dihedral_presentation(1);
    }

    #[test]
    #[should_panic(expected = "Dicyclic group requires n >= 2")]
    fn test_dicyclic_invalid() {
        dicyclic_presentation(1);
    }
}
