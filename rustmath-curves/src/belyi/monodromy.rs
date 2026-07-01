//! Permutations, Belyi monodromy triples, and the Riemann--Hurwitz genus from
//! branch cycles.
//!
//! Ported from `dessin_engine/src/permutation.rs` in
//! `/home/john/inverse_galois/M23/dessin_engine` (Layer 6b, D0/D3). The reference
//! implementation's private helpers are retyped onto RustMath conventions and the
//! transitivity check is delegated to
//! [`rustmath_groups::perm_predicates::is_transitive`] (image-list convention).
//!
//! A dessin of degree `n` is a transitive triple `(σ0, σ1, σ∞)` of permutations
//! of `{0,…,n−1}` with `σ0 σ1 σ∞ = id`. The genus is read straight off the cycle
//! defects, with no function-field computation:
//! `2g − 2 = −2n + Σ_i defect(σ_i)`, `defect = n − #cycles`, i.e.
//! `g = 1 − n + (Σ defect)/2`. Genus lives *here*, with the covers, not in the
//! crate's placeholder `riemann_roch.rs`.

use rustmath_groups::perm_predicates::is_transitive;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PermError {
    #[error("permutation images are not a rearrangement of 0..n")]
    NotPermutation,
    #[error("permutations have different degrees")]
    DegreeMismatch,
    #[error("triple product is not identity")]
    ProductNotIdentity,
    #[error("generated action is not transitive")]
    NotTransitive,
    #[error("Riemann-Hurwitz defect sum is odd")]
    OddDefectSum,
}

/// A permutation as the image list `image[i] = σ(i)`, zero-based.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    image: Vec<usize>,
}

impl Permutation {
    pub fn new(image: Vec<usize>) -> Result<Self, PermError> {
        let n = image.len();
        let mut seen = vec![false; n];
        for &x in &image {
            if x >= n || seen[x] {
                return Err(PermError::NotPermutation);
            }
            seen[x] = true;
        }
        Ok(Self { image })
    }

    /// Build from disjoint cycles on `n` points (points not listed are fixed).
    pub fn from_cycles(n: usize, cycles: &[Vec<usize>]) -> Result<Self, PermError> {
        let mut image: Vec<usize> = (0..n).collect();
        for cyc in cycles {
            let len = cyc.len();
            for i in 0..len {
                if cyc[i] >= n {
                    return Err(PermError::NotPermutation);
                }
                image[cyc[i]] = cyc[(i + 1) % len];
            }
        }
        Self::new(image)
    }

    pub fn identity(n: usize) -> Self {
        Self {
            image: (0..n).collect(),
        }
    }

    pub fn degree(&self) -> usize {
        self.image.len()
    }

    /// The underlying image list `image[i] = σ(i)`.
    pub fn image(&self) -> &[usize] {
        &self.image
    }

    /// Function composition `self ∘ other`: `i ↦ self(other(i))`.
    pub fn compose(&self, other: &Self) -> Result<Self, PermError> {
        if self.degree() != other.degree() {
            return Err(PermError::DegreeMismatch);
        }
        let image = other.image.iter().map(|&i| self.image[i]).collect();
        Self::new(image)
    }

    pub fn inverse(&self) -> Self {
        let n = self.degree();
        let mut inv = vec![0; n];
        for (i, &j) in self.image.iter().enumerate() {
            inv[j] = i;
        }
        Self { image: inv }
    }

    pub fn is_identity(&self) -> bool {
        self.image.iter().enumerate().all(|(i, &j)| i == j)
    }

    /// Cycle lengths, descending (includes fixed points as length-1 cycles).
    pub fn cycle_lengths(&self) -> Vec<usize> {
        let n = self.degree();
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            let mut len = 0;
            let mut x = start;
            while !seen[x] {
                seen[x] = true;
                len += 1;
                x = self.image[x];
            }
            out.push(len);
        }
        out.sort_unstable_by(|a, b| b.cmp(a));
        out
    }

    pub fn cycle_count(&self) -> usize {
        self.cycle_lengths().len()
    }

    /// `n − #cycles`: the local Riemann--Hurwitz contribution.
    pub fn defect(&self) -> usize {
        self.degree() - self.cycle_count()
    }

    pub fn apply(&self, i: usize) -> usize {
        self.image[i]
    }
}

/// A dessin: `σ0 σ1 σ∞ = id`, transitive.
#[derive(Debug, Clone)]
pub struct BelyiTriple {
    pub sigma0: Permutation,
    pub sigma1: Permutation,
    pub sigmainf: Permutation,
}

impl BelyiTriple {
    /// Equal degrees, `σ0 ∘ σ1 ∘ σ∞ = id`, and transitivity.
    pub fn validate(&self) -> Result<(), PermError> {
        let n = self.sigma0.degree();
        if self.sigma1.degree() != n || self.sigmainf.degree() != n {
            return Err(PermError::DegreeMismatch);
        }
        let prod = self.sigma0.compose(&self.sigma1)?.compose(&self.sigmainf)?;
        if !prod.is_identity() {
            return Err(PermError::ProductNotIdentity);
        }
        // Transitivity via rustmath-groups (image-list convention shared with
        // `perm_predicates::Perm = Vec<usize>`).
        let gens = [
            self.sigma0.image().to_vec(),
            self.sigma1.image().to_vec(),
            self.sigmainf.image().to_vec(),
        ];
        if n == 0 || !is_transitive(&gens, n) {
            return Err(PermError::NotTransitive);
        }
        Ok(())
    }

    pub fn genus(&self) -> Result<i64, PermError> {
        genus_from_branch_cycles(
            self.sigma0.degree(),
            &[&self.sigma0, &self.sigma1, &self.sigmainf],
        )
    }

    /// The three cycle-type signatures (descending lengths), for reporting.
    pub fn cycle_types(&self) -> [Vec<usize>; 3] {
        [
            self.sigma0.cycle_lengths(),
            self.sigma1.cycle_lengths(),
            self.sigmainf.cycle_lengths(),
        ]
    }
}

/// `g = 1 − n + (Σ defect)/2`. Errors on an odd defect sum (impossible for a
/// genuine cover — a useful corruption check).
pub fn genus_from_branch_cycles(
    degree: usize,
    branch_cycles: &[&Permutation],
) -> Result<i64, PermError> {
    let mut defect_sum = 0usize;
    for sigma in branch_cycles {
        if sigma.degree() != degree {
            return Err(PermError::DegreeMismatch);
        }
        defect_sum += sigma.defect();
    }
    if defect_sum % 2 != 0 {
        return Err(PermError::OddDefectSum);
    }
    Ok(1 - degree as i64 + (defect_sum as i64) / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ported from `dessin_engine/src/permutation.rs::tests`.
    #[test]
    fn defect_genus_for_2_12_5_pattern_is_zero() {
        // 2^8 1^8 -> 16 cycles, defect 8;  12^2 -> 2 cycles, defect 22;
        // 5^4 1^4 -> 8 cycles, defect 16.  total 46, n=24 -> g = 1-24+23 = 0.
        let n = 24;
        let two_cycles: Vec<Vec<usize>> = (0..8).map(|k| vec![2 * k, 2 * k + 1]).collect();
        let sigma0 = Permutation::from_cycles(n, &two_cycles).unwrap();
        let sigma1 = Permutation::from_cycles(
            n,
            &[(0..12).collect::<Vec<_>>(), (12..24).collect::<Vec<_>>()],
        )
        .unwrap();
        let sigmainf = Permutation::from_cycles(
            n,
            &[
                vec![0, 1, 2, 3, 4],
                vec![5, 6, 7, 8, 9],
                vec![10, 11, 12, 13, 14],
                vec![15, 16, 17, 18, 19],
            ],
        )
        .unwrap();

        assert_eq!(sigma0.defect(), 8);
        assert_eq!(sigma1.defect(), 22);
        assert_eq!(sigmainf.defect(), 16);
        assert_eq!(
            genus_from_branch_cycles(n, &[&sigma0, &sigma1, &sigmainf]).unwrap(),
            0
        );
    }

    #[test]
    fn validate_catches_nonidentity_product() {
        let s0 = Permutation::from_cycles(3, &[vec![0, 1, 2]]).unwrap();
        let s1 = Permutation::identity(3);
        let sinf = s0.inverse();
        let t = BelyiTriple {
            sigma0: s0.clone(),
            sigma1: s1,
            sigmainf: sinf,
        };
        assert!(t.validate().is_ok());
        assert_eq!(t.genus().unwrap(), 0);

        let bad = BelyiTriple {
            sigma0: s0,
            sigma1: Permutation::identity(3),
            sigmainf: Permutation::identity(3),
        };
        assert_eq!(bad.validate(), Err(PermError::ProductNotIdentity));
    }

    #[test]
    fn validate_catches_nontransitive() {
        // Two disjoint transpositions on 4 points, product identity, but the
        // action splits into {0,1} and {2,3} — not transitive.
        let s0 = Permutation::from_cycles(4, &[vec![0, 1]]).unwrap();
        let s1 = Permutation::from_cycles(4, &[vec![2, 3]]).unwrap();
        let sinf = s0.compose(&s1).unwrap().inverse();
        let t = BelyiTriple {
            sigma0: s0,
            sigma1: s1,
            sigmainf: sinf,
        };
        assert_eq!(t.validate(), Err(PermError::NotTransitive));
    }
}
