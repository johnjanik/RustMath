//! GAP Permutation Group Algorithms
//!
//! This module provides permutation group algorithms using GAP's powerful
//! computational group theory functionality.
//!
//! # Overview
//!
//! GAP provides highly optimized algorithms for:
//! - Schreier-Sims algorithm for base and strong generating set
//! - Membership testing
//! - Stabilizer chains
//! - Normalizers and centralizers
//! - Subgroup lattice computations
//! - Point stabilizers
//!
//! # Example
//!
//! ```rust,ignore
//! use rustmath_interfaces::gap_permutation::GapPermutationGroup;
//!
//! let gap_group = GapPermutationGroup::symmetric(5)?;
//! let order = gap_group.order()?;
//! assert_eq!(order, 120);
//! ```

use crate::gap::{GapError, GapInterface, Result};
use crate::gap_parser::{parse_permutation, Permutation};

/// A permutation group interfaced through GAP
pub struct GapPermutationGroup {
    gap: GapInterface,
    gap_name: String,
    degree: usize,
}

impl GapPermutationGroup {
    /// Create a symmetric group S_n via GAP
    pub fn symmetric(n: usize) -> Result<Self> {
        let gap = GapInterface::new()?;
        let gap_name = format!("SymmetricGroup({})", n);

        // Verify it was created
        gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap,
            gap_name,
            degree: n,
        })
    }

    /// Create an alternating group A_n via GAP
    pub fn alternating(n: usize) -> Result<Self> {
        let gap = GapInterface::new()?;
        let gap_name = format!("AlternatingGroup({})", n);

        gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap,
            gap_name,
            degree: n,
        })
    }

    /// Create a cyclic group Z_n via GAP (as a permutation group)
    pub fn cyclic(n: usize) -> Result<Self> {
        let gap = GapInterface::new()?;
        let gap_name = format!("CyclicGroup(IsPermGroup, {})", n);

        gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap,
            gap_name,
            degree: n,
        })
    }

    /// Create a dihedral group D_n via GAP (as a permutation group)
    pub fn dihedral(n: usize) -> Result<Self> {
        let gap = GapInterface::new()?;
        let gap_name = format!("DihedralGroup(IsPermGroup, {})", n);

        gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap,
            gap_name,
            degree: n,
        })
    }

    /// Create a permutation group from generators
    pub fn from_generators(generators: &[Permutation]) -> Result<Self> {
        let gap = GapInterface::new()?;

        // Convert generators to GAP format
        let gap_gens: Vec<String> = generators
            .iter()
            .map(|perm| {
                let cycles: Vec<String> = perm
                    .cycles
                    .iter()
                    .map(|cycle| {
                        format!("({})", cycle.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                    })
                    .collect();
                if cycles.is_empty() {
                    "()".to_string()
                } else {
                    cycles.join("")
                }
            })
            .collect();

        let gap_name = format!("Group([{}])", gap_gens.join(", "));
        gap.execute(&format!("{};;", gap_name))?;

        let degree = generators.iter().map(|p| p.degree).max().unwrap_or(0);

        Ok(Self {
            gap,
            gap_name,
            degree,
        })
    }

    /// Get the order of the group
    pub fn order(&self) -> Result<usize> {
        self.gap.group_order(&self.gap_name)
    }

    /// Get the degree of the permutation group
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Check if the group is abelian
    pub fn is_abelian(&self) -> Result<bool> {
        self.gap.is_abelian(&self.gap_name)
    }

    /// Check if the group is simple
    pub fn is_simple(&self) -> Result<bool> {
        self.gap.is_simple(&self.gap_name)
    }

    /// Check if the group is solvable
    pub fn is_solvable(&self) -> Result<bool> {
        self.gap.is_solvable(&self.gap_name)
    }

    /// Get generators of the group
    pub fn generators(&self) -> Result<Vec<Permutation>> {
        let result = self.gap.execute(&format!("GeneratorsOfGroup({})", self.gap_name))?;

        // Parse the result which should be a list like [ (1,2), (1,2,3,4,5) ]
        let result = result.trim();
        if result.starts_with('[') && result.ends_with(']') {
            let content = &result[1..result.len() - 1];

            // Split by permutation boundaries
            let mut perms = Vec::new();
            let mut current = String::new();
            let mut depth = 0;

            for ch in content.chars() {
                if ch == '(' {
                    depth += 1;
                }
                if ch == ')' {
                    depth -= 1;
                }

                current.push(ch);

                if depth == 0 && ch == ')' {
                    if let Ok(perm) = parse_permutation(&current.trim()) {
                        perms.push(perm);
                    }
                    current.clear();
                }
            }

            Ok(perms)
        } else {
            Err(GapError::ParseError(format!(
                "Unexpected generators format: {}",
                result
            )))
        }
    }

    /// Compute the center of the group
    pub fn center(&self) -> Result<Self> {
        let gap_name = format!("Center({})", self.gap_name);
        self.gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap: GapInterface::new()?,
            gap_name,
            degree: self.degree,
        })
    }

    /// Compute the derived subgroup (commutator subgroup)
    pub fn derived_subgroup(&self) -> Result<Self> {
        let gap_name = format!("DerivedSubgroup({})", self.gap_name);
        self.gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap: GapInterface::new()?,
            gap_name,
            degree: self.degree,
        })
    }

    /// Get the number of conjugacy classes
    pub fn num_conjugacy_classes(&self) -> Result<usize> {
        self.gap.num_conjugacy_classes(&self.gap_name)
    }

    /// Compute the stabilizer of a point
    pub fn stabilizer(&self, point: usize) -> Result<Self> {
        let gap_name = format!("Stabilizer({}, {})", self.gap_name, point);
        self.gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap: GapInterface::new()?,
            gap_name,
            degree: self.degree,
        })
    }

    /// Compute the orbit of a point
    pub fn orbit(&self, point: usize) -> Result<Vec<usize>> {
        let result = self
            .gap
            .execute(&format!("Orbit({}, {})", self.gap_name, point))?;

        // Parse the orbit which should be a list like [ 1, 2, 3, 4, 5 ]
        crate::gap_parser::parse_integer_list(&result)
            .map(|v| v.into_iter().map(|x| x as usize).collect())
            .map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Get all orbits of the group
    pub fn orbits(&self) -> Result<Vec<Vec<usize>>> {
        let result = self.gap.execute(&format!("Orbits({})", self.gap_name))?;

        // Parse nested list structure
        // This is a simplified parser - may need enhancement for complex cases
        let result = result.trim();
        if !result.starts_with('[') || !result.ends_with(']') {
            return Err(GapError::ParseError(format!(
                "Invalid orbits format: {}",
                result
            )));
        }

        let content = &result[1..result.len() - 1];
        let mut orbits = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in content.chars() {
            if ch == '[' {
                depth += 1;
                if depth == 1 {
                    current.clear();
                    current.push('[');
                    continue;
                }
            }
            if ch == ']' {
                depth -= 1;
                current.push(']');
                if depth == 0 {
                    if let Ok(orbit) = crate::gap_parser::parse_integer_list(&current) {
                        orbits.push(orbit.into_iter().map(|x| x as usize).collect());
                    }
                    current.clear();
                    continue;
                }
            }
            if depth > 0 {
                current.push(ch);
            }
        }

        Ok(orbits)
    }

    /// Check if the group is transitive
    pub fn is_transitive(&self) -> Result<bool> {
        let result = self
            .gap
            .execute(&format!("IsTransitive({})", self.gap_name))?;
        crate::gap_parser::parse_boolean(&result)
            .map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Check if the group is primitive
    pub fn is_primitive(&self) -> Result<bool> {
        let result = self
            .gap
            .execute(&format!("IsPrimitive({})", self.gap_name))?;
        crate::gap_parser::parse_boolean(&result)
            .map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Get a base and strong generating set (Schreier-Sims algorithm)
    pub fn base_and_strong_generators(&self) -> Result<(Vec<usize>, Vec<Permutation>)> {
        // Get base
        let base_result = self
            .gap
            .execute(&format!("BaseOfGroup({})", self.gap_name))?;
        let base = crate::gap_parser::parse_integer_list(&base_result)
            .map(|v| v.into_iter().map(|x| x as usize).collect())
            .map_err(|e| GapError::ParseError(e.to_string()))?;

        // Get strong generators
        let sgs_result = self
            .gap
            .execute(&format!("StrongGeneratorsStabChain(StabChain({}))", self.gap_name))?;

        // Parse generators
        let sgs = if sgs_result.starts_with('[') && sgs_result.ends_with(']') {
            let content = &sgs_result[1..sgs_result.len() - 1];
            let mut perms = Vec::new();
            let mut current = String::new();
            let mut depth = 0;

            for ch in content.chars() {
                if ch == '(' {
                    depth += 1;
                }
                if ch == ')' {
                    depth -= 1;
                }

                current.push(ch);

                if depth == 0 && ch == ')' {
                    if let Ok(perm) = parse_permutation(&current.trim()) {
                        perms.push(perm);
                    }
                    current.clear();
                }
            }

            perms
        } else {
            Vec::new()
        };

        Ok((base, sgs))
    }

    /// Test membership of a permutation in the group
    pub fn contains(&self, perm: &Permutation) -> Result<bool> {
        // Convert permutation to GAP format
        let cycles: Vec<String> = perm
            .cycles
            .iter()
            .map(|cycle| {
                format!("({})", cycle.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
            })
            .collect();

        let perm_str = if cycles.is_empty() {
            "()".to_string()
        } else {
            cycles.join("")
        };

        let result = self
            .gap
            .execute(&format!("{} in {}", perm_str, self.gap_name))?;

        crate::gap_parser::parse_boolean(&result)
            .map_err(|e| GapError::ParseError(e.to_string()))
    }

    /// Compute the normalizer of a subgroup
    pub fn normalizer(&self, subgroup: &GapPermutationGroup) -> Result<Self> {
        let gap_name = format!("Normalizer({}, {})", self.gap_name, subgroup.gap_name);
        self.gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap: GapInterface::new()?,
            gap_name,
            degree: self.degree,
        })
    }

    /// Compute the centralizer of a subgroup
    pub fn centralizer(&self, subgroup: &GapPermutationGroup) -> Result<Self> {
        let gap_name = format!("Centralizer({}, {})", self.gap_name, subgroup.gap_name);
        self.gap.execute(&format!("{};;", gap_name))?;

        Ok(Self {
            gap: GapInterface::new()?,
            gap_name,
            degree: self.degree,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GAP
    fn test_symmetric_group_order() {
        let s5 = GapPermutationGroup::symmetric(5).unwrap();
        assert_eq!(s5.order().unwrap(), 120);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_alternating_group_order() {
        let a5 = GapPermutationGroup::alternating(5).unwrap();
        assert_eq!(a5.order().unwrap(), 60);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_group_properties() {
        let s3 = GapPermutationGroup::symmetric(3).unwrap();
        assert!(!s3.is_abelian().unwrap());
        assert!(s3.is_solvable().unwrap());
        assert!(!s3.is_simple().unwrap());

        let a5 = GapPermutationGroup::alternating(5).unwrap();
        assert!(!a5.is_abelian().unwrap());
        assert!(a5.is_simple().unwrap());
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_generators() {
        let s3 = GapPermutationGroup::symmetric(3).unwrap();
        let gens = s3.generators().unwrap();
        assert!(!gens.is_empty());
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_orbit() {
        let s5 = GapPermutationGroup::symmetric(5).unwrap();
        let orbit = s5.orbit(1).unwrap();
        assert_eq!(orbit.len(), 5); // S5 acts transitively on {1,2,3,4,5}
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_stabilizer() {
        let s5 = GapPermutationGroup::symmetric(5).unwrap();
        let stab = s5.stabilizer(1).unwrap();
        assert_eq!(stab.order().unwrap(), 24); // |Stab(1)| = 4! = 24
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_conjugacy_classes() {
        let s4 = GapPermutationGroup::symmetric(4).unwrap();
        assert_eq!(s4.num_conjugacy_classes().unwrap(), 5);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_is_transitive() {
        let s5 = GapPermutationGroup::symmetric(5).unwrap();
        assert!(s5.is_transitive().unwrap());
    }
}
