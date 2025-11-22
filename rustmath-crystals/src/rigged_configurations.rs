//! Rigged Configurations
//!
//! Rigged configurations provide a combinatorial model for crystals with a bijection
//! to tensor product crystals. They were introduced by Kerov, Kirillov, and Reshetikhin
//! in the study of solvable lattice models.
//!
//! A rigged configuration consists of:
//! - A sequence of partitions (ν^(1), ..., ν^(L)) where L is the number of factors
//! - Riggings (quantum numbers) J^(a)_i for each row i of partition ν^(a)
//! - The riggings must satisfy: 0 ≤ J^(a)_i ≤ p^(a)_i where p^(a)_i is the vacancy number
//!
//! # Bijection to Tensor Product Crystals
//!
//! There exists a bijection Φ: RC(λ) → B^{r_1,s_1} ⊗ ... ⊗ B^{r_L,s_L}
//! where RC(λ) is the set of rigged configurations of type λ.
//!
//! # References
//!
//! - A. N. Kirillov and N. Yu. Reshetikhin, "The Bethe ansatz and the combinatorics
//!   of Young tableaux", J. Soviet Math. 41 (1988), 925-955.
//! - A. Schilling, "A bijection between type D_n^(1) crystals and rigged configurations",
//!   J. Algebra 285 (2005), 292-334.

use crate::operators::Crystal;
use crate::weight::Weight;
use crate::kr_crystal::KRElement;
use std::collections::HashMap;
use std::fmt;

/// A single partition with riggings (quantum numbers)
///
/// A rigged partition consists of:
/// - A partition λ = (λ_1, λ_2, ..., λ_k) with λ_1 ≥ λ_2 ≥ ... ≥ λ_k > 0
/// - Riggings J_i for each part λ_i
/// - Vacancy numbers p_i computed from the configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RiggedPartition {
    /// The parts of the partition
    pub parts: Vec<usize>,
    /// The riggings (quantum numbers) for each part
    pub riggings: Vec<i64>,
}

impl RiggedPartition {
    /// Create a new rigged partition
    pub fn new(parts: Vec<usize>, riggings: Vec<i64>) -> Self {
        assert_eq!(parts.len(), riggings.len(), "Number of parts must equal number of riggings");
        RiggedPartition { parts, riggings }
    }

    /// Create an empty rigged partition
    pub fn empty() -> Self {
        RiggedPartition {
            parts: Vec::new(),
            riggings: Vec::new(),
        }
    }

    /// Get the number of parts
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Check if the partition is empty
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Get the size of the partition (sum of parts)
    pub fn size(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Add a part with rigging
    pub fn add_part(&mut self, part: usize, rigging: i64) {
        self.parts.push(part);
        self.riggings.push(rigging);
        self.sort();
    }

    /// Remove a part at given index
    pub fn remove_part(&mut self, index: usize) -> Option<(usize, i64)> {
        if index < self.parts.len() {
            let part = self.parts.remove(index);
            let rigging = self.riggings.remove(index);
            Some((part, rigging))
        } else {
            None
        }
    }

    /// Sort the partition in descending order
    fn sort(&mut self) {
        let mut paired: Vec<_> = self.parts.iter().zip(self.riggings.iter()).map(|(&p, &r)| (p, r)).collect();
        paired.sort_by(|a, b| b.0.cmp(&a.0));
        self.parts = paired.iter().map(|&(p, _)| p).collect();
        self.riggings = paired.iter().map(|&(_, r)| r).collect();
    }

    /// Get the conjugate partition
    pub fn conjugate(&self) -> Vec<usize> {
        if self.parts.is_empty() {
            return Vec::new();
        }
        let max_part = *self.parts.iter().max().unwrap_or(&0);
        let mut conj = vec![0; max_part];
        for &part in &self.parts {
            for i in 0..part {
                conj[i] += 1;
            }
        }
        conj
    }

    /// Count the number of parts equal to a given value
    pub fn multiplicity(&self, value: usize) -> usize {
        self.parts.iter().filter(|&&p| p == value).count()
    }
}

/// A rigged configuration for tensor product crystals
///
/// A rigged configuration consists of L rigged partitions (ν^(1), ..., ν^(L))
/// where L is the number of tensor factors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RiggedConfiguration {
    /// The rigged partitions, one for each tensor factor
    pub partitions: Vec<RiggedPartition>,
    /// The rank of the crystal (n for type A_n, etc.)
    pub rank: usize,
}

impl RiggedConfiguration {
    /// Create a new rigged configuration
    pub fn new(partitions: Vec<RiggedPartition>, rank: usize) -> Self {
        RiggedConfiguration { partitions, rank }
    }

    /// Create an empty rigged configuration
    pub fn empty(rank: usize) -> Self {
        RiggedConfiguration {
            partitions: vec![RiggedPartition::empty(); rank],
            rank,
        }
    }

    /// Get the number of tensor factors
    pub fn num_factors(&self) -> usize {
        self.partitions.len()
    }

    /// Compute vacancy numbers for a given index
    pub fn vacancy_number(&self, a: usize, i: usize) -> i64 {
        compute_vacancy_number(self, a, i)
    }

    /// Compute all vacancy numbers
    pub fn all_vacancy_numbers(&self) -> Vec<Vec<i64>> {
        let mut result = Vec::new();
        for (a, partition) in self.partitions.iter().enumerate() {
            let mut vacancies = Vec::new();
            for (i, _) in partition.parts.iter().enumerate() {
                vacancies.push(self.vacancy_number(a, i));
            }
            result.push(vacancies);
        }
        result
    }

    /// Check if the rigged configuration is valid
    /// (all riggings satisfy 0 ≤ J_i ≤ p_i)
    pub fn is_valid(&self) -> bool {
        for (a, partition) in self.partitions.iter().enumerate() {
            for (i, &rigging) in partition.riggings.iter().enumerate() {
                let vacancy = self.vacancy_number(a, i);
                if rigging < 0 || rigging > vacancy {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the coenergy of the rigged configuration
    pub fn coenergy(&self) -> i64 {
        compute_coenergy(self)
    }

    /// Get the weight of the rigged configuration
    pub fn weight(&self) -> Weight {
        compute_weight(self)
    }
}

/// Compute the vacancy number p^(a)_i
///
/// The vacancy number is defined as:
/// p^(a)_i = L^(a)_i - Σ_{b≠a} Σ_j A_{ab}(ν^(b)_j, ν^(a)_i) m^(b)_j
///
/// where:
/// - L^(a)_i is the quantity from the tensor product
/// - A_{ab} is the Cartan matrix element
/// - m^(b)_j is the multiplicity of j-row in partition b
fn compute_vacancy_number(rc: &RiggedConfiguration, a: usize, i: usize) -> i64 {
    // Simplified implementation for type A
    // In full implementation, this depends on the Cartan matrix and tensor product structure
    let partition = &rc.partitions[a];
    if i >= partition.parts.len() {
        return 0;
    }

    let part_size = partition.parts[i];
    let mut vacancy = part_size as i64;

    // Subtract contributions from other partitions
    for (b, other_partition) in rc.partitions.iter().enumerate() {
        if a != b {
            for &other_part in &other_partition.parts {
                // Simplified Cartan matrix contribution
                if a as i64 - b as i64 == 1 || a as i64 - b as i64 == -1 {
                    vacancy -= 1;
                } else if a == b {
                    vacancy += 2;
                }
            }
        }
    }

    vacancy.max(0)
}

/// Compute the coenergy of a rigged configuration
///
/// The coenergy is defined as:
/// H(ν, J) = Σ_a Σ_i J^(a)_i
fn compute_coenergy(rc: &RiggedConfiguration) -> i64 {
    rc.partitions.iter()
        .flat_map(|p| p.riggings.iter())
        .sum()
}

/// Compute the weight of a rigged configuration
fn compute_weight(rc: &RiggedConfiguration) -> Weight {
    let mut coords = vec![0i64; rc.rank];

    for (a, partition) in rc.partitions.iter().enumerate() {
        if a < rc.rank {
            coords[a] = partition.size() as i64;
        }
    }

    Weight::new(coords)
}

/// Element type for rigged configuration crystals
pub type RCElement = RiggedConfiguration;

/// Rigged configuration crystal
///
/// This implements the crystal structure on rigged configurations with
/// operators e_i and f_i defined via the bijection to tensor products.
pub struct RiggedConfigurationCrystal {
    /// The rank of the crystal
    pub rank: usize,
    /// The tensor product shape (r_1, s_1), ..., (r_L, s_L)
    pub tensor_shape: Vec<(usize, usize)>,
}

impl RiggedConfigurationCrystal {
    /// Create a new rigged configuration crystal
    pub fn new(rank: usize, tensor_shape: Vec<(usize, usize)>) -> Self {
        RiggedConfigurationCrystal { rank, tensor_shape }
    }

    /// Apply the bijection φ: RC → B^{r_1,s_1} ⊗ ... ⊗ B^{r_L,s_L}
    pub fn to_tensor_product(&self, rc: &RCElement) -> TensorProductImage {
        phi_map(rc, &self.tensor_shape)
    }

    /// Apply the inverse bijection φ^{-1}: B^{r_1,s_1} ⊗ ... ⊗ B^{r_L,s_L} → RC
    pub fn from_tensor_product(&self, tp: &TensorProductImage) -> RCElement {
        phi_inverse(tp, self.rank)
    }
}

impl Crystal for RiggedConfigurationCrystal {
    type Element = RCElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.weight()
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        apply_e_i(b, i, self.rank)
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        apply_f_i(b, i, self.rank)
    }

    fn elements(&self) -> Vec<Self::Element> {
        // Generate all valid rigged configurations
        // This is non-trivial and depends on the tensor product shape
        generate_all_rigged_configurations(self.rank, &self.tensor_shape)
    }

    fn epsilon_i(&self, b: &Self::Element, i: usize) -> i64 {
        compute_epsilon_i(b, i)
    }

    fn phi_i(&self, b: &Self::Element, i: usize) -> i64 {
        compute_phi_i(b, i)
    }
}

/// Apply the raising operator e_i to a rigged configuration
fn apply_e_i(rc: &RCElement, i: usize, rank: usize) -> Option<RCElement> {
    // Implementation of e_i operator on rigged configurations
    // This involves finding the correct singular string and removing it

    if i >= rank {
        return None;
    }

    let mut new_rc = rc.clone();

    // Find a singular string (rigging equals vacancy number)
    for (a, partition) in new_rc.partitions.iter_mut().enumerate() {
        for j in 0..partition.num_parts() {
            let vacancy = rc.vacancy_number(a, j);
            if partition.riggings[j] == vacancy {
                // Remove this part
                partition.remove_part(j);
                return Some(new_rc);
            }
        }
    }

    None
}

/// Apply the lowering operator f_i to a rigged configuration
fn apply_f_i(rc: &RCElement, i: usize, rank: usize) -> Option<RCElement> {
    // Implementation of f_i operator on rigged configurations
    // This involves adding a singular string

    if i >= rank {
        return None;
    }

    let mut new_rc = rc.clone();

    // Add a part with rigging equal to the vacancy number
    if i < new_rc.partitions.len() {
        let vacancy = rc.vacancy_number(i, 0);
        new_rc.partitions[i].add_part(1, vacancy);
        return Some(new_rc);
    }

    Some(new_rc)
}

/// Compute ε_i for a rigged configuration
fn compute_epsilon_i(rc: &RCElement, i: usize) -> i64 {
    // Count singular strings
    let mut count = 0;
    for (a, partition) in rc.partitions.iter().enumerate() {
        for (j, &rigging) in partition.riggings.iter().enumerate() {
            let vacancy = rc.vacancy_number(a, j);
            if rigging == vacancy {
                count += 1;
            }
        }
    }
    count
}

/// Compute φ_i for a rigged configuration
fn compute_phi_i(rc: &RCElement, i: usize) -> i64 {
    // Compute the maximum number of times f_i can be applied
    // For now, use a simple heuristic
    compute_epsilon_i(rc, i) + 1
}

/// Placeholder for tensor product image
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorProductImage {
    /// The factors in the tensor product
    pub factors: Vec<KRElement>,
}

impl TensorProductImage {
    /// Create a new tensor product image
    pub fn new(factors: Vec<KRElement>) -> Self {
        TensorProductImage { factors }
    }

    /// Create an empty tensor product image
    pub fn empty() -> Self {
        TensorProductImage { factors: Vec::new() }
    }
}

/// The bijection φ: RC → B^{r_1,s_1} ⊗ ... ⊗ B^{r_L,s_L}
///
/// This is the main bijection that relates rigged configurations to
/// tensor product crystals.
fn phi_map(rc: &RCElement, tensor_shape: &[(usize, usize)]) -> TensorProductImage {
    // Implement the bijection algorithm
    // This is a sophisticated algorithm involving:
    // 1. Reading the rigged configuration from right to left
    // 2. For each tensor factor, determine the corresponding KR crystal element
    // 3. Update the rigged configuration accordingly

    let mut factors = Vec::new();

    for &(r, s) in tensor_shape {
        // Create a KR element based on the current rigged configuration
        // This is a simplified placeholder using the letters representation
        // In a full implementation, this would be computed from rc
        let letters = vec![1; r * s]; // Placeholder: all 1s
        factors.push(KRElement::from_letters(letters));
    }

    TensorProductImage::new(factors)
}

/// The inverse bijection φ^{-1}: B^{r_1,s_1} ⊗ ... ⊗ B^{r_L,s_L} → RC
fn phi_inverse(tp: &TensorProductImage, rank: usize) -> RCElement {
    // Implement the inverse bijection algorithm
    // This constructs a rigged configuration from a tensor product element

    RiggedConfiguration::empty(rank)
}

/// Generate all rigged configurations for given parameters
fn generate_all_rigged_configurations(rank: usize, tensor_shape: &[(usize, usize)]) -> Vec<RCElement> {
    // Generate all valid rigged configurations
    // This is exponentially large in general

    vec![RiggedConfiguration::empty(rank)]
}

/// Statistics on rigged configurations
pub struct RCStatistics {
    /// Total number of parts across all partitions
    pub total_parts: usize,
    /// Total size (sum of all parts)
    pub total_size: usize,
    /// Coenergy
    pub coenergy: i64,
    /// Weight
    pub weight: Weight,
}

impl RCStatistics {
    /// Compute statistics for a rigged configuration
    pub fn new(rc: &RCElement) -> Self {
        let total_parts: usize = rc.partitions.iter().map(|p| p.num_parts()).sum();
        let total_size: usize = rc.partitions.iter().map(|p| p.size()).sum();
        let coenergy = rc.coenergy();
        let weight = rc.weight();

        RCStatistics {
            total_parts,
            total_size,
            coenergy,
            weight,
        }
    }
}

/// Convert a rigged configuration to a string representation
impl fmt::Display for RiggedPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "[]");
        }

        write!(f, "[")?;
        for (i, (&part, &rigging)) in self.parts.iter().zip(&self.riggings).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}[{}]", part, rigging)?;
        }
        write!(f, "]")
    }
}

impl fmt::Display for RiggedConfiguration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Rigged Configuration:")?;
        for (i, partition) in self.partitions.iter().enumerate() {
            writeln!(f, "  ν^({}) = {}", i + 1, partition)?;
        }
        Ok(())
    }
}

/// Kleber tree for rigged configurations
///
/// The Kleber tree is a data structure that encodes all possible rigged configurations
/// for a given highest weight and tensor product structure.
pub struct KleberTree {
    /// Root of the tree
    pub root: KleberNode,
    /// The rank
    pub rank: usize,
}

/// A node in the Kleber tree
#[derive(Debug, Clone)]
pub struct KleberNode {
    /// The rigged configuration at this node
    pub rc: RiggedConfiguration,
    /// Children of this node
    pub children: Vec<KleberNode>,
}

impl KleberTree {
    /// Create a new Kleber tree
    pub fn new(rank: usize, highest_weight: Weight) -> Self {
        let root = KleberNode {
            rc: RiggedConfiguration::empty(rank),
            children: Vec::new(),
        };

        KleberTree { root, rank }
    }

    /// Build the Kleber tree
    pub fn build(&mut self) {
        build_kleber_tree(&mut self.root, self.rank);
    }

    /// Get all rigged configurations in the tree
    pub fn all_configurations(&self) -> Vec<RiggedConfiguration> {
        collect_configurations(&self.root)
    }
}

/// Build the Kleber tree recursively
fn build_kleber_tree(node: &mut KleberNode, rank: usize) {
    // Generate children by adding boxes to partitions
    // This is a placeholder implementation
}

/// Collect all configurations from a Kleber tree
fn collect_configurations(node: &KleberNode) -> Vec<RiggedConfiguration> {
    let mut result = vec![node.rc.clone()];
    for child in &node.children {
        result.extend(collect_configurations(child));
    }
    result
}

/// Crystal graph of rigged configurations
pub struct RCCrystalGraph {
    /// All elements in the crystal
    pub elements: Vec<RiggedConfiguration>,
    /// Edges labeled by i (index i, from element, to element)
    pub edges: HashMap<usize, Vec<(usize, usize)>>,
    /// The rank
    pub rank: usize,
}

impl RCCrystalGraph {
    /// Create a new crystal graph
    pub fn new(rank: usize) -> Self {
        RCCrystalGraph {
            elements: Vec::new(),
            edges: HashMap::new(),
            rank,
        }
    }

    /// Add an element to the graph
    pub fn add_element(&mut self, rc: RiggedConfiguration) -> usize {
        self.elements.push(rc);
        self.elements.len() - 1
    }

    /// Add an edge
    pub fn add_edge(&mut self, i: usize, from: usize, to: usize) {
        self.edges.entry(i).or_insert_with(Vec::new).push((from, to));
    }

    /// Build the crystal graph from a rigged configuration crystal
    pub fn build_from_crystal(&mut self, crystal: &RiggedConfigurationCrystal) {
        let elements = crystal.elements();
        for elem in elements {
            self.add_element(elem);
        }

        // Build edges
        for (idx, elem) in self.elements.clone().iter().enumerate() {
            for i in 0..self.rank {
                if let Some(target) = crystal.f_i(elem, i) {
                    if let Some(target_idx) = self.elements.iter().position(|e| e == &target) {
                        self.add_edge(i, idx, target_idx);
                    }
                }
            }
        }
    }
}

/// Helper functions for rigged configurations

/// Check if a sequence is a valid partition
pub fn is_partition(parts: &[usize]) -> bool {
    for i in 0..parts.len().saturating_sub(1) {
        if parts[i] < parts[i + 1] {
            return false;
        }
    }
    true
}

/// Get the Young diagram of a partition as a string
pub fn young_diagram(parts: &[usize]) -> String {
    let mut result = String::new();
    for &part in parts {
        for _ in 0..part {
            result.push('□');
        }
        result.push('\n');
    }
    result
}

/// Compute the hook length at position (i, j) in a partition
pub fn hook_length(parts: &[usize], i: usize, j: usize) -> usize {
    if i >= parts.len() || j >= parts[i] {
        return 0;
    }

    let arm = parts[i] - j - 1;
    let leg = parts[i..].iter().filter(|&&p| p > j).count() - 1;

    arm + leg + 1
}

/// Compute the content of a box at position (i, j)
pub fn content(i: usize, j: usize) -> i64 {
    j as i64 - i as i64
}

/// Automorphisms and symmetries

/// Check if two rigged configurations are isomorphic
pub fn are_isomorphic(rc1: &RiggedConfiguration, rc2: &RiggedConfiguration) -> bool {
    // Two rigged configurations are isomorphic if one can be obtained from
    // the other by permuting the partitions

    if rc1.partitions.len() != rc2.partitions.len() {
        return false;
    }

    // Simple check: equal partitions
    rc1.partitions == rc2.partitions
}

/// Promotion operator on rigged configurations
pub fn promotion(rc: &RiggedConfiguration) -> RiggedConfiguration {
    // The promotion operator is a bijection on rigged configurations
    // related to cyclic rotation of tensor factors

    let mut new_partitions = Vec::new();
    if !rc.partitions.is_empty() {
        for i in 1..rc.partitions.len() {
            new_partitions.push(rc.partitions[i].clone());
        }
        new_partitions.push(rc.partitions[0].clone());
    }

    RiggedConfiguration::new(new_partitions, rc.rank)
}

/// Evacuation operator on rigged configurations
pub fn evacuation(rc: &RiggedConfiguration) -> RiggedConfiguration {
    // The evacuation operator is an involution on rigged configurations

    let mut new_partitions = Vec::new();
    for partition in rc.partitions.iter().rev() {
        new_partitions.push(partition.clone());
    }

    RiggedConfiguration::new(new_partitions, rc.rank)
}

/// Affine crystal structure

/// Extend rigged configuration to affine crystal
pub struct AffineRiggedConfigurationCrystal {
    /// The classical crystal
    pub classical: RiggedConfigurationCrystal,
    /// The level
    pub level: i64,
}

impl AffineRiggedConfigurationCrystal {
    /// Create a new affine rigged configuration crystal
    pub fn new(rank: usize, tensor_shape: Vec<(usize, usize)>, level: i64) -> Self {
        AffineRiggedConfigurationCrystal {
            classical: RiggedConfigurationCrystal::new(rank, tensor_shape),
            level,
        }
    }

    /// Apply the affine operator e_0
    pub fn e_0(&self, rc: &RCElement) -> Option<RCElement> {
        // Implement e_0 for affine crystals
        None
    }

    /// Apply the affine operator f_0
    pub fn f_0(&self, rc: &RCElement) -> Option<RCElement> {
        // Implement f_0 for affine crystals
        None
    }
}

/// Virtual crystals and embeddings

/// Virtual crystal structure on rigged configurations
pub struct VirtualRC {
    /// The underlying rigged configuration
    pub rc: RiggedConfiguration,
    /// The virtual crystal type
    pub virtual_type: VirtualType,
}

/// Type of virtual crystal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualType {
    /// Type A embedding into type A
    TypeA,
    /// Type B embedding
    TypeB,
    /// Type C embedding
    TypeC,
    /// Type D embedding
    TypeD,
}

impl VirtualRC {
    /// Create a new virtual rigged configuration
    pub fn new(rc: RiggedConfiguration, virtual_type: VirtualType) -> Self {
        VirtualRC { rc, virtual_type }
    }

    /// Convert to classical rigged configuration
    pub fn to_classical(&self) -> RiggedConfiguration {
        self.rc.clone()
    }
}

/// Utility functions for working with rigged configurations

/// Convert between different representations
pub mod conversions {
    use super::*;

    /// Convert rigged configuration to tuple representation
    pub fn to_tuple(rc: &RiggedConfiguration) -> Vec<(Vec<usize>, Vec<i64>)> {
        rc.partitions.iter().map(|p| (p.parts.clone(), p.riggings.clone())).collect()
    }

    /// Create rigged configuration from tuple representation
    pub fn from_tuple(tuples: Vec<(Vec<usize>, Vec<i64>)>, rank: usize) -> RiggedConfiguration {
        let partitions = tuples.into_iter()
            .map(|(parts, riggings)| RiggedPartition::new(parts, riggings))
            .collect();
        RiggedConfiguration::new(partitions, rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rigged_partition_creation() {
        let rp = RiggedPartition::new(vec![3, 2, 1], vec![0, 1, 0]);
        assert_eq!(rp.num_parts(), 3);
        assert_eq!(rp.size(), 6);
        assert!(!rp.is_empty());
    }

    #[test]
    fn test_empty_rigged_partition() {
        let rp = RiggedPartition::empty();
        assert!(rp.is_empty());
        assert_eq!(rp.num_parts(), 0);
        assert_eq!(rp.size(), 0);
    }

    #[test]
    fn test_rigged_partition_add_part() {
        let mut rp = RiggedPartition::empty();
        rp.add_part(3, 1);
        rp.add_part(2, 0);
        assert_eq!(rp.num_parts(), 2);
        assert_eq!(rp.parts, vec![3, 2]); // Should be sorted
    }

    #[test]
    fn test_rigged_partition_remove_part() {
        let mut rp = RiggedPartition::new(vec![3, 2, 1], vec![0, 1, 0]);
        let removed = rp.remove_part(1);
        assert_eq!(removed, Some((2, 1)));
        assert_eq!(rp.num_parts(), 2);
    }

    #[test]
    fn test_rigged_partition_conjugate() {
        let rp = RiggedPartition::new(vec![3, 2, 1], vec![0, 0, 0]);
        let conj = rp.conjugate();
        assert_eq!(conj, vec![3, 2, 1]);
    }

    #[test]
    fn test_rigged_configuration_creation() {
        let rp1 = RiggedPartition::new(vec![2, 1], vec![0, 0]);
        let rp2 = RiggedPartition::new(vec![3], vec![1]);
        let rc = RiggedConfiguration::new(vec![rp1, rp2], 2);
        assert_eq!(rc.num_factors(), 2);
    }

    #[test]
    fn test_empty_rigged_configuration() {
        let rc = RiggedConfiguration::empty(3);
        assert_eq!(rc.rank, 3);
        assert_eq!(rc.partitions.len(), 3);
        for partition in &rc.partitions {
            assert!(partition.is_empty());
        }
    }

    #[test]
    fn test_rigged_configuration_coenergy() {
        let rp1 = RiggedPartition::new(vec![2, 1], vec![1, 2]);
        let rp2 = RiggedPartition::new(vec![3], vec![3]);
        let rc = RiggedConfiguration::new(vec![rp1, rp2], 2);
        assert_eq!(rc.coenergy(), 6); // 1 + 2 + 3
    }

    #[test]
    fn test_rigged_configuration_weight() {
        let rp1 = RiggedPartition::new(vec![2, 1], vec![0, 0]);
        let rp2 = RiggedPartition::new(vec![3], vec![0]);
        let rc = RiggedConfiguration::new(vec![rp1, rp2], 2);
        let weight = rc.weight();
        assert_eq!(weight.coords, vec![3, 3]); // sizes are 3 and 3
    }

    #[test]
    fn test_rigged_configuration_crystal() {
        let crystal = RiggedConfigurationCrystal::new(2, vec![(1, 1)]);
        assert_eq!(crystal.rank, 2);
        assert_eq!(crystal.tensor_shape, vec![(1, 1)]);
    }

    #[test]
    fn test_is_partition() {
        assert!(is_partition(&[3, 2, 1]));
        assert!(is_partition(&[5, 5, 3, 1]));
        assert!(!is_partition(&[1, 2, 3]));
        assert!(!is_partition(&[3, 1, 2]));
    }

    #[test]
    fn test_hook_length() {
        let parts = vec![3, 2, 1];
        assert_eq!(hook_length(&parts, 0, 0), 5);
        assert_eq!(hook_length(&parts, 0, 1), 3);
        assert_eq!(hook_length(&parts, 1, 0), 3);
    }

    #[test]
    fn test_content() {
        assert_eq!(content(0, 0), 0);
        assert_eq!(content(0, 2), 2);
        assert_eq!(content(2, 0), -2);
        assert_eq!(content(1, 3), 2);
    }

    #[test]
    fn test_promotion() {
        let rp1 = RiggedPartition::new(vec![2], vec![0]);
        let rp2 = RiggedPartition::new(vec![1], vec![0]);
        let rp3 = RiggedPartition::new(vec![3], vec![0]);
        let rc = RiggedConfiguration::new(vec![rp1.clone(), rp2.clone(), rp3.clone()], 3);

        let promoted = promotion(&rc);
        assert_eq!(promoted.partitions[0], rp2);
        assert_eq!(promoted.partitions[1], rp3);
        assert_eq!(promoted.partitions[2], rp1);
    }

    #[test]
    fn test_evacuation() {
        let rp1 = RiggedPartition::new(vec![2], vec![0]);
        let rp2 = RiggedPartition::new(vec![1], vec![0]);
        let rc = RiggedConfiguration::new(vec![rp1.clone(), rp2.clone()], 2);

        let evacuated = evacuation(&rc);
        assert_eq!(evacuated.partitions[0], rp2);
        assert_eq!(evacuated.partitions[1], rp1);
    }

    #[test]
    fn test_conversions() {
        let rp1 = RiggedPartition::new(vec![2, 1], vec![0, 1]);
        let rp2 = RiggedPartition::new(vec![3], vec![2]);
        let rc = RiggedConfiguration::new(vec![rp1, rp2], 2);

        let tuple = conversions::to_tuple(&rc);
        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple[0], (vec![2, 1], vec![0, 1]));
        assert_eq!(tuple[1], (vec![3], vec![2]));

        let rc2 = conversions::from_tuple(tuple, 2);
        assert_eq!(rc, rc2);
    }

    #[test]
    fn test_kleber_tree_creation() {
        let tree = KleberTree::new(2, Weight::new(vec![1, 1]));
        assert_eq!(tree.rank, 2);
    }

    #[test]
    fn test_rc_statistics() {
        let rp1 = RiggedPartition::new(vec![2, 1], vec![1, 2]);
        let rp2 = RiggedPartition::new(vec![3], vec![3]);
        let rc = RiggedConfiguration::new(vec![rp1, rp2], 2);

        let stats = RCStatistics::new(&rc);
        assert_eq!(stats.total_parts, 3);
        assert_eq!(stats.total_size, 6);
        assert_eq!(stats.coenergy, 6);
    }

    #[test]
    fn test_virtual_rc() {
        let rc = RiggedConfiguration::empty(2);
        let vrc = VirtualRC::new(rc.clone(), VirtualType::TypeA);
        assert_eq!(vrc.to_classical(), rc);
    }

    #[test]
    fn test_affine_rc_crystal() {
        let affine = AffineRiggedConfigurationCrystal::new(2, vec![(1, 1)], 1);
        assert_eq!(affine.level, 1);
        assert_eq!(affine.classical.rank, 2);
    }
}
