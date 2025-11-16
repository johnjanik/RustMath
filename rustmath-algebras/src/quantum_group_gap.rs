//! Quantum Groups (GAP Interface)
//!
//! This module implements Drinfel'd-Jimbo quantum groups using interfaces similar
//! to GAP's QuaGroup package. It provides quantum group representations and their
//! crystal bases.
//!
//! # Mathematical Background
//!
//! A quantum group U_q(g) is a q-deformation of the universal enveloping algebra
//! of a Lie algebra g. It has generators:
//! - E_i: Raising operators
//! - F_i: Lowering operators
//! - K_i: Diagonal elements (q-Cartan subalgebra)
//!
//! These satisfy the quantum Serre relations and form a Hopf algebra.
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::quantum_group_gap::*;
//!
//! // Create a quantum group for sl_2
//! let qg = QuantumGroup::new("A1".to_string(), "q".to_string());
//!
//! // Create a highest weight module
//! let module = HighestWeightModule::new(&qg, vec![1]);
//! ```

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt;

/// Crystal graph vertex
///
/// Represents a vertex in the crystal graph of a quantum group module.
/// Each vertex corresponds to a weight vector in the module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrystalGraphVertex {
    /// Weight of the vertex (list of integers)
    weight: Vec<i32>,
    /// String representation
    label: String,
}

impl CrystalGraphVertex {
    /// Create a new crystal graph vertex
    pub fn new(weight: Vec<i32>, label: String) -> Self {
        Self { weight, label }
    }

    /// Get the weight
    pub fn weight(&self) -> &[i32] {
        &self.weight
    }

    /// Get the label
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl fmt::Display for CrystalGraphVertex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (wt: {:?})", self.label, self.weight)
    }
}

/// Element of a quantum group module
///
/// Base class for elements in quantum group representations.
#[derive(Debug, Clone)]
pub struct QuaGroupModuleElement<R: Ring> {
    /// Coefficients in the module basis
    coefficients: HashMap<usize, R>,
    /// Parent module identifier
    module_id: String,
}

impl<R: Ring> QuaGroupModuleElement<R> {
    /// Create a zero element
    pub fn zero(module_id: String) -> Self {
        Self {
            coefficients: HashMap::new(),
            module_id,
        }
    }

    /// Create an element from coefficients
    pub fn new(coefficients: HashMap<usize, R>, module_id: String) -> Self {
        Self {
            coefficients,
            module_id,
        }
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &HashMap<usize, R> {
        &self.coefficients
    }

    /// Apply Kashiwara operator e_i (raising)
    pub fn e_tilde(&self, _i: usize) -> Self {
        // Placeholder: would implement crystal operator
        Self::zero(self.module_id.clone())
    }

    /// Apply Kashiwara operator f_i (lowering)
    pub fn f_tilde(&self, _i: usize) -> Self {
        // Placeholder: would implement crystal operator
        Self::zero(self.module_id.clone())
    }

    /// Bar involution
    pub fn bar(&self) -> Self {
        // Placeholder: would implement bar involution
        self.clone()
    }
}

impl<R: Ring> fmt::Display for QuaGroupModuleElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        write!(f, "Element in {}", self.module_id)
    }
}

/// Element of a quantum group representation
///
/// Specialized element type for quantum group representations with crystal structure.
#[derive(Debug, Clone)]
pub struct QuaGroupRepresentationElement<R: Ring> {
    /// Base element
    base: QuaGroupModuleElement<R>,
    /// Crystal vertex information
    crystal_vertex: Option<CrystalGraphVertex>,
}

impl<R: Ring> QuaGroupRepresentationElement<R> {
    /// Create a new representation element
    pub fn new(
        coefficients: HashMap<usize, R>,
        module_id: String,
        crystal_vertex: Option<CrystalGraphVertex>,
    ) -> Self {
        Self {
            base: QuaGroupModuleElement::new(coefficients, module_id),
            crystal_vertex,
        }
    }

    /// Get the base element
    pub fn base(&self) -> &QuaGroupModuleElement<R> {
        &self.base
    }

    /// Get the crystal vertex
    pub fn crystal_vertex(&self) -> Option<&CrystalGraphVertex> {
        self.crystal_vertex.as_ref()
    }
}

/// Quantum Group
///
/// Represents the quantum group U_q(g) for a given Cartan type.
#[derive(Debug, Clone)]
pub struct QuantumGroup {
    /// Cartan type (e.g., "A2", "B3", etc.)
    cartan_type: String,
    /// Quantum parameter q (as a string for symbolic computation)
    q_param: String,
    /// Rank of the Lie algebra
    rank: usize,
}

impl QuantumGroup {
    /// Create a new quantum group
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type (e.g., "A2", "B3")
    /// * `q_param` - The quantum parameter as a string
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::quantum_group_gap::QuantumGroup;
    ///
    /// let qg = QuantumGroup::new("A2".to_string(), "q".to_string());
    /// assert_eq!(qg.rank(), 2);
    /// ```
    pub fn new(cartan_type: String, q_param: String) -> Self {
        // Parse rank from Cartan type (simplified)
        let rank = cartan_type
            .chars()
            .skip(1)
            .collect::<String>()
            .parse::<usize>()
            .unwrap_or(1);

        Self {
            cartan_type,
            q_param,
            rank,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &str {
        &self.cartan_type
    }

    /// Get the quantum parameter
    pub fn q_param(&self) -> &str {
        &self.q_param
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get raising operator E_i
    pub fn e(&self, i: usize) -> String {
        assert!(i < self.rank, "Index out of range");
        format!("E_{}", i)
    }

    /// Get lowering operator F_i
    pub fn f(&self, i: usize) -> String {
        assert!(i < self.rank, "Index out of range");
        format!("F_{}", i)
    }

    /// Get K operator K_i
    pub fn k(&self, i: usize) -> String {
        assert!(i < self.rank, "Index out of range");
        format!("K_{}", i)
    }

    /// Get all generators
    pub fn generators(&self) -> Vec<String> {
        let mut gens = Vec::new();
        for i in 0..self.rank {
            gens.push(self.e(i));
            gens.push(self.f(i));
            gens.push(self.k(i));
        }
        gens
    }
}

impl fmt::Display for QuantumGroup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Quantum group U_{{{}}}({}) with rank {}",
            self.q_param, self.cartan_type, self.rank
        )
    }
}

/// Quantum group module
///
/// Base class for modules over a quantum group.
#[derive(Debug, Clone)]
pub struct QuantumGroupModule {
    /// Associated quantum group
    quantum_group: QuantumGroup,
    /// Module identifier
    id: String,
    /// Dimension (if finite-dimensional)
    dimension: Option<usize>,
}

impl QuantumGroupModule {
    /// Create a new quantum group module
    pub fn new(quantum_group: QuantumGroup, id: String, dimension: Option<usize>) -> Self {
        Self {
            quantum_group,
            id,
            dimension,
        }
    }

    /// Get the quantum group
    pub fn quantum_group(&self) -> &QuantumGroup {
        &self.quantum_group
    }

    /// Get the module ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the dimension
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Check if finite-dimensional
    pub fn is_finite_dimensional(&self) -> bool {
        self.dimension.is_some()
    }
}

/// Highest weight module
///
/// A finite-dimensional irreducible representation of the quantum group
/// characterized by its highest weight.
#[derive(Debug, Clone)]
pub struct HighestWeightModule {
    /// Base module
    base: QuantumGroupModule,
    /// Highest weight (fundamental weights)
    highest_weight: Vec<i32>,
}

impl HighestWeightModule {
    /// Create a new highest weight module
    ///
    /// # Arguments
    ///
    /// * `quantum_group` - The quantum group
    /// * `highest_weight` - The highest weight as a vector
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::quantum_group_gap::*;
    ///
    /// let qg = QuantumGroup::new("A2".to_string(), "q".to_string());
    /// let module = HighestWeightModule::new(&qg, vec![1, 0]);
    /// ```
    pub fn new(quantum_group: &QuantumGroup, highest_weight: Vec<i32>) -> Self {
        assert_eq!(
            highest_weight.len(),
            quantum_group.rank(),
            "Weight length must match rank"
        );

        let id = format!("V({:?})", highest_weight);
        // Dimension would be computed from Weyl dimension formula
        let dimension = Some(Self::compute_dimension(&highest_weight));

        Self {
            base: QuantumGroupModule::new(quantum_group.clone(), id, dimension),
            highest_weight,
        }
    }

    /// Compute dimension using Weyl dimension formula (simplified)
    fn compute_dimension(weight: &[i32]) -> usize {
        // Placeholder: actual formula involves product over positive roots
        weight.iter().map(|&w| (w + 1) as usize).product()
    }

    /// Get the highest weight
    pub fn highest_weight(&self) -> &[i32] {
        &self.highest_weight
    }

    /// Get the base module
    pub fn base(&self) -> &QuantumGroupModule {
        &self.base
    }

    /// Get the quantum group
    pub fn quantum_group(&self) -> &QuantumGroup {
        self.base.quantum_group()
    }

    /// Get the highest weight vector
    pub fn highest_weight_vector<R: Ring>(&self) -> QuaGroupModuleElement<R>
    where
        R: From<i32>,
    {
        let mut coefficients = HashMap::new();
        coefficients.insert(0, R::from(1));
        QuaGroupModuleElement::new(coefficients, self.base.id().to_string())
    }
}

impl fmt::Display for HighestWeightModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Highest weight module V({:?}) of {}",
            self.highest_weight,
            self.base.quantum_group()
        )
    }
}

/// Highest weight submodule
///
/// An irreducible submodule of a larger module, typically appearing
/// in tensor product decompositions.
#[derive(Debug, Clone)]
pub struct HighestWeightSubmodule {
    /// The ambient module
    ambient_module_id: String,
    /// The highest weight
    highest_weight: Vec<i32>,
    /// Associated quantum group
    quantum_group: QuantumGroup,
}

impl HighestWeightSubmodule {
    /// Create a new highest weight submodule
    pub fn new(
        ambient_module_id: String,
        highest_weight: Vec<i32>,
        quantum_group: QuantumGroup,
    ) -> Self {
        Self {
            ambient_module_id,
            highest_weight,
            quantum_group,
        }
    }

    /// Get the highest weight
    pub fn highest_weight(&self) -> &[i32] {
        &self.highest_weight
    }

    /// Get the ambient module ID
    pub fn ambient_module_id(&self) -> &str {
        &self.ambient_module_id
    }
}

/// Tensor product of highest weight modules
///
/// Represents V_1 ⊗ V_2 ⊗ ... ⊗ V_n where each V_i is a highest weight module.
#[derive(Debug, Clone)]
pub struct TensorProductOfHighestWeightModules {
    /// The factors in the tensor product
    factors: Vec<HighestWeightModule>,
    /// Associated quantum group
    quantum_group: QuantumGroup,
}

impl TensorProductOfHighestWeightModules {
    /// Create a new tensor product
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::quantum_group_gap::*;
    ///
    /// let qg = QuantumGroup::new("A1".to_string(), "q".to_string());
    /// let v1 = HighestWeightModule::new(&qg, vec![1]);
    /// let v2 = HighestWeightModule::new(&qg, vec![1]);
    /// let tensor = TensorProductOfHighestWeightModules::new(vec![v1, v2]);
    /// ```
    pub fn new(factors: Vec<HighestWeightModule>) -> Self {
        assert!(!factors.is_empty(), "Tensor product must have at least one factor");

        let quantum_group = factors[0].quantum_group().clone();

        Self {
            factors,
            quantum_group,
        }
    }

    /// Get the factors
    pub fn factors(&self) -> &[HighestWeightModule] {
        &self.factors
    }

    /// Get the number of factors
    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Decompose into irreducible components
    ///
    /// Returns the highest weights of irreducible submodules (multiplicities not included)
    pub fn decompose(&self) -> Vec<Vec<i32>> {
        // Placeholder: would use Clebsch-Gordan/tensor product rules
        vec![vec![0; self.quantum_group.rank()]]
    }
}

/// Lower half quantum group
///
/// The subalgebra U_q^-(g) generated by the lowering operators F_i.
#[derive(Debug, Clone)]
pub struct LowerHalfQuantumGroup {
    /// Associated quantum group
    quantum_group: QuantumGroup,
}

impl LowerHalfQuantumGroup {
    /// Create the lower half quantum group
    pub fn new(quantum_group: QuantumGroup) -> Self {
        Self { quantum_group }
    }

    /// Get the quantum group
    pub fn quantum_group(&self) -> &QuantumGroup {
        &self.quantum_group
    }

    /// Get the lowering generators
    pub fn generators(&self) -> Vec<String> {
        (0..self.quantum_group.rank())
            .map(|i| self.quantum_group.f(i))
            .collect()
    }
}

/// Quantum group morphism
///
/// A morphism between quantum group modules.
#[derive(Debug, Clone)]
pub struct QuantumGroupMorphism {
    /// Source module ID
    source: String,
    /// Target module ID
    target: String,
    /// Description of the morphism
    description: String,
}

impl QuantumGroupMorphism {
    /// Create a new morphism
    pub fn new(source: String, target: String, description: String) -> Self {
        Self {
            source,
            target,
            description,
        }
    }

    /// Get the source
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the target
    pub fn target(&self) -> &str {
        &self.target
    }
}

/// Quantum group homset
///
/// The set of morphisms between two quantum group modules.
#[derive(Debug, Clone)]
pub struct QuantumGroupHomset {
    /// Source module
    source: String,
    /// Target module
    target: String,
}

impl QuantumGroupHomset {
    /// Create a new homset
    pub fn new(source: String, target: String) -> Self {
        Self { source, target }
    }

    /// Get the source
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the target
    pub fn target(&self) -> &str {
        &self.target
    }
}

/// Projection to lower half
///
/// Computes the projection of an element to the lower half quantum group.
pub fn projection_lower_half<R: Ring>(element: &QuaGroupModuleElement<R>) -> QuaGroupModuleElement<R> {
    // Placeholder: would compute projection
    QuaGroupModuleElement::zero(element.module_id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crystal_vertex() {
        let vertex = CrystalGraphVertex::new(vec![1, 0], "v1".to_string());
        assert_eq!(vertex.weight(), &[1, 0]);
        assert_eq!(vertex.label(), "v1");
    }

    #[test]
    fn test_quantum_group_creation() {
        let qg = QuantumGroup::new("A2".to_string(), "q".to_string());
        assert_eq!(qg.cartan_type(), "A2");
        assert_eq!(qg.q_param(), "q");
        assert_eq!(qg.rank(), 2);
    }

    #[test]
    fn test_quantum_group_generators() {
        let qg = QuantumGroup::new("A1".to_string(), "q".to_string());
        assert_eq!(qg.e(0), "E_0");
        assert_eq!(qg.f(0), "F_0");
        assert_eq!(qg.k(0), "K_0");

        let gens = qg.generators();
        assert_eq!(gens.len(), 3);
    }

    #[test]
    fn test_module_element() {
        let elem: QuaGroupModuleElement<i32> = QuaGroupModuleElement::zero("test".to_string());
        assert!(elem.is_zero());
    }

    #[test]
    fn test_highest_weight_module() {
        let qg = QuantumGroup::new("A2".to_string(), "q".to_string());
        let module = HighestWeightModule::new(&qg, vec![1, 0]);
        assert_eq!(module.highest_weight(), &[1, 0]);
        assert_eq!(module.quantum_group().rank(), 2);
    }

    #[test]
    #[should_panic(expected = "Weight length must match rank")]
    fn test_highest_weight_module_invalid_weight() {
        let qg = QuantumGroup::new("A2".to_string(), "q".to_string());
        HighestWeightModule::new(&qg, vec![1]); // Wrong length
    }

    #[test]
    fn test_highest_weight_vector() {
        let qg = QuantumGroup::new("A1".to_string(), "q".to_string());
        let module = HighestWeightModule::new(&qg, vec![1]);
        let hwv: QuaGroupModuleElement<i32> = module.highest_weight_vector();
        assert!(!hwv.is_zero());
    }

    #[test]
    fn test_tensor_product() {
        let qg = QuantumGroup::new("A1".to_string(), "q".to_string());
        let v1 = HighestWeightModule::new(&qg, vec![1]);
        let v2 = HighestWeightModule::new(&qg, vec![1]);
        let tensor = TensorProductOfHighestWeightModules::new(vec![v1, v2]);
        assert_eq!(tensor.num_factors(), 2);
    }

    #[test]
    fn test_lower_half_quantum_group() {
        let qg = QuantumGroup::new("A2".to_string(), "q".to_string());
        let lower = LowerHalfQuantumGroup::new(qg);
        let gens = lower.generators();
        assert_eq!(gens.len(), 2);
        assert_eq!(gens[0], "F_0");
        assert_eq!(gens[1], "F_1");
    }

    #[test]
    fn test_quantum_group_morphism() {
        let morphism = QuantumGroupMorphism::new(
            "V1".to_string(),
            "V2".to_string(),
            "inclusion".to_string(),
        );
        assert_eq!(morphism.source(), "V1");
        assert_eq!(morphism.target(), "V2");
    }

    #[test]
    fn test_quantum_group_homset() {
        let homset = QuantumGroupHomset::new("V1".to_string(), "V2".to_string());
        assert_eq!(homset.source(), "V1");
        assert_eq!(homset.target(), "V2");
    }

    #[test]
    fn test_highest_weight_submodule() {
        let qg = QuantumGroup::new("A1".to_string(), "q".to_string());
        let submodule = HighestWeightSubmodule::new(
            "V(2)".to_string(),
            vec![1],
            qg,
        );
        assert_eq!(submodule.highest_weight(), &[1]);
        assert_eq!(submodule.ambient_module_id(), "V(2)");
    }

    #[test]
    fn test_projection_lower_half() {
        let elem: QuaGroupModuleElement<i32> = QuaGroupModuleElement::zero("test".to_string());
        let projected = projection_lower_half(&elem);
        assert!(projected.is_zero());
    }

    #[test]
    fn test_representation_element() {
        let vertex = CrystalGraphVertex::new(vec![1, 0], "v".to_string());
        let mut coeffs = HashMap::new();
        coeffs.insert(0, 1);

        let elem = QuaGroupRepresentationElement::new(
            coeffs,
            "test".to_string(),
            Some(vertex.clone()),
        );

        assert!(elem.crystal_vertex().is_some());
        assert_eq!(elem.crystal_vertex().unwrap().weight(), &[1, 0]);
    }

    #[test]
    fn test_kashiwara_operators() {
        let elem: QuaGroupModuleElement<i32> = QuaGroupModuleElement::zero("test".to_string());
        let raised = elem.e_tilde(0);
        let lowered = elem.f_tilde(0);

        assert!(raised.is_zero());
        assert!(lowered.is_zero());
    }

    #[test]
    fn test_bar_involution() {
        let mut coeffs = HashMap::new();
        coeffs.insert(0, 1);
        let elem = QuaGroupModuleElement::new(coeffs, "test".to_string());
        let barred = elem.bar();
        assert!(!barred.is_zero());
    }

    #[test]
    fn test_quantum_group_display() {
        let qg = QuantumGroup::new("A2".to_string(), "q".to_string());
        let display = format!("{}", qg);
        assert!(display.contains("U_{q}(A2)"));
        assert!(display.contains("rank 2"));
    }

    #[test]
    fn test_highest_weight_module_display() {
        let qg = QuantumGroup::new("A1".to_string(), "q".to_string());
        let module = HighestWeightModule::new(&qg, vec![2]);
        let display = format!("{}", module);
        assert!(display.contains("V([2])"));
    }
}
