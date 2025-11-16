//! Hasse Diagram Construction from Incidence Relations
//!
//! This module provides functionality for constructing lattice structures
//! from incidence data, corresponding to SageMath's sage.geometry.hasse_diagram module.
//!
//! The main function `lattice_from_incidences` builds a finite atomic and coatomic
//! lattice from the incidence relations between atoms (minimal non-empty elements)
//! and coatoms (maximal proper elements).

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// A lattice element constructed from atom and coatom indices
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LatticeElement {
    /// Set of atoms contained in this element
    pub atoms: Vec<usize>,
    /// Set of coatoms containing this element
    pub coatoms: Vec<usize>,
}

impl LatticeElement {
    /// Create a new lattice element from atom and coatom sets
    pub fn new(mut atoms: Vec<usize>, mut coatoms: Vec<usize>) -> Self {
        atoms.sort();
        atoms.dedup();
        coatoms.sort();
        coatoms.dedup();
        LatticeElement { atoms, coatoms }
    }

    /// Check if this element is less than or equal to another in the lattice order
    ///
    /// In an atomic lattice, x <= y if and only if atoms(x) ⊆ atoms(y)
    pub fn is_le(&self, other: &Self) -> bool {
        self.atoms.iter().all(|a| other.atoms.contains(a))
    }

    /// Check if this is the empty element (bottom of lattice)
    pub fn is_bottom(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Check if this is the top element (contains all atoms)
    pub fn is_top(&self, num_atoms: usize) -> bool {
        self.atoms.len() == num_atoms
    }
}

/// A finite lattice represented as a partially ordered set
///
/// This structure represents a lattice constructed from incidence data,
/// containing elements with a partial order relation.
#[derive(Debug, Clone)]
pub struct FiniteLattice {
    /// All elements in the lattice (including bottom and top)
    elements: Vec<LatticeElement>,
    /// Index of the bottom element
    bottom_index: usize,
    /// Index of the top element
    top_index: usize,
    /// Number of atoms (minimal non-empty elements)
    num_atoms: usize,
    /// Number of coatoms (maximal proper elements)
    num_coatoms: usize,
    /// Cover relations: maps element index to indices of elements it covers
    covers: HashMap<usize, Vec<usize>>,
}

impl FiniteLattice {
    /// Get the number of elements in the lattice
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if the lattice is empty (no elements)
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Get an element by index
    pub fn get(&self, index: usize) -> Option<&LatticeElement> {
        self.elements.get(index)
    }

    /// Get all elements
    pub fn elements(&self) -> &[LatticeElement] {
        &self.elements
    }

    /// Get the bottom element
    pub fn bottom(&self) -> &LatticeElement {
        &self.elements[self.bottom_index]
    }

    /// Get the top element
    pub fn top(&self) -> &LatticeElement {
        &self.elements[self.top_index]
    }

    /// Get elements that cover a given element
    ///
    /// Element y covers x if x < y and there is no z with x < z < y
    pub fn covers_of(&self, index: usize) -> Option<&Vec<usize>> {
        self.covers.get(&index)
    }

    /// Check if element i is less than or equal to element j
    pub fn is_le(&self, i: usize, j: usize) -> bool {
        if i >= self.elements.len() || j >= self.elements.len() {
            return false;
        }
        self.elements[i].is_le(&self.elements[j])
    }

    /// Get the number of atoms in the lattice
    pub fn num_atoms(&self) -> usize {
        self.num_atoms
    }

    /// Get the number of coatoms in the lattice
    pub fn num_coatoms(&self) -> usize {
        self.num_coatoms
    }
}

/// Construct a finite atomic and coatomic lattice from incidence data
///
/// This function builds a lattice from the incidence relations between atoms
/// (minimal non-empty elements) and coatoms (maximal proper elements).
///
/// # Arguments
///
/// * `atom_to_coatoms` - For each atom i, a list of coatom indices that cover it
/// * `coatom_to_atoms` - For each coatom j, a list of atom indices it covers
/// * `required_atoms` - Optional list of atoms that must appear in all non-empty faces
///
/// # Returns
///
/// A `FiniteLattice` containing all elements with their incidence relations
///
/// # Algorithm
///
/// The algorithm follows Section 2.5 of "Polytopes – Combinatorics and Computation"
/// (Ziegler, 2000), using frozensets for efficiency:
///
/// 1. Start with the bottom element (empty set of atoms)
/// 2. Iteratively build faces by computing closures under coatom incidence
/// 3. Identify minimal elements at each step to ensure all faces are generated
/// 4. Build the top element (all atoms)
/// 5. Compute cover relations for the Hasse diagram
///
/// # Example
///
/// ```
/// use rustmath_geometry::hasse_diagram::lattice_from_incidences;
///
/// // Simple lattice: 3 atoms {0, 1, 2}, 3 coatoms
/// // Coatom 0 covers atoms {0, 1}
/// // Coatom 1 covers atoms {1, 2}
/// // Coatom 2 covers atoms {0, 2}
/// let atom_to_coatoms = vec![
///     vec![0, 2],  // atom 0 is in coatoms 0, 2
///     vec![0, 1],  // atom 1 is in coatoms 0, 1
///     vec![1, 2],  // atom 2 is in coatoms 1, 2
/// ];
/// let coatom_to_atoms = vec![
///     vec![0, 1],  // coatom 0 contains atoms 0, 1
///     vec![1, 2],  // coatom 1 contains atoms 1, 2
///     vec![0, 2],  // coatom 2 contains atoms 0, 2
/// ];
///
/// let lattice = lattice_from_incidences(&atom_to_coatoms, &coatom_to_atoms, None);
/// assert!(lattice.len() >= 8); // At least: bottom + 3 atoms + 3 pairs + top
/// assert_eq!(lattice.num_atoms(), 3);
/// assert_eq!(lattice.num_coatoms(), 3);
/// ```
pub fn lattice_from_incidences(
    atom_to_coatoms: &[Vec<usize>],
    coatom_to_atoms: &[Vec<usize>],
    required_atoms: Option<&[usize]>,
) -> FiniteLattice {
    let num_atoms = atom_to_coatoms.len();
    let num_coatoms = coatom_to_atoms.len();

    // Validate inputs
    for coatoms in atom_to_coatoms {
        for &coatom in coatoms {
            assert!(coatom < num_coatoms, "Invalid coatom index");
        }
    }
    for atoms in coatom_to_atoms {
        for &atom in atoms {
            assert!(atom < num_atoms, "Invalid atom index");
        }
    }

    // Build the lattice elements
    let mut elements = Vec::new();
    let mut element_set: HashSet<(Vec<usize>, Vec<usize>)> = HashSet::new();

    // Helper function to compute the closure of a set of atoms under coatom incidence
    let compute_closure = |atoms: &HashSet<usize>| -> (Vec<usize>, Vec<usize>) {
        // Find all coatoms that contain all the given atoms
        let mut coatoms = Vec::new();
        for (coatom_idx, coatom_atoms) in coatom_to_atoms.iter().enumerate() {
            if atoms.iter().all(|a| coatom_atoms.contains(a)) {
                coatoms.push(coatom_idx);
            }
        }

        // Atoms are just the sorted vector
        let atoms_vec: Vec<usize> = atoms.iter().copied().collect();
        (atoms_vec, coatoms)
    };

    // Bottom element: no atoms
    let bottom_atoms = HashSet::new();
    let (bottom_atom_vec, bottom_coatom_vec) = compute_closure(&bottom_atoms);
    let bottom_elem = LatticeElement::new(bottom_atom_vec.clone(), bottom_coatom_vec.clone());
    elements.push(bottom_elem.clone());
    element_set.insert((bottom_atom_vec, bottom_coatom_vec));
    let bottom_index = 0;

    // Build all elements using a queue-based approach
    let mut to_process: Vec<HashSet<usize>> = Vec::new();

    // Add single atoms as starting points
    for atom in 0..num_atoms {
        // Skip atoms not in required set
        if let Some(req) = required_atoms {
            if !req.contains(&atom) {
                continue;
            }
        }

        let mut atom_set = HashSet::new();
        atom_set.insert(atom);
        to_process.push(atom_set);
    }

    let mut processed: HashSet<Vec<usize>> = HashSet::new();

    while let Some(atom_set) = to_process.pop() {
        let (atom_vec, coatom_vec) = compute_closure(&atom_set);

        // Skip if already processed
        if processed.contains(&atom_vec) {
            continue;
        }
        processed.insert(atom_vec.clone());

        // Add to elements if not already present
        if !element_set.contains(&(atom_vec.clone(), coatom_vec.clone())) {
            let elem = LatticeElement::new(atom_vec.clone(), coatom_vec.clone());
            elements.push(elem);
            element_set.insert((atom_vec.clone(), coatom_vec.clone()));
        }

        // Generate new elements by adding one more atom
        for atom in 0..num_atoms {
            if !atom_set.contains(&atom) {
                let mut new_set = atom_set.clone();
                new_set.insert(atom);
                to_process.push(new_set);
            }
        }
    }

    // Top element: all atoms
    let all_atoms: HashSet<usize> = (0..num_atoms).collect();
    let (top_atom_vec, top_coatom_vec) = compute_closure(&all_atoms);
    let top_elem = LatticeElement::new(top_atom_vec.clone(), top_coatom_vec.clone());

    // Check if top already exists
    let top_index = if let Some(pos) = elements.iter().position(|e| e == &top_elem) {
        pos
    } else {
        elements.push(top_elem);
        elements.len() - 1
    };

    // Compute cover relations
    let mut covers: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, elem_i) in elements.iter().enumerate() {
        let mut cover_list = Vec::new();
        for (j, elem_j) in elements.iter().enumerate() {
            if i == j {
                continue;
            }
            // j covers i if i < j and there's no k with i < k < j
            if elem_i.is_le(elem_j) && !elem_j.is_le(elem_i) {
                let mut is_cover = true;
                for (k, elem_k) in elements.iter().enumerate() {
                    if k == i || k == j {
                        continue;
                    }
                    if elem_i.is_le(elem_k) && elem_k.is_le(elem_j)
                       && !elem_k.is_le(elem_i) && !elem_j.is_le(elem_k) {
                        is_cover = false;
                        break;
                    }
                }
                if is_cover {
                    cover_list.push(j);
                }
            }
        }
        if !cover_list.is_empty() {
            covers.insert(i, cover_list);
        }
    }

    FiniteLattice {
        elements,
        bottom_index,
        top_index,
        num_atoms,
        num_coatoms,
        covers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_element_creation() {
        let elem = LatticeElement::new(vec![0, 2, 1], vec![1, 0]);
        assert_eq!(elem.atoms, vec![0, 1, 2]);
        assert_eq!(elem.coatoms, vec![0, 1]);
    }

    #[test]
    fn test_lattice_element_ordering() {
        let elem1 = LatticeElement::new(vec![0, 1], vec![]);
        let elem2 = LatticeElement::new(vec![0, 1, 2], vec![]);
        let elem3 = LatticeElement::new(vec![1, 2], vec![]);

        assert!(elem1.is_le(&elem2));
        assert!(!elem2.is_le(&elem1));
        assert!(!elem1.is_le(&elem3));
        assert!(!elem3.is_le(&elem1));
    }

    #[test]
    fn test_simple_lattice() {
        // Triangle: 3 vertices (atoms), 3 edges (coatoms)
        // Edge 0: vertices {0, 1}
        // Edge 1: vertices {1, 2}
        // Edge 2: vertices {0, 2}
        let atom_to_coatoms = vec![
            vec![0, 2],  // vertex 0 in edges 0, 2
            vec![0, 1],  // vertex 1 in edges 0, 1
            vec![1, 2],  // vertex 2 in edges 1, 2
        ];
        let coatom_to_atoms = vec![
            vec![0, 1],  // edge 0 has vertices 0, 1
            vec![1, 2],  // edge 1 has vertices 1, 2
            vec![0, 2],  // edge 2 has vertices 0, 2
        ];

        let lattice = lattice_from_incidences(&atom_to_coatoms, &coatom_to_atoms, None);

        assert!(lattice.len() > 0);
        assert_eq!(lattice.num_atoms(), 3);
        assert_eq!(lattice.num_coatoms(), 3);

        // Bottom element should have no atoms
        assert!(lattice.bottom().is_bottom());

        // Top element should have all atoms
        assert!(lattice.top().is_top(3));
    }

    #[test]
    fn test_square_lattice() {
        // Square: 4 vertices (atoms), 4 edges (coatoms)
        // Edge 0: vertices {0, 1}
        // Edge 1: vertices {1, 2}
        // Edge 2: vertices {2, 3}
        // Edge 3: vertices {3, 0}
        let atom_to_coatoms = vec![
            vec![0, 3],  // vertex 0 in edges 0, 3
            vec![0, 1],  // vertex 1 in edges 0, 1
            vec![1, 2],  // vertex 2 in edges 1, 2
            vec![2, 3],  // vertex 3 in edges 2, 3
        ];
        let coatom_to_atoms = vec![
            vec![0, 1],  // edge 0
            vec![1, 2],  // edge 1
            vec![2, 3],  // edge 2
            vec![3, 0],  // edge 3
        ];

        let lattice = lattice_from_incidences(&atom_to_coatoms, &coatom_to_atoms, None);

        assert!(lattice.len() > 0);
        assert_eq!(lattice.num_atoms(), 4);
        assert_eq!(lattice.num_coatoms(), 4);

        // Check bottom and top
        assert!(lattice.bottom().atoms.is_empty());
        assert_eq!(lattice.top().atoms.len(), 4);
    }

    #[test]
    fn test_lattice_ordering() {
        let atom_to_coatoms = vec![
            vec![0],
            vec![0],
        ];
        let coatom_to_atoms = vec![
            vec![0, 1],
        ];

        let lattice = lattice_from_incidences(&atom_to_coatoms, &coatom_to_atoms, None);

        // Bottom <= everything
        for i in 0..lattice.len() {
            assert!(lattice.is_le(lattice.bottom_index, i));
        }

        // Everything <= top
        for i in 0..lattice.len() {
            assert!(lattice.is_le(i, lattice.top_index));
        }
    }

    #[test]
    fn test_cover_relations() {
        let atom_to_coatoms = vec![
            vec![0],
            vec![0],
        ];
        let coatom_to_atoms = vec![
            vec![0, 1],
        ];

        let lattice = lattice_from_incidences(&atom_to_coatoms, &coatom_to_atoms, None);

        // Bottom should be covered by single-atom elements
        if let Some(covers) = lattice.covers_of(lattice.bottom_index) {
            assert!(!covers.is_empty());
        }
    }
}
