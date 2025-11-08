//! Representation theory for finite groups
//!
//! This module implements representation theory including:
//! - Linear representations of finite groups
//! - Character theory
//! - Irreducible representations
//! - Tensor products

use rustmath_core::Field;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use rustmath_complex::Complex;
use std::collections::HashMap;
use std::fmt;

/// A linear representation of a group
///
/// A representation ρ: G → GL(V) assigns a matrix to each group element
/// such that ρ(g₁g₂) = ρ(g₁)ρ(g₂)
#[derive(Clone, Debug)]
pub struct Representation<F: Field> {
    /// Dimension of the representation (degree)
    degree: usize,
    /// Map from group element labels to matrices
    matrices: HashMap<String, Matrix<F>>,
}

impl<F: Field> Representation<F> {
    /// Create a new representation
    pub fn new(degree: usize, matrices: HashMap<String, Matrix<F>>) -> Result<Self, String> {
        // Validate all matrices are square and correct dimension
        for (label, mat) in &matrices {
            if mat.rows() != degree || mat.cols() != degree {
                return Err(format!(
                    "Matrix for {} has dimension {}×{}, expected {}×{}",
                    label,
                    mat.rows(),
                    mat.cols(),
                    degree,
                    degree
                ));
            }
        }

        Ok(Representation { degree, matrices })
    }

    /// Get the degree (dimension) of the representation
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the matrix for a group element
    pub fn matrix(&self, element: &str) -> Option<&Matrix<F>> {
        self.matrices.get(element)
    }

    /// Get all group elements
    pub fn elements(&self) -> Vec<&String> {
        self.matrices.keys().collect()
    }

    /// Check if this is the trivial representation (all identity matrices)
    pub fn is_trivial(&self) -> bool {
        if self.degree != 1 {
            return false;
        }

        for mat in self.matrices.values() {
            if let Ok(elem) = mat.get(0, 0) {
                if *elem != F::one() {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Compute the trace of a representation element
    pub fn trace(&self, element: &str) -> Option<F> {
        self.matrices.get(element).and_then(|mat| mat.trace().ok())
    }
}

/// A character of a representation
///
/// The character χ(g) = tr(ρ(g)) is the trace of the representation matrix
#[derive(Clone, Debug)]
pub struct Character {
    /// Degree of the representation
    degree: usize,
    /// Map from conjugacy class representatives to character values
    values: HashMap<String, Complex>,
}

impl Character {
    /// Create a new character
    pub fn new(degree: usize, values: HashMap<String, Complex>) -> Self {
        Character { degree, values }
    }

    /// Create a character from a representation over rationals
    pub fn from_representation_rational(rep: &Representation<Rational>) -> Self {
        let mut values = HashMap::new();

        for (element, _) in &rep.matrices {
            if let Some(trace) = rep.trace(element) {
                // Convert rational to complex
                let trace_f64 = trace.numerator().to_string().parse::<f64>().unwrap_or(0.0)
                    / trace.denominator().to_string().parse::<f64>().unwrap_or(1.0);
                let trace_complex = Complex::new(trace_f64, 0.0);
                values.insert(element.clone(), trace_complex);
            }
        }

        Character {
            degree: rep.degree(),
            values,
        }
    }

    /// Get the degree of the character
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the character value for an element
    pub fn value(&self, element: &str) -> Option<&Complex> {
        self.values.get(element)
    }

    /// Get all elements
    pub fn elements(&self) -> Vec<&String> {
        self.values.keys().collect()
    }

    /// Compute the inner product of two characters
    ///
    /// ⟨χ₁, χ₂⟩ = (1/|G|) ∑_{g∈G} χ₁(g) χ₂(g)*
    pub fn inner_product(&self, other: &Character) -> Complex {
        let mut sum = Complex::new(0.0, 0.0);
        let mut count = 0;

        for (element, value1) in &self.values {
            if let Some(value2) = other.values.get(element) {
                sum = sum + value1.clone() * value2.conjugate();
                count += 1;
            }
        }

        if count > 0 {
            sum / Complex::new(count as f64, 0.0)
        } else {
            Complex::new(0.0, 0.0)
        }
    }

    /// Check if this character is irreducible
    ///
    /// A character is irreducible if ⟨χ, χ⟩ = 1
    pub fn is_irreducible(&self) -> bool {
        let inner = self.inner_product(self);
        (inner.real() - 1.0).abs() < 1e-10 && inner.imag().abs() < 1e-10
    }

    /// Compute the norm squared of the character
    pub fn norm_squared(&self) -> f64 {
        let inner = self.inner_product(self);
        inner.real()
    }
}

impl fmt::Display for Character {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Character(degree={})", self.degree)
    }
}

/// A character table for a finite group
///
/// Stores all irreducible characters and conjugacy classes
#[derive(Clone, Debug)]
pub struct CharacterTable {
    /// Group order
    order: usize,
    /// Conjugacy class representatives
    conjugacy_classes: Vec<String>,
    /// Conjugacy class sizes
    class_sizes: Vec<usize>,
    /// Irreducible characters (rows of the character table)
    irreducible_characters: Vec<Character>,
}

impl CharacterTable {
    /// Create a new character table
    pub fn new(
        order: usize,
        conjugacy_classes: Vec<String>,
        class_sizes: Vec<usize>,
        irreducible_characters: Vec<Character>,
    ) -> Result<Self, String> {
        if conjugacy_classes.len() != class_sizes.len() {
            return Err("Number of conjugacy classes must match number of sizes".to_string());
        }

        // Verify class sizes sum to group order
        let total: usize = class_sizes.iter().sum();
        if total != order {
            return Err(format!(
                "Class sizes sum to {}, expected group order {}",
                total, order
            ));
        }

        Ok(CharacterTable {
            order,
            conjugacy_classes,
            class_sizes,
            irreducible_characters,
        })
    }

    /// Get the group order
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the number of conjugacy classes
    pub fn num_classes(&self) -> usize {
        self.conjugacy_classes.len()
    }

    /// Get the number of irreducible representations
    pub fn num_irreducibles(&self) -> usize {
        self.irreducible_characters.len()
    }

    /// Get conjugacy class representatives
    pub fn conjugacy_classes(&self) -> &[String] {
        &self.conjugacy_classes
    }

    /// Get irreducible characters
    pub fn irreducible_characters(&self) -> &[Character] {
        &self.irreducible_characters
    }

    /// Create the character table for the cyclic group Z/nZ
    pub fn cyclic_group(n: usize) -> Self {
        let conjugacy_classes: Vec<String> = (0..n).map(|i| format!("g^{}", i)).collect();
        let class_sizes = vec![1; n];

        let mut irreducible_characters = Vec::new();

        for k in 0..n {
            let mut values = HashMap::new();
            for j in 0..n {
                // χₖ(gʲ) = exp(2πijk/n)
                let angle = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                let value = Complex::new(angle.cos(), angle.sin());
                values.insert(format!("g^{}", j), value);
            }
            irreducible_characters.push(Character::new(1, values));
        }

        CharacterTable {
            order: n,
            conjugacy_classes,
            class_sizes,
            irreducible_characters,
        }
    }

    /// Create the character table for the symmetric group S₃
    pub fn symmetric_group_s3() -> Self {
        let conjugacy_classes = vec!["e".to_string(), "(12)".to_string(), "(123)".to_string()];
        let class_sizes = vec![1, 3, 2];

        let mut irreducible_characters = Vec::new();

        // Trivial representation
        let mut trivial = HashMap::new();
        trivial.insert("e".to_string(), Complex::new(1.0, 0.0));
        trivial.insert("(12)".to_string(), Complex::new(1.0, 0.0));
        trivial.insert("(123)".to_string(), Complex::new(1.0, 0.0));
        irreducible_characters.push(Character::new(1, trivial));

        // Sign representation
        let mut sign = HashMap::new();
        sign.insert("e".to_string(), Complex::new(1.0, 0.0));
        sign.insert("(12)".to_string(), Complex::new(-1.0, 0.0));
        sign.insert("(123)".to_string(), Complex::new(1.0, 0.0));
        irreducible_characters.push(Character::new(1, sign));

        // Standard representation
        let mut standard = HashMap::new();
        standard.insert("e".to_string(), Complex::new(2.0, 0.0));
        standard.insert("(12)".to_string(), Complex::new(0.0, 0.0));
        standard.insert("(123)".to_string(), Complex::new(-1.0, 0.0));
        irreducible_characters.push(Character::new(2, standard));

        CharacterTable {
            order: 6,
            conjugacy_classes,
            class_sizes,
            irreducible_characters,
        }
    }

    /// Verify orthogonality relations
    ///
    /// ∑ᵢ χᵢ(C)* χᵢ(C') = |G|/|C| δ_{C,C'}
    pub fn verify_column_orthogonality(&self) -> bool {
        for (i, class1) in self.conjugacy_classes.iter().enumerate() {
            for (j, class2) in self.conjugacy_classes.iter().enumerate() {
                let mut sum = Complex::new(0.0, 0.0);

                for char in &self.irreducible_characters {
                    if let (Some(v1), Some(v2)) = (char.value(class1), char.value(class2)) {
                        sum = sum + v1.conjugate() * v2.clone();
                    }
                }

                let expected = if i == j {
                    self.order as f64 / self.class_sizes[i] as f64
                } else {
                    0.0
                };

                if (sum.real() - expected).abs() > 1e-8 || sum.imag().abs() > 1e-8 {
                    return false;
                }
            }
        }

        true
    }
}

impl fmt::Display for CharacterTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Character Table (order = {})", self.order)?;
        writeln!(f, "Classes: {:?}", self.conjugacy_classes)?;
        writeln!(f, "Sizes:   {:?}", self.class_sizes)?;

        for (i, char) in self.irreducible_characters.iter().enumerate() {
            write!(f, "χ{}: ", i)?;
            for class in &self.conjugacy_classes {
                if let Some(value) = char.value(class) {
                    write!(f, "{:6.2} ", value.real())?;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// Direct sum of representations
///
/// (ρ₁ ⊕ ρ₂)(g) = [ρ₁(g)  0   ]
///                 [0      ρ₂(g)]
pub fn direct_sum<F: Field>(
    rep1: &Representation<F>,
    rep2: &Representation<F>,
) -> Result<Representation<F>, String> {
    let new_degree = rep1.degree + rep2.degree;
    let mut new_matrices = HashMap::new();

    // Find common elements
    for element in rep1.elements() {
        if let (Some(mat1), Some(mat2)) = (rep1.matrix(element), rep2.matrix(element)) {
            // Create block diagonal matrix
            let mut entries = Vec::new();

            for i in 0..new_degree {
                for j in 0..new_degree {
                    if i < rep1.degree && j < rep1.degree {
                        entries.push(mat1.get(i, j).map_err(|e| format!("{:?}", e))?.clone());
                    } else if i >= rep1.degree && j >= rep1.degree {
                        entries.push(mat2.get(i - rep1.degree, j - rep1.degree).map_err(|e| format!("{:?}", e))?.clone());
                    } else {
                        entries.push(F::zero());
                    }
                }
            }

            let mat = Matrix::from_vec(new_degree, new_degree, entries)
                .map_err(|e| format!("{:?}", e))?;
            new_matrices.insert(element.clone(), mat);
        }
    }

    Representation::new(new_degree, new_matrices)
}

/// Tensor product of representations
///
/// (ρ₁ ⊗ ρ₂)(g) = ρ₁(g) ⊗ ρ₂(g)
pub fn tensor_product<F: Field>(
    rep1: &Representation<F>,
    rep2: &Representation<F>,
) -> Result<Representation<F>, String> {
    let new_degree = rep1.degree * rep2.degree;
    let mut new_matrices = HashMap::new();

    // Find common elements
    for element in rep1.elements() {
        if let (Some(mat1), Some(mat2)) = (rep1.matrix(element), rep2.matrix(element)) {
            // Compute Kronecker product
            let mut entries = Vec::new();

            for i1 in 0..rep1.degree {
                for i2 in 0..rep2.degree {
                    for j1 in 0..rep1.degree {
                        for j2 in 0..rep2.degree {
                            let elem1 = mat1.get(i1, j1).map_err(|e| format!("{:?}", e))?;
                            let elem2 = mat2.get(i2, j2).map_err(|e| format!("{:?}", e))?;
                            entries.push(elem1.clone() * elem2.clone());
                        }
                    }
                }
            }

            let mat = Matrix::from_vec(new_degree, new_degree, entries)
                .map_err(|e| format!("{:?}", e))?;
            new_matrices.insert(element.clone(), mat);
        }
    }

    Representation::new(new_degree, new_matrices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn make_rational(n: i64) -> Rational {
        Rational::new(n, 1).unwrap()
    }

    #[test]
    fn test_representation_creation() {
        let mut matrices = HashMap::new();

        // Identity element
        let id = Matrix::<Rational>::identity(2);
        matrices.insert("e".to_string(), id);

        let rep = Representation::new(2, matrices);
        assert!(rep.is_ok());
    }

    #[test]
    fn test_trivial_representation() {
        let mut matrices = HashMap::new();

        // All 1x1 identity matrices
        let one = Matrix::from_vec(1, 1, vec![make_rational(1)]).unwrap();
        matrices.insert("e".to_string(), one.clone());
        matrices.insert("g".to_string(), one);

        let rep = Representation::new(1, matrices).unwrap();
        assert!(rep.is_trivial());
    }

    #[test]
    fn test_representation_trace() {
        let mut matrices = HashMap::new();

        // 2x2 matrix with trace = 3
        let mat = Matrix::from_vec(
            2,
            2,
            vec![
                make_rational(2),
                make_rational(0),
                make_rational(0),
                make_rational(1),
            ],
        )
        .unwrap();

        matrices.insert("g".to_string(), mat);

        let rep = Representation::new(2, matrices).unwrap();
        let trace = rep.trace("g").unwrap();

        assert_eq!(trace, make_rational(3));
    }

    #[test]
    fn test_character_from_representation() {
        let mut matrices = HashMap::new();

        let id = Matrix::identity(2);
        matrices.insert("e".to_string(), id);

        let mat = Matrix::from_vec(
            2,
            2,
            vec![
                make_rational(-1),
                make_rational(0),
                make_rational(0),
                make_rational(1),
            ],
        )
        .unwrap();
        matrices.insert("g".to_string(), mat);

        let rep = Representation::new(2, matrices).unwrap();
        let char = Character::from_representation_rational(&rep);

        assert_eq!(char.degree(), 2);

        // Character of identity should be 2
        let chi_e = char.value("e").unwrap();
        assert!((chi_e.real() - 2.0).abs() < 1e-10);

        // Character of g should be 0
        let chi_g = char.value("g").unwrap();
        assert!(chi_g.real().abs() < 1e-10);
    }

    #[test]
    fn test_character_inner_product() {
        let mut values1 = HashMap::new();
        values1.insert("e".to_string(), Complex::new(1.0, 0.0));
        values1.insert("g".to_string(), Complex::new(1.0, 0.0));
        let char1 = Character::new(1, values1);

        let mut values2 = HashMap::new();
        values2.insert("e".to_string(), Complex::new(1.0, 0.0));
        values2.insert("g".to_string(), Complex::new(-1.0, 0.0));
        let char2 = Character::new(1, values2);

        // Different irreducible characters should be orthogonal
        let inner = char1.inner_product(&char2);
        assert!(inner.real().abs() < 1e-10);
    }

    #[test]
    fn test_character_is_irreducible() {
        let mut values = HashMap::new();
        values.insert("e".to_string(), Complex::new(1.0, 0.0));
        values.insert("g".to_string(), Complex::new(1.0, 0.0));
        let char = Character::new(1, values);

        assert!(char.is_irreducible());
    }

    #[test]
    fn test_cyclic_group_character_table() {
        let table = CharacterTable::cyclic_group(3);

        assert_eq!(table.order(), 3);
        assert_eq!(table.num_classes(), 3);
        assert_eq!(table.num_irreducibles(), 3);

        // All conjugacy classes have size 1 in abelian groups
        assert_eq!(table.class_sizes, vec![1, 1, 1]);

        // Verify orthogonality
        assert!(table.verify_column_orthogonality());
    }

    #[test]
    fn test_s3_character_table() {
        let table = CharacterTable::symmetric_group_s3();

        assert_eq!(table.order(), 6);
        assert_eq!(table.num_classes(), 3);
        assert_eq!(table.num_irreducibles(), 3);

        // Class sizes: {e}, {(12), (13), (23)}, {(123), (132)}
        assert_eq!(table.class_sizes, vec![1, 3, 2]);

        // Verify orthogonality
        assert!(table.verify_column_orthogonality());
    }

    #[test]
    fn test_direct_sum() {
        let mut matrices1 = HashMap::new();
        let id1 = Matrix::<Rational>::identity(1);
        matrices1.insert("e".to_string(), id1);
        let rep1 = Representation::new(1, matrices1).unwrap();

        let mut matrices2 = HashMap::new();
        let id2 = Matrix::<Rational>::identity(2);
        matrices2.insert("e".to_string(), id2);
        let rep2 = Representation::new(2, matrices2).unwrap();

        let sum = direct_sum(&rep1, &rep2).unwrap();
        assert_eq!(sum.degree(), 3);

        // Check it's block diagonal
        let mat = sum.matrix("e").unwrap();
        assert_eq!(*mat.get(0, 0).unwrap(), make_rational(1));
        assert_eq!(*mat.get(1, 1).unwrap(), make_rational(1));
        assert_eq!(*mat.get(2, 2).unwrap(), make_rational(1));
        assert_eq!(*mat.get(0, 1).unwrap(), make_rational(0));
        assert_eq!(*mat.get(0, 2).unwrap(), make_rational(0));
    }

    #[test]
    fn test_tensor_product() {
        let mut matrices1 = HashMap::new();
        let mat1 = Matrix::from_vec(1, 1, vec![make_rational(2)]).unwrap();
        matrices1.insert("g".to_string(), mat1);
        let rep1 = Representation::new(1, matrices1).unwrap();

        let mut matrices2 = HashMap::new();
        let mat2 = Matrix::from_vec(1, 1, vec![make_rational(3)]).unwrap();
        matrices2.insert("g".to_string(), mat2);
        let rep2 = Representation::new(1, matrices2).unwrap();

        let tensor = tensor_product(&rep1, &rep2).unwrap();
        assert_eq!(tensor.degree(), 1);

        // 2 ⊗ 3 = 6
        let mat = tensor.matrix("g").unwrap();
        assert_eq!(*mat.get(0, 0).unwrap(), make_rational(6));
    }

    #[test]
    fn test_representation_dimension_validation() {
        let mut matrices = HashMap::new();

        // Wrong dimension matrix
        let mat = Matrix::from_vec(2, 3, vec![make_rational(1); 6]).unwrap();
        matrices.insert("g".to_string(), mat);

        let rep = Representation::<Rational>::new(2, matrices);
        assert!(rep.is_err());
    }

    #[test]
    fn test_character_table_creation() {
        let classes = vec!["e".to_string(), "g".to_string()];
        let sizes = vec![1, 1];
        let chars = vec![];

        let table = CharacterTable::new(2, classes, sizes, chars);
        assert!(table.is_ok());
    }

    #[test]
    fn test_character_table_invalid_sizes() {
        let classes = vec!["e".to_string(), "g".to_string()];
        let sizes = vec![1, 2]; // Sum = 3, not matching order
        let chars = vec![];

        let table = CharacterTable::new(2, classes, sizes, chars);
        assert!(table.is_err());
    }

    #[test]
    fn test_cyclic_group_irreducibles() {
        let table = CharacterTable::cyclic_group(4);

        // Z/4Z has 4 irreducible 1-dimensional representations
        for char in table.irreducible_characters() {
            assert_eq!(char.degree(), 1);
            assert!(char.is_irreducible());
        }
    }

    #[test]
    fn test_s3_trivial_character() {
        let table = CharacterTable::symmetric_group_s3();

        let trivial = &table.irreducible_characters()[0];
        assert_eq!(trivial.degree(), 1);

        // All values should be 1
        for class in table.conjugacy_classes() {
            let val = trivial.value(class).unwrap();
            assert!((val.real() - 1.0).abs() < 1e-10);
            assert!(val.imag().abs() < 1e-10);
        }
    }

    #[test]
    fn test_s3_sign_character() {
        let table = CharacterTable::symmetric_group_s3();

        let sign = &table.irreducible_characters()[1];
        assert_eq!(sign.degree(), 1);

        // Identity: +1, Transposition: -1, 3-cycle: +1
        let e_val = sign.value("e").unwrap();
        assert!((e_val.real() - 1.0).abs() < 1e-10);

        let trans_val = sign.value("(12)").unwrap();
        assert!((trans_val.real() + 1.0).abs() < 1e-10);

        let cycle_val = sign.value("(123)").unwrap();
        assert!((cycle_val.real() - 1.0).abs() < 1e-10);
    }
}
