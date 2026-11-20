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

/// Why a character inner product, or a character with class sizes, was refused.
///
/// The inner product `<chi1, chi2> = (1/|G|) sum_C |C| chi1(C) conj(chi2(C))` needs the class
/// sizes `|C|` (MAGMA ch. 91, `InnerProduct`). Tracker T-215: the old product divided by the
/// number of keys and never weighted by class size. These errors replace the plausible wrong
/// values it returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CharacterError {
    /// One character carries class sizes and the other does not.
    SizesOnOneSide,
    /// Both characters carry class sizes, and they differ at this key.
    SizeMismatch {
        /// The class where the two size maps differ (or that only one of them has).
        key: String,
    },
    /// Neither character carries class sizes, and their key sets differ at this key.
    KeyMismatch {
        /// A key of one character that the other lacks.
        key: String,
    },
    /// A size map whose keys differ from the value keys, at this key.
    SizeKeysDiffer {
        /// A key of one map that the other lacks.
        key: String,
    },
    /// A class of size 0.
    ZeroSize {
        /// The class.
        key: String,
    },
    /// A character without a value on a class of the table.
    MissingValue {
        /// The class.
        key: String,
    },
    /// A character with no values at all.
    NoValues,
}

impl fmt::Display for CharacterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CharacterError::SizesOnOneSide => {
                write!(f, "only one of the two characters carries class sizes")
            }
            CharacterError::SizeMismatch { key } => {
                write!(f, "the two characters have different class sizes at {key}")
            }
            CharacterError::KeyMismatch { key } => {
                write!(
                    f,
                    "the two characters have different classes: {key} is on one side only"
                )
            }
            CharacterError::SizeKeysDiffer { key } => {
                write!(
                    f,
                    "the class sizes and the values have different keys: {key}"
                )
            }
            CharacterError::ZeroSize { key } => write!(f, "class {key} has size 0"),
            CharacterError::MissingValue { key } => {
                write!(f, "the character has no value on class {key}")
            }
            CharacterError::NoValues => write!(f, "the character has no values"),
        }
    }
}

impl std::error::Error for CharacterError {}

/// The first key of `a` missing from `b`, or of `b` missing from `a` (sorted, so the
/// reported key does not depend on the hash order).
fn first_key_difference<V, W>(a: &HashMap<String, V>, b: &HashMap<String, W>) -> Option<String> {
    let mut missing: Vec<&String> = a
        .keys()
        .filter(|k| !b.contains_key(*k))
        .chain(b.keys().filter(|k| !a.contains_key(*k)))
        .collect();
    missing.sort();
    missing.first().map(|k| (*k).clone())
}

/// A character of a representation
///
/// The character χ(g) = tr(ρ(g)) is the trace of the representation matrix.
///
/// With class sizes (`with_class_sizes`) the keys are conjugacy classes and the inner product
/// is weighted by `|C|`. Without them each key stands for one group element (the old,
/// element-keyed reading), which is right only when the keys list the whole group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Character {
    /// Degree of the representation
    degree: usize,
    /// Map from conjugacy class representatives to character values
    values: HashMap<String, Complex>,
    /// Class sizes `|C|` for the keys of `values`, if known (T-215).
    class_sizes: Option<HashMap<String, usize>>,
}

impl Character {
    /// Create a new character (element-keyed, no class sizes)
    pub fn new(degree: usize, values: HashMap<String, Complex>) -> Self {
        Character {
            degree,
            values,
            class_sizes: None,
        }
    }

    /// A character on conjugacy classes with their sizes. `Err` if the keys of `sizes` differ
    /// from those of `values`, or a size is 0.
    ///
    /// The keys must be all the classes of the group. Nothing checks this: with only some of
    /// the classes, [`Character::try_inner_product`] divides by the partial sum of the sizes,
    /// and its value is only a weighted mean over those classes, not the inner product.
    pub fn with_class_sizes(
        degree: usize,
        values: HashMap<String, Complex>,
        sizes: HashMap<String, usize>,
    ) -> Result<Self, CharacterError> {
        if let Some(key) = first_key_difference(&values, &sizes) {
            return Err(CharacterError::SizeKeysDiffer { key });
        }
        let mut zero: Vec<&String> = sizes
            .iter()
            .filter(|(_, &v)| v == 0)
            .map(|(k, _)| k)
            .collect();
        zero.sort();
        if let Some(key) = zero.first() {
            return Err(CharacterError::ZeroSize {
                key: (*key).clone(),
            });
        }
        Ok(Character {
            degree,
            values,
            class_sizes: Some(sizes),
        })
    }

    /// The class sizes, if this character carries them.
    pub fn class_sizes(&self) -> Option<&HashMap<String, usize>> {
        self.class_sizes.as_ref()
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
            class_sizes: None,
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

    /// Get a reference to the internal values HashMap
    pub fn values(&self) -> &HashMap<String, Complex> {
        &self.values
    }

    /// The inner product of two characters, or why it cannot be computed (T-215).
    ///
    /// | Class sizes | Result |
    /// |---|---|
    /// | on both sides, equal maps | `sum_C |C| chi1(C) conj(chi2(C)) / sum_C |C|` |
    /// | on neither side, equal key sets | the mean of `chi1(g) conj(chi2(g))` over the keys (each key is one group element) |
    /// | on neither side, different key sets | `Err(KeyMismatch)` |
    /// | on one side only | `Err(SizesOnOneSide)` |
    /// | on both sides, different maps | `Err(SizeMismatch)` |
    ///
    /// `Err(NoValues)` for characters with no values.
    ///
    /// With class sizes, the keys must be all the classes of the group, so that `sum_C |C|`
    /// is `|G|`. Otherwise the value is only a weighted mean over the given classes (for the
    /// standard character of S3 on `{e: 1, (12): 3}` it is 1).
    pub fn try_inner_product(&self, other: &Character) -> Result<Complex, CharacterError> {
        match (&self.class_sizes, &other.class_sizes) {
            (Some(a), Some(b)) => {
                if let Some(key) = first_key_difference(a, b) {
                    return Err(CharacterError::SizeMismatch { key });
                }
                let mut differ: Vec<&String> = a
                    .iter()
                    .filter(|(k, v)| b.get(*k) != Some(v))
                    .map(|(k, _)| k)
                    .collect();
                differ.sort();
                if let Some(key) = differ.first() {
                    return Err(CharacterError::SizeMismatch {
                        key: (*key).clone(),
                    });
                }
                if a.is_empty() {
                    return Err(CharacterError::NoValues);
                }
                // with_class_sizes makes the value keys equal to the size keys on both sides.
                let mut sum = Complex::new(0.0, 0.0);
                let mut total = 0usize;
                for (key, &size) in a {
                    let v1 = &self.values[key];
                    let v2 = &other.values[key];
                    sum = sum + Complex::new(size as f64, 0.0) * v1.clone() * v2.conjugate();
                    total += size;
                }
                Ok(sum / Complex::new(total as f64, 0.0))
            }
            (None, None) => {
                if let Some(key) = first_key_difference(&self.values, &other.values) {
                    return Err(CharacterError::KeyMismatch { key });
                }
                if self.values.is_empty() {
                    return Err(CharacterError::NoValues);
                }
                let mut sum = Complex::new(0.0, 0.0);
                for (element, value1) in &self.values {
                    let value2 = &other.values[element];
                    sum = sum + value1.clone() * value2.conjugate();
                }
                Ok(sum / Complex::new(self.values.len() as f64, 0.0))
            }
            _ => Err(CharacterError::SizesOnOneSide),
        }
    }

    /// Compute the inner product of two characters
    ///
    /// ⟨χ₁, χ₂⟩ = (1/|G|) ∑_{g∈G} χ₁(g) χ₂(g)*, weighted by the class sizes when the
    /// characters carry them (see [`Character::try_inner_product`]).
    ///
    /// # Panics
    ///
    /// When `try_inner_product` returns `Err` (class sizes on one side only, two different
    /// size maps, different key sets, or no values), with the error text. A character with
    /// itself never panics unless it has no values.
    pub fn inner_product(&self, other: &Character) -> Complex {
        match self.try_inner_product(other) {
            Ok(v) => v,
            Err(e) => panic!("Character::inner_product: {e}"),
        }
    }

    /// Check if this character is irreducible
    ///
    /// A character is irreducible if ⟨χ, χ⟩ = 1
    /// A character with no values is not irreducible.
    pub fn is_irreducible(&self) -> bool {
        match self.try_inner_product(self) {
            Ok(inner) => (inner.real() - 1.0).abs() < 1e-10 && inner.imag().abs() < 1e-10,
            Err(_) => false,
        }
    }

    /// Compute the norm squared of the character (0 for a character with no values, the
    /// empty sum).
    pub fn norm_squared(&self) -> f64 {
        match self.try_inner_product(self) {
            Ok(inner) => inner.real(),
            Err(_) => 0.0,
        }
    }
}

impl fmt::Display for Character {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Character(degree={})", self.degree)
    }
}

impl Default for Character {
    /// Create a default character with degree 0 and no values
    fn default() -> Self {
        Character {
            degree: 0,
            values: HashMap::new(),
            class_sizes: None,
        }
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

    /// The class sizes, in the order of [`CharacterTable::conjugacy_classes`].
    pub fn class_sizes(&self) -> &[usize] {
        &self.class_sizes
    }

    /// `<a, b> = (1/|G|) sum_C |C| a(C) conj(b(C))` over the classes of this table, with the
    /// table's class sizes (T-215). `Err(MissingValue)` if `a` or `b` has no value on a class
    /// of the table; `Err(SizeMismatch)` if a character carries class sizes that differ from
    /// the table's.
    pub fn inner_product(&self, a: &Character, b: &Character) -> Result<Complex, CharacterError> {
        for chi in [a, b] {
            if let Some(sizes) = chi.class_sizes() {
                for (class, &size) in self.conjugacy_classes.iter().zip(&self.class_sizes) {
                    if sizes.get(class) != Some(&size) {
                        return Err(CharacterError::SizeMismatch { key: class.clone() });
                    }
                }
            }
        }
        let mut sum = Complex::new(0.0, 0.0);
        for (class, &size) in self.conjugacy_classes.iter().zip(&self.class_sizes) {
            let missing = || CharacterError::MissingValue { key: class.clone() };
            let va = a.value(class).ok_or_else(missing)?;
            let vb = b.value(class).ok_or_else(missing)?;
            sum = sum + Complex::new(size as f64, 0.0) * va.clone() * vb.conjugate();
        }
        Ok(sum / Complex::new(self.order as f64, 0.0))
    }

    /// Create the character table for the cyclic group Z/nZ
    pub fn cyclic_group(n: usize) -> Self {
        let conjugacy_classes: Vec<String> = (0..n).map(|i| format!("g^{}", i)).collect();
        let class_sizes = vec![1; n];

        let mut irreducible_characters = Vec::new();

        let sizes: HashMap<String, usize> =
            conjugacy_classes.iter().map(|c| (c.clone(), 1)).collect();
        for k in 0..n {
            let mut values = HashMap::new();
            for j in 0..n {
                // χₖ(gʲ) = exp(2πijk/n)
                let angle = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                let value = Complex::new(angle.cos(), angle.sin());
                values.insert(format!("g^{}", j), value);
            }
            irreducible_characters.push(
                Character::with_class_sizes(1, values, sizes.clone())
                    .expect("the classes of C_n have size 1"),
            );
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
        let sizes: HashMap<String, usize> = conjugacy_classes
            .iter()
            .cloned()
            .zip(class_sizes.iter().copied())
            .collect();
        let with_sizes = |degree: usize, values: HashMap<String, Complex>| {
            Character::with_class_sizes(degree, values, sizes.clone())
                .expect("the S3 values are given on its three classes")
        };

        let mut irreducible_characters = Vec::new();

        // Trivial representation
        let mut trivial = HashMap::new();
        trivial.insert("e".to_string(), Complex::new(1.0, 0.0));
        trivial.insert("(12)".to_string(), Complex::new(1.0, 0.0));
        trivial.insert("(123)".to_string(), Complex::new(1.0, 0.0));
        irreducible_characters.push(with_sizes(1, trivial));

        // Sign representation
        let mut sign = HashMap::new();
        sign.insert("e".to_string(), Complex::new(1.0, 0.0));
        sign.insert("(12)".to_string(), Complex::new(-1.0, 0.0));
        sign.insert("(123)".to_string(), Complex::new(1.0, 0.0));
        irreducible_characters.push(with_sizes(1, sign));

        // Standard representation
        let mut standard = HashMap::new();
        standard.insert("e".to_string(), Complex::new(2.0, 0.0));
        standard.insert("(12)".to_string(), Complex::new(0.0, 0.0));
        standard.insert("(123)".to_string(), Complex::new(-1.0, 0.0));
        irreducible_characters.push(with_sizes(2, standard));

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

    // T-215. The tracker's failure case: the S3 standard character (2, 0, -1) on classes of
    // sizes 1, 3, 2 has <chi, chi> = (4 + 0 + 2) / 6 = 1; the old unweighted mean gave
    // (4 + 0 + 1) / 3 = 1.667. Reference: the S3 table of
    // GAP 4.12.1 (Irr(CharacterTable(SymmetricGroup(3))), SizesConjugacyClasses [1, 3, 2]).
    #[test]
    fn t215_s3_standard_character_is_irreducible() {
        let table = CharacterTable::symmetric_group_s3();
        let standard = &table.irreducible_characters()[2];
        let ip = standard.inner_product(standard);
        assert!(
            (ip.real() - 1.0).abs() < 1e-12 && ip.imag().abs() < 1e-12,
            "{ip:?}"
        );
        assert!(standard.is_irreducible());
        assert!((standard.norm_squared() - 1.0).abs() < 1e-12);
        let ip = table.inner_product(standard, standard).unwrap();
        assert!((ip.real() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn t215_s3_rows_are_orthonormal() {
        let table = CharacterTable::symmetric_group_s3();
        let chars = table.irreducible_characters();
        for (i, a) in chars.iter().enumerate() {
            for (j, b) in chars.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                let ip = a.try_inner_product(b).unwrap();
                assert!(
                    (ip.real() - want).abs() < 1e-12 && ip.imag().abs() < 1e-12,
                    "{i} {j}"
                );
                let ip = table.inner_product(a, b).unwrap();
                assert!(
                    (ip.real() - want).abs() < 1e-12 && ip.imag().abs() < 1e-12,
                    "{i} {j}"
                );
            }
        }
        // trivial and sign are orthogonal
        let ip = chars[0].inner_product(&chars[1]);
        assert!(ip.real().abs() < 1e-12);
    }

    fn s3_values(v: [f64; 3]) -> HashMap<String, Complex> {
        ["e", "(12)", "(123)"]
            .iter()
            .zip(v)
            .map(|(k, x)| (k.to_string(), Complex::new(x, 0.0)))
            .collect()
    }

    fn s3_sizes(v: [usize; 3]) -> HashMap<String, usize> {
        ["e", "(12)", "(123)"]
            .iter()
            .zip(v)
            .map(|(k, x)| (k.to_string(), x))
            .collect()
    }

    #[test]
    fn t215_try_inner_product_edge_cases() {
        let sized =
            Character::with_class_sizes(2, s3_values([2.0, 0.0, -1.0]), s3_sizes([1, 3, 2]))
                .unwrap();
        let unsized_char = Character::new(2, s3_values([2.0, 0.0, -1.0]));
        assert_eq!(
            sized.try_inner_product(&unsized_char),
            Err(CharacterError::SizesOnOneSide)
        );
        assert_eq!(
            unsized_char.try_inner_product(&sized),
            Err(CharacterError::SizesOnOneSide)
        );
        let other_sizes =
            Character::with_class_sizes(2, s3_values([2.0, 0.0, -1.0]), s3_sizes([1, 2, 3]))
                .unwrap();
        assert!(matches!(
            sized.try_inner_product(&other_sizes),
            Err(CharacterError::SizeMismatch { .. })
        ));
        // review 2, n3: different key sets without sizes are refused
        let mut fewer = s3_values([2.0, 0.0, -1.0]);
        fewer.remove("(123)");
        let fewer = Character::new(2, fewer);
        assert_eq!(
            unsized_char.try_inner_product(&fewer),
            Err(CharacterError::KeyMismatch {
                key: "(123)".to_string()
            })
        );
        // the element-keyed reading with equal keys is the old mean
        let ip = unsized_char.try_inner_product(&unsized_char).unwrap();
        assert!((ip.real() - 5.0 / 3.0).abs() < 1e-12);
        assert_eq!(
            Character::default().try_inner_product(&Character::default()),
            Err(CharacterError::NoValues)
        );
        assert!(!Character::default().is_irreducible());
    }

    #[test]
    fn t215_with_class_sizes_refuses_bad_maps() {
        let mut sizes = s3_sizes([1, 3, 2]);
        sizes.remove("(12)");
        assert_eq!(
            Character::with_class_sizes(2, s3_values([2.0, 0.0, -1.0]), sizes),
            Err(CharacterError::SizeKeysDiffer {
                key: "(12)".to_string()
            })
        );
        assert_eq!(
            Character::with_class_sizes(2, s3_values([2.0, 0.0, -1.0]), s3_sizes([1, 0, 2])),
            Err(CharacterError::ZeroSize {
                key: "(12)".to_string()
            })
        );
        let table = CharacterTable::symmetric_group_s3();
        let mut partial = s3_values([1.0, 1.0, 1.0]);
        partial.remove("e");
        let partial = Character::new(1, partial);
        assert_eq!(
            table.inner_product(&partial, &table.irreducible_characters()[0]),
            Err(CharacterError::MissingValue {
                key: "e".to_string()
            })
        );
    }

    #[test]
    #[should_panic(expected = "different class sizes")]
    fn t215_inner_product_panics_on_two_size_maps() {
        let a = Character::with_class_sizes(2, s3_values([2.0, 0.0, -1.0]), s3_sizes([1, 3, 2]))
            .unwrap();
        let b = Character::with_class_sizes(2, s3_values([2.0, 0.0, -1.0]), s3_sizes([1, 2, 3]))
            .unwrap();
        let _ = a.inner_product(&b);
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
