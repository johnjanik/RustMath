//! Free Zinbiel Algebras
//!
//! A Zinbiel algebra is a non-associative algebra with binary operation (∘)
//! satisfying the defining relation:
//!   (a ∘ b) ∘ c = a ∘ (b ∘ c) + a ∘ (c ∘ b)
//!
//! These algebras are the Koszul dual to Leibniz algebras and have interesting
//! divided power properties. The free Zinbiel algebra on n generators has a
//! basis indexed by words in those generators, with multiplication defined
//! via shuffle products.
//!
//! Corresponds to sage.algebras.free_zinbiel_algebra
//!
//! References:
//! - Loday, J.-L. "Dialgebras" (1995)
//! - Loday, J.-L. and Ronco, M. "Trialgebras and families of polytopes" (2002)

use rustmath_core::Ring;
use rustmath_modules::CombinatorialFreeModuleElement;
use std::collections::HashMap;
use std::fmt::{self, Display};

/// A word in the generators of a free Zinbiel algebra
///
/// Represented as a sequence of generator indices
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ZinbielWord {
    /// Sequence of generator indices
    letters: Vec<usize>,
}

impl ZinbielWord {
    /// Create a new word
    pub fn new(letters: Vec<usize>) -> Self {
        ZinbielWord { letters }
    }

    /// Create the empty word (identity)
    pub fn empty() -> Self {
        ZinbielWord {
            letters: Vec::new(),
        }
    }

    /// Create a single-letter word (generator)
    pub fn generator(index: usize) -> Self {
        ZinbielWord {
            letters: vec![index],
        }
    }

    /// Length of the word (degree)
    pub fn len(&self) -> usize {
        self.letters.len()
    }

    /// Check if this is the empty word
    pub fn is_empty(&self) -> bool {
        self.letters.is_empty()
    }

    /// Get the letters in this word
    pub fn letters(&self) -> &[usize] {
        &self.letters
    }

    /// Concatenate two words
    pub fn concatenate(&self, other: &ZinbielWord) -> ZinbielWord {
        let mut result = self.letters.clone();
        result.extend_from_slice(&other.letters);
        ZinbielWord::new(result)
    }

    /// Compute all shuffles of two words
    ///
    /// A shuffle interleaves the letters while preserving relative order
    fn shuffles(&self, other: &ZinbielWord) -> Vec<ZinbielWord> {
        if self.is_empty() {
            return vec![other.clone()];
        }
        if other.is_empty() {
            return vec![self.clone()];
        }

        let mut result = Vec::new();

        // Take first letter from self
        let mut rest1 = ZinbielWord::new(self.letters[1..].to_vec());
        for shuffle in rest1.shuffles(other) {
            let mut letters = vec![self.letters[0]];
            letters.extend_from_slice(shuffle.letters());
            result.push(ZinbielWord::new(letters));
        }

        // Take first letter from other
        let mut rest2 = ZinbielWord::new(other.letters[1..].to_vec());
        for shuffle in self.shuffles(&rest2) {
            let mut letters = vec![other.letters[0]];
            letters.extend_from_slice(shuffle.letters());
            result.push(ZinbielWord::new(letters));
        }

        result
    }

    /// Left Zinbiel product via shuffles
    ///
    /// Uses the left-sided convention
    pub fn zinbiel_product_left(&self, other: &ZinbielWord) -> Vec<ZinbielWord> {
        // Simplified shuffle product for Zinbiel operation
        self.shuffles(other)
    }

    /// Right Zinbiel product via shuffles
    ///
    /// Uses the right-sided convention
    pub fn zinbiel_product_right(&self, other: &ZinbielWord) -> Vec<ZinbielWord> {
        // Simplified shuffle product for Zinbiel operation
        other.shuffles(self)
    }
}

impl Display for ZinbielWord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            write!(f, "1")
        } else {
            let parts: Vec<String> = self.letters.iter().map(|i| format!("x{}", i)).collect();
            write!(f, "{}", parts.join(""))
        }
    }
}

/// Free Zinbiel Algebra
///
/// The free Zinbiel algebra on n generators over a ring R.
/// Elements are linear combinations of words in the generators.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::FreeZinbielAlgebra;
/// # use rustmath_integers::Integer;
/// let algebra: FreeZinbielAlgebra<Integer> = FreeZinbielAlgebra::new(3, false);
/// assert_eq!(algebra.num_generators(), 3);
/// ```
pub struct FreeZinbielAlgebra<R: Ring> {
    /// Number of generators
    num_generators: usize,
    /// Whether to use right-sided convention (default is left)
    use_right_convention: bool,
    /// Names of generators
    generator_names: Vec<String>,
    /// Coefficient ring marker
    coefficient_ring: std::marker::PhantomData<R>,
}

impl<R: Ring + Clone> FreeZinbielAlgebra<R> {
    /// Create a new free Zinbiel algebra
    ///
    /// # Arguments
    ///
    /// * `num_generators` - Number of generators
    /// * `use_right_convention` - Whether to use right convention (default false)
    pub fn new(num_generators: usize, use_right_convention: bool) -> Self {
        let generator_names = (0..num_generators)
            .map(|i| format!("x{}", i))
            .collect();

        FreeZinbielAlgebra {
            num_generators,
            use_right_convention,
            generator_names,
            coefficient_ring: std::marker::PhantomData,
        }
    }

    /// Create with custom generator names
    pub fn with_names(
        generator_names: Vec<String>,
        use_right_convention: bool,
    ) -> Self {
        let num_generators = generator_names.len();
        FreeZinbielAlgebra {
            num_generators,
            use_right_convention,
            generator_names,
            coefficient_ring: std::marker::PhantomData,
        }
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Get the name of generator i
    pub fn generator_name(&self, i: usize) -> Option<&str> {
        self.generator_names.get(i).map(|s| s.as_str())
    }

    /// Get a generator as an algebra element
    pub fn generator(&self, i: usize) -> FreeZinbielAlgebraElement<R>
    where
        R: From<i64>,
    {
        FreeZinbielAlgebraElement::basis_element(ZinbielWord::generator(i))
    }

    /// Get all generators
    pub fn generators(&self) -> Vec<FreeZinbielAlgebraElement<R>>
    where
        R: From<i64>,
    {
        (0..self.num_generators)
            .map(|i| self.generator(i))
            .collect()
    }

    /// Get the degree of a basis word (its length)
    pub fn degree_on_basis(&self, word: &ZinbielWord) -> usize {
        word.len()
    }

    /// Compute the product of two basis words
    ///
    /// Returns a map from words to coefficients
    pub fn product_on_basis(
        &self,
        w1: &ZinbielWord,
        w2: &ZinbielWord,
    ) -> HashMap<ZinbielWord, R>
    where
        R: From<i64> + std::ops::Add<Output = R>,
    {
        let product_words = if self.use_right_convention {
            w1.zinbiel_product_right(w2)
        } else {
            w1.zinbiel_product_left(w2)
        };

        let mut result = HashMap::new();
        for word in product_words {
            let entry = result.entry(word).or_insert_with(|| R::from(0));
            *entry = entry.clone() + R::from(1);
        }

        result
    }

    /// Get the identity element
    pub fn one(&self) -> FreeZinbielAlgebraElement<R>
    where
        R: From<i64>,
    {
        FreeZinbielAlgebraElement::basis_element(ZinbielWord::empty())
    }

    /// Dimension in a given degree
    ///
    /// For n generators, dimension in degree d is n^d
    pub fn dimension_in_degree(&self, degree: usize) -> usize {
        self.num_generators.pow(degree as u32)
    }
}

/// Element of a free Zinbiel algebra
///
/// Represented as a linear combination of words
pub struct FreeZinbielAlgebraElement<R: Ring> {
    /// Coefficients for each word
    coefficients: HashMap<ZinbielWord, R>,
}

impl<R: Ring + Clone> FreeZinbielAlgebraElement<R> {
    /// Create a new element
    pub fn new(coefficients: HashMap<ZinbielWord, R>) -> Self {
        FreeZinbielAlgebraElement { coefficients }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        FreeZinbielAlgebraElement {
            coefficients: HashMap::new(),
        }
    }

    /// Create a basis element (word with coefficient 1)
    pub fn basis_element(word: ZinbielWord) -> Self
    where
        R: From<i64>,
    {
        let mut coefficients = HashMap::new();
        coefficients.insert(word, R::from(1));
        FreeZinbielAlgebraElement { coefficients }
    }

    /// Get the coefficient of a word
    pub fn coefficient(&self, word: &ZinbielWord) -> Option<&R> {
        self.coefficients.get(word)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.coefficients.values().all(|c| c.is_zero())
    }

    /// Get all words with non-zero coefficients
    pub fn support(&self) -> Vec<&ZinbielWord> {
        self.coefficients.keys().collect()
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for FreeZinbielAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.coefficients.len() != other.coefficients.len() {
            return false;
        }

        for (word, coeff) in &self.coefficients {
            if let Some(other_coeff) = other.coefficients.get(word) {
                if coeff != other_coeff {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for FreeZinbielAlgebraElement<R> {}

/// Zinbiel Functor
///
/// A construction functor for creating Zinbiel algebras over different base rings.
/// This enables systematic algebra creation and morphism induction.
pub struct ZinbielFunctor {
    /// Number of generators
    num_generators: usize,
    /// Whether to use right convention
    use_right_convention: bool,
}

impl ZinbielFunctor {
    /// Create a new Zinbiel functor
    pub fn new(num_generators: usize, use_right_convention: bool) -> Self {
        ZinbielFunctor {
            num_generators,
            use_right_convention,
        }
    }

    /// Apply the functor to a ring to get a Zinbiel algebra
    pub fn apply<R: Ring + Clone>(&self) -> FreeZinbielAlgebra<R> {
        FreeZinbielAlgebra::new(self.num_generators, self.use_right_convention)
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Check if using right convention
    pub fn uses_right_convention(&self) -> bool {
        self.use_right_convention
    }
}

/// Get the rank (number of generators) of a Zinbiel algebra
pub fn rank<R: Ring>(algebra: &FreeZinbielAlgebra<R>) -> usize {
    algebra.num_generators()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_creation() {
        let empty = ZinbielWord::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let gen = ZinbielWord::generator(0);
        assert_eq!(gen.len(), 1);
        assert!(!gen.is_empty());
    }

    #[test]
    fn test_word_concatenation() {
        let w1 = ZinbielWord::new(vec![0, 1]);
        let w2 = ZinbielWord::new(vec![2]);
        let concat = w1.concatenate(&w2);
        assert_eq!(concat.letters(), &[0, 1, 2]);
    }

    #[test]
    fn test_algebra_creation() {
        let algebra: FreeZinbielAlgebra<i64> = FreeZinbielAlgebra::new(3, false);
        assert_eq!(algebra.num_generators(), 3);
        assert_eq!(algebra.generator_name(0), Some("x0"));
    }

    #[test]
    fn test_algebra_with_names() {
        let algebra: FreeZinbielAlgebra<i64> = FreeZinbielAlgebra::with_names(
            vec!["a".to_string(), "b".to_string()],
            false,
        );
        assert_eq!(algebra.generator_name(0), Some("a"));
        assert_eq!(algebra.generator_name(1), Some("b"));
    }

    #[test]
    fn test_degree() {
        let algebra: FreeZinbielAlgebra<i64> = FreeZinbielAlgebra::new(2, false);
        let word = ZinbielWord::new(vec![0, 1, 0]);
        assert_eq!(algebra.degree_on_basis(&word), 3);
    }

    #[test]
    fn test_dimension() {
        let algebra: FreeZinbielAlgebra<i64> = FreeZinbielAlgebra::new(3, false);
        // Dimension in degree d is n^d
        assert_eq!(algebra.dimension_in_degree(0), 1);  // 3^0 = 1
        assert_eq!(algebra.dimension_in_degree(1), 3);  // 3^1 = 3
        assert_eq!(algebra.dimension_in_degree(2), 9);  // 3^2 = 9
    }

    #[test]
    fn test_element_creation() {
        let elem: FreeZinbielAlgebraElement<i64> = FreeZinbielAlgebraElement::zero();
        assert!(elem.is_zero());

        let word = ZinbielWord::generator(0);
        let elem2: FreeZinbielAlgebraElement<i64> =
            FreeZinbielAlgebraElement::basis_element(word);
        assert!(!elem2.is_zero());
    }

    #[test]
    fn test_functor() {
        let functor = ZinbielFunctor::new(3, false);
        assert_eq!(functor.num_generators(), 3);
        assert!(!functor.uses_right_convention());

        let algebra: FreeZinbielAlgebra<i64> = functor.apply();
        assert_eq!(algebra.num_generators(), 3);
    }

    #[test]
    fn test_rank_function() {
        let algebra: FreeZinbielAlgebra<i64> = FreeZinbielAlgebra::new(5, false);
        assert_eq!(rank(&algebra), 5);
    }
}
