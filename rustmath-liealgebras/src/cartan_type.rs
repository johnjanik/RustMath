//! Cartan Types - Classification of Root Systems
//!
//! Cartan types classify root systems, Dynkin diagrams, Weyl groups, and Lie algebras.
//! The finite irreducible types are: A_n, B_n, C_n, D_n (classical families) and
//! E_6, E_7, E_8, F_4, G_2 (exceptional types).
//!
//! Affine types (Kac-Moody algebras) are denoted with ^(1) superscript, e.g., A_n^(1).
//!
//! Corresponds to sage.combinat.root_system.cartan_type

use std::fmt::{self, Display};

/// The affinity level of a Cartan type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Affinity {
    /// Finite (classical) Cartan type
    Finite,
    /// Affine Cartan type (Kac-Moody algebra)
    /// The usize represents the affinity level (usually 1)
    Affine(usize),
}

/// The letter component of a Cartan type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CartanLetter {
    /// Type A_n: Special linear Lie algebra sl(n+1)
    A,
    /// Type B_n: Odd orthogonal Lie algebra so(2n+1)
    B,
    /// Type C_n: Symplectic Lie algebra sp(2n)
    C,
    /// Type D_n: Even orthogonal Lie algebra so(2n)
    D,
    /// Type E_6, E_7, E_8: Exceptional Lie algebras
    E,
    /// Type F_4: Exceptional Lie algebra
    F,
    /// Type G_2: Exceptional Lie algebra
    G,
}

impl Display for CartanLetter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CartanLetter::A => write!(f, "A"),
            CartanLetter::B => write!(f, "B"),
            CartanLetter::C => write!(f, "C"),
            CartanLetter::D => write!(f, "D"),
            CartanLetter::E => write!(f, "E"),
            CartanLetter::F => write!(f, "F"),
            CartanLetter::G => write!(f, "G"),
        }
    }
}

/// A Cartan type specifying a root system
///
/// Cartan types are specified by a letter (A, B, C, D, E, F, G), a rank (positive integer),
/// and an affinity (finite or affine).
/// Some types are only valid for specific ranks (e.g., E_6, E_7, E_8).
///
/// Examples:
/// - A_3: finite type A of rank 3
/// - A_3^(1): affine type A of rank 3
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CartanType {
    /// The letter component
    pub letter: CartanLetter,
    /// The rank (dimension of the root system)
    pub rank: usize,
    /// The affinity (finite or affine)
    pub affinity: Affinity,
}

impl CartanType {
    /// Create a new finite Cartan type
    ///
    /// # Arguments
    ///
    /// * `letter` - The type letter (A, B, C, D, E, F, G)
    /// * `rank` - The rank (must be valid for the given letter)
    ///
    /// # Returns
    ///
    /// `Some(CartanType)` if the combination is valid, `None` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
    ///
    /// let type_a3 = CartanType::new(CartanLetter::A, 3).unwrap();
    /// let type_e8 = CartanType::new(CartanLetter::E, 8).unwrap();
    /// ```
    pub fn new(letter: CartanLetter, rank: usize) -> Option<Self> {
        // Validate the combination for finite types
        let valid = match letter {
            CartanLetter::A => rank >= 1,
            CartanLetter::B => rank >= 2,
            CartanLetter::C => rank >= 2,
            CartanLetter::D => rank >= 3,
            CartanLetter::E => rank >= 6 && rank <= 8,
            CartanLetter::F => rank == 4,
            CartanLetter::G => rank == 2,
        };

        if valid {
            Some(CartanType {
                letter,
                rank,
                affinity: Affinity::Finite,
            })
        } else {
            None
        }
    }

    /// Create a new affine Cartan type
    ///
    /// # Arguments
    ///
    /// * `letter` - The type letter (A, B, C, D, E, F, G)
    /// * `rank` - The rank (must be valid for the given letter)
    /// * `affinity_level` - The affinity level (usually 1)
    ///
    /// # Returns
    ///
    /// `Some(CartanType)` if the combination is valid, `None` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
    ///
    /// let type_a3_aff = CartanType::new_affine(CartanLetter::A, 3, 1).unwrap();
    /// ```
    pub fn new_affine(letter: CartanLetter, rank: usize, affinity_level: usize) -> Option<Self> {
        // Validate the combination for affine types
        // Affine types generally have fewer restrictions than finite types
        let valid = match letter {
            CartanLetter::A => rank >= 1,
            CartanLetter::B => rank >= 2,
            CartanLetter::C => rank >= 2,
            CartanLetter::D => rank >= 3,
            CartanLetter::E => rank >= 6 && rank <= 8,
            CartanLetter::F => rank == 4,
            CartanLetter::G => rank == 2,
        };

        if valid && affinity_level >= 1 {
            Some(CartanType {
                letter,
                rank,
                affinity: Affinity::Affine(affinity_level),
            })
        } else {
            None
        }
    }

    /// Check if this is a finite Cartan type
    pub fn is_finite(&self) -> bool {
        matches!(self.affinity, Affinity::Finite)
    }

    /// Check if this is an affine Cartan type
    pub fn is_affine(&self) -> bool {
        matches!(self.affinity, Affinity::Affine(_))
    }

    /// Get the affinity level (returns 0 for finite types, n for affine^(n))
    pub fn affinity_level(&self) -> usize {
        match self.affinity {
            Affinity::Finite => 0,
            Affinity::Affine(n) => n,
        }
    }

    /// Get the rank (dimension) of this Cartan type
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the letter of this Cartan type
    pub fn letter(&self) -> CartanLetter {
        self.letter
    }

    /// Check if this Cartan type is simply-laced
    ///
    /// Simply-laced types have all edges in the Dynkin diagram of the same type.
    /// These are: A_n (n≥1), D_n (n≥4), E_6, E_7, E_8
    pub fn is_simply_laced(&self) -> bool {
        matches!(
            self.letter,
            CartanLetter::A | CartanLetter::D | CartanLetter::E
        )
    }

    /// Check if this is a classical Cartan type (A, B, C, D)
    pub fn is_classical(&self) -> bool {
        matches!(
            self.letter,
            CartanLetter::A | CartanLetter::B | CartanLetter::C | CartanLetter::D
        )
    }

    /// Check if this is an exceptional Cartan type (E, F, G)
    pub fn is_exceptional(&self) -> bool {
        matches!(
            self.letter,
            CartanLetter::E | CartanLetter::F | CartanLetter::G
        )
    }

    /// Get the dimension of the associated Lie algebra
    ///
    /// Returns the number of roots (positive + negative) plus the rank
    pub fn lie_algebra_dimension(&self) -> usize {
        match self.letter {
            CartanLetter::A => (self.rank + 1) * (self.rank + 1) - 1,
            CartanLetter::B => self.rank * (2 * self.rank + 1),
            CartanLetter::C => self.rank * (2 * self.rank + 1),
            CartanLetter::D => self.rank * (2 * self.rank - 1),
            CartanLetter::E => match self.rank {
                6 => 78,
                7 => 133,
                8 => 248,
                _ => 0,
            },
            CartanLetter::F => 52, // F_4
            CartanLetter::G => 14, // G_2
        }
    }

    /// Get the number of positive roots
    pub fn num_positive_roots(&self) -> usize {
        match self.letter {
            CartanLetter::A => self.rank * (self.rank + 1) / 2,
            CartanLetter::B | CartanLetter::C => self.rank * self.rank,
            CartanLetter::D => self.rank * (self.rank - 1),
            CartanLetter::E => match self.rank {
                6 => 36,
                7 => 63,
                8 => 120,
                _ => 0,
            },
            CartanLetter::F => 24, // F_4
            CartanLetter::G => 6,  // G_2
        }
    }

    /// Get the dual Cartan type
    ///
    /// The dual swaps B ↔ C, and leaves all others unchanged
    pub fn dual(&self) -> Self {
        let dual_letter = match self.letter {
            CartanLetter::B => CartanLetter::C,
            CartanLetter::C => CartanLetter::B,
            other => other,
        };
        CartanType {
            letter: dual_letter,
            rank: self.rank,
            affinity: self.affinity,
        }
    }
}

impl Display for CartanType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.affinity {
            Affinity::Finite => write!(f, "{}_{}", self.letter, self.rank),
            Affinity::Affine(level) => write!(f, "{}_{}^({})", self.letter, self.rank, level),
        }
    }
}

/// Parse a Cartan type from a string like "A3", "E8", "A3^(1)", etc.
impl std::str::FromStr for CartanType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err("Empty Cartan type string".to_string());
        }

        let letter_char = s.chars().next().unwrap();
        let letter = match letter_char {
            'A' => CartanLetter::A,
            'B' => CartanLetter::B,
            'C' => CartanLetter::C,
            'D' => CartanLetter::D,
            'E' => CartanLetter::E,
            'F' => CartanLetter::F,
            'G' => CartanLetter::G,
            _ => return Err(format!("Invalid Cartan letter: {}", letter_char)),
        };

        // Check if there's an affinity marker ^(n)
        let remainder = &s[1..];
        if let Some(caret_pos) = remainder.find('^') {
            // Affine type
            let rank_str = &remainder[..caret_pos];
            let rank = rank_str
                .parse::<usize>()
                .map_err(|_| format!("Invalid rank: {}", rank_str))?;

            // Parse affinity level from ^(n)
            let affinity_str = &remainder[caret_pos + 1..];
            if !affinity_str.starts_with('(') || !affinity_str.ends_with(')') {
                return Err(format!("Invalid affinity notation: {}", affinity_str));
            }

            let level_str = &affinity_str[1..affinity_str.len() - 1];
            let level = level_str
                .parse::<usize>()
                .map_err(|_| format!("Invalid affinity level: {}", level_str))?;

            CartanType::new_affine(letter, rank, level)
                .ok_or_else(|| format!("Invalid affine Cartan type: {}", s))
        } else {
            // Finite type
            let rank = remainder
                .parse::<usize>()
                .map_err(|_| format!("Invalid rank: {}", remainder))?;

            CartanType::new(letter, rank)
                .ok_or_else(|| format!("Invalid Cartan type: {}", s))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartan_type_creation() {
        assert!(CartanType::new(CartanLetter::A, 3).is_some());
        assert!(CartanType::new(CartanLetter::E, 8).is_some());
        assert!(CartanType::new(CartanLetter::E, 5).is_none()); // E_5 doesn't exist
        assert!(CartanType::new(CartanLetter::B, 1).is_none()); // B_1 is not standard
    }

    #[test]
    fn test_simply_laced() {
        let a3 = CartanType::new(CartanLetter::A, 3).unwrap();
        let b3 = CartanType::new(CartanLetter::B, 3).unwrap();
        let e8 = CartanType::new(CartanLetter::E, 8).unwrap();

        assert!(a3.is_simply_laced());
        assert!(!b3.is_simply_laced());
        assert!(e8.is_simply_laced());
    }

    #[test]
    fn test_dual() {
        let b3 = CartanType::new(CartanLetter::B, 3).unwrap();
        let c3 = CartanType::new(CartanLetter::C, 3).unwrap();

        assert_eq!(b3.dual(), c3);
        assert_eq!(c3.dual(), b3);

        let a3 = CartanType::new(CartanLetter::A, 3).unwrap();
        assert_eq!(a3.dual(), a3); // A is self-dual
    }

    #[test]
    fn test_from_str() {
        let a3: CartanType = "A3".parse().unwrap();
        assert_eq!(a3.letter, CartanLetter::A);
        assert_eq!(a3.rank, 3);

        let e8: CartanType = "E8".parse().unwrap();
        assert_eq!(e8.letter, CartanLetter::E);
        assert_eq!(e8.rank, 8);

        assert!("E5".parse::<CartanType>().is_err());
        assert!("X3".parse::<CartanType>().is_err());
    }

    #[test]
    fn test_dimensions() {
        let a2 = CartanType::new(CartanLetter::A, 2).unwrap();
        assert_eq!(a2.lie_algebra_dimension(), 8); // sl(3) has dimension 8

        let e8 = CartanType::new(CartanLetter::E, 8).unwrap();
        assert_eq!(e8.lie_algebra_dimension(), 248);

        let g2 = CartanType::new(CartanLetter::G, 2).unwrap();
        assert_eq!(g2.num_positive_roots(), 6);
    }

    #[test]
    fn test_classification() {
        let a3 = CartanType::new(CartanLetter::A, 3).unwrap();
        let e8 = CartanType::new(CartanLetter::E, 8).unwrap();

        assert!(a3.is_classical());
        assert!(!a3.is_exceptional());

        assert!(e8.is_exceptional());
        assert!(!e8.is_classical());
    }

    #[test]
    fn test_affine_creation() {
        let a3_aff = CartanType::new_affine(CartanLetter::A, 3, 1).unwrap();
        assert_eq!(a3_aff.letter, CartanLetter::A);
        assert_eq!(a3_aff.rank, 3);
        assert!(a3_aff.is_affine());
        assert!(!a3_aff.is_finite());
        assert_eq!(a3_aff.affinity_level(), 1);

        let b4_aff = CartanType::new_affine(CartanLetter::B, 4, 1).unwrap();
        assert!(b4_aff.is_affine());
        assert_eq!(b4_aff.affinity_level(), 1);
    }

    #[test]
    fn test_affine_display() {
        let a3_aff = CartanType::new_affine(CartanLetter::A, 3, 1).unwrap();
        assert_eq!(format!("{}", a3_aff), "A_3^(1)");

        let e6_aff = CartanType::new_affine(CartanLetter::E, 6, 1).unwrap();
        assert_eq!(format!("{}", e6_aff), "E_6^(1)");
    }

    #[test]
    fn test_affine_from_str() {
        let a3_aff: CartanType = "A3^(1)".parse().unwrap();
        assert_eq!(a3_aff.letter, CartanLetter::A);
        assert_eq!(a3_aff.rank, 3);
        assert!(a3_aff.is_affine());
        assert_eq!(a3_aff.affinity_level(), 1);

        let b4_aff: CartanType = "B4^(1)".parse().unwrap();
        assert_eq!(b4_aff.letter, CartanLetter::B);
        assert_eq!(b4_aff.rank, 4);
        assert!(b4_aff.is_affine());

        // Test finite type still works
        let a3_finite: CartanType = "A3".parse().unwrap();
        assert!(a3_finite.is_finite());
        assert!(!a3_finite.is_affine());
    }

    #[test]
    fn test_affine_invalid() {
        // Invalid affinity notation
        assert!("A3^1".parse::<CartanType>().is_err());
        assert!("A3^(a)".parse::<CartanType>().is_err());

        // Affinity level 0 should fail
        assert!(CartanType::new_affine(CartanLetter::A, 3, 0).is_none());
    }
}
