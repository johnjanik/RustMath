//! Weight lattice and weight operations
//!
//! Weights are elements of the weight lattice, which is a free abelian group.
//! In type A_n, weights can be represented as vectors of integers.

use std::fmt;
use std::ops::{Add, Sub, Neg, Mul};

/// A weight in the weight lattice
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Weight {
    /// Coordinates of the weight in the fundamental weight basis
    pub coords: Vec<i64>,
}

impl Weight {
    /// Create a new weight from coordinates
    pub fn new(coords: Vec<i64>) -> Self {
        Weight { coords }
    }

    /// Create the zero weight
    pub fn zero(rank: usize) -> Self {
        Weight {
            coords: vec![0; rank],
        }
    }

    /// Create a fundamental weight
    pub fn fundamental(index: usize, rank: usize) -> Self {
        let mut coords = vec![0; rank];
        if index < rank {
            coords[index] = 1;
        }
        Weight { coords }
    }

    /// Get the rank (dimension) of the weight
    pub fn rank(&self) -> usize {
        self.coords.len()
    }

    /// Inner product with another weight
    pub fn dot(&self, other: &Weight) -> i64 {
        assert_eq!(self.coords.len(), other.coords.len());
        self.coords.iter().zip(&other.coords).map(|(a, b)| a * b).sum()
    }

    /// Check if this weight is dominant (all coordinates >= 0)
    pub fn is_dominant(&self) -> bool {
        self.coords.iter().all(|&x| x >= 0)
    }

    /// Level of the weight (sum of coordinates)
    pub fn level(&self) -> i64 {
        self.coords.iter().sum()
    }
}

impl Add for Weight {
    type Output = Weight;

    fn add(self, other: Weight) -> Weight {
        assert_eq!(self.coords.len(), other.coords.len());
        Weight {
            coords: self.coords.iter().zip(&other.coords).map(|(a, b)| a + b).collect(),
        }
    }
}

impl Add for &Weight {
    type Output = Weight;

    fn add(self, other: &Weight) -> Weight {
        assert_eq!(self.coords.len(), other.coords.len());
        Weight {
            coords: self.coords.iter().zip(&other.coords).map(|(a, b)| a + b).collect(),
        }
    }
}

impl Sub for Weight {
    type Output = Weight;

    fn sub(self, other: Weight) -> Weight {
        assert_eq!(self.coords.len(), other.coords.len());
        Weight {
            coords: self.coords.iter().zip(&other.coords).map(|(a, b)| a - b).collect(),
        }
    }
}

impl Sub for &Weight {
    type Output = Weight;

    fn sub(self, other: &Weight) -> Weight {
        assert_eq!(self.coords.len(), other.coords.len());
        Weight {
            coords: self.coords.iter().zip(&other.coords).map(|(a, b)| a - b).collect(),
        }
    }
}

impl Neg for Weight {
    type Output = Weight;

    fn neg(self) -> Weight {
        Weight {
            coords: self.coords.iter().map(|x| -x).collect(),
        }
    }
}

impl Neg for &Weight {
    type Output = Weight;

    fn neg(self) -> Weight {
        Weight {
            coords: self.coords.iter().map(|x| -x).collect(),
        }
    }
}

impl Mul<i64> for Weight {
    type Output = Weight;

    fn mul(self, scalar: i64) -> Weight {
        Weight {
            coords: self.coords.iter().map(|x| x * scalar).collect(),
        }
    }
}

impl Mul<i64> for &Weight {
    type Output = Weight;

    fn mul(self, scalar: i64) -> Weight {
        Weight {
            coords: self.coords.iter().map(|x| x * scalar).collect(),
        }
    }
}

impl fmt::Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, coord) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_operations() {
        let w1 = Weight::new(vec![1, 2, 3]);
        let w2 = Weight::new(vec![2, -1, 1]);

        let sum = &w1 + &w2;
        assert_eq!(sum.coords, vec![3, 1, 4]);

        let diff = &w1 - &w2;
        assert_eq!(diff.coords, vec![-1, 3, 2]);

        let neg = -&w1;
        assert_eq!(neg.coords, vec![-1, -2, -3]);

        let scaled = &w1 * 2;
        assert_eq!(scaled.coords, vec![2, 4, 6]);
    }

    #[test]
    fn test_fundamental_weights() {
        let w0 = Weight::fundamental(0, 3);
        assert_eq!(w0.coords, vec![1, 0, 0]);

        let w1 = Weight::fundamental(1, 3);
        assert_eq!(w1.coords, vec![0, 1, 0]);

        let w2 = Weight::fundamental(2, 3);
        assert_eq!(w2.coords, vec![0, 0, 1]);
    }

    #[test]
    fn test_dominant() {
        assert!(Weight::new(vec![1, 2, 0]).is_dominant());
        assert!(!Weight::new(vec![1, -1, 0]).is_dominant());
    }
}
