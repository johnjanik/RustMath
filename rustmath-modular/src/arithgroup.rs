//! Arithmetic subgroups of SL(2, Z)
//!
//! This module implements arithmetic subgroups of the modular group SL(2, Z),
//! including the full modular group, congruence subgroups Gamma0(N), Gamma1(N),
//! and GammaH(N, H).

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{Zero, One, Signed};
use std::fmt;

/// Element of an arithmetic subgroup (2x2 matrix with integer entries)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithmeticSubgroupElement {
    /// Matrix entries [[a, b], [c, d]]
    pub a: BigInt,
    pub b: BigInt,
    pub c: BigInt,
    pub d: BigInt,
}

impl ArithmeticSubgroupElement {
    /// Create a new arithmetic subgroup element from four integers
    pub fn new(a: BigInt, b: BigInt, c: BigInt, d: BigInt) -> Self {
        ArithmeticSubgroupElement { a, b, c, d }
    }

    /// Create from i64 values
    pub fn from_i64(a: i64, b: i64, c: i64, d: i64) -> Self {
        ArithmeticSubgroupElement {
            a: BigInt::from(a),
            b: BigInt::from(b),
            c: BigInt::from(c),
            d: BigInt::from(d),
        }
    }

    /// Compute the determinant of the matrix
    pub fn determinant(&self) -> BigInt {
        &self.a * &self.d - &self.b * &self.c
    }

    /// Check if this element is in SL(2, Z) (determinant = 1)
    pub fn is_sl2z(&self) -> bool {
        self.determinant() == BigInt::one()
    }

    /// Check if this element is in GL(2, Z) (determinant = ±1)
    pub fn is_gl2z(&self) -> bool {
        let det = self.determinant();
        det == BigInt::one() || det == -BigInt::one()
    }

    /// Matrix multiplication
    pub fn multiply(&self, other: &ArithmeticSubgroupElement) -> ArithmeticSubgroupElement {
        ArithmeticSubgroupElement {
            a: &self.a * &other.a + &self.b * &other.c,
            b: &self.a * &other.b + &self.b * &other.d,
            c: &self.c * &other.a + &self.d * &other.c,
            d: &self.c * &other.b + &self.d * &other.d,
        }
    }

    /// Compute the inverse (only for det = ±1)
    pub fn inverse(&self) -> Option<ArithmeticSubgroupElement> {
        let det = self.determinant();
        if det == BigInt::one() {
            Some(ArithmeticSubgroupElement {
                a: self.d.clone(),
                b: -&self.b,
                c: -&self.c,
                d: self.a.clone(),
            })
        } else if det == -BigInt::one() {
            Some(ArithmeticSubgroupElement {
                a: -&self.d,
                b: self.b.clone(),
                c: self.c.clone(),
                d: -&self.a,
            })
        } else {
            None
        }
    }

    /// Identity matrix
    pub fn identity() -> Self {
        ArithmeticSubgroupElement::from_i64(1, 0, 0, 1)
    }

    /// Apply the Mobius transformation z -> (az + b)/(cz + d)
    pub fn act_on_complex(&self, z: &num_complex::Complex<f64>) -> Option<num_complex::Complex<f64>> {
        use num_complex::Complex;

        let a = self.a.to_string().parse::<f64>().ok()?;
        let b = self.b.to_string().parse::<f64>().ok()?;
        let c = self.c.to_string().parse::<f64>().ok()?;
        let d = self.d.to_string().parse::<f64>().ok()?;

        let numerator = Complex::new(a, 0.0) * z + Complex::new(b, 0.0);
        let denominator = Complex::new(c, 0.0) * z + Complex::new(d, 0.0);

        if denominator.norm() < 1e-10 {
            None
        } else {
            Some(numerator / denominator)
        }
    }
}

impl fmt::Display for ArithmeticSubgroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[[{}, {}], [{}, {}]]", self.a, self.b, self.c, self.d)
    }
}

/// Trait for arithmetic subgroups of SL(2, Z)
pub trait ArithmeticSubgroup {
    /// Check if an element is in this subgroup
    fn contains(&self, element: &ArithmeticSubgroupElement) -> bool;

    /// Return the level of the subgroup (if it's a congruence subgroup)
    fn level(&self) -> Option<u64>;

    /// Return the index of the subgroup in SL(2, Z)
    fn index(&self) -> Option<u64>;

    /// Check if this is a congruence subgroup
    fn is_congruence(&self) -> bool {
        self.level().is_some()
    }

    /// Return generators of the subgroup
    fn generators(&self) -> Vec<ArithmeticSubgroupElement>;

    /// Number of cusps
    fn cusp_count(&self) -> u64;
}

/// Trait for congruence subgroups
pub trait CongruenceSubgroup: ArithmeticSubgroup {
    /// Return the level N
    fn get_level(&self) -> u64;
}

/// The full modular group SL(2, Z)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SL2Z;

impl SL2Z {
    pub fn new() -> Self {
        SL2Z
    }

    /// Standard generators S and T of SL(2, Z)
    /// S = [[0, -1], [1, 0]]  (order 4)
    /// T = [[1, 1], [0, 1]]   (order infinity)
    pub fn standard_generators() -> (ArithmeticSubgroupElement, ArithmeticSubgroupElement) {
        let s = ArithmeticSubgroupElement::from_i64(0, -1, 1, 0);
        let t = ArithmeticSubgroupElement::from_i64(1, 1, 0, 1);
        (s, t)
    }
}

impl Default for SL2Z {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithmeticSubgroup for SL2Z {
    fn contains(&self, element: &ArithmeticSubgroupElement) -> bool {
        element.is_sl2z()
    }

    fn level(&self) -> Option<u64> {
        Some(1)
    }

    fn index(&self) -> Option<u64> {
        Some(1)
    }

    fn generators(&self) -> Vec<ArithmeticSubgroupElement> {
        let (s, t) = Self::standard_generators();
        vec![s, t]
    }

    fn cusp_count(&self) -> u64 {
        1
    }
}

/// The congruence subgroup Gamma0(N)
/// Consists of matrices [[a, b], [c, d]] in SL(2, Z) with c ≡ 0 (mod N)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gamma0 {
    level: u64,
}

impl Gamma0 {
    pub fn new(level: u64) -> Self {
        assert!(level > 0, "Level must be positive");
        Gamma0 { level }
    }

    /// Compute the index [SL(2,Z) : Gamma0(N)]
    pub fn compute_index(&self) -> u64 {
        if self.level == 1 {
            return 1;
        }

        // Formula: N * prod_{p|N} (1 + 1/p)
        let n = self.level;
        let mut result = n;
        let mut temp_n = n;
        let mut p = 2;

        while p * p <= temp_n {
            if temp_n % p == 0 {
                result += result / p;
                while temp_n % p == 0 {
                    temp_n /= p;
                }
            }
            p += 1;
        }
        if temp_n > 1 {
            result += result / temp_n;
        }

        result
    }

    /// Compute number of cusps for Gamma0(N)
    pub fn compute_cusp_count(&self) -> u64 {
        let n = self.level;
        if n == 1 {
            return 1;
        }

        // Number of cusps = sum_{d | N} gcd(d, N/d)
        let mut count = 0;
        for d in 1..=n {
            if n % d == 0 {
                count += Integer::gcd(&d, &(n / d));
            }
        }
        count
    }
}

impl ArithmeticSubgroup for Gamma0 {
    fn contains(&self, element: &ArithmeticSubgroupElement) -> bool {
        if !element.is_sl2z() {
            return false;
        }
        // Check if c ≡ 0 (mod N)
        let n = BigInt::from(self.level);
        (&element.c % &n).is_zero()
    }

    fn level(&self) -> Option<u64> {
        Some(self.level)
    }

    fn index(&self) -> Option<u64> {
        Some(self.compute_index())
    }

    fn generators(&self) -> Vec<ArithmeticSubgroupElement> {
        // For Gamma0(N), standard generators include:
        // S = [[0, -1], [1, 0]]
        // T = [[1, 1], [0, 1]]
        // and additional elements depending on N
        let s = ArithmeticSubgroupElement::from_i64(0, -1, 1, 0);
        let t = ArithmeticSubgroupElement::from_i64(1, 1, 0, 1);
        vec![s, t]
    }

    fn cusp_count(&self) -> u64 {
        self.compute_cusp_count()
    }
}

impl CongruenceSubgroup for Gamma0 {
    fn get_level(&self) -> u64 {
        self.level
    }
}

/// The congruence subgroup Gamma1(N)
/// Consists of matrices [[a, b], [c, d]] in SL(2, Z) with c ≡ 0 (mod N) and a ≡ d ≡ 1 (mod N)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gamma1 {
    level: u64,
}

impl Gamma1 {
    pub fn new(level: u64) -> Self {
        assert!(level > 0, "Level must be positive");
        Gamma1 { level }
    }

    /// Compute the index [SL(2,Z) : Gamma1(N)]
    pub fn compute_index(&self) -> u64 {
        if self.level == 1 {
            return 1;
        }
        if self.level == 2 {
            return 3;
        }

        // Formula: N^2 * prod_{p|N} (1 - 1/p^2)
        let n = self.level;
        let mut result = n * n;
        let mut temp_n = n;
        let mut p = 2;

        while p * p <= temp_n {
            if temp_n % p == 0 {
                result -= result / (p * p);
                while temp_n % p == 0 {
                    temp_n /= p;
                }
            }
            p += 1;
        }
        if temp_n > 1 {
            result -= result / (temp_n * temp_n);
        }

        result
    }
}

impl ArithmeticSubgroup for Gamma1 {
    fn contains(&self, element: &ArithmeticSubgroupElement) -> bool {
        if !element.is_sl2z() {
            return false;
        }
        let n = BigInt::from(self.level);
        // Check c ≡ 0 (mod N) and a ≡ d ≡ 1 (mod N)
        (&element.c % &n).is_zero()
            && (&element.a % &n) == BigInt::one()
            && (&element.d % &n) == BigInt::one()
    }

    fn level(&self) -> Option<u64> {
        Some(self.level)
    }

    fn index(&self) -> Option<u64> {
        Some(self.compute_index())
    }

    fn generators(&self) -> Vec<ArithmeticSubgroupElement> {
        let s = ArithmeticSubgroupElement::from_i64(0, -1, 1, 0);
        let t = ArithmeticSubgroupElement::from_i64(1, 1, 0, 1);
        vec![s, t]
    }

    fn cusp_count(&self) -> u64 {
        // Number of cusps for Gamma1(N)
        if self.level == 1 {
            1
        } else if self.level == 2 {
            2
        } else {
            // This is an approximation; exact formula is more complex
            self.compute_index() / 6
        }
    }
}

impl CongruenceSubgroup for Gamma1 {
    fn get_level(&self) -> u64 {
        self.level
    }
}

/// The congruence subgroup GammaH(N, H)
/// Generalization of Gamma1(N) with a subgroup H of (Z/NZ)*
#[derive(Debug, Clone)]
pub struct GammaH {
    level: u64,
    h_subgroup: Vec<u64>, // Elements of H
}

impl GammaH {
    pub fn new(level: u64, h_subgroup: Vec<u64>) -> Self {
        assert!(level > 0, "Level must be positive");
        // Verify all elements are in (Z/NZ)*
        for &h in &h_subgroup {
            assert!(h < level && Integer::gcd(&h, &level) == 1);
        }
        GammaH { level, h_subgroup }
    }

    /// Create Gamma0(N) as GammaH(N, (Z/NZ)*)
    pub fn gamma0(level: u64) -> Self {
        let mut h = Vec::new();
        for i in 1..level {
            if Integer::gcd(&i, &level) == 1 {
                h.push(i);
            }
        }
        GammaH::new(level, h)
    }

    /// Create Gamma1(N) as GammaH(N, {1})
    pub fn gamma1(level: u64) -> Self {
        GammaH::new(level, vec![1])
    }
}

impl ArithmeticSubgroup for GammaH {
    fn contains(&self, element: &ArithmeticSubgroupElement) -> bool {
        if !element.is_sl2z() {
            return false;
        }
        let n = BigInt::from(self.level);
        // Check c ≡ 0 (mod N)
        if !(&element.c % &n).is_zero() {
            return false;
        }
        // Check a ≡ d (mod N) and both in H
        if (&element.a % &n) != (&element.d % &n) {
            return false;
        }

        // Check if a (mod N) is in H
        let a_mod = ((&element.a % &n + &n) % &n).to_string().parse::<u64>().unwrap_or(0);
        self.h_subgroup.contains(&a_mod)
    }

    fn level(&self) -> Option<u64> {
        Some(self.level)
    }

    fn index(&self) -> Option<u64> {
        // Index depends on the size of H
        None // Computing this requires more sophisticated methods
    }

    fn generators(&self) -> Vec<ArithmeticSubgroupElement> {
        let s = ArithmeticSubgroupElement::from_i64(0, -1, 1, 0);
        let t = ArithmeticSubgroupElement::from_i64(1, 1, 0, 1);
        vec![s, t]
    }

    fn cusp_count(&self) -> u64 {
        // Depends on H; requires more sophisticated computation
        1
    }
}

impl CongruenceSubgroup for GammaH {
    fn get_level(&self) -> u64 {
        self.level
    }
}

/// The principal congruence subgroup Gamma(N)
/// Consists of matrices ≡ I (mod N)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gamma {
    level: u64,
}

impl Gamma {
    pub fn new(level: u64) -> Self {
        assert!(level > 0, "Level must be positive");
        Gamma { level }
    }
}

impl ArithmeticSubgroup for Gamma {
    fn contains(&self, element: &ArithmeticSubgroupElement) -> bool {
        if !element.is_sl2z() {
            return false;
        }
        let n = BigInt::from(self.level);
        // Check matrix ≡ I (mod N)
        (&element.a - BigInt::one()) % &n == BigInt::zero()
            && (&element.b % &n).is_zero()
            && (&element.c % &n).is_zero()
            && (&element.d - BigInt::one()) % &n == BigInt::zero()
    }

    fn level(&self) -> Option<u64> {
        Some(self.level)
    }

    fn index(&self) -> Option<u64> {
        if self.level == 1 {
            return Some(1);
        }
        // Index formula: N^3 * prod_{p|N} (1 - 1/p^2)
        // This is complex to compute exactly
        None
    }

    fn generators(&self) -> Vec<ArithmeticSubgroupElement> {
        vec![]
    }

    fn cusp_count(&self) -> u64 {
        // Many cusps for Gamma(N)
        if self.level == 1 {
            1
        } else {
            self.level * self.level // Approximation
        }
    }
}

impl CongruenceSubgroup for Gamma {
    fn get_level(&self) -> u64 {
        self.level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_subgroup_element_basic() {
        let e = ArithmeticSubgroupElement::from_i64(1, 0, 0, 1);
        assert_eq!(e.determinant(), BigInt::one());
        assert!(e.is_sl2z());
        assert!(e.is_gl2z());
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = ArithmeticSubgroupElement::from_i64(1, 1, 0, 1);
        let b = ArithmeticSubgroupElement::from_i64(1, 0, 1, 1);
        let c = a.multiply(&b);
        assert_eq!(c.a, BigInt::from(2));
        assert_eq!(c.b, BigInt::from(1));
        assert_eq!(c.c, BigInt::from(1));
        assert_eq!(c.d, BigInt::from(1));
    }

    #[test]
    fn test_matrix_inverse() {
        let a = ArithmeticSubgroupElement::from_i64(1, 1, 0, 1);
        let inv = a.inverse().unwrap();
        let identity = a.multiply(&inv);
        assert_eq!(identity, ArithmeticSubgroupElement::identity());
    }

    #[test]
    fn test_sl2z() {
        let sl2z = SL2Z::new();
        let identity = ArithmeticSubgroupElement::identity();
        assert!(sl2z.contains(&identity));

        let (s, t) = SL2Z::standard_generators();
        assert!(sl2z.contains(&s));
        assert!(sl2z.contains(&t));
        assert_eq!(sl2z.level(), Some(1));
        assert_eq!(sl2z.index(), Some(1));
    }

    #[test]
    fn test_gamma0() {
        let gamma0_2 = Gamma0::new(2);
        assert_eq!(gamma0_2.level(), Some(2));
        assert_eq!(gamma0_2.compute_index(), 3);

        // [[1, 0], [0, 1]] is in Gamma0(2)
        let identity = ArithmeticSubgroupElement::identity();
        assert!(gamma0_2.contains(&identity));

        // [[1, 1], [2, 3]] is in Gamma0(2) (c=2 is divisible by 2)
        let m = ArithmeticSubgroupElement::from_i64(1, 1, 2, 3);
        assert!(gamma0_2.contains(&m));

        // [[1, 1], [1, 2]] is NOT in Gamma0(2) (c=1 is not divisible by 2)
        let m2 = ArithmeticSubgroupElement::from_i64(1, 1, 1, 2);
        assert!(!gamma0_2.contains(&m2));
    }

    #[test]
    fn test_gamma1() {
        let gamma1_3 = Gamma1::new(3);
        assert_eq!(gamma1_3.level(), Some(3));

        // [[1, 0], [0, 1]] is in Gamma1(3)
        let identity = ArithmeticSubgroupElement::identity();
        assert!(gamma1_3.contains(&identity));

        // [[1, 2], [3, 4]] has c=3≡0, a=1≡1, d=4≡1 (mod 3), so it's in Gamma1(3)
        let m = ArithmeticSubgroupElement::from_i64(1, 2, 3, 4);
        assert!(gamma1_3.contains(&m));

        // [[2, 1], [3, 2]] has a=2≢1 (mod 3), so NOT in Gamma1(3)
        let m2 = ArithmeticSubgroupElement::from_i64(2, 1, 3, 2);
        assert!(!gamma1_3.contains(&m2));
    }

    #[test]
    fn test_gamma() {
        let gamma_2 = Gamma::new(2);
        assert_eq!(gamma_2.level(), Some(2));

        // Identity is in Gamma(2)
        let identity = ArithmeticSubgroupElement::identity();
        assert!(gamma_2.contains(&identity));

        // [[3, 2], [2, 3]] ≡ [[1, 0], [0, 1]] (mod 2), so it's in Gamma(2)
        let m = ArithmeticSubgroupElement::from_i64(3, 2, 2, 3);
        assert!(gamma_2.contains(&m));

        // [[1, 1], [0, 1]] has b=1≢0 (mod 2), so NOT in Gamma(2)
        let m2 = ArithmeticSubgroupElement::from_i64(1, 1, 0, 1);
        assert!(!gamma_2.contains(&m2));
    }

    #[test]
    fn test_cusp_count() {
        assert_eq!(Gamma0::new(1).cusp_count(), 1);
        assert_eq!(Gamma0::new(2).cusp_count(), 2);
        assert_eq!(Gamma0::new(3).cusp_count(), 2);
        assert_eq!(Gamma0::new(4).cusp_count(), 3);
    }
}
