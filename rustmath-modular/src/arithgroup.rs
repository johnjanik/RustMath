//! Arithmetic subgroups of SL(2, Z)
//!
//! This module implements arithmetic subgroups of the modular group SL(2, Z),
//! including the full modular group, congruence subgroups Gamma0(N), Gamma1(N),
//! and GammaH(N, H).

use rustmath_complex::Complex;
use rustmath_integers::Integer;
use std::fmt;

/// Greatest common divisor for u64 arguments.
fn gcd_u64(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd_u64(b, a % b) }
}

/// The inverse of `a` mod `n`, or `None` if `gcd(a, n) != 1`.
fn mod_inverse(a: u64, n: u64) -> Option<u64> {
    if n == 1 {
        return Some(0);
    }
    let (mut old_r, mut r) = (a as i128, n as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    if old_r != 1 {
        return None;
    }
    Some(old_s.rem_euclid(n as i128) as u64)
}

/// Euler's totient function φ(n), for small u64 arguments.
fn euler_phi(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    let mut m = n;
    let mut p = 2;
    while p * p <= m {
        if m % p == 0 {
            while m % p == 0 {
                m /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if m > 1 {
        result -= result / m;
    }
    result
}

/// Element of an arithmetic subgroup (2x2 matrix with integer entries)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithmeticSubgroupElement {
    /// Matrix entries [[a, b], [c, d]]
    pub a: Integer,
    pub b: Integer,
    pub c: Integer,
    pub d: Integer,
}

impl ArithmeticSubgroupElement {
    /// Create a new arithmetic subgroup element from four integers
    pub fn new(a: Integer, b: Integer, c: Integer, d: Integer) -> Self {
        ArithmeticSubgroupElement { a, b, c, d }
    }

    /// Create from i64 values
    pub fn from_i64(a: i64, b: i64, c: i64, d: i64) -> Self {
        ArithmeticSubgroupElement {
            a: Integer::from(a),
            b: Integer::from(b),
            c: Integer::from(c),
            d: Integer::from(d),
        }
    }

    /// Compute the determinant of the matrix
    pub fn determinant(&self) -> Integer {
        &self.a * &self.d - &self.b * &self.c
    }

    /// Check if this element is in SL(2, Z) (determinant = 1)
    pub fn is_sl2z(&self) -> bool {
        self.determinant() == Integer::one()
    }

    /// Check if this element is in GL(2, Z) (determinant = ±1)
    pub fn is_gl2z(&self) -> bool {
        let det = self.determinant();
        det == Integer::one() || det == -Integer::one()
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
        if det == Integer::one() {
            Some(ArithmeticSubgroupElement {
                a: self.d.clone(),
                b: -&self.b,
                c: -&self.c,
                d: self.a.clone(),
            })
        } else if det == -Integer::one() {
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
    pub fn act_on_complex(&self, z: &Complex) -> Option<Complex> {
        let a = self.a.to_string().parse::<f64>().ok()?;
        let b = self.b.to_string().parse::<f64>().ok()?;
        let c = self.c.to_string().parse::<f64>().ok()?;
        let d = self.d.to_string().parse::<f64>().ok()?;

        let numerator = Complex::new(a, 0.0) * z.clone() + Complex::new(b, 0.0);
        let denominator = Complex::new(c, 0.0) * z.clone() + Complex::new(d, 0.0);

        if denominator.abs() < 1e-10 {
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

        // Number of cusps of Gamma0(N) = sum_{d | N} phi(gcd(d, N/d)).
        // (The previous code summed gcd(d, N/d) itself, omitting the Euler
        // phi and overcounting, e.g. giving 4 instead of 3 for N = 4.)
        let mut count = 0;
        for d in 1..=n {
            if n % d == 0 {
                count += euler_phi(gcd_u64(d, n / d));
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
        let n = Integer::from(self.level);
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

    /// The exact number of cusps of Gamma1(N).
    ///
    /// The cusps of Gamma1(N) are the orbits of the pairs `(a, c)` in
    /// `(Z/N)^2` with `gcd(a, c, N) = 1` under `(a, c) -> (a + c, c)` and
    /// `(a, c) -> (-a, -c)` (Diamond-Shurman Prop. 3.8.3).  Counting those
    /// orbits gives
    ///
    /// ```text
    ///     eps_inf(Gamma1(N)) = (1/2) sum_{d | N} phi(d) phi(N/d)      (N >= 5)
    /// ```
    ///
    /// with the four small levels exceptional (the +/- identification is not
    /// free there): 1, 2, 2, 3 for N = 1, 2, 3, 4.  Note N = 3 is NOT exceptional
    /// -- the sum already gives 2 -- but N = 1, 2, 4 are.
    ///
    /// The tests re-derive this by literally counting the orbits above, for every
    /// N <= 40.  (This used to return `index / 6`, e.g. 8 instead of 6 at
    /// N = 7.)
    pub fn compute_cusp_count(&self) -> u64 {
        let n = self.level;
        match n {
            1 => 1,
            2 => 2,
            4 => 3,
            _ => {
                let mut total = 0u64;
                for d in 1..=n {
                    if n.is_multiple_of(d) {
                        total += euler_phi(d) * euler_phi(n / d);
                    }
                }
                debug_assert!(total.is_multiple_of(2));
                total / 2
            }
        }
    }
}

impl ArithmeticSubgroup for Gamma1 {
    fn contains(&self, element: &ArithmeticSubgroupElement) -> bool {
        if !element.is_sl2z() {
            return false;
        }
        let n = Integer::from(self.level);
        // Check c ≡ 0 (mod N) and a ≡ d ≡ 1 (mod N)
        (&element.c % &n).is_zero()
            && (&element.a % &n) == Integer::one()
            && (&element.d % &n) == Integer::one()
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
        self.compute_cusp_count()
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
    /// `GammaH(N, H)` where `H` is the subgroup of `(Z/NZ)*` GENERATED by the
    /// given elements.
    ///
    /// The generators are reduced mod `N` and closed under multiplication: a
    /// caller who hands in a set that is not already a subgroup gets the subgroup
    /// it generates (this is what SageMath does), and `Gamma_H` is then a group.
    /// Previously the list was stored verbatim, so a non-closed list gave a
    /// `contains` predicate that did not define a group at all -- and every
    /// count below depends on `|H|`.
    pub fn new(level: u64, h_subgroup: Vec<u64>) -> Self {
        assert!(level > 0, "Level must be positive");
        for &h in &h_subgroup {
            assert!(
                gcd_u64(h % level, level) == 1,
                "GammaH: {h} is not a unit mod {level}"
            );
        }

        // Close the generators under multiplication mod N (N = 1: the unit group
        // is trivial, represented by {0}).
        let mut elements = vec![1 % level];
        let gens: Vec<u64> = h_subgroup.iter().map(|&h| h % level).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for &g in &gens {
                for i in 0..elements.len() {
                    let p = (elements[i] * g) % level;
                    if !elements.contains(&p) {
                        elements.push(p);
                        changed = true;
                    }
                }
            }
        }
        elements.sort_unstable();

        GammaH {
            level,
            h_subgroup: elements,
        }
    }

    /// The elements of `H` (the full subgroup, not just the given generators).
    pub fn h_elements(&self) -> &[u64] {
        &self.h_subgroup
    }

    /// The index `[SL(2,Z) : Gamma_H(N)]`.
    ///
    /// `Gamma_1(N) <= Gamma_H(N) <= Gamma_0(N)` and `gamma |-> a mod N` is a
    /// surjection `Gamma_H(N) -> H` with kernel `Gamma_1(N)`, so
    /// `[Gamma_H : Gamma_1] = |H|` and hence
    /// `[SL(2,Z) : Gamma_H] = [SL(2,Z) : Gamma_1(N)] / |H|`.
    /// (This is exact; `index()` used to return `None`.)
    pub fn compute_index(&self) -> u64 {
        Gamma1::new(self.level).compute_index() / (self.h_subgroup.len() as u64)
    }

    /// The exact number of cusps of `Gamma_H(N)`.
    ///
    /// The cusps of a group `G` with `Gamma(N) <= G <= SL(2,Z)` are the double
    /// cosets `G \ SL(2,Z) / Gamma_inf`, and reducing mod `N` this is the set of
    /// orbits of the primitive column vectors `(a, c)` in `(Z/N)^2`
    /// (`gcd(a, c, N) = 1`, which are exactly the first columns of `SL(2, Z/N)`)
    /// under the image of `G` on the left and of `Gamma_inf` on the right.  For
    /// `G = Gamma_H(N)` that image is `{[[h, b], [0, h^-1]] : h in H, b in Z/N}`,
    /// so the orbits are generated by
    ///
    /// ```text
    ///     (a, c) -> (a + b c, c)      (b in Z/N)
    ///     (a, c) -> (h a, h^-1 c)     (h in H)
    ///     (a, c) -> (-a, -c)          (-I in Gamma_inf)
    /// ```
    ///
    /// Counting those orbits is what this does -- it is the definition, not a
    /// formula, so it is correct for every `H` (it used to return a flat `1`).
    /// It costs `O(N^2 |H|)`; that is fine at the levels this crate works at.
    ///
    /// The tests pin the two ends against the independently-certified counts:
    /// `H = {1}` must reproduce `Gamma1::cusp_count` and `H = (Z/NZ)*` must
    /// reproduce `Gamma0::cusp_count`.
    pub fn compute_cusp_count(&self) -> u64 {
        let n = self.level;
        if n == 1 {
            return 1;
        }
        let mut seen = std::collections::HashSet::new();
        let mut orbits = 0u64;
        for a0 in 0..n {
            for c0 in 0..n {
                if gcd_u64(gcd_u64(a0, c0), n) != 1 || seen.contains(&(a0, c0)) {
                    continue;
                }
                orbits += 1;
                let mut stack = vec![(a0, c0)];
                while let Some((a, c)) = stack.pop() {
                    if !seen.insert((a, c)) {
                        continue;
                    }
                    // (a, c) -> (a + b c, c) for every b mod N
                    for b in 0..n {
                        stack.push(((a + b * c) % n, c));
                    }
                    // (a, c) -> (h a, h^{-1} c) for h in H
                    for &h in &self.h_subgroup {
                        let h_inv = mod_inverse(h, n).expect("H consists of units mod N");
                        stack.push(((a * h) % n, (c * h_inv) % n));
                    }
                    // (a, c) -> (-a, -c)
                    stack.push(((n - a) % n, (n - c) % n));
                }
            }
        }
        orbits
    }

    /// Create Gamma0(N) as GammaH(N, (Z/NZ)*)
    pub fn gamma0(level: u64) -> Self {
        let mut h = Vec::new();
        for i in 1..level {
            if gcd_u64(i, level) == 1 {
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
        let n = Integer::from(self.level);
        // Check c ≡ 0 (mod N)
        if !(&element.c % &n).is_zero() {
            return false;
        }
        // Check a ≡ d (mod N) and both in H
        if (&element.a % &n) != (&element.d % &n) {
            return false;
        }

        // Check if a (mod N) is in H
        let a_mod = (&(&(&element.a % &n) + &n) % &n).to_string().parse::<u64>().unwrap_or(0);
        self.h_subgroup.contains(&a_mod)
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
        self.compute_cusp_count()
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

    /// `J_2(N) = N^2 prod_{p | N} (1 - 1/p^2)`, the Jordan totient: the number of
    /// PRIMITIVE vectors `(a, c)` in `(Z/N)^2`, i.e. those with `gcd(a, c, N) = 1`.
    /// It is also `|SL(2, Z/N)| / N`.
    fn jordan_totient_2(&self) -> u64 {
        let n = self.level;
        let mut result = n * n;
        let mut m = n;
        let mut p = 2;
        while p * p <= m {
            if m % p == 0 {
                result -= result / (p * p);
                while m % p == 0 {
                    m /= p;
                }
            }
            p += 1;
        }
        if m > 1 {
            result -= result / (m * m);
        }
        result
    }

    /// The index `[SL(2,Z) : Gamma(N)] = |SL(2, Z/N)| = N^3 prod_{p|N} (1 - 1/p^2)`
    /// (reduction mod `N` is onto `SL(2, Z/N)` with kernel `Gamma(N)`).
    ///
    /// Exact for every `N >= 1`: 1, 6, 24, 48, 120, ... (`index()` used to return
    /// `None` "this is complex to compute exactly").
    pub fn compute_index(&self) -> u64 {
        self.level * self.jordan_totient_2()
    }

    /// The exact number of cusps of `Gamma(N)`.
    ///
    /// The cusps of `Gamma(N)` are `Gamma(N) \ SL(2,Z) / Gamma_inf`; reducing mod
    /// `N` (the image of `Gamma(N)` is trivial) they are the primitive vectors
    /// `(a, c)` in `(Z/N)^2` -- the first columns of `SL(2, Z/N)`, the right
    /// `Gamma_inf`-cosets being determined by the first column -- taken modulo the
    /// single identification `(a, c) ~ (-a, -c)` coming from `-I in Gamma_inf`.
    /// Hence
    ///
    /// ```text
    ///     eps_inf(Gamma(N)) = J_2(N) / 2 = (N^2 / 2) prod_{p|N} (1 - 1/p^2)   (N >= 3)
    /// ```
    ///
    /// The `-I` action is free exactly when `N >= 3` (a fixed point needs
    /// `2a = 2c = 0` with `(a, c)` primitive, which forces `N <= 2`), so `N = 1`
    /// and `N = 2` are exceptional and are counted directly: 1 and 3.
    ///
    /// Note this is `index / (2N)` for `N >= 3`, NOT `index / N`: `-I` is not in
    /// `Gamma(N)` there, so the index in `PSL(2,Z)` is half the index in
    /// `SL(2,Z)`, and each of the cusps has width `N`.  Values: 1, 3, 4, 6, 12,
    /// 12, 24, ... for `N = 1, 2, 3, 4, 5, 6, 7`.  (This used to return `N^2`:
    /// 4 instead of 3 at `N = 2`, 9 instead of 4 at `N = 3`, 25 instead of 12 at
    /// `N = 5`.)  The tests gate it against a brute-force orbit count.
    pub fn compute_cusp_count(&self) -> u64 {
        match self.level {
            1 => 1,
            2 => 3,
            _ => {
                let j2 = self.jordan_totient_2();
                debug_assert!(j2.is_multiple_of(2));
                j2 / 2
            }
        }
    }
}

impl ArithmeticSubgroup for Gamma {
    fn contains(&self, element: &ArithmeticSubgroupElement) -> bool {
        if !element.is_sl2z() {
            return false;
        }
        let n = Integer::from(self.level);
        // Check matrix ≡ I (mod N)
        (&(&element.a - &Integer::one()) % &n).is_zero()
            && (&element.b % &n).is_zero()
            && (&element.c % &n).is_zero()
            && (&(&element.d - &Integer::one()) % &n).is_zero()
    }

    fn level(&self) -> Option<u64> {
        Some(self.level)
    }

    fn index(&self) -> Option<u64> {
        Some(self.compute_index())
    }

    fn generators(&self) -> Vec<ArithmeticSubgroupElement> {
        vec![]
    }

    fn cusp_count(&self) -> u64 {
        self.compute_cusp_count()
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

    /// The cusps of Gamma1(N), counted by DEFINITION: the orbits of the pairs
    /// (a, c) in (Z/N)^2 with gcd(a, c, N) = 1 under (a, c) -> (a + c, c) and
    /// (a, c) -> (-a, -c) (Diamond-Shurman Prop. 3.8.3).
    fn brute_gamma1_cusps(n: u64) -> u64 {
        let mut seen = std::collections::HashSet::new();
        let mut orbits = 0u64;
        for a in 0..n {
            for c in 0..n {
                if gcd_u64(gcd_u64(a, c), n) != 1 || seen.contains(&(a, c)) {
                    continue;
                }
                orbits += 1;
                let mut stack = vec![(a, c)];
                while let Some((x, y)) = stack.pop() {
                    if !seen.insert((x, y)) {
                        continue;
                    }
                    stack.push(((x + y) % n, y % n));
                    stack.push(((n - x) % n, (n - y) % n));
                }
            }
        }
        orbits
    }

    /// GATE: the closed formula for the number of cusps of Gamma1(N) against the
    /// orbit count that defines it, for every N <= 40 -- including the
    /// exceptional small levels.  (`cusp_count` used to return `index / 6`, which
    /// gives 8 at N = 7 where the true count is 6.)
    #[test]
    fn test_gamma1_cusp_count_against_brute_force() {
        for n in 1..=40u64 {
            assert_eq!(
                Gamma1::new(n).cusp_count(),
                brute_gamma1_cusps(n),
                "#cusps(Gamma1({n}))"
            );
        }
        // the classical small values
        for (n, c) in [(1u64, 1u64), (2, 2), (3, 2), (4, 3), (5, 4), (7, 6), (11, 10)] {
            assert_eq!(Gamma1::new(n).cusp_count(), c, "#cusps(Gamma1({n}))");
        }
        // the old approximation index/6 was simply wrong
        assert_ne!(Gamma1::new(7).compute_index() / 6, 6);
    }

    /// GATE: the number of cusps of Gamma0(N) against `dims::gamma0_invariants`,
    /// which derives it multiplicatively and is itself brute-force certified.
    #[test]
    fn test_gamma0_cusp_count_agrees_with_dims() {
        for n in 1..=60u64 {
            let inv = crate::dims::gamma0_invariants(n).unwrap();
            assert_eq!(
                Gamma0::new(n).cusp_count() as u128,
                inv.cusps,
                "#cusps(Gamma0({n}))"
            );
            assert_eq!(
                Gamma0::new(n).compute_index() as u128,
                inv.index,
                "psi({n})"
            );
        }
    }

    #[test]
    fn test_arithmetic_subgroup_element_basic() {
        let e = ArithmeticSubgroupElement::from_i64(1, 0, 0, 1);
        assert_eq!(e.determinant(), Integer::one());
        assert!(e.is_sl2z());
        assert!(e.is_gl2z());
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = ArithmeticSubgroupElement::from_i64(1, 1, 0, 1);
        let b = ArithmeticSubgroupElement::from_i64(1, 0, 1, 1);
        let c = a.multiply(&b);
        assert_eq!(c.a, Integer::from(2));
        assert_eq!(c.b, Integer::from(1));
        assert_eq!(c.c, Integer::from(1));
        assert_eq!(c.d, Integer::from(1));
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

        // [[4, 1], [3, 1]] has det = 4·1 - 1·3 = 1 (so it lies in SL(2,Z)),
        // and c=3≡0, a=4≡1, d=1≡1 (mod 3), so it's in Gamma1(3).
        // (The old [[1,2],[3,4]] had det -2 and was never in SL(2,Z).)
        let m = ArithmeticSubgroupElement::from_i64(4, 1, 3, 1);
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

        // [[3, 2], [4, 3]] has det = 3·3 - 2·4 = 1 (so it lies in SL(2,Z)),
        // and ≡ [[1, 0], [0, 1]] (mod 2), so it's in Gamma(2).
        // (The old [[3,2],[2,3]] had det 5 and was never in SL(2,Z).)
        let m = ArithmeticSubgroupElement::from_i64(3, 2, 4, 3);
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

    /// The cusps of Gamma(N), counted by DEFINITION: the primitive vectors (a, c)
    /// in (Z/N)^2 -- the first columns of SL(2, Z/N), which index the right
    /// Gamma_inf-cosets -- modulo the single identification (a, c) -> (-a, -c)
    /// coming from -I in Gamma_inf.
    fn brute_gamma_cusps(n: u64) -> u64 {
        let mut seen = std::collections::HashSet::new();
        let mut orbits = 0u64;
        for a in 0..n {
            for c in 0..n {
                if gcd_u64(gcd_u64(a, c), n) != 1 || seen.contains(&(a, c)) {
                    continue;
                }
                orbits += 1;
                seen.insert((a, c));
                seen.insert(((n - a) % n, (n - c) % n));
            }
        }
        orbits
    }

    /// |SL(2, Z/N)|, counted by definition: the 2x2 matrices mod N of determinant 1.
    fn brute_sl2_order(n: u64) -> u64 {
        let mut count = 0u64;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        if (a * d + n * n - b * c) % n == 1 % n {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// GATE: the closed formula for the number of cusps of Gamma(N) against the
    /// orbit count that defines it, for every N <= 16, and the index against a
    /// literal count of SL(2, Z/N).
    ///
    /// `Gamma::cusp_count` used to return N^2: 4 instead of 3 at N = 2, 9 instead
    /// of 4 at N = 3, 25 instead of 12 at N = 5.
    #[test]
    fn test_gamma_cusp_count_against_brute_force() {
        for n in 1..=16u64 {
            assert_eq!(
                Gamma::new(n).cusp_count(),
                brute_gamma_cusps(n),
                "#cusps(Gamma({n}))"
            );
        }
        // the classical small values (PARI-checked)
        for (n, c) in [
            (1u64, 1u64),
            (2, 3),
            (3, 4),
            (4, 6),
            (5, 12),
            (6, 12),
            (7, 24),
            (12, 48),
        ] {
            assert_eq!(Gamma::new(n).cusp_count(), c, "#cusps(Gamma({n}))");
        }
        // the old approximation N^2 was simply wrong
        assert_ne!(Gamma::new(3).cusp_count(), 9);
        assert_ne!(Gamma::new(5).cusp_count(), 25);

        for n in 1..=10u64 {
            assert_eq!(
                Gamma::new(n).compute_index(),
                brute_sl2_order(n),
                "[SL(2,Z) : Gamma({n})] = |SL(2, Z/{n})|"
            );
        }
        // For N >= 3 the count is index/(2N) -- NOT index/N: -I is not in
        // Gamma(N) there, so the index in PSL(2,Z) is half of it.
        for n in 3..=16u64 {
            assert_eq!(
                Gamma::new(n).cusp_count(),
                Gamma::new(n).compute_index() / (2 * n),
                "eps_inf(Gamma({n})) = index / 2N"
            );
        }
    }

    /// GATE: GammaH's orbit count must reproduce the two ends it interpolates --
    /// H = {1} is Gamma1(N) and H = (Z/NZ)* is Gamma0(N) -- both of which are
    /// independently certified above.  It used to return a flat 1 for every H.
    #[test]
    fn test_gammah_cusp_count_and_index_at_both_ends() {
        for n in 1..=20u64 {
            assert_eq!(
                GammaH::gamma1(n).cusp_count(),
                Gamma1::new(n).cusp_count(),
                "GammaH(N, {{1}}) = Gamma1({n}): cusps"
            );
            assert_eq!(
                GammaH::gamma0(n).cusp_count(),
                Gamma0::new(n).cusp_count(),
                "GammaH(N, (Z/NZ)*) = Gamma0({n}): cusps"
            );
            assert_eq!(
                GammaH::gamma1(n).compute_index(),
                Gamma1::new(n).compute_index(),
                "GammaH(N, {{1}}) = Gamma1({n}): index"
            );
            assert_eq!(
                GammaH::gamma0(n).compute_index(),
                Gamma0::new(n).compute_index(),
                "GammaH(N, (Z/NZ)*) = Gamma0({n}): index"
            );
        }
    }

    /// The generators handed to `GammaH::new` are closed into the subgroup they
    /// generate; |H| then divides phi(N) (Lagrange) and drives the index.
    #[test]
    fn test_gammah_generates_its_subgroup() {
        // <5> = {1, 5} mod 12 (5^2 = 25 = 1)
        assert_eq!(GammaH::new(12, vec![5]).h_elements(), &[1, 5]);
        // <5, 7> = {1, 5, 7, 11} = all of (Z/12Z)*
        assert_eq!(GammaH::new(12, vec![5, 7]).h_elements(), &[1, 5, 7, 11]);
        assert_eq!(
            GammaH::new(12, vec![5, 7]).cusp_count(),
            Gamma0::new(12).cusp_count()
        );
        // <2> = {1, 2, 4} mod 7
        assert_eq!(GammaH::new(7, vec![2]).h_elements(), &[1, 2, 4]);
        // index [SL2(Z) : GammaH] = [SL2(Z) : Gamma1(N)] / |H|
        assert_eq!(
            GammaH::new(7, vec![2]).compute_index(),
            Gamma1::new(7).compute_index() / 3
        );
        // Gamma1(N) <= Gamma_H(N) <= Gamma0(N), so the cusp counts are sandwiched
        for n in [7u64, 12, 13, 16] {
            for h in 2..n {
                if gcd_u64(h, n) != 1 {
                    continue;
                }
                let gh = GammaH::new(n, vec![h]);
                assert!(gh.cusp_count() <= Gamma1::new(n).cusp_count());
                assert!(gh.cusp_count() >= Gamma0::new(n).cusp_count());
                assert_eq!(
                    Gamma1::new(n).compute_index() % (gh.h_elements().len() as u64),
                    0
                );
            }
        }
    }
}
