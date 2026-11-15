//! Frobenius polynomials of curves over finite fields from point counts, and exact checks on
//! them.
//!
//! Specification: MAGMA Handbook ch. 114 *Algebraic Curves*, 114.8.1 (`LPolynomial`,
//! `ZetaFunction`, `NumberOfPlacesOfDegreeOneECF`; digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`) and ch. 42 *Algebraic Function Fields*
//! (the same intrinsics for function fields; `Chapter_42_Algebraic_Function_Fields.md`). MAGMA
//! computes these for any curve; this module only turns point counts into the polynomial and
//! checks it. The counts come from elsewhere (for the M23 curve:
//! [`crate::nodal_biprojective::normalization_point_counts`]).
//!
//! Conventions. For a smooth projective curve of genus `g` over `F_p`,
//! `Z(T) = L(T) / ((1 - T)(1 - pT))` with `L(T) = prod (1 - alpha_i T)` of degree `2g` (MAGMA
//! `LPolynomial`, ascending) and `P(T) = T^(2g) L(1/T) = prod (T - alpha_i)`, the characteristic
//! polynomial of Frobenius (descending, monic). `s_k = sum alpha_i^k = p^k + 1 - N_k`.
//!
//! Mathematics:
//! - \[Weil\] A. Weil, *Sur les courbes algebriques et les varietes qui s'en deduisent* (1948):
//!   `L` has degree `2g`, `L(T) = p^g T^(2g) L(1/(pT))`, and `|alpha_i| = sqrt p`.
//! - Newton's identities `m e_m = sum_{i=1..m} (-1)^(i-1) e_(m-i) s_i` (I. G. Macdonald,
//!   *Symmetric functions and Hall polynomials*, I.2).
//! - \[Sturm\] C. Sturm (1835); S. Basu, R. Pollack, M.-F. Roy, *Algorithms in real algebraic
//!   geometry*, Thm. 2.50: the real-root counts of `rustmath_polynomials::real_roots`.
//! - The real Weil polynomial: `P(T) = T^g R(T + p/T)` with `R` monic of degree `g` in `Z[x]`;
//!   the roots of `P` have absolute value `sqrt p` iff every root of `R` is real and lies in
//!   `[-2 sqrt p, 2 sqrt p]` (a root `x` of `R` gives the roots of `z^2 - x z + p`).
//! - The 2-torsion: `P mod 2` is the characteristic polynomial of Frobenius on
//!   `Jac[2] = T_2 Jac / 2` (J. Tate, *Endomorphisms of abelian varieties over finite fields*,
//!   1966, Thm. 1 and its proof; \[HJLPPZ\] and `docs/workstreams/m23_golay/HANDOFF.md` 3.1).

use rustmath_integers::Integer;
use rustmath_polynomials::{fp_factor, real_roots, zx};
use rustmath_rationals::Rational;
use std::fmt;

/// Why a Frobenius polynomial could not be formed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeilError {
    /// `p < 2`, or a coefficient list that is not monic of even degree.
    InvalidInput(String),
    /// Newton's identities gave a non-integral `e_index = numerator / denominator`: the counts
    /// are not those of a smooth curve of this genus over `F_p` (wrong counts or bad reduction).
    NonIntegral {
        index: usize,
        numerator: Integer,
        denominator: usize,
    },
    /// The coefficients do not satisfy the functional equation `a_(2g-i) = p^(g-i) a_i`.
    FunctionalEquation { index: usize },
}

impl fmt::Display for WeilError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeilError::InvalidInput(s) => write!(f, "invalid input: {s}"),
            WeilError::NonIntegral {
                index,
                numerator,
                denominator,
            } => write!(
                f,
                "Newton's identities give e_{index} = {numerator}/{denominator}, not an integer"
            ),
            WeilError::FunctionalEquation { index } => {
                write!(f, "functional equation fails at coefficient {index}")
            }
        }
    }
}

impl std::error::Error for WeilError {}

/// The order of Frobenius on `Jac[2]` as far as `P mod 2` decides it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TwoTorsionOrder {
    /// `P mod 2` is squarefree: Frobenius is semisimple on `Jac[2]` and its order is the lcm of
    /// the orders of `x` modulo the irreducible factors. This is a statement about `P`: it is the
    /// order on `Jac[2]` of the curve only when `P` is certified to be its Frobenius polynomial.
    DeterminedByP(u64),
    /// `P mod 2` has a repeated factor: Frobenius may have a unipotent part. Its order is a
    /// multiple of `semisimple_order` (the lcm over the distinct factors) and divides `divides`
    /// (`semisimple_order * 2^ceil(log2 e_max)`, `e_max` the largest multiplicity).
    Unresolved { semisimple_order: u64, divides: u64 },
}

/// The characteristic polynomial of Frobenius of a curve of genus `g` over `F_p`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrobeniusPolynomial {
    p: Integer,
    genus: usize,
    /// Descending: `coeffs[i]` is the coefficient of `T^(2g-i)`; `coeffs[0] = 1`,
    /// `coeffs[2g] = p^g`.
    coeffs: Vec<Integer>,
}

fn int(x: i64) -> Integer {
    Integer::from(x)
}

fn modulo(a: &Integer, m: &Integer) -> Integer {
    let r = a % m;
    if r.signum() < 0 {
        r + m.clone()
    } else {
        r
    }
}

impl FrobeniusPolynomial {
    /// `P` from `N_1, ..., N_g` (`g = counts.len()`), by Newton's identities, an integrality check
    /// of every `e_m`, and the functional equation for the upper half.
    pub fn from_point_counts(p: &Integer, counts: &[Integer]) -> Result<Self, WeilError> {
        if *p < int(2) {
            return Err(WeilError::InvalidInput(format!("p = {p} < 2")));
        }
        let g = counts.len();
        let s: Vec<Integer> = counts
            .iter()
            .enumerate()
            .map(|(k, n)| p.pow(k as u32 + 1) + Integer::one() - n.clone())
            .collect();
        let mut e: Vec<Integer> = vec![Integer::one()];
        for m in 1..=g {
            let mut acc = Integer::zero();
            for i in 1..=m {
                let t = e[m - i].clone() * s[i - 1].clone();
                acc = if (i - 1) % 2 == 0 { acc + t } else { acc - t };
            }
            let den = int(m as i64);
            if !(&acc % &den).is_zero() {
                return Err(WeilError::NonIntegral {
                    index: m,
                    numerator: acc,
                    denominator: m,
                });
            }
            e.push(acc / den);
        }
        for i in (g + 1)..=(2 * g) {
            let v = p.pow((i - g) as u32) * e[2 * g - i].clone();
            e.push(v);
        }
        let coeffs = e
            .into_iter()
            .enumerate()
            .map(|(i, x)| if i % 2 == 0 { x } else { -x })
            .collect();
        Ok(FrobeniusPolynomial {
            p: p.clone(),
            genus: g,
            coeffs,
        })
    }

    /// [`Self::from_point_counts`] for machine-word inputs.
    pub fn from_counts_u64(p: u64, counts: &[u64]) -> Result<Self, WeilError> {
        let c: Vec<Integer> = counts.iter().map(|&n| Integer::from(n)).collect();
        Self::from_point_counts(&Integer::from(p), &c)
    }

    /// `P` from its descending coefficients `[1, a_1, ..., a_(2g)]`; checks that the list is monic
    /// of even length minus one and satisfies the functional equation.
    pub fn from_coefficients(p: &Integer, coeffs: Vec<Integer>) -> Result<Self, WeilError> {
        if *p < int(2) {
            return Err(WeilError::InvalidInput(format!("p = {p} < 2")));
        }
        if coeffs.is_empty() || coeffs.len().is_multiple_of(2) || coeffs[0] != Integer::one() {
            return Err(WeilError::InvalidInput(
                "expected a monic polynomial of even degree".into(),
            ));
        }
        let g = (coeffs.len() - 1) / 2;
        for i in 0..g {
            // a_(2g-i) = p^(g-i) a_i
            if coeffs[2 * g - i] != p.pow((g - i) as u32) * coeffs[i].clone() {
                return Err(WeilError::FunctionalEquation { index: 2 * g - i });
            }
        }
        Ok(FrobeniusPolynomial {
            p: p.clone(),
            genus: g,
            coeffs,
        })
    }

    /// The prime `p`.
    pub fn p(&self) -> &Integer {
        &self.p
    }

    /// The genus `g` (`deg P = 2g`).
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Descending coefficients `[1, a_1, ..., a_(2g)]` of `P(T)`.
    pub fn coefficients(&self) -> &[Integer] {
        &self.coeffs
    }

    /// `L(T) = T^(2g) P(1/T)`, ascending (MAGMA `LPolynomial`): the same list as
    /// [`Self::coefficients`].
    pub fn l_polynomial(&self) -> Vec<Integer> {
        self.coeffs.clone()
    }

    /// `N_1, ..., N_kmax` predicted by `P`: `N_k = p^k + 1 - s_k` with `s_k` from Newton's
    /// identities (`e_i = 0` for `i > 2g`).
    pub fn point_counts(&self, kmax: usize) -> Vec<Integer> {
        let two_g = 2 * self.genus;
        // e_i = (-1)^i * coeffs[i]
        let e: Vec<Integer> = (0..=two_g)
            .map(|i| {
                if i % 2 == 0 {
                    self.coeffs[i].clone()
                } else {
                    -self.coeffs[i].clone()
                }
            })
            .collect();
        let mut s: Vec<Integer> = Vec::with_capacity(kmax);
        for k in 1..=kmax {
            // s_k = sum_{i=1}^{k-1} (-1)^(i-1) e_i s_(k-i) + (-1)^(k-1) k e_k
            let mut acc = Integer::zero();
            for i in 1..k {
                if i > two_g {
                    break;
                }
                let t = e[i].clone() * s[k - i - 1].clone();
                acc = if (i - 1) % 2 == 0 { acc + t } else { acc - t };
            }
            if k <= two_g {
                let t = int(k as i64) * e[k].clone();
                acc = if (k - 1) % 2 == 0 { acc + t } else { acc - t };
            }
            s.push(acc);
        }
        s.into_iter()
            .enumerate()
            .map(|(k, sk)| self.p.pow(k as u32 + 1) + Integer::one() - sk)
            .collect()
    }

    /// True iff `counts` (`N_1, N_2, ...`, any length) are the counts predicted by `P`. With more
    /// than `g` counts this is a consistency check of the counts beyond those that define `P`.
    pub fn predicts(&self, counts: &[Integer]) -> bool {
        self.point_counts(counts.len()) == counts
    }

    /// The real Weil polynomial `R` (ascending, monic of degree `g`) with `P(T) = T^g R(T + p/T)`.
    pub fn real_weil_polynomial(&self) -> Vec<Integer> {
        let g = self.genus;
        // c_j = coefficient of T^(g+j) = coeffs[g - j]
        let c: Vec<Integer> = (0..=g).map(|j| self.coeffs[g - j].clone()).collect();
        // D_0 = 2, D_1 = x, D_(j+1) = x D_j - p D_(j-1): T^j + (p/T)^j = D_j(T + p/T).
        let mut d: Vec<Vec<Integer>> = vec![vec![int(2)], vec![int(0), int(1)]];
        for j in 1..g {
            let mut nxt = vec![Integer::zero(); j + 2];
            for (i, v) in d[j].iter().enumerate() {
                nxt[i + 1] = nxt[i + 1].clone() + v.clone();
            }
            for (i, v) in d[j - 1].iter().enumerate() {
                nxt[i] = nxt[i].clone() - self.p.clone() * v.clone();
            }
            d.push(nxt);
        }
        let mut r = vec![Integer::zero(); g + 1];
        r[0] = c[0].clone();
        for j in 1..=g {
            for (i, v) in d[j].iter().enumerate() {
                r[i] = r[i].clone() + c[j].clone() * v.clone();
            }
        }
        r
    }

    /// The exact Weil test: every root of `P` has absolute value `sqrt p`. Decided with
    /// multiplicities by Sturm sequences on each squarefree factor of `R` (all roots real) and of
    /// `S(y) = R(sqrt y) R(-sqrt y)` (all roots `x^2` in `[0, 4p]`, closed at `4p`). A finite
    /// exact computation: the answer is a boolean, not a status.
    pub fn is_weil(&self) -> bool {
        let r = self.real_weil_polynomial();
        if self.genus == 0 {
            return true;
        }
        let to_q = |v: &[Integer]| -> Vec<Rational> {
            v.iter()
                .map(|c| Rational::from_integer(c.clone()))
                .collect()
        };
        for (ri, _) in zx::squarefree_decomposition(&r) {
            let deg = ri.len() - 1;
            if real_roots::count_real_roots(&to_q(&ri)) != deg {
                return false;
            }
        }
        // R(x) = E(x^2) + x O(x^2); S(y) = E(y)^2 - y O(y)^2.
        let ev: Vec<Integer> = r.iter().step_by(2).cloned().collect();
        let od: Vec<Integer> = r.iter().skip(1).step_by(2).cloned().collect();
        let e2 = zx::mul(&ev, &ev);
        let s = if od.is_empty() {
            e2
        } else {
            let o2 = zx::mul(&od, &od);
            zx::sub(&e2, &zx::shift(&o2, 1))
        };
        let lo = Rational::from_integer(int(-1));
        let hi = Rational::from_integer(int(4) * self.p.clone());
        for (si, _) in zx::squarefree_decomposition(&s) {
            let deg = si.len() - 1;
            if real_roots::count_real_roots_in(&to_q(&si), &lo, &hi) != deg {
                return false;
            }
        }
        true
    }

    /// `P(x) mod m` in `[0, |m|)`.
    pub fn eval_mod(&self, x: &Integer, m: &Integer) -> Integer {
        let m = m.abs();
        let mut acc = Integer::zero();
        for c in &self.coeffs {
            acc = modulo(&(acc * x.clone() + c.clone()), &m);
        }
        acc
    }

    /// `P mod l` (ascending, coefficients in `[0, l)`).
    pub fn reduce_mod(&self, l: i64) -> Vec<i64> {
        let li = int(l);
        self.coeffs
            .iter()
            .rev()
            .map(|c| modulo(c, &li).to_i64().expect("residue fits i64"))
            .collect()
    }

    /// The factorization of `P mod l` (monic irreducible factors, ascending, with multiplicity),
    /// by [`fp_factor::factor_with_multiplicity`]. Panics if `l` is not a prime `<= MAX_P`.
    pub fn factor_mod(&self, l: i64) -> Vec<(Vec<i64>, u32)> {
        fp_factor::factor_with_multiplicity(&self.reduce_mod(l), l)
    }

    /// Every `Q` of degree `g` over `F_2` with `Q Q^rev = P mod 2` (`Q^rev = x^g Q(1/x)`),
    /// ascending bit vectors, sorted. `Q` and `Q^rev` both appear when they differ. Found by
    /// enumerating the products of the irreducible factors of `P mod 2` (exhaustive).
    pub fn reciprocal_halves_mod2(&self) -> Vec<Vec<i64>> {
        let g = self.genus;
        let target = fp_factor::trim(&self.reduce_mod(2));
        let fac = self.factor_mod(2);
        let mut sols: Vec<Vec<i64>> = Vec::new();
        fn rec(
            idx: usize,
            cur: Vec<i64>,
            fac: &[(Vec<i64>, u32)],
            g: usize,
            target: &[i64],
            sols: &mut Vec<Vec<i64>>,
        ) {
            if idx == fac.len() {
                if fp_factor::degree(&cur) == g as i64 {
                    let rev: Vec<i64> = fp_factor::trim(&cur).iter().rev().cloned().collect();
                    let prod = fp_factor::trim(&fp_factor::mul(&cur, &rev, 2));
                    if prod == target && !sols.contains(&cur) {
                        sols.push(cur);
                    }
                }
                return;
            }
            let (f, m) = &fac[idx];
            let mut pw = vec![1i64];
            for _ in 0..=*m {
                let nxt = fp_factor::trim(&fp_factor::mul(&cur, &pw, 2));
                if fp_factor::degree(&nxt) <= g as i64 {
                    rec(idx + 1, nxt, fac, g, target, sols);
                }
                pw = fp_factor::mul(&pw, f, 2);
            }
        }
        rec(0, vec![1], &fac, g, &target, &mut sols);
        sols.sort();
        sols
    }

    /// The order of Frobenius on `Jac[2]` as far as `P mod 2` decides it (see
    /// [`TwoTorsionOrder`]).
    ///
    /// Cost: for each irreducible factor of `P mod 2`, of degree `d`, the order of `x` is the
    /// least divisor of `2^d - 1` that works. The divisors are found by trial division up to
    /// `2^(d/2)`, so about `2^31` steps near `d = 62`. For genus 4 (`d <= 8`) this is instant.
    ///
    /// # Panics
    ///
    /// Panics if `P mod 2` has an irreducible factor of degree above 62, which is possible from
    /// genus 32 on (`deg P = 2g`), or if the order or the bound `divides` does not fit in a
    /// `u64`, which is possible from genus 64 on (the order is below `2^(g + 1)`). It never
    /// returns a wrapped value.
    pub fn two_torsion_frobenius_order(&self) -> TwoTorsionOrder {
        let fac = self.factor_mod(2);
        let mut s = 1u64;
        let mut emax = 1u32;
        for (f, e) in &fac {
            s = lcm(s, order_of_x_mod2(f));
            emax = emax.max(*e);
        }
        if emax == 1 {
            TwoTorsionOrder::DeterminedByP(s)
        } else {
            // a unipotent block of size <= emax has order 2^ceil(log2 emax)
            let mut two = 1u64;
            while two < emax as u64 {
                two *= 2;
            }
            TwoTorsionOrder::Unresolved {
                semisimple_order: s,
                divides: s.checked_mul(two).expect(ORDER_OVERFLOW),
            }
        }
    }
}

impl fmt::Display for FrobeniusPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.coeffs.len() - 1;
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let e = n - i;
            let sign = if c.signum() < 0 { "-" } else { "+" };
            let a = c.abs();
            if first {
                if c.signum() < 0 {
                    write!(f, "-")?;
                }
            } else {
                write!(f, " {sign} ")?;
            }
            first = false;
            let one = a == Integer::one();
            match (e, one) {
                (0, _) => write!(f, "{a}")?,
                (1, true) => write!(f, "T")?,
                (1, false) => write!(f, "{a}*T")?,
                (_, true) => write!(f, "T^{e}")?,
                (_, false) => write!(f, "{a}*T^{e}")?,
            }
        }
        Ok(())
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

const ORDER_OVERFLOW: &str = "two_torsion_frobenius_order: the order does not fit in a u64";

fn lcm(a: u64, b: u64) -> u64 {
    (a / gcd(a, b)).checked_mul(b).expect(ORDER_OVERFLOW)
}

/// The multiplicative order of `x` in `F_2[x]/(f)` for an irreducible `f != x` (ascending bits).
/// It divides `2^d - 1`; found as the least divisor `n` with `x^n = 1`.
fn order_of_x_mod2(f: &[i64]) -> u64 {
    let d = fp_factor::degree(f) as u32;
    assert!((1..=62).contains(&d), "order_of_x_mod2: degree {d}");
    let n = (1u64 << d) - 1;
    let xpow = |e: u64| -> Vec<i64> { fp_factor::pow_mod(&[0, 1], &Integer::from(e), f, 2) };
    let mut best = n;
    let mut divs: Vec<u64> = (1..=n)
        .take_while(|k| k * k <= n)
        .filter(|k| n.is_multiple_of(*k))
        .collect();
    let big: Vec<u64> = divs.iter().map(|k| n / k).collect();
    divs.extend(big);
    divs.sort();
    for k in divs {
        if fp_factor::trim(&xpow(k)) == vec![1] {
            best = k;
            break;
        }
    }
    best
}

/// The polynomial over `F_2` as a string in `x`, descending (`x^4+x^3+1`).
pub fn f2_poly_string(c: &[i64]) -> String {
    let mut t = Vec::new();
    for i in (0..c.len()).rev() {
        if c[i] % 2 != 0 {
            t.push(match i {
                0 => "1".to_string(),
                1 => "x".to_string(),
                _ => format!("x^{i}"),
            });
        }
    }
    if t.is_empty() {
        "0".into()
    } else {
        t.join("+")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    /// Reference rows of `zeta_X.json` (johnjanik/M23 at d05475b, FLINT reference) used here:
    /// p, N_1..N_4, (a_1..a_4), and the table's Q candidates.
    const X_ROWS: &[(i64, [i64; 4], [i64; 4])] = &[
        (7, [11, 69, 383, 2433], [3, 14, 46, 138]),
        (11, [15, 133, 1368, 14905], [3, 10, 33, 145]),
        (13, [11, 161, 2135, 29089], [-3, 0, -12, 188]),
        (17, [21, 285, 4923, 82985], [3, 2, 0, -130]),
        (19, [17, 405, 7265, 130913], [-3, 26, 66, 74]),
        (29, [35, 789, 24773, 708785], [5, -14, 16, 1060]),
        (31, [33, 993, 29640, 926713], [1, 16, -35, 875]),
        (37, [43, 1433, 50881, 1881609], [5, 44, 254, 3156]),
        (41, [42, 1780, 69408, 2827728], [0, 49, 162, 1692]),
        (43, [44, 1996, 80192, 3422024], [0, 73, 228, 3470]),
        (47, [47, 2349, 104321, 4882281], [-1, 70, 96, 2934]),
        (53, [54, 2804, 148704, 7881808], [0, -3, -58, -2164]),
        (59, [72, 3560, 205056, 12120780], [12, 111, 648, 3991]),
        (61, [64, 3740, 226852, 13855856], [2, 11, -24, 2476]),
        (67, [80, 4460, 302192, 20163724], [12, 57, 584, 8759]),
        (71, [68, 5048, 358256, 25432076], [-4, 11, 92, 4679]),
        (73, [83, 5541, 390089, 28393065], [9, 146, 1428, 12030]),
        (79, [77, 6369, 491636, 38964585], [-3, 68, -663, 7335]),
    ];

    #[test]
    fn newton_round_trip_and_weil_on_the_x_table() {
        for &(p, n, a) in X_ROWS {
            let fp = FrobeniusPolynomial::from_point_counts(&int(p), &ints(&n)).unwrap();
            assert_eq!(&fp.coefficients()[1..5], &ints(&a)[..], "a at p = {p}");
            assert_eq!(fp.coefficients()[8], int(p).pow(4));
            assert_eq!(fp.point_counts(4), ints(&n));
            assert!(fp.is_weil(), "Weil at p = {p}");
            // chi = (-23/p); P(chi) = P(chi p) = 0 mod 23 (HANDOFF 3.1 step 6).
            let chi = int(-23).kronecker_symbol(&int(p)) as i64;
            assert_eq!(fp.eval_mod(&int(chi), &int(23)), Integer::zero());
            assert_eq!(fp.eval_mod(&int(chi * p), &int(23)), Integer::zero());
            // a Q with Q Q^rev = P mod 2 exists at every row
            assert!(!fp.reciprocal_halves_mod2().is_empty());
        }
    }

    #[test]
    fn mod2_halves_match_the_table() {
        // zeta_X.json "Q candidates" at 7, 11, 41, 59.
        let q = |p: i64, n: [i64; 4]| {
            FrobeniusPolynomial::from_point_counts(&int(p), &ints(&n))
                .unwrap()
                .reciprocal_halves_mod2()
                .iter()
                .map(|c| f2_poly_string(c))
                .collect::<Vec<_>>()
        };
        let mut s7 = q(7, [11, 69, 383, 2433]);
        s7.sort();
        assert_eq!(s7, vec!["x^4+x^2+x+1", "x^4+x^3+x^2+1"]);
        let mut s11 = q(11, [15, 133, 1368, 14905]);
        s11.sort();
        assert_eq!(s11, vec!["x^4+x+1", "x^4+x^3+1"]);
        assert_eq!(q(41, [42, 1780, 69408, 2827728]), vec!["x^4+x^3+x+1"]);
        assert_eq!(q(59, [72, 3560, 205056, 12120780]), vec!["x^4+x^3+x^2+x+1"]);
    }

    #[test]
    fn two_torsion_orders() {
        // P mod 2 squarefree at 11, 31, 79: order 15 (HANDOFF 3.3 item 5).
        for &(p, n) in &[
            (11i64, [15i64, 133, 1368, 14905]),
            (31, [33, 993, 29640, 926713]),
            (79, [77, 6369, 491636, 38964585]),
        ] {
            let fp = FrobeniusPolynomial::from_point_counts(&int(p), &ints(&n)).unwrap();
            assert_eq!(
                fp.two_torsion_frobenius_order(),
                TwoTorsionOrder::DeterminedByP(15)
            );
        }
        // p = 7: (x+1)^2 (x^3+x+1)(x^3+x^2+1): semisimple order 7, divides 14.
        let fp =
            FrobeniusPolynomial::from_point_counts(&int(7), &ints(&[11, 69, 383, 2433])).unwrap();
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::Unresolved {
                semisimple_order: 7,
                divides: 14
            }
        );
    }

    fn poly_from_desc(p: i64, c: &[i64]) -> FrobeniusPolynomial {
        FrobeniusPolynomial::from_coefficients(&int(p), ints(c)).unwrap()
    }

    #[test]
    fn exact_weil_test_edge_cases() {
        for &p in &[5i64, 7, 11, 13, 101] {
            // (T^2 + p)^4: R = x^4, a fourfold root at 0.
            let p2 = p * p;
            let c = [1, 0, 4 * p, 0, 6 * p2, 0, 4 * p2 * p, 0, p2 * p2];
            assert!(poly_from_desc(p, &c).is_weil());
            // (T^2 - p)^4 = T^8 - 4p T^6 + 6p^2 T^4 - 4p^3 T^2 + p^4: roots +-sqrt p, real Weil
            // numbers; R = (x^2 - 4p)^2, double roots on the boundary +-2 sqrt p.
            let c = [1, 0, -4 * p, 0, 6 * p2, 0, -4 * p2 * p, 0, p2 * p2];
            let fp = poly_from_desc(p, &c);
            assert_eq!(fp.real_weil_polynomial(), ints(&[16 * p2, 0, -8 * p, 0, 1]));
            assert!(fp.is_weil());
            // genus 1: T^2 + aT + p is Weil iff a^2 <= 4p.
            let a = (4.0 * p as f64).sqrt().floor() as i64;
            assert!(a * a <= 4 * p && (a + 1) * (a + 1) > 4 * p);
            assert!(poly_from_desc(p, &[1, a, p]).is_weil());
            assert!(poly_from_desc(p, &[1, -a, p]).is_weil());
            assert!(!poly_from_desc(p, &[1, a + 1, p]).is_weil());
            assert!(!poly_from_desc(p, &[1, p, p]).is_weil());
            // T^4 + (2p + 1) T^2 + p^2: R = x^2 + 1, non-real roots.
            let fp = poly_from_desc(p, &[1, 0, 2 * p + 1, 0, p2]);
            assert_eq!(fp.real_weil_polynomial(), ints(&[1, 0, 1]));
            assert!(!fp.is_weil());
            // T^4 + (2p - 1) T^2 + p^2: R = x^2 - 1, roots +-1 inside.
            assert!(poly_from_desc(p, &[1, 0, 2 * p - 1, 0, p2]).is_weil());
            // R = x^2 - (4p + 1): roots just outside; R = x^2 - (4p - 1): just inside.
            assert!(!poly_from_desc(p, &[1, 0, -2 * p - 1, 0, p2]).is_weil());
            assert!(poly_from_desc(p, &[1, 0, -2 * p + 1, 0, p2]).is_weil());
        }
    }

    #[test]
    fn reference_rule_row_at_5_is_not_weil() {
        // zeta_X.json p = 5: counts [8, 36, 170, 560] (reference rule), "not a Weil polynomial".
        let fp = FrobeniusPolynomial::from_point_counts(&int(5), &ints(&[8, 36, 170, 560]));
        match fp {
            Ok(fp) => assert!(!fp.is_weil()),
            Err(e) => panic!("unexpected {e}"),
        }
        // The exact counts 7, 37, 169, 561 (WP-F design 7.1) give a Weil polynomial.
        let fp =
            FrobeniusPolynomial::from_point_counts(&int(5), &ints(&[7, 37, 169, 561])).unwrap();
        assert_eq!(
            fp.coefficients(),
            &ints(&[1, 1, 6, 20, 16, 100, 150, 125, 625])[..]
        );
        assert!(fp.is_weil());
    }

    #[test]
    fn non_integral_and_functional_equation_errors() {
        // N_1 = 0, N_2 = 0 at p = 3: s = (4, 10); e_2 = (4*4 - 10)/2 = 3 integral; use N_2 = 1:
        // e_2 = (16 - 9)/2 not integral.
        let r = FrobeniusPolynomial::from_point_counts(&int(3), &ints(&[0, 1]));
        assert!(matches!(r, Err(WeilError::NonIntegral { index: 2, .. })));
        assert!(matches!(
            FrobeniusPolynomial::from_coefficients(&int(5), ints(&[1, 1, 6])),
            Err(WeilError::FunctionalEquation { index: 2 })
        ));
        assert!(matches!(
            FrobeniusPolynomial::from_coefficients(&int(5), ints(&[1, 1])),
            Err(WeilError::InvalidInput(_))
        ));
    }

    #[test]
    fn predicted_counts_beyond_g() {
        // y^2 = x^3 + 3x + 5 over F_7: a_7 = 1 (PARI/GP ellap(ellinit([0,0,0,3,5]), 7) = 1),
        // N_1 = 7; N_2 = 49 + 1 - (1 - 14) = 63.
        let fp = FrobeniusPolynomial::from_point_counts(&int(7), &ints(&[7])).unwrap();
        assert_eq!(fp.coefficients(), &ints(&[1, -1, 7])[..]);
        assert!(fp.predicts(&ints(&[7, 63])));
        assert!(!fp.predicts(&ints(&[7, 64])));
        assert_eq!(format!("{fp}"), "T^2 - T + 7");
    }

    #[test]
    #[should_panic(expected = "does not fit in a u64")]
    fn two_torsion_order_above_u64_panics() {
        // Genus 67, p = 3, P mod 2 = f f^rev for f = x^31 + x^3 + 1, x^19 + x^5 + x^2 + x + 1 and
        // x^17 + x^3 + 1, irreducible over F_2 with prime 2^d - 1. The order of Frobenius on
        // Jac[2] is (2^31 - 1)(2^19 - 1)(2^17 - 1), above 2^64 (PARI/GP polisirreducible,
        // factormod, lcm; docs/workstreams/m23_golay/wpf/gp/two_torsion_overflow.gp). Before the
        // fix this wrapped in release builds.
        let mut f2 = vec![1i64];
        for exps in [&[31, 3, 0][..], &[19, 5, 2, 1, 0], &[17, 3, 0]] {
            let mut f = vec![0i64; exps[0] + 1];
            for &e in exps {
                f[e] = 1;
            }
            let rev: Vec<i64> = f.iter().rev().cloned().collect();
            f2 = fp_factor::mul(&fp_factor::mul(&f2, &f, 2), &rev, 2);
        }
        let g = 67;
        assert_eq!(f2.len(), 2 * g + 1);
        // Descending, c[i] = c[2g - i] mod 2 (f2 is palindromic) and c[2g - i] = 3^(g - i) c[i].
        let mut c: Vec<Integer> = f2.iter().rev().map(|&b| Integer::from(b)).collect();
        for i in 0..g {
            c[2 * g - i] = int(3).pow((g - i) as u32) * c[i].clone();
        }
        let fp = FrobeniusPolynomial::from_coefficients(&int(3), c).unwrap();
        fp.two_torsion_frobenius_order();
    }
}
