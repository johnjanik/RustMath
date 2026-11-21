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
//!   The module structure of `Jac[2]` as far as `P` decides it, from the 2-ranks of the
//!   Jacobian and of its quadratic twists over `F_(q^k)` and the Weil pairing, is in
//!   [`two_torsion`] (design C57; REQUESTS C-5).
//! - Base change: `P^(k)(T) = prod (T - alpha_i^k)` is the Frobenius polynomial over `F_(q^k)`
//!   (MAGMA `LPolynomial(C, k)`, reversed), and `#Jac(F_(q^k)) = P^(k)(1) = Res(P, x^k - 1)`.

mod two_torsion;

pub use two_torsion::{
    LagrangianType, ModuleType, PrimaryFactor, RankBound, TwoTorsionAnalysis, TwoTorsionError,
    MAX_BASE_CHANGE_DEGREE, MAX_CANDIDATES, MAX_LR_SIZE,
};

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

/// The order of Frobenius on `Jac[2]` as far as `P` decides it (design C57 2.6). Every variant
/// is a statement about `P`: it is the order on `Jac[2]` of the curve only when `P` is certified
/// to be its Frobenius polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TwoTorsionOrder {
    /// `P mod 2` is squarefree: Frobenius is semisimple on `Jac[2]` and its order is the lcm of
    /// the orders of `x` modulo the irreducible factors.
    DeterminedByP(u64),
    /// `P mod 2` has a repeated factor and every module type allowed by the rank bounds and the
    /// Weil pairing has this order. `bounds` is a minimal set of rank bounds that decides it on
    /// its own ([`TwoTorsionAnalysis::decisive_bounds`]).
    DeterminedByTwoRank { order: u64, bounds: Vec<RankBound> },
    /// Several orders remain: `candidates`, sorted, each a multiple of `semisimple_order` (the
    /// lcm over the distinct factors) dividing `divides` (`semisimple_order * 2^ceil(log2
    /// e_max)`, `e_max` the largest multiplicity).
    Unresolved {
        semisimple_order: u64,
        divides: u64,
        candidates: Vec<u64>,
    },
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

    /// The size `q` of the base field: the prime `p` for the curves here, `q^k` after
    /// [`Self::base_change`]`(k)`.
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
        self.power_sums(kmax)
            .into_iter()
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

    /// `P^(k)(T) = prod (T - alpha_i^k)`: the Frobenius polynomial over `F_(q^k)` (MAGMA
    /// `LPolynomial(C, k)`, reversed). The power sums `s_1..s_(2gk)` of the `alpha_i` by Newton's
    /// identities, keeping only the last `2g` and the `s_(jk)` (those of the `alpha_i^k`), then
    /// the coefficients with exact division. Its [`Self::p`] is `q^k`.
    ///
    /// Cost: `s_m` has about `m log2(q) / 2` bits. Each of the `2gk` steps multiplies `2g`
    /// coefficients of `P` by such sums, so the time grows like `g^4 k^2 (log2 q)^2` (quadratic
    /// in `k`) and the memory like `g^2 k log2 q` bits, the size of the result (genus 12,
    /// `k = 4094`, `q = 10^9 + 7`: 33 s and 13 MB in a release build). The analysis of `Jac[2]`
    /// does not use this function.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0` or `k > MAX_BASE_CHANGE_DEGREE` (4096).
    pub fn base_change(&self, k: u32) -> FrobeniusPolynomial {
        assert!(
            (1..=MAX_BASE_CHANGE_DEGREE).contains(&k),
            "base_change: k = {k} outside 1..={MAX_BASE_CHANGE_DEGREE}"
        );
        self.base_change_by_power_sums(k as usize)
    }

    /// `#Jac(F_(q^k)) = P^(k)(1) = Res(P, x^k - 1)` (MAGMA `ClassNumber` of the constant field
    /// extension of degree `k`). Same panics as [`Self::base_change`].
    pub fn jacobian_order(&self, k: u32) -> Integer {
        self.base_change(k).eval_at_sign(false)
    }

    /// The order of the quadratic twist of `Jac` over `F_(q^k)` (Frobenius `-F^k`):
    /// `P^(k)(-1) = Res(P, x^k + 1)`. Same panics as [`Self::base_change`].
    pub fn twisted_jacobian_order(&self, k: u32) -> Integer {
        self.base_change(k).eval_at_sign(true)
    }

    /// The module `Jac[2]` over `F_2[x]` as far as `P` decides it (design C57 2.6; see
    /// [`TwoTorsionAnalysis`]): the types allowed by the Weil pairing (S1, S2) and by the rank
    /// bounds `dim ker(F^k -+ 1) <= v_2(P^(k)(+-1))` for every `k | N_max`.
    ///
    /// Hypothesis: `P` is the Frobenius polynomial of a principally polarized abelian variety
    /// over `F_q`, `q` odd (a Jacobian). For any other `P` with the functional equation the
    /// result is about a hypothetical variety; an empty set of types is `Err(Inconsistent)`.
    ///
    /// Errors: `EvenCharacteristic` (`q` even), `TooManyCandidates` (more than
    /// [`MAX_CANDIDATES`] types), `TooLarge` (`N_max` above [`MAX_BASE_CHANGE_DEGREE`]),
    /// `Inconsistent`.
    ///
    /// # Panics
    ///
    /// As [`Self::two_torsion_frobenius_order`]: a factor of `P mod 2` of degree above 62, or an
    /// order that does not fit in a `u64`.
    pub fn two_torsion_analysis(&self) -> Result<TwoTorsionAnalysis, TwoTorsionError> {
        two_torsion::analyse(self)
    }

    /// The order of Frobenius on `Jac[2]` as far as `P` decides it (see [`TwoTorsionOrder`]):
    /// [`Self::two_torsion_analysis`]`.order()`. When the analysis is not made
    /// (`TooManyCandidates`, `TooLarge`) or finds no type (`Inconsistent`, a `P` that is not the
    /// Frobenius polynomial of a principally polarized abelian variety), it returns the bound of
    /// `P mod 2` alone, `Unresolved` with the candidates `S 2^t`, `0 <= t <= ceil(log2 e_max)`:
    /// a true statement about any operator with characteristic polynomial `P mod 2`.
    ///
    /// Cost: for each irreducible factor of `P mod 2`, of degree `d`, the order of `x` is the
    /// least divisor of `2^d - 1` that works. The divisors are found by trial division up to
    /// `2^(d/2)`, so about `2^31` steps near `d = 62`. The rank bounds take `O(g^2 log k + g^3)`
    /// operations on 128-bit words for each `k | N_max`, whatever the size of `q` (genus 12 with
    /// `N_max = 4094` over `q = 10^9 + 7`: 0.02 s in a debug build). The types are at most
    /// [`MAX_CANDIDATES`]. For genus 4 (`d <= 8`) the whole analysis takes milliseconds.
    ///
    /// # Panics
    ///
    /// Panics if `q` is even (`P mod 2` is then not the characteristic polynomial of Frobenius
    /// on `Jac[2]`), if `P mod 2` has an irreducible factor of degree above 62, which is possible
    /// from genus 32 on (`deg P = 2g`), or if the order or the bound `divides` does not fit in a
    /// `u64`, which is possible from genus 64 on (the order is below `2^(g + 1)`). It never
    /// returns a wrapped value.
    pub fn two_torsion_frobenius_order(&self) -> TwoTorsionOrder {
        match self.two_torsion_analysis() {
            Ok(a) => a.order(),
            Err(TwoTorsionError::EvenCharacteristic) => panic!(
                "two_torsion_frobenius_order: q is even, so P mod 2 is not the characteristic \
                 polynomial of Frobenius on Jac[2]"
            ),
            Err(_) => self.mod2_bound(),
        }
    }

    /// The bound of `P mod 2` alone: `Unresolved` with every `S 2^t` (`DeterminedByP` when
    /// `P mod 2` is squarefree).
    fn mod2_bound(&self) -> TwoTorsionOrder {
        let fac = self.factor_mod(2);
        let mut s = 1u64;
        let mut emax = 1u32;
        for (f, e) in &fac {
            s = lcm(s, order_of_x_mod2(f));
            emax = emax.max(*e);
        }
        if emax == 1 {
            return TwoTorsionOrder::DeterminedByP(s);
        }
        // a unipotent block of size <= emax has order 2^ceil(log2 emax)
        let mut two = 1u64;
        let mut candidates = vec![s];
        while two < emax as u64 {
            two *= 2;
            candidates.push(s.checked_mul(two).expect(ORDER_OVERFLOW));
        }
        TwoTorsionOrder::Unresolved {
            semisimple_order: s,
            divides: s.checked_mul(two).expect(ORDER_OVERFLOW),
            candidates,
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

const ORDER_OVERFLOW: &str = "the order of Frobenius on Jac[2] does not fit in a u64";

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
        // p = 7: (x+1)^2 (x^3+x+1)(x^3+x^2+1): semisimple order 7, divides 14; both orders are
        // possible (v2 #Jac(F_7) = 5, design C57 Section 4.1).
        let fp =
            FrobeniusPolynomial::from_point_counts(&int(7), &ints(&[11, 69, 383, 2433])).unwrap();
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::Unresolved {
                semisimple_order: 7,
                divides: 14,
                candidates: vec![7, 14],
            }
        );
        // p = 17: #Jac(F_17) = P(1) = 98714 = 2 * 49357 (REQUESTS 1.1; PARI keyvals_pari.txt), so
        // one Jordan block of size 2 at the eigenvalue 1: order 14 (REQUESTS C-5).
        let fp = FrobeniusPolynomial::from_point_counts(&int(17), &ints(&[21, 285, 4923, 82985]))
            .unwrap();
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByTwoRank {
                order: 14,
                bounds: vec![RankBound {
                    k: 1,
                    twisted: false,
                    v2: 1
                }],
            }
        );
    }

    #[test]
    fn base_change_predicts_the_counts() {
        // P^(k) is the Frobenius polynomial over F_(p^k): its N_1 is N_k of the curve, and the
        // N_1, N_2 of P^(2) are N_2, N_4 (zeta_X.json counts).
        for &(p, n, _) in X_ROWS {
            let fp = FrobeniusPolynomial::from_point_counts(&int(p), &ints(&n)).unwrap();
            for k in 1..=4u32 {
                let b = fp.base_change(k);
                assert_eq!(b.p(), &int(p).pow(k), "p = {p}, k = {k}");
                assert_eq!(b.genus(), 4);
                assert_eq!(
                    b.point_counts(1),
                    ints(&[n[k as usize - 1]]),
                    "p = {p}, k = {k}"
                );
            }
            assert_eq!(
                fp.base_change(2).point_counts(2),
                ints(&[n[1], n[3]]),
                "p = {p}"
            );
            assert_eq!(fp.base_change(1), fp);
        }
    }

    #[test]
    fn jacobian_orders_match_the_owner() {
        // order_of_Jac_Fp of data/a8_frobenius_predictions.json (johnjanik/M23 at e0d25b8,
        // vendored in tests/data/m23/) for the X_ROWS primes.
        let owner: &[(i64, i64)] = &[
            (7, 4640),
            (11, 20399),
            (13, 21988),
            (17, 98714),
            (19, 120548),
            (29, 818984),
            (31, 968461),
            (37, 2200520),
            (41, 2916676),
            (43, 3567354),
            (47, 4938100),
            (53, 7876756),
            (59, 15011295),
            (61, 14341736),
            (67, 24064691),
            (71, 24046799),
            (73, 32795286),
            (79, 37849713),
        ];
        for &(p, n, _) in X_ROWS {
            let fp = FrobeniusPolynomial::from_point_counts(&int(p), &ints(&n)).unwrap();
            let o = owner.iter().find(|r| r.0 == p).expect("owner row").1;
            assert_eq!(fp.jacobian_order(1), int(o), "p = {p}");
        }
        // PARI/GP (tests/data/two_torsion/keyvals.gp, keyvals_pari.txt): P(-1),
        // Res(P, x^4 + 1) and Res(P, x^5 - 1) at 17 and 59.
        let row = |p: i64| X_ROWS.iter().find(|r| r.0 == p).unwrap();
        let big = |s: &str| Integer::from_decimal_str(s).unwrap();
        for (p, tw1, tw4, j5) in [
            (
                17,
                "69230",
                "48975933951329703300",
                "4069914032622861116386154",
            ),
            (
                59,
                "10004415",
                "21553097128538527354887445681",
                "261272374040302192032654432854223600",
            ),
        ] {
            let fp = FrobeniusPolynomial::from_point_counts(&int(p), &ints(&row(p).1)).unwrap();
            assert_eq!(fp.twisted_jacobian_order(1), big(tw1), "p = {p}");
            assert_eq!(fp.twisted_jacobian_order(4), big(tw4), "p = {p}");
            assert_eq!(fp.jacobian_order(5), big(j5), "p = {p}");
        }
    }

    /// The descending coefficients of a product of polynomials given descending.
    fn mul_desc(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
        let mut r = vec![Integer::zero(); a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                r[i + j] = r[i + j].clone() + x.clone() * y.clone();
            }
        }
        r
    }

    /// `(T^2 + q)^g`: a Weil polynomial (roots `+-i sqrt q`) with `P mod 2 = (x + 1)^(2g)`.
    fn t2_plus_q_pow(q: i64, g: usize) -> FrobeniusPolynomial {
        let base = ints(&[1, 0, q]);
        let mut c = ints(&[1]);
        for _ in 0..g {
            c = mul_desc(&c, &base);
        }
        FrobeniusPolynomial::from_coefficients(&int(q), c).unwrap()
    }

    #[test]
    fn degenerate_genus_0_and_1() {
        // Genus 0: P = 1, Jac[2] = 0.
        let fp = FrobeniusPolynomial::from_coefficients(&int(5), ints(&[1])).unwrap();
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByP(1)
        );
        let a = fp.two_torsion_analysis().unwrap();
        assert_eq!(a.candidates, vec![ModuleType(vec![])]);
        assert!(a.bounds.is_empty());
        assert_eq!(a.eigenvalue_one_types(), None);
        let w = a.lagrangian_types(&[1]).unwrap();
        assert_eq!(w.len(), 1);
        assert!(w[0].parts.is_empty());
        assert_eq!(w[0].eigenvalue_one(), Vec::<u32>::new());
        assert_eq!(w[0].order(), 1);
        // Elliptic curves (PARI/GP ellcard, factormod; tests/data/two_torsion/elliptic.gp and
        // elliptic_pari.txt). y^2 = x^3 + 3x + 5 over F_7: a = 1, the cubic is irreducible.
        let fp = poly_from_desc(7, &[1, -1, 7]);
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByP(3)
        );
        // y^2 = x^3 + x + 1 over F_5: #E = 9, a = -3.
        let fp = poly_from_desc(5, &[1, 3, 5]);
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByP(3)
        );
        // y^2 = x^3 + 1 over F_5: #E = 6 (one root of the cubic, E[2](F_5) of order 2),
        // P = T^2 + 5: one block of size 2, order 2.
        let fp = poly_from_desc(5, &[1, 0, 5]);
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByTwoRank {
                order: 2,
                bounds: vec![RankBound {
                    k: 1,
                    twisted: false,
                    v2: 1
                }],
            }
        );
        let a = fp.two_torsion_analysis().unwrap();
        assert_eq!(a.eigenvalue_one_types(), Some(vec![vec![2]]));
        // y^2 = x^3 - x over F_5: #E = 8, P = T^2 + 2T + 5, P(-1) = 4; three roots, so the true
        // order is 1, but P alone allows 1 and 2: Unresolved, never a guess.
        let fp = poly_from_desc(5, &[1, 2, 5]);
        assert_eq!(fp.jacobian_order(1), int(8));
        assert_eq!(fp.twisted_jacobian_order(1), int(4));
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::Unresolved {
                semisimple_order: 1,
                divides: 2,
                candidates: vec![1, 2],
            }
        );
        let a = fp.two_torsion_analysis().unwrap();
        assert_eq!(a.eigenvalue_one_types(), Some(vec![vec![2], vec![1, 1]]));
    }

    #[test]
    fn symplectic_filter() {
        // Design C57 2.4, S2: at the eigenvalue 1 every odd part has even multiplicity.
        // P mod 2 = (x + 1)^4: [3, 1] is removed (one of the five partitions).
        let a = t2_plus_q_pow(5, 2).two_torsion_analysis().unwrap();
        assert_eq!(a.symplectic_removed, 1);
        let one = a.eigenvalue_one_types().unwrap();
        assert!(!one.contains(&vec![3, 1]));
        // (x + 1)^8: 8 of the 22 partitions are removed (7 1, 5 3, 5 2 1, 5 1^3, 4 3 1, 3 2^2 1,
        // 3 2 1^3, 3 1^5).
        let a = t2_plus_q_pow(5, 4).two_torsion_analysis().unwrap();
        assert_eq!(a.symplectic_removed, 8);
        let one = a.eigenvalue_one_types().unwrap();
        for t in [vec![7, 1], vec![5, 3], vec![5, 2, 1], vec![3, 2, 1, 1, 1]] {
            assert!(!one.contains(&t), "{t:?}");
        }
    }

    #[test]
    fn too_many_candidates_and_errors() {
        // (T^2 + 5)^40: P mod 2 = (x + 1)^80 has p(80) = 15796476 types (above 10^6).
        let fp = t2_plus_q_pow(5, 40);
        assert_eq!(
            fp.two_torsion_analysis(),
            Err(TwoTorsionError::TooManyCandidates { count: 15796476 })
        );
        // The fallback is the bound of P mod 2 alone, a true statement.
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::Unresolved {
                semisimple_order: 1,
                divides: 128,
                candidates: vec![1, 2, 4, 8, 16, 32, 64, 128],
            }
        );
        // C57 review 1, m1: (T^2 + 5)^210, (x + 1)^420, p(420) = 22755290216580025259 (PARI/GP
        // numbpart(420)) is above u64::MAX; the count once went through partition_count, which
        // panics there. Above multiplicity 1000 the count saturates.
        let fp = t2_plus_q_pow(5, 210);
        assert_eq!(
            fp.two_torsion_analysis(),
            Err(TwoTorsionError::TooManyCandidates {
                count: 22755290216580025259
            })
        );
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::Unresolved {
                semisimple_order: 1,
                divides: 512,
                candidates: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            }
        );
        let fp = t2_plus_q_pow(5, 501);
        assert_eq!(
            fp.two_torsion_analysis(),
            Err(TwoTorsionError::TooManyCandidates { count: u128::MAX })
        );
        // q even: P mod 2 is not the characteristic polynomial of Frobenius on Jac[2].
        let fp = poly_from_desc(4, &[1, 0, 4]);
        assert_eq!(
            fp.two_torsion_analysis(),
            Err(TwoTorsionError::EvenCharacteristic)
        );
        // A w with w w^rev != P mod 2.
        let fp =
            FrobeniusPolynomial::from_point_counts(&int(7), &ints(&[11, 69, 383, 2433])).unwrap();
        let a = fp.two_torsion_analysis().unwrap();
        assert_eq!(
            a.lagrangian_types(&[1, 1, 0, 0, 1]),
            Err(TwoTorsionError::NotAReciprocalHalf)
        );
        // C57 review 1, m2: multiplicity 8 is the largest for lagrangian_types; (x + 1)^10 is
        // refused (w = (x + 1)^5 = 1 + x + x^4 + x^5).
        let a = t2_plus_q_pow(5, 4).two_torsion_analysis().unwrap();
        assert!(a.lagrangian_types(&[1, 0, 0, 0, 1]).is_ok());
        let a = t2_plus_q_pow(5, 5).two_torsion_analysis().unwrap();
        assert!(matches!(
            a.lagrangian_types(&[1, 1, 0, 0, 1, 1]),
            Err(TwoTorsionError::TooLarge(_))
        ));
    }

    #[test]
    fn zero_group_orders_give_no_bound() {
        // P = (T - 1)(T - 3) satisfies the functional equation over q = 3 but has the root 1, so
        // #Jac(F_3) = P(1) = 0 and #Jac(F_9) = 0: those bounds are skipped (design review 1,
        // m1), never v_2(0). The twisted bounds stay: P(-1) = 8, P^(2)(-1) = 20.
        let fp = poly_from_desc(3, &[1, -4, 3]);
        let a = fp.two_torsion_analysis().unwrap();
        assert_eq!(a.skipped_bounds, vec![(1, false), (2, false)]);
        assert_eq!(
            a.bounds,
            vec![
                RankBound {
                    k: 1,
                    twisted: true,
                    v2: 3
                },
                RankBound {
                    k: 2,
                    twisted: true,
                    v2: 2
                }
            ]
        );
        assert_eq!(a.orders(), vec![1, 2]);
    }

    /// A `P` of genus `g` with the functional equation over `q` whose reduction mod 2 is the
    /// palindromic `bits` (descending, `2g + 1` entries): `c_i = bits_i` for `i <= g` and
    /// `c_(2g - i) = q^(g - i) c_i` (C57 review 1). It need not be a Weil polynomial: the
    /// analysis of `Jac[2]` does not use the Riemann hypothesis.
    fn synthetic(q: &Integer, bits: &[i64]) -> FrobeniusPolynomial {
        let g = (bits.len() - 1) / 2;
        let mut c: Vec<Integer> = bits.iter().map(|&b| int(b)).collect();
        for i in 0..g {
            c[2 * g - i] = q.pow((g - i) as u32) * c[i].clone();
        }
        FrobeniusPolynomial::from_coefficients(q, c).unwrap()
    }

    /// `(x + 1)^2 f f*` with `f = x^11 + x^2 + 1` (descending bits, genus 12): `x` has order
    /// 2047 modulo `f`, so `N_max = 4094`.
    fn large_n_max_bits() -> Vec<i64> {
        // ascending: f = 1 + x^2 + x^11, f* = 1 + x^9 + x^11, (x + 1)^2 = 1 + x^2
        let f: Vec<i64> = (0..12)
            .map(|i| i64::from([0, 2, 11].contains(&i)))
            .collect();
        let fr: Vec<i64> = f.iter().rev().cloned().collect();
        let prod = fp_factor::mul(&fp_factor::mul(&f, &fr, 2), &[1, 0, 1], 2);
        assert_eq!(prod.len(), 25);
        prod.iter().rev().cloned().collect()
    }

    /// `f()` on its own thread; fails the test if it takes more than `secs` seconds.
    fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
        use std::sync::mpsc::{channel, RecvTimeoutError};
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => panic!("not finished after {secs} s"),
            Err(RecvTimeoutError::Disconnected) => panic!("the computation panicked"),
        }
    }

    #[test]
    fn large_n_max_bounds_are_cheap() {
        // C57 review 1, B1. Genus 12, P mod 2 = (x + 1)^2 f f*, f = x^11 + x^2 + 1: N_max = 4094
        // (k | 4094: 1, 2, 23, 46, 89, 178, 2047, 4094). The bounds once came from the
        // 2g N_max = 98256 power sums of P at full precision: at q = 3 5.7 s and 558 MB (debug),
        // at q = 10^9 + 7 about 10 GB (the process aborted at a 3 GB cap after 72 s). Reference
        // values: PARI/GP, v_2 of norm(Mod(x, P)^k -+ 1) (tests/data/two_torsion/large_nmax.gp,
        // large_nmax_pari.txt).
        // (q, [(k, v2 untwisted, v2 twisted) for k | 4094])
        type Case = (&'static str, [(u64, u32, u32); 8]);
        let cases: [Case; 2] = [
            (
                "3",
                [
                    (1, 4, 3),
                    (2, 7, 2),
                    (23, 4, 3),
                    (46, 7, 2),
                    (89, 4, 3),
                    (178, 7, 2),
                    (2047, 26, 25),
                    (4094, 51, 24),
                ],
            ),
            (
                "1000000007",
                [
                    (1, 2, 2),
                    (2, 4, 2),
                    (23, 2, 2),
                    (46, 4, 2),
                    (89, 2, 2),
                    (178, 4, 2),
                    (2047, 24, 24),
                    (4094, 48, 24),
                ],
            ),
        ];
        for (q, want) in cases {
            let fp = synthetic(&Integer::from_decimal_str(q).unwrap(), &large_n_max_bits());
            let f2 = fp.clone();
            let a = within(60, move || f2.two_torsion_analysis()).unwrap();
            assert_eq!((a.semisimple_order, a.max_order), (2047, 4094), "q = {q}");
            let expect: Vec<RankBound> = [false, true]
                .iter()
                .flat_map(|&twisted| {
                    want.iter().map(move |&(k, u, t)| RankBound {
                        k,
                        twisted,
                        v2: if twisted { t } else { u },
                    })
                })
                .collect();
            assert_eq!(a.bounds, expect, "q = {q}");
            assert!(a.skipped_bounds.is_empty(), "q = {q}");
            // lambda_(x+1) = [2] or [1, 1]: r_1 = 1 or 2, both allowed by every bound.
            assert_eq!(a.orders(), vec![2047, 4094], "q = {q}");
            let f2 = fp.clone();
            assert_eq!(
                within(60, move || f2.two_torsion_frobenius_order()),
                TwoTorsionOrder::Unresolved {
                    semisimple_order: 2047,
                    divides: 4094,
                    candidates: vec![2047, 4094],
                },
                "q = {q}"
            );
        }
    }

    #[test]
    fn bounds_equal_the_exact_valuations() {
        // The bounds of the analysis (valuations computed modulo 2^128) equal the 2-adic
        // valuations of the exact group orders of base_change (power sums, a second method) at
        // every k | N_max, on the X_ROWS rows with a repeated factor mod 2, (T^2 + q)^g and
        // the genus-12 P of large_n_max_bounds_are_cheap at q = 3 for k | 178.
        let mut cases: Vec<FrobeniusPolynomial> = X_ROWS
            .iter()
            .map(|&(p, n, _)| FrobeniusPolynomial::from_point_counts(&int(p), &ints(&n)).unwrap())
            .collect();
        cases.extend([
            t2_plus_q_pow(5, 1),
            t2_plus_q_pow(3, 2),
            t2_plus_q_pow(7, 4),
        ]);
        let mut checked = 0;
        for fp in &cases {
            let a = fp.two_torsion_analysis().unwrap();
            for b in &a.bounds {
                let n = if b.twisted {
                    fp.twisted_jacobian_order(b.k as u32)
                } else {
                    fp.jacobian_order(b.k as u32)
                };
                assert_eq!(n.valuation(&int(2)), b.v2, "{fp}: {b:?}");
                checked += 1;
            }
            assert!(a.skipped_bounds.is_empty(), "{fp}");
        }
        assert!(checked >= 100, "{checked}");
        let fp = synthetic(&int(3), &large_n_max_bits());
        let a = fp.two_torsion_analysis().unwrap();
        for b in a.bounds.iter().filter(|b| 178 % b.k == 0) {
            let n = if b.twisted {
                fp.twisted_jacobian_order(b.k as u32)
            } else {
                fp.jacobian_order(b.k as u32)
            };
            assert_eq!(n.valuation(&int(2)), b.v2, "genus 12: {b:?}");
        }
    }

    #[test]
    fn base_change_of_genus_12() {
        // The exact group orders P^(k)(1) = Res(P, x^k - 1) and P^(k)(-1) = Res(P, x^k + 1) of
        // the genus-12 P of large_n_max_bounds_are_cheap modulo 10^30, and their 2-adic
        // valuations (PARI/GP, tests/data/two_torsion/large_nmax.gp, large_nmax_pari.txt).
        let ten30 = int(10).pow(30);
        let big = |s: &str| Integer::from_decimal_str(s).unwrap();
        for (q, k, untwisted, twisted, v) in [
            ("3", 1, "538032", "537976", (4, 3)),
            (
                "3",
                23,
                "307867245892628370137315787216",
                "894414162359307374796230493256",
                (4, 3),
            ),
            (
                "3",
                89,
                "187948291803259651120252897008",
                "706212311980273080899907610392",
                (4, 3),
            ),
            (
                "1000000007",
                1,
                "751646660650734509420847052348",
                "749646660608734509126847051660",
                (2, 2),
            ),
            (
                "1000000007",
                23,
                "324819430060498873988995265812",
                "320645424931915580285248254340",
                (2, 2),
            ),
            (
                "1000000007",
                89,
                "632307901554557974918740542668",
                "81424749539139308662858201180",
                (2, 2),
            ),
        ] {
            let fp = synthetic(&big(q), &large_n_max_bits());
            let n = fp.jacobian_order(k);
            let t = fp.twisted_jacobian_order(k);
            assert_eq!(n.modulo(&ten30), big(untwisted), "q = {q}, k = {k}");
            assert_eq!(t.modulo(&ten30), big(twisted), "q = {q}, k = {k}");
            assert_eq!(
                (n.valuation(&int(2)), t.valuation(&int(2))),
                v,
                "q = {q}, k = {k}"
            );
            assert_eq!(fp.base_change(k).p(), &big(q).pow(k), "q = {q}, k = {k}");
        }
    }

    #[test]
    #[should_panic(expected = "q is even")]
    fn two_torsion_order_in_even_characteristic_panics() {
        poly_from_desc(4, &[1, 0, 4]).two_torsion_frobenius_order();
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
