//! # MacLane inductive valuations on K[x]
//!
//! Real machinery for (shifted) Gauss valuations, augmented valuations
//! `w = [v, w(phi) = lambda]`, key-polynomial detection, and MacLane
//! approximants of the extensions of a p-adic valuation to `Q[x]/(f)`.
//!
//! Semantic reference: `sage.rings.valuation` (Sage) and MacLane,
//! *A construction for absolute values in polynomial rings*, Trans. AMS 40
//! (1936); residual-polynomial normalization as in Guardia-Montes-Nart,
//! *Newton polygons of higher order in algebraic number theory* (order one).
//!
//! ## What is real here
//!
//! - [`InductiveValuation::gauss`]: the Gauss valuation
//!   `v_0(sum a_i x^i) = min_i v(a_i)` over any [`BaseValuation`] (the p-adic
//!   valuation on `Q` via [`PAdicBaseValuation`], or a place valuation on the
//!   stage-1 typed rational function field via [`PlaceBaseValuation`]).
//! - [`InductiveValuation::gauss_shifted`]: the monomial valuation
//!   `v_0(sum a_i x^i) = min_i (v(a_i) + i*lambda)` for any rational
//!   `lambda` (i.e. `v_0(x) = lambda`).
//! - [`InductiveValuation::augment`] (p-adic base): the MacLane augmentation
//!   `w = [v, w(phi) = lambda]` computed by the phi-adic expansion
//!   `f = sum f_i phi^i`, `w(f) = min_i (v(f_i) + i*lambda)`. The key
//!   polynomial condition (monic, equivalence-irreducible, v-minimal) is
//!   VERIFIED before augmenting; a non-key `phi` is an `Err`, because the min
//!   formula is not multiplicative for non-keys (there is a should-fail test
//!   demonstrating exactly that).
//! - [`InductiveValuation::is_key`] (p-adic base): the standard effective
//!   criterion at EVERY augmentation level. At the Gauss level: monic,
//!   p-integral, irreducible reduction mod p. At level `k >= 1`: degree
//!   divisible by `tau_k * d_k`, one-sided Newton polygon of slope
//!   `-lambda_k` (`w(phi) = deg(phi)/d_k * lambda_k`), and irreducible
//!   residual polynomial over the residue-field tower `kappa_k`, not
//!   divisible by `y` (plus the same-degree equivalent-key cases).
//! - Residue machinery through the whole tower
//!   `kappa_0 = GF(p) ⊂ kappa_j = kappa_{j-1}[y_{j-1}]/(psi_j)` (see
//!   [`crate::valuation::residue_tower`]): a recursive reduction map with
//!   the coherent `Q_j^t` normalization of the GMN residual polynomials,
//!   certified residual factorization over `GF(p^d)` for any `d`, and
//!   key-polynomial lifting from residual factors by GF(p)-linear algebra
//!   over the standard degree-bounded monomials — every lift is re-checked
//!   (value, residual associate to the chosen factor) before use, so a
//!   lifting bug is an `Err`, never a wrong key.
//! - [`mac_lane_approximants`]: the MacLane tree for monic squarefree
//!   p-integral `f` over `Q`, iterating `mac_lane_step` (now real at every
//!   level) until `sum E(w)*F(w) = deg f` over the leaves. Each leaf
//!   approximates exactly one irreducible factor of `f` over `Q_p`, with
//!   `E` and `F` its ramification index and residue degree. The packaged
//!   factorization with congruence-certified approximations is
//!   [`crate::padics::om_factorization`].
//!
//! ## Honest limitations
//!
//! - Key checks / steps for *shifted* Gauss valuations (`lambda0 != 0`) are
//!   not implemented; positive shifts are available as the honest
//!   augmentation `[v0, v(x) = lambda]` instead.
//! - Value computation ([`InductiveValuation::value`]) is fully generic in
//!   the chain length and base valuation.
//!
//! Every expected value in the tests was verified independently (sympy /
//! PARI-gp `idealprimedec` + `factorpadic` / first-principles p-adic
//! computations) before being asserted; see the stage reports.
//!
//! ## Example: extensions of the 2-adic valuation to Q[x]/(x^2+1)
//!
//! `x^2 + 1` is irreducible over `Q_2` (`-1 = 7 mod 8` is not a 2-adic
//! square) and `Q_2(i)/Q_2` is ramified: `(1+i)^2 = 2i` gives
//! `v(1+i) = 1/2`, so `e = 2`, `f = 1`:
//!
//! ```
//! use rustmath_rings::valuation::maclane::{mac_lane_approximants, QVal};
//! use rustmath_polynomials::UnivariatePolynomial;
//! use rustmath_rationals::Rational;
//!
//! // f = x^2 + 1
//! let f = UnivariatePolynomial::new(vec![
//!     Rational::from_i64(1), Rational::from_i64(0), Rational::from_i64(1),
//! ]);
//! let leaves = mac_lane_approximants(&f, 2).unwrap();
//! assert_eq!(leaves.len(), 1); // one extension of v_2 to Q_2(i)
//! let w = &leaves[0];
//! assert_eq!(w.ramification_index(), 2); // e = 2
//! assert_eq!(w.residue_degree(), 1); // f = 1
//! // the approximant is [ Gauss, v(x + 1) = 1/2 ]
//! assert_eq!(w.augmentations().len(), 1);
//! assert_eq!(w.augmentations()[0].lambda(), &QVal::from_frac(1, 2));
//! ```

use crate::function_field::typed::element::RationalFunction;
use crate::function_field::typed::place::Place;
use crate::valuation::residue_tower::{ResidueTower, TowerElt};
use rustmath_core::{EuclideanDomain, Field, MathError, Result, Ring};
use rustmath_integers::{prime::is_prime, Integer};
use rustmath_polynomials::{fp_factor, UnivariatePolynomial};
use rustmath_rationals::Rational;
use std::cmp::Ordering;
use std::fmt;

// ---------------------------------------------------------------------------
// Rational helpers
// ---------------------------------------------------------------------------

fn rat_i64(n: i64) -> Rational {
    Rational::from_i64(n)
}

fn rat_frac(n: i64, d: i64) -> Rational {
    Rational::new(n, d).expect("nonzero denominator")
}

/// gcd of two rationals: the positive generator of the group Z*a + Z*b.
/// `rat_gcd(0, b) = |b|`.
pub fn rat_gcd(a: &Rational, b: &Rational) -> Rational {
    if a.is_zero() {
        return b.abs();
    }
    if b.is_zero() {
        return a.abs();
    }
    let n1 = a.numerator().clone() * b.denominator().clone();
    let n2 = b.numerator().clone() * a.denominator().clone();
    let den = a.denominator().clone() * b.denominator().clone();
    Rational::new(n1.gcd(&n2), den)
        .expect("denominator nonzero")
        .abs()
}

/// Convert an `Integer` to `i64`, `Err` if it does not fit.
fn int_to_i64(n: &Integer) -> Result<i64> {
    if n.bit_length() > 62 {
        return Err(MathError::NumericalError(
            "maclane: integer too large for i64".to_string(),
        ));
    }
    Ok(n.to_i64())
}

// ---------------------------------------------------------------------------
// QVal: Q union {+infinity}
// ---------------------------------------------------------------------------

/// A value in `Q ∪ {+∞}`: the codomain of an inductive valuation.
///
/// (The pre-existing `valuation::valuation::ValuationValue` is `i64`-valued
/// and cannot represent the rational values of augmented valuations.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QVal {
    /// A finite rational value.
    Finite(Rational),
    /// `+∞` (the value of 0, and of the last key under an infinite
    /// augmentation).
    Infinity,
}

impl QVal {
    /// Finite integer value.
    pub fn from_int(n: i64) -> Self {
        QVal::Finite(rat_i64(n))
    }

    /// Finite value `n/d` (`d != 0`).
    pub fn from_frac(n: i64, d: i64) -> Self {
        QVal::Finite(rat_frac(n, d))
    }

    /// Is this `+∞`?
    pub fn is_infinite(&self) -> bool {
        matches!(self, QVal::Infinity)
    }

    /// The finite value, if any.
    pub fn finite(&self) -> Option<&Rational> {
        match self {
            QVal::Finite(r) => Some(r),
            QVal::Infinity => None,
        }
    }

    /// `self + other` (`∞` absorbs).
    pub fn add(&self, other: &QVal) -> QVal {
        match (self, other) {
            (QVal::Finite(a), QVal::Finite(b)) => QVal::Finite(a.clone() + b.clone()),
            _ => QVal::Infinity,
        }
    }

    /// `min(a, b)`.
    pub fn min_of(a: &QVal, b: &QVal) -> QVal {
        if a <= b {
            a.clone()
        } else {
            b.clone()
        }
    }
}

impl PartialOrd for QVal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QVal {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (QVal::Infinity, QVal::Infinity) => Ordering::Equal,
            (QVal::Infinity, QVal::Finite(_)) => Ordering::Greater,
            (QVal::Finite(_), QVal::Infinity) => Ordering::Less,
            (QVal::Finite(a), QVal::Finite(b)) => a.cmp(b),
        }
    }
}

impl fmt::Display for QVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QVal::Finite(r) => write!(f, "{}", r),
            QVal::Infinity => write!(f, "+Infinity"),
        }
    }
}

// ---------------------------------------------------------------------------
// Base valuations on the coefficient field K
// ---------------------------------------------------------------------------

/// A discrete (rank-one, rational-valued) valuation on the coefficient
/// field `K`, used as the base of a Gauss valuation on `K[x]`.
pub trait BaseValuation<K: Field + EuclideanDomain>: Clone + fmt::Debug {
    /// `v(c)`; `QVal::Infinity` iff `c = 0`.
    fn value(&self, c: &K) -> QVal;

    /// An element of value 1 (in the normalization of this valuation).
    fn uniformizer(&self) -> K;

    /// The positive generator of the value group `v(K^*)` (usually 1).
    fn value_group_generator(&self) -> Rational {
        rat_i64(1)
    }

    /// Human-readable description, e.g. "2-adic valuation".
    fn describe(&self) -> String;
}

/// The p-adic valuation `v_p` on `Q`, normalized with `v_p(p) = 1`.
///
/// This is the same valuation computed by the in-crate p-adics
/// (`crate::padics::PadicRational::valuation`); a test asserts the agreement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PAdicBaseValuation {
    p: i64,
    p_int: Integer,
}

impl PAdicBaseValuation {
    /// Create `v_p`. `Err` if `p` is not a prime in `[2, 2^31)` (the residue
    /// arithmetic uses `i64` and `fp_factor`).
    pub fn new(p: i64) -> Result<Self> {
        if p < 2 {
            return Err(MathError::InvalidArgument(format!(
                "PAdicBaseValuation: p = {} must be >= 2",
                p
            )));
        }
        if p >= (1_i64 << 31) {
            return Err(MathError::NotSupported(
                "PAdicBaseValuation: p >= 2^31 not supported by the i64 residue arithmetic"
                    .to_string(),
            ));
        }
        let p_int = Integer::from(p);
        if !is_prime(&p_int) {
            return Err(MathError::InvalidArgument(format!(
                "PAdicBaseValuation: p = {} is not prime",
                p
            )));
        }
        Ok(Self { p, p_int })
    }

    /// The prime p.
    pub fn prime(&self) -> i64 {
        self.p
    }
}

impl BaseValuation<Rational> for PAdicBaseValuation {
    fn value(&self, c: &Rational) -> QVal {
        if c.is_zero() {
            QVal::Infinity
        } else {
            QVal::from_int(c.valuation(&self.p_int) as i64)
        }
    }

    fn uniformizer(&self) -> Rational {
        rat_i64(self.p)
    }

    fn describe(&self) -> String {
        format!("{}-adic valuation", self.p)
    }
}

/// The valuation on the stage-1 typed rational function field `K(t)` at a
/// [`Place`] (finite or infinite), usable as the base of a Gauss valuation
/// on `K(t)[x]`.
#[derive(Clone, Debug)]
pub struct PlaceBaseValuation<K: Field + EuclideanDomain + fmt::Debug> {
    place: Place<K>,
}

impl<K: Field + EuclideanDomain + fmt::Debug> PlaceBaseValuation<K> {
    /// Wrap a place of `K(t)` as a base valuation.
    pub fn new(place: Place<K>) -> Self {
        Self { place }
    }

    /// The underlying place.
    pub fn place(&self) -> &Place<K> {
        &self.place
    }
}

impl<K: Field + EuclideanDomain + fmt::Debug> BaseValuation<RationalFunction<K>>
    for PlaceBaseValuation<K>
{
    fn value(&self, c: &RationalFunction<K>) -> QVal {
        match self.place.valuation(c) {
            None => QVal::Infinity,
            Some(v) => QVal::from_int(v),
        }
    }

    fn uniformizer(&self) -> RationalFunction<K> {
        self.place.uniformizer()
    }

    fn describe(&self) -> String {
        if self.place.is_infinite() {
            "valuation at the infinite place".to_string()
        } else {
            match self.place.polynomial() {
                Some(p) => format!("valuation at the place ({})", p),
                None => "valuation at a finite place".to_string(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// phi-adic expansion
// ---------------------------------------------------------------------------

/// The `phi`-adic expansion of `f`: coefficients `f_0, ..., f_n` with
/// `f = sum f_i phi^i` and `deg f_i < deg phi`. Returns an empty vector for
/// `f = 0`. `Err` if `deg phi < 1`.
pub fn phi_adic_expansion<K: Field + EuclideanDomain>(
    f: &UnivariatePolynomial<K>,
    phi: &UnivariatePolynomial<K>,
) -> Result<Vec<UnivariatePolynomial<K>>> {
    match phi.degree() {
        Some(d) if d >= 1 => {}
        _ => {
            return Err(MathError::InvalidArgument(
                "phi_adic_expansion: phi must have degree >= 1".to_string(),
            ))
        }
    }
    let mut out = Vec::new();
    let mut g = f.clone();
    while !g.is_zero() {
        let (q, r) = g.div_rem(phi)?;
        out.push(r);
        g = q;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Inductive valuations
// ---------------------------------------------------------------------------

/// One augmentation step `v(phi) = lambda` in an inductive valuation.
#[derive(Clone, Debug)]
pub struct Augmentation<K: Field + EuclideanDomain> {
    phi: UnivariatePolynomial<K>,
    lambda: QVal,
    /// Degree of the residual (irreducible) polynomial of `phi` as a key of
    /// the previous level; the residue degree `F` is the product of these.
    residual_degree: usize,
}

impl<K: Field + EuclideanDomain> Augmentation<K> {
    /// The key polynomial of this step.
    pub fn phi(&self) -> &UnivariatePolynomial<K> {
        &self.phi
    }

    /// The assigned value `lambda` (finite rational, or `+∞` for a final
    /// pseudo-valuation).
    pub fn lambda(&self) -> &QVal {
        &self.lambda
    }

    /// The residual degree contributed by this step to `F`.
    pub fn residual_degree(&self) -> usize {
        self.residual_degree
    }
}

/// A MacLane inductive valuation on `K[x]`:
/// `[v_0(x) = lambda0, v(phi_1) = lambda_1, ..., v(phi_k) = lambda_k]`
/// where `v_0` is the (shifted) Gauss valuation over a base valuation on `K`.
///
/// Values are computed recursively by the phi-adic min formula at each level.
/// Chains are only constructible through [`Self::gauss`],
/// [`Self::gauss_shifted`] and the key-verified `augment`, so every value of
/// this type is a genuine valuation (or pseudo-valuation when the last
/// `lambda` is `+∞`).
#[derive(Clone, Debug)]
pub struct InductiveValuation<K: Field + EuclideanDomain, V: BaseValuation<K>> {
    base: V,
    lambda0: Rational,
    augmentations: Vec<Augmentation<K>>,
}

impl<K: Field + EuclideanDomain, V: BaseValuation<K>> InductiveValuation<K, V> {
    /// The Gauss valuation: `v_0(sum a_i x^i) = min_i v(a_i)`, `v_0(x) = 0`.
    pub fn gauss(base: V) -> Self {
        Self {
            base,
            lambda0: Rational::zero(),
            augmentations: Vec::new(),
        }
    }

    /// The shifted Gauss (monomial) valuation `v_0(x) = lambda0`:
    /// `v_0(sum a_i x^i) = min_i (v(a_i) + i*lambda0)`, for ANY rational
    /// `lambda0` (also negative). This is a valuation for every `lambda0`
    /// (the associated graded ring is a domain); multiplicativity is gated
    /// by tests.
    pub fn gauss_shifted(base: V, lambda0: Rational) -> Self {
        Self {
            base,
            lambda0,
            augmentations: Vec::new(),
        }
    }

    /// The base valuation on `K`.
    pub fn base(&self) -> &V {
        &self.base
    }

    /// The Gauss-level value of `x`.
    pub fn lambda0(&self) -> &Rational {
        &self.lambda0
    }

    /// The augmentation chain (possibly empty).
    pub fn augmentations(&self) -> &[Augmentation<K>] {
        &self.augmentations
    }

    /// Number of augmentations.
    pub fn level(&self) -> usize {
        self.augmentations.len()
    }

    /// Is this a plain (shifted) Gauss valuation?
    pub fn is_gauss(&self) -> bool {
        self.augmentations.is_empty()
    }

    /// Is this final, i.e. is the last `lambda` infinite (a pseudo-valuation
    /// that can no longer be augmented)?
    pub fn is_final(&self) -> bool {
        self.augmentations
            .last()
            .map(|a| a.lambda.is_infinite())
            .unwrap_or(false)
    }

    /// The last key polynomial, if any augmentation has been made.
    pub fn last_key(&self) -> Option<&UnivariatePolynomial<K>> {
        self.augmentations.last().map(|a| &a.phi)
    }

    /// Drop the last augmentation (internal, for collapsing same-degree
    /// augmentations).
    fn truncated(&self) -> Self {
        let mut t = self.clone();
        t.augmentations.pop();
        t
    }

    /// The value `w(f)`, computed by the recursive phi-adic min formula.
    /// `w(0) = +∞`.
    pub fn value(&self, f: &UnivariatePolynomial<K>) -> QVal {
        self.value_at_level(f, self.augmentations.len())
    }

    fn value_at_level(&self, f: &UnivariatePolynomial<K>, level: usize) -> QVal {
        if f.is_zero() {
            return QVal::Infinity;
        }
        if level == 0 {
            let mut best = QVal::Infinity;
            for (i, c) in f.coefficients().iter().enumerate() {
                if c.is_zero() {
                    continue;
                }
                let mut term = self.base.value(c);
                if !self.lambda0.is_zero() {
                    term = term.add(&QVal::Finite(rat_i64(i as i64) * self.lambda0.clone()));
                }
                best = QVal::min_of(&best, &term);
            }
            return best;
        }
        let aug = &self.augmentations[level - 1];
        let mut best = QVal::Infinity;
        let mut g = f.clone();
        let mut i: i64 = 0;
        while !g.is_zero() {
            let (q, r) = g.div_rem(&aug.phi).expect("key polynomial is nonzero");
            if !r.is_zero() {
                let vi = self.value_at_level(&r, level - 1);
                let term = match &aug.lambda {
                    QVal::Infinity => {
                        if i == 0 {
                            vi
                        } else {
                            QVal::Infinity
                        }
                    }
                    QVal::Finite(l) => vi.add(&QVal::Finite(rat_i64(i) * l.clone())),
                };
                best = QVal::min_of(&best, &term);
            }
            g = q;
            i += 1;
        }
        best
    }

    /// The positive generator of the value group
    /// `Gamma_w = Gamma_base + Z*lambda0 + sum_i Z*lambda_i`
    /// (infinite lambdas contribute nothing).
    pub fn value_group_generator(&self) -> Rational {
        let mut g = self.base.value_group_generator();
        if !self.lambda0.is_zero() {
            g = rat_gcd(&g, &self.lambda0);
        }
        for aug in &self.augmentations {
            if let QVal::Finite(l) = &aug.lambda {
                if !l.is_zero() {
                    g = rat_gcd(&g, l);
                }
            }
        }
        g
    }

    /// The ramification index `E = [Gamma_w : Gamma_base]`.
    pub fn ramification_index(&self) -> u64 {
        let ratio = self.base.value_group_generator() / self.value_group_generator();
        debug_assert!(ratio.is_integer());
        int_to_i64(ratio.numerator()).expect("E fits in i64") as u64
            / int_to_i64(ratio.denominator()).expect("E fits in i64") as u64
    }

    /// The residue degree `F = [k_w : k_base]`, the product of the residual
    /// degrees of the keys along the chain.
    pub fn residue_degree(&self) -> u64 {
        self.augmentations
            .iter()
            .map(|a| a.residual_degree as u64)
            .product()
    }
}

impl<K: Field + EuclideanDomain, V: BaseValuation<K>> fmt::Display for InductiveValuation<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[ Gauss valuation induced by {}", self.base.describe())?;
        if !self.lambda0.is_zero() {
            write!(f, ", v(x) = {}", self.lambda0)?;
        }
        for aug in &self.augmentations {
            write!(f, ", v({}) = {}", aug.phi, aug.lambda)?;
        }
        write!(f, " ]")
    }
}

// ---------------------------------------------------------------------------
// Key polynomials, augmentation and MacLane steps over (Q, v_p)
// ---------------------------------------------------------------------------

/// Cached residue machinery for a finite prefix of a p-adic inductive
/// chain: the residue-field tower `psi_1, ..., psi_k`, the relative
/// ramification indices `tau_j = [Gamma_j : Gamma_{j-1}]`, the coherent
/// normalizers `Q_j` (standard elements of value `tau_j * lambda_j`), and
/// the value-group generators `gens[j]` of `Gamma_j`.
struct ResidueData {
    tower: ResidueTower,
    taus: Vec<usize>,
    qs: Vec<UnivariatePolynomial<Rational>>,
    gens: Vec<Rational>,
}

/// `p^c` as an exact rational (`c` may be negative).
fn p_power(p: i64, c: i64) -> Rational {
    let pk = Integer::from(p).pow(c.unsigned_abs() as u32);
    if c >= 0 {
        Rational::new(pk, Integer::one()).expect("nonzero")
    } else {
        Rational::new(Integer::one(), pk).expect("nonzero")
    }
}

/// Solve `sum_j n_j * cols[j] = rhs` over `GF(p)` by Gaussian elimination;
/// `None` if inconsistent.
fn solve_mod_p(cols: &[Vec<i64>], rhs: &[i64], p: i64) -> Option<Vec<i64>> {
    let nrows = rhs.len();
    let ncols = cols.len();
    let mut mat: Vec<Vec<i64>> = (0..nrows)
        .map(|r| {
            let mut row: Vec<i64> = cols
                .iter()
                .map(|c| c.get(r).copied().unwrap_or(0).rem_euclid(p))
                .collect();
            row.push(rhs[r].rem_euclid(p));
            row
        })
        .collect();
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    let mut rank_row = 0usize;
    for col in 0..ncols {
        if rank_row >= nrows {
            break;
        }
        let Some(piv) = (rank_row..nrows).find(|&r| mat[r][col] != 0) else {
            continue;
        };
        mat.swap(rank_row, piv);
        let inv = fp_factor::mod_inv(mat[rank_row][col], p)?;
        for j in col..=ncols {
            mat[rank_row][j] =
                (mat[rank_row][j] as i128 * inv as i128).rem_euclid(p as i128) as i64;
        }
        for r in 0..nrows {
            if r != rank_row && mat[r][col] != 0 {
                let f = mat[r][col];
                for j in col..=ncols {
                    mat[r][j] = (mat[r][j] as i128 - f as i128 * mat[rank_row][j] as i128)
                        .rem_euclid(p as i128) as i64;
                }
            }
        }
        pivots.push((rank_row, col));
        rank_row += 1;
    }
    for row in mat.iter().take(nrows).skip(rank_row) {
        if row[ncols] != 0 {
            return None;
        }
    }
    let mut sol = vec![0i64; ncols];
    for (r, c) in pivots {
        sol[c] = mat[r][ncols];
    }
    Some(sol)
}

/// Result of a key-polynomial check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyCheck {
    /// `phi` is a key polynomial; `residual_degree` is the degree of its
    /// residual polynomial over the current residue field (the relative
    /// residue-degree contribution of augmenting with it).
    Key {
        /// Degree of the residual polynomial of `phi`.
        residual_degree: usize,
    },
    /// `phi` is not a key polynomial, with the reason.
    NotKey(String),
}

/// An inductive valuation on `Q[x]` over the p-adic valuation `v_p`.
pub type PAdicInductiveValuation = InductiveValuation<Rational, PAdicBaseValuation>;

impl InductiveValuation<Rational, PAdicBaseValuation> {
    fn p(&self) -> i64 {
        self.base.prime()
    }

    /// Residue of a p-adic unit `q` (requires `v_p(q) = 0`) in `GF(p)`,
    /// as an element of `[0, p)`.
    fn residue_unit(&self, q: &Rational) -> Result<i64> {
        let p = self.p();
        let p_int = Integer::from(p);
        if q.is_zero() || q.valuation(&p_int) != 0 {
            return Err(MathError::InvalidArgument(
                "maclane: residue of a non-unit".to_string(),
            ));
        }
        let num = ((q.numerator().clone() % p_int.clone()) + p_int.clone()) % p_int.clone();
        let den = ((q.denominator().clone() % p_int.clone()) + p_int.clone()) % p_int;
        let num = int_to_i64(&num)?;
        let den = int_to_i64(&den)?;
        let den_inv = fp_factor::mod_inv(den, p).ok_or_else(|| {
            MathError::InvalidArgument("maclane: denominator not invertible mod p".to_string())
        })?;
        Ok(((num as i128 * den_inv as i128) % p as i128) as i64)
    }

    /// Reduce a p-integral polynomial (all coefficients with `v_p >= 0`)
    /// coefficient-wise mod p.
    fn reduce_poly_mod_p(&self, g: &UnivariatePolynomial<Rational>) -> Result<Vec<i64>> {
        let p = self.p();
        let p_int = Integer::from(p);
        let mut out = Vec::with_capacity(g.coefficients().len());
        for c in g.coefficients() {
            if c.is_zero() {
                out.push(0);
                continue;
            }
            let v = c.valuation(&p_int);
            if v < 0 {
                return Err(MathError::InvalidArgument(
                    "maclane: polynomial is not p-integral".to_string(),
                ));
            }
            if v > 0 {
                out.push(0);
            } else {
                out.push(self.residue_unit(c)?);
            }
        }
        Ok(fp_factor::trim(&out))
    }

    // -----------------------------------------------------------------
    // General residue machinery (any augmentation level): the residue
    // field tower kappa_0 = GF(p) ⊂ kappa_j = kappa_{j-1}[y_{j-1}]/(psi_j),
    // the recursive reduction map, residual polynomials with the coherent
    // Q_j^t normalization, and key-polynomial lifting by GF(p)-linear
    // algebra over the standard degree-< deg(phi_j) monomials.
    // -----------------------------------------------------------------

    /// Build the [`ResidueData`] (tower `psi_1..psi_k`, relative
    /// ramifications `tau_j`, normalizers `Q_j`, value-group generators)
    /// for this chain. Requires `lambda0 = 0` and all finite lambdas up to
    /// `levels` (callers reject final valuations first).
    fn residue_data(&self, levels: usize) -> Result<ResidueData> {
        if !self.lambda0.is_zero() {
            return Err(MathError::NotSupported(
                "maclane: residue computations over shifted Gauss valuations not implemented"
                    .to_string(),
            ));
        }
        let mut data = ResidueData {
            tower: ResidueTower::new(self.p())?,
            taus: Vec::new(),
            qs: Vec::new(),
            gens: vec![rat_i64(1)],
        };
        for j in 1..=levels {
            let aug = &self.augmentations[j - 1];
            let lambda = aug.lambda.finite().ok_or_else(|| {
                MathError::InvalidArgument(
                    "maclane: residue data of an infinite augmentation".to_string(),
                )
            })?;
            let gen_prev = data.gens[j - 1].clone();
            let gen_new = rat_gcd(&gen_prev, lambda);
            let ratio = gen_prev.clone() / gen_new.clone();
            if !ratio.is_integer() {
                return Err(MathError::NumericalError(
                    "maclane: internal error: non-integral value-group index".to_string(),
                ));
            }
            let tau = int_to_i64(ratio.numerator())? as usize;
            let q_j = self.element_with_valuation(
                j - 1,
                &(rat_i64(tau as i64) * lambda.clone()),
                &data,
            )?;
            let psi_j = if j == 1 {
                self.reduce_poly_mod_p(&aug.phi)?
                    .into_iter()
                    .map(TowerElt::Base)
                    .collect::<Vec<_>>()
            } else {
                // the residual polynomial of phi_j w.r.t. the level-(j-1)
                // prefix, which must start at i0 = 0 (phi_j is a key there)
                let (r, i0) = self.residual_polynomial_general(j - 1, &aug.phi, &data)?;
                if i0 != 0 {
                    return Err(MathError::NumericalError(
                        "maclane: internal error: key residual polynomial divisible by y"
                            .to_string(),
                    ));
                }
                data.tower.poly_monic(&r)?
            };
            data.tower.push_level(psi_j)?; // re-certifies irreducibility
            data.taus.push(tau);
            data.qs.push(q_j);
            data.gens.push(gen_new);
        }
        Ok(data)
    }

    /// An element of `Q[x]` in standard form `p^c * prod phi_i^{a_i}`
    /// (`0 <= a_i < tau_i`, `c` any integer) with `w_level`-value exactly
    /// `t`. `Err` if `t` is not in the value group of the level.
    fn element_with_valuation(
        &self,
        level: usize,
        t: &Rational,
        data: &ResidueData,
    ) -> Result<UnivariatePolynomial<Rational>> {
        let mut rem = t.clone();
        let mut result = UnivariatePolynomial::one();
        for i in (1..=level).rev() {
            let lambda_i = self.augmentations[i - 1]
                .lambda
                .finite()
                .expect("residue data exists only for finite chains")
                .clone();
            let gen_prev = &data.gens[i - 1];
            let mut found = false;
            for a in 0..data.taus[i - 1] {
                let cand = rem.clone() - rat_i64(a as i64) * lambda_i.clone();
                if (cand.clone() / gen_prev.clone()).is_integer() {
                    for _ in 0..a {
                        result = result * self.augmentations[i - 1].phi.clone();
                    }
                    rem = cand;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(MathError::InvalidArgument(format!(
                    "maclane: {} is not in the value group at level {}",
                    t, level
                )));
            }
        }
        if !rem.is_integer() {
            return Err(MathError::InvalidArgument(format!(
                "maclane: {} is not in the value group at level {}",
                t, level
            )));
        }
        let c = int_to_i64(rem.numerator())?;
        Ok(result.scalar_mul(&p_power(self.p(), c)))
    }

    /// The reduction of `g` (with `w_level(g) = 0`) into the residue ring
    /// `kappa_level[y_level]`, computed recursively through the tower:
    /// `red_0` is coefficient-wise reduction mod p, and
    /// `red_j(g) = sum_t [red_{j-1}(g_{t*tau_j} Q_j^t) mod psi_j] y_j^t`
    /// over the phi_j-adic expansion of `g`.
    fn reduce_general(
        &self,
        level: usize,
        g: &UnivariatePolynomial<Rational>,
        data: &ResidueData,
    ) -> Result<Vec<TowerElt>> {
        if self.value_at_level(g, level) != QVal::from_int(0) {
            return Err(MathError::NumericalError(
                "maclane: internal error: reduce of an element of nonzero value".to_string(),
            ));
        }
        if level == 0 {
            return Ok(self
                .reduce_poly_mod_p(g)?
                .into_iter()
                .map(TowerElt::Base)
                .collect());
        }
        let aug = &self.augmentations[level - 1];
        let lambda = aug.lambda.finite().expect("finite chain").clone();
        let tau = data.taus[level - 1];
        let exp = phi_adic_expansion(g, &aug.phi)?;
        let sub_tower = data.tower.truncate(level - 1);
        let psi = data.tower.modulus_at(level);
        let mut out: Vec<TowerElt> = Vec::new();
        let mut q_pow = UnivariatePolynomial::<Rational>::one();
        let mut next_t = 0usize;
        for (i, gi) in exp.iter().enumerate() {
            if gi.is_zero() {
                continue;
            }
            let u = match self.value_at_level(gi, level - 1) {
                QVal::Finite(u) => u,
                QVal::Infinity => unreachable!("nonzero coefficient"),
            };
            if !(u + rat_i64(i as i64) * lambda.clone()).is_zero() {
                continue; // above the critical line: reduces to 0
            }
            if i % tau != 0 {
                return Err(MathError::NumericalError(
                    "maclane: internal error: critical index of a value-0 element not divisible by tau"
                        .to_string(),
                ));
            }
            let t = i / tau;
            while next_t < t {
                q_pow = q_pow * data.qs[level - 1].clone();
                next_t += 1;
            }
            let c = gi.clone() * q_pow.clone();
            let r = self.reduce_general(level - 1, &c, data)?;
            let rem = sub_tower.poly_divmod(&r, psi)?.1;
            while out.len() <= t {
                out.push(data.tower.e_zero(level));
            }
            out[t] = data.tower.make_ext(level, rem)?;
        }
        if out.is_empty() {
            return Err(MathError::NumericalError(
                "maclane: internal error: reduce of a value-0 element is zero".to_string(),
            ));
        }
        Ok(out)
    }

    /// The residual polynomial of nonzero `g` with respect to the level-
    /// `level` prefix of this chain (GMN normalization, coherent with
    /// [`Self::reduce_general`], up to one global unit):
    ///
    /// `R(y) = sum_t [red_{level-1}(g_{i0 + t*tau} Q^t N_0) mod psi] y^t`
    ///
    /// over `kappa_level`, where the support is the critical line
    /// `u_i + i*lambda = w(g)` of the phi-adic expansion, `i0` is its least
    /// index and `N_0` is a standard element with value `-u_{i0}`. Returns
    /// `(R, i0)`; `R(0) != 0` by construction of `i0`.
    fn residual_polynomial_general(
        &self,
        level: usize,
        g: &UnivariatePolynomial<Rational>,
        data: &ResidueData,
    ) -> Result<(Vec<TowerElt>, usize)> {
        if level == 0 || level > self.augmentations.len() {
            return Err(MathError::InvalidArgument(
                "maclane: residual polynomial level out of range".to_string(),
            ));
        }
        if g.is_zero() {
            return Err(MathError::InvalidArgument(
                "maclane: residual polynomial of zero".to_string(),
            ));
        }
        let aug = &self.augmentations[level - 1];
        let lambda = match &aug.lambda {
            QVal::Finite(l) => l.clone(),
            QVal::Infinity => {
                return Err(MathError::InvalidArgument(
                    "maclane: final valuation has no residual polynomials".to_string(),
                ))
            }
        };
        let tau = data.taus[level - 1];
        let exp = phi_adic_expansion(g, &aug.phi)?;
        let mut us: Vec<Option<Rational>> = Vec::with_capacity(exp.len());
        for c in &exp {
            us.push(match self.value_at_level(c, level - 1) {
                QVal::Finite(u) => Some(u),
                QVal::Infinity => None,
            });
        }
        let mut mu: Option<Rational> = None;
        for (i, u) in us.iter().enumerate() {
            if let Some(u) = u {
                let val = u.clone() + rat_i64(i as i64) * lambda.clone();
                if mu.as_ref().map(|m| &val < m).unwrap_or(true) {
                    mu = Some(val);
                }
            }
        }
        let mu = mu.ok_or_else(|| {
            MathError::InvalidArgument("maclane: residual polynomial of zero".to_string())
        })?;
        let on_line = |i: usize| -> bool {
            match &us[i] {
                None => false,
                Some(u) => u.clone() + rat_i64(i as i64) * lambda.clone() == mu,
            }
        };
        let i0 = (0..us.len()).find(|&i| on_line(i)).expect("mu attained");
        let imax = (0..us.len()).rev().find(|&i| on_line(i)).expect("mu attained");
        for i in i0..=imax {
            if on_line(i) && (i - i0) % tau != 0 {
                return Err(MathError::NumericalError(
                    "maclane: internal error: critical index not congruent mod tau".to_string(),
                ));
            }
        }
        let u_i0 = us[i0].clone().expect("on line");
        let n0 = self.element_with_valuation(level - 1, &(-u_i0), data)?;
        let m = (imax - i0) / tau;
        let sub_tower = data.tower.truncate(level - 1);
        let psi = data.tower.modulus_at(level);
        let mut r: Vec<TowerElt> = Vec::with_capacity(m + 1);
        let mut q_pow = UnivariatePolynomial::<Rational>::one();
        for t in 0..=m {
            let i = i0 + t * tau;
            if on_line(i) {
                let c = exp[i].clone() * q_pow.clone() * n0.clone();
                let red = self.reduce_general(level - 1, &c, data)?;
                let rem = sub_tower.poly_divmod(&red, psi)?.1;
                r.push(data.tower.make_ext(level, rem)?);
            } else {
                r.push(data.tower.e_zero(level));
            }
            if t < m {
                q_pow = q_pow * data.qs[level - 1].clone();
            }
        }
        Ok((r, i0))
    }

    /// Lift a monic irreducible residual factor `psi` over `kappa_level`
    /// (with `psi(0) != 0`) to a key polynomial of the level-`level` chain
    /// with residual polynomial an associate of `psi`:
    ///
    /// `phi_new = phi^{m*tau} + sum_{t<m} c_t phi^{t*tau}`, `deg c_t < deg phi`,
    ///
    /// where each `c_t` is a GF(p)-combination of standard monomials
    /// `p^c x^{a_0} prod phi_i^{a_i}` of value `(m-t)*tau*lambda`, solved by
    /// linear algebra so that `red(c_t Q^t N_0) = gamma * psi_t`
    /// (`gamma = red(Q^m N_0)`). The result is CERTIFIED: its value and its
    /// residual polynomial (up to the unit `gamma`) are recomputed and
    /// checked, so a lifting bug is an `Err`, never a wrong key.
    fn lift_residual_to_key(
        &self,
        level: usize,
        psi: &[TowerElt],
        data: &ResidueData,
    ) -> Result<UnivariatePolynomial<Rational>> {
        let tower = &data.tower;
        let aug = &self.augmentations[level - 1];
        let lambda = aug.lambda.finite().expect("finite chain").clone();
        let tau = data.taus[level - 1];
        let m = tower.poly_degree(psi);
        if m < 1 {
            return Err(MathError::InvalidArgument(
                "maclane: lift of a constant residual factor".to_string(),
            ));
        }
        let m = m as usize;
        if psi[m] != tower.e_one(level) {
            return Err(MathError::NumericalError(
                "maclane: internal error: residual factor is not monic".to_string(),
            ));
        }
        if tower.poly_degree(&[psi[0].clone()]) < 0 {
            return Err(MathError::NumericalError(
                "maclane: internal error: residual factor divisible by y".to_string(),
            ));
        }
        let p = self.p();
        // gamma = red(Q^m N_0), N_0 of value -m*tau*lambda
        let full_value = rat_i64((m * tau) as i64) * lambda.clone();
        let n0 = self.element_with_valuation(level - 1, &(-full_value.clone()), data)?;
        let mut q_m = UnivariatePolynomial::<Rational>::one();
        for _ in 0..m {
            q_m = q_m * data.qs[level - 1].clone();
        }
        let sub_tower = tower.truncate(level - 1);
        let psi_mod = tower.modulus_at(level);
        let reduce_mod = |c: &UnivariatePolynomial<Rational>| -> Result<TowerElt> {
            let red = self.reduce_general(level - 1, c, data)?;
            let rem = sub_tower.poly_divmod(&red, psi_mod)?.1;
            tower.make_ext(level, rem)
        };
        let gamma = reduce_mod(&(q_m * n0.clone()))?;
        if tower.e_is_zero(&gamma) {
            return Err(MathError::NumericalError(
                "maclane: internal error: gamma = 0 in key lift".to_string(),
            ));
        }
        // standard monomials x^{a_0} prod phi_i^{a_i} of degree < deg(phi_level)
        let monomials = self.standard_monomials(level, data)?;
        // assemble phi_new = phi^{m tau} + sum c_t phi^{t tau}
        let mut phi_pow_tau = UnivariatePolynomial::<Rational>::one();
        for _ in 0..tau {
            phi_pow_tau = phi_pow_tau * aug.phi.clone();
        }
        let mut phi_new = UnivariatePolynomial::<Rational>::one();
        for _ in 0..m {
            phi_new = phi_new * phi_pow_tau.clone();
        }
        let mut q_pow = UnivariatePolynomial::<Rational>::one();
        let mut phi_tau_pow = UnivariatePolynomial::<Rational>::one();
        for (t, psi_t) in psi.iter().enumerate().take(m) {
            if !tower.e_is_zero(psi_t) {
                let target = tower.e_mul(level, &gamma, psi_t);
                let v_t = rat_i64(((m - t) * tau) as i64) * lambda.clone();
                // columns: red(M * p^{c_M} * Q^t * N_0) for each monomial M,
                // where p^{c_M} normalizes M to value v_t
                let mut scaled_mons: Vec<(Rational, UnivariatePolynomial<Rational>)> = Vec::new();
                for mono in &monomials {
                    let val = match self.value_at_level(mono, level - 1) {
                        QVal::Finite(v) => v,
                        QVal::Infinity => continue,
                    };
                    let shift = v_t.clone() - val.clone();
                    if !shift.is_integer() {
                        continue;
                    }
                    let scaled = mono.scalar_mul(&p_power(p, int_to_i64(shift.numerator())?));
                    scaled_mons.push((shift, scaled));
                }
                // prefer nonnegative p-shifts as pivots (integral lifts)
                scaled_mons.sort_by(|a, b| b.0.cmp(&a.0));
                let mut cols: Vec<Vec<i64>> = Vec::new();
                let mut mons: Vec<UnivariatePolynomial<Rational>> = Vec::new();
                for (_, scaled) in &scaled_mons {
                    let red = reduce_mod(&(scaled.clone() * q_pow.clone() * n0.clone()))?;
                    cols.push(tower.flatten(&red));
                    mons.push(scaled.clone());
                }
                let rhs = tower.flatten(&target);
                let sol = solve_mod_p(&cols, &rhs, p).ok_or_else(|| {
                    MathError::NumericalError(
                        "maclane: internal error: monomial reductions do not span kappa"
                            .to_string(),
                    )
                })?;
                let mut c_t = UnivariatePolynomial::<Rational>::zero();
                for (n_m, mono) in sol.iter().zip(mons.iter()) {
                    if *n_m != 0 {
                        c_t = c_t + mono.scalar_mul(&rat_i64(*n_m));
                    }
                }
                phi_new = phi_new + c_t * phi_tau_pow.clone();
            }
            q_pow = q_pow * data.qs[level - 1].clone();
            phi_tau_pow = phi_tau_pow * phi_pow_tau.clone();
        }
        // CERTIFY the lift
        let expected_deg = m * tau * aug.phi.degree().expect("key degree");
        if phi_new.degree() != Some(expected_deg) || !phi_new.is_monic() {
            return Err(MathError::NumericalError(
                "maclane: internal error: lifted key has wrong degree/leading coefficient"
                    .to_string(),
            ));
        }
        let expected_val = rat_i64((m * tau) as i64) * lambda.clone();
        if self.value_at_level(&phi_new, level) != QVal::Finite(expected_val) {
            return Err(MathError::NumericalError(
                "maclane: internal error: lifted key has wrong value".to_string(),
            ));
        }
        let (r_new, i0) = self.residual_polynomial_general(level, &phi_new, data)?;
        if i0 != 0 || tower.poly_degree(&r_new) != m as i64 {
            return Err(MathError::NumericalError(
                "maclane: internal error: lifted key residual has wrong shape".to_string(),
            ));
        }
        let lead = r_new[m].clone();
        for (t, psi_t) in psi.iter().enumerate() {
            let expect = tower.e_mul(level, &lead, psi_t);
            if r_new[t] != expect {
                return Err(MathError::NumericalError(
                    "maclane: internal error: lifted key residual is not an associate of the factor"
                        .to_string(),
                ));
            }
        }
        Ok(phi_new)
    }

    /// The standard monomials `x^{a_0} prod_{i=1}^{level-1} phi_i^{a_i}`
    /// with `a_0 < deg psi_1` and `a_i < tau_i * deg psi_{i+1}`; all have
    /// degree < deg(phi_level) and their normalized reductions span
    /// `kappa_level` (certified downstream by the linear solves).
    fn standard_monomials(
        &self,
        level: usize,
        data: &ResidueData,
    ) -> Result<Vec<UnivariatePolynomial<Rational>>> {
        let mut ranges: Vec<usize> = vec![data.tower.modulus_degree(1)];
        for i in 1..level {
            ranges.push(data.taus[i - 1] * data.tower.modulus_degree(i + 1));
        }
        let mut out: Vec<UnivariatePolynomial<Rational>> =
            vec![UnivariatePolynomial::one()];
        for (i, &range) in ranges.iter().enumerate() {
            let base = if i == 0 {
                UnivariatePolynomial::new(vec![rat_i64(0), rat_i64(1)])
            } else {
                self.augmentations[i - 1].phi.clone()
            };
            let mut next = Vec::with_capacity(out.len() * range);
            for m in &out {
                let mut pow = UnivariatePolynomial::<Rational>::one();
                for _ in 0..range {
                    next.push(m.clone() * pow.clone());
                    pow = pow * base.clone();
                }
            }
            out = next;
        }
        Ok(out)
    }

    /// Is `phi` a key polynomial for this valuation? Implements the standard
    /// effective criterion (monic + equivalence-irreducible + v-minimal):
    ///
    /// - Gauss level: p-integral with irreducible reduction mod p.
    /// - Level `k >= 1`: `deg phi` a multiple of `tau_k * d_k`, one-sided
    ///   Newton polygon (`w(phi) = (deg phi / d_k) * lambda_k`), residual
    ///   polynomial irreducible over the residue tower `kappa_k` and not
    ///   divisible by `y`; plus the same-degree equivalent-key cases.
    ///
    /// `Err(NotSupported)` only for shifted Gauss valuations
    /// (`lambda0 != 0`).
    pub fn is_key(&self, phi: &UnivariatePolynomial<Rational>) -> Result<KeyCheck> {
        if !self.lambda0.is_zero() {
            return Err(MathError::NotSupported(
                "maclane: key checks for shifted Gauss valuations not implemented".to_string(),
            ));
        }
        if self.is_final() {
            return Err(MathError::InvalidArgument(
                "maclane: a final (infinite) valuation has no key polynomials".to_string(),
            ));
        }
        let n = match phi.degree() {
            Some(n) if n >= 1 => n,
            _ => return Ok(KeyCheck::NotKey("phi must have degree >= 1".to_string())),
        };
        if !phi.is_monic() {
            return Ok(KeyCheck::NotKey("phi must be monic".to_string()));
        }
        match self.augmentations.len() {
            0 => {
                let p_int = Integer::from(self.p());
                for c in phi.coefficients() {
                    if !c.is_zero() && c.valuation(&p_int) < 0 {
                        return Ok(KeyCheck::NotKey(
                            "phi must be p-integral".to_string(),
                        ));
                    }
                }
                if n == 1 {
                    return Ok(KeyCheck::Key { residual_degree: 1 });
                }
                let fbar = self.reduce_poly_mod_p(phi)?;
                if is_irreducible_fp(&fbar, self.p())? {
                    Ok(KeyCheck::Key { residual_degree: n })
                } else {
                    Ok(KeyCheck::NotKey(
                        "reduction of phi mod p is not irreducible".to_string(),
                    ))
                }
            }
            level => {
                let aug = &self.augmentations[level - 1];
                let d = aug.phi.degree().expect("key has degree >= 1");
                let lambda = aug
                    .lambda
                    .finite()
                    .expect("non-final valuation has finite lambda")
                    .clone();
                let tau = self.tau_at(level)?;
                if n % d != 0 {
                    return Ok(KeyCheck::NotKey(format!(
                        "deg phi = {} is not a multiple of the key degree {}",
                        n, d
                    )));
                }
                let ell = n / d;
                if ell % tau != 0 {
                    if n == d {
                        // tau >= 2, same degree: keys are exactly phi_old + r
                        // with w(r) > lambda (equivalent keys).
                        let diff = phi.clone() - aug.phi.clone();
                        return if self.value(&diff) > QVal::Finite(lambda) {
                            Ok(KeyCheck::Key {
                                residual_degree: aug.residual_degree,
                            })
                        } else {
                            Ok(KeyCheck::NotKey(
                                "phi is not equivalent to the key (w(phi - key) <= lambda)"
                                    .to_string(),
                            ))
                        };
                    }
                    return Ok(KeyCheck::NotKey(format!(
                        "deg phi / d = {} is not a multiple of tau = {}",
                        ell, tau
                    )));
                }
                let m = ell / tau;
                // v-minimality: the phi_old-Newton polygon of phi must be
                // one-sided of slope -lambda ending at (ell, 0), i.e.
                // w(phi) = ell * lambda.
                let expected = rat_i64(ell as i64) * lambda.clone();
                if self.value(phi) != QVal::Finite(expected) {
                    return Ok(KeyCheck::NotKey(
                        "phi is not v-minimal (w(phi) < deg(phi)/d * lambda)".to_string(),
                    ));
                }
                let data = self.residue_data(level)?;
                let (r, i0) = self.residual_polynomial_general(level, phi, &data)?;
                if i0 == 0 {
                    // residual polynomial R of degree m with R(0) != 0
                    if data.tower.poly_degree(&r) == m as i64 && data.tower.is_irreducible(&r)? {
                        Ok(KeyCheck::Key { residual_degree: m })
                    } else {
                        Ok(KeyCheck::NotKey(
                            "residual polynomial is not irreducible".to_string(),
                        ))
                    }
                } else if n == d {
                    // ell = 1, tau = 1, support {1}: R = y, phi = key + r
                    // with w(r) > lambda: an equivalent key.
                    Ok(KeyCheck::Key {
                        residual_degree: aug.residual_degree,
                    })
                } else {
                    Ok(KeyCheck::NotKey(
                        "residual polynomial is divisible by y (phi is equivalence-divisible by a power of the key)"
                            .to_string(),
                    ))
                }
            }
        }
    }

    /// Does `self` dominate `other` pointwise (`self(g) >= other(g)` for
    /// all `g`)? Decided by the standard criterion for inductive
    /// valuations over the same (unshifted) base: domination holds iff
    /// `self(phi) >= lambda` for every augmentation `(phi, lambda)` of
    /// `other`.
    pub fn dominates(&self, other: &Self) -> bool {
        if self.base != other.base || self.lambda0 != other.lambda0 {
            return false;
        }
        other
            .augmentations
            .iter()
            .all(|aug| self.value(&aug.phi) >= aug.lambda)
    }

    /// `tau_level = [Gamma_level : Gamma_{level-1}]`, the relative
    /// ramification of the last augmentation of the level-`level` prefix.
    fn tau_at(&self, level: usize) -> Result<usize> {
        let mut gen = rat_i64(1);
        for j in 1..level {
            if let QVal::Finite(l) = &self.augmentations[j - 1].lambda {
                gen = rat_gcd(&gen, l);
            }
        }
        let lambda = self.augmentations[level - 1].lambda.finite().ok_or_else(|| {
            MathError::InvalidArgument(
                "maclane: tau of an infinite augmentation".to_string(),
            )
        })?;
        let ratio = gen.clone() / rat_gcd(&gen, lambda);
        if !ratio.is_integer() {
            return Err(MathError::NumericalError(
                "maclane: internal error: non-integral value-group index".to_string(),
            ));
        }
        Ok(int_to_i64(ratio.numerator())? as usize)
    }

    /// The MacLane augmentation `[self, v(phi) = lambda]`.
    ///
    /// Verifies that `phi` is a key polynomial (Err otherwise: the min
    /// formula would not be multiplicative) and that
    /// `lambda > self.value(phi)` for finite `lambda`. A same-degree key
    /// collapses the chain: `[.., v(phi_k) = lambda_k, v(phi) = lambda]`
    /// is stored as `[.., v(phi) = lambda]` (the standard equality of
    /// valuations for same-degree keys), with `phi` re-verified as a key of
    /// the truncated valuation.
    pub fn augment(
        &self,
        phi: UnivariatePolynomial<Rational>,
        lambda: QVal,
    ) -> Result<Self> {
        if self.is_final() {
            return Err(MathError::InvalidArgument(
                "maclane: cannot augment a final (infinite) valuation".to_string(),
            ));
        }
        if !self.lambda0.is_zero() {
            return Err(MathError::NotSupported(
                "maclane: augmenting shifted Gauss valuations not implemented (use the augmentation [v0, v(x) = lambda] instead)"
                    .to_string(),
            ));
        }
        let rd = match self.is_key(&phi)? {
            KeyCheck::Key { residual_degree } => residual_degree,
            KeyCheck::NotKey(reason) => {
                return Err(MathError::InvalidArgument(format!(
                    "maclane: phi is not a key polynomial: {}",
                    reason
                )))
            }
        };
        if let QVal::Finite(_) = &lambda {
            if lambda <= self.value(&phi) {
                return Err(MathError::InvalidArgument(format!(
                    "maclane: lambda = {} must exceed v(phi) = {}",
                    lambda,
                    self.value(&phi)
                )));
            }
        }
        let same_degree = self
            .last_key()
            .map(|k| k.degree() == phi.degree())
            .unwrap_or(false);
        if same_degree {
            let trunc = self.truncated();
            let rd_t = match trunc.is_key(&phi)? {
                KeyCheck::Key { residual_degree } => residual_degree,
                KeyCheck::NotKey(reason) => {
                    return Err(MathError::NumericalError(format!(
                        "maclane: internal error: same-degree key of the augmented valuation is not a key of the truncation: {}",
                        reason
                    )))
                }
            };
            let mut out = trunc;
            out.augmentations.push(Augmentation {
                phi,
                lambda,
                residual_degree: rd_t,
            });
            Ok(out)
        } else {
            let mut out = self.clone();
            out.augmentations.push(Augmentation {
                phi,
                lambda,
                residual_degree: rd,
            });
            Ok(out)
        }
    }

    /// Children of this valuation for the key `phi` and target `f`: an
    /// infinite augmentation if `phi | f` exactly, plus one augmentation
    /// `[self, v(phi) = lambda']` for each side of the phi-Newton polygon of
    /// `f` (points `(i, w(f_i))`) with `lambda' = -slope > w(phi)`.
    fn children_for_key(
        &self,
        f: &UnivariatePolynomial<Rational>,
        phi: &UnivariatePolynomial<Rational>,
    ) -> Result<Vec<Self>> {
        let exp = phi_adic_expansion(f, phi)?;
        let cur = self.value(phi);
        let mut kids = Vec::new();
        if exp.first().map(|c| c.is_zero()).unwrap_or(false) {
            kids.push(self.augment(phi.clone(), QVal::Infinity)?);
        }
        let mut pts: Vec<(usize, Rational)> = Vec::new();
        for (i, c) in exp.iter().enumerate() {
            if !c.is_zero() {
                match self.value(c) {
                    QVal::Finite(u) => pts.push((i, u)),
                    QVal::Infinity => unreachable!("nonzero coefficient has finite value"),
                }
            }
        }
        for ((i1, u1), (i2, u2)) in lower_hull_sides(&pts) {
            let slope =
                (u2 - u1) / (rat_i64(i2 as i64) - rat_i64(i1 as i64));
            let lambda = -slope;
            if QVal::Finite(lambda.clone()) > cur {
                kids.push(self.augment(phi.clone(), QVal::Finite(lambda))?);
            }
        }
        // deterministic order: by lambda of the new step, finite ascending,
        // infinity last
        kids.sort_by(|a, b| {
            let la = a.augmentations.last().expect("augmented").lambda.clone();
            let lb = b.augmentations.last().expect("augmented").lambda.clone();
            la.cmp(&lb)
        });
        Ok(kids)
    }

    /// One step of the MacLane algorithm toward the monic squarefree
    /// p-integral target `f`: returns the valuations replacing `self` in the
    /// leaf set (each strictly closer to the extensions of `v_p` determined
    /// by the irreducible factors of `f` over `Q_p`).
    ///
    /// Implemented at EVERY augmentation level (residue-field towers,
    /// certified residual factorization over `GF(p^d)`, verified key
    /// lifts; same-degree refinements collapse, so degree-preserving
    /// chains can be iterated indefinitely). `Err(NotSupported)` only for
    /// shifted Gauss valuations.
    pub fn mac_lane_step(&self, f: &UnivariatePolynomial<Rational>) -> Result<Vec<Self>> {
        if self.is_final() {
            return Err(MathError::InvalidArgument(
                "maclane: cannot step a final valuation".to_string(),
            ));
        }
        if !self.lambda0.is_zero() {
            return Err(MathError::NotSupported(
                "maclane: mac_lane_step over shifted Gauss valuations not implemented".to_string(),
            ));
        }
        match self.augmentations.len() {
            0 => {
                let fbar = self.reduce_poly_mod_p(f)?;
                let factors = factor_fp_certified(&fbar, self.p())?;
                let mut kids = Vec::new();
                for (psi, _mult) in &factors {
                    let phi = UnivariatePolynomial::new(
                        psi.iter().map(|&c| rat_i64(c)).collect::<Vec<_>>(),
                    );
                    kids.extend(self.children_for_key(f, &phi)?);
                }
                if kids.is_empty() {
                    return Err(MathError::NumericalError(
                        "maclane: internal error: Gauss step made no progress".to_string(),
                    ));
                }
                Ok(kids)
            }
            level => {
                let phi_last = self.augmentations[level - 1].phi.clone();
                // (a) roots closer to the current key: steeper polygon sides
                // (and an exact-division infinite leaf); same-degree
                // augmentations collapse inside augment().
                let mut kids = self.children_for_key(f, &phi_last)?;
                // (b) residual factors away from y: new or refined keys,
                // lifted from the certified factorization of the residual
                // polynomial of f over the residue tower kappa_level.
                let data = self.residue_data(level)?;
                let (r, _i0) = self.residual_polynomial_general(level, f, &data)?;
                if data.tower.poly_degree(&r) >= 1 {
                    for (psi, _mult) in data.tower.factor_certified(&r)? {
                        let phi_new = self.lift_residual_to_key(level, &psi, &data)?;
                        kids.extend(self.children_for_key(f, &phi_new)?);
                    }
                }
                if kids.is_empty() {
                    return Err(MathError::NumericalError(
                        "maclane: internal error: step made no progress".to_string(),
                    ));
                }
                Ok(kids)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Certified GF(p) factorization.
//
// `rustmath_polynomials::fp_factor::factor` is WRONG on inseparable inputs
// (zero derivative mod p): `factor(x^2 mod 2)` returns `[x, x]` (duplicates)
// and `factor(x^4 + x^2 mod 2)` returns `[x+1, x, x^2+x]` with the reducible
// `x^2+x` listed as a factor, because `squarefree_factor` does not handle
// `f' = 0` (a pre-existing bug in the read-only rustmath-polynomials crate).
// The helpers below implement the p-th-root reduction `g(x^p) = g(x)^p`
// correctly, certify irreducibility with Rabin's test, and certify the whole
// factorization by reconstruction, so a wrong answer is an `Err`, never a
// silent lie.
// ---------------------------------------------------------------------------

/// Rabin irreducibility test for a monic polynomial over GF(p):
/// `q` of degree `d >= 1` is irreducible iff `x^{p^d} = x (mod q)` and
/// `gcd(x^{p^{d/l}} - x, q) = 1` for every prime `l | d`.
fn is_irreducible_fp(q: &[i64], p: i64) -> Result<bool> {
    let d = fp_factor::degree(q);
    if d < 1 {
        return Ok(false);
    }
    if d == 1 {
        return Ok(true);
    }
    let x = vec![0i64, 1];
    let pd = Integer::from(p).pow(d as u32);
    let xp = fp_factor::pow_mod(&x, &pd, q, p);
    if !fp_factor::is_zero(&fp_factor::sub(&xp, &x, p)) {
        return Ok(false);
    }
    let mut m = d;
    let mut ell = 2i64;
    let mut prime_divs = Vec::new();
    while ell * ell <= m {
        if m % ell == 0 {
            prime_divs.push(ell);
            while m % ell == 0 {
                m /= ell;
            }
        }
        ell += 1;
    }
    if m > 1 {
        prime_divs.push(m);
    }
    for l in prime_divs {
        let e = Integer::from(p).pow((d / l) as u32);
        let xe = fp_factor::pow_mod(&x, &e, q, p);
        let diff = fp_factor::sub(&xe, &x, p);
        let g = fp_factor::gcd(&diff, q, p);
        if fp_factor::degree(&g) != 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The p-th root of an inseparable polynomial: `f` with `f' = 0 (mod p)` is
/// `g(x^p) = g(x)^p` over GF(p); returns `g`.
fn fp_pth_root(f: &[i64], p: i64) -> Result<Vec<i64>> {
    let d = fp_factor::degree(f);
    if d <= 0 || d % p != 0 {
        return Err(MathError::NumericalError(
            "maclane: internal error: p-th root of a non-p-power".to_string(),
        ));
    }
    let mut g = vec![0i64; (d / p) as usize + 1];
    for (i, &c) in f.iter().enumerate() {
        if c != 0 {
            if i % (p as usize) != 0 {
                return Err(MathError::NumericalError(
                    "maclane: internal error: inseparable polynomial is not in x^p".to_string(),
                ));
            }
            g[i / (p as usize)] = c;
        }
    }
    Ok(g)
}

/// Certified factorization over GF(p): the distinct monic irreducible
/// factors of `f` with multiplicities. Every factor is verified irreducible
/// (Rabin) and the product `lc * prod q_i^{m_i}` is verified equal to `f`;
/// any mismatch is an `Err`.
fn factor_fp_certified(f: &[i64], p: i64) -> Result<Vec<(Vec<i64>, usize)>> {
    let f = fp_factor::trim(f);
    if fp_factor::degree(&f) < 1 {
        return Err(MathError::InvalidArgument(
            "factor_fp_certified: constant polynomial".to_string(),
        ));
    }
    let fm = fp_factor::make_monic(&f, p);
    let mut result: Vec<(Vec<i64>, usize)> = Vec::new();
    // (work, multiplicity-multiplier) pieces still to be factored
    let mut pending: Vec<(Vec<i64>, usize)> = vec![(fm.clone(), 1)];
    let mut guard = 0;
    while let Some((work, mult0)) = pending.pop() {
        guard += 1;
        if guard > 200 {
            return Err(MathError::NumericalError(
                "factor_fp_certified: internal error: no progress".to_string(),
            ));
        }
        if fp_factor::degree(&work) < 1 {
            continue;
        }
        let deriv = fp_factor::derivative_of(&work, p);
        if fp_factor::is_zero(&deriv) {
            // work = g(x)^p
            let g = fp_pth_root(&work, p)?;
            pending.push((g, mult0 * p as usize));
            continue;
        }
        let gcd = fp_factor::gcd(&work, &deriv, p);
        let (sf, rem) = fp_factor::div_mod(&work, &gcd, p);
        if !fp_factor::is_zero(&rem) {
            return Err(MathError::NumericalError(
                "factor_fp_certified: internal error: gcd does not divide".to_string(),
            ));
        }
        // sf is squarefree (and separable); factor it with DDF/EDF and
        // certify each piece.
        let mut cofactor = work.clone();
        for (d, gd) in fp_factor::distinct_degree_factor(&sf, p) {
            for piece in fp_factor::equal_degree_factor(&gd, d, p) {
                let q = fp_factor::make_monic(&piece, p);
                if fp_factor::degree(&q) != d || !is_irreducible_fp(&q, p)? {
                    return Err(MathError::NumericalError(
                        "factor_fp_certified: internal error: EDF piece not irreducible"
                            .to_string(),
                    ));
                }
                // multiplicity of q in the current work piece
                let mut mult = 0usize;
                loop {
                    let (quo, r) = fp_factor::div_mod(&cofactor, &q, p);
                    if fp_factor::is_zero(&r) {
                        cofactor = quo;
                        mult += 1;
                    } else {
                        break;
                    }
                }
                if mult == 0 {
                    return Err(MathError::NumericalError(
                        "factor_fp_certified: internal error: factor does not divide".to_string(),
                    ));
                }
                result.push((q, mult * mult0));
            }
        }
        if fp_factor::degree(&cofactor) >= 1 {
            // remaining factors all have multiplicity divisible by p
            pending.push((cofactor, mult0));
        }
    }
    // merge duplicates (a factor can reappear from different pieces)
    let mut merged: Vec<(Vec<i64>, usize)> = Vec::new();
    for (q, m) in result {
        if let Some(entry) = merged.iter_mut().find(|(q2, _)| *q2 == q) {
            entry.1 += m;
        } else {
            merged.push((q, m));
        }
    }
    // certify by reconstruction: lc * prod q^m == f
    let lc = *f.last().expect("nonzero");
    let mut acc = vec![lc];
    for (q, m) in &merged {
        for _ in 0..*m {
            acc = fp_factor::mul(&acc, q, p);
        }
    }
    if fp_factor::trim(&acc) != f {
        return Err(MathError::NumericalError(
            "factor_fp_certified: reconstruction check failed".to_string(),
        ));
    }
    Ok(merged)
}

/// Sides of the lower convex hull of `points` (x strictly increasing).
fn lower_hull_sides(points: &[(usize, Rational)]) -> Vec<((usize, Rational), (usize, Rational))> {
    let mut stack: Vec<(usize, Rational)> = Vec::new();
    for pt in points {
        while stack.len() >= 2 {
            let a = &stack[stack.len() - 2];
            let b = &stack[stack.len() - 1];
            // pop b if slope(a,b) >= slope(b,pt):
            // (b.u - a.u) * (pt.i - b.i) >= (pt.u - b.u) * (b.i - a.i)
            let lhs = (b.1.clone() - a.1.clone())
                * (rat_i64(pt.0 as i64) - rat_i64(b.0 as i64));
            let rhs = (pt.1.clone() - b.1.clone())
                * (rat_i64(b.0 as i64) - rat_i64(a.0 as i64));
            if lhs >= rhs {
                stack.pop();
            } else {
                break;
            }
        }
        stack.push(pt.clone());
    }
    stack.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect()
}

/// The MacLane approximants of the extensions of `v_p` to `Q[x]/(f)`:
/// inductive valuations `w_1, ..., w_r` in bijection with the irreducible
/// factors of `f` over `Q_p`, iterated until
/// `sum_i E(w_i) * F(w_i) = deg f`. For each leaf, `E` is the ramification
/// index and `F` the residue degree of the corresponding extension of `Q_p`.
///
/// Requirements (honest `Err` otherwise): `p` prime (`< 2^31`), `f` monic of
/// degree >= 1, p-integral coefficients, squarefree over `Q`.
pub fn mac_lane_approximants(
    f: &UnivariatePolynomial<Rational>,
    p: i64,
) -> Result<Vec<PAdicInductiveValuation>> {
    let base = PAdicBaseValuation::new(p)?;
    let n = match f.degree() {
        Some(n) if n >= 1 => n,
        _ => {
            return Err(MathError::InvalidArgument(
                "mac_lane_approximants: f must have degree >= 1".to_string(),
            ))
        }
    };
    if !f.is_monic() {
        return Err(MathError::NotSupported(
            "mac_lane_approximants: f must be monic".to_string(),
        ));
    }
    let p_int = Integer::from(p);
    for c in f.coefficients() {
        if !c.is_zero() && c.valuation(&p_int) < 0 {
            return Err(MathError::NotSupported(
                "mac_lane_approximants: f must be p-integral".to_string(),
            ));
        }
    }
    let deriv = f.derivative();
    let g = f.gcd(&deriv);
    if g.degree() != Some(0) {
        return Err(MathError::InvalidArgument(
            "mac_lane_approximants: f must be squarefree (the tree would not terminate)"
                .to_string(),
        ));
    }
    let mut leaves: Vec<PAdicInductiveValuation> = vec![InductiveValuation::gauss(base)];
    for round in 0..64 {
        if round > 0 {
            let total: u64 = leaves
                .iter()
                .map(|w| w.ramification_index() * w.residue_degree())
                .sum();
            debug_assert!(total <= n as u64);
            if total == n as u64 {
                return Ok(leaves);
            }
        }
        let mut next = Vec::new();
        for w in &leaves {
            if w.is_final() {
                next.push(w.clone());
            } else {
                next.extend(w.mac_lane_step(f)?);
            }
        }
        leaves = next;
    }
    Err(MathError::NumericalError(
        "mac_lane_approximants: did not terminate within 64 rounds".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Tests. Every expected value below was verified independently BEFORE being
// asserted, in scratchpad/verify_maclane.py (sympy + first-principles 2-adic
// computations: Eisenstein, mod-8 square criterion, GF(p) factorization,
// Hensel lifting, Newton polygons). 54/54 checks passed.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_field::typed::element::RationalFunction;

    fn qpoly(coeffs: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.iter().map(|&n| rat_i64(n)).collect())
    }

    fn v2_gauss() -> PAdicInductiveValuation {
        InductiveValuation::gauss(PAdicBaseValuation::new(2).unwrap())
    }

    /// w1 = [ Gauss(v_2), v(x+1) = 1/2 ]
    fn w1() -> PAdicInductiveValuation {
        v2_gauss().augment(qpoly(&[1, 1]), QVal::from_frac(1, 2)).unwrap()
    }

    /// w2 = [ Gauss(v_2), v(x) = 1/2, v(x^2 - 2) = 3/2 ]
    fn w2() -> PAdicInductiveValuation {
        v2_gauss()
            .augment(qpoly(&[0, 1]), QVal::from_frac(1, 2))
            .unwrap()
            .augment(qpoly(&[-2, 0, 1]), QVal::from_frac(3, 2))
            .unwrap()
    }

    /// Simple LCG for the random batteries (deterministic).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn rat(&mut self) -> Rational {
            let num = (self.next() % 41) as i64 - 20;
            let den = [1, 1, 2, 4, 3][(self.next() % 5) as usize];
            rat_frac(num, den)
        }
        fn poly(&mut self, maxdeg: usize) -> UnivariatePolynomial<Rational> {
            let d = (self.next() as usize) % (maxdeg + 1);
            let mut coeffs: Vec<Rational> = (0..=d).map(|_| self.rat()).collect();
            if coeffs.iter().all(|c| c.is_zero()) {
                coeffs[d] = rat_i64(1);
            }
            UnivariatePolynomial::new(coeffs)
        }
    }

    #[test]
    fn test_qval_ordering_and_ops() {
        let half = QVal::from_frac(1, 2);
        let one = QVal::from_int(1);
        let inf = QVal::Infinity;
        assert!(half < one);
        assert!(one < inf);
        assert_eq!(half.add(&one), QVal::from_frac(3, 2));
        assert_eq!(half.add(&inf), QVal::Infinity);
        assert_eq!(QVal::min_of(&half, &one), half);
        assert_eq!(QVal::min_of(&inf, &one), one);
        assert_eq!(format!("{}", QVal::from_frac(1, 2)), "1/2");
        assert_eq!(format!("{}", QVal::Infinity), "+Infinity");
    }

    #[test]
    fn test_rat_gcd() {
        // verified: gcd(1, 1/2) = 1/2; gcd(1/2, 3/2) = 1/2; gcd(1, 2) = 1
        assert_eq!(rat_gcd(&rat_i64(1), &rat_frac(1, 2)), rat_frac(1, 2));
        assert_eq!(rat_gcd(&rat_frac(1, 2), &rat_frac(3, 2)), rat_frac(1, 2));
        assert_eq!(rat_gcd(&rat_i64(1), &rat_i64(2)), rat_i64(1));
        assert_eq!(rat_gcd(&rat_i64(0), &rat_frac(-1, 3)), rat_frac(1, 3));
    }

    #[test]
    fn test_padic_base_valuation_gates() {
        assert!(PAdicBaseValuation::new(2).is_ok());
        assert!(PAdicBaseValuation::new(1).is_err());
        assert!(PAdicBaseValuation::new(4).is_err());
        assert!(PAdicBaseValuation::new(-7).is_err());
    }

    #[test]
    fn test_gauss_valuation_values_v2() {
        let v0 = v2_gauss();
        // v(12) = 2, v(4) = 2, v(8) = 3 -> min = 2
        assert_eq!(v0.value(&qpoly(&[8, 4, 0, 12])), QVal::from_int(2));
        assert_eq!(v0.value(&qpoly(&[0])), QVal::Infinity);
        assert_eq!(v0.value(&qpoly(&[0, 1])), QVal::from_int(0));
        assert_eq!(v0.value(&qpoly(&[7])), QVal::from_int(0));
        // constant 3/8 has v_2 = -3
        let c = UnivariatePolynomial::new(vec![rat_frac(3, 8)]);
        assert_eq!(v0.value(&c), QVal::from_int(-3));
        assert_eq!(v0.ramification_index(), 1);
        assert_eq!(v0.residue_degree(), 1);
        assert!(v0.is_gauss() && !v0.is_final());
    }

    #[test]
    fn test_gauss_agrees_with_padics() {
        // The Gauss valuation on constants is the p-adic valuation computed
        // by the in-crate padics module.
        use crate::padics::PadicRational;
        let v0 = v2_gauss();
        for (num, den) in [(3i64, 8i64), (12, 1), (5, 6), (-40, 7)] {
            let q = rat_frac(num, den);
            let via_padics =
                PadicRational::from_rational(q.clone(), Integer::from(2), 20).unwrap().valuation();
            let via_gauss = v0.value(&UnivariatePolynomial::new(vec![q]));
            assert_eq!(via_gauss, QVal::from_int(via_padics as i64));
        }
    }

    #[test]
    fn test_shifted_gauss_values() {
        // verified in python (B2)
        let sh = InductiveValuation::gauss_shifted(
            PAdicBaseValuation::new(2).unwrap(),
            rat_frac(1, 2),
        );
        assert_eq!(sh.value(&qpoly(&[2, 0, 1])), QVal::from_int(1)); // x^2+2
        let sh53 = InductiveValuation::gauss_shifted(
            PAdicBaseValuation::new(5).unwrap(),
            rat_frac(-2, 3),
        );
        assert_eq!(sh53.value(&qpoly(&[1, 5, 0, 25])), QVal::from_int(0));
        assert_eq!(sh53.value(&qpoly(&[0, 0, 0, 5])), QVal::from_int(-1));
        // value group gains lambda0: gcd(1, 1/2) = 1/2
        let sh2 = InductiveValuation::gauss_shifted(
            PAdicBaseValuation::new(2).unwrap(),
            rat_frac(1, 2),
        );
        assert_eq!(sh2.value_group_generator(), rat_frac(1, 2));
        assert_eq!(sh2.ramification_index(), 2);
    }

    #[test]
    fn test_gauss_multiplicativity_battery() {
        // THE law: v(fg) = v(f) + v(g), 25 random pairs (python B4)
        let v0 = v2_gauss();
        let mut rng = Lcg(0x5EED_0001);
        for _ in 0..25 {
            let f = rng.poly(4);
            let g = rng.poly(4);
            let lhs = v0.value(&(f.clone() * g.clone()));
            let rhs = v0.value(&f).add(&v0.value(&g));
            assert_eq!(lhs, rhs, "v0(fg) != v0(f)+v0(g) for f={}, g={}", f, g);
            // ultrametric: v(f+g) >= min(v f, v g)
            let s = v0.value(&(f.clone() + g.clone()));
            assert!(s >= QVal::min_of(&v0.value(&f), &v0.value(&g)));
        }
    }

    #[test]
    fn test_shifted_gauss_multiplicativity_battery() {
        let sh = InductiveValuation::gauss_shifted(
            PAdicBaseValuation::new(2).unwrap(),
            rat_frac(1, 2),
        );
        let sh53 = InductiveValuation::gauss_shifted(
            PAdicBaseValuation::new(5).unwrap(),
            rat_frac(-2, 3),
        );
        let mut rng = Lcg(0x5EED_0002);
        for _ in 0..15 {
            let f = rng.poly(4);
            let g = rng.poly(4);
            for w in [&sh, &sh53] {
                let lhs = w.value(&(f.clone() * g.clone()));
                let rhs = w.value(&f).add(&w.value(&g));
                assert_eq!(lhs, rhs, "shifted Gauss not multiplicative");
            }
        }
    }

    #[test]
    fn test_gauss_over_function_field_place() {
        // Gauss valuation on Q(t)[X] over the place valuation v_t of the
        // stage-1 typed function field.
        let place = Place::<Rational>::finite_linear(Rational::zero());
        let base = PlaceBaseValuation::new(place);
        let v0 = InductiveValuation::gauss(base);
        let t = RationalFunction::<Rational>::gen();
        let t_inv = RationalFunction::constant(rat_i64(1)) / t.clone();
        // f = (1/t) X + t: min(v_t(1/t), v_t(t)) = min(-1, 1) = -1
        let f = UnivariatePolynomial::new(vec![t.clone(), t_inv.clone()]);
        assert_eq!(v0.value(&f), QVal::from_int(-1));
        assert_eq!(v0.value(&UnivariatePolynomial::new(vec![t.clone()])), QVal::from_int(1));
        // multiplicativity battery with coefficients c * t^k
        let mut rng = Lcg(0x5EED_0003);
        let mut rand_ff_poly = |rng: &mut Lcg| {
            let d = (rng.next() as usize) % 3;
            let coeffs: Vec<RationalFunction<Rational>> = (0..=d)
                .map(|_| {
                    let c = rng.rat();
                    if c.is_zero() {
                        return RationalFunction::constant(rat_i64(1));
                    }
                    let k = (rng.next() % 5) as i64 - 2;
                    let mut e = RationalFunction::constant(c);
                    for _ in 0..k.unsigned_abs() {
                        if k > 0 {
                            e = e * t.clone();
                        } else {
                            e = e * t_inv.clone();
                        }
                    }
                    e
                })
                .collect();
            UnivariatePolynomial::new(coeffs)
        };
        for _ in 0..10 {
            let f = rand_ff_poly(&mut rng);
            let g = rand_ff_poly(&mut rng);
            let lhs = v0.value(&(f.clone() * g.clone()));
            let rhs = v0.value(&f).add(&v0.value(&g));
            assert_eq!(lhs, rhs, "Gauss over place base not multiplicative");
        }
    }

    #[test]
    fn test_augment_rejects_non_key() {
        let v0 = v2_gauss();
        // x^2+1 reduces to (x+1)^2 mod 2: not equivalence-irreducible
        assert!(matches!(v0.is_key(&qpoly(&[1, 0, 1])).unwrap(), KeyCheck::NotKey(_)));
        assert!(v0.augment(qpoly(&[1, 0, 1]), QVal::from_int(1)).is_err());
        // non-monic
        assert!(matches!(v0.is_key(&qpoly(&[1, 2])).unwrap(), KeyCheck::NotKey(_)));
        // non-integral: x + 1/2
        let phi = UnivariatePolynomial::new(vec![rat_frac(1, 2), rat_i64(1)]);
        assert!(matches!(v0.is_key(&phi).unwrap(), KeyCheck::NotKey(_)));
        // lambda not exceeding v(phi): v0(x) = 0
        assert!(v0.augment(qpoly(&[0, 1]), QVal::from_int(0)).is_err());
        // constant
        assert!(matches!(v0.is_key(&qpoly(&[1])).unwrap(), KeyCheck::NotKey(_)));
    }

    #[test]
    fn test_should_fail_non_key_min_formula() {
        // THE counterexample (python B5, hand-worked): with the NON-key
        // phi = x^2+1 over Gauss(v_2), the min formula is not multiplicative:
        //   "w"((x+1)^2) = 1 but "w"(x+1) + "w"(x+1) = 0.
        // Constructed by bypassing the key check (test-only struct literal).
        let bad = InductiveValuation {
            base: PAdicBaseValuation::new(2).unwrap(),
            lambda0: Rational::zero(),
            augmentations: vec![Augmentation {
                phi: qpoly(&[1, 0, 1]),
                lambda: QVal::from_int(1),
                residual_degree: 1,
            }],
        };
        let f = qpoly(&[1, 1]); // x + 1
        let prod = f.clone() * f.clone(); // x^2 + 2x + 1
        assert_eq!(bad.value(&f), QVal::from_int(0));
        assert_eq!(bad.value(&prod), QVal::from_int(1));
        // multiplicativity FAILS, exactly because phi is not a key:
        assert_ne!(bad.value(&prod), bad.value(&f).add(&bad.value(&f)));
    }

    #[test]
    fn test_augmented_w1_values() {
        // python B1: w1 = [Gauss(v_2), v(x+1) = 1/2]
        let w = w1();
        assert_eq!(w.value(&qpoly(&[1, 0, 1])), QVal::from_int(1)); // x^2+1
        assert_eq!(w.value(&qpoly(&[1, 1])), QVal::from_frac(1, 2)); // x+1
        assert_eq!(w.value(&qpoly(&[0, 1])), QVal::from_int(0)); // x
        assert_eq!(w.value(&qpoly(&[2])), QVal::from_int(1)); // 2
        let cube = qpoly(&[1, 0, 1]) * qpoly(&[1, 0, 1]) * qpoly(&[1, 0, 1]);
        assert_eq!(w.value(&cube), QVal::from_int(3)); // (x^2+1)^3
        assert_eq!(w.ramification_index(), 2);
        assert_eq!(w.residue_degree(), 1);
        assert_eq!(w.value_group_generator(), rat_frac(1, 2));
    }

    #[test]
    fn test_augmented_multiplicativity_battery() {
        // w(fg) = w(f) + w(g) for the augmented valuation (python B4)
        let w = w1();
        let mut rng = Lcg(0x5EED_0004);
        for _ in 0..25 {
            let f = rng.poly(4);
            let g = rng.poly(4);
            let lhs = w.value(&(f.clone() * g.clone()));
            let rhs = w.value(&f).add(&w.value(&g));
            assert_eq!(lhs, rhs, "w1(fg) != w1(f)+w1(g) for f={}, g={}", f, g);
        }
    }

    #[test]
    fn test_w_dominates_v_battery() {
        // w(f) >= v(f) for all f, with equality when the phi-adic expansion
        // has degree 0 (python B6)
        let v0 = v2_gauss();
        let w = w1();
        let w1s = v0.augment(qpoly(&[0, 1]), QVal::from_frac(1, 2)).unwrap();
        let w2 = w2();
        let mut rng = Lcg(0x5EED_0005);
        for _ in 0..40 {
            let f = rng.poly(4);
            assert!(w.value(&f) >= v0.value(&f), "w1 >= v0 fails for {}", f);
            assert!(w2.value(&f) >= w1s.value(&f), "w2 >= w1' fails for {}", f);
            if f.degree().map(|d| d < 2).unwrap_or(true) {
                // expansion degree 0 at the last level of w2
                assert_eq!(w2.value(&f), w1s.value(&f), "equality (deg 0) for {}", f);
            }
        }
        // constants: expansion degree 0 for w1
        for c in [2i64, 3, 12, -8] {
            assert_eq!(w.value(&qpoly(&[c])), v0.value(&qpoly(&[c])));
        }
        // w1(x^2+1) = 1 > v0(x^2+1) = 0: phi equivalence-divides
        assert!(w.value(&qpoly(&[1, 0, 1])) > v0.value(&qpoly(&[1, 0, 1])));
    }

    #[test]
    fn test_two_step_chain_values() {
        // python B3: w2 = [Gauss(v_2), v(x)=1/2, v(x^2-2)=3/2]
        let w = w2();
        assert_eq!(w.level(), 2);
        let spots: [(&[i64], (i64, i64)); 8] = [
            (&[-2, 0, 1], (3, 2)),      // x^2-2
            (&[0, 1], (1, 2)),          // x
            (&[0, 0, 0, 1], (3, 2)),    // x^3
            (&[4, 0, -4, 0, 1], (3, 1)), // (x^2-2)^2 = x^4-4x^2+4
            (&[4, 2], (3, 2)),          // 2x+4
            (&[2, 0, 1], (3, 2)),       // x^2+2 = (x^2-2)+4
            (&[0, 0, 1], (1, 1)),       // x^2
            (&[-2, 8, 1], (3, 2)),      // x^2+8x-2
        ];
        for (coeffs, (n, d)) in spots {
            assert_eq!(
                w.value(&qpoly(coeffs)),
                QVal::from_frac(n, d),
                "w2 of {:?}",
                coeffs
            );
        }
        assert_eq!(w.ramification_index(), 2);
        assert_eq!(w.residue_degree(), 1);
        assert_eq!(w.value_group_generator(), rat_frac(1, 2));
    }

    #[test]
    fn test_two_step_multiplicativity_battery() {
        let w = w2();
        let mut rng = Lcg(0x5EED_0006);
        for _ in 0..25 {
            let f = rng.poly(4);
            let g = rng.poly(4);
            let lhs = w.value(&(f.clone() * g.clone()));
            let rhs = w.value(&f).add(&w.value(&g));
            assert_eq!(lhs, rhs, "w2(fg) != w2(f)+w2(g) for f={}, g={}", f, g);
        }
    }

    #[test]
    fn test_value_group_bookkeeping() {
        // Gamma_w = Gamma_v + Z*lambda, checked step by step
        let v0 = v2_gauss();
        assert_eq!(v0.value_group_generator(), rat_i64(1));
        let w1s = v0.augment(qpoly(&[0, 1]), QVal::from_frac(1, 2)).unwrap();
        assert_eq!(
            w1s.value_group_generator(),
            rat_gcd(&v0.value_group_generator(), &rat_frac(1, 2))
        );
        assert_eq!(w1s.ramification_index(), 2);
        let w2 = w1s
            .augment(qpoly(&[-2, 0, 1]), QVal::from_frac(3, 2))
            .unwrap();
        assert_eq!(
            w2.value_group_generator(),
            rat_gcd(&w1s.value_group_generator(), &rat_frac(3, 2))
        );
        assert_eq!(w2.ramification_index(), 2);
        // integer lambda keeps Gamma = Z
        let wz = v0.augment(qpoly(&[-2, 1]), QVal::from_int(2)).unwrap();
        assert_eq!(wz.value_group_generator(), rat_i64(1));
        assert_eq!(wz.ramification_index(), 1);
        assert_eq!(wz.residue_degree(), 1);
    }

    #[test]
    fn test_is_key_level1() {
        // over w1' = [Gauss(v_2), v(x)=1/2] (python B7a/B7b)
        let w = v2_gauss().augment(qpoly(&[0, 1]), QVal::from_frac(1, 2)).unwrap();
        // x^2-2: R = y+1 irreducible => key with residual degree 1
        assert_eq!(
            w.is_key(&qpoly(&[-2, 0, 1])).unwrap(),
            KeyCheck::Key { residual_degree: 1 }
        );
        // x^2+2: R = y+1 => key
        assert_eq!(
            w.is_key(&qpoly(&[2, 0, 1])).unwrap(),
            KeyCheck::Key { residual_degree: 1 }
        );
        // x^2 = phi^2: residual polynomial y => NOT a key
        assert!(matches!(w.is_key(&qpoly(&[0, 0, 1])).unwrap(), KeyCheck::NotKey(_)));
        // x+1: same degree but w(x+1-x) = 0 <= 1/2 => not equivalent, not key
        assert!(matches!(w.is_key(&qpoly(&[1, 1])).unwrap(), KeyCheck::NotKey(_)));
        // x+2: w(2) = 1 > 1/2 => equivalent key
        assert_eq!(
            w.is_key(&qpoly(&[2, 1])).unwrap(),
            KeyCheck::Key { residual_degree: 1 }
        );
        // x^3: degree not a multiple of e*d = 2
        assert!(matches!(w.is_key(&qpoly(&[0, 0, 0, 1])).unwrap(), KeyCheck::NotKey(_)));
    }

    #[test]
    fn test_is_key_level1_residue_degree_two() {
        // w = [Gauss(v_2), v(x^2+x+1) = 1]: residue field GF(4) (d = 2)
        let w = v2_gauss().augment(qpoly(&[1, 1, 1]), QVal::from_int(1)).unwrap();
        assert_eq!(w.residue_degree(), 2);
        // x^2+x+3 = phi + 2: R = 1 + y (degree 1, trivially irreducible)
        assert_eq!(
            w.is_key(&qpoly(&[3, 1, 1])).unwrap(),
            KeyCheck::Key { residual_degree: 1 }
        );
        // x^2+x+5 = phi + 4: w(4) = 2 > 1 => equivalent key (rd of phi = 2)
        assert_eq!(
            w.is_key(&qpoly(&[5, 1, 1])).unwrap(),
            KeyCheck::Key { residual_degree: 2 }
        );
        // x^2+x+2 = phi + 1: w(1) = 0 < 1 => w(phi') = 0 != lambda: not minimal
        assert!(matches!(w.is_key(&qpoly(&[2, 1, 1])).unwrap(), KeyCheck::NotKey(_)));
    }

    #[test]
    fn test_phi_adic_expansion() {
        // x^2+3x+2 = (x+1)^2 + (x+1) + 0
        let f = qpoly(&[2, 3, 1]);
        let phi = qpoly(&[1, 1]);
        let exp = phi_adic_expansion(&f, &phi).unwrap();
        assert_eq!(exp.len(), 3);
        assert!(exp[0].is_zero());
        assert_eq!(exp[1], qpoly(&[1]));
        assert_eq!(exp[2], qpoly(&[1]));
        // reconstruct
        let mut acc = UnivariatePolynomial::zero();
        for c in exp.iter().rev() {
            acc = acc * phi.clone() + c.clone();
        }
        assert_eq!(acc, f);
        // zero polynomial -> empty expansion
        assert!(phi_adic_expansion(&qpoly(&[0]), &phi).unwrap().is_empty());
        // constant phi is an error
        assert!(phi_adic_expansion(&f, &qpoly(&[1])).is_err());
    }

    #[test]
    fn test_infinite_augmentation_pseudo_valuation() {
        // [Gauss(v_2), v(x^2+x+1) = +infinity]
        let w = v2_gauss().augment(qpoly(&[1, 1, 1]), QVal::Infinity).unwrap();
        assert!(w.is_final());
        assert_eq!(w.value(&qpoly(&[1, 1, 1])), QVal::Infinity);
        // w(x^2+x+3) = v0(2) = 1 (remainder mod phi)
        assert_eq!(w.value(&qpoly(&[3, 1, 1])), QVal::from_int(1));
        assert_eq!(w.value(&qpoly(&[1, 1])), QVal::from_int(0));
        // cannot augment or step a final valuation
        assert!(w.augment(qpoly(&[0, 1]), QVal::from_int(1)).is_err());
        assert!(w.mac_lane_step(&qpoly(&[1, 1, 1])).is_err());
        assert_eq!(w.ramification_index(), 1);
        assert_eq!(w.residue_degree(), 2);
    }

    #[test]
    fn test_display() {
        let w = w1();
        let s = format!("{}", w);
        assert!(s.contains("Gauss valuation induced by 2-adic valuation"), "{}", s);
        assert!(s.contains("1/2"), "{}", s);
        let winf = v2_gauss().augment(qpoly(&[1, 1, 1]), QVal::Infinity).unwrap();
        assert!(format!("{}", winf).contains("+Infinity"));
    }

    // -- MacLane approximants: the target examples ---------------------------
    // Independent (e, f) derivations in python part A of verify_maclane.py.

    #[test]
    fn test_approximants_x2_plus_1() {
        // x^2+1 over Q_2: irreducible (-1 = 7 mod 8 not a square), ramified:
        // v(theta+1) = 1/2, e = 2, f = 1. Leaf: [Gauss, v(x+1) = 1/2].
        let leaves = mac_lane_approximants(&qpoly(&[1, 0, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 2);
        assert_eq!(w.residue_degree(), 1);
        assert_eq!(w.level(), 1);
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[1, 1]));
        assert_eq!(w.augmentations()[0].lambda(), &QVal::from_frac(1, 2));
    }

    #[test]
    fn test_approximants_x2_minus_2() {
        // x^2-2 over Q_2: Eisenstein, e = 2, f = 1. Leaf: [Gauss, v(x) = 1/2].
        let leaves = mac_lane_approximants(&qpoly(&[-2, 0, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 2);
        assert_eq!(w.residue_degree(), 1);
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[0, 1]));
        assert_eq!(w.augmentations()[0].lambda(), &QVal::from_frac(1, 2));
    }

    #[test]
    fn test_approximants_x2_plus_x_plus_1() {
        // x^2+x+1 over Q_2: irreducible mod 2, unramified: e = 1, f = 2.
        // Leaf: [Gauss, v(x^2+x+1) = +infinity] (f is its own key).
        let leaves = mac_lane_approximants(&qpoly(&[1, 1, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 1);
        assert_eq!(w.residue_degree(), 2);
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[1, 1, 1]));
        assert_eq!(w.augmentations()[0].lambda(), &QVal::Infinity);
    }

    #[test]
    fn test_approximants_x2_plus_x_plus_3() {
        // x^2+x+3 over Q_2: disc = -11 = 5 mod 8, unramified, e=1, f=2;
        // phi(theta) = -2 has v = 1, so the leaf is [Gauss, v(x^2+x+1) = 1].
        let leaves = mac_lane_approximants(&qpoly(&[3, 1, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 1);
        assert_eq!(w.residue_degree(), 2);
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[1, 1, 1]));
        assert_eq!(w.augmentations()[0].lambda(), &QVal::from_int(1));
    }

    #[test]
    fn test_approximants_x2_plus_2x_plus_4() {
        // x^2+2x+4 over Q_2: roots -1 +- sqrt(-3), -3 = 5 mod 8: unramified
        // e = 1, f = 2, root valuation 1 (collinear polygon slope -1).
        // Chain: [Gauss, v(x) = 1, v(x^2+2x+4) = +infinity].
        let leaves = mac_lane_approximants(&qpoly(&[4, 2, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 1);
        assert_eq!(w.residue_degree(), 2);
        assert_eq!(w.level(), 2);
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[0, 1]));
        assert_eq!(w.augmentations()[0].lambda(), &QVal::from_int(1));
        assert_eq!(w.augmentations()[1].phi(), &qpoly(&[4, 2, 1]));
        assert_eq!(w.augmentations()[1].lambda(), &QVal::Infinity);
    }

    #[test]
    fn test_approximants_split_x2_minus_17() {
        // 17 = 1 mod 8 is a 2-adic square: two extensions, e = f = 1.
        // Hensel (python A6): roots +-a with v(a+1) = 1, v(-a+1) = 3.
        // Leaves: [Gauss, v(x+1) = 1] and [Gauss, v(x+1) = 3].
        let leaves = mac_lane_approximants(&qpoly(&[-17, 0, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 2);
        let mut lambdas: Vec<QVal> = leaves
            .iter()
            .map(|w| {
                assert_eq!(w.ramification_index(), 1);
                assert_eq!(w.residue_degree(), 1);
                assert_eq!(w.augmentations()[0].phi(), &qpoly(&[1, 1]));
                w.augmentations()[0].lambda().clone()
            })
            .collect();
        lambdas.sort();
        assert_eq!(lambdas, vec![QVal::from_int(1), QVal::from_int(3)]);
    }

    #[test]
    fn test_approximants_split_with_refinement() {
        // (x-2)(x-10) = x^2-12x+20: both roots have v = 1 and residue 1, so
        // the tree must refine x -> x+2 -> x+6 before separating (python A7):
        // leaves [Gauss, v(x+6) = 3] and [Gauss, v(x+6) = 4]
        // (v(2+6) = 3, v(10+6) = 4).
        let leaves = mac_lane_approximants(&qpoly(&[20, -12, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 2);
        let mut lambdas: Vec<QVal> = leaves
            .iter()
            .map(|w| {
                assert_eq!(w.ramification_index(), 1);
                assert_eq!(w.residue_degree(), 1);
                assert_eq!(w.level(), 1, "collapse must keep the chain at level 1");
                assert_eq!(w.augmentations()[0].phi(), &qpoly(&[6, 1]));
                w.augmentations()[0].lambda().clone()
            })
            .collect();
        lambdas.sort();
        assert_eq!(lambdas, vec![QVal::from_int(3), QVal::from_int(4)]);
    }

    #[test]
    fn test_approximants_split_x2_plus_3x_plus_2() {
        // (x+1)(x+2): exact rational factors. Leaves: [Gauss, v(x+1) = +inf]
        // (exact divisor) and [Gauss, v(x) = 1] (root -2).
        let leaves = mac_lane_approximants(&qpoly(&[2, 3, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 2);
        for w in &leaves {
            assert_eq!(w.ramification_index() * w.residue_degree(), 1);
        }
        let inf_leaf = leaves
            .iter()
            .find(|w| w.is_final())
            .expect("one exact-divisor leaf");
        assert_eq!(inf_leaf.augmentations()[0].phi(), &qpoly(&[1, 1]));
        let fin_leaf = leaves.iter().find(|w| !w.is_final()).expect("one finite leaf");
        assert_eq!(fin_leaf.augmentations()[0].phi(), &qpoly(&[0, 1]));
        assert_eq!(fin_leaf.augmentations()[0].lambda(), &QVal::from_int(1));
    }

    #[test]
    fn test_approximants_x3_minus_2() {
        // Eisenstein cubic: e = 3, f = 1, leaf [Gauss, v(x) = 1/3].
        let leaves = mac_lane_approximants(&qpoly(&[-2, 0, 0, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 3);
        assert_eq!(w.residue_degree(), 1);
        assert_eq!(w.augmentations()[0].lambda(), &QVal::from_frac(1, 3));
    }

    #[test]
    fn test_approximants_x3_minus_x_minus_1() {
        // x^3-x-1 mod 2 = x^3+x+1 irreducible: unramified, e = 1, f = 3.
        // phi(theta) = 2(theta+1) with theta a unit whose residue is not 1,
        // so lambda = 1: leaf [Gauss, v(x^3+x+1) = 1].
        let leaves = mac_lane_approximants(&qpoly(&[-1, -1, 0, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 1);
        assert_eq!(w.residue_degree(), 3);
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[1, 1, 0, 1]));
        assert_eq!(w.augmentations()[0].lambda(), &QVal::from_int(1));
    }

    #[test]
    fn test_approximants_quartic_two_factors() {
        // x^4-x^2-2 = (x^2+1)(x^2-2) over Q_2 (both irreducible): two
        // extensions with e = 2, f = 1; leaves [Gauss, v(x) = 1/2] and
        // [Gauss, v(x+1) = 1/2] (python A11).
        let leaves = mac_lane_approximants(&qpoly(&[-2, 0, -1, 0, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 2);
        let mut keys: Vec<UnivariatePolynomial<Rational>> = Vec::new();
        for w in &leaves {
            assert_eq!(w.ramification_index(), 2);
            assert_eq!(w.residue_degree(), 1);
            assert_eq!(w.augmentations()[0].lambda(), &QVal::from_frac(1, 2));
            keys.push(w.augmentations()[0].phi().clone());
        }
        assert!(keys.contains(&qpoly(&[0, 1])));
        assert!(keys.contains(&qpoly(&[1, 1])));
    }

    #[test]
    fn test_approximants_x2_plus_1_over_v3() {
        // x^2+1 over Q_3: irreducible mod 3, unramified: e = 1, f = 2.
        let leaves = mac_lane_approximants(&qpoly(&[1, 0, 1]), 3).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 1);
        assert_eq!(w.residue_degree(), 2);
        assert_eq!(w.augmentations()[0].lambda(), &QVal::Infinity);
    }

    #[test]
    fn test_approximants_linear() {
        // x - 6 over Q_2: v(6) = 1: leaf [Gauss, v(x) = 1].
        let leaves = mac_lane_approximants(&qpoly(&[-6, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].augmentations()[0].phi(), &qpoly(&[0, 1]));
        assert_eq!(leaves[0].augmentations()[0].lambda(), &QVal::from_int(1));
    }

    #[test]
    fn test_approximants_input_gates() {
        // non-monic
        assert!(mac_lane_approximants(&qpoly(&[1, 0, 2]), 2).is_err());
        // non-squarefree
        assert!(mac_lane_approximants(&qpoly(&[0, 0, 1]), 2).is_err());
        // non-integral
        let f = UnivariatePolynomial::new(vec![rat_frac(1, 2), rat_i64(1)]);
        assert!(mac_lane_approximants(&f, 2).is_err());
        // composite p
        assert!(mac_lane_approximants(&qpoly(&[1, 0, 1]), 4).is_err());
        // constant
        assert!(mac_lane_approximants(&qpoly(&[5]), 2).is_err());
    }

    // -- Stage-2 gates: level >= 2 trees and GF(p^d) residual factoring.
    // Every (e, f) expectation below was derived independently BEFORE being
    // asserted (scratchpad/verify_om_gates.py: PARI/gp idealprimedec +
    // factorpadic + scripted Newton-polygon hand derivations; 55/55 checks).

    #[test]
    fn test_approximants_level2_x4_plus_4x2_plus_12() {
        // x^4 + 4x^2 + 12 over Q_2: irreducible, e = 4, f = 1, and the tree
        // NEEDS a second augmentation. Hand-derived chain (verified in
        // verify_om_gates.py):
        //   polygon slope -1/2, residual (y+1)^2 -> lift key x^2+2 (level 2)
        //   residual (Y+1)^2 again -> same-degree refinement x^2+2x+2
        //   polygon w.r.t. x^2+2x+2: one side of slope -7/4 -> leaf
        //   [Gauss, v(x) = 1/2, v(x^2+2x+2) = 7/4], E = 4, F = 1.
        let f = qpoly(&[12, 0, 4, 0, 1]);
        let leaves = mac_lane_approximants(&f, 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 4);
        assert_eq!(w.residue_degree(), 1);
        assert_eq!(w.level(), 2, "the tree must reach augmentation level 2");
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[0, 1]));
        assert_eq!(w.augmentations()[0].lambda(), &QVal::from_frac(1, 2));
        assert_eq!(w.augmentations()[1].phi(), &qpoly(&[2, 2, 1]));
        assert_eq!(w.augmentations()[1].lambda(), &QVal::from_frac(7, 4));
    }

    #[test]
    fn test_approximants_gf4_residual_irreducible() {
        // x^4 + 2x^3 + 5x^2 + 8x + 3 = phi^2 + 2 phi + 4x (phi = x^2+x+1)
        // over Q_2: residual polynomial y^2 + y + w over GF(4) with
        // Tr(w) = 1: irreducible => single unramified quartic factor,
        // e = 1, f = 4 (gp idealprimedec-confirmed). Exercises residual
        // factoring over GF(4) and a degree-4 key lift to level 2.
        let f = qpoly(&[3, 8, 5, 2, 1]);
        let leaves = mac_lane_approximants(&f, 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 1);
        assert_eq!(w.residue_degree(), 4);
        assert_eq!(w.level(), 2);
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[1, 1, 1]));
        assert_eq!(
            w.augmentations()[1].phi().degree(),
            Some(4),
            "level-2 key must have full degree 4"
        );
    }

    #[test]
    fn test_approximants_gf4_residual_split() {
        // x^4 + 2x^3 + 5x^2 + 4x + 7 = phi^2 + 2 phi + 4 (phi = x^2+x+1)
        // over Q_2: residual y^2 + y + 1 = (y+w)(y+w^2) over GF(4):
        // TWO unramified quadratic factors, e = 1, f = 2 each
        // (gp-confirmed). Exercises GF(4) root finding + same-degree lifts.
        let f = qpoly(&[7, 4, 5, 2, 1]);
        let leaves = mac_lane_approximants(&f, 2).unwrap();
        assert_eq!(leaves.len(), 2);
        for w in &leaves {
            assert_eq!(w.ramification_index(), 1);
            assert_eq!(w.residue_degree(), 2);
            assert_eq!(w.last_key().unwrap().degree(), Some(2));
        }
        // the two leaves carry distinct keys
        assert_ne!(
            leaves[0].last_key().unwrap(),
            leaves[1].last_key().unwrap()
        );
    }

    #[test]
    fn test_approximants_x4_plus_1() {
        // x^4 + 1 over Q_2: Q_2(zeta_8) is totally ramified of degree 4:
        // irreducible, e = 4, f = 1; (x+1)-polygon is Eisenstein-like with
        // slope -1/4 (gp-confirmed). Leaf [Gauss, v(x+1) = 1/4].
        let leaves = mac_lane_approximants(&qpoly(&[1, 0, 0, 0, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 4);
        assert_eq!(w.residue_degree(), 1);
        assert_eq!(w.augmentations()[0].phi(), &qpoly(&[1, 1]));
        assert_eq!(w.augmentations()[0].lambda(), &QVal::from_frac(1, 4));
    }

    #[test]
    fn test_approximants_x6_plus_2x3_plus_4() {
        // x^6 + 2x^3 + 4 over Q_2: single slope -1/3 with residual
        // y^2 + y + 1 irreducible over GF(2): irreducible, e = 3, f = 2
        // (gp-confirmed).
        let leaves = mac_lane_approximants(&qpoly(&[4, 0, 0, 2, 0, 0, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 1);
        let w = &leaves[0];
        assert_eq!(w.ramification_index(), 3);
        assert_eq!(w.residue_degree(), 2);
    }

    #[test]
    fn test_approximants_x3_minus_x_minus_1_q23() {
        // x^3 - x - 1 over Q_23 (disc = -23): one linear factor and one
        // ramified quadratic: (e,f) = (1,1) and (2,1) (gp-confirmed).
        let leaves = mac_lane_approximants(&qpoly(&[-1, -1, 0, 1]), 23).unwrap();
        assert_eq!(leaves.len(), 2);
        let mut efs: Vec<(u64, u64)> = leaves
            .iter()
            .map(|w| (w.ramification_index(), w.residue_degree()))
            .collect();
        efs.sort();
        assert_eq!(efs, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn test_approximants_mixed_slopes_q3() {
        // (x^2 - 3)(x^3 - 3) = x^5 - 3x^3 - 3x^2 + 9 over Q_3: mixed slopes
        // 1/2 and 1/3: two factors with e = 2 and e = 3, f = 1
        // (gp-confirmed).
        let leaves = mac_lane_approximants(&qpoly(&[9, 0, -3, -3, 0, 1]), 3).unwrap();
        assert_eq!(leaves.len(), 2);
        let mut efs: Vec<(u64, u64)> = leaves
            .iter()
            .map(|w| (w.ramification_index(), w.residue_degree()))
            .collect();
        efs.sort();
        assert_eq!(efs, vec![(2, 1), (3, 1)]);
    }

    #[test]
    fn test_approximants_product_two_unramified_quadratics() {
        // (x^2+x+1)(x^2+x+3) over Q_2: both factors reduce to the SAME
        // irreducible x^2+x+1 mod 2, so the tree must refine an equivalent
        // key of degree 2 before separating: two leaves, e = 1, f = 2 each
        // (gp-confirmed).
        let leaves = mac_lane_approximants(&qpoly(&[3, 4, 5, 2, 1]), 2).unwrap();
        assert_eq!(leaves.len(), 2);
        for w in &leaves {
            assert_eq!(w.ramification_index(), 1);
            assert_eq!(w.residue_degree(), 2);
        }
    }

    #[test]
    fn test_leaves_union_coprime_product() {
        // Consistency law: the approximants of f*g (f, g coprime monic
        // squarefree) are the union of those of f and of g, compared by the
        // (E, F, level, last key degree) signature.
        let f = qpoly(&[12, 0, 4, 0, 1]); // x^4+4x^2+12 (level-2 tree)
        let g = qpoly(&[1, 1, 1]); // x^2+x+1 (unramified quadratic)
        let fg = f.clone() * g.clone();
        let sig = |w: &PAdicInductiveValuation| {
            (
                w.ramification_index(),
                w.residue_degree(),
                w.last_key().unwrap().degree().unwrap(),
            )
        };
        let mut union: Vec<_> = mac_lane_approximants(&f, 2)
            .unwrap()
            .iter()
            .map(sig)
            .chain(mac_lane_approximants(&g, 2).unwrap().iter().map(sig))
            .collect();
        let mut product: Vec<_> = mac_lane_approximants(&fg, 2)
            .unwrap()
            .iter()
            .map(sig)
            .collect();
        union.sort();
        product.sort();
        assert_eq!(union, product);
    }

    #[test]
    fn test_level2_residual_multiplicativity_battery() {
        // THE graded law behind the OM tree: for the level-2 valuation
        // w = [Gauss(v_2), v(x) = 1/2, v(x^2+2) = 3/2],
        // R(fg) is an associate of R(f) * R(g) * y^s. A coherence bug in the
        // Q^t-normalization of the residual coefficients would break this.
        let w = w2();
        let data = w.residue_data(2).unwrap();
        let tower = &data.tower;
        let mut rng = Lcg(0x5EED_0009);
        let mut checked = 0;
        for _ in 0..40 {
            let f = rng.poly(3);
            let g = rng.poly(3);
            if f.is_zero() || g.is_zero() {
                continue;
            }
            let (rf, i0f) = w.residual_polynomial_general(2, &f, &data).unwrap();
            let (rg, i0g) = w.residual_polynomial_general(2, &g, &data).unwrap();
            let (rfg, i0fg) = w
                .residual_polynomial_general(2, &(f.clone() * g.clone()), &data)
                .unwrap();
            // y-shifts multiply: i0(fg) = i0(f) + i0(g)
            assert_eq!(i0fg, i0f + i0g, "i0 additivity for f={}, g={}", f, g);
            let prod = tower.poly_mul(&rf, &rg);
            // associate check: rfg = c * prod for a nonzero constant c
            assert_eq!(
                tower.poly_degree(&rfg),
                tower.poly_degree(&prod),
                "degree of R(fg) for f={}, g={}",
                f,
                g
            );
            let d = tower.poly_degree(&rfg) as usize;
            let c = tower
                .e_mul(2, &rfg[d], &tower.e_inv(2, &prod[d]).unwrap());
            for t in 0..=d {
                assert_eq!(
                    rfg[t],
                    tower.e_mul(2, &c, &prod[t]),
                    "R(fg) !~ R(f)R(g) at coeff {} for f={}, g={}",
                    t,
                    f,
                    g
                );
            }
            checked += 1;
        }
        assert!(checked >= 30, "battery too small: {}", checked);
    }

    #[test]
    fn test_approximant_lambda_matches_root_valuation() {
        // Consistency: for w1 = [Gauss, v(x+1) = 1/2] approximating x^2+1,
        // the target satisfies w(f) = min-formula value 1 = v(f(theta))
        // truncated at the approximation level; and stepping the leaf again
        // refines lambda upward without changing (E, F).
        let f = qpoly(&[1, 0, 1]);
        let leaves = mac_lane_approximants(&f, 2).unwrap();
        let w = &leaves[0];
        assert_eq!(w.value(&f), QVal::from_int(1));
        let refined = w.mac_lane_step(&f).unwrap();
        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0].ramification_index(), 2);
        assert_eq!(refined[0].residue_degree(), 1);
        assert!(
            refined[0].augmentations().last().unwrap().lambda() > w.augmentations()[0].lambda()
        );
        // the refined valuation still dominates and agrees below the key
        assert!(refined[0].value(&f) > w.value(&f));
    }
}
