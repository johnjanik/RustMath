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
//!   criterion. At the Gauss level: monic, p-integral, irreducible reduction
//!   mod p. At augmentation level one: degree divisible by `e*d`, one-sided
//!   Newton polygon of slope `-lambda` (`w(phi) = deg(phi)/d * lambda`), and
//!   irreducible residual polynomial not divisible by `y` (plus the
//!   same-degree equivalent-key cases).
//! - [`mac_lane_approximants`]: the MacLane tree for monic squarefree
//!   p-integral `f` over `Q`, iterating `mac_lane_step` until
//!   `sum E(w)*F(w) = deg f` over the leaves. Each leaf approximates exactly
//!   one irreducible factor of `f` over `Q_p`, with `E` and `F` its
//!   ramification index and residue degree.
//!
//! ## Honest limitations (this chunk)
//!
//! - Key checks / `mac_lane_step` need residue-field computations. These are
//!   implemented for the p-adic base at the Gauss level and at augmentation
//!   level one (residue fields `GF(p)` and `GF(p)[z]/(phibar)`); residual
//!   polynomial *factorization* over `GF(p^d)` with `d > 1` and residue
//!   towers at level >= 2 are honest `Err(NotSupported)`, never a guess.
//!   Same-degree (collapsing) augmentations recurse to the truncated
//!   valuation, so chains like `[v0, v(x+2)=2] -> [v0, v(x+6)=3]` stay in
//!   scope at any depth reachable by degree-preserving refinement.
//! - Key checks for *shifted* Gauss valuations (`lambda0 != 0`) are not
//!   implemented; positive shifts are available as the honest augmentation
//!   `[v0, v(x) = lambda]` instead.
//! - Value computation ([`InductiveValuation::value`]) is fully generic in
//!   the chain length and base valuation.
//!
//! Every expected value in the tests was verified independently (sympy /
//! first-principles 2-adic computations) before being asserted; see the
//! stage-2 report.
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

    /// `g / p^u` (exact rational scaling).
    fn scale_by_p_power(&self, g: &UnivariatePolynomial<Rational>, u: i64) -> UnivariatePolynomial<Rational> {
        if u == 0 {
            return g.clone();
        }
        let pk = Integer::from(self.p()).pow(u.unsigned_abs() as u32);
        let factor = if u > 0 {
            Rational::new(Integer::one(), pk).expect("p^u nonzero")
        } else {
            Rational::new(pk, Integer::one()).expect("1 nonzero")
        };
        g.scalar_mul(&factor)
    }

    /// The Gauss-level value of `g` as an exact integer (requires
    /// `lambda0 = 0`, so Gauss values lie in `Z`).
    fn gauss_value_int(&self, g: &UnivariatePolynomial<Rational>) -> Result<Option<i64>> {
        match self.value_at_level(g, 0) {
            QVal::Infinity => Ok(None),
            QVal::Finite(r) => {
                if !r.is_integer() {
                    return Err(MathError::NumericalError(
                        "maclane: non-integral Gauss value".to_string(),
                    ));
                }
                Ok(Some(int_to_i64(r.numerator())?))
            }
        }
    }

    /// Residual polynomial data of `g` with respect to a level-one chain
    /// `[v_0, v(phi) = h/e]` (GMN order-one normalization, up to the global
    /// unit `p^{-u_{i0}}` which does not change the factor structure):
    ///
    /// `R(y) = sum_j res(f_{i0+j*e} / p^{u_{i0+j*e}}) y^j` over
    /// `k_1 = GF(p)[z]/(phibar)`, where `f = sum f_i phi^i`, `u_i = v_0(f_i)`,
    /// the support is the critical line `u_i + i*lambda = w(g)`, and `i0` is
    /// its least index. Returns `(R as k1-coefficient vectors, i0, e, h)`.
    fn residual_polynomial_level1(
        &self,
        g: &UnivariatePolynomial<Rational>,
    ) -> Result<(Vec<Vec<i64>>, usize, usize, i64)> {
        if self.augmentations.len() != 1 {
            return Err(MathError::NotSupported(
                "maclane: residual polynomials only implemented at augmentation level one"
                    .to_string(),
            ));
        }
        if !self.lambda0.is_zero() {
            return Err(MathError::NotSupported(
                "maclane: residual polynomials over shifted Gauss valuations not implemented"
                    .to_string(),
            ));
        }
        let aug = &self.augmentations[0];
        let lambda = match &aug.lambda {
            QVal::Finite(l) => l.clone(),
            QVal::Infinity => {
                return Err(MathError::InvalidArgument(
                    "maclane: final valuation has no residual polynomials".to_string(),
                ))
            }
        };
        if g.is_zero() {
            return Err(MathError::InvalidArgument(
                "maclane: residual polynomial of zero".to_string(),
            ));
        }
        let h = int_to_i64(lambda.numerator())?;
        let e = int_to_i64(lambda.denominator())? as usize;
        let exp = phi_adic_expansion(g, &aug.phi)?;
        // Gauss values of the expansion coefficients (deg < deg phi).
        let mut us: Vec<Option<i64>> = Vec::with_capacity(exp.len());
        for c in &exp {
            us.push(self.gauss_value_int(c)?);
        }
        // w(g) = min(u_i + i*lambda)
        let mut mu: Option<Rational> = None;
        for (i, u) in us.iter().enumerate() {
            if let Some(u) = u {
                let val = rat_i64(*u) + rat_i64(i as i64) * lambda.clone();
                if mu.as_ref().map(|m| &val < m).unwrap_or(true) {
                    mu = Some(val);
                }
            }
        }
        let mu = mu.ok_or_else(|| {
            MathError::InvalidArgument("maclane: residual polynomial of zero".to_string())
        })?;
        let on_line = |i: usize, u: &Option<i64>| -> bool {
            match u {
                None => false,
                Some(u) => rat_i64(*u) + rat_i64(i as i64) * lambda.clone() == mu,
            }
        };
        let i0 = (0..us.len())
            .find(|&i| on_line(i, &us[i]))
            .expect("mu is attained");
        let imax = (0..us.len())
            .rev()
            .find(|&i| on_line(i, &us[i]))
            .expect("mu is attained");
        // All critical indices are congruent mod e (gcd(h, e) = 1).
        for i in i0..=imax {
            if on_line(i, &us[i]) && (i - i0) % e != 0 {
                return Err(MathError::NumericalError(
                    "maclane: internal error: critical index not congruent mod e".to_string(),
                ));
            }
        }
        let m = (imax - i0) / e;
        let mut r: Vec<Vec<i64>> = Vec::with_capacity(m + 1);
        for j in 0..=m {
            let i = i0 + j * e;
            if on_line(i, &us[i]) {
                let scaled = self.scale_by_p_power(&exp[i], us[i].expect("on-line is finite"));
                r.push(self.reduce_poly_mod_p(&scaled)?);
            } else {
                r.push(vec![0]);
            }
        }
        Ok((r, i0, e, h))
    }

    /// Is `phi` a key polynomial for this valuation? Implements the standard
    /// effective criterion (monic + equivalence-irreducible + v-minimal):
    ///
    /// - Gauss level: p-integral with irreducible reduction mod p.
    /// - Level one: `deg phi` a multiple of `e*d`, one-sided Newton polygon
    ///   (`w(phi) = (deg phi / d) * lambda`), residual polynomial irreducible
    ///   and not divisible by `y`; plus the same-degree equivalent-key cases.
    ///
    /// `Err(NotSupported)` where the required residue computation is out of
    /// scope (shifted Gauss; level >= 2 with growing degree; residual
    /// factorization over `GF(p^d)`, `d > 1`).
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
            1 => {
                let aug = &self.augmentations[0];
                let d = aug.phi.degree().expect("key has degree >= 1");
                let lambda = aug
                    .lambda
                    .finite()
                    .expect("non-final valuation has finite lambda")
                    .clone();
                let e = int_to_i64(lambda.denominator())? as usize;
                if n % d != 0 {
                    return Ok(KeyCheck::NotKey(format!(
                        "deg phi = {} is not a multiple of the key degree {}",
                        n, d
                    )));
                }
                let ell = n / d;
                if ell % e != 0 {
                    if n == d {
                        // e >= 2, same degree: keys are exactly phi_old + r
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
                        "deg phi / d = {} is not a multiple of e = {}",
                        ell, e
                    )));
                }
                let m = ell / e;
                // v-minimality: the phi_old-Newton polygon of phi must be
                // one-sided of slope -lambda ending at (ell, 0), i.e.
                // w(phi) = ell * lambda.
                let expected = rat_i64(ell as i64) * lambda.clone();
                if self.value(phi) != QVal::Finite(expected) {
                    return Ok(KeyCheck::NotKey(
                        "phi is not v-minimal (w(phi) < deg(phi)/d * lambda)".to_string(),
                    ));
                }
                let (r, i0, _e, _h) = self.residual_polynomial_level1(phi)?;
                if i0 == 0 {
                    // residual polynomial R of degree m with R(0) != 0
                    debug_assert_eq!(r.len(), m + 1);
                    if d == 1 {
                        let flat: Vec<i64> =
                            r.iter().map(|c| c.first().copied().unwrap_or(0)).collect();
                        let flat = fp_factor::trim(&flat);
                        if fp_factor::degree(&flat) as usize == m
                            && is_irreducible_fp(&flat, self.p())?
                        {
                            Ok(KeyCheck::Key { residual_degree: m })
                        } else {
                            Ok(KeyCheck::NotKey(
                                "residual polynomial is not irreducible".to_string(),
                            ))
                        }
                    } else if m == 1 {
                        // degree-1 residual polynomials are irreducible over
                        // any field
                        Ok(KeyCheck::Key { residual_degree: 1 })
                    } else {
                        Err(MathError::NotSupported(
                            "maclane: factoring residual polynomials over GF(p^d), d > 1, not implemented"
                                .to_string(),
                        ))
                    }
                } else if n == d {
                    // ell = 1, e = 1, support {1}: R = y, phi = key + r with
                    // w(r) > lambda: an equivalent key.
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
            _ => Err(MathError::NotSupported(
                "maclane: key checks at augmentation level >= 2 (residue field towers) not implemented"
                    .to_string(),
            )),
        }
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

    /// Lift a monic residual polynomial `psi` over `k_1 = GF(p)[z]/(phibar)`
    /// (coefficients as `GF(p)[z]`-vectors of degree < d) to a key polynomial
    /// of this level-one valuation with residual polynomial `psi`:
    /// `phi_new = sum_j B_j p^{(m-j)h} phi^{j e}`.
    fn lift_residual_factor(
        &self,
        psi: &[Vec<i64>],
        e: usize,
        h: i64,
    ) -> Result<UnivariatePolynomial<Rational>> {
        let aug = &self.augmentations[0];
        let m = psi.len() - 1;
        if psi[m] != vec![1] {
            return Err(MathError::NumericalError(
                "maclane: internal error: residual factor is not monic".to_string(),
            ));
        }
        // phi^e
        let mut phi_e = UnivariatePolynomial::one();
        for _ in 0..e {
            phi_e = phi_e * aug.phi.clone();
        }
        let mut result = UnivariatePolynomial::zero();
        let mut phi_e_pow = UnivariatePolynomial::one(); // phi^{j*e}
        for (j, b) in psi.iter().enumerate() {
            if !fp_factor::is_zero(b) {
                let lift = UnivariatePolynomial::new(
                    b.iter().map(|&c| rat_i64(c)).collect::<Vec<_>>(),
                );
                let exponent = ((m - j) as i64) * h;
                let pk = Integer::from(self.p()).pow(exponent as u32);
                let scale = Rational::new(pk, Integer::one()).expect("nonzero");
                result = result + (lift * phi_e_pow.clone()).scalar_mul(&scale);
            }
            if j < m {
                phi_e_pow = phi_e_pow * phi_e.clone();
            }
        }
        Ok(result)
    }

    /// One step of the MacLane algorithm toward the monic squarefree
    /// p-integral target `f`: returns the valuations replacing `self` in the
    /// leaf set (each strictly closer to the extensions of `v_p` determined
    /// by the irreducible factors of `f` over `Q_p`).
    ///
    /// Implemented at the Gauss level and at augmentation level one (with
    /// same-degree refinements collapsing, so degree-preserving chains can
    /// be iterated indefinitely); an honest `Err(NotSupported)` where the
    /// residue tower is out of scope.
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
            1 => {
                let phi_last = self.augmentations[0].phi.clone();
                let d = phi_last.degree().expect("key has degree >= 1");
                // (a) roots closer to the current key: steeper polygon sides
                // (and an exact-division infinite leaf); same-degree
                // augmentations collapse inside augment().
                let mut kids = self.children_for_key(f, &phi_last)?;
                // (b) residual factors away from y: new or refined keys.
                let (r, _i0, e, h) = self.residual_polynomial_level1(f)?;
                if r.len() >= 2 {
                    let psis: Vec<Vec<Vec<i64>>> = if d == 1 {
                        let flat: Vec<i64> =
                            r.iter().map(|c| c.first().copied().unwrap_or(0)).collect();
                        let flat = fp_factor::trim(&flat);
                        factor_fp_certified(&flat, self.p())?
                            .into_iter()
                            .map(|(psi, _mult)| psi.into_iter().map(|c| vec![c]).collect())
                            .collect()
                    } else if r.len() == 2 {
                        // degree-1 residual polynomial: monicize over k_1
                        vec![monicize_k1(&r, &self.reduce_poly_mod_p(&phi_last)?, self.p())?]
                    } else {
                        return Err(MathError::NotSupported(
                            "maclane: factoring residual polynomials over GF(p^d), d > 1, not implemented"
                                .to_string(),
                        ));
                    };
                    for psi in &psis {
                        let phi_new = self.lift_residual_factor(psi, e, h)?;
                        kids.extend(self.children_for_key(f, &phi_new)?);
                    }
                }
                if kids.is_empty() {
                    return Err(MathError::NumericalError(
                        "maclane: internal error: level-one step made no progress".to_string(),
                    ));
                }
                Ok(kids)
            }
            _ => Err(MathError::NotSupported(
                "maclane: mac_lane_step at augmentation level >= 2 (residue field towers) not implemented"
                    .to_string(),
            )),
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

/// Monicize a degree-one polynomial over `k_1 = GF(p)[z]/(phibar)`.
fn monicize_k1(r: &[Vec<i64>], phibar: &[i64], p: i64) -> Result<Vec<Vec<i64>>> {
    debug_assert_eq!(r.len(), 2);
    let lc = &r[1];
    // invert lc modulo (phibar, p)
    let (g, s, _t) = fp_factor::extended_gcd(lc, phibar, p);
    if fp_factor::degree(&g) != 0 {
        return Err(MathError::NumericalError(
            "maclane: internal error: residual leading coefficient not invertible".to_string(),
        ));
    }
    let g0_inv = fp_factor::mod_inv(g[0], p).ok_or_else(|| {
        MathError::NumericalError("maclane: internal error: gcd unit not invertible".to_string())
    })?;
    let inv = fp_factor::mul(&s, &[g0_inv], p);
    let (_q, inv) = fp_factor::div_mod(&inv, phibar, p);
    let c0 = fp_factor::mul(&r[0], &inv, p);
    let (_q, c0) = fp_factor::div_mod(&c0, phibar, p);
    Ok(vec![c0, vec![1]])
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
