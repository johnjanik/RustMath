//! Frobenius on `Jac[2]` from the Frobenius polynomial: the module type of `Jac[2]` over
//! `F_2[x]`, the order of Frobenius on it, its type at the eigenvalue 1, and the types of an
//! `F`-stable Lagrangian subspace, each decided only when every module type allowed by the rank
//! bounds and the Weil pairing (S1, S2) agrees. These are not all the constraints that `P`
//! imposes: the `Z_2[F]`-stable lattices of `Q_2[x]/(P)` (design C57 2.8) are not used.
//!
//! Specification: MAGMA Handbook ch. 114 *Algebraic Curves* (`LPolynomial(C, m)`,
//! `ZetaFunction(C, m)`: the L-polynomial over the degree-m extension, digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md:418`; `ClassGroupAbelianInvariants(C)`,
//! `:599`) and ch. 42 *Algebraic Function Fields* (`LPolynomial(F, m)`, `ClassNumber(F)`,
//! `ClassGroupAbelianInvariants(F)`; `Chapter_42_Algebraic_Function_Fields.md:241`, `:266`,
//! `:268`); ch. 146 (Schur functions and their products). MAGMA computes the class group of a
//! function field by index calculus; this module uses only what `P` says about `Jac[2]`. The M23
//! requests: `docs/workstreams/m23_golay/REQUESTS.md` 1.2 (the rule "2-rank <= v_2(#Jac)") and
//! C-5. The proofs are in the design C57, Section 2: `docs/workstreams/m23_golay/c57/design/C57.md`
//! (copied there with its reviews by the Docs phase of the same PR, as for `wpf/` and `wpp/`).
//!
//! Mathematics (design C57 Section 2):
//! - `J = Jac[2] = T_2 / 2 T_2` is an `F_2[x]`-module through Frobenius `F`, with characteristic
//!   polynomial `P mod 2` (J. Tate, *Endomorphisms of abelian varieties over finite fields*,
//!   Invent. Math. 2 (1966), Thm. 1 and its proof, `l = 2 != p`). Its type is one partition
//!   `lambda_f` of `e_f` for each irreducible factor `f` of multiplicity `e_f`.
//! - Lemma A (rank bound): for `g` in `Z[x]` with `Res(P, g) != 0`,
//!   `dim ker gbar(F) <= v_2(Res(P, g))` (Smith normal form of `g(F)` on `T_2`). With
//!   `g = x^k - 1` this is the rule of REQUESTS 1.2 over `F_(q^k)`: `Res(P, x^k - 1) =
//!   #Jac(F_(q^k))`; with `g = x^k + 1`, the order of the quadratic twist over `F_(q^k)`.
//!   Only `k | N_max` matter (design Lemma E). The valuations are computed modulo `2^128` as
//!   `v_2 det` of multiplication by `x^k -+ 1` on `Z[x]/(P)`, by elimination with pivots of least
//!   valuation (C57 review 1, B1): exact when every elementary divisor has valuation below 128,
//!   else the bound is skipped (it would be vacuous for `2g < 128`; skipping a bound never makes
//!   a result wrong). The cost does not depend on the size of `q`.
//! - The Weil pairing (D. Mumford, *Abelian varieties*, 1970, Sections 20 and 23): `F` lies in
//!   `Sp(J)`, so `lambda_f = lambda_(f*)` (S1, `f*` the reciprocal) and at `f = x + 1` every
//!   odd part has even multiplicity (S2; W. H. Hesselink, *Nilpotency in classical groups over a
//!   field of characteristic 2*, Math. Z. 166 (1979); a complete proof is in the design, 2.4).
//!   They never change the set of possible orders (design review 1, 1.4), only the lists of
//!   types.
//! - An `F`-stable Lagrangian `W` with quotient `J/W` (J. A. Green, *The characters of the
//!   finite general linear groups*, Trans. AMS 80 (1955); T. Klein, *The multiplication of
//!   Schur-functions and extensions of p-modules*, J. London Math. Soc. 43 (1968); I. G.
//!   Macdonald, *Symmetric functions and Hall polynomials*, 2nd ed., II.4 (4.3)): a submodule of
//!   type `mu` with quotient of type `nu` exists only if the Littlewood-Richardson coefficient
//!   `c^lambda_(mu, nu)` is nonzero. `J_f / W_f` has type `mu_f` (`f = f*`) or `mu_(f*)` (design
//!   Lemma F). The coefficients come from `rustmath_symmetricfunctions::multiply` in the Schur
//!   basis (checked against GAP, `tests/data/two_torsion/lr_gap.txt`).
//!
//! Every result is a statement about `P`. It is a statement about a curve only when `P` is
//! certified to be its Frobenius polynomial, and it assumes that `P` is the Frobenius polynomial
//! of a principally polarized abelian variety over `F_q`, `q` odd (a Jacobian).

use super::{lcm, order_of_x_mod2, FrobeniusPolynomial, TwoTorsionOrder, ORDER_OVERFLOW};
use rustmath_combinatorics::{number_of_partitions, partitions, Partition};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;
use rustmath_rationals::Rational;
use rustmath_symmetricfunctions::{multiply, schur_function, SymFun};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;

/// The largest `k` for [`FrobeniusPolynomial::base_change`], and the largest `N_max` for which
/// the rank bounds are computed.
pub const MAX_BASE_CHANGE_DEGREE: u32 = 4096;

/// The largest multiplicity of a factor of `P mod 2` for
/// [`TwoTorsionAnalysis::lagrangian_types`] (Schur products of degree up to 8): the range in
/// which the Littlewood-Richardson coefficients are checked against GAP and the criterion by
/// enumeration (the tests of this module), and all that genus 4 needs. The cost grows fast with
/// the multiplicity: `P mod 2 = (x + 1)^m`, every type, in a debug build: 0.4 s at `m = 8`,
/// 4.1 s at 10, 51 s at 12, more than 20 min at 16 (C57 review 1, m2).
pub const MAX_LR_SIZE: u32 = 8;

/// The largest number of types (the product of the partition numbers of the multiplicities)
/// that the analysis enumerates.
pub const MAX_CANDIDATES: u128 = 1_000_000;

/// Why the analysis of `Jac[2]` was not made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TwoTorsionError {
    /// `q` is even: `T_2` is not the right object (`l = p`), and `P mod 2` is not the
    /// characteristic polynomial of Frobenius on `Jac[2]`.
    EvenCharacteristic,
    /// The number of types to enumerate (saturating) is above [`MAX_CANDIDATES`].
    TooManyCandidates { count: u128 },
    /// `N_max` above [`MAX_BASE_CHANGE_DEGREE`], or a multiplicity above [`MAX_LR_SIZE`] in
    /// [`TwoTorsionAnalysis::lagrangian_types`].
    TooLarge(String),
    /// `lagrangian_types`: `w w^rev != P mod 2`.
    NotAReciprocalHalf,
    /// No type survives (or `P mod 2` is not self-reciprocal): `P` is not the Frobenius
    /// polynomial of a principally polarized abelian variety over `F_q`.
    Inconsistent,
}

impl fmt::Display for TwoTorsionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TwoTorsionError::EvenCharacteristic => {
                write!(f, "q is even: P mod 2 is not Frobenius on Jac[2]")
            }
            TwoTorsionError::TooManyCandidates { count } => write!(
                f,
                "{count} module types to enumerate, above the limit {MAX_CANDIDATES}"
            ),
            TwoTorsionError::TooLarge(s) => write!(f, "too large: {s}"),
            TwoTorsionError::NotAReciprocalHalf => write!(f, "w w^rev is not P mod 2"),
            TwoTorsionError::Inconsistent => write!(
                f,
                "no module type is consistent with P: P is not the Frobenius polynomial of a \
                 principally polarized abelian variety over F_q"
            ),
        }
    }
}

impl std::error::Error for TwoTorsionError {}

/// One bound of Lemma A: `dim ker(F^k - 1 | Jac[2]) <= v2`, with `v2 = v_2(#Jac(F_(q^k)))`
/// (`twisted = false`) or `v2` of the order of the quadratic twist of `Jac` over `F_(q^k)`
/// (`twisted = true`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RankBound {
    pub k: u64,
    pub twisted: bool,
    pub v2: u32,
}

impl RankBound {
    /// `v2 #Jac(F_q) = 1`, `v2 #Jac(F_q^5) = 4`, `v2 #Jac'(F_q^4) = 6` (the prime marks the twist).
    pub fn label(&self) -> String {
        let field = if self.k == 1 {
            "F_q".to_string()
        } else {
            format!("F_q^{}", self.k)
        };
        let twist = if self.twisted { "'" } else { "" };
        format!("v2 #Jac{twist}({field}) = {}", self.v2)
    }
}

/// An irreducible factor `f` of `P mod 2` (ascending bits) with its multiplicity `e_f`, the
/// order of `x` modulo `f`, and the index of the reciprocal `f*` in the list of factors.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PrimaryFactor {
    pub f: Vec<i64>,
    pub multiplicity: u32,
    pub order_of_x: u64,
    pub reciprocal: usize,
}

impl PrimaryFactor {
    /// `deg f`.
    pub fn degree(&self) -> usize {
        self.f.len() - 1
    }

    /// `f = x + 1`.
    pub fn is_x_plus_1(&self) -> bool {
        self.f == [1, 1]
    }
}

/// A type of `Jac[2]`: one partition (non-increasing) per factor of `P mod 2`, aligned with
/// [`TwoTorsionAnalysis::factors`]. The part `a` of the partition at `f` is a summand
/// `F_2[x]/(f^a)`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleType(pub Vec<Vec<u32>>);

/// The analysis of `Jac[2]` from `P` (design C57 2.6): the types allowed by the symplectic
/// constraints and by every rank bound with `k | N_max`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TwoTorsionAnalysis {
    /// The size of the base field.
    pub q: Integer,
    /// The irreducible factors of `P mod 2`, in the order of `factor_mod(2)`.
    pub factors: Vec<PrimaryFactor>,
    /// `S`, the lcm of the orders of `x` modulo the factors.
    pub semisimple_order: u64,
    /// `N_max = S 2^ceil(log2 e_max)`: every possible order divides it.
    pub max_order: u64,
    /// Every bound computed: `k | N_max` ascending, untwisted first, then twisted; `v2` is
    /// exact. Empty when `P mod 2` is squarefree (one type).
    pub bounds: Vec<RankBound>,
    /// `(k, twisted)` where the computation modulo `2^128` shows only `v2 >= 128` (some
    /// elementary divisor of `x^k -+ 1` on `Z[x]/(P)` has valuation 128 or more), in particular
    /// where the group order is 0 (possible only when a root of `P` is a root of unity, never
    /// for a Weil polynomial over `F_q`, `q > 1`). No bound is taken from them; for `2g < 128`
    /// such a bound would be vacuous (`dim ker <= 2g`).
    pub skipped_bounds: Vec<(u64, bool)>,
    /// The candidate types, sorted, never empty.
    pub candidates: Vec<ModuleType>,
    /// The number of types removed by S1 and S2 before the bounds.
    pub symplectic_removed: usize,
    /// `P mod 2`, ascending.
    p_mod_2: Vec<i64>,
    /// Every type allowed by S1 and S2 (before the bounds), for the decisive bounds.
    symplectic: Vec<ModuleType>,
}

/// A type of an `F`-stable Lagrangian `W` (design C57 2.5), from
/// [`TwoTorsionAnalysis::lagrangian_types`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LagrangianType {
    /// `(f, mu_f)` for each factor `f` of the characteristic polynomial `w` of `F` on `W`, in the
    /// order of `factor_mod(2)` of `P`: `mu_f` is a partition of the multiplicity of `f` in `w`.
    pub parts: Vec<(Vec<i64>, Vec<u32>)>,
    /// The indices of the candidates of `Jac[2]` this type is compatible with.
    pub compatible: Vec<usize>,
    orders_of_x: Vec<u64>,
}

impl LagrangianType {
    /// `mu_(x+1)`: the type of `W` at the eigenvalue 1, empty if `x + 1` does not divide `w`.
    pub fn eigenvalue_one(&self) -> Vec<u32> {
        self.parts
            .iter()
            .find(|(f, _)| f == &[1, 1])
            .map(|(_, mu)| mu.clone())
            .unwrap_or_default()
    }

    /// The order of `F` on `W` (Lemma D on `W`).
    pub fn order(&self) -> u64 {
        self.parts
            .iter()
            .zip(&self.orders_of_x)
            .fold(1, |acc, ((_, mu), &o)| lcm(acc, block_order(o, mu)))
    }

    /// `dim ker(F - 1 | W)`: the number of parts of `mu_(x+1)`.
    pub fn fixed_dimension(&self) -> usize {
        self.eigenvalue_one().len()
    }
}

/// `ceil(log2 a)` for `a >= 1`.
fn ceil_log2(a: u64) -> u32 {
    let mut t = 0;
    while (1u64 << t) < a {
        t += 1;
    }
    t
}

/// The order of `x` on `sum_i F_2[x]/(f^(mu_i))` for `f` with `x` of order `o` (Lemma D).
fn block_order(o: u64, mu: &[u32]) -> u64 {
    let top = mu.iter().copied().max().unwrap_or(0) as u64;
    if top == 0 {
        return 1;
    }
    o.checked_mul(1u64 << ceil_log2(top)).expect(ORDER_OVERFLOW)
}

/// The partitions of `n`, non-increasing parts, in reverse lexicographic order (`8` before `6 2`).
fn partition_list(n: u32) -> Vec<Vec<u32>> {
    let mut v: Vec<Vec<u32>> = partitions(n as usize)
        .iter()
        .map(|p| p.parts().iter().map(|&x| x as u32).collect())
        .collect();
    v.sort();
    v.reverse();
    v
}

fn reverse_f2(f: &[i64]) -> Vec<i64> {
    fp_factor::trim(&f.iter().rev().cloned().collect::<Vec<_>>())
}

/// The irreducible factors of `pbar` over `F_2` (`pbar(0) = 1`) with their orders and
/// reciprocals. `Inconsistent` if `x` divides `pbar` or `pbar` is not self-reciprocal up to the
/// multiplicities. Panics (in `order_of_x_mod2`) on a factor of degree above 62.
pub(crate) fn primary_factors(pbar: &[i64]) -> Result<Vec<PrimaryFactor>, TwoTorsionError> {
    let mut out: Vec<PrimaryFactor> = Vec::new();
    for (f, e) in fp_factor::factor_with_multiplicity(pbar, 2) {
        let f = fp_factor::trim(&f);
        if f == [0, 1] {
            return Err(TwoTorsionError::Inconsistent);
        }
        let o = order_of_x_mod2(&f);
        out.push(PrimaryFactor {
            f,
            multiplicity: e,
            order_of_x: o,
            reciprocal: 0,
        });
    }
    for i in 0..out.len() {
        let r = reverse_f2(&out[i].f);
        let j = out
            .iter()
            .position(|g| g.f == r)
            .ok_or(TwoTorsionError::Inconsistent)?;
        if out[j].multiplicity != out[i].multiplicity {
            return Err(TwoTorsionError::Inconsistent);
        }
        out[i].reciprocal = j;
    }
    Ok(out)
}

/// S1 and S2 (design C57 2.4): `lambda_f = lambda_(f*)`, and at `x + 1` every odd part has even
/// multiplicity.
pub(crate) fn symplectic_ok(factors: &[PrimaryFactor], t: &ModuleType) -> bool {
    for (i, f) in factors.iter().enumerate() {
        if t.0[i] != t.0[f.reciprocal] {
            return false;
        }
        if f.is_x_plus_1() {
            let lam = &t.0[i];
            let mut j = 0;
            while j < lam.len() {
                let mut m = 1;
                while j + m < lam.len() && lam[j + m] == lam[j] {
                    m += 1;
                }
                if lam[j] % 2 == 1 && m % 2 == 1 {
                    return false;
                }
                j += m;
            }
        }
    }
    true
}

/// Littlewood-Richardson coefficients `c^lambda_(mu, nu)` from the Schur expansion of
/// `s_mu s_nu` (`rustmath_symmetricfunctions::multiply`), one product per pair `(mu, nu)`.
#[derive(Default)]
pub(crate) struct LrCache {
    products: HashMap<(Vec<u32>, Vec<u32>), SymFun>,
}

fn contains(lambda: &[u32], mu: &[u32]) -> bool {
    mu.len() <= lambda.len() && mu.iter().zip(lambda).all(|(m, l)| m <= l)
}

fn schur(p: &[u32]) -> SymFun {
    schur_function(Partition::new(p.iter().map(|&x| x as usize).collect()))
}

impl LrCache {
    /// `c^lambda_(mu, nu)` (partitions non-increasing).
    pub(crate) fn coefficient(&mut self, lambda: &[u32], mu: &[u32], nu: &[u32]) -> u64 {
        let size = |v: &[u32]| v.iter().map(|&x| x as u64).sum::<u64>();
        if size(lambda) != size(mu) + size(nu) {
            return 0;
        }
        if mu.is_empty() {
            return u64::from(lambda == nu);
        }
        if nu.is_empty() {
            return u64::from(lambda == mu);
        }
        if !contains(lambda, mu) || !contains(lambda, nu) {
            return 0;
        }
        let prod = self
            .products
            .entry((mu.to_vec(), nu.to_vec()))
            .or_insert_with(|| multiply(&schur(mu), &schur(nu)));
        let c: Rational = prod.coeff(&Partition::new(
            lambda.iter().map(|&x| x as usize).collect(),
        ));
        assert!(
            c.denominator().is_one(),
            "Littlewood-Richardson coefficient {c} is not an integer"
        );
        c.numerator()
            .to_i64()
            .and_then(|x| u64::try_from(x).ok())
            .expect("a Littlewood-Richardson coefficient is a small non-negative integer")
    }
}

/// Lemma F with the Green-Klein criterion: the type `mu` of `W` (one partition per factor of
/// `P mod 2`, of the multiplicity of the factor in `w`) is compatible with the type `lambda` of
/// `J` iff `c^(lambda_f)_(mu_f, mu_(f*)) != 0` for every factor `f`.
pub(crate) fn lagrangian_compatible(
    cache: &mut LrCache,
    factors: &[PrimaryFactor],
    lambda: &ModuleType,
    mu: &[Vec<u32>],
) -> bool {
    factors
        .iter()
        .enumerate()
        .all(|(i, f)| cache.coefficient(&lambda.0[i], &mu[i], &mu[f.reciprocal]) != 0)
}

/// `x mod 2^128` (two's complement for a negative `x`).
fn residue_mod_2_128(x: &Integer) -> u128 {
    let (sign, digits) = x.as_bigint().to_u64_digits();
    let lo = u128::from(digits.first().copied().unwrap_or(0))
        | (u128::from(digits.get(1).copied().unwrap_or(0)) << 64);
    if sign == num_bigint::Sign::Minus {
        lo.wrapping_neg()
    } else {
        lo
    }
}

/// `x^n mod P` in `(Z/2^128)[x]` for a monic `P` (ascending, `p[deg] = 1`): `r` is reduced in
/// place from its top coefficient down, and keeps `deg P` coefficients.
fn reduce_mod_monic(mut r: Vec<u128>, p: &[u128]) -> Vec<u128> {
    let n = p.len() - 1;
    for d in (n..r.len()).rev() {
        let c = r[d];
        if c != 0 {
            for (x, &pi) in r[d - n..d].iter_mut().zip(&p[..n]) {
                *x = x.wrapping_sub(c.wrapping_mul(pi));
            }
        }
    }
    r.resize(n, 0);
    r
}

/// `a b mod P` in `(Z/2^128)[x]` (`a`, `b` of degree below `deg P`).
fn mul_mod_monic(a: &[u128], b: &[u128], p: &[u128]) -> Vec<u128> {
    let mut r = vec![0u128; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            r[i + j] = r[i + j].wrapping_add(x.wrapping_mul(y));
        }
    }
    reduce_mod_monic(r, p)
}

/// The inverse of an odd `u` modulo `2^128` (Newton: each step doubles the correct bits, from 3).
fn inverse_odd_mod_2_128(u: u128) -> u128 {
    debug_assert!(u & 1 == 1);
    let mut x = u; // u u = 1 mod 8
    for _ in 0..6 {
        x = x.wrapping_mul(2u128.wrapping_sub(u.wrapping_mul(x)));
    }
    x
}

/// `v_2(det M)` for a square matrix `M` over `Z_2` known modulo `2^128` (rows). Gaussian
/// elimination with a pivot of least valuation in the whole remaining block: over `Z_2` this
/// computes the elementary divisors `2^(e_i)` of `M`, and modulo `2^128` it sees `e_i` exactly
/// while `e_i < 128` (the multipliers agree with the exact ones modulo `2^(128 - e)`, and the
/// pivot row is divisible by `2^e`). `Some(sum e_i)` when every `e_i < 128` (exact, even if the
/// sum is 128 or more); `None` when the remaining block vanishes modulo `2^128`, so that
/// `v_2(det M) >= 128` (`det M = 0` included).
fn det_valuation_2(mut m: Vec<Vec<u128>>) -> Option<u32> {
    let n = m.len();
    let mut v = 0u32;
    for c in 0..n {
        let mut best: Option<(u32, usize, usize)> = None;
        for (i, row) in m.iter().enumerate().skip(c) {
            for (j, &x) in row.iter().enumerate().skip(c) {
                let t = x.trailing_zeros();
                if t < 128 && best.is_none_or(|b| t < b.0) {
                    best = Some((t, i, j));
                }
            }
        }
        let (t, pi, pj) = best?;
        m.swap(c, pi);
        for row in m.iter_mut() {
            row.swap(c, pj);
        }
        let uinv = inverse_odd_mod_2_128(m[c][c] >> t);
        let pivot_row = m[c].clone();
        for row in m.iter_mut().skip(c + 1) {
            if row[c] == 0 {
                continue;
            }
            let mult = (row[c] >> t).wrapping_mul(uinv);
            for (x, &y) in row[c..].iter_mut().zip(&pivot_row[c..]) {
                *x = x.wrapping_sub(mult.wrapping_mul(y));
            }
            debug_assert_eq!(row[c], 0);
        }
        v += t;
    }
    Some(v)
}

/// `[v_2(Res(P, x^k - 1)), v_2(Res(P, x^k + 1))]` from the coefficients of `P` modulo `2^128`
/// (ascending, monic, degree at least 1). `Res(P, g)` is the determinant of multiplication by
/// `g` on `Z[x]/(P)`, so with `r = x^k mod P` it is `det` of the matrix with columns
/// `x^j (r -+ 1) mod P`, `0 <= j < deg P`; see [`det_valuation_2`] for `None`. Cost: `log2 k`
/// products modulo `P` and two eliminations, `O(n^2 log k + n^3)` word operations for
/// `n = deg P`, whatever the size of `q`.
fn resultant_valuations(p: &[u128], k: u64) -> [Option<u32>; 2] {
    let n = p.len() - 1;
    let mut r = reduce_mod_monic(vec![1], p);
    let mut base = reduce_mod_monic(vec![0, 1], p);
    let mut e = k;
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod_monic(&r, &base, p);
        }
        e >>= 1;
        if e > 0 {
            base = mul_mod_monic(&base, &base, p);
        }
    }
    [1u128.wrapping_neg(), 1].map(|s| {
        let mut g = r.clone();
        g[0] = g[0].wrapping_add(s);
        let mut cols = Vec::with_capacity(n);
        for _ in 0..n {
            let mut next = vec![0u128; n + 1];
            next[1..].copy_from_slice(&g);
            cols.push(g);
            g = reduce_mod_monic(next, p);
        }
        // the transpose has the same determinant
        det_valuation_2(cols)
    })
}

/// Every tuple of choices, one from each list (the lists are non-empty).
fn product_of<T: Clone>(lists: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut out: Vec<Vec<T>> = vec![Vec::new()];
    for l in lists {
        let mut next = Vec::with_capacity(out.len() * l.len());
        for prefix in &out {
            for x in l {
                let mut v = prefix.clone();
                v.push(x.clone());
                next.push(v);
            }
        }
        out = next;
    }
    out
}

/// The analysis (design C57 2.6). See [`FrobeniusPolynomial::two_torsion_analysis`].
pub(super) fn analyse(fp: &FrobeniusPolynomial) -> Result<TwoTorsionAnalysis, TwoTorsionError> {
    if fp.p.is_even() {
        return Err(TwoTorsionError::EvenCharacteristic);
    }
    let pbar = fp_factor::trim(&fp.reduce_mod(2));
    let factors = primary_factors(&pbar)?;
    let s = factors.iter().fold(1, |a, f| lcm(a, f.order_of_x));
    let emax = factors.iter().map(|f| f.multiplicity).max().unwrap_or(1);
    let n_max = s
        .checked_mul(1u64 << ceil_log2(emax as u64))
        .expect(ORDER_OVERFLOW);
    if emax <= 1 {
        let t = ModuleType(factors.iter().map(|_| vec![1]).collect());
        return Ok(TwoTorsionAnalysis {
            q: fp.p.clone(),
            factors,
            semisimple_order: s,
            max_order: n_max,
            bounds: Vec::new(),
            skipped_bounds: Vec::new(),
            candidates: vec![t.clone()],
            symplectic_removed: 0,
            p_mod_2: pbar,
            symplectic: vec![t],
        });
    }
    // p(e) exactly for e <= 1000 (p(1000) < 2^105), saturated above (C57 review 1, m1:
    // partition_count panics from p(417) > u64::MAX on).
    let count = factors.iter().fold(1u128, |acc, f| {
        let n = if f.multiplicity > 1000 {
            u128::MAX
        } else {
            u128::try_from(&number_of_partitions(u64::from(f.multiplicity))).unwrap_or(u128::MAX)
        };
        acc.saturating_mul(n)
    });
    if count > MAX_CANDIDATES {
        return Err(TwoTorsionError::TooManyCandidates { count });
    }
    if n_max > MAX_BASE_CHANGE_DEGREE as u64 {
        return Err(TwoTorsionError::TooLarge(format!(
            "N_max = {n_max} above {MAX_BASE_CHANGE_DEGREE}"
        )));
    }
    let lists: Vec<Vec<Vec<u32>>> = factors
        .iter()
        .map(|f| partition_list(f.multiplicity))
        .collect();
    let all = product_of(&lists);
    let total = all.len();
    let symplectic: Vec<ModuleType> = all
        .into_iter()
        .map(ModuleType)
        .filter(|t| symplectic_ok(&factors, t))
        .collect();
    let mut a = TwoTorsionAnalysis {
        q: fp.p.clone(),
        factors,
        semisimple_order: s,
        max_order: n_max,
        bounds: Vec::new(),
        skipped_bounds: Vec::new(),
        candidates: Vec::new(),
        symplectic_removed: total - symplectic.len(),
        p_mod_2: pbar,
        symplectic,
    };
    // The bounds of Lemma A for k | N_max (Lemma E): v_2(P^(k)(1)) = v_2(Res(P, x^k - 1)) and
    // v_2(P^(k)(-1)) = v_2(Res(P, x^k + 1)), computed modulo 2^128 (C57 review 1, B1: no power
    // sums at full precision, whose size grew like (g N_max)^2 log2 q).
    let divisors: Vec<u64> = (1..=n_max).filter(|k| n_max.is_multiple_of(*k)).collect();
    let two_g = 2 * fp.genus;
    let p_res: Vec<u128> = (0..=two_g)
        .map(|j| residue_mod_2_128(&fp.coeffs[two_g - j]))
        .collect();
    let vals: Vec<[Option<u32>; 2]> = divisors
        .iter()
        .map(|&k| resultant_valuations(&p_res, k))
        .collect();
    for (s, twisted) in [false, true].into_iter().enumerate() {
        for (&k, v) in divisors.iter().zip(&vals) {
            match v[s] {
                None => a.skipped_bounds.push((k, twisted)),
                Some(v2) => a.bounds.push(RankBound { k, twisted, v2 }),
            }
        }
    }
    let mut cands: Vec<ModuleType> = a
        .symplectic
        .iter()
        .filter(|t| a.passes(t, &a.bounds))
        .cloned()
        .collect();
    if cands.is_empty() {
        return Err(TwoTorsionError::Inconsistent);
    }
    cands.sort();
    a.candidates = cands;
    Ok(a)
}

impl TwoTorsionAnalysis {
    /// `ord(lambda) = lcm_f o_f 2^ceil(log2 max lambda_f)` (design Lemma D). Panics if it does
    /// not fit in a `u64`.
    pub fn order_of(&self, t: &ModuleType) -> u64 {
        self.factors
            .iter()
            .zip(&t.0)
            .fold(1, |acc, (f, lam)| lcm(acc, block_order(f.order_of_x, lam)))
    }

    /// `r_k(lambda) = dim ker(F^k - 1)` on a module of type `t` (design Lemma C): with
    /// `k = 2^s k'`, `k'` odd, the sum over the factors with `o_f | k'` of
    /// `d_f sum_i min(2^s, lambda_i)`.
    pub fn kernel_dimension(&self, t: &ModuleType, k: u64) -> usize {
        assert!(k >= 1, "kernel_dimension: k = 0");
        let s = k.trailing_zeros();
        let kp = k >> s;
        let cap: u64 = if s >= 32 { u64::MAX } else { 1u64 << s };
        self.factors
            .iter()
            .zip(&t.0)
            .filter(|(f, _)| kp.is_multiple_of(f.order_of_x))
            .map(|(f, lam)| {
                f.degree()
                    * lam
                        .iter()
                        .map(|&x| (x as u64).min(cap) as usize)
                        .sum::<usize>()
            })
            .sum()
    }

    fn passes(&self, t: &ModuleType, bounds: &[RankBound]) -> bool {
        bounds
            .iter()
            .all(|b| self.kernel_dimension(t, b.k) as u64 <= b.v2 as u64)
    }

    /// The possible orders of Frobenius on `Jac[2]`, sorted, distinct.
    pub fn orders(&self) -> Vec<u64> {
        let s: BTreeSet<u64> = self.candidates.iter().map(|t| self.order_of(t)).collect();
        s.into_iter().collect()
    }

    fn orders_under(&self, bounds: &[RankBound]) -> BTreeSet<u64> {
        self.symplectic
            .iter()
            .filter(|t| self.passes(t, bounds))
            .map(|t| self.order_of(t))
            .collect()
    }

    /// A minimal set of bounds that decides the order on its own (design 2.6): bounds are
    /// dropped from the last (twisted after untwisted, larger `k` later) while the order stays
    /// decided. Empty unless the order is decided and `P mod 2` has a repeated factor.
    pub fn decisive_bounds(&self) -> Vec<RankBound> {
        if self.bounds.is_empty() || self.orders().len() != 1 {
            return Vec::new();
        }
        let mut keep = self.bounds.clone();
        for b in self.bounds.iter().rev() {
            let trial: Vec<RankBound> = keep.iter().filter(|x| *x != b).copied().collect();
            if self.orders_under(&trial).len() == 1 {
                keep = trial;
            }
        }
        keep
    }

    /// The order of Frobenius on `Jac[2]` as far as `P` decides it.
    pub fn order(&self) -> TwoTorsionOrder {
        let orders = self.orders();
        if self.factors.iter().all(|f| f.multiplicity == 1) {
            return TwoTorsionOrder::DeterminedByP(self.semisimple_order);
        }
        if orders.len() == 1 {
            TwoTorsionOrder::DeterminedByTwoRank {
                order: orders[0],
                bounds: self.decisive_bounds(),
            }
        } else {
            TwoTorsionOrder::Unresolved {
                semisimple_order: self.semisimple_order,
                divides: self.max_order,
                candidates: orders,
            }
        }
    }

    fn index_of_x_plus_1(&self) -> Option<usize> {
        self.factors.iter().position(|f| f.is_x_plus_1())
    }

    /// The types of `Jac[2]` at the eigenvalue 1 (`lambda_(x+1)`), distinct, in reverse
    /// lexicographic order (`8` before `6 2`); `None` when `x + 1` does not divide `P mod 2`.
    pub fn eigenvalue_one_types(&self) -> Option<Vec<Vec<u32>>> {
        let i = self.index_of_x_plus_1()?;
        let s: BTreeSet<Vec<u32>> = self.candidates.iter().map(|t| t.0[i].clone()).collect();
        Some(s.into_iter().rev().collect())
    }

    /// The types of an `F`-stable Lagrangian on which `F` has characteristic polynomial `w`
    /// (ascending bits, `w w^rev = P mod 2`), by Lemma F and the Littlewood-Richardson
    /// criterion (design C57 2.5), each with the candidates it is compatible with; types
    /// compatible with no candidate are left out. Rests on the hypothesis that such a `W`
    /// exists (for the M23 curves: HJLPPZ Thm. 4.8, `Gal = M23`); the order on `Jac[2]` does not.
    ///
    /// Errors: `NotAReciprocalHalf` if `w w^rev != P mod 2`; `TooLarge` if a multiplicity of
    /// `P mod 2` is above [`MAX_LR_SIZE`] (8; see there for the cost).
    pub fn lagrangian_types(&self, w: &[i64]) -> Result<Vec<LagrangianType>, TwoTorsionError> {
        let w = fp_factor::trim(&w.iter().map(|c| c.rem_euclid(2)).collect::<Vec<_>>());
        let prod = fp_factor::trim(&fp_factor::mul(&w, &reverse_f2(&w), 2));
        if prod != self.p_mod_2 || w.first() != Some(&1) {
            return Err(TwoTorsionError::NotAReciprocalHalf);
        }
        if let Some(f) = self.factors.iter().find(|f| f.multiplicity > MAX_LR_SIZE) {
            return Err(TwoTorsionError::TooLarge(format!(
                "multiplicity {} of {:?} above {MAX_LR_SIZE}",
                f.multiplicity, f.f
            )));
        }
        // The multiplicity of each factor in w.
        let mut m = vec![0u32; self.factors.len()];
        let mut rest = w.clone();
        for (i, f) in self.factors.iter().enumerate() {
            loop {
                let (q, r) = fp_factor::div_mod(&rest, &f.f, 2);
                if !fp_factor::trim(&r).iter().all(|&c| c == 0) || fp_factor::degree(&rest) < 1 {
                    break;
                }
                rest = fp_factor::trim(&q);
                m[i] += 1;
            }
        }
        let lists: Vec<Vec<Vec<u32>>> = m.iter().map(|&mi| partition_list(mi)).collect();
        let mut cache = LrCache::default();
        let mut out = Vec::new();
        for mu in product_of(&lists) {
            let compatible: Vec<usize> = self
                .candidates
                .iter()
                .enumerate()
                .filter(|(_, lam)| lagrangian_compatible(&mut cache, &self.factors, lam, &mu))
                .map(|(i, _)| i)
                .collect();
            if compatible.is_empty() {
                continue;
            }
            let keep: Vec<usize> = (0..self.factors.len()).filter(|&i| m[i] > 0).collect();
            out.push(LagrangianType {
                parts: keep
                    .iter()
                    .map(|&i| (self.factors[i].f.clone(), mu[i].clone()))
                    .collect(),
                compatible,
                orders_of_x: keep.iter().map(|&i| self.factors[i].order_of_x).collect(),
            });
        }
        Ok(out)
    }
}

impl FrobeniusPolynomial {
    /// Newton's identities for the power sums `s_1, ..., s_n` of the roots of `P` (`e_i = 0` for
    /// `i > 2g`), each handed to `keep(m, s_m)`. Only the last `2g` sums are held.
    fn for_each_power_sum(&self, n: usize, mut keep: impl FnMut(usize, &Integer)) {
        let two_g = 2 * self.genus;
        let e: Vec<Integer> = (0..=two_g)
            .map(|i| {
                if i % 2 == 0 {
                    self.coeffs[i].clone()
                } else {
                    -self.coeffs[i].clone()
                }
            })
            .collect();
        // window[window.len() - i] = s_(m - i)
        let mut window: VecDeque<Integer> = VecDeque::with_capacity(two_g + 1);
        for m in 1..=n {
            // s_m = sum_{i=1}^{m-1} (-1)^(i-1) e_i s_(m-i) + (-1)^(m-1) m e_m
            let mut acc = Integer::zero();
            for i in 1..m.min(two_g + 1) {
                let t = e[i].clone() * window[window.len() - i].clone();
                acc = if (i - 1) % 2 == 0 { acc + t } else { acc - t };
            }
            if m <= two_g {
                let t = Integer::from(m as i64) * e[m].clone();
                acc = if (m - 1) % 2 == 0 { acc + t } else { acc - t };
            }
            keep(m, &acc);
            if two_g > 0 {
                if window.len() == two_g {
                    window.pop_front();
                }
                window.push_back(acc);
            }
        }
    }

    /// `s_1, ..., s_n`, the power sums of the roots of `P`.
    pub(super) fn power_sums(&self, n: usize) -> Vec<Integer> {
        let mut s = Vec::with_capacity(n);
        self.for_each_power_sum(n, |_, x| s.push(x.clone()));
        s
    }

    /// `P^(k)` (see [`FrobeniusPolynomial::base_change`]): the power sums `s_k, s_(2k), ..,
    /// s_(2gk)` of the `alpha_i^k`, kept from a pass of Newton's identities that holds only the
    /// last `2g` sums, then the elementary symmetric functions of the `alpha_i^k` by Newton's
    /// identities with exact division.
    pub(super) fn base_change_by_power_sums(&self, k: usize) -> FrobeniusPolynomial {
        let two_g = 2 * self.genus;
        let mut t: Vec<Integer> = Vec::with_capacity(two_g);
        self.for_each_power_sum(two_g * k, |m, x| {
            if m % k == 0 {
                t.push(x.clone());
            }
        });
        let mut e: Vec<Integer> = vec![Integer::one()];
        for m in 1..=two_g {
            let mut acc = Integer::zero();
            for i in 1..=m {
                let x = e[m - i].clone() * t[i - 1].clone();
                acc = if (i - 1) % 2 == 0 { acc + x } else { acc - x };
            }
            let den = Integer::from(m as i64);
            assert!(
                (&acc % &den).is_zero(),
                "base_change: e_{m} is not integral (impossible for an integer P)"
            );
            e.push(acc / den);
        }
        let coeffs = e
            .into_iter()
            .enumerate()
            .map(|(i, x)| if i % 2 == 0 { x } else { -x })
            .collect();
        FrobeniusPolynomial {
            p: self.p.pow(k as u32),
            genus: self.genus,
            coeffs,
        }
    }

    /// `P(1)` (`twisted = false`) or `P(-1)` (`twisted = true`).
    pub(super) fn eval_at_sign(&self, twisted: bool) -> Integer {
        self.coeffs
            .iter()
            .enumerate()
            .fold(Integer::zero(), |acc, (i, c)| {
                if twisted && i % 2 == 1 {
                    acc - c.clone()
                } else {
                    acc + c.clone()
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn data(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/two_torsion")
            .join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// The bracketed lists of a line: `[ 2, 1 ] [ 1 ] [ 1, 1 ] 1`.
    fn bracketed(line: &str) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = line;
        while let Some(a) = rest.find('[') {
            let b = rest[a..].find(']').unwrap() + a;
            out.push(
                rest[a + 1..b]
                    .split(',')
                    .filter(|t| !t.trim().is_empty())
                    .map(|t| t.trim().parse().unwrap())
                    .collect(),
            );
            rest = &rest[b + 1..];
        }
        out
    }

    /// `(lambda, mu, nu)`.
    type Triple = (Vec<u32>, Vec<u32>, Vec<u32>);

    fn check_lr_against_gap(max_n: u32) {
        // GAP 4.12.1, characters of the symmetric groups (tests/data/two_torsion/lr.g,
        // lr_gap.txt): every nonzero c^lam_(mu, nu), |mu|, |nu| >= 1, |lam| <= 8.
        let mut gap: HashMap<Triple, u64> = HashMap::new();
        for line in data("lr_gap.txt").lines() {
            let l = bracketed(line);
            let c: u64 = line.rsplit(' ').next().unwrap().parse().unwrap();
            gap.insert((l[0].clone(), l[1].clone(), l[2].clone()), c);
        }
        assert_eq!(gap.len(), 1197);
        let mut cache = LrCache::default();
        let mut checked = 0;
        let mut nonzero = 0;
        for n in 2..=max_n {
            for a in 1..n {
                for lam in partition_list(n) {
                    for mu in partition_list(a) {
                        for nu in partition_list(n - a) {
                            let want = gap
                                .get(&(lam.clone(), mu.clone(), nu.clone()))
                                .copied()
                                .unwrap_or(0);
                            assert_eq!(
                                cache.coefficient(&lam, &mu, &nu),
                                want,
                                "c^{lam:?}_({mu:?}, {nu:?})"
                            );
                            checked += 1;
                            nonzero += usize::from(want != 0);
                        }
                    }
                }
            }
        }
        if max_n == 8 {
            assert_eq!(nonzero, 1197);
        }
        assert!(checked > 0);
        // The trivial cases.
        assert_eq!(cache.coefficient(&[2, 1], &[2, 1], &[]), 1);
        assert_eq!(cache.coefficient(&[2, 1], &[], &[1, 1, 1]), 0);
        assert_eq!(cache.coefficient(&[], &[], &[]), 1);
        assert_eq!(cache.coefficient(&[3], &[1, 1], &[1]), 0);
    }

    #[test]
    fn lr_coefficients_match_gap() {
        check_lr_against_gap(6);
    }

    #[test]
    #[ignore = "release; every |lambda| <= 8"]
    fn lr_coefficients_match_gap_to_8() {
        check_lr_against_gap(8);
    }

    // ------------------------------------------------------------------ F_2 linear algebra
    // Vectors and matrices over F_2 with at most 32 coordinates: a vector is a bit mask, a
    // matrix the list of the images of the basis vectors (columns).

    fn apply(m: &[u32], v: u32) -> u32 {
        let mut out = 0;
        for (j, c) in m.iter().enumerate() {
            if (v >> j) & 1 == 1 {
                out ^= c;
            }
        }
        out
    }

    fn compose(a: &[u32], b: &[u32]) -> Vec<u32> {
        // a after b
        b.iter().map(|&c| apply(a, c)).collect()
    }

    fn identity(n: usize) -> Vec<u32> {
        (0..n).map(|i| 1u32 << i).collect()
    }

    fn add(a: &[u32], b: &[u32]) -> Vec<u32> {
        a.iter().zip(b).map(|(x, y)| x ^ y).collect()
    }

    /// A row echelon basis of the span (reduced against itself).
    fn echelon(vs: &[u32]) -> Vec<u32> {
        let mut basis: Vec<u32> = Vec::new();
        for &v in vs {
            let mut x = v;
            for &b in &basis {
                x = x.min(x ^ b);
            }
            if x != 0 {
                basis.push(x);
                basis.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        basis
    }

    fn rank(vs: &[u32]) -> usize {
        echelon(vs).len()
    }

    fn in_span(basis: &[u32], v: u32) -> bool {
        let mut x = v;
        for &b in basis {
            x = x.min(x ^ b);
        }
        x == 0
    }

    /// `f(M)` for `f` ascending bits.
    fn poly_of(f: &[i64], m: &[u32]) -> Vec<u32> {
        let n = m.len();
        let mut r = vec![0u32; n];
        for &c in f.iter().rev() {
            r = compose(m, &r);
            if c % 2 == 1 {
                r = add(&r, &identity(n));
            }
        }
        r
    }

    /// `dim ker A` restricted to the subspace with basis `w` (all of `F_2^n` if `w` is None).
    fn kernel_dim(a: &[u32], w: Option<&[u32]>) -> usize {
        match w {
            None => a.len() - rank(a),
            Some(w) => w.len() - rank(&w.iter().map(|&v| apply(a, v)).collect::<Vec<_>>()),
        }
    }

    /// The partition with `dims[j] - dims[j-1]` parts of size `>= j` (`dims[0] = 0`), given the
    /// kernel dimensions divided by `d_f`.
    fn partition_from_kernel_dims(dims: &[usize]) -> Vec<u32> {
        let mut at_least: Vec<usize> = Vec::new();
        for j in 1..dims.len() {
            at_least.push(dims[j] - dims[j - 1]);
        }
        let mut parts = Vec::new();
        for (j, &c) in at_least.iter().enumerate() {
            let next = at_least.get(j + 1).copied().unwrap_or(0);
            for _ in 0..(c - next) {
                parts.push(j as u32 + 1);
            }
        }
        parts.sort_unstable_by(|a, b| b.cmp(a));
        parts
    }

    /// The characteristic polynomial of `m` over `F_2` from the generalized eigenspaces of the
    /// irreducible polynomials of degree `<= n`, and the type of `m` (restricted to the stable
    /// subspace `w` for the second result, if given).
    fn module_type(m: &[u32], w: Option<&[u32]>, factors: &[PrimaryFactor]) -> Vec<Vec<u32>> {
        let n = w.map_or(m.len(), |w| w.len());
        factors
            .iter()
            .map(|f| {
                let d = f.degree();
                let fm = poly_of(&f.f, m);
                let mut pw = identity(m.len());
                let mut dims = vec![0usize];
                for _ in 0..n.div_ceil(d) {
                    pw = compose(&fm, &pw);
                    dims.push(kernel_dim(&pw, w) / d);
                }
                partition_from_kernel_dims(&dims)
            })
            .collect()
    }

    fn irreducibles_up_to(n: usize) -> Vec<Vec<i64>> {
        let mut out = Vec::new();
        for d in 1..=n {
            for bits in 0u32..(1 << d) {
                let mut f: Vec<i64> = (0..d).map(|i| ((bits >> i) & 1) as i64).collect();
                f.push(1);
                let fac = fp_factor::factor_with_multiplicity(&f, 2);
                if fac.len() == 1 && fac[0].1 == 1 {
                    out.push(f);
                }
            }
        }
        out
    }

    /// The characteristic polynomial of `m` (ascending bits), as the product of `f^(e_f)` with
    /// `d_f e_f` the dimension of the generalized eigenspace of `f`.
    fn charpoly(m: &[u32], irr: &[Vec<i64>]) -> Vec<i64> {
        let n = m.len();
        let mut cp = vec![1i64];
        for f in irr {
            let d = f.len() - 1;
            let mut fm = identity(n);
            let fpow = poly_of(f, m);
            for _ in 0..n {
                fm = compose(&fpow, &fm);
            }
            let dim = kernel_dim(&fm, None);
            for _ in 0..dim / d {
                cp = fp_factor::mul(&cp, f, 2);
            }
        }
        fp_factor::trim(&cp)
    }

    /// The classes of `Sp(n, 2)` from GAP (`sp_classes.g`): the form (rows) and the
    /// representatives (rows of the matrix acting on row vectors, as GAP does).
    fn sp_classes(name: &str) -> (Vec<u32>, Vec<Vec<u32>>) {
        let text = data(name);
        let parse_rows = |s: &str| -> Vec<u32> {
            bracketed(s.trim().trim_start_matches('[').trim_end_matches(']'))
                .iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .fold(0u32, |acc, (j, &b)| acc | (b << j))
                })
                .collect()
        };
        let mut form = Vec::new();
        let mut reps = Vec::new();
        for line in text.lines() {
            if let Some(r) = line.strip_prefix("FORM ") {
                form = parse_rows(r);
            } else if let Some(r) = line.strip_prefix("CLASS ") {
                let mut it = r.splitn(3, ' ');
                it.next();
                it.next();
                reps.push(parse_rows(it.next().unwrap()));
            }
        }
        (form, reps)
    }

    /// The bilinear form `x B y^T` with `B` given by rows.
    fn pairing(b: &[u32], x: u32, y: u32) -> u32 {
        // x B: the combination of the rows of B with the coefficients of x
        let xb = apply(b, x);
        (xb & y).count_ones() % 2
    }

    /// Every subspace of `F_2^n` of dimension `k`, by its reduced row echelon basis.
    fn subspaces(n: usize, k: usize) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        // choose pivots p_0 > p_1 > ... (highest bit is the pivot of each row)
        fn rec(n: usize, k: usize, start: usize, pivots: &mut Vec<usize>, out: &mut Vec<Vec<u32>>) {
            if pivots.len() == k {
                // free positions of row i: below its pivot, not a pivot
                let free: Vec<Vec<usize>> = pivots
                    .iter()
                    .map(|&p| (0..p).filter(|j| !pivots.contains(j)).collect())
                    .collect();
                let total: usize = free.iter().map(|f| f.len()).sum();
                for bits in 0u64..(1u64 << total) {
                    let mut b = bits;
                    let rows: Vec<u32> = pivots
                        .iter()
                        .zip(&free)
                        .map(|(&p, fr)| {
                            let mut r = 1u32 << p;
                            for &j in fr {
                                if b & 1 == 1 {
                                    r |= 1 << j;
                                }
                                b >>= 1;
                            }
                            r
                        })
                        .collect();
                    out.push(rows);
                }
                return;
            }
            for p in start..n {
                pivots.push(p);
                rec(n, k, p + 1, pivots, out);
                pivots.pop();
            }
        }
        rec(n, k, 0, &mut Vec::new(), &mut out);
        out
    }

    #[test]
    fn sp_classes_pass_s1_s2_and_their_lagrangians_pass_lr() {
        // Design review 1, m3 (GAP 4.12.1, tests/data/two_torsion/sp_classes.g): for every
        // conjugacy class of Sp(4, 2), Sp(6, 2) and Sp(8, 2), the type of the representative
        // passes S1 and S2, and the type of every stable Lagrangian is accepted by the LR rule
        // of lagrangian_types (Lemma F). A constraint applied too widely would fail here.
        let irr = irreducibles_up_to(8);
        let mut cache = LrCache::default();
        for (name, n, nclasses, nlag) in [
            ("sp4_classes.txt", 4usize, 11usize, 15usize),
            ("sp6_classes.txt", 6, 30, 135),
            ("sp8_classes.txt", 8, 81, 2295),
        ] {
            let (form, reps) = sp_classes(name);
            assert_eq!(reps.len(), nclasses, "{name}");
            let lagrangians: Vec<Vec<u32>> = subspaces(n, n / 2)
                .into_iter()
                .filter(|w| {
                    w.iter()
                        .all(|&x| w.iter().all(|&y| pairing(&form, x, y) == 0))
                })
                .collect();
            assert_eq!(lagrangians.len(), nlag, "{name}");
            let mut stable_pairs = 0;
            for rows in &reps {
                // GAP acts on row vectors: v -> v M. As columns: the image of e_i is row i.
                let m: Vec<u32> = rows.clone();
                // M preserves the form: e(vM, wM) = e(v, w).
                for i in 0..n {
                    for j in 0..n {
                        assert_eq!(
                            pairing(&form, m[i], m[j]),
                            pairing(&form, 1 << i, 1 << j),
                            "{name}: not symplectic"
                        );
                    }
                }
                let cp = charpoly(&m, &irr);
                assert_eq!(cp.len(), n + 1, "{name}: charpoly");
                let factors = primary_factors(&cp).expect("self-reciprocal");
                let lam = ModuleType(module_type(&m, None, &factors));
                for (f, l) in factors.iter().zip(&lam.0) {
                    assert_eq!(l.iter().sum::<u32>(), f.multiplicity, "{name}");
                }
                assert!(symplectic_ok(&factors, &lam), "{name}: {lam:?} fails S1/S2");
                for w in &lagrangians {
                    let wb = echelon(w);
                    if !w.iter().all(|&v| in_span(&wb, apply(&m, v))) {
                        continue;
                    }
                    stable_pairs += 1;
                    let mu = module_type(&m, Some(w), &factors);
                    assert!(
                        lagrangian_compatible(&mut cache, &factors, &lam, &mu),
                        "{name}: J {lam:?}, W {mu:?} rejected"
                    );
                }
            }
            assert!(stable_pairs > 0);
        }
    }

    #[test]
    fn s1_ties_reciprocal_factors() {
        // Design review 1, m3 (b): P mod 2 = (x^3+x+1)^2 (x^3+x^2+1)^2 (genus 6). S1 keeps the
        // types with equal partitions at f and f*; the operator A + A^-T on the hyperbolic space
        // V + V^*, A the companion matrix of f^2, has type ([2], [2]) and is kept.
        let f = vec![1i64, 1, 0, 1];
        let fr = vec![1i64, 0, 1, 1];
        let f2 = fp_factor::mul(&f, &f, 2);
        let pbar = fp_factor::mul(&f2, &fp_factor::mul(&fr, &fr, 2), 2);
        let factors = primary_factors(&pbar).unwrap();
        assert_eq!(factors.len(), 2);
        let (i, j) = if factors[0].f == f { (0, 1) } else { (1, 0) };
        assert_eq!(factors[i].reciprocal, j);
        let ty = |a: Vec<u32>, b: Vec<u32>| {
            let mut t = vec![Vec::new(); 2];
            t[i] = a;
            t[j] = b;
            ModuleType(t)
        };
        assert!(symplectic_ok(&factors, &ty(vec![2], vec![2])));
        assert!(symplectic_ok(&factors, &ty(vec![1, 1], vec![1, 1])));
        assert!(!symplectic_ok(&factors, &ty(vec![2], vec![1, 1])));
        assert!(!symplectic_ok(&factors, &ty(vec![1, 1], vec![2])));
        // A: the companion matrix of f^2 (degree 6) on V = F_2^6.
        let c = fp_factor::trim(&f2);
        let n = 6;
        let a: Vec<u32> = (0..n)
            .map(|k| {
                if k + 1 < n {
                    1u32 << (k + 1)
                } else {
                    (0..n).fold(0u32, |acc, i| acc | ((c[i] as u32 & 1) << i))
                }
            })
            .collect();
        // A^-T on V^*: the inverse of the transpose.
        let at: Vec<u32> = (0..n)
            .map(|i| (0..n).fold(0u32, |acc, j| acc | (((a[j] >> i) & 1) << j)))
            .collect();
        let at_inv = {
            // brute force: the inverse permutes F_2^6
            let mut inv = vec![0u32; n];
            for v in 0u32..(1 << n) {
                let img = apply(&at, v);
                if img.count_ones() == 1 {
                    inv[img.trailing_zeros() as usize] = v;
                }
            }
            inv
        };
        assert_eq!(compose(&at, &at_inv), identity(n));
        let m: Vec<u32> = a
            .iter()
            .copied()
            .chain(at_inv.iter().map(|&v| v << n))
            .collect();
        // the hyperbolic form [[0, I], [I, 0]] is preserved
        let form: Vec<u32> = (0..2 * n)
            .map(|i| {
                if i < n {
                    1u32 << (i + n)
                } else {
                    1u32 << (i - n)
                }
            })
            .collect();
        for x in 0..2 * n {
            for y in 0..2 * n {
                assert_eq!(
                    pairing(&form, apply(&m, 1 << x), apply(&m, 1 << y)),
                    pairing(&form, 1 << x, 1 << y)
                );
            }
        }
        let lam = ModuleType(module_type(&m, None, &factors));
        assert_eq!(lam, ty(vec![2], vec![2]));
        assert!(symplectic_ok(&factors, &lam));
        // V is a stable Lagrangian with characteristic polynomial f^2: compatible.
        let mut cache = LrCache::default();
        let v: Vec<u32> = (0..n).map(|i| 1u32 << i).collect();
        let mu = module_type(&m, Some(&v), &factors);
        assert!(lagrangian_compatible(&mut cache, &factors, &lam, &mu));
    }

    /// For each partition `lambda` of `n` (even): the pairs (type of `U`, type of `V / U`) over
    /// the subspaces `U` of dimension `n / 2` stable under a nilpotent of type `lambda` over
    /// `F_2` equal the pairs with `c^lambda_(mu, nu) != 0` (Green, Klein; design 3.3).
    fn check_lr_criterion(n: usize) {
        let mut cache = LrCache::default();
        let subs = subspaces(n, n / 2);
        for lam in partition_list(n as u32) {
            // N: Jordan blocks, e_i -> e_(i-1) inside a block, the first vector of a block to 0
            let mut nmat = Vec::new();
            let mut start = 0;
            for &b in &lam {
                for i in 0..b as usize {
                    nmat.push(if i == 0 { 0 } else { 1u32 << (start + i - 1) });
                }
                start += b as usize;
            }
            let mut found: BTreeSet<(Vec<u32>, Vec<u32>)> = BTreeSet::new();
            let powers: Vec<Vec<u32>> = (0..=n)
                .scan(identity(n), |acc, _| {
                    let cur = acc.clone();
                    *acc = compose(&nmat, acc);
                    Some(cur)
                })
                .collect();
            for u in &subs {
                let ub = echelon(u);
                if !u.iter().all(|&v| in_span(&ub, apply(&nmat, v))) {
                    continue;
                }
                // type of U: dims of ker N^j on U
                let du: Vec<usize> = (0..=n).map(|j| kernel_dim(&powers[j], Some(u))).collect();
                // type of V/U: dim {v : N^j v in U} - dim U
                //   = dim ker N^j + dim (U cap Im N^j) - dim U
                let dq: Vec<usize> = (0..=n)
                    .map(|j| {
                        let im = echelon(&powers[j]);
                        let sum = rank(&[ub.clone(), im.clone()].concat());
                        let cap = ub.len() + im.len() - sum;
                        kernel_dim(&powers[j], None) + cap - ub.len()
                    })
                    .collect();
                found.insert((
                    partition_from_kernel_dims(&du),
                    partition_from_kernel_dims(&dq),
                ));
            }
            let mut want = BTreeSet::new();
            for mu in partition_list(n as u32 / 2) {
                for nu in partition_list(n as u32 / 2) {
                    if cache.coefficient(&lam, &mu, &nu) != 0 {
                        want.insert((mu.clone(), nu));
                    }
                }
            }
            assert_eq!(found, want, "lambda = {lam:?}");
        }
    }

    #[test]
    fn lr_criterion_by_enumeration() {
        for n in [2, 4, 6] {
            check_lr_criterion(n);
        }
    }

    #[test]
    #[ignore = "release; 200787 subspaces of F_2^8"]
    fn lr_criterion_by_enumeration_8() {
        check_lr_criterion(8);
    }

    #[test]
    fn valuations_modulo_2_128() {
        // C57 review 1, B1: the residues and the valuations of Res(P, x^k -+ 1) modulo 2^128,
        // on polynomials whose resultants are computed by hand in the comments.
        let two = Integer::from(2);
        assert_eq!(residue_mod_2_128(&Integer::from(-1)), u128::MAX);
        assert_eq!(residue_mod_2_128(&(two.pow(128) + Integer::from(5))), 5);
        assert_eq!(residue_mod_2_128(&-two.pow(130)), 0);
        assert_eq!(
            residue_mod_2_128(&-two.pow(64)),
            0u128.wrapping_sub(1u128 << 64)
        );
        assert_eq!(residue_mod_2_128(&Integer::from(12345)), 12345);
        let c = 1u128 + (1u128 << 70);
        // P = x^2 - (1 + 2^70), k = 2: x^2 - 1 = 2^70 mod P, so multiplication by x^2 - 1 is
        // 2^70 times the identity: two elementary divisors 2^70, v = 140 exactly (above 128).
        // x^2 + 1 = 2 + 2^70: v = 2.
        let p = [c.wrapping_neg(), 0, 1];
        assert_eq!(resultant_valuations(&p, 2), [Some(140), Some(2)]);
        // k = 1: Res(P, x - 1) = 1 - c = -2^70 and Res(P, x + 1) = 1 - c as well (v = 70).
        assert_eq!(resultant_valuations(&p, 1), [Some(70), Some(70)]);
        // P = (x - a)(x - b), a = 1 + 2^70, b = 1 + 2^80, k = 1: Res(P, x - 1) = 2^150, but the
        // multiplication by x - 1 has elementary divisors 1 and 2^150 (its matrix has the entry
        // 1), so modulo 2^128 only "at least 128" is known: None. Res(P, x + 1) =
        // (2 + 2^70)(2 + 2^80): v = 2.
        let a = c;
        let b = 1u128 + (1u128 << 80);
        let p = [a.wrapping_mul(b), a.wrapping_add(b).wrapping_neg(), 1];
        assert_eq!(resultant_valuations(&p, 1), [None, Some(2)]);
        // P = (x - 1)(x - 3): Res(P, x^k - 1) = 0, None; Res(P, x + 1) = 2 * 4 = 8,
        // Res(P, x^2 + 1) = 2 * 10 = 20.
        let p = [3u128, 4u128.wrapping_neg(), 1];
        assert_eq!(resultant_valuations(&p, 1), [None, Some(3)]);
        assert_eq!(resultant_valuations(&p, 2), [None, Some(2)]);
        // The elimination on its own: det [[32, 64], [64, 96]] = -1024 (a row operation below
        // a pivot of valuation 5), det [[0, 4], [2, 6]] = -8 (a pivot off the diagonal), a
        // matrix vanishing modulo 2^128, the empty matrix.
        assert_eq!(det_valuation_2(vec![vec![32, 64], vec![64, 96]]), Some(10));
        assert_eq!(det_valuation_2(vec![vec![0, 4], vec![2, 6]]), Some(3));
        assert_eq!(det_valuation_2(vec![vec![0, 0], vec![0, 0]]), None);
        assert_eq!(det_valuation_2(Vec::new()), Some(0));
        for u in [1u128, 3, 5, 12345, u128::MAX, (1u128 << 127) + 1] {
            assert_eq!(u.wrapping_mul(inverse_odd_mod_2_128(u)), 1, "u = {u}");
        }
    }
}
