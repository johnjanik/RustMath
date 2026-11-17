//! The Frobenius automorphism of a Galois ring as a matrix, Frobenius orbits, traces and orbit
//! polynomials, and the Frobenius-ordered roots of a polynomial that is irreducible mod `p`.
//!
//! MAGMA: ch. 47 (unramified extensions of `Z_p`: `GaloisImage`, `Trace`), ch. 48 (Galois rings
//! `GR(p^a, d)`, `Eltseq`). Digests: `rustmath-lava/docs/Chapter_47_p-adic_Rings_and_their_Extensions.md`,
//! `Chapter_48_Galois_Rings.md`.
//!
//! Mathematics:
//! - B. R. McDonald, *Finite Rings with Identity*, Dekker 1974, ch. XVI, and Z.-X. Wan,
//!   *Lectures on Finite Fields and Galois Rings*, World Scientific 2003, ch. 14: the Frobenius
//!   `phi` of `R = GR(p^a, D)` is the unique automorphism lifting `t -> t^p` on the residue
//!   field; it generates `Gal(R / (Z/p^a))`, cyclic of order `d`; its fixed ring is `Z/p^a`; the
//!   trace is `Tr(t) = sum_{i<d} phi^i(t)`.
//! - J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed., Cambridge 2013, §9.4
//!   (Newton iteration with precision doubling) and §15.4 (Hensel lifting).
//! - The Steiner labelling exercise of the M23 Golay workstream: the paper `golay_m23.tex` of the
//!   owner's M23 repository (`d05475b`), Section 5, and its reference implementation
//!   `m23golay/arith.py` (`frobenius_lift`), ported here as an algorithm. Design notes:
//!   `docs/workstreams/m23_golay/` (design SL, Sections 2.1 and 2.6).
//!
//! Method ([`FrobeniusMatrix::new`]). Let `w` be the generator of `R = GR(p^a, D)`. The image
//! `y = phi(w)` is the unique root of `D` in `R` with `y = w^p mod p` (Hensel: `D` is separable
//! mod `p`, so `D'(y)` is a unit). It is computed by Newton iteration on `D(y) = 0` together with
//! the inverse `u = 1/D'(y)`, doubling the precision `k -> min(2k, a)`: `u <- u (2 - D'(y) u)`,
//! `y <- y - D(y) u`, `u <- u (2 - D'(y) u)`. Then `phi(sum c_j w^j) = sum c_j y^j` because `phi`
//! fixes `Z/p^a`, so `phi` is the `Z/p^a`-linear map whose matrix has column `j` equal to the
//! coefficient vector of `y^j`. Checks, each an `Err` on failure (they would reveal an arithmetic
//! bug, never a property of the input): `D(y) = 0` in `R`, `y = w^p` mod `p`, `phi^d(w) = w`.
//!
//! Traces: `Tr(w^j)` is the power sum `P_j` of the `d` roots `phi^i(w)` of `D` in `R`, by Newton's
//! identities over `Z/p^a` (`P_0 = d`, `P_k = -k D_{d-k} - sum_{j<k} D_{d-j} P_{k-j}`; no
//! division). The `phi^i(w)` are pairwise distinct mod `p`, so `D = prod (X - phi^i(w))` in
//! `R[X]` and the identities hold.
//!
//! Orbit polynomials ([`orbit_polynomial`]): for a family `v_1..v_m` that `phi` permutes, the
//! coefficients of `prod (Z - v_i)` are fixed by `phi` and lie in `Z/p^a`. The function computes
//! the product in `R[Z]` and checks this; a coefficient outside `Z/p^a` is
//! `Err(NotInBaseRing)`, which disproves the caller's claim that the family is stable.

use std::fmt;

use rustmath_core::{Field, MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;

use crate::finite_field::{FiniteField, FiniteFieldElement};
use crate::galois_ring::{mul_reduce, GaloisRing, GaloisRingElement};

// ---------------------------------------------------------------------------
// Coefficient-vector helpers (canonical vectors of length d, entries in [0, m))
// ---------------------------------------------------------------------------

fn vec_add(a: &[Integer], b: &[Integer], m: &Integer) -> Vec<Integer> {
    a.iter().zip(b).map(|(x, y)| (x + y).modulo(m)).collect()
}

fn vec_sub(a: &[Integer], b: &[Integer], m: &Integer) -> Vec<Integer> {
    a.iter().zip(b).map(|(x, y)| (x - y).modulo(m)).collect()
}

fn unit_vector(d: usize, j: usize) -> Vec<Integer> {
    let mut v = vec![Integer::zero(); d];
    v[j] = Integer::one();
    v
}

/// `g(y)` in `(Z/m)[x]/(D)` by Horner; `g` over `Z`, little-endian.
fn horner(g: &[Integer], y: &[Integer], d_mod_m: &[Integer], m: &Integer) -> Vec<Integer> {
    let d = d_mod_m.len() - 1;
    let mut acc = vec![Integer::zero(); d];
    for c in g.iter().rev() {
        acc = mul_reduce(&acc, y, d_mod_m, m);
        acc[0] = (&acc[0] + c).modulo(m);
    }
    acc
}

/// `u (2 - t u)`: one Newton step for the inverse of `t`.
fn inverse_step(u: &[Integer], t: &[Integer], d_mod_m: &[Integer], m: &Integer) -> Vec<Integer> {
    let tu = mul_reduce(t, u, d_mod_m, m);
    let mut two_minus = tu.iter().map(|c| (-c).modulo(m)).collect::<Vec<_>>();
    two_minus[0] = (&two_minus[0] + &Integer::from(2u64)).modulo(m);
    mul_reduce(u, &two_minus, d_mod_m, m)
}

fn reduce_all(v: &[Integer], m: &Integer) -> Vec<Integer> {
    v.iter().map(|c| c.modulo(m)).collect()
}

/// The canonical coefficient vector of `a` in `ring` (an unbound constant is bound first).
/// Panics when `a` lives in another Galois ring (as the ring arithmetic does).
fn coefficients_in(ring: &GaloisRing, a: &GaloisRingElement, who: &str) -> Vec<Integer> {
    if !a.is_bound() {
        return ring.from_int(a.eltseq()[0].clone()).eltseq().to_vec();
    }
    assert!(
        a.ring() == ring,
        "{who}: the element lives in {}, not in {ring}",
        a.ring()
    );
    a.eltseq().to_vec()
}

// ---------------------------------------------------------------------------
// FrobeniusMatrix
// ---------------------------------------------------------------------------

/// The Frobenius `phi` of `R = GR(p^a, D)` through `phi(w)`, `w` the generator: its matrix on the
/// power basis (column `j` is `phi(w)^j`), the orbit `w, phi(w), ..., phi^{d-1}(w)` and the
/// traces `Tr(w^j)`. See the module docs for the method and its checks.
///
/// For `d = 1`, `phi` is the identity of `Z/p^a` (no Newton step).
#[derive(Clone, Debug)]
pub struct FrobeniusMatrix {
    ring: GaloisRing,
    /// `phi^i(w)` for `i < d` (`orbit[0] = w`).
    orbit: Vec<GaloisRingElement>,
    /// Column `j` = coefficient vector of `phi(w)^j`.
    columns: Vec<Vec<Integer>>,
    /// `Tr(w^j)` for `j < d`, in `[0, p^a)`.
    traces: Vec<Integer>,
}

impl FrobeniusMatrix {
    /// Newton lift of `w^p` (mod `p`) to the root `phi(w)` of `D` in `R`, with the checks
    /// `D(phi(w)) = 0`, `phi(w) = w^p` mod `p` and `phi^d(w) = w`.
    ///
    /// # Errors
    ///
    /// `Err(MathError::NumericalError)` when a check fails or `D'(w^p)` is not invertible in the
    /// residue field. Neither can happen for a Galois ring (its `D` is irreducible, hence
    /// separable, mod `p`); an `Err` reveals an arithmetic bug.
    pub fn new(ring: &GaloisRing) -> Result<Self> {
        let residue = ring.residue_field();
        let d = ring.degree();
        if d == 1 {
            return Self::identity(ring);
        }
        let mut x = vec![Integer::zero(); d];
        x[1] = Integer::one();
        let y0 = residue.element(x).frobenius();
        Self::lift(ring, &residue, y0.eltseq().to_vec(), 1, &y0)
    }

    fn identity(ring: &GaloisRing) -> Result<Self> {
        Ok(FrobeniusMatrix {
            ring: ring.clone(),
            orbit: vec![ring.generator()],
            columns: vec![vec![Integer::one()]],
            traces: vec![Integer::one().modulo(ring.characteristic())],
        })
    }

    /// Newton from `y` (correct mod `p^start`, entries in `[0, p^start)`) up to `p^a`, then the
    /// matrix, the orbit, the checks and the traces. `y0` is `w^p` in the residue field.
    fn lift(
        ring: &GaloisRing,
        residue: &FiniteField,
        y_start: Vec<Integer>,
        start: u32,
        y0: &FiniteFieldElement,
    ) -> Result<Self> {
        let p = ring.prime().clone();
        let a = ring.exponent();
        let d = ring.degree();
        let big_d = ring.defining_polynomial().to_vec(); // monic mod p^a
        let big_d_prime: Vec<Integer> = (1..=d)
            .map(|i| &big_d[i] * &Integer::from(i as u64))
            .collect();
        let failed = |what: &str| {
            MathError::NumericalError(format!(
                "FrobeniusMatrix: {what} in {ring} (an arithmetic bug; a Galois ring always passes)"
            ))
        };

        // u = 1/D'(y) in the residue field.
        let dfy0 = {
            let mut acc = residue.zero();
            for c in big_d_prime.iter().rev() {
                acc = acc * y0.clone() + residue.from_int(c.clone());
            }
            acc
        };
        let mut u: Vec<Integer> = dfy0
            .inverse()
            .map_err(|_| failed("D'(w^p) is not invertible mod p"))?
            .eltseq()
            .to_vec();
        let mut y = y_start;
        // Lift u alone to precision p^start (y is fixed there).
        let mut u_prec = 1u32;
        while u_prec < start {
            u_prec = (2 * u_prec).min(start);
            let q = p.pow(u_prec);
            let dq = reduce_all(&big_d, &q);
            let yq = reduce_all(&y, &q);
            let dfy = horner(&big_d_prime, &yq, &dq, &q);
            u = inverse_step(&reduce_all(&u, &q), &dfy, &dq, &q);
        }
        // Joint Newton for y and u.
        let mut prec = start;
        while prec < a {
            prec = (2 * prec).min(a);
            let q = p.pow(prec);
            let dq = reduce_all(&big_d, &q);
            let dfy = horner(&big_d_prime, &y, &dq, &q);
            u = inverse_step(&u, &dfy, &dq, &q);
            let fy = horner(&big_d, &y, &dq, &q);
            let corr = mul_reduce(&fy, &u, &dq, &q);
            y = vec_sub(&y, &corr, &q);
            let dfy = horner(&big_d_prime, &y, &dq, &q);
            u = inverse_step(&u, &dfy, &dq, &q);
        }
        let m = ring.characteristic().clone();
        let y = reduce_all(&y, &m);
        if !horner(&big_d, &y, &big_d, &m).iter().all(|c| c.is_zero()) {
            return Err(failed("D(phi(w)) != 0"));
        }
        if reduce_all(&y, &p) != y0.eltseq() {
            return Err(failed("phi(w) != w^p mod p"));
        }
        let mut columns = Vec::with_capacity(d);
        columns.push(unit_vector(d, 0));
        for j in 1..d {
            let next = mul_reduce(&columns[j - 1], &y, &big_d, &m);
            columns.push(next);
        }
        let mut fm = FrobeniusMatrix {
            ring: ring.clone(),
            orbit: Vec::with_capacity(d),
            columns,
            traces: Vec::new(),
        };
        let w = unit_vector(d, 1);
        let mut cur = w.clone();
        for _ in 0..d {
            fm.orbit.push(ring.canonical_element(cur.clone()));
            cur = fm.apply_vec(&cur);
        }
        if cur != w {
            return Err(failed("phi^d(w) != w"));
        }
        fm.traces = power_sums(&big_d, &m);
        Ok(fm)
    }

    /// The same Frobenius in `GR(p^b, D)` for `1 <= b <= a`: the projection (the canonical
    /// surjection commutes with `phi`). Builds the ring `GR(p^b, D)` (one irreducibility proof).
    /// To raise the precision use [`FrobeniusMatrix::lift_to`]: `D mod p^a` does not determine
    /// `D mod p^b` for `b > a`.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` for `b = 0` or `b > a`.
    pub fn with_precision(&self, b: u32) -> Result<Self> {
        let a = self.ring.exponent();
        if b == 0 || b > a {
            return Err(MathError::InvalidArgument(format!(
                "FrobeniusMatrix::with_precision: b = {b} is not in 1..={a} (use lift_to to raise \
                 the precision)"
            )));
        }
        if b == a {
            return Ok(self.clone());
        }
        let target = self.ring.reduce_precision(b)?;
        let m = target.characteristic().clone();
        Ok(FrobeniusMatrix {
            orbit: self
                .orbit
                .iter()
                .map(|e| target.canonical_element(reduce_all(e.eltseq(), &m)))
                .collect(),
            columns: self.columns.iter().map(|c| reduce_all(c, &m)).collect(),
            traces: reduce_all(&self.traces, &m),
            ring: target,
        })
    }

    /// The Frobenius of `target = GR(p^b, D')` with `b >= a` and `D' = D` mod `p^a`, by Newton
    /// continued from the current `phi(w)` (no new `w^p`, half the Newton steps of
    /// [`FrobeniusMatrix::new`]). The result projects to `self` (Hensel uniqueness).
    ///
    /// # Errors
    ///
    /// `InvalidArgument` when `target` has another prime, a lower exponent, or a defining
    /// polynomial that does not reduce to `D` mod `p^a`; a failed check of
    /// [`FrobeniusMatrix::new`].
    pub fn lift_to(&self, target: &GaloisRing) -> Result<Self> {
        let a = self.ring.exponent();
        let p = self.ring.prime().clone();
        let pa = self.ring.characteristic();
        if target.prime() != &p
            || target.exponent() < a
            || reduce_all(target.defining_polynomial(), pa) != self.ring.defining_polynomial()
        {
            return Err(MathError::InvalidArgument(format!(
                "FrobeniusMatrix::lift_to: {target} is not a lift of {}",
                self.ring
            )));
        }
        if target.exponent() == a {
            return Ok(self.clone());
        }
        if self.ring.degree() == 1 {
            return Self::identity(target);
        }
        let residue = target.residue_field();
        let y = self.image_of_generator().eltseq().to_vec();
        let y0 = residue.element(reduce_all(&y, &p));
        Self::lift(target, &residue, y, a, &y0)
    }

    /// The ring `R`.
    pub fn ring(&self) -> &GaloisRing {
        &self.ring
    }

    /// `phi(w)`.
    pub fn image_of_generator(&self) -> &GaloisRingElement {
        &self.orbit[1 % self.orbit.len()]
    }

    /// The orbit `w, phi(w), ..., phi^{d-1}(w)` of the generator (computed once).
    pub fn generator_orbit(&self) -> &[GaloisRingElement] {
        &self.orbit
    }

    fn apply_vec(&self, v: &[Integer]) -> Vec<Integer> {
        let d = self.columns.len();
        let m = self.ring.characteristic();
        let mut out = vec![Integer::zero(); d];
        for (vj, col) in v.iter().zip(&self.columns) {
            if vj.is_zero() {
                continue;
            }
            for (o, c) in out.iter_mut().zip(col) {
                if !c.is_zero() {
                    *o = &*o + &(vj * c);
                }
            }
        }
        reduce_all(&out, m)
    }

    /// `phi(a)`: one matrix-vector product (`d^2` products of integers below `p^a`). An unbound
    /// constant is bound into `R` first. Panics when `a` lives in another Galois ring.
    pub fn apply(&self, a: &GaloisRingElement) -> GaloisRingElement {
        let v = coefficients_in(&self.ring, a, "FrobeniusMatrix::apply");
        self.ring.canonical_element(self.apply_vec(&v))
    }

    /// `a, phi(a), ..., phi^{d-1}(a)`.
    pub fn orbit(&self, a: &GaloisRingElement) -> Vec<GaloisRingElement> {
        let d = self.columns.len();
        let mut v = coefficients_in(&self.ring, a, "FrobeniusMatrix::orbit");
        let mut out = Vec::with_capacity(d);
        for i in 0..d {
            if i > 0 {
                v = self.apply_vec(&v);
            }
            out.push(self.ring.canonical_element(v.clone()));
        }
        out
    }

    /// The traces `Tr(w^j)`, `j < d`, in `[0, p^a)` (the power sums of the roots of `D`).
    pub fn generator_power_traces(&self) -> &[Integer] {
        &self.traces
    }

    /// `Tr(a) = sum_{i<d} phi^i(a)`, in `[0, p^a)`: `sum_j a_j Tr(w^j)`.
    pub fn trace(&self, a: &GaloisRingElement) -> Integer {
        let v = coefficients_in(&self.ring, a, "FrobeniusMatrix::trace");
        trace_of_vec(&v, &self.traces, self.ring.characteristic())
    }

    /// `prod_{i<d} (Z - phi^i(a))`, ascending, monic, coefficients in `[0, p^a)` (the
    /// characteristic polynomial of `a` over `Z/p^a`; its minimal polynomial when the `d`
    /// conjugates are distinct).
    ///
    /// # Errors
    ///
    /// `Err(MathError::NumericalError)` when a coefficient is not in `Z/p^a`. The orbit is
    /// stable under `phi` by construction, so this reveals an arithmetic bug.
    pub fn orbit_polynomial(&self, a: &GaloisRingElement) -> Result<Vec<Integer>> {
        orbit_polynomial(&self.orbit(a)).map_err(|e| {
            MathError::NumericalError(format!(
                "FrobeniusMatrix::orbit_polynomial: {e} for a Frobenius orbit (an arithmetic bug)"
            ))
        })
    }
}

/// `sum_j v_j t_j mod m`.
pub(crate) fn trace_of_vec(v: &[Integer], traces: &[Integer], m: &Integer) -> Integer {
    let mut s = Integer::zero();
    for (vj, tj) in v.iter().zip(traces) {
        if !vj.is_zero() {
            s = &s + &(vj * tj);
        }
    }
    s.modulo(m)
}

/// Power sums `P_0..P_{d-1}` of the roots of the monic `D` (degree `d`) by Newton's identities,
/// reduced mod `m`. No division, so every `m` works.
fn power_sums(big_d: &[Integer], m: &Integer) -> Vec<Integer> {
    let d = big_d.len() - 1;
    let mut pw: Vec<Integer> = vec![Integer::from(d as u64).modulo(m)];
    for k in 1..d {
        let mut s = -(&Integer::from(k as u64) * &big_d[d - k]);
        for j in 1..k {
            s = &s - &(&big_d[d - j] * &pw[k - j]);
        }
        pw.push(s.modulo(m));
    }
    pw
}

// ---------------------------------------------------------------------------
// Orbit polynomials
// ---------------------------------------------------------------------------

/// A coefficient of an orbit polynomial that is not in the base ring `Z/p^a`: the family given
/// to [`orbit_polynomial`] is not stable under the Frobenius.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotInBaseRing {
    /// The degree (in `Z`) of the first coefficient outside `Z/p^a`.
    pub index: usize,
}

impl fmt::Display for NotInBaseRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the coefficient of Z^{} of the orbit polynomial is not in the base ring Z/p^a",
            self.index
        )
    }
}

impl std::error::Error for NotInBaseRing {}

/// `prod (Z - v)` over the given elements, ascending and monic, with coefficients in `[0, p^a)`,
/// for a family the caller claims is permuted by the Frobenius (then every coefficient lies in
/// the fixed ring `Z/p^a`). `[1]` for an empty family.
///
/// Cost: `m (m + 1) / 2` ring products for `m` values.
///
/// # Errors
///
/// `Err(NotInBaseRing { index })` when the coefficient of `Z^index` is not in `Z/p^a`, which
/// disproves the claim. Never a guess.
///
/// # Panics
///
/// When the values do not all live in one Galois ring (unbound constants are bound into the
/// ring of the first bound value; a family of unbound constants only panics).
pub fn orbit_polynomial(
    values: &[GaloisRingElement],
) -> std::result::Result<Vec<Integer>, NotInBaseRing> {
    if values.is_empty() {
        return Ok(vec![Integer::one()]);
    }
    let ring = values
        .iter()
        .find(|v| v.is_bound())
        .expect("orbit_polynomial: every value is an unbound constant (no Galois ring)")
        .ring()
        .clone();
    let d = ring.degree();
    let m = ring.characteristic().clone();
    let big_d = ring.defining_polynomial();
    let mut poly: Vec<Vec<Integer>> = vec![unit_vector(d, 0)];
    for v in values {
        let v = coefficients_in(&ring, v, "orbit_polynomial");
        let mut next = vec![vec![Integer::zero(); d]; poly.len() + 1];
        for (k, c) in poly.iter().enumerate() {
            next[k + 1] = vec_add(&next[k + 1], c, &m);
            let cv = mul_reduce(c, &v, big_d, &m);
            next[k] = vec_sub(&next[k], &cv, &m);
        }
        poly = next;
    }
    poly.into_iter()
        .enumerate()
        .map(|(index, c)| {
            if c[1..].iter().all(|x| x.is_zero()) {
                Ok(c[0].clone())
            } else {
                Err(NotInBaseRing { index })
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// FrobeniusRoots
// ---------------------------------------------------------------------------

/// Why [`FrobeniusRoots::new`] refused its input, or (`CheckFailed`) a failed internal check.
#[derive(Clone, Debug, PartialEq)]
pub enum FrobeniusRootsError {
    /// `f` is empty or its leading coefficient is not `1`.
    NotMonic,
    /// `deg f < 2`.
    DegreeTooSmall(usize),
    /// Precision `N = 0`.
    ZeroPrecision,
    /// `p` is not prime, or its primality proof is unresolved (`try_is_prime`).
    Prime(MathError),
    /// `f mod p` has a repeated factor (`p` divides `disc f`).
    NotSquarefreeModP,
    /// `f mod p` is squarefree but reducible: `p` is not inert. The factor degrees (sorted) are
    /// given when `p <= fp_factor::MAX_P`.
    NotInertModP {
        /// Degrees of the irreducible factors of `f mod p`, when computed.
        factor_degrees: Option<Vec<usize>>,
    },
    /// `f(phi(x)) != 0`, `phi(x) != x^p mod p` or `phi^n != id`: an arithmetic bug, never a
    /// property of the input.
    CheckFailed(String),
}

impl fmt::Display for FrobeniusRootsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMonic => write!(f, "f is not monic"),
            Self::DegreeTooSmall(n) => write!(f, "deg f = {n} < 2"),
            Self::ZeroPrecision => write!(f, "precision N = 0"),
            Self::Prime(e) => write!(f, "p is not a proven prime: {e}"),
            Self::NotSquarefreeModP => write!(f, "f mod p is not squarefree"),
            Self::NotInertModP {
                factor_degrees: Some(d),
            } => write!(f, "f mod p is reducible, factor degrees {d:?}"),
            Self::NotInertModP {
                factor_degrees: None,
            } => write!(f, "f mod p is reducible"),
            Self::CheckFailed(why) => write!(f, "internal check failed: {why}"),
        }
    }
}

impl std::error::Error for FrobeniusRootsError {}

/// The roots `alpha_i = phi^i(x)`, `i < n`, of a monic `f` in `Z[x]` of degree `n >= 2` that is
/// irreducible mod the prime `p`, in `R = GR(p^N, f) = (Z/p^N)[x]/(f)`.
///
/// `R` is the reduction mod `p^N` of the ring of integers of the unramified extension of `Q_p` of
/// degree `n`, and `f` splits there. The `alpha_i` are the reductions of the `n` distinct `p`-adic
/// roots, ordered by the Frobenius: `phi(alpha_i) = alpha_{i+1}` (indices mod `n`),
/// `alpha_0 = x` (MAGMA ch. 38 `GaloisRoot` with the Frobenius order of ch. 47 `GaloisImage`).
#[derive(Clone, Debug)]
pub struct FrobeniusRoots {
    f: Vec<Integer>,
    p: Integer,
    precision: u32,
    frobenius: FrobeniusMatrix,
}

impl FrobeniusRoots {
    /// The Frobenius-ordered roots of `f` mod `p^precision` (module docs, method).
    ///
    /// Refusals, in this order and before any lifting: `NotMonic`; `DegreeTooSmall`;
    /// `ZeroPrecision`; `Prime` (by `try_is_prime`); `NotSquarefreeModP`; `NotInertModP`.
    /// The squarefree and irreducibility tests are exact: `fp_factor::factor_with_multiplicity`
    /// for `p <= fp_factor::MAX_P`, a gcd with `f'` and Rabin's test over `F_p` above it.
    ///
    /// # Errors
    ///
    /// The refusals above, and `CheckFailed` for a failed internal check (a bug).
    pub fn new(
        f: &[Integer],
        p: &Integer,
        precision: u32,
    ) -> std::result::Result<Self, FrobeniusRootsError> {
        if f.last().map(|c| !c.is_one()).unwrap_or(true) {
            return Err(FrobeniusRootsError::NotMonic);
        }
        let n = f.len() - 1;
        if n < 2 {
            return Err(FrobeniusRootsError::DegreeTooSmall(n));
        }
        if precision == 0 {
            return Err(FrobeniusRootsError::ZeroPrecision);
        }
        match rustmath_integers::prime::try_is_prime(p) {
            Ok(true) => {}
            Ok(false) => {
                return Err(FrobeniusRootsError::Prime(MathError::InvalidArgument(
                    format!("{p} is not prime"),
                )))
            }
            Err(e) => return Err(FrobeniusRootsError::Prime(e)),
        }
        check_inert(f, p)?;
        let ring = GaloisRing::with_modulus(p.clone(), precision, f.to_vec()).map_err(|e| {
            FrobeniusRootsError::CheckFailed(format!("GaloisRing::with_modulus: {e}"))
        })?;
        let frobenius = FrobeniusMatrix::new(&ring)
            .map_err(|e| FrobeniusRootsError::CheckFailed(e.to_string()))?;
        Ok(FrobeniusRoots {
            f: f.to_vec(),
            p: p.clone(),
            precision,
            frobenius,
        })
    }

    /// The same roots at another precision: the projection when lower, Newton continued from
    /// the current `phi(x)` when higher (no new `x^p`; one irreducibility proof for the new
    /// ring). The roots at the new precision reduce to the current ones (Hensel uniqueness).
    ///
    /// # Errors
    ///
    /// `ZeroPrecision`; `CheckFailed` for a failed internal check.
    pub fn with_precision(&self, precision: u32) -> std::result::Result<Self, FrobeniusRootsError> {
        if precision == 0 {
            return Err(FrobeniusRootsError::ZeroPrecision);
        }
        let check = |e: MathError| FrobeniusRootsError::CheckFailed(e.to_string());
        let frobenius = if precision <= self.precision {
            self.frobenius.with_precision(precision).map_err(check)?
        } else {
            let target = GaloisRing::with_modulus(self.p.clone(), precision, self.f.clone())
                .map_err(check)?;
            self.frobenius.lift_to(&target).map_err(check)?
        };
        Ok(FrobeniusRoots {
            f: self.f.clone(),
            p: self.p.clone(),
            precision,
            frobenius,
        })
    }

    /// `f`, little-endian over `Z`.
    pub fn polynomial(&self) -> &[Integer] {
        &self.f
    }

    /// The prime `p`.
    pub fn prime(&self) -> &Integer {
        &self.p
    }

    /// The precision `N`.
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// `p^N`.
    pub fn modulus(&self) -> &Integer {
        self.frobenius.ring().characteristic()
    }

    /// `R = GR(p^N, f)`.
    pub fn ring(&self) -> &GaloisRing {
        self.frobenius.ring()
    }

    /// The Frobenius of `R`.
    pub fn frobenius(&self) -> &FrobeniusMatrix {
        &self.frobenius
    }

    /// `alpha_0, ..., alpha_{n-1}` with `alpha_0 = x` and `alpha_{i+1} = phi(alpha_i)`.
    pub fn roots(&self) -> &[GaloisRingElement] {
        self.frobenius.generator_orbit()
    }
}

/// The refusals `NotSquarefreeModP` and `NotInertModP` (exact), for a proven prime `p`.
fn check_inert(f: &[Integer], p: &Integer) -> std::result::Result<(), FrobeniusRootsError> {
    let small = p.to_i64().filter(|&q| q <= fp_factor::MAX_P);
    if let Some(q) = small {
        let pi = Integer::from(q);
        let fq: Vec<i64> = f
            .iter()
            .map(|c| c.modulo(&pi).to_i64().expect("residue below MAX_P"))
            .collect();
        let factors = fp_factor::factor_with_multiplicity(&fq, q);
        if factors.iter().any(|(_, e)| *e > 1) {
            return Err(FrobeniusRootsError::NotSquarefreeModP);
        }
        if factors.len() > 1 {
            let mut degrees: Vec<usize> = factors
                .iter()
                .map(|(g, _)| fp_factor::trim(g).len() - 1)
                .collect();
            degrees.sort_unstable();
            return Err(FrobeniusRootsError::NotInertModP {
                factor_degrees: Some(degrees),
            });
        }
        return Ok(());
    }
    if !crate::poly_factor::is_squarefree_fp_unchecked(f, p) {
        return Err(FrobeniusRootsError::NotSquarefreeModP);
    }
    let fp: Vec<Integer> = f.iter().map(|c| c.modulo(p)).collect();
    if !crate::poly_factor::is_irreducible_fp_unchecked(&fp, p) {
        return Err(FrobeniusRootsError::NotInertModP {
            factor_degrees: None,
        });
    }
    Ok(())
}
