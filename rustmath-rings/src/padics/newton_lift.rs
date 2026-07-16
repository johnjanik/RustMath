//! Coupled multivariate Newton / Hensel lift of a mod-p seed solution to a
//! bivariate power series over Z_p.
//!
//! Given a square polynomial system `F(z_1, …, z_n) = 0` whose coefficients also
//! depend on two extra *series variables* `u, v`, and a mod-p seed for the
//! constant terms `z_i(0, 0)`, this module lifts the seed to genuine power
//! series `z_i(u, v)` over Z_p with
//!
//! ```text
//!     F(z_1(u,v), …, z_n(u,v))  ≡  0   (mod p^k, (u,v)^order).
//! ```
//!
//! # Home crate (deviation from the handoff)
//!
//! The handoff proposed `PolySystem::newton_lift_bivariate`, an inherent method
//! on [`PolySystem`]. That is architecturally impossible: [`PolySystem`] lives in
//! `rustmath-polynomials`, while [`MPowerSeries`] and [`PadicRational`] live here
//! in `rustmath-rings`, and one cannot add a cross-crate inherent method. Because
//! `rustmath-rings` *depends on* `rustmath-polynomials` (verified in `Cargo.toml`),
//! both `PolySystem` and `MPowerSeries<PadicRational>` are visible here, so this
//! is implemented as the **free function** [`newton_lift_bivariate`] in
//! `rustmath-rings` — no dependency cycle is introduced.
//!
//! # System layout
//!
//! The system is a [`PolySystem`] in `n + 2` variables. The first `n` variables
//! are the unknowns `z_1, …, z_n`; the last two variables (indices `n` and
//! `n+1`) are the series variables `u` and `v`. The Jacobian used by Newton is
//! taken with respect to the unknowns only. The number of equations must equal
//! the number of unknowns `n` (a square system) so the Jacobian is a square
//! matrix; it must be invertible modulo `p` at the seed (rank-full), otherwise
//! the lift returns an honest `Err`.
//!
//! # Algorithm
//!
//! Work in the complete local ring `R = Z_p[[u, v]]` with maximal ideal
//! `m = (p, u, v)`. The residue field is `R/m = F_p`. If `F(seed) ≡ 0 (mod m)`
//! and the mod-p Jacobian at the seed is invertible over `F_p`, Hensel/Newton
//! guarantees a unique root, reached by the quadratically convergent iteration
//!
//! ```text
//!     z_{t+1} = z_t - J(z_t)^{-1} F(z_t),      z_t ≡ root (mod m^{2^t}).
//! ```
//!
//! Each step DOUBLES both the p-adic precision and the (u,v)-truncation order
//! (a single `m`-adic accuracy counter `N`, working over
//! `Z/p^N[[u,v]]/(u,v)^N`). The linear solve `J·δ = -F` over the truncated
//! power-series ring is done *exactly*, degree-by-degree in `(u,v)`:
//!
//! ```text
//!     J_0 δ^{(d)} = -F^{(d)} - Σ_{a=1..d} J^{(a)} δ^{(d-a)},
//! ```
//!
//! where `J^{(a)}` / `δ^{(d)}` are the homogeneous total-degree parts and
//! `J_0 = J^{(0)}` is the (u,v)-constant matrix over `Z/p^N`, inverted once per
//! step by Gauss–Jordan with unit pivots. Running the doubling until
//! `N ≥ p_prec + uv_order - 1` makes every monomial `p^a u^i v^j` with
//! `a < p_prec` and `i+j < uv_order` accurate, so the final truncation to
//! `(p^{p_prec}, (u,v)^{uv_order})` is exact.

use crate::multi_power_series_ring_element::MPowerSeries;
use crate::padics::padic_rational::PadicRational;
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::multivariate::MultivariatePolynomial;
use rustmath_polynomials::poly_system::PolySystem;
use rustmath_rationals::Rational;
use std::collections::BTreeMap;

/// A bivariate power series over `Z/m`: `(u-exp, v-exp) -> coefficient in [0, m)`.
/// Zero coefficients are never stored, so `is_empty()` means the zero series.
type Biv = BTreeMap<(u32, u32), Integer>;

// --- bivariate power-series arithmetic over Z/m, truncated to total degree < order

/// `dst[key] += c` reduced mod `m`, dropping the entry if it becomes zero.
fn add_into(dst: &mut Biv, key: (u32, u32), c: Integer, m: &Integer) {
    let cur = dst.get(&key).cloned().unwrap_or_else(Integer::zero);
    let s = (cur + c).modulo(m);
    if s.is_zero() {
        dst.remove(&key);
    } else {
        dst.insert(key, s);
    }
}

fn series_add(a: &Biv, b: &Biv, m: &Integer) -> Biv {
    let mut r = a.clone();
    for (k, c) in b {
        add_into(&mut r, *k, c.clone(), m);
    }
    r
}

fn series_sub(a: &Biv, b: &Biv, m: &Integer) -> Biv {
    let mut r = a.clone();
    for (k, c) in b {
        add_into(&mut r, *k, Integer::zero() - c.clone(), m);
    }
    r
}

/// Multiply every coefficient by the scalar `s` (mod `m`).
fn series_scalar_mul(a: &Biv, s: &Integer, m: &Integer) -> Biv {
    let mut r = Biv::new();
    let s = s.modulo(m);
    if s.is_zero() {
        return r;
    }
    for (k, c) in a {
        let v = (c.clone() * s.clone()).modulo(m);
        if !v.is_zero() {
            r.insert(*k, v);
        }
    }
    r
}

/// Power-series product truncated to total degree `< order`, coefficients mod `m`.
fn series_mul(a: &Biv, b: &Biv, m: &Integer, order: usize) -> Biv {
    let mut r = Biv::new();
    for ((i1, j1), c1) in a {
        for ((i2, j2), c2) in b {
            let i = i1 + i2;
            let j = j1 + j2;
            if (i as usize + j as usize) >= order {
                continue;
            }
            let v = (c1.clone() * c2.clone()).modulo(m);
            if !v.is_zero() {
                add_into(&mut r, (i, j), v, m);
            }
        }
    }
    r
}

/// The constant series `c` (mod `m`).
fn series_const(c: &Integer, m: &Integer) -> Biv {
    let mut r = Biv::new();
    let v = c.modulo(m);
    if !v.is_zero() {
        r.insert((0, 0), v);
    }
    r
}

/// `a^e` truncated to total degree `< order` (binary exponentiation).
fn series_pow(a: &Biv, e: u32, m: &Integer, order: usize) -> Biv {
    let mut result = series_const(&Integer::one(), m);
    let mut base = a.clone();
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            result = series_mul(&result, &base, m, order);
        }
        e >>= 1;
        if e > 0 {
            base = series_mul(&base, &base, m, order);
        }
    }
    result
}

/// Reduce coefficients mod `m` and drop everything of total degree `>= order`.
fn reduce_deg(a: &Biv, m: &Integer, order: usize) -> Biv {
    let mut r = Biv::new();
    for ((i, j), c) in a {
        if (*i as usize + *j as usize) < order {
            let v = c.modulo(m);
            if !v.is_zero() {
                r.insert((*i, *j), v);
            }
        }
    }
    r
}

/// The homogeneous total-degree-`d` part of a series.
fn homo(a: &Biv, d: usize) -> Biv {
    let mut r = Biv::new();
    for ((i, j), c) in a {
        if (*i as usize + *j as usize) == d {
            r.insert((*i, *j), c.clone());
        }
    }
    r
}

/// Evaluate an integer-coefficient polynomial at a vector of power series.
///
/// `point[var]` supplies the series substituted for variable `var`; the result
/// is reduced mod `m` and truncated to total degree `< order`.
fn eval_poly(
    poly: &MultivariatePolynomial<Integer>,
    point: &[Biv],
    m: &Integer,
    order: usize,
) -> Biv {
    let mut acc = Biv::new();
    for (mono, coeff) in poly.terms() {
        let mut term = series_const(coeff, m);
        if term.is_empty() {
            continue;
        }
        for (&var, &exp) in mono.iter_exponents() {
            if exp > 0 {
                let pw = series_pow(&point[var], exp, m, order);
                term = series_mul(&term, &pw, m, order);
            }
        }
        acc = series_add(&acc, &term, m);
    }
    acc
}

// --- linear algebra over Z/m ------------------------------------------------

/// Inverse of `a` modulo `m` (requires `gcd(a, m) = 1`).
fn mod_inverse(a: &Integer, m: &Integer) -> Result<Integer> {
    let (g, s, _) = a.extended_gcd(m);
    if !g.is_one() {
        return Err(MathError::NotInvertible);
    }
    Ok(s.modulo(m))
}

/// Invert an `n x n` matrix over `Z/m` by Gauss–Jordan, pivoting on entries that
/// are units mod `p` (i.e. coprime to `p`, hence to `m = p^k`).
///
/// Returns an honest `Err` if no unit pivot exists in some column — that is
/// exactly the statement that the (reduced mod p) matrix is rank-deficient.
fn mat_inverse_mod(mat: &[Vec<Integer>], m: &Integer, p: &Integer) -> Result<Vec<Vec<Integer>>> {
    let n = mat.len();
    let mut aug: Vec<Vec<Integer>> = Vec::with_capacity(n);
    for (i, row_in) in mat.iter().enumerate() {
        let mut row: Vec<Integer> = row_in.iter().map(|x| x.modulo(m)).collect();
        for j in 0..n {
            row.push(if i == j { Integer::one() } else { Integer::zero() });
        }
        aug.push(row);
    }

    for col in 0..n {
        // Pivot: a row (at or below `col`) whose column entry is a unit mod p.
        let mut piv = None;
        for r in col..n {
            if !aug[r][col].modulo(p).is_zero() {
                piv = Some(r);
                break;
            }
        }
        let piv = piv.ok_or_else(|| {
            MathError::InvalidArgument(
                "Jacobian is rank-deficient mod p at the seed (not invertible over F_p); \
                 cannot lift"
                    .to_string(),
            )
        })?;
        aug.swap(col, piv);

        let inv = mod_inverse(&aug[col][col], m)?;
        for j in 0..2 * n {
            aug[col][j] = (aug[col][j].clone() * inv.clone()).modulo(m);
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r][col].clone();
            if factor.is_zero() {
                continue;
            }
            for j in 0..2 * n {
                let sub = (factor.clone() * aug[col][j].clone()).modulo(m);
                aug[r][j] = (aug[r][j].clone() + m.clone() - sub).modulo(m);
            }
        }
    }

    Ok(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// Solve `J·δ = -F` exactly over `Z/m[[u,v]]/(u,v)^order`, degree-by-degree.
///
/// `jac[i][k]` are the (power-series) Jacobian entries, `f[i]` the residual
/// series, and `j0inv` the inverse of the (u,v)-constant Jacobian `J_0`.
fn solve_series_system(
    jac: &[Vec<Biv>],
    f: &[Biv],
    j0inv: &[Vec<Integer>],
    m: &Integer,
    order: usize,
    n: usize,
) -> Vec<Biv> {
    let mut delta: Vec<Biv> = vec![Biv::new(); n];
    // delta_homo[d][i] = homogeneous degree-d part of delta[i]
    let mut delta_homo: Vec<Vec<Biv>> = Vec::new();

    for d in 0..order {
        // rhs = -F^{(d)} - Σ_{a=1..d} J^{(a)} δ^{(d-a)}
        let mut rhs: Vec<Biv> = vec![Biv::new(); n];
        for i in 0..n {
            let fd = homo(&f[i], d);
            rhs[i] = series_sub(&rhs[i], &fd, m);
        }
        for a in 1..=d {
            for (i, rhs_i) in rhs.iter_mut().enumerate() {
                for (k, delta_hk) in delta_homo[d - a].iter().enumerate() {
                    if delta_hk.is_empty() {
                        continue;
                    }
                    let ja = homo(&jac[i][k], a);
                    if ja.is_empty() {
                        continue;
                    }
                    let prod = series_mul(&ja, delta_hk, m, order);
                    *rhs_i = series_sub(rhs_i, &prod, m);
                }
            }
        }
        // δ^{(d)} = J_0^{-1} · rhs
        let mut dd: Vec<Biv> = vec![Biv::new(); n];
        for i in 0..n {
            for (k, rhs_k) in rhs.iter().enumerate() {
                let s = &j0inv[i][k];
                if s.is_zero() {
                    continue;
                }
                let contrib = series_scalar_mul(rhs_k, s, m);
                dd[i] = series_add(&dd[i], &contrib, m);
            }
        }
        for i in 0..n {
            delta[i] = series_add(&delta[i], &dd[i], m);
        }
        delta_homo.push(dd);
    }

    delta
}

// --- boundary conversions between PadicRational and Z/m residues ------------

/// Value of a `PadicRational` reduced to an integer residue mod `m`.
///
/// Requires the coefficient to be p-integral (non-negative valuation); a
/// negative valuation is an honest error (the coefficient is not in Z_p).
fn padic_to_int(pr: &PadicRational, m: &Integer, p: &Integer) -> Result<Integer> {
    let v = pr.valuation();
    if v < 0 {
        return Err(MathError::InvalidArgument(
            "coefficient is not p-integral (negative p-adic valuation)".to_string(),
        ));
    }
    let mut val = pr.unit().value().clone();
    for _ in 0..v {
        val = val * p.clone();
    }
    Ok(val.modulo(m))
}

/// Extract the constant term `series(0,0)` as a residue mod `m` (mod p for a
/// seed). A series with no constant term is treated as `0`.
fn constant_residue(series: &MPowerSeries<PadicRational>, m: &Integer, p: &Integer) -> Result<Integer> {
    for (exp, coeff) in series.monomial_coefficients() {
        if exp.iter().all(|&e| e == 0) {
            return padic_to_int(coeff, m, p);
        }
    }
    Ok(Integer::zero())
}

/// Multivariate Hensel/Newton lift of a mod-p seed to a bivariate power series
/// solution over Z_p.
///
/// * `system` — a square [`PolySystem`] in `n + 2` variables: the first `n` are
///   the unknowns `z_1, …, z_n`, the last two (indices `n`, `n+1`) are the
///   series variables `u`, `v`. `system.num_equations()` must equal `n`.
/// * `seed` — one [`MPowerSeries`] per unknown supplying its mod-p constant
///   term `z_i(0,0)` (any higher terms are ignored; the lift recomputes them).
/// * `base` — the prime `p` (the campaign uses `17`).
/// * `uv_order` — target `(u,v)`-truncation order (keep total degree `< uv_order`).
/// * `p_prec` — target p-adic precision `k` (coefficients known mod `p^k`).
///
/// On success the returned series `z_i(u,v)` satisfy
/// `F(z) ≡ 0 (mod p^{p_prec}, (u,v)^{uv_order})`. Returns `Err` if `base` is not
/// prime, the arities are inconsistent, the seed is not a solution mod `p`, or
/// the mod-p Jacobian at the seed is rank-deficient (not invertible over `F_p`).
///
/// Note: `base` is required to be prime because p-adic Newton doubling relies on
/// "unit mod p ⇒ unit mod p^k" and on `Z_p` being a complete DVR.
pub fn newton_lift_bivariate(
    system: &PolySystem,
    seed: &[MPowerSeries<PadicRational>],
    base: i64,
    uv_order: usize,
    p_prec: u32,
) -> Result<Vec<MPowerSeries<PadicRational>>> {
    if base < 2 {
        return Err(MathError::InvalidArgument(format!(
            "base must be a prime >= 2, got {base}"
        )));
    }
    let p = Integer::from(base);
    if !p.is_prime() {
        return Err(MathError::InvalidArgument(format!(
            "base must be prime for a p-adic lift, got {base}"
        )));
    }
    if uv_order == 0 {
        return Err(MathError::InvalidArgument(
            "uv_order must be >= 1".to_string(),
        ));
    }
    if p_prec == 0 {
        return Err(MathError::InvalidArgument(
            "p_prec must be >= 1".to_string(),
        ));
    }

    let n = seed.len();
    if n == 0 {
        return Err(MathError::InvalidArgument(
            "seed must supply at least one unknown".to_string(),
        ));
    }
    if system.num_variables() != n + 2 {
        return Err(MathError::InvalidArgument(format!(
            "system must have n+2 variables (n = seed.len() = {}, plus u,v); got {}",
            n,
            system.num_variables()
        )));
    }
    if system.num_equations() != n {
        return Err(MathError::InvalidArgument(format!(
            "system must be square: {} equations for {} unknowns",
            system.num_equations(),
            n
        )));
    }

    // Seed residues mod p (the constant terms of z_i).
    let mut z: Vec<Biv> = Vec::with_capacity(n);
    for s in seed {
        let r = constant_residue(s, &p, &p)?;
        let mut series = Biv::new();
        if !r.is_zero() {
            series.insert((0, 0), r);
        }
        z.push(series);
    }

    // Verify the seed is a solution modulo p (F(seed) ≡ 0 mod (p, u, v)).
    {
        let order1 = 1usize; // only the (u,v)-constant term survives
        let mut point: Vec<Biv> = z.iter().map(|s| reduce_deg(s, &p, order1)).collect();
        point.push(Biv::new()); // u = 0
        point.push(Biv::new()); // v = 0
        for poly in system.polynomials() {
            let val = eval_poly(poly, &point, &p, order1);
            if !val.is_empty() {
                return Err(MathError::InvalidArgument(
                    "seed is not a solution modulo p (nonzero constant residual)".to_string(),
                ));
            }
        }
    }

    // m-adic accuracy needed so the final (p^{p_prec}, (u,v)^{uv_order})
    // truncation is exact: every monomial p^a u^i v^j with a < p_prec and
    // i+j < uv_order has a+i+j <= (p_prec-1)+(uv_order-1) < target.
    let target = p_prec as usize + uv_order - 1;

    let mut n_acc = 1usize; // z is accurate mod m^{n_acc}
    while n_acc < target {
        let work = (2 * n_acc).min(target);
        let m = p.pow(work as u32);
        let order = work;

        // Evaluation point: lifted unknowns, then the u and v monomial series.
        let mut point: Vec<Biv> = z.iter().map(|s| reduce_deg(s, &m, order)).collect();
        let mut u_series = Biv::new();
        u_series.insert((1, 0), Integer::one());
        let mut v_series = Biv::new();
        v_series.insert((0, 1), Integer::one());
        point.push(u_series);
        point.push(v_series);

        // Residual F(z) and Jacobian J(z) (w.r.t. unknowns only) as series.
        let polys = system.polynomials();
        let mut f: Vec<Biv> = Vec::with_capacity(n);
        let mut jac: Vec<Vec<Biv>> = Vec::with_capacity(n);
        for poly in polys {
            f.push(eval_poly(poly, &point, &m, order));
            let mut row: Vec<Biv> = Vec::with_capacity(n);
            for k in 0..n {
                let d = poly.partial_derivative(k);
                row.push(eval_poly(&d, &point, &m, order));
            }
            jac.push(row);
        }

        // Constant (u,v)-part J_0 of the Jacobian, and its inverse over Z/m.
        let mut j0: Vec<Vec<Integer>> = Vec::with_capacity(n);
        for row in &jac {
            let mut r: Vec<Integer> = Vec::with_capacity(n);
            for entry in row {
                r.push(entry.get(&(0, 0)).cloned().unwrap_or_else(Integer::zero));
            }
            j0.push(r);
        }
        let j0inv = mat_inverse_mod(&j0, &m, &p)?;

        // δ = -J^{-1} F, then z += δ (truncated).
        let delta = solve_series_system(&jac, &f, &j0inv, &m, order, n);
        for i in 0..n {
            z[i] = reduce_deg(&series_add(&z[i], &delta[i], &m), &m, order);
        }

        n_acc = work;
    }

    // Truncate to the requested output ring and pack into MPowerSeries.
    let out_mod = p.pow(p_prec);
    let mut result = Vec::with_capacity(n);
    for zi in &z {
        let mut series = MPowerSeries::<PadicRational>::with_variables(2, uv_order);
        for ((e0, e1), c) in zi {
            if (*e0 as usize + *e1 as usize) >= uv_order {
                continue;
            }
            let cc = c.modulo(&out_mod);
            if cc.is_zero() {
                continue;
            }
            let pr = PadicRational::from_rational(
                Rational::from_integer(cc),
                p.clone(),
                p_prec as usize,
            )?;
            series.set_coefficient(vec![*e0 as usize, *e1 as usize], pr);
        }
        result.push(series);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_polynomials::poly_system::PolySystem;

    // --- independent oracle (no reference to the code under test) -----------

    fn egcd(a: i64, b: i64) -> (i64, i64, i64) {
        if b == 0 {
            (a, 1, 0)
        } else {
            let (g, x, y) = egcd(b, a % b);
            (g, y, x - (a / b) * y)
        }
    }

    fn modinv(a: i64, m: i64) -> i64 {
        let (g, x, _) = egcd(((a % m) + m) % m, m);
        assert_eq!(g, 1, "not invertible");
        ((x % m) + m) % m
    }

    /// Coefficient of `u^i v^j` in `sqrt(1+u+v)`, reduced mod `m`, derived from
    /// the closed form C(1/2, i+j)·C(i+j, i) = [∏_{t<d}(1-2t)] / (2^d · i! · j!).
    /// (Cross-checked against sympy's series expansion mod 17^3.)
    fn sqrt_coeff_mod(i: usize, j: usize, m: i64) -> i64 {
        let d = i + j;
        let mut num: i64 = 1;
        for t in 0..d {
            num *= 1 - 2 * (t as i64);
        }
        let mut den: i64 = 1;
        for _ in 0..d {
            den *= 2;
        }
        for t in 1..=i {
            den *= t as i64;
        }
        for t in 1..=j {
            den *= t as i64;
        }
        let num_m = ((num % m) + m) % m;
        num_m * modinv(((den % m) + m) % m, m) % m
    }

    fn seed_const(value: i64, p: i64, prec: usize) -> MPowerSeries<PadicRational> {
        let mut s = MPowerSeries::<PadicRational>::with_variables(2, 1);
        if value != 0 {
            let pr = PadicRational::from_rational(
                Rational::from_integer(Integer::from(value)),
                Integer::from(p),
                prec,
            )
            .unwrap();
            s.set_coefficient(vec![0, 0], pr);
        }
        s
    }

    #[test]
    fn lifts_sqrt_of_one_plus_u_plus_v() {
        // Unknown z (var 0), series vars u (1), v (2).
        // F(z) = z^2 - 1 - u - v ;  seed z(0,0) = 1 (mod 17).
        // Expected lift z = sqrt(1 + u + v).
        let system = PolySystem::from_terms(
            3,
            &[vec![
                (vec![2, 0, 0], 1), // z^2
                (vec![0, 0, 0], -1),
                (vec![0, 1, 0], -1), // -u
                (vec![0, 0, 1], -1), // -v
            ]],
        );

        let base = 17i64;
        let uv_order = 5usize;
        let p_prec = 3u32;
        let m = 17i64.pow(p_prec); // 4913
        let m_int = Integer::from(m);
        let p_int = Integer::from(base);

        let seed = vec![seed_const(1, base, p_prec as usize)];
        let z = newton_lift_bivariate(&system, &seed, base, uv_order, p_prec).unwrap();
        assert_eq!(z.len(), 1);

        // Compare EVERY coefficient (total degree < uv_order) to the independent
        // closed-form sqrt series reduced mod 17^3.
        for d in 0..uv_order {
            for i in 0..=d {
                let j = d - i;
                let expected = sqrt_coeff_mod(i, j, m);
                let got = z[0]
                    .get_coefficient(&vec![i, j])
                    .map(|pr| padic_to_int(pr, &m_int, &p_int).unwrap().to_i64())
                    .unwrap_or(0);
                assert_eq!(
                    got, expected,
                    "coefficient u^{i} v^{j} mismatch: got {got}, expected {expected}"
                );
            }
        }

        // Independently confirm F(z) ≡ 0 mod (17^3, (u,v)^5) by direct
        // substitution in the power-series ring.
        let mut point: Vec<Biv> = Vec::new();
        let mut zbiv = Biv::new();
        for (exp, pr) in z[0].monomial_coefficients() {
            let c = padic_to_int(pr, &m_int, &p_int).unwrap();
            if !c.is_zero() {
                zbiv.insert((exp[0] as u32, exp[1] as u32), c);
            }
        }
        point.push(zbiv);
        let mut us = Biv::new();
        us.insert((1, 0), Integer::one());
        let mut vs = Biv::new();
        vs.insert((0, 1), Integer::one());
        point.push(us);
        point.push(vs);
        let residual = eval_poly(&system.polynomials()[0], &point, &m_int, uv_order);
        assert!(
            residual.is_empty(),
            "F(z) should vanish mod (17^3,(u,v)^5) but got {residual:?}"
        );
    }

    #[test]
    fn honest_err_on_rank_deficient_jacobian() {
        // F(z) = z^2 - u - v ; seed z(0,0) = 0. Then dF/dz = 2z = 0 mod 17 at
        // the seed, so the 1x1 Jacobian is NOT invertible over F_17. The lift
        // must return an honest Err rather than a fabricated series.
        let system = PolySystem::from_terms(
            3,
            &[vec![
                (vec![2, 0, 0], 1), // z^2
                (vec![0, 1, 0], -1), // -u
                (vec![0, 0, 1], -1), // -v
            ]],
        );
        // seed z = 0 (empty series; never build PadicRational(0), which would
        // loop on the infinite valuation of zero).
        let seed = vec![MPowerSeries::<PadicRational>::with_variables(2, 1)];
        let res = newton_lift_bivariate(&system, &seed, 17, 4, 3);
        assert!(
            res.is_err(),
            "rank-deficient Jacobian must produce Err, got Ok"
        );
    }

    #[test]
    fn honest_err_on_non_prime_base() {
        let system = PolySystem::from_terms(
            3,
            &[vec![(vec![2, 0, 0], 1), (vec![0, 0, 0], -1), (vec![0, 1, 0], -1)]],
        );
        let seed = vec![seed_const(1, 17, 2)];
        assert!(newton_lift_bivariate(&system, &seed, 16, 3, 2).is_err());
    }

    #[test]
    fn honest_err_on_bad_arity() {
        // 2 variables but seed says 1 unknown -> needs 1+2 = 3 variables.
        let system = PolySystem::from_terms(2, &[vec![(vec![1, 0], 1), (vec![0, 1], -1)]]);
        let seed = vec![seed_const(0, 17, 2)];
        assert!(newton_lift_bivariate(&system, &seed, 17, 3, 2).is_err());
    }

    #[test]
    fn honest_err_when_seed_not_a_mod_p_solution() {
        // F(z) = z^2 - 2 - u - v ; seed z = 1 gives F(seed) = 1 - 2 = -1 != 0 mod 17.
        let system = PolySystem::from_terms(
            3,
            &[vec![
                (vec![2, 0, 0], 1),
                (vec![0, 0, 0], -2),
                (vec![0, 1, 0], -1),
                (vec![0, 0, 1], -1),
            ]],
        );
        let seed = vec![seed_const(1, 17, 2)];
        assert!(newton_lift_bivariate(&system, &seed, 17, 3, 2).is_err());
    }

    /// A second, genuinely coupled 2x2 system with a known lift, to exercise the
    /// matrix path (n = 2). Unknowns z (0), w (1); series vars u (2), v (3).
    ///   f1 = z + w - 2 - u        (=> z + w = 2 + u)
    ///   f2 = z - w - v            (=> z - w = v)
    /// Linear, so the unique lift is z = 1 + u/2 + v/2, w = 1 + u/2 - v/2.
    #[test]
    fn lifts_small_linear_two_by_two() {
        let system = PolySystem::from_terms(
            4,
            &[
                vec![
                    (vec![1, 0, 0, 0], 1),  // z
                    (vec![0, 1, 0, 0], 1),  // w
                    (vec![0, 0, 0, 0], -2), // -2
                    (vec![0, 0, 1, 0], -1), // -u
                ],
                vec![
                    (vec![1, 0, 0, 0], 1),  // z
                    (vec![0, 1, 0, 0], -1), // -w
                    (vec![0, 0, 0, 1], -1), // -v
                ],
            ],
        );

        let base = 17i64;
        let p_prec = 3u32;
        let m = 17i64.pow(p_prec);
        let m_int = Integer::from(m);
        let p_int = Integer::from(base);

        let seed = vec![seed_const(1, base, p_prec as usize), seed_const(1, base, p_prec as usize)];
        let sol = newton_lift_bivariate(&system, &seed, base, 3, p_prec).unwrap();
        assert_eq!(sol.len(), 2);

        let half = modinv(2, m); // 1/2 mod 17^3
        let get = |s: &MPowerSeries<PadicRational>, i: usize, j: usize| -> i64 {
            s.get_coefficient(&vec![i, j])
                .map(|pr| padic_to_int(pr, &m_int, &p_int).unwrap().to_i64())
                .unwrap_or(0)
        };
        // z = 1 + u/2 + v/2
        assert_eq!(get(&sol[0], 0, 0), 1);
        assert_eq!(get(&sol[0], 1, 0), half);
        assert_eq!(get(&sol[0], 0, 1), half);
        // w = 1 + u/2 - v/2
        assert_eq!(get(&sol[1], 0, 0), 1);
        assert_eq!(get(&sol[1], 1, 0), half);
        assert_eq!(get(&sol[1], 0, 1), ((m - half) % m));
    }
}
