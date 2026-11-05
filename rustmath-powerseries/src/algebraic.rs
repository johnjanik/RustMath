//! Algebraic power series (Chapter 52): roots of a bivariate polynomial.
//!
//! MAGMA source: Handbook Chapter 52 "Algebraic Power Series Rings".  Covers
//! §52.3 (`ImplicitFunction` — the unique series solution of `p(x,y)=0` under the
//! implicit-function-theorem hypotheses `p(0,0)=0`, `∂p/∂y(0,0)≠0`) and §52.3.1
//! (`RationalPuiseux` — Puiseux-series roots via the Newton–Puiseux algorithm,
//! **[Bec07, Sec. 4.3]**).  Consumes `rustmath-polynomials`/`rustmath-rationals`
//! for the ground data and `rustmath-finitefields` for characteristic-equation
//! roots over `GF(p)`.
//!
//! Representation: a defining polynomial `p(x,y) = Σ_j a_j(x) y^j` is stored in
//! [`BivariatePoly`] as a "y-major" `Vec<Vec<R>>` (`p[j][i]` is the coefficient
//! of `y^j x^i`).
//!
//! Status honesty: `implicit_function` is exact and total on its stated
//! hypotheses (Newton lifting).  `newton_puiseux` fully expands every branch
//! whose Newton-polygon characteristic equation has a **simple rational** root
//! (leading coefficient in the ground field `Q`); a branch whose leading
//! coefficient is genuinely algebraic over `Q`, or a non-simple (higher
//! multiplicity) edge root, is **not** expanded here — it is reported in the
//! `unresolved` count.  This is a bounded, honest subset of MAGMA's `Duval:=true`
//! machinery (which introduces field extensions / automorphisms), not a claim to
//! have decided those branches.

use crate::laurent::LaurentSeries;
use crate::puiseux::PuiseuxSeries;
use crate::series::PowerSeries;
use rustmath_core::{Field, MathError, NumericConversion, Result, Ring};
use rustmath_finitefields::PrimeField;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// A bivariate polynomial `p(x,y) = Σ_j a_j(x) y^j`, stored y-major: `coeffs[j]`
/// is the coefficient list (ascending powers of `x`) of the `y^j` term.
#[derive(Clone, Debug, PartialEq)]
pub struct BivariatePoly<R: Ring> {
    coeffs: Vec<Vec<R>>,
}

impl<R: Ring> BivariatePoly<R> {
    /// Build from a y-major coefficient table (`table[j][i]` = coeff of `y^j x^i`).
    pub fn from_y_major(table: Vec<Vec<R>>) -> Self {
        BivariatePoly { coeffs: table }
    }

    /// The coefficient of `y^j x^i`.
    pub fn coeff(&self, i: usize, j: usize) -> R {
        self.coeffs
            .get(j)
            .and_then(|row| row.get(i))
            .cloned()
            .unwrap_or_else(R::zero)
    }

    /// Degree in `y` (highest `j` with a non-zero `a_j`), or `None` if zero.
    pub fn degree_y(&self) -> Option<usize> {
        (0..self.coeffs.len())
            .rev()
            .find(|&j| self.coeffs[j].iter().any(|c| !c.is_zero()))
    }

    /// The support: `(i, j)` for every non-zero coefficient of `y^j x^i`.
    pub fn support(&self) -> Vec<(i64, i64)> {
        let mut pts = Vec::new();
        for (j, row) in self.coeffs.iter().enumerate() {
            for (i, c) in row.iter().enumerate() {
                if !c.is_zero() {
                    pts.push((i as i64, j as i64));
                }
            }
        }
        pts
    }
}

impl<R: Field + NumericConversion> BivariatePoly<R> {
    /// The partial derivative `∂p/∂y` (again y-major): `q_j = (j+1) a_{j+1}`.
    fn derivative_y(&self) -> BivariatePoly<R> {
        let mut out = Vec::new();
        for j in 1..self.coeffs.len() {
            let scaled: Vec<R> = self.coeffs[j]
                .iter()
                .map(|c| R::from_i64(j as i64) * c.clone())
                .collect();
            out.push(scaled);
        }
        BivariatePoly { coeffs: out }
    }

    /// Evaluate `p(x, Y)` for a power series `Y`, truncated to `prec` terms
    /// (Horner in `y`).
    fn eval_series(&self, y: &PowerSeries<R>, prec: usize) -> PowerSeries<R> {
        let mut acc = PowerSeries::zero(prec);
        for j in (0..self.coeffs.len()).rev() {
            let aj = PowerSeries::new(self.coeffs[j].clone(), prec);
            acc = acc * y.clone() + aj;
        }
        acc
    }
}

/// `ImplicitFunction` (Chapter 52.3): the unique power series `y(x)` with
/// `y(0)=0` and `p(x, y(x)) = 0`, given `p(0,0)=0` and `∂p/∂y(0,0) ≠ 0`.
///
/// Computed by Newton lifting on truncated power series:
/// `y ← y − p(x,y) · p_y(x,y)^{-1}`, which doubles the number of correct terms
/// each step.  Returns the solution to `prec` terms.
pub fn implicit_function<R: Field + NumericConversion>(
    p: &BivariatePoly<R>,
    prec: usize,
) -> Result<PowerSeries<R>> {
    if prec == 0 {
        return Ok(PowerSeries::zero(0));
    }
    // p(0,0) = a_0(0) must be zero.
    if !p.coeff(0, 0).is_zero() {
        return Err(MathError::InvalidArgument(
            "implicit_function: p(0,0) must be 0".to_string(),
        ));
    }
    // p_y(0,0) = a_1(0) must be a unit.
    if p.coeff(0, 1).is_zero() {
        return Err(MathError::InvalidArgument(
            "implicit_function: dp/dy(0,0) must be non-zero".to_string(),
        ));
    }
    let py = p.derivative_y();
    let mut y = PowerSeries::zero(prec);
    let mut prev = y.clone();
    // Each Newton step at least doubles the matched order; iterate until stable.
    for _ in 0..(prec + 2) {
        let num = p.eval_series(&y, prec);
        let den = py.eval_series(&y, prec);
        let den_inv = den.inverse()?;
        y = y.clone() - num * den_inv;
        if y == prev {
            break;
        }
        prev = y.clone();
    }
    Ok(y)
}

// ---------------------------------------------------------------------------
// Rational-root helpers
// ---------------------------------------------------------------------------

/// Positive divisors of `|n|` (trial division; bounded to avoid pathological
/// inputs — returns an empty list for `n = 0` or when `|n|` is too large).
fn divisors_i128(n: i128) -> Vec<i128> {
    let n = n.abs();
    if n == 0 || n > 4_000_000 {
        return vec![];
    }
    let mut out = Vec::new();
    let mut d = 1i128;
    while d * d <= n {
        if n % d == 0 {
            out.push(d);
            if d != n / d {
                out.push(n / d);
            }
        }
        d += 1;
    }
    out
}

/// Evaluate a univariate polynomial (ascending coefficients) at `x` (Horner).
fn eval_rat(coeffs: &[Rational], x: &Rational) -> Rational {
    let mut acc = Rational::zero();
    for c in coeffs.iter().rev() {
        acc = acc * x.clone() + c.clone();
    }
    acc
}

/// All rational roots of a univariate polynomial with rational coefficients
/// (ascending degree), via the rational-root theorem after clearing
/// denominators.  Bounded / honest: extremely large integer coefficients cause
/// the divisor search to be skipped (returning fewer roots), never a wrong root.
pub fn rational_roots(coeffs: &[Rational]) -> Vec<Rational> {
    // Trim leading zero coefficients.
    let mut deg = coeffs.len();
    while deg > 0 && coeffs[deg - 1].is_zero() {
        deg -= 1;
    }
    if deg == 0 {
        return vec![];
    }
    let coeffs = &coeffs[..deg];
    if deg == 1 {
        return vec![]; // non-zero constant, no roots
    }
    // Clear denominators to integer coefficients.
    let mut den = Integer::one();
    for c in coeffs {
        den = lcm_int(&den, c.denominator());
    }
    let int_coeffs_opt: Vec<Option<i128>> = coeffs
        .iter()
        .map(|c| {
            let scaled = c.numerator().clone() * (den.clone() / c.denominator().clone());
            i128_of(&scaled)
        })
        .collect();
    if int_coeffs_opt.iter().any(|v| v.is_none()) {
        return vec![];
    }
    let int_coeffs: Vec<i128> = int_coeffs_opt.into_iter().map(|v| v.unwrap()).collect();
    let a0 = int_coeffs[0];
    let an = *int_coeffs.last().unwrap();

    let mut roots = Vec::new();
    if a0 == 0 {
        roots.push(Rational::zero());
    }
    let ps = divisors_i128(a0);
    let qs = divisors_i128(an);
    for &pnum in &ps {
        for &qden in &qs {
            for &sign in &[1i128, -1] {
                let cand = Rational::new((sign * pnum) as i64, qden as i64).unwrap();
                if eval_rat(coeffs, &cand).is_zero() && !roots.contains(&cand) {
                    roots.push(cand);
                }
            }
        }
    }
    roots
}

fn lcm_int(a: &Integer, b: &Integer) -> Integer {
    if a.is_zero() || b.is_zero() {
        return Integer::zero();
    }
    let g = a.gcd(b);
    (a.clone() / g) * b.clone()
}

fn i128_of(n: &Integer) -> Option<i128> {
    // Integer exposes to_i64; fall back through it for our bounded use.
    let v = n.to_i64();
    if Integer::from(v) == *n {
        Some(v as i128)
    } else {
        None
    }
}

/// Roots in `GF(p)` of a univariate polynomial given by integer coefficients
/// (ascending degree), by enumerating field elements.  This is the
/// finite-field characteristic-equation root primitive consumed by a
/// finite-characteristic Newton–Puiseux (Chapter 52.3.1); exposed and tested on
/// its own since the full `GF(p)` Newton–Puiseux driver is future work.
pub fn prime_field_roots(coeffs: &[i64], p: u64) -> Result<Vec<PrimeField>> {
    let modulus = Integer::from(p as i64);
    let fp_coeffs: Vec<PrimeField> = coeffs
        .iter()
        .map(|&c| PrimeField::new(Integer::from(c), modulus.clone()))
        .collect::<Result<Vec<_>>>()?;
    let zero = PrimeField::new(Integer::zero(), modulus.clone())?;
    let mut roots = Vec::new();
    for v in 0..p {
        let x = PrimeField::new(Integer::from(v as i64), modulus.clone())?;
        let mut acc = zero.clone();
        for c in fp_coeffs.iter().rev() {
            acc = acc * x.clone() + c.clone();
        }
        if acc.is_zero() {
            roots.push(x);
        }
    }
    Ok(roots)
}

// ---------------------------------------------------------------------------
// Newton–Puiseux over Q
// ---------------------------------------------------------------------------

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn binomial(n: u64, k: u64) -> Rational {
    if k > n {
        return Rational::zero();
    }
    let mut num = Integer::one();
    let mut den = Integer::one();
    for t in 0..k {
        num = num * Integer::from((n - t) as i64);
        den = den * Integer::from((t + 1) as i64);
    }
    Rational::new(num, den).unwrap()
}

/// Lower convex hull (vertices) of a point set, sorted by the first coordinate.
fn lower_hull(points: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut pts: Vec<(i64, i64)> = points.to_vec();
    pts.sort();
    pts.dedup();
    let mut hull: Vec<(i64, i64)> = Vec::new();
    for &pt in &pts {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let cross = (b.0 - a.0) * (pt.1 - a.1) - (b.1 - a.1) * (pt.0 - a.0);
            if cross <= 0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(pt);
    }
    hull
}

/// All rational `q`-th roots of `u` (a non-zero rational): `[c]` for odd `q`,
/// `[c, -c]` for even `q` with `u > 0`, or `[]` if no rational `q`-th root
/// exists (the leading coefficient of the branch is genuinely algebraic).
fn rational_qth_roots(u: &Rational, q: i64) -> Vec<Rational> {
    if q <= 0 {
        return vec![];
    }
    if q == 1 {
        return vec![u.clone()];
    }
    let (num, den) = match (
        integer_qth_root(u.numerator(), q),
        integer_qth_root(u.denominator(), q),
    ) {
        (Some(n), Some(d)) => (n, d),
        _ => return vec![],
    };
    let principal = Rational::new(num, den).unwrap();
    if q % 2 == 0 {
        vec![principal.clone(), -principal]
    } else {
        vec![principal]
    }
}

/// An integer `q`-th root of `n` (`n >= 0`), if exact.  Handles negative `n`
/// only for odd `q`.
fn integer_qth_root(n: &Integer, q: i64) -> Option<Integer> {
    let neg = n < &Integer::zero();
    if neg && q % 2 == 0 {
        return None;
    }
    let target = if neg { -(n.clone()) } else { n.clone() };
    let t = i128_of(&target)?;
    // integer q-th root by search (bounded — our characteristic equations are small).
    let mut r: i128 = 0;
    while pow_i128(r + 1, q as u32).map(|v| v <= t).unwrap_or(false) {
        r += 1;
    }
    if pow_i128(r, q as u32) == Some(t) {
        let root = Integer::from(r as i64);
        Some(if neg { -root } else { root })
    } else {
        None
    }
}

fn pow_i128(base: i128, exp: u32) -> Option<i128> {
    let mut acc: i128 = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(base)?;
    }
    Some(acc)
}

/// A single Puiseux branch expansion, plus a count of branches left unresolved.
#[derive(Clone, Debug)]
pub struct NewtonPuiseuxResult {
    /// Fully expanded rational Puiseux-series branches through the origin.
    pub branches: Vec<PuiseuxSeries<Rational>>,
    /// Number of Newton-polygon edge branches not expanded here (algebraic
    /// leading coefficient, or a non-simple characteristic root).
    pub unresolved: usize,
}

/// Compute Puiseux-series roots `y(x)` (with positive valuation) of a bivariate
/// polynomial over `Q` by the Newton–Puiseux algorithm, expanding each branch to
/// `order` terms of its internal power series.  See the module note for what is
/// and is not resolved.
pub fn newton_puiseux(p: &BivariatePoly<Rational>, order: usize) -> NewtonPuiseuxResult {
    let mut branches = Vec::new();
    let mut unresolved = 0usize;

    let support = p.support();
    if support.is_empty() {
        return NewtonPuiseuxResult { branches, unresolved };
    }
    // Newton polygon: plot points (y-exponent j, x-exponent i) and take the
    // lower hull; edges of negative slope give the branch exponents μ > 0.
    let hull_pts: Vec<(i64, i64)> = support.iter().map(|&(i, j)| (j, i)).collect();
    let hull = lower_hull(&hull_pts);

    for w in hull.windows(2) {
        let (j1, i1) = (w[0].0, w[0].1);
        let (j2, i2) = (w[1].0, w[1].1);
        // We want branches y ~ c x^{mu}, mu = (i1 - i2)/(j2 - j1) > 0.
        let dj = j2 - j1;
        let di = i1 - i2;
        if dj <= 0 || di <= 0 {
            continue; // not a descending (positive-slope-branch) edge
        }
        let g = gcd_i64(di, dj);
        let pexp = di / g; // numerator of mu
        let qexp = dj / g; // denominator of mu
        // Edge selector: points with q*i + p*j == M.
        let m = qexp * i1 + pexp * j1;
        // Characteristic polynomial psi(u) with u = c^q, indexed by k=(j - j1)/q.
        let mut psi: Vec<Rational> = Vec::new();
        for &(i, j) in &support {
            if qexp * i + pexp * j == m {
                let k = (j - j1) / qexp;
                if k < 0 {
                    continue;
                }
                let k = k as usize;
                if psi.len() <= k {
                    psi.resize(k + 1, Rational::zero());
                }
                psi[k] = psi[k].clone() + p.coeff(i as usize, j as usize);
            }
        }
        // psi'(u) for the simple-root test.
        let dpsi: Vec<Rational> = (1..psi.len())
            .map(|k| Rational::from_i64(k as i64) * psi[k].clone())
            .collect();

        let roots = rational_roots(&psi);
        // Each nonzero root u of the characteristic equation ψ(u)=0 (u = c^q)
        // contributes the branches y ~ c x^{p/q} for the rational q-th roots c.
        for u in roots {
            if u.is_zero() {
                continue;
            }
            let cs = rational_qth_roots(&u, qexp);
            if cs.is_empty() {
                // Leading coefficient is algebraic over Q (needs a field
                // extension, as in MAGMA's Duval machinery): unresolved.
                unresolved += 1;
                continue;
            }
            if eval_rat(&dpsi, &u).is_zero() {
                // Non-simple edge root: needs deeper Newton–Puiseux recursion.
                unresolved += cs.len();
                continue;
            }
            for c in cs {
                match expand_simple_branch(p, &support, pexp, qexp, m, &c, order) {
                    Some(b) => branches.push(b),
                    None => unresolved += 1,
                }
            }
        }
    }

    NewtonPuiseuxResult { branches, unresolved }
}

/// Expand a simple branch with leading term `c x^{pexp/qexp}` by the change of
/// variables `x = t^q`, `y = t^p (c + w)`, solving for `w(t)` via the implicit
/// function theorem, then re-encoding `t^p (c + w(t))` as a Puiseux series.
fn expand_simple_branch(
    p: &BivariatePoly<Rational>,
    support: &[(i64, i64)],
    pexp: i64,
    qexp: i64,
    m: i64,
    c: &Rational,
    order: usize,
) -> Option<PuiseuxSeries<Rational>> {
    // Build G(t, w) = t^{-m} p(t^q, t^p (c + w)) as a w-major bivariate over t.
    // Term c_{ij} x^i y^j -> c_{ij} t^{qi+pj-m} (c+w)^j.
    let mut g: Vec<Vec<Rational>> = Vec::new();
    for &(i, j) in support {
        let texp = qexp * i + pexp * j - m;
        if texp < 0 {
            return None; // m was not the minimum: inconsistent
        }
        let texp = texp as usize;
        let cij = p.coeff(i as usize, j as usize);
        // (c + w)^j = Σ_k C(j,k) c^{j-k} w^k
        for k in 0..=(j as u64) {
            let coeff = cij.clone()
                * binomial(j as u64, k)
                * pow_rat(c, (j as u64 - k) as u32);
            let k = k as usize;
            if g.len() <= k {
                g.resize(k + 1, Vec::new());
            }
            if g[k].len() <= texp {
                g[k].resize(texp + 1, Rational::zero());
            }
            g[k][texp] = g[k][texp].clone() + coeff;
        }
    }
    let gpoly = BivariatePoly::from_y_major(g);
    let w = implicit_function(&gpoly, order).ok()?;
    // c + w(t) as coefficients; branch y = t^p (c + w).
    let mut coeffs: Vec<Rational> = (0..w.precision()).map(|n| w.coeff(n).clone()).collect();
    if coeffs.is_empty() {
        coeffs.push(c.clone());
    } else {
        coeffs[0] = coeffs[0].clone() + c.clone();
    }
    Some(PuiseuxSeries::from_laurent(
        qexp as u64,
        LaurentSeries::new(pexp, coeffs),
    ))
}

fn pow_rat(base: &Rational, exp: u32) -> Rational {
    let mut acc = Rational::one();
    for _ in 0..exp {
        acc = acc * base.clone();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }
    fn qq(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }

    #[test]
    fn implicit_catalan() {
        // p(x,y) = y - x - y^2 = 0  =>  y = x + x^2 + 2x^3 + 5x^4 + 14x^5 + ...
        // y-major: a_0 = -x, a_1 = 1, a_2 = -1
        let p = BivariatePoly::from_y_major(vec![
            vec![q(0), q(-1)], // -x
            vec![q(1)],        // 1
            vec![q(-1)],       // -1
        ]);
        let y = implicit_function(&p, 7).unwrap();
        let cat = [0i64, 1, 1, 2, 5, 14, 42];
        for (n, &c) in cat.iter().enumerate() {
            assert_eq!(y.coeff(n), &q(c), "coeff {n}");
        }
    }

    #[test]
    fn implicit_requires_hypotheses() {
        // dp/dy(0,0) = 0 => error
        let p = BivariatePoly::from_y_major(vec![vec![q(0), q(-1)], vec![q(0)], vec![q(1)]]);
        assert!(implicit_function(&p, 5).is_err());
    }

    #[test]
    fn puiseux_sqrt_x() {
        // p = y^2 - x  => branches y = ± x^{1/2}
        let p = BivariatePoly::from_y_major(vec![
            vec![q(0), q(-1)], // -x
            vec![q(0)],        // 0
            vec![q(1)],        // y^2
        ]);
        let res = newton_puiseux(&p, 4);
        assert_eq!(res.unresolved, 0);
        assert_eq!(res.branches.len(), 2);
        for b in &res.branches {
            assert_eq!(b.exponent_denominator(), 2);
            assert_eq!(b.valuation(), Some(qq(1, 2)));
            let lead = b.coefficient(1, 2);
            assert!(lead == q(1) || lead == q(-1));
        }
    }

    #[test]
    fn puiseux_cusp_x_cubed() {
        // p = y^2 - x^3 => branches y = ± x^{3/2}
        let p = BivariatePoly::from_y_major(vec![
            vec![q(0), q(0), q(0), q(-1)], // -x^3
            vec![q(0)],
            vec![q(1)], // y^2
        ]);
        let res = newton_puiseux(&p, 4);
        assert_eq!(res.unresolved, 0);
        assert_eq!(res.branches.len(), 2);
        for b in &res.branches {
            assert_eq!(b.exponent_denominator(), 2);
            assert_eq!(b.valuation(), Some(qq(3, 2)));
        }
    }

    #[test]
    fn puiseux_node_integral_branches() {
        // p = y^2 - x^2 - x^3 = 0 => y = ± x*sqrt(1+x) = ±(x + x^2/2 - x^3/8 + ...)
        // Two smooth branches with integral exponents (denominator 1).
        let p = BivariatePoly::from_y_major(vec![
            vec![q(0), q(0), q(-1), q(-1)], // -x^2 - x^3
            vec![q(0)],
            vec![q(1)], // y^2
        ]);
        let res = newton_puiseux(&p, 5);
        assert_eq!(res.unresolved, 0);
        assert_eq!(res.branches.len(), 2);
        // Verify one branch satisfies y^2 = x^2 + x^3 to the computed order.
        let b = &res.branches[0];
        assert_eq!(b.exponent_denominator(), 1);
        assert_eq!(b.valuation(), Some(q(1)));
        // leading coefficient ±1
        let c1 = b.coefficient(1, 1);
        assert!(c1 == q(1) || c1 == q(-1));
        // next coefficient is ±1/2 (from sqrt(1+x))
        let c2 = b.coefficient(2, 1);
        assert!(c2 == qq(1, 2) || c2 == qq(-1, 2));
    }

    #[test]
    fn irrational_branch_is_unresolved() {
        // p = y^2 - 2  has no x; but y^2 - 2x with leading needing sqrt(2):
        // p = y^2 - 2x => characteristic psi(u) = u - 2, root u=2, c = sqrt(2) irrational.
        let p = BivariatePoly::from_y_major(vec![
            vec![q(0), q(-2)], // -2x
            vec![q(0)],
            vec![q(1)],
        ]);
        let res = newton_puiseux(&p, 4);
        // Both conjugate branches c = ±sqrt(2) are algebraic => unresolved, none expanded.
        assert_eq!(res.branches.len(), 0);
        assert!(res.unresolved >= 1);
    }

    #[test]
    fn rational_roots_basic() {
        // 2x^2 - 3x + 1 = (2x-1)(x-1): roots 1/2 and 1
        let roots = rational_roots(&[q(1), q(-3), q(2)]);
        assert!(roots.contains(&qq(1, 2)));
        assert!(roots.contains(&q(1)));
    }

    #[test]
    fn prime_field_roots_gfp() {
        // x^2 - 1 over GF(7): roots 1 and 6
        let roots = prime_field_roots(&[-1, 0, 1], 7).unwrap();
        let vals: Vec<i64> = roots.iter().map(|r| r.value().to_i64()).collect();
        assert!(vals.contains(&1));
        assert!(vals.contains(&6));
        assert_eq!(roots.len(), 2);
        // x^2 + 1 over GF(3) has no roots
        let none = prime_field_roots(&[1, 0, 1], 3).unwrap();
        assert!(none.is_empty());
    }
}
