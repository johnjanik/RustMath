//! Resultant and discriminant of a bivariate polynomial in `X` over the rational
//! function field `ℚ(t)` — and polynomial square roots over `ℚ`.
//!
//! These are the P0 enablers for the Mestre–Vila `A_24` construction: the pencil
//! discriminant `Δ_X(P − T·H) ∈ ℚ[T]` (must be a square for `Gal ⊆ A_m`), the
//! descent discriminant `Δ_X(F(τ,X)) ∈ ℚ(τ)`, and the perfect-square test on the
//! Mestre identity `P'H − PH' = R²` and on `S(T)²`.
//!
//! A bivariate polynomial is `Vec<Vec<Rational>>` with `f[i][j]` = coefficient of
//! `Xⁱ tʲ` (X-major; each `f[i]` is a polynomial in `t`). The resultant in `X` is a
//! polynomial in `t`, computed by **evaluation–interpolation**: evaluate `t` at
//! enough good rational points (where the `X`-leading coefficients don't vanish),
//! take the resultant over `ℚ` at each, and Lagrange-interpolate. This reuses the
//! validated field resultant and avoids rational-function GCD.

use rustmath_rationals::Rational;

fn rzero() -> Rational {
    Rational::from(0i64)
}

// --------------------------------------------------------------------------- //
// Univariate-over-ℚ helpers (polynomials in one variable, Vec<Rational>)
// --------------------------------------------------------------------------- //
fn deg(p: &[Rational]) -> i64 {
    let mut n = p.len();
    while n > 0 && p[n - 1] == rzero() {
        n -= 1;
    }
    n as i64 - 1
}

fn is_zero(p: &[Rational]) -> bool {
    p.iter().all(|c| *c == rzero())
}

fn poly_add(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let n = a.len().max(b.len());
    let mut out = vec![rzero(); n];
    for (i, c) in a.iter().enumerate() {
        out[i] = out[i].clone() + c.clone();
    }
    for (i, c) in b.iter().enumerate() {
        out[i] = out[i].clone() + c.clone();
    }
    out
}

fn poly_scale(a: &[Rational], s: &Rational) -> Vec<Rational> {
    a.iter().map(|c| c.clone() * s.clone()).collect()
}

fn poly_mul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    if is_zero(a) || is_zero(b) {
        return vec![rzero()];
    }
    let mut out = vec![rzero(); a.len() + b.len() - 1];
    for (i, ca) in a.iter().enumerate() {
        if *ca == rzero() {
            continue;
        }
        for (j, cb) in b.iter().enumerate() {
            out[i + j] = out[i + j].clone() + ca.clone() * cb.clone();
        }
    }
    out
}

/// Remainder of `a` by `b` over `ℚ` (`b ≠ 0`).
fn poly_rem(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let db = deg(b);
    let lcb_inv = b[db as usize].reciprocal().expect("nonzero leading coeff");
    let mut r: Vec<Rational> = a.to_vec();
    while deg(&r) >= db && !is_zero(&r) {
        let dr = deg(&r) as usize;
        let coeff = r[dr].clone() * lcb_inv.clone();
        let shift = dr - db as usize;
        for j in 0..b.len() {
            r[j + shift] = r[j + shift].clone() - coeff.clone() * b[j].clone();
        }
        // force the cleared leading term to exact zero
        while r.len() > 1 && *r.last().unwrap() == rzero() {
            r.pop();
        }
    }
    r
}

/// Resultant of `a, b` over `ℚ` via the Euclidean recurrence
/// `Res(a,b) = (−1)^{deg a · deg b} lc(b)^{deg a − deg r} Res(b, r)`.
pub fn resultant_q(a: &[Rational], b: &[Rational]) -> Rational {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    if is_zero(&a) || is_zero(&b) {
        return rzero();
    }
    let mut sign = 1i64;
    let mut res = Rational::from(1i64);
    if deg(&a) < deg(&b) {
        if (deg(&a) * deg(&b)) % 2 == 1 {
            sign = -sign;
        }
        std::mem::swap(&mut a, &mut b);
    }
    loop {
        let db = deg(&b);
        if db < 0 {
            return rzero();
        }
        let da = deg(&a);
        if db == 0 {
            // Res = lc(b)^{deg a}
            let lcb = b[0].clone();
            res = res * pow_q(&lcb, da as u32);
            break;
        }
        let r = poly_rem(&a, &b);
        let dr = deg(&r);
        if dr < 0 {
            return rzero();
        }
        if (da * db) % 2 == 1 {
            sign = -sign;
        }
        let lcb = b[db as usize].clone();
        res = res * pow_q(&lcb, (da - dr) as u32);
        a = b;
        b = r;
    }
    if sign < 0 {
        -res
    } else {
        res
    }
}

fn pow_q(a: &Rational, e: u32) -> Rational {
    let mut acc = Rational::from(1i64);
    for _ in 0..e {
        acc = acc * a.clone();
    }
    acc
}

// --------------------------------------------------------------------------- //
// Bivariate (X over ℚ[t]) — Vec<Vec<Rational>>, f[i] = coeff of X^i (poly in t)
// --------------------------------------------------------------------------- //
fn deg_x(f: &[Vec<Rational>]) -> i64 {
    let mut n = f.len();
    while n > 0 && is_zero(&f[n - 1]) {
        n -= 1;
    }
    n as i64 - 1
}

fn deg_t_max(f: &[Vec<Rational>]) -> i64 {
    f.iter().map(|c| deg(c)).max().unwrap_or(-1)
}

/// Evaluate `t := val`, giving a univariate polynomial in `X` over `ℚ`.
fn eval_t(f: &[Vec<Rational>], val: &Rational) -> Vec<Rational> {
    f.iter()
        .map(|c| {
            // Horner in t
            let mut acc = rzero();
            for coeff in c.iter().rev() {
                acc = acc * val.clone() + coeff.clone();
            }
            acc
        })
        .collect()
}

/// `∂f/∂X`.
pub fn derivative_x(f: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
    if f.len() <= 1 {
        return vec![vec![rzero()]];
    }
    (1..f.len()).map(|i| poly_scale(&f[i], &Rational::from(i as i64))).collect()
}

/// Lagrange interpolation over `ℚ`: the polynomial through `(xs[k], ys[k])`.
fn interpolate(xs: &[Rational], ys: &[Rational]) -> Vec<Rational> {
    let n = xs.len();
    let mut result = vec![rzero()];
    for i in 0..n {
        // basis L_i(t) = Π_{j≠i} (t - xs[j])/(xs[i]-xs[j])
        let mut num = vec![Rational::from(1i64)];
        let mut den = Rational::from(1i64);
        for j in 0..n {
            if j == i {
                continue;
            }
            num = poly_mul(&num, &[-xs[j].clone(), Rational::from(1i64)]);
            den = den * (xs[i].clone() - xs[j].clone());
        }
        let scale = ys[i].clone() * den.reciprocal().expect("distinct nodes");
        result = poly_add(&result, &poly_scale(&num, &scale));
    }
    // trim
    while result.len() > 1 && *result.last().unwrap() == rzero() {
        result.pop();
    }
    result
}

/// Resultant in `X` of two bivariate polynomials, returned as a polynomial in `t`.
pub fn resultant_in_t(f: &[Vec<Rational>], g: &[Vec<Rational>]) -> Vec<Rational> {
    let m = deg_x(f);
    let n = deg_x(g);
    if m < 0 || n < 0 {
        return vec![rzero()];
    }
    // degree bound: deg_t(Res) ≤ n·D_f + m·D_g
    let bound = (n.max(0) * deg_t_max(f).max(0) + m.max(0) * deg_t_max(g).max(0)) as usize;
    let lcf = &f[m as usize];
    let lcg = &g[n as usize];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut node = 0i64;
    while xs.len() <= bound {
        let val = Rational::from(node);
        node += 1;
        // good point: X-leading coefficients stay nonzero (degree preserved)
        if eval_poly_t(lcf, &val) == rzero() || eval_poly_t(lcg, &val) == rzero() {
            continue;
        }
        let fx = eval_t(f, &val);
        let gx = eval_t(g, &val);
        xs.push(val);
        ys.push(resultant_q(&fx, &gx));
    }
    interpolate(&xs, &ys)
}

fn eval_poly_t(c: &[Rational], val: &Rational) -> Rational {
    let mut acc = rzero();
    for coeff in c.iter().rev() {
        acc = acc * val.clone() + coeff.clone();
    }
    acc
}

/// Discriminant in `X`, as a polynomial in `t`:
/// `Δ_X(f) = (−1)^{m(m−1)/2} · Res_X(f, ∂f/∂X) / lc_X(f)`, with the division over
/// `ℚ[t]` exact. Requires `lc_X(f)` constant in `t` (true for the monic-in-`X`
/// Mestre pencils); panics otherwise.
pub fn discriminant_in_t(f: &[Vec<Rational>]) -> Vec<Rational> {
    let m = deg_x(f);
    if m < 1 {
        return vec![rzero()];
    }
    let res = resultant_in_t(f, &derivative_x(f));
    let lc = &f[m as usize];
    assert!(deg(lc) <= 0, "discriminant_in_t: leading X-coefficient must be constant in t");
    let lc0 = lc[0].clone();
    let inv = lc0.reciprocal().expect("nonzero leading coeff");
    let scaled = poly_scale(&res, &inv);
    let mm = m * (m - 1) / 2;
    if mm % 2 == 1 {
        scaled.iter().map(|c| -c.clone()).collect()
    } else {
        scaled
    }
}

// --------------------------------------------------------------------------- //
// Polynomial square root over ℚ (Mestre identity / S(T)²)
// --------------------------------------------------------------------------- //
/// Rational square root, or `None` if `s` is not a square of a rational.
pub fn rational_sqrt(s: &Rational) -> Option<Rational> {
    if *s < rzero() {
        return None;
    }
    if *s == rzero() {
        return Some(rzero());
    }
    let num = s.numerator().clone();
    let den = s.denominator().clone();
    if !num.is_perfect_square() || !den.is_perfect_square() {
        return None;
    }
    let rn = num.sqrt().ok()?;
    let rd = den.sqrt().ok()?;
    Rational::new(rn, rd).ok()
}

/// Polynomial square root over `ℚ`: returns `Some(R)` with `R·R = s`, or `None` if
/// `s` is not a perfect square in `ℚ[x]`. Coefficient recursion from the top.
pub fn poly_sqrt(s: &[Rational]) -> Option<Vec<Rational>> {
    let ds = deg(s);
    if ds < 0 {
        return Some(vec![rzero()]);
    }
    if ds % 2 == 1 {
        return None;
    }
    let k = (ds / 2) as usize;
    let mut r = vec![rzero(); k + 1];
    // leading coefficient
    r[k] = rational_sqrt(&s[2 * k])?;
    let two_rk = Rational::from(2i64) * r[k].clone();
    let inv = two_rk.reciprocal().ok()?;
    for j in (0..k).rev() {
        let d = k + j;
        // (R²)_d = 2·r_k·r_j + Σ_{a=j+1}^{k-1} r_a·r_{d-a}; the loop visits each
        // ordered cross pair (a, d-a) once, so subtract a single product each.
        let mut acc = s[d].clone();
        for a in (j + 1)..k {
            let b = d - a;
            acc = acc - r[a].clone() * r[b].clone();
        }
        r[j] = acc * inv.clone();
    }
    // verify R*R == s exactly
    let sq = poly_mul(&r, &r);
    if poly_eq(&sq, s) {
        Some(r)
    } else {
        None
    }
}

fn poly_eq(a: &[Rational], b: &[Rational]) -> bool {
    let n = a.len().max(b.len());
    for i in 0..n {
        let av = a.get(i).cloned().unwrap_or_else(rzero);
        let bv = b.get(i).cloned().unwrap_or_else(rzero);
        if av != bv {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Rational {
        Rational::from(n)
    }
    fn qs(v: &[i64]) -> Vec<Rational> {
        v.iter().map(|&x| q(x)).collect()
    }

    #[test]
    fn test_resultant_q() {
        // Res(x^2-1, x-2) = (2)^2-1 = 3  (eval of x^2-1 at root 2)
        assert_eq!(resultant_q(&qs(&[-1, 0, 1]), &qs(&[-2, 1])), q(3));
        // common root → 0
        assert_eq!(resultant_q(&qs(&[-1, 0, 1]), &qs(&[-1, 1])), q(0));
    }

    #[test]
    fn test_rational_sqrt() {
        assert_eq!(rational_sqrt(&q(9)), Some(q(3)));
        assert_eq!(rational_sqrt(&Rational::new(4, 9).unwrap()), Some(Rational::new(2, 3).unwrap()));
        assert_eq!(rational_sqrt(&q(8)), None);
        assert_eq!(rational_sqrt(&q(-4)), None);
    }

    #[test]
    fn test_poly_sqrt() {
        // (x+1)^2 = x^2+2x+1
        assert_eq!(poly_sqrt(&qs(&[1, 2, 1])), Some(qs(&[1, 1])));
        // (2x^2-3)^2 = 4x^4-12x^2+9
        assert_eq!(poly_sqrt(&qs(&[9, 0, -12, 0, 4])), Some(qs(&[-3, 0, 2])));
        // not a square
        assert_eq!(poly_sqrt(&qs(&[1, 1, 1])), None);
        assert_eq!(poly_sqrt(&qs(&[1, 2, 3])), None);
    }

    #[test]
    fn test_resultant_in_t_linear_pencil() {
        // f = X^2 - t (coeffs of X: [-t, 0, 1]); g = X - t  ([-t, 1])
        // Res_X(f,g) = g-root t plugged into f = t^2 - t
        let f = vec![qs(&[0, -1]), vec![q(0)], qs(&[1])]; // X^0: -t, X^1: 0, X^2: 1
        let g = vec![qs(&[0, -1]), qs(&[1])]; // X^0: -t, X^1: 1
        let res = resultant_in_t(&f, &g);
        // t^2 - t  → [0, -1, 1]
        assert_eq!(deg(&res), 2);
        assert_eq!(res[0], q(0));
        assert_eq!(res[1], q(-1));
        assert_eq!(res[2], q(1));
    }

    #[test]
    fn test_discriminant_in_t_cubic_vs_gp() {
        // f = X^3 + t·X + 1  → disc_X = -4t^3 - 27 (PARI poldisc)
        let f = vec![qs(&[1]), qs(&[0, 1]), qs(&[0]), qs(&[1])]; // 1 + t·X + X^3
        let d = discriminant_in_t(&f);
        assert_eq!(d[0], q(-27));
        assert_eq!(d.get(1).cloned().unwrap_or(q(0)), q(0));
        assert_eq!(d.get(2).cloned().unwrap_or(q(0)), q(0));
        assert_eq!(d[3], q(-4));
    }

    #[test]
    fn test_discriminant_in_t_quadratic() {
        // f = X^2 + t·X + 1  → disc = t^2 - 4
        let f = vec![qs(&[1]), qs(&[0, 1]), qs(&[1])]; // X^0:1, X^1: t, X^2: 1
        let d = discriminant_in_t(&f);
        assert_eq!(d[0], q(-4));
        assert_eq!(d.get(1).cloned().unwrap_or(q(0)), q(0));
        assert_eq!(d[2], q(1));
    }
}
