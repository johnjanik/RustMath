//! Mestre–Vila construction operators over `ℚ` (the `A_n` square-discriminant
//! engine). Given a separable odd-degree `P` with square discriminant and a Mestre
//! auxiliary polynomial `H` solving the identity `P'H − PH' = R²`, these build and
//! verify the regular `A_n`-family and its degree-reduction descent.
//!
//! Per the construction plan (lab Entry 26), finding `H` (Step 3, the nonlinear
//! root-square system) is left to an external solver (Magma/Sage are suggested);
//! everything else — the identity verifier, the pencil-discriminant square test,
//! the descent `Φ(τ,X)`, and specialization — is native and exact here.
//!
//! Polynomials over `ℚ` are `Vec<Rational>` (ascending); bivariate results are
//! `Vec<Vec<Rational>>` (X-major, each entry a polynomial in the parameter).

use crate::bivariate::{discriminant_in_t, poly_sqrt};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

fn rz() -> Rational {
    Rational::from(0i64)
}

fn deg(p: &[Rational]) -> i64 {
    let mut n = p.len();
    while n > 0 && p[n - 1] == rz() {
        n -= 1;
    }
    n as i64 - 1
}

fn padd(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let n = a.len().max(b.len());
    let mut o = vec![rz(); n];
    for (i, c) in a.iter().enumerate() {
        o[i] = o[i].clone() + c.clone();
    }
    for (i, c) in b.iter().enumerate() {
        o[i] = o[i].clone() + c.clone();
    }
    o
}

fn psub(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let n = a.len().max(b.len());
    let mut o = vec![rz(); n];
    for (i, c) in a.iter().enumerate() {
        o[i] = o[i].clone() + c.clone();
    }
    for (i, c) in b.iter().enumerate() {
        o[i] = o[i].clone() - c.clone();
    }
    o
}

fn pscale(a: &[Rational], s: &Rational) -> Vec<Rational> {
    a.iter().map(|c| c.clone() * s.clone()).collect()
}

fn pmul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    if deg(a) < 0 || deg(b) < 0 {
        return vec![rz()];
    }
    let mut o = vec![rz(); a.len() + b.len() - 1];
    for (i, ca) in a.iter().enumerate() {
        if *ca == rz() {
            continue;
        }
        for (j, cb) in b.iter().enumerate() {
            o[i + j] = o[i + j].clone() + ca.clone() * cb.clone();
        }
    }
    o
}

/// `dp/dX`.
pub fn derivative(p: &[Rational]) -> Vec<Rational> {
    if p.len() <= 1 {
        return vec![rz()];
    }
    (1..p.len()).map(|i| p[i].clone() * Rational::from(i as i64)).collect()
}

/// The Mestre Wronskian `W(P,H) = P'·H − P·H'`.
pub fn wronskian(p: &[Rational], h: &[Rational]) -> Vec<Rational> {
    psub(&pmul(&derivative(p), h), &pmul(p, &derivative(h)))
}

/// Verify the Mestre identity `P'H − PH' = R²`: returns `Some(R)` if the Wronskian
/// is a perfect square in `ℚ[X]`, else `None`.
pub fn verify_identity(p: &[Rational], h: &[Rational]) -> Option<Vec<Rational>> {
    poly_sqrt(&wronskian(p, h))
}

/// Discriminant of the Mestre pencil `P − T·H`, as a polynomial in `T`.
pub fn pencil_discriminant(p: &[Rational], h: &[Rational]) -> Vec<Rational> {
    let m = deg(p).max(deg(h)) as usize;
    // bivariate in T: coeff of X^i is (p_i − T·h_i) = [p_i, −h_i]
    let f: Vec<Vec<Rational>> = (0..=m)
        .map(|i| {
            let pi = p.get(i).cloned().unwrap_or_else(rz);
            let hi = h.get(i).cloned().unwrap_or_else(rz);
            vec![pi, -hi]
        })
        .collect();
    discriminant_in_t(&f)
}

/// `Some(S(T))` if the pencil discriminant `Δ_X(P−T·H)` is a perfect square in
/// `ℚ[T]` (so `Gal(P−TH/ℚ(T)) ⊆ A_m`), else `None`. By Mestre's lemma this holds
/// exactly when [`verify_identity`] succeeds and `disc(P)` is a square.
pub fn pencil_disc_is_square(p: &[Rational], h: &[Rational]) -> Option<Vec<Rational>> {
    poly_sqrt(&pencil_discriminant(p, h))
}

// --------------------------------------------------------------------------- //
// Degree descent  Φ(τ,X) = (P(X)H(τ) − P(τ)H(X)) / (X − τ)
// --------------------------------------------------------------------------- //
/// The even-degree descent `Φ(τ,X) = (P(X)H(τ) − P(τ)H(X))/(X−τ)`, returned as a
/// bivariate polynomial (X-major; each `out[i]` is a polynomial in `τ`).
/// `deg_X Φ = deg_X P − 1`. When `P(0)=0`, `Φ(0,X) = H(0)·(P/X)`.
pub fn descent_phi(p: &[Rational], h: &[Rational]) -> Vec<Vec<Rational>> {
    let d = deg(p).max(deg(h)) as usize;
    // numerator N(τ,X) = P(X)H(τ) − P(τ)H(X): coeff of X^i is p_i·H(τ) − h_i·P(τ)
    let n_coeffs: Vec<Vec<Rational>> = (0..=d)
        .map(|i| {
            let pi = p.get(i).cloned().unwrap_or_else(rz);
            let hi = h.get(i).cloned().unwrap_or_else(rz);
            psub(&pscale(h, &pi), &pscale(p, &hi)) // poly in τ
        })
        .collect();
    // synthetic division in X by (X − τ): q_{d-1}=N_d, q_{i-1}=N_i + τ·q_i
    let dn = n_coeffs.len() - 1;
    let mut q = vec![vec![rz()]; dn]; // q[0..dn-1]
    q[dn - 1] = n_coeffs[dn].clone();
    for i in (1..dn).rev() {
        // q[i-1] = N_i + τ·q[i]   (τ· = shift τ-poly up by one)
        q[i - 1] = padd(&n_coeffs[i], &shift_tau(&q[i]));
    }
    q
}

/// Multiply a polynomial-in-`τ` by `τ` (shift coefficients up by one).
fn shift_tau(p: &[Rational]) -> Vec<Rational> {
    let mut o = vec![rz()];
    o.extend_from_slice(p);
    o
}

/// Specialize a bivariate `Φ(τ,X)` at `τ = val`, giving a univariate poly in `ℚ[X]`.
pub fn specialize(biv: &[Vec<Rational>], val: &Rational) -> Vec<Rational> {
    biv.iter()
        .map(|c| {
            let mut acc = rz();
            for coeff in c.iter().rev() {
                acc = acc * val.clone() + coeff.clone();
            }
            acc
        })
        .collect()
}

/// Clear denominators and content: the primitive integer model of `f ∈ ℚ[X]`.
pub fn to_integer_primitive(f: &[Rational]) -> Vec<Integer> {
    // common denominator
    let mut lcm = Integer::one();
    for c in f {
        let d = c.denominator().clone();
        let g = lcm.gcd(&d);
        lcm = lcm.clone() / g * d;
    }
    let ints: Vec<Integer> =
        f.iter().map(|c| c.numerator().clone() * (lcm.clone() / c.denominator().clone())).collect();
    // divide by content
    let mut content = Integer::zero();
    for c in &ints {
        if !c.is_zero() {
            content = content.gcd(c);
        }
    }
    if content.is_zero() || content.is_one() {
        ints
    } else {
        ints.iter().map(|c| c.clone() / content.clone()).collect()
    }
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
    fn wronskian_known_identity() {
        // P = X^3 - X, H = X^2 + X  →  P'H - PH' = (X^2 + X)^2
        let p = qs(&[0, -1, 0, 1]);
        let h = qs(&[0, 1, 1]);
        let w = wronskian(&p, &h);
        // X^4 + 2X^3 + X^2
        assert_eq!(w, qs(&[0, 0, 1, 2, 1]));
        assert_eq!(verify_identity(&p, &h), Some(qs(&[0, 1, 1])));
    }

    #[test]
    fn wronskian_non_square_rejected() {
        let p = qs(&[0, -1, 0, 1]); // X^3 - X
        let h = qs(&[1, 0, 0]); // H = 1  →  W = 3X^2 - 1, not a square
        assert_eq!(verify_identity(&p, &h), None);
    }

    #[test]
    fn pencil_disc_is_square_for_mestre_solution() {
        // For the solved identity, the pencil P - T·H must have square discriminant
        // in ℚ[T] (Mestre's lemma), since disc(P=X^3-X)=4 is a square.
        let p = qs(&[0, -1, 0, 1]);
        let h = qs(&[0, 1, 1]);
        let s = pencil_disc_is_square(&p, &h);
        assert!(s.is_some(), "pencil discriminant must be a perfect square in ℚ[T]");
        let d = pencil_discriminant(&p, &h);
        let sq = s.unwrap();
        assert_eq!(d, pmul(&sq, &sq), "poly_sqrt squared must reproduce the discriminant");
        // Mestre's lemma: Δ(P−TH) = Δ(P)·S(T)². disc(P)=4, so d/4 is also a square.
        let d_over_disc = pscale(&d, &Rational::new(1, 4).unwrap());
        assert!(poly_sqrt(&d_over_disc).is_some(), "d/disc(P) must be a perfect square");
    }

    #[test]
    fn descent_recovers_seed_at_zero() {
        // P = X·g with g = X^2 + 1 (so P = X^3 + X, P(0)=0), H with H(0) ≠ 0.
        // Φ(0,X) must equal H(0)·g(X).
        let g = qs(&[1, 0, 1]); // X^2 + 1
        let p = qs(&[0, 1, 0, 1]); // X·g = X^3 + X
        let h = qs(&[2, 1, 3]); // H = 3X^2 + X + 2, H(0)=2
        let phi = descent_phi(&p, &h);
        let phi0 = specialize(&phi, &q(0));
        // H(0)·g = 2·(X^2+1) = [2,0,2]
        let expect = pscale(&g, &q(2));
        // compare up to trailing zeros
        assert_eq!(deg(&phi0), deg(&expect));
        for i in 0..=(deg(&expect) as usize) {
            assert_eq!(phi0.get(i).cloned().unwrap_or_else(rz), expect[i]);
        }
    }

    #[test]
    fn descent_division_is_exact() {
        // Φ(τ,X)·(X−τ) must equal N(τ,X) = P(X)H(τ) − P(τ)H(X); check at a few (τ,X).
        let p = qs(&[1, -2, 0, 1]); // X^3 - 2X + 1
        let h = qs(&[3, 1, 2]); // 2X^2 + X + 3
        let phi = descent_phi(&p, &h);
        for &tau in &[0i64, 1, 2, -1] {
            for &xx in &[0i64, 1, 3, -2] {
                let tv = q(tau);
                let xv = q(xx);
                // Φ(τ,xx)·(xx-τ)
                let phi_t = specialize(&phi, &tv); // poly in X
                let lhs = {
                    let mut acc = rz();
                    for c in phi_t.iter().rev() {
                        acc = acc * xv.clone() + c.clone();
                    }
                    acc * (xv.clone() - tv.clone())
                };
                let eval = |poly: &[Rational], at: &Rational| {
                    let mut a = rz();
                    for c in poly.iter().rev() {
                        a = a * at.clone() + c.clone();
                    }
                    a
                };
                let rhs = eval(&p, &xv) * eval(&h, &tv) - eval(&p, &tv) * eval(&h, &xv);
                assert_eq!(lhs, rhs, "descent exactness at tau={tau}, X={xx}");
            }
        }
    }

    #[test]
    fn integer_primitive_model() {
        // (1/2)X^2 + (3/4)X + 1  →  clear to 2X^2 + 3X + 4
        let f = vec![q(1), Rational::new(3, 4).unwrap(), Rational::new(1, 2).unwrap()];
        assert_eq!(to_integer_primitive(&f), vec![Integer::from(4), Integer::from(3), Integer::from(2)]);
    }
}

// --------------------------------------------------------------------------- //
// Deterministic core of the Mestre identity solver
// --------------------------------------------------------------------------- //
// Reducing P'H − PH' = R² modulo the separable P (so gcd(P,P')=1): P'H ≡ R²
// (mod P) forces H ≡ R²·(P')⁻¹ (mod P), which (deg H < deg P) determines H from R.
// The full identity then holds iff the residual K − H' vanishes, where
// K = (P'H − R²)/P. So the remaining solver is exactly: find R with residual 0.

fn divmod(a: &[Rational], b: &[Rational]) -> (Vec<Rational>, Vec<Rational>) {
    let db = deg(b);
    let lcb_inv = b[db as usize].reciprocal().expect("nonzero leading coeff");
    let mut r = a.to_vec();
    let mut q = vec![rz(); (deg(a).max(0) - db).max(0) as usize + 1];
    while deg(&r) >= db && deg(&r) >= 0 {
        let dr = deg(&r) as usize;
        let coeff = r[dr].clone() * lcb_inv.clone();
        let shift = dr - db as usize;
        q[shift] = coeff.clone();
        for j in 0..b.len() {
            r[j + shift] = r[j + shift].clone() - coeff.clone() * b[j].clone();
        }
        while r.len() > 1 && *r.last().unwrap() == rz() {
            r.pop();
        }
        if deg(&r) < db {
            break;
        }
    }
    while q.len() > 1 && *q.last().unwrap() == rz() {
        q.pop();
    }
    (q, r)
}

fn prem(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    divmod(a, b).1
}

/// Extended GCD over `ℚ[X]`: returns `(g, s, t)` with `s·a + t·b = g`.
fn ext_gcd(a: &[Rational], b: &[Rational]) -> (Vec<Rational>, Vec<Rational>, Vec<Rational>) {
    let (mut r0, mut r1) = (a.to_vec(), b.to_vec());
    let (mut s0, mut s1) = (vec![Rational::from(1i64)], vec![rz()]);
    let (mut t0, mut t1) = (vec![rz()], vec![Rational::from(1i64)]);
    while deg(&r1) >= 0 {
        let (q, r) = divmod(&r0, &r1);
        r0 = r1;
        r1 = r;
        let ns = psub(&s0, &pmul(&q, &s1));
        s0 = s1;
        s1 = ns;
        let nt = psub(&t0, &pmul(&q, &t1));
        t0 = t1;
        t1 = nt;
    }
    (r0, s0, t0)
}

/// Inverse of `a` modulo `m` over `ℚ[X]`, or `None` if not coprime.
fn mod_inverse(a: &[Rational], m: &[Rational]) -> Option<Vec<Rational>> {
    let (g, s, _) = ext_gcd(a, m);
    if deg(&g) != 0 {
        return None; // not a unit
    }
    let inv_lead = g[0].reciprocal().ok()?;
    Some(prem(&pscale(&s, &inv_lead), m))
}

/// Given `P` (separable) and a candidate `R`, return the forced auxiliary
/// polynomial `H = R²·(P')⁻¹ mod P` together with the **residual** `K − H'`
/// (`K = (P'H − R²)/P`). The pair `(P, H)` solves the Mestre identity
/// `P'H − PH' = R²` **iff the residual is zero**. `None` if `P` is inseparable.
pub fn solve_h_given_r(p: &[Rational], r: &[Rational]) -> Option<(Vec<Rational>, Vec<Rational>)> {
    let pp = derivative(p);
    let pinv = mod_inverse(&pp, p)?;
    let r2 = pmul(r, r);
    let h = prem(&pmul(&r2, &pinv), p);
    // K = (P'H − R²)/P
    let pe = psub(&pmul(&pp, &h), &r2);
    let (k, rem) = divmod(&pe, p);
    debug_assert!(deg(&rem) < 0 || rem.iter().all(|c| *c == rz()), "P'H−R² not divisible by P");
    let residual = psub(&k, &derivative(&h));
    Some((h, residual))
}

#[cfg(test)]
mod solver_tests {
    use super::*;

    fn q(n: i64) -> Rational {
        Rational::from(n)
    }
    fn qs(v: &[i64]) -> Vec<Rational> {
        v.iter().map(|&x| q(x)).collect()
    }

    #[test]
    fn h_from_r_recovers_known_solution() {
        // P = X^3 - X, R = X^2 + X.  The forced H must give residual 0 and match the
        // known solution H = X^2 + X.
        let p = qs(&[0, -1, 0, 1]);
        let r = qs(&[0, 1, 1]);
        let (h, residual) = solve_h_given_r(&p, &r).unwrap();
        assert!(residual.iter().all(|c| *c == q(0)), "residual must vanish for a true solution");
        // and (P,H) indeed solves the identity
        assert_eq!(verify_identity(&p, &h), Some(r));
    }

    #[test]
    fn h_from_r_nonsolution_has_residual() {
        // A random R that does not solve the identity → nonzero residual.
        let p = qs(&[0, -1, 0, 1]);
        let r = qs(&[1, 0, 1]); // X^2 + 1
        let (_h, residual) = solve_h_given_r(&p, &r).unwrap();
        assert!(residual.iter().any(|c| *c != q(0)), "non-solution must have nonzero residual");
    }
}
