//! Panayi root-finding in local fields, and the Galois decision for a local field
//! (Module 9, the `pAdicGaloisGroup` endgame — base layer).
//!
//! Finding roots of `f ∈ O_K[x]` in a local field `K` cannot use plain Hensel/Newton
//! lifting once the root is multiple modulo the maximal ideal (then `f'(root)` is not
//! a unit). **Panayi's method** finds roots by repeated substitution + deflation: at
//! each level it solves `f̄ = 0` in the residue field, and for each residue root `r`
//! substitutes `x = r + π·y` (`π` a uniformizer), strips the `π`-content of the
//! shifted polynomial `f(r+πy)`, and recurses. The content-stripping (*deflation*)
//! is essential: it makes the count the true number of roots in `K`, not the
//! over-count of solutions modulo a power of the uniformizer (e.g. `x²−1` has 4
//! solutions mod `2^k` for every `k ≥ 3` but only the two roots `±1` in `ℤ_2`).
//!
//! `roots_in_zp` / `count_roots_qp` do this over `ℚ_p` (residue field `F_p`,
//! uniformizer `p`); `roots_in_eisenstein` does it over a totally ramified
//! `K = ℤ_p[π]/(g)` (uniformizer `π`, using `EisensteinElement` arithmetic and the
//! division-by-`π` primitive). On top sits the Galois decision `is_eisenstein_galois`:
//! `K = ℚ_p[x]/(g)` is Galois over `ℚ_p` iff `g` splits completely in `K`, i.e. iff it
//! has `deg g` roots in `K` (the number found `= |Aut(K/ℚ_p)|`). Validated against
//! PARI/GP `polrootspadic` and known local Galois groups.

use crate::local_field::EisensteinElement;
use rustmath_integers::Integer;

fn eval(f: &[Integer], x: &Integer, modulus: &Integer) -> Integer {
    let mut acc = Integer::zero();
    for c in f.iter().rev() {
        acc = (acc * x.clone() + c.clone()) % modulus.clone();
    }
    let r = acc % modulus.clone();
    if r.signum() < 0 {
        r + modulus.clone()
    } else {
        r
    }
}

fn binom(n: usize, k: usize) -> Integer {
    if k > n {
        return Integer::zero();
    }
    let k = k.min(n - k);
    let mut num = Integer::one();
    let mut den = Integer::one();
    for i in 0..k {
        num = num * Integer::from((n - i) as i64);
        den = den * Integer::from((i + 1) as i64);
    }
    num / den
}

/// `h(y) = f(r + p·y)`: `h_k = p^k · Σ_{i≥k} f_i C(i,k) r^{i−k}`.
fn compose_shift(f: &[Integer], r: i64, p: i64) -> Vec<Integer> {
    let d = f.len();
    let ri = Integer::from(r);
    let pk_base = Integer::from(p);
    let mut h = vec![Integer::zero(); d];
    // precompute r^j
    let mut rpow = vec![Integer::one(); d];
    for j in 1..d {
        rpow[j] = rpow[j - 1].clone() * ri.clone();
    }
    for k in 0..d {
        let mut s = Integer::zero();
        for i in k..d {
            if f[i].is_zero() {
                continue;
            }
            s = s + f[i].clone() * binom(i, k) * rpow[i - k].clone();
        }
        h[k] = pk_base.pow(k as u32) * s;
    }
    h
}

/// Strip a polynomial of its `p`-content (divide out the highest power of `p`
/// dividing every coefficient) and trim trailing zeros.
fn strip_content(mut f: Vec<Integer>, p: i64) -> Vec<Integer> {
    while f.len() > 1 && f.last().map(|c| c.is_zero()).unwrap_or(false) {
        f.pop();
    }
    let pi = Integer::from(p);
    loop {
        if f.iter().all(|c| c.is_zero()) {
            break;
        }
        if f.iter().all(|c| (c.clone() % pi.clone()).is_zero()) {
            for c in f.iter_mut() {
                *c = c.clone() / pi.clone();
            }
        } else {
            break;
        }
    }
    f
}

fn poly_degree(f: &[Integer]) -> usize {
    let mut d = f.len();
    while d > 1 && f[d - 1].is_zero() {
        d -= 1;
    }
    d - 1
}

/// All roots of `f ∈ ℤ[x]` in `ℤ_p`, returned as residues modulo `p^prec` (the
/// **distinct genuine roots** to that precision). Proper Panayi: at each level find
/// the residue roots `r` of `f̄` over `F_p`, substitute `x = r + p·y`, strip the
/// `p`-content of `f(r+py)`, and recurse — the content-stripping deflates the
/// singular (`f'(r) ≡ 0`) case so the count is the true `ℤ_p` root count (not the
/// over-count of solutions mod `p^prec`).
pub fn roots_in_zp(f: &[Integer], p: i64, prec: u32) -> Vec<Integer> {
    fn helper(f: &[Integer], p: i64, prec: u32, bound: u32) -> Vec<Integer> {
        if prec == 0 {
            return vec![Integer::zero()];
        }
        let f = strip_content(f.to_vec(), p);
        if poly_degree(&f) == 0 {
            return Vec::new(); // nonzero constant: no roots
        }
        // bound coefficient growth: work modulo p^bound.
        let modulus = Integer::from(p).pow(bound);
        let pi = Integer::from(p);
        let pk = pi.clone();
        let mut result: Vec<Integer> = Vec::new();
        for r in 0..p {
            if !eval(&f, &Integer::from(r), &pi).is_zero() {
                continue;
            }
            let h: Vec<Integer> =
                compose_shift(&f, r, p).into_iter().map(|c| c % modulus.clone()).collect();
            for y0 in helper(&h, p, prec - 1, bound) {
                result.push(Integer::from(r) + pk.clone() * y0);
            }
        }
        result
    }
    let bound = prec + 4;
    let mut roots = helper(f, p, prec, bound);
    let modulus = Integer::from(p).pow(prec);
    for r in roots.iter_mut() {
        *r = r.clone() % modulus.clone();
        if r.signum() < 0 {
            *r = r.clone() + modulus.clone();
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

/// The number of roots of `f` in `ℚ_p` (= the number of degree-1 factors of `f` over
/// `ℚ_p`, counted without multiplicity for separable `f`).
pub fn count_roots_qp(f: &[Integer], p: i64) -> usize {
    roots_in_zp(f, p, 20).len()
}

// --------------------------------------------------------------------------- //
// Root-finding in a totally ramified extension, and the Galois decision
// --------------------------------------------------------------------------- //

/// Evaluate a polynomial with `O_K` coefficients at `x ∈ O_K` (Horner).
fn eval_ok(phi: &[EisensteinElement], x: &EisensteinElement, p: i64, n: u32, g: &[Integer]) -> EisensteinElement {
    let mut acc = EisensteinElement::zero(p, n, g.to_vec());
    for c in phi.iter().rev() {
        acc = acc.mul(x).add(c);
    }
    acc
}

/// Strip the `π`-content of a polynomial over `O_K`: divide every coefficient by `π`
/// while they all have positive valuation (residue digit `0`); trim trailing zeros.
fn strip_pi_content(mut phi: Vec<EisensteinElement>) -> Vec<EisensteinElement> {
    while phi.len() > 1 && phi.last().map(|c| c.is_zero()).unwrap_or(false) {
        phi.pop();
    }
    loop {
        if phi.iter().all(|c| c.is_zero()) {
            break;
        }
        if phi.iter().all(|c| c.residue_digit() == 0) {
            for c in phi.iter_mut() {
                *c = c.div_by_uniformizer();
            }
        } else {
            break;
        }
    }
    phi
}

fn ok_degree(phi: &[EisensteinElement]) -> usize {
    let mut d = phi.len();
    while d > 1 && phi[d - 1].is_zero() {
        d -= 1;
    }
    d - 1
}

/// All roots of the integer polynomial `phi` in the totally ramified local field
/// `K = ℤ_p[π]/(g)` (`g` Eisenstein, residue field `F_p`), to the working precision.
/// Proper Panayi over `O_K`, mirroring [`roots_in_zp`]: at each level find the
/// residue roots `r ∈ F_p` of `φ̄`, substitute `x = r + π·y`, strip the `π`-content
/// of `φ(r+πy)` (the deflation), and recurse.
pub fn roots_in_eisenstein(phi: &[Integer], g: &[Integer], p: i64) -> Vec<EisensteinElement> {
    let e = g.len() - 1;
    let depth = (e * e + 2 * e + 6) as u32; // π-adic digits to resolve the conjugates
    let n = depth / (e as u32) + (e as u32) + 8; // p-adic precision
    let gv = g.to_vec();
    let pi = EisensteinElement::uniformizer(p, n, gv.clone());
    let phi0: Vec<EisensteinElement> =
        phi.iter().map(|c| EisensteinElement::from_int(p, n, gv.clone(), c.clone())).collect();

    // helper returns the per-branch correction `y` (an O_K element) so that the root
    // is `r + π·y`; at the top these compose into the actual roots.
    fn helper(
        phi: &[EisensteinElement],
        depth: u32,
        p: i64,
        n: u32,
        g: &[Integer],
        pi: &EisensteinElement,
    ) -> Vec<EisensteinElement> {
        if depth == 0 {
            return vec![EisensteinElement::zero(p, n, g.to_vec())];
        }
        let phi = strip_pi_content(phi.to_vec());
        if ok_degree(&phi) == 0 {
            return Vec::new(); // unit constant: no roots
        }
        let mut out = Vec::new();
        for r in 0..p {
            let rk = EisensteinElement::from_int(p, n, g.to_vec(), Integer::from(r));
            if eval_ok(&phi, &rk, p, n, g).residue_digit() != 0 {
                continue; // φ̄(r) ≠ 0 in F_p
            }
            // ψ(y) = φ(r + π·y).
            let psi: Vec<EisensteinElement> = compose_pi(&phi, &rk, pi);
            for y0 in helper(&psi, depth - 1, p, n, g, pi) {
                out.push(rk.add(&pi.mul(&y0)));
            }
        }
        out
    }

    let roots = helper(&phi0, depth, p, n, &gv, &pi);
    // Dedupe by residue at a fixed precision.
    let mut uniq: Vec<EisensteinElement> = Vec::new();
    for r in roots {
        if !uniq.iter().any(|u| u == &r) {
            uniq.push(r);
        }
    }
    uniq
}

/// `ψ(y) = φ(r + π·y)` for a polynomial `φ` over `O_K`: `ψ_k = Σ_{i≥k} φ_i C(i,k) r^{i−k} π^k`.
fn compose_pi(
    phi: &[EisensteinElement],
    r: &EisensteinElement,
    pi: &EisensteinElement,
) -> Vec<EisensteinElement> {
    let d = phi.len();
    let p = r.prime;
    let n = r.precision;
    let g = &r.g;
    // r^j
    let mut rpow = vec![EisensteinElement::one(p, n, g.clone())];
    for j in 1..d {
        rpow.push(rpow[j - 1].mul(r));
    }
    // π^k
    let mut pipow = vec![EisensteinElement::one(p, n, g.clone())];
    for k in 1..d {
        pipow.push(pipow[k - 1].mul(pi));
    }
    let mut psi = Vec::with_capacity(d);
    for k in 0..d {
        let mut s = EisensteinElement::zero(p, n, g.clone());
        for i in k..d {
            let c = phi[i].scale_int(&binom(i, k)).mul(&rpow[i - k]);
            s = s.add(&c);
        }
        psi.push(s.mul(&pipow[k]));
    }
    psi
}

/// Decide whether `K = ℚ_p[x]/(g)` (`g` Eisenstein, totally ramified) is **Galois**
/// over `ℚ_p`: true iff `g` splits completely in `K`, i.e. iff it has `deg g` roots
/// in `K`. Returns `(is_galois, number_of_roots_in_K)`.
pub fn is_eisenstein_galois(g: &[Integer], p: i64) -> (bool, usize) {
    let e = g.len() - 1;
    let roots = roots_in_eisenstein(g, g, p);
    (roots.len() == e, roots.len())
}

/// The **true** lower-numbering ramification filtration of `K = ℚ_p[x]/(g)` when it
/// is Galois — computed from the automorphisms themselves: the roots `β` of `g` in
/// `K` are the images `σ(π)`, and `|G_i| = 1 + #{β : v_K(β − π) ≥ i+1}`. Returns
/// `None` if `K/ℚ_p` is not Galois (`g` does not split in `K`). This cross-checks
/// `ramification::wild_filtration_from_eisenstein`, which gets the same filtration
/// from the ramification polygon without root-finding.
pub fn galois_filtration(g: &[Integer], p: i64) -> Option<crate::ramification::RamificationFiltration> {
    let e = g.len() - 1;
    let roots = roots_in_eisenstein(g, g, p);
    if roots.len() != e {
        return None;
    }
    let n = roots[0].precision;
    let pi = EisensteinElement::uniformizer(p, n, g.to_vec());
    // m(β) = v_K(β − π); the identity (β = π) gives ∞ and is skipped.
    let ms: Vec<i64> = roots.iter().filter_map(|b| b.sub(&pi).valuation()).collect();
    let maxm = ms.iter().copied().max().unwrap_or(0);
    let orders: Vec<usize> =
        (0..maxm).map(|i| 1 + ms.iter().filter(|&&m| m >= i + 1).count()).collect();
    Some(crate::ramification::RamificationFiltration::new(orders))
}

// --------------------------------------------------------------------------- //
// Naming the local Galois group of a quartic (the splitting field / closure)
// --------------------------------------------------------------------------- //

/// Is the integer `d` a square in `ℚ_p`? (`v_p(d)` even and the unit part is a square
/// — a QR mod `p` for odd `p`, or `≡ 1 mod 8` for `p = 2`.)
pub fn is_padic_square(d: &Integer, p: i64) -> bool {
    if d.is_zero() {
        return true;
    }
    let pi = Integer::from(p);
    let v = d.abs().valuation(&pi);
    if v % 2 == 1 {
        return false;
    }
    let mut u = d.clone() / pi.pow(v);
    if p == 2 {
        let mut m = u % Integer::from(8);
        if m.signum() < 0 {
            m = m + Integer::from(8);
        }
        m == Integer::one()
    } else {
        let mut m = u.clone() % pi.clone();
        if m.signum() < 0 {
            m = m + pi.clone();
        }
        u = m;
        // Legendre symbol via Euler's criterion: u^{(p-1)/2} ≡ 1 (mod p).
        let exp = Integer::from((p - 1) / 2);
        u.modpow(&exp, &pi).map(|r| r == Integer::one()).unwrap_or(false)
    }
}

/// Is `f` irreducible modulo `p` (so `K = ℚ_p[x]/(f)` is the unramified extension of
/// degree `deg f`, hence Galois cyclic)?
fn irreducible_mod_p(f: &[Integer], p: i64) -> bool {
    let fbar: Vec<i64> = f
        .iter()
        .map(|c| {
            let r = (c.clone() % Integer::from(p)).to_i64();
            ((r % p) + p) % p
        })
        .collect();
    let deg = {
        let mut d = fbar.len();
        while d > 1 && fbar[d - 1] == 0 {
            d -= 1;
        }
        d - 1
    };
    if deg < 1 {
        return false;
    }
    let fac = rustmath_polynomials::fp_factor::factor(&fbar, p);
    fac.len() == 1 && {
        let g = &fac[0];
        let mut gd = g.len();
        while gd > 1 && g[gd - 1] == 0 {
            gd -= 1;
        }
        gd - 1 == deg
    }
}

/// The resolvent cubic of a monic quartic `f = x⁴ + b x³ + c x² + d x + e`:
/// `y³ − c y² + (bd − 4e) y − (b²e − 4ce + d²)`.
fn resolvent_cubic(f: &[Integer]) -> Vec<Integer> {
    let e = f[0].clone();
    let d = f[1].clone();
    let c = f[2].clone();
    let b = f[3].clone();
    let four = Integer::from(4);
    let c0 = -(b.clone() * b.clone() * e.clone() - four.clone() * c.clone() * e.clone()
        + d.clone() * d.clone());
    let c1 = b.clone() * d.clone() - four * e;
    let c2 = -c;
    vec![c0, c1, c2, Integer::one()]
}

/// The local Galois group of an **irreducible quartic** `f` over `ℚ_p`: its name
/// among the transitive degree-4 groups, the group order (= degree of the splitting
/// field / Galois closure), and the `4Tt` label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarticGalois {
    pub name: &'static str,
    pub order: usize,
    pub label: &'static str,
}

/// Identify `Gal(f / ℚ_p)` for an irreducible quartic `f`, via the resolvent cubic
/// factorization over `ℚ_p`, the `ℚ_p`-square class of `disc(f)`, and (for the
/// `C₄`/`D₄` split) whether `K = ℚ_p[x]/(f)` is itself Galois. Returns `None` if the
/// `C₄`/`D₄` case cannot be decided (mixed ramification not covered here).
pub fn quartic_local_galois_group(f: &[Integer], p: i64) -> Option<QuarticGalois> {
    if f.len() != 5 {
        return None;
    }
    let r3 = resolvent_cubic(f);
    let nr = count_roots_qp(&r3, p);
    let disc = rustmath_polynomials::disc::discriminant(f);
    let dsq = is_padic_square(&disc, p);
    match nr {
        3 => Some(QuarticGalois { name: "V4", order: 4, label: "4T2" }),
        0 => {
            if dsq {
                Some(QuarticGalois { name: "A4", order: 12, label: "4T4" })
            } else {
                Some(QuarticGalois { name: "S4", order: 24, label: "4T5" })
            }
        }
        1 => {
            // C₄ ⟺ the splitting field is K itself (degree 4, K Galois); else D₄.
            let k_galois = if irreducible_mod_p(f, p) {
                Some(true) // unramified ⇒ cyclic C₄
            } else if crate::local_field::is_eisenstein(f, p) {
                Some(is_eisenstein_galois(f, p).0)
            } else {
                None // mixed ramification: not decided here
            };
            match k_galois {
                Some(true) => Some(QuarticGalois { name: "C4", order: 4, label: "4T1" }),
                Some(false) => Some(QuarticGalois { name: "D4", order: 8, label: "4T3" }),
                None => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn roots_in_qp_counts_match_pari_gp() {
        // Cross-checked against PARI/GP polrootspadic.
        assert_eq!(count_roots_qp(&iz(&[-2, 0, 1]), 7), 2); // x²−2 /7
        assert_eq!(count_roots_qp(&iz(&[-2, 0, 1]), 3), 0); // x²−2 /3
        assert_eq!(count_roots_qp(&iz(&[1, 0, 1]), 5), 2); // x²+1 /5
        assert_eq!(count_roots_qp(&iz(&[1, 0, 1]), 3), 0); // x²+1 /3
        assert_eq!(count_roots_qp(&iz(&[-2, 0, 0, 1]), 7), 0); // x³−2 /7
        assert_eq!(count_roots_qp(&iz(&[-2, 0, 0, 1]), 31), 3); // x³−2 /31
        assert_eq!(count_roots_qp(&iz(&[-1, 0, 1]), 2), 2); // x²−1 /2 (multiple mod 2)
        assert_eq!(count_roots_qp(&iz(&[0, -1, 0, 1]), 5), 3); // x³−x /5
        // (x−3)(x−10)(x²+1) = x⁴−13x³+31x²−13x+30 over Q_5: 4 roots.
        assert_eq!(count_roots_qp(&iz(&[30, -13, 31, -13, 1]), 5), 4);
    }

    #[test]
    fn roots_in_zp_values_are_actual_roots() {
        // x²−2 in Q_7 to precision 7^5: both roots square to 2.
        let prec = 5u32;
        let m = Integer::from(7).pow(prec);
        let rs = roots_in_zp(&iz(&[-2, 0, 1]), 7, prec);
        assert_eq!(rs.len(), 2);
        for r in &rs {
            assert_eq!((r.clone() * r.clone()) % m.clone(), Integer::from(2));
        }
    }

    #[test]
    fn eisenstein_galois_decision() {
        // The number of roots of g in K = ℚ_p[x]/(g) is |Aut(K/ℚ_p)|; K is Galois
        // iff that equals deg g (g splits completely in K).
        // Degree-2 extensions are always Galois.
        assert_eq!(is_eisenstein_galois(&iz(&[-2, 0, 1]), 2), (true, 2)); // Q_2(√2)
        assert_eq!(is_eisenstein_galois(&iz(&[2, -2, 1]), 2), (true, 2)); // Q_2(i)
        // Q_2(ζ_8) = ℚ_2[x]/Φ_8(x+1): Galois V_4, splits completely (4 roots).
        assert_eq!(is_eisenstein_galois(&iz(&[2, 4, 6, 4, 1]), 2), (true, 4));
        // Q_3(ζ_9) = ℚ_3[x]/Φ_9(x+1): Galois C_6, 6 roots.
        assert_eq!(is_eisenstein_galois(&iz(&[3, 9, 18, 21, 15, 6, 1]), 3), (true, 6));
        // x⁴−2 over ℚ_2: Galois group D_4 (needs i); Aut(K/ℚ_2) = {2^¼↦±2^¼} = 2.
        assert_eq!(is_eisenstein_galois(&iz(&[-2, 0, 0, 0, 1]), 2), (false, 2));
        // x³−3 over ℚ_3 = ℚ_3(3^⅓): not Galois (needs ζ_3); only the identity aut.
        assert_eq!(is_eisenstein_galois(&iz(&[-3, 0, 0, 1]), 3), (false, 1));
    }

    #[test]
    fn galois_filtration_from_automorphisms() {
        use crate::ramification::{wild_filtration_from_eisenstein, RamificationFiltration};
        let f = |v: &[usize]| RamificationFiltration::new(v.to_vec());
        // The filtration computed from the actual automorphisms (Panayi roots) agrees
        // with the ramification-polygon filtration and the gp-validated known ones.
        let cases: &[(&[i64], i64, &[usize])] = &[
            (&[-2, 0, 1], 2, &[2, 2, 2]),          // Q_2(√2)
            (&[2, -2, 1], 2, &[2, 2]),             // Q_2(i)
            (&[2, 4, 6, 4, 1], 2, &[4, 4, 2, 2]),  // Q_2(ζ_8), V_4
            (&[3, 9, 18, 21, 15, 6, 1], 3, &[6, 3, 3]), // Q_3(ζ_9), C_6
        ];
        for &(g, p, expected) in cases {
            let gz: Vec<Integer> = g.iter().map(|&x| Integer::from(x)).collect();
            let from_auts = galois_filtration(&gz, p).expect("Galois");
            assert_eq!(from_auts, f(expected), "automorphism filtration for {g:?}");
            // matches the ramification-polygon route (no root-finding):
            assert_eq!(from_auts, wild_filtration_from_eisenstein(&gz, p).unwrap());
        }
        // Non-Galois ⇒ None.
        assert!(galois_filtration(&iz(&[-2, 0, 0, 0, 1]), 2).is_none()); // x⁴−2
    }

    #[test]
    fn padic_square_classes() {
        assert!(!is_padic_square(&Integer::from(-2048), 2)); // v_2=11 odd
        assert!(is_padic_square(&Integer::from(256), 2)); // 2^8
        assert!(!is_padic_square(&Integer::from(125), 2)); // 5^3, unit 5 mod 8
        assert!(is_padic_square(&Integer::from(2), 7)); // 2 is a QR mod 7 (3²=2)
        assert!(!is_padic_square(&Integer::from(3), 7)); // 3 is not a QR mod 7
    }

    #[test]
    fn quartic_galois_groups_named() {
        let q = |v: &[i64], p: i64| quartic_local_galois_group(&iz(v), p).unwrap();
        // x⁴−2 over ℚ_2: the Galois CLOSURE is ℚ_2(2^¼, i), degree 8 ⇒ D₄ = 4T3.
        assert_eq!(q(&[-2, 0, 0, 0, 1], 2), QuarticGalois { name: "D4", order: 8, label: "4T3" });
        // x⁴+1 over ℚ_2 = ℚ_2(ζ_8): V₄, splitting field degree 4.
        assert_eq!(q(&[1, 0, 0, 0, 1], 2), QuarticGalois { name: "V4", order: 4, label: "4T2" });
        // Φ_5 over ℚ_2 (irreducible mod 2, unramified): cyclic C₄.
        assert_eq!(q(&[1, 1, 1, 1, 1], 2), QuarticGalois { name: "C4", order: 4, label: "4T1" });
        // x⁴+x+1 over ℚ_2 (irreducible mod 2, unramified): C₄.
        assert_eq!(q(&[1, 1, 0, 0, 1], 2), QuarticGalois { name: "C4", order: 4, label: "4T1" });
    }

    #[test]
    fn x4_minus_2_closure_degree_is_group_order() {
        // The Galois closure degree [L:ℚ_2] equals |Gal| = 8: K=ℚ_2(2^¼) is degree 4
        // and NOT Galois (2 automorphisms), and the splitting field adjoins i (the
        // root of the irreducible factor x²+√2 of x⁴−2 over K), giving 4·2 = 8.
        let (gal, auts) = is_eisenstein_galois(&iz(&[-2, 0, 0, 0, 1]), 2);
        assert!(!gal && auts == 2); // K not Galois, [K:ℚ_2]=4, |Aut(K)|=2
        let g = quartic_local_galois_group(&iz(&[-2, 0, 0, 0, 1]), 2).unwrap();
        assert_eq!(g.order, 8); // |Gal| = closure degree [L:ℚ_2]
        assert_eq!(g.order / 4, 2); // [L:K] = [L:ℚ_2] / [K:ℚ_2] = 8/4 = 2
    }
}
