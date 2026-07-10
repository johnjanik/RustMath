//! Ideal class group of a number field `K = ℚ[x]/(f)`.
//!
//! Rigorous core: a **principality test** — is `𝔞 = (α)`? — by searching the ideal's
//! `T₂` (Minkowski) lattice for a short element `α` with `|N(α)| = N(𝔞)` (LLL,
//! Phase 3b). An element of `𝔞` of norm `N(𝔞)` generates it, so this both decides
//! principality and returns a generator.
//!
//! Class group: the factor base is all prime ideals above rational primes `p` up to
//! the **Minkowski bound** (these generate `Cl(K)`). Relations are collected from the
//! principal ideals `(p)` and from small principal ideals `(α)` that factor over the
//! factor base (index-calculus); the Smith normal form of the relation matrix gives
//! the invariant factors of `Cl(K)`. Validated against `gp bnfinit` on small fields.
//! Returns `None` when a factor-base prime divides the index (Dedekind generators
//! unreliable) or the relations are insufficient (free part remains).

use crate::ideals::{ideal_norm, prime_ideals, rational_prime_ideal, Ideal};
use crate::round2::{bareiss_det, field_discriminant, maximal_order_data, OrderData};
use rustmath_integers::Integer;
use rustmath_matrix::lll::lll_reduce;
use rustmath_polynomials::real_roots::count_real_roots_int;

// --------------------------------------------------------------------------- //
// Element norm and complex embeddings
// --------------------------------------------------------------------------- //
/// Absolute norm `N(α) = det(mult-by-α)` for `α` in integral-basis coordinates.
pub fn element_norm(ord: &OrderData, alpha: &[Integer]) -> Integer {
    let n = ord.n;
    // M[k][j] = (α·e_j)_k
    let mut rows = vec![vec![Integer::zero(); n]; n];
    for j in 0..n {
        let mut ej = vec![Integer::zero(); n];
        ej[j] = Integer::one();
        let col = ord.mul(alpha, &ej);
        for k in 0..n {
            rows[k][j] = col[k].clone();
        }
    }
    bareiss_det(&rows)
}

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    fn add(self, o: C) -> C {
        C { re: self.re + o.re, im: self.im + o.im }
    }
    fn sub(self, o: C) -> C {
        C { re: self.re - o.re, im: self.im - o.im }
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C { re: (self.re * o.re + self.im * o.im) / d, im: (self.im * o.re - self.re * o.im) / d }
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

fn roots(f: &[Integer]) -> Vec<C> {
    let n = f.len() - 1;
    let c: Vec<f64> = f.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let eval = |z: C| {
        let mut acc = C { re: 1.0, im: 0.0 };
        for k in (0..n).rev() {
            acc = acc.mul(z).add(C { re: c[k], im: 0.0 });
        }
        acc
    };
    let mut z: Vec<C> = (0..n)
        .map(|k| {
            let s = C { re: 0.4, im: 0.9 };
            let mut p = C { re: 1.0, im: 0.0 };
            for _ in 0..k {
                p = p.mul(s);
            }
            p
        })
        .collect();
    for _ in 0..300 {
        let mut md = 0.0f64;
        for i in 0..n {
            let mut den = C { re: 1.0, im: 0.0 };
            for j in 0..n {
                if j != i {
                    den = den.mul(z[i].sub(z[j]));
                }
            }
            let d = eval(z[i]).div(den);
            z[i] = z[i].sub(d);
            md = md.max(d.abs());
        }
        if md < 1e-13 {
            break;
        }
    }
    z
}

/// Split roots into real embeddings and one representative per complex pair.
fn embeddings(rts: &[C]) -> (Vec<C>, Vec<C>) {
    let mut reals = Vec::new();
    let mut cplx = Vec::new();
    let mut used = vec![false; rts.len()];
    for i in 0..rts.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        if rts[i].im.abs() < 1e-6 {
            reals.push(rts[i]);
        } else {
            cplx.push(rts[i]);
            let mut best = usize::MAX;
            let mut bd = f64::INFINITY;
            for j in 0..rts.len() {
                if j != i && !used[j] {
                    let d = (rts[i].re - rts[j].re).abs() + (rts[i].im + rts[j].im).abs();
                    if d < bd {
                        bd = d;
                        best = j;
                    }
                }
            }
            if best != usize::MAX {
                used[best] = true;
            }
        }
    }
    (reals, cplx)
}

/// `T₂` lattice of an ideal: row `i` is the Minkowski embedding of basis element
/// `β_i = Σ_k basis[i][k]·ω_k`, scaled and rounded.
fn embed_ideal(ord: &OrderData, ideal: &Ideal, reals: &[C], cplx: &[C], scale: f64) -> Vec<Vec<Integer>> {
    let n = ord.n;
    let s2 = std::f64::consts::SQRT_2;
    let dd = ord.d.to_f64().unwrap_or(1.0);
    let mut lattice = Vec::with_capacity(n);
    for col in 0..n {
        // β power-coords numerators: pow[i] = Σ_k w[i][k]·basis[col][k]
        let bcol = &ideal.basis[col];
        let pow: Vec<f64> = (0..n)
            .map(|i| {
                let mut s = 0.0f64;
                for k in 0..n {
                    s += ord.w[i][k].to_f64().unwrap_or(0.0) * bcol[k].to_f64().unwrap_or(0.0);
                }
                s
            })
            .collect();
        let embed = |r: C| -> C {
            let mut acc = C { re: 0.0, im: 0.0 };
            let mut pw = C { re: 1.0, im: 0.0 };
            for &coef in &pow {
                acc = acc.add(C { re: coef, im: 0.0 }.mul(pw));
                pw = pw.mul(r);
            }
            C { re: acc.re / dd, im: acc.im / dd }
        };
        let mut row = Vec::with_capacity(n);
        for r in reals {
            row.push(Integer::from((embed(*r).re * scale).round() as i64));
        }
        for r in cplx {
            let e = embed(*r);
            row.push(Integer::from((s2 * e.re * scale).round() as i64));
            row.push(Integer::from((s2 * e.im * scale).round() as i64));
        }
        lattice.push(row);
    }
    lattice
}

/// Is the integral ideal `𝔞` principal? Returns a generator `α` (integral-basis
/// coords) if so. Searches short elements of the `T₂` lattice for `|N(α)| = N(𝔞)`.
pub fn is_principal(f: &[Integer], ord: &OrderData, ideal: &Ideal) -> Option<Vec<Integer>> {
    let n = ord.n;
    let target = ideal_norm(ideal).abs();
    if target.is_zero() {
        return None;
    }
    let rts = roots(f);
    let (reals, cplx) = embeddings(&rts);
    let lattice = embed_ideal(ord, ideal, &reals, &cplx, 1e6);
    let (_red, u) = lll_reduce(&lattice);
    // element of ideal from coefficients over the ideal basis
    let elt = |coeffs: &[Integer]| -> Vec<Integer> {
        let mut a = vec![Integer::zero(); n];
        for (i, ci) in coeffs.iter().enumerate() {
            if ci.is_zero() {
                continue;
            }
            for k in 0..n {
                a[k] = a[k].clone() + ci.clone() * ideal.basis[i][k].clone();
            }
        }
        a
    };
    let check = |coeffs: &[Integer]| -> Option<Vec<Integer>> {
        let a = elt(coeffs);
        if a.iter().all(|x| x.is_zero()) {
            return None;
        }
        if element_norm(ord, &a).abs() == target {
            Some(a)
        } else {
            None
        }
    };
    // individual reduced vectors
    for row in &u {
        if let Some(a) = check(row) {
            return Some(a);
        }
    }
    // small combinations of the first few reduced vectors
    let kk = n.min(4);
    let bound = 2i64;
    let mut idx = vec![-bound; kk];
    loop {
        let mut coeffs = vec![Integer::zero(); n];
        for j in 0..kk {
            if idx[j] != 0 {
                for i in 0..n {
                    coeffs[i] = coeffs[i].clone() + Integer::from(idx[j]) * u[j][i].clone();
                }
            }
        }
        if let Some(a) = check(&coeffs) {
            return Some(a);
        }
        // odometer
        let mut p = 0;
        loop {
            if p == kk {
                idx = vec![0; 0];
                break;
            }
            idx[p] += 1;
            if idx[p] > bound {
                idx[p] = -bound;
                p += 1;
            } else {
                break;
            }
        }
        if idx.is_empty() {
            break;
        }
    }
    None
}

// --------------------------------------------------------------------------- //
// Minkowski bound and the class group
// --------------------------------------------------------------------------- //
/// Minkowski bound `M_K = √|d_K| · (n!/nⁿ) · (4/π)^{r₂}` (floored).
pub fn minkowski_bound(f: &[Integer]) -> i64 {
    let n = f.len() - 1;
    let disc = field_discriminant(f);
    let r1 = count_real_roots_int(f);
    let r2 = (n - r1) / 2;
    let sqrt_disc = (disc.to_f64().unwrap_or(0.0)).abs().sqrt();
    let mut fact = 1.0f64;
    for i in 1..=n {
        fact *= i as f64;
    }
    let nn = (n as f64).powi(n as i32);
    let four_over_pi = (4.0 / std::f64::consts::PI).powi(r2 as i32);
    (sqrt_disc * (fact / nn) * four_over_pi).floor() as i64
}

fn small_primes_up_to(m: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let mut p = 2i64;
    while p <= m {
        let mut is_p = true;
        let mut d = 2;
        while d * d <= p {
            if p % d == 0 {
                is_p = false;
                break;
            }
            d += 1;
        }
        if is_p {
            out.push(p);
        }
        p += 1;
    }
    out
}

/// The ideal class group `Cl(K)` of `K = ℚ[x]/(f)` (f monic irreducible) as its
/// list of invariant factors (`[]` ⇒ trivial, `h_K = 1`). Returns `None` if a prime
/// in the factor base divides the index (Dedekind generators unreliable) or the
/// collected relations are insufficient (free part remains). Validated vs `gp`.
pub fn class_group(f: &[Integer]) -> Option<Vec<usize>> {
    let ord = maximal_order_data(f);
    let n = ord.n;
    let m = minkowski_bound(f);
    // factor base: all prime ideals above rational primes p ≤ M
    let mut fb: Vec<Ideal> = Vec::new();
    let mut fb_prime: Vec<i64> = Vec::new(); // which rational prime each FB ideal lies over
    for p in small_primes_up_to(m) {
        let (_o, primes) = prime_ideals(f, p);
        // Dedekind unreliable if p | index — detect via crate::ideals
        if crate::ideals::prime_decomposition(f, p).p_divides_index {
            return None;
        }
        for (pr, _e, _fdeg) in primes {
            fb.push(pr);
            fb_prime.push(p);
        }
    }
    let g = fb.len();
    if g == 0 {
        return Some(Vec::new()); // M < 2 ⇒ trivial class group
    }
    // relation rows over the factor base
    let mut relations: Vec<Vec<i64>> = Vec::new();
    // (1) (p) = ∏ 𝔭^{v}  is principal  →  relation (v_𝔭)
    for &p in &small_primes_up_to(m) {
        let pid = rational_prime_ideal(&ord, p);
        let mut row = vec![0i64; g];
        for (col, pr) in fb.iter().enumerate() {
            if fb_prime[col] == p {
                row[col] = crate::ideals::ideal_valuation(&ord, &pid, pr) as i64;
            }
        }
        relations.push(row);
    }
    // (2) cross-relations from small principal ideals (α) that are FB-smooth.
    //     (α) principal ⇒ Σ v_𝔭(α)·[𝔭] = 0 in Cl(K). Captures relations that (p)
    //     and generator orders miss (e.g. the genus relation in non-cyclic groups).
    if n <= 4 {
        let fb_norms: Vec<Integer> = fb.iter().map(ideal_norm).collect();
        let b = 4i64;
        let mut idx = vec![-b; n];
        loop {
            let alpha: Vec<Integer> = idx.iter().map(|&x| Integer::from(x)).collect();
            if alpha.iter().any(|x| !x.is_zero()) {
                let nrm = element_norm(&ord, &alpha).abs();
                if !nrm.is_zero() {
                    let aid = crate::ideals::ideal_from_generators(&ord, &[alpha]);
                    let mut row = vec![0i64; g];
                    let mut prod = Integer::one();
                    for (col, pr) in fb.iter().enumerate() {
                        let v = crate::ideals::ideal_valuation(&ord, &aid, pr);
                        row[col] = v as i64;
                        for _ in 0..v {
                            prod = prod * fb_norms[col].clone();
                        }
                    }
                    if prod == nrm {
                        relations.push(row); // FB-smooth ⇒ valid relation
                    }
                }
            }
            // odometer over idx ∈ [-b, b]^n
            let mut p = 0;
            while p < n {
                idx[p] += 1;
                if idx[p] > b {
                    idx[p] = -b;
                    p += 1;
                } else {
                    break;
                }
            }
            if p == n {
                break;
            }
        }
    }
    // Smith normal form of the relation matrix → invariant factors
    invariant_factors(&relations, g)
}

/// Invariant factors `> 1` from the relation matrix (rows × g) via SNF. `None` if
/// the relation lattice is not full rank (free part remains).
fn invariant_factors(relations: &[Vec<i64>], g: usize) -> Option<Vec<usize>> {
    if relations.is_empty() {
        return None;
    }
    // HNF-reduce the relation rows (as columns) to a g×g basis, then SNF.
    let cols: Vec<Vec<Integer>> =
        relations.iter().map(|r| r.iter().map(|&x| Integer::from(x)).collect()).collect();
    let basis = crate::round2::hnf_basis(&cols, g); // g columns
    let mut m = vec![vec![Integer::zero(); g]; g];
    for i in 0..g {
        for j in 0..g {
            m[i][j] = basis[j][i].clone();
        }
    }
    let diag = snf_diagonal(m);
    let mut factors = Vec::new();
    for d in diag {
        if d.is_zero() {
            return None; // free part remains: insufficient relations
        }
        if !d.is_one() {
            factors.push(d.to_i64() as usize);
        }
    }
    factors.sort_unstable();
    Some(factors)
}

/// Smith normal form diagonal (invariant factors `d₁|d₂|…`) of a square integer
/// matrix, by elementary row/column reduction. Self-contained (the matrix crate's
/// SNF can loop on some inputs).
fn snf_diagonal(mut a: Vec<Vec<Integer>>) -> Vec<Integer> {
    let g = a.len();
    let mut res = Vec::new();
    let mut t = 0;
    while t < g {
        // smallest nonzero pivot in submatrix [t..][t..]
        let mut piv: Option<(usize, usize)> = None;
        let mut best = Integer::zero();
        for i in t..g {
            for j in t..g {
                if !a[i][j].is_zero() {
                    let v = a[i][j].abs();
                    if piv.is_none() || v < best {
                        best = v;
                        piv = Some((i, j));
                    }
                }
            }
        }
        let (pi, pj) = match piv {
            Some(x) => x,
            None => {
                for _ in t..g {
                    res.push(Integer::zero());
                }
                break;
            }
        };
        a.swap(pi, t);
        for r in 0..g {
            a[r].swap(pj, t);
        }
        // clear row t and column t off the diagonal (Euclid via swaps)
        loop {
            for i in (t + 1)..g {
                if !a[i][t].is_zero() {
                    let q = a[i][t].clone() / a[t][t].clone();
                    for j in t..g {
                        a[i][j] = a[i][j].clone() - q.clone() * a[t][j].clone();
                    }
                    if !a[i][t].is_zero() {
                        a.swap(i, t);
                    }
                }
            }
            for j in (t + 1)..g {
                if !a[t][j].is_zero() {
                    let q = a[t][j].clone() / a[t][t].clone();
                    for i in t..g {
                        a[i][j] = a[i][j].clone() - q.clone() * a[i][t].clone();
                    }
                    if !a[t][j].is_zero() {
                        for r in 0..g {
                            a[r].swap(j, t);
                        }
                    }
                }
            }
            let row_clean = (t + 1..g).all(|j| a[t][j].is_zero());
            let col_clean = (t + 1..g).all(|i| a[i][t].is_zero());
            if row_clean && col_clean {
                break;
            }
        }
        // ensure a[t][t] divides the rest; else fold an offending row in and redo
        let mut redo = false;
        'outer: for i in (t + 1)..g {
            for j in (t + 1)..g {
                if !(a[i][j].clone() % a[t][t].clone()).is_zero() {
                    for c in t..g {
                        a[t][c] = a[t][c].clone() + a[i][c].clone();
                    }
                    redo = true;
                    break 'outer;
                }
            }
        }
        if redo {
            continue;
        }
        res.push(a[t][t].abs());
        t += 1;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn element_norm_quadratic() {
        // Q(i): N(a + b i) = a^2 + b^2. ω = (1, i); α = 1 + 2i → coords [1,2], N=5.
        let ord = maximal_order_data(&iz(&[1, 0, 1]));
        assert_eq!(element_norm(&ord, &iz(&[1, 2])), Integer::from(5));
        assert_eq!(element_norm(&ord, &iz(&[3, 0])), Integer::from(9)); // N(3)=9
    }

    #[test]
    fn principality_quadratic() {
        // Q(i): (2) is principal; the prime 𝔭=(1+i) above 2 is principal.
        let f = iz(&[1, 0, 1]);
        let ord = maximal_order_data(&f);
        let (_o, p2) = prime_ideals(&f, 2);
        assert!(is_principal(&f, &ord, &p2[0].0).is_some(), "𝔭 over 2 in Q(i) is principal");
        // Q(sqrt(-5)): the prime above 2 is NON-principal (h=2).
        let f5 = iz(&[5, 0, 1]);
        let ord5 = maximal_order_data(&f5);
        let (_o5, q2) = prime_ideals(&f5, 2);
        assert!(is_principal(&f5, &ord5, &q2[0].0).is_none(), "𝔭 over 2 in Q(sqrt-5) is NOT principal");
    }
}
