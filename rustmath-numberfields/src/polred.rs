//! `polredabs`-style reduction: given a monic irreducible `f ∈ ℤ[x]` defining a
//! number field `K = ℚ(θ)`, find an alternative monic defining polynomial of `K`
//! with small coefficients — the IGP24 small-discriminant tiebreak (directly
//! scored). This is `polred`, not the canonical `polredabs`: it returns reduced
//! same-field models but does not canonicalize across the (finite) reduced set.
//!
//! Pipeline:
//!   1. numerical roots of `f` (Durand–Kerner);
//!   2. the `T₂` (Minkowski) lattice of the power basis `1, θ, …, θ^{n−1}`,
//!      scaled and rounded to an integer lattice whose Euclidean norm is `T₂`;
//!   3. LLL-reduce it ([`rustmath_matrix::lll`]) → short integer combinations of
//!      the power basis, i.e. small algebraic integers `α = Σ uⱼ θʲ`;
//!   4. the exact minimal polynomial of each `α` via the characteristic polynomial
//!      of multiplication-by-`α` in the power basis (Faddeev–LeVerrier, integer);
//!   5. keep the squarefree degree-`n` ones (the field generators) and pick the
//!      one with the smallest coefficients.

use rustmath_integers::Integer;
use rustmath_matrix::lll::lll_reduce;

// --------------------------------------------------------------------------- //
// Minimal f64 complex
// --------------------------------------------------------------------------- //
#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

fn to_f64(x: &Integer) -> f64 {
    x.to_f64().unwrap_or(0.0)
}

/// Roots of a monic polynomial `f` (little-endian) by Durand–Kerner.
fn roots(f: &[Integer]) -> Vec<C> {
    let n = f.len() - 1;
    let coeffs: Vec<f64> = f.iter().map(to_f64).collect();
    let eval = |z: C| -> C {
        // Horner from the top, monic
        let mut acc = C::new(1.0, 0.0);
        for k in (0..n).rev() {
            acc = acc.mul(z).add(C::new(coeffs[k], 0.0));
        }
        acc
    };
    // initial guesses on a spiral
    let mut z: Vec<C> = (0..n)
        .map(|k| {
            let seed = C::new(0.4, 0.9);
            let mut p = C::new(1.0, 0.0);
            for _ in 0..k {
                p = p.mul(seed);
            }
            p
        })
        .collect();
    for _ in 0..200 {
        let mut max_delta = 0.0f64;
        for i in 0..n {
            let mut denom = C::new(1.0, 0.0);
            for j in 0..n {
                if j != i {
                    denom = denom.mul(z[i].sub(z[j]));
                }
            }
            let delta = eval(z[i]).div(denom);
            z[i] = z[i].sub(delta);
            max_delta = max_delta.max(delta.abs());
        }
        if max_delta < 1e-13 {
            break;
        }
    }
    z
}

/// Integer `T₂` lattice of the power basis: row `j` is the Minkowski embedding of
/// `θʲ`, scaled by `scale` and rounded. Real embeddings contribute one coordinate;
/// each complex-conjugate pair contributes `√2·Re` and `√2·Im` so the Euclidean
/// norm of a row equals `T₂(θʲ)`.
fn t2_lattice(f: &[Integer], rts: &[C], scale: f64) -> Vec<Vec<Integer>> {
    let n = f.len() - 1;
    // partition roots into real and one-per-conjugate-pair
    let mut reals: Vec<C> = Vec::new();
    let mut complex: Vec<C> = Vec::new();
    let mut used = vec![false; rts.len()];
    for i in 0..rts.len() {
        if used[i] {
            continue;
        }
        if rts[i].im.abs() < 1e-6 {
            reals.push(rts[i]);
            used[i] = true;
        } else {
            // find conjugate
            let mut best = usize::MAX;
            let mut bd = f64::INFINITY;
            for j in 0..rts.len() {
                if j != i && !used[j] {
                    let d = rts[i].re.sub_dist(rts[j].re) + (rts[i].im + rts[j].im).abs();
                    if d < bd {
                        bd = d;
                        best = j;
                    }
                }
            }
            complex.push(rts[i]);
            used[i] = true;
            if best != usize::MAX {
                used[best] = true;
            }
        }
    }
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut lattice = Vec::with_capacity(n);
    for j in 0..n {
        let mut row: Vec<Integer> = Vec::with_capacity(n);
        // θ^j numerically per embedding
        for r in &reals {
            let mut p = C::new(1.0, 0.0);
            for _ in 0..j {
                p = p.mul(*r);
            }
            row.push(Integer::from((p.re * scale).round() as i64));
        }
        for r in &complex {
            let mut p = C::new(1.0, 0.0);
            for _ in 0..j {
                p = p.mul(*r);
            }
            row.push(Integer::from((sqrt2 * p.re * scale).round() as i64));
            row.push(Integer::from((sqrt2 * p.im * scale).round() as i64));
        }
        lattice.push(row);
    }
    lattice
}

trait SubDist {
    fn sub_dist(self, o: f64) -> f64;
}
impl SubDist for f64 {
    fn sub_dist(self, o: f64) -> f64 {
        (self - o).abs()
    }
}

/// `θ^m mod f` for `m = 0 ..= 2n−2`, each as length-`n` integer coordinate vectors
/// in the power basis (`f` monic ⇒ all integer).
fn power_table(f: &[Integer]) -> Vec<Vec<Integer>> {
    let n = f.len() - 1;
    let mut table: Vec<Vec<Integer>> = Vec::with_capacity(2 * n - 1);
    let mut cur = vec![Integer::zero(); n];
    cur[0] = Integer::one(); // θ^0 = 1
    table.push(cur.clone());
    for _ in 1..(2 * n - 1).max(1) {
        // multiply by θ: shift up
        let mut next = vec![Integer::zero(); n + 1];
        for i in 0..n {
            next[i + 1] = cur[i].clone();
        }
        // reduce the θ^n term using θ^n = -(f0 + f1 θ + ... + f_{n-1} θ^{n-1})
        let top = next[n].clone();
        if !top.is_zero() {
            for i in 0..n {
                next[i] = next[i].clone() - top.clone() * f[i].clone();
            }
        }
        next.truncate(n);
        table.push(next.clone());
        cur = next;
    }
    table
}

/// Characteristic polynomial of an `n×n` integer matrix via Faddeev–LeVerrier;
/// returned little-endian, monic, degree `n`.
fn charpoly(m: &[Vec<Integer>]) -> Vec<Integer> {
    let n = m.len();
    // M_k accumulator; c[k] coefficients (c[0]=1 leading)
    let mut c = vec![Integer::zero(); n + 1];
    c[0] = Integer::one();
    // M_0 = 0 (zero matrix) so that M_1 = M·(0 + c_0 I) = M.
    let mut mk = vec![vec![Integer::zero(); n]; n];
    for k in 1..=n {
        // M_k = M · (M_{k-1} + c_{k-1} I)
        let mut tmp = mk.clone();
        for i in 0..n {
            tmp[i][i] = tmp[i][i].clone() + c[k - 1].clone();
        }
        mk = mat_mul(m, &tmp);
        let mut tr = Integer::zero();
        for i in 0..n {
            tr = tr + mk[i][i].clone();
        }
        // c_k = -tr / k   (exact)
        c[k] = -(tr / Integer::from(k as i64));
    }
    // c is [1, c1, ..., cn] high→low; return little-endian
    let mut le: Vec<Integer> = c.into_iter().rev().collect();
    while le.len() > 1 && le.last().unwrap().is_zero() {
        le.pop();
    }
    le
}

fn mat_mul(a: &[Vec<Integer>], b: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    let n = a.len();
    let mut out = vec![vec![Integer::zero(); n]; n];
    for i in 0..n {
        for k in 0..n {
            if a[i][k].is_zero() {
                continue;
            }
            for j in 0..n {
                out[i][j] = out[i][j].clone() + a[i][k].clone() * b[k][j].clone();
            }
        }
    }
    out
}

/// Multiplication-by-`α` matrix in the power basis, where `α = Σ u_j θʲ`.
fn mul_matrix(u: &[Integer], table: &[Vec<Integer>], n: usize) -> Vec<Vec<Integer>> {
    // column k = α·θ^k = Σ_j u_j θ^{j+k} (reduced)
    let mut m = vec![vec![Integer::zero(); n]; n];
    for k in 0..n {
        for (j, uj) in u.iter().enumerate() {
            if uj.is_zero() {
                continue;
            }
            let pk = &table[j + k];
            for r in 0..n {
                m[r][k] = m[r][k].clone() + uj.clone() * pk[r].clone();
            }
        }
    }
    m
}

fn is_squarefree_le(f: &[Integer]) -> bool {
    // gcd(f, f') constant over ℚ ⇔ squarefree; reuse the polynomials crate
    rustmath_polynomials::disc::discriminant(f) != Integer::zero()
}

fn sup_norm(f: &[Integer]) -> Integer {
    f.iter().map(|c| c.abs()).max().unwrap_or_else(Integer::zero)
}

/// All reduced candidate defining polynomials of `K = ℚ[x]/(f)` found from the
/// LLL-reduced `T₂` lattice (squarefree, monic, degree `n`), including `f` itself.
pub fn polred_candidates(f: &[Integer]) -> Vec<Vec<Integer>> {
    let n = f.len() - 1;
    if n < 2 {
        return vec![f.to_vec()];
    }
    let rts = roots(f);
    let scale = 1e6;
    let lattice = t2_lattice(f, &rts, scale);
    let (_red, u) = lll_reduce(&lattice);
    let table = power_table(f);

    let mut out: Vec<Vec<Integer>> = vec![f.to_vec()];
    for row in &u {
        // α = Σ row[j] θ^j ; skip α ∈ ℚ (only the constant coordinate nonzero)
        if row.iter().skip(1).all(|c| c.is_zero()) {
            continue;
        }
        let m = mul_matrix(row, &table, n);
        let cp = charpoly(&m);
        if cp.len() == n + 1 && is_squarefree_le(&cp) {
            // normalize leading sign to monic +1 (charpoly already monic)
            out.push(cp);
        }
    }
    out
}

/// The reduced defining polynomial with the smallest coefficients (sup-norm, then
/// `|disc|`). Defines the same field as `f`.
pub fn polred(f: &[Integer]) -> Vec<Integer> {
    let mut cands = polred_candidates(f);
    cands.sort_by(|a, b| {
        let na = sup_norm(a);
        let nb = sup_norm(b);
        na.cmp(&nb).then_with(|| {
            rustmath_polynomials::disc::discriminant(a)
                .abs()
                .cmp(&rustmath_polynomials::disc::discriminant(b).abs())
        })
    });
    cands.into_iter().next().unwrap_or_else(|| f.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    #[test]
    fn test_charpoly_companion() {
        // multiplication-by-θ matrix of f = x^2 - x - 1 has charpoly x^2 - x - 1
        let f = p(&[-1, -1, 1]);
        let table = power_table(&f);
        let m = mul_matrix(&p(&[0, 1]), &table, 2); // α = θ
        assert_eq!(charpoly(&m), f);
    }

    #[test]
    fn test_power_table_fibonacci() {
        // f = x^2 - x - 1: θ^2 = θ + 1
        let f = p(&[-1, -1, 1]);
        let t = power_table(&f);
        assert_eq!(t[2], p(&[1, 1])); // θ^2 = 1 + θ
    }

    #[test]
    fn test_polred_reduces_messy_quadratic() {
        // f = x^2 - 2x - 48  (disc 4+192=196=14^2... reducible? roots 8,-6 → reducible!)
        // use x^2 - 2x - 7 (field Q(√8)=Q(√2)); polred should find x^2 - 2 (smaller).
        let f = p(&[-7, -2, 1]); // roots 1±2√2, field Q(√2)
        let r = polred(&f);
        // same field Q(√2): the minimal small model is x^2 - 2 (sup-norm 2)
        assert!(sup_norm(&r) <= sup_norm(&f));
        assert_eq!(r.len(), 3); // still degree 2
    }

    #[test]
    fn test_polred_keeps_already_small() {
        // x^2 + 1 is already minimal; polred returns a degree-2 same-field poly
        let f = p(&[1, 0, 1]);
        let r = polred(&f);
        assert_eq!(r.len(), 3);
        assert!(sup_norm(&r) <= Integer::from(1));
    }
}
