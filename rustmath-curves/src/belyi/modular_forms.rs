//! KMSV §4 (Path A): power-series expansions of modular forms on Γ.
//!
//! A modular form f of weight k for Γ satisfies f(γz) = j(γ,z)^k f(z), j(γ,z)=cz+d.
//! Around p ∈ ℍ it has an expansion (eq 4.3) f(z) = (1−w)^k Σ_{n≥0} b_n w^n in the
//! disk coordinate w = w_p(z) = (z−p)/(z−p̄). The coefficients b are recovered as the
//! eigenvalue-1 eigenvector of the matrix A (eq 4.6): sampling Q points w_m on a
//! circle of radius ρ ⊇ D_Γ, reducing each into D_Γ via Algorithm 3.14 (γ_m, w'_m),
//!
//!   a_{nr} = (1/Q) Σ_m  j(γ_m, z_m)^{-k} · (w'_m)^r (1−w'_m)^k / ( w_m^n (1−w_m)^k ),
//!
//! and Ab ≈ b. The 1-eigenspace of A (equivalently the null space of A−I) is a basis
//! for S_k(Γ) = M_k(Γ) (Γ cocompact ⇒ no cusps). This f64 prototype validates the
//! plumbing; a high-precision (rug) port follows for recognition.

use super::coset_graph::CosetGraph;
use super::triangle_group::TriangleGroup;
use num_complex::Complex64;
use std::collections::HashSet;

const I: Complex64 = Complex64::new(0.0, 1.0);

/// Disk coordinate w_p(z) = (z−p)/(z−p̄), here p = z_a = i.
fn wp(z: Complex64) -> Complex64 {
    (z - I) / (z + I)
}
/// Inverse: z = i(1+w)/(1−w).
fn wp_inv(w: Complex64) -> Complex64 {
    I * (1.0 + w) / (1.0 - w)
}

/// Radius ρ of a circle about the origin (in the w_p chart) containing D_Γ: the max
/// |w_p(α_i·v)| over coset reps α_i and triangle vertices v.
pub fn domain_radius(tg: &TriangleGroup, cg: &CosetGraph) -> f64 {
    let verts = [tg.z_a, tg.z_b, tg.z_c, -tg.z_c.conj()];
    let mut rho: f64 = 0.0;
    for rep in &cg.reps {
        for &v in &verts {
            let r = wp(rep.apply(v)).norm();
            if r.is_finite() {
                rho = rho.max(r);
            }
        }
    }
    rho
}

/// Result of the eigenvector computation: a basis of S_k(Γ) as coefficient vectors
/// (each of length N+1), plus the sorted elimination pivot magnitudes (diagnostic —
/// the gap between "large" and "≈0" pivots reveals the null-space dimension).
pub struct ModularForms {
    pub k: i64,
    pub n: usize,
    pub basis: Vec<Vec<Complex64>>,
    pub pivots: Vec<f64>,
}

/// Assemble A (eq 4.6) and return the null space of A−I = a basis for S_k(Γ).
pub fn compute_forms(
    tg: &TriangleGroup,
    cg: &CosetGraph,
    k: i64,
    big_n: usize,
    q: usize,
    rho: f64,
    tol: f64,
) -> ModularForms {
    let dim = big_n + 1;
    let ku = k as u32;
    let mut a = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    let two_pi = 2.0 * std::f64::consts::PI;

    for m in 1..=q {
        let theta = two_pi * (m as f64) / (q as f64);
        let wm = Complex64::from_polar(rho, theta);
        let zm = wp_inv(wm);
        let (gamma, _) = cg.reduce(tg, zm);
        let zpm = gamma.apply(zm);
        let wpm = wp(zpm);
        // j(γ_m, z_m) = c·z_m + d
        let jm = Complex64::new(gamma.c, 0.0) * zm + Complex64::new(gamma.d, 0.0);
        let jm_neg_k = (Complex64::new(1.0, 0.0) / jm).powu(ku);
        let base = jm_neg_k * (1.0 - wpm).powu(ku) / (1.0 - wm).powu(ku);
        // (w'_m)^r and w_m^{-n}
        let mut wpm_r = vec![Complex64::new(1.0, 0.0); dim];
        for r in 1..dim {
            wpm_r[r] = wpm_r[r - 1] * wpm;
        }
        let inv_wm = Complex64::new(1.0, 0.0) / wm;
        let mut wm_neg_n = vec![Complex64::new(1.0, 0.0); dim];
        for n in 1..dim {
            wm_neg_n[n] = wm_neg_n[n - 1] * inv_wm;
        }
        for n in 0..dim {
            let cn = base * wm_neg_n[n];
            let row = &mut a[n];
            for r in 0..dim {
                row[r] += cn * wpm_r[r];
            }
        }
    }
    let qc = Complex64::new(q as f64, 0.0);
    for row in a.iter_mut() {
        for e in row.iter_mut() {
            *e /= qc;
        }
    }
    // Precondition: the entries span ρ^{-n} (from w_m^{-n}), a huge dynamic range that
    // wrecks f64 elimination. Scale a[n][r] ·= ρ^{n−r}: this fixes the −I diagonal
    // (ρ^{n}·ρ^{-n}=1) and brings the A part to O(1), preserving nullity (invertible
    // diagonal scalings on both sides).
    let mut rho_pow = vec![1.0f64; dim];
    for n in 1..dim {
        rho_pow[n] = rho_pow[n - 1] * rho;
    }
    for n in 0..dim {
        for r in 0..dim {
            a[n][r] *= rho_pow[n] / rho_pow[r];
        }
    }
    // M = A − I
    for i in 0..dim {
        a[i][i] -= Complex64::new(1.0, 0.0);
    }
    let (basis, pivots) = null_space(a, tol);
    ModularForms { k, n: big_n, basis, pivots }
}

/// Evaluate a modular form from its coefficient vector: f(z) = (1−w)^k Σ b_n w^n,
/// w = w_p(z) with p = z_a = i (eq 4.3).
pub fn eval_form(b: &[Complex64], k: i64, z: Complex64) -> Complex64 {
    let w = wp(z);
    let mut s = Complex64::new(0.0, 0.0);
    let mut wn = Complex64::new(1.0, 0.0);
    for &bn in b {
        s += bn * wn;
        wn *= w;
    }
    (1.0 - w).powu(k as u32) * s
}

/// Null space of a square complex matrix by Gauss–Jordan with partial pivoting.
/// Returns the basis vectors (one per free column) and the pivot magnitudes.
fn null_space(mut m: Vec<Vec<Complex64>>, tol: f64) -> (Vec<Vec<Complex64>>, Vec<f64>) {
    let rows = m.len();
    let cols = m[0].len();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut pivots_mag: Vec<f64> = Vec::new();
    let mut r = 0usize;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        // pivot = largest-magnitude entry in column c at or below row r
        let mut best = r;
        let mut best_val = m[r][c].norm();
        for i in (r + 1)..rows {
            let v = m[i][c].norm();
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        if best_val < tol {
            continue; // free column
        }
        m.swap(r, best);
        pivots_mag.push(best_val);
        pivot_cols.push(c);
        let piv = m[r][c];
        for j in c..cols {
            m[r][j] /= piv;
        }
        for i in 0..rows {
            if i != r {
                let f = m[i][c];
                if f.norm() > 0.0 {
                    for j in c..cols {
                        m[i][j] = m[i][j] - f * m[r][j];
                    }
                }
            }
        }
        r += 1;
    }
    let pivot_set: HashSet<usize> = pivot_cols.iter().copied().collect();
    let mut basis = Vec::new();
    for fc in 0..cols {
        if pivot_set.contains(&fc) {
            continue;
        }
        let mut v = vec![Complex64::new(0.0, 0.0); cols];
        v[fc] = Complex64::new(1.0, 0.0);
        for (ri, &pc) in pivot_cols.iter().enumerate() {
            v[pc] = -m[ri][fc];
        }
        basis.push(v);
    }
    pivots_mag.sort_by(|a, b| b.partial_cmp(a).unwrap());
    (basis, pivots_mag)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Paper Example 5.7 first triple: Δ(5,3,3), σ0=(1 5 4 3 2), σ1=(1 2 3), degree 5.
    fn setup_5_3_3() -> (TriangleGroup, CosetGraph) {
        let tg = TriangleGroup::new(5, 3, 3);
        let s0 = vec![4, 0, 1, 2, 3]; // (1 5 4 3 2) 0-indexed
        let s1 = vec![1, 2, 0, 3, 4]; // (1 2 3)     0-indexed
        let mut cg = CosetGraph::build(&tg, &s0, &s1);
        cg.compactify(&tg); // compact (Dirichlet) domain ⇒ small ρ
        (tg, cg)
    }

    #[test]
    fn radius_5_3_3_matches_paper() {
        let (tg, cg) = setup_5_3_3();
        let rho = domain_radius(&tg, &cg);
        // paper: fundamental domain contained in a circle of radius ρ = 0.528935…
        assert!(
            (rho - 0.528935).abs() < 1e-3,
            "ρ = {rho}, expected ≈ 0.528935"
        );
    }

    #[test]
    fn k2_no_modular_forms() {
        // dim S_2(Γ) = g = 0: A−I is full rank. This is the one dimension count f64
        // resolves cleanly (all pivots are O(1), well clear of any noise floor).
        let (tg, cg) = setup_5_3_3();
        let rho = domain_radius(&tg, &cg);
        let mf = compute_forms(&tg, &cg, 2, 28, 56, rho, 1e-2);
        assert_eq!(mf.basis.len(), 0, "dim S_2 should be 0");
    }

    // The exact null-space dimension for k ≥ 4 (dim S_4 = 1, dim S_6 = 3) cannot be
    // resolved in f64: the entries span ρ^{-N}, capping accuracy at a ~10⁻³ noise
    // floor (a high-precision port is needed — the paper runs at 10⁻³⁰). What f64 DOES
    // show robustly is the *emergence* of the eigenspace: the smallest eigen-pivot of
    // A−I drops sharply as modular forms appear (dim S_k = 0,1,3 for k = 2,4,6).
    #[test]
    fn eigenspace_emergence_5_3_3() {
        let (tg, cg) = setup_5_3_3();
        let rho = domain_radius(&tg, &cg);
        let (n, q) = (28usize, 56usize);
        // tol below the noise floor ⇒ null_space keeps every pivot; last() = smallest.
        let min_pivot = |k: i64| {
            let mf = compute_forms(&tg, &cg, k, n, q, rho, 1e-30);
            *mf.pivots.last().unwrap()
        };
        let (p2, p4, p6) = (min_pivot(2), min_pivot(4), min_pivot(6));
        assert!(p2 > 1e-1, "k=2 (no forms): min pivot {p2:.2e} should be O(1)");
        assert!(p6 < 1e-2, "k=6 (3 forms): min pivot {p6:.2e} should be small");
        assert!(
            p2 > p4 && p4 > p6,
            "smallest eigen-pivot should drop with k as forms appear: {p2:.2e}, {p4:.2e}, {p6:.2e}"
        );
    }
}
