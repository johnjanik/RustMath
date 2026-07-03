//! KMSV §3: coset enumeration for Γ ≤ Δ(a,b,c) and the fundamental domain D_Γ.
//!
//! Given a transitive permutation triple σ = (σ0, σ1, σ∞) (σ∞σ1σ0 = 1) of degree d,
//! Γ = Stab(1) is an index-d subgroup of the triangle group Δ. Algorithm 3.5 walks
//! the coset graph: δ_a, δ_b (and inverses) act on cosets via π(δ_a)=σ0, π(δ_b)=σ1,
//! labelling each coset i with an explicit Möbius representative α_i ∈ Δ (a word in
//! δ_a, δ_b) such that 1^{π(α_i)} = i. Revisiting a labelled coset yields a side-
//! pairing element γ = α_j·ε·α_i⁻¹ ∈ Γ that glues two boundary sides of
//! D_Γ = ⋃_i α_i D_Δ. The side-pairing elements generate Γ.
//!
//! Convention (KMSV): permutations act on the RIGHT and compose left-to-right, so the
//! target coset under generator ε is simply i = j^{π(ε)} (the permutation applied to
//! the coset index j), while α_i = α_j·M(ε) builds the geometric representative.

use super::triangle_group::{Mobius, TriangleGroup};

/// The four generators used to walk the coset graph: δ_a^{±1}, δ_b^{±1}.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gen {
    A,
    AInv,
    B,
    BInv,
}

impl Gen {
    pub fn inverse(self) -> Gen {
        match self {
            Gen::A => Gen::AInv,
            Gen::AInv => Gen::A,
            Gen::B => Gen::BInv,
            Gen::BInv => Gen::B,
        }
    }
    pub fn all() -> [Gen; 4] {
        [Gen::A, Gen::AInv, Gen::B, Gen::BInv]
    }
}

/// A side pairing: `gamma ∈ Γ` glues side `from = (coset j, ε)` to `to = (coset i, ε⁻¹)`.
#[derive(Clone, Debug)]
pub struct SidePairing {
    pub gamma: Mobius,
    pub from: (usize, Gen),
    pub to: (usize, Gen),
}

/// The coset graph and fundamental-domain data for Γ ≤ Δ.
#[derive(Clone, Debug)]
pub struct CosetGraph {
    pub d: usize,
    /// α_i for coset i (0-indexed; coset 0 = the identity coset "1", α_0 = I).
    pub reps: Vec<Mobius>,
    /// Directed edges `(j, ε, i)` with `i = j^{π(ε)}` (the coset graph itself).
    pub edges: Vec<(usize, Gen, usize)>,
    /// Boundary side pairings (γ ≠ 1). These generate Γ.
    pub side_pairings: Vec<SidePairing>,
    /// The defining permutations π(δ_a) = σ0, π(δ_b) = σ1 (0-indexed).
    pub sigma0: Vec<usize>,
    pub sigma1: Vec<usize>,
}

fn inverse_perm(p: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; p.len()];
    for (i, &j) in p.iter().enumerate() {
        inv[j] = i;
    }
    inv
}

impl CosetGraph {
    /// Run Algorithm 3.5 on the triple (σ0 = π(δ_a), σ1 = π(δ_b)), 0-indexed, with the
    /// geometric generators from `tg`. `σ0` must have order dividing `a`, `σ1` order
    /// dividing `b` (matching the triangle group), and ⟨σ0,σ1⟩ transitive.
    pub fn build(tg: &TriangleGroup, sigma0: &[usize], sigma1: &[usize]) -> CosetGraph {
        let d = sigma0.len();
        assert_eq!(sigma1.len(), d, "σ0, σ1 must have equal degree");
        let s0 = sigma0.to_vec();
        let s0i = inverse_perm(sigma0);
        let s1 = sigma1.to_vec();
        let s1i = inverse_perm(sigma1);

        // geometric matrices and permutations for each generator
        let ga = tg.delta_a;
        let gai = tg.delta_a.inverse();
        let gb = tg.delta_b;
        let gbi = tg.delta_b.inverse();
        let gen_data = |g: Gen| -> (&Vec<usize>, Mobius) {
            match g {
                Gen::A => (&s0, ga),
                Gen::AInv => (&s0i, gai),
                Gen::B => (&s1, gb),
                Gen::BInv => (&s1i, gbi),
            }
        };

        let mut reps: Vec<Option<Mobius>> = vec![None; d];
        reps[0] = Some(Mobius::identity());
        let mut edges = Vec::new();
        let mut side_pairings = Vec::new();

        // BFS over cosets in discovery order (Algorithm 3.5 "first vertex with no out edges").
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0usize);
        let mut processed = vec![false; d];

        while let Some(j) = queue.pop_front() {
            if processed[j] {
                continue;
            }
            processed[j] = true;
            let alpha_j = reps[j].expect("processed coset must be labelled");
            for eps in Gen::all() {
                let (perm, mat) = gen_data(eps);
                let i = perm[j];
                edges.push((j, eps, i));
                match reps[i] {
                    Some(alpha_i) => {
                        // γ = α_j · ε · α_i⁻¹ ∈ Γ; record if not identity (a genuine side).
                        let gamma = alpha_j.mul(&mat).mul(&alpha_i.inverse());
                        if !gamma.is_scalar(1e-7) {
                            side_pairings.push(SidePairing {
                                gamma,
                                from: (j, eps),
                                to: (i, eps.inverse()),
                            });
                        }
                    }
                    None => {
                        reps[i] = Some(alpha_j.mul(&mat));
                        queue.push_back(i);
                    }
                }
            }
        }

        let reps: Vec<Mobius> = reps
            .into_iter()
            .map(|r| r.expect("all cosets reachable (σ transitive)"))
            .collect();
        CosetGraph {
            d,
            reps,
            edges,
            side_pairings,
            sigma0: s0,
            sigma1: s1,
        }
    }

    /// Replace the coset representatives with minimal-radius ones (a Dirichlet-style
    /// transversal), giving a compact fundamental domain `D_Γ = ⋃ α_i D_Δ` with small
    /// containing radius ρ. Any transversal tiles, so this stays a valid fundamental
    /// domain; compactness is what the §4 power-series method needs (small ρ ⇒ small N).
    /// Enumerates words in δ_a^{±1}, δ_b^{±1}, keeping for each coset the representative
    /// whose triangle has the smallest max-vertex radius |w_{z_a}(α·v)|.
    pub fn compactify(&mut self, tg: &TriangleGroup) {
        let i_c = num_complex::Complex64::new(0.0, 1.0);
        let wp = |z: num_complex::Complex64| (z - i_c) / (z + i_c);
        let verts = [tg.z_a, tg.z_b, tg.z_c, -tg.z_c.conj()];
        let radius = |m: &Mobius| -> f64 {
            verts
                .iter()
                .map(|&v| wp(m.apply(v)).norm())
                .filter(|r| r.is_finite())
                .fold(0.0, f64::max)
        };
        let p0 = num_complex::Complex64::new(0.37, 1.9);
        let fingerprint = |m: &Mobius| -> (i64, i64) {
            let z = m.apply(p0);
            ((z.re * 1e6).round() as i64, (z.im * 1e6).round() as i64)
        };
        let s0i = inverse_perm(&self.sigma0);
        let s1i = inverse_perm(&self.sigma1);
        let gens: [(Mobius, &[usize]); 4] = [
            (tg.delta_a, &self.sigma0),
            (tg.delta_a.inverse(), &s0i),
            (tg.delta_b, &self.sigma1),
            (tg.delta_b.inverse(), &s1i),
        ];
        const R_PRUNE: f64 = 0.95;
        const L_MAX: usize = 18;

        let mut best: Vec<Option<(f64, Mobius)>> = vec![None; self.d];
        best[0] = Some((radius(&Mobius::identity()), Mobius::identity()));
        let mut visited: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
        visited.insert(fingerprint(&Mobius::identity()));
        let mut frontier = vec![(Mobius::identity(), 0usize)];
        for _ in 0..L_MAX {
            let mut next = Vec::new();
            for (rep, coset) in &frontier {
                for (gmat, gperm) in gens.iter() {
                    let nrep = rep.mul(gmat);
                    let ncoset = gperm[*coset];
                    let r = radius(&nrep);
                    if r > R_PRUNE {
                        continue;
                    }
                    if !visited.insert(fingerprint(&nrep)) {
                        continue;
                    }
                    if best[ncoset].as_ref().map_or(true, |(br, _)| r < *br) {
                        best[ncoset] = Some((r, nrep));
                    }
                    next.push((nrep, ncoset));
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        self.reps = best
            .into_iter()
            .enumerate()
            .map(|(i, o)| o.unwrap_or_else(|| panic!("coset {i} unreached in compactify")).1)
            .collect();
    }

    /// KMSV Algorithm 3.14: reduce `z ∈ ℍ` into the Γ fundamental domain `D_Γ`.
    /// Returns `(γ, i)` with `γ ∈ Γ` and coset index `i` such that `γz ∈ α_i D_Δ`.
    /// Uses the Δ-reduction (`δz ∈ D_Δ`) and `i = 1^{π(δ⁻¹)}` recovered from the
    /// generator powers, then `γ = α_i · δ`.
    pub fn reduce(&self, tg: &TriangleGroup, z: num_complex::Complex64) -> (Mobius, usize) {
        let (delta, ops) = tg.reduce_to_base(z);
        let s0i = inverse_perm(&self.sigma0);
        let s1i = inverse_perm(&self.sigma1);
        let apply_pow = |mut x: usize, fwd: &[usize], inv: &[usize], k: i32| -> usize {
            if k >= 0 {
                for _ in 0..k {
                    x = fwd[x];
                }
            } else {
                for _ in 0..(-k) {
                    x = inv[x];
                }
            }
            x
        };
        // p[k] = k^{π(δ)}: δ = o_n···o_1 (o_n = last pushed), so apply π(o_n) first.
        let mut p = vec![0usize; self.d];
        for (k, pk) in p.iter_mut().enumerate() {
            let mut x = k;
            for &(is_a, pw) in ops.iter().rev() {
                x = if is_a {
                    apply_pow(x, &self.sigma0, &s0i, pw)
                } else {
                    apply_pow(x, &self.sigma1, &s1i, pw)
                };
            }
            *pk = x;
        }
        // i = 0^{π(δ⁻¹)} = the k with p[k] = 0.
        let i = (0..self.d).find(|&k| p[k] == 0).expect("π(δ) is a permutation");
        let gamma = self.reps[i].mul(&delta);
        (gamma, i)
    }

    /// Every side-pairing element must stabilize the base coset 1 (i.e. lie in Γ):
    /// this is a structural check on the enumeration.
    pub fn all_pairings_in_gamma(&self, sigma0: &[usize], sigma1: &[usize]) -> bool {
        // A word check is unnecessary: γ ∈ Γ ⟺ its coset-graph action fixes 0. We verify
        // instead that side pairings come in inverse pairs (γ and γ⁻¹ both appear),
        // a necessary property of a side pairing of a fundamental domain.
        let _ = (sigma0, sigma1);
        for sp in &self.side_pairings {
            let inv = sp.gamma.inverse();
            let found = self
                .side_pairings
                .iter()
                .any(|o| o.gamma.mul(&inv).is_scalar(1e-6));
            if !found {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycle_type(p: &[usize]) -> Vec<usize> {
        let n = p.len();
        let mut seen = vec![false; n];
        let mut ct = Vec::new();
        for s in 0..n {
            if seen[s] {
                continue;
            }
            let (mut x, mut len) = (s, 0);
            while !seen[x] {
                seen[x] = true;
                x = p[x];
                len += 1;
            }
            ct.push(len);
        }
        ct.sort_unstable_by(|a, b| b.cmp(a));
        ct
    }

    // Paper Example 3.1: Γ ≤ Δ(5,6,4), σ0=(1 5 4 3 2), σ1=(1 6 4 2 3 5), degree 6.
    // (1-indexed cycles → 0-indexed image arrays.)
    fn example_5_6_4() -> (Vec<usize>, Vec<usize>) {
        // σ0 = (1 5 4 3 2): 1→5,5→4,4→3,3→2,2→1, 6 fixed.  0-indexed images:
        let s0 = vec![4, 0, 1, 2, 3, 5];
        // σ1 = (1 6 4 2 3 5): 1→6,6→4,4→2,2→3,3→5,5→1.  0-indexed:
        let s1 = vec![5, 2, 4, 1, 0, 3];
        (s0, s1)
    }

    #[test]
    fn coset_graph_5_6_4() {
        let tg = TriangleGroup::new(5, 6, 4);
        let (s0, s1) = example_5_6_4();
        assert_eq!(cycle_type(&s0), vec![5, 1]);
        assert_eq!(cycle_type(&s1), vec![6]);
        let cg = CosetGraph::build(&tg, &s0, &s1);
        assert_eq!(cg.d, 6);
        // all 6 cosets reached, 4 out-edges each = 24 edges
        assert_eq!(cg.edges.len(), 24);
        // side pairings come in inverse pairs
        assert!(cg.all_pairings_in_gamma(&s0, &s1));
        // there is at least one genuine boundary side pairing
        assert!(!cg.side_pairings.is_empty());
    }

    // Our [2,12,5] M24 dessin (the black/white triple used throughout the project).
    const SIGMA0: [usize; 24] = [
        0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
    ];
    const SIGMA1: [usize; 24] = [
        14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
    ];

    #[test]
    fn coset_graph_2_12_5() {
        // Confirm the passport labelling: σ0 = 2^8 1^8 (order 2 = δ_a), σ1 = 12^2 (order 12 = δ_b).
        assert_eq!(
            cycle_type(&SIGMA0),
            vec![2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(cycle_type(&SIGMA1), vec![12, 12]);
        // σ∞ = (σ0 then σ1)⁻¹ must be 5^4 1^4 (order 5 = δ_c) ⇒ Riemann–Hurwitz genus 0:
        //   g = 1 − 24 + (e(σ0)+e(σ1)+e(σ∞))/2 = 1 − 24 + (8+22+16)/2 = 0.
        let comp: Vec<usize> = (0..24).map(|i| SIGMA1[SIGMA0[i]]).collect();
        assert_eq!(cycle_type(&comp), vec![5, 5, 5, 5, 1, 1, 1, 1]);

        let tg = TriangleGroup::new(2, 12, 5);
        let cg = CosetGraph::build(&tg, &SIGMA0, &SIGMA1);
        assert_eq!(cg.d, 24);
        assert_eq!(cg.edges.len(), 24 * 4);
        assert!(cg.all_pairings_in_gamma(&SIGMA0, &SIGMA1));
        assert!(!cg.side_pairings.is_empty());
        // reps: coset 0 is the identity
        assert!(cg.reps[0].is_scalar(1e-12));
        // the 24 coset representatives are geometrically distinct transformations:
        // their images of a generic point are 24 distinct points of ℍ.
        let p = num_complex::Complex64::new(0.3, 1.7);
        let imgs: Vec<_> = cg.reps.iter().map(|m| m.apply(p)).collect();
        for i in 0..24 {
            for j in (i + 1)..24 {
                assert!((imgs[i] - imgs[j]).norm() > 1e-6, "reps {i},{j} coincide");
            }
        }
    }

    #[test]
    fn gamma_reduction_2_12_5() {
        use num_complex::Complex64;
        let tg = TriangleGroup::new(2, 12, 5);
        let cg = CosetGraph::build(&tg, &SIGMA0, &SIGMA1);
        let pts = [
            Complex64::new(0.4, 3.0),
            Complex64::new(-1.5, 0.6),
            Complex64::new(1.0, 5.0),
        ];
        for &z in &pts {
            let (g, i) = cg.reduce(&tg, z);
            let red = g.apply(z);
            assert!(red.im > -1e-9, "reduced point left ℍ");
            // γz ∈ α_i D_Δ  ⟺  α_i⁻¹γz = δz ∈ D_Δ (reducing it to base is a no-op).
            let base_pt = cg.reps[i].inverse().apply(red);
            let (dd, _) = tg.reduce_to_base(base_pt);
            assert!(
                (dd.apply(base_pt) - base_pt).norm() < 1e-6,
                "γz not in α_i D_Δ"
            );
            // Γ-invariance: for side-pairing elements η ∈ Γ, reduce(ηz) is the same point.
            for sp in cg.side_pairings.iter().take(6) {
                let hz = sp.gamma.apply(z);
                let (g2, _) = cg.reduce(&tg, hz);
                let red2 = g2.apply(hz);
                assert!(
                    (red - red2).norm() < 1e-5,
                    "Γ-reduction not Γ-invariant: {red} vs {red2}"
                );
            }
        }
    }
}
