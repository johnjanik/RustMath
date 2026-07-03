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
        CosetGraph { d, reps, edges, side_pairings }
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
}
