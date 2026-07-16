//! Base and strong generating sets (BSGS) via the deterministic Schreier–Sims
//! algorithm, for permutation groups given by image-list generators.
//!
//! A [`StabilizerChain`] for `G = ⟨gens⟩ ≤ Sym({0,…,n−1})` stores a base
//! `B = (b₀, …, b_{k−1})` and, for each level `i`, the strong generators of the
//! pointwise stabilizer `G^{(i)} = G_{b₀,…,b_{i−1}}` together with the orbit
//! `b_i^{G^{(i)}}` and a Schreier transversal (explicit coset representatives).
//! The defining invariant, *proved* during construction by Schreier's lemma
//! (every Schreier generator of level `i` sifts to the identity through levels
//! `i+1, …`), is `G^{(i+1)} = Stab_{G^{(i)}}(b_i)`. It yields:
//!
//! - **order**: `|G| = ∏ᵢ |b_i^{G^{(i)}}|` (orbit–stabilizer, telescoping);
//! - **membership**: sifting/stripping `g` through the transversals — `g ∈ G`
//!   iff every base image lies in the level orbit and the final residue is the
//!   identity.
//!
//! The construction here is the classical deterministic Schreier–Sims
//! (Sims 1970; Holt–Eick–O'Brien, *Handbook of Computational Group Theory*,
//! §4.4.2), with the standard pruning: tree-edge Schreier generators and
//! identity residues are discarded, duplicate strong generators are never
//! added, and verification restarts at the deepest level whose generating set
//! changed. No randomization — the result is exact, not Monte Carlo.
//!
//! Permutations are image lists (`perm[i]` = image of `i`), the same
//! convention as [`crate::perm_predicates`]; inputs are validated (length and
//! bijectivity) with an honest `Err` on garbage.
//!
//! ```
//! use rustmath_groups::bsgs::group_order;
//!
//! // S4 = ⟨(0 1), (0 1 2 3)⟩ has order 24.
//! let gens = vec![vec![1, 0, 2, 3], vec![1, 2, 3, 0]];
//! assert_eq!(group_order(&gens, 4).unwrap().to_string(), "24");
//! ```

use rustmath_integers::Integer;

/// A permutation as an image list on `{0,…,n−1}` (same as
/// [`crate::perm_predicates::Perm`]).
type Perm = Vec<usize>;

/// Compose two permutations: `compose(a, b)[i] = a[b[i]]` (apply `b`, then `a`).
fn compose(a: &[usize], b: &[usize]) -> Perm {
    b.iter().map(|&i| a[i]).collect()
}

/// Inverse permutation: `inverse(a)[a[i]] = i`.
fn inverse(a: &[usize]) -> Perm {
    let mut inv = vec![0usize; a.len()];
    for (i, &ai) in a.iter().enumerate() {
        inv[ai] = i;
    }
    inv
}

fn is_identity(a: &[usize]) -> bool {
    a.iter().enumerate().all(|(i, &ai)| ai == i)
}

/// Validate that `g` is a permutation of `{0,…,degree−1}` given as an image
/// list: correct length, entries in range, no repeats.
fn validate_perm(g: &[usize], degree: usize) -> Result<(), String> {
    if g.len() != degree {
        return Err(format!(
            "permutation has length {} but the degree is {degree}",
            g.len()
        ));
    }
    let mut seen = vec![false; degree];
    for (i, &gi) in g.iter().enumerate() {
        if gi >= degree {
            return Err(format!("image {gi} of point {i} is out of range 0..{degree}"));
        }
        if seen[gi] {
            return Err(format!("not a bijection: image {gi} occurs twice"));
        }
        seen[gi] = true;
    }
    Ok(())
}

/// One level of the chain: base point, strong generators of
/// `G^{(i)} = G_{b₀,…,b_{i−1}}`, and the Schreier transversal of the orbit
/// `b_i^{G^{(i)}}` (`transversal[p]` maps `b_i ↦ p`).
struct Level {
    point: usize,
    gens: Vec<Perm>,
    transversal: Vec<Option<Perm>>,
    orbit: Vec<usize>,
}

impl Level {
    fn new(point: usize, degree: usize) -> Level {
        Level { point, gens: Vec::new(), transversal: vec![None; degree], orbit: Vec::new() }
    }

    /// Recompute the orbit of `self.point` under `self.gens` with a Schreier
    /// transversal (BFS, as in [`crate::perm_predicates::stabilizer`]).
    fn recompute_orbit(&mut self, degree: usize) {
        self.transversal = vec![None; degree];
        self.orbit.clear();
        self.transversal[self.point] = Some((0..degree).collect());
        self.orbit.push(self.point);
        let mut head = 0;
        while head < self.orbit.len() {
            let beta = self.orbit[head];
            head += 1;
            let ub = self.transversal[beta].clone().expect("orbit point has a transversal entry");
            for g in &self.gens {
                let gamma = g[beta];
                if self.transversal[gamma].is_none() {
                    self.transversal[gamma] = Some(compose(g, &ub));
                    self.orbit.push(gamma);
                }
            }
        }
    }
}

/// A base and strong generating set for a finite permutation group, built by
/// the deterministic Schreier–Sims algorithm. See the module docs for the
/// invariant and the guarantees behind [`order`](StabilizerChain::order) and
/// [`contains`](StabilizerChain::contains).
pub struct StabilizerChain {
    degree: usize,
    base: Vec<usize>,
    levels: Vec<Level>,
    /// All strong generators: the union of the level generating sets. The
    /// subset fixing `b₀,…,b_{i−1}` generates `G^{(i)}`.
    strong: Vec<Perm>,
}

impl StabilizerChain {
    /// Deterministic Schreier–Sims on validated generators. `Err` on any
    /// non-permutation input (wrong length / non-bijective); the identity and
    /// duplicate generators are harmless. An empty generator list (or all
    /// identities) yields the trivial group.
    pub fn new(gens: &[Vec<usize>], degree: usize) -> Result<Self, String> {
        for g in gens {
            validate_perm(g, degree)?;
        }
        let mut chain = StabilizerChain {
            degree,
            base: Vec::new(),
            levels: Vec::new(),
            strong: Vec::new(),
        };

        // Seed base and nested level generating sets: every non-identity
        // generator must move some base point.
        let mut seed: Vec<Perm> = Vec::new();
        for g in gens {
            if !is_identity(g) && !seed.contains(g) {
                seed.push(g.clone());
            }
        }
        for g in &seed {
            if chain.base.iter().all(|&b| g[b] == b) {
                let moved = (0..degree).find(|&p| g[p] != p).expect("non-identity moves a point");
                chain.base.push(moved);
                chain.levels.push(Level::new(moved, degree));
            }
        }
        // S⁽ⁱ⁾ = generators fixing b₀,…,b_{i−1} (nested downward).
        for g in seed {
            let stop = chain
                .base
                .iter()
                .position(|&b| g[b] != b)
                .expect("generator moves a base point");
            for level in chain.levels[..=stop].iter_mut() {
                level.gens.push(g.clone());
            }
        }

        chain.schreier_sims()?;
        // Full strong generating set: the union of the level sets (new strong
        // generators are recorded at levels ≥ 1 only; the union restricted to
        // elements fixing b₀,…,b_{i−1} is exactly S⁽ⁱ⁾ by the nesting of the
        // additions).
        for level in &chain.levels {
            for g in &level.gens {
                if !chain.strong.contains(g) {
                    chain.strong.push(g.clone());
                }
            }
        }
        Ok(chain)
    }

    /// Sift `g` through levels `start..`: repeatedly replace `g` by
    /// `u⁻¹·g` where `u` is the transversal element for `g(b_i)`. Returns
    /// `(j, residue)` where `j` is the first level whose orbit does not
    /// contain the base image (residue then *moves* `b_j`), or
    /// `(levels.len(), residue)` after a full pass (residue fixes every base
    /// point; it is the identity iff `g ∈ ⟨S^{(start)}⟩` given the invariant).
    fn sift_from(&self, g: &[usize], start: usize) -> (usize, Perm) {
        let mut residue: Perm = g.to_vec();
        for (j, level) in self.levels.iter().enumerate().skip(start) {
            let image = residue[level.point];
            match &level.transversal[image] {
                None => return (j, residue),
                Some(u) => residue = compose(&inverse(u), &residue),
            }
        }
        (self.levels.len(), residue)
    }

    /// The deterministic verification loop (Holt et al., SCHREIERSIMS):
    /// verify levels bottom-up; a Schreier generator whose sift residue is
    /// nontrivial becomes a new strong generator at levels `i+1..=j` (base
    /// extended if the residue fixes every base point) and verification
    /// restarts at level `j`. On success, Schreier's lemma proves
    /// `G^{(i+1)} = Stab_{G^{(i)}}(b_i)` at every level.
    fn schreier_sims(&mut self) -> Result<(), String> {
        if self.levels.is_empty() {
            return Ok(());
        }
        let degree = self.degree;
        let mut i = self.levels.len() - 1;
        loop {
            self.levels[i].recompute_orbit(degree);
            let mut fix_at: Option<usize> = None;

            // Test all Schreier generators u_{g(p)}⁻¹ · g · u_p of level i.
            'search: for oi in 0..self.levels[i].orbit.len() {
                let p = self.levels[i].orbit[oi];
                let up = self.levels[i].transversal[p]
                    .clone()
                    .expect("orbit point has a transversal entry");
                for gi in 0..self.levels[i].gens.len() {
                    let g = self.levels[i].gens[gi].clone();
                    let gp = g[p];
                    let ugp = self.levels[i].transversal[gp]
                        .as_ref()
                        .expect("orbit is closed under generators");
                    let schreier = compose(&inverse(ugp), &compose(&g, &up));
                    if is_identity(&schreier) {
                        continue; // pruning: tree edges and trivial Schreier gens
                    }
                    let (j, residue) = self.sift_from(&schreier, i + 1);
                    if is_identity(&residue) {
                        continue;
                    }
                    // New strong generator: fixes b₀..b_{j−1}, moves b_j
                    // (or fixes the whole base ⇒ extend it).
                    if j == self.levels.len() {
                        let moved = (0..degree)
                            .find(|&q| residue[q] != q)
                            .expect("nontrivial residue moves a point");
                        self.base.push(moved);
                        self.levels.push(Level::new(moved, degree));
                    }
                    for level in self.levels[i + 1..=j].iter_mut() {
                        if !level.gens.contains(&residue) {
                            level.gens.push(residue.clone());
                        }
                    }
                    fix_at = Some(j);
                    break 'search;
                }
            }

            match fix_at {
                Some(j) => i = j, // re-verify from the deepest changed level
                None => {
                    if i == 0 {
                        return Ok(());
                    }
                    i -= 1;
                }
            }
        }
    }

    /// `|G|` as the exact product of the level orbit lengths
    /// (orbit–stabilizer along the verified chain).
    pub fn order(&self) -> Integer {
        let mut n = Integer::one();
        for level in &self.levels {
            n = n * Integer::from(level.orbit.len() as i64);
        }
        n
    }

    /// Membership by sifting. A slice that is not a permutation of the right
    /// degree is not an element of the group (`false`), as is any permutation
    /// whose sift leaves a nontrivial residue.
    pub fn contains(&self, g: &[usize]) -> bool {
        if validate_perm(g, self.degree).is_err() {
            return false;
        }
        let (j, residue) = self.sift_from(g, 0);
        j == self.levels.len() && is_identity(&residue)
    }

    /// The base `(b₀, …, b_{k−1})`.
    pub fn base(&self) -> &[usize] {
        &self.base
    }

    /// The strong generating set (generators of `G`; every element fixing
    /// `b₀,…,b_{i−1}` lies in the level-`i` subset by construction).
    pub fn strong_generators(&self) -> &[Vec<usize>] {
        &self.strong
    }

    /// Degree of the action (number of points).
    pub fn degree(&self) -> usize {
        self.degree
    }
}

/// Order of `⟨gens⟩ ≤ Sym({0,…,degree−1})` — the simple entry point.
/// `Err` on invalid (non-permutation) generators.
pub fn group_order(gens: &[Vec<usize>], degree: usize) -> Result<Integer, String> {
    Ok(StabilizerChain::new(gens, degree)?.order())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ord(gens: &[Vec<usize>], n: usize) -> String {
        group_order(gens, n).unwrap().to_string()
    }

    /// n-cycle (0 1 2 … n−1).
    fn cycle(n: usize) -> Perm {
        (0..n).map(|i| (i + 1) % n).collect()
    }

    /// Transposition (0 1) on n points.
    fn transposition(n: usize) -> Perm {
        let mut t: Perm = (0..n).collect();
        t.swap(0, 1);
        t
    }

    fn sn_gens(n: usize) -> Vec<Perm> {
        vec![transposition(n), cycle(n)]
    }

    /// A_n from consecutive 3-cycles (i i+1 i+2), the standard generating set.
    fn an_gens(n: usize) -> Vec<Perm> {
        (0..n - 2)
            .map(|i| {
                let mut g: Perm = (0..n).collect();
                g[i] = i + 1;
                g[i + 1] = i + 2;
                g[i + 2] = i;
                g
            })
            .collect()
    }

    // |S_n| = n! and |A_n| = n!/2 for n = 3..8; factorials written out
    // independently.
    #[test]
    fn symmetric_and_alternating_orders() {
        let fact: [(usize, u64); 6] =
            [(3, 6), (4, 24), (5, 120), (6, 720), (7, 5040), (8, 40320)];
        for (n, f) in fact {
            assert_eq!(ord(&sn_gens(n), n), f.to_string(), "|S_{n}|");
            assert_eq!(ord(&an_gens(n), n), (f / 2).to_string(), "|A_{n}|");
        }
    }

    #[test]
    fn cyclic_and_dihedral_orders() {
        for n in 1..=12 {
            assert_eq!(ord(&[cycle(n)], n), n.to_string(), "|C_{n}|");
        }
        // D_n = ⟨rotation, reflection i ↦ −i mod n⟩, order 2n (n ≥ 3).
        for n in 3..=10 {
            let refl: Perm = (0..n).map(|i| (n - i) % n).collect();
            assert_eq!(ord(&[cycle(n), refl], n), (2 * n).to_string(), "|D_{n}|");
        }
    }

    #[test]
    fn trivial_group_and_identity_generators() {
        let chain = StabilizerChain::new(&[], 5).unwrap();
        assert!(chain.order().is_one());
        assert!(chain.base().is_empty());
        assert!(chain.strong_generators().is_empty());
        assert!(chain.contains(&[0, 1, 2, 3, 4]));
        assert!(!chain.contains(&[1, 0, 2, 3, 4]));
        // identity generators are ignored
        let chain = StabilizerChain::new(&[(0..5).collect()], 5).unwrap();
        assert!(chain.order().is_one());
        // degree 0: the empty group is trivial
        let chain = StabilizerChain::new(&[], 0).unwrap();
        assert!(chain.order().is_one());
        assert!(chain.contains(&[]));
    }

    #[test]
    fn invalid_generators_are_rejected() {
        assert!(StabilizerChain::new(&[vec![0, 0, 1]], 3).is_err()); // not bijective
        assert!(StabilizerChain::new(&[vec![0, 1]], 3).is_err()); // wrong length
        assert!(StabilizerChain::new(&[vec![0, 3]], 2).is_err()); // out of range
        assert!(group_order(&[vec![1, 1]], 2).is_err());
    }

    #[test]
    fn membership_in_s4_and_a4() {
        let s4 = StabilizerChain::new(&sn_gens(4), 4).unwrap();
        let a4 = StabilizerChain::new(&an_gens(4), 4).unwrap();
        assert_eq!(s4.order().to_string(), "24");
        assert_eq!(a4.order().to_string(), "12");
        // Every strong generator is a member.
        for g in s4.strong_generators() {
            assert!(s4.contains(g));
        }
        // A transposition (odd) is in S4, not in A4.
        let t = transposition(4);
        assert!(s4.contains(&t));
        assert!(!a4.contains(&t));
        // A product of A4 generators is in A4.
        let g = an_gens(4);
        let prod = compose(&g[0], &compose(&g[1], &g[0]));
        assert!(a4.contains(&prod));
        // Garbage is never a member.
        assert!(!a4.contains(&[0, 0, 1, 2]));
        assert!(!a4.contains(&[0, 1, 2]));
    }

    #[test]
    fn base_and_strong_generators_are_consistent() {
        let s5 = StabilizerChain::new(&sn_gens(5), 5).unwrap();
        assert_eq!(s5.degree(), 5);
        // Base is nonredundant for S5: orbit sizes multiply to 120 over 4 levels.
        assert_eq!(s5.base().len(), 4);
        // Level-i strong generators fixing the base prefix generate the
        // stabilizer; spot-check via the point stabilizer of b0 having index 5.
        let fixing_b0: Vec<Perm> = s5
            .strong_generators()
            .iter()
            .filter(|g| g[s5.base()[0]] == s5.base()[0])
            .cloned()
            .collect();
        assert_eq!(ord(&fixing_b0, 5), "24", "Stab_S5(b0) ≅ S4");
    }

    // ------------------------------------------------------------------ //
    // Mathieu groups: the classical generators (Carmichael's, as listed in
    // standard references for M23/M24), 1-indexed cycles
    //   a = (1 2 … 23)
    //   b = (3 17 10 7 9)(4 13 14 19 5)(8 18 11 12 23)(15 20 22 21 16)
    //   c = (1 24)(2 23)(3 12)(4 16)(5 18)(6 10)(7 20)(8 14)(9 21)(11 17)(13 22)(15 19)
    // with M23 = ⟨a,b⟩ (the stabilizer of 24) and M24 = ⟨a,b,c⟩.
    // Re-verified independently for this port with sympy's PermutationGroup
    // (its own Schreier–Sims): orders 10200960 and 244823040, cycle
    // structures 23·1 / 5⁴·1⁴ / 2¹², ⟨a,b⟩ fixes 24, ⟨a,b,c⟩ transitive.
    // ------------------------------------------------------------------ //

    fn mathieu_a() -> Perm {
        let mut g: Perm = (0..24).map(|i| i + 1).collect();
        g[22] = 0; // (0 1 … 22), fixes 23   [0-indexed]
        g[23] = 23;
        g
    }

    fn mathieu_b() -> Perm {
        vec![
            0, 1, 16, 12, 3, 5, 8, 17, 2, 6, 11, 22, 13, 18, 19, 14, 9, 10, 4, 21, 15, 20, 7,
            23,
        ]
    }

    fn mathieu_c() -> Perm {
        vec![
            23, 22, 11, 15, 17, 9, 19, 13, 20, 5, 16, 2, 21, 7, 18, 3, 10, 4, 14, 6, 8, 12, 1,
            0,
        ]
    }

    #[test]
    fn mathieu_m23_order() {
        // |M23| = 10200960; ⟨a,b⟩ fixes point 23 (0-indexed), so this is the
        // M23 ≤ M24 point stabilizer on 24 points.
        assert_eq!(ord(&[mathieu_a(), mathieu_b()], 24), "10200960");
    }

    #[test]
    fn mathieu_m24_order_and_membership() {
        let m24 =
            StabilizerChain::new(&[mathieu_a(), mathieu_b(), mathieu_c()], 24).unwrap();
        assert_eq!(m24.order().to_string(), "244823040");
        // Positive membership: products of the generators.
        let ab = compose(&mathieu_a(), &mathieu_b());
        let abc = compose(&ab, &mathieu_c());
        assert!(m24.contains(&ab));
        assert!(m24.contains(&abc));
        assert!(m24.contains(&inverse(&abc)));
        // Negative membership: M24 contains no transpositions (its
        // involutions have cycle type 2⁸1⁸ or 2¹²).
        assert!(!m24.contains(&transposition(24)));
        // A 23-cycle moving 23 but fixing 0: conjugate a by c — still in M24…
        let conj = compose(&mathieu_c(), &compose(&mathieu_a(), &inverse(&mathieu_c())));
        assert!(m24.contains(&conj));
        // …but a random odd permutation is not (M24 ⊆ A24).
        let mut odd: Perm = (0..24).collect();
        odd.swap(2, 17);
        assert!(!m24.contains(&odd));
    }

    #[test]
    fn m23_membership_respects_point_stabilizer() {
        let m23 = StabilizerChain::new(&[mathieu_a(), mathieu_b()], 24).unwrap();
        // c moves point 23, so c ∉ ⟨a,b⟩ even though c ∈ M24.
        assert!(!m23.contains(&mathieu_c()));
        assert!(m23.contains(&compose(&mathieu_a(), &mathieu_b())));
        // M23 on its natural 23 points gives the same order.
        let a23: Perm = mathieu_a()[..23].to_vec();
        let b23: Perm = mathieu_b()[..23].to_vec();
        assert_eq!(ord(&[a23, b23], 23), "10200960");
    }
}
