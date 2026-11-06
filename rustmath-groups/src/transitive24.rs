//! Degree-24 transitive groups and Frobenius cycle-type narrowing of Gal(f).
//!
//! Loads the 25,000 degree-24 transitive groups (LMFDB `gps_transitive`, n=24) from
//! `data/transitive_24.jsonl` — each with its permutation generators `⟨gens⟩ ≤ S₂₄`,
//! order, parity, primitivity. Given the Frobenius cycle types of a polynomial
//! `f ∈ ℤ[x]` at several unramified primes (computed natively by
//! `rustmath_polynomials::padic_factor::cycle_type`), [`candidates`] narrows the
//! Galois group to the transitive groups *consistent* with those cycle types —
//! **soundly**: the true group is never excluded.
//!
//! Filters, cheapest first: parity (a group `⊆ A₂₄` cannot have an odd Frobenius);
//! element order (a cycle type's `lcm` of parts must divide `|G|`); and, for groups
//! small enough to enumerate, exact membership of every observed type in the group's
//! cycle-type set. Larger groups are kept (sound) and separated by resolvents
//! (future work). This is the verifier-independent narrowing of the Phase-4 plan —
//! it lets a candidate's Galois group be checked without MAGMA or a submission slot.

use rustmath_integers::Integer;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;

pub const DEG: usize = 24;
/// A permutation of `{0,…,23}` (image list).
pub type Perm = [u8; DEG];

pub fn identity() -> Perm {
    let mut p = [0u8; DEG];
    for i in 0..DEG {
        p[i] = i as u8;
    }
    p
}

/// Compose: `(compose(a,b))[i] = a[b[i]]` (apply `b`, then `a`).
pub fn compose(a: &Perm, b: &Perm) -> Perm {
    let mut p = [0u8; DEG];
    for i in 0..DEG {
        p[i] = a[b[i] as usize];
    }
    p
}

/// Build a permutation from disjoint cycles given in **1-indexed** points.
pub fn perm_from_cycles(cycles: &[Vec<u8>]) -> Perm {
    let mut p = identity();
    for cyc in cycles {
        let m = cyc.len();
        for k in 0..m {
            let from = (cyc[k] - 1) as usize;
            let to = (cyc[(k + 1) % m] - 1) as usize;
            p[from] = to as u8;
        }
    }
    p
}

/// Cycle type: sorted (ascending) multiset of cycle lengths, including 1-cycles;
/// sums to 24. Matches `rustmath_polynomials::cycle_type`'s ordering.
pub fn cycle_type(p: &Perm) -> Vec<usize> {
    let mut seen = [false; DEG];
    let mut ct = Vec::new();
    for i in 0..DEG {
        if !seen[i] {
            let mut len = 0usize;
            let mut j = i;
            while !seen[j] {
                seen[j] = true;
                j = p[j] as usize;
                len += 1;
            }
            ct.push(len);
        }
    }
    ct.sort_unstable();
    ct
}

/// True if the cycle type is an odd permutation (`sign = (−1)^(24−#cycles)`).
pub fn is_odd_type(ct: &[usize]) -> bool {
    (DEG - ct.len()) % 2 == 1
}

/// `lcm` of the cycle lengths = the order of an element of this cycle type.
pub fn type_order(ct: &[usize]) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let mut l = 1u64;
    for &c in ct {
        l = l / gcd(l, c as u64) * c as u64;
    }
    l
}

/// Enumerate the group `⟨gens⟩` (BFS closure). `None` if it exceeds `cap` elements.
pub fn group_closure(gens: &[Perm], cap: usize) -> Option<HashSet<Perm>> {
    let mut set = HashSet::new();
    let id = identity();
    set.insert(id);
    let mut frontier = vec![id];
    while let Some(g) = frontier.pop() {
        for s in gens {
            let h = compose(s, &g);
            if set.insert(h) {
                if set.len() > cap {
                    return None;
                }
                frontier.push(h);
            }
        }
    }
    Some(set)
}

/// The set of cycle types occurring in `⟨gens⟩`. `None` if the group exceeds `cap`.
pub fn cycle_type_set(gens: &[Perm], cap: usize) -> Option<BTreeSet<Vec<usize>>> {
    let elements = group_closure(gens, cap)?;
    Some(elements.iter().map(cycle_type).collect())
}

/// One degree-24 transitive group `24Tt`.
#[derive(Clone)]
pub struct TransitiveGroup24 {
    pub t: usize,
    pub order: Integer,
    /// LMFDB parity: `1` ⇒ `G ⊆ A₂₄` (all elements even); `-1` ⇒ contains odd elements.
    pub parity: i64,
    pub primitive: bool,
    pub solvable: bool,
    pub gens: Vec<Perm>,
    /// cached cycle-type set (filled lazily for enumerable groups)
    ct_set: Option<BTreeSet<Vec<usize>>>,
    cached: bool,
}

/// The loaded degree-24 transitive-group database.
pub struct Db {
    pub groups: Vec<TransitiveGroup24>,
}

impl Db {
    /// Load from `data/transitive_24.jsonl` (one JSON object per line, with keys
    /// `t`, `order`, `parity`, `prim`, `solv`, `gens`; see `data/README.md`).
    ///
    /// A line that is not valid JSON or lacks `t`/`gens` (e.g. the cycle-type
    /// support file, which has `t`/`types`) is an `InvalidData` error, not a panic.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Db> {
        Db::parse_jsonl(&std::fs::read_to_string(path)?)
    }

    /// Parse the JSONL text of `transitive_24.jsonl` (see [`load`](Self::load)).
    fn parse_jsonl(text: &str) -> std::io::Result<Db> {
        fn bad(line_no: usize, what: &str) -> std::io::Error {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("transitive_24.jsonl line {}: {}", line_no, what),
            )
        }
        let mut groups = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let t = v["t"].as_u64().ok_or_else(|| bad(line_no, "missing integer `t`"))? as usize;
            let order = parse_int(v["order"].as_str().unwrap_or("0"));
            let parity = v["parity"].as_i64().unwrap_or(0);
            let primitive = v["prim"].as_i64().unwrap_or(0) == 1;
            let solvable = v["solv"].as_i64().unwrap_or(0) == 1;
            let gens_json = v["gens"]
                .as_array()
                .ok_or_else(|| bad(line_no, "missing `gens` (is this the cycle-type support file?)"))?;
            let mut gens: Vec<Perm> = Vec::with_capacity(gens_json.len());
            for g in gens_json {
                let mut cycles: Vec<Vec<u8>> = Vec::new();
                for c in g.as_array().ok_or_else(|| bad(line_no, "generator is not a list of cycles"))? {
                    let mut cyc = Vec::new();
                    for x in c.as_array().ok_or_else(|| bad(line_no, "cycle is not a list"))? {
                        match x.as_u64() {
                            Some(p) if (1..=DEG as u64).contains(&p) => cyc.push(p as u8),
                            _ => return Err(bad(line_no, "cycle entry is not a point in 1..=24")),
                        }
                    }
                    cycles.push(cyc);
                }
                gens.push(perm_from_cycles(&cycles));
            }
            groups.push(TransitiveGroup24 {
                t,
                order,
                parity,
                primitive,
                solvable,
                gens,
                ct_set: None,
                cached: false,
            });
        }
        Ok(Db { groups })
    }

    /// Default path for the generator-based DB: `rustmath-groups/data/transitive_24.jsonl`.
    ///
    /// That file (the full `⟨gens⟩` of all 25,000 groups) is large and gitignored,
    /// so a fresh checkout does not have it; regenerate it with GAP
    /// (`gap -q build_transitive_24.g`, see `data/README.md`). When it is absent
    /// this returns a `NotFound` error pointing at
    /// [`CycleTypeSupport::load_default`] (`data/transitive24_cycletypes.jsonl`),
    /// which is enough for Frobenius cycle-type narrowing.
    pub fn load_default() -> std::io::Result<Db> {
        Db::load_generator_atlas(&Path::new(env!("CARGO_MANIFEST_DIR")).join("data/transitive_24.jsonl"))
    }

    /// [`load`](Self::load) with a clear `NotFound` error when `path` is absent.
    fn load_generator_atlas(path: &Path) -> std::io::Result<Db> {
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "generator-based degree-24 DB not found at {} (gitignored; regenerate \
                     with `gap -q build_transitive_24.g`, see data/README.md). For \
                     Frobenius cycle-type narrowing alone, use \
                     CycleTypeSupport::load_default() (data/transitive24_cycletypes.jsonl).",
                    path.display()
                ),
            ));
        }
        Db::load(path)
    }
}

/// Parse a decimal string into an `Integer`.
fn parse_int(s: &str) -> Integer {
    let mut acc = Integer::zero();
    let ten = Integer::from(10);
    for ch in s.bytes() {
        if ch.is_ascii_digit() {
            acc = acc * ten.clone() + Integer::from((ch - b'0') as i64);
        }
    }
    acc
}

/// Candidate `24Tt` groups consistent with the `observed` Frobenius cycle types.
///
/// Sound: the true Galois group is always present. Groups with order `≤ enum_cap`
/// are checked exactly (cycle-type-set membership); larger groups pass the cheap
/// parity/order filters and are kept for resolvent separation. Results memoize the
/// per-group cycle-type set in `db`.
pub fn candidates(db: &mut Db, observed: &[Vec<usize>], enum_cap: usize) -> Vec<usize> {
    let any_odd = observed.iter().any(|t| is_odd_type(t));
    let obs_orders: Vec<u64> = observed.iter().map(|t| type_order(t)).collect();
    let mut out = Vec::new();
    for g in db.groups.iter_mut() {
        // parity: a group in A_24 cannot exhibit an odd Frobenius
        if g.parity == 1 && any_odd {
            continue;
        }
        // element order: lcm of each observed type must divide |G|
        if obs_orders.iter().any(|&o| !(g.order.clone() % Integer::from(o as i64)).is_zero()) {
            continue;
        }
        // exact check for enumerable groups
        let small = g.order <= Integer::from(enum_cap as i64);
        if small {
            if !g.cached {
                g.ct_set = cycle_type_set(&g.gens, enum_cap);
                g.cached = true;
            }
            if let Some(set) = &g.ct_set {
                if observed.iter().all(|t| set.contains(t)) {
                    out.push(g.t);
                }
                continue;
            }
        }
        out.push(g.t); // large / un-enumerable ⇒ keep (sound)
    }
    out
}

// --------------------------------------------------------------------------- //
// Resolvent separation: pair-orbit signatures (the group side of Module 18)
// --------------------------------------------------------------------------- //
/// Pair-orbit signature of a group: sorted orbit lengths of `⟨gens⟩` on 2-subsets
/// of the 24 points. Equal to the irreducible-factor degrees of the group's
/// pair-sum resolvent (`rustmath_polynomials::resolvent::pair_sum_resolvent`) when
/// that resolvent is separable.
pub fn pair_orbit_signature(gens: &[Perm]) -> Vec<usize> {
    crate::ksubset_orbits::orbit_lengths_on_pairs(gens, DEG)
}

/// k-subset orbit signature of a group: sorted orbit lengths of `⟨gens⟩` on
/// `k`-subsets of the 24 points = the irreducible-factor degrees of the group's
/// `k`-subset resolvent (`rustmath_polynomials::resolvent::subset_sum_resolvent`)
/// when separable. This is the Stauduhar resolvent for descending to the `k`-set
/// stabilizer `Sₖ × S_{24−k}`.
pub fn ksubset_orbit_signature(gens: &[Perm], k: usize) -> Vec<usize> {
    crate::ksubset_orbits::orbit_lengths_on_ksubsets(gens, DEG, k)
}

/// Separate a blind class by the **k-subset resolvent**: keep candidates whose
/// `k`-subset orbit signature equals `observed` (the factor-degree multiset of the
/// candidate polynomial's `k`-subset resolvent). Sound when the resolvent is
/// separable; larger `k` resolves finer structure (Stauduhar descent). Returns
/// sorted `t`'s.
pub fn separate_by_ksubset_orbits(
    db: &Db,
    cands: &[usize],
    k: usize,
    observed: &[usize],
) -> Vec<usize> {
    let mut want = observed.to_vec();
    want.sort_unstable();
    let mut out: Vec<usize> = cands
        .iter()
        .copied()
        .filter(|&t| {
            db.groups
                .iter()
                .find(|g| g.t == t)
                .map(|g| ksubset_orbit_signature(&g.gens, k) == want)
                .unwrap_or(false)
        })
        .collect();
    out.sort_unstable();
    out
}

/// Separate a blind class by the **pair-sum resolvent**: keep the candidate `24Tt`
/// whose pair-orbit signature equals `observed` — the factor-degree multiset of a
/// candidate polynomial's pair-sum resolvent.
///
/// Sound when the resolvent is separable (distinct pair-sums `αᵢ+αⱼ`): the true
/// group's pair-orbit signature then equals `observed`, so it is retained. Groups
/// in the blind class with a different pair-orbit signature are eliminated. Returns
/// sorted `t`'s; computing each signature is a 276-pair BFS, cheap even for large
/// groups.
pub fn separate_by_pair_orbits(db: &Db, cands: &[usize], observed: &[usize]) -> Vec<usize> {
    let mut want = observed.to_vec();
    want.sort_unstable();
    let mut out: Vec<usize> = cands
        .iter()
        .copied()
        .filter(|&t| {
            db.groups
                .iter()
                .find(|g| g.t == t)
                .map(|g| pair_orbit_signature(&g.gens) == want)
                .unwrap_or(false)
        })
        .collect();
    out.sort_unstable();
    out
}

// --------------------------------------------------------------------------- //
// Sharp narrowing via precomputed cycle-type support (the Frobenius-blind data)
// --------------------------------------------------------------------------- //
/// Per-group cycle-type **support** (the set of cycle types that occur in each
/// `24Tt`) — the complete Frobenius-blind data. Native [`group_closure`] cannot
/// enumerate the large groups, so the support is loaded from
/// `data/transitive24_cycletypes.jsonl` (derived from the LMFDB cycle-type
/// distribution; cross-checked against `group_closure` on enumerable groups).
pub struct CycleTypeSupport {
    pub by_t: HashMap<usize, HashSet<Vec<usize>>>,
}

impl CycleTypeSupport {
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<CycleTypeSupport> {
        let text = std::fs::read_to_string(path)?;
        let mut by_t = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let t = v["t"].as_u64().unwrap() as usize;
            let set: HashSet<Vec<usize>> = v["types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|ct| ct.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect())
                .collect();
            by_t.insert(t, set);
        }
        Ok(CycleTypeSupport { by_t })
    }

    pub fn load_default() -> std::io::Result<CycleTypeSupport> {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/transitive24_cycletypes.jsonl");
        CycleTypeSupport::load(path)
    }

    /// Groups `24Tt` consistent with the `observed` Frobenius cycle types: every
    /// observed type must occur in `G`. Sound (the true group is always included)
    /// and **sharp** — the candidate set is exactly the Frobenius-blind class once
    /// enough primes have been sampled. Returns sorted `t`'s.
    pub fn candidates(&self, observed: &[Vec<usize>]) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .by_t
            .iter()
            .filter(|(_, set)| observed.iter().all(|o| set.contains(o)))
            .map(|(&t, _)| t)
            .collect();
        out.sort_unstable();
        out
    }

    /// The Frobenius-blind class: groups whose cycle-type support is **exactly**
    /// the `observed` set. When enough primes have been sampled that `observed`
    /// equals the true group's support (Chebotarev — every conjugacy class is hit),
    /// this is the sharpest cycle-types-alone answer: Gal(f) lies here and no prime
    /// can separate the members. Resolvents are needed to go further.
    pub fn blind_class(&self, observed: &[Vec<usize>]) -> Vec<usize> {
        let obs: BTreeSet<Vec<usize>> = observed.iter().cloned().collect();
        let mut out: Vec<usize> = self
            .by_t
            .iter()
            .filter(|(_, set)| {
                set.len() == obs.len() && set.iter().all(|c| obs.contains(c))
            })
            .map(|(&t, _)| t)
            .collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_generator_atlas_is_a_clear_not_found_error() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/definitely_missing_transitive_24.jsonl");
        assert!(!path.exists());
        let err = Db::load_generator_atlas(&path).err().expect("missing file must be an error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("CycleTypeSupport::load_default"), "{err}");
    }

    #[test]
    fn wrong_schema_is_invalid_data_not_a_panic() {
        // A cycle-type support row: has `t` and `types`, no `gens`.
        let err = Db::parse_jsonl("{\"t\": 1, \"types\": [[24]]}\n")
            .err()
            .expect("wrong schema must be an error");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("gens"), "{err}");
        // A point outside 1..=24 is rejected rather than wrapping in u8 arithmetic.
        let err = Db::parse_jsonl("{\"t\": 1, \"gens\": [[[1, 25]]]}").err().unwrap();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn parses_a_generator_row() {
        // 24T1 in the format of data/transitive_24.jsonl: C24 generated by a 24-cycle.
        let row = format!(
            "{{\"label\":\"24T1\",\"t\":1,\"order\":\"24\",\"parity\":-1,\"prim\":0,\"solv\":1,\"gens\":[[{:?}]]}}\n",
            (1..=24).collect::<Vec<u8>>()
        );
        let db = Db::parse_jsonl(&row).expect("valid row");
        assert_eq!(db.groups.len(), 1);
        let g = &db.groups[0];
        assert_eq!((g.t, g.parity, g.primitive, g.solvable), (1, -1, false, true));
        assert_eq!(g.order, Integer::from(24));
        assert_eq!(group_closure(&g.gens, 100).unwrap().len(), 24);
    }
}
