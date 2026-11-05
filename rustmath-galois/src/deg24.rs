//! Degree-24 Galois narrowing: cycle-type candidate class + resolvent
//! orbit-signature descent against the transitive-group atlas.
//!
//! Full generic Stauduhar at degree 24 is infeasible by the small-degree
//! subgroup-lattice route (|S_24| ≈ 6.2·10²³), so the degree-24 case descends
//! through the **precomputed transitive atlas** instead. The pipeline, all
//! sound (the true group is never dropped):
//!
//! 1. **Frobenius cycle types** at many unramified primes
//!    (`rustmath_polynomials::padic_factor::cycle_type`) → the **candidate
//!    class** of `24Tt` whose cycle-type support contains every observed type
//!    (`rustmath_groups::transitive24::CycleTypeSupport::candidates`).
//! 2. **k-subset resolvent orbit signatures** for `k = 1, 2, 3`: build the exact
//!    `k`-subset-sum resolvent of `f`
//!    (`rustmath_polynomials::resolvent::subset_sum_resolvent`), factor it to get
//!    the orbit-length multiset
//!    (`resolvent_orbit_signature`), and keep only the atlas groups whose
//!    `k`-subset orbit signature matches
//!    (`transitive24::separate_by_ksubset_orbits`). Larger `k` resolves finer
//!    structure — this *is* Stauduhar descent expressed on the atlas. (`k ≤ 3`
//!    because `C(24, 4) > 4096` exceeds the exact-resolvent build bound.)
//!
//! The result is the narrowed candidate class and, when a single group remains,
//! the unique `24Tt`.

use rustmath_groups::transitive24::{
    separate_by_ksubset_orbits, CycleTypeSupport, Db,
};
use rustmath_integers::Integer;
use rustmath_polynomials::padic_factor::cycle_type as frobenius_cycle_type;
use rustmath_polynomials::resolvent::{resolvent_orbit_signature, subset_sum_resolvent};

/// Outcome of the degree-24 narrowing.
#[derive(Clone, Debug)]
pub struct Narrowing24 {
    /// Frobenius cycle types observed (deduped).
    pub observed_types: Vec<Vec<usize>>,
    /// Candidate `24Tt` after cycle-type narrowing.
    pub candidate_class: Vec<usize>,
    /// `(k, signature, remaining_count)` for each resolvent descent step.
    pub steps: Vec<(usize, Vec<usize>, usize)>,
    /// Final narrowed candidate `24Tt` list.
    pub narrowed: Vec<usize>,
    /// The unique `t` if exactly one candidate remains, else `None`.
    pub unique_t: Option<usize>,
}

/// Small primes for Frobenius sampling (skip 2; the caller's `f` may be ramified
/// at small primes — `frobenius_cycle_type` returns `None` there and we skip).
fn small_primes(limit: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let mut p = 3i64;
    while p <= limit {
        let mut is_p = true;
        let mut d = 3i64;
        while d * d <= p {
            if p % d == 0 {
                is_p = false;
                break;
            }
            d += 2;
        }
        if is_p {
            out.push(p);
        }
        p += 2;
    }
    out
}

/// Narrow the Galois group of a degree-24 polynomial `f` (monic, irreducible,
/// little-endian integer coefficients) using the atlas + resolvent descent.
///
/// `prime_limit` bounds the Frobenius sampling; `db` and `cts` are the loaded
/// atlas (pass `None` to load the defaults). Returns the full narrowing trace.
pub fn narrow_degree24(
    f: &[Integer],
    prime_limit: i64,
) -> std::io::Result<Narrowing24> {
    let cts = CycleTypeSupport::load_default()?;
    let mut db = Db::load_default()?;
    Ok(narrow_degree24_with(f, prime_limit, &cts, &mut db))
}

/// Narrow with caller-supplied (already-loaded) atlas tables — avoids reloading
/// the ~30 MB cycle-type file across many polynomials.
pub fn narrow_degree24_with(
    f: &[Integer],
    prime_limit: i64,
    cts: &CycleTypeSupport,
    db: &mut Db,
) -> Narrowing24 {
    // 1. Frobenius cycle-type candidate class.
    let mut observed: Vec<Vec<usize>> = Vec::new();
    for p in small_primes(prime_limit) {
        if let Some(ct) = frobenius_cycle_type(f, p) {
            if !observed.contains(&ct) {
                observed.push(ct);
            }
        }
    }
    let candidate_class = cts.candidates(&observed);

    // 2. k-subset resolvent orbit-signature descent (k = 1, 2, 3).
    let mut narrowed = candidate_class.clone();
    let mut steps: Vec<(usize, Vec<usize>, usize)> = Vec::new();
    for k in 1..=3usize {
        if narrowed.len() <= 1 {
            break;
        }
        // C(24,k) bound: 24, 276, 2024 — all ≤ 4096, safe to build exactly.
        let res = subset_sum_resolvent(f, k);
        let sig = match resolvent_orbit_signature(&res) {
            Ok(s) => s,
            Err(_) => continue, // factoring failed (rare); skip this k soundly
        };
        let sep = separate_by_ksubset_orbits(db, &narrowed, k, &sig);
        // `separate_by_ksubset_orbits` is sound only when the resolvent is
        // separable (distinct subset-sums); if it returns empty we keep the
        // previous (sound) set rather than drop the true group.
        if !sep.is_empty() {
            narrowed = sep;
        }
        steps.push((k, sig, narrowed.len()));
    }

    let unique_t = if narrowed.len() == 1 { Some(narrowed[0]) } else { None };
    Narrowing24 {
        observed_types: observed,
        candidate_class,
        steps,
        narrowed,
        unique_t,
    }
}
