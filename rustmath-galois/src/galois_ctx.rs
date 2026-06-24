//! p-adic `GaloisCtx` — roots of a monic irreducible `f ∈ ℤ[x]` in an
//! **unramified** p-adic setting, **labeled so that Frobenius is an explicit
//! permutation** (mirrors OSCAR/Hecke's `GaloisCtx`).
//!
//! # Why this exists (the Stauduhar keystone)
//!
//! The degree-24 narrowing in [`crate::deg24`] currently builds a degree-2024
//! absolute resolvent over ℤ (the >300 s bottleneck). The OSCAR/Magma pipeline
//! avoids it by working *p-adically*: pick a good prime `p`, compute the roots
//! of `f` in an unramified extension of `Q_p` to controllable precision, and
//! **label the roots so that the Frobenius automorphism `x ↦ x^p` acts as an
//! explicit permutation `σ` of the labels**. With `σ` known, the Frobenius
//! short cosets ([`crate::short_coset`]) become usable: an evaluation-free
//! filter on the candidate subgroups, and the relative invariant then only has
//! to be evaluated p-adically on the *few* surviving cosets.
//!
//! # The construction
//!
//! Let `f` be monic, squarefree, degree `n`. We choose a prime `p` with
//! `p ∤ lc(f)` and `p ∤ disc(f)` (so `f mod p` is squarefree and the lift is
//! unramified). Factor `f mod p` into distinct monic irreducibles
//! `g_1, …, g_r` over `F_p`. A factor `g_i` of degree `d` has its `d` roots in
//! `GF(p^d)`, and Frobenius `y ↦ y^p` permutes them as a **single `d`-cycle**.
//!
//! For each `g_i` we:
//!
//! 1. Hensel-lift `g_i` to a monic `G_i ∈ (Z/p^k)[x]` dividing `f mod p^k`
//!    (via [`rustmath_polynomials::zp_hensel::hensel_lift_all`]).
//! 2. Work in the unramified ring `R_i = (Z/p^k)[x] / (G_i)`, representing each
//!    element by its coefficient vector of length `d` in the **power basis**
//!    `{1, x, …, x^{d-1}}`, reduced mod `p^k`. The class `x̄` is a root of `f`
//!    (since `G_i | f` in `R_i`), and it is our base root `r_0`.
//! 3. Enumerate the **Frobenius orbit** `r_0, σ(r_0), σ²(r_0), …, σ^{d-1}(r_0)`:
//!    `σ(r)` is the unique root of `G_i` in `R_i` whose reduction mod `p` equals
//!    `r^p mod (g_i, p)`. We compute `r^p` mod `p` in `GF(p^d)`, then lift it to
//!    a root of `G_i` in `R_i` by Newton's iteration
//!    `z ← z − G_i(z)·G_i'(z)^{-1}` over `R_i`.
//!
//! Concatenating the factors' orbits into the global label set `0..n-1`, the
//! global Frobenius permutation `σ` is the disjoint product of the per-factor
//! `d`-cycles, so **`σ`'s cycle type is exactly the multiset of mod-`p` factor
//! degrees, by construction.**
//!
//! # Representation
//!
//! A p-adic root is a [`PadicElt`]: which factor `R_i` it lives in (`factor`),
//! and its coefficient vector mod `p^k` in that factor's power basis. A rational
//! integer `m ∈ Z` is the element `[m mod p^k, 0, …, 0]` in *every* factor; the
//! [`GaloisCtx::is_integer`] test recognises this shape.
//!
//! # Scope / limitations
//!
//! * Only the **unramified** case is built (the prime is chosen so `p ∤ disc`),
//!   which is exactly what the Frobenius-labeling needs. Ramified primes are
//!   rejected during prime selection.
//! * `f` must be monic (the Stauduhar inputs are). Non-monic / non-squarefree
//!   inputs return `None` from [`galois_ctx`].
//! * Precision is a fixed power `p^k`; [`GaloisCtx::raise_precision`] re-lifts
//!   from scratch to a larger `k` (simple but not incremental).

use crate::perm::{self, Perm};
use rustmath_finitefields::ff_poly::{FiniteFieldElement, Gfpn};
use rustmath_integers::Integer;
use rustmath_polynomials::{disc, fp_factor, zp_hensel};

// ---------------------------------------------------------------------------
// Small Integer helpers
// ---------------------------------------------------------------------------

#[inline]
fn izero() -> Integer {
    Integer::from(0i64)
}

#[inline]
fn ione() -> Integer {
    Integer::from(1i64)
}

/// Reduce `v` into the canonical range `[0, m)` (m > 0).
fn redm(v: &Integer, m: &Integer) -> Integer {
    let r = v % m;
    &(&r + m) % m
}

/// Balanced representative of `c` mod `m` in `(-m/2, m/2]`.
fn balanced(c: &Integer, m: &Integer) -> Integer {
    let r = redm(c, m);
    let half = m / &Integer::from(2i64);
    if r > half {
        &r - m
    } else {
        r
    }
}

// ---------------------------------------------------------------------------
// Polynomial helpers over Z/p^k (little-endian Vec<Integer>, coeffs in [0,m))
// ---------------------------------------------------------------------------

/// Reduce a coefficient vector mod `m`, keeping a fixed length (no trimming).
fn vec_redm(a: &[Integer], m: &Integer) -> Vec<Integer> {
    a.iter().map(|c| redm(c, m)).collect()
}

// ---------------------------------------------------------------------------
// PadicElt — an element of one unramified factor ring R_i = (Z/p^k)[x]/(G_i)
// ---------------------------------------------------------------------------

/// An element of the unramified ring `R_i = (Z/p^k)[x] / (G_i)` for one
/// irreducible mod-`p` factor `g_i` of `f` (the factor that `G_i` lifts).
///
/// The coefficient vector has length `d = deg(g_i)` (the power-basis dimension),
/// with entries in `[0, p^k)`. A degree-1 factor gives `d = 1`, i.e. a plain
/// element of `Z/p^k` (a root in `Z_p`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PadicElt {
    /// Index of the factor `R_i` this element lives in.
    pub factor: usize,
    /// Power-basis coefficients (low degree first), length `d`, each in `[0,p^k)`.
    pub coeffs: Vec<Integer>,
}

impl PadicElt {
    /// The power-basis coefficient vector.
    pub fn coeffs(&self) -> &[Integer] {
        &self.coeffs
    }

    /// Which factor ring this element belongs to.
    pub fn factor_index(&self) -> usize {
        self.factor
    }
}

// ---------------------------------------------------------------------------
// Per-factor ring data
// ---------------------------------------------------------------------------

/// Bundles one factor's lifted modulus `G_i` and the field context `GF(p^d)`
/// used to drive the Frobenius orbit at the mod-`p` level.
#[derive(Clone, Debug)]
struct FactorRing {
    /// Degree `d` of the factor (= power-basis dimension).
    degree: usize,
    /// Lifted monic modulus `G_i` over `Z/p^k`, little-endian, length `d+1`,
    /// coeffs in `[0, p^k)`. `G_i[d] == 1`.
    modulus: Vec<Integer>,
    /// Derivative `G_i'` over `Z/p^k`, little-endian, coeffs in `[0, p^k)`.
    deriv: Vec<Integer>,
    /// The mod-`p` factor `g_i` as a `GF(p^d)` defining polynomial (length d+1,
    /// coeffs in `[0,p)`), used to build `Gfpn` elements for the Frobenius map.
    gf_modulus: Vec<Integer>,
}

impl FactorRing {
    /// Multiply two power-basis vectors in `R_i` (reduce mod `G_i` and `p^k`).
    fn ring_mul(&self, a: &[Integer], b: &[Integer], pk: &Integer) -> Vec<Integer> {
        let d = self.degree;
        let mut acc = vec![izero(); 2 * d - 1];
        for (i, ai) in a.iter().enumerate() {
            if ai.is_zero() {
                continue;
            }
            for (j, bj) in b.iter().enumerate() {
                if bj.is_zero() {
                    continue;
                }
                acc[i + j] = &acc[i + j] + &(ai * bj);
            }
        }
        for v in acc.iter_mut() {
            *v = redm(v, pk);
        }
        // Reduce modulo the monic G_i from the top down.
        for deg in (d..acc.len()).rev() {
            if acc[deg].is_zero() {
                continue;
            }
            let lead = acc[deg].clone();
            let base = deg - d;
            for i in 0..=d {
                acc[base + i] = redm(&(&acc[base + i] - &(&lead * &self.modulus[i])), pk);
            }
        }
        let mut out = vec![izero(); d];
        out[..d].clone_from_slice(&acc[..d]);
        out
    }

    /// Evaluate a little-endian `Z/p^k` polynomial `poly` at the ring element
    /// `x` (a power-basis vector) via Horner, in `R_i`.
    fn ring_eval(&self, poly: &[Integer], x: &[Integer], pk: &Integer) -> Vec<Integer> {
        let d = self.degree;
        let mut acc = vec![izero(); d];
        for c in poly.iter().rev() {
            // acc = acc * x + c
            acc = self.ring_mul(&acc, x, pk);
            acc[0] = redm(&(&acc[0] + c), pk);
        }
        acc
    }

    /// Invert a ring element `a` modulo `p` only (used to seed Newton): returns
    /// the inverse in `GF(p^d)` lifted to a `Z/p^k` vector (entries in `[0,p)`).
    /// `None` if `a ≡ 0 mod p`.
    fn ring_inverse_modp(&self, a: &[Integer], p: &Integer) -> Option<Vec<Integer>> {
        let gf_mod: Vec<Integer> = self.gf_modulus.clone();
        let a_modp: Vec<Integer> = a.iter().map(|c| redm(c, p)).collect();
        if a_modp.iter().all(|c| c.is_zero()) {
            return None;
        }
        let elt = Gfpn::new(a_modp, p.clone(), gf_mod);
        let inv = elt.invert().ok()?;
        // Gfpn::coeffs has length d, entries in [0,p).
        Some(inv.coeffs().to_vec())
    }
}

// ---------------------------------------------------------------------------
// GaloisCtx
// ---------------------------------------------------------------------------

/// A p-adic Galois context for a monic irreducible `f ∈ ℤ[x]`: the labeled
/// p-adic roots in an unramified setting plus the explicit Frobenius
/// permutation.
#[derive(Clone, Debug)]
pub struct GaloisCtx {
    /// The polynomial `f` (little-endian integer coefficients, monic).
    f: Vec<Integer>,
    /// The chosen good prime.
    p: i64,
    /// Precision exponent `k` (roots known mod `p^k`).
    prec_power: u32,
    /// `p^k` as an [`Integer`].
    pk: Integer,
    /// Degrees of the mod-`p` irreducible factors, in label-block order.
    factor_degrees: Vec<usize>,
    /// Per-factor ring data, parallel to `factor_degrees`.
    rings: Vec<FactorRing>,
    /// The explicit Frobenius permutation `σ` of the labels `0..n-1`.
    frobenius: Perm,
    /// The labeled p-adic roots, indexed `0..n-1`.
    roots: Vec<PadicElt>,
}

impl GaloisCtx {
    /// The chosen good prime.
    pub fn prime(&self) -> i64 {
        self.p
    }

    /// The precision exponent `k` (roots known mod `p^k`).
    pub fn prec_power(&self) -> u32 {
        self.prec_power
    }

    /// `p^k`.
    pub fn modulus(&self) -> &Integer {
        &self.pk
    }

    /// The degrees of the mod-`p` irreducible factors, in label-block order.
    /// Equal (as a multiset) to the Frobenius cycle type.
    pub fn factor_degrees(&self) -> &[usize] {
        &self.factor_degrees
    }

    /// The explicit Frobenius permutation `σ` of the root labels `0..n-1`.
    pub fn frobenius(&self) -> &Perm {
        &self.frobenius
    }

    /// The labeled p-adic roots, indexed by their global label.
    pub fn roots(&self) -> &[PadicElt] {
        &self.roots
    }

    /// Evaluate `f` at a [`PadicElt`] in its factor ring (returns the power-basis
    /// residue vector mod `p^k`). For a true root this is the zero vector.
    pub fn eval_f(&self, value: &PadicElt) -> Vec<Integer> {
        let ring = &self.rings[value.factor];
        let f_modpk = vec_redm(&self.f, &self.pk);
        ring.ring_eval(&f_modpk, &value.coeffs, &self.pk)
    }

    /// Recognise a p-adic value as a small rational integer `n`: the element must
    /// be a *scalar* (all higher power-basis coordinates `≡ 0 mod p^k`) and its
    /// constant coordinate must be congruent mod `p^k` to an integer `n` with
    /// `|n| ≤ height_bound`. Returns that `n`, else `None`.
    ///
    /// This mirrors OSCAR's `isinteger(ctx, bound, value)`: it is the test that
    /// decides whether a p-adically-evaluated relative invariant is a genuine
    /// rational integer (hence a descent certificate).
    pub fn is_integer(&self, value: &PadicElt, height_bound: &Integer) -> Option<Integer> {
        // All non-constant power-basis coordinates must vanish mod p^k.
        for c in value.coeffs.iter().skip(1) {
            if !redm(c, &self.pk).is_zero() {
                return None;
            }
        }
        let c0 = value
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(izero);
        let n = balanced(&c0, &self.pk);
        if n.abs() <= height_bound.abs() {
            Some(n)
        } else {
            None
        }
    }

    /// Construct an integer `n` as a [`PadicElt`] in the given factor ring.
    /// Useful for tests / for comparing an evaluated value against a known `n`.
    pub fn integer_elt(&self, n: &Integer, factor: usize) -> PadicElt {
        let d = self.rings[factor].degree;
        let mut coeffs = vec![izero(); d];
        coeffs[0] = redm(n, &self.pk);
        PadicElt { factor, coeffs }
    }

    /// Re-lift the context to a larger precision `new_k > k`. Re-runs the
    /// construction from scratch with the same prime; the labeling and Frobenius
    /// permutation are reproduced identically (the construction is deterministic
    /// for a fixed prime). Returns `false` if `new_k ≤ k` (no-op) or the relift
    /// fails.
    pub fn raise_precision(&mut self, new_k: u32) -> bool {
        if new_k <= self.prec_power {
            return false;
        }
        match build_ctx(&self.f, self.p, new_k) {
            Some(ctx) => {
                *self = ctx;
                true
            }
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Prime selection + top-level constructor
// ---------------------------------------------------------------------------

fn is_prime_i64(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2i64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// Strip trailing zero coefficients (keep at least one).
fn trim(p: &[Integer]) -> Vec<Integer> {
    let mut c = p.to_vec();
    while c.len() > 1 && c.last().map(|x| x.is_zero()).unwrap_or(false) {
        c.pop();
    }
    if c.is_empty() {
        c.push(izero());
    }
    c
}

/// Build a p-adic [`GaloisCtx`] for the monic integer polynomial `f`
/// (little-endian coefficients), choosing a good prime automatically and lifting
/// the roots to precision `p^target_prec_power`.
///
/// Returns `None` if `f` is not monic / not squarefree, has degree < 1, or no
/// good prime is found in the sampled range.
pub fn galois_ctx(f: &[Integer], target_prec_power: u32) -> Option<GaloisCtx> {
    let f = trim(f);
    let n = (f.len() as i64) - 1;
    if n < 1 {
        return None;
    }
    // Monic check (the Stauduhar inputs are monic).
    if !f[f.len() - 1].is_one() {
        return None;
    }
    let prec = target_prec_power.max(1);

    let disc_f = disc::discriminant(&f);
    if disc_f.is_zero() {
        return None; // not squarefree
    }

    // Sample small primes; require p ∤ lc(f) (lc=1 here) and p ∤ disc(f).
    let mut p: i64 = 2;
    let mut tried = 0;
    while tried < 200 {
        if is_prime_i64(p) {
            let pp = Integer::from(p);
            let lc_ok = !redm(&f[f.len() - 1], &pp).is_zero();
            let disc_ok = !redm(&disc_f, &pp).is_zero();
            if lc_ok && disc_ok && (p as i128) * (p as i128) <= i64::MAX as i128 {
                if let Some(ctx) = build_ctx(&f, p, prec) {
                    return Some(ctx);
                }
            }
            tried += 1;
        }
        p += 1;
        if p > 1_000_000 {
            break;
        }
    }
    None
}

/// Build a [`GaloisCtx`] for a *given* prime `p` and precision `k`. Assumes `p`
/// is good (squarefree mod p, p ∤ lc). Returns `None` if the lift fails.
pub fn build_ctx(f: &[Integer], p: i64, k: u32) -> Option<GaloisCtx> {
    let f = trim(f);
    let n = (f.len() as i64) - 1;
    if n < 1 {
        return None;
    }
    let pp = Integer::from(p);
    let pk = pp.pow(k);

    // 1. Factor f mod p into distinct monic irreducibles over F_p.
    let f_fp: Vec<i64> = f
        .iter()
        .map(|c| {
            let r = c % &pp;
            (&(&r + &pp) % &pp).to_i64()
        })
        .collect();
    let f_fp = fp_factor::trim(&f_fp);
    let mut factors_fp = fp_factor::factor(&f_fp, p);
    if factors_fp.is_empty() {
        return None;
    }
    // The factors must be distinct (squarefree mod p) and their degrees sum to n.
    let deg_sum: i64 = factors_fp.iter().map(|g| fp_factor::degree(g)).sum();
    if deg_sum != n {
        // Repeated factor mod p ⇒ ramified / not squarefree at p; reject.
        return None;
    }
    // Deterministic order: by (degree, lexicographic) so labeling is stable.
    factors_fp.sort_by(|a, b| {
        let da = fp_factor::degree(a);
        let db = fp_factor::degree(b);
        da.cmp(&db).then_with(|| a.cmp(b))
    });

    // 2. Hensel-lift all factors to monic G_i over Z/p^k.
    let lifted = zp_hensel::hensel_lift_all(&f, &factors_fp, p, k)?;
    if lifted.len() != factors_fp.len() {
        return None;
    }

    // 3. Per factor: build ring data, enumerate the Frobenius orbit, lift roots.
    let mut factor_degrees: Vec<usize> = Vec::new();
    let mut rings: Vec<FactorRing> = Vec::new();
    let mut roots: Vec<PadicElt> = Vec::new();
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    let mut next_label = 0usize;

    for (fi, (gfp, gi_lift)) in factors_fp.iter().zip(lifted.iter()).enumerate() {
        let d = fp_factor::degree(gfp) as usize;
        debug_assert!(d >= 1);

        // GF(p^d) defining poly = g_i (monic, coeffs in [0,p)), length d+1.
        let mut gf_modulus: Vec<Integer> = gfp.iter().map(|&c| Integer::from(c)).collect();
        while gf_modulus.len() < d + 1 {
            gf_modulus.push(izero());
        }

        // Lifted modulus G_i, made monic length d+1, coeffs in [0, p^k).
        let mut modulus: Vec<Integer> = vec_redm(gi_lift, &pk);
        while modulus.len() < d + 1 {
            modulus.push(izero());
        }
        modulus.truncate(d + 1);
        // Force monic (Hensel preserves monicity; this guards rounding).
        modulus[d] = ione();

        // Derivative G_i' over Z/p^k.
        let mut deriv: Vec<Integer> = Vec::with_capacity(d);
        for i in 1..modulus.len() {
            deriv.push(redm(&(&modulus[i] * &Integer::from(i as i64)), &pk));
        }
        if deriv.is_empty() {
            deriv.push(izero());
        }

        let ring = FactorRing {
            degree: d,
            modulus: modulus.clone(),
            deriv,
            gf_modulus: gf_modulus.clone(),
        };

        // Base root r_0 = x̄ in R_i = power-basis vector [0,1,0,...].
        let mut r0 = vec![izero(); d];
        if d >= 2 {
            r0[1] = ione();
        } else {
            // d == 1: G_i = x + c, the root is -c mod p^k (a scalar in Z/p^k).
            // x̄ in (Z/p^k)[x]/(x + c) equals -c.
            r0[0] = redm(&(-&ring.modulus[0]), &pk);
        }

        // Enumerate the Frobenius orbit r_0, σ(r_0), ..., σ^{d-1}(r_0).
        let mut orbit_elts: Vec<Vec<Integer>> = Vec::with_capacity(d);
        let mut cur = r0.clone();
        for _ in 0..d {
            orbit_elts.push(cur.clone());
            if d >= 2 {
                cur = frobenius_step(&ring, &cur, p, &pp, &pk);
            }
        }

        // Assign global labels for this block (in Frobenius-cycle order) and
        // record the d-cycle (next_label, next_label+1, ..., next_label+d-1).
        let mut cyc: Vec<usize> = Vec::with_capacity(d);
        for elt in orbit_elts.into_iter() {
            roots.push(PadicElt { factor: fi, coeffs: elt });
            cyc.push(next_label);
            next_label += 1;
        }
        cycles.push(cyc);
        factor_degrees.push(d);
        rings.push(ring);
    }

    let total = next_label;
    if total != n as usize {
        return None;
    }
    let frobenius = perm::from_cycles(total, &cycles);

    Some(GaloisCtx {
        f,
        p,
        prec_power: k,
        pk,
        factor_degrees,
        rings,
        frobenius,
        roots,
    })
}

/// Frobenius step in one factor ring: given a root `r` of `G_i` in
/// `R_i = (Z/p^k)[x]/(G_i)` (power-basis vector mod `p^k`), return the next root
/// in the Frobenius cycle — the unique root of `G_i` whose reduction mod `p` is
/// `r^p mod (g_i, p)`.
///
/// Method: compute the target residue `t = r^p` in `GF(p^d)` (mod `p`), then
/// Newton-lift it to a root of `G_i` mod `p^k` via `z ← z − G_i(z)·G_i'(z)^{-1}`,
/// seeding `G_i'(z)^{-1}` with the mod-`p` inverse and refining by linear Hensel.
fn frobenius_step(
    ring: &FactorRing,
    r: &[Integer],
    p: i64,
    pp: &Integer,
    pk: &Integer,
) -> Vec<Integer> {
    let d = ring.degree;
    // t = r^p in GF(p^d) (entries in [0,p)).
    let r_modp: Vec<Integer> = r.iter().map(|c| redm(c, pp)).collect();
    let elt = Gfpn::new(r_modp, pp.clone(), ring.gf_modulus.clone());
    let t = elt.pow(&Integer::from(p)); // y ↦ y^p, the GF(p^d) Frobenius
    let mut z: Vec<Integer> = t.coeffs().to_vec(); // length d, entries in [0,p)
    while z.len() < d {
        z.push(izero());
    }
    z.truncate(d);

    // Newton-lift z (a simple root of G_i mod p) to a root of G_i mod p^k.
    // Compute G_i'(z)^{-1} mod p once (z is a simple root since disc != 0).
    let dz_modp = ring.ring_eval(&ring.deriv, &z, pp); // G_i'(z) mod p
    let dz_inv_modp = match ring.ring_inverse_modp(&dz_modp, pp) {
        Some(v) => v,
        None => return z, // should not happen (simple root); fall back to mod-p
    };

    // Linear Hensel: one power of p at a time, with the fixed mod-p inverse.
    let mut cur_prec: u32 = 1;
    let target = exponent_of(pk, p);
    while cur_prec < target {
        let next = cur_prec + 1;
        let mod_next = pp.pow(next);
        // f(z) mod p^next (a power-basis vector).
        let g_modpk = vec_redm(&ring.modulus, &mod_next);
        let fz = ring.ring_eval_mod(&g_modpk, &z, &mod_next);
        // correction = f(z) * G_i'(z)^{-1}  (use the mod-p inverse; converges).
        let corr = ring.ring_mul(&fz, &dz_inv_modp, &mod_next);
        for i in 0..d {
            z[i] = redm(&(&z[i] - &corr[i]), &mod_next);
        }
        cur_prec = next;
    }
    vec_redm(&z, pk)
}

/// The exponent `k` such that `p^k == pk`. (pk was built as `p^k`.)
fn exponent_of(pk: &Integer, p: i64) -> u32 {
    let pp = Integer::from(p);
    let mut acc = ione();
    let mut k = 0u32;
    while &acc < pk {
        acc = &acc * &pp;
        k += 1;
    }
    k
}

impl FactorRing {
    /// `ring_eval` against an explicit modulus power (for Newton at intermediate
    /// precision `p^next`). Same as [`FactorRing::ring_eval`] but reducing mod
    /// the supplied modulus `m`.
    fn ring_eval_mod(&self, poly: &[Integer], x: &[Integer], m: &Integer) -> Vec<Integer> {
        let d = self.degree;
        let mut acc = vec![izero(); d];
        for c in poly.iter().rev() {
            acc = self.ring_mul(&acc, x, m);
            acc[0] = redm(&(&acc[0] + c), m);
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// Tests (known answers; run with `cargo test -p rustmath-galois`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&c| Integer::from(c)).collect()
    }

    /// f(root) ≡ 0 mod p^k for every labeled root.
    fn assert_roots_vanish(ctx: &GaloisCtx) {
        for r in ctx.roots() {
            let v = ctx.eval_f(r);
            for c in &v {
                assert!(
                    redm(c, ctx.modulus()).is_zero(),
                    "f(root) != 0 mod p^k: coeff {:?}, factor {}",
                    c,
                    r.factor
                );
            }
        }
    }

    /// frobenius cycle type == sorted mod-p factor degrees (the invariant).
    fn assert_frobenius_invariant(ctx: &GaloisCtx) {
        let ct = perm::cycle_type(ctx.frobenius());
        let mut degs = ctx.factor_degrees().to_vec();
        degs.sort_unstable();
        assert_eq!(ct, degs, "frobenius cycle type != mod-p factor degrees");
    }

    // ---- x^2 + 1, p = 7 (inert): one factor of degree 2, cycle type [2] ----

    #[test]
    fn x2p1_p7_inert() {
        // x^2 + 1 mod 7: -1 is not a QR mod 7 ⇒ irreducible, single factor deg 2.
        let ctx = build_ctx(&ints(&[1, 0, 1]), 7, 6).unwrap();
        assert_eq!(ctx.factor_degrees(), &[2]);
        assert_eq!(perm::cycle_type(ctx.frobenius()), vec![2]);
        assert_frobenius_invariant(&ctx);
        assert_roots_vanish(&ctx);
        // Frobenius is a 2-cycle (0 1): σ(0)=1, σ(1)=0.
        assert_eq!(ctx.frobenius(), &vec![1usize, 0usize]);
        // The two roots are Frobenius-conjugate and distinct.
        assert_ne!(ctx.roots()[0], ctx.roots()[1]);
    }

    // ---- x^2 + 1, p = 5 (splits): (x-2)(x-3), cycle type [1,1] ----

    #[test]
    fn x2p1_p5_split() {
        // x^2 + 1 mod 5 = (x+2)(x+3) = (x-3)(x-2): two linear factors.
        let ctx = build_ctx(&ints(&[1, 0, 1]), 5, 6).unwrap();
        assert_eq!(ctx.factor_degrees(), &[1, 1]);
        assert_eq!(perm::cycle_type(ctx.frobenius()), vec![1, 1]);
        assert_frobenius_invariant(&ctx);
        assert_roots_vanish(&ctx);
        // Frobenius is the identity (each root is in Z_5, fixed by Frobenius).
        assert_eq!(ctx.frobenius(), &perm::identity(2));
        // Roots are ±2 mod 5 (i.e. residues 2 and 3 mod 5).
        let mut res: Vec<i64> = ctx
            .roots()
            .iter()
            .map(|r| redm(&r.coeffs[0], &Integer::from(5)).to_i64())
            .collect();
        res.sort();
        assert_eq!(res, vec![2, 3]);
        // r^2 ≡ -1 mod 5^k (so r ≡ ±2 to full precision).
        let pk = ctx.modulus().clone();
        for r in ctx.roots() {
            let c = &r.coeffs[0];
            let sq = redm(&(c * c), &pk);
            assert_eq!(redm(&(&sq + &ione()), &pk), izero());
        }
    }

    // ---- x^3 - 2, p = 7: factor degrees == cycle type, roots vanish ----

    #[test]
    fn x3m2_p7() {
        // x^3 - 2 mod 7: 2 is not a cube mod 7 (cubes mod 7 are {0,1,6}),
        // so there is one linear factor (the cube root of 2 in F_7? none) —
        // actually 2 has no cube root in F_7, giving an irreducible cubic.
        // Whatever the split, the invariant must hold and roots must vanish.
        let ctx = build_ctx(&ints(&[-2, 0, 0, 1]), 7, 5).unwrap();
        assert_frobenius_invariant(&ctx);
        assert_roots_vanish(&ctx);
        // Degrees sum to 3.
        let s: usize = ctx.factor_degrees().iter().sum();
        assert_eq!(s, 3);
    }

    // ---- degree-4 with mod-p split 1+1+2 ⇒ cycle type [1,1,2] ----

    #[test]
    fn deg4_split_1_1_2() {
        // Construct f over Z whose reduction mod p factors as (linear)(linear)(quad).
        // Use f = (x-1)(x-2)(x^2 + 1) over Z; mod 7, x^2+1 is irreducible (deg 2),
        // and (x-1),(x-2) are distinct linear ⇒ degrees [1,1,2].
        // f = (x^2 - 3x + 2)(x^2 + 1) = x^4 - 3x^3 + 3x^2 - 3x + 2.
        let f = ints(&[2, -3, 3, -3, 1]);
        let ctx = build_ctx(&f, 7, 5).unwrap();
        let mut degs = ctx.factor_degrees().to_vec();
        degs.sort_unstable();
        assert_eq!(degs, vec![1, 1, 2]);
        assert_eq!(perm::cycle_type(ctx.frobenius()), vec![1, 1, 2]);
        assert_frobenius_invariant(&ctx);
        assert_roots_vanish(&ctx);
        // The degree-2 block forms a 2-cycle; the two linear roots are fixed.
        // Exactly two fixed points and one 2-cycle.
        let frob = ctx.frobenius();
        let fixed: usize = (0..frob.len()).filter(|&i| frob[i] == i).count();
        assert_eq!(fixed, 2);
    }

    // ---- is_integer: scalar recognized, generic root not ----

    #[test]
    fn is_integer_recognition() {
        // f = (x-1)(x-2)(x^2+1); the degree-1 roots are the integers 1 and 2.
        let f = ints(&[2, -3, 3, -3, 1]);
        let ctx = build_ctx(&f, 7, 5).unwrap();
        let bound = Integer::from(100);

        // Each degree-1 root recognizes as the small integer 1 or 2.
        let mut recognized: Vec<i64> = Vec::new();
        for r in ctx.roots() {
            if ctx.factor_degrees()[r.factor] == 1 {
                let n = ctx.is_integer(r, &bound).expect("linear root is an integer");
                recognized.push(n.to_i64());
            } else {
                // A root of x^2+1 in GF(7^2) is NOT a rational integer.
                assert!(
                    ctx.is_integer(r, &bound).is_none(),
                    "quadratic root wrongly recognized as integer"
                );
            }
        }
        recognized.sort();
        assert_eq!(recognized, vec![1, 2]);

        // An explicitly built integer element recognizes back to itself.
        let elt = ctx.integer_elt(&Integer::from(42), 0);
        assert_eq!(ctx.is_integer(&elt, &bound), Some(Integer::from(42)));
        // Out of the height bound ⇒ rejected.
        let big = ctx.integer_elt(&Integer::from(1000), 0);
        assert_eq!(ctx.is_integer(&big, &Integer::from(100)), None);
        // Negative integer round-trips (balanced residue).
        let neg = ctx.integer_elt(&Integer::from(-7), 0);
        assert_eq!(ctx.is_integer(&neg, &bound), Some(Integer::from(-7)));
    }

    // ---- raise_precision re-lifts and preserves labeling ----

    #[test]
    fn raise_precision_preserves_labeling() {
        let mut ctx = build_ctx(&ints(&[1, 0, 1]), 7, 3).unwrap();
        let frob_before = ctx.frobenius().clone();
        let degs_before = ctx.factor_degrees().to_vec();
        assert!(ctx.raise_precision(8));
        assert_eq!(ctx.prec_power(), 8);
        assert_eq!(ctx.frobenius(), &frob_before);
        assert_eq!(ctx.factor_degrees(), degs_before.as_slice());
        assert_roots_vanish(&ctx);
        // Lowering / equal is a no-op returning false.
        assert!(!ctx.raise_precision(8));
        assert!(!ctx.raise_precision(2));
    }

    // ---- top-level galois_ctx chooses a good prime automatically ----

    #[test]
    fn galois_ctx_auto_prime() {
        // x^3 - x - 1 (irreducible, Galois group S_3); disc = -23.
        let ctx = galois_ctx(&ints(&[-1, -1, 0, 1]), 5).unwrap();
        // p must not divide disc = -23 ⇒ p != 23. First good prime is 2 (disc odd).
        assert!(ctx.prime() != 23);
        assert_frobenius_invariant(&ctx);
        assert_roots_vanish(&ctx);
        let s: usize = ctx.factor_degrees().iter().sum();
        assert_eq!(s, 3);
    }

    #[test]
    fn galois_ctx_rejects_non_monic_and_nonsquarefree() {
        // non-monic
        assert!(galois_ctx(&ints(&[1, 0, 2]), 4).is_none());
        // non-squarefree: x^2 (disc 0)
        assert!(galois_ctx(&ints(&[0, 0, 1]), 4).is_none());
    }
}
