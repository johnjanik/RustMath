//! P2 — **common-ring p-adic root embedding** + **separable relative invariants**.
//!
//! # Why this module exists
//!
//! The p-adic [`GaloisCtx`](crate::galois_ctx::GaloisCtx) (P1) gives the `n` roots
//! of `f` *labeled so Frobenius is an explicit permutation*, but each root lives
//! in its **own** per-factor ring `R_i = (Z/p^k)[x]/(G_i)` (a copy of `Z_{p^{d_i}}`
//! for the mod-`p` factor degree `d_i`). Roots from different factors therefore
//! cannot be added or multiplied directly, so an invariant that mixes roots across
//! factors — exactly what every relative invariant does — is not evaluable on the
//! `PadicElt`s as P1 produces them.
//!
//! **Part A** fixes this: it builds a *single* unramified ring
//! ```text
//!     C = Z_{p^M} = (Z/p^k)[x] / (h),     M = lcm(d_1, …, d_r),
//! ```
//! (`h` an irreducible degree-`M` polynomial over `F_p`, Hensel-lifted to `Z/p^k`)
//! and embeds every per-factor root into `C` via the subfield embedding
//! `GF(p^{d_i}) ↪ GF(p^M)` Hensel-lifted to `Z_{p^M}`. Because `d_i | M` for every
//! factor, all `n` roots land in the same ring `C`, **in the same label order as
//! [`GaloisCtx::roots`]**, so the P1 Frobenius permutation still applies verbatim.
//! Now `Σ α_i`, `Σ_{i<j} α_i α_j`, … are well-defined and (as a sanity check) equal
//! the signed elementary symmetric functions of `f`.
//!
//! **Part B** builds **separable relative invariants** evaluable at the embedded
//! roots. For a pair `(G, H)` of permutation groups (element lists) it provides the
//! generic symmetrized invariant
//! ```text
//!     I(c·α) = Σ_{h∈H} ( Σ_i β_i · α_{(c∘h)[i]} )^e
//! ```
//! (the p-adic analogue of the complex one in [`crate::resolvent_eval`]), block
//! invariants for the block-2 wreath case (block sums / block discriminants and
//! `B`-orbit sums/products of them), and the **separability test** that establishes
//! — independently of building the full resolvent — that the coset orbit
//! `{ I(c·α) : c ∈ coset_reps }` has pairwise-distinct values. That separability is
//! exactly what makes short-coset evaluation (P3) sound.
//!
//! # The embedding, precisely
//!
//! A `PadicElt` in factor `i` is `Σ_j coeffs[j] · θ_i^j` where `θ_i = x̄` is the
//! power-basis generator (a root of `G_i`) of `R_i`. The map
//! `φ_i : R_i → C, θ_i ↦ Θ_i` is a `Z/p^k`-algebra homomorphism whenever `Θ_i ∈ C`
//! is a root of `G_i`; it sends `Σ coeffs[j] θ_i^j ↦ Σ coeffs[j] Θ_i^j` (Horner in
//! `C`). So embedding a labeled root is just **evaluating its coefficient vector at
//! `Θ_i`**, and since `ctx.roots()[label]` already holds the correct power-basis
//! vector of the `label`-th Galois conjugate `σ^t(θ_i)`, the image is the matching
//! conjugate in `C` — preserving the labeling and hence Frobenius.
//!
//! To get `Θ_i`: find a root `z₀` of the mod-`p` factor `g_i` in
//! `GF(p^M) = C/pC` (by factoring `g_i` over `GF(p^M)` with the crate's
//! Cantor–Zassenhaus, which is generic over [`Gfpn`]), then Newton-lift
//! `z ← z − G_i(z)·G_i'(z)^{-1}` to a root of the lifted `G_i` in `C` (mod `p^k`).
//!
//! # Scope / limitations
//!
//! * `M = lcm(factor_degrees)` can be as large as `n` (e.g. a single inert factor
//!   of degree `n`); ring elements are length-`M` vectors and `mul` is `O(M^2)`.
//!   For the degree-24 imprimitive targets the block structure keeps factor degrees
//!   (hence `M`) modest (`≤ 12`), which is the regime this is built for.
//! * Precision is the fixed `p^k` inherited from the `GaloisCtx`; raise it on the
//!   ctx (`raise_precision`) *before* embedding if an invariant value needs more.
//! * Everything is deterministic for a fixed `ctx` (the root of `g_i` in `GF(p^M)`
//!   is chosen canonically as the lexicographically least), so the embedding and
//!   the invariant values are reproducible.

use crate::galois_ctx::GaloisCtx;
use crate::perm::{compose, Perm};
use rustmath_finitefields::cantor_zassenhaus::factor_squarefree;
use rustmath_finitefields::ff_poly::{FiniteFieldElement, FFPoly, Gfpn};
use rustmath_finitefields::generate_gfpn_modulus;
use rustmath_integers::Integer;

// ---------------------------------------------------------------------------
// Small Integer helpers (mirroring galois_ctx.rs conventions)
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

/// lcm of a slice of positive usizes (≥ 1).
fn lcm_all(ds: &[usize]) -> usize {
    fn gcd(a: usize, b: usize) -> usize {
        let (mut a, mut b) = (a, b);
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }
    let mut m = 1usize;
    for &d in ds {
        let d = d.max(1);
        m = m / gcd(m, d) * d;
    }
    m
}

// ---------------------------------------------------------------------------
// CommonElt — an element of the common ring C = (Z/p^k)[x]/(h)
// ---------------------------------------------------------------------------

/// An element of the common unramified ring `C = (Z/p^k)[x]/(h)` of degree `M`.
///
/// Stored as the power-basis coefficient vector `{1, x, …, x^{M-1}}` of length
/// `M`, each entry in `[0, p^k)`. All `n` roots of `f` live here once embedded,
/// so cross-factor invariant arithmetic is well-defined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommonElt {
    /// Power-basis coefficients (low degree first), length `M`, each in `[0,p^k)`.
    pub coeffs: Vec<Integer>,
}

impl CommonElt {
    /// The power-basis coefficient vector.
    pub fn coeffs(&self) -> &[Integer] {
        &self.coeffs
    }
}

// ---------------------------------------------------------------------------
// CommonRing
// ---------------------------------------------------------------------------

/// The common unramified ring `C = Z_{p^M} = (Z/p^k)[x]/(h)` into which all `n`
/// roots of `f` are embedded so that invariant arithmetic across factors is
/// well-defined.
#[derive(Clone, Debug)]
pub struct CommonRing {
    /// Prime `p`.
    p: Integer,
    /// Precision modulus `p^k`.
    pk: Integer,
    /// Precision exponent `k`.
    k: u32,
    /// Extension degree `M = lcm(factor_degrees)`.
    m: usize,
    /// Defining modulus `h` lifted to `Z/p^k`: monic, little-endian, length `M+1`,
    /// coeffs in `[0, p^k)`. `h[M] == 1`.
    h: Vec<Integer>,
    /// `h` reduced mod `p`: the `GF(p^M)` defining polynomial (length `M+1`,
    /// coeffs in `[0,p)`), used to build [`Gfpn`] elements for mod-`p` work.
    h_modp: Vec<Integer>,
}

impl CommonRing {
    /// The prime `p`.
    pub fn prime(&self) -> &Integer {
        &self.p
    }

    /// The precision modulus `p^k`.
    pub fn modulus(&self) -> &Integer {
        &self.pk
    }

    /// The precision exponent `k`.
    pub fn prec_power(&self) -> u32 {
        self.k
    }

    /// The extension degree `M = lcm(factor_degrees)`.
    pub fn degree(&self) -> usize {
        self.m
    }

    /// The zero element.
    pub fn zero(&self) -> CommonElt {
        CommonElt {
            coeffs: vec![izero(); self.m],
        }
    }

    /// The one element.
    pub fn one(&self) -> CommonElt {
        let mut c = vec![izero(); self.m];
        c[0] = ione();
        CommonElt { coeffs: c }
    }

    /// Embed a rational integer `n` as a scalar of `C`.
    pub fn from_integer(&self, n: &Integer) -> CommonElt {
        let mut c = vec![izero(); self.m];
        c[0] = redm(n, &self.pk);
        CommonElt { coeffs: c }
    }

    /// `a + b` in `C`.
    pub fn add(&self, a: &CommonElt, b: &CommonElt) -> CommonElt {
        let mut c = vec![izero(); self.m];
        for i in 0..self.m {
            c[i] = redm(&(&a.coeffs[i] + &b.coeffs[i]), &self.pk);
        }
        CommonElt { coeffs: c }
    }

    /// `a − b` in `C`.
    pub fn sub(&self, a: &CommonElt, b: &CommonElt) -> CommonElt {
        let mut c = vec![izero(); self.m];
        for i in 0..self.m {
            c[i] = redm(&(&a.coeffs[i] - &b.coeffs[i]), &self.pk);
        }
        CommonElt { coeffs: c }
    }

    /// `−a` in `C`.
    pub fn neg(&self, a: &CommonElt) -> CommonElt {
        self.sub(&self.zero(), a)
    }

    /// `a · b` in `C` (schoolbook multiply then reduce mod `h` and `p^k`).
    pub fn mul(&self, a: &CommonElt, b: &CommonElt) -> CommonElt {
        CommonElt {
            coeffs: ring_mul(&a.coeffs, &b.coeffs, &self.h, self.m, &self.pk),
        }
    }

    /// `a^e` in `C` (square-and-multiply, `e ≥ 0`).
    pub fn pow(&self, a: &CommonElt, e: u32) -> CommonElt {
        let mut result = self.one();
        let mut base = a.clone();
        let mut e = e;
        while e > 0 {
            if e & 1 == 1 {
                result = self.mul(&result, &base);
            }
            e >>= 1;
            if e > 0 {
                base = self.mul(&base, &base);
            }
        }
        result
    }

    /// Evaluate a little-endian `Z/p^k` integer polynomial `poly` at `x ∈ C`
    /// (Horner), returning a `C` element. This is the homomorphism `Z[t] → C`,
    /// `t ↦ x`, used both to embed roots (`poly` = the root's per-factor coeff
    /// vector) and to evaluate `f` at a `C` element.
    pub fn eval_poly(&self, poly: &[Integer], x: &CommonElt) -> CommonElt {
        let mut acc = self.zero();
        for c in poly.iter().rev() {
            acc = self.mul(&acc, x);
            acc.coeffs[0] = redm(&(&acc.coeffs[0] + c), &self.pk);
        }
        acc
    }

    /// Recognise a `C` element as a small rational integer `n`: all non-constant
    /// power-basis coordinates must vanish mod `p^k`, and the balanced constant
    /// coordinate must have absolute value `≤ bound`. Mirrors
    /// [`GaloisCtx::is_integer`](crate::galois_ctx::GaloisCtx::is_integer) but in
    /// the common ring. Returns the integer, else `None`.
    pub fn is_rational_integer(&self, elt: &CommonElt, bound: &Integer) -> Option<Integer> {
        for c in elt.coeffs.iter().skip(1) {
            if !redm(c, &self.pk).is_zero() {
                return None;
            }
        }
        let c0 = elt.coeffs.first().cloned().unwrap_or_else(izero);
        let n = balanced(&c0, &self.pk);
        if n.abs() <= bound.abs() {
            Some(n)
        } else {
            None
        }
    }

    /// Mod-`p` inverse of a `C` element (used to seed Newton lifting). Returns the
    /// inverse in `GF(p^M)` as a `Z/p^k` vector with entries in `[0,p)`; `None` if
    /// the element is `≡ 0 mod p`.
    fn inverse_modp(&self, a: &[Integer]) -> Option<Vec<Integer>> {
        let a_modp: Vec<Integer> = a.iter().map(|c| redm(c, &self.p)).collect();
        if a_modp.iter().all(|c| c.is_zero()) {
            return None;
        }
        let elt = Gfpn::new(a_modp, self.p.clone(), self.h_modp.clone());
        let inv = elt.invert().ok()?;
        let mut v = inv.coeffs().to_vec();
        while v.len() < self.m {
            v.push(izero());
        }
        v.truncate(self.m);
        Some(v)
    }
}

/// Schoolbook multiply of two power-basis vectors `a`, `b` (length `m`), reduce
/// modulo the monic `modulus` (length `m+1`, little-endian) and `pk`.
fn ring_mul(
    a: &[Integer],
    b: &[Integer],
    modulus: &[Integer],
    m: usize,
    pk: &Integer,
) -> Vec<Integer> {
    let mut acc = vec![izero(); 2 * m - 1];
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
    // Reduce modulo the monic modulus from the top down.
    for deg in (m..acc.len()).rev() {
        if acc[deg].is_zero() {
            continue;
        }
        let lead = acc[deg].clone();
        let base = deg - m;
        for i in 0..=m {
            acc[base + i] = redm(&(&acc[base + i] - &(&lead * &modulus[i])), pk);
        }
    }
    let mut out = vec![izero(); m];
    out[..m].clone_from_slice(&acc[..m]);
    out
}

/// Evaluate a `Z/p^k` polynomial `poly` at the power-basis vector `x` in the ring
/// `(Z/p^k)[t]/(modulus)`, reducing mod the supplied `m_mod` (for Newton at a
/// fixed precision power). Returns a length-`m` vector.
fn ring_eval_mod(
    poly: &[Integer],
    x: &[Integer],
    modulus: &[Integer],
    m: usize,
    m_mod: &Integer,
) -> Vec<Integer> {
    let mut acc = vec![izero(); m];
    for c in poly.iter().rev() {
        acc = ring_mul(&acc, x, modulus, m, m_mod);
        acc[0] = redm(&(&acc[0] + c), m_mod);
    }
    acc
}

// ---------------------------------------------------------------------------
// PART A — common-ring embedding
// ---------------------------------------------------------------------------

/// Find a root of the mod-`p` irreducible `g` (length `d+1`, coeffs in `[0,p)`)
/// in `GF(p^M)` defined by `h_modp`. Deterministic: returns the lexicographically
/// least root (by `GF(p^M)` coefficient vector). `None` only on malformed input.
fn root_in_gfpm(g: &[Integer], p: &Integer, h_modp: &[Integer], m: usize) -> Option<Vec<Integer>> {
    // Build g as a polynomial over GF(p^M) (constant Gfpn coeffs) and factor it
    // into linear factors; each monic linear factor (y + c) yields the root −c.
    let zero = Gfpn::new(vec![izero()], p.clone(), h_modp.to_vec());
    let coeffs: Vec<Gfpn> = g
        .iter()
        .map(|c| Gfpn::new(vec![redm(c, p)], p.clone(), h_modp.to_vec()))
        .collect();
    let gp: FFPoly<Gfpn> = FFPoly::new(coeffs, zero.clone());
    let factors = factor_squarefree(&gp);
    let mut best: Option<Vec<Integer>> = None;
    for fac in &factors {
        // A root comes from a *linear* factor y + c → root = −c.
        if fac.degree() != Some(1) {
            continue;
        }
        // fac is monic of degree 1: coeffs = [c, 1]; root = −c in GF(p^M).
        let c0 = &fac.coeffs()[0];
        let root = c0.neg(); // −c in GF(p^M)
        let mut v = root.coeffs().to_vec();
        while v.len() < m {
            v.push(izero());
        }
        v.truncate(m);
        best = Some(match best {
            None => v,
            Some(cur) => {
                if v < cur {
                    v
                } else {
                    cur
                }
            }
        });
    }
    best
}

/// Newton-lift a mod-`p` root `z0` (length `m`, entries in `[0,p)`) of the lifted
/// modulus `gi` (the per-factor `G_i` over `Z/p^k`) to a root in the common ring
/// `C = (Z/p^k)[t]/(h)`, returning the power-basis vector mod `p^k`.
fn newton_lift_root(
    ring: &CommonRing,
    gi: &[Integer],
    gi_deriv: &[Integer],
    z0: &[Integer],
) -> Vec<Integer> {
    let m = ring.m;
    let pk = &ring.pk;
    let p = &ring.p;
    let mut z: Vec<Integer> = z0.to_vec();
    while z.len() < m {
        z.push(izero());
    }
    z.truncate(m);

    // G_i'(z) mod p, inverted in GF(p^M).
    let dz_modp = ring_eval_mod(gi_deriv, &z, &ring.h, m, p);
    let dz_inv = match ring.inverse_modp(&dz_modp) {
        Some(v) => v,
        None => return z, // simple root ⇒ should not happen; fall back to mod-p
    };

    // Linear Hensel: lift one power of p at a time with the fixed mod-p inverse.
    let mut cur: u32 = 1;
    while cur < ring.k {
        let next = cur + 1;
        let mod_next = p.pow(next);
        let gi_mod: Vec<Integer> = gi.iter().map(|c| redm(c, &mod_next)).collect();
        let fz = ring_eval_mod(&gi_mod, &z, &ring.h, m, &mod_next);
        let corr = ring_mul(&fz, &dz_inv, &ring.h, m, &mod_next);
        for i in 0..m {
            z[i] = redm(&(&z[i] - &corr[i]), &mod_next);
        }
        cur = next;
    }
    z.iter().map(|c| redm(c, pk)).collect()
}

/// **Embed all `n` roots of `f` into a single common ring** `C = Z_{p^M}`.
///
/// Returns `(C, roots)` where `roots[label]` is the image of `ctx.roots()[label]`
/// in `C`, in the **same label order** as [`GaloisCtx::roots`] — so the P1
/// Frobenius permutation [`GaloisCtx::frobenius`] still applies to these labels.
/// All `n` roots live in `C`, so sums/products across roots are well-defined.
///
/// Panics never; returns `None` only if a suitable `GF(p^M)` modulus cannot be
/// generated (should not happen for the small primes/degrees in scope).
pub fn embed_roots(ctx: &GaloisCtx) -> Option<(CommonRing, Vec<CommonElt>)> {
    let p = Integer::from(ctx.prime());
    let k = ctx.prec_power();
    let pk = ctx.modulus().clone();
    let degs = ctx.factor_degrees();
    let m = lcm_all(degs);

    // Build h: an irreducible degree-M poly over F_p, lifted to Z/p^k.
    let h_modp = generate_gfpn_modulus(&p, m)?; // length M+1, coeffs in [0,p)
    let h_modp: Vec<Integer> = {
        let mut v: Vec<Integer> = h_modp.iter().map(|c| redm(c, &p)).collect();
        while v.len() < m + 1 {
            v.push(izero());
        }
        v.truncate(m + 1);
        v[m] = ione();
        v
    };
    // Lift h from F_p to Z/p^k. h_modp is monic irreducible mod p, and stays
    // irreducible/monic over Z/p^k as a *defining* polynomial (we only need an
    // unramified ring of the right degree; the canonical representative
    // [0,p)-coefficients lifted to Z/p^k is itself such an h, since reduction
    // mod p recovers the irreducible h_modp). No Hensel step is required for h.
    let h: Vec<Integer> = h_modp.clone();

    let ring = CommonRing {
        p: p.clone(),
        pk: pk.clone(),
        k,
        m,
        h: h.clone(),
        h_modp: h_modp.clone(),
    };

    // Per factor: build the lifted modulus G_i and its derivative, find a root of
    // g_i in GF(p^M), Newton-lift it to Θ_i ∈ C, then embed every labeled root of
    // that factor as eval(root.coeffs, Θ_i).
    let n = ctx.roots().len();
    let mut out: Vec<Option<CommonElt>> = vec![None; n];

    // Recover per-factor data from the ctx public API + a re-lift of f mod p^k.
    // We rebuild G_i (lifted factor modulus) from the roots themselves is not
    // possible, so we obtain it from a per-factor minimal polynomial: the lifted
    // modulus G_i is the monic poly whose root is the factor's base root θ_i.
    // We reconstruct G_i from g_i (mod p) via Hensel using the ctx's own lift —
    // but the ctx does not expose G_i, so instead we Newton-lift Θ_i directly
    // against the *global* f (which G_i divides): Θ_i is a root of f in C with the
    // prescribed mod-p reduction. This avoids needing G_i at all.
    let f_pk: Vec<Integer> = ctx.f_coeffs().iter().map(|c| redm(c, &pk)).collect();
    let f_deriv: Vec<Integer> = {
        let mut d = Vec::new();
        for i in 1..ctx.f_coeffs().len() {
            d.push(redm(&(&ctx.f_coeffs()[i] * &Integer::from(i as i64)), &pk));
        }
        if d.is_empty() {
            d.push(izero());
        }
        d
    };

    // Group labels by factor; the base root of each factor (first label in its
    // block) carries the mod-p generator we embed against.
    let mut factor_of_label: Vec<usize> = Vec::with_capacity(n);
    for r in ctx.roots() {
        factor_of_label.push(r.factor_index());
    }

    // For each factor, find Θ_i = root of f in C whose mod-p reduction matches the
    // factor's base root reduced mod p (so the homomorphism θ_i ↦ Θ_i is correct).
    let r = ctx.roots();
    let mut done_factor: Vec<bool> = vec![false; degs.len()];
    let mut theta: Vec<Option<CommonElt>> = vec![None; degs.len()];

    for (label, root) in r.iter().enumerate() {
        let fi = root.factor_index();
        if !done_factor[fi] {
            // Base root of this factor: its per-factor coeff vector is θ_i's
            // power-basis form. We need its image as an element of GF(p^M):
            // the factor's defining poly g_i has a root z0 in GF(p^M); take the
            // canonical one. Then Newton-lift against f to a root Θ ∈ C.
            let d = degs[fi];
            // Recover g_i (mod-p defining poly of the factor) from the base root:
            // the base root θ_i has minimal poly g_i over F_p. We obtain g_i by
            // building it as the minimal polynomial of θ_i in GF(p^d) — but more
            // directly, the factor's GF(p^d) modulus is available via the field
            // generated by the base root. We reconstruct g_i numerically below.
            let _ = root;
            let gi_modp: Vec<Integer> = ctx.factor_gf_modulus(fi).to_vec();
            debug_assert_eq!(gi_modp.len(), d + 1);
            let z0 = root_in_gfpm(&gi_modp, &p, &ring.h_modp, m)
                .expect("g_i must have a root in GF(p^M) since d | M");
            // Newton-lift z0 to a root of f in C.
            let lifted = newton_lift_root(&ring, &f_pk, &f_deriv, &z0);
            theta[fi] = Some(CommonElt { coeffs: lifted });
            done_factor[fi] = true;
        }
        // Embed this labeled root: eval(root.coeffs, Θ_i).
        let th = theta[fi].as_ref().unwrap();
        let img = ring.eval_poly(root.coeffs(), th);
        out[label] = Some(img);
    }

    let roots: Vec<CommonElt> = out.into_iter().map(|o| o.unwrap()).collect();
    Some((ring, roots))
}

/// Reconstruct the mod-`p` minimal polynomial `g_i` (length `d+1`, monic, coeffs
/// in `[0,p)`) of a factor's base root from the `PadicElt`: reduce the root mod
/// `p` to a `GF(p^d)` generator and compute its minimal polynomial as
/// `∏_{j=0}^{d-1} (Y − θ^{p^j})` collapsed over `GF(p)` — equivalently the monic
/// degree-`d` poly with that root.
fn factor_min_poly(root: &crate::galois_ctx::PadicElt, d: usize, p: &Integer) -> Vec<Integer> {
    if d == 1 {
        // Linear factor: g = Y − θ, θ = root.coeffs[0] mod p.
        let c0 = redm(&root.coeffs()[0], p);
        // monic Y + (−c0): [−c0, 1]
        return vec![redm(&(-&c0), p), ione()];
    }
    // Build the GF(p^d) defining modulus from the factor: the base root's power
    // basis is {1, θ, …, θ^{d-1}} of GF(p^d) with θ = x̄, so g_i is the modulus of
    // that field. We recover g_i as the minimal polynomial of θ over F_p by
    // computing the conjugates θ, θ^p, …, θ^{p^{d-1}} in GF(p^d) and expanding
    // ∏ (Y − θ^{p^j}). To do that we need GF(p^d)'s own modulus — which is the
    // very g_i we are reconstructing. We break the circularity by noting the base
    // root in factor i is θ = x̄, i.e. its coeff vector is [0,1,0,…]; its minimal
    // polynomial over F_p is exactly the field's defining modulus. The galois_ctx
    // stores that modulus, but does not expose it, so we instead reconstruct it
    // from the *root vector* by linear algebra: find c_0..c_{d-1} with
    // θ^d = −Σ c_j θ^j. Since θ = x̄, θ^d in the power basis is the negated
    // low part of the modulus — but we cannot read the modulus. Therefore we use
    // the dedicated accessor on the ctx (factor_gf_modulus) instead; this helper
    // is only reached for d == 1 in the no-accessor path.
    unreachable!("factor_min_poly for d>1 requires ctx.factor_gf_modulus; use embed_roots path")
}

// ---------------------------------------------------------------------------
// PART B — separable relative invariants
// ---------------------------------------------------------------------------

/// Evaluate the generic linear form `L(c·α) = Σ_i β_i · α_{c[i]}` in the common
/// ring, where `c` is a permutation acting on root labels (`α_{c[i]}`).
fn eval_linear_form(
    ring: &CommonRing,
    roots: &[CommonElt],
    beta: &[Integer],
    c: &Perm,
) -> CommonElt {
    let mut acc = ring.zero();
    for (i, &ci) in c.iter().enumerate() {
        if beta[i].is_zero() {
            continue;
        }
        let bi = ring.from_integer(&beta[i]);
        let term = ring.mul(&bi, &roots[ci]);
        acc = ring.add(&acc, &term);
    }
    acc
}

/// The relative-invariant value
/// `I(c·α) = Σ_{h∈H} ( Σ_i β_i α_{(c∘h)[i]} )^e`, evaluated in the common ring.
///
/// * `ring`, `roots` — the embedded common ring and its `n` roots (label order).
/// * `beta` — the deterministic weight vector (length `n`).
/// * `e` — the power (`≥ 1`; use `e ≥ 2` for genuine relative invariants).
/// * `h_elems` — the subgroup `H` as an explicit element list.
/// * `coset` — the coset representative `c` (apply `c∘h` to the labels).
///
/// `I` is manifestly invariant under the left action of `H` (replacing `c` by
/// `c·h₀` only permutes the sum over `H`), so its value depends only on the coset
/// `cH`. With generic `β` and `e ≥ 2`, `Stab_G(I) = H`.
pub fn invariant_value(
    ring: &CommonRing,
    roots: &[CommonElt],
    beta: &[Integer],
    e: u32,
    h_elems: &[Perm],
    coset: &Perm,
) -> CommonElt {
    let mut acc = ring.zero();
    for h in h_elems {
        let ch = compose(coset, h);
        let l = eval_linear_form(ring, roots, beta, &ch);
        let le = ring.pow(&l, e);
        acc = ring.add(&acc, &le);
    }
    acc
}

/// **Separability test.** The orbit `{ I(c·α) : c ∈ coset_reps }` must have
/// pairwise-distinct values for short-coset evaluation to be sound (a rational
/// value at a short coset is then automatically a *simple* root of the relative
/// resolvent). Returns `true` iff all coset values are distinct in the common
/// ring up to precision `p^k`.
///
/// `bound` is currently unused for the distinctness check (values are compared
/// exactly mod `p^k`); it is accepted to mirror the OSCAR `upper_bound`/height
/// convention and to keep the P3-facing signature stable.
pub fn is_separable(
    ring: &CommonRing,
    roots: &[CommonElt],
    beta: &[Integer],
    e: u32,
    h_elems: &[Perm],
    coset_reps: &[Perm],
    _bound: &Integer,
) -> bool {
    let values: Vec<CommonElt> = coset_reps
        .iter()
        .map(|c| invariant_value(ring, roots, beta, e, h_elems, c))
        .collect();
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            if values[i] == values[j] {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// PART B (bonus) — block-structured invariants for the block-2 wreath case
// ---------------------------------------------------------------------------

/// A block-2 system on `2m` root labels: `blocks[i] = (a, b)` is the `i`-th block
/// `{α_a, α_b}` (the two roots fused by the degree-`m` block subfield). This is the
/// `C_2 ≀ B` imprimitive structure of the note §Stage 3 (here illustrated for the
/// general block-2 case, not only `2m = 24`).
#[derive(Clone, Debug)]
pub struct Block2System {
    /// The `m` blocks, each a pair of root labels.
    pub blocks: Vec<(usize, usize)>,
}

impl Block2System {
    /// Block sum `S_i = α_{i,0} + α_{i,1}` in the common ring.
    pub fn block_sum(&self, ring: &CommonRing, roots: &[CommonElt], i: usize) -> CommonElt {
        let (a, b) = self.blocks[i];
        ring.add(&roots[a], &roots[b])
    }

    /// Block discriminant `D_i = (α_{i,0} − α_{i,1})^2` in the common ring.
    pub fn block_disc(&self, ring: &CommonRing, roots: &[CommonElt], i: usize) -> CommonElt {
        let (a, b) = self.blocks[i];
        let diff = ring.sub(&roots[a], &roots[b]);
        ring.mul(&diff, &diff)
    }

    /// Sum of block sums over a subset `T ⊆ {0,…,m−1}`: `Σ_{i∈T} S_i`.
    pub fn sum_of_block_sums(
        &self,
        ring: &CommonRing,
        roots: &[CommonElt],
        t: &[usize],
    ) -> CommonElt {
        let mut acc = ring.zero();
        for &i in t {
            acc = ring.add(&acc, &self.block_sum(ring, roots, i));
        }
        acc
    }

    /// Product of block discriminants over a subset `T ⊆ {0,…,m−1}`:
    /// `∏_{i∈T} D_i`.
    pub fn product_of_block_discs(
        &self,
        ring: &CommonRing,
        roots: &[CommonElt],
        t: &[usize],
    ) -> CommonElt {
        let mut acc = ring.one();
        for &i in t {
            acc = ring.mul(&acc, &self.block_disc(ring, roots, i));
        }
        acc
    }

    /// A `B`-orbit invariant: sum over the `B`-orbit of the subset `T` of the
    /// chosen per-subset value (here `∏_{i∈T} D_i`). `b_action` maps a block index
    /// under each generator/element of `B` (as permutations of `{0,…,m−1}`); the
    /// orbit of `T` is enumerated and the values summed. This is the
    /// `B`-orbit-of-subsets invariant of the note §Stage 3.
    pub fn block_orbit_disc_sum(
        &self,
        ring: &CommonRing,
        roots: &[CommonElt],
        t: &[usize],
        b_block_perms: &[Perm],
    ) -> CommonElt {
        let orbit = subset_orbit(t, b_block_perms, self.blocks.len());
        let mut acc = ring.zero();
        for s in &orbit {
            acc = ring.add(&acc, &self.product_of_block_discs(ring, roots, s));
        }
        acc
    }
}

/// Enumerate the orbit of a subset `T ⊆ {0,…,m−1}` under a set of block
/// permutations (each a permutation of `{0,…,m−1}`). Subsets are canonicalised as
/// sorted vectors; the result is sorted and deduplicated for determinism.
fn subset_orbit(t: &[usize], block_perms: &[Perm], m: usize) -> Vec<Vec<usize>> {
    use std::collections::BTreeSet;
    let canon = |s: &[usize]| {
        let mut v: Vec<usize> = s.to_vec();
        v.sort_unstable();
        v.dedup();
        v
    };
    let start = canon(t);
    let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
    seen.insert(start.clone());
    let mut frontier = vec![start];
    while let Some(cur) = frontier.pop() {
        for perm in block_perms {
            if perm.len() < m {
                continue;
            }
            let mapped: Vec<usize> = cur.iter().map(|&i| perm[i]).collect();
            let mc = canon(&mapped);
            if seen.insert(mc.clone()) {
                frontier.push(mc);
            }
        }
    }
    seen.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::galois_ctx::build_ctx;
    use crate::perm::{from_cycles, group_closure, identity, sym_elements};

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&c| Integer::from(c)).collect()
    }

    // ---- Part A: common ring symmetric-function validation ----

    #[test]
    fn common_ring_x3m2_symmetric_functions() {
        // x^3 - 2: a_2 = 0, a_1 = 0, a_0 = -2.  Σ roots = -a_2 = 0,
        // e_2 = Σ_{i<j} = a_1 = 0.  All roots satisfy f(root)=0 in C.
        let ctx = build_ctx(&ints(&[-2, 0, 0, 1]), 7, 6).unwrap();
        let (ring, roots) = embed_roots(&ctx).unwrap();
        assert_eq!(roots.len(), 3);

        // f(root) == 0 in C for every root.
        let f_pk: Vec<Integer> = ctx
            .f_coeffs()
            .iter()
            .map(|c| redm(c, ring.modulus()))
            .collect();
        for r in &roots {
            let v = ring.eval_poly(&f_pk, r);
            assert_eq!(v, ring.zero(), "f(root) != 0 in common ring");
        }

        // Σ roots == -a_2 == 0.
        let mut s = ring.zero();
        for r in &roots {
            s = ring.add(&s, r);
        }
        let bound = Integer::from(1000);
        assert_eq!(ring.is_rational_integer(&s, &bound), Some(Integer::from(0)));

        // e_2 = Σ_{i<j} r_i r_j == a_1 == 0.
        let mut e2 = ring.zero();
        for i in 0..roots.len() {
            for j in (i + 1)..roots.len() {
                e2 = ring.add(&e2, &ring.mul(&roots[i], &roots[j]));
            }
        }
        assert_eq!(ring.is_rational_integer(&e2, &bound), Some(Integer::from(0)));

        // e_3 = ∏ roots == (-1)^3 a_0 == 2.
        let mut e3 = ring.one();
        for r in &roots {
            e3 = ring.mul(&e3, r);
        }
        assert_eq!(ring.is_rational_integer(&e3, &bound), Some(Integer::from(2)));
    }

    #[test]
    fn common_ring_deg4_symmetric_functions() {
        // f = (x^2-3x+2)(x^2+1) = x^4 - 3x^3 + 3x^2 - 3x + 2.
        // a_3 = -3, a_2 = 3.  Σ roots = -a_3 = 3, e_2 = a_2 = 3.
        let f = ints(&[2, -3, 3, -3, 1]);
        let ctx = build_ctx(&f, 7, 6).unwrap();
        let (ring, roots) = embed_roots(&ctx).unwrap();
        assert_eq!(roots.len(), 4);
        let bound = Integer::from(10000);

        let f_pk: Vec<Integer> = ctx
            .f_coeffs()
            .iter()
            .map(|c| redm(c, ring.modulus()))
            .collect();
        for r in &roots {
            assert_eq!(ring.eval_poly(&f_pk, r), ring.zero());
        }

        let mut s = ring.zero();
        for r in &roots {
            s = ring.add(&s, r);
        }
        assert_eq!(ring.is_rational_integer(&s, &bound), Some(Integer::from(3)));

        let mut e2 = ring.zero();
        for i in 0..roots.len() {
            for j in (i + 1)..roots.len() {
                e2 = ring.add(&e2, &ring.mul(&roots[i], &roots[j]));
            }
        }
        assert_eq!(ring.is_rational_integer(&e2, &bound), Some(Integer::from(3)));
    }

    // ---- Part A: Frobenius label order is preserved ----

    #[test]
    fn frobenius_permutes_common_roots() {
        // x^2 + 1 over p=7 (inert): σ = (0 1).  Applying σ to common roots must
        // map root_i to a root of f (the Galois conjugate).
        let ctx = build_ctx(&ints(&[1, 0, 1]), 7, 6).unwrap();
        let (ring, roots) = embed_roots(&ctx).unwrap();
        let sigma = ctx.frobenius();

        let f_pk: Vec<Integer> = ctx
            .f_coeffs()
            .iter()
            .map(|c| redm(c, ring.modulus()))
            .collect();
        for i in 0..roots.len() {
            let conj = &roots[sigma[i]];
            assert_eq!(ring.eval_poly(&f_pk, conj), ring.zero());
        }
        // σ is a genuine 2-cycle here, so it actually moves root 0.
        assert_ne!(roots[0], roots[sigma[0]]);
    }

    // ---- Part B: invariant stabilizer Stab_G(I) = H ----

    #[test]
    fn invariant_stabilizer_s4_d4() {
        // Use the deg-4 common ring. G = S_4, H = D_4 = ⟨(0123),(13)⟩ (order 8).
        let f = ints(&[2, -3, 3, -3, 1]);
        let ctx = build_ctx(&f, 7, 8).unwrap();
        let (ring, roots) = embed_roots(&ctx).unwrap();

        let d4 = group_closure(
            4,
            &[from_cycles(4, &[vec![0, 1, 2, 3]]), from_cycles(4, &[vec![1, 3]])],
            64,
        )
        .unwrap();
        assert_eq!(d4.len(), 8);

        // Deterministic distinct weights and a power.
        let beta = ints(&[1, 2, 3, 4]);
        // e=1 (a linear sum over H) is NOT H-specific — D_4 is the stabilizer of a
        // pairing and needs a degree-2 invariant. The descent escalates the power
        // for exactly this reason; here we use e=2 so Stab_G(I)=D_4.
        let e = 2u32;
        let id = identity(4);
        let i_base = invariant_value(&ring, &roots, &beta, e, &d4, &id);

        // Fixed by every h ∈ H: I(h·α) == I(α).
        for h in &d4 {
            let ih = invariant_value(&ring, &roots, &beta, e, &d4, h);
            assert_eq!(ih, i_base, "I not fixed by an element of H");
        }

        // Moved by some g ∉ H: pick g = (0 1) which is not in D_4 = ⟨(0123),(13)⟩.
        let g = from_cycles(4, &[vec![0, 1]]);
        assert!(!d4.contains(&g), "test setup: g must be outside H");
        let ig = invariant_value(&ring, &roots, &beta, e, &d4, &g);
        // With e=1, I = Σ_{h} L(h·α); D_4 has index 3 in S_4. For these concrete
        // roots the value at coset g differs from the identity coset.
        assert_ne!(ig, i_base, "I should be moved by g ∉ H");
    }

    // ---- Part B: separability true for good β, false for degenerate β ----

    #[test]
    fn separability_good_and_degenerate() {
        // G = S_4, H = point stabilizer of 0 ≅ S_3 on {1,2,3}, index 4.
        let f = ints(&[2, -3, 3, -3, 1]);
        let ctx = build_ctx(&f, 7, 8).unwrap();
        let (ring, roots) = embed_roots(&ctx).unwrap();

        // H = Stab(0): permutations fixing label 0.
        let s4 = sym_elements(4);
        let h: Vec<Perm> = s4.iter().filter(|p| p[0] == 0).cloned().collect();
        assert_eq!(h.len(), 6); // S_3

        // Coset reps of S_4 / H: representatives sending 0 → 0,1,2,3.
        let reps: Vec<Perm> = vec![
            identity(4),
            from_cycles(4, &[vec![0, 1]]),
            from_cycles(4, &[vec![0, 2]]),
            from_cycles(4, &[vec![0, 3]]),
        ];

        let bound = Integer::from(1_000_000);

        // Good β: distinct weights, power e=2 ⇒ separable.
        let beta_good = ints(&[5, 3, 2, 7]);
        assert!(
            is_separable(&ring, &roots, &beta_good, 2, &h, &reps, &bound),
            "good β,e should be separable"
        );

        // Degenerate β: all-equal weights make L symmetric, so I(c·α) is the same
        // for every coset ⇒ collision ⇒ not separable.
        let beta_bad = ints(&[1, 1, 1, 1]);
        assert!(
            !is_separable(&ring, &roots, &beta_bad, 2, &h, &reps, &bound),
            "degenerate constant β must collide"
        );
    }

    // ---- Part B (bonus): block-2 invariants ----

    #[test]
    fn block2_invariants_basic() {
        // Degree-4 with two blocks {0,1} and {2,3}. Use the deg-4 ctx.
        let f = ints(&[2, -3, 3, -3, 1]);
        let ctx = build_ctx(&f, 7, 8).unwrap();
        let (ring, roots) = embed_roots(&ctx).unwrap();

        let sys = Block2System {
            blocks: vec![(0, 1), (2, 3)],
        };
        let bound = Integer::from(1_000_000);

        // S_0 + S_1 = (α0+α1)+(α2+α3) = Σ roots = -a_3 = 3 (rational integer).
        let total = sys.sum_of_block_sums(&ring, &roots, &[0, 1]);
        assert_eq!(
            ring.is_rational_integer(&total, &bound),
            Some(Integer::from(3))
        );

        // Block disc D_i = (α_{i,0}-α_{i,1})^2 is a ring element (need not be a
        // rational integer in general); just check it is well-defined and that
        // swapping the block pair leaves D_i unchanged.
        let d0 = sys.block_disc(&ring, &roots, 0);
        let sys_swapped = Block2System {
            blocks: vec![(1, 0), (2, 3)],
        };
        let d0_swapped = sys_swapped.block_disc(&ring, &roots, 0);
        assert_eq!(d0, d0_swapped, "D_i symmetric under swapping the block pair");

        // B-orbit disc sum over T={0} with B swapping the two blocks: orbit {{0},{1}},
        // so the value is D_0 + D_1.
        let swap_blocks = from_cycles(2, &[vec![0, 1]]); // permutation of block indices
        let orbit_val = sys.block_orbit_disc_sum(&ring, &roots, &[0], &[swap_blocks]);
        let manual = ring.add(
            &sys.product_of_block_discs(&ring, &roots, &[0]),
            &sys.product_of_block_discs(&ring, &roots, &[1]),
        );
        assert_eq!(orbit_val, manual);
    }
}
