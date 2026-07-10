//! `Hom` and `End` of modules over a Dedekind domain.
//!
//! For pseudo-basis (projective) modules `M = ⊕ᵢ 𝔞ᵢvᵢ` and `N = ⊕ⱼ 𝔟ⱼwⱼ`, an
//! `O`-linear map extends `K`-linearly, so in pseudo-basis coordinates a
//! homomorphism *is* a matrix `T = (t_{ij})` with `φ(vᵢ) = Σⱼ t_{ij}wⱼ`, and
//! `φ(M) ⊆ N` ⟺ `t_{ij} ∈ 𝔟ⱼ·𝔞ᵢ⁻¹`. Hence
//!
//! `Hom_O(M, N) ≅ ⊕_{i,j} (𝔟ⱼ𝔞ᵢ⁻¹)·E_{ij}`
//!
//! is itself a projective `O`-module of rank `rank(M)·rank(N)`, with an
//! explicit pseudo-basis of elementary matrices — everything here is exact
//! ideal arithmetic, no class-group data required.
//!
//! For torsion modules given by elementary divisors,
//! `Hom(⊕ O/𝔡ᵢ, ⊕ O/𝔢ⱼ) ≅ ⊕_{i,j} O/(𝔡ᵢ + 𝔢ⱼ)` (the local computation
//! `Hom(O/𝔭^a, O/𝔭^b) = O/𝔭^{min(a,b)}` glued by CRT).
//!
//! The object-safe layer uses [`rustmath_core::morphism::Morphism`]: a
//! [`PseudoHom`] applies a coordinate matrix and can be boxed, composed, and
//! stored polymorphically.

use super::pseudo::PseudoMatrix;
use super::{DedekindContext, DedekindError};
use rustmath_core::morphism::Morphism;

/// `Hom_O(M, N)` for pseudo-basis modules, as an explicit pseudo-structure:
/// the coefficient ideal of the elementary matrix `E_{ij}` is `𝔟ⱼ·𝔞ᵢ⁻¹`.
#[derive(Debug, Clone)]
pub struct HomModule<C: DedekindContext> {
    src: PseudoMatrix<C>,
    tgt: PseudoMatrix<C>,
    /// `coeff[i][j] = 𝔟ⱼ·𝔞ᵢ⁻¹`
    coeff: Vec<Vec<C::Ideal>>,
}

/// Compute `Hom_O(M, N)` of the modules spanned by two pseudo-matrices.
/// Inputs are Hermite-reduced internally, so they may be arbitrary
/// pseudo-matrices; the stored source/target are their pseudo-bases.
pub fn hom_module<C: DedekindContext>(
    ctx: &C,
    m: &PseudoMatrix<C>,
    n: &PseudoMatrix<C>,
) -> Result<HomModule<C>, DedekindError> {
    let src = m.hnf(ctx)?;
    let tgt = n.hnf(ctx)?;
    let mut coeff = Vec::with_capacity(src.nrows());
    for a in src.ideals() {
        let a_inv = ctx.ideal_inv(a);
        coeff.push(
            tgt.ideals()
                .iter()
                .map(|b| ctx.ideal_mul(b, &a_inv))
                .collect(),
        );
    }
    Ok(HomModule { src, tgt, coeff })
}

/// `End_O(M) = Hom_O(M, M)`.
pub fn end_module<C: DedekindContext>(
    ctx: &C,
    m: &PseudoMatrix<C>,
) -> Result<HomModule<C>, DedekindError> {
    hom_module(ctx, m, m)
}

impl<C: DedekindContext> HomModule<C> {
    /// Rank of the source pseudo-basis.
    pub fn source_rank(&self) -> usize {
        self.src.nrows()
    }
    /// Rank of the target pseudo-basis.
    pub fn target_rank(&self) -> usize {
        self.tgt.nrows()
    }
    /// Rank of `Hom(M, N)` as a projective `O`-module.
    pub fn rank(&self) -> usize {
        self.source_rank() * self.target_rank()
    }
    /// The source pseudo-basis.
    pub fn source(&self) -> &PseudoMatrix<C> {
        &self.src
    }
    /// The target pseudo-basis.
    pub fn target(&self) -> &PseudoMatrix<C> {
        &self.tgt
    }
    /// The coefficient ideal `𝔟ⱼ·𝔞ᵢ⁻¹` of the elementary matrix `E_{ij}`.
    pub fn coefficient_ideal(&self, i: usize, j: usize) -> &C::Ideal {
        &self.coeff[i][j]
    }

    /// Is the coordinate matrix `t` (source_rank × target_rank over `K`) an
    /// `O`-module homomorphism `M → N`?
    pub fn is_hom(&self, ctx: &C, t: &[Vec<C::Elem>]) -> bool {
        if t.len() != self.source_rank() || t.iter().any(|r| r.len() != self.target_rank()) {
            return false;
        }
        t.iter()
            .zip(&self.coeff)
            .all(|(trow, crow)| {
                trow.iter()
                    .zip(crow)
                    .all(|(e, ideal)| ctx.ideal_contains(ideal, e))
            })
    }

    /// The Steinitz-class representative of `Hom(M, N)`:
    /// `∏_{i,j} 𝔟ⱼ𝔞ᵢ⁻¹ = (∏𝔟)^{rank M}·(∏𝔞)^{-rank N}`.
    pub fn steinitz_ideal(&self, ctx: &C) -> C::Ideal {
        let mut s = ctx.unit_ideal();
        for row in &self.coeff {
            for ideal in row {
                s = ctx.ideal_mul(&s, ideal);
            }
        }
        s
    }

    /// `Hom(M, N)` as a pseudo-matrix in its own right: the module
    /// `⊕_{i,j} (𝔟ⱼ𝔞ᵢ⁻¹)·E_{ij} ⊆ K^{rank M · rank N}` over the flattened
    /// elementary-matrix basis (row-major `(i, j)` flattening).
    pub fn as_pseudo_matrix(&self, ctx: &C) -> PseudoMatrix<C> {
        let dim = self.rank();
        let mut pseudo_rows = Vec::with_capacity(dim);
        for (i, crow) in self.coeff.iter().enumerate() {
            for (j, ideal) in crow.iter().enumerate() {
                let mut row = vec![ctx.zero(); dim];
                row[i * self.target_rank() + j] = ctx.one();
                pseudo_rows.push((ideal.clone(), row));
            }
        }
        PseudoMatrix::new(dim, pseudo_rows).expect("elementary rows have the right length")
    }
}

/// Matrix product of two coordinate homs: `(t1·t2)[i][k] = Σⱼ t1[i][j]·t2[j][k]`
/// — the composition `M → N → P` when `t1 : M → N` and `t2 : N → P`.
pub fn compose_homs<C: DedekindContext>(
    ctx: &C,
    t1: &[Vec<C::Elem>],
    t2: &[Vec<C::Elem>],
) -> Result<Vec<Vec<C::Elem>>, DedekindError> {
    let mid = match t1.first() {
        None => return Ok(Vec::new()),
        Some(r) => r.len(),
    };
    if t2.len() != mid {
        return Err(DedekindError::Shape(format!(
            "inner dimensions {} and {} do not match",
            mid,
            t2.len()
        )));
    }
    let out_cols = t2.first().map_or(0, |r| r.len());
    let mut out = Vec::with_capacity(t1.len());
    for r1 in t1 {
        let mut row = Vec::with_capacity(out_cols);
        for k in 0..out_cols {
            let mut acc = ctx.zero();
            for (j, e) in r1.iter().enumerate() {
                acc = ctx.add(&acc, &ctx.mul(e, &t2[j][k]));
            }
            row.push(acc);
        }
        out.push(row);
    }
    Ok(out)
}

/// A homomorphism between pseudo-basis modules as an object-safe
/// [`Morphism`] on coordinate vectors: `c ↦ c·T`. Compose/box it with the
/// combinators from [`rustmath_core::morphism`].
pub struct PseudoHom<'a, C: DedekindContext> {
    ctx: &'a C,
    /// `t[i][j]`: coefficient of the `j`-th target basis vector in the image
    /// of the `i`-th source basis vector.
    t: Vec<Vec<C::Elem>>,
}

impl<'a, C: DedekindContext> PseudoHom<'a, C> {
    /// Wrap a coordinate matrix. Correctness of the ideal conditions can be
    /// checked against a [`HomModule`] via [`HomModule::is_hom`].
    pub fn new(ctx: &'a C, t: Vec<Vec<C::Elem>>) -> Self {
        PseudoHom { ctx, t }
    }

    /// The coordinate matrix.
    pub fn matrix(&self) -> &[Vec<C::Elem>] {
        &self.t
    }
}

impl<C: DedekindContext> Morphism for PseudoHom<'_, C> {
    type Domain = Vec<C::Elem>;
    type Codomain = Vec<C::Elem>;

    /// `c ↦ c·T` in pseudo-basis coordinates.
    fn apply(&self, c: &Vec<C::Elem>) -> Vec<C::Elem> {
        assert_eq!(c.len(), self.t.len(), "coordinate length must match source rank");
        let out_cols = self.t.first().map_or(0, |r| r.len());
        let mut out = vec![self.ctx.zero(); out_cols];
        for (ci, trow) in c.iter().zip(&self.t) {
            if self.ctx.is_zero(ci) {
                continue;
            }
            for (o, e) in out.iter_mut().zip(trow) {
                *o = self.ctx.add(o, &self.ctx.mul(ci, e));
            }
        }
        out
    }
}

/// Cyclic factors of `Hom(⊕ᵢ O/𝔡ᵢ, ⊕ⱼ O/𝔢ⱼ) ≅ ⊕_{i,j} O/(𝔡ᵢ + 𝔢ⱼ)` for
/// torsion modules given by (nonzero) elementary divisor ideals. The returned
/// factors are in `(i, j)` order, **not** sorted into a divisibility chain —
/// use [`divisibility_chain`] for the canonical elementary divisors.
pub fn torsion_hom_cyclic_factors<C: DedekindContext>(
    ctx: &C,
    ds: &[C::Ideal],
    es: &[C::Ideal],
) -> Vec<C::Ideal> {
    let mut out = Vec::with_capacity(ds.len() * es.len());
    for d in ds {
        for e in es {
            out.push(ctx.ideal_add(d, e));
        }
    }
    out
}

/// Sort a multiset of (nonzero, integral) cyclic-factor ideals into a
/// divisibility chain `𝔠₁ | 𝔠₂ | … ` with the same direct sum, by repeated
/// `(𝔞, 𝔟) → (𝔞+𝔟, 𝔞∩𝔟)` (gcd/lcm) passes — pure ideal arithmetic, valid
/// over any Dedekind domain since `O/𝔞 ⊕ O/𝔟 ≅ O/(𝔞+𝔟) ⊕ O/(𝔞∩𝔟)` by CRT
/// on each prime.
pub fn divisibility_chain<C: DedekindContext>(
    ctx: &C,
    factors: &[C::Ideal],
) -> Result<Vec<C::Ideal>, DedekindError> {
    let mut v: Vec<C::Ideal> = factors.to_vec();
    let k = v.len();
    if k < 2 {
        return Ok(v);
    }
    // Each effective step strictly enlarges an earlier ideal, so the process
    // terminates (ACC); the pass bound is defensive against a buggy context.
    let max_passes = 64 + k * k;
    for _ in 0..max_passes {
        let mut changed = false;
        for i in 0..k {
            for j in (i + 1)..k {
                if ctx.ideal_subset(&v[j], &v[i]) {
                    continue; // already 𝔞ᵢ | 𝔞ⱼ
                }
                let g = ctx.ideal_add(&v[i], &v[j]);
                let l = ctx.ideal_intersect(&v[i], &v[j]);
                v[i] = g;
                v[j] = l;
                changed = true;
            }
        }
        if !changed {
            return Ok(v);
        }
    }
    Err(DedekindError::Internal(
        "divisibility chain did not stabilize — inconsistent ideal arithmetic".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dedekind::nfctx::NfDedekind;
    use crate::dedekind::zctx::{ZDedekind, ZIdeal};
    use rustmath_core::morphism::compose;
    use rustmath_rationals::Rational;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }
    fn qq(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }

    #[test]
    fn z_hom_coefficient_ideals_and_membership() {
        let ctx = ZDedekind;
        // M = 2ℤ·e₁ ⊕ 3ℤ·e₂ ⊆ ℚ², N = 4ℤ·e₁ ⊆ ℚ (as pseudo-modules in K¹)
        let m = PseudoMatrix::new(
            2,
            vec![
                (ZIdeal::from_int(2).unwrap(), vec![q(1), q(0)]),
                (ZIdeal::from_int(3).unwrap(), vec![q(0), q(1)]),
            ],
        )
        .unwrap();
        let n = PseudoMatrix::new(1, vec![(ZIdeal::from_int(4).unwrap(), vec![q(1)])]).unwrap();
        let hom = hom_module(&ctx, &m, &n).unwrap();
        assert_eq!(hom.rank(), 2);
        // coeff ideals: 4/2 = (2), 4/3 = (4/3)
        assert_eq!(hom.coefficient_ideal(0, 0), &ZIdeal::from_int(2).unwrap());
        assert_eq!(hom.coefficient_ideal(1, 0), &ZIdeal::new(qq(4, 3)).unwrap());
        // T = [[1],[x]] is not a hom (1 ∉ 2ℤ); T = [[2],[4/3]] is
        assert!(!hom.is_hom(&ctx, &[vec![q(1)], vec![q(0)]]));
        assert!(hom.is_hom(&ctx, &[vec![q(2)], vec![qq(4, 3)]]));
        // Steinitz ideal of Hom = (2)·(4/3) = (8/3)
        assert_eq!(hom.steinitz_ideal(&ctx), ZIdeal::new(qq(8, 3)).unwrap());
        // Hom as a module: rank-2 pseudo-matrix with those ideals
        let hp = hom.as_pseudo_matrix(&ctx);
        assert_eq!(hp.nrows(), 2);
        assert_eq!(hp.ideals()[0], ZIdeal::from_int(2).unwrap());
        assert_eq!(hp.ideals()[1], ZIdeal::new(qq(4, 3)).unwrap());
    }

    #[test]
    fn z_pseudo_hom_is_a_core_morphism_and_composes() {
        let ctx = ZDedekind;
        let t1 = PseudoHom::new(&ctx, vec![vec![q(2), q(0)], vec![q(1), q(3)]]);
        let t2 = PseudoHom::new(&ctx, vec![vec![q(0), q(1)], vec![q(1), q(0)]]);
        // (1, 1)·T₁ = (3, 3); then swap → (3, 3)
        assert_eq!(t1.apply(&vec![q(1), q(1)]), vec![q(3), q(3)]);
        let both = compose(t1, t2);
        assert_eq!(both.apply(&vec![q(1), q(1)]), vec![q(3), q(3)]);
        // matrix composition agrees with the composed morphism
        let m1 = vec![vec![q(2), q(0)], vec![q(1), q(3)]];
        let m2 = vec![vec![q(0), q(1)], vec![q(1), q(0)]];
        let prod = compose_homs(&ctx, &m1, &m2).unwrap();
        let as_one = PseudoHom::new(&ctx, prod);
        assert_eq!(as_one.apply(&vec![q(1), q(1)]), vec![q(3), q(3)]);
    }

    /// Hom(ℤ/4 ⊕ ℤ/6, ℤ/8) ≅ ℤ/gcd(4,8) ⊕ ℤ/gcd(6,8) = ℤ/4 ⊕ ℤ/2,
    /// elementary divisor chain (2, 4) — verified independently with sympy.
    #[test]
    fn z_torsion_hom() {
        let ctx = ZDedekind;
        let ds = vec![ZIdeal::from_int(4).unwrap(), ZIdeal::from_int(6).unwrap()];
        let es = vec![ZIdeal::from_int(8).unwrap()];
        let factors = torsion_hom_cyclic_factors(&ctx, &ds, &es);
        assert_eq!(
            factors,
            vec![ZIdeal::from_int(4).unwrap(), ZIdeal::from_int(2).unwrap()]
        );
        let chain = divisibility_chain(&ctx, &factors).unwrap();
        assert_eq!(
            chain,
            vec![ZIdeal::from_int(2).unwrap(), ZIdeal::from_int(4).unwrap()]
        );
    }

    #[test]
    fn z_divisibility_chain_general() {
        let ctx = ZDedekind;
        let factors: Vec<ZIdeal> = [6, 10, 15]
            .iter()
            .map(|&n| ZIdeal::from_int(n).unwrap())
            .collect();
        let chain = divisibility_chain(&ctx, &factors).unwrap();
        // ℤ/6 ⊕ ℤ/10 ⊕ ℤ/15 ≅ ℤ/1 ⊕ ℤ/30 ⊕ ℤ/30: gcd-content 1, then 30, 30
        assert_eq!(
            chain,
            vec![
                ZIdeal::from_int(1).unwrap(),
                ZIdeal::from_int(30).unwrap(),
                ZIdeal::from_int(30).unwrap()
            ]
        );
        // sanity: product (= group order 900) is preserved
    }

    #[test]
    fn nf_end_of_p_oplus_o() {
        // K = ℚ(√-5), M = 𝔭₂·e₁ ⊕ O·e₂: End coefficient ideals
        // [[O, 𝔭⁻¹], [𝔭, O]]; the swap matrix is not an endomorphism but
        // 2·(v₂ ↦ v₁) is.
        let k = NfDedekind::from_i64_poly(&[5, 0, 1]);
        let p2 = k
            .ideal_from_elems(&[k.elem(&[2, 0], 1), k.elem(&[1, 1], 1)])
            .unwrap();
        let m = PseudoMatrix::new(
            2,
            vec![
                (p2.clone(), vec![k.one(), k.zero()]),
                (k.unit_ideal(), vec![k.zero(), k.one()]),
            ],
        )
        .unwrap();
        let end = end_module(&k, &m).unwrap();
        assert_eq!(end.rank(), 4);
        assert_eq!(end.coefficient_ideal(0, 0), &k.unit_ideal());
        assert_eq!(end.coefficient_ideal(0, 1), &k.ideal_inv(&p2));
        assert_eq!(end.coefficient_ideal(1, 0), &p2);
        assert_eq!(end.coefficient_ideal(1, 1), &k.unit_ideal());
        // identity is an endomorphism
        let id = vec![
            vec![k.one(), k.zero()],
            vec![k.zero(), k.one()],
        ];
        assert!(end.is_hom(&k, &id));
        // swap: needs 1 ∈ 𝔭 → false
        let swap = vec![
            vec![k.zero(), k.one()],
            vec![k.one(), k.zero()],
        ];
        assert!(!end.is_hom(&k, &swap));
        // 2·E₂₁ (v₂ ↦ 2v₁): 2 ∈ 𝔭 → a genuine endomorphism
        let two_e21 = vec![
            vec![k.zero(), k.zero()],
            vec![k.elem(&[2, 0], 1), k.zero()],
        ];
        assert!(end.is_hom(&k, &two_e21));
        // Steinitz ideal of End: O·𝔭⁻¹·𝔭·O = O
        assert_eq!(end.steinitz_ideal(&k), k.unit_ideal());
    }

    #[test]
    fn nf_torsion_hom_qi() {
        // O = ℤ[i]: Hom(O/(1+i)², O/(2)) — note ((1+i)²) = (2), so the single
        // cyclic factor is (2) + (2) = (2).
        let k = NfDedekind::from_i64_poly(&[1, 0, 1]);
        let p = k.principal_ideal(&k.elem(&[1, 1], 1)).unwrap();
        let psq = k.ideal_mul(&p, &p);
        let two = k.principal_ideal(&k.elem(&[2, 0], 1)).unwrap();
        assert_eq!(psq, two);
        let factors = torsion_hom_cyclic_factors(&k, &[psq], &[two.clone()]);
        assert_eq!(factors, vec![two.clone()]);
        // Hom(O/(1+i), O/(2)) = O/(1+i)
        let factors2 = torsion_hom_cyclic_factors(&k, &[p.clone()], &[two]);
        assert_eq!(factors2, vec![p]);
    }
}
