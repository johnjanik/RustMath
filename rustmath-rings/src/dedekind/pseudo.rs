//! Pseudo-matrices over a Dedekind domain: Hermite-like reduction to a
//! pseudo-basis (Cohen, GTM 193, Algorithm 1.4.7 flavor), module membership,
//! Steinitz class, and elementary divisors of `Oⁿ/M` via determinantal
//! (Fitting) ideals.

use super::{DedekindContext, DedekindError, IsoDecision, Principality};

/// A pseudo-matrix `[(𝔞ᵢ, vᵢ)]` over a Dedekind context: the module is
/// `M = Σᵢ 𝔞ᵢ·vᵢ ⊆ Kⁿ` where each `vᵢ ∈ Kⁿ` and each `𝔞ᵢ` is a nonzero
/// fractional ideal.
#[derive(Debug, Clone)]
pub struct PseudoMatrix<C: DedekindContext> {
    ncols: usize,
    rows: Vec<Vec<C::Elem>>,
    ideals: Vec<C::Ideal>,
}

impl<C: DedekindContext> PseudoMatrix<C> {
    /// Build from `(ideal, row)` pairs; every row must have length `ncols`.
    pub fn new(
        ncols: usize,
        pseudo_rows: Vec<(C::Ideal, Vec<C::Elem>)>,
    ) -> Result<Self, DedekindError> {
        for (_, r) in &pseudo_rows {
            if r.len() != ncols {
                return Err(DedekindError::Shape(format!(
                    "row length {} ≠ ncols {}",
                    r.len(),
                    ncols
                )));
            }
        }
        let (ideals, rows) = pseudo_rows.into_iter().unzip();
        Ok(PseudoMatrix { ncols, rows, ideals })
    }

    /// A free module presentation: unit coefficient ideal on every row.
    pub fn from_rows(ctx: &C, ncols: usize, rows: Vec<Vec<C::Elem>>) -> Result<Self, DedekindError> {
        let unit = ctx.unit_ideal();
        Self::new(ncols, rows.into_iter().map(|r| (unit.clone(), r)).collect())
    }

    /// The standard free module `Oⁿ` (identity pseudo-matrix, unit ideals).
    pub fn standard(ctx: &C, n: usize) -> Self {
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut r = vec![ctx.zero(); n];
            r[i] = ctx.one();
            rows.push(r);
        }
        PseudoMatrix {
            ncols: n,
            rows,
            ideals: vec![ctx.unit_ideal(); n],
        }
    }

    /// Ambient dimension (number of columns).
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    /// Number of pseudo-generators (rows).
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }
    /// The matrix rows.
    pub fn rows(&self) -> &[Vec<C::Elem>] {
        &self.rows
    }
    /// The coefficient ideals (one per row).
    pub fn ideals(&self) -> &[C::Ideal] {
        &self.ideals
    }

    fn first_nonzero(ctx: &C, row: &[C::Elem]) -> Option<usize> {
        row.iter().position(|e| !ctx.is_zero(e))
    }

    /// Is this matrix in echelon pseudo-basis form: pivot columns strictly
    /// increasing, every pivot entry exactly `1`, no zero rows?
    pub fn is_echelon(&self, ctx: &C) -> bool {
        let mut last: Option<usize> = None;
        for row in &self.rows {
            match Self::first_nonzero(ctx, row) {
                None => return false,
                Some(j) => {
                    if let Some(l) = last {
                        if j <= l {
                            return false;
                        }
                    }
                    if row[j] != ctx.one() {
                        return false;
                    }
                    last = Some(j);
                }
            }
        }
        true
    }

    /// Hermite-like reduction over the Dedekind domain (Cohen Alg. 1.4.7
    /// flavor): returns an echelon pseudo-matrix spanning the **same module**,
    /// with strictly increasing pivot columns, pivot entries exactly `1`
    /// (matrix content pushed into the coefficient ideals), and zero rows
    /// dropped. The result is a pseudo-basis: the module is the *direct sum*
    /// `⊕ᵢ 𝔞ᵢ·vᵢ` of the returned pseudo-elements.
    ///
    /// Elimination of two rows with entries `a₁, a₂` and ideals `𝔞₁, 𝔞₂` uses
    /// the ideal Bézout step: with `𝔡 = a₁𝔞₁ + a₂𝔞₂` and the idempotent
    /// splitting `e₁ + e₂ = 1`, `eᵢ ∈ aᵢ𝔞ᵢ𝔡⁻¹`, the transformation
    /// `(u, v; a₂, -a₁)` with `u = e₁/a₁`, `v = e₂/a₂` has determinant `-1`
    /// and replaces the ideals by `(𝔡, 𝔞₁𝔞₂𝔡⁻¹)` — an exactly
    /// module-preserving step.
    pub fn hnf(&self, ctx: &C) -> Result<PseudoMatrix<C>, DedekindError> {
        let mut rows = self.rows.clone();
        let mut ideals = self.ideals.clone();
        let nrows = rows.len();
        let mut pivot = 0usize;
        for col in 0..self.ncols {
            // find a row ≥ pivot with a nonzero entry in this column
            let Some(first) = (pivot..nrows).find(|&r| !ctx.is_zero(&rows[r][col])) else {
                continue;
            };
            rows.swap(pivot, first);
            ideals.swap(pivot, first);
            let mut normalized = false;
            for r in (pivot + 1)..nrows {
                if ctx.is_zero(&rows[r][col]) {
                    continue;
                }
                let a1 = rows[pivot][col].clone();
                let a2 = rows[r][col].clone();
                let ia1 = ideals[pivot].clone();
                let ia2 = ideals[r].clone();
                let sa1 = ctx
                    .scaled_ideal(&a1, &ia1)
                    .ok_or_else(|| DedekindError::Internal("zero pivot entry".into()))?;
                let sa2 = ctx
                    .scaled_ideal(&a2, &ia2)
                    .ok_or_else(|| DedekindError::Internal("zero elimination entry".into()))?;
                let d = ctx.ideal_add(&sa1, &sa2);
                let d_inv = ctx.ideal_inv(&d);
                let c1 = ctx.ideal_mul(&sa1, &d_inv);
                let c2 = ctx.ideal_mul(&sa2, &d_inv);
                // c1, c2 are integral and coprime by construction
                let (e1, e2) = ctx.idempotents(&c1, &c2)?;
                let u = ctx
                    .div(&e1, &a1)
                    .ok_or_else(|| DedekindError::Internal("division by zero pivot".into()))?;
                let v = ctx
                    .div(&e2, &a2)
                    .ok_or_else(|| DedekindError::Internal("division by zero entry".into()))?;
                // new_pivot = u·R₁ + v·R₂ (entry `1` at col), ideal 𝔡
                // new_r     = a₂·R₁ - a₁·R₂ (entry `0` at col), ideal 𝔞₁𝔞₂𝔡⁻¹
                let mut new_p = Vec::with_capacity(self.ncols);
                let mut new_r = Vec::with_capacity(self.ncols);
                for j in 0..self.ncols {
                    let x = &rows[pivot][j];
                    let y = &rows[r][j];
                    new_p.push(ctx.add(&ctx.mul(&u, x), &ctx.mul(&v, y)));
                    new_r.push(ctx.sub(&ctx.mul(&a2, x), &ctx.mul(&a1, y)));
                }
                debug_assert!(new_p[col] == ctx.one(), "pivot must be exactly 1");
                debug_assert!(ctx.is_zero(&new_r[col]), "eliminated entry must be exactly 0");
                rows[pivot] = new_p;
                rows[r] = new_r;
                ideals[pivot] = d;
                ideals[r] = ctx.ideal_mul(&ctx.ideal_mul(&ia1, &ia2), &ctx.ideal_inv(&ideals[pivot]));
                normalized = true;
            }
            if !normalized {
                // single nonzero row at this column: scale the pivot to 1,
                // pushing the entry into the ideal: 𝔞·v = (a𝔞)·(v/a)
                let a = rows[pivot][col].clone();
                if a != ctx.one() {
                    let a_inv = ctx
                        .inv(&a)
                        .ok_or_else(|| DedekindError::Internal("zero pivot entry".into()))?;
                    for j in 0..self.ncols {
                        rows[pivot][j] = ctx.mul(&a_inv, &rows[pivot][j]);
                    }
                    ideals[pivot] = ctx
                        .scaled_ideal(&a, &ideals[pivot])
                        .ok_or_else(|| DedekindError::Internal("zero pivot entry".into()))?;
                }
            }
            pivot += 1;
            if pivot == nrows {
                break;
            }
        }
        // drop zero rows (they contribute nothing to the module)
        let mut out_rows = Vec::new();
        let mut out_ideals = Vec::new();
        for (row, ideal) in rows.into_iter().zip(ideals) {
            if Self::first_nonzero(ctx, &row).is_some() {
                out_rows.push(row);
                out_ideals.push(ideal);
            }
        }
        let out = PseudoMatrix {
            ncols: self.ncols,
            rows: out_rows,
            ideals: out_ideals,
        };
        debug_assert!(out.nrows() == 0 || out.is_echelon(ctx));
        Ok(out)
    }

    /// Coordinates of `v` over the rows of an **echelon** pseudo-matrix
    /// (K-linear span); `None` when `v` is outside the span.
    pub fn span_coords(&self, ctx: &C, v: &[C::Elem]) -> Option<Vec<C::Elem>> {
        assert_eq!(v.len(), self.ncols, "vector length must equal ncols");
        assert!(
            self.nrows() == 0 || self.is_echelon(ctx),
            "span_coords requires echelon form (call hnf first)"
        );
        let mut residual = v.to_vec();
        let mut coords = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let j = Self::first_nonzero(ctx, row).expect("echelon rows are nonzero");
            let c = residual[j].clone();
            if !ctx.is_zero(&c) {
                for (res, e) in residual.iter_mut().zip(row) {
                    *res = ctx.sub(res, &ctx.mul(&c, e));
                }
            }
            coords.push(c);
        }
        if residual.iter().all(|e| ctx.is_zero(e)) {
            Some(coords)
        } else {
            None
        }
    }

    /// Is the single element `v ∈ M`? (Requires echelon form.)
    pub fn contains_element(&self, ctx: &C, v: &[C::Elem]) -> bool {
        match self.span_coords(ctx, v) {
            None => false,
            Some(coords) => coords
                .iter()
                .zip(&self.ideals)
                .all(|(c, ideal)| ctx.ideal_contains(ideal, c)),
        }
    }

    /// Is the pseudo-element `𝔞·v ⊆ M`? (Requires echelon form.)
    pub fn contains_pseudo_gen(&self, ctx: &C, a: &C::Ideal, v: &[C::Elem]) -> bool {
        match self.span_coords(ctx, v) {
            None => false,
            Some(coords) => coords.iter().zip(&self.ideals).all(|(c, ideal)| {
                match ctx.scaled_ideal(c, a) {
                    None => true, // zero coefficient
                    Some(scaled) => ctx.ideal_subset(&scaled, ideal),
                }
            }),
        }
    }

    /// Is `other ⊆ self` as modules? (`self` must be echelon; `other` need not.)
    pub fn contains_module(&self, ctx: &C, other: &PseudoMatrix<C>) -> bool {
        other
            .rows
            .iter()
            .zip(&other.ideals)
            .all(|(row, ideal)| self.contains_pseudo_gen(ctx, ideal, row))
    }

    /// Exact module equality, decided by mutual inclusion of pseudo-generators
    /// after Hermite reduction.
    pub fn module_eq(ctx: &C, a: &PseudoMatrix<C>, b: &PseudoMatrix<C>) -> Result<bool, DedekindError> {
        let ha = a.hnf(ctx)?;
        let hb = b.hnf(ctx)?;
        Ok(ha.contains_module(ctx, b) && hb.contains_module(ctx, a))
    }

    /// The Steinitz-class representative `𝔞₁···𝔞ₖ` of `M`: by the Steinitz
    /// theorem `M ≅ O^{k-1} ⊕ (𝔞₁···𝔞ₖ)`, and the ideal class of the product
    /// is a complete isomorphism invariant together with the rank `k`.
    /// (The zero module returns the unit ideal by convention.)
    pub fn steinitz_ideal(&self, ctx: &C) -> Result<C::Ideal, DedekindError> {
        let h = self.hnf(ctx)?;
        let mut s = ctx.unit_ideal();
        for ideal in &h.ideals {
            s = ctx.ideal_mul(&s, ideal);
        }
        Ok(s)
    }

    /// Is `M` free? Decided exactly when the bounded principality search
    /// certifies the Steinitz ideal principal: `Principal(g)` means
    /// `M ≅ Oᵏ` (with `(g) = 𝔞₁···𝔞ₖ`). `Unresolved` is **not** a decision —
    /// deciding non-freeness needs class-group data. See [`Principality`].
    pub fn steinitz_is_trivial(&self, ctx: &C) -> Result<Principality<C::Elem>, DedekindError> {
        Ok(ctx.principal_generator(&self.steinitz_ideal(ctx)?))
    }

    /// Isomorphism test for the (projective) modules spanned by two
    /// pseudo-matrices: rank + Steinitz class classify them completely.
    /// Unequal ranks decide `NotIsomorphic`; equal ranks decide `Isomorphic`
    /// only when `s(A)·s(B)⁻¹` is *certified* principal, and stay honestly
    /// `Unresolved` otherwise (deciding non-triviality of an ideal class needs
    /// class-group discrete logarithms this crate cannot compute).
    pub fn is_isomorphic(ctx: &C, a: &PseudoMatrix<C>, b: &PseudoMatrix<C>) -> Result<IsoDecision, DedekindError> {
        let ha = a.hnf(ctx)?;
        let hb = b.hnf(ctx)?;
        if ha.nrows() != hb.nrows() {
            return Ok(IsoDecision::NotIsomorphic);
        }
        let sa = ha.steinitz_ideal(ctx)?;
        let sb = hb.steinitz_ideal(ctx)?;
        let quot = ctx.ideal_mul(&sa, &ctx.ideal_inv(&sb));
        match ctx.principal_generator(&quot) {
            Principality::Principal(_) => Ok(IsoDecision::Isomorphic),
            Principality::Unresolved => Ok(IsoDecision::Unresolved),
        }
    }

    /// Elementary divisors of the quotient `Oⁿ/M` (with `n = ncols`), computed
    /// from determinantal (Fitting) ideals:
    /// `F_k = Σ_{|R|=|C|=k} det(A[R,C])·∏_{i∈R} 𝔞ᵢ` and `𝔡_k = F_k·F_{k-1}⁻¹`.
    /// Over a Dedekind domain the Fitting ideals localize, so
    /// `Oⁿ/M ≅ ⊕_k O/𝔡_k ⊕ O^{n-r}` with the divisibility chain
    /// `𝔡₁ | 𝔡₂ | … | 𝔡_r`.
    ///
    /// Errors with [`DedekindError::NotIntegral`] when `M ⊄ Oⁿ`.
    pub fn elementary_divisors(&self, ctx: &C) -> Result<ElementaryDivisors<C>, DedekindError> {
        let h = self.hnf(ctx)?;
        let unit = ctx.unit_ideal();
        // integrality: every pseudo-generator must lie in Oⁿ
        for (row, ideal) in h.rows.iter().zip(&h.ideals) {
            for e in row {
                if let Some(scaled) = ctx.scaled_ideal(e, ideal) {
                    if !ctx.ideal_subset(&scaled, &unit) {
                        return Err(DedekindError::NotIntegral(
                            "a pseudo-generator entry times its ideal is not integral".into(),
                        ));
                    }
                }
            }
        }
        let r = h.nrows();
        let n = h.ncols;
        // guard the combinatorial explosion honestly
        let mut work: u128 = 0;
        for k in 1..=r {
            work = work.saturating_add(binomial(r, k).saturating_mul(binomial(n, k)));
        }
        if work > 2_000_000 {
            return Err(DedekindError::Shape(
                "minor-based elementary divisors would need too many determinants".into(),
            ));
        }
        let mut torsion = Vec::with_capacity(r);
        let mut f_prev = ctx.unit_ideal(); // F₀ = O
        for k in 1..=r {
            let mut f_k: Option<C::Ideal> = None;
            for rows_sel in k_subsets(r, k) {
                // ∏_{i∈R} 𝔞ᵢ
                let mut prod = ctx.unit_ideal();
                for &i in &rows_sel {
                    prod = ctx.ideal_mul(&prod, &h.ideals[i]);
                }
                for cols_sel in k_subsets(n, k) {
                    let d = minor_det(ctx, &h.rows, &rows_sel, &cols_sel);
                    if ctx.is_zero(&d) {
                        continue;
                    }
                    let term = ctx
                        .scaled_ideal(&d, &prod)
                        .expect("nonzero determinant scales an ideal");
                    f_k = Some(match f_k {
                        None => term,
                        Some(acc) => ctx.ideal_add(&acc, &term),
                    });
                }
            }
            let f_k = f_k.ok_or_else(|| {
                DedekindError::Internal(
                    "echelon pseudo-matrix of rank r must have a nonzero k×k minor for k ≤ r".into(),
                )
            })?;
            let d_k = ctx.ideal_mul(&f_k, &ctx.ideal_inv(&f_prev));
            if !ctx.ideal_is_integral(&d_k) {
                return Err(DedekindError::Internal(
                    "elementary divisor quotient F_k/F_{k-1} must be integral".into(),
                ));
            }
            torsion.push(d_k);
            f_prev = f_k;
        }
        // divisibility chain sanity: 𝔡_k | 𝔡_{k+1}, i.e. 𝔡_{k+1} ⊆ 𝔡_k
        for w in torsion.windows(2) {
            if !ctx.ideal_subset(&w[1], &w[0]) {
                return Err(DedekindError::Internal(
                    "elementary divisors must form a divisibility chain".into(),
                ));
            }
        }
        Ok(ElementaryDivisors {
            torsion,
            free_rank: n - r,
        })
    }
}

/// The structure of a quotient `Oⁿ/M ≅ O/𝔡₁ ⊕ … ⊕ O/𝔡_r ⊕ O^{free_rank}`
/// with the divisibility chain `𝔡₁ | 𝔡₂ | … | 𝔡_r` (unit ideals allowed —
/// they contribute trivial summands and align with SNF diagonals).
#[derive(Debug, Clone)]
pub struct ElementaryDivisors<C: DedekindContext> {
    /// The elementary divisor ideals, smallest (most divisible into) first.
    pub torsion: Vec<C::Ideal>,
    /// The free rank of the quotient.
    pub free_rank: usize,
}

/// The explicit `O^{k-1} ⊕ 𝔞` normal form (a pseudo-basis whose first `k-1`
/// ideals are the unit ideal).
///
/// HONEST UNIMPLEMENTED: the transformation (Cohen Alg. 1.3.16 flavor) needs
/// ideal *reduction* / two-element representations steered by class-group
/// data, which this crate cannot compute. The class *invariant* is available
/// exactly via [`PseudoMatrix::steinitz_ideal`]; only the explicit basis
/// transformation is missing.
pub fn steinitz_normal_form<C: DedekindContext>(
    _ctx: &C,
    _m: &PseudoMatrix<C>,
) -> Result<PseudoMatrix<C>, DedekindError> {
    Err(DedekindError::NeedsClassGroup(
        "explicit O^{k-1} ⊕ 𝔞 pseudo-basis needs ideal reduction / two-element \
         representations; the Steinitz invariant itself is available via steinitz_ideal()"
            .into(),
    ))
}

fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc.saturating_mul((n - i) as u128) / (i as u128 + 1);
    }
    acc
}

/// All k-element subsets of `{0, …, n-1}` in lexicographic order.
fn k_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(k);
    fn rec(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        let need = k - cur.len();
        for i in start..=(n - need) {
            cur.push(i);
            rec(i + 1, n, k, cur, out);
            cur.pop();
        }
    }
    if k <= n {
        rec(0, n, k, &mut cur, &mut out);
    }
    out
}

/// Determinant of the submatrix `rows[R][C]` by Laplace expansion along the
/// first selected row (exact field arithmetic; intended for small `k`).
fn minor_det<C: DedekindContext>(
    ctx: &C,
    rows: &[Vec<C::Elem>],
    rsel: &[usize],
    csel: &[usize],
) -> C::Elem {
    if rsel.len() == 1 {
        return rows[rsel[0]][csel[0]].clone();
    }
    let mut acc = ctx.zero();
    let sub_r = &rsel[1..];
    for (pos, &c) in csel.iter().enumerate() {
        let a = &rows[rsel[0]][c];
        if ctx.is_zero(a) {
            continue;
        }
        let sub_c: Vec<usize> = csel
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &cc)| cc)
            .collect();
        let cofactor = minor_det(ctx, rows, sub_r, &sub_c);
        let term = ctx.mul(a, &cofactor);
        acc = if pos % 2 == 0 {
            ctx.add(&acc, &term)
        } else {
            ctx.sub(&acc, &term)
        };
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dedekind::nfctx::NfDedekind;
    use crate::dedekind::zctx::{ZDedekind, ZIdeal};
    use rustmath_integers::Integer;
    use rustmath_matrix::Matrix;
    use rustmath_rationals::Rational;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }
    fn qq(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }
    fn zrows(rows: &[&[i64]]) -> Vec<Vec<Rational>> {
        rows.iter().map(|r| r.iter().map(|&x| q(x)).collect()).collect()
    }

    #[test]
    fn z_hnf_is_echelon_and_module_preserving() {
        let ctx = ZDedekind;
        let m = PseudoMatrix::from_rows(&ctx, 2, zrows(&[&[6, 10], &[15, 4]])).unwrap();
        let h = m.hnf(&ctx).unwrap();
        assert!(h.is_echelon(&ctx));
        assert_eq!(h.nrows(), 2);
        assert!(PseudoMatrix::module_eq(&ctx, &m, &h).unwrap());
        // (6,10) ∈ M; (1,0) ∉ M (6a+15b = 1 is impossible mod 3)
        assert!(h.contains_element(&ctx, &[q(6), q(10)]));
        assert!(h.contains_element(&ctx, &[q(15), q(4)]));
        assert!(!h.contains_element(&ctx, &[q(1), q(0)]));
        // cross-check the lattice against rustmath-matrix HNF rows
        let a = Matrix::from_vec(
            2,
            2,
            vec![
                Integer::from(6),
                Integer::from(10),
                Integer::from(15),
                Integer::from(4),
            ],
        )
        .unwrap();
        let mut hf = a.hermite_normal_form().unwrap();
        hf.canonicalize().unwrap();
        let hnf_rows: Vec<Vec<Rational>> = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| Rational::from_integer(hf.h.get(i, j).unwrap().clone()))
                    .collect()
            })
            .collect();
        let from_hnf = PseudoMatrix::from_rows(&ctx, 2, hnf_rows).unwrap();
        assert!(PseudoMatrix::module_eq(&ctx, &m, &from_hnf).unwrap());
    }

    #[test]
    fn z_hnf_drops_dependent_rows() {
        let ctx = ZDedekind;
        let m = PseudoMatrix::from_rows(&ctx, 3, zrows(&[&[1, 2, 3], &[2, 4, 6]])).unwrap();
        let h = m.hnf(&ctx).unwrap();
        assert_eq!(h.nrows(), 1);
        assert!(PseudoMatrix::module_eq(&ctx, &m, &h).unwrap());
        // the surviving pseudo-generator ideal must make 𝔞·v = ℤ·(1,2,3)
        assert!(h.contains_element(&ctx, &[q(1), q(2), q(3)]));
        assert!(!h.contains_element(&ctx, &[q(1), q(2), q(4)]));
    }

    /// Deliverable (3): the PID special case cross-checked against matrix SNF
    /// on three examples, expected values verified independently with sympy:
    /// [[2,4],[6,8]] → (2,4); diag(2,3,4) → (1,2,12); [[1,2,3],[2,4,6]] → (1)
    /// with free rank 2.
    #[test]
    fn z_elementary_divisors_match_matrix_snf_three_examples() {
        let ctx = ZDedekind;
        let examples: [(&[&[i64]], usize); 3] = [
            (&[&[2, 4], &[6, 8]], 2),
            (&[&[2, 0, 0], &[0, 3, 0], &[0, 0, 4]], 3),
            (&[&[1, 2, 3], &[2, 4, 6]], 3),
        ];
        for (rows, n) in examples {
            let m = PseudoMatrix::from_rows(&ctx, n, zrows(rows)).unwrap();
            let ed = m.elementary_divisors(&ctx).unwrap();
            // independent computation: SNF from rustmath-matrix
            let data: Vec<Integer> = rows
                .iter()
                .flat_map(|r| r.iter().map(|&x| Integer::from(x)))
                .collect();
            let a = Matrix::from_vec(rows.len(), n, data).unwrap();
            let mut snf = a.smith_normal_form().unwrap();
            snf.canonicalize_signs();
            let mut diag: Vec<Integer> = (0..rows.len().min(n))
                .map(|i| snf.s.get(i, i).unwrap().clone())
                .collect();
            // drop zero diagonal entries (they are free rank, not torsion)
            diag.retain(|d| !d.is_zero());
            assert_eq!(
                ed.torsion.len(),
                diag.len(),
                "torsion rank mismatch vs SNF for {rows:?}"
            );
            for (mine, theirs) in ed.torsion.iter().zip(&diag) {
                assert_eq!(
                    mine,
                    &ZIdeal::new(Rational::from_integer(theirs.clone())).unwrap(),
                    "invariant factor mismatch vs SNF for {rows:?}"
                );
            }
            assert_eq!(ed.free_rank, n - diag.len(), "free rank mismatch for {rows:?}");
        }
    }

    #[test]
    fn z_fractional_pseudo_basis_and_honest_integrality_error() {
        let ctx = ZDedekind;
        // M = (1/2)ℤ·e₁ ⊕ 3ℤ·e₂
        let m = PseudoMatrix::new(
            2,
            vec![
                (ZIdeal::new(qq(1, 2)).unwrap(), vec![q(1), q(0)]),
                (ZIdeal::from_int(3).unwrap(), vec![q(0), q(1)]),
            ],
        )
        .unwrap();
        let s = m.steinitz_ideal(&ctx).unwrap();
        assert_eq!(s, ZIdeal::new(qq(3, 2)).unwrap());
        match m.steinitz_is_trivial(&ctx).unwrap() {
            Principality::Principal(g) => assert_eq!(g, qq(3, 2)),
            Principality::Unresolved => panic!("ℤ ideals are always principal"),
        }
        // M ⊄ ℤ²: the quotient construction must refuse honestly
        assert!(matches!(
            m.elementary_divisors(&ctx),
            Err(DedekindError::NotIntegral(_))
        ));
        // membership honours the fractional ideals: (1/2, 0) ∈ M, (1/4, 0) ∉ M
        assert!(m.contains_element(&ctx, &[qq(1, 2), q(0)]));
        assert!(!m.contains_element(&ctx, &[qq(1, 4), q(0)]));
        assert!(!m.contains_element(&ctx, &[q(0), q(1)]));
        assert!(m.contains_element(&ctx, &[q(0), q(3)]));
    }

    #[test]
    fn z_isomorphism_decisions() {
        let ctx = ZDedekind;
        let a = PseudoMatrix::from_rows(&ctx, 2, zrows(&[&[2, 0], &[0, 3]])).unwrap();
        let b = PseudoMatrix::standard(&ctx, 2);
        // both are free of rank 2 over ℤ
        assert_eq!(
            PseudoMatrix::is_isomorphic(&ctx, &a, &b).unwrap(),
            IsoDecision::Isomorphic
        );
        let c = PseudoMatrix::from_rows(&ctx, 2, zrows(&[&[1, 1]])).unwrap();
        assert_eq!(
            PseudoMatrix::is_isomorphic(&ctx, &a, &c).unwrap(),
            IsoDecision::NotIsomorphic
        );
    }

    #[test]
    fn steinitz_normal_form_is_honestly_unimplemented() {
        let ctx = ZDedekind;
        let m = PseudoMatrix::standard(&ctx, 2);
        assert!(matches!(
            steinitz_normal_form(&ctx, &m),
            Err(DedekindError::NeedsClassGroup(_))
        ));
    }

    // ---- number-field tests ----

    fn w5() -> NfDedekind {
        NfDedekind::from_i64_poly(&[5, 0, 1]) // ℚ(√-5), h = 2
    }

    #[test]
    fn nf_hnf_collapses_generators_to_prime_ideal() {
        let k = w5();
        // M = O·(2,0) + O·(1+w,0) = 𝔭₂·e₁ where 𝔭₂ = (2, 1+w)
        let m = PseudoMatrix::from_rows(
            &k,
            2,
            vec![
                vec![k.elem(&[2, 0], 1), k.zero()],
                vec![k.elem(&[1, 1], 1), k.zero()],
            ],
        )
        .unwrap();
        let h = m.hnf(&k).unwrap();
        assert_eq!(h.nrows(), 1);
        assert!(h.is_echelon(&k));
        let p2 = k
            .ideal_from_elems(&[k.elem(&[2, 0], 1), k.elem(&[1, 1], 1)])
            .unwrap();
        assert_eq!(&h.ideals()[0], &p2);
        assert_eq!(h.rows()[0], vec![k.one(), k.zero()]);
        assert!(PseudoMatrix::module_eq(&k, &m, &h).unwrap());
    }

    #[test]
    fn nf_steinitz_p_oplus_p_is_free_but_p_oplus_o_unresolved() {
        let k = w5();
        let p2 = k
            .ideal_from_elems(&[k.elem(&[2, 0], 1), k.elem(&[1, 1], 1)])
            .unwrap();
        let e1 = vec![k.one(), k.zero()];
        let e2 = vec![k.zero(), k.one()];
        // 𝔭₂·e₁ ⊕ 𝔭₂·e₂: Steinitz ideal 𝔭₂² = (2) is principal → free (≅ O²)
        let m = PseudoMatrix::new(2, vec![(p2.clone(), e1.clone()), (p2.clone(), e2.clone())])
            .unwrap();
        let s = m.steinitz_ideal(&k).unwrap();
        assert_eq!(s, k.principal_ideal(&k.elem(&[2, 0], 1)).unwrap());
        match m.steinitz_is_trivial(&k).unwrap() {
            Principality::Principal(g) => {
                assert_eq!(k.principal_ideal(&g).unwrap(), s);
            }
            Principality::Unresolved => panic!("𝔭₂² = (2): the generator 2 must be found"),
        }
        // 𝔭₂·e₁ ⊕ O·e₂: Steinitz ideal 𝔭₂ is non-principal; the bounded search
        // must stay honestly Unresolved (deciding needs class-group data)
        let m2 = PseudoMatrix::new(2, vec![(p2.clone(), e1), (k.unit_ideal(), e2)]).unwrap();
        assert_eq!(m2.steinitz_ideal(&k).unwrap(), p2);
        assert!(matches!(
            m2.steinitz_is_trivial(&k).unwrap(),
            Principality::Unresolved
        ));
        // …and the isomorphism test with O² stays Unresolved as well (they are
        // in fact NOT isomorphic, but certifying that needs Cl(K))
        let free = PseudoMatrix::standard(&k, 2);
        assert_eq!(
            PseudoMatrix::is_isomorphic(&k, &m2, &free).unwrap(),
            IsoDecision::Unresolved
        );
        // while 𝔭⊕𝔭 vs O² is certified isomorphic
        assert_eq!(
            PseudoMatrix::is_isomorphic(&k, &m, &free).unwrap(),
            IsoDecision::Isomorphic
        );
    }

    #[test]
    fn nf_elementary_divisors_qi() {
        // O = ℤ[i]; M = (1+i)O·e₁ ⊕ 2O·e₂ ⊆ O²:
        // F₁ = (1+i, 2) = (1+i), F₂ = (2(1+i)) → 𝔡₁ = (1+i), 𝔡₂ = (2).
        let k = NfDedekind::from_i64_poly(&[1, 0, 1]);
        let one_plus_i = k.elem(&[1, 1], 1);
        let two = k.elem(&[2, 0], 1);
        let m = PseudoMatrix::from_rows(
            &k,
            2,
            vec![
                vec![one_plus_i.clone(), k.zero()],
                vec![k.zero(), two.clone()],
            ],
        )
        .unwrap();
        let ed = m.elementary_divisors(&k).unwrap();
        assert_eq!(ed.free_rank, 0);
        assert_eq!(ed.torsion.len(), 2);
        assert_eq!(ed.torsion[0], k.principal_ideal(&one_plus_i).unwrap());
        assert_eq!(ed.torsion[1], k.principal_ideal(&two).unwrap());
    }
}
