//! # Residue-field towers over GF(p) with certified polynomial factorization
//!
//! The MacLane/OM machinery walks a chain of inductive valuations
//! `w_0 < w_1 < ... < w_k`; the residue field of `w_j` is a finite extension
//! `kappa_j` of `GF(p)` built as a tower
//!
//! ```text
//! kappa_0 = GF(p),   kappa_j = kappa_{j-1}[y_{j-1}] / (psi_j)
//! ```
//!
//! where `psi_j` is the (monicized) residual polynomial of the key `phi_j`
//! with respect to `w_{j-1}` (irreducible over `kappa_{j-1}` by the key
//! property). This module implements that tower directly — elements are
//! nested coefficient vectors, no flattening to a primitive element is
//! needed — together with the polynomial arithmetic over the top field that
//! `mac_lane_step` needs, most importantly **certified factorization**:
//!
//! - squarefree reduction handling inseparable inputs by the correct
//!   `p`-th-root reduction `f(y) = g(y^p) => g` (coefficient-wise inverse
//!   Frobenius `c -> c^{q/p}`); this mirrors the fix for the KNOWN HAZARD in
//!   `rustmath_polynomials::fp_factor::factor`, which is wrong on inseparable
//!   inputs — nothing here trusts that routine,
//! - distinct-degree factorization via `y^{q^i} mod f`,
//! - equal-degree splitting: Cantor–Zassenhaus for odd `q`, the GF(2)-trace
//!   map `r + r^2 + ... + r^{2^{Fd-1}}` for `p = 2`,
//! - **certification**: every reported factor is verified monic and
//!   irreducible (Rabin's criterion over the tower), multiplicities are
//!   established by explicit trial division, and the product of the factors
//!   is re-multiplied and compared with the input. Any mismatch is an `Err`,
//!   never a silent lie. The randomized splitting uses a deterministic LCG;
//!   if it fails to split within the iteration guard the result is an
//!   honest `Err(NumericalError)` (possible incompleteness, never a wrong
//!   answer).
//!
//! Scope: the fields stay small in OM trees (`q^d` bounded by the tree
//! data), and all arithmetic is exact `i64`-mod-`p` at the leaves (enforced
//! `p < 2^31` upstream).

use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use std::fmt;

/// An element of a level of the tower: `Base` for `GF(p)` (value in
/// `[0, p)`), `Ext` for `kappa_j = kappa_{j-1}[y]/(psi_j)` with **exactly**
/// `deg psi_j` coefficients at level `j-1` (fixed length, so derived
/// equality is canonical).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TowerElt {
    /// An element of `GF(p)`, in `[0, p)`.
    Base(i64),
    /// An element of an extension level, as `deg psi` coefficients of the
    /// previous level.
    Ext(Vec<TowerElt>),
}

/// A tower `GF(p) = kappa_0 ⊂ kappa_1 ⊂ ... ⊂ kappa_L` of finite fields,
/// each level a quotient by a certified-irreducible monic modulus.
#[derive(Clone, Debug)]
pub struct ResidueTower {
    p: i64,
    /// `moduli[j]` is `psi_{j+1}`: a monic polynomial over level `j`
    /// (little-endian, trimmed, degree >= 1).
    moduli: Vec<Vec<TowerElt>>,
}

impl ResidueTower {
    /// The trivial tower `kappa = GF(p)`. `p` must be prime (enforced by the
    /// callers via `PAdicBaseValuation`); this checks only `2 <= p < 2^31`.
    pub fn new(p: i64) -> Result<Self> {
        if !(2..(1_i64 << 31)).contains(&p) {
            return Err(MathError::InvalidArgument(format!(
                "ResidueTower: p = {} out of range",
                p
            )));
        }
        Ok(Self { p, moduli: Vec::new() })
    }

    /// The characteristic `p`.
    pub fn p(&self) -> i64 {
        self.p
    }

    /// Number of extension levels (`0` = plain `GF(p)`).
    pub fn levels(&self) -> usize {
        self.moduli.len()
    }

    /// The absolute degree `F = [kappa_L : GF(p)]`.
    pub fn degree(&self) -> u64 {
        self.moduli.iter().map(|m| (m.len() - 1) as u64).product()
    }

    /// Degree of the modulus at level `l` (1-based: `psi_l`).
    pub fn modulus_degree(&self, l: usize) -> usize {
        self.moduli[l - 1].len() - 1
    }

    /// The modulus `psi_l` (1-based), a monic polynomial over level `l-1`.
    pub fn modulus_at(&self, l: usize) -> &[TowerElt] {
        &self.moduli[l - 1]
    }

    /// Extend the tower by a monic modulus `psi` (a polynomial over the
    /// current top level, little-endian). `psi` is verified monic of degree
    /// >= 1 and irreducible over the current top field.
    pub fn push_level(&mut self, psi: Vec<TowerElt>) -> Result<()> {
        let l = self.levels();
        let psi = self.ptrim(l, psi);
        let d = psi.len().saturating_sub(1);
        if d < 1 {
            return Err(MathError::InvalidArgument(
                "ResidueTower::push_level: modulus must have degree >= 1".to_string(),
            ));
        }
        if psi.last() != Some(&self.e_one(l)) {
            return Err(MathError::InvalidArgument(
                "ResidueTower::push_level: modulus must be monic".to_string(),
            ));
        }
        if !self.is_irreducible_at(l, &psi)? {
            return Err(MathError::InvalidArgument(
                "ResidueTower::push_level: modulus is not irreducible over the current top level"
                    .to_string(),
            ));
        }
        self.moduli.push(psi);
        Ok(())
    }

    /// The truncated tower with the top `self.levels() - l` levels dropped.
    pub fn truncate(&self, l: usize) -> Self {
        Self {
            p: self.p,
            moduli: self.moduli[..l].to_vec(),
        }
    }

    // ------------------------------------------------------------------
    // Element arithmetic (level-indexed internals; `level` = which kappa)
    // ------------------------------------------------------------------

    /// The zero of level `level`.
    pub fn e_zero(&self, level: usize) -> TowerElt {
        if level == 0 {
            TowerElt::Base(0)
        } else {
            let d = self.moduli[level - 1].len() - 1;
            TowerElt::Ext(vec![self.e_zero(level - 1); d])
        }
    }

    /// The one of level `level`.
    pub fn e_one(&self, level: usize) -> TowerElt {
        self.e_constant(level, 1)
    }

    /// The image of the integer `c` in level `level`.
    pub fn e_constant(&self, level: usize, c: i64) -> TowerElt {
        let c = c.rem_euclid(self.p);
        if level == 0 {
            TowerElt::Base(c)
        } else {
            let d = self.moduli[level - 1].len() - 1;
            let mut v = vec![self.e_zero(level - 1); d];
            v[0] = self.e_constant(level - 1, c);
            TowerElt::Ext(v)
        }
    }

    /// The class of the generator `y_{level-1}` in level `level`
    /// (`level >= 1`). If the modulus is linear this is a constant (the
    /// root), computed as `-psi[0]`.
    pub fn e_gen(&self, level: usize) -> TowerElt {
        let psi = &self.moduli[level - 1];
        if psi.len() == 2 {
            // linear: y = -psi_0
            self.e_neg(level - 1, &psi[0]).promoted_one(self, level)
        } else {
            let d = psi.len() - 1;
            let mut v = vec![self.e_zero(level - 1); d];
            v[1] = self.e_one(level - 1);
            TowerElt::Ext(v)
        }
    }

    /// Promote an element of level `from` to the top level by nesting.
    pub fn promote(&self, e: &TowerElt, from: usize) -> TowerElt {
        let mut cur = e.clone();
        for l in (from + 1)..=self.levels() {
            cur = cur.promoted_one(self, l);
        }
        cur
    }

    /// Is this the zero element?
    pub fn e_is_zero(&self, e: &TowerElt) -> bool {
        match e {
            TowerElt::Base(v) => *v == 0,
            TowerElt::Ext(v) => v.iter().all(|c| self.e_is_zero(c)),
        }
    }

    /// `a + b` at level `level`.
    pub fn e_add(&self, level: usize, a: &TowerElt, b: &TowerElt) -> TowerElt {
        match (a, b) {
            (TowerElt::Base(x), TowerElt::Base(y)) => TowerElt::Base((x + y).rem_euclid(self.p)),
            (TowerElt::Ext(x), TowerElt::Ext(y)) => {
                debug_assert_eq!(x.len(), y.len());
                TowerElt::Ext(
                    x.iter()
                        .zip(y.iter())
                        .map(|(u, v)| self.e_add(level - 1, u, v))
                        .collect(),
                )
            }
            _ => panic!("ResidueTower: level mismatch in e_add"),
        }
    }

    /// `-a` at level `level`.
    pub fn e_neg(&self, level: usize, a: &TowerElt) -> TowerElt {
        match a {
            TowerElt::Base(x) => TowerElt::Base((-x).rem_euclid(self.p)),
            TowerElt::Ext(v) => {
                TowerElt::Ext(v.iter().map(|c| self.e_neg(level - 1, c)).collect())
            }
        }
    }

    /// `a - b` at level `level`.
    pub fn e_sub(&self, level: usize, a: &TowerElt, b: &TowerElt) -> TowerElt {
        self.e_add(level, a, &self.e_neg(level, b))
    }

    /// `a * b` at level `level` (schoolbook, reduced mod the level modulus).
    pub fn e_mul(&self, level: usize, a: &TowerElt, b: &TowerElt) -> TowerElt {
        match (a, b) {
            (TowerElt::Base(x), TowerElt::Base(y)) => {
                TowerElt::Base(((*x as i128 * *y as i128) % self.p as i128) as i64)
            }
            (TowerElt::Ext(x), TowerElt::Ext(y)) => {
                let sub = level - 1;
                let prod = self.raw_mul(sub, x, y);
                let rem = self.raw_rem(sub, prod, &self.moduli[level - 1]);
                TowerElt::Ext(self.fixed_len(sub, rem, self.moduli[level - 1].len() - 1))
            }
            _ => panic!("ResidueTower: level mismatch in e_mul"),
        }
    }

    /// `a^exp` at level `level` (`exp >= 0`).
    pub fn e_pow(&self, level: usize, a: &TowerElt, exp: &Integer) -> TowerElt {
        let zero = Integer::from(0i64);
        let two = Integer::from(2i64);
        let mut result = self.e_one(level);
        let mut b = a.clone();
        let mut e = exp.clone();
        while e > zero {
            if (&e % &two).is_odd() {
                result = self.e_mul(level, &result, &b);
            }
            e = &e / &two;
            if e > zero {
                b = self.e_mul(level, &b, &b);
            }
        }
        result
    }

    /// `a^{-1}` at level `level`; `Err` if `a = 0`. Since the field has
    /// `q = p^F` elements, `a^{-1} = a^{q-2}`; the result is certified by
    /// re-multiplication.
    pub fn e_inv(&self, level: usize, a: &TowerElt) -> Result<TowerElt> {
        if self.e_is_zero(a) {
            return Err(MathError::DivisionByZero);
        }
        let mut q = Integer::from(1i64);
        let p = Integer::from(self.p);
        // order of the level: p^(prod of moduli degrees up to `level`)
        let mut f = 1u64;
        for m in self.moduli.iter().take(level) {
            f *= (m.len() - 1) as u64;
        }
        for _ in 0..f {
            q = &q * &p;
        }
        let inv = self.e_pow(level, a, &(&q - &Integer::from(2i64)));
        if !self.e_is_one(level, &self.e_mul(level, a, &inv)) {
            return Err(MathError::NumericalError(
                "ResidueTower::e_inv: certification failed (a * a^{q-2} != 1)".to_string(),
            ));
        }
        Ok(inv)
    }

    fn e_is_one(&self, level: usize, e: &TowerElt) -> bool {
        *e == self.e_one(level)
    }

    /// Assemble a level-`level` element (`level >= 1`) from at most
    /// `deg psi_level` coefficients of level `level - 1` (zero-padded).
    pub fn make_ext(&self, level: usize, coeffs: Vec<TowerElt>) -> Result<TowerElt> {
        let d = self.moduli[level - 1].len() - 1;
        if coeffs.len() > d {
            return Err(MathError::InvalidArgument(
                "ResidueTower::make_ext: too many coefficients".to_string(),
            ));
        }
        Ok(TowerElt::Ext(self.fixed_len(level - 1, coeffs, d)))
    }

    /// Flatten a top-level element to its `F` coordinates over `GF(p)`
    /// (concatenation of the recursive coefficient vectors).
    pub fn flatten(&self, e: &TowerElt) -> Vec<i64> {
        let mut out = Vec::new();
        fn rec(e: &TowerElt, out: &mut Vec<i64>) {
            match e {
                TowerElt::Base(v) => out.push(*v),
                TowerElt::Ext(v) => {
                    for c in v {
                        rec(c, out);
                    }
                }
            }
        }
        rec(e, &mut out);
        out
    }

    /// Inverse of [`Self::flatten`] at the top level.
    pub fn unflatten(&self, coords: &[i64]) -> TowerElt {
        fn rec(t: &ResidueTower, level: usize, coords: &[i64]) -> TowerElt {
            if level == 0 {
                debug_assert_eq!(coords.len(), 1);
                return TowerElt::Base(coords[0].rem_euclid(t.p));
            }
            let d = t.moduli[level - 1].len() - 1;
            let step = coords.len() / d;
            TowerElt::Ext(
                (0..d)
                    .map(|i| rec(t, level - 1, &coords[i * step..(i + 1) * step]))
                    .collect(),
            )
        }
        debug_assert_eq!(coords.len() as u64, self.degree());
        rec(self, self.levels(), coords)
    }

    // ------------------------------------------------------------------
    // Raw polynomial helpers over an arbitrary level (little-endian vecs)
    // ------------------------------------------------------------------

    fn raw_mul(&self, level: usize, a: &[TowerElt], b: &[TowerElt]) -> Vec<TowerElt> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![self.e_zero(level); a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            if self.e_is_zero(x) {
                continue;
            }
            for (j, y) in b.iter().enumerate() {
                let t = self.e_mul(level, x, y);
                out[i + j] = self.e_add(level, &out[i + j], &t);
            }
        }
        out
    }

    /// Remainder of `a` by the MONIC `m`, over level `level`.
    fn raw_rem(&self, level: usize, mut a: Vec<TowerElt>, m: &[TowerElt]) -> Vec<TowerElt> {
        let dm = m.len() - 1;
        debug_assert!(self.e_is_one(level, &m[dm]));
        while a.len() > dm {
            let lead = a.pop().expect("nonempty");
            if self.e_is_zero(&lead) {
                continue;
            }
            let k = a.len() - dm; // a.len() is old len - 1 = degree of popped term
            for (i, mi) in m.iter().take(dm).enumerate() {
                let t = self.e_mul(level, &lead, mi);
                a[k + i] = self.e_sub(level, &a[k + i], &t);
            }
        }
        a
    }

    fn fixed_len(&self, level: usize, mut v: Vec<TowerElt>, d: usize) -> Vec<TowerElt> {
        while v.len() < d {
            v.push(self.e_zero(level));
        }
        debug_assert!(v.len() == d);
        v
    }

    fn ptrim(&self, level: usize, mut v: Vec<TowerElt>) -> Vec<TowerElt> {
        let _ = level;
        while let Some(last) = v.last() {
            if self.e_is_zero(last) {
                v.pop();
            } else {
                break;
            }
        }
        v
    }

    // ------------------------------------------------------------------
    // Polynomials over the TOP level (public layer)
    // ------------------------------------------------------------------

    /// Trim trailing zeros (top-level polynomials).
    pub fn poly_trim(&self, v: Vec<TowerElt>) -> Vec<TowerElt> {
        self.ptrim(self.levels(), v)
    }

    /// Degree (`-1` for the zero polynomial).
    pub fn poly_degree(&self, v: &[TowerElt]) -> i64 {
        let mut d = v.len() as i64 - 1;
        while d >= 0 && self.e_is_zero(&v[d as usize]) {
            d -= 1;
        }
        d
    }

    /// `a + b` over the top level.
    pub fn poly_add(&self, a: &[TowerElt], b: &[TowerElt]) -> Vec<TowerElt> {
        let l = self.levels();
        let n = a.len().max(b.len());
        let z = self.e_zero(l);
        let out = (0..n)
            .map(|i| self.e_add(l, a.get(i).unwrap_or(&z), b.get(i).unwrap_or(&z)))
            .collect();
        self.poly_trim(out)
    }

    /// `a - b` over the top level.
    pub fn poly_sub(&self, a: &[TowerElt], b: &[TowerElt]) -> Vec<TowerElt> {
        let neg: Vec<TowerElt> = b.iter().map(|c| self.e_neg(self.levels(), c)).collect();
        self.poly_add(a, &neg)
    }

    /// `a * b` over the top level.
    pub fn poly_mul(&self, a: &[TowerElt], b: &[TowerElt]) -> Vec<TowerElt> {
        let out = self.raw_mul(self.levels(), a, b);
        self.poly_trim(out)
    }

    /// `(quotient, remainder)` of `a / b` over the top level; `Err` on `b = 0`.
    pub fn poly_divmod(&self, a: &[TowerElt], b: &[TowerElt]) -> Result<(Vec<TowerElt>, Vec<TowerElt>)> {
        let l = self.levels();
        let db = self.poly_degree(b);
        if db < 0 {
            return Err(MathError::DivisionByZero);
        }
        let db = db as usize;
        let lead_inv = self.e_inv(l, &b[db])?;
        let mut rem: Vec<TowerElt> = self.poly_trim(a.to_vec());
        let mut quo = vec![self.e_zero(l); rem.len().saturating_sub(db)];
        while self.poly_degree(&rem) >= db as i64 {
            let dr = self.poly_degree(&rem) as usize;
            let c = self.e_mul(l, &rem[dr], &lead_inv);
            quo[dr - db] = c.clone();
            for i in 0..=db {
                let t = self.e_mul(l, &c, &b[i]);
                rem[dr - db + i] = self.e_sub(l, &rem[dr - db + i], &t);
            }
            rem = self.poly_trim(rem);
        }
        Ok((self.poly_trim(quo), rem))
    }

    /// Make monic (nonzero input).
    pub fn poly_monic(&self, a: &[TowerElt]) -> Result<Vec<TowerElt>> {
        let l = self.levels();
        let d = self.poly_degree(a);
        if d < 0 {
            return Err(MathError::InvalidArgument(
                "poly_monic: zero polynomial".to_string(),
            ));
        }
        let inv = self.e_inv(l, &a[d as usize])?;
        Ok(self.poly_trim(
            a.iter().map(|c| self.e_mul(l, c, &inv)).collect(),
        ))
    }

    /// Monic gcd over the top level.
    pub fn poly_gcd(&self, a: &[TowerElt], b: &[TowerElt]) -> Result<Vec<TowerElt>> {
        let mut x = self.poly_trim(a.to_vec());
        let mut y = self.poly_trim(b.to_vec());
        while self.poly_degree(&y) >= 0 {
            let (_, r) = self.poly_divmod(&x, &y)?;
            x = y;
            y = r;
        }
        if self.poly_degree(&x) < 0 {
            Ok(x)
        } else {
            self.poly_monic(&x)
        }
    }

    /// `base^exp mod modulus` over the top level.
    pub fn poly_pow_mod(
        &self,
        base: &[TowerElt],
        exp: &Integer,
        modulus: &[TowerElt],
    ) -> Result<Vec<TowerElt>> {
        let zero = Integer::from(0i64);
        let two = Integer::from(2i64);
        let mut result = vec![self.e_one(self.levels())];
        let mut b = self.poly_divmod(base, modulus)?.1;
        let mut e = exp.clone();
        while e > zero {
            if (&e % &two).is_odd() {
                result = self.poly_divmod(&self.poly_mul(&result, &b), modulus)?.1;
            }
            e = &e / &two;
            if e > zero {
                b = self.poly_divmod(&self.poly_mul(&b, &b), modulus)?.1;
            }
        }
        Ok(result)
    }

    /// Formal derivative over the top level.
    pub fn poly_derivative(&self, a: &[TowerElt]) -> Vec<TowerElt> {
        let l = self.levels();
        let out: Vec<TowerElt> = a
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| {
                let k = self.e_constant(l, (i as i64) % self.p);
                self.e_mul(l, &k, c)
            })
            .collect();
        self.poly_trim(out)
    }

    /// The order `q = p^F` of the top field.
    pub fn field_order(&self) -> Integer {
        let mut q = Integer::from(1i64);
        let p = Integer::from(self.p);
        for _ in 0..self.degree() {
            q = &q * &p;
        }
        q
    }

    /// The `p`-th root of an inseparable polynomial `f` (`f' = 0`, so
    /// `f(y) = g(y^p)` with `g_i = f_{pi}^{q/p}`); `Err` if `f` is not of
    /// that shape.
    pub fn poly_pth_root(&self, f: &[TowerElt]) -> Result<Vec<TowerElt>> {
        let l = self.levels();
        let d = self.poly_degree(f);
        let p = self.p as usize;
        if d <= 0 || (d as usize) % p != 0 {
            return Err(MathError::NumericalError(
                "poly_pth_root: degree not a positive multiple of p".to_string(),
            ));
        }
        // inverse Frobenius exponent p^(F-1)
        let mut e = Integer::from(1i64);
        let pi = Integer::from(self.p);
        for _ in 0..(self.degree() - 1) {
            e = &e * &pi;
        }
        let mut g = vec![self.e_zero(l); d as usize / p + 1];
        for (i, c) in f.iter().enumerate() {
            if self.e_is_zero(c) {
                continue;
            }
            if i % p != 0 {
                return Err(MathError::NumericalError(
                    "poly_pth_root: polynomial is not in y^p".to_string(),
                ));
            }
            g[i / p] = self.e_pow(l, c, &e);
        }
        Ok(self.poly_trim(g))
    }

    /// Rabin irreducibility over the top field: monic `f` of degree `d >= 1`
    /// is irreducible iff `y^{q^d} = y (mod f)` and
    /// `gcd(y^{q^{d/l}} - y, f) = 1` for every prime `l | d`.
    pub fn is_irreducible(&self, f: &[TowerElt]) -> Result<bool> {
        self.is_irreducible_at(self.levels(), f)
    }

    fn is_irreducible_at(&self, level: usize, f: &[TowerElt]) -> Result<bool> {
        // work in the truncated tower whose TOP is `level`
        let t = self.truncate(level);
        let d = t.poly_degree(f);
        if d < 1 {
            return Ok(false);
        }
        if d == 1 {
            return Ok(true);
        }
        let d = d as usize;
        let q = t.field_order();
        let y = vec![t.e_zero(level), t.e_one(level)];
        let mut qd = Integer::from(1i64);
        for _ in 0..d {
            qd = &qd * &q;
        }
        let yqd = t.poly_pow_mod(&y, &qd, f)?;
        if t.poly_degree(&t.poly_sub(&yqd, &y)) >= 0 {
            return Ok(false);
        }
        let mut m = d;
        let mut ell = 2usize;
        let mut prime_divs = Vec::new();
        while ell * ell <= m {
            if m % ell == 0 {
                prime_divs.push(ell);
                while m % ell == 0 {
                    m /= ell;
                }
            }
            ell += 1;
        }
        if m > 1 {
            prime_divs.push(m);
        }
        for l in prime_divs {
            let mut qe = Integer::from(1i64);
            for _ in 0..(d / l) {
                qe = &qe * &q;
            }
            let ye = t.poly_pow_mod(&y, &qe, f)?;
            let g = t.poly_gcd(&t.poly_sub(&ye, &y), f)?;
            if t.poly_degree(&g) != 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // ------------------------------------------------------------------
    // Certified factorization over the top field
    // ------------------------------------------------------------------

    /// Distinct monic irreducible factors of `f` with multiplicities,
    /// certified: each factor Rabin-irreducible, multiplicities by trial
    /// division, and `lc(f) * prod q_i^{m_i} == f` re-verified. Randomized
    /// splitting uses a deterministic LCG; failure to split within the guard
    /// is an honest `Err`.
    pub fn factor_certified(&self, f: &[TowerElt]) -> Result<Vec<(Vec<TowerElt>, usize)>> {
        let f = self.poly_trim(f.to_vec());
        if self.poly_degree(&f) < 1 {
            return Err(MathError::InvalidArgument(
                "factor_certified: constant polynomial".to_string(),
            ));
        }
        let fm = self.poly_monic(&f)?;
        let mut rng = Lcg(0x0F0F_5EED_2026_0710);
        let mut result: Vec<(Vec<TowerElt>, usize)> = Vec::new();
        let mut pending: Vec<(Vec<TowerElt>, usize)> = vec![(fm, 1)];
        let mut guard = 0;
        while let Some((work, mult0)) = pending.pop() {
            guard += 1;
            if guard > 300 {
                return Err(MathError::NumericalError(
                    "factor_certified: internal error: no progress".to_string(),
                ));
            }
            if self.poly_degree(&work) < 1 {
                continue;
            }
            let deriv = self.poly_derivative(&work);
            if self.poly_degree(&deriv) < 0 {
                // work = g(y^p)
                let g = self.poly_pth_root(&work)?;
                pending.push((g, mult0 * self.p as usize));
                continue;
            }
            let gcd = self.poly_gcd(&work, &deriv)?;
            let (sf, rem) = self.poly_divmod(&work, &gcd)?;
            if self.poly_degree(&rem) >= 0 {
                return Err(MathError::NumericalError(
                    "factor_certified: internal error: gcd does not divide".to_string(),
                ));
            }
            // sf is squarefree; DDF + EDF, certify each piece.
            let mut cofactor = work.clone();
            for (d, gd) in self.distinct_degree_split(&sf)? {
                for piece in self.equal_degree_split(&gd, d, &mut rng)? {
                    let q = self.poly_monic(&piece)?;
                    if self.poly_degree(&q) != d as i64 || !self.is_irreducible(&q)? {
                        return Err(MathError::NumericalError(
                            "factor_certified: internal error: EDF piece not irreducible"
                                .to_string(),
                        ));
                    }
                    let mut mult = 0usize;
                    loop {
                        let (quo, r) = self.poly_divmod(&cofactor, &q)?;
                        if self.poly_degree(&r) < 0 {
                            cofactor = quo;
                            mult += 1;
                        } else {
                            break;
                        }
                    }
                    if mult == 0 {
                        return Err(MathError::NumericalError(
                            "factor_certified: internal error: factor does not divide".to_string(),
                        ));
                    }
                    result.push((q, mult * mult0));
                }
            }
            if self.poly_degree(&cofactor) >= 1 {
                pending.push((cofactor, mult0));
            }
        }
        // merge duplicates
        let mut merged: Vec<(Vec<TowerElt>, usize)> = Vec::new();
        for (q, m) in result {
            if let Some(entry) = merged.iter_mut().find(|(q2, _)| *q2 == q) {
                entry.1 += m;
            } else {
                merged.push((q, m));
            }
        }
        // certify by reconstruction
        let lc = f[self.poly_degree(&f) as usize].clone();
        let mut acc = vec![lc];
        for (q, m) in &merged {
            for _ in 0..*m {
                acc = self.poly_mul(&acc, q);
            }
        }
        if self.poly_trim(acc) != f {
            return Err(MathError::NumericalError(
                "factor_certified: reconstruction check failed".to_string(),
            ));
        }
        Ok(merged)
    }

    /// DDF: pairs `(d, product of the irreducible factors of degree d)` for
    /// squarefree monic input.
    fn distinct_degree_split(&self, f: &[TowerElt]) -> Result<Vec<(usize, Vec<TowerElt>)>> {
        let l = self.levels();
        let q = self.field_order();
        let mut out = Vec::new();
        let mut rest = self.poly_trim(f.to_vec());
        let y = vec![self.e_zero(l), self.e_one(l)];
        let mut h = self.poly_divmod(&y, &rest)?.1; // y^{q^0}
        let mut d = 0usize;
        while self.poly_degree(&rest) >= 1 {
            d += 1;
            if 2 * d as i64 > self.poly_degree(&rest) {
                // remainder is irreducible
                out.push((self.poly_degree(&rest) as usize, rest.clone()));
                break;
            }
            h = self.poly_pow_mod(&h, &q, &rest)?;
            let g = self.poly_gcd(&self.poly_sub(&h, &y), &rest)?;
            if self.poly_degree(&g) >= 1 {
                out.push((d, g.clone()));
                let (quo, r) = self.poly_divmod(&rest, &g)?;
                if self.poly_degree(&r) >= 0 {
                    return Err(MathError::NumericalError(
                        "distinct_degree_split: internal error: gcd does not divide".to_string(),
                    ));
                }
                rest = quo;
                h = self.poly_divmod(&h, &rest)?.1;
            }
        }
        Ok(out)
    }

    /// EDF: split a monic product of degree-`d` irreducibles into its factors.
    fn equal_degree_split(
        &self,
        g: &[TowerElt],
        d: usize,
        rng: &mut Lcg,
    ) -> Result<Vec<Vec<TowerElt>>> {
        let l = self.levels();
        let deg = self.poly_degree(g);
        if deg == d as i64 {
            return Ok(vec![g.to_vec()]);
        }
        let mut attempts = 0;
        loop {
            attempts += 1;
            if attempts > 200 {
                return Err(MathError::NumericalError(
                    "equal_degree_split: splitting did not converge (honest failure, not a wrong answer)"
                        .to_string(),
                ));
            }
            // random polynomial of degree < deg
            let r: Vec<TowerElt> = (0..deg as usize)
                .map(|_| self.random_elt(rng))
                .collect();
            let r = self.poly_trim(r);
            if self.poly_degree(&r) < 1 {
                continue;
            }
            let h = if self.p == 2 {
                // GF(2)-trace map: r + r^2 + r^4 + ... + r^{2^{F d - 1}} mod g
                let iters = self.degree() as usize * d;
                let mut acc = self.poly_divmod(&r, g)?.1;
                let mut cur = acc.clone();
                for _ in 1..iters {
                    cur = self.poly_divmod(&self.poly_mul(&cur, &cur), g)?.1;
                    acc = self.poly_add(&acc, &cur);
                }
                acc
            } else {
                // Cantor-Zassenhaus: r^{(q^d - 1)/2} - 1 mod g
                let q = self.field_order();
                let mut qd = Integer::from(1i64);
                for _ in 0..d {
                    qd = &qd * &q;
                }
                let e = &(&qd - &Integer::from(1i64)) / &Integer::from(2i64);
                let s = self.poly_pow_mod(&r, &e, g)?;
                self.poly_sub(&s, &[self.e_one(l)])
            };
            let w = self.poly_gcd(&h, g)?;
            let dw = self.poly_degree(&w);
            if dw >= 1 && dw < self.poly_degree(g) {
                let (quo, rr) = self.poly_divmod(g, &w)?;
                if self.poly_degree(&rr) >= 0 {
                    return Err(MathError::NumericalError(
                        "equal_degree_split: internal error: gcd does not divide".to_string(),
                    ));
                }
                let mut out = self.equal_degree_split(&w, d, rng)?;
                out.extend(self.equal_degree_split(&quo, d, rng)?);
                return Ok(out);
            }
        }
    }

    fn random_elt(&self, rng: &mut Lcg) -> TowerElt {
        let coords: Vec<i64> = (0..self.degree())
            .map(|_| (rng.next() % self.p as u64) as i64)
            .collect();
        self.unflatten(&coords)
    }

    /// Human-readable form of a top-level element (nested coefficient lists).
    pub fn describe_elt(&self, e: &TowerElt) -> String {
        format!("{:?}", self.flatten(e))
    }
}

impl TowerElt {
    /// Wrap into one more level (as the constant coefficient).
    fn promoted_one(self, t: &ResidueTower, level: usize) -> TowerElt {
        let d = t.moduli[level - 1].len() - 1;
        let mut v = vec![t.e_zero(level - 1); d];
        v[0] = self;
        TowerElt::Ext(v)
    }
}

impl fmt::Display for TowerElt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TowerElt::Base(v) => write!(f, "{}", v),
            TowerElt::Ext(v) => {
                write!(f, "[")?;
                for (i, c) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", c)?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Deterministic LCG (same generator as the maclane test batteries).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

// ---------------------------------------------------------------------------
// Tests. Expected values verified by hand / against classical finite-field
// facts (GF(4) = GF(2)[w]/(w^2+w+1), Artin-Schreier irreducibility
// Tr(c) = 1, x^q - x = product of all monic irreducibles of degree | d).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// GF(4) as GF(2)[y]/(y^2+y+1).
    fn gf4() -> ResidueTower {
        let mut t = ResidueTower::new(2).unwrap();
        t.push_level(vec![
            TowerElt::Base(1),
            TowerElt::Base(1),
            TowerElt::Base(1),
        ])
        .unwrap();
        t
    }

    /// GF(9) as GF(3)[y]/(y^2+1).
    fn gf9() -> ResidueTower {
        let mut t = ResidueTower::new(3).unwrap();
        t.push_level(vec![
            TowerElt::Base(1),
            TowerElt::Base(0),
            TowerElt::Base(1),
        ])
        .unwrap();
        t
    }

    /// GF(16) as a TOWER: GF(4)[z]/(z^2 + z + w) with w the GF(4) generator
    /// (irreducible since Tr_{GF4/GF2}(w) = w + w^2 = 1).
    fn gf16_tower() -> ResidueTower {
        let mut t = gf4();
        let w = t.e_gen(1);
        let one = t.e_one(1);
        t.push_level(vec![w, one.clone(), one]).unwrap();
        t
    }

    #[test]
    fn test_gf4_arithmetic() {
        let t = gf4();
        assert_eq!(t.degree(), 2);
        let w = t.e_gen(1);
        // w^2 = w + 1
        let w2 = t.e_mul(1, &w, &w);
        assert_eq!(w2, t.e_add(1, &w, &t.e_one(1)));
        // w^3 = 1
        let w3 = t.e_mul(1, &w2, &w);
        assert_eq!(w3, t.e_one(1));
        // inverse: w * w^2 = 1
        assert_eq!(t.e_inv(1, &w).unwrap(), w2);
        assert!(t.e_inv(1, &t.e_zero(1)).is_err());
    }

    #[test]
    fn test_gf9_arithmetic() {
        let t = gf9();
        let i = t.e_gen(1); // i^2 = -1
        let i2 = t.e_mul(1, &i, &i);
        assert_eq!(i2, t.e_constant(1, -1));
        // multiplicative order of i is 4
        assert_eq!(t.e_pow(1, &i, &Integer::from(4i64)), t.e_one(1));
        assert!(t.e_pow(1, &i, &Integer::from(2i64)) != t.e_one(1));
        // every nonzero element x: x^8 = 1
        for a in 0..3i64 {
            for b in 0..3i64 {
                if a == 0 && b == 0 {
                    continue;
                }
                let e = t.unflatten(&[a, b]);
                assert_eq!(t.e_pow(1, &e, &Integer::from(8i64)), t.e_one(1));
            }
        }
    }

    #[test]
    fn test_push_level_rejects_reducible() {
        let mut t = ResidueTower::new(2).unwrap();
        // y^2 + 1 = (y+1)^2 mod 2: not irreducible
        assert!(t
            .push_level(vec![TowerElt::Base(1), TowerElt::Base(0), TowerElt::Base(1)])
            .is_err());
        // non-monic
        let mut t3 = ResidueTower::new(3).unwrap();
        assert!(t3
            .push_level(vec![TowerElt::Base(1), TowerElt::Base(0), TowerElt::Base(2)])
            .is_err());
    }

    #[test]
    fn test_flatten_roundtrip() {
        let t = gf16_tower();
        assert_eq!(t.degree(), 4);
        for coords in [[1i64, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]] {
            let e = t.unflatten(&coords);
            assert_eq!(t.flatten(&e), coords.to_vec());
        }
    }

    #[test]
    fn test_field_law_battery_gf16_tower() {
        // associativity/distributivity/inverse on deterministic elements of
        // the DEPTH-2 tower GF(16) = GF(4)[z]/(z^2+z+w)
        let t = gf16_tower();
        let l = t.levels();
        let mut rng = Lcg(0xABCD_EF01);
        let mut elts = Vec::new();
        for _ in 0..8 {
            let coords: Vec<i64> = (0..4).map(|_| (rng.next() % 2) as i64).collect();
            elts.push(t.unflatten(&coords));
        }
        for a in &elts {
            for b in &elts {
                for c in &elts {
                    let ab_c = t.e_mul(l, &t.e_mul(l, a, b), c);
                    let a_bc = t.e_mul(l, a, &t.e_mul(l, b, c));
                    assert_eq!(ab_c, a_bc, "associativity");
                    let d1 = t.e_mul(l, a, &t.e_add(l, b, c));
                    let d2 = t.e_add(l, &t.e_mul(l, a, b), &t.e_mul(l, a, c));
                    assert_eq!(d1, d2, "distributivity");
                }
            }
            if !t.e_is_zero(a) {
                let inv = t.e_inv(l, a).unwrap();
                assert_eq!(t.e_mul(l, a, &inv), t.e_one(l), "inverse");
                // x^15 = 1 in GF(16)
                assert_eq!(t.e_pow(l, a, &Integer::from(15i64)), t.e_one(l));
            }
        }
    }

    #[test]
    fn test_irreducibility_gf4() {
        let t = gf4();
        let w = t.e_gen(1);
        let one = t.e_one(1);
        // y^2 + y + w irreducible over GF(4) (Artin-Schreier: Tr(w) = 1)
        assert!(t
            .is_irreducible(&[w.clone(), one.clone(), one.clone()])
            .unwrap());
        // y^2 + y + 1 = (y + w)(y + w^2): reducible over GF(4)
        assert!(!t
            .is_irreducible(&[one.clone(), one.clone(), one.clone()])
            .unwrap());
        // y^2 + w y + 1: no roots in GF(4) (0 -> 1, 1 -> w, w -> 1,
        // w^2 -> w), equivalently y = w z gives z^2 + z + w with Tr(w) = 1:
        // irreducible
        assert!(t
            .is_irreducible(&[one.clone(), w.clone(), one.clone()])
            .unwrap());
    }

    #[test]
    fn test_factor_gf4_split_and_irreducible() {
        // (y + w)(y + w^2) = y^2 + y + 1 over GF(4)
        let t = gf4();
        let w = t.e_gen(1);
        let one = t.e_one(1);
        let f = vec![one.clone(), one.clone(), one.clone()];
        let factors = t.factor_certified(&f).unwrap();
        assert_eq!(factors.len(), 2);
        let w2 = t.e_mul(1, &w, &w);
        let mut roots: Vec<TowerElt> = factors
            .iter()
            .map(|(q, m)| {
                assert_eq!(*m, 1);
                assert_eq!(t.poly_degree(q), 1);
                q[0].clone()
            })
            .collect();
        roots.sort_by_key(|r| t.flatten(r));
        let mut expect = vec![w.clone(), w2.clone()];
        expect.sort_by_key(|r| t.flatten(r));
        assert_eq!(roots, expect);
        // irreducible input comes back whole
        let g = vec![w.clone(), one.clone(), one.clone()];
        let factors = t.factor_certified(&g).unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
        assert_eq!(t.poly_degree(&factors[0].0), 2);
    }

    #[test]
    fn test_factor_gf4_inseparable() {
        // (y + w)^2 = y^2 + w^2: derivative 0 (char 2) -- the KNOWN-HAZARD
        // shape. Must factor as (y + w) with multiplicity 2.
        let t = gf4();
        let w = t.e_gen(1);
        let w2 = t.e_mul(1, &w, &w);
        let f = vec![w2.clone(), t.e_zero(1), t.e_one(1)];
        let factors = t.factor_certified(&f).unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 2);
        assert_eq!(t.poly_degree(&factors[0].0), 1);
        assert_eq!(factors[0].0[0], w);
    }

    #[test]
    fn test_factor_gf9_mixed() {
        // (y - i)(y + i)(y^2 + y + 2) over GF(9), i^2 = -1.
        // y^2+1 = (y-i)(y+i) over GF(9); y^2+y+2 irreducible over GF(3)
        // (disc = 1-8 = -7 = 2 mod 3, non-square mod 3), and stays
        // irreducible over GF(9)? NO: any quadratic over GF(3) splits over
        // GF(9). So over GF(9) it has two roots: 4 linear factors total.
        let t = gf9();
        let one = t.e_one(1);
        let zero = t.e_zero(1);
        let two = t.e_constant(1, 2);
        // f = (y^2 + 1)(y^2 + y + 2)
        let a = vec![one.clone(), zero.clone(), one.clone()];
        let b = vec![two.clone(), one.clone(), one.clone()];
        let f = t.poly_mul(&a, &b);
        let factors = t.factor_certified(&f).unwrap();
        assert_eq!(factors.len(), 4);
        for (q, m) in &factors {
            assert_eq!(*m, 1);
            assert_eq!(t.poly_degree(q), 1);
        }
    }

    #[test]
    fn test_factor_multiplicities_gf2() {
        // Base-level tower (no extension): (y+1)^2 y^3 (y^2+y+1) over GF(2)
        let t = ResidueTower::new(2).unwrap();
        let one = TowerElt::Base(1);
        let zero = TowerElt::Base(0);
        let y1 = vec![one.clone(), one.clone()];
        let y = vec![zero.clone(), one.clone()];
        let q = vec![one.clone(), one.clone(), one.clone()];
        let mut f = t.poly_mul(&y1, &y1);
        for _ in 0..3 {
            f = t.poly_mul(&f, &y);
        }
        f = t.poly_mul(&f, &q);
        let mut factors = t.factor_certified(&f).unwrap();
        factors.sort_by_key(|(q, _)| (t.poly_degree(q), t.flatten(&q[0])));
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0], (y, 3));
        assert_eq!(factors[1], (y1, 2));
        assert_eq!(factors[2], (q, 1));
    }

    #[test]
    fn test_factor_gf16_tower_frobenius_orbit() {
        // In the depth-2 tower GF(16): y^4 + y + 1 (a GF(2)-irreducible
        // quartic) splits into 4 linear factors over GF(16) (its roots
        // generate GF(16)). Verified: it is one of the three irreducible
        // quartics over GF(2) with splitting field GF(16).
        let t = gf16_tower();
        let one = t.e_constant(2, 1);
        let zero = t.e_constant(2, 0);
        let f = vec![one.clone(), one.clone(), zero.clone(), zero.clone(), one.clone()];
        let factors = t.factor_certified(&f).unwrap();
        assert_eq!(factors.len(), 4);
        for (q, m) in &factors {
            assert_eq!(*m, 1);
            assert_eq!(t.poly_degree(q), 1);
        }
        // and the roots are closed under the Frobenius x -> x^2
        let roots: Vec<TowerElt> = factors
            .iter()
            .map(|(q, _)| t.e_neg(2, &q[0]))
            .collect();
        for r in &roots {
            let r2 = t.e_mul(2, r, r);
            assert!(roots.contains(&r2), "roots not Frobenius-closed");
        }
    }

    #[test]
    fn test_factor_odd_char_edf_gf23() {
        // GF(23) base level: x^2 - 1 = (x-1)(x+1); x^2 - 5: 5 QR mod 23?
        // squares mod 23: 1,2,3,4,6,8,9,12,13,16,18: 5 is NOT a square,
        // so x^2 - 5 is irreducible over GF(23).
        let t = ResidueTower::new(23).unwrap();
        let f = vec![TowerElt::Base(22), TowerElt::Base(0), TowerElt::Base(1)];
        let factors = t.factor_certified(&f).unwrap();
        assert_eq!(factors.len(), 2);
        let g = vec![TowerElt::Base(18), TowerElt::Base(0), TowerElt::Base(1)]; // x^2 - 5
        let factors = t.factor_certified(&g).unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(t.poly_degree(&factors[0].0), 2);
    }

    #[test]
    fn test_poly_divmod_and_gcd() {
        let t = gf4();
        let w = t.e_gen(1);
        let one = t.e_one(1);
        // f = (y + w)(y^2 + y + w): divmod and gcd sanity
        let a = vec![w.clone(), one.clone()];
        let b = vec![w.clone(), one.clone(), one.clone()];
        let f = t.poly_mul(&a, &b);
        let (q, r) = t.poly_divmod(&f, &a).unwrap();
        assert_eq!(t.poly_degree(&r), -1);
        assert_eq!(t.poly_trim(q), b);
        let g = t.poly_gcd(&f, &a).unwrap();
        assert_eq!(g, a);
        // gcd with a coprime polynomial is 1
        let c = vec![t.e_add(1, &w, &one), one.clone()]; // y + (w+1), w+1 != w and not a root of b
        let g2 = t.poly_gcd(&a, &c).unwrap();
        assert_eq!(t.poly_degree(&g2), 0);
    }
}
