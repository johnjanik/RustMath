//! Linear codes over an arbitrary field `F: Field`
//!
//! [`GenericLinearCode`] is the field-generic counterpart of the `u64`-based
//! [`crate::LinearCode`]: generator/parity-check construction by row reduction
//! (RREF with pivot tracking, the same construction as the proven `u64` code
//! path, written over any [`Field`]), encoding, syndromes, dual codes, and
//! exhaustive weight enumeration under an explicit, honest size budget.
//!
//! The module also provides [`macwilliams_transform`], the exact-integer
//! MacWilliams transform `W_{C^perp}(x, y) = |C|^{-1} W_C(x + (q-1)y, x - y)`,
//! so a code's dual weight enumerator can be computed two independent ways
//! (direct enumeration of the dual vs. transform of the primal) and compared.
//!
//! # Enumeration is only as honest as the element list
//!
//! Rust cannot enumerate the elements of an abstract `F: Field`, so
//! [`GenericLinearCode::weight_enumerator`] takes the full element list from
//! the caller. The list is validated for distinctness and closure under `+`
//! and `*` (so it is at least a subfield/subring containing every entry it
//! generates), but completeness relative to `F` cannot be checked
//! generically: pass all `q` elements. Helpers [`prime_field_elements`] and
//! [`finite_field_elements`] build correct lists for `GF(p)` and `GF(p^n)`.

use rustmath_core::{Field, NumericConversion};
use rustmath_finitefields::{FiniteField, FiniteFieldElement, PrimeField};
use rustmath_integers::Integer;
use std::collections::HashSet;

use crate::LinearCode;

/// A linear `[n, k]` code over an arbitrary field `F`.
#[derive(Clone, Debug)]
pub struct GenericLinearCode<F: Field> {
    /// Generator matrix, `k` rows of length `n`.
    generator: Vec<Vec<F>>,
    /// Parity check matrix, `n - k` rows of length `n`, with `G * H^T = 0`.
    parity_check: Vec<Vec<F>>,
    /// Code length `n`.
    length: usize,
    /// Code dimension `k`.
    dimension: usize,
}

/// Row-reduce `mat` to RREF over `F`, returning the reduced matrix and the
/// pivot columns (one per row of the row space, in order).
fn rref<F: Field>(mat: &[Vec<F>]) -> Result<(Vec<Vec<F>>, Vec<usize>), String> {
    let rows = mat.len();
    if rows == 0 {
        return Ok((vec![], vec![]));
    }
    let cols = mat[0].len();
    if mat.iter().any(|r| r.len() != cols) {
        return Err("matrix rows have unequal lengths".to_string());
    }

    let mut m: Vec<Vec<F>> = mat.to_vec();
    let mut pivot_cols: Vec<usize> = Vec::with_capacity(rows);
    let mut current = 0usize;

    for col in 0..cols {
        if current >= rows {
            break;
        }
        let sel = match (current..rows).find(|&r| !m[r][col].is_zero()) {
            Some(r) => r,
            None => continue,
        };
        m.swap(current, sel);

        let inv = m[current][col]
            .inverse()
            .map_err(|e| format!("pivot entry has no inverse: {e:?}"))?;
        for c in 0..cols {
            m[current][c] = m[current][c].clone() * inv.clone();
        }

        for r in 0..rows {
            if r == current || m[r][col].is_zero() {
                continue;
            }
            let factor = m[r][col].clone();
            for c in 0..cols {
                m[r][c] = m[r][c].clone() - factor.clone() * m[current][c].clone();
            }
        }

        pivot_cols.push(col);
        current += 1;
    }

    Ok((m, pivot_cols))
}

/// The `[I|P] -> [-P^T|I]` complement construction generalized to permuted
/// pivot columns (mirror of the `u64` `LinearCode` construction): given the
/// RREF of a full-row-rank matrix and its pivot columns, build a basis of the
/// orthogonal complement, one row per free column.
fn orthogonal_complement_rows<F: Field>(
    rref_mat: &[Vec<F>],
    pivot_cols: &[usize],
    cols: usize,
) -> Vec<Vec<F>> {
    let pivot_set: HashSet<usize> = pivot_cols.iter().copied().collect();
    let free_cols: Vec<usize> = (0..cols).filter(|c| !pivot_set.contains(c)).collect();

    let mut out = Vec::with_capacity(free_cols.len());
    for &fc in &free_cols {
        let mut row = vec![F::zero(); cols];
        row[fc] = F::one();
        for (i, &pc) in pivot_cols.iter().enumerate() {
            row[pc] = -rref_mat[i][fc].clone();
        }
        out.push(row);
    }
    out
}

impl<F: Field> GenericLinearCode<F> {
    /// Build a code from a full-row-rank generator matrix (`k` rows of
    /// length `n`). Returns `Err` if the matrix is empty, ragged, or does not
    /// have full row rank (an honest failure, not a silent rank reduction).
    pub fn from_generator(generator: Vec<Vec<F>>) -> Result<Self, String> {
        let k = generator.len();
        if k == 0 {
            return Err("generator matrix has no rows".to_string());
        }
        let n = generator[0].len();
        if n < k {
            return Err(format!("generator matrix is {k} x {n} with k > n"));
        }

        let (m, pivots) = rref(&generator)?;
        if pivots.len() != k {
            return Err(format!(
                "generator matrix does not have full row rank: rank {} < k = {}",
                pivots.len(),
                k
            ));
        }
        let parity_check = orthogonal_complement_rows(&m, &pivots, n);

        Ok(GenericLinearCode {
            generator,
            parity_check,
            length: n,
            dimension: k,
        })
    }

    /// Build a code from a full-row-rank parity check matrix (`r` rows of
    /// length `n`); the code has dimension `k = n - r`.
    pub fn from_parity_check(parity_check: Vec<Vec<F>>) -> Result<Self, String> {
        let r = parity_check.len();
        if r == 0 {
            return Err("parity check matrix has no rows".to_string());
        }
        let n = parity_check[0].len();
        if n < r {
            return Err(format!("parity check matrix is {r} x {n} with r > n"));
        }

        let (m, pivots) = rref(&parity_check)?;
        if pivots.len() != r {
            return Err(format!(
                "parity check matrix does not have full row rank: rank {} < r = {}",
                pivots.len(),
                r
            ));
        }
        let generator = orthogonal_complement_rows(&m, &pivots, n);
        let k = n - r;

        Ok(GenericLinearCode {
            generator,
            parity_check,
            length: n,
            dimension: k,
        })
    }

    /// Code length `n`.
    pub fn length(&self) -> usize {
        self.length
    }

    /// Code dimension `k`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Redundancy `n - k`.
    pub fn redundancy(&self) -> usize {
        self.length - self.dimension
    }

    /// The generator matrix.
    pub fn generator_matrix(&self) -> &[Vec<F>] {
        &self.generator
    }

    /// The parity check matrix.
    pub fn parity_check_matrix(&self) -> &[Vec<F>] {
        &self.parity_check
    }

    /// Encode a length-`k` message as `m * G`.
    pub fn encode(&self, message: &[F]) -> Result<Vec<F>, String> {
        if message.len() != self.dimension {
            return Err(format!(
                "message length {} does not match code dimension {}",
                message.len(),
                self.dimension
            ));
        }
        let mut codeword = Vec::with_capacity(self.length);
        for j in 0..self.length {
            let mut sum = F::zero();
            for (mi, row) in message.iter().zip(self.generator.iter()) {
                sum = sum + mi.clone() * row[j].clone();
            }
            codeword.push(sum);
        }
        Ok(codeword)
    }

    /// Syndrome `H * w^T` of a length-`n` word.
    pub fn syndrome(&self, word: &[F]) -> Result<Vec<F>, String> {
        if word.len() != self.length {
            return Err(format!(
                "word length {} does not match code length {}",
                word.len(),
                self.length
            ));
        }
        let mut syn = Vec::with_capacity(self.parity_check.len());
        for row in &self.parity_check {
            let mut sum = F::zero();
            for (h, w) in row.iter().zip(word.iter()) {
                sum = sum + h.clone() * w.clone();
            }
            syn.push(sum);
        }
        Ok(syn)
    }

    /// Whether `word` lies in the code (zero syndrome). Words of the wrong
    /// length are not codewords.
    pub fn is_codeword(&self, word: &[F]) -> bool {
        match self.syndrome(word) {
            Ok(s) => s.iter().all(|x| x.is_zero()),
            Err(_) => false,
        }
    }

    /// The dual code `C^perp`: its generator is this code's parity check and
    /// vice versa (`G * H^T = 0` is symmetric in the two roles).
    pub fn dual_code(&self) -> Self {
        GenericLinearCode {
            generator: self.parity_check.clone(),
            parity_check: self.generator.clone(),
            length: self.length,
            dimension: self.length - self.dimension,
        }
    }

    /// Validate a caller-supplied field element list: pairwise distinct,
    /// contains 0 and 1, and is closed under `+` and `*`. Returns q.
    fn validate_elements(&self, elements: &[F]) -> Result<usize, String> {
        let q = elements.len();
        if q < 2 {
            return Err("field element list must contain at least 0 and 1".to_string());
        }
        for i in 0..q {
            for j in (i + 1)..q {
                if elements[i] == elements[j] {
                    return Err(format!(
                        "field element list has duplicate entries at indices {i} and {j}"
                    ));
                }
            }
        }
        if !elements.iter().any(|e| e.is_zero()) {
            return Err("field element list does not contain zero".to_string());
        }
        if !elements.iter().any(|e| e.is_one()) {
            return Err("field element list does not contain one".to_string());
        }
        for a in elements {
            for b in elements {
                let s = a.clone() + b.clone();
                let p = a.clone() * b.clone();
                if !elements.contains(&s) || !elements.contains(&p) {
                    return Err(
                        "field element list is not closed under + and * (incomplete list?)"
                            .to_string(),
                    );
                }
            }
        }
        Ok(q)
    }

    /// The weight enumerator `[A_0, A_1, ..., A_n]` (`A_i` = number of
    /// codewords of Hamming weight `i`) by exhaustive enumeration of all
    /// `q^k` codewords.
    ///
    /// `elements` must be the complete list of field elements (see the module
    /// docs for what is and is not validated). If `q^k > budget` this returns
    /// `Err` — never an estimate.
    pub fn weight_enumerator(&self, elements: &[F], budget: u64) -> Result<Vec<u64>, String> {
        let q = self.validate_elements(elements)?;

        let mut total: u128 = 1;
        for _ in 0..self.dimension {
            total = total
                .checked_mul(q as u128)
                .ok_or_else(|| "q^k overflows u128".to_string())?;
            if total > budget as u128 {
                return Err(format!(
                    "enumeration of q^k = {}^{} codewords exceeds budget {}",
                    q, self.dimension, budget
                ));
            }
        }

        let mut counts = vec![0u64; self.length + 1];
        // Odometer over message symbols.
        let mut idx = vec![0usize; self.dimension];
        loop {
            let message: Vec<F> = idx.iter().map(|&i| elements[i].clone()).collect();
            let codeword = self.encode(&message)?;
            let weight = codeword.iter().filter(|x| !x.is_zero()).count();
            counts[weight] += 1;

            // increment odometer
            let mut pos = 0;
            loop {
                if pos == self.dimension {
                    // sanity: we enumerated exactly q^k words
                    debug_assert_eq!(counts.iter().sum::<u64>() as u128, total);
                    return Ok(counts);
                }
                idx[pos] += 1;
                if idx[pos] < q {
                    break;
                }
                idx[pos] = 0;
                pos += 1;
            }
        }
    }

    /// Minimum distance by exhaustive search (smallest nonzero-weight index
    /// of the weight enumerator), under the same honest budget.
    pub fn minimum_distance(&self, elements: &[F], budget: u64) -> Result<usize, String> {
        let counts = self.weight_enumerator(elements, budget)?;
        counts
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, &c)| c > 0)
            .map(|(i, _)| i)
            .ok_or_else(|| "code has no nonzero codewords (k = 0?)".to_string())
    }
}

impl GenericLinearCode<PrimeField> {
    /// Bridge from the `u64`-based [`LinearCode`]: reinterpret its generator
    /// matrix over [`PrimeField`] with the same characteristic.
    pub fn from_u64_code(code: &LinearCode) -> Result<Self, String> {
        let p = code.field_characteristic();
        let generator = code
            .generator_matrix()
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&v| {
                        PrimeField::new(Integer::from(v), Integer::from(p))
                            .map_err(|e| format!("bad entry {v} mod {p}: {e:?}"))
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;
        Self::from_generator(generator)
    }
}

/// All `p` elements of `GF(p)` as [`PrimeField`] values, `0..p-1`.
pub fn prime_field_elements(p: u64) -> Result<Vec<PrimeField>, String> {
    if p < 2 {
        return Err(format!("GF({p}) is not a field"));
    }
    (0..p)
        .map(|v| {
            PrimeField::new(Integer::from(v), Integer::from(p))
                .map_err(|e| format!("cannot build {v} mod {p}: {e:?}"))
        })
        .collect()
}

/// All `p^n` elements of `GF(p^n)`, indexed by base-`p` digits (little-endian
/// coefficient vectors of the residue polynomial).
pub fn finite_field_elements(field: &FiniteField) -> Result<Vec<FiniteFieldElement>, String> {
    let p = field
        .characteristic()
        .to_u64()
        .ok_or_else(|| "characteristic too large to enumerate".to_string())?;
    let n = field.degree();
    let q = p
        .checked_pow(n as u32)
        .ok_or_else(|| "field too large to enumerate".to_string())?;
    let mut out = Vec::with_capacity(q as usize);
    for idx in 0..q {
        let mut digits = Vec::with_capacity(n);
        let mut t = idx;
        for _ in 0..n {
            digits.push(Integer::from(t % p));
            t /= p;
        }
        out.push(field.element(digits));
    }
    Ok(out)
}

/// The MacWilliams transform: given the weight enumerator
/// `[A_0, ..., A_n]` of a linear code `C` over `GF(q)`, compute the weight
/// enumerator of the dual code `C^perp` via
///
/// `B_j = |C|^{-1} * sum_i A_i * K_j(i)`,
///
/// where `K_j(i) = sum_s (-1)^s (q-1)^{j-s} C(i,s) C(n-i,j-s)` is the
/// Krawtchouk polynomial (the coefficient of `x^{n-j} y^j` in
/// `(x+(q-1)y)^{n-i} (x-y)^i`). All arithmetic is exact in `i128`; if any
/// `B_j` is negative or the division by `|C|` is inexact, the input was not
/// the weight enumerator of a linear code and this returns `Err`.
pub fn macwilliams_transform(weight_enum: &[u64], q: u64) -> Result<Vec<u64>, String> {
    if weight_enum.is_empty() {
        return Err("empty weight enumerator".to_string());
    }
    if q < 2 {
        return Err(format!("q = {q} is not a prime power >= 2"));
    }
    let n = weight_enum.len() - 1;
    let total: i128 = weight_enum.iter().map(|&a| a as i128).sum();
    if total == 0 {
        return Err("weight enumerator counts no codewords".to_string());
    }
    // A linear code over GF(q) has exactly q^k codewords. Requiring |C| to be
    // a power of q catches the wrong-q footgun (a mismatched q otherwise
    // yields a wrong-but-plausible transform whenever the divisions happen to
    // land on integers).
    {
        let mut m = total;
        while m % (q as i128) == 0 {
            m /= q as i128;
        }
        if m != 1 {
            return Err(format!(
                "sum of the weight enumerator is {total}, which is not a power of \
                 q = {q} — not the enumerator of a linear code over GF({q}) \
                 (wrong q, or a non-linear count)"
            ));
        }
    }

    // Pascal's triangle up to n (rows are zero-padded, so C(i-1, i) = 0).
    let mut binom = vec![vec![0i128; n + 1]; n + 1];
    for i in 0..=n {
        binom[i][0] = 1;
        for j in 1..=i {
            binom[i][j] = binom[i - 1][j - 1] + binom[i - 1][j];
        }
    }

    let qm1 = (q - 1) as i128;
    let mut out = Vec::with_capacity(n + 1);
    for j in 0..=n {
        let mut sum: i128 = 0;
        for (i, &ai) in weight_enum.iter().enumerate() {
            if ai == 0 {
                continue;
            }
            let mut kraw: i128 = 0;
            for s in 0..=j.min(i) {
                if j - s > n - i {
                    continue;
                }
                let sign = if s % 2 == 0 { 1i128 } else { -1i128 };
                let term = sign
                    .checked_mul(
                        qm1.checked_pow((j - s) as u32)
                            .ok_or_else(|| "overflow in (q-1)^j".to_string())?,
                    )
                    .and_then(|t| t.checked_mul(binom[i][s]))
                    .and_then(|t| t.checked_mul(binom[n - i][j - s]))
                    .ok_or_else(|| "overflow in Krawtchouk term".to_string())?;
                kraw += term;
            }
            sum = sum
                .checked_add(
                    (ai as i128)
                        .checked_mul(kraw)
                        .ok_or_else(|| "overflow in A_i * K_j(i)".to_string())?,
                )
                .ok_or_else(|| "overflow in MacWilliams sum".to_string())?;
        }
        if sum % total != 0 || sum < 0 {
            return Err(format!(
                "MacWilliams transform is not a nonnegative integer at j = {j} \
                 (sum = {sum}, |C| = {total}): input is not a linear code's weight enumerator"
            ));
        }
        out.push((sum / total) as u64);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golay::TernaryGolayCode;
    use rustmath_core::Ring;

    fn gf(v: i64, p: u64) -> PrimeField {
        PrimeField::new(Integer::from(v), Integer::from(p)).unwrap()
    }

    fn rows(m: &[&[i64]], p: u64) -> Vec<Vec<PrimeField>> {
        m.iter()
            .map(|r| r.iter().map(|&v| gf(v, p)).collect())
            .collect()
    }

    /// G * H^T = 0 for every generic construction, over GF(7).
    #[test]
    fn test_parity_check_orthogonality_gf7() {
        let g = rows(&[&[1, 0, 3, 5], &[0, 1, 2, 6]], 7);
        let code = GenericLinearCode::from_generator(g).unwrap();
        assert_eq!(code.length(), 4);
        assert_eq!(code.dimension(), 2);
        assert_eq!(code.redundancy(), 2);
        for grow in code.generator_matrix() {
            for hrow in code.parity_check_matrix() {
                let mut sum = PrimeField::zero();
                for (a, b) in grow.iter().zip(hrow.iter()) {
                    sum = sum + a.clone() * b.clone();
                }
                assert!(sum.is_zero(), "G * H^T != 0");
            }
        }
        // encode/syndrome round trip: codewords have zero syndrome
        let msg = vec![gf(2, 7), gf(5, 7)];
        let cw = code.encode(&msg).unwrap();
        assert!(code.is_codeword(&cw));
        // a non-codeword: perturb one coordinate
        let mut bad = cw.clone();
        bad[3] = bad[3].clone() + gf(1, 7);
        assert!(!code.is_codeword(&bad));
    }

    /// from_parity_check builds the dual construction; dual_code() swaps roles.
    #[test]
    fn test_from_parity_check_and_dual_gf3() {
        let h = rows(&[&[1, 0, 1, 2], &[0, 1, 1, 1]], 3);
        let code = GenericLinearCode::from_parity_check(h).unwrap();
        assert_eq!(code.dimension(), 2);
        let dual = code.dual_code();
        assert_eq!(dual.dimension(), 2);
        // every dual codeword is orthogonal to every codeword
        let elements = prime_field_elements(3).unwrap();
        let a = code.weight_enumerator(&elements, 1000).unwrap();
        let b_direct = dual.weight_enumerator(&elements, 1000).unwrap();
        let b_transform = macwilliams_transform(&a, 3).unwrap();
        assert_eq!(b_direct, b_transform, "MacWilliams identity failed");
    }

    /// Rank-deficient generator is an honest error, not a silent fix.
    #[test]
    fn test_rank_deficient_generator_rejected() {
        let g = rows(&[&[1, 2, 3], &[2, 4, 6]], 7); // row2 = 2 * row1
        assert!(GenericLinearCode::from_generator(g).is_err());
    }

    /// Budget exhaustion is an honest error, never an estimate.
    #[test]
    fn test_budget_honesty() {
        let golay = TernaryGolayCode::new();
        let u64_code = LinearCode::from_generator_matrix(golay.generator_matrix().clone(), 3);
        let code = GenericLinearCode::from_u64_code(&u64_code).unwrap();
        let elements = prime_field_elements(3).unwrap();
        let err = code.weight_enumerator(&elements, 100).unwrap_err();
        assert!(err.contains("budget"), "unexpected error: {err}");
    }

    /// Element-list validation: duplicates and incomplete lists are rejected.
    #[test]
    fn test_element_list_validation() {
        let g = rows(&[&[1, 0], &[0, 1]], 5);
        let code = GenericLinearCode::from_generator(g).unwrap();
        let mut dup = prime_field_elements(5).unwrap();
        dup[3] = dup[2].clone();
        assert!(code.weight_enumerator(&dup, 1000).is_err());
        // {0, 1, 2} is not closed under + in GF(5)
        let partial = vec![gf(0, 5), gf(1, 5), gf(2, 5)];
        assert!(code.weight_enumerator(&partial, 1000).is_err());
    }

    // ---- MacWilliams gates -------------------------------------------------
    //
    // Every expected enumerator below was derived independently in Python
    // (GF(p) arithmetic by hand; GF(4)/GF(9) via explicit polynomial
    // arithmetic with the same Conway moduli the finitefields crate uses),
    // and each gate ALSO checks the identity internally: dual enumerator by
    // direct enumeration == MacWilliams transform of the primal enumerator.

    fn macwilliams_gate<F: Field>(
        code: &GenericLinearCode<F>,
        elements: &[F],
        q: u64,
        expect_a: &[u64],
        expect_b: &[u64],
    ) {
        let a = code.weight_enumerator(elements, 1_000_000).unwrap();
        assert_eq!(a, expect_a, "primal weight enumerator mismatch");
        let b_direct = code
            .dual_code()
            .weight_enumerator(elements, 1_000_000)
            .unwrap();
        assert_eq!(b_direct, expect_b, "dual weight enumerator mismatch");
        let b_transform = macwilliams_transform(&a, q).unwrap();
        assert_eq!(
            b_direct, b_transform,
            "MacWilliams identity failed: direct dual enumeration disagrees with transform"
        );
    }

    /// GF(2), Hamming(7,4) — via the u64 bridge, cross-checked both paths.
    #[test]
    fn test_macwilliams_hamming_7_4_gf2() {
        let g = vec![
            vec![1, 0, 0, 0, 1, 1, 0],
            vec![0, 1, 0, 0, 1, 0, 1],
            vec![0, 0, 1, 0, 0, 1, 1],
            vec![0, 0, 0, 1, 1, 1, 1],
        ];
        let mut u64_code = LinearCode::from_generator_matrix(g, 2);
        let code = GenericLinearCode::from_u64_code(&u64_code).unwrap();
        let elements = prime_field_elements(2).unwrap();
        // Python-derived: A = 1 + 7 z^3 + 7 z^4 + z^7, dual (simplex) = 1 + 7 z^4.
        macwilliams_gate(
            &code,
            &elements,
            2,
            &[1, 0, 0, 7, 7, 0, 0, 1],
            &[1, 0, 0, 0, 7, 0, 0, 0],
        );
        // The u64 path and the generic path agree on the minimum distance.
        assert_eq!(u64_code.minimum_distance(), 3);
        assert_eq!(code.minimum_distance(&elements, 1_000_000).unwrap(), 3);
    }

    /// GF(2), the [5,3] toy code from the u64 tests.
    #[test]
    fn test_macwilliams_toy_5_3_gf2() {
        let g = rows(&[&[1, 0, 0, 1, 1], &[0, 1, 0, 1, 0], &[0, 0, 1, 0, 1]], 2);
        let code = GenericLinearCode::from_generator(g).unwrap();
        let elements = prime_field_elements(2).unwrap();
        // Python-derived.
        macwilliams_gate(&code, &elements, 2, &[1, 0, 2, 4, 1, 0], &[1, 0, 0, 2, 1, 0]);
    }

    /// GF(3), the ternary Golay [11,6,5] — via the u64 bridge from the
    /// crate's own Golay construction; enumerators pinned to the classical
    /// values (independently rederived in Python from the cyclic generator
    /// g(x) = 2 + x^2 + 2x^3 + x^4 + x^5, a factor of x^11 - 1 over GF(3)).
    #[test]
    fn test_macwilliams_ternary_golay_gf3() {
        let golay = TernaryGolayCode::new();
        let u64_code = LinearCode::from_generator_matrix(golay.generator_matrix().clone(), 3);
        let code = GenericLinearCode::from_u64_code(&u64_code).unwrap();
        let elements = prime_field_elements(3).unwrap();
        macwilliams_gate(
            &code,
            &elements,
            3,
            &[1, 0, 0, 0, 0, 132, 132, 0, 330, 110, 0, 24],
            &[1, 0, 0, 0, 0, 0, 132, 0, 0, 110, 0, 0],
        );
        assert_eq!(code.minimum_distance(&elements, 1_000_000).unwrap(), 5);
        assert_eq!(golay.minimum_distance(), 5);
    }

    /// GF(7), a [4,2] code.
    #[test]
    fn test_macwilliams_gf7() {
        let g = rows(&[&[1, 0, 3, 5], &[0, 1, 2, 6]], 7);
        let code = GenericLinearCode::from_generator(g).unwrap();
        let elements = prime_field_elements(7).unwrap();
        // Python-derived: self-dual-enumerator pair.
        macwilliams_gate(&code, &elements, 7, &[1, 0, 0, 24, 24], &[1, 0, 0, 24, 24]);
    }

    /// GF(4) = GF(2)[w]/(w^2+w+1) — the point of the exercise: the identity
    /// connects two independent enumerations over an extension field,
    /// exercising the whole generic stack on ExtensionField arithmetic.
    #[test]
    fn test_macwilliams_gf4_extension_field() {
        let f = FiniteField::new(Integer::from(2), 2).unwrap();
        let one = f.one();
        let zero = f.zero();
        let w = f.generator();
        let w1 = w.clone() + one.clone();
        let g = vec![
            vec![one.clone(), zero.clone(), one.clone(), w.clone()],
            vec![zero.clone(), one.clone(), w.clone(), w1.clone()],
        ];
        let code = GenericLinearCode::from_generator(g).unwrap();
        let elements = finite_field_elements(&f).unwrap();
        assert_eq!(elements.len(), 4);
        // Python-derived over the same modulus w^2 + w + 1.
        macwilliams_gate(&code, &elements, 4, &[1, 0, 3, 6, 6], &[1, 0, 3, 6, 6]);
    }

    /// GF(9) = GF(3)[w]/(w^2+2w+2) — extension field of odd characteristic.
    #[test]
    fn test_macwilliams_gf9_extension_field() {
        let f = FiniteField::new(Integer::from(3), 2).unwrap();
        let one = f.one();
        let zero = f.zero();
        let two = f.from_int(Integer::from(2));
        let w = f.generator();
        let w1 = w.clone() + one.clone();
        let g = vec![
            vec![one.clone(), zero.clone(), w.clone(), two.clone()],
            vec![zero.clone(), one.clone(), w1.clone(), w.clone()],
        ];
        let code = GenericLinearCode::from_generator(g).unwrap();
        let elements = finite_field_elements(&f).unwrap();
        assert_eq!(elements.len(), 9);
        // Python-derived over the same modulus w^2 + 2w + 2.
        macwilliams_gate(&code, &elements, 9, &[1, 0, 0, 32, 48], &[1, 0, 0, 32, 48]);
    }
}
