//! Conway polynomials for finite field construction
//!
//! Conway polynomials are a standard choice of irreducible polynomials
//! for constructing finite field extensions GF(p^n). They ensure
//! compatibility between different extensions of the same prime field:
//! if `x_n` denotes the class of `x` in `GF(p^n) = F_p[x]/(C_{p,n})`, then
//! `x_n^{(p^n-1)/(p^m-1)}` is a root of `C_{p,m}` whenever `m | n`, so the
//! norm-compatible embeddings `GF(p^m) -> GF(p^n)` of
//! [`crate::embedding::FieldEmbedding`] commute.
//!
//! # Coverage
//!
//! The table below covers:
//! * all primes `p < 20` (2, 3, 5, 7, 11, 13, 17, 19) with `1 <= n <= 6`,
//! * additionally `(2, 7)` and `(2, 8)`,
//! * `p` in {23, 29, 31} with `n <= 2`.
//!
//! Values follow Frank Lübeck's standard Conway polynomial table. Each entry
//! was recomputed from the definition (lexicographically first — in Lübeck's
//! word ordering — monic *primitive* polynomial of degree `n` compatible with
//! `C_{p,m}` for every proper divisor `m | n`) and cross-verified with an
//! independent implementation (sympy `galoistools`): irreducibility,
//! primitivity of `x`, and the norm-compatibility condition all hold for
//! every entry. The `conway_table_is_conway` test re-checks all of this in
//! pure Rust.
//!
//! Outside this table, [`crate::FiniteField::new`] falls back to an arbitrary
//! monic irreducible ([`crate::poly_factor::find_irreducible`]); such fields
//! report [`crate::FiniteField::is_conway`]` == false` and the compatibility
//! promise above does **not** hold for them (embeddings still exist but are
//! only canonical up to the choice of root).

use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;

/// The Conway polynomial table: `((p, n), [a_0, a_1, ..., a_n])`,
/// little-endian, coefficients in `[0, p)`.
///
/// Source: Frank Lübeck, "Conway polynomials for finite fields"
/// (<http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/>),
/// recomputed from the definition and independently verified (see module docs).
static CONWAY_TABLE: &[((u32, usize), &[u32])] = &[
    // --- p = 2 ---
    ((2, 1), &[1, 1]),                      // x + 1
    ((2, 2), &[1, 1, 1]),                   // x^2 + x + 1
    ((2, 3), &[1, 1, 0, 1]),                // x^3 + x + 1
    ((2, 4), &[1, 1, 0, 0, 1]),             // x^4 + x + 1
    ((2, 5), &[1, 0, 1, 0, 0, 1]),          // x^5 + x^2 + 1
    ((2, 6), &[1, 1, 0, 1, 1, 0, 1]),       // x^6 + x^4 + x^3 + x + 1
    ((2, 7), &[1, 1, 0, 0, 0, 0, 0, 1]),    // x^7 + x + 1
    ((2, 8), &[1, 0, 1, 1, 1, 0, 0, 0, 1]), // x^8 + x^4 + x^3 + x^2 + 1
    // --- p = 3 ---
    ((3, 1), &[1, 1]),                   // x + 1
    ((3, 2), &[2, 2, 1]),                // x^2 + 2*x + 2
    ((3, 3), &[1, 2, 0, 1]),             // x^3 + 2*x + 1
    ((3, 4), &[2, 0, 0, 2, 1]),          // x^4 + 2*x^3 + 2
    ((3, 5), &[1, 2, 0, 0, 0, 1]),       // x^5 + 2*x + 1
    ((3, 6), &[2, 2, 1, 0, 2, 0, 1]),    // x^6 + 2*x^4 + x^2 + 2*x + 2
    // --- p = 5 ---
    ((5, 1), &[3, 1]),                   // x + 3
    ((5, 2), &[2, 4, 1]),                // x^2 + 4*x + 2
    ((5, 3), &[3, 3, 0, 1]),             // x^3 + 3*x + 3
    ((5, 4), &[2, 4, 4, 0, 1]),          // x^4 + 4*x^2 + 4*x + 2
    ((5, 5), &[3, 4, 0, 0, 0, 1]),       // x^5 + 4*x + 3
    ((5, 6), &[2, 0, 1, 4, 1, 0, 1]),    // x^6 + x^4 + 4*x^3 + x^2 + 2
    // --- p = 7 ---
    ((7, 1), &[4, 1]),                   // x + 4
    ((7, 2), &[3, 6, 1]),                // x^2 + 6*x + 3
    ((7, 3), &[4, 0, 6, 1]),             // x^3 + 6*x^2 + 4
    ((7, 4), &[3, 4, 5, 0, 1]),          // x^4 + 5*x^2 + 4*x + 3
    ((7, 5), &[4, 1, 0, 0, 0, 1]),       // x^5 + x + 4
    ((7, 6), &[3, 6, 4, 5, 1, 0, 1]),    // x^6 + x^4 + 5*x^3 + 4*x^2 + 6*x + 3
    // --- p = 11 ---
    ((11, 1), &[9, 1]),                  // x + 9
    ((11, 2), &[2, 7, 1]),               // x^2 + 7*x + 2
    ((11, 3), &[9, 2, 0, 1]),            // x^3 + 2*x + 9
    ((11, 4), &[2, 10, 8, 0, 1]),        // x^4 + 8*x^2 + 10*x + 2
    ((11, 5), &[9, 0, 10, 0, 0, 1]),     // x^5 + 10*x^2 + 9
    ((11, 6), &[2, 7, 6, 4, 3, 0, 1]),   // x^6 + 3*x^4 + 4*x^3 + 6*x^2 + 7*x + 2
    // --- p = 13 ---
    ((13, 1), &[11, 1]),                 // x + 11
    ((13, 2), &[2, 12, 1]),              // x^2 + 12*x + 2
    ((13, 3), &[11, 2, 0, 1]),           // x^3 + 2*x + 11
    ((13, 4), &[2, 12, 3, 0, 1]),        // x^4 + 3*x^2 + 12*x + 2
    ((13, 5), &[11, 4, 0, 0, 0, 1]),     // x^5 + 4*x + 11
    ((13, 6), &[2, 11, 11, 10, 0, 0, 1]), // x^6 + 10*x^3 + 11*x^2 + 11*x + 2
    // --- p = 17 ---
    ((17, 1), &[14, 1]),                 // x + 14
    ((17, 2), &[3, 16, 1]),              // x^2 + 16*x + 3
    ((17, 3), &[14, 1, 0, 1]),           // x^3 + x + 14
    ((17, 4), &[3, 10, 7, 0, 1]),        // x^4 + 7*x^2 + 10*x + 3
    ((17, 5), &[14, 1, 0, 0, 0, 1]),     // x^5 + x + 14
    ((17, 6), &[3, 3, 10, 0, 2, 0, 1]),  // x^6 + 2*x^4 + 10*x^2 + 3*x + 3
    // --- p = 19 ---
    ((19, 1), &[17, 1]),                 // x + 17
    ((19, 2), &[2, 18, 1]),              // x^2 + 18*x + 2
    ((19, 3), &[17, 4, 0, 1]),           // x^3 + 4*x + 17
    ((19, 4), &[2, 11, 2, 0, 1]),        // x^4 + 2*x^2 + 11*x + 2
    ((19, 5), &[17, 5, 0, 0, 0, 1]),     // x^5 + 5*x + 17
    ((19, 6), &[2, 6, 17, 17, 0, 0, 1]), // x^6 + 17*x^3 + 17*x^2 + 6*x + 2
    // --- p = 23 ---
    ((23, 1), &[18, 1]),                 // x + 18
    ((23, 2), &[5, 21, 1]),              // x^2 + 21*x + 5
    // --- p = 29 ---
    ((29, 1), &[27, 1]),                 // x + 27
    ((29, 2), &[2, 24, 1]),              // x^2 + 24*x + 2
    // --- p = 31 ---
    ((31, 1), &[28, 1]),                 // x + 28
    ((31, 2), &[3, 29, 1]),              // x^2 + 29*x + 3
];

fn lookup(p: u32, n: usize) -> Option<&'static [u32]> {
    CONWAY_TABLE
        .iter()
        .find(|((tp, tn), _)| *tp == p && *tn == n)
        .map(|(_, coeffs)| *coeffs)
}

/// Get the Conway polynomial for GF(p^n)
///
/// Returns the standard irreducible (indeed primitive) polynomial of degree
/// `n` over GF(p), from a lookup table of pre-computed Conway polynomials for
/// small `p` and `n` (see the module docs for exact coverage).
///
/// # Arguments
///
/// * `p` - Prime characteristic
/// * `n` - Degree of the extension
///
/// # Returns
///
/// The Conway polynomial as a `UnivariatePolynomial<Integer>` with coefficients in [0, p).
/// Returns `None` if the Conway polynomial is not in the lookup table.
///
/// # Format
///
/// Polynomials are given as coefficient vectors [a_0, a_1, ..., a_n]
/// representing a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
pub fn conway_polynomial(p: u32, n: usize) -> Option<UnivariatePolynomial<Integer>> {
    lookup(p, n).map(|coeffs| {
        let int_coeffs: Vec<Integer> = coeffs.iter().map(|&c| Integer::from(c as i64)).collect();
        UnivariatePolynomial::new(int_coeffs)
    })
}

/// Check if a Conway polynomial is available for GF(p^n)
pub fn has_conway_polynomial(p: u32, n: usize) -> bool {
    lookup(p, n).is_some()
}

/// Get all available Conway polynomials
///
/// Returns a sorted vector of (p, n) pairs for which Conway polynomials are
/// available.
pub fn available_conway_polynomials() -> Vec<(u32, usize)> {
    let mut keys: Vec<(u32, usize)> = CONWAY_TABLE.iter().map(|(k, _)| *k).collect();
    keys.sort();
    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::prime::factor;

    #[test]
    fn test_conway_gf2() {
        // GF(2^2): x^2 + x + 1
        let poly = conway_polynomial(2, 2).unwrap();
        assert_eq!(poly.degree(), Some(2));
        assert_eq!(*poly.coeff(0), Integer::from(1));
        assert_eq!(*poly.coeff(1), Integer::from(1));
        assert_eq!(*poly.coeff(2), Integer::from(1));
    }

    #[test]
    fn test_conway_gf3() {
        // GF(3^2): x^2 + 2*x + 2 (Lübeck's table). The former entry
        // x^2 + x + 2 is primitive too, but comes strictly later in Lübeck's
        // word ordering, so it is not the Conway polynomial.
        let poly = conway_polynomial(3, 2).unwrap();
        assert_eq!(poly.degree(), Some(2));
        assert_eq!(*poly.coeff(0), Integer::from(2));
        assert_eq!(*poly.coeff(1), Integer::from(2));
        assert_eq!(*poly.coeff(2), Integer::from(1));
    }

    #[test]
    fn test_known_luebeck_values() {
        // Spot checks against well-known published values.
        // C_{2,6} = x^6 + x^4 + x^3 + x + 1 (NOT x^6 + x + 1, which is
        // primitive but violates norm compatibility with GF(4) and GF(8)).
        let c26 = conway_polynomial(2, 6).unwrap();
        let expect = [1i64, 1, 0, 1, 1, 0, 1];
        for (i, e) in expect.iter().enumerate() {
            assert_eq!(*c26.coeff(i), Integer::from(*e));
        }
        // C_{5,4} = x^4 + 4x^2 + 4x + 2.
        let c54 = conway_polynomial(5, 4).unwrap();
        let expect = [2i64, 4, 4, 0, 1];
        for (i, e) in expect.iter().enumerate() {
            assert_eq!(*c54.coeff(i), Integer::from(*e));
        }
        // C_{7,2} = x^2 + 6x + 3.
        let c72 = conway_polynomial(7, 2).unwrap();
        let expect = [3i64, 6, 1];
        for (i, e) in expect.iter().enumerate() {
            assert_eq!(*c72.coeff(i), Integer::from(*e));
        }
        // C_{3,1} = x + 1 (root 2, the smallest primitive root mod 3).
        let c31 = conway_polynomial(3, 1).unwrap();
        assert_eq!(*c31.coeff(0), Integer::from(1));
        assert_eq!(*c31.coeff(1), Integer::from(1));
    }

    #[test]
    fn test_has_conway() {
        assert!(has_conway_polynomial(2, 4));
        assert!(has_conway_polynomial(3, 3));
        assert!(has_conway_polynomial(19, 6));
        assert!(!has_conway_polynomial(2, 100)); // Not in table
        assert!(!has_conway_polynomial(1000, 1)); // Prime not in table
    }

    #[test]
    fn test_available_polynomials() {
        let available = available_conway_polynomials();

        // Full coverage p < 20, n <= 6.
        for p in [2u32, 3, 5, 7, 11, 13, 17, 19] {
            for n in 1..=6usize {
                assert!(available.contains(&(p, n)), "missing ({p},{n})");
            }
        }
        assert!(available.contains(&(2, 8)));
        assert!(available.contains(&(31, 2)));

        // Should be sorted.
        for i in 1..available.len() {
            let (p1, n1) = available[i - 1];
            let (p2, n2) = available[i];
            assert!(p1 < p2 || (p1 == p2 && n1 < n2));
        }
    }

    #[test]
    fn test_conway_polynomial_format() {
        // Verify that all polynomials are monic with coefficients in [0, p).
        for ((p, n), coeffs) in CONWAY_TABLE.iter() {
            assert_eq!(
                coeffs.len(),
                n + 1,
                "Polynomial for GF({}^{}) has wrong length",
                p,
                n
            );
            assert_eq!(coeffs[*n], 1, "Polynomial for GF({}^{}) is not monic", p, n);

            for &coeff in coeffs.iter() {
                assert!(
                    coeff < *p,
                    "Coefficient {} >= {} in polynomial for GF({}^{})",
                    coeff,
                    p,
                    p,
                    n
                );
            }
        }
    }

    // --- full mathematical self-check of the table -------------------------

    fn to_int_vec(coeffs: &[u32]) -> Vec<Integer> {
        coeffs.iter().map(|&c| Integer::from(c as i64)).collect()
    }

    fn trim(mut v: Vec<Integer>) -> Vec<Integer> {
        while v.last().map(|c| c.is_zero()).unwrap_or(false) {
            v.pop();
        }
        v
    }

    /// `a * b mod (f, p)` for little-endian coefficient vectors, monic `f`.
    /// Small helper independent of `poly_factor` internals.
    fn mul_mod(a: &[Integer], b: &[Integer], f: &[Integer], p: &Integer) -> Vec<Integer> {
        let n = f.len() - 1;
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![Integer::zero(); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                out[i + j] = (out[i + j].clone() + ai.clone() * bj.clone()) % p.clone();
            }
        }
        // reduce mod monic f
        for k in (n..out.len()).rev() {
            let lead = out[k].clone();
            if lead.is_zero() {
                continue;
            }
            for i in 0..n {
                let v = (out[k - n + i].clone() - lead.clone() * f[i].clone()) % p.clone();
                out[k - n + i] = if v.signum() < 0 { v + p.clone() } else { v };
            }
            out[k] = Integer::zero();
        }
        out.truncate(n);
        trim(out)
    }

    /// x^e mod f over F_p (little-endian).
    fn x_pow_mod(e: &Integer, f: &[Integer], p: &Integer) -> Vec<Integer> {
        let mut result = vec![Integer::one()];
        let mut base = vec![Integer::zero(), Integer::one()];
        let mut e = e.clone();
        let two = Integer::from(2);
        while e > Integer::zero() {
            if (e.clone() % two.clone()).is_one() {
                result = mul_mod(&result, &base, f, p);
            }
            e = e / two.clone();
            if e > Integer::zero() {
                base = mul_mod(&base, &base, f, p);
            }
        }
        result
    }

    /// Evaluate `g(y) mod (f, p)` by Horner, `g` with F_p coefficients.
    fn eval_mod(g: &[Integer], y: &[Integer], f: &[Integer], p: &Integer) -> Vec<Integer> {
        let mut acc: Vec<Integer> = Vec::new();
        for c in g.iter().rev() {
            acc = mul_mod(&acc, y, f, p);
            if !c.is_zero() {
                if acc.is_empty() {
                    acc.push(c.clone());
                } else {
                    acc[0] = (acc[0].clone() + c.clone()) % p.clone();
                    acc = trim(acc);
                }
            }
        }
        acc
    }

    /// Every table entry must be irreducible, primitive (x generates
    /// GF(p^n)^*), and norm-compatible with the entries for all proper
    /// divisors of n. This is the defining property of Conway polynomials
    /// (short of lexicographic minimality, which was verified offline).
    #[test]
    fn conway_table_is_conway() {
        for ((p, n), coeffs) in CONWAY_TABLE.iter() {
            let pi = Integer::from(*p as i64);
            let f = to_int_vec(coeffs);
            // irreducible over F_p
            assert!(
                crate::poly_factor::is_irreducible_fp(&f, &pi),
                "C_{{{p},{n}}} not irreducible"
            );
            // primitive: ord(x) = p^n - 1
            let qm1 = pi.pow(*n as u32) - Integer::one();
            assert_eq!(
                x_pow_mod(&qm1, &f, &pi),
                vec![Integer::one()],
                "x^(q-1) != 1 for C_{{{p},{n}}}"
            );
            for (r, _) in factor(&qm1) {
                let e = qm1.clone() / r.clone();
                assert_ne!(
                    x_pow_mod(&e, &f, &pi),
                    vec![Integer::one()],
                    "x has order dividing (q-1)/{r} in GF({p}^{n}) — not primitive"
                );
            }
            // norm compatibility with proper-divisor entries in the table
            for m in 1..*n {
                if n % m != 0 {
                    continue;
                }
                let sub = lookup(*p, m)
                    .unwrap_or_else(|| panic!("subfield entry ({p},{m}) missing from table"));
                let sub = to_int_vec(sub);
                let e = qm1.clone() / (pi.pow(m as u32) - Integer::one());
                let y = x_pow_mod(&e, &f, &pi);
                let val = eval_mod(&sub, &y, &f, &pi);
                assert!(
                    val.is_empty(),
                    "C_{{{p},{m}}}(x^{{(q-1)/(p^{m}-1)}}) != 0 mod C_{{{p},{n}}}: norm compatibility broken"
                );
            }
        }
    }
}
