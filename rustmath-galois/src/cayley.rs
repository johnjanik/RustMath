//! The Cayley–Dummit sextic resolvent: the exact solvability test for
//! quintics, i.e. the decision `Gal(f) ⊆ F20` (up to conjugacy) that closes
//! the one degree-5 gap of the cycle-type sieve (F20 vs S5).
//!
//! # Theory
//!
//! `F20 = ⟨(0 1 2 3 4), (1 2 4 3)⟩` (the normalizer of a Sylow 5-subgroup,
//! order 20) has index 6 in `S5`; the transitive subgroups of `S5` split into
//! the **solvable** chain `C5 ⊂ D5 ⊂ F20` — exactly the groups contained in a
//! conjugate of `F20` — and the **non-solvable** `A5 ⊂ S5`. Dummit's
//! degree-4 invariant
//!
//! ```text
//! θ = x0²(x1x4 + x2x3) + x1²(x0x2 + x3x4) + x2²(x0x4 + x1x3)
//!   + x3²(x0x1 + x2x4) + x4²(x0x3 + x1x2)
//! ```
//!
//! has stabilizer exactly `F20` (verified: 20 of the 120 permutations fix θ,
//! including both generators), so its `S5`-orbit `{θ_1,…,θ_6}` is indexed by
//! the six cosets, and `R(T) = ∏ᵢ (T − θᵢ)` has coefficients symmetric in the
//! roots — integers for monic integral `f`. If `Gal(f)` lies in a conjugate of
//! `F20` it fixes a coset, hence fixes some `θᵢ`, which is therefore a
//! rational root of `R`; conversely, when `R` is **squarefree** (θ-values
//! pairwise distinct), a rational root forces `Gal(f)` to fix that coset, i.e.
//! `Gal(f) ⊆ F20^σ`. Squarefreeness is checked exactly; on failure a
//! Tschirnhaus transformation `β = α² + kα` (same splitting field — see
//! [`tschirnhaus`]) is applied and the resolvent recomputed.
//!
//! This is Theorem 1 of D. S. Dummit, *Solving solvable quintics*, Math.
//! Comp. 57 (1991) 387–401, with the resolvent re-derived from scratch for
//! this port (Dummit states it for the trinomial-free general depressed
//! quintic, as here).
//!
//! # Derivation of the baked coefficients
//!
//! For the depressed quintic `y⁵ + p y³ + q y² + r y + s` (any monic quintic
//! is depressed integrally by `y = 5x + a₄`, i.e. `g(y) = 5⁵·f((y−a₄)/5)`,
//! which rescales-and-shifts the roots — splitting field unchanged), the
//! coefficient of `T^{6−k}` in `R` is `(−1)^k e_k(θ₁,…,θ₆)`, a symmetric,
//! isobaric polynomial of weight `4k` in `p, q, r, s` (weights 2, 3, 4, 5).
//! The tables below were derived **twice, independently**, with the results
//! agreeing term-for-term:
//!
//! 1. sympy: expand `∏ᵢ (T − θᵢ)` over `x0..x4`, reduce each coefficient with
//!    `sympy.polys.polyfuncs.symmetrize` (remainder verified 0), substitute
//!    `e1 = 0, e2 = p, e3 = −q, e4 = r, e5 = −s`;
//! 2. exact linear algebra: sample integer root tuples with `Σxᵢ = 0`,
//!    evaluate `e_k(θ)` exactly, solve for the isobaric monomial coefficients
//!    over ℚ (overdetermined; verified on 30 fresh samples per coefficient).
//!
//! Both were verified against 60-digit numerical root products (mpmath) on 8
//! quintics, and the full criterion against PARI/GP `polgalois` on 36
//! quintics covering all five groups (see the battery test in `crate`).

use crate::types::Evidence;
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::{bivariate, zassenhaus, zx};
use rustmath_rationals::Rational;

// ------------------------------------------------------------------------- //
// Baked resolvent coefficients (see module docs for the double derivation).
// Each table lists (integer coefficient, [exponent of p, q, r, s]) for one
// T-coefficient of R(T) = T^6 + C5·T^5 + … + C0.
// ------------------------------------------------------------------------- //

const C5: &[(i64, [u32; 4])] = &[(8, [0, 0, 1, 0])];
const C4: &[(i64, [u32; 4])] = &[
    (40, [0, 0, 2, 0]),
    (-50, [0, 1, 0, 1]),
    (2, [1, 2, 0, 0]),
    (-6, [2, 0, 1, 0]),
];
const C3: &[(i64, [u32; 4])] = &[
    (160, [0, 0, 3, 0]),
    (-400, [0, 1, 1, 1]),
    (-2, [0, 4, 0, 0]),
    (125, [1, 0, 0, 2]),
    (21, [1, 2, 1, 0]),
    (-40, [2, 0, 2, 0]),
    (-15, [2, 1, 0, 1]),
];
const C2: &[(i64, [u32; 4])] = &[
    (400, [0, 0, 4, 0]),
    (-1400, [0, 1, 2, 1]),
    (625, [0, 2, 0, 2]),
    (-8, [0, 4, 1, 0]),
    (500, [1, 0, 1, 2]),
    (76, [1, 2, 2, 0]),
    (-50, [1, 3, 0, 1]),
    (-136, [2, 0, 3, 0]),
    (90, [2, 1, 1, 1]),
    (1, [2, 4, 0, 0]),
    (-6, [3, 2, 1, 0]),
    (9, [4, 0, 2, 0]),
];
const C1: &[(i64, [u32; 4])] = &[
    (-3125, [0, 0, 0, 4]),
    (512, [0, 0, 5, 0]),
    (-2400, [0, 1, 3, 1]),
    (2750, [0, 2, 1, 2]),
    (3, [0, 4, 2, 0]),
    (-58, [0, 5, 0, 1]),
    (-500, [1, 0, 2, 2]),
    (625, [1, 1, 0, 3]),
    (76, [1, 2, 3, 0]),
    (105, [1, 3, 1, 1]),
    (-2, [1, 6, 0, 0]),
    (-256, [2, 0, 4, 0]),
    (260, [2, 1, 2, 1]),
    (-325, [2, 2, 0, 2]),
    (19, [2, 4, 1, 0]),
    (525, [3, 0, 1, 2]),
    (-51, [3, 2, 2, 0]),
    (-31, [3, 3, 0, 1]),
    (32, [4, 0, 3, 0]),
    (117, [4, 1, 1, 1]),
    (-108, [5, 0, 0, 2]),
];
const C0: &[(i64, [u32; 4])] = &[
    (-9375, [0, 0, 1, 4]),
    (256, [0, 0, 6, 0]),
    (-1600, [0, 1, 4, 1]),
    (3250, [0, 2, 2, 2]),
    (17, [0, 4, 3, 0]),
    (-124, [0, 5, 1, 1]),
    (1, [0, 8, 0, 0]),
    (-2000, [1, 0, 3, 2]),
    (-1250, [1, 1, 1, 3]),
    (-16, [1, 2, 4, 0]),
    (590, [1, 3, 2, 1]),
    (-125, [1, 4, 0, 2]),
    (-13, [1, 6, 1, 0]),
    (3125, [2, 0, 0, 4]),
    (-192, [2, 0, 5, 0]),
    (-160, [2, 1, 3, 1]),
    (-725, [2, 2, 1, 2]),
    (65, [2, 4, 2, 0]),
    (-12, [2, 5, 0, 1]),
    (1200, [3, 0, 2, 2]),
    (-128, [3, 2, 3, 0]),
    (12, [3, 3, 1, 1]),
    (48, [4, 0, 4, 0]),
    (196, [4, 1, 2, 1]),
    (-150, [4, 2, 0, 2]),
    (-99, [5, 0, 1, 2]),
    (1, [5, 2, 2, 0]),
    (-4, [5, 3, 0, 1]),
    (-4, [6, 0, 3, 0]),
    (18, [6, 1, 1, 1]),
    (-27, [7, 0, 0, 2]),
];

fn eval_table(table: &[(i64, [u32; 4])], p: &Integer, q: &Integer, r: &Integer, s: &Integer) -> Integer {
    let mut acc = Integer::zero();
    for (c, [a, b, cc, d]) in table {
        acc = acc + Integer::from(*c) * p.pow(*a) * q.pow(*b) * r.pow(*cc) * s.pow(*d);
    }
    acc
}

/// The Cayley–Dummit sextic resolvent of the **depressed** monic quintic
/// `y⁵ + p y³ + q y² + r y + s`, little-endian, monic of degree 6.
fn resolvent_sextic(p: &Integer, q: &Integer, r: &Integer, s: &Integer) -> Vec<Integer> {
    vec![
        eval_table(C0, p, q, r, s),
        eval_table(C1, p, q, r, s),
        eval_table(C2, p, q, r, s),
        eval_table(C3, p, q, r, s),
        eval_table(C4, p, q, r, s),
        eval_table(C5, p, q, r, s),
        Integer::one(),
    ]
}

/// Integral depression of a monic quintic: `g(y) = 5⁵ · f((y − a₄)/5)` with
/// `a₄` the `x⁴`-coefficient. `g` is monic and integral with zero `y⁴`-term;
/// its roots are `5αᵢ + a₄`, so the splitting field and the Galois action are
/// unchanged.
fn depress_quintic(f: &[Integer]) -> Result<Vec<Integer>> {
    if f.len() != 6 || !f[5].is_one() {
        return Err(MathError::InvalidArgument(
            "depress_quintic needs a monic degree-5 polynomial".to_string(),
        ));
    }
    let a4 = f[4].clone();
    // g = Σ_j f_j · 5^{5−j} · (y − a4)^j
    let mut g = vec![Integer::zero(); 6];
    let five = Integer::from(5i64);
    let mut pow_y_minus = vec![Integer::one()]; // (y − a4)^0
    for j in 0..=5usize {
        if !f[j].is_zero() {
            let scale = f[j].clone() * five.pow((5 - j) as u32);
            for (k, c) in pow_y_minus.iter().enumerate() {
                g[k] = g[k].clone() + scale.clone() * c.clone();
            }
        }
        if j < 5 {
            // multiply by (y − a4)
            let mut next = vec![Integer::zero(); pow_y_minus.len() + 1];
            for (k, c) in pow_y_minus.iter().enumerate() {
                next[k + 1] = next[k + 1].clone() + c.clone();
                next[k] = next[k].clone() - a4.clone() * c.clone();
            }
            pow_y_minus = next;
        }
    }
    debug_assert!(g[5].is_one());
    if !g[4].is_zero() {
        return Err(MathError::InvalidOperation(
            "depression failed: y⁴-term nonzero (internal error)".to_string(),
        ));
    }
    Ok(g)
}

/// Tschirnhaus transformation: the monic degree-5 polynomial with roots
/// `βᵢ = αᵢ² + k·αᵢ`, computed exactly as `±Res_x(f(x), y − x² − kx)`.
///
/// Soundness when the result is squarefree: the five `βᵢ` are then distinct,
/// so no `βᵢ` is rational (the conjugates of `β₁` would all coincide), hence
/// `[ℚ(β₁):ℚ] = 5` (a divisor of the prime `[ℚ(α₁):ℚ] = 5`, not 1) and
/// `ℚ(β₁) = ℚ(α₁)`: same splitting field, and `αᵢ ↦ βᵢ` intertwines the two
/// degree-5 root actions, so the Galois group is the same subgroup of `S5` up
/// to conjugacy. A non-squarefree result is reported as `Err` and the caller
/// tries the next `k`.
fn tschirnhaus(f: &[Integer], k: i64) -> Result<Vec<Integer>> {
    // f(x), constant in y: index [x-power][y-power].
    let fbiv: Vec<Vec<Rational>> = f
        .iter()
        .map(|c| vec![Rational::from_integer(c.clone())])
        .collect();
    // h(x, y) = y − x² − kx.
    let hbiv: Vec<Vec<Rational>> = vec![
        vec![Rational::from_i64(0), Rational::from_i64(1)], // x^0: y
        vec![Rational::from_i64(-k)],                       // x^1: −k
        vec![Rational::from_i64(-1)],                       // x^2: −1
    ];
    let res = bivariate::resultant_in_t(&fbiv, &hbiv);
    // Expect ∏ᵢ (y − βᵢ) up to sign: degree 5, leading ±1.
    let mut out: Vec<Integer> = Vec::with_capacity(res.len());
    for c in &res {
        if !c.is_integer() {
            return Err(MathError::InvalidOperation(
                "Tschirnhaus resultant coefficient not integral".to_string(),
            ));
        }
        out.push(c.numerator().clone());
    }
    let out = zx::trim(&out);
    if zx::degree(&out) != 5 {
        return Err(MathError::InvalidOperation(
            "Tschirnhaus resultant does not have degree 5".to_string(),
        ));
    }
    let out = if out[5].is_one() {
        out
    } else if (-out[5].clone()).is_one() {
        zx::neg(&out)
    } else {
        return Err(MathError::InvalidOperation(
            "Tschirnhaus resultant not ±monic".to_string(),
        ));
    };
    // Squarefree ⇔ the βᵢ are distinct (required for soundness; see above).
    let gcd = zx::subresultant_gcd(&out, &zx::derivative(&out));
    if zx::degree(&gcd) != 0 {
        return Err(MathError::InvalidOperation(format!(
            "Tschirnhaus β = α² + {k}α has colliding values (result not squarefree)"
        )));
    }
    Ok(out)
}

/// Outcome of the exact solvability test.
pub(crate) struct CayleyOutcome {
    /// `true` ⟺ `Gal(f)` is contained in a conjugate of `F20` ⟺ `f` is
    /// solvable by radicals.
    pub solvable: bool,
    /// Irreducible-factor degrees of the (verified squarefree) sextic — the
    /// orbit lengths of `Gal(f)` on the six cosets of `F20`.
    pub factor_degrees: Vec<usize>,
    /// Which polynomial the certified resolvent was computed from.
    pub description: String,
}

/// Decide `Gal(f) ⊆ F20` (up to conjugacy) for a monic irreducible quintic
/// `f ∈ ℤ[x]` via the Cayley–Dummit sextic resolvent, retrying through
/// Tschirnhaus transformations until the resolvent is squarefree. Every
/// returned verdict is certified (rational root exhibited by exact
/// factorization, or its proven absence, on a squarefree resolvent); if no
/// attempt produces a squarefree resolvent an honest `Err` is returned.
pub(crate) fn f20_membership(f: &[Integer], ev: &mut Evidence) -> Result<CayleyOutcome> {
    if f.len() != 6 || !f[5].is_one() {
        return Err(MathError::InvalidArgument(
            "f20_membership needs a monic degree-5 polynomial".to_string(),
        ));
    }
    // Attempt 0: f itself; attempts k = 1..: β = α² + kα.
    for attempt in 0..=6i64 {
        let (h, desc) = if attempt == 0 {
            (f.to_vec(), "the input quintic".to_string())
        } else {
            match tschirnhaus(f, attempt) {
                Ok(h) => (h, format!("the Tschirnhaus transform β = α² + {attempt}α")),
                Err(e) => {
                    ev.notes.push(format!("Cayley resolvent attempt {attempt}: {e}"));
                    continue;
                }
            }
        };
        let g = depress_quintic(&h)?;
        let resolvent = resolvent_sextic(&g[3], &g[2], &g[1], &g[0]);
        let (_, factors) = zassenhaus::factor(&resolvent).map_err(|_| {
            MathError::NotSupported("factor recombination limit exceeded".to_string())
        })?;
        if factors.iter().any(|(_, m)| *m > 1) {
            ev.notes.push(format!(
                "Cayley sextic resolvent of {desc} is not squarefree (θ-value collision); \
                 applying a Tschirnhaus transformation"
            ));
            continue;
        }
        let mut degs: Vec<usize> = factors.iter().map(|(g, _)| g.len() - 1).collect();
        degs.sort_unstable();
        let solvable = degs.first() == Some(&1);
        return Ok(CayleyOutcome { solvable, factor_degrees: degs, description: desc });
    }
    Err(MathError::NotSupported(
        "no squarefree Cayley sextic resolvent found (input and 6 Tschirnhaus \
         transformations all degenerate)"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    /// Decimal-string → `Integer`, to pin gp-derived values exceeding `i64`.
    fn int(s: &str) -> Integer {
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        let ten = Integer::from(10);
        let mut acc = Integer::zero();
        for ch in digits.chars() {
            assert!(ch.is_ascii_digit());
            acc = acc * ten.clone() + Integer::from((ch as u8 - b'0') as i64);
        }
        if neg {
            -acc
        } else {
            acc
        }
    }

    // All expected values below were computed independently with PARI/GP
    // (g = 5^5·subst(f, x, (x−a4)/5); R from the same tables re-typed in GP;
    // factor degrees and rational roots via factor(R)) — see the module docs
    // for the derivation and its two-way + numerical verification.

    #[test]
    fn depression_pins() {
        // x^5 − x − 1 → y^5 − 625y − 3125 (a4 = 0: pure scaling by 5).
        assert_eq!(
            depress_quintic(&iz(&[-1, -1, 0, 0, 0, 1])).unwrap(),
            iz(&[-3125, -625, 0, 0, 0, 1])
        );
        // Lehmer C5 quintic x^5+x^4−4x^3−3x^2+3x+1 (a4 = 1):
        // gp: 5^5·subst(f,x,(x−1)/5) = y^5 − 110y^3 − 55y^2 + 2310y + 979.
        assert_eq!(
            depress_quintic(&iz(&[1, 3, -3, -4, 1, 1])).unwrap(),
            iz(&[979, 2310, -55, -110, 0, 1])
        );
        // Non-monic input refused.
        assert!(depress_quintic(&iz(&[1, 0, 0, 0, 0, 2])).is_err());
    }

    #[test]
    fn resolvent_pins_against_gp() {
        // x^5 − 2 (depressed: y^5 − 6250): R = T^6 − 4768371582031250000·T.
        let g = depress_quintic(&iz(&[-2, 0, 0, 0, 0, 1])).unwrap();
        let r = resolvent_sextic(&g[3], &g[2], &g[1], &g[0]);
        let mut expect = iz(&[0, 0, 0, 0, 0, 0, 1]);
        expect[1] = -Integer::from(4768371582031250000i64);
        assert_eq!(r, expect);

        // x^5 − x − 1 (S5): gp-pinned vector.
        let g = depress_quintic(&iz(&[-1, -1, 0, 0, 0, 1])).unwrap();
        let r = resolvent_sextic(&g[3], &g[2], &g[1], &g[0]);
        let expect: Vec<Integer> = [
            "574052333831787109375",
            "-346851348876953125",
            "61035156250000",
            "-39062500000",
            "15625000",
            "-5000",
            "1",
        ]
        .iter()
        .map(|s| int(s))
        .collect();
        assert_eq!(r, expect);

        // Lehmer C5 quintic: gp-pinned vector (exercises p, q, r, s ≠ 0).
        let g = depress_quintic(&iz(&[1, 3, -3, -4, 1, 1])).unwrap();
        let r = resolvent_sextic(&g[3], &g[2], &g[1], &g[0]);
        let expect: Vec<Integer> = [
            "-360260685644469671875",
            "2980357148316659375",
            "-1796651418959375",
            "-580262760000",
            "47764750",
            "18480",
            "1",
        ]
        .iter()
        .map(|s| int(s))
        .collect();
        assert_eq!(r, expect);
    }

    #[test]
    fn solvability_verdicts_match_gp_polgalois() {
        // (f, expected solvable, expected orbit lengths on the 6 cosets)
        let cases: [(&[i64], bool, &[usize]); 6] = [
            (&[-2, 0, 0, 0, 0, 1], true, &[1, 5]),   // F20 (rational root 0)
            (&[12, 15, 0, 0, 0, 1], true, &[1, 5]),  // F20 (rational root 0)
            (&[12, -5, 0, 0, 0, 1], true, &[1, 5]),  // D5
            (&[1, 3, -3, -4, 1, 1], true, &[1, 5]),  // C5 (root −9955, gp)
            (&[-1, -1, 0, 0, 0, 1], false, &[6]),    // S5
            (&[16, 20, 0, 0, 0, 1], false, &[6]),    // A5
        ];
        for (f, solvable, degs) in cases {
            let mut ev = Evidence::default();
            let out = f20_membership(&iz(f), &mut ev).unwrap();
            assert_eq!(out.solvable, solvable, "solvability of {f:?}");
            assert_eq!(out.factor_degrees, degs.to_vec(), "coset orbits of {f:?}");
            assert_eq!(out.description, "the input quintic");
        }
    }

    #[test]
    fn tschirnhaus_pin_and_soundness() {
        // β = α² + α for f = x^5 − 2. gp:
        //   polresultant(x^5-2, y-x^2-x, x) = y^5 − 10y² − 10y − 6
        // (monic; polisirreducible = 1; polgalois of it = F(5) = 5:4).
        let t = tschirnhaus(&iz(&[-2, 0, 0, 0, 0, 1]), 1).unwrap();
        assert_eq!(t, iz(&[-6, -10, -10, 0, 0, 1]));
        // The transform must preserve the Galois group: its own Cayley
        // resolvent must yield the same verdict (F20 ⇒ solvable).
        let g = depress_quintic(&t).unwrap();
        let resolvent = resolvent_sextic(&g[3], &g[2], &g[1], &g[0]);
        let (_, factors) = zassenhaus::factor(&resolvent).unwrap();
        assert!(factors.iter().all(|(_, m)| *m == 1), "resolvent squarefree");
        assert!(
            factors.iter().any(|(g, _)| g.len() == 2),
            "transformed F20 quintic still has a rational resolvent root"
        );
    }
}
